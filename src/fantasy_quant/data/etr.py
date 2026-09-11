"""Establish The Run rankings ingest.

ETR has no API and their terms prohibit automated scraping, so the only path is
the CSV download button on each rankings chart. This reads what that button
produces.

Three things about the export shape drive the design:

* **It is RANKINGS, not projections.** The exports carry rank, and sometimes ADP and
  the gap between them -- no component stats. So they cannot feed the ensemble,
  which combines at the stat level precisely so one projection set can serve
  leagues with different scoring. ETR enters instead as an independent opinion to
  arbitrage against, which is arguably where a human ranking set belongs anyway.
* **There is more than one board, and they collide.** The Draft Kit chart and Matt
  Silva's Top 150 are separate downloads that share a folder and a scoring format,
  so a board needs a second axis to be addressable at all. The rank column's own
  name is that axis and the only stable thing that carries it: `"ETR Rank"` against
  `"Silva Rank"`. It becomes `EtrRankings.kind`, and `load_all` keys on
  `(kind, scoring)` rather than scoring alone.
* **Only one of them ships the ESPN player id.** The Draft Kit export's `id` column
  joined 300/300 against our 2026 pool with no fuzzy matching. The Top 150 has no id
  column at all, so it resolves by name through `EspnIdIndex`, which refuses
  ambiguity by design -- measured 150/150 on the live file with no override table,
  and a miss is loud where a wrong player would be invisible.

Scoring is per file, not per row: the Full PPR and Half PPR charts are separate
downloads. Keep them in separate files and load the one matching the league.

**Substitution across scoring is the caller's call, not this module's.** `for_league`
still never substitutes silently. But a board's *positional* ordering barely moves with
the scoring format, because half-vs-full PPR reorders receivers against running backs
rather than against each other: measured on the two 300-row Draft Kit boards, the
within-position Spearman between them is +0.9967 (RB) to +1.0000 (QB, K, D/ST), with a
mean drift of at most 1.24 positional ranks. So a consumer that reads only
`positional()` can use the other format deliberately; `applies_to` is how it says so.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..core import DST, QB, RB, TE, WR, K

log = logging.getLogger(__name__)

DEFAULT_DIR = Path("data/manual/etr")
DEFAULT_ARCHIVE = DEFAULT_DIR / "archive"

#: Rank columns by the analyst whose board it is. Two boards can share a scoring
#: format and a folder, and this prefix is the only thing in the export that tells
#: them apart, so it is load-bearing rather than decorative.
RANK_COLUMNS: Mapping[str, str] = {"etrrank": "etr", "silvarank": "silva"}
POS_RANK_COLUMNS: Mapping[str, str] = {"etrposrank": "etr", "silvaposrank": "silva"}
COMMENT_COLUMNS: Mapping[str, str] = {"etrcommentary": "etr", "silvacommentary": "silva"}

#: Column names as ETR writes them, mapped to ours. The export has a BOM and
#: quotes every field, so read with encoding="utf8-lossy" and normalise here
#: rather than trusting the header bytes. The rank, positional-rank and commentary
#: columns are not here because their names carry the board's kind; see above.
COLUMNS: dict[str, str] = {
    "player": "player",
    "position": "position",
    "team": "team",
    "adp": "adp",
    "rankingdiff": "rank_vs_adp",
    "adpposrank": "adp_pos_rank",
    "posrankdiff": "pos_rank_vs_adp",
    "rankchange": "rank_change",
    "id": "espn_id",
}

#: `player` and a rank are the whole of what a board must carry. `espn_id` used to be
#: required and is not: the Top 150 has no id column, and refusing it over that would
#: refuse the only in-season board we get.
REQUIRED = ("player", "etr_rank")

#: Position abbreviation as the export writes it -> ESPN `defaultPositionId`.
POSITION_IDS: Mapping[str, int] = {
    "QB": QB, "RB": RB, "WR": WR, "TE": TE, "K": K, "PK": K,
    "DST": DST, "DEF": DST, "D/ST": DST,
}

#: Below this share of rows resolved to an ESPN id, a name-keyed board is refused.
#: The live Top 150 resolves 150/150; anything near this floor means the crosswalk has
#: drifted or the file is not what it claims, and a half-matched board silently
#: re-ranks only the half it matched.
MIN_NAME_MATCH = 0.90

_POS_RANK = re.compile(r"^([A-Za-z/]+)\s*(\d+)$")


class EtrError(RuntimeError):
    """The export could not be read as an ETR rankings file."""


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True, slots=True)
class EtrRankings:
    """One ETR rankings chart, joined to ESPN ids.

    `kind` names the board ("etr", "silva"); `scoring` names the format it was
    downloaded under. Together they address a board uniquely, which scoring alone
    does not.
    """

    scoring: str
    kind: str
    path: Path
    frame: pl.DataFrame
    #: (resolved, total) when the board had no id column and was matched by name;
    #: None when it shipped ESPN ids and no matching was needed.
    name_match: tuple[int, int] | None = None

    @property
    def n(self) -> int:
        return self.frame.height

    def applies_to(self, scoring_variant: str) -> bool:
        """Whether this board was downloaded under a league's scoring format.

        False is not "unusable" -- see the module docstring on the measured
        within-position drift -- it is "say so out loud before you use it".
        """
        return self.scoring == scoring_variant

    def rank_of(self, espn_id: int) -> int | None:
        hit = self.frame.filter(pl.col("espn_id") == espn_id)
        return int(hit["etr_rank"][0]) if hit.height else None

    def as_map(self) -> dict[int, int]:
        return dict(zip(self.frame["espn_id"], self.frame["etr_rank"], strict=True))

    def positional(self) -> dict[int, tuple[int, int]]:
        """`{espn_id: (position_id, rank within that position)}`.

        The accessor every consumer should reach for, and the overall rank is not.
        An overall 1..150 rank conflates the analyst's player evaluation -- which is
        what we are buying -- with his cross-positional value model, which this repo
        already solves per league at the flex fixed point in `decide/valuation.py`.
        Adopting the overall rank would overwrite a solved, league-specific model
        with a generic one, silently.
        """
        out: dict[int, tuple[int, int]] = {}
        for row in self.frame.iter_rows(named=True):
            pid, pos, rank = row["espn_id"], row["position_id"], row["pos_rank"]
            if pid is None or pos is None or rank is None:
                continue
            out[int(pid)] = (int(pos), int(rank))
        return out

    def comments(self) -> dict[int, str]:
        """`{espn_id: the analyst's note}`, empty when the board carries none."""
        if "comment" not in self.frame.columns:
            return {}
        return {
            int(r["espn_id"]): r["comment"]
            for r in self.frame.iter_rows(named=True)
            if r["espn_id"] is not None and r["comment"]
        }

    def disagreements(self, *, minimum: float = 12.0) -> pl.DataFrame:
        """Players ETR rates very differently from the field's ADP.

        `rank_vs_adp` is positive when ADP is LATER than ETR's rank, i.e. ETR
        likes the player more than the room does. ETR publishes this column
        itself; we recompute it so a schema change cannot silently invert it.
        """
        if "adp" not in self.frame.columns:
            raise EtrError(
                f"{self.path.name} carries no ADP column, so there is no field price to "
                "disagree with. The Top 150 is an in-season board; ADP only exists on the "
                "Draft Kit chart."
            )
        return (
            self.frame.with_columns((pl.col("adp") - pl.col("etr_rank")).alias("etr_edge"))
            .filter(pl.col("etr_edge").abs() >= minimum)
            .sort("etr_edge", descending=True)
        )


def _rename_map(columns: Iterable[str]) -> tuple[dict[str, str], str]:
    """`(original -> ours, kind)`. Raises when no rank column is recognisable."""
    mapping: dict[str, str] = {}
    kinds: list[str] = []
    for col in columns:
        key = _norm(col)
        if key in RANK_COLUMNS:
            mapping[col] = "etr_rank"
            kinds.append(RANK_COLUMNS[key])
        elif key in POS_RANK_COLUMNS:
            mapping[col] = "pos_rank_raw"
        elif key in COMMENT_COLUMNS:
            mapping[col] = "comment"
        elif key in COLUMNS:
            mapping[col] = COLUMNS[key]
    return mapping, (kinds[0] if kinds else "")


def _split_pos_rank(value: object) -> int | None:
    """`"RB01"` -> 1. The prefix is dropped: the `position` column is the authority."""
    if value is None:
        return None
    m = _POS_RANK.match(str(value).strip())
    return int(m.group(2)) if m else None


def load_file(
    path: Path | str,
    *,
    scoring: str | None = None,
    id_index: object | None = None,
) -> EtrRankings:
    """Read one ETR rankings CSV.

    Validates the columns on every ingest: the schema is undocumented and can
    move without warning, and a silently mis-parsed ranking set is worse than a
    missing one.

    A board with no `id` column is resolved by name through `EspnIdIndex`, which
    returns None rather than a plausible wrong player. Pass `id_index` to supply
    one; the default is built lazily so a board that ships ids never pays for the
    crosswalk.
    """
    path = Path(path)
    if not path.exists():
        raise EtrError(f"no ETR export at {path}")

    raw = pl.read_csv(path, encoding="utf8-lossy", infer_schema_length=None)
    mapping, kind = _rename_map(raw.columns)
    frame = raw.rename(mapping).select(list(mapping.values()))

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise EtrError(
            f"{path.name} is missing {missing}; got columns {raw.columns}. "
            "The ETR export schema is undocumented -- re-download, and if the "
            "header really changed, update COLUMNS or RANK_COLUMNS."
        )

    casts = [pl.col("etr_rank").cast(pl.Int64, strict=False)]
    if "adp" in frame.columns:
        casts.append(pl.col("adp").cast(pl.Float64, strict=False))
    if "espn_id" in frame.columns:
        casts.append(pl.col("espn_id").cast(pl.Int64, strict=False))
    frame = frame.with_columns(casts).drop_nulls(["etr_rank"])

    name_match: tuple[int, int] | None = None
    if "espn_id" in frame.columns:
        frame = frame.drop_nulls(["espn_id"])
    else:
        frame, name_match = _resolve_names(frame, path, id_index)

    frame = _add_positions(frame)

    if frame.is_empty():
        raise EtrError(f"{path.name} parsed to zero usable rows")

    return EtrRankings(
        scoring=scoring or _scoring_from_name(path),
        kind=kind or "etr",
        path=path,
        frame=frame,
        name_match=name_match,
    )


def _resolve_names(
    frame: pl.DataFrame, path: Path, id_index: object | None
) -> tuple[pl.DataFrame, tuple[int, int]]:
    """Name -> ESPN id for a board with no id column.

    Imported here rather than at module scope: `projections.sources` pulls in the
    whole id crosswalk, and a board that ships ESPN ids must not pay for it.
    """
    if id_index is None:
        from ..projections.sources import default_id_index

        id_index = default_id_index()

    resolve = id_index.from_name  # type: ignore[attr-defined]
    teams = frame["team"].to_list() if "team" in frame.columns else [None] * frame.height
    poss = frame["position"].to_list() if "position" in frame.columns else [None] * frame.height
    ids = [resolve(n, t, p) for n, t, p in zip(frame["player"], teams, poss, strict=True)]

    total = len(ids)
    hit = sum(1 for i in ids if i is not None)
    if total and hit / total < MIN_NAME_MATCH:
        unmatched = [n for n, i in zip(frame["player"], ids, strict=True) if i is None]
        raise EtrError(
            f"{path.name} has no id column and only {hit}/{total} names resolved to an "
            f"ESPN id, below the {MIN_NAME_MATCH:.0%} floor. Unresolved: {unmatched[:10]}. "
            "A half-matched board silently re-ranks only the half it matched."
        )
    if hit < total:
        log.info("%s: %d/%d names resolved to an ESPN id", path.name, hit, total)

    frame = frame.with_columns(pl.Series("espn_id", ids, dtype=pl.Int64)).drop_nulls(["espn_id"])
    return frame, (hit, total)


def _add_positions(frame: pl.DataFrame) -> pl.DataFrame:
    """Add `position_id` and `pos_rank`, deriving the rank when the board omits it."""
    if "position" in frame.columns:
        pos_ids = [POSITION_IDS.get(str(p).strip().upper()) for p in frame["position"]]
    else:
        pos_ids = [None] * frame.height
    frame = frame.with_columns(pl.Series("position_id", pos_ids, dtype=pl.Int64))

    if "pos_rank_raw" in frame.columns:
        ranks = [_split_pos_rank(v) for v in frame["pos_rank_raw"]]
        frame = frame.with_columns(pl.Series("pos_rank", ranks, dtype=pl.Int64)).drop(
            "pos_rank_raw"
        )
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("pos_rank"))

    # A board that publishes no positional rank still has one implied by its own
    # ordering, and `positional()` is the accessor everything downstream uses. Deriving
    # it is exact -- it is a dense rank of `etr_rank` inside each position.
    return frame.with_columns(
        pl.when(pl.col("pos_rank").is_null())
        .then(pl.col("etr_rank").rank("ordinal").over("position_id").cast(pl.Int64))
        .otherwise(pl.col("pos_rank"))
        .alias("pos_rank")
    )


def _scoring_from_name(path: Path) -> str:
    stem = _norm(path.stem)
    if "halfppr" in stem or "half" in stem:
        return "half_ppr"
    if "fullppr" in stem or "ppr" in stem:
        return "ppr"
    return "unknown"


def load_all(
    directory: Path | str = DEFAULT_DIR, *, id_index: object | None = None
) -> dict[tuple[str, str], EtrRankings]:
    """Every export in the watched folder, keyed by `(kind, scoring)`.

    Keyed on the pair rather than scoring alone because the Draft Kit board and the
    Top 150 are both half PPR and would otherwise overwrite each other, leaving
    whichever sorted last.

    One `EspnIdIndex` is built for the whole folder and only if some board actually
    needs it, rather than one per file.
    """
    directory = Path(directory)
    if not directory.exists():
        return {}
    out: dict[tuple[str, str], EtrRankings] = {}
    shared = _IndexCache(id_index)
    for path in sorted(directory.glob("*.csv")):
        try:
            board = load_file(path, id_index=shared)
        except EtrError as exc:
            log.warning("skipping %s: %s", path.name, exc)
            continue
        out[(board.kind, board.scoring)] = board
    return out


class _IndexCache:
    """Builds the id crosswalk at most once, and only if a board asks for it."""

    __slots__ = ("_index", "_built")

    def __init__(self, index: object | None = None) -> None:
        self._index = index
        self._built = index is not None

    def from_name(self, name: str, team: object = None, position: object = None) -> int | None:
        if not self._built:
            from ..projections.sources import default_id_index

            self._index = default_id_index()
            self._built = True
        return self._index.from_name(name, team, position)  # type: ignore[union-attr]


def for_league(
    scoring_variant: str,
    directory: Path | str = DEFAULT_DIR,
    *,
    kind: str = "etr",
    id_index: object | None = None,
) -> EtrRankings | None:
    """The board matching a league's scoring, or None if it was not downloaded.

    Never silently substitutes the other format: a full-PPR board applied to a
    half-PPR league systematically overrates receivers, which is exactly the sort
    of quiet error this project keeps finding. A caller that wants the other format
    anyway asks for it by name and says so -- see `applies_to` and the module
    docstring's measurement of what that costs.
    """
    return load_all(directory, id_index=id_index).get((kind, scoring_variant))


def best_available(
    scoring_variant: str,
    directory: Path | str = DEFAULT_DIR,
    *,
    kind: str = "etr",
    id_index: object | None = None,
) -> tuple[EtrRankings | None, bool]:
    """`(board, matches_scoring)` -- the league's own board, else the other format.

    The deliberate substitution `for_league` refuses to make on its own. The flag is
    not decoration: the caller is expected to log it, because a consumer that reads
    the overall rank rather than `positional()` must not take the fallback at all.
    """
    boards = load_all(directory, id_index=id_index)
    exact = boards.get((kind, scoring_variant))
    if exact is not None:
        return exact, True
    other = [b for (k, _), b in sorted(boards.items()) if k == kind]
    return (other[0] if other else None), False


def coverage(board: EtrRankings, espn_ids: Iterable[int]) -> tuple[int, int]:
    """(matched, total) for a set of players -- e.g. a roster or a waiver board."""
    ids = list(espn_ids)
    known = set(board.frame["espn_id"].to_list())
    return sum(1 for i in ids if i in known), len(ids)


def archive(
    path: Path | str,
    *,
    root: Path | str = DEFAULT_ARCHIVE,
    today: dt.date | None = None,
) -> Path | None:
    """Keep a dated copy of one export. Returns the copy, or None if it already exists.

    The only thing that can ever answer "does this analyst's ordering actually beat
    ESPN's". No historical boards exist, ETR overwrites each chart in place, and
    `Rank Change` came back zero on 148 of the Top 150's rows because this is the
    season's first in-season publication -- so the comparison set has to be
    accumulated going forward or it never exists at all. Costs one file copy a week
    and cannot be backfilled.

    Never overwrites: a second run on the same day is a no-op, so this is safe to
    call on every load.
    """
    path = Path(path)
    if not path.exists():
        raise EtrError(f"no ETR export at {path}")
    stamp = (today or dt.date.today()).isoformat()
    target = Path(root) / stamp / path.name
    if target.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    log.info("archived %s to %s", path.name, target)
    return target
