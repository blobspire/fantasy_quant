"""Establish The Run rankings ingest.

ETR has no API and their terms prohibit automated scraping, so the only path is
the CSV download button on each rankings chart. This reads what that button
produces.

Two things about the export shape drive the design:

* **It is RANKINGS, not projections.** The Draft Kit export carries rank, ADP and
  the gap between them -- no component stats. So it cannot feed the ensemble,
  which combines at the stat level precisely so one projection set can serve
  leagues with different scoring. ETR enters instead as an independent opinion to
  arbitrage against, which is arguably where a human ranking set belongs anyway.
* **It ships the ESPN player id.** The `id` column joined 300/300 against our 2026
  pool with no fuzzy matching, which makes this the cleanest external source we
  have. Nothing else we ingest is keyed to ESPN for free.

Scoring is per file, not per row: the Full PPR and Half PPR charts are separate
downloads. Keep them in separate files and load the one matching the league.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

DEFAULT_DIR = Path("data/manual/etr")

#: Column names as ETR writes them, mapped to ours. The export has a BOM and
#: quotes every field, so read with encoding="utf8-lossy" and normalise here
#: rather than trusting the header bytes.
COLUMNS: dict[str, str] = {
    "player": "player",
    "position": "position",
    "team": "team",
    "etrrank": "etr_rank",
    "adp": "adp",
    "rankingdiff": "rank_vs_adp",
    "etrposrank": "etr_pos_rank",
    "adpposrank": "adp_pos_rank",
    "posrankdiff": "pos_rank_vs_adp",
    "id": "espn_id",
}

REQUIRED = ("player", "etr_rank", "espn_id")


class EtrError(RuntimeError):
    """The export could not be read as an ETR rankings file."""


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True, slots=True)
class EtrRankings:
    """One ETR rankings chart, joined to ESPN ids."""

    scoring: str
    path: Path
    frame: pl.DataFrame

    @property
    def n(self) -> int:
        return self.frame.height

    def rank_of(self, espn_id: int) -> int | None:
        hit = self.frame.filter(pl.col("espn_id") == espn_id)
        return int(hit["etr_rank"][0]) if hit.height else None

    def as_map(self) -> dict[int, int]:
        return dict(zip(self.frame["espn_id"], self.frame["etr_rank"], strict=True))

    def disagreements(self, *, minimum: float = 12.0) -> pl.DataFrame:
        """Players ETR rates very differently from the field's ADP.

        `rank_vs_adp` is positive when ADP is LATER than ETR's rank, i.e. ETR
        likes the player more than the room does. ETR publishes this column
        itself; we recompute it so a schema change cannot silently invert it.
        """
        return (
            self.frame.with_columns((pl.col("adp") - pl.col("etr_rank")).alias("etr_edge"))
            .filter(pl.col("etr_edge").abs() >= minimum)
            .sort("etr_edge", descending=True)
        )


def load_file(path: Path | str, *, scoring: str | None = None) -> EtrRankings:
    """Read one ETR rankings CSV.

    Validates the columns on every ingest: the schema is undocumented and can
    move without warning, and a silently mis-parsed ranking set is worse than a
    missing one.
    """
    path = Path(path)
    if not path.exists():
        raise EtrError(f"no ETR export at {path}")

    raw = pl.read_csv(path, encoding="utf8-lossy", infer_schema_length=None)
    mapping = {c: COLUMNS[_norm(c)] for c in raw.columns if _norm(c) in COLUMNS}
    frame = raw.rename(mapping).select(list(mapping.values()))

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise EtrError(
            f"{path.name} is missing {missing}; got columns {raw.columns}. "
            "The ETR export schema is undocumented -- re-download, and if the "
            "header really changed, update COLUMNS."
        )

    frame = frame.with_columns(
        pl.col("espn_id").cast(pl.Int64, strict=False),
        pl.col("etr_rank").cast(pl.Int64, strict=False),
        pl.col("adp").cast(pl.Float64, strict=False),
    ).drop_nulls(["espn_id", "etr_rank"])

    if frame.is_empty():
        raise EtrError(f"{path.name} parsed to zero usable rows")

    return EtrRankings(scoring=scoring or _scoring_from_name(path), path=path, frame=frame)


def _scoring_from_name(path: Path) -> str:
    stem = _norm(path.stem)
    if "halfppr" in stem or "half" in stem:
        return "half_ppr"
    if "fullppr" in stem or "ppr" in stem:
        return "ppr"
    return "unknown"


def load_all(directory: Path | str = DEFAULT_DIR) -> dict[str, EtrRankings]:
    """Every export in the watched folder, keyed by scoring format."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    out: dict[str, EtrRankings] = {}
    for path in sorted(directory.glob("*.csv")):
        try:
            board = load_file(path)
        except EtrError as exc:
            log.warning("skipping %s: %s", path.name, exc)
            continue
        out[board.scoring] = board
    return out


def for_league(scoring_variant: str, directory: Path | str = DEFAULT_DIR) -> EtrRankings | None:
    """The board matching a league's scoring, or None if it was not downloaded.

    Never silently substitutes the other format: a full-PPR board applied to a
    half-PPR league systematically overrates receivers, which is exactly the sort
    of quiet error this project keeps finding.
    """
    return load_all(directory).get(scoring_variant)


def coverage(board: EtrRankings, espn_ids: Iterable[int]) -> tuple[int, int]:
    """(matched, total) for a set of players -- e.g. a roster or a waiver board."""
    ids = list(espn_ids)
    known = set(board.frame["espn_id"].to_list())
    return sum(1 for i in ids if i in known), len(ids)
