"""Projection sources, each presented as one `core.ProjectionSource`.

Everything upstream of the ensemble lands here and leaves in one shape:
`ComponentLine`s keyed by ESPN player id, carrying raw ESPN statIds. That is the
only vocabulary in which four feeds that disagree about everything else can be
averaged, and it is what lets one ensemble serve a 14-team full-PPR league and
two 12-team half-PPR ones without re-projecting anything.

Two design rules do most of the work here.

**A source being unavailable is normal, not exceptional.** Sleeper can 200 with a
skeleton, FanDuel can close a tab slug, the ETR folder can simply be empty
because nobody downloaded this week's CSV yet. So every adapter raises
`SourceUnavailable` rather than something bespoke, and `collect` records the
failure in `SourceBundle.errors` and returns whatever else answered. The ensemble
then renormalizes over the sources that are present -- which is exactly the
missing-source path it is built around, so an outage exercises a tested code path
instead of an untested one.

**Silence must be unambiguous.** Downstream, an absent stat means "this source
abstained" and gets renormalized away; it never means zero. That is only sound if
a source that *did* project a player emits explicit zeros for the stats it covers
but does not credit him with. So the full-slate adapters (ESPN, Sleeper, ETR)
densify: every stat in their vocabulary is present, zero included. Props do the
opposite and stay sparse, deliberately -- no posted receiving-yards market means
the book said nothing, not that it forecast zero yards. `dense` records which of
the two a source is.

Measured facts this module encodes, all from our own corpus rather than from docs:

* **ESPN's raw `stats` restate themselves.** On 9,360 weekly projection rows from
  the 2026 pool, statId 27 is exactly `floor(RY/5)`, 47 is exactly
  `floor(REY/5)`, 22/40/61 are byte-equal to 3/24/42, 2 is `PA - PC`, 39 is
  `RY/RA`, 60 is `REY/REC`, 73 is `INTT + FUML` -- zero mismatches in ~25k checks
  each. Ensembling those alongside the primitives would average a number with a
  stale restatement of itself, so `DERIVED_STAT_IDS` is dropped on ingest and
  `ensemble.add_derived_stats` puts it back from the *combined* primitives.
* **statId 210 (games played) is 1.0 on every weekly row**, projected or not. It
  is shape, not forecast, so it moves to `ComponentLine.games` instead of being
  averaged as if it were a stat.
* **Rush share of projected touchdowns**, needed to split an anytime-TD market
  into rushing and receiving TDs: QB 1.000, RB 0.802, WR 0.031, TE 0.014
  (ESPN 2026 weeks 1-18, 9,360 rows). See `RUSH_SHARE_PRIOR`.

Establish The Run is manual-CSV only. Their ToS prohibits automated collection and
there is no API, so there is deliberately no fetch path in `EtrCsvSource` -- it
watches `data/manual/etr/` and nothing else. Because that CSV's schema is
undocumented and has no stability guarantee, every ingest re-validates the header
and a drifted file fails loudly with the columns actually seen.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from ..core import ComponentLine
from ..data import ids as ids_module
from ..data import props as props_module
from ..data import sleeper as sleeper_module
from ..espn.statrows import SOURCE_PROJECTED, SPLIT_GAME

log = logging.getLogger(__name__)


#: Score floor for a position-hinted name retry: exact matches only, fuzzy
#: nickname stage off. See `EspnIdIndex.from_name` for what happens without it.
HINT_MIN_SCORE = 1.0


class SourceUnavailable(RuntimeError):
    """This source cannot answer for this week. Expected, and never fatal.

    Every adapter narrows its own upstream exception to this one type so that
    `collect` has a single thing to catch, and so that a caller cannot
    accidentally treat "Sleeper is down" as a programming error.
    """


# --------------------------------------------------------------------------------------
# ESPN statId vocabulary
# --------------------------------------------------------------------------------------
# Names and ids from `seasons/{year}?view=chui_default_platformsettings`, not from a
# blog post. Only the ones some adapter actually maps to are named.

PASS_ATT = "0"
PASS_CMP = "1"
PASS_YDS = "3"
PASS_TD = "4"
PASS_2PT = "19"
PASS_INT = "20"
RUSH_ATT = "23"
RUSH_YDS = "24"
RUSH_TD = "25"
RUSH_2PT = "26"
REC_YDS = "42"
REC_TD = "43"
REC_2PT = "44"
RECEPTIONS = "53"
TARGETS = "58"
FUMBLES_LOST = "72"
FG_MADE = "83"
FG_ATT = "84"
PAT_MADE = "86"
PAT_ATT = "87"
GAMES_PLAYED = "210"
PASS_FIRST_DOWNS = "211"
RUSH_FIRST_DOWNS = "212"
REC_FIRST_DOWNS = "213"

#: statIds ESPN publishes that are exact restatements of other statIds in the same
#: row. Verified on the 2026 corpus (see the module docstring); the counts are the
#: number of rows in which each rule held with zero exceptions.
#:
#: Dropped on ingest rather than ensembled. They are recoverable from the combined
#: primitives -- `ensemble.add_derived_stats` does exactly that -- and keeping them
#: would mean averaging, say, receiving yards and a floor-bucket of a *different*
#: source's receiving yards into one line that no longer agrees with itself.
DERIVED_STAT_IDS: frozenset[str] = frozenset(
    {
        "2",  # incompletions = PA - PC                          (1,088 rows)
        "5",  # every 5 passing yards = floor(PY/5)              (1,088)
        "6",  # every 10 passing yards                           (555)
        "7",  # every 20                                         (544)
        "8",  # every 25                                         (544)
        "9",  # every 50                                         (544)
        "10",  # every 100                                       (544)
        "11",  # every 5 completions = floor(PC/5)               (544)
        "12",  # every 10 completions                            (544)
        "13",  # every 5 incompletions -- never present live
        "14",  # every 10 incompletions -- never present live
        "21",  # completion pct = PC/PA                          (1,088)
        "22",  # passing yards per game == PY on a weekly row    (1,088)
        "27",  # every 5 rushing yards = floor(RY/5)             (1,881)
        "28",  # every 10                                        (1,602)
        "29",  # every 20                                        (1,123)
        "30",  # every 25                                        (957)
        "31",  # every 50                                        (519)
        "32",  # every 100 -- never present live
        "33",  # every 5 rush attempts = floor(RA/5)             (1,012)
        "34",  # every 10 rush attempts                          (632)
        "39",  # yards per carry = RY/RA                         (4,044)
        "40",  # rushing yards per game == RY                    (4,044)
        "41",  # receptions restated -- never present live
        "47",  # every 5 receiving yards = floor(REY/5)          (4,385)
        "48",  # every 10                                        (3,530)
        "49",  # every 20                                        (2,412)
        "50",  # every 25                                        (2,003)
        "51",  # every 50                                        (730)
        "52",  # every 100 -- never present live
        "54",  # every 5 receptions -- never present live
        "55",  # every 10 receptions -- never present live
        "60",  # yards per catch = REY/REC                       (5,894)
        "61",  # receiving yards per game == REY                 (5,894)
        "73",  # total turnovers = INTT + FUML                   (7,097)
    }
)


# --------------------------------------------------------------------------------------
# Source containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceLines:
    """One source's answer for one player-week slate.

    Richer than the bare `Sequence[ComponentLine]` the `ProjectionSource` protocol
    requires, because the ensemble needs two things `ComponentLine` deliberately
    does not carry: the position (nothing can be *scored* without it, and only
    some sources know it) and whether silence means zero.

    `unresolved` is not decoration. A feed that publishes names rather than ids
    loses rows at the crosswalk, and a source that silently halves in coverage
    looks exactly like a source that got quieter -- so the count travels with the
    data and `collect` logs it.
    """

    source: str
    season: int
    week: int
    lines: tuple[ComponentLine, ...] = ()
    #: espn player id -> defaultPositionId, for whatever subset this source knows.
    positions: Mapping[int, int] = field(default_factory=dict)
    #: espn player id -> display name, for reporting.
    names: Mapping[int, str] = field(default_factory=dict)
    #: True when an absent stat inside this source's vocabulary means zero.
    #: False for market data, where absence means "nothing was posted".
    dense: bool = True
    #: Rows this source published that could not be mapped to an ESPN id.
    unresolved: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def player_ids(self) -> frozenset[int]:
        return frozenset(line.player_id for line in self.lines)


class SourceLinesLike(Protocol):
    """Structural type the ensemble consumes. `SourceLines` satisfies it."""

    source: str
    lines: tuple[ComponentLine, ...]
    positions: Mapping[int, int]
    names: Mapping[int, str]


@dataclass(slots=True)
class SourceBundle:
    """Whatever answered this week, and what each failure was.

    Same shape and same reasoning as `props.PropsBundle`: these feeds are free,
    undocumented and independently mortal, so the pipeline runs on what is left.
    """

    season: int
    week: int
    sources: list[SourceLines] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def live(self) -> tuple[str, ...]:
        return tuple(s.source for s in self.sources)

    def get(self, source: str) -> SourceLines | None:
        for s in self.sources:
            if s.source == source:
                return s
        return None

    def summary(self) -> str:
        parts = [f"{s.source}:{len(s.lines)}" for s in self.sources]
        parts += [f"{name}:FAILED" for name in self.errors]
        return f"season {self.season} week {self.week} -- " + " ".join(parts or ["nothing"])


class ProjectionAdapter(Protocol):
    """What `collect` needs. A superset of `core.ProjectionSource`."""

    name: str

    def source_lines(self, season: int, week: int) -> SourceLines: ...

    def component_lines(self, season: int, week: int) -> Sequence[ComponentLine]: ...


def collect(
    adapters: Iterable[ProjectionAdapter],
    season: int,
    week: int,
) -> SourceBundle:
    """Run every adapter, tolerating any subset of them being dead.

    A source that raises lands in `errors` and the rest continue. An adapter that
    returns zero lines is *also* recorded as an error: a 200 with an empty body is
    this codebase's most frequently observed failure mode (ESPN's unknown `view=`,
    Sleeper's bogus week, FanDuel's bad tab slug), and treating it as a successful
    empty slate is how that failure gets into a projection silently.
    """
    bundle = SourceBundle(season=season, week=week)
    for adapter in adapters:
        name = getattr(adapter, "name", adapter.__class__.__name__)
        try:
            lines = adapter.source_lines(season, week)
        except SourceUnavailable as exc:
            log.warning("source %s unavailable for %d week %d: %s", name, season, week, exc)
            bundle.errors[name] = str(exc)
            continue
        if not lines.lines:
            bundle.errors[name] = "returned no lines"
            log.warning("source %s returned no lines for %d week %d", name, season, week)
            continue
        if lines.unresolved:
            log.warning(
                "source %s: %d rows dropped for want of an ESPN id (e.g. %s)",
                name,
                len(lines.unresolved),
                ", ".join(lines.unresolved[:5]),
            )
        bundle.sources.append(lines)
    if not bundle.sources:
        log.error("every projection source failed: %s", bundle.errors)
    return bundle


# --------------------------------------------------------------------------------------
# ID resolution
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class EspnIdIndex:
    """Anything -> ESPN player id, the id space `ComponentLine` is keyed in.

    Thin on purpose: `data.ids` already owns the hard parts (the `"NA"` trap, the
    32-row D/ST table, refusing ambiguous names). This adds the last hop --
    canonical id to ESPN id as an `int` -- plus a manual override table, because
    a props feed will always have a handful of names the crosswalk cannot place
    and a hand-fix must not require editing the crosswalk.
    """

    resolver: ids_module.IdResolver | None = None
    #: Normalized name -> ESPN id. Escape hatch for the residual name misses.
    overrides: Mapping[str, int] = field(default_factory=dict)
    _name_cache: dict[tuple[str, str | None, str | None, tuple[str, ...]], int | None] = field(
        default_factory=dict, repr=False
    )

    def from_canonical(self, canonical: str | None) -> int | None:
        if not canonical or self.resolver is None:
            return None
        raw = self.resolver.from_canonical(canonical, ids_module.ESPN)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def from_sleeper(self, sleeper_id: str) -> int | None:
        """Sleeper id -> ESPN id, defenses included.

        Sleeper's own `espn_id` field covers only ~24% of the master and is not
        used; this goes through the crosswalk spine, where a team defense is the
        bare abbreviation ``"LV"`` and resolves to ESPN's synthetic ``-16013``.
        """
        if self.resolver is None:
            return None
        return self.from_canonical(self.resolver.to_canonical(sleeper_id, ids_module.SLEEPER))

    def from_espn(self, value: object) -> int | None:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def from_name(
        self,
        name: str,
        team: object = None,
        position: object = None,
        position_hints: Sequence[str] = (),
    ) -> int | None:
        """Display name -> ESPN id, or None rather than a plausible wrong player.

        `IdResolver.resolve_name` refuses ambiguity by design (measured: zero wrong
        matches over 1,028 live pool entries, at the cost of ~2% misses). Keep it
        that way -- a wrong player is invisible in a projection and a missing one
        is loud.

        `position_hints` is a *fallback* for feeds that publish a bare name, and it
        does not weaken that discipline. It is tried only after the unhinted
        resolution has already returned None, each hint is resolved separately, and
        the answer is taken only when every hint that matched agrees on one player.
        The hinted retry runs at `min_score=1.0`, i.e. exact/suffix-stripped/
        compacted matches only, with the fuzzy nickname stage switched off. That
        is measured, not cautious-by-default: at the ordinary floor the retry
        matched the book's "Kyle Williams" onto **Kyren Williams** (ESPN 4430737)
        -- a confident wrong player, invisible in a projection. At 1.0 it resolves
        "Justin Jefferson" and "Lamar Jackson", who share their exact names with
        defensive backs in the crosswalk and were being dropped, and refuses
        "Kyle Williams" as it should. On the live 2026 week-1 board that took the
        props feed from 9 unresolved players to 1, with zero wrong matches.
        """
        # The hints are part of the key. Without them, an ETR row that resolved
        # to None with no context would poison the props lookup for the same name
        # a moment later, and the two adapters share one index by design.
        key = (
            name,
            str(team) if team is not None else None,
            str(position) if position is not None else None,
            tuple(position_hints),
        )
        if key in self._name_cache:
            return self._name_cache[key]
        result = self.overrides.get(_normalize_key(name))
        if result is None and self.resolver is not None:
            match = self.resolver.resolve_name(name, team, position)
            result = self.from_canonical(match.canonical) if match else None
        if result is None and self.resolver is not None and position_hints:
            candidates = {
                found
                for hint in position_hints
                if (m := self.resolver.resolve_name(name, team, hint, min_score=HINT_MIN_SCORE))
                is not None
                and (found := self.from_canonical(m.canonical)) is not None
            }
            result = candidates.pop() if len(candidates) == 1 else None
        if result is None and (team_defense := book_defense(name)) is not None:
            result = team_defense.espn_player_id
        self._name_cache[key] = result
        return result


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


#: Longest first: "New England Team Defense" must strip "teamdefense", not "defense".
_DEFENSE_SUFFIXES: tuple[str, ...] = ("teamdefense", "defense", "dst", "def", "d")

#: Location -> defense, for the locations that name exactly one team. New York
#: and Los Angeles are absent by construction: two teams each, and guessing one
#: is precisely the silent cross-wiring `data.ids` refuses to do.
_DST_BY_LOCATION: Mapping[str, ids_module.TeamDefense] = {
    key: team
    for key, team in (
        (_normalize_key(t.location), t)
        for t in ids_module.DST_TEAMS
        if sum(1 for o in ids_module.DST_TEAMS if o.location == t.location) == 1
    )
}


def book_defense(name: str) -> ids_module.TeamDefense | None:
    """A sportsbook's spelling of a team defense -> the defense.

    `ids.resolve_dst` knows "Seahawks D/ST", "Seattle Seahawks" and "SEA"; live
    FanDuel posts **"Seattle Defense"**, which none of those match, and every
    D/ST prop was landing in `unresolved` because of it. RESEARCH is blunt that
    every join failure in testing was a team defense, so this is the expected
    place for one.

    The location index is derived from `ids.DST_TEAMS` rather than typed out
    again, and it deliberately omits New York and Los Angeles: "New York
    Defense" is two teams and answering with either one is a wrong answer that
    nothing downstream can see.
    """
    direct = ids_module.resolve_dst(name)
    if direct is not None:
        return direct
    normalized = _normalize_key(name)
    for suffix in _DEFENSE_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            stem = normalized[: -len(suffix)]
            return ids_module.resolve_dst(stem) or _DST_BY_LOCATION.get(stem)
    return None


def default_id_index(*, allow_download: bool = False, season: int | None = None) -> EspnIdIndex:
    """Build an index from the cached crosswalks, or an empty one that resolves nothing.

    Never raises. A missing crosswalk must not take the projection feeds offline --
    ESPN's own rows are already keyed by ESPN id and need no resolution at all --
    but it is logged, because a silently id-less props feed just looks like a
    quiet week at the sportsbook.
    """
    try:
        resolver = ids_module.IdResolver.load(season, allow_download=allow_download)
    except Exception:  # noqa: BLE001 -- any failure here is degraded, never fatal
        log.warning(
            "id crosswalk unavailable (allow_download=%s); name- and Sleeper-keyed "
            "sources will resolve nothing",
            allow_download,
            exc_info=True,
        )
        return EspnIdIndex()
    return EspnIdIndex(resolver=resolver)


# --------------------------------------------------------------------------------------
# ESPN
# --------------------------------------------------------------------------------------

DEFAULT_SNAPSHOT_ROOT = Path("data/snapshots/espn")

ESPN_SOURCE = "espn"


def _stats_from_lists(
    stat_ids: Sequence[str] | None,
    stat_values: Sequence[float] | None,
    *,
    keep_derived: bool,
) -> dict[str, float]:
    if not stat_ids or not stat_values:
        return {}
    out: dict[str, float] = {}
    for stat_id, value in zip(stat_ids, stat_values, strict=False):
        key = str(stat_id)
        if not keep_derived and key in DERIVED_STAT_IDS:
            continue
        out[key] = float(value)
    return out


def snapshot_files(
    season: int, *, root: Path = DEFAULT_SNAPSHOT_ROOT, variant: str = "ppr"
) -> list[Path]:
    """Capture files for one season, oldest first. Filenames are ISO dates, so they sort."""
    directory = Path(root) / f"season={season}" / f"variant={variant}"
    return sorted(directory.glob("*.parquet"))


class EspnSnapshotSource:
    """ESPN's own weekly projections, read from the Parquet corpus.

    The corpus rather than the live pool by default, for two reasons: it is the
    only way to get *past* weeks (which is what calibration and any backtest
    need), and the daily capture already ran. `EspnLiveSource` is the same rows
    fetched fresh when today's file does not exist yet.

    Two filters here are load-bearing rather than tidy:

    * `stat_season == season` -- a 2026 request returns 2025 rows in the same
      array, and array position tells you nothing.
    * `statSourceId == 1, statSplitTypeId == 1` -- projected, single game. The
      season-total projection (split 0) is frozen at preseason and never revised,
      and the ROS row (split 2) is a per-game rate that must never be summed.
    """

    name = ESPN_SOURCE

    def __init__(
        self,
        *,
        root: Path = DEFAULT_SNAPSHOT_ROOT,
        variant: str = "ppr",
        frame: pl.DataFrame | None = None,
        keep_derived: bool = False,
    ) -> None:
        self.root = Path(root)
        self.variant = variant
        self.keep_derived = keep_derived
        self._frame = frame
        self._cache: dict[int, pl.DataFrame] = {}

    def _load(self, season: int) -> pl.DataFrame:
        if self._frame is not None:
            return self._frame
        cached = self._cache.get(season)
        if cached is not None:
            return cached
        files = snapshot_files(season, root=self.root, variant=self.variant)
        if not files:
            raise SourceUnavailable(
                f"no ESPN snapshot for season {season} variant {self.variant} under "
                f"{self.root}; run `fq snapshot` or `fq backfill` first"
            )
        frame = pl.read_parquet(files[-1])
        self._cache[season] = frame
        return frame

    def positions(self, season: int, week: int) -> Mapping[int, int]:
        """espn id -> defaultPositionId. The one source that always knows."""
        return self.source_lines(season, week).positions

    def source_lines(self, season: int, week: int) -> SourceLines:
        frame = self._load(season)
        wanted = frame.filter(
            (pl.col("stat_season") == season)
            & (pl.col("stat_source_id") == SOURCE_PROJECTED)
            & (pl.col("stat_split_type_id") == SPLIT_GAME)
            & (pl.col("scoring_period_id") == week)
        )
        lines: list[ComponentLine] = []
        positions: dict[int, int] = {}
        names: dict[int, str] = {}
        for row in wanted.iter_rows(named=True):
            player_id = row.get("espn_id")
            if player_id is None:
                continue
            stats = _stats_from_lists(
                row.get("stat_ids"), row.get("stat_values"), keep_derived=self.keep_derived
            )
            if not stats:
                continue
            games = stats.pop(GAMES_PLAYED, None)
            player_id = int(player_id)
            lines.append(
                ComponentLine(
                    player_id=player_id,
                    season=season,
                    week=week,
                    source=self.name,
                    stats=stats,
                    games=games,
                )
            )
            position = row.get("default_position_id")
            if position is not None:
                positions[player_id] = int(position)
            if row.get("full_name"):
                names[player_id] = str(row["full_name"])
        return SourceLines(
            source=self.name,
            season=season,
            week=week,
            lines=tuple(lines),
            positions=positions,
            names=names,
            dense=True,
        )

    def component_lines(self, season: int, week: int) -> Sequence[ComponentLine]:
        return self.source_lines(season, week).lines


class EspnLiveSource(EspnSnapshotSource):
    """The same rows, fetched from the public player pool instead of from disk.

    Unauthenticated: `leaguedefaults/{n}` is a real, league-independent pool. The
    scoring variant only affects `appliedTotal`, which this adapter never reads --
    the raw component counts are identical across variants, which is the whole
    premise of component-level ensembling.
    """

    def __init__(
        self,
        *,
        variant: str = "ppr",
        limit: int = 600,
        client: Any | None = None,
        keep_derived: bool = False,
    ) -> None:
        super().__init__(variant=variant, keep_derived=keep_derived)
        self.limit = limit
        self._client = client

    def _pool(self, season: int) -> list[dict[str, Any]]:
        from ..espn.client import EspnClient, EspnError, sort_by_projection
        from ..espn.endpoints import league_default_url

        client = self._client or EspnClient()
        owned = self._client is None
        try:
            return list(
                client.player_pool(
                    league_default_url(season, self.variant),
                    limit=min(self.limit, 250),
                    max_players=self.limit,
                    # A sort is mandatory (limit without one is a 400), and ranking
                    # by league-scored projection is the only way to get the top of
                    # the pool rather than 600 arbitrary players.
                    sort=sort_by_projection(season),
                    params={"view": "kona_player_info"},
                )
            )
        except EspnError as exc:
            raise SourceUnavailable(f"ESPN player pool unavailable: {exc}") from exc
        finally:
            if owned:
                client.close()

    def source_lines(self, season: int, week: int) -> SourceLines:
        from ..espn.statrows import parse_rows

        lines: list[ComponentLine] = []
        positions: dict[int, int] = {}
        names: dict[int, str] = {}
        for entry in self._pool(season):
            player = entry.get("player") or {}
            player_id = entry.get("id") or player.get("id")
            if player_id is None:
                continue
            player_id = int(player_id)
            for parsed in parse_rows(player.get("stats") or []):
                if parsed.season != season or parsed.scoring_period != week:
                    continue
                if parsed.source != SOURCE_PROJECTED or parsed.split != SPLIT_GAME:
                    continue
                stats = {
                    k: v
                    for k, v in parsed.stats.items()
                    if self.keep_derived or k not in DERIVED_STAT_IDS
                }
                if not stats:
                    continue
                games = stats.pop(GAMES_PLAYED, None)
                lines.append(
                    ComponentLine(
                        player_id=player_id,
                        season=season,
                        week=week,
                        source=self.name,
                        stats=stats,
                        games=games,
                    )
                )
                if player.get("defaultPositionId") is not None:
                    positions[player_id] = int(player["defaultPositionId"])
                if player.get("fullName"):
                    names[player_id] = str(player["fullName"])
                break
        return SourceLines(
            source=self.name,
            season=season,
            week=week,
            lines=tuple(lines),
            positions=positions,
            names=names,
            dense=True,
        )


# --------------------------------------------------------------------------------------
# Sleeper
# --------------------------------------------------------------------------------------

SLEEPER_SOURCE = "sleeper"

#: Sleeper component key -> ESPN statId. Sleeper is the only free feed that
#: publishes counts rather than points, which is the entire reason it is here.
#:
#: Every mapping here was checked against ESPN's own projection for the same 373
#: player-weeks (2026 week 1). Median Sleeper/ESPN ratio: passing attempts 1.03,
#: completions 1.03, passing yards 1.02, passing TDs 1.16, interceptions 1.11,
#: rush attempts 0.94, rushing yards 0.89, rushing TDs 0.90, receiving yards
#: 1.01, receiving TDs 0.98, receptions 0.98, targets 0.98. Those are two
#: forecasters disagreeing, which is the point of an ensemble.
#:
#: Three keys are deliberately NOT mapped, and this is the one place the obvious
#: mapping is wrong:
#:
#: **`pass_fd` / `rush_fd` / `rec_fd` are not ESPN's 211/212/213.** The names line
#: up and the numbers do not: median ratios against ESPN are 2.11, 1.58 and 2.07
#: over 32/102/207 players. It is not a calibration difference, it is a different
#: quantity -- live 2026 week 1, Puka Nacua carries `rec` 7.28 and `rec_fd` 9.61,
#: more first downs than catches, and a projected RB shows `rush_fd` 9.07 on 17.29
#: carries (a 52% conversion rate against a league norm near 23%). Whatever
#: Sleeper means by `fd`, it is not the count ESPN scores in a PPFD league, and
#: mapping it would corrupt exactly those leagues. ESPN remains the only source
#: for 211/212/213, at `source_count == 1`.
#:
#: `gp` is absent on purpose -- it is 1.0 on every weekly row and moves to
#: `ComponentLine.games`. The reception-distance buckets (`rec_0_4` ... `rec_40p`)
#: are absent too: ESPN's 47-52 are "every N receiving *yards*", a different thing
#: entirely, so there is no honest target statId and inventing a key outside the
#: ESPN space would put something in `stats` that no league scorer can read.
SLEEPER_TO_ESPN: Mapping[str, str] = {
    "pass_att": PASS_ATT,
    "pass_cmp": PASS_CMP,
    "pass_yd": PASS_YDS,
    "pass_td": PASS_TD,
    "pass_int": PASS_INT,
    "pass_2pt": PASS_2PT,
    "rush_att": RUSH_ATT,
    "rush_yd": RUSH_YDS,
    "rush_td": RUSH_TD,
    "rush_2pt": RUSH_2PT,
    "rec_tgt": TARGETS,
    "rec": RECEPTIONS,
    "rec_yd": REC_YDS,
    "rec_td": REC_TD,
    "rec_2pt": REC_2PT,
    "fum_lost": FUMBLES_LOST,
}

#: Sleeper keys we deliberately do not map. Named rather than merely absent, so
#: the exclusion is enforceable and a future "obvious" mapping has to argue with
#: a measurement; `test_ensemble.py` pins it.
#:
#: `fum` is here for a different reason than the first downs: `SleeperProjection`
#: reads through `components()`, which is keyed on `sleeper.COMPONENT_FIELDS`, and
#: total fumbles is not in it. Mapping it would zero-fill ESPN statId 68 on every
#: Sleeper row -- 0 of 373 live rows published it -- and halve the consensus for
#: the leagues that score total fumbles. Every mapped key must exist in
#: `COMPONENT_FIELDS` or the dense fill invents data rather than recording it.
SLEEPER_UNMAPPED: frozenset[str] = frozenset({"pass_fd", "rush_fd", "rec_fd", "fum"})

#: Sleeper `fantasy_positions` -> ESPN defaultPositionId.
SLEEPER_POSITION_IDS: Mapping[str, int] = {"QB": 1, "RB": 2, "WR": 3, "TE": 4, "K": 5, "DEF": 16}


class SleeperSource:
    """Sleeper's weekly component projections.

    Uses the weekly endpoint, never the season one: `rec_tgt` is published only
    on the weekly rows, and target volume is the most forward-stable component
    there is.

    Dense within `SLEEPER_TO_ESPN`. A Sleeper row that projects a receiver and
    omits `pass_yd` means zero passing yards, and saying so is what lets the
    ensemble read a *genuinely* absent source as an abstention.
    """

    name = SLEEPER_SOURCE

    def __init__(
        self,
        *,
        client: sleeper_module.SleeperClient | None = None,
        id_index: EspnIdIndex | None = None,
        positions: Sequence[str] = sleeper_module.SKILL_POSITIONS,
        company: str | None = None,
    ) -> None:
        self._client = client
        self.id_index = id_index or EspnIdIndex()
        self.positions = tuple(positions)
        self.company = company

    def _fetch(self, season: int, week: int) -> list[sleeper_module.SleeperProjection]:
        client = self._client or sleeper_module.SleeperClient()
        owned = self._client is None
        try:
            return client.weekly_projections(
                season, week, positions=self.positions, company=self.company
            )
        except sleeper_module.SleeperError as exc:
            raise SourceUnavailable(f"Sleeper unavailable: {exc}") from exc
        finally:
            if owned:
                client.close()

    def source_lines(self, season: int, week: int) -> SourceLines:
        lines: list[ComponentLine] = []
        positions: dict[int, int] = {}
        names: dict[int, str] = {}
        unresolved: list[str] = []
        for row in self._fetch(season, week):
            player_id = self.id_index.from_sleeper(row.sleeper_id)
            if player_id is None:
                unresolved.append(f"{row.name} ({row.sleeper_id})")
                continue
            components = row.components()
            stats = {
                stat_id: float(components.get(key, 0.0)) for key, stat_id in SLEEPER_TO_ESPN.items()
            }
            lines.append(
                ComponentLine(
                    player_id=player_id,
                    season=season,
                    week=week,
                    source=self.name,
                    stats=stats,
                    games=components.get("gp"),
                )
            )
            position = _sleeper_position_id(row)
            if position is not None:
                positions[player_id] = position
            names[player_id] = row.name
        return SourceLines(
            source=self.name,
            season=season,
            week=week,
            lines=tuple(lines),
            positions=positions,
            names=names,
            dense=True,
            unresolved=tuple(unresolved),
        )

    def component_lines(self, season: int, week: int) -> Sequence[ComponentLine]:
        return self.source_lines(season, week).lines


def _sleeper_position_id(row: sleeper_module.SleeperProjection) -> int | None:
    """Read `fantasy_positions`, not `position`.

    Sleeper's own filter keys on `fantasy_positions`, and `player.position` holds
    FB/CB/DB for rows Sleeper itself classifies as skill players. Trusting
    `position` here would tag four fullbacks as an unknown position and drop them.
    """
    for candidate in row.fantasy_positions:
        position = SLEEPER_POSITION_IDS.get(str(candidate).upper())
        if position is not None:
            return position
    return SLEEPER_POSITION_IDS.get(str(row.position or "").upper())


# --------------------------------------------------------------------------------------
# Sportsbook props
# --------------------------------------------------------------------------------------

PROPS_SOURCE = "props"

#: Prop component name -> ESPN statId.
PROPS_TO_ESPN: Mapping[str, str] = {
    props_module.PASSING_YARDS: PASS_YDS,
    props_module.PASSING_TDS: PASS_TD,
    props_module.PASSING_ATTEMPTS: PASS_ATT,
    props_module.PASSING_COMPLETIONS: PASS_CMP,
    props_module.INTERCEPTIONS: PASS_INT,
    props_module.RUSHING_YARDS: RUSH_YDS,
    props_module.RUSHING_ATTEMPTS: RUSH_ATT,
    props_module.RUSHING_TDS: RUSH_TD,
    props_module.RECEPTIONS: RECEPTIONS,
    props_module.RECEIVING_YARDS: REC_YDS,
    props_module.RECEIVING_TDS: REC_TD,
    props_module.FIELD_GOALS_MADE: FG_MADE,
    props_module.EXTRA_POINTS_MADE: PAT_MADE,
}

#: Market quantities that are already *scored*, in somebody else's league.
#: Ensembling them at the component level is a category error, so they are dropped
#: with a name rather than falling through an unmatched-key branch.
PROPS_ALREADY_SCORED: frozenset[str] = frozenset(
    {props_module.FANTASY_POINTS, props_module.KICKING_POINTS}
)

#: Share of a player's projected touchdowns that are rushing, by defaultPositionId.
#: Measured on 9,360 ESPN weekly projection rows (2026, weeks 1-18): the ratio of
#: summed projected rushing TDs to summed projected rushing + receiving TDs.
#:
#: An anytime-TD market prices P(any TD) and says nothing about which kind, so a
#: split is unavoidable. This is a prior, not a measurement of the market; a caller
#: with a better per-player split should pass `rush_share_for`.
RUSH_SHARE_PRIOR: Mapping[int, float] = {1: 1.000, 2: 0.802, 3: 0.031, 4: 0.014}

#: Used only when the player's position is unknown, which is the one case where
#: there is genuinely nothing to condition on. Half of an anytime TD in each
#: bucket is wrong for everybody and biased for nobody.
DEFAULT_RUSH_SHARE = 0.5

#: A mapping of espn id -> defaultPositionId, or something that builds one for a
#: given week. Positions are per-week because a player can change teams and, more
#: to the point, because the ESPN snapshot is the thing that knows them.
PositionLookup = Mapping[int, int] | Any


def _resolve_positions(lookup: PositionLookup | None, season: int, week: int) -> Mapping[int, int]:
    if lookup is None:
        return {}
    if isinstance(lookup, Mapping):
        return lookup
    try:
        return dict(lookup(season, week))
    except Exception:  # noqa: BLE001 -- a position hint is never worth a hard failure
        log.warning("position lookup failed for %d week %d", season, week, exc_info=True)
        return {}


#: Market -> the positions a player carrying it can plausibly be. Used only as a
#: disambiguation hint for the name join; see `EspnIdIndex.from_name`.
_PASSING_MARKETS: frozenset[str] = frozenset(
    {
        props_module.PASSING_YARDS,
        props_module.PASSING_TDS,
        props_module.PASSING_ATTEMPTS,
        props_module.PASSING_COMPLETIONS,
        props_module.INTERCEPTIONS,
    }
)
_RECEIVING_MARKETS: frozenset[str] = frozenset(
    {props_module.RECEPTIONS, props_module.RECEIVING_YARDS, props_module.RECEIVING_TDS}
)
_RUSHING_MARKETS: frozenset[str] = frozenset(
    {props_module.RUSHING_YARDS, props_module.RUSHING_ATTEMPTS, props_module.RUSHING_TDS}
)
_KICKING_MARKETS: frozenset[str] = frozenset(
    {props_module.KICKING_POINTS, props_module.FIELD_GOALS_MADE, props_module.EXTRA_POINTS_MADE}
)


def market_position_hints(components: Mapping[str, float]) -> tuple[str, ...]:
    """What positions a player with these markets could be.

    A book that posts passing yards for someone has told you he is a quarterback,
    which is all the crosswalk needs to separate the Ravens' Lamar Jackson from
    the cornerback of the same name. Coarse on purpose -- the hints are only ever
    used to break a tie, and a wrong one refuses rather than guesses.
    """
    keys = set(components)
    if keys & _PASSING_MARKETS:
        return ("QB",)
    if keys & _KICKING_MARKETS:
        return ("K",)
    if keys & _RECEIVING_MARKETS:
        return ("WR", "TE", "RB")
    if keys & _RUSHING_MARKETS:
        return ("RB", "QB", "WR")
    # An anytime-TD-only line says nothing about position.
    return ("QB", "RB", "WR", "TE")


class PropsSource:
    """Market-implied component means.

    Genuinely sharp and genuinely partial: a book posts receiving yards for two
    dozen players and nothing at all for the rest of the slate, so this source is
    **not** dense. Absence means "no market", never zero, and the ensemble
    renormalizes around it.

    Everything the market gives is already a mean rather than a median by the time
    it arrives here -- `props.project_event` does the ladder fit and the devig.
    That is the difference between reading a 29.5-yard line as 29.5 and reading it
    as the ~42 it implies.
    """

    name = PROPS_SOURCE

    def __init__(
        self,
        *,
        bundle: props_module.PropsBundle | None = None,
        id_index: EspnIdIndex | None = None,
        positions: PositionLookup | None = None,
        rush_share_for: Mapping[int, float] | None = None,
        collect_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._bundle = bundle
        self.id_index = id_index or EspnIdIndex()
        # A book publishes a name and a number and no position, so the anytime-TD
        # split has nothing to key its prior on unless somebody else supplies the
        # positions. `default_adapters` wires ESPN's in.
        self.positions = positions
        self.rush_share = dict(rush_share_for or RUSH_SHARE_PRIOR)
        self.collect_kwargs = dict(collect_kwargs or {})

    def _collect(self) -> props_module.PropsBundle:
        if self._bundle is not None:
            return self._bundle
        try:
            return props_module.collect_props(**self.collect_kwargs)
        except props_module.PropsError as exc:
            raise SourceUnavailable(f"props unavailable: {exc}") from exc

    def source_lines(self, season: int, week: int) -> SourceLines:
        bundle = self._collect()
        if not bundle.sources_live:
            raise SourceUnavailable(f"no prop source answered: {bundle.errors}")

        projections: list[props_module.PropProjection] = []
        for event in bundle.fanduel:
            projections.extend(props_module.project_event(event))
        for line in bundle.pinnacle:
            # No ALT ladder at Pinnacle, so there is no shape to fit and the posted
            # line has to stand in for the mean. That is known to be biased LOW for
            # every right-skewed stat -- `source="line"` marks it, and
            # `components_by_player` prefers a FanDuel ladder wherever one exists.
            projections.append(
                props_module.PropProjection(
                    player=line.player,
                    stat=line.stat,
                    book=line.book,
                    source="line",
                    mean=line.line,
                    median=line.line,
                    posted_line=line.line,
                )
            )

        known = _resolve_positions(self.positions, season, week)
        by_player = props_module.components_by_player(projections)
        # `components_by_player` keys on the book's NAME STRING, and two books --
        # or one book across two tabs -- spell the same player differently
        # ("Marvin Mims" / "Marvin Mims Jr."). Both resolve to the same ESPN id,
        # so merge them here into one line: two lines would give the market two
        # votes in the ensemble, and simply dropping one would throw away the
        # markets that only the other spelling carried.
        # The two spellings are merged as MARKETS and converted afterwards, not
        # converted and then merged. Doing it the other way round breaks the rule
        # two lines below: if "Marvin Mims" carries only an anytime-TD market and
        # "Marvin Mims Jr." carries a posted rushing-TD line, converting first
        # writes the anytime split into statId 25 for the first spelling and the
        # merge then refuses to replace it -- so the prior wins over the price,
        # and which spelling arrives first is dict order.
        merged: dict[int, dict[str, float]] = {}
        positions: dict[int, int] = {}
        names: dict[int, str] = {}
        unresolved: list[str] = []
        for player, components in by_player.items():
            player_id = self.id_index.from_name(
                player, position_hints=market_position_hints(components)
            )
            if player_id is None:
                unresolved.append(player)
                continue
            position_id = known.get(player_id)
            if position_id is not None:
                positions[player_id] = position_id
            target = merged.setdefault(player_id, {})
            for stat, value in components.items():
                target.setdefault(stat, float(value))
            names.setdefault(player_id, player)

        lines = []
        for player_id, components in merged.items():
            stats = self._to_espn_stats(components, positions.get(player_id))
            if not stats:
                continue
            lines.append(
                ComponentLine(
                    player_id=player_id,
                    season=season,
                    week=week,
                    source=self.name,
                    stats=stats,
                    games=None,
                )
            )
        return SourceLines(
            source=self.name,
            season=season,
            week=week,
            lines=tuple(lines),
            positions=positions,
            names=names,
            dense=False,
            unresolved=tuple(unresolved),
        )

    def _to_espn_stats(
        self, components: Mapping[str, float], position_id: int | None
    ) -> dict[str, float]:
        stats: dict[str, float] = {}
        anytime = components.get(props_module.ANYTIME_TDS)
        for stat, value in components.items():
            if stat in PROPS_ALREADY_SCORED or stat == props_module.ANYTIME_TDS:
                continue
            target = PROPS_TO_ESPN.get(stat)
            if target is None:
                continue
            stats[target] = float(value)
        if anytime is not None:
            share = self.rush_share.get(position_id or -1, DEFAULT_RUSH_SHARE)
            rush, rec = props_module.split_anytime_tds(float(anytime), share)
            # A posted rushing- or receiving-TD market is a real price; the split of
            # an anytime market is a prior. Never let the prior overwrite the price.
            stats.setdefault(RUSH_TD, rush)
            stats.setdefault(REC_TD, rec)
        return stats

    def component_lines(self, season: int, week: int) -> Sequence[ComponentLine]:
        return self.source_lines(season, week).lines


# --------------------------------------------------------------------------------------
# Establish The Run -- watched-folder CSV ingest
# --------------------------------------------------------------------------------------

ETR_SOURCE = "etr"
DEFAULT_ETR_DIR = Path("data/manual/etr")

#: `SourceLines.season`/`.week` when one file covers several. The lines inside it
#: still carry their own; only the slate label is undefined.
MIXED_SLATE = -1


class EtrSchemaError(ValueError):
    """An ETR CSV did not have the columns we know how to read.

    Raised on every ingest, not once at setup. ETR publishes no schema and
    guarantees nothing about their export; a renamed column is silent data loss
    if the loader shrugs and reads what it recognizes, so the message always
    carries the columns actually seen.
    """


def _norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


#: Canonical field -> the header spellings we accept, normalized to lowercase
#: alphanumerics (so "Pass Yds", "pass_yds" and "PASS-YDS" are one key).
#:
#: Extend this table when ETR moves; do not loosen the matcher. A fuzzy header
#: matcher is how "Rush Yds" silently becomes "Rec Yds" in a season where the
#: export changes and nobody notices for six weeks.
ETR_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "player": ("player", "playername", "name", "fullname"),
    "team": ("team", "tm", "nflteam", "teamabbrev"),
    "position": ("position", "pos"),
    "opponent": ("opponent", "opp"),
    "week": ("week", "wk"),
    "season": ("season", "year"),
    "pass_att": ("passatt", "passattempts", "passingattempts", "patt"),
    "pass_cmp": ("passcmp", "completions", "passingcompletions", "pcmp", "cmp"),
    "pass_yds": ("passyds", "passyards", "passingyards", "pyds"),
    "pass_td": ("passtd", "passtds", "passingtds", "passingtouchdowns", "ptd"),
    "pass_int": ("int", "ints", "interceptions", "passint", "passingints"),
    "rush_att": ("rushatt", "rushattempts", "carries", "rushingattempts", "ratt"),
    "rush_yds": ("rushyds", "rushyards", "rushingyards", "ryds"),
    "rush_td": ("rushtd", "rushtds", "rushingtds", "rushingtouchdowns", "rtd"),
    "targets": ("targets", "tgt", "tgts", "receivingtargets"),
    "receptions": ("rec", "receptions", "catches"),
    "rec_yds": ("recyds", "recyards", "receivingyards", "reyds"),
    "rec_td": ("rectd", "rectds", "receivingtds", "receivingtouchdowns", "retd"),
    "fumbles_lost": ("fl", "fum", "fumbles", "fumbleslost", "fumlost"),
    "games": ("g", "gp", "games", "gamesplayed"),
}

#: Canonical field -> ESPN statId. Fields absent from this map (player, team,
#: week, ...) are metadata, not stats.
ETR_STAT_IDS: Mapping[str, str] = {
    "pass_att": PASS_ATT,
    "pass_cmp": PASS_CMP,
    "pass_yds": PASS_YDS,
    "pass_td": PASS_TD,
    "pass_int": PASS_INT,
    "rush_att": RUSH_ATT,
    "rush_yds": RUSH_YDS,
    "rush_td": RUSH_TD,
    "targets": TARGETS,
    "receptions": RECEPTIONS,
    "rec_yds": REC_YDS,
    "rec_td": REC_TD,
    "fumbles_lost": FUMBLES_LOST,
}

#: Without a name there is nothing to join on, and without at least one stat there
#: is nothing to project. Everything else is optional.
ETR_REQUIRED_FIELDS: tuple[str, ...] = ("player",)
ETR_MIN_STAT_COLUMNS = 1

_ALIAS_LOOKUP: Mapping[str, str] = {
    alias: field_name for field_name, aliases in ETR_COLUMN_ALIASES.items() for alias in aliases
}


@dataclass(frozen=True, slots=True)
class EtrSchema:
    """The result of validating one CSV header."""

    path: Path | None
    columns: tuple[str, ...]
    #: canonical field -> the raw column it was found in
    fields: Mapping[str, str]
    #: raw columns we did not recognize. Not fatal -- ETR ships ranks, tiers and
    #: ownership alongside the projection -- but reported so drift is visible.
    unknown: tuple[str, ...]

    @property
    def stat_columns(self) -> dict[str, str]:
        """raw column -> ESPN statId."""
        return {
            column: ETR_STAT_IDS[field_name]
            for field_name, column in self.fields.items()
            if field_name in ETR_STAT_IDS
        }


def validate_etr_columns(columns: Sequence[str], *, path: Path | None = None) -> EtrSchema:
    """Map a CSV header onto our fields, or fail with what was actually there."""
    fields: dict[str, str] = {}
    unknown: list[str] = []
    for column in columns:
        field_name = _ALIAS_LOOKUP.get(_norm_header(column))
        if field_name is None:
            unknown.append(str(column))
        elif field_name not in fields:
            fields[field_name] = str(column)

    schema = EtrSchema(
        path=path,
        columns=tuple(str(c) for c in columns),
        fields=fields,
        unknown=tuple(unknown),
    )
    missing = [f for f in ETR_REQUIRED_FIELDS if f not in fields]
    stat_columns = schema.stat_columns
    if missing or len(stat_columns) < ETR_MIN_STAT_COLUMNS:
        where = f" in {path}" if path else ""
        problem = (
            f"missing required column(s) {missing}"
            if missing
            else f"found {len(stat_columns)} stat column(s), need at least {ETR_MIN_STAT_COLUMNS}"
        )
        raise EtrSchemaError(
            f"ETR CSV schema drift{where}: {problem}.\n"
            f"  columns seen ({len(schema.columns)}): {list(schema.columns)}\n"
            f"  recognized: {dict(fields)}\n"
            f"  unrecognized: {list(schema.unknown)}\n"
            "ETR publishes no schema and does not version their export. Update "
            "ETR_COLUMN_ALIASES in projections/sources.py to teach it the new "
            "spelling -- do not relax the matcher."
        )
    return schema


_WEEK_IN_NAME = re.compile(r"(?:^|[^a-z0-9])(?:wk|week)[^0-9]?(\d{1,2})", re.IGNORECASE)
_SEASON_IN_NAME = re.compile(r"(?:^|[^0-9])(20\d{2})(?:[^0-9]|$)")


def week_from_filename(path: Path) -> int | None:
    """`etr_2026_wk03.csv` -> 3. None when the name says nothing."""
    match = _WEEK_IN_NAME.search(path.stem)
    return int(match.group(1)) if match else None


def season_from_filename(path: Path) -> int | None:
    match = _SEASON_IN_NAME.search(path.stem)
    return int(match.group(1)) if match else None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "N/A", "-", "--", "NULL", "NONE"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_etr_csv(
    path: Path,
    *,
    season: int | None = None,
    week: int | None = None,
    id_index: EspnIdIndex | None = None,
) -> SourceLines:
    """One ETR CSV -> component lines, validating the header first.

    `season`/`week` come from the caller, then from a column, then from the
    filename. A file that cannot say which week it is for is rejected: an ETR
    export silently filed under the wrong week is worse than a missing one,
    because it will quietly out-vote two live sources.
    """
    path = Path(path)
    try:
        # Every column as Utf8: an export whose missing marker is "NA" or a blank
        # gets a mix of nulls and literals under type inference, depending on which
        # rows the sampler happened to see.
        frame = pl.read_csv(path, infer_schema_length=0)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise EtrSchemaError(f"could not read ETR CSV {path}: {exc}") from exc

    schema = validate_etr_columns(frame.columns, path=path)
    stat_columns = schema.stat_columns
    index = id_index or EspnIdIndex()

    file_week = week if week is not None else week_from_filename(path)
    file_season = season if season is not None else season_from_filename(path)

    lines: list[ComponentLine] = []
    positions: dict[int, int] = {}
    names: dict[int, str] = {}
    unresolved: list[str] = []
    weeks_seen: set[int] = set()

    for row in frame.iter_rows(named=True):
        name = str(row.get(schema.fields.get("player", ""), "") or "").strip()
        if not name:
            continue
        row_week = _as_float(row.get(schema.fields["week"])) if "week" in schema.fields else None
        row_season = (
            _as_float(row.get(schema.fields["season"])) if "season" in schema.fields else None
        )
        this_week = int(row_week) if row_week is not None else file_week
        this_season = int(row_season) if row_season is not None else file_season
        if this_week is None or this_season is None:
            raise EtrSchemaError(
                f"{path}: cannot tell which season/week this file covers. It has no "
                f"season/week column, and the filename does not carry them "
                f"(expected something like 'etr_2026_wk03.csv'). Columns seen: "
                f"{list(schema.columns)}"
            )
        weeks_seen.add(this_week)
        if week is not None and this_week != week:
            continue
        if season is not None and this_season != season:
            continue

        team = row.get(schema.fields.get("team", ""))
        position = row.get(schema.fields.get("position", ""))
        player_id = index.from_name(name, team, position)
        if player_id is None:
            unresolved.append(name)
            continue

        # Dense within the columns this file actually has: a blank cell in a
        # present column is a projected zero. A column the file does not have at
        # all is an abstention, and simply never reaches `stats`.
        stats = {
            stat_id: (_as_float(row.get(column)) or 0.0) for column, stat_id in stat_columns.items()
        }
        games = None
        if "games" in schema.fields:
            games = _as_float(row.get(schema.fields["games"]))
        lines.append(
            ComponentLine(
                player_id=player_id,
                season=this_season,
                week=this_week,
                source=ETR_SOURCE,
                stats=stats,
                games=games,
            )
        )
        names[player_id] = name
        position_id = _etr_position_id(position)
        if position_id is not None:
            positions[player_id] = position_id

    # The lines always carry their own true season and week. These two fields
    # describe the *slate*, so they are MIXED_SLATE when one file spans several
    # weeks (a season-long export) rather than a plausible-looking single week.
    resolved_week = (
        week if week is not None else (weeks_seen.pop() if len(weeks_seen) == 1 else MIXED_SLATE)
    )
    seasons_seen = {line.season for line in lines}
    resolved_season = (
        season
        if season is not None
        else (seasons_seen.pop() if len(seasons_seen) == 1 else (file_season or MIXED_SLATE))
    )
    return SourceLines(
        source=ETR_SOURCE,
        season=resolved_season,
        week=resolved_week,
        lines=tuple(lines),
        positions=positions,
        names=names,
        dense=True,
        unresolved=tuple(unresolved),
    )


def _etr_position_id(value: object) -> int | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    return SLEEPER_POSITION_IDS.get(text) or {"DST": 16, "D/ST": 16, "PK": 5}.get(text)


class EtrCsvSource:
    """Watched-folder ingest of Establish The Run projections.

    ETR has no API and their terms of service prohibit automated collection, so a
    hand-downloaded CSV dropped into `data/manual/etr/` is the only path and this
    class deliberately has no fetch method. There is nothing to build later; do
    not add one.

    Every file in the folder is re-validated on every ingest, because the export's
    schema is undocumented and unversioned. A drifted file raises rather than
    contributing a partial line -- a projection that silently lost its receiving
    yards column still looks like a projection.
    """

    name = ETR_SOURCE

    def __init__(
        self,
        directory: Path = DEFAULT_ETR_DIR,
        *,
        id_index: EspnIdIndex | None = None,
        strict: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.id_index = id_index or EspnIdIndex()
        #: When False, a drifted file is logged and skipped instead of raising.
        #: Only for a batch backfill over a folder of historical exports.
        self.strict = strict

    def files(self) -> list[Path]:
        return sorted(self.directory.glob("*.csv"))

    def candidate_files(self, season: int, week: int) -> list[Path]:
        """Files whose name does not rule them out for this season/week.

        A file whose name carries a different week is skipped without being read.
        A file whose name says nothing is read, because the week may be in a
        column.
        """
        out = []
        for path in self.files():
            named_week = week_from_filename(path)
            named_season = season_from_filename(path)
            if named_week is not None and named_week != week:
                continue
            if named_season is not None and named_season != season:
                continue
            out.append(path)
        return out

    def source_lines(self, season: int, week: int) -> SourceLines:
        if not self.directory.exists():
            raise SourceUnavailable(
                f"no ETR folder at {self.directory}; ETR is manual-CSV only "
                "(no API, and their ToS forbids scraping)"
            )
        candidates = self.candidate_files(season, week)
        if not candidates:
            raise SourceUnavailable(
                f"no ETR CSV for {season} week {week} in {self.directory} "
                f"(files present: {[p.name for p in self.files()]})"
            )

        lines: list[ComponentLine] = []
        positions: dict[int, int] = {}
        names: dict[int, str] = {}
        unresolved: list[str] = []
        for path in candidates:
            try:
                part = read_etr_csv(path, season=season, week=week, id_index=self.id_index)
            except EtrSchemaError:
                if self.strict:
                    raise
                log.warning("skipping drifted ETR file %s", path, exc_info=True)
                continue
            lines.extend(part.lines)
            positions.update(part.positions)
            names.update(part.names)
            unresolved.extend(part.unresolved)

        # Two exports for the same week (a re-download, say) would double-weight
        # ETR against every other source. Last file wins, by player.
        deduped: dict[int, ComponentLine] = {line.player_id: line for line in lines}
        return SourceLines(
            source=self.name,
            season=season,
            week=week,
            lines=tuple(deduped.values()),
            positions=positions,
            names=names,
            dense=True,
            unresolved=tuple(unresolved),
        )

    def component_lines(self, season: int, week: int) -> Sequence[ComponentLine]:
        return self.source_lines(season, week).lines


# --------------------------------------------------------------------------------------
# The standard set
# --------------------------------------------------------------------------------------


def default_adapters(
    *,
    id_index: EspnIdIndex | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    etr_dir: Path = DEFAULT_ETR_DIR,
    include_props: bool = True,
) -> list[ProjectionAdapter]:
    """The sources we actually run, in no particular order -- weights are equal.

    A new source is a small class with `name` and `source_lines`; adding it here
    is the whole integration. Nothing downstream enumerates sources by name.
    """
    index = id_index if id_index is not None else default_id_index()
    espn = EspnSnapshotSource(root=snapshot_root)
    adapters: list[ProjectionAdapter] = [
        espn,
        SleeperSource(id_index=index),
        EtrCsvSource(etr_dir, id_index=index),
    ]
    if include_props:
        # ESPN is the only source that reliably knows a player's position, and the
        # anytime-TD split needs one.
        adapters.append(PropsSource(id_index=index, positions=espn.positions))
    return adapters


def iter_lines(bundle: SourceBundle) -> Iterator[ComponentLine]:
    for source in bundle.sources:
        yield from source.lines
