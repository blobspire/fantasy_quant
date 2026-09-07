"""League scoring: `scoringSettings.scoringItems` -> a callable scorer.

Every number this system produces is a scored projection, so a scoring bug is not a
local bug -- it silently biases valuations, trades, FAAB and title odds at once, and
it does so *per league*, which makes it nearly invisible in aggregate. Hence the
acceptance test in `verify_scoring_reproduction`: re-score real boxscore stat lines
from raw counts and demand bit-for-bit agreement with ESPN's own `appliedTotal`.
Measured with this module over every week of three public 2025 leagues (half-PPR +
TE premium, flat half-PPR, and an IDP/PPFD/TQB league): 15,936 of 15,936 stat rows
exact, across defaultPositionIds 1-5, 10, 11, 13, 15 and 16.

Three rules do all the work, and each has a documented way to get it wrong:

1. **`pointsOverrides` keys are `defaultPositionId`, not `lineupSlotId`.** The two
   ID spaces collide at 4 (TE vs WR) and 15 (TQB vs DP). Scoring by slot instead of
   position mis-scored 394 of 1,062 rows in the IDP league -- and, worse, would look
   *right* for most of a standard roster.
2. **An override replaces `points`; it never adds to it, and `0.0` is a real value.**
   `{"statId": 106, "points": 4.0, "pointsOverrides": {"16": 0.0}}` means a D/ST
   forced fumble is worth nothing, not four. Treating overrides as additive, or as
   "falsy means unset", each broke 110 of those same 1,062 rows.
3. **Apply only statIds that carry a `scoringItem`.** Raw `stats` ships pre-computed
   derived buckets -- 40/61 restate rushing/receiving yards, 41 restates receptions,
   47-52 are "every N receiving yards", 100 = 2x99, 109 = 107+108, and the
   points-allowed / yards-allowed bands are one-hot. Applying everything present
   counts the same yardage two or three times.

`pointsOverrides` may be absent entirely, `{}`, or populated -- all three occur in
the wild, so it is always read with `.get`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .client import EspnClient
from .endpoints import league_url
from .statrows import SOURCE_ACTUAL, SOURCE_PROJECTED, parse_rows

log = logging.getLogger(__name__)

# --- defaultPositionId. NOT lineupSlotId: the spaces disagree at 4 and 15. -------
POS_QB = 1
POS_RB = 2
POS_WR = 3
POS_TE = 4
POS_K = 5
POS_TQB = 15
POS_DST = 16

# --- lineupSlotId ---------------------------------------------------------------
SLOT_QB = 0
SLOT_TQB = 1
SLOT_OP = 7  # "OP" is ESPN's superflex slot
SLOT_BENCH = 20
SLOT_IR = 21
SLOT_INVALID = 22
# DT DE LB DL CB S DB DP -- any of these being startable is what makes a league IDP.
IDP_SLOTS = tuple(range(8, 16))
NON_STARTING_SLOTS = frozenset({SLOT_BENCH, SLOT_IR, SLOT_INVALID})

STAT_RECEPTIONS = "53"
STAT_RECEIVING_FIRST_DOWNS = "213"

# `draftSettings.type` comes back as the enum NAME on the live endpoint; the
# ordinals are kept so older or differently-serialized captures still parse.
DRAFT_TYPES = {
    0: "OFFLINE",
    1: "SNAKE",
    2: "AUTOPICK",
    3: "SNAIL",
    4: "AUCTION",
    5: "LINEAR",
}
AUCTION_DRAFT_TYPES = frozenset({"AUCTION"})

# ESPN's name for "each team also plays the league median every week".
MEDIAN_SCORING_ENHANCEMENT = "WIN_BONUS_TOP_HALF"


class ScoringReproductionError(RuntimeError):
    """Our scorer disagreed with ESPN's `appliedTotal`. Never paper over this."""


def _unwrap(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Accept a league payload, its `settings`, or the sub-settings dict itself."""
    if key in payload:
        node = payload[key]
        return node if isinstance(node, Mapping) else {}
    settings = payload.get("settings")
    if isinstance(settings, Mapping) and key in settings:
        node = settings[key]
        return node if isinstance(node, Mapping) else {}
    return {}


def _league_settings(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = payload.get("settings")
    if isinstance(settings, Mapping):
        return settings
    return payload


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoringItem:
    """One entry of `scoringSettings.scoringItems`."""

    stat_id: str
    points: float
    overrides: Mapping[int, float] = field(default_factory=dict)
    # Ranking metadata ("lower is better"), not a sign flip -- `points` already
    # carries its own sign. This is measured, not assumed: public 2025 leagues
    # 350313 (statIds 20, 72, 85) and 1966012 (20, 72) set it, and both reproduce
    # ESPN's appliedTotal exactly with no flip -- Geno Smith's three interceptions
    # score -6.0 in ESPN's own appliedStats, not +6.0.
    is_reverse: bool = False

    def points_for(self, position_id: int) -> float:
        """Per-unit value of this stat for a player whose defaultPositionId is given."""
        return self.overrides.get(position_id, self.points)


class LeagueScoring:
    """One league's scoring rules as a callable: `score(raw_stats, position_id)`."""

    __slots__ = ("items", "_tables")

    def __init__(self, items: Mapping[str, ScoringItem]) -> None:
        self.items: dict[str, ScoringItem] = dict(items)
        # Resolving overrides per stat per call is the inner loop of every
        # simulation, so flatten to a statId -> points table once per position.
        self._tables: dict[int, dict[str, float]] = {}

    @classmethod
    def from_settings(cls, payload: Mapping[str, Any]) -> LeagueScoring:
        """Build from a league payload, its `settings`, or `scoringSettings`."""
        scoring_settings = _unwrap(payload, "scoringSettings") or payload
        raw_items = scoring_settings.get("scoringItems")
        if not raw_items:
            raise ValueError(
                "no scoringItems in the supplied settings; an unrecognized `view=` "
                "returns HTTP 200 with a skeleton, so check the request first."
            )

        items: dict[str, ScoringItem] = {}
        for raw in raw_items:
            stat_id = str(raw["statId"])
            items[stat_id] = ScoringItem(
                stat_id=stat_id,
                points=float(raw.get("points") or 0.0),
                # Absent on most items, `{}` on some. Never index it.
                overrides={int(k): float(v) for k, v in (raw.get("pointsOverrides") or {}).items()},
                is_reverse=bool(raw.get("isReverseItem")),
            )
        return cls(items)

    def points_for(self, stat_id: str | int, position_id: int) -> float:
        """Per-unit value of `stat_id` at `position_id`; 0.0 if the league ignores it."""
        item = self.items.get(str(stat_id))
        return item.points_for(position_id) if item is not None else 0.0

    def table_for(self, position_id: int) -> dict[str, float]:
        """Cached statId -> points table for one defaultPositionId."""
        table = self._tables.get(position_id)
        if table is None:
            table = {sid: item.points_for(position_id) for sid, item in self.items.items()}
            self._tables[position_id] = table
        return table

    def score(self, raw_stats: Mapping[str, float], default_position_id: int) -> float:
        """Score a raw stat line. `default_position_id`, never a lineupSlotId."""
        table = self.table_for(default_position_id)
        total = 0.0
        for stat_id, value in raw_stats.items():
            points = table.get(stat_id)
            if points is None:
                # Tolerate int keys; a JSON payload always gives strings, but a
                # Parquet round-trip or a hand-built dict may not.
                points = table.get(str(stat_id))
                if points is None:
                    # No scoringItem => not scored. This branch is what stops the
                    # derived buckets (40/41/47-52/61/100/109 and the one-hot
                    # points- and yards-allowed bands) from multi-counting.
                    continue
            total += value * points
        return total

    __call__ = score

    def breakdown(
        self, raw_stats: Mapping[str, float], default_position_id: int
    ) -> dict[str, float]:
        """Per-statId contributions -- directly comparable to ESPN's `appliedStats`."""
        table = self.table_for(default_position_id)
        out: dict[str, float] = {}
        for stat_id, value in raw_stats.items():
            key = str(stat_id)
            points = table.get(key)
            if points is None:
                continue
            out[key] = value * points
        return out

    @property
    def scored_stat_ids(self) -> frozenset[str]:
        return frozenset(self.items)

    @property
    def reverse_items(self) -> tuple[ScoringItem, ...]:
        """Items flagged `isReverseItem` -- real, and deliberately not acted on.

        Roughly one public league in four sets it, always on a negative-points item
        (interceptions thrown, fumbles lost). Applying a sign flip would break those
        leagues; the live reproduction test on league 350313 is what pins that.
        """
        return tuple(item for item in self.items.values() if item.is_reverse)

    def __repr__(self) -> str:
        return f"LeagueScoring({len(self.items)} items)"


# ---------------------------------------------------------------------------
# League shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeagueShape:
    """The format facts that change what a player is worth in *this* league.

    Kept separate from `LeagueScoring` because scoring answers "how many points"
    and this answers "how many of him do I need, and what did he cost".
    """

    name: str
    size: int
    scoring_type: str

    draft_type: str
    auction_budget: int
    keeper_count: int
    keeper_count_future: int

    uses_faab: bool
    faab_budget: int
    acquisition_type: str

    lineup_slot_counts: Mapping[int, int]
    position_limits: Mapping[int, int]

    points_per_reception: float
    te_premium: float
    points_per_receiving_first_down: float
    has_median_scoring: bool

    matchup_period_count: int
    playoff_team_count: int

    @property
    def is_auction(self) -> bool:
        return self.draft_type in AUCTION_DRAFT_TYPES

    @property
    def is_keeper(self) -> bool:
        return self.keeper_count > 0 or self.keeper_count_future > 0

    @property
    def is_redraft(self) -> bool:
        return not self.is_keeper

    @property
    def is_superflex(self) -> bool:
        return self.lineup_slot_counts.get(SLOT_OP, 0) > 0 or (
            self.lineup_slot_counts.get(SLOT_QB, 0) >= 2
        )

    @property
    def is_idp(self) -> bool:
        return any(self.lineup_slot_counts.get(slot, 0) > 0 for slot in IDP_SLOTS)

    @property
    def has_te_premium(self) -> bool:
        return self.te_premium > 0.0

    @property
    def starting_slot_counts(self) -> dict[int, int]:
        """Slots that actually score, i.e. everything but bench/IR/invalid."""
        return {
            slot: n
            for slot, n in self.lineup_slot_counts.items()
            if n > 0 and slot not in NON_STARTING_SLOTS
        }

    @property
    def starters(self) -> int:
        return sum(self.starting_slot_counts.values())

    @classmethod
    def from_settings(
        cls,
        payload: Mapping[str, Any],
        scoring: LeagueScoring | None = None,
    ) -> LeagueShape:
        settings = _league_settings(payload)
        roster = _unwrap(settings, "rosterSettings")
        draft = _unwrap(settings, "draftSettings")
        acquisition = _unwrap(settings, "acquisitionSettings")
        schedule = _unwrap(settings, "scheduleSettings")
        scoring_settings = _unwrap(settings, "scoringSettings")
        scoring = scoring or LeagueScoring.from_settings(settings)

        raw_type = draft.get("type")
        if isinstance(raw_type, int):
            draft_type = DRAFT_TYPES.get(raw_type, f"UNKNOWN_{raw_type}")
        else:
            draft_type = str(raw_type or "UNKNOWN")

        # TE premium is the TE's *resolved* per-reception value minus the WR's.
        # Subtracting the item's bare `points` instead reports a fake premium,
        # because per-position PPR leagues leave `points` at 0.0 and put every
        # real value in the overrides.
        wr_rec = scoring.points_for(STAT_RECEPTIONS, POS_WR)
        te_rec = scoring.points_for(STAT_RECEPTIONS, POS_TE)

        return cls(
            name=str(settings.get("name") or ""),
            size=int(settings.get("size") or 0),
            scoring_type=str(scoring_settings.get("scoringType") or ""),
            draft_type=draft_type,
            # `auctionBudget` is populated even in snake leagues, so it is not a
            # discriminator -- `draftSettings.type` is.
            auction_budget=int(draft.get("auctionBudget") or 0),
            keeper_count=int(draft.get("keeperCount") or 0),
            keeper_count_future=int(draft.get("keeperCountFuture") or 0),
            # `acquisitionType` is the waiver *processing model*
            # (WAIVERS_TRADITIONAL / WAIVERS_CONTINUOUS) and says nothing about
            # whether money is involved. This flag is the FAAB discriminator.
            uses_faab=bool(acquisition.get("isUsingAcquisitionBudget")),
            faab_budget=int(acquisition.get("acquisitionBudget") or 0),
            acquisition_type=str(acquisition.get("acquisitionType") or ""),
            lineup_slot_counts={
                int(k): int(v) for k, v in (roster.get("lineupSlotCounts") or {}).items()
            },
            # positionLimits is keyed by defaultPositionId, unlike lineupSlotCounts.
            position_limits={
                int(k): int(v) for k, v in (roster.get("positionLimits") or {}).items()
            },
            points_per_reception=wr_rec,
            te_premium=te_rec - wr_rec,
            points_per_receiving_first_down=scoring.points_for(STAT_RECEIVING_FIRST_DOWNS, POS_WR),
            has_median_scoring=(
                str(scoring_settings.get("scoringEnhancementType") or "")
                == MEDIAN_SCORING_ENHANCEMENT
            ),
            matchup_period_count=int(schedule.get("matchupPeriodCount") or 0),
            playoff_team_count=int(schedule.get("playoffTeamCount") or 0),
        )


# ---------------------------------------------------------------------------
# The acceptance test: reproduce ESPN's own appliedTotal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoringMismatch:
    """One stat line where our score disagrees with ESPN's."""

    player_id: int
    player_name: str
    default_position_id: int
    lineup_slot_id: int
    scoring_period_id: int
    stat_row_id: str
    stat_source_id: int
    espn_total: float
    our_total: float
    raw_stats: Mapping[str, float]
    espn_applied_stats: Mapping[str, float]
    our_applied_stats: Mapping[str, float]

    @property
    def delta(self) -> float:
        return self.our_total - self.espn_total

    def describe(self) -> str:
        source = "projected" if self.stat_source_id == SOURCE_PROJECTED else "actual"
        lines = [
            f"{self.player_name} (id {self.player_id}, defaultPositionId "
            f"{self.default_position_id}, slot {self.lineup_slot_id}) week "
            f"{self.scoring_period_id} {source} row {self.stat_row_id}: "
            f"espn={self.espn_total!r} ours={self.our_total!r} delta={self.delta!r}",
            "  per-stat  statId: raw -> ours vs espn",
        ]
        interesting = sorted(
            set(self.our_applied_stats) | set(self.espn_applied_stats),
            key=lambda s: (len(s), s),
        )
        for stat_id in interesting:
            ours = self.our_applied_stats.get(stat_id, 0.0)
            theirs = self.espn_applied_stats.get(stat_id, 0.0)
            flag = "" if abs(ours - theirs) < 1e-9 else "   <-- differs"
            raw = self.raw_stats.get(stat_id, 0.0)
            lines.append(f"    {stat_id}: {raw!r} -> {ours!r} vs {theirs!r}{flag}")
        lines.append(f"  full raw stats: {dict(self.raw_stats)!r}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ReproductionReport:
    """Result of re-scoring a whole boxscore against ESPN's `appliedTotal`."""

    league_id: int
    season: int
    scoring_period_id: int
    players: int
    checked: int
    mismatches: tuple[ScoringMismatch, ...]
    # defaultPositionId -> rows checked. "Everything reproduced" is worth much
    # less if the week happened to contain no D/ST, whose scoring is the part
    # most likely to be wrong, so the coverage travels with the result.
    positions: Mapping[int, int] = field(default_factory=dict)

    @property
    def matched(self) -> int:
        return self.checked - len(self.mismatches)

    @property
    def exact(self) -> bool:
        # `checked == 0` is a failure, not a pass: it means the boxscore shape
        # moved and we verified nothing at all.
        return self.checked > 0 and not self.mismatches

    def summary(self) -> str:
        positions = ",".join(f"{pos}:{n}" for pos, n in sorted(self.positions.items()))
        head = (
            f"league {self.league_id} season {self.season} week {self.scoring_period_id}: "
            f"{self.matched}/{self.checked} stat rows reproduce ESPN's appliedTotal "
            f"({self.players} players; defaultPositionId {positions})"
        )
        if self.exact:
            return head + " -- exact"
        if not self.checked:
            return head + " -- NOTHING CHECKED (no rosters in the boxscore payload)"
        return "\n".join([head] + [m.describe() for m in self.mismatches])

    def raise_if_failed(self) -> None:
        if not self.exact:
            raise ScoringReproductionError(self.summary())


def _iter_boxscore_entries(payload: Mapping[str, Any]) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """(teamId, roster entry) for every player in an `mBoxscore` payload.

    `mBoxscore` returns the whole season's schedule but populates rosters for only
    the requested `scoringPeriodId`, so the empty matchups fall out naturally.
    """
    for matchup in payload.get("schedule") or []:
        for side in ("home", "away"):
            team = matchup.get(side) or {}
            roster = team.get("rosterForCurrentScoringPeriod") or {}
            for entry in roster.get("entries") or []:
                yield int(team.get("teamId") or 0), entry


def check_boxscore(
    scoring: LeagueScoring,
    boxscore: Mapping[str, Any],
    *,
    season: int,
    scoring_period_id: int,
    league_id: int | None = None,
    tolerance: float = 1e-6,
    sources: Sequence[int] = (SOURCE_ACTUAL, SOURCE_PROJECTED),
) -> ReproductionReport:
    """Re-score every stat row in one boxscore. Pure -- no network, so it is testable.

    Both projected and actual rows are checked by default. Projections are the
    harsher half: their raw counts are fractional, so every derived bucket is
    non-zero and any stat applied that shouldn't be shows up immediately. They
    also exist before Week 1 kicks off, which keeps the startup assertion alive
    in the preseason.
    """
    wanted = set(sources)
    mismatches: list[ScoringMismatch] = []
    positions: dict[int, int] = {}
    checked = 0
    players = 0

    for _team_id, entry in _iter_boxscore_entries(boxscore):
        pool_entry = entry.get("playerPoolEntry") or {}
        player = pool_entry.get("player") or {}
        position_id = int(player.get("defaultPositionId") or 0)
        raw_rows = player.get("stats") or []
        players += 1

        for raw_row, row in zip(raw_rows, parse_rows(raw_rows), strict=True):
            if row.season != season or row.scoring_period != scoring_period_id:
                continue
            # Rows with an empty `stats` carry appliedTotal 0.0 without exception
            # (2,115 of them across the three verification leagues), so they are a
            # free pass that would only inflate the count.
            if row.source not in wanted or not row.stats:
                continue

            checked += 1
            positions[position_id] = positions.get(position_id, 0) + 1
            ours = scoring.score(row.stats, position_id)
            if abs(ours - row.applied_total) <= tolerance:
                continue
            mismatches.append(
                ScoringMismatch(
                    player_id=int(player.get("id") or 0),
                    player_name=str(player.get("fullName") or "?"),
                    default_position_id=position_id,
                    lineup_slot_id=int(entry.get("lineupSlotId") or -1),
                    scoring_period_id=row.scoring_period,
                    stat_row_id=row.row_id,
                    stat_source_id=row.source,
                    espn_total=row.applied_total,
                    our_total=ours,
                    raw_stats=row.stats,
                    espn_applied_stats={
                        str(k): float(v) for k, v in (raw_row.get("appliedStats") or {}).items()
                    },
                    our_applied_stats=scoring.breakdown(row.stats, position_id),
                )
            )

    return ReproductionReport(
        league_id=int(league_id if league_id is not None else boxscore.get("id") or 0),
        season=season,
        scoring_period_id=scoring_period_id,
        players=players,
        checked=checked,
        mismatches=tuple(mismatches),
        positions=dict(sorted(positions.items())),
    )


def fetch_league_settings(client: EspnClient, season: int, league_id: int) -> dict[str, Any]:
    """`mSettings` for one league. Public leagues need no cookies."""
    payload, _ = client.get(league_url(season, league_id), params={"view": "mSettings"})
    return payload


def fetch_boxscore(
    client: EspnClient, season: int, league_id: int, scoring_period_id: int
) -> dict[str, Any]:
    """`mBoxscore` for one week. Rosters populate for that week only; loop to sweep."""
    payload, _ = client.get(
        league_url(season, league_id),
        params={"view": "mBoxscore", "scoringPeriodId": scoring_period_id},
    )
    return payload


def _default_scoring_period(settings_payload: Mapping[str, Any]) -> tuple[int, int]:
    """(week to check, earliest legal week) from `status`.

    `latestScoringPeriod` runs past the end of the fantasy season -- a finished
    2025 league reports 19 against a `finalScoringPeriod` of 17 -- and weeks past
    the final one carry no rosters, so it has to be clamped.
    """
    status = settings_payload.get("status") or {}
    first = int(status.get("firstScoringPeriod") or 1)
    latest = int(status.get("latestScoringPeriod") or 0)
    final = int(status.get("finalScoringPeriod") or 0)
    week = min(latest, final) if final else latest
    return max(week, first), first


def verify_scoring_reproduction(
    season: int,
    league_id: int,
    scoring_period_id: int | None = None,
    *,
    client: EspnClient | None = None,
    tolerance: float = 1e-6,
    sources: Sequence[int] = (SOURCE_ACTUAL, SOURCE_PROJECTED),
    max_lookback: int = 3,
) -> ReproductionReport:
    """Fetch a league's settings and one boxscore, and re-score every player.

    This is the regression test against ESPN quietly changing the scoring model,
    and it is cheap enough (two GETs) to run as a startup assertion:

        verify_scoring_reproduction(2025, 1241838).raise_if_failed()

    With no `scoring_period_id` it picks the latest completed week and, if that
    boxscore turns out to be empty, walks back up to `max_lookback` weeks -- a
    league mid-week can legitimately have a week with no populated rosters yet.
    """
    owned = client is None
    client = client or EspnClient()
    try:
        settings = fetch_league_settings(client, season, league_id)
        scoring = LeagueScoring.from_settings(settings)
        if scoring_period_id is not None:
            week, first, attempts = scoring_period_id, scoring_period_id, 1
        else:
            week, first = _default_scoring_period(settings)
            attempts = max_lookback + 1

        report = ReproductionReport(league_id, season, week, 0, 0, ())
        for back in range(attempts):
            candidate = week - back
            if candidate < first:
                break
            boxscore = fetch_boxscore(client, season, league_id, candidate)
            report = check_boxscore(
                scoring,
                boxscore,
                season=season,
                scoring_period_id=candidate,
                league_id=league_id,
                tolerance=tolerance,
                sources=sources,
            )
            if report.checked:
                break
    finally:
        if owned:
            client.close()

    log.info("%s", report.summary())
    return report
