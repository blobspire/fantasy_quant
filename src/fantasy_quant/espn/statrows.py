"""Parsing for ESPN player `stats[]` rows.

A stat row's `id` encodes its own identity: ``{statSourceId}{statSplitTypeId}{externalId}``.

    002026        actual, season total
    102026        projected, season total  <- FROZEN at preseason, see below
    1120261       projected, 2026 week 1
    01401772835   actual, one game; suffix is the NFL event id
    122026        projected, rest-of-season expressed as a per-game RATE

Two traps this module exists to prevent:

1. **Mixed seasons.** A 2026 request returns 2025 rows in the same array. Always
   filter on `seasonId`; never trust array position.
2. **The frozen season total.** ESPN fixes the season projection at preseason and
   never revises it, while revising the weeklies all year. The widely-repeated
   "ESPN is ~11% optimistic" finding is really this staleness. Rest-of-season must
   be built by summing remaining weekly rows -- see `rest_of_season_projection`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# statSourceId
SOURCE_ACTUAL = 0
SOURCE_PROJECTED = 1

# statSplitTypeId
SPLIT_SEASON = 0
SPLIT_GAME = 1
SPLIT_REST_OF_SEASON = 2


@dataclass(frozen=True, slots=True)
class StatRow:
    """One entry from a player's `stats[]` array."""

    row_id: str
    season: int
    source: int
    split: int
    scoring_period: int
    applied_total: float
    pro_team_id: int
    stats: dict[str, float]

    @property
    def is_projection(self) -> bool:
        return self.source == SOURCE_PROJECTED

    @property
    def is_weekly(self) -> bool:
        return self.split == SPLIT_GAME


def parse_rows(raw: Iterable[dict]) -> list[StatRow]:
    out: list[StatRow] = []
    for r in raw:
        out.append(
            StatRow(
                row_id=str(r.get("id", "")),
                season=int(r.get("seasonId") or 0),
                source=int(r.get("statSourceId") or 0),
                split=int(r.get("statSplitTypeId") or 0),
                scoring_period=int(r.get("scoringPeriodId") or 0),
                applied_total=float(r.get("appliedTotal") or 0.0),
                pro_team_id=int(r.get("proTeamId") or 0),
                stats={str(k): float(v) for k, v in (r.get("stats") or {}).items()},
            )
        )
    return out


def weekly_projections(rows: Sequence[StatRow], season: int) -> dict[int, float]:
    """Week -> projected points, for one season only."""
    return {
        r.scoring_period: r.applied_total
        for r in rows
        if r.season == season and r.is_projection and r.is_weekly and r.scoring_period > 0
    }


def weekly_actuals(rows: Sequence[StatRow], season: int) -> dict[int, float]:
    """Week -> actual points, for one season only.

    Actual weekly rows carry the NFL event id in `id`, so `scoringPeriodId` is the
    only reliable week key.
    """
    return {
        r.scoring_period: r.applied_total
        for r in rows
        if r.season == season and r.source == SOURCE_ACTUAL and r.is_weekly and r.scoring_period > 0
    }


def frozen_season_projection(rows: Sequence[StatRow], season: int) -> float | None:
    """ESPN's season-total projection *as published* -- stale after preseason.

    Exposed only so we can measure the staleness. Do not use it as a forecast;
    `espn-api`'s `projected_total_points` reads this field, which is why most
    tools inherit the error.
    """
    for r in rows:
        if r.season == season and r.is_projection and r.split == SPLIT_SEASON:
            return r.applied_total
    return None


def rest_of_season_projection(rows: Sequence[StatRow], season: int, from_week: int) -> float:
    """Sum the remaining weekly projections. This is the number to actually use.

    `from_week` is inclusive, so passing the current scoring period includes the
    week now in progress.
    """
    weekly = weekly_projections(rows, season)
    return sum(pts for wk, pts in weekly.items() if wk >= from_week)
