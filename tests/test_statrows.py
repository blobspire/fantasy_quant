"""Stat-row parsing, including the two traps that silently corrupt downstream math."""

from __future__ import annotations

from fantasy_quant.espn.statrows import (
    frozen_season_projection,
    parse_rows,
    rest_of_season_projection,
    weekly_actuals,
    weekly_projections,
)

# Shape mirrors a live ESPN payload: two seasons interleaved in one array.
RAW = [
    {
        "id": "102026",
        "seasonId": 2026,
        "statSourceId": 1,
        "statSplitTypeId": 0,
        "scoringPeriodId": 0,
        "appliedTotal": 300.0,
        "stats": {},
    },
    {
        "id": "1120261",
        "seasonId": 2026,
        "statSourceId": 1,
        "statSplitTypeId": 1,
        "scoringPeriodId": 1,
        "appliedTotal": 18.0,
        "stats": {"53": 4.0},
    },
    {
        "id": "1120262",
        "seasonId": 2026,
        "statSourceId": 1,
        "statSplitTypeId": 1,
        "scoringPeriodId": 2,
        "appliedTotal": 20.0,
        "stats": {},
    },
    {
        "id": "1120263",
        "seasonId": 2026,
        "statSourceId": 1,
        "statSplitTypeId": 1,
        "scoringPeriodId": 3,
        "appliedTotal": 22.0,
        "stats": {},
    },
    # 2025 rows arrive in the same array and must never leak into 2026 aggregates.
    {
        "id": "002025",
        "seasonId": 2025,
        "statSourceId": 0,
        "statSplitTypeId": 0,
        "scoringPeriodId": 0,
        "appliedTotal": 250.0,
        "stats": {},
    },
    {
        "id": "01401772835",
        "seasonId": 2025,
        "statSourceId": 0,
        "statSplitTypeId": 1,
        "scoringPeriodId": 2,
        "appliedTotal": 19.4,
        "stats": {},
    },
    {
        "id": "1120251",
        "seasonId": 2025,
        "statSourceId": 1,
        "statSplitTypeId": 1,
        "scoringPeriodId": 1,
        "appliedTotal": 17.0,
        "stats": {},
    },
]


def test_parses_every_row():
    assert len(parse_rows(RAW)) == len(RAW)


def test_weekly_projections_are_season_scoped():
    """The whole point: a 2026 query returns 2025 rows too."""
    assert weekly_projections(parse_rows(RAW), 2026) == {1: 18.0, 2: 20.0, 3: 22.0}
    assert weekly_projections(parse_rows(RAW), 2025) == {1: 17.0}


def test_weekly_actuals_keyed_by_scoring_period():
    """Actual weekly rows carry an event id, so scoringPeriodId is the only week key."""
    assert weekly_actuals(parse_rows(RAW), 2025) == {2: 19.4}


def test_rest_of_season_sums_remaining_weeks_not_the_frozen_field():
    rows = parse_rows(RAW)
    # The frozen field says 300; the weeks that remain say 42.
    assert frozen_season_projection(rows, 2026) == 300.0
    assert rest_of_season_projection(rows, 2026, from_week=2) == 42.0
    assert rest_of_season_projection(rows, 2026, from_week=1) == 60.0


def test_rest_of_season_is_inclusive_of_the_current_week():
    rows = parse_rows(RAW)
    assert rest_of_season_projection(rows, 2026, from_week=3) == 22.0


def test_missing_fields_do_not_explode():
    assert parse_rows([{"id": "x"}])[0].applied_total == 0.0
