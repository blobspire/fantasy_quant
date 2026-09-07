"""Snapshot flattening, and a live regression test on the stat-filter trap."""

from __future__ import annotations

import polars as pl
import pytest

from fantasy_quant.espn.client import EspnClient
from fantasy_quant.espn.endpoints import league_default_url
from fantasy_quant.snapshot import _flatten

ENTRY = {
    "id": 4429795,
    "status": "FREEAGENT",
    "onTeamId": 0,
    "keeperValue": 38,
    "player": {
        "fullName": "Test Player",
        "defaultPositionId": 3,
        "proTeamId": 8,
        "injuryStatus": "ACTIVE",
        "eligibleSlots": [3, 4, 23],
        "ownership": {"percentOwned": 99.9, "percentStarted": 98.0},
        "stats": [
            {
                "id": "1120261",
                "seasonId": 2026,
                "statSourceId": 1,
                "statSplitTypeId": 1,
                "scoringPeriodId": 1,
                "appliedTotal": 18.0,
                "stats": {"53": 4.0, "42": 80.0},
            },
        ],
    },
}


def test_flatten_emits_one_row_per_stat_row():
    rows = _flatten(ENTRY, 2026)
    assert len(rows) == 1
    assert rows[0]["full_name"] == "Test Player"
    assert rows[0]["percent_owned"] == 99.9
    assert rows[0]["applied_total"] == 18.0


def test_flatten_stores_stats_as_parallel_lists():
    """Dicts become Parquet structs whose schema varies per file and refuse to concat."""
    r = _flatten(ENTRY, 2026)[0]
    assert r["stat_ids"] == ["53", "42"]
    assert r["stat_values"] == [4.0, 80.0]


def test_flatten_keeps_players_with_no_stats():
    entry = {"id": 1, "player": {"fullName": "Rookie", "stats": []}}
    rows = _flatten(entry, 2026)
    assert len(rows) == 1
    assert rows[0]["stat_ids"] == []


def test_schema_is_stable_across_differing_stat_ids():
    a = _flatten(ENTRY, 2026)
    b = _flatten(
        {
            **ENTRY,
            "player": {
                **ENTRY["player"],
                "stats": [
                    {
                        "id": "1120262",
                        "seasonId": 2026,
                        "statSourceId": 1,
                        "statSplitTypeId": 1,
                        "scoringPeriodId": 2,
                        "appliedTotal": 9.0,
                        "stats": {"999": 1.0},
                    }
                ],
            },
        },
        2026,
    )
    overrides = {
        "stat_ids": pl.List(pl.Utf8),
        "stat_values": pl.List(pl.Float64),
        "eligible_slots": pl.List(pl.Int64),
    }
    da = pl.DataFrame(a, infer_schema_length=None, schema_overrides=overrides)
    db = pl.DataFrame(b, infer_schema_length=None, schema_overrides=overrides)
    assert pl.concat([da, db], how="diagonal").height == 2


@pytest.mark.network
def test_unfiltered_pool_returns_weekly_projections():
    """Pins the trap that cost us a bad first capture.

    Narrowing the payload with `filterStatsForTopScoringPeriodIds` silently drops
    the weekly PROJECTION rows and returns actuals only. If a future change
    reintroduces a stat filter, this fails.
    """
    with EspnClient() as client:
        season, _ = client.current_season_and_week()
        players = client.player_pool(
            league_default_url(season, "ppr"),
            limit=10,
            params={"view": "kona_player_info"},
            max_players=10,
        )
    assert players
    weeks = {
        s.get("scoringPeriodId")
        for p in players
        for s in (p["player"].get("stats") or [])
        if s.get("seasonId") == season
        and s.get("statSourceId") == 1
        and s.get("statSplitTypeId") == 1
    }
    assert len(weeks) >= 15, f"expected a full weekly projection set, got {sorted(weeks)}"
