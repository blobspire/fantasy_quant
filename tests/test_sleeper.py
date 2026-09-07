"""Sleeper adapter: offline parsing of a recorded payload, plus live canaries.

The fixtures below are verbatim records from the live 2026 week-1 response, trimmed
only of fields nothing reads. Every offline test here corresponds to a documented
trap in `data/sleeper.py`; the point is to fail when Sleeper's shape moves, not to
re-assert that dictionaries have keys.
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import polars as pl
import pytest

from fantasy_quant.data.sleeper import (
    ADP_FIELDS,
    ADP_SENTINELS,
    BASE,
    RECEPTION_BUCKETS,
    SKILL_POSITIONS,
    SLEEPER_SOURCE,
    SleeperClient,
    SleeperError,
    TrendingPlayer,
    WaiverDemand,
    attach_internal_ids,
    crosswalk_rows,
    default_id_resolver,
    parse_projections,
    parse_trending,
    to_frame,
)

# Ben Mason: `player.position` is FB but `fantasy_positions` is ["TE"], which is why
# the server hands him back on a TE request. He also carries nothing but the ADP
# sentinel -- the shape the great majority of every response takes.
JUNK_FB = {
    "stats": {"adp_dd_ppr": 1000.0},
    "category": "proj",
    "week": 1,
    "season": "2026",
    "season_type": "regular",
    "player": {
        "fantasy_positions": ["TE"],
        "first_name": "Ben",
        "last_name": "Mason",
        "position": "FB",
        "team": None,
        "injury_status": None,
        "years_exp": 3,
    },
    "team": None,
    "player_id": "7808",
    "opponent": None,
    "game_id": None,
    "company": "rotowire",
}

ALLEN = {
    "stats": {
        "adp_dd_ppr": 37.0,
        "bonus_rush_td_qb": 0.55,
        "cmp_pct": 64.8,
        "fum_lost": 0.17,
        "gp": 1.0,
        "pass_2pt": 0.09,
        "pass_att": 31.5,
        "pass_cmp": 20.41,
        "pass_fd": 23.52,
        "pass_int": 0.72,
        "pass_td": 1.64,
        "pass_yd": 235.21,
        "pos_adp_dd_ppr": 1.0,
        "pts_half_ppr": 23.26,
        "pts_ppr": 23.26,
        "pts_std": 23.26,
        "rush_att": 5.82,
        "rush_fd": 2.63,
        "rush_td": 0.55,
        "rush_yd": 26.3,
    },
    "category": "proj",
    "week": 1,
    "season": "2026",
    "season_type": "regular",
    "player": {
        "fantasy_positions": ["QB"],
        "first_name": "Josh",
        "last_name": "Allen",
        "position": "QB",
        "team": "BUF",
        "injury_status": None,
        "years_exp": 8,
    },
    "team": "BUF",
    "player_id": "4984",
    "opponent": "HOU",
    "updated_at": 1788799844721,
    "game_id": "202610113",
    "company": "rotowire",
}

GIBBS = {
    "stats": {
        "adp_dd_ppr": 1.0,
        "bonus_rec_rb": 4.6,
        "fum_lost": 0.1,
        "gp": 1.0,
        "pts_half_ppr": 21.38,
        "pts_ppr": 23.68,
        "pts_std": 19.08,
        "rec": 4.6,
        "rec_0_4": 0.92,
        "rec_5_9": 0.92,
        "rec_10_19": 1.38,
        "rec_20_29": 0.92,
        "rec_30_39": 0.46,
        "rec_40p": 0.46,
        "rec_2pt": 0.01,
        "rec_fd": 3.07,
        "rec_td": 0.22,
        "rec_tgt": 5.43,
        "rec_yd": 30.67,
        "rush_att": 17.29,
        "rush_fd": 9.07,
        "rush_td": 0.94,
        "rush_yd": 90.73,
    },
    "category": "proj",
    "week": 1,
    "season": "2026",
    "season_type": "regular",
    "player": {
        "fantasy_positions": ["RB"],
        "first_name": "Jahmyr",
        "last_name": "Gibbs",
        "position": "RB",
        "team": "DET",
        "injury_status": None,
        "years_exp": 3,
    },
    "team": "DET",
    "player_id": "9221",
    "opponent": "NO",
    "updated_at": 1788799844758,
    "game_id": "202610111",
    "company": "rotowire",
}

# Server order, verbatim in spirit: junk first, and the higher-scoring player last.
WEEKLY_PAYLOAD = [JUNK_FB, ALLEN, GIBBS]

# Season row: the 12 ADP flavors, two of them sentinel, and NO rec_tgt.
NACUA_SEASON = {
    "stats": {
        "adp_2qb": 5.7,
        "adp_dynasty": 999.0,
        "adp_dynasty_2qb": 7.9,
        "adp_dynasty_half_ppr": 4.4,
        "adp_dynasty_ppr": 4.0,
        "adp_dynasty_std": 4.3,
        "adp_half_ppr": 6.9,
        "adp_idp": 5.3,
        "adp_idp_1qb": 4.8,
        "adp_ppr": 4.2,
        "adp_rookie": 999.0,
        "adp_std": 8.9,
        "gp": 18.0,
        "pts_ppr": 312.5,
        "rec": 107.0,
        "rec_0_4": 21.4,
        "rec_5_9": 21.4,
        "rec_10_19": 32.1,
        "rec_20_29": 21.4,
        "rec_30_39": 10.7,
        "rec_40p": 10.7,
        "rec_fd": 140.0,
        "rec_td": 10.0,
        "rec_yd": 1400.0,
        "rush_att": 10.0,
        "rush_yd": 55.0,
    },
    "category": "proj",
    "week": None,
    "season": "2026",
    "season_type": "regular",
    "player": {
        "fantasy_positions": ["WR"],
        "first_name": "Puka",
        "last_name": "Nacua",
        "position": "WR",
        "team": "LAR",
        "injury_status": None,
    },
    "team": "LAR",
    "player_id": "9493",
    "game_id": "season",
    "company": "rotowire",
}

# What a bad week, a bad season, or an unpopulated `company=` returns: HTTP 200 and
# the whole player list, every row sentinel-only.
SKELETON_PAYLOAD = [JUNK_FB, {**JUNK_FB, "player_id": "4504"}]


def _client_with(handler) -> SleeperClient:
    """A SleeperClient wired to an in-process transport. No network, no sleeping."""
    client = SleeperClient()
    client._client.close()
    client._client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler))
    return client


# --------------------------------------------------------------------------- parsing


def test_projectionless_rows_are_dropped_by_default():
    """Live week 1 returned 3,115 rows and 398 projections; the rest are noise."""
    rows = parse_projections(WEEKLY_PAYLOAD)
    assert [r.name for r in rows] == ["Jahmyr Gibbs", "Josh Allen"]

    kept = parse_projections(WEEKLY_PAYLOAD, projected_only=False)
    assert len(kept) == 3
    assert [r.has_projection for r in kept] == [True, True, False]


def test_sentinel_only_row_is_not_a_projection():
    """`adp_dd_ppr: 1000.0` and a `gp` are not evidence that Sleeper projected anyone."""
    (mason,) = parse_projections([JUNK_FB], projected_only=False)
    assert mason.stats == {"adp_dd_ppr": 1000.0}
    assert mason.has_projection is False
    assert (
        parse_projections([{**JUNK_FB, "stats": {"gp": 1.0}}], projected_only=False)[
            0
        ].has_projection
        is False
    )


def test_return_game_noise_is_not_a_projection():
    """Verbatim live rows that a blocklist-based `has_projection` let through.

    Luke McCaffrey, Dyami Brown and Isaac Guerendo came back in the live 2026 week-1
    QB/RB/WR/TE pull carrying nothing but the ADP sentinel, `gp`, and return-game
    yardage. They have no points and no offensive component, so `components()` is
    empty and they reach `to_frame` as all-null rows -- but they are not on any
    blocklist of "keys that mean nothing", which is why the check has to be an
    allowlist. This is also the 401-vs-398 gap in the live counts.
    """
    for stats in (
        {
            "adp_dd_ppr": 999.0,
            "def_kr_yd": 24.74,
            "gp": 1.0,
            "pos_adp_dd_ppr": 164.0,
            "pr": 0.12,
            "pr_yd": 1.19,
        },
        {"adp_dd_ppr": 999.0, "def_kr_yd": 9.28, "gp": 1.0, "pos_adp_dd_ppr": 163.0},
    ):
        (row,) = parse_projections([{**JUNK_FB, "stats": stats}], projected_only=False)
        assert row.has_projection is False, stats
        assert {k: v for k, v in row.components().items() if k != "gp"} == {}

    assert parse_projections([{**JUNK_FB, "stats": stats}]) == []

    # ... while a genuine projection with no points but a real component stays.
    (real,) = parse_projections([{**JUNK_FB, "stats": {"rec_tgt": 4.0, "gp": 1.0}}])
    assert real.has_projection is True


def test_order_by_is_decorative_so_we_sort_client_side():
    """The server sent junk, then Allen (23.26), then Gibbs (23.68). We fix that."""
    rows = parse_projections(WEEKLY_PAYLOAD)
    assert rows[0].name == "Jahmyr Gibbs"
    assert [r.pts_ppr for r in rows] == sorted((r.pts_ppr for r in rows), reverse=True)


def test_component_fields_survive_parsing():
    gibbs = parse_projections(WEEKLY_PAYLOAD)[0]
    comps = gibbs.components()
    for key in ("rec_tgt", "rush_att", "rec_fd", *RECEPTION_BUCKETS):
        assert key in comps, key
    assert comps["rec_tgt"] == 5.43
    assert comps["rush_att"] == 17.29

    allen = parse_projections(WEEKLY_PAYLOAD)[1]
    assert allen.components()["pass_att"] == 31.5


def test_components_exclude_league_scored_and_bonus_fields():
    """Points are already scored under someone else's rules; bonuses aren't events."""
    comps = parse_projections(WEEKLY_PAYLOAD)[0].components()
    assert "pts_ppr" not in comps
    assert "bonus_rec_rb" not in comps
    assert "adp_dd_ppr" not in comps


def test_reception_buckets_do_not_sum_to_receptions():
    """Pins the fact that they are shape, not a partition -- 69.3 vs 63.0 live."""
    (nacua,) = parse_projections([NACUA_SEASON])
    bucket_total = sum(nacua.stats[b] for b in RECEPTION_BUCKETS)
    assert bucket_total == pytest.approx(117.7)
    assert nacua.stats["rec"] == 107.0


def test_adp_sentinels_are_scrubbed_but_raw_stats_keep_them():
    (nacua,) = parse_projections([NACUA_SEASON])
    adp = nacua.adp()
    assert "adp_dynasty" not in adp
    assert "adp_rookie" not in adp
    assert adp["adp_ppr"] == 4.2
    assert len(adp) == len(ADP_FIELDS) - 2
    # Averaging the raw dict is exactly the mistake; the raw value is still there
    # so a caller that wants "unranked" can see it.
    assert nacua.stats["adp_dynasty"] in ADP_SENTINELS


def test_all_twelve_adp_flavors_are_declared():
    assert len(ADP_FIELDS) == 12
    assert set(ADP_FIELDS) <= set(NACUA_SEASON["stats"])


def test_position_filter_keys_on_fantasy_positions_not_player_position():
    """Ben Mason is `position: FB`, `fantasy_positions: ["TE"]`.

    The server's `position[]` matches him on a TE request, so filtering on
    `player.position` would silently disagree with the pool we asked for.
    """
    kept = parse_projections([JUNK_FB], positions=["TE"], projected_only=False)
    assert [r.name for r in kept] == ["Ben Mason"]
    assert kept[0].position == "FB"

    assert parse_projections([JUNK_FB], positions=["QB"], projected_only=False) == []


def test_parse_handles_missing_fields_and_bad_types():
    rows = parse_projections(
        [{"player_id": "1", "stats": {"rec_tgt": "7.5", "rush_att": None}}],
        projected_only=False,
    )
    assert rows[0].name == "1"
    assert rows[0].stats == {"rec_tgt": 7.5}
    assert rows[0].season == 0
    assert rows[0].week is None
    assert rows[0].updated_at is None


def test_updated_at_is_parsed_from_epoch_millis():
    (allen,) = parse_projections([ALLEN])
    assert allen.updated_at == dt.datetime(2026, 9, 7, 16, 50, 44, 721000, tzinfo=dt.UTC)


def test_parse_rejects_a_non_list_payload():
    with pytest.raises(SleeperError):
        parse_projections({"error": "nope"})


# ------------------------------------------------------------------------- trending


def test_parse_trending_sorts_and_keeps_team_defenses():
    rows = parse_trending(
        [
            {"count": 5330, "player_id": "13307"},
            {"count": 91206, "player_id": "LV"},
            {"count": 239360, "player_id": "10235"},
            {"count": None, "player_id": "bad"},
        ]
    )
    assert [r.sleeper_id for r in rows] == ["10235", "LV", "13307"]
    # D/ST arrives as a team abbreviation, not a numeric id.
    assert rows[1] == TrendingPlayer("LV", 91206)


def test_waiver_demand_net_adds_catches_churn():
    """Live, the most-added player in the window was also the most-dropped."""
    demand = WaiverDemand(
        lookback_hours=24,
        captured_at=dt.datetime(2026, 9, 7, tzinfo=dt.UTC),
        adds={"10235": 239360, "11834": 132713, "9502": 102504},
        drops={"10235": 52160, "9502": 1000},
    )
    assert demand.net_adds("10235") == 187200
    assert demand.net_adds("11834") == 132713
    assert demand.net_adds("nobody") == 0


def test_demand_share_is_log_scaled_and_monotone():
    demand = WaiverDemand(
        lookback_hours=24,
        captured_at=dt.datetime(2026, 9, 7, tzinfo=dt.UTC),
        adds={"top": 239360, "mid": 20000, "low": 5330, "noise": 400},
    )
    assert demand.demand_share("top") == pytest.approx(0.99, abs=0.02)
    assert demand.demand_share("unknown") == 0.0
    # Below the floor is genuinely nobody, not "a little bit of interest".
    assert demand.demand_share("noise") == 0.0
    assert demand.demand_share("top") > demand.demand_share("mid") > demand.demand_share("low")
    # The whole reason for the log: on a linear scale "mid" would sit at 0.08,
    # whereas 20k adds in a 24h window is a genuinely contested player.
    assert demand.demand_share("mid") > 0.5


def test_demand_share_does_not_depend_on_the_rest_of_the_window():
    """Anchored on absolute counts, so `limit` and a quiet week don't move it.

    Normalizing by the window maximum would score a 20k-add player 0.83 in a quiet
    week and 0.63 in a hot one, which makes the number uncomparable across days --
    and the FAAB model reads it across days.
    """
    quiet = WaiverDemand(24, dt.datetime(2026, 9, 7, tzinfo=dt.UTC), adds={"x": 20000})
    hot = WaiverDemand(
        24, dt.datetime(2026, 9, 7, tzinfo=dt.UTC), adds={"x": 20000, "star": 900000}
    )
    assert quiet.demand_share("x") == hot.demand_share("x")


def test_demand_share_scales_with_the_lookback_window():
    """20k adds in 6 hours is four times the interest of 20k adds in a day."""
    day = WaiverDemand(24, dt.datetime(2026, 9, 7, tzinfo=dt.UTC), adds={"x": 20000})
    six_hours = WaiverDemand(6, dt.datetime(2026, 9, 7, tzinfo=dt.UTC), adds={"x": 20000})
    assert six_hours.demand_share("x") > day.demand_share("x")


def test_expected_bidders_is_bounded_and_clamped_by_league_size():
    demand = WaiverDemand(
        lookback_hours=24,
        captured_at=dt.datetime(2026, 9, 7, tzinfo=dt.UTC),
        adds={"top": 239360, "low": 5330},
    )
    assert demand.expected_bidders("top") == pytest.approx(5.0, abs=0.1)
    assert 1.0 < demand.expected_bidders("low") < 5.0
    assert demand.expected_bidders("nobody") == 1.0
    # A 4-team league cannot produce 5 rival bidders.
    assert demand.expected_bidders("top", league_size=4) == 3.0
    with pytest.raises(ValueError):
        demand.expected_bidders("top", league_size=1)


def test_waiver_demand_exposes_counts_not_just_a_top_n():
    """The FAAB model needs a per-player count; a ranked list alone is not enough."""
    demand = WaiverDemand(
        lookback_hours=24,
        captured_at=dt.datetime(2026, 9, 7, tzinfo=dt.UTC),
        adds={"a": 10, "b": 20},
    )
    assert demand.add_count("b") == 20
    assert [r.sleeper_id for r in demand.top_adds(1)] == ["b"]


# ---------------------------------------------------------------------------- client


def test_skeleton_response_raises_instead_of_returning_nothing():
    """Trap 1: a bad week is HTTP 200 with the full player list and no projections."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SKELETON_PAYLOAD)

    with _client_with(handler) as client, pytest.raises(SleeperError, match="skeleton"):
        client.weekly_projections(2026, 99)


def test_unpopulated_company_hits_the_same_alarm():
    """`company=sleeper` returns 750 rows of `{"adp_dd_ppr": 1000.0}`, not an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["company"] == "sleeper"
        return httpx.Response(200, json=SKELETON_PAYLOAD)

    with _client_with(handler) as client, pytest.raises(SleeperError):
        client.season_projections(2026, company="sleeper")


def test_positions_are_sent_as_repeated_bracket_params():
    seen: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["positions"] = request.url.params.get_list("position[]")
        return httpx.Response(200, json=WEEKLY_PAYLOAD)

    with _client_with(handler) as client:
        client.weekly_projections(2026, 1, positions=SKILL_POSITIONS)
    assert seen["positions"] == list(SKILL_POSITIONS)


def test_trending_limit_is_capped_at_the_server_maximum():
    """`limit=500` returned 100 rows live; don't pretend otherwise."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["limit"] = request.url.params["limit"]
        return httpx.Response(200, json=[{"player_id": "1", "count": 5}])

    with _client_with(handler) as client:
        client.trending("add", limit=500)
    assert seen["limit"] == "100"


def test_trending_rejects_an_unknown_kind():
    with _client_with(lambda r: httpx.Response(200, json=[])) as client, pytest.raises(ValueError):
        client.trending("steal")


def test_conditional_get_reuses_the_cached_body_on_304():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("if-none-match", ""))
        if request.headers.get("if-none-match"):
            return httpx.Response(304)
        return httpx.Response(200, json=WEEKLY_PAYLOAD, headers={"etag": 'W/"abc"'})

    with _client_with(handler) as client:
        first = client.weekly_projections(2026, 1)
        second = client.weekly_projections(2026, 1)

    assert calls == ["", 'W/"abc"']
    assert [r.sleeper_id for r in first] == [r.sleeper_id for r in second]


def test_http_errors_become_sleeper_errors():
    boom = _client_with(lambda r: httpx.Response(500, text="boom"))
    with boom, pytest.raises(SleeperError, match="500"):
        boom.trending()

    limited = _client_with(lambda r: httpx.Response(429, text="slow down"))
    with limited, pytest.raises(SleeperError, match="rate limited"):
        limited.trending()


def test_players_master_is_cached_to_disk(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"9221": {"full_name": "Jahmyr Gibbs"}})

    client = _client_with(handler)
    client._cache_dir = tmp_path
    assert client.players()["9221"]["full_name"] == "Jahmyr Gibbs"
    assert (tmp_path / "players_nfl.json").exists()

    # A second client, a cold process: the disk cache must still serve it.
    other = _client_with(handler)
    other._cache_dir = tmp_path
    assert other.players()["9221"]["full_name"] == "Jahmyr Gibbs"
    assert calls["n"] == 1

    assert other.players(force=True)
    assert calls["n"] == 2


def test_empty_player_master_is_refused(tmp_path):
    client = _client_with(lambda r: httpx.Response(200, json={}))
    client._cache_dir = tmp_path
    with pytest.raises(SleeperError):
        client.players()
    assert not (tmp_path / "players_nfl.json").exists()


# ------------------------------------------------------------------- ids + frames


PLAYER_MASTER = {
    "10235": {
        "active": True,
        "full_name": "Roschon Johnson",
        "search_full_name": "roschonjohnson",
        "position": "RB",
        "fantasy_positions": ["RB"],
        "team": "CHI",
        "espn_id": None,
        "gsis_id": None,
        # 520 players in the live master carry exactly this: an id that is present
        # as a key, blank as a value.
        "sportradar_id": "",
        "rotowire_id": 16772,
        "years_exp": 3,
    },
    "9221": {
        "active": True,
        "full_name": "Jahmyr Gibbs",
        "search_full_name": "jahmyrgibbs",
        "position": "RB",
        "fantasy_positions": ["RB"],
        "team": "DET",
        "espn_id": 4426502,
        "gsis_id": "00-0038543",
        "sportradar_id": "9f4d0a2e",
        "years_exp": 3,
    },
    # Verbatim shape of the live master's 32 defenses: no full_name, no
    # search_full_name, the team abbreviation as the player id.
    "LV": {
        "active": True,
        "position": "DEF",
        "fantasy_positions": ["DEF"],
        "team": "LV",
        "first_name": "Las Vegas",
        "last_name": "Raiders",
    },
    "old": {"active": False, "fantasy_positions": ["WR"], "full_name": "Retired Guy"},
}


def test_crosswalk_rows_filters_and_preserves_missing_ids_as_null():
    """24.3% ESPN coverage measured live -- a null must stay null."""
    rows = crosswalk_rows(PLAYER_MASTER)
    by_id = {r["sleeper_id"]: r for r in rows}
    assert "old" not in by_id, "inactive players must be filtered out"
    assert by_id["10235"]["espn_id"] is None
    assert by_id["9221"]["espn_id"] == 4426502
    assert by_id["10235"]["search_full_name"] == "roschonjohnson"


def test_crosswalk_blank_string_ids_read_as_missing_not_as_present():
    """The `db_playerids` "NA" trap in Sleeper's own idiom.

    RESEARCH: missing values in db_playerids are the literal string "NA", so naive
    truthiness reports a fake 100% coverage. Sleeper's version is `""` -- 520
    players in the live master carry `sportradar_id: ""`. A coverage check written
    the obvious way (`is not None`, which is exactly how the live coverage test
    below counts espn_id) would score those as joinable.
    """
    by_id = {r["sleeper_id"]: r for r in crosswalk_rows(PLAYER_MASTER)}
    assert by_id["10235"]["sportradar_id"] is None
    assert by_id["9221"]["sportradar_id"] == "9f4d0a2e"
    # And the same for the literal markers other feeds use.
    na = crosswalk_rows({"x": {**PLAYER_MASTER["9221"], "gsis_id": "NA", "team": " "}})
    assert na[0]["gsis_id"] is None
    assert na[0]["team"] is None


def test_crosswalk_keeps_team_defenses_despite_the_skill_position_default():
    """RESEARCH: "every join failure in testing was a team defense".

    `trending()` returns D/ST as a bare abbreviation ("LV"), so a crosswalk that
    dropped defenses under the default `positions=SKILL_POSITIONS` could not
    resolve the one row class that is known to break.
    """
    by_id = {r["sleeper_id"]: r for r in crosswalk_rows(PLAYER_MASTER)}
    assert "LV" in by_id, "team defenses were filtered out of the crosswalk"
    assert by_id["LV"]["position"] == "DEF"
    # Upstream leaves both name columns null on defenses; a null name column is a
    # hole in the only join key Sleeper populates 100% of the time.
    assert by_id["LV"]["full_name"] == "Las Vegas Raiders"
    assert by_id["LV"]["search_full_name"] == "lasvegasraiders"

    assert "LV" not in {r["sleeper_id"] for r in crosswalk_rows(PLAYER_MASTER, include_dst=False)}


def test_attach_internal_ids_degrades_to_empty_when_the_resolver_finds_nothing():
    rows = parse_projections(WEEKLY_PAYLOAD)
    assert attach_internal_ids(rows, resolver=lambda _: {}) == {}


def test_attach_internal_ids_uses_an_injected_resolver():
    rows = parse_projections(WEEKLY_PAYLOAD)
    mapping = attach_internal_ids(rows, resolver=lambda ps: {p.sleeper_id: p.name for p in ps})
    assert mapping == {"9221": "Jahmyr Gibbs", "4984": "Josh Allen"}


def test_a_broken_resolver_does_not_take_the_feed_down():
    """`data.ids` is being written in parallel; Sleeper must survive it being wrong."""

    def boom(_):
        raise RuntimeError("ids module exploded")

    assert attach_internal_ids(parse_projections(WEEKLY_PAYLOAD), resolver=boom) == {}


def test_attach_internal_ids_accepts_an_ids_index_object():
    """`data.ids.IdResolver` exposes `.to_canonical(value, source)`, not a callable.

    Passing one straight in must work, so a caller that already built the crosswalk
    does not have to rebuild it or hand-write an adapter.
    """

    class FakeIndex:
        def to_canonical(self, value, source):
            assert source == SLEEPER_SOURCE
            return {"9221": "00-0039139"}.get(str(value))

    mapping = attach_internal_ids(parse_projections(WEEKLY_PAYLOAD), resolver=FakeIndex())
    assert mapping == {"9221": "00-0039139"}


def test_default_id_resolver_binds_to_the_real_ids_module():
    """The duck-typed lookup must match the API `data.ids` actually ships.

    Guessed names are a silent-failure machine: if none of them exist,
    `attach_internal_ids` returns `{}` forever and every downstream join quietly
    degrades to "Sleeper ids only" with nothing to notice. Pin the binding to the
    real module rather than to the guess list.
    """
    ids = pytest.importorskip("fantasy_quant.data.ids")
    assert hasattr(ids, "IdResolver"), "data.ids no longer exposes IdResolver"
    assert ids.SLEEPER == SLEEPER_SOURCE

    # Skip on the external precondition (no crosswalk cached on this machine), never
    # on the outcome under test -- a skip keyed on `resolver is None` would go green
    # for the very bug this pins.
    if not any(ids.REFERENCE_DIR.glob("roster_*.parquet")):
        pytest.skip(f"no cached crosswalk under {ids.REFERENCE_DIR}; run the ids loader first")

    default_id_resolver.cache_clear()
    resolver = default_id_resolver()
    assert resolver is not None, (
        "data.ids ships IdResolver and a cached crosswalk, but the Sleeper binding "
        "found neither -- attach_internal_ids would return {} forever"
    )

    rows = parse_projections(WEEKLY_PAYLOAD)
    mapping = attach_internal_ids(rows)
    assert mapping, "default resolver produced no internal ids for a live-shaped payload"
    # Gibbs and Allen are both in every crosswalk we ship.
    assert set(mapping) == {"9221", "4984"}
    assert all(v and v != k for k, v in mapping.items())


def test_default_id_resolver_does_not_download_by_default():
    """Importing a projection feed must not silently trigger a crosswalk download."""
    import inspect

    sig = inspect.signature(default_id_resolver.__wrapped__)
    assert sig.parameters["allow_download"].default is False


def test_frame_schema_is_fixed_so_weeks_concatenate():
    """The `snapshot.py` lesson: a schema that follows the data cannot be appended."""
    qb = to_frame(parse_projections([ALLEN]))
    rb = to_frame(parse_projections([GIBBS]))
    assert qb.schema == rb.schema
    assert pl.concat([qb, rb]).height == 2
    assert qb.schema["rec_tgt"] == pl.Float64


def test_frame_distinguishes_missing_from_zero():
    """ "Sleeper did not project this" is not "Sleeper projected zero"."""
    df = to_frame(parse_projections([ALLEN]))
    assert df["rec_tgt"][0] is None
    assert df["pass_att"][0] == 31.5


def test_frame_scrubs_adp_sentinels():
    df = to_frame(parse_projections([NACUA_SEASON]))
    assert df["adp_dynasty"][0] is None
    assert df["adp_ppr"][0] == 4.2


def test_empty_frame_still_has_the_full_schema():
    df = to_frame([])
    assert df.height == 0
    assert "rec_tgt" in df.columns


def test_frame_carries_fantasy_positions_not_just_the_misleading_one():
    """Trap 3 does not stop at the API boundary.

    `player.position` is the field Sleeper does *not* filter on. Persisting only
    that column means a downstream `group_by("position")` invents an `FB` bucket
    and leaves four fullbacks out of the RB pool Sleeper itself put there -- live
    week 1 wrote exactly four `FB` rows and a `DB` row into the frame.
    """
    df = to_frame(parse_projections([JUNK_FB], positions=["TE"], projected_only=False))
    assert df["position"][0] == "FB"
    assert list(df["fantasy_positions"][0]) == ["TE"]

    # And the column must not break the fixed-schema concat contract.
    other = to_frame(parse_projections([GIBBS]))
    assert df.schema == other.schema
    assert pl.concat([df, other]).height == 2
    assert df.schema["fantasy_positions"] == pl.List(pl.Utf8)


# --------------------------------------------------------------------------- network


@pytest.mark.network
def test_live_weekly_projections_carry_component_fields():
    with SleeperClient() as client:
        season, week = client.current_season_and_week()
        rows = client.weekly_projections(season, week)

    assert len(rows) > 50, f"only {len(rows)} projected players in {season} week {week}"
    top = rows[:25]
    for key in ("rec_tgt", "rush_att", "rec_fd", *RECEPTION_BUCKETS):
        assert any(key in r.components() for r in top), f"no {key} anywhere in the top 25"
    assert any("pass_att" in r.components() for r in top), "no pass_att in the top 25"
    # Every kept row must carry a real projection, not just return-game noise.
    assert all(r.has_projection for r in rows)
    assert all(r.components() or r.pts_ppr is not None for r in rows)


@pytest.mark.network
def test_live_position_filter_really_does_key_on_fantasy_positions():
    """Trap 3, measured on the RAW payload rather than on our own filtered output.

    Asserting `fantasy_positions` intersects the requested pool on rows we already
    filtered that way is a tautology -- it stays green even if the filter is
    deleted. The falsifiable statement is about the server: `position[]` selects on
    `fantasy_positions` (so every raw row intersects) while `player.position` does
    not (so filtering on it would drop rows the server deliberately included).
    """
    with SleeperClient() as client:
        season, week = client.current_season_and_week()
        raw = client.get(
            f"/projections/nfl/{season}/{week}",
            params={"season_type": "regular", "position[]": list(SKILL_POSITIONS)},
        )

    assert isinstance(raw, list) and raw
    wanted = set(SKILL_POSITIONS)
    by_fantasy = [
        r for r in raw if wanted & set((r.get("player") or {}).get("fantasy_positions") or ())
    ]
    by_position = [r for r in raw if (r.get("player") or {}).get("position") in wanted]

    assert len(by_fantasy) == len(raw), (
        f"{len(raw) - len(by_fantasy)} rows do not intersect the requested pool on "
        "fantasy_positions -- position[] may no longer key on that field"
    )
    assert len(by_position) < len(raw), (
        "player.position now agrees with the requested pool on every row; the "
        "off-position rows (71 FBs, a CB and a DB live) are gone and trap 3 in the "
        "module docstring needs re-checking"
    )


@pytest.mark.network
def test_live_season_projections_carry_the_adp_flavors():
    with SleeperClient() as client:
        rows = client.season_projections(2026, positions=["RB"])

    assert rows
    seen = set().union(*(r.stats.keys() for r in rows))
    missing = set(ADP_FIELDS) - seen
    assert not missing, f"season endpoint lost ADP flavors: {sorted(missing)}"
    assert any(r.adp().get("adp_ppr") for r in rows)

    # Documenting a live disagreement with the brief: `rec_tgt` is published on the
    # weekly endpoint only. If that ever changes, say so rather than failing.
    if "rec_tgt" in seen:
        pytest.skip("Sleeper now publishes rec_tgt on the season endpoint -- update the docstring")


@pytest.mark.network
def test_live_bad_week_returns_a_skeleton_not_an_error():
    """The trap that makes status-code checking useless here."""
    with SleeperClient() as client:
        raw = client.get(
            "/projections/nfl/2026/99",
            params={"season_type": "regular", "position[]": ["QB"]},
        )
        assert isinstance(raw, list) and raw, "expected HTTP 200 with a full skeleton list"
        with pytest.raises(SleeperError):
            client.weekly_projections(2026, 99, positions=["QB"])


@pytest.mark.network
@pytest.mark.parametrize("company", ["sleeper", "fantasypros"])
def test_live_company_probe(company):
    """Only `rotowire` had values when tested. Re-probe; report, don't silently pass.

    The bare `except SleeperError: return` this replaced could not fail: a DNS
    failure, a 500, or an unplugged network all raise SleeperError and all read as
    "the documented state". Reach Sleeper first and assert on the payload, so the
    only skip is the one that actually means "Sleeper started publishing this".
    """
    with SleeperClient() as client:
        raw = client.get(
            "/projections/nfl/2026",
            params={"season_type": "regular", "position[]": ["RB"], "company": company},
        )
        assert isinstance(raw, list) and raw, f"company={company} returned no rows at all"

        if parse_projections(raw, positions=["RB"]):
            pytest.skip(f"Sleeper now populates company={company} -- revisit the module")

        # The documented state: HTTP 200, a full skeleton, and our guard firing on
        # the emptiness rather than on the status code.
        with pytest.raises(SleeperError, match="skeleton"):
            client.season_projections(2026, positions=["RB"], company=company)


@pytest.mark.network
def test_live_rotowire_is_populated():
    with SleeperClient() as client:
        rows = client.season_projections(2026, positions=["RB"], company="rotowire")
    assert len(rows) > 50
    assert all(r.company == "rotowire" for r in rows)


@pytest.mark.network
def test_live_trending_returns_per_player_counts():
    with SleeperClient() as client:
        demand = client.waiver_demand(lookback_hours=24, limit=100)

    assert demand.adds, "no trending adds returned"
    assert all(c > 0 for c in demand.adds.values())
    # The server caps at 100 regardless of what we ask for.
    assert len(demand.adds) <= 100
    top = max(demand.adds, key=lambda k: demand.adds[k])
    tail = min(demand.adds, key=lambda k: demand.adds[k])
    assert demand.demand_share(top) > demand.demand_share(tail)
    assert 0.0 <= demand.demand_share(top) <= 1.0
    assert 1.0 <= demand.expected_bidders(top) <= 5.0


@pytest.mark.network
def test_live_player_master_espn_coverage_is_still_too_low_to_join_on(tmp_path):
    """Pins the research claim with a live measurement rather than a citation."""
    client = SleeperClient()
    client._cache_dir = tmp_path
    try:
        players = client.players()
    finally:
        client.close()

    rows = crosswalk_rows(players)
    teamed = [r for r in rows if r["team"]]
    assert len(teamed) > 400
    coverage = sum(1 for r in teamed if r["espn_id"] is not None) / len(teamed)
    assert coverage < 0.5, f"ESPN coverage jumped to {coverage:.1%} -- Sleeper may be joinable now"
    # The name key is the one column Sleeper populates for everyone, defenses
    # included -- it is the only thing this table is actually good for.
    assert all(r["search_full_name"] for r in teamed)
    # All 32 defenses survive the skill-position default; RESEARCH says every join
    # failure in testing was a team defense.
    dst = [r for r in rows if r["position"] == "DEF"]
    assert len(dst) == 32, f"expected 32 team defenses in the crosswalk, got {len(dst)}"
    assert all(r["search_full_name"] and r["full_name"] for r in dst)


@pytest.mark.network
def test_live_payload_round_trips_through_the_frame():
    with SleeperClient() as client:
        season, week = client.current_season_and_week()
        rows = client.weekly_projections(season, week)
    df = to_frame(rows)
    assert df.height == len(rows)
    assert df["pts_ppr"].max() > 10
    # Serializable end to end -- this is what a daily capture would write.
    assert json.loads(df.head(3).write_json())
