"""Tests for the nflverse data plane.

The offline half runs the real `NflverseCache` against an `httpx.MockTransport`,
so the staleness decisions, the atomic download and the 404 handling are exercised
end to end without a network. The `network` half is a thin liveness check on each
loader; CI deselects it.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator

import httpx
import polars as pl
import pytest

from fantasy_quant.data import nflverse as nv

# --- fixtures -------------------------------------------------------------------


# 4 teams over 4 weeks, shaped like the real thing: 16 team-weeks, 6 games
# (12 team-games) and therefore exactly 4 byes. Week 3 is missing BBB-CCC and
# week 4 is missing AAA-DDD, so every team sits exactly once. Written out by hand
# so the expected byes below can be read off the table.
#
#   week, home, away, spread_line (home-relative), total_line
_GAMES = [
    (1, "AAA", "BBB", 3.0, 44.0),
    (1, "CCC", "DDD", -1.5, 40.0),
    (2, "AAA", "CCC", 7.0, 50.0),
    (2, "BBB", "DDD", 0.0, 41.0),
    (3, "AAA", "DDD", 2.5, 45.0),
    (4, "BBB", "CCC", -3.5, 39.0),
]
_EXPECTED_BYES = {"AAA": 4, "BBB": 3, "CCC": 3, "DDD": 4}


def _fake_schedule(games: list[tuple] | None = None, season: int = 2026) -> pl.DataFrame:
    """A games.parquet-shaped frame with the columns the schedule views require."""
    games = _GAMES if games is None else games
    return pl.DataFrame(
        {
            "season": pl.Series([season] * len(games), dtype=pl.Int32),
            "game_type": ["REG"] * len(games),
            "week": pl.Series([g[0] for g in games], dtype=pl.Int32),
            "home_team": [g[1] for g in games],
            "away_team": [g[2] for g in games],
            "spread_line": [g[3] for g in games],
            "total_line": [g[4] for g in games],
            "game_id": [f"{season}_{g[0]:02d}_{g[2]}_{g[1]}" for g in games],
        }
    )


@pytest.fixture
def schedule() -> pl.DataFrame:
    return _fake_schedule()


def _transport(bodies: dict[str, bytes], seen: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        body = bodies.get(str(request.url))
        if body is None:
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})

    return httpx.MockTransport(handler)


def _cache(tmp_path, bodies, seen=None, **kw) -> nv.NflverseCache:
    client = httpx.Client(transport=_transport(bodies, seen), follow_redirects=True)
    return nv.NflverseCache(tmp_path, client=client, **kw)


ASSET = nv.Asset("schedules", "games.parquet")
STAMP_URL = f"{nv.RELEASE_BASE}/schedules/timestamp.txt"


def _bodies(stamp: str, payload: bytes = b"v1") -> dict[str, bytes]:
    return {STAMP_URL: stamp.encode(), ASSET.url: payload}


# --- pure staleness logic -------------------------------------------------------


def _meta(stamp: str, checked_ago: dt.timedelta = dt.timedelta(0)) -> nv.CacheMeta:
    now = dt.datetime.now(dt.UTC)
    return nv.CacheMeta(stamp, now - checked_ago, now - checked_ago, 10)


def test_no_meta_always_polls_and_is_stale() -> None:
    now = dt.datetime.now(dt.UTC)
    assert nv.needs_poll(None, now, dt.timedelta(hours=4))
    assert nv.is_stale(None, "anything")


def test_poll_window_suppresses_the_check_then_reopens() -> None:
    now = dt.datetime.now(dt.UTC)
    window = dt.timedelta(hours=4)
    assert not nv.needs_poll(_meta("s", dt.timedelta(hours=1)), now, window)
    assert nv.needs_poll(_meta("s", dt.timedelta(hours=5)), now, window)


def test_stale_only_when_the_release_stamp_moved() -> None:
    meta = _meta("2026-09-07 12:51:09 EDT")
    assert not nv.is_stale(meta, "2026-09-07 12:51:09 EDT")
    assert nv.is_stale(meta, "2026-09-07 13:06:24 EDT")


def test_failed_poll_keeps_the_cached_copy() -> None:
    """A dead network must not be read as "the file changed" and trigger a refetch."""
    assert not nv.is_stale(_meta("s"), None)


def test_naive_timestamps_in_old_metadata_are_read_as_utc() -> None:
    raw = {
        "release_stamp": "s",
        "fetched_at": "2026-09-01T00:00:00",
        "checked_at": "2026-09-01T00:00:00",
        "size": 1,
    }
    meta = nv.CacheMeta.from_dict(raw)
    assert meta.checked_at.tzinfo is dt.UTC
    # and it must still be comparable against an aware "now"
    assert nv.needs_poll(meta, dt.datetime.now(dt.UTC), dt.timedelta(hours=4))


# --- cache behavior against a mock transport ------------------------------------


def test_first_fetch_downloads_and_writes_metadata(tmp_path) -> None:
    seen: list[str] = []
    with _cache(tmp_path, _bodies("stamp-1"), seen) as cache:
        path = cache.ensure(ASSET)
    assert path.read_bytes() == b"v1"
    meta = json.loads(path.with_name("games.parquet.meta.json").read_text())
    assert meta["release_stamp"] == "stamp-1"
    assert meta["size"] == 2
    assert ASSET.url in seen


def test_unchanged_stamp_polls_but_does_not_redownload(tmp_path) -> None:
    """The whole point: 24 bytes of timestamp instead of megabytes of Parquet."""
    seen: list[str] = []
    bodies = _bodies("stamp-1")
    with _cache(tmp_path, bodies, seen, poll_after=dt.timedelta(0)) as cache:
        cache.ensure(ASSET)
        cache._stamps.clear()  # a fresh process would re-poll
        seen.clear()
        cache.ensure(ASSET)
    assert seen == [STAMP_URL]


def test_changed_stamp_redownloads(tmp_path) -> None:
    bodies = _bodies("stamp-1")
    with _cache(tmp_path, bodies, poll_after=dt.timedelta(0)) as cache:
        path = cache.ensure(ASSET)
        assert path.read_bytes() == b"v1"
        bodies[STAMP_URL] = b"stamp-2"
        bodies[ASSET.url] = b"v2-longer"
        cache._stamps.clear()
        assert cache.ensure(ASSET).read_bytes() == b"v2-longer"


def test_inside_the_poll_window_nothing_is_requested(tmp_path) -> None:
    seen: list[str] = []
    with _cache(tmp_path, _bodies("stamp-1"), seen, poll_after=dt.timedelta(hours=4)) as cache:
        cache.ensure(ASSET)
        seen.clear()
        cache.ensure(ASSET)
    assert seen == []


def test_force_redownloads_even_inside_the_window(tmp_path) -> None:
    bodies = _bodies("stamp-1")
    with _cache(tmp_path, bodies, poll_after=dt.timedelta(hours=4)) as cache:
        cache.ensure(ASSET)
        bodies[ASSET.url] = b"forced"
        assert cache.ensure(ASSET, force=True).read_bytes() == b"forced"


def test_deleted_file_is_refetched_even_with_valid_metadata(tmp_path) -> None:
    with _cache(tmp_path, _bodies("stamp-1"), poll_after=dt.timedelta(hours=4)) as cache:
        path = cache.ensure(ASSET)
        path.unlink()
        assert cache.ensure(ASSET).read_bytes() == b"v1"


def test_missing_asset_raises_not_found(tmp_path) -> None:
    with _cache(tmp_path, {STAMP_URL: b"stamp-1"}) as cache, pytest.raises(nv.NflverseNotFound):
        cache.ensure(nv.Asset("schedules", "nope.parquet"))


def test_truncated_download_is_rejected_and_leaves_no_file(tmp_path) -> None:
    """A short body must not land as a plausible-looking cached Parquet."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == STAMP_URL:
            return httpx.Response(200, content=b"stamp-1")
        return httpx.Response(200, content=b"ab", headers={"content-length": "9999"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with nv.NflverseCache(tmp_path, client=client) as cache:
        with pytest.raises(nv.NflverseError, match="truncated"):
            cache.ensure(ASSET)
        assert not cache.local_path(ASSET).exists()
        assert not cache.local_path(ASSET).with_name("games.parquet.part").exists()


def test_offline_cache_serves_what_it_has_and_refuses_what_it_does_not(tmp_path) -> None:
    with _cache(tmp_path, _bodies("stamp-1")) as cache:
        cache.ensure(ASSET)
    with _cache(tmp_path, {}, offline=True) as cache:
        assert cache.ensure(ASSET).read_bytes() == b"v1"
        with pytest.raises(nv.NflverseError, match="offline"):
            cache.ensure(nv.Asset("rosters", "roster_2026.parquet"))


def test_corrupt_metadata_forces_a_refetch(tmp_path) -> None:
    with _cache(tmp_path, _bodies("stamp-1")) as cache:
        path = cache.ensure(ASSET)
        path.with_name("games.parquet.meta.json").write_text("{not json")
        seen: list[str] = []
    with _cache(tmp_path, _bodies("stamp-1"), seen) as cache:
        cache.ensure(ASSET)
    assert ASSET.url in seen


def test_a_failed_poll_does_not_shut_the_poll_window(tmp_path) -> None:
    """A network blip must not buy the cached copy another `poll_after` of trust.

    Regression: `ensure` used to rewrite `checked_at = now` on the "not stale"
    branch even when the branch was reached because the poll *failed*. A single
    dropped request then suppressed every check for four hours, and a Wednesday
    stat correction published one minute later was served as the old numbers with
    no request made at all -- the exact failure this module exists to prevent.
    """
    bodies = _bodies("stamp-1")
    with _cache(tmp_path, bodies, poll_after=dt.timedelta(hours=4)) as cache:
        meta_path = cache.ensure(ASSET).with_name("games.parquet.meta.json")

    # Reopen the window as if four hours had passed.
    meta = json.loads(meta_path.read_text())
    stale_check = (dt.datetime.now(dt.UTC) - dt.timedelta(hours=9)).isoformat()
    meta["checked_at"] = stale_check
    meta_path.write_text(json.dumps(meta))

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down", request=request)

    dead_client = httpx.Client(transport=httpx.MockTransport(dead))
    with nv.NflverseCache(tmp_path, client=dead_client, poll_after=dt.timedelta(hours=4)) as cache:
        assert cache.ensure(ASSET).read_bytes() == b"v1"  # cached copy still served
    assert json.loads(meta_path.read_text())["checked_at"] == stale_check

    # Network back, upstream moved: the very next call must notice.
    bodies[STAMP_URL] = b"stamp-2"
    bodies[ASSET.url] = b"v2-corrected"
    seen: list[str] = []
    with _cache(tmp_path, bodies, seen, poll_after=dt.timedelta(hours=4)) as cache:
        assert cache.ensure(ASSET).read_bytes() == b"v2-corrected"
    assert seen == [STAMP_URL, ASSET.url]


def test_a_sidecar_with_a_null_field_refetches_instead_of_raising(tmp_path) -> None:
    """`size: null` reaches int() as None -- a TypeError, not a ValueError."""
    with _cache(tmp_path, _bodies("stamp-1"), poll_after=dt.timedelta(0)) as cache:
        meta_path = cache.ensure(ASSET).with_name("games.parquet.meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "release_stamp": "stamp-1",
                    "fetched_at": "2026-09-01T00:00:00+00:00",
                    "checked_at": None,
                    "size": None,
                }
            )
        )
        assert cache.ensure(ASSET).read_bytes() == b"v1"
    assert json.loads(meta_path.read_text())["size"] == 2


def test_an_unparseable_content_length_surfaces_as_an_nflverse_error(tmp_path) -> None:
    """Callers guard fetches with `except NflverseError`; a bare ValueError escapes."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == STAMP_URL:
            return httpx.Response(200, content=b"stamp-1")
        return httpx.Response(200, content=b"abc", headers={"content-length": "seventeen"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with nv.NflverseCache(tmp_path, client=client) as cache:
        with pytest.raises(nv.NflverseError, match="content-length"):
            cache.ensure(ASSET)
        assert not cache.local_path(ASSET).exists()
        assert not cache.local_path(ASSET).with_name("games.parquet.part").exists()


def test_a_server_error_is_not_mistaken_for_a_missing_asset(tmp_path) -> None:
    """A 500 must not read as NflverseNotFound, or `allow_missing` swallows an outage."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == STAMP_URL:
            return httpx.Response(200, content=b"stamp-1")
        return httpx.Response(503, content=b"unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with nv.NflverseCache(tmp_path, client=client) as cache:
        with pytest.raises(nv.NflverseError) as exc:
            cache.ensure(ASSET)
        assert not isinstance(exc.value, nv.NflverseNotFound)


def test_an_unreadable_extension_is_refused(tmp_path) -> None:
    rds = nv.Asset("players", "players.rds")
    with (
        _cache(tmp_path, {STAMP_URL: b"s", rds.url: b"x"}) as cache,
        pytest.raises(nv.NflverseError, match="no reader"),
    ):
        cache.frame(rds)


def test_tag_timestamp_is_polled_once_per_process(tmp_path) -> None:
    seen: list[str] = []
    bodies = {
        f"{nv.RELEASE_BASE}/rosters/timestamp.txt": b"stamp-1",
        nv.Asset("rosters", "roster_2025.parquet").url: b"a",
        nv.Asset("rosters", "roster_2026.parquet").url: b"b",
    }
    with _cache(tmp_path, bodies, seen) as cache:
        cache.ensure(nv.Asset("rosters", "roster_2025.parquet"))
        cache.ensure(nv.Asset("rosters", "roster_2026.parquet"))
    assert seen.count(f"{nv.RELEASE_BASE}/rosters/timestamp.txt") == 1


# --- loader guards --------------------------------------------------------------


def test_depth_charts_refuses_to_span_the_2025_schema_change() -> None:
    with pytest.raises(nv.NflverseError, match="schema"):
        nv.depth_charts([2024, 2026])


def test_nextgen_rejects_an_unknown_stat() -> None:
    with pytest.raises(ValueError, match="unknown stat"):
        nv.nextgen_stats("kicking")


def test_pfr_weekly_rejects_pre_2018_seasons() -> None:
    with pytest.raises(ValueError, match="start in 2018"):
        nv.pfr_advstats([2015], frequency="week")


def test_pfr_weekly_requires_explicit_seasons() -> None:
    with pytest.raises(ValueError, match="explicit seasons"):
        nv.pfr_advstats(frequency="week")


def test_a_string_season_is_rejected_rather_than_shredded_into_digits() -> None:
    """`"2026"` is Iterable, so the generic branch turned it into (2, 0, 2, 6).

    Nothing raised: every downstream `is_in` matched no rows and the caller got a
    silently empty frame instead of a season.
    """
    with pytest.raises(TypeError, match="not the str"):
        nv._season_tuple("2026")
    for call in (
        lambda: nv.schedules("2026"),
        lambda: nv.rosters("2026"),
        lambda: nv.nextgen_stats("receiving", seasons="2026"),
        lambda: nv.pfr_advstats("2026", frequency="season"),
    ):
        with pytest.raises(TypeError, match="not the str"):
            call()


def test_an_empty_season_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="no seasons"):
        nv._season_tuple([])


# --- schedule views -------------------------------------------------------------


def test_bye_weeks_from_a_hand_checked_schedule(schedule: pl.DataFrame) -> None:
    assert nv.bye_weeks(schedule, 2026) == _EXPECTED_BYES


def test_idle_weeks_reports_every_gap(schedule: pl.DataFrame) -> None:
    assert nv.idle_weeks(schedule, 2026) == {t: [w] for t, w in _EXPECTED_BYES.items()}


def test_a_cancelled_game_leaves_two_idle_weeks_and_the_earlier_one_wins(caplog) -> None:
    """The 2022 shape: a game vanishes from the schedule entirely.

    BUF and CIN never replayed Week 17 that year, so nflverse carries 271 games,
    not 272, and both teams look like they have two byes. The later gap is the
    cancellation; the earlier one is the real bye.
    """
    # Weeks 1-4 as above plus a two-game week 5, so cancelling one late game
    # leaves the week itself standing -- exactly the 2022 shape.
    full = [*_GAMES, (5, "AAA", "BBB", 1.0, 43.0), (5, "CCC", "DDD", 2.0, 42.0)]
    assert nv.bye_weeks(_fake_schedule(full), 2026) == _EXPECTED_BYES

    cancelled = _fake_schedule([g for g in full if g != (5, "AAA", "BBB", 1.0, 43.0)])
    assert nv.idle_weeks(cancelled, 2026)["AAA"] == [4, 5]
    assert nv.idle_weeks(cancelled, 2026)["BBB"] == [3, 5]
    with caplog.at_level("WARNING"):
        byes = nv.bye_weeks(cancelled, 2026)
    assert byes == _EXPECTED_BYES
    assert "more than one idle week" in caplog.text


def test_bye_weeks_raises_when_a_team_never_sits(schedule: pl.DataFrame) -> None:
    """No idle week at all means the frame is wrong; there is nothing to guess."""
    packed = pl.concat(
        [
            schedule,
            schedule.head(1).with_columns(
                pl.lit(4, dtype=pl.Int32).alias("week"),
                pl.lit("AAA").alias("home_team"),
                pl.lit("DDD").alias("away_team"),
                pl.lit("2026_04_DDD_AAA").alias("game_id"),
            ),
        ]
    )
    with pytest.raises(nv.NflverseError, match="no bye"):
        nv.bye_weeks(packed, 2026)


def test_bye_weeks_ignores_the_postseason(schedule: pl.DataFrame) -> None:
    """Playoff weeks are numbered 19-22 in the same frame.

    Counting them would stretch the week grid to 19 and hand every team fifteen
    extra "byes".
    """
    playoff = schedule.head(1).with_columns(
        pl.lit("WC").alias("game_type"), pl.lit(19, dtype=pl.Int32).alias("week")
    )
    assert nv.bye_weeks(pl.concat([schedule, playoff]), 2026) == _EXPECTED_BYES


def test_unknown_season_raises_rather_than_returning_empty(schedule: pl.DataFrame) -> None:
    with pytest.raises(nv.NflverseError, match="no REG games"):
        nv.bye_weeks(schedule, 1998)


def test_team_weeks_has_a_row_per_team_week_including_byes(schedule: pl.DataFrame) -> None:
    tw = nv.team_weeks(schedule, 2026)
    assert tw.height == 4 * 4
    assert tw["is_bye"].sum() == 4
    assert tw.filter(pl.col("is_bye"))["opponent"].is_null().all()


def test_team_spread_is_negated_for_the_away_side(schedule: pl.DataFrame) -> None:
    """spread_line is home-relative; a team-oriented view that forgets this
    reports every road favorite as an underdog."""
    tw = nv.team_weeks(schedule, 2026)
    wk1 = tw.filter((pl.col("week") == 1) & pl.col("team").is_in(["AAA", "BBB"]))
    got = dict(zip(wk1["team"].to_list(), wk1["team_spread"].to_list(), strict=True))
    assert got == {"AAA": 3.0, "BBB": -3.0}  # AAA hosts and is favored by 3


def test_implied_totals_split_the_total_by_the_spread(schedule: pl.DataFrame) -> None:
    tw = nv.team_weeks(schedule, 2026).filter((pl.col("week") == 1) & (pl.col("team") == "AAA"))
    row = tw.row(0, named=True)
    # total 44, home favored by 3 -> 23.5 / 20.5
    assert row["team_implied_total"] == pytest.approx(23.5)
    assert row["opponent_implied_total"] == pytest.approx(20.5)
    assert row["team_implied_total"] + row["opponent_implied_total"] == pytest.approx(44.0)


def test_remaining_opponents_is_inclusive_of_from_week(schedule: pl.DataFrame) -> None:
    ros = nv.remaining_opponents(schedule, 2026, from_week=3)
    assert sorted(set(ros["week"].to_list())) == [3, 4]
    no_byes = nv.remaining_opponents(schedule, 2026, from_week=3, include_byes=False)
    assert no_byes["is_bye"].sum() == 0


def test_games_remaining_counts_only_real_games(schedule: pl.DataFrame) -> None:
    assert nv.games_remaining(schedule, 2026, from_week=3) == {
        "AAA": 1,
        "BBB": 1,
        "CCC": 1,
        "DDD": 1,
    }
    assert nv.games_remaining(schedule, 2026, from_week=1) == {
        "AAA": 3,
        "BBB": 3,
        "CCC": 3,
        "DDD": 3,
    }


def test_games_remaining_past_the_end_is_zeros_not_a_missing_team(schedule: pl.DataFrame) -> None:
    """A dropped key reads downstream as an unknown team, not as a finished one."""
    assert nv.games_remaining(schedule, 2026, from_week=99) == dict.fromkeys(_EXPECTED_BYES, 0)


# --- live smoke tests -----------------------------------------------------------


@pytest.fixture(scope="module")
def live(tmp_path_factory) -> Iterator[nv.NflverseCache]:
    cache = nv.NflverseCache(tmp_path_factory.mktemp("nflverse"))
    yield cache
    cache.close()


@pytest.mark.network
def test_live_schedules(live: nv.NflverseCache) -> None:
    df = nv.schedules(cache=live)
    assert df.height > 7000
    assert int(df["season"].min()) == 1999
    for col in ("spread_line", "total_line", "roof", "surface", "espn", "pfr", "game_id"):
        assert col in df.columns
    # The CSV build types these as Int64, which will not join against ESPN.
    assert df.schema["espn"] == pl.String
    assert df.schema["old_game_id"] == pl.String
    # The claim is about the build, not about polars, so read the CSV and contrast.
    csv = live.frame(nv.Asset("schedules", "games.csv"))
    assert csv.height == df.height
    assert csv.schema["espn"] == pl.Int64 != df.schema["espn"]
    # game_types filter, which nothing else covers
    reg = nv.schedules(2026, game_types=["REG"], cache=live)
    assert reg.height == 272
    assert reg["game_type"].unique().to_list() == ["REG"]


@pytest.mark.network
def test_live_2026_byes_span_weeks_5_to_14(live: nv.NflverseCache) -> None:
    byes = nv.bye_weeks(nv.schedules(cache=live), 2026)
    assert len(byes) == 32
    assert min(byes.values()) == 5
    assert max(byes.values()) == 14


@pytest.mark.network
def test_live_every_season_since_1999_yields_a_bye_for_every_team(live: nv.NflverseCache) -> None:
    sched = nv.schedules(cache=live)
    for season in range(1999, 2027):
        byes = nv.bye_weeks(sched, season)
        # 31 teams until Houston arrives in 2002.
        assert len(byes) == (31 if season < 2002 else 32), season


@pytest.mark.network
def test_live_2022_cancelled_game_is_handled(live: nv.NflverseCache) -> None:
    """The Hamlin game: nflverse carries 271 REG games for 2022, not 272.

    BUF and CIN therefore have two idle weeks each and the naive "exactly one
    bye" rule reports Week 17 as their bye.
    """
    sched = nv.schedules(cache=live)
    idle = nv.idle_weeks(sched, 2022)
    assert idle["BUF"] == [7, 17]
    assert idle["CIN"] == [10, 17]
    byes = nv.bye_weeks(sched, 2022)
    assert byes["BUF"] == 7
    assert byes["CIN"] == 10


@pytest.mark.network
def test_live_remaining_opponents(live: nv.NflverseCache) -> None:
    sched = nv.schedules(cache=live)
    ros = nv.remaining_opponents(sched, 2026, from_week=10)
    assert set(ros["team"].to_list()) == set(nv.bye_weeks(sched, 2026))
    # Weeks 10-18 inclusive, one row per team per week.
    assert ros.height == 32 * 9
    # Every team's remaining opponents are themselves the other 31 teams.
    assert set(ros["opponent"].drop_nulls().to_list()) <= set(ros["team"].to_list())


@pytest.mark.network
def test_live_team_view_is_reconstructed_from_the_raw_home_relative_schedule(
    live: nv.NflverseCache,
) -> None:
    """Rebuild every team-side column from `games.parquet` independently.

    Checking `team_implied + opponent_implied == total_line` proves nothing: it is
    an algebraic identity of the two expressions the module writes, and it holds
    just as well with the spread sign inverted. The only real check joins back to
    the raw home-relative row and recomputes.
    """
    sched = nv.schedules(cache=live)
    tw = nv.team_weeks(sched, 2025, include_byes=False)
    raw = sched.filter((pl.col("season") == 2025) & (pl.col("game_type") == "REG")).select(
        "game_id", "home_team", "spread_line", "total_line", "home_rest", "away_rest"
    )
    j = tw.join(raw, on="game_id", how="inner", suffix="_raw")
    assert j.height == tw.height == 544

    home = pl.col("team") == pl.col("home_team")
    expected_spread = (
        pl.when(home).then(pl.col("spread_line_raw")).otherwise(-pl.col("spread_line_raw"))
    )
    expected_rest = pl.when(home).then(pl.col("home_rest")).otherwise(pl.col("away_rest"))
    half = pl.col("total_line_raw") / 2
    bad = j.filter(
        (pl.col("is_home") != home)
        | (pl.col("team_spread") != expected_spread)
        | (pl.col("team_rest") != expected_rest)
        | ((pl.col("team_implied_total") - (half + expected_spread / 2)).abs() > 1e-9)
        | ((pl.col("opponent_implied_total") - (half - expected_spread / 2)).abs() > 1e-9)
    )
    assert bad.height == 0, bad.head(3).to_dicts()


@pytest.mark.network
def test_live_team_spread_points_the_right_way_against_realized_margins(
    live: nv.NflverseCache,
) -> None:
    """The sign convention, checked against outcomes rather than against itself.

    A flipped `team_spread` would still satisfy every internal identity, so pin it
    on 2025 results: favorites must actually win by more.
    """
    sched = nv.schedules(cache=live)
    reg = sched.filter((pl.col("season") == 2025) & (pl.col("game_type") == "REG"))
    margins = pl.concat(
        [
            reg.select(
                "game_id",
                pl.col("home_team").alias("team"),
                (pl.col("home_score") - pl.col("away_score")).alias("margin"),
            ),
            reg.select(
                "game_id",
                pl.col("away_team").alias("team"),
                (pl.col("away_score") - pl.col("home_score")).alias("margin"),
            ),
        ]
    )
    j = (
        nv.team_weeks(sched, 2025, include_byes=False)
        .join(margins, on=["game_id", "team"], how="inner")
        .drop_nulls(["team_spread", "margin"])
    )
    assert j.height > 500
    assert j.select(pl.corr("team_spread", "margin")).item() > 0.3
    assert j.filter(pl.col("team_spread") > 3)["margin"].mean() > 3
    assert j.filter(pl.col("team_spread") < -3)["margin"].mean() < -3


@pytest.mark.network
def test_live_player_week_stats(live: nv.NflverseCache) -> None:
    df = nv.player_week_stats(2025, cache=live)
    assert df.height > 15000
    for col in ("player_id", "position", "team", "opponent_team", "week", "targets", "wopr"):
        assert col in df.columns
    assert set(df["season_type"].unique().to_list()) == {"REG"}


@pytest.mark.network
def test_live_unbuilt_season_raises_not_found_not_an_empty_frame(live: nv.NflverseCache) -> None:
    """nflverse builds a season's weekly file only once games have been played.

    The in-season pipeline hits this every preseason, and an empty frame would be
    read downstream as "nobody scored".
    """
    with pytest.raises(nv.NflverseNotFound):
        nv.player_week_stats(2035, cache=live)
    # allow_missing tolerates a gap, but not a request that yields nothing at all.
    with pytest.raises(nv.NflverseNotFound):
        nv.player_week_stats([2035, 2036], allow_missing=True, cache=live)
    mixed = nv.player_week_stats([2025, 2035], allow_missing=True, cache=live)
    assert set(mixed["season"].unique().to_list()) == {2025}


@pytest.mark.network
def test_live_rosters_carry_the_espn_crosswalk(live: nv.NflverseCache) -> None:
    df = nv.rosters(2026, cache=live)
    assert df.height > 2000
    assert df.schema["espn_id"] == pl.String
    skill = df.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"]))
    coverage = 1 - skill.select(pl.col("espn_id").is_null().mean()).item()
    assert coverage > 0.80, f"ESPN id coverage fell to {coverage:.1%}"


@pytest.mark.network
def test_live_snap_counts(live: nv.NflverseCache) -> None:
    df = nv.snap_counts(2025, cache=live)
    assert df.height > 20000
    for col in ("game_id", "player", "pfr_player_id", "offense_snaps", "offense_pct"):
        assert col in df.columns


@pytest.mark.network
def test_live_depth_charts_dedupe_to_one_snapshot(live: nv.NflverseCache) -> None:
    raw = nv.depth_charts(2026, latest_only=False, cache=live)
    latest = nv.depth_charts(2026, latest_only=True, cache=live)
    assert raw.height > 100_000
    assert raw["dt"].n_unique() > 100
    assert latest.height < raw.height
    # `<= 32` was unfalsifiable: the max-per-team filter guarantees it. What the
    # module actually claims is that every team shares one snapshot.
    assert latest["dt"].n_unique() == 1
    assert latest["dt"].item(0) == raw["dt"].max()
    assert latest["team"].n_unique() == 32
    # Dedupe is per snapshot, not per player: multi-slot players survive.
    assert latest.height > latest["gsis_id"].n_unique()
    # ...and every retained row really is from the newest snapshot for its team.
    assert (
        latest.join(raw.group_by("team").agg(pl.col("dt").max().alias("newest")), on="team")
        .filter(pl.col("dt") != pl.col("newest"))
        .is_empty()
    )


@pytest.mark.network
def test_live_legacy_depth_charts_keep_their_own_schema(live: nv.NflverseCache) -> None:
    """`latest_only` is a documented no-op before 2025; prove it does not drop rows."""
    raw = nv.depth_charts(2024, latest_only=False, cache=live)
    same = nv.depth_charts(2024, latest_only=True, cache=live)
    assert "dt" not in raw.columns
    assert {"club_code", "week", "depth_team", "position"} <= set(raw.columns)
    assert same.height == raw.height > 10_000


@pytest.mark.network
def test_live_players_dictionary(live: nv.NflverseCache) -> None:
    """`players()` had no coverage at all; it is the fallback ID spine."""
    df = nv.players(cache=live)
    assert df.height > 20_000
    for col in ("gsis_id", "espn_id", "pfr_id", "display_name", "position"):
        assert col in df.columns
    assert df.schema["espn_id"] == pl.String
    assert df.filter(pl.col("espn_id") == "NA").is_empty()  # nflverse nulls, not R's "NA"


@pytest.mark.network
def test_live_nextgen_consolidated_beats_the_per_season_stubs(live: nv.NflverseCache) -> None:
    df = nv.nextgen_stats("receiving", cache=live)
    assert df.height > 10000
    for col in ("avg_separation", "avg_cushion", "avg_yac_above_expectation", "targets"):
        assert col in df.columns
    assert nv.nextgen_stats("receiving", seasons=2024, cache=live).height < df.height

    # week 0 is a season aggregate that repeats the weekly rows -- summing the raw
    # frame per player doubles everyone. Documented; pinned here.
    chase = df.filter(
        (pl.col("season") == 2024) & (pl.col("player_display_name") == "Ja'Marr Chase")
    )
    assert chase.filter(pl.col("week") == 0)["targets"].item() == (
        chase.filter(pl.col("week") > 0)["targets"].sum()
    )

    # The trap this loader exists to avoid: the per-season file parses fine and
    # looks like data, but 2024 was abandoned after the opener.
    stub = live.frame(nv.Asset("nextgen_stats", "ngs_2024_receiving.csv.gz"))
    real_2024 = df.filter(pl.col("season") == 2024).height
    assert stub.height < 50 < real_2024
    # And 2025 onward has no per-season file at all.
    with pytest.raises(nv.NflverseNotFound):
        live.frame(nv.Asset("nextgen_stats", "ngs_2025_receiving.csv.gz"))


@pytest.mark.network
def test_live_pfr_advstats(live: nv.NflverseCache) -> None:
    weekly = nv.pfr_advstats([2025], stat="rec", frequency="week", cache=live)
    assert weekly.height > 4000
    assert "receiving_drop_pct" in weekly.columns
    seasonal = nv.pfr_advstats(stat="rec", frequency="season", cache=live)
    assert seasonal.height > 3000
    one = nv.pfr_advstats(2024, stat="rec", frequency="season", cache=live)
    assert 0 < one.height < seasonal.height
    assert one["season"].unique().to_list() == [2024]


@pytest.mark.network
def test_live_unchanged_release_is_not_redownloaded(live: nv.NflverseCache) -> None:
    """A second process polls timestamp.txt and leaves the 519 KB Parquet alone."""
    before = live.ensure(nv.SCHEDULES).stat().st_mtime_ns
    with nv.NflverseCache(live.root, poll_after=dt.timedelta(0)) as fresh:
        assert fresh.ensure(nv.SCHEDULES).stat().st_mtime_ns == before
