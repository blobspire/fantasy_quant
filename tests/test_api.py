"""API-layer tests. Nothing here touches ESPN and nothing here runs a simulation.

The API computes no analysis, so testing it against the real pipeline would test the
pipeline -- which has 1,749 tests of its own -- and would take minutes to say nothing
about the transport. Instead every `report.*_payload` is replaced by a synthetic builder
that records how often it was called, which turns each property this layer is actually
responsible for into a direct assertion:

*The shape.* Every endpoint returns one envelope -- `ok`, `computed_at`, `stale_seconds`,
`data`, `error` -- so the dashboard has one thing to render and one thing to check.

*The cache.* A second call returns byte-identical data and does not call the builder
again; `POST /api/refresh` makes the next call recompute *and drops the workspace*,
because a refresh that kept the built `LeagueSim` would recompute the same numbers off
the same tensor and present them as new.

*The degradation.* A league that raises comes back as `ok: false` with the exception type
intact, with HTTP 200, while the other two leagues still return their numbers. This is
the property the whole error path exists for and it is asserted at the HTTP level.

*The credentials.* `test_no_credentials_in_any_response` plants sentinel values in the
environment, plants them *inside a payload* to simulate a leak from an exception message,
and greps every response body for them. It fails without printing the value it found.

*The bind.* Loopback only, CORS loopback only, no wildcard.

`edges/portfolio` is exercised through real dataclasses -- a real `ActionQueue` of real
`QueueItem`s over real `Recommendation`s -- with synthetic numbers, so the serialisers are
checked against the shapes the module actually produces rather than against a mock of
them. Anything that would hit live ESPN is marked `network`.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fantasy_quant import registry as R
from fantasy_quant import report
from fantasy_quant.api import cache as C
from fantasy_quant.api import server as S
from fantasy_quant.core import Move, MoveKind, PlayerMove, Recommendation

# The three leagues in the fixture registry. `BROKEN` is the one every degradation test
# points at; it is a real id shape so nothing can pass by accident on a sentinel.
WINE, BADDIES, BROKEN = 272150391, 161496047, 634537479

SURFACES = ("odds", "weekly", "waivers", "trades", "lineup", "stream", "roster")


# --------------------------------------------------------------------------------------
# Fixtures: a synthetic registry, a fake workspace, fake payload builders
# --------------------------------------------------------------------------------------


def _registry() -> R.Registry:
    leagues = {
        (WINE, 2026): R.LeagueConfig(
            league_id=WINE, season=2026, name="Wine Wednesday", team_id=1, scoring_variant="ppr"
        ),
        (BADDIES, 2026): R.LeagueConfig(
            league_id=BADDIES,
            season=2026,
            name="Blacksburg Baddies",
            team_id=1,
            scoring_variant="half_ppr",
        ),
        (BROKEN, 2026): R.LeagueConfig(
            league_id=BROKEN,
            season=2026,
            name="Type shi season 2",
            team_id=2,
            scoring_variant="half_ppr",
        ),
    }
    return R.Registry(defaults=R.RegistryDefaults(season=2026), leagues=leagues)


class FakeWorkspace:
    """Stands in for `report.Workspace`: counts builds, closes like the real one."""

    def __init__(self, cfg: R.LeagueConfig, sims: int) -> None:
        self.cfg = cfg
        self.n_sims = sims
        self.closed = False
        self.client = SimpleNamespace(authenticated=True)

    def close(self) -> None:
        self.closed = True


class Recorder:
    """Synthetic `report.*_payload` builders that record their calls."""

    def __init__(self, *, fail: set[int] | None = None, delay: dict[int, float] | None = None):
        self.calls: list[tuple[str, int]] = []
        self.fail = fail or set()
        self.delay = delay or {}
        self.extra: dict[str, Any] = {}

    def build(self, surface: str):
        def _fn(ws: Any, cfg: R.LeagueConfig, **kw: Any) -> dict[str, Any]:
            self.calls.append((surface, cfg.league_id))
            if cfg.league_id in self.delay:
                time.sleep(self.delay[cfg.league_id])
            if cfg.league_id in self.fail:
                raise RuntimeError(f"401 from ESPN for league {cfg.league_id}")
            return {
                "league_id": cfg.league_id,
                "season": cfg.season,
                "name": cfg.name,
                "surface": surface,
                "n_sims": ws.n_sims,
                "kwargs": {k: v for k, v in kw.items() if v is not None},
                **self.extra,
            }

        return _fn

    def count(self, surface: str, league_id: int) -> int:
        return sum(1 for s, lid in self.calls if s == surface and lid == league_id)


def _install(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> None:
    """Replace every real payload builder, including the one this module owns."""
    monkeypatch.setattr(report, "odds_payload", rec.build("odds"))
    monkeypatch.setattr(report, "weekly_payload", rec.build("weekly"))
    monkeypatch.setattr(report, "waivers_payload", rec.build("waivers"))
    monkeypatch.setattr(report, "trades_payload", rec.build("trades"))
    monkeypatch.setattr(report, "lineup_payload", rec.build("lineup"))
    monkeypatch.setattr(report, "stream_payload", rec.build("stream"))
    monkeypatch.setattr(S, "roster_payload", rec.build("roster"))
    monkeypatch.setattr(
        S,
        "_probe_league",
        lambda eng, cfg, sims: {"reachable": True, "error": None, "espn_name": cfg.name},
    )
    monkeypatch.setattr(
        S,
        "_auth_probe",
        lambda eng, sims: {
            "credentials": {"swid": "set", "espn_s2": "set", "complete": True},
            "espn_reachable": True,
            "authenticated": True,
            "season": 2026,
            "week": 1,
        },
    )


def _engine(monkeypatch: pytest.MonkeyPatch, rec: Recorder, **kw: Any) -> S.Engine:
    _install(monkeypatch, rec)
    settings = S.Settings(web_dir=Path("/nonexistent-web-dist"), **kw)
    return S.Engine(
        settings=settings,
        registry_loader=_registry,
        workspace_factory=lambda cfg, sims: FakeWorkspace(cfg, sims),
    )


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> S.Engine:
    return _engine(monkeypatch, rec)


@pytest.fixture
def client(engine: S.Engine) -> TestClient:
    return TestClient(S.create_app(engine.settings, engine=engine))


def _url(surface: str, league_id: int = WINE) -> str:
    return f"/api/leagues/{league_id}/{surface}"


# --------------------------------------------------------------------------------------
# The envelope every endpoint shares
# --------------------------------------------------------------------------------------

ENVELOPE_KEYS = {
    "ok",
    "endpoint",
    "league_id",
    "season",
    "sims",
    "schema_version",
    "cached",
    "computed_at",
    "stale_seconds",
    "compute_seconds",
    "data",
    "error",
}


@pytest.mark.parametrize("surface", SURFACES)
def test_every_surface_returns_the_documented_envelope(client: TestClient, surface: str) -> None:
    body = client.get(_url(surface)).json()
    assert set(body) >= ENVELOPE_KEYS, f"{surface} is missing {ENVELOPE_KEYS - set(body)}"
    assert body["ok"] is True
    assert body["error"] is None
    assert body["endpoint"] == surface
    assert body["league_id"] == WINE
    assert body["season"] == 2026
    assert body["schema_version"] == report.SCHEMA_VERSION
    assert body["data"]["surface"] == surface
    assert body["stale_seconds"] >= 0.0
    assert body["computed_at"].endswith("+00:00")


@pytest.mark.parametrize("surface", SURFACES)
def test_every_surface_degrades_to_a_structured_error(
    monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """A 401 on one league is a row, not a page: HTTP 200, `ok: false`, type intact."""
    rec = Recorder(fail={BROKEN})
    eng = _engine(monkeypatch, rec)
    client = TestClient(S.create_app(eng.settings, engine=eng))

    bad = client.get(_url(surface, BROKEN))
    assert bad.status_code == 200
    body = bad.json()
    assert body["ok"] is False
    assert body["data"] is None
    assert body["error"]["type"] == "RuntimeError"
    assert "401 from ESPN" in body["error"]["message"]
    assert body["error"]["league_id"] == BROKEN

    # ... and the other two leagues are untouched.
    for league_id in (WINE, BADDIES):
        good = client.get(_url(surface, league_id)).json()
        assert good["ok"] is True, f"{surface} for {league_id} died with {BROKEN}"
        assert good["data"]["league_id"] == league_id


def test_unknown_league_is_404_not_a_structured_error(client: TestClient) -> None:
    """A league not in the registry is a client mistake, not a degraded league."""
    r = client.get(_url("odds", 12345))
    assert r.status_code == 404
    assert "not in the registry" in r.json()["detail"]


def test_unknown_weekly_section_is_rejected(client: TestClient) -> None:
    r = client.get(_url("weekly"), params={"skip": ["oddz"]})
    assert r.status_code == 422
    assert "oddz" in json.dumps(r.json())


def test_request_parameters_reach_the_builder(client: TestClient, rec: Recorder) -> None:
    body = client.get(_url("waivers"), params={"limit": 12, "week": 3}).json()
    assert body["data"]["kwargs"] == {"limit": 12, "week": 3}
    body = client.get(_url("trades"), params={"limit": 2, "min_gain": 0.5}).json()
    assert body["data"]["kwargs"] == {"limit": 2, "min_gain": 0.5}


def test_sims_is_clamped_and_reaches_the_workspace(client: TestClient) -> None:
    """A URL cannot ask for a ten-minute run, and cannot ask for a meaningless one."""
    assert client.get(_url("odds"), params={"sims": 10}).json()["sims"] == S.MIN_SIMS
    assert client.get(_url("odds"), params={"sims": 10**9}).json()["sims"] == S.MAX_SIMS
    body = client.get(_url("odds"), params={"sims": 1000}).json()
    assert body["sims"] == 1000 and body["data"]["n_sims"] == 1000


# --------------------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------------------


def test_warm_call_is_identical_and_does_not_recompute(client: TestClient, rec: Recorder) -> None:
    first = client.get(_url("odds")).json()
    second = client.get(_url("odds")).json()

    assert first["cached"] is False and second["cached"] is True
    assert rec.count("odds", WINE) == 1
    # Identical bytes, not merely equal values: the data is served from the cache and
    # nothing in the payload is regenerated per request.
    assert json.dumps(first["data"], sort_keys=True) == json.dumps(second["data"], sort_keys=True)
    assert first["computed_at"] == second["computed_at"]
    assert second["stale_seconds"] >= first["stale_seconds"]


def test_refresh_invalidates_and_recomputes(client: TestClient, rec: Recorder) -> None:
    before = client.get(_url("odds")).json()
    client.get(_url("odds"))
    assert rec.count("odds", WINE) == 1

    r = client.post("/api/refresh")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["invalidated"]["entries"] >= 1

    after = client.get(_url("odds")).json()
    assert rec.count("odds", WINE) == 2
    assert after["cached"] is False
    assert after["computed_at"] != before["computed_at"]


def test_refresh_drops_the_workspace_so_the_simulation_is_rebuilt(
    client: TestClient, engine: S.Engine
) -> None:
    """The refresh that matters: the built `LeagueSim` goes, not just the payload."""
    client.get(_url("odds"))
    ws = engine.workspace(engine.config(WINE), engine.settings.sims)
    client.post("/api/refresh", params={"league_id": WINE})
    assert ws.closed is True
    assert engine.workspace(engine.config(WINE), engine.settings.sims) is not ws


def test_refresh_of_one_league_leaves_the_others_warm(client: TestClient, rec: Recorder) -> None:
    client.get(_url("odds", WINE))
    client.get(_url("odds", BADDIES))
    client.post("/api/refresh", params={"league_id": WINE})
    client.get(_url("odds", WINE))
    client.get(_url("odds", BADDIES))
    assert rec.count("odds", WINE) == 2
    assert rec.count("odds", BADDIES) == 1


def test_refresh_of_an_unknown_league_is_404(client: TestClient) -> None:
    assert client.post("/api/refresh", params={"league_id": 999}).status_code == 404


def test_different_parameters_are_different_answers(client: TestClient, rec: Recorder) -> None:
    """A six-row waiver board must not be served to a request that asked for twenty."""
    client.get(_url("waivers"), params={"limit": 6})
    client.get(_url("waivers"), params={"limit": 20})
    client.get(_url("waivers"), params={"limit": 6})
    assert rec.count("waivers", WINE) == 2


def test_a_failure_is_retried_once_its_backoff_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 stands for `error_ttl` and is then retried; a success never expires."""
    now = [1000.0]
    rec = Recorder(fail={BROKEN})
    eng = _engine(monkeypatch, rec, error_ttl=30.0)
    eng.cache.clock = lambda: now[0]
    client = TestClient(S.create_app(eng.settings, engine=eng))

    assert client.get(_url("odds", BROKEN)).json()["ok"] is False
    client.get(_url("odds", BROKEN))
    assert rec.count("odds", BROKEN) == 1, "a failure inside its backoff must not re-run"

    now[0] += 31.0
    client.get(_url("odds", BROKEN))
    assert rec.count("odds", BROKEN) == 2

    now[0] += 10_000.0
    client.get(_url("odds", WINE))
    client.get(_url("odds", WINE))
    assert rec.count("odds", WINE) == 1, "a successful analysis must never expire on its own"


def test_staleness_is_reported_in_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [500.0]
    rec = Recorder()
    eng = _engine(monkeypatch, rec)
    eng.cache.clock = lambda: now[0]
    client = TestClient(S.create_app(eng.settings, engine=eng))

    assert client.get(_url("odds")).json()["stale_seconds"] == 0.0
    now[0] += 247.0
    body = client.get(_url("odds")).json()
    assert body["stale_seconds"] == 247.0
    assert body["cached"] is True


# --------------------------------------------------------------------------------------
# The cache module on its own
# --------------------------------------------------------------------------------------


def test_cache_key_drops_none_parameters() -> None:
    """`week=None` and an omitted `week` are the same question and must share a key."""
    assert C.Key.of("waivers", league_id=1, week=None) == C.Key.of("waivers", league_id=1)
    assert C.Key.of("waivers", league_id=1, week=3) != C.Key.of("waivers", league_id=1)
    assert C.Key.of("waivers", limit=True).params == (("limit", "true"),)
    assert C.Key.of("weekly", skip=["trades", "odds"]).params == (("skip", "trades,odds"),)


def test_cache_returns_the_same_object_and_ages_it() -> None:
    now = [0.0]
    cache = C.Cache(clock=lambda: now[0])
    value = {"a": 1}
    entry = cache.put(C.Key.of("odds", league_id=1), value)
    now[0] = 12.5
    again = cache.peek(C.Key.of("odds", league_id=1))
    assert again is entry
    assert again.value is value
    assert again.staleness(clock=cache.clock)["stale_seconds"] == 12.5


def test_cache_invalidate_by_league_takes_the_cross_league_rows_with_it() -> None:
    """A queue that ranks a league the user just refreshed must not survive the refresh."""
    cache = C.Cache()
    cache.put(C.Key.of("odds", league_id=1), 1)
    cache.put(C.Key.of("odds", league_id=2), 2)
    cache.put(C.Key.of("queue"), 3)
    assert cache.invalidate(league_id=1) == 2
    assert cache.peek(C.Key.of("odds", league_id=2)) is not None
    assert cache.peek(C.Key.of("queue")) is None


def test_cache_max_age_expires_only_when_asked() -> None:
    now = [0.0]
    cache = C.Cache(clock=lambda: now[0])
    key = C.Key.of("health")
    cache.put(key, {"ok": True})
    now[0] = 99.0
    assert cache.peek(key) is not None
    assert cache.peek(key, max_age=50.0) is None
    assert cache.peek(key, max_age=100.0) is not None


def test_cache_stats_never_include_values() -> None:
    cache = C.Cache()
    cache.put(C.Key.of("odds", league_id=1), {"secret-ish": "value"})
    assert "secret-ish" not in json.dumps(cache.stats())


# --------------------------------------------------------------------------------------
# The event loop, and one league not freezing another
# --------------------------------------------------------------------------------------


def _asgi_client(app: Any) -> Any:
    import httpx

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_a_slow_league_does_not_block_a_fast_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU-bound build runs off the event loop, so a cheap answer overtakes it."""
    rec = Recorder(delay={BROKEN: 0.6})
    eng = _engine(monkeypatch, rec)
    app = S.create_app(eng.settings, engine=eng)

    async def _run() -> tuple[float, float]:
        async with _asgi_client(app) as http:
            started = time.perf_counter()
            done: dict[str, float] = {}

            async def _get(surface: str, league_id: int, tag: str) -> None:
                await http.get(_url(surface, league_id))
                done[tag] = time.perf_counter() - started

            await asyncio.gather(_get("odds", BROKEN, "slow"), _get("odds", WINE, "fast"))
            return done["fast"], done["slow"]

    fast, slow = asyncio.run(_run())
    assert slow >= 0.6
    assert fast < slow / 2, f"the fast league waited {fast:.3f}s behind the slow one"


def test_two_simultaneous_cold_requests_compute_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-flight: two tabs on one league must not both mutate one Workspace."""
    rec = Recorder(delay={WINE: 0.25})
    eng = _engine(monkeypatch, rec)
    app = S.create_app(eng.settings, engine=eng)

    async def _run() -> list[dict[str, Any]]:
        async with _asgi_client(app) as http:
            responses = await asyncio.gather(
                http.get(_url("odds")), http.get(_url("odds")), http.get(_url("odds"))
            )
            return [r.json() for r in responses]

    bodies = asyncio.run(_run())
    assert rec.count("odds", WINE) == 1
    assert all(b["ok"] for b in bodies)
    assert len({b["computed_at"] for b in bodies}) == 1


def test_a_refresh_mid_build_does_not_cache_the_answer_it_invalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh during a slow build closes the client under it. The failure must not stand.

    `POST /api/refresh` discards the workspace, so a build already reading through that
    workspace's client raises. Caching that failure would leave the refresh button
    looking broken for `error_ttl` seconds -- the opposite of what it does -- so the
    request that was in flight gets its answer and the next one recomputes.
    """
    rec = Recorder(delay={WINE: 0.4})
    eng = _engine(monkeypatch, rec)
    app = S.create_app(eng.settings, engine=eng)

    async def _run() -> dict[str, Any]:
        async with _asgi_client(app) as http:
            slow = asyncio.ensure_future(http.get(_url("odds")))
            await asyncio.sleep(0.15)
            await http.post("/api/refresh")
            return (await slow).json()

    body = asyncio.run(_run())
    assert body["ok"] is True
    assert (
        eng.cache.peek(C.Key.of("odds", league_id=WINE, season=2026, sims=eng.settings.sims))
        is None
    )


# --------------------------------------------------------------------------------------
# Cross-league: portfolio and queue, against the real dataclasses
# --------------------------------------------------------------------------------------


def _rec(delta: float, stderr: float, *, kind: MoveKind = MoveKind.WAIVER_CLAIM, **kw: Any):
    return Recommendation(
        move=Move(kind=kind, league_id=WINE, players=(PlayerMove(1, None, 1),)),
        delta_title=delta,
        delta_points=1.5,
        stderr=stderr,
        leverage=0.8,
        rationale="Claim Somebody. He is the best free agent on the wire.",
        **kw,
    )


def _fake_queue(portfolio_mod: Any) -> Any:
    """A real `ActionQueue` of real `QueueItem`s. Synthetic numbers, genuine shapes."""
    items = (
        portfolio_mod.QueueItem(
            league_id=WINE,
            league_name="Wine Wednesday",
            team_id=1,
            surface="waivers",
            rec=_rec(0.012, 0.002),
            baseline_title=0.022,
            n_considered=12,
        ),
        portfolio_mod.QueueItem(
            league_id=BADDIES,
            league_name="Blacksburg Baddies",
            team_id=1,
            surface="trades",
            rec=_rec(0.011, 0.005, kind=MoveKind.TRADE, confidence="low"),
            baseline_title=0.051,
            n_considered=40,
        ),
    )
    return portfolio_mod.ActionQueue(
        items=items,
        limit=20,
        leverage_weight=0.0,
        failures=(("Type shi season 2", "trades", "EspnError: 401"),),
        ranked=items,
    )


def _fake_book() -> Any:
    stake = SimpleNamespace(
        league_id=WINE,
        name="Wine Wednesday",
        team_id=1,
        team_name="Cole",
        title=0.022,
        n_sims=4000,
        state=SimpleNamespace(season=2026),
    )
    return SimpleNamespace(stakes=(stake,), seed=1, n_sims=4000)


def _fake_report(portfolio_mod: Any, queue: Any) -> Any:
    P = portfolio_mod
    odds = P.PortfolioOdds(
        names=("Wine Wednesday",),
        titles=(0.022,),
        p_at_least_one=0.15,
        p_zero=0.85,
        p_two_plus=0.01,
        expected_titles=0.156,
        variance_titles=0.14,
        independent=0.152,
        sum_bound=0.156,
        max_bound=0.083,
        stderr=0.0056,
        dependence_cost=0.002,
        dependence_cost_stderr=0.0009,
        n_sims=4000,
    )
    exposure = P.Exposure(
        player_id=4262921,
        name="Somebody",
        position_id=3,
        pro_team_id=12,
        holdings=(
            P.Holding(
                league_id=WINE,
                league_name="Wine Wednesday",
                team_id=1,
                slot_id=2,
                start_share=0.9,
                title_added=0.004,
                title_added_stderr=0.001,
            ),
        ),
        equity_at_risk=0.004,
        equity_share=0.03,
        portfolio_damage=0.003,
        portfolio_damage_stderr=0.0008,
    )
    conc = P.Concentration(
        kind="player",
        label="Somebody",
        removed={"Wine Wednesday": ("Somebody",)},
        before=0.15,
        after=0.147,
        stderr=0.0008,
        expected_before=0.156,
        expected_after=0.152,
    )
    bye = P.ByeExposure(
        week=7,
        starters_out={"Wine Wednesday": ("Somebody",)},
        normal_points={"Wine Wednesday": 11.2},
        bye_points={"Wine Wednesday": 0.1},
        weekly_mean={"Wine Wednesday": 118.0},
        unpriced=(("Wine Wednesday", "Jaguars D/ST", 3.28, 4.71),),
    )
    corr = P.Correlations(
        pairs=(
            P.PairCorrelation(
                a="Wine Wednesday",
                b="Blacksburg Baddies",
                shared_players=("Somebody",),
                weekly=0.11,
                season=0.09,
                champion=-0.01,
                champion_stderr=0.02,
            ),
        ),
        by_week={9: 0.12, 11: 0.115},
        worst_week_gap_stderr=0.03,
    )
    return P.PortfolioReport(
        odds=odds,
        coupling=(
            P.CouplingCheck(
                a="Wine Wednesday",
                b="Blacksburg Baddies",
                shared_pool=180,
                median_player_correlation=0.98,
                min_player_correlation=0.81,
                availability_match=1.0,
            ),
        ),
        correlations=corr,
        exposures=(exposure,),
        concentration=(conc,),
        pro_teams=(conc,),
        byes=(bye,),
        diversification=P.diversification(SimpleNamespace(odds=lambda: odds), odds=odds),
        queue=queue,
    )


@pytest.fixture
def cross_client(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> TestClient:
    from fantasy_quant.edges import portfolio as portfolio_mod

    eng = _engine(monkeypatch, rec)
    queue = _fake_queue(portfolio_mod)
    book = _fake_book()
    monkeypatch.setattr(S.Engine, "book", lambda self, sims: book)
    monkeypatch.setattr(S.Engine, "queue", lambda self, sims, *, limit, actionable_only: queue)
    monkeypatch.setattr(
        portfolio_mod, "report", lambda b, **kw: _fake_report(portfolio_mod, kw.get("queue"))
    )
    return TestClient(S.create_app(eng.settings, engine=eng))


def test_queue_endpoint_has_the_cli_queue_shape(cross_client: TestClient) -> None:
    body = cross_client.get("/api/queue").json()
    assert body["ok"] is True and body["league_id"] is None
    data = body["data"]
    assert data["command"] == "queue"
    assert data["schema_version"] == report.SCHEMA_VERSION
    assert data["ranked_by"].startswith("edges.portfolio.")
    assert data["n_actions"] == 2
    assert [a["surface"] for a in data["actions"]] == ["waivers", "trades"]
    assert data["errors"] == [
        {"league": "Type shi season 2", "surface": "trades", "error": "EspnError: 401"}
    ]
    # The queue is the surface the whole system is ranked in, so `delta_title` and the
    # uncertainty beside it have to survive the wire.
    top = data["actions"][0]
    assert top["delta_title"] == pytest.approx(0.012)
    assert top["stderr"] == pytest.approx(0.002)
    assert top["z"] == pytest.approx(6.0)
    assert isinstance(top["significant"], bool)


def test_queue_marks_a_search_winner_as_not_significant(cross_client: TestClient) -> None:
    """The selection correction survives serialisation, which is the point of reusing it.

    The trade row is +1.10pp +/- 0.50 -- 2.2 sigma, "significant" on a naive test -- and
    it won a search of forty. `report.verdict_for` calls it noise and the API must ship
    that verdict rather than a recomputed one.
    """
    actions = cross_client.get("/api/queue").json()["data"]["actions"]
    trade = next(a for a in actions if a["surface"] == "trades")
    assert trade["significant"] is False
    assert trade["verdict"] in {"noise", "null", "harm"}
    assert any("best of 40" in c for c in trade["caveats"])


def test_portfolio_endpoint_serialises_the_whole_picture(cross_client: TestClient) -> None:
    body = cross_client.get("/api/portfolio").json()
    assert body["ok"] is True
    data = body["data"]
    assert set(data) >= {
        "odds",
        "coupling",
        "correlations",
        "exposures",
        "concentration",
        "pro_teams",
        "byes",
        "diversification",
        "leagues",
    }
    assert data["odds"]["p_at_least_one"] == pytest.approx(0.15)
    assert data["odds"]["expected_titles"] == pytest.approx(0.156)
    assert data["odds"]["dependence_significant"] is True
    exposure = data["exposures"][0]
    assert exposure["name"] == "Somebody"
    assert exposure["significant"] is True
    assert exposure["holdings"][0]["slot"] == "RB"
    assert data["concentration"][0]["damage"] == pytest.approx(0.003)
    assert data["byes"][0]["unpriced"][0]["player"] == "Jaguars D/ST"
    assert "E[titles]" in data["diversification"]["verdict"]
    assert "queue" not in data, "the queue is /api/queue's; two copies would drift"


def test_portfolio_and_queue_share_one_draw(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> None:
    """Both endpoints must be built from the same `Portfolio`, or they measure differently."""
    from fantasy_quant.edges import portfolio as portfolio_mod

    eng = _engine(monkeypatch, rec)
    built: list[Any] = []
    queue = _fake_queue(portfolio_mod)
    monkeypatch.setattr(
        portfolio_mod,
        "build_portfolio",
        lambda leagues, season, **kw: built.append((tuple(leagues), season)) or _fake_book(),
    )
    monkeypatch.setattr(portfolio_mod, "action_queue", lambda book, **kw: queue)
    monkeypatch.setattr(
        portfolio_mod, "report", lambda b, **kw: _fake_report(portfolio_mod, kw.get("queue"))
    )
    client = TestClient(S.create_app(eng.settings, engine=eng))
    assert client.get("/api/queue").json()["ok"] is True
    assert client.get("/api/portfolio").json()["ok"] is True
    # One build per (season, sims) -- `Engine.book` memoizes -- so the second endpoint
    # reuses the first one's draw rather than re-simulating three leagues onto a second,
    # incomparable, shared season.
    assert len(built) == 1
    leagues, season = built[0]
    assert season == 2026
    assert sorted(leagues) == [(BADDIES, 1), (WINE, 1), (BROKEN, 2)]


def test_portfolio_and_queue_share_one_draw_when_asked_at_the_same_time(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    """The same invariant, but concurrently -- which is the case that has to be locked.

    `/api/portfolio` and `/api/queue` are two cache keys and therefore two
    `asyncio.Lock`s, and both builders call `Engine.book` from inside a worker thread.
    Nothing about the per-key coroutine lock stops them arriving together, so without a
    lock a *thread* can hold, two tabs opened a moment apart draw two thirty-second
    portfolios through one shared ESPN client.
    """
    from fantasy_quant.edges import portfolio as portfolio_mod

    eng = _engine(monkeypatch, rec)
    built: list[float] = []
    queued: list[float] = []
    queue = _fake_queue(portfolio_mod)

    def _build(leagues: Any, season: int, **kw: Any) -> Any:
        built.append(time.perf_counter())
        time.sleep(0.3)
        return _fake_book()

    def _action_queue(book: Any, **kw: Any) -> Any:
        queued.append(time.perf_counter())
        time.sleep(0.1)
        return queue

    monkeypatch.setattr(portfolio_mod, "build_portfolio", _build)
    monkeypatch.setattr(portfolio_mod, "action_queue", _action_queue)
    monkeypatch.setattr(
        portfolio_mod, "report", lambda b, **kw: _fake_report(portfolio_mod, kw.get("queue"))
    )
    app = S.create_app(eng.settings, engine=eng)

    async def _run() -> list[dict[str, Any]]:
        async with _asgi_client(app) as http:
            responses = await asyncio.gather(http.get("/api/portfolio"), http.get("/api/queue"))
            return [r.json() for r in responses]

    bodies = asyncio.run(_run())
    assert all(b["ok"] for b in bodies)
    assert len(built) == 1, f"build_portfolio ran {len(built)} times for one shared season"
    assert len(queued) == 1, f"action_queue ran {len(queued)} times for one shared season"


def test_queue_degrades_to_the_local_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A portfolio that cannot be built must not take the ranked list down with it."""
    rec = Recorder()
    eng = _engine(monkeypatch, rec)

    def _boom(self: Any, sims: int, *, limit: int, actionable_only: bool) -> Any:
        raise RuntimeError("EspnError: 401 on one league")

    monkeypatch.setattr(S.Engine, "queue", _boom)
    monkeypatch.setattr(
        report, "queue_payload", lambda weekly, **kw: {"command": "queue", "actions": []}
    )
    client = TestClient(S.create_app(eng.settings, engine=eng))
    body = client.get("/api/queue").json()
    assert body["ok"] is True
    assert "401" in body["data"]["degraded"]
    assert rec.count("weekly", WINE) == 1, "the merge falls back to the weekly reports"
    # Same keys either way. The UI must not have to know which path produced a queue
    # before it can read one.
    assert set(body["data"]) >= {"selection_note", "n_significant", "n_actionable"}
    assert body["data"]["n_significant"] == 0 and body["data"]["n_actionable"] == 0


# --------------------------------------------------------------------------------------
# Registry, health
# --------------------------------------------------------------------------------------


def test_leagues_lists_the_registry_with_reachability(client: TestClient) -> None:
    body = client.get("/api/leagues").json()
    assert body["ok"] is True
    assert body["n_leagues"] == 3
    assert body["n_reachable"] == 3
    rows = {r["league_id"]: r for r in body["leagues"]}
    assert set(rows) == {WINE, BADDIES, BROKEN}
    wine = rows[WINE]
    assert wine["name"] == "Wine Wednesday"
    assert wine["team_id"] == 1
    assert wine["scoring_variant"] == "ppr"
    assert wine["reachable"] is True
    assert wine["stale_seconds"] >= 0.0


def test_leagues_reports_an_unreachable_league_without_failing(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    eng = _engine(monkeypatch, rec)

    def _probe(engine: Any, cfg: R.LeagueConfig, sims: int) -> dict[str, Any]:
        if cfg.league_id == BROKEN:
            raise RuntimeError("401 from ESPN: credentials missing or stale")
        return {"reachable": True, "error": None}

    monkeypatch.setattr(S, "_probe_league", _probe)
    client = TestClient(S.create_app(eng.settings, engine=eng))
    body = client.get("/api/leagues").json()
    assert body["n_reachable"] == 2
    broken = next(r for r in body["leagues"] if r["league_id"] == BROKEN)
    assert broken["reachable"] is False
    assert "401" in broken["error"]["message"]


def test_health_reports_liveness_and_auth(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["espn"]["authenticated"] is True
    assert body["espn"]["credentials"] == {"swid": "set", "espn_s2": "set", "complete": True}
    assert body["server"]["host"] == S.HOST
    assert body["leagues"] == 3
    assert "entries" in body["cache"]


def test_health_reports_a_dead_session_without_dying(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    eng = _engine(monkeypatch, rec)
    monkeypatch.setattr(
        S,
        "_auth_probe",
        lambda engine, sims: {
            "credentials": {"swid": "set", "espn_s2": "set", "complete": True},
            "espn_reachable": True,
            "authenticated": False,
            "error": "EspnError: 401 from ESPN: credentials missing or stale",
        },
    )
    body = TestClient(S.create_app(eng.settings, engine=eng)).get("/api/health").json()
    assert body["ok"] is True
    assert body["espn"]["authenticated"] is False
    assert "re-copy" in body["espn"]["error"] or "401" in body["espn"]["error"]


# --------------------------------------------------------------------------------------
# Credentials must never reach the browser
# --------------------------------------------------------------------------------------

SWID_SENTINEL = "{DEADBEEF-1111-2222-3333-444455556666}"
S2_SENTINEL = "AEBxxxxSENTINELespns2cookieVALUExxxx%2Bxxxx"


def test_no_credentials_in_any_response(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> None:
    """Grep every response body for the cookie values. Fails without printing them.

    The payloads are *deliberately poisoned*: the sentinels are planted inside a builder
    result and inside an exception message, which is the realistic leak -- `httpx` will
    put a request URL in an error, and a request URL can carry a cookie. If the scrubber
    is removed this test fails on every endpoint at once.
    """
    monkeypatch.setenv("ESPN_SWID", SWID_SENTINEL)
    monkeypatch.setenv("ESPN_S2", S2_SENTINEL)
    S.set_scrub_secrets(None)
    try:
        secrets = S.load_secrets()
        assert SWID_SENTINEL in secrets and S2_SENTINEL in secrets

        rec.extra = {"leak": f"cookie SWID={SWID_SENTINEL}; espn_s2={S2_SENTINEL}"}
        rec.fail = {BROKEN}

        def _leaky(ws: Any, cfg: R.LeagueConfig, **kw: Any) -> dict[str, Any]:
            raise RuntimeError(f"401 for https://espn.com?swid={SWID_SENTINEL}&s2={S2_SENTINEL}")

        eng = _engine(monkeypatch, rec)
        # After `_engine`, which installs the whole synthetic set over the top.
        monkeypatch.setattr(report, "odds_payload", _leaky)
        client = TestClient(S.create_app(eng.settings, engine=eng))

        paths = ["/api/health", "/api/leagues"]
        paths += [_url(s, lid) for s in SURFACES for lid in (WINE, BROKEN)]
        for path in paths:
            body = client.get(path).text
            assert SWID_SENTINEL not in body, f"SWID leaked into {path}"
            assert S2_SENTINEL not in body, f"espn_s2 leaked into {path}"
            found = _leaks(body, secrets)
            assert found == 0, f"{found} credential value(s) leaked into {path}"
        # ... and the scrubber left something behind so the leak is visible in the logs.
        assert S.REDACTED in client.get(_url("odds")).text
    finally:
        S.set_scrub_secrets(None)


def _leaks(body: str, secrets: Sequence[str]) -> int:
    """How many secrets are in `body`. A COUNT, never the value.

    `assert secret not in body` would put the credential itself into pytest's assertion
    rewriting and from there into the terminal and CI log -- which is the leak the test
    is meant to prevent. Only the integer is ever asserted on.
    """
    return sum(1 for secret in secrets if secret and secret in body)


def test_real_credentials_never_appear_either(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    """The same grep against whatever is actually in `.env`, if anything is."""
    S.set_scrub_secrets(None)
    real = S.load_secrets()
    if not real:
        pytest.skip("no ESPN credentials configured; nothing to leak")
    eng = _engine(monkeypatch, rec)
    client = TestClient(S.create_app(eng.settings, engine=eng))
    for path in ["/api/health", "/api/leagues", _url("odds"), _url("weekly")]:
        found = _leaks(client.get(path).text, real)
        assert found == 0, f"{found} real credential value(s) appeared in {path}"


def test_scrub_replaces_raw_and_percent_encoded_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cookie inside a URL is percent-encoded; scrubbing the raw form alone misses it."""
    monkeypatch.setenv("ESPN_SWID", SWID_SENTINEL)
    monkeypatch.setenv("ESPN_S2", S2_SENTINEL)
    S.set_scrub_secrets(None)
    try:
        encoded = "%7BDEADBEEF-1111-2222-3333-444455556666%7D"
        assert encoded in S.load_secrets()
        assert S.scrub(f"a {SWID_SENTINEL} b") == f"a {S.REDACTED} b"
        assert S.scrub(f"https://espn.com?swid={encoded}") == f"https://espn.com?swid={S.REDACTED}"
    finally:
        S.set_scrub_secrets(None)


def test_scrub_ignores_values_too_short_to_be_secrets() -> None:
    """A two-character "secret" would corrupt every payload it appeared in."""
    S.set_scrub_secrets(["ab"])
    try:
        assert S.scrub("a cabbage") == "a cabbage"
    finally:
        S.set_scrub_secrets(None)


# --------------------------------------------------------------------------------------
# Bind and CORS: this is reachable from this machine and nowhere else
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    assert S._require_loopback(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "10.0.0.1", "example.com", ""])
def test_non_loopback_hosts_are_refused(host: str) -> None:
    with pytest.raises(ValueError, match="refusing to bind"):
        S._require_loopback(host)


def test_default_settings_bind_to_loopback() -> None:
    assert S.Settings().host == S.HOST == "127.0.0.1"
    assert S._require_loopback(S.Settings().host)


@pytest.mark.parametrize("peer", ["127.0.0.1", "127.0.0.5", "::1", "testclient", "", None])
def test_a_peer_on_this_machine_is_allowed(peer: str | None) -> None:
    """Loopback addresses, and the non-address peers an in-process ASGI caller reports."""
    assert S._is_loopback_peer(peer) is True


@pytest.mark.parametrize("peer", ["10.0.0.190", "192.168.1.20", "8.8.8.8", "2001:db8::1"])
def test_a_peer_off_this_machine_is_not(peer: str) -> None:
    assert S._is_loopback_peer(peer) is False


def test_a_request_from_the_network_is_refused_whatever_the_bind_address(
    engine: S.Engine,
) -> None:
    """`_require_loopback` only runs inside `fq dash`.

    `uvicorn fantasy_quant.api.server:app --host 0.0.0.0` is the module's other
    documented entry point and this file never sees its `--host`. Started that way the
    process is an unauthenticated proxy to the user's ESPN account for the whole
    network, so the peer is checked as well as the bind: every endpoint, before CORS,
    before routing, and before anything reads a league.
    """
    app = S.create_app(engine.settings, engine=engine)
    with TestClient(app, client=("10.0.0.190", 51234)) as remote:
        for path in ("/api/health", "/api/leagues", _url("odds"), "/api/queue", "/"):
            r = remote.get(path)
            assert r.status_code == 403, f"{path} answered a non-loopback peer"
            assert r.json()["error"]["type"] == "NotLoopback"
        assert remote.post("/api/refresh").status_code == 403

    with TestClient(app, client=("127.0.0.1", 51234)) as local:
        assert local.get("/api/health").status_code == 200


@pytest.mark.parametrize(
    "origin", ["http://localhost:5173", "http://127.0.0.1:8765", "http://localhost"]
)
def test_cors_allows_local_origins(client: TestClient, origin: str) -> None:
    r = client.get("/api/health", headers={"Origin": origin})
    assert r.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.com",
        "http://localhost.evil.com",
        "http://127.0.0.1.evil.com",
        "https://espn.com",
    ],
)
def test_cors_refuses_everything_else(client: TestClient, origin: str) -> None:
    r = client.get("/api/health", headers={"Origin": origin})
    assert "access-control-allow-origin" not in r.headers


def test_cors_is_never_a_wildcard(client: TestClient) -> None:
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert r.headers.get("access-control-allow-credentials") is None
    assert "*" not in S.ALLOWED_ORIGIN_REGEX.replace(r"\d", "")


# --------------------------------------------------------------------------------------
# Serving the built frontend
# --------------------------------------------------------------------------------------


def test_missing_frontend_explains_itself_instead_of_500ing(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["ok"] is True
    assert "/api" in body["message"]


def test_frontend_is_served_and_unknown_paths_fall_back_to_index(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder, tmp_path: Path
) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>fq</title>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    eng = _engine(monkeypatch, rec)
    client = TestClient(S.create_app(S.Settings(web_dir=tmp_path), engine=eng))

    assert "<title>fq</title>" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    # A client-side route the server has never heard of gets the app, not a 404.
    assert "<title>fq</title>" in client.get("/leagues/272150391/waivers").text
    # ... but a mistyped API path is still a JSON 404, never an HTML page.
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")


def test_the_frontend_cannot_read_outside_its_directory(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder, tmp_path: Path
) -> None:
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.html").write_text("<!doctype html>ok")
    (tmp_path / "secret.txt").write_text("not for the browser")
    eng = _engine(monkeypatch, rec)
    client = TestClient(S.create_app(S.Settings(web_dir=tmp_path / "dist"), engine=eng))
    assert "not for the browser" not in client.get("/../secret.txt").text


# --------------------------------------------------------------------------------------
# Payloads survive the wire
# --------------------------------------------------------------------------------------


def test_jsonable_converts_numpy_and_kills_non_finite_floats() -> None:
    """`inf` and `nan` are not JSON; one of them anywhere takes the whole page down."""
    out = S.jsonable(
        {
            "z": math.inf,
            "nan": math.nan,
            "i": np.int64(3),
            "f": np.float64(2.5),
            "b": np.bool_(True),
            "arr": np.array([1.0, np.nan]),
            "nested": [{"w": np.int32(7)}],
            "tag": frozenset({"b", "a"}),
        }
    )
    assert out["z"] is None and out["nan"] is None
    assert out["i"] == 3 and isinstance(out["i"], int)
    assert out["f"] == 2.5 and out["b"] is True
    assert out["arr"] == [1.0, None]
    assert out["nested"][0]["w"] == 7
    assert out["tag"] == ["a", "b"]
    json.dumps(out, allow_nan=False)  # the property the whole function exists for


def test_a_payload_with_numpy_in_it_still_serves(
    monkeypatch: pytest.MonkeyPatch, rec: Recorder
) -> None:
    rec.extra = {"delta_title": np.float64(0.0123), "n": np.int64(40), "z": math.inf}
    eng = _engine(monkeypatch, rec)
    client = TestClient(S.create_app(eng.settings, engine=eng))
    data = client.get(_url("odds")).json()["data"]
    assert data["delta_title"] == pytest.approx(0.0123)
    assert data["n"] == 40
    assert data["z"] is None


# --------------------------------------------------------------------------------------
# Live ESPN. Deselect with -m 'not network'.
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_health_against_live_espn() -> None:
    client = TestClient(S.create_app(S.Settings(web_dir=Path("/nonexistent"))))
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["espn"]["espn_reachable"] is True
    assert body["espn"]["credentials"]["complete"] is True
    assert body["espn"]["authenticated"] is True
    found = _leaks(json.dumps(body), S.load_secrets())
    assert found == 0, f"{found} credential value(s) reached the health payload"


@pytest.mark.network
def test_leagues_are_reachable_live() -> None:
    client = TestClient(S.create_app(S.Settings(web_dir=Path("/nonexistent"))))
    body = client.get("/api/leagues").json()
    assert body["n_leagues"] >= 1
    assert body["n_reachable"] == body["n_leagues"], [
        (r["league_id"], r["error"]) for r in body["leagues"] if not r["reachable"]
    ]
    for row in body["leagues"]:
        assert row["size"] >= 2
        assert row["current_week"] >= 1


@pytest.mark.network
def test_roster_against_a_live_league() -> None:
    """The one payload this module assembles itself, against a real roster.

    Everything in a row comes from somewhere else -- the projection from `pipeline`, the
    value from `decide/valuation`, the slot and the injury from ESPN -- so the assertions
    are that those three actually arrived and line up on the same player.
    """
    client = TestClient(S.create_app(S.Settings(web_dir=Path("/nonexistent"), sims=500)))
    league_id = client.get("/api/leagues").json()["leagues"][0]["league_id"]
    body = client.get(_url("roster", league_id)).json()
    assert body["ok"] is True, body["error"]
    data = body["data"]
    assert data["n_players"] >= 12
    assert data["n_starting"] >= 7
    assert data["current_known"] is True
    assert data["values_ok"] is True, data["values_error"]
    top = data["players"][0]
    assert top["name"] and top["position"] in {"QB", "RB", "WR", "TE", "K", "DST"}
    assert top["ros_vorp"] is not None and top["week_mean"] > 0
    assert top["outlook"], "a rostered player with no weekly outlook is a broken join"
    # Ordered by value, which is the only ordering decision `roster_payload` makes.
    vorps = [p["ros_vorp"] for p in data["players"] if p["ros_vorp"] is not None]
    assert vorps == sorted(vorps, reverse=True)


@pytest.mark.network
def test_odds_against_a_live_league() -> None:
    """One real league, end to end, at the smallest honest simulation count."""
    client = TestClient(S.create_app(S.Settings(web_dir=Path("/nonexistent"), sims=500)))
    league_id = client.get("/api/leagues").json()["leagues"][0]["league_id"]
    body = client.get(_url("odds", league_id)).json()
    assert body["ok"] is True, body["error"]
    teams = body["data"]["teams"]
    assert len(teams) >= 8
    assert sum(t["championship"] for t in teams) == pytest.approx(1.0, abs=1e-6)
    assert all(t["championship_stderr"] > 0 for t in teams)
    # Warm on the second call, and instantly so.
    started = time.perf_counter()
    again = client.get(_url("odds", league_id)).json()
    assert again["cached"] is True
    assert time.perf_counter() - started < 0.5
