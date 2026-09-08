"""The JSON the dashboard reads. Every number in it was computed by something else.

This is a transport, and the discipline that makes it one is worth stating up front:
**no probability, projection, ranking or significance test is computed in this file.**
`report.py` already builds every payload the CLI prints, `edges/portfolio.py` already
builds the cross-league picture, and both publish a `SCHEMA_VERSION`. So the handlers
here call those functions and serialise the result. If `fq waivers --json` and
`/api/leagues/{id}/waivers` ever disagree it is a bug in exactly one place, and there is
nowhere in this module for the disagreement to have come from.

Four things this layer does have to get right on its own.

**Staleness, never silence.** A cold league costs three seconds to build and a trade
search tens of seconds on top of that; a dashboard that recomputes on every paint is
unusable and one that serves breakfast's numbers without saying so is dangerous. Every
response carries `computed_at`, `stale_seconds` and `compute_seconds` off `cache.Entry`,
and `POST /api/refresh` is the only thing that expires an analysis. Refreshing a league
also **discards its `report.Workspace`**, because the surfaces are computed off a cached
`LeagueSim` and a refresh that kept it would recompute the same numbers off the same
three-second-old tensor and call them new.

**The event loop never blocks.** The simulations are CPU-bound NumPy inside a
synchronous call stack, so every one of them runs under `asyncio.to_thread`. One league
that ESPN is answering slowly cannot stop another league's cached answer from returning
instantly, and the browser's six parallel requests do not queue behind each other.
Concurrency *within* one league is deliberately not exploited: `report.Workspace` is a
mutable cache of one `LeagueSim` shared by all of that league's surfaces, so a per-league
lock serialises them. That costs nothing -- they are CPU-bound on one machine and would
have run one after another anyway -- and it is what makes the shared workspace safe.

**A broken league is a row, not a page.** `espn_s2` expires roughly yearly and dies
silently, one league can go private, one surface can raise on a roster ESPN will not
serve. Every failure is caught at the league boundary and returned as `ok: false` with a
structured `error` and an HTTP 200, because the dashboard renders a card per league and a
non-2xx would turn "two of your three leagues are fine" into a page-level error. The
error is cached for `error_ttl` seconds so a 401 does not re-time-out on every paint.

**Credentials never reach the browser.** `SWID` and `espn_s2` are session cookies for the
user's own ESPN account. They are not in any payload by construction, and that is not
enough: every response is rendered through `ScrubbedJSONResponse`, which replaces the
literal cookie values -- raw and percent-encoded -- with `[redacted]` on the way out.
`/api/health` reports `set`/`unset` and never a value. The server binds to loopback for
the same reason; see `HOST`.

Mounting: `cli.py` is not edited by this module. One line there --
`from .api.server import mount as _mount_dash; _mount_dash(app)` -- turns this into
`fq dash`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import registry, report
from ..core import DST
from .cache import Cache, Key, KeyedLocks

log = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_ORIGIN_REGEX",
    "DEFAULT_PORT",
    "HOST",
    "Engine",
    "ScrubbedJSONResponse",
    "Settings",
    "app",
    "app_cli",
    "create_app",
    "jsonable",
    "mount",
    "portfolio_payload",
    "queue_payload",
    "roster_payload",
    "scrub",
]

#: **Loopback only, and not configurable to anything else.**
#:
#: This process holds the user's live ESPN session cookies in memory and will happily
#: read any league those cookies can see. Bound to `0.0.0.0` on a coffee-shop network it
#: is an unauthenticated proxy to somebody's ESPN account: there is no login here, no
#: CSRF token and no rate limit, because there does not need to be one on a socket only
#: this machine can open. `_require_loopback` enforces it at startup rather than
#: documenting it, and the CORS policy below is the same rule for the browser side.
HOST = "127.0.0.1"

#: 8765 rather than 8000: 8000 is the first port every other local tool takes.
DEFAULT_PORT = 8765

#: Browsers may call this API only from a page served by this machine. The dev frontend
#: runs on Vite (`http://localhost:5173`) and the built one is served from this same
#: origin, so a loopback pattern covers both and nothing else. No wildcard, and
#: `allow_credentials` stays off -- there is nothing here to authenticate with.
ALLOWED_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"

#: Where `fq dash` looks for the built frontend. `src/fantasy_quant/api/server.py`
#: -> repo root -> `web/dist`.
DEFAULT_WEB_DIR = Path(__file__).resolve().parents[3] / "web" / "dist"

REDACTED = "[redacted]"

#: A secret shorter than this is not scrubbed. Nothing real is this short, and blindly
#: replacing a two-character value would corrupt every payload it appeared in.
MIN_SECRET_LENGTH = 8

#: Simulation counts a request may ask for. The lower bound is honesty -- at 200 sims a
#: 2% championship probability carries +/-1pp and the table is noise -- and the upper
#: bound stops one URL from tying up the machine for ten minutes.
MIN_SIMS, MAX_SIMS = 200, 40_000

#: How long a *failure* is allowed to stand before the next request retries it. A league
#: whose cookies expired fails slowly, and retrying that on every paint is what turns one
#: broken league into a broken dashboard. Successful analyses never expire; they are
#: invalidated explicitly.
DEFAULT_ERROR_TTL = 60.0

#: Liveness checks are not analyses: "does ESPN answer" and "are these cookies still
#: good" are allowed to age out on their own, because a stale `yes` there is a lie about
#: the present rather than a measurement of the past.
DEFAULT_PROBE_TTL = 120.0

# --------------------------------------------------------------------------------------
# Credentials never leave this process
# --------------------------------------------------------------------------------------

_scrub_lock = threading.Lock()
_secrets: tuple[str, ...] | None = None


def _percent_encoded(value: str) -> str:
    """`{SWID}` as it would appear inside a URL or a `Cookie:` header."""
    from urllib.parse import quote

    return quote(value, safe="")


def load_secrets(env_file: Path | str | None = None) -> tuple[str, ...]:
    """The literal credential strings to scrub, raw and percent-encoded.

    Read through `registry.load_credentials`, which is the only reader of `ESPN_SWID` /
    `ESPN_S2` in the codebase. Values are held in memory and never logged: this function
    returns them so the response renderer can *remove* them, which is the one legitimate
    reason to hold them here at all.
    """
    creds = registry.load_credentials(env_file)
    out: list[str] = []
    for value in (creds.swid, creds.espn_s2):
        if value and len(value) >= MIN_SECRET_LENGTH:
            out.append(value)
            encoded = _percent_encoded(value)
            if encoded != value:
                out.append(encoded)
    return tuple(out)


def scrub_secrets() -> tuple[str, ...]:
    """The cached secret list, loaded once."""
    global _secrets
    with _scrub_lock:
        if _secrets is None:
            _secrets = load_secrets()
        return _secrets


def set_scrub_secrets(values: Sequence[str] | None) -> None:
    """Replace (or, with `None`, reload) the scrub list. Used by the leak test."""
    global _secrets
    with _scrub_lock:
        if values is None:
            _secrets = None
        else:
            _secrets = tuple(v for v in values if len(v) >= MIN_SECRET_LENGTH)


def scrub(text: str, secrets: Sequence[str] | None = None) -> str:
    """Remove credential values from a rendered response body.

    Defence in depth, not the primary control: nothing in `report.py` or
    `edges/portfolio.py` puts a cookie in a payload. But an exception message is written
    by whatever raised it, and `httpx` is entirely willing to put a request URL in one.
    A leak here is unrecoverable -- the value is in the browser and in its devtools log
    -- so the check is on every byte on the way out rather than on the paths anyone
    thought of.
    """
    for secret in secrets if secrets is not None else scrub_secrets():
        if secret and secret in text:
            text = text.replace(secret, REDACTED)
    return text


class ScrubbedJSONResponse(JSONResponse):
    """`JSONResponse` that cannot emit a credential value. The app's default class."""

    def render(self, content: Any) -> bytes:
        body = json.dumps(
            jsonable(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        )
        return scrub(body).encode("utf-8")


# --------------------------------------------------------------------------------------
# Making an analysis payload survive the wire
# --------------------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """A payload converted to types `json.dumps` will accept, applied once at the boundary.

    Two conversions, and both of them are bugs waiting in a payload that came out of
    NumPy.

    **Non-finite floats become `null`.** A z-score divided by a zero standard error, a
    correlation over a constant column, a ratio against an empty week -- these produce
    `inf` and `nan`, and `json.dumps` will happily write the bare tokens `Infinity` and
    `NaN`, which are not JSON and which `JSON.parse` refuses. One of those anywhere in a
    weekly report takes the whole page down; `null` renders as a dash.

    **NumPy scalars become Python scalars.** `np.float64` is a `float` subclass and slips
    through; `np.int64` and `np.bool_` are not, and FastAPI's encoder raises on them
    (`'numpy.int64' object is not iterable`) *before* this response class ever sees the
    payload. So the conversion happens where the analysis output enters the transport --
    in the worker thread, in `Engine.compute` -- and what the cache holds is already
    JSON-ready.
    """
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
    """How this server is configured. Every field is a server-level default.

    Request parameters may override `sims`, `limit` and the rest per call; they cannot
    override `host`, and `seed` is deliberately not per-request. Two workspaces built
    under different seeds are two different universes and their levels must not be
    differenced -- the same rule `report.Workspace` states -- so the seed is a property
    of the server, and changing it is a restart.
    """

    host: str = HOST
    port: int = DEFAULT_PORT
    season: int | None = None
    sims: int = 4000
    seed: int = 1
    config: Path | None = None
    web_dir: Path = DEFAULT_WEB_DIR
    error_ttl: float = DEFAULT_ERROR_TTL
    probe_ttl: float = DEFAULT_PROBE_TTL
    #: Rows per surface in the weekly report and on the waiver board.
    limit: int = 6
    queue_limit: int = 20


def _require_loopback(host: str) -> str:
    """Refuse to bind anywhere the network can reach. See `HOST`."""
    if host in {"localhost", "::1"}:
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise ValueError(
        f"refusing to bind to {host!r}: this process holds your live ESPN session "
        "cookies and has no authentication, so it must only be reachable from this "
        f"machine. Use {HOST}."
    )


def _is_loopback_peer(host: str | None) -> bool:
    """Whether the peer on this connection is on this machine.

    `_require_loopback` guards the bind address, but it only runs inside `fq dash`, and
    the module docstring names a second entry point: `uvicorn
    fantasy_quant.api.server:app`, whose `--host` this module never sees. Run that way
    with `--host 0.0.0.0` the server is an unauthenticated proxy to the user's ESPN
    account for everyone on the network -- a coffee shop, a hotel, a shared office --
    and there is no login here to stop them, because there was never supposed to be a
    socket to try. So the peer is checked as well as the bind: the two together mean the
    guarantee holds however the app was started.

    A peer that is not an address at all is an in-process ASGI caller -- Starlette's
    `TestClient` reports the literal `testclient`, and a raw ASGI scope may carry no
    client at all. Those never came off a socket, so they are allowed.
    """
    if not host:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


# --------------------------------------------------------------------------------------
# Envelopes
# --------------------------------------------------------------------------------------


def _error_body(err: BaseException, *, endpoint: str, league_id: int | None) -> dict[str, Any]:
    """One failure, in the shape the dashboard renders as a red row.

    The exception type is kept. "EspnError: 401 from ESPN: credentials missing or stale"
    tells the user to re-copy a cookie; "an error occurred" sends him to the logs.
    """
    return {
        "type": type(err).__name__,
        "message": scrub(str(err)) or type(err).__name__,
        "endpoint": endpoint,
        "league_id": league_id,
    }


def _envelope(
    entry: Any,
    *,
    cached: bool,
    endpoint: str,
    league_id: int | None,
    season: int | None,
    sims: int | None,
    clock: Callable[[], float],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One `cache.Entry` as an API response. Success and failure share the shape."""
    return {
        "ok": entry.ok,
        "endpoint": endpoint,
        "league_id": league_id,
        "season": season,
        "sims": sims,
        "schema_version": report.SCHEMA_VERSION,
        "cached": cached,
        **entry.staleness(clock=clock),
        "data": entry.value if entry.ok else None,
        "error": None if entry.ok else entry.value,
        **(dict(extra) if extra else {}),
    }


# --------------------------------------------------------------------------------------
# Payloads this module has to assemble itself
# --------------------------------------------------------------------------------------


def roster_payload(
    ws: Any, cfg: registry.LeagueConfig, *, week: int | None = None
) -> dict[str, Any]:
    """The user's roster, with each player's own projected outlook and value beside him.

    The one surface with no `report.py` builder, and it still computes nothing: names,
    positions and per-week moments come off `pipeline`'s `PlayerOutlook`s, value over
    replacement comes from `decide/valuation.value_league` (through
    `edges/market.context_from`, which is the existing way to get a `LeagueContext` out
    of a built `LeagueSim`), and the current lineup slot and injury status come off
    ESPN's own roster.

    Two degradations are reported rather than hidden. If the live roster cannot be read,
    `current_known` is false and no player is marked as starting -- the same distinction
    `report.lineup_payload` draws, and for the same reason: a manager who has not logged
    in since the draft is a different problem from one whose lineup is one point light.
    If the valuation fails, `values_ok` is false and the rows still carry their
    projections.
    """
    sim = ws.sim(cfg)
    team_id = ws.my_team_id(cfg)
    state = sim.state
    weeks = tuple(state.weeks)
    from_week = int(week) if week is not None else (weeks[0] if weeks else 1)
    franchise = state.franchise(team_id)
    outlooks = {o.player_id: o for o in sim.outlooks}
    names = ws.names(cfg)
    eligibility = state.slot_eligibility

    current = ws.current_starters(cfg)
    starting = set(current or ())

    entries: dict[int, Any] = {}
    try:
        entries = {e.player_id: e for e in sim.league.roster(team_id, week=from_week).entries}
    except Exception as err:  # pragma: no cover - live-only path
        log.info("no live roster detail for league %s (%s)", cfg.league_id, err)

    values: dict[int, Any] = {}
    replacement: list[dict[str, Any]] = []
    values_error = ""
    try:
        from ..decide import valuation as valuation_mod
        from ..edges import market as market_mod

        ctx = market_mod.context_from(sim, my_team_id=team_id)
        valued = valuation_mod.value_league(ctx, sim.outlooks, from_week=from_week or 1)
        values = {v.player_id: v for v in valued.values}
        replacement = [
            {
                "position_id": pos,
                "position": valuation_mod.POSITION_ABBREV.get(pos, str(pos)),
                "rostered_rank": level.demand.rostered,
                "points_per_week": level.per_week,
                "supply_limited": level.supply_limited,
            }
            for pos, level in sorted(valued.replacement.levels.items())
        ]
    except Exception as err:
        log.info("no valuation for league %s (%s)", cfg.league_id, err, exc_info=True)
        values_error = f"{type(err).__name__}: {err}"

    rows: list[dict[str, Any]] = []
    for pid in franchise.player_ids:
        outlook = outlooks.get(pid)
        entry = entries.get(pid)
        value = values.get(pid)
        this_week = outlook.weeks.get(from_week) if outlook and from_week else None
        # The label follows whichever id we actually resolved. Reading it off the
        # outlook alone leaves a player ESPN lists but the projection pool does not --
        # a just-signed free agent -- carrying a position id with a blank position.
        position_id = (
            outlook.position_id if outlook else (entry.default_position_id if entry else None)
        )
        rows.append(
            {
                "player_id": int(pid),
                "name": names.get(int(pid), str(pid)),
                "position_id": position_id,
                "position": report.POSITION_ABBREV.get(
                    position_id if position_id is not None else -1, ""
                ),
                "pro_team_id": outlook.pro_team_id
                if outlook
                else (entry.pro_team_id if entry else None),
                "lineup_slot_id": entry.lineup_slot_id if entry else None,
                "lineup_slot": (
                    report.slot_label(entry.lineup_slot_id, eligibility) if entry else ""
                ),
                "starting": int(pid) in starting,
                "injury_status": entry.injury_status if entry else "",
                "injured": bool(entry.injured) if entry else None,
                "percent_owned": entry.percent_owned if entry else None,
                "percent_started": entry.percent_started if entry else None,
                "week": from_week,
                "week_mean": this_week.mean if this_week else None,
                "week_sd": this_week.sd if this_week else None,
                "week_p_zero": this_week.p_zero if this_week else None,
                "week_playing": this_week.playing if this_week else None,
                "ros_points": value.ros_points if value else None,
                "ros_vorp": value.ros_vorp if value else None,
                "ros_vorp_per_week": value.ros_vorp_per_week if value else None,
                "ros_weeks": value.ros_weeks if value else None,
                "playoff_points": value.playoff_points if value else None,
                "playoff_vorp": value.playoff_vorp if value else None,
                "playoff_weeks": value.playoff_weeks if value else None,
                "outlook": [
                    {
                        "week": w,
                        "mean": o.mean,
                        "sd": o.sd,
                        "p_zero": o.p_zero,
                        "playing": o.playing,
                    }
                    for w, o in sorted((outlook.weeks if outlook else {}).items())
                    if w in weeks
                ],
                "projected": outlook is not None,
            }
        )

    # Value first when we have it, then this week's projection: the same order the user
    # reads a roster in, and the only ordering decision in this function. `is None` and
    # not `or`: a replacement-level player's VORP is exactly 0.0, and treating that as
    # "no value" would file him below every player who is *worse* than replacement.
    def _rank(r: dict[str, Any]) -> tuple[float, float]:
        vorp, mean = r["ros_vorp"], r["week_mean"]
        return (1e18 if vorp is None else -vorp, 0.0 if mean is None else -mean)

    rows.sort(key=_rank)
    return {
        "league_id": cfg.league_id,
        "season": cfg.season,
        "name": state.name,
        "team_id": team_id,
        "team_name": franchise.name,
        "week": from_week,
        "weeks": list(weeks),
        "current_known": current is not None,
        "n_players": len(rows),
        "n_starting": sum(1 for r in rows if r["starting"]),
        "starting_slot_counts": {str(k): v for k, v in sorted(state.lineup_slot_counts.items())},
        "values_ok": not values_error,
        "values_error": values_error,
        "replacement": replacement,
        "players": rows,
    }


def _holding(h: Any) -> dict[str, Any]:
    return {
        "league_id": h.league_id,
        "league_name": h.league_name,
        "team_id": h.team_id,
        "slot_id": h.slot_id,
        "slot": h.slot,
        "start_share": h.start_share,
        "starts": h.starts,
        "title_added": h.title_added,
        "title_added_stderr": h.title_added_stderr,
    }


def _exposure(e: Any) -> dict[str, Any]:
    return {
        "player_id": e.player_id,
        "name": e.name,
        "position_id": e.position_id,
        "position": e.position,
        "pro_team_id": e.pro_team_id,
        "n_leagues": e.n_leagues,
        "n_starting": e.n_starting,
        "equity_at_risk": e.equity_at_risk,
        "equity_share": e.equity_share,
        "portfolio_damage": e.portfolio_damage,
        "portfolio_damage_stderr": e.portfolio_damage_stderr,
        "significant": e.significant,
        "holdings": [_holding(h) for h in e.holdings],
    }


def _concentration(c: Any) -> dict[str, Any]:
    return {
        "kind": c.kind,
        "label": c.label,
        "removed": {k: list(v) for k, v in c.removed.items()},
        "n_removed": c.n_removed,
        "before": c.before,
        "after": c.after,
        "damage": c.damage,
        "stderr": c.stderr,
        "significant": c.significant,
        "expected_before": c.expected_before,
        "expected_after": c.expected_after,
        "driver": c.driver,
        "driver_damage": c.driver_damage,
        "increment": c.increment,
        "increment_stderr": c.increment_stderr,
        "increment_significant": c.increment_significant,
        "describe": c.describe(),
    }


def portfolio_payload(book: Any, rep: Any) -> dict[str, Any]:
    """`edges.portfolio.PortfolioReport` as JSON. Serialisation only.

    Every field is copied off the dataclass, including the derived properties
    (`significant`, `coupled`, `damage`, `verdict`) -- they are the module's own answers
    about its own numbers, and recomputing any of them in TypeScript is how the UI starts
    disagreeing with the CLI. The action queue is **not** here: `/api/queue` owns it, and
    both endpoints are built from the same cached `Portfolio` so they cannot disagree.
    """
    odds = rep.odds
    corr = rep.correlations
    worst_week, worst_value = corr.worst_week
    return {
        "season": book.stakes[0].state.season if book.stakes else None,
        "seed": book.seed,
        "n_sims": book.n_sims,
        "leagues": [
            {
                "league_id": s.league_id,
                "name": s.name,
                "team_id": s.team_id,
                "team_name": s.team_name,
                "title": s.title,
                "n_sims": s.n_sims,
            }
            for s in book.stakes
        ],
        "odds": {
            "names": list(odds.names),
            "titles": list(odds.titles),
            "p_at_least_one": odds.p_at_least_one,
            "p_zero": odds.p_zero,
            "p_two_plus": odds.p_two_plus,
            "expected_titles": odds.expected_titles,
            "variance_titles": odds.variance_titles,
            "independent": odds.independent,
            "sum_bound": odds.sum_bound,
            "max_bound": odds.max_bound,
            "stderr": odds.stderr,
            "dependence_cost": odds.dependence_cost,
            "dependence_cost_stderr": odds.dependence_cost_stderr,
            "dependence_significant": odds.dependence_significant,
            "n_sims": odds.n_sims,
        },
        "coupling": [
            {
                "a": c.a,
                "b": c.b,
                "shared_pool": c.shared_pool,
                "median_player_correlation": c.median_player_correlation,
                "min_player_correlation": c.min_player_correlation,
                "availability_match": c.availability_match,
                "same_seed": c.same_seed,
                "coupled": c.coupled,
            }
            for c in rep.coupling
        ],
        "correlations": {
            "pairs": [
                {
                    "a": p.a,
                    "b": p.b,
                    "shared_players": list(p.shared_players),
                    "n_shared": len(p.shared_players),
                    "weekly": p.weekly,
                    "weekly_stderr": p.weekly_stderr,
                    "season": p.season,
                    "season_stderr": p.season_stderr,
                    "champion": p.champion,
                    "champion_stderr": p.champion_stderr,
                    "champion_significant": p.champion_significant,
                }
                for p in corr.pairs
            ],
            "by_week": {str(w): v for w, v in sorted(corr.by_week.items())},
            "worst_week": worst_week,
            "worst_week_value": worst_value,
            "worst_week_gap_stderr": corr.worst_week_gap_stderr,
            "worst_week_separable": corr.worst_week_separable,
        },
        "exposures": [_exposure(e) for e in rep.exposures],
        "concentration": [_concentration(c) for c in rep.concentration],
        "pro_teams": [_concentration(c) for c in rep.pro_teams],
        "byes": [
            {
                "week": b.week,
                "starters_out": {k: list(v) for k, v in b.starters_out.items()},
                "normal_points": dict(b.normal_points),
                "bye_points": dict(b.bye_points),
                "weekly_mean": dict(b.weekly_mean),
                "unpriced": [
                    {"league": lg, "player": who, "bye_points": bye, "season_mean": mean}
                    for lg, who, bye, mean in b.unpriced
                ],
                "total_out": b.total_out,
                "leagues_hit": b.leagues_hit,
                "worst_share": b.worst_share,
                "describe": b.describe(),
            }
            for b in rep.byes
        ],
        "diversification": {
            "p_at_least_one": rep.diversification.p_at_least_one,
            "independent": rep.diversification.independent,
            "sum_bound": rep.diversification.sum_bound,
            "max_bound": rep.diversification.max_bound,
            "expected_titles": rep.diversification.expected_titles,
            "marginal_sum": rep.diversification.marginal_sum,
            "expected_titles_invariant": rep.diversification.expected_titles_invariant,
            "concentration_headroom": rep.diversification.concentration_headroom,
            "diversification_headroom": rep.diversification.diversification_headroom,
            "variance_titles": rep.diversification.variance_titles,
            "variance_independent": rep.diversification.variance_independent,
            "dependence_cost": rep.diversification.dependence_cost,
            "dependence_cost_stderr": rep.diversification.dependence_cost_stderr,
            "verdict": rep.diversification.verdict,
        },
    }


def queue_payload(
    configs: Sequence[registry.LeagueConfig], queue: Any, *, season: int
) -> dict[str, Any]:
    """One `edges.portfolio.ActionQueue` in the shape `fq queue --json` emits.

    The ranking, the significance and the blockers all come off the queue; the row
    translation is `report._action_from_item`, which is the function `report.
    portfolio_queue` uses for exactly this and which re-tests every row through
    `report.verdict_for`. It is private by name, and calling it anyway is the correct
    trade: the alternative is a second copy of the rule that decides whether a trade that
    won a search of forty is real, and two copies of that rule is how the dashboard ends
    up telling the user to go and negotiate something the weekly page calls noise.

    `holds` is composed here rather than borrowed because it is a sentence, not a
    measurement: a league with nothing to do gets a line saying so in its own numbers,
    so an empty queue never reads as a broken tool.
    """
    actions = [report._action_from_item(item, season=season) for item in queue.items]
    holds: list[str] = []
    for league in sorted({a.league_name for a in actions}):
        rows = [a for a in actions if a.league_name == league]
        if not any(a.significant and a.actionable for a in rows):
            best = max(rows, key=lambda a: a.delta_title, default=None)
            detail = (
                f" Best found: {best.headline} at {report.pp(best.delta_title)}." if best else ""
            )
            holds.append(f"{league}: nothing clears its own error and is executable today.{detail}")
    rows_out = [a.to_dict() for a in actions]
    return {
        **report._envelope(
            "queue",
            [
                {"ok": True, "league_id": c.league_id, "season": c.season, "name": c.name}
                for c in configs
            ],
            actions=rows_out,
            holds=holds,
            errors=[
                {"league": league, "surface": surface, "error": err}
                for league, surface, err in getattr(queue, "failures", ())
            ],
            ranked_by=f"edges.portfolio.{report.PORTFOLIO_HOOK}",
            n_actions=len(actions),
            first_actionable=report._first_actionable(rows_out),
        ),
        "selection_note": queue.selection_note(),
        "n_significant": sum(1 for a in actions if a.significant),
        "n_actionable": sum(1 for a in actions if a.actionable),
    }


# --------------------------------------------------------------------------------------
# The engine: registry, workspaces, cache
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Engine:
    """Owns the expensive state: one `report.Workspace` per league and one cache.

    A `Workspace` is the reason this class exists. It holds an open ESPN client and a
    built `LeagueSim`, both of which the surfaces need and neither of which is cheap;
    `pipeline.build` also closes any client it opened itself, so the caller has to own
    one for `sim.league.rosters()` -- the currently-set lineup -- to be readable at all.
    Keyed on `(league_id, season, sims)` because a different simulation count is a
    different tensor.

    `workspace_factory` and `registry_loader` are injectable so the tests can run every
    endpoint against a synthetic league without a network, and so a failure in this file
    is never confused for a failure in `pipeline.py`.
    """

    settings: Settings
    cache: Cache = field(default_factory=Cache)
    workspace_factory: Callable[[registry.LeagueConfig, int], Any] | None = None
    registry_loader: Callable[[], registry.Registry] | None = None
    _registry: registry.Registry | None = field(default=None, repr=False)
    _workspaces: dict[tuple[int, int, int], Any] = field(default_factory=dict, repr=False)
    _books: dict[tuple[int, int, int], Any] = field(default_factory=dict, repr=False)
    _queues: dict[tuple[int, int, int, int, bool], Any] = field(default_factory=dict, repr=False)
    _locks: KeyedLocks = field(default_factory=KeyedLocks, repr=False)
    _mutex: threading.Lock = field(default_factory=threading.Lock, repr=False)
    #: Single-flight for the two builds that happen in a *worker thread* rather than on
    #: the event loop. `_locks` cannot serve them: an `asyncio.Lock` is only held by the
    #: coroutine that awaits it, and `book()`/`queue()` are called from inside
    #: `asyncio.to_thread`. See `_build_lock`.
    _build_locks: dict[Any, threading.Lock] = field(default_factory=dict, repr=False)
    #: Bumped by every `invalidate`. A computation that finishes after the refresh that
    #: discarded its workspace is still returned to the caller who asked for it, but is
    #: not cached: see `compute`.
    _epoch: int = field(default=0, repr=False)

    # -- registry ----------------------------------------------------------------------

    def registry(self) -> registry.Registry:
        """The league registry, loaded once. Reloaded by a full refresh."""
        with self._mutex:
            if self._registry is None:
                loader = self.registry_loader or (
                    lambda: report.load_registry(self.settings.config)
                )
                self._registry = loader()
            return self._registry

    def configs(self) -> list[registry.LeagueConfig]:
        """Every league in the registry for this season, enabled or not.

        Disabled leagues are listed rather than dropped: `/api/leagues` is the page that
        tells the user what this tool is looking at, and a league silently missing from
        it is indistinguishable from one it cannot reach.
        """
        reg = self.registry()
        season = self.settings.season
        return [c for c in reg.sorted() if season is None or c.season == season]

    def enabled(self) -> list[registry.LeagueConfig]:
        return [c for c in self.configs() if c.enabled]

    def config(self, league_id: int) -> registry.LeagueConfig:
        for cfg in self.configs():
            if cfg.league_id == int(league_id):
                return cfg
        known = ", ".join(str(c.league_id) for c in self.configs()) or "nothing"
        raise HTTPException(
            status_code=404,
            detail=f"league {league_id} is not in the registry; it holds {known}",
        )

    # -- workspaces --------------------------------------------------------------------

    def workspace(self, cfg: registry.LeagueConfig, sims: int) -> Any:
        """This league's `report.Workspace`, built once per `(league, sims)`.

        The ESPN client is created eagerly under the mutex rather than lazily on first
        use. `Workspace.client` is a lazy property, and two surfaces racing into it would
        open two clients and leak one; doing it here means `_client` is never `None` by
        the time a worker thread touches the workspace.
        """
        key = (cfg.league_id, cfg.season, int(sims))
        with self._mutex:
            ws = self._workspaces.get(key)
            if ws is None:
                factory = self.workspace_factory or self._default_workspace
                ws = factory(cfg, int(sims))
                self._workspaces[key] = ws
            return ws

    def _build_lock(self, key: Any) -> threading.Lock:
        """One `threading.Lock` per cross-league build, made on demand.

        `Engine.compute`'s `asyncio.Lock` serialises the *coroutines* that ask for a
        surface, which is enough while one key means one computation. It is not enough
        for the portfolio: `/api/portfolio` and `/api/queue` are two different cache
        keys and therefore two different `asyncio.Lock`s, and both of their builders run
        in worker threads and call `book()`. Without a lock that a *thread* can hold,
        two simultaneous requests draw two thirty-second portfolios off one shared ESPN
        client -- which is both twice the work and two objects where this module
        promises one.
        """
        with self._mutex:
            lock = self._build_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._build_locks[key] = lock
            return lock

    def _default_workspace(self, cfg: registry.LeagueConfig, sims: int) -> Any:
        ws = report.Workspace(season=cfg.season, n_sims=sims, seed=self.settings.seed)
        ws.client  # noqa: B018 - see `workspace`: open the client before any thread can
        return ws

    def portfolio_workspace(self, sims: int) -> Any:
        """A workspace shared by every league, for `build_portfolio`'s one client.

        Cross-league work must run on ONE seed and ONE client: simulation `s` in Wine
        Wednesday has to be the same football as simulation `s` in Blacksburg or the
        joint probabilities are the independent ones with a coupled label on them. Keyed
        under league id 0, which no ESPN league uses.
        """
        cfg = registry.LeagueConfig(league_id=0, season=self.settings.season or 0)
        return self.workspace(cfg, sims)

    def sims_for(self, requested: int | None) -> int:
        if requested is None:
            return self.settings.sims
        return max(MIN_SIMS, min(int(requested), MAX_SIMS))

    # -- invalidation ------------------------------------------------------------------

    def invalidate(self, *, league_id: int | None = None) -> dict[str, Any]:
        """Drop cached answers, and the built simulations they were computed from.

        Dropping only the payload cache would be a refresh button that changes nothing:
        every surface is computed off a `LeagueSim` held on the workspace, so the next
        request would rebuild the same numbers from the same three-second-old tensor.
        The workspaces go too, and their clients are closed.
        """
        removed = self.cache.invalidate(league_id=league_id)
        with self._mutex:
            self._epoch += 1
            doomed = [
                k for k in self._workspaces if league_id is None or k[0] in (int(league_id), 0)
            ]
            for key in doomed:
                ws = self._workspaces.pop(key)
                try:
                    ws.close()
                except Exception:  # pragma: no cover - a closed client is still gone
                    log.debug("workspace for %s did not close cleanly", key, exc_info=True)
            self._books.clear()
            self._queues.clear()
            if league_id is None:
                self._registry = None
        return {
            "entries": removed,
            "workspaces": len(doomed),
            "league_id": league_id,
        }

    def close(self) -> None:
        self.invalidate(league_id=None)

    def computing(self) -> list[str]:
        """Keys with a computation in flight, for `/api/health`'s "work in progress"."""
        return [":".join(str(part) for part in key) for key in self._locks.held()]

    # -- the one path every computation takes ------------------------------------------

    async def compute(
        self,
        key: Key,
        lock_key: Any,
        build: Callable[[], Any],
        *,
        force: bool = False,
    ) -> tuple[Any, bool]:
        """`(entry, was_cached)`. Runs `build` in a worker thread, once, under a lock.

        Three properties, in the order they matter. The cache is checked *before* the
        lock, so a warm answer never queues behind a cold one. `build` runs under
        `asyncio.to_thread`, so a thirty-second trade search does not stop the event loop
        from serving another league. And the cache is re-checked *after* the lock, which
        is what makes two simultaneous cold requests for the same league produce one
        computation instead of two racing mutations of one shared `Workspace`.
        """
        hit = self._fresh(key, force=force)
        if hit is not None:
            return hit, True
        async with self._locks(lock_key):
            hit = self._fresh(key, force=force)
            if hit is not None:
                return hit, True
            started = time.perf_counter()
            epoch = self._epoch
            try:
                # `jsonable` runs in the worker thread too: it is a walk over a payload
                # with a few thousand nodes and it belongs with the work, not on the
                # event loop. What lands in the cache is already JSON-ready.
                value = await asyncio.to_thread(lambda: jsonable(build()))
                ok = True
            except Exception as err:
                log.warning("%s failed for %s: %s", key.endpoint, key.league_id, err, exc_info=True)
                value = _error_body(err, endpoint=key.endpoint, league_id=key.league_id)
                ok = False
            elapsed = time.perf_counter() - started
            if epoch != self._epoch:
                # A refresh landed while this was building, which closed the client this
                # build was reading through. The caller who asked still gets an answer,
                # but it describes a workspace that no longer exists -- and caching the
                # failure it usually is would leave the refresh button looking broken for
                # `error_ttl` seconds, which is exactly the opposite of what it does.
                log.info(
                    "%s for %s finished after a refresh; not cached", key.endpoint, key.league_id
                )
                return self.cache.entry(key, value, ok=ok, compute_seconds=elapsed), False
            entry = self.cache.put(key, value, ok=ok, compute_seconds=elapsed)
            return entry, False

    def _fresh(self, key: Key, *, force: bool) -> Any:
        if force:
            return None
        hit = self.cache.peek(key)
        if hit is None:
            return None
        # A successful analysis never expires on its own; a failure is retried once its
        # backoff is up. See `DEFAULT_ERROR_TTL`.
        if hit.ok or hit.age(clock=self.cache.clock) <= self.settings.error_ttl:
            return hit
        return None

    async def surface(
        self,
        endpoint: str,
        cfg: registry.LeagueConfig,
        build: Callable[[Any, registry.LeagueConfig], Any],
        *,
        sims: int | None = None,
        force: bool = False,
        **params: Any,
    ) -> dict[str, Any]:
        """One per-league endpoint, end to end: key, lock, thread, envelope."""
        n_sims = self.sims_for(sims)
        key = Key.of(endpoint, league_id=cfg.league_id, season=cfg.season, sims=n_sims, **params)
        ws = self.workspace(cfg, n_sims)
        entry, cached = await self.compute(
            key, ("league", cfg.league_id, cfg.season, n_sims), lambda: build(ws, cfg), force=force
        )
        return _envelope(
            entry,
            cached=cached,
            endpoint=endpoint,
            league_id=cfg.league_id,
            season=cfg.season,
            sims=n_sims,
            clock=self.cache.clock,
            extra={"name": cfg.name, "team_id": cfg.team_id},
        )

    async def cross(
        self,
        endpoint: str,
        build: Callable[[], Any],
        *,
        sims: int | None = None,
        force: bool = False,
        **params: Any,
    ) -> dict[str, Any]:
        """One cross-league endpoint. Same path, keyed with `league_id=None`."""
        n_sims = self.sims_for(sims)
        key = Key.of(endpoint, season=self.settings.season, sims=n_sims, **params)
        entry, cached = await self.compute(key, ("cross", endpoint, n_sims), build, force=force)
        return _envelope(
            entry,
            cached=cached,
            endpoint=endpoint,
            league_id=None,
            season=self.settings.season,
            sims=n_sims,
            clock=self.cache.clock,
        )

    # -- the shared portfolio ----------------------------------------------------------

    def book(self, sims: int) -> Any:
        """The `edges.portfolio.Portfolio`, built once and shared by two endpoints.

        `/api/portfolio` and `/api/queue` are two views of one object. Building it twice
        would cost another three seconds per league *and* would put the two pages on two
        different draws, so the exposure table and the action ranking would be measured
        against different championship probabilities.

        The memo is checked, the build lock taken, and the memo checked **again**: the
        two endpoints have different cache keys, so nothing upstream stops them from
        arriving here at the same time in two worker threads.
        """
        from ..edges import portfolio as portfolio_mod

        key = (0, self.settings.season or 0, int(sims))
        with self._mutex:
            book = self._books.get(key)
        if book is not None:
            return book
        with self._build_lock(("book", key)):
            with self._mutex:
                book = self._books.get(key)
            if book is not None:
                return book
            configs = self.enabled()
            if not configs:
                raise report.ReportError("no enabled leagues in the registry")
            seasons = {c.season for c in configs}
            if len(seasons) != 1:
                raise report.ReportError(
                    f"the enabled leagues span seasons {sorted(seasons)}; a portfolio is one "
                    "shared NFL season, so pick one with --season"
                )
            ws = self.portfolio_workspace(sims)
            book = portfolio_mod.build_portfolio(
                [(c.league_id, c.team_id if c.team_id is not None else 1) for c in configs],
                seasons.pop(),
                seed=self.settings.seed,
                n_sims=int(sims),
                client=ws.client,
            )
            with self._mutex:
                self._books[key] = book
            return book

    def queue(self, sims: int, *, limit: int, actionable_only: bool) -> Any:
        """The `ActionQueue` off that portfolio, built once per `(limit, actionable)`.

        Held as an object rather than only as JSON so `/api/portfolio` can be handed the
        queue it already ran instead of `edges.portfolio.report` re-running every surface
        in every league to produce one it then throws away.
        """
        from ..edges import portfolio as portfolio_mod

        key = (0, self.settings.season or 0, int(sims), int(limit), bool(actionable_only))
        with self._mutex:
            queue = self._queues.get(key)
        if queue is not None:
            return queue
        # Same double-check as `book`, and the lock order is always queue-then-book, so
        # the nested `self.book(sims)` below cannot invert it into a deadlock.
        with self._build_lock(("queue", key)):
            with self._mutex:
                queue = self._queues.get(key)
            if queue is not None:
                return queue
            queue = portfolio_mod.action_queue(
                self.book(sims), limit=int(limit), actionable_only=bool(actionable_only)
            )
            with self._mutex:
                self._queues[key] = queue
            return queue


# --------------------------------------------------------------------------------------
# Probes: is ESPN answering, and are the cookies still good
# --------------------------------------------------------------------------------------


def _probe_league(engine: Engine, cfg: registry.LeagueConfig, sims: int) -> dict[str, Any]:
    """One `mSettings` fetch: the cheapest proof that a league is actually readable.

    Deliberately not a simulation. `/api/leagues` is the first thing the dashboard loads
    and it must answer in under a second for every league, so it asks the one question a
    card needs -- can I see this league, and what does ESPN call it today -- and leaves
    the three-second build to the page the user then opens.
    """
    from ..espn.league import League

    ws = engine.workspace(cfg, sims)
    settings = League(ws.client, cfg.league_id, cfg.season).settings()
    return {
        "reachable": True,
        "error": None,
        "espn_name": settings.name,
        "size": settings.size,
        "is_public": settings.is_public,
        "current_week": settings.status.current_week,
        "final_week": settings.status.final_scoring_period,
        "playoff_team_count": settings.schedule.playoff_team_count,
        "uses_faab": settings.acquisition.uses_faab,
        "faab_budget": settings.acquisition.budget,
        "format_tags": list(settings.format_tags),
    }


def _auth_probe(engine: Engine, sims: int) -> dict[str, Any]:
    """Whether ESPN answers at all, and whether these cookies still see a private league.

    `espn_s2` expires roughly yearly and dies *silently* -- the failure is a 401 on one
    league, not an error at startup -- so the health check reads one private league
    rather than trusting that a cookie exists. No credential value appears in the result,
    only `set` / `unset`.
    """
    from ..espn.client import EspnClient

    creds = registry.load_credentials()
    out: dict[str, Any] = {
        "credentials": {
            "swid": "set" if creds.swid else "unset",
            "espn_s2": "set" if creds.espn_s2 else "unset",
            "complete": creds.complete,
        },
        "espn_reachable": False,
        "authenticated": None,
        "season": None,
        "week": None,
        "checked_league_id": None,
        "error": None,
    }
    client: Any = None
    configs: list[registry.LeagueConfig] = []
    try:
        configs = engine.enabled()
        if configs:
            client = engine.workspace(configs[0], sims).client
        else:
            client = EspnClient(**creds.as_kwargs())
        season, week = client.current_season_and_week()
        out["espn_reachable"] = True
        out["season"] = season
        out["week"] = week
    except Exception as err:
        out["error"] = scrub(f"{type(err).__name__}: {err}")
        return out

    if not configs:
        return out
    cfg = configs[0]
    out["checked_league_id"] = cfg.league_id
    try:
        from ..espn.league import League

        League(client, cfg.league_id, cfg.season).settings()
        out["authenticated"] = True
    except Exception as err:
        out["authenticated"] = False
        out["error"] = scrub(f"{type(err).__name__}: {err}")
    return out


# --------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------


def _sections(skip: Sequence[str] | None) -> tuple[str, ...]:
    """`--skip` for the weekly report, refusing a misspelled section.

    Same rule as the CLI: silently ignoring an unknown section is how a caller ends up
    believing he skipped the slow trade search when he did not.
    """
    skipped = {s.strip().lower() for s in (skip or []) if s.strip()}
    unknown = sorted(skipped - set(report.WEEKLY_SECTIONS))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown section(s) {unknown}; expected one of "
            f"{', '.join(report.WEEKLY_SECTIONS)}",
        )
    return tuple(s for s in report.WEEKLY_SECTIONS if s not in skipped)


def create_app(settings: Settings | None = None, *, engine: Engine | None = None) -> FastAPI:
    """Build the ASGI app. `engine=` is the seam the tests inject a synthetic league at."""
    cfg = settings or Settings()
    eng = engine or Engine(settings=cfg)

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> Any:
        """Close every ESPN client on the way out; there is nothing to do on the way in.

        Deliberately no warm-up: a server that simulated three leagues at startup would
        take ten seconds to accept its first connection and would do it again on every
        `--reload`. The first request pays, and it is told what it is waiting for.
        """
        yield
        eng.close()

    app = FastAPI(
        title="fantasy_quant",
        version=str(report.SCHEMA_VERSION),
        summary="Local JSON over the fantasy_quant analytics stack. Loopback only.",
        default_response_class=ScrubbedJSONResponse,
        lifespan=_lifespan,
    )
    app.state.engine = eng
    app.state.settings = cfg

    # Loopback only, no wildcard, no credentials: see ALLOWED_ORIGIN_REGEX and HOST.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOWED_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Added last, so it is the OUTERMOST middleware: a peer that is not on this machine
    # is turned away before CORS, before routing, and before anything reads a league.
    # See `_is_loopback_peer` for why the bind guard is not sufficient on its own.
    @app.middleware("http")
    async def _loopback_only(request: Request, call_next: Callable[..., Any]) -> Any:
        peer = request.client.host if request.client else None
        if not _is_loopback_peer(peer):
            log.warning(
                "refused %s %s from non-loopback peer %s", request.method, request.url.path, peer
            )
            return ScrubbedJSONResponse(
                {
                    "ok": False,
                    "error": {
                        "type": "NotLoopback",
                        "message": (
                            "this server is readable only from the machine it runs on; "
                            "it holds live ESPN session cookies and has no login. Start "
                            f"it on {HOST}."
                        ),
                    },
                },
                status_code=403,
            )
        return await call_next(request)

    # FastAPI's built-in handlers construct a plain `JSONResponse`, so a 404 or a 422
    # would otherwise be the one class of body that leaves without passing the scrubber
    # -- and a validation error echoes back whatever the caller put in the query string.
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> Any:
        return ScrubbedJSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> Any:
        return ScrubbedJSONResponse({"detail": jsonable(exc.errors())}, status_code=422)

    api = APIRouter(prefix="/api")

    # -- meta --------------------------------------------------------------------------

    @api.get("/health")
    async def health(refresh: bool = False) -> dict[str, Any]:
        """Liveness, plus whether the ESPN session cookies are still valid right now.

        The auth answer is allowed to age out (`probe_ttl`) because it is a fact about
        the present, not a measurement of the past -- unlike every analysis here, which
        expires only when the user asks.
        """
        sims = eng.sims_for(None)
        key = Key.of("health", sims=sims)
        hit = None if refresh else eng.cache.peek(key, max_age=cfg.probe_ttl)
        cached = hit is not None
        if hit is None:
            entry, _ = await eng.compute(
                key, ("cross", "health", sims), lambda: _auth_probe(eng, sims), force=True
            )
        else:
            entry = hit
        probe = entry.value if entry.ok else {}
        return {
            "ok": True,
            "endpoint": "health",
            "schema_version": report.SCHEMA_VERSION,
            "cached": cached,
            **entry.staleness(clock=eng.cache.clock),
            "server": {
                "host": cfg.host,
                "port": cfg.port,
                "sims": cfg.sims,
                "seed": cfg.seed,
                "season": cfg.season,
                "web_dist": str(cfg.web_dir),
                "web_dist_present": cfg.web_dir.is_dir(),
            },
            "espn": probe if entry.ok else {"error": entry.value},
            "leagues": len(eng.configs()),
            "cache": eng.cache.stats(),
            "computing": eng.computing(),
        }

    @api.get("/leagues")
    async def leagues(refresh: bool = False, probe: bool = True) -> dict[str, Any]:
        """The registry, plus whether each league is actually reachable right now.

        Registry rows are returned even when the probe fails, and disabled leagues are
        listed as disabled: the point of this endpoint is to say what the tool is looking
        at, and a league missing from it would be indistinguishable from one that does
        not exist.
        """
        configs = eng.configs()
        sims = eng.sims_for(None)

        async def _row(c: registry.LeagueConfig) -> dict[str, Any]:
            base = {
                **c.to_dict(),
                "team_id": c.team_id,
                "notes": c.notes,
                "tags": list(c.tags),
                "reachable": None,
                "error": None,
                "checked_at": None,
                "stale_seconds": None,
            }
            if not probe or not c.enabled:
                return base
            key = Key.of("league-probe", league_id=c.league_id, season=c.season)
            hit = None if refresh else eng.cache.peek(key, max_age=cfg.probe_ttl)
            if hit is None:
                hit, _ = await eng.compute(
                    key,
                    ("probe", c.league_id, c.season),
                    lambda cc=c: _probe_league(eng, cc, sims),
                    force=True,
                )
            if hit.ok:
                return {**base, **hit.value, **hit.staleness(clock=eng.cache.clock)}
            return {
                **base,
                "reachable": False,
                "error": hit.value,
                **hit.staleness(clock=eng.cache.clock),
            }

        rows = await asyncio.gather(*(_row(c) for c in configs))
        return {
            "ok": True,
            "endpoint": "leagues",
            "schema_version": report.SCHEMA_VERSION,
            "season": cfg.season,
            "sims": cfg.sims,
            "seed": cfg.seed,
            "generated_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "n_leagues": len(rows),
            "n_reachable": sum(1 for r in rows if r.get("reachable")),
            "leagues": list(rows),
        }

    @api.post("/refresh")
    async def refresh_endpoint(league_id: int | None = None) -> dict[str, Any]:
        """Invalidate. With no `league_id`, everything, registry included.

        This is the only thing that expires an analysis, and it discards the built
        simulations as well as the payloads -- see `Engine.invalidate`.
        """
        if league_id is not None:
            eng.config(league_id)  # 404 rather than a silent no-op on a typo
        dropped = eng.invalidate(league_id=league_id)
        return {
            "ok": True,
            "endpoint": "refresh",
            "invalidated": dropped,
            "at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }

    # -- per-league surfaces -----------------------------------------------------------

    @api.get("/leagues/{league_id}/odds")
    async def odds(
        league_id: int, refresh: bool = False, sims: int | None = None
    ) -> dict[str, Any]:
        """`report.odds_payload`: the championship table, with its Monte Carlo error."""
        c = eng.config(league_id)
        return await eng.surface("odds", c, report.odds_payload, sims=sims, force=refresh)

    @api.get("/leagues/{league_id}/weekly")
    async def weekly(
        league_id: int,
        refresh: bool = False,
        sims: int | None = None,
        limit: int | None = None,
        skip: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        """`report.weekly_payload`: every section, each failing independently."""
        c = eng.config(league_id)
        sections = _sections(skip)
        rows = limit if limit is not None else cfg.limit
        return await eng.surface(
            "weekly",
            c,
            lambda ws, cc: report.weekly_payload(ws, cc, sections=sections, limit=rows),
            sims=sims,
            force=refresh,
            limit=rows,
            skip=sorted(set(report.WEEKLY_SECTIONS) - set(sections)) or None,
        )

    @api.get("/leagues/{league_id}/waivers")
    async def waivers(
        league_id: int,
        refresh: bool = False,
        sims: int | None = None,
        limit: int = 10,
        week: int | None = None,
    ) -> dict[str, Any]:
        """`report.waivers_payload`: the board and the priority threshold it must clear."""
        c = eng.config(league_id)
        return await eng.surface(
            "waivers",
            c,
            lambda ws, cc: report.waivers_payload(ws, cc, limit=limit, week=week),
            sims=sims,
            force=refresh,
            limit=limit,
            week=week,
        )

    @api.get("/leagues/{league_id}/trades")
    async def trades(
        league_id: int,
        refresh: bool = False,
        sims: int | None = None,
        limit: int = 5,
        min_gain: float = 0.0,
    ) -> dict[str, Any]:
        """`report.trades_payload`: confirmed Pareto trades, with the selection caveat."""
        c = eng.config(league_id)
        return await eng.surface(
            "trades",
            c,
            lambda ws, cc: report.trades_payload(ws, cc, limit=limit, min_gain=min_gain),
            sims=sims,
            force=refresh,
            limit=limit,
            min_gain=min_gain,
        )

    @api.get("/leagues/{league_id}/lineup")
    async def lineup(
        league_id: int, refresh: bool = False, sims: int | None = None, week: int | None = None
    ) -> dict[str, Any]:
        """`report.lineup_payload`: start/sit measured against the lineup actually set."""
        c = eng.config(league_id)
        return await eng.surface(
            "lineup",
            c,
            lambda ws, cc: report.lineup_payload(ws, cc, week=week),
            sims=sims,
            force=refresh,
            week=week,
        )

    @api.get("/leagues/{league_id}/stream")
    async def stream(
        league_id: int,
        refresh: bool = False,
        sims: int | None = None,
        position: int = DST,
        plan_weeks: int = 8,
    ) -> dict[str, Any]:
        """`report.stream_payload`: this week's action and the plan that prices it."""
        c = eng.config(league_id)
        return await eng.surface(
            "stream",
            c,
            lambda ws, cc: report.stream_payload(
                ws, cc, position_id=position, plan_weeks=plan_weeks
            ),
            sims=sims,
            force=refresh,
            position=position,
            plan_weeks=plan_weeks,
        )

    @api.get("/leagues/{league_id}/roster")
    async def roster(
        league_id: int, refresh: bool = False, sims: int | None = None, week: int | None = None
    ) -> dict[str, Any]:
        """`roster_payload`: the roster with each player's outlook and value."""
        c = eng.config(league_id)
        return await eng.surface(
            "roster",
            c,
            lambda ws, cc: roster_payload(ws, cc, week=week),
            sims=sims,
            force=refresh,
            week=week,
        )

    # -- cross-league ------------------------------------------------------------------

    @api.get("/portfolio")
    async def portfolio(
        refresh: bool = False,
        sims: int | None = None,
        min_leagues: int = 2,
        top_exposures: int = 5,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """`edges.portfolio.report`: exposure, correlation and concentration damage.

        Handed the action queue this server already ran, so the surfaces are not run a
        second time to build a queue this endpoint then discards.
        """
        rows = limit if limit is not None else cfg.queue_limit

        def _build() -> dict[str, Any]:
            from ..edges import portfolio as portfolio_mod

            n = eng.sims_for(sims)
            book = eng.book(n)
            queue = eng.queue(n, limit=rows, actionable_only=False)
            rep = portfolio_mod.report(
                book,
                limit=rows,
                min_leagues=min_leagues,
                top_exposures=top_exposures,
                queue=queue,
            )
            return portfolio_payload(book, rep)

        return await eng.cross(
            "portfolio",
            _build,
            sims=sims,
            force=refresh,
            min_leagues=min_leagues,
            top_exposures=top_exposures,
            limit=rows,
        )

    @api.get("/queue")
    async def queue(
        refresh: bool = False,
        sims: int | None = None,
        limit: int | None = None,
        actionable: bool = False,
    ) -> dict[str, Any]:
        """The ranked cross-league action list, in `fq queue --json`'s own shape.

        Falls back to `report.queue_payload` over the per-league weekly reports when the
        shared-season portfolio cannot be built -- a league ESPN will not serve today
        takes the coupled ranking down with it, and a merged queue with a warning on it
        beats no queue at all.
        """
        rows = limit if limit is not None else cfg.queue_limit

        def _build() -> dict[str, Any]:
            n = eng.sims_for(sims)
            configs = eng.enabled()
            season = configs[0].season if configs else (cfg.season or 0)
            try:
                q = eng.queue(n, limit=rows, actionable_only=actionable)
                return queue_payload(configs, q, season=season)
            except Exception as err:
                log.warning("shared-season queue failed (%s); merging weekly reports", err)
                weekly_rows: list[dict[str, Any]] = []
                for c in configs:
                    ws = eng.workspace(c, n)
                    try:
                        weekly_rows.append({"ok": True, **report.weekly_payload(ws, c)})
                    except Exception as inner:
                        weekly_rows.append(
                            {
                                "ok": False,
                                "league_id": c.league_id,
                                "season": c.season,
                                "name": c.name,
                                "error": f"{type(inner).__name__}: {inner}",
                            }
                        )
                merged = report.queue_payload({"leagues": weekly_rows}, limit=rows)
                merged["degraded"] = scrub(f"{type(err).__name__}: {err}")
                # The three keys `queue_payload` (which is `fq queue --local`'s shape)
                # does not carry. The UI must not have to ask which path produced a
                # queue before it can read one; there is no selection to note on this
                # path, and an empty string is how that is said.
                merged_actions = merged.get("actions") or []
                merged.setdefault("selection_note", "")
                merged["n_significant"] = sum(1 for a in merged_actions if a.get("significant"))
                merged["n_actionable"] = sum(1 for a in merged_actions if a.get("actionable"))
                return merged

        return await eng.cross(
            "queue", _build, sims=sims, force=refresh, limit=rows, actionable=actionable
        )

    app.include_router(api)
    _mount_frontend(app, cfg.web_dir)
    return app


def _mount_frontend(app: FastAPI, web_dir: Path) -> None:
    """Serve `web/dist` if it has been built, so `fq dash` is one command.

    Registered after the API router, and every path is checked against the real
    directory before `index.html` is served as a fallback, which is what a client-side
    router needs. An unknown `/api/...` path still 404s as JSON rather than being handed
    the SPA -- a mistyped endpoint that returns an HTML page is a debugging afternoon.
    """
    root = web_dir.resolve()
    assets = root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> Any:
        if path.startswith("api/") or path == "api":
            raise HTTPException(status_code=404, detail=f"no such endpoint: /{path}")
        if not root.is_dir():
            return ScrubbedJSONResponse(
                {
                    "ok": True,
                    "service": "fantasy_quant",
                    "message": (
                        f"no built frontend at {root}. The JSON API is at /api "
                        "(/api/health, /api/leagues, /api/queue)."
                    ),
                }
            )
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail=f"nothing to serve for /{path}")


#: The ASGI app `uvicorn fantasy_quant.api.server:app` imports. Constructing it is cheap:
#: the registry is read on the first request, not here, so an unreadable `leagues.toml`
#: is a structured error on one endpoint rather than a server that will not start.
app = create_app()


# --------------------------------------------------------------------------------------
# fq dash
# --------------------------------------------------------------------------------------

app_cli = typer.Typer(help="The local dashboard.")


@app_cli.command("dash")
def dash_command(
    port: Annotated[int, typer.Option("--port", help="Loopback port to serve on.")] = DEFAULT_PORT,
    host: Annotated[
        str, typer.Option("--host", help="Must be a loopback address; see the module docstring.")
    ] = HOST,
    sims: Annotated[int, typer.Option("--sims", help="Simulations per league.")] = 4000,
    seed: Annotated[int, typer.Option("--seed", help="Common-random-numbers seed.")] = 1,
    season: Annotated[int | None, typer.Option("--season", help="Restrict to one season.")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Path to leagues.toml.")] = None,
    web: Annotated[
        Path | None, typer.Option("--web", help="Directory holding the built frontend.")
    ] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on code changes.")] = False,
) -> None:
    """Serve the dashboard on 127.0.0.1. Read-only against ESPN; it recommends, you click."""
    import uvicorn

    try:
        bind = _require_loopback(host)
    except ValueError as err:
        typer.secho(str(err), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from err

    settings = Settings(
        host=bind,
        port=port,
        season=season,
        sims=max(MIN_SIMS, min(sims, MAX_SIMS)),
        seed=seed,
        config=config,
        web_dir=(web or DEFAULT_WEB_DIR).resolve(),
    )
    typer.secho(f"fantasy_quant dashboard -> http://{bind}:{port}", fg=typer.colors.CYAN)
    if not settings.web_dir.is_dir():
        typer.secho(
            f"  no built frontend at {settings.web_dir}; serving the JSON API only "
            "(build it with `npm run build` in web/).",
            fg=typer.colors.YELLOW,
        )
    if reload:  # pragma: no cover - developer convenience
        uvicorn.run(
            "fantasy_quant.api.server:app", host=bind, port=port, reload=True, log_level="info"
        )
        return
    uvicorn.run(create_app(settings), host=bind, port=port, log_level="info")


def mount(parent: typer.Typer) -> typer.Typer:
    """Attach `dash` to `fq`. One line in `cli.py`; nothing there is edited by this file.

    from .api.server import mount as _mount_dash; _mount_dash(app)
    """
    for command in app_cli.registered_commands:
        if command not in parent.registered_commands:
            parent.registered_commands.append(command)
    return parent
