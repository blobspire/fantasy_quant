"""Sleeper adapter: free component-level projections and waiver demand.

Why this source is worth a module of its own: we ensemble at the **component-stat**
level, not the points level, so a projection set only transfers across leagues with
different scoring if it publishes the underlying counts. Sleeper does -- targets,
carries, attempts, first downs, and the reception-distance buckets its own bonus
scoring needs -- for free and without auth. Nobody else free does.

Everything below was probed live on 2026-09-07. Where the probe disagreed with the
research notes, the probe wins and the disagreement is called out.

Traps this module exists to prevent
-----------------------------------

1. **A bogus season or week returns HTTP 200 with a full skeleton.** Asking for week
   99 or season 1999 yields the entire player list with `stats` either empty or
   holding only the ADP sentinel. Same failure class as ESPN's bogus `view=` and
   FanDuel's bad tab slug: *alarm on empty, not on status*. `_require_content`
   does exactly that.

2. **`order_by` does not order.** With `order_by=ppr` the live week-1 payload came
   back leading with Greg Ward, Quincy McDuffie and a 2009 tight end -- all of them
   projectionless. Never take the head of the list as "the top players"; sort
   client-side on `pts_ppr`.

3. **`position[]` filters on `fantasy_positions`, not `player.position`.** A
   QB/RB/WR/TE request returned 71 FBs, a CB and a DB, because their
   `fantasy_positions` include a skill position. Filter on the right field or a
   position-keyed aggregate silently gains rows. This does not stop at the API
   boundary: `to_frame` carries **both** columns, because persisting only
   `position` writes a frame whose `group_by("position")` invents an `FB` bucket
   and drops four fullbacks out of the RB pool Sleeper itself selected.

4. **ADP 999.0 / 1000.0 are "unranked" sentinels, not an ADP.** Live, *every* one of
   the 355 QB rows carried `adp_dynasty: 999.0`. Averaging them raw poisons any ADP
   comparison. `adp()` scrubs them; the raw dict still has them.

5. **`trending` silently caps at 100.** `limit=500` returned 100 rows.

6. **Most rows carry no projection at all.** Live week 1: 3,115 rows for QB/RB/WR/TE,
   398 with `pts_ppr`. The rest are inactive or camp bodies. `has_projection`
   discriminates; parsing defaults to dropping them.

7. **"Not the ADP sentinel" is not the same as "projected".** Three live week-1
   rows carried the sentinel, `gp`, and nothing but return-game yardage
   (`pr`, `pr_yd`, `def_kr_yd`) -- no points, no offensive component. A blocklist
   of known-useless keys admitted them and they reached the frame as all-null
   rows, which is the 401-vs-398 gap between "kept" and "has a point total".
   `has_projection` is therefore an allowlist; see `_PROJECTION_EVIDENCE`.

8. **Missing string ids are `""`, not null.** 520 players in the live master carry
   `sportradar_id: ""`. This is RESEARCH's `db_playerids` "NA" trap in Sleeper's
   own idiom: a coverage check written as `is not None` scores them as joinable.
   `crosswalk_rows` normalises every string id through `_as_str`.

Measured on the live player master (12,226 players; 816 active, teamed, skill
position): `espn_id` coverage **24.3%**, `gsis_id` 18.9%. This confirms the research
note -- **Sleeper is not a usable ESPN crosswalk on its own**. `crosswalk_rows`
exposes what it does have (and a 100%-populated `search_full_name`) so `data.ids`
can use it as one input among several, but do not key a join on it.

`company=` probe (asked for explicitly): `rotowire` is still the only value with
data. `company=sleeper` and `company=fantasypros` both return HTTP 200 and 750 rows
whose entire `stats` payload is `{"adp_dd_ppr": 1000.0}`. Handled by the same
empty-content alarm as trap 1.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import polars as pl

log = logging.getLogger(__name__)

BASE = "https://api.sleeper.app"

# Sleeper publishes a 1000 req/min ceiling. 0.12s between calls caps us at 500/min
# and we make single-digit calls per run anyway; the throttle is here so a future
# per-week loop cannot turn into a burst.
_MIN_INTERVAL_S = 0.12

# Sleeper asks that the 14 MB player master be pulled at most once a day.
PLAYERS_TTL_S = 24 * 3600
DEFAULT_CACHE_DIR = Path("data/cache/sleeper")

# In-memory conditional-GET bodies are capped so the player master (14.6 MB) is
# not retained for the process lifetime alongside its own disk cache.
_MAX_CACHED_BODY_BYTES = 4_000_000

SKILL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Receptions split by the yardage of the catch. Sleeper's own bonus-scoring
# categories; commonly (loosely) called the air-yard buckets. They are counts of
# receptions, not yards, and they do NOT sum to `rec` -- live, Gibbs' buckets total
# 69.3 against 63.0 receptions -- so use them as shape, never as a partition.
RECEPTION_BUCKETS: tuple[str, ...] = (
    "rec_0_4",
    "rec_5_9",
    "rec_10_19",
    "rec_20_29",
    "rec_30_39",
    "rec_40p",
)

# The league-scoring-independent counts the ensemble combines. Deliberately excludes
# `pts_*` (already scored, and scored in someone else's league) and the bonus_* keys
# (Sleeper's own scoring artifacts, not football events).
COMPONENT_FIELDS: tuple[str, ...] = (
    "pass_att",
    "pass_cmp",
    "pass_yd",
    "pass_td",
    "pass_int",
    "pass_fd",
    "pass_2pt",
    "rush_att",
    "rush_yd",
    "rush_td",
    "rush_fd",
    "rush_2pt",
    "rec_tgt",
    "rec",
    "rec_yd",
    "rec_td",
    "rec_fd",
    "rec_2pt",
    *RECEPTION_BUCKETS,
    "fum_lost",
    "gp",
)

# All 12 flavors, confirmed present on every season row.
ADP_FIELDS: tuple[str, ...] = (
    "adp_std",
    "adp_ppr",
    "adp_half_ppr",
    "adp_2qb",
    "adp_idp",
    "adp_idp_1qb",
    "adp_dynasty",
    "adp_dynasty_std",
    "adp_dynasty_ppr",
    "adp_dynasty_half_ppr",
    "adp_dynasty_2qb",
    "adp_rookie",
)

# "Undrafted" placeholders. Both values turn up on both endpoints -- live week 1
# carried `adp_dd_ppr: 999.0` and `adp_dd_ppr: 1000.0` on different rows -- so scrub
# the pair, not one per endpoint. Never average these.
ADP_SENTINELS: frozenset[float] = frozenset({999.0, 1000.0})

_POINT_FIELDS: tuple[str, ...] = ("pts_ppr", "pts_half_ppr", "pts_std")

# What counts as evidence that Sleeper actually projected a player.
#
# This is an ALLOWLIST on purpose. A blocklist of "keys that mean nothing" leaks:
# live week 1 returned three rows (Luke McCaffrey, Dyami Brown, Isaac Guerendo)
# whose entire stats payload was the ADP sentinel plus return-game noise --
# `pr`, `pr_yd`, `def_kr_yd` -- with no points and no offensive component at all.
# Any blocklist that had not enumerated those three keys in advance let them
# through as "projections", and they land in `to_frame` as all-null rows. Sleeper
# publishes ~48 stat keys on the weekly endpoint and adds more per position, so
# enumerating the useless ones is a losing game; enumerate the useful ones.
#
# `gp` is excluded: it is 1.0 (weekly) or 18.0 (season) on every row Sleeper
# returns, projected or not, so it is a shape artifact rather than a projection.
_PROJECTION_EVIDENCE: frozenset[str] = frozenset(_POINT_FIELDS) | (
    frozenset(COMPONENT_FIELDS) - {"gp"}
)

# Missing string ids arrive from Sleeper as null *or* as the empty string -- 520
# players in the live master carry `sportradar_id: ""`. RESEARCH flags the same
# class of trap in `db_playerids`, where the missing marker is the literal "NA"
# and naive truthiness reports fake 100% coverage. Normalise both to None so a
# coverage check written as `is not None` cannot be fooled.
_STRING_NULLS: frozenset[str] = frozenset({"", "na", "n/a", "null", "none", "-", "--"})

# Absolute anchors for scoring waiver demand, measured over a live 24h window on
# 2026-09-07: the most-added player drew ~240k adds platform-wide, the 100th-most
# ~5.3k. A player under the floor is not being chased by anyone.
SATURATED_ADDS_PER_DAY = 250_000
NEGLIGIBLE_ADDS_PER_DAY = 1_000


class SleeperError(RuntimeError):
    """A Sleeper call failed, or succeeded with a payload that carries no data."""


@dataclass(frozen=True, slots=True)
class SleeperProjection:
    """One player-week (or player-season) projection row."""

    sleeper_id: str
    name: str
    position: str | None
    fantasy_positions: tuple[str, ...]
    team: str | None
    opponent: str | None
    season: int
    week: int | None
    game_id: str | None
    company: str | None
    injury_status: str | None
    updated_at: dt.datetime | None
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def has_projection(self) -> bool:
        """True when Sleeper actually projected this player.

        Most rows in any response do not: the endpoint returns the whole position
        universe and leaves `stats` holding nothing but the ADP sentinel, `gp`,
        and -- for a handful of players -- return-game noise. See
        `_PROJECTION_EVIDENCE` for why this is an allowlist.
        """
        return not _PROJECTION_EVIDENCE.isdisjoint(self.stats)

    @property
    def pts_ppr(self) -> float | None:
        return self.stats.get("pts_ppr")

    @property
    def pts_half_ppr(self) -> float | None:
        return self.stats.get("pts_half_ppr")

    @property
    def pts_std(self) -> float | None:
        return self.stats.get("pts_std")

    def components(self) -> dict[str, float]:
        """The league-scoring-independent counts, for the component-level ensemble."""
        return {k: self.stats[k] for k in COMPONENT_FIELDS if k in self.stats}

    def adp(self) -> dict[str, float]:
        """ADP flavors with the unranked sentinels removed.

        Season rows only; the weekly endpoint carries a different, near-useless
        `adp_dd_ppr` that is 1000.0 for almost everyone.
        """
        out: dict[str, float] = {}
        for key in ADP_FIELDS:
            value = self.stats.get(key)
            if value is not None and value not in ADP_SENTINELS:
                out[key] = value
        return out


@dataclass(frozen=True, slots=True)
class TrendingPlayer:
    """One row of the trending add/drop board."""

    sleeper_id: str
    count: int


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    return None if f is None else int(f)


def _as_str(value: Any) -> str | None:
    """A string id, or None for every flavour of "missing" Sleeper publishes."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return None if text.lower() in _STRING_NULLS else text


def _search_key(name: str | None) -> str | None:
    """Sleeper's own `search_full_name` convention: lowercase, alphanumerics only.

    The 32 team defenses in the master carry `search_full_name: null`, so we
    synthesise it rather than hand `data.ids` a name-join column with a hole in
    exactly the rows RESEARCH says always fail to join.
    """
    if not name:
        return None
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    return key or None


def _epoch_ms(value: Any) -> dt.datetime | None:
    ms = _as_float(value)
    if ms is None:
        return None
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.UTC)


def parse_projection(record: Mapping[str, Any]) -> SleeperProjection:
    """One raw record -> a typed row. Tolerant: every field is optional upstream."""
    player = record.get("player") or {}
    name = " ".join(
        p for p in (player.get("first_name"), player.get("last_name")) if p
    ).strip() or str(record.get("player_id", ""))

    stats_raw = record.get("stats") or {}
    stats = {str(k): f for k, v in stats_raw.items() if (f := _as_float(v)) is not None}

    return SleeperProjection(
        sleeper_id=str(record.get("player_id", "")),
        name=name,
        position=player.get("position"),
        fantasy_positions=tuple(player.get("fantasy_positions") or ()),
        # `team` sits at the top level *and* inside `player`; they agree live, but
        # the top-level one is the one attached to the projected game.
        team=record.get("team") or player.get("team"),
        opponent=record.get("opponent"),
        season=_as_int(record.get("season")) or 0,
        week=_as_int(record.get("week")),
        game_id=record.get("game_id"),
        company=record.get("company"),
        injury_status=player.get("injury_status"),
        updated_at=_epoch_ms(record.get("updated_at") or record.get("last_modified")),
        stats=stats,
    )


def parse_projections(
    payload: Any,
    *,
    positions: Sequence[str] | None = None,
    projected_only: bool = True,
) -> list[SleeperProjection]:
    """Parse a projections payload, newest-useful-first.

    `positions` filters on `fantasy_positions`, which is what the server's
    `position[]` parameter actually keys on -- filtering on `player.position`
    instead lets fullbacks and defensive backs through (71 FBs, a CB and a DB in
    the live week-1 QB/RB/WR/TE pull).

    `projected_only` drops the rows Sleeper returns with no projection, which are
    the large majority of any response.
    """
    if not isinstance(payload, list):
        raise SleeperError(f"expected a list of projection records, got {type(payload).__name__}")

    wanted = {p.upper() for p in positions} if positions else None
    out: list[SleeperProjection] = []
    for record in payload:
        if not isinstance(record, Mapping):
            continue
        proj = parse_projection(record)
        if projected_only and not proj.has_projection:
            continue
        if wanted and not wanted.intersection(proj.fantasy_positions):
            continue
        out.append(proj)

    # The server's `order_by` is decorative -- see module docstring, trap 2.
    out.sort(key=lambda p: p.pts_ppr if p.pts_ppr is not None else -1.0, reverse=True)
    return out


def parse_trending(payload: Any) -> list[TrendingPlayer]:
    """Parse `[{player_id, count}, ...]`, descending by count.

    `player_id` is a team abbreviation ("LV", "NE") for a team defense, not a
    numeric id -- every platform names D/ST differently and this is Sleeper's way.
    """
    if not isinstance(payload, list):
        raise SleeperError(f"expected a list of trending rows, got {type(payload).__name__}")

    rows: list[TrendingPlayer] = []
    for record in payload:
        if not isinstance(record, Mapping):
            continue
        pid = record.get("player_id")
        count = _as_int(record.get("count"))
        if pid is None or count is None:
            continue
        rows.append(TrendingPlayer(sleeper_id=str(pid), count=count))

    rows.sort(key=lambda r: r.count, reverse=True)
    return rows


@dataclass(frozen=True, slots=True)
class WaiverDemand:
    """Per-player add and drop counts over one lookback window.

    This is the input to `n` in the FAAB bid model -- the number of rival managers
    who will contest a claim -- which is why it is exposed as a per-player mapping
    rather than a top-N list.

    Sleeper's counts are platform-wide (the top target drew 239k adds in 24h), so
    they are a *relative* demand signal, not a count of bidders in your league.
    `expected_bidders` turns one into the other through an explicitly provisional
    prior; read its docstring before trusting the number.
    """

    lookback_hours: int
    captured_at: dt.datetime
    adds: dict[str, int] = field(default_factory=dict)
    drops: dict[str, int] = field(default_factory=dict)

    def add_count(self, sleeper_id: str) -> int:
        return self.adds.get(sleeper_id, 0)

    def drop_count(self, sleeper_id: str) -> int:
        return self.drops.get(sleeper_id, 0)

    def net_adds(self, sleeper_id: str) -> int:
        """Adds minus drops.

        Worth having: the single most-added player in the live window was also the
        single most-dropped, so raw adds alone read churn as demand.
        """
        return self.add_count(sleeper_id) - self.drop_count(sleeper_id)

    def demand_share(self, sleeper_id: str) -> float:
        """Add interest on [0, 1].

        Log-scaled, because counts in one window span three orders of magnitude and
        a linear share flattens everything below the top few to zero.

        Anchored on the absolute constants above rather than on this window's own
        maximum. Normalizing by the window max makes the score depend on `limit`
        and on how hot the week happened to be -- the 5th-most-added player scored
        0.90 against a window max and 0.78 against a fixed anchor, and only the
        second number means the same thing next week.
        """
        count = self.add_count(sleeper_id)
        scale = max(self.lookback_hours, 1) / 24.0
        floor = NEGLIGIBLE_ADDS_PER_DAY * scale
        ceiling = SATURATED_ADDS_PER_DAY * scale
        if count <= floor:
            return 0.0
        span = math.log1p(ceiling) - math.log1p(floor)
        return min(1.0, (math.log1p(count) - math.log1p(floor)) / span)

    def expected_bidders(
        self,
        sleeper_id: str,
        *,
        league_size: int = 12,
        ceiling: float = 5.0,
    ) -> float:
        """Provisional estimate of `n` for the FAAB winner's-curse haircut.

        The research anchor is that the relevant `n` is the **2-5 teams actually
        bidding**, not league size. This maps demand share onto [1, `ceiling`] and
        clamps at `league_size - 1`.

        It is a prior, not a measurement: Sleeper does not publish a league count,
        so the platform-wide add total cannot be converted to a per-league rate.
        Replace it as soon as your own leagues' `Transaction.bid_amount` history
        gives a fitted bidder distribution -- that history is the thing the FAAB
        model is supposed to be parameterized on.
        """
        if league_size < 2:
            raise ValueError("league_size must be at least 2")
        n = 1.0 + (ceiling - 1.0) * self.demand_share(sleeper_id)
        return min(n, float(league_size - 1))

    def top_adds(self, limit: int = 25) -> list[TrendingPlayer]:
        rows = [TrendingPlayer(pid, c) for pid, c in self.adds.items()]
        rows.sort(key=lambda r: r.count, reverse=True)
        return rows[:limit]


def _require_content(rows: Sequence[SleeperProjection], what: str) -> None:
    """Alarm on empty, not on status.

    Sleeper answers a nonsense week or an unpopulated `company` with HTTP 200 and
    the full player list, every row's `stats` empty or sentinel-only. A caller that
    checks the status code will happily write a file of nothing.
    """
    if not any(r.has_projection for r in rows):
        raise SleeperError(
            f"Sleeper returned {len(rows)} rows for {what} but none carry a projection. "
            "That is what a bad season/week or an unpopulated company= looks like: "
            "HTTP 200 with a skeleton."
        )


class SleeperClient:
    """Synchronous Sleeper reader: throttled, conditional-GET cached, no auth."""

    def __init__(
        self,
        timeout: float = 30.0,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._client = httpx.Client(
            base_url=BASE,
            headers={"Accept": "application/json", "User-Agent": "fantasy_quant/0.1"},
            timeout=timeout,
            follow_redirects=True,
        )
        self._cache_dir = Path(cache_dir)
        self._etags: dict[str, str] = {}
        self._bodies: dict[str, Any] = {}
        self._last_call = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SleeperClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        self._last_call = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET and return the decoded payload.

        Sleeper serves weak ETags and a 600s s-maxage on everything here, and
        honors `If-None-Match` with a 304 (verified), so a repeated pull inside a
        run costs one conditional request instead of 14 MB.
        """
        key = f"{path}?{json.dumps(params, sort_keys=True, default=str)}"
        headers: dict[str, str] = {}
        if key in self._etags:
            headers["If-None-Match"] = self._etags[key]

        self._throttle()
        try:
            resp = self._client.get(path, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise SleeperError(f"Sleeper request failed ({path}): {exc}") from exc

        if resp.status_code == 304 and key in self._bodies:
            return self._bodies[key]
        if resp.status_code == 429:
            raise SleeperError(
                "429 from Sleeper: rate limited. The published ceiling is 1000/min; "
                "something is calling in a loop without the throttle."
            )
        if resp.status_code != 200:
            raise SleeperError(f"HTTP {resp.status_code} from Sleeper ({path}): {resp.text[:300]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise SleeperError(f"non-JSON body from Sleeper ({path}): {resp.text[:200]}") from exc

        # Etag and body are cached together or not at all: a stored etag with no
        # stored body would produce a 304 that falls through to the non-200 branch
        # below and raises. The size guard keeps the 14.6 MB player master out of
        # memory for the life of the process -- it has a 24h disk cache already.
        if (etag := resp.headers.get("etag")) and len(resp.content) <= _MAX_CACHED_BODY_BYTES:
            self._etags[key] = etag
            self._bodies[key] = payload
        return payload

    def state(self) -> dict[str, Any]:
        """Sleeper's own view of the current season and week.

        Use it instead of a calendar computation, the same way we use ESPN's
        `currentScoringPeriod` rather than deriving the week.
        """
        payload = self.get("/v1/state/nfl")
        if not isinstance(payload, dict):
            raise SleeperError(f"unexpected /v1/state/nfl payload: {type(payload).__name__}")
        return payload

    def current_season_and_week(self) -> tuple[int, int]:
        state = self.state()
        return int(state["season"]), int(state["week"])

    def weekly_projections(
        self,
        season: int,
        week: int,
        positions: Sequence[str] = SKILL_POSITIONS,
        company: str | None = None,
        projected_only: bool = True,
    ) -> list[SleeperProjection]:
        """Component-level projections for one scoring week.

        This is the endpoint that carries `rec_tgt`; the season endpoint does not
        (see `season_projections`).
        """
        params: dict[str, Any] = {
            "season_type": "regular",
            "position[]": list(positions),
            "order_by": "ppr",
        }
        if company:
            params["company"] = company

        payload = self.get(f"/projections/nfl/{season}/{week}", params=params)
        rows = parse_projections(payload, positions=positions, projected_only=projected_only)
        _require_content(rows, f"{season} week {week}" + (f" company={company}" if company else ""))
        return rows

    def season_projections(
        self,
        season: int,
        positions: Sequence[str] = SKILL_POSITIONS,
        company: str | None = None,
        projected_only: bool = True,
    ) -> list[SleeperProjection]:
        """Full-season projections plus all 12 ADP flavors.

        Note what is *missing* relative to the weekly endpoint: `rec_tgt` is not
        published here at any position (checked across QB, RB, WR and TE). Season
        target volume has to be built by summing the weekly rows, which is the same
        shape as the ESPN frozen-season-total workaround.
        """
        params: dict[str, Any] = {
            "season_type": "regular",
            "position[]": list(positions),
            "order_by": "pts_ppr",
        }
        if company:
            params["company"] = company

        payload = self.get(f"/projections/nfl/{season}", params=params)
        rows = parse_projections(payload, positions=positions, projected_only=projected_only)
        _require_content(rows, f"{season} season" + (f" company={company}" if company else ""))
        return rows

    def trending(
        self,
        kind: str = "add",
        lookback_hours: int = 24,
        limit: int = 100,
    ) -> list[TrendingPlayer]:
        """Trending adds or drops. `limit` is capped at 100 by the server."""
        if kind not in ("add", "drop"):
            raise ValueError(f"kind must be 'add' or 'drop', got {kind!r}")
        payload = self.get(
            f"/v1/players/nfl/trending/{kind}",
            params={"lookback_hours": lookback_hours, "limit": min(limit, 100)},
        )
        return parse_trending(payload)

    def waiver_demand(self, lookback_hours: int = 24, limit: int = 100) -> WaiverDemand:
        """Adds and drops in one object, keyed by player -- the FAAB `n` input."""
        adds = self.trending("add", lookback_hours=lookback_hours, limit=limit)
        drops = self.trending("drop", lookback_hours=lookback_hours, limit=limit)
        return WaiverDemand(
            lookback_hours=lookback_hours,
            captured_at=dt.datetime.now(dt.UTC),
            adds={r.sleeper_id: r.count for r in adds},
            drops={r.sleeper_id: r.count for r in drops},
        )

    def players(self, force: bool = False, ttl_s: float = PLAYERS_TTL_S) -> dict[str, Any]:
        """The player master, cached on disk.

        14.6 MB and 12,226 players live. Sleeper explicitly asks for at most one
        pull per day, so this goes to disk rather than to memory and honors a TTL
        across processes.
        """
        cache = self._cache_dir / "players_nfl.json"
        if not force and cache.exists():
            age = time.time() - cache.stat().st_mtime
            if age < ttl_s:
                with cache.open() as fh:
                    return json.load(fh)

        payload = self.get("/v1/players/nfl")
        if not isinstance(payload, dict) or not payload:
            raise SleeperError("empty or unexpected player master from Sleeper")

        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(payload, fh)
        tmp.replace(cache)
        log.info("cached %d Sleeper players -> %s", len(payload), cache)
        return payload


DST_POSITION = "DEF"


def crosswalk_rows(
    players: Mapping[str, Mapping[str, Any]],
    positions: Sequence[str] | None = SKILL_POSITIONS,
    active_only: bool = True,
    include_dst: bool = True,
) -> list[dict[str, Any]]:
    """Reduce the 14 MB player master to the id columns a crosswalk needs.

    Offered to `data.ids` as *one* input, not as the spine. Measured live over 816
    active, teamed skill players: `espn_id` present on 24.3%, `gsis_id` on 18.9%.
    `search_full_name` is Sleeper's own normalized key, which makes this table
    useful for name matching and useless for id joining.

    `include_dst` keeps the 32 team defenses regardless of `positions`. RESEARCH is
    blunt about this -- *"every join failure in testing was a team defense"* -- and
    `trending()` hands back D/ST as bare team abbreviations ("LV", "JAX"), so a
    crosswalk that filtered them out with the default `positions=SKILL_POSITIONS`
    would be unable to resolve the one class of row that is known to break. Their
    `search_full_name` is null upstream and is synthesised here.

    Every string id is normalised through `_as_str`, so `""` (520 players in the
    live master carry `sportradar_id: ""`) reads as missing rather than as a
    present-but-blank id.
    """
    wanted = {p.upper() for p in positions} if positions else None
    rows: list[dict[str, Any]] = []
    for pid, p in players.items():
        if not isinstance(p, Mapping):
            continue
        if active_only and not p.get("active"):
            continue
        fantasy_positions = tuple(p.get("fantasy_positions") or ())
        is_dst = DST_POSITION in fantasy_positions or p.get("position") == DST_POSITION
        keep = not wanted or bool(wanted.intersection(fantasy_positions))
        if not keep and not (include_dst and is_dst):
            continue
        full_name = _as_str(p.get("full_name")) or _as_str(
            " ".join(x for x in (p.get("first_name"), p.get("last_name")) if x)
        )
        rows.append(
            {
                "sleeper_id": str(pid),
                "full_name": full_name,
                "search_full_name": _as_str(p.get("search_full_name")) or _search_key(full_name),
                "position": _as_str(p.get("position")),
                "fantasy_positions": list(fantasy_positions),
                "team": _as_str(p.get("team")),
                "espn_id": _as_int(p.get("espn_id")),
                "gsis_id": _as_str(p.get("gsis_id")),
                "yahoo_id": _as_int(p.get("yahoo_id")),
                "rotowire_id": _as_int(p.get("rotowire_id")),
                "sportradar_id": _as_str(p.get("sportradar_id")),
                "fantasy_data_id": _as_int(p.get("fantasy_data_id")),
                "years_exp": _as_int(p.get("years_exp")),
                "depth_chart_order": _as_int(p.get("depth_chart_order")),
                "injury_status": _as_str(p.get("injury_status")),
            }
        )
    return rows


IdResolver = Callable[[Sequence[SleeperProjection]], Mapping[str, str]]

# The source name `data.ids` files Sleeper ids under.
SLEEPER_SOURCE = "sleeper"

# Names a future `data.ids` might expose as a purpose-built Sleeper helper. Checked
# first for forward compatibility; the real coupling below is to `ids.IdResolver`.
_RESOLVER_NAMES = (
    "resolve_sleeper_ids",
    "sleeper_to_internal",
    "resolve_sleeper",
    "from_sleeper",
)


def _index_resolver(index: Any, source: str = SLEEPER_SOURCE) -> IdResolver:
    """Adapt anything exposing `.to_canonical(value, source)` -- i.e. `ids.IdResolver`."""

    def resolve(projections: Sequence[SleeperProjection]) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in projections:
            canonical = index.to_canonical(p.sleeper_id, source)
            if canonical:
                out[p.sleeper_id] = str(canonical)
        return out

    return resolve


@lru_cache(maxsize=1)
def default_id_resolver(allow_download: bool = False) -> IdResolver | None:
    """Build a Sleeper->canonical-id resolver from `data.ids`, or return None.

    Bound to `ids.IdResolver.to_canonical(value, "sleeper")`, which is the API that
    module actually ships -- and which resolves team defenses too, mapping the bare
    abbreviations `trending()` returns ("LV" -> "DST-LV").

    Defaults to `allow_download=False` so importing a projection feed never
    silently triggers a crosswalk download; call it explicitly with True (or hand
    `attach_internal_ids` a prebuilt resolver) when a refresh is wanted. Cached,
    because building the index parses ~13k crosswalk records.

    Every failure path returns None *and logs*: a crosswalk that is not ready must
    not take the Sleeper feed offline, but it must also not fail silently.
    """
    try:
        from . import ids
    except Exception:
        log.warning("data.ids is unavailable; Sleeper rows keep their native ids", exc_info=True)
        return None

    for name in _RESOLVER_NAMES:
        fn = getattr(ids, name, None)
        if callable(fn):
            return fn

    index_cls = getattr(ids, "IdResolver", None)
    if index_cls is None:
        log.warning("data.ids exposes no IdResolver and no %s; no internal ids", _RESOLVER_NAMES)
        return None
    try:
        index = index_cls.load(allow_download=allow_download)
    except Exception:
        log.warning(
            "data.ids crosswalk could not be loaded (allow_download=%s); "
            "Sleeper rows keep their native ids",
            allow_download,
            exc_info=True,
        )
        return None
    return _index_resolver(index, getattr(ids, "SLEEPER", SLEEPER_SOURCE))


def attach_internal_ids(
    projections: Sequence[SleeperProjection],
    resolver: IdResolver | Any | None = None,
) -> dict[str, str]:
    """sleeper_id -> internal id, empty when no resolver is available.

    `resolver` may be a callable over the projection sequence, or an index object
    exposing `.to_canonical(value, source)` -- so a caller that already built an
    `ids.IdResolver` can pass it straight in rather than rebuilding the crosswalk.

    Loose on purpose. A resolver that raises is logged and treated as absent: the
    Sleeper feed is useful on its own (component projections keyed by Sleeper id),
    and must not be taken offline by a crosswalk that is not ready.
    """
    if resolver is not None and hasattr(resolver, "to_canonical"):
        resolver = _index_resolver(resolver)
    if resolver is None:
        resolver = default_id_resolver()
    if resolver is None:
        log.warning(
            "no Sleeper->internal id resolver available; %d rows keep their native ids",
            len(projections),
        )
        return {}
    try:
        mapping = resolver(projections)
    except Exception:
        log.warning("Sleeper id resolver failed; continuing without internal ids", exc_info=True)
        return {}
    return {str(k): str(v) for k, v in dict(mapping).items()}


def to_frame(projections: Iterable[SleeperProjection]) -> pl.DataFrame:
    """Wide Polars frame with a fixed column set.

    The column list is explicit rather than derived from whichever stat keys turned
    up, for the same reason `snapshot.py` stores stats as parallel lists: a schema
    that follows the data means two weeks' files refuse to concatenate. Missing
    stats are null, never zero -- "Sleeper did not project this" and "Sleeper
    projected zero" are different facts.

    `fantasy_positions` is carried alongside `position` because trap 3 does not stop
    at the API boundary. `player.position` is the field Sleeper does *not* filter
    on: the live week-1 QB/RB/WR/TE pull yields a frame whose `position` column
    holds four `FB` rows and a `DB` row, so a downstream `group_by("position")`
    invents phantom positions and leaves four fullbacks out of the RB pool that
    Sleeper itself put there. Group on `fantasy_positions`.
    """
    numeric = (*_POINT_FIELDS, *COMPONENT_FIELDS, *ADP_FIELDS)
    records: list[dict[str, Any]] = []
    for p in projections:
        adp = p.adp()
        row: dict[str, Any] = {
            "sleeper_id": p.sleeper_id,
            "name": p.name,
            "position": p.position,
            "fantasy_positions": list(p.fantasy_positions),
            "team": p.team,
            "opponent": p.opponent,
            "season": p.season,
            "week": p.week,
            "game_id": p.game_id,
            "company": p.company,
            "injury_status": p.injury_status,
            "updated_at": p.updated_at.replace(tzinfo=None) if p.updated_at else None,
        }
        for key in numeric:
            row[key] = adp.get(key) if key in ADP_FIELDS else p.stats.get(key)
        records.append(row)
    schema: dict[str, Any] = {
        "sleeper_id": pl.Utf8,
        "name": pl.Utf8,
        "position": pl.Utf8,
        "fantasy_positions": pl.List(pl.Utf8),
        "team": pl.Utf8,
        "opponent": pl.Utf8,
        "season": pl.Int64,
        "week": pl.Int64,
        "game_id": pl.Utf8,
        "company": pl.Utf8,
        "injury_status": pl.Utf8,
        "updated_at": pl.Datetime("us"),
        **{k: pl.Float64 for k in numeric},
    }
    return pl.DataFrame(records, schema=schema)
