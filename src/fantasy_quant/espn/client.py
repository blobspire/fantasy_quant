"""Thin read client for the ESPN Fantasy API.

Deliberately not `espn-api`. That library hard-codes `pointsOverrides.get('16')`
(D/ST only), so it silently mis-scores any league with TE premium or per-position
PPR; and its `free_agents()` pins the sort to percent-owned, so it cannot rank the
player pool by league-scored projected points -- the single most useful thing this
API can do.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .endpoints import DEFAULT_HEADERS, game_meta_url

log = logging.getLogger(__name__)

# ESPN publishes no rate limit and none was observed under concurrency, but there
# is no reason to be rude to an endpoint we depend on all season.
_MIN_INTERVAL_S = 0.2


class EspnError(RuntimeError):
    """An ESPN API call failed in a way worth surfacing."""


class EspnClient:
    """Synchronous ESPN reader with conditional-GET caching and filter paging."""

    def __init__(
        self,
        swid: str | None = None,
        espn_s2: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        cookies: dict[str, str] = {}
        if swid and espn_s2:
            # SWID keeps its braces; espn_s2 stays URL-encoded as copied.
            cookies = {"SWID": swid, "espn_s2": espn_s2}
        self._authed = bool(cookies)
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            cookies=cookies,
            timeout=timeout,
            follow_redirects=False,
        )
        self._etags: dict[str, str] = {}
        self._last_call = 0.0

    @property
    def authenticated(self) -> bool:
        return self._authed

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EspnClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        self._last_call = time.monotonic()

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        fantasy_filter: dict[str, Any] | None = None,
        use_etag: bool = False,
    ) -> tuple[Any, httpx.Headers]:
        """GET and return (payload, response headers).

        Raises EspnError on anything that isn't a 200 or a 304.
        """
        headers: dict[str, str] = {}
        if fantasy_filter is not None:
            headers["x-fantasy-filter"] = json.dumps(fantasy_filter, separators=(",", ":"))

        cache_key = "|".join(
            (url, json.dumps(params, sort_keys=True), headers.get("x-fantasy-filter", ""))
        )
        if use_etag and cache_key in self._etags:
            headers["If-None-Match"] = self._etags[cache_key]

        self._throttle()
        resp = self._client.get(url, params=params, headers=headers)

        if resp.status_code == 304:
            return None, resp.headers
        if resp.status_code == 400:
            # Overwhelmingly this is a `limit` with no accompanying sort.
            raise EspnError(f"400 from ESPN ({url}): {resp.text[:300]}")
        if resp.status_code == 401:
            raise EspnError(
                "401 from ESPN: credentials missing or stale. espn_s2 expires roughly "
                "yearly and dies silently -- re-copy it from your browser."
            )
        if resp.status_code == 404:
            # Measured 2026-09-07 across six league ids: a league that exists but
            # is not visible to you returns 401 AUTH_LEAGUE_NOT_VISIBLE, while one
            # that does not exist returns 404. So a 404 really does mean "no such
            # thing" -- for a leaguedefaults variant too, several of which exist
            # only in recent seasons.
            raise EspnError(f"404 from ESPN ({url}): no such league or resource.")
        if resp.status_code != 200:
            raise EspnError(f"HTTP {resp.status_code} from ESPN ({url}): {resp.text[:300]}")

        if use_etag and (etag := resp.headers.get("etag")):
            self._etags[cache_key] = etag

        return resp.json(), resp.headers

    def game_meta(self) -> dict[str, Any]:
        """currentSeasonId and currentScoringPeriod, straight from ESPN."""
        payload, _ = self.get(game_meta_url())
        return payload

    def current_season_and_week(self) -> tuple[int, int]:
        """(season, scoring period) as ESPN reports them.

        Use this rather than deriving the week from a calendar; ESPN's scoring
        period is the authority and it does not track weeks naively.
        """
        meta = self.game_meta()
        season = int(meta["currentSeasonId"])
        # `currentScoringPeriod` lives under `currentSeason`; tolerate a top-level
        # one too, since this is exactly the sort of thing ESPN moves without notice.
        holder = meta.get("currentSeason") or meta
        period = holder.get("currentScoringPeriod") or meta.get("currentScoringPeriod")
        if not period:
            raise EspnError(
                f"no currentScoringPeriod in ESPN game metadata; keys were {sorted(meta)}"
            )
        return season, int(period["id"])

    def player_pool(
        self,
        url: str,
        *,
        limit: int = 250,
        sort: dict[str, Any] | None = None,
        extra_filter: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        max_players: int | None = None,
    ) -> list[dict[str, Any]]:
        """Page the whole player pool.

        ESPN rejects a `limit` that arrives without a sort, so one is always sent.
        Paging terminates on the `x-fantasy-filter-player-count` response header,
        which reports total matches *before* the limit is applied.
        """
        sort = sort or {"sortPercOwned": {"sortAsc": False, "sortPriority": 1}}
        players: list[dict[str, Any]] = []
        offset = 0
        total: int | None = None

        while True:
            pfilter: dict[str, Any] = {"limit": limit, "offset": offset, **sort}
            if extra_filter:
                pfilter.update(extra_filter)

            payload, headers = self.get(url, params=params, fantasy_filter={"players": pfilter})
            if total is None:
                raw_total = headers.get("x-fantasy-filter-player-count")
                total = int(raw_total) if raw_total else None

            batch = (payload or {}).get("players", []) if isinstance(payload, dict) else []
            if not batch:
                break

            players.extend(batch)
            offset += len(batch)

            if max_players is not None and len(players) >= max_players:
                return players[:max_players]
            if total is not None and offset >= total:
                break
            if len(batch) < limit:
                break

        log.info("pulled %d players from %s", len(players), url.rsplit("/", 1)[-1])
        return players


def sort_by_projection(season: int, week: int | None = None) -> dict[str, Any]:
    """Sort spec ranking the pool by league-scored projected points.

    Season projection by default; pass `week` for a single week -- note that a
    weekly sort ALSO requires `scoringPeriodId` on the query string, or the rows
    silently vanish.
    """
    stat_id = f"11{season}{week}" if week is not None else f"10{season}"
    return {"sortAppliedStatTotal": {"sortAsc": False, "sortPriority": 1, "value": stat_id}}
