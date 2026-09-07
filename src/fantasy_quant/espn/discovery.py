"""Find every ESPN league a SWID belongs to.

There is no "list my leagues" endpoint on the fantasy API -- `/leagues` returns 405. The
only route is the Fan API:

    GET https://fan.api.espn.com/apis/v2/fans/{SWID}?featureFlags=expandAthlete
    Cookies: SWID={...}; espn_s2=...

**The response shape here is community-reported and unverified.** We have no credentials in
this environment, and the endpoint answers `404 {"message": "fan not found"}` to an
unauthenticated request, so nothing about the body could be confirmed live. Everything
below is written on that assumption:

* The first successful call dumps the raw body to `data/reference/fan_api_raw.json` (the
  `data/` tree is git-ignored) so a human can read what actually came back.
* The parser **searches** for fields instead of walking a fixed path. It carries context
  down the tree, so `preferences[].metaData.entry.groups[].groupId` still resolves when a
  nesting level is renamed, inserted or removed -- only the leaf names have to survive.
* When the football filter (`typeId == 9`, `gameId == 1`) matches nothing but the payload
  clearly holds league-shaped records, it retries unfiltered and says so loudly. Better to
  over-collect and let `verify` drop the non-football ids than to report zero leagues.
* `manual_league_ids` is a first-class input, not a fallback. It is the only path that is
  *known* to work, so a registry that lists ids explicitly never depends on any of the above.

Nothing here logs a credential. The SWID goes in the request path because the endpoint is
addressed by it; it is never written to a log line, and `espn_s2` is never touched outside
the cookie jar.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..registry import Credentials, load_credentials, normalize_swid
from .client import EspnClient, EspnError
from .endpoints import FAN_API
from .league import League, owner_key

log = logging.getLogger(__name__)

#: Where the first raw Fan API body lands for human inspection. Under the git-ignored
#: `data/` tree, because the body contains the account's own SWID.
FAN_RAW_PATH = Path("data/reference/fan_api_raw.json")

#: ESPN preference type for a fantasy team, and the gameId for fantasy football.
FFL_PREFERENCE_TYPE_ID = 9
FFL_GAME_ID = 1
FFL_ABBREVS = frozenset({"ffl"})

# Leaf names the parser hunts for. Order matters: earlier keys win.
_LEAGUE_ID_KEYS = ("groupId", "leagueId", "groupID")
_TEAM_ID_KEYS = ("entryId", "teamId", "entryID")
_SEASON_KEYS = ("seasonId", "seasonID", "season", "gameSeason", "year")
_NAME_KEYS = ("groupName", "leagueName", "name")
_GAME_KEYS = ("gameId", "gameID")
#: Authoritative: whatever these say the sport is, it is.
_GAME_ABBREV_KEYS = ("gameAbbrev", "sportAbbrev")
#: A hint only. `abbrev` is also a team's short name ("BINI"), so it can confirm football
#: but must never be read as a denial.
_WEAK_ABBREV_KEYS = ("abbrev",)
_TYPE_KEYS = ("typeId", "typeID")

# Context keys inherited down the tree while walking.
_CONTEXT_KEYS = (
    *_TEAM_ID_KEYS,
    *_SEASON_KEYS,
    *_GAME_KEYS,
    *_GAME_ABBREV_KEYS,
    *_WEAK_ABBREV_KEYS,
    *_TYPE_KEYS,
)


class DiscoveryError(RuntimeError):
    """Fan API discovery failed in a way the caller has to act on."""


@dataclass(frozen=True, slots=True)
class DiscoveredLeague:
    league_id: int
    season: int
    team_id: int | None = None
    name: str | None = None
    #: "fan_api" | "fan_api_unfiltered" | "manual"
    source: str = "fan_api"

    @property
    def key(self) -> tuple[int, int]:
        return (self.league_id, self.season)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    leagues: tuple[DiscoveredLeague, ...]
    #: False when the Fan API could not be reached or refused; manual ids still populate
    #: `leagues`, so an empty tuple plus `fan_api_ok=True` really does mean "no leagues".
    fan_api_ok: bool
    reason: str | None = None
    raw_path: Path | None = None

    def __len__(self) -> int:
        return len(self.leagues)

    def __iter__(self) -> Iterator[DiscoveredLeague]:
        return iter(self.leagues)

    def for_season(self, season: int) -> tuple[DiscoveredLeague, ...]:
        return tuple(dl for dl in self.leagues if dl.season == season)


def fan_api_url(swid: str) -> str:
    """The Fan API is addressed by SWID, braces and all."""
    braced = normalize_swid(swid)
    if not braced:
        raise DiscoveryError("no SWID: set ESPN_SWID (with braces) in the environment or .env")
    return f"{FAN_API}/{braced}"


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _first(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _as_int(value: Any) -> int | None:
    """League and team ids arrive as ints or as strings depending on the caller."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _is_football(ctx: Mapping[str, Any]) -> bool:
    """Whether the accumulated context says this record is fantasy football.

    Signals are ranked rather than OR'd. `typeId == 9` means "a fantasy team", not "a
    fantasy *football* team" -- the same account's basketball and hockey entries carry it
    too -- so it only decides when nothing more specific is present. A `gameId` or a
    `gameAbbrev` is definitive in both directions; anything weaker can confirm football but
    never deny it, so a shape that drops `gameId` still discovers leagues.
    """
    game = _as_int(_first(ctx, _GAME_KEYS))
    if game is not None:
        return game == FFL_GAME_ID
    abbrev = _first(ctx, _GAME_ABBREV_KEYS)
    if isinstance(abbrev, str):
        return abbrev.lower() in FFL_ABBREVS
    hint = _first(ctx, _WEAK_ABBREV_KEYS)
    if isinstance(hint, str) and hint.lower() in FFL_ABBREVS:
        return True
    return _as_int(_first(ctx, _TYPE_KEYS)) == FFL_PREFERENCE_TYPE_ID


def _context_from(node: Mapping[str, Any]) -> dict[str, Any]:
    return {k: node[k] for k in _CONTEXT_KEYS if k in node and node[k] is not None}


def _walk(node: Any, ctx: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield (league-bearing dict, inherited context) for every node carrying a league id.

    Context accumulates on the way down, so `entryId` and `seasonId` found on an ancestor
    still reach a `groupId` several levels below -- which is what makes the parser survive
    an inserted or renamed wrapper.
    """
    if isinstance(node, Mapping):
        merged = {**ctx, **_context_from(node)}
        if _first(node, _LEAGUE_ID_KEYS) is not None:
            yield dict(node), merged
        for value in node.values():
            yield from _walk(value, merged)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item, ctx)


def parse_fan_payload(
    payload: Any,
    *,
    require_football: bool = True,
    season: int | None = None,
    source: str = "fan_api",
) -> list[DiscoveredLeague]:
    """Pull (league_id, team_id, season) out of a Fan API body, whatever its shape.

    `require_football` applies the `typeId == 9` / `gameId == 1` filter. Turn it off only
    after the filtered pass came back empty; other ESPN fantasy sports share this endpoint
    and their league ids are worthless here.
    """
    found: dict[tuple[int, int], DiscoveredLeague] = {}
    for node, ctx in _walk(payload, {}):
        if require_football and not _is_football(ctx):
            continue
        league_id = _as_int(_first(node, _LEAGUE_ID_KEYS))
        if league_id is None:
            continue
        # The node's own season/team beat an inherited one; a group can carry its own.
        node_season = _as_int(_first(node, _SEASON_KEYS))
        ctx_season = _as_int(_first(ctx, _SEASON_KEYS))
        resolved_season = node_season or ctx_season or season
        if resolved_season is None:
            log.debug("skipping league %s: no season anywhere in its context", league_id)
            continue
        team_id = _as_int(_first(node, _TEAM_ID_KEYS)) or _as_int(_first(ctx, _TEAM_ID_KEYS))
        name = _first(node, _NAME_KEYS)

        key = (league_id, resolved_season)
        candidate = DiscoveredLeague(
            league_id=league_id,
            season=resolved_season,
            team_id=team_id,
            name=str(name) if isinstance(name, str) and name else None,
            source=source,
        )
        existing = found.get(key)
        if existing is None:
            found[key] = candidate
        else:
            # Two nodes described the same league; keep whichever fields are populated.
            found[key] = replace(
                existing,
                team_id=existing.team_id if existing.team_id is not None else candidate.team_id,
                name=existing.name or candidate.name,
            )
    return [found[k] for k in sorted(found)]


def manual_leagues(
    league_ids: Iterable[int],
    season: int,
    team_ids: Mapping[int, int] | None = None,
) -> list[DiscoveredLeague]:
    """Turn a hand-maintained id list into discovery results.

    First-class, not a consolation prize: the Fan API shape is unverified, so a registry
    that lists its league ids explicitly is the configuration that cannot break.
    """
    lookup = team_ids or {}
    return [
        DiscoveredLeague(
            league_id=int(lid),
            season=season,
            team_id=lookup.get(int(lid)),
            source="manual",
        )
        for lid in dict.fromkeys(int(i) for i in league_ids)
    ]


# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------


def _dump_raw(payload: Any, raw_path: Path | None, *, force: bool = False) -> Path | None:
    """Write the untouched body once, so a human can see the real shape.

    Contains the account's own SWID (the endpoint is addressed by it) and nothing else
    sensitive -- `espn_s2` never appears in a response. `data/` is git-ignored.
    """
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    log.info("wrote raw Fan API body to %s -- inspect it and tighten the parser", path)
    return path


def fetch_fan_payload(
    credentials: Credentials,
    *,
    client: EspnClient | None = None,
    raw_path: Path | None = FAN_RAW_PATH,
    dump_raw: bool = True,
) -> Any:
    """GET the Fan API. Raises `DiscoveryError` with a usable message on refusal."""
    if not credentials.complete:
        raise DiscoveryError(
            "Fan API discovery needs both ESPN_SWID and ESPN_S2. Without them, list your "
            "league ids under [discovery].manual_league_ids in config/leagues.toml."
        )
    assert credentials.swid is not None  # narrowed by `complete`
    url = fan_api_url(credentials.swid)

    owned = client is None
    api = client or EspnClient(**credentials.as_kwargs())
    try:
        payload, _ = api.get(url, params={"featureFlags": "expandAthlete"})
    except EspnError as err:
        # EspnClient's 404 text is written for the fantasy API ("private league, or no such
        # league"); here a 404 means the Fan API did not recognize the SWID at all.
        text = str(err)
        if text.startswith("404"):
            raise DiscoveryError(
                "Fan API answered 404 'fan not found'. The SWID is wrong or malformed -- "
                "it must include the braces exactly as the browser cookie has them."
            ) from err
        if text.startswith("401"):
            raise DiscoveryError(
                "Fan API rejected the credentials. espn_s2 lasts about a year and dies "
                "silently, usually mid-season; re-copy it from the browser."
            ) from err
        raise DiscoveryError(f"Fan API request failed: {text}") from err
    finally:
        if owned:
            api.close()

    if dump_raw:
        _dump_raw(payload, raw_path)
    return payload


def discover_leagues(
    credentials: Credentials | None = None,
    *,
    manual_league_ids: Iterable[int] = (),
    season: int | None = None,
    client: EspnClient | None = None,
    raw_path: Path | None = FAN_RAW_PATH,
) -> DiscoveryResult:
    """Every league this SWID plays in, from the Fan API plus any manual ids.

    Manual ids are merged in first and win on conflict, so a hand-entered `team_id` is not
    overwritten by a guess. If the Fan API is unreachable or unauthenticated the manual list
    still comes back, with `fan_api_ok=False` and a reason.

    `season` is a *fallback* for records that do not carry one, and the season manual ids
    belong to -- it is not a filter. Prior seasons of the same league come back too, which
    is what the backfill wants; use `DiscoveryResult.for_season` to narrow.
    """
    credentials = credentials or load_credentials()

    leagues: dict[tuple[int, int], DiscoveredLeague] = {}
    manual_ids = list(manual_league_ids)
    if manual_ids:
        if season is None:
            raise DiscoveryError("manual_league_ids needs a season; the registry keys on both.")
        for dl in manual_leagues(manual_ids, season):
            leagues[dl.key] = dl

    try:
        payload = fetch_fan_payload(credentials, client=client, raw_path=raw_path)
    except DiscoveryError as err:
        log.warning("Fan API discovery unavailable: %s", err)
        return DiscoveryResult(
            leagues=tuple(leagues[k] for k in sorted(leagues)),
            fan_api_ok=False,
            reason=str(err),
            raw_path=None,
        )

    found = parse_fan_payload(payload, require_football=True, season=season)
    if not found:
        # The football filter matched nothing. Either there really are no football leagues,
        # or the fields it keys on moved. Retry unfiltered and say so; `verify_leagues`
        # drops anything that is not a real FFL league.
        unfiltered = parse_fan_payload(
            payload, require_football=False, season=season, source="fan_api_unfiltered"
        )
        if unfiltered:
            log.warning(
                "Fan API returned %d league-shaped records but none matched the football "
                "filter (typeId==9 / gameId==1). The response shape has probably drifted; "
                "inspect %s. Taking them unfiltered -- verify before trusting them.",
                len(unfiltered),
                raw_path,
            )
            found = unfiltered

    for dl in found:
        existing = leagues.get(dl.key)
        if existing is None:
            leagues[dl.key] = dl
            continue
        # A manual id says *which* league to pull; it does not say who we are in it or what
        # it is called. `manual_league_ids` is a bare id list, so a manual entry's team_id
        # and name are always blank -- letting it win outright would throw away the one
        # thing discovery actually knows and leave `team_id=None` on a league we play in.
        # Manual keeps its provenance and any field it did set; blanks are filled.
        leagues[dl.key] = replace(
            existing,
            team_id=existing.team_id if existing.team_id is not None else dl.team_id,
            name=existing.name or dl.name,
        )

    return DiscoveryResult(
        leagues=tuple(leagues[k] for k in sorted(leagues)),
        fan_api_ok=True,
        raw_path=Path(raw_path) if raw_path else None,
    )


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def _rejection_reason(err: EspnError) -> str:
    """Say *why* a league id failed, in the terms the caller can act on.

    Measured across six league ids on 2026-09-07: a private league that really exists
    answers **401 `AUTH_LEAGUE_NOT_VISIBLE`** and an id that does not exist answers
    **404 `GENERAL_NOT_FOUND`**. `docs/RESEARCH.md` records these as indistinguishable
    without cookies; they are not, and the difference is exactly the one worth surfacing --
    "add your cookies" and "fix this id" are different jobs.
    """
    text = str(err)
    if text.startswith("401"):
        return (
            "401: the league exists but is not visible to us. Set ESPN_SWID/ESPN_S2, or "
            "re-copy espn_s2 if they are already set (it expires silently)."
        )
    if text.startswith("404"):
        return "404: no such league in this season -- wrong id, or wrong season."
    return text


def verify_leagues(
    client: EspnClient,
    leagues: Iterable[DiscoveredLeague],
    *,
    swid: str | None = None,
) -> tuple[list[DiscoveredLeague], list[tuple[DiscoveredLeague, str]]]:
    """Confirm each id is a readable FFL league; fill in its name and our team id.

    Returns (verified, rejected-with-reason). This is what makes the unfiltered parser
    fallback safe: a basketball league id, a typo in a manual list and a league whose
    cookies have expired all fail here with distinguishable messages.

    `swid` resolves which franchise is ours by matching the team's `owners`, which is more
    reliable than the Fan API's `entryId` because it is checked against the league itself.
    """
    verified: list[DiscoveredLeague] = []
    rejected: list[tuple[DiscoveredLeague, str]] = []
    owner = owner_key(swid) if swid else None

    for dl in leagues:
        league = League(client, dl.league_id, dl.season)
        try:
            settings = league.settings()
        except EspnError as err:
            rejected.append((dl, _rejection_reason(err)))
            continue

        team_id = dl.team_id
        if owner:
            try:
                mine = league.teams().team_for_owner(owner)
            except EspnError:
                mine = None
            if mine is not None:
                team_id = mine.id
        verified.append(replace(dl, name=settings.name or dl.name, team_id=team_id))

    return verified, rejected
