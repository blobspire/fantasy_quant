"""Health checks for the things that fail silently.

Three failures in this system are quiet rather than loud, which is what makes
them worth a dedicated command:

* **`espn_s2` expires**, roughly yearly and usually mid-season. Nothing announces
  it; private-league reads simply start returning 401 and any surface that
  degrades gracefully will report a partial answer instead of an error.
* **The snapshot stops running.** Percent rostered, ADP drift and injury
  designations are point-in-time and unrecoverable, so a cron that died in
  October is a hole in the corpus that cannot be backfilled at any price.
* **ESPN moves the schema.** `NFL_MIGRATION` has been live in the 2026 settings
  all season, and a view that starts returning a skeleton answers HTTP 200.

Run it from cron alongside the snapshot; it exits non-zero when something needs
a human.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from . import corpus
from .espn.client import EspnClient, EspnError
from .espn.endpoints import league_url
from .registry import Registry

#: A snapshot older than this means the cron is not running. Two days rather than
#: one so a single missed night is not an alarm.
STALE_SNAPSHOT_HOURS = 48


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False


def check_credentials(client: EspnClient) -> Check:
    if not client.authenticated:
        return Check(
            "credentials",
            False,
            "no ESPN_SWID/ESPN_S2 in the environment; private leagues will 401. "
            "Copy them from DevTools -> Application -> Cookies -> espn.com into .env.",
            fatal=True,
        )
    return Check("credentials", True, "SWID and espn_s2 are set")


def check_league_access(client: EspnClient, registry: Registry) -> list[Check]:
    """The real auth test: can we actually read each configured league?

    A cookie that is present but expired looks identical to a good one until it
    is used, so this reads rather than inspects.
    """
    out: list[Check] = []
    for cfg in registry.active():
        try:
            client.get(league_url(cfg.season, cfg.league_id), params={"view": "mTeam"})
            out.append(Check(f"league {cfg.league_id}", True, f"{cfg.name}: readable"))
        except EspnError as exc:
            expired = "401" in str(exc) or "credentials" in str(exc).lower()
            out.append(
                Check(
                    f"league {cfg.league_id}",
                    False,
                    f"{cfg.name}: {exc}"
                    + (" -- espn_s2 has most likely expired; re-copy it." if expired else ""),
                    fatal=True,
                )
            )
    return out


def check_snapshot_freshness(root: Path | str = corpus.DEFAULT_ROOT) -> Check:
    files = corpus.corpus_files(root=root)
    if not files:
        return Check(
            "snapshot", False, "no corpus at all; run `fq backfill` then `fq snapshot`", fatal=True
        )
    newest = max(files, key=lambda p: p.stat().st_mtime)
    age = dt.datetime.now() - dt.datetime.fromtimestamp(newest.stat().st_mtime)
    hours = age.total_seconds() / 3600
    if hours > STALE_SNAPSHOT_HOURS:
        return Check(
            "snapshot",
            False,
            f"newest capture is {hours:.0f}h old ({newest.name}). Ownership, ADP and injury "
            "designations are point-in-time and cannot be backfilled -- every day the cron is "
            "down is a permanent hole. Check your crontab.",
            fatal=True,
        )
    return Check("snapshot", True, f"newest capture {hours:.1f}h old ({newest.name})")


def check_schema(client: EspnClient) -> Check:
    """Does the player pool still answer with the shape we parse?

    ESPN returns HTTP 200 with a skeleton for an unrecognised view, so a status
    code proves nothing; this asserts on the content.
    """
    from .espn.endpoints import league_default_url

    try:
        players = client.player_pool(
            league_default_url(2026, "ppr"),
            limit=5,
            max_players=5,
            params={"view": "kona_player_info"},
        )
    except EspnError as exc:
        return Check("schema", False, f"player pool unreadable: {exc}", fatal=True)
    if not players:
        return Check("schema", False, "player pool came back empty", fatal=True)
    stats = players[0].get("player", {}).get("stats") or []
    weekly = [s for s in stats if s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 1]
    if not weekly:
        return Check(
            "schema",
            False,
            "no weekly PROJECTION rows in the player pool. This is the documented trap: a "
            "stats filter silently drops them and ESPN answers 200 either way. Check that no "
            "filterStatsForTopScoringPeriodIds crept back into snapshot.py.",
            fatal=True,
        )
    return Check("schema", True, f"{len(weekly)} weekly projection rows on the first player")


def run(root: Path | str = corpus.DEFAULT_ROOT) -> list[Check]:
    from .pipeline import client_from_env

    client = client_from_env()
    try:
        checks = [check_credentials(client), check_snapshot_freshness(root), check_schema(client)]
        registry = Registry.load()
        if len(registry):
            checks.extend(check_league_access(client, registry))
        else:
            checks.append(Check("registry", False, "no leagues configured in config/leagues.toml"))
        return checks
    finally:
        client.close()
