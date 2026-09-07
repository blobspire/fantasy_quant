"""Capture of the ESPN player pool.

What is and is not recoverable, measured rather than assumed:

Weekly projections paired with weekly actuals ARE retrievable for past seasons --
verified back to 2022, each from its own season endpoint (~18 projected and ~17
actual weekly rows per player). So the calibration corpus can be backfilled in one
pass; see `backfill_seasons`. A widely repeated claim that these rows are purged
appears to come from querying through a stats filter, which silently drops the
projection half of the pair. This module therefore sends no stat filter at all.

What genuinely is NOT recoverable is the daily state: percent rostered, percent
started, ADP and its drift, auction values, and injury designations are all
point-in-time and overwritten in place. Those only exist if captured as they
happen, which is what the daily run is for.

Runs unauthenticated. One file per (date, season, scoring variant), written
atomically. Never deletes.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import polars as pl

from .espn.client import EspnClient, EspnError
from .espn.endpoints import LEAGUE_DEFAULTS, league_default_url
from .espn.statrows import parse_rows

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/snapshots/espn")

# Deliberately NO stat filter.
#
# It is tempting to narrow the payload with `filterStatsForTopScoringPeriodIds`,
# and it is a trap: any form of it we tested silently drops the weekly PROJECTION
# rows and returns actuals only -- which is precisely the half of the pair the
# calibration work needs. Unfiltered returns ~57 stat rows per player covering
# both seasons, projected and actual, season and weekly. Verified against the
# live endpoint; there is a regression test pinning this.
_STAT_FILTER: dict[str, Any] | None = None


def _flatten(entry: dict[str, Any], season: int) -> list[dict[str, Any]]:
    """One player entry -> one row per stat row, with player attributes denormalized.

    Long format rather than wide: stat rows are ragged across players and seasons,
    and a long table is what every downstream calibration query wants anyway.
    """
    player = entry.get("player") or {}
    ownership = player.get("ownership") or {}
    rows = parse_rows(player.get("stats") or [])

    base = {
        "espn_id": entry.get("id"),
        "full_name": player.get("fullName"),
        "default_position_id": player.get("defaultPositionId"),
        "pro_team_id": player.get("proTeamId"),
        "injury_status": player.get("injuryStatus"),
        "injured": player.get("injured"),
        "active": player.get("active"),
        "eligible_slots": player.get("eligibleSlots"),
        "status": entry.get("status"),
        "on_team_id": entry.get("onTeamId"),
        "keeper_value": entry.get("keeperValue"),
        "keeper_value_future": entry.get("keeperValueFuture"),
        "draft_auction_value": entry.get("draftAuctionValue"),
        "percent_owned": ownership.get("percentOwned"),
        "percent_started": ownership.get("percentStarted"),
        "percent_change": ownership.get("percentChange"),
        "average_draft_position": ownership.get("averageDraftPosition"),
        "adp_percent_change": ownership.get("averageDraftPositionPercentChange"),
        "auction_value_average": ownership.get("auctionValueAverage"),
        "auction_value_change": ownership.get("auctionValueAverageChange"),
    }

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                **base,
                "stat_row_id": r.row_id,
                "stat_season": r.season,
                "stat_source_id": r.source,
                "stat_split_type_id": r.split,
                "scoring_period_id": r.scoring_period,
                "applied_total": r.applied_total,
                "stat_pro_team_id": r.pro_team_id,
                # Parallel lists, not a dict. A dict lands in Parquet as a struct
                # whose field set is whatever statIds happened to appear in that
                # file, so two days' files then refuse to concatenate.
                "stat_ids": list(r.stats.keys()),
                "stat_values": list(r.stats.values()),
            }
        )

    if not out:
        # Keep players with no stat rows so ownership history stays complete.
        out.append(
            {
                **base,
                "stat_row_id": None,
                "stat_season": season,
                "stat_source_id": None,
                "stat_split_type_id": None,
                "scoring_period_id": None,
                "applied_total": None,
                "stat_pro_team_id": None,
                "stat_ids": [],
                "stat_values": [],
            }
        )
    return out


def snapshot_variant(
    client: EspnClient,
    season: int,
    variant: str,
    root: Path = DEFAULT_ROOT,
    captured_at: dt.datetime | None = None,
) -> Path:
    """Capture one scoring variant's full pool to Parquet. Returns the path written."""
    captured_at = captured_at or dt.datetime.now(dt.UTC)
    url = league_default_url(season, variant)

    entries = client.player_pool(
        url,
        limit=250,
        extra_filter=_STAT_FILTER,
        params={"view": "kona_player_info"},
    )
    if not entries:
        raise RuntimeError(f"ESPN returned an empty pool for {variant}; refusing to write.")

    records: list[dict[str, Any]] = []
    for e in entries:
        records.extend(_flatten(e, season))

    df = pl.DataFrame(
        records,
        infer_schema_length=None,
        schema_overrides={
            "stat_ids": pl.List(pl.Utf8),
            "stat_values": pl.List(pl.Float64),
            "eligible_slots": pl.List(pl.Int64),
        },
    ).with_columns(
        pl.lit(captured_at.replace(tzinfo=None)).alias("captured_at"),
        pl.lit(season).alias("request_season"),
        pl.lit(variant).alias("scoring_variant"),
    )

    day = captured_at.date().isoformat()
    out_dir = root / f"season={season}" / f"variant={variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{day}.parquet"

    # Atomic: a partial file from an interrupted run would silently poison the corpus.
    tmp = dest.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(dest)

    log.info(
        "snapshot %s: %d players, %d stat rows -> %s",
        variant,
        len(entries),
        df.height,
        dest,
    )
    return dest


def snapshot_all(
    season: int | None = None,
    variants: list[str] | None = None,
    root: Path = DEFAULT_ROOT,
) -> list[Path]:
    """Capture every scoring variant. This is what the daily cron calls.

    Not every variant exists in every season -- `leaguedefaults/8` (half PPR) is
    2026-only and 404s for 2022-2025. A variant that is missing is skipped with a
    warning rather than aborting the run, because this is a cron job and losing
    the whole day's capture over one absent scoring preset is the worse outcome.
    """
    variants = variants or list(LEAGUE_DEFAULTS)
    written: list[Path] = []
    with EspnClient() as client:
        if season is None:
            season, _ = client.current_season_and_week()
        for variant in variants:
            try:
                written.append(snapshot_variant(client, season, variant, root=root))
            except EspnError as exc:
                log.warning("skipping variant %r for season %d: %s", variant, season, exc)
    if not written:
        raise RuntimeError(
            f"no variants captured for season {season}; tried {variants}. "
            "Every request failed -- check connectivity before assuming ESPN changed."
        )
    return written


def backfill_seasons(
    seasons: list[int],
    variant: str = "ppr",
    root: Path = DEFAULT_ROOT,
) -> list[Path]:
    """Pull completed seasons for the projection-vs-actual calibration corpus.

    Each season endpoint serves only its own season, so this is one request set per
    year. Files are dated by the season they cover rather than by capture date --
    a completed season is immutable, so re-running overwrites harmlessly.
    """
    written: list[Path] = []
    with EspnClient() as client:
        for season in seasons:
            path = snapshot_variant(
                client,
                season,
                variant,
                root=root,
                captured_at=dt.datetime(season, 12, 31, tzinfo=dt.UTC),
            )
            written.append(path)
    return written
