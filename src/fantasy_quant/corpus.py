"""The one correct way to read the snapshot corpus.

The corpus is append-only: `fq snapshot` writes a file per (season, variant, day),
so a season accumulates many captures. Most rows in them are identical -- a 2025
game that has already been played does not change -- which makes naive
concatenation silently double-count. Measured on this repo: globbing every 2024
PPR file yields 162,446 weekly rows where 131,231 are distinct, because 2024 was
captured twice.

That is a quiet failure. Nothing errors; every per-position count, MAE and
correlation simply gets computed on duplicated rows and looks plausible. So this
module exists to be the single reader, and consumers should not glob the corpus
themselves.

Two different questions need two different answers, which is why there are two
functions rather than one:

* `load_stat_rows` -- "what is true about these player-weeks?" Deduplicated to
  one row per stat identity, keeping the most recent capture, because a later
  capture supersedes an earlier one (ESPN revises weekly projections all season).
* `load_ownership_history` -- "how did the field's view change over time?" NOT
  deduplicated across captures, because the whole value of that series is that
  each day differs. Percent rostered, ADP drift and injury designations are
  point-in-time and unrecoverable if collapsed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

DEFAULT_ROOT = Path("data/snapshots/espn")

#: One row per player per stat entry. A later capture of the same identity
#: supersedes an earlier one.
STAT_IDENTITY: tuple[str, ...] = (
    "espn_id",
    "stat_season",
    "stat_source_id",
    "stat_split_type_id",
    "scoring_period_id",
)

SOURCE_ACTUAL = 0
SOURCE_PROJECTED = 1
SPLIT_SEASON = 0
SPLIT_GAME = 1
SPLIT_REST_OF_SEASON = 2


class CorpusError(RuntimeError):
    """The corpus is missing or unusable for the request."""


def corpus_files(
    seasons: Iterable[int] | None = None,
    *,
    root: Path | str = DEFAULT_ROOT,
    variant: str = "ppr",
) -> list[Path]:
    """Capture files, oldest first. Filenames are ISO dates, so they sort lexically."""
    base = Path(root)
    if not base.exists():
        return []
    wanted = set(seasons) if seasons is not None else None
    out: list[Path] = []
    for season_dir in sorted(base.glob("season=*")):
        try:
            season = int(season_dir.name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        if wanted is not None and season not in wanted:
            continue
        out.extend(sorted((season_dir / f"variant={variant}").glob("*.parquet")))
    return out


def available_seasons(*, root: Path | str = DEFAULT_ROOT, variant: str = "ppr") -> list[int]:
    files = corpus_files(root=root, variant=variant)
    return sorted({int(p.parent.parent.name.split("=", 1)[1]) for p in files})


def load_stat_rows(
    seasons: Iterable[int] | None = None,
    *,
    root: Path | str = DEFAULT_ROOT,
    variant: str = "ppr",
    own_season_only: bool = True,
) -> pl.DataFrame:
    """Every stat row, deduplicated to one per identity, latest capture winning.

    `own_season_only` drops rows whose `stat_season` differs from the season file
    they came from. A 2026 request returns 2025 rows in the same array, so without
    this the same 2025 week arrives from several season files at once.
    """
    files = corpus_files(seasons, root=root, variant=variant)
    if not files:
        raise CorpusError(
            f"no corpus under {root} for variant {variant!r}"
            + (f" seasons {sorted(seasons)}" if seasons is not None else "")
            + "; run `fq backfill` (past seasons) or `fq snapshot` (current)"
        )
    frame = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    if own_season_only:
        frame = frame.filter(pl.col("stat_season") == pl.col("request_season"))
    # Sort by capture time so `keep="last"` really is the newest observation;
    # do not rely on concat order, which follows the glob.
    return frame.sort("captured_at").unique(subset=list(STAT_IDENTITY), keep="last")


def load_weekly_pairs(
    seasons: Iterable[int] | None = None,
    *,
    root: Path | str = DEFAULT_ROOT,
    variant: str = "ppr",
    positions: Sequence[int] | None = (1, 2, 3, 4),
    positive_projections_only: bool = True,
) -> pl.DataFrame:
    """Projected vs actual, one row per played player-week.

    `positive_projections_only` drops rows projected at zero. This is not
    cosmetic: the pool carries ~20k deep-bench players projected at ~0 who never
    play, and including them halves every MAE and triples every observed
    zero-rate. The published calibration constants are all computed on the
    positive-projection subset, and dropping this filter silently changes what
    the numbers mean.
    """
    rows = load_stat_rows(seasons, root=root, variant=variant)
    weekly = rows.filter(
        (pl.col("stat_split_type_id") == SPLIT_GAME) & (pl.col("scoring_period_id") > 0)
    )
    if positions is not None:
        weekly = weekly.filter(pl.col("default_position_id").is_in(list(positions)))

    key = ["espn_id", "default_position_id", "stat_season", "scoring_period_id"]
    proj = (
        weekly.filter(pl.col("stat_source_id") == SOURCE_PROJECTED)
        .select([*key, "full_name", pl.col("applied_total").alias("projected")])
        .unique(subset=key, keep="last")
    )
    actual = (
        weekly.filter(pl.col("stat_source_id") == SOURCE_ACTUAL)
        .select([*key, pl.col("applied_total").alias("actual")])
        .unique(subset=key, keep="last")
    )
    paired = proj.join(actual, on=key, how="inner")
    if positive_projections_only:
        paired = paired.filter(pl.col("projected") > 0)
    return paired.sort(["stat_season", "scoring_period_id", "espn_id"])


def load_ownership_history(
    seasons: Iterable[int] | None = None,
    *,
    root: Path | str = DEFAULT_ROOT,
    variant: str = "ppr",
) -> pl.DataFrame:
    """Point-in-time roster/ADP state, one row per player per capture.

    Deliberately NOT deduplicated across captures. Percent rostered, ADP and its
    drift, auction value and injury designation are overwritten in place by ESPN,
    so this series only exists because we captured it daily. Collapsing it would
    throw away the only genuinely unrecoverable data in the corpus.
    """
    files = corpus_files(seasons, root=root, variant=variant)
    if not files:
        raise CorpusError(f"no corpus under {root} for variant {variant!r}")
    frame = pl.concat([pl.read_parquet(f) for f in files], how="diagonal")
    cols = [
        "espn_id",
        "full_name",
        "default_position_id",
        "pro_team_id",
        "captured_at",
        "request_season",
        "percent_owned",
        "percent_started",
        "percent_change",
        "average_draft_position",
        "adp_percent_change",
        "auction_value_average",
        "auction_value_change",
        "injury_status",
        "injured",
        "status",
        "on_team_id",
    ]
    present = [c for c in cols if c in frame.columns]
    return (
        frame.select(present)
        .unique(subset=["espn_id", "captured_at", "request_season"], keep="last")
        .sort(["captured_at", "espn_id"])
    )
