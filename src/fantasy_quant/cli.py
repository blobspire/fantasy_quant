"""fq — command line entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .snapshot import DEFAULT_ROOT, backfill_seasons, snapshot_all

app = typer.Typer(help="fantasy_quant — multi-league fantasy football decision analytics.")
console = Console()


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def snapshot(
    season: int | None = typer.Option(None, help="Defaults to ESPN's current season."),
    root: Path = typer.Option(DEFAULT_ROOT, help="Corpus root directory."),
    variant: list[str] = typer.Option(
        None, "--variant", help="Scoring variant(s); repeatable. Default: all."
    ),
) -> None:
    """Capture the ESPN player pool to Parquet.

    Run daily. ESPN purges historical weekly projections, so a day not captured is
    a day lost permanently.
    """
    paths = snapshot_all(season=season, variants=list(variant) if variant else None, root=root)
    table = Table("variant", "rows", "path", title="snapshot written")
    import polars as pl

    for p in paths:
        height = pl.scan_parquet(p).select(pl.len()).collect().item()
        table.add_row(p.parent.name.replace("variant=", ""), f"{height:,}", str(p))
    console.print(table)


@app.command()
def backfill(
    seasons: str = typer.Option(
        "2022,2023,2024,2025", help="Comma-separated completed seasons to pull."
    ),
    variant: str = typer.Option("ppr", help="Scoring variant to capture."),
    root: Path = typer.Option(DEFAULT_ROOT, help="Corpus root directory."),
) -> None:
    """Pull completed seasons for the projection-vs-actual calibration corpus.

    Weekly projections and actuals are both retrievable for past seasons, so the
    calibration set does not have to be accumulated live.
    """
    years = [int(y) for y in seasons.split(",") if y.strip()]
    paths = backfill_seasons(years, variant=variant, root=root)

    import polars as pl

    table = Table("season", "rows", "proj wk", "actual wk", title="backfill written")
    for season, path in zip(years, paths, strict=True):
        df = pl.scan_parquet(path)
        rows = df.select(pl.len()).collect().item()
        pw = (
            df.filter(
                (pl.col("stat_season") == season)
                & (pl.col("stat_source_id") == 1)
                & (pl.col("stat_split_type_id") == 1)
            )
            .select(pl.len())
            .collect()
            .item()
        )
        aw = (
            df.filter(
                (pl.col("stat_season") == season)
                & (pl.col("stat_source_id") == 0)
                & (pl.col("stat_split_type_id") == 1)
            )
            .select(pl.len())
            .collect()
            .item()
        )
        table.add_row(str(season), f"{rows:,}", f"{pw:,}", f"{aw:,}")
    console.print(table)


@app.command()
def status() -> None:
    """Show ESPN's current season and scoring period."""
    from .espn.client import EspnClient

    with EspnClient() as client:
        season, week = client.current_season_and_week()
    console.print(f"ESPN current season [bold]{season}[/bold], scoring period [bold]{week}[/bold]")


# Reporting lives in report.py but mounts flat, so the user types `fq odds` rather
# than `fq report odds`. Imported at the bottom to keep the data-plane commands
# above independent of the analytics stack -- `fq snapshot` must keep working even
# if a decision surface fails to import.
from .api.server import mount as _mount_api  # noqa: E402
from .report import mount as _mount_report  # noqa: E402

_mount_report(app)
_mount_api(app)


if __name__ == "__main__":
    app()
