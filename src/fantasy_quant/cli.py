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


@app.command()
def doctor() -> None:
    """Check the things that fail silently: cookies, the cron, and ESPN's schema.

    Exits non-zero when something needs a human, so it can run from cron next to
    the snapshot and actually tell you.
    """
    import sys

    from .doctor import run as run_checks

    checks = run_checks()
    table = Table("check", "", "detail", title="fq doctor")
    for c in checks:
        table.add_row(c.name, "[green]ok[/green]" if c.ok else "[red]FAIL[/red]", c.detail)
    console.print(table)
    broken = [c for c in checks if not c.ok and c.fatal]
    if broken:
        console.print(f"[red]{len(broken)} problem(s) need attention.[/red]")
        sys.exit(1)
    console.print("[green]all good.[/green]")


@app.command()
def managers(
    league: str | None = typer.Option(None, help="League id or name; default all configured."),
    season: int = typer.Option(2026),
) -> None:
    """Per-manager tendencies from the transaction log: who to trade with, who to ignore.

    Most published fantasy "manager bias" analysis does not survive a permutation
    test, and neither did most of ours: five of nine traits collapsed when the
    labels were shuffled and are refused outright rather than reported. What is
    printed here is what survived, with its signal-to-noise. Read the refusals as
    findings too -- "we cannot tell" is the honest answer for a trait measured on
    thirteen draft picks.
    """
    from .edges.behavioral import behavioral_report, build_panel
    from .pipeline import client_from_env
    from .registry import Registry

    registry = Registry.load()
    targets = [c for c in registry.active() if league in (None, str(c.league_id), c.name)]
    if not targets:
        console.print(f"[red]no league matching {league!r} in config/leagues.toml[/red]")
        raise typer.Exit(1)

    client = client_from_env()
    try:
        for cfg in targets:
            console.rule(f"{cfg.name} ({cfg.league_id})")
            try:
                panel = build_panel(cfg.league_id, season, client=client)
                report = behavioral_report(panel)
            except Exception as exc:  # a league with no history is normal, not fatal
                console.print(f"  [yellow]unavailable: {exc}[/yellow]")
                continue

            usable = getattr(report, "usable_traits", ()) or ()
            marginal = getattr(report, "marginal_traits", ()) or ()
            refused = getattr(report, "refused_traits", ()) or ()
            console.print(
                f"  usable: {', '.join(map(str, usable)) or 'none'}\n"
                f"  marginal (read, do not act): {', '.join(map(str, marginal)) or 'none'}\n"
                f"  refused as noise: {', '.join(map(str, refused)) or 'none'}"
            )
            actions = getattr(report, "actions", ()) or ()
            if actions:
                table = Table("manager", "read", title="what to do about it")
                for a in actions:
                    table.add_row(str(getattr(a, "manager", "?")), str(getattr(a, "note", a)))
                console.print(table)
            else:
                console.print("  [dim]no per-manager read clears its own error yet.[/dim]")
    finally:
        client.close()


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
