# fantasy_quant

Multi-league fantasy football decision analytics for competitive ESPN redraft leagues.

Public tools optimize projected points. This one optimizes **Δ P(championship)** — every
surface (trade, waiver claim, start/sit) reports in that same unit, so recommendations are
directly comparable across all of your leagues at once.

## Status

Phase 0 — the ESPN snapshotter. See [the plan](docs/PLAN.md) for the full roadmap.

## Why the snapshotter ships first

ESPN purges historical weekly projections and does not backfill them. The 2025 weekly rows
are already unrecoverable at any price. Every calibration constant this system depends on —
projection-vs-actual variance, per-position hurdle rates, correlation blocks — comes from
pairing weekly projections against weekly actuals, so the corpus has to accumulate live.

A day not captured is a day lost permanently.

## Quickstart

```bash
uv sync
uv run fq status      # ESPN's current season and week
uv run fq snapshot    # capture the pool; writes data/snapshots/espn/...
```

Phase 0 needs no credentials. Private-league features (Phase 1+) need `SWID` and `espn_s2`
cookies — copy `.env.example` to `.env` and fill them in.

## Daily capture

```bash
# crontab -e — 6am local, before waiver processing
0 6 * * * cd /path/to/fantasy_quant && /path/to/uv run fq snapshot >> data/snapshot.log 2>&1
```

## Layout

```
src/fantasy_quant/
  espn/          endpoints, read client, stat-row parsing
  snapshot.py    the daily corpus capture
  cli.py         fq
```

## Notes

Read-only against ESPN. It recommends; you click the buttons.
