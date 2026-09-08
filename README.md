# fantasy_quant

Multi-league decision analytics for competitive ESPN fantasy football.

Public tools optimize projected points. This one optimizes **Δ P(championship)** — every
surface (waiver claim, trade, start/sit, streaming plan) reports in that same unit, so a
claim in one league is directly comparable to a lineup change in another, and all of your
leagues collapse into one ranked list of what to actually do.

Read-only against ESPN. It recommends; you click the buttons.

```
uv sync
cp .env.example .env          # add your SWID and espn_s2 cookies
uv run fq queue               # every league's best moves, one ranked list
uv run fq dash                # the same thing as a local dashboard
```

## What it does

| Command | |
|---|---|
| `fq queue` | every league's best moves in one list, ranked by Δ P(title) |
| `fq odds` | championship table, with the Monte Carlo error on every row |
| `fq weekly` | the full picture per league |
| `fq waivers` | the board, plus the priority threshold a claim has to clear |
| `fq trades` | searches for trades — 2-team and 3-team — rather than grading yours |
| `fq lineup` | start/sit by win probability, not expected points |
| `fq stream` | rest-of-season DST/K plan |
| `fq dash` | all of it in a browser, on 127.0.0.1 only |
| `fq snapshot` / `fq backfill` | the data corpus |

## How it works

```
ESPN league ─┬─ settings, rosters, schedule, transactions
             └─ scoring ──────────────► a callable that reproduces ESPN's own
                                        appliedTotal exactly (verified 933/933)
corpus ──────► component projections ──► ensemble ──► per-position calibration
                                                          │
                                        hurdle-gamma outlooks (mean, sd, P(zero))
                                                          │
                              correlated draw, block-diagonal by NFL team
                                                          │
                          [sim, week, player] ──► optimal lineups ──► bracket
                                                          │
                                              Δ P(championship)
                                                          │
              ┌──────────────┬──────────────┬─────────────┴──────┬──────────────┐
           waivers        trades         lineups            streaming       portfolio
```

Everything is calibrated against a corpus this repo builds: **~24,000 paired
projection-vs-actual player-weeks** pulled from ESPN, 2022–2025. `fq backfill` reproduces it
in about ten seconds.

## Some things it does that most tools don't

- **Values a player per league.** Replacement level is solved as a flex-demand fixed point
  from each league's real slot counts, so a receiver is genuinely worth different amounts in a
  14-team full-PPR league and a 12-team half-PPR one. There is no API for a league-independent
  player value, because there is no such quantity.
- **Models the distribution, not the point estimate.** Weekly outcomes are hurdle gamma
  (measured skew up to +1.8, excess kurtosis to +4.0 — normal is the wrong family), correlated
  by NFL team via a Gaussian copula. Under point projections bench depth is worth exactly zero;
  under simulation it is a weekly exchange option, which is the whole argument for simulating.
- **Knows when a decision doesn't matter.** One projected point is worth ~1.16pp of weekly win
  probability at an even matchup and ~14% of that in a blowout. "This week barely matters" is
  often the most useful thing the tool can say.
- **Flips the variance objective by standing.** A swap improves win probability iff
  `Δμ − z·Δσ > 0`. A team chasing the playoff cut should *want* variance; a lock for the bye
  should not. Same roster, same week, opposite answer.
- **Reports its own error.** Every recommendation carries the standard error of the *paired*
  CRN difference, and anything smaller than its own noise is shown as noise rather than
  dressed up as a recommendation.
- **Refuses signals that don't survive.** Behavioral traits are label-shuffle tested; five of
  nine collapsed and are refused outright rather than reported with confidence.

## Layout

```
src/fantasy_quant/
  core.py         the contracts every layer shares
  corpus.py       the one correct reader for the snapshot corpus
  pipeline.py     league id -> championship probabilities
  espn/           client, generated constants, scoring, league state, discovery
  data/           nflverse, Sleeper, sportsbook props, the ID crosswalk
  projections/    component ensemble, calibration
  sim/            distributions, lineup solver, season + bracket
  decide/         Δ P(title) engine, waivers, trades, lineups, streaming, valuation
  edges/          behavioral, market, portfolio
  analysis/       opponent-adjusted DvP, opportunity metrics
  api/            FastAPI, localhost-only
web/              Vite + React dashboard
```

## Notes

- Python 3.13 (`nflreadpy` does not support 3.14 yet).
- `.env` and `config/leagues.toml` are git-ignored; this repo is public.
- The API binds to loopback and refuses non-local peers whatever the bind address — it holds
  your ESPN session cookies.
- `docs/RESEARCH.md` records what was believed going in. The code and its tests record what
  turned out to be true; where they disagree, the tests win.
