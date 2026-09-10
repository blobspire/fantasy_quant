# The first-principles audit: findings, fixes, and what is left

Status as of 2026-09-10. Four of nine verified findings are fixed and pushed; five remain,
all located, none started. Read the **Remaining work** section to continue.

## Why this file exists

A seven-subsystem adversarial audit verified **nine findings** and correctly dropped a tenth
(a trade-budget effect that was a max-of-N artifact: +0.028pp mean with the sign flipping
across two seeds). The findings are not a random scatter. They share one shape:

> **a component computing something defensible-looking that is not the right quantity.**

None were caught by tests, because in every case the code did correctly what it was written
to do. More unit tests will not find the next one. Two lessons worth carrying:

- **Extracting the canonical implementation is necessary, not sufficient.** `decide/wire.py`
  was extracted so the replacement floor would have exactly one definition. The bug came
  back anyway, because the *caller* handed it the wrong input (a pool with no free agents in
  it). One definition does not help if the inputs disagree.
- **The tell is usually an invariant, not a number.** The D/ST floor came back as 4.704 in
  all three leagues. A league-specific floor cannot be league-independent. That was visible
  long before anyone measured the floor itself.

---

## Fixed

### #1 — the engine never saw the wire (`cc624f6`)

`streaming_levels` read the wire from `_outlooks_from_panel(draw, state)`, but `pipeline.build`
pools only rostered players, so the read returned 0.00 at all seven slots in all three leagues
and fell through to the VOLS roster-bottom fallback — the exact bug `decide/wire.py` exists to
prevent. Fix was one argument: `from_sim` passes `sim.outlooks`.

Before: Wine RB floor 9.22 against a true 4.55. On Blacksburg's drop board Harrison Butker (K)
priced at −0.925pp against Luther Burden III (WR) at −0.375pp; the truth is −0.375 and −2.200.

### #4 + #7 — K/DST calibration and D/ST byes (`197490f`)

**Had to land together.** They hit the same 12–15 assets and point opposite ways.

`calibration.load_pairs` filtered the corpus to QB/RB/WR/TE before anything was fitted, so
`for_position` fell K and D/ST through to the pooled *skill* line. For a defence that is the
wrong **sign**: pooled shrinks a projection (slope 0.9414) where a defence must be expanded
(1.4237). Separately, `pipeline.build` passed no bye table, so `has_game` was True in all
seventeen weeks — ESPN zeroes skill players and kickers on a bye but projects 31 of 32
**defences** normally on theirs (Rams 7.46 against a 6.57 season mean).

Measured on Blacksburg, season points per rostered asset:

| | D/ST | K |
|---|---|---|
| before (both bugs) | 97.21 | 137.22 |
| bye fix only | 91.58 | 137.04 |
| calibration fix only | 115.95 | 147.53 |
| both fixed | 109.26 | 147.24 |

**The acceptance criterion is bias, not MAE, and here they disagree.** Leave-one-season-out at
D/ST, MAE prefers no correction at all (4.7058 raw against 4.7193 fitted) while bias prefers
the fit by two orders of magnitude (−0.6715 against −0.0034). The simulator draws from a
hurdle gamma whose mean is exactly the calibrated value and sums nine starters over seventeen
weeks: a level error is paid every week in the same direction; a per-week absolute error
cancels. `test_calibration_improves_held_out_bias` is the acceptance test now; the MAE one is
the report. Both print.

**The kicker's slope is pinned at 1.0 deliberately.** Its level fit has r² 0.014 and the
fitted slope swings 0.67–1.22 across held-out seasons; applying it makes the held-out *slope*
calibration worse (1.414 against a raw 1.018, target 1.0). `_fit_level` drops to a level shift
below `LEVEL_SLOPE_MIN_R2 = 0.05`, picked from the measured gap: K 0.014, D/ST 0.112, QB 0.211.

Independent corroboration that 1.4237 is real: `streaming.MATCHUP_MODELS[DST]` fitted the same
quantity off the same discarded rows and got **1.440**. A test pins them together.

**Still approximated, on purpose:** 14.0% of D/ST weeks are strictly negative and a hurdle
gamma cannot draw one. `HurdleCurve` fits `P(actual ≤ 0)`, so the negative mass lands on the
zero spike and the mean and SD stay exact by construction; only the third moment is wrong.
Fixing it means a location-shifted family, which would touch CRN and every other position.

### #3 — waivers charged priority to players who cost nothing (`57ecd25`)

One line decided everything:

```python
claims = tuple(r for r in finished if r.delta_title > 0.0 and r.delta_title >= threshold)
```

and every row came from `free_agent_pool`, which meant nothing more than "not on a roster in
this league". The distinction did not exist anywhere upstream either: `PlayerOutlook` had no
status, `FreeAgent` had no status, `League` had no player-pool method at all, and the corpus's
`status` column comes from the league-**independent** `leaguedefaults` pool and is read by
nobody.

| league | threshold | claims | free adds | suppressed |
|---|---|---|---|---|
| Blacksburg | 0.163pp | 0 | 23 | 19 |
| Wine Wednesday | 0.140pp | 0 | 27 | **24** |
| Type shi | 0.000pp | 0 | 12 | 0 |

Note `claims` is zero everywhere: nothing that actually costs a claim is worth one. That was
always true and was being reported as "hold everything", which is a different statement.

Three layers: `League.availability()` (one request; `filterStatus: ["WAIVERS"]` for the ids, a
`limit=1` FREEAGENT probe for the count header), `FreeAgent.on_waivers` defaulting to `True`
(unknown availability must stay *expensive* — the reverse default turns "ESPN did not answer"
into an irreversible spend on no evidence), and a separate `free_adds` tuple rather than
merging into `claims`, because `_SURFACE_COST`, the queue's `actionable` and
`portfolio._waiver_recs` all price a claim at "waiver priority".

**The timing advice was also wrong, for all three leagues.** Every row ended with a hardcoded
"submit as late as Tuesday night allows, ESPN processes around 3-4am ET Wednesday".
`AcquisitionConfig` has carried `waiverHours` / `waiverProcessDays` / `waiverProcessHour` all
along with zero readers, and it says these leagues run a **24-hour** period and process on
**six** days at hour 11 — and **Tuesday is the one day none of them process**.

One thing I got wrong first and caught on the live board: the free adds are **alternatives,
not a shopping list**. 23 of them dropping the same two players, each priced on its own
against today's roster, is the same submodularity the claim waterfall already warns about.

### #2 — the screen could not tell week 16 from week 3 (`5ef67b5`)

`_screen_one` averaged a `(weeks,)` delta over the calendar, and the surface was fitted by
broadcasting a *scalar* `d_mu` across every week. Neither half could represent a playoff-only
gain.

| league | +17 pts spread evenly | +17 pts in the bracket |
|---|---|---|
| Blacksburg | +1.29pp | +3.08pp |
| Wine Wednesday | +0.93pp | +1.92pp |
| Type shi | +1.54pp | +4.16pp |

`_design` gained a third axis `p` (extra points a week in bracket weeks only), ten terms, with
**the playoff terms appended last** so indices 0–5 keep their meaning. Paid for by coarsening
the mu grid to 2-point steps — measured, 16 nodes reproduce 31 to rmse 0.00314 vs 0.00322.

`acceptable_only` pruned on the screen's own signs before confirming; it now **reorders**
rather than deletes.

> **Scope, honestly:** `TitleEngine.screen`, `evaluate` and the surrogate have **no caller in
> `src/` today**. `waiver_board` deliberately does not route through this engine by default,
> `decide/trades.py` has its own screen, and `report.py` builds a `TitleEngine` only for
> `week_leverage`, which never touches the fit. This is a latent correctness fix, not a change
> to what any surface prints today. The audit ranked it CRITICAL partly on the premise that it
> changes recommendations. It does not, yet.

### Also fixed along the way

- Three live-market tests asserting more than the market promises (`7ea0553`). Pre-existing
  failures, unrelated to this work. All three asserted a per-item exactness the module itself
  does not claim; now judged on distributions. See the commit for the measurements.
- An off-by-one in the live D/ST streaming test: my own defence is a candidate, so the ceiling
  is `size - 1` rostered, not `size`. It only passed while some rival carried no defence.

---

## Remaining work

Ranked. Everything below is located and measured; none is started.

### #5 MAJOR — `streaming.evaluate` prices against an empty seat

Compares the streaming plan to a seat scoring nothing rather than to the wire floor, inflating
every plan by roughly the floor. **D/ST +3.33 → +1.85pp; the kicker plan flips sign in two of
three leagues.** Streaming still ranks #1 in all three after the fix.

Same root cause as #1 — "what does an empty seat score?" had five live answers. Route it
through `decide/wire.wire_levels` like every other surface. Confirm the exact call site before
editing; `streaming.py:1841` is already the one production path that passes byes, so this
module is close to correct and the change should be small.

Correction to report to the user when this lands: I previously told them "the top line
changes". It does not — streaming stays #1 in all three leagues; the magnitude halves
(+1.85/+1.62/+2.45pp) and the **kicker** plan is what flips sign.

### #6 MAJOR — the bench-hoarding prior is never overridden

`DEFAULT_BENCH_HOARDING` (`valuation.py:89`) sums to **1.80** against a measured **7.07–7.25**
on the user's real rosters. The module docstring at `valuation.py:46-52` already says so and
names the fix. `bench_hoarding_from_rosters` (`valuation.py:1271`) exists and **has no caller
outside its own module.**

The single live call is `api/server.py:439`:

```python
valued = valuation_mod.value_league(ctx, sim.outlooks, from_week=from_week or 1)
```

— no `bench_hoarding`, so the dashboard's `ros_vorp` sort runs on a bench a quarter of the real
size, which reorders players *across* positions. Wire the measured value through.
`test_valuation.py:1148` pins the constant's sum, not the caller, and stays valid.

### #8 MAJOR — `counterparty-loses` is a coin flip printed as fact

`trades.py:1515`: `any(i.delta_title < 0.0 for i in ev.impacts if i.team_id != team)` — a bare
sign test on a noisy paired estimate that two seeds disagree about 75% / 60% of the time.
`report.py:402` renders it as a flat assertion.

`TeamImpact.delta_title_stderr` is already populated (`trades.py:1465`) and the module already
computes a selection-adjusted `z` (`trades.py:1490`). Gate the tag on that, and give the
caveat three states rather than one: *falls*, *cannot tell*, *does not fall*.

### #9 MAJOR — `settle`'s forced cut is decided by ESPN's roster order

`trades.py:982-990` picks the cut by greedy leave-one-out with `if value > best_value`, which
keeps the **first** maximum. **100% of forced cuts have ≥2 bit-exact ties** — deep-bench
players who never start contribute exactly zero — so the player cut is whoever ESPN happened to
list first. Break ties on a meaningful key (lowest playoff-weighted ROS value, then lowest
player id) and surface the tie count rather than hiding it.

### Leftovers

- **`trades.wire_pool:718`** — `rostered = set(state.pool.player_ids)` treats the whole pool as
  rostered. Accidentally right under `pipeline.build` (which pools only rostered players) and
  wrong the moment `waivers.augment` or `sim_with_free_agents` widens it: every free agent then
  counts as rostered and the wire goes empty. Union over `state.franchises` instead, which is
  what `waivers._all_rostered` already does.
- **`floor_noise` is not threaded into `waivers._lineup_scores` (`:388`) or
  `portfolio._column` (`:367`)**, so those two surfaces still credit an empty seat a constant
  where `title` credits a draw. Same shape as #1.
- **`edges/portfolio._floors_for:216` duplicates `title._floors_for:1001`** — extract one
  shared helper. `tests/test_portfolio.py:664` is deliberately written to fail when
  `leave_one_out` is fixed; resolve both together.
- **Unify the `championship_table` baseline** (user already approved). `fq odds` scores an
  empty slot at zero while every recommendation surface floors it at the wire — two baselines
  for one league, disclosed today in a CLI footer ("baseline: championship_table (unfilled slot
  scores zero)"). Expect the headline odds to move slightly down.
- **`omitted` → `(weeks,)`** — step 7 of the stochastic-floor plan, never done. Promote the
  scalar in all three copies so an always-streamed slot carries a streamer's variance.

### Raised by this work, needs a decision rather than a patch

**`decide/trades.PLAYOFF_WEIGHT = 1.2`** is a hand-set constant for a quantity the surrogate
now measures: `SurrogateFit.playoff_premium` reads **1.34–1.44×** on the three live leagues.
It moves the Pareto gate, so it is the user's call, not a silent change.

---

## How to work on this

- `uv run pytest -q -m "not network"` — 1,883 offline tests, ~70s.
- `uv run pytest -q -m network` — hits ESPN and FanDuel with the real credentials, ~4.5 min.
  Live-market tests are inherently a little flaky; judge on distributions, not single items.
- `uv run ruff check src tests` before every commit.
- `npm --prefix web run build` and `npx tsc --noEmit` (from `web/`) after touching the dashboard.
- **Every change needs a negative control** that proves it did what it claimed and nothing
  else. The pattern used throughout: run the new code path with the new input disabled and
  assert the output is byte-identical to before.
- Measure on the real leagues, not just the fixture. `Registry.load().active()` gives all three;
  `cfg.team_id` and `cfg.scoring_variant` are the field names (not `my_team_id` / `variant`).
