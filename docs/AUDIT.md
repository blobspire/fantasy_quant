# The first-principles audit: findings, fixes, and what is left

Status as of 2026-09-10. Eight findings are fixed and pushed -- six from the original nine,
plus two the audit did not have. Three remain, all located. Read the **Remaining work**
section to continue.

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

### #1 again, in the portfolio (`edges/portfolio.py`)

**Not one of the nine.** Found while reading for #5. `build_portfolio` fitted its floor with

```python
floors = streaming_replacement(sim.state, sim.draw) if stream_replacement else None
```

-- no `outlooks=`. That is the exact omission `cc624f6` fixed elsewhere, in the one surface
whose whole purpose is ranking a move in one league against a move in another, and
`title.streaming_levels` warns this caller by name: *"Callers holding `sim.outlooks` must
pass it."* Without it the pool comes from `_outlooks_from_panel`, `pipeline.build` pools only
rostered players, the wire reads 0.00 at every slot and the board falls through to
`_vols_replacement`. `sim` was bound six lines above.

The tell was an invariant again, and a sharper one than "identical across leagues":
**Blacksburg and Type shi came back byte-identical at six of seven slots** -- 15.286786281772235
at QB, 8.805586651381187 at RB -- because reading the demand-rank player off a *rostered* pool
reads the same NFL players in every league. Wine Wednesday differed only in size.

Per-slot floor, week 1 of 2026:

| slot | Blacksburg | | Wine Wednesday | | Type shi | |
|---|---|---|---|---|---|---|
| | before | after | before | after | before | after |
| QB | 15.29 | 14.01 | 15.10 | 13.59 | 15.29 | 14.01 |
| RB | 8.81 | 5.57 | **9.22** | **4.55** | 8.81 | 4.98 |
| WR | 8.73 | 6.31 | 9.91 | 5.25 | 8.73 | 5.77 |
| TE | 6.56 | 6.13 | 7.78 | 7.68 | 6.56 | 6.74 |
| D/ST | **5.29** | **7.39** | 5.29 | 7.26 | 5.29 | 7.45 |
| K | 8.16 | 8.82 | 8.14 | 8.68 | 8.14 | 8.87 |
| FLEX | 8.16 | 6.69 | 9.24 | 7.71 | 8.16 | 6.82 |

RB 9.22 against a true 4.55 is `decide/wire.py`'s own docstring figure, arrived at from the
other direction. **The error is not a level shift and does not cancel:** skill floors were too
high, K and D/ST too low, so it reorders *across* positions.

What it moved:

| | before | after |
|---|---|---|
| Blacksburg P(title) | 5.30% | 6.83% |
| Wine Wednesday | 3.17% | 4.40% |
| Type shi | 7.42% | 9.38% |
| P(>=1 title) | 15.07% | **19.10%** |

and **39 of 40 rows of `exposures` change rank**. Harrison Butker sat 11 places above where he
belongs (14 -> 25, 0.80pp -> 0.40pp) and the Chargers D/ST five (11 -> 16); Jalen Hurts rose
6 -> 4 (3.60 -> 6.60pp) and Justin Jefferson 8 -> 6. That is the kicker-over-a-first-round-back
inversion `decide/wire.py` exists to prevent, arriving through the caller rather than the
definition.

> **Scope, honestly:** the ranked **action queue is byte-identical** before and after, top to
> bottom. Every surface adapter (`_waiver_recs`, `_trade_recs`, `_lineup_recs`,
> `_streaming_recs`) takes `stake.sim` and rebuilds its own draw and its own floors, exactly as
> `QueueItem.baseline_title` documents. So this is a *levels and exposures* fix, not a change
> to what the queue tells the user to do today. `_starting_shares` also moves (5/12, 3/10 and
> 3/14 players change recorded slot or share), and that feeds `exposures`.

Nothing caught it because `tests/test_portfolio.py` put **every** synthetic player on a roster,
so the panel and the wire were the same set and the two calls could not disagree. The fixture
now takes `wire_strength=` and grows free agents that are in `outlooks` and absent from
`state.pool`; the regression guard drives `build_portfolio` itself rather than the test helper,
because the helper mirrored the bug and would have passed either way.

### The empty-slot-group guard lived at the wrong altitude

**Also not one of the nine**, and it is the reason #5 could not be done first. The audit
listed "the duplicated `_floors_for`" as a tidiness leftover. It was not tidiness: the guard
was missing from `sim/season._floors`, which is the one place every floor in the system is
built, so it had been reimplemented twice on top -- `title.TitleEngine._floors_for` and
`portfolio._floors_for` -- and any caller that reached `_floors` directly got no guard at all.

`lineup.monotone_floor` raises a slot group's floor to that of every group whose eligible set
it *contains*, and eligibility is computed against this roster. A roster with nobody at a
position leaves that group's eligible set empty, and the empty set is contained in every
other. Measured on Blacksburg minus its only quarterback:

```
floor_slot_ids      (QB, RB, WR, TE, DST, K, FLEX)
unguarded groups    [14.007, 14.007, 14.007, 14.007, 14.007, 14.007, 14.007]
guarded groups      [ 0.0,    5.569,  6.312,  6.135,  7.386,  8.819,  6.692]

   unguarded: 147 of 153 slot-weeks left EMPTY (96.1%)
     guarded:  36 of 153 slot-weeks left EMPTY (23.5%)
```

The tell again: **seven different slot groups, one identical number.** The whole team is
benched behind a phantom quarterback floor.

The guard now lives in `_floors`, which returns the points a week it holds out as a fourth
value; `_franchise_scores` adds them back. Both private copies are deleted.

**The negative control is the whole point of this one.** A digest of every guarded path on the
three live leagues -- portfolio `champions`, `weekly`, `scores`, `replacement`, `starting`,
three `champions_without` counterfactuals per league, `TitleEngine._base_scores` and
`_roster_moments` on full and QB-less rosters -- is **byte-identical** before and after.

Exactly one path moves, and it is the one that never had a guard:
`TitleEngine._hindsight_moments` read `self.replacement` raw while its two siblings went
through `_floors_for`. On a roster with an empty group it floored every slot at the missing
position's level, so 96% of them were left empty and the Clark bonus it exists to compute came
back very nearly zero. Per-season starting mean, roster minus QB: 1611.17 -> 1823.64
(Blacksburg), 1779.94 -> 1885.72 (Wine Wednesday), 1714.54 -> 1838.81 (Type shi). It is a
diagnostic -- reached only through `screen(hindsight_max=True)` and `agreement()`, neither of
which any production surface calls -- so this is a latent fix, not a change to what anything
prints.

`tests/test_portfolio.py` carried a test written to fail the moment `leave_one_out` was fixed.
It fired: `title_added` for the only quarterback went from **-0.5075** -- deleting him *raises*
the title odds by more than half -- to **+0.1025**. It is now a test that the two paths agree.

### #5 — the streaming plan was priced against a seat that scored nothing

`build_grid` floored an unfilled streamed slot at **zero** and `evaluate` passed no
`replacement` to `S.team_week_scores` at all, so every unfilled seat in the simulated league
was paid nothing. Every plan was therefore credited with the replacement level it would have
collected by doing nothing. On Blacksburg's D/ST grid the hold baseline read **100.8 points
against a wire paying 149.5**.

| | D/ST before | after | K before | after |
|---|---|---|---|---|
| Blacksburg | +4.075pp | **+1.550pp** ±0.444 | +0.800pp | **+0.025pp** ±0.288 |
| Wine Wednesday | +4.250pp | **+1.250pp** ±0.342 | +0.250pp | **+0.175pp** ±0.307 |
| Type shi | +5.825pp | **+0.675pp** ±0.492 | +1.050pp | **−0.200pp** ±0.384 |

**Corrections to what the audit and I said before.** Two of them:

1. *"Streaming still ranks #1 in all three"* — it does, but the action queue never depended on
   this: `move.kind` is `hold` with `no-action-this-week` in all six cases, before and after.
   What moved is the **price**, not the advice.
2. *"The kicker plan flips sign in two of three leagues"* — it flips in **one**. In the other
   two it collapses to a fraction of its own standard error. The honest statement is stronger
   than the audit's: **no kicker plan is distinguishable from zero in any league**, and one is
   negative. D/ST survives in two of three; Type shi's `+0.675 ± 0.492` no longer clears.

**The floor has to be in the grid's units, and that is not a detail.** `wire_levels` ranks the
raw calibrated projection; `grid.value` is the matchup model's conditional expectation, and
`_reward` compares a candidate against `grid.floor` directly. Same definition, same bodies,
same depth — extracted as `wire.kth_best_index` so there is still one — but measured live the
two differ by **−0.97 to −1.01 points a week at D/ST** (the market term re-spreads the top of
the field) and **+0.09 to +0.10 at K** (no usable market term, so the 0.32 shrink dominates).

The tell was an impossibility. A floor meaning "the second-best body on this wire" can never
exceed the best body the plan may sign. Against the raw-projection floor the grid's best value
fell *below* it in 1 of 17 D/ST weeks and **up to 7 of 17 kicker weeks** — and those spurious
"leave it empty" weeks are most of what the audit saw as the kicker flipping sign. Against the
grid-units floor it is 0 of 17 everywhere.

**Three things the fix dragged out that were not in the audit:**

- **The per-week floor had to be reduced to one number.** The grid priced an empty week at
  *that* week's floor while the simulator priced it at the season average, and a plan leaves
  empty exactly the weeks the floor is high. That selection put `model_points` at +8.64
  against a simulated +13.50 — a 56% disagreement between optimiser and simulator about the
  same plan. One number, and they agree to 0.5% (13.57 vs 13.50).
- **`ValueSplit.bye_cover` and `as_recommendation`'s `covers_bye` both meant "started
  nobody".** That was the same statement as "on a bye" only while an empty seat scored zero.
  Under a floor there is a second reason to bench an incumbent — he is worth less than the
  wire — and on the live D/ST grids that is *every* week, so the old spelling reported the
  entire 13.0-point gain as a bye cover on a schedule with one bye. Worse, `covers_bye` is the
  sole exemption that lets a `not-streamable` position transact at all, so it would have
  waved through exactly the junk kicker moves the gate exists to suppress. Both now mean
  "holds somebody, none of them playing".
- **`apply_matchup_model=False` was scoring the plan in one currency against a baseline in
  another.** The sensitivity run turns the re-pricing off, so the floor override — which
  exists *because* of the re-pricing — has to come off with it. Contaminated, the check read
  +0.475pp/+5.1 points; corrected, +0.90pp/+11.2. That is a wrong conclusion drawn from the
  run whose whole job is to check the conclusion.

**The honest bound got worse, and it is the number worth reading.** Only **2.1–2.4** of the
13.4–15.7 model points sit in weeks a bookmaker has priced (it used to be 13.4–14.9 of a much
larger total). `through_week=6` now gives `+0.23 ± 0.28`, `+0.35 ± 0.23` and `−0.28 ± 0.29` —
**none clears two standard errors and one is negative**, where it used to clear by a hair.

**Negative control:** `build_grid(floor=0.0)` + `evaluate(replacement=0.0)` on the three live
leagues reproduces the parent commit **byte for byte** — floors, plan, hold, `delta_title`,
`stderr`, `delta_points`, `model_points`, both title levels, `commit_delta`, `leverage` and
all six `ValueSplit` fields, for D/ST and K. Separately verified that `replacement=0.0` and
`replacement=None` give bit-identical franchise scores, so the old default really is
expressible.

The offline fixture had to change too. `_sim_league` gave its free agents alternating 3 and 9,
which put three bodies at exactly 9.0 — and a wire with three identical best bodies is
bottomless: the depth-2 floor equals the depth-1 pick, signing one is worth exactly nothing,
and every test measured zero. That was the fixture being degenerate for the question, not the
floor being wrong. It is a descending ladder now, like a real wire.

### #6 — the bench-hoarding estimator had no caller anywhere

`DEFAULT_BENCH_HOARDING` sums to **1.80**. `bench_hoarding_from_rosters` existed to replace
it, the module docstring named it as the fix, and `grep` over `src/` found **zero callers** —
not merely none outside its own module, none at all. Every valuation ran on the prior.

Measured on the live rosters at week 1 of 2026:

| | Blacksburg | Wine Wednesday | Type shi |
|---|---|---|---|
| carried RB/team | 4.58 | 4.57 | 4.83 |
| carried WR/team | 5.67 | 5.86 | 5.83 |
| **sum(beta)** | **7.25** | **7.07** | **7.17** |
| prior | 1.80 | 1.80 | 1.80 |

**It is not a level error and does not cancel.** The prior understates the bench most at the
positions a bench is made of, so replacement level moves very unevenly (Blacksburg, points a
week): RB **8.26 → 3.55**, WR **7.95 → 5.15**, QB 15.05 → 13.54, TE 7.18 → 5.91, but K only
8.75 → 8.63 and D/ST 6.47 → 6.31. A running back gains 4.7 points a week of VORP and a kicker
gains 0.13.

So it reorders *across* positions: **543–552 of the ~598 valued players change rank**, and all
16–17 of the user's own in every league. Harrison Butker falls 120 → 188, the Chargers D/ST
148 → 214, Chris Boswell 151 → 235; Jordan Mason rises 156 → 76, Tank Bigsby 288 → 177,
MarShawn Lloyd 188 → 80. The live dashboard board changes at the top too — Blacksburg's first
four go from `[Jefferson (WR), Chase Brown, Etienne, Hurts (QB)]` to
`[Chase Brown, Etienne, Jefferson, Warren]`.

**Where the fix went.** Into `value_league(rosters=...)` rather than into the one caller. It
has to be two passes — `bench_hoarding_from_rosters` needs a `PositionDemand` to subtract
starters from what is carried, and demand only exists once a model is solved — so putting the
two-pass at the entry point is the difference between a fix and the next caller forgetting
again. Without rosters it logs, at INFO, that it is falling back to the prior and what that
costs. `api/server.py` fetches the rosters in their own guard: losing them should cost the
measured coefficient, not the whole board.

The bench count for the sanity check is **not** on `ctx.lineup_slot_counts` — that is
`settings.roster.starting_slots`, with the bench filtered out — so it comes from
`settings.roster.lineup_slot_counts[SLOT_BENCH]` and a mismatch warns rather than raises.

**Negative control:** `value_league` with no `rosters` on the three live leagues reproduces
the parent commit's every `ros_vorp` and `playoff_vorp` to a **byte-identical digest**;
`rosters={}` and an explicit `DEFAULT_BENCH_HOARDING` give the same. Five new tests fail on
the parent commit and pass here.

### Also fixed along the way

- Three live-market tests asserting more than the market promises (`7ea0553`). Pre-existing
  failures, unrelated to this work. All three asserted a per-item exactness the module itself
  does not claim; now judged on distributions. See the commit for the measurements.
- An off-by-one in the live D/ST streaming test: my own defence is a candidate, so the ceiling
  is `size - 1` rostered, not `size`. It only passed while some rival carried no defence.

---

## Remaining work

Ranked. Everything below is located and measured; none is started.

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
