# The first-principles audit: findings, fixes, and what is left

Status as of 2026-09-10. **Every finding in this audit is fixed and pushed** -- all nine
verified findings, plus three the audit did not have -- and the one constant that was left
as the user's decision has been measured and answered: `PLAYOFF_WEIGHT` stays at 1.2. See
**Remaining work** for why, including a recommendation I got wrong first.

Since the audit closed, one feature has been built on top of it: **a second opinion**
(`decide/opinion.py`), which prices the wire and the trade board against Establish The
Run's rankings instead of ESPN's projections alone. It is recorded here rather than in a
separate file because it found three defects of exactly the audit's shape, two of them in
its own first draft. See **After the audit: the second opinion**.

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

### #8 — a noisy sign printed as a flat assertion, twice

`trades.py` tagged on two bare sign tests over a paired title delta:

```python
if mine.delta_title <= 0.0:
    tags.append("harmful")
if any(i.delta_title < 0.0 for i in ev.impacts if i.team_id != team):
    tags.append("counterparty-loses")
```

and `report.py` rendered the second as *"a counterparty's simulated title odds FALL under
this"*. **The module already knew.** `TradeEvaluation.title_pareto`'s own docstring says of
that exact quantity: *"the per-side deltas are individually noisy at affordable simulation
counts, and a gate on a noisy quantity is a gate on noise."* Two properties later, both tags
gate on it.

Measured across three seeds, on the forty trades all three runs found:

| | Blacksburg | Wine Wednesday | Type shi |
|---|---|---|---|
| `counterparty-loses` disagrees with itself | **75%** | 52% | 35% |
| `harmful` disagrees with itself | 42% | **78%** | 18% |
| `title-pareto` disagrees with itself | 25% | 58% | 40% |

And the reason, which is worse than instability. The tag fired on **45 of 120** trades. Of
the **50 counterparty impacts that read negative, not one cleared the selection-adjusted
threshold** (z = 3.23 at forty candidates) and only four cleared even a naive two sigma. The
median |delta|/stderr is 0.71–1.18. Every one of those 45 assertions was the sign of noise.

Both tags now have three states against the `z` that was already computed two lines above
them — *falls* / *cannot tell* / *nothing to say* — and the rationale prints the two cases in
different sentences instead of quoting a standard error beside a number it had not tested
against it. After the fix, on the same three seeds:

| | fires | disagrees across seeds |
|---|---|---|
| `counterparty-loses` | **0 / 120** | **0%** |
| `harmful` | 0 / 120 | 0–2% |
| `counterparty-loses-unclear` | 45 / 120 | 35–75% |
| `harmful-unclear` | 61 / 120 | 18–78% |

The instability is all in the `-unclear` rows, which is where it belongs: those say they did
not resolve. **No protection was lost** — `_confidence` returns `"low"` on the sign
independently of the tag, and both `-unclear` names are in `report._LOW_CONFIDENCE_TAGS`, so
the reader still gets the warning and now gets an honest reason for it.
`counterparty-loses-unclear` is deliberately **not** in `portfolio.CONTESTED_TAGS`: a blocker
that fires on "cannot tell" blocks the whole board.

**Negative control:** with `z` forced to 0 every `-unclear` collapses and the resolved counts
land exactly on the `-unclear` counts at the real threshold — 14 / 10 / 21 for the
counterparty and 27 / 29 / 5 for `harmful`. `_resolved_loss` uses `>=` rather than `>`
precisely so `z = 0` reproduces the old test on an exactly-zero delta too. Three of the seven
new tests fail on the parent commit.

> **Left as a tradeoff rather than patched: `title_pareto`.** It is a sign test over every
> impact and it disagrees with itself 25–58% of the time. Its own docstring already declares
> it *"reported rather than gated on... Still not the gate"*, and the surface's actual gate is
> the points Pareto, so the noise is disclosed rather than acted on. Gating it on `z` would
> empty it — no per-side delta on these leagues clears the threshold — which is a decision
> about what the tag is *for*, not a bug fix. The stale claim in that docstring ("24 to 27 of
> every 40") is now 14 / 10 / 21 of 40 on today's rosters.

### #9 — the forced cut was a function of ESPN's roster ordering

```python
if value > best_value:          # keeps the FIRST maximum
    best_pid, best_value = pid, value
```

over `roster = list(dict.fromkeys(player_ids))`, which traces back to
`self.rosters = {f.team_id: tuple(f.player_ids) ...}` — ESPN's own ordering, straight
through. `value_of` is a whole-roster starting-lineup objective, so a deep-bench player who
never cracks a lineup contributes exactly zero and removing any of them leaves the objective
**bit-identical**.

Measured on the three live leagues: **65 of 69 forced cuts (94%) had at least two bit-exact
ties**, median tie group three to five, maximum six. The spread between the best and worst
leave-one-out is 97–140 points, so the choice matters enormously in general; it is only among
the *top* candidates that it is a dead heat.

The proof, and the reason this is a defect rather than an untidiness — shuffle the roster
before handing it to `settle` and see whether the advice changes:

| | before | after |
|---|---|---|
| forced cuts whose chosen player moves under a shuffled input | **65 / 69** | **0 / 69** |

Ties now break on the least valuable asset — lowest playoff-weighted rest-of-season points
(`self._mu[pid] @ self._w`, the expression `startable_count` already writes), then lowest
player id so the answer cannot depend on an input ordering at all. A relative tolerance
rather than exact equality, because letting a 1e-12 difference in a ~100-point objective
decide the cut is the same defect one level down. The equals come back on
`TeamImpact.cut_alternatives` and the rationale names them: *"cut this one"* and *"cut any of
these five, they are indistinguishable"* are different pieces of advice, and after the fix
**all 69 settlements report a tie**.

> **Scope, honestly: this does not make the trades better.** The screen cannot see which of
> the tied bodies it throws away, but the confirmation can, so a better tie-break might have
> shown up as better confirmed deltas. It does not. Across three seeds and three leagues the
> mean confirmed delta moves −0.064 → −0.072pp, +0.095 → +0.092pp and +0.963 → +0.958pp, the
> best moves +0.900 → +0.858pp, +0.742 → +0.733pp and +2.317 → +2.308pp, and the counts
> clearing the selection threshold are identical. All of it is inside the noise. What the fix
> buys is a **determinate and disclosed** answer, not a better one.

**A live test fired, and it was over-asserting.**
`test_no_trade_on_the_live_board_survives_the_field_it_was_selected_out_of` asserted
`not any(i.significant for i in trades)`. A Bonferroni threshold at α = 0.05 is built to admit
about one board in twenty, so "no trade ever clears" is an invariant nothing promises — the
same over-assertion as the three live-market tests rewritten in `7ea0553`. It fired when the
tie-break moved a borderline row by 0.04pp, and the row is noise: across seeds 1, 2 and 3 the
significant set is `{}`, `{}` and one trade, and that trade is not in the other seeds' top
three. It now asserts what the correction is *for* — rows a naive two-sigma test would call
significant must stop being significant once the threshold accounts for the field they won,
and anything that survives must be a hair over the bar rather than comfortably clear of it.

### Leftover — `trades.wire_pool` called the whole pool "rostered"

`rostered = set(state.pool.player_ids)`. Accidentally right under `pipeline.build`, which pools
only rostered players: on all three live leagues the pool and the rostered set are the *same
195 / 225 / 194 ids*, so no test could tell them apart. Wrong the moment anything widens the
pool — and `waivers.augment` does exactly that, adding the sixty best free agents so they have
columns to be simulated in.

The audit predicted "the wire goes empty". It does not; it reads the **dregs behind the
augmented sixty**, which is worse because it looks plausible. Measured on the live leagues with
`augment`'s own candidate set, floor points a week:

| pos | read as | truth | | pos | read as | truth |
|---|---|---|---|---|---|---|
| QB | 10.97 | 14.07 | | TE | 4.75 | 6.94 |
| RB | 4.62 | 5.72 | | **K** | **0.29** | **8.87** |
| WR | 4.84 | 6.54 | | D/ST | 4.96 | 6.99 |

0.74 to 8.58 points a week too low at every position. The kicker is catastrophic because all
thirty plausible free-agent kickers get pulled into the pool, leaving the floor to read
whatever is behind them.

> **Scope, honestly: latent.** No production path hands `wire_pool` a widened state —
> `find_trades` is only ever called with a `pipeline.build` sim. **The negative control is the
> fix being a provable no-op today:** pool and rostered are identical sets on all three live
> leagues, so not a number moves. The two new tests are what keep it that way.

`waivers._all_rostered` was the second copy of the same one-liner and is now an alias for
`wire.all_rostered`. The third spelling was the one that mattered.

### Leftover — two surfaces paid an empty seat a constant while `title` paid it a draw

`sim/season.FloorNoise` has existed since the stochastic floor landed and `decide/title.py`
has drawn its empty seats ever since. `decide/waivers` and `edges/portfolio` never picked it
up: `_lineup_scores` and `_column` both called `_franchise_scores` with no `floor_noise`, so
every empty seat was paid its mean with **zero variance**. Same shape as #1 — the canonical
machinery was right and the callers did not reach for it.

Not a rare path. Measured on the user's three rosters:

| | Blacksburg | Wine Wednesday | Type shi |
|---|---|---|---|
| empty slot-weeks | 15.7% | 19.6% | 18.5% |
| emptiest single slot | 71% | 88% | 71% |
| weekly SD understated by | 5.9% | 6.8% | 5.9% |
| season SD understated by | 6.0% | 6.0% | 5.6% |

The level was right and the **spread** was missing, and a bracket is decided by the spread —
so the bias landed on exactly the question both surfaces exist to answer. Live effect:

| | before | after |
|---|---|---|
| Blacksburg P(title) | 6.75% | 6.50% |
| Wine Wednesday | 4.20% | 4.35% |
| Type shi | 9.20% | 8.60% |
| P(≥1 title) | 18.60% | 17.95% |

The waiver board reorders (Wine Wednesday's Vikings D/ST rises past two rows, and several
rows change which player they drop), Type shi goes from 1 claim to 2, and **every standard
error rises** — correctly, because the wire is now a random variable and the paired
difference has genuine extra variance it was previously pretending away.

**Three things this dragged out:**

- **`augment` fitted `wire_floor`, not `wire_levels`.** The board is built on `wide.floor`, so
  the draw could never have reached it however well `RosterSimulator` threaded the uniforms.
  `wire_floor` is documented as the mean of `wire_levels`, so the level is unchanged.
- **`week_scores`'s `team_id` was documented as ignored, and now is not.** The solve still
  depends only on the players, slots and floor, but the seats are drawn per team on purpose:
  `FloorNoise` keeps franchises independent because byes are league-wide, and one shared body
  cancels in `Var(A) + Var(B) − 2Cov(A,B)`, throwing away 40% of the spread.
- **`waivers._roster_floor` was the third copy of the empty-slot-group guard**, after
  `title._floors_for` and `portfolio._floors_for`. It phrased the test over
  `state.slot_eligibility` where the surviving copy phrases it over the compiled plan's
  eligibility matrix; the two were verified to agree on **all 190 roster shapes** across the
  three live leagues before it was deleted. It would also have crashed on a `WireLevel`.

`season.has_spread` gates the `FloorNoise` allocation — `(sims, weeks, teams, seats)` float64,
about **68MB** on a 14-team league at 4,000 simulations — so a mean-only caller does not pay
for an array nothing reads.

**Negative control:** a mean-only mapping reproduces the parent commit's `_base_scores`,
`_base_champ`, `week_scores`, exchange rate and the QB-less empty-group path to a
**byte-identical digest** on all three live leagues. And CRN survives the drawn wire: a null
claim is exactly `0.0`, and two identical paired drops return bit-identical arrays. Seven of
the nine new tests fail on the parent commit.

**A fixture had to change again, for the same reason as `_sim_league`.** `test_portfolio`'s
`overlapping` has every player on a roster, so its wire is empty, `streaming_levels` falls
through to the deterministic VOLS rank and every slot comes back `sd = 0.0`. A spread test run
on it would have passed while measuring nothing. The spread tests use a `wire_strength=` build.

This also closes the **`omitted` → `(weeks,)`** leftover: with the guard living in `_floors`
and the seat drawn rather than credited a constant, there is one scalar in one place and
nothing left to promote.

### Leftover — one league, two published baselines

`pipeline.championship_table` scored an unfilled starting slot at **zero** while every
recommendation surface floored it at the wire. That is one league answering two different
questions, and the gap was disclosed only as a footer string on `fq odds` — impossible to
reconcile from the output. An empty seat does not score nothing; the wire always has a defence.

**The audit predicted the headline odds would move "slightly down". They move UP, and not
slightly.** Championship probabilities sum to one, so this is a *relative* game: flooring an
empty seat helps whoever has the most empty seats, and that is the thin rosters. The user sits
mid-to-low in all three leagues, so the user is who it helps.

| | before | after |
|---|---|---|
| Blacksburg (rank 9/12) | 5.03% | **6.60%** |
| Wine Wednesday (rank 13/14) | 2.70% | **4.15%** |
| Type shi (rank 7/12) | 7.83% | **8.60%** |

And it is a re-ranking, not a rescale: **7 to 11 of the 12–14 teams change rank** in each
league. The largest single move is −3.53pp (Pukachu) and the largest rise +3.25pp (Rice
Farmers). Every table still sums to 1.00000000, which `championship_table` enforces anyway.

`LeagueSim.floors()` caches `title.streaming_levels` over `self.outlooks` — the same call every
other surface makes — and `simulate()` defaults to it, passing the `FloorNoise` with it so
`fq odds` draws its empty seats like everything else. `S.simulate` gained the `noise` pass-through
it was missing; `team_week_scores` already took one.

**Negative control:** `sim.simulate(replacement=0.0)` reproduces `S.simulate(..., replacement=0.0)`
byte for byte on all three live leagues. The old convention is still reachable, exactly. Three
of the four new tests fail on the parent commit.

**The disclosure strings were updated rather than deleted**, because the surfaces still do not
match and it is worth being precise about why. The floor is no longer the reason — that is
closed. What remains is that each surface draws its own season: `championship_table` on
`pipeline.build`'s draw, `decide/waivers` on a widened panel that gives free agents columns,
`edges/portfolio` on one seed shared across three leagues. Those are genuinely different Monte
Carlo universes, so their levels will never match to the decimal. Compare deltas, not levels.

### Also fixed along the way

- Three live-market tests asserting more than the market promises (`7ea0553`). Pre-existing
  failures, unrelated to this work. All three asserted a per-item exactness the module itself
  does not claim; now judged on distributions. See the commit for the measurements.
- An off-by-one in the live D/ST streaming test: my own defence is a candidate, so the ceiling
  is `size - 1` rostered, not `size`. It only passed while some rival carried no defence.

---

## After the audit: the second opinion

`decide/opinion.py`, `data/etr.py`, and the two surfaces that consume them
(`ee4e8d2`, `099cb12`, `7e80b0b`, `170c136`, `48481ee`, `90dde59`, `61c6589`).

**What it is.** Every number in this repo runs on one valuation:
`pipeline.league_projections` reads ESPN's own weekly projections out of the corpus and
re-scores them per league. So "our number" and "the counterparty's number" have always
been the same number seen from two sides, which can answer *is this trade good* and cannot
answer *is it mispriced*. Matt Silva's Top 150 is the second opinion, and it enters in two
shapes: `bench_upgrades` compares ranks to ranks and stops there, and `tilt_outlooks`
carries the ordering into the numbers by **transporting our own points along it** -- within
a position, our value ladder re-dealt in the board's order. A permutation, not a model, so
replacement level, the scarcity curves and the wire do not move.

It never prices a rank. `valuation.py:390` records why -- comparisons against an outside
opinion stay in rank space -- and a fitted `points(rank)` would also overwrite this
league's solved, per-league replacement model with a national one.

### Three defects it found, all of them the audit's shape

**1. The wire read off `state.pool`, which holds only rostered players.** The first
`upgrades_for` reported `FREE=0` in all three leagues. `pipeline.build` pools the rostered
194-225 and nothing else, so a free agent read off the pool is a free agent that does not
exist -- and it returns a clean zero rather than raising, which is how `cc624f6` and
`ef36808` each survived as long as they did. The wire is `sim.outlooks` (598 players).

**2. A rest-of-season rank transported onto a player ESPN projects as absent.** The tilt is
multiplicative, which is exact only when both sources agree on games played. Jordyn Tyson
is WR51 on a board whose own note says "recurring hamstring injuries will sideline Tyson
through September", against 0.064/wk with `p_zero` 0.84 for weeks 1-8 in the corpus. The
transport handed the WR51 season total to a ten-week player: a 60% higher per-game rate,
all of it in the back half where `PLAYOFF_WEIGHT` pays about three times. He became the top
trade target in **two of three leagues**, once at +5.78pp with the counterparty 65
playoff-weighted points out of pocket. `MAX_ABSENT_WEEKS = 1` keeps him out; exactly two of
the 150 are affected.

**3. The trade screen pruned counterparties on numbers they cannot see.** `_paper_gain` and
`asset_ranking` run before `evaluate`, so the market gate alone changed nothing: Blacksburg
returned 38 trades with zero arbitrage rows and Wine Wednesday returned none at all.
`_side(team_id)` routes the preference graph, the pruning bound and `settle` through the
finder whose numbers that team reads.

A fourth, found the same way: `confirm_titles` scores on the **draw**, not on `_mu`, so
reusing `sim.draw` left screen and simulation in different currencies -- +17.8 points
screened, +0.10pp +/- 0.36 confirmed.

A fifth and sixth, found by a fresh-eyes review after the feature shipped, and a crash.
**The value ladder was built over the wrong horizon.** `pipeline._fill_weeks` only ever
ADDS weeks to an outlook, so `sim.outlooks` carries every week ESPN projected -- measured
on the live leagues, **weeks 1 to 22** against a `state.weeks` of 1 to 17. The ladder
summed all of them. Five weeks the league never scores were being ranked on day one (98
players' rest-of-season value moved by more than half a point when it was fixed, up to
4.59), and from week 2 the already-played weeks would have joined them, ranking a player
who was excellent through September level with one about to carry you. Its sibling:
`partial_season` counted absences over the same span, so a returning player stayed
excluded for the rest of the season -- Tyson would never have come back. Both now take
`weeks=state.weeks`. **And `trades_payload` crashed with no board on disk**: `tag_value`
returns `"-"` for a missing tag, `"-"` is truthy, and `float("-")` raises -- so every
fresh checkout, every `--no-rankings`, and the dashboard's own trades route was broken.
Nothing caught it because `test_report.py`'s CLI fixture stubs `trades_payload` wholesale.

That last one is the lesson worth carrying: **a stub standing in for the code under test
hides exactly what it stands in for.** It is the same shape as the `make_portfolio` helper
that mirrored the bug it was supposed to catch (`#1 again, in the portfolio`), one level
up.

### What it is worth, measured (week 1 of 2026, all three leagues)

**On the wire, very little.** The top claim is the same player at every weight in all three
leagues; only its price moves, by about 40% (Blacksburg +0.237 -> +0.343pp). One row in one
league carried a board endorsement. That is structural: the Top 150 is 150 skill players, a
league rosters 194-225 of them, and the top of a week-1 wire is defences and kickers, which
the board does not rank. Only 4 to 7 of the 150 are unrostered.

**On the trade board, substantially.** Top rows at weight 1.0:

| league | ΔP(title) | trade | spread |
|---|---|---|---|
| Blacksburg | +1.70pp | Daniel Jones for Blake Corum | −7.9 |
| Wine Wednesday | +2.30pp | Mike Evans for DK Metcalf | +14.2 |
| Type shi | +3.10pp | Lawrence + Mahomes for Carnell Tate | +6.8 |

`spread` is how much better the counterparty reads the deal by their own projections than
by the board. The gate got **looser**, not tighter, and that is the object: a deal good for
me and good for them *by my reckoning* needed no second opinion to find.

### The one input here with no measured verdict

Everything else in this repo ships with one -- calibration bias, the ensemble's twelve
seasons of head-to-heads, analyst dispersion at +0.187 (p=0.004). This does not, because
ETR overwrites each chart in place and publishes no history, so there is nothing to
back-test against: the first board we hold is the one being used. The user chose to run it
anyway and to measure later. Two things make that reversible:

- `data.etr.archive` keeps a dated copy on every read, which is the only way the comparison
  set ever exists. Score archived boards against realized weekly points once several weeks
  of them exist -- roughly week 6-8.
- `rankings_weight` is one constant, and `0.0` is byte-identical to not having the board.

`edges/portfolio.py` deliberately takes no board: it ranks waiver, streaming and lineup
rows in one currency and the latter two run on ESPN's numbers.

---

## After the board: making the advice worth acting on

The second opinion shipped and the user's verdict was that the recommendations were bad:
the trade board acquired **multiple quarterbacks** in one-QB leagues, and the waiver board
wanted to **add defenses while dropping high-upside running backs**. Both complaints were
correct. Neither was caused by the ETR work.

### The four-QB trade was a coin flip the board kept losing

At **8,000 simulations under three seeds**, "Lawrence for Tate" against "Lawrence +
Mahomes for Tate" came out **+0.563pp, −0.175pp and +0.125pp** apart. The sign flips: the
model cannot separate them, so which one tops the board is decided by the draw — and the
one that won left four quarterbacks on a sixteen-man roster and forced Jordyn Tyson out.
Twenty of forty candidates had a strictly leaner sibling already in the same search.

`prune_throw_ins` drops a candidate when a strict **subset** of its own moves cannot be
shown to be worse. The comparison is paired — both ran against the same baseline on the
same draw, so the error on `A − B` is ~20× smaller than either one's own, and comparing
the two published means (±0.4pp each, on a 0.06pp gap) would never have fired. Live, the
four-QB trades are gone and 40 candidates become 10–18 with **zero orphans**: everything
dropped is a fatter copy of something kept.

Two things it is not. It is not the redundancy gate the plan called for — measured on the
user's roster, a "do not exceed the startable requirement" rule refuses adds at QB, RB,
WR, K *and* D/ST, and would have blocked the +14-point Lawrence upgrade too. And it
cannot fix a throw-in whose lean variant the search never enumerated; one such case
survives on Type shi.

### Bench upside is real, nobody captures it

`sim/season.py` already said a rank-less tensor gives "the upper bound you need to price
bench option value". Priced, in rest-of-season points:

| dropping | costs, ex ante | at the hindsight ceiling |
|---|---|---|
| Jordyn Tyson | +0.27 | **+19.13** |
| Tank Bigsby | +0.56 to +0.81 | +7.4 to +9.8 |
| Makai Lemon | +0.8 | **+45.9** |

So the upside is real, by a factor of ten to seventy. **And nobody captures it.**
`measure_hindsight_ratio` puts the projection-optimal lineup at **0.886/0.900/0.886** of
that ceiling, against a measured manager efficiency of **0.775** — real managers land
eleven points *below* simply following projections. A foresight parameter above zero
would price skill nobody has ever demonstrated, which is `PLAYOFF_WEIGHT` again.

So it is reported and never charged: `cut_cost` and `drop_cost` return both numbers, the
trade rationale prints the gap, the waiver board tags it, and the output says why the
second number is not charged. The cut becomes the user's decision with the one fact the
objective cannot see — the same move `cut_alternatives` made for ties.

**The roster-spot half of the plan collapsed.** A spot is worth what the marginal player
it holds is worth, and that is the same zero for the same reason. Pricing it separately
would double-count a number that is not there.

### Streaming, and a framing error that nearly shipped

`stream_advantage` reports per single-body slot what the best available every week yields
against holding the best rosterable body. The first version quoted the positive gap as
evidence that streaming wins. **It is not**: `sum_w max_i >= max_i sum_w` for any table of
numbers, so the sign is an identity. Only the size across positions is information —
D/ST **+38.4**, QB +27.5, K +13.2, TE +12.9. D/ST is three times K and TE, and that is
the measured reason a defense is the seat everybody streams. Neither column pays for the
weekly transaction, so it is a ceiling, not a policy.

### One decision printed three times

At 4,000 simulations the top **three** rows of Type shi were all "Trevor Lawrence for
Carnell Tate" — +2.67pp, +2.43pp and +2.25pp against standard errors of 0.55 — routed
through one, two and two counterparties. Wine Wednesday's top three were all "Mike Evans
for DK Metcalf". One decision each, taking three of five rows and pushing genuinely
different ideas off the board.

They are not duplicates and are not dropped: a different middleman is a different person
to persuade and carries its own spread, which is the number you want when choosing whom
to ask. `order_routes` groups them by what the subject actually gets and gives, leads with
the easiest to sign (fewest teams, then widest spread), and marks the rest `same-return`.
The published sort is group-aware, or the global ranking would split a family and let a
three-team version outrank the two-team one handing over identical players.

### Both ranks, on every player

`opinion.rank_pairs` puts our positional rank and the analyst's beside every name the
board prints: **Lawrence (QB12/QB10) for Tate (WR29/WR45)**. It is the arbitrage stated
in the one unit both sources publish, and without it the reader takes the claim on trust.
Our side is ranked on rest-of-season points over the remaining weeks, not ESPN's frozen
preseason ordering, which `PlayerOutlook.mean_from` is explicit is never revised. Players
the board does not rank carry our side alone — it covers 150 of ~598, and implying an
opinion it has not published would be worse than silence.

### Why the boards are full of three-way trades

They are, and it is economics rather than a defect. A two-team swap needs a **bilateral
coincidence of wants** — I have what you want *and* you have what I want — while a cycle
only needs a chain, so cycles are simply more findable. Measured at week 1 of 2026 the
screen returned **13 / 0 / 4** two-team candidates against **27 / 40 / 36** three-team
ones, and Wine Wednesday had no bilateral trade that cleared the Pareto gate at all.

Where a two-team trade does exist it tends to be *better* (+1.82 against +1.68pp on
Blacksburg, +2.83 against +2.43 on Type shi), and two of the three leagues already lead
with one. `--max-teams 2` restricts the board to deals needing one other manager, and the
output says how many of the shown rows already qualify.

One real bias was found and fixed while checking this: `_cycle_gain` **summed** one
non-negative term per leg, so a three-cycle outscored a two-cycle by construction and
took the search budget with it — the same shape as ranking positions by a sum of per-week
maxima. It is the minimum leg now, because the Pareto gate makes a cycle worth exactly
what its weakest leg can carry. **The fix is latent**: the candidate composition is
unchanged on all three leagues, because the 200-cycle budget was never binding at these
league sizes. It will matter when it is.

### The queue could never load on a cold server

Reported as `timed out after 120s waiting for /queue` on a freshly restarted dashboard.
Measured: **`/api/queue` takes 334s cold and 0.0s warm.** It runs every surface in every
league before it can rank anything — the three waiver boards alone are ~74s each — so it
is not a slow version of a single-league call, it is a different order of work. The 120s
client default was written for single-league surfaces, and `_lifespan`'s "deliberately no
warm-up: the first request pays" was written when that meant ten seconds.

The browser gave up while the server carried on and finished, so the *next* attempt
returned instantly from a cache with no TTL — a timeout reporting failure for work that
was succeeding. The cross-league endpoints now carry a 600s timeout measured against the
334s cost, and their loading state says five minutes and says it is paid once per server
run rather than "the decision surfaces take longer".

`drop_cost` is also memoised per simulator. Every row pairs its add with the same
cheapest drop, so one player sat under thirty rows at four full-tensor lineup solves
each: 4.6s of a 74s board, three times over in the queue. It was **not** the cause — the
334s is almost entirely pre-existing `price()` calls at ~0.5s each — and saying so
matters more than the saving.

### Two regressions this work introduced, both caught

- **Pruning shrank the selection field.** `portfolio` sized the multiplicity off
  `len(find_trades(...))`, which parsimony had just cut from forty to ten — softening the
  correction on the noisiest surface in the app. The count now travels on a
  `considered:N` tag. Only the live test caught it; no offline fixture has enough
  candidates.
- **The dashboard was pricing on ESPN alone.** Two cwd-relative data roots (`data/manual/etr`
  and the `data/reference` crosswalk) resolved silently to nothing off the repo root.
  `paths.data_dir` resolves both, and absence is now loud.

---

## Remaining work

### Raised by this work, and now answered: `PLAYOFF_WEIGHT`

**Recommendation: leave it at 1.2.** An earlier revision of this section recommended 1.29 on
the strength of `SurrogateFit.playoff_premium`. That was wrong twice over, and both errors are
worth writing down because they are the audit's own recurring shape — *a component computing
something defensible-looking that is not the right quantity*.

#### What the constant is for

`self._w = self.weights.vector` is the week-weighting of the **trade screen's** objective,
`value_of = weekly_points @ self._w`. It does three things: it gates `TradeEvaluation.pareto`
(`all(i.delta_points > 0)`), which decides what is even proposed; it ranks candidates for the
expensive paired-CRN confirmation; and it now breaks `settle`'s forced-cut ties. It does *not*
touch `delta_title`, which is pure simulation.

Note what that implies: a trade that changes a roster's points roughly uniformly scores the
same under any weighting that conserves the total. **The weight only discriminates between
trades whose gains arrive at different times** — a bye in the bracket, a playoff-schedule edge.

#### Where 1.2 came from

`c10a319` (2026-09-07), the Wave 4 build, with no cited source. `docs/RESEARCH.md` says the
objective is "playoff-weighted" and never gives a number. It is a hand-set constant, as the
audit said.

#### Why 1.36 is not the right target

`playoff_premium` is `1 + dP/dp ÷ dP/dx`, and those are derivatives **of the surrogate's own
axes**: `x` is points a week across *all 17 weeks* and `p` is points a week added to the
bracket *on top of x*. So `dP/dx` already contains the bracket's contribution, and dividing by
it produces a heavily diluted ratio. Verified against the fit: `delta_title(0, 1, 1)` returns
exactly `dP/dp`, so `p` really is the on-top axis.

Written out, with `B` and `R` the title value of one point in a bracket and a regular week:

```
dP/dx = n_n*R + n_p*B
dP/dp = n_p*B
  =>  B/R = (dP/dp / n_p) / ((dP/dx - dP/dp) / n_n)
```

Three independent routes then agree, and none of them is 1.36:

| league | `playoff_premium` | B/R from the same fit | B/R by direct simulation |
|---|---|---|---|
| Blacksburg | 1.364 | 2.667 | 3.07 |
| Wine Wednesday | 1.363 | 2.664 | 2.59 |
| Type shi | 1.402 | 3.131 | 4.22 |

The direct measurement adds the same total points to a team's bracket weeks and then to its
regular weeks at 20,000 paired simulations — no surrogate involved. So the true bracket
premium is **~2.6–4.2**, and both 1.2537 (what `w_p = 1.2` actually delivers, since
`playoff_weights` conserves the total and scales the regular weeks down) and 1.3755 (the
earlier recommendation) are well under it.

#### Why it should still not be raised

Because "what is a bracket point worth" is not the question the screen answers.
`find_trades`'s own docstring settles it: **"the screen is a recall filter, not a ranker."**
The confirm does the ranking. What the screen must do is keep the best trade alive long
enough to be simulated.

Both metrics, measured across seeds and leagues:

| `w_p` | rank correlation with the confirm (9 runs) | **best trade surviving the screen** (6 runs) |
|---|---|---|
| 1.0 | 0.502 | **1.346pp** |
| **1.2** (shipped) | 0.507 | **1.325pp** |
| 1.29 | 0.519 | — |
| 1.6 | 0.555 | 1.104pp |
| 2.0 | 0.573 | 1.179pp |
| 2.5 | 0.573 | 0.829pp |

**They point opposite ways, and the second one is the one that matters.** Raising the weight
does improve rank correlation — consistently, in 8 of 9 runs, not a seed artifact — but it
*costs recall of the best trade*: 1.0 and 1.2 are tied at the top and every higher value is
worse, with 2.5 losing 38% of the best available trade. On Wine Wednesday at `w_p = 2.0` the
+0.47pp trade stops surviving the screen at all and the board's best becomes +0.38pp.

The mechanism is plain once stated: weighting weeks 15–17 at two to three times pushes the
screen to chase players projected well *fifteen weeks out*, which is the least reliable part
of the forecast. It buys a better ordering of a field it has already mis-selected.

1.0 and 1.2 are not distinguishable from each other here (1.346 against 1.325, well inside the
spread). There is no evidence for raising the constant and no case for lowering it, so the
honest action is to leave it alone and record why.

---

## How to work on this

- `uv run pytest -q -m "not network"` — 2,050 offline tests, ~100s.
- `uv run pytest -q -m network` — hits ESPN and FanDuel with the real credentials, ~4.5 min.
  Live-market tests are inherently a little flaky; judge on distributions, not single items.
- `uv run ruff check src tests` before every commit.
- `npm --prefix web run build` and `npx tsc --noEmit` (from `web/`) after touching the dashboard.
- **Every change needs a negative control** that proves it did what it claimed and nothing
  else. The pattern used throughout: run the new code path with the new input disabled and
  assert the output is byte-identical to before.
- Measure on the real leagues, not just the fixture. `Registry.load().active()` gives all three;
  `cfg.team_id` and `cfg.scoring_variant` are the field names (not `my_team_id` / `variant`).
