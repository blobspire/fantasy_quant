# fantasy_quant — professional-grade multi-league fantasy football analyzer

## Context

You play in multiple competitive, real-money ESPN redraft leagues (PPR / half-PPR) and want every available edge. Today you make decisions the way everyone else does: eyeball ESPN's projections, read a ranking list, guess at FAAB. That loses to a tool that (a) models uncertainty instead of point estimates, (b) knows your *specific* league's scoring and roster rules, and (c) optimizes the thing that actually pays — championship probability — rather than projected points.

Three research passes (each empirically verified against live 2026 ESPN endpoints) turned up a set of exploitable inefficiencies that no existing public tool implements. This plan builds a local web dashboard over all your leagues at once, read-only against ESPN, that surfaces trades, waiver/FAAB claims, and start/sit calls ranked by their effect on your title odds.

**Decisions locked with you:** local web dashboard · read-only ESPN, recommend-only (you click the buttons) · Establish the Run + free sources + a betting-odds API · redraft PPR/half-PPR day one · all leagues, not one · **the full edges list below is in scope, not a stretch goal** · public GitHub repo under `blobspire`, committed incrementally.

---

## ⚠️ Do this first — a 3-day window

**ESPN purges weekly projections and does not backfill them.** Verified: `112026` weekly projection rows are live now; the equivalent `112025` rows for the completed 2025 season are already gone. Every calibration number this system depends on — variance-vs-mean, hurdle rates, correlation blocks — comes from comparing weekly projections to weekly actuals. If we don't start capturing before Week 1 kicks off **Sept 10 — three days out**, we lose the 2026 corpus permanently and wait a year.

**Phase 0 is a ~150-line snapshotter, run on a daily cron, that dumps the full ESPN player pool to Parquet.** It ships before anything else, including the UI. Everything downstream can be rebuilt later; this data cannot.

```
GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leaguedefaults/3?view=kona_player_info
Header: x-fantasy-filter: {"players":{"limit":600,"sortPercOwned":{"sortAsc":false,"sortPriority":1}}}
```
No auth, no league ID, no key. Returns the whole pool with projections (`statSourceId=1`) and actuals (`statSourceId=0`), already scored in a standard PPR context. Snapshot daily to `data/snapshots/espn/{date}.parquet` and never delete.

---

## The core thesis

> Every public tool optimizes expected points. **This one optimizes Δ P(championship).** Every surface — trade, waiver, start/sit — returns the same unit, so they're directly comparable across all your leagues.

Two calibration results from the research anchor the whole product:

- **1 projected point ≈ 1.16 percentage points of weekly win probability** at an even matchup — decaying to **14% of that** when the matchup is a 2-sigma blowout. So the most valuable thing the tool can tell you each week is often *"this decision doesn't matter, don't spend a waiver on it."*
- **The sign on variance flips with your standing.** A fringe playoff team should *maximize* variance; a lock for the 1-seed should minimize it. Same roster, same week, opposite optimal lineup. No projection-based tool can produce this.

---

## Architecture

Pure-Python analytics core; the dashboard is a thin client over it. Keeps the hard part testable and the UI replaceable.

```
fantasy_quant/
  espn/           thin requests.Session client on lm-api-reads (NOT espn-api — see below)
    client.py       filters, If-None-Match caching, x-fantasy-filter-player-count paging
    constants.py    generated from chui_default_platformsettings — never hand-maintained
    scoring.py      scoringItems -> callable scorer, position-keyed pointsOverrides
    discovery.py    Fan API -> every league your SWID belongs to
  data/           nflverse, FantasyPros, Sleeper, sportsbook props, ETR ingestion
  projections/    component-stat ensemble, per-position calibration, xFP
  sim/            correlated hurdle-gamma weekly sim, season sim, playoff bracket
  decide/         Δ P(title) engine + valuation, trades, FAAB, streaming, lineups
  api/            FastAPI — serves JSON to the dashboard
web/              Vite + React + TS + Tailwind + Recharts
```

**Stack:** Python **3.13** via `uv` (not the installed 3.14 — `nflreadpy` supports 3.10–3.13 only). Polars + NumPy + SciPy. **DuckDB over Parquet** for the snapshot corpus and feature store. PuLP or OR-Tools for the integer programs. FastAPI backend, React frontend.

**Multi-league model.** Data plane is shared and pulled once; only league state is per-league. Critically, **a player's value is a function of `(player, league_settings)`, never a global number** — scoring, roster slots and league depth all move replacement level. The registry keys everything on `(league_id, season)`.

---

## Why a hand-rolled ESPN client instead of `espn-api`

`espn-api` (957★, maintained) has a good object model but three disqualifying gaps for this use case:

1. **It does `pointsOverrides.get('16')`** — hard-coded to D/ST. It silently drops TE-premium and per-position PPR, so it computes the wrong score for any league with them.
2. **`free_agents()` hard-codes `sortPercOwned`** and cannot sort the pool by projection. Sorting the free-agent pool by *league-scored projected points* server-side is the single most useful ESPN capability and no library exposes it.
3. **`projected_total_points` reads the frozen season field** (below).

The client is ~200 lines. Keep `espn-api` as an optional convenience for league/roster objects; never use it as scoring ground truth.

### Verified ESPN gotchas the client must encode

| | |
|---|---|
| **Frozen season projection** | The season-total row is fixed at preseason and never updates. Everyone's measured "ESPN is ~11% optimistic" is really this staleness. **Build rest-of-season by summing the remaining weekly rows.** A real edge, cheaply had. |
| `limit` without a sort | → HTTP 400. The #1 thing that bites people. |
| Bogus `view=` | → HTTP 200 with a skeleton. Typos are undetectable from status code. |
| `pointsOverrides` keys | are `defaultPositionId` (4=TE, 16=D/ST), **not** `lineupSlotId`. The two ID spaces collide at 4 and 15. Most common source of silent scoring corruption. |
| Raw `stats` | contains pre-computed derived buckets (statIds 47–52 etc.). Apply **only** statIds that have a `scoringItem` or you multi-count. |
| Weekly projections | need **both** `?scoringPeriodId=N` and `"11{season}{N}"` in `additionalValue`. |
| `matchupPeriodId` | ≠ `scoringPeriodId`; `matchupPeriods` inner lists are unsorted. |
| `fantasy.espn.com/apis/v3` | 302s to a marketing page. Use `lm-api-reads`. `site.api.espn.com` 403s from datacenter IPs. |
| `NFL_MIGRATION: true` | is live in 2026 settings — add a schema canary. |

**Startup assertion:** parse `scoringItems`, re-score a known boxscore, and assert it reproduces ESPN's `appliedTotal` for every player. The research hit 160/160 exact. This is the regression test against ESPN changing anything.

### Multi-league auto-discovery

No "list my leagues" endpoint exists on the fantasy API (it 405s). The Fan API does it:

```
GET https://fan.api.espn.com/apis/v2/fans/{SWID}?featureFlags=expandAthlete
Cookies: SWID={...}; espn_s2=...
```
Filter `preferences` to `typeId == 9` and `metaData.entry.gameId == 1`, then read `groups[].groupId` (league) and `entryId` (your team). Response *shape* is community-reported, not verified — log the raw response on first run and adapt. Keep a manual league-ID override list as fallback.

**Cookie hygiene:** `espn_s2` lasts ~a year and **dies silently, usually mid-season**. A daily authed canary is the highest-value piece of ops in the project.

---

## Build phases

### Phase 0 — Snapshotter (before Sept 10) 🔴
Daily Parquet dump of the ESPN pool; nflverse mirror. Nothing else. Ships standalone.

### Phase 1 — ESPN client + scoring + discovery
Generate constants from `chui_default_platformsettings` (235 statIds, slot→eligible-position maps, every enum). Scoring engine with the 160/160 assertion. Fan API discovery → league registry. CLI: `fq sync`.

### Phase 2 — Projection ensemble
Ensemble at the **component-stat level** (pass yards, receptions, rush TDs), never at the points level — that's what makes projections transfer across your leagues' different scoring. **Equal weights** (a 12-season audit found equal weighting beat accuracy-weighting in 64% of head-to-heads; source accuracy doesn't persist). Hodges–Lehmann location estimator. Zero-weight FantasyPros/FFN as inputs — they're themselves consensuses and would double-count.

Then the **highest-ROI few lines in the system — per-position calibration:**
```
E[actual | proj] = a_q + b_q · proj    b_QB=0.67  b_RB=0.79  b_WR=0.85  b_TE=0.72
```
Every public projection set is 20–35% too wide. This matters enormously for anything consuming point *differences* — which is everything we compute.

### Phase 3 — Simulator
**Hurdle gamma**, per position (normal is empirically wrong: skew to +1.42, excess kurtosis to +4.5). Parameters measured on 5,337 player-weeks:
- Hurdle P(≤0): QB 6%, RB 15%, WR 25%, TE 22%
- **Heteroskedastic variance: σ(μ) ≈ 2.5 + 0.30·μ**
- Correlation, block-diagonal by NFL team via a Gaussian copula: ρ(QB,WR)=0.30, ρ(QB,TE)=0.20, ρ(QB,RB)=0.08, ρ(TE,TE)=0.13, ρ(RB,RB)=−0.09, else 0. **Different-team pairs correlate +0.003 and a league-wide week factor explains 0.71% of variance** — so there is no market-wide factor and the matrix is cheaply block-diagonal.
- **Injuries must be absorbing** (a hazard + duration draw). `ffsimulator` draws IID per week, so a Week 3 season-ender doesn't persist — badly understating the left tail.
- Lineup efficiency haircut ~0.775 ± 0.05 when modelling opponents.

**Lineup solving: greedy is provably exact** for nested slot eligibility (dedicated ⊂ FLEX ⊂ SUPERFLEX), so don't run an LP in the inner loop — vectorize a partial sort over the `[sim, week, player]` tensor. Keep a PuLP path as a correctness oracle and assert equality on a sample. Budget ~5.4M cells, sub-second in NumPy.

**Common random numbers** on paired comparisons: 2,000 sims suffice where 14,400 would be needed independently.

### Phase 4 — The Δ P(title) engine
Two-tier, and this is the heart of the product:
- **Tier 1 (microseconds, thousands of candidates):** fit a surrogate response surface `P_title = g(μ, σ)` on the baseline sim, then evaluate candidates analytically via an order-statistic approximation on the affected slot. Refit weekly; **condition on standing and week**, because `∂P/∂σ` flips sign at the playoff cut line.
- **Tier 2 (top ~50):** full CRN sim, re-optimizing lineups only for affected franchises, plus the playoff bracket.

Port `ffsimulator::ff_wins_added()`'s leave-one-out structure, but add the bracket (it stops at regular-season wins), report all-play win% alongside H2H to strip schedule luck, and flag in the UI that the metric is **not additive** — it's a one-term Shapley approximation.

*Calibration anchor:* a 53%-per-matchup team has only ~28% title odds vs 25% for a coin flip. **Playoffs are near-random; the marginal value of seed-improving moves is much lower than managers assume.** If the surrogate disagrees, it's wrong.

### Phase 5 — Decision surfaces
- **Valuation** — VOLS with a **flex-demand fixed point**: replacement level depends on which position wins the flex at the margin, which depends on the projections. Iterate to convergence (3–4 passes). Compute twice: rest-of-season, and playoff-weeks-only.
- **Trades** — objective is **Δ(starting-lineup points) with a free-agent floor, playoff-weighted, subject to strict Pareto improvement for both sides**; chart-sum equality is a negotiation heuristic, not the objective. Build the **verdict layer before refining values** — an ablation on 3,003 judged trades showed the consolidation credit (a dropped roster spot is worth ~425 on FantasyCalc's scale) lifts accuracy 61%→82%, more than the value list itself. Multi-team: multi-edge preference graph → DFS all simple cycles ≤6 → longest-first greedy. Naive single-edge TTC degenerates to 2-cycles under concentrated preferences.
- **FAAB** — budget DP for the shadow price `λ`, then `b* + F(b*)/f(b*) = Δ/λ`. **Fit `F` from your own leagues' bid history** (`Transaction.bid_amount` exposes every historical bid, per manager) rather than population priors. Unspent FAAB has zero salvage, so `λ→0` at season end — the DP correctly predicts "bid $0 most weeks, bid enormous rarely." Bid odd dollars; rivals cluster on 5s and 10s.
- **Streaming (QB/TE/DST/K)** — rolling-horizon IP over the rest of season (roster flow, one streamed starter per week, FAAB budget coupled via the same `λ`). ~1,700 binaries, milliseconds in CBC. Best-minus-worst-starter VORP is 64 pts for DST and 20 for K, and the asset is unpredictable while the matchup is — so **all the value is in the weekly assignment.**
- **Lineups** — swap improves win probability iff `Δμ − z·Δσ > 0`, where `z` is your current standardized margin. Only override the expected-points lineup when `|z| > 0.4` and the `Δμ` sacrifice is under ~2 points; below that, start your studs. Weeks 10–14, `z` is measured against **the playoff cut line, not your opponent**.

### Phase 6 — Dashboard
Cross-league action queue (every recommendation, all leagues, ranked by Δ P(title)) · per-league team view with leverage index · trade explorer · waiver/FAAB board · start/sit with the variance-objective flag · **portfolio view** (exposure to each player across leagues — concentration risk) · market-arbitrage screens.

---

## Phase 7 — The edges (in scope)

You asked for these explicitly, so they're deliverables, not a backlog. Ordered by value-per-effort; each is a module under `decide/edges/` feeding the same Δ P(title) unit and surfacing as its own dashboard panel.

1. **League-mate behavioral modeling — the largest untapped edge.** No academic work exists on it, and ESPN hands you the raw material in the transaction log. Per manager, estimable: recency bias (regress their adds on last-week points vs RoS projection), endowment effect (implied WTA for owned vs WTP for same-tier unowned — experiments put WTA/WTP ≈ 2×), sunk-cost (draft round vs hold duration, controlling for production), name-brand bias, home-team bias, FAAB depletion curve, and transaction-timestamp latency. The FAAB curve **directly parameterizes** the bidding model.
2. **The leverage index.** Tell me each week which decisions actually matter. A start/sit call in a blowout is worth one-seventh of the same call in a coin-flip week.
3. **ESPN's own analyst boards as a signal.** `player.rankings` carries per-analyst ranks — Mike Clay (id 7), Karabell, Cockcroft, Field Yates, Bowen, Moody, Loza — separately from the consensus. Clay's projections *are* what power the ESPN game, so disagreement between Clay and the consensus is a leading indicator of where your league-mates' defaults will move.
4. **Opponent-adjusted defense-vs-position.** Raw "fantasy points allowed" measures schedule, not defense, and is corrupted by game-script endogeneity (bad defenses trail → opponents pass more). A two-way `offense + defense + home` decomposition with empirical-Bayes shrinkage moved defenses a mean of 3.6 ranks of 32 in testing, one moving 11 places. Nobody sells this cheaply.
5. **Props-implied projections.** FanDuel's `ALT_` ladders give 11–13 rungs per player — effectively a full survival function, so you can recover a **mean** rather than the median a single line gives you. On one test case that was a 41% difference. Underdog publishes literal `Fantasy Points` props — a market-set projection needing no modeling at all.
6. **Bench slots as option value, not expected points.** `E[max] > max[E]`; a bench player's value is a weekly exchange option. Under point projections bench depth is worth literally zero — the strongest argument for simulating at all. Corollary: a stash only earns its slot if it *isn't re-acquirable later*, which is exactly why handcuffs and IR returnees dominate.
7. **The waiver timing window.** Sunday night → Tuesday night is the exploitable gap: claims are blind, and snap counts and route participation post Monday/Tuesday. Highest signal, lowest attention. Submit as late as possible.
8. **Stochastic over-performance detection.** Split metrics into stable (target share, air-yards share, WOPR, route participation, red-zone share — target share stabilizes in ~3 games) vs unstable (TD rate, YAC, catch rate over expected, YPC → regress to positional mean). Flag directly off `ffopportunity`'s `*_diff` columns. Archetype: Kupp 2019 at 10 TDs vs 5.6 expected — 26 points of pure variance that won't repeat.
9. **Don't build an injury-prediction model.** Best published models have RMSE ≈ their own mean. Price off position base rates only. Two corrections worth encoding: the RB-vs-WR injury gap is ~0.3 games/season (handcuff value comes from *concentration of opportunity*, not RB fragility), and games-missed does **not** predict next-season decline (R²=0.005).
10. **Reject "buy low / sell high"** as a frame — it's market timing and conflates actual with perceived value. The rule is *buy undervalued, sell overvalued, regardless of price level.* What does hold is systematic mispricing: in a study of AFL picks, each additional time a pick changed hands cut the drafted player's exit hazard by 0.269 with no performance advantage — the more you paid, the longer you irrationally hold.
11. **Frame trade offers around what the counterparty receives**, not what they give up. Eye-tracking work shows sellers fixate on the good and buyers on the price, and attention amplifies perceived importance.
12. **Check whether any league pays for points-for.** High-stakes formats (NFFC) pay a points-leader bonus and admit wildcards on total points. In those, a tool optimizing weekly win probability **actively destroys value**. Make the objective per-league configurable.

**Anti-features** the research says to skip: hard-coded positional priors or "winning roster shapes" (published ones contradict each other across sources and years); naive contrarianism (common constructions outscore rare ones); paying a real price to avoid bye weeks (~0.7% of a season); trusting preseason playoff SoS (Vegas totals already contain every input).

---

## Data stack

All 17 core free endpoints were verified returning 200 in a single pass. Highlights, with the traps that matter:

| Source | Use | Note |
|---|---|---|
| ESPN `leaguedefaults/3` | the thing we arbitrage against | no auth; ESPN vs RotoWire season projections differ by **MAE 20.5 pts** across 240 players — the surface is real |
| ESPN `propBets` (`sports.core.api`) | line movement | **the only prop source pre-joined to ESPN athlete IDs**; carries `open` *and* `current` targets. Values only, no prices — can't devig |
| Sleeper `/projections/nfl/2026/{wk}` | component-level projections | undocumented, no auth; gives `rec_tgt`, `rush_att`, air-yard buckets + 12 ADP flavors |
| Sleeper `/trending/add` | rival demand | feeds `n` in the FAAB bid model |
| Underdog `/beta/v6/over_under_lines` | market-set fantasy projection | a literal `Fantasy Points` stat type — no modeling needed. Half-PPR; UUIDs only, so name-match |
| FanDuel `sbapi` event-page | **the ALT ladders** | plain curl works, `_ak` is static. A bad slug returns **200 with empty attachments** — alarm on empty, not status |
| Pinnacle guest Arcadia | two-sided prices | devig-able, and `maxRiskStake` is a sharpness signal DK/FD don't publish |
| nflverse `games.csv` | schedule + closing Vegas lines | 7,388 games back to 1999 — free historical totals for backtesting |
| nflverse `stats_player_week` | the DvP input | |
| MFL `TYPE=injuries` / `adp` / `aav` | injuries + auction values | **replaces nflverse's dead injuries feed**; AAV from 572 real auctions |
| RotoWire RSS | fastest news | minute-fresh; beat ESPN's own data on a live practice report during research |
| Open-Meteo + api.weather.gov | weather | keyless; nflverse `airports.csv` has team lat/long |

**Establish the Run — confirmed: manual CSV download only.** No API, no session-accessible JSON. There's a CSV link on each rankings chart, refreshed daily at 9am. Their ToS explicitly prohibits automated scraping and above-human request volumes, so **we will not build a cookie-replay scraper** — that's the one paid source where the language is unambiguous. Ingestion is a watched folder: you drop the weekly CSV, a loader normalizes it to our IDs and validates columns on every ingest (the schema is undocumented and can move). ETR then plugs into the ensemble as one more component-stat contributor. Worth it for the human projections plus their explicit "Suggested 1% FAAB Bid" per player.

**Paid stack, ~$108/mo.** Start with **FantasyPros $8.99** — best value in the whole brief: ECR + projections + practice reports + **an ESPN ID crosswalk via `external_ids`**, the join the entire tool is keyed on. Its practice fields carry `team_practice_N_submitted` booleans that distinguish "team hasn't filed yet" from "player isn't on the report" — without which a Wednesday model reads missing data as good news. Then balldontlie $39.99 (dated practice arrays + Sunday inactives) and The Odds API $59 **for historical backfill only** — buy one month at the $119 tier, backfill, downgrade. Live props are free to pull.

**ID spine:** nflverse `roster_2026` first (86% ESPN coverage, daily), fall back to DynastyProcess `db_playerids`, FantasyPros `external_ids` as a third check. Two traps: **missing values in `db_playerids` are the literal string `"NA"`**, so naive truthiness reports a fake 100% coverage; and **Sleeper is no longer a usable ESPN crosswalk** (25%). **Hard-code a 32-row D/ST table** keyed on nflverse `team_abbr` — every join failure in testing was a team defense, because every platform names them differently.

**Kill list** (don't waste time): nflverse `injuries` (dead since March) · `nfl_data_py` (deprecated) · `fantasy.espn.com/apis/v3` and `site.api.espn.com` (both blocked) · FAABLab (live API, 2022-vintage data) · PFF (projections not exposed at any price) · direct DK/BetMGM/Caesars sportsbook (Akamai) · DK contest scraping (`Disallow: /contest/`, needs a real-money login — account-suspension risk) · PFR scraping (nflverse already carries it) · Reddit JSON (OAuth mandatory now).

---

## Verification

- **Scoring:** assert `appliedTotal` reproduction across a full boxscore, all positions, every league. Fails loudly at startup.
- **Lineup solver:** assert greedy == PuLP optimum on a random sample every run.
- **Simulator calibration:** simulated team-score distribution must match the empirical anchor — mean ~122, SD ~24.4, skew ~+0.27 for 12-team PPR. Backtest on 2025 actuals.
- **Projections:** hold-out MAE per position vs raw ESPN, on the snapshot corpus. The ensemble must beat ESPN or something's wrong.
- **Δ P(title):** sanity-check against the 28%-for-a-53%-team anchor; verify a null move returns 0.00 ± MC noise under CRN.
- **End-to-end:** `fq sync && fq report --all-leagues` produces a ranked action queue across every discovered league; then drive the dashboard in a browser and confirm the numbers match the CLI.
- **Ops canary:** daily authed request that alerts when `espn_s2` dies, plus a schema canary on `NFL_MIGRATION`.

---

## Repo and delivery

- **Public GitHub repo under `blobspire`** (already the active `gh` account; git identity is set). `gh repo create fantasy_quant --public --source=. `, initialized in this directory.
- **Branch-per-phase, PR per phase**, squash-merged. I'll self-approve and merge within this session, as you authorized. Commits land incrementally within a phase, not one dump at the end.
- **Secrets never touch the repo.** `SWID`/`espn_s2` and API keys live in a git-ignored `.env`; `.env.example` is committed. Since the repo is public, the snapshot corpus under `data/` is git-ignored too — it's large, it's rebuildable-forward-only, and it's yours. Add a `.gitignore` before the first commit, not after.
- **CI:** a GitHub Actions workflow running `ruff` + `pytest` on PRs. The scoring-reproduction and greedy-vs-LP assertions run there as real tests.
- Execution uses a **dynamic workflow** — parallel agents per phase where the work is genuinely independent (e.g. the edge modules in Phase 7, the data adapters in Phase 2), serial where phases depend on each other.

---

## Open items

1. **Fan API response shape** — endpoint verified reachable and cookie-authenticated, but the response body shape is community-reported rather than confirmed. Log the raw response on first run and adapt; a manual league-ID list is the fallback.
2. **Your ESPN cookies** — I'll need `SWID` and `espn_s2` from your browser (DevTools → Application → Cookies → espn.com) before anything league-specific works. Phase 0 needs no auth, so this isn't blocking today.
3. **Which leagues pay for points-for.** If any use a total-points bonus or wildcard (common in high-stakes formats), the objective inverts there — optimizing weekly win probability actively destroys value. It's per-league configurable; I just need you to tell me which.
