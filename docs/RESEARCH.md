# Research notes

Empirically verified against live endpoints, 2026-09-06/07. Where a claim was checked
against our own data it is marked **[measured]**. Where research and our data disagree,
**our data wins** — the numbers here are what the code should use.

## ESPN API

**Host:** `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl`. Everything else is a
dead end: `fantasy.espn.com/apis/v3` 302s to a marketing page (since Apr 2024), and
`site.api.espn.com` 403s from datacenter IPs regardless of headers. `site.web.api.espn.com`
does work.

**Auth:** SWID (braces included) + `espn_s2` cookies. No OAuth/bearer path exists.
`espn_s2` lasts ~a year and dies silently, usually mid-season.

### Views that work
`mSettings` `mTeam` `mRoster` `mMatchup` `mMatchupScore` `mBoxscore` `mLiveScoring`
`mStandings` `mDraftDetail` `mPositionalRatings` `kona_player_info` `kona_playercard`
`mTransactions2` (auth-gated) `mPendingTransactions` (auth-gated) `allon`.

- `mRoster` **does** honor `scoringPeriodId` (community docs claiming otherwise are wrong).
- `mMatchup` has **no** `playoffTierType` — use `mMatchupScore`/`mBoxscore`.
- `mBoxscore` populates rosters for **one** week per call; loop weeks.
- An unrecognized `view=` returns **HTTP 200 with a default skeleton**, not an error.

### x-fantasy-filter
`limit` without a sort → **HTTP 400**. Always send a sort.

```jsonc
{"players": {
  "filterStatus":  {"value": ["FREEAGENT","WAIVERS","ONTEAM"]},
  "filterSlotIds": {"value": [0,2,4,6,17,16,23]},
  "filterIds":     {"value": [4429795]},          // works WITHOUT a sort
  "filterName":    {"value": "Gibbs"},            // substring; filterFullName is IGNORED
  "filterStatsForSourceIds":    {"value": [1]},   // 0 actual / 1 projected
  "filterStatsForSplitTypeIds": {"value": [0]},   // 0 season / 1 game / 2 ROS
  "filterStatsForExternalIds":  {"value": [2026]},
  "limit": 250, "offset": 0,
  "sortPercOwned":        {"sortPriority": 1, "sortAsc": false},
  "sortAdp":              {"sortPriority": 1, "sortAsc": true},
  "sortDraftRanks":       {"sortPriority": 1, "sortAsc": true,  "value": "PPR"},
  "sortAppliedStatTotal": {"sortPriority": 1, "sortAsc": false, "value": "102026"}
}}
```

`sortAppliedStatTotal` ranks the pool by **league-scored** projected points server-side.
No Python library exposes this; it is the single most useful capability of the API.

**Do not use:** `filterFullName` (silently ignored), `sortAppliedStatTotalForScoringPeriodId`
(accepted, garbage ordering), `sortAuctionValueAverage` (not real — unknown keys are
silently ignored, which is its own trap).

Page on the `x-fantasy-filter-player-count` response header (total *before* limit).

**⚠️ Never send `filterStatsForTopScoringPeriodIds`** — every form we tested silently drops
the weekly *projection* rows and returns actuals only. Send no stat filter; unfiltered
returns ~57 stat rows/player covering both seasons, projected and actual. Pinned by a test.

### stats[] rows
`id` = `{statSourceId}{statSplitTypeId}{externalId}`, so you can select by string prefix.

| src | split | example id | meaning |
|---|---|---|---|
| 0 | 0 | `002026` | actual season total (+`appliedAverage`) |
| 0 | 1 | `01401772835` | actual, one game — suffix is the NFL **event** id |
| 1 | 0 | `102026` | projected season total — **FROZEN at preseason** |
| 1 | 1 | `1120261` | projected week 1 |
| 1 | 2 | `122026` | ROS expressed as a per-game **rate** (never sum it) |

- **Mixed seasons.** A 2026 request returns 2025 rows in the same array. Filter on `seasonId`.
- **The frozen season total** never updates while weeklies are revised all year. Build ROS by
  summing remaining weekly rows. `espn-api`'s `projected_total_points` reads the frozen field.
- Actual weekly rows carry an event id, so `scoringPeriodId` is the only reliable week key.
- `stats` = raw counts (league-independent). `appliedStats` = points per statId (league-scored).
  Raw `stats` contains **pre-computed derived buckets** (47–52 = "every N receiving yards",
  `3↔22`, `24↔40`, `42↔61`, `41↔53`, `100 = 2×99`, `109 = 107+108`). Apply **only** statIds
  that have a `scoringItem`, or you multi-count.
- `player.proTeamId` is the player's **current** team; for historical weeks read the stat
  row's `proTeamId` or traded players get the wrong bye.

### Scoring
```json
{"statId": 53, "points": 0.0, "isReverseItem": false,
 "pointsOverrides": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 1.0, "15": 0.5}}
```
**`pointsOverrides` keys are `defaultPositionId`, NOT `lineupSlotId`.** The override
**replaces** `points` for that position (it does not add). A `points: 0` item is not unscored.
`pointsOverrides` may be **absent** rather than `{}` — always `.get(...)`.

Verified: parsing this way reproduces `appliedTotal` **160/160 exactly** on a full boxscore.
Assert that at startup; it is the regression test against ESPN changing anything.

### Two colliding ID spaces — the top source of silent corruption
`defaultPositionId`: 1 QB · 2 RB · 3 WR · **4 TE** · 5 K · 7 P · 9 DT · 10 DE · 11 LB ·
12 CB · 13 S · 14 HC · **15 TQB** · 16 D/ST · 17 EDR · 18 BE

`lineupSlotId`: 0 QB · 1 TQB · 2 RB · 3 RB/WR · **4 WR** · 5 WR/TE · 6 TE · **7 OP
(superflex)** · 8 DT · 9 DE · 10 LB · 11 DL · 12 CB · 13 S · 14 DB · **15 DP** · 16 D/ST ·
17 K · 18 P · 19 HC · 20 BE · 21 IR · 22 INV · **23 FLEX** · 24 EDR · 25 ALL

They **disagree at 4 and 15**. `lineupSlotCounts` keys are slot IDs; `positionLimits` and
`pointsOverrides` keys are position IDs.

`proTeamId`: 0 FA · 1 ATL · 2 BUF · 3 CHI · 4 CIN · 5 CLE · 6 DAL · 7 DEN · 8 DET · 9 GB ·
10 TEN · 11 IND · 12 KC · 13 LV · 14 LAR · 15 MIA · 16 MIN · 17 NE · 18 NO · 19 NYG ·
20 NYJ · 21 PHI · 22 ARI · 23 PIT · 24 LAC · 25 SF · 26 SEA · 27 TB · 28 WSH · 29 CAR ·
30 JAX · 33 BAL · 34 HOU. **31 and 32 do not exist.**

Key statIds: 3 PY · 4 PTD · 20 INTT · 24 RY · 25 RTD · 42 REY · 43 RETD · **53 REC** ·
58 targets · 72 FUML · 95 INT · 96 FR · 99 SK · **103 INT-return TD / 104 fumble-return TD**
· 106 FF · 107 TKA · 108 TKS · 109 TK · 113 PD · 89–92 & 121–125 points allowed ·
128–136 yards allowed · **211/212/213 passing/rushing/receiving first downs (PPFD)**.

### Generate constants, don't hand-maintain them
```
GET {BASE}/seasons/{year}?view=chui_default_platformsettings
```
No auth. Returns `statSettings.stats` (all 235 statIds with `abbrev` and `apiIdentifier`),
`positions`, `lineupSlots` **with `eligiblePositions`**, `proTeams` (with `byeWeek`),
`statIdToOverridePosition`, and every enum under `types`/`typeNames`.

### League-type detection
| Format | Detect via |
|---|---|
| Redraft | `draftSettings.keeperCount == 0 && keeperCountFuture == 0` |
| Keeper | `keeperCount > 0`; per-player `keeperValue` (auction $ in auction, draft ROUND in snake) |
| Auction vs snake | `draftSettings.type` ∈ OFFLINE(0) SNAKE(1) AUTOPICK(2) SNAIL(3) AUCTION(4) LINEAR(5). `auctionBudget` is present even in snake — not a discriminator |
| **FAAB** | **`acquisitionSettings.isUsingAcquisitionBudget`** — NOT `acquisitionType`, which is the processing model |
| Superflex | `lineupSlotCounts["7"] > 0` or `["0"] >= 2` |
| IDP | `lineupSlotCounts` nonzero on 8–15 |
| TE premium | `pointsOverrides["4"] > pointsOverrides["3"]` on statId 53 |
| Median scoring | `scoringSettings.scoringEnhancementType == "WIN_BONUS_TOP_HALF"` |

`scheduleSettings.matchupPeriods` maps matchupPeriodId → [scoringPeriodId…] and inner lists
are **unsorted**. `matchupPeriodId ≠ scoringPeriodId`. `playoffMatchupPeriodLength: 0` does
not mean "no playoffs" — read `playoffMatchupPeriodLengthByRound` when
`variablePlayoffMatchupPeriodLength`. `-1` means unlimited. Use `status.latestScoringPeriod`,
never a computed week.

### Multi-league discovery
No "list my leagues" endpoint exists (`/leagues` 405s). Use the Fan API:
```
GET https://fan.api.espn.com/apis/v2/fans/{SWID}?featureFlags=expandAthlete
Cookies: SWID={...}; espn_s2=...
```
Filter `preferences` to `typeId == 9` and `metaData.entry.gameId == 1`; read
`groups[].groupId` (league) and `entryId` (your team). **Response shape is community-reported,
not verified** — log the raw body on first run and adapt. Keep a manual override list.

### Errors
401 `AUTH_MISSING_CREDENTIALS` / `AUTH_LEAGUE_NOT_VISIBLE` · 404 for a private league **and**
for one that doesn't exist (indistinguishable without cookies) · 400 on limit-without-sort.
No rate limit observed (60 requests at concurrency 20, no 429). Be polite anyway: 2–5 req/s
and `If-None-Match`.

---

## Calibration constants **[measured]**

From our own corpus, 23,999 paired player-weeks, ESPN PPR, seasons 2022–2025.
Regenerate with `fq backfill`. **These supersede the research figures.**

| pos | n | MAE | RMSE | weekly slope | P(actual ≤ 0) | skew | excess kurt |
|---|---|---|---|---|---|---|---|
| QB | 2,210 | 5.87 | 7.43 | 0.948 | **3.2%** | 0.28 | −0.01 |
| RB | 6,312 | 4.10 | 5.76 | 0.921 | **18.5%** | 1.40 | 2.13 |
| WR | 10,004 | 4.27 | 5.92 | 0.943 | **25.9%** | 1.45 | 2.31 |
| TE | 5,473 | 3.24 | 4.65 | 0.961 | **32.5%** | 1.79 | 4.02 |

**σ(μ) = 3.67 + 0.273·μ** (research claimed 2.5 + 0.30·μ).

Positive skew and excess kurtosis at every position ⇒ **hurdle gamma**, not normal.
Note the weekly slopes (~0.92–0.96) are NOT the season-level calibration slopes
(QB 0.67 / RB 0.79 / WR 0.85 / TE 0.72) — don't conflate them.

**Residual correlation, block-diagonal by NFL team** (a Gaussian copula over gamma
marginals): ρ(QB,WR)=0.30 · ρ(QB,TE)=0.20 · ρ(QB,RB)=0.08 · ρ(TE,TE)=0.13 ·
ρ(RB,RB)=−0.09 · everything else 0. Different-team pairs correlate **+0.003** and a
league-wide weekly factor explains **0.71%** of residual variance — there is no market
factor, so the matrix really is block-diagonal.

**Team-score anchor** (12-team PPR, 9 skill starters, optimal lineups): mean 121.9,
SD 24.35, skew +0.27. σ_D = √2 × 24.35 = 34.4, so **1 projected point ≈ 1.16 pp of weekly
win probability** at an even matchup — falling to 32% of that at |z|=1.5 and 14% at |z|=2.

Injury hazard per game: RB 5.2% · WR 4.5% · TE 4.9% · QB 2.5%. Must be **absorbing**
(hazard + duration), not IID per week. Lineup-efficiency haircut for modelling opponents:
×0.775 ± 0.05.

---

## External sources

All verified 200 on 2026-09-06.

| Source | URL |
|---|---|
| Sleeper projections | `https://api.sleeper.app/projections/nfl/{season}/{week}?season_type=regular&position[]=QB&order_by=ppr` |
| Sleeper trending | `https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=25` |
| Underdog props | `https://api.underdogfantasy.com/beta/v6/over_under_lines` (has a literal `Fantasy Points` stat type) |
| FanDuel props | `https://sbapi.nj.sportsbook.fanduel.com/api/event-page?_ak=FhMFpcPWXMeyZxOx&eventId={id}&tab=receiving-props` |
| ESPN propBets | `https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{gid}/competitions/{gid}/odds/100/propBets` |
| Pinnacle | `https://guest.api.arcadia.pinnacle.com/0.1/leagues/889/matchups` and `/markets/straight` |
| nflverse | `https://github.com/nflverse/nflverse-data/releases/download/{tag}/{file}` |
| MFL | `https://api.myfantasyleague.com/{season}/export?TYPE=injuries&W={wk}&JSON=1` (also `adp`, `aav`) |
| FantasyPros ECR | `https://partners.fantasypros.com/api/v1/consensus-rankings.php?sport=NFL&year=...&week=...&position=ALL&type=weekly&scoring=PPR` |
| db_playerids | `https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv` |
| FantasyCalc | `https://api.fantasycalc.com/values/current?isDynasty=false&numQbs=1&numTeams=12&ppr=1` |
| Weather | `https://api.open-meteo.com/v1/forecast?...` and `https://api.weather.gov/points/{lat},{lon}` |

nflverse tags/files that matter: `schedules/games.csv` (has `spread_line`, `total_line`,
`roof`, and `espn`/`pfr` game-id crosswalks, back to 1999) · `stats_player/stats_player_week_{season}.csv.gz`
· `rosters/roster_{season}.csv.gz` · `depth_charts/` (47 MB, append-only, dedupe on `dt`) ·
`snap_counts/` · `nextgen_stats/` (**consolidated files only — per-season are empty stubs**) ·
`pfr_advstats/` · `players/players.csv`. Poll `timestamp.txt` per tag rather than
re-downloading. Re-pull Wed/Thu for stat corrections.

**FanDuel ALT ladders → a mean, not a median.** `PLAYER_X_ALT_*` markets give 11–13 rungs =
a survival function S(k)=P(X≥k). Do **not** sum implied probabilities — each rung carries
5–8% overround and it compounds. Devig by anchoring: fit a lognormal to the rungs subject to
S(main_line)=0.5. Worked example: posted median 30.5 → fitted median 30.9, **fitted mean 43.6**
(+41%). Fantasy scoring is a mean; sportsbook lines are medians. For anytime-TD, devig first
then λ = −ln(1−p) (at +150: naive 0.400 → 0.511, +27.7%).
FanDuel tab slugs are kebab-case (`receiving-props`), not the numeric ids in `layout.tabs`,
and a bad slug returns **200 with empty `attachments`** — alarm on empty, not on status.

**ID crosswalk.** Prefer nflverse `roster_{season}` (86% ESPN coverage, daily), fall back to
`db_playerids` (73%). **Missing values in db_playerids are the literal string `"NA"`**, so
naive truthiness reports a fake 100% coverage. **Sleeper is no longer a usable ESPN crosswalk**
(25%). **Hard-code a 32-row D/ST table** keyed on nflverse `team_abbr` — every join failure in
testing was a team defense, because every platform names them differently.

**Dead / not worth it:** nflverse `injuries` (nothing since 2026-03) · `nfl_data_py`
(deprecated → `nflreadpy`, or just read the Parquet URLs) · FAABLab (live API, 2022 data) ·
PFF (projections not exposed at any price) · direct DK/BetMGM/Caesars (Akamai) · DK contest
scraping (`Disallow: /contest/`, needs a real-money login) · PFR scraping · Reddit JSON.

**Establish the Run:** manual CSV download only, no API. Their ToS explicitly prohibits
automated scraping — do not build a cookie-replay scraper. Ingest via a watched folder.

---

## Method notes

**Valuation.** VOLS with a flex-demand fixed point: replacement level depends on which
position wins the flex at the margin, which depends on the projections — iterate (3–4 passes
converges). 12-team 1QB/2RB/3WR/1TE/1FLEX PPR lands near QB12/RB30/WR42/TE12. Compute twice:
rest-of-season and playoff-weeks-only.

Positional scarcity, fitted `pts(rank) = A·e^{−b·rank}` over the top 60: QB b=0.040 (half-life
17 ranks) · RB 0.026 (26) · **WR 0.013 (52)** · TE 0.039 (18). WR's curve is ~4× flatter than
QB/TE — the quantitative core of late-round-QB and Zero-RB. Share of total positive VORP:
RB 41.9% · WR 46.8% · TE 6.5% · QB 4.8%.

**Lineup solving.** Greedy is **provably exact** for nested slot eligibility
(dedicated ⊂ FLEX ⊂ SUPERFLEX) — fill dedicated slots best-first, then flex from the
remainder. Don't run an LP in the inner loop; keep one as a correctness oracle.

**Start/sit under uncertainty.** A swap improves win probability iff **Δμ − z·Δσ > 0**, where
z is your current standardized margin. Favorite ⇒ reject variance; underdog ⇒ accept negative
Δμ for enough Δσ. Only override the expected-points lineup when |z| > 0.4 and the Δμ sacrifice
is under ~2 points. Weeks 10–14, measure z against the **playoff cut line**, not the opponent.
Variance-tuning is worth ~0.3–0.5 pp/week and **flips sign** with standing.

**Monte Carlo.** Use common random numbers on paired comparisons — 2,000 sims suffice where
14,400 would be needed independently. Two-tier: an analytic surrogate `P_title = g(μ,σ)` for
screening thousands of candidates, then full CRN sim for the top ~50. Refit the surrogate
weekly and **condition it on standing and week**, because ∂P/∂σ flips sign at the cut line.
Calibration anchor: a 53%-per-matchup team has only ~28% title odds vs 25% for a coin flip —
playoffs are near-random, so seed-improving moves are worth less than managers assume.

**Ensembling.** Combine at the **component-stat level**, not the points level, so projections
transfer across leagues with different scoring. **Equal weights** (equal weighting beat
accuracy-weighting in 64% of head-to-heads over 12 seasons; source accuracy doesn't persist).
Hodges–Lehmann location estimator (median of Walsh averages). **Zero-weight FantasyPros and
FantasyFootballNerd as inputs** — they are themselves consensuses and would double-count.
Handle missing sources by renormalizing weights, not imputing.

**Trades.** Objective: Δ(starting-lineup points) with a free-agent floor, playoff-weighted,
subject to strict Pareto improvement for both sides. Chart-sum equality is a negotiation
heuristic, not the objective. **Build the verdict layer before refining values** — on 3,003
judged trades, a consolidation credit (a dropped roster spot ≈ 425 on FantasyCalc's scale)
lifted accuracy 61%→82%, more than the value list itself. Positional decay for surplus:
multiply incoming value by ≈ 1 − (rostered − required)/rostered. Multi-team: multi-edge
preference graph → DFS all simple cycles ≤6 → longest-first greedy selection. Naive
single-edge top-trading-cycles degenerates to 2-cycles under concentrated preferences.

**FAAB.** Budget DP for the shadow price λ, then bid `b* + F(b*)/f(b*) = Δ/λ`. Unspent FAAB
has zero salvage so λ→0 at season end, which correctly yields "bid $0 most weeks, bid enormous
rarely". Fit F from your own league's bid history (`Transaction.bid_amount`), not population
priors. n is the 2–5 teams actually bidding, not league size. Winner's curse (haircut ≈
σ√(2 ln n)) is separate from shading — apply once each, to different quantities. Bid odd
dollars; rivals cluster on 5s and 10s. Population prior: median winning bid 1.4% of budget,
lognormal with a fat right tail.

**Claim EV is submodular in adds**, and `E[max] > max[E]` means bench depth is worth literally
zero under point projections — the strongest argument for simulating. Blocking a rival is worth
only ~1/(N−1) of their gain.

**Streaming (QB/TE/DST/K).** Rolling-horizon IP: roster flow `h[i,w] = h[i,w−1] + a − d`,
start⇒hold, one streamed starter per week, roster-size and FAAB-budget constraints (λ couples
it to the bidding model). ~1,700 binaries, milliseconds in CBC. Season best-minus-worst-starter
VORP: WR 224.7 · RB 194.7 · QB 134.2 · TE 122.6 · DST 64.0 · K 20.0.

**Stable vs unstable metrics.** Project forward: target share, air-yards share, WOPR, route
participation, red-zone touch share (target share stabilizes in ~3 games). Regress to
positional mean: TD rate, YAC, catch rate over expected, YPC, FPOE. WOPR's 1.5/0.7 coefficients
are undocumented folklore — worth re-fitting on nflverse data.

**Do not build an injury-prediction model.** Best published models have RMSE ≈ their own mean.
Price off position base rates. The RB-vs-WR injury gap is ~0.3 games/season; handcuff value
comes from concentration of opportunity on absence, not RB fragility. Games missed does **not**
predict next-season decline (R²=0.005).
