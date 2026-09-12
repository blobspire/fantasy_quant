/**
 * The wire contract.
 *
 * Every type in this file mirrors a payload that already exists in
 * `src/fantasy_quant/report.py` (and `edges/portfolio.py`). The dashboard is a
 * viewer: it does not project, simulate, rank, or decide what is significant.
 * If you are about to write arithmetic on a probability here, stop -- the CLI
 * already computed it and the two must not be allowed to disagree.
 *
 * UNITS, once, so no view has to guess:
 *   - Probabilities and probability deltas cross the wire as FRACTIONS.
 *     `delta_title: 0.0277` is +2.77 percentage points. `fmt.pp` and `fmt.pct`
 *     are the only places that multiply by 100.
 *   - `stderr` is in the same unit as the quantity it qualifies.
 *   - Points are points. `delta_points` is not a probability and never renders
 *     as one.
 *
 * SIGNIFICANCE IS SERVER-SIDE. `significant` and `verdict` come off
 * `report.verdict_for`, which applies a different test per surface (a trade
 * uses a selection-adjusted threshold; a lineup's zero-effect case is
 * structural). The client never derives either from `delta` and `stderr`.
 */

/* ==========================================================================
   Configuration
   ========================================================================== */

/** Same-origin by default: the API serves `web/dist` and mounts the JSON here. */
export const API_BASE = '/api';

/** A cold league simulation is ~3s and the decision surfaces are slower. */
const DEFAULT_TIMEOUT_MS = 120_000;

/**
 * The cross-league endpoints run EVERY surface in EVERY league before they can rank
 * anything, so they are not slow versions of the single-league calls — they are a
 * different order of work.
 *
 * Measured cold on the three live leagues at 4,000 sims: `/queue` takes **334s**, of
 * which the waiver boards alone are ~74s each. At the 120s default the browser gave up
 * every time on a fresh server while the server carried on and finished, so the next
 * attempt returned in 0.0s from cache — a timeout that reported failure for work that
 * was succeeding. Ten minutes is the measured cost with headroom, and the server caches
 * with no TTL, so this is paid once per server run and never again.
 */
const CROSS_LEAGUE_TIMEOUT_MS = 600_000;

/**
 * `report.WEEKLY_SECTIONS`, in its order. The server rejects a name it does not
 * know (deliberately: a silently ignored `skip` means the slow trade search ran
 * after all), so this list must stay in step with the Python one.
 */
export const WEEKLY_SECTION_NAMES = [
  'odds',
  'leverage',
  'lineup',
  'waivers',
  'trades',
  'stream',
] as const;

/* ==========================================================================
   Errors
   ========================================================================== */

/**
 * A failed request, with enough structure for a view to render a partial page.
 *
 * The API reports per-league failures two ways and both land here: a non-2xx
 * with a JSON body, and a 200 whose payload carries `ok: false` (which is how
 * `report._section` reports a surface that blew up without costing the reader
 * the rest of the page). `leagueId` is set when the failure is scoped to one
 * league, so a three-league view can grey out one card instead of erroring
 * whole.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: string;
  readonly leagueId: number | null;
  readonly retryable: boolean;

  constructor(init: {
    message: string;
    status?: number;
    code?: string;
    detail?: string;
    leagueId?: number | null;
    retryable?: boolean;
  }) {
    super(init.message);
    this.name = 'ApiError';
    this.status = init.status ?? 0;
    this.code = init.code ?? 'error';
    this.detail = init.detail ?? '';
    this.leagueId = init.leagueId ?? null;
    this.retryable = init.retryable ?? (this.status === 0 || this.status >= 500);
  }
}

/** True for the abort a component fires when it unmounts mid-flight. */
export function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/* ==========================================================================
   Cache metadata
   ==========================================================================
   A cold computation is slow enough that the API caches it, so every payload
   says when it was computed. The UI shows the age rather than pretending the
   number is live: "computed 4 minutes ago" is a fact, an unlabelled stale
   probability is a lie.
   ========================================================================== */

export interface Meta {
  /** ISO-8601 UTC instant the underlying simulation finished, if known. */
  computed_at: string | null;
  /** Server's own age for the cached entry, in seconds. Preferred over local clock arithmetic. */
  stale_seconds: number | null;
  /** Whether this response was served from cache rather than computed now. */
  cached: boolean;
  /** Whether a refresh for this resource is currently running server-side. */
  computing: boolean;
  /** `report.SCHEMA_VERSION` -- bump means the shapes below may have moved. */
  schema_version: number | null;
  /** Simulations behind the numbers. Drives every Monte Carlo error on the page. */
  n_sims: number | null;
  /** Common-random-numbers seed. Two payloads with different seeds are not comparable. */
  seed: number | null;
  /** `report._envelope`'s timestamp, when the payload carries one. */
  generated_at: string | null;
}

/** Any payload, plus the freshness facts about it. */
export type WithMeta<T> = T & { meta: Meta };

const EMPTY_META: Meta = {
  computed_at: null,
  stale_seconds: null,
  cached: false,
  computing: false,
  schema_version: null,
  n_sims: null,
  seed: null,
  generated_at: null,
};

function pickNumber(source: Record<string, unknown>, key: string): number | null {
  const value = source[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function pickString(source: Record<string, unknown>, key: string): string | null {
  const value = source[key];
  return typeof value === 'string' && value ? value : null;
}

/**
 * Read freshness off a payload, whether the API puts it at the top level or in
 * a nested `meta` object. Tolerating both is three lines and saves a lockstep
 * change with the API module.
 */
export function readMeta(payload: unknown): Meta {
  if (!payload || typeof payload !== 'object') return { ...EMPTY_META };
  const top = payload as Record<string, unknown>;
  const nested =
    top.meta && typeof top.meta === 'object' ? (top.meta as Record<string, unknown>) : {};
  const src: Record<string, unknown> = { ...top, ...nested };
  return {
    computed_at: pickString(src, 'computed_at') ?? pickString(src, 'generated_at'),
    stale_seconds: pickNumber(src, 'stale_seconds') ?? pickNumber(src, 'age_seconds'),
    cached: src.cached === true,
    computing: src.computing === true || src.refreshing === true,
    schema_version: pickNumber(src, 'schema_version'),
    // The envelope calls it `sims`; the payloads inside call it `n_sims`.
    n_sims: pickNumber(src, 'n_sims') ?? pickNumber(src, 'sims'),
    seed: pickNumber(src, 'seed'),
    generated_at: pickString(src, 'generated_at'),
  };
}

/** Outer wins where it has an answer; the inner payload fills the rest. */
function mergeMeta(outer: Meta, inner: Meta): Meta {
  return {
    computed_at: outer.computed_at ?? inner.computed_at,
    stale_seconds: outer.stale_seconds ?? inner.stale_seconds,
    cached: outer.cached || inner.cached,
    computing: outer.computing || inner.computing,
    schema_version: outer.schema_version ?? inner.schema_version,
    n_sims: outer.n_sims ?? inner.n_sims,
    seed: outer.seed ?? inner.seed,
    generated_at: outer.generated_at ?? inner.generated_at,
  };
}

/**
 * Age of a payload in seconds: the server's number if it sent one, otherwise
 * derived from `computed_at`. The server's is preferred because it is immune to
 * a browser clock that disagrees with the host.
 */
export function ageSeconds(meta: Meta | null | undefined, now: number = Date.now()): number | null {
  if (!meta) return null;
  if (meta.stale_seconds !== null) return Math.max(0, meta.stale_seconds);
  if (!meta.computed_at) return null;
  const then = Date.parse(meta.computed_at);
  if (Number.isNaN(then)) return null;
  return Math.max(0, (now - then) / 1000);
}

/* ==========================================================================
   Sections: a payload part that can fail on its own
   ==========================================================================
   `report._section` wraps every section so a trade search that blows up does
   not cost the reader his waiver board. Views must handle both arms.
   ========================================================================== */

export type Section<T> = (T & { ok?: true }) | { ok: false; error: string };

/** The section's payload, or null if it failed. */
export function sectionData<T>(section: Section<T> | null | undefined): T | null {
  if (!section) return null;
  if ((section as { ok?: boolean }).ok === false) return null;
  return section as T;
}

/** The section's error message, or null if it succeeded. */
export function sectionError<T>(section: Section<T> | null | undefined): string | null {
  if (!section) return null;
  const s = section as { ok?: boolean; error?: string };
  return s.ok === false ? (s.error ?? 'section failed') : null;
}

/* ==========================================================================
   Shared shapes
   ========================================================================== */

/**
 * Which significance test the producing surface applied, and what it concluded.
 * Four states rather than a boolean, straight off `report.Verdict`:
 *
 *   act   -- cleared its own error. The only kind the queue ranks on.
 *   noise -- a real measurement that did not clear its error. Must still show.
 *   null  -- nothing was measured (a hold, an already-optimal lineup). Its zero
 *            stderr means *nothing was measured*, not *measured precisely*.
 *   harm  -- resolved, in the wrong direction. NOT the same fact as noise.
 */
export type Verdict = 'act' | 'noise' | 'null' | 'harm';

export type Confidence = 'high' | 'medium' | 'low' | string;

export interface PlayerRef {
  player_id: number;
  name: string;
  /**
   * Where OUR projections put him at his position this rest-of-season, e.g. `"WR29"`.
   * Absent for a player we do not project.
   */
  espn?: string;
  espn_rank?: number;
  /** Where the analyst board puts him, e.g. `"WR45"`. Absent when it does not rank him. */
  etr?: string;
  etr_rank?: number;
  /** `espn_rank - etr_rank`: positive when the analyst likes him MORE than we do. */
  gap?: number;
}

export interface MovedPlayer extends PlayerRef {
  from_team: number;
  to_team: number;
}

export interface TeamRef {
  team_id: number;
  name: string;
}

/** `report.rec_payload`: one `core.Recommendation`, named and judged. */
export interface Recommendation {
  kind: string;
  league_id: number;
  /** Change in championship probability, as a fraction. The unit of everything. */
  delta_title: number;
  /** Points are an intermediate quantity, never the answer. */
  delta_points: number;
  /** Monte Carlo error on `delta_title`, same units. */
  stderr: number;
  /** |delta| / stderr, or null when nothing was measured. */
  z: number | null;
  /** Server-side verdict. Never recomputed here. */
  significant: boolean;
  verdict: Verdict;
  verdict_note: string;
  /** Marginal win probability per point, relative to a coin flip. 1.00 = every point lands. */
  leverage: number;
  confidence: Confidence;
  tags: string[];
  rationale: string;
  receive: PlayerRef[];
  send: PlayerRef[];
  players: MovedPlayer[];
  /** `report.caveats_for` -- what is wrong with the NUMBER. Present on trades and streams. */
  caveats?: string[];
}

/* ==========================================================================
   /api/leagues
   ========================================================================== */

/**
 * One league in the switcher. `ok: false` is expected and must render: ESPN
 * refuses a league now and then, and the other two still work.
 */
/**
 * One row of `/api/leagues`: the registry entry plus a live `mSettings` probe.
 *
 * The probe answers in `reachable`, NOT in `ok` -- there is no `ok` on a row,
 * and a switcher that tested `row.ok === false` would render a league whose
 * cookies expired as a perfectly ordinary chip. Use `leagueFailed()`.
 *
 * The probe is deliberately not a simulation, so there is no `championship`
 * here: the title odds come from `/api/leagues/{id}/odds` and cost three
 * seconds. The optional odds fields below are read if the API ever adds them
 * and are simply absent today.
 */
export interface LeagueSummary {
  league_id: number;
  season: number;
  /** The registry's name. `espn_name` is what ESPN calls it today. */
  name: string;
  espn_name?: string | null;
  enabled?: boolean;
  /** The probe's verdict. `null` when the row was not probed. */
  reachable?: boolean | null;
  error?: string | null;
  /** Present only if the API grows an explicit ok flag; `reachable` is today's. */
  ok?: boolean;
  /** From `registry.LeagueConfig` and the settings probe. */
  team_id?: number | null;
  team_name?: string | null;
  size?: number | null;
  scoring_variant?: string | null;
  current_week?: number | null;
  final_week?: number | null;
  playoff_team_count?: number | null;
  uses_faab?: boolean | null;
  tags?: string[];
  /** Not returned by the probe today. Read if present, never required. */
  championship?: number | null;
  championship_stderr?: number | null;
  playoffs?: number | null;
  rank?: number | null;
  /** Freshness of this row's probe. */
  computed_at?: string | null;
  stale_seconds?: number | null;
}

/** Whether this league can be read at all right now. The one failure test. */
export function leagueFailed(league: LeagueSummary): boolean {
  return league.ok === false || league.reachable === false;
}

/** What to call a league on screen: ESPN's current name, else the registry's. */
export function leagueLabel(league: LeagueSummary): string {
  return league.espn_name || league.name || String(league.league_id);
}

export interface LeaguesResponse {
  leagues: LeagueSummary[];
  season?: number | null;
  schema_version?: number | null;
  n_leagues?: number | null;
  n_reachable?: number | null;
}

/* ==========================================================================
   /api/leagues/:id/odds
   ========================================================================== */

export interface TeamOdds {
  rank: number;
  team_id: number;
  name: string;
  is_me: boolean;
  championship: number;
  playoffs: number;
  bye: number;
  expected_wins: number;
  wins: number | null;
  losses: number | null;
  ties: number | null;
  points_for: number | null;
  espn_projected_rank: number | null;
  /** sqrt(p(1-p)/n). The table prints decimals it does not have without this. */
  championship_stderr: number;
  /**
   * Whether this row is actually above the next one, on this draw. Null on the
   * last row. Most adjacent pairs on a live 14-team board are NOT separated:
   * render an unseparated boundary as the tie it is.
   */
  separated_from_next?: boolean | null;
}

export interface OddsPayload {
  league_id: number;
  season: number;
  name: string;
  size: number;
  week: number | null;
  team_id: number | null;
  team_name: string;
  n_sims: number;
  corpus_variant: string;
  corpus_variant_requested: string;
  baseline: string;
  teams: TeamOdds[];
  my_rank: number | null;
  my_championship: number | null;
  my_championship_stderr: number | null;
  my_playoffs: number | null;
  n_ranks_separated: number;
  n_ranks: number;
  ranking_note: string;
}

/* ==========================================================================
   /api/leagues/:id/leverage
   ========================================================================== */

export interface LeverageWeek {
  week: number;
  matchup_period: number;
  weeks_in_matchup: number;
  opponent_id: number;
  opponent: string;
  margin: number;
  sd_diff: number;
  win_probability: number;
  /** Marginal win probability per projected point, relative to a coin flip. */
  leverage: number;
  decided: boolean;
  /** What one projected point is worth, as a fraction of this game. */
  points_per_win_pct: number;
}

export interface LeveragePayload {
  league_id: number;
  name: string;
  team_id: number | null;
  mean_leverage: number;
  this_week: LeverageWeek | null;
  least_leveraged: LeverageWeek | null;
  weeks: LeverageWeek[];
}

/* ==========================================================================
   /api/leagues/:id/lineup
   ========================================================================== */

export interface LineupSlot {
  slot_id: number;
  slot: string;
  player_id: number;
  name: string;
}

export interface LineupChange {
  slot_id: number;
  slot: string;
  out_player_id: number;
  out: string;
  in_player_id: number;
  in: string;
  d_mean: number;
  d_sd: number;
}

export interface StackRow {
  name: string;
  opponent_name: string;
  rho: number;
  modelled: boolean;
}

export interface LineupPayload {
  league_id: number;
  name: string;
  team_id: number | null;
  team_name: string;
  week: number;
  threshold_kind: string;
  target: number;
  margin: number;
  sd_diff: number;
  z: number;
  leverage: number;
  /** False means the deltas are against a hypothetical, not against what is set. Say so. */
  current_known: boolean;
  unpriced_current: number[];
  opponent: { team_id: number; name: string; mean: number; sd: number } | null;
  recommended: LineupSlot[];
  baseline: LineupSlot[];
  changes: LineupChange[];
  n_changes: number;
  points_lineup_mean: number;
  win_prob_lineup_mean: number;
  delta_win_prob: number;
  delta_win_prob_stderr: number;
  noise_floor: number;
  points_sacrifice: number;
  guard: string | null;
  differ: boolean;
  /** Stricter than `Recommendation.significant`; an optimal lineup measures exactly zero. */
  significant: boolean;
  n_lineups: number;
  sd_independent: number;
  stacks: StackRow[];
  recommendation: Recommendation;
}

/* ==========================================================================
   /api/leagues/:id/waivers
   ========================================================================== */

/**
 * The analyst board a surface was priced against, when one was. Null means ESPN alone.
 *
 * `unverified` is not boilerplate: every other input in this system carries a measured
 * verdict and this one cannot yet, because no historical boards exist to score it
 * against. `weight` is the single constant that turns it off -- `0.0` is byte-identical
 * to no board at all -- so render it.
 */
export interface RankingsRef {
  kind: string;
  scoring: string;
  /** False when the half-PPR board is standing in for a full-PPR league (~1 rank drift). */
  matches_league_scoring: boolean;
  n: number;
  weight: number;
  file: string;
  unverified: string;
}

export interface WaiverRow extends Recommendation {
  add: string;
  position: string;
  drop: string;
  /**
   * "55-over-60": the analyst ranks the add 55th and the drop 60th at the position.
   * A rank comparison read straight off the board, NOT derived from `delta_title`,
   * which already carries the board's tilt. "-" when the board has no opinion.
   */
  board?: string;
  /** The analyst's one-line note on the add. "-" when absent. */
  note?: string;
  /** `"WR84 / WR55"`: our positional rank for the add, then the analyst's. */
  add_ranks?: string;
  /** The same pair for the player being dropped. */
  drop_ranks?: string;
  /**
   * `"+0.0/+16.3"`: what dropping this player costs as lineups are actually set,
   * against what he would have been worth to someone who knew which weeks to start
   * him. The second number is deliberately NOT charged -- a projection-set lineup
   * already captures ~0.89 of the ceiling and real managers capture 0.78, so nobody
   * measured beats the projection. Shown so the cut is the user's call. "-" when the
   * gap is under 10 points.
   */
  drop_ceiling?: string;
  bracket_title: number;
  bracket_stderr: number;
  agrees: boolean;
  /**
   * Whether adding him costs a waiver claim. False means ESPN has him as a plain
   * FREEAGENT: first come, no priority spent, no threshold to clear. Roughly 780-810
   * of the ~840 available players in each league are in that state.
   */
  on_waivers: boolean;
  /** "waiver priority" or "free". Derived from `on_waivers`; do not rank on it. */
  cost: string;
  /** `delta_title >= threshold`, printed as a fact by the CLI -- see `clears_certain`. */
  clears_threshold: boolean;
  clears_margin: number;
  /** False means the margin over the threshold is inside this row's own error: a coin flip. */
  clears_certain: boolean;
}

export interface StreamAdvantage {
  slot: number;
  position: string;
  stream_points: number;
  hold_points: number;
  hold_player: string;
  gap: number;
}

export interface WaiversPayload {
  league_id: number;
  name: string;
  team_id: number | null;
  team_name: string;
  week: number;
  /** All three of the user's leagues are rolling priority, not FAAB. */
  uses_faab: boolean;
  priority: number | null;
  priority_known: boolean;
  budget: number | null;
  /** What a claim has to clear to be worth spending a queue position. The point of the surface. */
  threshold: number;
  baseline_title: number;
  baseline: string;
  title_per_point: number;
  title_per_point_stderr: number;
  week_leverage: number;
  sd_diff: number;
  n_free_agents: number;
  /** How many unrostered players actually cost a claim. Null when ESPN was not asked. */
  n_on_waivers: number | null;
  /** How many are simply free. Null when ESPN was not asked. */
  n_free_agents_available: number | null;
  rankings: RankingsRef | null;
  /**
   * Per single-body slot, over the remaining season: what the seat yields taking the
   * best available every week, against holding the best rosterable body.
   *
   * `gap` is non-negative by CONSTRUCTION -- streaming sums a per-week max and holding
   * maximises a per-week sum -- so never render the sign as evidence. Render the
   * comparison ACROSS positions: D/ST runs ~38 points against ~13 at K and TE, and
   * that ratio is the measured reason a defense is the seat you stream. Neither column
   * pays for the weekly transaction.
   */
  stream_advantage: StreamAdvantage[];
  board: WaiverRow[];
  /** Costs waiver priority, and clears the continuation value of holding it. */
  claims: WaiverRow[];
  /**
   * Positive adds that cost NOTHING. Kept apart from `claims` because everything
   * downstream prices a claim at "waiver priority", and merging the two would label a
   * free add as spending the scarcest thing on the board.
   *
   * These are ALTERNATIVES, not a shopping list: each was priced on its own against
   * today's roster and most share a drop, so the gains are not additive.
   */
  free_adds: WaiverRow[];
  blocks: WaiverRow[];
  hold: Recommendation;
  best: Recommendation;
  any_claim: boolean;
  any_action: boolean;
  waterfall_note: string;
}

/* ==========================================================================
   /api/leagues/:id/trades
   ========================================================================== */

export interface TradeRow extends Recommendation {
  partners: TeamRef[];
  caveats: string[];
  /**
   * The arbitrage in one number. For every counterparty, how much better the deal
   * looks by the projections on THEIR screen than by the analyst board -- summed, and
   * NOT including our side. Positive is the case worth having. Null with no board.
   */
  spread: number | null;
  /** `spread > 0`: the other side reads it as better for them than we think it is. */
  mispriced: boolean;
  /** The analyst's note per player in the deal, keyed by name. Empty with no board. */
  notes: Record<string, string>;
  /**
   * How many OTHER routes deliver this exact return for this exact price, differing
   * only in who stands in the middle. Set on the row that leads the family, which is
   * the easiest one to get signed (fewest teams, then widest spread).
   */
  routes: number;
  /** This row is one of those alternatives -- the same return, routed differently. */
  same_return: boolean;
}

export interface TradesPayload {
  league_id: number;
  name: string;
  team_id: number | null;
  n_found: number;
  min_gain: number;
  rankings: RankingsRef | null;
  /** Largest cycle searched. 2 means every row needs only one other manager. */
  max_teams: number;
  /**
   * How many of the shown trades need only ONE other manager. Three-way chains dominate
   * because they need no bilateral coincidence of wants -- measured at week 1 of 2026,
   * the screen found 13/0/4 two-team candidates against 27/40/36 three-team ones -- not
   * because they are better. Request `max_teams=2` for bilateral only.
   */
  n_two_team: number;
  /**
   * "analyst board" or "espn projections". With a board, every `delta_title` on the
   * page and the paired baseline behind it are priced on the re-dealt projections, so
   * the deltas are consistent with each other and NOT with the odds page's level.
   */
  priced_on: string;
  significance_test: string;
  /**
   * The top row is the maximum of `n_found` noisy paired estimates and is biased
   * upward; the +/- is this draw's Monte Carlo error, NOT the uncertainty on the
   * trade being worth what it says. Render this note wherever the number renders.
   */
  selection_note: string;
  trades: TradeRow[];
}

/* ==========================================================================
   /api/leagues/:id/stream
   ========================================================================== */

export interface StreamPlanRow {
  week: number;
  [key: string]: unknown;
}

export interface PlanSummary {
  weeks?: number;
  weeks_shown?: number;
  total_gain_points?: number;
  priced_gain_points?: number;
  /** Points from weeks no bookmaker has priced yet. Those numbers will move. */
  unpriced_gain_points?: number;
  unpriced_share?: number;
  /** Week one's gain over holding. Routinely 0.00 while the plan's headline is +2.77pp. */
  week_one_gain_points?: number;
  n_acquisitions?: number;
}

export interface StreamRecommendation extends Recommendation {
  position: string;
  caveats: string[];
  plan: StreamPlanRow[];
  plan_source: 'rederived' | 'unavailable' | string;
  plan_summary: PlanSummary;
  /** "hold" when this week's move is nothing, even though the plan is worth something. */
  action: string;
}

export interface StreamPayload {
  league_id: number;
  name: string;
  team_id: number | null;
  stream: StreamRecommendation;
}

/* ==========================================================================
   /api/leagues/:id/weekly
   ========================================================================== */

/** `report.Action.to_dict` -- one thing to do, priced in the unit that crosses leagues. */
export interface ActionRow {
  league_id: number;
  league_name: string;
  season: number;
  team_id: number;
  surface: string;
  headline: string;
  delta_title: number;
  stderr: number;
  z: number | null;
  leverage: number;
  significant: boolean;
  verdict: Verdict;
  confidence: Confidence;
  /** Not ranked on -- a cheaper action is not a better one -- but needed to execute. */
  cost: string;
  deadline: string;
  tags: string[];
  rationale: string;
  /** False when there is nothing to execute today (a plan whose first week is a hold). */
  actionable: boolean;
  /** Why it is not simply "do it", in the producing surface's own words. */
  blockers: string[];
  /** What is wrong with the NUMBER, as opposed to with executing it. */
  caveats: string[];
}

export interface WeeklyPayload {
  league_id: number;
  season: number;
  name: string;
  ok: boolean;
  error?: string;
  team_id?: number;
  week?: number | null;
  n_sims?: number;
  corpus_variant?: string;
  corpus_variant_requested?: string;
  odds?: Section<OddsPayload>;
  leverage?: Section<LeveragePayload>;
  lineup?: Section<LineupPayload>;
  waivers?: Section<WaiversPayload>;
  trades?: Section<TradesPayload>;
  stream?: Section<StreamPayload>;
  /** The one line to read if you read nothing else. Composed from measured numbers. */
  headline?: string;
  actions?: ActionRow[];
}

/* ==========================================================================
   /api/queue
   ========================================================================== */

export interface QueueError {
  league_id?: number;
  league?: string;
  surface?: string;
  error: string;
}

export interface QueuePayload {
  schema_version: number;
  command: string;
  generated_at: string;
  leagues: Array<{ ok: boolean; league_id: number; season: number; name: string; error?: string }>;
  actions: ActionRow[];
  /** A league with nothing to do says so, in its own numbers. Never dropped. */
  holds: string[];
  errors: QueueError[];
  /** "edges.portfolio.action_queue" (one shared NFL season) or "report.rank_actions". */
  ranked_by: string;
  n_actions: number;
  /** Index of the first row that is both significant and executable today, or null. */
  first_actionable: number | null;
}

/* ==========================================================================
   /api/portfolio
   ==========================================================================
   `edges/portfolio.py`. Every league drawn against ONE shared NFL season, which
   is what makes cross-league statements measurements rather than assumptions.
   ========================================================================== */

export interface PortfolioOdds {
  names: string[];
  titles: number[];
  p_at_least_one: number;
  p_zero: number;
  p_two_plus: number;
  /** The only aggregate that is additive across leagues, exactly, for any dependence. */
  expected_titles: number;
  variance_titles: number;
  independent: number;
  sum_bound: number;
  max_bound: number;
  stderr: number;
  dependence_cost: number;
  dependence_cost_stderr: number;
  dependence_significant: boolean;
  n_sims: number;
}

export interface PairCorrelation {
  a: string;
  b: string;
  shared_players: string[];
  weekly: number;
  season: number;
  /** A phi between two rare indicators. It changes sign across seeds -- show the error. */
  champion: number;
  weekly_stderr: number;
  season_stderr: number;
  champion_stderr: number;
  champion_significant: boolean;
}

export interface Holding {
  league_id: number;
  league_name: string;
  team_id: number;
  slot_id: number;
  slot: string;
  start_share: number;
  title_added: number;
  title_added_stderr: number;
  starts?: boolean;
}

export interface Exposure {
  player_id: number;
  name: string;
  position_id: number;
  position: string;
  pro_team_id: number;
  holdings: Holding[];
  /** Sum over leagues of the title probability he holds up, in E[titles]. */
  equity_at_risk: number;
  equity_share: number;
  /** P(>=1 title) today minus P(>=1 title) with him gone everywhere, re-simulated. */
  portfolio_damage: number;
  portfolio_damage_stderr: number;
  significant: boolean;
  n_leagues?: number;
  n_starting?: number;
}

export interface Concentration {
  kind: string;
  label: string;
  removed: Record<string, string[]>;
  before: number;
  after: number;
  stderr: number;
  expected_before: number;
  expected_after: number;
  driver: string;
  driver_damage: number;
  /** What the GROUPING adds over its single biggest holding. The number to read. */
  increment: number;
  increment_stderr: number;
  damage?: number;
}

export interface ByeExposure {
  week: number;
  [key: string]: unknown;
}

export interface PortfolioPayload {
  season?: number;
  odds?: PortfolioOdds | null;
  correlations?: { pairs: PairCorrelation[]; by_week?: Record<string, number> } | null;
  exposures?: Exposure[];
  concentration?: Concentration[];
  byes?: ByeExposure[];
  diversification?: Record<string, unknown> | null;
  queue?: QueuePayload | null;
  errors?: QueueError[];
}

/* ==========================================================================
   /api/refresh, /api/health
   ========================================================================== */

/**
 * What the refresh button is scoped to. `league` sends a `league_id`; `all`
 * sends nothing and drops the registry too. The API has exactly these two
 * behaviours -- there is no server-side notion of refreshing only the queue.
 */
export type RefreshScope = 'all' | 'league';

export interface RefreshRequest {
  scope?: RefreshScope;
  /** Present means "this league only". Absent means everything. */
  league_id?: number;
}

/** `POST /api/refresh`. It only invalidates; nothing is recomputed until asked. */
export interface RefreshResponse {
  ok: boolean;
  endpoint?: string;
  invalidated?: {
    entries: number;
    workspaces: number;
    league_id: number | null;
  };
  at?: string | null;
}

export interface HealthResponse {
  ok: boolean;
  schema_version?: number;
  season?: number;
  n_leagues?: number;
  version?: string;
  /** Never a credential. If the API ever puts one here, that is a bug in the API. */
  [key: string]: unknown;
}

/* ==========================================================================
   Transport
   ========================================================================== */

type QueryValue = string | number | boolean | undefined | null | string[];

interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  query?: Record<string, QueryValue>;
  method?: 'GET' | 'POST';
  body?: unknown;
}

function buildQuery(query: RequestOptions['query']): string {
  if (!query) return '';
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === '') continue;
    // FastAPI reads a repeated key as a list (`?skip=trades&skip=stream`);
    // a comma-joined value would be one section literally named "a,b".
    if (Array.isArray(value)) {
      for (const item of value) if (item !== '') params.append(key, String(item));
      continue;
    }
    params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded ? `?${encoded}` : '';
}

/**
 * Take the payload out of the API's envelope.
 *
 * `api/server._envelope` answers every computed surface as
 * `{ok, endpoint, league_id, season, sims, schema_version, cached, computed_at,
 *   stale_seconds, compute_seconds, data, error}` -- the freshness facts on the
 * outside and the `report.py` payload under `data`. `/api/health` and
 * `/api/leagues` are flat. This flattens the first shape and passes the second
 * through, so a view reads `payload.teams` without knowing which it got, and
 * `meta` is assembled from both levels (the envelope has `sims` and the age;
 * the payload inside has `n_sims`, `seed` and `generated_at`).
 */
function unwrap<T>(parsed: unknown): WithMeta<T> {
  const top = parsed as Record<string, unknown>;
  const outer = readMeta(parsed);
  const inner = top?.['data'];
  if (
    top &&
    typeof top.endpoint === 'string' &&
    'data' in top &&
    inner !== null &&
    typeof inner === 'object' &&
    !Array.isArray(inner)
  ) {
    return { ...(inner as T), meta: mergeMeta(outer, readMeta(inner)) };
  }
  return { ...(parsed as T), meta: outer };
}

/** Pull the most useful message out of whatever the API put in an error body. */
function errorFromBody(status: number, body: unknown, fallback: string): ApiError {
  if (body && typeof body === 'object') {
    const b = body as Record<string, unknown>;
    // FastAPI's own shape is `{detail: "..."}`; `_error_body` nests under it.
    const nested =
      b.detail && typeof b.detail === 'object' ? (b.detail as Record<string, unknown>) : b;
    const message =
      pickString(nested, 'message') ??
      pickString(nested, 'error') ??
      (typeof b.detail === 'string' ? b.detail : null) ??
      fallback;
    return new ApiError({
      message,
      status,
      code: pickString(nested, 'type') ?? pickString(nested, 'code') ?? `http_${status}`,
      detail: pickString(nested, 'detail') ?? pickString(nested, 'traceback') ?? '',
      leagueId: pickNumber(nested, 'league_id'),
    });
  }
  return new ApiError({ message: fallback, status, code: `http_${status}` });
}

/**
 * A 200 whose envelope says `ok: false`.
 *
 * `api/server._envelope` puts the failure in `error`, and it is an OBJECT --
 * `{type, message, endpoint, league_id}` off `_error_body` -- not a string. A
 * client that only recognises the string form hands the view an empty payload
 * and no error at all, which is the one failure mode a dashboard must not have:
 * a page that looks computed and is blank. Both shapes land here.
 */
function errorFromEnvelope(status: number, top: Record<string, unknown>, path: string): ApiError {
  const raw = top.error;
  if (raw && typeof raw === 'object') {
    const e = raw as Record<string, unknown>;
    return new ApiError({
      message: pickString(e, 'message') ?? `${path} failed`,
      status,
      code: pickString(e, 'type') ?? 'league_failed',
      detail: pickString(e, 'endpoint') ?? pickString(top, 'endpoint') ?? '',
      leagueId: pickNumber(e, 'league_id') ?? pickNumber(top, 'league_id'),
    });
  }
  return new ApiError({
    message: pickString(top, 'error') ?? pickString(top, 'message') ?? `${path} failed`,
    status,
    code: 'league_failed',
    leagueId: pickNumber(top, 'league_id'),
  });
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<WithMeta<T>> {
  const { signal, timeoutMs = DEFAULT_TIMEOUT_MS, query, method = 'GET', body } = options;

  // One controller for our own timeout, chained to the caller's signal so a
  // component that unmounts mid-simulation actually cancels the request.
  const controller = new AbortController();
  const timer = window.setTimeout(
    () => controller.abort(new DOMException('timeout', 'TimeoutError')),
    timeoutMs,
  );
  const onAbort = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener('abort', onAbort, { once: true });
  }

  try {
    const response = await fetch(`${API_BASE}${path}${buildQuery(query)}`, {
      method,
      signal: controller.signal,
      headers: body
        ? { Accept: 'application/json', 'Content-Type': 'application/json' }
        : { Accept: 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
      credentials: 'same-origin',
    });

    const text = await response.text();
    let parsed: unknown = null;
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = null;
      }
    }

    if (!response.ok) {
      throw errorFromBody(
        response.status,
        parsed,
        `${response.status} ${response.statusText || 'request failed'} for ${path}`,
      );
    }
    if (parsed === null) {
      throw new ApiError({
        message: `${path} returned a body that is not JSON`,
        status: response.status,
        code: 'bad_payload',
        detail: text.slice(0, 400),
      });
    }

    // A 200 that says `ok: false` is a real failure with a useful message --
    // the API returns one (with HTTP 200, deliberately) for a league ESPN would
    // not serve, or a surface that raised. Section-level `ok: false` is NOT an
    // error: those are handled per section so one dead surface does not cost
    // the reader the rest of the page.
    const top = parsed as Record<string, unknown>;
    if (top.ok === false) throw errorFromEnvelope(response.status, top, path);

    return unwrap<T>(parsed);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof DOMException && err.name === 'TimeoutError') {
      throw new ApiError({
        message: `timed out after ${Math.round(timeoutMs / 1000)}s waiting for ${path}`,
        code: 'timeout',
        retryable: true,
      });
    }
    if (isAbort(err)) throw err;
    throw new ApiError({
      message: err instanceof Error ? err.message : `could not reach ${path}`,
      code: 'network',
      retryable: true,
    });
  } finally {
    window.clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

/**
 * Per-league path.
 *
 * `/api/leagues/{id}/{leaf}`, verified against the running server. There used
 * to be a shim here that retried the singular `/api/league/...` on any 404;
 * it was worse than nothing, because the API's 404 for a league that is not in
 * the registry ("league 999 is not in the registry; it holds ...") is the most
 * useful message on the endpoint, and the retry replaced it with the router's
 * "no such endpoint".
 */
function leagueRequest<T>(
  leagueId: number,
  leaf: string,
  options: RequestOptions = {},
): Promise<WithMeta<T>> {
  return request<T>(`/leagues/${leagueId}/${leaf}`, options);
}

/* ==========================================================================
   The client
   ========================================================================== */

export interface ListOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export const api = {
  /** Liveness plus schema version. Cheap; never touches ESPN. */
  health(options: ListOptions = {}): Promise<WithMeta<HealthResponse>> {
    return request<HealthResponse>('/health', { ...options, timeoutMs: options.timeoutMs ?? 10_000 });
  },

  /** The league switcher's source of truth, from `config/leagues.toml`. */
  async leagues(options: ListOptions = {}): Promise<WithMeta<LeaguesResponse>> {
    const raw = await request<LeaguesResponse | LeagueSummary[]>('/leagues', options);
    if (Array.isArray(raw)) {
      // Tolerate a bare array: the list is the payload either way.
      return { leagues: raw as LeagueSummary[], meta: readMeta(raw) };
    }
    const body = raw as WithMeta<LeaguesResponse>;
    return { ...body, leagues: body.leagues ?? [] };
  },

  /**
   * The ranked cross-league list. The home view.
   *
   * The server's flag is `actionable`, not `actionable_only` -- FastAPI ignores
   * a query parameter it does not declare, so the misspelling silently returned
   * the whole queue including the rows that did not clear their error.
   */
  queue(
    params: { limit?: number; actionable?: boolean } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<QueuePayload>> {
    return request<QueuePayload>('/queue', {
      timeoutMs: CROSS_LEAGUE_TIMEOUT_MS,
      ...options,
      query: { ...params },
    });
  },

  /**
   * Cross-league exposure, correlated risk and concentration damage.
   *
   * Takes the signal in either position -- `portfolio({signal})` and
   * `portfolio({limit}, {signal})` both work -- because a view that passes an
   * AbortSignal into the wrong slot would otherwise send it as `?signal=` and
   * never cancel.
   */
  portfolio(
    params: ListOptions & {
      min_leagues?: number;
      top_exposures?: number;
      limit?: number;
    } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<PortfolioPayload>> {
    const { signal, timeoutMs, ...query } = params;
    return request<PortfolioPayload>('/portfolio', {
      signal: options.signal ?? signal,
      timeoutMs: options.timeoutMs ?? timeoutMs ?? CROSS_LEAGUE_TIMEOUT_MS,
      query,
    });
  },

  /**
   * One league's whole picture. Every section fails independently.
   *
   * `skip` names the sections NOT to compute (the server's own parameter), and
   * it is repeated, not comma-joined. This is also where leverage comes from:
   * there is no `/api/leagues/{id}/leverage` route, so read
   * `sectionData(weekly.leverage)` rather than asking for one.
   */
  weekly(
    leagueId: number,
    params: { limit?: number; skip?: string[] } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<WeeklyPayload>> {
    return leagueRequest<WeeklyPayload>(leagueId, 'weekly', { ...options, query: { ...params } });
  },

  odds(leagueId: number, options: ListOptions = {}): Promise<WithMeta<OddsPayload>> {
    return leagueRequest<OddsPayload>(leagueId, 'odds', options);
  },

  /**
   * This week's leverage.
   *
   * THERE IS NO `/api/leagues/{id}/leverage` ROUTE. Leverage is a section of the
   * weekly report, so this asks for the weekly report with every other section
   * skipped -- `?skip=odds&skip=lineup&...`, which the server honours and which
   * drops a cold call from ~19s to ~4s. The numbers are `report.weekly_payload`'s
   * own; nothing is derived here. A section that failed becomes an ApiError
   * rather than an empty object, so the caller renders the server's message.
   */
  async leverage(leagueId: number, options: ListOptions = {}): Promise<WithMeta<LeveragePayload>> {
    const others = WEEKLY_SECTION_NAMES.filter((s) => s !== 'leverage');
    const weekly = await leagueRequest<WeeklyPayload>(leagueId, 'weekly', {
      ...options,
      query: { skip: others },
    });
    const failure = sectionError(weekly.leverage);
    if (failure) {
      throw new ApiError({
        message: failure,
        code: 'section_failed',
        detail: 'weekly.leverage',
        leagueId,
      });
    }
    const data = sectionData(weekly.leverage);
    if (!data) {
      throw new ApiError({
        message: `league ${leagueId} returned no leverage section`,
        code: 'section_missing',
        detail: 'weekly.leverage',
        leagueId,
      });
    }
    return { ...data, meta: weekly.meta };
  },

  /** The user's roster with each player's outlook and value beside him. */
  roster(
    leagueId: number,
    params: { week?: number } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<Record<string, unknown>>> {
    return leagueRequest<Record<string, unknown>>(leagueId, 'roster', {
      ...options,
      query: { ...params },
    });
  },

  lineup(
    leagueId: number,
    params: { week?: number } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<LineupPayload>> {
    return leagueRequest<LineupPayload>(leagueId, 'lineup', { ...options, query: { ...params } });
  },

  waivers(
    leagueId: number,
    params: { limit?: number; week?: number } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<WaiversPayload>> {
    return leagueRequest<WaiversPayload>(leagueId, 'waivers', { ...options, query: { ...params } });
  },

  trades(
    leagueId: number,
    params: { limit?: number; min_gain?: number } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<TradesPayload>> {
    return leagueRequest<TradesPayload>(leagueId, 'trades', { ...options, query: { ...params } });
  },

  /** `position` is an ESPN position id (16 is D/ST), not a label. */
  stream(
    leagueId: number,
    params: { position?: number; plan_weeks?: number } = {},
    options: ListOptions = {},
  ): Promise<WithMeta<StreamPayload>> {
    return leagueRequest<StreamPayload>(leagueId, 'stream', { ...options, query: { ...params } });
  },

  /**
   * Drop the server-side cache. Explicit, because a cold run is seconds per
   * league and the cross-league queue is over a minute: nothing in this app
   * silently recomputes behind the user.
   *
   * `POST /api/refresh?league_id=...` -- a QUERY parameter. The server declares
   * no request body, so sending `{league_id}` as JSON was read as "no league
   * given", i.e. invalidate everything: the per-league refresh button threw
   * away all three leagues and the 78-second queue every time it was pressed.
   * Scoping is by presence of `league_id`; there is no `scope` or `force` on
   * the wire, and asking for a refresh at all is the force.
   */
  refresh(
    body: RefreshRequest = {},
    options: ListOptions = {},
  ): Promise<WithMeta<RefreshResponse>> {
    return request<RefreshResponse>('/refresh', {
      ...options,
      method: 'POST',
      query: body.league_id === undefined ? {} : { league_id: body.league_id },
    });
  },
};

/* ==========================================================================
   Formatters
   ==========================================================================
   The wire is fractions; the display is percentage points. That conversion
   lives here and nowhere else, so a view cannot accidentally print 0.0277 as
   "0.03%". These mirror `report.pp` / `report.pct` / `report.signed` exactly,
   so a number in the browser reads the same as the same number in the CLI.
   ========================================================================== */

const DASH = '—'; // em dash: "no value", distinct from a minus sign.

function isNum(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * `toFixed`, except it rounds the way Python's `format` does.
 *
 * THIS IS NOT PEDANTRY, IT IS THE CLI-AGREEMENT RULE. Python rounds an exact
 * tie to even; JavaScript's `toFixed` rounds it away from zero. The same double
 * therefore prints as 29.2% in `fq odds` and 29.3% in the browser -- and with
 * `n_sims = 4000` every probability is k/4000, so `p * 100` lands on an exact
 * half at one decimal place about a quarter of the time. A dashboard whose last
 * digit disagrees with the CLI for no reason a reader can see is worse than one
 * that shows a number less precisely.
 *
 * `Intl.NumberFormat`'s `halfEven` is not sufficient: it rounds the shortest
 * decimal representation, so it turns 0.05 (whose exact double is a hair ABOVE
 * a half) into "0.0" where Python gives "0.1". So this reconstructs the double's
 * exact decimal value from its bits and rounds that, in BigInt, which is what
 * Python does.
 */
export function toFixedHalfEven(value: number, digits: number): string {
  if (!Number.isFinite(value)) return String(value);
  const d = Math.max(0, Math.min(20, Math.trunc(digits)));

  const bits = new DataView(new ArrayBuffer(8));
  bits.setFloat64(0, value);
  const hi = BigInt(bits.getUint32(0));
  const lo = BigInt(bits.getUint32(4));
  const raw = (hi << 32n) | lo;
  const negative = (raw >> 63n) & 1n ? '-' : '';
  const exponent = Number((raw >> 52n) & 0x7ffn);
  const fraction = raw & 0xf_ffff_ffff_ffffn;

  // value = mantissa * 2 ** exp2, both exact.
  const mantissa = exponent === 0 ? fraction : fraction | (1n << 52n);
  const exp2 = (exponent === 0 ? -1074 : exponent - 1075) | 0;

  // Scale to an integer count of 10**-d units, keeping the division exact.
  let scaled: bigint;
  if (exp2 >= 0) {
    scaled = mantissa * (1n << BigInt(exp2)) * 10n ** BigInt(d);
  } else {
    // m / 2**k == (m * 5**k) / 10**k, exactly.
    const k = -exp2;
    const numerator = mantissa * 5n ** BigInt(k);
    if (d >= k) {
      scaled = numerator * 10n ** BigInt(d - k);
    } else {
      const divisor = 10n ** BigInt(k - d);
      const quotient = numerator / divisor;
      const remainder = numerator % divisor;
      const twice = remainder * 2n;
      scaled =
        twice > divisor || (twice === divisor && quotient % 2n === 1n) ? quotient + 1n : quotient;
    }
  }

  const text = scaled.toString().padStart(d + 1, '0');
  const whole = text.slice(0, text.length - d) || '0';
  return d === 0 ? `${negative}${whole}` : `${negative}${whole}.${text.slice(text.length - d)}`;
}

export const fmt = {
  /** Nothing to show. One glyph everywhere so an empty cell is unambiguous. */
  dash: DASH,

  /** A probability as a percentage: 0.051 -> "5.1%". */
  pct(value: number | null | undefined, digits = 1): string {
    return isNum(value) ? `${toFixedHalfEven(value * 100, digits)}%` : DASH;
  },

  /** A probability DIFFERENCE in signed percentage points: 0.0277 -> "+2.77pp". */
  pp(value: number | null | undefined, digits = 2): string {
    if (!isNum(value)) return DASH;
    const shown = value * 100;
    // Avoid "-0.00pp" for a negative value that rounds to zero.
    const safe = Math.abs(shown) < 0.5 / 10 ** digits ? 0 : shown;
    return `${safe >= 0 ? '+' : ''}${toFixedHalfEven(safe, digits)}pp`;
  },

  /** An unsigned error bar in percentage points: 0.0035 -> "0.35pp". */
  ppError(value: number | null | undefined, digits = 2): string {
    return isNum(value) ? `${toFixedHalfEven(Math.abs(value * 100), digits)}pp` : DASH;
  },

  /** A plain number. */
  num(value: number | null | undefined, digits = 2): string {
    return isNum(value) ? toFixedHalfEven(value, digits) : DASH;
  },

  /** A signed plain number, for points and other non-probability deltas. */
  signed(value: number | null | undefined, digits = 2): string {
    if (!isNum(value)) return DASH;
    const safe = Math.abs(value) < 0.5 / 10 ** digits ? 0 : value;
    return `${safe >= 0 ? '+' : ''}${toFixedHalfEven(safe, digits)}`;
  },

  /** An integer with thousands separators, for simulation counts. */
  int(value: number | null | undefined): string {
    return isNum(value) ? Math.round(value).toLocaleString('en-US') : DASH;
  },

  /** Sigma, for a row that wants to show how resolved it is. Server-supplied. */
  z(value: number | null | undefined, digits = 1): string {
    return isNum(value) ? `${toFixedHalfEven(value, digits)}σ` : DASH;
  },

  /** "4m ago". Coarse on purpose: a cache age is not a stopwatch. */
  ago(seconds: number | null | undefined): string {
    if (!isNum(seconds)) return 'never computed';
    if (seconds < 10) return 'just now';
    if (seconds < 90) return `${Math.round(seconds)}s ago`;
    if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`;
    if (seconds < 172800) return `${Math.round(seconds / 3600)}h ago`;
    return `${Math.round(seconds / 86400)}d ago`;
  },

  /** Local wall-clock time of an ISO instant, for the "computed at" tooltip. */
  clock(iso: string | null | undefined): string {
    if (!iso) return DASH;
    const then = Date.parse(iso);
    if (Number.isNaN(then)) return iso;
    return new Date(then).toLocaleString();
  },

  /** A win-loss record. */
  record(wins: number | null | undefined, losses: number | null | undefined, ties?: number | null): string {
    if (!isNum(wins) || !isNum(losses)) return DASH;
    return isNum(ties) && ties > 0 ? `${wins}-${losses}-${ties}` : `${wins}-${losses}`;
  },
};

/**
 * Map the server's `significant` flag onto a verdict when a payload carries the
 * boolean but not the four-state kind (some surfaces predate it).
 *
 * This is a rename, not a test. The client never decides significance: the
 * right test differs per surface and lives in `report.verdict_for`.
 */
export function verdictOf(
  row: { verdict?: Verdict | null; significant?: boolean | null; delta_title?: number | null },
): Verdict {
  if (row.verdict === 'act' || row.verdict === 'noise' || row.verdict === 'null' || row.verdict === 'harm') {
    return row.verdict;
  }
  if (row.significant === true) return 'act';
  if (isNum(row.delta_title) && row.delta_title === 0) return 'null';
  return 'noise';
}
