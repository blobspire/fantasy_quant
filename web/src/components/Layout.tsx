import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { NavLink, useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  ageSeconds,
  api,
  ApiError,
  fmt,
  isAbort,
  leagueFailed,
  leagueLabel,
  type LeagueSummary,
  type Meta,
  type RefreshScope,
  type WithMeta,
} from '../api';

/* ==========================================================================
   Routes
   ==========================================================================
   One table, used twice: `Layout` renders it as the nav, `App` mounts it as
   the router. A view module lives at `src/views/<view>.tsx` and default-exports
   a component. Adding a view is adding a row here and a file there; nothing
   else in the shell needs to change.
   ========================================================================== */

export interface ViewRoute {
  key: string;
  label: string;
  /** Module basename under `src/views/`. `Queue` -> `src/views/Queue.tsx`. */
  view: string;
  /** React Router pattern. `:leagueId` is filled from the active league. */
  pattern: string;
  /** True when the view is about one league and the nav link needs an id. */
  needsLeague: boolean;
  /** Hover text on the nav link. Say what the view answers. */
  help?: string;
}

export const ROUTES: ViewRoute[] = [
  {
    key: 'queue',
    label: 'Queue',
    view: 'Queue',
    pattern: '/queue',
    needsLeague: false,
    help: 'Every league in one order, ranked on the increment to E[titles]',
  },
  {
    key: 'league',
    label: 'League',
    view: 'League',
    pattern: '/league/:leagueId',
    needsLeague: true,
    help: 'Championship odds, this week’s leverage, and the headline',
  },
  {
    key: 'lineup',
    label: 'Lineup',
    view: 'Lineup',
    pattern: '/league/:leagueId/lineup',
    needsLeague: true,
    help: 'Start/sit against the lineup that is actually set',
  },
  {
    key: 'waivers',
    label: 'Waivers',
    view: 'Waivers',
    pattern: '/league/:leagueId/waivers',
    needsLeague: true,
    help: 'The board, and the priority threshold a claim has to clear',
  },
  {
    key: 'trades',
    label: 'Trades',
    view: 'Trades',
    pattern: '/league/:leagueId/trades',
    needsLeague: true,
    help: 'Confirmed Pareto trades, with the selection bias stated',
  },
  {
    key: 'stream',
    label: 'Stream',
    view: 'Stream',
    pattern: '/league/:leagueId/stream',
    needsLeague: true,
    help: 'This week’s D/ST action and the plan that justifies it',
  },
  {
    key: 'portfolio',
    label: 'Portfolio',
    view: 'Portfolio',
    pattern: '/portfolio',
    needsLeague: false,
    help: 'Cross-league exposure, correlated risk, concentration damage',
  },
];

/** The concrete href for a route, given the league in the switcher. */
export function hrefFor(route: ViewRoute, leagueId: number | null): string {
  if (!route.needsLeague) return route.pattern;
  if (leagueId === null) return ROUTES[0]!.pattern;
  return route.pattern.replace(':leagueId', String(leagueId));
}

/* ==========================================================================
   App state
   ========================================================================== */

export interface AppState {
  leagues: LeagueSummary[];
  leaguesLoading: boolean;
  leaguesError: ApiError | null;
  /** The league in the switcher, or null before `/api/leagues` answers. */
  leagueId: number | null;
  league: LeagueSummary | null;
  setLeagueId: (id: number) => void;
  /** Freshness of whatever is currently on screen. Views report it via `useResource`. */
  meta: Meta | null;
  reportMeta: (meta: Meta | null) => void;
  refreshing: boolean;
  /** Drop the server cache and recompute. Explicit: nothing recomputes silently. */
  refresh: (scope?: RefreshScope) => Promise<void>;
  /** Bumped by a refresh. `useResource` watches it, so every view refetches. */
  refreshToken: number;
  /** Re-read `/api/leagues` without a recompute. */
  reloadLeagues: () => void;
}

const AppStateContext = createContext<AppState | null>(null);

export function useAppState(): AppState {
  const state = useContext(AppStateContext);
  if (!state) throw new Error('useAppState must be used inside <AppStateProvider>');
  return state;
}

/** The active league id from the URL, falling back to the switcher's. */
export function useLeagueId(): number | null {
  const params = useParams();
  const { leagueId } = useAppState();
  const fromUrl = params.leagueId ? Number(params.leagueId) : NaN;
  return Number.isFinite(fromUrl) ? fromUrl : leagueId;
}

const LAST_LEAGUE_KEY = 'fq.league';

function readStoredLeague(): number | null {
  try {
    const raw = window.localStorage.getItem(LAST_LEAGUE_KEY);
    const parsed = raw ? Number(raw) : NaN;
    return Number.isFinite(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [leagues, setLeagues] = useState<LeagueSummary[]>([]);
  const [leaguesLoading, setLeaguesLoading] = useState(true);
  const [leaguesError, setLeaguesError] = useState<ApiError | null>(null);
  const [leagueId, setLeagueIdState] = useState<number | null>(readStoredLeague);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);
  const [leaguesToken, setLeaguesToken] = useState(0);

  // Read the league out of the path rather than out of `useParams`: this
  // provider sits ABOVE the routes, where the route params are not bound yet.
  const { pathname } = useLocation();
  const urlLeagueId = Number(/^\/league\/(\d+)/.exec(pathname)?.[1] ?? NaN);

  useEffect(() => {
    const controller = new AbortController();
    setLeaguesLoading(true);
    api
      .leagues({ signal: controller.signal })
      .then((response) => {
        setLeagues(response.leagues);
        setLeaguesError(null);
        setLeaguesLoading(false);
      })
      .catch((err: unknown) => {
        if (isAbort(err)) return;
        setLeaguesError(
          err instanceof ApiError ? err : new ApiError({ message: String(err), code: 'unknown' }),
        );
        setLeaguesLoading(false);
      });
    return () => controller.abort();
  }, [leaguesToken, refreshToken]);

  // Pick a league as soon as we know what there is: the URL wins, then the last
  // one used, then the first that actually loaded.
  useEffect(() => {
    if (!leagues.length) return;
    setLeagueIdState((current) => {
      const known = (id: number | null) => leagues.some((l) => l.league_id === id);
      if (Number.isFinite(urlLeagueId) && known(urlLeagueId)) return urlLeagueId;
      if (known(current)) return current;
      const healthy = leagues.find((l) => !leagueFailed(l));
      return (healthy ?? leagues[0])!.league_id;
    });
  }, [leagues, urlLeagueId]);

  const setLeagueId = useCallback((id: number) => {
    setLeagueIdState(id);
    try {
      window.localStorage.setItem(LAST_LEAGUE_KEY, String(id));
    } catch {
      // A browser with storage disabled still gets a working switcher.
    }
  }, []);

  const refresh = useCallback(
    async (scope: RefreshScope = 'all') => {
      setRefreshing(true);
      try {
        // Only a `league_id` scopes the invalidation. Sending one when the
        // reader asked for "this league" is the difference between dropping one
        // three-second simulation and dropping every league plus the
        // minute-and-a-quarter cross-league queue.
        await api.refresh({
          scope,
          ...(scope === 'league' && leagueId !== null ? { league_id: leagueId } : {}),
        });
      } catch (err) {
        if (!isAbort(err)) {
          setLeaguesError(
            err instanceof ApiError
              ? err
              : new ApiError({ message: String(err), code: 'refresh_failed' }),
          );
        }
      } finally {
        setRefreshing(false);
        // Bump regardless: if the API recomputes in the background, refetching
        // is how the page picks up the new `computed_at` when it lands.
        setRefreshToken((n) => n + 1);
      }
    },
    [leagueId],
  );

  const value = useMemo<AppState>(
    () => ({
      leagues,
      leaguesLoading,
      leaguesError,
      leagueId,
      league: leagues.find((l) => l.league_id === leagueId) ?? null,
      setLeagueId,
      meta,
      reportMeta: setMeta,
      refreshing,
      refresh,
      refreshToken,
      reloadLeagues: () => setLeaguesToken((n) => n + 1),
    }),
    [leagues, leaguesLoading, leaguesError, leagueId, setLeagueId, meta, refreshing, refresh, refreshToken],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

/* ==========================================================================
   Data hook
   ========================================================================== */

export interface Resource<T> {
  data: WithMeta<T> | null;
  error: ApiError | null;
  loading: boolean;
  /** Refetch without a server-side recompute. */
  reload: () => void;
}

/**
 * Fetch one payload, cancel it on unmount, and refetch when the global refresh
 * fires. The shell's staleness readout is driven from here, so a view that uses
 * this gets "computed 4 minutes ago" in the header for free.
 *
 *   const odds = useResource((signal) => api.odds(leagueId, { signal }), [leagueId]);
 *
 * `deps` behaves like a `useEffect` dependency list. The refresh token is added
 * automatically -- do not include it yourself.
 */
export function useResource<T>(
  fetcher: (signal: AbortSignal) => Promise<WithMeta<T>>,
  deps: unknown[],
): Resource<T> {
  const state = useContext(AppStateContext);
  const refreshToken = state?.refreshToken ?? 0;
  const reportMeta = state?.reportMeta;
  const [data, setData] = useState<WithMeta<T> | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [localToken, setLocalToken] = useState(0);

  // Keep the latest fetcher without making it a dependency: a view that builds
  // the closure inline would otherwise refetch on every render.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    const controller = new AbortController();
    let live = true;
    setLoading(true);
    fetcherRef
      .current(controller.signal)
      .then((payload) => {
        if (!live) return;
        setData(payload);
        setError(null);
        setLoading(false);
        reportMeta?.(payload.meta);
      })
      .catch((err: unknown) => {
        if (!live || isAbort(err)) return;
        setError(
          err instanceof ApiError ? err : new ApiError({ message: String(err), code: 'unknown' }),
        );
        setLoading(false);
      });
    return () => {
      live = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, refreshToken, localToken]);

  return { data, error, loading, reload: () => setLocalToken((n) => n + 1) };
}

/** A ticking clock, so "4m ago" becomes "5m ago" without a refetch. */
export function useNow(intervalMs = 15_000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs]);
  return now;
}

/* ==========================================================================
   Shell
   ========================================================================== */

/** Anything older than this is called out. A stale simulation is a wrong one. */
const STALE_AFTER_SECONDS = 30 * 60;

function LeagueSwitcher() {
  const { leagues, leaguesLoading, leagueId, setLeagueId } = useAppState();
  const navigate = useNavigate();
  const location = useLocation();

  function choose(id: number) {
    setLeagueId(id);
    // Stay on the same view when switching leagues -- a reader comparing waiver
    // boards should not be thrown back to the queue.
    const current = ROUTES.find(
      (r) => r.needsLeague && hrefFor(r, leagueId) === location.pathname,
    );
    if (current) navigate(hrefFor(current, id));
  }

  if (leaguesLoading && leagues.length === 0) {
    return (
      <div className="leagues">
        {[0, 1, 2].map((i) => (
          <div className="league league--skeleton" key={i}>
            <span className="league__name skeleton">league name</span>
            <span className="league__odds skeleton">0.0%</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="leagues" role="tablist" aria-label="League">
      {leagues.map((league) => {
        const active = league.league_id === leagueId;
        const failed = leagueFailed(league);
        const label = leagueLabel(league);
        // `/api/leagues` is a settings probe, not a simulation, so it carries no
        // championship probability. The sub-line says what the row actually
        // knows -- size, scoring, week -- rather than an em dash pretending the
        // odds are missing. If the API ever adds them, they render here.
        const hasOdds = typeof league.championship === 'number';
        return (
          <button
            type="button"
            role="tab"
            aria-selected={active}
            key={league.league_id}
            className={`league${active ? ' league--active' : ''}${failed ? ' league--errored' : ''}`}
            onClick={() => choose(league.league_id)}
            title={
              failed
                ? `${label}: ${league.error ?? 'ESPN would not serve this league'}`
                : `${label} · ${league.size ?? '?'} teams · ${
                    league.scoring_variant ?? 'ppr'
                  }${league.current_week ? ` · week ${league.current_week}` : ''}`
            }
          >
            <span className="league__name">{label}</span>
            <span className="league__odds">
              {failed ? (
                'unavailable'
              ) : hasOdds ? (
                <>
                  {fmt.pct(league.championship, 1)}
                  {league.championship_stderr ? (
                    <span className="err">±{fmt.ppError(league.championship_stderr, 2)}</span>
                  ) : null}
                </>
              ) : (
                <span className="faint">
                  {league.size ? `${league.size} teams` : league.scoring_variant ?? '—'}
                </span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function Staleness() {
  const { meta, refreshing, refresh, leagueId } = useAppState();
  const location = useLocation();
  const now = useNow(15_000);
  const age = ageSeconds(meta, now);
  const stale = age !== null && age > STALE_AFTER_SECONDS;

  // Refresh what is on screen: one league on a league view, everything on the
  // cross-league ones. The button says which, because a full recompute is
  // several seconds per league and the user should know what he is buying.
  const scope: RefreshScope = location.pathname.startsWith('/league/') && leagueId ? 'league' : 'all';
  const what = scope === 'league' ? 'this league' : 'all leagues';

  return (
    <div className="row">
      <div
        className={`staleness${stale ? ' staleness--stale' : ''}${
          refreshing || meta?.computing ? ' staleness--computing' : ''
        }`}
        title={meta?.computed_at ? `computed ${fmt.clock(meta.computed_at)}` : 'nothing computed yet'}
      >
        <span className="staleness__age">
          {refreshing ? 'computing…' : age === null ? 'not computed' : `computed ${fmt.ago(age)}`}
        </span>
        <span className="caption">
          {meta?.n_sims ? `${fmt.int(meta.n_sims)} sims` : meta?.cached ? 'cached' : ' '}
        </span>
      </div>
      <button
        type="button"
        className="btn btn--sm"
        onClick={() => void refresh(scope)}
        disabled={refreshing}
        title={`Recompute ${what} from ESPN. A cold run is a few seconds per league.`}
      >
        {refreshing ? <span className="spin" /> : null}
        {refreshing ? 'refreshing' : 'refresh'}
      </button>
    </div>
  );
}

/** The app chrome. Renders sensibly while data is loading and when a league failed. */
export function Layout({ children }: { children: ReactNode }) {
  const { leagueId, leaguesError, leagues, reloadLeagues } = useAppState();
  const failed = leagues.filter(leagueFailed);

  return (
    <div className="shell">
      <header className="shell__header">
        <div className="shell__bar">
          <div className="shell__mark" title="fantasy_quant">
            fq<span>·</span>
          </div>
          <LeagueSwitcher />
          <nav className="shell__nav">
            {ROUTES.map((route) => (
              <NavLink
                key={route.key}
                to={hrefFor(route, leagueId)}
                end={route.pattern === '/league/:leagueId'}
                title={route.help}
                className={({ isActive }) => `navlink${isActive ? ' navlink--active' : ''}`}
              >
                {route.label}
              </NavLink>
            ))}
          </nav>
          <div className="spacer" />
          <Staleness />
        </div>
        {leaguesError ? (
          <div className="shell__banner">
            <span>
              <strong>/api/leagues failed:</strong> {leaguesError.message}
            </span>
            <button type="button" className="btn btn--sm btn--ghost" onClick={reloadLeagues}>
              retry
            </button>
          </div>
        ) : null}
        {failed.length ? (
          <div className="shell__banner">
            <span>
              {failed.length === 1
                ? `${leagueLabel(failed[0]!)} could not be read: ${
                    failed[0]!.error ?? 'unknown error'
                  }`
                : `${failed.length} leagues could not be read. The rest of the page is unaffected.`}
            </span>
          </div>
        ) : null}
      </header>

      <main className="shell__main">{children}</main>

      <footer className="shell__footer">
        <span>
          Every number here is computed by <code className="mono">fantasy_quant</code> and read, not
          recomputed, by this page.
        </span>
        <span className="spacer" />
        <span>Δ P(title) in percentage points · ± is Monte Carlo error at the stated sims</span>
      </footer>
    </div>
  );
}

/* ==========================================================================
   States every view needs
   ========================================================================== */

/** A view waiting on a cold simulation. Says what it is waiting for. */
/**
 * `slow` is for the cross-league views, which run every surface in every league before
 * they can rank anything. Measured cold at 4,000 sims: ~334s for the queue against ~3s
 * for a league simulation. Saying "the decision surfaces take longer" in front of a
 * five-minute wait reads as a hang, and the honest number is the difference between
 * waiting and reaching for the reload.
 */
export function Loading({ what = 'simulation', slow = false }: { what?: string; slow?: boolean }) {
  return (
    <div className="state">
      <div className="row">
        <span className="spin" />
        <span className="state__title">running the {what}</span>
      </div>
      <span>
        {slow
          ? 'Every surface in every league, about five minutes on a freshly started server. It is cached afterwards, so this is paid once and later loads are instant.'
          : 'A cold league is about three seconds; the decision surfaces take longer.'}
      </span>
    </div>
  );
}

/** A failed fetch, with the server's own message. Never a bare "something went wrong". */
export function Failure({ error, onRetry }: { error: ApiError | Error; onRetry?: () => void }) {
  const api_ = error instanceof ApiError ? error : null;
  return (
    <div className="state state--error">
      <div className="state__title">
        {api_?.code === 'timeout' ? 'timed out' : 'could not load this view'}
      </div>
      <div className="prose">{error.message}</div>
      {api_?.detail ? <div className="state__detail">{api_.detail}</div> : null}
      {onRetry ? (
        <button type="button" className="btn btn--sm" onClick={onRetry}>
          try again
        </button>
      ) : null}
    </div>
  );
}

/** A section the payload reported as failed. One dead surface, not a dead page. */
export function SectionFailure({ name, error }: { name: string; error: string }) {
  return (
    <div className="note note--bad">
      <strong>{name}</strong> could not be computed: {error}
    </div>
  );
}
