import { Component, Suspense, lazy, type ComponentType, type ReactNode } from 'react';
import { HashRouter, Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import {
  AppStateProvider,
  Layout,
  Loading,
  ROUTES,
  useAppState,
  useLeagueId,
  type ViewRoute,
} from './components/Layout';

/**
 * Wiring.
 *
 * ROUTING is hash-based on purpose. The built bundle is served as static files
 * by the FastAPI app; a hash route means a deep link like
 * `#/league/272150391/waivers` resolves in the browser and never asks the
 * server for a path it has no handler for. No SPA-fallback route is needed on
 * the Python side.
 *
 * VIEWS ARE DISCOVERED, NOT IMPORTED. `import.meta.glob` is resolved by Vite at
 * build time and returns an empty map when `src/views/` is empty, so this shell
 * compiles and runs before a single view exists -- which is the point, since the
 * views are written in parallel. A route whose module is missing renders a
 * placeholder naming the file it wants, and a view module that exists but is not
 * in `ROUTES` is still mounted, at `/<name>`, so nothing anyone writes is
 * unreachable.
 *
 * WHAT A VIEW GETS. Every prop is optional; a view may ignore all of them and
 * read the league out of the URL instead.
 *
 *   leagueId    the league in the switcher, or the one in the route
 *   season      that league's season
 *   onNavigate  push a path through the router (do NOT use history.pushState
 *               directly -- this is a HashRouter and a raw push would desync it)
 *
 * To add a view: create `src/views/<View>.tsx` with a default-exported
 * component and add a row to `ROUTES` in `components/Layout.tsx` for its nav
 * entry. Nothing here needs to change.
 */
const viewModules = import.meta.glob('./views/*.tsx');

/** What the shell passes a view. All optional, by design. */
export interface ShellViewProps {
  leagueId?: number | null;
  season?: number | null;
  onNavigate?: (path: string) => void;
}

type ViewModule = { default: ComponentType<ShellViewProps> };

function loaderFor(view: string): (() => Promise<unknown>) | null {
  const wanted = `./views/${view.toLowerCase()}.tsx`;
  const key = Object.keys(viewModules).find((path) => path.toLowerCase() === wanted);
  return key ? (viewModules[key] as () => Promise<unknown>) : null;
}

/** Every view module on disk, by basename. */
function discoveredViews(): string[] {
  return Object.keys(viewModules)
    .map((path) => path.replace('./views/', '').replace(/\.tsx$/, ''))
    .sort();
}

// `lazy()` must be called once per module, not once per render, or the view
// remounts (and refetches) on every parent render.
const lazyCache = new Map<string, ComponentType<ShellViewProps>>();

function lazyView(view: string): ComponentType<ShellViewProps> | null {
  const cached = lazyCache.get(view);
  if (cached) return cached;
  const loader = loaderFor(view);
  if (!loader) return null;
  const component = lazy(async () => {
    const module = (await loader()) as Partial<ViewModule>;
    if (typeof module.default !== 'function') {
      const name = view;
      return {
        default: () => (
          <div className="state state--error">
            <div className="state__title">{name} has no default export</div>
            <div className="state__detail">web/src/views/{name}.tsx must default-export a component</div>
          </div>
        ),
      } satisfies ViewModule;
    }
    return module as ViewModule;
  });
  lazyCache.set(view, component);
  return component;
}

/** What a route renders before its view module exists. Honest about why. */
function MissingView({ route }: { route: ViewRoute }) {
  return (
    <div className="state">
      <div className="state__title">{route.label} is not built yet</div>
      <div className="prose">{route.help}</div>
      <div className="state__detail">expected: web/src/views/{route.view}.tsx (default export)</div>
    </div>
  );
}

/**
 * One broken view must not take the shell down with it -- the league switcher
 * and the refresh button are how a reader gets to a view that does work.
 */
class ViewBoundary extends Component<
  { children: ReactNode; label: string },
  { error: Error | null }
> {
  constructor(props: { children: ReactNode; label: string }) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  override render() {
    if (this.state.error) {
      return (
        <div className="state state--error">
          <div className="state__title">{this.props.label} failed to render</div>
          <div className="prose">{this.state.error.message}</div>
          <div className="state__detail">{this.state.error.stack?.split('\n')[1]?.trim()}</div>
          <button
            type="button"
            className="btn btn--sm"
            onClick={() => this.setState({ error: null })}
          >
            try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

/** Mounts one view with the shell's context wired into its props. */
function ViewHost({ route, view }: { route: ViewRoute; view: string }) {
  const { league } = useAppState();
  const leagueId = useLeagueId();
  const navigate = useNavigate();
  const View = lazyView(view);
  if (!View) return <MissingView route={route} />;
  return (
    <ViewBoundary label={route.label}>
      <Suspense fallback={<Loading what={route.label.toLowerCase()} />}>
        <View
          leagueId={leagueId}
          season={league?.season ?? null}
          onNavigate={(path: string) => navigate(path)}
        />
      </Suspense>
    </ViewBoundary>
  );
}

function NotFound() {
  return (
    <div className="state">
      <div className="state__title">no such view</div>
      <div className="prose">
        Pick one from the nav. The queue is the place to start: it is every league in one order.
      </div>
    </div>
  );
}

export default function App() {
  const named = new Set(ROUTES.map((route) => route.view.toLowerCase()));
  const extras = discoveredViews().filter((view) => !named.has(view.toLowerCase()));

  return (
    <HashRouter>
      <AppStateProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/queue" replace />} />
            {ROUTES.map((route) => (
              <Route
                key={route.key}
                path={route.pattern}
                element={<ViewHost route={route} view={route.view} />}
              />
            ))}
            {/* Reachable but not in the nav: a view somebody wrote that the
                route table above does not name. Better than a 404 on a file
                that exists. */}
            {extras.map((view) => (
              <Route
                key={`extra:${view}`}
                path={`/${view.toLowerCase()}`}
                element={
                  <ViewHost
                    route={{
                      key: view.toLowerCase(),
                      label: view,
                      view,
                      pattern: `/${view.toLowerCase()}`,
                      needsLeague: false,
                    }}
                    view={view}
                  />
                }
              />
            ))}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Layout>
      </AppStateProvider>
    </HashRouter>
  );
}
