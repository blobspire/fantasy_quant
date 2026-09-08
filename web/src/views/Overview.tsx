/**
 * Where I stand: every league at a glance.
 *
 * Title odds, playoff odds, rank, this week's matchup and its leverage, one
 * card per league, with the error on every number that has one. Two leagues
 * working and one refusing to build renders as two cards and one stated
 * failure — a per-league error never costs the reader the leagues that worked.
 *
 * The rank in particular is mostly Monte Carlo noise: `report.odds_payload`
 * counts how many adjacent pairs on the board are actually separated by more
 * than twice the paired error, and at 4,000 simulations on a 14-team league
 * that is a small minority of them. So the count sits directly under the rank
 * rather than in a footnote, and the rank recedes when most of the order is
 * unresolved. Nothing here is computed client-side.
 */
import {
  ApiError,
  api,
  fmt,
  leagueFailed,
  leagueLabel,
  readMeta,
  sectionData,
  sectionError,
  type LeagueSummary,
  type LeveragePayload,
  type OddsPayload,
} from '../api';
import { Failure, Loading, useAppState, useResource } from '../components/Layout';
import { Stat } from '../components/Stat';
import { Panel, type ViewProps } from './Queue';

/** Below this a marginal point is worth under a quarter of its coin-flip value. */
export const DECIDED_LEVERAGE = 0.25;

interface LeagueCard {
  league: LeagueSummary;
  odds: OddsPayload | null;
  oddsError: string | null;
  leverage: LeveragePayload | null;
  leverageError: string | null;
  /** Simulations behind this league's numbers, off the weekly envelope. */
  nSims: number | null;
}

/**
 * Odds and leverage in one request per league.
 *
 * There is no `/leverage` route -- the leverage schedule is a section of the
 * weekly payload -- and `skip` drops the three expensive surfaces, so this
 * costs one league simulation rather than the whole decision stack.
 */
const OVERVIEW_SKIP = ['lineup', 'waivers', 'trades', 'stream'];

function messageOf(reason: unknown): string {
  if (reason instanceof ApiError) return reason.message;
  if (reason instanceof Error) return reason.message;
  return String(reason);
}

export default function Overview({ onNavigate }: ViewProps) {
  const { leagues, leaguesLoading, leaguesError } = useAppState();
  // `leagueFailed`, not `row.ok`. A row of `/api/leagues` is a settings probe
  // and carries `reachable`; it has no `ok` at all, so `ok !== false` was true
  // for a league whose cookies had expired -- this view then spent a request on
  // it and rendered the weekly failure instead of the registry's own message,
  // and the "did not build" list below could never contain anything.
  const ids = leagues
    .filter((league) => !leagueFailed(league))
    .map((league) => league.league_id)
    .join(',');

  const board = useResource<{ cards: LeagueCard[] }>(
    async (signal) => {
      const wanted = leagues.filter((league) => !leagueFailed(league));
      const settled = await Promise.allSettled(
        wanted.map((league) =>
          api.weekly(league.league_id, { skip: OVERVIEW_SKIP }, { signal }),
        ),
      );
      let meta = readMeta(null);
      const cards = wanted.map((league, index): LeagueCard => {
        const result = settled[index];
        if (!result || result.status === 'rejected') {
          const reason = result ? messageOf(result.reason) : 'no response';
          return {
            league,
            odds: null,
            oddsError: reason,
            leverage: null,
            leverageError: reason,
            nSims: null,
          };
        }
        const weekly = result.value;
        meta = weekly.meta;
        return {
          league,
          odds: sectionData<OddsPayload>(weekly.odds),
          oddsError: weekly.ok === false ? (weekly.error ?? 'league did not build') : sectionError(weekly.odds),
          leverage: sectionData<LeveragePayload>(weekly.leverage),
          leverageError: sectionError(weekly.leverage),
          nSims: weekly.n_sims ?? null,
        };
      });
      return { cards, meta };
    },
    [ids],
  );

  if (leaguesLoading && !leagues.length) return <Loading what="league registry" />;
  if (leaguesError && !leagues.length) return <Failure error={leaguesError} />;
  if (board.loading && !board.data) return <Loading what="championship simulation for every league" />;
  if (board.error && !board.data) return <Failure error={board.error} onRetry={board.reload} />;

  const cards = board.data?.cards ?? [];
  const failedToBuild = leagues.filter((league) => leagueFailed(league));

  return (
    <div className="grid">
      <header>
        <h1 className="panel__title" style={{ fontSize: 'var(--fs-lg)' }}>
          Where I stand
        </h1>
        <div className="caption">
          Title odds, playoff odds, rank and this week’s matchup, across every league.
        </div>
      </header>

      {failedToBuild.map((league) => (
        <p className="note note--bad" key={league.league_id}>
          <strong>{leagueLabel(league)}</strong> could not be read:{' '}
          {league.error ?? 'ESPN would not serve this league'}. It was not simulated, and every
          card below is unaffected.
        </p>
      ))}

      {cards.length === 0 ? (
        <div className="state">
          <div className="state__title">No league came back.</div>
          <p className="prose">
            The registry is empty, or every league in it failed. Check{' '}
            <span className="mono">config/leagues.toml</span> and the ESPN credentials the API loads
            from <span className="mono">.env</span> — the browser never sees them.
          </p>
        </div>
      ) : (
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(440px, 1fr))' }}>
          {cards.map((card) => (
            <Card
              key={card.league.league_id}
              card={card}
              onOpen={onNavigate ? () => onNavigate(`/league/${card.league.league_id}`) : undefined}
            />
          ))}
        </div>
      )}

      <div className="caption">
        Championship probabilities are `fq odds`: the baseline where an unfilled starting slot
        scores zero. The waiver board reports a different level — it streams a replacement into the
        empty seat, worth about 1.4pp on a thin roster — so compare deltas across surfaces, never
        levels.
      </div>
    </div>
  );
}

function Card({ card, onOpen }: { card: LeagueCard; onOpen?: () => void }) {
  const { league, odds } = card;

  if (!odds) {
    return (
      <Panel title={leagueLabel(league)}>
        <div className="grid" style={{ gap: 'var(--sp-4)' }}>
          <p className="note note--bad">
            The championship table failed: {card.oddsError ?? 'unknown error'}
          </p>
        </div>
      </Panel>
    );
  }

  const me = odds.teams.find((team) => team.is_me) ?? null;
  const rank = odds.my_rank ?? me?.rank ?? null;
  const separated = odds.n_ranks_separated;
  const nRanks = odds.n_ranks;

  return (
    <Panel
      title={
        onOpen ? (
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={onOpen}
            style={{ padding: 0, border: 'none', background: 'none', fontSize: 'var(--fs-md)' }}
          >
            {odds.name || leagueLabel(league)}
          </button>
        ) : (
          odds.name || leagueLabel(league)
        )
      }
      right={
        <div className="row">
          {me ? <span className="chip">{fmt.record(me.wins, me.losses, me.ties)}</span> : null}
          <span className="chip">{odds.size} teams</span>
          {odds.week ? <span className="chip">week {odds.week}</span> : null}
        </div>
      }
    >
      <div className="grid" style={{ gap: 'var(--sp-5)' }}>
        <div className="row row--wrap" style={{ gap: 'var(--sp-7)', alignItems: 'flex-end' }}>
          <Stat
            size="lg"
            label="P(championship)"
            value={odds.my_championship}
            format="pct"
            digits={2}
            stderr={odds.my_championship_stderr}
            sub={`${fmt.int(card.nSims ?? odds.n_sims)} simulations`}
          />
          <Stat size="sm" label="P(playoffs)" value={odds.my_playoffs} format="pct" />
          <Stat size="sm" label="P(bye)" value={me?.bye ?? null} format="pct" />
          <Stat size="sm" label="E[wins]" value={me?.expected_wins ?? null} format="points" />
          <Stat
            size="sm"
            label="rank"
            value={rank === null ? null : `${rank} of ${odds.size}`}
            format="raw"
            note={odds.ranking_note}
            sub={nRanks > 0 ? `${separated} of ${nRanks} adjacent pairs separated` : undefined}
          />
        </div>

        <ThisWeek card={card} />

        {odds.corpus_variant !== odds.corpus_variant_requested ? (
          <p className="note note--warn">
            No <span className="mono">{odds.corpus_variant_requested}</span> corpus for this season;
            component stats were read from the <span className="mono">{odds.corpus_variant}</span>{' '}
            capture and re-scored with this league’s own scoring function.
          </p>
        ) : null}
      </div>
    </Panel>
  );
}

/**
 * This week's matchup, and whether it is worth caring about.
 *
 * When the leverage says the game is effectively decided, that sentence is the
 * output — the start/sit call in a blowout is worth a fraction of the same call
 * in a coin flip, and hiding that sells busywork.
 */
function ThisWeek({ card }: { card: LeagueCard }) {
  if (!card.leverage) {
    return (
      <div className="caption">
        This week’s matchup did not load{card.leverageError ? `: ${card.leverageError}` : ''}.
      </div>
    );
  }
  const week = card.leverage.this_week;
  if (!week) return <div className="caption">No unplayed week left in the regular season.</div>;
  const decided = week.decided || week.leverage < DECIDED_LEVERAGE;
  return (
    <div className="grid" style={{ gap: 'var(--sp-3)' }}>
      <div className="row row--wrap" style={{ gap: 'var(--sp-6)', alignItems: 'flex-end' }}>
        <Stat size="xs" label={`week ${week.week} vs`} value={week.opponent} format="raw" />
        <Stat
          size="xs"
          label="projected margin"
          value={fmt.signed(week.margin, 1)}
          format="raw"
          sub={`± ${fmt.num(week.sd_diff, 1)}`}
        />
        <Stat size="xs" label="P(win)" value={week.win_probability} format="pct" />
        <Stat
          size="xs"
          label="leverage"
          value={week.leverage}
          format="number"
          digits={2}
          significant={week.decided ? false : undefined}
          sub={`${fmt.num(week.points_per_win_pct, 2)}pp of the game per point`}
        />
      </div>
      {decided ? (
        <p className="note note--warn">
          This week is close to decided (leverage {fmt.num(week.leverage, 2)}). A marginal upgrade
          moves it by about {fmt.num(week.points_per_win_pct, 2)}pp per projected point — the
          start/sit call barely matters here.
        </p>
      ) : null}
    </div>
  );
}
