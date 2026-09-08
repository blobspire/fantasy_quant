/**
 * The portfolio: three leagues as one position.
 *
 * The two numbers a multi-league player actually wants are P(≥1 title) and
 * P(0 titles), and neither is obtainable from the three individual
 * probabilities without knowing how they covary — which is why
 * `edges/portfolio.py` draws every league against ONE simulated NFL season and
 * reads the joint outcome off it. E[titles] is the exception: it is the sum of
 * the marginals exactly, for any dependence whatsoever, and that is the entire
 * reason the action queue can rank across leagues at all.
 *
 * Concentration is reported as a re-simulated event, never as a rule. There is
 * no defensible "never more than X% in one player" constant, so every row reads
 * as one sentence: if this happens, the portfolio goes from A to B, ± this much.
 */
import { useState } from 'react';

import {
  api,
  fmt,
  type Concentration,
  type Exposure,
  type PairCorrelation,
  type PortfolioOdds,
  type PortfolioPayload,
} from '../api';
import { Delta } from '../components/Delta';
import { Failure, Loading, useResource } from '../components/Layout';
import { Stat } from '../components/Stat';
import { Table, type Column } from '../components/Table';
import { Legend, Panel } from './Queue';

/* ==========================================================================
   Shapes the wire carries but `api.ts` does not name yet
   ==========================================================================
   `PortfolioReport` also publishes the pro-team concentration rows, the bye
   table and the diversification verdict, and several of the fields below are
   Python `@property` values that a plain `asdict` drops. Everything here is
   read defensively: a missing `significant` renders as an unjudged number, not
   as a verdict this client invented.
   ========================================================================== */

type ConcentrationRow = Concentration & {
  significant?: boolean;
  increment_significant?: boolean;
  n_removed?: number;
};

interface ByeRow {
  week: number;
  starters_out?: Record<string, string[]>;
  normal_points?: Record<string, number>;
  bye_points?: Record<string, number>;
  weekly_mean?: Record<string, number>;
  /** Starters whose bye the projections did NOT zero. `api/server.portfolio_payload`. */
  unpriced?: Array<{ league: string; player: string; bye_points: number; season_mean: number }>;
  total_out?: number;
  worst_share?: number;
}

interface DiversificationView {
  concentration_headroom?: number;
  diversification_headroom?: number;
  verdict?: string;
}

function extras(payload: PortfolioPayload) {
  const record = payload as unknown as Record<string, unknown>;
  return {
    proTeams: (record['pro_teams'] as ConcentrationRow[] | undefined) ?? [],
    byes: (record['byes'] as ByeRow[] | undefined) ?? [],
    diversification: (record['diversification'] as DiversificationView | null | undefined) ?? null,
  };
}

/**
 * A signed portfolio effect, judged only if the server judged it.
 *
 * `Delta` resolves an absent `significant` to "noise", which would be this
 * client asserting a significance test it did not run. So a row whose flag the
 * payload omits renders as a plain number beside its error instead.
 */
function Effect({
  value,
  stderr,
  significant,
  digits = 2,
}: {
  value: number | null | undefined;
  stderr?: number | null;
  significant?: boolean;
  digits?: number;
}) {
  if (typeof significant === 'boolean') {
    return <Delta value={value} stderr={stderr} significant={significant} digits={digits} size="sm" />;
  }
  return (
    <span className="delta">
      <span className="delta__value">{fmt.pp(value, digits)}</span>
      {typeof stderr === 'number' && stderr > 0 ? (
        <span className="delta__err">±{fmt.ppError(stderr, digits)}</span>
      ) : null}
    </span>
  );
}

/* ==========================================================================
   View
   ========================================================================== */

export default function Portfolio() {
  const book = useResource<PortfolioPayload>((signal) => api.portfolio({ signal }), []);

  if (book.loading && !book.data) {
    return <Loading what="portfolio across one shared NFL season" />;
  }
  if (book.error && !book.data) return <Failure error={book.error} onRetry={book.reload} />;
  if (!book.data) return null;

  const data = book.data;
  const { proTeams, byes, diversification } = extras(data);
  const exposures = data.exposures ?? [];

  if (!data.odds && !exposures.length) {
    return (
      <div className="state">
        <div className="state__title">No portfolio came back.</div>
        <p className="prose">
          The cross-league view needs every league on one seed and one simulated NFL season
          (`edges.portfolio.build_portfolio`). If one league will not build, the portfolio cannot be
          coupled and the API says so rather than mixing two universes.
        </p>
      </div>
    );
  }

  return (
    <div className="grid">
      <header className="row row--wrap" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1 className="panel__title" style={{ fontSize: 'var(--fs-lg)' }}>
            Portfolio
          </h1>
          <div className="caption">
            Every league drawn against one shared NFL season, so the overlap is measured rather than
            assumed.
          </div>
        </div>
      </header>

      {book.error ? (
        <p className="note note--warn">Showing the last good portfolio: {book.error.message}</p>
      ) : null}
      {(data.errors ?? []).map((failure, index) => (
        <p className="note note--bad" key={index}>
          <strong>{failure.league ?? failure.league_id ?? 'a league'}</strong>
          {failure.surface ? ` / ${failure.surface}` : ''}: {failure.error}
        </p>
      ))}

      {data.odds ? <Odds odds={data.odds} diversification={diversification} /> : null}
      <Exposures rows={exposures} />
      <ConcentrationPanel
        title="If one of these happens"
        subtitle="re-simulated per event, not a rule of thumb"
        rows={(data.concentration ?? []) as ConcentrationRow[]}
      />
      <ConcentrationPanel
        title="NFL-team concentration"
        subtitle="every roster spot on one pro team, deleted together"
        rows={proTeams}
      />
      <Byes rows={byes} />
      <Correlations pairs={data.correlations?.pairs ?? []} />
      <Legend />
    </div>
  );
}

/* ==========================================================================
   Joint odds
   ========================================================================== */

function Odds({
  odds,
  diversification,
}: {
  odds: PortfolioOdds;
  diversification: DiversificationView | null;
}) {
  return (
    <Panel
      title="Joint odds"
      right={<span className="chip">{fmt.int(odds.n_sims)} shared seasons</span>}
    >
      <div className="grid" style={{ gap: 'var(--sp-6)' }}>
        <div className="row row--wrap" style={{ gap: 'var(--sp-8)', alignItems: 'flex-end' }}>
          <Stat
            size="lg"
            label="P(at least one title)"
            value={odds.p_at_least_one}
            format="pct"
            digits={2}
            stderr={odds.stderr}
          />
          <Stat size="md" label="P(zero titles)" value={odds.p_zero} format="pct" digits={2} />
          <Stat size="sm" label="P(two or more)" value={odds.p_two_plus} format="pct" digits={2} />
          <Stat
            size="sm"
            label="E[titles]"
            value={odds.expected_titles}
            format="number"
            digits={4}
            sub="the sum of the column below, exactly, whatever the dependence"
          />
          <Stat
            size="sm"
            label="Var[titles]"
            value={odds.variance_titles}
            format="number"
            digits={4}
          />
        </div>

        <div className="row row--wrap" style={{ gap: 'var(--sp-6)', alignItems: 'flex-start' }}>
          <div style={{ minWidth: 280, flex: '1 1 280px' }}>
            {/* These are the COUPLED draw's marginals, not `fq odds`. Both are
                the same estimand at 4,000 sims and they land a few tenths of a
                point apart (Blacksburg is 5.12% on its own draw and 5.83%
                here). A reader who flips between this page and the league page
                and finds two championship probabilities with no explanation
                will conclude one of them is wrong, so the table says which
                draw it is from. */}
            <Table
              compact
              columns={[
                { key: 'league', header: 'league', render: (row: LeagueTitle) => row.name },
                {
                  key: 'p',
                  header: 'P(title)',
                  help: 'this league’s marginal on the shared-season draw',
                  num: true,
                  width: 90,
                  render: (row: LeagueTitle) => fmt.pct(row.title, 2),
                },
              ]}
              rows={odds.names.map((name, index) => ({ name, title: odds.titles[index] ?? null }))}
              rowKey={(row) => row.name}
              footer={
                <span className="caption">
                  Read off the one shared NFL season this page is built on, at{' '}
                  {fmt.int(odds.n_sims)} sims. That is a different draw from the per-league
                  simulation behind <span className="mono">fq odds</span> and the league page, so
                  each of these sits within a few tenths of a point of the figure shown there. The
                  sum is E[titles] either way.
                </span>
              }
            />
          </div>
          <div style={{ minWidth: 320, flex: '1 1 320px' }}>
            <Table
              compact
              columns={[
                { key: 'case', header: 'P(≥1) under…', render: (row: BoundRow) => row.label },
                {
                  key: 'value',
                  header: 'value',
                  num: true,
                  width: 90,
                  render: (row: BoundRow) => fmt.pct(row.value, 2),
                },
              ]}
              rows={[
                { label: 'the measured dependence', value: odds.p_at_least_one, dim: false },
                { label: 'independence', value: odds.independent, dim: true },
                { label: 'never winning together (Fréchet upper)', value: odds.sum_bound, dim: true },
                { label: 'always winning together (Fréchet lower)', value: odds.max_bound, dim: true },
              ]}
              rowKey={(row) => row.label}
              dim={(row) => row.dim}
            />
          </div>
        </div>

        <div className="row row--wrap" style={{ gap: 'var(--sp-7)', alignItems: 'flex-end' }}>
          <div className="stat stat--sm">
            <div className="stat__label">overlap’s effect on P(≥1 title)</div>
            <div className="stat__figure">
              <Effect
                value={-odds.dependence_cost}
                stderr={odds.dependence_cost_stderr}
                significant={odds.dependence_significant}
                digits={3}
              />
            </div>
            <div className="stat__sub">
              {/* Signed as an effect, so the overlap reads negative. The
                  server's verdict prose below quotes the same figure as a
                  positive COST; say so here rather than leaving a +/- clash
                  between two numbers a few lines apart. */}
              measured minus independent — the verdict below quotes the same figure as a cost, so
              it is positive there.{' '}
              {odds.dependence_significant
                ? 'A real effect on P(≥1 title).'
                : 'Inside its own error — not distinguishable from zero.'}
            </div>
          </div>
          {diversification?.concentration_headroom !== undefined ? (
            <Stat
              size="sm"
              label="if fully concentrated"
              value={-diversification.concentration_headroom}
              format="pp"
            />
          ) : null}
          {diversification?.diversification_headroom !== undefined ? (
            <Stat
              size="sm"
              label="if perfectly diversified"
              value={diversification.diversification_headroom}
              format="pp"
            />
          ) : null}
        </div>

        {diversification?.verdict ? <p className="prose">{diversification.verdict}</p> : null}
      </div>
    </Panel>
  );
}

interface LeagueTitle {
  name: string;
  title: number | null;
}

interface BoundRow {
  label: string;
  value: number;
  dim: boolean;
}

/* ==========================================================================
   Exposure
   ========================================================================== */

function Exposures({ rows }: { rows: Exposure[] }) {
  const [showAll, setShowAll] = useState(false);
  const shared = rows.filter((row) => (row.n_leagues ?? row.holdings?.length ?? 1) > 1);
  const shown = showAll ? rows : shared.length ? shared : rows.slice(0, 12);

  if (!rows.length) {
    return (
      <Panel title="Exposure">
        <div className="state">
          <div className="state__title">No exposures in this payload.</div>
        </div>
      </Panel>
    );
  }

  const columns: Array<Column<Exposure>> = [
    { key: 'name', header: 'player', width: '22%', value: (row) => row.name },
    { key: 'pos', header: 'pos', width: 52, value: (row) => row.position },
    {
      key: 'leagues',
      header: 'lg / st',
      help: 'leagues he is held in / of those, the ones he starts in',
      num: true,
      width: 70,
      value: (row) => row.n_leagues ?? row.holdings?.length ?? 0,
      render: (row) =>
        `${row.n_leagues ?? row.holdings?.length ?? fmt.dash} / ${row.n_starting ?? fmt.dash}`,
    },
    {
      key: 'equity',
      header: 'equity at risk',
      help: 'title probability he holds up, summed over leagues, in E[titles]',
      num: true,
      width: 106,
      defaultDesc: true,
      value: (row) => row.equity_at_risk,
      render: (row) => fmt.pp(row.equity_at_risk, 3),
    },
    {
      key: 'share',
      header: 'share',
      num: true,
      width: 68,
      value: (row) => row.equity_share,
      render: (row) => <span className="faint">{fmt.pct(row.equity_share, 1)}</span>,
    },
    {
      key: 'damage',
      header: 'ΔP(≥1 title)',
      help: 'P(≥1 title) today minus P(≥1 title) with him gone from every roster',
      num: true,
      width: 128,
      value: (row) => -row.portfolio_damage,
      render: (row) => (
        <Effect
          value={-row.portfolio_damage}
          stderr={row.portfolio_damage_stderr}
          significant={row.significant}
          digits={3}
        />
      ),
    },
    {
      key: 'where',
      header: 'where',
      sortable: false,
      render: (row) => (
        <div className="caption" style={{ whiteSpace: 'normal' }}>
          {(row.holdings ?? [])
            .map(
              (holding) =>
                `${holding.league_name} (${holding.slot}, starts ${fmt.pct(holding.start_share, 0)})`,
            )
            .join(' · ')}
        </div>
      ),
    },
  ];

  return (
    <Panel
      flush
      title="Exposure"
      right={
        <div className="row">
          <span className="caption">
            {shared.length} player{shared.length === 1 ? '' : 's'} held in more than one league
          </span>
          <button type="button" className="btn btn--sm btn--ghost" onClick={() => setShowAll((v) => !v)}>
            {showAll ? 'shared only' : `all ${rows.length}`}
          </button>
        </div>
      }
    >
      <Table
        columns={columns}
        rows={shown}
        rowKey={(row) => row.player_id}
        dim={(row) => row.significant === false}
        maxHeight={560}
        footer={
          <span className="caption">
            The ± is Monte Carlo only: it says how well this draw pins the number down, not how well
            the projections do. A dimmed row is a holding whose loss this simulation cannot
            distinguish from costing nothing. Equity at risk is a one-term Shapley approximation and
            is <strong>not additive across players</strong> — removing the WR1 promotes the WR4.
          </span>
        }
      />
    </Panel>
  );
}

/* ==========================================================================
   Concentration
   ========================================================================== */

function ConcentrationPanel({
  title,
  subtitle,
  rows,
}: {
  title: string;
  subtitle: string;
  rows: ConcentrationRow[];
}) {
  if (!rows.length) return null;
  return (
    <Panel title={title} right={<span className="caption">{subtitle}</span>}>
      <div className="grid" style={{ gap: 'var(--sp-5)' }}>
        {rows.map((row, index) => (
          <ConcentrationLine key={`${row.kind}:${row.label}:${index}`} row={row} />
        ))}
      </div>
    </Panel>
  );
}

function ConcentrationLine({ row }: { row: ConcentrationRow }) {
  // `damage` is the Python property `before - after` and does not survive a
  // plain `asdict`. Subtracting two published probabilities is arithmetic on
  // the payload, not a re-derivation of the analysis.
  const damage = row.damage ?? row.before - row.after;
  const removed = Object.entries(row.removed ?? {});
  const nRemoved =
    row.n_removed ?? removed.reduce((total, [, players]) => total + players.length, 0);
  const recedes = row.significant === false;

  return (
    <div style={{ opacity: recedes ? 0.72 : 1 }}>
      <div className="row row--wrap" style={{ gap: 'var(--sp-5)', alignItems: 'baseline' }}>
        <span style={{ color: 'var(--fg)', fontSize: 'var(--fs-md)' }}>
          {row.kind === 'player' ? 'If ' : ''}
          <strong>{row.label}</strong>
          {row.kind === 'player' ? ' misses the season' : ` (${nRemoved} roster spots) is out`}
        </span>
        <span className="num">
          {fmt.pct(row.before, 2)} <span className="faint">→</span> {fmt.pct(row.after, 2)}
        </span>
        <Effect value={-damage} stderr={row.stderr} significant={row.significant} />
        <span className="caption">
          E[titles] {fmt.num(row.expected_before, 4)} → {fmt.num(row.expected_after, 4)}
        </span>
        {recedes ? <span className="chip chip--warn">inside its own error</span> : null}
      </div>
      {removed.length ? (
        <div className="caption">
          {removed.map(([league, players]) => `${league}: ${players.join(', ')}`).join(' · ')}
        </div>
      ) : null}
      {nRemoved > 1 && row.driver ? (
        <div className="caption">
          {row.driver} alone is {fmt.ppError(row.driver_damage, 2)} of it; the other {nRemoved - 1}{' '}
          {nRemoved - 1 === 1 ? 'adds' : 'add'} {fmt.pp(row.increment, 2)} ±
          {fmt.ppError(row.increment_stderr, 2)}
          {row.increment_significant === undefined
            ? ''
            : row.increment_significant
              ? ' — the grouping is worth more than its driver'
              : ' — not separable from its driver alone'}
        </div>
      ) : null}
    </div>
  );
}

/* ==========================================================================
   Byes
   ========================================================================== */

function Byes({ rows }: { rows: ByeRow[] }) {
  if (!rows.length) return null;
  const unpriced = rows.flatMap((row) =>
    (row.unpriced ?? []).map((entry) => ({ week: row.week, ...entry })),
  );
  const columns: Array<Column<ByeRow>> = [
    { key: 'week', header: 'wk', num: true, width: 44, value: (row) => row.week },
    {
      key: 'out',
      header: 'out',
      num: true,
      width: 56,
      value: (row) => row.total_out,
      render: (row) => row.total_out ?? fmt.dash,
    },
    {
      key: 'share',
      header: 'worst share',
      help: 'largest share of one team’s normal weekly scoring that is on bye',
      num: true,
      width: 96,
      value: (row) => row.worst_share,
      render: (row) => fmt.pct(row.worst_share, 0),
    },
    {
      key: 'who',
      header: 'who',
      sortable: false,
      render: (row) => (
        <div className="caption" style={{ whiteSpace: 'normal' }}>
          {Object.entries(row.starters_out ?? {})
            .map(([league, players]) => {
              const normal = row.normal_points?.[league];
              const priced = row.bye_points?.[league];
              const weekly = row.weekly_mean?.[league];
              return `${league}: ${players.join(', ')} — ${fmt.num(normal, 1)} of ${fmt.num(
                weekly,
                1,
              )} weekly pts, priced at ${fmt.num(priced, 1)}`;
            })
            .join(' · ')}
        </div>
      ),
    },
  ];
  return (
    <Panel
      flush
      title="Bye weeks"
      right={
        <span className="caption">
          no title figure here: the projections already price most of a bye, and converting one
          would double-count it
        </span>
      }
    >
      <Table
        compact
        columns={columns}
        rows={rows}
        rowKey={(row) => row.week}
        footer={
          unpriced.length ? (
            <span className="note note--bad">
              {unpriced.length} starter(s) whose bye the projections did NOT zero, so the model
              starts them and scores them on a week they do not play:{' '}
              {unpriced
                .map(
                  (row) =>
                    `w${row.week} ${row.player} (${row.league}) projects ${fmt.num(
                      row.bye_points,
                      2,
                    )} against a season mean of ${fmt.num(row.season_mean, 2)}`,
                )
                .join('; ')}
              .
            </span>
          ) : undefined
        }
      />
    </Panel>
  );
}

/* ==========================================================================
   Correlation
   ========================================================================== */

function Correlations({ pairs }: { pairs: PairCorrelation[] }) {
  if (!pairs.length) return null;
  const anyUnresolved = pairs.some((pair) => !pair.champion_significant);
  const columns: Array<Column<PairCorrelation>> = [
    {
      key: 'pair',
      header: 'pair',
      width: '30%',
      value: (row) => `${row.a} / ${row.b}`,
      render: (row) => `${row.a} / ${row.b}`,
    },
    {
      key: 'shared',
      header: 'shared',
      help: 'players held in both leagues',
      num: true,
      width: 70,
      value: (row) => row.shared_players?.length ?? 0,
      render: (row) => row.shared_players?.length ?? 0,
    },
    {
      key: 'weekly',
      header: 'weekly',
      help: 'correlation of the two teams’ weekly starting totals',
      num: true,
      width: 78,
      value: (row) => row.weekly,
      render: (row) => fmt.signed(row.weekly, 3),
    },
    {
      key: 'season',
      header: 'season',
      help: 'correlation of season points-for',
      num: true,
      width: 78,
      value: (row) => row.season,
      render: (row) => fmt.signed(row.season, 3),
    },
    {
      key: 'champion',
      header: 'title',
      help: 'phi between the two championship indicators — the one that feeds P(≥1)',
      num: true,
      width: 78,
      value: (row) => row.champion,
      render: (row) => fmt.signed(row.champion, 3),
    },
    {
      key: 'stderr',
      header: '±',
      num: true,
      width: 66,
      value: (row) => row.champion_stderr,
      render: (row) => <span className="faint">{fmt.num(row.champion_stderr, 3)}</span>,
    },
    {
      key: 'sig',
      header: 'title ≠ 0',
      align: 'center',
      width: 78,
      value: (row) => (row.champion_significant ? 1 : 0),
      render: (row) =>
        row.champion_significant ? (
          <span className="up">yes</span>
        ) : (
          <span style={{ color: 'var(--noise)' }}>no</span>
        ),
    },
  ];
  return (
    <Panel
      flush
      title="How the leagues move together"
      right={<span className="caption">measured on the shared draw, not modelled</span>}
    >
      <Table
        compact
        columns={columns}
        rows={pairs}
        rowKey={(row) => `${row.a}:${row.b}`}
        dim={(row) => !row.champion_significant}
        footer={
          anyUnresolved ? (
            <span className="note">
              A “no” is a title correlation inside twice its own Monte Carlo error: it changes sign
              from seed to seed, so read it as zero. The weekly and season figures are stable to the
              third decimal; the championship one is a phi between two indicators that are 1 in
              three to seven per cent of seasons.
            </span>
          ) : undefined
        }
      />
    </Panel>
  );
}
