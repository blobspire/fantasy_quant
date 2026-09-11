/**
 * The waiver board, and the priority threshold a claim has to clear.
 *
 * The threshold IS the screen, so it is the biggest number on it. All three of
 * these leagues run rolling waiver priority, not FAAB: a claim costs a queue
 * position rather than money, so the question is never "is this player good"
 * but "is he better than the option of claiming somebody later".
 * `decide/waivers.py` answers that by backward induction and publishes the
 * continuation value as `threshold`. No bid amount is rendered anywhere on this
 * page, and `WaiversPayload.budget` is never read.
 *
 * The second thing this screen has to say is that clearing the threshold is
 * itself uncertain. On the live boards the marginal candidate clears by less
 * than its own standard error — Blacksburg's fifth claim by +0.003pp against
 * ±0.024pp — so `clears_certain` gets its own column and a row that clears by a
 * coin flip is not allowed to look like a row that clears.
 */
import { api, fmt, type WaiverRow, type WaiversPayload } from '../api';
import { RankingsNote } from '../components/RankingsNote';
import { DeltaCell, ErrorBar } from '../components/Delta';
import { Failure, Loading, useLeagueId, useResource } from '../components/Layout';
import { Stat } from '../components/Stat';
import { Table, type Column } from '../components/Table';
import { Legend, Panel } from './Queue';

export default function Waivers() {
  const leagueId = useLeagueId();
  const board = useResource<WaiversPayload>(
    (signal) =>
      leagueId === null
        ? Promise.reject(new Error('no league selected'))
        : api.waivers(leagueId, { limit: 12 }, { signal }),
    [leagueId],
  );

  if (leagueId === null) {
    return (
      <div className="state">
        <div className="state__title">No league selected</div>
        <p className="prose">Pick one in the switcher above.</p>
      </div>
    );
  }
  if (board.loading && !board.data) return <Loading what="waiver board" />;
  if (board.error && !board.data) return <Failure error={board.error} onRetry={board.reload} />;
  if (!board.data) return null;

  const data = board.data;

  return (
    <div className="grid">
      <header className="row row--wrap" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1 className="panel__title" style={{ fontSize: 'var(--fs-lg)' }}>
            Waiver board — {data.name}
          </h1>
          <div className="caption">
            Rolling waiver priority: a claim costs your place in the queue, never money.
          </div>
        </div>
        {board.error ? (
          <p className="note note--warn">Showing the last good board: {board.error.message}</p>
        ) : null}
      </header>

      <Rule data={data} />
      {data.rankings ? <RankingsNote rankings={data.rankings} surface="waivers" /> : null}
      <Claims data={data} />
      <Board data={data} />
      {data.blocks?.length ? <Blocks rows={data.blocks} /> : null}
      <Legend />
    </div>
  );
}

/**
 * The decision rule, at the size of the decision it makes.
 *
 * `threshold` is the continuation value of KEEPING the priority: the title
 * probability you expect to buy with it in a later week. A claim below it is a
 * real measured gain that you should still not make.
 */
function Rule({ data }: { data: WaiversPayload }) {
  const claims = data.claims ?? [];
  return (
    <Panel
      title="The rule"
      right={
        <div className="row">
          <span className="chip">week {data.week}</span>
          {data.uses_faab ? (
            <span className="chip chip--warn">FAAB reported</span>
          ) : (
            <span
              className="chip"
              title={data.priority_known ? undefined : 'the waiver order could not be read'}
            >
              priority {data.priority ?? '?'}
              {data.priority_known ? '' : ' (assumed)'}
            </span>
          )}
          <span className="chip">{data.n_free_agents} free agents priced</span>
        </div>
      }
    >
      <div className="grid" style={{ gap: 'var(--sp-5)' }}>
        <div className="row row--wrap" style={{ gap: 'var(--sp-8)', alignItems: 'flex-end' }}>
          <Stat
            size="lg"
            label="a claim must clear"
            value={data.threshold}
            format="pp"
            digits={3}
            sub="what your waiver priority is worth if you keep it"
          />
          <Stat
            size="sm"
            label="claims that clear"
            value={String(claims.length)}
            format="raw"
            sub={claims.length ? 'submit all of them, in order' : 'hold your priority'}
          />
          <Stat
            size="sm"
            label="baseline P(title)"
            value={data.baseline_title}
            format="pct"
            digits={2}
            sub="this board streams a replacement into an empty slot"
          />
          <Stat
            size="sm"
            label="title per point"
            value={data.title_per_point}
            format="pp"
            digits={3}
            stderr={data.title_per_point_stderr}
            sub="rest of season"
          />
          <Stat
            size="sm"
            label="this week’s leverage"
            value={data.week_leverage}
            format="number"
            digits={2}
            sub={`sd of the margin ${fmt.num(data.sd_diff, 1)}`}
          />
        </div>

        {data.uses_faab ? (
          <p className="note note--warn">
            The API reports this league as FAAB. Everything here is priced in championship
            probability and no bid is shown — the registry has all three leagues on rolling
            priority, so check the league settings before treating this as a budget decision.
          </p>
        ) : null}
        {data.priority_known ? null : (
          <p className="note note--warn">
            The waiver order could not be read, so the priority above is an assumption and the
            threshold is only as good as it.
          </p>
        )}
      </div>
    </Panel>
  );
}

/**
 * Digits on this page, and why they are not the two the rest of the app uses.
 *
 * `report.render_waivers` prints three: `+0.151pp` with `0.014pp` of error. The
 * whole board lives inside a tenth of a percentage point -- twelve rows from
 * +0.151pp down to +0.040pp, with standard errors from 0.008pp to 0.015pp -- so
 * at two decimals every error on the page rounds to the same `±0.01pp` and rows
 * that the CLI separates print identically. A dashboard whose numbers agree
 * with `fq waivers` only to the digit before the one that decides the claim is
 * not agreeing with it.
 */
const WAIVER_DIGITS = 3;

/** The claim columns, shared by the waterfall and the full board. */
function claimColumns(withRank: boolean): Array<Column<WaiverRow>> {
  const columns: Array<Column<WaiverRow>> = [
    {
      key: 'add',
      header: 'add',
      width: '24%',
      value: (row) => row.add,
      // The analyst's note rides under the name when the board has one. It is the
      // one thing on this row that is not a number, and the one thing a human wrote.
      render: (row) =>
        row.note && row.note !== '-' ? (
          <span>
            {row.add}
            <span className="caption" style={{ display: 'block' }} title={row.note}>
              {row.note}
            </span>
          </span>
        ) : (
          row.add
        ),
    },
    { key: 'pos', header: 'pos', width: 54, value: (row) => row.position },
    {
      key: 'board',
      header: 'board',
      help: "the analyst's positional rank of the add over the drop, read straight off the board -- not derived from ΔP(title)",
      align: 'center',
      width: 74,
      value: (row) => (row.board && row.board !== '-' ? row.board : ''),
      render: (row) =>
        row.board && row.board !== '-' ? (
          <span className="up mono">{row.board}</span>
        ) : (
          <span className="faint">—</span>
        ),
    },
    {
      key: 'drop',
      header: 'drop',
      width: '20%',
      value: (row) => row.drop,
      render: (row) =>
        row.drop && row.drop !== '-' ? row.drop : <span className="faint">nobody</span>,
    },
    {
      key: 'delta',
      header: 'ΔP(title)',
      num: true,
      width: 100,
      defaultDesc: true,
      value: (row) => row.delta_title,
      render: (row) => (
        <DeltaCell
          value={row.delta_title}
          verdict={row.verdict}
          significant={row.significant}
          note={row.verdict_note}
          digits={WAIVER_DIGITS}
        />
      ),
    },
    {
      key: 'stderr',
      header: '±',
      num: true,
      width: 76,
      value: (row) => row.stderr,
      render: (row) => <ErrorBar stderr={row.stderr} digits={WAIVER_DIGITS} />,
    },
    {
      key: 'cost',
      header: 'cost',
      help: 'a plain free agent is first come and spends no waiver priority at all',
      align: 'center',
      width: 76,
      value: (row) => (row.on_waivers ? 1 : 0),
      render: (row) =>
        row.on_waivers ? (
          <span className="faint">claim</span>
        ) : (
          <span style={{ color: 'var(--up)' }} title="first come, no priority spent">
            free
          </span>
        ),
    },
    {
      key: 'margin',
      header: 'over the bar',
      help: 'delta_title minus the priority threshold; n/a for a player who costs nothing',
      num: true,
      width: 96,
      value: (row) => (row.on_waivers ? row.clears_margin : Number.POSITIVE_INFINITY),
      render: (row) =>
        row.on_waivers ? (
          fmt.pp(row.clears_margin, WAIVER_DIGITS)
        ) : (
          <span className="faint">n/a</span>
        ),
    },
    {
      key: 'resolved',
      header: 'resolved',
      help: 'is that margin larger than twice this row’s own error?',
      align: 'center',
      width: 84,
      value: (row) => (row.clears_certain || !row.on_waivers ? 1 : 0),
      render: (row) =>
        !row.on_waivers ? (
          // Nothing to resolve: there is no threshold for a free agent to be near.
          <span className="faint">n/a</span>
        ) : row.clears_certain ? (
          <span className="faint">yes</span>
        ) : (
          <span
            style={{ color: 'var(--noise)' }}
            title="the margin over the threshold is inside this row’s own error"
          >
            coin flip
          </span>
        ),
    },
  ];
  if (!withRank) return columns;
  return [
    {
      key: 'order',
      header: '#',
      num: true,
      width: 38,
      sortable: false,
      render: (_row, index) => index + 1,
    },
    ...columns,
  ];
}

function Claims({ data }: { data: WaiversPayload }) {
  const claims = data.claims ?? [];
  if (!claims.length) {
    const best = (data.board ?? [])[0];
    return (
      <Panel title="Claims">
        <div className="state">
          <div className="state__title">Nothing on the wire clears the threshold.</div>
          <p className="prose">
            {best
              ? `The best claim available (${best.add}) is worth ${fmt.pp(
                  best.delta_title,
                  WAIVER_DIGITS,
                )} against a ${fmt.pp(data.threshold, WAIVER_DIGITS)} cost of spending your ` +
                'priority. '
              : ''}
            {/* Never assert "hold" over a board the payload contradicts: if the
                surface returned no claims but its own best row is still marked
                as clearing, say that rather than talking past it. */}
            {best?.clears_threshold
              ? 'That row is marked as clearing, yet the surface returned no claim — read the ' +
                'board below rather than this sentence, and treat the count as unresolved.'
              : 'Holding is the answer: the priority is worth more kept than spent.'}
          </p>
          {data.hold?.rationale ? <p className="prose">{data.hold.rationale}</p> : null}
        </div>
      </Panel>
    );
  }
  return (
    <Panel
      flush
      title="Submit these, in this order"
      right={<span className="chip chip--ok">{claims.length} clear the threshold</span>}
    >
      <Table
        columns={claimColumns(true)}
        rows={claims}
        rowKey={(row, index) => `${row.add}:${index}`}
        dim={(row) => !row.significant}
        footer={data.waterfall_note ? <span className="note">{data.waterfall_note}</span> : undefined}
      />
    </Panel>
  );
}

function Board({ data }: { data: WaiversPayload }) {
  const rows = data.board ?? [];
  if (!rows.length) {
    return (
      <Panel title="The board">
        <div className="state">
          <div className="state__title">Nothing was priced.</div>
          <p className="prose">
            No free agent beat the wire floor for any of your roster spots this week.
          </p>
        </div>
      </Panel>
    );
  }
  const columns: Array<Column<WaiverRow>> = [
    ...claimColumns(false),
    {
      key: 'clears',
      header: 'clears',
      align: 'center',
      width: 68,
      value: (row) => (row.clears_threshold ? 1 : 0),
      render: (row) =>
        row.clears_threshold ? (
          row.clears_certain ? (
            <span className="up">yes</span>
          ) : (
            <span style={{ color: 'var(--noise)' }} title="inside its own error">
              ~yes
            </span>
          )
        ) : (
          <span className="faint">no</span>
        ),
    },
    {
      key: 'bracket',
      header: 'bracket',
      help: 'the same claim priced through the bracket rather than the points rate',
      num: true,
      width: 86,
      value: (row) => row.bracket_title,
      render: (row) => <span className="faint">{fmt.pp(row.bracket_title, WAIVER_DIGITS)}</span>,
    },
    {
      key: 'agrees',
      header: 'agrees',
      help: 'do the two pricings agree on the sign?',
      align: 'center',
      width: 68,
      value: (row) => (row.agrees ? 1 : 0),
      render: (row) => <span className="faint">{row.agrees ? 'yes' : 'no'}</span>,
    },
  ];
  return (
    <Panel flush title="Everything priced" right={<span className="caption">{data.baseline}</span>}>
      <Table
        compact
        columns={columns}
        rows={rows}
        rowKey={(row, index) => `${row.add}:${index}`}
        dim={(row) => !row.significant || !row.clears_threshold}
        maxHeight={560}
      />
    </Panel>
  );
}

/**
 * Denying a rival the player he wants most.
 *
 * Already discounted by 1/(N-1) server-side, and never a reason to claim on its
 * own — the surface says so and so does this panel.
 */
function Blocks({ rows }: { rows: WaiverRow[] }) {
  const columns: Array<Column<WaiverRow>> = [
    { key: 'add', header: 'add', width: '22%', value: (row) => row.add },
    { key: 'pos', header: 'pos', width: 54, value: (row) => row.position },
    {
      key: 'delta',
      header: 'ΔP(title)',
      num: true,
      width: 100,
      value: (row) => row.delta_title,
      render: (row) => (
        <DeltaCell
          value={row.delta_title}
          verdict={row.verdict}
          significant={row.significant}
          digits={WAIVER_DIGITS}
        />
      ),
    },
    {
      key: 'stderr',
      header: '±',
      num: true,
      width: 76,
      value: (row) => row.stderr,
      render: (row) => <ErrorBar stderr={row.stderr} digits={WAIVER_DIGITS} />,
    },
    {
      key: 'why',
      header: 'why',
      sortable: false,
      render: (row) => (
        <div className="caption" style={{ whiteSpace: 'normal' }}>
          {row.rationale}
        </div>
      ),
    },
  ];
  return (
    <Panel
      flush
      title="Denying a rival"
      right={
        <span className="caption">
          discounted by 1/(N−1) already — never claim for this reason alone
        </span>
      }
    >
      <Table
        compact
        columns={columns}
        rows={rows}
        rowKey={(row, index) => `block:${index}`}
        dim={() => true}
      />
    </Panel>
  );
}
