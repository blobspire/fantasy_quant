/**
 * Start/sit: what is set, what maximises points, what maximises the win.
 *
 * The loudest thing on this page is the leverage, because it is usually the
 * answer. Leverage is the marginal win probability per projected point relative
 * to a coin flip: at 1.00 every point lands, and below 0.25 the game is
 * effectively decided and the whole start/sit decision is worth less than the
 * time spent making it. A dashboard that buries that under a lineup grid is
 * selling busywork, so when the week is decided "this week barely matters" is
 * the headline, in the largest type on the screen.
 *
 * The deltas are measured against the lineup actually set, read live off ESPN.
 * When that read fails `current_known` is false and the page says the comparison
 * is against a hypothetical rather than quietly substituting the optimum —
 * which would show "no change" to the one manager this surface exists for.
 */
import { api, fmt, type LineupChange, type LineupPayload, type LineupSlot } from '../api';
import { Delta } from '../components/Delta';
import { Failure, Loading, useLeagueId, useResource } from '../components/Layout';
import { Stat } from '../components/Stat';
import { Table, type Column } from '../components/Table';
import { Legend, Panel } from './Queue';
import { DECIDED_LEVERAGE } from './Overview';

export default function Lineup() {
  const leagueId = useLeagueId();
  const advice = useResource<LineupPayload>(
    (signal) =>
      leagueId === null
        ? Promise.reject(new Error('no league selected'))
        : api.lineup(leagueId, {}, { signal }),
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
  if (advice.loading && !advice.data) return <Loading what="lineup search" />;
  if (advice.error && !advice.data) return <Failure error={advice.error} onRetry={advice.reload} />;
  if (!advice.data) return null;

  const data = advice.data;

  return (
    <div className="grid">
      {advice.error ? (
        <p className="note note--warn">Showing the last good advice: {advice.error.message}</p>
      ) : null}
      <Headline data={data} />
      <Changes data={data} />
      <Lineups data={data} />
      <Matchup data={data} />
      <Legend />
    </div>
  );
}

/* ==========================================================================
   Does this week matter at all
   ========================================================================== */

function Headline({ data }: { data: LineupPayload }) {
  const decided = data.leverage < DECIDED_LEVERAGE;
  const rec = data.recommendation;
  return (
    <Panel
      title={`${data.name} — week ${data.week}`}
      right={
        <div className="row">
          {data.opponent ? <span className="chip">vs {data.opponent.name}</span> : null}
          <span className="chip" title={`threshold: ${data.threshold_kind}`}>
            {data.threshold_kind === 'opponent' ? 'vs your opponent' : data.threshold_kind}
          </span>
          {data.n_changes ? (
            <span className="chip chip--ok">
              {data.n_changes} change{data.n_changes === 1 ? '' : 's'}
            </span>
          ) : null}
        </div>
      }
    >
      <div className="grid" style={{ gap: 'var(--sp-5)' }}>
        <div style={{ maxWidth: '74ch' }}>
          <div
            className="stat__value"
            style={{ fontSize: 'var(--fs-2xl)', color: decided ? 'var(--noise)' : 'var(--fg)' }}
          >
            {decided ? 'This week barely matters.' : 'This week is live.'}
          </div>
          <p className="prose" style={{ marginTop: 'var(--sp-3)' }}>
            Leverage {fmt.num(data.leverage, 2)} of a coin flip, on a projected margin of{' '}
            {fmt.signed(data.margin, 1)} ± {fmt.num(data.sd_diff, 1)} (z {fmt.signed(data.z, 2)}).{' '}
            {decided
              ? 'The game is close to decided either way, so the whole start/sit call is worth very little — do not spend the morning on it.'
              : 'A projected point moves this game, so the lineup call is worth making.'}
          </p>
        </div>

        <div className="row row--wrap" style={{ gap: 'var(--sp-8)', alignItems: 'flex-end' }}>
          <div className="stat stat--md">
            <div className="stat__label">the whole decision, in title</div>
            <div className="stat__figure">
              <Delta
                size="lg"
                value={rec?.delta_title}
                stderr={rec?.stderr}
                verdict={rec?.verdict}
                significant={data.significant}
                note={rec?.verdict_note}
              />
            </div>
            <div className="stat__sub">
              {rec?.verdict_note ??
                (data.significant ? '' : 'one week of lineup is a small thing')}
            </div>
          </div>
          {/* NOT "what the recommendation buys you". `LineupAdvice` defines
              `delta_win_prob` as the win-probability lineup against the
              EXPECTED-POINTS lineup, which is a different pair from the title
              delta beside it (that one is against the baseline actually set).
              When a guard blocks the variance override the two disagree, so
              the label has to name the comparison. The noise floor is a
              magnitude, not a signed change, so it does not carry a "+". */}
          <Stat
            size="sm"
            label="ΔP(win): win-prob lineup vs points lineup"
            value={data.delta_win_prob}
            format="pp"
            stderr={data.delta_win_prob_stderr}
            sub={`must clear ${fmt.ppError(data.noise_floor)} to be believed — the paired error inflated for having taken the best of ${fmt.int(data.n_lineups)} lineups`}
          />
          <Stat
            size="sm"
            label="points given up"
            value={data.points_sacrifice}
            format="points"
            sub="win-probability lineup vs points lineup"
          />
          <Stat
            size="sm"
            label="leverage"
            value={data.leverage}
            format="number"
            digits={2}
          />
        </div>

        {data.current_known ? null : (
          <p className="note note--warn">
            Your live lineup could not be read, so everything here is measured against the
            projected-best lineup rather than against what you actually have set.
          </p>
        )}
        {data.unpriced_current?.length ? (
          <p className="note note--bad">
            {data.unpriced_current.length} starter(s) you have set cannot be legally assigned to
            this week’s slots and were not priced — the lineup is broken, and the deltas above are
            against the projected-best lineup.
          </p>
        ) : null}
        {data.guard ? (
          <p className="note">
            The win-probability lineup was <em>not</em> adopted: {data.guard}
          </p>
        ) : null}
      </div>
    </Panel>
  );
}

/* ==========================================================================
   Changes
   ========================================================================== */

function Changes({ data }: { data: LineupPayload }) {
  const changes = data.changes ?? [];
  if (!changes.length) {
    return (
      <Panel title="Changes">
        <div className="state">
          <div className="state__title">Your lineup is already the one to start.</div>
          <p className="prose">
            No swap improves either the expected points or the win probability. Nothing to do before
            kickoff.
          </p>
        </div>
      </Panel>
    );
  }
  const columns: Array<Column<LineupChange>> = [
    { key: 'slot', header: 'slot', width: 72, value: (row) => row.slot },
    {
      key: 'in',
      header: 'start',
      width: '30%',
      value: (row) => row.in,
      render: (row) => <span className="up">{row.in}</span>,
    },
    {
      key: 'out',
      header: 'over',
      width: '30%',
      value: (row) => row.out,
      render: (row) => <span className="faint">{row.out}</span>,
    },
    {
      key: 'dmean',
      header: 'Δmean',
      help: 'projected points gained by the swap',
      num: true,
      width: 82,
      defaultDesc: true,
      value: (row) => row.d_mean,
      render: (row) => fmt.signed(row.d_mean, 1),
    },
    {
      key: 'dsd',
      header: 'Δsd',
      help: 'change in the spread of your weekly total — the variance is the point near a cut line',
      num: true,
      width: 74,
      value: (row) => row.d_sd,
      render: (row) => fmt.signed(row.d_sd, 1),
    },
  ];
  return (
    <Panel flush title="Change these">
      <Table columns={columns} rows={changes} rowKey={(row, index) => `${row.slot_id}:${index}`} />
    </Panel>
  );
}

/* ==========================================================================
   The lineups
   ========================================================================== */

interface LineupComparison {
  index: number;
  slot: string;
  current: LineupSlot | undefined;
  recommended: LineupSlot | undefined;
  differs: boolean;
}

function Lineups({ data }: { data: LineupPayload }) {
  const baseline = data.baseline ?? [];
  const recommended = data.recommended ?? [];
  const length = Math.max(baseline.length, recommended.length);
  if (!length) {
    return (
      <Panel title="Lineups">
        <div className="state">
          <div className="state__title">No lineup in this payload.</div>
        </div>
      </Panel>
    );
  }

  const rows: LineupComparison[] = [];
  for (let index = 0; index < length; index += 1) {
    const current = baseline[index];
    const better = recommended[index];
    rows.push({
      index,
      slot: better?.slot ?? current?.slot ?? fmt.dash,
      current,
      recommended: better,
      differs: current?.player_id !== better?.player_id,
    });
  }

  const currentLabel = data.current_known ? 'currently set' : 'projected best (live lineup unread)';
  const columns: Array<Column<LineupComparison>> = [
    { key: 'slot', header: 'slot', width: 72, sortable: false, render: (row) => row.slot },
    {
      key: 'current',
      header: currentLabel,
      width: '38%',
      sortable: false,
      render: (row) =>
        row.current ? (
          <span className={row.differs ? 'down' : undefined}>{row.current.name}</span>
        ) : (
          <span className="faint">empty</span>
        ),
    },
    {
      key: 'recommended',
      header: 'recommended',
      width: '38%',
      sortable: false,
      render: (row) =>
        row.recommended ? (
          <span className={row.differs ? 'up' : undefined}>{row.recommended.name}</span>
        ) : (
          <span className="faint">empty</span>
        ),
    },
  ];

  return (
    <Panel
      flush
      title="Lineups"
      right={
        <div className="row row--wrap">
          <Stat
            size="xs"
            align="right"
            label="expected-points lineup"
            value={data.points_lineup_mean}
            format="points"
          />
          <Stat
            size="xs"
            align="right"
            label="win-probability lineup"
            value={data.win_prob_lineup_mean}
            format="points"
          />
          <span className="caption">
            {data.differ
              ? 'the two disagree — the recommended column is the one adopted'
              : 'the two agree this week'}
          </span>
        </div>
      }
    >
      <Table
        compact
        columns={columns}
        rows={rows}
        rowKey={(row) => row.index}
        dim={(row) => !row.differs}
        footer={
          <span className="caption">
            {data.differ ? (
              <>
                Only the adopted lineup&rsquo;s roster is in this payload; the other option is
                represented by its projected total ({fmt.num(data.points_lineup_mean, 1)} points
                against {fmt.num(data.win_prob_lineup_mean, 1)}). Swapping to the win-probability
                lineup buys {fmt.pp(data.delta_win_prob)} of this game for{' '}
                {fmt.num(data.points_sacrifice, 1)} projected points.
              </>
            ) : (
              <>
                The expected-points lineup and the win-probability lineup are the same eleven this
                week, so there is no second roster to show and nothing is being traded off.
              </>
            )}
          </span>
        }
      />
    </Panel>
  );
}

/* ==========================================================================
   The matchup
   ========================================================================== */

function Matchup({ data }: { data: LineupPayload }) {
  const stacks = data.stacks ?? [];
  return (
    <Panel title="The matchup">
      <div className="grid" style={{ gap: 'var(--sp-5)' }}>
        <div className="row row--wrap" style={{ gap: 'var(--sp-7)', alignItems: 'flex-end' }}>
          <Stat
            size="sm"
            label={data.opponent ? `${data.opponent.name} projects` : 'opponent projects'}
            value={data.opponent?.mean ?? null}
            format="points"
            sub={data.opponent ? `± ${fmt.num(data.opponent.sd, 1)}` : undefined}
          />
          <Stat
            size="sm"
            label="margin"
            value={fmt.signed(data.margin, 1)}
            format="raw"
            sub={`± ${fmt.num(data.sd_diff, 1)} · ${fmt.num(data.sd_independent, 1)} with no covariance`}
          />
          <Stat size="sm" label="z" value={fmt.signed(data.z, 2)} format="raw" />
          <Stat size="sm" label="lineups searched" value={data.n_lineups} format="int" />
        </div>
        {stacks.length ? (
          <div>
            <div className="label">correlated with your opponent</div>
            <ul className="caption" style={{ margin: 'var(--sp-2) 0 0' }}>
              {stacks.map((stack, index) => (
                <li key={index}>
                  {stack.name} vs {stack.opponent_name}: ρ {fmt.num(stack.rho, 2)}
                  {stack.modelled ? '' : ' (not modelled in this draw)'}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </Panel>
  );
}
