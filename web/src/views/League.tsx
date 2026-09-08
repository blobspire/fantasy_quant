/**
 * One league in depth: the championship table, the roster's weekly outlooks,
 * and what is left of the schedule.
 *
 * The championship table is the first thing anyone reads and the number most
 * easily over-read: at 4,000 simulations a 13% favourite carries ±0.53pp, and
 * most adjacent pairs on a live 14-team board are not separated by more than
 * twice the paired Monte Carlo error. `report.odds_payload` measures that per
 * row, so every row shows its error and an unseparated boundary renders as the
 * tie it is rather than as an ordering.
 *
 * One request carries most of this page: `weekly` with the three expensive
 * decision surfaces skipped, which is also the only place the leverage schedule
 * exists (there is no `/leverage` route). Each section fails independently and
 * says so in place, so a dead surface costs one panel rather than the page. The
 * roster is its own endpoint and its own request.
 */
import {
  api,
  fmt,
  sectionData,
  sectionError,
  type LeveragePayload,
  type OddsPayload,
  type TeamOdds,
  type WeeklyPayload,
} from '../api';
import { Failure, Loading, SectionFailure, useLeagueId, useResource } from '../components/Layout';
import { Stat } from '../components/Stat';
import { Table, type Column } from '../components/Table';
import { Panel } from './Queue';
import { DECIDED_LEVERAGE } from './Overview';

/** Sections this page does not need. Each one costs seconds server-side. */
const LEAGUE_SKIP = ['lineup', 'waivers', 'trades', 'stream'];

/* ==========================================================================
   Roster
   ==========================================================================
   `/api/leagues/{id}/roster` is the one payload with no `report.py` builder, so
   `api.ts` types it as an open record and the shapes below name what the server
   actually sends (`api/server.roster_payload`). Every number is read: the
   weekly moments come off `core.WeeklyOutlook` and the value columns off
   `decide/valuation.value_league`. Nothing is projected in the browser.
   ========================================================================== */

interface OutlookWeek {
  week: number;
  mean: number;
  sd: number;
  p_zero: number;
  playing: boolean;
}

interface RosterPlayer {
  player_id: number;
  name: string;
  position: string;
  lineup_slot: string;
  starting: boolean;
  injury_status?: string;
  injured?: boolean | null;
  week?: number;
  week_mean?: number | null;
  week_sd?: number | null;
  week_playing?: boolean | null;
  ros_points?: number | null;
  ros_vorp?: number | null;
  ros_vorp_per_week?: number | null;
  playoff_vorp?: number | null;
  outlook?: OutlookWeek[];
  projected?: boolean;
}

interface ReplacementLevel {
  position: string;
  rostered_rank?: number;
  points_per_week?: number;
  supply_limited?: boolean;
}

interface RosterPayload {
  week?: number;
  current_known?: boolean;
  n_players?: number;
  n_starting?: number;
  values_ok?: boolean;
  values_error?: string;
  replacement?: ReplacementLevel[];
  players?: RosterPlayer[];
}

/** Healthy is `NORMAL` on ESPN's wire, not `ACTIVE`. Anything else is worth a flag. */
const HEALTHY = new Set(['', 'ACTIVE', 'NORMAL']);

function isHurt(status: string | undefined): boolean {
  return Boolean(status) && !HEALTHY.has(status!.toUpperCase());
}

/**
 * `report.slot_label` names a slot after what it accepts, and a bench seat
 * accepts nothing, so it arrives as the bare id. `edges/portfolio.SLOT_ABBREV`
 * is the project's own name for those two.
 */
const NUMERIC_SLOT: Record<string, string> = { '20': 'BE', '21': 'IR' };

/** The roster payload, or an empty one. The endpoint is typed as an open record. */
function asRoster(body: Record<string, unknown> | null): RosterPayload {
  return (body ?? {}) as RosterPayload;
}

/* ==========================================================================
   View
   ========================================================================== */

export default function League() {
  const leagueId = useLeagueId();
  // A null league is a state, not a request: the shell has not resolved the
  // switcher yet, and `/api/leagues/null/weekly` is not a question to ask.
  const weekly = useResource<WeeklyPayload>(
    (signal) =>
      leagueId === null
        ? Promise.reject(new Error('no league selected'))
        : api.weekly(leagueId, { skip: LEAGUE_SKIP }, { signal }),
    [leagueId],
  );
  const roster = useResource<Record<string, unknown>>(
    (signal) =>
      leagueId === null
        ? Promise.reject(new Error('no league selected'))
        : api.roster(leagueId, {}, { signal }),
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
  if (weekly.loading && !weekly.data) return <Loading what="championship simulation" />;
  if (weekly.error && !weekly.data) return <Failure error={weekly.error} onRetry={weekly.reload} />;

  const payload = weekly.data;
  const odds = sectionData<OddsPayload>(payload?.odds);
  const leverage = sectionData<LeveragePayload>(payload?.leverage);
  const oddsError = sectionError(payload?.odds);

  return (
    <div className="grid">
      {odds ? <Header odds={odds} /> : null}
      {weekly.error ? (
        <p className="note note--warn">Showing the last good table: {weekly.error.message}</p>
      ) : null}
      {payload && payload.ok === false ? (
        <p className="note note--bad">
          This league did not build: {payload.error ?? 'unknown error'}
        </p>
      ) : null}
      {oddsError ? <SectionFailure name="The championship table" error={oddsError} /> : null}

      {odds ? <ChampionshipTable odds={odds} /> : null}

      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(520px, 1fr))' }}>
        <RosterPanel
          data={asRoster(roster.data)}
          loading={roster.loading}
          error={roster.error?.message ?? null}
        />
        <SchedulePanel
          data={leverage}
          loading={weekly.loading}
          error={sectionError(payload?.leverage)}
        />
      </div>
    </div>
  );
}

function Header({ odds }: { odds: OddsPayload }) {
  const me = odds.teams.find((team) => team.is_me) ?? null;
  return (
    <header className="row row--wrap" style={{ justifyContent: 'space-between' }}>
      <div>
        <h1 className="panel__title" style={{ fontSize: 'var(--fs-lg)' }}>
          {odds.name}
        </h1>
        <div className="caption">
          {odds.size} teams
          {odds.week ? ` · week ${odds.week}` : ''}
          {` · ${fmt.int(odds.n_sims)} simulations`}
          {odds.team_name ? ` · you are ${odds.team_name}` : ''}
        </div>
      </div>
      <div className="row row--wrap" style={{ gap: 'var(--sp-7)' }}>
        <Stat
          size="md"
          align="right"
          label="P(championship)"
          value={odds.my_championship}
          format="pct"
          digits={2}
          stderr={odds.my_championship_stderr}
        />
        <Stat size="sm" align="right" label="P(playoffs)" value={odds.my_playoffs} format="pct" />
        <Stat
          size="sm"
          align="right"
          label="rank"
          value={odds.my_rank === null ? null : `${odds.my_rank} of ${odds.size}`}
          format="raw"
          note={odds.ranking_note}
          sub={me ? fmt.record(me.wins, me.losses, me.ties) : undefined}
        />
      </div>
    </header>
  );
}

/* ==========================================================================
   Championship table
   ========================================================================== */

function ChampionshipTable({ odds }: { odds: OddsPayload }) {
  // `report.render_odds` colours its ranking note when NO adjacent pair on the
  // board separates at all. Same rule here rather than a second one.
  const nothingSeparates = odds.n_ranks > 0 && odds.n_ranks_separated === 0;
  const columns: Array<Column<TeamOdds>> = [
    { key: 'rank', header: '#', num: true, width: 38, value: (row) => row.rank },
    {
      key: 'name',
      header: 'team',
      width: '26%',
      value: (row) => row.name,
      render: (row) => (
        <>
          {row.name}
          {row.is_me ? <span className="faint"> ← you</span> : null}
        </>
      ),
    },
    {
      key: 'record',
      header: 'record',
      num: true,
      width: 78,
      value: (row) => row.wins,
      render: (row) => fmt.record(row.wins, row.losses, row.ties),
    },
    {
      key: 'pf',
      header: 'PF',
      help: 'points for, from ESPN’s own standings',
      num: true,
      width: 74,
      value: (row) => row.points_for,
      render: (row) => (row.points_for === null ? fmt.dash : fmt.num(row.points_for, 0)),
    },
    {
      key: 'title',
      header: 'P(title)',
      num: true,
      width: 82,
      defaultDesc: true,
      value: (row) => row.championship,
      render: (row) => fmt.pct(row.championship, 2),
    },
    {
      key: 'stderr',
      header: '±',
      help: 'sqrt(p(1−p)/n): the error on this row’s own probability',
      num: true,
      width: 66,
      value: (row) => row.championship_stderr,
      render: (row) => <span className="faint">{fmt.ppError(row.championship_stderr, 2)}</span>,
    },
    {
      key: 'separated',
      header: 'vs next',
      help: 'is this row actually above the one below, on this draw?',
      align: 'center',
      width: 76,
      sortable: false,
      render: (row) =>
        row.separated_from_next === null || row.separated_from_next === undefined ? (
          <span className="faint">{fmt.dash}</span>
        ) : row.separated_from_next ? (
          <span className="faint">yes</span>
        ) : (
          <span style={{ color: 'var(--noise)' }} title="inside the paired Monte Carlo error">
            tied
          </span>
        ),
    },
    {
      key: 'playoffs',
      header: 'P(playoffs)',
      num: true,
      width: 92,
      value: (row) => row.playoffs,
      render: (row) => fmt.pct(row.playoffs, 1),
    },
    {
      key: 'bye',
      header: 'P(bye)',
      num: true,
      width: 76,
      value: (row) => row.bye,
      render: (row) => fmt.pct(row.bye, 1),
    },
    {
      key: 'wins',
      header: 'E[wins]',
      num: true,
      width: 78,
      value: (row) => row.expected_wins,
      render: (row) => fmt.num(row.expected_wins, 1),
    },
  ];
  return (
    <Panel
      flush
      title="Championship table"
      right={
        <div className="row">
          <span className={nothingSeparates ? 'chip chip--warn' : 'chip'}>
            {odds.n_ranks_separated}/{odds.n_ranks} ranks separated
          </span>
          <span className="caption">{odds.baseline}</span>
        </div>
      }
    >
      <Table
        columns={columns}
        rows={odds.teams}
        rowKey={(row) => row.team_id}
        highlight={(row) => row.is_me}
        footer={
          odds.ranking_note ? (
            <span className={nothingSeparates ? 'note note--warn' : 'note'}>
              {odds.ranking_note}
            </span>
          ) : undefined
        }
      />
    </Panel>
  );
}

/* ==========================================================================
   Roster
   ========================================================================== */

function RosterPanel({
  data,
  loading,
  error,
}: {
  data: RosterPayload;
  loading: boolean;
  error: string | null;
}) {
  const players = data.players ?? [];

  if (loading && !players.length) {
    return (
      <Panel title="Roster">
        <Loading what="roster outlooks" />
      </Panel>
    );
  }
  if (!players.length) {
    return (
      <Panel title="Roster">
        <div className="state state--error">
          <div className="state__title">The roster did not load</div>
          <p className="prose">
            {error ?? 'The endpoint answered with no players.'} Every projection on this page comes
            from the API; nothing is computed in the browser to fill the gap.
          </p>
        </div>
      </Panel>
    );
  }

  // Which upcoming weeks get a column: whatever the payload carries, capped so
  // the table stays readable beside the schedule.
  const weeks: number[] = [];
  for (const player of players) {
    for (const entry of player.outlook ?? []) {
      if (!weeks.includes(entry.week)) weeks.push(entry.week);
    }
  }
  weeks.sort((a, b) => a - b);
  const shownWeeks = weeks.filter((week) => week !== data.week).slice(0, 5);
  const replacement = data.replacement ?? [];

  const columns: Array<Column<RosterPlayer>> = [
    {
      key: 'name',
      header: 'player',
      width: '26%',
      value: (row) => row.name,
      render: (row) => (
        <>
          {row.name}
          {isHurt(row.injury_status) ? (
            <span style={{ color: 'var(--warn)' }} title="ESPN injury status">
              {' '}
              {row.injury_status?.toLowerCase().replace(/_/g, ' ')}
            </span>
          ) : null}
        </>
      ),
    },
    { key: 'pos', header: 'pos', width: 52, value: (row) => row.position },
    {
      key: 'slot',
      header: 'slot',
      help: 'the slot he is in right now, off ESPN',
      width: 60,
      value: (row) => row.lineup_slot,
      render: (row) => (
        <span className={row.starting ? undefined : 'faint'}>
          {NUMERIC_SLOT[row.lineup_slot] ?? row.lineup_slot}
        </span>
      ),
    },
    {
      key: 'week',
      header: data.week ? `w${data.week}` : 'this week',
      help: 'calibrated mean ± sd of the full distribution, the mass at zero included',
      num: true,
      width: 108,
      defaultDesc: true,
      value: (row) => row.week_mean,
      render: (row) =>
        row.week_playing === false ? (
          <span className="faint">bye</span>
        ) : (
          <>
            {fmt.num(row.week_mean, 1)}
            {typeof row.week_sd === 'number' ? (
              <span className="faint"> ±{fmt.num(row.week_sd, 1)}</span>
            ) : null}
          </>
        ),
    },
    {
      key: 'ros',
      header: 'RoS',
      help: 'rest-of-season projected points',
      num: true,
      width: 66,
      value: (row) => row.ros_points,
      render: (row) => fmt.num(row.ros_points, 0),
    },
    {
      key: 'vorp',
      header: 'VORP',
      help: 'rest-of-season points over the replacement level at his position',
      num: true,
      width: 72,
      value: (row) => row.ros_vorp,
      render: (row) => fmt.signed(row.ros_vorp, 0),
    },
    {
      key: 'playoff',
      header: 'playoff VORP',
      help: 'the same, over the playoff weeks only',
      num: true,
      width: 104,
      value: (row) => row.playoff_vorp,
      render: (row) => <span className="faint">{fmt.signed(row.playoff_vorp, 0)}</span>,
    },
    ...shownWeeks.map<Column<RosterPlayer>>((week) => ({
      key: `w${week}`,
      header: `w${week}`,
      num: true,
      width: 54,
      value: (row) => row.outlook?.find((entry) => entry.week === week)?.mean,
      render: (row) => {
        const cell = row.outlook?.find((entry) => entry.week === week);
        if (!cell) return <span className="faint">{fmt.dash}</span>;
        if (!cell.playing) return <span className="faint">bye</span>;
        return <span title={`± ${fmt.num(cell.sd, 1)}`}>{fmt.num(cell.mean, 1)}</span>;
      },
    })),
  ];

  return (
    <Panel
      flush
      title="Roster"
      right={
        <span className="caption">
          {data.n_starting ?? 0} starting of {data.n_players ?? players.length} · weekly mean ± sd
        </span>
      }
    >
      <Table
        compact
        columns={columns}
        rows={players}
        rowKey={(row) => row.player_id}
        highlight={(row) => row.starting}
        dim={(row) => row.projected === false}
        maxHeight={540}
        footer={
          <div className="grid" style={{ gap: 'var(--sp-2)' }}>
            {data.current_known === false ? (
              <span className="note note--warn">
                Your live lineup could not be read, so no player is marked as starting.
              </span>
            ) : null}
            {data.values_ok === false ? (
              <span className="note note--warn">
                The valuation failed, so the VORP columns are empty: {data.values_error}
              </span>
            ) : null}
            {replacement.length ? (
              <span className="caption">
                replacement level, points per week —{' '}
                {replacement
                  .map(
                    (level) =>
                      `${level.position} ${fmt.num(level.points_per_week, 1)}${
                        level.supply_limited ? ' (supply-limited)' : ''
                      }`,
                  )
                  .join(' · ')}
              </span>
            ) : null}
          </div>
        }
      />
    </Panel>
  );
}

/* ==========================================================================
   Schedule
   ========================================================================== */

function SchedulePanel({
  data,
  loading,
  error,
}: {
  data: LeveragePayload | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading && !data) {
    return (
      <Panel title="Remaining schedule">
        <Loading what="leverage schedule" />
      </Panel>
    );
  }
  if (!data) {
    return (
      <Panel title="Remaining schedule">
        <div className="state state--error">
          <div className="state__title">The schedule did not load</div>
          <div className="prose">{error ?? 'the weekly payload carried no leverage section'}</div>
        </div>
      </Panel>
    );
  }
  const columns: Array<Column<LeveragePayload['weeks'][number]>> = [
    { key: 'week', header: 'wk', num: true, width: 42, value: (row) => row.week },
    {
      key: 'opponent',
      header: 'opponent',
      width: '30%',
      value: (row) => row.opponent,
      render: (row) => (
        <>
          {row.opponent}
          {row.decided ? (
            <span
              className="faint"
              title="a marginal point is worth under a quarter of its coin-flip value"
            >
              {' '}
              · decided
            </span>
          ) : null}
        </>
      ),
    },
    {
      key: 'margin',
      header: 'margin',
      num: true,
      width: 78,
      value: (row) => row.margin,
      render: (row) => fmt.signed(row.margin, 1),
    },
    {
      key: 'sd',
      header: '±sd',
      num: true,
      width: 62,
      value: (row) => row.sd_diff,
      render: (row) => <span className="faint">{fmt.num(row.sd_diff, 1)}</span>,
    },
    {
      key: 'win',
      header: 'P(win)',
      num: true,
      width: 74,
      value: (row) => row.win_probability,
      render: (row) => fmt.pct(row.win_probability, 1),
    },
    {
      key: 'leverage',
      header: 'leverage',
      help: 'marginal win probability per point, relative to a coin flip',
      num: true,
      width: 82,
      value: (row) => row.leverage,
      render: (row) => fmt.num(row.leverage, 2),
    },
    {
      key: 'ppp',
      header: 'pp/pt',
      help: 'percentage points of this game bought by one projected point',
      num: true,
      width: 68,
      value: (row) => row.points_per_win_pct,
      // Already in percentage points server-side (`WeekLeverage.points_per_win_pct`
      // multiplies by 100), so this must not go through a percentage formatter.
      render: (row) => <span className="faint">{fmt.num(row.points_per_win_pct, 2)}</span>,
    },
  ];
  return (
    <Panel
      flush
      title="Remaining schedule"
      right={
        <div className="row">
          <Stat
            size="xs"
            align="right"
            label="mean leverage"
            value={data.mean_leverage}
            format="number"
            digits={2}
          />
          {data.least_leveraged ? (
            <span className="caption">
              least leveraged: week {data.least_leveraged.week} at{' '}
              {fmt.num(data.least_leveraged.leverage, 2)}
            </span>
          ) : null}
        </div>
      }
    >
      <Table
        compact
        columns={columns}
        rows={data.weeks}
        rowKey={(row) => row.week}
        dim={(row) => row.decided || row.leverage < DECIDED_LEVERAGE}
        maxHeight={540}
      />
    </Panel>
  );
}
