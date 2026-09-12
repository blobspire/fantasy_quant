/**
 * The cross-league action queue: every league's best moves in ONE ranked list.
 *
 * This is the landing view and the point of the product. Three leagues each run
 * four decision surfaces, and every one of them reports its effect in the same
 * unit -- percentage points of championship probability -- which is the only
 * reason a single ordering can exist. The server ranks it (`edges.portfolio
 * .action_queue` on one shared NFL season, or `report.rank_actions` as the
 * fallback) and this renders that order without re-deriving it. The rule the
 * order encodes is worth knowing while reading the table: **anything inside its
 * own Monte Carlo error sorts below everything that is not**, whatever its point
 * estimate, because the largest number on a live board is routinely a noisy one.
 *
 * Nothing here computes analysis. Sorting a column and hiding a row are the only
 * client-side operations, and both are opt-in.
 */
import { useMemo, useState, type ReactNode } from 'react';

import { api, fmt, type ActionRow, type QueuePayload } from '../api';
import { DeltaCell, ErrorBar } from '../components/Delta';
import { Failure, Loading, useResource } from '../components/Layout';
import { Table, type Column } from '../components/Table';

/** What the shell passes a view. Every field optional; see `App.tsx`. */
export interface ViewProps {
  leagueId?: number | null;
  season?: number | null;
  onNavigate?: (path: string) => void;
}

/**
 * A titled card. `theme.css` styles `.panel`, and there is no `Panel.tsx` in
 * `components/`, so this is the minimum implementation of that markup rather
 * than a second design. Exported because every view in this module needs it.
 */
export function Panel(props: {
  title?: ReactNode;
  right?: ReactNode;
  flush?: boolean;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      {props.title !== undefined || props.right !== undefined ? (
        <header className="panel__head">
          <div className="panel__title">{props.title}</div>
          {props.right}
        </header>
      ) : null}
      <div className={props.flush ? 'panel__body panel__body--flush' : 'panel__body'}>
        {props.children}
      </div>
    </section>
  );
}

/** What the markers beside a number mean. Printed under anything that uses them. */
export function Legend() {
  return (
    <div className="caption">
      <span className="mono">~</span> inside its own Monte Carlo error (shown, not acted on)
      {'  ·  '}
      <span className="mono">!</span> measured in the WRONG direction
      {'  ·  '}
      deltas are signed; pp = percentage points of championship probability
    </div>
  );
}

/** An action with the server's own rank frozen onto it, so sorting cannot lie. */
type RankedAction = ActionRow & { rank: number };

const SURFACE_PAGE: Record<string, string> = {
  waiver: 'waivers',
  waivers: 'waivers',
  trade: 'trades',
  trades: 'trades',
  lineup: 'lineup',
  lineups: 'lineup',
  stream: 'stream',
  streaming: 'stream',
};

export default function Queue({ onNavigate }: ViewProps) {
  const queue = useResource<QueuePayload>((signal) => api.queue({ limit: 24 }, { signal }), []);
  const [doTodayOnly, setDoTodayOnly] = useState(false);
  const [hideNoise, setHideNoise] = useState(false);

  const payload = queue.data;
  const ranked = useMemo<RankedAction[]>(
    () => (payload?.actions ?? []).map((action, index) => ({ ...action, rank: index + 1 })),
    [payload],
  );
  const rows = useMemo(
    () =>
      ranked.filter((row) => {
        if (doTodayOnly && !row.actionable) return false;
        if (hideNoise && !row.significant) return false;
        return true;
      }),
    [ranked, doTodayOnly, hideNoise],
  );

  const columns = useMemo<Array<Column<RankedAction>>>(
    () => [
      {
        key: 'rank',
        header: '#',
        num: true,
        width: 38,
        value: (row) => row.rank,
        render: (row) => row.rank,
      },
      {
        key: 'league',
        header: 'league',
        width: 150,
        value: (row) => row.league_name,
        render: (row) => row.league_name,
      },
      {
        key: 'surface',
        header: 'surface',
        width: 84,
        value: (row) => row.surface,
        render: (row) => <span className="chip">{row.surface}</span>,
      },
      {
        key: 'action',
        header: 'action',
        width: '46%',
        sortable: false,
        render: (row) => <ActionCell row={row} />,
      },
      {
        key: 'delta',
        header: 'ΔP(title)',
        help: 'increment to this league’s championship probability',
        num: true,
        width: 92,
        defaultDesc: true,
        value: (row) => row.delta_title,
        render: (row) => (
          <DeltaCell
            value={row.delta_title}
            verdict={row.verdict}
            significant={row.significant}
            note={row.rationale}
          />
        ),
      },
      {
        key: 'stderr',
        header: '±',
        help: 'Monte Carlo standard error on the delta',
        num: true,
        width: 66,
        value: (row) => row.stderr,
        render: (row) => <ErrorBar stderr={row.stderr} />,
      },
      {
        key: 'z',
        header: 'z',
        help: 'delta / standard error, as the server measured it',
        num: true,
        width: 52,
        value: (row) => row.z,
        render: (row) => (row.z === null ? <span className="faint">{fmt.dash}</span> : fmt.z(row.z)),
      },
      {
        key: 'leverage',
        header: 'lev',
        help: 'marginal win probability per point, relative to a coin flip',
        num: true,
        width: 52,
        value: (row) => row.leverage,
        render: (row) => fmt.num(row.leverage, 2),
      },
      {
        key: 'cost',
        header: 'cost / deadline',
        width: 176,
        value: (row) => row.cost,
        render: (row) => (
          <div>
            <div>{row.cost}</div>
            <div className="caption">{row.deadline}</div>
          </div>
        ),
      },
    ],
    [],
  );

  if (queue.loading && !payload) return <Loading what="queue across every league" slow />;
  if (queue.error && !payload) return <Failure error={queue.error} onRetry={queue.reload} />;

  const lead =
    payload && payload.first_actionable !== null
      ? ranked[payload.first_actionable]
      : undefined;

  // `report.queue_payload` publishes a one-line summary of which surfaces
  // cleared their own field and which passed a two-sigma test but failed the
  // selection-adjusted one. `api.ts` does not name it yet, so it is read
  // defensively rather than being left on the floor: it is the page's only
  // statement about *why* the ordering below is not simply by point estimate.
  const rawNote = (payload as unknown as { selection_note?: unknown } | null)?.selection_note;
  const selectionNote = typeof rawNote === 'string' ? rawNote.trim() || null : null;

  return (
    <div className="grid">
      <header className="row row--wrap" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1 className="panel__title" style={{ fontSize: 'var(--fs-lg)' }}>
            What to do this week
          </h1>
          <div className="caption">
            {lead ? (
              <>
                First thing to actually do — <strong>{lead.league_name}</strong>: {lead.headline}
              </>
            ) : (
              'Every league’s best move, in one order, priced in the same unit.'
            )}
          </div>
        </div>
        <div className="row row--wrap">
          <label className="caption row" style={{ gap: 'var(--sp-2)' }}>
            <input
              type="checkbox"
              checked={doTodayOnly}
              onChange={(event) => setDoTodayOnly(event.currentTarget.checked)}
            />
            do-today only
          </label>
          <label className="caption row" style={{ gap: 'var(--sp-2)' }}>
            <input
              type="checkbox"
              checked={hideNoise}
              onChange={(event) => setHideNoise(event.currentTarget.checked)}
            />
            hide inside-error rows
          </label>
        </div>
      </header>

      {queue.error ? (
        <p className="note note--warn">
          Showing the last good queue; the refetch failed: {queue.error.message}
        </p>
      ) : null}

      {selectionNote ? <p className="note note--warn">{selectionNote}</p> : null}

      {(payload?.leagues ?? [])
        .filter((league) => !league.ok)
        .map((league) => (
          <p className="note note--bad" key={league.league_id}>
            <strong>{league.name || league.league_id}</strong> did not build:{' '}
            {league.error ?? 'unknown error'}. Every other league below is unaffected.
          </p>
        ))}

      {ranked.length === 0 ? (
        <div className="state">
          <div className="state__title">Nothing to do anywhere today.</div>
          <p className="prose">
            Every surface in every league ran and none of them found a move worth making. That is a
            result, not a failure — most weeks in most leagues are holds, and the per-league reasons
            are below.
          </p>
        </div>
      ) : (
        <Panel
          flush
          title="Action queue"
          right={
            <span className="caption">
              {rows.length} of {payload?.n_actions ?? ranked.length} shown · ranked by{' '}
              <span className="mono">{payload?.ranked_by}</span>
            </span>
          }
        >
          <Table
            columns={columns}
            rows={rows}
            rowKey={(row) => `${row.league_id}:${row.surface}:${row.rank}`}
            dim={(row) => !row.significant || !row.actionable}
            highlight={(row) => row.rank - 1 === payload?.first_actionable}
            onRowClick={
              onNavigate
                ? (row) => {
                    // An unknown surface still has a league to open; a bare
                    // trailing slash would land on the router's 404.
                    const page = SURFACE_PAGE[row.surface];
                    onNavigate(page ? `/league/${row.league_id}/${page}` : `/league/${row.league_id}`);
                  }
                : undefined
            }
            empty={
              <>
                Every row is filtered out. {ranked.length} measured action
                {ranked.length === 1 ? '' : 's'} are still there — clear the filters to see them.
              </>
            }
          />
        </Panel>
      )}

      {payload?.holds?.length ? <Holds holds={payload.holds} /> : null}

      {payload?.errors?.length ? (
        <Panel title="Surfaces that failed">
          <div className="grid" style={{ gap: 'var(--sp-3)' }}>
            {payload.errors.map((failure, index) => (
              <p className="note note--bad" key={index}>
                <strong>{failure.league ?? failure.league_id ?? 'a league'}</strong>
                {failure.surface ? ` / ${failure.surface}` : ''}: {failure.error}
              </p>
            ))}
          </div>
        </Panel>
      ) : null}

      <div className="grid" style={{ gap: 'var(--sp-3)' }}>
        <Legend />
        <div className="caption">
          The three deltas are increments to three <em>different</em> probabilities and must never
          be added into one. What makes them rankable together is that each is an increment to
          E[titles] = Σ P(title), which is additive across leagues exactly, by linearity of
          expectation, with no independence assumption anywhere.
        </div>
      </div>
    </div>
  );
}

/**
 * The action itself, plus why it is not simply "do it".
 *
 * `blockers` is NOT "not today". `report.Action` defines it as "why it is not
 * simply 'do it', in the producing surface's own vocabulary", and `actionable`
 * is the separate flag for whether there is anything to execute now. Printing
 * "not today" over a blocker contradicted the payload on the one row that
 * matters most: the queue's `first_actionable` is Blacksburg's trade, which is
 * `actionable: true` with a `counterparty-loses` blocker, so the page
 * highlighted a row as "the first thing to actually do" and captioned it "not
 * today" — where `fq queue` prints "Do now: #4" for the same row. The prefix
 * now follows `actionable`, which is the field that answers that question.
 */
function ActionCell({ row }: { row: RankedAction }) {
  return (
    <div style={{ whiteSpace: 'normal', paddingTop: 4, paddingBottom: 4 }}>
      <div>{row.headline}</div>
      {row.blockers?.length ? (
        <div className="caption" style={{ color: 'var(--noise)' }}>
          {row.actionable ? 'not simply “do it”: ' : 'nothing to execute today: '}
          {row.blockers.join('; ')}
        </div>
      ) : null}
      {row.caveats?.length ? (
        <ul className="caption" style={{ margin: '2px 0 0', paddingLeft: '1.1em' }}>
          {row.caveats.map((caveat, index) => (
            <li key={index}>{caveat}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/**
 * "Nothing to do in this league today" as a first-class, calm result.
 *
 * `report.headline` composes these sentences out of measured numbers — what the
 * best available move was worth, what it had to clear, what a point is worth
 * this week — so a hold arrives as a quantitative answer. Rendering it as an
 * error would teach the reader that a quiet week means the tool broke.
 */
function Holds({ holds }: { holds: string[] }) {
  return (
    <Panel title="Nothing to do today">
      <div className="grid" style={{ gap: 'var(--sp-5)' }}>
        {holds.map((hold, index) => {
          const cut = hold.indexOf(': ');
          const league = cut > 0 ? hold.slice(0, cut) : '';
          const why = cut > 0 ? hold.slice(cut + 2) : hold;
          return (
            <div key={index}>
              <div className="row" style={{ gap: 'var(--sp-4)' }}>
                {league ? <strong style={{ color: 'var(--fg)' }}>{league}</strong> : null}
                <span className="chip">hold</span>
              </div>
              <p className="prose">{why}</p>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
