/**
 * Discovered trades, two-team and three-team, written from the other guy's side
 * of the table.
 *
 * The framing is not decoration. Eye-tracking work on exchange finds sellers
 * fixate on the good and buyers on the price, so a proposal that leads with what
 * the counterparty gives up reads as a demand; `decide/trades.py` writes its
 * pitch counterparty-first for the same reason, and the copyable text here keeps
 * that order.
 *
 * Two things this screen must not do. It must not present the top row's
 * `delta_title` as a measurement of that trade — it is the maximum of `n_found`
 * positively correlated noisy estimates and is biased upward, which is what
 * `selection_note` says and why that note sits above the list rather than under
 * it. And it must not present a trade tagged `counterparty-loses` as a clean
 * offer: the simulation says that side's title odds FALL under a trade the
 * points gate called Pareto-improving, which is the difference between a trade
 * offer and a trick.
 */
import { useCallback, useState } from 'react';

import { api, fmt, type MovedPlayer, type TradeRow, type TradesPayload } from '../api';
import { RankingsNote } from '../components/RankingsNote';
import { Delta } from '../components/Delta';
import { Failure, Loading, useLeagueId, useResource } from '../components/Layout';
import { Stat } from '../components/Stat';
import { Legend, Panel } from './Queue';

export default function Trades() {
  const leagueId = useLeagueId();
  const found = useResource<TradesPayload>(
    (signal) =>
      leagueId === null
        ? Promise.reject(new Error('no league selected'))
        : api.trades(leagueId, { limit: 6 }, { signal }),
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
  if (found.loading && !found.data) return <Loading what="trade search (the slow one)" />;
  if (found.error && !found.data) return <Failure error={found.error} onRetry={found.reload} />;
  if (!found.data) return null;

  const data = found.data;
  const trades = data.trades ?? [];

  return (
    <div className="grid">
      <header className="row row--wrap" style={{ justifyContent: 'space-between' }}>
        <div>
          <h1 className="panel__title" style={{ fontSize: 'var(--fs-lg)' }}>
            Trades — {data.name}
          </h1>
          <div className="caption">
            {data.rankings
              ? 'Two questions, asked of different people: does it help you by the analyst board, and does it help them by the projections on their own screen.'
              : 'Every side has to gain on its own starting lineup, or it never gets proposed.'}{' '}
            {data.n_found} candidate{data.n_found === 1 ? '' : 's'} confirmed
            {data.min_gain ? ` at a minimum gain of ${fmt.num(data.min_gain, 1)} points` : ''}.
          </div>
        </div>
      </header>

      {found.error ? (
        <p className="note note--warn">Showing the last good search: {found.error.message}</p>
      ) : null}
      {/* Both notes describe the LIST. With no list they describe nothing, and a
          selection warning about "the top row" printed above "no trade improves
          both sides" is a caveat on a number that is not on the page. */}
      {trades.length > 0 && data.selection_note ? (
        <p className="note note--warn">{data.selection_note}</p>
      ) : null}
      {data.rankings ? <RankingsNote rankings={data.rankings} surface="trades" /> : null}
      {trades.length > 0 ? <CounterpartySide priced={Boolean(data.rankings)} /> : null}

      {trades.length === 0 ? (
        <div className="state">
          <div className="state__title">No trade improves both sides here.</div>
          <p className="prose">
            The search gates on a strict Pareto improvement in playoff-weighted starting points
            before it prices anything in championship probability, and nothing on this board cleared
            it. That is the common case and it is a result: there is nothing to go and propose
            today.
          </p>
        </div>
      ) : (
        trades.map((trade, index) => (
          <TradeCard
            key={`${trade.league_id}:${index}`}
            trade={trade}
            rank={index + 1}
            myTeamId={data.team_id}
            leagueName={data.name}
            nFound={data.n_found}
          />
        ))
      )}
      <Legend />
    </div>
  );
}

/* ==========================================================================
   One trade
   ========================================================================== */

interface Receipt {
  teamId: number | null;
  label: string;
  gets: string[];
}

/**
 * What each team walks away with, off the move's own player list.
 *
 * Counterparties come first because the proposal is framed around what they
 * receive. This is a regrouping of `players[]` by `to_team`, not a computation.
 */
function receiptsFor(trade: TradeRow, myTeamId: number | null): Receipt[] {
  const names = new Map<number, string>();
  for (const partner of trade.partners ?? []) names.set(partner.team_id, partner.name);

  const byTeam = new Map<number, MovedPlayer[]>();
  for (const player of trade.players ?? []) {
    const list = byTeam.get(player.to_team) ?? [];
    list.push(player);
    byTeam.set(player.to_team, list);
  }

  const receipts: Receipt[] = [];
  for (const [teamId, players] of byTeam) {
    if (teamId === myTeamId) continue;
    receipts.push({
      teamId,
      label: names.get(teamId) ?? `Team ${teamId}`,
      gets: players.map((player) => player.name),
    });
  }
  const mine =
    myTeamId !== null && byTeam.has(myTeamId)
      ? (byTeam.get(myTeamId) as MovedPlayer[]).map((player) => player.name)
      : (trade.receive ?? []).map((player) => player.name);
  receipts.push({ teamId: myTeamId, label: 'You', gets: mine });
  return receipts;
}

/** The producing surface's own opening sentences. Its words, not a template. */
function firstSentences(text: string, count: number): string {
  const parts = text
    .split('. ')
    .map((part) => part.trim())
    .filter(Boolean);
  if (!parts.length) return '';
  const head = parts.slice(0, count).join('. ');
  return head.endsWith('.') ? head : `${head}.`;
}

function proposalText(trade: TradeRow, receipts: Receipt[], leagueName: string): string {
  const lines: string[] = [];
  const pitch = firstSentences(trade.rationale ?? '', 2);
  if (pitch) lines.push(pitch, '');
  for (const receipt of receipts) {
    if (receipt.label === 'You') continue;
    lines.push(`${receipt.label} gets: ${receipt.gets.join(', ') || 'nothing'}`);
  }
  const mine = receipts.find((receipt) => receipt.label === 'You');
  lines.push(`I get: ${mine?.gets.join(', ') || 'nothing'}`);
  if (leagueName) lines.push('', `(${leagueName})`);
  return lines.join('\n').trim();
}

function analysisText(trade: TradeRow, receipts: Receipt[]): string {
  const lines = [
    receipts
      .map((receipt) => `${receipt.label} gets ${receipt.gets.join(', ') || 'nothing'}`)
      .join(' | '),
    `my ΔP(title) ${fmt.pp(trade.delta_title)} ±${fmt.ppError(trade.stderr)}` +
      (trade.z === null ? '' : ` (${fmt.z(trade.z)})`) +
      `, verdict ${trade.verdict}, confidence ${trade.confidence}`,
    `my Δpoints ${fmt.signed(trade.delta_points, 1)} playoff-weighted starters`,
    trade.spread === null || trade.spread === undefined
      ? ''
      : `their read vs ours ${fmt.signed(trade.spread, 1)} (by the projections on their screen)`,
    ...(trade.caveats ?? []).map((caveat) => `caveat: ${caveat}`),
    ...Object.entries(trade.notes ?? {}).map(([who, note]) => `${who}: ${note}`),
    trade.rationale ?? '',
  ];
  return lines.filter(Boolean).join('\n');
}

function TradeCard({
  trade,
  rank,
  myTeamId,
  leagueName,
  nFound,
}: {
  trade: TradeRow;
  rank: number;
  myTeamId: number | null;
  leagueName: string;
  nFound: number;
}) {
  const receipts = receiptsFor(trade, myTeamId);
  const counterparties = receipts.filter((receipt) => receipt.label !== 'You');
  const mine = receipts.find((receipt) => receipt.label === 'You');
  const nTeams = (trade.tags ?? []).find((tag) => /^\d+-team$/.test(tag)) ?? `${receipts.length}-team`;
  const counterpartyLoses = (trade.tags ?? []).includes('counterparty-loses');
  // The server's own verdict, never a test run here. Half this board is `noise`
  // on a live search and the cards for those are otherwise identical to the
  // ones that cleared -- same size, same headline, same copy buttons -- which
  // is exactly how a search winner gets proposed as if it were a measurement.
  const resolved = trade.verdict === 'act' || (trade.verdict === undefined && trade.significant);

  return (
    <Panel
      title={
        <span className="row" style={{ gap: 'var(--sp-4)' }}>
          <span className="faint">#{rank}</span>
          <span>{counterparties.map((receipt) => receipt.label).join(' + ') || 'a rival'}</span>
        </span>
      }
      right={
        <div className="row">
          <span className="chip">{nTeams}</span>
          <span className={trade.confidence === 'low' ? 'chip chip--warn' : 'chip'}>
            {trade.confidence} confidence
          </span>
          {resolved ? null : (
            <span className="chip chip--warn" title={trade.verdict_note}>
              inside its own error
            </span>
          )}
          {counterpartyLoses ? <span className="chip chip--bad">counterparty loses</span> : null}
        </div>
      }
    >
      <div className="grid" style={{ gap: 'var(--sp-5)' }}>
        {/* The offer. Their side leads. */}
        <div className="grid" style={{ gap: 'var(--sp-3)' }}>
          {counterparties.map((receipt) => (
            <div key={receipt.teamId ?? receipt.label} style={{ fontSize: 'var(--fs-md)' }}>
              <strong style={{ color: 'var(--fg)' }}>{receipt.label}</strong>
              <span className="dim"> gets </span>
              <strong style={{ color: 'var(--fg)' }}>
                {receipt.gets.join(', ') || 'nothing'}
              </strong>
            </div>
          ))}
          <div className="dim" style={{ fontSize: 'var(--fs-sm)' }}>
            You get {mine?.gets.join(', ') || 'nothing'}
          </div>
        </div>

        <div className="row row--wrap" style={{ gap: 'var(--sp-7)', alignItems: 'flex-end' }}>
          <div className="stat stat--md">
            <div className="stat__label">your ΔP(title)</div>
            <div className="stat__figure">
              <Delta
                size="lg"
                value={trade.delta_title}
                stderr={trade.stderr}
                verdict={trade.verdict}
                significant={trade.significant}
                note={trade.verdict_note}
              />
            </div>
            {trade.verdict_note ? <div className="stat__sub">{trade.verdict_note}</div> : null}
          </div>
          <Stat
            size="xs"
            label="your Δpoints"
            value={fmt.signed(trade.delta_points, 1)}
            format="raw"
            sub="playoff-weighted starters"
          />
          {trade.spread !== null && trade.spread !== undefined ? (
            // The arbitrage. How much better the OTHER side reads this by the
            // projections on their screen than by the analyst board. Positive means
            // they think they are getting more than we think they are.
            <Stat
              size="xs"
              label="their read vs ours"
              value={fmt.signed(trade.spread, 1)}
              format="raw"
              sub={trade.mispriced ? 'they read it richer than it is' : 'both boards agree'}
            />
          ) : null}
          <Stat
            size="xs"
            label="z"
            value={trade.z === null ? null : fmt.z(trade.z)}
            format="raw"
            sub={`best of ${nFound}`}
          />
          <Stat size="xs" label="leverage" value={trade.leverage} format="number" digits={2} />
        </div>

        {resolved ? null : (
          <p className="note note--warn">
            {trade.verdict_note ||
              'this did not clear the selection-adjusted threshold for a search winner'}{' '}
            — not a proposal yet. Re-run with another seed before you send it.
          </p>
        )}

        {trade.caveats?.length ? (
          <ul style={{ margin: 0, paddingLeft: '1.1em' }}>
            {trade.caveats.map((caveat, index) => (
              <li
                key={index}
                className="caption"
                style={counterpartyLoses && index === 0 ? { color: 'var(--bad)' } : undefined}
              >
                {caveat}
              </li>
            ))}
          </ul>
        ) : null}

        {trade.same_return ? (
          <p className="caption">
            Same players, same price as a card above — routed through a different team, so
            it needs a different person to say yes.
          </p>
        ) : trade.routes ? (
          <p className="caption">
            {trade.routes} other route{trade.routes > 1 ? 's' : ''} below deliver the same
            return; this is the easiest to get signed.
          </p>
        ) : null}

        {trade.notes && Object.keys(trade.notes).length ? (
          <ul style={{ margin: 0, paddingLeft: '1.1em' }}>
            {Object.entries(trade.notes).map(([who, note]) => (
              <li key={who} className="caption">
                <strong>{who}:</strong> {note}
              </li>
            ))}
          </ul>
        ) : null}

        {trade.rationale ? <p className="prose">{trade.rationale}</p> : null}

        <div className="row">
          <CopyButton label="Copy proposal" text={proposalText(trade, receipts, leagueName)} />
          <CopyButton ghost label="Copy analysis" text={analysisText(trade, receipts)} />
        </div>
      </div>
    </Panel>
  );
}

/**
 * The other side of the table.
 *
 * A trade that helps only you does not get accepted, so the counterparty's own
 * simulated delta is what decides whether this is worth sending — and it is the
 * one number this payload does not carry. `decide/trades.TradeEvaluation` holds
 * a `TeamImpact` per team, `report.rec_payload` flattens the recommendation to
 * the user's side, and the counterparty's figure survives only in the pitch (a
 * points gain) and in the `counterparty-loses` tag. Stated rather than
 * approximated: an estimate of their side computed in the browser would be a
 * number this system never measured.
 *
 * Stated ONCE, above the list. It is the same sixty words on every card, and
 * repeated per trade it became the longest text on the page -- a notice about a
 * missing API field outweighing the analysis it qualifies.
 */
function CounterpartySide({ priced }: { priced: boolean }) {
  if (priced) {
    return (
      <p className="note">
        Their simulated ΔP(title) is not in this payload. What is: every counterparty gains in{' '}
        <em>playoff-weighted starting points</em> by the projections on their own screen, and
        "their read vs ours" on each card is how much richer they read the deal than the analyst
        board does. That is an acceptance <em>condition</em>, not a prediction that they accept.
        A <span className="mono">counterparty-loses</span> tag means the title simulation
        disagrees about their side.
      </p>
    );
  }
  return (
    <p className="note">
      Their simulated ΔP(title) is not in this payload. The gate every side cleared is a strict
      Pareto improvement in <em>playoff-weighted starting points</em>, quoted per counterparty in
      the pitch below; a <span className="mono">counterparty-loses</span> tag means the title
      simulation disagrees with that gate about their side. Add `impacts[]` (team_id, name,
      delta_title, delta_title_stderr) to the trades payload and both sides render here as numbers.
    </p>
  );
}

function CopyButton({ label, text, ghost }: { label: string; text: string; ghost?: boolean }) {
  const [done, setDone] = useState(false);
  const copy = useCallback(() => {
    const flash = () => {
      setDone(true);
      window.setTimeout(() => setDone(false), 1400);
    };
    const fallback = () => {
      const area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.left = '-9999px';
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand('copy');
      } finally {
        document.body.removeChild(area);
      }
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(flash, () => {
        fallback();
        flash();
      });
    } else {
      fallback();
      flash();
    }
  }, [text]);
  return (
    <button type="button" className={ghost ? 'btn btn--sm btn--ghost' : 'btn btn--sm'} onClick={copy}>
      {done ? 'copied' : label}
    </button>
  );
}
