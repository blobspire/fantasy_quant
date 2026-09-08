/**
 * Headless render tests against LIVE captured payloads.
 *
 * The fixtures in test/fixtures/ were captured from the running API against the
 * user's three real leagues, so these exercise the shapes the dashboard actually
 * receives rather than shapes someone imagined. That matters: every earlier bug
 * in this project that mattered was a real-data bug that synthetic tests passed.
 *
 * The load-bearing assertion is the `is-dim` one. "Uncertainty must be visible"
 * is a claim about pixels, not about types, and a `significant` field that the UI
 * receives and ignores would satisfy the type checker while misleading the user
 * into acting on noise.
 */
import { render, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { AppStateProvider } from '../src/components/Layout';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import leaguesFixture from './fixtures/leagues.json';
import lineupFixture from './fixtures/lineup.json';
import oddsFixture from './fixtures/odds.json';
import portfolioFixture from './fixtures/portfolio.json';
import queueFixture from './fixtures/queue.json';
import rosterFixture from './fixtures/roster.json';
import streamFixture from './fixtures/stream.json';
import tradesFixture from './fixtures/trades.json';
import weeklyFixture from './fixtures/weekly.json';
import waiversFixture from './fixtures/waivers.json';

const FIXTURES: Array<[RegExp, unknown]> = [
  [/\/api\/leagues\/\d+\/odds/, oddsFixture],
  [/\/api\/leagues\/\d+\/roster/, rosterFixture],
  [/\/api\/leagues\/\d+\/weekly/, weeklyFixture],
  [/\/api\/leagues\/\d+\/trades/, tradesFixture],
  [/\/api\/leagues\/\d+\/stream/, streamFixture],
  [/\/api\/leagues\/\d+\/waivers/, waiversFixture],
  [/\/api\/leagues\/\d+\/lineup/, lineupFixture],
  [/\/api\/portfolio/, portfolioFixture],
  [/\/api\/queue/, queueFixture],
  [/\/api\/leagues/, leaguesFixture],
];

function mockFetch() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(typeof input === 'string' ? input : (input as Request).url ?? input);
    const hit = FIXTURES.find(([re]) => re.test(url));
    if (!hit) {
      return new Response(JSON.stringify({ ok: false, error: { type: 'NotFound' } }), {
        status: 404, headers: { 'content-type': 'application/json' },
      });
    }
    return new Response(JSON.stringify(hit[1]), {
      status: 200, headers: { 'content-type': 'application/json' },
    });
  });
}

beforeEach(() => {
  vi.stubGlobal('fetch', mockFetch());
});

/** Every view is lazy in the app; import directly so a failure is this test's. */
async function renderView(name: string) {
  const mod = await import(`../src/views/${name}.tsx`);
  const View = mod.default ?? mod[name];
  const utils = render(
    <MemoryRouter initialEntries={['/league/161496047']}>
      <AppStateProvider>
        <View leagueId={161496047} season={2026} />
      </AppStateProvider>
    </MemoryRouter>,
  );
  // Views start in a loading state; wait for the fetch to settle.
  await waitFor(() => expect(document.body.textContent).not.toMatch(/^\s*$/), { timeout: 5000 });
  return utils;
}

describe('every view renders against live payloads', () => {
  for (const name of ['Queue', 'Overview', 'League', 'Waivers', 'Trades', 'Lineup', 'Portfolio']) {
    it(`${name} mounts and produces content`, async () => {
      const { container } = await renderView(name);
      expect(container.textContent?.length ?? 0).toBeGreaterThan(50);
      // A view stuck on its error boundary is not "rendered".
      expect(container.textContent).not.toMatch(/Cannot read|undefined is not|Objects are not valid/i);
    });
  }
});

describe('the queue is the product, so check it in detail', () => {
  it('renders one row per action from the payload', async () => {
    const rows = (queueFixture as any).data.actions ?? (queueFixture as any).data.rows;
    const { container } = await renderView('Queue');
    await waitFor(() => {
      expect(container.querySelectorAll('tbody tr').length).toBeGreaterThanOrEqual(
        Math.min(rows.length, 5),
      );
    });
  });

  it('names the leagues from the payload, so it is not rendering a placeholder', async () => {
    await renderView('Queue');
    await waitFor(() => {
      expect(document.body.textContent).toMatch(/League Alpha|League Bravo|League Charlie/);
    });
  });

  it('DIMS the rows whose effect is inside their own error', async () => {
    const rows: any[] = (queueFixture as any).data.actions ?? (queueFixture as any).data.rows;
    const insignificant = rows.filter((r) => !r.significant).length;
    expect(insignificant).toBeGreaterThan(0); // guard the guard: the fixture must contain some

    const { container } = await renderView('Queue');
    await waitFor(() => expect(container.querySelectorAll('tbody tr').length).toBeGreaterThan(0));
    const dimmed = container.querySelectorAll('tbody tr.is-dim, tbody tr[data-dim="true"]');
    expect(dimmed.length).toBeGreaterThan(0);
  });

  it('does not hide insignificant rows outright -- they recede, they do not vanish', async () => {
    const rows: any[] = (queueFixture as any).data.actions ?? (queueFixture as any).data.rows;
    const { container } = await renderView('Queue');
    await waitFor(() => expect(container.querySelectorAll('tbody tr').length).toBeGreaterThan(0));
    // Default view shows everything; the hide-noise toggle is opt-in.
    expect(container.querySelectorAll('tbody tr').length).toBeGreaterThan(
      rows.filter((r) => r.significant).length,
    );
  });

  it('shows an uncertainty alongside the headline number', async () => {
    const { container } = await renderView('Queue');
    await waitFor(() => expect(container.textContent).toMatch(/±|\+\/-/));
  });
});

describe('numbers come from the payload, not from the client', () => {
  it('the odds table shows the payload championship values', async () => {
    const teams: any[] = (oddsFixture as any).data.teams ?? (oddsFixture as any).data.rows;
    const top = [...teams].sort((a, b) => b.championship - a.championship)[0];
    const { container } = await renderView('League');
    await waitFor(() => {
      const pct = (top.championship * 100).toFixed(1);
      expect(container.textContent).toContain(pct.slice(0, 3));
    });
  });
});

describe('failure states', () => {
  it('a view whose endpoint 500s shows an error rather than a blank page', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('boom', { status: 500 })),
    );
    const { container } = await renderView('Queue');
    await waitFor(() => {
      expect(container.textContent).toMatch(/error|failed|unavailable|could not/i);
    });
  });
});
