/**
 * The one line every board-priced surface owes the reader.
 *
 * Names the analyst board the numbers were priced against and says, in the same
 * breath, that it is unverified. That second half is not hedging: every other input
 * in this system carries a measured verdict before it ships, and this one cannot
 * yet, because no historical boards exist to score it against. The archive that
 * makes a verdict possible is being accumulated; until then `weight` is the one
 * constant that turns the whole thing off.
 */
import type { RankingsRef } from '../api';

export function RankingsNote({
  rankings,
  surface,
}: {
  rankings: RankingsRef;
  surface: 'trades' | 'waivers';
}) {
  const fit = rankings.matches_league_scoring
    ? ''
    : ` (${rankings.scoring.replace('_', ' ')} board standing in; within-position orderings drift about one rank)`;
  return (
    <p className="note">
      Priced against the <strong>{rankings.kind}</strong> board — {rankings.n} players, weight{' '}
      {rankings.weight.toFixed(1)}
      {fit}.{' '}
      {surface === 'trades'
        ? 'ΔP(title) on every card is priced on the re-dealt projections, so compare the deltas between cards, not their level to the odds page.'
        : 'Adds the board ranks above somebody you hold carry his note and both ranks.'}{' '}
      <em>{rankings.unverified}</em>
    </p>
  );
}
