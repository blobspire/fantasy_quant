"""Three leagues as one position, which is not the same as three separate bets.

Every other surface in this system prices one league at a time, and that is the right
scope for a waiver claim: the claim happens in a league, the bracket happens in a
league, `delta_title` means "this league's title". But the user does not own three
leagues, he owns one **portfolio** of three, and three numbers that each mean something
in isolation still have to be combined by somebody. Combining them wrongly is easy and
silent, so this module exists to do it once and to say what the combination is.

Four things live here, in order of how much they change what the user does.

**1. The action queue, which is the reason `delta_title` is the unit.** Every surface
-- `decide/waivers`, `decide/trades`, `decide/lineups`, `decide/streaming` -- already
returns a `core.Recommendation`, so a waiver claim in Wine Wednesday and a start/sit
call in Type shi are directly comparable *without any conversion*. This module runs all
four in all three leagues and merges the output into one ranked list. What that list
ranks is stated in `ActionQueue`: it is NOT a single probability, because the three
`delta_title`s are increments to three different probabilities and adding them is a
category error. It is a ranking of `d E[titles] / d action`, and that is legitimate
because E[titles] genuinely is additive across leagues (see below).

**And the ordering that ranking produces across *surfaces* is mostly selection, which
is the one thing in this module a reader would otherwise act on and should not.** Each
surface hands over the argmax of its own search, and the searches are not the same size:
`decide/waivers` confirms about a dozen claims at a standard error of 0.013pp,
`decide/trades` confirms forty packages at 0.48pp. The maximum of forty draws of pure
noise at 0.48pp is roughly 1.5pp, which is *larger than every trade on the live board*,
and `decide/trades` had already stamped every one of them `confidence="low"` and
published `selection_threshold` to say why. Sorted by raw `delta_title` the live queue
puts three trades at +0.8 to +1.2pp above nine waiver claims at +0.11 to +0.18pp -- an
order of magnitude, and the wrong way round: the waiver claims clear their own
selection-adjusted threshold five times over and not one trade clears its. So
`QueueItem.significant` tests against `decide/trades.selection_threshold(n_considered)`,
`ActionQueue.by_selection_bound` ranks on the corrected key, and `table()` prints the
field size and the surface's own `confidence` beside every row.

**2. What the portfolio odds actually are.** `P(at least one title)` is neither the sum
of the three nor their product complement -- unless the leagues are independent, which
has to be measured rather than assumed. It is measured here by coupling the three
simulations onto **one NFL season**: `sim/distributions.WeeklySampler` keys every
player's random stream on `(seed, player_id)` rather than on his column, so three
leagues drawn with the same seed see the same football, and simulation `s` in Wine
Wednesday is the same week of the same season as simulation `s` in Blacksburg.
`Portfolio.verify_coupling` proves that on the live data rather than trusting the
docstring: on the user's leagues the shared players' season totals correlate at a median
of **0.999** across panels and their availability masks are **identical**, so the
coupling is real and not approximate.

**And then the measured answer is that it barely matters, which is the honest headline.**
On the three real leagues at 4,000 simulations the *weekly scores* of the user's three
teams correlate at **+0.40 / +0.25 / +0.11** -- substantial, and exactly what four shared
players between Wine Wednesday and Blacksburg should produce. The *championship
indicators* correlate at **+0.065 +/- 0.023 / +0.031 +/- 0.020 / -0.022 +/- 0.013**, and
only the first of those three is distinguishable from zero; across five seeds the third
one reads -0.022, +0.005, +0.030, +0.014, -0.029, so its sign is a coin flip and
`PairCorrelation.champion_significant` says so on the row. `P(>=1 title)` comes out
**14.70%** against **14.91%** for the independent calculation and 15.68% for the naive
sum: the whole dependence correction is **0.21pp, against 0.15pp of bootstrapped error on
the correction itself** and 0.56pp on the level. It does not clear two standard errors.
Pooled over five seeds it settles near +0.31pp with a spread of 0.20pp, so the effect is
probably real and is still an order of magnitude too small to be a decision input.

So the correlation is real in the scores and is very nearly annihilated by the bracket,
because a title is a rank statistic inside your own league: a good week from a shared
running back lifts your score against eleven or thirteen rivals who mostly do not own
him, and it has to survive a whole regular season and three playoff rounds before it
shows up as a title. The premise this module was built on -- "the three teams are not
independent bets" -- is true of their points and is *barely* true of their titles.
Anyone quoting a correlation correction on a three-league portfolio should be asked
which of the two he measured.

**3. Concentration, priced rather than ruled.** There is no defensible "no more than X%
in one player" rule, so none is offered. Instead every exposure is re-simulated:
`Exposure.portfolio_damage` is `P(>=1 title)` today minus `P(>=1 title)` with that
player deleted from every roster that holds him for the rest of the season, computed by
re-running the bracket, not by scaling anything.

The same machinery answers it for an NFL team, and **the tempting headline there is not
a finding.** "The NFL-team number beats any single player -- KC costs 5.9pp against the
worst player's 5.0pp" is true and is arithmetic: an NFL-team row deletes two to six
roster spots and a player row deletes one, so the group has to win. KC *is* Kenneth
Walker III, held twice, at -4.93pp, plus Harrison Butker in the third league. What is
worth reading is the remainder, which `Concentration.increment` computes as a paired
per-simulation difference against the group's own biggest holding: +1.00pp +/- 0.34 for
KC, and +0.00pp for PHI, whose five deleted roster spots are Jalen Hurts and four
players worth nothing at all.

Byes were the exception, and this module is where the gap was found and reported for
long enough that it eventually got fixed at the source. `pipeline.build` now passes
ESPN's own bye table, so `has_game` is False on a player's bye and the seat is left
empty. Before that it passed nothing, and byes reached the simulation only *indirectly*,
through the projection: ESPN collapses a skill player's or a kicker's bye week to
0.06-0.29 points and the lineup solver duly benched him. That worked for five positions
out of six and silently failed at the sixth -- ESPN projects a defence normally on its
own bye (the Jaguars at 3.28 in week 7 against a season mean of 4.71), so every defence
played seventeen games. `ByeExposure.unpriced` still names offenders per week, and now
reads `has_game` rather than the projection alone, so a properly modelled bye does not
report itself forever.

**Two floors, and picking the wrong one makes a kicker the most valuable asset the user
owns.** An unfilled starting slot streams a replacement; it does not score zero. This
module therefore fits `decide/title.streaming_levels` by default. `championship_table`
now floors the same way, so the two no longer disagree about the CONVENTION; the levels
still will not match, because this module couples three leagues onto one shared seed and
that is a different drawn season. Compare deltas across surfaces, never levels.
`build_portfolio(stream_replacement=False)` returns to the empty-seat convention if a
caller wants to reconcile.

It fits that floor **off `sim.outlooks`**, and the argument is load-bearing rather than
tidy. Omitting it hands `streaming_replacement` the panel instead, `pipeline.build` pools
only rostered players, so the wire reads empty at every slot and the whole board falls
through to the VOLS roster-bottom rank -- which is not merely too high but too high at
some positions and too LOW at others, so it does not cancel. Measured on the three live
leagues it put RB at 9.22 against a true 4.55 and D/ST at 5.29 against a true 7.39, and
it reordered **39 of 40** rows of `exposures`: Harrison Butker sat 11 places above where
he belongs and Justin Jefferson two below. That is the same inversion `decide/wire.py`
was extracted to eliminate, arriving here through the caller rather than the definition.

There is a live trap behind that choice: `lineup.monotone_floor` raises every slot's
floor to that of any slot whose eligible set it contains, eligibility is computed against
*this roster*, and a roster with nobody at a position has an empty eligible set that is
contained in every other. Delete the user's only quarterback and all nine slots lift to
the QB replacement level -- a 137-point-a-week team with zero variance at 93% title odds.
All three of the user's rosters carry exactly one quarterback, one kicker and one defence,
so this fires on the first interesting exposure rather than on an edge case. The guard now
lives in `sim/season._floors`, which is the one place every floor in the system is built;
this module and `decide/title.py` each used to carry a private copy of it.

**4. Diversification, with the two objectives that disagree.** `E[titles]` is a sum of
three marginals and is therefore *completely blind* to how the leagues covary -- linearity
of expectation is not an approximation. `P(>=1 title)` is not: at fixed marginals it falls
as the leagues become more dependent, over the whole Frechet range from `min(1, sum p)`
down to `max p`. So the two objectives really do give opposite advice about holding the
same player everywhere, and `Diversification` reports the size of the disagreement rather
than the direction alone. On the live portfolio the range is 6.55% (perfectly concentrated)
to 15.68% (perfectly diversified) around a measured 14.70%: there is a lot at stake in
principle -- 8.15pp of downside -- and the user is already sitting within 0.98pp of the
best attainable end of it, so the argument is not worth having on this portfolio today.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ..core import MoveKind, Recommendation, WireLevel
from ..espn.client import EspnClient
from ..pipeline import LeagueSim, build, client_from_env
from ..sim import season as S
from ..sim.distributions import DEFAULT_SEED, Draw
from ..sim.lineup import plan_from_slots

log = logging.getLogger(__name__)

#: Labels for the six positions a normal league starts. Local rather than imported from
#: `espn/constants.py` so a report can be rendered without a season's platform payload.
POSITION_ABBREV: Mapping[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}

#: `lineupSlotId` labels. Slot ids, not position ids -- the two spaces collide at 4 and
#: 15, which is the bug `espn/constants.py` exists to prevent.
SLOT_ABBREV: Mapping[int, str] = {
    0: "QB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",
    16: "DST",
    17: "K",
    20: "BE",
    21: "IR",
    23: "FLEX",
}

SLOT_BENCH = 20

#: Tags and move kinds that mean "this is not something to do today". A streaming plan
#: whose first week is a hold is worth several points of title probability *as a plan*
#: and nothing at all *as an action*, and it out-ranks every real move on the board if
#: nobody separates the two -- on all three live leagues the D/ST plan opens with a hold
#: and prices at +2.3 to +4.4pp.
NOT_ACTIONABLE_TAGS: frozenset[str] = frozenset(
    {"no-action-this-week", "null-plan", "null", "action-suppressed"}
)

#: Tags that mean the move is real but somebody else has to agree to it, or that it is
#: priced as an add with no drop. Kept out of `blockers` for `hold`, which is legitimate.
#:
#: `counterparty-loses-unclear` is deliberately NOT here. It is the honest reading of a
#: counterparty delta too noisy to have a sign, and on the live leagues that describes
#: nearly every trade -- 45 of 120 carried the old bare-sign tag and not one of the 50
#: negative impacts behind it cleared the selection-adjusted threshold. A blocker that
#: fires on "cannot tell" blocks the whole board and stops meaning anything.
CONTESTED_TAGS: frozenset[str] = frozenset(
    {"counterparty-loses", "unilateral", "roster_size", "partially-unpriced"}
)

#: A bye week whose projection keeps more than this share of the player's ordinary week
#: is one the projections did not zero. The live separation is stark -- skill players
#: keep about 0.5% of their projection on a bye and the Jaguars defence keeps 70% -- so
#: the constant only has to land between the two, not be tuned.
BYE_ZERO_TOLERANCE = 0.25

#: How many rows a human will actually work through in one sitting.
DEFAULT_QUEUE_LIMIT = 10

#: Bootstrap resamples for the standard error on a *difference of two probabilities
#: computed from the same simulations*. The two share every sim, so their difference is
#: far better determined than either level and a naive binomial error overstates it by
#: an order of magnitude; resampling the sim index is the only cheap way to see that.
DEFAULT_BOOTSTRAP = 2000

#: Resamples for the correlation standard errors. Fewer than `DEFAULT_BOOTSTRAP`
#: because every correlation in `Correlations` shares one weight matrix, so the same
#: resample is applied to every pair and every week and the *differences* between them
#: (which is what `worst_week` needs) are properly paired.
CORR_BOOTSTRAP = 500


class PortfolioError(ValueError):
    """The three leagues cannot be combined into one portfolio as posed."""


# --------------------------------------------------------------------------------------
# One league's stake
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class LeagueStake:
    """One league in the portfolio, reduced to what the portfolio needs from it.

    The two arrays are the whole point. `champions` is the per-simulation title
    indicator for the user's own franchise, and `weekly` is that franchise's weekly
    starting-lineup total; both are indexed by the *shared* simulation axis, so column
    `s` of one league and column `s` of another are the same NFL season. Every joint
    probability in this module is an arithmetic operation on those columns, which is why
    none of them needs a copula, a correlation matrix or an assumption.
    """

    sim: LeagueSim
    team_id: int
    team_name: str
    #: The fitted streaming level per lineup slot, or None for the empty-seat baseline.
    #: `WireLevel`s rather than bare means, because an empty seat is PAID a draw and not
    #: a constant -- see `noise`.
    replacement: Mapping[int, WireLevel] | None
    #: `(sims,)` 1.0 where the user's franchise won this league.
    champions: np.ndarray
    #: `(sims, weeks)` the user's franchise's starting-lineup total.
    weekly: np.ndarray
    #: `(sims, weeks, teams)` every franchise's total, held so a removal only has to
    #: re-score the one column it changes.
    scores: np.ndarray = field(repr=False)
    #: `(sims, teams)` lineup-efficiency multipliers, drawn once so two scenarios meet
    #: the same opposing managers.
    factors: np.ndarray = field(repr=False)
    #: player_id -> (modal starting slot id, share of remaining weeks he starts).
    starting: Mapping[int, tuple[int, float]] = field(repr=False)
    #: Uniforms for what each empty seat streams, fixed per `(seed, n_sims)` so two
    #: candidate rosters meet the same football AND the same wire. Without it an empty
    #: seat is paid its mean and carries no variance at all, and on these rosters
    #: 15.7-19.6% of slot-weeks are empty -- one slot is empty 71-88% of the time. The
    #: level was right and the SPREAD was missing: threading this raises the team's
    #: weekly SD by 5.9-6.8% and its season SD by 5.6-6.0%, which is most of what decides
    #: a bracket. `decide/title.py` has done this since the stochastic floor landed;
    #: this module and `decide/waivers` were the two that never picked it up.
    noise: S.FloorNoise | None = field(default=None, repr=False)
    #: The lineup the manager has actually submitted this week, captured at build time
    #: while the ESPN client is still open. Empty when it could not be read, in which
    #: case the start/sit surface is measured against a hypothetical optimum and its
    #: `delta_title` is zero by construction rather than by measurement.
    current_starters: tuple[int, ...] = ()

    @property
    def state(self) -> S.LeagueState:
        return self.sim.state

    @property
    def draw(self) -> Draw:
        return self.sim.draw

    @property
    def league_id(self) -> int:
        return self.sim.state.league_id

    @property
    def name(self) -> str:
        return self.sim.state.name

    @property
    def season(self) -> int:
        return self.sim.state.season

    @property
    def n_sims(self) -> int:
        return int(self.champions.size)

    @property
    def title(self) -> float:
        """P(championship) for the user's franchise, on this module's floor convention."""
        return float(self.champions.mean())

    @property
    def roster(self) -> tuple[int, ...]:
        return self.state.franchise(self.team_id).player_ids

    def owner_of(self, player_id: int) -> int | None:
        for f in self.state.franchises:
            if player_id in f.player_ids:
                return f.team_id
        return None

    def slot_of(self, player_id: int) -> tuple[int, float]:
        """The slot he starts in and how often, or `(bench, 0.0)` when he never does."""
        return self.starting.get(player_id, (SLOT_BENCH, 0.0))

    # -- counterfactuals ---------------------------------------------------------------

    def champions_without(self, player_ids: Sequence[int]) -> np.ndarray:
        """`(sims,)` title indicator with these players deleted for the rest of the season.

        Only the franchises that actually rostered one of them are re-scored; everybody
        else's weekly total is reused verbatim, so the per-simulation difference against
        `champions` isolates the loss instead of re-rolling the season. The whole league
        is re-stood regardless, because taking a player off one roster changes who his
        opponents beat and therefore who takes the last playoff seed -- which is most of
        the difference between "wins lost" and "title lost".

        Players nobody in this league rosters are ignored rather than raising: the caller
        is asking a portfolio-wide question ("what if this receiver tears an ACL") and two
        of the three leagues will not have him.
        """
        wanted = {int(p) for p in player_ids}
        state = self.state
        touched: dict[int, S.Franchise] = {}
        for f in state.franchises:
            hit = wanted.intersection(f.player_ids)
            if hit:
                touched[f.team_id] = f.with_players(p for p in f.player_ids if p not in hit)
        if not touched:
            return self.champions
        scores = self.scores.copy()
        reduced = state
        points, rank = S._as_points_and_rank(state, self.draw, None)
        rank_source = S._rank_tensor(points, rank, points.shape[1], points.shape[2])
        for team_id, franchise in touched.items():
            reduced = reduced.with_franchise(franchise)
            t = state.team_index[team_id]
            scores[:, :, t] = (
                self._column(franchise, points, rank_source) * self.factors[:, t : t + 1]
            )
        result = S.simulate_from_scores(reduced, scores, all_play=False)
        return result.champions[:, state.team_index[self.team_id]].astype(np.float64)

    def _column(
        self, franchise: S.Franchise, points: np.ndarray, rank_source: np.ndarray
    ) -> np.ndarray:
        """One franchise's `(sims, weeks)` starting total, before the efficiency factor."""
        plan = plan_from_slots(
            self.state.lineup_slot_counts,
            self.state.slot_eligibility,
            self.state.pool.positions_of(franchise.player_ids),
        )
        # The empty-group guard lives in `sim/season._floors`, and `_franchise_scores`
        # adds back what it holds out. This module used to carry its own copy of both.
        return S._franchise_scores(
            self.state.pool,
            franchise,
            plan,
            points,
            rank_source,
            self.replacement,
            floor_noise=None
            if self.noise is None
            else self.noise.for_plan(plan, self.state.team_index[franchise.team_id]),
        )


def _starting_shares(
    state: S.LeagueState,
    draw: Draw,
    franchise: S.Franchise,
    replacement: Mapping[int, WireLevel] | None,
) -> dict[int, tuple[int, float]]:
    """Which slot each rostered player starts in, and in what share of remaining weeks.

    Solved on the projected means with byes masked out -- the lineup a manager sets on
    Sunday morning -- rather than per simulation. The per-simulation answer differs only
    where an injury has already fired, and a roster's *exposure* is a statement about the
    plan, not about one drawn season. A player who never makes the lineup comes back as
    bench with a 0.0 share, which is the honest answer for a handcuff.
    """
    cols = state.pool.columns(franchise.player_ids)
    positions = state.pool.positions_of(franchise.player_ids)
    plan = plan_from_slots(state.lineup_slot_counts, state.slot_eligibility, positions)
    groups, _per_slot, _credit, _omitted = S._floors(plan, replacement)
    mean = np.asarray(draw.panel.mean, dtype=np.float64)[:, cols]
    playing = np.asarray(draw.panel.has_game)[:, cols]
    rank = np.where(playing, mean, -np.inf)
    assignment = plan.solve(rank, floor=groups, assignment=True).assignment
    assert assignment is not None  # assignment=True always populates it

    n_weeks = mean.shape[0]
    counts: dict[int, dict[int, int]] = {}
    for w in range(n_weeks):
        for i, local in enumerate(assignment[w]):
            if local < 0:
                continue
            pid = int(franchise.player_ids[int(local)])
            counts.setdefault(pid, {}).setdefault(plan.slot_ids[i], 0)
            counts[pid][plan.slot_ids[i]] += 1
    return {
        pid: (max(by_slot, key=lambda s: by_slot[s]), sum(by_slot.values()) / n_weeks)
        for pid, by_slot in counts.items()
    }


# --------------------------------------------------------------------------------------
# The portfolio
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Portfolio:
    """The user's three teams on one shared NFL season.

    Built by `build_portfolio`, which is also where the coupling is established: every
    league is drawn with the SAME seed, and `WeeklySampler` keys each player's random
    stream on `(seed, player_id)`, so the leagues share their football. Two portfolios
    built with different seeds are two universes and their numbers must not be
    differenced against each other -- the same rule that holds inside one league.
    """

    stakes: tuple[LeagueStake, ...]
    seed: int
    n_sims: int

    def __post_init__(self) -> None:
        if not self.stakes:
            raise PortfolioError("a portfolio needs at least one league")
        sizes = {s.n_sims for s in self.stakes}
        if len(sizes) != 1:
            raise PortfolioError(
                f"the leagues were drawn at different simulation counts {sorted(sizes)}; "
                "the joint probabilities below are elementwise operations on the sim axis "
                "and there is no alignment between two different axes"
            )
        seeds = {s.draw.seed for s in self.stakes}
        if len(seeds) != 1:
            # Not an error: a decoupled portfolio is a legitimate control arm, and it is
            # what `Diversification` is measured against. It is a warning because it is
            # the one mistake that produces three perfectly plausible numbers and a
            # `P(>=1 title)` that is silently the independent calculation.
            log.warning(
                "the leagues were drawn from different seeds %s, so they are NOT on one "
                "NFL season and every joint probability here is an independent one; "
                "verify_coupling() will say so",
                sorted(seeds),
            )
        weeks = {s.state.weeks for s in self.stakes}
        if len(weeks) != 1:
            log.warning(
                "the leagues do not share a week axis (%s); the per-player random streams "
                "still line up but the weekly-score correlations below compare different "
                "calendars",
                sorted(weeks),
            )

    # -- the shared axis ---------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stakes)

    @property
    def champions(self) -> np.ndarray:
        """`(sims, leagues)` title indicators on one shared season."""
        return np.stack([s.champions for s in self.stakes], axis=1)

    @property
    def titles(self) -> tuple[float, ...]:
        return tuple(s.title for s in self.stakes)

    def stake(self, league_id: int) -> LeagueStake:
        for s in self.stakes:
            if s.league_id == league_id:
                return s
        raise PortfolioError(f"league {league_id} is not in this portfolio")

    # -- headline numbers ---------------------------------------------------------------

    def odds(self) -> PortfolioOdds:
        return portfolio_odds(self.champions, names=self.names)

    def correlations(self) -> Correlations:
        return correlations(self)

    def exposures(self, *, min_leagues: int = 1) -> tuple[Exposure, ...]:
        return exposures(self, min_leagues=min_leagues)

    def verify_coupling(self) -> tuple[CouplingCheck, ...]:
        return verify_coupling(self)


def build_portfolio(
    leagues: Sequence[tuple[int, int]],
    season: int,
    *,
    seed: int = DEFAULT_SEED,
    n_sims: int = 4000,
    client: EspnClient | None = None,
    stream_replacement: bool = True,
    efficiency: S.LineupEfficiency | None = None,
    **build_kwargs: object,
) -> Portfolio:
    """Build every league onto one shared NFL season. `leagues` is `(league_id, team_id)`.

    The single seed is the load-bearing argument and it is not a convenience: it is what
    makes simulation `s` in one league the same football as simulation `s` in another, and
    therefore what makes `P(at least one title)` a measurement rather than an assumption.
    `Portfolio.verify_coupling` checks it held.

    `stream_replacement` fits `decide/title.streaming_replacement` for each league, so an
    unfilled starting slot streams a body off the wire instead of scoring zero. Leave it
    on unless you are reconciling against `pipeline.championship_table`, whose levels this
    deliberately does not match -- see the module docstring. It is fitted off
    `sim.outlooks`; passing the state and draw alone reads a pool with no free agents in
    it and silently returns the VOLS roster-bottom rank instead of the wire.

    The manager's currently submitted lineup is captured here and nowhere else, because
    `pipeline.build` closes the client it opened and `League.rosters` is not memoised, so
    by the time the action queue runs there is nothing left to ask. Without it
    `decide/lineups` prices its advice against a hypothetical projection-optimal lineup
    and returns exactly zero by construction, which is not the same statement as "your
    lineup is already right".
    """
    if not leagues:
        raise PortfolioError("no leagues given")
    from ..decide.title import streaming_levels

    own_client = client is None
    client = client or client_from_env()
    stakes: list[LeagueStake] = []
    try:
        for league_id, team_id in leagues:
            sim = build(
                int(league_id),
                season,
                my_team_id=int(team_id),
                client=client,
                n_sims=n_sims,
                seed=seed,
                **build_kwargs,  # type: ignore[arg-type]
            )
            # `outlooks=` is the WIRE, and it is not optional -- see the warning
            # `title.streaming_levels` addresses to this caller by name. Without it the
            # pool comes from the panel, `pipeline.build` pools only ROSTERED players,
            # and every slot falls through to the VOLS roster-bottom rank.
            floors = (
                streaming_levels(sim.state, sim.draw, outlooks=sim.outlooks)
                if stream_replacement
                else None
            )
            stakes.append(
                _stake(
                    sim,
                    int(team_id),
                    floors,
                    efficiency,
                    current_starters=_submitted_lineup(sim, int(team_id)),
                )
            )
    finally:
        if own_client:
            client.close()
    return Portfolio(stakes=tuple(stakes), seed=seed, n_sims=n_sims)


def _submitted_lineup(sim: LeagueSim, team_id: int) -> tuple[int, ...]:
    """The starters the manager has actually set, or `()` if ESPN would not say.

    A missing lineup is a degraded start/sit surface, not a failed portfolio, so this
    swallows the error and says so in the log rather than sinking a build over a
    reporting nicety.
    """
    from ..decide.lineups import starters_from_roster

    try:
        return starters_from_roster(sim.league.roster(team_id))
    except Exception as err:  # noqa: BLE001 - a lineup read must not fail a build
        log.info(
            "could not read team %d's submitted lineup in league %s (%s); the start/sit "
            "surface will be priced against the projection-optimal lineup instead",
            team_id,
            sim.state.league_id,
            err,
        )
        return ()


def _stake(
    sim: LeagueSim,
    team_id: int,
    replacement: Mapping[int, WireLevel] | None,
    efficiency: S.LineupEfficiency | None,
    *,
    current_starters: Sequence[int] = (),
) -> LeagueStake:
    """Score one league's baseline season once, and keep everything a counterfactual needs."""
    state = sim.state
    if team_id not in state.team_index:
        raise PortfolioError(f"team {team_id} is not in league {state.league_id}")
    eff = efficiency if efficiency is not None else S.LineupEfficiency()
    factors = eff.draw(state, sim.draw.n_sims)
    # Built once per stake, never per call: it is common random numbers for the wire, and
    # a fresh draw per counterfactual would put the noise back into every difference.
    noise = S.FloorNoise(state, sim.draw) if S.has_spread(replacement) else None
    points, rank = S._as_points_and_rank(state, sim.draw, None)
    rank_source = S._rank_tensor(points, rank, points.shape[1], points.shape[2])

    scores = np.zeros((sim.draw.n_sims, len(state.weeks), state.size), dtype=np.float32)
    stub = LeagueStake(
        sim=sim,
        team_id=team_id,
        team_name=state.franchise(team_id).name,
        replacement=replacement,
        noise=noise,
        champions=np.zeros(sim.draw.n_sims),
        weekly=np.zeros((sim.draw.n_sims, len(state.weeks))),
        scores=scores,
        factors=factors,
        starting={},
    )
    for t, franchise in enumerate(state.franchises):
        scores[:, :, t] = stub._column(franchise, points, rank_source) * factors[:, t : t + 1]
    result = S.simulate_from_scores(state, scores, all_play=False)
    t = state.team_index[team_id]
    return LeagueStake(
        sim=sim,
        team_id=team_id,
        team_name=state.franchise(team_id).name,
        replacement=replacement,
        noise=noise,
        champions=result.champions[:, t].astype(np.float64),
        weekly=scores[:, :, t].astype(np.float64),
        scores=scores,
        factors=factors,
        starting=_starting_shares(state, sim.draw, state.franchise(team_id), replacement),
        current_starters=tuple(int(p) for p in current_starters),
    )


# --------------------------------------------------------------------------------------
# Portfolio odds
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PortfolioOdds:
    """What playing three leagues is actually worth, and what it is not.

    The two numbers a multi-league player wants are `p_at_least_one` and `p_zero`, and
    neither is obtainable from the three individual probabilities without knowing how
    they covary. `sum_bound` and `max_bound` are the Frechet bounds -- the best and worst
    `p_at_least_one` any dependence structure could produce at these marginals -- so a
    reader can see immediately how much of the range is even in play.

    `expected_titles` is the one aggregate that *is* additive: `E[N] = sum p_i` by
    linearity of expectation, exactly, for any dependence whatsoever. That is not a
    convenient approximation, it is why the action queue can rank across leagues at all.
    """

    names: tuple[str, ...]
    titles: tuple[float, ...]
    p_at_least_one: float
    p_zero: float
    p_two_plus: float
    expected_titles: float
    variance_titles: float
    #: `1 - prod(1 - p_i)`: what `p_at_least_one` would be if the leagues were independent.
    independent: float
    #: `min(1, sum p_i)` and `max p_i`: the Frechet bounds on `p_at_least_one`.
    sum_bound: float
    max_bound: float
    #: Monte Carlo standard error on the level of `p_at_least_one`.
    stderr: float
    #: `independent - p_at_least_one`: what the dependence costs. Bootstrapped over the
    #: shared simulation axis, which is far tighter than the error on either level.
    dependence_cost: float
    dependence_cost_stderr: float
    n_sims: int

    @property
    def dependence_significant(self) -> bool:
        """Whether the leagues can be distinguished from independent at all."""
        return abs(self.dependence_cost) > 2.0 * self.dependence_cost_stderr

    def table(self) -> str:
        rows = [
            f"{'league':24s} {'P(title)':>9s}",
            "-" * 34,
        ]
        rows += [
            f"{n[:24]:24s} {p * 100:8.2f}%" for n, p in zip(self.names, self.titles, strict=True)
        ]
        rows += [
            "",
            f"P(>=1 title)        {self.p_at_least_one * 100:7.2f}%  +/- {self.stderr * 100:.2f}pp",
            f"P(0 titles)         {self.p_zero * 100:7.2f}%",
            f"P(>=2 titles)       {self.p_two_plus * 100:7.2f}%",
            f"E[titles]           {self.expected_titles:7.4f}   "
            f"(= sum of the column above, exactly, for any dependence)",
            f"Var[titles]         {self.variance_titles:7.4f}",
            "",
            f"if independent      {self.independent * 100:7.2f}%",
            f"naive sum           {self.sum_bound * 100:7.2f}%  (Frechet upper bound)",
            f"perfectly coupled   {self.max_bound * 100:7.2f}%  (Frechet lower bound)",
            f"dependence costs    {self.dependence_cost * 100:+7.3f}pp "
            f"+/- {self.dependence_cost_stderr * 100:.3f}pp -- "
            + ("real" if self.dependence_significant else "INSIDE its own error, so unmeasurable"),
        ]
        return "\n".join(rows)


def portfolio_odds(
    champions: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 7,
) -> PortfolioOdds:
    """Joint outcomes from a `(sims, leagues)` matrix of title indicators.

    Everything here is arithmetic on the shared simulation axis, so the dependence is
    whatever the coupled draw produced and nothing is modelled a second time. The one
    number that needs care is `dependence_cost`: it is a difference between two
    quantities computed from the *same* simulations, so its error is much smaller than
    the binomial error on either, and a bootstrap over the sim index is the cheapest
    honest way to get it. Reported with a naive standard error it looks significant on
    every portfolio; reported properly it is significant on almost none.
    """
    c = np.asarray(champions, dtype=np.float64)
    if c.ndim != 2:
        raise PortfolioError(f"champions must be (sims, leagues), got shape {c.shape}")
    n_sims, n_leagues = c.shape
    labels = tuple(names) if names is not None else tuple(f"league {i}" for i in range(n_leagues))
    if len(labels) != n_leagues:
        raise PortfolioError(f"{len(labels)} names for {n_leagues} leagues")

    p = c.mean(axis=0)
    count = c.sum(axis=1)
    any_ = (count > 0).astype(np.float64)
    at_least_one = float(any_.mean())
    independent = float(1.0 - np.prod(1.0 - p))

    rng = np.random.default_rng(seed)
    diffs = np.empty(max(bootstrap, 0))
    for b in range(diffs.size):
        idx = rng.integers(0, n_sims, n_sims)
        sample = c[idx]
        diffs[b] = (1.0 - np.prod(1.0 - sample.mean(axis=0))) - float(
            (sample.sum(axis=1) > 0).mean()
        )
    cost_stderr = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0

    return PortfolioOdds(
        names=labels,
        titles=tuple(p.tolist()),
        p_at_least_one=at_least_one,
        p_zero=float(1.0 - at_least_one),
        p_two_plus=float((count >= 2).mean()),
        expected_titles=float(count.mean()),
        variance_titles=float(count.var(ddof=1)) if n_sims > 1 else 0.0,
        independent=independent,
        sum_bound=float(min(1.0, p.sum())),
        max_bound=float(p.max()),
        stderr=float(math.sqrt(max(at_least_one * (1.0 - at_least_one), 0.0) / n_sims)),
        dependence_cost=independent - at_least_one,
        dependence_cost_stderr=cost_stderr,
        n_sims=n_sims,
    )


# --------------------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairCorrelation:
    """How two of the user's teams move together, at three levels of aggregation.

    **Every one of the three carries a standard error, and the third one needs it.**
    The weekly and season correlations are estimated off thousands of sim-weeks and are
    stable to the third decimal: on the user's leagues the Wine/Blacksburg weekly figure
    comes back +0.404, +0.407, +0.401, +0.404, +0.403 across five seeds. The
    *championship* correlation is a phi coefficient between two indicators that are 1 in
    three to seven per cent of seasons, and on those same five seeds the same pair reads
    +0.065, +0.074, +0.049, +0.024, +0.026 and the Blacksburg/Type-shi pair reads -0.022,
    +0.005, +0.030, +0.014, -0.029 -- it changes sign. Printed to three decimals with no
    error bar, a reader takes "-0.022" for a measured negative dependence between two of
    his teams; it is a draw from a distribution centred near zero. So the error bars are
    bootstrapped over the shared simulation axis and `champion_significant` says which of
    them is distinguishable from zero at all. On the live portfolio, one of three is.
    """

    a: str
    b: str
    shared_players: tuple[str, ...]
    #: Correlation of the two teams' weekly starting totals, pooled over sim-weeks.
    weekly: float
    #: Correlation of their season points-for totals. Higher than weekly, because a
    #: shared player's season is one draw of his injury clock rather than seventeen.
    season: float
    #: Phi coefficient between the two championship indicators. This is the one that
    #: feeds `P(>=1 title)`, and it is where the other two go to die.
    champion: float
    #: Bootstrap standard errors over the sim axis. Monte Carlo only -- they say how
    #: well this draw pins the number down, not how well the projections do.
    weekly_stderr: float = 0.0
    season_stderr: float = 0.0
    champion_stderr: float = 0.0

    @property
    def champion_significant(self) -> bool:
        """Whether the title-level dependence is distinguishable from zero at all."""
        return self.champion_stderr > 0.0 and abs(self.champion) > 2.0 * self.champion_stderr


@dataclass(frozen=True, slots=True)
class Correlations:
    """The correlation structure of the portfolio, measured rather than modelled.

    Nothing here is fitted. The three leagues were drawn against one NFL season through
    `sim/distributions`' block-diagonal correlation model, so these numbers are what that
    model plus the shared rosters plus the bracket actually produced.
    """

    pairs: tuple[PairCorrelation, ...]
    #: Correlation of my three teams' weekly scores within each remaining week, averaged
    #: over pairs. A spike is a week where the whole portfolio is exposed at once.
    by_week: Mapping[int, float]
    #: Bootstrap standard error on the *gap* between the most-correlated week and the
    #: runner-up, paired over one shared resample of the sim axis. Zero when there is
    #: no runner-up to compare against.
    worst_week_gap_stderr: float = 0.0

    @property
    def worst_week(self) -> tuple[int, float]:
        """The week where the three teams are most tied together."""
        if not self.by_week:
            return (0, 0.0)
        week = max(self.by_week, key=lambda w: self.by_week[w])
        return (week, self.by_week[week])

    @property
    def worst_week_separable(self) -> bool:
        """Whether the argmax week is actually distinguishable from the runner-up.

        It usually is not, and the report says so rather than naming a week. On the
        user's leagues weeks 9 and 11 sit within 0.01 of each other and the argmax
        flips between them from seed to seed; naming one of them as "the week the
        portfolio is most exposed" is reading a coin flip.
        """
        if len(self.by_week) < 2 or self.worst_week_gap_stderr <= 0.0:
            return False
        top, runner = sorted(self.by_week.values(), reverse=True)[:2]
        return (top - runner) > 2.0 * self.worst_week_gap_stderr

    def table(self) -> str:
        head = (
            f"{'pair':40s} {'shared':>7s} {'weekly':>8s} {'season':>8s} "
            f"{'title':>8s} {'+/-':>7s} {'title!=0':>9s}"
        )
        lines = [head, "-" * len(head)]
        for pair in self.pairs:
            lines.append(
                f"{(pair.a + ' / ' + pair.b)[:40]:40s} {len(pair.shared_players):7d} "
                f"{pair.weekly:+8.3f} {pair.season:+8.3f} {pair.champion:+8.3f} "
                f"{pair.champion_stderr:7.3f} "
                f"{'yes' if pair.champion_significant else 'NO':>9s}"
            )
        if any(not p.champion_significant for p in self.pairs):
            lines.append(
                "  a 'NO' in the last column is a title correlation inside twice its own "
                "Monte Carlo error: it changes sign from seed to seed, so read it as zero."
            )
        week, value = self.worst_week
        if self.worst_week_separable:
            lines.append(f"  most-correlated week: {week} at {value:+.3f} mean pairwise")
        else:
            ordered = sorted(self.by_week.items(), key=lambda kv: -kv[1])[:3]
            near = ", ".join(f"w{w} {v:+.3f}" for w, v in ordered)
            lines.append(
                f"  most-correlated week is NOT separable ({near}; gap +/- "
                f"{self.worst_week_gap_stderr:.3f}) -- the argmax is noise, the level is not"
            )
        return "\n".join(lines)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() <= 0 or y.std() <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _corr_stats(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`(sims, 6)` per-simulation sufficient statistics for `corr(x, y)`.

    A correlation is a function of six sums, and every one of them is a sum over the
    simulation axis. Holding them per simulation turns a bootstrap over whole seasons
    -- the only resample that respects the fact that a team's seventeen weeks are one
    football season and not seventeen independent observations -- into a matrix product
    against a weight matrix, which is what makes error bars on 3 pairs x 17 weeks cheap
    enough to compute every time rather than never.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    return np.stack(
        [
            np.full(a.shape[0], float(a.shape[1])),
            a.sum(axis=1),
            b.sum(axis=1),
            (a * a).sum(axis=1),
            (b * b).sum(axis=1),
            (a * b).sum(axis=1),
        ],
        axis=1,
    )


def _corr_from_totals(totals: np.ndarray) -> np.ndarray:
    """Correlation(s) from `(..., 6)` summed sufficient statistics."""
    t = np.asarray(totals, dtype=np.float64)
    n, sx, sy, sxx, syy, sxy = (t[..., k] for k in range(6))
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx * sx, 0.0) * np.maximum(n * syy - sy * sy, 0.0))
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)


def _bootstrap_weights(n_sims: int, resamples: int, seed: int) -> np.ndarray:
    """`(resamples, sims)` multinomial counts: one nonparametric resample per row.

    Equivalent to drawing `n_sims` simulation indices with replacement, and shared
    across every correlation in one `Correlations` so that differences between them --
    which is what `worst_week_separable` tests -- are paired rather than independent.
    """
    rng = np.random.default_rng(seed)
    p = np.full(n_sims, 1.0 / n_sims)
    return rng.multinomial(n_sims, p, size=max(resamples, 0)).astype(np.float64)


def correlations(
    portfolio: Portfolio, *, bootstrap: int = CORR_BOOTSTRAP, seed: int = 13
) -> Correlations:
    """Pairwise correlation of the user's teams, at weekly, season and title level.

    The three levels are reported together because they disagree by an order of
    magnitude and only the disagreement is informative. A shared player couples two
    weekly scores strongly and two *titles* almost not at all, because a title is a rank
    statistic against eleven or thirteen rivals and the shared player is a small share of
    the variance that decides it.

    Every figure carries a bootstrap standard error over the shared simulation axis,
    because the title-level number is the one this module exists to report and it is
    also the one small enough to be mistaken for a measurement when it is not. See
    `PairCorrelation`.
    """
    stakes = portfolio.stakes
    n_sims = stakes[0].n_sims
    weights = _bootstrap_weights(n_sims, bootstrap, seed) if bootstrap > 0 else None

    def measure(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray | None]:
        stats = _corr_stats(x, y)
        point = float(_corr_from_totals(stats.sum(axis=0)))
        if weights is None or weights.size == 0:
            return point, 0.0, None
        draws = _corr_from_totals(weights @ stats)
        return point, float(np.std(draws, ddof=1)), draws

    pairs: list[PairCorrelation] = []
    for i in range(len(stakes)):
        for j in range(i + 1, len(stakes)):
            a, b = stakes[i], stakes[j]
            shared = sorted(set(a.roster) & set(b.roster))
            weekly, weekly_se, _ = measure(a.weekly, b.weekly)
            season, season_se, _ = measure(a.weekly.sum(axis=1), b.weekly.sum(axis=1))
            champ, champ_se, _ = measure(a.champions, b.champions)
            pairs.append(
                PairCorrelation(
                    a=a.name,
                    b=b.name,
                    shared_players=tuple(a.state.pool.name(p) for p in shared),
                    weekly=weekly,
                    season=season,
                    champion=champ,
                    weekly_stderr=weekly_se,
                    season_stderr=season_se,
                    champion_stderr=champ_se,
                )
            )

    by_week: dict[int, float] = {}
    gap_stderr = 0.0
    weeks = stakes[0].state.weeks
    if all(s.state.weeks == weeks for s in stakes) and len(stakes) > 1:
        per_week_draws: list[np.ndarray] = []
        for w, _ in enumerate(weeks):
            values = []
            draws = []
            for i in range(len(stakes)):
                for j in range(i + 1, len(stakes)):
                    point, _, d = measure(stakes[i].weekly[:, w], stakes[j].weekly[:, w])
                    values.append(point)
                    if d is not None:
                        draws.append(d)
            by_week[weeks[w]] = float(np.mean(values)) if values else 0.0
            if draws:
                per_week_draws.append(np.mean(draws, axis=0))
        if len(per_week_draws) > 1:
            # The gap between the two weeks the point estimate ranked first and second,
            # resample by resample. Paired -- every week rode the same bootstrap weights
            # -- which is the whole reason the argmax turns out not to be separable.
            order = sorted(range(len(per_week_draws)), key=lambda k: -by_week[weeks[k]])
            top, runner = per_week_draws[order[0]], per_week_draws[order[1]]
            gap_stderr = float(np.std(top - runner, ddof=1))
    return Correlations(pairs=tuple(pairs), by_week=by_week, worst_week_gap_stderr=gap_stderr)


@dataclass(frozen=True, slots=True)
class CouplingCheck:
    """Evidence that two leagues really were drawn against the same NFL season.

    Asserted by `sim/distributions`' docstring and checked here, because everything this
    module says about joint probabilities is false if it is not true, and it fails
    silently: two leagues built with different seeds still produce three plausible
    numbers and a `P(>=1 title)` that is simply the independent one.
    """

    a: str
    b: str
    shared_pool: int
    #: Median correlation of a shared player's season total across the two panels. The
    #: block-diagonal repair depends on which team-mates are in each panel, so this is
    #: near one rather than exactly one.
    median_player_correlation: float
    min_player_correlation: float
    #: Share of shared player-weeks whose availability mask matches. Injuries are drawn
    #: from the same `(seed, player_id)` stream, so this should be exactly 1.0.
    availability_match: float
    #: Whether the two draws were made from the same seed. This is the *cause* of the
    #: coupling and the measurements above are the evidence for it; both are reported
    #: because two panels with no player in common have no evidence to offer.
    same_seed: bool = True

    @property
    def coupled(self) -> bool:
        """Whether these two leagues are on one NFL season.

        With no shared player there is nothing to measure and the seed is the whole
        answer: the leagues still meet the same football, they just have no observable
        in common to prove it on. Reporting `False` there would call a correctly built
        portfolio broken; reporting `True` on a seed mismatch would be worse.
        """
        if not self.same_seed:
            return False
        if self.shared_pool == 0:
            return True
        return self.median_player_correlation > 0.9 and self.availability_match > 0.99


def verify_coupling(portfolio: Portfolio) -> tuple[CouplingCheck, ...]:
    """Prove the leagues share a season, on the shared players they actually have.

    Run this once on a new portfolio. The alternative is trusting that `build_portfolio`
    passed one seed to three `WeeklySampler`s and that a sampler keyed its streams the
    way its docstring says -- and the failure mode of that trust is not a crash, it is a
    `P(>=1 title)` that quietly equals the independent calculation.
    """
    out: list[CouplingCheck] = []
    stakes = portfolio.stakes
    for i in range(len(stakes)):
        for j in range(i + 1, len(stakes)):
            a, b = stakes[i], stakes[j]
            same_seed = a.draw.seed == b.draw.seed
            shared = sorted(
                set(a.draw.panel.player_ids.tolist()) & set(b.draw.panel.player_ids.tolist())
            )
            if not shared:
                out.append(
                    CouplingCheck(
                        a=a.name,
                        b=b.name,
                        shared_pool=0,
                        median_player_correlation=float("nan"),
                        min_player_correlation=float("nan"),
                        availability_match=float("nan"),
                        same_seed=same_seed,
                    )
                )
                continue
            ia = a.draw.panel.index_of(shared)
            ib = b.draw.panel.index_of(shared)
            ta = a.draw.points[:, :, ia].sum(axis=1, dtype=np.float64)
            tb = b.draw.points[:, :, ib].sum(axis=1, dtype=np.float64)
            cors = np.array([_corr(ta[:, k], tb[:, k]) for k in range(len(shared))])
            match = float((a.draw.available[:, :, ia] == b.draw.available[:, :, ib]).mean())
            out.append(
                CouplingCheck(
                    a=a.name,
                    b=b.name,
                    shared_pool=len(shared),
                    median_player_correlation=float(np.median(cors)),
                    min_player_correlation=float(cors.min()),
                    availability_match=match,
                    same_seed=same_seed,
                )
            )
    return tuple(out)


# --------------------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Holding:
    """One league's stake in one player."""

    league_id: int
    league_name: str
    team_id: int
    #: The slot he most often starts in; `SLOT_BENCH` when he does not start.
    slot_id: int
    #: Share of remaining weeks he is in the ex-ante starting lineup.
    start_share: float
    #: This league's P(title) lost if he vanishes for the season, re-simulated.
    title_added: float
    title_added_stderr: float

    @property
    def slot(self) -> str:
        return SLOT_ABBREV.get(self.slot_id, str(self.slot_id))

    @property
    def starts(self) -> bool:
        return self.slot_id != SLOT_BENCH and self.start_share > 0.0


@dataclass(frozen=True, slots=True)
class Exposure:
    """One player, across every league the user holds him in.

    **Not additive across players.** `equity_at_risk` is a one-term Shapley
    approximation -- the marginal value of this player against exactly one coalition,
    the full roster -- for the same reason `sim/season.PlayerContribution` is. Removing
    the WR1 promotes the WR4, so summing the receiving corps badly overstates what it is
    worth. This answers "what happens if I lose him", which is the question an injury
    asks; it does not divide credit for a roster.

    **`portfolio_damage` is NOT bounded above by `equity_at_risk`, and it would be a
    mistake to assert that it is.** The tempting union bound -- losing a player cannot
    cost the union more than it costs the leagues separately -- needs the post-removal
    win set to be a subset of the pre-removal one, and it is not: `champions_without`
    re-stands the whole league, so a season in which the removal reshuffles the last
    playoff seed can hand the user a title he would not otherwise have won. When that
    happens in a season he was *already* winning another league in, the union is
    unchanged while the per-league term goes negative, and the damage exceeds the sum.
    On the user's live portfolio at 4,000 simulations this happens for eleven of the
    forty rostered players, by 0.025 to 0.125pp -- one to five simulations each. It is
    small, it is real, and it is not a sign of a misaligned simulation axis.
    """

    player_id: int
    name: str
    position_id: int
    pro_team_id: int
    holdings: tuple[Holding, ...]
    #: Sum over leagues of the title probability he is holding up. In units of
    #: E[titles], which is the only cross-league aggregate that is additive.
    equity_at_risk: float
    #: `equity_at_risk` as a share of the portfolio's whole `E[titles]`.
    equity_share: float
    #: `P(>=1 title)` today minus `P(>=1 title)` with him gone from every roster,
    #: re-simulated jointly rather than combined from the per-league numbers.
    portfolio_damage: float
    portfolio_damage_stderr: float

    @property
    def position(self) -> str:
        return POSITION_ABBREV.get(self.position_id, str(self.position_id))

    @property
    def n_leagues(self) -> int:
        return len(self.holdings)

    @property
    def n_starting(self) -> int:
        return sum(1 for h in self.holdings if h.starts)

    @property
    def significant(self) -> bool:
        return abs(self.portfolio_damage) > 2.0 * self.portfolio_damage_stderr

    def __lt__(self, other: Exposure) -> bool:
        return self.equity_at_risk < other.equity_at_risk


def exposures(portfolio: Portfolio, *, min_leagues: int = 1) -> tuple[Exposure, ...]:
    """Every player the user rosters anywhere, priced by what losing him would cost.

    One re-simulation per (player, league) pair, reusing the baseline scores for every
    franchise the removal does not touch. On the user's three leagues that is about fifty
    pairs and a few seconds.

    `min_leagues=2` restricts to the concentrated bets, which is usually what a reader
    wants: a player in one league is an ordinary roster spot, a player in three is a
    position taken three times without anybody deciding to take it.
    """
    base = portfolio.champions
    base_any = (base.sum(axis=1) > 0).astype(np.float64)
    expected = float(base.sum(axis=1).mean())

    everywhere: dict[int, list[LeagueStake]] = {}
    for stake in portfolio.stakes:
        for pid in stake.roster:
            everywhere.setdefault(int(pid), []).append(stake)

    out: list[Exposure] = []
    for pid, stakes in everywhere.items():
        if len(stakes) < min_leagues:
            continue
        holdings: list[Holding] = []
        columns = base.copy()
        for stake in stakes:
            after = stake.champions_without([pid])
            paired = stake.champions - after
            slot_id, share = stake.slot_of(pid)
            holdings.append(
                Holding(
                    league_id=stake.league_id,
                    league_name=stake.name,
                    team_id=stake.team_id,
                    slot_id=slot_id,
                    start_share=share,
                    title_added=float(paired.mean()),
                    title_added_stderr=float(paired.std(ddof=1) / math.sqrt(paired.size)),
                )
            )
            columns[:, portfolio.stakes.index(stake)] = after
        damage = base_any - (columns.sum(axis=1) > 0).astype(np.float64)
        at_risk = sum(h.title_added for h in holdings)
        first = stakes[0].state.pool
        cols = first.columns([pid])
        out.append(
            Exposure(
                player_id=pid,
                name=first.name(pid),
                position_id=int(first.position_ids[int(cols[0])]),
                pro_team_id=int(first.pro_team_ids[int(cols[0])]),
                holdings=tuple(holdings),
                equity_at_risk=at_risk,
                equity_share=at_risk / expected if expected > 0 else 0.0,
                portfolio_damage=float(damage.mean()),
                portfolio_damage_stderr=float(damage.std(ddof=1) / math.sqrt(damage.size)),
            )
        )
    return tuple(sorted(out, reverse=True))


# --------------------------------------------------------------------------------------
# Concentration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Concentration:
    """What one event would do to the whole portfolio, re-simulated for that event.

    No limit, no threshold, no "never more than X% in one player" -- there is no
    defensible constant, and a number that says *how much* damage beats a rule that says
    *too much*. `before` and `after` are both `P(>=1 title)`, so the row reads as one
    sentence: if this happens, the portfolio goes from A to B.
    """

    kind: str
    label: str
    #: The players the event removes, per league name.
    removed: Mapping[str, tuple[str, ...]]
    before: float
    after: float
    stderr: float
    #: `E[titles]` before and after, which moves for a different reason and by a
    #: different amount: it cannot see the dependence at all.
    expected_before: float
    expected_after: float
    #: For a GROUP row: the name of the single roster spot inside the group that does
    #: most of the damage, and what deleting only him costs. Empty for a one-player row.
    driver: str = ""
    driver_damage: float = 0.0
    #: `damage - driver_damage`, computed as a paired per-simulation difference rather
    #: than by subtracting two means, and its own standard error. This is the number a
    #: group row is worth reading: everything above `driver_damage` is what the *grouping*
    #: adds over its single biggest holding.
    increment: float = 0.0
    increment_stderr: float = 0.0

    @property
    def n_removed(self) -> int:
        """Roster spots deleted. A group row that deletes six is not a player row."""
        return sum(len(v) for v in self.removed.values())

    @property
    def damage(self) -> float:
        return self.before - self.after

    @property
    def significant(self) -> bool:
        return abs(self.damage) > 2.0 * self.stderr

    @property
    def increment_significant(self) -> bool:
        """Whether the group is worth more than its own biggest single holding."""
        return self.increment_stderr > 0.0 and abs(self.increment) > 2.0 * self.increment_stderr

    def __lt__(self, other: Concentration) -> bool:
        return self.damage < other.damage

    def describe(self) -> str:
        where = "; ".join(f"{k}: {', '.join(v)}" for k, v in self.removed.items())
        tail = "" if self.significant else "  (inside its own error)"
        head = (
            f"{self.kind} {self.label}: P(>=1 title) {self.before * 100:.2f}% -> "
            f"{self.after * 100:.2f}% ({-self.damage * 100:+.2f}pp +/- {self.stderr * 100:.2f}), "
            f"E[titles] {self.expected_before:.4f} -> {self.expected_after:.4f} [{where}]{tail}"
        )
        if self.n_removed > 1 and self.driver:
            verdict = (
                "the grouping is worth more than its driver"
                if self.increment_significant
                else "NOT separable from its driver alone"
            )
            head += (
                f"\n      {self.n_removed} roster spots; {self.driver} alone is "
                f"{self.driver_damage * 100:.2f}pp of it, the other "
                f"{self.n_removed - 1} add {self.increment * 100:+.2f}pp "
                f"+/- {self.increment_stderr * 100:.2f} -- {verdict}"
            )
        return head


def _removed_columns(
    portfolio: Portfolio, players: Mapping[int, Sequence[int]]
) -> tuple[np.ndarray, dict[str, tuple[str, ...]]]:
    """`(sims, leagues)` title indicators after the deletion, and what was deleted."""
    columns = portfolio.champions.copy()
    removed: dict[str, tuple[str, ...]] = {}
    for i, stake in enumerate(portfolio.stakes):
        ids = [p for p in players.get(stake.league_id, ()) if p in stake.roster]
        if not ids:
            continue
        columns[:, i] = stake.champions_without(ids)
        removed[stake.name] = tuple(stake.state.pool.name(p) for p in ids)
    return columns, removed


def _concentration(
    portfolio: Portfolio, kind: str, label: str, players: Mapping[int, Sequence[int]]
) -> Concentration:
    """Re-simulate the portfolio with `players` (per league id) deleted, and diff it."""
    base = portfolio.champions
    columns, removed = _removed_columns(portfolio, players)
    before_any = (base.sum(axis=1) > 0).astype(np.float64)
    after_any = (columns.sum(axis=1) > 0).astype(np.float64)
    paired = before_any - after_any
    return Concentration(
        kind=kind,
        label=label,
        removed=removed,
        before=float(before_any.mean()),
        after=float(after_any.mean()),
        stderr=float(paired.std(ddof=1) / math.sqrt(paired.size)) if paired.size > 1 else 0.0,
        expected_before=float(base.sum(axis=1).mean()),
        expected_after=float(columns.sum(axis=1).mean()),
    )


def _with_driver(
    portfolio: Portfolio, row: Concentration, players: Mapping[int, Sequence[int]]
) -> Concentration:
    """Attribute a group row to its single biggest holding, and price the remainder.

    **A group deletion is a superset of a player deletion, so ranking the two together
    and reporting that the group won is not a finding.** On the user's live portfolio
    the largest "NFL team" row is KC at -5.92pp and the largest player row is Chase
    Brown at -5.02pp, and the tempting sentence -- "the team concentration beats any
    single player even though no player is held three times" -- is arithmetic: KC
    deletes Kenneth Walker III out of two leagues (-4.93pp on its own) *and* Harrison
    Butker out of a third. What is actually worth knowing is the remainder, which is
    +1.00pp +/- 0.34 here and is a real if modest number; on PHI the same remainder is
    +0.00pp, because that row is Jalen Hurts and two players worth nothing.

    So each member is deleted on its own, the biggest is named as the `driver`, and the
    increment over him is taken as a *paired* per-simulation difference. One extra
    re-simulation per member, which is a few hundredths of a second each.
    """
    # A member is a PLAYER, deleted from every league in this group that holds him --
    # not a (league, player) roster spot. Kenneth Walker III is one holding taken twice,
    # and pricing his two seats separately would halve the driver and inflate the
    # remainder, which is the error this function exists to avoid making.
    by_player: dict[int, dict[int, tuple[int, ...]]] = {}
    for league_id, ids in players.items():
        for p in ids:
            by_player.setdefault(int(p), {})[league_id] = (int(p),)
    if row.n_removed < 2:
        return row
    base = portfolio.champions
    before_any = (base.sum(axis=1) > 0).astype(np.float64)
    group_cols, _ = _removed_columns(portfolio, players)
    group_any = (group_cols.sum(axis=1) > 0).astype(np.float64)

    best_name, best_damage, best_any = "", -math.inf, None
    for per_league in by_player.values():
        cols, named = _removed_columns(portfolio, per_league)
        if not named:
            continue
        any_ = (cols.sum(axis=1) > 0).astype(np.float64)
        damage = float((before_any - any_).mean())
        if damage > best_damage:
            best_name = next(iter(named.values()))[0]
            best_damage, best_any = damage, any_
    if best_any is None:  # pragma: no cover - n_removed >= 2 guarantees a member
        return row
    paired = best_any - group_any
    return replace(
        row,
        driver=best_name,
        driver_damage=best_damage,
        increment=float(paired.mean()),
        increment_stderr=(
            float(paired.std(ddof=1) / math.sqrt(paired.size)) if paired.size > 1 else 0.0
        ),
    )


def player_concentration(portfolio: Portfolio, exposure: Exposure) -> Concentration:
    """`if this player misses the season, P(>=1 title) falls from A to B`, re-simulated."""
    return _concentration(
        portfolio,
        "player",
        exposure.name,
        {h.league_id: (exposure.player_id,) for h in exposure.holdings},
    )


def pro_team_concentration(portfolio: Portfolio, *, limit: int = 5) -> tuple[Concentration, ...]:
    """What a whole NFL offence going dark would do, per NFL team, worst first.

    This is the correlated risk the shared-player view misses: the user can hold no
    player twice and still have four Chiefs across three leagues. Removing the group is a
    blunt instrument -- a real collapse is a degradation, not a deletion -- so read the
    number as the upper bound on that team's concentration rather than as a forecast.

    **These rows are NOT comparable with the player rows and must not be ranked against
    them.** A group row deletes between two and six roster spots and a player row deletes
    one, so a group row winning the combined ranking says nothing at all. Each row is
    therefore decomposed by `_with_driver` into its single biggest holding and the
    remainder, and it is the remainder that carries the information: on the live
    portfolio KC's -5.92pp is Kenneth Walker III's -4.93pp plus +1.00pp +/- 0.34 for
    Harrison Butker, while PHI's -3.90pp is Jalen Hurts and nothing else at all.
    """
    groups: dict[int, dict[int, list[int]]] = {}
    for stake in portfolio.stakes:
        pool = stake.state.pool
        for pid in stake.roster:
            team = int(pool.pro_team_ids[pool.index[pid]])
            if team <= 0:
                continue
            groups.setdefault(team, {}).setdefault(stake.league_id, []).append(pid)
    from ..espn.constants import FALLBACK_PRO_TEAM_ABBREV

    rows = [
        (
            _concentration(
                portfolio, "NFL team", FALLBACK_PRO_TEAM_ABBREV.get(team, str(team)), per
            ),
            per,
        )
        for team, per in groups.items()
        if sum(len(v) for v in per.values()) > 1
    ]
    rows.sort(key=lambda pair: -pair[0].damage)
    return tuple(_with_driver(portfolio, row, per) for row, per in rows[:limit])


@dataclass(frozen=True, slots=True)
class ByeExposure:
    """One NFL bye week, seen across all three teams at once.

    Byes reach the simulated season by two routes and this audit exists because for a
    long time only one of them was live. `SimPanel.from_outlooks` marks a bye as "no
    game" when it is handed a `byes=` table; `pipeline.build` now hands it one, so that
    route is the primary and correct one. The second route is the *projection*: ESPN's
    weekly numbers collapse a skill player's or kicker's bye to almost nothing, the
    calibrated outlook inherits it, and the lineup solver benches him because his rank
    that week is 0.06 rather than his usual 11.65. Measured on the user's Wine Wednesday
    roster, every quarterback, back, receiver, tight end and kicker's bye week projects
    **0.06 to 0.29 points against a season mean of 2.2 to 17.1**.

    **That second route never covered D/ST**, which is why relying on it was a bug: the
    Jaguars defence projects **3.28 points in week 7, its own bye**, against a season
    mean of 4.71. With no bye table, a defence on bye was started and scored, and every
    one of the user's three teams rosters exactly one D/ST. `unpriced` names any starter
    whose bye week keeps more than `BYE_ZERO_TOLERANCE` of his usual projection **and is
    still marked as playing**, so it reports a real gap rather than re-reporting the
    projection quirk the bye table already handles.

    There is deliberately no title-probability figure here. Converting one would mean
    removing a starter for one week from a season that already prices his bye at 0.06,
    which double-counts the ones that are handled and answers a question about a third
    season for the one that is not.
    """

    week: int
    #: league name -> starters whose NFL team is on bye this week.
    starters_out: Mapping[str, tuple[str, ...]]
    #: league name -> what those starters are worth in an ordinary week, summed.
    normal_points: Mapping[str, float]
    #: league name -> what the model projects them for in the bye week itself. The gap
    #: against `normal_points` is how much of the bye the projections already price.
    bye_points: Mapping[str, float]
    #: league name -> that team's mean simulated weekly total, for scale.
    weekly_mean: Mapping[str, float]
    #: `(league, player, bye projection, season mean)` for starters whose bye the
    #: projections did NOT zero. Empty is the good case.
    unpriced: tuple[tuple[str, str, float, float], ...]

    @property
    def total_out(self) -> int:
        return sum(len(v) for v in self.starters_out.values())

    @property
    def leagues_hit(self) -> int:
        return len(self.starters_out)

    @property
    def worst_share(self) -> float:
        """The largest share of one team's normal weekly scoring that is on bye."""
        return max(
            (
                self.normal_points[k] / self.weekly_mean[k]
                for k in self.normal_points
                if self.weekly_mean[k] > 0
            ),
            default=0.0,
        )

    def describe(self) -> str:
        parts = []
        for name, out in self.starters_out.items():
            share = self.normal_points[name] / max(self.weekly_mean[name], 1e-9)
            parts.append(
                f"{name}: {len(out)} out ({', '.join(out)}), "
                f"{self.normal_points[name]:.1f} of {self.weekly_mean[name]:.1f} weekly pts "
                f"({share * 100:.0f}%), priced at {self.bye_points[name]:.1f}"
            )
        tail = ""
        if self.unpriced:
            named = ", ".join(
                f"{who} ({lg}, {bye:.1f} vs {mean:.1f})" for lg, who, bye, mean in self.unpriced
            )
            tail = f"  [UNPRICED BYE: {named}]"
        return (
            f"week {self.week}: {self.total_out} starters across {self.leagues_hit} leagues; "
            + "; ".join(parts)
            + tail
        )


def bye_concentration(
    portfolio: Portfolio,
    *,
    byes: Mapping[int, int] | None = None,
    limit: int = 3,
    tolerance: float = BYE_ZERO_TOLERANCE,
) -> tuple[ByeExposure, ...]:
    """The weeks where the portfolio is thinnest, worst first by starters lost.

    `byes` is `proTeamId -> bye week`; left `None` it is read from ESPN's platform
    settings for the season, which are cached to disk and need no network after the first
    run. A portfolio whose bye table cannot be loaded comes back empty rather than
    guessing -- an invented bye week is worse than no bye analysis.

    The point of the cross-league view is the coincidence, not the count: one team losing
    three starters in week 7 is a lineup problem, and all three teams losing starters in
    week 7 is the only week the whole portfolio is thin at once.
    """
    stakes = portfolio.stakes
    weeks = stakes[0].state.weeks
    if not all(s.state.weeks == weeks for s in stakes):
        raise PortfolioError("the leagues do not share a week axis; bye alignment is undefined")
    if byes is None:
        from ..sim.distributions import espn_bye_weeks

        try:
            byes = espn_bye_weeks(stakes[0].season)
        except Exception as err:  # noqa: BLE001 - offline, or an unparsable payload
            log.warning("no bye table for %s (%s); skipping bye exposure", stakes[0].season, err)
            return ()

    out: list[ByeExposure] = []
    for w, week in enumerate(weeks):
        starters: dict[str, tuple[str, ...]] = {}
        normal: dict[str, float] = {}
        priced: dict[str, float] = {}
        weekly: dict[str, float] = {}
        unpriced: list[tuple[str, str, float, float]] = []
        for stake in stakes:
            panel = stake.draw.panel
            names: list[str] = []
            usual = 0.0
            here = 0.0
            for pid in stake.roster:
                if stake.slot_of(int(pid))[1] <= 0.0:
                    continue
                col = int(panel.index_of([int(pid)])[0])
                if byes.get(int(panel.pro_team_ids[w, col])) != week:
                    continue
                name = stake.state.pool.name(int(pid))
                # The season mean over the OTHER weeks, so a bye that is already priced
                # at zero does not drag down the "what he is normally worth" figure it is
                # being compared against.
                others = np.delete(np.asarray(panel.mean[:, col], dtype=np.float64), w)
                season_mean = float(others.mean()) if others.size else 0.0
                # What the SIM scores this week, not what the projection says. `has_game`
                # is the switch a bye actually flips, and `panel.mean` is written before
                # it -- so a properly modelled bye still carries a five-point mean here.
                # Reading the mean alone would make this audit fire forever on exactly the
                # defences it was written to catch, which is worse than not having it.
                bye_mean = float(panel.mean[w, col]) if bool(panel.has_game[w, col]) else 0.0
                names.append(name)
                usual += season_mean
                here += bye_mean
                if season_mean > 0 and bye_mean > tolerance * season_mean:
                    unpriced.append((stake.name, name, bye_mean, season_mean))
            if names:
                starters[stake.name] = tuple(names)
                normal[stake.name] = usual
                priced[stake.name] = here
                weekly[stake.name] = float(stake.weekly[:, w].mean())
        if starters:
            out.append(
                ByeExposure(
                    week=week,
                    starters_out=starters,
                    normal_points=normal,
                    bye_points=priced,
                    weekly_mean=weekly,
                    unpriced=tuple(unpriced),
                )
            )
    return tuple(sorted(out, key=lambda b: (-b.total_out, -b.worst_share))[:limit])


# --------------------------------------------------------------------------------------
# Diversification
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diversification:
    """Whether to hold the same player everywhere, answered twice because it has two answers.

    Under `E[titles]` the question is empty. `E[N] = sum p_i` holds for *any* joint
    distribution, so no rearrangement of who is correlated with whom changes it by a
    single basis point, and the right play is simply to hold the best player available in
    each league -- which, when the same man is best in all three, means concentrating
    without a second thought. `expected_titles_invariant` checks that numerically rather
    than asserting it: the measured `E[N]` and the sum of the marginals agree to machine
    precision on any coupled draw.

    Under `P(>=1 title)` it is not empty at all. At fixed marginals that probability is
    strictly decreasing in dependence, from `sum_bound` (the leagues never win together)
    down to `max_bound` (they always do). `concentration_headroom` is how much is left to
    lose and `diversification_headroom` how much is left to gain, so a reader can see
    whether the argument is worth having on his actual portfolio before having it.
    """

    p_at_least_one: float
    independent: float
    sum_bound: float
    max_bound: float
    expected_titles: float
    marginal_sum: float
    #: `p_at_least_one - max_bound`: everything full concentration could still take away.
    concentration_headroom: float
    #: `sum_bound - p_at_least_one`: everything perfect diversification could still add.
    diversification_headroom: float
    #: Measured `Var[N]` against the variance three independent leagues would have.
    variance_titles: float
    variance_independent: float
    #: What the shared players are costing right now, from `PortfolioOdds`.
    dependence_cost: float
    dependence_cost_stderr: float

    @property
    def expected_titles_invariant(self) -> bool:
        """E[N] equals the sum of the marginals exactly, whatever the dependence."""
        return math.isclose(self.expected_titles, self.marginal_sum, rel_tol=1e-9, abs_tol=1e-12)

    @property
    def verdict(self) -> str:
        if not self.expected_titles_invariant:  # pragma: no cover - arithmetic guarantee
            return "E[titles] does not match the sum of the marginals; the draw is misaligned"
        cost = self.dependence_cost
        measurable = abs(cost) > 2.0 * self.dependence_cost_stderr
        head = (
            "Maximising E[titles]: hold the best player in each league and ignore overlap "
            "entirely -- E[N] is a sum of marginals and cannot see dependence. "
            "Maximising P(>=1 title): prefer the less-correlated of two otherwise equal "
            "players, because P(>=1) falls as the leagues move together. "
        )
        size = (
            f"On this portfolio the overlap currently costs {cost * 100:+.3f}pp of "
            f"P(>=1 title) (+/- {self.dependence_cost_stderr * 100:.3f}pp), which is "
            + ("a real effect. " if measurable else "not distinguishable from zero. ")
        )
        room = (
            f"Full concentration would cost {self.concentration_headroom * 100:.2f}pp more and "
            f"perfect diversification would add {self.diversification_headroom * 100:.2f}pp, "
            f"so {self._position}."
        )
        return head + size + room

    @property
    def _position(self) -> str:
        """Where this portfolio actually sits between the two Frechet bounds.

        Read off `concentration_headroom` and `diversification_headroom` rather than
        asserted. The sentence used to end "the position is already near the diversified
        end" unconditionally, which is true of the user's live portfolio and false of any
        portfolio that is not: on two identical leagues the same string claimed a
        perfectly comonotone position was near the diversified end while printing
        0.00pp of concentration headroom directly in front of it.
        """
        span = self.concentration_headroom + self.diversification_headroom
        if span <= 0:
            return "the two bounds coincide and there is nothing to trade off"
        toward_diversified = self.concentration_headroom / span
        if toward_diversified >= 0.8:
            return "the position is already near the diversified end and the argument is small"
        if toward_diversified <= 0.2:
            return (
                "the position is near the CONCENTRATED end -- almost all of the available "
                "P(>=1 title) is still on the table"
            )
        return "the position is in the middle of the range and both directions are live"


def diversification(portfolio: Portfolio, *, odds: PortfolioOdds | None = None) -> Diversification:
    """The concentrate-or-diversify answer, computed on this portfolio's own draw."""
    o = odds or portfolio.odds()
    p = np.asarray(o.titles, dtype=np.float64)
    return Diversification(
        p_at_least_one=o.p_at_least_one,
        independent=o.independent,
        sum_bound=o.sum_bound,
        max_bound=o.max_bound,
        expected_titles=o.expected_titles,
        marginal_sum=float(p.sum()),
        concentration_headroom=o.p_at_least_one - o.max_bound,
        diversification_headroom=o.sum_bound - o.p_at_least_one,
        variance_titles=o.variance_titles,
        variance_independent=float((p * (1.0 - p)).sum()),
        dependence_cost=o.dependence_cost,
        dependence_cost_stderr=o.dependence_cost_stderr,
    )


# --------------------------------------------------------------------------------------
# The action queue
# --------------------------------------------------------------------------------------


def _tag_int(rec: Recommendation, prefix: str) -> int:
    """Read a `key:integer` tag off a recommendation, or 0."""
    for tag in rec.tags:
        if tag.startswith(prefix):
            try:
                return int(tag[len(prefix) :])
            except ValueError:
                return 0
    return 0


@functools.cache
def _selection_z(n_candidates: int) -> float:
    """`decide/trades.selection_threshold`, memoised. Not reimplemented -- imported."""
    from ..decide.trades import selection_threshold

    return selection_threshold(n_candidates)


@dataclass(frozen=True, slots=True)
class Candidates:
    """What one surface returned, and how many it chose from.

    `n_considered` is the size of the field the top row won, and it is the only thing
    that makes a cross-surface `delta_title` comparison honest. A waiver board confirms
    a dozen claims with a standard error of 0.013pp; a trade finder confirms forty
    packages with a standard error of 0.48pp. The maximum of forty draws of pure noise
    at 0.48pp is about 1.1pp all by itself, which is larger than every trade on the live
    board -- so a queue that ranks the two together by raw `delta_title` puts the
    noisiest surface on top by construction. See `QueueItem.significant`.

    A surface that returns a bare list of recommendations is taken at `len(list)`, which
    is a lower bound on its multiplicity and the right default for a custom surface.
    """

    recommended: tuple[Recommendation, ...]
    n_considered: int


@dataclass(frozen=True, slots=True)
class QueueItem:
    """One recommendation from one surface in one league, ready to be ranked against the rest."""

    league_id: int
    league_name: str
    team_id: int
    surface: str
    rec: Recommendation
    #: The subject team's P(title) under no move, **on this portfolio's own baseline and
    #: not on the surface's**. The two differ: every surface builds its own draw and its
    #: own floors, so `decide/waivers` reports a different level for the same team. It is
    #: here for scale on the delta, and no delta should ever be added to it.
    baseline_title: float
    #: How many candidates this surface ranked before handing over its best. See
    #: `Candidates`; 1 means the row was not selected out of a field.
    n_considered: int = 1

    @property
    def delta_title(self) -> float:
        return self.rec.delta_title

    @property
    def stderr(self) -> float:
        return self.rec.stderr

    @property
    def leverage(self) -> float:
        return self.rec.leverage

    @property
    def confidence(self) -> str:
        """The surface's own verdict on its own row, carried through rather than dropped."""
        return self.rec.confidence

    @property
    def selection_z(self) -> float:
        """Standard errors a *selected* delta has to clear, Bonferroni over the field.

        `decide/trades.selection_threshold`, imported rather than reimplemented, because
        the argument it makes is exactly the one this queue needs: `Recommendation.
        significant` is a two-sigma test and two sigma is the test for one pre-specified
        candidate, not for the candidate that won a search of forty.
        """
        return _selection_z(max(self.n_considered, 1))

    @property
    def significant(self) -> bool:
        """Whether the effect survives the field it was selected out of.

        Two corrections on `core.Recommendation.significant`, and the second one changes
        the live board. First, a zero standard error is read there as certainty, which is
        right for an analytic surface and wrong for a move that changed nothing -- both
        arms were bit-identical and nothing was measured -- so a zero delta is never
        significant here. Second, the threshold is `selection_z`, not 2.0.

        The difference is not cosmetic. On the user's leagues on 2026-09-07 the top two
        actionable rows on the whole board were Blacksburg trades at +1.225pp +/- 0.477
        and +1.125pp +/- 0.476, both "significant" at two sigma. They were selected out
        of forty confirmed packages, where the threshold is 3.23 sigma and the bar is
        1.54pp: neither clears it, `decide/trades` itself assigned both `confidence=
        "low"`, and both are consistent with zero. Every waiver claim on the same board
        clears its own threshold by a factor of five. Ranking by `delta_title` alone
        would have sent the user to negotiate two trades that are noise, ahead of nine
        waiver claims that are not.
        """
        if self.rec.delta_title == 0.0:
            return False
        if self.rec.stderr <= 0.0:
            return True
        return abs(self.rec.delta_title) > self.selection_z * self.rec.stderr

    @property
    def naive_significant(self) -> bool:
        """The uncorrected two-sigma answer, kept so the two can be compared."""
        return self.rec.delta_title != 0.0 and self.rec.significant

    @property
    def lower_bound(self) -> float:
        """`delta - 2 * stderr`: the risk-adjusted key, for a caller who wants one."""
        return self.rec.delta_title - 2.0 * self.rec.stderr

    @property
    def selection_lower_bound(self) -> float:
        """`delta - selection_z * stderr`: the same key, corrected for the field size.

        This is the honest cross-surface key. On the live board it is positive for every
        waiver claim and negative for every trade, which is the opposite of the order
        `delta_title` gives and is the reason both are reported.
        """
        return self.rec.delta_title - self.selection_z * self.rec.stderr

    @property
    def blockers(self) -> tuple[str, ...]:
        """Why this row is not simply "do it", in the surface's own words.

        `no-action-this-week` is the one that matters most and it is not cosmetic: a
        streaming plan prices the whole rest-of-season plan and its `move` is only the
        first week's action, which on all three live D/ST grids is a hold. Ranked naively
        it tops the queue at +2 to +4pp for doing nothing today.
        """
        tags = frozenset(self.rec.tags)
        out = sorted(tags & (NOT_ACTIONABLE_TAGS | CONTESTED_TAGS))
        if self.rec.move.kind is MoveKind.HOLD and "no-action-this-week" not in out:
            out.append("hold")
        return tuple(out)

    @property
    def actionable(self) -> bool:
        """Whether there is something to do about this row today."""
        return not (frozenset(self.rec.tags) & NOT_ACTIONABLE_TAGS) and (
            self.rec.move.kind is not MoveKind.HOLD
        )


@dataclass(frozen=True, slots=True)
class ActionQueue:
    """Every surface in every league, in one order.

    **What this ranks, and why it is the right ranking.** The three `delta_title`s are
    increments to three *different* probabilities, so the queue is not a probability and
    the numbers in it must never be added into one. What they are is increments to
    `E[titles] = sum_l P_l(title)`, and *that* is additive across leagues exactly, by
    linearity of expectation, with no independence assumption anywhere. So the queue
    ranks `d E[titles] / d action`: how much expected championship one action buys,
    wherever it happens. That is the correct greedy order for a user spending scarce
    attention, and it is correct whether or not the leagues are correlated -- the
    dependence would only matter if the objective were `P(>=1 title)`, and even then the
    measured correction on this portfolio (0.21pp) is smaller than the queue's fourth row.

    **The sort key is right and the sort key is not enough, and the second half of that
    sentence is the one a reader acts on.** `delta_title` is a common unit but it is not
    a common *precision*: each surface hands over the argmax of its own search, and the
    searches are wildly different sizes with wildly different standard errors. Measured
    on the user's three leagues on 2026-09-07:

    ==========  ==========  ==========  ==========  ===================================
    surface     candidates  best delta  its stderr  max of pure noise at that stderr
    ==========  ==========  ==========  ==========  ===================================
    waivers     12          +0.18pp     0.013pp     0.03pp
    trades      40          +1.23pp     0.477pp     1.54pp
    streaming   2           +3.83pp     0.469pp     1.05pp
    lineups     1           +0.00pp     0.000pp     --
    ==========  ==========  ==========  ==========  ===================================

    So the queue's raw ordering -- trades seven times a waiver claim -- is what the
    selection produced, not what the leagues contain: every trade on the live board is
    *below* the noise ceiling of its own search, and `decide/trades` marked every one of
    them `confidence="low"` before this module ever saw it. The waiver rows, an order of
    magnitude smaller, clear their own threshold five times over. `QueueItem.significant`
    therefore tests against `decide/trades.selection_threshold(n_considered)` rather than
    against 2.0, `by_selection_bound` ranks on the corrected key, and `table()` prints the
    field size next to every row so the comparison cannot be read without it.

    Two things are deliberately not folded into the sort key.

    `leverage` is a property of the *team's remaining schedule*, not of the move, and for
    a surface that prices a whole rest-of-season it is already inside `delta_title`.
    Multiplying it back in double-counts. `leverage_weight` exists for a caller who wants
    to try it, defaults to zero, and `reordered_by_leverage` reports whether it changed
    anything at all -- on the live leagues the three teams' mean leverage spans 0.947 to
    0.999 and weighting by it reorders not one row.

    Significance is reported, not sorted on. A big noisy trade genuinely might be the
    best move available; it is just not *known* to be. `by_lower_bound` gives the
    two-sigma risk-adjusted order and `by_selection_bound` the selection-adjusted one --
    and at the live sizes only the second one flips the board, because `2 * 0.477pp` does
    not price a field of forty and the top trade's two-sigma lower bound (+0.27pp) still
    beats the top waiver claim's (+0.16pp). Both are re-ranked over the whole board and
    then truncated, so a corrected key can actually promote a row from below the fold.
    """

    items: tuple[QueueItem, ...]
    limit: int
    leverage_weight: float
    #: Surfaces that raised rather than returning, as `(league, surface, error)`.
    failures: tuple[tuple[str, str, str], ...] = ()
    #: Every row before `limit` truncated the list. `reordered_by_leverage` reads this,
    #: because a reordering that pulls row 15 into the top ten is invisible in `items`.
    ranked: tuple[QueueItem, ...] = ()

    @property
    def actionable(self) -> tuple[QueueItem, ...]:
        return tuple(i for i in self.items if i.actionable)

    @property
    def significant(self) -> tuple[QueueItem, ...]:
        """Rows that clear their own selection-adjusted threshold. See `QueueItem`."""
        return tuple(i for i in self.items if i.significant)

    @property
    def by_lower_bound(self) -> tuple[QueueItem, ...]:
        """The same rows ranked by `delta - 2 * stderr` instead of by `delta`."""
        return self._reranked(lambda i: -i.lower_bound)

    @property
    def by_selection_bound(self) -> tuple[QueueItem, ...]:
        """Ranked by `delta - selection_z * stderr`: the honest cross-surface order.

        This is the order to act on when a user has one hour, because it is the only one
        of the three that charges a surface for the size of the field its top row won.

        Re-ranked over the WHOLE board and then truncated, not over the `limit` rows the
        raw key already chose. That distinction is the difference between the property
        working and not working: on the user's leagues the nine waiver claims sit at raw
        ranks 16 to 24, below eleven trades and streaming plans, so a re-ranking confined
        to a top-ten `items` would have had nothing to promote and would have reported
        the raw order back with a corrected label on it.
        """
        return self._reranked(lambda i: -i.selection_lower_bound)

    def _reranked(self, key: Callable[[QueueItem], float]) -> tuple[QueueItem, ...]:
        rows = self.ranked or self.items
        return tuple(sorted(rows, key=key)[: self.limit or len(rows)])

    @property
    def reordered_by_leverage(self) -> int:
        """How many rows change place if the key is weighted by leverage. Usually zero.

        Measured over `ranked` -- the whole board -- rather than over the truncated
        `items`, because weighting can promote a row from below the fold and a count
        taken inside the fold would report zero without having looked.
        """
        rows = self.ranked or self.items
        plain = sorted(rows, key=lambda i: -i.delta_title)
        weighted = sorted(rows, key=lambda i: -(i.delta_title * i.leverage))
        return sum(1 for a, b in zip(plain, weighted, strict=True) if a is not b)

    def selection_note(self) -> str:
        """One sentence on which surfaces survive the field they were selected out of."""
        selected = [i for i in (self.ranked or self.items) if i.n_considered > 1 and i.stderr > 0]
        if not selected:
            return ""
        lost = sorted({i.surface for i in selected if not i.significant and i.naive_significant})
        kept = sorted({i.surface for i in selected if i.significant})
        parts = []
        if kept:
            parts.append(f"clears its own field: {', '.join(kept)}")
        if lost:
            parts.append(
                f"passes a two-sigma test and FAILS the selection-adjusted one: {', '.join(lost)}"
            )
        return "  " + "; ".join(parts) if parts else ""

    def table(self, limit: int | None = None) -> str:
        head = (
            f"{'#':>2s} {'league':18s} {'surface':10s} {'dTitle':>9s} {'+/-':>7s} {'of':>4s} "
            f"{'z':>5s} {'lev':>5s} {'sig':>4s} {'conf':>6s} {'do now':>7s}  action"
        )
        lines = [head, "-" * (len(head) + 30)]
        for n, item in enumerate(self.items[: limit or self.limit], start=1):
            what = item.rec.rationale.split(".")[0][:70] or item.rec.move.kind.value
            lines.append(
                f"{n:2d} {item.league_name[:18]:18s} {item.surface:10s} "
                f"{item.delta_title * 100:+8.3f}pp {item.stderr * 100:6.3f} "
                f"{item.n_considered:4d} {item.selection_z:5.2f} "
                f"{item.leverage:5.2f} {'yes' if item.significant else 'NO':>4s} "
                f"{item.confidence[:6]:>6s} "
                f"{'yes' if item.actionable else 'no':>7s}  {what}"
            )
            if item.blockers:
                lines.append(f"{'':>2s} {'':18s} {'':10s} -> {', '.join(item.blockers)}")
        lines.append(
            "  'of' is how many candidates the surface ranked to produce this row and 'z' the "
            "standard errors a winner of that field must clear; 'sig' is tested against z, "
            "not against 2."
        )
        note = self.selection_note()
        if note:
            lines.append(note)
        for league, surface, err in self.failures:
            lines.append(f"   {league[:18]:18s} {surface:10s} FAILED: {err}")
        return "\n".join(lines)


#: The four surfaces, each reduced to `(portfolio stake) -> Recommendations`. Held as a
#: table rather than four branches so a caller can drop one that is slow, or add a fifth
#: without touching the merge. A surface may return a bare iterable, in which case its
#: field size is taken as the length of what it returned, or a `Candidates` when it knows
#: how many it actually ranked -- which is the honest number and usually the larger one.
Surface = Callable[[LeagueStake], "Iterable[Recommendation] | Candidates"]


def _waiver_recs(stake: LeagueStake) -> Candidates:
    """Waiver claims, with the size of the board they were picked off.

    `report.claims` is the above-threshold subset; `report.board` is every confirmed
    candidate, and that is the field the top claim won. Reporting `len(claims)` would
    understate the multiplicity by a factor of four on the live leagues -- though it
    barely matters here, because the waiver standard errors are 0.013pp and the claims
    clear even a forty-candidate threshold.
    """
    from ..decide.waivers import waiver_board

    # No analyst board here, deliberately. The portfolio ranks waiver rows against
    # streaming and lineup rows in one currency, and those two surfaces run on ESPN's
    # numbers; handing the wire a different projection set would rank a Silva-priced
    # claim against an ESPN-priced stream. `fq waivers` carries the board; this does not,
    # and the two will disagree about the wire until the rest of the portfolio does too.
    report = waiver_board(stake.sim, team_id=stake.team_id)
    recs = list(report.claims) or [report.hold]
    return Candidates(tuple(recs), max(len(report.board), len(recs), 1))


def _trade_recs(stake: LeagueStake) -> Candidates:
    """Confirmed trades, with the size of the confirmed field rather than of the winners.

    `find_trades` drops the packages the simulation says would *lower* this team's title
    probability, so the returned list is the positive tail of a much larger confirmed
    set -- eight of forty on Wine Wednesday, twenty-two of forty on Blacksburg. The
    multiplicity that matters is forty, because all forty competed for the top of the
    ranking, so the full list is asked for with `include_harmful=True` (the same
    computation, one filter fewer) and only the positive rows are handed on.
    """
    from ..decide.trades import find_trades

    # Same reason as `_waiver_recs`: one currency across the portfolio, so no board.
    every = list(find_trades(stake.sim, for_team=stake.team_id, include_harmful=True))
    positive = tuple(r for r in every if r.delta_title > 0.0)
    # The field is what was CONFIRMED, not what survived being published. `find_trades`
    # prunes a candidate when a strict subset of its moves is worth the same, which
    # happens after every one of them has already competed for the top of the list --
    # so `len(every)` understates the multiplicity and the correction would go soft on
    # precisely the surface it was built for. The count travels on the tag.
    considered = max((_tag_int(r, "considered:") for r in every), default=0)
    return Candidates(positive, max(considered, len(every), 1))


def _lineup_recs(stake: LeagueStake) -> list[Recommendation]:
    """The start/sit call, priced against the lineup the manager has actually submitted.

    `current=` is what makes the number mean anything. Without it the surface compares
    its recommendation to the projection-optimal lineup it just built, which is the same
    lineup, so it returns exactly 0.00pp in every league every week -- an answer that
    looks like "nothing to do" and is really "nothing was asked".
    """
    from ..decide.lineups import advise_sim

    advice = advise_sim(stake.sim, team_id=stake.team_id, current=stake.current_starters or None)
    return [advice.recommendation]


def _streaming_recs(stake: LeagueStake) -> list[Recommendation]:
    """D/ST and kicker plans. One position failing must not lose the other.

    The two grids are independent problems that happen to share a surface, and the
    kicker's is the fragile one -- it needs a market line the corpus does not always
    carry. Letting it take the defence's plan down with it would drop the largest
    `delta_title` on the whole board for a position the module itself calls not
    streamable.
    """
    from ..decide.streaming import recommend

    out: list[Recommendation] = []
    for position_id in (16, 5):
        try:
            out.append(recommend(stake.sim, position_id=position_id))
        except Exception as err:  # noqa: BLE001 - one grid must not sink the other
            log.warning("streaming position %d failed on %s: %s", position_id, stake.name, err)
    if not out:
        raise PortfolioError(f"no streaming grid could be built for {stake.name}")
    return out


DEFAULT_SURFACES: Mapping[str, Surface] = {
    "waivers": _waiver_recs,
    "trades": _trade_recs,
    "lineups": _lineup_recs,
    "streaming": _streaming_recs,
}


def action_queue(
    portfolio: Portfolio,
    *,
    limit: int = DEFAULT_QUEUE_LIMIT,
    surfaces: Mapping[str, Surface] | None = None,
    leverage_weight: float = 0.0,
    per_surface: int = 3,
    actionable_only: bool = False,
) -> ActionQueue:
    """Run every surface in every league and merge the output into one ranked list.

    `per_surface` caps how many rows one surface may contribute per league before the
    merge, which is the difference between a queue and a waiver board with two other
    leagues appended: `decide/waivers` alone returns a dozen claims, all of the same
    shape, and without a cap they crowd out the single trade or start/sit call that is
    the only reason to look across leagues at all.

    A surface that raises is recorded in `failures` and the rest of the queue is still
    produced. Losing the trade finder in one league is not a reason to withhold the
    waiver claim in another, and a queue that silently drops a surface is worse than one
    that says which surface it lost.
    """
    table = surfaces if surfaces is not None else DEFAULT_SURFACES
    items: list[QueueItem] = []
    failures: list[tuple[str, str, str]] = []
    for stake in portfolio.stakes:
        for name, surface in table.items():
            try:
                result = surface(stake)
            except Exception as err:  # noqa: BLE001 - one surface must not sink the queue
                log.warning("%s failed on %s: %s", name, stake.name, err)
                failures.append((stake.name, name, f"{type(err).__name__}: {err}"))
                continue
            if isinstance(result, Candidates):
                recs, considered = list(result.recommended), max(result.n_considered, 1)
            else:
                recs = list(result)
                considered = max(len(recs), 1)
            recs.sort(key=lambda r: -r.delta_title)
            for rec in recs[: max(per_surface, 1)]:
                items.append(
                    QueueItem(
                        league_id=stake.league_id,
                        league_name=stake.name,
                        team_id=stake.team_id,
                        surface=name,
                        rec=rec,
                        baseline_title=stake.title,
                        n_considered=considered,
                    )
                )
    if actionable_only:
        items = [i for i in items if i.actionable]
    items.sort(key=lambda i: -(i.delta_title * (i.leverage**leverage_weight)))
    return ActionQueue(
        items=tuple(items[:limit]),
        limit=limit,
        leverage_weight=leverage_weight,
        failures=tuple(failures),
        ranked=tuple(items),
    )


# --------------------------------------------------------------------------------------
# The whole picture
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PortfolioReport:
    """Everything this module knows, assembled once so a caller can print it."""

    odds: PortfolioOdds
    coupling: tuple[CouplingCheck, ...]
    correlations: Correlations
    exposures: tuple[Exposure, ...]
    concentration: tuple[Concentration, ...]
    pro_teams: tuple[Concentration, ...]
    byes: tuple[ByeExposure, ...]
    diversification: Diversification
    queue: ActionQueue

    def text(self) -> str:
        lines = ["=== PORTFOLIO ODDS ===", self.odds.table(), ""]
        lines += ["=== COUPLING (are the leagues really on one NFL season?) ==="]
        for c in self.coupling:
            lines.append(
                f"  {c.a} / {c.b}: {c.shared_pool} shared pool players, median player "
                f"correlation {c.median_player_correlation:.3f} "
                f"(min {c.min_player_correlation:.3f}), "
                f"availability match {c.availability_match:.4f} -> "
                + ("coupled" if c.coupled else "NOT COUPLED")
            )
        lines += ["", "=== CORRELATION ===", self.correlations.table(), ""]
        lines += ["=== EXPOSURE (players held in more than one league) ==="]
        head = (
            f"{'player':22s} {'pos':>4s} {'lg':>3s} {'st':>3s} {'equity':>9s} "
            f"{'share':>7s} {'dP(>=1)':>9s} {'+/-':>7s} {'sig':>4s}"
        )
        lines += [head, "-" * len(head)]
        for e in self.exposures:
            lines.append(
                f"{e.name[:22]:22s} {e.position:>4s} {e.n_leagues:3d} {e.n_starting:3d} "
                f"{e.equity_at_risk * 100:8.3f}pp {e.equity_share * 100:6.2f}% "
                f"{e.portfolio_damage * 100:+8.3f}pp {e.portfolio_damage_stderr * 100:7.3f} "
                f"{'yes' if e.significant else 'NO':>4s}"
            )
        lines.append(
            "  the +/- is Monte Carlo only: it is how well THIS draw pins the number down, "
            "not how well the projections do. A 'NO' row is a holding whose loss this "
            "simulation cannot distinguish from costing nothing."
        )
        lines += ["", "=== CONCENTRATION (re-simulated, not a rule) ==="]
        lines += [f"  {c.describe()}" for c in self.concentration]
        lines += [f"  {c.describe()}" for c in self.pro_teams]
        lines += ["", "=== BYE WEEKS ==="]
        lines += [f"  {b.describe()}" for b in self.byes]
        lines += ["", "=== DIVERSIFICATION ===", "  " + self.diversification.verdict, ""]
        lines += ["=== ACTION QUEUE ===", self.queue.table()]
        return "\n".join(lines)


def report(
    portfolio: Portfolio,
    *,
    limit: int = DEFAULT_QUEUE_LIMIT,
    min_leagues: int = 2,
    top_exposures: int = 5,
    queue: ActionQueue | None = None,
) -> PortfolioReport:
    """Assemble the whole cross-league picture. `queue=` skips re-running the surfaces."""
    odds = portfolio.odds()
    shared = portfolio.exposures(min_leagues=min_leagues)
    everyone = portfolio.exposures(min_leagues=1)
    return PortfolioReport(
        odds=odds,
        coupling=portfolio.verify_coupling(),
        correlations=portfolio.correlations(),
        exposures=shared,
        concentration=tuple(
            player_concentration(portfolio, e) for e in everyone[: max(top_exposures, 0)]
        ),
        pro_teams=pro_team_concentration(portfolio),
        byes=bye_concentration(portfolio),
        diversification=diversification(portfolio, odds=odds),
        queue=queue if queue is not None else action_queue(portfolio, limit=limit),
    )


__all__ = [
    "BYE_ZERO_TOLERANCE",
    "CONTESTED_TAGS",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_QUEUE_LIMIT",
    "DEFAULT_SURFACES",
    "NOT_ACTIONABLE_TAGS",
    "POSITION_ABBREV",
    "SLOT_ABBREV",
    "CORR_BOOTSTRAP",
    "ActionQueue",
    "ByeExposure",
    "Candidates",
    "Concentration",
    "Correlations",
    "CouplingCheck",
    "Diversification",
    "Exposure",
    "Holding",
    "LeagueStake",
    "PairCorrelation",
    "Portfolio",
    "PortfolioError",
    "PortfolioOdds",
    "PortfolioReport",
    "QueueItem",
    "action_queue",
    "build_portfolio",
    "bye_concentration",
    "correlations",
    "diversification",
    "exposures",
    "player_concentration",
    "portfolio_odds",
    "pro_team_concentration",
    "report",
    "verify_coupling",
]
