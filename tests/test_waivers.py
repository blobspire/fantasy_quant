"""Waiver-surface tests.

Four layers, kept apart because they fail for different reasons.

*Claim arithmetic* runs on a tiny synthetic league whose rosters are hand-built, so a
submodularity or exchange-option failure is a claim-pricing failure and nothing else.
The league is small enough that 200 simulations resolve everything the tests assert.

*The exchange option* gets its own section because it is the one result that cannot be
reproduced without simulating, and because the mechanism is not the one the research
note implies. Under HINDSIGHT lineups a bench add is worth a lot and the `E[max] >
max[E]` story is the whole explanation. Under EX-ANTE lineups -- which `sim/season.py`
insists on, correctly -- the lineup is chosen on projections that do not know the
outcome, so the bench player is worth exactly zero from realised-score variance alone.
What is left is availability: the starter gets hurt, and the bench player is the option
that covers it. Both numbers are measured here and the ordering between them is pinned.

*The continuation value* is pure dynamic programming with no tensor anywhere, so a
failure there is the optimal-stopping model.

*FAAB* is likewise closed-form where it can be: the first-order condition against
uniform rivals has to reproduce the textbook `(n-1)/n * v`, exactly.

Everything is offline and fast. No test here touches ESPN.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fantasy_quant.core import Move, MoveKind, PlayerOutlook, Recommendation, WeeklyOutlook
from fantasy_quant.decide.waivers import (
    DEFAULT_CONTEST_RATE,
    MIN_LOG_WEEKS,
    FreeAgent,
    LognormalRivalBids,
    OpportunityDistribution,
    RosterSimulator,
    UniformRivalBids,
    WaiverError,
    _confidence,
    _hold_rationale,
    _promotion_matrix,
    _rival_gain_after_drop,
    _roster_floor,
    arrival_rate_from_log,
    blocking_value,
    claim_move,
    contest_rate_from_log,
    continuation_values,
    cross_league_board,
    faab_shadow_price,
    first_order_bid,
    free_agent_pool,
    odd_dollars,
    waiver_board,
    wire_floor,
)
from fantasy_quant.sim.distributions import Draw, InjuryModel, WeeklySampler
from fantasy_quant.sim.season import (
    Franchise,
    LeagueState,
    PlayerPool,
    ScheduledGame,
    panel_for,
    team_week_scores,
)

#: ESPN's standard redraft start: 1QB/2RB/2WR/1TE/1FLEX/1DST/1K. Slot ids, not position
#: ids -- the two spaces collide at 4 and 15.
SLOT_COUNTS = {0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 23: 1}
SLOT_ELIGIBILITY = {
    0: frozenset({1}),
    2: frozenset({2}),
    4: frozenset({3}),
    6: frozenset({4}),
    16: frozenset({16}),
    17: frozenset({5}),
    23: frozenset({2, 3, 4}),
}

QB, RB, WR, TE, K, DST = 1, 2, 3, 4, 5, 16

#: Nine forced starters: QB, 2RB, 2WR, TE, a flex-able WR, D/ST, K.
ROSTER_POSITIONS = (QB, RB, RB, WR, WR, TE, WR, DST, K)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _outlook(
    player_id: int,
    position_id: int,
    pro_team_id: int,
    means: dict[int, float],
    *,
    name: str | None = None,
    shape: float = 2.0,
    p_zero: float = 0.05,
    season: int = 2026,
) -> PlayerOutlook:
    """A hurdle-gamma outlook whose stated mean the sampler actually reproduces.

    `mean = (1 - p_zero) * shape * scale`, so the scale is solved rather than guessed;
    the SD follows from the same two parameters. Getting this wrong would make every
    assertion below about a number the panel does not carry.
    """
    weeks: dict[int, WeeklyOutlook] = {}
    for week, mu in means.items():
        scale = mu / ((1.0 - p_zero) * shape) if mu > 0 else 0.0
        second = (1.0 - p_zero) * shape * (shape + 1.0) * scale * scale
        weeks[week] = WeeklyOutlook(
            player_id=player_id,
            season=season,
            week=week,
            position_id=position_id,
            mean=mu,
            sd=math.sqrt(max(second - mu * mu, 0.0)),
            p_zero=p_zero,
            shape=shape,
            scale=scale,
            pro_team_id=pro_team_id,
            playing=mu > 0,
        )
    return PlayerOutlook(
        player_id=player_id,
        name=name or f"P{player_id}",
        position_id=position_id,
        pro_team_id=pro_team_id,
        weeks=weeks,
    )


#: Weekly mean by roster slot, descending within position so a lineup has real choice.
_BASE_MEANS = (18.0, 14.0, 11.0, 13.0, 10.0, 9.0, 7.0, 8.0, 8.0)


def _league(
    n_teams: int = 4,
    weeks: tuple[int, ...] = (1, 2, 3, 4),
    playoff_rounds: tuple[tuple[int, ...], ...] = ((4,),),
    *,
    extra: tuple[PlayerOutlook, ...] = (),
    strength: float = 1.0,
) -> tuple[LeagueState, list[PlayerOutlook]]:
    """A tiny league plus the outlooks for everyone in it, free agents included.

    Player ids encode the team so a failure is readable, and every player sits on his own
    pro team so the correlation blocks stay empty -- these tests are about lineup and
    claim arithmetic, not about the copula.
    """
    reg = tuple(w for w in weeks if not any(w in r for r in playoff_rounds))
    outlooks: list[PlayerOutlook] = []
    rosters: list[tuple[int, ...]] = []
    for t in range(n_teams):
        ids = []
        for j, pos in enumerate(ROSTER_POSITIONS):
            pid = 1000 * (t + 1) + j
            ids.append(pid)
            mu = _BASE_MEANS[j] * (strength if t == 0 else 1.0)
            outlooks.append(_outlook(pid, pos, 10 * (t + 1) + j, dict.fromkeys(weeks, mu)))
        rosters.append(tuple(ids))
    outlooks.extend(extra)

    pool = PlayerPool.of(
        [(o.player_id, o.position_id, o.pro_team_id, o.name) for o in outlooks],
    )
    games = tuple(
        ScheduledGame(matchup_period=w, weeks=(w,), home_team_id=a + 1, away_team_id=b + 1)
        for w in reg
        for a, b in _pairs(n_teams, w)
    )
    state = LeagueState(
        league_id=77,
        season=2026,
        name="tiny",
        franchises=tuple(
            Franchise(team_id=i + 1, name=f"T{i + 1}", player_ids=rosters[i], is_user=(i == 0))
            for i in range(n_teams)
        ),
        pool=pool,
        weeks=weeks,
        remaining_games=games,
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=2,
        playoff_rounds=playoff_rounds,
        my_team_id=1,
    )
    return state, outlooks


def _pairs(n: int, week: int) -> list[tuple[int, int]]:
    """Circle-method pairings for one week."""
    ids = list(range(n))
    for _ in range(week - 1):
        ids = [ids[0], ids[-1], *ids[1:-1]]
    return [(ids[i], ids[n - 1 - i]) for i in range(n // 2)]


def _draw(state: LeagueState, outlooks, n_sims: int = 200, *, seed: int = 5, injuries=None) -> Draw:
    panel = panel_for(state, outlooks)
    sampler = WeeklySampler(panel, seed=seed, injuries=injuries)
    return sampler.draw(n_sims)


def _point_projection_draw(state: LeagueState, outlooks) -> Draw:
    """One 'simulation' in which every player scores exactly his projection.

    This is the world a points-based tool lives in, and it is the control arm for the
    exchange-option test: a bench player who never out-projects a starter is worth
    exactly zero here, and the arithmetic is exact rather than approximate.
    """
    panel = panel_for(state, outlooks)
    points = np.asarray(panel.mean, dtype=np.float32)[None, :, :].copy()
    available = np.asarray(panel.has_game, dtype=bool)[None, :, :].copy()
    return Draw(panel=panel, seed=0, n_sims=1, points=points, available=available)


def _engine(state, outlooks, *, n_sims=200, seed=5, replacement=None, injuries=None):
    return RosterSimulator.build(
        state,
        _draw(state, outlooks, n_sims, seed=seed, injuries=injuries),
        1,
        replacement=replacement,
    )


# --------------------------------------------------------------------------------------
# The free-agent pool and the wire floor
# --------------------------------------------------------------------------------------


class TestFreeAgentPool:
    def test_rostered_players_are_never_free_agents(self):
        state, outlooks = _league()
        agents = free_agent_pool(outlooks, state.pool.player_ids, state.weeks)
        assert agents == ()

    def test_the_board_is_ordered_above_the_wire_not_on_raw_points(self):
        """A deep position's best available player is not the best available player.

        Three quarterbacks at 20/19/18 and two backs at 12/2: the back is worth ten
        points more than what would replace him and the quarterback is worth one, so
        ordering on raw points -- which every public waiver list does -- puts the wrong
        player first.
        """
        state, outlooks = _league()
        weeks = state.weeks
        free = [
            _outlook(9001, QB, 90, dict.fromkeys(weeks, 20.0), name="QB1"),
            _outlook(9002, QB, 91, dict.fromkeys(weeks, 19.0), name="QB2"),
            _outlook(9003, QB, 92, dict.fromkeys(weeks, 18.0), name="QB3"),
            _outlook(9004, RB, 93, dict.fromkeys(weeks, 12.0), name="RB1"),
            _outlook(9005, RB, 94, dict.fromkeys(weeks, 2.0), name="RB2"),
        ]
        agents = free_agent_pool([*outlooks, *free], state.pool.player_ids, weeks)
        assert agents[0].name == "RB1"
        assert agents[0].ros_points < agents[1].ros_points  # and he is not the top scorer

    def test_a_player_projected_at_nothing_is_not_a_free_agent(self):
        state, outlooks = _league()
        dead = _outlook(9100, WR, 95, dict.fromkeys(state.weeks, 0.0))
        agents = free_agent_pool([*outlooks, dead], state.pool.player_ids, state.weeks)
        assert agents == ()

    def test_the_floor_reads_the_second_best_not_the_best(self):
        """Otherwise a claim is priced against the player being claimed."""
        state, outlooks = _league()
        weeks = state.weeks
        free = [
            _outlook(9001, WR, 90, dict.fromkeys(weeks, 15.0)),
            _outlook(9002, WR, 91, dict.fromkeys(weeks, 9.0)),
            _outlook(9003, WR, 92, dict.fromkeys(weeks, 4.0)),
        ]
        floor = wire_floor(
            [*outlooks, *free], state.pool.player_ids, weeks, SLOT_ELIGIBILITY, depth=2
        )
        assert floor[4] == pytest.approx(9.0)
        assert wire_floor(
            [*outlooks, *free], state.pool.player_ids, weeks, SLOT_ELIGIBILITY, depth=1
        )[4] == pytest.approx(15.0)

    def test_the_floor_is_read_week_by_week_and_not_off_a_season_total(self):
        """An empty slot streams whoever is best THAT WEEK, not one fixed player.

        Two defenses that alternate -- 12 one week, 2 the next, out of phase -- have
        identical season totals, so a floor read off season totals scores the slot at
        7.0 a week. What the slot actually streams is 12.0 every week, because a
        different one of them is the best available each time. On the user's real
        leagues the gap is 1.1-1.3 points a week at D/ST and quarterback, roughly 20
        points of rest-of-season score, and under the season-total reading every board
        came back dominated by "add a second defense" whose entire value was that
        understatement.

        The regression is pinned at `depth=1` because the arithmetic is exact there; the
        board's default `depth=2` inherits the same per-week reading.
        """
        weeks = (1, 2, 3, 4)
        free = [
            _outlook(9001, DST, 90, {1: 12.0, 2: 2.0, 3: 12.0, 4: 2.0}, name="odd weeks"),
            _outlook(9002, DST, 91, {1: 2.0, 2: 12.0, 3: 2.0, 4: 12.0}, name="even weeks"),
        ]
        state, outlooks = _league()
        floor = wire_floor(
            [*outlooks, *free], state.pool.player_ids, weeks, SLOT_ELIGIBILITY, depth=1
        )
        assert floor[16] == pytest.approx(12.0), (
            "the D/ST floor was read off one player's season average, not off the wire"
        )
        # And the second-best each week is the other one, not a third defense.
        deeper = wire_floor(
            [*outlooks, *free], state.pool.player_ids, weeks, SLOT_ELIGIBILITY, depth=2
        )
        assert deeper[16] == pytest.approx(2.0)

    def test_a_slot_with_nothing_available_floors_at_zero(self):
        state, outlooks = _league()
        floor = wire_floor(outlooks, state.pool.player_ids, state.weeks, SLOT_ELIGIBILITY)
        assert set(floor) == set(SLOT_ELIGIBILITY)
        assert all(v == 0.0 for v in floor.values())


# --------------------------------------------------------------------------------------
# Claim pricing: add AND drop
# --------------------------------------------------------------------------------------


class TestClaimPricing:
    def test_a_null_move_is_exactly_zero(self):
        """Common random numbers, not approximately zero. This is the whole design."""
        state, outlooks = _league()
        eng = _engine(state, outlooks)
        assert float(eng.marginal_points((), ()).max()) == 0.0
        price = eng.price((), (), confirm=True)
        assert price.delta_points == 0.0
        assert price.delta_title == 0.0

    def test_a_claim_is_add_and_drop_together(self):
        """Adding a stud and dropping a starter is worth less than the add alone."""
        state, outlooks = _league()
        weeks = state.weeks
        stud = _outlook(9001, WR, 90, dict.fromkeys(weeks, 22.0), name="stud")
        state, outlooks = _league(extra=(stud,))
        eng = _engine(state, outlooks)
        add_only = float(eng.marginal_points((9001,), ()).mean())
        with_drop = float(eng.marginal_points((9001,), (1003,)).mean())
        assert add_only > with_drop > 0.0

    def test_dropping_a_starter_for_nothing_costs_points(self):
        state, outlooks = _league()
        eng = _engine(state, outlooks)
        assert float(eng.marginal_points((), (1000,)).mean()) < 0.0

    def test_dropping_a_player_the_team_does_not_have_is_an_error(self):
        state, outlooks = _league()
        eng = _engine(state, outlooks)
        with pytest.raises(WaiverError):
            eng.marginal_points((), (4242,))

    def test_the_evaluator_protocol_is_satisfied(self):
        """`screen` and `confirm` both return one Recommendation per Move, in order."""
        state, outlooks = _league(
            extra=(_outlook(9001, WR, 90, dict.fromkeys((1, 2, 3, 4), 22.0)),)
        )
        eng = _engine(state, outlooks)
        moves = [claim_move(77, 1, 9001, 1003), Move(kind=MoveKind.HOLD, league_id=77)]
        for recs in (eng.screen(moves), eng.confirm(moves)):
            assert len(recs) == 2
            assert all(isinstance(r, Recommendation) for r in recs)
            assert recs[1].delta_title == 0.0
        assert eng.confirm(moves)[0].stderr >= 0.0

    def test_the_screen_and_the_confirm_agree_on_the_sign(self):
        state, outlooks = _league(
            extra=(_outlook(9001, WR, 90, dict.fromkeys((1, 2, 3, 4), 22.0)),)
        )
        eng = _engine(state, outlooks, n_sims=400)
        move = claim_move(77, 1, 9001, 1003)
        assert eng.screen([move])[0].delta_title > 0
        assert eng.confirm([move])[0].delta_title > 0

    def test_an_effect_inside_its_own_error_is_reported_as_not_significant(self):
        """A recommendation smaller than its Monte Carlo error must say so.

        Unconditionally. The version of this test that guarded the assertion behind
        `if abs(delta) <= 2 * stderr` could not fail: whenever the engine got the
        significance wrong the guard was false and the test asserted nothing at all.

        The claim is engineered to be genuinely unresolvable rather than structurally
        zero: the free agent out-projects the incumbent flex by a hair, so the ex-ante
        lineup really does swap and the paired difference is a small mean on top of a
        full-sized realised-score spread.
        """
        weeks = (1, 2, 3, 4)
        # The flex-able receiver on the roster is 7.0 a week; this one is 7.2, so he
        # starts, and the difference between two noisy gammas is the whole signal.
        state, outlooks = _league(
            extra=(_outlook(9001, WR, 90, dict.fromkeys(weeks, 7.2), name="coin flip"),)
        )
        eng = _engine(state, outlooks, n_sims=200)
        price = eng.price((9001,), (1006,), confirm=True)
        assert price.delta_points > 0.0, "the swap has to actually happen to be a fair test"
        assert abs(price.delta_title) <= 2.0 * price.stderr, (
            f"engineered claim came out resolvable: {price}"
        )
        assert not price.significant
        assert eng.confirm([claim_move(77, 1, 9001, 1006)])[0].confidence == "low"

    def test_a_claim_worth_exactly_nothing_is_not_called_significant(self):
        """Zero error is the one case `core.Recommendation` reads backwards.

        A free agent strictly below the wire floor adds nothing in every simulation, so
        the paired difference and its standard error are both identically zero -- and
        `abs(delta) > 2 * stderr if stderr > 0 else True` then calls that certainty.
        `ClaimPrice` overrides it, and the rendered table reads the override.
        """
        state, outlooks = _league(extra=(_outlook(9002, WR, 91, dict.fromkeys((1, 2, 3, 4), 1.0)),))
        eng = _engine(state, outlooks, n_sims=100, replacement=FLOOR)
        price = eng.price((9002,), (), confirm=True)
        assert price.delta_points == 0.0 and price.stderr == 0.0
        assert not price.significant
        assert _confidence(price, True) == "low"


# --------------------------------------------------------------------------------------
# Submodularity
# --------------------------------------------------------------------------------------


FLOOR = {0: 12.0, 2: 4.0, 4: 5.0, 6: 6.0, 16: 5.0, 17: 7.0, 23: 6.0}


class TestEmptySlotFloor:
    """The bug that made the first live board recommend dropping a starting quarterback.

    A slot no rostered player is eligible for has an EMPTY player set, and the empty set
    is a subset of every other set, so `lineup.monotone_floor` treats it as nested inside
    every slot and lifts the whole floor vector to its level. On the live Blacksburg
    roster that turned all nine slots into 12.7-point streamers the moment the only
    quarterback was dropped, and the board reported dropping Jalen Hurts as **+337 points
    and +50pp of title probability** -- as the top recommendation, in two of the three
    leagues. Nothing raised; the arithmetic was internally consistent and completely wrong.
    """

    def test_dropping_the_only_quarterback_costs_points(self):
        state, outlooks = _league()
        eng = _engine(state, outlooks, replacement=FLOOR)
        qb = next(p for p in eng.roster if state.pool.positions_of([p])[0] == QB)
        delta = float(eng.marginal_points((), (qb,)).mean())
        assert delta < 0.0, f"dropping the only QB reads as a gain of {delta:+.1f}"
        # He projects 18 a week against a 12-point wire, over four weeks, less whatever
        # the zero mass and the injury draw take off him.
        assert -30.0 < delta < -10.0, delta

    def test_the_quarterback_floor_does_not_leak_into_the_other_slots(self):
        """Raise ONLY the QB floor and only the QB slot may move. Nine times, pre-fix."""
        state, outlooks = _league()
        draw = _draw(state, outlooks, 200)
        low = RosterSimulator.build(state, draw, 1, replacement=FLOOR)
        high = RosterSimulator.build(state, draw, 1, replacement={**FLOOR, 0: 100.0})
        qb = next(p for p in low.roster if state.pool.positions_of([p])[0] == QB)
        without = [p for p in low.roster if p != qb]
        gap = float((high.season_points(without) - low.season_points(without)).mean())
        assert gap == pytest.approx(len(state.weeks) * (100.0 - 12.0), abs=1e-3)

    def test_a_roster_that_can_fill_every_slot_is_left_alone(self):
        state, _ = _league()
        positions = state.pool.positions_of(state.franchise(1).player_ids)
        floor, bonus = _roster_floor(state, FLOOR, positions)
        assert floor is FLOOR
        assert bonus == 0.0

    def test_an_unfillable_slot_is_zeroed_and_its_value_handed_back_separately(self):
        state, _ = _league()
        roster = [p for p in state.franchise(1).player_ids if state.pool.positions_of([p])[0] != QB]
        floor, bonus = _roster_floor(state, FLOOR, state.pool.positions_of(roster))
        assert floor[0] == 0.0
        assert bonus == pytest.approx(12.0)
        assert all(floor[s] == FLOOR[s] for s in FLOOR if s != 0)

    def test_a_scalar_or_absent_floor_needs_no_correction(self):
        state, _ = _league()
        positions = state.pool.positions_of(state.franchise(1).player_ids[:1])
        assert _roster_floor(state, None, positions) == (None, 0.0)
        assert _roster_floor(state, 3.0, positions) == (3.0, 0.0)


class TestExchangeRate:
    """The points-to-title rate, which is what the published `delta_title` runs through."""

    def test_the_rate_is_positive_and_team_specific(self):
        state, outlooks = _league(n_teams=6, strength=1.6)
        eng = _engine(state, outlooks, n_sims=600)
        mine, _ = eng.exchange_rate(1)
        assert mine > 0.0
        assert all(eng.exchange_rate(t)[0] >= 0.0 for t in state.team_ids)

    def test_the_bracket_error_does_not_shrink_with_the_claim_and_the_published_one_does(self):
        """The whole reason `delta_title` is the plug-in and not the bracket difference.

        A paired bracket difference is a difference of two 0/1 champion indicators, so its
        standard error is set by the title probability and barely moves when the claim
        gets smaller. The plug-in's error is the points error through the rate, and it
        shrinks with the claim. Below some size the bracket simply cannot see the effect
        it is being asked to rank -- which on the live leagues is *every* waiver claim:
        they are worth 0.1-0.7pp against a bracket error near 0.2-0.3pp at 4,000
        simulations, so a board ranked on the bracket is a board ranked on noise.
        """
        big = _outlook(9001, WR, 90, dict.fromkeys((1, 2, 3, 4), 15.0), name="big")
        small = _outlook(9002, WR, 91, dict.fromkeys((1, 2, 3, 4), 9.0), name="small")
        state, outlooks = _league(extra=(big, small))
        eng = _engine(state, outlooks, n_sims=800, replacement=FLOOR)
        a = eng.price((9001,), (1006,), confirm=True)
        b = eng.price((9002,), (1006,), confirm=True)

        assert b.delta_points < 0.5 * a.delta_points
        assert b.stderr < 0.6 * a.stderr, "the published error does not track the claim"
        assert b.bracket_stderr > 0.7 * a.bracket_stderr, "the bracket error tracks the claim"
        assert b.bracket_stderr > 2.0 * b.stderr
        assert a.agrees and b.agrees


class TestSubmodularity:
    def test_each_extra_receiver_is_worth_less_than_the_last(self):
        """Four identical good receivers, added one at a time.

        The roster starts two dedicated receivers plus a flex, so the first add displaces
        a weak starter, the second fills the flex, and the third and fourth are pure
        bench. The gains must be non-increasing -- that is what submodular means -- and
        the drop from the first to the fourth is what a ranking list cannot express,
        because a list gives all four the same number.
        """
        weeks = (1, 2, 3, 4)
        clones = tuple(
            _outlook(9000 + i, WR, 90 + i, dict.fromkeys(weeks, 16.0), name=f"WR{i}")
            for i in range(4)
        )
        state, outlooks = _league(extra=clones)
        eng = _engine(state, outlooks, n_sims=300)

        roster = list(eng.roster)
        gains = []
        previous = eng.season_points(roster)
        for i in range(4):
            roster.append(9000 + i)
            now = eng.season_points(roster)
            gains.append(float((now - previous).mean()))
            previous = now

        assert gains[0] > 0.0
        for a, b in zip(gains, gains[1:], strict=False):
            assert b <= a + 1e-9, f"gains not submodular: {gains}"
        assert gains[3] < 0.35 * gains[0], f"a fourth receiver is not cheap enough: {gains}"


# --------------------------------------------------------------------------------------
# The exchange option
# --------------------------------------------------------------------------------------


class TestExchangeOption:
    """A bench add under three lineup regimes, which is where the research note is loose.

    RESEARCH.md says `E[max] > max[E]` means bench depth is worth literally zero under
    point projections, and that is right. What it does not say is that the same argument
    kills most of the bench value under *ex-ante* lineups too, because the lineup is set
    on projections that do not know the outcome. What survives is availability. The
    three numbers are measured below and their ordering pinned.
    """

    def _setup(self):
        weeks = (1, 2, 3, 4)
        # Strictly worse than every receiver already starting, so he never out-projects
        # one and can only be worth something through somebody else's absence.
        bench = _outlook(9001, WR, 90, dict.fromkeys(weeks, 6.0), name="bench WR")
        return _league(extra=(bench,))

    def test_a_bench_add_is_worth_exactly_zero_under_point_projections(self):
        state, outlooks = self._setup()
        eng = RosterSimulator.build(state, _point_projection_draw(state, outlooks), 1)
        assert float(eng.marginal_points((9001,), ()).mean()) == 0.0

    def test_the_same_add_is_strictly_positive_under_simulation(self):
        state, outlooks = self._setup()
        eng = _engine(state, outlooks, n_sims=600)
        gain = float(eng.marginal_points((9001,), ()).mean())
        assert gain > 0.0, "a bench receiver has no exchange-option value at all"

    def test_turning_injuries_off_removes_the_ex_ante_option(self):
        """Under ex-ante lineups the option IS availability, not realised-score variance.

        With injuries off, every player's availability is fixed by the schedule, the
        ex-ante lineup never changes, and a permanently worse receiver adds exactly zero
        even though the realised scores still bounce around. That is the refinement to
        the research note: it is the absences that pay, not `E[max]` over the scores.
        """
        state, outlooks = self._setup()
        eng = _engine(state, outlooks, n_sims=400, injuries=InjuryModel.off())
        assert float(eng.marginal_points((9001,), ()).mean()) == 0.0

    def test_hindsight_lineups_pay_far_more_for_the_same_bench_player(self):
        """The upper bound, and the reason `sim/season.py` refuses to set lineups that way.

        Ranking on the realised tensor is the `E[max]` world in its pure form. The same
        receiver is worth several times more there, which is exactly the inflation a
        hindsight simulator would hand a deep bench.
        """
        state, outlooks = self._setup()
        draw = _draw(state, outlooks, 400)
        eng = RosterSimulator.build(state, draw, 1)
        ex_ante = float(eng.marginal_points((9001,), ()).mean())

        wide = state.with_franchise(
            state.franchise(1).with_players((*state.franchise(1).player_ids, 9001))
        )
        # rank=None on a bare tensor is season.py's documented opt-in to hindsight.
        hind_base = team_week_scores(state, draw.points, rank=None)[:, :, 0].sum(axis=1)
        hind_wide = team_week_scores(wide, draw.points, rank=None)[:, :, 0].sum(axis=1)
        hindsight = float((hind_wide - hind_base).mean())

        assert hindsight > ex_ante > 0.0
        assert hindsight > 3.0 * ex_ante


# --------------------------------------------------------------------------------------
# Blocking
# --------------------------------------------------------------------------------------


class TestBlocking:
    def test_blocking_value_is_one_over_n_minus_one(self):
        assert blocking_value(0.09, 10) == pytest.approx(0.01)
        assert blocking_value(0.12, 13) == pytest.approx(0.01)
        with pytest.raises(WaiverError):
            blocking_value(0.1, 1)

    def test_a_rivals_gain_is_the_rest_of_the_leagues_loss_exactly(self):
        """Championship probability sums to one, so the identity is arithmetic.

        Every simulation has exactly one champion in both worlds, so the per-team title
        changes sum to zero to machine precision. That is what makes `1/(N-1)` the right
        average share and not merely a plausible one.
        """
        weeks = (1, 2, 3, 4)
        stud = _outlook(9001, WR, 90, dict.fromkeys(weeks, 24.0), name="stud")
        state, outlooks = _league(n_teams=6, extra=(stud,))
        eng = _engine(state, outlooks, n_sims=800)

        rival = 2
        roster = state.franchise(rival).player_ids
        deltas = {}
        base_all = None
        for team in state.team_ids:
            before = eng.champions(state.franchise(team).player_ids, team)
            if base_all is None:
                base_all = before
        # One re-simulation gives every team's new title probability at once.
        scores = eng._base_scores.copy()
        index = state.team_index[rival]
        scores[:, :, index] = (
            eng.week_scores((*roster, 9001), rival) * eng._factors[:, index : index + 1]
        )
        from fantasy_quant.sim.season import simulate_from_scores

        after = simulate_from_scores(state, scores, all_play=False).champions.astype(float)
        before = simulate_from_scores(state, eng._base_scores, all_play=False).champions.astype(
            float
        )
        deltas = (after - before).mean(axis=0)

        assert abs(float(deltas.sum())) < 1e-12
        gain = float(deltas[index])
        assert gain > 0.0
        others = np.delete(deltas, index)
        assert float(-others.mean()) == pytest.approx(blocking_value(gain, state.size), rel=1e-9)

    def test_a_rivals_claim_is_an_add_and_a_drop_too(self):
        """The blocking number was screening rivals on the add alone.

        A rival's roster is as full as ours, so he pays for the stud with whatever he
        cuts, and pricing his gain as a free add is the exact error this module's opening
        paragraph rejects for our own claims. On the live boards it inflated every block.
        """
        weeks = (1, 2, 3, 4)
        stud = _outlook(9001, WR, 90, dict.fromkeys(weeks, 24.0), name="stud")
        state, outlooks = _league(n_teams=6, extra=(stud,))
        eng = _engine(state, outlooks, n_sims=300, replacement=FLOOR)

        rival = 2
        roster = state.franchise(rival).player_ids
        add_only = float(
            (eng.season_points((*roster, 9001), rival) - eng.season_points(roster, rival)).mean()
        )
        paid = _rival_gain_after_drop(eng, rival, 9001, add_only)
        assert 0.0 < paid < add_only, (paid, add_only)
        # And the best cut really is the best one available, not merely *a* cut.
        base = eng.season_points(roster, rival)
        exhaustive = max(
            float(
                (eng.season_points((*(p for p in roster if p != drop), 9001), rival) - base).mean()
            )
            for drop in roster
        )
        assert paid == pytest.approx(exhaustive, rel=1e-9)

    def test_a_player_no_rival_would_pay_a_roster_spot_for_is_not_a_block(self):
        """Clamped at zero: a rival who would lose by claiming simply does not claim."""
        weeks = (1, 2, 3, 4)
        scrub = _outlook(9002, WR, 91, dict.fromkeys(weeks, 0.5), name="scrub")
        state, outlooks = _league(n_teams=4, extra=(scrub,))
        eng = _engine(state, outlooks, n_sims=200, replacement=FLOOR)
        assert _rival_gain_after_drop(eng, 2, 9002, 0.0) == 0.0

    def test_owning_a_player_is_an_order_of_magnitude_better_than_blocking_him(self):
        """At the user's real league sizes, blocking is worth ~8% of owning.

        The gap is the league size, so it is measured at twelve teams rather than at the
        four the rest of the file uses: `1/(N-1)` is only "an order of magnitude" once
        `N` is a real league.
        """
        weeks = (1, 2, 3, 4)
        stud = _outlook(9001, WR, 90, dict.fromkeys(weeks, 24.0), name="stud")
        state, outlooks = _league(n_teams=12, extra=(stud,))
        eng = _engine(state, outlooks, n_sims=800)
        own = eng.price((9001,), (1006,), confirm=True).delta_title

        rival = 2
        roster = state.franchise(rival).player_ids
        rival_gain = float(
            (eng.champions((*roster, 9001), rival) - eng.champions(roster, rival)).mean()
        )
        block = blocking_value(rival_gain, state.size)
        assert own > 5.0 * block, f"own {own:.4f} against block {block:.4f}"


# --------------------------------------------------------------------------------------
# The continuation value of priority
# --------------------------------------------------------------------------------------


def _wire(mean: float = 0.02) -> OpportunityDistribution:
    return OpportunityDistribution.exponential(mean)


class TestContinuationValue:
    def test_the_last_waiver_run_has_no_continuation_value(self):
        """Priority has no salvage: `C_T(p) = 0` for every p."""
        table = continuation_values(range(1, 15), 12, _wire())
        assert np.allclose(table.values[-1], 0.0)

    def test_the_worst_priority_costs_nothing_to_spend(self):
        """`C_t(N) = 0`: winning sends you to the back, and you are already there."""
        table = continuation_values(range(1, 15), 12, _wire())
        assert np.allclose(table.values[:, -1], 0.0)

    def test_the_continuation_value_falls_through_the_season(self):
        table = continuation_values(range(1, 15), 12, _wire())
        for p in range(12):
            column = table.values[:, p]
            assert np.all(np.diff(column) <= 1e-15), f"priority {p + 1} is not decreasing"
        assert table.values[0, 0] > table.values[-2, 0]

    def test_a_better_priority_is_never_cheaper_to_spend(self):
        table = continuation_values(range(1, 15), 12, _wire())
        for t in range(table.values.shape[0]):
            row = table.values[t]
            assert np.all(np.diff(row) <= 1e-15), f"week {t} thresholds rise with priority"
        assert np.all(table.values >= -1e-15)

    def test_a_heavier_right_tail_raises_the_cost_of_spending(self):
        """`C` responds to the tail, which is the whole reason it is not just `E[v]`."""
        thin = continuation_values(range(1, 15), 12, _wire(0.01))
        fat = continuation_values(range(1, 15), 12, _wire(0.04))
        assert fat.values[0, 0] > thin.values[0, 0]

    def test_a_dead_wire_makes_priority_worthless(self):
        empty = OpportunityDistribution(values=(0.0,), weights=(1.0,))
        table = continuation_values(range(1, 15), 12, empty)
        assert np.allclose(table.values, 0.0)

    def test_the_contest_rate_cannot_flip_a_claim(self):
        """It scales the value function and cancels out of the threshold's sign.

        Worth pinning: it is the parameter we are least able to measure, and a reader
        should be able to see that being wrong about it is not dangerous.
        """
        a = continuation_values(range(1, 15), 12, _wire(), contest_rate=0.02)
        b = continuation_values(range(1, 15), 12, _wire(), contest_rate=0.5)
        assert np.all(np.sign(a.values) == np.sign(b.values))

    def test_queue_movement_compresses_the_threshold_but_keeps_the_boundaries(self):
        still = continuation_values(range(1, 15), 12, _wire(), promotion=0.0)
        moving = continuation_values(range(1, 15), 12, _wire(), promotion=0.3)
        assert moving.values[0, 0] < still.values[0, 0]
        assert np.allclose(moving.values[:, -1], 0.0)
        assert np.allclose(moving.values[-1], 0.0)

    def test_the_promotion_matrix_is_a_distribution(self):
        for promotion in (0.0, 0.25, 1.0):
            m = _promotion_matrix(8, promotion)
            assert np.allclose(m.sum(axis=1), 1.0)
            assert np.all(m >= 0.0)

    def test_a_week_outside_the_table_has_nothing_left_to_protect(self):
        table = continuation_values((5, 6, 7), 12, _wire())
        assert table.threshold(99, 1) == 0.0
        with pytest.raises(WaiverError):
            table.threshold(5, 13)

    def test_bad_inputs_are_refused(self):
        with pytest.raises(WaiverError):
            continuation_values((), 12, _wire())
        with pytest.raises(WaiverError):
            continuation_values((1, 2), 12, _wire(), contest_rate=1.0)
        with pytest.raises(WaiverError):
            continuation_values((1, 2), 12, [_wire()])

    def test_the_table_renders(self):
        table = continuation_values((1, 2, 3), 4, _wire())
        assert "week" in table.table()
        assert table.size == 4


class TestOpportunityDistribution:
    def test_expected_excess_is_the_option_value_and_decreases(self):
        dist = OpportunityDistribution(values=(0.01, 0.05), weights=(0.5, 0.5))
        assert dist.expected_excess(0.0) == pytest.approx(0.03)
        assert dist.expected_excess(0.02) == pytest.approx(0.015)
        assert dist.expected_excess(1.0) == 0.0

    def test_arrival_scales_the_whole_wire_down(self):
        full = OpportunityDistribution.from_board([0.04, 0.02], arrival=1.0)
        quiet = OpportunityDistribution.from_board([0.04, 0.02], arrival=0.25)
        assert quiet.mean == pytest.approx(full.mean * 0.25)

    def test_an_empty_board_is_a_wire_worth_nothing(self):
        dist = OpportunityDistribution.from_board([-0.1, 0.0])
        assert dist.mean == 0.0

    def test_weights_must_be_a_sub_probability(self):
        with pytest.raises(WaiverError):
            OpportunityDistribution(values=(0.1,), weights=(1.5,))
        with pytest.raises(WaiverError):
            OpportunityDistribution(values=(0.1, 0.2), weights=(1.0,))


class _Log:
    """The two things `TransactionLog` is read for: iteration and `available`."""

    def __init__(self, rows, available=True):
        self.available = available
        self.transactions = tuple(
            SimpleNamespace(is_waiver=w, is_executed=e, scoring_period_id=p, bid_amount=b)
            for w, e, p, b in rows
        )

    def __iter__(self):
        return iter(self.transactions)

    def __len__(self):
        # `_history` guards on `len(log)`; without this the fake silently reads as "no
        # log at all" and every test that hands one over exercises the fallback instead.
        return len(self.transactions)


class TestLogDrivenRates:
    def test_an_unavailable_log_falls_back_rather_than_reporting_a_dead_wire(self):
        """ESPN refusing the log is not the same as the league making no claims."""
        assert contest_rate_from_log(_Log((), available=False), 12, 14) == DEFAULT_CONTEST_RATE
        assert arrival_rate_from_log(_Log((), available=False), 14) == 1.0

    def test_rates_come_from_executed_waivers_only(self):
        rows = [
            (True, True, 3, 0),
            (True, True, 3, 0),
            (True, True, 4, 0),
            (True, False, 5, 0),  # pending
            (False, True, 6, 0),  # a free-agent add, not a waiver
        ]
        assert contest_rate_from_log(_Log(rows), 12, 10) == pytest.approx(3 / 120)
        assert arrival_rate_from_log(_Log(rows), 10) == pytest.approx(0.2)

    def test_a_quiet_league_still_gets_a_usable_rate(self):
        """A single claim all season is clamped rather than driving the DP to zero."""
        assert contest_rate_from_log(_Log([(True, True, 3, 0)]), 14, 14) == pytest.approx(0.01)

    def test_the_board_ignores_a_log_too_short_to_mean_anything(self):
        """One week of a fresh season is not a sample, and it reads as a dead wire.

        Every league has made almost no waiver claims by week 1, so a rate measured
        there clamps to the floor and reports waiver priority as nearly worthless --
        measured, it moved Wine Wednesday's week-1 threshold from 0.250pp to 0.171pp on
        no evidence. `MIN_LOG_WEEKS` is the guard.

        The league here *serves* a log rather than refusing one, and records whether it
        was asked. The version of this test that ran against `_NoLeague` compared two
        different spellings of the same fallback and could not fail whatever
        `MIN_LOG_WEEKS` was set to.
        """
        assert MIN_LOG_WEEKS >= 2
        season = tuple(range(1, 9))
        # A league that claims hard: three executed waivers a week, every week. Reading
        # that gives a contest rate far above the 0.15 prior and an arrival rate of 1.
        rows = [(True, True, w, 0) for w in season for _ in range(3)]
        week = MIN_LOG_WEEKS + 1
        assert week < season[-1], "the probe week must not be the terminal one"

        early_league = _LoggedLeague(rows)
        early = _board(priority=1, week=1, weeks=season, league=early_league)
        assert early_league.asked == 0, "week 1 must not consult the log at all"

        late_league = _LoggedLeague(rows)
        late = _board(priority=1, week=week, weeks=season, league=late_league)
        assert late_league.asked == 1, "past MIN_LOG_WEEKS the log has to be read"

        blind_league = _LoggedLeague(rows)
        blind = _board(priority=1, week=week, weeks=season, league=blind_league, use_log=False)
        assert blind_league.asked == 0

        # Same week, same board, log against no log: if reading it did not move the
        # threshold the guard would be untestable rather than merely untested.
        assert late.threshold != pytest.approx(blind.threshold)
        assert early.threshold > 0.0


# --------------------------------------------------------------------------------------
# FAAB
# --------------------------------------------------------------------------------------


class TestFaab:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
    def test_the_first_order_condition_recovers_the_textbook_uniform_bid(self, n):
        """`b + F(b)/f(b) = v/lambda` must give `b = (n-1)/n * v` against uniform rivals.

        With `n-1` rivals bidding uniform on `[0, v]`, the best rival bid has
        `F(b) = (b/v)^(n-1)` and `F/f = b/(n-1)`, so the condition collapses to
        `b * n/(n-1) = v`. This is the one place the bidding maths has a closed form and
        it is worth pinning exactly.
        """
        value = 40.0
        rivals = UniformRivalBids(rivals=n - 1, high=value)
        bid = first_order_bid(value, rivals, shadow_price=1.0, high=value)
        assert bid == pytest.approx(value * (n - 1) / n, rel=1e-6)

    def test_a_cheap_dollar_buys_a_bigger_bid(self):
        rivals = UniformRivalBids(rivals=3, high=100.0)
        expensive = first_order_bid(40.0, rivals, shadow_price=1.0, high=100.0)
        cheap = first_order_bid(40.0, rivals, shadow_price=0.4, high=100.0)
        assert cheap > expensive

    def test_a_worthless_dollar_bids_the_cap(self):
        """Unspent budget has zero salvage, so the last week is a spend-it-all week."""
        rivals = UniformRivalBids(rivals=3, high=100.0)
        assert first_order_bid(40.0, rivals, shadow_price=0.0, high=100.0) == 100.0

    def test_nothing_is_bid_on_a_worthless_player(self):
        assert first_order_bid(0.0, UniformRivalBids(rivals=3, high=50.0)) == 0.0

    def test_bids_are_odd_dollars(self):
        assert odd_dollars(12.2) == 13
        assert odd_dollars(0.1) == 1
        assert odd_dollars(99.0, budget=40) == 39
        assert all(odd_dollars(b) % 2 == 1 for b in range(1, 60))

    def test_the_shadow_price_falls_as_the_season_runs_out(self):
        """`V_{T+1} = 0`, so a dollar is worth less the fewer weeks are left to use it.

        Measured on the whole budget rather than on its last dollar: at the very top of
        the budget the DP is flat, because a bid that high already wins almost surely and
        the marginal dollar buys nothing in any week. The binding dollars are the ones in
        the middle, and those fall cleanly.
        """
        lam = faab_shadow_price(6, 40, _wire(0.05), UniformRivalBids(rivals=3, high=40.0))
        whole = lam.sum(axis=1)  # V_t(budget), telescoped
        assert np.all(np.diff(whole) < 0.0), f"budget value is not decaying: {whole}"
        for b in (5, 10, 20, 30):
            column = lam[:, b]
            assert np.all(np.diff(column) <= 1e-12), f"dollar {b} is not decaying: {column}"

    def test_the_last_dollar_of_a_large_budget_is_nearly_worthless_at_the_end(self):
        lam = faab_shadow_price(4, 60, _wire(0.05), UniformRivalBids(rivals=3, high=20.0))
        assert lam[-1, -1] < 1e-4

    def test_the_population_prior_is_a_fantasy_wire_not_a_uniform_draw(self):
        """Median at 1.4% of budget with a fat tail, which is what FAAB leagues do.

        The uniform-on-the-whole-budget fallback this replaced is not a prior, it is an
        assumption that rivals bid at random, and on the live board it wanted $55 of a
        $100 budget for a marginal D/ST upgrade.
        """
        prior = LognormalRivalBids.population(100)
        assert prior.cdf(1.4) == pytest.approx(0.5, abs=0.02)
        assert 0.9 < prior.cdf(30.0) < 1.0
        bid = first_order_bid(0.006, prior, shadow_price=8e-05, high=100.0)
        assert 3.0 < bid < 20.0, bid
        assert (
            first_order_bid(
                0.006, UniformRivalBids(rivals=3, high=100.0), shadow_price=8e-05, high=100.0
            )
            > 2.0 * bid
        )

    def test_a_winning_bid_history_is_already_a_maximum(self):
        """`rivals` defaults to 1: the log records the winner, not one rival's bid.

        Fitting the winning bid and then taking the max over three "rivals" applies the
        auction twice, and the difference is a factor of six on a real board.
        """
        history = [4, 9, 15, 22, 40]
        once = LognormalRivalBids.from_history(history)
        assert once.rivals == 1
        twice = LognormalRivalBids.from_history(history, rivals=3)
        b_once = first_order_bid(0.006, once, shadow_price=8e-05, high=100.0)
        b_twice = first_order_bid(0.006, twice, shadow_price=8e-05, high=100.0)
        assert b_twice > b_once

    def test_bid_history_is_fitted_without_the_uncontested_dollar_zeros(self):
        fitted = LognormalRivalBids.from_history([0, 0, 0, 8, 12, 20, 31], rivals=3)
        assert fitted.cdf(1.0) < fitted.cdf(40.0)
        assert fitted.pdf(15.0) > 0.0
        with pytest.raises(WaiverError):
            LognormalRivalBids.from_history([0, 0, 5], rivals=3)

    def test_the_fitted_bid_rises_with_the_players_value(self):
        rivals = LognormalRivalBids.from_history([4, 9, 15, 22, 40], rivals=3)
        low = first_order_bid(5.0, rivals, shadow_price=0.2, high=100.0)
        high = first_order_bid(50.0, rivals, shadow_price=0.2, high=100.0)
        assert high > low

    def test_a_budget_dp_needs_a_budget(self):
        with pytest.raises(WaiverError):
            faab_shadow_price(0, 10, _wire(), UniformRivalBids(rivals=2, high=10.0))


# --------------------------------------------------------------------------------------
# The board, end to end
# --------------------------------------------------------------------------------------


class _NoLeague:
    """Stands in for `League` when the test has no network. Every accessor refuses.

    `waiver_board` has to survive this: it is also what a rate-limited or logged-out run
    looks like, and losing the whole board because the settings call failed would be a
    worse failure than falling back to the arguments.
    """

    def settings(self):
        raise RuntimeError("offline")

    def teams(self):
        raise RuntimeError("offline")

    def transactions(self):
        raise RuntimeError("offline")


class _LoggedLeague(_NoLeague):
    """A league that refuses settings and the waiver order but does serve a log.

    `asked` is the point: `MIN_LOG_WEEKS` is a guard on whether the log is *consulted*,
    so a test of it has to be able to see the call that did not happen.
    """

    def __init__(self, rows):
        self._rows = rows
        self.asked = 0

    def transactions(self):
        self.asked += 1
        return _Log(self._rows)


def _fake_sim(state, outlooks, *, n_sims=200, seed=5, league=None):
    return SimpleNamespace(
        league=league or _NoLeague(),
        state=state,
        draw=None,
        outlooks=outlooks,
        n_sims=n_sims,
        seed=seed,
    )


def _board(*, league=None, weeks=(1, 2, 3, 4), **kw):
    extra = tuple(
        _outlook(9000 + i, WR, 90 + i, dict.fromkeys(weeks, mu), name=f"FA{i}")
        for i, mu in enumerate((21.0, 16.0, 12.0, 8.0, 5.0, 3.0))
    )
    state, outlooks = _league(weeks=weeks, playoff_rounds=((weeks[-1],),), extra=extra)
    sim = _fake_sim(state, outlooks, league=league)
    kw.setdefault("candidates", 6)
    kw.setdefault("confirm", 4)
    kw.setdefault("drops", 3)
    kw.setdefault("screen_sims", 120)
    kw.setdefault("blocks", 0)
    return waiver_board(sim, **kw)


class TestBoard:
    def test_the_board_is_ordered_by_title_probability(self):
        report = _board(priority=4)
        assert report.board
        deltas = [r.delta_title for r in report.board]
        assert deltas == sorted(deltas, reverse=True)
        assert all("waiver" in r.tags for r in report.board)

    def test_every_claim_clears_the_threshold_and_nothing_else_does(self):
        """The stopping rule, applied. This is the one decision the module makes."""
        report = _board(priority=1)
        assert all(r.delta_title >= report.threshold for r in report.claims)
        rejected = [r for r in report.board if r not in report.claims]
        assert all(r.delta_title < report.threshold or r.delta_title <= 0.0 for r in rejected)

    def test_the_worst_priority_has_a_zero_threshold_so_every_gain_is_claimable(self):
        report = _board(priority=4)
        assert report.threshold == 0.0
        assert all(r.delta_title > 0.0 for r in report.claims)

    def test_a_better_priority_is_harder_to_spend(self):
        best = _board(priority=1)
        worst = _board(priority=4)
        assert best.threshold >= worst.threshold

    def test_the_output_says_a_losing_claim_is_free(self):
        report = _board(priority=4)
        text = report.hold.rationale + " ".join(r.rationale for r in report.claims)
        assert "waterfall" in text or "Hold" in report.hold.rationale
        assert "Tuesday" in report.hold.rationale

    def test_every_recommendation_carries_an_actionable_rationale(self):
        report = _board(priority=2)
        for rec in report.board:
            assert rec.rationale
            assert "title probability" in rec.rationale
            assert "3-4am ET Wednesday" in rec.rationale

    def test_a_claim_is_an_add_and_a_drop(self):
        report = _board(priority=4)
        for rec in report.board:
            adds = [p for p in rec.move.players if p.to_team == report.team_id]
            drops = [p for p in rec.move.players if p.from_team == report.team_id]
            assert len(adds) == 1
            # No roster limit is readable offline, so a drop is required, which is the
            # conservative branch and the one a full roster is actually in.
            assert len(drops) == 1
            assert rec.move.kind is MoveKind.WAIVER_CLAIM

    def test_a_priority_league_prices_in_priority_not_dollars(self):
        report = _board(priority=2)
        assert not report.uses_faab
        assert report.priority == 2
        assert report.continuation is not None
        assert all(r.move.bid is None for r in report.board)

    def test_a_faab_league_prices_in_odd_dollars(self):
        report = _board(uses_faab=True, budget=100)
        assert report.uses_faab
        assert report.continuation is None
        bids = [r.move.bid for r in report.board]
        assert all(b is not None and b % 2 == 1 for b in bids)
        assert all(0 < b <= 100 for b in bids)

    def test_an_injected_evaluator_is_used_for_the_confirm_tier(self):
        """The whole point of taking a `core.MoveEvaluator`: `decide/title.py` can drive it."""
        seen: list[int] = []

        class Fake:
            def screen(self, moves):
                return self.confirm(moves)

            def confirm(self, moves):
                seen.append(len(moves))
                return [
                    Recommendation(
                        move=m, delta_title=0.01 * (i + 1), delta_points=1.0, stderr=0.001
                    )
                    for i, m in enumerate(moves)
                ]

            def baseline_title(self, team_id):
                return 0.25

        report = _board(priority=4, evaluator=Fake())
        assert report.baseline_title == 0.25
        # More moves are CONFIRMED than are shown: each add is paired against several
        # drops and only its best pairing reaches the board, because the points screen
        # and the title objective disagree about which drop is right.
        assert seen and seen[0] >= len(report.board)
        adds = [
            next(p.player_id for p in r.move.players if p.to_team is not None) for r in report.board
        ]
        assert len(adds) == len(set(adds)), "one row per player, not one per pairing"

    def test_blocking_candidates_come_back_tagged_and_tiny(self):
        report = _board(priority=4, blocks=2)
        # At most `blocks` candidates, and only the ones a rival would actually claim:
        # a player worth nothing to anybody once his roster spot is paid for is not a
        # blocking candidate, he is a non-event.
        assert 0 < len(report.blocks) <= 2
        for rec in report.blocks:
            assert "blocking" in rec.tags
            assert "Never claim for this reason alone" in rec.rationale
            # A player who helps a rival cannot be worth negative to block. Pricing this
            # off a direct bracket difference produced exactly that on the live board,
            # which is why it goes through the rival's own exchange rate instead.
            assert rec.delta_title >= 0.0
        best_own = max(r.delta_title for r in report.board)
        assert max(r.delta_title for r in report.blocks) < best_own

    def test_an_unreadable_waiver_order_assumes_the_FRONT_of_the_queue_and_says_so(self):
        """The bug that turned the whole optimal-stopping model off in production.

        `pipeline.build` closes the client it opened and `League.teams()` is not cached,
        so on every sim built the documented way the waiver-order read failed with
        "Cannot send a request, as the client has been closed" -- silently. The board then
        fell back to `state.size`, the WORST priority, whose continuation value is zero by
        construction, so every positive claim cleared a threshold of nothing and the
        report printed "priority 14/14" as if ESPN had said so. Measured against the live
        league the user actually holds priority **1**: twelve claims were recommended
        where three clear the real threshold.

        Unknown priority is now the front of the queue -- the most expensive place to
        spend from -- and the report says it is an assumption in the table and in every
        rationale. `_NoLeague` raises from `teams()` and carries no league_id/season, so
        this exercises the fallback without touching the network.
        """
        report = _board()  # no priority= : the surface has to work it out
        assert not report.priority_known
        assert report.priority == 1, "an unknown queue position must not be assumed worst"
        assert report.threshold > 0.0
        assert "ASSUMED" in report.table()
        assert "assumed" in report.hold.rationale
        for rec in report.board:
            assert "assumed" in rec.rationale

    def test_a_priority_that_was_actually_read_is_not_flagged_as_assumed(self):
        report = _board(priority=2)
        assert report.priority_known
        assert "ASSUMED" not in report.table()
        assert "assumed" not in report.hold.rationale

    def test_the_board_stamps_its_own_waiver_tag_whatever_engine_confirmed_it(self):
        """`decide/title.py`'s engine tags with its own vocabulary, and `waiver_board`
        acquired a live dependency on the sibling module's tag strings the day it grew an
        `evaluator_for`. The board's own contract is stamped here, not inherited."""

        class Untagged:
            def screen(self, moves):
                return self.confirm(moves)

            def confirm(self, moves):
                return [
                    Recommendation(move=m, delta_title=0.01, delta_points=1.0, tags=("mine",))
                    for m in moves
                ]

            def baseline_title(self, team_id):
                return 0.1

        report = _board(priority=4, evaluator=Untagged())
        assert report.board
        assert all("waiver" in r.tags and "mine" in r.tags for r in report.board)

    def test_a_waterfall_of_claims_that_drop_different_players_is_flagged(self):
        """ "Submit them all" is only free when at most one of them can execute.

        ESPN re-processes the rest of your list at your new place at the back of the
        queue after a successful claim, so two claims that drop different players can both
        land in the same run -- and each was priced against today's roster on its own,
        which submodularity says overstates the pair.
        """
        move = claim_move(77, 1, 9001, 1003)
        same = [
            Recommendation(move=move, delta_title=0.02, delta_points=1.0, tags=("drop:A",)),
            Recommendation(move=move, delta_title=0.01, delta_points=1.0, tags=("drop:A",)),
        ]
        mixed = [
            same[0],
            Recommendation(move=move, delta_title=0.01, delta_points=1.0, tags=("drop:B",)),
        ]
        one = _hold_rationale(False, 3, 0.005, same, same)
        two = _hold_rationale(False, 3, 0.005, mixed, mixed)
        assert "at most one of them can execute" in one
        assert "Careful" not in one
        assert "2 different players" in two
        assert "worth less than the sum" in two

    def test_the_report_renders(self):
        report = _board(priority=3)
        text = report.table()
        assert "tiny" in text and "threshold" in text

    def test_a_league_with_no_free_agents_is_an_error_not_an_empty_board(self):
        state, outlooks = _league()
        with pytest.raises(WaiverError):
            waiver_board(_fake_sim(state, outlooks), screen_sims=60)

    def test_a_sim_with_no_team_is_refused(self):
        weeks = (1, 2, 3, 4)
        state, outlooks = _league(extra=(_outlook(9001, WR, 90, dict.fromkeys(weeks, 20.0)),))
        headless = state.__class__(
            **{
                **{f.name: getattr(state, f.name) for f in state.__dataclass_fields__.values()},
                "my_team_id": None,
            }
        )
        with pytest.raises(WaiverError):
            waiver_board(_fake_sim(headless, outlooks), screen_sims=60)

    def test_cross_league_ranking_puts_the_best_claim_first(self):
        a = _board(priority=4)
        b = _board(priority=4)
        merged = cross_league_board([a, b], limit=3)
        assert 0 < len(merged) <= 3
        assert len(merged) == min(3, len(a.claims) + len(b.claims))
        assert [r.delta_title for r in merged] == sorted(
            [r.delta_title for r in merged], reverse=True
        )


def _narrowed(state, outlooks, drop_id):
    return state.__class__(
        **{
            **{f.name: getattr(state, f.name) for f in state.__dataclass_fields__.values()},
            "pool": PlayerPool.of(
                [
                    (o.player_id, o.position_id, o.pro_team_id, o.name)
                    for o in outlooks
                    if o.player_id != drop_id
                ]
            ),
        }
    )


def test_augment_leaves_players_on_untouched_pro_teams_alone():
    """`WeeklySampler` keys each player's stream on his id, so a *disjoint* pool widens
    without re-rolling anyone.

    This is the property, stated at the width it actually holds. Every player in this
    fixture sits on his own pro team, so every correlation block is a singleton and the
    id-keyed streams pass straight through -- which is exactly why the version of this
    test that asserted the same equality over a shared-team panel could not fail. See
    `test_augment_does_re_roll_the_free_agents_own_pro_team` for the other half.
    """
    from fantasy_quant.decide.waivers import augment

    weeks = (1, 2, 3, 4)
    fa = _outlook(9001, WR, 90, dict.fromkeys(weeks, 15.0), name="FA")
    state, outlooks = _league(extra=(fa,))
    narrow_state = _narrowed(state, outlooks, 9001)
    narrow = _draw(narrow_state, outlooks, 50, seed=3)
    sim = _fake_sim(narrow_state, outlooks, n_sims=50, seed=3)
    wide = augment(
        sim,
        (
            FreeAgent(
                player_id=9001,
                name="FA",
                position_id=WR,
                pro_team_id=90,
                ros_points=60.0,
                next_points=15.0,
            ),
        ),
        screen_sims=10,
    )
    cols = wide.draw.panel.index_of(narrow_state.pool.player_ids)
    assert np.array_equal(wide.draw.points[:, :, cols], narrow.points)


def test_augment_does_re_roll_the_free_agents_own_pro_team():
    """And the invariance stops at the correlation block, which the docstring used to deny.

    `WeeklySampler._simulate` pushes each week's raw normals through the per-pro-team
    correlation blocks, and `PlayerPool.of` holds ids ascending. So a free agent who
    joins an incumbent's NFL huddle *and sorts ahead of him* moves him down inside the
    block, changes the Cholesky row that mixes him, and re-rolls his whole season -- even
    though his own random stream never moved. Measured on the live Wine Wednesday pool:
    160 of 225 incumbents change, every one of them on a pro team a candidate joined,
    with single player-weeks moving by up to ~370 points.

    Nothing about the board depends on this -- every candidate is priced inside one draw,
    which is what the pairing needs -- but the module said "bit-identical" and it is not,
    and `WaiverReport.baseline_title` is therefore a different universe from
    `pipeline.championship_table`'s. Pinned so the claim cannot come back.
    """
    from fantasy_quant.decide.waivers import augment

    weeks = (1, 2, 3, 4)
    # pro_team_id 10 is team 1's quarterback's NFL team, and id 500 sorts ahead of his
    # 1000, so he stops being row 0 of the block. QB-WR carries the largest measured
    # same-team rho (0.30), so the mixing is not a rounding artefact.
    fa = _outlook(500, WR, 10, dict.fromkeys(weeks, 15.0), name="teammate")
    state, outlooks = _league(extra=(fa,))
    narrow_state = _narrowed(state, outlooks, 500)
    narrow = _draw(narrow_state, outlooks, 50, seed=3)
    sim = _fake_sim(narrow_state, outlooks, n_sims=50, seed=3)
    wide = augment(
        sim,
        (
            FreeAgent(
                player_id=500,
                name="teammate",
                position_id=WR,
                pro_team_id=10,
                ros_points=60.0,
                next_points=15.0,
            ),
        ),
        screen_sims=10,
    )
    shared = wide.draw.panel.index_of([1000])  # the quarterback on pro team 10
    narrow_shared = narrow.panel.index_of([1000])
    assert not np.array_equal(wide.draw.points[:, :, shared], narrow.points[:, :, narrow_shared]), (
        "the correlation block did not re-roll; this test no longer proves anything"
    )

    # The invariant that the board actually rests on, and it holds regardless: inside one
    # draw a null move is exactly zero, so every claim is measured against itself.
    eng = RosterSimulator.build(wide.state, wide.draw, 1)
    assert float(np.abs(eng.marginal_points((), ())).max()) == 0.0


class TestEveryShortlistedPlayerReachesConfirm:
    """Points decide the ORDER of the queue; they must not decide who is measured.

    Both filters ahead of `confirm` ranked on marginal points, and a bench body's
    marginal points with no drop are ~0 because he does not crack the current
    lineup. So the players whose entire value is covering an injury -- exactly what
    a bench slot is for -- were cut before the objective that would have priced
    them was ever computed. On the live board that meant Josh Downs, the third-best
    body on the wire by the pool's own ranking, never appeared while the
    fourth-best available defense did.
    """

    def test_a_points_neutral_candidate_is_still_measured(self):
        """A player who adds nothing to the current lineup must still be priced."""
        report = _board(priority=4)
        adds = [
            next(p.player_id for p in r.move.players if p.to_team is not None) for r in report.board
        ]
        assert len(adds) == len(set(adds)), "one row per player, not one per pairing"
        assert len(report.board) > 1

    def test_the_board_shows_one_row_per_player_not_per_pairing(self):
        report = _board(priority=4, drop_pairs=3)
        adds = [
            next(p.player_id for p in r.move.players if p.to_team is not None) for r in report.board
        ]
        assert len(adds) == len(set(adds))

    def test_drop_pairs_of_one_restores_the_single_pairing_behaviour(self):
        """The knob is real: at 1 each add is committed to its points-best drop."""
        wide = _board(priority=4, drop_pairs=3)
        narrow = _board(priority=4, drop_pairs=1)
        assert len(wide.board) >= len(narrow.board)


class TestWireLevelsCarriesADistribution:
    """The wire floor is a distribution, not a number.

    Crediting an empty seat a constant gave it zero variance, and on a live league
    15.7% of all slot-weeks fall back to that constant -- 70.6% of the WR3 slot.
    The spread turns out to be nearly as large as the mean (WR 6.40 +/- 5.33), so
    treating the floor as a point value discarded most of what a streamed seat does.
    """

    WEEKS = (1, 2, 3)

    def _outlooks(self, specs):
        """specs: list of (player_id, mean, sd, p_zero)."""
        from fantasy_quant.core import PlayerOutlook, WeeklyOutlook

        return [
            PlayerOutlook(
                player_id=pid,
                name=f"p{pid}",
                position_id=3,
                pro_team_id=1,
                weeks={
                    w: WeeklyOutlook(
                        player_id=pid,
                        season=2026,
                        week=w,
                        position_id=3,
                        mean=mean,
                        sd=sd,
                        p_zero=pz,
                        shape=2.0,
                        scale=max(mean / 2, 0.1),
                    )
                    for w in self.WEEKS
                },
            )
            for pid, mean, sd, pz in specs
        ]

    def test_the_mean_matches_wire_floor_exactly(self):
        from fantasy_quant.decide import wire

        outlooks = self._outlooks([(1, 9.0, 7.0, 0.2), (2, 6.0, 5.0, 0.3), (3, 4.0, 3.0, 0.4)])
        el = {4: frozenset({3})}
        levels = wire.wire_levels(outlooks, [], self.WEEKS, el, depth=2)
        floors = wire.wire_floor(outlooks, [], self.WEEKS, el, depth=2)
        assert levels[4].mean == pytest.approx(floors[4])

    def test_the_spread_comes_from_the_body_that_fills_the_seat(self):
        """Not the k-th largest sd -- that would pair one body's mean with another's
        variance. Player 2 is 2nd by projection, so his sd and p_zero are the ones."""
        from fantasy_quant.decide import wire

        outlooks = self._outlooks([(1, 9.0, 1.0, 0.05), (2, 6.0, 5.0, 0.30), (3, 4.0, 9.0, 0.60)])
        level = wire.wire_levels(outlooks, [], self.WEEKS, {4: frozenset({3})}, depth=2)[4]
        assert level.mean == pytest.approx(6.0)
        assert level.sd == pytest.approx(5.0)
        assert level.p_zero == pytest.approx(0.30)

    def test_the_triple_is_reproducible_by_the_hurdle_gamma(self):
        """All three moments come off one body, so the credit can hit them exactly."""
        from fantasy_quant.decide import wire
        from fantasy_quant.projections.calibration import hurdle_gamma_from_moments

        outlooks = self._outlooks([(1, 9.0, 7.0, 0.2), (2, 6.4, 5.3, 0.15)])
        level = wire.wire_levels(outlooks, [], self.WEEKS, {4: frozenset({3})}, depth=2)[4]
        mean, sd = hurdle_gamma_from_moments(level.mean, level.sd, level.p_zero).moments()
        assert mean == pytest.approx(level.mean, abs=1e-6)
        assert sd == pytest.approx(level.sd, abs=1e-3)

    def test_an_empty_wire_reports_no_body_rather_than_a_free_one(self):
        from fantasy_quant.decide import wire

        level = wire.wire_levels([], [], self.WEEKS, {4: frozenset({3})})[4]
        assert (level.mean, level.sd, level.p_zero) == (0.0, 0.0, 1.0)
