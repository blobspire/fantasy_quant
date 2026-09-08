"""The two-tier delta-P(championship) engine.

The acceptance tests here are the ones that would catch a silently wrong answer: a null
move must be *exactly* zero rather than small, a strictly better player must price
positive, the paired error must beat the independent one, a drop must agree with
`season.leave_one_out` to the last bit, and the sign of the variance derivative must
reverse across the playoff cut. Everything runs offline off one synthetic league built
once per session; the live-league checks are marked `network`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from fantasy_quant.core import (
    Move,
    MoveKind,
    PlayerMove,
    PlayerOutlook,
    Recommendation,
    WeeklyOutlook,
)
from fantasy_quant.core import leverage as core_leverage
from fantasy_quant.decide import title as T
from fantasy_quant.decide import wire
from fantasy_quant.projections.calibration import load as load_calibration
from fantasy_quant.sim import season as S
from fantasy_quant.sim.distributions import InjuryModel, SimPanel, WeeklySampler

LEAGUES = [
    (272150391, "Wine Wednesday", 1, 14),
    (161496047, "Blacksburg Baddies", 1, 12),
    (634537479, "Type shi season 2 actually", 2, 12),
]

#: 1QB/2RB/2WR/1TE/1FLEX/1DST/1K, which is what all three real leagues run.
SLOT_COUNTS = {0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1}
SLOT_ELIGIBILITY = {
    0: frozenset({1}),
    2: frozenset({2}),
    4: frozenset({3}),
    6: frozenset({4}),
    16: frozenset({16}),
    17: frozenset({5}),
    23: frozenset({2, 3, 4}),
}
#: Nine starters then a three-man bench, so the flex has a real choice and the bench has
#: option value -- both of which the screen has to price.
POSITIONS = (1, 2, 2, 3, 3, 4, 3, 16, 5, 3, 2, 4)

#: Measured baselines, so a synthetic starter scores like a real one.
_BASE_POINTS = {1: 18.0, 2: 12.0, 3: 11.0, 4: 8.0, 5: 8.0, 16: 7.0}

#: The real fitted calibration, loaded from disk with no network. Building the synthetic
#: league through the same hurdle-gamma fit the pipeline uses is not decoration: a
#: hand-rolled `(p_zero, shape, scale)` is easy to make internally inconsistent -- the
#: pooled 25.9% blank rate for WR applied to an 11-point receiver implies a total SD of
#: 6.7 with essentially no spread in the positive part, which is not a football player --
#: and the engine reads `panel.sd` for its variance model, so an inconsistent fixture
#: would test the screen against a tensor that does not match its own stated moments.
_CALIBRATION = load_calibration("ppr")


def _weekly(pid: int, week: int, pos: int, mean: float, team: int) -> WeeklyOutlook:
    return _CALIBRATION.outlook(
        player_id=pid,
        season=2026,
        week=week,
        position_id=pos if pos in (1, 2, 3, 4) else 0,
        projection=mean,
        pro_team_id=team,
    )


def build_league(
    *,
    n_teams: int = 12,
    strength: list[float] | None = None,
    reg_weeks: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    rounds: tuple[tuple[int, ...], ...] = ((9,), (10,), (11,)),
    playoff_team_count: int = 6,
    n_sims: int = 1200,
    seed: int = 5,
    with_outlooks: bool = False,
) -> tuple[S.LeagueState, object]:
    """A whole league, rosters and tensor, deterministic in `seed`.

    Team `i` scores `strength[i]` times a baseline roster, so the standings are ordered
    by construction and a test can name a team that is above or below the cut without
    having to discover which one it is.
    """
    strength = strength or [0.90 + 0.018 * i for i in range(n_teams)]
    weeks = tuple(sorted(set(reg_weeks) | {w for r in rounds for w in r}))
    rows: list[tuple[int, int, int, str]] = []
    rosters: list[tuple[int, ...]] = []
    outlooks: list[PlayerOutlook] = []
    for t in range(n_teams):
        ids = []
        for j, pos in enumerate(POSITIONS):
            pid = 1000 * (t + 1) + j
            nfl = (t % 30) + 1
            ids.append(pid)
            rows.append((pid, pos, nfl, f"p{pid}"))
            mean = max(_BASE_POINTS[pos] * strength[t] * (1.0 - 0.07 * j), 0.5)
            outlooks.append(
                PlayerOutlook(
                    player_id=pid,
                    name=f"p{pid}",
                    position_id=pos,
                    pro_team_id=nfl,
                    weeks={w: _weekly(pid, w, pos, mean, nfl) for w in weeks},
                )
            )
        rosters.append(tuple(ids))

    order = list(range(n_teams))
    games = []
    for w in reg_weeks:
        for i in range(n_teams // 2):
            games.append(
                S.ScheduledGame(
                    matchup_period=w,
                    weeks=(w,),
                    home_team_id=order[i] + 1,
                    away_team_id=order[n_teams - 1 - i] + 1,
                )
            )
        order = [order[0], order[-1], *order[1:-1]]

    state = S.LeagueState(
        league_id=7,
        season=2026,
        name="synthetic",
        franchises=tuple(
            S.Franchise(team_id=i + 1, name=f"T{i + 1}", player_ids=rosters[i], is_user=(i == 0))
            for i in range(n_teams)
        ),
        pool=S.PlayerPool.of(rows),
        weeks=weeks,
        remaining_games=tuple(games),
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=playoff_team_count,
        playoff_rounds=rounds,
        my_team_id=1,
    )
    draw = WeeklySampler(S.panel_for(state, outlooks), seed=seed).draw(n_sims)
    return (state, draw, outlooks) if with_outlooks else (state, draw)


@pytest.fixture(scope="module")
def league():
    return build_league()


@pytest.fixture(scope="module")
def engine(league):
    state, draw = league
    return T.TitleEngine(state, draw)


def _swap(state, *, incoming: int, outgoing: int) -> Move:
    return T.swap_move(
        state.league_id,
        my_team_id=1,
        incoming=incoming,
        outgoing=outgoing,
        from_team=incoming // 1000,
    )


# --------------------------------------------------------------------------------------
# The surrogate surface
# --------------------------------------------------------------------------------------


class TestSurrogateFit:
    def test_the_surface_reproduces_the_simulated_baseline_at_the_origin(self, engine, league):
        state, _ = league
        for team in (1, state.size // 2, state.size):
            fit = engine.surrogate(team)
            assert fit.title() == pytest.approx(engine.baseline_title(team), abs=0.02)
            assert fit.playoffs() == pytest.approx(engine.baseline_playoffs(team), abs=0.03)

    def test_the_fit_is_tight_enough_to_screen_on(self, engine):
        """RMSE in probability units. A surface that cannot resolve a percentage point
        cannot rank candidates that differ by one."""
        fit = engine.surrogate(1)
        assert fit.rmse < 0.01
        assert fit.playoff_rmse < 0.02

    def test_more_points_is_more_title_probability(self, engine):
        fit = engine.surrogate(5)
        assert fit.title(-10.0) < fit.title(0.0) < fit.title(10.0)
        assert fit.d_title_d_mu() > 0.0

    def test_the_mu_derivative_is_the_right_order_of_magnitude(self, engine):
        """A point a week over a whole season is worth low single-digit pp of title odds.

        Anything near 1pp per point would mean a single waiver claim decides a season,
        which is exactly the overconfidence this whole module is arranged to avoid.
        """
        fit = engine.surrogate(5)
        assert 0.0005 < fit.d_title_d_mu() < 0.05

    def test_a_team_that_never_wins_still_fits(self):
        """A hopeless team wins zero titles at every node, and a plain binomial likelihood
        answers that with `eta -> -inf` and a wall of NaN.

        Over the real perturbation grid, not a degenerate one: a design of all-identical
        rows is rank one and can be solved by the ridge alone, so it would pass whether or
        not the half-count correction was there.
        """
        grid_mu, grid_ratio = np.meshgrid(
            np.array(T.DEFAULT_MU_GRID), np.array(T.DEFAULT_SIGMA_GRID), indexing="ij"
        )
        beta, rmse = T._fit_binomial_surface(grid_mu, grid_ratio, np.zeros_like(grid_mu), 2000)
        assert np.all(np.isfinite(beta))
        assert rmse == pytest.approx(0.0, abs=1e-3)
        assert T._sigmoid(T._design(0.0, 1.0) @ beta) < 0.01
        # and it stays finite everywhere on the grid it was fitted over
        assert np.all(np.isfinite(T._sigmoid(T._design(grid_mu, grid_ratio) @ beta)))

    def test_a_team_that_always_wins_still_fits(self):
        grid_mu, grid_ratio = np.meshgrid(
            np.array(T.DEFAULT_MU_GRID), np.array(T.DEFAULT_SIGMA_GRID), indexing="ij"
        )
        beta, _ = T._fit_binomial_surface(grid_mu, grid_ratio, np.full_like(grid_mu, 2000.0), 2000)
        assert np.all(np.isfinite(beta))
        assert T._sigmoid(T._design(0.0, 1.0) @ beta) > 0.99

    def test_a_sparse_but_nonzero_team_is_not_shrunk_away_by_the_correction(self):
        """The Jeffreys half-count must not move a node that has real mass.

        Claimed at "under 0.03pp"; asserted rather than trusted, because the correction is
        the one thing standing between a low-probability team and a NaN surface and it
        would be easy to over-apply.
        """
        grid_mu, grid_ratio = np.meshgrid(
            np.array(T.DEFAULT_MU_GRID), np.array(T.DEFAULT_SIGMA_GRID), indexing="ij"
        )
        truth = 0.02 + 0.001 * grid_mu
        beta, rmse = T._fit_binomial_surface(grid_mu, grid_ratio, truth * 2000, 2000)
        assert rmse < 0.003
        assert float(T._sigmoid(T._design(0.0, 1.0) @ beta)) == pytest.approx(0.02, abs=0.003)


@pytest.fixture(scope="module")
def near_cut():
    """Strengths spread so the field straddles the cut rather than splitting into locks
    and no-hopers, which is where the derivative would be zero and a sign test vacuous."""
    return build_league(
        n_teams=12,
        playoff_team_count=6,
        strength=[0.88 + 0.022 * i for i in range(12)],
        n_sims=2000,
        seed=11,
    )


@pytest.fixture(scope="module")
def near_cut_engine(near_cut):
    state, draw = near_cut
    return T.TitleEngine(state, draw)


class TestVarianceDerivativeFlipsAtTheCut:
    """The known failure mode, pinned.

    A surface fitted across standings would carry ONE sign for dP/dsigma and would
    therefore mis-screen every variance-changing move for one whole half of the league.
    """

    def test_the_playoff_variance_derivative_reverses_across_the_cut(
        self, near_cut, near_cut_engine
    ):
        state, _ = near_cut
        engine = near_cut_engine
        below = engine.surrogate(1)
        above = engine.surrogate(state.size)
        assert below.baseline_playoffs < 0.5 < above.baseline_playoffs
        assert below.d_playoffs_d_sigma() > 0.0, "a team below the cut must want variance"
        assert above.d_playoffs_d_sigma() < 0.0, "a team above the cut must reject it"

    def test_a_favourite_also_rejects_variance_on_the_title_surface(
        self, near_cut, near_cut_engine
    ):
        state, _ = near_cut
        engine = near_cut_engine
        assert engine.surrogate(state.size).d_title_d_sigma() < 0.0
        assert engine.surrogate(1).d_title_d_sigma() > 0.0

    def test_one_pooled_surface_must_mis_sign_one_of_them(self, near_cut, near_cut_engine):
        """Fit a single surface over both teams' nodes and watch it get one team wrong.

        This is the concrete failure a global surrogate produces: the pooled fit lands on
        one sign, and whichever sign it lands on, the team on the other side of the cut is
        now being told to buy variance when it should sell it.
        """
        state, _ = near_cut
        engine = near_cut_engine
        below, above = engine.surrogate(1), engine.surrogate(state.size)
        grid_mu, grid_ratio = np.meshgrid(
            np.array(below.mu_grid), np.array(below.sigma_grid), indexing="ij"
        )
        pooled, _ = T._fit_binomial_surface(
            np.concatenate([grid_mu, grid_mu]),
            np.concatenate([grid_ratio, grid_ratio]),
            np.concatenate([below.playoff_nodes, above.playoff_nodes]) * below.n_sims,
            below.n_sims,
        )
        # The pooled slope in the sigma direction at the origin.
        p = float(T._sigmoid(T._design(0.0, 1.0) @ pooled))
        pooled_slope = p * (1.0 - p) * pooled[2]
        per_team = [below.d_playoffs_d_sigma(), above.d_playoffs_d_sigma()]
        assert any(np.sign(pooled_slope) != np.sign(s) for s in per_team)


# --------------------------------------------------------------------------------------
# Acceptance: the numbers a recommendation is made of
# --------------------------------------------------------------------------------------


class TestNullMoves:
    def test_a_hold_is_exactly_zero(self, engine, league):
        state, _ = league
        move = Move(kind=MoveKind.HOLD, league_id=state.league_id)
        for rec in (engine.screen([move])[0], engine.confirm([move])[0]):
            assert rec.delta_title == 0.0
            assert rec.stderr == 0.0
            assert "null" in rec.tags

    def test_a_lineup_move_is_zero_and_says_why(self, engine, league):
        """The simulator already starts the ex-ante optimal lineup, so there is nothing
        left to gain by telling it to. Pretending otherwise would be a fabricated edge."""
        state, _ = league
        move = Move(kind=MoveKind.LINEUP, league_id=state.league_id, lineup={0: 1000})
        rec = engine.confirm([move])[0]
        assert rec.delta_title == 0.0
        assert "optimal lineup" in rec.rationale

    def test_a_player_traded_to_his_own_team_returns_exactly_zero(self, engine, league):
        """The move goes all the way through the simulator rather than short-circuiting,
        so this is the real common-random-numbers check: identical football, identical
        bracket, a difference of exactly 0.0 rather than Monte Carlo fog."""
        state, _ = league
        move = Move(
            kind=MoveKind.TRADE,
            league_id=state.league_id,
            players=(PlayerMove(player_id=1001, from_team=1, to_team=1),),
        )
        rec = engine.confirm([move])[0]
        assert rec.delta_title == 0.0
        assert rec.stderr == 0.0
        assert "null" not in rec.tags, "this one must have been simulated, not short-cut"


class TestAStrictlyBetterPlayer:
    """The same slot, a strictly higher projection every week, everything else held."""

    def test_confirm_prices_the_upgrade_positive(self, engine, league):
        state, _ = league
        # Team 1 is the weakest roster; team 10's RB1 dominates team 1's RB3 in every week.
        move = _swap(state, incoming=10001, outgoing=1010)
        rec = engine.confirm([move])[0]
        assert rec.delta_points > 0.0
        assert rec.delta_title > 0.0
        assert rec.significant, f"{rec.delta_title} +/- {rec.stderr}"

    def test_screen_prices_the_upgrade_positive(self, engine, league):
        state, _ = league
        rec = engine.screen([_swap(state, incoming=10001, outgoing=1010)])[0]
        assert rec.delta_title > 0.0
        assert rec.delta_points > 0.0

    def test_the_reverse_trade_is_priced_negative(self, engine, league):
        """Symmetry. Handing your best running back away cannot come out positive."""
        state, _ = league
        move = T.swap_move(
            state.league_id, my_team_id=1, incoming=10010, outgoing=1001, from_team=10
        )
        assert engine.confirm([move])[0].delta_title < 0.0

    def test_a_dropped_starter_costs_title_probability(self, engine, league):
        state, _ = league
        rec = engine.confirm([T.drop_move(state.league_id, team_id=1, player_id=1001)])[0]
        assert rec.delta_title < 0.0


class TestCommonRandomNumbers:
    def test_the_paired_error_beats_the_independent_one(self, engine, league):
        state, _ = league
        move = _swap(state, incoming=10001, outgoing=1010)
        paired = engine.confirm([move])[0].stderr
        assert 0.0 < paired < engine.independent_stderr(move)

    def test_pairing_removes_far_more_noise_from_wins_than_from_the_title(self, engine, league):
        """The measured ordering, and it is the reason this module screens on a surface.

        Common random numbers are worth orders of magnitude on points-for, a lot on wins,
        and comparatively little on the championship indicator -- a bracket is three coin
        flips laid on top of the season, and pairing cannot remove a coin flip.
        """
        state, _ = league
        ratios = engine.crn_variance_ratio(_swap(state, incoming=10001, outgoing=1010))
        assert ratios["title"] > 1.0
        assert ratios["wins"] > ratios["title"]
        assert ratios["points_for"] > ratios["wins"]

    def test_a_null_move_has_zero_paired_error_against_a_real_independent_one(self, engine, league):
        """The cleanest proof there is: the paired error is 0 where the independent error
        is half a percentage point, so the ratio is not large, it is infinite."""
        state, _ = league
        move = Move(
            kind=MoveKind.TRADE,
            league_id=state.league_id,
            players=(PlayerMove(player_id=1001, from_team=1, to_team=1),),
        )
        assert engine.confirm([move])[0].stderr == 0.0
        assert engine.independent_stderr(move) > 0.001

    def test_confirm_agrees_with_leave_one_out_on_a_drop(self, engine, league):
        """`season.leave_one_out` prices exactly the same question by a different route.

        If these two ever disagree, one of them is applying the efficiency factors or the
        replacement floor differently, which is silent and would poison every number the
        product prints.

        Player 1002 is one of two running backs, so dropping him leaves the RB group with
        somebody in it. That matters: the two routes genuinely diverge when a drop empties
        a whole position group, and the reason is pinned in
        `TestTheFreeAgentFloor.test_emptying_a_position_does_not_lift_every_other_slot`.
        """
        state, draw = league
        contribution = S.leave_one_out(
            state,
            draw,
            player_ids=[1002],
            efficiency=engine._factors,
            all_play=False,
            replacement=engine.replacement,
        )[0]
        rec = engine.confirm([T.drop_move(state.league_id, team_id=1, player_id=1002)])[0]
        assert rec.delta_title == pytest.approx(-contribution.title_added, abs=1e-12)
        assert rec.stderr == pytest.approx(contribution.title_added_stderr, abs=1e-12)


class TestTheFreeAgentFloor:
    """An unfilled slot streams a replacement; it does not sit empty for the season.

    Getting this wrong is not a rounding error, it inverts the board. Measured on the
    user's real leagues against an empty seat, dropping Harrison Butker priced at -4.15pp
    of title probability against Justin Jefferson's -3.50pp, and Evan McPherson at -6.53pp
    against Jonathan Taylor's -5.67pp. With the streaming level fitted, both kickers come
    in under a point and the first-round backs are back on top.
    """

    def test_the_engine_fits_a_floor_rather_than_leaving_the_seat_empty(self, engine):
        assert isinstance(engine.replacement, dict) and engine.replacement
        # Levels now, not bare floats: the seat has to be PAID, and paying it the mean
        # gives it zero variance. The claim is unchanged -- a real floor, not empty.
        assert all(v.mean > 0.0 for v in engine.replacement.values())

    def test_the_floor_is_ordered_like_the_positions_it_prices(self, league):
        state, draw = league
        levels = T.streaming_replacement(state, draw)
        # QB is the highest-scoring position and the shallowest demand; the D/ST slot is
        # the cheapest thing on the wire. Anything else and the levels are transposed.
        assert levels[0] > levels[2] > levels[16]
        assert levels[23] >= min(levels[2], levels[4])

    def test_a_dropped_singleton_is_priced_against_the_wire_not_against_nothing(self, league):
        """The mechanism behind the kicker inversion, on a team the fixture can express.

        Priced for the STRONGEST team, not the weakest: team 1 is the worst roster in a
        league built by scaling one template, so its kicker and its quarterback *are* the
        replacement level by construction and every floor comparison there is a tie. The
        inversion itself needs real projections -- a real kicker projects 8 a week where
        this fixture's projects 3 -- so it is asserted against the live leagues in
        `TestAgainstTheRealLeagues.test_a_kicker_does_not_outrank_the_best_starter`.
        """
        state, draw = league
        team = state.size
        ids = state.franchise(team).player_ids
        kicker = next(p for p in ids if state.pool.positions_of([p])[0] == 5)
        drop = T.drop_move(state.league_id, team_id=team, player_id=kicker)
        empty = T.TitleEngine(state, draw, replacement=0.0).confirm([drop])[0]
        floored = T.TitleEngine(state, draw).confirm([drop])[0]
        assert empty.delta_points < floored.delta_points < 0.0
        # The K slot is the only one that empties, so the whole difference is its floor,
        # every week, with nothing else moving.
        level = T.streaming_replacement(state, draw)[17]
        gap = floored.delta_points - empty.delta_points
        assert gap == pytest.approx(level * len(state.weeks), rel=0.05)

    def test_emptying_a_position_does_not_lift_every_other_slot(self, league):
        """`lineup.monotone_floor` computes nesting from the ROSTER's eligibility, so a
        roster with nobody at a position has an empty group -- a subset of every other
        group -- and every slot on the team is lifted to the missing position's floor.

        Unguarded, dropping the only quarterback made the user's Blacksburg roster score
        137.6 points a week with zero variance and a 93% title probability. Here the same
        drop must simply cost points, and the team's weekly spread must survive it.
        """
        state, draw = league
        engine = T.TitleEngine(state, draw)
        team = state.size
        ids = state.franchise(team).player_ids
        assert (state.pool.positions_of(ids) == 1).sum() == 1, "fixture must have one QB"
        rec = engine.confirm([T.drop_move(state.league_id, team_id=team, player_id=ids[0])])[0]
        assert rec.delta_points < 0.0, "losing your only QB cannot add points"
        assert rec.delta_title < 0.0, f"losing your only QB priced at {rec.delta_title}"
        without = engine._roster_moments(tuple(ids[1:]))[1]
        assert np.all(without > 0.0), "the whole lineup collapsed to a deterministic floor"

    def test_the_correction_is_exactly_the_floor_it_held_out(self, league):
        """The held-out floor is deterministic, so the corrected mean lands on the
        hand-computed one exactly rather than merely near it.

        Isolated by flooring the QB slot alone: against an all-zero floor, a roster with
        no quarterback must come out higher by precisely the QB level, every week.
        """
        state, draw = league
        ids = state.franchise(state.size).player_ids
        qb_less = tuple(p for p in ids if state.pool.positions_of([p])[0] != 1)
        level = T.streaming_replacement(state, draw)[0]
        only_qb = dict.fromkeys(T.streaming_replacement(state, draw), 0.0) | {0: level}
        bare = T.TitleEngine(state, draw, replacement=0.0)._roster_moments(qb_less)[0]
        floored = T.TitleEngine(state, draw, replacement=only_qb)._roster_moments(qb_less)[0]
        assert floored == pytest.approx(bare + level, abs=1e-9)


class TestScreenAgreesWithConfirm:
    def test_the_screen_ranks_like_the_confirmation(self, engine, league):
        state, _ = league
        moves = T.one_for_one_candidates(state, 1, opponents=[8, 9, 10])[:120]
        report = engine.agreement(moves, shortlist=30, top_k=8)
        assert report.spearman > 0.7, report
        assert report.recall_at >= 0.75, report

    def test_the_screen_is_far_cheaper_than_the_confirmation(self, engine, league):
        """Not a timing assertion on wall clock -- a ratio, which is stable enough to
        pin. If the screen ever stops being an order of magnitude cheaper it has stopped
        being a screen."""
        state, _ = league
        moves = T.one_for_one_candidates(state, 1, opponents=[9])[:60]
        engine.screen(moves)  # warm the surrogate cache, which is the one-time cost
        report = engine.agreement(moves, shortlist=20, top_k=5)
        assert report.per_screen_ms * 10 < report.per_confirm_ms

    def test_the_ex_ante_order_statistic_beats_clarks_hindsight_max(self, engine, league):
        """The research note's `E[max]` over realised points prices a lineup nobody can
        set. Measured rather than argued: it ranks candidates worse."""
        state, _ = league
        moves = T.one_for_one_candidates(state, 1, opponents=[8, 9, 10])[:120]
        ours = engine.agreement(moves, shortlist=30, top_k=8)
        theirs = engine.agreement(moves, shortlist=30, top_k=8, hindsight_max=True)
        assert ours.spearman > theirs.spearman

    def test_the_screen_is_calibrated_in_level_and_not_only_in_order(self, engine, league):
        """A ranking key is not enough: other surfaces will print the screened number.

        The slope of confirm on screen comes out at 0.94-1.01 on the user's real leagues,
        so a screened `delta_title` can be read as a percentage point rather than only as
        a position in a list.
        """
        state, _ = league
        moves = T.one_for_one_candidates(state, 1, opponents=[8, 9, 10])[:120]
        assert 0.6 < engine.agreement(moves, shortlist=30, top_k=8).scale < 1.6

    def test_evaluate_confirms_only_the_shortlist(self, engine, league):
        state, _ = league
        moves = T.one_for_one_candidates(state, 1, opponents=[10])
        out = engine.evaluate(moves, keep=12)
        assert len(out) == 12
        assert all("confirm" in r.tags for r in out)
        assert out == sorted(out, reverse=True)


# --------------------------------------------------------------------------------------
# Leverage, and the contract
# --------------------------------------------------------------------------------------


class TestAMoveNobodyWouldAccept:
    """A `delta_title` is not a recommendation until somebody would agree to it.

    `one_for_one_candidates` proposes every swap in the league, so the top of any
    unfiltered ranking is a rival's best player for a bench body. On the user's three
    real leagues the headline of `evaluate` was "+Jahmyr Gibbs, -Keaton Mitchell,
    +11.95pp" on all three -- arithmetically correct, and not a move that exists.
    """

    def test_a_lopsided_trade_is_labelled_unilateral(self, engine, league):
        state, _ = league
        rec = engine.confirm([_swap(state, incoming=10001, outgoing=1010)])[0]
        assert rec.delta_title > 0.0
        assert "unilateral" in rec.tags
        assert "not a trade they accept" in rec.rationale
        assert state.franchise(10).name in rec.rationale

    def test_the_screen_reaches_the_same_verdict_for_free(self, engine, league):
        state, _ = league
        rec = engine.screen([_swap(state, incoming=10001, outgoing=1010)])[0]
        assert "unilateral" in rec.tags

    def test_evaluate_can_drop_the_moves_nobody_would_accept(self, engine, league):
        state, _ = league
        moves = T.one_for_one_candidates(state, 1, opponents=[10])
        loose = engine.evaluate(moves, keep=25)
        gated = engine.evaluate(moves, keep=25, acceptable_only=True)
        assert any("unilateral" in r.tags for r in loose), "fixture has no robberies to gate"
        assert not any("unilateral" in r.tags for r in gated)
        assert gated, "the gate removed every candidate, so it is not a gate"
        assert max(r.delta_title for r in gated) < max(r.delta_title for r in loose)

    def test_an_add_with_no_drop_is_labelled_rather_than_ranked_beside_legal_moves(
        self, engine, league
    ):
        """`Move` can name an add with no drop and nothing below here knows a league has a
        roster limit, so "+De'Von Achane" alone priced at +5.85pp on a real league."""
        state, _ = league
        claim = Move(
            kind=MoveKind.TRADE,
            league_id=state.league_id,
            players=(PlayerMove(player_id=10001, from_team=10, to_team=1),),
        )
        rec = engine.confirm([claim])[0]
        assert rec.delta_title > 0.0
        assert "roster_size" in rec.tags
        assert "changes a roster's size" in rec.rationale

    def test_a_balanced_swap_is_not_flagged(self, engine, league):
        state, _ = league
        rec = engine.confirm([_swap(state, incoming=10001, outgoing=1010)])[0]
        assert "roster_size" not in rec.tags


class TestClarksHindsightMax:
    """The research note's `E[max]`, implemented as the note states it.

    `E[max(X, Y)] - E[X] = theta * phi(alpha) + (mu_y - mu_x) * Phi(-alpha)`. Dropping the
    second term -- which the module did -- leaves a bonus 3.5 times too large and quietly
    decides the comparison against the ex-ante order statistic in the ex-ante statistic's
    favour, which is exactly the comparison the module reports on.
    """

    def test_the_bonus_matches_a_brute_forced_expectation(self):
        """Against `title.clark_bonus` itself, which is what `_hindsight_moments` calls.

        Re-deriving the closed form inside the test and comparing it to Monte Carlo would
        prove the test's arithmetic and leave the module's free to be wrong -- which is
        precisely the state this found.
        """
        rng = np.random.default_rng(11)
        cases = ((12.0, 7.0, 0.0), (12.0, 7.0, 9.0), (4.0, 3.0, 6.0), (25.0, 5.0, 8.0))
        for mu, sd, floor in cases:
            x = rng.normal(mu, sd, 400_000)
            y = rng.normal(floor, sd, 400_000)
            brute = float(np.maximum(x, y).mean() - x.mean())
            assert float(T.clark_bonus(mu, sd, floor)) == pytest.approx(brute, abs=0.03)

    def test_the_bonus_vanishes_when_the_starter_always_beats_the_wire(self):
        """A starter thirty points clear of the floor has no option value in him. The
        version that dropped the `Phi` term still paid 1.4 points here."""
        assert float(T.clark_bonus(40.0, 5.0, 5.0)) < 0.01
        assert float(T.clark_bonus(40.0, 5.0, 5.0)) < float(T.clark_bonus(10.0, 5.0, 5.0))

    def test_the_hindsight_lineup_is_worth_more_than_the_ex_ante_one(self, engine, league):
        """`E[max] > max[E]`: the gap is the bench option value a manager cannot realise.

        Bounded by the analytic maximum of the bonus, `0.399 * sqrt(2) * sd` a slot, which
        is attained when the starter sits exactly on the floor.
        """
        state, _ = league
        ids = state.franchise(1).player_ids
        ex_ante, _ = engine._roster_moments(ids)
        hindsight, _ = engine._hindsight_moments(ids)
        gap = float(np.mean(hindsight - ex_ante))
        ceiling = 0.3990 * math.sqrt(2.0) * float(np.max(engine._sd)) * 9
        assert 0.0 < gap < ceiling, (gap, ceiling)


class TestTheWaiverBoardCanReachThisEngine:
    def test_evaluator_for_exists_and_satisfies_the_protocol(self, league):
        """`waivers.default_evaluator` looks up `title.evaluator_for` by name and falls
        back to its own simulator when it is missing, which it was."""
        from fantasy_quant.decide import waivers as W

        state, draw = league
        evaluator = W.default_evaluator(state, draw, 3)
        assert isinstance(evaluator, T.TitleEngine)
        assert evaluator.subject_team(T.drop_move(state.league_id, team_id=3, player_id=3001)) == 3

    def test_the_default_evaluator_gets_a_floor_and_not_an_empty_seat(self, league):
        state, draw = league
        from fantasy_quant.decide import waivers as W

        evaluator = W.default_evaluator(state, draw, 1)
        assert isinstance(evaluator.replacement, dict)
        assert all(v.mean > 0.0 for v in evaluator.replacement.values())


class TestLeverage:
    def test_every_matchup_reports_a_leverage_in_range(self, engine, league):
        state, _ = league
        rows = engine.week_leverage(1)
        mine = [g for g in state.remaining_games if 1 in (g.home_team_id, g.away_team_id)]
        assert len(rows) == len(mine)
        assert all(0.0 <= r.leverage <= 1.0 for r in rows)
        assert all(0.0 <= r.win_probability <= 1.0 for r in rows)

    def test_the_spread_of_a_matchup_margin_matches_the_measured_constant(self, engine, league):
        """sd_diff = 34.4 on the corpus, 26.5-28.7 on the user's three real leagues.

        Two claims, because the gap between those numbers is itself a property worth
        pinning. The spread has to be in the right neighbourhood -- the original window of
        20 to 50 would have admitted a conversion constant off by a third either way --
        and flooring an unfilled slot at a constant has to be the thing that lowers it,
        since a streamed replacement really varies and `sim/season._floors` pretends it
        does not. If that ordering ever reverses, the floor has stopped being a floor.
        """
        state, draw = league
        rows = engine.week_leverage(1)
        spread = float(np.mean([r.sd_diff for r in rows]))
        assert 18.0 < spread < 40.0
        empty = T.TitleEngine(state, draw, replacement=0.0)
        unfloored = float(np.mean([r.sd_diff for r in empty.week_leverage(1)]))
        assert spread < unfloored, "constant floors must not raise the modelled spread"
        assert spread > 0.6 * unfloored, "the floors removed most of the team's variance"

    def test_a_mismatch_has_less_leverage_than_a_coin_flip(self, engine, league):
        """The most valuable thing the tool can say is 'this week does not matter'.

        The previous form of this test filtered for `abs(margin) > sd_diff` and asserted
        over the survivors: the fixture never produces one, so `all(...)` ran over an
        empty list and the test passed no matter what `leverage` returned. Compare the
        weakest team's most lopsided remaining matchup against its most even one instead,
        which is a claim the fixture can actually express.
        """
        _ = league
        rows = engine.week_leverage(1)
        assert len(rows) > 1
        by_z = sorted(rows, key=lambda r: abs(r.margin) / r.sd_diff)
        even, lopsided = by_z[0], by_z[-1]
        assert abs(lopsided.margin) > abs(even.margin)
        assert lopsided.leverage < even.leverage
        # And leverage is the analytic function of z it claims to be, not a proxy.
        for row in rows:
            assert row.leverage == pytest.approx(
                math.exp(-0.5 * (row.margin / row.sd_diff) ** 2), rel=1e-9
            )

    def test_a_point_is_worth_about_a_point_of_win_probability(self, engine):
        rows = engine.week_leverage(1)
        even = min(rows, key=lambda r: abs(r.margin))
        assert 0.5 < even.points_per_win_pct < 2.0


class TestTheContract:
    def test_the_engine_satisfies_the_move_evaluator_protocol(self, engine, league):
        state, _ = league
        evaluator = engine  # typed as core.MoveEvaluator at every call site
        move = Move(kind=MoveKind.HOLD, league_id=state.league_id)
        assert isinstance(evaluator.screen([move])[0], Recommendation)
        assert isinstance(evaluator.confirm([move])[0], Recommendation)
        assert isinstance(evaluator.baseline_title(1), float)

    def test_an_effect_inside_its_own_error_is_not_reported_as_significant(self, engine, league):
        """Swapping two near-identical bench bodies cannot be dressed up as an edge.

        Asserted outright rather than under `if abs(delta) <= 2 * stderr:` -- guarding the
        assertion with the very condition it is checking makes the test pass whenever the
        thing it exists to catch does not happen to occur.
        """
        state, _ = league
        # A receiver swapped for the next team's receiver: a real but tiny edge, which is
        # the case that has to be labelled. Two deep bench bodies would come back as
        # *exactly* zero now that an unfilled slot streams a replacement -- bench depth
        # below the wire is worth nothing -- and zero with zero error reads as
        # significant under `core.Recommendation`, which is a different test.
        rec = engine.confirm([_swap(state, incoming=2003, outgoing=1006)])[0]
        assert 0.0 < abs(rec.delta_title) <= 2.0 * rec.stderr, (
            f"the fixture stopped producing a marginal swap: {rec.delta_title} +/- {rec.stderr}"
        )
        assert not rec.significant
        assert "inside the error bar" in rec.rationale

    def test_every_recommendation_carries_an_actionable_rationale(self, engine, league):
        state, _ = league
        rec = engine.confirm([_swap(state, incoming=10001, outgoing=1010)])[0]
        assert rec.move.kind is MoveKind.TRADE
        assert "title" in rec.rationale and "%" in rec.rationale
        assert "starting-lineup points" in rec.rationale
        assert 0.0 <= rec.leverage <= 1.0

    def test_screen_reports_the_surfaces_own_error_rather_than_zero(self, engine, league):
        """A screened estimate with `stderr = 0` would report every rounding artifact as
        significant. The surrogate's RMSE is the floor, and it is only the floor.

        The screen's real disagreement with `confirm` grows with the size of the effect --
        measured at roughly 0.25pp plus a fifth of it -- so the RMSE alone would call a
        +12pp screened trade accurate to a third of a point. Both terms are checked here:
        a tiny move reports the RMSE, a large one reports materially more.
        """
        state, _ = league
        rmse = engine.surrogate(1).rmse
        tiny = engine.screen([_swap(state, incoming=5011, outgoing=1011)])[0]
        assert tiny.stderr == pytest.approx(rmse, rel=0.05)
        assert tiny.confidence == "low"
        big = engine.screen([_swap(state, incoming=10001, outgoing=1010)])[0]
        assert big.stderr > 2.0 * rmse, "a large screened effect must carry a larger error"
        assert big.stderr == pytest.approx(
            math.hypot(rmse, T.SCREEN_RELATIVE_ERROR * big.delta_title), rel=1e-9
        )

    def test_a_player_outside_the_panel_is_a_loud_error(self, engine, league):
        state, _ = league
        move = Move(
            kind=MoveKind.WAIVER_CLAIM,
            league_id=state.league_id,
            players=(PlayerMove(player_id=999999, from_team=None, to_team=1),),
        )
        with pytest.raises(T.TitleError, match="drawn panel"):
            engine.confirm([move])

    def test_a_move_from_a_team_that_does_not_roster_the_player_is_refused(self, engine, league):
        state, _ = league
        move = Move(
            kind=MoveKind.TRADE,
            league_id=state.league_id,
            players=(PlayerMove(player_id=1001, from_team=2, to_team=3),),
        )
        with pytest.raises(T.TitleError, match="does not roster"):
            engine.confirm([move])

    def test_the_subject_of_a_trade_is_the_user(self, engine, league):
        state, _ = league
        assert engine.subject_team(_swap(state, incoming=10001, outgoing=1010)) == 1

    def test_a_trade_can_be_priced_from_the_counterpartys_side(self, engine, league):
        """A proposal is only accepted if it is positive for them too, so a trade surface
        has to be able to ask. The two sides of a lopsided swap must have opposite signs."""
        state, _ = league
        move = _swap(state, incoming=10001, outgoing=1010)
        mine = engine.confirm([move])[0]
        theirs = engine.confirm([move], subject=10)[0]
        assert mine.delta_title > 0.0 > theirs.delta_title
        assert theirs.rationale.startswith(state.franchise(10).name)

    def test_pricing_for_a_team_the_league_does_not_have_is_refused(self, engine, league):
        state, _ = league
        with pytest.raises(T.TitleError, match="no team"):
            engine.confirm([_swap(state, incoming=10001, outgoing=1010)], subject=99)


class TestRosterMoments:
    def test_a_bye_week_costs_the_slot_its_starter(self, league):
        """Availability is read off the drawn tensor, so a week a player cannot play must
        show up as a lower expected lineup mean, not as a full-strength week."""
        state, draw = league
        engine = T.TitleEngine(state, draw)
        mean, var = engine._roster_moments(state.franchise(1).player_ids)
        assert mean.shape == (len(state.weeks),)
        assert np.all(mean > 0.0)
        assert np.all(var > 0.0)

    def test_a_better_roster_has_a_higher_expected_lineup(self, league):
        state, draw = league
        engine = T.TitleEngine(state, draw)
        weak, _ = engine._roster_moments(state.franchise(1).player_ids)
        strong, _ = engine._roster_moments(state.franchise(state.size).player_ids)
        assert strong.mean() > weak.mean()

    def test_the_modelled_mean_is_within_reach_of_the_simulated_one(self, engine, league):
        """The screen only ever uses differences, so a bias here is survivable -- but a
        bias of tens of points would mean the depth-chart model is describing a different
        roster, and the sigma ratio it feeds the surface would be nonsense."""
        state, _ = league
        modelled, _ = engine._roster_moments(state.franchise(1).player_ids)
        simulated = np.array(engine.moments(1).mean_by_week)
        assert abs(modelled.mean() - simulated.mean()) < 0.15 * simulated.mean()


# --------------------------------------------------------------------------------------
# The real leagues
# --------------------------------------------------------------------------------------


@pytest.mark.network
class TestAgainstTheRealLeagues:
    @pytest.fixture(scope="class")
    def client(self):
        from fantasy_quant import pipeline as P

        c = P.client_from_env()
        yield c
        c.close()

    @pytest.mark.parametrize("league_id,name,my_team,size", LEAGUES)
    def test_the_engine_runs_and_the_odds_are_believable(
        self, client, league_id, name, my_team, size
    ):
        from fantasy_quant import pipeline as P

        sim = P.build(league_id, 2026, my_team_id=my_team, client=client, n_sims=1000)
        engine = T.TitleEngine.from_sim(sim)
        total = sum(engine.baseline_title(f.team_id) for f in sim.state.franchises)
        assert total == pytest.approx(1.0, abs=1e-6)
        # The user is below average by projection in all three; a surface that says
        # otherwise is the failure this project exists to avoid.
        assert engine.baseline_title(my_team) < 1.5 / size

    def test_a_real_trade_prices_and_the_screen_agrees_with_the_confirmation(self, client):
        from fantasy_quant import pipeline as P

        sim = P.build(161496047, 2026, my_team_id=1, client=client, n_sims=1000)
        engine = T.TitleEngine.from_sim(sim)
        moves = T.one_for_one_candidates(sim.state, 1, opponents=[2, 3])[:150]
        # Warm the surrogate cache first: a fit is half a second and belongs to the team,
        # not to the candidate, so timing it per candidate would understate the screen by
        # two orders of magnitude on a 150-move sample.
        engine.screen(moves)
        report = engine.agreement(moves, shortlist=40, top_k=10)
        assert report.spearman > 0.7, report
        assert 0.6 < report.scale < 1.6, report
        assert report.per_screen_ms * 10 < report.per_confirm_ms

    @pytest.mark.parametrize("league_id,name,my_team,size", LEAGUES)
    def test_a_kicker_does_not_outrank_the_best_starter(
        self, client, league_id, name, my_team, size
    ):
        """The inversion an empty-seat floor produces, checked against real projections.

        With `replacement=0.0` this fails on all three of the user's leagues: Harrison
        Butker prices at -4.15pp against Justin Jefferson's -3.50pp, and Evan McPherson at
        -6.53pp against Jonathan Taylor's -5.67pp. A board that tells the user his kicker
        is his second most valuable asset is a board he should ignore, so this is pinned
        against the live rosters rather than against a fixture that cannot express it.
        """
        from fantasy_quant import pipeline as P

        sim = P.build(league_id, 2026, my_team_id=my_team, client=client, n_sims=1000)
        engine = T.TitleEngine.from_sim(sim)
        ids = sim.state.franchise(my_team).player_ids
        positions = dict(zip(ids, sim.state.pool.positions_of(ids), strict=True))
        recs = engine.confirm([T.drop_move(league_id, team_id=my_team, player_id=p) for p in ids])
        cost = dict(zip(ids, (r.delta_title for r in recs), strict=True))
        streamable = [cost[p] for p in ids if positions[p] in (5, 16)]
        skill = [cost[p] for p in ids if positions[p] in (1, 2, 3, 4)]
        assert min(skill) < min(streamable), (
            f"{name}: a kicker or defence is priced above every skill player -- "
            f"streamable {min(streamable):.4f} vs skill {min(skill):.4f}"
        )

    def test_the_variance_derivative_still_flips_on_a_real_league(self, client):
        from fantasy_quant import pipeline as P

        sim = P.build(272150391, 2026, my_team_id=1, client=client, n_sims=2000)
        engine = T.TitleEngine.from_sim(sim)
        odds = sorted(
            ((engine.baseline_playoffs(f.team_id), f.team_id) for f in sim.state.franchises),
        )
        worst, best = odds[0][1], odds[-1][1]
        assert engine.surrogate(worst).d_playoffs_d_sigma() > 0.0
        assert engine.surrogate(best).d_playoffs_d_sigma() < 0.0


class TestFreeAgentsCanBePriced:
    """Without an extended panel no waiver claim can be evaluated at all: `pipeline.build`
    pools only rostered players, and a move naming anyone else indexes off the tensor."""

    def test_a_move_naming_a_free_agent_is_refused_before_the_panel_is_extended(
        self, engine, league
    ):
        state, _ = league
        claim = Move(
            kind=MoveKind.WAIVER_CLAIM,
            league_id=state.league_id,
            players=(PlayerMove(player_id=424242, from_team=None, to_team=1),),
        )
        with pytest.raises(T.TitleError, match="sim_with_free_agents"):
            engine.screen([claim])

    def test_extending_the_panel_makes_the_claim_priceable(self):
        state, draw, outlooks = build_league(
            n_teams=6,
            playoff_team_count=4,
            n_sims=400,
            reg_weeks=(1, 2, 3, 4),
            rounds=((5,), (6,)),
            with_outlooks=True,
        )
        wire = PlayerOutlook(
            player_id=424242,
            name="A Free Agent",
            position_id=2,
            pro_team_id=31,
            weeks={w: _weekly(424242, w, 2, 16.0, 31) for w in state.weeks},
        )
        # And some scrubs behind him. Without these he IS the wire, so the floor
        # equals his own projection and claiming him is correctly worth nothing --
        # a claim is only worth what it beats.
        scrubs = [
            PlayerOutlook(
                player_id=424243 + i,
                name=f"Wire body {i}",
                position_id=2,
                pro_team_id=31,
                weeks={w: _weekly(424243 + i, w, 2, 3.0, 31) for w in state.weeks},
            )
            for i in range(3)
        ]
        sim = _FakeSim(state=state, draw=draw, outlooks=outlooks, n_sims=400, seed=5)
        extended = T.sim_with_free_agents(sim, [wire, *scrubs])
        assert 424242 in extended.state.pool.player_ids
        assert extended.state.pool.size == state.pool.size + 4

        engine = T.TitleEngine(extended.state, extended.draw)
        claim = Move(
            kind=MoveKind.WAIVER_CLAIM,
            league_id=state.league_id,
            players=(
                PlayerMove(player_id=424242, from_team=None, to_team=1),
                PlayerMove(player_id=state.franchise(1).player_ids[-1], from_team=1, to_team=None),
            ),
        )
        rec = engine.confirm([claim])[0]
        assert rec.delta_points > 0.0
        assert "A Free Agent" in rec.rationale

    def test_nothing_new_returns_the_same_sim(self):
        state, draw, outlooks = build_league(
            n_teams=6,
            playoff_team_count=4,
            n_sims=200,
            reg_weeks=(1, 2),
            rounds=((3,), (4,)),
            with_outlooks=True,
        )
        sim = _FakeSim(state=state, draw=draw, outlooks=outlooks, n_sims=200, seed=5)
        assert T.sim_with_free_agents(sim, outlooks) is sim


@dataclass
class _FakeSim:
    """The five fields `sim_with_free_agents` reads off a `pipeline.LeagueSim`.

    Standing in for the real thing so the panel-extension path is covered offline; the
    live wiring is exercised by `TestAgainstTheRealLeagues`.
    """

    state: S.LeagueState
    draw: object
    outlooks: list
    n_sims: int
    seed: int
    league: object = None


def test_module_exports_are_importable():
    for name in T.__all__:
        assert hasattr(T, name), name


def test_sigmoid_is_stable_at_the_tails():
    assert T._sigmoid(np.array([-1e6, 0.0, 1e6])).tolist() == pytest.approx(
        [0.0, 0.5, 1.0], abs=1e-12
    )


def test_a_draw_from_another_league_is_refused_at_construction():
    """The engine assembles its own baseline rather than calling `team_week_scores`, so
    the panel/pool alignment check has to be made explicitly or a misaligned draw scores
    the right numbers against the wrong players in silence."""
    state, _ = build_league(
        n_teams=4, playoff_team_count=2, n_sims=60, reg_weeks=(1, 2), rounds=((3,),)
    )
    _, other_draw = build_league(
        n_teams=6, playoff_team_count=2, n_sims=60, reg_weeks=(1, 2), rounds=((3,),)
    )
    with pytest.raises(S.SeasonError):
        T.TitleEngine(state, other_draw)


def test_a_panel_and_a_pool_that_disagree_are_refused():
    """`panel_for` is the contract between the tensor and the pool; a mismatch scores the
    right numbers against the wrong players and nothing raises downstream."""
    state, _ = build_league(
        n_teams=4, playoff_team_count=2, n_sims=50, reg_weeks=(1, 2), rounds=((3,),)
    )
    with pytest.raises(S.SeasonError):
        S.panel_for(state, [])


def test_hurdle_gamma_helper_matches_the_stated_moments():
    """The synthetic league here has to be a fair test bed, not a caricature."""
    outlook = _weekly(1, 1, 3, 11.0, 5)
    assert outlook.position_id == 3
    panel = SimPanel.from_outlooks([outlook])
    # Injuries off: with them on the sampler is correctly reporting a 4.5% chance the
    # receiver never suits up, which is a different quantity from the stated marginal.
    draw = WeeklySampler(panel, seed=3, injuries=InjuryModel.off()).draw(20000)
    # The calibration deliberately shrinks an 11-point projection to a 10.4-point mean;
    # the sampler must reproduce the calibrated moments, not the raw projection.
    assert outlook.mean < 11.0
    assert float(draw.points.mean()) == pytest.approx(outlook.mean, rel=0.03)
    assert float(draw.points.std()) == pytest.approx(outlook.sd, rel=0.10)


def test_leverage_matches_the_measured_conversion():
    """1 projected point ~= 1.16pp of weekly win probability at an even matchup."""
    row = T.WeekLeverage(
        week=1,
        matchup_period=1,
        team_id=1,
        opponent_id=2,
        margin=0.0,
        sd_diff=34.4,
        win_probability=0.5,
        leverage=1.0,
    )
    assert row.points_per_win_pct == pytest.approx(1.16, abs=0.01)
    assert not row.decided


def test_leverage_falls_off_as_the_matchup_decides():
    """Built through `core.leverage`, which is what the engine actually calls.

    Constructing the row with a hand-computed `leverage=` and then asserting on the field
    tests `math.exp` and nothing else -- the dataclass hands the value straight back.
    """

    def at(z: float) -> T.WeekLeverage:
        margin, sd = z * 34.4, 34.4
        return T.WeekLeverage(
            week=1,
            matchup_period=1,
            team_id=1,
            opponent_id=2,
            margin=margin,
            sd_diff=sd,
            win_probability=0.5,
            leverage=core_leverage(margin, sd),
        )

    assert at(0.0).leverage == pytest.approx(1.0, abs=1e-12)
    assert at(1.5).leverage == pytest.approx(0.32, abs=0.01)
    assert at(2.0).leverage == pytest.approx(0.14, abs=0.01)
    assert at(2.0).decided and not at(0.0).decided
    # A point is worth less the more decided the week is, which is the whole claim.
    assert at(2.0).points_per_win_pct < at(1.5).points_per_win_pct < at(0.0).points_per_win_pct


class TestTheFloorIsTheRealWireNotTheRosterBottom:
    """The replacement floor must come from who is actually unrostered.

    Found by a user asking why a clearly better free agent was never recommended.
    `streaming_replacement` read the VOLS demand rank -- WR28 in a 12-team league --
    but in a league that carries five receivers WR28 is ROSTERED. Measured on the
    real league the floor claimed an empty WR slot streams 8.73 points a week while
    the best genuinely available receiver (WR55) projected 6.48.

    The consequence was not a small bias. Every bench receiver below the phantom
    floor contributed EXACTLY zero, so swapping one for another returned
    +0.000pp +/- 0.000 on bit-identical seasons -- the option value that justifies
    carrying a bench at all was silently zero.
    """

    WEEKS = (1, 2, 3)

    def _outlooks(self, rostered_means, free_means):
        """Returns (all outlooks, rostered ids, calibrated free means best-first).

        The calibrated mean is what the floor actually sees -- `_weekly` puts the
        projection through the level correction -- so the expectations are read off
        the fixture rather than restated, which also keeps this honest if the
        calibration is ever refit.
        """
        out, pid, rostered_ids, free_rates = [], 1, [], []
        for m in rostered_means:
            out.append(_outlook_at(pid, 3, m, self.WEEKS))
            rostered_ids.append(pid)
            pid += 1
        for m in free_means:
            o = _outlook_at(pid, 3, m, self.WEEKS)
            out.append(o)
            free_rates.append(sum(o.weeks[w].mean for w in self.WEEKS) / len(self.WEEKS))
            pid += 1
        return out, rostered_ids, sorted(free_rates, reverse=True)

    def test_the_floor_reads_the_wire_not_the_roster(self):
        """Twenty strong rostered receivers, three weak free ones."""
        outlooks, rostered, free = self._outlooks([12.0] * 20, [6.5, 6.0, 5.5])
        floors = wire.wire_floor(outlooks, rostered, self.WEEKS, {4: frozenset({3})}, depth=2)
        assert floors[4] == pytest.approx(free[1])  # 2nd best AVAILABLE
        assert floors[4] < free[0] * 1.6, "the floor must not be read off rostered players"

    def test_a_deeper_wire_depth_takes_a_worse_body(self):
        outlooks, rostered, free = self._outlooks([12.0] * 20, [6.5, 6.0, 5.5])
        el = {4: frozenset({3})}
        got = [wire.wire_floor(outlooks, rostered, self.WEEKS, el, depth=d)[4] for d in (1, 2, 3)]
        assert got == [pytest.approx(f) for f in free]
        assert got[0] > got[1] > got[2], "a deeper wire must be a worse body"

    def test_an_empty_wire_floors_at_zero_so_the_caller_can_fall_back(self):
        """`streaming_replacement` detects this and substitutes the VOLS rank; the
        primitive itself must report honestly that it found nobody."""
        outlooks, rostered, _ = self._outlooks([12.0] * 20, [])
        assert wire.wire_floor(outlooks, rostered, self.WEEKS, {4: frozenset({3})})[4] == 0.0

    def test_the_streamer_is_chosen_per_week_not_once_for_the_season(self):
        """Two free agents who alternate: each is best in different weeks, so the
        week-by-week floor is strictly above either one's season average."""
        weeks = (1, 2)
        a = PlayerOutlook(
            player_id=91,
            name="a",
            position_id=3,
            pro_team_id=1,
            weeks={1: _weekly(91, 1, 3, 10.0, 1), 2: _weekly(91, 2, 3, 2.0, 1)},
        )
        b = PlayerOutlook(
            player_id=92,
            name="b",
            position_id=3,
            pro_team_id=1,
            weeks={1: _weekly(92, 1, 3, 2.0, 1), 2: _weekly(92, 2, 3, 10.0, 1)},
        )
        floor = wire.wire_floor([a, b], [], weeks, {4: frozenset({3})}, depth=1)[4]
        season_avg = max(sum(o.weeks[w].mean for w in weeks) / len(weeks) for o in (a, b))
        assert floor > season_avg, "the streamer is picked per week, not once for the season"


def _outlook_at(pid: int, position_id: int, mean: float, weeks) -> PlayerOutlook:
    """A player projected at a flat `mean` every week."""
    return PlayerOutlook(
        player_id=pid,
        name=f"p{pid}",
        position_id=position_id,
        pro_team_id=1,
        weeks={w: _weekly(pid, w, position_id, mean, 1) for w in weeks},
    )
