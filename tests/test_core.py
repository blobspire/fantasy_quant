"""The shared contracts, and the win-probability math every surface depends on."""

from __future__ import annotations

import math

import pytest

from fantasy_quant.core import (
    LeagueContext,
    Move,
    MoveKind,
    Objective,
    PlayerMove,
    PlayerOutlook,
    Recommendation,
    WeeklyOutlook,
    leverage,
    points_to_win_prob,
    swap_improves_win_probability,
    win_probability,
)

# The measured team-score anchor: 12-team PPR, SD 24.35, so sd(you - opp) = sqrt(2)*24.35.
SD_DIFF = 34.4


def _outlook(week: int, mean: float, sd: float = 6.0, **kw) -> WeeklyOutlook:
    return WeeklyOutlook(
        player_id=1,
        season=2026,
        week=week,
        position_id=3,
        mean=mean,
        sd=sd,
        p_zero=0.25,
        shape=2.0,
        scale=mean / 2 or 1.0,
        **kw,
    )


class TestWinProbability:
    def test_one_point_is_about_1_16pp_at_an_even_matchup(self):
        """The conversion constant the whole product rests on."""
        assert points_to_win_prob(SD_DIFF) * 100 == pytest.approx(1.16, abs=0.01)

    def test_leverage_decays_the_way_it_was_measured(self):
        assert leverage(0, SD_DIFF) == pytest.approx(1.0)
        assert leverage(1.5 * SD_DIFF, SD_DIFF) == pytest.approx(0.32, abs=0.01)
        assert leverage(2.0 * SD_DIFF, SD_DIFF) == pytest.approx(0.14, abs=0.01)

    def test_leverage_is_symmetric_in_the_margin(self):
        assert leverage(20, SD_DIFF) == pytest.approx(leverage(-20, SD_DIFF))

    def test_win_probability_is_a_coin_flip_at_zero_margin(self):
        assert win_probability(0, SD_DIFF) == pytest.approx(0.5)

    def test_win_probability_is_monotone_in_margin(self):
        probs = [win_probability(m, SD_DIFF) for m in range(-40, 41, 10)]
        assert probs == sorted(probs)

    def test_zero_variance_degenerates_to_a_certainty(self):
        assert win_probability(5, 0) == 1.0
        assert win_probability(-5, 0) == 0.0
        assert win_probability(0, 0) == 0.5


class TestVarianceObjectiveFlips:
    """The headline strategic result: the same swap is right or wrong by standing."""

    def test_underdog_accepts_variance_at_a_cost_in_mean(self):
        assert swap_improves_win_probability(-1.0, 3.0, margin=-15, sd_diff=SD_DIFF)

    def test_favorite_rejects_the_identical_swap(self):
        assert not swap_improves_win_probability(-1.0, 3.0, margin=+15, sd_diff=SD_DIFF)

    def test_at_an_even_matchup_only_the_mean_matters(self):
        """z = 0, so the variance term drops out entirely."""
        assert swap_improves_win_probability(0.1, 99.0, margin=0, sd_diff=SD_DIFF)
        assert not swap_improves_win_probability(-0.1, 99.0, margin=0, sd_diff=SD_DIFF)

    def test_the_hurdle_rate_is_the_standardized_margin(self):
        """d_mean - z*d_sd > 0, so the break-even d_mean is exactly z*d_sd."""
        z, d_sd = 0.5, 4.0
        margin = z * SD_DIFF
        assert not swap_improves_win_probability(z * d_sd - 0.01, d_sd, margin, SD_DIFF)
        assert swap_improves_win_probability(z * d_sd + 0.01, d_sd, margin, SD_DIFF)


class TestWeeklyOutlook:
    def test_rejects_an_impossible_hurdle_probability(self):
        with pytest.raises(ValueError, match="p_zero"):
            WeeklyOutlook(
                player_id=1,
                season=2026,
                week=1,
                position_id=3,
                mean=10,
                sd=5,
                p_zero=1.5,
                shape=2,
                scale=5,
            )

    def test_rejects_negative_sd(self):
        with pytest.raises(ValueError, match="sd"):
            WeeklyOutlook(
                player_id=1,
                season=2026,
                week=1,
                position_id=3,
                mean=10,
                sd=-1,
                p_zero=0.2,
                shape=2,
                scale=5,
            )

    def test_zeroed_marks_a_player_as_not_playing(self):
        z = _outlook(5, 12.0).zeroed()
        assert (z.mean, z.sd, z.p_zero, z.playing) == (0.0, 0.0, 1.0, False)


class TestPlayerOutlook:
    def test_rest_of_season_sums_remaining_weeks_only(self):
        p = PlayerOutlook(
            player_id=1,
            name="x",
            position_id=3,
            pro_team_id=8,
            weeks={w: _outlook(w, 10.0) for w in range(1, 6)},
        )
        assert p.mean_from(1) == pytest.approx(50.0)
        assert p.mean_from(4) == pytest.approx(20.0)

    def test_playoff_mean_selects_only_playoff_weeks(self):
        p = PlayerOutlook(
            player_id=1,
            name="x",
            position_id=3,
            pro_team_id=8,
            weeks={w: _outlook(w, float(w)) for w in range(1, 18)},
        )
        assert p.playoff_mean([15, 16, 17]) == pytest.approx(48.0)


class TestLeagueContext:
    def _ctx(self, **kw):
        base = dict(
            league_id=1,
            season=2026,
            name="t",
            size=12,
            lineup_slot_counts={0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1, 20: 7, 21: 1},
            slot_eligibility={},
            scorer=lambda s, p: 0.0,
            playoff_team_count=6,
            playoff_weeks=(15, 16, 17),
            regular_season_weeks=tuple(range(1, 15)),
        )
        return LeagueContext(**{**base, **kw})

    def test_bench_and_ir_are_not_starting_slots(self):
        ctx = self._ctx()
        assert 20 not in ctx.starting_slots and 21 not in ctx.starting_slots
        assert ctx.starters_per_team == 9

    def test_zero_count_slots_are_excluded(self):
        ctx = self._ctx(lineup_slot_counts={0: 1, 2: 2, 7: 0, 20: 7})
        assert 7 not in ctx.starting_slots

    def test_objective_defaults_to_championship(self):
        assert self._ctx().objective is Objective.CHAMPIONSHIP


class TestMoveAndRecommendation:
    def test_move_reports_every_team_it_touches(self):
        m = Move(
            kind=MoveKind.TRADE,
            league_id=1,
            players=(
                PlayerMove(player_id=1, from_team=3, to_team=7),
                PlayerMove(player_id=2, from_team=7, to_team=9),
                PlayerMove(player_id=3, from_team=9, to_team=3),
            ),
        )
        assert m.teams == frozenset({3, 7, 9})

    def test_a_waiver_add_from_the_wire_has_no_from_team(self):
        m = Move(
            kind=MoveKind.WAIVER_CLAIM,
            league_id=1,
            players=(PlayerMove(player_id=1, from_team=None, to_team=4),),
        )
        assert m.teams == frozenset({4})

    def test_an_effect_smaller_than_its_own_error_is_not_significant(self):
        move = Move(kind=MoveKind.HOLD, league_id=1)
        assert not Recommendation(move, delta_title=0.001, delta_points=1, stderr=0.01).significant
        assert Recommendation(move, delta_title=0.05, delta_points=1, stderr=0.01).significant

    def test_recommendations_sort_by_title_impact(self):
        move = Move(kind=MoveKind.HOLD, league_id=1)
        recs = [Recommendation(move, delta_title=d, delta_points=0) for d in (0.01, 0.05, -0.02)]
        assert [r.delta_title for r in sorted(recs)] == [-0.02, 0.01, 0.05]


def test_points_to_win_prob_matches_a_numeric_derivative():
    """dP/dmu should equal the finite difference of win_probability."""
    h = 1e-4
    for margin in (-20.0, 0.0, 13.0):
        hi = win_probability(margin + h, SD_DIFF)
        lo = win_probability(margin - h, SD_DIFF)
        numeric = (hi - lo) / (2 * h)
        assert points_to_win_prob(SD_DIFF, margin) == pytest.approx(numeric, rel=1e-6)


def test_win_probability_matches_the_normal_cdf():
    assert win_probability(10, 20) == pytest.approx(0.5 * (1 + math.erf(0.5 / math.sqrt(2))))
