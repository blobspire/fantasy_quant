"""Start/sit tests.

The fixtures hand the surface a tensor it did not draw, which is the only way to test a
win-probability rule honestly: the interesting cases are "a steady ten points" against "a
coin flip between nothing and nineteen", and those are exact statements about a
distribution, not approximations of one. `_league` therefore builds the `(sim, week,
player)` points array directly and derives each player's stated mean and sd from it, so
the projections the surface ranks on and the football it is scored against cannot
disagree.

The correlation test is the exception and uses the real `WeeklySampler`, because the
claim under test -- that starting a receiver whose quarterback the opponent starts
shrinks `sd_diff` -- is a claim about `sim/distributions.py`'s measured blocks, and
faking the draw would test nothing.

Everything here is offline and runs in a few seconds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from fantasy_quant.core import (
    QB,
    RB,
    WR,
    MoveKind,
    WeeklyOutlook,
    leverage,
    swap_improves_win_probability,
)
from fantasy_quant.decide import lineups as L
from fantasy_quant.sim import season as S
from fantasy_quant.sim.distributions import Draw, SimPanel, WeeklySampler
from fantasy_quant.sim.lineup import plan_from_slots

# Slot ids, not position ids. See espn/constants.py for why that distinction matters.
SLOT_QB, SLOT_RB, SLOT_WR, SLOT_FLEX = 0, 2, 4, 23

SIMPLE_SLOTS: Mapping[int, int] = {SLOT_QB: 1, SLOT_RB: 1}
SIMPLE_ELIGIBILITY: Mapping[int, frozenset[int]] = {
    SLOT_QB: frozenset({QB}),
    SLOT_RB: frozenset({RB}),
}
FLEX_SLOTS: Mapping[int, int] = {SLOT_QB: 1, SLOT_RB: 2, SLOT_WR: 2, SLOT_FLEX: 1}
FLEX_ELIGIBILITY: Mapping[int, frozenset[int]] = {
    SLOT_QB: frozenset({QB}),
    SLOT_RB: frozenset({RB}),
    SLOT_WR: frozenset({WR}),
    SLOT_FLEX: frozenset({RB, WR}),
}


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Player:
    player_id: int
    position_id: int
    pro_team_id: int
    name: str
    #: (sims,) points this player scores every week of the fixture, or None for a bye.
    points: np.ndarray | None


def _league(
    rosters: Mapping[int, Sequence[Player]],
    *,
    weeks: Sequence[int],
    games: Sequence[tuple[int, int, int]],
    playoff_rounds: Sequence[Sequence[int]] = ((),),
    playoff_team_count: int = 2,
    slot_counts: Mapping[int, int] = SIMPLE_SLOTS,
    slot_eligibility: Mapping[int, frozenset[int]] = SIMPLE_ELIGIBILITY,
    my_team_id: int = 1,
    wins: Mapping[int, int] | None = None,
    points_for: Mapping[int, float] | None = None,
) -> tuple[S.LeagueState, Draw]:
    """A whole league with a hand-built tensor. `games` is `(week, home, away)`."""
    players = [p for roster in rosters.values() for p in roster]
    n_sims = next(p.points.shape[0] for p in players if p.points is not None)
    outlooks: list[WeeklyOutlook] = []
    for p in players:
        for w in weeks:
            if p.points is None:
                outlooks.append(
                    WeeklyOutlook(
                        player_id=p.player_id,
                        season=2026,
                        week=w,
                        position_id=p.position_id,
                        mean=0.0,
                        sd=0.0,
                        p_zero=1.0,
                        shape=1.0,
                        scale=1.0,
                        pro_team_id=p.pro_team_id,
                        playing=False,
                    )
                )
                continue
            outlooks.append(
                WeeklyOutlook(
                    player_id=p.player_id,
                    season=2026,
                    week=w,
                    position_id=p.position_id,
                    # Stated moments are read off the tensor, so the ranking the surface
                    # uses and the football it is scored against are the same numbers.
                    mean=float(p.points.mean()),
                    sd=float(p.points.std(ddof=0)),
                    p_zero=0.0,
                    shape=2.0,
                    scale=2.0,
                    pro_team_id=p.pro_team_id,
                    playing=True,
                )
            )
    panel = SimPanel.from_outlooks(outlooks, weeks=list(weeks))
    points = np.zeros((n_sims, panel.n_weeks, panel.n_players), dtype=np.float32)
    for p in players:
        if p.points is None:
            continue
        j = int(np.searchsorted(panel.player_ids, p.player_id))
        points[:, :, j] = p.points[:, None]
    draw = Draw(
        panel=panel,
        seed=0,
        n_sims=n_sims,
        points=points,
        available=np.broadcast_to(panel.has_game[None, :, :], points.shape).copy(),
    )
    pool = S.PlayerPool.of((p.player_id, p.position_id, p.pro_team_id, p.name) for p in players)
    franchises = tuple(
        S.Franchise(
            team_id=tid,
            name=f"team {tid}",
            player_ids=tuple(p.player_id for p in roster),
            wins=(wins or {}).get(tid, 0),
            points_for=(points_for or {}).get(tid, 0.0),
            is_user=tid == my_team_id,
        )
        for tid, roster in sorted(rosters.items())
    )
    state = S.LeagueState(
        league_id=99,
        season=2026,
        name="fixture",
        franchises=franchises,
        pool=pool,
        weeks=tuple(weeks),
        remaining_games=tuple(
            S.ScheduledGame(matchup_period=w, weeks=(w,), home_team_id=h, away_team_id=a)
            for w, h, a in games
        ),
        lineup_slot_counts=slot_counts,
        slot_eligibility=slot_eligibility,
        playoff_team_count=playoff_team_count,
        playoff_rounds=tuple(tuple(r) for r in playoff_rounds),
        my_team_id=my_team_id,
    )
    return state, draw


def _normal(n: int, mean: float, sd: float, seed: int) -> np.ndarray:
    """A fixed sample whose realised mean and sd are exactly what was asked for."""
    raw = np.random.default_rng(seed).standard_normal(n)
    raw = (raw - raw.mean()) / raw.std(ddof=0)
    return mean + sd * raw


def _coin(n: int, high: float, seed: int) -> np.ndarray:
    """Exactly half the simulations score `high`, the other half nothing."""
    out = np.zeros(n)
    out[np.random.default_rng(seed).permutation(n)[: n // 2]] = high
    return out


def _variance_fixture(
    *,
    opponent_total: float,
    steady_mean: float,
    volatile_high: float,
    alternative: np.ndarray | None = None,
    n_sims: int = 4000,
) -> tuple[S.LeagueState, Draw]:
    """Me with one steady back and one boom/bust back, against a fixed opponent score.

    The quarterback carries all of the lineup-independent randomness, so `sd_diff` is
    non-degenerate and the *only* thing a lineup choice changes is how much spread sits
    on top of it. The opponent is deterministic so the standardized margin is whatever
    the test says it is.

    Six regular-season weeks rather than one, deliberately: with a one-week season the
    run-in rule would catch week 1 and every one of these tests would be silently
    exercising the playoff-cut threshold instead of the matchup. That mistake was live in
    an earlier draft of this file, and it is invisible from the assertions.
    """
    me = [
        Player(101, QB, 1, "my qb", _normal(n_sims, 20.0, 10.0, 7)),
        Player(102, RB, 2, "steady", np.full(n_sims, steady_mean)),
        Player(
            103,
            RB,
            3,
            "volatile",
            _coin(n_sims, volatile_high, 11) if alternative is None else alternative,
        ),
    ]
    them = [
        Player(201, QB, 4, "their qb", np.full(n_sims, opponent_total)),
        Player(202, RB, 5, "their rb", np.zeros(n_sims)),
    ]
    filler = {
        3: [
            Player(301, QB, 6, "qb3", np.full(n_sims, 15.0)),
            Player(302, RB, 7, "rb3", np.full(n_sims, 10.0)),
        ],
        4: [
            Player(401, QB, 8, "qb4", np.full(n_sims, 15.0)),
            Player(402, RB, 9, "rb4", np.full(n_sims, 10.0)),
        ],
    }
    return _league(
        {1: me, 2: them, **filler},
        weeks=tuple(range(1, 8)),
        games=tuple((w, h, a) for w in range(1, 7) for h, a in ((1, 2), (3, 4))),
        playoff_rounds=((7,),),
        playoff_team_count=2,
    )


def _benched_opponent_fixture(n_sims: int = 2000) -> tuple[S.LeagueState, Draw]:
    """Like `_variance_fixture`, but the opponent has a bench they could set wrong.

    Needed because every other fixture gives the opponent exactly as many players as
    starting slots, which makes "the lineup they set" and "the lineup we assume they set"
    the same object and any test of `opponent_starters` vacuous.
    """
    me = [
        Player(101, QB, 1, "my qb", _normal(n_sims, 20.0, 10.0, 7)),
        Player(102, RB, 2, "my rb", np.full(n_sims, 10.0)),
    ]
    them = [
        Player(201, QB, 4, "their qb", np.full(n_sims, 20.0)),
        Player(202, RB, 5, "their good rb", np.full(n_sims, 12.0)),
        Player(203, RB, 6, "their bad rb", np.full(n_sims, 4.0)),
    ]
    filler = {
        3: [
            Player(301, QB, 7, "qb3", np.full(n_sims, 15.0)),
            Player(302, RB, 8, "rb3", np.full(n_sims, 10.0)),
        ],
        4: [
            Player(401, QB, 9, "qb4", np.full(n_sims, 15.0)),
            Player(402, RB, 10, "rb4", np.full(n_sims, 10.0)),
        ],
    }
    return _league(
        {1: me, 2: them, **filler},
        weeks=tuple(range(1, 8)),
        games=tuple((w, h, a) for w in range(1, 7) for h, a in ((1, 2), (3, 4))),
        playoff_rounds=((7,),),
        playoff_team_count=2,
    )


# --------------------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------------------


class TestSwapRule:
    @pytest.mark.parametrize("z", [-2.0, -0.8, -0.25, 0.0, 0.25, 0.8, 2.0])
    @pytest.mark.parametrize("d_sd", [0.5, 3.0, 12.0])
    def test_break_even_is_exactly_z_times_delta_sigma(self, z: float, d_sd: float) -> None:
        """The frontier is `d_mu = z * d_sd`, with strictly better on one side of it."""
        sd_diff = 34.4
        margin = z * sd_diff
        break_even = z * d_sd
        assert not swap_improves_win_probability(break_even, d_sd, margin, sd_diff)
        assert swap_improves_win_probability(break_even + 1e-9, d_sd, margin, sd_diff)
        assert not swap_improves_win_probability(break_even - 1e-9, d_sd, margin, sd_diff)

    def test_favorite_and_underdog_disagree_on_one_swap(self) -> None:
        """Identical swap, opposite verdicts. This is the whole module in one assertion."""
        d_mean, d_sd, sd_diff = -1.0, 6.0, 34.4
        favorite = swap_improves_win_probability(d_mean, d_sd, +20.0, sd_diff)
        underdog = swap_improves_win_probability(d_mean, d_sd, -20.0, sd_diff)
        assert not favorite
        assert underdog

    def test_leverage_falls_to_a_seventh_at_two_sigma(self) -> None:
        sd = 34.4
        assert leverage(0.0, sd) == pytest.approx(1.0)
        assert leverage(1.5 * sd, sd) == pytest.approx(0.325, abs=0.01)
        assert leverage(2.0 * sd, sd) == pytest.approx(0.135, abs=0.005)
        # "One seventh of the same call in a coin flip", stated as a ratio.
        assert 1.0 / leverage(2.0 * sd, sd) == pytest.approx(7.4, abs=0.2)


# --------------------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------------------


class TestEnumeration:
    def _flex_roster(self, n_sims: int = 200) -> tuple[S.LeagueState, Draw]:
        """A flex roster in an *ordinary* week.

        Six regular-season weeks, not one, and for the reason `_variance_fixture` gives:
        a one-week season puts week 1 inside `CUT_LINE_WINDOW`, and with two teams and a
        two-team bracket the cut line degenerates to "everyone is in and nobody gets a
        bye" -- every candidate lineup then scores `P = 0` against a `+inf` threshold,
        `wp == ep` for free, and every assertion about the win-probability lineup below
        passes without the win-probability path ever running. That was live here and it
        is invisible from the assertions.
        """
        rng = np.random.default_rng(3)
        me = [
            Player(101, QB, 1, "qb1", _normal(n_sims, 18.0, 6.0, 1)),
            Player(102, QB, 2, "qb2", _normal(n_sims, 14.0, 9.0, 2)),
            *[
                Player(110 + i, RB, 3 + i, f"rb{i}", _normal(n_sims, 14.0 - 2.0 * i, 6.0, 10 + i))
                for i in range(4)
            ],
            *[
                Player(120 + i, WR, 10 + i, f"wr{i}", _normal(n_sims, 13.0 - 1.5 * i, 7.0, 20 + i))
                for i in range(4)
            ],
        ]
        them = [
            Player(200 + i, p, 20 + i, f"opp{i}", _normal(n_sims, 12.0, 6.0, 40 + i))
            for i, p in enumerate((QB, RB, RB, WR, WR, RB))
        ]
        assert rng is not None
        return _league(
            {1: me, 2: them},
            weeks=tuple(range(1, 8)),
            games=tuple((w, 1, 2) for w in range(1, 7)),
            playoff_rounds=((7,),),
            playoff_team_count=2,
            slot_counts=FLEX_SLOTS,
            slot_eligibility=FLEX_ELIGIBILITY,
        )

    def test_the_fixture_really_is_an_ordinary_matchup_week(self) -> None:
        """Guard the guard: every test below is meaningless against a settled threshold."""
        state, draw = self._flex_roster()
        advice = L.advise(state, draw, week=1)
        assert advice.kind is L.ThresholdKind.OPPONENT
        assert advice.decisive_share == 1.0
        assert 0.0 < advice.points_lineup.win_prob < 1.0
        assert advice.sd_diff > 0.0

    def test_enumeration_contains_the_points_optimal_lineup(self) -> None:
        """The pruned pool must still hold the lineup the proven-exact solver returns."""
        state, draw = self._flex_roster()
        advice = L.advise(state, draw, week=1)
        positions = np.asarray(state.pool.positions_of(state.franchise(1).player_ids))
        plan = plan_from_slots(state.lineup_slot_counts, state.slot_eligibility, positions)
        means = np.asarray(draw.panel.mean[0, state.pool.columns(state.franchise(1).player_ids)])
        assert advice.points_lineup.mean == pytest.approx(float(plan.solve(means).total))

    def test_enumeration_is_a_few_hundred_lineups(self) -> None:
        state, draw = self._flex_roster()
        advice = L.advise(state, draw, week=1)
        assert 10 < advice.n_lineups < 500

    def test_a_player_on_bye_is_never_started(self) -> None:
        state, draw = self._flex_roster()
        # Sit the best receiver down with a bye and he must vanish from every candidate.
        players = [
            Player(
                p, int(state.pool.position_ids[i]), int(state.pool.pro_team_ids[i]), str(p), None
            )
            for i, p in enumerate(state.pool.player_ids)
            if p == 120
        ]
        assert players
        rosters = {
            f.team_id: [
                Player(
                    pid,
                    int(state.pool.position_ids[state.pool.index[pid]]),
                    int(state.pool.pro_team_ids[state.pool.index[pid]]),
                    str(pid),
                    None
                    if pid == 120
                    else np.asarray(draw.points[:, 0, state.pool.index[pid]], dtype=np.float64),
                )
                for pid in f.player_ids
            ]
            for f in state.franchises
        }
        benched_state, benched_draw = _league(
            rosters,
            weeks=tuple(range(1, 8)),
            games=tuple((w, 1, 2) for w in range(1, 7)),
            playoff_rounds=((7,),),
            playoff_team_count=2,
            slot_counts=FLEX_SLOTS,
            slot_eligibility=FLEX_ELIGIBILITY,
        )
        advice = L.advise(benched_state, benched_draw, week=1)
        # Both lineups, and the win-probability one only says something because the
        # threshold is live -- see `test_the_fixture_really_is_an_ordinary_matchup_week`.
        assert advice.kind is L.ThresholdKind.OPPONENT
        assert 120 not in advice.points_lineup.player_ids
        assert 120 not in advice.win_prob_lineup.player_ids

    def test_slot_map_is_keyed_by_player_not_slot(self) -> None:
        """Two RB slots would collide under the direction `core.Move` suggests."""
        state, draw = self._flex_roster()
        advice = L.advise(state, draw, week=1)
        lineup = advice.recommendation.move.lineup
        assert lineup is not None
        assert set(lineup) == set(advice.recommended.player_ids)
        assert set(lineup.values()) <= set(FLEX_SLOTS)
        assert sum(1 for s in lineup.values() if s == SLOT_RB) == 2


# --------------------------------------------------------------------------------------
# Favorite against underdog, on one roster
# --------------------------------------------------------------------------------------


class TestVarianceDirection:
    """The same boom/bust back, started by the underdog and benched by the favorite."""

    UNDERDOG, FAVORITE = 38.0, 22.0

    def test_underdog_starts_the_volatile_back(self) -> None:
        state, draw = _variance_fixture(
            opponent_total=self.UNDERDOG, steady_mean=10.0, volatile_high=19.0
        )
        advice = L.advise(state, draw, week=1)
        assert advice.z < -L.OVERRIDE_Z
        assert 102 in advice.points_lineup.player_ids  # steady is the points lineup
        assert 103 in advice.recommended.player_ids  # volatile is the recommendation
        assert advice.guard == ""
        assert advice.recommendation.move.kind is MoveKind.LINEUP
        assert advice.delta_win_prob > 0.0
        assert advice.recommendation.delta_points < 0.0  # paid points for the spread
        assert advice.win_prob_lineup.sd_diff > advice.points_lineup.sd_diff

    def test_favorite_keeps_the_steady_back(self) -> None:
        state, draw = _variance_fixture(
            opponent_total=self.FAVORITE, steady_mean=10.0, volatile_high=19.0
        )
        advice = L.advise(state, draw, week=1)
        assert advice.z > L.OVERRIDE_Z
        assert 102 in advice.recommended.player_ids
        assert 103 not in advice.recommended.player_ids
        assert advice.recommendation.move.kind is MoveKind.HOLD

    def test_favorite_pays_points_to_shed_variance(self) -> None:
        """The mirror image: the points lineup is the volatile one and the favorite sells it."""
        state, draw = _variance_fixture(
            opponent_total=self.FAVORITE, steady_mean=10.0, volatile_high=21.0
        )
        advice = L.advise(state, draw, week=1)
        assert advice.z > L.OVERRIDE_Z
        assert 103 in advice.points_lineup.player_ids  # volatile projects higher
        assert 102 in advice.recommended.player_ids  # and is benched anyway
        assert advice.guard == ""
        assert advice.win_prob_lineup.sd_diff < advice.points_lineup.sd_diff
        assert advice.recommendation.delta_points < 0.0

    def test_underdog_keeps_the_volatile_back_on_the_same_roster(self) -> None:
        state, draw = _variance_fixture(
            opponent_total=self.UNDERDOG, steady_mean=10.0, volatile_high=21.0
        )
        advice = L.advise(state, draw, week=1)
        assert 103 in advice.recommended.player_ids
        assert advice.recommendation.move.kind is MoveKind.HOLD

    def test_the_analytic_rule_agrees_with_the_empirical_argmax(self) -> None:
        for opponent in (self.FAVORITE, self.UNDERDOG):
            state, draw = _variance_fixture(
                opponent_total=opponent, steady_mean=10.0, volatile_high=21.0
            )
            advice = L.advise(state, draw, week=1)
            assert advice.rule_agrees


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


class TestGuards:
    def test_the_z_guard_binds(self) -> None:
        """A better lineup by win probability is refused inside |z| = 0.4."""
        state, draw = _variance_fixture(opponent_total=28.5, steady_mean=10.0, volatile_high=21.0)
        advice = L.advise(state, draw, week=1)
        assert abs(advice.z) < L.OVERRIDE_Z
        assert advice.differ
        assert advice.win_prob_lineup.win_prob > advice.points_lineup.win_prob
        assert advice.recommended.player_ids == advice.points_lineup.player_ids
        assert "guard" in advice.guard
        assert advice.recommendation.move.kind is MoveKind.HOLD

    def test_the_two_point_guard_binds(self) -> None:
        """A big enough favorite still refuses to pay more than two projected points."""
        state, draw = _variance_fixture(opponent_total=15.0, steady_mean=7.0, volatile_high=21.0)
        advice = L.advise(state, draw, week=1)
        assert advice.z > L.OVERRIDE_Z
        assert advice.differ
        assert advice.points_sacrifice > L.MAX_POINTS_SACRIFICE
        assert advice.recommended.player_ids == advice.points_lineup.player_ids
        assert "projected points" in advice.guard

    def test_the_noise_guard_binds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An override that cannot be measured is not taken, however good it looks."""
        monkeypatch.setattr(L, "MIN_EDGE_SIGMA", 1.0e6)
        state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1)
        assert advice.differ
        assert "noise floor" in advice.guard
        assert advice.recommended.player_ids == advice.points_lineup.player_ids

    def test_the_selection_penalty_grows_with_the_candidate_count(self) -> None:
        """Best-of-n is biased high by about `sqrt(2 ln n)`; two candidates barely at all."""
        assert L.selection_penalty(2) == pytest.approx(1.177, abs=0.001)
        assert L.selection_penalty(200) == pytest.approx(3.255, abs=0.001)
        assert L.selection_penalty(1) == L.selection_penalty(2)
        assert L.selection_penalty(2000) > L.selection_penalty(200)

    def test_the_noise_floor_is_the_penalty_times_the_paired_stderr(self) -> None:
        state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1)
        assert advice.noise_floor == pytest.approx(
            L.MIN_EDGE_SIGMA * L.selection_penalty(advice.n_lineups) * advice.delta_win_prob_stderr
        )
        # Two candidate lineups here, so the correction is small and the real edge
        # survives it. On a full roster it is a factor of three and almost nothing does.
        assert advice.n_lineups == 2
        assert advice.delta_win_prob > advice.noise_floor

    def test_leverage_is_core_leverage_times_the_decisive_share(self) -> None:
        state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1)
        assert advice.decisive_share == 1.0
        assert advice.leverage == pytest.approx(leverage(advice.margin, advice.sd_diff))
        assert advice.leverage < 1.0


# --------------------------------------------------------------------------------------
# The playoff cut line
# --------------------------------------------------------------------------------------


def _race_fixture(
    n_sims: int = 1500, wins: Mapping[int, int] | None = None
) -> tuple[S.LeagueState, Draw]:
    """Four teams, eight regular-season weeks left and a two-round bracket.

    Long enough that weeks 1-3 are ordinary and weeks 4-8 are the run-in, which is what
    `CUT_LINE_WINDOW` selects.
    """

    def roster(base: int, qb_mean: float, rb_mean: float, seed: int) -> list[Player]:
        return [
            Player(base + 1, QB, base, "qb", _normal(n_sims, qb_mean, 8.0, seed)),
            Player(base + 2, RB, base + 50, "steady", np.full(n_sims, rb_mean)),
            Player(
                base + 3, RB, base + 60, "volatile", _coin(n_sims, 2.0 * rb_mean - 1.0, seed + 1)
            ),
        ]

    rosters = {
        1: roster(100, 20.0, 10.0, 5),
        2: roster(200, 19.0, 10.0, 6),
        3: roster(300, 19.0, 10.0, 7),
        4: roster(400, 19.0, 10.0, 8),
    }
    weeks = tuple(range(1, 11))
    games = []
    pairings = [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]
    for w in range(1, 9):
        for home, away in pairings[(w - 1) % 3]:
            games.append((w, home, away))
    return _league(
        rosters,
        weeks=weeks,
        games=tuple(games),
        playoff_rounds=((9,), (10,)),
        playoff_team_count=3,
        wins=dict(wins or {}),
    )


class TestCutLine:
    def test_ordinary_weeks_price_against_the_opponent(self) -> None:
        state, draw = _race_fixture()
        advice = L.advise(state, draw, week=1)
        assert advice.kind is L.ThresholdKind.OPPONENT
        assert advice.opponent is not None
        assert advice.decisive_share == 1.0

    def test_the_run_in_prices_against_the_playoff_cut(self) -> None:
        state, draw = _race_fixture()
        advice = L.advise(state, draw, week=6)
        assert advice.kind is L.ThresholdKind.PLAYOFF_CUT
        assert advice.target == "a playoff berth"
        # The opponent is still known and still reported -- it is simply not the threshold.
        assert advice.opponent is not None
        assert advice.decisive_share < 1.0

    def test_the_cut_line_is_a_different_number_from_the_opponent(self) -> None:
        """Two thresholds for one week, and they must not agree by construction."""
        state, draw = _race_fixture()
        cut = L.advise(state, draw, week=6)
        opponent = L.advise(state, draw, week=6, cut_line_window=0)
        assert opponent.kind is L.ThresholdKind.OPPONENT
        assert cut.margin != pytest.approx(opponent.margin, abs=1.0)
        assert cut.sd_diff != pytest.approx(opponent.sd_diff, abs=1.0)

    def test_a_locked_in_team_prices_against_the_bye_line(self) -> None:
        state, draw = _race_fixture(wins={1: 8})
        advice = L.advise(state, draw, week=6)
        assert advice.kind is L.ThresholdKind.BYE_CUT
        assert advice.target == "a first-round bye"

    def test_a_bracket_week_prices_against_the_title(self) -> None:
        state, draw = _race_fixture()
        advice = L.advise(state, draw, week=9)
        assert advice.kind is L.ThresholdKind.TITLE
        assert advice.opponent is None
        assert advice.stacks == ()

    def test_a_settled_season_says_so_instead_of_recommending(self) -> None:
        """No bracket at all: nothing this week can change, and the surface says it."""
        state, draw = _race_fixture()
        no_bracket = S.LeagueState(
            **{
                **{f.name: getattr(state, f.name) for f in state.__dataclass_fields__.values()},
                "playoff_rounds": (),
                "playoff_team_count": 0,
                "weeks": state.weeks,
            }
        )
        advice = L.advise(no_bracket, draw, week=6)
        assert advice.decisive_share == 0.0
        assert advice.leverage == 0.0
        assert "settled" in advice.recommendation.rationale
        assert advice.recommendation.confidence == "low"

    def test_the_threshold_marks_settled_simulations_as_infinite(self) -> None:
        state, draw = _race_fixture(wins={1: 8})
        scores = S.team_week_scores(state, draw, efficiency=S.LineupEfficiency.symmetric())
        threshold = L.season_threshold(
            state,
            scores,
            week_index=state.week_index[6],
            team_index=state.team_index[1],
            field="made_playoffs",
            kind=L.ThresholdKind.PLAYOFF_CUT,
            label="a playoff berth",
        )
        assert np.isneginf(threshold.score[~threshold.decisive]).all()
        assert threshold.decisive_share < 0.05

    def test_the_threshold_reproduces_the_simulator_simulation_by_simulation(self) -> None:
        """The load-bearing claim: `score > threshold` *is* the simulator's own answer.

        The whole cut-line design rests on a team's own week score being monotone in
        whether it makes the bracket, so that one critical score per simulation replaces
        re-running the season for every candidate lineup. Check it the only way that
        settles it: shift the week score around and confirm the threshold predicts the
        re-simulated outcome in every single simulation, not merely on average.
        """
        state, draw = _race_fixture()
        wi, t = state.week_index[6], state.team_index[1]
        scores = S.team_week_scores(state, draw, efficiency=S.LineupEfficiency.symmetric())
        threshold = L.season_threshold(
            state,
            scores,
            week_index=wi,
            team_index=t,
            field="made_playoffs",
            kind=L.ThresholdKind.PLAYOFF_CUT,
            label="a playoff berth",
        )
        for offset in (-25.0, -5.0, 0.0, 5.0, 25.0):
            trial = scores.copy()
            trial[:, wi, t] = scores[:, wi, t] + offset
            truth = S.simulate_from_scores(state, trial, all_play=False).made_playoffs[:, t]
            predicted = trial[:, wi, t] > threshold.score
            # A handful of simulations sit within the bisection's own resolution of the
            # line; anything more would mean the monotonicity assumption is wrong.
            assert np.count_nonzero(truth != predicted) <= 2


# --------------------------------------------------------------------------------------
# Opponent correlation
# --------------------------------------------------------------------------------------


def _stack_league(wr_pro_team: int, *, n_sims: int = 8000) -> tuple[S.LeagueState, Draw]:
    """Me (QB + WR) against them (QB + WR), drawn by the real correlated sampler.

    My receiver's NFL team is the only thing that changes between the two arms of the
    test, so any move in `sd_diff` is the measured QB/WR block and nothing else.

    Six regular-season weeks for the same reason `_variance_fixture` has them: with one
    week the run-in rule catches week 1 and `sd_diff` is measured against the playoff
    cut line rather than against the opponent, which is not the claim under test. The
    assertions still passed that way -- the correlation leaks into the cut line through
    the opponent's own week score -- which is exactly why it has to be pinned down.
    """
    people = [
        (101, QB, 11, "my qb", 1),
        (102, WR, wr_pro_team, "my wr", 1),
        (201, QB, 22, "their qb", 2),
        (202, WR, 33, "their wr", 2),
        (301, QB, 44, "qb3", 3),
        (302, WR, 55, "wr3", 3),
        (401, QB, 66, "qb4", 4),
        (402, WR, 77, "wr4", 4),
    ]
    weeks = tuple(range(1, 8))
    outlooks = [
        WeeklyOutlook(
            player_id=pid,
            season=2026,
            week=w,
            position_id=pos,
            # Gamma(shape 4, scale 4.5): mean 18, sd 9, no zero mass.
            mean=18.0,
            sd=9.0,
            p_zero=0.0,
            shape=4.0,
            scale=4.5,
            pro_team_id=team,
            playing=True,
        )
        for pid, pos, team, _, _ in people
        for w in weeks
    ]
    panel = SimPanel.from_outlooks(outlooks, weeks=list(weeks))
    draw = WeeklySampler(panel, seed=4242, injuries=None).draw(n_sims)
    pool = S.PlayerPool.of((pid, pos, team, name) for pid, pos, team, name, _ in people)
    rosters: dict[int, list[int]] = {}
    for pid, _, _, _, tid in people:
        rosters.setdefault(tid, []).append(pid)
    state = S.LeagueState(
        league_id=7,
        season=2026,
        name="stack",
        franchises=tuple(
            S.Franchise(team_id=t, name=f"team {t}", player_ids=tuple(ps), is_user=t == 1)
            for t, ps in sorted(rosters.items())
        ),
        pool=pool,
        weeks=weeks,
        remaining_games=tuple(
            S.ScheduledGame(matchup_period=w, weeks=(w,), home_team_id=h, away_team_id=a)
            for w in range(1, 7)
            for h, a in ((1, 2), (3, 4))
        ),
        lineup_slot_counts={SLOT_QB: 1, SLOT_WR: 1},
        slot_eligibility={SLOT_QB: frozenset({QB}), SLOT_WR: frozenset({WR})},
        playoff_team_count=2,
        playoff_rounds=((7,),),
        my_team_id=1,
    )
    return state, draw


class TestOpponentCorrelation:
    def test_stacking_the_opponents_quarterback_shrinks_the_spread(self) -> None:
        """rho(QB, WR) = 0.30 on one NFL team, and it lands in `sd_diff` where it should."""
        neutral_state, neutral_draw = _stack_league(wr_pro_team=99)
        stacked_state, stacked_draw = _stack_league(wr_pro_team=22)
        neutral = L.advise(neutral_state, neutral_draw, week=1)
        stacked = L.advise(stacked_state, stacked_draw, week=1)
        # Independent halves are unchanged; only the covariance term moves.
        assert stacked.sd_independent == pytest.approx(neutral.sd_independent, abs=0.6)
        assert stacked.sd_diff < neutral.sd_diff - 0.8
        assert stacked.points_lineup.covariance > neutral.points_lineup.covariance + 10.0

    def test_the_stack_is_reported_with_its_measured_rho(self) -> None:
        state, draw = _stack_league(wr_pro_team=22)
        advice = L.advise(state, draw, week=1)
        assert [(e.player_id, e.opponent_player_id, e.rho) for e in advice.stacks] == [
            (102, 201, 0.30)
        ]
        assert advice.stacks[0].shrinks_spread
        assert advice.stacks[0].modelled

    def test_no_stack_is_reported_when_nobody_shares_a_team(self) -> None:
        state, draw = _stack_league(wr_pro_team=99)
        assert L.advise(state, draw, week=1).stacks == ()


# --------------------------------------------------------------------------------------
# What the recommendation says
# --------------------------------------------------------------------------------------


class TestRecommendation:
    def test_a_lineup_that_is_already_set_is_a_hold_worth_nothing(self) -> None:
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1, current=(101, 102))
        assert advice.recommendation.move.kind is MoveKind.HOLD
        assert advice.recommendation.delta_title == 0.0
        assert advice.recommendation.delta_points == 0.0
        # `core.Recommendation.significant` calls a zero effect significant; we do not.
        assert advice.recommendation.significant
        assert not advice.significant
        assert "not_significant" in advice.recommendation.tags

    def test_a_lineup_is_not_churned_for_a_hundredth_of_a_point(self) -> None:
        """Found live: two receivers projected 0.01 apart, and the advice cost 0.25pp.

        `argmax` is happy to break a tie, and every guard above this one was satisfied
        because nothing about the swap was *wrong* -- it was merely worth nothing, and
        the paired season simulation scored it slightly negative. Ties go to the lineup
        already in.
        """
        state, draw = _variance_fixture(
            opponent_total=30.0,
            steady_mean=10.0,
            volatile_high=0.0,
            alternative=np.full(4000, 10.05),
        )
        assert L.advise(state, draw, week=1).points_lineup.player_ids == (101, 103)
        advice = L.advise(state, draw, week=1, current=(101, 102))
        assert advice.recommendation.move.kind is MoveKind.HOLD
        assert advice.recommended.player_ids == (101, 102)
        assert "worth touching" in advice.guard
        assert advice.recommendation.delta_title == 0.0

    def test_a_variance_override_is_not_blocked_by_the_churn_floor(self) -> None:
        """The churn floor is denominated in threshold probability and must not gate this.

        Half a point of *weekly win* probability is nothing; half a point of
        *championship* probability is a large recommendation. One materiality number
        cannot police both, so a deliberate override -- which has already cleared the
        |z|, two-point and noise guards written for it -- is exempt.
        """
        state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1, current=(101, 102))
        assert advice.baseline.player_ids == advice.points_lineup.player_ids
        assert advice.guard == ""
        assert 103 in advice.recommended.player_ids
        assert advice.recommendation.move.kind is MoveKind.LINEUP

    def test_a_lineup_that_is_set_wrong_is_priced_against_what_is_set(self) -> None:
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1, current=(101, 103))
        assert advice.baseline.player_ids == (101, 103)
        assert advice.recommendation.move.kind is MoveKind.LINEUP
        assert advice.recommendation.delta_points > 0.0
        assert [s.in_player_id for s in advice.swaps] == [102]
        assert [s.out_player_id for s in advice.swaps] == [103]

    def test_an_illegal_current_lineup_falls_back_rather_than_raising(self) -> None:
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1, current=(102, 103))  # two RBs, no QB
        assert advice.baseline.player_ids == advice.points_lineup.player_ids

    def test_an_unknown_player_is_an_error(self) -> None:
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        with pytest.raises(L.StartSitError):
            L.advise(state, draw, week=1, current=(101, 999))

    def test_a_week_off_the_axis_is_an_error(self) -> None:
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        with pytest.raises(L.StartSitError):
            L.advise(state, draw, week=17)

    def test_the_opponents_actual_lineup_can_be_supplied(self) -> None:
        """A *suboptimal* opponent lineup, because the optimal one proves nothing.

        The obvious version of this test hands back the only legal lineup the opponent
        has and checks the margin did not move -- which passes just as happily if
        `opponent_starters` is thrown away unread. The opponent here has a bench, so the
        supplied lineup is eight points worse than the one they would be assumed to set,
        and the margin has to move by exactly that.
        """
        state, draw = _benched_opponent_fixture()
        assumed = L.advise(state, draw, week=1)
        told = L.advise(state, draw, week=1, opponent_starters=(201, 203))
        assert assumed.opponent is not None and told.opponent is not None
        assert set(assumed.opponent.starters) == {201, 202}  # their projection-optimal one
        assert set(told.opponent.starters) == {201, 203}
        assert told.opponent.mean == pytest.approx(assumed.opponent.mean - 8.0)
        assert told.margin == pytest.approx(assumed.margin + 8.0, abs=1e-6)

    def test_an_opponent_starter_who_is_not_on_their_roster_is_an_error(self) -> None:
        state, draw = _benched_opponent_fixture()
        with pytest.raises(L.StartSitError):
            L.advise(state, draw, week=1, opponent_starters=(201, 102))

    def test_the_delta_title_is_a_paired_difference_with_an_honest_stderr(self) -> None:
        """A real weekly edge, and a title effect that is honestly reported as noise.

        This is the shape of nearly every start/sit answer and it should not be dressed
        up: the win-probability gain is measurable, and what it does to a championship
        three rounds away is not.
        """
        state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1)
        rec = advice.recommendation
        assert advice.guard == ""
        assert advice.delta_win_prob > advice.noise_floor > 0.0
        assert abs(rec.delta_title) <= max(2.0 * rec.stderr, 0.005)
        assert not advice.significant
        assert "not_significant" in rec.tags
        assert rec.leverage == advice.leverage
        assert "start_sit" in rec.tags

    def test_the_report_names_the_threshold_and_the_leverage(self) -> None:
        state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
        text = L.advise(state, draw, week=1).report()
        assert "threshold" in text
        assert "leverage" in text
        assert "dTitle" in text


def test_module_constants_match_the_research_note() -> None:
    """The guards are the ones RESEARCH.md names, not something drifted."""
    assert L.OVERRIDE_Z == 0.4
    assert L.MAX_POINTS_SACRIFICE == 2.0
    assert L.CUT_LINE_WINDOW == 5
    assert math.isclose(L.MAX_SWING, 150.0)


# --------------------------------------------------------------------------------------
# Bounding the enumeration against brute force
# --------------------------------------------------------------------------------------


def _deep_roster(n_sims: int = 400, opponent_mean: float = 13.0) -> tuple[S.LeagueState, Draw]:
    """Fourteen players for six slots, with a mean-for-spread trade-off down the roster.

    `_flex_roster` has exactly four backs and four receivers against a pool depth of
    `count + 2`, so nothing is ever pruned there and "the enumeration contains the
    argmax" is true for free. Six of each puts the exhaustive set at 1,200 started sets,
    which the tests below prune to 88 and then check against.

    Each back and receiver trades projected mean for spread as you go down the list, so
    `opponent_mean` selects which end of the roster the win-probability argmax wants: a
    favorite takes the steady top of it, a big underdog buys the volatile bottom.
    """
    me = [
        Player(101, QB, 1, "qb1", _normal(n_sims, 18.0, 6.0, 1)),
        Player(102, QB, 2, "qb2", _normal(n_sims, 15.0, 11.0, 2)),
        *[
            Player(
                110 + i, RB, 3 + i, f"rb{i}", _normal(n_sims, 14.0 - 0.7 * i, 4.0 + 5.0 * i, 10 + i)
            )
            for i in range(6)
        ],
        *[
            Player(
                120 + i,
                WR,
                20 + i,
                f"wr{i}",
                _normal(n_sims, 13.0 - 0.6 * i, 5.0 + 4.5 * i, 30 + i),
            )
            for i in range(6)
        ],
    ]
    them = [
        Player(200 + i, p, 40 + i, f"opp{i}", _normal(n_sims, opponent_mean, 6.0, 50 + i))
        for i, p in enumerate((QB, RB, RB, WR, WR, RB))
    ]
    return _league(
        {1: me, 2: them},
        weeks=tuple(range(1, 8)),
        games=tuple((w, 1, 2) for w in range(1, 7)),
        playoff_rounds=((7,),),
        playoff_team_count=2,
        slot_counts=FLEX_SLOTS,
        slot_eligibility=FLEX_ELIGIBILITY,
    )


def _every_legal_started_set(
    positions: Mapping[int, int],
    counts: Mapping[int, int],
    eligibility: Mapping[int, frozenset[int]],
) -> list[frozenset[int]]:
    """Exhaustive: every distinct SET of players a legal lineup can start.

    Written from the slot table rather than from `lineups._slot_groups`, so it is an
    independent enumeration and not the code under test rephrased.
    """
    slots = [s for s, n in counts.items() for _ in range(n)]
    out: set[frozenset[int]] = set()

    def walk(i: int, used: frozenset[int]) -> None:
        if i == len(slots):
            out.add(used)
            return
        allowed = eligibility[slots[i]]
        placed = False
        for pid, pos in positions.items():
            if pid not in used and pos in allowed:
                placed = True
                walk(i + 1, used | {pid})
        if not placed:
            walk(i + 1, used)

    walk(0, frozenset())
    return sorted(out, key=sorted)


class TestEnumerationIsBounded:
    """The one claim the module cannot make by inspection: that the pruned candidate set
    still contains the true argmax. The module's own caveats admit a pathological roster
    could prune it away, and nothing checked it. Enumerate every legal started set
    independently of `lineups._slot_groups`, score it against the same draw and the same
    opponent, and require the shipped answer to be the exhaustive one.
    """

    @pytest.mark.parametrize("opponent_mean", [10.0, 11.5, 13.0, 14.5, 16.0, 19.0, 22.0])
    def test_the_pruned_pool_finds_the_exhaustive_win_probability_argmax(
        self, opponent_mean: float
    ) -> None:
        state, draw = _deep_roster(opponent_mean=opponent_mean)
        advice = L.advise(state, draw, week=1)
        assert advice.kind is L.ThresholdKind.OPPONENT
        assert advice.opponent is not None

        best_p, best_mean, best_set = _exhaustive_best(state, draw, advice.opponent.starters)
        assert advice.n_lineups < 1200, "the enumeration is meant to be a pruned set"
        assert advice.win_prob_lineup.win_prob == pytest.approx(best_p)
        assert advice.win_prob_lineup.mean == pytest.approx(best_mean, abs=1e-6)
        assert set(advice.win_prob_lineup.player_ids) == set(best_set)

    def test_the_objective_actually_moves_across_that_sweep(self) -> None:
        """Otherwise the parametrisation above proves one lineup seven times.

        A favourite and a big underdog must not be told to start the same nine players,
        or the win-probability path is not being exercised at all.
        """
        favourite = L.advise(*_deep_roster(opponent_mean=10.0), week=1)
        underdog = L.advise(*_deep_roster(opponent_mean=22.0), week=1)
        assert not favourite.differ  # the steady top of the roster is also the best bet
        assert underdog.differ  # and the underdog wants the volatile bottom of it
        assert set(favourite.win_prob_lineup.player_ids) != set(underdog.win_prob_lineup.player_ids)
        assert underdog.win_prob_lineup.sd_diff > underdog.points_lineup.sd_diff

    def test_a_shallower_pool_can_miss_it_which_is_why_the_depth_is_what_it_is(self) -> None:
        """`DEFAULT_EXTRA_DEPTH` is load-bearing, not decoration.

        At `extra_depth=0` the pool is exactly the slot count per ranking key, and on this
        roster that prunes the true argmax away in an even matchup. The default finds it.
        Pinning the failure is what makes the passing case above mean something.
        """
        state, draw = _deep_roster(opponent_mean=13.0)
        default = L.advise(state, draw, week=1)
        shallow = L.advise(state, draw, week=1, extra_depth=0)
        assert state.franchise(1).player_ids  # sanity: the roster is the deep one
        assert shallow.n_lineups < default.n_lineups
        assert shallow.win_prob_lineup.win_prob < default.win_prob_lineup.win_prob
        assert L.DEFAULT_EXTRA_DEPTH >= 1


def _exhaustive_best(
    state: S.LeagueState, draw: Draw, opponent: Sequence[int], week: int = 1
) -> tuple[float, float, frozenset[int]]:
    """`(P, projected mean, players)` of the best legal lineup, over *every* legal lineup."""
    mine = state.franchise(1).player_ids
    positions = {int(p): int(state.pool.position_ids[state.pool.index[p]]) for p in mine}
    wi = state.week_index[week]
    threshold = np.asarray(
        draw.points[:, wi][:, state.pool.columns(opponent)], dtype=np.float64
    ).sum(axis=1)
    means = dict(zip(mine, np.asarray(draw.panel.mean[wi, state.pool.columns(mine)]), strict=True))
    best: tuple[float, float, frozenset[int]] = (-1.0, -1.0, frozenset())
    for started in _every_legal_started_set(positions, FLEX_SLOTS, FLEX_ELIGIBILITY):
        total = np.asarray(
            draw.points[:, wi][:, state.pool.columns(sorted(started))], dtype=np.float64
        ).sum(axis=1)
        candidate = (float((total > threshold).mean()), float(sum(means[p] for p in started)))
        if candidate > best[:2]:
            best = (*candidate, started)
    return best


# --------------------------------------------------------------------------------------
# What the manager actually has in his lineup
# --------------------------------------------------------------------------------------


class TestCurrentLineup:
    def test_a_current_lineup_the_enumeration_pruned_away_is_still_priced(self) -> None:
        """The append path: a legal lineup outside the candidate set becomes a candidate.

        Without it the baseline silently falls back to the projected-best lineup and the
        recommendation is measured against a lineup the manager never set -- the same
        failure as an unassignable lineup, but quieter, because nothing about it is
        illegal.
        """
        state, draw = _deep_roster()
        mine = state.franchise(1).player_ids
        positions = {int(p): int(state.pool.position_ids[state.pool.index[p]]) for p in mine}
        wi = state.week_index[1]
        means = dict(
            zip(mine, np.asarray(draw.panel.mean[wi, state.pool.columns(mine)]), strict=True)
        )
        sets = _every_legal_started_set(positions, FLEX_SLOTS, FLEX_ELIGIBILITY)
        # The worst legal lineup on the roster is certainly not in a pool ranked on mean,
        # mean+sd and mean-sd.
        worst = min(sets, key=lambda s: sum(means[p] for p in s))
        without = L.advise(state, draw, week=1, extra_depth=0)
        advice = L.advise(state, draw, week=1, extra_depth=0, current=sorted(worst))
        assert advice.n_lineups == without.n_lineups + 1, "the lineup was not appended"
        assert set(advice.baseline.player_ids) == set(worst)
        assert advice.unpriced_current == ()
        assert advice.recommendation.move.kind is MoveKind.LINEUP
        assert advice.recommendation.delta_points > 0.0
        assert advice.baseline.mean == pytest.approx(sum(means[p] for p in worst), abs=1e-6)

    def test_an_unassignable_current_lineup_is_reported_not_silently_dropped(self) -> None:
        """The failure that matters most: a broken lineup told "nothing to change".

        `_with_current` cannot price a set of starters with no legal slot assignment, so
        it falls back to the projected-best lineup. Falling back is right; falling back
        *silently* is not -- the manager then reads a HOLD on a lineup he cannot field.
        """
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1, current=(102, 103))  # two RBs, no QB
        assert advice.unpriced_current == (102, 103)
        assert "unpriced_current" in advice.recommendation.tags
        assert advice.recommendation.move.kind is MoveKind.LINEUP
        assert "not priced" in advice.recommendation.rationale
        assert "fix it" in advice.recommendation.rationale
        assert "nothing to change" not in advice.recommendation.rationale
        assert "WARNING" in advice.report()
        # It still prices against the projected-best lineup, as documented.
        assert set(advice.baseline.player_ids) == set(advice.points_lineup.player_ids)

    def test_a_legal_current_lineup_is_never_flagged_as_unpriced(self) -> None:
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        for current in ((101, 102), (101, 103), None):
            advice = L.advise(state, draw, week=1, current=current)
            assert advice.unpriced_current == ()
            assert "unpriced_current" not in advice.recommendation.tags


def _twin_fixture(n_sims: int = 4000) -> tuple[S.LeagueState, Draw]:
    """Two boom/bust backs with *identical* realisations, and one steady one.

    Player 104 is player 103's outcome vector, simulation for simulation, so swapping one
    for the other changes nothing that can be measured -- the paired difference is the
    zero vector. Any surface that recommends the swap is recommending noise.
    """
    volatile = _coin(n_sims, 19.0, 11)
    me = [
        Player(101, QB, 1, "my qb", _normal(n_sims, 20.0, 10.0, 7)),
        Player(102, RB, 2, "steady", np.full(n_sims, 10.0)),
        Player(103, RB, 3, "volatile a", volatile),
        Player(104, RB, 4, "volatile b", volatile.copy()),
    ]
    them = [
        Player(201, QB, 5, "their qb", np.full(n_sims, 38.0)),
        Player(202, RB, 6, "their rb", np.zeros(n_sims)),
    ]
    filler = {
        3: [
            Player(301, QB, 7, "qb3", np.full(n_sims, 15.0)),
            Player(302, RB, 8, "rb3", np.full(n_sims, 10.0)),
        ],
        4: [
            Player(401, QB, 9, "qb4", np.full(n_sims, 15.0)),
            Player(402, RB, 10, "rb4", np.full(n_sims, 10.0)),
        ],
    }
    return _league(
        {1: me, 2: them, **filler},
        weeks=tuple(range(1, 8)),
        games=tuple((w, h, a) for w in range(1, 7) for h, a in ((1, 2), (3, 4))),
        playoff_rounds=((7,),),
        playoff_team_count=2,
    )


class TestOverrideAgainstWhatIsSet:
    def test_an_override_must_beat_the_lineup_already_set_not_just_the_points_lineup(
        self,
    ) -> None:
        """The |z|, two-point and noise guards are all measured against the POINTS
        lineup, because that is the reference the variance argument is made in. What the
        manager is actually asked to do is move off the lineup he has set, and when that
        lineup already carries the spread the override buys, the change is worth exactly
        nothing. Here the two candidates are the same outcome vector, so the paired gain
        is the zero vector and the answer has to be "leave it alone".
        """
        state, draw = _twin_fixture()
        free = L.advise(state, draw, week=1)
        assert free.differ and free.guard == ""  # the override itself is sound
        assert free.z < -L.OVERRIDE_Z
        twin = next(p for p in (103, 104) if p not in free.win_prob_lineup.player_ids)

        advice = L.advise(state, draw, week=1, current=(101, twin))
        assert set(advice.baseline.player_ids) == {101, twin}
        assert advice.recommended.player_ids == advice.baseline.player_ids
        assert advice.recommendation.move.kind is MoveKind.HOLD
        assert advice.recommendation.delta_title == 0.0
        assert "already set" in advice.guard

    def test_the_override_still_fires_against_the_points_lineup_it_beats(self) -> None:
        """The mirror: the new check must not swallow a real override."""
        state, draw = _twin_fixture()
        advice = L.advise(state, draw, week=1, current=(101, 102))
        assert set(advice.baseline.player_ids) == {101, 102}
        assert advice.guard == ""
        assert advice.recommendation.move.kind is MoveKind.LINEUP
        assert set(advice.recommended.player_ids) & {103, 104}


# --------------------------------------------------------------------------------------
# Common random numbers: the stderr has to be the paired one
# --------------------------------------------------------------------------------------


class TestPairing:
    def test_the_title_stderr_is_the_paired_difference_not_an_arm(self) -> None:
        """An unpaired stderr would be several times larger and would hide a real effect.

        `Recommendation.significant` is `|delta| > 2 * stderr`, so reporting a per-arm
        error rather than the error on the *difference* silently reclassifies findings as
        noise. Rebuild both arms here, measure both quantities, and require the reported
        number to be the paired one by a margin no rounding can explain.
        """
        state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1)
        assert advice.guard == ""
        assert set(advice.recommended.player_ids) != set(advice.baseline.player_ids)

        wi, t = state.week_index[1], state.team_index[1]
        base_scores = S.team_week_scores(state, draw, efficiency=S.LineupEfficiency.symmetric())
        arms = []
        for ids in (advice.baseline.player_ids, advice.recommended.player_ids):
            scores = base_scores.copy()
            scores[:, wi, t] = np.asarray(
                draw.points[:, wi][:, state.pool.columns(ids)], dtype=np.float64
            ).sum(axis=1)
            arms.append(
                S.simulate_from_scores(state, scores, all_play=False)
                .champions[:, t]
                .astype(np.float64)
            )
        a, b = arms
        n = a.size
        paired = float((b - a).std(ddof=1) / math.sqrt(n))
        unpaired = math.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)

        assert advice.recommendation.delta_title == pytest.approx(float((b - a).mean()))
        assert advice.recommendation.stderr == pytest.approx(paired)
        assert paired < 0.5 * unpaired, "common random numbers bought nothing; check the pairing"

    def test_a_no_op_recommendation_has_exactly_zero_error(self) -> None:
        """Bit-identical arms, so both the difference and its error must be exactly 0.0.

        This is what an unpaired implementation cannot do: two independent runs of the
        same lineup disagree by ~0.5pp at these sample sizes.
        """
        state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
        advice = L.advise(state, draw, week=1, current=(101, 102))
        assert advice.recommendation.move.kind is MoveKind.HOLD
        assert advice.recommendation.delta_title == 0.0
        assert advice.recommendation.stderr == 0.0
        assert not advice.significant


# --------------------------------------------------------------------------------------
# Multi-week matchup periods, and the small public surface
# --------------------------------------------------------------------------------------


def test_a_multi_week_matchup_folds_the_other_week_into_the_threshold() -> None:
    """A two-week matchup period is not two matchups: only one week is a lineup decision.

    Whatever both teams are projected to add in the other week shifts the score this
    week's lineup has to clear, one for one. Untested code before this, and it runs in
    every league whose `playoffMatchupPeriodLength` is two.
    """
    import dataclasses

    state, draw = _variance_fixture(opponent_total=22.0, steady_mean=10.0, volatile_high=19.0)
    single = L.advise(state, draw, week=1)
    paired_games = tuple(
        dataclasses.replace(g, weeks=(1, 2)) if g.weeks == (1,) else g
        for g in state.remaining_games
        if g.weeks != (2,)
    )
    twin = dataclasses.replace(state, remaining_games=paired_games)
    both = L.advise(twin, draw, week=1)

    scores = S.team_week_scores(twin, draw, efficiency=S.LineupEfficiency.symmetric())
    wj = twin.week_index[2]
    shift = float((scores[:, wj, twin.team_index[2]] - scores[:, wj, twin.team_index[1]]).mean())
    assert both.kind is L.ThresholdKind.OPPONENT
    assert both.margin == pytest.approx(single.margin - shift, abs=0.05)


def test_starters_from_roster_reads_the_starting_slots_only() -> None:
    from fantasy_quant.espn.league import RosterEntry, TeamRoster

    def entry(pid: int, slot: int) -> RosterEntry:
        return RosterEntry(
            player_id=pid,
            name=str(pid),
            lineup_slot_id=slot,
            default_position_id=RB,
            pro_team_id=1,
            eligible_slots=(slot,),
            injury_status="ACTIVE",
            injured=False,
            status="ONTEAM",
            acquisition_type="DRAFT",
            acquisition_date=None,
            keeper_value=0.0,
            keeper_value_future=0.0,
            percent_owned=0.0,
            percent_started=0.0,
        )

    roster = TeamRoster(
        team_id=1,
        scoring_period=1,
        entries=(entry(11, SLOT_RB), entry(12, 20), entry(13, SLOT_QB), entry(14, 21)),
    )
    assert L.starters_from_roster(roster) == (11, 13)


def test_advise_sim_forwards_state_and_draw() -> None:
    from types import SimpleNamespace

    state, draw = _variance_fixture(opponent_total=38.0, steady_mean=10.0, volatile_high=19.0)
    direct = L.advise(state, draw, week=1, team_id=1)
    through = L.advise_sim(SimpleNamespace(state=state, draw=draw), week=1, team_id=1)
    assert through.recommended.player_ids == direct.recommended.player_ids
    assert through.delta_win_prob == direct.delta_win_prob


# --------------------------------------------------------------------------------------
# The sentence a manager actually reads
# --------------------------------------------------------------------------------------


def _rotation_fixture(n_sims: int = 600) -> tuple[S.LeagueState, Draw]:
    """A roster where ranking both sides of a swap by projection crosses positions.

    The quarterback coming in projects for ten and the receiver coming in for twenty,
    while the quarterback going out projects for six and the receiver going out for four.
    Sort both sides by mean and zip, and the surface tells you to start a receiver over a
    quarterback and a quarterback over a receiver -- two instructions ESPN will not let
    you carry out.
    """
    me = [
        Player(101, QB, 1, "good qb", _normal(n_sims, 10.0, 3.0, 1)),
        Player(102, QB, 2, "bad qb", _normal(n_sims, 6.0, 3.0, 2)),
        Player(110, RB, 3, "rb a", _normal(n_sims, 14.0, 5.0, 3)),
        Player(111, RB, 4, "rb b", _normal(n_sims, 12.0, 5.0, 4)),
        Player(112, RB, 5, "rb c", _normal(n_sims, 9.0, 5.0, 5)),
        Player(120, WR, 6, "good wr", _normal(n_sims, 20.0, 6.0, 6)),
        Player(121, WR, 7, "bad wr", _normal(n_sims, 4.0, 6.0, 7)),
        Player(122, WR, 8, "wr c", _normal(n_sims, 11.0, 6.0, 8)),
        Player(123, WR, 9, "wr d", _normal(n_sims, 10.0, 6.0, 9)),
    ]
    them = [
        Player(200 + i, p, 20 + i, f"opp{i}", _normal(n_sims, 13.0, 6.0, 40 + i))
        for i, p in enumerate((QB, RB, RB, WR, WR, RB))
    ]
    return _league(
        {1: me, 2: them},
        weeks=tuple(range(1, 8)),
        games=tuple((w, 1, 2) for w in range(1, 7)),
        playoff_rounds=((7,),),
        playoff_team_count=2,
        slot_counts=FLEX_SLOTS,
        slot_eligibility=FLEX_ELIGIBILITY,
    )


class TestSwapsAreActionable:
    CURRENT = (102, 110, 111, 121, 122, 123)

    def test_a_swap_never_asks_for_a_move_the_lineup_screen_forbids(self) -> None:
        """The invariant: whoever is benched must be someone who could hold that slot.

        Measured on the user's own league before this was enforced: 31 of 167 swap
        sentences named an out-player who was never eligible for the in-player's slot --
        "start <quarterback> over <receiver>". The lineup in `Move.lineup` was right the
        whole time; the sentence the manager reads was not, and the sentence is the
        product.
        """
        state, draw = _rotation_fixture()
        advice = L.advise(state, draw, week=1, current=self.CURRENT)
        assert advice.swaps, "this fixture is meant to produce a multi-position change"
        positions = {
            int(p): int(state.pool.position_ids[state.pool.index[p]])
            for p in state.franchise(1).player_ids
        }
        for swap in advice.swaps:
            assert positions[swap.out_player_id] in state.slot_eligibility[swap.slot_id], (
                f"{swap.describe()} benches a "
                f"{positions[swap.out_player_id]} out of slot {swap.slot_id}"
            )

    def test_the_swaps_pair_the_quarterback_with_the_quarterback(self) -> None:
        """The concrete pairing, so a regression is named rather than merely detected."""
        state, draw = _rotation_fixture()
        advice = L.advise(state, draw, week=1, current=self.CURRENT)
        assert {(s.in_player_id, s.out_player_id) for s in advice.swaps} == {(120, 121), (101, 102)}
        # And projection order still drives which sentence comes first.
        assert advice.swaps[0].in_player_id == 120

    def test_every_swap_names_a_player_who_is_really_leaving_the_lineup(self) -> None:
        state, draw = _rotation_fixture()
        advice = L.advise(state, draw, week=1, current=self.CURRENT)
        started = set(advice.recommended.player_ids)
        was = set(advice.baseline.player_ids)
        for swap in advice.swaps:
            assert swap.in_player_id in started and swap.in_player_id not in was
            assert swap.out_player_id in was and swap.out_player_id not in started
        assert len({s.out_player_id for s in advice.swaps}) == len(advice.swaps)
