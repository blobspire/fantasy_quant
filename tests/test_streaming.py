"""Multi-week streaming: the relaxation, the exact IP, and the constraints.

Two things are being defended here and they are different in kind.

The first is arithmetic: `solve` claims to be an exact optimum of a stated integer
program, and a claim of optimality that has never met an exhaustive enumeration is a
comment rather than a fact. So every solver is checked against `brute_force` on
instances small enough to enumerate, across random values, byes, ownership patterns and
acquisition costs -- and the LAP relaxation is checked against its own separate
exhaustion over injective assignments, because it solves a *different* problem.

The second is that the plan has to be executable. A plan that starts a defense on its
bye, holds two when the roster allows one, or starts two players in a one-starter slot
is arithmetic that cannot be typed into ESPN, and every one of those is a silent failure
-- the objective goes up and nothing raises.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from fantasy_quant.core import DST, QB, TE, K, MoveKind, PlayerOutlook, WeeklyOutlook
from fantasy_quant.data.ids import DST_TEAMS
from fantasy_quant.decide import streaming as ST
from fantasy_quant.projections.calibration import load as load_calibration
from fantasy_quant.sim import season as S

LEAGUES = [
    (272150391, "Wine Wednesday", 1, 14),
    (161496047, "Blacksburg Baddies", 1, 12),
    (634537479, "Type shi season 2", 2, 12),
]


# --------------------------------------------------------------------------------------
# Fixtures: synthetic grids we can enumerate
# --------------------------------------------------------------------------------------


def make_grid(
    value: np.ndarray,
    *,
    playing: np.ndarray | None = None,
    available: np.ndarray | None = None,
    held: tuple[int, ...] = (0,),
    floor: float = 0.0,
    position_id: int = DST,
    my_team_id: int = 1,
) -> ST.StreamGrid:
    value = np.asarray(value, dtype=float)
    n, n_w = value.shape
    playing = np.ones((n, n_w), bool) if playing is None else np.asarray(playing, bool)
    available = np.ones((n, n_w), bool) if available is None else np.asarray(available, bool)
    held_mask = np.zeros(n, bool)
    for i in held:
        held_mask[i] = True
    streamers = tuple(
        ST.Streamer(
            player_id=-16000 - i,
            name=f"D{i}",
            position_id=position_id,
            pro_team_id=i + 1,
            team=f"T{i}",
            owner=my_team_id if held_mask[i] else None,
            mine=bool(held_mask[i]),
        )
        for i in range(n)
    )
    return ST.StreamGrid(
        league_id=1,
        season=2026,
        position_id=position_id,
        my_team_id=my_team_id,
        model=ST.MATCHUP_MODELS[position_id],
        weeks=tuple(range(1, n_w + 1)),
        streamers=streamers,
        value=value,
        projection=value.copy(),
        playing=playing,
        available=available,
        held=held_mask,
        priced=tuple([True] * n_w),
        floor=np.full(n_w, float(floor)),
    )


def random_grid(n: int, n_w: int, seed: int, *, bye_rate: float = 0.15, owned: int = 0):
    rng = np.random.default_rng(seed)
    value = rng.uniform(0.0, 12.0, (n, n_w))
    playing = rng.random((n, n_w)) > bye_rate
    available = np.ones((n, n_w), bool)
    for i in range(1, 1 + owned):
        available[i, :] = False
    return make_grid(value, playing=playing, available=available)


def brute_lap(grid: ST.StreamGrid) -> float:
    """Exhaustive optimum of the no-reuse assignment, including leaving a week empty."""
    scored = grid.scored()
    best = -np.inf
    for combo in itertools.product(range(-1, grid.n), repeat=grid.n_weeks):
        used = [c for c in combo if c >= 0]
        if len(set(used)) != len(used):
            continue
        total = 0.0
        ok = True
        for j, c in enumerate(combo):
            if c < 0:
                total += float(grid.floor[j])
            elif np.isfinite(scored[c, j]):
                total += float(scored[c, j])
            else:
                ok = False
                break
        if ok:
            best = max(best, total)
    return best


# --------------------------------------------------------------------------------------
# The measured matchup model
# --------------------------------------------------------------------------------------


class TestMatchupModels:
    def test_every_streamed_position_has_a_fit(self):
        assert set(ST.MATCHUP_MODELS) == {QB, TE, DST, K}

    def test_kicker_is_the_least_streamable_position_not_the_most(self):
        """The brief ranks K > DST > TE > QB. Measured within-week, K is last."""
        sd = {p: m.fitted_sd for p, m in ST.MATCHUP_MODELS.items()}
        assert sd[K] == min(sd.values())
        assert sd[K] < ST.STREAMABLE_SD < sd[DST] < sd[QB] < sd[TE]

    def test_only_dst_and_qb_carry_a_market_term(self):
        """TE's and K's fitted Vegas coefficients were noise, so they are zeroed."""
        assert ST.MATCHUP_MODELS[DST].uses_market
        assert ST.MATCHUP_MODELS[QB].uses_market
        assert not ST.MATCHUP_MODELS[TE].uses_market
        assert not ST.MATCHUP_MODELS[K].uses_market

    def test_dst_market_term_is_negative_and_the_projection_is_shrunk(self):
        """A high implied opponent total is bad for a defense, and ESPN's projection is
        under-dispersed, so it gets amplified alone and shrunk beside the market."""
        m = ST.MATCHUP_MODELS[DST]
        assert m.opp_total_coef < 0
        assert m.proj_only_coef > 1.0
        assert m.proj_coef < m.proj_only_coef

    def test_the_market_strictly_adds_information_where_it_is_used(self):
        for pos in (DST, QB):
            m = ST.MATCHUP_MODELS[pos]
            assert m.r2 > m.r2_proj_only


# --------------------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------------------


def _outlook(pid, week, mean, pos=DST, team=1, playing=True):
    return WeeklyOutlook(
        player_id=pid,
        season=2026,
        week=week,
        position_id=pos,
        mean=mean,
        sd=max(mean * 0.6, 1.0),
        p_zero=0.2,
        shape=2.0,
        scale=max(mean / 2, 0.5),
        pro_team_id=team,
        playing=playing,
    )


def _player(pid, name, pos, team, means):
    return PlayerOutlook(
        player_id=pid,
        name=name,
        position_id=pos,
        pro_team_id=team,
        weeks={w: _outlook(pid, w, m, pos, team) for w, m in means.items()},
    )


def _market(weeks, teams, *, priced=(), bye=None):
    """A tiny MarketSchedule. `priced` maps week -> {team: (team_total, opp_total)}."""
    tt, ot, plays = {}, {}, set()
    for t in teams:
        for w in weeks:
            if bye and bye.get(t) == w:
                continue
            plays.add((t, w))
    for w, entries in dict(priced).items():
        for t, (a, b) in entries.items():
            tt[(t, w)] = a
            ot[(t, w)] = b
    return ST.MarketSchedule(
        season=2026,
        team_total=tt,
        opponent_total=ot,
        plays=frozenset(plays),
        weeks=tuple(weeks),
    )


class TestBuildGrid:
    def test_a_rival_roster_removes_a_candidate_and_mine_keeps_him(self):
        outs = [
            _player(-16001, "A D/ST", DST, 1, {1: 6.0, 2: 6.0}),
            _player(-16002, "B D/ST", DST, 2, {1: 7.0, 2: 7.0}),
            _player(-16003, "C D/ST", DST, 3, {1: 8.0, 2: 8.0}),
        ]
        market = _market((1, 2), ("ATL", "BUF", "CHI"))
        g = ST.build_grid(
            outs,
            league_id=1,
            season=2026,
            position_id=DST,
            weeks=(1, 2),
            ownership={-16001: 1, -16002: 5},
            my_team_id=1,
            market=market,
        )
        ids = {s.player_id for s in g.streamers}
        assert ids == {-16001, -16003}
        assert g.held_index == (list(g.streamers).index(next(s for s in g.streamers if s.mine)),)

    def test_a_bye_is_marked_unplayable_even_though_espn_still_projects_it(self):
        """A D/ST projection survives its own bye, so the grid must not trust it.

        ESPN zeroes skill players and kickers on a bye but projects 31 of 32 defences
        normally -- every 2026 player has eighteen weekly rows and a defence's bye row
        is a full one. `pipeline.build` handles that with a bye table now, and this grid
        independently re-derives it from `market.plays`. Both belts, deliberately: this
        module is fed by callers that do not all go through the pipeline.
        """
        outs = [_player(-16001, "A D/ST", DST, 1, {1: 9.0, 2: 9.0})]
        market = _market((1, 2), ("ATL",), bye={"ATL": 2})
        g = ST.build_grid(
            outs,
            league_id=1,
            season=2026,
            position_id=DST,
            weeks=(1, 2),
            ownership={-16001: 1},
            my_team_id=1,
            market=market,
        )
        assert g.projection[0, 1] == pytest.approx(9.0)
        assert g.playing[0, 0] and not g.playing[0, 1]
        assert not np.isfinite(g.scored()[0, 1])

    def test_the_market_reorders_defenses_the_projection_ranks_the_other_way(self):
        """The whole reason the module exists: ESPN's D/ST projection under-reacts."""
        outs = [
            _player(-16001, "A D/ST", DST, 1, {1: 7.0}),
            _player(-16002, "B D/ST", DST, 2, {1: 6.0}),
        ]
        # A faces a 30-point offense, B faces a 13-point one.
        market = _market(
            (1,), ("ATL", "BUF"), priced={1: {"ATL": (20.0, 30.0), "BUF": (20.0, 13.0)}}
        )
        g = ST.build_grid(
            outs,
            league_id=1,
            season=2026,
            position_id=DST,
            weeks=(1,),
            ownership={},
            my_team_id=1,
            market=market,
        )
        assert g.priced == (True,)
        by_id = {s.player_id: i for i, s in enumerate(g.streamers)}
        assert g.projection[by_id[-16001], 0] > g.projection[by_id[-16002], 0]
        assert g.value[by_id[-16002], 0] > g.value[by_id[-16001], 0]

    def test_an_unpriced_week_falls_back_rather_than_inventing_a_line(self):
        outs = [
            _player(-16001, "A D/ST", DST, 1, {1: 7.0, 2: 7.0}),
            _player(-16002, "B D/ST", DST, 2, {1: 4.0, 2: 4.0}),
        ]
        market = _market(
            (1, 2), ("ATL", "BUF"), priced={1: {"ATL": (20.0, 30.0), "BUF": (20.0, 13.0)}}
        )
        g = ST.build_grid(
            outs,
            league_id=1,
            season=2026,
            position_id=DST,
            weeks=(1, 2),
            ownership={},
            my_team_id=1,
            market=market,
        )
        assert g.priced == (True, False)
        i = next(i for i, s in enumerate(g.streamers) if s.player_id == -16001)
        bar = 5.5
        assert g.value[i, 1] == pytest.approx(
            bar + ST.MATCHUP_MODELS[DST].proj_only_coef * (7.0 - bar), abs=1e-6
        )

    def test_a_position_with_no_fitted_model_is_refused(self):
        outs = [_player(1, "A RB", 2, 1, {1: 12.0})]
        with pytest.raises(ST.StreamingError, match="no fitted matchup model"):
            ST.build_grid(
                outs,
                league_id=1,
                season=2026,
                position_id=2,
                weeks=(1,),
                ownership={},
                my_team_id=1,
                market=_market((1,), ("ATL",)),
            )


# --------------------------------------------------------------------------------------
# The relaxation
# --------------------------------------------------------------------------------------


class TestRelaxation:
    @pytest.mark.parametrize("seed", range(6))
    def test_lap_matches_exhaustive_no_reuse_assignment(self, seed):
        g = random_grid(4, 4, seed)
        assert ST.relax_assignment(g).points == pytest.approx(brute_lap(g), abs=1e-9)

    def test_lap_matches_exhaustion_when_a_week_cannot_be_covered(self):
        value = np.array([[5.0, 6.0], [4.0, 7.0]])
        playing = np.array([[True, False], [True, False]])
        g = make_grid(value, playing=playing)
        plan = ST.relax_assignment(g)
        assert plan.start[1] == -1
        assert plan.points == pytest.approx(brute_lap(g), abs=1e-9)

    def test_lap_never_starts_the_same_streamer_twice(self):
        # One candidate is best in every week, so an unconstrained planner reuses him.
        value = np.array([[9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        g = make_grid(value)
        plan = ST.relax_assignment(g)
        used = [i for i in plan.start if i >= 0]
        assert len(set(used)) == len(used)
        assert plan.points == pytest.approx(9.0 + 3.0 + 3.0)

    def test_the_ip_re_acquires_where_the_relaxation_may_not(self):
        """The single named difference between the two models, made to bite."""
        value = np.array([[9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        g = make_grid(value, held=(0,))
        lap = ST.relax_assignment(g)
        ip = ST.solve(g)
        assert ip.start == (0, 0, 0)
        assert ip.points == pytest.approx(27.0)
        assert ip.points > lap.points
        assert ST.brute_force(g).value == pytest.approx(ip.value)

    def test_the_relaxation_is_a_lower_bound_and_the_argmax_an_upper_one(self):
        for seed in range(8):
            g = random_grid(6, 4, seed)
            lap = ST.relax_assignment(g).value
            exact = ST.solve(g).value
            assert lap <= exact + 1e-9
            assert exact <= ST.upper_bound(g) + 1e-9


# --------------------------------------------------------------------------------------
# The exact integer program
# --------------------------------------------------------------------------------------


class TestExactSolve:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("cost", [0.0, 1.0, 3.0])
    def test_dp_equals_brute_force_with_one_roster_slot(self, seed, cost):
        g = random_grid(4, 4, seed)
        dp = ST.solve(g, kappa=1, acquisition_cost=cost)
        bf = ST.brute_force(g, kappa=1, acquisition_cost=cost)
        assert dp.value == pytest.approx(bf.value, abs=1e-9)
        assert dp.optimal

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("cost,slot", [(0.0, 0.0), (2.0, 1.0), (1.0, 4.0)])
    def test_dp_equals_brute_force_with_two_roster_slots(self, seed, cost, slot):
        g = random_grid(4, 4, seed)
        dp = ST.solve(g, kappa=2, acquisition_cost=cost, extra_slot_cost=slot)
        bf = ST.brute_force(g, kappa=2, acquisition_cost=cost, extra_slot_cost=slot)
        assert dp.value == pytest.approx(bf.value, abs=1e-9)

    def test_dp_equals_brute_force_under_a_randomised_sweep(self):
        """The parametrised cases above are hand-picked; this is not.

        Byes, availability that moves week to week, an empty starting roster, a
        non-zero floor and both roster sizes, all drawn at random. A hundred instances
        here and four hundred in the same shape offline have never disagreed with
        exhaustion by more than floating point.
        """
        rng = np.random.default_rng(0)
        checked = 0
        for _ in range(100):
            n, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            value = rng.uniform(0.0, 12.0, (n, w))
            playing = rng.random((n, w)) > rng.uniform(0.0, 0.5)
            available = rng.random((n, w)) > rng.uniform(0.0, 0.5)
            held = (0,) if rng.random() < 0.8 else ()
            for i in held:
                available[i, :] = True
            g = make_grid(
                value,
                playing=playing,
                available=available,
                held=held,
                floor=float(rng.uniform(0.0, 3.0)),
            )
            kappa = int(rng.integers(1, 3))
            if len(held) > kappa:
                continue
            cost, slot = float(rng.choice([0.0, 0.7, 2.5])), float(rng.choice([0.0, 1.5]))
            dp = ST.solve(g, kappa=kappa, acquisition_cost=cost, extra_slot_cost=slot)
            bf = ST.brute_force(g, kappa=kappa, acquisition_cost=cost, extra_slot_cost=slot)
            assert dp.value == pytest.approx(bf.value, abs=1e-9)
            assert_executable(g, dp, kappa)
            checked += 1
        assert checked > 80

    def test_the_dp_and_highs_agree_on_a_real_scale_instance(self):
        """Twenty candidates and seventeen weeks is the live shape, and it is far past
        what brute force can reach. Two independent optimisers landing on the same
        number there is the strongest evidence of optimality available."""
        rng = np.random.default_rng(11)
        value = rng.uniform(0.0, 12.0, (20, 17))
        playing = rng.random((20, 17)) > 0.06
        available = np.ones((20, 17), bool)
        available[5:8, :] = False
        g = make_grid(value, playing=playing, available=available, held=(0,))
        for cost in (0.0, 2.0):
            dp = ST.solve(g, kappa=1, acquisition_cost=cost, method="dp")
            hi = ST.solve(g, kappa=1, acquisition_cost=cost, method="milp")
            assert dp.value == pytest.approx(hi.value, abs=1e-6)
            assert dp.optimal and hi.optimal
            assert_executable(g, hi, 1)

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("kappa", [1, 2])
    def test_the_milp_matches_brute_force(self, seed, kappa):
        rng = np.random.default_rng(100 + seed)
        n, w = 4, 4
        value = rng.uniform(0.0, 12.0, (n, w))
        available = rng.random((n, w)) > 0.3
        available[0, :] = True
        g = make_grid(
            value,
            playing=rng.random((n, w)) > 0.3,
            available=available,
            held=(0,),
            floor=float(rng.uniform(0.0, 2.0)),
        )
        m = ST.solve(g, kappa=kappa, acquisition_cost=1.5, extra_slot_cost=1.0, method="milp")
        bf = ST.brute_force(g, kappa=kappa, acquisition_cost=1.5, extra_slot_cost=1.0)
        assert m.value == pytest.approx(bf.value, abs=1e-6)

    def test_auto_takes_the_dp_at_one_roster_slot_and_the_solver_above_it(self):
        g = random_grid(6, 5, 1)
        assert ST.solve(g, kappa=1).method == "dp-kappa1"
        assert ST.solve(g, kappa=2).method == "milp-kappa2"

    def test_the_pruned_dp_admits_it_is_not_optimal_and_the_solver_is_not_worse(self):
        """`optimal=False` is the whole reason that path is still reachable: it is fast
        to say 'I pruned' and slow to notice you shipped a pruned answer as an optimum."""
        g = random_grid(20, 8, 7)
        pruned = ST.solve(g, kappa=2, acquisition_cost=1.0, method="dp", max_candidates=6)
        exact = ST.solve(g, kappa=2, acquisition_cost=1.0, method="milp")
        assert not pruned.optimal
        assert exact.optimal
        assert exact.value >= pruned.value - 1e-9

    def test_the_milp_honours_the_acquisition_budget(self):
        g = random_grid(10, 8, 5)
        for cap in (0, 2):
            plan = ST.solve(g, kappa=2, max_acquisitions=cap, method="milp")
            assert plan.n_acquisitions <= cap

    def test_a_rival_owned_candidate_is_never_acquired(self):
        value = np.array([[1.0, 1.0], [9.0, 9.0]])
        g = random_grid(2, 2, 0)
        g = make_grid(value, available=np.array([[True, True], [False, False]]))
        plan = ST.solve(g)
        assert all(1 not in r for r in plan.roster)
        assert plan.points == pytest.approx(2.0)

    def test_free_acquisition_closes_the_gap_to_the_upper_bound(self):
        """With nothing to pay, the optimum IS the per-week argmax. A gap here means
        the DP is losing something the bound can see."""
        for seed in range(6):
            g = random_grid(8, 6, seed)
            assert ST.solve(g, acquisition_cost=0.0).gap == pytest.approx(0.0, abs=1e-9)

    def test_a_dear_enough_claim_stops_the_churn_entirely(self):
        """A free optimiser swaps every week; the acquisition cost is the only thing
        standing between the plan and unlimited add/drops."""
        g = random_grid(10, 8, 3)
        free = ST.solve(g, acquisition_cost=0.0)
        dear = ST.solve(g, acquisition_cost=1e6)
        assert free.n_acquisitions >= 6
        assert dear.n_acquisitions == 0
        assert dear.points == pytest.approx(ST.hold_plan(g).points)
        assert free.points > dear.points

    def test_a_cardinality_cap_on_claims_is_honoured(self):
        g = random_grid(10, 8, 5)
        for cap in (0, 1, 3):
            plan = ST.solve(g, max_acquisitions=cap)
            assert plan.n_acquisitions <= cap

    def test_holding_more_than_kappa_already_is_refused_not_silently_dropped(self):
        g = make_grid(np.ones((4, 3)), held=(0, 1))
        with pytest.raises(ST.StreamingError, match="kappa"):
            ST.solve(g, kappa=1)
        ST.solve(g, kappa=2)

    def test_kappa_two_can_stash_a_streamer_ahead_of_a_bye(self):
        """The reason the roster slot couples the weeks at all."""
        value = np.array([[6.0, 6.0], [0.0, 9.0]])
        # The 9-point streamer is only acquirable in week 1, and only plays in week 2.
        playing = np.array([[True, True], [False, True]])
        available = np.array([[True, True], [True, False]])
        g = make_grid(value, playing=playing, available=available, held=(0,))
        assert ST.solve(g, kappa=1).points == pytest.approx(12.0)
        assert ST.solve(g, kappa=2).points == pytest.approx(15.0)


# --------------------------------------------------------------------------------------
# Constraints the plan must satisfy to be executable
# --------------------------------------------------------------------------------------


def assert_executable(grid: ST.StreamGrid, plan: ST.StreamPlan, kappa: int) -> None:
    assert len(plan.start) == grid.n_weeks
    assert len(plan.roster) == grid.n_weeks
    prev = set(grid.held_index)
    for j in range(grid.n_weeks):
        held = set(plan.roster[j])
        assert len(held) <= kappa, f"week {grid.weeks[j]} holds {len(held)} > kappa"
        adds = held - prev
        assert all(grid.available[i, j] for i in adds), "acquired an unavailable streamer"
        assert set(plan.acquired[j]) == adds
        assert set(plan.dropped[j]) == prev - held
        i = plan.start[j]
        # One streamed starter per week: `start` is a single index, so the only way to
        # violate it is to start someone off the roster or on a bye.
        if i >= 0:
            assert i in held, "started a streamer who is not on the roster"
            assert grid.playing[i, j], "started a streamer on a bye"
            assert grid.value[i, j] >= grid.floor[j] - 1e-9, "started a streamer under the floor"
        else:
            # Empty is legitimate for exactly two reasons: nobody held has a game, or
            # the replacement level beats everybody who does. Anything else is a slot
            # left unfilled for nothing.
            best = max(
                (grid.value[k, j] for k in held if grid.playing[k, j]), default=-float("inf")
            )
            assert best <= grid.floor[j] + 1e-9, "left the slot empty for nothing"
        prev = held


class TestConstraints:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("kappa", [1, 2])
    def test_the_exact_plan_is_executable(self, seed, kappa):
        g = random_grid(6, 6, seed, bye_rate=0.3)
        assert_executable(g, ST.solve(g, kappa=kappa), kappa)

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("horizon", [1, 2, 4])
    def test_the_rolling_plan_is_executable(self, seed, horizon):
        g = random_grid(6, 6, seed, bye_rate=0.3)
        assert_executable(g, ST.rolling_plan(g, horizon=horizon), 1)

    def test_a_bye_is_never_started_even_when_it_is_the_best_value(self):
        value = np.array([[99.0, 99.0], [1.0, 1.0]])
        playing = np.array([[False, False], [True, True]])
        g = make_grid(value, playing=playing, held=(0,))
        for plan in (ST.solve(g), ST.rolling_plan(g, horizon=2)):
            assert plan.start == (1, 1), plan.method
            assert plan.points == pytest.approx(2.0)
        # The relaxation may not reuse him, so it covers one week and leaves the other
        # empty -- but it still never starts the 99-point defense on its bye.
        lap = ST.relax_assignment(g)
        assert sorted(lap.start) == [-1, 1]
        assert lap.points == pytest.approx(1.0)

    def test_a_week_nobody_can_cover_scores_the_floor_not_a_phantom_starter(self):
        value = np.array([[5.0, 5.0]])
        playing = np.array([[True, False]])
        g = make_grid(value, playing=playing, floor=1.5)
        plan = ST.solve(g)
        assert plan.start == (0, -1)
        assert plan.points == pytest.approx(6.5)

    def test_the_hold_baseline_leaves_the_slot_empty_on_its_own_bye(self):
        value = np.array([[5.0, 5.0], [8.0, 8.0]])
        playing = np.array([[True, False], [True, True]])
        g = make_grid(value, playing=playing, held=(0,))
        hold = ST.hold_plan(g)
        assert hold.start == (0, -1)
        assert hold.points == pytest.approx(5.0)


# --------------------------------------------------------------------------------------
# Rolling horizon and the measured gap
# --------------------------------------------------------------------------------------


class TestRollingHorizon:
    @pytest.mark.parametrize("seed", range(6))
    def test_full_horizon_rolling_equals_the_exact_solve(self, seed):
        """With the grid frozen, re-solving is arithmetic. The value of rolling is the
        information that arrives between weeks, and this pins that claim."""
        g = random_grid(6, 6, seed)
        assert ST.rolling_plan(g, horizon=6).value == pytest.approx(ST.solve(g).value, abs=1e-9)

    def test_a_short_horizon_can_lose_and_the_loss_is_measured_not_asserted(self):
        """A one-week horizon cannot see a stash-before-the-bye, so it must give some
        value back -- and the report has to be able to say how much."""
        value = np.array([[6.0, 6.0], [0.0, 9.0]])
        playing = np.array([[True, True], [False, True]])
        available = np.array([[True, True], [True, False]])
        g = make_grid(value, playing=playing, available=available, held=(0,))
        report = ST.optimality_gap(g, kappa=2, horizons=(1, 2), brute=True)
        assert report.verified
        assert report.rolling_gap[1] > 0
        assert report.rolling_gap[2] == pytest.approx(0.0, abs=1e-9)

    def test_the_gap_report_ranks_the_bounds_in_the_right_order(self):
        g = random_grid(10, 7, 2)
        r = ST.optimality_gap(g, brute=False)
        assert r.hold <= r.lap <= r.exact <= r.bound + 1e-9

    def test_revalue_is_what_makes_re_solving_worth_anything(self):
        """The hook exists so a caller can fold in lines that posted since. If it is
        ignored, a rolling solve is the same arithmetic twice."""
        g = random_grid(5, 5, 1)
        calls: list[int] = []

        def revalue(grid, j):
            calls.append(j)
            return grid

        ST.rolling_plan(g, horizon=2, revalue=revalue)
        assert calls == list(range(5))


# --------------------------------------------------------------------------------------
# Value decomposition and the recommendation contract
# --------------------------------------------------------------------------------------


class TestValueSplit:
    def test_the_bye_cover_is_separated_from_the_matchup_edge(self):
        value = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        playing = np.array([[True, False, True], [True, True, True]])
        g = make_grid(value, playing=playing, held=(0,))
        plan, hold = ST.solve(g), ST.hold_plan(g)
        split = plan.value_split(g, hold)
        assert split.bye_cover == pytest.approx(6.0)
        assert split.matchup == pytest.approx(2.0)
        assert split.total == pytest.approx(8.0)

    def test_priced_and_unpriced_weeks_add_back_to_the_total(self):
        g = random_grid(8, 6, 4)
        g = ST.replace(g, priced=(True, True, False, False, False, False))
        plan, hold = ST.solve(g), ST.hold_plan(g)
        s = plan.value_split(g, hold)
        assert s.priced + s.unpriced == pytest.approx(s.total + s.cost)
        assert s.bye_cover + s.matchup == pytest.approx(s.total + s.cost)


class TestRecommendationContract:
    def _grid_and_plans(self, position_id=DST):
        value = np.array([[4.0, 4.0, 4.0], [9.0, 9.0, 9.0]])
        g = make_grid(value, held=(0,), position_id=position_id)
        return g, ST.solve(g), ST.hold_plan(g)

    def _eval(self, dt=0.02, se=0.005):
        return ST.StreamEvaluation(
            baseline_title=0.05,
            plan_title=0.05 + dt,
            delta_title=dt,
            stderr=se,
            delta_points=15.0,
            model_points=15.0,
            commit_title=0.06,
            commit_delta=0.01,
            commit_stderr=0.004,
            leverage=0.9,
            n_sims=2000,
        )

    def test_an_upgrade_becomes_an_add_drop_naming_both_players(self):
        g, plan, hold = self._grid_and_plans()
        rec = ST.as_recommendation(g, plan, hold, self._eval())
        assert rec.move.kind is MoveKind.ADD_DROP
        assert [p.player_id for p in rec.move.players] == [
            g.streamers[1].player_id,
            g.streamers[0].player_id,
        ]
        assert rec.move.players[0].to_team == 1 and rec.move.players[1].from_team == 1
        assert rec.move.bid is None, "these leagues are rolling priority, not FAAB"

    def test_no_change_this_week_is_a_hold_with_no_players(self):
        value = np.array([[9.0, 1.0], [1.0, 9.0]])
        g = make_grid(value, held=(0,))
        plan, hold = ST.solve(g), ST.hold_plan(g)
        rec = ST.as_recommendation(g, plan, hold, self._eval())
        assert rec.move.kind is MoveKind.HOLD
        assert rec.move.players == ()

    def test_an_effect_inside_its_own_error_is_reported_as_not_significant(self):
        g, plan, hold = self._grid_and_plans()
        rec = ST.as_recommendation(g, plan, hold, self._eval(dt=0.002, se=0.003))
        assert not rec.significant

    def test_the_kicker_recommendation_says_it_is_not_streamable(self):
        g, plan, hold = self._grid_and_plans(position_id=K)
        rec = ST.as_recommendation(g, plan, hold, self._eval())
        assert "not-streamable" in rec.tags
        assert rec.confidence == "low"
        assert "not meaningfully streamable" in rec.rationale

    def test_the_dst_recommendation_does_not(self):
        g, plan, hold = self._grid_and_plans(position_id=DST)
        rec = ST.as_recommendation(g, plan, hold, self._eval())
        assert "not-streamable" not in rec.tags

    def test_a_position_whose_fit_uses_no_line_is_not_reported_as_unpriced(self):
        """TE's market coefficients are zero, so it is never short of a line -- calling
        it 'partially-unpriced' would promise information that would never arrive."""
        g, plan, hold = self._grid_and_plans(position_id=TE)
        g = ST.replace(g, priced=(False,) * g.n_weeks)
        rec = ST.as_recommendation(g, plan, hold, self._eval())
        assert "partially-unpriced" not in rec.tags
        assert rec.confidence == "high"
        assert "indistinguishable from zero" in rec.rationale

    def test_a_dst_week_with_no_line_is_flagged_and_downgraded(self):
        g, plan, hold = self._grid_and_plans(position_id=DST)
        g = ST.replace(g, priced=(True, False, False))
        rec = ST.as_recommendation(g, plan, hold, self._eval())
        assert "partially-unpriced" in rec.tags
        assert rec.confidence == "medium"

    def test_the_rationale_carries_the_week_by_week_plan(self):
        g, plan, hold = self._grid_and_plans()
        rec = ST.as_recommendation(g, plan, hold, self._eval())
        for w in g.weeks:
            assert f"w{w} " in rec.rationale


class TestPlanTable:
    def test_one_row_per_week_with_the_hold_it_replaces(self):
        value = np.array([[4.0, 4.0], [9.0, 9.0]])
        g = make_grid(value, held=(0,))
        rows = ST.plan_table(g, ST.solve(g))
        assert [r["week"] for r in rows] == [1, 2]
        assert rows[0]["start"] == "D1" and rows[0]["hold"] == "D0"
        assert rows[0]["gain"] == pytest.approx(5.0)
        assert rows[0]["add"] == ("D1",) and rows[0]["drop"] == ("D0",)


# --------------------------------------------------------------------------------------
# The replacement level, which every solver has to price the same way
# --------------------------------------------------------------------------------------


class TestTheFloorIsAChoiceNotAFallback:
    """A non-zero `floor` is what a caller passes when an unfilled slot still streams
    something. Reading it only as "nobody is startable" makes benching impossible, and
    that mistake was invisible for as long as the DP and the exhaustion shared the
    function that made it -- so every check here also runs against HiGHS, which does not.
    """

    def test_a_held_streamer_under_the_replacement_level_is_benched(self):
        g = make_grid(np.array([[1.0, 1.0]]), held=(0,), floor=5.0)
        hold = ST.hold_plan(g)
        assert hold.start == (-1, -1), "started a 1.0 defense against a 5.0 replacement level"
        assert hold.points == pytest.approx(10.0)

    def test_the_dp_does_not_have_to_drop_him_to_bench_him(self):
        """The distinction the old code could not express. With a claim priced at 100
        the only way to collect the floor was to drop the incumbent, so the DP either
        started a player worth less than the floor or paid to be rid of him."""
        g = make_grid(np.array([[1.0, 1.0]]), held=(0,), floor=5.0)
        for method in ("dp", "milp"):
            plan = ST.solve(g, kappa=1, acquisition_cost=100.0, method=method)
            assert plan.value == pytest.approx(10.0), method
            assert plan.n_acquisitions == 0, method

    @pytest.mark.parametrize("cost", [0.0, 1.0, 3.0])
    def test_dp_brute_force_and_highs_agree_once_the_floor_binds(self, cost):
        """Two implementations of one mistake agreeing is not a proof of optimality.
        Before the fix this disagreed with HiGHS on 33 of 300 instances by up to three
        points, with `optimal=True` on every one of them."""
        rng = np.random.default_rng(int(cost * 17) + 4)
        checked = 0
        for _ in range(60):
            n, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            available = rng.random((n, w)) > 0.25
            available[0, :] = True
            g = make_grid(
                rng.uniform(0.0, 12.0, (n, w)),
                playing=rng.random((n, w)) > 0.25,
                available=available,
                held=(0,),
                floor=float(rng.uniform(1.0, 6.0)),
            )
            dp = ST.solve(g, kappa=1, acquisition_cost=cost, method="dp")
            hi = ST.solve(g, kappa=1, acquisition_cost=cost, method="milp")
            bf = ST.brute_force(g, kappa=1, acquisition_cost=cost)
            assert dp.value == pytest.approx(hi.value, abs=1e-6)
            assert bf.value == pytest.approx(hi.value, abs=1e-6)
            assert dp.optimal
            assert_executable(g, dp, 1)
            checked += 1
        assert checked == 60

    def test_the_subset_dp_prices_the_floor_the_same_way(self):
        rng = np.random.default_rng(21)
        for _ in range(30):
            g = make_grid(
                rng.uniform(0.0, 12.0, (4, 3)),
                playing=rng.random((4, 3)) > 0.25,
                available=rng.random((4, 3)) > 0.25,
                held=(0,),
                floor=float(rng.uniform(1.0, 6.0)),
            )
            kw = {"kappa": 2, "acquisition_cost": 1.5, "extra_slot_cost": 1.0}
            assert ST.solve(g, method="dp", **kw).value == pytest.approx(
                ST.solve(g, method="milp", **kw).value, abs=1e-6
            )

    def test_the_upper_bound_is_actually_an_upper_bound_when_the_floor_binds(self):
        """`gap` clamps at zero, so a bound below the optimum is silent rather than
        loud. With floor 5 over candidates worth 1 this used to return 3 against a true
        optimum of 15 and still report a gap of 0.0."""
        g = make_grid(np.full((1, 3), 1.0), held=(0,), floor=5.0)
        assert ST.upper_bound(g) == pytest.approx(15.0)
        assert ST.solve(g).value <= ST.upper_bound(g) + 1e-9
        assert ST.solve(g, method="milp").value <= ST.upper_bound(g) + 1e-9

    def test_the_upper_bound_holds_when_availability_moves_week_to_week(self):
        """A streamer signed in week 1 is yours in week 2 whether or not he was still on
        the wire then, so a bound that reads only the week's own availability mask is
        not a bound: the DP scored 24.39 against a 'bound' of 24.15 on a grid this
        shape."""
        rng = np.random.default_rng(99)
        for _ in range(80):
            n, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
            available = rng.random((n, w)) > 0.4
            available[0, :] = True
            g = make_grid(
                rng.uniform(0.0, 12.0, (n, w)),
                playing=rng.random((n, w)) > 0.2,
                available=available,
                held=(0,),
                floor=float(rng.uniform(0.0, 4.0)),
            )
            bound = ST.upper_bound(g)
            assert ST.solve(g, method="milp").value <= bound + 1e-9
            assert ST.solve(g, method="dp").value <= bound + 1e-9


# --------------------------------------------------------------------------------------
# A drop is final
# --------------------------------------------------------------------------------------


def _te_outlooks(*, incumbent_bye: int, weeks: tuple[int, ...]) -> list[PlayerOutlook]:
    """An elite incumbent with one bye, and a replacement-level free agent who does not.

    Deliberately the shape of the live QB and TE grids: the only week the free agent is
    worth starting is the week the incumbent has no game.
    """
    return [
        _player(-1, "Star TE", TE, 1, {w: (0.0 if w == incumbent_bye else 14.0) for w in weeks}),
        _player(-2, "Waiver TE", TE, 2, dict.fromkeys(weeks, 5.0)),
    ]


class TestADropIsFinal:
    def test_the_plan_does_not_drop_a_starter_and_re_sign_him_next_week(self):
        """The live failure this exists for: the QB grid returned 'drop Jalen Hurts in
        week 10, re-add him in week 11' and the TE grid the same for Colston Loveland,
        because a player you own was marked acquirable in every week forever."""
        weeks = (1, 2, 3)
        g = ST.build_grid(
            _te_outlooks(incumbent_bye=2, weeks=weeks),
            league_id=1,
            season=2026,
            position_id=TE,
            weeks=weeks,
            ownership={-1: 1},
            my_team_id=1,
            market=_market(weeks, ("ATL", "BUF"), bye={"ATL": 2}),
        )
        plan = ST.solve(g)
        star = g.index_of(-1)
        assert plan.roster == ((star,),) * 3, "dropped the starter to rent a bye week"
        assert plan.n_acquisitions == 0
        assert plan.start == (star, -1, star)

    def test_the_option_is_still_available_when_the_caller_says_it_is_real(self):
        weeks = (1, 2, 3)
        kw = {
            "league_id": 1,
            "season": 2026,
            "position_id": TE,
            "weeks": weeks,
            "ownership": {-1: 1},
            "my_team_id": 1,
            "market": _market(weeks, ("ATL", "BUF"), bye={"ATL": 2}),
        }
        outs = _te_outlooks(incumbent_bye=2, weeks=weeks)
        loose = ST.build_grid(outs, readd_dropped=True, **kw)
        tight = ST.build_grid(outs, **kw)
        star = loose.index_of(-1)
        assert ST.solve(loose).start[1] == loose.index_of(-2), "should rent the bye week"
        assert ST.solve(loose).roster[2] == (star,), "and take him back"
        assert ST.solve(loose).value > ST.solve(tight).value
        # The difference is exactly one week of the waiver replacement, which is what
        # the free option is worth and what the default declines to spend.
        assert ST.solve(loose).value - ST.solve(tight).value == pytest.approx(
            loose.value[loose.index_of(-2), 1], abs=1e-9
        )

    def test_a_player_you_own_is_held_not_acquirable(self):
        weeks = (1, 2)
        g = ST.build_grid(
            [_player(-1, "Mine D/ST", DST, 1, dict.fromkeys(weeks, 8.0))],
            league_id=1,
            season=2026,
            position_id=DST,
            weeks=weeks,
            ownership={-1: 1},
            my_team_id=1,
            market=_market(weeks, ("ATL",)),
        )
        i = g.index_of(-1)
        assert g.held[i]
        assert not g.available[i].any(), "owning him today is not a standing right to re-sign him"
        # Holding is never gated on availability, so the null plan is still feasible.
        assert ST.hold_plan(g).points == pytest.approx(16.0)
        assert ST.solve(g).start == (i, i)


# --------------------------------------------------------------------------------------
# The scoring scale is a league fact, not a fact about who your rivals rostered
# --------------------------------------------------------------------------------------


class TestMarketRescaling:
    def test_a_rival_claim_does_not_move_the_market_coefficient(self):
        """The rescaling exists so a league that scores D/ST differently moves the
        points-per-point market term. Estimating it off the *candidate pool* instead
        made it move when a rival rostered the good defenses: on Wine Wednesday the pool
        mean was 4.28 against 4.91 for all thirty-two, shrinking the coefficient 18%."""
        weeks = (1,)
        outs = [
            _player(-16001, "A D/ST", DST, 1, {1: 9.0}),
            _player(-16002, "B D/ST", DST, 2, {1: 6.0}),
            _player(-16003, "C D/ST", DST, 3, {1: 3.0}),
        ]
        market = _market(
            weeks,
            ("ATL", "BUF", "CHI"),
            priced={1: {"ATL": (21.0, 27.0), "BUF": (21.0, 21.0), "CHI": (21.0, 15.0)}},
        )
        kw = {
            "league_id": 1,
            "season": 2026,
            "position_id": DST,
            "weeks": weeks,
            "my_team_id": 1,
            "market": market,
        }
        wide = ST.build_grid(outs, ownership={-16002: 1}, **kw)
        # A rival takes the best defense off the board. Nothing about the league changed.
        narrow = ST.build_grid(outs, ownership={-16002: 1, -16001: 5}, **kw)
        b_wide = wide.value[wide.index_of(-16002), 0] - wide.projection[wide.index_of(-16002), 0]
        b_narrow = (
            narrow.value[narrow.index_of(-16002), 0] - narrow.projection[narrow.index_of(-16002), 0]
        )
        # Both grids still centre on their own pool mean, so compare the market term
        # itself: the spread the Vegas coefficient buys per point of implied total.
        assert wide.priced == (True,) and narrow.priced == (True,)
        assert b_wide - b_narrow == pytest.approx(
            wide.value[wide.index_of(-16002), 0]
            - narrow.value[narrow.index_of(-16002), 0]
            - (
                wide.projection[wide.index_of(-16002), 0]
                - narrow.projection[narrow.index_of(-16002), 0]
            ),
            abs=1e-9,
        )
        spread_wide = wide.value[wide.index_of(-16003), 0] - wide.value[wide.index_of(-16002), 0]
        spread_narrow = (
            narrow.value[narrow.index_of(-16003), 0] - narrow.value[narrow.index_of(-16002), 0]
        )
        assert spread_wide == pytest.approx(spread_narrow, abs=1e-9)


# --------------------------------------------------------------------------------------
# The simulated half, offline
# --------------------------------------------------------------------------------------

_SLOT_COUNTS = {0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1}
_SLOT_ELIGIBILITY = {
    0: frozenset({1}),
    2: frozenset({2}),
    4: frozenset({3}),
    6: frozenset({4}),
    16: frozenset({16}),
    17: frozenset({5}),
    23: frozenset({2, 3, 4}),
}
_SKILL = ((1, 18.0), (2, 12.0), (2, 11.0), (3, 11.0), (3, 10.0), (4, 8.0), (3, 9.0), (5, 8.0))
_CALIBRATION = load_calibration("ppr")


def _sim_league(*, n_teams: int = 8, weeks: tuple[int, ...] = (1, 2, 3, 4, 5, 6)):
    """A whole league plus a D/ST wire, offline and deterministic.

    Exists because the module's simulated half -- `_extended_state`, `_grid_outlooks`,
    the paired arms, `_first_week_leverage` -- had no coverage at all outside the
    network tests, and those assert things that cannot fail (`-1 <= delta_title <= 1`).

    Skill players sit on `pro_team_id = 0` so the nflverse bye map, which is keyed by
    real team ids, cannot reach them; the defenses use real ids because the bye map is
    the whole point of passing a `MarketSchedule` in.
    """
    reg, playoff = weeks[:-2], weeks[-2:]
    defenses = DST_TEAMS[: n_teams + 6]
    rows: list[tuple[int, int, int, str]] = []
    rosters: list[tuple[int, ...]] = []
    outlooks: list[PlayerOutlook] = []

    def outlook(pid, name, pos, team, means):
        return PlayerOutlook(
            player_id=pid,
            name=name,
            position_id=pos,
            pro_team_id=team,
            weeks={
                w: _CALIBRATION.outlook(
                    player_id=pid,
                    season=2026,
                    week=w,
                    position_id=pos if pos in (1, 2, 3, 4) else 0,
                    projection=m,
                    pro_team_id=team,
                )
                for w, m in means.items()
            },
        )

    for t in range(n_teams):
        strength = 0.92 + 0.02 * t
        ids: list[int] = []
        for j, (pos, base) in enumerate(_SKILL):
            pid = 1000 * (t + 1) + j
            ids.append(pid)
            rows.append((pid, pos, 0, f"p{pid}"))
            outlooks.append(
                outlook(pid, f"p{pid}", pos, 0, dict.fromkeys(weeks, max(base * strength, 0.5)))
            )
        d = defenses[t]
        ids.append(d.pro_team_id)
        rows.append((d.pro_team_id, DST, d.pro_team_id, f"{d.nflverse} D/ST"))
        rosters.append(tuple(ids))

    # Every defense in the league, rostered or not, gets an outlook. The rostered ones
    # are worth 5; the free agents alternate 3 and 9 so streaming has something to find.
    for k, d in enumerate(defenses):
        mine = k < n_teams
        base = 5.0 if mine else (9.0 if k % 2 else 3.0)
        outlooks.append(
            outlook(
                d.pro_team_id,
                f"{d.nflverse} D/ST",
                DST,
                d.pro_team_id,
                dict.fromkeys(weeks, base),
            )
        )

    games = []
    order = list(range(n_teams))
    for w in reg:
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
        league_id=99,
        season=2026,
        name="offline",
        franchises=tuple(
            S.Franchise(team_id=i + 1, name=f"T{i + 1}", player_ids=rosters[i], is_user=(i == 0))
            for i in range(n_teams)
        ),
        pool=S.PlayerPool.of(rows),
        weeks=weeks,
        remaining_games=tuple(games),
        lineup_slot_counts=_SLOT_COUNTS,
        slot_eligibility=_SLOT_ELIGIBILITY,
        playoff_team_count=4,
        playoff_rounds=((playoff[0],), (playoff[1],)),
        my_team_id=1,
    )
    market = _market(weeks, tuple(d.nflverse for d in DST_TEAMS), bye={defenses[0].nflverse: 3})
    return state, outlooks, market


class _FakeSim:
    """The three attributes `recommend` reads off a `pipeline.LeagueSim`."""

    def __init__(self, state, outlooks, n_sims=400):
        self.state, self.outlooks, self.n_sims = state, outlooks, n_sims


class TestTheSimulatedHalf:
    @staticmethod
    @pytest.fixture(scope="class")
    def league():
        return _sim_league()

    def _grid(self, state, outlooks, market, **kw):
        own = {pid: f.team_id for f in state.franchises for pid in f.player_ids}
        return ST.build_grid(
            outlooks,
            league_id=state.league_id,
            season=2026,
            position_id=DST,
            weeks=state.weeks,
            ownership=own,
            my_team_id=1,
            market=market,
            **kw,
        )

    def test_a_plan_identical_to_holding_prices_at_exactly_zero(self, league):
        """The strongest available check on the pairing: the two arms are the same
        season, so every simulation must cancel to the bit. A delta that is merely small
        here means the arms are not actually paired."""
        state, outlooks, market = league
        g = self._grid(state, outlooks, market)
        hold = ST.hold_plan(g)
        ev = ST.evaluate(
            state, outlooks, g, hold, baseline=hold, my_team_id=1, n_sims=400, market=market
        )
        assert ev.delta_title == 0.0
        assert ev.delta_points == 0.0
        assert ev.stderr == 0.0

    def test_the_paired_error_beats_the_unpaired_one(self, league):
        """`stderr` must be the SD of the per-simulation difference. If it were the two
        arms' independent errors the module would call every true effect noise."""
        state, outlooks, market = league
        g = self._grid(state, outlooks, market)
        plan, hold = ST.solve(g), ST.hold_plan(g)
        ev = ST.evaluate(
            state, outlooks, g, plan, baseline=hold, my_team_id=1, n_sims=1500, market=market
        )
        p, b = ev.plan_title, ev.baseline_title
        unpaired = math.sqrt((p * (1 - p) + b * (1 - b)) / 1500)
        assert 0.0 < ev.stderr < unpaired
        assert ev.delta_title > 0.0

    def test_the_simulated_gain_tracks_the_grid_it_was_planned_on(self, league):
        """The optimiser and the simulator have to agree about what a defense is worth.
        A large disagreement means the grid is promising points the roster cannot start
        -- which is exactly what happens if `apply_matchup_model` stops feeding the
        plan's own forecast into the tensor."""
        state, outlooks, market = league
        g = self._grid(state, outlooks, market)
        plan, hold = ST.solve(g), ST.hold_plan(g)
        ev = ST.evaluate(
            state, outlooks, g, plan, baseline=hold, my_team_id=1, n_sims=1500, market=market
        )
        assert ev.model_points > 0
        assert ev.delta_points == pytest.approx(ev.model_points, rel=0.15)

    def test_the_bye_week_is_not_quietly_covered_for_free(self, league):
        """Week 3 is the incumbent's bye. Holding must score nothing in that slot, which
        is what makes the bye cover worth a whole starter rather than an upgrade."""
        state, outlooks, market = league
        g = self._grid(state, outlooks, market)
        hold = ST.hold_plan(g)
        j = g.weeks.index(3)
        assert hold.start[j] == -1
        split = ST.solve(g).value_split(g, hold)
        assert split.bye_cover > 0

    def test_recommend_plans_around_a_roster_that_already_holds_more_than_kappa(self, league):
        """A roster with two quarterbacks is the ordinary case and this used to raise
        `StreamingError` on the user's own league rather than plan for it."""
        state, outlooks, market = league
        spare = DST_TEAMS[len(state.franchises) + 1]
        me = state.franchise(1)
        state2 = state.with_franchise(me.with_players((*me.player_ids, spare.pro_team_id)))
        state2 = ST.replace(state2, pool=S.PlayerPool.of([*_pool_rows(state), _dst_row(spare)]))
        rec = ST.recommend(
            _FakeSim(state2, outlooks), position_id=DST, n_sims=300, market=market, kappa=1
        )
        assert rec.move.league_id == 99
        assert rec.rationale

    def test_recommend_reports_a_hold_as_a_hold_and_says_the_delta_is_the_plan(self, league):
        state, outlooks, market = league
        rec = ST.recommend(_FakeSim(state, outlooks), position_id=DST, n_sims=300, market=market)
        if rec.move.kind is MoveKind.HOLD:
            assert "no-action-this-week" in rec.tags
            assert "nothing to execute today" in rec.rationale
        assert "not the value of the single move above" in rec.rationale


def _pool_rows(state):
    return [
        (int(p), int(pos), int(team), state.pool.name(int(p)))
        for p, pos, team in zip(
            state.pool.player_ids, state.pool.position_ids, state.pool.pro_team_ids, strict=True
        )
    ]


def _dst_row(team):
    return (team.pro_team_id, DST, team.pro_team_id, f"{team.nflverse} D/ST")


# --------------------------------------------------------------------------------------
# Live leagues
# --------------------------------------------------------------------------------------


@pytest.mark.network
class TestAgainstTheRealLeagues:
    @staticmethod
    @pytest.fixture(scope="class")
    def market():
        return ST.market_schedule(2026)

    @staticmethod
    @pytest.fixture(scope="class")
    def sims():
        from fantasy_quant import pipeline as P

        client = P.client_from_env()
        try:
            yield {
                lid: P.build(lid, 2026, my_team_id=me, client=client, n_sims=500, seed=7)
                for lid, _, me, _ in LEAGUES
            }
        finally:
            client.close()

    @pytest.mark.parametrize("league_id,name,my_team,size", LEAGUES)
    def test_the_dst_plan_is_executable_and_beats_holding(
        self, sims, market, league_id, name, my_team, size
    ):
        sim = sims[league_id]
        own = {pid: f.team_id for f in sim.state.franchises for pid in f.player_ids}
        g = ST.build_grid(
            sim.outlooks,
            league_id=league_id,
            season=2026,
            position_id=DST,
            weeks=sim.state.weeks,
            ownership=own,
            my_team_id=my_team,
            market=market,
        )
        # 32 defenses minus the ones RIVALS roster -- my own is a candidate, because
        # holding it is one of the choices the plan is picking between. So the ceiling is
        # `size - 1` rostered, not `size`. The old bound only held while some rival was
        # carrying no defence at all, and it started failing the week they all had one.
        assert 32 - (size - 1) >= g.n >= 32 - 2 * size
        plan = ST.solve(g)
        assert_executable(g, plan, 1)
        assert plan.optimal
        assert plan.value > ST.hold_plan(g).value
        # Two independent optimisers on a 20x17 instance far past exhaustion. This is
        # the only optimality evidence available at real scale, and it is worth more
        # than the old `gap == 0`, which merely restated that free acquisition reaches
        # the per-week argmax and was true before the module solved anything correctly.
        assert plan.value == pytest.approx(ST.solve(g, method="milp").value, abs=1e-6)
        assert ST.hold_plan(g).value <= plan.value <= ST.upper_bound(g) + 1e-9

    @pytest.mark.parametrize("league_id,name,my_team,size", LEAGUES)
    def test_the_plan_never_drops_a_starter_and_re_signs_him(
        self, sims, market, league_id, name, my_team, size
    ):
        """The live failure that changed the default. At QB and TE the incumbent beats
        every free agent in every week, so the only way to 'gain' is to rent his bye
        week -- which means dropping Jalen Hurts and picking him back up seven days
        later. Nothing this module returns may require that."""
        sim = sims[league_id]
        own = {pid: f.team_id for f in sim.state.franchises for pid in f.player_ids}
        for pos in (QB, TE, DST, K):
            g = ST.build_grid(
                sim.outlooks,
                league_id=league_id,
                season=2026,
                position_id=pos,
                weeks=sim.state.weeks,
                ownership=own,
                my_team_id=my_team,
                market=market,
            )
            mine = {i for i, s in enumerate(g.streamers) if s.mine}
            plan = ST.solve(g, kappa=max(1, len(g.held_index)))
            dropped: set[int] = set()
            for j in range(g.n_weeks):
                re_added = mine & set(plan.acquired[j]) & dropped
                assert not re_added, (
                    f"{name} {pos}: plan re-signs "
                    f"{[g.streamers[i].name for i in re_added]} after dropping him"
                )
                dropped |= set(plan.dropped[j])

    @pytest.mark.parametrize("league_id,name,my_team,size", LEAGUES)
    def test_a_streaming_recommendation_is_a_comparable_recommendation(
        self, sims, league_id, name, my_team, size
    ):
        sim = sims[league_id]
        rec = ST.recommend(sim, position_id=DST, n_sims=500)
        assert rec.rationale
        assert rec.move.league_id == league_id
        assert rec.move.bid is None
        # The move has to be typeable into ESPN: you may only add someone nobody
        # rosters, and you may only drop someone you actually have.
        own = {pid: f.team_id for f in sim.state.franchises for pid in f.player_ids}
        for p in rec.move.players:
            if p.to_team is not None:
                assert p.to_team == my_team
                assert own.get(p.player_id) in (None, my_team), "claimed a rostered player"
            if p.from_team is not None:
                assert p.from_team == my_team
                assert own.get(p.player_id) == my_team, "dropped a player you do not roster"
        if rec.move.kind is MoveKind.HOLD:
            assert rec.move.players == ()
            assert "no-action-this-week" in rec.tags, (
                "a hold that carries the plan's delta must say the delta is not the move's"
            )

    @pytest.mark.parametrize("league_id,name,my_team,size", LEAGUES)
    def test_the_reported_error_actually_covers_the_seed_to_seed_spread(
        self, sims, league_id, name, my_team, size
    ):
        """`stderr` is the paired Monte Carlo error, and the claim it makes is that a
        rerun lands inside it. Two independent seeds must agree to within a few of
        them, or `significant` is certifying an effect smaller than its own noise."""
        sim = sims[league_id]
        a = ST.recommend(sim, position_id=DST, n_sims=1500, seed=11)
        b = ST.recommend(sim, position_id=DST, n_sims=1500, seed=404)
        pooled = math.sqrt(a.stderr**2 + b.stderr**2)
        assert pooled > 0
        assert abs(a.delta_title - b.delta_title) <= 4.0 * pooled

    def test_the_kicker_gain_is_mostly_a_bye_cover_not_a_matchup_edge(self, sims, market):
        """If this ever inverts, the K model has found signal it did not have in the
        2022-2025 fit and the constants should be re-measured before believing it."""
        sim = sims[161496047]
        own = {pid: f.team_id for f in sim.state.franchises for pid in f.player_ids}
        g = ST.build_grid(
            sim.outlooks,
            league_id=161496047,
            season=2026,
            position_id=K,
            weeks=sim.state.weeks,
            ownership=own,
            my_team_id=1,
            market=market,
        )
        split = ST.solve(g).value_split(g)
        assert split.bye_cover > split.matchup

    def test_the_2026_market_is_only_partly_posted(self, market):
        """160 of 272 games unpriced on 2026-09-07. The plan must degrade, not invent."""
        assert market.priced_weeks
        assert len(market.priced_weeks) < 18
        assert len(market.bye_by_pro_team_id()) == 32
