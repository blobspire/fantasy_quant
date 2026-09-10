"""The cross-league portfolio view.

The acceptance tests here are the ones that would catch a portfolio number that looks
plausible and is wrong: `P(>=1 title)` must lie inside the Frechet bounds and must not be
the sum of the marginals; two identical leagues must produce a *different* portfolio
number from two independent ones, which is the whole claim that coupling the draws buys
something; the exposure count must survive a player held in two leagues out of three;
the concentration damage must move when the simulated season moves, because a heuristic
would not; and dropping a roster's only quarterback must make it worse rather than
turning it into a 93%-title machine, which is what the shared floor code does if its
empty-position guard is missing.

Everything runs offline off small synthetic leagues built once per session. The live
three-league check is marked `network`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fantasy_quant.core import Move, MoveKind, PlayerMove, PlayerOutlook, Recommendation
from fantasy_quant.decide.title import streaming_replacement
from fantasy_quant.edges import portfolio as PF
from fantasy_quant.pipeline import LeagueSim
from fantasy_quant.projections.calibration import load as load_calibration
from fantasy_quant.sim import season as S
from fantasy_quant.sim.distributions import WeeklySampler
from fantasy_quant.sim.lineup import plan_from_slots

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
#: Nine starters then a three-man bench, so the flex has a real choice, the bench has
#: option value, and exactly one quarterback / kicker / defence sits on the roster --
#: which is what makes the empty-position floor trap reachable, as it is on every one of
#: the user's real teams.
POSITIONS = (1, 2, 2, 3, 3, 4, 3, 16, 5, 3, 2, 4)

_BASE_POINTS = {1: 18.0, 2: 12.0, 3: 11.0, 4: 8.0, 5: 8.0, 16: 7.0}

#: The real fitted calibration, off disk with no network, so a synthetic player's
#: `(p_zero, shape, scale)` is internally consistent with the mean and SD the simulator
#: reads back off the panel.
_CALIBRATION = load_calibration("ppr")

REG_WEEKS = (1, 2, 3, 4, 5, 6)
ROUNDS = ((7,), (8,))
WEEKS = tuple(sorted(set(REG_WEEKS) | {w for r in ROUNDS for w in r}))
N_SIMS = 800


def _weekly(pid: int, week: int, pos: int, mean: float, team: int):
    return _CALIBRATION.outlook(
        player_id=pid,
        season=2026,
        week=week,
        position_id=pos if pos in (1, 2, 3, 4) else 0,
        projection=mean,
        pro_team_id=team,
    )


def build_sim(
    league_id: int,
    *,
    n_teams: int = 8,
    seed: int = 11,
    id_offset: int = 0,
    my_id_offset: int | None = None,
    n_sims: int = N_SIMS,
    strength: list[float] | None = None,
    wire_strength: float | None = None,
) -> LeagueSim:
    """A whole synthetic league as a `pipeline.LeagueSim`, deterministic in `seed`.

    `id_offset` shifts every player id, and `my_id_offset` shifts team 1's separately.
    Two leagues built with the same `my_id_offset` and the same `seed` therefore hold the
    *same players* on the user's team against different opposition, which is the shape
    the whole module exists to reason about: leagues that share a roster and nothing else.

    Team 1 -- the user -- is the *strongest* team by construction, which is not cosmetic.
    `streaming_replacement` fits the wire off the pool, so on a league where the user has
    the worst quarterback the replacement level *is* his quarterback, deleting him costs
    nothing, and every concentration test would pass for the wrong reason.

    `wire_strength` adds two UNROSTERED players per position at that strength -- present
    in `outlooks`, absent from `state.pool`. That is the live shape (`pipeline.build`
    pools only rostered players while `league_projections` reads the whole corpus) and
    without it this fixture cannot tell the wire from the roster bottom, which is exactly
    why it did not catch `build_portfolio` reading the panel. Default `None` keeps every
    existing expectation in this file unchanged.
    """
    strength = strength or [1.08 - 0.04 * i for i in range(n_teams)]
    mine = id_offset if my_id_offset is None else my_id_offset
    rows: list[tuple[int, int, int, str]] = []
    rosters: list[tuple[int, ...]] = []
    outlooks: list[PlayerOutlook] = []
    for t in range(n_teams):
        base = (mine if t == 0 else id_offset) + 1000 * (t + 1)
        ids = []
        for j, pos in enumerate(POSITIONS):
            pid = base + j
            nfl = ((base // 1000 + j) % 30) + 1
            ids.append(pid)
            rows.append((pid, pos, nfl, f"p{pid}"))
            mean = max(_BASE_POINTS[pos] * strength[t] * (1.0 - 0.07 * j), 0.5)
            outlooks.append(
                PlayerOutlook(
                    player_id=pid,
                    name=f"p{pid}",
                    position_id=pos,
                    pro_team_id=nfl,
                    weeks={w: _weekly(pid, w, pos, mean, nfl) for w in WEEKS},
                )
            )
        rosters.append(tuple(ids))

    if wire_strength is not None:
        # `outlooks` only. These have no column in `state.pool`, which is what makes them
        # free agents rather than a thirteenth roster.
        for k, pos in enumerate(sorted(set(POSITIONS))):
            for d in range(2):
                pid = id_offset + 900_000 + 10 * k + d
                nfl = ((pid // 1000 + k) % 30) + 1
                mean = max(_BASE_POINTS[pos] * wire_strength * (1.0 - 0.05 * d), 0.5)
                outlooks.append(
                    PlayerOutlook(
                        player_id=pid,
                        name=f"fa{pid}",
                        position_id=pos,
                        pro_team_id=nfl,
                        weeks={w: _weekly(pid, w, pos, mean, nfl) for w in WEEKS},
                    )
                )

    order = list(range(n_teams))
    games = []
    for w in REG_WEEKS:
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
        league_id=league_id,
        season=2026,
        name=f"league {league_id}",
        franchises=tuple(
            S.Franchise(team_id=i + 1, name=f"T{i + 1}", player_ids=rosters[i], is_user=(i == 0))
            for i in range(n_teams)
        ),
        pool=S.PlayerPool.of(rows),
        weeks=WEEKS,
        remaining_games=tuple(games),
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=4,
        playoff_rounds=ROUNDS,
        my_team_id=1,
    )
    # `panel_for` indexes on `state.pool`, so the free agents drop out here on their own
    # -- which is the whole point: the panel is the ROSTER and `outlooks` is the wire.
    draw = WeeklySampler(S.panel_for(state, outlooks), seed=seed).draw(n_sims)
    return LeagueSim(
        league=None,  # type: ignore[arg-type]
        state=state,
        draw=draw,
        outlooks=outlooks,
        n_sims=n_sims,
        seed=seed,
    )


def make_portfolio(sims, *, stream_replacement: bool = True) -> PF.Portfolio:
    """A `Portfolio` from pre-built sims, bypassing the ESPN client."""
    stakes = []
    for sim in sims:
        floors = (
            streaming_replacement(sim.state, sim.draw, outlooks=sim.outlooks)
            if stream_replacement
            else None
        )
        stakes.append(PF._stake(sim, 1, floors, None))
    return PF.Portfolio(stakes=tuple(stakes), seed=sims[0].seed, n_sims=sims[0].n_sims)


@pytest.fixture(scope="module")
def overlapping():
    """Three leagues on one NFL season: A and B share the user's whole roster, C does not.

    The shape mirrors the live portfolio -- shared players between two leagues and none
    across all three -- so every exposure count in the tests below has a hand-checkable
    answer.
    """
    return make_portfolio(
        [
            build_sim(101, seed=11, id_offset=0, my_id_offset=0),
            build_sim(102, seed=11, id_offset=200_000, my_id_offset=0),
            build_sim(103, seed=11, id_offset=400_000, my_id_offset=400_000),
        ]
    )


@pytest.fixture(scope="module")
def twins():
    """Two identical leagues on one season: the comonotone extreme."""
    return make_portfolio([build_sim(201, seed=11), build_sim(202, seed=11)])


@pytest.fixture(scope="module")
def strangers():
    """The same two leagues drawn from two different universes: the independent extreme."""
    return make_portfolio([build_sim(301, seed=11), build_sim(302, seed=99)])


# --------------------------------------------------------------------------------------
# The shared season
# --------------------------------------------------------------------------------------


class TestCoupling:
    def test_one_seed_puts_the_leagues_on_the_same_football(self, overlapping):
        checks = {(c.a, c.b): c for c in overlapping.verify_coupling()}
        shared = checks[("league 101", "league 102")]
        assert shared.shared_pool == len(POSITIONS)
        assert shared.median_player_correlation > 0.99
        assert shared.availability_match == 1.0
        assert all(c.coupled and c.same_seed for c in checks.values())

    def test_two_panels_with_nobody_in_common_have_nothing_to_measure(self, overlapping):
        """The seed is the cause; the correlations are only the evidence for it.

        Leagues 101 and 103 share no player at all, so there is no observable to check.
        Calling that "not coupled" would report a correctly built portfolio as broken.
        """
        check = next(
            c for c in overlapping.verify_coupling() if (c.a, c.b) == ("league 101", "league 103")
        )
        assert check.shared_pool == 0
        assert math.isnan(check.median_player_correlation)
        assert check.same_seed and check.coupled

    def test_two_seeds_are_two_universes_and_the_check_says_so(self, strangers):
        (check,) = strangers.verify_coupling()
        assert check.shared_pool > 0
        assert abs(check.median_player_correlation) < 0.2
        assert check.availability_match < 0.99
        assert not check.same_seed
        assert not check.coupled

    def test_identical_leagues_on_one_seed_produce_identical_seasons(self, twins):
        a, b = twins.stakes
        assert np.array_equal(a.champions, b.champions)

    def test_a_portfolio_of_mismatched_sim_counts_is_refused(self):
        small = build_sim(401, seed=11, n_sims=200)
        big = build_sim(402, seed=11, n_sims=400)
        with pytest.raises(PF.PortfolioError, match="different simulation counts"):
            make_portfolio([small, big])


# --------------------------------------------------------------------------------------
# Portfolio odds
# --------------------------------------------------------------------------------------


class TestPortfolioOdds:
    def test_p_at_least_one_is_not_the_sum_and_lies_between_max_and_sum(self, overlapping):
        odds = overlapping.odds()
        assert odds.max_bound <= odds.p_at_least_one <= odds.sum_bound
        assert odds.p_at_least_one < odds.sum_bound
        assert odds.p_at_least_one != pytest.approx(sum(odds.titles), abs=1e-6)
        assert odds.p_zero == pytest.approx(1.0 - odds.p_at_least_one)

    def test_expected_titles_is_exactly_the_sum_of_the_marginals(self, overlapping):
        """Linearity of expectation, which is what lets the action queue rank across leagues.

        This has to hold to machine precision for *any* dependence. If it ever drifts,
        the champions matrix is not aligned on one simulation axis and every joint number
        in the module is being computed across mismatched columns.
        """
        odds = overlapping.odds()
        assert odds.expected_titles == pytest.approx(sum(odds.titles), rel=1e-12)

    def test_the_frechet_extremes_are_reproduced_exactly(self):
        """Hand-built champion matrices, so the arithmetic is checked without a simulator."""
        n = 1000
        both = np.zeros((n, 2))
        both[:100, :] = 1.0  # comonotone: the same 10% of seasons win both
        odds = PF.portfolio_odds(both, bootstrap=0)
        assert odds.p_at_least_one == pytest.approx(0.1)
        assert odds.p_at_least_one == pytest.approx(odds.max_bound)

        disjoint = np.zeros((n, 2))
        disjoint[:100, 0] = 1.0
        disjoint[100:200, 1] = 1.0
        odds = PF.portfolio_odds(disjoint, bootstrap=0)
        assert odds.p_at_least_one == pytest.approx(0.2)
        assert odds.p_at_least_one == pytest.approx(odds.sum_bound)
        assert odds.p_two_plus == 0.0

    def test_independence_reproduces_one_minus_the_product(self):
        rng = np.random.default_rng(3)
        c = (rng.random((40_000, 3)) < np.array([0.05, 0.10, 0.20])).astype(float)
        odds = PF.portfolio_odds(c, bootstrap=0)
        assert odds.p_at_least_one == pytest.approx(odds.independent, abs=0.006)
        assert abs(odds.dependence_cost) < 0.006

    def test_a_ragged_matrix_is_refused(self):
        with pytest.raises(PF.PortfolioError, match="must be"):
            PF.portfolio_odds(np.zeros(10), bootstrap=0)
        with pytest.raises(PF.PortfolioError, match="names"):
            PF.portfolio_odds(np.zeros((10, 2)), names=("only one",), bootstrap=0)


class TestCorrelationChangesTheAnswer:
    def test_correlated_leagues_give_a_different_portfolio_number_than_independent_ones(
        self, twins, strangers
    ):
        """The claim the coupled draw exists to support, on two versions of one league.

        `twins` and `strangers` hold the *same rosters in the same leagues*; the only
        difference is whether the two seasons are the same football. So every difference
        below is dependence and nothing else.
        """
        coupled = twins.odds()
        independent = strangers.odds()
        assert coupled.titles[0] == pytest.approx(coupled.titles[1])
        # Perfectly coupled: winning both is the same event as winning either.
        assert coupled.p_at_least_one == pytest.approx(coupled.max_bound)
        assert coupled.p_at_least_one < coupled.independent
        # Decoupled: the independent formula is right, and it is a strictly better number.
        assert independent.p_at_least_one == pytest.approx(independent.independent, abs=0.03)
        assert independent.p_at_least_one > coupled.p_at_least_one

    def test_dependence_costs_more_than_it_can_be_measured_to(self, twins):
        odds = twins.odds()
        assert odds.dependence_cost > 0.0
        assert odds.dependence_significant

    def test_expected_titles_does_not_move_with_the_dependence(self, twins, strangers):
        """The half of the diversification answer that is a theorem, checked numerically."""
        assert twins.odds().expected_titles == pytest.approx(
            strangers.odds().expected_titles, abs=0.02
        )

    def test_the_title_correlation_is_far_weaker_than_the_weekly_one(self, overlapping):
        """The measured shape on the real leagues, reproduced on a synthetic overlap.

        Two teams sharing a whole roster correlate strongly week to week and much less on
        the championship, because a title is a rank statistic inside its own league. A
        portfolio tool that quoted the weekly number as the dependence would badly
        overstate it.
        """
        pair = next(
            p
            for p in overlapping.correlations().pairs
            if p.a == "league 101" and p.b == "league 102"
        )
        assert len(pair.shared_players) == len(POSITIONS)
        assert pair.weekly > 0.5
        assert pair.champion < pair.weekly

    def test_every_correlation_carries_an_error_bar_and_the_title_one_needs_it(self, overlapping):
        """Printed bare, a phi coefficient between two rare indicators reads as a finding.

        On the live leagues the Blacksburg/Type-shi title correlation is -0.022 and across
        five seeds it reads -0.022, +0.005, +0.030, +0.014, -0.029 -- it changes sign. The
        weekly figure over the same five seeds is +0.404, +0.407, +0.401, +0.404, +0.403.
        So the table has to distinguish them, and here that is checked on two leagues that
        share NOTHING: their title correlation must come back inside its own error.
        """
        corr = overlapping.correlations()
        shared = next(p for p in corr.pairs if p.a == "league 101" and p.b == "league 102")
        disjoint = next(p for p in corr.pairs if p.a == "league 102" and p.b == "league 103")

        assert shared.champion_stderr > 0.0 and disjoint.champion_stderr > 0.0
        # The weekly figure is pinned an order of magnitude tighter than the title one.
        assert shared.weekly_stderr < shared.champion_stderr / 5.0
        # Two leagues with no player in common have no title dependence to find.
        assert not disjoint.champion_significant
        assert abs(disjoint.champion) < 2.0 * disjoint.champion_stderr

        text = corr.table()
        assert "title!=0" in text and "NO" in text

    def test_the_most_correlated_week_is_not_named_when_it_cannot_be_separated(self, overlapping):
        """Ranking seventeen noisy weeks and printing the argmax is reading a coin flip.

        On the live leagues the argmax is week 11 on four seeds and week 9 on the fifth,
        and the two sit within 0.01 of each other against a spread of 0.034. The synthetic
        weeks here are exchangeable by construction -- every player projects the same mean
        every week -- so there is no true worst week at all and the module must not name
        one.
        """
        corr = overlapping.correlations()
        assert corr.worst_week_gap_stderr > 0.0
        assert not corr.worst_week_separable
        assert "NOT separable" in corr.table()

        # The property still answers, for a caller who wants the argmax anyway.
        week, value = corr.worst_week
        assert week in corr.by_week and corr.by_week[week] == value

    def test_a_planted_week_is_separable_so_the_check_is_not_vacuous(self, overlapping):
        """The negative control above needs a positive one, or it only tests a constant."""
        corr = overlapping.correlations()
        spiked = PF.Correlations(
            pairs=corr.pairs,
            by_week=dict.fromkeys(corr.by_week, 0.10) | {next(iter(corr.by_week)): 0.90},
            worst_week_gap_stderr=corr.worst_week_gap_stderr,
        )
        assert spiked.worst_week_separable
        assert "NOT separable" not in spiked.table()


# --------------------------------------------------------------------------------------
# Exposure
# --------------------------------------------------------------------------------------


class TestExposure:
    def test_the_counts_are_right_across_the_three_rosters(self, overlapping):
        a, b, c = overlapping.stakes
        every = overlapping.exposures(min_leagues=1)
        by_id = {e.player_id: e for e in every}

        union = set(a.roster) | set(b.roster) | set(c.roster)
        assert set(by_id) == union
        assert len(every) == len(union)

        shared = set(a.roster) & set(b.roster)
        assert shared == set(a.roster)  # A and B hold the identical user roster
        for pid in shared:
            assert by_id[pid].n_leagues == 2
            assert {h.league_id for h in by_id[pid].holdings} == {101, 102}
        for pid in c.roster:
            assert by_id[pid].n_leagues == 1

        two_up = overlapping.exposures(min_leagues=2)
        assert {e.player_id for e in two_up} == shared
        assert all(e.n_leagues == 2 for e in two_up)

    def test_a_player_is_credited_to_the_slot_he_actually_starts_in(self, overlapping):
        stake = overlapping.stakes[0]
        starters = stake.roster[:9]
        bench = stake.roster[9:]
        assert stake.slot_of(starters[0])[0] == 0  # the only QB fills the QB slot
        assert all(stake.slot_of(p)[1] > 0.5 for p in starters)
        # The bench here is a strictly worse copy of the starters, so it never plays.
        assert all(stake.slot_of(p) == (PF.SLOT_BENCH, 0.0) for p in bench)

    def test_a_bench_body_carries_no_equity_and_a_starter_does(self, overlapping):
        by_id = {e.player_id: e for e in overlapping.exposures(min_leagues=1)}
        stake = overlapping.stakes[0]
        best = by_id[stake.roster[0]]
        worst = by_id[stake.roster[-1]]
        assert best.equity_at_risk > 0.01
        assert best.n_starting == 2
        assert abs(worst.equity_at_risk) < 0.005
        assert worst.n_starting == 0

    def test_equity_share_is_denominated_in_expected_titles(self, overlapping):
        odds = overlapping.odds()
        for e in overlapping.exposures(min_leagues=1):
            assert e.equity_share == pytest.approx(e.equity_at_risk / odds.expected_titles)

    def test_portfolio_damage_tracks_the_per_league_losses_without_being_bounded_by_them(
        self, overlapping
    ):
        """`portfolio_damage` and `equity_at_risk` measure the same loss two ways.

        This test used to assert the union bound -- `damage <= equity`, called
        "arithmetically impossible" to violate -- and the live portfolio violates it for
        eleven of forty players. The bound needs the post-removal win set to be a subset
        of the pre-removal one and it is not: `champions_without` re-stands the whole
        league, so a removal can hand the user a title he would not otherwise have won,
        and when that lands in a season he was already winning elsewhere the union does
        not move while the per-league term goes negative. See `Exposure`.

        What is actually true is the weaker statement, which is what is checked here: the
        two agree to within the Monte Carlo error of the difference, and the union number
        is never LARGER by more than that -- a real axis misalignment shows up as a gross
        disagreement, not as a two-simulation excess.
        """
        checked = 0
        for e in overlapping.exposures(min_leagues=1):
            error = math.hypot(
                e.portfolio_damage_stderr, sum(h.title_added_stderr for h in e.holdings)
            )
            assert e.portfolio_damage <= e.equity_at_risk + 4.0 * error + 1e-9
            assert e.portfolio_damage >= -1.0 and e.equity_at_risk >= -1.0
            checked += 1
        assert checked > 10

    def test_the_union_can_lose_less_than_the_leagues_do_because_a_title_moves(self, overlapping):
        """The mechanism behind the line above, on the champion matrix itself.

        Deleting a player is not a monotone operation on the win set, so `damage` and
        `equity` genuinely can straddle each other. Demonstrated arithmetically rather
        than hoped for: a season the user wins in league B, loses in A before the
        removal and wins in A after it, contributes -1 to `equity` and 0 to `damage`.
        """
        # Simulation 0: the user wins league B either way, loses A before the removal and
        # wins it after. Simulation 1: nothing moves.
        before = np.array([[0.0, 1.0], [1.0, 0.0]])
        after = np.array([[1.0, 1.0], [1.0, 0.0]])
        equity = float((before - after).sum(axis=1).mean())
        damage = float(
            ((before.sum(axis=1) > 0).astype(float) - (after.sum(axis=1) > 0).astype(float)).mean()
        )
        assert equity == pytest.approx(-0.5)
        assert damage == pytest.approx(0.0)
        assert damage > equity  # the bound the old test asserted, violated by construction


# --------------------------------------------------------------------------------------
# Concentration
# --------------------------------------------------------------------------------------


class TestConcentration:
    def test_the_damage_is_a_re_simulation_not_a_heuristic(self, overlapping):
        """Two independent proofs that the number came out of the season model.

        First, `champions_without` differs from the baseline *per simulation*, which no
        closed-form discount of a projection could produce. Second, the per-league loss it
        implies agrees with `sim/season.leave_one_out` -- a completely separate
        implementation of the same counterfactual -- to well inside the paired error.
        """
        stake = overlapping.stakes[0]
        # One of three running backs, so removing him empties no position group and the
        # two implementations are answering the identical question. See
        # `TestEmptyPositionFloorTrap` for what happens when it does empty one.
        victim = stake.roster[1]
        after = stake.champions_without([victim])
        paired = stake.champions - after
        assert not np.array_equal(after, stake.champions)
        assert set(np.unique(after)) <= {0.0, 1.0}
        assert paired.std(ddof=1) > 0.0

        (contribution,) = S.leave_one_out(
            stake.state,
            stake.draw,
            player_ids=[victim],
            all_play=False,
            replacement=stake.replacement,
        )
        error = math.hypot(
            float(paired.std(ddof=1) / math.sqrt(paired.size)),
            contribution.title_added_stderr,
        )
        assert float(paired.mean()) == pytest.approx(contribution.title_added, abs=4.0 * error)

    def test_removing_a_starter_hurts_and_removing_a_never_used_bench_body_does_not(
        self, overlapping
    ):
        by_id = {e.player_id: e for e in overlapping.exposures(min_leagues=1)}
        stake = overlapping.stakes[0]
        starter = PF.player_concentration(overlapping, by_id[stake.roster[0]])
        bench = PF.player_concentration(overlapping, by_id[stake.roster[-1]])
        assert starter.after < starter.before
        assert starter.damage > 0.01
        assert starter.significant
        assert abs(bench.damage) < 0.005

    def test_the_damage_moves_the_expected_title_count_too(self, overlapping):
        by_id = {e.player_id: e for e in overlapping.exposures(min_leagues=1)}
        stake = overlapping.stakes[0]
        c = PF.player_concentration(overlapping, by_id[stake.roster[0]])
        assert c.expected_after < c.expected_before
        assert c.removed  # the row names what it deleted, per league

    def test_an_nfl_team_group_removes_every_league_at_once(self, overlapping):
        rows = PF.pro_team_concentration(overlapping, limit=3)
        assert rows
        assert all(r.kind == "NFL team" for r in rows)
        assert rows == tuple(sorted(rows, reverse=True))
        # Every returned group has to touch more than one roster spot, or it is just a
        # player row wearing a team label.
        assert all(r.n_removed > 1 for r in rows)

    def test_a_group_row_is_attributed_to_its_driver_so_it_cannot_beat_a_player_by_size(
        self, overlapping
    ):
        """A group deletion is a superset of a player deletion, so the group has to win.

        "The NFL-team number beats any single player" was the module's headline finding
        and it is arithmetic: on the live portfolio KC's -5.92pp is Kenneth Walker III,
        held in two leagues, at -4.93pp, plus a kicker in the third. The row now names its
        driver and prices the remainder as a paired per-simulation difference, and the
        remainder is what a reader is entitled to call a concentration.
        """
        rows = PF.pro_team_concentration(overlapping, limit=4)
        for row in rows:
            assert row.driver, row.label
            assert row.driver in {n for names in row.removed.values() for n in names}
            # The group cannot be beaten by a strict subset of itself by more than the
            # error on the difference between them.
            assert row.driver_damage <= row.damage + 4.0 * max(row.increment_stderr, 1e-9) + 1e-9
            assert row.increment == pytest.approx(row.damage - row.driver_damage, abs=1e-9)
            assert "roster spots" in row.describe()
            assert ("NOT separable from its driver" in row.describe()) == (
                not row.increment_significant
            )

    def test_a_group_that_is_one_player_twice_adds_nothing_over_him(self, overlapping):
        """A and B hold the identical roster, so an NFL team with one player on it there
        is that player taken twice -- and the increment over him must be exactly zero,
        not a second helping of his value."""
        stake = overlapping.stakes[0]
        pool = stake.state.pool
        counts: dict[int, set[int]] = {}
        for pid in stake.roster:
            counts.setdefault(int(pool.pro_team_ids[pool.index[pid]]), set()).add(int(pid))
        solo = {team for team, ids in counts.items() if len(ids) == 1}
        rows = [r for r in PF.pro_team_concentration(overlapping, limit=30) if r.n_removed == 2]
        assert rows, "the fixture should produce at least one two-league single-player group"
        singles = [r for r in rows if len({n for v in r.removed.values() for n in v}) == 1]
        assert singles and solo
        for row in singles:
            assert row.increment == pytest.approx(0.0, abs=1e-12)
            assert not row.increment_significant

    def test_a_player_nobody_rosters_changes_nothing(self, overlapping):
        stake = overlapping.stakes[0]
        assert np.array_equal(stake.champions_without([-999]), stake.champions)


@pytest.fixture(scope="module")
def wired():
    """One league that has a wire: free agents in `outlooks`, absent from `state.pool`."""
    return build_sim(701, seed=5, wire_strength=0.55)


class TestTheFloorIsFittedOffTheWireAndNotThePanel:
    """`build_portfolio` must hand `streaming_replacement` the wire, not the roster.

    `decide/title.streaming_levels` warns this caller by name: without `outlooks=` the
    pool comes from the panel, `pipeline.build` pools only ROSTERED players, the wire
    reads 0.00 at every slot and the whole board silently falls through to the VOLS
    roster-bottom rank. Measured on the three live leagues that put RB at 9.22 against a
    true 4.55 and D/ST at 5.29 against a true 7.39 -- wrong in both directions, so it does
    not cancel -- and reordered 39 of 40 rows of `exposures`.

    The old fixture could not see any of it, because every synthetic player was on a
    roster and so the panel and the wire were the same set. `build_sim(wire_strength=...)`
    is what makes the two distinguishable; these tests are worthless without it.
    """

    def test_the_fixture_really_has_a_wire(self, wired):
        """Guards the guard: free agents in `outlooks`, absent from `state.pool`."""
        pool = set(wired.state.pool.player_ids)
        free = [o for o in wired.outlooks if o.player_id not in pool]
        assert free, "no free agents, so neither test below can fail"
        assert all(o.player_id not in pool for o in free)

    def test_the_panel_and_the_wire_give_different_floors(self, wired):
        panel = streaming_replacement(wired.state, wired.draw)
        wire = streaming_replacement(wired.state, wired.draw, outlooks=wired.outlooks)
        assert set(panel) == set(wire)
        # The wire here is deliberately weaker than the roster bottom, which is the live
        # direction at RB/WR/QB/FLEX. The point is that they DISAGREE, not the sign.
        assert panel != wire
        assert any(wire[s] < panel[s] for s in panel)

    def test_build_portfolio_fits_the_floor_off_the_wire(self, wired, monkeypatch):
        """The regression guard on `portfolio.py`'s own call, not on a helper's.

        Driven through `build_portfolio` rather than `make_portfolio` because the helper
        is the thing that mirrored the bug: a test that only exercised the helper would
        have passed both before and after the fix.
        """
        monkeypatch.setattr(PF, "build", lambda *a, **kw: wired)
        # A non-None client keeps `build_portfolio` from reaching for real credentials,
        # and `_submitted_lineup` swallows the AttributeError off `league=None`.
        got = PF.build_portfolio([(701, 1)], 2026, client=object(), n_sims=wired.n_sims)
        assert got.stakes[0].replacement == streaming_replacement(
            wired.state, wired.draw, outlooks=wired.outlooks
        )
        assert got.stakes[0].replacement != streaming_replacement(wired.state, wired.draw)

    def test_turning_the_floor_off_still_means_off(self, wired, monkeypatch):
        """The negative control: `stream_replacement=False` is untouched by any of this."""
        monkeypatch.setattr(PF, "build", lambda *a, **kw: wired)
        got = PF.build_portfolio(
            [(701, 1)], 2026, client=object(), n_sims=wired.n_sims, stream_replacement=False
        )
        assert got.stakes[0].replacement is None


class TestEmptyPositionFloorTrap:
    """Dropping a roster's only quarterback must make it worse, not invincible.

    `lineup.monotone_floor` lifts every slot to the floor of any slot whose eligible set
    it contains; a roster with nobody at a position has an *empty* eligible set, which is
    contained in every other. `decide/title.py` measured what that does -- a 137-point-a-
    week team with zero variance at 93% title odds. Every one of the user's three real
    rosters carries exactly one quarterback, one kicker and one defence, so this fires on
    the first interesting exposure rather than on a contrived one.

    The guard used to be reimplemented here as `_floors_for`. It now lives in
    `sim/season._floors`, so these tests reach for it there -- but they stay in this file
    as well as in `test_season.py`, because this module is where the empty position is
    routine rather than exceptional.
    """

    def test_the_guard_zeroes_the_empty_group_and_hands_back_its_points(self, overlapping):
        stake = overlapping.stakes[0]
        state = stake.state
        franchise = state.franchise(1)
        without_qb = franchise.with_players(
            p for p in franchise.player_ids if state.pool.positions_of([p])[0] != 1
        )
        plan = plan_from_slots(
            state.lineup_slot_counts,
            state.slot_eligibility,
            state.pool.positions_of(without_qb.player_ids),
        )
        groups, per_slot, _credit, omitted = S._floors(plan, stake.replacement)
        assert omitted > 0.0
        # The QB group lifts nothing any more, and nothing else was flattened with it.
        qb_group = plan.floor_slot_ids.index(0)
        assert groups[qb_group] == 0.0
        assert any(float(v) > 0.0 for i, v in enumerate(groups) if i != qb_group)
        assert per_slot[[i for i, s in enumerate(plan.slot_ids) if s == 0]].tolist() == [0.0]

    def test_a_scalar_or_absent_replacement_needs_no_guard(self, overlapping):
        stake = overlapping.stakes[0]
        plan = plan_from_slots(
            stake.state.lineup_slot_counts,
            stake.state.slot_eligibility,
            stake.state.pool.positions_of(stake.roster),
        )
        groups, per_slot, credit, omitted = S._floors(plan, None)
        assert (groups, credit, omitted) == (None, None, 0.0)
        assert not per_slot.any()

        groups, per_slot, credit, omitted = S._floors(plan, 3.0)
        assert credit is None and omitted == 0.0
        # A scalar is uniform, so the lift is a no-op and every slot sits at the number.
        assert np.allclose(np.asarray(groups), 3.0)
        assert np.allclose(per_slot, 3.0)

    def test_dropping_the_only_quarterback_lowers_the_title_odds(self, overlapping):
        stake = overlapping.stakes[0]
        qb = next(p for p in stake.roster if stake.state.pool.positions_of([p])[0] == 1)
        after = stake.champions_without([qb])
        assert float(after.mean()) < stake.title
        # The failure this guards against is not a small overstatement: unguarded, the
        # team scores its replacement level in every slot with no variance and runs away
        # with the league.
        assert float(after.mean()) < 0.5

    def test_season_leave_one_out_now_agrees_with_this_module_about_the_quarterback(
        self, overlapping
    ):
        """The predecessor of this test was written to fail once `leave_one_out` was fixed.

        It did exactly that. `sim/season.leave_one_out` used to forward a Mapping
        `replacement` straight into `_floors` with no empty-group handling, so on this
        roster -- one quarterback, as on all three of the user's real teams -- it reported
        that **deleting the quarterback RAISED the title probability by more than half**
        (`title_added` -0.5075) while this module's own counterfactual, over the same
        state and the same draw, had him worth a gain. Two answers of opposite sign for
        one question was the whole reason `_floors_for` existed here.

        The guard now lives in `sim/season._floors`, so both paths route through it and
        the disagreement is gone. This asserts they agree, which is the property the two
        copies were maintaining by hand.
        """
        stake = overlapping.stakes[0]
        qb = next(p for p in stake.roster if stake.state.pool.positions_of([p])[0] == 1)
        (upstream,) = S.leave_one_out(
            stake.state,
            stake.draw,
            player_ids=[qb],
            all_play=False,
            replacement=stake.replacement,
        )
        guarded = float((stake.champions - stake.champions_without([qb])).mean())
        assert upstream.title_added > 0.0, "the quarterback must be worth having"
        assert guarded > 0.0
        # Not identical -- `leave_one_out` re-stands the league from the draw while
        # `champions_without` re-scores only the touched franchises -- but they must not
        # disagree about the sign, which is what the unguarded path did.
        assert abs(upstream.title_added - guarded) < 0.1


# --------------------------------------------------------------------------------------
# Byes
# --------------------------------------------------------------------------------------


class TestByeExposure:
    def test_a_bye_the_projections_zeroed_is_not_flagged_and_one_they_did_not_is(self, overlapping):
        """The live gap is D/ST, where ESPN does not zero the bye. Reproduced by handing
        the same portfolio two bye tables: one on a week the synthetic projections do not
        zero at all, which every starter must be flagged for."""
        stake = overlapping.stakes[0]
        pool = stake.state.pool
        team = int(pool.pro_team_ids[pool.index[stake.roster[0]]])
        rows = PF.bye_concentration(overlapping, byes={team: 3}, limit=5)
        assert rows
        week3 = next(r for r in rows if r.week == 3)
        # These synthetic outlooks project the same mean every week, so NOTHING about
        # week 3 is priced as a bye and every affected starter has to say so.
        assert week3.unpriced
        assert all(bye == pytest.approx(mean, rel=0.05) for _, _, bye, mean in week3.unpriced)

        # A tolerance above 1.0 can never be exceeded, so nothing is flagged and the
        # exposure is still reported -- the two are separate questions.
        relaxed = PF.bye_concentration(overlapping, byes={team: 3}, limit=5, tolerance=2.0)
        assert not next(r for r in relaxed if r.week == 3).unpriced
        assert next(r for r in relaxed if r.week == 3).total_out > 0

    def test_a_bye_table_covering_nobody_produces_nothing(self, overlapping):
        assert PF.bye_concentration(overlapping, byes={}) == ()

    def test_the_cross_league_count_adds_up(self, overlapping):
        stake = overlapping.stakes[0]
        pool = stake.state.pool
        team = int(pool.pro_team_ids[pool.index[stake.roster[0]]])
        (row,) = PF.bye_concentration(overlapping, byes={team: 3}, limit=1)
        assert row.total_out == sum(len(v) for v in row.starters_out.values())
        assert row.leagues_hit == len(row.starters_out)
        assert set(row.normal_points) == set(row.starters_out)


# --------------------------------------------------------------------------------------
# Diversification
# --------------------------------------------------------------------------------------


class TestDiversification:
    def test_the_two_objectives_disagree_and_the_module_says_by_how_much(self, twins):
        d = PF.diversification(twins)
        assert d.expected_titles_invariant
        # Concentration costs P(>=1) and leaves E[N] untouched. That is the disagreement.
        assert d.p_at_least_one < d.independent
        assert d.expected_titles == pytest.approx(d.marginal_sum, rel=1e-12)
        assert d.concentration_headroom == pytest.approx(0.0, abs=1e-9)
        assert d.diversification_headroom > 0.0
        assert "E[titles]" in d.verdict and "P(>=1 title)" in d.verdict

    def test_the_verdict_reads_the_portfolios_position_instead_of_asserting_one(
        self, twins, strangers
    ):
        """The closing clause has to move with the numbers, and it used not to.

        `verdict` ended "the position is already near the diversified end" unconditionally.
        That is true of the user's live portfolio and false of `twins`, where the same
        sentence claimed a perfectly comonotone pair of leagues was near the diversified
        end while printing `0.00pp` of concentration headroom immediately in front of it.
        Asserting on the presence of the substrings "E[titles]" and "P(>=1 title)" -- which
        is all the test above does -- could never have caught that, because both are
        literals in the same f-string.
        """
        concentrated = PF.diversification(twins)
        assert concentrated.concentration_headroom == pytest.approx(0.0, abs=1e-9)
        assert "CONCENTRATED end" in concentrated.verdict
        assert "near the diversified end" not in concentrated.verdict

        middle = PF.diversification(strangers)
        assert middle.concentration_headroom > 0.0 and middle.diversification_headroom > 0.0
        assert "CONCENTRATED end" not in middle.verdict

        # And the diversified end, on a hand-built portfolio where the two leagues never
        # win together, so P(>=1) sits exactly on the Frechet upper bound.
        disjoint = np.zeros((1000, 2))
        disjoint[:100, 0] = 1.0
        disjoint[100:200, 1] = 1.0
        diversified = PF.Diversification(
            p_at_least_one=0.2,
            independent=0.19,
            sum_bound=0.2,
            max_bound=0.1,
            expected_titles=0.2,
            marginal_sum=0.2,
            concentration_headroom=0.1,
            diversification_headroom=0.0,
            variance_titles=0.16,
            variance_independent=0.18,
            dependence_cost=-0.01,
            dependence_cost_stderr=0.02,
        )
        assert "near the diversified end" in diversified.verdict

    def test_concentration_raises_the_variance_of_the_title_count(self, twins, strangers):
        """The other half of "higher variance of the portfolio outcome", measured."""
        coupled = PF.diversification(twins)
        independent = PF.diversification(strangers)
        assert coupled.variance_titles > coupled.variance_independent
        assert independent.variance_titles == pytest.approx(
            independent.variance_independent, rel=0.15
        )

    def test_a_decoupled_portfolio_has_no_headroom_left_to_recover(self, strangers):
        d = PF.diversification(strangers)
        assert abs(d.dependence_cost) < 0.02
        assert d.concentration_headroom > 0.0


# --------------------------------------------------------------------------------------
# The action queue
# --------------------------------------------------------------------------------------


def _rec(delta: float, *, stderr: float = 0.0, leverage: float = 1.0, tags=(), kind=None):
    return Recommendation(
        move=Move(
            kind=kind or MoveKind.ADD_DROP,
            league_id=1,
            players=(PlayerMove(player_id=1, from_team=None, to_team=1),),
        ),
        delta_title=delta,
        delta_points=10.0 * delta,
        stderr=stderr,
        leverage=leverage,
        rationale="synthetic",
        tags=tuple(tags),
    )


class TestActionQueue:
    def test_the_queue_is_ordered_by_delta_title_across_leagues(self, overlapping):
        deltas = {101: (0.004, 0.001), 102: (0.010, 0.002), 103: (0.007, 0.003)}
        queue = PF.action_queue(
            overlapping,
            limit=10,
            surfaces={"fake": lambda s: [_rec(d) for d in deltas[s.league_id]]},
        )
        assert len(queue.items) == 6
        assert [round(i.delta_title, 6) for i in queue.items] == sorted(
            [round(d, 6) for pair in deltas.values() for d in pair], reverse=True
        )
        assert queue.items[0].league_id == 102
        assert queue.items[0].surface == "fake"

    def test_an_effect_inside_its_own_error_is_marked_not_significant(self, overlapping):
        queue = PF.action_queue(
            overlapping,
            surfaces={
                "fake": lambda s: [
                    _rec(0.020, stderr=0.002),  # 10 sigma
                    _rec(0.010, stderr=0.009),  # 1.1 sigma
                    _rec(0.000, stderr=0.000),  # nothing measured at all
                ]
            },
        )
        by_delta = {round(i.delta_title, 4): i for i in queue.items}
        assert by_delta[0.02].significant
        assert not by_delta[0.01].significant
        # `core.Recommendation.significant` calls a zero effect with zero error certain;
        # the queue must not, because both arms were bit-identical and nothing was tested.
        assert by_delta[0.0].rec.significant
        assert not by_delta[0.0].significant
        assert len(queue.significant) == 3  # one per league

    def test_a_plan_whose_first_week_is_a_hold_is_not_an_action(self, overlapping):
        queue = PF.action_queue(
            overlapping,
            surfaces={
                "fake": lambda s: [
                    _rec(0.040, stderr=0.004, tags=("streaming", "no-action-this-week")),
                    _rec(0.005, stderr=0.001),
                ]
            },
        )
        top = queue.items[0]
        assert top.delta_title == pytest.approx(0.04)
        assert not top.actionable
        assert "no-action-this-week" in top.blockers
        assert all(i.delta_title == pytest.approx(0.005) for i in queue.actionable)

        only = PF.action_queue(
            overlapping,
            actionable_only=True,
            surfaces={
                "fake": lambda s: [
                    _rec(0.040, stderr=0.004, tags=("streaming", "no-action-this-week")),
                    _rec(0.005, stderr=0.001),
                ]
            },
        )
        assert all(i.actionable for i in only.items)
        assert len(only.items) == 3

    def test_a_bare_hold_is_flagged_even_without_a_tag(self, overlapping):
        queue = PF.action_queue(
            overlapping, surfaces={"fake": lambda s: [_rec(0.01, kind=MoveKind.HOLD)]}
        )
        assert not queue.items[0].actionable
        assert "hold" in queue.items[0].blockers

    def test_the_risk_adjusted_order_is_available_and_differs(self, overlapping):
        queue = PF.action_queue(
            overlapping,
            surfaces={"fake": lambda s: [_rec(0.020, stderr=0.009), _rec(0.015, stderr=0.001)]},
        )
        assert queue.items[0].delta_title == pytest.approx(0.020)
        assert queue.by_lower_bound[0].delta_title == pytest.approx(0.015)

    def test_leverage_weighting_is_off_by_default_and_reachable(self, overlapping):
        surfaces = {
            "fake": lambda s: [
                _rec(0.010, leverage=0.10),
                _rec(0.008, leverage=1.00),
            ]
        }
        plain = PF.action_queue(overlapping, surfaces=surfaces)
        weighted = PF.action_queue(overlapping, surfaces=surfaces, leverage_weight=1.0)
        assert plain.items[0].delta_title == pytest.approx(0.010)
        assert weighted.items[0].delta_title == pytest.approx(0.008)
        assert plain.reordered_by_leverage > 0

    def test_one_surface_per_league_is_capped_so_it_cannot_crowd_the_queue_out(self, overlapping):
        queue = PF.action_queue(
            overlapping,
            limit=50,
            per_surface=2,
            surfaces={"noisy": lambda s: [_rec(0.01 - 0.001 * k) for k in range(10)]},
        )
        assert len(queue.items) == 6  # two per league, not ten

    def test_a_failing_surface_is_recorded_and_the_rest_of_the_queue_survives(self, overlapping):
        def broken(stake):
            raise RuntimeError("no wire data")

        queue = PF.action_queue(
            overlapping,
            surfaces={"broken": broken, "fine": lambda s: [_rec(0.01)]},
        )
        assert len(queue.items) == 3
        assert len(queue.failures) == 3
        assert all(f[1] == "broken" and "no wire data" in f[2] for f in queue.failures)
        assert "FAILED" in queue.table()

    def test_the_start_sit_surface_is_priced_against_the_submitted_lineup(self, overlapping):
        """Without `current`, `decide/lineups` compares its answer to the lineup it just
        built and returns 0.00pp forever. The portfolio captures the real one at build
        time; here it is captured by hand and made deliberately wrong."""
        stake = overlapping.stakes[0]
        optimal = [p for p in stake.roster if stake.slot_of(p)[1] > 0.0]
        bench = [p for p in stake.roster if stake.slot_of(p)[1] == 0.0]
        assert optimal and bench

        (idle,) = PF._lineup_recs(stake)
        assert idle.delta_title == 0.0

        # Swap the flex starter for a bench body of the same position, which is a legal
        # lineup and a worse one.
        pos = stake.state.pool.positions_of
        swap_out = next(p for p in reversed(optimal) if pos([p])[0] in (2, 3, 4))
        swap_in = next(p for p in bench if pos([p])[0] == pos([swap_out])[0])
        wrong = tuple(swap_in if p == swap_out else p for p in optimal)
        (rec,) = PF._lineup_recs(
            PF.LeagueStake(
                sim=stake.sim,
                team_id=stake.team_id,
                team_name=stake.team_name,
                replacement=stake.replacement,
                champions=stake.champions,
                weekly=stake.weekly,
                scores=stake.scores,
                factors=stake.factors,
                starting=stake.starting,
                current_starters=wrong,
            )
        )
        # The surface now has a real baseline to beat: it finds the swap and prices the
        # points. It is asserted on `delta_points`, not `delta_title`, because a
        # single week's lineup is worth about two projected points and the title effect
        # of that is below the Monte Carlo resolution -- which is what `LineupAdvice`
        # says about itself and what the live queue shows.
        assert rec.delta_points > 1.0
        assert rec.delta_title != 0.0
        assert rec.move.kind is MoveKind.LINEUP

    def test_the_table_renders(self, overlapping):
        queue = PF.action_queue(
            overlapping, surfaces={"fake": lambda s: [_rec(0.01, stderr=0.001)]}
        )
        text = queue.table()
        assert "dTitle" in text and "league 101" in text


class TestSelectionAdjustedSignificance:
    """The winner of a forty-candidate search does not get a two-sigma test.

    This is the defect the queue shipped with, measured on the user's own leagues on
    2026-09-07. The top two *actionable* rows on the whole board were Blacksburg trades
    at +1.225pp +/- 0.477 and +1.125pp +/- 0.476 -- 2.6 and 2.4 sigma, marked
    significant. Both were the argmax of forty confirmed packages, where
    `decide/trades.selection_threshold(40)` is 3.23 and the bar is 1.54pp. Neither
    cleared it, `decide/trades` had already stamped both `confidence="low"`, and the
    nine waiver claims they outranked by a factor of seven clear their own threshold
    five times over.
    """

    def test_the_field_size_reaches_the_item_and_raises_the_bar(self, overlapping):
        one = PF.action_queue(overlapping, surfaces={"fake": lambda s: [_rec(0.012, stderr=0.005)]})
        assert one.items[0].n_considered == 1
        assert one.items[0].selection_z == pytest.approx(1.96, abs=0.01)
        assert one.items[0].significant  # 2.4 sigma against a single-candidate test

        many = PF.action_queue(
            overlapping,
            per_surface=1,
            surfaces={
                "fake": lambda s: PF.Candidates((_rec(0.012, stderr=0.005),), n_considered=40)
            },
        )
        item = many.items[0]
        assert item.n_considered == 40
        assert item.selection_z == pytest.approx(3.227, abs=0.005)
        assert item.naive_significant  # the old answer, kept for comparison
        assert not item.significant  # 0.012 < 3.227 * 0.005 = 0.0161
        assert item.selection_lower_bound < 0.0 < item.lower_bound

    def test_the_live_shape_puts_the_small_precise_surface_above_the_big_noisy_one(
        self, overlapping
    ):
        """A waiver claim and a trade at the sizes and errors the live board produced."""

        def surfaces(stake):
            return {
                "waivers": lambda s: PF.Candidates((_rec(0.00182, stderr=0.00013),), 12),
                "trades": lambda s: PF.Candidates((_rec(0.01225, stderr=0.00477),), 40),
            }

        queue = PF.action_queue(overlapping, surfaces=surfaces(None), per_surface=1)
        trade = next(i for i in queue.items if i.surface == "trades")
        waiver = next(i for i in queue.items if i.surface == "waivers")

        # Raw delta puts the trade seven times above the claim...
        assert queue.items[0].surface == "trades"
        assert trade.delta_title > 6 * waiver.delta_title
        # ...and it is the claim, not the trade, that is actually established.
        assert trade.naive_significant and not trade.significant
        assert waiver.significant
        assert queue.by_selection_bound[0].surface == "waivers"
        # And the two-sigma lower bound is NOT enough on its own: at these sizes it still
        # puts the trade first, because 2 * 0.477pp does not price a field of forty. That
        # is the whole reason `by_selection_bound` exists next to `by_lower_bound`.
        assert queue.by_lower_bound[0].surface == "trades"
        assert trade.lower_bound > waiver.lower_bound
        assert trade.selection_lower_bound < waiver.selection_lower_bound
        assert trade.selection_lower_bound < 0.0 < waiver.selection_lower_bound
        assert "trades" in queue.selection_note() and "waivers" in queue.selection_note()
        assert "z" in queue.table()

    def test_the_corrected_order_re_ranks_the_whole_board_not_just_the_visible_fold(
        self, overlapping
    ):
        """A promotion from below the fold is the only thing the corrected key is for.

        On the live board the nine waiver claims sit at raw ranks 16 to 24, so a
        re-ranking confined to `items` -- the top ten the raw key already picked -- would
        have had nothing to promote and would have handed the raw order back wearing a
        corrected label. Here one precise row is planted below a wall of noisy ones.
        """
        surfaces = {
            "noisy": lambda s: PF.Candidates(
                tuple(_rec(0.020 - 0.001 * k, stderr=0.009) for k in range(3)), 40
            ),
            "precise": lambda s: PF.Candidates((_rec(0.004, stderr=0.0002),), 6),
        }
        queue = PF.action_queue(overlapping, limit=3, per_surface=3, surfaces=surfaces)
        assert len(queue.items) == 3
        assert len(queue.ranked) == 12
        assert all(i.surface == "noisy" for i in queue.items)  # the precise row is cut
        assert queue.by_selection_bound[0].surface == "precise"
        assert len(queue.by_selection_bound) == 3

    def test_a_bare_list_surface_still_works_and_counts_itself(self, overlapping):
        queue = PF.action_queue(
            overlapping,
            per_surface=5,
            surfaces={"fake": lambda s: [_rec(0.01 - 0.001 * k) for k in range(5)]},
        )
        assert all(i.n_considered == 5 for i in queue.items)

    def test_the_trade_surface_counts_the_whole_confirmed_field_not_its_positive_tail(
        self, overlapping
    ):
        """`find_trades` drops the packages that lower the user's title probability.

        All of them competed for the top of the ranking, so all of them are the
        multiplicity: on the live leagues the returned list is eight, twenty-two and
        twenty-nine rows out of forty confirmed candidates each time, and taking the
        multiplicity from the returned list would understate the threshold by a factor of
        five on Wine Wednesday. The waiver surface makes the same distinction between
        `report.board` and `report.claims`; it needs a wire to run, so it is checked on
        the live leagues instead.
        """
        from fantasy_quant.decide.trades import find_trades

        stake = overlapping.stakes[0]
        result = PF._trade_recs(stake)
        assert isinstance(result, PF.Candidates)
        every = find_trades(stake.sim, for_team=stake.team_id, include_harmful=True)
        positive = [r for r in every if r.delta_title > 0.0]
        assert result.n_considered == max(len(every), 1)
        assert len(result.recommended) == len(positive)
        assert result.n_considered >= len(result.recommended)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


class TestReport:
    def test_the_whole_report_assembles_and_renders(self, overlapping):
        queue = PF.action_queue(
            overlapping, surfaces={"fake": lambda s: [_rec(0.01, stderr=0.001)]}
        )
        r = PF.report(overlapping, queue=queue, top_exposures=2, min_leagues=2)
        text = r.text()
        for heading in ("PORTFOLIO ODDS", "COUPLING", "CORRELATION", "EXPOSURE", "ACTION QUEUE"):
            assert heading in text
        assert r.odds.p_at_least_one == overlapping.odds().p_at_least_one
        assert len(r.concentration) == 2

    def test_an_unknown_league_is_refused(self, overlapping):
        with pytest.raises(PF.PortfolioError, match="not in this portfolio"):
            overlapping.stake(999)

    def test_an_empty_portfolio_is_refused(self):
        with pytest.raises(PF.PortfolioError, match="at least one league"):
            PF.Portfolio(stakes=(), seed=1, n_sims=10)


# --------------------------------------------------------------------------------------
# The live portfolio
# --------------------------------------------------------------------------------------

LIVE = [(272150391, 1), (161496047, 1), (634537479, 2)]


@pytest.mark.network
class TestLive:
    def test_the_three_real_leagues_share_one_season_and_the_odds_are_bounded(self):
        portfolio = PF.build_portfolio(LIVE, 2026, n_sims=2000)
        for check in portfolio.verify_coupling():
            assert check.coupled, check

        odds = portfolio.odds()
        assert len(odds.titles) == 3
        assert odds.max_bound <= odds.p_at_least_one <= odds.sum_bound
        assert odds.p_at_least_one < sum(odds.titles)
        assert odds.expected_titles == pytest.approx(sum(odds.titles), rel=1e-12)

        shared = portfolio.exposures(min_leagues=2)
        assert shared, "the user really does hold players in more than one league"
        assert all(e.n_leagues >= 2 for e in shared)
        assert max(e.portfolio_damage for e in shared) > 0.01

        # The waiver board is a bigger field than the claims it returns, and the trade
        # finder a bigger one than its positive tail. Both are what the threshold is
        # Bonferroni over, and neither is available offline.
        stake = portfolio.stakes[0]
        for surface in (PF._waiver_recs, PF._trade_recs):
            result = surface(stake)
            assert isinstance(result, PF.Candidates)
            assert result.n_considered >= len(result.recommended) >= 1
            assert result.n_considered > 1

    def test_the_selection_correction_bites_on_the_live_trade_board(self):
        """The defect this module shipped with, pinned on the data that produced it.

        On 2026-09-07 the two top actionable rows were Blacksburg trades at +1.225pp
        +/- 0.477 and +1.125pp +/- 0.476, both "significant" at two sigma and both the
        argmax of forty confirmed packages, where the bar is 3.23 sigma.

        **This asserted `not any(i.significant for i in trades)` and that was asserting
        more than the statistics claim.** A Bonferroni threshold at `alpha = 0.05` is
        built to admit about one board in twenty, so "no trade ever clears" is an
        invariant nothing promises -- the same over-assertion as the three live-market
        tests rewritten in `7ea0553`. It fired the first time a forced-cut tie-break
        moved a borderline row by 0.04pp, and the row it fired on is noise: across seeds
        1, 2 and 3 the significant set is `{}`, `{}` and one trade, and that one trade
        does not appear in the other seeds' top three at all. That is the max-of-N
        signature the audit already correctly dropped a finding for.

        What the correction is *for* survives being stated properly: rows that a naive
        two-sigma test would call significant must stop being significant once the
        threshold accounts for the field they won. That cannot rot into a coin flip.
        """
        portfolio = PF.build_portfolio(LIVE, 2026, n_sims=2000)
        queue = PF.action_queue(portfolio, limit=40)
        trades = [i for i in queue.items if i.surface == "trades"]
        waivers = [i for i in queue.items if i.surface == "waivers"]
        assert trades and waivers

        assert all(i.n_considered >= 20 for i in trades)
        assert all(i.selection_z > 3.0 for i in trades)

        naive = [i for i in trades if abs(i.delta_title) > 2.0 * i.stderr > 0.0]
        corrected = [i for i in trades if i.significant]
        assert naive, "no trade cleared even two sigma; the correction has nothing to bite on"
        assert len(corrected) < len(naive), [
            (i.delta_title, i.stderr, i.selection_z) for i in corrected
        ]
        # Whatever survives is a hair over the bar, never comfortably clear of it. A
        # trade at five sigma against a 3.23 threshold would be a real finding.
        for i in corrected:
            assert abs(i.delta_title) < 1.5 * i.selection_z * i.stderr, (
                i.delta_title,
                i.stderr,
                i.selection_z,
            )
        # And the small, precise surface is the one that is actually established.
        assert all(i.significant for i in waivers)
        assert max(i.delta_title for i in trades) > 3.0 * max(i.delta_title for i in waivers)
        assert queue.by_selection_bound[0].surface != "trades"
