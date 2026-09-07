"""Lineup-solver tests. The acceptance test is greedy against an assignment-problem oracle.

Everything here runs offline. The two corpus-gated tests read the Parquet snapshot
already in the repo and skip if it is absent.

The randomized suite is the point of the file. `optimal_lineup` claims that a greedy
fill of the most restrictive slot first is *exactly* optimal whenever the eligibility
family is laminar, and that claim is worth almost nothing asserted on three hand-built
rosters. So it is checked on thousands of randomized structures -- real ESPN slot
tables with random counts, plus randomly grown laminar forests -- against
`scipy.optimize.linear_sum_assignment`, which is exact by construction. The non-laminar
half is checked the other way round: it is not enough that the detector fires, the
cases it fires on must actually contain disagreements, or the detector is untested.

**That oracle is not independent enough on its own**, which is worth stating because it
is the kind of thing a test suite congratulates itself for having. `plan.solve(
method="exact")` and `method="greedy"` both run off the *same* compiled `LineupPlan`:
the same `_eligible` matrix, the same per-group counts, the same `_effective_scores` and
the same `_floor_matrix`. An error in compiling positions into an eligibility matrix, in
expanding groups into slot instances, or in normalising a floor vector would move both
answers together and every one of those thousands of comparisons would pass. So
`TestAgainstAnIndependentOracle` re-poses the problem from the raw inputs -- slot
eligibility by position, roster positions, counts, scores, floors -- and solves it by
exhaustive DP over used-player bitmasks, sharing no line of code with the module.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from fantasy_quant.core import DST, QB, RB, TE, WR, K
from fantasy_quant.sim.lineup import (
    LineupError,
    LineupPlan,
    is_laminar,
    laminar_violations,
    monotone_floor,
    optimal_lineup,
    optimal_lineup_with_floor,
    plan_from_context,
    plan_from_slots,
)

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data" / "snapshots" / "espn"
REFERENCE = REPO / "data" / "reference"
HAS_CORPUS = bool(sorted(CORPUS.glob("season=*/variant=ppr/*.parquet")))
needs_corpus = pytest.mark.skipif(not HAS_CORPUS, reason="no ESPN Parquet corpus present")

# lineupSlotId -> eligible defaultPositionIds, straight out of ESPN's platform settings.
# 4 is WR here and TE in the position space; 23 is FLEX, 7 is OP/superflex, 3 is RB/WR
# and 5 is WR/TE -- and those last two are the pair that breaks laminarity.
SLOT_ELIGIBILITY: Mapping[int, frozenset[int]] = {
    0: frozenset({QB}),
    2: frozenset({RB}),
    3: frozenset({RB, WR}),
    4: frozenset({WR}),
    5: frozenset({WR, TE}),
    6: frozenset({TE}),
    7: frozenset({QB, RB, WR, TE}),
    16: frozenset({DST}),
    17: frozenset({K}),
    23: frozenset({RB, WR, TE}),
}

# The IDP block, also ESPN's. Despite the overlapping-sounding slot names, a clean tree.
IDP_ELIGIBILITY: Mapping[int, frozenset[int]] = {
    8: frozenset({9}),  # DT
    9: frozenset({10}),  # DE
    10: frozenset({11}),  # LB
    11: frozenset({9, 10}),  # DL
    12: frozenset({12}),  # CB
    13: frozenset({13}),  # S
    14: frozenset({12, 13}),  # DB
    15: frozenset({9, 10, 11, 12, 13}),  # DP
}

#: The user's three leagues: 1QB/2RB/2WR/1TE/1FLEX/1DST/1K.
STANDARD_SLOTS: Mapping[int, int] = {0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1, 20: 7, 21: 1}
SUPERFLEX_SLOTS: Mapping[int, int] = {**STANDARD_SLOTS, 7: 1}
#: A shape ESPN allows and greedy cannot solve: RB/WR and WR/TE started together.
TWO_FLEX_SLOTS: Mapping[int, int] = {0: 1, 2: 1, 4: 1, 6: 1, 3: 1, 5: 1, 20: 7}

#: A plausible 16-man roster: two QB, four RB, five WR, two TE, a kicker, two defenses.
ROSTER = np.array([QB, QB, RB, RB, RB, RB, WR, WR, WR, WR, WR, TE, TE, K, DST, DST])


def make_plan(
    slots: Mapping[int, int] = STANDARD_SLOTS,
    roster: np.ndarray = ROSTER,
    eligibility: Mapping[int, frozenset[int]] = SLOT_ELIGIBILITY,
) -> LineupPlan:
    return plan_from_slots(slots, eligibility, roster)


def assert_assignment_is_legal(plan: LineupPlan, roster: np.ndarray, result) -> None:
    """No player in two slots, and nobody in a slot he cannot fill.

    Double-starting is the failure this whole file is guarding: it inflates every
    simulated score by roughly one starter and nothing downstream would notice.
    """
    assignment = np.asarray(result.assignment).reshape(-1, plan.n_slots)
    eligible = np.repeat(plan._eligible, plan.group_counts, axis=0)
    for row in assignment:
        used = [p for p in row if p >= 0]
        assert len(used) == len(set(used)), f"player started twice: {row}"
        for slot, player in enumerate(row):
            if player >= 0:
                assert eligible[slot, player], (
                    f"slot {plan.slot_ids[slot]} started position {roster[player]}"
                )


# --------------------------------------------------------------------------------------
# Laminarity: is greedy allowed here at all
# --------------------------------------------------------------------------------------


class TestLaminarity:
    def test_the_users_leagues_are_laminar(self) -> None:
        eligible = {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 4, 6, 23, 16, 17)}
        assert is_laminar(eligible)
        assert make_plan().laminar

    def test_superflex_is_laminar(self) -> None:
        eligible = {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 4, 6, 23, 7, 16, 17)}
        assert is_laminar(eligible), "dedicated < FLEX < OP is the nested case"
        assert make_plan(SUPERFLEX_SLOTS).laminar

    def test_the_idp_block_is_laminar(self) -> None:
        """The obvious guess at the non-laminar case is IDP. The guess is wrong.

        DT and DE nest inside DL, CB and S inside DB, and DL, LB and DB all inside DP.
        That is a tree, so greedy is exact on a full IDP lineup. (RESEARCH.md and
        PLAN.md say only that greedy is exact for nested eligibility; neither names a
        violating slot table, so this is measured here rather than contradicted.)
        """
        assert is_laminar(IDP_ELIGIBILITY)

    def test_rb_wr_against_wr_te_is_the_real_violation(self) -> None:
        assert laminar_violations({3: SLOT_ELIGIBILITY[3], 5: SLOT_ELIGIBILITY[5]}) == ((3, 5),)
        assert not is_laminar({s: SLOT_ELIGIBILITY[s] for s in (0, 2, 3, 4, 5, 6)})

    def test_greedy_actually_loses_on_the_two_flex_shape(self) -> None:
        """The violation is not theoretical: it costs a whole starter."""
        roster = np.array([WR, RB, TE])
        plan = plan_from_slots({3: 1, 5: 1}, SLOT_ELIGIBILITY, roster)
        scores = np.array([[10.0, 8.0, 1.0]])
        assert not plan.laminar
        greedy = plan.solve(scores, method="greedy").total[0]
        exact = plan.solve(scores, method="exact").total[0]
        # Greedy hands the lone receiver to RB/WR and strands WR/TE on the 1.0 tight end.
        assert greedy == pytest.approx(11.0)
        assert exact == pytest.approx(18.0)
        assert plan.solve(scores).total[0] == pytest.approx(18.0), "auto must not use greedy"

    def test_laminarity_is_a_property_of_the_roster_not_the_league(self) -> None:
        """Drop the tight end and the same illegal league becomes greedy-safe.

        This is why the plan compiles the player matrix rather than the position table:
        the position-level answer would send this roster down the exact solver for
        nothing, at a couple of hundred times the cost.
        """
        assert not is_laminar({3: SLOT_ELIGIBILITY[3], 5: SLOT_ELIGIBILITY[5]})
        roster = np.array([WR, RB])
        plan = plan_from_slots({3: 1, 5: 1}, SLOT_ELIGIBILITY, roster)
        assert plan.laminar
        scores = np.array([[10.0, 8.0]])
        assert plan.solve(scores, method="greedy").total[0] == pytest.approx(18.0)
        assert plan.solve(scores, method="exact").total[0] == pytest.approx(18.0)

    def test_a_full_two_flex_league_routes_to_the_exact_solver(self) -> None:
        """The whole league shape, not just the offending pair, on a real roster."""
        plan = make_plan(TWO_FLEX_SLOTS)
        assert plan.violations == ((3, 5),)
        rng = np.random.default_rng(9)
        scores = rng.gamma(3.0, 5.0, size=(300, len(ROSTER)))
        auto = plan.solve(scores)
        exact = plan.solve(scores, method="exact")
        greedy = plan.solve(scores, method="greedy")
        assert np.allclose(auto.total, exact.total)
        assert np.all(greedy.total <= exact.total + 1e-9)
        assert np.any(greedy.total < exact.total - 1e-9), "and greedy really does lose"
        assert_assignment_is_legal(plan, ROSTER, auto)

    @pytest.mark.skipif(
        not (REFERENCE / "platform_settings_2026.json").exists(), reason="no cached settings"
    )
    def test_espn_slot_table_violation_is_rb_wr_vs_wr_te(self) -> None:
        """Pinned against ESPN's own table, so this fails if ESPN changes the slots."""
        import json

        payload = json.loads((REFERENCE / "platform_settings_2026.json").read_text())
        slots = payload["payload"]["settings"]["lineupSlots"]
        starters = {
            int(s["id"]): frozenset(s.get("eligiblePositions") or ())
            for s in slots
            if s.get("starter") and s.get("eligiblePositions")
        }
        assert laminar_violations(starters) == ((3, 5),), (
            "the only crossing pair among ESPN's 22 starting slots is RB/WR against WR/TE"
        )


# --------------------------------------------------------------------------------------
# What the greedy fill actually does
# --------------------------------------------------------------------------------------


class TestGreedySemantics:
    def test_flex_takes_the_best_leftover_regardless_of_position(self) -> None:
        plan = make_plan()
        scores = np.zeros((3, len(ROSTER)))
        scores[:, [0, 1]] = [20.0, 5.0]  # QBs
        scores[:, [2, 3, 4, 5]] = [18.0, 16.0, 9.0, 1.0]  # RBs
        scores[:, [6, 7, 8, 9, 10]] = [17.0, 15.0, 8.0, 1.0, 1.0]  # WRs
        scores[:, [11, 12]] = [12.0, 2.0]  # TEs
        scores[:, 13], scores[:, [14, 15]] = 7.0, [6.0, 1.0]
        # Row 0 leaves RB3 (9.0) as the best leftover. Row 1 promotes a receiver into
        # the WR slots and pushes WR2 (15.0) down to the flex. Row 2 does the same to
        # the tight ends, so the flex becomes a TE (12.0). Same slot, three positions.
        scores[1, 8] = 30.0
        scores[2, 12] = 40.0
        result = plan.solve(scores)
        flex = plan.slot_ids.index(23)
        picks = result.assignment[:, flex]
        assert [ROSTER[p] for p in picks] == [RB, WR, TE]
        assert [scores[r, picks[r]] for r in range(3)] == [9.0, 15.0, 12.0]
        assert_assignment_is_legal(plan, ROSTER, result)

    def test_superflex_takes_a_quarterback_when_the_qb_is_the_best_remaining(self) -> None:
        plan = make_plan(SUPERFLEX_SLOTS)
        scores = np.zeros((2, len(ROSTER)))
        scores[:, [0, 1]] = [25.0, 22.0]
        scores[:, [2, 3, 4, 5]] = [18.0, 16.0, 9.0, 1.0]
        scores[:, [6, 7, 8, 9, 10]] = [17.0, 15.0, 8.0, 1.0, 1.0]
        scores[:, [11, 12]] = [12.0, 2.0]
        op = plan.slot_ids.index(7)
        flex = plan.slot_ids.index(23)
        result = plan.solve(scores)
        assert result.assignment[0, op] == 1, "QB2 at 22 beats every leftover skill player"
        assert result.assignment[0, flex] == 4
        # Drop QB2 below the leftovers and the superflex must stop being a QB slot.
        scores[1, 1] = 3.0
        result = plan.solve(scores)
        assert ROSTER[result.assignment[1, op]] != QB
        assert_assignment_is_legal(plan, ROSTER, result)

    def test_a_short_roster_leaves_slots_empty_at_zero(self) -> None:
        roster = np.array([QB, RB, WR])
        plan = plan_from_slots(STANDARD_SLOTS, SLOT_ELIGIBILITY, roster)
        result = plan.solve(np.array([[20.0, 10.0, 8.0]]))
        assert result.total[0] == pytest.approx(38.0)
        assert (result.assignment[0] == -1).sum() == plan.n_slots - 3
        assert_assignment_is_legal(plan, roster, result)

    def test_a_bye_scores_zero_and_is_not_started(self) -> None:
        """`WeeklyOutlook.zeroed()` gives mean 0, and 0 is not greater than 0.

        Starting him and leaving the slot open are the same number, so the solver
        prefers the honest assignment: the slot reads empty rather than naming a
        player who was never going to play.
        """
        roster = np.array([RB, RB, RB])
        plan = plan_from_slots({2: 2}, SLOT_ELIGIBILITY, roster)
        result = plan.solve(np.array([[12.0, 0.0, 4.0]]))
        assert result.total[0] == pytest.approx(16.0)
        assert sorted(result.assignment[0]) == [0, 2]

    def test_an_unavailable_player_cannot_be_started_whatever_he_scores(self) -> None:
        roster = np.array([RB, RB])
        plan = plan_from_slots({2: 1}, SLOT_ELIGIBILITY, roster)
        scores = np.array([[30.0, 9.0]])
        available = np.array([[False, True]])
        result = plan.solve(scores, available=available)
        assert result.total[0] == pytest.approx(9.0)
        assert result.assignment[0, 0] == 1

    def test_a_negative_score_is_benched_unless_the_slot_must_be_filled(self) -> None:
        """Fantasy points go negative, so 'empty scores zero' is a real decision."""
        roster = np.array([DST])
        plan = plan_from_slots({16: 1}, SLOT_ELIGIBILITY, roster)
        scores = np.array([[-4.0]])
        assert plan.solve(scores).total[0] == pytest.approx(0.0)
        assert plan.solve(scores).assignment[0, 0] == -1
        forced = plan.solve(scores, allow_empty=False)
        assert forced.total[0] == pytest.approx(-4.0)
        assert forced.assignment[0, 0] == 0

    def test_one_player_cannot_fill_two_slots(self) -> None:
        """The single most damaging bug this module could have.

        One enormous running back and nine zeroes: the total must be his score once,
        not once per slot he is eligible for.
        """
        roster = np.array([RB, WR, WR, TE, QB, K, DST])
        plan = plan_from_slots(STANDARD_SLOTS, SLOT_ELIGIBILITY, roster)
        scores = np.zeros((1, len(roster)))
        scores[0, 0] = 100.0
        result = plan.solve(scores)
        assert result.total[0] == pytest.approx(100.0)
        assert (result.assignment[0] == 0).sum() == 1
        assert_assignment_is_legal(plan, roster, result)

    def test_totals_do_not_depend_on_whether_the_assignment_was_asked_for(self) -> None:
        plan = make_plan()
        rng = np.random.default_rng(11)
        scores = rng.gamma(3.0, 4.0, size=(500, len(ROSTER)))
        with_ids = plan.solve(scores)
        without = plan.solve(scores, assignment=False)
        assert np.allclose(with_ids.total, without.total)
        assert without.assignment is None

    def test_a_nan_score_does_not_split_the_two_paths_apart(self) -> None:
        """Clean gamma draws cannot catch this, and one NaN cost eleven points.

        The two paths build their descending queue differently -- `argsort(-x)` with the
        assignment, a plain sort without -- and numpy sorts NaN *high*. Reversing an
        ascending sort therefore put NaN at the head of a queue, where it fails
        `> floor`, stops the cursor and strands every player behind it: the whole
        receiving corps went unstarted on the totals-only path only. A non-finite score
        must behave the way `available=False` does, in all three solvers.
        """
        plan = make_plan()
        scores = np.zeros((1, len(ROSTER)))
        scores[0, 0] = 25.0
        scores[0, 2:6] = [20.0, 18.0, 16.0, 14.0]
        scores[0, 6:11] = [19.0, 17.0, 15.0, 13.0, 11.0]
        assert plan.solve(scores, assignment=False).total[0] == pytest.approx(115.0)
        scores[0, 3] = np.nan  # RB2 -- the second entry of a five-deep queue
        with_ids = plan.solve(scores).total[0]
        without = plan.solve(scores, assignment=False).total[0]
        exact = plan.solve(scores, method="exact").total[0]
        assert with_ids == pytest.approx(112.0), "20 + 16 + 19 + 17 + 25 + 15 (flex)"
        assert without == pytest.approx(with_ids), "the fast path must not strand RB3-5"
        assert exact == pytest.approx(with_ids)
        assert 3 not in plan.solve(scores).assignment[0].tolist(), "NaN is never started"

    def test_leading_axes_are_preserved(self) -> None:
        plan = make_plan()
        rng = np.random.default_rng(3)
        scores = rng.gamma(3.0, 4.0, size=(7, 5, 3, len(ROSTER)))
        result = plan.solve(scores)
        assert result.total.shape == (7, 5, 3)
        assert result.assignment.shape == (7, 5, 3, plan.n_slots)
        flat = plan.solve(scores.reshape(-1, len(ROSTER)))
        assert np.allclose(result.total.reshape(-1), flat.total)

    def test_bench_and_ir_slots_never_start_anyone(self) -> None:
        plan = make_plan()
        assert 20 not in plan.slot_ids and 21 not in plan.slot_ids
        assert plan.n_slots == 9


# --------------------------------------------------------------------------------------
# The acceptance test: greedy against the assignment-problem optimum
# --------------------------------------------------------------------------------------


def random_laminar_family(
    rng: np.random.Generator, positions: Sequence[int]
) -> dict[int, frozenset[int]]:
    """Grow a laminar forest by merging disjoint roots, then keep a random subset.

    Merging disjoint sets can only produce nested-or-disjoint pairs, so the family is
    laminar by construction rather than by rejection sampling.
    """
    nodes = [frozenset({p}) for p in positions]
    roots = list(range(len(nodes)))
    while len(roots) > 1 and rng.random() < 0.75:
        k = int(rng.integers(2, len(roots) + 1))
        chosen = rng.choice(len(roots), size=k, replace=False)
        merged = frozenset().union(*(nodes[roots[int(i)]] for i in chosen))
        nodes.append(merged)
        roots = [r for i, r in enumerate(roots) if i not in set(int(c) for c in chosen)]
        roots.append(len(nodes) - 1)
    keep = [n for n in nodes if rng.random() < 0.6] or [nodes[0]]
    return {i: s for i, s in enumerate(keep)}


def random_crossing_family(
    rng: np.random.Generator, positions: Sequence[int]
) -> dict[int, frozenset[int]]:
    """Random subsets, resampled until at least one pair genuinely crosses."""
    for _ in range(64):
        family = {}
        for i in range(int(rng.integers(2, 6))):
            size = int(rng.integers(1, len(positions) + 1))
            family[i] = frozenset(rng.choice(positions, size=size, replace=False).tolist())
        if laminar_violations(family):
            return family
    raise AssertionError("could not draw a crossing family")


def random_case(
    rng: np.random.Generator, family: Mapping[int, frozenset[int]], positions: Sequence[int]
) -> tuple[LineupPlan, np.ndarray, np.ndarray]:
    """A random roster and score matrix for a given eligibility family."""
    roster: list[int] = []
    for p in positions:
        roster.extend([p] * int(rng.integers(0, 5)))
    if not roster:
        roster = [int(rng.choice(positions))]
    roster_arr = np.array(roster)
    counts = {s: int(rng.integers(1, 3)) for s in family}
    plan = plan_from_slots(counts, family, roster_arr)
    rows = int(rng.integers(1, 4))
    scores = rng.gamma(2.0, 6.0, size=(rows, len(roster_arr)))
    # Zeroes are byes, negatives are three-interception games; both must be handled.
    scores[rng.random(scores.shape) < 0.15] = 0.0
    negative = rng.random(scores.shape) < 0.05
    scores[negative] = -rng.gamma(1.0, 3.0, size=int(negative.sum()))
    return plan, roster_arr, scores


POSITIONS = (QB, RB, WR, TE, K, DST)

#: Real ESPN shapes the randomized suite draws from, alongside the grown forests.
REAL_SHAPES: tuple[Mapping[int, frozenset[int]], ...] = (
    {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 4, 6, 23, 16, 17)},
    {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 4, 6, 23, 7, 16, 17)},
    {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 4, 6, 23)},
    {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 4, 5, 6)},
    {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 3, 4, 6)},
    IDP_ELIGIBILITY,
)


class TestAgainstTheOracle:
    def test_greedy_equals_the_assignment_optimum_on_laminar_structures(self) -> None:
        """The claim the whole module rests on, on 6,000 randomized rosters."""
        rng = np.random.default_rng(20260907)
        checked = 0
        for _ in range(2500):
            family = random_laminar_family(rng, POSITIONS)
            plan, roster, scores = random_case(rng, family, POSITIONS)
            assert plan.laminar, "a merged forest is laminar by construction"
            greedy = plan.solve(scores, method="greedy")
            exact = plan.solve(scores, method="exact")
            assert np.allclose(greedy.total, exact.total), (
                f"greedy lost on a laminar family: {family} "
                f"roster={roster.tolist()} greedy={greedy.total} exact={exact.total}"
            )
            assert_assignment_is_legal(plan, roster, greedy)
            checked += scores.shape[0]
        for shape in REAL_SHAPES:
            positions = tuple(sorted({p for s in shape.values() for p in s}))
            for _ in range(250):
                plan, roster, scores = random_case(rng, shape, positions)
                if not plan.laminar:
                    continue
                greedy = plan.solve(scores, method="greedy")
                exact = plan.solve(scores, method="exact")
                assert np.allclose(greedy.total, exact.total), (
                    f"greedy lost on ESPN shape {dict(shape)} roster={roster.tolist()}"
                )
                checked += scores.shape[0]
        assert checked > 5000, f"only {checked} rosters actually got checked"

    def test_the_detector_fires_on_every_structure_where_greedy_loses(self) -> None:
        """Non-laminar cases: greedy may lose, and it must never be trusted when it can.

        The assertion that matters is not that the detector fires -- these families were
        drawn crossing -- but that greedy really does lose on some of them, so the
        detector is guarding something. If this stops finding disagreements the
        generator has gone weak, not the solver.
        """
        rng = np.random.default_rng(4242)
        disagreements = 0
        for _ in range(600):
            family = random_crossing_family(rng, POSITIONS)
            plan, roster, scores = random_case(rng, family, POSITIONS)
            greedy = plan.solve(scores, method="greedy")
            exact = plan.solve(scores, method="exact")
            assert np.all(greedy.total <= exact.total + 1e-9), "greedy beat the optimum"
            if not np.allclose(greedy.total, exact.total):
                assert not plan.laminar, (
                    f"greedy disagreed with the optimum but the detector said laminar: "
                    f"{dict(family)} roster={roster.tolist()}"
                )
                disagreements += 1
                # `auto` is the contract: a crossing structure must route to exact.
                assert np.allclose(plan.solve(scores).total, exact.total)
        assert disagreements > 20, (
            f"only {disagreements} disagreements found; the generator is not exercising "
            "the non-laminar path hard enough to call this a test"
        )

    def test_greedy_and_exact_agree_on_which_players_start_not_just_the_total(self) -> None:
        """Same roster, same scores, two methods: the started *set* has to match.

        `method="auto"` picks between them on roster shape, and `sim/season.py` chooses
        the lineup on a projection tensor and then scores the *realised* tensor at those
        indices -- so an assignment that starts a projected-zero player is not free even
        when the two totals agree. The assignment-problem solver is indifferent between
        a player who exactly ties his floor and an empty slot, so left alone it filled
        every slot it could reach; a roster of byes made it start nine players where
        greedy started three. Which *slot* holds a given player is still free to differ.
        """
        plan = make_plan()
        rng = np.random.default_rng(3)
        for _ in range(150):
            scores = rng.gamma(2.0, 6.0, size=(4, len(ROSTER)))
            scores[rng.random(scores.shape) < 0.35] = 0.0  # byes: exactly the floor
            greedy = plan.solve(scores, method="greedy")
            exact = plan.solve(scores, method="exact")
            assert np.allclose(greedy.total, exact.total)
            for row in range(4):
                g = {int(p) for p in greedy.assignment[row] if p >= 0}
                e = {int(p) for p in exact.assignment[row] if p >= 0}
                assert g == e, f"greedy started {sorted(g)}, exact started {sorted(e)}"

    def test_a_player_who_only_ties_the_floor_is_never_started_by_either_method(self) -> None:
        """The rule the module states, held to by the fallback as well as the kernel."""
        roster = np.array([RB, RB, RB])
        plan = plan_from_slots({2: 2}, SLOT_ELIGIBILITY, roster)
        for method in ("greedy", "exact"):
            result = plan.solve(np.array([[12.0, 5.0, 4.0]]), floor=5.0, method=method)
            assert result.total[0] == pytest.approx(17.0)
            assert result.assignment[0].tolist() == [0, -1], method
        # ... and with floor 0, a bye is a tie at 0 and stays on the bench.
        for method in ("greedy", "exact"):
            result = plan.solve(np.array([[12.0, 0.0, 0.0]]), method=method)
            assert (result.assignment[0] >= 0).sum() == 1, method

    def test_floors_match_the_oracle_too(self) -> None:
        rng = np.random.default_rng(77)
        for _ in range(400):
            family = random_laminar_family(rng, POSITIONS)
            plan, roster, scores = random_case(rng, family, POSITIONS)
            raw = rng.gamma(2.0, 4.0, size=plan.n_groups)
            floor = monotone_floor(plan, raw)
            greedy = plan.solve(scores, floor=floor, method="greedy")
            exact = plan.solve(scores, floor=floor, method="exact")
            assert np.allclose(greedy.total, exact.total), (
                f"greedy lost against a monotone floor: {dict(family)} floor={floor}"
            )


# --------------------------------------------------------------------------------------
# A second opinion that shares no code with the module
# --------------------------------------------------------------------------------------


def independent_optimum(
    slot_players: Sequence[tuple[int, ...]], scores: Sequence[float], floors: Sequence[float]
) -> float:
    """Exhaustive search over injective slot->player maps, by DP on used-player bitmasks.

    Deliberately written from nothing: it takes per-slot-*instance* eligible player
    tuples, a score per player and a floor per slot, and enumerates. No `LineupPlan`, no
    eligibility matrix, no solve order, no scipy. Slot order does not affect the maximum,
    so the caller does not have to reproduce the module's ordering either -- which is
    exactly the point, since reproducing it would reintroduce the shared assumption.
    """
    n_slots = len(slot_players)

    @cache
    def best(index: int, used: int) -> float:
        if index == n_slots:
            return 0.0
        top = floors[index] + best(index + 1, used)
        for player in slot_players[index]:
            bit = 1 << player
            if used & bit:
                continue
            candidate = scores[player] + best(index + 1, used | bit)
            if candidate > top:
                top = candidate
        return top

    answer = best(0, 0)
    best.cache_clear()
    return answer


def independent_problem(
    slot_counts: Mapping[int, int],
    family: Mapping[int, frozenset[int]],
    roster: np.ndarray,
    floor: Mapping[int, float] | None,
) -> tuple[list[tuple[int, ...]], list[float]]:
    """Re-pose the lineup problem straight from positions, without the plan."""
    instances: list[tuple[int, ...]] = []
    floors: list[float] = []
    for slot in sorted(slot_counts):
        eligible = tuple(i for i, pos in enumerate(roster) if pos in family[slot])
        for _ in range(slot_counts[slot]):
            instances.append(eligible)
            floors.append(0.0 if floor is None else float(floor[slot]))
    return instances, floors


class TestAgainstAnIndependentOracle:
    """The check `TestAgainstTheOracle` structurally cannot make.

    Greedy and the scipy fallback both consume the same compiled plan, so an error in
    compiling positions to an eligibility matrix, expanding groups into slot instances,
    or normalising a floor vector would move both together. This re-derives the answer
    from the raw inputs.
    """

    def test_solve_matches_an_exhaustive_search_built_from_the_raw_inputs(self) -> None:
        rng = np.random.default_rng(19)
        checked = 0
        for _ in range(500):
            family = random_laminar_family(rng, POSITIONS)
            counts = {s: int(rng.integers(1, 3)) for s in family}
            if sum(counts.values()) > 7:
                continue
            roster: list[int] = []
            for position in POSITIONS:
                roster.extend([position] * int(rng.integers(0, 4)))
            roster_arr = np.array(roster[:12] or [int(rng.choice(POSITIONS))])
            plan = plan_from_slots(counts, family, roster_arr)
            scores = rng.gamma(2.0, 6.0, size=(2, len(roster_arr)))
            scores[rng.random(scores.shape) < 0.2] = 0.0
            negative = rng.random(scores.shape) < 0.08
            scores[negative] = -rng.gamma(1.0, 3.0, size=int(negative.sum()))
            # A floor that is admissible by construction and derived from the *player*
            # sets, so it does not borrow the module's notion of which slot is wider.
            floor = {s: 0.75 * sum(1 for pos in roster_arr if pos in family[s]) for s in counts}
            instances, floors = independent_problem(counts, family, roster_arr, floor)
            for method in ("greedy", "exact"):
                result = plan.solve(scores, floor=floor, method=method)
                for row in range(scores.shape[0]):
                    want = independent_optimum(instances, tuple(scores[row]), floors)
                    assert result.total[row] == pytest.approx(want, abs=1e-9), (
                        f"{method} returned {result.total[row]} against an exhaustive "
                        f"{want}: family={dict(family)} counts={counts} "
                        f"roster={roster_arr.tolist()} scores={scores[row].tolist()}"
                    )
            checked += 1
        assert checked > 300, f"only {checked} structures survived the size filter"

    def test_the_espn_shapes_match_an_exhaustive_search_too(self) -> None:
        """Including the non-laminar two-flex shape, where `auto` must route to exact."""
        rng = np.random.default_rng(202)
        for family in (*REAL_SHAPES, {s: SLOT_ELIGIBILITY[s] for s in (0, 3, 5, 6)}):
            positions = tuple(sorted({p for s in family.values() for p in s}))
            for _ in range(25):
                counts = {s: 1 for s in family}
                roster: list[int] = []
                for position in positions:
                    roster.extend([position] * int(rng.integers(0, 3)))
                roster_arr = np.array(roster[:11] or [int(rng.choice(positions))])
                if len(family) > 7:
                    continue
                plan = plan_from_slots(counts, family, roster_arr)
                scores = rng.gamma(2.0, 6.0, size=len(roster_arr))
                scores[rng.random(scores.shape) < 0.2] = 0.0
                instances, floors = independent_problem(counts, family, roster_arr, None)
                want = independent_optimum(instances, tuple(scores), floors)
                assert plan.solve(scores).total == pytest.approx(want, abs=1e-9), (
                    f"auto lost on {dict(family)} roster={roster_arr.tolist()}"
                )

    def test_the_users_own_league_shape_matches_an_exhaustive_search(self) -> None:
        """The shape that actually runs, on the roster that actually runs, with floors."""
        family = {s: SLOT_ELIGIBILITY[s] for s in (0, 2, 4, 6, 23, 16, 17)}
        counts = {s: n for s, n in STANDARD_SLOTS.items() if s not in (20, 21)}
        plan = make_plan()
        rng = np.random.default_rng(808)
        floor = {0: 6.0, 2: 5.0, 4: 5.0, 6: 4.0, 23: 7.0, 16: 5.0, 17: 6.0}
        instances, floors = independent_problem(counts, family, ROSTER, floor)
        for _ in range(40):
            scores = rng.gamma(2.0, 6.0, size=len(ROSTER))
            scores[rng.random(scores.shape) < 0.25] = 0.0
            want = independent_optimum(instances, tuple(scores), floors)
            assert plan.solve(scores, floor=floor).total == pytest.approx(want, abs=1e-9)
            plain = independent_optimum(instances, tuple(scores), [0.0] * len(instances))
            assert plan.solve(scores).total == pytest.approx(plain, abs=1e-9)


# --------------------------------------------------------------------------------------
# The free-agent floor
# --------------------------------------------------------------------------------------


class TestFreeAgentFloor:
    def test_a_bench_player_worse_than_the_wire_is_worth_nothing(self) -> None:
        """The whole point: `E[max]` over a bench is not value if the wire is deeper."""
        eligibility, counts = np.array([[True, True]]), np.array([1])
        scores = np.array([[14.0, 6.0], [3.0, 2.0]])
        plain = optimal_lineup(scores, eligibility, counts)
        floored = optimal_lineup_with_floor(scores, eligibility, counts, 8.0)
        assert plain.total.tolist() == [14.0, 3.0]
        # Week 1 the starter beats the streamer; week 2 the whole roster is worth the
        # streamer and nothing more, which is exactly the drop decision.
        assert floored.total.tolist() == [14.0, 8.0]
        assert floored.assignment[1, 0] == -1

    def test_the_floor_is_taken_per_slot(self) -> None:
        plan = make_plan()
        scores = np.zeros((1, len(ROSTER)))
        floor = {0: 12.0, 2: 8.0, 4: 8.0, 6: 5.0, 23: 8.0, 16: 6.0, 17: 7.0}
        result = plan.solve(scores, floor=floor)
        expected = sum(floor[s] for s in plan.slot_ids)
        assert result.total[0] == pytest.approx(expected)
        assert (result.assignment[0] == -1).all()

    def test_a_narrower_slot_may_not_have_a_higher_floor(self) -> None:
        """The one hypothesis of the exactness proof a caller can break, so it is loud."""
        plan = make_plan()
        floor = {0: 12.0, 2: 9.5, 4: 8.0, 6: 5.0, 23: 0.0, 16: 6.0, 17: 7.0}
        with pytest.raises(LineupError, match="subset of slot 23"):
            plan.solve(np.zeros((1, len(ROSTER))), floor=floor)

    def test_that_rejection_is_not_pedantry(self) -> None:
        """With the bad floor let through, greedy really would return 10.1 against 19.5.

        Reaches past the guard on purpose -- the point is to show what the guard buys.
        """
        roster = np.array([RB, RB])
        plan = plan_from_slots({2: 1, 23: 1}, SLOT_ELIGIBILITY, roster)
        assert plan.group_slot_ids == (2, 23)
        eff = np.array([[10.0, 0.1]])
        bad = np.array([[9.5, 0.0]])
        with pytest.raises(LineupError):
            plan.solve(eff, floor=bad)
        assert plan._greedy(eff, bad, False)[0][0] == pytest.approx(10.1)
        assert plan._exact(eff, bad, False)[0][0] == pytest.approx(19.5)

    def test_two_slots_that_start_the_same_players_must_agree_on_the_floor_order(self) -> None:
        """The equal-sets case the randomized oracle found, at minimum size.

        No kicker on the roster, so a WR slot and a WR/K slot accept exactly the same
        two players. Filling the higher-floor slot first wastes the better receiver on
        the slot with less to gain from him.
        """
        roster = np.array([WR, WR])
        plan = plan_from_slots({4: 1, 5: 1}, {4: frozenset({WR}), 5: frozenset({WR, K})}, roster)
        assert plan.group_slot_ids == (4, 5) and plan.laminar
        eff = np.array([[20.0, 11.28]])
        with pytest.raises(LineupError, match="starts the same players as"):
            plan.solve(eff, floor={4: 14.72, 5: 9.24})
        assert plan._greedy(eff, np.array([[14.72, 9.24]]), False)[0][0] == pytest.approx(31.28)
        assert plan._exact(eff, np.array([[14.72, 9.24]]), False)[0][0] == pytest.approx(34.72)
        # In the admissible order the same floors are exact.
        ordered = plan.solve(eff, floor={4: 9.24, 5: 14.72})
        assert ordered.total[0] == pytest.approx(34.72)

    def test_monotone_floor_lifts_a_wider_slot_to_its_children(self) -> None:
        plan = make_plan()
        raw = {0: 12.0, 2: 9.5, 4: 8.0, 6: 5.0, 23: 0.0, 16: 6.0, 17: 7.0}
        lifted = monotone_floor(plan, raw)
        # `monotone_floor` returns a positional vector, so it is indexed by
        # `floor_slot_ids` -- the caller's own row order -- not by the solve order.
        flex = plan.floor_slot_ids.index(23)
        rb = plan.floor_slot_ids.index(2)
        assert lifted[0, flex] == pytest.approx(9.5), "FLEX can stream the RB slot's free agent"
        assert lifted[0, rb] == pytest.approx(9.5), "and the RB slot is untouched"
        plan.solve(np.zeros((1, len(ROSTER))), floor=lifted)  # now admissible

    def test_monotone_floor_output_is_always_admissible(self) -> None:
        """One pass must reach the transitive closure, on every shape and per row.

        Asserted through `solve`, not `_check_monotone`, so the permutation between the
        caller's floor order and the solve order is inside the loop being tested.
        """
        rng = np.random.default_rng(31)
        for _ in range(200):
            family = random_laminar_family(rng, POSITIONS)
            plan, roster, _ = random_case(rng, family, POSITIONS)
            raw = rng.gamma(2.0, 4.0, size=(3, plan.n_groups))
            plan.solve(np.zeros((3, len(roster))), floor=monotone_floor(plan, raw))

    def test_a_missing_slot_in_a_floor_mapping_is_an_error(self) -> None:
        plan = make_plan()
        with pytest.raises(LineupError, match="missing starting slots"):
            plan.solve(np.zeros((1, len(ROSTER))), floor={0: 1.0})

    def test_floors_may_vary_by_row(self) -> None:
        roster = np.array([RB, RB])
        plan = plan_from_slots({2: 1}, SLOT_ELIGIBILITY, roster)
        scores = np.array([[10.0, 1.0], [10.0, 1.0]])
        floor = np.array([[4.0], [14.0]])
        result = plan.solve(scores, floor=floor)
        assert result.total.tolist() == [10.0, 14.0]

    def test_a_positional_floor_is_read_in_the_callers_row_order(self) -> None:
        """Not the solve order, which the caller of the front door cannot see.

        The plan sorts slots most-restrictive-first internally, so a floor array read in
        solve order is silently permuted relative to the `eligibility` rows the caller
        handed in. Here the wide slot is row 0 and the narrow slot is row 1, but the
        narrow slot is solved first -- so the two orders are exactly reversed, and the
        old behaviour turned an inadmissible floor (narrow above wide) into a
        perfectly legal-looking answer instead of the error the mapping form raises.
        """
        eligibility = np.array([[True, True], [True, False]])  # row 0 wide, row 1 narrow
        plan = LineupPlan.build(eligibility, [1, 1])
        assert plan.group_slot_ids == (1, 0), "narrow slot is solved first"
        assert plan.floor_slot_ids == (0, 1), "but floors are read in the caller's order"
        scores = np.array([[10.0, 0.1]])
        with pytest.raises(LineupError, match="higher free-agent floor"):
            optimal_lineup_with_floor(scores, eligibility, [1, 1], np.array([0.0, 9.5]))
        with pytest.raises(LineupError, match="higher free-agent floor"):
            plan.solve(scores, floor={0: 0.0, 1: 9.5})
        # The admissible direction agrees between the two spellings, value for value.
        positional = optimal_lineup_with_floor(scores, eligibility, [1, 1], np.array([9.5, 0.1]))
        mapping = plan.solve(scores, floor={0: 9.5, 1: 0.1})
        assert positional.total[0] == pytest.approx(19.5)
        assert mapping.total[0] == pytest.approx(19.5)

    def test_monotone_floor_returns_the_order_solve_reads_back(self) -> None:
        """Round-tripping is the whole contract; a permutation between them is silent."""
        eligibility = np.array([[True, True], [True, False]])
        plan = LineupPlan.build(eligibility, [1, 1])
        lifted = monotone_floor(plan, np.array([0.0, 9.5]))  # wide 0.0, narrow 9.5
        assert lifted[0].tolist() == [9.5, 9.5], "the wide slot is lifted, in caller order"
        assert plan.solve(np.array([[10.0, 0.1]]), floor=lifted).total[0] == pytest.approx(19.5)


class TestFunctionalApi:
    def test_optimal_lineup_accepts_a_bare_eligibility_matrix(self) -> None:
        eligibility = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
        result = optimal_lineup(np.array([[5.0, 9.0, 7.0]]), eligibility, [1, 1])
        assert result.total[0] == pytest.approx(16.0)
        assert sorted(result.assignment[0].tolist()) == [1, 2]

    def test_counts_must_line_up_with_the_matrix(self) -> None:
        with pytest.raises(LineupError, match="one entry per eligibility row"):
            optimal_lineup(np.zeros((1, 3)), np.ones((2, 3), dtype=bool), [1])

    def test_the_score_axis_must_match_the_roster(self) -> None:
        plan = make_plan()
        with pytest.raises(LineupError, match="last axis"):
            plan.solve(np.zeros((1, 4)))

    def test_a_slot_with_no_eligible_player_is_simply_empty(self) -> None:
        roster = np.array([RB, RB])
        plan = plan_from_slots({2: 1, 0: 1}, SLOT_ELIGIBILITY, roster)
        result = plan.solve(np.array([[9.0, 4.0]]))
        assert result.total[0] == pytest.approx(9.0)
        assert result.assignment[0].tolist().count(-1) == 1

    def test_an_unknown_method_is_rejected_rather_than_falling_back(self) -> None:
        with pytest.raises(LineupError, match="method must be"):
            make_plan().solve(np.zeros((1, len(ROSTER))), method="lp")  # type: ignore[arg-type]

    def test_a_one_dimensional_roster_gives_a_scalar_total(self) -> None:
        plan = LineupPlan.build(np.array([[True, True]]), [1])
        result = plan.solve(np.array([3.0, 5.0]))
        assert result.total.shape == ()
        assert float(result.total) == pytest.approx(5.0)
        assert result.assignment.tolist() == [1]

    @pytest.mark.parametrize("method", ["greedy", "exact"])
    def test_an_empty_roster_leaves_every_slot_empty(self, method: str) -> None:
        """Degenerate but reachable: a plan built before the roster is loaded."""
        plan = LineupPlan.build(np.zeros((2, 0), dtype=bool), [1, 1])
        result = plan.solve(np.zeros((3, 0)), method=method)
        assert result.total.tolist() == [0.0, 0.0, 0.0]
        assert (result.assignment == -1).all()

    def test_a_slot_that_cannot_be_filled_at_all_is_minus_infinity_when_forced(self) -> None:
        """`allow_empty=False` means what it says, loudly, rather than quietly scoring 0."""
        plan = LineupPlan.build(np.array([[True], [False]]), [1, 1])
        for method in ("greedy", "exact"):
            result = plan.solve(np.array([[5.0]]), allow_empty=False, method=method)
            assert result.total[0] == -np.inf

    def test_plan_from_context_reads_a_league_context(self) -> None:
        from fantasy_quant.core import LeagueContext

        ctx = LeagueContext(
            league_id=272150391,
            season=2026,
            name="Wine Wednesday",
            size=14,
            lineup_slot_counts=dict(STANDARD_SLOTS),
            slot_eligibility=SLOT_ELIGIBILITY,
            scorer=lambda stats, position: 0.0,
            playoff_team_count=6,
            playoff_weeks=(15, 16, 17),
            regular_season_weeks=tuple(range(1, 15)),
        )
        plan = plan_from_context(ctx, ROSTER)
        assert plan.n_slots == 9 and plan.laminar
        assert sorted(plan.slot_ids) == [0, 2, 2, 4, 4, 6, 16, 17, 23]


# --------------------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------------------


class TestThroughput:
    def test_the_stated_workload_runs_in_well_under_a_second(self) -> None:
        """2,000 sims x 14 weeks x 12 teams x 16 players = 5.4M cells.

        Measured on the development machine: 0.16s with the assignment, 0.09s for
        totals only. The bound here is 1.0s, which is loose enough to survive a slow
        CI box and tight enough to catch a regression that reintroduces a Python loop
        over simulations -- that would take minutes, not milliseconds.
        """
        plan = make_plan()
        rng = np.random.default_rng(5)
        scores = rng.gamma(3.0, 4.0, size=(2000, 14, 12, len(ROSTER)))
        start = time.perf_counter()
        result = plan.solve(scores, assignment=False)
        elapsed = time.perf_counter() - start
        assert result.total.shape == (2000, 14, 12)
        assert elapsed < 1.0, f"{scores.size / 1e6:.1f}M cells took {elapsed:.3f}s"

    def test_the_exact_fallback_is_slow_enough_to_be_worth_avoiding(self) -> None:
        """States the cost of a non-laminar league rather than leaving it a surprise.

        **Warm the import first.** `_exact` imports `scipy.optimize` lazily, and that
        import is ~130ms. Timed cold on a 400-row batch it reads as 330us a row and the
        fallback looks 400x slower than greedy; timed warm it is 10-13us a row, about
        35x. The module docstring quoted the cold number for a while, which turned "a
        2,000-sim season on a non-laminar league costs 3.5s" into "half a minute" and
        made the fallback sound unusable when it is merely expensive. So the first solve
        below is thrown away, and the bound is a modest 5x because the real ratio is 35x
        and greedy's fixed per-call overhead dominates at only 400 rows.
        """
        plan = make_plan()
        rng = np.random.default_rng(6)
        scores = rng.gamma(3.0, 4.0, size=(400, len(ROSTER)))
        plan.solve(scores[:8], method="exact")  # pay the scipy import outside the clock
        plan.solve(scores[:8], method="greedy", assignment=False)
        start = time.perf_counter()
        plan.solve(scores, method="greedy", assignment=False)
        fast = time.perf_counter() - start
        start = time.perf_counter()
        plan.solve(scores, method="exact")
        slow = time.perf_counter() - start
        assert slow > 5 * fast, f"greedy {fast:.4f}s against exact {slow:.4f}s"
        # And the number the caveats quote: tens of microseconds a row, not hundreds.
        assert slow / 400 < 100e-6, f"exact ran at {slow / 400 * 1e6:.0f}us a roster-week"


# --------------------------------------------------------------------------------------
# Against the corpus: does an optimal lineup actually score 121.9 +/- 24.35
# --------------------------------------------------------------------------------------

#: 12-team PPR, optimal lineups. The number this module has to be able to produce.
RESEARCH_TEAM_MEAN = 121.9
RESEARCH_TEAM_SD = 24.35

#: A 16-man roster shape, which is what all three of the user's leagues carry.
DRAFT_QUOTA: Mapping[int, int] = {QB: 2, RB: 5, WR: 5, TE: 2, K: 1, DST: 1}


def _draft_and_score(season: int, teams: int = 12) -> np.ndarray:
    """Snake a 12-team league off ESPN's frozen preseason projections, then score it.

    Real actual weekly scores, real preseason draft board, this module's lineup solver.
    A player with no row in a week -- bye, inactive, not yet signed -- scores 0, which
    is the same thing `WeeklyOutlook.zeroed()` means.
    """
    import polars as pl

    from fantasy_quant import corpus

    rows = corpus.load_stat_rows([season], root=CORPUS, variant="ppr")
    actual = rows.filter(
        (pl.col("stat_split_type_id") == corpus.SPLIT_GAME)
        & (pl.col("stat_source_id") == corpus.SOURCE_ACTUAL)
        & pl.col("scoring_period_id").is_between(1, 14)
        & pl.col("default_position_id").is_in(list(DRAFT_QUOTA))
    )
    board = rows.filter(
        (pl.col("stat_split_type_id") == corpus.SPLIT_SEASON)
        & (pl.col("stat_source_id") == corpus.SOURCE_PROJECTED)
    )
    # A usable board needs a real starter at every one of the twelve QB1 spots, not
    # just a plausible number one. Measured: 2024 runs 314 down to 256 across the top
    # fourteen; 2023 runs 270, 211, 42, 19, 18, 15 and then zeroes all the way down.
    qb12 = (
        board.filter(pl.col("default_position_id") == QB)
        .sort("applied_total", descending=True)["applied_total"]
        .to_list()
    )
    if len(qb12) < 12 or qb12[11] < 150:
        pytest.skip(f"{season} has no usable preseason season-projection board")

    rosters: dict[int, list[tuple[int, int]]] = {t: [] for t in range(teams)}
    for position, depth in DRAFT_QUOTA.items():
        ranked = (
            board.filter(pl.col("default_position_id") == position)
            .sort("applied_total", descending=True)
            .head(teams * depth)["espn_id"]
            .to_list()
        )
        for round_ in range(depth):
            picks = ranked[round_ * teams : (round_ + 1) * teams]
            if round_ % 2:  # snake
                picks = picks[::-1]
            for team, player in enumerate(picks):
                rosters[team].append((player, position))

    weeks = sorted(actual["scoring_period_id"].unique().to_list())
    points = {
        (r["espn_id"], r["scoring_period_id"]): r["applied_total"]
        for r in actual.select("espn_id", "scoring_period_id", "applied_total").iter_rows(
            named=True
        )
    }
    starting = {s: n for s, n in STANDARD_SLOTS.items() if s not in (20, 21)}
    totals = []
    for team in range(teams):
        ids = [p for p, _ in rosters[team]]
        plan = plan_from_slots(starting, SLOT_ELIGIBILITY, np.array([q for _, q in rosters[team]]))
        weekly = np.array([[points.get((p, w), 0.0) for p in ids] for w in weeks])
        totals.append(plan.solve(weekly, assignment=False).total)
    return np.concatenate(totals)


@needs_corpus
class TestAgainstMeasuredReality:
    @pytest.mark.parametrize("season", [2023, 2024])
    def test_optimal_team_scores_reproduce_the_research_anchor(self, season: int) -> None:
        """Measured here: 2024 lands at mean 123.96, SD 23.95 against 121.9 / 24.35.

        2023 skips, and the reason is worth keeping for whoever reads the corpus next:
        its only capture is dated 2023-12-31, and a season-total projection captured
        after the season is over comes back as **zero** for all but a handful of
        players -- six nonzero quarterbacks out of 133. Ranking on it drafts a league
        of backups and the weekly mean collapses to 46.7 (measured by lifting the
        `qb12` guard below). A December capture is not a draft board, and nothing in
        the file says so.
        """
        scores = _draft_and_score(season)
        assert len(scores) >= 12 * 14
        assert scores.mean() == pytest.approx(RESEARCH_TEAM_MEAN, rel=0.06)
        assert scores.std(ddof=1) == pytest.approx(RESEARCH_TEAM_SD, rel=0.12)

    def test_perfect_hindsight_beats_a_real_draft_by_about_ten_percent(self) -> None:
        """A sanity check on the anchor itself, and on where 121.9 comes from.

        Drafting on realized season totals instead of preseason projections lifts the
        same optimal-lineup machinery to ~134 points a week. The gap is the value of
        foresight, not a bug in the solver, and it is why a lineup optimizer measured
        against hindsight rosters will always look better than it is.
        """
        import polars as pl

        from fantasy_quant import corpus

        rows = corpus.load_stat_rows([2024], root=CORPUS, variant="ppr")
        actual = rows.filter(
            (pl.col("stat_split_type_id") == corpus.SPLIT_GAME)
            & (pl.col("stat_source_id") == corpus.SOURCE_ACTUAL)
            & pl.col("scoring_period_id").is_between(1, 14)
            & pl.col("default_position_id").is_in(list(DRAFT_QUOTA))
        )
        board = actual.group_by(["espn_id", "default_position_id"]).agg(
            applied_total=pl.col("applied_total").sum()
        )
        rosters: dict[int, list[tuple[int, int]]] = {t: [] for t in range(12)}
        for position, depth in DRAFT_QUOTA.items():
            ranked = (
                board.filter(pl.col("default_position_id") == position)
                .sort("applied_total", descending=True)
                .head(12 * depth)["espn_id"]
                .to_list()
            )
            for round_ in range(depth):
                picks = ranked[round_ * 12 : (round_ + 1) * 12]
                if round_ % 2:
                    picks = picks[::-1]
                for team, player in enumerate(picks):
                    rosters[team].append((player, position))
        weeks = sorted(actual["scoring_period_id"].unique().to_list())
        points = {
            (r["espn_id"], r["scoring_period_id"]): r["applied_total"]
            for r in actual.select("espn_id", "scoring_period_id", "applied_total").iter_rows(
                named=True
            )
        }
        starting = {s: n for s, n in STANDARD_SLOTS.items() if s not in (20, 21)}
        totals = []
        for team in range(12):
            ids = [p for p, _ in rosters[team]]
            plan = plan_from_slots(
                starting, SLOT_ELIGIBILITY, np.array([q for _, q in rosters[team]])
            )
            weekly = np.array([[points.get((p, w), 0.0) for p in ids] for w in weeks])
            totals.append(plan.solve(weekly, assignment=False).total)
        hindsight = np.concatenate(totals)
        assert hindsight.mean() > RESEARCH_TEAM_MEAN * 1.05
        assert hindsight.mean() < RESEARCH_TEAM_MEAN * 1.20
