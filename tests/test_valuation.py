"""Valuation tests.

The synthetic pool is not points -- it is component stat lines, scored through the
real `LeagueScoring`. That matters: the whole claim of this module is that the same
player is worth different amounts in two leagues, and a fixture that hands the
engine pre-scored points would test that claim with the interesting half removed.
Full-PPR and half-PPR outlooks here differ exactly the way two real leagues differ,
because they are the same components through two different scorers.

The slot-eligibility table is ESPN's, copied from `chui_default_platformsettings`,
and `test_slot_eligibility_matches_espn` re-reads the cached payload to catch drift.

Corpus-backed tests read the Parquet snapshot already in the repo and are skipped
if it is absent. They are offline. Only the tests marked `network` touch ESPN.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from fantasy_quant.core import (
    DST,
    QB,
    RB,
    TE,
    WR,
    K,
    LeagueContext,
    PlayerOutlook,
    WeeklyOutlook,
)
from fantasy_quant.decide.valuation import (
    DEFAULT_BENCH_HOARDING,
    FLEX_IDENTIFICATION_TOLERANCE,
    MARKET_INFORMATIVE_RANGE,
    MARKET_METRICS,
    MarketQuote,
    PositionDemand,
    ScarcityCurve,
    ValuationError,
    bench_hoarding_from_rosters,
    build_replacement_model,
    fit_scarcity_curve,
    market_disagreements,
    player_values,
    position_demand,
    positive_vorp_share,
    remaining_weeks,
    replacement_levels,
    rostered_per_team,
    scarcity_curves,
    solve_flex_shares,
    starting_shape,
    uniform_shares,
    value_league,
)
from fantasy_quant.espn.scoring import LeagueScoring

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data/snapshots/espn/season=2026/variant=ppr/2026-09-07.parquet"
REFERENCE = REPO / "data/reference"

# lineupSlotId -> eligible defaultPositionIds, from ESPN's own platform settings.
# 4 is WR here and TE in the position space; 23 is FLEX and 7 is OP/superflex.
SLOT_ELIGIBILITY: Mapping[int, frozenset[int]] = {
    0: frozenset({QB}),
    2: frozenset({RB}),
    4: frozenset({WR}),
    5: frozenset({WR, TE}),
    6: frozenset({TE}),
    7: frozenset({QB, RB, WR, TE}),
    16: frozenset({DST}),
    17: frozenset({K}),
    20: frozenset({QB, RB, WR, TE, K, DST}),
    21: frozenset({QB, RB, WR, TE, K, DST}),
    23: frozenset({RB, WR, TE}),
}

STANDARD_SLOTS: Mapping[int, int] = {0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 23: 1, 20: 7, 21: 1}

REGULAR_WEEKS = tuple(range(1, 15))
PLAYOFF_WEEKS = (15, 16, 17)

# --- Scoring. Two leagues that differ in exactly one number, the way the user's do.
_SHARED_ITEMS = [
    {"statId": 3, "points": 0.04},  # passing yards
    {"statId": 4, "points": 4.0},  # passing TD
    {"statId": 24, "points": 0.1},  # rushing yards
    {"statId": 25, "points": 6.0},  # rushing TD
    {"statId": 42, "points": 0.1},  # receiving yards
    {"statId": 43, "points": 6.0},  # receiving TD
    {"statId": 83, "points": 3.0},  # field goals made
    {"statId": 86, "points": 1.0},  # extra points made
    {"statId": 95, "points": 2.0},  # defensive interceptions
    {"statId": 99, "points": 1.0},  # sacks
]


def _scoring(points_per_reception: float) -> LeagueScoring:
    return LeagueScoring.from_settings(
        {"scoringItems": [*_SHARED_ITEMS, {"statId": 53, "points": points_per_reception}]}
    )


FULL_PPR = _scoring(1.0)
HALF_PPR = _scoring(0.5)


# --------------------------------------------------------------------------------------
# Synthetic pool
# --------------------------------------------------------------------------------------

#: (position, players, curve intercept, decay). Shaped so RB and WR cross near the
#: flex margin, which is what makes the fixed point do work rather than pick a winner
#: on the first pass.
_POOL_SHAPE = (
    (QB, 40, 24.0, 0.030),
    (RB, 90, 22.0, 0.026),
    (WR, 150, 17.0, 0.013),
    (TE, 80, 15.0, 0.039),
    (K, 40, 9.5, 0.010),
    (DST, 32, 8.0, 0.020),
)


def _components(position_id: int, strength: float) -> dict[str, float]:
    """A stat line worth roughly `strength` points, split the way the position scores."""
    if position_id == QB:
        return {"3": strength * 20.0, "4": strength * 0.05}
    if position_id == RB:
        return {
            "24": strength * 7.0,
            "25": strength * 0.04,
            "42": strength * 1.5,
            "53": strength * 0.15,
        }
    if position_id in (WR, TE):
        rec = 0.28 if position_id == WR else 0.30
        return {"42": strength * 7.0, "43": strength * 0.035, "53": strength * rec}
    if position_id == K:
        return {"83": strength * 0.25, "86": strength * 0.25}
    return {"99": strength * 0.4, "95": strength * 0.3}


def _wobble(index: int, week: int) -> float:
    """Deterministic +/-15% weekly variation. No RNG, so every run is identical."""
    return 1.0 + 0.15 * math.sin(0.7 * week + 0.37 * index)


def synthetic_outlooks(
    scorer: Callable[[Mapping[str, float], int], float],
    *,
    weeks: tuple[int, ...] = REGULAR_WEEKS + PLAYOFF_WEEKS,
    byes: bool = True,
) -> list[PlayerOutlook]:
    """A whole league's projection universe, scored by one league's rules."""
    out: list[PlayerOutlook] = []
    for position_id, count, a, b in _POOL_SHAPE:
        for rank in range(1, count + 1):
            index = position_id * 1000 + rank
            player_id = position_id * 10_000 + rank
            base = a * math.exp(-b * rank)
            bye = 5 + (rank % 10) if byes else None
            weekly: dict[int, WeeklyOutlook] = {}
            for week in weeks:
                playing = week != bye
                mean = (
                    scorer(_components(position_id, base * _wobble(index, week)), position_id)
                    if playing
                    else 0.0
                )
                weekly[week] = WeeklyOutlook(
                    player_id=player_id,
                    season=2026,
                    week=week,
                    position_id=position_id,
                    mean=mean,
                    sd=3.67 + 0.273 * mean,
                    p_zero=0.2 if playing else 1.0,
                    shape=1.5,
                    scale=max(mean, 0.01),
                    pro_team_id=1 + (rank % 32),
                    playing=playing,
                )
            out.append(
                PlayerOutlook(
                    player_id=player_id,
                    name=f"{position_id}-{rank:03d}",
                    position_id=position_id,
                    pro_team_id=1 + (rank % 32),
                    weeks=weekly,
                )
            )
    return out


def make_context(
    *,
    size: int = 12,
    slots: Mapping[int, int] | None = None,
    scorer: Callable[[Mapping[str, float], int], float] = HALF_PPR,
    league_id: int = 1,
    name: str = "test",
) -> LeagueContext:
    return LeagueContext(
        league_id=league_id,
        season=2026,
        name=name,
        size=size,
        lineup_slot_counts=dict(slots or STANDARD_SLOTS),
        slot_eligibility=SLOT_ELIGIBILITY,
        scorer=scorer,
        playoff_team_count=6,
        playoff_weeks=PLAYOFF_WEEKS,
        regular_season_weeks=REGULAR_WEEKS,
    )


@pytest.fixture(scope="module")
def half_ppr_outlooks() -> list[PlayerOutlook]:
    return synthetic_outlooks(HALF_PPR)


@pytest.fixture(scope="module")
def full_ppr_outlooks() -> list[PlayerOutlook]:
    return synthetic_outlooks(FULL_PPR)


# --------------------------------------------------------------------------------------
# League shape
# --------------------------------------------------------------------------------------


class TestStartingShape:
    def test_dedicated_and_flex_are_derived_not_hardcoded(self) -> None:
        ctx = make_context()
        dedicated, flex = starting_shape(ctx, (QB, RB, WR, TE, K, DST))
        assert dedicated == {QB: 1.0, RB: 2.0, WR: 2.0, TE: 1.0, DST: 1.0, K: 1.0}
        assert flex == {23: (1.0, frozenset({RB, WR, TE}))}

    def test_bench_and_ir_are_not_starting_slots(self) -> None:
        ctx = make_context()
        dedicated, flex = starting_shape(ctx, (QB, RB, WR, TE, K, DST))
        assert 20 not in flex and 21 not in flex
        assert sum(dedicated.values()) + sum(c for c, _ in flex.values()) == 9

    def test_superflex_is_a_flex_slot_not_a_second_qb_slot(self) -> None:
        ctx = make_context(slots={**STANDARD_SLOTS, 7: 1})
        dedicated, flex = starting_shape(ctx, (QB, RB, WR, TE, K, DST))
        assert dedicated[QB] == 1.0
        assert flex[7] == (1.0, frozenset({QB, RB, WR, TE}))

    def test_a_slot_that_accepts_no_projected_position_is_dropped(self) -> None:
        # An IDP slot in a league where we project no defensive players.
        ctx = make_context(slots={**STANDARD_SLOTS, 10: 2})
        ctx = LeagueContext(
            **{
                **{f.name: getattr(ctx, f.name) for f in ctx.__dataclass_fields__.values()},
                "slot_eligibility": {**SLOT_ELIGIBILITY, 10: frozenset({11})},
            }
        )
        dedicated, flex = starting_shape(ctx, (QB, RB, WR, TE, K, DST))
        assert 11 not in dedicated
        assert 10 not in flex

    def test_uniform_shares_sum_to_one_per_slot(self) -> None:
        ctx = make_context(slots={**STANDARD_SLOTS, 7: 1})
        _, flex = starting_shape(ctx, (QB, RB, WR, TE, K, DST))
        shares = uniform_shares(flex)
        for slot_shares in shares.values():
            assert sum(slot_shares.values()) == pytest.approx(1.0)

    @pytest.mark.skipif(not (REFERENCE / "platform_settings_2026.json").exists(), reason="no cache")
    def test_slot_eligibility_matches_espn(self) -> None:
        """The fixture is ESPN's table; this fails if ESPN moves and we do not."""
        from fantasy_quant.espn.constants import load_platform_settings, slot_id

        settings = load_platform_settings(2026, root=REFERENCE, offline=True)
        for slot, expected in SLOT_ELIGIBILITY.items():
            live = settings.slots[slot_id(slot)]
            assert {int(p) for p in live.eligible_positions} >= expected, slot


# --------------------------------------------------------------------------------------
# The fixed point
# --------------------------------------------------------------------------------------


class TestFlexFixedPoint:
    def test_converges(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        solution = solve_flex_shares(ctx, half_ppr_outlooks, weeks)
        assert solution.converged
        assert solution.iterations <= 8, "research says 3-4 passes; 8 is already generous"
        assert sum(solution.shares[23].values()) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "guess",
        [
            None,
            {23: {RB: 1.0, WR: 0.0, TE: 0.0}},
            {23: {RB: 0.0, WR: 1.0, TE: 0.0}},
            {23: {RB: 0.0, WR: 0.0, TE: 1.0}},
            {23: {RB: 0.34, WR: 0.33, TE: 0.33}},
        ],
    )
    def test_stable_to_the_starting_guess(
        self,
        half_ppr_outlooks: list[PlayerOutlook],
        guess: Mapping[int, Mapping[int, float]] | None,
    ) -> None:
        """A fixed point that remembers its guess is not a fixed point."""
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        reference = solve_flex_shares(ctx, half_ppr_outlooks, weeks).shares[23]
        solution = solve_flex_shares(ctx, half_ppr_outlooks, weeks, initial_shares=guess)
        assert solution.converged
        for position, share in reference.items():
            assert solution.shares[23][position] == pytest.approx(share, abs=0.01)

    def test_stable_to_the_starting_guess_when_damped(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        undamped = solve_flex_shares(ctx, half_ppr_outlooks, weeks, damping=1.0)
        damped = solve_flex_shares(
            ctx, half_ppr_outlooks, weeks, damping=0.4, initial_shares={23: {TE: 1.0}}
        )
        assert damped.iterations > undamped.iterations
        for position, share in undamped.shares[23].items():
            assert damped.shares[23][position] == pytest.approx(share, abs=0.01)

    def test_zero_bench_hoarding_saturates_and_falls_back(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """With no bench, every survivor starts and the pool cannot select. See the docstring."""
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        a = solve_flex_shares(ctx, half_ppr_outlooks, weeks, bench_hoarding={})
        b = solve_flex_shares(
            ctx, half_ppr_outlooks, weeks, bench_hoarding={}, initial_shares={23: {TE: 1.0}}
        )
        assert a.saturated and b.saturated
        # The open-pool fallback is what keeps it guess-independent; without it the
        # degenerate branch would hand back {TE: 1.0} and look perfectly converged.
        assert b.shares[23][TE] < 0.05
        for position, share in a.shares[23].items():
            assert b.shares[23][position] == pytest.approx(share, abs=0.01)

    def test_bench_hoarding_makes_the_pool_selective(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        assert not solve_flex_shares(ctx, half_ppr_outlooks, weeks).saturated

    def test_rejects_a_meaningless_damping(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        ctx = make_context()
        with pytest.raises(ValueError, match="damping"):
            solve_flex_shares(ctx, half_ppr_outlooks, remaining_weeks(ctx, 1), damping=0.0)

    def test_flex_share_responds_to_scoring(
        self, half_ppr_outlooks: list[PlayerOutlook], full_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """Full PPR pushes receivers into the flex. Same slots, different equilibrium."""
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        half = solve_flex_shares(ctx, half_ppr_outlooks, weeks).shares[23]
        full = solve_flex_shares(ctx, full_ppr_outlooks, weeks).shares[23]
        assert full[WR] > half[WR] + 0.05
        assert full[RB] < half[RB] - 0.05

    def test_no_flex_slots_means_no_iteration(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        ctx = make_context(slots={0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 20: 7})
        solution = solve_flex_shares(ctx, half_ppr_outlooks, remaining_weeks(ctx, 1))
        assert solution.shares == {} and solution.converged and solution.iterations == 0


class TestFlexIdentification:
    """`converged` is not the acceptance test; `identified` is.

    The measurement map degrades continuously into the identity as bench hoarding
    shrinks -- with a thin bench every rostered survivor starts, so the count of who
    filled the flex is just a transcription of the pool, which was built from the
    guess. The identity converges on pass one at whatever it was handed and reports
    `converged=True`. These tests exist because that is invisible from the outside.
    """

    THIN = {RB: 0.033, WR: 0.033, TE: 0.033}  # sum 0.1 against a 12-team 1-FLEX league

    def test_a_thin_bench_cannot_identify_alpha_and_says_so(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        solution = solve_flex_shares(ctx, half_ppr_outlooks, weeks, bench_hoarding=self.THIN)
        assert not solution.identified
        assert solution.guess_spread > FLEX_IDENTIFICATION_TOLERANCE
        # Flagged *and* repaired: the reported shares came from the open pool.
        assert solution.saturated

    @pytest.mark.parametrize(
        "guess",
        [None, {23: {RB: 1.0}}, {23: {WR: 1.0}}, {23: {TE: 1.0}}, {23: {RB: 0.6, WR: 0.4}}],
    )
    def test_the_repaired_answer_is_the_same_from_every_guess(
        self,
        half_ppr_outlooks: list[PlayerOutlook],
        guess: Mapping[int, Mapping[int, float]] | None,
    ) -> None:
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        reference = solve_flex_shares(ctx, half_ppr_outlooks, weeks, bench_hoarding=self.THIN)
        solution = solve_flex_shares(
            ctx, half_ppr_outlooks, weeks, bench_hoarding=self.THIN, initial_shares=guess
        )
        for position, share in reference.shares[23].items():
            assert solution.shares[23][position] == pytest.approx(share, abs=1e-12)

    def test_the_guard_is_load_bearing_not_decorative(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """Switch it off and the same call hands the guess straight back.

        If this ever stops failing to reproduce the guess, the degeneracy is gone and
        the guard can go with it. Until then it is the only thing standing between a
        thin-bench league and a valuation built on the caller's arbitrary prior.
        """
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        unguarded = solve_flex_shares(
            ctx,
            half_ppr_outlooks,
            weeks,
            bench_hoarding=self.THIN,
            initial_shares={23: {RB: 1.0, WR: 0.0, TE: 0.0}},
            identification_tolerance=math.inf,
        )
        assert unguarded.converged, "the degenerate map converges instantly -- that is the trap"
        assert unguarded.shares[23][RB] == pytest.approx(1.0), "it echoed the guess"

        guarded = solve_flex_shares(
            ctx,
            half_ppr_outlooks,
            weeks,
            bench_hoarding=self.THIN,
            initial_shares={23: {RB: 1.0, WR: 0.0, TE: 0.0}},
        )
        assert guarded.shares[23][RB] < 0.9

    def test_a_real_bench_identifies_alpha(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        """The research prior and a measured seven-man bench both survive the check."""
        ctx = make_context()
        weeks = remaining_weeks(ctx, 1)
        for beta in (None, {RB: 2.4, WR: 3.1, TE: 0.6, QB: 0.7, K: 0.1, DST: 0.1}):
            solution = solve_flex_shares(ctx, half_ppr_outlooks, weeks, bench_hoarding=beta)
            assert solution.identified, beta
            assert solution.guess_spread <= FLEX_IDENTIFICATION_TOLERANCE
            assert not solution.saturated

    def test_no_bench_at_all_is_saturated_but_still_identified(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """beta=0 takes the open pool by the necessary condition, before any probing."""
        ctx = make_context()
        solution = solve_flex_shares(
            ctx, half_ppr_outlooks, remaining_weeks(ctx, 1), bench_hoarding={}
        )
        assert solution.saturated and solution.identified
        assert solution.guess_spread == 0.0


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


class TestBaselines:
    def test_demand_is_the_formula(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        ctx = make_context()
        model = build_replacement_model(ctx, half_ppr_outlooks)
        rb = model.demand(RB)
        assert rb is not None
        share = model.flex.share(23, RB)
        assert rb.dedicated_slots == 2.0
        assert rb.flex_slots == pytest.approx(share)
        assert rb.bench_hoarding == DEFAULT_BENCH_HOARDING[RB]
        assert rb.rostered == pytest.approx(12 * (2.0 + share + 0.5))
        assert rb.starters == pytest.approx(12 * (2.0 + share))

    def test_pure_vols_baselines_are_the_starting_slots(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """With beta=0 the QB and TE baselines must be exactly one per team."""
        ctx = make_context()
        model = build_replacement_model(ctx, half_ppr_outlooks, bench_hoarding={})
        assert model.levels[QB].demand.rostered == pytest.approx(12.0)
        assert model.levels[TE].demand.rostered == pytest.approx(12.0, abs=0.5)
        total_flex = sum(
            model.levels[p].demand.flex_slots for p in (RB, WR, TE) if p in model.levels
        )
        assert total_flex == pytest.approx(1.0)

    def test_fourteen_teams_are_deeper_than_twelve(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        twelve = build_replacement_model(make_context(size=12), half_ppr_outlooks)
        fourteen = build_replacement_model(make_context(size=14, league_id=2), half_ppr_outlooks)
        for position in (QB, RB, WR, TE, K, DST):
            assert (
                fourteen.levels[position].demand.rostered > twelve.levels[position].demand.rostered
            ), position
            # Deeper demand means a worse player is freely available.
            assert fourteen.levels[position].per_week < twelve.levels[position].per_week, position

    def test_superflex_moves_the_qb_baseline_sharply(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        base = build_replacement_model(make_context(), half_ppr_outlooks)
        flexed = build_replacement_model(
            make_context(slots={**STANDARD_SLOTS, 7: 1}, league_id=3), half_ppr_outlooks
        )
        op = flexed.flex.shares[7]
        assert op[QB] > 0.8, "a superflex is a quarterback slot in practice"
        assert op[QB] > 5 * max(op[p] for p in (RB, WR, TE))
        # 12 teams x one more quarterback each: the baseline should roughly double.
        assert flexed.levels[QB].demand.rostered > base.levels[QB].demand.rostered + 10
        assert flexed.levels[QB].per_week < base.levels[QB].per_week * 0.85
        # Everything else moves only by the sliver of the OP slot it wins, which is
        # an order of magnitude smaller. K and D/ST are not flex-eligible at all.
        qb_shift = flexed.levels[QB].demand.rostered - base.levels[QB].demand.rostered
        for position in (RB, WR, TE):
            shift = abs(
                flexed.levels[position].demand.rostered - base.levels[position].demand.rostered
            )
            assert shift < 0.25 * qb_shift, position
        for position in (K, DST):
            assert flexed.levels[position].demand.rostered == pytest.approx(
                base.levels[position].demand.rostered
            ), position

    def test_replacement_level_varies_by_week(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        """Byes thin a position; a single season-average baseline hides that."""
        model = build_replacement_model(make_context(), half_ppr_outlooks)
        weekly = list(model.levels[WR].by_week.values())
        assert len(set(round(v, 6) for v in weekly)) > 1
        assert min(weekly) < max(weekly)

    def test_fractional_baseline_interpolates_between_the_right_two_players(self) -> None:
        """N_q = 3.4 means 60% of WR3 and 40% of WR4, in that order.

        Worth pinning by hand: the module's whole argument for interpolating rather
        than rounding is that value moves smoothly across the baseline, and an
        interpolation running the wrong way between the same two players is
        invisible in every aggregate the other tests look at.
        """
        means = [10.0, 8.0, 5.0, 4.0, 1.0]
        outlooks = [
            PlayerOutlook(
                player_id=500 + i,
                name=f"wr{i}",
                position_id=WR,
                pro_team_id=1,
                weeks={
                    1: WeeklyOutlook(
                        player_id=500 + i,
                        season=2026,
                        week=1,
                        position_id=WR,
                        mean=mu,
                        sd=1.0,
                        p_zero=0.1,
                        shape=1.0,
                        scale=1.0,
                    )
                },
            )
            for i, mu in enumerate(means)
        ]
        ctx = make_context(size=10)

        def level_at(rostered: float):
            demand = {
                WR: PositionDemand(
                    position_id=WR,
                    teams=10,
                    dedicated_slots=rostered / 10.0,
                    flex_slots=0.0,
                    bench_hoarding=0.0,
                )
            }
            return replacement_levels(ctx, outlooks, [1], demand)[WR]

        # 5 + 0.4 * (4 - 5) = 4.6. Interpolating the other way would give 4.4.
        assert level_at(3.4).points(1) == pytest.approx(4.6)
        assert level_at(3.0).points(1) == pytest.approx(5.0)
        assert level_at(4.0).points(1) == pytest.approx(4.0)
        assert level_at(1.5).points(1) == pytest.approx(9.0)
        # Above the pool the baseline pins to the worst projection and says so;
        # landing exactly on the last player is a resolved answer, not a pinned one.
        assert not level_at(5.0).supply_limited
        assert level_at(5.0).points(1) == pytest.approx(1.0)
        assert level_at(6.0).supply_limited
        assert level_at(6.0).points(1) == pytest.approx(1.0)

    def test_supply_limited_is_flagged_not_silently_zero(self) -> None:
        """A pool shallower than the demand must say so rather than return 0.0."""
        ctx = make_context(size=40)
        outlooks = synthetic_outlooks(HALF_PPR)
        model = build_replacement_model(ctx, outlooks)
        assert model.levels[K].supply_limited
        assert model.levels[K].per_week > 0.0

    def test_needs_projections(self) -> None:
        with pytest.raises(ValuationError, match="no projections"):
            build_replacement_model(make_context(), [])

    def test_needs_weeks(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        with pytest.raises(ValuationError, match="no weeks"):
            build_replacement_model(make_context(), half_ppr_outlooks, from_week=99)

    def test_from_week_shortens_the_horizon(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        late = build_replacement_model(make_context(), half_ppr_outlooks, from_week=12)
        assert late.weeks == (12, 13, 14, 15, 16, 17)


# --------------------------------------------------------------------------------------
# VORP
# --------------------------------------------------------------------------------------


class TestPlayerValues:
    def test_vorp_is_points_minus_the_weekly_baseline(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        model = build_replacement_model(ctx, half_ppr_outlooks)
        values = player_values(half_ppr_outlooks, model, playoff_weeks=PLAYOFF_WEEKS)
        best = values[0]
        outlook = next(o for o in half_ppr_outlooks if o.player_id == best.player_id)
        level = model.levels[best.position_id]
        expected = sum(
            (outlook.weeks[w].mean if outlook.weeks[w].playing else 0.0) - level.points(w)
            for w in model.weeks
        )
        assert best.ros_vorp == pytest.approx(expected)

    def test_playoff_vorp_covers_only_the_playoff_weeks(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        model = build_replacement_model(ctx, half_ppr_outlooks)
        values = player_values(half_ppr_outlooks, model, playoff_weeks=PLAYOFF_WEEKS)
        assert all(v.playoff_weeks <= 3 for v in values)
        assert all(v.ros_weeks == 17 for v in values)
        assert values[0].playoff_vorp < values[0].ros_vorp

    def test_a_player_who_misses_the_playoffs_is_not_the_same_asset(self) -> None:
        """The point of doing this per week: equal season totals, opposite playoff value."""
        weeks = REGULAR_WEEKS + PLAYOFF_WEEKS
        elite_weeks = tuple(range(1, 12))

        def build(pid: int, name: str, per_week: float, playing_weeks: tuple[int, ...]):
            return PlayerOutlook(
                player_id=pid,
                name=name,
                position_id=WR,
                pro_team_id=1,
                weeks={
                    w: WeeklyOutlook(
                        player_id=pid,
                        season=2026,
                        week=w,
                        position_id=WR,
                        mean=per_week if w in playing_weeks else 0.0,
                        sd=1.0,
                        p_zero=0.2 if w in playing_weeks else 1.0,
                        shape=1.0,
                        scale=1.0,
                        playing=w in playing_weeks,
                    )
                    for w in weeks
                },
            )

        pool = synthetic_outlooks(HALF_PPR)
        # Equal rest-of-season points: 11 x 25.5 = 280.5 vs 17 x 16.5 = 280.5.
        pool.append(build(90_001, "elite-then-gone", 25.5, elite_weeks))
        pool.append(build(90_002, "steady", 16.5, weeks))
        report = value_league(make_context(), pool)
        gone = report.value_for(90_001)
        steady = report.value_for(90_002)
        assert gone is not None and steady is not None
        assert gone.ros_points == pytest.approx(steady.ros_points, abs=0.01)
        assert gone.playoff_vorp < 0 < steady.playoff_vorp
        assert gone.playoff_vorp < steady.playoff_vorp - 20

    def test_an_omitted_week_is_unknown_and_an_absent_week_is_charged(self) -> None:
        """Two different facts, deliberately not collapsed into one."""
        weeks = REGULAR_WEEKS + PLAYOFF_WEEKS
        played = tuple(range(1, 12))

        def build(pid: int, present: tuple[int, ...], playing: tuple[int, ...]):
            return PlayerOutlook(
                player_id=pid,
                name=str(pid),
                position_id=WR,
                pro_team_id=1,
                weeks={
                    w: WeeklyOutlook(
                        player_id=pid,
                        season=2026,
                        week=w,
                        position_id=WR,
                        mean=25.5 if w in playing else 0.0,
                        sd=1.0,
                        p_zero=0.2 if w in playing else 1.0,
                        shape=1.0,
                        scale=1.0,
                        playing=w in playing,
                    )
                    for w in present
                },
            )

        pool = synthetic_outlooks(HALF_PPR)
        pool.append(build(90_003, played, played))  # weeks 12-17 simply absent
        pool.append(build(90_004, weeks, played))  # weeks 12-17 present, not playing
        report = value_league(make_context(), pool)
        unknown = report.value_for(90_003)
        benched = report.value_for(90_004)
        assert unknown is not None and benched is not None
        assert unknown.ros_weeks == 11
        assert benched.ros_weeks == 17
        assert unknown.ros_vorp > benched.ros_vorp

    def test_positive_vorp_share_sums_to_one_and_ignores_the_tail(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        model = build_replacement_model(ctx, half_ppr_outlooks)
        values = player_values(half_ppr_outlooks, model, playoff_weeks=PLAYOFF_WEEKS)
        share = positive_vorp_share(values)
        assert sum(share.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in share.values())
        # Kickers and defenses are nearly interchangeable; the skill positions own
        # essentially the whole scarcity budget.
        assert share[RB] + share[WR] > 0.7
        assert share.get(K, 0.0) < 0.05

    def test_values_are_sorted_and_deterministic(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        first = value_league(ctx, half_ppr_outlooks)
        second = value_league(ctx, half_ppr_outlooks)
        assert [v.player_id for v in first.values] == [v.player_id for v in second.values]
        assert all(
            a.ros_vorp >= b.ros_vorp for a, b in zip(first.values, first.values[1:], strict=False)
        )


# --------------------------------------------------------------------------------------
# The thing that must not regress: values are per league
# --------------------------------------------------------------------------------------


class TestLeagueSpecificity:
    def test_ppr_moves_receivers_and_leaves_kickers_alone(
        self, half_ppr_outlooks: list[PlayerOutlook], full_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """Same size, same slots; only the per-reception value differs."""
        half = value_league(make_context(scorer=HALF_PPR), half_ppr_outlooks)
        full = value_league(make_context(scorer=FULL_PPR, league_id=2), full_ppr_outlooks)

        for position in (WR, TE):
            ranked = [a for a in half.values if a.position_id == position]
            # Every startable receiver moves by points, not by rounding. Below the
            # baseline VORP is ~0 in both leagues by construction, so the top of the
            # position is where the claim has content.
            for a in ranked[:10]:
                b = full.value_for(a.player_id)
                assert b is not None
                assert abs(a.ros_vorp - b.ros_vorp) > 1.0, a.name
            # And nothing at the position survives unchanged.
            assert not any(
                full.value_for(a.player_id).ros_vorp == pytest.approx(a.ros_vorp, abs=1e-9)
                for a in ranked
            )

        for a in [v for v in half.values if v.position_id == K]:
            b = full.value_for(a.player_id)
            assert b is not None
            assert b.ros_vorp == pytest.approx(a.ros_vorp), "kickers do not catch passes"
        for a in [v for v in half.values if v.position_id == DST]:
            b = full.value_for(a.player_id)
            assert b is not None
            assert b.ros_vorp == pytest.approx(a.ros_vorp)

    def test_two_leagues_never_share_a_number(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        """The regression guard for a globally cached valuation.

        A module-level cache, a memo keyed on player id, or a baseline computed once
        and reused would all make these two reports agree. Interleaved deliberately,
        so an order-dependent cache fails here too.
        """
        small = make_context(size=12, league_id=101, name="small")
        big = make_context(size=14, league_id=102, name="big")

        first_small = value_league(small, half_ppr_outlooks)
        big_report = value_league(big, half_ppr_outlooks)
        second_small = value_league(small, half_ppr_outlooks)

        assert first_small.replacement.baseline_ranks() != big_report.replacement.baseline_ranks()
        for position in (QB, RB, WR, TE, K, DST):
            assert first_small.replacement.levels[position].per_week != pytest.approx(
                big_report.replacement.levels[position].per_week
            ), position

        identical = 0
        for value in first_small.values[:100]:
            other = big_report.value_for(value.player_id)
            assert other is not None
            if other.ros_vorp == pytest.approx(value.ros_vorp, abs=1e-9):
                identical += 1
        assert identical == 0, f"{identical} of the top 100 carried the same value into a 14-team"

        # Re-running the first league must reproduce it exactly: differing is a bug
        # in the other direction (leaked state between calls).
        for a, b in zip(first_small.values, second_small.values, strict=True):
            assert a == b

    def test_the_same_settings_give_the_same_answer(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """Two of the user's leagues are identical in shape; they must agree exactly."""
        a = value_league(make_context(league_id=161496047, name="Baddies"), half_ppr_outlooks)
        b = value_league(make_context(league_id=634537479, name="Type shi"), half_ppr_outlooks)
        assert a.replacement.baseline_ranks() == b.replacement.baseline_ranks()
        for left, right in zip(a.values, b.values, strict=True):
            assert left.player_id == right.player_id
            assert left.ros_vorp == pytest.approx(right.ros_vorp)


# --------------------------------------------------------------------------------------
# Scarcity
# --------------------------------------------------------------------------------------


class TestScarcityCurve:
    @pytest.mark.parametrize(
        ("a", "b"), [(25.0, 0.040), (22.0, 0.026), (17.0, 0.013), (15.0, 0.039)]
    )
    def test_recovers_known_coefficients(self, a: float, b: float) -> None:
        points = [a * math.exp(-b * rank) for rank in range(1, 61)]
        curve = fit_scarcity_curve(points, RB)
        assert curve.a == pytest.approx(a, rel=1e-6)
        assert curve.b == pytest.approx(b, rel=1e-6)
        assert curve.r2 == pytest.approx(1.0)
        assert curve.half_life == pytest.approx(math.log(2) / b)
        assert curve.n == 60

    def test_recovers_coefficients_through_noise(self) -> None:
        points = [
            22.0 * math.exp(-0.026 * rank) * (1.0 + 0.05 * math.sin(rank)) for rank in range(1, 61)
        ]
        curve = fit_scarcity_curve(points, RB)
        assert curve.b == pytest.approx(0.026, rel=0.05)
        assert curve.r2 > 0.98

    def test_window_is_reported_because_the_fit_depends_on_it(self) -> None:
        points = [22.0 * math.exp(-0.026 * rank) for rank in range(1, 121)]
        assert fit_scarcity_curve(points, RB, ranks=30).n == 30
        assert fit_scarcity_curve(points, RB, ranks=120).n == 120

    def test_predict_round_trips(self) -> None:
        curve = ScarcityCurve(position_id=WR, a=17.0, b=0.013, r2=1.0, n=60)
        assert curve.predict(1) == pytest.approx(17.0 * math.exp(-0.013))
        assert curve.half_life == pytest.approx(53.3, abs=0.1)

    def test_flat_curve_has_infinite_half_life(self) -> None:
        curve = ScarcityCurve(position_id=WR, a=10.0, b=0.0, r2=1.0, n=10)
        assert curve.half_life == math.inf

    def test_refuses_to_fit_almost_nothing(self) -> None:
        with pytest.raises(ValuationError, match="at least 3"):
            fit_scarcity_curve([10.0, 5.0], WR)

    def test_per_game_divides_by_games_played_not_by_the_horizon(self) -> None:
        """A bye is not scarcity. Two identical receivers, one with a week off.

        `per_game` must divide by the weeks a player actually plays; dividing by the
        horizon reads a bye as a 6% talent gap and bends the curve. Ten identical
        players, half of them on a bye, must fit a perfectly flat curve.
        """
        weeks = tuple(range(1, 18))

        def wr(pid: int, bye: int | None):
            return PlayerOutlook(
                player_id=pid,
                name=str(pid),
                position_id=WR,
                pro_team_id=1,
                weeks={
                    w: WeeklyOutlook(
                        player_id=pid,
                        season=2026,
                        week=w,
                        position_id=WR,
                        mean=10.0 if w != bye else 0.0,
                        sd=1.0,
                        p_zero=0.1 if w != bye else 1.0,
                        shape=1.0,
                        scale=1.0,
                        playing=w != bye,
                    )
                    for w in weeks
                },
            )

        pool = [wr(600 + i, 5 if i % 2 else None) for i in range(10)]
        flat = scarcity_curves(pool, weeks, ranks=10)[WR]
        assert flat.b == pytest.approx(0.0, abs=1e-12)
        assert flat.half_life == math.inf
        # Season totals are a different question and must show the missing week.
        seasonal = scarcity_curves(pool, weeks, ranks=10, per_game=False)[WR]
        assert seasonal.b > 0.0

    def test_wr_curve_is_flatter_than_qb_and_te(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """The quantitative core of late-round-QB, on our own numbers."""
        curves = scarcity_curves(half_ppr_outlooks, REGULAR_WEEKS + PLAYOFF_WEEKS)
        assert curves[WR].b < curves[QB].b
        assert curves[WR].b < curves[TE].b
        assert curves[WR].half_life > curves[TE].half_life


# --------------------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------------------


class TestMarket:
    def _quotes(self, values, *, offset: int = 0) -> dict[int, MarketQuote]:
        return {
            v.player_id: MarketQuote(
                player_id=v.player_id, adp=float(i + 1 + offset), percent_owned=100.0 - i * 0.2
            )
            for i, v in enumerate(values)
        }

    def test_agreement_produces_no_disagreement(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        report = value_league(make_context(), half_ppr_outlooks)
        quotes = self._quotes(report.values)
        edges = market_disagreements(
            report.values, quotes, metric="adp", use_informative_range=False
        )
        assert len(edges) == len(report.values)
        assert all(e.rank_delta == 0 for e in edges)

    def test_positive_delta_means_we_like_him_more(self) -> None:
        values = value_league(make_context(), synthetic_outlooks(HALF_PPR)).values
        best, worst = values[0], values[40]
        quotes = {
            best.player_id: MarketQuote(player_id=best.player_id, adp=50.0),
            worst.player_id: MarketQuote(player_id=worst.player_id, adp=1.0),
        }
        edges = {
            e.player_id: e
            for e in market_disagreements(values, quotes, metric="adp", use_informative_range=False)
        }
        assert edges[best.player_id].rank_delta > 0 and edges[best.player_id].is_buy
        assert edges[worst.player_id].rank_delta < 0 and not edges[worst.player_id].is_buy

    def test_censored_adp_is_excluded_by_default(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """ESPN parks every undrafted player at ~170; ranking inside that is noise."""
        report = value_league(make_context(), half_ppr_outlooks)
        ceiling = MARKET_INFORMATIVE_RANGE["adp"][1]
        assert ceiling is not None
        quotes = {
            v.player_id: MarketQuote(
                player_id=v.player_id, adp=float(i + 1) if i < 150 else 169.9 + (i % 17) * 0.1
            )
            for i, v in enumerate(report.values)
        }
        filtered = market_disagreements(report.values, quotes, metric="adp")
        unfiltered = market_disagreements(
            report.values, quotes, metric="adp", use_informative_range=False
        )
        assert len(filtered) == 150
        assert len(unfiltered) == len(report.values)
        assert max(abs(e.rank_delta) for e in unfiltered) > max(abs(e.rank_delta) for e in filtered)

    def test_percent_owned_is_ranked_the_other_way(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        report = value_league(make_context(), half_ppr_outlooks)
        quotes = {
            v.player_id: MarketQuote(player_id=v.player_id, percent_owned=1.0 + i * 0.5)
            for i, v in enumerate(report.values[:50])
        }
        edges = market_disagreements(report.values, quotes, metric="percent_owned")
        # Our #1 is the field's least-rostered player, so he is the biggest buy.
        assert edges[0].player_id == report.values[0].player_id
        assert edges[0].rank_delta > 0

    def test_positions_filter_ranks_within_the_position(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        report = value_league(make_context(), half_ppr_outlooks)
        quotes = self._quotes(report.values)
        edges = market_disagreements(
            report.values, quotes, metric="adp", positions=(WR,), use_informative_range=False
        )
        assert edges and all(e.position_id == WR for e in edges)
        assert max(e.our_rank for e in edges) == len(edges)

    def test_unknown_metric_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown market metric"):
            market_disagreements((), {}, metric="vibes")
        with pytest.raises(ValueError, match="unknown market metric"):
            MarketQuote(player_id=1).metric("vibes")

    def test_the_informative_ceiling_is_exclusive(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """169.0 is the first censored pick, so 169.0 itself is out and 168.9 is in.

        The boundary is the whole point of the filter -- ESPN's censored pile starts
        exactly at the ceiling -- and an off-by-one there readmits the noise the
        constant exists to exclude.
        """
        report = value_league(make_context(), half_ppr_outlooks)
        ceiling = MARKET_INFORMATIVE_RANGE["adp"][1]
        assert ceiling is not None
        two = report.values[:2]
        for adp, expected in ((ceiling, 0), (ceiling - 0.1, 2)):
            quotes = {v.player_id: MarketQuote(player_id=v.player_id, adp=adp) for v in two}
            assert len(market_disagreements(two, quotes, metric="adp")) == expected, adp

    def test_draft_rank_is_wired_end_to_end_though_the_corpus_lacks_it(self) -> None:
        """`draftRanksByRankType` is not captured by snapshot.py -- see the docstring.

        Asserting the field defaults to None proves nothing, so this screens on it:
        widening snapshot.py must need no change here, and this is what says so.
        """
        assert MarketQuote(player_id=1).draft_rank is None
        assert MARKET_METRICS["draft_rank"] is True, "a smaller draft rank is a better player"
        assert "draft_rank" in MARKET_INFORMATIVE_RANGE

        values = value_league(make_context(), synthetic_outlooks(HALF_PPR)).values
        best, mid = values[0], values[40]
        quotes = {
            best.player_id: MarketQuote(player_id=best.player_id, draft_rank=90.0),
            mid.player_id: MarketQuote(player_id=mid.player_id, draft_rank=1.0),
        }
        edges = {e.player_id: e for e in market_disagreements(values, quotes, metric="draft_rank")}
        assert edges[best.player_id].is_buy
        assert edges[best.player_id].market_metric == "draft_rank"
        assert not edges[mid.player_id].is_buy

    def test_no_quotes_is_empty_not_an_error(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        report = value_league(make_context(), half_ppr_outlooks)
        assert report.market == ()
        assert market_disagreements(report.values, {}) == ()


# --------------------------------------------------------------------------------------
# Bench hoarding
# --------------------------------------------------------------------------------------


class _FakeEntry:
    def __init__(self, position_id: int, slot: int = 20) -> None:
        self.default_position_id = position_id
        self.lineup_slot_id = slot


class _FakeRoster:
    def __init__(self, positions: list[int], ir: list[int] | None = None) -> None:
        self.entries = [_FakeEntry(p) for p in positions]
        self.entries += [_FakeEntry(p, slot=21) for p in ir or []]


class TestBenchHoarding:
    def _rosters(self, per_team: Mapping[int, int], teams: int = 12):
        roster = [p for p, n in per_team.items() for _ in range(n)]
        return {i: _FakeRoster(roster) for i in range(1, teams + 1)}

    def test_measures_what_the_league_actually_carries(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        model = build_replacement_model(ctx, half_ppr_outlooks)
        demand = {p: lv.demand for p, lv in model.levels.items()}
        rosters = self._rosters({QB: 2, RB: 5, WR: 6, TE: 2, K: 1, DST: 1})
        beta = bench_hoarding_from_rosters(rosters, demand)
        assert beta[QB] == pytest.approx(1.0)
        assert beta[K] == pytest.approx(0.0)
        assert beta[RB] == pytest.approx(5.0 - demand[RB].starters_per_team)
        # The check worth running: beta must add up to the bench.
        assert sum(beta.values()) == pytest.approx(17 - 9, abs=0.01)

    def test_an_under_carried_position_floors_at_zero_rather_than_going_negative(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """Not every team rosters a kicker. That is not negative bench hoarding.

        Without the floor a league that carries 0.5 kickers per team would push
        N_K below the number of kickers the league is obliged to start, and the
        replacement level would be read off a rank nobody can reach.
        """
        ctx = make_context()
        model = build_replacement_model(ctx, half_ppr_outlooks)
        demand = {p: lv.demand for p, lv in model.levels.items()}
        rosters = {
            i: _FakeRoster([QB, RB, RB, RB, WR, WR, WR, TE] + ([K] if i % 2 else []))
            for i in range(1, 13)
        }
        beta = bench_hoarding_from_rosters(rosters, demand)
        assert beta[K] == 0.0, "half a kicker per team is still zero hoarding, not -0.5"
        assert all(v >= 0.0 for v in beta.values())
        assert beta[DST] == 0.0, "a position nobody rosters must not go negative either"

    def test_the_research_prior_does_not_add_up_to_a_bench(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        """1.8 bench players against a seven-man bench -- and it moves the answer.

        The documented sanity check is `sum(beta) == bench slots per team`. The prior
        fails it by a factor of four, so this asserts the consequence rather than the
        constant: the prior's baselines are materially shallower than the ones a real
        seven-man bench produces, which is why the daily run must pass measured betas.
        """
        ctx = make_context()
        bench_slots = ctx.lineup_slot_counts[20]
        assert sum(DEFAULT_BENCH_HOARDING.values()) == pytest.approx(1.8)
        assert sum(DEFAULT_BENCH_HOARDING.values()) < bench_slots / 3

        prior = build_replacement_model(ctx, half_ppr_outlooks)
        demand = {p: lv.demand for p, lv in prior.levels.items()}
        # 16 carried, 9 of them starters: the bench is exactly the seven slots.
        rosters = self._rosters({QB: 2, RB: 5, WR: 5, TE: 2, K: 1, DST: 1})
        measured = bench_hoarding_from_rosters(rosters, demand)
        assert sum(measured.values()) == pytest.approx(bench_slots, abs=0.01)

        deep = build_replacement_model(ctx, half_ppr_outlooks, bench_hoarding=measured)
        for position in (RB, WR):
            assert (
                deep.levels[position].demand.rostered > prior.levels[position].demand.rostered + 10
            ), position

    def test_recalibrating_deepens_the_baseline(
        self, half_ppr_outlooks: list[PlayerOutlook]
    ) -> None:
        ctx = make_context()
        prior = build_replacement_model(ctx, half_ppr_outlooks)
        demand = {p: lv.demand for p, lv in prior.levels.items()}
        rosters = self._rosters({QB: 2, RB: 5, WR: 6, TE: 2, K: 1, DST: 1})
        beta = bench_hoarding_from_rosters(rosters, demand)
        measured = build_replacement_model(ctx, half_ppr_outlooks, bench_hoarding=beta)
        for position in (RB, WR):
            assert (
                measured.levels[position].demand.rostered > prior.levels[position].demand.rostered
            )
            assert measured.levels[position].per_week < prior.levels[position].per_week

    def test_ir_counts_by_default_and_is_dropped_on_request(self) -> None:
        """Slot 21 is IR in the *slot* space. 21 in the position space is nothing.

        `include_ir=False` has to key on the lineup slot, and the two id spaces are
        the documented source of silent corruption in this codebase, so the branch
        is exercised rather than assumed. An IR'd receiver is still off the wire,
        which is why counting him is the default.
        """
        rosters = {i: _FakeRoster([QB, RB, RB, WR, WR], ir=[WR]) for i in range(1, 13)}
        assert rostered_per_team(rosters)[WR] == pytest.approx(3.0)
        assert rostered_per_team(rosters, include_ir=False)[WR] == pytest.approx(2.0)
        assert rostered_per_team(rosters, include_ir=False)[RB] == pytest.approx(2.0)

    def test_empty_rosters_are_not_a_division_by_zero(self) -> None:
        assert bench_hoarding_from_rosters({}, {}) == {}


# --------------------------------------------------------------------------------------
# Ways the caller can be wrong
# --------------------------------------------------------------------------------------


class TestGuards:
    def test_a_duplicated_player_is_refused(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        """A merged projection set with a duplicate would shift every baseline."""
        doubled = [*half_ppr_outlooks, half_ppr_outlooks[0]]
        with pytest.raises(ValuationError, match="appears twice"):
            build_replacement_model(make_context(), doubled)

    def test_missing_slot_eligibility_is_loud(
        self, half_ppr_outlooks: list[PlayerOutlook], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Demand silently vanishing is the worst outcome, so it warns."""
        ctx = make_context()
        ctx = LeagueContext(
            **{
                **{f.name: getattr(ctx, f.name) for f in ctx.__dataclass_fields__.values()},
                "slot_eligibility": {k: v for k, v in SLOT_ELIGIBILITY.items() if k != 23},
            }
        )
        with caplog.at_level("WARNING"):
            dedicated, flex = starting_shape(ctx, (QB, RB, WR, TE, K, DST))
        assert flex == {}
        assert "eligibility" in caplog.text

    def test_report_accessors(self, half_ppr_outlooks: list[PlayerOutlook]) -> None:
        report = value_league(
            make_context(),
            half_ppr_outlooks,
            quotes={
                v.player_id: MarketQuote(player_id=v.player_id, adp=float(i + 1))
                for i, v in enumerate(value_league(make_context(), half_ppr_outlooks).values)
            },
        )
        assert len(report.top(5)) == 5
        assert all(v.position_id == RB for v in report.top(5, RB))
        assert report.value_for(-1) is None
        assert len(report.buys(3)) <= 3 and len(report.fades(3)) <= 3
        text = report.replacement.describe()
        assert "RB" in text and "flex" in text
        assert str(report.teams) in text


# --------------------------------------------------------------------------------------
# Corpus-backed: does the engine reproduce what we measured?
# --------------------------------------------------------------------------------------


def _corpus_outlooks(scorer, weeks):
    import polars as pl

    frame = pl.read_parquet(CORPUS).filter(
        (pl.col("stat_season") == 2026)
        & (pl.col("stat_source_id") == 1)
        & (pl.col("stat_split_type_id") == 1)
        & (pl.col("scoring_period_id").is_in(list(weeks)))
    )
    grouped: dict[int, dict] = {}
    for row in frame.iter_rows(named=True):
        pid, pos = row["espn_id"], row["default_position_id"]
        if pid is None or pos is None:
            continue
        stats = dict(zip(row["stat_ids"], row["stat_values"], strict=True))
        record = grouped.setdefault(
            pid, {"name": row["full_name"] or str(pid), "pos": pos, "weeks": {}}
        )
        record["weeks"][row["scoring_period_id"]] = max(scorer(stats, pos), 0.0)

    return [
        PlayerOutlook(
            player_id=pid,
            name=rec["name"],
            position_id=rec["pos"],
            pro_team_id=0,
            weeks={
                w: WeeklyOutlook(
                    player_id=pid,
                    season=2026,
                    week=w,
                    position_id=rec["pos"],
                    mean=mu,
                    sd=3.67 + 0.273 * mu,
                    p_zero=0.2,
                    shape=1.5,
                    scale=max(mu, 0.01),
                )
                for w, mu in rec["weeks"].items()
            },
        )
        for pid, rec in grouped.items()
    ]


def _corpus_quotes() -> dict[int, MarketQuote]:
    import polars as pl

    frame = (
        pl.read_parquet(CORPUS)
        .select("espn_id", "average_draft_position", "percent_owned", "auction_value_average")
        .unique(subset=["espn_id"])
    )
    return {
        row["espn_id"]: MarketQuote(
            player_id=row["espn_id"],
            adp=row["average_draft_position"],
            percent_owned=row["percent_owned"],
            auction_value=row["auction_value_average"],
        )
        for row in frame.iter_rows(named=True)
        if row["espn_id"] is not None
    }


@pytest.fixture(scope="module")
def corpus_full_ppr() -> list[PlayerOutlook]:
    return _corpus_outlooks(FULL_PPR, REGULAR_WEEKS + PLAYOFF_WEEKS)


@pytest.fixture(scope="module")
def corpus_half_ppr() -> list[PlayerOutlook]:
    return _corpus_outlooks(HALF_PPR, REGULAR_WEEKS + PLAYOFF_WEEKS)


@pytest.mark.skipif(not CORPUS.exists(), reason="no 2026 snapshot in the corpus")
class TestAgainstTheCorpus:
    def test_scarcity_reproduces_the_measured_decay(
        self, corpus_full_ppr: list[PlayerOutlook]
    ) -> None:
        """RB/WR/TE land on research's coefficients. QB does not, and it is not a fit.

        Research: QB 0.040 / RB 0.026 / WR 0.013 / TE 0.039 over the top 60. Ours
        reproduces three of those to two decimals. The QB pool has a cliff instead of
        a curve -- ESPN projects ~33 startable quarterbacks and then backups at half
        a point a week -- so an exponential over 60 ranks fits it badly (r^2 0.84 vs
        0.96+ everywhere else) and comes out twice as steep.
        """
        curves = scarcity_curves(corpus_full_ppr, REGULAR_WEEKS + PLAYOFF_WEEKS)
        assert curves[RB].b == pytest.approx(0.026, abs=0.004)
        assert curves[WR].b == pytest.approx(0.013, abs=0.004)
        assert curves[TE].b == pytest.approx(0.039, abs=0.006)
        assert curves[QB].b > 0.06, "the QB cliff, not research's 0.040"
        assert curves[QB].r2 < min(curves[p].r2 for p in (RB, WR, TE))
        # The claim that survives: receivers are far flatter than anyone else.
        assert curves[WR].half_life > 2 * curves[TE].half_life

    def test_positive_vorp_share_is_near_the_measured_split(
        self, corpus_full_ppr: list[PlayerOutlook]
    ) -> None:
        """Research: RB 41.9 / WR 46.8 / TE 6.5 / QB 4.8, skill positions only."""
        ctx = make_context(size=14, scorer=FULL_PPR, league_id=272150391)
        report = value_league(ctx, corpus_full_ppr)
        share = report.positive_vorp_share
        skill = sum(share[p] for p in (QB, RB, WR, TE))
        assert share[RB] / skill == pytest.approx(0.419, abs=0.06)
        assert share[WR] / skill == pytest.approx(0.468, abs=0.06)
        assert share[TE] / skill < 0.12
        assert share[QB] / skill < 0.12

    def test_vols_baselines_land_near_the_published_figures(
        self, corpus_full_ppr: list[PlayerOutlook]
    ) -> None:
        """12-team 1QB/2RB/3WR/1TE/1FLEX PPR, starters only -> QB12/RB30/WR42/TE12."""
        ctx = make_context(size=12, scorer=FULL_PPR, slots={**STANDARD_SLOTS, 4: 3}, league_id=1)
        model = build_replacement_model(ctx, corpus_full_ppr, bench_hoarding={})
        ranks = model.baseline_ranks()
        assert ranks[QB] == pytest.approx(12.0)
        assert ranks[TE] == pytest.approx(12.0, abs=0.5)
        assert ranks[RB] == pytest.approx(30.0, abs=4.0)
        assert ranks[WR] == pytest.approx(42.0, abs=4.0)

    def test_the_three_real_leagues_do_not_share_a_valuation(
        self, corpus_full_ppr: list[PlayerOutlook], corpus_half_ppr: list[PlayerOutlook]
    ) -> None:
        """14-team full PPR vs 12-team half PPR, on the real pool and the real slots."""
        wine = value_league(
            make_context(size=14, scorer=FULL_PPR, league_id=272150391, name="Wine Wednesday"),
            corpus_full_ppr,
        )
        baddies = value_league(
            make_context(size=12, scorer=HALF_PPR, league_id=161496047, name="Baddies"),
            corpus_half_ppr,
        )
        type_shi = value_league(
            make_context(size=12, scorer=HALF_PPR, league_id=634537479, name="Type shi"),
            corpus_half_ppr,
        )

        assert wine.replacement.baseline_ranks() != baddies.replacement.baseline_ranks()
        for position in (QB, RB, WR, TE, K, DST):
            assert (
                wine.replacement.levels[position].demand.rostered
                > baddies.replacement.levels[position].demand.rostered
            ), position

        shared = 0
        for value in wine.values[:200]:
            other = baddies.value_for(value.player_id)
            if other is not None and other.ros_vorp == pytest.approx(value.ros_vorp, abs=1e-9):
                shared += 1
        assert shared == 0

        # The two 12-team half-PPR leagues are the same league twice; they must agree.
        for a, b in zip(baddies.values, type_shi.values, strict=True):
            assert a.player_id == b.player_id and a.ros_vorp == pytest.approx(b.ros_vorp)

    def test_the_real_leagues_identify_alpha_on_the_real_pool(
        self, corpus_full_ppr: list[PlayerOutlook], corpus_half_ppr: list[PlayerOutlook]
    ) -> None:
        """The synthetic pool spreads by 0.0000; the real one does not. Pin the gap.

        Measured 2026-09-07 on the 596-player corpus at the research prior: the
        14-team full-PPR league is guess-independent to machine precision, the
        12-team half-PPR one moves by 0.0098 -- real, an eighth of a roster spot,
        and half of FLEX_IDENTIFICATION_TOLERANCE. If that ever grows past the
        tolerance the engine will silently switch to the open pool, so it is worth
        knowing before it happens rather than after.
        """
        for ctx, pool, ceiling in (
            (make_context(size=14, scorer=FULL_PPR, league_id=272150391), corpus_full_ppr, 0.002),
            (make_context(size=12, scorer=HALF_PPR, league_id=161496047), corpus_half_ppr, 0.015),
        ):
            solution = solve_flex_shares(ctx, pool, REGULAR_WEEKS + PLAYOFF_WEEKS)
            assert solution.identified, ctx.league_id
            assert not solution.saturated, ctx.league_id
            assert solution.guess_spread < ceiling, (ctx.league_id, solution.guess_spread)
            assert solution.guess_spread < FLEX_IDENTIFICATION_TOLERANCE

    def test_a_thin_bench_degenerates_on_the_real_pool_too(
        self, corpus_half_ppr: list[PlayerOutlook]
    ) -> None:
        """The degeneracy is a property of the model, not of the synthetic fixture."""
        ctx = make_context(size=12, scorer=HALF_PPR, league_id=161496047)
        weeks = REGULAR_WEEKS + PLAYOFF_WEEKS
        thin = {RB: 0.033, WR: 0.033, TE: 0.033}
        guarded = solve_flex_shares(ctx, corpus_half_ppr, weeks, bench_hoarding=thin)
        assert not guarded.identified and guarded.guess_spread > 0.5

        echoed = solve_flex_shares(
            ctx,
            corpus_half_ppr,
            weeks,
            bench_hoarding=thin,
            initial_shares={23: {RB: 1.0, WR: 0.0, TE: 0.0}},
            identification_tolerance=math.inf,
        )
        assert echoed.converged and echoed.shares[23][RB] == pytest.approx(1.0)

    def test_espn_adp_really_is_censored(self) -> None:
        """The measurement behind MARKET_INFORMATIVE_RANGE, pinned so it cannot rot."""
        quotes = _corpus_quotes()
        adps = sorted(q.adp for q in quotes.values() if q.adp is not None)
        ceiling = MARKET_INFORMATIVE_RANGE["adp"][1]
        assert ceiling is not None
        below = [a for a in adps if a < ceiling]
        assert len(below) < 0.25 * len(adps), "if this fails ESPN stopped censoring ADP"
        assert max(adps) - ceiling < 5.0, "the whole censored pile sits just above the ceiling"

    def test_the_market_screen_finds_league_specific_edges(
        self, corpus_full_ppr: list[PlayerOutlook], corpus_half_ppr: list[PlayerOutlook]
    ) -> None:
        quotes = _corpus_quotes()
        wine = value_league(
            make_context(size=14, scorer=FULL_PPR, league_id=272150391),
            corpus_full_ppr,
            quotes=quotes,
        )
        baddies = value_league(
            make_context(size=12, scorer=HALF_PPR, league_id=161496047),
            corpus_half_ppr,
            quotes=quotes,
        )
        assert wine.market and baddies.market
        assert wine.buys() and wine.fades()
        # Same ADP, different valuations, so the disagreements must differ.
        left = {e.player_id: e.rank_delta for e in wine.market}
        right = {e.player_id: e.rank_delta for e in baddies.market}
        assert any(left[p] != right.get(p) for p in left)


# --------------------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------------------

#: Verified live 2026-09-07 against all three of the user's leagues.
REAL_LEAGUES = (
    (272150391, 14, 1.0),
    (161496047, 12, 0.5),
    (634537479, 12, 0.5),
)


@pytest.mark.network
@pytest.mark.parametrize(("league_id", "size", "ppr"), REAL_LEAGUES)
def test_real_league_shape_is_still_what_we_modelled(league_id: int, size: int, ppr: float) -> None:
    """The fixtures above encode these leagues; this is the drift alarm."""
    from fantasy_quant.espn.client import EspnClient
    from fantasy_quant.espn.league import League
    from fantasy_quant.registry import load_credentials

    credentials = load_credentials(REPO / ".env")
    if not credentials.complete:
        pytest.skip("no ESPN credentials")
    with EspnClient(**credentials.as_kwargs()) as client:
        settings = League(client, league_id, 2026).settings()

    assert settings.size == size
    assert settings.scoring.points_for(53, WR) == pytest.approx(ppr)
    assert settings.scoring.points_for(53, TE) == pytest.approx(ppr), "no TE premium"
    assert settings.roster.starting_slots == {0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 23: 1}
    assert settings.roster.lineup_slot_counts[20] == 7
    assert settings.schedule.playoff_team_count == 6
    assert settings.schedule.playoff_weeks == PLAYOFF_WEEKS
    assert not settings.acquisition.uses_faab

    ctx = make_context(
        size=settings.size,
        slots=settings.roster.lineup_slot_counts,
        scorer=LeagueScoring.from_settings({"scoringItems": list(settings.scoring.scoring_items)}),
        league_id=league_id,
        name=settings.name,
    )
    demand = position_demand(ctx, {23: {RB: 0.5, WR: 0.5}}, positions=(QB, RB, WR, TE, K, DST))
    assert demand[QB].rostered == pytest.approx(size * 1.4)
