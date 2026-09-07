"""The ensemble and its source adapters.

Every test here corresponds to a documented claim in `projections/ensemble.py` or
`projections/sources.py`. The point is to fail when one of those claims stops
being true -- when Hodges-Lehmann quietly becomes a mean, when a missing source
starts being imputed as zero, when a consensus product sneaks in as an input, or
when ETR renames a column -- not to re-assert that dictionaries have keys.

Offline except where marked. The corpus checks read the Parquet already on disk
and skip if it is not there.
"""

from __future__ import annotations

import dataclasses
import itertools
import random
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats as scipy_stats

from fantasy_quant.core import ComponentLine, LeagueContext, Objective
from fantasy_quant.data import ids as ids_module
from fantasy_quant.data import props as props_module
from fantasy_quant.data import sleeper as sleeper_module
from fantasy_quant.espn.scoring import LeagueScoring
from fantasy_quant.projections import ensemble as ens
from fantasy_quant.projections import sources as src

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_2026 = REPO_ROOT / "data/snapshots/espn/season=2026/variant=ppr"


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


def scoring_settings(points_per_reception: float) -> dict:
    """A minimal but real `scoringSettings` payload, in ESPN's own shape."""
    return {
        "scoringSettings": {
            "scoringItems": [
                {"statId": 3, "points": 0.04},
                {"statId": 4, "points": 4.0},
                {"statId": 20, "points": -2.0},
                {"statId": 24, "points": 0.1},
                {"statId": 25, "points": 6.0},
                {"statId": 42, "points": 0.1},
                {"statId": 43, "points": 6.0},
                {"statId": 53, "points": points_per_reception},
                {"statId": 72, "points": -2.0},
                {"statId": 83, "points": 3.0},
                {"statId": 86, "points": 1.0},
            ]
        }
    }


def league(points_per_reception: float, *, league_id: int = 1, size: int = 12) -> LeagueContext:
    """One of the user's league shapes: 1QB/2RB/2WR/1TE/1FLEX/1DST/1K, 7 bench, 1 IR."""
    scorer = LeagueScoring.from_settings(scoring_settings(points_per_reception))
    return LeagueContext(
        league_id=league_id,
        season=2026,
        name=f"{points_per_reception} ppr",
        size=size,
        lineup_slot_counts={0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 16: 1, 17: 1, 20: 7, 21: 1},
        slot_eligibility={
            0: frozenset({1}),
            2: frozenset({2}),
            4: frozenset({3}),
            6: frozenset({4}),
            23: frozenset({2, 3, 4}),
            16: frozenset({16}),
            17: frozenset({5}),
        },
        scorer=scorer,
        playoff_team_count=6,
        playoff_weeks=(15, 16, 17),
        regular_season_weeks=tuple(range(1, 15)),
        objective=Objective.CHAMPIONSHIP,
        uses_faab=False,
        my_team_id=1,
    )


FULL_PPR = league(1.0, league_id=272150391, size=14)
HALF_PPR = league(0.5, league_id=161496047, size=12)


def line(source: str, player_id: int, stats: dict[str, float], week: int = 1) -> ComponentLine:
    return ComponentLine(player_id=player_id, season=2026, week=week, source=source, stats=stats)


def source_lines(source: str, lines, positions=None, names=None) -> src.SourceLines:
    return src.SourceLines(
        source=source,
        season=2026,
        week=1,
        lines=tuple(lines),
        positions=positions or {},
        names=names or {},
    )


# --------------------------------------------------------------------------------------
# Hodges-Lehmann
# --------------------------------------------------------------------------------------


def test_hodges_lehmann_hand_computed():
    """[1, 2, 6]: Walsh averages 1, 1.5, 3.5, 2, 4, 6 -> sorted median 2.75."""
    values = [1.0, 2.0, 6.0]
    assert sorted(ens.walsh_averages(values)) == [1.0, 1.5, 2.0, 3.5, 4.0, 6.0]
    assert ens.hodges_lehmann(values) == pytest.approx(2.75)
    # Not the mean (3.0) and not the median (2.0). If it ever equals either of
    # those for this input, someone swapped the estimator out.
    assert ens.hodges_lehmann(values) != pytest.approx(np.mean(values))
    assert ens.hodges_lehmann(values) != pytest.approx(np.median(values))


def test_walsh_averages_include_the_raw_values():
    """i <= j, not i < j. The diagonal is what puts the raw values in the set."""
    values = [3.0, 9.0]
    assert sorted(ens.walsh_averages(values)) == [3.0, 6.0, 9.0]
    assert len(ens.walsh_averages([1.0] * 5)) == 5 * 6 // 2


def test_hodges_lehmann_matches_brute_force():
    rng = random.Random(20260907)
    for n in range(1, 9):
        for _ in range(40):
            values = [rng.gauss(12.0, 5.0) for _ in range(n)]
            brute = float(
                np.median(
                    [(a + b) / 2 for a, b in itertools.combinations_with_replacement(values, 2)]
                )
            )
            assert ens.hodges_lehmann(values) == pytest.approx(brute)


def test_hodges_lehmann_sits_on_the_wilcoxon_null_expectation():
    """The HL estimator is defined by W+(theta) = #{Walsh averages > theta}, so
    subtracting it leaves the signed-rank statistic on its null expectation
    n(n+1)/4. scipy computes that statistic independently of us.

    Only for n where the Walsh count n(n+1)/2 is EVEN. When it is odd the median
    is one of the Walsh averages rather than the midpoint of two, the statistic
    lands half a rank off the expectation, and if that average happens to be a
    diagonal element scipy drops the resulting exact zero and renumbers the ranks.
    Asserting the identity for every n would be asserting something false; the
    odd-n case is covered by `test_hodges_lehmann_balances_the_walsh_averages`.
    """
    rng = random.Random(7)
    for n in (3, 4, 7, 8, 11, 12):
        assert (n * (n + 1) // 2) % 2 == 0
        for _ in range(10):
            values = [rng.gauss(10.0, 4.0) for _ in range(n)]
            hl = ens.hodges_lehmann(values)
            result = scipy_stats.wilcoxon(np.array(values) - hl, method="exact")
            assert float(result.statistic) == pytest.approx(n * (n + 1) / 4.0, abs=1e-9)
            assert float(result.pvalue) == pytest.approx(1.0)


def test_hodges_lehmann_balances_the_walsh_averages():
    rng = random.Random(4)
    for n in range(2, 11):
        for _ in range(20):
            values = [rng.gauss(10.0, 4.0) for _ in range(n)]
            hl = ens.hodges_lehmann(values)
            walsh = ens.walsh_averages(values)
            assert sum(1 for w in walsh if w > hl) == sum(1 for w in walsh if w < hl)


def test_hodges_lehmann_maximizes_the_wilcoxon_p_value():
    """The other way to say the same thing, and the one that holds for every n:
    HL is the shift that makes the one-sample Wilcoxon test least significant."""
    rng = random.Random(5)
    for n in (5, 6, 7, 8):
        for _ in range(10):
            values = np.array([rng.gauss(10.0, 4.0) for _ in range(n)])
            hl = ens.hodges_lehmann(list(values))
            at_hl = float(scipy_stats.wilcoxon(values - hl, method="exact").pvalue)
            for shift in (3.0, -3.0):
                moved = float(scipy_stats.wilcoxon(values - (hl + shift), method="exact").pvalue)
                assert moved <= at_hl


def test_hodges_lehmann_degenerate_cases():
    assert ens.hodges_lehmann([7.25]) == 7.25
    assert ens.hodges_lehmann([4.0, 6.0]) == 5.0
    assert ens.hodges_lehmann([2.0, 2.0, 2.0]) == 2.0
    with pytest.raises(ValueError):
        ens.hodges_lehmann([])


def test_hodges_lehmann_equals_mean_for_symmetric_data():
    values = [8.0, 9.0, 10.0, 11.0, 12.0]
    assert ens.hodges_lehmann(values) == pytest.approx(10.0)
    assert ens.hodges_lehmann(values) == pytest.approx(float(np.mean(values)))


def test_hodges_lehmann_resists_one_broken_source():
    """29% breakdown: one source blowing up moves HL far less than the mean."""
    sane = [12.0, 13.0, 12.5, 13.5]
    broken = [*sane, 400.0]
    assert ens.hodges_lehmann(sane) == pytest.approx(12.75)
    hl_shift = abs(ens.hodges_lehmann(broken) - ens.hodges_lehmann(sane))
    mean_shift = abs(float(np.mean(broken)) - float(np.mean(sane)))
    assert hl_shift < 2.0
    assert mean_shift > 70.0


def test_weighted_median_reproduces_numpy_median_when_equal():
    rng = random.Random(11)
    for n in range(1, 12):
        values = [rng.gauss(0.0, 3.0) for _ in range(n)]
        assert ens.weighted_median(values, [1.0] * n) == pytest.approx(float(np.median(values)))
        # Scale invariance in the weights is what makes "renormalize when a
        # source is missing" a no-op rather than a code path.
        assert ens.weighted_median(values, [3.7] * n) == pytest.approx(float(np.median(values)))


def test_weighted_median_excludes_a_zero_weight_value():
    """A zero weight must not be able to decide the answer. Left in the sorted
    order it can be picked as the partner of an exact-half tie: weights
    [1, 0, 1] over [1, 2, 3] returns 1.5 rather than 2.0."""
    assert ens.weighted_median([1.0, 2.0, 3.0], [1.0, 0.0, 1.0]) == pytest.approx(2.0)
    assert ens.weighted_median([1.0, 2.0, 3.0], [0.0, 1.0, 1.0]) == pytest.approx(2.5)
    assert ens.weighted_median([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        ens.weighted_median([1.0, 2.0], [0.0, 0.0])


def test_weighted_hodges_lehmann_reduces_to_unweighted():
    values = [1.0, 2.0, 6.0, 4.5]
    assert ens.hodges_lehmann(values, [1.0] * 4) == pytest.approx(ens.hodges_lehmann(values))
    assert ens.hodges_lehmann(values, [0.25] * 4) == pytest.approx(ens.hodges_lehmann(values))


def test_weighted_hodges_lehmann_moves_toward_the_heavier_source():
    values = [10.0, 20.0, 30.0]
    even = ens.hodges_lehmann(values)
    heavy_low = ens.hodges_lehmann(values, [8.0, 1.0, 1.0])
    heavy_high = ens.hodges_lehmann(values, [1.0, 1.0, 8.0])
    assert heavy_low < even < heavy_high


def test_weighted_median_is_scale_invariant_in_the_weights():
    """Doubling every weight, or dividing every weight by a billion, cannot move a
    median. The tie tolerance used to carry an absolute floor (`max(total, 1.0)`),
    so once the Walsh pair weights fell below ~1e-12 the exact-half branch fired on
    the first value and `hodges_lehmann([1, 2, 3, 10], [1e-6] * 4)` returned 2.25
    -- and 1.25 at 1e-7 -- against the correct 2.75."""
    rng = random.Random(20260908)
    for _ in range(300):
        n = rng.randint(1, 7)
        values = [round(rng.uniform(-5.0, 5.0), 3) for _ in range(n)]
        weights = [rng.choice([0.0, 0.5, 1.0, 2.0, 7.0]) for _ in range(n)]
        if sum(weights) <= 0:
            continue
        base = ens.weighted_median(values, weights)
        for scale in (1e-9, 1e-4, 1.0, 1e4, 1e9):
            assert ens.weighted_median(values, [w * scale for w in weights]) == (
                pytest.approx(base)
            )

    values = [1.0, 2.0, 3.0, 10.0]
    for weight in (1e3, 1.0, 1e-6, 1e-9):
        assert ens.hodges_lehmann(values, [weight] * 4) == pytest.approx(2.75)


def test_weighted_median_minimizes_weighted_absolute_deviation():
    """The definition, checked against the implementation rather than the other way
    round: a weighted median is any minimiser of sum w_i |x_i - m|, and the optimum
    is always attained at one of the data points."""
    rng = random.Random(77)
    for _ in range(400):
        n = rng.randint(1, 7)
        values = [round(rng.uniform(-9.0, 9.0), 3) for _ in range(n)]
        weights = [rng.choice([0.0, 0.5, 1.0, 3.0, 11.0]) for _ in range(n)]
        if sum(weights) <= 0:
            continue
        m = ens.weighted_median(values, weights)

        def cost(t, values=values, weights=weights):
            return sum(w * abs(v - t) for v, w in zip(values, weights, strict=True))

        assert cost(m) <= min(cost(v) for v in values) + 1e-9


def test_weighted_hodges_lehmann_rejects_bad_weights():
    with pytest.raises(ValueError):
        ens.hodges_lehmann([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ens.hodges_lehmann([1.0, 2.0, 3.0], [1.0, -2.0, 1.0])


# --------------------------------------------------------------------------------------
# Renormalization, not imputation
# --------------------------------------------------------------------------------------


def test_missing_source_renormalizes_rather_than_imputing():
    """Three sources on receptions, two on receiving yards. The receiving-yards
    consensus is over those two only -- no zero-fill, no mean-fill."""
    lines = [
        line("espn", 1, {"53": 4.0, "42": 50.0}),
        line("sleeper", 1, {"53": 6.0, "42": 70.0}),
        line("props", 1, {"53": 8.0}),
    ]
    result = ens.combine_component_lines(lines)
    combined = result.lines[1]

    assert combined.components["53"].value == pytest.approx(ens.hodges_lehmann([4.0, 6.0, 8.0]))
    assert combined.components["53"].source_count == 3
    # Two votes on 42, so 60.0. Imputing props at zero would give 50.0; imputing
    # it at the mean would give 60.0 too, which is why the source_count assertion
    # below is the one that actually pins the behaviour.
    assert combined.components["42"].value == pytest.approx(60.0)
    assert combined.components["42"].source_count == 2
    assert combined.components["42"].sources == ("espn", "sleeper")


def test_absence_is_not_a_zero():
    """A source that abstains must not drag the consensus toward zero."""
    both = ens.combine_component_lines(
        [line("espn", 1, {"42": 60.0}), line("sleeper", 1, {"42": 80.0})]
    ).lines[1]
    one_abstains = ens.combine_component_lines(
        [line("espn", 1, {"42": 60.0}), line("sleeper", 1, {"53": 5.0})]
    ).lines[1]
    assert both.components["42"].value == pytest.approx(70.0)
    assert one_abstains.components["42"].value == pytest.approx(60.0)
    assert one_abstains.components["42"].source_count == 1


def test_a_dropped_source_leaves_the_others_untouched():
    """The outage path: whatever is left combines exactly as if it were the whole set."""
    all_three = [
        line("espn", 1, {"53": 4.0}),
        line("sleeper", 1, {"53": 6.0}),
        line("etr", 1, {"53": 11.0}),
    ]
    without_etr = all_three[:2]
    assert ens.combine_component_lines(without_etr).lines[1].components["53"].value == (
        pytest.approx(5.0)
    )
    assert ens.combine_component_lines(all_three).lines[1].components["53"].value == (
        pytest.approx(ens.hodges_lehmann([4.0, 6.0, 11.0]))
    )


def test_source_count_and_coverage_are_exposed():
    lines = [
        line("espn", 1, {"53": 4.0}),
        line("sleeper", 1, {"53": 6.0}),
        line("espn", 2, {"53": 1.0}),
    ]
    result = ens.combine_component_lines(lines)
    assert result.lines[1].source_count == 2
    assert result.lines[2].source_count == 1
    assert result.lines[1].sources == ("espn", "sleeper")
    assert result.coverage() == {1: 2, 2: 1}
    assert result.coverage_histogram() == {1: 1, 2: 1}


def test_min_sources_filters_on_coverage():
    lines = [
        line("espn", 1, {"53": 4.0}),
        line("sleeper", 1, {"53": 6.0}),
        line("espn", 2, {"53": 1.0}),
    ]
    result = ens.combine_component_lines(lines, min_sources=2)
    assert set(result.lines) == {1}


def test_combine_refuses_to_mix_weeks():
    lines = [line("espn", 1, {"53": 4.0}, week=1), line("espn", 1, {"53": 9.0}, week=2)]
    with pytest.raises(ValueError, match="mix of seasons"):
        ens.combine_component_lines(lines)
    # Selecting one is fine.
    assert ens.combine_component_lines(lines, week=2).lines[1].components["53"].value == 9.0


def test_duplicate_line_from_one_source_is_one_vote(caplog):
    lines = [line("espn", 1, {"53": 4.0}), line("espn", 1, {"53": 10.0})]
    with caplog.at_level("WARNING"):
        result = ens.combine_component_lines(lines)
    assert result.lines[1].source_count == 1
    assert "duplicate" in caplog.text


# --------------------------------------------------------------------------------------
# Consensus sources
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "fantasypros",
        "FantasyPros",
        "FantasyPros ECR",
        "fantasy_pros",
        "fantasyfootballnerd",
        "Fantasy Football Nerd",
        "FFN",
        "ecr",
        "ensemble",
    ],
)
def test_consensus_sources_are_rejected_as_inputs(name):
    with pytest.raises(ens.ConsensusSourceError):
        ens.combine_component_lines([line("espn", 1, {"53": 4.0}), line(name, 1, {"53": 6.0})])


@pytest.mark.parametrize("name", ["espn", "sleeper", "etr", "props", "underdog", "ffngrades"])
def test_real_sources_are_not_mistaken_for_consensuses(name):
    assert not ens.is_consensus_source(name)
    result = ens.combine_component_lines([line(name, 1, {"53": 4.0})])
    assert result.lines[1].source_count == 1


def test_consensus_source_cannot_arrive_through_the_weights_either():
    with pytest.raises(ens.ConsensusSourceError):
        ens.resolve_weights(["espn", "sleeper"], {"fantasypros": 2.0})


def test_weights_reject_a_name_that_is_not_present():
    """A typo must not silently leave the source it meant to down-weight at 1.0."""
    with pytest.raises(ValueError, match="not present"):
        ens.resolve_weights(["espn", "sleeper"], {"slepper": 0.5})


def test_default_weights_are_equal():
    assert ens.resolve_weights(["espn", "sleeper", "etr"], None) == {
        "espn": 1.0,
        "sleeper": 1.0,
        "etr": 1.0,
    }


def test_zero_weight_drops_a_source_entirely():
    lines = [line("espn", 1, {"53": 4.0}), line("sleeper", 1, {"53": 10.0})]
    result = ens.combine_component_lines(lines, weights={"sleeper": 0.0})
    assert result.sources == ("espn",)
    assert result.lines[1].components["53"].value == pytest.approx(4.0)
    assert result.lines[1].source_count == 1


# --------------------------------------------------------------------------------------
# One ensemble, three leagues
# --------------------------------------------------------------------------------------


def test_same_ensemble_scores_differently_in_full_and_half_ppr():
    """A receiver's points differ by exactly 0.5 x receptions; a kicker's do not.

    This is the whole reason the ensemble is built on components. Nothing is
    re-projected between the two leagues -- the identical `EnsembleLine` is
    handed to two different `LeagueContext.scorer`s.
    """
    receiver = ens.combine_component_lines(
        [
            line("espn", 1, {"53": 5.0, "42": 60.0, "43": 0.4}),
            line("sleeper", 1, {"53": 7.0, "42": 80.0, "43": 0.6}),
        ]
    ).lines[1]
    kicker = ens.combine_component_lines(
        [line("espn", 2, {"83": 1.8, "86": 2.4}), line("sleeper", 2, {"83": 2.2, "86": 2.6})]
    ).lines[2]

    receptions = receiver.components["53"].value
    assert receptions == pytest.approx(6.0)

    full = receiver.points(FULL_PPR.scorer, position_id=3)
    half = receiver.points(HALF_PPR.scorer, position_id=3)
    # 6 rec + 70 yds + 0.5 TD = 6.0 + 7.0 + 3.0 = 16.0 full PPR, 13.0 half.
    assert full == pytest.approx(16.0)
    assert half == pytest.approx(13.0)
    assert full - half == pytest.approx(0.5 * receptions)

    k_full = kicker.points(FULL_PPR.scorer, position_id=5)
    k_half = kicker.points(HALF_PPR.scorer, position_id=5)
    assert k_full == pytest.approx(2.0 * 3.0 + 2.5 * 1.0)
    assert k_full == k_half


def test_ensemble_score_runs_one_line_through_every_league():
    result = ens.combine_component_lines(
        [line("espn", 1, {"53": 5.0}), line("sleeper", 1, {"53": 7.0})],
        positions={1: 3},
    )
    assert result.score(FULL_PPR) == {1: pytest.approx(6.0)}
    assert result.score(HALF_PPR) == {1: pytest.approx(3.0)}


def test_te_premium_is_a_position_lookup_not_a_slot_lookup():
    """`pointsOverrides` keys are defaultPositionIds. TE is position 4 and slot 6;
    slot 4 is WR. Scoring a TE through the slot space silently pays WR rates."""
    settings = scoring_settings(0.5)
    settings["scoringSettings"]["scoringItems"] = [
        {"statId": 53, "points": 0.5, "pointsOverrides": {"4": 1.5}},
    ]
    scorer = LeagueScoring.from_settings(settings)
    combined = ens.combine_component_lines([line("espn", 1, {"53": 4.0})]).lines[1]
    assert combined.points(scorer, position_id=4) == pytest.approx(6.0)  # TE
    assert combined.points(scorer, position_id=3) == pytest.approx(2.0)  # WR


# --------------------------------------------------------------------------------------
# Disagreement
# --------------------------------------------------------------------------------------


def test_stat_spread_is_the_sample_sd_of_the_votes():
    combined = ens.combine_component_lines(
        [
            line("espn", 1, {"42": 60.0}),
            line("sleeper", 1, {"42": 80.0}),
            line("etr", 1, {"42": 70.0}),
        ]
    ).lines[1]
    consensus = combined.components["42"]
    assert consensus.spread == pytest.approx(float(np.std([60.0, 80.0, 70.0], ddof=1)))
    assert consensus.span == pytest.approx(20.0)


def test_a_single_source_has_no_spread():
    combined = ens.combine_component_lines([line("espn", 1, {"42": 60.0})]).lines[1]
    assert combined.components["42"].spread == 0.0
    assert combined.points_spread(FULL_PPR.scorer, position_id=3) == 0.0


def test_points_disagreement_is_league_specific():
    """Two sources three receptions apart disagree by 3.0 points in full PPR and
    1.5 in half. A league-independent uncertainty number would be wrong twice."""
    combined = ens.combine_component_lines(
        [line("espn", 1, {"53": 4.0}), line("sleeper", 1, {"53": 10.0})]
    ).lines[1]
    full = combined.points_spread(FULL_PPR.scorer, position_id=3)
    half = combined.points_spread(HALF_PPR.scorer, position_id=3)
    assert full == pytest.approx(float(np.std([4.0, 10.0], ddof=1)))
    assert half == pytest.approx(full / 2.0)
    assert combined.points_by_source(FULL_PPR.scorer, 3) == {
        "espn": pytest.approx(4.0),
        "sleeper": pytest.approx(10.0),
    }


def test_points_spread_over_can_exclude_the_sparse_sources():
    combined = ens.combine_component_lines(
        [
            line("espn", 1, {"53": 5.0, "42": 60.0}),
            line("sleeper", 1, {"53": 7.0, "42": 70.0}),
            line("props", 1, {"42": 65.0}),
        ]
    ).lines[1]
    everything = combined.points_spread(FULL_PPR.scorer, 3)
    dense_only = combined.points_spread_over(FULL_PPR.scorer, ["espn", "sleeper"], 3)
    assert everything != pytest.approx(dense_only)
    assert dense_only == pytest.approx(float(np.std([11.0, 14.0], ddof=1)))


def test_to_component_line_returns_the_contract_type():
    combined = ens.combine_component_lines(
        [line("espn", 1, {"53": 4.0}), line("sleeper", 1, {"53": 6.0})]
    ).lines[1]
    out = combined.to_component_line()
    assert isinstance(out, ComponentLine)
    assert out.source == ens.ENSEMBLE_SOURCE
    assert out.stats["53"] == pytest.approx(5.0)
    assert out.points(FULL_PPR.scorer, 3) == pytest.approx(5.0)


# --------------------------------------------------------------------------------------
# Derived stats
# --------------------------------------------------------------------------------------


def test_add_derived_stats_rebuilds_the_floor_buckets():
    stats = {"24": 85.026, "23": 17.691, "42": 31.561, "53": 3.897}
    derived = ens.add_derived_stats(stats)
    assert derived["27"] == 17.0  # floor(85.026 / 5)
    assert derived["28"] == 8.0
    assert derived["29"] == 4.0
    assert derived["30"] == 3.0
    assert derived["31"] == 1.0
    assert derived["33"] == 3.0  # floor(17.691 / 5)
    assert derived["47"] == 6.0  # floor(31.561 / 5)
    assert derived["40"] == pytest.approx(85.026)  # rushing yards per game
    assert derived["39"] == pytest.approx(85.026 / 17.691)
    assert derived["60"] == pytest.approx(31.561 / 3.897)


def test_add_derived_stats_skips_a_zero_denominator():
    """ESPN does not publish yards-per-carry for a player with no carries, and
    neither do we -- a 0/0 would land in the line as a NaN and poison the sum."""
    derived = ens.add_derived_stats({"24": 0.0, "23": 0.0})
    assert "39" not in derived


@pytest.mark.skipif(not CORPUS_2026.exists(), reason="ESPN corpus not present")
def test_derived_rules_reproduce_espns_own_rows():
    """The rules against every real ESPN projection row in the corpus.

    Two assertions, and the second is the one that has teeth: not only must every
    rebuilt value match ESPN's, but every derived statId ESPN *publishes* must be
    rebuilt at all. Checking only the ids we happen to emit lets a whole rule
    family stop firing and still pass, because the missing id is silently skipped.
    """
    files = sorted(CORPUS_2026.glob("*.parquet"))
    frame = pl.read_parquet(files[-1]).filter(
        (pl.col("stat_season") == 2026)
        & (pl.col("stat_source_id") == 1)
        & (pl.col("stat_split_type_id") == 1)
    )
    checked: dict[str, int] = {}
    unreproduced: dict[str, int] = {}
    for row in frame.iter_rows(named=True):
        espn = dict(zip(row["stat_ids"], row["stat_values"], strict=True))
        primitives = {k: v for k, v in espn.items() if k not in src.DERIVED_STAT_IDS}
        rebuilt = ens.add_derived_stats(primitives)
        for stat_id, value in espn.items():
            if stat_id not in src.DERIVED_STAT_IDS:
                continue
            if stat_id not in rebuilt:
                unreproduced[stat_id] = unreproduced.get(stat_id, 0) + 1
                continue
            # ESPN stores these rounded to ~9 significant digits, so an exact
            # comparison is not available: measured over the whole corpus the
            # worst deviation is 7e-8 absolute / 3.5e-8 relative, and this is
            # ~30x that. Loose enough for ESPN's rounding, far too tight for a
            # wrong divisor (the original rel=1e-3 would not have noticed a
            # yards-per-carry rule drifting in the seventh digit).
            assert rebuilt[stat_id] == pytest.approx(value, rel=1e-6, abs=1e-7), (
                f"statId {stat_id} for {row['full_name']} week {row['scoring_period_id']}"
            )
            checked[stat_id] = checked.get(stat_id, 0) + 1

    assert not unreproduced, f"ESPN published derived statIds we never rebuilt: {unreproduced}"
    assert sum(checked.values()) > 20000, f"only {sum(checked.values())} checks; the filter moved"
    # The families ESPN actually publishes weekly: floor buckets off passing,
    # rushing, receiving and completions; the per-game copies; the three ratios;
    # incompletions; and total turnovers. A rule quietly dropping out of the table
    # would show up here as a missing key rather than as a smaller count.
    assert {"2", "5", "11", "21", "22", "27", "33", "39", "40", "47", "60", "61", "73"} <= set(
        checked
    ), f"a whole derived family stopped being reproduced; got {sorted(checked, key=int)}"


# --------------------------------------------------------------------------------------
# ESPN source
# --------------------------------------------------------------------------------------


def espn_frame(stat_ids, stat_values, *, week=1, season=2026, source=1, split=1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "espn_id": [4429795],
            "full_name": ["Jahmyr Gibbs"],
            "default_position_id": [2],
            "stat_season": [season],
            "stat_source_id": [source],
            "stat_split_type_id": [split],
            "scoring_period_id": [week],
            "stat_ids": [stat_ids],
            "stat_values": [stat_values],
        }
    )


def test_espn_source_drops_the_derived_restatements():
    frame = espn_frame(["24", "40", "27", "53", "210"], [85.0, 85.0, 17.0, 3.9, 1.0])
    lines = src.EspnSnapshotSource(frame=frame).source_lines(2026, 1)
    assert len(lines.lines) == 1
    stats = lines.lines[0].stats
    assert set(stats) == {"24", "53"}
    # Games played is shape, not forecast: it moves off the stat line.
    assert lines.lines[0].games == 1.0
    assert lines.positions == {4429795: 2}


def test_espn_source_keeps_derived_when_asked():
    frame = espn_frame(["24", "40"], [85.0, 85.0])
    lines = src.EspnSnapshotSource(frame=frame, keep_derived=True).source_lines(2026, 1)
    assert set(lines.lines[0].stats) == {"24", "40"}


def test_espn_source_filters_the_mixed_season_and_the_frozen_total():
    """A 2026 request returns 2025 rows in the same array, and the season-total
    projection is frozen at preseason. Both must be invisible here."""
    frame = pl.concat(
        [
            espn_frame(["24"], [85.0], season=2025),
            espn_frame(["24"], [1500.0], split=0),  # frozen season total
            espn_frame(["24"], [70.0], source=0),  # actual, not projected
            espn_frame(["24"], [85.0]),
        ]
    )
    lines = src.EspnSnapshotSource(frame=frame).source_lines(2026, 1)
    assert [line_.stats["24"] for line_ in lines.lines] == [85.0]


def test_espn_source_without_a_snapshot_is_unavailable_not_fatal(tmp_path):
    with pytest.raises(src.SourceUnavailable):
        src.EspnSnapshotSource(root=tmp_path).source_lines(2026, 1)


class StubPool:
    """An `EspnClient` that answers `player_pool` from a canned payload."""

    def __init__(self, entries):
        self.entries = entries
        self.calls: list[dict] = []

    def player_pool(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.entries

    def close(self):  # pragma: no cover -- only reached for a client we own
        raise AssertionError("EspnLiveSource must not close a client it was handed")


def pool_stat_row(stats, *, season=2026, source=1, split=1, week=1):
    return {
        "id": f"{source}{split}{season}{week}",
        "seasonId": season,
        "statSourceId": source,
        "statSplitTypeId": split,
        "scoringPeriodId": week,
        "stats": stats,
    }


def test_espn_live_source_parses_the_pool_like_the_snapshot():
    """The live adapter had no offline test at all; it is the same filtering and
    the same derived-stat drop as the snapshot path, and nothing was pinning that.

    The 2025 row and the frozen season total (split 0) are the two rows that must
    be invisible: array position tells you nothing, so only the season/source/split
    triple can separate them.
    """
    entry = {
        "id": 4429795,
        "player": {
            "id": 4429795,
            "fullName": "Jahmyr Gibbs",
            "defaultPositionId": 2,
            "stats": [
                pool_stat_row({"24": 999.0}, season=2025),  # mixed season
                pool_stat_row({"24": 1500.0}, split=0),  # frozen season total
                pool_stat_row({"24": 70.0}, source=0),  # actual, not projected
                pool_stat_row({"24": 85.0, "40": 85.0, "27": 17.0, "53": 3.9, "210": 1.0}),
                pool_stat_row({"24": 111.0}, week=2),  # another week
            ],
        },
    }
    client = StubPool([entry])
    source = src.EspnLiveSource(client=client)
    lines = source.source_lines(2026, 1)

    assert len(lines.lines) == 1
    assert lines.lines[0].stats == {"24": pytest.approx(85.0), "53": pytest.approx(3.9)}
    assert lines.lines[0].games == 1.0
    assert lines.positions == {4429795: 2}
    assert lines.names == {4429795: "Jahmyr Gibbs"}
    assert lines.dense is True
    # A sort is mandatory -- `limit` without one is an HTTP 400.
    assert client.calls[0]["sort"]

    # `keep_derived` behaves the same as on the snapshot path.
    kept = src.EspnLiveSource(client=StubPool([entry]), keep_derived=True).source_lines(2026, 1)
    assert set(kept.lines[0].stats) == {"24", "40", "27", "53"}


def test_espn_live_source_and_snapshot_agree_on_the_same_row():
    """Same row through both adapters must give byte-identical component lines --
    the claim that makes `EspnLiveSource` a drop-in for a missing capture."""
    stats = {"24": 85.026, "40": 85.026, "27": 17.0, "42": 31.561, "53": 3.897, "210": 1.0}
    live = src.EspnLiveSource(
        client=StubPool(
            [
                {
                    "id": 4429795,
                    "player": {
                        "id": 4429795,
                        "fullName": "Jahmyr Gibbs",
                        "defaultPositionId": 2,
                        "stats": [pool_stat_row(stats)],
                    },
                }
            ]
        )
    ).source_lines(2026, 1)
    snapshot = src.EspnSnapshotSource(
        frame=espn_frame(list(stats), list(stats.values()))
    ).source_lines(2026, 1)

    assert live.lines[0].stats == snapshot.lines[0].stats
    assert live.lines[0].games == snapshot.lines[0].games
    assert dict(live.positions) == dict(snapshot.positions)


# --------------------------------------------------------------------------------------
# Sleeper source
# --------------------------------------------------------------------------------------


def sleeper_row(sleeper_id: str, name: str, stats: dict, positions=("RB",)):
    return sleeper_module.SleeperProjection(
        sleeper_id=sleeper_id,
        name=name,
        position=positions[0],
        fantasy_positions=tuple(positions),
        team="DET",
        opponent="GB",
        season=2026,
        week=1,
        game_id=None,
        company="rotowire",
        injury_status=None,
        updated_at=None,
        stats=stats,
    )


class StubSleeper:
    def __init__(self, rows):
        self.rows = rows

    def weekly_projections(self, season, week, positions=None, company=None):
        return self.rows


def id_index() -> src.EspnIdIndex:
    """An `IdResolver` over two hand-built records. No crosswalk download."""
    resolver = ids_module.IdResolver(
        [
            ids_module.PlayerIds(
                canonical="00-0037746",
                name="Jahmyr Gibbs",
                team="DET",
                position="RB",
                origin="test",
                ids={ids_module.ESPN: "4429795", ids_module.SLEEPER: "9509"},
            ),
            ids_module.PlayerIds(
                canonical="00-0036389",
                name="Justin Jefferson",
                team="MIN",
                position="WR",
                origin="test",
                ids={ids_module.ESPN: "4262921", ids_module.SLEEPER: "6794"},
            ),
        ]
    )
    return src.EspnIdIndex(resolver=resolver)


def test_sleeper_source_maps_components_onto_espn_stat_ids():
    rows = [
        sleeper_row(
            "9509",
            "Jahmyr Gibbs",
            {"rush_att": 17.7, "rush_yd": 85.0, "rec": 3.9, "rec_tgt": 5.0, "gp": 1.0},
        )
    ]
    source = src.SleeperSource(client=StubSleeper(rows), id_index=id_index())
    lines = source.source_lines(2026, 1)
    assert len(lines.lines) == 1
    stats = lines.lines[0].stats
    assert lines.lines[0].player_id == 4429795
    assert stats["23"] == pytest.approx(17.7)  # rush attempts
    assert stats["24"] == pytest.approx(85.0)  # rushing yards
    assert stats["53"] == pytest.approx(3.9)  # receptions
    assert stats["58"] == pytest.approx(5.0)  # targets
    assert lines.lines[0].games == 1.0
    # gp is not a stat, and neither is anything outside the ESPN space.
    assert "210" not in stats
    assert lines.positions == {4429795: 2}


def test_sleeper_source_is_dense_so_silence_means_zero():
    """A Sleeper row that projects a receiver and omits `pass_yd` means zero
    passing yards. Saying so explicitly is what lets a genuinely absent source
    read as an abstention downstream."""
    rows = [sleeper_row("6794", "Justin Jefferson", {"rec": 6.0}, positions=("WR",))]
    lines = src.SleeperSource(client=StubSleeper(rows), id_index=id_index()).source_lines(2026, 1)
    stats = lines.lines[0].stats
    assert stats["3"] == 0.0  # passing yards, explicitly zero
    assert set(stats) == set(src.SLEEPER_TO_ESPN.values())


def test_sleeper_map_only_names_fields_sleeper_actually_publishes():
    """Every mapped key must exist in `sleeper.COMPONENT_FIELDS`, because the
    dense fill writes 0.0 for a mapped key that never arrives. Mapping `fum` (not
    in COMPONENT_FIELDS) zero-filled ESPN statId 68 on all 373 live rows and would
    have halved the consensus in leagues that score total fumbles."""
    assert set(src.SLEEPER_TO_ESPN) <= set(sleeper_module.COMPONENT_FIELDS)
    assert not set(src.SLEEPER_TO_ESPN) & src.SLEEPER_UNMAPPED


def test_sleeper_first_downs_are_not_espns_first_downs():
    """Measured, not assumed: Sleeper's `*_fd` run 1.6-2.1x ESPN's 211/212/213 and
    exceed their own reception counts. Whatever they are, they are not the counts
    a PPFD league scores."""
    for key in ("pass_fd", "rush_fd", "rec_fd"):
        assert key in sleeper_module.COMPONENT_FIELDS  # Sleeper does publish them
        assert key in src.SLEEPER_UNMAPPED
        assert key not in src.SLEEPER_TO_ESPN
    assert src.PASS_FIRST_DOWNS not in src.SLEEPER_TO_ESPN.values()
    assert src.REC_FIRST_DOWNS not in src.SLEEPER_TO_ESPN.values()


def test_sleeper_source_reports_rows_it_cannot_resolve():
    rows = [sleeper_row("999999", "Nobody At All", {"rec": 3.0})]
    lines = src.SleeperSource(client=StubSleeper(rows), id_index=id_index()).source_lines(2026, 1)
    assert lines.lines == ()
    assert lines.unresolved and "Nobody At All" in lines.unresolved[0]


def test_sleeper_position_comes_from_fantasy_positions():
    """`player.position` is FB/CB/DB on rows Sleeper itself classifies as skill
    players; `fantasy_positions` is the field its own filter keys on."""
    row = sleeper_row("9509", "Jahmyr Gibbs", {"rush_yd": 40.0}, positions=("RB",))
    row = dataclasses.replace(row, position="FB")
    lines = src.SleeperSource(client=StubSleeper([row]), id_index=id_index()).source_lines(2026, 1)
    assert lines.positions == {4429795: 2}


# --------------------------------------------------------------------------------------
# Props source
# --------------------------------------------------------------------------------------


def test_book_defense_reads_the_sportsbooks_own_spelling():
    """`ids.resolve_dst` knows "Seahawks D/ST" and "SEA"; FanDuel posts
    "Seattle Defense", and every D/ST prop was landing in `unresolved`."""
    assert src.book_defense("Seattle Defense").nflverse == "SEA"
    assert src.book_defense("New England Team Defense").nflverse == "NE"
    assert src.book_defense("Seahawks D/ST").nflverse == "SEA"
    assert src.book_defense("SEA").nflverse == "SEA"
    # Two teams share each of these locations. Answering with either is a wrong
    # answer that nothing downstream can see, so it refuses.
    assert src.book_defense("New York Defense") is None
    assert src.book_defense("Los Angeles Defense") is None
    assert src.book_defense("Jahmyr Gibbs") is None


def test_defense_resolves_through_from_name():
    assert id_index().from_name("Seattle Defense") == -16026
    assert id_index().from_name("Chicago Defense") == -16003


def test_market_position_hints():
    assert src.market_position_hints({props_module.PASSING_YARDS: 250.0}) == ("QB",)
    assert src.market_position_hints({props_module.RECEPTIONS: 5.0}) == ("WR", "TE", "RB")
    assert src.market_position_hints({props_module.RUSHING_YARDS: 60.0}) == ("RB", "QB", "WR")
    assert src.market_position_hints({props_module.FIELD_GOALS_MADE: 1.5}) == ("K",)
    # An anytime-TD-only line says nothing about position.
    assert src.market_position_hints({props_module.ANYTIME_TDS: 0.5}) == ("QB", "RB", "WR", "TE")


def namesake_index() -> src.EspnIdIndex:
    """Two people with one exact name, plus a near-name at a third position.

    The shape that broke the live props join: `IdResolver.resolve_name` refuses
    "Lamar Jackson" with no context because a defensive back shares the name.
    """
    return src.EspnIdIndex(
        resolver=ids_module.IdResolver(
            [
                ids_module.PlayerIds(
                    canonical="00-0034796",
                    name="Lamar Jackson",
                    team="BAL",
                    position="QB",
                    origin="test",
                    ids={ids_module.ESPN: "3916387"},
                ),
                ids_module.PlayerIds(
                    canonical="00-0035705",
                    name="Lamar Jackson",
                    team="LV",
                    position="WR",
                    origin="test",
                    ids={ids_module.ESPN: "4038815"},
                ),
                # Two near-namesakes at different positions: enough for the bare
                # name to be ambiguous and for a position hint to make it unique.
                ids_module.PlayerIds(
                    canonical="00-0037746",
                    name="Kyler Williams",
                    team="LA",
                    position="RB",
                    origin="test",
                    ids={ids_module.ESPN: "4430737"},
                ),
                ids_module.PlayerIds(
                    canonical="00-0039901",
                    name="Kylen Williams",
                    team="NYJ",
                    position="WR",
                    origin="test",
                    ids={ids_module.ESPN: "4599999"},
                ),
            ]
        )
    )


def test_position_hints_break_a_tie_the_bare_name_cannot():
    index = namesake_index()
    assert index.from_name("Lamar Jackson") is None
    assert index.from_name("Lamar Jackson", position_hints=("QB",)) == 3916387
    # Every hint that matches must agree. QB and WR find different people, so the
    # pair refuses rather than picking the first.
    assert index.from_name("Lamar Jackson", position_hints=("QB", "WR")) is None


def test_position_hints_do_not_open_the_fuzzy_path():
    """At the default score floor the hinted retry matched the live board's "Kyle
    Williams" onto Kyren Williams -- a confident wrong player, and invisible in a
    projection. HINT_MIN_SCORE turns that stage off; the miss is the right answer."""
    index = namesake_index()
    # With no context the bare name is ambiguous and correctly refused ...
    assert index.from_name("Kyle Williams") is None
    # ... but a position hint narrows it to one fuzzy candidate, which at the
    # ordinary floor resolves, and at HINT_MIN_SCORE does not.
    assert index.resolver.resolve_name("Kyle Williams", None, "RB") is not None
    assert (
        index.resolver.resolve_name("Kyle Williams", None, "RB", min_score=src.HINT_MIN_SCORE)
        is None
    )
    assert index.from_name("Kyle Williams", position_hints=("RB", "WR", "TE")) is None


def test_props_merges_two_spellings_of_one_player():
    """`components_by_player` keys on the book's name string, so one player can
    arrive twice. Two lines would give the market two votes in the ensemble."""

    def line_for(name, stat, value):
        return props_module.PropLine(
            book="fanduel",
            event_id=1,
            player=name,
            stat=stat,
            line=value,
            over_odds=-110.0,
            under_odds=-110.0,
        )

    event = props_module.FanDuelEventProps(
        event_id=1,
        lines=(
            line_for("Jahmyr Gibbs", props_module.RUSHING_YARDS, 70.5),
            line_for("Jahmyr Gibbs Jr.", props_module.RECEPTIONS, 3.5),
        ),
        ladders=(),
        touchdowns=(),
        tabs_seen=(),
    )
    source = src.PropsSource(
        bundle=props_module.PropsBundle(fanduel=[event]),
        id_index=id_index(),
        positions={4429795: 2},
    )
    lines = source.source_lines(2026, 1)
    assert len(lines.lines) == 1
    assert lines.lines[0].stats == {"24": pytest.approx(70.5), "53": pytest.approx(3.5)}


def test_props_source_is_sparse_and_splits_anytime_touchdowns():
    event = props_module.FanDuelEventProps(
        event_id=1,
        lines=(),
        ladders=(),
        touchdowns=(props_module.AnytimeTd("fanduel", 1, "Jahmyr Gibbs", -150.0),),
        tabs_seen=("touchdown-scorer-props",),
    )
    bundle = props_module.PropsBundle(fanduel=[event])
    source = src.PropsSource(bundle=bundle, id_index=id_index(), positions={4429795: 2})
    lines = source.source_lines(2026, 1)
    assert lines.dense is False
    stats = lines.lines[0].stats
    lam = props_module.expected_touchdowns(-150.0)
    share = src.RUSH_SHARE_PRIOR[2]
    assert stats["25"] == pytest.approx(lam * share)
    assert stats["43"] == pytest.approx(lam * (1.0 - share))
    # Sparse: nothing the market did not price.
    assert set(stats) == {"25", "43"}


def test_props_source_never_lets_the_split_prior_overwrite_a_posted_price():
    event = props_module.FanDuelEventProps(
        event_id=1,
        lines=(
            props_module.PropLine(
                book="fanduel",
                event_id=1,
                player="Jahmyr Gibbs",
                stat=props_module.RUSHING_TDS,
                line=0.5,
                over_odds=-110.0,
                under_odds=-110.0,
            ),
        ),
        ladders=(),
        touchdowns=(props_module.AnytimeTd("fanduel", 1, "Jahmyr Gibbs", -150.0),),
        tabs_seen=(),
    )
    source = src.PropsSource(
        bundle=props_module.PropsBundle(fanduel=[event]),
        id_index=id_index(),
        positions={4429795: 2},
    )
    stats = source.source_lines(2026, 1).lines[0].stats
    assert stats["25"] == pytest.approx(0.5)  # the posted line, not the prior


def test_props_prior_never_beats_a_posted_price_across_two_spellings():
    """The two spellings have to be merged as MARKETS and converted afterwards.

    Converting first and merging second writes the anytime-TD split into statId 25
    for whichever spelling arrives first, and the merge then refuses to replace it
    -- so the prior beats the posted price, and dict order decides which.
    """

    def event(event_id, *, lines=(), touchdowns=()):
        return props_module.FanDuelEventProps(
            event_id=event_id, lines=lines, ladders=(), touchdowns=touchdowns, tabs_seen=()
        )

    posted = props_module.PropLine(
        book="fanduel",
        event_id=2,
        player="Jahmyr Gibbs Jr.",
        stat=props_module.RUSHING_TDS,
        line=0.5,
        over_odds=-110.0,
        under_odds=-110.0,
    )
    anytime = props_module.AnytimeTd("fanduel", 1, "Jahmyr Gibbs", -150.0)

    for bundle in (
        # The anytime-only spelling first is the order that used to lose the price.
        props_module.PropsBundle(
            fanduel=[event(1, touchdowns=(anytime,)), event(2, lines=(posted,))]
        ),
        props_module.PropsBundle(
            fanduel=[event(2, lines=(posted,)), event(1, touchdowns=(anytime,))]
        ),
    ):
        source = src.PropsSource(bundle=bundle, id_index=id_index(), positions={4429795: 2})
        lines = source.source_lines(2026, 1)
        assert len(lines.lines) == 1
        stats = lines.lines[0].stats
        assert stats["25"] == pytest.approx(0.5), "the anytime-TD prior overwrote a posted price"
        lam = props_module.expected_touchdowns(-150.0)
        assert stats["43"] == pytest.approx(lam * (1.0 - src.RUSH_SHARE_PRIOR[2]))


def test_props_source_drops_already_scored_markets():
    """A `Fantasy Points` market is scored in somebody else's league. Ensembling
    it at the component level is a category error."""
    assert props_module.FANTASY_POINTS in src.PROPS_ALREADY_SCORED
    assert props_module.FANTASY_POINTS not in src.PROPS_TO_ESPN

    def line_for(stat, value):
        return props_module.PropLine(
            book="fanduel",
            event_id=1,
            player="Jahmyr Gibbs",
            stat=stat,
            line=value,
            over_odds=-110.0,
            under_odds=-110.0,
        )

    event = props_module.FanDuelEventProps(
        event_id=1,
        lines=(
            line_for(props_module.FANTASY_POINTS, 17.5),
            line_for(props_module.RUSHING_YARDS, 70.5),
        ),
        ladders=(),
        touchdowns=(),
        tabs_seen=(),
    )
    source = src.PropsSource(
        bundle=props_module.PropsBundle(fanduel=[event]),
        id_index=id_index(),
        positions={4429795: 2},
    )
    # The whole market is gone, not merely unmapped onto some statId.
    assert source.source_lines(2026, 1).lines[0].stats == {"24": pytest.approx(70.5)}


def test_props_source_with_every_book_dead_is_unavailable():
    bundle = props_module.PropsBundle(errors={"fanduel": "boom", "pinnacle": "boom"})
    source = src.PropsSource(bundle=bundle, id_index=id_index())
    with pytest.raises(src.SourceUnavailable):
        source.source_lines(2026, 1)


# --------------------------------------------------------------------------------------
# ETR CSV ingest
# --------------------------------------------------------------------------------------

ETR_HEADER = (
    "Player,Team,Pos,Week,Pass Att,Pass Yds,Pass TD,INT,"
    "Rush Att,Rush Yds,Rush TD,Targets,Rec,Rec Yds,Rec TD,FL"
)
ETR_ROW = "Jahmyr Gibbs,DET,RB,1,0,0,0,0,17.7,85.0,0.6,5.0,3.9,31.6,0.2,0.05"


def write_etr(tmp_path: Path, header: str, rows: list[str], name="etr_2026_wk01.csv") -> Path:
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_etr_loader_reads_a_well_formed_export(tmp_path):
    path = write_etr(tmp_path, ETR_HEADER, [ETR_ROW])
    lines = src.read_etr_csv(path, season=2026, week=1, id_index=id_index())
    assert len(lines.lines) == 1
    stats = lines.lines[0].stats
    assert lines.lines[0].player_id == 4429795
    assert stats["23"] == pytest.approx(17.7)
    assert stats["24"] == pytest.approx(85.0)
    assert stats["53"] == pytest.approx(3.9)
    assert stats["58"] == pytest.approx(5.0)
    assert stats["42"] == pytest.approx(31.6)
    assert stats["72"] == pytest.approx(0.05)
    assert lines.positions == {4429795: 2}


def test_etr_loader_rejects_drifted_columns(tmp_path):
    """The schema is undocumented and unversioned, so it is re-validated on every
    ingest and a rename fails loudly with the columns actually seen."""
    header = "Athlete,Club,Position,Week,Rushing Yards Projection,Receptions Projection"
    path = write_etr(tmp_path, header, ["Jahmyr Gibbs,DET,RB,1,85.0,3.9"])
    with pytest.raises(src.EtrSchemaError) as excinfo:
        src.read_etr_csv(path, season=2026, week=1, id_index=id_index())
    message = str(excinfo.value)
    assert "Athlete" in message and "Rushing Yards Projection" in message
    assert "columns seen (6)" in message
    assert "ETR_COLUMN_ALIASES" in message


def test_etr_loader_rejects_a_header_with_a_name_but_no_stats(tmp_path):
    path = write_etr(tmp_path, "Player,Team,Pos,Week,Rank,Tier", ["Jahmyr Gibbs,DET,RB,1,1,1"])
    with pytest.raises(src.EtrSchemaError, match="stat column"):
        src.read_etr_csv(path, season=2026, week=1, id_index=id_index())


def test_etr_loader_reports_unrecognized_columns_without_failing(tmp_path):
    path = write_etr(tmp_path, ETR_HEADER + ",Ownership,Tier", [ETR_ROW + ",14.2,2"])
    schema = src.validate_etr_columns(pl.read_csv(path, infer_schema_length=0).columns, path=path)
    assert schema.unknown == ("Ownership", "Tier")
    assert "receptions" in schema.fields


def test_etr_loader_needs_to_know_which_week_it_is(tmp_path):
    header = ETR_HEADER.replace("Week,", "")
    row = ",".join(ETR_ROW.split(",")[:3] + ETR_ROW.split(",")[4:])
    path = write_etr(tmp_path, header, [row], name="etr_projections.csv")
    with pytest.raises(src.EtrSchemaError, match="which season/week"):
        src.read_etr_csv(path, id_index=id_index())
    # ... and the filename is enough when it carries them.
    named = write_etr(tmp_path, header, [row], name="etr_2026_wk03.csv")
    lines = src.read_etr_csv(named, id_index=id_index())
    assert lines.lines[0].week == 3
    assert lines.lines[0].season == 2026


def test_etr_blank_cell_in_a_present_column_is_a_projected_zero(tmp_path):
    row = ETR_ROW.replace(",0.2,", ",,")
    path = write_etr(tmp_path, ETR_HEADER, [row])
    lines = src.read_etr_csv(path, season=2026, week=1, id_index=id_index())
    assert lines.lines[0].stats["43"] == 0.0


def test_etr_watched_folder_skips_files_for_other_weeks(tmp_path):
    write_etr(tmp_path, ETR_HEADER, [ETR_ROW], name="etr_2026_wk01.csv")
    write_etr(tmp_path, "Athlete,Junk", ["x,1"], name="etr_2026_wk02.csv")
    source = src.EtrCsvSource(tmp_path, id_index=id_index())
    lines = source.source_lines(2026, 1)
    assert len(lines.lines) == 1
    # The week-2 file is drifted, and asking for week 2 must surface that rather
    # than quietly returning nothing.
    with pytest.raises(src.EtrSchemaError):
        source.source_lines(2026, 2)


def test_etr_source_is_unavailable_when_nobody_downloaded_the_csv(tmp_path):
    with pytest.raises(src.SourceUnavailable):
        src.EtrCsvSource(tmp_path / "nope", id_index=id_index()).source_lines(2026, 1)
    with pytest.raises(src.SourceUnavailable):
        src.EtrCsvSource(tmp_path, id_index=id_index()).source_lines(2026, 1)


def test_etr_source_has_no_fetch_path():
    """ETR's ToS prohibits automated collection and they publish no API. There is
    nothing to build here later; this test is the reminder."""
    forbidden = {"fetch", "download", "scrape", "get", "sync"}
    assert not forbidden.intersection(dir(src.EtrCsvSource))


# --------------------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------------------


class DeadSource:
    name = "dead"

    def source_lines(self, season, week):
        raise src.SourceUnavailable("the endpoint closed")

    def component_lines(self, season, week):
        return self.source_lines(season, week).lines


class EmptySource:
    name = "empty"

    def source_lines(self, season, week):
        return src.SourceLines(source=self.name, season=season, week=week)

    def component_lines(self, season, week):
        return ()


class LiveSource:
    name = "live"

    def source_lines(self, season, week):
        return source_lines(self.name, [line(self.name, 1, {"53": 4.0})])

    def component_lines(self, season, week):
        return self.source_lines(season, week).lines


def test_collect_tolerates_a_dead_source():
    bundle = src.collect([LiveSource(), DeadSource()], 2026, 1)
    assert bundle.live == ("live",)
    assert "dead" in bundle.errors
    assert bundle.get("live") is not None


def test_collect_treats_an_empty_answer_as_a_failure():
    """HTTP 200 with an empty body is this codebase's most common failure mode.
    Alarm on empty, not on status."""
    bundle = src.collect([LiveSource(), EmptySource()], 2026, 1)
    assert bundle.errors["empty"] == "returned no lines"


def test_combine_accepts_the_collected_bundle():
    bundle = src.collect([LiveSource(), DeadSource()], 2026, 1)
    result = ens.combine(bundle.sources, season=2026, week=1)
    assert len(result) == 1
    assert result.lines[1].source_count == 1


def test_combine_carries_positions_and_names_from_the_sources():
    espn = source_lines(
        "espn", [line("espn", 1, {"53": 4.0})], positions={1: 3}, names={1: "Justin Jefferson"}
    )
    sleeper = source_lines("sleeper", [line("sleeper", 1, {"53": 6.0})])
    result = ens.combine([espn, sleeper])
    assert result.lines[1].position_id == 3
    assert result.lines[1].name == "Justin Jefferson"
    assert result.lines[1].points(FULL_PPR.scorer) == pytest.approx(5.0)


def test_a_line_with_no_position_is_reported_loudly(caplog):
    """`defaultPositionId == 0` matches no `pointsOverrides` key, so such a line
    scores off the base `points` and a TE-premium league under-scores it with no
    error anywhere. The warning is the only place it is visible."""
    with caplog.at_level("WARNING"):
        result = ens.combine_component_lines([line("props", 1, {"42": 60.0})])
    assert result.unpositioned == frozenset({1})
    assert "carry no position" in caplog.text
    with_position = ens.combine_component_lines([line("props", 1, {"42": 60.0})], positions={1: 3})
    assert with_position.unpositioned == frozenset()


def test_position_precedence_is_first_source_wins_and_the_caller_outranks_everyone(caplog):
    """`pointsOverrides` is keyed on ESPN's `defaultPositionId`, so whichever source
    wins a disagreement decides whether a TE-premium league pays TE or WR rates.

    This used to be `position_map.update(source.positions)`: the LAST adapter in
    the list won, which in `default_adapters()` order means Sleeper and ETR
    overrode ESPN, and an explicit `positions=` argument was discarded entirely.
    Measured on the live 2026 week-1 slate, ESPN calls Riley Nowakowski (4693370)
    an RB and Sleeper a TE.
    """
    espn = source_lines("espn", [line("espn", 1, {"53": 4.0})], positions={1: 2}, names={1: "ESPN"})
    sleeper = source_lines(
        "sleeper", [line("sleeper", 1, {"53": 6.0})], positions={1: 4}, names={1: "SLEEPER"}
    )

    with caplog.at_level("WARNING"):
        first_wins = ens.combine([espn, sleeper])
    assert first_wins.lines[1].position_id == 2
    assert "disputed defaultPositionId" in caplog.text
    # Order decides, and it is the first source rather than the last.
    assert ens.combine([sleeper, espn]).lines[1].position_id == 4
    # Names already worked this way; positions now match them.
    assert first_wins.lines[1].name == "ESPN"

    # An explicit map is the caller speaking, and it outranks every source.
    assert ens.combine([espn, sleeper], positions={1: 4}).lines[1].position_id == 4
    assert ens.combine([sleeper, espn], positions={1: 2}).lines[1].position_id == 2


def test_score_rebuilds_derived_stats_when_the_league_prices_one():
    """`sources.py` drops statId 47 on ingest, so a league that scores "every 5
    receiving yards" would silently be short those points. None of our three price
    one, which is exactly why the bug would never surface here."""
    settings = {
        "scoringSettings": {
            "scoringItems": [{"statId": 42, "points": 0.1}, {"statId": 47, "points": 0.5}]
        }
    }
    scorer = LeagueScoring.from_settings(settings)
    assert ens.priced_derived_stats(scorer) == frozenset({"47"})
    assert ens.priced_derived_stats(FULL_PPR.scorer) == frozenset()

    every_five = dataclasses.replace(FULL_PPR, scorer=scorer, league_id=999)
    combined = ens.combine_component_lines(
        [line("espn", 1, {"42": 63.0}), line("sleeper", 1, {"42": 67.0})], positions={1: 3}
    )
    # 65 receiving yards -> 6.5 points of yardage plus floor(65/5) = 13 buckets.
    assert combined.score(every_five) == {1: pytest.approx(6.5 + 13 * 0.5)}
    assert combined.score(every_five, derived=False) == {1: pytest.approx(6.5)}
    assert ens.rank(combined, every_five)[0].points == pytest.approx(6.5 + 6.5)
    # A league that does not price one is untouched, and pays nothing for the check.
    assert combined.score(FULL_PPR) == {1: pytest.approx(6.5)}


def test_priced_derived_stats_tolerates_a_scorer_that_cannot_answer():
    """`LeagueContext.scorer` is declared as a bare callable, so a test double is
    allowed to be one. That must degrade to today's behaviour, not raise."""
    assert ens.priced_derived_stats(lambda stats, pos: 0.0) == frozenset()
    # A zero-points item with no override is not scored, and must not trigger a rebuild.
    unscored = LeagueScoring.from_settings(
        {"scoringSettings": {"scoringItems": [{"statId": 47, "points": 0.0}]}}
    )
    assert ens.priced_derived_stats(unscored) == frozenset()
    # ... but a position override on a zero-points item is.
    overridden = LeagueScoring.from_settings(
        {
            "scoringSettings": {
                "scoringItems": [{"statId": 47, "points": 0.0, "pointsOverrides": {"4": 0.5}}]
            }
        }
    )
    assert ens.priced_derived_stats(overridden) == frozenset({"47"})


def test_rank_orders_by_league_scored_points():
    espn = source_lines(
        "espn",
        [line("espn", 1, {"53": 10.0}), line("espn", 2, {"53": 2.0})],
        positions={1: 3, 2: 3},
        names={1: "A", 2: "B"},
    )
    ranked = ens.rank(ens.combine([espn]), FULL_PPR)
    assert [r.name for r in ranked] == ["A", "B"]
    assert ranked[0].points == pytest.approx(10.0)
    assert ranked[0].source_count == 1


# --------------------------------------------------------------------------------------
# Live canaries
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_sleeper_and_espn_agree_on_the_stats_we_map():
    """The mapping is only right if the two sources' numbers land in the same
    units. Measured on 2026 week 1: every mapped statId has a median
    Sleeper/ESPN ratio inside [0.6, 1.6]. The `*_fd` keys, which look like they
    should map, sit at 1.6-2.1 -- which is why they are in SLEEPER_UNMAPPED."""
    import statistics

    if not CORPUS_2026.exists():
        pytest.skip("ESPN corpus not present")
    index = src.default_id_index(allow_download=False, season=2026)
    if index.resolver is None:
        pytest.skip("id crosswalk not cached")

    espn = src.EspnSnapshotSource(root=REPO_ROOT / "data/snapshots/espn").source_lines(2026, 1)
    sleeper = src.SleeperSource(id_index=index).source_lines(2026, 1)
    e = {line_.player_id: line_.stats for line_ in espn.lines}
    s = {line_.player_id: line_.stats for line_ in sleeper.lines}
    common = set(e) & set(s)
    assert len(common) > 200, f"only {len(common)} players joined; the crosswalk moved"

    for stat_id in sorted(set(src.SLEEPER_TO_ESPN.values()), key=int):
        ratios = [
            s[pid][stat_id] / e[pid][stat_id]
            for pid in common
            if e[pid].get(stat_id, 0.0) > 0.5 and stat_id in s[pid]
        ]
        if len(ratios) < 5:
            continue
        median = statistics.median(ratios)
        assert 0.6 < median < 1.6, f"statId {stat_id}: median ratio {median:.3f} over {len(ratios)}"


@pytest.mark.network
def test_the_whole_pipeline_serves_both_league_shapes_from_one_ensemble():
    if not CORPUS_2026.exists():
        pytest.skip("ESPN corpus not present")
    index = src.default_id_index(allow_download=False, season=2026)
    adapters = [
        src.EspnSnapshotSource(root=REPO_ROOT / "data/snapshots/espn"),
        src.SleeperSource(id_index=index),
    ]
    bundle = src.collect(adapters, 2026, 1)
    assert bundle.live == ("espn", "sleeper"), bundle.errors

    result = ens.combine(bundle.sources, season=2026, week=1)
    assert len(result) > 400
    assert max(result.coverage_histogram()) == 2

    # The identity the component-level design exists to guarantee, on real data.
    for combined in result.lines.values():
        full = combined.points(FULL_PPR.scorer)
        half = combined.points(HALF_PPR.scorer)
        receptions = combined.components["53"].value if "53" in combined.components else 0.0
        assert full - half == pytest.approx(0.5 * receptions, abs=1e-9)
