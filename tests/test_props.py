"""Props: the devig math, the ladder fit, and the traps each source hides.

Fixtures are trimmed from live payloads captured 2026-09-07, so the shapes here are the
shapes the books actually serve, not the ones the docs describe. Where those disagree
the test encodes reality -- see `test_bogus_tab_slug_is_not_empty_attachments`.
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from fantasy_quant.data.props import (
    ANYTIME_TDS,
    DEVIG_METHODS,
    PASSING_YARDS,
    RECEIVING_YARDS,
    RECEPTIONS,
    RUSHING_TDS,
    AnytimeTd,
    FanDuelEventProps,
    FanDuelProps,
    LadderFit,
    LadderRung,
    PropLadder,
    PropLine,
    PropsError,
    american_to_implied,
    collect_props,
    components_by_player,
    devig,
    devig_two_sided,
    expected_touchdowns,
    fanduel_stat,
    fetch_espn_prop_moves,
    fetch_pinnacle_player_props,
    fetch_underdog,
    fit_ladder_with_line,
    fit_lognormal_ladder,
    implied_to_american,
    naive_ladder_mean,
    parse_pinnacle,
    parse_underdog,
    project_event,
    project_fantasy_points,
    split_anytime_tds,
    touchdown_probability,
)

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

# Cooper Kupp - Alt Receiving Yds, FanDuel NJ, NE @ SEA, captured 2026-09-07. Thirteen
# rungs against a two-sided main line of 29.5 at -114/-114. This is the ladder shape
# RESEARCH.md's worked example describes (posted median 30.5 -> fitted median 30.9,
# fitted mean 43.6).
KUPP_RUNGS = [
    LadderRung(5, -1600),
    LadderRung(10, -650),
    LadderRung(15, -360),
    LadderRung(20, -240),
    LadderRung(25, -160),
    LadderRung(30, -114),
    LadderRung(40, 154),
    LadderRung(50, 260),
    LadderRung(60, 400),
    LadderRung(70, 630),
    LadderRung(80, 880),
    LadderRung(90, 1200),
    LadderRung(100, 1700),
]


def _fd_runner(name: str, odds: int, handicap: float | None = None, side: str = "") -> dict:
    runner: dict[str, Any] = {
        "runnerName": name,
        "winRunnerOdds": {"americanDisplayOdds": {"americanOdds": odds, "americanOddsInt": odds}},
    }
    if handicap is not None:
        runner["handicap"] = handicap
    if side:
        runner["result"] = {"type": side}
    return runner


FD_ALT_MARKET = {
    "marketId": "734.181887920",
    "marketName": "Cooper Kupp - Alt Receiving Yds",
    "marketType": "PLAYER_X_ALT_RECEIVING_YARDS_HIGH",
    "runners": [
        _fd_runner(f"Cooper Kupp {int(r.threshold)}+ Yards", int(r.american)) for r in KUPP_RUNGS
    ],
}

FD_TWO_SIDED_MARKET = {
    "marketId": "734.181887919",
    "marketName": "Cooper Kupp - Receiving Yds",
    "marketType": "PLAYER_X_RECEIVING_YARDS_HIGH",
    "runners": [
        _fd_runner("Cooper Kupp Over", -114, handicap=29.5, side="OVER"),
        _fd_runner("Cooper Kupp Under", -114, handicap=29.5, side="UNDER"),
    ],
}

FD_ANYTIME_TD_MARKET = {
    "marketName": "Any Time Touchdown Scorer",
    "marketType": "ANY_TIME_TOUCHDOWN_SCORER",
    "runners": [
        _fd_runner("Rhamondre Stevenson", 120),
        _fd_runner("Cooper Kupp", 250),
    ],
}

# What a *wrong* slug actually returns: HTTP 200, a non-empty market set, and not one
# player prop in it. Captured from `tab=bogus-slug`.
FD_FALLBACK_MARKETS = [
    {"marketName": "Total Points", "marketType": "TOTAL_POINTS_(OVER/UNDER)", "runners": []},
    {"marketName": "Spread", "marketType": "MATCH_HANDICAP_(2-WAY)", "runners": []},
    {"marketName": "Moneyline", "marketType": "MONEY_LINE", "runners": []},
]


def _fd_client(pages: dict[str, dict[str, Any]]) -> httpx.Client:
    """A FanDuel stub keyed on the `tab` query parameter (or 'events' for the NFL page)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "content-managed-page" in request.url.path:
            return httpx.Response(200, json=pages["events"])
        tab = request.url.params.get("tab", "")
        if tab not in pages:
            return httpx.Response(503, text="host down")
        return httpx.Response(200, json=pages[tab])

    return httpx.Client(transport=httpx.MockTransport(handler))


def _fd_page(markets: list[dict]) -> dict:
    return {"layout": {}, "attachments": {"markets": {str(i): m for i, m in enumerate(markets)}}}


# --------------------------------------------------------------------------------------
# American odds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("odds", "expected"),
    [
        (100, 0.5),
        (-100, 0.5),
        (150, 0.4),
        (-110, 0.5238095238095238),
        (-200, 2 / 3),
        (400, 0.2),
        (-1600, 16 / 17),
    ],
)
def test_american_to_implied(odds: float, expected: float) -> None:
    assert american_to_implied(odds) == pytest.approx(expected)


@pytest.mark.parametrize("odds", [0, float("inf"), float("nan")])
def test_american_to_implied_rejects_nonsense(odds: float) -> None:
    with pytest.raises(ValueError, match="American odds"):
        american_to_implied(odds)


@pytest.mark.parametrize("odds", [-500, -110, 120, 150, 1700])
def test_american_odds_round_trip(odds: float) -> None:
    assert implied_to_american(american_to_implied(odds)) == pytest.approx(odds)


def test_even_money_canonicalizes_to_plus_100() -> None:
    """-100 and +100 are the same price; the round trip picks the conventional spelling."""
    assert american_to_implied(-100) == american_to_implied(100) == 0.5
    assert implied_to_american(0.5) == 100.0


# --------------------------------------------------------------------------------------
# Devigging
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", DEVIG_METHODS)
def test_symmetric_pair_devigs_to_a_coin_flip(method: str) -> None:
    """-110/-110 prices to 1.0476; every method must return exactly 0.5/0.5."""
    result = devig_two_sided(30.5, -110, -110, method)
    assert result.overround == pytest.approx(1.0476190476190477)
    assert result.vig_pct == pytest.approx(4.761904761904767)
    assert result.p_over == pytest.approx(0.5)
    assert result.p_under == pytest.approx(0.5)


@pytest.mark.parametrize("method", DEVIG_METHODS)
def test_devig_is_a_probability_distribution(method: str) -> None:
    fair = devig([american_to_implied(400), american_to_implied(-600)], method)
    assert sum(fair) == pytest.approx(1.0)
    assert all(0.0 < p < 1.0 for p in fair)


def test_multiplicative_devig_is_hand_computable() -> None:
    # +400 -> 0.2, -600 -> 6/7. Sum 1.0571428...; proportional shares are p_i / sum.
    raw = [0.2, 6 / 7]
    over, under = devig(raw, "multiplicative")
    assert over == pytest.approx(0.2 / (0.2 + 6 / 7))
    assert over == pytest.approx(0.1891891891891892)
    assert under == pytest.approx(0.8108108108108109)


def test_additive_devig_is_hand_computable() -> None:
    # Subtract half the overround from each side: 0.2 - 0.0571.../2.
    raw = [0.2, 6 / 7]
    excess = sum(raw) - 1.0
    over, under = devig(raw, "additive")
    assert over == pytest.approx(0.2 - excess / 2)
    assert under == pytest.approx(6 / 7 - excess / 2)


def test_shin_and_additive_coincide_on_two_way_markets() -> None:
    """A real result, not an accident: for n == 2 Shin reduces to the balanced book.

    Verified against live FanDuel and Pinnacle pairs. It is why implementing both is
    free, and why the default choice between them turns on numerical robustness rather
    than on the answers differing.
    """
    for pair in [(-110, -110), (400, -600), (150, -190), (-2000, 1000), (1700, -3000)]:
        raw = [american_to_implied(pair[0]), american_to_implied(pair[1])]
        assert devig(raw, "shin") == pytest.approx(devig(raw, "additive"))


def test_shin_shrinks_the_longshot_harder_than_multiplicative() -> None:
    """The favourite-longshot correction. It is why Shin is the default."""
    raw = [american_to_implied(400), american_to_implied(-600)]
    assert devig(raw, "shin")[0] < devig(raw, "multiplicative")[0]
    assert devig(raw, "power")[0] < devig(raw, "shin")[0]


def test_shin_falls_back_when_the_solve_will_not_bracket() -> None:
    """A pathological book has no Shin root; additive would go negative. Neither is fatal."""
    raw = [0.02, 1.04]
    fair = devig(raw, "shin")
    assert fair == pytest.approx(devig(raw, "multiplicative"))
    assert all(p > 0.0 for p in fair)


def test_underround_is_renormalized_proportionally_not_left_alone() -> None:
    """An underround has no margin to remove, so the shares survive and the level does not.

    Shin/additive/power are all derived from a positive overround and do not apply, but
    the caller still needs a distribution, so the two prices are scaled UP to sum to 1.
    Naming this "left alone" would be wrong: 0.40 comes back as 0.421.
    """
    fair = devig([0.40, 0.55], "shin")
    assert sum(fair) == pytest.approx(1.0)
    assert fair[0] == pytest.approx(0.40 / 0.95)
    assert fair[0] > 0.40, "an underround is scaled up, not preserved"
    # Shares are preserved even though levels are not.
    assert fair[0] / fair[1] == pytest.approx(0.40 / 0.55)


def test_unknown_devig_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown devig method"):
        devig([0.55, 0.55], "vibes")


def test_devig_needs_at_least_two_outcomes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        devig([0.55])


# --------------------------------------------------------------------------------------
# Anytime touchdowns -- devig FIRST, then take the log
# --------------------------------------------------------------------------------------


def test_documented_anytime_td_case() -> None:
    """+150: naive read 0.400 expected TDs, correct read 0.511, +27.7%."""
    naive = american_to_implied(150)
    assert naive == pytest.approx(0.400)

    lam = expected_touchdowns(150)
    assert lam == pytest.approx(0.5108256237659907)
    assert lam / naive - 1.0 == pytest.approx(0.2771, abs=1e-4)


def test_devig_before_log_is_not_the_same_as_log_before_devig() -> None:
    """The ordering trap. Doing it backwards systematically over-rates the short prices.

    lambda = -ln(1 - p) is convex in p, so shading p down first and shading lambda down
    afterwards are different operations -- and the gap widens exactly where goal-line
    backs live.
    """
    overround = 1.07
    right = expected_touchdowns(150, overround=overround)
    wrong = -math.log1p(-american_to_implied(150)) / overround

    assert right == pytest.approx(0.4681362150709401)
    assert wrong == pytest.approx(0.4774071250149446)
    assert right < wrong  # the wrong order inflates every anytime-TD price


def test_anytime_td_with_a_real_no_td_side_uses_a_real_devig() -> None:
    """When the book posts both sides there is nothing to assume; it is a two-way devig."""
    lam = expected_touchdowns(150, no_td_odds=-190)
    p_fair = devig_two_sided(0.5, 150, -190).p_over
    assert lam == pytest.approx(-math.log1p(-p_fair))
    assert lam < expected_touchdowns(150)


def test_touchdown_probability_inverts_the_poisson_conversion() -> None:
    assert touchdown_probability(expected_touchdowns(150)) == pytest.approx(0.4)
    assert touchdown_probability(0.0) == 0.0


def test_split_anytime_tds() -> None:
    rush, rec = split_anytime_tds(0.5, 0.8)
    assert (rush, rec) == pytest.approx((0.4, 0.1))
    with pytest.raises(ValueError, match="rush_share"):
        split_anytime_tds(0.5, 1.4)


def test_expected_touchdowns_rejects_impossible_inputs() -> None:
    # Over-devigging a near-certainty pushes p past 1, where the log blows up.
    with pytest.raises(ValueError, match="out of range"):
        expected_touchdowns(-100000, overround=0.9)
    with pytest.raises(ValueError, match="overround must be positive"):
        expected_touchdowns(150, overround=0.0)


# --------------------------------------------------------------------------------------
# Ladder -> distribution
# --------------------------------------------------------------------------------------


def test_documented_ladder_worked_example() -> None:
    """The whole thesis in one assertion: a 30.5 median implies a ~43 mean.

    RESEARCH.md's case is posted median 30.5 -> fitted median 30.9, fitted mean 43.6
    (+41%). Its ladder was posted at 30.5; this one is posted at 29.5 and re-anchored,
    so the levels land ~1.5% low while the claim that matters -- the skew premium --
    reproduces almost exactly (1.407 against the documented 1.410). The premium is
    therefore pinned tightly and the levels loosely, rather than the other way round.
    """
    anchor = devig_two_sided(30.5, -110, -110)
    fit = fit_ladder_with_line(KUPP_RUNGS, anchor)

    assert fit.n_rungs == 13
    assert fit.mean / 30.5 == pytest.approx(1.41, abs=0.03), "the +41% skew premium"
    assert fit.median == pytest.approx(30.4, abs=0.5)
    assert fit.mean == pytest.approx(42.9, abs=1.5)
    assert fit.sigma == pytest.approx(0.83, abs=0.05)
    assert fit.rms_error < 0.05
    # A well-behaved ladder leaves the shading parameter room to move.
    assert not fit.overround_pinned


def test_ladder_fit_exposes_the_whole_distribution_not_a_point_estimate() -> None:
    fit = fit_ladder_with_line(KUPP_RUNGS, devig_two_sided(29.5, -114, -114))

    # Lognormal identities, so the simulator can draw rather than take a point estimate.
    assert fit.median == pytest.approx(math.exp(fit.mu))
    assert fit.mean == pytest.approx(math.exp(fit.mu + fit.sigma**2 / 2))
    assert fit.variance == pytest.approx(
        (math.exp(fit.sigma**2) - 1) * math.exp(2 * fit.mu + fit.sigma**2)
    )
    assert fit.sd == pytest.approx(math.sqrt(fit.variance))
    assert fit.mean_over_median == pytest.approx(fit.mean / fit.median)

    assert fit.quantile(0.5) == pytest.approx(fit.median)
    assert fit.survival(fit.quantile(0.25)) == pytest.approx(0.75)
    assert fit.survival(0.0) == 1.0
    assert fit.survival(10.0) > fit.survival(100.0)


def _synthetic_ladder(
    mu: float, sigma: float, shade: float, thresholds: list[int]
) -> list[LadderRung]:
    """Rungs a book would post for lognormal(mu, sigma) shaded by `shade`.

    The rung "k+" is priced on P(X >= k), and X is integer-valued, so the true
    probability behind that price is S(k - 0.5) -- NOT S(k). Generating the fixture the
    other way round would bake the fit's own convention into its ground truth and the
    test could never catch a wrong one.
    """
    truth = LadderFit(mu, sigma, 1.0, 0, 0.0)
    return [LadderRung(k, implied_to_american(shade * truth.survival(k - 0.5))) for k in thresholds]


def test_fit_recovers_a_known_lognormal_through_a_uniform_overround() -> None:
    """Synthetic ground truth: rungs generated from lognormal(ln 60, 0.9), shaded 7%.

    If the fit cannot invert a shading it *knows* is there, it will not invert a real
    book's. Rungs start at 20 so the shaded probabilities stay well below 1.
    """
    mu, sigma, shade = math.log(60.0), 0.9, 1.07
    thresholds = [20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]
    rungs = _synthetic_ladder(mu, sigma, shade, thresholds)

    fit = fit_lognormal_ladder(rungs, anchor_line=60.0, anchor_prob=0.5)
    assert fit.mu == pytest.approx(mu, abs=0.02)
    assert fit.sigma == pytest.approx(sigma, abs=0.03)
    assert fit.overround == pytest.approx(shade, abs=0.02)
    assert fit.median == pytest.approx(60.0, rel=0.02)
    assert fit.mean == pytest.approx(math.exp(mu + sigma**2 / 2), rel=0.04)


def test_dropping_the_continuity_correction_corrupts_the_recovered_overround() -> None:
    """The half-unit correction is load-bearing, and this is the falsification test.

    Same synthetic ladder, but on a low-count stat where the rungs sit one unit apart.
    With `continuity=0.5` the fit inverts the 7% shading it was given. With
    `continuity=0.0` -- asking the model for P(X >= k+1) against a price for
    P(X >= k) -- the shading parameter absorbs the off-by-one and runs away, because a
    nuisance parameter with nothing else to do will always eat a systematic bias.
    """
    mu, sigma, shade = math.log(3.5), 0.65, 1.07
    rungs = _synthetic_ladder(mu, sigma, shade, [2, 3, 4, 5, 6, 7, 8])

    good = fit_lognormal_ladder(rungs, anchor_line=3.5, anchor_prob=0.5, continuity=0.5)
    assert good.overround == pytest.approx(shade, abs=0.001)
    assert good.sigma == pytest.approx(sigma, abs=0.001)
    assert not good.overround_pinned
    assert good.rms_error < 1e-6, "exact ground truth is recovered exactly"

    naive = fit_lognormal_ladder(rungs, anchor_line=3.5, anchor_prob=0.5, continuity=0.0)
    assert naive.overround > 1.20, "the bias has nowhere to go but the shading parameter"
    assert naive.rms_error > 0.01


def test_rungs_sit_above_the_fitted_curve_because_each_carries_its_own_vig() -> None:
    """The compounding-overround trap, stated as an inequality.

    The posted 30+ rung is -114, an implied 0.533. It prices X >= 30, which is the same
    event as the OVER at 29.5 -- and FanDuel posts that OVER at -114 too, the identical
    price. The *devigged* two-sided market at 29.5 says the fair number is 0.500. That
    3.3-point gap is one rung's margin; summing thirteen of them is how naive
    integration ends up over the truth with no modelling error to blame it on.

    The comparison has to be made at `rung_survival(k)`, not `survival(k)`: the latter
    is P(X >= k + 1) and would flatter the inequality by half a rung of pure convention.

    The inequality is asserted only over the body of the ladder. In the tails the fitted
    curve runs ABOVE the posted rungs -- a real limitation, not a bug: a lognormal has no
    atom at zero, so it cannot represent a receiver's ~6% chance of being blanked and it
    over-states every low rung. That misfit is what `LadderFit.rms_error` is for.
    """
    fit = fit_ladder_with_line(KUPP_RUNGS, devig_two_sided(29.5, -114, -114))

    posted_at_30 = american_to_implied(-114)
    assert posted_at_30 == pytest.approx(0.5327, abs=1e-3)
    # The rung and the anchor price the same event, so they are compared at the same x.
    assert fit.rung_survival(30.0) == pytest.approx(fit.survival(29.5))
    assert fit.rung_survival(30.0) < posted_at_30

    body = [r for r in KUPP_RUNGS if 20 <= r.threshold <= 50]
    assert len(body) == 5
    assert all(r.implied > fit.rung_survival(r.threshold) for r in body)

    # Both tails run the other way -- the lognormal has no atom at zero.
    assert KUPP_RUNGS[0].implied < fit.rung_survival(KUPP_RUNGS[0].threshold)
    assert KUPP_RUNGS[-1].implied < fit.rung_survival(KUPP_RUNGS[-1].threshold)

    assert naive_ladder_mean(KUPP_RUNGS) > fit.mean


def test_the_anchor_sets_the_level_and_the_rungs_set_the_shape() -> None:
    """Move only the anchor and the median follows it; the skew premium barely moves."""
    low = fit_ladder_with_line(KUPP_RUNGS, devig_two_sided(25.0, -110, -110))
    high = fit_ladder_with_line(KUPP_RUNGS, devig_two_sided(40.0, -110, -110))

    assert low.median < high.median
    assert low.median == pytest.approx(25.0, rel=0.12)
    assert high.median == pytest.approx(40.0, rel=0.12)
    assert low.mean_over_median == pytest.approx(high.mean_over_median, rel=0.25)


def test_an_asymmetric_anchor_is_not_forced_to_a_median() -> None:
    """S(line) == p_over, not 0.5 -- a -150/+125 pair is not a coin flip."""
    anchor = devig_two_sided(29.5, -150, 125)
    assert anchor.p_over > 0.5
    fit = fit_ladder_with_line(KUPP_RUNGS, anchor)
    assert fit.survival(29.5) == pytest.approx(anchor.p_over, abs=0.02)
    assert fit.median > 29.5


def test_fit_without_an_anchor_still_works() -> None:
    """A ladder with no matched two-sided market is degraded, not unusable."""
    fit = fit_lognormal_ladder(KUPP_RUNGS)
    assert fit.mean > fit.median > 0
    assert fit_ladder_with_line(KUPP_RUNGS, None).mu == pytest.approx(fit.mu)


def test_short_ladders_and_junk_thresholds_are_refused_not_fudged() -> None:
    with pytest.raises(PropsError, match="usable rungs"):
        fit_lognormal_ladder(KUPP_RUNGS[:3])

    # A "0+" rung carries no information and lognormal support starts above zero.
    with_zero = [LadderRung(0, -5000), *KUPP_RUNGS]
    assert fit_lognormal_ladder(with_zero, anchor_line=29.5).n_rungs == 13

    # Duplicated thresholds must not double-weight a point.
    doubled = [*KUPP_RUNGS, LadderRung(30, -114)]
    assert fit_lognormal_ladder(doubled, anchor_line=29.5).n_rungs == 13


def test_naive_ladder_mean_needs_something_to_integrate() -> None:
    with pytest.raises(PropsError, match="two rungs"):
        naive_ladder_mean(KUPP_RUNGS[:1])


def test_a_ladder_that_is_not_a_distribution_raises_instead_of_returning_a_number() -> None:
    """A flat or inverted ladder pins sigma at its bound and yields a 2,000-yard mean.

    Before the guard both of these returned a `LadderFit` indistinguishable in type from
    a good one, with sigma == 3.0 and mean ~= 2,290 -- median * exp(sigma^2/2) is 90x at
    the ceiling. `project_event` catches `PropsError`, so a suspended market now drops
    out of the pool with a log line instead of poisoning it.
    """
    flat = [LadderRung(k, -110) for k in (10, 20, 30, 40, 50)]
    with pytest.raises(PropsError, match="degenerate"):
        fit_lognormal_ladder(flat, anchor_line=25.0)

    inverted = [
        LadderRung(10, 1700),
        LadderRung(20, 880),
        LadderRung(30, 400),
        LadderRung(40, -160),
        LadderRung(50, -650),
    ]
    with pytest.raises(PropsError, match="degenerate"):
        fit_lognormal_ladder(inverted, anchor_line=30.0)


def test_a_pinned_overround_is_visible_to_the_caller() -> None:
    """`overround_pinned` exists so nobody has to hard-code 1.30 to spot a saturated fit."""
    fit = fit_ladder_with_line(KUPP_RUNGS, devig_two_sided(29.5, -114, -114))
    assert fit.max_overround == 1.30
    assert not fit.overround_pinned

    # A ladder shaded 7% against a ceiling of 2%: the solve runs out of room and says so.
    shaded = _synthetic_ladder(math.log(60.0), 0.9, 1.07, [20, 30, 40, 50, 60, 80, 100, 150])
    squeezed = fit_lognormal_ladder(shaded, anchor_line=60.0, max_overround=1.02)
    assert squeezed.overround == pytest.approx(1.02)
    assert squeezed.overround_pinned
    assert squeezed.rms_error > fit_lognormal_ladder(shaded, anchor_line=60.0).rms_error


def test_a_ceiling_below_the_seed_is_rejected_not_crashed_on() -> None:
    """max_overround < 1.0 is meaningless; a ceiling under the solver's seed is not."""
    with pytest.raises(ValueError, match="max_overround"):
        fit_lognormal_ladder(KUPP_RUNGS, max_overround=0.9)
    # 1.01 sits below the 1.05 seed -- scipy would raise "initial guess outside bounds".
    assert fit_lognormal_ladder(KUPP_RUNGS, anchor_line=29.5, max_overround=1.01).overround <= 1.01


def test_a_continuity_correction_cannot_push_a_rung_to_or_below_zero() -> None:
    """A "0.5+" rung would land on log(0). It is dropped, like the "0+" rung."""
    rungs = [LadderRung(0.5, -5000), *KUPP_RUNGS]
    assert fit_lognormal_ladder(rungs, anchor_line=29.5).n_rungs == 13
    assert fit_lognormal_ladder(rungs, anchor_line=29.5, continuity=0.0).n_rungs == 14


def test_continuity_must_be_a_sub_unit_correction() -> None:
    with pytest.raises(ValueError, match="continuity"):
        fit_lognormal_ladder(KUPP_RUNGS, continuity=1.0)


# --------------------------------------------------------------------------------------
# FanDuel
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("market_type", "expected"),
    [
        ("PLAYER_X_RECEIVING_YARDS_HIGH", RECEIVING_YARDS),
        ("PLAYER_X_ALT_RECEIVING_YARDS_LOW", RECEIVING_YARDS),
        ("PLAYER_X_ALT_RECEPTIONS_MEDIUM", RECEPTIONS),
        ("PLAYER_X_ALT_PASSING_YARDS_HIGH", PASSING_YARDS),
        ("ANY_TIME_TOUCHDOWN_SCORER", None),
        ("PLAYERS_WITH_20+_YARDS_RECEPTION", None),
    ],
)
def test_fanduel_stat_collapses_alt_and_tier_variants(
    market_type: str, expected: str | None
) -> None:
    """HIGH/MEDIUM/LOW is a display tier, not a different stat, and ALT is the same market."""
    assert fanduel_stat(market_type) == expected


def test_bogus_tab_slug_is_not_empty_attachments() -> None:
    """RESEARCH.md says a bad slug returns 200 with EMPTY attachments. It does not.

    Live traffic returns 200 with a non-empty fallback set of game markets and a layout
    block identical to a good response, so `len(markets) > 0` passes on a typo. The check
    that actually works is per-slug: does this tab contain the markets it promises?
    """
    pages = {"events": {}, "bogus-slug": _fd_page(FD_FALLBACK_MARKETS)}
    with FanDuelProps(hosts=["nj"], client=_fd_client(pages)) as fd:
        # Unknown slugs are refused outright -- nothing in the body can validate them.
        with pytest.raises(PropsError, match="unknown FanDuel tab slug"):
            fd.markets(1, "bogus-slug")

        # A known slug served the fallback set is caught by the content check.
        with pytest.raises(PropsError, match="none matching"):
            fd.markets(1, "bogus-slug", required=("PLAYER_X_",))

        # Opting out is possible but explicit.
        assert len(fd.markets(1, "bogus-slug", required=())) == 3


def test_genuinely_empty_attachments_still_raises() -> None:
    pages = {"events": {}, "receiving-props": {"layout": {}, "attachments": {"markets": {}}}}
    with (
        FanDuelProps(hosts=["nj"], client=_fd_client(pages)) as fd,
        pytest.raises(PropsError, match="empty attachments"),
    ):
        fd.markets(1, "receiving-props")


def test_fanduel_falls_back_across_state_hosts() -> None:
    """One dead state must not read as a dead book."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host.startswith("sbapi.nj"):
            return httpx.Response(451, text="geo")
        return httpx.Response(200, json=_fd_page([FD_TWO_SIDED_MARKET]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with FanDuelProps(hosts=["nj", "va", "oh"], client=client) as fd:
        assert len(fd.markets(1, "receiving-props")) == 1
        # The working host is promoted, so the dead one is not retried every call.
        fd.markets(1, "receiving-props")
    assert calls[0].startswith("sbapi.nj")
    assert calls[-1].startswith("sbapi.va")
    assert sum(host.startswith("sbapi.nj") for host in calls) == 1


def test_every_host_failing_raises_with_all_the_reasons() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with (
        FanDuelProps(hosts=["nj", "va"], client=client) as fd,
        pytest.raises(PropsError, match="every FanDuel host failed"),
    ):
        fd.markets(1, "receiving-props")


def test_event_props_parses_ladders_lines_and_touchdowns() -> None:
    pages = {
        "events": {},
        "receiving-props": _fd_page([FD_ALT_MARKET, FD_TWO_SIDED_MARKET]),
        "td-scorer-props": _fd_page([FD_ANYTIME_TD_MARKET]),
    }
    with FanDuelProps(hosts=["nj"], client=_fd_client(pages)) as fd:
        props = fd.event_props(35607262, tabs=("receiving-props", "td-scorer-props"))

    assert props.tabs_seen == ("receiving-props", "td-scorer-props")

    (ladder,) = props.ladders
    assert (ladder.player, ladder.stat) == ("Cooper Kupp", RECEIVING_YARDS)
    assert [r.threshold for r in ladder.rungs] == [float(r.threshold) for r in KUPP_RUNGS]
    assert ladder.rungs[5].american == -114

    (line,) = props.lines
    assert (line.player, line.line, line.over_odds, line.under_odds) == (
        "Cooper Kupp",
        29.5,
        -114,
        -114,
    )

    assert {td.player: td.american for td in props.touchdowns} == {
        "Rhamondre Stevenson": 120,
        "Cooper Kupp": 250,
    }


def test_the_longest_copy_of_a_ladder_wins() -> None:
    """The same ALT market is served on several tabs and the copies are not always equal."""
    truncated = {**FD_ALT_MARKET, "runners": FD_ALT_MARKET["runners"][:5]}
    pages = {
        "events": {},
        "popular": _fd_page([truncated]),
        "receiving-props": _fd_page([FD_ALT_MARKET]),
    }
    with FanDuelProps(hosts=["nj"], client=_fd_client(pages)) as fd:
        props = fd.event_props(1, tabs=("popular", "receiving-props"))
    (ladder,) = props.ladders
    assert len(ladder.rungs) == 13


def test_one_dead_tab_does_not_take_the_event_down() -> None:
    """Sources are independently optional, and so are the tabs within one source."""
    pages = {"events": {}, "receiving-props": _fd_page([FD_ALT_MARKET, FD_TWO_SIDED_MARKET])}
    with FanDuelProps(hosts=["nj"], client=_fd_client(pages)) as fd:
        props = fd.event_props(1, tabs=("receiving-props", "rushing-props"))
    assert props.tabs_seen == ("receiving-props",)
    assert len(props.ladders) == 1

    with (
        FanDuelProps(hosts=["nj"], client=_fd_client({"events": {}})) as fd,
        pytest.raises(PropsError, match="no tab served"),
    ):
        fd.event_props(1, tabs=("receiving-props",))


def test_events_filters_futures_and_specials() -> None:
    page = {
        "attachments": {
            "events": {
                "1": {"eventId": 1, "name": "NFL Futures", "openDate": "2026-03-06T00:00:00Z"},
                "2": {
                    "eventId": 2,
                    "name": "Dallas Cowboys @ New York Giants",
                    "openDate": "2026-09-14T00:20:00Z",
                },
                "3": {
                    "eventId": 3,
                    "name": "New England Patriots @ Seattle Seahawks",
                    "openDate": "2026-09-10T00:15:00Z",
                },
            }
        }
    }
    with FanDuelProps(hosts=["nj"], client=_fd_client({"events": page})) as fd:
        events = fd.events()
    assert [e.event_id for e in events] == [3, 2]  # sorted by kickoff
    assert events[0].teams == ("New England Patriots", "Seattle Seahawks")


# --------------------------------------------------------------------------------------
# Underdog
# --------------------------------------------------------------------------------------

UD_PAYLOAD = {
    "players": [
        {
            "id": "p-nfl",
            "first_name": "A.J.",
            "last_name": "Brown",
            "position_name": "WR",
            "team_id": "t-phi",
            "sport_id": "NFL",
        },
        {
            "id": "p-mlb",
            "first_name": "Shohei",
            "last_name": "Ohtani",
            "position_name": "Designated Hitter",
            "team_id": "t-lad",
            "sport_id": "MLB",
        },
    ],
    "appearances": [
        {"id": "a-nfl", "player_id": "p-nfl"},
        {"id": "a-mlb", "player_id": "p-mlb"},
    ],
    "over_under_lines": [
        {
            "stat_value": "11.55",
            "status": "active",
            "options": [
                {"choice": "higher", "american_price": "-112"},
                {"choice": "lower", "american_price": "-112"},
            ],
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a-nfl",
                    "display_stat": "Fantasy Points",
                }
            },
        },
        {
            "stat_value": "9.50",
            "status": "active",
            "options": [
                {"choice": "higher", "american_price": "-120"},
                {"choice": "lower", "american_price": "-105"},
            ],
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a-mlb",
                    "display_stat": "Fantasy Points",
                }
            },
        },
        {
            "stat_value": "72.5",
            "status": "active",
            "options": [{"choice": "higher", "american_price": "-115"}],
            "over_under": {
                "appearance_stat": {
                    "appearance_id": "a-nfl",
                    "display_stat": "Receiving Yards",
                }
            },
        },
    ],
}


def test_underdog_must_be_filtered_by_sport_not_just_by_stat_name() -> None:
    """The trap: one URL serves every sport, and "Fantasy Points" exists in all of them.

    A live pull carried 235 Fantasy Points lines -- 122 NFL, 106 MLB, 7 CFB. Filtering on
    `display_stat` alone drags baseball into the football projections, and since Underdog
    publishes only its own UUIDs those rows name-match against nothing and disappear
    silently instead of failing.
    """
    lines = parse_underdog(UD_PAYLOAD)
    assert [line.player for line in lines] == ["A.J. Brown"]

    assert [line.player for line in parse_underdog(UD_PAYLOAD, sport="MLB")] == ["Shohei Ohtani"]


def test_underdog_line_parsing_and_devig() -> None:
    (line,) = parse_underdog(UD_PAYLOAD)
    assert (line.line, line.position, line.stat) == (11.55, "WR", "fantasy_points")
    assert (line.over_odds, line.under_odds) == (-112.0, -112.0)

    fair = line.devigged()
    assert fair is not None
    assert fair.p_over == pytest.approx(0.5)
    assert fair.overround > 1.0


def test_underdog_can_pull_other_stats_and_tolerates_a_missing_side() -> None:
    lines = parse_underdog(UD_PAYLOAD, stats=None)
    stats = {line.stat for line in lines}
    assert stats == {"fantasy_points", RECEIVING_YARDS}
    one_sided = next(line for line in lines if line.stat == RECEIVING_YARDS)
    assert one_sided.under_odds is None
    assert one_sided.devigged() is None


def test_underdog_cache_is_used_and_repaired(tmp_path) -> None:
    cache = tmp_path / "underdog.json"
    cache.write_text(json.dumps(UD_PAYLOAD))
    # A fresh cache short-circuits the network entirely; no client is even constructed.
    assert fetch_underdog(cache_path=cache)["players"][0]["id"] == "p-nfl"

    cache.write_text("{not json")
    with pytest.raises(PropsError):
        # Corrupt cache falls through to the network, which we deny by pointing at a
        # transport that always fails.
        fetch_underdog(
            cache_path=cache,
            book=_book_that_always_fails(),
        )


def _book_that_always_fails():
    from fantasy_quant.data.props import _Book

    return _Book(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))


# --------------------------------------------------------------------------------------
# Pinnacle
# --------------------------------------------------------------------------------------

PIN_MATCHUPS = [
    {
        "id": 1636006629,
        "parentId": 1630865291,
        "type": "special",
        "units": "Receptions",
        "special": {"category": "Player Props", "description": "Quentin Johnston Total Receptions"},
        "participants": [
            {"id": 1636006632, "name": "Over"},
            {"id": 1636006633, "name": "Under"},
        ],
    },
    {
        "id": 1636006700,
        "parentId": 1630865291,
        "type": "special",
        "units": "Regular",
        "special": {"category": "Season Long Player Props", "description": "Someone Total Wins"},
        "participants": [{"id": 1, "name": "Over"}, {"id": 2, "name": "Under"}],
    },
]

PIN_MARKETS = [
    {
        "matchupId": 1636006629,
        "type": "total",
        "period": 0,
        "key": "s;0;ou",
        "limits": [{"amount": 250, "type": "maxRiskStake"}],
        "prices": [
            {"participantId": 1636006632, "points": 3.5, "price": 116},
            {"participantId": 1636006633, "points": 3.5, "price": -154},
        ],
    },
    {
        # First half. Joining without the period filter folds this into the full-game line.
        "matchupId": 1636006629,
        "type": "total",
        "period": 1,
        "key": "s;1;ou",
        "prices": [
            {"participantId": 1636006632, "points": 1.5, "price": -110},
            {"participantId": 1636006633, "points": 1.5, "price": -110},
        ],
    },
]


def test_pinnacle_join_keeps_full_game_periods_only() -> None:
    """The markets feed carries several rows per matchup; period 0 is the fantasy-relevant one."""
    (prop,) = parse_pinnacle(PIN_MATCHUPS, PIN_MARKETS)
    assert prop.player == "Quentin Johnston"
    assert prop.stat == RECEPTIONS
    assert prop.line == 3.5
    assert (prop.over_odds, prop.under_odds) == (116.0, -154.0)
    assert prop.max_risk_stake == 250.0
    assert prop.event_id == 1630865291

    fair = prop.devigged()
    assert fair.overround == pytest.approx(1.0692621755613883)
    assert fair.p_over + fair.p_under == pytest.approx(1.0)
    assert fair.p_over < american_to_implied(116)


def test_pinnacle_drops_matchups_with_no_market() -> None:
    assert parse_pinnacle(PIN_MATCHUPS, []) == []
    assert parse_pinnacle([], PIN_MARKETS) == []


# --------------------------------------------------------------------------------------
# ESPN propBets -- movement, not levels
# --------------------------------------------------------------------------------------

ESPN_ITEMS = [
    {
        "athlete": {"$ref": "http://sports.core.api.espn.com/v2/.../athletes/3912547?lang=en"},
        "type": {"id": "8", "name": "Total Passing Yards (incl. overtime)"},
        "lastUpdated": "2026-09-07T19:12Z",
        "current": {"target": {"value": 226.5}},
        "open": {"target": {"value": 229.5}},
    },
    {
        # A team prop: same feed, no athlete ref.
        "type": {"id": "40", "name": "Team Total Points"},
        "current": {"target": {"value": 24.5}},
    },
    {
        "athlete": {"$ref": "http://sports.core.api.espn.com/v2/.../athletes/4361307?lang=en"},
        "type": {"id": "9", "name": "Passing Yards Milestones - 1st Half"},
        "current": {"target": {"value": 120.5}},
    },
]


class _StubEspn:
    """Enough of EspnClient to exercise the pager without touching the network."""

    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages
        self.requested: list[int] = []
        self.closed = False

    def get(self, url: str, params: dict | None = None, **_: object):
        page = int((params or {}).get("page", 1))
        self.requested.append(page)
        return self.pages[page - 1], {}

    def close(self) -> None:
        self.closed = True


def test_espn_prop_moves_are_keyed_to_athlete_ids_and_carry_drift() -> None:
    stub = _StubEspn([{"pageCount": 1, "items": ESPN_ITEMS}])
    moves = fetch_espn_prop_moves(401872656, client=stub)

    # The team prop has no athlete ref and is dropped rather than mis-attributed.
    assert [m.athlete_id for m in moves] == [3912547, 4361307]

    passing = moves[0]
    assert passing.stat == PASSING_YARDS
    assert (passing.open_value, passing.current_value) == (229.5, 226.5)
    assert passing.drift == pytest.approx(-3.0)

    # Period-restricted props stay unmapped: they are real markets that score nothing.
    half = moves[1]
    assert half.stat is None
    assert half.drift is None  # no open target, so no movement to report


def test_espn_prop_moves_walk_every_page() -> None:
    # Distinct athletes per page: identical rows would (correctly) collapse into one and
    # the count would no longer say anything about whether the pager walked.
    pages = [
        {
            "pageCount": 3,
            "items": [{**ESPN_ITEMS[0], "athlete": {"$ref": f".../athletes/{n}?lang=en"}}],
        }
        for n in (11, 22, 33)
    ]
    stub = _StubEspn(pages)
    moves = fetch_espn_prop_moves(1, client=stub)
    assert [m.athlete_id for m in moves] == [11, 22, 33]
    assert stub.requested == [1, 2, 3]
    assert not stub.closed  # a caller-supplied client is not ours to close


def test_espn_prop_moves_raise_rather_than_return_nothing() -> None:
    with pytest.raises(PropsError, match="nothing usable"):
        fetch_espn_prop_moves(1, client=_StubEspn([{"pageCount": 1, "items": []}]))


def test_espn_pager_refuses_to_return_half_a_board() -> None:
    """Exhausting `max_pages` mid-walk used to return partial props and report success.

    `page_size=25` is ESPN's default and gives 29 pages on a live game against a
    `max_pages` of 20, so this is reachable without anyone doing anything strange. A
    truncated board is indistinguishable from a board where the rest was never posted.
    """
    pages = [
        {"pageCount": 29, "items": [{**ESPN_ITEMS[0], "athlete": {"$ref": f".../athletes/{n}"}}]}
        for n in range(1, 30)
    ]
    with pytest.raises(PropsError, match="stopped after 20 pages"):
        fetch_espn_prop_moves(1, client=_StubEspn(pages), max_pages=20)

    # Enough budget, same feed: every page is walked and nothing is dropped.
    moves = fetch_espn_prop_moves(1, client=_StubEspn(pages), max_pages=29)
    assert len(moves) == 29


def test_espn_repeats_itself_and_the_repeats_are_dropped() -> None:
    """The feed, not the pager, duplicates rows: 705 items, 465 distinct props, live.

    165 of 465 props came back two to five times, byte-identical down to `lastUpdated`.
    Anything averaging drift over the result would double-weight a third of the board.
    """
    passing = ESPN_ITEMS[0]
    milestone_a = {
        "athlete": {"$ref": ".../athletes/4361307?lang=en"},
        "type": {"id": "194", "name": "Receiving Yards Milestones"},
        "current": {"target": {"value": 50.5}},
        "open": {"target": {"value": 50.5}},
    }
    # Same athlete, same type id, DIFFERENT threshold -- a real second row, not a repeat.
    milestone_b = {**milestone_a, "current": {"target": {"value": 75.5}}}

    stub = _StubEspn(
        [
            {"pageCount": 2, "items": [passing, passing, milestone_a]},
            {"pageCount": 2, "items": [passing, milestone_a, milestone_b]},
        ]
    )
    moves = fetch_espn_prop_moves(1, client=stub)

    assert len(moves) == 3, "exact repeats collapse"
    assert sum(m.espn_type_id == "8" for m in moves) == 1
    # Milestones share (athlete, type) legitimately, so they must both survive.
    milestones = [m for m in moves if m.espn_type_id == "194"]
    assert {m.current_value for m in milestones} == {50.5, 75.5}


# --------------------------------------------------------------------------------------
# Components -> fantasy points
# --------------------------------------------------------------------------------------


def half_ppr(stats):
    """Stand-in for a league scorer; the real one comes from espn/scoring.py."""
    return (
        stats.get(RECEIVING_YARDS, 0.0) * 0.1
        + stats.get(RECEPTIONS, 0.0) * 0.5
        + stats.get(RUSHING_TDS, 0.0) * 6.0
        + stats.get("receiving_tds", 0.0) * 6.0
    )


def test_project_fantasy_points_takes_an_injected_scorer() -> None:
    components = {RECEIVING_YARDS: 42.1, RECEPTIONS: 3.3}
    assert project_fantasy_points(components, half_ppr) == pytest.approx(5.86)


def test_project_fantasy_points_expands_anytime_tds_when_asked() -> None:
    components = {RECEIVING_YARDS: 42.1, ANYTIME_TDS: 0.5}
    # Without a share the scorer never sees the TDs; with one they split rush/rec.
    assert project_fantasy_points(components, half_ppr) == pytest.approx(4.21)
    assert project_fantasy_points(components, half_ppr, rush_share=0.0) == pytest.approx(7.21)
    assert project_fantasy_points(components, half_ppr, rush_share=1.0) == pytest.approx(7.21)
    # The caller's dict is not mutated.
    assert ANYTIME_TDS in components


def test_the_rush_receive_split_goes_the_way_round_it_claims_to() -> None:
    """`half_ppr` prices rush and receiving TDs identically, so it cannot see a swap.

    A scorer that distinguishes them can, and that is the only assertion that pins the
    direction of `split_anytime_tds`: share 1.0 means ALL rushing.
    """

    def rush_only(stats):
        return stats.get(RUSHING_TDS, 0.0) * 10.0

    components = {ANYTIME_TDS: 0.5}
    assert project_fantasy_points(components, rush_only, rush_share=1.0) == pytest.approx(5.0)
    assert project_fantasy_points(components, rush_only, rush_share=0.0) == pytest.approx(0.0)
    assert project_fantasy_points(components, rush_only, rush_share=0.25) == pytest.approx(1.25)


def test_scoring_a_median_instead_of_a_mean_is_the_error_this_module_prevents() -> None:
    """Same player, same scorer: the posted line prices ~1.2 fewer points than the mean."""
    fit = fit_ladder_with_line(KUPP_RUNGS, devig_two_sided(29.5, -114, -114))
    from_median = project_fantasy_points({RECEIVING_YARDS: fit.median}, half_ppr)
    from_mean = project_fantasy_points({RECEIVING_YARDS: fit.mean}, half_ppr)
    assert from_mean - from_median > 1.2


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def _event_props() -> FanDuelEventProps:
    return FanDuelEventProps(
        event_id=1,
        lines=(
            PropLine("fanduel", 1, "Cooper Kupp", RECEIVING_YARDS, 29.5, -114, -114),
            PropLine("fanduel", 1, "Hunter Henry", RECEPTIONS, 3.5, -110, -110),
        ),
        ladders=(PropLadder("fanduel", 1, "Cooper Kupp", RECEIVING_YARDS, tuple(KUPP_RUNGS)),),
        touchdowns=(AnytimeTd("fanduel", 1, "Cooper Kupp", 250),),
        tabs_seen=("receiving-props",),
    )


def test_project_event_prefers_ladders_and_flags_bare_lines() -> None:
    projections = {(p.player, p.stat): p for p in project_event(_event_props())}

    kupp = projections[("Cooper Kupp", RECEIVING_YARDS)]
    assert kupp.source == "ladder"
    assert kupp.posted_line == 29.5
    assert kupp.mean > kupp.median
    assert kupp.skew_premium > 1.35
    assert kupp.fit is not None and kupp.sd is not None

    # No ladder for this one, so the posted line is all we have -- and it is flagged as
    # biased low rather than silently mixed in with the fitted numbers.
    henry = projections[("Hunter Henry", RECEPTIONS)]
    assert henry.source == "line"
    assert henry.mean == henry.median == 3.5
    assert henry.fit is None

    td = projections[("Cooper Kupp", ANYTIME_TDS)]
    assert td.source == "anytime_td"
    assert td.mean == pytest.approx(expected_touchdowns(250))


def test_components_by_player_collapses_to_one_mean_per_stat() -> None:
    components = components_by_player(project_event(_event_props()))
    assert set(components) == {"Cooper Kupp", "Hunter Henry"}
    assert set(components["Cooper Kupp"]) == {RECEIVING_YARDS, ANYTIME_TDS}
    assert components["Cooper Kupp"][RECEIVING_YARDS] > 40.0
    assert components["Hunter Henry"][RECEPTIONS] == 3.5

    points = project_fantasy_points(components["Cooper Kupp"], half_ppr, rush_share=0.0)
    assert points > 4.0


def test_collect_props_survives_every_source_dying(monkeypatch) -> None:
    """The headline claim of the module: one dead book never takes it down.

    Four undocumented endpoints, each of which can close without notice, so the failure
    path is the load-bearing one and it deserves an offline test rather than a note
    saying it is hard to stub.
    """
    import fantasy_quant.data.props as mod

    def dead(*_a, **_k):
        raise PropsError("book is gone")

    monkeypatch.setattr(mod, "FanDuelProps", dead)
    monkeypatch.setattr(mod, "fetch_underdog", dead)
    monkeypatch.setattr(mod, "fetch_pinnacle_player_props", dead)
    monkeypatch.setattr(mod, "fetch_espn_prop_moves", dead)

    bundle = collect_props(espn_game_ids=(7,))
    assert bundle.sources_live == []
    assert set(bundle.errors) == {"fanduel", "underdog", "pinnacle", "espn:7"}
    assert all("book is gone" in msg for msg in bundle.errors.values())


def test_collect_props_keeps_the_survivors_when_two_of_four_die(monkeypatch) -> None:
    import fantasy_quant.data.props as mod

    def dead(*_a, **_k):
        raise PropsError("book is gone")

    class _FakeFanDuel:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def events(self):
            return [mod.FanDuelEvent(1, "A @ B", "2026-09-10T00:15:00Z")]

        def event_props(self, event_id, tabs=()):
            return _event_props()

    monkeypatch.setattr(mod, "FanDuelProps", _FakeFanDuel)
    monkeypatch.setattr(mod, "fetch_underdog", lambda **_k: UD_PAYLOAD)
    monkeypatch.setattr(mod, "fetch_pinnacle_player_props", dead)
    monkeypatch.setattr(mod, "fetch_espn_prop_moves", dead)

    bundle = collect_props(espn_game_ids=(7,))
    assert bundle.sources_live == ["fanduel", "underdog"]
    assert set(bundle.errors) == {"pinnacle", "espn:7"}
    assert [line.player for line in bundle.underdog] == ["A.J. Brown"]
    assert len(bundle.fanduel) == 1
    assert bundle.fanduel[0].ladders


def test_collect_props_isolates_one_bad_event_from_the_rest(monkeypatch) -> None:
    """A game whose board has not opened lands in `errors` keyed by event, not globally."""
    import fantasy_quant.data.props as mod

    class _FanDuelWithOneDeadGame:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def events(self):
            return [
                mod.FanDuelEvent(1, "A @ B", "2026-09-10T00:15:00Z"),
                mod.FanDuelEvent(2, "C @ D", "2026-12-26T01:15:00Z"),
            ]

        def event_props(self, event_id, tabs=()):
            if event_id == 2:
                raise PropsError("no tab served")
            return _event_props()

    monkeypatch.setattr(mod, "FanDuelProps", _FanDuelWithOneDeadGame)
    monkeypatch.setattr(mod, "fetch_underdog", lambda **_k: UD_PAYLOAD)
    monkeypatch.setattr(mod, "fetch_pinnacle_player_props", lambda: [])

    bundle = collect_props()
    assert len(bundle.fanduel) == 1
    assert set(bundle.errors) == {"fanduel:2"}
    assert "fanduel" in bundle.sources_live


# --------------------------------------------------------------------------------------
# Live smoke tests -- skip, never fail, when a book is unreachable
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_live_fanduel_ladders() -> None:
    try:
        with FanDuelProps() as fd:
            event = fd.events()[0]
            props = fd.event_props(event.event_id)
    except PropsError as exc:
        pytest.skip(f"FanDuel unreachable: {exc}")

    assert props.ladders, "no ALT ladders -- the market shape has moved"
    projections = [p for p in project_event(props) if p.source == "ladder"]
    assert projections

    yards = [p for p in projections if p.stat == RECEIVING_YARDS and p.posted_line]
    assert yards, "no receiving-yards ladder paired with a two-sided line"
    for projection in yards:
        # The whole point: the fitted median tracks the posted line, the mean clears it.
        assert projection.median == pytest.approx(projection.posted_line, rel=0.10)
        assert projection.mean > projection.posted_line
        assert projection.fit is not None and projection.fit.rms_error < 0.06

    # The shading parameter must stay in the region a book actually charges per rung.
    # Against `continuity=0.0` this fails outright on receptions and passing TDs, where
    # the off-by-one drives the median estimate to ~1.21 and pins 1.30.
    for projection in projections:
        assert projection.fit is not None
        assert projection.fit.overround < 1.15, (
            f"{projection.player} {projection.stat}: fitted per-rung overround "
            f"{projection.fit.overround:.3f} is not a bookmaker's margin"
        )
        assert not projection.fit.overround_pinned


@pytest.mark.network
def test_live_fanduel_prices_a_rung_and_its_half_point_line_identically() -> None:
    """The empirical basis for the continuity correction, taken from the book itself.

    The ALT rung "k+" and the OVER at k-0.5 are the same event for an integer stat, and
    FanDuel quotes them at the same American price. If this ever stops being true the
    half-unit shift in `fit_lognormal_ladder` has lost its justification.
    """
    try:
        with FanDuelProps() as fd:
            props = fd.event_props(fd.events()[0].event_id)
    except PropsError as exc:
        pytest.skip(f"FanDuel unreachable: {exc}")

    lines = {(line.player, line.stat): line for line in props.lines}
    matched = 0
    for ladder in props.ladders:
        line = lines.get((ladder.player, ladder.stat))
        if line is None:
            continue
        rung = next((r for r in ladder.rungs if r.threshold == math.ceil(line.line)), None)
        if rung is None:
            continue  # the ladder simply does not carry a rung at that threshold
        matched += 1
        assert rung.implied == pytest.approx(american_to_implied(line.over_odds), abs=0.02), (
            f"{ladder.player} {ladder.stat}: rung {rung.threshold:g}+ at "
            f"{rung.american:g} vs OVER {line.line:g} at {line.over_odds:g}"
        )
    assert matched >= 3, "no ladder carried a rung at ceil(line) to compare"


@pytest.mark.network
def test_live_fanduel_rejects_a_bad_slug_with_http_200() -> None:
    try:
        with FanDuelProps() as fd:
            event_id = fd.events()[0].event_id
            with pytest.raises(PropsError):
                fd.markets(event_id, "receving-props", required=("PLAYER_X_",))
    except PropsError as exc:
        pytest.skip(f"FanDuel unreachable: {exc}")


@pytest.mark.network
def test_live_underdog_fantasy_points() -> None:
    try:
        payload = fetch_underdog()
    except PropsError as exc:
        pytest.skip(f"Underdog unreachable: {exc}")

    lines = parse_underdog(payload)
    if not lines:
        pytest.skip("Underdog has no NFL fantasy-points lines posted right now")
    assert all(line.stat == "fantasy_points" for line in lines)
    assert all(0.0 < line.line < 60.0 for line in lines)
    # If sport filtering ever regresses, baseball positions show up here.
    assert not {line.position for line in lines} - {"QB", "RB", "WR", "TE", "K", "FLEX"}


@pytest.mark.network
def test_live_pinnacle_player_props() -> None:
    try:
        props = fetch_pinnacle_player_props()
    except PropsError as exc:
        pytest.skip(f"Pinnacle unreachable: {exc}")

    assert props
    for prop in props[:25]:
        fair = prop.devigged()
        assert 1.0 < fair.overround < 1.30
        assert fair.p_over + fair.p_under == pytest.approx(1.0)
    assert any(prop.max_risk_stake for prop in props), "maxRiskStake is the sharpness signal"


@pytest.mark.network
def test_live_espn_prop_bets_expose_line_movement() -> None:
    # nflverse games.csv `espn` column, 2026 week 1 NE @ SEA.
    try:
        moves = fetch_espn_prop_moves(401872656)
    except PropsError as exc:
        pytest.skip(f"ESPN propBets unreachable: {exc}")

    assert moves
    assert all(move.athlete_id > 0 for move in moves)
    mapped = [m for m in moves if m.stat]
    assert mapped, "no prop type names mapped -- ESPN has renamed them"
    assert any(m.drift for m in mapped), "open and current are identical; movement is the point"
    # The feed ships exact duplicates; nothing that survives the fetch may be one.
    assert len(moves) == len(set(moves))
