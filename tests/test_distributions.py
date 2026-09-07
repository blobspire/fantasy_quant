"""Acceptance tests for the correlated weekly sampler.

The point of these is not that the code runs. It is that the tensor coming out of it
has the four properties the rest of the system assumes: exact marginals, the measured
correlations, absorbing injuries, and paired draws. Each of those is checked against a
number that was measured rather than chosen, and the tolerances are Monte Carlo
standard errors rather than round numbers.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
from scipy.special import gammainc, gammaincinv

from fantasy_quant.core import QB, RB, TE, WR, PlayerOutlook, WeeklyOutlook
from fantasy_quant.projections import calibration as cal
from fantasy_quant.sim import distributions as D

# Every test in this file that samples uses one of these, so a failure points at the
# sampler rather than at a fixture that happened to be pathological.
CALIBRATION = cal.default_calibration()


def outlook(
    player_id: int,
    position_id: int,
    mean: float,
    *,
    week: int = 1,
    pro_team_id: int = 1,
    playing: bool = True,
) -> WeeklyOutlook:
    return CALIBRATION.outlook_from_mean(
        player_id=player_id,
        season=2026,
        week=week,
        position_id=position_id,
        mean=mean,
        pro_team_id=pro_team_id,
        playing=playing,
    )


def season(
    player_id: int, position_id: int, mean: float, *, pro_team_id: int, weeks: range
) -> list[WeeklyOutlook]:
    return [outlook(player_id, position_id, mean, week=w, pro_team_id=pro_team_id) for w in weeks]


# --------------------------------------------------------------------------------------
# The quantile transform
# --------------------------------------------------------------------------------------


def test_hurdle_quantile_is_the_inverse_cdf_of_the_stated_mixture():
    """A stratified sweep of u must integrate back to the outlook's own moments.

    Deterministic on purpose: this isolates the mixture algebra from sampling noise, so
    a failure here is an algebra bug and never a lucky seed.
    """
    u = (np.arange(200_000) + 0.5) / 200_000
    for position_id, mean in ((QB, 18.0), (RB, 11.0), (WR, 9.0), (TE, 5.0), (WR, 0.6)):
        o = outlook(1, position_id, mean)
        x = D.hurdle_gamma_quantile(
            u, np.float64(o.p_zero), np.float64(o.shape), np.float64(o.scale)
        )
        assert x.mean() == pytest.approx(o.mean, rel=2e-3, abs=2e-3)
        assert x.std() == pytest.approx(o.sd, rel=5e-3, abs=5e-3)
        assert (x <= 0).mean() == pytest.approx(o.p_zero, abs=1e-4)


def test_quantile_table_matches_scipy_to_four_decimals():
    """The interpolated table is the hot path; it has to be indistinguishable."""
    rng = np.random.default_rng(0)
    shape = np.exp(rng.uniform(np.log(0.2), np.log(60.0), 200_000))
    u = rng.random(200_000)
    table = D.GammaQuantileTable.build(0.2, 60.0)
    approx = D.hurdle_gamma_quantile(u, np.zeros_like(u), shape, np.ones_like(u), table=table)
    exact = D.hurdle_gamma_quantile(u, np.zeros_like(u), shape, np.ones_like(u))
    rel = np.abs(approx - exact) / np.maximum(exact, 1e-9)
    assert rel.max() < 2e-3
    assert np.quantile(rel, 0.999) < 1e-3


def test_quantile_table_clamps_instead_of_extrapolating():
    """Off the end of the grid a bilinear read diverges; it must saturate instead."""
    table = D.GammaQuantileTable.build(1.0, 2.0)
    far = table(np.array([1e6, 1e-8]), np.array([0.0, 0.0]))
    assert np.all(np.isfinite(far))
    assert np.all(far > 0)


def test_log_quantile_is_right_where_the_quantile_itself_underflows():
    """`gammaincinv` returns 0.0 long before the true quantile is zero.

    At a shape of 1e-5 the median of Gamma(a, 1) is `exp(-69315)`; scipy hands back 0.0
    and `log(max(x, 1e-300))` records that as `exp(-691)`. This does not move any moment
    the sampler produces today -- both are zero to anything that sums them -- and the
    inflation that DID move them is pinned by the next test, on the shape floor. What
    this protects is that the stored surface is the real one, so the table stays right if
    someone moves `z_max` or reads the low corner of the grid. The small-x branch is
    checked the only way it can be: push its answer back through the FORWARD incomplete
    gamma and require the probability to come out.
    """
    a = np.array([1e-8, 1e-6, 1e-5, 1e-3])[:, None]
    q = np.array([1e-6, 0.01, 0.5, 0.9, 0.999])[None, :]
    log_x = D.log_gamma_quantile(a, q)
    representable = log_x > -700.0
    back = gammainc(np.broadcast_to(a, log_x.shape), np.exp(np.minimum(log_x, 700.0)))
    assert np.allclose(
        back[representable], np.broadcast_to(q, log_x.shape)[representable], rtol=1e-9
    )
    # The unrepresentable ones are not clipped to log(1e-300); they carry the real value.
    assert log_x.min() < -1e6
    # And on ordinary shapes it is scipy, to eight digits.
    u = (np.arange(50_000) + 0.5) / 50_000
    for shape in (0.005, 0.2, 1.0, 60.0):
        exact = gammaincinv(shape, u)
        mine = np.exp(D.log_gamma_quantile(np.float64(shape), u))
        assert np.abs(mine - exact).max() / exact.max() < 1e-8


def test_a_near_zero_projection_is_not_inflated_into_a_real_player():
    """A sub-0.01-point outlook has a gamma shape of 1e-5, and it must not be clamped.

    `shape_index` clamps rather than extrapolates, so a shape floor ABOVE what the panel
    contains does not blur a player's draw -- it replaces his distribution. Against the
    1e-4 floor this module used to carry, a WR calibrated at mu = 0.004 (shape 1.4e-5)
    sampled at a mean of 0.051 and an SD of 10.1 against a stated 0.004 and 2.71: two
    orders of magnitude of invented points and 3.7x the modelled width, silently, on
    exactly the free-agent pool a waiver search walks. Nothing warned. His SD cannot be
    recovered -- it lives in a one-in-a-million draw of several thousand points -- but it
    must not be EXCEEDED, because a variance-seeking lineup would then reach for him.
    """
    scrub = CALIBRATION.outlook_from_mean(
        player_id=2, season=2026, week=1, position_id=WR, mean=0.004, pro_team_id=2
    )
    assert scrub.shape < 1e-4  # the regime under test really is reached

    # Stratified rather than sampled. This marginal's variance rides on one draw in a
    # million of ~1,800 points, so ANY Monte Carlo estimate of its SD is itself wildly
    # variable -- 2.7 on one seed and 6.5 on the next -- and an assertion on a sampled SD
    # would be an assertion about the seed. A quantile sweep has no such noise.
    table = D.GammaQuantileTable.build(scrub.shape, 3.0)
    u = (np.arange(1_000_000) + 0.5) / 1_000_000
    args = (u, np.float64(scrub.p_zero), np.float64(scrub.shape), np.float64(scrub.scale))
    x = D.hurdle_gamma_quantile(*args, table=table)
    assert x.mean() < 0.01  # the old floor gave 0.0284 against a stated 0.004
    assert x.std() < scrub.sd  # the old floor gave 6.83 against a stated 2.71
    # Approaching from below and still climbing, which is the honest shape of the miss.
    assert (
        x.mean()
        < D.hurdle_gamma_quantile(
            (np.arange(4_000_000) + 0.5) / 4_000_000,
            *args[1:],
            table=table,
        ).mean()
        < scrub.mean * 1.05
    )

    # The table is sold as an approximation of the exact path; in this regime the exact
    # path underflows, and the two used to give completely different distributions.
    exact = D.hurdle_gamma_quantile(*args)
    assert x.mean() == pytest.approx(exact.mean(), rel=1e-3)
    assert x.std() == pytest.approx(exact.std(), rel=1e-3)

    # And an ordinary team-mate in the same panel is untouched by any of it.
    normal = outlook(1, WR, 12.0, pro_team_id=2)
    panel = D.SimPanel.from_outlooks([normal, scrub])
    drawn = (
        D.WeeklySampler(panel, seed=5, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(200_000)
        .points[:, 0, 0]
    )
    assert drawn.mean() == pytest.approx(normal.mean, rel=0.01)
    assert drawn.std() == pytest.approx(normal.sd, rel=0.01)


def test_an_unresolvable_width_is_warned_about_rather_than_quietly_wrong(caplog):
    scrub = CALIBRATION.outlook_from_mean(
        player_id=1, season=2026, week=1, position_id=WR, mean=0.004, pro_team_id=2
    )
    panel = D.SimPanel.from_outlooks([scrub, outlook(2, WR, 12.0, pro_team_id=2)])
    with caplog.at_level(logging.WARNING, logger="fantasy_quant.sim.distributions"):
        D.WeeklySampler(panel, seed=1)
    assert any("gamma shape under" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------------------
# Marginals
# --------------------------------------------------------------------------------------


def _standard_errors(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(se of the mean, se of the sd) estimated from the sample itself.

    The sd of a hurdle gamma is NOT `sd/sqrt(2n)` -- at a shape of 0.005 the excess
    kurtosis runs past 1,000 and the normal-theory error understates the true one
    twenty-fold. Using the delta-method error with the sample's own fourth moment is
    the difference between a test that passes because the sampler is right and one that
    fails because the estimator is slow.
    """
    n = x.shape[0]
    sd = x.std(axis=0)
    centred = x - x.mean(axis=0)
    m4 = (centred**4).mean(axis=0)
    var_s = np.maximum(m4 - sd**4, 0.0) / (4.0 * n * np.maximum(sd**2, 1e-12))
    return sd / np.sqrt(n), np.sqrt(var_s)


def test_sampled_marginals_recover_every_outlook():
    """Mean, sd and P(X<=0) of the tensor must be the outlook's own stated values."""
    rng = np.random.default_rng(4)
    positions = rng.choice([QB, RB, WR, TE], 120, p=[0.12, 0.28, 0.42, 0.18])
    means = rng.gamma(2.0, 4.5, 120)
    outlooks = [
        outlook(i + 1, int(positions[i]), float(means[i]), pro_team_id=1 + i % 32)
        for i in range(120)
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    n = 40_000
    draw = D.WeeklySampler(panel, seed=17, injuries=D.InjuryModel.off(), dtype=np.float64).draw(n)
    x = draw.points[:, 0, :]

    se_mean, se_sd = _standard_errors(x)
    z_mean = (x.mean(axis=0) - panel.mean[0]) / se_mean
    z_sd = (x.std(axis=0) - panel.sd[0]) / se_sd
    p_zero = (x <= 0).mean(axis=0)
    se_p = np.sqrt(panel.p_zero[0] * (1 - panel.p_zero[0]) / n)
    z_p = (p_zero - panel.p_zero[0]) / np.maximum(se_p, 1e-12)

    assert np.abs(z_mean).max() < 4.5
    assert np.abs(z_sd).max() < 4.5
    assert np.abs(z_p).max() < 4.5


def test_injuries_do_not_touch_the_available_weeks_marginal():
    """Conditioning on availability must leave the outlook's own distribution intact."""
    outlooks = season(1, RB, 12.0, pro_team_id=9, weeks=range(1, 19))
    panel = D.SimPanel.from_outlooks(outlooks)
    draw = D.WeeklySampler(panel, seed=2, dtype=np.float64).draw(4_000)
    played = draw.points[:, :, 0][draw.available[:, :, 0]]
    assert played.mean() == pytest.approx(panel.mean[0, 0], rel=0.02)
    assert (played <= 0).mean() == pytest.approx(panel.p_zero[0, 0], abs=0.01)


# --------------------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------------------

# One NFL offense: a QB, two backs, two receivers, two tight ends. Small enough that
# the measured rhos are jointly consistent, which is the case the constants describe.
OFFENSE = (
    (1, QB, 18.0),
    (2, RB, 11.0),
    (3, RB, 7.0),
    (4, WR, 13.0),
    (5, WR, 9.0),
    (6, TE, 8.0),
    (7, TE, 4.0),
)


# The same offense projected as a bottom-of-the-roster one. Blank rates run to 69%
# here, which is where a first-order attenuation correction falls apart -- and it is
# most of the free-agent pool a waiver search evaluates.
SCRUB_OFFENSE = (
    (1, QB, 3.0),
    (2, RB, 1.2),
    (3, RB, 0.6),
    (4, WR, 0.8),
    (5, WR, 0.5),
    (6, TE, 0.7),
    (7, TE, 0.4),
)


def _offense_draw(n: int, offense=OFFENSE, **kwargs) -> np.ndarray:
    outlooks = [outlook(p, pos, mu, pro_team_id=12) for p, pos, mu in offense]
    panel = D.SimPanel.from_outlooks(outlooks)
    sampler = D.WeeklySampler(
        panel, seed=11, injuries=D.InjuryModel.off(), dtype=np.float64, **kwargs
    )
    return sampler.draw(n).points[:, 0, :]


def _correlation_errors(offense, n: int, **kwargs) -> dict[tuple[int, int], float]:
    x = _offense_draw(n, offense=offense, **kwargs)
    corr = np.corrcoef(x.T)
    model = D.CorrelationModel()
    return {
        (i, j): corr[i, j] - model.rho(pos_i, pos_j)
        for i, (_, pos_i, _) in enumerate(offense)
        for j, (_, pos_j, _) in enumerate(offense[i + 1 :], start=i + 1)
    }


def test_sampled_correlation_recovers_the_measured_rhos():
    """Pearson correlation of the POINTS, which is what the constants were measured on.

    A Gaussian copula does not pass a correlation through a hurdle gamma unchanged, so
    this is really a test of the Hermite inversion. Every measured pair comes back
    within about three Monte Carlo standard errors, and every unmeasured pair -- WR/WR
    on the same team included -- comes back at zero.
    """
    n = 400_000
    errors = _correlation_errors(OFFENSE, n)
    model = D.CorrelationModel()
    se = 1.0 / np.sqrt(n)
    checked = 0
    for (i, j), error in errors.items():
        target = model.rho(OFFENSE[i][1], OFFENSE[j][1])
        if target == 0.0:
            assert abs(error) < 5 * se
            continue
        checked += 1
        assert abs(error) < 0.008
    assert checked == 8


def test_correlation_survives_the_free_agent_end_of_the_pool():
    """Blank rates of 40-69%, where the textbook first-order correction fails outright.

    Keeping one Hermite term here asks for a latent 0.707 and realizes 0.518 against a
    target of 0.300. Four terms hold it to a few thousandths, which is why the module
    carries the polynomial solve rather than a division.
    """
    n = 400_000
    errors = _correlation_errors(SCRUB_OFFENSE, n)
    assert max(abs(e) for e in errors.values()) < 0.008


def test_raw_copula_reading_lands_low_which_is_why_the_correction_exists():
    """Documented, not merely believed: uncorrected, rho(QB,WR) comes out near 0.29."""
    x = _offense_draw(300_000, correlation=D.CorrelationModel(match_pearson=False))
    qb_wr = np.corrcoef(x[:, 0], x[:, 3])[0, 1]
    assert 0.283 < qb_wr < 0.297


def test_four_hermite_terms_are_where_the_series_stops_moving():
    """Convergence check on the truncation, over blank rates from 0.6% to 69%.

    Parseval mass is the wrong diagnostic here -- a hurdle gamma's transform is close to
    a step function and its spectrum has a long tail -- but those terms enter the
    inversion as `rho^k`, and at the largest measured rho of 0.30 the fifth power is
    already 0.002. Doubling the order must therefore not move the solved latent
    correlation, and it does not.
    """
    p_zero = np.array([0.006, 0.20, 0.40, 0.69])
    shape = np.array([5.3, 14.7, 2.0, 0.9])
    target = np.full((4, 4), 0.30)

    def solved(order: int) -> np.ndarray:
        g = D.hermite_coefficients(p_zero, shape, order=order)
        return D.solve_latent_correlation(target, g[:, :, None] * g[:, None, :])

    assert np.abs(solved(D.HERMITE_ORDER) - solved(8)).max() < 0.005
    # One term is not enough, which is the whole reason for the polynomial.
    assert np.abs(solved(1) - solved(8)).max() > 0.1

    spectrum = D.hermite_coefficients(p_zero, shape, order=8)
    assert np.all((spectrum**2).sum(axis=0) <= 1.0 + 1e-6)
    assert np.all(np.abs(spectrum[0]) == np.abs(spectrum).max(axis=0))


def test_latent_solver_inverts_its_own_polynomial():
    spectrum = np.array([[0.9, 0.5, 0.7], [0.2, 0.3, 0.1], [0.05, 0.1, 0.02], [0.01, 0.02, 0.0]])
    target = np.array([0.30, 0.20, -0.09])
    rho = D.solve_latent_correlation(target, spectrum)
    powers = np.stack([rho ** (k + 1) for k in range(spectrum.shape[0])])
    assert (spectrum * powers).sum(axis=0) == pytest.approx(target, abs=1e-6)


def test_latent_solver_clamps_an_unreachable_target():
    """Two near-certain blanks cannot correlate 0.30 however you set the copula."""
    spectrum = np.array([[0.2], [0.02], [0.0], [0.0]])
    rho = D.solve_latent_correlation(np.array([0.30]), spectrum, max_latent=0.99)
    assert rho[0] == pytest.approx(0.99, abs=1e-6)


def test_correlation_does_not_depend_on_where_a_player_sits_in_the_panel():
    """Every other fixture here lists a team's positions in ascending order. Real ones do not.

    A panel is sorted by ESPN player id, which has nothing to do with position, so the
    common case is a block whose positions run TE, WR, RB, QB. `SAME_TEAM_RHO` is keyed
    by the SORTED position pair, and a lookup that forgot to sort would find nothing for
    (WR, QB) -- so every same-team correlation would silently drop to zero and the block
    would not even be built. Same offense, both orders, same answer.
    """
    n = 200_000
    corrs = []
    for ids in ([1, 2, 3, 4], [4, 3, 2, 1]):
        outlooks = [
            outlook(ids[0], TE, 8.0, pro_team_id=12),
            outlook(ids[1], WR, 13.0, pro_team_id=12),
            outlook(ids[2], RB, 11.0, pro_team_id=12),
            outlook(ids[3], QB, 18.0, pro_team_id=12),
        ]
        panel = D.SimPanel.from_outlooks(outlooks)
        draw = D.WeeklySampler(panel, seed=13, injuries=D.InjuryModel.off(), dtype=np.float64).draw(
            n
        )
        x = draw.points_for([ids[3], ids[1], ids[2], ids[0]])[:, 0, :]  # QB, WR, RB, TE
        corrs.append(np.corrcoef(x.T))
    for corr in corrs:
        assert corr[0, 1] == pytest.approx(0.30, abs=0.01)  # QB/WR
        assert corr[0, 2] == pytest.approx(0.08, abs=0.01)  # QB/RB
        assert corr[0, 3] == pytest.approx(0.20, abs=0.01)  # QB/TE
    assert np.abs(corrs[0] - corrs[1]).max() < 0.01


def test_different_team_pairs_are_uncorrelated():
    """+0.003 measured across teams; the model treats it as exactly zero."""
    outlooks = [
        outlook(1, QB, 18.0, pro_team_id=12),
        outlook(2, WR, 13.0, pro_team_id=12),
        outlook(3, WR, 13.0, pro_team_id=21),
        outlook(4, TE, 8.0, pro_team_id=21),
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    n = 200_000
    x = (
        D.WeeklySampler(panel, seed=8, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(n)
        .points[:, 0, :]
    )
    corr = np.corrcoef(x.T)
    se = 1.0 / np.sqrt(n)
    assert corr[0, 1] == pytest.approx(0.30, abs=0.01)  # same team
    assert abs(corr[0, 2]) < 4 * se  # QB vs the other team's WR
    assert abs(corr[1, 3]) < 4 * se
    assert abs(corr[2, 0]) < 4 * se


def test_the_hurdle_shares_its_uniform_with_the_magnitude():
    """The design decision the module is built around, checked behaviourally.

    A QB who blanks must drag his receiver down with him. With independent Bernoulli
    hurdles P(WR blanks | QB blanks) would equal P(WR blanks); driving both from one
    uniform makes it five times larger, which is what a real shut-out offense looks
    like.
    """
    outlooks = [outlook(1, QB, 18.0, pro_team_id=12), outlook(2, WR, 13.0, pro_team_id=12)]
    panel = D.SimPanel.from_outlooks(outlooks)
    x = (
        D.WeeklySampler(panel, seed=5, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(200_000)
        .points[:, 0, :]
    )
    qb_blank = x[:, 0] <= 0
    marginal = (x[:, 1] <= 0).mean()
    conditional = (x[qb_blank, 1] <= 0).mean()
    assert conditional > 4 * marginal
    # ... and the magnitude moves with it, in both directions.
    top_decile = x[:, 0] >= np.quantile(x[:, 0], 0.9)
    assert x[qb_blank, 1].mean() < 0.6 * x[:, 1].mean()
    assert x[top_decile, 1].mean() > 1.3 * x[:, 1].mean()


#: A whole NFL depth chart, starters and fringe together, which is what a candidate
#: search puts in the panel the moment it considers the free-agent pool.
DEPTH_CHART = (
    (QB, 20.0),
    (QB, 0.4),
    (RB, 13.0),
    (RB, 7.0),
    (RB, 2.0),
    (RB, 0.8),
    (WR, 15.0),
    (WR, 11.0),
    (WR, 6.0),
    (WR, 2.0),
    (WR, 0.9),
    (WR, 0.4),
    (TE, 8.0),
    (TE, 1.5),
    (TE, 0.4),
)


def test_a_full_depth_chart_keeps_the_starters_correlations():
    """The case the two-stage repair exists for, and the one that matters in practice.

    Fifteen players on one team cannot all hold their measured rhos at once. Everything
    turns on which ones give way. Relaxing the transform correction first and weighting
    the projection by variance holds QB-WR1 at 0.285 against a target of 0.300 and lets
    the sub-one-point fringe absorb the rest; repairing naively instead gives 0.153,
    which would quietly halve the value of a QB-WR stack.
    """
    outlooks = [
        outlook(i + 1, position_id, mean, pro_team_id=12)
        for i, (position_id, mean) in enumerate(DEPTH_CHART)
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    x = (
        D.WeeklySampler(panel, seed=4, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(300_000)
        .points[:, 0, :]
    )
    corr = np.corrcoef(x.T)
    model = D.CorrelationModel()
    starters = [i for i, (_, mean) in enumerate(DEPTH_CHART) if mean >= 6.0]
    for i in starters:
        for j in starters:
            if j <= i:
                continue
            target = model.rho(DEPTH_CHART[i][0], DEPTH_CHART[j][0])
            if target == 0.0:
                continue
            # Within 16% of the measured value, and never with the sign reversed. The
            # bound is not 12%: the systematic loss here, computed noise-free from the
            # repaired latent through the Hermite spectrum, is 9.6% at worst (QB against
            # the second back), and one Monte Carlo standard error on that pair is
            # another 2.3% -- so a 12% bound was under one standard error from failing on
            # a reseed. It still separates this repair from the alternatives by a wide
            # margin: an unweighted projection of the unrelaxed matrix lands 49% low.
            assert corr[i, j] == pytest.approx(target, rel=0.16)
    # The headline pair carries its own bound, because a blanket tolerance loose enough
    # to survive a reseed is also loose enough to hide the thing this repair is for.
    # QB1-WR1 realizes 0.284 here; drop the variance weighting and it is 0.264, drop the
    # relaxation too and it is 0.153, against a Monte Carlo standard error of 0.002.
    qb1, wr1 = 0, 6
    assert (DEPTH_CHART[qb1][0], DEPTH_CHART[wr1][0]) == (QB, WR)
    assert corr[qb1, wr1] > 0.275
    # Marginals are untouched by any of the repair; that is the invariant to protect.
    assert x.mean(axis=0) == pytest.approx(panel.mean[0], abs=0.06)
    assert x.std(axis=0) == pytest.approx(panel.sd[0], abs=0.08)


def test_the_repair_warning_states_the_loss_in_points_not_in_latent_space(caplog):
    """The one honest disclosure of a known modelling loss, checked against the tensor.

    The repair operates on the LATENT matrix; the constants are Pearson correlations of
    POINTS. Differencing the repaired latent against the target -- which is what this
    warning used to do -- reported 0.053 on the depth chart below where the realized loss
    is 0.258, understating the module's own worst-case damage fivefold. So the number in
    the log is compared here against the correlation actually measured on the draw.
    """
    outlooks = [
        outlook(i + 1, position_id, mean, pro_team_id=12)
        for i, (position_id, mean) in enumerate(DEPTH_CHART)
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    with caplog.at_level(logging.WARNING, logger="fantasy_quant.sim.distributions"):
        sampler = D.WeeklySampler(panel, seed=4, injuries=D.InjuryModel.off(), dtype=np.float64)
    warnings = [r for r in caplog.records if "positive-definite" in r.msg]
    assert len(warnings) == 1
    logged_overall, logged_core = warnings[0].args[2], warnings[0].args[3]

    x = sampler.draw(200_000).points[:, 0, :]
    corr = np.corrcoef(x.T)
    model = D.CorrelationModel()
    target = model.target_matrix([p for p, _ in DEPTH_CHART])
    error = np.abs(corr - target)
    np.fill_diagonal(error, 0.0)
    sd = panel.sd[0]
    core = sd >= np.median(sd)

    assert logged_overall == pytest.approx(error.max(), abs=0.02)
    assert logged_core == pytest.approx(error[np.ix_(core, core)].max(), abs=0.02)
    # And it is a big number, not a rounding error: a warning that reads 0.05 invites
    # the reader to ignore it, and this block really does give up a quarter of a rho.
    assert logged_overall > 0.15


def test_relaxation_is_skipped_when_the_block_is_already_feasible():
    positions = [QB, RB, WR, TE]
    target = D.CorrelationModel().target_matrix(positions)
    relaxed, t = D.relax_to_feasible(target * 0.5 + np.eye(4) * 0.5, target)
    assert t == 0.0
    assert np.array_equal(relaxed, target * 0.5 + np.eye(4) * 0.5)


def test_relaxation_falls_all_the_way_back_when_even_the_targets_are_infeasible():
    positions = [QB, QB] + [WR] * 7
    target = D.CorrelationModel().target_matrix(positions)
    latent = np.clip(target * 2.5, -0.99, 0.99)
    np.fill_diagonal(latent, 1.0)
    relaxed, t = D.relax_to_feasible(latent, target)
    assert t == 1.0
    assert np.allclose(relaxed, target)


def test_non_positive_definite_block_is_repaired_not_crashed():
    """Two QBs and seven receivers is jointly inconsistent under the measured rhos.

    Both quarterbacks correlate 0.30 with all seven receivers and zero with each other,
    which no correlation matrix can do (min eigenvalue -0.21 here). A candidate search
    that adds a whole depth chart to the panel produces exactly this, so it must repair
    rather than raise.
    """
    positions = [QB, QB] + [WR] * 7
    target = D.CorrelationModel().target_matrix(positions)
    assert np.linalg.eigvalsh(target).min() < 0

    factor, adjusted = D.nearest_correlation_factor(target)
    assert adjusted
    repaired = factor @ factor.T
    assert np.allclose(np.diag(repaired), 1.0)
    assert np.linalg.eigvalsh(repaired).min() >= -1e-10
    # Close, but honestly not identical: QB-WR lands near 0.265 rather than 0.300.
    assert np.abs(repaired - target).max() < 0.08

    outlooks = [outlook(i + 1, pos, 12.0, pro_team_id=12) for i, pos in enumerate(positions)]
    panel = D.SimPanel.from_outlooks(outlooks)
    x = (
        D.WeeklySampler(panel, seed=1, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(40_000)
        .points[:, 0, :]
    )
    assert np.isfinite(x).all()
    assert np.corrcoef(x[:, 0], x[:, 2])[0, 1] > 0.2
    # The repair keeps the unit diagonal, so every marginal survives it intact. That is
    # the invariant that matters: a wide block may lose some correlation, never a mean.
    assert x.mean(axis=0) == pytest.approx(panel.mean[0], abs=0.12)
    assert x.std(axis=0) == pytest.approx(panel.sd[0], abs=0.12)


def test_positive_definite_block_uses_cholesky_unchanged():
    target = D.CorrelationModel().target_matrix([QB, RB, WR, TE])
    factor, adjusted = D.nearest_correlation_factor(target)
    assert not adjusted
    assert np.allclose(factor @ factor.T, target)


# --------------------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------------------


def test_absence_distribution_is_a_proper_pmf():
    model = D.InjuryModel()
    assert sum(model.absence_pmf) == pytest.approx(1.0, abs=1e-9)
    assert all(p >= 0 for p in model.absence_pmf)
    assert model.expected_absence == pytest.approx(4.27, abs=0.05)
    # The number to sanity-check against a real season: a starting back misses about
    # three games in seventeen.
    assert model.expected_games_missed(RB) == pytest.approx(3.1, abs=0.3)
    assert model.expected_games_missed(QB) < model.expected_games_missed(RB)


def test_absence_pmf_must_sum_to_one():
    with pytest.raises(ValueError):
        D.InjuryModel(absence_pmf=(0.5, 0.2))


def _availability(seed: int = 9, n: int = 4_000) -> np.ndarray:
    outlooks = [
        o
        for p in range(1, 121)
        for o in season(p, RB, 12.0, pro_team_id=1 + p % 32, weeks=range(1, 19))
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    return D.WeeklySampler(panel, seed=seed).draw(n).available


def test_injuries_are_absorbing_not_iid():
    """The property `ffsimulator` does not have.

    Under an IID per-week hazard, P(playing next week | out this week) would be
    1 - hazard, i.e. ~0.95. Absorbing durations put it near 0.24. That difference is
    the entire left tail of a season.
    """
    available = _availability()
    was_out = ~available[:, :-1, :]
    plays_next = available[:, 1:, :]
    recovery = plays_next[was_out].mean()
    assert recovery < 0.4
    iid_recovery = 1.0 - D.INJURY_HAZARD[RB]
    assert recovery < 0.5 * iid_recovery
    # Healthy players, by contrast, keep playing at very nearly 1 - hazard.
    still_healthy = plays_next[available[:, :-1, :]].mean()
    assert still_healthy == pytest.approx(iid_recovery, abs=0.01)


def test_availability_survival_curve_is_monotone():
    available = _availability()
    by_week = available.mean(axis=(0, 2))
    assert np.all(np.diff(by_week) <= 1e-9)
    assert by_week[0] > 0.93
    assert by_week[-1] < 0.85


def test_absences_come_in_runs():
    """Mean run length has to look like the duration model, not like single weeks."""
    available = _availability(n=1_000)
    out = ~available[:, :, 0]
    starts = out[:, 1:] & ~out[:, :-1]
    total_out = out[:, 1:].sum()
    assert total_out > 0
    mean_run = total_out / max(starts.sum(), 1)
    assert mean_run > 2.5


def _absence_chain(n_weeks: int, hazard: float, pmf: np.ndarray) -> dict[str, np.ndarray]:
    """The injury process solved exactly, as a Markov chain over games-still-to-miss.

    Written from the two constants and the stated rule -- one hazard draw per at-risk
    game, then a duration that runs down -- and NOT from the sampler. It pins the level
    of the process and not just its shape: "recovery is under 0.4" passes for any
    absorbing model at all, including one where the hazard keeps re-firing while a player
    is already out and doubles his time on the shelf.
    """
    n_states = len(pmf) + 1
    state = np.zeros(n_states)
    state[0] = 1.0
    available, joint_out_then_play, p_out = [], [], []
    for _ in range(n_weeks):
        available.append(state[0] * (1.0 - hazard))
        after = np.zeros(n_states)
        after[: len(pmf)] += state[0] * hazard * pmf  # fires: out now, ends at d-1
        after[:-1] += state[1:]  # already out: one game burnt
        p_out.append(1.0 - available[-1])
        joint_out_then_play.append(after[0] * (1.0 - hazard))
        state = after.copy()
        state[0] += available[-1]  # the healthy branch stays healthy
    return {
        "available": np.array(available),
        "recovery": np.array(joint_out_then_play[:-1]).sum() / np.array(p_out[:-1]).sum(),
        "games_missed": float(n_weeks - np.array(available).sum()),
    }


def test_the_absence_process_matches_an_independently_solved_markov_chain():
    """Level, not just shape. Every number here is solved for, not measured off a run."""
    chain = _absence_chain(18, D.INJURY_HAZARD[RB], np.array(D.ABSENCE_PMF))
    available = _availability()
    by_week = available.mean(axis=(0, 2))
    was_out = ~available[:, :-1, :]
    recovery = available[:, 1:, :][was_out].mean()

    assert recovery == pytest.approx(chain["recovery"], abs=0.01)
    assert np.abs(by_week - chain["available"]).max() < 0.005
    assert (~available).sum(axis=1).mean() == pytest.approx(chain["games_missed"], abs=0.05)
    # The steady state the helper advertises, against the chain it is meant to describe.
    # `1 + h*E[D]` in the denominator instead of `1 - h + h*E[D]` reads 3.09 here where
    # the process runs at 3.23; the run-length of the healthy spell is (1-h)/h, because
    # the week the hazard fires is the first week missed rather than the last played.
    long_run = _absence_chain(4_000, D.INJURY_HAZARD[RB], np.array(D.ABSENCE_PMF))
    assert D.InjuryModel().expected_games_missed(RB) == pytest.approx(
        long_run["games_missed"] / 4_000.0 * 17.0, abs=0.02
    )


def test_points_are_zero_wherever_the_player_was_not_available():
    """The invariant every consumer of the tensor relies on and none of them re-checks.

    `available` is the flag a lineup filters on, but a caller that sums `points` over a
    roster -- `Draw.totals` does exactly that -- gets the injured weeks for free unless
    they are actually zero in the tensor. Byes are covered elsewhere; this is the injury
    half, and it is the half that only appears once a hazard has fired.
    """
    outlooks = [
        o
        for p in range(1, 41)
        for o in season(p, RB, 12.0, pro_team_id=1 + p % 8, weeks=range(1, 19))
    ]
    panel = D.SimPanel.from_outlooks(outlooks, byes={3: 9})
    draw = D.WeeklySampler(panel, seed=21, dtype=np.float64).draw(1_000)
    unavailable = ~draw.available
    assert unavailable.any()
    assert np.all(draw.points[unavailable] == 0.0)
    # Distinguish an injury from a bye: there must be unavailable player-weeks that are
    # neither, or the assertion above would be about byes alone.
    injured = unavailable & panel.has_game[None, :, :]
    assert injured.mean() > 0.05


def test_the_default_float32_tensor_still_carries_the_marginals():
    """The precision-sensitive tests all ask for float64; the shipped default is not."""
    outlooks = [outlook(i + 1, WR, 14.0, pro_team_id=1 + i) for i in range(4)]
    panel = D.SimPanel.from_outlooks(outlooks)
    x = D.WeeklySampler(panel, seed=2, injuries=D.InjuryModel.off()).draw(100_000).points[:, 0, :]
    assert x.dtype == np.float32
    assert x.mean(axis=0) == pytest.approx(panel.mean[0], abs=0.06)
    assert x.std(axis=0) == pytest.approx(panel.sd[0], abs=0.06)


def test_injuries_can_be_turned_off_without_moving_anything_else():
    """Off is the same season with everybody healthy, not a different season."""
    outlooks = season(1, RB, 12.0, pro_team_id=9, weeks=range(1, 19))
    panel = D.SimPanel.from_outlooks(outlooks)
    healthy = D.WeeklySampler(panel, seed=3, injuries=D.InjuryModel.off(), dtype=np.float64)
    injured = D.WeeklySampler(panel, seed=3, dtype=np.float64)
    a, b = injured.draw(500), healthy.draw(500)
    assert b.available.all()
    assert not a.available.all()
    assert np.array_equal(a.points[a.available], b.points[a.available])


# --------------------------------------------------------------------------------------
# Byes and availability bookkeeping
# --------------------------------------------------------------------------------------


def test_a_player_on_bye_scores_zero_and_is_not_startable():
    outlooks = season(1, WR, 14.0, pro_team_id=12, weeks=range(1, 19))
    panel = D.SimPanel.from_outlooks(outlooks, byes={12: 7})
    draw = D.WeeklySampler(panel, seed=1).draw(400)
    bye = panel.week_index(7)
    assert not draw.available[:, bye, 0].any()
    assert np.all(draw.points[:, bye, 0] == 0)
    assert draw.available[:, panel.week_index(6), 0].mean() > 0.8


def test_a_bye_does_not_burn_a_game_off_an_injury_clock():
    """Durations are counted in games, so an absence that straddles a bye ends later."""
    weeks = range(1, 19)
    with_bye = D.SimPanel.from_outlooks(
        season(1, RB, 12.0, pro_team_id=12, weeks=weeks), byes={12: 9}
    )
    without = D.SimPanel.from_outlooks(season(1, RB, 12.0, pro_team_id=12, weeks=weeks))
    games_missed_with = (~D.WeeklySampler(with_bye, seed=6).draw(4_000).available).sum(axis=1)[:, 0]
    games_missed_without = (~D.WeeklySampler(without, seed=6).draw(4_000).available).sum(axis=1)[
        :, 0
    ]
    # The bye adds exactly one unavailable week and does not shorten the absences.
    assert games_missed_with.mean() == pytest.approx(games_missed_without.mean() + 1.0, abs=0.25)


def test_a_player_not_playing_is_compiled_as_no_game():
    o = outlook(1, WR, 14.0, week=3).zeroed()
    panel = D.SimPanel.from_outlooks([outlook(1, WR, 14.0, week=2), o])
    assert panel.has_game[panel.week_index(2), 0]
    assert not panel.has_game[panel.week_index(3), 0]


def test_a_week_with_no_outlook_is_a_week_with_no_game():
    panel = D.SimPanel.from_outlooks(
        [outlook(1, WR, 14.0, week=1), outlook(2, WR, 14.0, week=1), outlook(2, WR, 14.0, week=2)]
    )
    assert panel.n_weeks == 2
    assert not panel.has_game[1, 0]
    draw = D.WeeklySampler(panel, seed=1).draw(200)
    assert np.all(draw.points[:, 1, 0] == 0)


def test_espn_bye_weeks_are_real_bye_weeks():
    byes = D.espn_bye_weeks(2026, offline=True)
    assert len(byes) >= 30
    assert all(4 <= w <= 15 for w in byes.values())


# --------------------------------------------------------------------------------------
# Common random numbers
# --------------------------------------------------------------------------------------


def _panel(extra: bool = False, n_players: int = 40) -> D.SimPanel:
    outlooks = [
        o
        for p in range(1, n_players + 1)
        for o in season(
            p, [QB, RB, WR, TE][p % 4], 4.0 + p % 13, pro_team_id=1 + p % 16, weeks=range(1, 15)
        )
    ]
    if extra:
        outlooks += season(9_999, WR, 11.0, pro_team_id=99, weeks=range(1, 15))
    return D.SimPanel.from_outlooks(outlooks)


def test_the_same_sampler_returns_the_same_tensor_object():
    sampler = D.WeeklySampler(_panel(), seed=7)
    first = sampler.draw(200)
    assert sampler.draw(200) is first


def test_two_samplers_with_the_same_seed_are_bit_identical():
    a = D.WeeklySampler(_panel(), seed=7).draw(300)
    b = D.WeeklySampler(_panel(), seed=7).draw(300)
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.available, b.available)


def test_a_different_seed_is_a_different_universe():
    a = D.WeeklySampler(_panel(), seed=7).draw(300)
    b = D.WeeklySampler(_panel(), seed=8).draw(300)
    assert not np.array_equal(a.points, b.points)


def test_adding_a_player_leaves_every_other_offense_untouched():
    """Why the base streams are keyed on player id rather than array position.

    A waiver search adds one free agent to the panel and re-evaluates. If that
    reshuffled the whole league the paired comparison would be worthless.
    """
    base = D.WeeklySampler(_panel(), seed=7).draw(300)
    grown = D.WeeklySampler(_panel(extra=True), seed=7).draw(300)
    shared = base.panel.player_ids.tolist()
    assert np.array_equal(base.points_for(shared), grown.points_for(shared))
    assert np.array_equal(base.available_for(shared), grown.available_for(shared))


def test_the_new_player_does_not_have_to_sort_last():
    """The version of the above that actually distinguishes the two designs.

    `_panel(extra=True)` adds id 9,999, which sorts after everyone -- so every other
    player keeps his column index and a stream keyed on the COLUMN is indistinguishable
    from one keyed on the player id. That is not the real case: ESPN ids are handed out
    over time, so the veteran sitting on the wire in October has a lower id than half the
    roster, and inserting him shifts every column after his. Under column-keyed streams
    that re-rolls the entire league and the paired comparison silently becomes an
    unpaired one -- with no symptom other than a noisier answer.
    """
    incumbents = [2 * p for p in range(1, 41)]  # even ids, so an odd one lands between
    newcomer = incumbents[len(incumbents) // 2] + 1
    assert newcomer not in incumbents and incumbents[0] < newcomer < incumbents[-1]

    def build(extra: bool) -> D.SimPanel:
        outlooks = [
            o
            for p in incumbents
            for o in season(
                p, [QB, RB, WR, TE][p % 4], 4.0 + p % 13, pro_team_id=1 + p % 16, weeks=range(1, 15)
            )
        ]
        if extra:
            outlooks += season(newcomer, WR, 11.0, pro_team_id=99, weeks=range(1, 15))
        return D.SimPanel.from_outlooks(outlooks)

    grown_panel = build(True)
    inserted = int(grown_panel.index_of([newcomer])[0])
    assert 0 < inserted < grown_panel.n_players - 1  # columns really do shift behind him

    base = D.WeeklySampler(build(False), seed=7).draw(300)
    grown = D.WeeklySampler(grown_panel, seed=7).draw(300)
    assert np.array_equal(base.points_for(incumbents), grown.points_for(incumbents))
    assert np.array_equal(base.available_for(incumbents), grown.available_for(incumbents))


def test_a_new_free_agent_re_mixes_at_most_his_own_nfl_team():
    """The blast radius of a panel change, checked rather than assumed."""

    def panel(extra: bool) -> D.SimPanel:
        outlooks = [
            outlook(p, [QB, RB, WR, TE][p % 4], 4.0 + p % 13, week=w, pro_team_id=1 + p % 8)
            for p in range(1, 41)
            for w in range(1, 8)
        ]
        if extra:
            outlooks += [outlook(9_999, WR, 11.0, week=w, pro_team_id=3) for w in range(1, 8)]
        return D.SimPanel.from_outlooks(outlooks)

    base = D.WeeklySampler(panel(False), seed=7).draw(300)
    grown = D.WeeklySampler(panel(True), seed=7).draw(300)
    elsewhere = [p for p in range(1, 41) if 1 + p % 8 != 3]
    same_team = [p for p in range(1, 41) if 1 + p % 8 == 3]
    assert same_team  # the addition really does land on an occupied block
    assert np.array_equal(base.points_for(elsewhere), grown.points_for(elsewhere))
    # And on this block, which still factors by Cholesky, even the team-mates survive.
    assert np.array_equal(base.points_for(same_team), grown.points_for(same_team))


def test_paired_differences_kill_most_of_the_monte_carlo_noise():
    """The whole reason 2,000 sims suffice: swap one player and difference.

    Independent draws give the difference the SUM of the two arms' variances. Sharing
    one tensor leaves only the two swapped players, and the fourteen the rosters have
    in common cancel exactly. Measured here on a fifteen-man roster: the paired
    difference has 19% of an arm's variance and a tenth of the unpaired difference's,
    against the 7.2x that RESEARCH.md's "2,000 where 14,400 would be needed" implies.
    """
    roster = list(range(1, 16))
    sampler = D.WeeklySampler(_panel(n_players=40), seed=7)
    draw = sampler.draw(2_000)

    def team_total(ids: list[int]) -> np.ndarray:
        return draw.points_for(ids).sum(axis=(1, 2), dtype=np.float64)

    arm_a = team_total(roster)
    arm_b = team_total([*roster[:-1], 25])
    paired = arm_b - arm_a

    assert paired.var() < 0.3 * min(arm_a.var(), arm_b.var())
    independent = D.WeeklySampler(_panel(n_players=40), seed=8)
    other = (
        independent.draw(2_000).points_for([*roster[:-1], 25]).sum(axis=(1, 2), dtype=np.float64)
    )
    unpaired = other - arm_a
    assert unpaired.var() > 5.0 * paired.var()
    # Both estimate the same thing, so the saving is pure precision.
    assert paired.mean() == pytest.approx(unpaired.mean(), abs=4 * unpaired.std() / np.sqrt(2_000))


def test_common_random_draw_is_a_one_liner_over_the_same_machinery():
    outlooks = list(season(1, WR, 12.0, pro_team_id=3, weeks=range(1, 5)))
    a = D.common_random_draw(outlooks, n_sims=200, seed=3)
    b = D.common_random_draw(outlooks, n_sims=200, seed=3)
    assert np.array_equal(a.points, b.points)


# --------------------------------------------------------------------------------------
# Panel plumbing
# --------------------------------------------------------------------------------------


def test_panel_indexing_is_by_player_id():
    panel = _panel(n_players=10)
    idx = panel.index_of([5, 2])
    assert panel.player_ids[idx].tolist() == [5, 2]
    with pytest.raises(KeyError):
        panel.index_of([5, 12345])
    with pytest.raises(KeyError):
        panel.week_index(99)


def test_panel_builds_from_player_outlooks():
    weeks = {w: outlook(4, RB, 10.0, week=w, pro_team_id=6) for w in range(1, 5)}
    player = PlayerOutlook(player_id=4, name="A Back", position_id=RB, pro_team_id=6, weeks=weeks)
    panel = D.SimPanel.from_players([player])
    assert panel.n_players == 1
    assert panel.n_weeks == 4


def test_empty_panel_is_an_error_not_an_empty_tensor():
    with pytest.raises(ValueError):
        D.SimPanel.from_outlooks([])


def test_draw_rejects_a_non_positive_sim_count():
    with pytest.raises(ValueError):
        D.WeeklySampler(_panel(n_players=4), seed=1).draw(0)


def test_totals_sum_the_requested_weeks_only():
    outlooks = season(1, WR, 12.0, pro_team_id=3, weeks=range(1, 5))
    draw = D.common_random_draw(outlooks, n_sims=300, seed=1)
    everything = draw.totals([1])
    playoffs = draw.totals([1], weeks=[3, 4])
    assert playoffs.mean() < everything.mean()
    assert playoffs.shape == (300,)


# --------------------------------------------------------------------------------------
# Against the corpus
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_pairs():
    pairs = cal.load_pairs()
    if pairs.height < 1_000:
        pytest.skip("no snapshot corpus available")
    return pairs


def test_sampler_reproduces_the_measured_blank_rates(corpus_pairs):
    """Feed the corpus's own ESPN projections through and the zero rate must come back.

    RESEARCH.md: QB 3.2%, RB 18.5%, WR 25.9%, TE 32.5%, over 23,999 paired
    player-weeks. This is the end-to-end check that calibration's hurdle curve and this
    module's hurdle draw agree with what actually happened.
    """
    position_ids = corpus_pairs["position_id"].to_numpy()
    projections = corpus_pairs["projection"].to_numpy()
    actuals = corpus_pairs["actual"].to_numpy()
    outlooks = [
        CALIBRATION.outlook(
            player_id=i,
            season=2026,
            week=1,
            position_id=int(position_ids[i]),
            projection=float(projections[i]),
        )
        for i in range(len(projections))
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    x = (
        D.WeeklySampler(panel, seed=42, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(200)
        .points[:, 0, :]
    )
    for position_id in (QB, RB, WR, TE):
        mask = position_ids == position_id
        observed = (actuals[mask] <= 0).mean()
        simulated = (x[:, mask] <= 0).mean()
        assert simulated == pytest.approx(observed, abs=0.01)
        assert x[:, mask].mean() == pytest.approx(actuals[mask].mean(), rel=0.03)
        assert x[:, mask].std() == pytest.approx(actuals[mask].std(), rel=0.05)


def test_simulated_spread_line_matches_the_corpus_spread_line(corpus_pairs):
    """`sigma(mu)` refitted on simulated draws must land on the corpus's own line.

    Note what it does NOT land on: RESEARCH.md's pooled `3.67 + 0.273*mu`. Fitting the
    same count-weighted quantile bins to the corpus gives `2.71 + 0.364*mu` and to the
    simulated draws `2.72 + 0.364*mu` -- so the sampler reproduces the data, and the
    published pooled constant is the binning artifact calibration.py already documents.
    """
    position_ids = corpus_pairs["position_id"].to_numpy()
    projections = corpus_pairs["projection"].to_numpy()
    actuals = corpus_pairs["actual"].to_numpy()
    outlooks = [
        CALIBRATION.outlook(
            player_id=i,
            season=2026,
            week=1,
            position_id=int(position_ids[i]),
            projection=float(projections[i]),
        )
        for i in range(len(projections))
    ]
    mu = np.array([o.mean for o in outlooks])
    panel = D.SimPanel.from_outlooks(outlooks)
    x = (
        D.WeeklySampler(panel, seed=42, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(200)
        .points[:, 0, :]
    )

    def line(squared_deviation: np.ndarray, n_bins: int = 20) -> tuple[float, float]:
        order = np.argsort(mu)
        m, d = mu[order], squared_deviation[order]
        edges = np.linspace(0, m.size, n_bins + 1).astype(int)
        xs, ys, ws = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            if hi <= lo:
                continue
            xs.append(m[lo:hi].mean())
            ys.append(np.sqrt(d[lo:hi].mean()))
            ws.append(hi - lo)
        xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, dtype=float)
        design = np.vstack([np.ones_like(xs), xs]).T * np.sqrt(ws)[:, None]
        coef, *_ = np.linalg.lstsq(design, ys * np.sqrt(ws), rcond=None)
        return float(coef[0]), float(coef[1])

    corpus_intercept, corpus_slope = line((actuals - mu) ** 2)
    sim_intercept, sim_slope = line(((x - mu[None, :]) ** 2).mean(axis=0))
    assert sim_intercept == pytest.approx(corpus_intercept, abs=0.15)
    assert sim_slope == pytest.approx(corpus_slope, abs=0.02)


LINEUP = (
    (QB, 21.5),
    (RB, 15.0),
    (RB, 12.0),
    (WR, 16.0),
    (WR, 13.0),
    (TE, 10.0),
    (WR, 11.5),
    (5, 8.5),
    (16, 8.5),
)


def _lineup_totals(*, stacked: bool, n: int = 100_000) -> np.ndarray:
    outlooks = [
        outlook(i + 1, position_id, mean, pro_team_id=(12 if stacked else 1 + i))
        for i, (position_id, mean) in enumerate(LINEUP)
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    return (
        D.WeeklySampler(panel, seed=3, injuries=D.InjuryModel.off(), dtype=np.float64)
        .draw(n)
        .points[:, 0, :]
        .sum(axis=1)
    )


def test_a_full_lineup_lands_on_the_team_score_anchor():
    """12-team PPR optimal lineups measure at mean 121.9, SD 24.35 -- a CV of 0.200.

    Nine starters on nine different NFL teams, which is what an ordinary lineup is, so
    this checks that the MARGINAL widths compose -- the correlation structure contributes
    nothing here and the docstring should not claim it does. What it is really pinning is
    that summing nine hurdle gammas does not drift off the measured team-score width; the
    tolerance is loose because the anchor is a different roster than this one.
    """
    totals = _lineup_totals(stacked=False)
    cv = totals.std() / totals.mean()
    assert cv == pytest.approx(24.35 / 121.9, rel=0.15)
    # It really is the independent sum, so record that rather than implying otherwise.
    outlooks = [
        outlook(i + 1, position_id, mean, pro_team_id=1 + i)
        for i, (position_id, mean) in enumerate(LINEUP)
    ]
    assert totals.std() == pytest.approx(np.sqrt(sum(o.variance for o in outlooks)), rel=0.01)
    skew = float(((totals - totals.mean()) ** 3).mean() / totals.std() ** 3)
    assert 0.1 < skew < 0.6


def test_stacking_a_lineup_widens_it_by_the_measured_amount():
    """The correlation half of the composition, which the anchor test does not reach.

    Put the same nine starters on one NFL team and the total's variance must gain
    `2 * sum_{i<j} rho_ij sigma_i sigma_j` over the independent sum -- computed here from
    the constants and the outlooks' own SDs, with nothing asked of the sampler but the
    draw. A copula that induced its correlation on the normals and lost it through the
    gamma transform would land short of this.
    """
    outlooks = [
        outlook(i + 1, position_id, mean, pro_team_id=12)
        for i, (position_id, mean) in enumerate(LINEUP)
    ]
    model = D.CorrelationModel()
    independent = sum(o.variance for o in outlooks)
    cross = 2.0 * sum(
        model.rho(a.position_id, b.position_id) * a.sd * b.sd
        for i, a in enumerate(outlooks)
        for b in outlooks[i + 1 :]
    )
    assert cross > 0.15 * independent  # the effect is worth measuring at all
    totals = _lineup_totals(stacked=True, n=400_000)
    assert totals.var() == pytest.approx(independent + cross, rel=0.02)


# --------------------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------------------


def test_a_season_sized_draw_is_fast_enough_to_sit_inside_a_search():
    """2,000 x 18 x 500 is the real workload. Measured at ~1.4s on the dev machine.

    The bound is deliberately loose -- this is a regression guard against someone
    reintroducing a Python loop over simulations, not a benchmark.
    """
    rng = np.random.default_rng(0)
    n_players = 250
    positions = rng.choice([QB, RB, WR, TE], n_players, p=[0.12, 0.28, 0.42, 0.18])
    means = rng.gamma(2.0, 4.0, n_players)
    outlooks = [
        outlook(p + 1, int(positions[p]), float(means[p]), week=w, pro_team_id=1 + p % 32)
        for p in range(n_players)
        for w in range(1, 19)
    ]
    panel = D.SimPanel.from_outlooks(outlooks)
    sampler = D.WeeklySampler(panel, seed=1)
    start = time.perf_counter()
    draw = sampler.draw(1_000)
    elapsed = time.perf_counter() - start
    assert draw.points.shape == (1_000, 18, n_players)
    assert elapsed < 12.0
    # And a second evaluation is free, which is the point of the memo.
    start = time.perf_counter()
    sampler.draw(1_000)
    assert time.perf_counter() - start < 0.05


class TestNegativePlayerIdsFromRealLeagues:
    """ESPN encodes team defenses as negative ids, and every real roster starts one.

    Found by running the season simulator against a live league after the whole
    module's own suite passed on synthetic positive-id panels. The failure mode is
    worth remembering: 100% of tests green, 0% of real leagues able to draw.
    """

    def test_a_real_dst_id_can_be_drawn(self):
        from fantasy_quant.core import WeeklyOutlook
        from fantasy_quant.sim.distributions import SimPanel, WeeklySampler

        outlooks = [
            WeeklyOutlook(
                player_id=pid,
                season=2026,
                week=1,
                position_id=pos,
                mean=8.0,
                sd=6.0,
                p_zero=0.2,
                shape=2.0,
                scale=4.0,
                pro_team_id=24,
            )
            for pid, pos in [(-16024, 16), (4429795, 2)]
        ]
        draw = WeeklySampler(SimPanel.from_outlooks(outlooks), seed=7).draw(64)
        points = getattr(draw, "points", draw)
        assert points.shape[0] == 64
        assert np.isfinite(points).all()

    def test_a_defense_does_not_share_a_stream_with_its_mirror_player_id(self):
        """`abs()` is the obvious fix and it is wrong.

        abs(-16024) == 16024, which is a live player id, so the two would draw
        identical seasons -- silent corruption rather than a loud crash.
        """
        from fantasy_quant.core import WeeklyOutlook
        from fantasy_quant.sim.distributions import SimPanel, WeeklySampler

        outlooks = [
            WeeklyOutlook(
                player_id=pid,
                season=2026,
                week=1,
                position_id=pos,
                mean=8.0,
                sd=6.0,
                p_zero=0.2,
                shape=2.0,
                scale=4.0,
                pro_team_id=24,
            )
            for pid, pos in [(-16024, 16), (16024, 3)]
        ]
        panel = SimPanel.from_outlooks(outlooks)
        points = getattr(draw := WeeklySampler(panel, seed=7).draw(256), "points", draw)
        ids = list(panel.player_ids)
        a = points[:, :, ids.index(-16024)]
        b = points[:, :, ids.index(16024)]
        assert not np.array_equal(a, b)

    def test_stream_key_is_injective_and_non_negative(self):
        from fantasy_quant.sim.distributions import _stream_key

        ids = list(range(-20000, -15900)) + list(range(0, 5000))
        keys = [_stream_key(i) for i in ids]
        assert min(keys) >= 0
        assert len(set(keys)) == len(ids), "distinct ids must keep distinct streams"
