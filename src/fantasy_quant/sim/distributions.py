"""How a set of `WeeklyOutlook`s becomes a `[sim, week, player]` tensor of points.

Everything above this module reasons about one player at a time. Everything below it
-- lineups, matchups, playoff brackets, title odds -- only ever sees the tensor. So
this is the single place where the four things that make a fantasy season *not* a bag
of independent draws are modelled, and each of them is here because getting it wrong
biases the answer in a specific, known direction.

**1. One uniform per player-week, not two.**
A hurdle gamma is a mixture, and its inverse CDF is

    F(x) = p_zero + (1 - p_zero) * G(x)   =>   F^-1(u) = 0            if u <= p_zero
                                                        G^-1(q), q = (u - p_zero)/(1 - p_zero)

so a *single* uniform decides both whether the player blanks and how big the game is
when he does not. The obvious alternative -- flip an independent Bernoulli for the
hurdle, then draw a correlated gamma for the magnitude -- looks equivalent marginally
and is not. Under a copula it decouples the blanks: Josh Allen can post 4 points while
his WR1 still draws a median game, because only the magnitudes were correlated. Real
weeks do not work that way; an offense that gets shut out shuts everyone out together.
Driving the hurdle from the same uniform makes the copula act on the *whole* mixture,
which is both the statistically correct thing (it is literally the inverse CDF of the
marginal, so marginals are exact by construction) and the behaviourally correct one:
a QB dud drags his receivers toward zero and a shootout lifts them together.

**2. Correlation is block-diagonal by NFL team, and the blocks are tiny.**
Measured on our corpus: rho(QB,WR)=0.30, rho(QB,TE)=0.20, rho(QB,RB)=0.08,
rho(TE,TE)=0.13, rho(RB,RB)=-0.09, everything else 0. Different-team pairs correlate
+0.003 and a league-wide weekly factor explains 0.71% of residual variance -- there is
no market factor to model. So a 500-player panel is never a 500x500 matrix; it is ~32
blocks of under twenty, each with its own Cholesky. That is what makes a full-season
draw take a second rather than a minute.

The blocks are not always positive definite, and this is not an edge case. Two
quarterbacks on one team each correlate 0.30 with all of that team's receivers and zero
with each other, which no correlation matrix can do -- so a panel holding whole depth
charts (a 500-player pool is ~15 per NFL team) trips it on most blocks. Marginal
estimates are under no obligation to be jointly consistent, and the question is not
whether to give something up but *what*: naive repair spreads the damage evenly and
halves the one correlation the model exists to represent. So a block that will not
factor is first relaxed toward the raw constants (`relax_to_feasible`, which gives back
the transform correction on the fringe players who forced the infeasibility) and only
then projected, weighted by variance (`nearest_correlation_factor`, which spends the
remaining error where a lineup total is least sensitive to it). On a realistic
fifteen-man depth chart with two quarterbacks that holds QB-WR1 at 0.285 against a
target of 0.300, where an unweighted projection of the unrelaxed matrix gives 0.153.
Marginals survive all of it untouched, because every step preserves the unit diagonal.

**3. The measured rhos are Pearson correlations of points, and a Gaussian copula does
not reproduce them at face value.**
Push a latent rho of 0.30 through a hurdle gamma with 26% zero mass and the realized
Pearson correlation of the *points* comes out near 0.28 -- the non-linear marginal
transform attenuates it, and the attenuation gets worse the more mass sits at zero. So
the latent correlation is solved for rather than assumed, from the exact identity
`Corr(X, Y) = sum_k g_k(X) g_k(Y) rho^k` over the marginals' Hermite spectra (see
`hermite_coefficients`). Keeping only `k = 1` is the familiar attenuation factor and it
is not enough: for the sub-one-point free agents a waiver search spends its time on, it
asks for a latent 0.707 and realizes 0.518 against a target of 0.300. Four terms hold
every regime to within about two Monte Carlo standard errors. Set
`CorrelationModel(match_pearson=False)` to read the constants as copula parameters
instead, which lands about 4% low on ordinary starters and much further off below that.

**4. Injuries are absorbing, and this is the biggest single modelling gap in the
open-source alternatives.**
`ffsimulator` and friends draw each week independently, so a running back who misses
week 3 is fully healthy in week 4 with probability ~0.95. The real conditional
probability is under 0.4. An IID hazard produces the right *number* of missed games
and completely the wrong *shape*: it sprinkles single-week absences across every team
instead of removing one RB1 from one roster for six straight weeks. The left tail of a
season -- the outcome a title chase actually turns on -- is where those two differ, and
the IID version understates it badly. Here the hazard fires once and draws a duration,
and the player stays out until it runs down.

**Common random numbers are the default, not an option.**
`WeeklySampler.draw` memoizes, so calling it twice returns the *same* tensor object; a
candidate search evaluates roster A and roster B against one shared draw and differences
out the Monte Carlo noise. A further property makes that robust across *different*
panels: a player's own base randomness is derived from `(seed, player_id)` rather than
from his position in the array, so adding a free agent to the panel re-mixes at most his
own NFL team's block and leaves every other team's season bit-identical. (When that
block still factors by Cholesky and the new id sorts last, even his new team-mates are
untouched, because the Cholesky of a leading principal submatrix is the leading block of
the whole factor. A block that needed the PD repair loses that, since the eigenvalue
route is not triangular.) Still: build one panel over the union of every player any
candidate might touch, draw once, and index.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import gammaincinv, gammaln, ndtr, ndtri

from ..core import QB, RB, TE, WR, PlayerOutlook, WeeklyOutlook
from ..projections.calibration import MAX_GAMMA_SHAPE

log = logging.getLogger(__name__)

#: Fixed so that a run is reproducible without the caller having to remember a seed.
#: Any int works; what matters is that one search uses one seed throughout.
DEFAULT_SEED = 20260907

#: RESEARCH.md: 2,000 paired sims buy what 14,400 independent ones would.
DEFAULT_SIMS = 2000

# --------------------------------------------------------------------------------------
# Measured constants
# --------------------------------------------------------------------------------------

#: Residual correlation between two players on the SAME NFL team, keyed by the sorted
#: pair of `defaultPositionId`s. From 23,999 paired player-weeks, 2022-2025. Pairs not
#: listed here are zero -- including WR/WR, which is the interesting one: a shootout
#: lifts both receivers but target competition pulls them apart, and the two effects
#: measure out to nothing.
SAME_TEAM_RHO: Mapping[tuple[int, int], float] = {
    (QB, RB): 0.08,
    (QB, WR): 0.30,
    (QB, TE): 0.20,
    (RB, RB): -0.09,
    (TE, TE): 0.13,
}

#: Different-team pairs. Measured at +0.003, and a league-wide weekly factor explains
#: 0.71% of residual variance. Both are noise; the model treats cross-team as exactly
#: zero and this constant exists to record what was measured, not to be used.
CROSS_TEAM_RHO = 0.003

#: Per *at-risk game* probability that a player picks up an absence. An at-risk game is
#: one he would otherwise have played: byes and weeks he is already out do not count.
INJURY_HAZARD: Mapping[int, float] = {QB: 0.025, RB: 0.052, WR: 0.045, TE: 0.049}

#: P(absence lasts exactly k games) for k = 1..17, with index 17 absorbing "18 or more",
#: which in a fantasy season means "gone".
#:
#: Kaplan-Meier over 1,030 absence spells by RB/WR/TE regulars (>=40% of team offensive
#: snaps in the game before they vanished), nflverse snap counts 2019-2025, regular
#: season only. 28.8% of spells are right-censored by the end of the season, which is
#: exactly why this is a KM estimate and not a histogram: counting only spells that
#: closed would throw away every season-ending injury and halve the tail. Masses for
#: k=1..8 are the raw KM steps; from k=9 the risk set is under 50 and the estimate is
#: continued at the constant hazard measured over k=9..17 (7.96%/game), with the
#: remaining 7.1% at "rest of season". Mean 4.3 games.
#:
#: Pooled over RB/WR/TE deliberately. Per-position curves are within noise of each
#: other, and QB is excluded outright because a snap-count absence at QB is mostly a
#: benching -- see `INJURY_HAZARD` note in the module tests.
ABSENCE_PMF: tuple[float, ...] = (
    0.3845,
    0.1865,
    0.1005,
    0.0654,
    0.0461,
    0.0279,
    0.0217,
    0.0170,
    0.0120,
    0.0110,
    0.0101,
    0.0093,
    0.0086,
    0.0079,
    0.0073,
    0.0067,
    0.0062,
    0.0713,
)


# --------------------------------------------------------------------------------------
# Gamma quantiles, fast
# --------------------------------------------------------------------------------------

#: Smallest gamma shape the quantile machinery will represent. A hurdle gamma fitted to
#: a sub-0.02-point projection lands here: `shape ~= mu^2 / ((1-p) sd^2)`, and the spread
#: line puts a floor of ~2.2-4.7 on `sd` however small `mu` gets, so a WR calibrated at
#: mu = 0.004 comes out at shape 1.4e-5 and one at mu = 0.001 at 8.6e-7. Those are real
#: panel entries -- `CalibrationSet.outlook_from_mean` takes an ensemble mean and does
#: not floor it -- so they have to be handled rather than assumed away.
MIN_GAMMA_SHAPE = 1.0e-10

#: Below this shape the sampler cannot reproduce the outlook's SD, and says so once.
#: Such a marginal carries its whole width in a one-in-a-million draw of several thousand
#: points -- at mu = 0.004 the scale is 1,840 and 99.99% of the positive part is under
#: 0.001 -- so no finite sample reproduces it and no float64 quantile even represents the
#: bulk of it (`exp(-69315)` is zero). Measured on a stratified sweep at shape 1.4e-5:
#: 200k strata recover a mean of 0.0022 and an SD of 0.90, 4M strata 0.0039 and 2.44,
#: against a stated 0.004 and 2.71 -- converging from below, and at 2,000 sims not
#: converging at all. The mean survives in expectation; the width does not.
_UNRESOLVED_SHAPE = 1.0e-4

#: `log G^-1(q; a, 1)` switches to the small-x branch below this. See
#: `log_gamma_quantile`; at `x = 1.5e-8` the neglected term is under 1e-8 relative.
_SMALL_LOG_QUANTILE = -18.0


def log_gamma_quantile(shape: np.ndarray, q: np.ndarray) -> np.ndarray:
    """`log G^-1(q; shape, 1)`, correct where the quantile itself underflows float64.

    `scipy.special.gammaincinv` returns exactly 0.0 once the true quantile falls under
    ~1e-308, which for a shape of 1e-5 is every quantile below the 99.9th percentile.
    For small `x` the regularized lower incomplete gamma is
    `P(a, x) = x^a / Gamma(a+1) * (1 + O(a x))`, so

        log x = (log q + lgamma(a + 1)) / a

    exactly in the limit, and to better than 1e-8 relative wherever it returns
    `x < 1.5e-8`. Above that `gammaincinv` is itself exact and is used directly. Checked
    both ways: the two branches agree with `gammaincinv` to 1.5e-8 relative on
    `a = 0.005..60`, and the small-x branch round-trips through the forward `gammainc`
    to 1e-15.

    Be clear about what this does and does not buy, because the two are easy to
    conflate. It does NOT move any moment the sampler currently produces: a quantile of
    `exp(-69315)` and one of `exp(-691)` both contribute nothing to a mean, and building
    the table with `log(max(gammaincinv(...), 1e-300))` instead gives the same draw to
    six digits. The 12x mean and 3.7x SD inflation this module used to have on near-zero
    projections came from the SHAPE FLOOR in `GammaQuantileTable.build`, not from here.
    What it buys is that the stored surface is not a lie: the table says -69315 where the
    truth is -69315, so its accuracy no longer depends on nobody ever moving `z_max` or
    reading the low corner of the grid, and the exact and interpolated paths are the same
    function rather than two functions that happen to agree.
    """
    a, qq = np.broadcast_arrays(np.asarray(shape, dtype=float), np.asarray(q, dtype=float))
    a = np.maximum(a, MIN_GAMMA_SHAPE)
    out = (np.log(qq) + gammaln(a + 1.0)) / a
    big = out >= _SMALL_LOG_QUANTILE
    if big.any():
        out[big] = np.log(np.maximum(gammaincinv(a[big], qq[big]), 1e-300))
    return out


@dataclass(frozen=True, slots=True)
class GammaQuantileTable:
    """`log G^-1(Phi(z); shape, 1)` on a regular grid, bilinearly interpolated.

    `scipy.special.gammaincinv` costs ~320ns per element. An 18-million-element draw
    therefore spends about six seconds inside it, which is most of the run. The
    quantile surface is smooth in `(log shape, z)` -- log-quantile is near-quadratic in
    z at the low end and near-logarithmic at the high end -- so a grid at 0.01 spacing
    in both directions and a bilinear read is accurate to about 4e-4 relative and
    seventeen times faster. On a 20-point week that is an error of 0.008 points.

    Indexed by `z = Phi^-1(q)` rather than by `q` because the interesting part of a
    fantasy distribution is the right tail, and a uniform grid in `q` has no resolution
    there at all.
    """

    shape_lo: float
    log_shape_step: float
    z_lo: float
    z_step: float
    #: (n_shape, n_z) of log quantiles.
    table: np.ndarray

    @classmethod
    def build(
        cls,
        shape_lo: float,
        shape_hi: float,
        *,
        log_shape_step: float = 0.01,
        z_step: float = 0.01,
        z_max: float = 6.5,
        max_shape_nodes: int = 4096,
    ) -> GammaQuantileTable:
        """Cover `[shape_lo, shape_hi]`, padded, at the requested resolution.

        The floor is `MIN_GAMMA_SHAPE`, not something comfortable: `shape_index` CLAMPS,
        so a floor above what the panel actually contains does not degrade a player's
        draw, it replaces his distribution with a different one. A WR calibrated at
        mu = 0.004 has shape 1.4e-5; against the 1e-4 floor this class used to carry, he
        was sampled as shape 1e-4 -- a mean of 0.051 against a stated 0.004 and an SD of
        10.1 against a stated 2.71. Nothing warned, and the free-agent pool a waiver
        search walks is full of him.
        """
        lo = float(max(min(shape_lo, shape_hi), MIN_GAMMA_SHAPE))
        hi = float(min(max(shape_lo, shape_hi), MAX_GAMMA_SHAPE * 1.01))
        hi = max(hi, lo * 1.001)
        lo, hi = lo * 0.99, hi * 1.01
        span = np.log(hi) - np.log(lo)
        n_shape = min(max(int(np.ceil(span / log_shape_step)) + 1, 8), max_shape_nodes)
        n_z = max(int(np.ceil(2.0 * z_max / z_step)) + 1, 8)

        shapes = np.exp(np.linspace(np.log(lo), np.log(hi), n_shape))
        zs = np.linspace(-z_max, z_max, n_z)
        q = np.clip(ndtr(zs), 1e-16, 1.0 - 1e-16)
        # At tiny shapes the quantile underflows float64 outright, so the table is built
        # in log space by `log_gamma_quantile` rather than by logging a clipped value.
        # The interpolation error scales with |log x|, which is what makes a grid this
        # coarse survive a surface this steep: it is largest where the quantile is
        # 1e-30000 and smallest where the mass is.
        table = log_gamma_quantile(shapes[:, None], q[None, :])
        return cls(
            shape_lo=float(shapes[0]),
            log_shape_step=float(np.log(shapes[1]) - np.log(shapes[0])),
            z_lo=float(zs[0]),
            z_step=float(zs[1] - zs[0]),
            table=table,
        )

    @property
    def z_hi(self) -> float:
        return self.z_lo + self.z_step * (self.table.shape[1] - 1)

    def shape_index(self, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid cell and interpolation weight for a shape. Hoisted out of the hot loop.

        Shape is constant across simulations for a given player-week, so this is
        computed once over `(week, player)` and reused for every one of the thousands
        of sims that share it.
        """
        n = self.table.shape[0]
        # Clamped, not extrapolated: a bilinear read run off the end of the grid does
        # not degrade, it diverges. The build pads the range so this only bites on a
        # shape the panel never declared.
        f = np.clip(
            (np.log(np.maximum(shape, 1e-300)) - np.log(self.shape_lo)) / self.log_shape_step,
            0.0,
            n - 1.0,
        )
        idx = np.clip(f.astype(np.int64), 0, n - 2)
        return idx, f - idx

    def lookup(self, shape_idx: np.ndarray, shape_weight: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Quantiles of Gamma(shape, 1). `shape_idx`/`shape_weight` broadcast against `z`."""
        fz = (np.clip(z, self.z_lo, self.z_hi) - self.z_lo) / self.z_step
        iz = np.clip(fz.astype(np.int64), 0, self.table.shape[1] - 2)
        wz = fz - iz
        lo = self.table[shape_idx, iz] * (1.0 - wz) + self.table[shape_idx, iz + 1] * wz
        hi = self.table[shape_idx + 1, iz] * (1.0 - wz) + self.table[shape_idx + 1, iz + 1] * wz
        return np.exp(lo * (1.0 - shape_weight) + hi * shape_weight)

    def __call__(self, shape: np.ndarray, z: np.ndarray) -> np.ndarray:
        idx, weight = self.shape_index(np.asarray(shape, dtype=float))
        return self.lookup(idx, weight, z)


def hurdle_gamma_quantile(
    u: np.ndarray,
    p_zero: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
    *,
    table: GammaQuantileTable | None = None,
) -> np.ndarray:
    """`F^-1(u)` for the hurdle gamma -- the whole mixture, from one uniform.

    Exposed on its own because it is the piece worth testing directly: feed it uniforms
    and the sampled mean, sd and P(X<=0) must come back as the `WeeklyOutlook` stated
    them. `table=None` takes the exact `gammaincinv` path.
    """
    p = np.minimum(p_zero, 1.0 - 1e-12)
    q = np.clip((u - p) / (1.0 - p), 1e-12, 1.0 - 1e-12)
    x = np.exp(log_gamma_quantile(shape, q)) if table is None else table(shape, ndtri(q))
    return np.where(u <= p_zero, 0.0, x * scale)


# --------------------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------------------


def _unit_diagonal_factor(corr: np.ndarray, min_eigenvalue: float) -> np.ndarray:
    """Symmetric square root with clipped eigenvalues, rows rescaled to unit norm.

    The rescaling is what turns a merely-PSD repair into a *correlation* matrix, and
    returning the factor rather than the matrix is deliberate: the factor is exact,
    whereas re-factoring the repaired matrix can fail again on the eigenvalue floor.
    """
    values, vectors = np.linalg.eigh(corr)
    root = (vectors * np.sqrt(np.clip(values, min_eigenvalue, None))) @ vectors.T
    norms = np.linalg.norm(root, axis=1)
    norms[norms <= 0] = 1.0
    return root / norms[:, None]


def nearest_correlation_factor(
    corr: np.ndarray, *, weights: np.ndarray | None = None, min_eigenvalue: float = 1e-8
) -> tuple[np.ndarray, bool]:
    """A factor `L` with `L @ L.T` a valid correlation matrix close to `corr`.

    Cholesky when the matrix is already positive definite, which is the ordinary case.
    It is not always: the measured rhos are marginal pairwise estimates and were never
    constrained to be jointly consistent. Two quarterbacks on one team each correlate
    0.30 with all of that team's receivers and zero with each other, which no
    correlation matrix can do -- and a candidate search that adds a whole depth chart
    to the panel produces exactly that. Crashing on it would make the search fail on
    the rosters it exists to explore.

    `weights` decides *whose* correlations get sacrificed, and it matters more than the
    projection method does. What a simulation is finally asked for is the variance of a
    lineup total, `sum_ij rho_ij sigma_i sigma_j`, so an error `d_ij` costs
    `d_ij sigma_i sigma_j`. Minimising `||W^(1/2) (X - R) W^(1/2)||_F` with `W` the
    per-player VARIANCE is exactly minimising the squared covariance error, which
    concentrates the damage on the fringe players whose correlations nothing depends on.
    Measured on a fifteen-man depth chart with two quarterbacks: unweighted, the headline
    QB-WR1 correlation is repaired from 0.300 down to 0.266; variance-weighted it holds
    at 0.284, and the deep bench absorbs the difference. One weighted eigenvalue clip
    gets within 0.005 of what two hundred alternating projections converge to, so this
    does one.
    """
    try:
        return np.linalg.cholesky(corr), False
    except np.linalg.LinAlgError:
        pass
    if weights is not None:
        scale = np.sqrt(np.maximum(np.asarray(weights, dtype=float), 1e-12))
        outer = np.outer(scale, scale)
        values, vectors = np.linalg.eigh(outer * corr)
        rebuilt = ((vectors * np.clip(values, 0.0, None)) @ vectors.T) / outer
        np.fill_diagonal(rebuilt, 1.0)
        corr = rebuilt
    return _unit_diagonal_factor(corr, min_eigenvalue), True


def relax_to_feasible(
    latent: np.ndarray, target: np.ndarray, *, min_eigenvalue: float = 1e-6, steps: int = 16
) -> tuple[np.ndarray, float]:
    """Slide the solved latent matrix back toward the raw targets until it is PD.

    Solving each pair exactly for its own marginals makes the *block* harder to satisfy
    than the raw constants were: a receiver who blanks 85% of the time needs a latent
    0.72 to realize a Pearson 0.30, and six of those against one quarterback is not a
    correlation matrix even though six raw 0.30s is. Interpolating
    `(1-t)*latent + t*target` and taking the smallest feasible `t` gives up the
    transform correction exactly where the block cannot carry it -- which is on the
    fringe players, since a starter's latent barely differs from his target in the first
    place. Measured on a full one-quarterback depth chart: `t = 0.51`, the QB-WR1
    correlation still lands at 0.295, and the sixth receiver drops to 0.238.

    Returns the relaxed matrix and the `t` used. `t = 1` means even the raw constants
    are infeasible for this block and the caller must still project.
    """
    if np.linalg.eigvalsh(latent).min() >= min_eigenvalue:
        return latent, 0.0
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        candidate = latent + mid * (target - latent)
        np.fill_diagonal(candidate, 1.0)
        if np.linalg.eigvalsh(candidate).min() >= min_eigenvalue:
            hi = mid
        else:
            lo = mid
    relaxed = latent + hi * (target - latent)
    np.fill_diagonal(relaxed, 1.0)
    return relaxed, hi


@dataclass(frozen=True, slots=True)
class CorrelationModel:
    """Within-team residual correlation, as a Gaussian copula.

    `match_pearson` says how to read the constants. True (the default) treats them as
    what they were measured as -- Pearson correlations between realized fantasy points
    -- and solves for the latent normal correlation that reproduces them through each
    player-week's own marginal. False treats them as the copula parameters directly,
    which is faster to set up and lands roughly 7% low on the realized correlation.
    """

    same_team: Mapping[tuple[int, int], float] = field(default_factory=lambda: SAME_TEAM_RHO)
    match_pearson: bool = True
    #: Latent correlations are clipped here. A block of many receivers with an inflated
    #: QB correlation can otherwise ask for something outside [-1, 1] outright.
    max_latent: float = 0.99

    def rho(self, position_a: int, position_b: int) -> float:
        """Correlation for two players on the same team. Zero for anything unmeasured."""
        key = (position_a, position_b) if position_a <= position_b else (position_b, position_a)
        return float(self.same_team.get(key, 0.0))

    def target_matrix(self, position_ids: Sequence[int]) -> np.ndarray:
        """The measured Pearson correlation matrix for one team block."""
        n = len(position_ids)
        out = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                r = self.rho(int(position_ids[i]), int(position_ids[j]))
                out[i, j] = out[j, i] = r
        return out

    def couples(self, position_ids: Sequence[int]) -> bool:
        """Whether this block has any non-zero off-diagonal at all."""
        return any(
            self.rho(int(a), int(b)) != 0.0
            for i, a in enumerate(position_ids)
            for b in position_ids[i + 1 :]
        )


#: Terms kept in the Hermite expansion of the copula transform. One is not enough --
#: see `hermite_coefficients` -- and beyond four nothing moves.
HERMITE_ORDER = 4


def hermite_coefficients(
    p_zero: np.ndarray,
    shape: np.ndarray,
    *,
    table: GammaQuantileTable | None = None,
    n_nodes: int = 512,
    order: int = HERMITE_ORDER,
) -> np.ndarray:
    """`g_k = E[X He_k(Z)] / (sd(X) sqrt(k!))` for `k = 1..order`, the copula's spectrum.

    Expand the transform `X = F^-1(Phi(Z))` in Hermite polynomials. Because
    `E[He_j He_k] = k! delta_jk` under the standard normal, a pair of such variables
    with latent correlation `rho` satisfies, exactly,

        Corr(X, Y) = sum_k g_k(X) g_k(Y) rho^k

    so recovering a measured Pearson correlation means solving that polynomial for
    `rho`, which `solve_latent_correlation` does.

    Truncating at `k = 1` is the textbook attenuation factor `Corr(X, Z)` and it is a
    trap. For two ordinary starters it is fine (target 0.300 comes back at 0.304), but
    the term it drops grows with the zero mass, and a hurdle gamma's zero mass is
    exactly what makes fantasy points fantasy points. For a QB projected at 3 and a
    receiver at 0.8 -- 40% and 69% blank rates, the free-agent pool a waiver search
    spends all its time in -- first order asks for a latent 0.707 and realizes 0.518
    against a target of 0.300. Four terms bring that to 0.301, and six change nothing.

    `g` does not depend on `scale`: both the numerator and `sd(X)` are linear in it. So
    this is a function of `(p_zero, shape)` alone and can be evaluated once per
    player-week. The integral is Gauss-Legendre in `u`, where the transform is the
    quantile function and therefore smooth.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    z = ndtri(np.clip(u, 1e-15, 1.0 - 1e-15))

    p = np.minimum(np.asarray(p_zero, dtype=float), 1.0 - 1e-9)[..., None]
    a = np.asarray(shape, dtype=float)[..., None]
    q = np.clip((u - p) / (1.0 - p), 1e-12, 1.0 - 1e-12)
    x = np.exp(log_gamma_quantile(a, q)) if table is None else table(a, ndtri(q))
    x = np.where(u > p, x, 0.0)
    xw = w * x

    # Scale-free moments: with scale = 1 the positive part has mean = variance = shape.
    p_flat, a_flat = p[..., 0], a[..., 0]
    sd = np.sqrt((1.0 - p_flat) * a_flat + p_flat * (1.0 - p_flat) * a_flat * a_flat)
    inv_sd = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0)

    out = np.empty((order, *sd.shape))
    previous, current = np.ones_like(z), z  # He_0, He_1
    factorial = 1.0
    for k in range(1, order + 1):
        factorial *= k
        out[k - 1] = np.sum(xw * current, axis=-1) * inv_sd / np.sqrt(factorial)
        previous, current = current, z * current - k * previous
    # A degenerate marginal carries no correlation; make it ask for the target itself
    # rather than divide by zero downstream.
    out[0] = np.where(sd > 0, out[0], 1.0)
    return out


def pearson_from_latent(latent: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """`sum_k spectrum[k] * latent^(k+1)` -- the forward direction of the inversion.

    `solve_latent_correlation` runs this backwards; this runs it forwards, which is what
    a *diagnostic* needs. The repair works on the latent matrix, but the number a reader
    of the log cares about is what the simulated POINTS end up correlating at, and on a
    block with heavy zero mass those two are nowhere near each other: a repaired latent
    of 0.251 against a target of 0.300 sounds like a 0.05 loss and is a 0.26 one. Checked
    against direct two-dimensional quadrature over the copula on a fifteen-man depth
    chart: agrees on all 105 pairs to 1.5e-4.
    """
    out = np.zeros_like(latent)
    power = np.ones_like(latent)
    for k in range(spectrum.shape[0]):
        power = power * latent
        out = out + spectrum[k] * power
    return out


def solve_latent_correlation(
    target: np.ndarray, spectrum: np.ndarray, *, max_latent: float = 0.99, iterations: int = 32
) -> np.ndarray:
    """Invert `sum_k spectrum[k] * rho^(k+1) = target` for `rho`.

    Bisection rather than Newton: the polynomial is monotone on the bracket in every
    case that arises, the brackets are `[0, max]` or `[-max, 0]` by the sign of the
    target, and thirty-two halvings put it past float noise. A target the marginals
    cannot reach even at `rho = 1` -- two near-certain blanks asked to correlate 0.30 --
    converges to the bracket edge, which is the right clamp rather than a divergence.
    """
    lo = np.where(target >= 0.0, 0.0, -max_latent)
    hi = np.where(target >= 0.0, max_latent, 0.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        value = np.zeros_like(mid)
        power = np.ones_like(mid)
        for k in range(spectrum.shape[0]):
            power = power * mid
            value = value + spectrum[k] * power
        below = value < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InjuryModel:
    """An absorbing absence: one hazard draw, then a duration that has to run down.

    Contrast the IID-per-week model used by `ffsimulator` and most public simulators,
    which re-rolls availability every week. Both reproduce the marginal games-missed
    rate; only this one reproduces the *runs*. A roster's season is ruined by one back
    missing weeks 6 through 12, not by six teams each missing a different single week,
    and only the absorbing version ever generates the first shape.

    Duration is counted in GAMES, not calendar weeks, so an absence that straddles the
    bye comes back a week later than it started -- which is what actually happens.
    """

    hazard: Mapping[int, float] = field(default_factory=lambda: INJURY_HAZARD)
    absence_pmf: tuple[float, ...] = ABSENCE_PMF
    enabled: bool = True

    def __post_init__(self) -> None:
        total = float(sum(self.absence_pmf))
        if not self.absence_pmf or total <= 0:
            raise ValueError("absence_pmf must have positive mass")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"absence_pmf must sum to 1, got {total}")

    @classmethod
    def off(cls) -> InjuryModel:
        """No injuries, for isolating the marginal and copula behaviour.

        Turning injuries off does not change any other random number: the hazard draws
        are taken from each player's stream either way and simply never fire, so a draw
        with injuries off is the same season with everybody healthy rather than a
        different season.
        """
        return cls(enabled=False)

    @property
    def expected_absence(self) -> float:
        return float(sum((k + 1) * p for k, p in enumerate(self.absence_pmf)))

    def hazard_vector(self, position_ids: np.ndarray) -> np.ndarray:
        """Per-position hazard as an array. Unmodelled positions (K, D/ST) get zero."""
        out = np.zeros(len(position_ids), dtype=float)
        if not self.enabled:
            return out
        for i, pos in enumerate(position_ids):
            out[i] = self.hazard.get(int(pos), 0.0)
        return out

    def duration_cdf(self) -> np.ndarray:
        cdf = np.cumsum(np.asarray(self.absence_pmf, dtype=float))
        cdf[-1] = 1.0
        return cdf

    def expected_games_missed(self, position_id: int, n_games: int = 17) -> float:
        """Steady-state games missed in a season. The number to check against reality.

        A renewal cycle is the healthy run plus the absence. The week the hazard fires is
        the first week MISSED, not the last week played, so the healthy run is `(1-h)/h`
        games and not `1/h` -- the cycle is `(1-h)/h + E[D]` and the share missed is

            h*E[D] / (1 - h + h*E[D])

        At the measured RB hazard and a 4.3-game mean absence that is 3.2 of 17, which is
        what a starting back actually misses. Writing `1 + h*E[D]` for the denominator, as
        this did, describes a slightly different chain than the one `WeeklySampler` runs
        and understates it by 4%. A simulated season starts everyone healthy, so an
        observed run comes in under this either way -- 2.71 over eighteen weeks.
        """
        h = self.hazard.get(int(position_id), 0.0)
        share = h * self.expected_absence
        return n_games * share / (1.0 - h + share)


# --------------------------------------------------------------------------------------
# The compiled panel
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimPanel:
    """`WeeklyOutlook`s flattened into the `(week, player)` arrays the sampler wants.

    Built once per search over the UNION of every player any candidate might roster.
    That is not a performance nicety: two panels are two different random universes, so
    comparing a draw from one against a draw from the other throws away the variance
    reduction that the whole design exists for.

    A player-week with no outlook is not an error -- it is a player who is not on an NFL
    roster that week -- and it is compiled as "no game", which is distinct from a game
    he might blank in.
    """

    #: (P,) ESPN player ids, ascending. Ascending because the order fixes which base
    #: random stream mixes into which, and a stable order keeps that stable.
    player_ids: np.ndarray
    #: (W,) scoring period ids, ascending.
    weeks: np.ndarray
    #: (P,) defaultPositionId.
    position_ids: np.ndarray
    #: (W, P) proTeamId that week. Per-week rather than per-player because a traded
    #: player changes both his correlation block and his bye mid-season.
    pro_team_ids: np.ndarray
    #: (W, P) hurdle-gamma parameters.
    p_zero: np.ndarray
    shape: np.ndarray
    scale: np.ndarray
    #: (W, P) True when the player has an NFL game that week and is not already ruled
    #: out. Byes and known absences are folded in here, injuries are drawn later.
    has_game: np.ndarray
    #: (W, P) the stated moments, kept so a caller can assert the sampler recovers them.
    mean: np.ndarray
    sd: np.ndarray

    @property
    def n_players(self) -> int:
        return int(self.player_ids.size)

    @property
    def n_weeks(self) -> int:
        return int(self.weeks.size)

    def index_of(self, player_ids: Iterable[int]) -> np.ndarray:
        """Column indices for these players, in the order given."""
        wanted = np.asarray(list(player_ids), dtype=np.int64)
        pos = np.searchsorted(self.player_ids, wanted)
        pos = np.clip(pos, 0, self.n_players - 1)
        if self.n_players == 0 or not np.all(self.player_ids[pos] == wanted):
            missing = sorted(set(wanted.tolist()) - set(self.player_ids.tolist()))
            raise KeyError(f"players not in the panel: {missing[:10]}")
        return pos

    def week_index(self, week: int) -> int:
        idx = int(np.searchsorted(self.weeks, week))
        if idx >= self.n_weeks or int(self.weeks[idx]) != int(week):
            raise KeyError(f"week {week} is not in the panel")
        return idx

    @classmethod
    def from_outlooks(
        cls,
        outlooks: Iterable[WeeklyOutlook],
        *,
        weeks: Sequence[int] | None = None,
        byes: Mapping[int, int] | None = None,
    ) -> SimPanel:
        """Compile a flat stream of outlooks.

        `byes` maps proTeamId -> bye week; a player whose week-`w` team is on bye that
        week is compiled as having no game, which zeroes his points and makes him
        unstartable rather than merely likely to blank.
        """
        rows = list(outlooks)
        if not rows:
            raise ValueError("no outlooks to compile")

        players = sorted({o.player_id for o in rows})
        week_list = sorted({o.week for o in rows}) if weeks is None else sorted(set(weeks))
        p_index = {p: i for i, p in enumerate(players)}
        w_index = {w: i for i, w in enumerate(week_list)}
        n_w, n_p = len(week_list), len(players)

        position_ids = np.zeros(n_p, dtype=np.int64)
        pro_team_ids = np.zeros((n_w, n_p), dtype=np.int64)
        p_zero = np.ones((n_w, n_p))
        shape = np.ones((n_w, n_p))
        scale = np.zeros((n_w, n_p))
        has_game = np.zeros((n_w, n_p), dtype=bool)
        mean = np.zeros((n_w, n_p))
        sd = np.zeros((n_w, n_p))

        for o in rows:
            j = p_index[o.player_id]
            position_ids[j] = o.position_id
            wi = w_index.get(o.week)
            if wi is None:
                continue
            pro_team_ids[wi, j] = o.pro_team_id
            p_zero[wi, j] = min(max(o.p_zero, 0.0), 1.0)
            shape[wi, j] = _clean_shape(o.shape)
            scale[wi, j] = max(o.scale, 0.0)
            mean[wi, j] = o.mean
            sd[wi, j] = o.sd
            on_bye = byes is not None and byes.get(o.pro_team_id) == o.week
            has_game[wi, j] = bool(o.playing) and o.p_zero < 1.0 and not on_bye

        # A player carried into a week he has no outlook for keeps his position but has
        # no game; leaving the team id at 0 keeps him out of every correlation block.
        return cls(
            player_ids=np.asarray(players, dtype=np.int64),
            weeks=np.asarray(week_list, dtype=np.int64),
            position_ids=position_ids,
            pro_team_ids=pro_team_ids,
            p_zero=p_zero,
            shape=shape,
            scale=scale,
            has_game=has_game,
            mean=mean,
            sd=sd,
        )

    @classmethod
    def from_players(
        cls,
        players: Iterable[PlayerOutlook],
        *,
        weeks: Sequence[int] | None = None,
        byes: Mapping[int, int] | None = None,
    ) -> SimPanel:
        return cls.from_outlooks(
            [o for p in players for o in p.weeks.values()], weeks=weeks, byes=byes
        )


def _clean_shape(shape: float) -> float:
    """Clamp into the range the quantile machinery represents, both ends.

    The lower clamp is a numerical backstop, not a modelling choice: below
    `MIN_GAMMA_SHAPE` the quantile grid would have to span more than forty e-folds and
    the interpolation would coarsen for every other player in the panel to no purpose,
    since a shape that small is a point mass at zero to any accuracy anyone can measure.
    """
    if not np.isfinite(shape) or shape <= 0.0:
        return 1.0
    return float(min(max(shape, MIN_GAMMA_SHAPE), MAX_GAMMA_SHAPE))


def espn_bye_weeks(season: int, **kwargs: Any) -> dict[int, int]:
    """proTeamId -> bye week, from ESPN's own platform-settings table.

    Imported lazily: this pulls in the ESPN client, and the sampler must stay usable
    with nothing but numpy in the room.
    """
    from ..espn.constants import load_platform_settings

    settings = load_platform_settings(season, **kwargs)
    return {t.id.value: t.bye_week for t in settings.pro_teams if t.bye_week}


# --------------------------------------------------------------------------------------
# The draw
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Draw:
    """One `[sim, week, player]` realization of a season, plus who was available.

    `points` and `available` carry the same shape and are both needed: a zero in
    `points` where `available` is True is a player who suited up and did nothing, which
    a lineup can still be stuck with, while a zero where `available` is False is a bye
    or an injury and the slot has to be filled by someone else.
    """

    panel: SimPanel
    seed: int
    n_sims: int
    #: (S, W, P)
    points: np.ndarray
    #: (S, W, P)
    available: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.points.shape  # type: ignore[return-value]

    def points_for(self, player_ids: Iterable[int]) -> np.ndarray:
        """(S, W, k) for these players, in the order given."""
        return self.points[:, :, self.panel.index_of(player_ids)]

    def available_for(self, player_ids: Iterable[int]) -> np.ndarray:
        return self.available[:, :, self.panel.index_of(player_ids)]

    def week(self, week: int) -> tuple[np.ndarray, np.ndarray]:
        """(points, available) for one scoring period, each (S, P)."""
        i = self.panel.week_index(week)
        return self.points[:, i, :], self.available[:, i, :]

    def totals(
        self, player_ids: Iterable[int], *, weeks: Sequence[int] | None = None
    ) -> np.ndarray:
        """(S,) total points these players score, summed over weeks. Explanatory only --
        a real lineup starts a subset, which is the caller's job."""
        cols = self.panel.index_of(player_ids)
        block = self.points[:, :, cols]
        if weeks is not None:
            rows = [self.panel.week_index(w) for w in weeks]
            block = block[:, rows, :]
        return block.sum(axis=(1, 2), dtype=np.float64)


def _stream_key(player_id: int) -> int:
    """Map a player id into the non-negative space SeedSequence requires.

    ESPN encodes team defenses as NEGATIVE ids -- `-16{proTeamId:03d}`, so the
    Chargers D/ST is -16024 -- and `SeedSequence` rejects a negative spawn_key
    outright. Every roster in a normal league starts a defense, so without this
    the sampler raises on every real league while passing every synthetic test.

    Zigzag rather than `abs`, because `abs(-16024) == 16024` is itself a live
    player id: two different entities would silently share one random stream,
    which is far worse than the crash it replaces. Zigzag is injective over all
    of Z, so distinct ids keep distinct streams.
    """
    return 2 * player_id if player_id >= 0 else -2 * player_id - 1


class WeeklySampler:
    """Turns a `SimPanel` into draws, and owns the common random numbers.

    Construct one per search, ask it for a draw, and hand that draw to every candidate.
    `draw` memoizes on `n_sims`, so this is hard to get wrong: two calls return the
    same object, and a candidate evaluator that indexes it cannot accidentally
    resample. If you genuinely want a fresh universe -- to put a standard error on a
    result rather than to compare two rosters -- build a second sampler with a
    different `seed`.
    """

    __slots__ = (
        "_blocks",
        "_cache",
        "_hazard",
        "_duration_cdf",
        "_shape_idx",
        "_shape_w",
        "correlation",
        "dtype",
        "injuries",
        "panel",
        "seed",
        "table",
    )

    def __init__(
        self,
        panel: SimPanel,
        *,
        seed: int = DEFAULT_SEED,
        correlation: CorrelationModel | None = None,
        injuries: InjuryModel | None = None,
        exact_quantiles: bool = False,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        self.panel = panel
        self.seed = int(seed)
        self.correlation = correlation if correlation is not None else CorrelationModel()
        self.injuries = injuries if injuries is not None else InjuryModel()
        self.dtype = np.dtype(dtype)
        self._cache: dict[int, Draw] = {}

        live = panel.shape[panel.has_game] if panel.has_game.any() else panel.shape
        unresolved = int(np.count_nonzero(live < _UNRESOLVED_SHAPE))
        if unresolved:
            log.warning(
                "%d of %d live player-weeks have a gamma shape under %.0e (a calibrated "
                "mean under about 0.005 points against a spread line that floors the SD "
                "near 2.7). Their MEAN is reproduced; their SD is not, and comes out low "
                "-- the stated width lives in a one-in-a-million draw of several thousand "
                "points that no finite sample and no float64 quantile can carry. They are "
                "not startable players, but do not read their variance",
                unresolved,
                live.size,
                _UNRESOLVED_SHAPE,
            )
        self.table = (
            None
            if exact_quantiles
            else GammaQuantileTable.build(float(live.min()), float(live.max()))
        )
        self._shape_idx, self._shape_w = (
            (None, None) if self.table is None else self.table.shape_index(panel.shape)
        )
        self._hazard = self.injuries.hazard_vector(panel.position_ids)
        self._duration_cdf = self.injuries.duration_cdf()
        self._blocks = self._build_blocks()

    # -- setup ---------------------------------------------------------------------

    def _build_blocks(self) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        """Per week, the correlated team blocks and their factors.

        Per week because a block is (team, week): a traded player joins a different
        offense, and the attenuation correction depends on that week's marginals. Each
        block is a handful of players, so this is ~600 tiny Choleskys, not one big one.
        """
        panel = self.panel
        model = self.correlation
        spectrum = self._spectrum()
        out: list[list[tuple[np.ndarray, np.ndarray]]] = []
        repaired = 0
        relaxed = 0
        worst = 0.0
        worst_core = 0.0
        for w in range(panel.n_weeks):
            teams = panel.pro_team_ids[w]
            week_blocks: list[tuple[np.ndarray, np.ndarray]] = []
            for team in np.unique(teams):
                if team <= 0:
                    continue
                idx = np.flatnonzero((teams == team) & panel.has_game[w])
                if idx.size < 2:
                    continue
                positions = panel.position_ids[idx]
                if not model.couples(positions.tolist()):
                    continue
                target = model.target_matrix(positions.tolist())
                if spectrum is not None:
                    g = spectrum[:, w, idx]
                    latent = solve_latent_correlation(
                        target, g[:, :, None] * g[:, None, :], max_latent=model.max_latent
                    )
                    np.fill_diagonal(latent, 1.0)
                    latent, t = relax_to_feasible(latent, target)
                    relaxed += int(t > 0.0)
                else:
                    latent = target
                block_sd = panel.sd[w, idx]
                factor, adjusted = nearest_correlation_factor(latent, weights=block_sd**2)
                if adjusted:
                    repaired += 1
                    # In POINTS space, not latent space. The repair moves the latent
                    # matrix, but `target` is a Pearson correlation between realized
                    # fantasy points, and on a block carrying 80% zero mass the two live
                    # a long way apart: on the fifteen-man depth chart in the tests the
                    # latent difference reads 0.053 where the realized loss is 0.258.
                    # Differencing them directly understated this warning fivefold.
                    repaired_corr = factor @ factor.T
                    if spectrum is not None:
                        g = spectrum[:, w, idx]
                        repaired_corr = pearson_from_latent(
                            repaired_corr, g[:, :, None] * g[:, None, :]
                        )
                    error = np.abs(repaired_corr - target)
                    np.fill_diagonal(error, 0.0)
                    worst = max(worst, float(error.max()))
                    # Reported separately because the whole design of the repair is that
                    # these two numbers differ: the error is deliberately pushed onto the
                    # low-variance pairs, and only the second number is a modelling loss.
                    core = block_sd >= np.median(block_sd)
                    if core.sum() >= 2:
                        worst_core = max(worst_core, float(error[np.ix_(core, core)].max()))
                week_blocks.append((idx, factor))
            out.append(week_blocks)
        total = sum(len(b) for b in out)
        if relaxed:
            log.info(
                "relaxed the marginal-transform correction on %d of %d team-week blocks; "
                "exact pairwise recovery for every fringe player at once is not a "
                "correlation matrix, so the fringe gives way and the starters do not",
                relaxed,
                total,
            )
        if repaired:
            log.warning(
                "projected %d of %d team-week correlation blocks onto the nearest "
                "positive-definite correlation matrix: realized Pearson correlation of "
                "the points off its measured rho by %.3f at worst overall, %.3f among "
                "the block's higher-variance half. The rhos are marginal estimates and "
                "are not jointly consistent once a block holds two quarterbacks and "
                "several receivers",
                repaired,
                total,
                worst,
                worst_core,
            )
        return out

    def _spectrum(self) -> np.ndarray | None:
        """(HERMITE_ORDER, W, P) copula spectra, or None when the rhos are read raw.

        Evaluated only on player-weeks with a game -- a bye carries no correlation and
        its `(p_zero, shape)` is a placeholder -- which is also most of the saving on a
        panel that spans a full season.
        """
        if not self.correlation.match_pearson:
            return None
        panel = self.panel
        spectrum = np.zeros((HERMITE_ORDER, panel.n_weeks, panel.n_players))
        spectrum[0] = 1.0
        live = panel.has_game
        if live.any():
            spectrum[:, live] = hermite_coefficients(
                panel.p_zero[live], panel.shape[live], table=self.table
            )
        return spectrum

    # -- drawing -------------------------------------------------------------------

    def draw(self, n_sims: int = DEFAULT_SIMS) -> Draw:
        """The tensor. Memoized, which is what makes common random numbers the default.

        Calling this twice with the same `n_sims` returns the identical object, so a
        candidate search that evaluates roster A and then roster B is automatically
        paired and differences out most of the Monte Carlo error.
        """
        n = int(n_sims)
        if n <= 0:
            raise ValueError(f"n_sims must be positive, got {n_sims}")
        cached = self._cache.get(n)
        if cached is None:
            cached = self._simulate(n)
            self._cache[n] = cached
        return cached

    def _streams(self) -> list[np.random.Generator]:
        """One generator per player, keyed on the player id rather than his column.

        This is the property that makes common random numbers survive a change of
        panel: a search that adds one free agent leaves every other player's season
        bit-identical, so the paired difference isolates the free agent instead of
        re-rolling the league.
        """
        return [
            np.random.Generator(
                np.random.PCG64(np.random.SeedSequence(self.seed, spawn_key=(_stream_key(pid),)))
            )
            for pid in self.panel.player_ids
        ]

    def _simulate(self, n_sims: int) -> Draw:
        panel = self.panel
        n_w, n_p = panel.n_weeks, panel.n_players
        points = np.zeros((n_sims, n_w, n_p), dtype=self.dtype)
        available = np.zeros((n_sims, n_w, n_p), dtype=bool)

        gens = self._streams()
        base = np.empty((n_p, n_sims))
        injury_u = np.empty((2, n_p, n_sims))
        # Games still to be missed. Absorbing: only a week with a game decrements it.
        remaining = np.zeros((n_p, n_sims), dtype=np.int16)
        cdf = self._duration_cdf
        hazard = self._hazard[:, None]

        for w in range(n_w):
            for j, g in enumerate(gens):
                base[j] = g.standard_normal(n_sims)
                injury_u[:, j, :] = g.random((2, n_sims))

            z = base.copy()
            for idx, factor in self._blocks[w]:
                z[idx] = factor @ base[idx]

            u = ndtr(z)
            p = np.minimum(panel.p_zero[w], 1.0 - 1e-12)[:, None]
            q = np.clip((u - p) / (1.0 - p), 1e-12, 1.0 - 1e-12)
            if self.table is None:
                x = np.exp(log_gamma_quantile(panel.shape[w][:, None], q))
            else:
                x = self.table.lookup(
                    self._shape_idx[w][:, None], self._shape_w[w][:, None], ndtri(q)
                )
            x *= panel.scale[w][:, None]
            x[u <= p] = 0.0

            plays = panel.has_game[w][:, None]
            fires = plays & (remaining <= 0) & (injury_u[0] < hazard)
            hit = np.flatnonzero(fires.ravel())
            if hit.size:
                draws = np.searchsorted(cdf, injury_u[1].ravel()[hit]) + 1
                np.put(remaining, hit, draws.astype(np.int16))

            out_now = remaining > 0
            usable = plays & ~out_now
            x[~usable] = 0.0
            points[:, w, :] = x.T.astype(self.dtype, copy=False)
            available[:, w, :] = usable.T
            # Only a game the player was absent for burns a game off the clock.
            remaining -= plays & out_now

        return Draw(panel=panel, seed=self.seed, n_sims=n_sims, points=points, available=available)

    # -- construction shortcuts ------------------------------------------------------

    @classmethod
    def from_outlooks(
        cls,
        outlooks: Iterable[WeeklyOutlook],
        *,
        weeks: Sequence[int] | None = None,
        byes: Mapping[int, int] | None = None,
        **kwargs: Any,
    ) -> WeeklySampler:
        return cls(SimPanel.from_outlooks(outlooks, weeks=weeks, byes=byes), **kwargs)

    @classmethod
    def from_players(
        cls,
        players: Iterable[PlayerOutlook],
        *,
        weeks: Sequence[int] | None = None,
        byes: Mapping[int, int] | None = None,
        **kwargs: Any,
    ) -> WeeklySampler:
        return cls(SimPanel.from_players(players, weeks=weeks, byes=byes), **kwargs)


def common_random_draw(
    outlooks: Iterable[WeeklyOutlook],
    *,
    n_sims: int = DEFAULT_SIMS,
    seed: int = DEFAULT_SEED,
    weeks: Sequence[int] | None = None,
    byes: Mapping[int, int] | None = None,
    **kwargs: Any,
) -> Draw:
    """One draw over one panel, for callers that do not need to keep the sampler.

    Keep the `Draw` and pass it around. Calling this twice is two universes, and
    comparing across them is exactly the mistake common random numbers exist to
    prevent.
    """
    sampler = WeeklySampler.from_outlooks(outlooks, weeks=weeks, byes=byes, seed=seed, **kwargs)
    return sampler.draw(n_sims)
