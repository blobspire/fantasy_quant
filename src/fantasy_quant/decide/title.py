"""Delta-P(championship): the engine every other decision surface is priced in.

A waiver claim in a 14-team full-PPR league and a trade in a 12-team half-PPR one are
comparable only because both come back as a change in championship probability. That
number comes from here, and it is expensive: a title probability is a bracket laid on
top of a season, so the Monte Carlo error on a *difference* of two title probabilities
stays around 0.55-0.63pp at 4,000 paired simulations even with common random numbers
(measured on the user's three leagues). Pricing a thousand candidates that way is a
minute and a half of compute for an answer that is mostly noise.

So this is two tiers, and the split is the whole design. Measured on the real leagues at
4,000 simulations: **2.7-5.8ms a candidate to screen against 73-80ms to confirm**, a
factor of **13 to 15**. (An earlier draft of this docstring claimed 0.95ms and a factor
of 90; re-measured over 200-3,344 candidates on each league it does not reproduce, and
since the factor is the entire argument for having two tiers it is worth being right
about. The screen is still an order of magnitude cheaper, which is enough.)

**An unfilled slot streams a replacement.** This engine's floors default to
`streaming_replacement`, not to an empty seat, and the difference is the difference
between a board worth reading and one that is not. With empty seats -- which is what
`sim/season` does when nobody passes a `replacement`, and what this module did -- the
user's own rosters priced Harrison Butker at -4.15pp of title probability against Justin
Jefferson's -3.50pp, and Evan McPherson at -6.53pp against Jonathan Taylor's -5.67pp. A
kicker cannot be the second most valuable asset on a roster. With the level fitted, every
kicker and defence on all three teams falls under a point and the first-round backs are
back at the top, which is also what `docs/RESEARCH.md` means when it says QB carries only
4.8% of positive VORP -- the marginal quarterback is nearly free, so losing one costs
little.

**Tier 1, `screen`.** A surrogate response surface `P_title = g(mu, sigma)` in the
team's own weekly scoring mean and standard deviation, fitted once per team (0.5s) by
perturbing that team's weekly scores over a 31 x 7 grid and re-running only the cheap
half of the simulator: standings, seeds and bracket, with the lineups left alone. A
candidate is then priced by an order-statistic calculation on the affected lineup slots,
which yields `(d_mu, d_sigma)`, and read off the surface. No simulation per candidate.

It is good enough to be read as a number, not just as a ranking key. Against `confirm`
over 200 one-for-one candidates on each real league: rank correlation 0.891 / 0.937 /
0.934, and -- the part that is easy to get wrong and easy to check -- a regression slope
of confirm on screen of **0.95 / 0.82 / 0.95**, so the screened percentage point is
roughly the size of the confirmed one. Recall of confirm's true top ten inside the
screen's top forty is 1.00 on all three, which is the number that decides whether the
two-tier design works. `agreement()` recomputes all of it, and a caller who does not
trust the screen on a new league should run it rather than assume.

The screen's own error bar is not the surface's RMSE alone. Regressing the actual
disagreement on the size of the effect gives roughly 0.25pp plus a fifth of the effect,
so `stderr` is reported as the two in quadrature; on the RMSE alone a +13pp screened
trade would claim an accuracy of a third of a point.

**Tier 2, `confirm`.** The survivors are re-simulated against the *same* pre-drawn
tensor and the *same* schedule, re-solving lineups only for the franchises the move
touches -- `sim/season.leave_one_out` is the pattern. The reported `stderr` is the
standard error of the PAIRED per-simulation difference, not of either arm.

**How much common random numbers are actually worth here, which is less than
`sim/season.py` implies, and the difference is structural rather than a disagreement.**
That module measures 1000x on points-for, 27x on wins and 5x on the championship
indicator by *scaling one player's realisations by 15%* -- a shift of the same football.
Reproduced here under the same protocol: 513x / 32x / 3.8x. But every candidate this
engine prices is a **substitution**, and the player coming in is a different random
variable whose own week-to-week variance no amount of pairing can remove. On a real
one-for-one trade in the user's leagues the same machinery gives 7-8x on points-for,
3.9-4.3x on wins and only **1.19-1.50x on the title indicator** -- a paired standard
error 1.09 to 1.22 times smaller than independent arms, not two or three times. Budget
simulations off the substitution numbers. `crn_variance_ratio` reports them per move, and
the pairing is still exact where it matters most: a move that changes no roster comes
back as 0.0 with zero error against an independent arm's 0.6pp.

**The failure mode this module exists to avoid: dP/dsigma flips sign at the cut.**
A team safely inside the playoff field wants less week-to-week variance; a team well
below the line wants more, because its only route to the bracket is a good tail. One
global surface fitted across teams would average those two populations together and
systematically mis-screen every variance-changing move for teams *near the cut*, which
is exactly the population a recommendation matters for. So a surface is fitted per team,
from that team's own perturbed scores, which conditions it on standing by construction;
and it is fitted off the current `LeagueState`, whose record is already reduced to facts,
so refitting as the season moves needs nothing but a rebuilt engine.

The flip is real and it is measured, but **it does not happen at the 50% cut**, which is
where the research note puts it. Sweeping `dP(playoffs)/dsigma` across all thirty-eight
franchises in the user's three leagues, the crossing sits at **35-40% playoff
probability**: at 23-34% odds the derivative is +4 to +9pp per unit of relative SD, by
42% it is already negative, and a 67% team is at -15 to -20. So a team on 45% playoff
odds should be *shedding* variance even though it is below the line -- the opposite of
what the naive rule says, and a real recommendation for two of the user's three teams.
The mechanism is that raising your own weekly SD also widens every opponent's margin
against you, which pulls the whole table toward the middle and helps the teams behind you
more than it helps you. `SurrogateFit.wants_variance` answers it per team rather than
from a rule of thumb.

**Where the order statistic actually lives, which is not where the research note put
it.** The note prescribes `d_mu ~= E[max(X_new, X_incumbent)] - E[X_incumbent]` over
the players' point distributions. That prices *hindsight*: `sim/season.py` sets lineups
ex ante on projected means, so a manager does not get the better realisation of two
players, he gets the one he started. The max that is really there is over
*availability*: the incumbent is out in some simulations and the newcomer starts then
instead. So the slot expectation here is the sequential order statistic

    E[slot] = a_1 mu_1 + (1-a_1)[ a_2 mu_2 + (1-a_2)[ ... + (1-a_k) f ] ]

down the ex-ante depth chart, with `a_i` the probability that player is available (read
straight off the drawn tensor, so byes and the absorbing injury model are both in it)
and `f` the slot's free-agent floor. It is an order-statistic approximation on the
affected slot, just of the *ranking* rather than of the outcome.

`screen(..., hindsight_max=True)` computes the research version so the two can be
measured instead of argued about. Re-measured on 200 one-for-one candidates in each real
league, with Clark's formula implemented in full -- the earlier figures here were taken
against a version missing its `Phi` term, which inflated the bonus 3.5-fold and so
flattered the comparison -- the rank correlation against `confirm` is **0.891 / 0.937 /
0.934** for the ex-ante order statistic against **0.851 / 0.834 / 0.916** for Clark's.
The ex-ante statistic wins on all three, and by more than the broken version suggested.
It also wins on the level, which is the part that never cancels: Clark's bonus pays a
bench for information nobody has on Sunday morning.

**What tier 1 cannot see, stated rather than hidden.** The surface is a function of one
team's own mu and sigma, so a trade is priced as "my roster got better" plus a
renormalisation for the counterparty's own gain (title probabilities sum to one, so a
rival's gain comes out of the rest of the field in proportion to their standing). It
cannot see schedule interactions -- that a rival's improvement lands on the team you
need to miss the playoffs. And the perturbation is uniform across remaining weeks, so a
player who only helps in weeks 15-17 screens as if he helped all year. All are
second-order, all are exactly what tier 2 gets right, and all are why `evaluate()`
confirms the survivors rather than shipping the screen.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..core import (
    Move,
    MoveKind,
    PlayerMove,
    PlayerOutlook,
    Recommendation,
    WeeklyOutlook,
    WireLevel,
    leverage,
    points_to_win_prob,
)
from ..sim import season as S
from ..sim.distributions import Draw
from ..sim.lineup import LineupPlan, plan_from_slots
from .wire import DEFAULT_WIRE_DEPTH, all_rostered, wire_levels

log = logging.getLogger(__name__)

#: Perturbation grid for the surrogate: +/-15 projected points a week in 2-point steps,
#: and +/-30% on the weekly standard deviation. The mu range covers every realistic
#: single-move swing (the best-minus-worst starter gap over a season is 225 points at WR,
#: about 13 a week) and the sigma range covers a whole starter's worth of variance.
#:
#: The mu step was 1 and is 2, which pays for the playoff axis below and costs nothing:
#: measured side by side on the fixture, 16 nodes reproduce 31 nodes to `rmse` 0.00314 vs
#: 0.00322 and `d_title_d_mu` to four significant figures. A quadratic in the logit does
#: not need 31 samples of its own smooth curve.
DEFAULT_MU_GRID: tuple[float, ...] = tuple(float(x) for x in range(-15, 16, 2))
DEFAULT_SIGMA_GRID: tuple[float, ...] = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30)

#: EXTRA points a week in the bracket weeks only, on top of `DEFAULT_MU_GRID`. This is
#: the axis the surface did not have, and its absence is why a screen priced a week-16
#: point exactly like a week-3 point.
#:
#: Measured on the `tests/test_title.py` fixture at 4,000 simulations, moving the SAME
#: total points around the calendar: +1.83pp spread evenly, +2.92pp concentrated in the
#: bracket, +1.25pp concentrated in the regular season. The old two-axis surface answered
#: +1.83pp to all three. Five nodes is enough for a quadratic plus its two interactions.
DEFAULT_PLAYOFF_GRID: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)

#: Simulations used to fit the surface. The fit is over ~200 nodes that share one draw,
#: so the node-to-node noise is highly correlated and the surface smooths what is left;
#: spending the full tensor here buys precision the fit cannot use.
DEFAULT_SURROGATE_SIMS = 2000

#: How many bench bodies behind a slot's starter are priced. Depth beyond the second
#: backup is worth less than the free-agent floor it competes with.
BENCH_DEPTH = 2

#: Candidates handed to tier 2 by `evaluate`.
DEFAULT_CONFIRM = 50

#: `eta` is clipped here before the logistic. exp(30) is already 1e13 to 1.
_LOGIT_CLIP = 30.0

#: Relative modelling error of the screen against `confirm`, measured by regressing
#: `|screen - confirm|` on `|confirm|` over 200 one-for-one candidates in each of the
#: user's three leagues: slopes 0.219 / 0.018 / 0.033 on intercepts of 0.24-0.31pp. The
#: intercept is what the surface's own RMSE already covers; the slope is not, and
#: without it a +12pp screened trade reports +/- 0.3pp when it is really +/- 2pp. The
#: worst of the three is used, because a screen that under-reports its error on one
#: league out of three is a screen that under-reports its error.
SCREEN_RELATIVE_ERROR = 0.22

#: Standardisation for the design matrix. Each is a half-width of its default grid, so
#: the quadratic terms stay order 1 and the normal equations stay well conditioned.
_MU_SCALE = 15.0
_SIGMA_SCALE = 0.30
_PLAYOFF_SCALE = 15.0

#: Column layout of `_design`. The playoff terms are APPENDED rather than interleaved, so
#: indices 0-5 keep the meaning they have always had -- `beta[2]` is still the sigma
#: coefficient the cut-line diagnostic reads, and a caller that hand-indexes the vector
#: does not silently start reading a different number.
_TERMS = 10


class TitleError(ValueError):
    """A candidate cannot be priced against this league as posed."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic, written so a large negative `eta` underflows to 0 rather than to inf."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(x, -_LOGIT_CLIP, _LOGIT_CLIP)))


#: Standard normal CDF over an array. numpy has no `erf`, and the only caller is the
#: hindsight diagnostic -- a few hundred elements -- so the loop costs nothing and the
#: alternative is pulling `scipy.special` in beside a hot path for one term.
_ERF = np.vectorize(math.erf, otypes=[float])


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def clark_bonus(mean: np.ndarray, sd: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """`E[max(X, Y)] - E[X]` for independent normals of equal spread. Clark, 1961.

    `X` is the starter, `Y` the notional replacement at the slot's floor. Both terms of

        E[max] - E[X] = theta * phi(alpha) + (mu_y - mu_x) * Phi(-alpha)

    with `theta = sqrt(sx^2 + sy^2)` and `alpha = (mu_x - mu_y) / theta`. Written out as
    its own function so the brute-force test exercises the code the engine runs rather
    than a copy of the formula in the test file; keeping the second term only in a
    comment is how it went missing in the first place, which left the bonus 3.5 times too
    large (1.89 against 0.54 at mu 12, sd 7, floor 0).
    """
    theta = np.sqrt(2.0) * np.maximum(np.asarray(sd, dtype=float), 1e-9)
    alpha = (np.asarray(mean, dtype=float) - floor) / theta
    phi = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    return theta * phi + (floor - mean) * _norm_cdf(-alpha)


# --------------------------------------------------------------------------------------
# The surrogate
# --------------------------------------------------------------------------------------


def _design(
    d_mu: np.ndarray | float,
    sigma_ratio: np.ndarray | float,
    d_mu_playoff: np.ndarray | float = 0.0,
) -> np.ndarray:
    """`[1, x, s, x^2, xs, s^2, p, p^2, xp, sp]` in standardised coordinates.

    A quadratic on the *logit* scale rather than a GAM or a spline, for three reasons
    that are all about what the surface is used for. It has ten parameters against ~560
    nodes, so it cannot chase the Monte Carlo noise the nodes carry. Its derivatives are
    closed form, and `dP/dsigma` at the origin is the number the cut-line diagnostic
    actually reports -- a smoother would have to be differenced numerically and would
    inherit the node noise doing it. And a logit link keeps every prediction inside
    (0, 1) without clipping, which matters because half the teams in a 14-team league sit
    under 5% and a linear surface walks them negative at the bottom of the mu grid.

    **`p` is extra points a week in the BRACKET weeks only**, on top of the `x` that
    applies everywhere. Without it the surface has no way to represent "+3 points in week
    16" and prices it exactly like "+3 points in week 3" -- which is what it did, and
    which is worth a factor of 2.3 on the fixture and about 4.5 on the live leagues. The
    module docstring conceded this ("a player who only helps in weeks 15-17 screens as if
    he helped all year") and `docs/PLAN.md` specified the fix that was never built.

    The playoff terms are appended LAST on purpose: indices 0-5 keep the meaning they
    have always had, so `beta[2]` is still the sigma coefficient and a caller that hand-
    indexes the vector -- the pooled-surface test does -- does not silently start reading
    a different number.

    Measured on the user's three leagues: RMSE 0.27-0.55pp on the title surface and
    0.70-0.86pp on the playoff surface, against baseline title probabilities of 2-14%.
    That is the number `screen` reports as its own `stderr`, so a screened effect the
    surface cannot resolve is labelled insignificant rather than dressed up.
    """
    x = np.asarray(d_mu, dtype=float) / _MU_SCALE
    s = (np.asarray(sigma_ratio, dtype=float) - 1.0) / _SIGMA_SCALE
    p = np.asarray(d_mu_playoff, dtype=float) / _PLAYOFF_SCALE
    x, s, p = np.broadcast_arrays(x, s, p)
    one = np.ones_like(x)
    return np.stack([one, x, s, x * x, x * s, s * s, p, p * p, x * p, s * p], axis=-1)


def playoff_mask(state: S.LeagueState) -> np.ndarray:
    """`(weeks,)` 1.0 on a bracket week, 0.0 elsewhere. The surrogate's third axis.

    Flattened from `state.playoff_rounds`, which `decide/title.py` referenced zero times
    before this. `trades.py` has had the same idiom since it was written.
    """
    bracket = {w for rnd in state.playoff_rounds for w in rnd}
    return np.array([1.0 if w in bracket else 0.0 for w in state.weeks], dtype=np.float64)


def split_by_bracket(delta: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """A per-week delta -> `(points a week everywhere, EXTRA a week in the bracket)`.

    The decomposition the surface is fitted on: `delta_w ~= x + p * mask_w`. Least
    squares over a two-level factor is just the two block means, so `x` is the regular-
    season mean and `p` is how much better the bracket weeks are than that.

    Both degenerate cases give `p = 0`, which is right rather than merely safe: with no
    bracket weeks left there is no tilt to price, and with nothing BUT bracket weeks left
    every point is a playoff point and `x` already carries it.
    """
    delta = np.asarray(delta, dtype=np.float64)
    bracket = mask > 0.0
    if not bracket.any() or bracket.all():
        return float(delta.mean()) if delta.size else 0.0, 0.0
    regular = float(delta[~bracket].mean())
    return regular, float(delta[bracket].mean()) - regular


def _fit_binomial_surface(
    d_mu: np.ndarray,
    sigma_ratio: np.ndarray,
    successes: np.ndarray,
    trials: int,
    *,
    d_mu_playoff: np.ndarray | float = 0.0,
    ridge: float = 1e-6,
    iterations: int = 64,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """IRLS for a binomial logit over the grid. Returns `(coefficients, RMSE)`.

    The half-count continuity correction is not cosmetic. A 14-team league's worst team
    wins zero titles in 2,000 simulations across the whole bottom half of the mu grid,
    and a plain binomial likelihood answers that with `eta -> -inf`: the fit diverges,
    numpy hands back a singular-matrix warning or a wall of `nan`, and the screen
    silently ranks every candidate for that team at zero. Fitting `(k + 1/2)` out of
    `(n + 1)` is the Jeffreys posterior mean, keeps every node interior, and moves a
    node with real mass by under 0.03pp.
    """
    design = _design(d_mu, sigma_ratio, d_mu_playoff).reshape(-1, _TERMS)
    k = np.asarray(successes, dtype=float).ravel() + 0.5
    n = float(trials) + 1.0
    observed = np.asarray(successes, dtype=float).ravel() / float(trials)

    eye = np.eye(_TERMS) * ridge
    start = np.log(k / (n - k))
    beta = np.linalg.solve(design.T @ design + eye, design.T @ start)
    for _ in range(iterations):
        eta = np.clip(design @ beta, -_LOGIT_CLIP, _LOGIT_CLIP)
        p = _sigmoid(eta)
        w = np.maximum(n * p * (1.0 - p), 1e-9)
        z = eta + (k - n * p) / w
        weighted = design * w[:, None]
        step = np.linalg.solve(weighted.T @ design + eye, weighted.T @ z)
        moved = float(np.max(np.abs(step - beta)))
        beta = step
        if moved < tolerance:
            break
    fitted = _sigmoid(design @ beta)
    return beta, float(np.sqrt(np.mean((fitted - observed) ** 2)))


@dataclass(frozen=True, slots=True, eq=False)
class SurrogateFit:
    """`P = g(mu + d_mu, sigma * ratio)` for ONE team, fitted at its current standing.

    Two surfaces come out of the same grid because one run of the cheap season model
    reports both: the title probability, which is what everything is denominated in, and
    the playoff probability, which is where the cut-line sign flip is cleanest and is
    therefore the honest diagnostic to show a user.

    Every number a caller reads off this is a *difference* between two points on the
    surface, never a level, so the fit's own level error cancels. `rmse` is kept so the
    caller can put an honest error bar on a screened recommendation instead of reporting
    a bare point estimate with `stderr = 0`.
    """

    team_id: int
    #: Baseline weekly scoring moments this surface is centred on.
    mu: float
    sigma: float
    baseline_title: float
    baseline_playoffs: float
    coefficients: np.ndarray
    playoff_coefficients: np.ndarray
    rmse: float
    playoff_rmse: float
    mu_grid: tuple[float, ...]
    sigma_grid: tuple[float, ...]
    n_sims: int
    #: `(len(mu_grid), len(sigma_grid), len(playoff_grid))` simulated probabilities at
    #: each node, kept so a caller can refit, plot, or -- as the tests do -- pool two
    #: teams' nodes and show that a single surface across standings must mis-sign one.
    title_nodes: np.ndarray
    playoff_nodes: np.ndarray
    #: Extra points a week in the bracket weeks only. Empty on a league with no bracket
    #: left, where the axis is degenerate and every node would be a duplicate.
    playoff_grid: tuple[float, ...] = ()

    # -- reading the surface ---------------------------------------------------------

    def title(
        self, d_mu: float = 0.0, sigma_ratio: float = 1.0, d_mu_playoff: float = 0.0
    ) -> float:
        return float(_sigmoid(_design(d_mu, sigma_ratio, d_mu_playoff) @ self.coefficients))

    def playoffs(
        self, d_mu: float = 0.0, sigma_ratio: float = 1.0, d_mu_playoff: float = 0.0
    ) -> float:
        return float(
            _sigmoid(_design(d_mu, sigma_ratio, d_mu_playoff) @ self.playoff_coefficients)
        )

    def delta_title(
        self, d_mu: float, sigma_ratio: float = 1.0, d_mu_playoff: float = 0.0
    ) -> float:
        """The screened effect: the surface at the candidate minus the surface at zero.

        Differenced on the surface rather than against the simulated baseline, so the
        fit's level error -- which is common to both points -- drops out exactly.
        """
        return self.title(d_mu, sigma_ratio, d_mu_playoff) - self.title(0.0, 1.0, 0.0)

    def delta_playoffs(
        self, d_mu: float, sigma_ratio: float = 1.0, d_mu_playoff: float = 0.0
    ) -> float:
        return self.playoffs(d_mu, sigma_ratio, d_mu_playoff) - self.playoffs(0.0, 1.0, 0.0)

    # -- derivatives, which is what the cut-line diagnostic reads ---------------------

    def _slope(
        self,
        beta: np.ndarray,
        d_mu: float,
        sigma_ratio: float,
        term: int,
        d_mu_playoff: float = 0.0,
    ) -> float:
        """d(probability)/d(one raw axis), by the chain rule through the logit.

        Written as a gradient over the standardised coordinates rather than as a pair of
        hand-expanded branches, because the design now carries three axes and four cross
        terms and an expansion that missed one would be wrong in a way nothing catches:
        `d/ds` alone gained `beta[9] * p`.
        """
        prob = float(_sigmoid(_design(d_mu, sigma_ratio, d_mu_playoff) @ beta))
        x = d_mu / _MU_SCALE
        sig = (sigma_ratio - 1.0) / _SIGMA_SCALE
        pl = d_mu_playoff / _PLAYOFF_SCALE
        inner, scale = {
            1: (beta[1] + 2.0 * beta[3] * x + beta[4] * sig + beta[8] * pl, _MU_SCALE),
            2: (beta[2] + beta[4] * x + 2.0 * beta[5] * sig + beta[9] * pl, _SIGMA_SCALE),
            3: (beta[6] + 2.0 * beta[7] * pl + beta[8] * x + beta[9] * sig, _PLAYOFF_SCALE),
        }[term]
        return prob * (1.0 - prob) * inner / scale

    def d_title_d_mu(self, d_mu: float = 0.0, sigma_ratio: float = 1.0) -> float:
        """Title probability gained per projected point a week. The screen's leverage."""
        return self._slope(self.coefficients, d_mu, sigma_ratio, 1)

    def d_title_d_playoff_mu(self, d_mu: float = 0.0, sigma_ratio: float = 1.0) -> float:
        """Title probability gained per EXTRA point a week in the bracket.

        Read against `d_title_d_mu`: the ratio is how much more a playoff point is worth
        than an ordinary one, measured for this team at this standing rather than assumed.
        `decide/trades.PLAYOFF_WEIGHT` is a hand-set 1.2 for the same quantity.
        """
        return self._slope(self.coefficients, d_mu, sigma_ratio, 3)

    def d_title_d_sigma(self, d_mu: float = 0.0, sigma_ratio: float = 1.0) -> float:
        """dP(title) per unit of *relative* weekly SD. Negative for a favourite."""
        return self._slope(self.coefficients, d_mu, sigma_ratio, 2)

    def d_playoffs_d_sigma(self, d_mu: float = 0.0, sigma_ratio: float = 1.0) -> float:
        """dP(playoffs) per unit of relative weekly SD. This is what flips at the cut."""
        return self._slope(self.playoff_coefficients, d_mu, sigma_ratio, 2)

    @property
    def playoff_premium(self) -> float:
        """How much more a bracket point is worth than an ordinary one, for this team.

        `1 + dP/dp / dP/dx`: a point in the bracket earns `dP/dx` like every other week
        PLUS `dP/dp` on top, so the ratio is the premium. Measured, not assumed -- and it
        is not small. On the fixture the same total points are worth +2.92pp concentrated
        in the bracket against +1.25pp concentrated in the regular season.

        1.0 when the axis is degenerate (no bracket left, or nothing but bracket), which
        is the honest answer there: with no regular-season week to be better than, there
        is no premium to measure.
        """
        ordinary = self.d_title_d_mu()
        if not self.playoff_grid or len(self.playoff_grid) < 2 or abs(ordinary) < 1e-12:
            return 1.0
        return 1.0 + self.d_title_d_playoff_mu() / ordinary

    @property
    def wants_variance(self) -> bool:
        """True when this team should be buying variance rather than selling it.

        Read off the playoff surface, not the title surface: below the cut the title is
        near zero and its derivative is numerically small, while the playoff derivative
        is unambiguous. This is the number a start/sit surface should be conditioning on
        in weeks 10-14.
        """
        return self.d_playoffs_d_sigma() > 0.0


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamMoments:
    """One franchise's weekly scoring distribution under the baseline roster."""

    team_id: int
    name: str
    weeks: tuple[int, ...]
    mean_by_week: tuple[float, ...]
    sd_by_week: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.mean_by_week)) if self.mean_by_week else 0.0

    @property
    def sd(self) -> float:
        """Root-mean-square weekly SD -- the scale the surrogate's sigma axis moves."""
        return float(np.sqrt(np.mean(np.square(self.sd_by_week)))) if self.sd_by_week else 0.0


@dataclass(frozen=True, slots=True)
class WeekLeverage:
    """Whether a decision in one matchup can move anything at all.

    `leverage` is `core.leverage`: the marginal win probability per point relative to its
    value in a coin flip. Near zero -- a blowout either way -- means the week is decided
    and the tool's most valuable output is to say so.
    """

    week: int
    matchup_period: int
    team_id: int
    opponent_id: int
    margin: float
    sd_diff: float
    win_probability: float
    leverage: float

    @property
    def points_per_win_pct(self) -> float:
        """Win probability gained per additional projected point, in percentage points.

        At an even matchup and the measured league `sd_diff` of 34.4 this is 1.16pp,
        which is the constant every start/sit conversation should be anchored on.
        """
        return 100.0 * points_to_win_prob(self.sd_diff, self.margin)

    @property
    def decided(self) -> bool:
        """Under a quarter of a coin flip's marginal value: this week is already over."""
        return self.leverage < 0.25


@dataclass(frozen=True, slots=True)
class ScreenAgreement:
    """How good a filter tier 1 actually is. Reported, never assumed.

    `recall_at` is the number that decides whether the two-tier design works: of the
    candidates `confirm` ranks best, how many did `screen` put in the shortlist it
    forwards? A screen with a mediocre rank correlation but perfect recall is fine; one
    with a good correlation that drops the true best move is not.
    """

    n: int
    spearman: float
    pearson: float
    #: Least-squares slope of confirm on screen through the origin. 1.0 means tier 1 is
    #: calibrated in level as well as in order; anything else means a screened
    #: `delta_title` may be read as a ranking key but not as a number.
    scale: float
    shortlist: int
    top_k: int
    recall_at: float
    screen_seconds: float
    confirm_seconds: float

    @property
    def per_screen_ms(self) -> float:
        return 1000.0 * self.screen_seconds / self.n if self.n else 0.0

    @property
    def per_confirm_ms(self) -> float:
        return 1000.0 * self.confirm_seconds / self.n if self.n else 0.0


def streaming_replacement(
    state: S.LeagueState,
    draw: Draw,
    *,
    wire_depth: int = DEFAULT_WIRE_DEPTH,
    outlooks: Sequence[PlayerOutlook] | None = None,
) -> dict[int, float]:
    """What an unfilled starting slot streams off the wire, per slot id.

    **This is the difference between a recommendation and a joke, and it is not
    optional.** `sim/season._floors` prices an unfilled slot at zero when no
    replacement level is supplied, which asks "what if this seat were empty for the
    rest of the season" -- a question nobody faces, because the wire always has a
    kicker. With an empty seat, dropping Harrison Butker cost 4.15pp of title
    probability against Justin Jefferson's 3.50pp. A surface that ranks a kicker
    above a first-round back is not one anyone should act on.

    Delegates to `decide.wire.wire_floor`, which owns the definition. This function
    used to carry a second, worse one -- the VOLS demand rank, which reads the bottom
    of a ROSTER rather than the top of the wire and so priced every bench receiver at
    exactly zero. See `decide/wire.py` for what that cost.

    Two pools, and which one is available decides the answer:

    * The panel covers free agents (`sim_with_free_agents`): the floor is the real
      wire, which is the number that matters and the one this exists to give.
    * The panel holds only rostered players (`pipeline.build` pools that way, and it
      is what `championship_table`, the API and the dashboard all run on): there is
      no wire to read, so fall back to the VOLS demand rank. That reads the bottom
      of a roster and is too high -- measured at WR28 8.90/wk against a true
      best-available WR55 of 6.48 -- but "too high" beats the alternative. Returning
      zero here would price an unfilled slot as empty for the season, which is the
      bug that made dropping a kicker cost more title probability than dropping
      Justin Jefferson.
    """
    return {
        slot: level.mean
        for slot, level in streaming_levels(
            state, draw, wire_depth=wire_depth, outlooks=outlooks
        ).items()
    }


def streaming_levels(
    state: S.LeagueState,
    draw: Draw,
    *,
    wire_depth: int = DEFAULT_WIRE_DEPTH,
    outlooks: Sequence[PlayerOutlook] | None = None,
) -> dict[int, WireLevel]:
    """`streaming_replacement`, plus how much that level VARIES.

    The mean is identical to `streaming_replacement` -- verified equal at every slot --
    so a caller that only solves a lineup can keep using the float form. This one is for
    the caller that has to PAY an empty seat, because paying it the mean gives the seat
    zero variance, and on a live league 15.7% of slot-weeks are paid that way.

    When the panel holds no wire there is no body to read a spread off either. The VOLS
    rank still gives a defensible LEVEL, so the seat keeps its floor and stays
    deterministic rather than being handed an invented spread.
    """
    # `outlooks` is the WIRE. Without it this reads the panel, and on every production
    # path `pipeline.build` pools only rostered players -- so the wire came back empty at
    # every slot and fell through to the VOLS roster-bottom rank, which is the precise
    # bug `decide/wire.py` was extracted to eliminate. Measured on the live leagues that
    # put the RB floor at 9.22 against a true 4.55, and ranked Harrison Butker as a more
    # costly drop than Luther Burden III. Callers holding `sim.outlooks` must pass it.
    pool = list(outlooks) if outlooks is not None else _outlooks_from_panel(draw, state)
    levels = wire_levels(
        pool, all_rostered(state), state.weeks, state.slot_eligibility, depth=wire_depth
    )
    missing = [slot for slot, v in levels.items() if v.mean <= 0.0]
    if missing:
        fallback = _vols_replacement(state, draw)
        for slot in missing:
            levels[slot] = WireLevel(fallback.get(slot, 0.0), 0.0, 0.0)
    return levels


def _vols_replacement(state: S.LeagueState, draw: Draw) -> dict[int, float]:
    """VOLS demand rank, used only when the panel shows no free agents at a slot.

    For each slot, rank every player it can start by projected points a week and read
    off the one just past league-wide starter demand. Demand counts every starting
    slot weighted by how much of its eligible pool this slot shares --
    `count * |E_t & E| / |E_t|` -- so a FLEX is charged in full for the RB, WR and TE
    slots it can raid and each of those is charged a third of the FLEX in return.
    """
    mean = np.asarray(draw.panel.mean, dtype=np.float64)
    played = mean > 0.0
    weeks_playing = played.sum(axis=0)
    rate = np.where(weeks_playing > 0, mean.sum(axis=0) / np.maximum(weeks_playing, 1), 0.0)
    positions = np.asarray(state.pool.position_ids, dtype=np.int64)

    plan = plan_from_slots(
        state.lineup_slot_counts,
        state.slot_eligibility,
        state.pool.positions_of(state.franchises[0].player_ids),
    )
    slots = plan.group_slot_ids
    eligible = {s: frozenset(int(p) for p in state.slot_eligibility[s]) for s in slots}
    counts = {s: int(state.lineup_slot_counts[s]) for s in slots}

    out: dict[int, float] = {}
    for s in slots:
        wanted = eligible[s]
        demand = state.size * sum(
            n * len(eligible[t] & wanted) / len(eligible[t]) for t, n in counts.items()
        )
        candidates = np.sort(rate[np.isin(positions, list(wanted))])[::-1]
        out[s] = (
            0.0
            if candidates.size == 0
            else float(candidates[min(int(round(demand)), candidates.size - 1)])
        )
    return out


def _outlooks_from_panel(draw: Draw, state: S.LeagueState) -> list[PlayerOutlook]:
    """Rebuild per-player weekly means from the drawn panel.

    `wire_floor` reads `PlayerOutlook`s and the engine holds a panel, so this is the
    adapter between them. Only the means are needed -- the floor is a per-week mean,
    not a distribution -- so the reconstructed outlooks are deliberately partial and
    are never handed back to a caller.
    """
    mean = np.asarray(draw.panel.mean, dtype=np.float64)
    # sd and p_zero as well as the mean: the floor is a DISTRIBUTION, and rebuilding
    # these as zero silently made every wire level deterministic -- which is exactly
    # what it did until a measurement on the real leagues showed sd 0.00 everywhere.
    sd = np.asarray(draw.panel.sd, dtype=np.float64)
    p_zero = np.asarray(draw.panel.p_zero, dtype=np.float64)
    weeks = tuple(int(w) for w in state.weeks)
    ids = list(state.pool.player_ids)
    positions = list(state.pool.position_ids)
    out: list[PlayerOutlook] = []
    for col, (pid, pos) in enumerate(zip(ids, positions, strict=True)):
        by_week = {
            int(w): WeeklyOutlook(
                player_id=int(pid),
                season=0,
                week=int(w),
                position_id=int(pos),
                mean=float(mean[i, col]),
                sd=float(sd[i, col]),
                p_zero=float(np.clip(p_zero[i, col], 0.0, 1.0)),
                shape=1.0,
                scale=1.0,
            )
            for i, w in enumerate(weeks)
        }
        out.append(
            PlayerOutlook(
                player_id=int(pid), name="", position_id=int(pos), pro_team_id=0, weeks=by_week
            )
        )
    return out


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, ties averaged. Written out rather than pulled from scipy so the
    diagnostic has no import cost in a hot path."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size < 3:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    # Average the ranks inside each tie group, which matters here: a screen that prices
    # many candidates at exactly zero would otherwise be credited with whatever order
    # the sort happened to find.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if unique.size != values.size:
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------


class TitleEngine:
    """`core.MoveEvaluator` over one league: screen thousands, confirm the survivors.

    Build one per league per sitting. It holds the drawn tensor, the baseline scores and
    the baseline season, so every candidate it prices meets identical football and the
    paired difference isolates the move. Two engines are two universes and their numbers
    must never be differenced against each other.

    `replacement` is the one argument that changes whether the answers are usable.
    Leaving it `None` fits `streaming_replacement` off the pool, which is what an
    unfilled slot really scores; passing `0.0` reproduces `sim/season`'s empty-seat
    baseline, under which a kicker outranks a first-round running back. Read
    `streaming_replacement` before choosing.
    """

    __slots__ = (
        "_avail",
        "_base",
        "_base_champ",
        "_base_scores",
        "_factors",
        "_leverage",
        "_moment_cache",
        "_noise",
        "_mu",
        "_plan_cache",
        "_points",
        "_pool_index",
        "_positions",
        "_rank",
        "_sd",
        "_surrogates",
        "_team_index",
        "all_play",
        "draw",
        "mu_grid",
        "playoff_grid",
        "replacement",
        "sigma_grid",
        "state",
        "surrogate_sims",
    )

    def __init__(
        self,
        state: S.LeagueState,
        draw: Draw,
        *,
        replacement: Mapping[int, float] | float | None = None,
        outlooks: Sequence[PlayerOutlook] | None = None,
        efficiency: S.LineupEfficiency | np.ndarray | None = None,
        all_play: bool = False,
        surrogate_sims: int | None = None,
        mu_grid: Sequence[float] = DEFAULT_MU_GRID,
        playoff_grid: Sequence[float] = DEFAULT_PLAYOFF_GRID,
        sigma_grid: Sequence[float] = DEFAULT_SIGMA_GRID,
    ) -> None:
        self.state = state
        self.draw = draw
        self.all_play = all_play
        self.mu_grid = tuple(float(x) for x in mu_grid)
        self.sigma_grid = tuple(float(x) for x in sigma_grid)
        self.playoff_grid = tuple(float(x) for x in playoff_grid)

        # Symmetric by default, matching sim/season.py's shipped default. The haircut is
        # an unverifiable asymmetry that turns the user's below-average teams into title
        # favourites, and a move evaluator must not manufacture that on its own.
        if efficiency is None:
            efficiency = S.LineupEfficiency()
        self._factors = (
            efficiency.draw(state, draw.n_sims)
            if isinstance(efficiency, S.LineupEfficiency)
            else np.asarray(efficiency, dtype=np.float32)
        )

        # Through `season`'s own normaliser rather than reading `draw.points` directly:
        # it is what enforces that the panel covers exactly the pool's players and
        # exactly the state's weeks. The baseline below no longer goes through
        # `team_week_scores`, and that check is the only thing standing between a
        # misaligned draw and a season scored against the wrong players in silence.
        self._points, rank = S._as_points_and_rank(state, draw, None)
        assert rank is not None  # a Draw always builds its own ex-ante rank
        self._rank = rank
        self._team_index = state.team_index

        # Only now that the panel is known to line up with the pool, because fitting the
        # replacement level indexes the draw's means by the pool's positions and a
        # mismatch there is an opaque IndexError rather than the SeasonError above.
        #
        # `None` means "work the level out yourself", NOT `sim/season`'s "leave the seat
        # empty" -- deliberately different, because the empty seat is the one setting
        # that reliably produces a wrong recommendation and it must not be what a caller
        # gets by forgetting an argument. Pass `replacement=0.0` for the empty-slot
        # baseline, which is what `season.team_week_scores(replacement=None)` does and is
        # right only for "how good is this lineup".
        if replacement is None:
            # Levels, not just means: an empty seat has to be PAID, and paying it
            # the mean gives it zero variance. `_floors` reads the means off these
            # for the solve, so the lineup decision is unchanged.
            replacement = streaming_levels(state, draw, outlooks=outlooks)
        self.replacement = replacement
        self._noise = S.FloorNoise(state, draw)

        # Stated per-player moments and the availability the tensor actually drew. Using
        # the drawn availability rather than a hazard formula keeps the screen and the
        # confirmation talking about the same byes and the same absorbing absences.
        self._mu = np.asarray(draw.panel.mean, dtype=np.float64)
        self._sd = np.asarray(draw.panel.sd, dtype=np.float64)
        self._avail = draw.available.mean(axis=0).astype(np.float64)
        self._pool_index = state.pool.index
        self._positions = np.asarray(state.pool.position_ids, dtype=np.int64)
        self._plan_cache: dict[tuple[int, ...], LineupPlan] = {}

        # Assembled a franchise at a time through the same helper `_rerun` uses, rather
        # than through `season.team_week_scores`, so that the baseline arm and the
        # candidate arm are the *same code* -- which is what makes the paired difference
        # exact -- and so the empty-group floor correction in `_floors_for` lands on both.
        self._base_scores = np.zeros((draw.n_sims, len(state.weeks), state.size), np.float32)
        for t, franchise in enumerate(state.franchises):
            self._base_scores[:, :, t] = self._franchise_column(franchise, t)
        self._base = S.simulate_from_scores(state, self._base_scores, all_play=all_play)
        self._base_champ = self._base.champions.astype(np.float64)

        self.surrogate_sims = min(
            draw.n_sims, DEFAULT_SURROGATE_SIMS if surrogate_sims is None else int(surrogate_sims)
        )
        self._surrogates: dict[int, SurrogateFit] = {}
        self._moment_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
        self._leverage: dict[int, tuple[WeekLeverage, ...]] = {}

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_sim(cls, sim, **kwargs) -> TitleEngine:
        """From a `pipeline.LeagueSim`, which is how a live league arrives.

        Passes `sim.outlooks` -- the whole wire, not just the rostered players the
        state's pool holds -- because otherwise the floor silently falls back to the
        VOLS roster-bottom rank on every production path.
        """
        kwargs.setdefault("outlooks", getattr(sim, "outlooks", None))
        return cls(sim.state, sim.draw, **kwargs)

    # -- the baseline ------------------------------------------------------------------

    @property
    def result(self) -> S.SeasonResult:
        """The unmoved season. Held so every delta is against one fixed baseline."""
        return self._base

    def baseline_title(self, team_id: int) -> float:
        return float(self._base_champ[:, self._index(team_id)].mean())

    def baseline_playoffs(self, team_id: int) -> float:
        return float(self._base.made_playoffs[:, self._index(team_id)].mean())

    def baseline_stderr(self, team_id: int) -> float:
        """The error on the *level*, which is much larger than the error on a delta."""
        p = self.baseline_title(team_id)
        return math.sqrt(max(p * (1.0 - p), 0.0) / self.draw.n_sims)

    def _index(self, team_id: int) -> int:
        try:
            return self._team_index[int(team_id)]
        except KeyError:
            raise TitleError(f"no team {team_id} in league {self.state.league_id}") from None

    def moments(self, team_id: int) -> TeamMoments:
        """Per-week mean and SD of this team's starting-lineup total."""
        t = self._index(team_id)
        col = self._base_scores[:, :, t].astype(np.float64)
        return TeamMoments(
            team_id=team_id,
            name=self.state.franchise(team_id).name,
            weeks=self.state.weeks,
            mean_by_week=tuple(col.mean(axis=0).tolist()),
            sd_by_week=tuple(col.std(axis=0, ddof=1).tolist()),
        )

    # -- leverage ----------------------------------------------------------------------

    def week_leverage(self, team_id: int) -> tuple[WeekLeverage, ...]:
        """Per remaining matchup: how much a point is worth, relative to a coin flip.

        The margin and its spread come from the drawn tensor rather than from a normal
        approximation, so a two-week playoff round is one comparison over the summed
        weeks -- which is the point, since summing halves the underdog's variance edge.

        The corpus constant for a one-week matchup is 34.4. Measured here the mean is
        **26.5-28.7** across the three leagues, which puts a projected point at 1.32-1.43pp
        of weekly win probability rather than 1.16pp. Two reasons, and the second is a
        known understatement rather than a finding. Three particular teams are not the
        pooled population the constant was fitted on (with empty seats, before this module
        floored its slots, the same three measured 29.0-31.3). And `sim/season._floors`
        scores an unfilled slot at a *constant* -- a streamed replacement really varies
        week to week, and pretending it does not removes that slot's variance from the
        team, which is worth about 8% of `sd_diff` on these rosters. The floor is still
        far better than the empty seat it replaced; it just biases every spread slightly
        low, and a start/sit surface reading the conversion constant off this should know
        it is reading a lower bound on the spread and therefore an upper bound on what a
        point is worth.
        """
        cached = self._leverage.get(team_id)
        if cached is not None:
            return cached
        t = self._index(team_id)
        windex = self.state.week_index
        out: list[WeekLeverage] = []
        for game in self.state.remaining_games:
            if team_id not in (game.home_team_id, game.away_team_id):
                continue
            other = game.away_team_id if game.home_team_id == team_id else game.home_team_id
            o = self._index(other)
            wis = [windex[w] for w in game.weeks]
            diff = (
                self._base_scores[:, wis, t].sum(axis=1) - self._base_scores[:, wis, o].sum(axis=1)
            ).astype(np.float64)
            margin, sd_diff = float(diff.mean()), float(diff.std(ddof=1))
            out.append(
                WeekLeverage(
                    week=game.weeks[0],
                    matchup_period=game.matchup_period,
                    team_id=team_id,
                    opponent_id=other,
                    margin=margin,
                    sd_diff=sd_diff,
                    win_probability=float((diff > 0).mean()),
                    leverage=leverage(margin, sd_diff),
                )
            )
        rows = tuple(sorted(out, key=lambda w: w.week))
        self._leverage[team_id] = rows
        return rows

    def mean_leverage(self, team_id: int) -> float:
        """Average leverage over the remaining schedule. 1.0 means every week is live."""
        rows = self.week_leverage(team_id)
        return float(np.mean([r.leverage for r in rows])) if rows else 1.0

    # -- tier 1: the surrogate ---------------------------------------------------------

    def surrogate(self, team_id: int, *, refit: bool = False) -> SurrogateFit:
        """Fit (and cache) this team's response surface.

        Per team, not per league. The sign of `dP/dsigma` depends on which side of the
        playoff cut the team sits, so a surface pooled across teams averages the two
        populations and mis-screens every variance-changing move near the line. Fitting
        from the team's own perturbed scores conditions on standing by construction, and
        because the state has already reduced played weeks to facts, refitting as the
        season moves needs nothing but a rebuilt engine.
        """
        if refit:
            self._surrogates.pop(team_id, None)
        cached = self._surrogates.get(team_id)
        if cached is None:
            cached = self._fit_surrogate(team_id)
            self._surrogates[team_id] = cached
        return cached

    def _fit_surrogate(self, team_id: int) -> SurrogateFit:
        t = self._index(team_id)
        n = self.surrogate_sims
        scores = np.array(self._base_scores[:n], dtype=np.float32, copy=True)
        col = scores[:, :, t].astype(np.float64)
        mu_week = col.mean(axis=0)
        resid = col - mu_week
        sigma = float(np.sqrt(np.mean(col.var(axis=0, ddof=1))))

        # The bracket axis is only meaningful while a bracket is still ahead of us. With
        # none left -- or with nothing BUT bracket weeks left -- every playoff node would
        # duplicate a node the mu axis already has, and the two columns would be exactly
        # collinear. Collapse to the single zero node instead of feeding the fit a
        # singular block and relying on the ridge to hide it.
        bracket = playoff_mask(self.state)
        degenerate = not bracket.any() or bool(bracket.all())
        playoff_grid = (0.0,) if degenerate else tuple(self.playoff_grid)

        shape = (len(self.mu_grid), len(self.sigma_grid), len(playoff_grid))
        titles = np.zeros(shape)
        playoffs = np.zeros_like(titles)
        grid_mu = np.zeros_like(titles)
        grid_ratio = np.zeros_like(titles)
        grid_playoff = np.zeros_like(titles)
        for i, d_mu in enumerate(self.mu_grid):
            for j, ratio in enumerate(self.sigma_grid):
                for k, d_po in enumerate(playoff_grid):
                    # A team total cannot go negative; the clip only ever binds at the
                    # bottom corner of the grid for a team scoring under 15 a week.
                    scores[:, :, t] = np.maximum(
                        mu_week + d_mu + d_po * bracket + ratio * resid, 0.0
                    )
                    res = S.simulate_from_scores(self.state, scores, all_play=False)
                    titles[i, j, k] = res.champions[:, t].sum()
                    playoffs[i, j, k] = res.made_playoffs[:, t].sum()
                    grid_mu[i, j, k] = d_mu
                    grid_ratio[i, j, k] = ratio
                    grid_playoff[i, j, k] = d_po

        beta, rmse = _fit_binomial_surface(
            grid_mu, grid_ratio, titles, n, d_mu_playoff=grid_playoff
        )
        beta_p, rmse_p = _fit_binomial_surface(
            grid_mu, grid_ratio, playoffs, n, d_mu_playoff=grid_playoff
        )
        fit = SurrogateFit(
            team_id=team_id,
            mu=float(mu_week.mean()),
            sigma=sigma,
            baseline_title=self.baseline_title(team_id),
            baseline_playoffs=self.baseline_playoffs(team_id),
            coefficients=beta,
            playoff_coefficients=beta_p,
            rmse=rmse,
            playoff_rmse=rmse_p,
            mu_grid=self.mu_grid,
            sigma_grid=self.sigma_grid,
            n_sims=n,
            title_nodes=titles / n,
            playoff_nodes=playoffs / n,
            playoff_grid=playoff_grid,
        )
        log.info(
            "surrogate for team %s: mu %.1f sigma %.1f, title %.2f%% (rmse %.3fpp), "
            "dP_title/dsigma %+.3fpp dP_playoff/dsigma %+.3fpp per 10%% of SD, "
            "a bracket point worth %.2fx an ordinary one",
            team_id,
            fit.mu,
            fit.sigma,
            100.0 * fit.baseline_title,
            100.0 * rmse,
            1000.0 * fit.d_title_d_sigma(),
            1000.0 * fit.d_playoffs_d_sigma(),
            fit.playoff_premium,
        )
        return fit

    # -- roster moments, the order statistic -------------------------------------------

    def _plan_for(self, positions: np.ndarray) -> LineupPlan:
        """A compiled lineup plan, cached on the roster's position *vector*.

        Not on the roster and not on a sorted multiset: the plan compiles an eligibility
        matrix whose columns are the roster in the order given, so two rosters with the
        same positions in the same order share a plan and any other pair must not. A
        one-for-one swap of like for like leaves the vector alone, which is most of what a
        trade search proposes, so the cache hits almost every time -- and compiling a plan
        is a large enough share of a candidate's cost that it is worth the dictionary.
        """
        key = tuple(int(p) for p in positions)
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = plan_from_slots(
                self.state.lineup_slot_counts, self.state.slot_eligibility, positions
            )
            self._plan_cache[key] = plan
        return plan

    def _floors_for(self, plan: LineupPlan) -> tuple[Mapping[int, float] | float | None, float]:
        """Floors it is safe to hand this roster's plan, and the points a week they omit.

        `lineup.monotone_floor` raises a slot's floor to that of every slot whose eligible
        set it contains, and eligibility is computed against **this roster**, not against
        the position table. A roster with nobody at a position leaves that group's
        eligible set empty; the empty set is contained in every other group; and so every
        slot on the team is lifted to the missing position's floor. It is not a small
        error: dropping the only quarterback off the user's Blacksburg roster lifts all
        nine slots to the QB replacement level and produces a 137.6-point-a-week team with
        *zero* variance and a 93% title probability, against 5.8% before the drop. Every
        candidate that empties a position -- trading away the only tight end, cutting the
        kicker -- lands on it, and those are candidates a search proposes by the hundred.

        So an empty group is handed a floor of zero, which lifts nothing, and the points
        its slots really stream are returned separately to be added back to the team's
        weekly total. That is exact rather than approximate: a group with nothing eligible
        takes its floor in every week of every simulation, so the omission is a constant.
        """
        floors = self.replacement
        if not isinstance(floors, Mapping):
            # A scalar floor is uniform, so the lift is a no-op and there is nothing to
            # correct; `None` is the empty-seat baseline and has no floors at all.
            return floors, 0.0
        empty = [g for g in range(plan.n_groups) if not plan._eligible[g].any()]
        if not empty:
            return floors, 0.0
        safe = dict(floors)
        omitted = 0.0
        for g in empty:
            slot = plan.group_slot_ids[g]
            level = floors[slot]
            # The floors may be WireLevels; the omission correction is a MEAN, because
            # a group with nothing eligible takes its floor in every week of every
            # simulation and the constant part is what the guard adds back. Zeroing the
            # entry still zeroes the credit, because `_floors` derives both from here.
            omitted += float(getattr(level, "mean", level)) * plan.group_counts[g]
            safe[slot] = WireLevel(0.0, 0.0, 0.0) if isinstance(level, WireLevel) else 0.0
        return safe, omitted

    def _franchise_column(self, franchise: S.Franchise, team: int) -> np.ndarray:
        """One franchise's `(sims, weeks)` starting-lineup total, efficiency applied.

        The single path both arms go through: the baseline in `__init__` and every
        candidate in `_rerun`. Two arms that differ only in a roster must not differ in
        the code that scores it, or the paired difference stops being a difference.
        """
        plan = plan_from_slots(
            self.state.lineup_slot_counts,
            self.state.slot_eligibility,
            self.state.pool.positions_of(franchise.player_ids),
        )
        floors, omitted = self._floors_for(plan)
        solo = S._franchise_scores(
            self.state.pool,
            franchise,
            plan,
            self._points,
            self._rank,
            floors,
            floor_noise=self._noise.for_plan(plan, team),
        )
        return (solo + np.float32(omitted)) * self._factors[:, team : team + 1]

    def _roster_moments(self, player_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """`(mean, variance)` per remaining week for this roster's starting lineup.

        The ex-ante lineup is solved on projected means -- exactly what the simulator
        does -- and each slot is then priced as the sequential order statistic down its
        depth chart: the starter when he is available, the best unused eligible backup
        when he is not, the free-agent floor when nobody is. `a_i` is read off the drawn
        tensor, so byes and the absorbing injury model are already in it.

        Two approximations, both of which difference out between two rosters that share
        most of their players. Backups are shared across the slot instances of a group
        beyond the offset applied here, so deep benches are credited slightly twice; and
        slot totals are added as if independent, which understates the team variance by
        the same within-team correlation on both sides of the comparison. `screen`
        calibrates the *difference* onto the simulated baseline moments rather than using
        these levels directly, which is what makes both harmless.
        """
        key = tuple(sorted(int(p) for p in player_ids))
        cached = self._moment_cache.get(key)
        if cached is not None:
            return cached

        cols = self._columns(player_ids)
        mu = self._mu[:, cols]
        sd = self._sd[:, cols]
        avail = self._avail[:, cols]
        n_weeks, size = mu.shape

        plan = self._plan_for(self._positions[cols])
        floors, omitted = self._floors_for(plan)
        groups, per_slot, credit = S._floors(plan, floors)
        floor_var = np.zeros(plan.n_slots) if credit is None else np.asarray(credit.variance)
        rank = np.where(avail > 0.0, mu, -np.inf)
        assignment = plan.solve(rank, floor=groups, assignment=True).assignment
        assert assignment is not None

        used = np.zeros((n_weeks, size), dtype=bool)
        rows = np.repeat(np.arange(n_weeks), plan.n_slots)
        flat = assignment.reshape(-1)
        taken = flat >= 0
        used[rows[taken], flat[taken]] = True

        second = mu * mu + sd * sd
        # Per slot group, the unused eligible bench in descending projected order. This
        # is the depth chart a manager would actually fall down to.
        bench: list[np.ndarray] = []
        for g in range(plan.n_groups):
            eligible = plan._eligible[g][None, :] & ~used & (avail > 0.0)
            bench.append(np.argsort(np.where(eligible, -mu, np.inf), axis=1, kind="stable"))

        group_of = np.repeat(np.arange(plan.n_groups), plan.group_counts)
        within = np.concatenate([np.arange(c) for c in plan.group_counts]) if plan.n_slots else ()
        rows = np.arange(n_weeks)
        mean = np.zeros(n_weeks)
        variance = np.zeros(n_weeks)
        for i in range(plan.n_slots):
            g = int(group_of[i])
            floor = float(per_slot[i])
            # Second moment carries the floor's own variance, so the screen and the
            # confirm price the same world. Without it they diverge on every
            # candidate that changes how many seats sit empty -- which is all of them.
            first, second_moment = floor, floor * floor + float(floor_var[i])
            order = bench[g]
            usable = plan._eligible[g][None, :] & ~used
            # Each instance in a group starts one place further down the shared bench, so
            # two empty RB slots do not both claim the same waiver body.
            for depth in range(BENCH_DEPTH - 1, -1, -1):
                pick = order[:, min(int(within[i]) + depth, size - 1)]
                # `argsort` still returns ineligible columns once the bench runs out, so
                # the pick is only real where it is genuinely eligible and unassigned.
                a = np.where(usable[rows, pick], avail[rows, pick], 0.0)
                first = a * mu[rows, pick] + (1.0 - a) * first
                second_moment = a * second[rows, pick] + (1.0 - a) * second_moment
            starter = assignment[:, i]
            has = starter >= 0
            safe = np.where(has, starter, 0)
            a0 = np.where(has, avail[rows, safe], 0.0)
            slot_mean = a0 * mu[rows, safe] + (1.0 - a0) * first
            slot_second = a0 * second[rows, safe] + (1.0 - a0) * second_moment
            mean += slot_mean
            variance += np.maximum(slot_second - slot_mean * slot_mean, 0.0)

        # The floor of any group with nothing eligible, held out of the solve so it
        # cannot lift every other slot. Deterministic, so it moves the mean and not the
        # variance. See `_floors_for`.
        mean += omitted
        out = (mean, variance)
        self._moment_cache[key] = out
        return out

    def _hindsight_moments(self, player_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """The research note's version: Clark's `E[max]` over *realised* points.

        Kept so the two can be measured against each other rather than argued about.
        Every starter is credited with `E[max(himself, his best backup)]`, which is what
        a manager would score if he set his lineup on Sunday night. `agreement()` shows
        it ranks candidates slightly worse than the ex-ante order statistic on all three
        real leagues (rho 0.958/0.979/0.981 against 0.969/0.983/0.982). The gap is small
        because Clark's bonus depends mostly on the incumbent's own spread and so is
        nearly constant across candidates -- it cancels out of a *ranking*. It does not
        cancel out of the level: it pays a bench for information nobody has.
        """
        mean, variance = self._roster_moments(player_ids)
        cols = self._columns(player_ids)
        mu, sd = self._mu[:, cols], self._sd[:, cols]
        plan = self._plan_for(self._positions[cols])
        groups, per_slot, _credit = S._floors(plan, self.replacement)
        rank = np.where(self._avail[:, cols] > 0.0, mu, -np.inf)
        assignment = plan.solve(rank, floor=groups, assignment=True).assignment
        assert assignment is not None

        rows = np.arange(mu.shape[0])
        bonus = np.zeros_like(mean)
        for i in range(plan.n_slots):
            starter = assignment[:, i]
            has = starter >= 0
            safe = np.where(has, starter, 0)
            m = mu[rows, safe]
            s = sd[rows, safe]
            floor = np.full_like(m, float(per_slot[i]))
            bonus += np.where(has, clark_bonus(m, s, floor), 0.0)
        return mean + bonus, variance

    def _columns(self, player_ids: Sequence[int]) -> np.ndarray:
        """Tensor columns for a roster. Against a cached index, not `PlayerPool.index`.

        `PlayerPool.index` is a property that rebuilds the whole dictionary on every
        access, and a screen calls it several times per candidate over a 200-player pool.
        Caching it, and the per-team leverage beside it, took tier 1 from 2.7ms a
        candidate to 0.95ms without changing a single number it reports.
        """
        try:
            return np.array([self._pool_index[int(p)] for p in player_ids], dtype=np.intp)
        except KeyError as err:
            raise TitleError(
                f"player {err.args[0]} is not in the drawn panel. Every player a candidate "
                "touches must be: build the pool over the union of every roster and every "
                "free agent the search might reach (see decide.title.sim_with_free_agents), "
                "or the comparison is against a different random universe"
            ) from None

    # -- moves -------------------------------------------------------------------------

    def subject_team(self, move: Move, subject: int | None = None) -> int:
        """Whose title probability this move is reported in.

        The user's own franchise whenever it is involved, which is what makes a trade
        offer and a waiver claim comparable on one list. Otherwise the lowest team id, so
        a blocking move against a rival still prices deterministically. Pass `subject`
        explicitly to price the *other* side: a trade surface has to show both, because a
        proposal only gets accepted if it is positive for the counterparty too, and the
        engine is the only thing that knows what it is worth to them.
        """
        if subject is not None:
            self._index(subject)  # raises if this league has no such franchise
            return int(subject)
        teams = move.teams
        if not teams:
            mine = self.state.my_team_id
            if mine is None:
                raise TitleError("a move with no players needs a team; set state.my_team_id")
            return mine
        if self.state.my_team_id in teams:
            return int(self.state.my_team_id)
        return int(min(teams))

    def _rosters_after(self, move: Move) -> dict[int, tuple[int, ...]]:
        """team_id -> new roster, for every franchise the move changes."""
        after: dict[int, tuple[int, ...]] = {}
        for pm in move.players:
            if pm.player_id not in self._pool_index:
                self._columns([pm.player_id])  # raises with the panel-coverage message
            if pm.from_team is not None:
                current = after.get(pm.from_team, self.state.franchise(pm.from_team).player_ids)
                if pm.player_id not in current:
                    raise TitleError(
                        f"move sends player {pm.player_id} from team {pm.from_team}, "
                        "which does not roster him"
                    )
                after[pm.from_team] = tuple(p for p in current if p != pm.player_id)
            if pm.to_team is not None:
                current = after.get(pm.to_team, self.state.franchise(pm.to_team).player_ids)
                after[pm.to_team] = (*current, pm.player_id)
        return {
            team: ids for team, ids in after.items() if ids != self.state.franchise(team).player_ids
        }

    # -- tier 1 ------------------------------------------------------------------------

    def screen(
        self,
        moves: Sequence[Move],
        *,
        subject: int | None = None,
        counterparty: bool = True,
        hindsight_max: bool = False,
    ) -> list[Recommendation]:
        """Analytic estimate for every candidate, in the order given.

        No simulation runs here: each candidate costs one lineup solve over the affected
        rosters' projected means and one read of a fitted surface. The surrogate for a
        team is fitted on first use and cached, so the first candidate touching a team
        pays about half a second and every one after it pays 0.95ms -- measured over
        2,800-3,344 candidates on each of the user's leagues at 4,000 simulations.

        `counterparty` renormalises for the other side of a trade. Title probabilities
        sum to one, so a rival's gain comes out of the rest of the field in proportion to
        their standing; without it every trade that helps both teams reads as a pure win.

        `stderr` here is the surrogate's own RMSE, not a Monte Carlo error -- so a
        screened effect smaller than the surface can resolve reports `significant =
        False` rather than being dressed up. Only `confirm` populates a real error bar.

        `subject` prices the move from another franchise's point of view, which is what a
        trade surface needs: a proposal is only accepted if it is positive for them too.
        """
        return [self._screen_one(m, subject, counterparty, hindsight_max) for m in moves]

    def _screen_one(
        self, move: Move, subject: int | None, counterparty: bool, hindsight_max: bool
    ) -> Recommendation:
        subject = self.subject_team(move, subject)
        rosters = self._rosters_after(move)
        if not rosters:
            return self._null(move, subject, "screen")

        moments = self._hindsight_moments if hindsight_max else self._roster_moments
        surrogate = self.surrogate(subject)
        delta = 0.0
        d_points = 0.0
        others: dict[int, float] = {}
        bracket = playoff_mask(self.state)
        for team, ids in rosters.items():
            base_mean, base_var = moments(self.state.franchise(team).player_ids)
            new_mean, new_var = moments(ids)
            eff = float(self._factors[:, self._index(team)].mean())
            # Split by calendar, not averaged over it. `np.mean(new_mean - base_mean)`
            # was the whole of the old computation, and it is what made a week-16 point
            # identical to a week-3 point -- the two are worth +3.82pp and +0.85pp of
            # title probability on the live leagues, a difference this collapsed to one
            # number and then screened on.
            delta_week = new_mean - base_mean
            d_mu, d_mu_playoff = split_by_bracket(delta_week, bracket)
            d_mu *= eff
            d_mu_playoff *= eff
            # One sigma axis still. A bracket-specific VARIANCE tilt is second order --
            # the sign flip it would carry is already on the level term -- and a fourth
            # axis would multiply the fit grid again for it. Noted, not modelled.
            d_var = eff * eff * float(np.mean(new_var - base_var))
            fit = surrogate if team == subject else self.surrogate(team)
            ratio = math.sqrt(max(fit.sigma**2 + d_var, 1e-6)) / max(fit.sigma, 1e-6)
            own = fit.delta_title(d_mu, ratio, d_mu_playoff)
            if team == subject:
                delta += own
                # The honest total, summed over the weeks it actually lands in. It was
                # `d_mu * len(weeks)`, which silently dropped every point of the bracket
                # tilt -- so a playoff-only upgrade reported ZERO points gained.
                d_points = eff * float(np.sum(delta_week))
            else:
                others[team] = own
                if counterparty:
                    # Title probabilities sum to one, so a rival's gain is drawn from the
                    # rest of the field in proportion to standing and our share of the
                    # loss is p_me / (1 - p_them). For two average teams that is 1/(N-1),
                    # which is exactly the research note's rule for what blocking a rival
                    # is worth -- arrived at here from the renormalisation rather than
                    # assumed.
                    share = surrogate.baseline_title / max(1.0 - fit.baseline_title, 1e-6)
                    delta -= share * own
        return self._recommend(
            move,
            subject,
            delta_title=delta,
            delta_points=d_points,
            # The surface's own RMSE is the error on a *level*, and the screen's real
            # error against `confirm` grows with the size of the effect: measured, it is
            # about 0.25pp plus a fifth of the effect. Reporting only the RMSE calls a
            # +12pp screened trade accurate to a third of a point.
            stderr=math.hypot(surrogate.rmse, SCREEN_RELATIVE_ERROR * delta),
            tags=("screen", *self._trade_tags(move, subject, rosters, others, delta)),
            confidence="low",
            others=others,
        )

    # -- tier 2 ------------------------------------------------------------------------

    def confirm(self, moves: Sequence[Move], *, subject: int | None = None) -> list[Recommendation]:
        """Paired CRN simulation, in the order given. This is the authoritative number.

        Measured on the user's leagues at 4,000 simulations: 79-87ms a candidate against
        0.95ms for `screen`. That factor of 90 is the entire argument for the two tiers,
        and it is why `evaluate` exists.
        """
        return [self._confirm_one(m, subject) for m in moves]

    def _rerun(self, move: Move) -> tuple[S.SeasonResult, np.ndarray] | None:
        """Re-simulate with the move applied. `None` when no roster actually changes.

        Only the touched franchises have their lineups re-solved; every other team's
        weekly total is reused verbatim, which is what makes the per-simulation
        difference exact. The whole league is still re-stood, because taking a player off
        one roster changes who his opponents beat and therefore who takes the last seed.
        """
        rosters = self._rosters_after(move)
        if not rosters:
            return None
        state = self.state
        scores = self._base_scores.copy()
        for team, ids in rosters.items():
            franchise = state.franchise(team).with_players(ids)
            state = state.with_franchise(franchise)
            t = self._index(team)
            scores[:, :, t] = self._franchise_column(franchise, t)
        return S.simulate_from_scores(state, scores, all_play=self.all_play), scores

    def _confirm_one(self, move: Move, subject: int | None = None) -> Recommendation:
        subject = self.subject_team(move, subject)
        rerun = self._rerun(move)
        if rerun is None:
            return self._null(move, subject, "confirm")
        alt, scores = rerun
        t = self._index(subject)
        paired = alt.champions[:, t].astype(np.float64) - self._base_champ[:, t]
        n = paired.size
        delta = float(paired.mean())
        stderr = float(paired.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        d_points = float(
            (scores[:, :, t].astype(np.float64) - self._base_scores[:, :, t].astype(np.float64))
            .sum(axis=1)
            .mean()
        )
        # Every other franchise the move touches, off the SAME rerun: the alternative
        # season already holds all of their champions, so knowing whether the trade is
        # one the counterparty would take costs nothing beyond the arithmetic. There is
        # no excuse for shipping only our own half of it.
        rosters = self._rosters_after(move)
        others: dict[int, float] = {}
        errors: dict[int, float] = {}
        for team in rosters:
            if team == subject:
                continue
            j = self._index(team)
            theirs = alt.champions[:, j].astype(np.float64) - self._base_champ[:, j]
            others[team] = float(theirs.mean())
            errors[team] = float(theirs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        return self._recommend(
            move,
            subject,
            delta_title=delta,
            delta_points=d_points,
            stderr=stderr,
            tags=("confirm", "paired", *self._trade_tags(move, subject, rosters, others, delta)),
            confidence=self._confidence(delta, stderr),
            others=others,
            errors=errors,
        )

    def independent_stderr(self, move: Move) -> float:
        """What the error on `delta_title` would have been WITHOUT common random numbers.

        The two arms' binomial errors in quadrature, which is exactly what differencing
        two separately drawn simulations gives you. The ratio against the paired error is
        the only proof that the pairing is doing anything, and `crn_variance_ratio`
        reports it alongside the same ratio on wins and points-for.
        """
        subject = self.subject_team(move)
        before = self.baseline_title(subject)
        rec = self._confirm_one(move)
        after = before + rec.delta_title
        n = self.draw.n_sims
        return math.sqrt(before * (1.0 - before) / n + after * (1.0 - after) / n)

    def crn_variance_ratio(self, move: Move) -> dict[str, float]:
        """Independent-arm variance over paired variance, per metric. Bigger is better.

        Reported rather than assumed, because the numbers `sim/season.py` documents --
        1000x on points-for, 27x on wins, 5x on the championship -- were measured by
        *scaling one player's realisations by 15%*, which is a shift of the same football.
        Every candidate this engine prices is a **substitution**: the incoming player is a
        different random variable, and his own week-to-week variance is irreducible by any
        amount of pairing. Reproducing their protocol here gives 513 / 32 / 3.8; the
        identical machinery on a real one-for-one swap in the user's leagues gives
        **7-8 / 3.9-4.3 / 1.19-1.50**. Budget simulations off the substitution numbers:
        resolving a 0.4pp edge on a trade takes roughly three times the simulations the
        perturbation figures imply.
        """
        subject = self.subject_team(move)
        t = self._index(subject)
        rerun = self._rerun(move)
        if rerun is None:
            return {"title": math.inf, "wins": math.inf, "points_for": math.inf}
        alt, _ = rerun
        out: dict[str, float] = {}
        for key, before, after in (
            ("title", self._base.champions[:, t], alt.champions[:, t]),
            ("wins", self._base.wins[:, t], alt.wins[:, t]),
            ("points_for", self._base.points_for[:, t], alt.points_for[:, t]),
        ):
            a = np.asarray(before, dtype=np.float64)
            b = np.asarray(after, dtype=np.float64)
            paired = float((b - a).var(ddof=1))
            independent = float(a.var(ddof=1) + b.var(ddof=1))
            out[key] = math.inf if paired <= 0 else independent / paired
        return out

    # -- the two-tier front door --------------------------------------------------------

    def evaluate(
        self,
        moves: Sequence[Move],
        *,
        keep: int = DEFAULT_CONFIRM,
        counterparty: bool = True,
        acceptable_only: bool = False,
    ) -> list[Recommendation]:
        """Screen everything, confirm the best `keep`, return the confirmed list sorted.

        The shortlist is taken on the *absolute* screened effect, not the signed one: a
        move that would cost two points of title probability is as much worth confirming
        as one that would gain it, and a surface that mis-signs a candidate near the cut
        would otherwise never get a second look.

        `acceptable_only` keeps only the candidates every party gains from, and drops the
        ones that quietly change a roster's size. **Read what the default returns before
        printing it at anybody.** Over `one_for_one_candidates` -- every swap in the
        league, which is what this module ships as a candidate generator -- the top of
        the unfiltered list on each of the user's three real leagues is a rival's best
        player for a bench body at about +13pp against a counterparty losing 7 to 9pp:
        correctly priced and completely unavailable. The screen already knows both sides,
        so the filter is free; it is off by default only because `evaluate` is also how
        the *engine* is measured, and a measurement that silently drops candidates
        measures the filter instead.

        An empty list from `acceptable_only` is an answer, not a failure. A one-for-one
        swap that lifts both teams has to take the probability out of the rest of the
        league, and between two rosters with no complementary hole there may be no such
        swap at all. "Nothing here is worth proposing" beats inventing something.

        One thing the list does not correct for: the top entry is the maximum of `keep`
        noisy estimates, so it carries a selection bias. Re-confirming the top ten in a
        second, independently drawn universe shrinks them by 0.15 / 0.45 / 0.37pp on
        average across the three leagues, and the single largest shrink was 1.6pp on a
        +12.8pp move -- so read the top of a long list as the top of a long list.
        """
        screened = self.screen(moves, counterparty=counterparty)
        keepers = list(range(len(moves)))
        prefer_acceptable = False

        def rank(i: int) -> tuple[float, float]:
            # Screened-acceptable first when asked, then by the size of the screened
            # effect. The second key alone is the unconditional order.
            unilateral = prefer_acceptable and "unilateral" in screened[i].tags
            return (1.0 if unilateral else 0.0, -abs(screened[i].delta_title))

        if acceptable_only:
            # `roster_size` is DELETED: roster-length arithmetic, identical under both
            # tiers, and free to prune on.
            keepers = [i for i in keepers if "roster_size" not in screened[i].tags]
            # `unilateral` only REORDERS. It is derived from the SIGNS of screened deltas
            # (`_trade_tags`), and deleting on it -- which is what this did -- threw a
            # candidate away before `confirm` could disagree, irrecoverably. Measured on
            # the fixture, 10 of 144 candidates the screen calls unacceptable are
            # acceptable once simulated, against 1 the other way. `decide/trades.py`
            # gates every sign-dependent statement on `ev.confirmed`; this is that rule.
            #
            # Sorting rather than filtering keeps the efficiency the filter was there for
            # -- the shortlist still leads with the trades the screen thinks are
            # acceptable -- while leaving the tail recoverable when there are not `keep`
            # of them. On a board that is all robberies the old code confirmed 25
            # acceptable-LOOKING moves and this confirms the 25 largest, which is where
            # a mis-signed one can actually be found.
            prefer_acceptable = True
        order = sorted(keepers, key=rank)
        shortlist = order[: max(keep, 0)]
        confirmed = self.confirm([moves[i] for i in shortlist])
        confirmed = [
            replace(rec, tags=(*rec.tags, f"screen_rank={rank}"))
            for rank, rec in enumerate(confirmed, start=1)
        ]
        if acceptable_only:
            confirmed = [
                r for r in confirmed if "unilateral" not in r.tags and "roster_size" not in r.tags
            ]
        confirmed.sort(reverse=True)
        return confirmed

    def agreement(
        self,
        moves: Sequence[Move],
        *,
        shortlist: int = DEFAULT_CONFIRM,
        top_k: int = 10,
        counterparty: bool = True,
        hindsight_max: bool = False,
    ) -> ScreenAgreement:
        """Run both tiers over the same candidates and report how well tier 1 filters.

        Confirming every candidate is the point: this is the diagnostic that decides
        whether the screen can be trusted, so it must not be computed on the screen's own
        shortlist. Run it on a sample of a few hundred, not on the full search.

        Time the *second* call on a league, or warm the surrogate cache first: a surface
        fit is half a second and belongs to the team rather than to the candidate, so on a
        150-move sample it lands entirely in `screen_seconds` and reports tier 1 as fifty
        times slower than it is.
        """
        start = time.perf_counter()
        screened = self.screen(moves, counterparty=counterparty, hindsight_max=hindsight_max)
        screen_seconds = time.perf_counter() - start
        start = time.perf_counter()
        confirmed = self.confirm(moves)
        confirm_seconds = time.perf_counter() - start

        a = np.array([r.delta_title for r in screened])
        b = np.array([r.delta_title for r in confirmed])
        keep = set(np.argsort(-np.abs(a))[:shortlist].tolist())
        best = np.argsort(-b)[:top_k].tolist()
        denominator = float(np.dot(a, a))
        return ScreenAgreement(
            n=len(moves),
            spearman=_spearman(a, b),
            pearson=float(np.corrcoef(a, b)[0, 1]) if len(moves) > 2 else float("nan"),
            scale=float(np.dot(a, b) / denominator) if denominator > 0 else float("nan"),
            shortlist=shortlist,
            top_k=top_k,
            recall_at=sum(1 for i in best if i in keep) / max(len(best), 1),
            screen_seconds=screen_seconds,
            confirm_seconds=confirm_seconds,
        )

    # -- assembling a Recommendation ----------------------------------------------------

    def _trade_tags(
        self,
        move: Move,
        subject: int,
        rosters: Mapping[int, tuple[int, ...]],
        others: Mapping[int, float],
        mine: float,
    ) -> tuple[str, ...]:
        """`unilateral` / `pareto`, and `roster_size` -- the two ways a number lies.

        **`unilateral`.** The engine's own candidate generator, `one_for_one_candidates`,
        proposes every swap in the league, and the best of them is always the one where a
        rival hands over his first-round pick for a bench body. Run against the user's
        three real leagues, the top of `evaluate` is "+Jahmyr Gibbs, -Ray Davis, +13.48pp"
        while the counterparty loses 7.15pp -- arithmetically right, and not a move
        anybody can make. A `delta_title` is not a recommendation until somebody would
        accept it, so a move that costs any party title probability is tagged and says so
        in its rationale.

        `pareto` means **every** party gains, the subject included. Testing only the
        counterparty is the same bug pointing the other way, and it is not hypothetical:
        the first version of this gate did exactly that, and `acceptable_only` came back
        led by "+Cooper Kupp, -Kenneth Walker III, -2.50pp" -- a giveaway, kept because
        the other side liked it. A trade both sides take must be positive on both sides.

        **`roster_size`.** `Move` can name an add with no drop, and nothing below here
        knows a league has a roster limit -- `LeagueState` does not carry one. Priced as
        posed, "+De'Von Achane" and nothing else is +5.85pp of title probability for a
        seventeenth man on a sixteen-man roster. That is a real number for an unreal move,
        so it is labelled rather than quietly ranked against legal ones.
        """
        tags: list[str] = []
        if any(len(ids) != len(self.state.franchise(t).player_ids) for t, ids in rosters.items()):
            tags.append("roster_size")
        if others:
            everyone = [mine, *others.values()]
            if min(everyone) < 0.0:
                tags.append("unilateral")
            elif min(everyone) > 0.0:
                tags.append("pareto")
        return tuple(tags)

    def _counterparty_note(
        self, subject: int, others: Mapping[int, float], significance: Mapping[int, float] | None
    ) -> str:
        """The half of a trade the subject's own `delta_title` cannot show."""
        if not others:
            return ""
        parts = []
        for team, value in sorted(others.items(), key=lambda kv: kv[1]):
            name = self.state.franchise(team).name
            error = f" +/- {100 * significance[team]:.2f}" if significance else ""
            parts.append(f"{name} {100 * value:+.2f}pp{error}")
        verdict = (
            " -- they lose, so this is not a trade they accept"
            if min(others.values()) < 0.0
            else " -- positive for them too"
        )
        return f"; for the other side: {', '.join(parts)}{verdict}"

    @staticmethod
    def _confidence(delta: float, stderr: float) -> str:
        if stderr <= 0:
            return "high"
        z = abs(delta) / stderr
        return "high" if z > 3.0 else "medium" if z > 2.0 else "low"

    def _null(self, move: Move, subject: int, tier: str) -> Recommendation:
        """A move that changes no roster. Exactly zero, not a small number.

        Lineup moves land here on purpose: the simulator already starts every team's
        ex-ante optimal lineup, so there is no title probability in telling it to. A
        start/sit decision has to be priced against the lineup the manager actually
        submitted, which is a different question and a different surface.
        """
        why = (
            "changes no roster"
            if move.kind is not MoveKind.LINEUP
            else "the season model already starts the optimal lineup, so a lineup change "
            "has no rest-of-season title effect; price start/sit against the submitted "
            "lineup instead"
        )
        return Recommendation(
            move=move,
            delta_title=0.0,
            delta_points=0.0,
            stderr=0.0,
            leverage=self.mean_leverage(subject),
            rationale=f"No effect: this {move.kind.value} {why}.",
            confidence="high",
            tags=(tier, "null"),
        )

    def _recommend(
        self,
        move: Move,
        subject: int,
        *,
        delta_title: float,
        delta_points: float,
        stderr: float,
        tags: tuple[str, ...],
        confidence: str,
        others: Mapping[int, float] | None = None,
        errors: Mapping[int, float] | None = None,
    ) -> Recommendation:
        base = self.baseline_title(subject)
        lever = self.mean_leverage(subject)
        rec = Recommendation(
            move=move,
            delta_title=delta_title,
            delta_points=delta_points,
            stderr=stderr,
            leverage=lever,
            rationale="",
            confidence=confidence,
            tags=tags,
        )
        return replace(rec, rationale=self._rationale(rec, subject, base, lever, others, errors))

    def _rationale(
        self,
        rec: Recommendation,
        subject: int,
        base: float,
        lever: float,
        others: Mapping[int, float] | None = None,
        errors: Mapping[int, float] | None = None,
    ) -> str:
        """One line a human can act on: the title move first, the points second.

        Points come second deliberately. A large points gain in a locked-up matchup is
        worth nothing, and showing both is the only way to demonstrate that rather than
        assert it.
        """
        pool = self.state.pool
        name = self.state.franchise(subject).name
        moved = ", ".join(
            f"{'+' if pm.to_team == subject else '-'}{pool.name(pm.player_id)}"
            for pm in rec.move.players
            if subject in (pm.to_team, pm.from_team)
        )
        head = f"{name}: {moved or rec.move.kind.value}"
        odds = (
            f"title {100 * base:.1f}% -> {100 * (base + rec.delta_title):.1f}% "
            f"({100 * rec.delta_title:+.2f}pp"
        )
        odds += f" +/- {100 * rec.stderr:.2f}pp)" if rec.stderr > 0 else ")"
        weeks = len(self.state.weeks)
        per_week = rec.delta_points / weeks if weeks else 0.0
        points = (
            f"{rec.delta_points:+.0f} starting-lineup points over {weeks} weeks "
            f"({per_week:+.1f} a week)"
        )
        if rec.delta_title == 0.0 and rec.stderr == 0.0:
            # `core.Recommendation.significant` is True whenever `stderr` is zero, which
            # is right for a real effect measured without error and wrong for this: the
            # two rosters scored identically in every simulation. Say so rather than let
            # a nothing inherit the word.
            tail = " -- identical in every simulation, so this changes nothing"
        elif not rec.significant:
            tail = " -- inside the error bar, treat as no change"
        elif lever < 0.35:
            tail = f" -- but weekly leverage is only {lever:.2f}, so most weeks are decided"
        else:
            tail = ""
        if "roster_size" in rec.tags:
            tail += (
                " -- WARNING: this move changes a roster's size, so it is priced as an "
                "add with no drop; pair it with the cut that pays for it"
            )
        note = self._counterparty_note(subject, others or {}, errors)
        return f"{head}: {odds}, {points}{tail}{note}."


# --------------------------------------------------------------------------------------
# Candidate helpers
# --------------------------------------------------------------------------------------


def evaluator_for(
    state: S.LeagueState,
    draw: Draw,
    team_id: int,
    *,
    replacement: Mapping[int, float] | float | None = None,
    **kwargs,
) -> TitleEngine:
    """The factory `decide/waivers.default_evaluator` looks for. Without it, nothing uses this.

    `waivers.default_evaluator` imports this module, asks for `evaluator_for`, and on not
    finding it logs "decide.title exists but exposes no evaluator_for(); using the local
    engine" and quietly falls back to its own `RosterSimulator` -- so the two-tier engine
    that every other surface is supposed to be priced in was never once reached from the
    waiver board. The name is a contract; it is cheaper to satisfy it than to document
    why it is unsatisfied.

    `team_id` is recorded as the state's own franchise so `subject_team` reports moves
    from the caller's side. A `None` replacement here means the same thing it means on
    `TitleEngine` -- fit the streaming level -- and not `season`'s empty seat, which is
    the setting that makes a kicker look like a first-round pick.
    """
    if state.my_team_id != team_id:
        state = replace(state, my_team_id=int(team_id))
    return TitleEngine(state, draw, replacement=replacement, **kwargs)


def swap_move(
    league_id: int, *, my_team_id: int, incoming: int, outgoing: int, from_team: int
) -> Move:
    """A one-for-one trade, which is the smallest candidate that touches two franchises."""
    return Move(
        kind=MoveKind.TRADE,
        league_id=league_id,
        players=(
            PlayerMove(player_id=incoming, from_team=from_team, to_team=my_team_id),
            PlayerMove(player_id=outgoing, from_team=my_team_id, to_team=from_team),
        ),
    )


def drop_move(league_id: int, *, team_id: int, player_id: int) -> Move:
    """Cut a player to the wire. Prices exactly what `season.leave_one_out` prices."""
    return Move(
        kind=MoveKind.ADD_DROP,
        league_id=league_id,
        players=(PlayerMove(player_id=player_id, from_team=team_id, to_team=None),),
    )


def one_for_one_candidates(
    state: S.LeagueState, team_id: int, *, opponents: Iterable[int] | None = None
) -> list[Move]:
    """Every 1-for-1 trade between `team_id` and the rest of the league.

    Here rather than in a trade surface because it is the engine's own benchmark: it is
    the only candidate set that needs no free agents in the panel, scales to thousands
    without any modelling choices of its own, and touches two franchises, which is the
    path a single-team candidate would not exercise. A real trade finder will want
    Pareto filters and multi-player packages; that is not this module's job.
    """
    mine = state.franchise(team_id).player_ids
    others = [f for f in state.franchises if f.team_id != team_id]
    if opponents is not None:
        wanted = set(opponents)
        others = [f for f in others if f.team_id in wanted]
    return [
        swap_move(
            state.league_id,
            my_team_id=team_id,
            incoming=theirs,
            outgoing=ours,
            from_team=f.team_id,
        )
        for f in others
        for theirs in f.player_ids
        for ours in mine
    ]


def sim_with_free_agents(sim, outlooks: Sequence, *, seed: int | None = None):
    """A `LeagueSim` whose pool and tensor also cover players nobody rosters yet.

    Without this no waiver claim can be priced at all: `pipeline.build` pools only
    rostered players, and a move naming anyone else indexes off the end of the tensor.
    Rebuilding the panel is safe for common random numbers because `WeeklySampler` keys
    each player's stream on his id rather than his column, so adding a free agent leaves
    every other player's season bit-identical except inside the NFL-team correlation
    block he joins. The baseline and every candidate must still be evaluated on the
    returned sim -- comparing a number from here against one from the original sim is
    comparing two universes.

    `outlooks` must cover every player in the extended pool for every remaining week,
    which is what `season.panel_for` enforces.
    """
    from .. import pipeline as P
    from ..sim.distributions import WeeklySampler

    state = sim.state
    known = set(state.pool.player_ids)
    added = [o for o in outlooks if o.player_id not in known]
    if not added:
        return sim
    existing = [
        (pid, pos, team, state.pool.name(pid))
        for pid, pos, team in zip(
            state.pool.player_ids, state.pool.position_ids, state.pool.pro_team_ids, strict=True
        )
    ]
    pool = S.PlayerPool.of(
        [*existing, *((o.player_id, o.position_id, o.pro_team_id, o.name) for o in added)]
    )
    extended = replace(state, pool=pool)
    combined = [*sim.outlooks, *added]
    seed = sim.seed if seed is None else int(seed)
    draw = WeeklySampler(S.panel_for(extended, combined), seed=seed).draw(sim.n_sims)
    return P.LeagueSim(
        league=sim.league,
        state=extended,
        draw=draw,
        outlooks=combined,
        n_sims=sim.n_sims,
        seed=seed,
    )


__all__ = [
    "BENCH_DEPTH",
    "DEFAULT_CONFIRM",
    "DEFAULT_MU_GRID",
    "DEFAULT_SIGMA_GRID",
    "DEFAULT_SURROGATE_SIMS",
    "SCREEN_RELATIVE_ERROR",
    "ScreenAgreement",
    "SurrogateFit",
    "TeamMoments",
    "TitleEngine",
    "TitleError",
    "WeekLeverage",
    "drop_move",
    "evaluator_for",
    "one_for_one_candidates",
    "sim_with_free_agents",
    "streaming_replacement",
    "swap_move",
]
