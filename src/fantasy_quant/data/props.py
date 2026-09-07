"""Sportsbook props -> component-stat means, and the devigging that makes them usable.

The one fact this whole module exists to enforce: **a posted sportsbook line is a
MEDIAN; fantasy scoring is a MEAN.** For the right-skewed stats that decide fantasy
weeks -- receiving yards above all -- those are not close. A live FanDuel receiving
ladder posted at 29.5 fits a lognormal with sigma ~= 0.84, whose mean is ~42 yards:
+42% over the line. Reading a posted line as a projection systematically under-rates
every pass-catcher in the pool, and the error is largest exactly where the market is
sharpest.

Three pieces of math earn their keep, and each has a documented way to get it wrong:

1. **Devigging.** A matched over/under pair prices to more than 1.00; the excess is the
   book's margin and must come out before the number means anything. Multiplicative,
   additive and Shin are all implemented. Default is Shin -- for a *two*-outcome market
   Shin and additive turn out to be numerically identical over the ordinary price range
   (verified against live FanDuel and Pinnacle pairs), and both shrink longshots harder
   than multiplicative, which is the empirically right correction for favourite-longshot
   bias. Shin is preferred over additive only because its solve is bounded: at extreme
   overrounds additive returns a negative probability while Shin either stays positive
   or fails to bracket, in which case we fall back to multiplicative rather than emit
   nonsense.

2. **Ladder -> distribution.** An `PLAYER_X_ALT_*` market is 11-13 rungs of
   S(k) = P(X >= k). Do NOT sum the rungs: every rung carries its own 5-8% overround
   and summing compounds it (see `naive_ladder_mean`, kept only as the counter-example).
   Instead fit a lognormal to the rungs *anchored* on the two-sided main line, which is
   the one point on the curve that can be honestly devigged. `LadderFit` returns median
   AND mean and exposes mu/sigma, because downstream needs the whole distribution --
   the simulator wants variance, not a point estimate.

   **The rung is a half-open interval and the model is continuous, so the two only line
   up under a continuity correction.** Every stat here is integer-valued, so the rung
   "k+" is the event X >= k, which is X > k - 0.5 -- and that is not a modelling
   opinion, it is what the book prices: on live FanDuel boards the ALT rung at
   ceil(line) is quoted at *exactly* the same American price as the OVER side of the
   two-sided line half a unit below it (Cooper Kupp 3+ Receptions -132, Over 2.5
   Receptions -132; 30+ Yards -114, Over 29.5 Yards -114). Evaluating the model
   survival at `k` instead of `k - 0.5` therefore asks it to fit P(X >= k+1) to a price
   for P(X >= k). On yardage ladders that is a rounding error; on receptions and
   passing TDs it is a full unit and it is fatal. Measured across 219 live ladders, the
   uncorrected fit drove the estimated per-rung overround to a nonsensical 1.21 median
   on receptions and pinned it at the 1.30 bound on 8 of 10 passing-TD ladders -- the
   nuisance parameter silently absorbing an off-by-one. With `continuity=0.5` the same
   ladders return 1.01 and 1.00, and the fit residual falls 41% on receptions
   (RMS 0.0324 -> 0.0191) and 13% on passing TDs. Nothing regresses on yards.

   Known limitation, measured on live ladders: a lognormal has no atom at zero, so it
   cannot carry a receiver's ~6% chance of being blanked and it over-states the bottom
   two or three rungs. Fits track the posted curve to an RMS of 0.015-0.026 in the body
   and drift in the tails; `rms_error` is exported so a ladder that stops looking
   lognormal is visible rather than silent. The mean is dominated by the right tail, so
   the left-tail misfit costs little -- but do not read `survival(5)` as a real number.

3. **Anytime TD.** Devig FIRST, then lambda = -ln(1 - p). The reverse order looks
   almost right and is not: it under-rates goal-line backs, whose prices are short
   enough that the log transform and the vig removal do not commute. At +150 the naive
   read is 0.400 expected TDs; the correct one is 0.511, +27.7%.

Every source here is an undocumented free surface that can close without notice, so each
is independently optional: a dead book raises `PropsError`, `collect_*` swallows it into
a `PropsBundle.errors` entry, and the rest of the pipeline keeps running.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from ..espn.client import EspnClient, EspnError

log = logging.getLogger(__name__)


class PropsError(RuntimeError):
    """A prop source failed or returned something we refuse to trust."""


# --------------------------------------------------------------------------------------
# Canonical component-stat vocabulary
# --------------------------------------------------------------------------------------
# Named to match nflverse's weekly stat columns so a props projection joins straight onto
# the rest of the corpus. Ensembling happens at the component level, never at the points
# level -- that is what lets one projection serve leagues with different scoring.

PASSING_YARDS: Final = "passing_yards"
PASSING_TDS: Final = "passing_tds"
PASSING_ATTEMPTS: Final = "attempts"
PASSING_COMPLETIONS: Final = "completions"
INTERCEPTIONS: Final = "passing_interceptions"
RUSHING_YARDS: Final = "rushing_yards"
RUSHING_ATTEMPTS: Final = "carries"
RUSHING_TDS: Final = "rushing_tds"
RECEPTIONS: Final = "receptions"
RECEIVING_YARDS: Final = "receiving_yards"
RECEIVING_TDS: Final = "receiving_tds"
ANYTIME_TDS: Final = "anytime_tds"
FANTASY_POINTS: Final = "fantasy_points"
KICKING_POINTS: Final = "kicking_points"
FIELD_GOALS_MADE: Final = "field_goals_made"
EXTRA_POINTS_MADE: Final = "extra_points_made"


# --------------------------------------------------------------------------------------
# Odds arithmetic
# --------------------------------------------------------------------------------------

DEVIG_METHODS: Final = ("shin", "multiplicative", "additive", "power")
DEFAULT_DEVIG: Final = "shin"


def american_to_implied(odds: float) -> float:
    """American odds -> raw implied probability (still carrying the book's margin)."""
    odds = float(odds)
    if odds == 0 or not math.isfinite(odds):
        raise ValueError(f"not valid American odds: {odds!r}")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def implied_to_american(prob: float) -> float:
    """Inverse of `american_to_implied`. Exact round trip; useful mostly for tests."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability out of range: {prob!r}")
    if prob > 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def _multiplicative(raw: np.ndarray) -> np.ndarray:
    return raw / raw.sum()


def _additive(raw: np.ndarray) -> np.ndarray:
    """Spread the overround equally in probability space (a "balanced book")."""
    return raw - (raw.sum() - 1.0) / raw.size


def _power(raw: np.ndarray) -> np.ndarray:
    """Solve for k with sum(p_i ** k) == 1. Shrinks longshots, like Shin."""

    def gap(k: float) -> float:
        return float(np.sum(raw**k) - 1.0)

    return raw ** brentq(gap, 0.2, 8.0, xtol=1e-12) if gap(1.0) > 0 else raw / raw.sum()


def _shin(raw: np.ndarray) -> np.ndarray:
    """Shin (1993): back out the insider-trading fraction z, then invert.

    q_i = (sqrt(z^2 + 4(1-z) p_i^2 / P) - z) / (2(1-z)),  z chosen so sum(q_i) == 1.

    There is no root for pathological books (tiny longshot against a huge overround);
    the caller falls back rather than pretending.
    """
    total = raw.sum()

    def q(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * raw * raw / total) - z) / (2.0 * (1.0 - z))

    def gap(z: float) -> float:
        return float(q(z).sum() - 1.0)

    return q(brentq(gap, 1e-12, 1.0 - 1e-9, xtol=1e-14))


_DEVIG_FUNCS: Final[dict[str, Callable[[np.ndarray], np.ndarray]]] = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
}


def devig(raw_probs: Sequence[float], method: str = DEFAULT_DEVIG) -> list[float]:
    """Strip the overround from a set of raw implied probabilities.

    Only valid over an exhaustive, mutually exclusive outcome set -- an over/under pair,
    a moneyline. Anytime-TD scorers are neither, so never route them through here.
    """
    if method not in _DEVIG_FUNCS:
        raise ValueError(f"unknown devig method {method!r}; expected one of {DEVIG_METHODS}")
    raw = np.asarray(raw_probs, dtype=float)
    if raw.size < 2:
        raise ValueError("devigging needs at least two outcomes")
    if np.any(raw <= 0.0):
        raise ValueError(f"non-positive implied probability in {list(raw_probs)!r}")
    if raw.sum() <= 1.0:
        # Underround: someone is offering an arb, or (far likelier) a stale price. There
        # is no margin to *remove*, and Shin/additive/power are all derived from a
        # positive overround, so they are not applicable. We still have to return a
        # distribution, so the probabilities are scaled UP proportionally -- the shares
        # are preserved, the level is not left alone.
        return [float(x) for x in _multiplicative(raw)]

    try:
        fair = _DEVIG_FUNCS[method](raw)
    except (ValueError, RuntimeError) as exc:
        log.warning("%s devig failed on %s (%s); falling back to multiplicative", method, raw, exc)
        fair = _multiplicative(raw)
    if np.any(fair <= 0.0) or not np.all(np.isfinite(fair)):
        log.warning("%s devig produced non-probabilities on %s; falling back", method, raw)
        fair = _multiplicative(raw)
    return [float(x) for x in fair / fair.sum()]


@dataclass(frozen=True, slots=True)
class TwoSided:
    """A devigged over/under pair. `p_over` is the vig-free P(X > line)."""

    line: float
    over_odds: float
    under_odds: float
    p_over: float
    p_under: float
    overround: float
    method: str

    @property
    def vig_pct(self) -> float:
        return 100.0 * (self.overround - 1.0)


def devig_two_sided(
    line: float,
    over_odds: float,
    under_odds: float,
    method: str = DEFAULT_DEVIG,
) -> TwoSided:
    """Devig a matched over/under pair into an honest point on the survival curve."""
    raw = [american_to_implied(over_odds), american_to_implied(under_odds)]
    p_over, p_under = devig(raw, method)
    return TwoSided(
        line=float(line),
        over_odds=float(over_odds),
        under_odds=float(under_odds),
        p_over=p_over,
        p_under=p_under,
        overround=float(sum(raw)),
        method=method,
    )


# --------------------------------------------------------------------------------------
# Anytime touchdowns
# --------------------------------------------------------------------------------------


def expected_touchdowns(
    anytime_odds: float,
    *,
    overround: float = 1.0,
    method: str = DEFAULT_DEVIG,
    no_td_odds: float | None = None,
) -> float:
    """Anytime-TD price -> expected touchdowns, under a Poisson count model.

    Order matters and is the documented trap. P(X >= 1) = 1 - exp(-lambda), so
    lambda = -ln(1 - p) -- but `p` must already be vig-free. Devigging afterwards
    (dividing lambda by the overround) is not the same operation and understates
    short-priced goal-line backs.

    Anytime-TD scorers are not a mutually exclusive outcome set, so there is nothing to
    normalize against. Supply the margin one of two ways: `no_td_odds` if the book posts
    the "no touchdown" side (then it is a real two-way devig), otherwise `overround` as
    a flat multiplicative shading factor -- 1.06 to 1.10 is typical for the market.

    With no margin supplied at all this returns the pure Poisson conversion, which is
    still the documented +27.7% over reading the implied probability as an expectation:
    +150 -> 0.400 implied -> 0.511 expected TDs.
    """
    if no_td_odds is not None:
        p = devig_two_sided(0.5, anytime_odds, no_td_odds, method).p_over
    else:
        if overround <= 0.0:
            raise ValueError(f"overround must be positive, got {overround!r}")
        p = american_to_implied(anytime_odds) / overround
    if not 0.0 < p < 1.0:
        raise ValueError(f"devigged anytime-TD probability out of range: {p!r}")
    return -math.log1p(-p)


def touchdown_probability(expected_tds: float) -> float:
    """Inverse of the Poisson conversion: lambda -> P(at least one TD)."""
    if expected_tds < 0.0:
        raise ValueError(f"expected touchdowns cannot be negative: {expected_tds!r}")
    return -math.expm1(-expected_tds)


def split_anytime_tds(expected_tds: float, rush_share: float) -> tuple[float, float]:
    """Split an anytime-TD expectation into (rushing, receiving).

    Anytime TD is rush+rec combined; leagues almost always score them identically, so
    this only matters for TE/RB-premium formats. `rush_share` is the caller's estimate
    from usage -- we have no market signal for the split.
    """
    if not 0.0 <= rush_share <= 1.0:
        raise ValueError(f"rush_share must be in [0, 1], got {rush_share!r}")
    return expected_tds * rush_share, expected_tds * (1.0 - rush_share)


# --------------------------------------------------------------------------------------
# ALT ladder -> fitted distribution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LadderRung:
    """One rung of an alternate-line ladder: P(X >= threshold) at this price."""

    threshold: float
    american: float

    @property
    def implied(self) -> float:
        """Raw implied probability. Carries the rung's own 5-8% margin -- do not sum."""
        return american_to_implied(self.american)


@dataclass(frozen=True, slots=True)
class LadderFit:
    """A lognormal fitted to a rung ladder, anchored on a devigged two-sided line.

    `mu` and `sigma` are the lognormal parameters, so downstream can draw from the whole
    distribution rather than passing a point estimate into a simulator that needs shape.
    `overround` is the single multiplicative shading the fit attributes to the rungs; it
    is estimated jointly rather than assumed, and hitting `max_overround` is a signal the
    ladder does not look lognormal. `max_overround` is carried on the fit precisely so
    that saturation is *detectable* -- see `overround_pinned`. A caller that has to
    hard-code the bound to notice the fit hit it has not been told anything.

    `continuity` is the half-unit correction applied to the rungs: the model's value for
    the rung "k+" is `survival(k - continuity)`, not `survival(k)`. Use `rung_survival`
    rather than `survival` whenever you are comparing the fit to a posted rung.
    """

    mu: float
    sigma: float
    overround: float
    n_rungs: int
    rms_error: float
    anchor_line: float | None = None
    anchor_prob: float | None = None
    continuity: float = 0.5
    max_overround: float = 1.30

    @property
    def overround_pinned(self) -> bool:
        """True when the jointly-estimated shading saturated its bound.

        A pinned overround means the least-squares solve ran out of room rather than
        converging: the ladder is not lognormal, or it is being evaluated under the
        wrong rung convention. Either way the mean is not trustworthy.
        """
        return self.overround >= self.max_overround - 1e-6

    @property
    def median(self) -> float:
        return math.exp(self.mu)

    @property
    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2.0)

    @property
    def variance(self) -> float:
        return math.expm1(self.sigma**2) * math.exp(2.0 * self.mu + self.sigma**2)

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)

    @property
    def mean_over_median(self) -> float:
        """The skew premium. ~1.4 on receiving yards; 1.0 would mean symmetric."""
        return math.exp(self.sigma**2 / 2.0)

    def survival(self, x: float) -> float:
        """P(X > x)."""
        if x <= 0.0:
            return 1.0
        return float(norm.sf((math.log(x) - self.mu) / self.sigma))

    def rung_survival(self, threshold: float) -> float:
        """The model's fair probability for the posted rung "threshold+", i.e. P(X >= k).

        This is `survival(k - continuity)`, and it is the only apples-to-apples
        comparison against a rung's implied price. Comparing a rung to `survival(k)`
        compares P(X >= k) against P(X >= k + 1).
        """
        return self.survival(float(threshold) - self.continuity)

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {q!r}")
        return float(math.exp(self.mu + self.sigma * norm.ppf(q)))


def _clean_rungs(
    rungs: Iterable[LadderRung], continuity: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, deduped rungs as (thresholds, raw probabilities), usable after `continuity`.

    A lognormal has support on x > 0, so a "0+ yards" rung carries no information and
    would drag the fit; duplicated thresholds (two books' rungs merged, or a re-post)
    would silently double-weight a point. A rung is also dropped when the continuity
    correction would push it to or below zero, where the log is undefined.
    """
    best: dict[float, float] = {}
    for r in rungs:
        k = float(r.threshold)
        if not math.isfinite(k) or k <= 0.0 or k - continuity <= 0.0:
            continue
        best[k] = r.implied
    ks = np.array(sorted(best), dtype=float)
    ps = np.array([best[k] for k in ks], dtype=float)
    return ks, np.clip(ps, 1e-6, 1.0 - 1e-6)


# Bounds on the fitted lognormal shape. Live NFL ladders sit at sigma 0.28-1.35, so a
# solve that reaches either bound has not converged onto anything real -- it has run out
# of room. mean = median * exp(sigma^2/2), so sigma at the ceiling is a 90x mean.
_SIGMA_BOUNDS: Final = (0.05, 3.0)


def fit_lognormal_ladder(
    rungs: Sequence[LadderRung],
    *,
    anchor_line: float | None = None,
    anchor_prob: float = 0.5,
    anchor_weight: float = 5.0,
    max_overround: float = 1.30,
    min_rungs: int = 4,
    continuity: float = 0.5,
) -> LadderFit:
    """Fit S(x) = P(X > x) as a lognormal, anchored on the two-sided main line.

    The rungs give the *shape* and the anchor gives the *level*. That division of labour
    is the whole trick: a one-sided rung price cannot be devigged on its own, but the
    matched over/under at the main line can be, so it is the only vig-free point on the
    curve and it is where the level gets pinned.

    `continuity` is the half-unit correction that makes those two things commensurable.
    The rung "k+" prices X >= k; the anchor at line L prices X > L. Every stat these
    ladders cover is integer-valued, so X >= k is X > k - 0.5, and the book agrees --
    it quotes the rung at ceil(L) and the OVER at L at the identical American price.
    The rung residual is therefore taken at `k - continuity`, while the anchor, already
    posted on a half-point, is taken as-is. Pass `continuity=0.0` to restore the naive
    convention; expect the fitted `overround` to absorb the resulting bias and pin
    against `max_overround` on receptions and passing TDs when you do.

    `anchor_weight` is deliberately finite rather than a hard constraint. The main line
    is the best single point but not a sacred one -- FanDuel's two-sided line and its ALT
    ladder are separate markets that can drift apart intraday, and a hard pin lets a
    stale main line override 13 fresh rungs. At the default weight the fitted median
    lands within ~1% of the posted line while the rungs still get a vote.

    Raises `PropsError` on a ladder that does not identify a lognormal at all -- too few
    usable rungs, or a solve that ends with sigma against a bound. Returning a shape the
    solver never converged onto would put a 2,000-yard receiving mean into the pool
    wearing the same type as a good fit.

    Returns median and mean; on receiving yards expect mean/median ~= 1.4.
    """
    if not 0.0 <= continuity < 1.0:
        raise ValueError(f"continuity must be in [0, 1), got {continuity!r}")
    if max_overround < 1.0:
        raise ValueError(f"max_overround must be at least 1.0, got {max_overround!r}")
    ks, ps = _clean_rungs(rungs, continuity)
    if ks.size < min_rungs:
        raise PropsError(f"ladder has {ks.size} usable rungs, need {min_rungs}")
    if anchor_line is not None and anchor_line <= 0.0:
        raise ValueError(f"anchor_line must be positive, got {anchor_line!r}")
    if not 0.0 < anchor_prob < 1.0:
        raise ValueError(f"anchor_prob must be in (0, 1), got {anchor_prob!r}")

    # P(X >= k) == P(X > k - 0.5) for an integer-valued X; see the module docstring.
    log_ks = np.log(ks - continuity)
    log_anchor = math.log(anchor_line) if anchor_line is not None else None

    def residuals(theta: np.ndarray) -> np.ndarray:
        mu, sigma, c = theta
        # The rungs are shaded by `c`; the anchor is already vig-free, so it is not.
        res = c * norm.sf((log_ks - mu) / sigma) - ps
        if log_anchor is None:
            return res
        anchor = float(norm.sf((log_anchor - mu) / sigma)) - anchor_prob
        return np.append(res, anchor_weight * anchor)

    # Seed mu at the anchor if we have one, else at the rung that straddles 50%.
    mu0 = (
        math.log(anchor_line)
        if anchor_line is not None
        else float(np.interp(0.5, ps[::-1], log_ks[::-1]))
    )
    lo_sigma, hi_sigma = _SIGMA_BOUNDS
    solution = least_squares(
        residuals,
        # The seed has to sit inside the box: a caller who narrows `max_overround` below
        # the seed would otherwise get an opaque scipy "initial guess outside bounds".
        x0=np.array([mu0, 0.85, min(1.05, max_overround)]),
        bounds=(
            np.array([math.log(1e-4), lo_sigma, 1.0]),
            np.array([math.log(1e5), hi_sigma, max_overround]),
        ),
    )
    mu, sigma, c = (float(v) for v in solution.x)
    if not solution.success:
        raise PropsError(f"ladder fit did not converge ({solution.message})")
    if sigma <= lo_sigma + 1e-6 or sigma >= hi_sigma - 1e-6:
        raise PropsError(
            f"ladder fit degenerate: sigma hit its bound at {sigma:.3f} "
            f"(bounds {_SIGMA_BOUNDS}). The rungs do not describe a lognormal -- a "
            f"flat, inverted or suspended market looks exactly like this."
        )
    rms = float(np.sqrt(np.mean(np.square(c * norm.sf((log_ks - mu) / sigma) - ps))))
    return LadderFit(
        mu=mu,
        sigma=sigma,
        overround=c,
        n_rungs=int(ks.size),
        rms_error=rms,
        anchor_line=anchor_line,
        anchor_prob=anchor_prob if anchor_line is not None else None,
        continuity=continuity,
        max_overround=max_overround,
    )


def naive_ladder_mean(rungs: Sequence[LadderRung]) -> float:
    """E[X] by summing the raw rungs. **This is the bug**, kept as the counter-example.

    Integrating the posted survival function directly inherits every rung's margin at
    once, and the margins do not cancel: through the body of the ladder -- where the mass
    is -- the posted curve runs above the fair one, so the area under it is biased high.
    On the live Kupp ladder this reads 43.8 against a fitted mean of 42.1. Compare the
    two in tests and in the dashboard's diagnostics.
    """
    ks, ps = _clean_rungs(rungs)
    if ks.size < 2:
        raise PropsError("need at least two rungs to integrate")
    total = float(ks[0])  # S(x) == 1 below the first rung
    total += float(np.sum(ps[:-1] * np.diff(ks)))
    # Extend the tail geometrically off the last two rungs; truncating there would
    # understate a heavy right tail and confuse the comparison we are trying to make.
    step = float(ks[-1] - ks[-2])
    ratio = float(ps[-1] / ps[-2]) if ps[-2] > 0.0 else 0.0
    if 0.0 < ratio < 1.0:
        total += step * float(ps[-1]) / (1.0 - ratio)
    return total


def fit_ladder_with_line(
    rungs: Sequence[LadderRung],
    two_sided: TwoSided | None,
    **kwargs: Any,
) -> LadderFit:
    """`fit_lognormal_ladder` with the anchor taken from a devigged two-sided market."""
    if two_sided is None:
        return fit_lognormal_ladder(rungs, **kwargs)
    return fit_lognormal_ladder(
        rungs,
        anchor_line=two_sided.line,
        anchor_prob=two_sided.p_over,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# Components -> fantasy points
# --------------------------------------------------------------------------------------


def project_fantasy_points(
    components: Mapping[str, float],
    scorer: Callable[[Mapping[str, float]], float],
    *,
    rush_share: float | None = None,
) -> float:
    """Score a bag of component means with a caller-supplied scoring function.

    The scorer is injected rather than imported so this module composes with
    `espn/scoring.py` (or any per-league scorer) without depending on it -- a player's
    value is a function of (player, league settings), never a global number.

    Scoring is linear in the component stats, so E[points] = points(E[components]) and
    plugging means straight in is exact. That is only true because we took the trouble
    to recover means; feeding medians here would propagate the median/mean error into
    every downstream decision.

    `rush_share` expands an `anytime_tds` entry into rushing/receiving TDs first, for
    scorers that price them differently.
    """
    stats = dict(components)
    if rush_share is not None and ANYTIME_TDS in stats:
        rush, rec = split_anytime_tds(stats.pop(ANYTIME_TDS), rush_share)
        stats[RUSHING_TDS] = stats.get(RUSHING_TDS, 0.0) + rush
        stats[RECEIVING_TDS] = stats.get(RECEIVING_TDS, 0.0) + rec
    return float(scorer(stats))


# --------------------------------------------------------------------------------------
# Shared HTTP
# --------------------------------------------------------------------------------------

_HEADERS: Final = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_MIN_INTERVAL_S: Final = 0.25


class _Book:
    """Polite JSON getter shared by the sportsbook adapters."""

    def __init__(self, timeout: float = 30.0, client: httpx.Client | None = None) -> None:
        self._owns = client is None
        self._client = client or httpx.Client(headers=_HEADERS, timeout=timeout)
        self._last = 0.0

    def close(self) -> None:
        if self._owns:
            self._client.close()

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        elapsed = time.monotonic() - self._last
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        self._last = time.monotonic()
        try:
            resp = self._client.get(url, params=dict(params or {}))
        except httpx.HTTPError as exc:
            raise PropsError(f"{url}: {exc}") from exc
        if resp.status_code != 200:
            raise PropsError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise PropsError(f"{url}: response was not JSON ({resp.text[:120]!r})") from exc


# --------------------------------------------------------------------------------------
# FanDuel
# --------------------------------------------------------------------------------------

FANDUEL_KEY: Final = "FhMFpcPWXMeyZxOx"
# Each state runs its own host behind the same code. Geo, maintenance and market
# availability all differ between them, so try several before declaring the book dead.
FANDUEL_HOSTS: Final = ("nj", "va", "oh", "mi", "pa", "az", "co")

# Kebab-case slugs, NOT the numeric ids in `layout.tabs`. A wrong slug is served with
# HTTP 200 and a plausible-looking body, so status is worthless as a health check.
FANDUEL_TABS: Final = (
    "popular",
    "passing-props",
    "receiving-props",
    "rushing-props",
    "td-scorer-props",
)

# What each tab MUST contain to count as served, and the reason unknown slugs are
# rejected outright rather than trusted.
#
# The documented failure mode is "a bad slug returns 200 with empty attachments". Live
# traffic says something worse: a bogus slug returns 200 with a small *non-empty*
# fallback set of game markets (moneyline, spread, a quarter total), and a `layout` block
# byte-identical to a good response. There is nothing in the body that names the tab that
# was actually served, so `len(markets) > 0` passes on a typo and no amount of inspecting
# the response can tell you which tab you got. The only workable check is a per-slug
# expectation of what that tab is supposed to contain.
TAB_REQUIRED_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "popular": ("PLAYER_X_",),
    "passing-props": ("PLAYER_X_PASSING", "PLAYER_X_ALT_PASSING"),
    "receiving-props": ("PLAYER_X_RECEIVING", "PLAYER_X_RECEPTIONS", "PLAYER_X_ALT_"),
    "rushing-props": ("PLAYER_X_RUSHING", "PLAYER_X_ALT_RUSHING"),
    "td-scorer-props": ("ANY_TIME_TOUCHDOWN_SCORER",),
}

# marketType with the ALT infix and the HIGH/MEDIUM/LOW tier suffix stripped.
FANDUEL_STATS: Final[dict[str, str]] = {
    "PLAYER_X_PASSING_YARDS": PASSING_YARDS,
    "PLAYER_X_PASSING_TOUCHDOWNS": PASSING_TDS,
    "PLAYER_X_PASSING_ATTEMPTS": PASSING_ATTEMPTS,
    "PLAYER_X_PASSING_COMPLETIONS": PASSING_COMPLETIONS,
    "PLAYER_X_INTERCEPTIONS": INTERCEPTIONS,
    "PLAYER_X_RUSHING_YARDS": RUSHING_YARDS,
    "PLAYER_X_RUSHING_ATTEMPTS": RUSHING_ATTEMPTS,
    "PLAYER_X_RECEIVING_YARDS": RECEIVING_YARDS,
    "PLAYER_X_RECEPTIONS": RECEPTIONS,
}

_TIER_SUFFIXES: Final = ("_HIGH", "_MEDIUM", "_LOW")
_RUNG_RE: Final = re.compile(r"(\d+(?:\.\d+)?)\s*\+")


def fanduel_stat(market_type: str) -> str | None:
    """Canonical stat for a FanDuel marketType, or None if we do not model it."""
    key = market_type
    for suffix in _TIER_SUFFIXES:
        key = key.removesuffix(suffix)
    key = key.replace("_ALT_", "_", 1)
    return FANDUEL_STATS.get(key)


def _rung_threshold(runner_name: str) -> float | None:
    """`'Cooper Kupp 30+ Yards'` -> 30.0. None when the runner is not a rung."""
    match = _RUNG_RE.search(runner_name or "")
    return float(match.group(1)) if match else None


def _runner_odds(runner: Mapping[str, Any]) -> float | None:
    odds = (runner.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}
    value = odds.get("americanOddsInt", odds.get("americanOdds"))
    return float(value) if value not in (None, 0) else None


def _market_player(market: Mapping[str, Any]) -> str:
    """`'Cooper Kupp - Alt Receiving Yds'` -> `'Cooper Kupp'`."""
    name = str(market.get("marketName") or "")
    return name.split(" - ", 1)[0].strip()


@dataclass(frozen=True, slots=True)
class FanDuelEvent:
    event_id: int
    name: str
    open_date: str

    @property
    def teams(self) -> tuple[str, str] | None:
        if " @ " not in self.name:
            return None
        away, home = self.name.split(" @ ", 1)
        return away.strip(), home.strip()


@dataclass(frozen=True, slots=True)
class PropLine:
    """A two-sided player prop: the market's own median, plus its devigged probability."""

    book: str
    event_id: int
    player: str
    stat: str
    line: float
    over_odds: float
    under_odds: float
    max_risk_stake: float | None = None

    def devigged(self, method: str = DEFAULT_DEVIG) -> TwoSided:
        return devig_two_sided(self.line, self.over_odds, self.under_odds, method)


@dataclass(frozen=True, slots=True)
class PropLadder:
    """An ALT market: a full survival function, one rung per threshold."""

    book: str
    event_id: int
    player: str
    stat: str
    rungs: tuple[LadderRung, ...]


@dataclass(frozen=True, slots=True)
class AnytimeTd:
    book: str
    event_id: int
    player: str
    american: float


@dataclass(frozen=True, slots=True)
class FanDuelEventProps:
    event_id: int
    lines: tuple[PropLine, ...]
    ladders: tuple[PropLadder, ...]
    touchdowns: tuple[AnytimeTd, ...]
    tabs_seen: tuple[str, ...]


class FanDuelProps:
    """Reader for FanDuel's public `sbapi` event pages -- the ALT ladders live here."""

    def __init__(
        self,
        hosts: Sequence[str] = FANDUEL_HOSTS,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not hosts:
            raise ValueError("need at least one FanDuel state host")
        self._hosts = list(hosts)
        self._book = _Book(timeout=timeout, client=client)

    def close(self) -> None:
        self._book.close()

    def __enter__(self) -> FanDuelProps:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        """Try each state host in turn; promote a working one to the front."""
        errors: list[str] = []
        for index, host in enumerate(list(self._hosts)):
            url = f"https://sbapi.{host}.sportsbook.fanduel.com/api/{path}"
            try:
                payload = self._book.get_json(url, {"_ak": FANDUEL_KEY, **params})
            except PropsError as exc:
                errors.append(f"{host}: {exc}")
                continue
            if index:
                self._hosts.insert(0, self._hosts.pop(index))
            return payload
        raise PropsError(f"every FanDuel host failed for {path}: {'; '.join(errors)}")

    def events(self) -> list[FanDuelEvent]:
        """Scheduled NFL games. Filters out futures/specials, which carry no player props."""
        payload = self._get(
            "content-managed-page",
            {"page": "CUSTOM", "customPageId": "nfl", "timezone": "America/New_York"},
        )
        raw = ((payload or {}).get("attachments") or {}).get("events") or {}
        events = [
            FanDuelEvent(
                event_id=int(e["eventId"]),
                name=str(e.get("name") or ""),
                open_date=str(e.get("openDate") or ""),
            )
            for e in raw.values()
            if e.get("eventId") is not None and " @ " in str(e.get("name") or "")
        ]
        if not events:
            raise PropsError("FanDuel returned no NFL games; the page or the key has moved")
        return sorted(events, key=lambda e: e.open_date)

    def markets(
        self,
        event_id: int,
        tab: str,
        required: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Markets for one tab, with the served-vs-typo check applied.

        Raises rather than returning empty. A silently wrong tab is how a typo'd slug and
        a geo block both look, and swallowing either poisons every downstream projection
        with quietly missing players instead of an error.

        `required` is a tuple of marketType prefixes at least one market must match;
        it defaults to `TAB_REQUIRED_PREFIXES[tab]`. An unknown slug with no expectation
        supplied is refused, because the response body cannot tell us whether the slug
        resolved -- pass `required=()` to accept a tab on non-emptiness alone.
        """
        if required is None:
            required = TAB_REQUIRED_PREFIXES.get(tab)
        if required is None:
            raise PropsError(
                f"unknown FanDuel tab slug {tab!r}. Slugs are kebab-case and a wrong one "
                f"is served as HTTP 200 with plausible-looking game markets, so we will "
                f"not fetch a slug we cannot validate. Known slugs: "
                f"{sorted(TAB_REQUIRED_PREFIXES)}. Pass required=(...) to add one, or "
                f"required=() to accept it on non-emptiness alone."
            )

        payload = self._get("event-page", {"eventId": event_id, "tab": tab})
        raw = ((payload or {}).get("attachments") or {}).get("markets") or {}
        markets = list(raw.values())
        if not markets:
            raise PropsError(f"FanDuel event {event_id} tab {tab!r}: empty attachments")
        prefixes = tuple(required)
        if prefixes and not any(str(m.get("marketType", "")).startswith(prefixes) for m in markets):
            raise PropsError(
                f"FanDuel event {event_id} tab {tab!r}: {len(markets)} markets but none "
                f"matching {prefixes} -- a bad slug is served as HTTP 200 with a "
                f"fallback set of game markets, so this is almost certainly a typo, a "
                f"geo block, or a market that has not opened yet."
            )
        return markets

    def event_props(
        self,
        event_id: int,
        tabs: Sequence[str] = FANDUEL_TABS,
        *,
        min_rungs: int = 4,
    ) -> FanDuelEventProps:
        """Every player prop we model for one game, deduped across tabs."""
        lines: dict[tuple[str, str], PropLine] = {}
        ladders: dict[tuple[str, str], PropLadder] = {}
        touchdowns: dict[str, AnytimeTd] = {}
        seen: list[str] = []
        errors: list[str] = []

        for tab in tabs:
            try:
                markets = self.markets(event_id, tab)
            except PropsError as exc:
                # One dead tab must not take the game down; ladders overlap across tabs.
                errors.append(str(exc))
                log.warning("%s", exc)
                continue
            seen.append(tab)
            for market in markets:
                self._absorb(event_id, market, lines, ladders, touchdowns, min_rungs)

        if not seen:
            raise PropsError(f"FanDuel event {event_id}: no tab served: {'; '.join(errors)}")
        return FanDuelEventProps(
            event_id=event_id,
            lines=tuple(lines.values()),
            ladders=tuple(ladders.values()),
            touchdowns=tuple(touchdowns.values()),
            tabs_seen=tuple(seen),
        )

    @staticmethod
    def _absorb(
        event_id: int,
        market: Mapping[str, Any],
        lines: dict[tuple[str, str], PropLine],
        ladders: dict[tuple[str, str], PropLadder],
        touchdowns: dict[str, AnytimeTd],
        min_rungs: int,
    ) -> None:
        market_type = str(market.get("marketType") or "")
        runners = market.get("runners") or []

        if market_type == "ANY_TIME_TOUCHDOWN_SCORER":
            for runner in runners:
                odds = _runner_odds(runner)
                name = str(runner.get("runnerName") or "").strip()
                if odds is not None and name:
                    touchdowns[name] = AnytimeTd("fanduel", event_id, name, odds)
            return

        stat = fanduel_stat(market_type)
        if stat is None:
            return
        player = _market_player(market)
        if not player:
            return
        key = (player, stat)

        if "_ALT_" in market_type:
            rungs = [
                LadderRung(threshold, odds)
                for runner in runners
                if (threshold := _rung_threshold(str(runner.get("runnerName") or ""))) is not None
                and (odds := _runner_odds(runner)) is not None
            ]
            # The same ALT market appears on several tabs and the copies are not always
            # equally complete -- `popular` sometimes carries a truncated ladder. More
            # rungs is a strictly better fit, so the longest copy wins.
            existing = ladders.get(key)
            if len(rungs) >= min_rungs and (existing is None or len(rungs) > len(existing.rungs)):
                ladders[key] = PropLadder("fanduel", event_id, player, stat, tuple(rungs))
            return

        # Two-sided: one runner per side, both at the same handicap.
        sides = {
            str((runner.get("result") or {}).get("type") or "").upper(): runner
            for runner in runners
        }
        over, under = sides.get("OVER"), sides.get("UNDER")
        if over is None or under is None:
            return
        over_odds, under_odds = _runner_odds(over), _runner_odds(under)
        handicap = over.get("handicap")
        if over_odds is None or under_odds is None or handicap is None:
            return
        if key not in lines:
            lines[key] = PropLine(
                book="fanduel",
                event_id=event_id,
                player=player,
                stat=stat,
                line=float(handicap),
                over_odds=over_odds,
                under_odds=under_odds,
            )


# --------------------------------------------------------------------------------------
# Underdog -- a market-set fantasy projection, no modelling required
# --------------------------------------------------------------------------------------

UNDERDOG_URL: Final = "https://api.underdogfantasy.com/beta/v6/over_under_lines"
UNDERDOG_STATS: Final[dict[str, str]] = {
    "Fantasy Points": FANTASY_POINTS,
    "Passing Yards": PASSING_YARDS,
    "Pass TDs": PASSING_TDS,
    "Passing Attempts": PASSING_ATTEMPTS,
    "Completions": PASSING_COMPLETIONS,
    "Interceptions": INTERCEPTIONS,
    "Rushing Yards": RUSHING_YARDS,
    "Rush Attempts": RUSHING_ATTEMPTS,
    "Receiving Yards": RECEIVING_YARDS,
    "Receptions": RECEPTIONS,
}


@dataclass(frozen=True, slots=True)
class UnderdogLine:
    """One Underdog over/under. Fantasy-points lines are HALF-PPR -- rescale before use."""

    player: str
    position: str
    team_id: str
    stat: str
    display_stat: str
    line: float
    over_odds: float | None
    under_odds: float | None
    status: str

    def devigged(self, method: str = DEFAULT_DEVIG) -> TwoSided | None:
        if self.over_odds is None or self.under_odds is None:
            return None
        return devig_two_sided(self.line, self.over_odds, self.under_odds, method)


def fetch_underdog(
    *,
    cache_path: Path | None = None,
    max_age_s: float = 900.0,
    book: _Book | None = None,
) -> dict[str, Any]:
    """Raw Underdog payload, cached on disk -- it is ~16 MB and rarely worth re-pulling."""
    if cache_path is not None and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_s:
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                log.warning("underdog cache at %s is corrupt; refetching", cache_path)

    owned = book is None
    book = book or _Book()
    try:
        payload = book.get_json(UNDERDOG_URL)
    finally:
        if owned:
            book.close()

    if not isinstance(payload, dict) or not payload.get("over_under_lines"):
        raise PropsError("Underdog returned no over_under_lines")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(cache_path)
    return payload


def parse_underdog(
    payload: Mapping[str, Any],
    *,
    sport: str = "NFL",
    stats: Iterable[str] | None = ("Fantasy Points",),
) -> list[UnderdogLine]:
    """Underdog payload -> typed lines for one sport.

    The sport filter is not optional. This endpoint serves every sport Underdog runs
    from one URL -- a single pull carried NFL, MLB and CFB lines interleaved, and
    "Fantasy Points" exists in all three. Filtering on `display_stat` alone silently
    mixes baseball players into the football projections, and since Underdog exposes
    only its own UUIDs, the resulting rows name-match against nothing and just vanish.
    """
    wanted = set(stats) if stats is not None else None
    players = {p["id"]: p for p in payload.get("players") or [] if p.get("id")}
    appearances = {a["id"]: a for a in payload.get("appearances") or [] if a.get("id")}

    out: list[UnderdogLine] = []
    for line in payload.get("over_under_lines") or []:
        over_under = line.get("over_under") or {}
        stat_block = over_under.get("appearance_stat") or {}
        display = str(stat_block.get("display_stat") or "")
        if wanted is not None and display not in wanted:
            continue
        appearance = appearances.get(stat_block.get("appearance_id"))
        player = players.get((appearance or {}).get("player_id"))
        if player is None or str(player.get("sport_id") or "") != sport:
            continue
        try:
            value = float(line.get("stat_value"))
        except (TypeError, ValueError):
            continue

        prices: dict[str, float] = {}
        for option in line.get("options") or []:
            try:
                prices[str(option.get("choice") or "")] = float(option.get("american_price"))
            except (TypeError, ValueError):
                continue
        name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        out.append(
            UnderdogLine(
                player=name,
                position=str(player.get("position_name") or ""),
                team_id=str(player.get("team_id") or ""),
                stat=UNDERDOG_STATS.get(display, display),
                display_stat=display,
                line=value,
                over_odds=prices.get("higher"),
                under_odds=prices.get("lower"),
                status=str(line.get("status") or ""),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# ESPN propBets -- line movement, pre-joined to ESPN athlete ids
# --------------------------------------------------------------------------------------

ESPN_PROPS_URL: Final = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{gid}"
    "/competitions/{gid}/odds/100/propBets"
)

# Full-game types only. The period-restricted variants ("... - 1st Half") are real
# markets but score nothing in fantasy, and folding them in would double-count.
ESPN_PROP_STATS: Final[dict[str, str]] = {
    "Total Passing Yards (incl. overtime)": PASSING_YARDS,
    "Total Passing Touchdowns (incl. overtime)": PASSING_TDS,
    "Total Passing Attempts (incl. overtime)": PASSING_ATTEMPTS,
    "Total Pass Completions (incl. overtime)": PASSING_COMPLETIONS,
    "Total Passing Interceptions (incl. overtime)": INTERCEPTIONS,
    "Total Rushing Yards (incl. overtime)": RUSHING_YARDS,
    "Total Carries (incl. overtime)": RUSHING_ATTEMPTS,
    "Total Receiving Yards (incl. overtime)": RECEIVING_YARDS,
    "Total Receptions (incl. overtime)": RECEPTIONS,
    "Total Kicking Points (incl. overtime)": KICKING_POINTS,
    "Total Field Goals Made (incl. overtime)": FIELD_GOALS_MADE,
    "Total Extra Points Made (incl. overtime)": EXTRA_POINTS_MADE,
}

_ATHLETE_RE: Final = re.compile(r"/athletes/(\d+)")


@dataclass(frozen=True, slots=True)
class PropMove:
    """Open vs current target for one prop. Values only -- there are no prices to devig.

    This is the only prop feed keyed to ESPN athlete ids, so it needs no name matching;
    the price of that is that it publishes targets without odds. Use it for *movement*,
    never for levels: `drift` is the signal, `current` is not a devigged projection.
    """

    athlete_id: int
    espn_type_id: str
    prop_name: str
    stat: str | None
    open_value: float | None
    current_value: float | None
    last_updated: str

    @property
    def drift(self) -> float | None:
        if self.open_value is None or self.current_value is None:
            return None
        return self.current_value - self.open_value


def fetch_espn_prop_moves(
    game_id: int | str,
    *,
    client: EspnClient | None = None,
    page_size: int = 100,
    max_pages: int = 20,
) -> list[PropMove]:
    """Every prop target for one game, both open and current.

    `game_id` is nflverse `games.csv`'s `espn` column. The endpoint pages at 25 by
    default and reports `pageCount` consistently with the `limit` you send, so ask for
    100 and walk it.

    **The feed repeats itself.** Not the pager -- the payload: one live game returned
    705 items covering 465 distinct (athlete, prop type) pairs, with byte-identical
    duplicates of the same row, same provider, same `lastUpdated`. Left alone that
    double-weights 165 of 465 props in anything that aggregates drift. Exact repeats are
    dropped here. Near-repeats are *kept*: the "... Milestones" types legitimately post
    several thresholds for one athlete under one type id, so deduping on (athlete, type)
    would throw real rows away.
    """
    owned = client is None
    client = client or EspnClient()
    url = ESPN_PROPS_URL.format(gid=game_id)
    out: list[PropMove] = []
    seen: set[PropMove] = set()
    try:
        page = 1
        while True:
            try:
                payload, _ = client.get(url, params={"page": page, "limit": page_size})
            except EspnError as exc:
                raise PropsError(f"ESPN propBets for game {game_id}: {exc}") from exc
            if not isinstance(payload, dict):
                raise PropsError(f"ESPN propBets for game {game_id}: unexpected body")
            items = payload.get("items") or []
            for item in items:
                move = _parse_prop_move(item)
                if move is not None and move not in seen:
                    seen.add(move)
                    out.append(move)
            page_count = int(payload.get("pageCount") or 1)
            if page >= page_count or not items:
                break
            page += 1
            if page > max_pages:
                # Running out of budget mid-walk is not "done". A half-read board looks
                # exactly like a board where half the props were never posted, and the
                # difference matters when the thing you read it for is line movement.
                raise PropsError(
                    f"ESPN propBets for game {game_id}: stopped after {max_pages} pages "
                    f"with {page_count} reported. Raise `max_pages`, or `page_size` "
                    f"above {page_size}, rather than trusting a partial board."
                )
    finally:
        if owned:
            client.close()

    if not out:
        raise PropsError(f"ESPN propBets for game {game_id} returned nothing usable")
    return out


def _parse_prop_move(item: Mapping[str, Any]) -> PropMove | None:
    ref = str((item.get("athlete") or {}).get("$ref") or "")
    match = _ATHLETE_RE.search(ref)
    if match is None:
        # Team and game props share this feed; they have no athlete ref.
        return None
    prop_type = item.get("type") or {}
    name = str(prop_type.get("name") or "")

    def target(key: str) -> float | None:
        value = ((item.get(key) or {}).get("target") or {}).get("value")
        return float(value) if value is not None else None

    return PropMove(
        athlete_id=int(match.group(1)),
        espn_type_id=str(prop_type.get("id") or ""),
        prop_name=name,
        stat=ESPN_PROP_STATS.get(name),
        open_value=target("open"),
        current_value=target("current"),
        last_updated=str(item.get("lastUpdated") or ""),
    )


# --------------------------------------------------------------------------------------
# Pinnacle -- two-sided, devig-able, with a published stake limit
# --------------------------------------------------------------------------------------

PINNACLE_MATCHUPS: Final = "https://guest.api.arcadia.pinnacle.com/0.1/leagues/889/matchups"
PINNACLE_MARKETS: Final = "https://guest.api.arcadia.pinnacle.com/0.1/leagues/889/markets/straight"

PINNACLE_STATS: Final[dict[str, str]] = {
    "Passing Yards": PASSING_YARDS,
    "Touchdown Passes": PASSING_TDS,
    "Pass Attempts": PASSING_ATTEMPTS,
    "Pass Completions": PASSING_COMPLETIONS,
    "Interceptions": INTERCEPTIONS,
    "Rushing Yards": RUSHING_YARDS,
    "Rush Attempts": RUSHING_ATTEMPTS,
    "Receiving Yards": RECEIVING_YARDS,
    "Receptions": RECEPTIONS,
}


def _pinnacle_player(description: str, units: str) -> str:
    """`'Quentin Johnston Total Receptions'` + units `'Receptions'` -> the name."""
    stripped = description.removesuffix(f" Total {units}")
    if stripped != description:
        return stripped.strip()
    return description.split(" Total ", 1)[0].strip()


def parse_pinnacle(
    matchups: Sequence[Mapping[str, Any]],
    markets: Sequence[Mapping[str, Any]],
    *,
    category: str = "Player Props",
) -> list[PropLine]:
    """Join matchups to straight markets on matchupId.

    Only full-game (`period == 0`) totals are kept. The markets feed carries several
    rows per matchup across periods and alternate handicaps, so joining without the
    period filter silently mixes a 1st-half line into a full-game projection.
    `maxRiskStake` rides along as a sharpness signal -- Pinnacle's limit is the closest
    thing to a public confidence interval any book publishes.
    """
    by_matchup: dict[int, list[Mapping[str, Any]]] = {}
    for market in markets:
        period, matchup_id = market.get("period"), market.get("matchupId")
        if market.get("type") != "total" or period is None or matchup_id is None:
            continue
        if int(period) != 0:
            continue
        by_matchup.setdefault(int(matchup_id), []).append(market)

    out: list[PropLine] = []
    for matchup in matchups:
        special = matchup.get("special") or {}
        if special.get("category") != category:
            continue
        units = str(matchup.get("units") or "")
        stat = PINNACLE_STATS.get(units)
        if stat is None:
            continue
        matchup_id = matchup.get("id")
        rows = by_matchup.get(int(matchup_id)) if matchup_id is not None else None
        if not rows:
            continue
        sides = {
            int(p["id"]): str(p.get("name") or "").upper()
            for p in matchup.get("participants") or []
            if p.get("id") is not None
        }
        player = _pinnacle_player(str(special.get("description") or ""), units)
        for market in rows:
            priced: dict[str, tuple[float, float]] = {}
            for price in market.get("prices") or []:
                side = sides.get(int(price.get("participantId", -1)), "")
                points, value = price.get("points"), price.get("price")
                if side in ("OVER", "UNDER") and points is not None and value is not None:
                    priced[side] = (float(points), float(value))
            if len(priced) != 2:
                continue
            limits = [
                float(limit.get("amount", 0.0))
                for limit in market.get("limits") or []
                if limit.get("type") == "maxRiskStake"
            ]
            out.append(
                PropLine(
                    book="pinnacle",
                    event_id=int(matchup.get("parentId") or matchup_id or 0),
                    player=player,
                    stat=stat,
                    line=priced["OVER"][0],
                    over_odds=priced["OVER"][1],
                    under_odds=priced["UNDER"][1],
                    max_risk_stake=max(limits) if limits else None,
                )
            )
    return out


def fetch_pinnacle_player_props(*, book: _Book | None = None) -> list[PropLine]:
    """Pinnacle's NFL player props, devig-able and limit-tagged."""
    owned = book is None
    book = book or _Book()
    try:
        matchups = book.get_json(PINNACLE_MATCHUPS)
        markets = book.get_json(PINNACLE_MARKETS)
    finally:
        if owned:
            book.close()
    if not isinstance(matchups, list) or not isinstance(markets, list):
        raise PropsError("Pinnacle returned an unexpected body shape")
    props = parse_pinnacle(matchups, markets)
    if not props:
        raise PropsError("Pinnacle returned no player props")
    return props


# --------------------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PropProjection:
    """A market-implied component mean, with the shape that produced it."""

    player: str
    stat: str
    book: str
    source: str  # "ladder" | "line" | "anytime_td"
    mean: float
    median: float
    sd: float | None = None
    posted_line: float | None = None
    fit: LadderFit | None = None

    @property
    def skew_premium(self) -> float | None:
        """mean / posted line. The number this whole module exists to recover."""
        if not self.posted_line:
            return None
        return self.mean / self.posted_line


def project_event(
    props: FanDuelEventProps,
    *,
    method: str = DEFAULT_DEVIG,
    td_overround: float = 1.0,
    **fit_kwargs: Any,
) -> list[PropProjection]:
    """FanDuel event props -> component means.

    Ladders are fitted against their own two-sided main line where one exists; a
    two-sided market with no ladder falls back to the posted line as both median and
    mean, which is *known to be biased low* for skewed stats and is flagged as
    `source == "line"` so downstream can prefer ladder-backed numbers.
    """
    two_sided = {(line.player, line.stat): line for line in props.lines}
    out: list[PropProjection] = []
    fitted: set[tuple[str, str]] = set()

    for ladder in props.ladders:
        key = (ladder.player, ladder.stat)
        line = two_sided.get(key)
        anchor = line.devigged(method) if line is not None else None
        try:
            fit = fit_ladder_with_line(list(ladder.rungs), anchor, **fit_kwargs)
        except (PropsError, ValueError) as exc:
            log.warning("ladder fit failed for %s %s: %s", ladder.player, ladder.stat, exc)
            continue
        if fit.overround_pinned:
            # Not fatal -- the median still tracks the anchor -- but the shading was
            # absorbed by a parameter that ran out of room, so the mean is soft.
            log.warning(
                "ladder overround pinned at %.2f for %s %s (rms %.4f); mean is suspect",
                fit.overround,
                ladder.player,
                ladder.stat,
                fit.rms_error,
            )
        fitted.add(key)
        out.append(
            PropProjection(
                player=ladder.player,
                stat=ladder.stat,
                book=ladder.book,
                source="ladder",
                mean=fit.mean,
                median=fit.median,
                sd=fit.sd,
                posted_line=line.line if line is not None else None,
                fit=fit,
            )
        )

    for key, line in two_sided.items():
        if key in fitted:
            continue
        out.append(
            PropProjection(
                player=line.player,
                stat=line.stat,
                book=line.book,
                source="line",
                mean=line.line,
                median=line.line,
                posted_line=line.line,
            )
        )

    for td in props.touchdowns:
        lam = expected_touchdowns(td.american, overround=td_overround, method=method)
        out.append(
            PropProjection(
                player=td.player,
                stat=ANYTIME_TDS,
                book=td.book,
                source="anytime_td",
                mean=lam,
                median=lam,
            )
        )
    return out


def components_by_player(
    projections: Iterable[PropProjection],
    *,
    prefer: Sequence[str] = ("ladder", "anytime_td", "line"),
) -> dict[str, dict[str, float]]:
    """Collapse projections to one mean per (player, stat), best source first."""
    rank = {source: i for i, source in enumerate(prefer)}
    best: dict[tuple[str, str], PropProjection] = {}
    for projection in projections:
        key = (projection.player, projection.stat)
        current = best.get(key)
        if current is None or rank.get(projection.source, 99) < rank.get(current.source, 99):
            best[key] = projection

    out: dict[str, dict[str, float]] = {}
    for (player, stat), projection in best.items():
        out.setdefault(player, {})[stat] = projection.mean
    return out


@dataclass(slots=True)
class PropsBundle:
    """Whatever the books gave us this run, plus what each failure was.

    Sources are independently optional by construction: these surfaces are undocumented
    and can close without notice, so a dead book lands in `errors` and the pipeline runs
    on what is left rather than failing the whole sync.
    """

    fanduel: list[FanDuelEventProps] = field(default_factory=list)
    underdog: list[UnderdogLine] = field(default_factory=list)
    pinnacle: list[PropLine] = field(default_factory=list)
    espn_moves: dict[str, list[PropMove]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def sources_live(self) -> list[str]:
        live = []
        if self.fanduel:
            live.append("fanduel")
        if self.underdog:
            live.append("underdog")
        if self.pinnacle:
            live.append("pinnacle")
        if self.espn_moves:
            live.append("espn")
        return live


def collect_props(
    *,
    fanduel_events: int | None = 4,
    espn_game_ids: Sequence[int | str] = (),
    underdog_cache: Path | None = None,
    tabs: Sequence[str] = FANDUEL_TABS,
) -> PropsBundle:
    """Pull every source, tolerating any subset of them being dead."""
    bundle = PropsBundle()

    if fanduel_events:
        try:
            with FanDuelProps() as fd:
                for event in fd.events()[:fanduel_events]:
                    try:
                        bundle.fanduel.append(fd.event_props(event.event_id, tabs))
                    except PropsError as exc:
                        bundle.errors[f"fanduel:{event.event_id}"] = str(exc)
        except PropsError as exc:
            bundle.errors["fanduel"] = str(exc)

    try:
        bundle.underdog = parse_underdog(fetch_underdog(cache_path=underdog_cache))
    except PropsError as exc:
        bundle.errors["underdog"] = str(exc)

    try:
        bundle.pinnacle = fetch_pinnacle_player_props()
    except PropsError as exc:
        bundle.errors["pinnacle"] = str(exc)

    for game_id in espn_game_ids:
        try:
            bundle.espn_moves[str(game_id)] = fetch_espn_prop_moves(game_id)
        except PropsError as exc:
            bundle.errors[f"espn:{game_id}"] = str(exc)

    if not bundle.sources_live:
        log.error("every prop source failed: %s", bundle.errors)
    return bundle
