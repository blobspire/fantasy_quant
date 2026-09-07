"""Turning a point projection into a distribution you can actually simulate.

Every downstream number -- lineup choice, waiver value, trade verdict, title odds --
is a function of *differences* between projections and of the *width* around them.
A point estimate answers neither question, so this module is where a raw projection
stops being a number and becomes a `WeeklyOutlook`.

Three corrections, each measured on our own corpus rather than assumed:

1. **Level.** `E[actual | proj] = a + b*proj` with b in 0.92-0.96. ESPN's weekly
   projections are mildly regressive-in-reverse: low projections under-shoot and
   high ones over-shoot. Held out on 2025 after fitting 2022-2024 this is worth
   1.4-1.5% of MAE at RB and WR, 0.3% at QB, and **nothing at TE** -- ESPN's TE
   weeklies are already mean-calibrated and the fitted line is a no-op there. That
   result is reported, not tuned away; see `calibration_report`.

2. **The hurdle is not a constant.** The marginal P(actual <= 0) figures everyone
   quotes (RB 18.5%, WR 25.9%, TE 32.5%) are averages over wildly different players.
   Measured by projection decile, a WR projected for 0.3 blanks 77% of the time and
   one projected for 17.2 blanks 1% of the time. Using the marginal rate would put
   a quarter of Ja'Marr Chase's simulated weeks at zero. Hence `HurdleCurve`:
   `logit P(zero) = alpha + beta*sqrt(mu)`, clamped to the fitted support.

3. **Width scales with level.** `sigma(mu) = c + m*mu`, per position. RB/WR/TE land
   near `2.2-2.7 + 0.39-0.45*mu`; **QB is a different animal** (`4.7 + 0.17*mu`) --
   a quarterback's floor is high and his ceiling is not proportionally higher, so
   the pooled line is badly wrong for him. Held out on 2025, per-position lines
   predict the conditional SD with weighted RMSE 0.33-0.77 points against 0.77-0.99
   for the pooled `3.67 + 0.273*mu` in RESEARCH.md. Position-specific wins at all
   four positions; use the pooled line only as a fallback.

The corpus definition matters and is easy to get wrong. Pairing every weekly
projection row with its actual gives 41,970 rows and a QB MAE of 2.37 -- because
20,729 of those rows are deep-bench players projected for ~0 who scored ~0, which
inflates nothing but the row count. Filtering to `proj > 0` gives exactly the 23,999
rows and every constant in RESEARCH.md to the digit (QB 5.87/0.948/3.2%,
RB 4.10/0.921/18.5%, WR 4.27/0.943/25.9%, TE 3.24/0.961/32.5%). `load_pairs`
applies that filter and `fit` reproduces the table; that agreement is the check
that this module is reading the same corpus the constants came from.

**The mixture-moment trap.** A hurdle gamma is a mixture, so its variance carries a
between-component term: with zero mass `p` and a positive part of mean `m_pos` and
variance `v_pos`,

    mean = (1-p) * m_pos
    var  = (1-p) * v_pos + p*(1-p) * m_pos^2

There are two ways to get that wrong and both are quiet. Fitting the gamma straight
to `(mean, sd)` and then declaring `p_zero` on top leaves every simulated player's
mean **low by a factor of `(1-p)`** -- 26% low for a typical WR, which is a whole
starter's worth of points per team per week. Solving `m_pos` correctly but setting
`v_pos = sd^2/(1-p)` -- dropping the between-component term -- leaves them too
**wide**: at `(mean 9, sd 7, p 0.259)` the mixture comes out at sd 8.79, +26%. Both
errors scale with `p`, so both are worst on exactly the volatile flex candidates a
lineup decision turns on. `hurdle_gamma_from_moments` inverts the pair above, is
asserted to round-trip on a grid, and both mistakes are pinned by a test.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..core import QB, RB, SKILL_POSITIONS, TE, WR, WeeklyOutlook
from ..espn.statrows import SOURCE_ACTUAL, SOURCE_PROJECTED, SPLIT_GAME

log = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_ROOT = Path("data/snapshots/espn")
DEFAULT_REFERENCE_DIR = Path("data/reference")

POSITION_NAMES: Mapping[int, str] = {QB: "QB", RB: "RB", WR: "WR", TE: "TE"}

#: Bump when the persisted shape changes incompatibly.
SCHEMA_VERSION = 1

#: A gamma this peaked is numerically a point mass; past it `scale` underflows.
MAX_GAMMA_SHAPE = 1.0e6
#: Above this the positive part's mean explodes as `mean/(1-p)`.
MAX_P_ZERO = 0.999

_LOGIT_CLIP = 30.0


# --------------------------------------------------------------------------------------
# Fitted pieces
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LevelLine:
    """`E[actual | proj] = intercept + slope * proj`.

    Least squares, deliberately: this estimates a conditional MEAN, and the
    simulator sums means. An L1 fit would land lower (the conditional median of a
    right-skewed score) and would score better on MAE while biasing every simulated
    team total downward. Optimising the reported metric is the wrong trade here.
    """

    intercept: float
    slope: float
    r2: float
    n: int

    def __call__(self, projection: float) -> float:
        return max(self.intercept + self.slope * projection, 0.0)

    @classmethod
    def identity(cls) -> LevelLine:
        """No correction. What `_fit_level` returns when there is nothing to fit."""
        return cls(intercept=0.0, slope=1.0, r2=0.0, n=0)


@dataclass(frozen=True, slots=True)
class HurdleCurve:
    """`logit P(actual <= 0) = intercept + slope * sqrt(mu)`.

    `sqrt(mu)` rather than `mu` because it holds up better out of sample: fitting
    2022-2024 and scoring 2025, log loss improves at QB (0.0955 -> 0.0871),
    RB (0.2833 -> 0.2800) and WR (0.4131 -> 0.4084) and degrades slightly at
    TE (0.4681 -> 0.4728). Every position beats the constant rate by a wide margin
    (QB 0.268, RB 0.452, WR 0.592, TE 0.599), which is the whole point.

    `mu` is clamped into `[mu_lo, mu_hi]` -- the 1st and 99th percentile of the
    fitted mu -- before the curve is evaluated. Outside that range there is no data,
    and an unclamped logit runs off to 0 or 1 with total confidence. `base_rate` is
    the marginal P(actual <= 0), kept both as the documented positional constant and
    as the answer when the curve cannot be fitted at all.
    """

    intercept: float
    slope: float
    mu_lo: float
    mu_hi: float
    base_rate: float
    n: int

    def __call__(self, mu: float) -> float:
        if self.n == 0:
            return self.base_rate
        m = min(max(mu, self.mu_lo), self.mu_hi)
        eta = self.intercept + self.slope * math.sqrt(max(m, 0.0))
        eta = min(max(eta, -_LOGIT_CLIP), _LOGIT_CLIP)
        return 1.0 / (1.0 + math.exp(-eta))

    @classmethod
    def constant(cls, rate: float) -> HurdleCurve:
        return cls(intercept=0.0, slope=0.0, mu_lo=0.0, mu_hi=0.0, base_rate=rate, n=0)


@dataclass(frozen=True, slots=True)
class SpreadLine:
    """`sd(actual | mu) = intercept + slope * mu`, the SD of the FULL distribution.

    Fitted on quantile bins of the calibrated mean, weighted by bin count, against
    the root mean squared deviation from `mu` (not from the bin's own mean) -- the
    simulator's error is measured about the model, so that is what must be modelled.
    Equal-width bins with equal weight give a sparse 36-row bin the same say as a
    2,400-row one; that is how RESEARCH.md's pooled `3.67 + 0.273*mu` arises (our
    equal-width unweighted refit is `3.40 + 0.280*mu`, near enough), and it costs
    real accuracy: held out on 2025, count-weighted RMSE of the predicted conditional
    SD is 0.44 for the weighted quantile fit, 0.67 equal-width unweighted, and 0.77
    for the published constant.

    Known limitation: a straight line with a positive intercept cannot also be right
    at `mu -> 0`, and in the bottom held-out decile it is not -- RB modelled at sd 2.65
    against an observed 1.51, WR 2.88 against 1.95, TE 2.51 against 2.77. Above one
    projected point every decile lands within 16%, so the line stands; sub-one-point
    players are not started and their width is not what a decision turns on.
    """

    intercept: float
    slope: float
    floor: float
    n: int

    def __call__(self, mu: float) -> float:
        return max(self.intercept + self.slope * max(mu, 0.0), self.floor)


@dataclass(frozen=True, slots=True)
class PositionCalibration:
    """The three fitted curves for one `defaultPositionId`."""

    position_id: int
    level: LevelLine
    hurdle: HurdleCurve
    spread: SpreadLine
    n: int

    @property
    def name(self) -> str:
        return POSITION_NAMES.get(self.position_id, str(self.position_id))


# --------------------------------------------------------------------------------------
# Hurdle-gamma algebra
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HurdleGamma:
    """Point mass `p_zero` at zero, Gamma(shape, scale) on the positive part."""

    p_zero: float
    shape: float
    scale: float

    @property
    def positive_mean(self) -> float:
        return self.shape * self.scale

    @property
    def positive_variance(self) -> float:
        return self.shape * self.scale * self.scale

    def moments(self) -> tuple[float, float]:
        """(mean, sd) of the full mixture, zero mass included."""
        q = 1.0 - self.p_zero
        m = self.positive_mean
        variance = q * self.positive_variance + self.p_zero * q * m * m
        return q * m, math.sqrt(max(variance, 0.0))


def minimum_sd(mean: float, p_zero: float) -> float:
    """Smallest SD a hurdle mixture with this mean and zero mass can have.

    Reached when the positive part is degenerate: the zero mass alone already
    supplies `p*(1-p)*m_pos^2` of variance. Asking for less is not a tolerance
    problem, it is infeasible, and `hurdle_gamma_from_moments` returns this instead.
    """
    if mean <= 0.0 or p_zero <= 0.0:
        return 0.0
    p = min(p_zero, MAX_P_ZERO)
    return mean * math.sqrt(p / (1.0 - p))


def hurdle_gamma_from_moments(mean: float, sd: float, p_zero: float) -> HurdleGamma:
    """Solve `shape`/`scale` so the FULL mixture has exactly this mean and sd.

    Inverting

        mean = (1-p) * m_pos
        var  = (1-p) * v_pos + p*(1-p) * m_pos^2

    gives `m_pos = mean/(1-p)` and `v_pos = (sd^2 - p*mean^2/(1-p)) / (1-p)`, then
    `shape = m_pos^2/v_pos`, `scale = v_pos/m_pos`. Dropping the `p*(1-p)*m_pos^2`
    term -- the obvious mistake -- OVERstates the SD, because that variance then gets
    added on top of a positive part already sized to carry all of it: at a WR's
    measured 25.9% blank rate, `(mean 9, sd 7)` comes back out at sd 8.79.

    When `sd < minimum_sd(mean, p_zero)` the request is infeasible and the positive
    part is made as tight as the numerics allow; the returned object then has a
    slightly larger SD than asked for, which callers can check with `.moments()`.
    """
    if not 0.0 <= p_zero <= 1.0:
        raise ValueError(f"p_zero must be a probability, got {p_zero}")
    if sd < 0.0:
        raise ValueError(f"sd must be non-negative, got {sd}")
    if mean <= 0.0:
        return HurdleGamma(p_zero=1.0, shape=1.0, scale=0.0)

    p = min(p_zero, MAX_P_ZERO)
    q = 1.0 - p
    m_pos = mean / q
    v_pos = (sd * sd - p * mean * mean / q) / q
    v_floor = m_pos * m_pos / MAX_GAMMA_SHAPE
    if v_pos < v_floor:
        v_pos = v_floor
    return HurdleGamma(p_zero=p, shape=m_pos * m_pos / v_pos, scale=v_pos / m_pos)


def mixture_moments(p_zero: float, shape: float, scale: float) -> tuple[float, float]:
    """(mean, sd) of a hurdle gamma. The inverse of `hurdle_gamma_from_moments`."""
    return HurdleGamma(p_zero=p_zero, shape=shape, scale=scale).moments()


# --------------------------------------------------------------------------------------
# The fitted set
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationSet:
    """Fitted calibration for one scoring variant, plus a pooled fallback.

    Keyed by variant because the intercept, the spread line and the hurdle curve are
    all denominated in league points: half-PPR receivers project lower and blank at a
    given projection more often than the same players in full PPR. Only the slope is
    roughly scoring-invariant.

    As of the 2026 corpus only `ppr` is fitted -- ESPN's half-PPR league default
    (`leaguedefaults/8`) is 2026-only and 404s for 2022-2025, so there is nothing to
    fit a `half_ppr` set on yet. The two half-PPR leagues therefore run on the PPR
    curves; expect the level correction to transfer cleanly and the hurdle and spread
    to be modestly wide at a given mu until a half-PPR corpus accumulates.
    """

    variant: str
    seasons: tuple[int, ...]
    fitted_at: str
    n_pairs: int
    source: str
    positions: Mapping[int, PositionCalibration]
    pooled: PositionCalibration

    def for_position(self, position_id: int) -> PositionCalibration:
        """The fitted curves for a position, or the pooled ones if it was not fitted.

        K and D/ST fall through to pooled. They are not modelled here -- their score
        distributions are nothing like a skill player's -- so a caller relying on
        this for them is getting a placeholder, not a projection.
        """
        return self.positions.get(position_id, self.pooled)

    # -- application ---------------------------------------------------------------

    def calibrate(self, projection: float, position_id: int) -> float:
        """Raw point projection -> calibrated conditional mean."""
        return self.for_position(position_id).level(projection)

    def p_zero(self, mu: float, position_id: int) -> float:
        return self.for_position(position_id).hurdle(mu)

    def sd(self, mu: float, position_id: int, source_spread: float | None = None) -> float:
        """Modelled SD at a calibrated mean, optionally widened by source disagreement.

        `source_spread` is the SD *across sources* of the point projection. It is added
        in quadrature, which is the conservative reading: the fitted `sigma(mu)` is the
        average conditional SD at that level, and a player the sources disagree about is
        harder than average. It does double-count a little -- part of the average width
        already reflects typical disagreement -- so it is opt-in and left at None until
        the ensemble can measure the typical spread at each mu and pass the excess.
        """
        base = self.for_position(position_id).spread(mu)
        if source_spread is None or source_spread <= 0.0:
            return base
        return math.hypot(base, source_spread)

    def outlook(
        self,
        *,
        player_id: int,
        season: int,
        week: int,
        position_id: int,
        projection: float,
        pro_team_id: int = 0,
        playing: bool = True,
        source_spread: float | None = None,
    ) -> WeeklyOutlook:
        """A raw point projection -> a calibrated `WeeklyOutlook`."""
        return self.outlook_from_mean(
            player_id=player_id,
            season=season,
            week=week,
            position_id=position_id,
            mean=self.calibrate(projection, position_id),
            pro_team_id=pro_team_id,
            playing=playing,
            source_spread=source_spread,
        )

    def outlook_from_mean(
        self,
        *,
        player_id: int,
        season: int,
        week: int,
        position_id: int,
        mean: float,
        pro_team_id: int = 0,
        playing: bool = True,
        source_spread: float | None = None,
    ) -> WeeklyOutlook:
        """Build the outlook from an ALREADY calibrated mean.

        This is the entry point for an ensemble that has done its own level
        correction; passing an ensemble mean through `outlook` would apply ESPN's
        level correction a second time.
        """
        mu = max(mean, 0.0)
        p = self.p_zero(mu, position_id)
        sd = self.sd(mu, position_id, source_spread)
        # Infeasible pairs are physically possible at the low end (a nearly-certain
        # blank has a large forced SD), so widen rather than silently mis-solve.
        sd = max(sd, minimum_sd(mu, p))
        gamma = hurdle_gamma_from_moments(mu, sd, p)
        actual_mean, actual_sd = gamma.moments()
        out = WeeklyOutlook(
            player_id=player_id,
            season=season,
            week=week,
            position_id=position_id,
            mean=actual_mean,
            sd=actual_sd,
            p_zero=gamma.p_zero,
            shape=gamma.shape,
            scale=gamma.scale,
            pro_team_id=pro_team_id,
            playing=playing,
        )
        return out if playing else out.zeroed()

    # -- persistence ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "variant": self.variant,
            "seasons": list(self.seasons),
            "fitted_at": self.fitted_at,
            "n_pairs": self.n_pairs,
            "source": self.source,
            "pooled": _position_to_dict(self.pooled),
            "positions": {str(k): _position_to_dict(v) for k, v in sorted(self.positions.items())},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CalibrationSet:
        # A half-written or hand-edited file can be perfectly valid JSON that is not
        # an object at all (`null`, `[]`, `42`). Reject it as a ValueError so `load`'s
        # documented "warn and fall back" holds instead of an AttributeError escaping.
        if not isinstance(raw, Mapping):
            raise ValueError(f"calibration payload must be an object, got {type(raw).__name__}")
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(f"calibration schema {version}, expected {SCHEMA_VERSION}")
        positions = raw.get("positions") or {}
        if not isinstance(positions, Mapping):
            raise ValueError(f"'positions' must be an object, got {type(positions).__name__}")
        return cls(
            variant=str(raw["variant"]),
            seasons=tuple(int(s) for s in raw["seasons"]),
            fitted_at=str(raw["fitted_at"]),
            n_pairs=int(raw["n_pairs"]),
            source=str(raw.get("source", "fit")),
            positions={int(k): _position_from_dict(v) for k, v in positions.items()},
            pooled=_position_from_dict(raw["pooled"]),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        tmp.replace(path)
        return path


def _position_to_dict(pc: PositionCalibration) -> dict[str, Any]:
    return {
        "position_id": pc.position_id,
        "n": pc.n,
        "level": {
            "intercept": pc.level.intercept,
            "slope": pc.level.slope,
            "r2": pc.level.r2,
            "n": pc.level.n,
        },
        "hurdle": {
            "intercept": pc.hurdle.intercept,
            "slope": pc.hurdle.slope,
            "mu_lo": pc.hurdle.mu_lo,
            "mu_hi": pc.hurdle.mu_hi,
            "base_rate": pc.hurdle.base_rate,
            "n": pc.hurdle.n,
        },
        "spread": {
            "intercept": pc.spread.intercept,
            "slope": pc.spread.slope,
            "floor": pc.spread.floor,
            "n": pc.spread.n,
        },
    }


def _position_from_dict(raw: Mapping[str, Any]) -> PositionCalibration:
    lv, hz, sp = raw["level"], raw["hurdle"], raw["spread"]
    return PositionCalibration(
        position_id=int(raw["position_id"]),
        n=int(raw["n"]),
        level=LevelLine(
            intercept=float(lv["intercept"]),
            slope=float(lv["slope"]),
            r2=float(lv["r2"]),
            n=int(lv["n"]),
        ),
        hurdle=HurdleCurve(
            intercept=float(hz["intercept"]),
            slope=float(hz["slope"]),
            mu_lo=float(hz["mu_lo"]),
            mu_hi=float(hz["mu_hi"]),
            base_rate=float(hz["base_rate"]),
            n=int(hz["n"]),
        ),
        spread=SpreadLine(
            intercept=float(sp["intercept"]),
            slope=float(sp["slope"]),
            floor=float(sp["floor"]),
            n=int(sp["n"]),
        ),
    )


# --------------------------------------------------------------------------------------
# Shipped defaults
# --------------------------------------------------------------------------------------

#: What `fit()` produced on the 2022-2025 PPR corpus (23,999 paired player-weeks).
#: Carried in source so a fresh checkout with no Parquet and no fitted JSON still
#: gets the measured numbers instead of an identity transform. Regenerate with
#: `fit(write=True)` and paste the printed block back here.
#: (level intercept, level slope, level r2, hurdle intercept, hurdle slope,
#:  hurdle mu_lo, hurdle mu_hi, base_rate, spread intercept, spread slope, n)
_DEFAULT_ROWS: Mapping[int, tuple[float, ...]] = {
    QB: (0.2883, 0.9480, 0.2112, 2.8922, -1.9037, 0.4352, 23.2287, 0.0321, 4.7219, 0.1716, 2210),
    RB: (0.2928, 0.9212, 0.4861, 2.4695, -2.1251, 0.3277, 20.6169, 0.1850, 2.4439, 0.3851, 6312),
    WR: (0.0637, 0.9434, 0.4160, 2.2126, -1.5725, 0.1206, 19.3886, 0.2592, 2.7112, 0.4085, 10004),
    TE: (0.2108, 0.9609, 0.3810, 2.4091, -1.7228, 0.5671, 15.0049, 0.3251, 2.1996, 0.4492, 5473),
}
_DEFAULT_POOLED = (
    0.1742,
    0.9414,
    0.4783,
    2.2638,
    -1.6992,
    0.2333,
    20.8511,
    0.2338,
    2.7021,
    0.3653,
    23999,
)

#: The measured floor. Even a projected-zero player who plays has this much spread.
SD_FLOOR = 1.0


def _position_from_row(position_id: int, row: Sequence[float]) -> PositionCalibration:
    a, b, r2, hi_, hs, lo, hi, base, si, ss, n = row
    return PositionCalibration(
        position_id=position_id,
        n=int(n),
        level=LevelLine(intercept=a, slope=b, r2=r2, n=int(n)),
        hurdle=HurdleCurve(intercept=hi_, slope=hs, mu_lo=lo, mu_hi=hi, base_rate=base, n=int(n)),
        spread=SpreadLine(intercept=si, slope=ss, floor=SD_FLOOR, n=int(n)),
    )


def default_calibration(variant: str = "ppr") -> CalibrationSet:
    """The shipped fit. Used when no `calibration_{variant}.json` exists."""
    return CalibrationSet(
        variant=variant,
        seasons=(2022, 2023, 2024, 2025),
        fitted_at="2026-09-07",
        n_pairs=int(_DEFAULT_POOLED[-1]),
        source="defaults",
        positions={p: _position_from_row(p, row) for p, row in _DEFAULT_ROWS.items()},
        pooled=_position_from_row(0, _DEFAULT_POOLED),
    )


def params_path(variant: str = "ppr", directory: Path = DEFAULT_REFERENCE_DIR) -> Path:
    return directory / f"calibration_{variant}.json"


def load(variant: str = "ppr", directory: Path = DEFAULT_REFERENCE_DIR) -> CalibrationSet:
    """Load fitted params, falling back to the shipped defaults.

    Never refits on import and never reads Parquet: a corrupt or absent file is a
    logged warning and the measured defaults, not an exception at call time.
    """
    path = params_path(variant, directory)
    if not path.exists():
        log.info("no calibration file at %s; using shipped defaults", path)
        return default_calibration(variant)
    try:
        return CalibrationSet.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        log.warning("unreadable calibration at %s (%s); using shipped defaults", path, exc)
        return default_calibration(variant)


# --------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------

PAIR_COLUMNS = (
    "espn_id",
    "full_name",
    "default_position_id",
    "stat_season",
    "request_season",
    "stat_source_id",
    "stat_split_type_id",
    "scoring_period_id",
    "applied_total",
    "captured_at",
)

PAIR_SCHEMA: Mapping[str, Any] = {
    "player_id": pl.Int64,
    "name": pl.Utf8,
    "position_id": pl.Int64,
    "season": pl.Int64,
    "week": pl.Int64,
    "projection": pl.Float64,
    "actual": pl.Float64,
}


def empty_pairs() -> pl.DataFrame:
    return pl.DataFrame(schema=PAIR_SCHEMA)


def load_pairs(
    root: Path = DEFAULT_SNAPSHOT_ROOT,
    variant: str = "ppr",
    seasons: Sequence[int] | None = None,
    min_projection: float = 0.0,
) -> pl.DataFrame:
    """Paired weekly (projection, actual) rows from the Parquet corpus.

    Four filters, each guarding a measured failure:

    * `stat_season == request_season` -- a 2026 request returns 26,198 rows of 2025
      alongside 12,535 of 2026 in the same file. Without this the "2026" fit is 68%
      last season.
    * `stat_split_type_id == SPLIT_GAME` and `scoring_period_id > 0` -- season totals
      and the ROS per-game rate are not player-weeks.
    * latest `captured_at` per (player, season, week, source) -- the corpus holds the
      same 2024 season under two capture dates, and 2024 rows would otherwise be
      double-weighted in every fit.
    * `projection > min_projection` -- see the module docstring. At the default of 0
      this yields exactly the 23,999 rows and every constant in RESEARCH.md; keeping
      the projected-zero rows yields 41,970 rows and a QB MAE of 2.37.
    """
    files = sorted(root.glob(f"season=*/variant={variant}/*.parquet"))
    if not files:
        log.warning("no %s snapshots under %s", variant, root)
        return empty_pairs()

    frames = [pl.read_parquet(f, columns=list(PAIR_COLUMNS)) for f in files]
    df = pl.concat(frames, how="vertical_relaxed")
    df = df.filter(
        (pl.col("stat_season") == pl.col("request_season"))
        & (pl.col("stat_split_type_id") == SPLIT_GAME)
        & (pl.col("scoring_period_id") > 0)
        & pl.col("default_position_id").is_in(list(SKILL_POSITIONS))
        & pl.col("applied_total").is_not_null()
        & pl.col("espn_id").is_not_null()
    )
    if seasons is not None:
        df = df.filter(pl.col("stat_season").is_in(list(seasons)))
    if df.height == 0:
        return empty_pairs()

    df = (
        df.sort("captured_at")
        .group_by(
            ["espn_id", "stat_season", "scoring_period_id", "stat_source_id"],
            maintain_order=True,
        )
        .last()
    )

    projected = df.filter(pl.col("stat_source_id") == SOURCE_PROJECTED).select(
        pl.col("espn_id").alias("player_id"),
        pl.col("full_name").alias("name"),
        pl.col("default_position_id").alias("position_id"),
        pl.col("stat_season").alias("season"),
        pl.col("scoring_period_id").alias("week"),
        pl.col("applied_total").alias("projection"),
    )
    actual = df.filter(pl.col("stat_source_id") == SOURCE_ACTUAL).select(
        pl.col("espn_id").alias("player_id"),
        pl.col("stat_season").alias("season"),
        pl.col("scoring_period_id").alias("week"),
        pl.col("applied_total").alias("actual"),
    )
    pairs = projected.join(actual, on=["player_id", "season", "week"], how="inner")
    return pairs.filter(pl.col("projection") > min_projection).select(list(PAIR_SCHEMA))


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


def _fit_level(proj: np.ndarray, actual: np.ndarray) -> LevelLine:
    n = int(proj.size)
    if n < 30 or float(np.ptp(proj)) < 1e-9:
        return LevelLine.identity()
    slope, intercept = np.polyfit(proj, actual, 1)
    pred = intercept + slope * proj
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return LevelLine(intercept=float(intercept), slope=float(slope), r2=r2, n=n)


def _irls_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> tuple[float, float]:
    """Two-parameter logistic by Newton/IRLS. numpy only; no sklearn in this tree."""
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(60):
        eta = np.clip(design @ beta, -_LOGIT_CLIP, _LOGIT_CLIP)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        lhs = design.T @ (design * w[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(lhs, design.T @ (w * z))
        if not np.all(np.isfinite(step)):
            break
        converged = float(np.max(np.abs(step - beta))) < 1e-10
        beta = step
        if converged:
            break
    return float(beta[0]), float(beta[1])


def _fit_hurdle(mu: np.ndarray, actual: np.ndarray) -> HurdleCurve:
    y = (actual <= 0.0).astype(float)
    base = float(y.mean()) if y.size else 0.0
    # Both classes must be present or the logit is unidentified.
    if y.size < 100 or y.sum() < 10 or y.sum() > y.size - 10:
        return HurdleCurve.constant(base)
    intercept, slope = _irls_logistic(np.sqrt(np.maximum(mu, 0.0)), y)
    lo, hi = (float(v) for v in np.quantile(mu, [0.01, 0.99]))
    return HurdleCurve(
        intercept=intercept,
        slope=slope,
        mu_lo=lo,
        mu_hi=max(hi, lo + 1e-6),
        base_rate=base,
        n=int(y.size),
    )


def _fit_spread(mu: np.ndarray, actual: np.ndarray, n_bins: int = 20) -> SpreadLine:
    """Regress the binned RMS deviation from `mu` on `mu`, weighted by bin count."""
    n = int(mu.size)
    if n < 200:
        return SpreadLine(
            intercept=float(np.sqrt(np.mean((actual - mu) ** 2))) if n else 4.0,
            slope=0.0,
            floor=SD_FLOOR,
            n=n,
        )
    edges = np.unique(np.quantile(mu, np.linspace(0.0, 1.0, n_bins + 1)))
    index = np.digitize(mu, edges[1:-1])
    xs, ys, ws = [], [], []
    for k in range(edges.size - 1):
        mask = index == k
        count = int(mask.sum())
        if count < 30:
            continue
        xs.append(float(mu[mask].mean()))
        ys.append(float(np.sqrt(np.mean((actual[mask] - mu[mask]) ** 2))))
        ws.append(float(count))
    if len(xs) < 3:
        return SpreadLine(
            intercept=float(np.sqrt(np.mean((actual - mu) ** 2))), slope=0.0, floor=SD_FLOOR, n=n
        )
    slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1, w=np.sqrt(np.array(ws)))
    return SpreadLine(intercept=float(intercept), slope=float(slope), floor=SD_FLOOR, n=n)


def _fit_position(position_id: int, proj: np.ndarray, actual: np.ndarray) -> PositionCalibration:
    level = _fit_level(proj, actual)
    mu = np.maximum(level.intercept + level.slope * proj, 0.0)
    return PositionCalibration(
        position_id=position_id,
        level=level,
        hurdle=_fit_hurdle(mu, actual),
        spread=_fit_spread(mu, actual),
        n=int(proj.size),
    )


def fit_from_pairs(
    pairs: pl.DataFrame, variant: str = "ppr", source: str = "fit"
) -> CalibrationSet:
    """Fit every position plus the pooled fallback from an in-memory pair table."""
    if pairs.height == 0:
        log.warning("no paired player-weeks; returning shipped defaults")
        return default_calibration(variant)

    proj = pairs["projection"].to_numpy()
    actual = pairs["actual"].to_numpy()
    pos = pairs["position_id"].to_numpy()

    positions: dict[int, PositionCalibration] = {}
    for pid in SKILL_POSITIONS:
        mask = pos == pid
        if int(mask.sum()) < 200:
            log.warning("only %d rows for position %d; leaving it to pooled", mask.sum(), pid)
            continue
        positions[pid] = _fit_position(pid, proj[mask], actual[mask])

    seasons = tuple(sorted(int(s) for s in pairs["season"].unique().to_list()))
    return CalibrationSet(
        variant=variant,
        seasons=seasons,
        fitted_at=dt.datetime.now(dt.UTC).date().isoformat(),
        n_pairs=int(pairs.height),
        source=source,
        positions=positions,
        pooled=_fit_position(0, proj, actual),
    )


def fit(
    root: Path = DEFAULT_SNAPSHOT_ROOT,
    variant: str = "ppr",
    seasons: Sequence[int] | None = None,
    write: bool = True,
    directory: Path = DEFAULT_REFERENCE_DIR,
) -> CalibrationSet:
    """Fit from the Parquet corpus and (by default) persist to data/reference/.

    The entry point behind `load`'s file. Deliberately explicit rather than lazy:
    refitting on import would make every process pay for 24k rows of Parquet and
    would let the corpus silently change the answer between two runs.
    """
    pairs = load_pairs(root=root, variant=variant, seasons=seasons)
    fitted = fit_from_pairs(pairs, variant=variant)
    if write and pairs.height > 0:
        path = fitted.save(params_path(variant, directory))
        log.info("wrote %s (%d pairs)", path, fitted.n_pairs)
    return fitted


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecileRow:
    """One reliability bucket: what we said, what happened, and what we modelled."""

    decile: int
    n: int
    mean_projection: float
    mean_calibrated: float
    mean_actual: float
    sd_actual: float
    sd_modelled: float
    p_zero_actual: float
    p_zero_modelled: float


@dataclass(frozen=True, slots=True)
class PositionReport:
    position_id: int
    n_train: int
    n_test: int
    mae_raw: float
    mae_calibrated: float
    rmse_raw: float
    rmse_calibrated: float
    slope_raw: float
    slope_calibrated: float
    deciles: tuple[DecileRow, ...]

    @property
    def name(self) -> str:
        return POSITION_NAMES.get(self.position_id, str(self.position_id))

    @property
    def mae_delta_pct(self) -> float:
        return 100.0 * (self.mae_calibrated - self.mae_raw) / self.mae_raw if self.mae_raw else 0.0

    @property
    def rmse_delta_pct(self) -> float:
        if not self.rmse_raw:
            return 0.0
        return 100.0 * (self.rmse_calibrated - self.rmse_raw) / self.rmse_raw

    @property
    def improves_mae(self) -> bool:
        return self.mae_calibrated < self.mae_raw


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    variant: str
    train_seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    positions: tuple[PositionReport, ...]
    pooled: PositionReport

    @property
    def all_positions_improve(self) -> bool:
        return all(p.improves_mae for p in self.positions)

    @property
    def regressions(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.positions if not p.improves_mae)


def _decile_rows(
    proj: np.ndarray,
    actual: np.ndarray,
    calibration: CalibrationSet,
    position_id: int,
    n_bins: int = 10,
) -> tuple[DecileRow, ...]:
    edges = np.unique(np.quantile(proj, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return ()
    index = np.digitize(proj, edges[1:-1])
    rows: list[DecileRow] = []
    for k in range(edges.size - 1):
        mask = index == k
        count = int(mask.sum())
        if count < 20:
            continue
        # Row-wise, then aggregate. Evaluating the curves once at the bin's mean
        # would flatter them: a decile that spans 0-12 projected points has a mean
        # that describes none of its members, and both curves are nonlinear.
        cal = np.array([calibration.calibrate(float(p), position_id) for p in proj[mask]])
        p_mod = np.array([calibration.p_zero(float(m), position_id) for m in cal])
        sd_mod = np.array([calibration.sd(float(m), position_id) for m in cal])
        rows.append(
            DecileRow(
                decile=k + 1,
                n=count,
                mean_projection=float(proj[mask].mean()),
                mean_calibrated=float(cal.mean()),
                mean_actual=float(actual[mask].mean()),
                sd_actual=float(np.sqrt(np.mean((actual[mask] - cal) ** 2))),
                # E[(actual - mu)^2] = sd^2 row by row, so the bin's model-implied
                # deviation is the ROOT MEAN SQUARE of the sds, not their average.
                sd_modelled=float(np.sqrt(np.mean(sd_mod**2))),
                p_zero_actual=float(np.mean(actual[mask] <= 0.0)),
                p_zero_modelled=float(p_mod.mean()),
            )
        )
    return tuple(rows)


def _position_report(
    position_id: int,
    train_proj: np.ndarray,
    test_proj: np.ndarray,
    test_actual: np.ndarray,
    calibration: CalibrationSet,
) -> PositionReport:
    cal = np.array([calibration.calibrate(float(p), position_id) for p in test_proj])
    return PositionReport(
        position_id=position_id,
        n_train=int(train_proj.size),
        n_test=int(test_proj.size),
        mae_raw=float(np.mean(np.abs(test_actual - test_proj))),
        mae_calibrated=float(np.mean(np.abs(test_actual - cal))),
        rmse_raw=float(np.sqrt(np.mean((test_actual - test_proj) ** 2))),
        rmse_calibrated=float(np.sqrt(np.mean((test_actual - cal) ** 2))),
        slope_raw=float(np.polyfit(test_proj, test_actual, 1)[0]),
        slope_calibrated=float(np.polyfit(cal, test_actual, 1)[0]),
        deciles=_decile_rows(test_proj, test_actual, calibration, position_id),
    )


def calibration_report(
    pairs: pl.DataFrame | None = None,
    variant: str = "ppr",
    train_seasons: Sequence[int] = (2022, 2023, 2024),
    test_seasons: Sequence[int] = (2025,),
    root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> CalibrationReport:
    """Fit on `train_seasons`, evaluate on `test_seasons`. Never the same rows.

    Split by season rather than at random: player-weeks within a season share a
    player, an offence and a projection regime, so a random split leaks and would
    show an improvement that does not survive to next Sunday.
    """
    if pairs is None:
        pairs = load_pairs(root=root, variant=variant)
    if pairs.height == 0:
        raise ValueError("no paired player-weeks to report on")
    overlap = set(train_seasons) & set(test_seasons)
    if overlap:
        raise ValueError(f"train and test share seasons {sorted(overlap)}; the split would leak")

    train = pairs.filter(pl.col("season").is_in(list(train_seasons)))
    test = pairs.filter(pl.col("season").is_in(list(test_seasons)))
    if train.height == 0 or test.height == 0:
        raise ValueError(
            f"empty split: {train.height} train rows, {test.height} test rows "
            f"(corpus has seasons {sorted(pairs['season'].unique().to_list())})"
        )

    fitted = fit_from_pairs(train, variant=variant, source="report")

    reports: list[PositionReport] = []
    for pid in SKILL_POSITIONS:
        tr = train.filter(pl.col("position_id") == pid)
        te = test.filter(pl.col("position_id") == pid)
        if te.height < 50 or tr.height < 200:
            continue
        reports.append(
            _position_report(
                pid,
                tr["projection"].to_numpy(),
                te["projection"].to_numpy(),
                te["actual"].to_numpy(),
                fitted,
            )
        )
    pooled = _position_report(
        0,
        train["projection"].to_numpy(),
        test["projection"].to_numpy(),
        test["actual"].to_numpy(),
        fitted,
    )
    return CalibrationReport(
        variant=variant,
        train_seasons=tuple(int(s) for s in train_seasons),
        test_seasons=tuple(int(s) for s in test_seasons),
        positions=tuple(reports),
        pooled=pooled,
    )


def render_report(report: CalibrationReport, fitted: CalibrationSet | None = None) -> str:
    """Plain text so it lands the same in a terminal, a log and a test failure."""
    lines: list[str] = []
    lines.append(
        f"Calibration report -- variant={report.variant} "
        f"train={'+'.join(str(s) for s in report.train_seasons)} "
        f"test={'+'.join(str(s) for s in report.test_seasons)}"
    )
    lines.append("")
    lines.append(
        f"{'pos':<5}{'n_tr':>7}{'n_te':>7}{'MAE raw':>9}{'MAE cal':>9}{'d%':>7}"
        f"{'RMSE raw':>10}{'RMSE cal':>10}{'d%':>7}{'slope raw':>11}{'slope cal':>11}"
    )
    lines.append("-" * 92)
    for p in (*report.positions, report.pooled):
        name = "ALL" if p.position_id == 0 else p.name
        lines.append(
            f"{name:<5}{p.n_train:>7}{p.n_test:>7}{p.mae_raw:>9.3f}{p.mae_calibrated:>9.3f}"
            f"{p.mae_delta_pct:>+7.2f}{p.rmse_raw:>10.3f}{p.rmse_calibrated:>10.3f}"
            f"{p.rmse_delta_pct:>+7.2f}{p.slope_raw:>11.3f}{p.slope_calibrated:>11.3f}"
        )
    lines.append("")
    if report.regressions:
        lines.append(
            "!! held-out MAE did NOT improve at: "
            + ", ".join(report.regressions)
            + " -- reported as measured, not tuned away."
        )
    else:
        lines.append("held-out MAE improved at every position.")
    lines.append("")

    for p in report.positions:
        lines.append(f"{p.name} reliability by projection decile (held out)")
        lines.append(
            f"  {'d':>2}{'n':>7}{'proj':>8}{'calib':>8}{'actual':>8}"
            f"{'sd obs':>9}{'sd mod':>9}{'P0 obs':>9}{'P0 mod':>9}"
        )
        for d in p.deciles:
            lines.append(
                f"  {d.decile:>2}{d.n:>7}{d.mean_projection:>8.2f}{d.mean_calibrated:>8.2f}"
                f"{d.mean_actual:>8.2f}{d.sd_actual:>9.2f}{d.sd_modelled:>9.2f}"
                f"{d.p_zero_actual * 100:>8.1f}%{d.p_zero_modelled * 100:>8.1f}%"
            )
        lines.append("")

    if fitted is not None:
        lines.append("Fitted parameters (full corpus)")
        lines.append(
            f"  {'pos':<5}{'n':>7}{'level':>22}{'R2':>7}{'sigma(mu)':>20}"
            f"{'logit P0':>26}{'base':>8}"
        )
        for pid in (*SKILL_POSITIONS, 0):
            pc = fitted.pooled if pid == 0 else fitted.positions.get(pid)
            if pc is None:
                continue
            name = "ALL" if pid == 0 else POSITION_NAMES[pid]
            level = f"{pc.level.intercept:+.3f} {pc.level.slope:+.3f}*p"
            sigma = f"{pc.spread.intercept:.3f} {pc.spread.slope:+.3f}*mu"
            logit = f"{pc.hurdle.intercept:+.4f} {pc.hurdle.slope:+.4f}*sqrt(mu)"
            lines.append(
                f"  {name:<5}{pc.n:>7}{level:>22}{pc.level.r2:>7.3f}{sigma:>20}"
                f"{logit:>26}{pc.hurdle.base_rate * 100:>7.1f}%"
            )
    return "\n".join(lines)


def print_report(report: CalibrationReport, fitted: CalibrationSet | None = None) -> None:
    print(render_report(report, fitted))
