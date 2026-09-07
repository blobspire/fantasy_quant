"""Opportunity, stability, expected fantasy points, and the regression screen.

Fantasy production decomposes into *opportunity* -- how many carries and targets a
player gets, and where on the field -- and *efficiency* -- what he does with them.
The two behave completely differently under repetition, and that difference is the
only edge in this module. Opportunity persists week to week; efficiency does not.
So a player's recent points are a mixture of a signal to project forward and a
noise to regress to the positional mean, and separating them is how you find the
buy-lows and sell-highs before the rest of the league does.

Everything below is **measured on our own copy of nflverse**, 2019-2025 regular
season, RB/WR/TE, 35,840 player-weeks. Reproduce any of it from `usage_weeks`.

**1. The stability split** (`stability_table`), on the default fantasy-relevant
population (>=6 games, >=2 opportunities per game; 24,603 usable player-weeks).
Reliability at n games is the correlation of a player's first n appearances with his
next n, aggregated volume-weighted -- a raw split-half correlation, **not**
Spearman-Brown corrected, because two disjoint n-game windows are already parallel
measurements of an n-game window. (Spearman-Brown here would report the reliability
of 2n while labelling it n, which halves every "games to 0.70" below.)

    metric               lag1   3-game rel   season-over-season   games to 0.70
    carry_share         0.875     0.808           0.727                 1
    snap_share          0.758     0.778           0.732                 1
    wopr                0.628     0.767           0.850                 2
    air_yards_share     0.619     0.791           0.898                 2
    target_share        0.567     0.701           0.762                 3
    rz_touch_share      0.342     0.498           0.696                --
    i10_touch_share     0.263     0.409           0.607                --
    ------------------------------------------------------------------------
    yac_per_reception   0.229     0.459           0.691                --
    catch_rate          0.120     0.307           0.541                --
    ypt                 0.074     0.198           0.432                --
    td_rate             0.050     0.078           0.243                --
    racr                0.045     0.068           0.103                --
    ypc                 0.043     0.040           0.270                --
    croe                0.035     0.081           0.155                --

`--` means "not inside the search window", and the window stops at 8 because
reliability at n needs 2n appearances in one season. Red-zone share gets to 0.699 at
8 and inside-10 share to 0.627, so both are *nearly* believable by the end of a
season; TD rate reaches 0.241 and never will be.

The line above the rule is the projectable half and the line below is the regress
half, and the gap at the boundary (0.263 to 0.229) is the only place the two are
even close. Target share clears 0.70 reliability at **3 games** (0.701) -- which is
exactly the folklore "stabilizes in ~3 games", and it survives contact with our data
without needing a correction to help it. It is worth being clear that it *only just*
clears: the curve is 0.53 / 0.65 / 0.70 / 0.72 / 0.72 / 0.74 / 0.76 / 0.78 over
n=1..8, so 3 games buys most of what 8 games buys and neither is definitive.

Two results worth not skipping. Red-zone and inside-10 share are *far* less stable
than raw share (0.34 and 0.26 against 0.57), because the denominator is ~4 touches
a game -- treat "he is the goal-line back" as a claim needing most of a season of
evidence, not one game. And YAC per reception is the honest middle case: 0.23 week
to week but 0.69 year to year. That is a real skill measured very noisily, so it
regresses hard for a weekly projection and much less for a season-long one.

**2. WOPR's 1.5/0.7 is calibrated for the wrong scoring system.** Regressing weekly
receiving points on target share and air-yards share (WR/TE, n=25,236) through the
origin, rescaled to WOPR's own 2.2 total:

    scoring           fitted (target_share, air_yards_share)   ratio    R2
    full PPR          (2.058, 0.142)                           14.5    0.602
    half PPR          (1.945, 0.255)                            7.6    0.550
    standard          (1.760, 0.440)                            4.0    0.472
    receiving yards   (1.746, 0.454)                            3.9      --
    canonical         (1.500, 0.700)                            2.1      --

The canonical pair is a *yardage* calibration and drifts further wrong the more a
league pays for receptions: with target share in the model, receptions regress on
air-yards share with a **negative** coefficient. The operational consequence is
small but one-directional. Holding the target-share weight at 1.5, the air-yards
weight that maximizes correlation with next week's receiving PPR is **0.15**
(r=0.5059) against 0.70 (r=0.4972), and against rest-of-season PPG it is 0.10
(r=0.6598 vs 0.6471). So the coefficients barely change the ranking -- which is
itself the finding: in full PPR, WOPR is target share wearing a hat. The default
stays canonical so our numbers reconcile with published ones (`usage_weeks`
reproduces nflverse's own `wopr` column to 0.0 on all 35,840 rows);
`REFIT_WOPR_PPR`, `REFIT_WOPR_HALF_PPR` and `REFIT_WOPR_STANDARD` are one argument
away, and `fit_wopr_weights` re-derives them for any reception value.

**3. Per-opportunity value** (`ConversionRates`), the input to xFP. Pooled RB/WR/TE
PPR points per opportunity, by distance to the end zone:

    yardline_100      carry   target   target/carry        n (carry / target)
    1-5 goal line      2.50     3.58       1.43             4,241 /  3,382
    6-10               0.92     2.66       2.91             3,127 /  3,618
    11-20 fringe       0.64     1.99       3.11             6,984 /  8,809
    21-50 midfield     0.51     1.63       3.22            26,289 / 36,772
    51-100 own half    0.48     1.53       3.16            44,172 / 67,453
    all (pooled)       0.62     1.69       2.72            84,813 /120,034

Read the `all` row as pooled RB/WR/TE on both sides. Split out, an average **RB**
carry is worth **0.611** PPR points (81,205 of the 84,813) and an average target
**1.687**, a ratio of 2.76.
The reference figures we were handed were 0.60 and 1.57, so the carry matches and
the target is 7% higher here. The compression at the goal line reproduces cleanly:
pooled inside the 10 the target premium is **1.70x** (reference 1.79x) against
~3.2x in open field, because a carry inside the 5 scores on 40.0% of attempts
against 0.3% from your own half.

xFP is built from component rates, not a points-per-touch constant, so one table
serves a full-PPR and a half-PPR league by re-scoring the components through that
league's own `LeagueContext.scorer`. And because points are linear in (receptions,
yards, TDs, fumbles), actual minus expected decomposes *exactly* into which
unstable metric drove the gap (max residual 7e-15 over 2024) -- which is the
regression screen.

**4. What the screen is and is not good for.** On 686 consecutive player-season
pairs with >=8 games in both years:

    predictor of next season's FP/g       R2
    FP/g this season                     0.407
    xFP/g this season                    0.353

So season-long xFP does **not** out-predict season-long actual points, and any
source that says it does is overselling. The reason is in the decomposition -- the
year-to-year correlation of each slice of the gap, per game:

    td_over_expected       r=0.10   sd 1.01 pts/g   <- does not repeat
    fumbles_over_expected  r=0.05   sd 0.11 pts/g   <- does not repeat
    yards_over_expected    r=0.33   sd 0.92 pts/g   <- partly a real player
    catch_over_expected    r=0.39   sd 0.38 pts/g   <- partly a real player
    total gap              r=0.28   sd 1.76 pts/g

Two thirds of the gap's variance sits in yardage and catch efficiency, which are
genuinely persistent -- a deep threat beats a league-average yards-per-target
forever. Only the touchdown slice is close to pure noise. Acting on the whole gap
is therefore a blunter instrument than acting on `unrepeatable_points`, and
`regression_screen(sort_by="unrepeatable")` exists for that.

It still works as a screen. Grouping those 686 pairs by this season's gap:

    sell (gap/g >= +1.5)   n=161   FP/g 15.99 -> 14.23   gap/g +2.87 -> +0.96
    fair                   n=452   FP/g 11.14 -> 11.03   gap/g -0.01 -> +0.08
    buy  (gap/g <= -1.5)   n= 73   FP/g 10.41 -> 10.72   gap/g -2.30 -> -0.49

Two thirds of a sell-high gap and four fifths of a buy-low gap are gone a year
later. Named cases the screen flags, and what happened next: 2019 Mark Ingram (15
TD vs 9.8 expected, 15.9 FP/g -> 2 TD, 5.3), Raheem Mostert (10 vs 3.7 -> 3),
Cooper Kupp (10 vs 7.6 -> 3), Aaron Jones (19 vs 11.2 -> 11); 2024 Terry McLaurin
(13 vs 6.0, 15.8 FP/g -> 3 TD, 11.4) and Mark Andrews (11 vs 4.6 -> 6). On the buy
side, 2024 Travis Etienne (2 TD vs 4.6, 8.7 FP/g -> 13 TD, 14.9) and Javonte
Williams (4 vs 6.5, 9.3 -> 13 TD, 15.2). It is not infallible: 2019 Derrick Henry
was the top sell-high at 18 TD vs 8.0 expected and scored 17 the next year.

**The Kupp 2019 archetype, honestly.** We flag him, driver `touchdowns`, but our
expected-TD count is 7.58 against the ~5.6 the archetype quotes; ffopportunity's
per-play model says 6.15. Our five field buckets price every target from the same
yard line identically, and Kupp's were unusually short, so a bucket model reads his
51 midfield targets as more TD equity than a model conditioned on air yards does.
Across 2019 WR/TE/RB our expected receiving TDs correlate 0.935 with
ffopportunity's per-play model with no mean bias (2.22 vs 2.39 against 2.32
actual), so the disagreement is concentrated in short-area receivers. Read
`expected_touchdowns` as a bucket estimate, not a per-play one.

Traps found while building this, all measured:

* **pbp targets reconcile with the weekly file exactly** -- 4,206 player-seasons
  over 2019-2025, 100.0% identical, max |diff| 0 -- but only after excluding
  two-point conversions, which carry a receiver and are not attempts.
* **`ffopportunity` has no fumble model.** Its `fantasy_points` is full PPR *minus
  2 per fumble lost* (verified exact on 5,878 of 6,005 rows of 2024, and -2.016
  mean on the 127 fumble rows) while `fantasy_points_exp` has no fumble term at
  all, so `total_fantasy_points_diff` is biased down by ~2 per fumble.
  `ffopportunity_weekly` corrects it. Note it is full PPR, not half. Our season xFP
  correlates 0.989 with theirs over 2024 (480 RB/WR/TE with a row in both, means
  75.5 vs 78.9; their season total runs ~4% higher because their per-play model
  prices throws we price by bucket).
* **No free nflverse source carries routes run.** Not `stats_player_week`, not
  `pfr_advstats` (rec gives broken tackles, drops, air yards -- no routes), not the
  consolidated NGS files. `route_participation` is therefore null unless the caller
  supplies a routes frame; `snap_share` is the substitute that does exist, and it
  joins only through the pfr_id <-> gsis_id crosswalk in `players.parquet` (99.9%
  coverage on this population).
* **Play-by-play lives under the `pbp` release tag**, which `data/nflverse.py` does
  not name. It is reached with `nv.Asset("pbp", ...)` here rather than by editing
  that module, and scanned lazily: the seven seasons are 140 MB and 372 columns, of
  which this module reads 18.
* **A season with no play-by-play is not a season with no opportunities.** Bucket
  columns are zero-filled only for seasons whose pbp actually loaded and left null
  otherwise, so `xfp` comes back null rather than quietly reading as "he never
  touched the ball", and the screen drops those weeks loudly.
* **Reliability is a property of the population, not the metric.** Widening from
  the default population to every RB/WR/TE with a stat line moves target share's
  one-game reliability from 0.532 to 0.578 and its lag-1 from 0.567 to 0.636 --
  between-player variance goes up when you throw in every third receiver who caught
  one pass, and every metric looks steadier for it. `min_games` /
  `min_opportunities_per_game` are arguments so that choice is visible rather than
  buried.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from ..core import RB, TE, WR
from ..data import nflverse as nv

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Positions and scoring
# --------------------------------------------------------------------------------------

#: The positions whose fantasy points are an opportunity story. QB is deliberately
#: absent: a quarterback's points are dominated by per-attempt efficiency (depth of
#: target, sack rate, interception luck), so "value each attempt at its league
#: average" is a far weaker model there and would invite a confident wrong answer.
OPPORTUNITY_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")

#: nflverse position string -> ESPN `defaultPositionId`, so a `LeagueContext.scorer`
#: applies directly. Position ids, never lineupSlotIds -- the two spaces collide.
POSITION_IDS: Mapping[str, int] = {"RB": RB, "WR": WR, "TE": TE}

# ESPN statIds for the six components an opportunity converts into.
STAT_RECEPTIONS = "53"
STAT_RECEIVING_YARDS = "42"
STAT_RECEIVING_TD = "43"
STAT_RUSHING_YARDS = "24"
STAT_RUSHING_TD = "25"
STAT_FUMBLE_LOST = "72"

COMPONENT_STATS: tuple[str, ...] = (
    STAT_RECEPTIONS,
    STAT_RECEIVING_YARDS,
    STAT_RECEIVING_TD,
    STAT_RUSHING_YARDS,
    STAT_RUSHING_TD,
    STAT_FUMBLE_LOST,
)

#: (raw ESPN stats, defaultPositionId) -> points. Same shape as `LeagueContext.scorer`.
Scorer = Callable[[Mapping[str, float], int], float]


def ppr_scorer(reception_points: float = 1.0) -> Scorer:
    """A standalone scorer over the six components, so this module runs league-free.

    Pass a real `LeagueContext.scorer` wherever one is available -- it carries TE
    premium and any non-standard yardage rate. Every function here takes the scorer
    as an argument for exactly that reason.
    """
    table = {
        STAT_RECEPTIONS: reception_points,
        STAT_RECEIVING_YARDS: 0.1,
        STAT_RECEIVING_TD: 6.0,
        STAT_RUSHING_YARDS: 0.1,
        STAT_RUSHING_TD: 6.0,
        STAT_FUMBLE_LOST: -2.0,
    }

    def score(stats: Mapping[str, float], position_id: int) -> float:  # noqa: ARG001
        return sum(value * table[key] for key, value in stats.items() if key in table)

    return score


PPR: Scorer = ppr_scorer(1.0)
HALF_PPR: Scorer = ppr_scorer(0.5)
STANDARD: Scorer = ppr_scorer(0.0)


# --------------------------------------------------------------------------------------
# Field position
# --------------------------------------------------------------------------------------

#: Distance-to-end-zone buckets, goal line outward. `yardline_100` is the nflverse
#: convention: 1 is the opponent's 1-yard line, 100 is your own goal line.
#:
#: The split at 5 is not cosmetic. The TD gradient is steepest there -- a pooled
#: carry is worth 2.50 PPR points inside the 5 and 0.92 from the 6 to the 10 -- and
#: collapsing the two into one "inside the 10" bucket is what makes a coarse xFP
#: model overstate expected touchdowns for short-area players.
FIELD_BUCKETS: tuple[str, ...] = (
    "goal_line",
    "inside_10",
    "fringe_rz",
    "midfield",
    "own_half",
)

#: Inclusive upper bound on `yardline_100` for each bucket.
BUCKET_MAX: Mapping[str, int] = {
    "goal_line": 5,
    "inside_10": 10,
    "fringe_rz": 20,
    "midfield": 50,
    "own_half": 100,
}

INSIDE_10_BUCKETS: tuple[str, ...] = ("goal_line", "inside_10")
RED_ZONE_BUCKETS: tuple[str, ...] = ("goal_line", "inside_10", "fringe_rz")

#: Bucket-count column names, e.g. `carries_inside_10` / `targets_own_half`.
BUCKET_COLUMNS: tuple[str, ...] = tuple(
    f"{plural}_{bucket}" for plural in ("carries", "targets") for bucket in FIELD_BUCKETS
)
TEAM_BUCKET_COLUMNS: tuple[str, ...] = tuple(f"team_{c}" for c in BUCKET_COLUMNS)


def bucket_of(yardline_100: float) -> str:
    """Which field bucket a snap at `yardline_100` belongs to."""
    for bucket in FIELD_BUCKETS:
        if yardline_100 <= BUCKET_MAX[bucket]:
            return bucket
    return FIELD_BUCKETS[-1]


def _bucket_expr() -> pl.Expr:
    first = FIELD_BUCKETS[0]
    expr = pl.when(pl.col("yardline_100") <= BUCKET_MAX[first]).then(pl.lit(first))
    for bucket in FIELD_BUCKETS[1:-1]:
        expr = expr.when(pl.col("yardline_100") <= BUCKET_MAX[bucket]).then(pl.lit(bucket))
    return expr.otherwise(pl.lit(FIELD_BUCKETS[-1])).alias("bucket")


def _plural(kind: str) -> str:
    return "carries" if kind == "carry" else "targets"


# --------------------------------------------------------------------------------------
# WOPR
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WoprWeights:
    """Coefficients for `wopr = a*target_share + b*air_yards_share`.

    Only the ratio carries information -- WOPR has no units -- so `rescaled` exists
    to put a refit back on the canonical scale, where the weights sum to 2.2 and a
    reading means what a reader of published WOPR thinks it means.
    """

    target_share: float
    air_yards_share: float

    def __call__(self, target_share: float, air_yards_share: float) -> float:
        return self.target_share * target_share + self.air_yards_share * air_yards_share

    def expr(
        self,
        target_share: str = "target_share",
        air_yards_share: str = "air_yards_share",
    ) -> pl.Expr:
        return self.target_share * pl.col(target_share) + self.air_yards_share * pl.col(
            air_yards_share
        )

    def rescaled(self, total: float = 2.2) -> WoprWeights:
        current = self.target_share + self.air_yards_share
        if current == 0:
            raise ValueError("cannot rescale weights that sum to zero")
        factor = total / current
        return WoprWeights(self.target_share * factor, self.air_yards_share * factor)

    @property
    def ratio(self) -> float:
        """How many times a point of target share outweighs a point of air-yards share."""
        if self.air_yards_share == 0:
            return math.inf
        return self.target_share / self.air_yards_share


#: The published weights. Every source states them; none shows the regression.
CANONICAL_WOPR = WoprWeights(1.5, 0.7)

#: Refits on 2019-2025 WR/TE weekly receiving points, rescaled to the canonical 2.2
#: total so they drop straight into an existing WOPR reading. See the module
#: docstring: the canonical pair behaves like a yardage-scoring calibration.
REFIT_WOPR_PPR = WoprWeights(2.058, 0.142)
REFIT_WOPR_HALF_PPR = WoprWeights(1.945, 0.255)
REFIT_WOPR_STANDARD = WoprWeights(1.760, 0.440)


@dataclass(frozen=True, slots=True)
class WoprFit:
    """The result of re-fitting WOPR, with the raw regression kept alongside."""

    weights: WoprWeights
    #: OLS coefficients before rescaling, in fantasy points per unit of share.
    raw_target_share: float
    raw_air_yards_share: float
    r_squared: float
    n: int
    reception_points: float
    positions: tuple[str, ...]

    @property
    def ratio(self) -> float:
        return self.weights.ratio

    def __str__(self) -> str:
        return (
            f"wopr = {self.weights.target_share:.3f}*target_share + "
            f"{self.weights.air_yards_share:.3f}*air_yards_share "
            f"(ratio {self.ratio:.1f} vs canonical {CANONICAL_WOPR.ratio:.1f}; raw "
            f"{self.raw_target_share:.1f}/{self.raw_air_yards_share:.1f} pts per share, "
            f"R2={self.r_squared:.3f}, n={self.n})"
        )


_WOPR_FIT_INPUTS = (
    "target_share",
    "air_yards_share",
    "position",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles_lost",
)


def fit_wopr_weights(
    frame: pl.DataFrame,
    *,
    reception_points: float = 1.0,
    positions: Sequence[str] = ("WR", "TE"),
    rescale_to: float | None = 2.2,
) -> WoprFit:
    """Regress receiving fantasy points on target share and air-yards share.

    Through the origin: both regressors are shares that are zero when a player is
    unused, so an intercept has no meaning -- and fitted anyway it lands within 0.01
    of zero on this data, moving the ratio by less than the seasonal spread.

    `reception_points` is what actually moves the answer; see the module docstring.
    RBs are excluded by default because their receiving points sit inside a rushing
    workload neither share can see.
    """
    if missing := [c for c in _WOPR_FIT_INPUTS if c not in frame.columns]:
        raise ValueError(f"frame is missing {missing}; build it with usage_weeks()")

    d = frame.filter(pl.col("position").is_in(list(positions))).with_columns(
        _y=(
            reception_points * pl.col("receptions")
            + 0.1 * pl.col("receiving_yards")
            + 6.0 * pl.col("receiving_tds")
            - 2.0 * pl.col("receiving_fumbles_lost")
        )
    )
    d = d.filter(
        pl.col("target_share").is_finite()
        & pl.col("air_yards_share").is_finite()
        & pl.col("_y").is_finite()
    )
    if d.height < 100:
        raise ValueError(f"only {d.height} usable rows; refusing to fit WOPR on that")

    x = np.column_stack([d["target_share"].to_numpy(), d["air_yards_share"].to_numpy()]).astype(
        float
    )
    y = d["_y"].to_numpy().astype(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    raw = WoprWeights(float(beta[0]), float(beta[1]))
    return WoprFit(
        weights=raw if rescale_to is None else raw.rescaled(rescale_to),
        raw_target_share=raw.target_share,
        raw_air_yards_share=raw.air_yards_share,
        r_squared=r2,
        n=d.height,
        reception_points=reception_points,
        positions=tuple(positions),
    )


# --------------------------------------------------------------------------------------
# Play-by-play: field-position usage
# --------------------------------------------------------------------------------------

PBP_TAG = "pbp"

# 18 of 372 columns. 140 MB of play-by-play does not need to be in memory to answer
# "how many carries did he take inside the 10".
_PBP_COLUMNS = (
    "season",
    "week",
    "season_type",
    "posteam",
    "yardline_100",
    "two_point_attempt",
    "rush_attempt",
    "pass_attempt",
    "complete_pass",
    "rusher_player_id",
    "receiver_player_id",
    "rushing_yards",
    "receiving_yards",
    "rush_touchdown",
    "pass_touchdown",
    "fumble_lost",
    "fumbled_1_player_id",
    "cp",
)


def pbp_asset(season: int) -> nv.Asset:
    """The play-by-play release asset for one season."""
    return nv.Asset(PBP_TAG, f"play_by_play_{season}.parquet")


def _seasons_tuple(seasons: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(seasons, str):
        # A str is Iterable, so "2026" would shred into (2, 0, 2, 6) and every
        # downstream filter would quietly match nothing.
        raise TypeError(f"seasons must be an int or an iterable of ints, not {seasons!r}")
    if isinstance(seasons, int):
        return (seasons,)
    out = tuple(int(s) for s in seasons)
    if not out:
        raise ValueError("no seasons requested")
    return out


def _pbp_paths(
    seasons: tuple[int, ...],
    *,
    cache: nv.NflverseCache | None,
    allow_missing: bool,
) -> tuple[list[Path], tuple[int, ...]]:
    """Local paths for each season's pbp file, plus the seasons that actually loaded."""
    resolved = cache if cache is not None else nv.default_cache()
    paths: list[Path] = []
    loaded: list[int] = []
    for season in seasons:
        asset = pbp_asset(season)
        try:
            paths.append(resolved.ensure(asset))
        except nv.NflverseNotFound:
            if not allow_missing:
                raise nv.NflverseNotFound(
                    f"{asset} is not published. nflverse builds play-by-play only once "
                    "games have been played; pass allow_missing=True to skip the season."
                ) from None
            log.warning("%s not published; its field-position columns will stay null", asset)
            continue
        loaded.append(season)
    if not paths:
        raise nv.NflverseNotFound(f"no play-by-play published for any of {list(seasons)}.")
    return paths, tuple(loaded)


def _pbp_scan(paths: Sequence[Path]) -> pl.LazyFrame:
    return (
        pl.scan_parquet(list(paths))
        .select(_PBP_COLUMNS)
        # Two-point plays carry a receiver and a rusher but are not attempts.
        # Leaving them in breaks the exact reconciliation with the weekly file.
        .filter(
            (pl.col("season_type") == "REG")
            & (pl.col("two_point_attempt").fill_null(0) == 0)
            & pl.col("yardline_100").is_not_null()
        )
        .with_columns(pl.col("season", "week").cast(pl.Int32))
    )


def field_position_usage(
    seasons: int | Iterable[int],
    *,
    cache: nv.NflverseCache | None = None,
    allow_missing: bool = False,
) -> pl.DataFrame:
    """Per player-week carries and targets, split by distance to the end zone.

    Every player is returned, not only RB/WR/TE, so team denominators computed from
    this frame include quarterback scrambles -- which is what "share of the team's
    red-zone work" is supposed to mean.

    `expected_receptions` is the sum of nflverse's own per-throw completion
    probability over the player's targets, a much sharper baseline for catch rate
    over expected than a bucket average because it conditions on the depth, down and
    distance of the actual throw. It is well calibrated on this data: mean `cp`
    matches realized catch rate to within 0.03 in 14 of the 15 position/bucket cells,
    the exception being TE at the goal line (0.564 against a realized 0.609, n=1,041).
    `cp` is null on 11 of 120,921 targets and polars sums those as zero, so corpus
    expected receptions run 405 low against 81,467 actual -- 0.3%, and in the same
    direction for every player.
    """
    paths, _ = _pbp_paths(_seasons_tuple(seasons), cache=cache, allow_missing=allow_missing)
    scan = _pbp_scan(paths)

    def side(id_col: str, gate: str, kind: str) -> pl.LazyFrame:
        plural = _plural(kind)
        return (
            scan.filter((pl.col(gate).fill_null(0) == 1) & pl.col(id_col).is_not_null())
            .select(
                "season",
                "week",
                pl.col("posteam").alias("team"),
                pl.col(id_col).alias("player_id"),
                _bucket_expr(),
            )
            .group_by("season", "week", "team", "player_id")
            .agg(
                [
                    (pl.col("bucket") == bucket).sum().cast(pl.Int32).alias(f"{plural}_{bucket}")
                    for bucket in FIELD_BUCKETS
                ]
            )
        )

    keys = ["season", "week", "team", "player_id"]
    expected_catches = (
        scan.filter(
            (pl.col("pass_attempt").fill_null(0) == 1) & pl.col("receiver_player_id").is_not_null()
        )
        .select(
            "season",
            "week",
            pl.col("posteam").alias("team"),
            pl.col("receiver_player_id").alias("player_id"),
            pl.col("cp"),
        )
        .group_by(keys)
        .agg(pl.col("cp").sum().alias("expected_receptions"))
    )

    targets = side("receiver_player_id", "pass_attempt", "target")
    out = (
        side("rusher_player_id", "rush_attempt", "carry")
        .join(targets, on=keys, how="full", coalesce=True)
        .join(expected_catches, on=keys, how="full", coalesce=True)
        .with_columns(
            [pl.col(c).fill_null(0).cast(pl.Int32) for c in BUCKET_COLUMNS]
            + [pl.col("expected_receptions").fill_null(0.0)]
        )
        .collect()
    )
    return out.sort("season", "week", "player_id")


# --------------------------------------------------------------------------------------
# The usage frame
# --------------------------------------------------------------------------------------

_WEEKLY_COLUMNS = (
    "season",
    "week",
    "player_id",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_tds",
    "receiving_fumbles_lost",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "target_share",
    "air_yards_share",
    "racr",
    "pacr",
    "fantasy_points_ppr",
)

# Cast to float and zero-fill: nflverse leaves a stat null rather than zero when a
# player recorded none of it, and every ratio below would otherwise go null.
_ZERO_FILL = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_tds",
    "receiving_fumbles_lost",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "target_share",
    "air_yards_share",
    "fantasy_points_ppr",
)


def usage_weeks(
    seasons: int | Iterable[int],
    *,
    positions: Sequence[str] = OPPORTUNITY_POSITIONS,
    wopr_weights: WoprWeights = CANONICAL_WOPR,
    field_position: bool = True,
    snaps: bool = True,
    routes: pl.DataFrame | None = None,
    cache: nv.NflverseCache | None = None,
    allow_missing: bool = False,
) -> pl.DataFrame:
    """One row per player-week with every opportunity metric this module measures.

    Stable metrics: `target_share`, `air_yards_share`, `wopr`, `carry_share`,
    `rz_touch_share`, `i10_touch_share`, `snap_share`, `route_participation`.
    Unstable: `td_rate`, `catch_rate`, `croe`, `yac_per_reception`, `ypc`, `ypt`,
    `racr`, `pacr`. `stability_table` is what says which is which, from this frame.

    `field_position=False` skips play-by-play entirely -- the only slow part -- at
    the cost of the red-zone columns, `expected_receptions`, and therefore xFP.

    `routes` is an optional frame of (season, week, player_id, routes,
    team_dropbacks). No free nflverse asset carries routes run, so
    `route_participation` is null without one; `snap_share` is the substitute that
    does exist.
    """
    wanted = _seasons_tuple(seasons)
    weekly = nv.player_week_stats(
        wanted, season_type="REG", allow_missing=allow_missing, cache=cache
    ).with_columns(pl.col("season", "week").cast(pl.Int32))

    # Team denominators come from every position, before the RB/WR/TE filter: a
    # quarterback's scrambles are part of the team's rushing workload.
    #
    # nflverse publishes `air_yards_share` but not the team air yards it divides by,
    # and that denominator is NOT the sum of `receiving_air_yards` (it comes off the
    # passing side, which counts throws with no charted receiver). It is recovered
    # exactly by inverting any one player's own share: over 2019-2025 every team-week
    # has at least one usable row and the implied value agrees across all of a team's
    # receivers to 0.0. Without it `season_usage` cannot re-form a season air-yards
    # share and has to average weekly shares instead, which biases it up ~22%.
    implied_team_air_yards = pl.when(pl.col("air_yards_share").abs() > 1e-12).then(
        pl.col("receiving_air_yards") / pl.col("air_yards_share")
    )
    team_totals = weekly.group_by("season", "week", "team").agg(
        pl.col("targets").sum().alias("team_targets"),
        pl.col("carries").sum().alias("team_carries"),
        implied_team_air_yards.drop_nulls().first().alias("team_air_yards"),
    )

    df = (
        weekly.filter(pl.col("position").is_in(list(positions)))
        .select([c for c in _WEEKLY_COLUMNS if c in weekly.columns])
        .with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in _ZERO_FILL])
        .rename({"player_display_name": "player_name", "opponent_team": "opponent"})
        .join(team_totals, on=["season", "week", "team"], how="left")
    )

    df = _add_field_position(df, wanted, field_position, cache=cache, allow_missing=allow_missing)
    df = _add_snap_share(df, wanted, snaps, cache=cache, allow_missing=allow_missing)
    df = _add_route_participation(df, routes)
    return _derive_metrics(df, wopr_weights).sort("season", "week", "player_id")


def _null_field_position(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [pl.lit(None, dtype=pl.Int32).alias(c) for c in BUCKET_COLUMNS]
        + [pl.lit(None, dtype=pl.Int32).alias(c) for c in TEAM_BUCKET_COLUMNS]
        + [pl.lit(None, dtype=pl.Float64).alias("expected_receptions")]
    )


def _add_field_position(
    df: pl.DataFrame,
    seasons: tuple[int, ...],
    enabled: bool,
    *,
    cache: nv.NflverseCache | None,
    allow_missing: bool,
) -> pl.DataFrame:
    if not enabled:
        return _null_field_position(df)
    try:
        # Resolve first, purely to learn which seasons actually exist: the zero-fill
        # below is only honest for those.
        _, loaded = _pbp_paths(seasons, cache=cache, allow_missing=allow_missing)
    except nv.NflverseError as exc:
        if not allow_missing:
            raise
        log.warning("no play-by-play for %s (%s); red-zone columns stay null", seasons, exc)
        return _null_field_position(df)

    fp = field_position_usage(loaded, cache=cache, allow_missing=allow_missing)
    team_fp = fp.group_by("season", "week", "team").agg(
        *[pl.col(c).sum().alias(f"team_{c}") for c in BUCKET_COLUMNS]
    )
    out = df.join(fp.drop("team"), on=["season", "week", "player_id"], how="left").join(
        team_fp, on=["season", "week", "team"], how="left"
    )
    # A player-week absent from the pbp aggregate had zero carries and zero targets
    # -- but only if that season's pbp was actually loaded. Elsewhere the null is
    # honest and must survive, or a missing file reads as a player who never played.
    covered = pl.col("season").is_in(list(loaded))
    return out.with_columns(
        [
            pl.when(covered).then(pl.col(c).fill_null(0)).otherwise(None).alias(c)
            for c in (*BUCKET_COLUMNS, *TEAM_BUCKET_COLUMNS)
        ]
        + [
            pl.when(covered)
            .then(pl.col("expected_receptions").fill_null(0.0))
            .otherwise(None)
            .alias("expected_receptions")
        ]
    )


def _add_snap_share(
    df: pl.DataFrame,
    seasons: tuple[int, ...],
    enabled: bool,
    *,
    cache: nv.NflverseCache | None,
    allow_missing: bool,
) -> pl.DataFrame:
    """Offensive snap share, joined via the nflverse pfr_id <-> gsis_id crosswalk."""
    if not enabled:
        return df.with_columns(snap_share=pl.lit(None, dtype=pl.Float64))
    try:
        snaps = nv.snap_counts(seasons, allow_missing=True, cache=cache)
        crosswalk = nv.players(cache=cache).select(
            pl.col("gsis_id").alias("player_id"), pl.col("pfr_id").alias("pfr_player_id")
        )
    except nv.NflverseError as exc:
        if not allow_missing:
            raise
        log.warning("snap counts unavailable (%s); snap_share stays null", exc)
        return df.with_columns(snap_share=pl.lit(None, dtype=pl.Float64))

    joined = (
        snaps.filter(pl.col("game_type") == "REG")
        .select(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            "pfr_player_id",
            pl.col("offense_pct").cast(pl.Float64).alias("snap_share"),
        )
        .join(crosswalk.drop_nulls(), on="pfr_player_id", how="inner")
        .select("season", "week", "player_id", "snap_share")
        .unique(subset=["season", "week", "player_id"])
    )
    return df.join(joined, on=["season", "week", "player_id"], how="left")


def _add_route_participation(df: pl.DataFrame, routes: pl.DataFrame | None) -> pl.DataFrame:
    if routes is None:
        return df.with_columns(
            routes=pl.lit(None, dtype=pl.Float64),
            route_participation=pl.lit(None, dtype=pl.Float64),
        )
    needed = {"season", "week", "player_id", "routes", "team_dropbacks"}
    if missing := needed - set(routes.columns):
        raise ValueError(f"routes frame is missing {sorted(missing)}")
    return (
        df.join(
            routes.select(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                "player_id",
                pl.col("routes").cast(pl.Float64),
                pl.col("team_dropbacks").cast(pl.Float64),
            ),
            on=["season", "week", "player_id"],
            how="left",
        )
        .with_columns(
            route_participation=pl.when(pl.col("team_dropbacks") > 0)
            .then(pl.col("routes") / pl.col("team_dropbacks"))
            .otherwise(None)
        )
        .drop("team_dropbacks")
    )


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def _sum_cols(columns: Iterable[str]) -> pl.Expr:
    expr: pl.Expr | None = None
    for column in columns:
        expr = pl.col(column) if expr is None else expr + pl.col(column)
    if expr is None:
        raise ValueError("no columns to sum")
    return expr


def _derive_metrics(df: pl.DataFrame, wopr_weights: WoprWeights) -> pl.DataFrame:
    df = df.with_columns(
        opportunities=pl.col("carries") + pl.col("targets"),
        total_tds=pl.col("rushing_tds") + pl.col("receiving_tds"),
        fumbles_lost=pl.col("rushing_fumbles_lost") + pl.col("receiving_fumbles_lost"),
        rz_touches=_sum_cols(f"{p}_{b}" for p in ("carries", "targets") for b in RED_ZONE_BUCKETS),
        i10_touches=_sum_cols(
            f"{p}_{b}" for p in ("carries", "targets") for b in INSIDE_10_BUCKETS
        ),
        team_rz_touches=_sum_cols(
            f"team_{p}_{b}" for p in ("carries", "targets") for b in RED_ZONE_BUCKETS
        ),
        team_i10_touches=_sum_cols(
            f"team_{p}_{b}" for p in ("carries", "targets") for b in INSIDE_10_BUCKETS
        ),
        wopr=wopr_weights.expr(),
    )
    return df.with_columns(
        carry_share=_safe_ratio(pl.col("carries"), pl.col("team_carries")),
        rz_touch_share=_safe_ratio(pl.col("rz_touches"), pl.col("team_rz_touches")),
        i10_touch_share=_safe_ratio(pl.col("i10_touches"), pl.col("team_i10_touches")),
        td_rate=_safe_ratio(pl.col("total_tds"), pl.col("opportunities")),
        catch_rate=_safe_ratio(pl.col("receptions"), pl.col("targets")),
        croe=_safe_ratio(pl.col("receptions") - pl.col("expected_receptions"), pl.col("targets")),
        yac_per_reception=_safe_ratio(pl.col("receiving_yards_after_catch"), pl.col("receptions")),
        ypc=_safe_ratio(pl.col("rushing_yards"), pl.col("carries")),
        ypt=_safe_ratio(pl.col("receiving_yards"), pl.col("targets")),
    )


def season_usage(frame: pl.DataFrame, *, min_games: int = 1) -> pl.DataFrame:
    """Collapse the weekly frame to one row per player-season, with per-game rates.

    **Every share here is re-formed from summed numerator and summed denominator**,
    never averaged over weeks, and the denominator is always the *team's* -- which is
    the part that is easy to get wrong. Weighting a player's weekly target shares by
    his own targets looks like the same thing and is not: it over-weights exactly the
    weeks his share was high, because his targets are the share's own numerator. On
    2019-2025 that error inflates season target share by +0.030 on average (0.211
    against a true 0.182, +16%, worst case +0.12) and air-yards share by +0.044
    (+22%, worst case +0.18). `test_season_usage_uses_the_team_denominator` pins it.

    `air_yards_share` needs `team_air_yards`, which `usage_weeks` recovers; a frame
    built some other way gets a null column and a warning rather than the biased
    stand-in.
    """
    has_team_air_yards = "team_air_yards" in frame.columns
    if not has_team_air_yards:
        log.warning(
            "frame has no team_air_yards column (build it with usage_weeks); "
            "season air_yards_share will be null rather than averaged over weeks"
        )
    return (
        frame.group_by("player_id", "season")
        .agg(
            pl.col("player_name").last().alias("name"),
            pl.col("position").last().alias("position"),
            pl.len().alias("games"),
            pl.col("carries").sum().alias("carries"),
            pl.col("targets").sum().alias("targets"),
            pl.col("receptions").sum().alias("receptions"),
            pl.col("opportunities").sum().alias("opportunities"),
            pl.col("team_carries").sum().alias("team_carries"),
            pl.col("team_targets").sum().alias("team_targets"),
            (
                pl.col("team_air_yards").sum()
                if has_team_air_yards
                else pl.lit(None, dtype=pl.Float64)
            ).alias("team_air_yards"),
            pl.col("total_tds").sum().alias("touchdowns"),
            pl.col("rushing_yards").sum().alias("rushing_yards"),
            pl.col("receiving_yards").sum().alias("receiving_yards"),
            pl.col("receiving_air_yards").sum().alias("receiving_air_yards"),
            pl.col("fantasy_points_ppr").sum().alias("fantasy_points_ppr"),
            pl.col("snap_share").mean().alias("snap_share"),
            pl.col("rz_touches").sum().alias("rz_touches"),
            pl.col("i10_touches").sum().alias("i10_touches"),
            pl.col("team_rz_touches").sum().alias("team_rz_touches"),
            pl.col("team_i10_touches").sum().alias("team_i10_touches"),
        )
        .filter(pl.col("games") >= min_games)
        .with_columns(
            carries_per_game=pl.col("carries") / pl.col("games"),
            targets_per_game=pl.col("targets") / pl.col("games"),
            points_per_game=pl.col("fantasy_points_ppr") / pl.col("games"),
            target_share=_safe_ratio(pl.col("targets"), pl.col("team_targets")),
            air_yards_share=_safe_ratio(pl.col("receiving_air_yards"), pl.col("team_air_yards")),
            carry_share=_safe_ratio(pl.col("carries"), pl.col("team_carries")),
            rz_touch_share=_safe_ratio(pl.col("rz_touches"), pl.col("team_rz_touches")),
            i10_touch_share=_safe_ratio(pl.col("i10_touches"), pl.col("team_i10_touches")),
            td_rate=_safe_ratio(pl.col("touchdowns"), pl.col("opportunities")),
            catch_rate=_safe_ratio(pl.col("receptions"), pl.col("targets")),
            ypc=_safe_ratio(pl.col("rushing_yards"), pl.col("carries")),
            ypt=_safe_ratio(pl.col("receiving_yards"), pl.col("targets")),
        )
        .sort("season", "fantasy_points_ppr", descending=[False, True])
    )


# --------------------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StabilityMetric:
    """One metric to test, and the volume column that weights and gates it."""

    name: str
    #: What the metric is a rate *per*. Aggregating a rate across games means
    #: re-forming the ratio from summed numerator and denominator, never averaging.
    volume: str
    #: Weeks below this much volume are dropped: a 1-for-1 week is a catch rate of
    #: 1.000 and pure noise. Shares use 0, because a zero share is a real reading.
    min_volume: float = 0.0


DEFAULT_STABILITY_METRICS: tuple[StabilityMetric, ...] = (
    StabilityMetric("target_share", "targets"),
    StabilityMetric("air_yards_share", "targets"),
    StabilityMetric("wopr", "targets"),
    StabilityMetric("carry_share", "carries"),
    StabilityMetric("snap_share", "opportunities"),
    StabilityMetric("rz_touch_share", "opportunities"),
    StabilityMetric("i10_touch_share", "opportunities"),
    StabilityMetric("td_rate", "opportunities", 3.0),
    StabilityMetric("catch_rate", "targets", 3.0),
    StabilityMetric("croe", "targets", 3.0),
    StabilityMetric("yac_per_reception", "receptions", 3.0),
    StabilityMetric("ypc", "carries", 5.0),
    StabilityMetric("ypt", "targets", 3.0),
    StabilityMetric("racr", "targets", 3.0),
)

#: A metric counts as projectable when its next-appearance autocorrelation clears
#: this. Every share metric measures 0.26 or better and every efficiency metric
#: except YAC measures 0.12 or worse, so the cut is uncontroversial for twelve of
#: the fourteen. It is *not* for the two either side of it -- i10_touch_share at
#: 0.263 and yac_per_reception at 0.229 -- and `MetricStability.verdict` is a
#: convenience, not the finding. Read the reliability curve before betting on
#: either of those.
STABLE_LAG1 = 0.25

#: Conventional reliability bar for "this many games is enough to believe it".
STABILIZATION_THRESHOLD = 0.70


@dataclass(frozen=True, slots=True)
class MetricStability:
    """How much of a metric is signal, measured three independent ways."""

    metric: str
    #: Pearson correlation of one appearance with the player's *next appearance*,
    #: pooled within player-season. Not strictly week w against week w+1: a week the
    #: volume gate drops (or a week he missed) is skipped rather than breaking the
    #: chain, so 14% of target-share pairs and 22% of td-rate pairs span a gap.
    #: Measured both ways on 2019-2025, the difference is at most 0.013 on every
    #: metric except racr (0.045 pooled, 0.018 strictly adjacent).
    lag1: float
    lag1_n: int
    #: Reliability of a 3-game window: first 3 appearances against the next 3. A raw
    #: split-half correlation, deliberately NOT Spearman-Brown corrected -- see
    #: `_split_half`.
    split_half_3: float
    #: Correlation of season t with season t+1 on volume-weighted aggregates.
    season_over_season: float
    season_n: int
    #: Games at which split-half reliability first clears STABILIZATION_THRESHOLD,
    #: or None if it never does inside the search window.
    games_to_stabilize: int | None
    reliability_curve: tuple[tuple[int, float], ...]

    @property
    def stable(self) -> bool:
        return math.isfinite(self.lag1) and self.lag1 >= STABLE_LAG1

    @property
    def verdict(self) -> str:
        return "project forward" if self.stable else "regress to mean"


def _relevant(
    frame: pl.DataFrame, min_games: int, min_opportunities_per_game: float
) -> pl.DataFrame:
    """Restrict to fantasy-relevant player-seasons.

    Reliability is a property of a population, not of a metric. Throw in every
    third receiver who caught one pass and between-player variance inflates, making
    every metric look more stable than it is (target share's 1-game reliability
    moves 0.532 -> 0.578, and its lag-1 0.567 -> 0.636, on the unfiltered
    population). This is roughly the set a manager actually chooses between.
    """
    keep = (
        frame.group_by("player_id", "season")
        .agg(pl.len().alias("_g"), pl.col("opportunities").mean().alias("_opg"))
        .filter((pl.col("_g") >= min_games) & (pl.col("_opg") >= min_opportunities_per_game))
        .select("player_id", "season")
    )
    return frame.join(keep, on=["player_id", "season"], how="inner")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _present(frame: pl.DataFrame, metric: StabilityMetric) -> bool:
    """Whether the frame even carries this metric.

    A caller may pass a frame built without play-by-play, or ask about a metric no
    free source publishes. Neither is an error -- the answer is "not measured".
    """
    return metric.name in frame.columns and metric.volume in frame.columns


def _usable(frame: pl.DataFrame, metric: StabilityMetric) -> pl.DataFrame:
    return frame.filter(
        pl.col(metric.volume).is_not_null()
        & (pl.col(metric.volume) >= metric.min_volume)
        & pl.col(metric.name).is_not_null()
        & pl.col(metric.name).is_finite()
    )


def _lag1(frame: pl.DataFrame, metric: StabilityMetric) -> tuple[float, int]:
    """Correlation of an appearance with the *next appearance*, within player-season.

    The shift is over rows that survived the volume gate, not over week numbers, so a
    missed or gated-out week is skipped rather than ending the chain. That is the
    operational question ("what does his last game say about his next one") and it is
    not literally lag-1 in weeks: 14% of target-share pairs and 22% of td-rate pairs
    span a gap, up to 16 weeks. Restricting to strictly adjacent weeks moves nothing
    that matters -- at most 0.013 on any metric except racr (0.045 -> 0.018) --
    which `test_lag1_is_next_appearance_not_next_week` pins.
    """
    if not _present(frame, metric):
        return float("nan"), 0
    d = _usable(frame, metric).sort("player_id", "season", "week")
    d = d.with_columns(_next=pl.col(metric.name).shift(-1).over("player_id", "season")).filter(
        pl.col("_next").is_not_null()
    )
    if d.height < 30:
        return float("nan"), d.height
    return _corr(d[metric.name].to_numpy(), d["_next"].to_numpy()), d.height


def _season_over_season(
    frame: pl.DataFrame, metric: StabilityMetric, min_games: int
) -> tuple[float, int]:
    if not _present(frame, metric):
        return float("nan"), 0
    d = _usable(frame, metric).filter(pl.col(metric.volume) > 0)
    agg = (
        d.group_by("player_id", "season")
        .agg(
            (pl.col(metric.name) * pl.col(metric.volume)).sum().alias("_num"),
            pl.col(metric.volume).sum().alias("_den"),
            pl.len().alias("_games"),
        )
        .filter((pl.col("_games") >= min_games) & (pl.col("_den") > 0))
        .select("player_id", "season", _value=pl.col("_num") / pl.col("_den"))
    )
    nxt = agg.with_columns(pl.col("season") - 1).rename({"_value": "_next"})
    j = agg.join(nxt, on=["player_id", "season"], how="inner").drop_nulls()
    if j.height < 30:
        return float("nan"), j.height
    return _corr(j["_value"].to_numpy(), j["_next"].to_numpy()), j.height


def _split_half(frame: pl.DataFrame, metric: StabilityMetric, games: int) -> float:
    """Reliability of a `games`-game window: two disjoint windows against each other.

    A player's first `games` appearances against his next `games`, each aggregated
    volume-weighted. **No Spearman-Brown correction.** Two disjoint windows of the
    same length are parallel measurements of one latent talent, so their correlation
    already *is* the reliability of a `games`-game window -- true-score variance over
    observed variance. Spearman-Brown belongs on a split of the window being reported
    on, into two halves of `games/2`; applied to two full-length windows it returns
    the reliability of `2*games` instead. That mistake is silent and flattering: it
    reads as "target share stabilizes at 2 games" when the honest answer is 3, and
    it doubles every reliability in the table. Pinned by a closed-form test
    (`test_split_half_matches_closed_form_reliability`).
    """
    if not _present(frame, metric):
        return float("nan")
    d = _usable(frame, metric).filter(pl.col(metric.volume) > 0).sort("player_id", "season", "week")
    d = d.with_columns(_idx=pl.int_range(pl.len()).over("player_id", "season")).filter(
        pl.col("_idx") < 2 * games
    )
    complete = (
        d.group_by("player_id", "season")
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") == 2 * games)
        .select("player_id", "season")
    )
    d = d.join(complete, on=["player_id", "season"], how="inner")
    halves = (
        d.with_columns(_half=pl.col("_idx") >= games)
        .group_by("player_id", "season", "_half")
        .agg(
            (pl.col(metric.name) * pl.col(metric.volume)).sum().alias("_num"),
            pl.col(metric.volume).sum().alias("_den"),
        )
        .filter(pl.col("_den") > 0)
        .with_columns(_value=pl.col("_num") / pl.col("_den"))
    )
    first = halves.filter(~pl.col("_half")).select("player_id", "season", _a=pl.col("_value"))
    second = halves.filter(pl.col("_half")).select("player_id", "season", _b=pl.col("_value"))
    j = first.join(second, on=["player_id", "season"], how="inner").drop_nulls()
    if j.height < 30:
        return float("nan")
    r = _corr(j["_a"].to_numpy(), j["_b"].to_numpy())
    return r if math.isfinite(r) else float("nan")


def stability_table(
    frame: pl.DataFrame,
    *,
    metrics: Sequence[StabilityMetric] = DEFAULT_STABILITY_METRICS,
    min_games: int = 6,
    min_opportunities_per_game: float = 2.0,
    max_window: int = 8,
) -> tuple[MetricStability, ...]:
    """The module's most valuable output: what to project and what to regress.

    Three views of the same question, because any one of them can be fooled. Lag-1
    autocorrelation is the cheapest and answers "does last week say anything about
    next week". Split-half reliability answers "how many games before I believe
    it". Season-over-season answers "is this a property of the player or of the
    situation" -- and it is the one that flatters unstable metrics, because a team's
    offense persists across a season boundary even when a player's efficiency does
    not. YAC per reception is the clean example: 0.23 week to week, 0.69 year to
    year. That is a real skill measured very noisily, not a projectable weekly one.

    `max_window` is capped by arithmetic, not taste: reliability at n games is read
    off two disjoint n-game windows, so it needs 2n appearances inside one season
    and n=8 is the most a 17-game season can support.
    """
    population = _relevant(frame, min_games, min_opportunities_per_game)
    out: list[MetricStability] = []
    for metric in metrics:
        lag1, lag1_n = _lag1(population, metric)
        sos, sos_n = _season_over_season(population, metric, min_games)
        curve = tuple((n, _split_half(population, metric, n)) for n in range(1, max_window + 1))
        out.append(
            MetricStability(
                metric=metric.name,
                lag1=lag1,
                lag1_n=lag1_n,
                split_half_3=dict(curve).get(3, float("nan")),
                season_over_season=sos,
                season_n=sos_n,
                games_to_stabilize=next(
                    (n for n, r in curve if math.isfinite(r) and r >= STABILIZATION_THRESHOLD),
                    None,
                ),
                reliability_curve=curve,
            )
        )
    return tuple(sorted(out, key=lambda r: -r.lag1 if math.isfinite(r.lag1) else 1.0))


def format_stability_table(rows: Sequence[MetricStability]) -> str:
    """Fixed-width rendering, for the CLI and for pasting into a note."""
    header = (
        f"{'metric':<20}{'lag1':>7}{'n':>8}{'3g rel':>9}{'season':>8}{'n':>7}{'games':>7}  verdict"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        stab = "--" if r.games_to_stabilize is None else str(r.games_to_stabilize)
        lines.append(
            f"{r.metric:<20}{r.lag1:>7.3f}{r.lag1_n:>8d}{r.split_half_3:>9.3f}"
            f"{r.season_over_season:>8.3f}{r.season_n:>7d}{stab:>7}  {r.verdict}"
        )
    return "\n".join(lines)


def shrink_to_mean(
    observed: float, prior: float, n_observations: float, stabilization: float
) -> float:
    """Regress an unstable metric toward its positional mean.

    `(n*observed + k*prior) / (n + k)`, with `k` the number of observations at which
    the metric is half signal -- which is what the reliability curve measures. This
    is the operational payoff of the stability table: an unstable metric does not
    mean "ignore it", it means "weight it by k".
    """
    if n_observations < 0 or stabilization <= 0:
        raise ValueError("n_observations must be >= 0 and stabilization > 0")
    return (n_observations * observed + stabilization * prior) / (n_observations + stabilization)


# --------------------------------------------------------------------------------------
# Expected fantasy points
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpportunityValue:
    """What one carry or one target converts into, on average, in one field bucket.

    Components rather than points, because points are league-specific: the same cell
    serves full PPR and half PPR by being re-scored. It is also what makes the
    regression screen exact -- points are linear in these four, so actual minus
    expected splits into four terms that sum back to the gap with no residual.
    """

    catch_rate: float
    yards: float
    td_rate: float
    fumble_lost_rate: float
    n: int

    def stats(self, kind: str) -> dict[str, float]:
        """The component vector as ESPN statIds, ready for a `LeagueContext.scorer`."""
        if kind == "carry":
            return {
                STAT_RUSHING_YARDS: self.yards,
                STAT_RUSHING_TD: self.td_rate,
                STAT_FUMBLE_LOST: self.fumble_lost_rate,
            }
        if kind == "target":
            return {
                STAT_RECEPTIONS: self.catch_rate,
                STAT_RECEIVING_YARDS: self.yards,
                STAT_RECEIVING_TD: self.td_rate,
                STAT_FUMBLE_LOST: self.fumble_lost_rate,
            }
        raise ValueError(f"kind must be 'carry' or 'target', got {kind!r}")

    def points(self, kind: str, scorer: Scorer, position_id: int) -> float:
        return scorer(self.stats(kind), position_id)


OPPORTUNITY_KINDS: tuple[str, ...] = ("carry", "target")


@dataclass(frozen=True, slots=True)
class ConversionRates:
    """(position, kind, bucket) -> `OpportunityValue`, with a pooled fallback.

    The pooled row is keyed on position `"*"`, and a thin cell falls back to it: WR
    carries inside the 10 are 149 plays in seven seasons, which is not a number a
    projection should lean on.
    """

    cells: Mapping[tuple[str, str, str], OpportunityValue]
    seasons: tuple[int, ...]
    #: Cells with fewer plays than this defer to the pooled row.
    min_cell: int = 300

    def get(self, position: str, kind: str, bucket: str) -> OpportunityValue:
        cell = self.cells.get((position, kind, bucket))
        if cell is not None and cell.n >= self.min_cell:
            return cell
        pooled = self.cells.get(("*", kind, bucket))
        if pooled is None:
            if cell is not None:
                return cell
            raise KeyError(f"no conversion rate for ({position!r}, {kind!r}, {bucket!r})")
        return pooled

    def points(self, position: str, kind: str, bucket: str, scorer: Scorer) -> float:
        return self.get(position, kind, bucket).points(kind, scorer, POSITION_IDS[position])

    def summary(self, scorer: Scorer = PPR) -> pl.DataFrame:
        """One row per cell with the scored value -- the table to eyeball."""
        order = {b: i for i, b in enumerate(FIELD_BUCKETS)}
        rows = [
            {
                "position": position,
                "kind": kind,
                "bucket": bucket,
                "n": cell.n,
                "catch_rate": cell.catch_rate,
                "yards": cell.yards,
                "td_rate": cell.td_rate,
                "fumble_lost_rate": cell.fumble_lost_rate,
                "points": cell.points(kind, scorer, POSITION_IDS.get(position, WR)),
            }
            for (position, kind, bucket), cell in self.cells.items()
        ]
        return pl.DataFrame(rows).sort(
            "position", "kind", pl.col("bucket").replace_strict(order, return_dtype=pl.Int32)
        )


#: Fitted on 2019-2025 regular-season play-by-play, RB/WR/TE, two-point plays
#: excluded, fumbles attributed to the ball carrier only. Regenerate with
#: `fit_conversion_rates(range(2019, 2026))`; a test checks these against a refit.
DEFAULT_CONVERSION_RATES = ConversionRates(
    cells={
        ("RB", "carry", "goal_line"): OpportunityValue(0.0, 1.102, 0.39737, 0.00779, 4107),
        ("RB", "carry", "inside_10"): OpportunityValue(0.0, 2.812, 0.10137, 0.00435, 2989),
        ("RB", "carry", "fringe_rz"): OpportunityValue(0.0, 3.7578, 0.04094, 0.00457, 6570),
        ("RB", "carry", "midfield"): OpportunityValue(0.0, 4.4916, 0.00912, 0.00396, 25006),
        ("RB", "carry", "own_half"): OpportunityValue(0.0, 4.7053, 0.00275, 0.00404, 42533),
        ("RB", "target", "goal_line"): OpportunityValue(0.67828, 1.6971, 0.49062, 0.00536, 373),
        ("RB", "target", "inside_10"): OpportunityValue(0.70588, 3.5342, 0.25278, 0.00318, 629),
        ("RB", "target", "fringe_rz"): OpportunityValue(0.74267, 4.6265, 0.08958, 0.00163, 1842),
        ("RB", "target", "midfield"): OpportunityValue(0.77771, 5.8986, 0.01912, 0.00533, 6748),
        ("RB", "target", "own_half"): OpportunityValue(0.78495, 6.0975, 0.00221, 0.00449, 12690),
        ("WR", "carry", "goal_line"): OpportunityValue(0.0, 1.3382, 0.5, 0.0, 68),
        ("WR", "carry", "inside_10"): OpportunityValue(0.0, 3.3086, 0.25926, 0.0, 81),
        ("WR", "carry", "fringe_rz"): OpportunityValue(0.0, 5.3623, 0.11594, 0.01449, 345),
        ("WR", "carry", "midfield"): OpportunityValue(0.0, 5.912, 0.0246, 0.00568, 1057),
        ("WR", "carry", "own_half"): OpportunityValue(0.0, 6.2649, 0.00363, 0.00653, 1378),
        ("WR", "target", "goal_line"): OpportunityValue(0.50559, 1.4527, 0.44614, 0.00102, 1968),
        ("WR", "target", "inside_10"): OpportunityValue(0.52596, 3.2863, 0.29452, 0.00146, 2061),
        ("WR", "target", "fringe_rz"): OpportunityValue(0.59743, 5.3697, 0.1506, 0.0029, 4834),
        ("WR", "target", "midfield"): OpportunityValue(0.62785, 7.9639, 0.04211, 0.00337, 22252),
        ("WR", "target", "own_half"): OpportunityValue(0.65015, 8.88, 0.00807, 0.00376, 40380),
        ("TE", "carry", "goal_line"): OpportunityValue(0.0, 0.8939, 0.48485, 0.0, 66),
        ("TE", "carry", "inside_10"): OpportunityValue(0.0, 3.193, 0.17544, 0.0, 57),
        ("TE", "carry", "fringe_rz"): OpportunityValue(0.0, 3.6522, 0.04348, 0.01449, 69),
        ("TE", "carry", "midfield"): OpportunityValue(0.0, 4.8319, 0.0177, 0.0177, 226),
        ("TE", "carry", "own_half"): OpportunityValue(0.0, 5.6207, 0.00766, 0.01149, 261),
        ("TE", "target", "goal_line"): OpportunityValue(0.60903, 1.6023, 0.53506, 0.00288, 1041),
        ("TE", "target", "inside_10"): OpportunityValue(0.56573, 3.5765, 0.31897, 0.00431, 928),
        ("TE", "target", "fringe_rz"): OpportunityValue(0.62072, 5.6962, 0.15659, 0.00422, 2133),
        ("TE", "target", "midfield"): OpportunityValue(0.70831, 7.8367, 0.02651, 0.00296, 7772),
        ("TE", "target", "own_half"): OpportunityValue(0.72057, 7.8932, 0.00174, 0.00494, 14383),
        ("*", "carry", "goal_line"): OpportunityValue(0.0, 1.1026, 0.40038, 0.00755, 4241),
        ("*", "carry", "inside_10"): OpportunityValue(0.0, 2.8318, 0.10681, 0.00416, 3127),
        ("*", "carry", "fringe_rz"): OpportunityValue(0.0, 3.8361, 0.04467, 0.00515, 6984),
        ("*", "carry", "midfield"): OpportunityValue(0.0, 4.5517, 0.00981, 0.00415, 26289),
        ("*", "carry", "own_half"): OpportunityValue(0.0, 4.7594, 0.00281, 0.00417, 44172),
        ("*", "target", "goal_line"): OpportunityValue(0.55648, 1.5257, 0.47842, 0.00207, 3382),
        ("*", "target", "inside_10"): OpportunityValue(0.56744, 3.4038, 0.29353, 0.00249, 3618),
        ("*", "target", "fringe_rz"): OpportunityValue(0.63344, 5.2933, 0.13929, 0.00295, 8809),
        ("*", "target", "midfield"): OpportunityValue(0.67236, 7.558, 0.03459, 0.00364, 36772),
        ("*", "target", "own_half"): OpportunityValue(0.69053, 8.1461, 0.00562, 0.00415, 67453),
    },
    seasons=tuple(range(2019, 2026)),
)


def fit_conversion_rates(
    seasons: int | Iterable[int],
    *,
    positions: Sequence[str] = OPPORTUNITY_POSITIONS,
    cache: nv.NflverseCache | None = None,
    allow_missing: bool = False,
    min_cell: int = 300,
) -> ConversionRates:
    """Re-measure per-opportunity conversion from play-by-play.

    Fumbles are attributed to the ball carrier only (`fumbled_1_player_id`), so a
    quarterback's aborted snap does not land on the back who was not holding it.
    """
    wanted = _seasons_tuple(seasons)
    paths, loaded = _pbp_paths(wanted, cache=cache, allow_missing=allow_missing)
    scan = _pbp_scan(paths)
    position_map = (
        nv.player_week_stats(loaded, season_type="REG", allow_missing=allow_missing, cache=cache)
        .select(pl.col("season").cast(pl.Int32), "player_id", "position")
        .unique()
    )

    def side(id_col: str, gate: str, yards: str, td: str, catch: bool) -> pl.DataFrame:
        return (
            scan.filter((pl.col(gate).fill_null(0) == 1) & pl.col(id_col).is_not_null())
            .select(
                "season",
                pl.col(id_col).alias("player_id"),
                _bucket_expr(),
                pl.col(yards).fill_null(0.0).alias("_yards"),
                pl.col(td).fill_null(0).cast(pl.Float64).alias("_td"),
                ((pl.col("fumble_lost") == 1) & (pl.col("fumbled_1_player_id") == pl.col(id_col)))
                .cast(pl.Float64)
                .alias("_fumble"),
                (
                    pl.col("complete_pass").fill_null(0).cast(pl.Float64) if catch else pl.lit(0.0)
                ).alias("_catch"),
            )
            .collect()
            .join(position_map, on=["season", "player_id"], how="left")
            .filter(pl.col("position").is_in(list(positions)))
        )

    plays = {
        "carry": side("rusher_player_id", "rush_attempt", "rushing_yards", "rush_touchdown", False),
        "target": side(
            "receiver_player_id", "pass_attempt", "receiving_yards", "pass_touchdown", True
        ),
    }
    aggs = (
        pl.len().alias("n"),
        pl.col("_catch").mean().alias("catch_rate"),
        pl.col("_yards").mean().alias("yards"),
        pl.col("_td").mean().alias("td_rate"),
        pl.col("_fumble").mean().alias("fumble_lost_rate"),
    )

    cells: dict[tuple[str, str, str], OpportunityValue] = {}
    for kind, df in plays.items():
        for keyed, group_by in ((True, ["position", "bucket"]), (False, ["bucket"])):
            for row in df.group_by(group_by).agg(*aggs).iter_rows(named=True):
                position = row["position"] if keyed else "*"
                cells[(position, kind, row["bucket"])] = OpportunityValue(
                    catch_rate=row["catch_rate"],
                    yards=row["yards"],
                    td_rate=row["td_rate"],
                    fumble_lost_rate=row["fumble_lost_rate"],
                    n=int(row["n"]),
                )
    return ConversionRates(cells=cells, seasons=loaded, min_cell=min_cell)


def component_prices(scorer: Scorer, position: str) -> dict[str, float]:
    """Points per unit of each component, in this league, for this position."""
    pid = POSITION_IDS[position]
    return {stat: scorer({stat: 1.0}, pid) for stat in COMPONENT_STATS}


def _per_position_expr(build: Callable[[str], pl.Expr]) -> pl.Expr:
    """Dispatch an expression on the `position` column; null for anything else."""
    expr = pl.when(pl.col("position") == OPPORTUNITY_POSITIONS[0]).then(
        build(OPPORTUNITY_POSITIONS[0])
    )
    for position in OPPORTUNITY_POSITIONS[1:]:
        expr = expr.when(pl.col("position") == position).then(build(position))
    return expr.otherwise(None)


EXPECTED_COLUMNS: tuple[str, ...] = (
    "expected_rushing_yards",
    "expected_receiving_yards",
    "expected_rushing_tds",
    "expected_receiving_tds",
    "expected_receptions_rate",
    "expected_fumbles_lost",
    "xfp",
)


def expected_points(
    frame: pl.DataFrame,
    *,
    rates: ConversionRates = DEFAULT_CONVERSION_RATES,
    scorer: Scorer = PPR,
) -> pl.DataFrame:
    """Add `xfp` and the expected components it decomposes into.

    Every carry and every target is priced at what that opportunity returns on
    average from that spot on the field. That is the whole model: there is no
    efficiency term, by construction, because efficiency is the part that does not
    repeat.

    Needs the field-position columns, so `usage_weeks(..., field_position=True)`.
    Rows whose buckets are null (a season nflverse has not built play-by-play for)
    come back with null everywhere rather than a confident zero.
    """
    if missing := [c for c in BUCKET_COLUMNS if c not in frame.columns]:
        raise ValueError(
            f"frame is missing {missing}; expected_points needs the play-by-play "
            "columns from usage_weeks(field_position=True)."
        )
    all_null = pl.all_horizontal([pl.col(c).is_null().all() for c in BUCKET_COLUMNS])
    if frame.height and frame.select(all_null).item():
        raise ValueError(
            "field-position columns are entirely null -- play-by-play was unavailable, "
            "so there is nothing to build xFP from."
        )

    def component(kind: str, field: str) -> Callable[[str], pl.Expr]:
        plural = _plural(kind)

        def build(position: str) -> pl.Expr:
            return _sum_cols_scaled(
                [
                    (f"{plural}_{bucket}", getattr(rates.get(position, kind, bucket), field))
                    for bucket in FIELD_BUCKETS
                ]
            )

        return build

    def value(position: str) -> pl.Expr:
        return _sum_cols_scaled(
            [
                (f"{_plural(kind)}_{bucket}", rates.points(position, kind, bucket, scorer))
                for kind in OPPORTUNITY_KINDS
                for bucket in FIELD_BUCKETS
            ]
        )

    return frame.with_columns(
        expected_rushing_yards=_per_position_expr(component("carry", "yards")),
        expected_receiving_yards=_per_position_expr(component("target", "yards")),
        expected_rushing_tds=_per_position_expr(component("carry", "td_rate")),
        expected_receiving_tds=_per_position_expr(component("target", "td_rate")),
        expected_receptions_rate=_per_position_expr(component("target", "catch_rate")),
        expected_fumbles_lost=_per_position_expr(component("carry", "fumble_lost_rate"))
        + _per_position_expr(component("target", "fumble_lost_rate")),
        xfp=_per_position_expr(value),
    )


def _sum_cols_scaled(terms: Sequence[tuple[str, float]]) -> pl.Expr:
    expr: pl.Expr | None = None
    for column, weight in terms:
        term = pl.col(column) * weight
        expr = term if expr is None else expr + term
    if expr is None:
        raise ValueError("no terms to sum")
    return expr


def actual_points_expr(scorer: Scorer = PPR) -> pl.Expr:
    """The same six components, scored -- the like-for-like partner of `xfp`.

    Deliberately not `fantasy_points_ppr`: that carries return yards, two-point
    conversions and passing, none of which xFP models, and a gap between actual and
    expected has to be a gap in the *modelled* components or the decomposition is a
    fiction. On 2024 the two differ for 1.9% of player-weeks, all of them returns
    and two-point plays.
    """

    def build(position: str) -> pl.Expr:
        price = component_prices(scorer, position)
        return (
            price[STAT_RECEPTIONS] * pl.col("receptions")
            + price[STAT_RECEIVING_YARDS] * pl.col("receiving_yards")
            + price[STAT_RECEIVING_TD] * pl.col("receiving_tds")
            + price[STAT_RUSHING_YARDS] * pl.col("rushing_yards")
            + price[STAT_RUSHING_TD] * pl.col("rushing_tds")
            + price[STAT_FUMBLE_LOST] * pl.col("fumbles_lost")
        )

    return _per_position_expr(build)


# --------------------------------------------------------------------------------------
# The regression screen
# --------------------------------------------------------------------------------------

#: Which unstable metric a points gap is attributed to.
DRIVERS: tuple[str, ...] = ("touchdowns", "catch_rate", "yards_per_opportunity", "fumbles")

#: Points per game of gap at which a player is worth acting on. 1.5 PPR points is
#: ~1.7pp of weekly win probability at an even matchup (sd_diff 34.4), which is the
#: smallest edge worth a roster move.
ACTIONABLE_GAP_PER_GAME = 1.5


@dataclass(frozen=True, slots=True)
class RegressionCandidate:
    """One player-season's gap between what he scored and what his usage was worth.

    `points_gap` is the season total; `gap_per_game` is what to act on, because 30
    points of variance banked over 17 games is a different problem from 30 over 6.

    The four `*_over_expected` terms sum to `points_gap` exactly -- points are linear
    in the components, so there is no residual to wave away. **Which term the gap
    lives in is the whole point.** Measured on 686 consecutive player-season pairs,
    the year-to-year correlation of each component per game is:

        td_over_expected      0.10   sd 1.01 pts/g   <- does not repeat
        fumbles_over_expected 0.05   sd 0.11 pts/g   <- does not repeat
        yards_over_expected   0.33   sd 0.92 pts/g   <- partly a real player
        catch_over_expected   0.39   sd 0.38 pts/g   <- partly a real player
        points_gap (total)    0.28   sd 1.76 pts/g

    So a 40-point sell-high built on touchdowns is a completely different animal
    from a 40-point sell-high built on yards per target, and `unrepeatable_points`
    is the slice the evidence says will be gone next year.
    """

    player_id: str
    name: str
    position: str
    season: int
    games: int
    carries: float
    targets: float
    actual_points: float
    expected_points: float
    points_gap: float
    gap_per_game: float
    td_over_expected: float
    catch_over_expected: float
    yards_over_expected: float
    fumbles_over_expected: float
    touchdowns: float
    expected_touchdowns: float
    driver: str
    verdict: str

    @property
    def unrepeatable_points(self) -> float:
        """The TD and fumble slice -- the part with ~zero year-to-year correlation."""
        return self.td_over_expected + self.fumbles_over_expected

    @property
    def unrepeatable_per_game(self) -> float:
        return self.unrepeatable_points / self.games if self.games else 0.0

    @property
    def significant(self) -> bool:
        return abs(self.gap_per_game) >= ACTIONABLE_GAP_PER_GAME


def regression_screen(
    frame: pl.DataFrame,
    *,
    season: int | None = None,
    through_week: int | None = None,
    rates: ConversionRates = DEFAULT_CONVERSION_RATES,
    scorer: Scorer = PPR,
    min_games: int = 4,
    min_opportunities: float = 40.0,
    sort_by: str = "gap",
) -> tuple[RegressionCandidate, ...]:
    """Rank sell-highs and buy-lows by actual minus expected points.

    A positive gap means the player scored more than his usage was worth; a negative
    gap is a buy-low. Returned most-overperforming first, so `[:10]` is the sell list
    and the last ten reversed are the buy list -- `format_screen` prints both.

    `sort_by="unrepeatable"` ranks on the touchdown-and-fumble slice instead of the
    whole gap. Prefer it when the question is "what will not repeat" rather than
    "who outscored his usage": measured over 686 player-season pairs, the total gap
    persists at r=0.27 while its TD component persists at r=0.09. The two lists are
    genuinely different -- a receiver overperforming on yards per target is often a
    real deep threat, not a lucky one.
    """
    if sort_by not in ("gap", "unrepeatable"):
        raise ValueError(f"sort_by must be 'gap' or 'unrepeatable', got {sort_by!r}")
    d = frame
    if season is not None:
        d = d.filter(pl.col("season") == season)
    if through_week is not None:
        d = d.filter(pl.col("week") <= through_week)
    if d.height == 0:
        return ()

    d = expected_points(d, rates=rates, scorer=scorer).with_columns(
        _actual=actual_points_expr(scorer)
    )
    dropped = d.select(pl.col("xfp").is_null().sum()).item()
    if dropped:
        # Null xFP means the season had no play-by-play. Summing over it would
        # silently understate the expected side instead of saying nothing.
        log.warning("dropping %d player-weeks with no play-by-play from the screen", dropped)
        d = d.filter(pl.col("xfp").is_not_null())
    if d.height == 0:
        return ()

    price = {position: component_prices(scorer, position) for position in OPPORTUNITY_POSITIONS}

    def priced(stat: str) -> pl.Expr:
        expr = pl.when(pl.col("position") == OPPORTUNITY_POSITIONS[0]).then(
            pl.lit(price[OPPORTUNITY_POSITIONS[0]][stat])
        )
        for position in OPPORTUNITY_POSITIONS[1:]:
            expr = expr.when(pl.col("position") == position).then(pl.lit(price[position][stat]))
        return expr.otherwise(0.0)

    d = d.with_columns(
        _td_oe=(
            priced(STAT_RECEIVING_TD) * (pl.col("receiving_tds") - pl.col("expected_receiving_tds"))
            + priced(STAT_RUSHING_TD) * (pl.col("rushing_tds") - pl.col("expected_rushing_tds"))
        ),
        _catch_oe=priced(STAT_RECEPTIONS)
        * (pl.col("receptions") - pl.col("expected_receptions_rate")),
        _yards_oe=(
            priced(STAT_RECEIVING_YARDS)
            * (pl.col("receiving_yards") - pl.col("expected_receiving_yards"))
            + priced(STAT_RUSHING_YARDS)
            * (pl.col("rushing_yards") - pl.col("expected_rushing_yards"))
        ),
        _fumble_oe=priced(STAT_FUMBLE_LOST)
        * (pl.col("fumbles_lost") - pl.col("expected_fumbles_lost")),
    )

    agg = (
        d.group_by("player_id", "season")
        .agg(
            pl.col("player_name").last().alias("name"),
            pl.col("position").last().alias("position"),
            pl.len().alias("games"),
            pl.col("carries").sum().alias("carries"),
            pl.col("targets").sum().alias("targets"),
            pl.col("opportunities").sum().alias("opportunities"),
            pl.col("_actual").sum().alias("actual_points"),
            pl.col("xfp").sum().alias("expected_points"),
            pl.col("_td_oe").sum().alias("td_over_expected"),
            pl.col("_catch_oe").sum().alias("catch_over_expected"),
            pl.col("_yards_oe").sum().alias("yards_over_expected"),
            pl.col("_fumble_oe").sum().alias("fumbles_over_expected"),
            pl.col("total_tds").sum().alias("touchdowns"),
            (pl.col("expected_receiving_tds") + pl.col("expected_rushing_tds"))
            .sum()
            .alias("expected_touchdowns"),
        )
        .filter((pl.col("games") >= min_games) & (pl.col("opportunities") >= min_opportunities))
        .with_columns(points_gap=pl.col("actual_points") - pl.col("expected_points"))
        .with_columns(gap_per_game=pl.col("points_gap") / pl.col("games"))
        .sort(
            "points_gap"
            if sort_by == "gap"
            else pl.col("td_over_expected") + pl.col("fumbles_over_expected"),
            descending=True,
        )
    )

    out: list[RegressionCandidate] = []
    for row in agg.iter_rows(named=True):
        components = {
            "touchdowns": row["td_over_expected"],
            "catch_rate": row["catch_over_expected"],
            "yards_per_opportunity": row["yards_over_expected"],
            "fumbles": row["fumbles_over_expected"],
        }
        driver = max(components, key=lambda k: abs(components[k]))
        gap = row["gap_per_game"]
        out.append(
            RegressionCandidate(
                player_id=row["player_id"],
                name=row["name"],
                position=row["position"],
                season=int(row["season"]),
                games=int(row["games"]),
                carries=float(row["carries"]),
                targets=float(row["targets"]),
                actual_points=float(row["actual_points"]),
                expected_points=float(row["expected_points"]),
                points_gap=float(row["points_gap"]),
                gap_per_game=float(gap),
                td_over_expected=float(row["td_over_expected"]),
                catch_over_expected=float(row["catch_over_expected"]),
                yards_over_expected=float(row["yards_over_expected"]),
                fumbles_over_expected=float(row["fumbles_over_expected"]),
                touchdowns=float(row["touchdowns"]),
                expected_touchdowns=float(row["expected_touchdowns"]),
                driver=driver,
                verdict=(
                    "sell_high"
                    if gap >= ACTIONABLE_GAP_PER_GAME
                    else "buy_low"
                    if gap <= -ACTIONABLE_GAP_PER_GAME
                    else "fair"
                ),
            )
        )
    return tuple(out)


def format_screen(candidates: Sequence[RegressionCandidate], *, top: int = 10) -> str:
    """The sell-high and buy-low lists, one block of text."""
    if not candidates:
        return "(no candidates)"
    header = (
        f"{'player':<24}{'pos':>4}{'g':>4}{'act':>8}{'xFP':>8}{'gap':>8}"
        f"{'/gm':>7}{'TD':>6}{'TDx':>7}  driver"
    )

    def block(rows: Sequence[RegressionCandidate], title: str) -> list[str]:
        lines = [title, header, "-" * len(header)]
        if not rows:
            lines.append("(none)")
        for c in rows:
            lines.append(
                f"{c.name[:23]:<24}{c.position:>4}{c.games:>4}{c.actual_points:>8.1f}"
                f"{c.expected_points:>8.1f}{c.points_gap:>8.1f}{c.gap_per_game:>7.2f}"
                f"{c.touchdowns:>6.0f}{c.expected_touchdowns:>7.1f}  {c.driver}"
            )
        return lines

    # The two blocks must not overlap. `candidates[:top]` and `candidates[-top:]`
    # intersect whenever there are fewer than 2*top of them, which printed the same
    # player as both a sell-high and a buy-low. When the list is too short to fill
    # both, split it at the midpoint rather than letting the sell side eat it.
    total = len(candidates)
    n_sell = min(top, (total + 1) // 2)
    n_buy = min(top, total - n_sell)
    sell = list(candidates[:n_sell])
    buy = list(reversed(candidates[total - n_buy :])) if n_buy else []
    lines = block(sell, "SELL HIGH (scored above opportunity)")
    lines.append("")
    lines.extend(block(buy, "BUY LOW (scored below opportunity)"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# ffopportunity (optional, and it has a hole in it)
# --------------------------------------------------------------------------------------

#: ffopportunity ships from its own repo, not nflverse-data, so `NflverseCache`'s
#: release base does not reach it and this fetches directly.
FFOPPORTUNITY_BASE = "https://github.com/ffverse/ffopportunity/releases/download"
FFOPPORTUNITY_TAG = "latest-data"

#: Measured fumble-loss rate per opportunity, pooled RB/WR/TE over every field
#: bucket. Used to give ffopportunity's expected side the fumble term it lacks.
FUMBLE_LOST_PER_OPPORTUNITY = 0.0042


def ffopportunity_weekly(
    season: int,
    *,
    root: Path | str = "data/cache/ffopportunity",
    client: httpx.Client | None = None,
    correct_fumbles: bool = True,
    fumble_rate: float = FUMBLE_LOST_PER_OPPORTUNITY,
) -> pl.DataFrame:
    """ffverse's expected-points model, with its fumble bias corrected.

    Their `fantasy_points` is **full PPR minus 2 per fumble lost** -- verified exact
    on 5,878 of 6,005 rows of 2024 and to a -2.016 mean on the 127 fumble rows --
    while `fantasy_points_exp` carries no fumble term at all. So
    `total_fantasy_points_diff` charges a fumble-prone back the full two points as
    if it were bad luck relative to expectation, when expectation never contained
    the fumble in the first place.

    `total_fantasy_points_diff_adj` puts an expected fumble cost back on the
    **expected** side and leaves the actual side alone:
    `diff + 2*opportunities*fumble_rate`. That is what our own xFP does structurally
    -- `actual - xfp` carries `-2*(fumbles_lost - expected_fumbles_lost)` -- which is
    why this is a cross-check rather than a dependency.

    Do not write `diff + 2*(fumbles_lost - opportunities*fumble_rate)`. It looks like
    the same repair and is not: the `+2*fumbles_lost` cancels the penalty on the
    *actual* side, so a back who fumbled five times is charged nothing for them while
    the expected side is charged an expected fumble anyway. Measured on 2024, that
    version drives corr(adj, fumbles) to 0.002 -- fumbling becomes free -- and leaves
    a constant -2*opp*rate bias on every clean row. The correct form keeps
    corr(adj, fumbles) at -0.072, matching the raw diff with only its mean removed.
    Note it is full PPR.
    """
    filename = f"ep_weekly_{season}.parquet"
    dest = Path(root) / filename
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{FFOPPORTUNITY_BASE}/{FFOPPORTUNITY_TAG}/{filename}"
        http = client if client is not None else httpx.Client(follow_redirects=True, timeout=120.0)
        try:
            response = http.get(url)
            if response.status_code == 404:
                raise nv.NflverseNotFound(f"ffopportunity has no {filename} ({url}).")
            response.raise_for_status()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(response.content)
            tmp.replace(dest)  # atomic: a half-written parquet must never be readable
        finally:
            if client is None:
                http.close()

    df = pl.read_parquet(dest)
    if not correct_fumbles:
        return df
    fumbles = pl.col("rec_fumble_lost").fill_null(0.0) + pl.col("rush_fumble_lost").fill_null(0.0)
    opportunities = pl.col("rec_attempt").fill_null(0.0) + pl.col("rush_attempt").fill_null(0.0)
    return df.with_columns(
        fumbles_lost=fumbles,
        expected_fumbles_lost=opportunities * fumble_rate,
        total_fantasy_points_diff_adj=(
            pl.col("total_fantasy_points_diff") + 2.0 * opportunities * fumble_rate
        ),
    )
