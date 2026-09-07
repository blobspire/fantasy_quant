"""Opponent-adjusted defense-vs-position, and the schedule strength that falls out of it.

The number every free site publishes -- "fantasy points allowed to WRs" -- is a
measure of schedule, not of defense. Three biases stack on it:

1. **Opponent quality.** A defense that drew Detroit, Baltimore and Cincinnati
   allows more than one that drew Cleveland, Tennessee and the Giants, and the
   raw table cannot tell those apart from actual defensive quality.
2. **Game-script endogeneity.** Bad defenses trail, trailing opponents throw, so
   the pass/run split that inflates "WR points allowed" is *caused by* the very
   quality being measured. The effect is circular and the raw number double-counts
   it.
3. **Small samples.** Six games against six offenses is a handful of observations
   against an outcome whose per-player SD is ~7 points. Most of the week-6 spread
   in a raw table is noise, and treating it as signal is worse than ignoring it.

The fix is the standard two-way additive decomposition

    y_ij = mu + offense_i + defense_j

over player-games, where `i` is the *player* who produced the line and `j` the
defense he faced, fitted by alternating shrunk means (block coordinate descent on
a ridge-penalised least squares problem, so it provably converges):

    offense_i = sum_j w (y - mu - defense_j) / (W_i + k_off)
    defense_j = sum_i w (y - mu - offense_i) / (W_j + k_def)

`k_off` and `k_def` are pseudo-observation counts -- the empirical-Bayes prior
weight -- and are the only free parameters. They are not guessed here: every
default in `DEFAULT_SHRINKAGE` came out of forward-chaining cross-validation on
2022-2025 (`cross_validate`), fit on weeks 1..w-1 and scored on week w. Because
they are pseudo-counts, a defense with two games of data is pulled almost all the
way to zero adjustment, which is the right early-season behaviour: the model
degrades to "no opinion" rather than to a loud wrong one.

Recency weighting is a half-life in games on top of that, and CV likes 6-10 --
roughly a trailing 8-game window, which is also where football sense puts it.
Note that the half-life and `k_off` interact: at the wrong `k_off`, recency
weighting looks actively harmful, because shortening the effective sample makes
an over-shrunk player term worse. Tune them together or not at all.

**Sign and rank conventions.** `DefenseEffect.effect` is in points per
player-game relative to the positional mean, so **negative = suppresses the
position**. `rank` is 1 for the toughest defense (fewest points allowed) through
32 for the softest -- ESPN's own convention, which we verified rather than
assumed: our team-total naive measure correlates rho=+1.000 with ESPN's
`mPositionalRatings` average and reproduces its QB number to the decimal
(16.29 half-PPR against 16.29, league 161496047, 2025). That is the finding that
makes this module worth having -- ESPN publishes the naive measure, so the
adjustment is a real edge over what the league page shows. Against ESPN's own
table ours disagrees by a mean of 3.9-4.9 places out of 32, up to 15.

**Measured against the research note, 2025, PPR.** The note claimed the
adjustment moves defenses a mean of 3.6 ranks of 32 with individual moves to 11,
CIN going from naive #32 to adjusted #21. At `k_off=4`, `k_def=30`, uniform
weights -- the note's own constants -- WR reproduces all three: mean 3.69, max
11, and CIN naive #32 -> adjusted #21 exactly. Note that is a WR-specific
figure. Pooled across positions the same settings give a mean of 2.67, and the
cross-validated constants below move WR further, to a mean of 4.4.

**How much signal is actually here.** Not much, and this is the number to keep in
mind before acting on any DvP table, ours included. Split-half correlation of a
defense's effect (weeks 1-9 against weeks 10-18) pooled over 128 defense-seasons,
2022-2025, **at the shipped `DEFAULT_SHRINKAGE`**:

    QB  naive +0.166 -> adjusted +0.204      RB  naive +0.177 -> adjusted +0.187
    WR  naive -0.027 -> adjusted +0.163      TE  naive +0.059 -> adjusted +0.120

(Fisher SE about 0.089, and that SE is optimistic -- the 32 defenses within a
season are coupled by the fit and the same clubs recur across the four seasons,
so treat these as suggestive, not as four significant results.) The same table
with the half-life switched off -- `Shrinkage(k_off, k_def, None)`, which is *not*
what this module runs -- reads +0.250 / +0.152 / +0.166 / +0.122. Quote whichever
you like, but say which: the recency weight moves QB by five hundredths.

The adjustment helps at all four positions, and the WR line is the headline:
**raw WR points-allowed does not predict itself at all** -- r = -0.03 half-season
to half-season -- while the adjusted version reaches +0.16. That is what a
schedule artefact looks like when you measure it.

Stepping the WR figure up to a full season (Spearman-Brown, 2r/(1+r)) gives a
reliability of about 0.28, so roughly seven parts in ten of a full-season DvP
table are noise even after the adjustment. That is why the cross-validated
`k_def` is so large; heavy shrinkage is the correct response to a measure this
unreliable, not a lack of nerve. Year over year it is weaker still and the sign
is not even stable across configurations: 2024 -> 2025 adjusted reads QB -0.01,
RB +0.11, WR -0.23, TE +0.22 at the defaults, and QB +0.07, RB +0.28, WR -0.10,
TE +0.35 with uniform weights. Never carry a defense rating across a season
boundary.

**What this does not fix.** The decomposition removes bias 1 (opponent quality)
and bias 3 (small samples). It does *not* identify bias 2, game-script
endogeneity, because a defense's own offense is not in the model. Measured on
2025 against the defense's own team's total skill-position points per game, the
correlation runs QB +0.24, RB +0.01, WR +0.22, TE +0.13, and the adjustment
leaves it roughly where the naive number had it (+0.25 / -0.10 / +0.17 / +0.11).
It is small, but it is there, and closing it would need a plays-faced or
neutral-script control rather than a bigger shrinkage constant.

**Fit one season at a time.** Defenses are not the same unit across a coaching
change or a free-agency cycle, and the recency weight is expressed in weeks,
which do not order across seasons. `fit` refuses a multi-season frame.

**Caution on preseason playoff strength of schedule.** `playoff_schedule_strength`
is deliberately available, but a preseason reading of it is a weak signal and
should not move a draft board. Vegas totals already contain every input that
goes into it -- and more, since they price the roster as it will be in December.
This is an in-season tool: it is worth something once the defense effects are
estimated on real games, and even then as a tiebreaker between two otherwise
comparable players, not as a first-order term.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl

from ..core import QB, RB, TE, WR
from ..data import nflverse
from ..data.ids import team_from_pro_team_id

log = logging.getLogger(__name__)

#: The four positions carrying a fitted distribution; matches `core.SKILL_POSITIONS`.
POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

#: ESPN `defaultPositionId` -> our label, for the `mPositionalRatings` join.
POSITION_LABELS: Mapping[int, str] = MappingProxyType({QB: "QB", RB: "RB", WR: "WR", TE: "TE"})

#: NFL weeks that carry a standard 12-team fantasy playoff bracket.
PLAYOFF_WEEKS: tuple[int, ...] = (15, 16, 17)

#: Scoring variants nflverse ships directly. Half-PPR is the midpoint, which is
#: exact rather than approximate: the two columns differ only in the per-reception
#: term, so their mean is the 0.5/rec line two of the user's three leagues use.
PPR = "fantasy_points_ppr"
STANDARD = "fantasy_points"
HALF_PPR = "fantasy_points_half_ppr"


class DvpError(RuntimeError):
    """A DvP fit was asked for something the data cannot support."""


# --------------------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Shrinkage:
    """Empirical-Bayes prior weights, in pseudo-observations.

    `k_off` is compared against a player's own weighted game count and `k_def`
    against a defense's, so `defense_j` with `k_def=90` and 78 weighted
    observations is pulled 54% of the way to zero. `halflife` is in games; None
    means uniform weighting.
    """

    k_off: float
    k_def: float
    halflife: float | None = None

    def __post_init__(self) -> None:
        if self.k_off < 0 or self.k_def < 0:
            raise ValueError(f"shrinkage constants must be non-negative, got {self}")
        if self.halflife is not None and self.halflife <= 0:
            raise ValueError(f"halflife must be positive or None, got {self.halflife}")


#: The argmin of `cross_validate` over 2022-2025 PPR player-weeks, forward-chaining
#: from week 6 -- k_off in {0.1..2}, k_def in {5..180, inf}, half-life in {4..16, none}.
#: Every position's optimum is *interior* on all three axes, which is the sanity
#: check that matters: a boundary optimum would mean the grid, not the data, chose
#: the constant. Held-out RMSE at the optimum against the same fit with the defense
#: term switched off (k_def -> inf):
#:
#:     QB  7.8068 vs 7.8414   RB  6.1499 vs 6.1613
#:     WR  6.1213 vs 6.1268   TE  5.0901 vs 5.0926
#:
#: Those gains are small in absolute terms -- player-week noise dominates
#: everything -- but they are the honest size of the effect. Note that "positive
#: at all four positions" is *not* by itself evidence, because `CvResult.gain` is
#: non-negative by construction (see its docstring). The evidence is a placebo:
#: randomise which defense each team-game faced and re-run the identical fit, and
#: the defense term stops paying -- placebo gain -0.0195 +/- 0.0111 (QB),
#: -0.0068 +/- 0.0028 (RB), -0.0035 +/- 0.0014 (WR), +0.0009 +/- 0.0022 (TE) over
#: six draws. The real gain sits 4.9 / 6.5 / 6.4 placebo SDs above that at
#: QB/RB/WR and **0.7 at TE**. A week-clustered bootstrap agrees: P(gain > 0) =
#: 0.97 / 0.99 / 0.94 / 0.83. Treat the TE defense term as unproven.
#:
#: The pattern across positions is the interesting part. QB wants a sixth of WR's
#: defense shrinkage because a QB line is one observation per team-game that
#: carries the whole passing offense, while a WR line is one of five noisy slices
#: of it. The half-life lands at 6-10 games everywhere, i.e. a trailing ~8-game
#: window, which is where football sense also puts it.
DEFAULT_SHRINKAGE: Mapping[str, Shrinkage] = MappingProxyType(
    {
        "QB": Shrinkage(k_off=0.5, k_def=15.0, halflife=6.0),
        "RB": Shrinkage(k_off=0.5, k_def=45.0, halflife=6.0),
        "WR": Shrinkage(k_off=0.75, k_def=90.0, halflife=10.0),
        "TE": Shrinkage(k_off=0.75, k_def=90.0, halflife=10.0),
    }
)


# --------------------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwoWaySolution:
    """Raw output of the alternating solver, in index space rather than labels."""

    mu: float
    offense: np.ndarray
    defense: np.ndarray
    weighted_offense_n: np.ndarray
    weighted_defense_n: np.ndarray
    iterations: int
    converged: bool
    #: Penalised objective after each sweep. Monotone non-increasing by
    #: construction; exposed because a violation means the solver is broken.
    objective: tuple[float, ...]


def solve_two_way(
    offense_idx: np.ndarray,
    defense_idx: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_offense: int,
    n_defense: int,
    shrinkage: Shrinkage,
    *,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> TwoWaySolution:
    """Fit `y = mu + offense[i] + defense[j]` by alternating shrunk means.

    Each sweep is the exact minimiser of the penalised objective

        sum_n w_n (y_n - mu - o_i - d_j)^2 + k_off*sum o^2 + k_def*sum d^2

    over one block at a time, so the objective decreases monotonically and the
    iteration converges to the unique optimum of a strictly convex problem. The
    intercept is left unpenalised, which is what identifies the split: without
    it, a constant could slide between `mu`, `offense` and `defense` for free.
    `test_solver_matches_a_direct_ridge_solve` pins this against the closed form.

    Two failure modes to know about, neither of which a real season hits:

    * **Disconnected play.** The offense/defense bipartite graph has to be
      connected for the two effects to be comparable. If two defenses share no
      opponent, the penalty still returns a unique answer, but the difference
      between them is an artefact of the shrinkage rather than a measurement.
    * **Slow convergence on thin designs.** Gauss-Seidel converges at a rate set
      by the conditioning, and a design with very few defenses or one game each
      can take thousands of sweeps. A real 32-defense season converges in
      146-182 sweeps at `tol=1e-9` (measured over 2022-2025 x QB/RB/WR/TE);
      `max_iter` is set well above that and non-convergence is logged rather
      than swallowed.
    """
    offense_idx = np.asarray(offense_idx, dtype=np.intp)
    defense_idx = np.asarray(defense_idx, dtype=np.intp)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not (offense_idx.shape == defense_idx.shape == y.shape == weights.shape):
        raise ValueError("offense_idx, defense_idx, y and weights must be the same length")
    if y.size == 0:
        raise DvpError("no observations to fit")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    total_w = float(weights.sum())
    if total_w <= 0:
        raise DvpError("all observation weights are zero")

    w_off = np.bincount(offense_idx, weights, n_offense)
    w_def = np.bincount(defense_idx, weights, n_defense)
    mu = float((weights * y).sum() / total_w)
    offense = np.zeros(n_offense)
    defense = np.zeros(n_defense)
    objective: list[float] = []

    converged = False
    step = 0
    while step < max_iter:
        step += 1
        resid = y - mu - defense[defense_idx]
        new_off = np.bincount(offense_idx, weights * resid, n_offense) / (w_off + shrinkage.k_off)
        delta = float(np.max(np.abs(new_off - offense))) if n_offense else 0.0
        offense = new_off

        resid = y - mu - offense[offense_idx]
        new_def = np.bincount(defense_idx, weights * resid, n_defense) / (w_def + shrinkage.k_def)
        delta = max(delta, float(np.max(np.abs(new_def - defense))) if n_defense else 0.0)
        defense = new_def

        mu = float((weights * (y - offense[offense_idx] - defense[defense_idx])).sum() / total_w)

        resid = y - mu - offense[offense_idx] - defense[defense_idx]
        objective.append(
            float((weights * resid * resid).sum())
            + _penalty(shrinkage.k_off, offense)
            + _penalty(shrinkage.k_def, defense)
        )
        if delta < tol:
            converged = True
            break

    if not converged:
        log.warning("two-way solver hit %d iterations without converging to %g", max_iter, tol)
    return TwoWaySolution(
        mu=mu,
        offense=offense,
        defense=defense,
        weighted_offense_n=w_off,
        weighted_defense_n=w_def,
        iterations=step,
        converged=converged,
        objective=tuple(objective),
    )


def _penalty(k: float, effects: np.ndarray) -> float:
    """Ridge penalty term. `k = inf` pins the block to exactly zero, and `inf * 0`
    is NaN rather than 0 in IEEE arithmetic, so that case is handled explicitly --
    otherwise the no-defense CV baseline reports a NaN objective."""
    if math.isinf(k):
        return 0.0
    return k * float((effects * effects).sum())


def recency_weights(weeks: np.ndarray, reference_week: int, halflife: float | None) -> np.ndarray:
    """Exponential decay in games. `halflife=None` gives uniform weights.

    A half-life of 6 puts ~11% weight on a game 18 weeks back and ~50% on one six
    weeks back; the effective sample is roughly `halflife / ln 2` recent games,
    which is where the "trailing eight" rule of thumb comes from.
    """
    weeks = np.asarray(weeks, dtype=float)
    if halflife is None:
        return np.ones_like(weeks)
    return np.power(0.5, np.maximum(reference_week - weeks, 0.0) / float(halflife))


# --------------------------------------------------------------------------------------
# Fitted model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DefenseEffect:
    """One defense's effect on one position, adjusted and naive side by side.

    `effect` and `naive` are both points per *player*-game against the positional
    mean, so they are directly comparable and their rank difference is the whole
    point of the module. Negative suppresses the position.
    """

    team: str
    position: str
    effect: float
    naive: float
    #: 1 = toughest (fewest points allowed), 32 = softest. ESPN's convention.
    rank: int
    naive_rank: int
    #: Distinct games in the fitting window in which this defense faced the position.
    games: int
    #: Player-game rows, and their sum of recency weights.
    observations: int
    weighted_observations: float

    @property
    def rank_move(self) -> int:
        """Places the adjustment moved this defense. Positive = looks tougher than naive."""
        return self.naive_rank - self.rank


@dataclass(frozen=True, slots=True)
class PositionDvp:
    """The fitted decomposition for one position in one season."""

    position: str
    season: int
    through_week: int
    points_column: str
    mu: float
    shrinkage: Shrinkage
    defense: Mapping[str, DefenseEffect]
    offense: Mapping[str, float]
    iterations: int
    converged: bool
    observations: int
    #: Mean player-game rows per team-game, i.e. how many of this position a
    #: defense faces at once. Converts a per-player effect to a team-total one.
    players_per_game: float

    def effect(self, team: str) -> float:
        """Points per player-game this defense adds (positive) or removes (negative).

        An unseen team returns 0.0 rather than raising: "no evidence" and "average"
        are the same prediction here, and a bye-week or expansion gap must not
        take down a lineup call.
        """
        entry = self.defense.get(team)
        return entry.effect if entry is not None else 0.0

    def team_effect(self, team: str) -> float:
        """The same effect expressed per team-game, which is what a DvP table shows."""
        return self.effect(team) * self.players_per_game

    def rank(self, team: str) -> int | None:
        entry = self.defense.get(team)
        return entry.rank if entry is not None else None

    def adjust(self, projection: float, opponent: str) -> float:
        """Shift a points projection by the opponent's effect, floored at zero.

        Additive, because that is the model that was fitted. The intuitive
        alternative -- scaling the projection by `1 + effect/mu`, so a stud loses
        more absolute points to a tough defense than a scrub does -- was tested
        head to head on the same forward-chaining folds and is *indistinguishable*:
        held-out RMSE moves by less than 0.003 and the sign of the difference
        varies by position (QB 7.8068 additive vs 7.8076 scaled, RB 6.1499 vs
        6.1476, WR 6.1213 vs 6.1193, TE 5.0901 vs 5.0901). There is no evidence
        for either form over the other, so this uses the one the estimator
        actually optimises.

        **Do not apply this to a projection that already prices the opponent.**
        Every vendor weekly projection -- ESPN's, Sleeper's, a market-derived one
        -- is conditioned on the matchup, so adding an effect on top double-counts
        it. This is for a projection built from a player's own form alone, or as
        the DvP term of an ensemble that deliberately excludes matchup-aware
        sources. The effect is small enough (+/-0.6 points per WR player-game in
        2025) that double-counting will not look wrong, which is exactly why it
        needs saying.
        """
        return max(0.0, projection + self.effect(opponent))

    def ranked(self) -> tuple[DefenseEffect, ...]:
        """Defenses toughest-first."""
        return tuple(sorted(self.defense.values(), key=lambda d: d.rank))

    def spread(self) -> float:
        """SD of the adjusted effects across defenses -- how much matchup is worth."""
        if not self.defense:
            return 0.0
        return float(np.std([d.effect for d in self.defense.values()]))


@dataclass(frozen=True, slots=True)
class DvpModel:
    """Every position's fit for one season, keyed by our position labels."""

    season: int
    through_week: int
    points_column: str
    positions: Mapping[str, PositionDvp]

    def __getitem__(self, position: str) -> PositionDvp:
        try:
            return self.positions[position]
        except KeyError:
            raise KeyError(f"no DvP fit for position {position!r}") from None

    def get(self, position: str) -> PositionDvp | None:
        return self.positions.get(position)

    def effect(self, position: str, team: str) -> float:
        fit = self.positions.get(position)
        return fit.effect(team) if fit is not None else 0.0

    def adjust(self, projection: float, position: str, opponent: str) -> float:
        fit = self.positions.get(position)
        if fit is None:
            return max(0.0, projection)
        return fit.adjust(projection, opponent)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def player_weeks(
    seasons: int | Iterable[int],
    *,
    positions: Sequence[str] = POSITIONS,
    season_type: str | None = "REG",
    cache: nflverse.NflverseCache | None = None,
    allow_missing: bool = False,
) -> pl.DataFrame:
    """The DvP input frame: one row per player-game, with half-PPR added.

    nflverse ships standard and full-PPR points; half-PPR is their midpoint
    exactly, since the columns differ only in the per-reception term. Two of the
    user's three leagues score 0.5/rec, so it is worth carrying rather than
    re-deriving at every call site.
    """
    wanted = list(positions)
    frame = nflverse.player_week_stats(
        seasons, season_type=season_type, cache=cache, allow_missing=allow_missing
    )
    return (
        frame.filter(pl.col("position").is_in(wanted))
        .with_columns(
            ((pl.col(STANDARD) + pl.col(PPR)) / 2.0).alias(HALF_PPR),
        )
        .select(
            "player_id",
            "player_display_name",
            "position",
            "season",
            "week",
            "team",
            "opponent_team",
            STANDARD,
            PPR,
            HALF_PPR,
        )
        .drop_nulls(["player_id", "opponent_team", "week"])
    )


def _prepare(
    frame: pl.DataFrame, position: str, points_column: str, through_week: int | None
) -> tuple[pl.DataFrame, int, int]:
    data = frame.filter(pl.col("position") == position)
    if through_week is not None:
        data = data.filter(pl.col("week") <= through_week)
    data = data.drop_nulls([points_column])
    if data.is_empty():
        raise DvpError(f"no {position} rows to fit (points column {points_column!r})")
    seasons = data["season"].unique().to_list()
    if len(seasons) != 1:
        raise DvpError(
            f"DvP is a within-season measure; got seasons {sorted(seasons)}. Defenses are "
            "not the same unit across a coaching change, and the recency half-life is "
            "expressed in weeks, which do not order across seasons."
        )
    return data, int(seasons[0]), int(data["week"].max())


def _naive_effects(data: pl.DataFrame, points_column: str) -> dict[str, float]:
    """Points allowed per player-game against the positional mean. What sites publish."""
    grand = float(data[points_column].mean())
    grouped = data.group_by("opponent_team").agg(pl.col(points_column).mean().alias("m"))
    return {row["opponent_team"]: float(row["m"]) - grand for row in grouped.iter_rows(named=True)}


def naive_team_totals(data: pl.DataFrame, points_column: str) -> dict[str, float]:
    """Mean points allowed to the whole position per team-game -- ESPN's measure.

    Kept separate from the per-player naive because it is the one that reproduces
    `mPositionalRatings` (rho=+1.000 on QB, and to the decimal on the average).

    The game key is `(season, week)` and not `week` alone. Unlike a mean, a
    per-game *sum* does not survive pooling: on a two-season frame a bare `week`
    key merges 2024 week 5 with 2025 week 5 into one "game" and returns roughly
    double the right number, which reads as a plausible team total rather than as
    an error. `season` is used when the column is present and skipped when it is
    not, so a single-season frame without one still works.
    """
    keys = ["opponent_team"] + (["season"] if "season" in data.columns else []) + ["week"]
    per_game = data.group_by(keys).agg(pl.col(points_column).sum().alias("t"))
    grouped = per_game.group_by("opponent_team").agg(pl.col("t").mean().alias("m"))
    return {row["opponent_team"]: float(row["m"]) for row in grouped.iter_rows(named=True)}


def _rank(values: Mapping[str, float]) -> dict[str, int]:
    """1 = lowest value = toughest defense. Ties broken by team name, for determinism."""
    order = sorted(values, key=lambda t: (values[t], t))
    return {team: i + 1 for i, team in enumerate(order)}


def fit_position(
    frame: pl.DataFrame,
    position: str,
    *,
    points_column: str = PPR,
    shrinkage: Shrinkage | None = None,
    through_week: int | None = None,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> PositionDvp:
    """Fit one position's two-way decomposition on a single season of player-games."""
    data, season, last_week = _prepare(frame, position, points_column, through_week)
    shrink = shrinkage or DEFAULT_SHRINKAGE.get(position, Shrinkage(1.0, 60.0, 8.0))

    players = data["player_id"].unique().sort().to_list()
    teams = sorted(set(data["opponent_team"].to_list()))
    player_ix = {p: i for i, p in enumerate(players)}
    team_ix = {t: i for i, t in enumerate(teams)}

    offense_idx = np.fromiter((player_ix[p] for p in data["player_id"]), np.intp, data.height)
    defense_idx = np.fromiter((team_ix[t] for t in data["opponent_team"]), np.intp, data.height)
    y = data[points_column].to_numpy().astype(float)
    weeks = data["week"].to_numpy().astype(float)
    # Anchor on the week we are predicting -- one past the window -- rather than on
    # the last week observed. That costs the newest game a factor of 0.5**(1/halflife)
    # (0.89 at halflife 6), which matters because the weights are also the
    # pseudo-counts `k_off`/`k_def` are compared against. `cross_validate` anchors on
    # its test week the same way, so `DEFAULT_SHRINKAGE` transfers here unchanged;
    # anchoring on `last_week` instead would silently make the tuned constants shrink
    # ~11% harder than they did in the CV that chose them. Pinned by
    # `test_recency_anchor_matches_cross_validation`.
    weights = recency_weights(weeks, last_week + 1, shrink.halflife)

    solution = solve_two_way(
        offense_idx,
        defense_idx,
        y,
        weights,
        len(players),
        len(teams),
        shrink,
        max_iter=max_iter,
        tol=tol,
    )

    adjusted = {t: float(solution.defense[i]) for t, i in team_ix.items()}
    naive = _naive_effects(data, points_column)
    adj_rank = _rank(adjusted)
    naive_rank = _rank(naive)

    counts = data.group_by("opponent_team").agg(
        pl.len().alias("rows"), pl.col("week").n_unique().alias("games")
    )
    rows_by_team = {
        r["opponent_team"]: (int(r["rows"]), int(r["games"])) for r in counts.iter_rows(named=True)
    }
    total_games = sum(g for _, g in rows_by_team.values())

    defense = {
        team: DefenseEffect(
            team=team,
            position=position,
            effect=adjusted[team],
            naive=naive[team],
            rank=adj_rank[team],
            naive_rank=naive_rank[team],
            games=rows_by_team[team][1],
            observations=rows_by_team[team][0],
            weighted_observations=float(solution.weighted_defense_n[team_ix[team]]),
        )
        for team in teams
    }

    return PositionDvp(
        position=position,
        season=season,
        through_week=last_week,
        points_column=points_column,
        mu=solution.mu,
        shrinkage=shrink,
        defense=MappingProxyType(defense),
        offense=MappingProxyType({p: float(solution.offense[i]) for p, i in player_ix.items()}),
        iterations=solution.iterations,
        converged=solution.converged,
        observations=data.height,
        players_per_game=data.height / total_games if total_games else 0.0,
    )


def fit(
    frame: pl.DataFrame,
    *,
    positions: Sequence[str] = POSITIONS,
    points_column: str = PPR,
    shrinkage: Mapping[str, Shrinkage] | None = None,
    through_week: int | None = None,
) -> DvpModel:
    """Fit every position. `shrinkage` overrides `DEFAULT_SHRINKAGE` per position."""
    overrides = dict(shrinkage or {})
    fits: dict[str, PositionDvp] = {}
    for position in positions:
        fits[position] = fit_position(
            frame,
            position,
            points_column=points_column,
            shrinkage=overrides.get(position),
            through_week=through_week,
        )
    if not fits:
        raise DvpError("fit() needs at least one position")
    any_fit = next(iter(fits.values()))
    return DvpModel(
        season=any_fit.season,
        through_week=max(f.through_week for f in fits.values()),
        points_column=points_column,
        positions=MappingProxyType(fits),
    )


# --------------------------------------------------------------------------------------
# Cross-validation -- how the shrinkage constants were chosen
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CvPoint:
    shrinkage: Shrinkage
    rmse: float
    mae: float
    n: int


@dataclass(frozen=True, slots=True)
class CvResult:
    """A forward-chaining CV sweep over the shrinkage grid for one position.

    `baseline_rmse` is the same sweep with the defense term switched off
    (`k_def = inf`) at the winning `k_off`/half-life, which isolates what the
    *defense* adjustment buys from what merely knowing the player's own average
    buys. The gain is small in RMSE terms -- player-week noise dominates
    everything.

    **`gain` cannot report harm, so it is not evidence.** `best` is the argmin
    over a curve that contains the baseline point, so `baseline_rmse >=
    best_rmse` identically and `gain >= 0` whatever the data says. On a synthetic
    season with the defense effects set to exactly zero, `k_def = 30` is a real
    0.088 RMSE *worse* than no defense term at all -- and `gain` still reports
    0.0000, because the search correctly falls back on the baseline. Read
    `gain > 0` as "a finite `k_def` won the search", nothing more. To ask whether
    the defense term is worth having, compare `curve` entries at a *fixed*
    `k_off`/half-life, or run the placebo in the `DEFAULT_SHRINKAGE` note.
    """

    position: str
    points_column: str
    seasons: tuple[int, ...]
    best: Shrinkage
    best_rmse: float
    baseline_rmse: float
    curve: tuple[CvPoint, ...]

    @property
    def gain(self) -> float:
        """RMSE the winning point removes against the no-defense point beside it.

        Bounded below by zero by construction -- see the class docstring. It says
        the search preferred a finite `k_def`; it does not say the defense term
        generalises.
        """
        return self.baseline_rmse - self.best_rmse

    def profile(self, over: str) -> tuple[CvPoint, ...]:
        """The curve along one axis with the other two held at the optimum."""
        if over not in {"k_off", "k_def", "halflife"}:
            raise ValueError(f"no CV axis {over!r}")
        fixed = [a for a in ("k_off", "k_def", "halflife") if a != over]
        held = [getattr(self.best, a) for a in fixed]
        points = [
            p
            for p in self.curve
            if all(getattr(p.shrinkage, a) == v for a, v in zip(fixed, held, strict=True))
        ]

        # `halflife=None` means "no decay", which sorts last. Written as an
        # explicit None test rather than `or math.inf`, which would also swallow
        # a legitimate k_off of 0.0.
        def sort_key(point: CvPoint) -> float:
            value = getattr(point.shrinkage, over)
            return math.inf if value is None else float(value)

        return tuple(sorted(points, key=sort_key))


def cross_validate(
    frame: pl.DataFrame,
    position: str,
    *,
    points_column: str = PPR,
    k_off_grid: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    k_def_grid: Sequence[float] = (10.0, 20.0, 30.0, 60.0, 90.0, 120.0, math.inf),
    halflife_grid: Sequence[float | None] = (6.0, 8.0, 12.0, 16.0, None),
    first_test_week: int = 6,
    max_iter: int = 2000,
    tol: float = 1e-7,
) -> CvResult:
    """Forward-chaining CV: fit on weeks 1..w-1, score week w, pooled over seasons.

    Forward chaining rather than random k-fold because that is the shape of the
    real decision -- on Tuesday of week 9 you have weeks 1-8 and nothing else.
    Random folds would leak later weeks into an earlier fit and flatter the model.

    An `inf` in `k_def_grid` is the no-defense-term baseline and is expected to be
    there; `CvResult.baseline_rmse` reads it back out.
    """
    data = frame.filter(pl.col("position") == position).drop_nulls([points_column])
    if data.is_empty():
        raise DvpError(f"no {position} rows to cross-validate")
    seasons = sorted(int(s) for s in data["season"].unique().to_list())

    grid = [
        Shrinkage(k_off=ko, k_def=kd, halflife=hl)
        for hl in halflife_grid
        for ko in k_off_grid
        for kd in k_def_grid
    ]
    errors: dict[int, list[np.ndarray]] = {i: [] for i in range(len(grid))}

    for season in seasons:
        season_rows = data.filter(pl.col("season") == season)
        last_week = int(season_rows["week"].max())
        for week in range(first_test_week, last_week + 1):
            train = season_rows.filter(pl.col("week") < week)
            test = season_rows.filter(pl.col("week") == week)
            if train.height < 50 or test.is_empty():
                continue

            players = train["player_id"].unique().sort().to_list()
            teams = sorted(set(train["opponent_team"].to_list()))
            player_ix = {p: i for i, p in enumerate(players)}
            team_ix = {t: i for i, t in enumerate(teams)}
            offense_idx = np.fromiter(
                (player_ix[p] for p in train["player_id"]), np.intp, train.height
            )
            defense_idx = np.fromiter(
                (team_ix[t] for t in train["opponent_team"]), np.intp, train.height
            )
            y_train = train[points_column].to_numpy().astype(float)
            train_weeks = train["week"].to_numpy().astype(float)
            # Unseen player or defense -> index -1 -> zero effect, i.e. the mean.
            test_off = np.fromiter(
                (player_ix.get(p, -1) for p in test["player_id"]), np.intp, test.height
            )
            test_def = np.fromiter(
                (team_ix.get(t, -1) for t in test["opponent_team"]), np.intp, test.height
            )
            y_test = test[points_column].to_numpy().astype(float)

            for i, shrink in enumerate(grid):
                weights = recency_weights(train_weeks, week, shrink.halflife)
                solution = solve_two_way(
                    offense_idx,
                    defense_idx,
                    y_train,
                    weights,
                    len(players),
                    len(teams),
                    shrink,
                    max_iter=max_iter,
                    tol=tol,
                )
                off = np.where(test_off >= 0, solution.offense[np.maximum(test_off, 0)], 0.0)
                dfn = np.where(test_def >= 0, solution.defense[np.maximum(test_def, 0)], 0.0)
                errors[i].append(y_test - (solution.mu + off + dfn))

    curve: list[CvPoint] = []
    for i, shrink in enumerate(grid):
        if not errors[i]:
            continue
        err = np.concatenate(errors[i])
        curve.append(
            CvPoint(
                shrinkage=shrink,
                rmse=float(np.sqrt(np.mean(err * err))),
                mae=float(np.mean(np.abs(err))),
                n=int(err.size),
            )
        )
    if not curve:
        raise DvpError(f"no CV folds produced for {position}; season too short?")

    best = min(curve, key=lambda p: p.rmse)
    baseline = [
        p
        for p in curve
        if math.isinf(p.shrinkage.k_def)
        and p.shrinkage.k_off == best.shrinkage.k_off
        and p.shrinkage.halflife == best.shrinkage.halflife
    ]
    return CvResult(
        position=position,
        points_column=points_column,
        seasons=tuple(seasons),
        best=best.shrinkage,
        best_rmse=best.rmse,
        baseline_rmse=baseline[0].rmse if baseline else float("nan"),
        curve=tuple(curve),
    )


# --------------------------------------------------------------------------------------
# Naive comparison
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankComparison:
    """How far the adjustment moves a naive DvP table."""

    position: str
    season: int
    teams: int
    mean_abs_move: float
    max_abs_move: int
    spearman: float
    #: team -> (naive rank, adjusted rank), biggest mover first.
    moves: tuple[tuple[str, int, int], ...]

    def top_movers(self, n: int = 5) -> tuple[tuple[str, int, int], ...]:
        return self.moves[:n]


def compare_naive(fit: PositionDvp) -> RankComparison:
    """Adjusted ranks against naive ranks for one position."""
    entries = list(fit.defense.values())
    if not entries:
        raise DvpError(f"no defenses fitted for {fit.position}")
    moves = sorted(
        ((e.team, e.naive_rank, e.rank) for e in entries),
        key=lambda m: (-abs(m[1] - m[2]), m[0]),
    )
    deltas = [abs(e.naive_rank - e.rank) for e in entries]
    return RankComparison(
        position=fit.position,
        season=fit.season,
        teams=len(entries),
        mean_abs_move=float(np.mean(deltas)),
        max_abs_move=int(max(deltas)),
        spearman=_spearman(
            [e.naive for e in entries],
            [e.effect for e in entries],
        ),
        moves=tuple(moves),
    )


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation without dragging scipy into a hot path."""
    if len(a) != len(b):
        raise ValueError("spearman needs equal-length sequences")
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = float(np.mean(ra)), float(np.mean(rb))
    da, db = ra - ma, rb - mb
    denom = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    return float((da * db).sum() / denom) if denom else float("nan")


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Average ranks, so ties do not distort the correlation."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranked = np.empty(arr.size, dtype=float)
    ranked[order] = np.arange(1, arr.size + 1, dtype=float)
    # Average within tie groups.
    sorted_vals = arr[order]
    start = 0
    for i in range(1, arr.size + 1):
        if i == arr.size or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranked[order[start:i]] = ranked[order[start:i]].mean()
            start = i
    return ranked


# --------------------------------------------------------------------------------------
# ESPN's own positional ratings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EspnComparison:
    """What ESPN's `mPositionalRatings` actually is, measured rather than assumed.

    `rho_team_total` is the tell. ESPN's published "average" is the mean fantasy
    points the position scored against that defense per team-game, with no
    opponent adjustment at all -- exactly the measure this module exists to
    replace. A rho near +1.0 against our naive team totals, together with matching
    absolute levels, is proof rather than suspicion.
    """

    position: str
    season: int
    teams: int
    espn_mean: float
    our_team_total_mean: float
    rho_team_total: float
    rho_naive_player: float
    rho_adjusted: float
    #: Mean and max places between ESPN's ranking and ours. This is the edge.
    mean_abs_move: float
    max_abs_move: int
    moves: tuple[tuple[str, int, int], ...]


def compare_with_espn(
    fit: PositionDvp,
    ratings: Mapping[int, Any],
    frame: pl.DataFrame,
) -> EspnComparison:
    """Rank-correlate ESPN's positional ratings against our naive and adjusted tables.

    `ratings` is what `espn.league.League.positional_ratings()` returns, keyed by
    `defaultPositionId`. `frame` must be the same player-week frame the fit came
    from, scored the same way ESPN scores this league -- comparing a half-PPR
    league's ratings against full-PPR points would compare two different things.
    """
    position_id = next(
        (pid for pid, label in POSITION_LABELS.items() if label == fit.position), None
    )
    if position_id is None or position_id not in ratings:
        raise DvpError(f"ESPN ratings carry no entry for {fit.position}")

    espn: dict[str, tuple[float, int]] = {}
    for entry in ratings[position_id].by_opponent.values():
        team = team_from_pro_team_id(int(entry.pro_team_id))
        if team is not None:
            espn[team] = (float(entry.average), int(entry.rank))
    if not espn:
        raise DvpError(
            f"ESPN returned no per-opponent ratings for {fit.position}. The table is empty "
            "before the season's first games; ask for a finished season."
        )

    data, _, _ = _prepare(frame, fit.position, fit.points_column, fit.through_week)
    totals = naive_team_totals(data, fit.points_column)
    naive = _naive_effects(data, fit.points_column)

    teams = sorted(set(espn) & set(fit.defense) & set(totals))
    if len(teams) < 2:
        raise DvpError("too few teams overlap between ESPN's ratings and our fit")

    espn_values = [espn[t][0] for t in teams]
    # ESPN ranks ascending by points allowed, same direction as ours; verified
    # rho(rank, average) = +1.000. Re-rank locally so a partial overlap still
    # compares like with like.
    espn_rank = _rank({t: espn[t][0] for t in teams})
    our_rank = _rank({t: fit.defense[t].effect for t in teams})
    moves = sorted(
        ((t, espn_rank[t], our_rank[t]) for t in teams),
        key=lambda m: (-abs(m[1] - m[2]), m[0]),
    )
    deltas = [abs(espn_rank[t] - our_rank[t]) for t in teams]

    return EspnComparison(
        position=fit.position,
        season=fit.season,
        teams=len(teams),
        espn_mean=float(np.mean(espn_values)),
        our_team_total_mean=float(np.mean([totals[t] for t in teams])),
        rho_team_total=_spearman(espn_values, [totals[t] for t in teams]),
        rho_naive_player=_spearman(espn_values, [naive[t] for t in teams]),
        rho_adjusted=_spearman(espn_values, [fit.defense[t].effect for t in teams]),
        mean_abs_move=float(np.mean(deltas)),
        max_abs_move=int(max(deltas)),
        moves=tuple(moves),
    )


# --------------------------------------------------------------------------------------
# Strength of schedule
# --------------------------------------------------------------------------------------


def team_schedule(
    season: int,
    *,
    cache: nflverse.NflverseCache | None = None,
) -> pl.DataFrame:
    """One row per played regular-season team-game: season, week, team, opponent, home.

    A thin projection of `nflverse.team_weeks`, which is the adapter that already
    owns the game-oriented -> team-oriented unpivot along with byes, market
    columns and rest days. This module needs five of those columns and no bye
    rows, so this narrows rather than re-derives; `schedule_strength` accepts the
    adapter's own frame equally well. A future season is fully populated well
    before kickoff -- only the Vegas columns are null -- so this works preseason.
    """
    weeks = nflverse.team_weeks(nflverse.schedules(season, cache=cache), season, include_byes=False)
    return weeks.select(
        pl.lit(season, dtype=pl.Int64).alias("season"),
        pl.col("week"),
        pl.col("team"),
        pl.col("opponent"),
        pl.col("is_home").alias("home"),
    ).sort(["team", "week"])


@dataclass(frozen=True, slots=True)
class ScheduleStrength:
    """Aggregated opponent difficulty for one team at one position over a week window.

    `per_game` is points per player-game: positive means the remaining opponents
    allow more than average, i.e. an easy schedule. `total` sums over the games
    actually played, so a team on bye in the window has one fewer term in `total`
    (pulling it toward zero from whichever side its schedule sat on) while
    `per_game` stays comparable -- which is the distinction that matters when the
    window is three playoff weeks and a bye would be catastrophic.

    `rank` is within one position. Comparing a QB's `per_game` against a WR's is
    legitimate -- both are points per game, and QB matchups genuinely swing
    several times more points than WR matchups (2025: 4.2 points best-to-worst at
    QB against 1.2 at WR) -- but comparing their *ranks* is not.
    """

    team: str
    position: str
    weeks: tuple[int, ...]
    games: int
    per_game: float
    total: float
    #: 1 = hardest schedule, matching the toughest-first defense ranking.
    rank: int
    opponents: tuple[tuple[int, str], ...]
    #: Weeks in the window with no game -- a bye, or an unscheduled future week.
    idle_weeks: tuple[int, ...]


def schedule_strength(
    fit: PositionDvp,
    schedule: pl.DataFrame,
    *,
    weeks: Iterable[int],
) -> dict[str, ScheduleStrength]:
    """Aggregate adjusted defense effects over each team's opponents in `weeks`.

    Takes any frame with `week`, `team` and `opponent` columns -- `team_schedule`
    or `nflverse.team_weeks`/`remaining_opponents` directly.

    **Bye rows are dropped, not scored.** The adapter's team-week grid carries a
    null-opponent row per bye, and a null opponent would otherwise fall through
    `PositionDvp.effect`'s unseen-team path to a 0.0 contribution, counting as a
    *played* game: 2025 KC then reads 18 games instead of 17, `per_game` is
    diluted toward zero, and `idle_weeks` -- the field whose whole job is to
    surface the bye -- comes back empty. Nothing about that output looks wrong.

    Unseen but *named* opponents still contribute 0.0 (see `PositionDvp.effect`),
    so a window reaching into weeks against a team we have no data for degrades
    toward "average schedule" rather than silently dropping games.
    """
    wanted = tuple(sorted({int(w) for w in weeks}))
    if not wanted:
        raise ValueError("schedule_strength needs at least one week")
    if missing := {"week", "team", "opponent"} - set(schedule.columns):
        raise ValueError(f"schedule frame is missing {sorted(missing)}")
    window = schedule.filter(pl.col("week").is_in(list(wanted)) & pl.col("opponent").is_not_null())
    if window.is_empty():
        raise DvpError(f"the schedule carries no games in weeks {wanted}")

    per_team: dict[str, list[tuple[int, str]]] = {}
    for row in window.iter_rows(named=True):
        per_team.setdefault(row["team"], []).append((int(row["week"]), row["opponent"]))

    raw = {
        team: sum(fit.effect(opponent) for _, opponent in games) for team, games in per_team.items()
    }
    means = {team: raw[team] / len(per_team[team]) for team in per_team}
    ranks = _rank(means)

    return {
        team: ScheduleStrength(
            team=team,
            position=fit.position,
            weeks=wanted,
            games=len(games),
            per_game=means[team],
            total=raw[team],
            rank=ranks[team],
            opponents=tuple(sorted(games)),
            idle_weeks=tuple(w for w in wanted if w not in {g[0] for g in games}),
        )
        for team, games in sorted(per_team.items())
    }


def rest_of_season_strength(
    fit: PositionDvp,
    schedule: pl.DataFrame,
    *,
    from_week: int,
    through_week: int = 17,
) -> dict[str, ScheduleStrength]:
    """Schedule strength over the remaining fantasy regular season and playoffs.

    Defaults to week 17 because week 18 is not a fantasy week in any of the three
    leagues and its starters are unpredictable anyway.
    """
    if from_week > through_week:
        raise ValueError(f"from_week {from_week} is after through_week {through_week}")
    return schedule_strength(fit, schedule, weeks=range(from_week, through_week + 1))


def playoff_schedule_strength(
    fit: PositionDvp,
    schedule: pl.DataFrame,
    *,
    weeks: Sequence[int] = PLAYOFF_WEEKS,
) -> dict[str, ScheduleStrength]:
    """Weeks 15-17 only.

    Read the module docstring's caution before acting on this preseason: with no
    games played the defense effects are all shrunk to ~0 and the answer is noise
    dressed as a ranking. In-season it is a legitimate tiebreaker, and nothing
    more -- the betting market has already priced everything it knows.
    """
    return schedule_strength(fit, schedule, weeks=weeks)


@dataclass(frozen=True, slots=True)
class PlayerSchedule:
    """One player's schedule strength, at his own position."""

    player_id: str
    name: str
    position: str
    team: str
    strength: ScheduleStrength

    @property
    def per_game(self) -> float:
        return self.strength.per_game


def player_schedule_strength(
    model: DvpModel,
    players: pl.DataFrame,
    schedule: pl.DataFrame,
    *,
    weeks: Iterable[int],
) -> list[PlayerSchedule]:
    """Per-player schedule strength for a frame of (player_id, position, team) rows.

    Extra columns are ignored, so a roster frame or the output of `player_weeks`
    deduplicated to one row per player both work. A player whose position has no
    fit, or whose team is not on the schedule, is dropped rather than given a
    zero -- a silent zero would rank him mid-pack instead of flagging him.
    """
    required = {"player_id", "position", "team"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"players frame is missing {sorted(missing)}")
    name_column = "player_display_name" if "player_display_name" in players.columns else None

    # Materialise once: `weeks` is an Iterable, so a generator would be drained by
    # the first position's table and every later position would raise "needs at
    # least one week" -- a failure whose message points nowhere near the cause.
    wanted = tuple(weeks)
    tables = {
        position: schedule_strength(fit, schedule, weeks=wanted)
        for position, fit in model.positions.items()
    }

    out: list[PlayerSchedule] = []
    for row in players.unique(subset=["player_id"], keep="first").iter_rows(named=True):
        table = tables.get(row["position"])
        if table is None:
            continue
        entry = table.get(row["team"])
        if entry is None:
            continue
        out.append(
            PlayerSchedule(
                player_id=row["player_id"],
                name=row[name_column] if name_column else row["player_id"],
                position=row["position"],
                team=row["team"],
                strength=entry,
            )
        )
    out.sort(key=lambda p: -p.per_game)
    return out
