"""Shared contracts between the projection, simulation, and decision layers.

This module is deliberately small and dependency-light. It exists so the layers
can be built and tested independently and still compose: a projection source
produces `ComponentLine`s, calibration turns those into `WeeklyOutlook`s, the
simulator consumes `WeeklyOutlook`s, and every decision surface returns a
`Recommendation` in the same unit.

The unit that matters is `delta_title` -- the change in championship probability.
Points are an intermediate quantity, never the answer. A waiver claim in one
league and a trade in another are comparable only because both report this.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

# --------------------------------------------------------------------------------------
# Positions and slots
# --------------------------------------------------------------------------------------

#: ESPN `defaultPositionId` for the positions we project. The full map lives in
#: espn/constants.py; these are the four that carry a fitted distribution.
QB, RB, WR, TE = 1, 2, 3, 4
K, DST = 5, 16

SKILL_POSITIONS: tuple[int, ...] = (QB, RB, WR, TE)

#: Every position that carries a calibration curve of its own. K and D/ST are here and
#: SKILL_POSITIONS is not enough, because the two are not interchangeable: the pooled
#: skill line SHRINKS a projection (slope 0.94) and a defence needs it EXPANDED (slope
#: 1.42). Measured leave-one-season-out on 2,144 D/ST player-weeks, the pooled line
#: leaves a -0.81/week level bias where the position's own line leaves -0.003.
FITTED_POSITIONS: tuple[int, ...] = (QB, RB, WR, TE, K, DST)


class Objective(StrEnum):
    """What a league actually pays for.

    Not cosmetic. In a points league, variance-reduction near the playoff cut is
    value-destroying rather than value-preserving, and the optimal lineup differs.
    """

    CHAMPIONSHIP = "championship"
    POINTS = "points"
    HYBRID = "hybrid"


# --------------------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComponentLine:
    """One source's projection for one player-week, in component stats.

    Component stats rather than points, because points are league-specific and
    components are not. A single ensemble then serves every league by applying
    each league's own scoring function -- which is the whole reason this system
    can run three leagues off one projection pipeline.

    `stats` is keyed by ESPN statId as a string, matching the raw ESPN payload
    and `LeagueScoring.score`.
    """

    player_id: int
    season: int
    week: int
    source: str
    stats: Mapping[str, float]
    #: None when the source does not say. Absence is informative -- a player a
    #: source omits entirely is different from one it projects at zero.
    games: float | None = None

    def points(
        self, scorer: Callable[[Mapping[str, float], int], float], position_id: int
    ) -> float:
        return scorer(self.stats, position_id)


@dataclass(frozen=True, slots=True)
class WeeklyOutlook:
    """A calibrated distribution over one player's fantasy points in one week.

    Hurdle gamma: `p_zero` mass at zero (injury, inactive, or a genuine goose
    egg) and a gamma over the positive part. Fitted on 24k paired player-weeks;
    a normal is the wrong family (skew to +1.8, excess kurtosis to +4.0).

    `mean` and `sd` are the moments of the FULL distribution including the zero
    mass, which is what lineup and simulation math wants. Do not reconstruct
    them from the gamma alone.
    """

    player_id: int
    season: int
    week: int
    position_id: int
    mean: float
    sd: float
    p_zero: float
    shape: float
    scale: float
    #: Team the player is on that week, or 0 if unknown. Drives the block-diagonal
    #: correlation structure -- different-team pairs are uncorrelated (+0.003).
    pro_team_id: int = 0
    #: False on a bye or when the player is out. Kept explicit rather than folded
    #: into p_zero so the simulator can distinguish "will not play" from "might blank".
    playing: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_zero <= 1.0:
            raise ValueError(f"p_zero must be a probability, got {self.p_zero}")
        if self.sd < 0:
            raise ValueError(f"sd must be non-negative, got {self.sd}")

    @property
    def variance(self) -> float:
        return self.sd * self.sd

    def zeroed(self) -> WeeklyOutlook:
        """This player, not playing. Used for byes and rest-of-season absences."""
        return replace(self, mean=0.0, sd=0.0, p_zero=1.0, playing=False)


@dataclass(frozen=True, slots=True)
class PlayerOutlook:
    """A player's calibrated week-by-week outlook for the rest of a season."""

    player_id: int
    name: str
    position_id: int
    pro_team_id: int
    weeks: Mapping[int, WeeklyOutlook]

    def mean_from(self, week: int) -> float:
        """Rest-of-season expected points from `week` inclusive.

        Always summed over remaining weeks. ESPN's own season-total field is
        frozen at preseason and never revised, so it must not be used here.
        """
        return sum(o.mean for w, o in self.weeks.items() if w >= week)

    def playoff_mean(self, playoff_weeks: Sequence[int]) -> float:
        return sum(o.mean for w, o in self.weeks.items() if w in set(playoff_weeks))


class ProjectionSource(Protocol):
    """Anything that can produce component projections.

    Sources are intentionally uniform so the ensemble can weight them equally and
    drop one that is unavailable without special-casing. `name` must be stable --
    it keys the historical-accuracy record and the ensemble's renormalization.
    """

    name: str

    def component_lines(self, season: int, week: int) -> Sequence[ComponentLine]: ...


# --------------------------------------------------------------------------------------
# League context
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeagueContext:
    """Everything a valuation or simulation needs to know about one league.

    A player's value is a function of (player, league). The same receiver is
    worth materially different amounts in a 14-team full-PPR league and a
    12-team half-PPR one, because both the scoring and the replacement level
    move. Nothing downstream may compute a league-independent player value.
    """

    league_id: int
    season: int
    name: str
    size: int
    #: lineupSlotId -> count. Slot ids, NOT position ids; the two spaces collide
    #: at 4 and 15. See espn/constants.py.
    lineup_slot_counts: Mapping[int, int]
    #: slotId -> the defaultPositionIds eligible for it, derived from ESPN.
    slot_eligibility: Mapping[int, frozenset[int]]
    #: (raw_stats, default_position_id) -> points, from espn/scoring.py.
    scorer: Callable[[Mapping[str, float], int], float]
    playoff_team_count: int
    playoff_weeks: tuple[int, ...]
    regular_season_weeks: tuple[int, ...]
    objective: Objective = Objective.CHAMPIONSHIP
    #: Only meaningful for HYBRID. Share of value from total points rather than titles.
    points_weight: float = 0.0
    uses_faab: bool = False
    faab_budget: int = 0
    my_team_id: int | None = None

    @property
    def starting_slots(self) -> dict[int, int]:
        """Slots that actually start a player, excluding bench/IR/invalid."""
        from .espn.scoring import NON_STARTING_SLOTS

        return {
            s: n
            for s, n in self.lineup_slot_counts.items()
            if n > 0 and s not in NON_STARTING_SLOTS
        }

    @property
    def starters_per_team(self) -> int:
        return sum(self.starting_slots.values())


# --------------------------------------------------------------------------------------
# Moves and recommendations
# --------------------------------------------------------------------------------------


class MoveKind(StrEnum):
    ADD_DROP = "add_drop"
    WAIVER_CLAIM = "waiver_claim"
    TRADE = "trade"
    LINEUP = "lineup"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class PlayerMove:
    """One player changing hands. `to_team`/`from_team` of None means the wire."""

    player_id: int
    from_team: int | None
    to_team: int | None


@dataclass(frozen=True, slots=True)
class Move:
    """A candidate action, possibly spanning several teams.

    A trade is just a `Move` whose `players` touch more than two teams, so
    two-team and multi-team trades share one representation and one evaluator.
    """

    kind: MoveKind
    league_id: int
    players: tuple[PlayerMove, ...] = ()
    #: Waiver claims only. None in a priority league, where the cost is the
    #: priority position rather than money.
    bid: int | None = None
    #: Lineup moves only: slotId -> playerId.
    lineup: Mapping[int, int] | None = None

    @property
    def teams(self) -> frozenset[int]:
        out: set[int] = set()
        for p in self.players:
            if p.from_team is not None:
                out.add(p.from_team)
            if p.to_team is not None:
                out.add(p.to_team)
        return frozenset(out)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """The common output of every decision surface.

    `delta_title` is the number that makes recommendations comparable across
    leagues. `delta_points` is kept for explanation only -- it is what the user
    intuitively expects to see, and showing both is how you demonstrate that a
    large points gain in a locked-up matchup is worth almost nothing.
    """

    move: Move
    delta_title: float
    delta_points: float
    #: Monte Carlo standard error on delta_title. A recommendation whose effect
    #: is smaller than its own error is noise and must be labelled as such.
    stderr: float = 0.0
    #: dPwin/dmu at the current margin, relative to its peak. Near zero means the
    #: decision does not matter this week regardless of who is better.
    leverage: float = 1.0
    rationale: str = ""
    confidence: str = "medium"
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def significant(self) -> bool:
        """True when the estimated effect exceeds two standard errors."""
        return abs(self.delta_title) > 2.0 * self.stderr if self.stderr > 0 else True

    def __lt__(self, other: Recommendation) -> bool:
        return self.delta_title < other.delta_title


@dataclass(frozen=True, slots=True)
class WireLevel:
    """What an unfilled starting slot streams off the wire, as a DISTRIBUTION.

    Lives here rather than in `decide/wire.py` because it is a contract between the
    layer that measures the wire (decide) and the layer that credits an empty seat
    (sim), and `sim` must not import `decide`.

    All three moments are read off the same body -- the player who is k-th best by
    projection in a given week -- so the triple is internally consistent and
    `calibration.hurdle_gamma_from_moments` reproduces it exactly. That is what lets
    the credited value have the solve floor as its mean by construction: the lineup
    decision is made on `mean`, and the seat is then paid a draw whose expectation is
    that same `mean`.
    """

    mean: float
    sd: float
    p_zero: float


class MoveEvaluator(Protocol):
    """Scores candidate moves in championship probability.

    Defined here rather than in `decide/title.py` so the decision surfaces can be
    written and tested against the contract without importing the engine -- and
    so a surface can be exercised with a cheap fake in unit tests while still
    being wired to the real two-tier engine in production.

    Implementations are expected to be two-tier: `screen` is an analytic
    approximation cheap enough to run over thousands of candidates, and `confirm`
    is a paired common-random-numbers simulation run only on the survivors. A
    caller that only ever uses `confirm` is correct but slow; one that only ever
    uses `screen` is fast and will occasionally rank a move wrongly near the
    playoff cut line, where the sign on the variance term flips.
    """

    def screen(self, moves: Sequence[Move]) -> list[Recommendation]:
        """Cheap analytic estimate for every candidate. May be approximate."""
        ...

    def confirm(self, moves: Sequence[Move]) -> list[Recommendation]:
        """Paired CRN simulation. Authoritative, and populates `stderr`."""
        ...

    def baseline_title(self, team_id: int) -> float:
        """P(championship) for a team under no move, for reporting a delta against."""
        ...


# --------------------------------------------------------------------------------------
# Win-probability helpers
# --------------------------------------------------------------------------------------


#: Standard normal pdf/cdf without pulling scipy into hot loops.
def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def win_probability(margin: float, sd_diff: float) -> float:
    """P(win) for a normal margin. `sd_diff` is the SD of (you - opponent)."""
    if sd_diff <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    return _Phi(margin / sd_diff)


def points_to_win_prob(sd_diff: float, margin: float = 0.0) -> float:
    """Win probability gained per additional projected point.

    dP/dmu = phi(z)/sd_diff. At an even matchup with the measured league sd_diff
    of 34.4 this is ~1.16 percentage points per point; by |z|=2 it has fallen to
    ~14% of that. This is why "does this decision matter at all" is usually more
    useful than "who is better".
    """
    if sd_diff <= 0:
        return 0.0
    return _phi(margin / sd_diff) / sd_diff


def leverage(margin: float, sd_diff: float) -> float:
    """Marginal value of a point relative to its value at an even matchup.

    1.0 in a coin flip, ~0.32 at |z|=1.5, ~0.14 at |z|=2.
    """
    if sd_diff <= 0:
        return 0.0
    return _phi(margin / sd_diff) / _phi(0.0)


def swap_improves_win_probability(
    d_mean: float, d_sd: float, margin: float, sd_diff: float
) -> bool:
    """Whether a lineup swap raises P(win), not expected points.

    The rule is `d_mean - z * d_sd > 0` where z is the current standardized
    margin. A favorite should reject variance; an underdog should accept a
    negative d_mean for enough d_sd. Same roster, same week, opposite answer
    depending on the scoreboard.
    """
    if sd_diff <= 0:
        return d_mean > 0
    z = margin / sd_diff
    return (d_mean - z * d_sd) > 0
