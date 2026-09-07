"""Start/sit priced in win probability, which is not the same thing as points.

A projection list ranks players by expected points and stops. That is the right answer
for exactly one scoreboard -- a coin flip -- and the wrong one everywhere else, because
what a lineup is actually asked to do is clear a threshold, and clearing a threshold is
a question about the whole distribution. `core.swap_improves_win_probability` states the
rule in one line:

    d_mu - z * d_sd > 0,    z = current standardized margin

A favorite is buying certainty and should refuse variance; an underdog is buying tail
mass and should pay projected points for it. Same roster, same week, opposite answer.

Four things this module insists on, each because getting it wrong produces a
confident-looking recommendation that is worth nothing.

**1. The selection step makes no distributional assumption.** Rosters are small, so the
feasible lineups are enumerated (a few hundred after de-duplication) and each one is
scored against the *same* correlated Monte Carlo draw. The winner is the empirical
argmax of `P(clear the threshold)`. The `d_mu - z*d_sd` rule above is used for the
guards and reported as a diagnostic, never as the objective -- weekly fantasy scores are
skewed hurdle-gamma sums and the normal rule is a linearization of them.

**2. The override is guarded, because the effect is small.** Variance tuning is worth
0.3-0.5pp of weekly win probability and it flips sign with the scoreboard; projection
error is worth several points. So the expected-points lineup stands unless `|z| >
OVERRIDE_Z` **and** the sacrifice is under `MAX_POINTS_SACRIFICE` projected points --
and, added here because the first two guards do not cover it, unless the measured gain
clears a noise floor that accounts for the argmax having been taken over hundreds of
candidates (`selection_penalty`). Without that third guard the surface trades real
projected points for a win-probability "gain" that is a handful of simulations wide,
which is how a tool like this destroys value while looking sophisticated. Measured on the
user's three leagues, 646 team-weeks at 2,000 simulations: the win-probability lineup
differed from the points lineup 201 times, the mean measured gain was 0.5pp against a
mean paired standard error of 0.68pp, and after the selection correction essentially
none of them survive. That is the honest answer -- start your studs -- and the guards
are what produce it.

The guards were then checked the only way that settles it, by holding out football the
argmax never saw. Draw 8,000 seasons on the three real leagues, pick the
win-probability lineup on the first 4,000, and score it against the opponent on the
second 4,000. Over the 89 team-weeks where the two lineups differed the in-sample gain
averaged **+0.35pp** and the out-of-sample gain averaged **-0.24pp**, positive in 30% of
them, with an in/out correlation of 0.14. The measured edge is not merely noise, it is
noise bought with real projected points. None of the 89 cleared `selection_penalty`, so
the shipped surface took none of them. Anyone tempted to relax `MIN_EDGE_SIGMA` or
`OVERRIDE_Z` should re-run that split first.

**3. Late in the regular season the threshold is the playoff cut line, not the
opponent.** This is the headline, and a pure-projection tool cannot produce it. In the
last `CUT_LINE_WINDOW` weeks of the regular season the thing at stake is a berth or a
bye, so the score to beat is derived from the simulator itself: with every other
outcome in a simulation held fixed, a team's own week score is *monotone* in whether it
makes the bracket, so there is a critical score per simulation and it is found by
bisection (`season_threshold`). That vector plays exactly the role the opponent's score
plays in an ordinary week, and everything downstream -- margin, z, leverage, the
enumeration's argmax -- is unchanged. A 7-3 team already in reads z >> 0 and refuses
variance; a 4-6 team needing to run the table reads z << 0 and buys it.

The same machinery covers a bracket week, where there is no scheduled opponent and the
monotone quantity is the title itself.

**4. Most weeks, the honest output is "this does not matter".** `core.leverage` is
reported with every recommendation: at |z| = 2 a start/sit call is worth one seventh of
the same call in a coin flip. And in the cut-line weeks a simulation where the outcome
is settled whatever the lineup does contributes nothing at all, so leverage is scaled by
the share of simulations the week is actually decisive in. A team locked into a bye gets
told plainly that its lineup cannot move its season, which is more valuable than a
confident recommendation that cannot.

**Opponent correlation is a real edge and half of it is not modellable here.** Season-
long is easier than DFS because the opponent's lineup is known, so the covariance term
in `sd_diff = sqrt(var_me + var_opp - 2 cov)` is computable rather than assumed. It
falls straight out of the draw: the correlation blocks in `sim/distributions.py` are
block-diagonal by NFL team, so starting a receiver whose quarterback the opponent starts
genuinely raises `cov` and shrinks `sd_diff` -- good when favored, bad when chasing.
`StackEdge` lists those pairs with the measured rho.

What is **not** available: the folklore play of starting a D/ST against the opponent's
quarterback. That is a *cross-team* correlation, and the corpus measured cross-team
pairs at +0.003 with a league-wide weekly factor explaining 0.71% of residual variance,
so `SAME_TEAM_RHO` carries no D/ST entry and no opposing-offense term at all. The draw
therefore gives that pairing exactly zero covariance. Rather than invent a constant,
this module reports the effect as unmodelled; see `StackEdge` and the module tests.

`Move.lineup` deliberately maps **player id -> lineup slot id**, the reverse of the
direction `core.Move` suggests. A slot id is not unique -- an ordinary league starts two
RB and two WR -- so a `Mapping[int, int]` keyed by slot silently drops half a lineup,
while keyed by player it is exact. The two spaces do not collide (slot ids are 0-23,
player ids are seven figures and D/ST are negative), so the direction is recoverable by
inspection. `LineupOption` carries the aligned `slot_ids`/`player_ids` pair and is the
authoritative form.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..core import (
    SKILL_POSITIONS,
    Move,
    MoveKind,
    Recommendation,
    leverage,
    swap_improves_win_probability,
)
from ..sim import season as S
from ..sim.distributions import CorrelationModel, Draw
from ..sim.lineup import LineupPlan, plan_from_slots

if TYPE_CHECKING:  # pragma: no cover - only the type checker needs these
    from ..espn.league import TeamRoster

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------

#: Standardized margin below which the expected-points lineup simply stands. RESEARCH.md
#: puts variance tuning at 0.3-0.5pp of weekly win probability; inside |z| = 0.4 the
#: whole effect is smaller than the error on the projections that ranked the players.
OVERRIDE_Z = 0.4

#: Projected points a win-probability override may spend. Two points is roughly half a
#: standard error on a single player's weekly projection, so beyond it the surface would
#: be trading a known quantity for a modelled one.
MAX_POINTS_SACRIFICE = 2.0

#: Standard errors the measured win-probability gain must clear before the override is
#: taken, *after* the winner's-curse correction below. Not in the research notes and
#: added here on purpose: the effect being chased is 0.3-0.5pp and the paired resolution
#: at a few thousand simulations is the same size, so without this the surface pays real
#: projected points for a difference of a handful of simulations.
MIN_EDGE_SIGMA = 1.0

#: How many of the regular season's final weeks price against the playoff cut line
#: rather than against the opponent. Five weeks of a fourteen-week regular season is
#: weeks 10-14, which is what RESEARCH.md names; derived from the schedule rather than
#: hard-coded so a league with a different length lands in the right place.
CUT_LINE_WINDOW = 5

#: Above this playoff probability the berth is not the live question and the cut line
#: that binds is the bye line.
LOCKED_PLAYOFFS = 0.98

#: Widest swing in a single week's team score the threshold search will consider. No
#: start/sit decision moves a week by 150 points, so a simulation whose outcome flips
#: only outside this range is settled *for this decision*, which is the honest reading.
MAX_SWING = 150.0

#: Halvings of the threshold bracket. Twelve puts the critical score inside 0.08 points.
BISECTION_STEPS = 12

#: Extra players past the slot count each slot group considers, per ranking key. The
#: keys are projected mean, mean + sd and mean - sd, so a boom/bust bench receiver an
#: underdog might want is in the pool even though his mean is not top-two.
DEFAULT_EXTRA_DEPTH = 2

#: What a change to an already-set lineup has to be worth before it is recommended at
#: all: a quarter of a projected point, or half a point of threshold probability. Both
#: are inside anyone's noise, and a tool that tells a manager to swap two receivers for
#: +0.01 projected points is not advising him, it is generating work. See `_churn_guard`
#: for the live case that put these here.
CHURN_TOLERANCE_POINTS = 0.25
CHURN_TOLERANCE_PROB = 0.005

#: Cap on enumerated lineups. Scoring is one matrix multiply, so this is about keeping
#: the enumeration honest rather than about speed: past a few thousand the argmax is
#: picking between lineups that differ by less than the Monte Carlo resolution.
MAX_LINEUPS = 2048


def selection_penalty(n_candidates: int) -> float:
    """How many standard errors the *best of n* noisy estimates beats the truth by.

    The winner's curse, and without it this surface is a machine for manufacturing
    edges. `advise` takes the argmax of `P(clear the threshold)` over a few hundred
    candidate lineups, each estimated from the same finite draw, so the winning estimate
    is biased high by roughly the expected maximum of n standard normals -- about
    `sqrt(2 ln n)`, which is 3.3 at 200 candidates. Comparing the winner against one
    standard error therefore certifies noise as signal in a fixed, predictable way.

    Deliberately conservative in two directions and neither is an accident. The bound is
    for *independent* estimates and these are anything but -- candidate lineups share
    seven of nine starters -- so the true expected maximum is smaller and this over-
    penalises. And it grows with the candidate count, which is right: a two-candidate
    roster has no selection problem, and a wide-open one has a large one. The cost of a
    false override is real projected points spent on nothing, so erring toward "start
    your studs" is the cheap mistake.

    Measured consequence, stated plainly: on the user's three leagues at 2,000
    simulations this floor is 1-2pp of weekly win probability against an effect of
    0.3-0.5pp, so the override essentially never fires on an ordinary matchup. Resolving
    a real 0.4pp edge against a 200-lineup argmax needs simulations in the tens of
    thousands.
    """
    return math.sqrt(2.0 * math.log(max(n_candidates, 2)))


class StartSitError(ValueError):
    """The start/sit question as posed cannot be answered for this week."""


class ThresholdKind(StrEnum):
    """What the lineup is being asked to beat."""

    OPPONENT = "opponent"
    PLAYOFF_CUT = "playoff_cut"
    BYE_CUT = "bye_cut"
    TITLE = "title"


# --------------------------------------------------------------------------------------
# The threshold
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Threshold:
    """The score this week has to beat, per simulation.

    One representation for both regimes, which is the point: an ordinary week's
    threshold is the opponent's total, and a cut-line week's is the score at which the
    season outcome flips. Everything downstream -- margin, z, leverage, the empirical
    argmax over lineups -- reads this and does not care which it got.

    `-inf` marks a simulation whose outcome is already secured however the lineup is
    set, `+inf` one already lost; both are exact under the `total > score` comparison and
    are excluded from the margin and spread, which are only meaningful where the week
    can still decide something.
    """

    kind: ThresholdKind
    #: (sims,)
    score: np.ndarray
    #: (sims,) where the week can still change the outcome.
    decisive: np.ndarray
    #: What clearing it wins, in words, for the rationale.
    label: str

    @property
    def decisive_share(self) -> float:
        return float(self.decisive.mean()) if self.decisive.size else 0.0

    @property
    def live(self) -> bool:
        """Whether the margin and spread are estimable at all."""
        return int(self.decisive.sum()) >= 2


# --------------------------------------------------------------------------------------
# Lineups, swaps and correlation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupOption:
    """One feasible starting lineup and how it performs against the threshold.

    `mean` is the *projected* total -- deterministic, and the number a points-maximizing
    tool would show. Everything else is measured on the draw, so the spread carries the
    within-team correlation and the covariance with whatever it is being compared to.
    """

    slot_ids: tuple[int, ...]
    player_ids: tuple[int, ...]
    names: tuple[str, ...]
    mean: float
    sd: float
    #: mean(total - threshold) over the decisive simulations.
    margin: float
    #: sd of that difference: `sqrt(var_me + var_them - 2 cov)`.
    sd_diff: float
    covariance: float
    win_prob: float

    @property
    def z(self) -> float:
        return self.margin / self.sd_diff if self.sd_diff > 0 else 0.0

    def slot_map(self) -> dict[int, int]:
        """player id -> lineup slot id. See the module docstring for the direction."""
        return dict(zip(self.player_ids, self.slot_ids, strict=True))

    def describe(self) -> str:
        parts = [f"{n} ({s})" for s, n in zip(self.slot_ids, self.names, strict=True)]
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class Swap:
    """One player in for one player out, with what the exchange buys and costs."""

    slot_id: int
    out_player_id: int
    out_name: str
    in_player_id: int
    in_name: str
    d_mean: float
    d_sd: float

    def describe(self) -> str:
        return (
            f"start {self.in_name} over {self.out_name} "
            f"({self.d_mean:+.1f} pts, {self.d_sd:+.1f} sd)"
        )


@dataclass(frozen=True, slots=True)
class StackEdge:
    """A player of mine who shares an NFL team with one of my opponent's starters.

    Positive rho raises the covariance and *shrinks* `sd_diff`: the two scores move
    together, so the matchup is decided by less randomness. That is what a favorite
    wants and what an underdog must avoid.

    `modelled` distinguishes the two ways a pairing can come out at zero. A skill-position
    pair the table does not list -- RB with TE, WR with WR -- was *measured* at zero and
    is simply not an exposure. A pairing involving a kicker or a D/ST was never measured
    at all: the corpus fitted QB/RB/WR/TE only, so the simulator gives it exactly zero
    covariance because it has no number, not because it found one. Those are reported
    rather than filtered, so the output never looks like it checked something it did not.
    """

    player_id: int
    name: str
    position_id: int
    pro_team_id: int
    opponent_player_id: int
    opponent_name: str
    opponent_position_id: int
    rho: float
    modelled: bool = True

    @property
    def shrinks_spread(self) -> bool:
        return self.rho > 0.0


@dataclass(frozen=True, slots=True)
class OpponentView:
    """Who I am playing and what they are starting. Known, unlike in DFS."""

    team_id: int
    name: str
    starters: tuple[int, ...]
    starter_names: tuple[str, ...]
    mean: float
    sd: float


# --------------------------------------------------------------------------------------
# The advice
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupAdvice:
    """Everything the start/sit surface found, with the comparable answer attached.

    `recommendation.delta_title` is measured against `baseline` -- the currently set
    lineup when one was supplied, otherwise the expected-points lineup -- by a paired
    common-random-numbers season simulation in which only this team's score in this one
    week differs. `delta_win_prob` is the weekly (or cut-line) probability difference
    between the win-probability lineup and the expected-points lineup, which is the
    number this module is actually about; the title delta is what makes it comparable
    with a waiver claim in another league and is usually below its own noise, because a
    single week's lineup is a small thing.
    """

    league_id: int
    season: int
    team_id: int
    team_name: str
    week: int
    kind: ThresholdKind
    target: str
    margin: float
    sd_diff: float
    z: float
    leverage: float
    decisive_share: float
    points_lineup: LineupOption
    win_prob_lineup: LineupOption
    recommended: LineupOption
    baseline: LineupOption
    swaps: tuple[Swap, ...]
    delta_win_prob: float
    delta_win_prob_stderr: float
    #: `delta_win_prob` has to clear this to be believed -- the paired standard error
    #: inflated by `selection_penalty` for having taken the best of `n_lineups`.
    noise_floor: float
    #: Projected points the win-probability lineup gives up against the points lineup.
    #: Never negative, and deliberately not the same quantity as
    #: `recommendation.delta_points`, which is measured against the baseline.
    points_sacrifice: float
    #: Empty when the win-probability lineup was adopted; otherwise why it was not.
    guard: str
    #: Whether `core.swap_improves_win_probability` agrees with the empirical argmax.
    rule_agrees: bool
    opponent: OpponentView | None
    stacks: tuple[StackEdge, ...]
    #: What `sd_diff` would be with no covariance at all. The gap is the correlation edge.
    sd_independent: float
    recommendation: Recommendation
    n_sims: int
    n_lineups: int
    #: The starters that were supplied as `current` and could **not** be priced, because
    #: no legal slot assignment exists for them this week. Empty in the ordinary case.
    #: Non-empty means the deltas below are measured against the projected-best lineup
    #: rather than against what the manager has set, and that his lineup is broken.
    unpriced_current: tuple[int, ...] = ()

    @property
    def differ(self) -> bool:
        """Whether the win-probability lineup and the points lineup disagree at all."""
        return set(self.win_prob_lineup.player_ids) != set(self.points_lineup.player_ids)

    @property
    def significant(self) -> bool:
        """Stricter than `Recommendation.significant`, which calls a zero effect real.

        `core.Recommendation.significant` returns True when `stderr` is zero, which is
        right for an analytic surface and wrong here: recommending the lineup that is
        already set makes both simulated arms bit-identical, so the difference and its
        standard error are both exactly zero. A no-op is not a significant finding.
        """
        return self.recommendation.delta_title != 0.0 and self.recommendation.significant

    def report(self) -> str:
        """A few lines a human can act on."""
        lines = [
            f"{self.team_name} -- {self.league_id} week {self.week}",
            f"  threshold      {self.kind.value} ({self.target})",
        ]
        if self.unpriced_current:
            lines.append(
                f"  WARNING        the {len(self.unpriced_current)} starters you have set "
                "cannot be legally assigned to this week's slots and were NOT priced; "
                "everything below is against the projected-best lineup"
            )
        lines += [
            f"  margin         {self.margin:+.1f} +/- {self.sd_diff:.1f}  z={self.z:+.2f}  "
            f"leverage={self.leverage:.2f}",
            f"  points lineup  {self.points_lineup.mean:.1f} pts, "
            f"P={self.points_lineup.win_prob * 100:.1f}%, sd_diff={self.points_lineup.sd_diff:.1f}",
            f"  win-prob lineup{self.win_prob_lineup.mean:8.1f} pts, "
            f"P={self.win_prob_lineup.win_prob * 100:.1f}%, "
            f"sd_diff={self.win_prob_lineup.sd_diff:.1f}",
            f"  differ         {self.differ} (points lineup against win-probability lineup)",
            "  change         "
            + ("; ".join(s.describe() for s in self.swaps) if self.swaps else "none"),
            f"  dP(threshold)  {self.delta_win_prob * 100:+.2f}pp "
            f"+/- {self.delta_win_prob_stderr * 100:.2f}pp for "
            f"-{self.points_sacrifice:.1f} pts "
            f"(noise floor {self.noise_floor * 100:.2f}pp over {self.n_lineups} lineups)",
            f"  guard          {self.guard or ('none -- taken' if self.differ else 'n/a')}",
            f"  dTitle         {self.recommendation.delta_title * 100:+.3f}pp "
            f"+/- {self.recommendation.stderr * 100:.3f}pp "
            f"({'significant' if self.significant else 'not significant'})",
            f"  correlation    sd_diff {self.sd_diff:.2f} against {self.sd_independent:.2f} "
            f"independent ({self.sd_diff - self.sd_independent:+.2f})",
        ]
        for edge in self.stacks:
            lines.append(
                f"    stack        {edge.name} with {edge.opponent_name} rho={edge.rho:+.2f}"
                + ("" if edge.modelled else " (UNMODELLED: no measured constant)")
            )
        lines.append(f"  {self.recommendation.rationale}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SlotGroup:
    slot_id: int
    count: int
    #: Local roster indices eligible for this slot and startable this week.
    eligible: np.ndarray


def _slot_groups(
    state: S.LeagueState, positions: np.ndarray, startable: np.ndarray
) -> tuple[_SlotGroup, ...]:
    """Starting slots, most restrictive first, against one roster's startable players.

    Restrictive-first is the same order `sim/lineup.py` solves in and it is what makes
    the enumeration cheap: the flex is filled last, from whoever the dedicated slots did
    not take, so the recursion never has to undo a choice.
    """
    from ..espn.scoring import NON_STARTING_SLOTS

    groups = []
    for slot, count in state.lineup_slot_counts.items():
        if count <= 0 or slot in NON_STARTING_SLOTS:
            continue
        allowed = np.fromiter(state.slot_eligibility[slot], dtype=np.int64)
        eligible = np.flatnonzero(startable & np.isin(positions, allowed))
        groups.append(_SlotGroup(slot_id=int(slot), count=int(count), eligible=eligible))
    groups.sort(key=lambda g: (g.eligible.size, g.slot_id))
    return tuple(groups)


def _pool(candidates: Sequence[int], keys: Sequence[np.ndarray], depth: int) -> list[int]:
    """The players worth considering for one slot group: the best `depth` by each key.

    Three keys rather than one, and this is not decoration. Ranking the pool by projected
    mean alone throws away exactly the players a win-probability lineup exists to find:
    the fourth-best receiver by mean who is the best by upside is the one an underdog
    wants, and he is outside a mean-ranked pool of three. `mean + sd` and `mean - sd`
    bracket the two directions the override can go.
    """
    if len(candidates) <= depth:
        return list(candidates)
    chosen: set[int] = set()
    idx = np.asarray(candidates, dtype=np.intp)
    for key in keys:
        order = idx[np.argsort(-key[idx], kind="stable")]
        chosen.update(int(p) for p in order[:depth])
    return sorted(chosen, key=lambda p: (-float(keys[0][p]), p))


def enumerate_lineups(
    groups: Sequence[_SlotGroup],
    keys: Sequence[np.ndarray],
    *,
    extra: int = DEFAULT_EXTRA_DEPTH,
    cap: int = MAX_LINEUPS,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Every distinct legal lineup worth scoring.

    Returns `(lineups, slot_ids, extra_used)` where `lineups` is
    `(n_lineups, n_slot_instances)` of local roster indices with -1 for an empty slot,
    and `slot_ids` names the instances.

    De-duplicated on the *set* of starters: two slot assignments of the same nine players
    score identically, and a real roster generates a lot of them (any RB in the flex is
    also any RB in an RB slot). Deduping is what keeps a 14-man roster at a few hundred
    candidates instead of a few thousand.

    `extra` shrinks until the enumeration fits under `cap`, so the cap binds by making
    the pools shallower rather than by truncating the list at an arbitrary point --
    truncation would silently drop whole regions of the lineup space.
    """
    slot_ids = tuple(g.slot_id for g in groups for _ in range(g.count))
    n_slots = len(slot_ids)
    for depth_extra in range(max(extra, 0), -1, -1):
        rows: dict[tuple[int, ...], tuple[int, ...]] = {}
        overflow = _walk(groups, keys, depth_extra, cap, rows, 0, frozenset(), ())
        if not overflow:
            break
    if not rows:  # pragma: no cover - _walk always emits at least the empty lineup
        rows = {(): tuple([-1] * n_slots)}
    lineups = np.array(list(rows.values()), dtype=np.intp).reshape(len(rows), n_slots)
    if overflow:
        log.warning(
            "start/sit enumeration hit the %d-lineup cap even at zero extra depth; the "
            "argmax is over a truncated candidate set",
            cap,
        )
    return lineups, slot_ids, depth_extra


def _walk(
    groups: Sequence[_SlotGroup],
    keys: Sequence[np.ndarray],
    extra: int,
    cap: int,
    rows: dict[tuple[int, ...], tuple[int, ...]],
    i: int,
    used: frozenset[int],
    chosen: tuple[int, ...],
) -> bool:
    """Depth-first over slot groups. True if the cap was hit."""
    if len(rows) > cap:
        return True
    if i == len(groups):
        key = tuple(sorted(p for p in chosen if p >= 0))
        rows.setdefault(key, chosen)
        return len(rows) > cap
    group = groups[i]
    available = [int(p) for p in group.eligible if p not in used]
    pool = _pool(available, keys, group.count + extra)
    combos: Iterable[tuple[int, ...]]
    if len(pool) <= group.count:
        combos = [tuple(pool)]
    else:
        combos = itertools.combinations(pool, group.count)
    for combo in combos:
        padded = combo + (-1,) * (group.count - len(combo))
        if _walk(groups, keys, extra, cap, rows, i + 1, used | set(combo), chosen + padded):
            return True
    return False


def _indicator(lineups: np.ndarray, n_players: int) -> np.ndarray:
    """`(players, lineups)` 0/1 matrix, so scoring every candidate is one matmul."""
    out = np.zeros((n_players, lineups.shape[0]), dtype=np.float32)
    for row, lineup in enumerate(lineups):
        for p in lineup:
            if p >= 0:
                out[p, row] = 1.0
    return out


# --------------------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------------------


def _week_outcome(
    state: S.LeagueState,
    scores: np.ndarray,
    week_index: int,
    team_index: int,
    offsets: np.ndarray,
    field: str,
) -> np.ndarray:
    """Whether the target outcome happens with this team's week score shifted per sim."""
    trial = scores.copy()
    trial[:, week_index, team_index] = scores[:, week_index, team_index] + offsets
    result = S.simulate_from_scores(state, trial, all_play=False)
    return np.asarray(getattr(result, field)[:, team_index], dtype=bool)


def season_threshold(
    state: S.LeagueState,
    scores: np.ndarray,
    *,
    week_index: int,
    team_index: int,
    field: str,
    kind: ThresholdKind,
    label: str,
) -> Threshold:
    """The week score at which a season outcome flips, found per simulation.

    With every other outcome in a simulation held fixed by common random numbers, a
    team's own score in one week is *monotone* in whether it makes the bracket: more
    points cannot lose a game it would have won, cannot lower its points-for tiebreak,
    and cannot help a rival. So there is a single critical score per simulation, and
    bisection finds it -- vectorised, because the offset may differ per simulation, so
    each halving is one `simulate_from_scores` call for the whole tensor.

    Monotonicity is checked rather than assumed: a simulation that misses the target with
    the largest possible boost *and* hits it with the largest possible penalty is logged.
    `champions` is the one field where that can genuinely happen, since winning a bracket
    game changes who you meet next; it is rare enough to report and not to model.
    """
    n_sims = scores.shape[0]
    base = scores[:, week_index, team_index].astype(np.float64)
    worst = _week_outcome(state, scores, week_index, team_index, np.full(n_sims, -MAX_SWING), field)
    best = _week_outcome(state, scores, week_index, team_index, np.full(n_sims, MAX_SWING), field)
    inverted = int(np.count_nonzero(worst & ~best))
    if inverted:
        log.warning(
            "%d of %d simulations are non-monotone in this team's week score for %r; "
            "their threshold is read off the upper branch",
            inverted,
            n_sims,
            field,
        )
    locked, lost = worst, ~best
    lo = np.full(n_sims, -MAX_SWING)
    hi = np.full(n_sims, MAX_SWING)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        hit = _week_outcome(state, scores, week_index, team_index, mid, field)
        lo = np.where(hit, lo, mid)
        hi = np.where(hit, mid, hi)
    critical = base + 0.5 * (lo + hi)
    score = np.where(locked, -np.inf, np.where(lost, np.inf, critical))
    return Threshold(kind=kind, score=score, decisive=~locked & ~lost, label=label)


def _cut_line_weeks(state: S.LeagueState, window: int) -> frozenset[int]:
    """The last `window` scheduled regular-season weeks, where a berth is the question.

    Derived from the schedule rather than hard-coded to 10-14: on a fourteen-week regular
    season it *is* 10-14, and on any other length it is still the run-in.
    """
    # `weeks[-0:]` is the whole season, not none of it, so a zero window has to be
    # spelled out: `cut_line_window=0` is how a caller asks for the opponent threshold
    # everywhere, and silently pricing every week against the cut line instead would be
    # a very quiet way to give the wrong answer.
    weeks = state.regular_season_weeks
    if window <= 0 or not weeks:
        return frozenset()
    return frozenset(weeks[-window:])


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


class HasSim(Protocol):
    """The part of `pipeline.LeagueSim` this module needs, without importing it."""

    state: S.LeagueState
    draw: Draw


def starters_from_roster(roster: TeamRoster) -> tuple[int, ...]:
    """The player ids a manager currently has in starting slots.

    The reason to pass these into `advise` rather than assume a projection-optimal
    lineup: what the surface is worth to a user is measured against what is actually in
    his lineup right now, and a manager who has not logged in is a different problem from
    a manager whose ninth starter is one point light.
    """
    return tuple(e.player_id for e in roster.starters)


@dataclass(frozen=True, slots=True)
class _Week:
    """One (team, week) slice, with every array the rest of the module reads.

    Assembled once because the pieces have to agree: the local roster index used by the
    enumeration, the tensor column it maps to, and the projected moments it is ranked on
    are three views of the same player and a mismatch between any two is silent.
    """

    state: S.LeagueState
    draw: Draw
    week: int
    wi: int
    team_id: int
    t: int
    franchise: S.Franchise
    #: Local roster index -> pool column.
    cols: np.ndarray
    positions: np.ndarray
    #: (sims, roster) realised points this week, already carrying the efficiency factor.
    points: np.ndarray
    #: (roster,) projected moments, and whether the player has a game at all.
    means: np.ndarray
    sds: np.ndarray
    startable: np.ndarray
    #: (sims, teams) lineup-efficiency multipliers, drawn once so every arm meets the
    #: same set of opposing managers.
    factors: np.ndarray
    #: (sims, weeks, teams) whole-league scores under the simulator's own ex-ante lineups.
    base_scores: np.ndarray

    @property
    def n_sims(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_players(self) -> int:
        return int(self.points.shape[1])

    def name(self, local: int) -> str:
        return self.state.pool.name(int(self.franchise.player_ids[local]))


def _prepare(
    state: S.LeagueState,
    draw: Draw,
    *,
    week: int,
    team_id: int,
    efficiency: S.LineupEfficiency,
) -> _Week:
    wi = state.week_index[week]
    t = state.team_index[team_id]
    franchise = state.franchise(team_id)
    cols = state.pool.columns(franchise.player_ids)
    factors = efficiency.draw(state, draw.n_sims)
    points = np.asarray(draw.points[:, wi][:, cols], dtype=np.float64) * factors[:, t : t + 1]
    panel = draw.panel
    return _Week(
        state=state,
        draw=draw,
        week=week,
        wi=wi,
        team_id=team_id,
        t=t,
        franchise=franchise,
        cols=cols,
        positions=np.asarray(state.pool.position_ids, dtype=np.int64)[cols],
        points=points,
        means=np.asarray(panel.mean[wi, cols], dtype=np.float64),
        sds=np.asarray(panel.sd[wi, cols], dtype=np.float64),
        # A bye or a player already ruled out is not a lineup choice. Injuries drawn
        # *inside* a simulation are not filtered here on purpose: on Sunday morning you
        # do not know about them, and pretending otherwise is the hindsight bug that
        # `sim/season.py` exists to avoid.
        startable=np.asarray(panel.has_game[wi, cols], dtype=bool),
        factors=factors,
        base_scores=S.team_week_scores(state, draw, efficiency=factors),
    )


def _game_for(state: S.LeagueState, week: int, team_id: int) -> S.ScheduledGame | None:
    for game in state.remaining_games:
        if week in game.weeks and team_id in (game.home_team_id, game.away_team_id):
            return game
    return None


def _opponent(
    w: _Week, game: S.ScheduledGame, starters: Sequence[int] | None
) -> tuple[OpponentView, np.ndarray]:
    """The opponent's view and the `(sims,)` score this week's lineup has to beat.

    Their weekly total is built from a **fixed** starting eleven -- the one they have
    actually set if it was supplied, otherwise their projection-optimal one -- rather
    than from the simulator's per-simulation lineup, so the comparison is like for like
    and the covariance with my own candidates is the real one.

    In a multi-week matchup period the other weeks are not a lineup decision, so they are
    folded into the threshold: whatever I am projected to add and they are projected to
    add in those weeks shifts the score I need this week, one for one.
    """
    other_id = game.away_team_id if game.home_team_id == w.team_id else game.home_team_id
    other = w.state.franchise(other_id)
    ot = w.state.team_index[other_id]
    cols = w.state.pool.columns(other.player_ids)
    positions = np.asarray(w.state.pool.position_ids, dtype=np.int64)[cols]
    panel = w.draw.panel
    means = np.asarray(panel.mean[w.wi, cols], dtype=np.float64)
    startable = np.asarray(panel.has_game[w.wi, cols], dtype=bool)

    if starters is None:
        plan = plan_from_slots(w.state.lineup_slot_counts, w.state.slot_eligibility, positions)
        result = plan.solve(np.where(startable, means, -np.inf), assignment=True)
        assert result.assignment is not None
        chosen = [int(p) for p in result.assignment if p >= 0]
    else:
        index = {int(pid): i for i, pid in enumerate(other.player_ids)}
        missing = [p for p in starters if p not in index]
        if missing:
            raise StartSitError(f"opponent starters {missing} are not on team {other_id}")
        chosen = [index[int(p)] for p in starters]

    raw = np.asarray(w.draw.points[:, w.wi][:, cols], dtype=np.float64)
    points = raw * w.factors[:, ot : ot + 1]
    total = points[:, chosen].sum(axis=1)
    threshold = total.copy()
    for other_week in game.weeks:
        if other_week == w.week:
            continue
        wj = w.state.week_index[other_week]
        threshold += w.base_scores[:, wj, ot] - w.base_scores[:, wj, w.t]
    view = OpponentView(
        team_id=other_id,
        name=other.name,
        starters=tuple(int(other.player_ids[i]) for i in chosen),
        starter_names=tuple(w.state.pool.name(int(other.player_ids[i])) for i in chosen),
        mean=float(means[chosen].sum()),
        sd=float(total.std(ddof=1)),
    )
    return view, threshold


def _pick_threshold(
    w: _Week,
    *,
    scores: np.ndarray,
    game: S.ScheduledGame | None,
    opponent_score: np.ndarray | None,
    cut_line_window: int,
) -> Threshold:
    """Opponent, playoff cut, bye cut or title -- whichever is the live question.

    The order matters and is the module's whole strategic claim. Inside the run-in the
    berth is what the week is for, so the cut line replaces the opponent *even though
    there is a scheduled game*; beating your opponent is then a means rather than the
    end, and the simulator already knows the difference.
    """
    if game is None:
        if not any(w.week in r for r in w.state.playoff_rounds):
            raise StartSitError(
                f"week {w.week} has no scheduled game for team {w.team_id} and is not a "
                "bracket week; nothing to set a lineup against"
            )
        return season_threshold(
            w.state,
            scores,
            week_index=w.wi,
            team_index=w.t,
            field="champions",
            kind=ThresholdKind.TITLE,
            label="the championship",
        )
    if w.week in _cut_line_weeks(w.state, cut_line_window):
        result = S.simulate_from_scores(w.state, scores, all_play=False)
        playoffs = float(result.made_playoffs[:, w.t].mean())
        if playoffs >= LOCKED_PLAYOFFS:
            return season_threshold(
                w.state,
                scores,
                week_index=w.wi,
                team_index=w.t,
                field="byes",
                kind=ThresholdKind.BYE_CUT,
                label="a first-round bye",
            )
        return season_threshold(
            w.state,
            scores,
            week_index=w.wi,
            team_index=w.t,
            field="made_playoffs",
            kind=ThresholdKind.PLAYOFF_CUT,
            label="a playoff berth",
        )
    assert opponent_score is not None
    return Threshold(
        kind=ThresholdKind.OPPONENT,
        score=opponent_score,
        decisive=np.ones(w.n_sims, dtype=bool),
        label="this week's matchup",
    )


@dataclass(frozen=True, slots=True)
class _Scored:
    """Every candidate lineup measured against one threshold."""

    win_prob: np.ndarray
    margin: np.ndarray
    sd_diff: np.ndarray
    covariance: np.ndarray
    sd_self: np.ndarray
    sd_threshold: float

    def sd_independent(self, index: int) -> float:
        return math.sqrt(self.sd_self[index] ** 2 + self.sd_threshold**2)


def _score_candidates(totals: np.ndarray, threshold: Threshold) -> _Scored:
    """`P(clear it)` and the moments of the difference, for every candidate at once.

    The probability is taken over *all* simulations, including the settled ones, because
    a simulation whose outcome is already secured really does contribute a win. The
    moments are taken over the decisive ones only, because a margin against `+inf` is not
    a number.
    """
    win = (totals > threshold.score[:, None]).mean(axis=0)
    n_lineups = totals.shape[1]
    if not threshold.live:
        zeros = np.zeros(n_lineups)
        return _Scored(win, zeros, zeros.copy(), zeros.copy(), totals.std(axis=0, ddof=1), 0.0)
    dec = threshold.decisive
    mine, theirs = totals[dec], threshold.score[dec]
    diff = mine - theirs[:, None]
    centred = mine - mine.mean(axis=0)
    cov = (centred * (theirs - theirs.mean())[:, None]).sum(axis=0) / (mine.shape[0] - 1)
    return _Scored(
        win_prob=win,
        margin=diff.mean(axis=0),
        sd_diff=diff.std(axis=0, ddof=1),
        covariance=cov,
        sd_self=mine.std(axis=0, ddof=1),
        sd_threshold=float(theirs.std(ddof=1)),
    )


def _legal_assignment(plan: LineupPlan, players: Sequence[int], n_local: int) -> dict[int, int]:
    """Slot id per player for a proposed set of starters, or {} if it is not legal.

    Solved with `sim/lineup.py`'s own solver rather than a hand-rolled first-fit, which
    can fail on a set that is perfectly legal: score 1 for the proposed players and
    `-inf` for everyone else, and a total short of the count means no legal assignment
    exists.
    """
    scores = np.full(n_local, -np.inf)
    for p in players:
        scores[p] = 1.0
    result = plan.solve(scores, assignment=True)
    assert result.assignment is not None
    if float(result.total) < len(players) - 1e-9:
        return {}
    return {
        int(p): int(s)
        for s, p in zip(result.slot_ids, result.assignment, strict=True)
        if int(p) >= 0
    }


def _row_for(slot_ids: Sequence[int], assignment: Mapping[int, int]) -> np.ndarray:
    """A lineup row in the enumeration's own slot-instance order."""
    by_slot: dict[int, list[int]] = {}
    for player, slot in assignment.items():
        by_slot.setdefault(slot, []).append(player)
    row = []
    for slot in slot_ids:
        bucket = by_slot.get(slot) or []
        row.append(bucket.pop() if bucket else -1)
    return np.asarray(row, dtype=np.intp)


def _paired_delta(
    w: _Week, baseline_total: np.ndarray, candidate_total: np.ndarray
) -> tuple[float, float, float]:
    """`(delta_title, its stderr, delta_playoffs)` from a paired season simulation.

    Two arms differing in exactly one team's score in exactly one week, against one
    drawn season. Everything else -- every other roster, every other week, the bracket
    -- is bit-identical, so the difference is the lineup and nothing else. It is still
    usually smaller than its own standard error, which is the honest answer: one week's
    start/sit is a small thing next to a bracket that is three coin flips.
    """
    base = w.base_scores.copy()
    base[:, w.wi, w.t] = baseline_total
    alt = w.base_scores.copy()
    alt[:, w.wi, w.t] = candidate_total
    a = S.simulate_from_scores(w.state, base, all_play=False)
    b = S.simulate_from_scores(w.state, alt, all_play=False)
    d_title = b.champions[:, w.t].astype(np.float64) - a.champions[:, w.t].astype(np.float64)
    d_playoffs = b.made_playoffs[:, w.t].astype(np.float64) - a.made_playoffs[:, w.t].astype(
        np.float64
    )
    return (
        float(d_title.mean()),
        float(d_title.std(ddof=1) / math.sqrt(d_title.size)),
        float(d_playoffs.mean()),
    )


def _paired_gain(hit: np.ndarray, a: int, b: int) -> tuple[float, float]:
    """`(mean, stderr)` of `P(clear it | a) - P(clear it | b)`, paired simulation by simulation.

    Paired rather than a difference of two marginal rates: the two lineups share seven of
    nine starters and meet the same football, so almost all of the variance cancels and
    the standard error is the honest one for the *difference*. Differencing two arms'
    marginal errors instead would report a resolution several times worse than the one
    actually available, which is the mirror image of the error this module guards
    against everywhere else.
    """
    paired = hit[:, a].astype(np.float64) - hit[:, b].astype(np.float64)
    if paired.size < 2:
        return float(paired.mean()) if paired.size else 0.0, 0.0
    return float(paired.mean()), float(paired.std(ddof=1) / math.sqrt(paired.size))


def _churn_guard(
    *,
    chosen: int,
    baseline_index: int,
    projected: np.ndarray,
    hit: np.ndarray,
    penalty: float,
) -> tuple[int, str]:
    """Refuse to move a lineup that is already set unless the move is worth something.

    Found on the user's own league: the currently set lineup and the projected-best
    lineup differed by two receivers projected within a hundredth of a point of each
    other, so `argmax` picked one, the surface told him to bench the other, and the
    paired season simulation scored the advice at -0.25pp of championship probability.
    Every part of that was working as designed and the answer was still wrong, because
    nothing in the pipeline had asked whether the change was worth making at all.

    So a change against a lineup the manager has already set has to clear something:
    `CHURN_TOLERANCE_POINTS` of projection, or a threshold-probability gain that is both
    materially large (`CHURN_TOLERANCE_PROB`) and outside its own selection-corrected
    noise. Materiality is not redundant with the noise floor and the twin-receiver case
    is why: two lineups that differ by five hundredths of a point straddle the threshold
    in one or two simulations out of four thousand, and *one simulation* clears a noise
    floor built from one simulation's worth of variance. Ties go to the lineup already in.
    """
    if chosen == baseline_index:
        return chosen, ""
    gain = float(projected[chosen] - projected[baseline_index])
    measured, stderr = _paired_gain(hit, chosen, baseline_index)
    floor = max(penalty * stderr, CHURN_TOLERANCE_PROB)
    if gain > CHURN_TOLERANCE_POINTS or measured > floor:
        return chosen, ""
    return baseline_index, (
        f"against the lineup already set it is worth {gain:+.2f} projected points and "
        f"{measured * 100:+.2f}pp, inside the {floor * 100:.2f}pp that would make it "
        "worth touching"
    )


def _option(
    w: _Week,
    row: np.ndarray,
    slot_ids: Sequence[int],
    projected: float,
    scored: _Scored,
    index: int,
) -> LineupOption:
    started = [(int(s), int(p)) for s, p in zip(slot_ids, row, strict=True) if p >= 0]
    return LineupOption(
        slot_ids=tuple(s for s, _ in started),
        player_ids=tuple(int(w.franchise.player_ids[p]) for _, p in started),
        names=tuple(w.name(p) for _, p in started),
        mean=float(projected),
        sd=float(scored.sd_self[index]),
        margin=float(scored.margin[index]),
        sd_diff=float(scored.sd_diff[index]),
        covariance=float(scored.covariance[index]),
        win_prob=float(scored.win_prob[index]),
    )


def _swaps(w: _Week, baseline: LineupOption, recommended: LineupOption) -> tuple[Swap, ...]:
    """Pair what left the lineup with what came in, **by slot**.

    The obvious pairing -- rank both sides by projection and zip -- produces sentences a
    manager cannot act on. Measured on the user's own league: 31 of 167 swap sentences
    read "start <quarterback> over <receiver>" because the two sides happened to sort
    that way, and no ESPN lineup screen will let him do that. The instruction has to name
    the seat: whoever is being asked to leave must be someone who could have held the
    slot the incoming player is taking.

    So the incoming player is matched to the man in his own slot first, then to anyone
    whose slot he could legally have filled, and only then to whoever is left. Ties and
    leftovers still fall back to projection order, which is what keeps a pure reshuffle
    reading in the order a human would say it. With more than one change the assignment
    remains an approximation -- the lineup is what it is -- but never an impossible one.
    """
    means = dict(zip(w.franchise.player_ids, w.means, strict=True))
    sds = dict(zip(w.franchise.player_ids, w.sds, strict=True))
    positions = dict(zip(w.franchise.player_ids, (int(p) for p in w.positions), strict=True))
    eligibility = w.state.slot_eligibility
    incoming = set(recommended.player_ids)
    outgoing = set(baseline.player_ids)
    out = sorted((p for p in baseline.player_ids if p not in incoming), key=lambda p: -means[p])
    coming = sorted(
        (p for p in recommended.player_ids if p not in outgoing), key=lambda p: -means[p]
    )
    slot_of = dict(zip(recommended.player_ids, recommended.slot_ids, strict=True))
    was_in = dict(zip(baseline.player_ids, baseline.slot_ids, strict=True))
    names = dict(zip(baseline.player_ids, baseline.names, strict=True))
    names.update(dict(zip(recommended.player_ids, recommended.names, strict=True)))

    swaps: list[Swap] = []
    available = list(out)
    for i in coming:
        if not available:
            break
        slot = slot_of[i]
        here = eligibility.get(slot, frozenset())
        # The reported slot is the seat that actually changes hands, which is not always
        # the one the incoming player ends up in. A flex rotation -- a back takes the RB
        # slot, the incumbent back slides to the flex, and the receiver who was in the
        # flex is the one who loses his place -- is named by the flex, because that is
        # the seat the manager empties.
        o = next((p for p in available if was_in[p] == slot), None)
        if o is None:
            o = next((p for p in available if positions[p] in here), None)
        if o is None:
            o = next(
                (p for p in available if positions[i] in eligibility.get(was_in[p], frozenset())),
                None,
            )
            if o is not None:
                slot = was_in[o]
        if o is None:
            o = available[0]
        available.remove(o)
        swaps.append(
            Swap(
                slot_id=slot,
                out_player_id=o,
                out_name=names[o],
                in_player_id=i,
                in_name=names[i],
                d_mean=float(means[i] - means[o]),
                d_sd=float(sds[i] - sds[o]),
            )
        )
    return tuple(swaps)


def _stacks(
    state: S.LeagueState,
    mine: Sequence[int],
    opponent: Sequence[int],
    correlation: CorrelationModel,
) -> tuple[StackEdge, ...]:
    """My starters who share an NFL team with one of my opponent's starters.

    Same-team only, because that is the only structure the measured constants have. A
    pairing the table has no entry for is still reported, with `modelled=False`, so the
    output distinguishes "measured at zero" from "we have no number for this".
    """
    pool = state.pool
    index = pool.index
    edges: list[StackEdge] = []
    for pid in mine:
        i = index[pid]
        team = pool.pro_team_ids[i]
        if team <= 0:
            continue
        for opp in opponent:
            j = index[opp]
            if pool.pro_team_ids[j] != team:
                continue
            a, b = pool.position_ids[i], pool.position_ids[j]
            rho = correlation.rho(a, b)
            known = a in SKILL_POSITIONS and b in SKILL_POSITIONS
            if rho == 0.0 and known:
                continue
            edges.append(
                StackEdge(
                    player_id=pid,
                    name=pool.name(pid),
                    position_id=a,
                    pro_team_id=int(team),
                    opponent_player_id=opp,
                    opponent_name=pool.name(opp),
                    opponent_position_id=b,
                    rho=rho,
                    modelled=known,
                )
            )
    return tuple(sorted(edges, key=lambda e: -abs(e.rho)))


def _confidence(significant: bool, lev: float) -> str:
    if lev < 0.15:
        return "low"
    return "high" if significant else "medium"


def _rationale(
    *,
    threshold: Threshold,
    opponent: OpponentView | None,
    z: float,
    margin: float,
    sd_diff: float,
    lev: float,
    swaps: tuple[Swap, ...],
    recommended: LineupOption,
    changed: bool,
    differ: bool,
    baseline_is_points: bool,
    guard: str,
    delta_win: float,
    delta_win_se: float,
    sacrifice: float,
    ep_sd_diff: float,
    wp_sd_diff: float,
    delta_title: float,
    stderr: float,
    significant: bool,
    unpriced: int = 0,
) -> str:
    """One paragraph a manager can act on without reading the rest of the object."""
    said: list[str] = []
    if unpriced:
        # This has to come first and it has to be an instruction, not a footnote. The
        # alternative -- and what this module did before -- was "nothing to change" on a
        # lineup that cannot legally be fielded, which is the most expensive sentence a
        # start/sit surface can print.
        said.append(
            f"WARNING: the {unpriced} starters you have set cannot be legally assigned to "
            "this week's slots (a bye, a player who is out, or an ineligible slot), so "
            "your lineup was not priced -- fix it first. Everything below is measured "
            "against the projected-best lineup, not against what you have in."
        )
    if unpriced:
        said.append(f"Set the projected-best lineup: {recommended.describe()}.")
    elif changed and swaps:
        said.append("; ".join(s.describe() for s in swaps).capitalize() + ".")
    elif changed:
        said.append("Reshuffle the slots; the same players start.")
    elif baseline_is_points:
        said.append("Start the projected-best lineup; nothing to change.")
    else:
        said.append("Leave the lineup exactly as you have it set.")

    side = "favorite" if z >= 0 else "underdog"
    if threshold.kind is ThresholdKind.OPPONENT and opponent is not None:
        said.append(
            f"You are a {abs(z):.2f}-sigma {side} against {opponent.name} "
            f"({margin:+.1f} +/- {sd_diff:.1f} points)."
        )
    elif threshold.live:
        need = margin
        said.append(
            f"Priced against {threshold.label} rather than the matchup: "
            f"{abs(need):.1f} points {'clear of' if need >= 0 else 'short of'} the line "
            f"({z:+.2f} sigma), decisive in {threshold.decisive_share * 100:.0f}% of "
            "simulations."
        )
    else:
        said.append(
            f"{threshold.label.capitalize()} is already settled in every simulation -- "
            "no lineup this week changes your season."
        )

    if lev < 0.15:
        said.append(
            f"Leverage {lev:.2f}: a point is worth about "
            f"{(1.0 / lev if lev > 0 else float('inf')):.0f}x less than in a coin flip, "
            "so this decision does not matter much."
        )
    else:
        said.append(f"Leverage {lev:.2f} against an even matchup.")

    if guard and differ:
        said.append(
            f"The win-probability lineup would move {threshold.label} probability "
            f"{delta_win * 100:+.2f}pp (+/- {delta_win_se * 100:.2f}) for "
            f"{sacrifice:.1f} projected points, but {guard}."
        )
    elif guard:
        said.append(f"A different lineup projects higher, but {guard}.")
    elif abs(delta_win) > 0:
        direction = "more" if wp_sd_diff > ep_sd_diff else "less"
        said.append(
            f"That buys {delta_win * 100:+.2f}pp (+/- {delta_win_se * 100:.2f}) of "
            f"{threshold.label} probability for {sacrifice:.1f} projected points, by "
            f"taking {direction} spread (sd_diff {ep_sd_diff:.1f} -> {wp_sd_diff:.1f})."
        )

    tail = f"Worth {delta_title * 100:+.3f}pp of championship probability"
    tail += (
        f" (+/- {stderr * 100:.3f}pp)."
        if significant
        else f", inside its own {stderr * 100:.3f}pp Monte Carlo error."
    )
    said.append(tail)
    return " ".join(said)


def advise(
    state: S.LeagueState,
    draw: Draw,
    *,
    week: int | None = None,
    team_id: int | None = None,
    opponent_starters: Sequence[int] | None = None,
    current: Sequence[int] | None = None,
    efficiency: S.LineupEfficiency | None = None,
    correlation: CorrelationModel | None = None,
    extra_depth: int = DEFAULT_EXTRA_DEPTH,
    max_lineups: int = MAX_LINEUPS,
    cut_line_window: int = CUT_LINE_WINDOW,
) -> LineupAdvice:
    """The start/sit call for one team in one week, priced in win probability.

    `week` defaults to the next unplayed one and `team_id` to the user's own team.
    `opponent_starters` is the lineup the opponent has actually set -- knowable in a
    season-long league, which is what makes the correlation term real rather than
    assumed; without it they are given their projection-optimal lineup. `current` is this
    team's own starters right now: supply it and the recommendation's deltas are measured
    against what is in the lineup today rather than against a hypothetical.

    The efficiency default is `LineupEfficiency.symmetric()` rather than
    `sim/season.py`'s own, because a start/sit surface that quietly haircuts the opponent
    is answering an easier question with a better-looking number.
    """
    week = int(state.weeks[0]) if week is None else int(week)
    resolved = state.my_team_id if team_id is None else int(team_id)
    if resolved is None:
        raise StartSitError("no team to advise: pass team_id, or set my_team_id on the state")
    if week not in state.week_index:
        raise StartSitError(f"week {week} is not on the simulated axis {list(state.weeks)}")
    efficiency = efficiency if efficiency is not None else S.LineupEfficiency.symmetric()
    correlation = correlation if correlation is not None else CorrelationModel()

    w = _prepare(state, draw, week=week, team_id=resolved, efficiency=efficiency)
    groups = _slot_groups(state, w.positions, w.startable)
    keys = (w.means, w.means + w.sds, w.means - w.sds)
    lineups, slot_ids, _ = enumerate_lineups(groups, keys, extra=extra_depth, cap=max_lineups)
    plan = plan_from_slots(state.lineup_slot_counts, state.slot_eligibility, w.positions)

    lineups, baseline_index = _with_current(w, plan, lineups, slot_ids, current)
    # A supplied lineup that could not be legally assigned is the one case where the
    # answer must not read as "nothing to change": the deltas below are then measured
    # against a lineup the manager has NOT set, and his own is broken. Carried through
    # rather than swallowed by the log line inside `_with_current`.
    unpriced_current = (
        () if current is None or baseline_index is not None else tuple(int(p) for p in current)
    )
    indicator = _indicator(lineups, w.n_players)
    totals = w.points @ indicator
    projected = w.means @ indicator

    optimum = float(plan.solve(np.where(w.startable, w.means, -np.inf)).total)
    if projected.max() < optimum - 1e-6:
        log.warning(
            "the enumerated best projected lineup is %.2f against the solver's %.2f; the "
            "candidate pool is too shallow to contain the points-optimal lineup",
            float(projected.max()),
            optimum,
        )

    ep = int(np.argmax(projected))
    scores_ep = w.base_scores.copy()
    scores_ep[:, w.wi, w.t] = totals[:, ep]

    game = _game_for(state, week, resolved)
    opponent_view: OpponentView | None = None
    opponent_score: np.ndarray | None = None
    if game is not None:
        opponent_view, opponent_score = _opponent(w, game, opponent_starters)
    threshold = _pick_threshold(
        w,
        scores=scores_ep,
        game=game,
        opponent_score=opponent_score,
        cut_line_window=cut_line_window,
    )
    scored = _score_candidates(totals, threshold)

    # Win probability first, projected points as the tiebreak: at a few thousand
    # simulations a great many candidates tie on probability, and picking the highest
    # projection among them is both the right default and what makes the answer stable.
    wp = int(np.lexsort((-projected, -scored.win_prob))[0])
    hit = totals > threshold.score[:, None]
    paired = hit[:, wp].astype(np.float64) - hit[:, ep].astype(np.float64)
    delta_win = float(paired.mean())
    delta_win_se = float(paired.std(ddof=1) / math.sqrt(paired.size)) if paired.size > 1 else 0.0

    z = float(scored.margin[ep] / scored.sd_diff[ep]) if scored.sd_diff[ep] > 0 else 0.0
    sacrifice = float(projected[ep] - projected[wp])
    d_sd = float(scored.sd_diff[wp] - scored.sd_diff[ep])
    penalty = selection_penalty(int(lineups.shape[0]))
    noise_floor = MIN_EDGE_SIGMA * penalty * delta_win_se
    guard = ""
    if wp != ep:
        if not threshold.live:
            guard = "the outcome is already settled in every simulation"
        elif abs(z) <= OVERRIDE_Z:
            guard = (
                f"|z| = {abs(z):.2f} is inside the {OVERRIDE_Z:.2f} guard, where "
                "projection error dominates any variance edge"
            )
        elif sacrifice > MAX_POINTS_SACRIFICE:
            guard = (
                f"it costs {sacrifice:.1f} projected points, past the "
                f"{MAX_POINTS_SACRIFICE:.1f}-point limit"
            )
        elif delta_win <= noise_floor:
            guard = (
                f"the gain is inside the {noise_floor * 100:.2f}pp noise floor "
                f"({delta_win_se * 100:.2f}pp of standard error, inflated "
                f"{penalty:.1f}x for taking the best of {lineups.shape[0]} candidates), "
                "so it is not measurably better"
            )
    chosen = ep if (wp == ep or guard) else wp
    baseline_index = ep if baseline_index is None else baseline_index
    churn = ""
    if chosen == ep:
        # Only a points-driven change is asked to justify itself here. A deliberate
        # variance override has already cleared three guards written for it, and the
        # materiality floor is denominated in threshold probability -- half a point of
        # *weekly win* probability is nothing and half a point of *championship*
        # probability is a large recommendation, so applying one number to both would
        # silently kill the bracket-week overrides.
        chosen, churn = _churn_guard(
            chosen=chosen,
            baseline_index=baseline_index,
            projected=projected,
            hit=hit,
            penalty=penalty,
        )
    elif baseline_index not in (ep, chosen):
        # Every guard above was measured against the *points* lineup, because that is the
        # reference the override argument is made in. What the manager would actually be
        # asked to do is move off the lineup he has already set, and that is a different
        # comparison: if his lineup already carries most of the spread the override buys,
        # the change is worth nothing even though the override cleared its own floor.
        # So the gain is re-measured against the baseline, paired, against the same
        # selection-corrected floor. Strictly more conservative -- it can only send the
        # answer back to the lineup already in.
        gain_b, se_b = _paired_gain(hit, chosen, baseline_index)
        floor_b = MIN_EDGE_SIGMA * penalty * se_b
        if gain_b <= floor_b:
            chosen = baseline_index
            churn = (
                f"against the lineup already set the override is worth {gain_b * 100:+.2f}pp, "
                f"inside the {floor_b * 100:.2f}pp that would make it measurable"
            )
    guard = "; ".join(reason for reason in (guard, churn) if reason)

    delta_title, stderr, _ = _paired_delta(w, totals[:, baseline_index], totals[:, chosen])
    lev = float(leverage(scored.margin[ep], scored.sd_diff[ep]) * threshold.decisive_share)

    points_lineup = _option(w, lineups[ep], slot_ids, projected[ep], scored, ep)
    win_prob_lineup = _option(w, lineups[wp], slot_ids, projected[wp], scored, wp)

    # Three candidate rows are in play -- the points lineup, the win-probability lineup
    # and whatever is already set -- and the churn guard can send the answer back to any
    # of them, so `chosen` is resolved against all three rather than assumed to be one of
    # the first two.
    def option_for(index: int) -> LineupOption:
        if index == ep:
            return points_lineup
        if index == wp:
            return win_prob_lineup
        return _option(w, lineups[index], slot_ids, projected[index], scored, index)

    baseline = option_for(baseline_index)
    recommended = option_for(chosen)
    swaps = _swaps(w, baseline, recommended)
    changed = set(recommended.player_ids) != set(baseline.player_ids)
    # The baseline is the projected-best lineup when the supplied one could not be
    # priced, so `changed` against it says nothing about whether the manager has to act.
    # He does: the players he has in are not the players being recommended.
    if unpriced_current and set(unpriced_current) != set(recommended.player_ids):
        changed = True
    move = Move(
        kind=MoveKind.LINEUP if changed else MoveKind.HOLD,
        league_id=state.league_id,
        lineup=recommended.slot_map(),
    )
    significant = abs(delta_title) > 2.0 * stderr if stderr > 0 else False
    tags = [
        "start_sit",
        threshold.kind.value,
        "favorite" if z >= 0 else "underdog",
        "override" if chosen == wp and wp != ep else ("guarded" if guard else "no_change"),
    ]
    if lev < 0.15:
        tags.append("low_leverage")
    if not significant:
        tags.append("not_significant")
    if not state.playoff_rounds:
        tags.append("no_bracket")
    if unpriced_current:
        tags.append("unpriced_current")
    recommendation = Recommendation(
        move=move,
        delta_title=delta_title,
        delta_points=float(projected[chosen] - projected[baseline_index]),
        stderr=stderr,
        leverage=lev,
        rationale=_rationale(
            threshold=threshold,
            opponent=opponent_view,
            z=z,
            margin=float(scored.margin[ep]),
            sd_diff=float(scored.sd_diff[ep]),
            lev=lev,
            swaps=swaps,
            recommended=recommended,
            changed=changed,
            differ=wp != ep,
            baseline_is_points=baseline_index == ep,
            guard=guard,
            delta_win=delta_win,
            delta_win_se=delta_win_se,
            sacrifice=sacrifice,
            ep_sd_diff=float(scored.sd_diff[ep]),
            wp_sd_diff=float(scored.sd_diff[wp]),
            delta_title=delta_title,
            stderr=stderr,
            significant=significant,
            unpriced=len(unpriced_current),
        ),
        confidence=_confidence(significant, lev),
        tags=tuple(tags),
    )
    return LineupAdvice(
        league_id=state.league_id,
        season=state.season,
        team_id=resolved,
        team_name=w.franchise.name,
        week=week,
        kind=threshold.kind,
        target=threshold.label,
        margin=float(scored.margin[ep]),
        sd_diff=float(scored.sd_diff[ep]),
        z=z,
        leverage=lev,
        decisive_share=threshold.decisive_share,
        points_lineup=points_lineup,
        win_prob_lineup=win_prob_lineup,
        recommended=recommended,
        baseline=baseline,
        swaps=swaps,
        delta_win_prob=delta_win,
        delta_win_prob_stderr=delta_win_se,
        noise_floor=noise_floor,
        points_sacrifice=max(sacrifice, 0.0),
        guard=guard,
        rule_agrees=swap_improves_win_probability(
            -sacrifice, d_sd, float(scored.margin[ep]), float(scored.sd_diff[ep])
        )
        == (delta_win > 0.0),
        opponent=opponent_view,
        stacks=(
            _stacks(state, recommended.player_ids, opponent_view.starters, correlation)
            if opponent_view is not None
            else ()
        ),
        sd_independent=scored.sd_independent(ep),
        recommendation=recommendation,
        n_sims=w.n_sims,
        n_lineups=int(lineups.shape[0]),
        unpriced_current=unpriced_current,
    )


def _with_current(
    w: _Week,
    plan: LineupPlan,
    lineups: np.ndarray,
    slot_ids: Sequence[int],
    current: Sequence[int] | None,
) -> tuple[np.ndarray, int | None]:
    """Append the lineup that is actually set, so it is scored on the same footing.

    Returns the candidate matrix and the row the current lineup landed on, or `None`
    when there is nothing legal to price against -- a manager with a player on bye in
    his lineup is a different (and more urgent) problem, and it is reported by falling
    back to the projected-best baseline rather than by raising.
    """
    if current is None:
        return lineups, None
    index = {int(p): i for i, p in enumerate(w.franchise.player_ids)}
    missing = [p for p in current if int(p) not in index]
    if missing:
        raise StartSitError(f"players {missing} are not on team {w.team_id}")
    locals_ = [index[int(p)] for p in current]
    assignment: dict[int, int] = {}
    if all(bool(w.startable[p]) for p in locals_):
        assignment = _legal_assignment(plan, locals_, w.n_players)
    if not assignment:
        log.info(
            "team %s's currently set lineup is not legally startable this week (a bye, "
            "an out player, or an illegal slot); pricing against the projected-best "
            "lineup instead",
            w.team_id,
        )
        return lineups, None
    row = _row_for(slot_ids, assignment)
    key = tuple(sorted(int(p) for p in row if p >= 0))
    for i, existing in enumerate(lineups):
        if tuple(sorted(int(p) for p in existing if p >= 0)) == key:
            return lineups, i
    return np.vstack([lineups, row[None, :]]), int(lineups.shape[0])


def advise_sim(sim: HasSim, **kwargs: object) -> LineupAdvice:
    """`advise` against a `pipeline.LeagueSim`, which already holds the state and draw."""
    return advise(sim.state, sim.draw, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "CHURN_TOLERANCE_POINTS",
    "CHURN_TOLERANCE_PROB",
    "CUT_LINE_WINDOW",
    "MAX_POINTS_SACRIFICE",
    "MIN_EDGE_SIGMA",
    "OVERRIDE_Z",
    "LineupAdvice",
    "LineupOption",
    "OpponentView",
    "StackEdge",
    "StartSitError",
    "Swap",
    "Threshold",
    "ThresholdKind",
    "advise",
    "advise_sim",
    "enumerate_lineups",
    "season_threshold",
    "selection_penalty",
    "starters_from_roster",
]
