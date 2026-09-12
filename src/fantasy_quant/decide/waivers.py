"""The waiver wire: what to claim, what it costs, and when priority is worth more.

Three of the user's leagues run **rolling waiver priority**, none runs FAAB, so the
currency is a queue position rather than money and the correct model is optimal
stopping rather than auction theory. That reordering is the point of this module: the
priority path is the developed one, and the FAAB path is a compact secondary kept
because the registry supports leagues we have not seen yet.

**A claim is an add AND a drop, never the add alone.**

    delta(j, k) = E[sum_w S(R + j - k, w)] - E[sum_w S(R, w)]

where `S` is the *optimal starting lineup* score from `sim/lineup.py` and the roster is
otherwise unchanged. Three consequences fall out of that definition, and a ranking list
-- which is what every public waiver tool is -- structurally cannot express any of them:

* **Submodular in adds.** A fourth good receiver is worth far less than the first,
  because the lineup only starts so many. `test_waivers.py` measures the decay.
* **`E[max] > max[E]`, so a bench add is a weekly exchange option.** Under point
  projections a player who never out-projects a starter adds exactly zero; under
  simulation he adds the weeks he happens to beat one. Bench depth having literally no
  value under point projections is the strongest single argument for simulating at all,
  and it is why claims here are priced off the drawn tensor rather than off means.
* **Blocking a rival is worth about `1/(N-1)` of his gain.** Title probability sums to
  one, so a rival's gain of `x` is a loss of `x` spread over the other `N-1` teams and
  our own share is an order of magnitude below owning the player. `blocking_value` is
  the only route to a blocking number here, and the board tags them so they cannot be
  mistaken for a real claim.

**Priority is a depreciating asset.** Spending it now forfeits the option to spend it
later, so a claim is worth making only when it beats the continuation value of holding:

    claim j  iff  delta_j >= C_t(p),   C_t(p) = U_{t+1}(p) - U_{t+1}(N)

with `C_t(N) = 0` -- the back of the queue costs nothing to spend -- and `C_T(p) = 0` --
priority has no salvage after the last waiver run. `continuation_values` solves that by
backward induction over (week, priority) against a per-week distribution of what the
wire offers, estimated from this league's own free-agent pool and its own transaction
history. `C` falls through the season and rises with the right-tail heaviness of what
shows up.

The win probability drops out of the threshold, which is not obvious and is worth
stating: winning moves you to `N` and losing leaves you where you were, so the
comparison is `w(p)*(v + U(N)) + (1-w(p))*U(p)` against `U(p)`, and `w(p)` cancels.
**A losing claim is therefore free**, and submitting the full conditional waterfall
every week -- every candidate above the threshold, in order -- is weakly dominant. The
report says so in its output rather than leaving it implied.

**The exploitable window is Sunday night to Tuesday night.** Claims are blind, and snap
counts and route participation post Monday and Tuesday, so information arrives after
most of a league has already submitted. ESPN processes around 3-4am ET Wednesday.
Submitting as late as possible is free option value; it is surfaced in the rationale
rather than buried in a docstring.

**Which of two estimates of `delta_title` gets published.** A claim's title value can be
had two ways: re-run the bracket on the modified roster and diff the champion indicator,
or take the paired points gain through the franchise's own measured points-to-title rate.
They estimate the same number and the first one cannot see it. Measured on the user's
leagues, a week-1 claim is worth 0.02 to 0.17pp while the paired bracket difference
carries a standard error near 0.1-0.3pp at 4,000 simulations -- so ranking a dozen
candidates on the bracket ranks the noise, and the leader's estimate is biased upward by
having been selected on it. Measured: routing the confirm tier through an engine that
publishes the bracket ranked a +2.4-point add fifth behind a +1.0-point one, called a
points-POSITIVE add -0.20pp and significant, and published a top claim at +0.90pp where
the points say +0.14pp. `sim/season.py` says the same thing in its own docstring: resolving a
0.4pp title edge takes on the order of 10,000 simulations. So the published estimate is
the plug-in, with the rate's own error propagated into `stderr`, and `confirm` keeps the
bracket difference beside it as a consistency check (`ClaimPrice.agrees`). The rate is
measured per franchise by central difference, because a contender and a team already out
of it convert the same point into very different amounts of title probability: 0.056,
0.079 and 0.090 pp per rest-of-season point on the three real leagues.

**What the floor is, and why it is not `decide/valuation.py`'s.** Dropping a kicker for
a receiver does not cost the kicker's whole score -- it costs the difference between him
and the kicker still sitting on the wire. So every lineup here is solved against a
per-slot free-agent floor. `valuation.replacement_levels` answers the same question from
the league-wide demand side; this module measures it off the players this league has
actually left unrostered, which is cheaper and closer to what a claim can really fall
back on. The floor deliberately reads the **second**-best unrostered player at a slot
rather than the best: the best one is usually the player the board is recommending, and
you cannot both claim him and stream him. It reads that second-best **week by week**
rather than once for the season -- at D/ST and kicker the identity of the best available
body changes every week, and pinning one player's season average understates the floor
by more than a point a week on all three real leagues, which is enough on its own to
manufacture a board full of "add a second defense". Pass `replacement=` to override it
with the valuation model's levels.

Everything comes back as a `core.Recommendation`, so a claim in Wine Wednesday and a
claim in Blacksburg are comparable in the only unit that matters.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ..core import (
    Move,
    MoveEvaluator,
    MoveKind,
    PlayerMove,
    PlayerOutlook,
    Recommendation,
    WeeklyOutlook,
    WireLevel,
)
from ..core import leverage as _leverage
from ..data.etr import EtrRankings
from ..sim import season as S
from ..sim.distributions import Draw, WeeklySampler
from ..sim.lineup import plan_from_slots
from .opinion import Upgrade, bench_upgrades, tilt_outlooks
from .valuation import POSITION_ABBREV
from .wire import DEFAULT_WIRE_DEPTH as _DEFAULT_WIRE_DEPTH
from .wire import all_rostered as _wire_all_rostered
from .wire import wire_floor as _wire_floor
from .wire import wire_levels

if TYPE_CHECKING:  # pragma: no cover - only the type checker needs these
    from ..espn.league import Availability, TransactionLog
    from ..pipeline import LeagueSim

log = logging.getLogger(__name__)

#: Measured SD of (my weekly score - my opponent's), 23,999 paired player-weeks. Used
#: only as the fallback when there is no tensor to measure it from; the board measures
#: its own per matchup and reports it next to this.
SD_DIFF = 34.4

#: Per-team-week probability that a rival contests the same claim. A prior until the
#: league's own log says otherwise (`contest_rate_from_log`). It cancels out of the claim
#: threshold and only scales the level of the value function, so a wrong value here
#: cannot flip a recommendation.
DEFAULT_CONTEST_RATE = 0.15

#: Which unrostered player at a slot the floor reads. 1 would be the best free agent --
#: usually the player being recommended, so the claim would be priced against itself. 2
#: is the honest "what could I still stream if I left this slot open".
DEFAULT_WIRE_DEPTH = _DEFAULT_WIRE_DEPTH

#: Free agents carried into the simulated pool. The tensor is `(sims, weeks, players)`,
#: so this is the memory knob: 60 adds about a quarter to a 14-team league's pool.
DEFAULT_CANDIDATES = 60

#: How far an outside ranking set moves this board when one is supplied. 1.0 adopts
#: the analyst's within-position ordering outright.
#:
#: **This is the one input in the repo that ships without a measured verdict, and the
#: user chose that deliberately.** Calibration got held-out bias, the ensemble's equal
#: weights got twelve seasons of head-to-heads, the analyst-dispersion screen got
#: +0.187 at p=0.004. This gets nothing, because Establish The Run overwrites each
#: chart in place and publishes no history, so there is no back-test to run: the first
#: board we hold is the one we are using. `data.etr.archive` starts the record that
#: makes a verdict possible in a few weeks. Until then `board_weight` is the single
#: constant that reverses the whole decision, and `0.0` is byte-identical to not
#: having the board at all.
DEFAULT_BOARD_WEIGHT = 1.0

#: Only mention a dropped player's hindsight ceiling when it exceeds what the board
#: charges by more than this, in rest-of-season points. Mirrors
#: `decide/trades._CEILING_WORTH_SAYING`: every bench player has some gap, so without a
#: floor the note appears on every row and carries no information. At 5.0 it fired on
#: 28 of 28 live rows, which is the same thing as not firing; at 10.0 it marks the
#: stashes whose ceiling is genuinely a startable week or more.
CEILING_WORTH_SAYING = 10.0

#: Simulations for the screen. Under common random numbers a points-for difference
#: carries ~1000x the variance reduction of a title difference, so a few hundred paired
#: simulations resolve `delta_points` far more sharply than 4,000 resolve `delta_title`.
DEFAULT_SCREEN_SIMS = 600

#: Points a week the exchange-rate probe shifts a franchise by, in each direction.
#: Roughly 0.13 of the measured 34.4-point matchup spread: large enough that the title
#: response clears the Monte Carlo floor, small enough that the response is still linear.
#: A one-point probe measures noise -- see `RosterSimulator.build`.
DEFAULT_PROBE = 4.0

#: Played weeks of transaction log needed before it is preferred to the priors. One
#: week of a fresh season is not a sample: every league has made nearly no waiver claims
#: by week 1, and reading a contest rate off that clamps it to the floor and reports
#: waiver priority as almost worthless. Measured: it moved Wine Wednesday's week-1
#: threshold from 0.250pp to 0.171pp on no evidence at all.
MIN_LOG_WEEKS = 3

#: Fallback timing advice for a CLAIM when the league's own acquisition settings cannot
#: be read. The edge is real -- claims are blind and information accrues right up to the
#: deadline -- but the schedule in it is the ESPN default, not this league's.
#:
#: It used to be stamped on every row unconditionally, including on players who cost
#: nothing and are first-come, where "wait until Tuesday" is the opposite of the right
#: advice. And its schedule is wrong for all three of the user's leagues: every one of
#: them processes on six days a week at hour 11 with a 24-hour waiver period, and
#: Tuesday is the one day none of them process. Prefer `timing_note`.
TIMING_NOTE = (
    "Submit as late as the deadline allows: claims are blind, and snap counts and route "
    "participation post Monday/Tuesday."
)

#: What a first-come free agent's rationale ends with. The opposite advice, for the
#: opposite situation: nothing is spent, nobody is bidding blind, and the only risk in
#: waiting is that somebody else clicks first.
FREE_AGENT_TIMING_NOTE = (
    "This one is first-come and costs no waiver priority, so there is nothing to wait "
    "for -- add now. Waiting only gives a rival the chance to take him."
)

#: Week order, so a league's process days read as a schedule rather than a set.
_DAY_ORDER = (
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
)


def timing_note(acquisition: object | None, *, on_waivers: bool) -> str:
    """When to act, from the league's own settings rather than from a folk rule.

    `waiverHours`, `waiverProcessDays` and `waiverProcessHour` have been parsed into
    `espn.league.AcquisitionConfig` all along and read by nobody, while every row of the
    board carried a hardcoded "ESPN processes around 3-4am ET Wednesday". Measured on the
    user's three leagues, that is wrong in both halves: all three run a 24-hour waiver
    period and process on six days at hour 11, and **Tuesday is the one day none of them
    process**. A reader who followed the old sentence would submit into a day the league
    skips.

    A free agent gets the opposite advice, because he is a different kind of decision.
    """
    if not on_waivers:
        return FREE_AGENT_TIMING_NOTE
    days = tuple(getattr(acquisition, "waiver_process_days", ()) or ())
    hour = int(getattr(acquisition, "waiver_process_hour", 0) or 0)
    hours = int(getattr(acquisition, "waiver_hours", 0) or 0)
    if not days:
        return TIMING_NOTE
    ordered = [d for d in _DAY_ORDER if d in {str(x).upper() for x in days}]
    listed = ", ".join(d.capitalize() for d in ordered) if ordered else "an unlisted day"
    period = f" A drop sits on waivers for {hours}h before the next run." if hours else ""
    return (
        f"Claims are blind and information accrues until they run, so submit as late as "
        f"you can: this league processes on {listed} at {hour:02d}:00 "
        f"(ESPN's `waiverProcessHour`, its own clock).{period}"
    )


class WaiverError(ValueError):
    """The league or roster as given cannot support a waiver evaluation."""


# --------------------------------------------------------------------------------------
# The free-agent pool
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FreeAgent:
    """One unrostered player, already scored for THIS league's remaining weeks."""

    player_id: int
    name: str
    position_id: int
    pro_team_id: int
    #: Projected points summed over the weeks still to be played.
    ros_points: float
    #: Projected points in the first remaining week -- the streaming question.
    next_points: float
    #: `ros_points` above the wire's own depth at this position. The screen orders on
    #: this rather than on raw points, or every board is quarterbacks.
    above_wire: float = 0.0
    #: Whether adding him costs a waiver claim. `True` is the SAFE default: an unknown
    #: availability must not turn into a board full of "add him now, it's free". See
    #: `waiver_board`, and `espn.league.Availability` for where the truth comes from.
    on_waivers: bool = True

    @property
    def position(self) -> str:
        return POSITION_ABBREV.get(self.position_id, str(self.position_id))

    @property
    def cost(self) -> str:
        return "waiver priority" if self.on_waivers else "free"


def _weekly_means(outlook: PlayerOutlook, weeks: Sequence[int]) -> list[float]:
    return [outlook.weeks[w].mean for w in weeks if w in outlook.weeks]


def free_agent_pool(
    outlooks: Sequence[PlayerOutlook],
    rostered: Iterable[int],
    weeks: Sequence[int],
    *,
    limit: int = DEFAULT_CANDIDATES,
    wire_depth: int = DEFAULT_WIRE_DEPTH,
    positions: Iterable[int] | None = None,
    on_waivers: Iterable[int] | None = None,
) -> tuple[FreeAgent, ...]:
    """Every unrostered player with a pulse, best first.

    "Best" is points above the *wire's own* depth at that position, not raw points. A
    league's twelfth-best available quarterback still projects for more points than its
    best available running back and is worth nothing, because the quarterback slot can
    be refilled for free and the running back slot cannot. Ordering on raw points gives
    a board that is all quarterbacks, and is the standard way this surface goes wrong.

    `rostered` is what makes a player available at all; `on_waivers` is what makes him
    cost something. They are different questions and this function used to answer only
    the first, which is how 809 first-come free agents came to be charged the price of a
    waiver claim. `None` means "we could not find out", and everything unrostered is then
    marked as costing a claim -- the expensive assumption, deliberately.
    """
    owned = {int(p) for p in rostered}
    claimed = None if on_waivers is None else {int(p) for p in on_waivers}
    wanted = None if positions is None else {int(p) for p in positions}
    rows: list[FreeAgent] = []
    for o in outlooks:
        if o.player_id in owned:
            continue
        if wanted is not None and o.position_id not in wanted:
            continue
        means = _weekly_means(o, weeks)
        if not means or sum(means) <= 0.0:
            continue
        rows.append(
            FreeAgent(
                player_id=o.player_id,
                name=o.name,
                position_id=o.position_id,
                pro_team_id=o.pro_team_id,
                ros_points=float(sum(means)),
                next_points=float(means[0]),
                on_waivers=True if claimed is None else o.player_id in claimed,
            )
        )
    by_position: dict[int, list[float]] = {}
    for r in rows:
        by_position.setdefault(r.position_id, []).append(r.ros_points)
    depth: dict[int, float] = {}
    for pos, values in by_position.items():
        values.sort(reverse=True)
        depth[pos] = values[min(max(wire_depth, 1), len(values)) - 1]
    scored = [replace(r, above_wire=r.ros_points - depth[r.position_id]) for r in rows]
    scored.sort(key=lambda r: (-r.above_wire, -r.ros_points, r.player_id))
    return tuple(scored[:limit])


#: Re-exported from `decide.wire`, which owns the one definition. Kept importable
#: from here because this module's callers have always found it at this name.
wire_floor = _wire_floor


# --------------------------------------------------------------------------------------
# Pricing a claim
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClaimPrice:
    """One add/drop pair, priced against a single common-random-numbers draw.

    **Two estimates of the same number, and the smaller-variance one is published.**
    `delta_title` is the paired points gain put through the franchise's own measured
    points-to-title rate; `bracket_title` re-runs the whole bracket on the modified
    roster and diffs the champion indicator. They estimate the same quantity, and the
    second one cannot see it: measured on the user's real leagues, a claim is worth
    0.02-0.17pp of title probability while the paired bracket difference carries a
    standard error near 0.3pp at 2,000 simulations, 0.2pp at 4,000 and still 0.05-0.10pp
    at 20,000. Ranking a dozen candidates on that ranks the noise, and the winner's
    estimate is biased upward by the selection -- which is exactly what the first live run
    of this module printed, and what a bracket-publishing engine printed again later.

    The plug-in's calibration is checked rather than assumed: at 20,000 simulations on
    Wine Wednesday the two estimators agree to within their errors over twelve candidates
    (mean bracket / mean plug-in = 0.88, rank correlation 0.81), and on a claim large
    enough for the bracket to resolve at all they agree to 2%. The rate itself is flat in
    the probe size from 1 to 4 points a week, which is what makes the linearization the
    plug-in relies on a measured fact rather than a hope.

    So the published number is the plug-in. Its bias is the curvature of the title
    response over a five-point move, which is negligible; its variance is a hundredth of
    the bracket's, because `delta_points` under common random numbers has a standard
    error of a few hundredths of a point. The bracket estimate is kept beside it as the
    consistency check `confirm` exists to provide, and `agrees` says whether it is one.
    """

    delta_points: float
    delta_points_stderr: float
    delta_title: float
    stderr: float
    #: The direct paired bracket difference, and its paired standard error. Zero when
    #: only the screen has been run.
    bracket_title: float = 0.0
    bracket_stderr: float = 0.0

    @property
    def significant(self) -> bool:
        """Whether the effect clears its own error. An exact zero never does.

        `core.Recommendation.significant` reads a zero error as "no Monte Carlo in this
        number, so trust it", which is right for an analytic result and wrong for the one
        case that actually produces it here: a free agent strictly below the wire floor
        adds *exactly* nothing in every simulation, so the paired difference is
        identically zero and so is its error. Calling that significant is the opposite of
        what a reader needs from the word.
        """
        if self.delta_title == 0.0:
            return False
        return abs(self.delta_title) > 2.0 * self.stderr if self.stderr > 0 else True

    @property
    def agrees(self) -> bool:
        """Whether the bracket check is consistent with the published estimate.

        Both errors count. A disagreement means the title response is not linear over a
        claim this size -- a real possibility for a claim that only helps in the playoff
        weeks -- and not that one of them is wrong.
        """
        if self.bracket_stderr <= 0:
            return True
        spread = 2.0 * math.hypot(self.bracket_stderr, self.stderr)
        return abs(self.bracket_title - self.delta_title) <= spread


def _paired(delta: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of a *paired* per-simulation difference."""
    n = int(delta.size)
    if n == 0:
        return 0.0, 0.0
    mean = float(delta.mean())
    if n < 2:
        return mean, 0.0
    return mean, float(delta.std(ddof=1) / math.sqrt(n))


def _move_players(move: Move, team_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    adds = tuple(p.player_id for p in move.players if p.to_team == team_id)
    drops = tuple(p.player_id for p in move.players if p.from_team == team_id)
    return adds, drops


def _lineup_scores(
    state: S.LeagueState,
    points: np.ndarray,
    rank: np.ndarray,
    replacement: Mapping[int, WireLevel] | Mapping[int, float] | float | None,
    player_ids: Sequence[int],
    *,
    noise: S.FloorNoise | None = None,
    team_index: int = 0,
) -> np.ndarray:
    """`(sims, weeks)` optimal-lineup totals for an arbitrary set of players.

    Deliberately the same kernel `sim.season.leave_one_out` runs on rather than a second
    copy of it: the floor-permutation trap `season._floors` documents is subtle enough
    that one implementation is the only safe number of implementations. The franchise
    handed in is a carrier for the player ids -- that kernel reads nothing else off it.

    `noise` is what an empty seat is actually PAID. Without it the seat gets its mean and
    carries no variance, and on the user's real rosters 15.7-19.6% of slot-weeks are
    empty -- one slot is empty 71-88% of the time -- so the team's weekly SD came out
    5.9-6.8% low and its season SD 5.6-6.0% low. The level was right; the spread was
    missing, and a bracket is decided by the spread. `decide/title.py` has drawn the seat
    since the stochastic floor landed; this surface and `edges/portfolio` did not.

    This used to hold a third copy of the empty-slot-group guard, as `_roster_floor`. It
    now lives in `sim/season._floors`, where every floor in the system is built, and the
    two tests were verified to agree on all 190 roster shapes across the three live
    leagues before the copy was deleted.
    """
    ids = tuple(int(p) for p in player_ids)
    positions = state.pool.positions_of(ids)
    plan = plan_from_slots(state.lineup_slot_counts, state.slot_eligibility, positions)
    carrier = S.Franchise(team_id=0, name="", player_ids=ids)
    return S._franchise_scores(
        state.pool,
        carrier,
        plan,
        points,
        rank,
        replacement,
        floor_noise=None if noise is None else noise.for_plan(plan, team_index),
    )


def claim_move(
    league_id: int, team_id: int, add: int | None, drop: int | None = None, bid: int | None = None
) -> Move:
    """The `core.Move` for one claim. `add=None` is a straight drop, which the screen uses."""
    players = []
    if add is not None:
        players.append(PlayerMove(player_id=add, from_team=None, to_team=team_id))
    if drop is not None:
        players.append(PlayerMove(player_id=drop, from_team=team_id, to_team=None))
    return Move(kind=MoveKind.WAIVER_CLAIM, league_id=league_id, players=tuple(players), bid=bid)


@dataclass(frozen=True, slots=True)
class RosterSimulator:
    """Prices any add/drop against one pre-drawn season, and nothing else.

    Every candidate meets identical football: the draw is an argument, the opposing
    managers are held fixed, and only the evaluated franchise's lineups are re-solved.
    That is the construction `sim.season.leave_one_out` uses, and it is why a null move
    returns exactly 0.0 rather than Monte Carlo fog.

    Satisfies `core.MoveEvaluator`, so the board can be driven by this or by
    `decide/title.py` interchangeably. `screen` is the analytic tier -- an exact paired
    `delta_points` converted through this franchise's own measured points-to-title
    exchange rate -- and `confirm` re-runs the bracket, which is where the standard error
    comes from.
    """

    state: S.LeagueState
    draw: Draw
    team_id: int
    replacement: Mapping[int, WireLevel] | Mapping[int, float] | float | None
    #: Points a week the exchange-rate probe shifts a franchise by, each way.
    probe: float
    #: dP(title)/d(one point of rest-of-season starting-lineup score) for THIS
    #: franchise, measured once by central difference on its own bracket.
    title_per_point: float
    #: Paired standard error of that rate. It is common to every candidate in the
    #: league, so it moves the level of a board and never its order.
    title_per_point_stderr: float
    baseline_points: float
    _index: int
    _points: np.ndarray
    _rank: np.ndarray
    _factors: np.ndarray
    _base_scores: np.ndarray
    _base_champ: np.ndarray
    _base_own: np.ndarray
    #: Uniforms for what each empty seat streams, fixed per `(seed, n_sims)`. This is
    #: common random numbers FOR THE WIRE: every candidate roster meets not only the same
    #: football but the same replacement bodies, so a paired difference still isolates
    #: the claim. Without it an empty seat is paid its mean with no variance at all, and
    #: 15.7-19.6% of slot-weeks on these rosters are empty.
    _noise: S.FloorNoise | None = None

    # -- construction ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        state: S.LeagueState,
        draw: Draw,
        team_id: int,
        *,
        replacement: Mapping[int, WireLevel] | Mapping[int, float] | float | None = None,
        efficiency: S.LineupEfficiency | None = None,
        probe: float = DEFAULT_PROBE,
    ) -> RosterSimulator:
        """Solve every roster once; everything after this touches one franchise."""
        if team_id not in state.team_ids:
            raise WaiverError(f"no team {team_id} in league {state.league_id}")
        # `LineupEfficiency()` is symmetric, which sim/season.py argues at length is the
        # only defensible default. Pass `.calibrated(...)` in if you disagree.
        efficiency = efficiency if efficiency is not None else S.LineupEfficiency()
        points = draw.points
        rank = S.ex_ante_rank(draw)
        factors = efficiency.draw(state, points.shape[0])
        # Drawn once and held, never per candidate: it is common random numbers for the
        # wire, and a fresh draw per candidate would put that noise back into every
        # paired difference this class exists to keep clean.
        noise = S.FloorNoise(state, draw) if S.has_spread(replacement) else None
        # Not `team_week_scores`: every franchise has to be scored through the same
        # empty-slot correction as the candidates, or a team that happens to carry no
        # kicker today gets the inflated baseline described in `sim/season._floors` and
        # every delta measured against it is wrong.
        base = (
            np.stack(
                [
                    _lineup_scores(
                        state,
                        points,
                        rank,
                        replacement,
                        f.player_ids,
                        noise=noise,
                        team_index=state.team_index[f.team_id],
                    )
                    for f in state.franchises
                ],
                axis=-1,
            ).astype(np.float32)
            * factors[:, None, :]
        )
        index = state.team_index[team_id]
        champ = S.simulate_from_scores(state, base, all_play=False).champions[:, index]

        self = cls(
            state=state,
            draw=draw,
            team_id=team_id,
            replacement=replacement,
            probe=float(probe),
            title_per_point=0.0,
            title_per_point_stderr=0.0,
            baseline_points=0.0,
            _index=index,
            _points=points,
            _rank=rank,
            _factors=factors,
            _base_scores=base,
            _base_champ=champ.astype(np.float64),
            _base_own=np.zeros(1),
            _noise=noise,
        )
        rate, rate_se = self.exchange_rate(team_id)
        object.__setattr__(self, "title_per_point", rate)
        object.__setattr__(self, "title_per_point_stderr", rate_se)
        # The baseline is the franchise's own un-haircut lineup total: `base` carries the
        # efficiency multiplier, and differencing a haircut total against a raw candidate
        # would report the haircut as the claim's value.
        own = self.season_points(self.roster)
        object.__setattr__(self, "_base_own", own)
        object.__setattr__(self, "baseline_points", float(own.mean()))
        return self

    def exchange_rate(self, team_id: int) -> tuple[float, float]:
        """`(dP(title)/d(one rest-of-season point), its paired standard error)`.

        Measured by central difference on that franchise's OWN bracket rather than
        borrowed from a league-wide constant: a contender and a team already out of it
        convert the same point into wildly different title probability, and on the user's
        three leagues the rate ranges over a factor of two (0.056, 0.079, 0.090 pp per
        rest-of-season point) in the same direction as their standings.

        `probe` is deliberately large, and the reason is VARIANCE rather than bias. A
        one-point-a-week bump moves a 95-point team by 1% and the title change it produces
        is barely above the paired Monte Carlo floor: measured on Wine Wednesday at 20,000
        simulations the rate comes out 0.0531 +/- 0.0033 at a one-point probe against
        0.0537 +/- 0.0014 at four -- the same number, at two and a half times the error,
        which is why a small probe disagrees by a factor of three at 4,000 and looks like
        a bias. Bias is what limits the probe from ABOVE: the same measurement gives
        0.0562 at eight points and 0.0636 at sixteen, so the response is visibly convex by
        then. Four points a week -- ~0.13 of the measured 34.4-point matchup spread -- is
        the flat part of that curve.
        """
        index = self.state.team_index[team_id]
        gained = np.zeros(self.n_sims, dtype=np.float64)
        for sign in (1.0, -1.0):
            shifted = self._base_scores.copy()
            shifted[:, :, index] += sign * self.probe
            arm = S.simulate_from_scores(self.state, shifted, all_play=False)
            gained += sign * arm.champions[:, index].astype(np.float64)
        return _paired(gained / (2.0 * self.probe * max(len(self.state.weeks), 1)))

    # -- primitives --------------------------------------------------------------------

    @property
    def n_sims(self) -> int:
        return int(self._points.shape[0])

    @property
    def roster(self) -> tuple[int, ...]:
        return self.state.franchise(self.team_id).player_ids

    def baseline_title(self, team_id: int | None = None) -> float:
        """`core.MoveEvaluator`: P(championship) under no move."""
        if team_id is None or team_id == self.team_id:
            return float(self._base_champ.mean())
        index = self.state.team_index[team_id]
        return float(
            S.simulate_from_scores(self.state, self._base_scores, all_play=False)
            .champions[:, index]
            .mean()
        )

    def roster_after(self, adds: Sequence[int], drops: Sequence[int]) -> tuple[int, ...]:
        gone = {int(p) for p in drops}
        missing = gone - set(self.roster)
        if missing:
            raise WaiverError(f"team {self.team_id} does not roster {sorted(missing)}")
        kept = [p for p in self.roster if p not in gone]
        return tuple([*kept, *(int(a) for a in adds)])

    def week_scores(self, player_ids: Sequence[int], team_id: int | None = None) -> np.ndarray:
        """`(sims, weeks)` optimal-lineup totals for this exact set of players.

        The lineup is chosen on the ex-ante rank and scored on the realised tensor, and
        every unfilled slot falls back to the wire floor -- so a roster with no kicker
        loses the gap between its kicker and the wire's, not eight points a week.

        `team_id` used to be ignored here, and is not any more: the lineup SOLVE still
        depends only on the players, the slots and the floor, but the empty seats are now
        paid a draw and those uniforms are per team. That is deliberate rather than
        incidental -- `S.FloorNoise` keeps each franchise's wire independent because byes
        are league-wide, so sharing one body across teams cancels in
        `Var(A) + Var(B) - 2Cov(A,B)` and throws away 40% of the spread it exists to
        restore. Defaulting to this simulator's own franchise keeps every candidate
        roster on the same uniforms, which is what makes the paired difference clean.
        """
        # The candidate's seats are drawn against the SAME uniforms as the baseline's,
        # keyed on the canonical seat rather than on a positional index, so a roster that
        # changes shape still meets the same wire. See `S.FloorNoise`.
        return _lineup_scores(
            self.state,
            self._points,
            self._rank,
            self.replacement,
            tuple(player_ids),
            noise=self._noise,
            team_index=self._index if team_id is None else self.state.team_index[team_id],
        )

    def drop_cost(self, player_id: int) -> tuple[float, float]:
        """`(what dropping him costs, what he was worth at the hindsight ceiling)`.

        Rest-of-season points, over the same draw. The first is what this board charges;
        the second is what he would have been worth to someone who knew which weeks he
        goes off, and the gap between them is the thing a points board cannot otherwise
        say. Neither is a reason on its own -- see `decide/trades.TradeFinder.cut_cost`
        for the measurement that says the ceiling must not be charged.
        """
        roster = list(self.roster)
        if player_id not in roster:
            return 0.0, 0.0
        kept = [p for p in roster if p != player_id]
        ex = float((self.season_points(roster) - self.season_points(kept)).mean())
        # Hindsight is "rank on the realisation": the lineup is chosen knowing what
        # every player actually scored. `S._franchise_scores` wants an explicit ranking
        # key, so the key IS the points tensor -- which is what `team_week_scores`
        # constructs internally when it is handed a bare tensor with no rank.
        full, less = (
            _lineup_scores(
                self.state, self._points, self._points, self.replacement, tuple(ids),
                noise=self._noise, team_index=self._index,
            ).sum(axis=1)
            for ids in (roster, kept)
        )
        return ex, float((full - less).mean())

    def season_points(self, player_ids: Sequence[int], team_id: int | None = None) -> np.ndarray:
        """`(sims,)` starting-lineup points over the whole remaining season."""
        return self.week_scores(player_ids, team_id).sum(axis=1, dtype=np.float64)

    def champions(self, player_ids: Sequence[int], team_id: int | None = None) -> np.ndarray:
        """`(sims,)` 0/1 title indicator for this roster, every other team held fixed.

        `simulate_from_scores` reads only records, ids and the schedule off the state, so
        the state's stale roster tuple is not consulted and does not need swapping.
        """
        team_id = self.team_id if team_id is None else team_id
        index = self.state.team_index[team_id]
        scores = self._base_scores.copy()
        scores[:, :, index] = (
            self.week_scores(player_ids, team_id) * self._factors[:, index : index + 1]
        )
        result = S.simulate_from_scores(self.state, scores, all_play=False)
        return result.champions[:, index].astype(np.float64)

    # -- the delta ---------------------------------------------------------------------

    def marginal_points(self, adds: Sequence[int], drops: Sequence[int] = ()) -> np.ndarray:
        """`(sims,)` paired `sum_w S(R + j - k, w) - sum_w S(R, w)`. The definition."""
        return self.season_points(self.roster_after(adds, drops)) - self._base_own

    def price(
        self, adds: Sequence[int], drops: Sequence[int] = (), *, confirm: bool = True
    ) -> ClaimPrice:
        """The claim's value in points and in championship probability.

        `confirm=True` additionally re-runs the bracket on the modified roster and
        records that as `bracket_title`, which is the cross-check, not the answer -- see
        `ClaimPrice` for why the plug-in is the published estimate either way.

        The published error propagates both terms: the paired points error through the
        rate, and the roster's points through the rate's own error. The second dominates
        and is common to every candidate in the league, so it sets how confidently a
        claim can be called positive at all without disturbing the board's order.
        """
        d_points = self.marginal_points(adds, drops)
        mean_points, se_points = _paired(d_points)
        title = mean_points * self.title_per_point
        stderr = math.hypot(
            se_points * abs(self.title_per_point), mean_points * self.title_per_point_stderr
        )
        if not confirm:
            return ClaimPrice(
                delta_points=mean_points,
                delta_points_stderr=se_points,
                delta_title=title,
                stderr=stderr,
            )
        after = self.champions(self.roster_after(adds, drops))
        bracket, bracket_se = _paired(after - self._base_champ)
        return ClaimPrice(
            delta_points=mean_points,
            delta_points_stderr=se_points,
            delta_title=title,
            stderr=stderr,
            bracket_title=bracket,
            bracket_stderr=bracket_se,
        )

    # -- core.MoveEvaluator ------------------------------------------------------------

    def screen(self, moves: Sequence[Move]) -> list[Recommendation]:
        return [self._recommend(m, confirm=False) for m in moves]

    def confirm(self, moves: Sequence[Move]) -> list[Recommendation]:
        return [self._recommend(m, confirm=True) for m in moves]

    def _recommend(self, move: Move, *, confirm: bool) -> Recommendation:
        adds, drops = _move_players(move, self.team_id)
        price = self.price(adds, drops, confirm=confirm)
        return Recommendation(
            move=move,
            delta_title=price.delta_title,
            delta_points=price.delta_points,
            stderr=price.stderr,
            leverage=1.0,
            confidence=_confidence(price, confirm),
            tags=(
                "waiver",
                "confirmed" if confirm else "screened",
                f"bracket:{price.bracket_title:.6f}",
                f"bracket_se:{price.bracket_stderr:.6f}",
            ),
        )


def _confidence(price: ClaimPrice, confirm: bool) -> str:
    """How much to trust one priced claim, in the three words `Recommendation` allows."""
    if not price.significant:
        return "low"
    if not confirm:
        return "medium"
    return "high" if price.agrees else "medium"


def default_evaluator(
    state: S.LeagueState,
    draw: Draw,
    team_id: int,
    *,
    replacement: Mapping[int, float] | float | None = None,
) -> MoveEvaluator:
    """`decide/title.py`'s engine if it is present, this module's own if it is not.

    Imported defensively rather than at module scope on purpose: `decide/title.py` is a
    sibling surface that may not exist in every build, and the waiver board is useful
    without it. Anything unexpected there is logged and stepped over rather than taking
    the whole board down with it.

    **`waiver_board` no longer calls this by default, and the reason is measured.**
    `decide/title.py`'s engine publishes the paired *bracket* difference as
    `delta_title`, which is the right answer for a trade worth several points a week and
    the wrong one for a waiver claim. Run on the user's three real leagues at the project
    default of 4,000 simulations, every claim on the board was worth between -0.2pp and
    +0.25pp and every one of them carried a standard error of 0.09pp to 0.36pp: the error
    is larger than the whole spread the ranking is made of. Concretely, on Wine Wednesday
    it ranked a +2.4-point add fifth behind a +1.0-point one, reported a +1.1-point add
    as **-0.20pp and significant**, and on Type shi published the top claim at +0.90pp
    where the points say +0.14pp -- the selected maximum of twelve noisy draws, which is
    exactly the bias `ClaimPrice` exists to avoid. Pass `evaluator=default_evaluator(...)`
    to `waiver_board` to route through it anyway; the board will then be ordered by
    something it cannot resolve.
    """
    try:
        from . import title  # type: ignore[attr-defined]

        factory = getattr(title, "evaluator_for", None)
        if factory is not None:
            return factory(state, draw, team_id, replacement=replacement)
        log.info("decide.title exists but exposes no evaluator_for(); using the local engine")
    except ImportError:
        log.debug("decide.title is not available; using the local engine")
    except Exception as err:  # pragma: no cover - defensive by design
        log.warning("decide.title could not build an evaluator (%s); using the local engine", err)
    return RosterSimulator.build(state, draw, team_id, replacement=replacement)


# --------------------------------------------------------------------------------------
# Priority as a depreciating asset
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpportunityDistribution:
    """What one week's wire is expected to offer, in `delta_title`.

    A discrete distribution rather than a fitted family, because the honest estimate is
    empirical: what a *future* week's wire offers looks like what this week's does, and
    this league's own board is the only sample that is about this league. The weights
    may sum to less than one; the missing mass is the weeks where nothing worth claiming
    appears, which is most of them.
    """

    values: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.weights):
            raise WaiverError("an opportunity distribution needs one weight per value")
        if any(w < 0 for w in self.weights):
            raise WaiverError("opportunity weights must be non-negative")
        total = sum(self.weights)
        if total <= 0.0 or total > 1.0 + 1e-9:
            raise WaiverError(f"opportunity weights must sum into (0, 1], got {total}")

    @property
    def mean(self) -> float:
        """`E[v]`, with the un-weighted mass counted as a week offering nothing."""
        return float(sum(v * w for v, w in zip(self.values, self.weights, strict=True)))

    def expected_excess(self, threshold: float) -> float:
        """`E[max(v - c, 0)]` -- the value of the option to claim above `c`.

        Decreasing and convex in `c`, which is what makes the backward induction below
        well behaved, and what makes `C` respond to the right tail rather than the mean.
        """
        return float(
            sum(w * max(v - threshold, 0.0) for v, w in zip(self.values, self.weights, strict=True))
        )

    def scaled(self, factor: float) -> OpportunityDistribution:
        """The same wire, `factor` times heavier in the tail. For sensitivity work."""
        return replace(self, values=tuple(v * factor for v in self.values))

    @classmethod
    def from_board(
        cls, deltas: Sequence[float], *, arrival: float = 1.0, top: int = 8
    ) -> OpportunityDistribution:
        """The empirical wire: the best few claims currently available, equally likely.

        `arrival` is the probability that a week offers anything at all; measure it from
        the league's own log with `arrival_rate_from_log`. The remaining mass sits at
        zero and is simply left out of the support.
        """
        best = sorted((float(d) for d in deltas if d > 0.0), reverse=True)[:top]
        if not best:
            return cls(values=(0.0,), weights=(1.0,))
        arrival = min(max(arrival, 1e-6), 1.0)
        return cls(values=tuple(best), weights=tuple([arrival / len(best)] * len(best)))

    @classmethod
    def exponential(cls, mean: float, *, points: int = 24) -> OpportunityDistribution:
        """A memoryless wire, discretised. For when there is no board to read yet."""
        if mean <= 0:
            raise WaiverError("an exponential opportunity distribution needs a positive mean")
        qs = (np.arange(points) + 0.5) / points
        values = -mean * np.log(1.0 - qs)
        return cls(values=tuple(values.tolist()), weights=tuple([1.0 / points] * points))


@dataclass(frozen=True, slots=True)
class ContinuationTable:
    """`C_t(p)`: what holding waiver priority `p` into week `t+1` is worth.

    Rows are `weeks`, columns are priorities 1..N. Claim iff a candidate's `delta_title`
    clears `values[t, p-1]`.
    """

    weeks: tuple[int, ...]
    #: (weeks, priorities) continuation value, in championship probability.
    values: np.ndarray
    #: (weeks, priorities) the value function itself, for inspection.
    utility: np.ndarray
    contest_rate: float
    promotion: float

    @property
    def size(self) -> int:
        return int(self.values.shape[1])

    def threshold(self, week: int, priority: int) -> float:
        """`C_t(p)`. Zero at the last waiver run and zero at the worst priority."""
        if priority < 1 or priority > self.size:
            raise WaiverError(f"priority {priority} is outside 1..{self.size}")
        if week not in self.weeks:
            # A week we are not modelling has no future left to protect.
            return 0.0
        return float(self.values[self.weeks.index(week), priority - 1])

    def table(self, priorities: Sequence[int] | None = None) -> str:
        cols = list(priorities or range(1, self.size + 1))
        head = "week " + " ".join(f"{p:>7d}" for p in cols)
        lines = [head, "-" * len(head)]
        for i, w in enumerate(self.weeks):
            lines.append(
                f"{w:4d} " + " ".join(f"{self.values[i, p - 1] * 100:6.3f}%" for p in cols)
            )
        return "\n".join(lines)


def _promotion_matrix(size: int, promotion: float) -> np.ndarray:
    """`(N, N)` distribution of next week's priority index given this week's.

    Index `p` (zero-based) has `p` teams ahead of it; each spends its claim independently
    with probability `promotion`, and every one that does moves us up a place. At
    `promotion = 0` this is the identity, which is the model the docstring states and
    the one the boundary conditions are stated for.
    """
    out = np.zeros((size, size))
    if promotion <= 0.0:
        np.fill_diagonal(out, 1.0)
        return out
    for p in range(size):
        for k in range(p + 1):
            out[p, p - k] = math.comb(p, k) * promotion**k * (1.0 - promotion) ** (p - k)
    return out


def continuation_values(
    weeks: Sequence[int],
    size: int,
    opportunity: OpportunityDistribution | Sequence[OpportunityDistribution],
    *,
    contest_rate: float = DEFAULT_CONTEST_RATE,
    promotion: float = 0.0,
) -> ContinuationTable:
    """Solve the optimal-stopping problem over (week, priority) by backward induction.

    The recursion is

        C_t(p) = Utilde_{t+1}(p) - Utilde_{t+1}(N)
        U_t(p) = Utilde_{t+1}(p) + w(p) * E[max(v - C_t(p), 0)]

    where `w(p) = (1 - contest_rate)^(p-1)` is the chance nobody ahead of us wants the
    same player and `Utilde` is `U` after the queue moves. Two boundaries fall out and
    both are pinned in the tests: `C` at the last modelled week is zero, because
    `U_{T+1} = 0` and unspent priority has no salvage; and `C_t(N)` is zero, because
    winning and passing lead to the same place from the back of the queue.

    `w(p)` cancels out of the threshold entirely. It scales the level of `U`, and
    therefore how much an earlier week values the queue, but it never decides a claim --
    which is why a rough `contest_rate` is good enough and a wrong one is not dangerous.
    """
    if size < 1:
        raise WaiverError("a league needs at least one team")
    if not 0.0 <= contest_rate < 1.0:
        raise WaiverError(f"contest_rate must be in [0, 1), got {contest_rate}")
    if not 0.0 <= promotion <= 1.0:
        raise WaiverError(f"promotion must be a probability, got {promotion}")
    order = tuple(int(w) for w in weeks)
    if not order:
        raise WaiverError("continuation values need at least one week")
    dists = (
        [opportunity] * len(order)
        if isinstance(opportunity, OpportunityDistribution)
        else list(opportunity)
    )
    if len(dists) != len(order):
        raise WaiverError(f"got {len(dists)} opportunity distributions for {len(order)} weeks")

    win = (1.0 - contest_rate) ** np.arange(size)
    move = _promotion_matrix(size, promotion)

    values = np.zeros((len(order), size))
    utility = np.zeros((len(order), size))
    forward = np.zeros(size)
    for t in range(len(order) - 1, -1, -1):
        drifted = move @ forward
        cont = drifted - drifted[size - 1]
        dist = dists[t]
        option = np.array([dist.expected_excess(float(c)) for c in cont])
        forward = drifted + win * option
        values[t] = cont
        utility[t] = forward
    return ContinuationTable(
        weeks=order,
        values=values,
        utility=utility,
        contest_rate=contest_rate,
        promotion=promotion,
    )


def contest_rate_from_log(log_: TransactionLog, size: int, weeks: int) -> float:
    """Per-team-week probability of a waiver claim, from the league's own history.

    Falls back to `DEFAULT_CONTEST_RATE` when the log is unavailable -- ESPN gates it
    behind credentials, and an empty log means "refused", not "nobody claims".
    """
    if not getattr(log_, "available", False) or size < 1 or weeks < 1:
        return DEFAULT_CONTEST_RATE
    claims = sum(1 for t in log_ if t.is_waiver and t.is_executed)
    if claims == 0:
        return DEFAULT_CONTEST_RATE
    return float(min(max(claims / (size * weeks), 0.01), 0.9))


def arrival_rate_from_log(log_: TransactionLog, weeks: int) -> float:
    """Probability that a given week's wire offers anything worth claiming.

    Measured as the share of weeks in which somebody in the league made an executed
    waiver claim. Crude, and better than a guess: a league claiming every week has a
    live wire and a higher continuation value than one that has gone quiet.
    """
    if not getattr(log_, "available", False) or weeks < 1:
        return 1.0
    active = {t.scoring_period_id for t in log_ if t.is_waiver and t.is_executed}
    if not active:
        return 1.0
    return float(min(max(len(active) / weeks, 0.05), 1.0))


# --------------------------------------------------------------------------------------
# FAAB -- the secondary path; none of the user's leagues run it
# --------------------------------------------------------------------------------------


class RivalBids(Protocol):
    """The distribution of the best *rival* bid, which is what a bid has to beat."""

    def cdf(self, bid: float) -> float: ...

    def pdf(self, bid: float) -> float: ...


@dataclass(frozen=True, slots=True)
class UniformRivalBids:
    """`rivals` opponents bidding uniform on `[0, high]`; the max is what you must beat.

    Kept because it is the case with a closed form: `F/f = b/rivals`, so the first-order
    condition collapses to `b = v * n/(n+1)` with `n = rivals`, the textbook first-price
    answer. That identity is what `test_waivers.py` pins the solver against.
    """

    rivals: int
    high: float

    def __post_init__(self) -> None:
        if self.rivals < 1 or self.high <= 0:
            raise WaiverError("uniform rival bids need at least one rival and a positive cap")

    def cdf(self, bid: float) -> float:
        return float(min(max(bid / self.high, 0.0), 1.0) ** self.rivals)

    def pdf(self, bid: float) -> float:
        x = min(max(bid / self.high, 0.0), 1.0)
        return float(self.rivals * x ** (self.rivals - 1) / self.high)


@dataclass(frozen=True, slots=True)
class LognormalRivalBids:
    """The max of `rivals` lognormal bids, fitted to the league's own winning bids.

    Lognormal because that is what the population data shows -- a median winning bid near
    1.4% of budget with a long right tail -- and fitted to *this* league because bid
    culture does not transfer.

    **`rivals` defaults to 1, and that is not a typo.** ESPN's log records the *winning*
    bid, which is already the maximum over everyone who bid on that player. Fitting that
    and then raising it to the power of "the two to five teams actually bidding" applies
    the auction twice and overbids badly: on a $100 budget the doubled version wanted $55
    for a marginal D/ST upgrade where the single application wants $9. So the fitted
    distribution IS the best-rival-bid distribution, with the number of bidders already
    folded into the sample. Pass `rivals > 1` only when the input is individual bids.
    """

    rivals: int
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.rivals < 1 or self.sigma <= 0:
            raise WaiverError("lognormal rival bids need a rival and a positive sigma")

    def _single(self, bid: float) -> float:
        if bid <= 0:
            return 0.0
        z = (math.log(bid) - self.mu) / self.sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def cdf(self, bid: float) -> float:
        return float(self._single(bid) ** self.rivals)

    def pdf(self, bid: float) -> float:
        if bid <= 0:
            return 0.0
        z = (math.log(bid) - self.mu) / self.sigma
        density = math.exp(-0.5 * z * z) / (bid * self.sigma * math.sqrt(2.0 * math.pi))
        return float(self.rivals * self._single(bid) ** (self.rivals - 1) * density)

    @classmethod
    def from_history(cls, bids: Sequence[int], *, rivals: int = 1) -> LognormalRivalBids:
        """Fit to executed winning bids. Zero-dollar claims are dropped, not logged as 0.

        A `$0` claim in ESPN's log is a waiver nobody contested, so it says nothing about
        what a contested player costs; keeping them drags the fit to zero and makes every
        recommended bid a dollar.
        """
        positive = [float(b) for b in bids if b and b > 0]
        if len(positive) < 3:
            raise WaiverError(
                f"only {len(positive)} positive winning bids in this league's log; "
                "not enough to fit a bid distribution"
            )
        logs = np.log(np.asarray(positive))
        return cls(rivals=rivals, mu=float(logs.mean()), sigma=float(max(logs.std(ddof=1), 1e-3)))

    @classmethod
    def population(cls, budget: int, *, median_share: float = 0.014, sigma: float = 1.3):
        """The population prior, for a league with no bid history of its own.

        Median winning bid at 1.4% of budget with a long right tail, which on $100 puts
        the median at $1, the 95th percentile near $12 and the 99th near $30 -- the shape
        FAAB leagues actually produce, where almost everything goes for a dollar and one
        player a season goes for a third of the budget.

        This replaces a uniform-on-the-whole-budget fallback, which is not a prior so much
        as an assumption that rivals bid at random: it priced a marginal D/ST upgrade at
        $55 of a $100 budget on the live board. Use the league's own history the moment
        there is any (`from_history`); bid culture does not transfer between leagues.
        """
        if budget < 1:
            raise WaiverError("a bid prior needs a positive budget")
        return cls(rivals=1, mu=math.log(max(budget * median_share, 0.5)), sigma=sigma)


def first_order_bid(
    value: float, rivals: RivalBids, *, shadow_price: float = 1.0, high: float | None = None
) -> float:
    """Solve `b + F(b)/f(b) = value / shadow_price` for the optimal bid.

    The first-order condition of `max_b F(b) * (value - shadow_price * b)`: a higher bid
    buys `f(b)` more probability of winning `value - shadow_price*b` and costs
    `shadow_price * F(b)` on the branch where you were winning anyway. `shadow_price` is
    the marginal value of a budget dollar from `faab_shadow_price`; it goes to zero at
    season end because unspent budget has no salvage, which is what correctly produces
    "bid nothing most weeks, bid enormous once".
    """
    if value <= 0:
        return 0.0
    if shadow_price <= 0:
        # A dollar is worthless: bid whatever the cap allows.
        return float(high) if high is not None else float("inf")
    target = value / shadow_price
    cap = float(high) if high is not None else target

    def gap(b: float) -> float:
        # `F/f` when the density is real. When it is not, the two ends are different
        # problems: deep in a lognormal's left tail both halves underflow together and
        # the true ratio tends to ZERO, so reading the 0/0 as infinity puts the whole
        # bracket above the root and every bid comes back at zero -- which it did before
        # this branch existed. A zero density with positive mass is the other end, where
        # more money really does buy nothing.
        f, cdf = rivals.pdf(b), rivals.cdf(b)
        ratio = cdf / f if f > 0 else (0.0 if cdf <= 0.0 else float("inf"))
        return b + ratio - target

    if gap(cap) < 0:
        return cap
    from scipy.optimize import brentq

    lo = 1e-12 * max(cap, 1.0)
    if gap(lo) > 0:
        return 0.0
    return float(brentq(gap, lo, cap, xtol=1e-10))


def odd_dollars(bid: float, *, budget: int | None = None) -> int:
    """Round up to the nearest odd dollar, at least one. Rivals cluster on 5s and 10s."""
    b = max(int(round(bid)), 1)
    if b % 2 == 0:
        b += 1
    if budget is not None and budget >= 1:
        cap = budget if budget % 2 == 1 else budget - 1
        b = min(b, max(cap, 1))
    return max(b, 1)


def faab_shadow_price(
    weeks: int,
    budget: int,
    opportunity: OpportunityDistribution,
    rivals: RivalBids,
    *,
    grid: int = 1,
) -> np.ndarray:
    """`(weeks, budget+1)` marginal value of the last budget dollar.

    The budget DP: with `b` dollars and the weeks from `t` on still to come,

        V_t(b) = E_v[ max_c  F(c)*(v + V_{t+1}(b-c)) + (1-F(c))*V_{t+1}(b) ]

    and `lambda_t(b) = V_t(b) - V_t(b-1)`. `V_{T+1} = 0` -- unspent budget has zero
    salvage -- so `lambda` decays to zero at the season's end and the shading in
    `first_order_bid` unwinds with it.
    """
    if weeks < 1 or budget < 1:
        raise WaiverError("a budget DP needs at least one week and one dollar")
    bids = np.arange(0, budget + 1, max(grid, 1))
    win = np.array([rivals.cdf(float(b)) for b in bids])
    values = np.asarray(opportunity.values)
    weights = np.asarray(opportunity.weights)
    nothing = float(max(1.0 - weights.sum(), 0.0))

    lam = np.zeros((weeks, budget + 1))
    forward = np.zeros(budget + 1)
    for t in range(weeks - 1, -1, -1):
        current = np.zeros(budget + 1)
        for b in range(budget + 1):
            legal = bids <= b
            after = forward[b - bids[legal]]
            # (opportunities, bids): what each bid is worth against each arriving player.
            gain = win[legal] * (values[:, None] + after[None, :]) + (1 - win[legal]) * forward[b]
            current[b] = float(weights @ gain.max(axis=1)) + nothing * forward[b]
        lam[t, 1:] = np.diff(current)
        forward = current
    return lam


# --------------------------------------------------------------------------------------
# Blocking
# --------------------------------------------------------------------------------------


def blocking_value(rival_gain: float, size: int) -> float:
    """What denying one rival a player is worth to US.

    Championship probability sums to one, so a rival's gain of `x` is a loss of `x`
    spread across the other `size - 1` teams, and our own share of it is `x / (size - 1)`
    on average. In a 12-team league that is 9% of what owning the player is worth, which
    is why a surface should almost never recommend a blocking claim -- and why this is
    the only route to a blocking number in this module.
    """
    if size < 2:
        raise WaiverError("blocking needs at least two teams")
    return float(rival_gain) / (size - 1)


# --------------------------------------------------------------------------------------
# Widening the pool so free agents live in the same tensor
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AugmentedSim:
    """A `LeagueSim` widened so free agents share a tensor with the rosters.

    The whole comparison depends on this: two rosters must be judged against one draw,
    and a free agent with no column has nothing to be judged in. The panel is rebuilt
    over the rostered players *plus* the candidates, which is why the candidate list is
    capped -- the tensor is `(sims, weeks, players)` and it is the memory budget.
    """

    state: S.LeagueState
    draw: Draw
    screen_draw: Draw
    agents: tuple[FreeAgent, ...]
    #: `WireLevel`s: an empty seat is paid a draw, not its mean. `wire_floor` is the
    #: mean of exactly this mapping, so nothing about the level changed.
    floor: Mapping[int, WireLevel]


def _availability(sim: LeagueSim) -> Availability | None:
    """Ask ESPN who is actually on waivers, or give up loudly.

    One request. A failure is not fatal but it IS expensive: with no answer every
    unrostered player is treated as costing a claim, which is what this whole split
    exists to stop doing. So it is logged at warning, not info.
    """
    league = getattr(sim, "league", None)
    getter = getattr(league, "availability", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception as err:  # noqa: BLE001 - offline, unauthed, or a moved endpoint
        log.warning(
            "could not read waiver status for league %s (%s); every free agent will be "
            "priced as if it cost a waiver claim, which understates the board",
            sim.state.league_id,
            err,
        )
        return None


#: The third copy of this was the one that mattered: `decide/trades.wire_pool` spelled it
#: `set(state.pool.player_ids)`, which is a different set as soon as anything widens the
#: pool -- which is exactly what `augment` below does.
_all_rostered = _wire_all_rostered


def _complete(
    outlooks: Sequence[PlayerOutlook], wanted: set[int], weeks: Sequence[int], season: int
) -> list[PlayerOutlook]:
    """Give every candidate an outlook in every remaining week, zeroed where absent.

    `panel_for` demands per-player-week coverage, and it is right to: an unsupplied week
    is silently drawn as zero, so a partial source makes a team score nothing rather than
    raising. Rostered players are already completed by `pipeline._fill_unprojected`; free
    agents are not, and on the live leagues exactly one does turn up short -- a D/ST
    missing week 8 -- which would otherwise take the whole board down.
    """
    span = set(weeks)
    out: list[PlayerOutlook] = []
    for o in outlooks:
        if o.player_id not in wanted or span <= set(o.weeks):
            out.append(o)
            continue
        filled = dict(o.weeks)
        for w in sorted(span - set(filled)):
            filled[w] = WeeklyOutlook(
                player_id=o.player_id,
                season=season,
                week=w,
                position_id=o.position_id,
                mean=0.0,
                sd=0.0,
                p_zero=1.0,
                shape=1.0,
                scale=1.0,
                pro_team_id=0,
                playing=False,
            )
        out.append(replace(o, weeks=filled))
    return out


def augment(
    sim: LeagueSim,
    agents: Sequence[FreeAgent],
    *,
    n_sims: int | None = None,
    screen_sims: int = DEFAULT_SCREEN_SIMS,
    seed: int | None = None,
    replacement: Mapping[int, float] | None = None,
    wire_depth: int = DEFAULT_WIRE_DEPTH,
    outlooks: Sequence[PlayerOutlook] | None = None,
) -> AugmentedSim:
    """Rebuild the pool, panel and draws with the candidate free agents included.

    Both draws come from one `WeeklySampler`, so every candidate priced against
    `AugmentedSim.draw` is paired with every other and with the baseline. **That pairing
    is what the claim prices rest on, and it is the only invariance this function has.**

    It is tempting to say more -- the sampler keys each player's normal stream on his
    *player id* rather than his column, which reads like "widening the pool leaves every
    incumbent's season bit-identical". It does not, and an earlier version of this
    docstring said it did. `WeeklySampler._simulate` mixes each week's raw normals
    through the per-pro-team correlation blocks and `PlayerPool.of` holds ids ascending,
    so a candidate who joins an incumbent's NFL huddle *and sorts ahead of him* pushes him
    down inside that block and re-rolls his whole season. Measured on Wine Wednesday:
    **160 of 225 incumbents change**, every one of them on a pro team a candidate joined,
    with single player-weeks moving by up to ~370 points. The invariance survives only
    for players whose correlation block the widening did not disturb.

    The consequence is bounded and worth stating: the whole board lives in one draw, so
    no delta is affected, but `WaiverReport.baseline_title` is a *different* Monte Carlo
    universe from `pipeline.championship_table`'s and will not match it exactly.
    """
    state = sim.state
    rows = [
        (pid, pos, team, state.pool.name(pid))
        for pid, pos, team in zip(
            state.pool.player_ids, state.pool.position_ids, state.pool.pro_team_ids, strict=True
        )
    ]
    extra = [(a.player_id, a.position_id, a.pro_team_id, a.name) for a in agents]
    wide = replace(state, pool=S.PlayerPool.of([*rows, *extra]))
    # `outlooks` overrides `sim.outlooks` so a caller holding a re-dealt projection set
    # can hand it in. Passed rather than swapped onto a copy of `sim`: the panel, the
    # sampler and the wire floor below all have to read the SAME set, and a caller that
    # tilted one of the three and not the others is the mixed-currency bug this codebase
    # has already found twice.
    source = sim.outlooks if outlooks is None else outlooks
    complete = _complete(source, {a.player_id for a in agents}, state.weeks, state.season)
    panel = S.panel_for(wide, complete)
    sampler = WeeklySampler(panel, seed=sim.seed if seed is None else seed)
    # `wire_levels`, not `wire_floor`: the mean is what the lineup solver compares a
    # roster player against, but the seat is PAID a draw and the spread is what a bracket
    # is decided by. `wire_floor` is the mean of exactly this, so the level is unchanged.
    floor = replacement or wire_levels(
        source, _all_rostered(state), state.weeks, state.slot_eligibility, depth=wire_depth
    )
    return AugmentedSim(
        state=wide,
        draw=sampler.draw(sim.n_sims if n_sims is None else n_sims),
        screen_draw=sampler.draw(screen_sims),
        agents=tuple(agents),
        floor=floor,
    )


# --------------------------------------------------------------------------------------
# The board
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WaiverReport:
    """One league's whole waiver picture, denominated in championship probability."""

    league_id: int
    season: int
    name: str
    team_id: int
    team_name: str
    week: int
    uses_faab: bool
    priority: int | None
    budget: int
    threshold: float
    #: P(championship) under no move, measured on THIS board's draw and with the wire
    #: floor applied to every team. It will still not match `pipeline.championship_table`
    #: exactly, but no longer because the two disagree about the floor -- that gap is
    #: closed and both now stream a replacement into an empty seat. What remains is that
    #: this board runs on a WIDENED panel, one that gives the candidate free agents
    #: columns to be simulated in, so it is a different Monte Carlo universe. Compare
    #: deltas across surfaces, not levels.
    baseline_title: float
    title_per_point: float
    title_per_point_stderr: float
    #: dP(win)/dmu for the coming matchup relative to its peak. Near zero means the claim
    #: cannot change this week's game regardless of who is better.
    week_leverage: float
    #: SD of (my score - my opponent's) for the coming matchup, measured off the tensor.
    #: Compare against the corpus-wide 34.4.
    sd_diff: float
    continuation: ContinuationTable | None
    #: Every confirmed candidate, best first.
    board: tuple[Recommendation, ...]
    #: The conditional waterfall: everything ON WAIVERS above the threshold, in
    #: submission order. These cost priority if they land.
    claims: tuple[Recommendation, ...]
    #: Blocking candidates, already discounted by 1/(N-1).
    blocks: tuple[Recommendation, ...]
    hold: Recommendation
    free_agents: tuple[FreeAgent, ...]
    #: Positive adds that cost NOTHING -- first-come free agents, no priority spent, no
    #: threshold to clear, available right now. Kept separate from `claims` rather than
    #: merged into it, because everything downstream prices a claim at "waiver priority"
    #: (`report._SURFACE_COST`, the queue's `actionable`, `portfolio._waiver_recs`) and
    #: mixing the two would relabel a free add as costing the thing it does not cost.
    free_adds: tuple[Recommendation, ...] = ()
    #: How many unrostered players actually cost a claim, and how many are simply free.
    #: `None` when ESPN was not asked, in which case every candidate was treated as
    #: costing one -- the expensive assumption.
    n_on_waivers: int | None = None
    n_free_agents_available: int | None = None
    #: False when the waiver order could not be read and `priority` is an assumption
    #: rather than a fact. The threshold is only as good as this.
    priority_known: bool = True
    #: `slot -> (streaming points, best rosterable points, that player)` for every
    #: single-body slot, over the remaining season. See `stream_advantage`: on the live
    #: leagues the D/ST gap is ~37 points a season against ~13 at K and TE. The gap is
    #: non-negative by construction; it is the RATIO between positions that says which
    #: seat is worth streaming. See `stream_advantage`.
    stream_advantage: Mapping[int, tuple[float, float, str]] = field(default_factory=dict)

    @property
    def best(self) -> Recommendation:
        """The single thing to do. A free add outranks a claim at equal value.

        Not because it is worth more -- it is worth the same -- but because it costs
        nothing and cannot be contested, so at a tie it strictly dominates.
        """
        if self.free_adds and self.claims:
            return (
                self.free_adds[0]
                if self.free_adds[0].delta_title >= self.claims[0].delta_title
                else self.claims[0]
            )
        if self.free_adds:
            return self.free_adds[0]
        return self.claims[0] if self.claims else self.hold

    @property
    def actions(self) -> tuple[Recommendation, ...]:
        """Everything worth doing, free adds first. Empty means hold."""
        return (*self.free_adds, *self.claims)

    def table(self, limit: int = 12) -> str:
        cost = (
            f"priority {self.priority}/{self.continuation.size if self.continuation else '?'}"
            f"{'' if self.priority_known else ' (ASSUMED -- ESPN would not say)'}"
            if not self.uses_faab
            else f"FAAB ${self.budget}"
        )
        head = (
            f"{'add':22s} {'pos':>4s} {'drop':20s} {'dPts':>7s} "
            f"{'dTitle':>8s} {'+/-':>7s} {'bracket':>9s} {'sig':>4s}"
        )
        lines = [
            f"{self.name} ({self.league_id}) week {self.week} -- {self.team_name}",
            f"  title {self.baseline_title * 100:.2f}%   "
            f"{self.title_per_point * 100:+.4f}pp per ROS point "
            f"(+/-{self.title_per_point_stderr * 100:.4f})   "
            f"leverage {self.week_leverage:.2f} (sd_diff {self.sd_diff:.1f})",
            f"  {cost}, claim threshold {self.threshold * 100:.3f}pp",
            head,
            "-" * len(head),
        ]
        for rec in self.board[:limit]:
            price = price_of(rec)
            bracket = (
                f"{price.bracket_title * 100:+6.2f}%" if price.bracket_stderr > 0 else "      -"
            )
            lines.append(
                f"{_tag(rec, 'add:')[:22]:22s} {_tag(rec, 'pos:'):>4s} "
                f"{_tag(rec, 'drop:')[:20]:20s} {rec.delta_points:7.1f} "
                f"{rec.delta_title * 100:7.3f}% {rec.stderr * 100:6.3f}% {bracket:>9s} "
                # `price.significant`, not `rec.significant`: `core.Recommendation` reads
                # a zero standard error as certainty, and a claim worth exactly nothing
                # has exactly that. See `ClaimPrice.significant`.
                f"{'yes' if price.significant else 'no':>4s}"
            )
        lines.append(f"  -> {self.best.rationale}")
        return "\n".join(lines)


def _tag(rec: Recommendation, prefix: str) -> str:
    for tag in rec.tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return "-"


def price_of(rec: Recommendation) -> ClaimPrice:
    """Recover the full price from a `Recommendation` this module produced.

    `core.Recommendation` is a fixed contract and carries one estimate, so the bracket
    cross-check travels in the tags. An injected evaluator that does not set them comes
    back with a zero cross-check, which `ClaimPrice.agrees` reads as "nothing to check".
    """
    bracket, bracket_se = _tag(rec, "bracket:"), _tag(rec, "bracket_se:")
    return ClaimPrice(
        delta_points=rec.delta_points,
        delta_points_stderr=0.0,
        delta_title=rec.delta_title,
        stderr=rec.stderr,
        bracket_title=float(bracket) if bracket != "-" else 0.0,
        bracket_stderr=float(bracket_se) if bracket_se != "-" else 0.0,
    )


def _matchup_leverage(engine: RosterSimulator, week: int) -> tuple[float, float]:
    """`(leverage, sd_diff)` for the coming matchup, measured off the tensor.

    A claim worth four points is worth 4.6pp of weekly win probability in a coin flip and
    0.6pp in a blowout. Reporting only the points hides that, and it is the difference
    between "make this claim now" and "this week does not care".
    """
    state = engine.state
    game = next(
        (
            g
            for g in state.remaining_games
            if week in g.weeks and engine.team_id in (g.home_team_id, g.away_team_id)
        ),
        None,
    )
    if game is None:
        return 0.0, SD_DIFF
    rows = [state.week_index[w] for w in game.weeks]
    mine = state.team_index[engine.team_id]
    other = game.away_team_id if game.home_team_id == engine.team_id else game.home_team_id
    theirs = state.team_index[other]
    diff = (
        engine._base_scores[:, rows, mine].sum(axis=1)
        - engine._base_scores[:, rows, theirs].sum(axis=1)
    ).astype(np.float64)
    sd = float(diff.std(ddof=1)) if diff.size > 1 else SD_DIFF
    sd = sd if sd > 0 else SD_DIFF
    return _leverage(float(diff.mean()), sd), sd


def _rationale(
    agent: FreeAgent,
    drop_name: str,
    price: ClaimPrice,
    threshold: float,
    priority: int | None,
    bid: int | None,
    lever: float,
    *,
    priority_known: bool = True,
    acquisition: object | None = None,
) -> str:
    lead = ("Claim " if agent.on_waivers else "Add ") + f"{agent.name} ({agent.position})"
    lead += f", drop {drop_name}" if drop_name and drop_name != "-" else " into the open spot"
    body = (
        f"{price.delta_points:+.1f} starting-lineup points over the rest of the season, "
        f"which is {price.delta_title * 100:+.2f}pp of title probability "
        f"(+/-{price.stderr * 100:.2f}pp)."
    )
    if not price.significant:
        body += " That is inside its own Monte Carlo error, so treat it as not significant."
    if price.bracket_stderr > 0:
        body += (
            f" A direct paired bracket run puts it at {price.bracket_title * 100:+.2f}pp "
            f"+/-{price.bracket_stderr * 100:.2f}pp"
            + (
                ", consistent, and too noisy at this simulation count to rank on."
                if price.agrees
                else " -- more than two combined standard errors away. Either the title "
                "response is not linear over a claim this size or the bracket drew a "
                "tail, which it does often enough across a board this wide. Read the "
                "points."
            )
        )
    if not agent.on_waivers:
        # He costs nothing, so there is no continuation value to clear and no threshold
        # to compare against. Charging him one is what suppressed 24 of 28 profitable
        # rows on a board where 809 of the 841 available players were free.
        verdict = (
            "ESPN has him as a plain free agent, not on waivers: no priority is spent "
            "and no claim is contested, so any positive number here is worth taking."
            if price.delta_title > 0.0
            else "He costs nothing, but he is not worth a roster spot either."
        )
    elif priority is not None:
        verdict = (
            f"Above the {threshold * 100:.3f}pp continuation value of holding priority "
            f"{priority}, so spending it is right."
            if price.delta_title >= threshold
            else f"Below the {threshold * 100:.3f}pp continuation value of priority "
            f"{priority}: hold the claim."
        )
        if not priority_known:
            verdict += (
                f" ESPN would not report the waiver order, so priority {priority} -- the "
                "front of the queue, the most expensive place to spend from -- is assumed "
                "rather than read. Check it before submitting."
            )
    else:
        verdict = f"Bid ${bid} -- odd dollars, because rivals cluster on 5s and 10s."
    lever_note = f"Matchup leverage {lever:.2f}" + (
        "; the coming game is close, so the points land."
        if lever > 0.6
        else "; the coming game barely cares, so this is a rest-of-season claim."
    )
    timing = timing_note(acquisition, on_waivers=agent.on_waivers)
    return " ".join([lead + ".", body, verdict, lever_note, timing])


def _hold_rationale(
    uses_faab: bool,
    priority: int | None,
    threshold: float,
    board: Sequence[Recommendation],
    claims: Sequence[Recommendation],
    free_adds: Sequence[Recommendation] = (),
    *,
    priority_known: bool = True,
    acquisition: object | None = None,
) -> str:
    best = board[0].delta_title if board else 0.0
    if free_adds:
        # "Hold" is the wrong word when something on the board costs nothing. This branch
        # used to be unreachable, because every free agent was charged a waiver claim and
        # filtered out before it could be reported.
        #
        # Singular on purpose. These are ALTERNATIVES: each was priced on its own against
        # today's roster and most of them drop the same player, so their gains are not
        # additive and "make all 23" is not a thing anyone can do. Same submodularity the
        # claim waterfall warns about, in a place where nothing stops the reader acting.
        drops = {_tag(r, "drop:") for r in free_adds}
        alternatives = len(free_adds) - 1
        lead = (
            f"Add {_tag(free_adds[0], 'add:')} now for "
            f"{free_adds[0].delta_title * 100:+.2f}pp. He is a plain free agent, not on "
            "waivers: no priority is spent, no claim can be contested, and there is no "
            "threshold to clear or deadline to wait for."
        )
        if alternatives:
            lead += (
                f" {alternatives} other free add(s) also help, but they are alternatives "
                "to this one rather than additions"
                + (
                    " -- every one of them drops the same player."
                    if len(drops) <= 1
                    else f": they drop {len(drops)} different players between them, and each "
                    "was priced on its own against today's roster, so two together are "
                    "worth less than the sum. Make one, then re-run."
                )
            )
        if not claims:
            return (
                f"{lead} Nothing that would cost a claim is worth one: the best is "
                f"{best * 100:+.2f}pp against a {threshold * 100:.3f}pp continuation value."
            )
        return lead
    caveat = (
        ""
        if uses_faab or priority_known
        else f" ESPN would not report the waiver order, so priority {priority} is assumed "
        "rather than read, and the threshold is only as good as that assumption."
    )
    if not claims:
        cost = (
            f"holding waiver priority {priority} is worth {threshold * 100:.3f}pp"
            if not uses_faab
            else "the budget is worth more later"
        )
        return (
            f"Hold. The best claim on the board is worth {best * 100:+.2f}pp and {cost}, "
            f"so spending now destroys value.{caveat} "
            f"{timing_note(acquisition, on_waivers=True)}"
        )
    currency = "budget" if uses_faab else "priority"
    # The "a losing claim is free" argument is exact. "Submit them all" is only exact
    # when they cannot BOTH land: ESPN re-processes the rest of your list at your new
    # (last) place in the queue after a successful claim, so two claims that drop
    # different players can execute in the same run. Each was priced against today's
    # roster on its own, and adds are submodular, so the pair is worth less than the sum.
    distinct = {_tag(r, "drop:") for r in claims}
    interaction = (
        " Every claim here drops the same player, so at most one of them can execute and "
        "the list really is a waterfall."
        if len(distinct) <= 1
        else f" Careful: these claims drop {len(distinct)} different players. After a "
        "successful claim ESPN processes the rest of your list at your new place at the "
        "back of the queue, so more than one can land in the same run -- and each was "
        "priced on its own against today's roster, so two together are worth less than "
        "the sum of the two. Submit the ones that share a drop, or accept the pair."
    )
    return (
        f"Submit all {len(claims)} claims as one conditional waterfall, best first: a "
        f"losing claim costs nothing, so submitting every candidate above the "
        f"{threshold * 100:.3f}pp threshold weakly dominates submitting one. Only a "
        f"successful claim spends {currency}.{interaction}{caveat} "
        f"{timing_note(acquisition, on_waivers=True)}"
    )


def _history(sim: LeagueSim, use_log: bool) -> TransactionLog | None:
    """This league's own transaction log, or `None` if it cannot be had.

    ESPN gates the log behind credentials and serves one scoring period per request, so
    it is fetched once and only when it will actually be used. A refusal is not a quiet
    zero: `contest_rate_from_log` and `arrival_rate_from_log` both fall back rather than
    reading "no claims" off an empty log, and this returns `None` so they are not called
    at all with a log that is not there.
    """
    if not use_log:
        return None
    try:
        history = sim.league.transactions()
    except Exception as err:  # pragma: no cover - live-only path
        log.info("no transaction log for league %s (%s)", sim.state.league_id, err)
        return None
    return history if getattr(history, "available", False) and len(history) else None


def _rank_from_teams(teams: object, team_id: int) -> int | None:
    for team in getattr(teams, "teams", ()):
        if getattr(team, "id", None) == team_id:
            return int(getattr(team, "waiver_rank", 0)) or None
    return None


def _priority_from_league(sim: LeagueSim, team_id: int) -> int | None:
    """This team's place in the waiver queue, or `None` if ESPN will not say.

    **`pipeline.build` closes the client it opened**, and `League.teams()` -- unlike
    `settings()` -- is not cached, so the obvious single call fails on every sim built
    the documented way: "Cannot send a request, as the client has been closed". That
    failure was silent and it disabled the entire optimal-stopping model, because the
    caller then fell back to the *worst* priority, whose continuation value is zero by
    construction, and every positive claim cleared a threshold of nothing. Measured on
    Wine Wednesday the user actually holds waiver priority **1**, worth 0.250pp to keep;
    the board was reporting "priority 14/14, threshold 0.000pp" and recommending a
    twelve-deep waterfall.

    So a closed client is retried once against a fresh one from the environment. If that
    also fails the caller is told `None` and must say so out loud rather than guessing --
    see `waiver_board`, which then assumes priority 1, the most expensive assumption.
    """
    try:
        return _rank_from_teams(sim.league.teams(), team_id)
    except Exception as err:
        log.info("could not read the waiver order (%s); retrying with a fresh client", err)

    league_id = getattr(sim.league, "league_id", None)
    season = getattr(sim.league, "season", None)
    if not isinstance(league_id, int) or not isinstance(season, int):
        log.warning("could not read the waiver order and there is no league to retry against")
        return None
    try:  # pragma: no cover - live-only path
        from ..espn.league import League
        from ..pipeline import client_from_env

        client = client_from_env()
        try:
            return _rank_from_teams(League(client, league_id, season).teams(), team_id)
        finally:
            client.close()
    except Exception as err:  # pragma: no cover - live-only path
        log.warning("could not read the waiver order: %s", err)
    return None


def _rival_gain_after_drop(
    engine: RosterSimulator, rival_id: int, add: int, add_only: float
) -> float:
    """What the player is worth to `rival_id` once the rival pays a roster spot for him.

    Never more than the add-only figure, and often much less: a rival carrying a full
    roster has to cut somebody, and the best cut is found by trying each of his players.
    Capped below at zero -- a rival who would lose by claiming simply does not claim, so
    there is nothing to block.
    """
    roster = engine.state.franchise(rival_id).player_ids
    if not roster:
        return max(add_only, 0.0)
    base = engine.season_points(roster, rival_id)
    best = float("-inf")
    for pid in roster:
        kept = (*(p for p in roster if p != pid), add)
        best = max(best, float((engine.season_points(kept, rival_id) - base).mean()))
    return max(min(best, add_only), 0.0)


def _blocking_board(
    engine: RosterSimulator,
    agents: Sequence[FreeAgent],
    team_id: int,
    limit: int,
    lever: float,
) -> tuple[Recommendation, ...]:
    """What denying the best free agents to the rival who wants them most is worth.

    The rival most helped is found on points, which is cheap, and his gain is converted
    through **his own** exchange rate rather than ours -- a contender and a team already
    out of it turn the same player into wildly different title probability, and using our
    rate for his roster is the same category error as a league-wide points-per-title
    constant.

    Priced through the rate rather than by re-running his bracket for exactly the reason
    `ClaimPrice` gives: the answer is `1/(N-1)` of a number that is already a fraction of
    a percentage point, so a direct bracket difference here is pure noise -- it came back
    NEGATIVE for a player who unambiguously helps, on the live board, before this was
    changed. They are ordered last and tagged, and the rationale says never to claim for
    this reason alone.

    **A rival's claim is an add AND a drop too.** Screening him on the add alone is the
    error this module's whole first paragraph exists to reject, and it inflates the block
    -- a rival's roster is as full as ours, so he pays for the player with whatever he
    cuts. The rival most helped is found on the add (cheap, and it never reorders the
    rivals by much because every roster is full), and the number that gets published is
    then re-priced with his own cheapest drop taken out.
    """
    state = engine.state
    rivals = [f.team_id for f in state.franchises if f.team_id != team_id]
    if not rivals:
        return ()
    out: list[Recommendation] = []
    for agent in agents[:limit]:
        best_gain, best_team = 0.0, rivals[0]
        for rid in rivals:
            roster = state.franchise(rid).player_ids
            gain = float(
                (
                    engine.season_points((*roster, agent.player_id), rid)
                    - engine.season_points(roster, rid)
                ).mean()
            )
            if gain > best_gain:
                best_gain, best_team = gain, rid
        best_gain = _rival_gain_after_drop(engine, best_team, agent.player_id, best_gain)
        if best_gain <= 0.0:
            continue
        rate, rate_se = engine.exchange_rate(best_team)
        rival_title = best_gain * rate
        rival_se = abs(best_gain) * rate_se
        mine = blocking_value(rival_title, state.size)
        out.append(
            Recommendation(
                move=claim_move(state.league_id, team_id, agent.player_id),
                delta_title=mine,
                delta_points=0.0,
                stderr=blocking_value(rival_se, state.size),
                leverage=lever,
                rationale=(
                    f"Blocking only. {agent.name} is worth {best_gain:+.1f} points and "
                    f"{rival_title * 100:+.2f}pp to {state.franchise(best_team).name}; denying "
                    f"him returns {mine * 100:+.3f}pp to us, 1/{state.size - 1} of his gain. "
                    "Never claim for this reason alone."
                ),
                confidence="low",
                tags=("waiver", "blocking", f"add:{agent.name}", f"rival:{best_team}"),
            )
        )
    out.sort(key=lambda r: -r.delta_title)
    return tuple(out)


def waiver_board(
    sim: LeagueSim,
    *,
    team_id: int | None = None,
    week: int | None = None,
    evaluator: MoveEvaluator | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    drops: int = 6,
    #: How many of the cheapest drops each add is paired against. 1 restores the old
    #: behaviour of committing to the points-best drop before title probability is known.
    drop_pairs: int = 3,
    confirm: int = 12,
    screen_sims: int = DEFAULT_SCREEN_SIMS,
    wire_depth: int = DEFAULT_WIRE_DEPTH,
    replacement: Mapping[int, float] | None = None,
    priority: int | None = None,
    uses_faab: bool | None = None,
    budget: int = 0,
    contest_rate: float = DEFAULT_CONTEST_RATE,
    arrival: float = 1.0,
    roster_limit: int | None = None,
    blocks: int = 3,
    faab_rivals: int = 1,
    use_log: bool = True,
    on_waivers: Iterable[int] | None = None,
    rankings: EtrRankings | None = None,
    rankings_weight: float = DEFAULT_BOARD_WEIGHT,
) -> WaiverReport:
    """Rank every plausible claim in one league by `delta_title`, and say whether to make it.

    Two tiers, as `core.MoveEvaluator` intends. The screen prices `delta_points` exactly
    -- common random numbers reduce that variance about a thousandfold -- on a small
    draw, and converts through the franchise's own measured points-to-title rate. The
    survivors are confirmed on the full draw, where the bracket is re-run and the
    standard error is real.

    `evaluator` drives the confirm tier and is injectable, so a caller can hand in
    `decide/title.py`'s engine (`evaluator=default_evaluator(...)`) or a cheap fake. When
    it is `None` the confirm tier is this module's own `RosterSimulator`, which publishes
    the plug-in and keeps the bracket beside it -- see `default_evaluator` for the
    measurement that says a bracket-ranked board at 4,000 simulations is a noise-ranked
    board, and `ClaimPrice` for why.

    `rankings` is an outside ranking set -- see `decide/opinion.py`. It is applied here
    and nowhere upstream: every surface in this repo reads `sim.outlooks`, so tilting them in
    `pipeline.build` would move the odds table, the streaming plan and the lineup advice
    along with the wire. This board's remit is the wire and the trade board, so this is
    where it enters. `board=None` or `board_weight=0.0` is exactly today's answer.

    The tilt lands on `free_agent_pool`'s ordering -- points above the wire's own depth
    at the position -- and then on everything `augment` rebuilds from the same outlooks,
    so the claim price, the FAAB bid and the continuation table all reprice off one
    change rather than several.
    """
    state = sim.state
    outlooks = _redealt(sim.outlooks, rankings, rankings_weight, state.weeks)
    team_id = team_id if team_id is not None else state.my_team_id
    if team_id is None:
        raise WaiverError("no team to advise: pass team_id or build the sim with my_team_id")
    week = week if week is not None else (state.weeks[0] if state.weeks else 0)

    settings = None
    try:
        settings = sim.league.settings()
    except Exception as err:  # pragma: no cover - live-only path
        log.info(
            "no settings for league %s (%s); using the arguments as given", state.league_id, err
        )
    acquisition = None if settings is None else settings.acquisition
    if settings is not None:
        uses_faab = settings.acquisition.uses_faab if uses_faab is None else uses_faab
        budget = budget or settings.acquisition.budget
        if roster_limit is None:
            roster_limit = settings.roster.starter_count + settings.roster.bench_slots
    uses_faab = bool(uses_faab)

    availability = _availability(sim) if on_waivers is None else None
    claimable = on_waivers if on_waivers is not None else getattr(availability, "on_waivers", None)
    agents = free_agent_pool(
        outlooks,
        _all_rostered(state),
        state.weeks,
        limit=candidates,
        wire_depth=wire_depth,
        on_waivers=claimable,
    )
    if not agents:
        raise WaiverError(f"league {state.league_id} has no projected free agents")
    wide = augment(
        sim,
        agents,
        screen_sims=screen_sims,
        replacement=replacement,
        wire_depth=wire_depth,
        outlooks=outlooks,
    )

    screener = RosterSimulator.build(wide.state, wide.screen_draw, team_id, replacement=wide.floor)
    confirmer = evaluator or RosterSimulator.build(
        wide.state, wide.draw, team_id, replacement=wide.floor
    )
    reporter = confirmer if isinstance(confirmer, RosterSimulator) else screener

    my_roster = list(screener.roster)
    if not my_roster:
        raise WaiverError(f"team {team_id} in league {state.league_id} rosters nobody")
    open_spot = roster_limit is not None and len(my_roster) < roster_limit
    lever, sd_diff = _matchup_leverage(reporter, week)

    # -- which of my own players are cheapest to lose ----------------------------------
    drop_cost = {pid: float(screener.marginal_points((), (pid,)).mean()) for pid in my_roster}
    droppable = sorted(my_roster, key=lambda p: -drop_cost[p])[: max(drops, 1)]

    # -- screen: every add on its own, then the best drop for the best adds ------------
    add_gain = {
        a.player_id: float(screener.marginal_points((a.player_id,), ()).mean()) for a in agents
    }
    # The shortlist is the UNION of two orderings, not just the first.
    #
    # `add_gain` prices an add with no drop, which is ~0 for anyone who does not
    # crack the current lineup -- so a bench receiver whose whole value is covering
    # an injury scores zero here and was cut before he could ever be paired. That is
    # how Josh Downs (ETR #92, the third-best body on the wire by the pool's own
    # ranking) failed to reach a board whose top rows were the fourth-best available
    # defense. `agents` is already ordered by points above the wire's depth at the
    # position, which is the ordering that knows a WR5 is not a D/ST2, so take the
    # top of both and let the paired confirm decide between them.
    width = max(confirm * 2, confirm)
    by_gain = sorted(agents, key=lambda a: -add_gain[a.player_id])[:width]
    seen_ids = {a.player_id for a in by_gain}
    shortlist = [*by_gain, *(a for a in agents[:width] if a.player_id not in seen_ids)]

    # Every add is paired with each plausible drop and ALL of them go to confirm,
    # rather than the points-best drop alone.
    #
    # The screen ranks on marginal POINTS and the board reports title probability,
    # and on a live board those disagree: dropping Makai Lemon for Josh Downs COSTS
    # 0.2 points and GAINS 0.267pp, because the pairing that loses expected points
    # can still raise the chance of a title through its effect on the spread. A
    # screen that committed to one drop per add therefore threw away the better
    # pairing before the objective that matters was ever computed.
    #
    # Points still order the queue -- it is the cheap signal and `confirm` is the
    # expensive one -- but the choice among drops is deferred to the confirm step.
    pairs: dict[int, list[tuple[FreeAgent, int | None, float]]] = {}
    for agent in shortlist:
        rows: list[tuple[FreeAgent, int | None, float]] = []
        if open_spot:
            rows.append((agent, None, add_gain[agent.player_id]))
        for pid in droppable[:drop_pairs]:
            gain = float(screener.marginal_points((agent.player_id,), (pid,)).mean())
            rows.append((agent, pid, gain))
        rows.sort(key=lambda row: -row[2])
        pairs[agent.player_id] = rows

    # EVERY shortlisted player gets at least one pairing through to confirm, and only
    # then does points ordering decide how the remaining budget is spent.
    #
    # Ranking pairs by points and truncating cut exactly the players this board exists
    # to find. Adding a bench receiver with no drop gains ~0 points because he does not
    # crack the current lineup, so all of Josh Downs' pairings sorted below the fourth-
    # best available defense and never reached the objective that would have shown him
    # at +0.17pp against a 0.12pp threshold. Points are the cheap screen; they are not
    # the thing being maximised, so they must not be the thing that decides who is
    # allowed to be measured.
    guaranteed = [rows[0] for rows in pairs.values() if rows]
    guaranteed.sort(key=lambda row: -row[2])
    extra = sorted((row for rows in pairs.values() for row in rows[1:]), key=lambda row: -row[2])
    # `confirm` bounds how many DISTINCT players are measured; the guaranteed set is
    # one pairing each, so the budget must clear it or the guarantee is undone by the
    # very next slice.
    budget = max(confirm * drop_pairs, confirm, len(guaranteed))
    screened = [*guaranteed, *extra][:budget]

    # -- confirm the survivors on the full draw ----------------------------------------
    names = {p: state.pool.name(p) for p in my_roster}
    # The board's own verdict on each pairing, as a rank comparison and nothing more.
    # It is computed from the board rather than read back out of the tilted numbers:
    # `delta_title` already carries the tilt, and a tag that merely restated it would
    # be a second view of one measurement dressed as corroboration.
    endorsed: dict[tuple[int, int], Upgrade] = {}
    if rankings is not None:
        wire_ids = [a.player_id for a in shortlist]
        for up in bench_upgrades(rankings, my_roster, wire_ids, names=names).upgrades:
            endorsed.setdefault((up.add, up.drop), up)
    moves = [
        claim_move(state.league_id, team_id, agent.player_id, drop_id)
        for agent, drop_id, _ in screened
    ]
    confirmed = confirmer.confirm(moves)
    recs: list[Recommendation] = []
    for (agent, drop_id, _), rec in zip(screened, confirmed, strict=True):
        recs.append(
            replace(
                rec,
                leverage=lever,
                # Appended, not replaced: the evaluator's own tags carry the bracket
                # cross-check that `price_of` reads back out. `waiver` is stamped here
                # rather than trusted from the evaluator, because an injected engine --
                # `decide/title.py`'s, say -- tags with its own vocabulary and the board's
                # own contract must not depend on which engine confirmed it.
                tags=(
                    *rec.tags,
                    *(() if "waiver" in rec.tags else ("waiver",)),
                    f"add:{agent.name}",
                    f"pos:{agent.position}",
                    f"drop:{names.get(drop_id, '-') if drop_id is not None else '-'}",
                    *_board_tags(endorsed.get((agent.player_id, drop_id))),
                    *_ceiling_tags(reporter, drop_id),
                ),
            )
        )
    # One row per player: several drops were confirmed for each add, and the board
    # should show the best of them rather than the same name three times.
    best_by_add: dict[int, Recommendation] = {}
    for rec in recs:
        add_id = next((p.player_id for p in rec.move.players if p.to_team is not None), None)
        if add_id is None:
            continue
        prior = best_by_add.get(add_id)
        if prior is None or rec.delta_title > prior.delta_title:
            best_by_add[add_id] = rec
    recs = sorted(best_by_add.values(), key=lambda r: -r.delta_title)

    # -- what the claim costs ----------------------------------------------------------
    played = max(week - 1, 0)
    history = _history(sim, use_log and played >= MIN_LOG_WEEKS)
    if history is not None:
        contest_rate = contest_rate_from_log(history, state.size, played)
        arrival = arrival_rate_from_log(history, played)
    streamable = stream_advantage(outlooks, state, _all_rostered(state))
    opportunity = OpportunityDistribution.from_board([r.delta_title for r in recs], arrival=arrival)
    table: ContinuationTable | None = None
    threshold = 0.0
    priority_known = True
    if not uses_faab:
        table = continuation_values(state.weeks, state.size, opportunity, contest_rate=contest_rate)
        if priority is None:
            read = _priority_from_league(sim, team_id)
            priority_known = read is not None
            # NOT `state.size`. The worst priority has a continuation value of zero by
            # construction, so guessing it turns "I could not read the waiver order" into
            # "spend the claim, it is free" -- the aggressive answer, on no evidence, and
            # the irreversible one: burning waiver priority 1 on a marginal claim cannot
            # be undone, while holding a claim you should have made can be made next week.
            # So an unknown queue position is assumed to be the FRONT of it.
            priority = read if read is not None else 1
        threshold = table.threshold(week, priority)

    # -- bids, rationales, and the waterfall -------------------------------------------
    rivals: RivalBids | None = None
    shadow = 0.0
    if uses_faab and budget > 0 and recs:
        rivals = LognormalRivalBids.population(budget)
        try:
            if history is None:
                raise WaiverError("no transaction log")
            rivals = LognormalRivalBids.from_history(
                [b for _, b in history.waiver_bids], rivals=faab_rivals
            )
        except WaiverError as err:
            log.info(
                "no usable bid history for league %s (%s); using the population prior",
                state.league_id,
                err,
            )
        shadow = float(
            faab_shadow_price(max(len(state.weeks), 1), budget, opportunity, rivals)[0, budget]
        )

    finished: list[Recommendation] = []
    for rec in recs:
        adds, drop_ids = _move_players(rec.move, team_id)
        agent = next(a for a in agents if a.player_id == adds[0])
        bid = (
            odd_dollars(
                first_order_bid(rec.delta_title, rivals, shadow_price=shadow, high=float(budget)),
                budget=budget,
            )
            if rivals is not None
            else None
        )
        finished.append(
            replace(
                rec,
                move=replace(rec.move, bid=bid),
                rationale=_rationale(
                    agent,
                    names.get(drop_ids[0], "-") if drop_ids else "-",
                    price_of(rec),
                    threshold,
                    None if uses_faab else priority,
                    bid,
                    lever,
                    priority_known=uses_faab or priority_known,
                    acquisition=acquisition,
                ),
            )
        )

    # The split this module existed without. A player ESPN has as FREEAGENT is
    # first-come and costs nothing, so the only question is whether he helps; a player
    # on WAIVERS costs a claim, and that claim has a continuation value he has to clear.
    # Charging the threshold to both is what suppressed 24 of 28 profitable rows on a
    # board where 809 of 841 available players were free.
    on_waivers_by_id = {a.player_id: a.on_waivers for a in agents}

    def _costs_a_claim(rec: Recommendation) -> bool:
        add_id = next((p.player_id for p in rec.move.players if p.to_team is not None), None)
        return on_waivers_by_id.get(add_id, True) if add_id is not None else True

    positive = [r for r in finished if r.delta_title > 0.0]
    claims = tuple(r for r in positive if _costs_a_claim(r) and r.delta_title >= threshold)
    free_adds = tuple(r for r in positive if not _costs_a_claim(r))
    hold = Recommendation(
        move=Move(kind=MoveKind.HOLD, league_id=state.league_id),
        delta_title=0.0,
        delta_points=0.0,
        stderr=0.0,
        leverage=lever,
        rationale=_hold_rationale(
            uses_faab,
            priority,
            threshold,
            finished,
            claims,
            free_adds,
            priority_known=uses_faab or priority_known,
            acquisition=acquisition,
        ),
        confidence="high",
        tags=("waiver", "hold"),
    )
    block_recs = _blocking_board(screener, agents, team_id, blocks, lever) if blocks else ()

    return WaiverReport(
        league_id=state.league_id,
        season=state.season,
        name=state.name,
        team_id=team_id,
        team_name=state.franchise(team_id).name,
        week=week,
        uses_faab=uses_faab,
        priority=None if uses_faab else priority,
        budget=budget,
        threshold=threshold,
        # The baseline comes from whoever confirmed the board, so an injected engine's
        # own view of the season is what gets reported next to its own deltas. The
        # exchange rate and the leverage are not on the protocol and come from the
        # local engine either way.
        baseline_title=confirmer.baseline_title(team_id),
        title_per_point=reporter.title_per_point,
        title_per_point_stderr=reporter.title_per_point_stderr,
        week_leverage=lever,
        sd_diff=sd_diff,
        continuation=table,
        board=tuple(finished),
        claims=claims,
        free_adds=free_adds,
        n_on_waivers=None if availability is None else availability.n_on_waivers,
        n_free_agents_available=None if availability is None else availability.n_free_agents,
        blocks=block_recs,
        hold=hold,
        stream_advantage=streamable,
        free_agents=agents,
        priority_known=uses_faab or priority_known,
    )


def cross_league_board(reports: Sequence[WaiverReport], limit: int = 10) -> list[Recommendation]:
    """Every league's claims in one ranking. The reason `delta_title` is the unit."""
    every = [r for report in reports for r in report.claims]
    every.sort(key=lambda r: -r.delta_title)
    return every[:limit]


__all__ = [
    "AugmentedSim",
    "ClaimPrice",
    "ContinuationTable",
    "FreeAgent",
    "LognormalRivalBids",
    "OpportunityDistribution",
    "RivalBids",
    "RosterSimulator",
    "UniformRivalBids",
    "WaiverError",
    "WaiverReport",
    "arrival_rate_from_log",
    "augment",
    "blocking_value",
    "claim_move",
    "contest_rate_from_log",
    "continuation_values",
    "cross_league_board",
    "default_evaluator",
    "faab_shadow_price",
    "first_order_bid",
    "free_agent_pool",
    "odd_dollars",
    "price_of",
    "waiver_board",
    "wire_floor",
]


def _redealt(
    outlooks: Sequence[PlayerOutlook],
    board: EtrRankings | None,
    weight: float,
    weeks: Sequence[int],
) -> Sequence[PlayerOutlook]:
    """The projection set this board runs on: ours, or ours re-dealt along a board.

    Returns the input object itself when there is nothing to do, so the no-board path
    is identical rather than merely equal.
    """
    if board is None or weight == 0.0:
        return outlooks
    return tilt_outlooks(outlooks, board, weight=weight, weeks=weeks)


def _board_tags(upgrade: Upgrade | None) -> tuple[str, ...]:
    """The board's own rank comparison for one claim, when it has an opinion.

    Deliberately separate from the price. `delta_title` on a tilted board already
    contains the board's view, so a tag derived from that number would be the same
    measurement twice. This is the raw ordering the analyst published -- checkable
    against his page, which is the whole use of it.
    """
    if upgrade is None:
        return ()
    return (
        f"board:{upgrade.add_rank}-over-{upgrade.drop_rank}",
        *((f"note:{upgrade.add_comment}",) if upgrade.add_comment else ()),
    )


def _ceiling_tags(engine: object, drop_id: int | None) -> tuple[str, ...]:
    """What the dropped player would have been worth to someone who knew.

    A points board charges a bench player what he adds to a lineup set on projections,
    which for a buried stash is zero. That is the right price and it is not the whole
    truth: measured at week 1 of 2026, dropping Jordyn Tyson costs -0.03 points and his
    hindsight ceiling is +16.26. The ceiling is reported and never charged -- the
    projection-optimal lineup already captures 0.886-0.900 of it while real managers
    capture 0.775, so nobody measured has beaten the projection and paying for the
    ceiling would price skill that has never been shown.

    Only when the gap is worth a sentence; every bench player has some gap.
    """
    cost = getattr(engine, "drop_cost", None)
    if drop_id is None or cost is None:
        return ()
    try:
        ex, ceiling = cost(drop_id)
    except Exception:  # pragma: no cover - never worth failing a board over
        return ()
    if ceiling - ex <= CEILING_WORTH_SAYING:
        return ()
    return (f"ceiling:{ex:+.1f}/{ceiling:+.1f}",)


def stream_advantage(
    outlooks: Sequence[PlayerOutlook],
    state: S.LeagueState,
    rostered: Iterable[int],
) -> dict[int, tuple[float, float, str]]:
    """`slot -> (streaming points, best rosterable points, that player)` per season.

    **The gap is non-negative by construction, so its sign proves nothing and only its
    size is information.** Streaming sums a per-week maximum and holding maximises a
    per-week sum, and `sum_w max_i >= max_i sum_w` for any table of numbers. Quoting the
    positive gap as evidence that streaming wins would be exactly the mistake this repo
    keeps finding: a defensible-looking number that is not the right quantity.

    What is informative is how the size varies by position, because that is a fact about
    week-to-week spread rather than about arithmetic. Measured at week 1 of 2026 on Type
    shi, over the remaining season:

      D/ST   stream 138.5   best held 100.1 (Chiefs D/ST)    +38.4
      QB     stream 251.0   best held 223.5 (C.J. Stroud)    +27.5
      TE     stream 120.6   best held 107.7 (Kenyon Sadiq)   +12.9
      K      stream 154.0   best held 140.8 (Cairo Santos)   +13.2

    D/ST is three times K and TE, and that is the measured reason a defense is the
    position everybody streams: its usable body changes week to week more than anything
    else on the roster. The other three leagues agree to within a point.

    None of this is achievable for free -- the streaming column assumes a transaction
    every week and pays nothing for it, so it is a ceiling and not a policy.
    `decide/streaming.py` plans the real thing, with acquisition costs and the
    single-slot constraint. This exists so the two surfaces cannot quietly disagree
    about the same seat, which is the defect `411ed47` fixed once already.

    Only slots whose eligible set is a single position and whose starting count is one.
    Everything else is a lineup, not a stream.
    """
    owned = {int(p) for p in rostered}
    out: dict[int, tuple[float, float, str]] = {}
    for slot, count in state.lineup_slot_counts.items():
        eligible = set(state.slot_eligibility.get(slot, ()))
        if count != 1 or len(eligible) != 1:
            continue
        pos = next(iter(eligible))
        free = [o for o in outlooks if o.position_id == pos and int(o.player_id) not in owned]
        if not free:
            continue
        stream = sum(
            max((o.weeks[w].mean for o in free if w in o.weeks), default=0.0)
            for w in state.weeks
        )
        best, who = max(
            ((sum(o.weeks[w].mean for w in state.weeks if w in o.weeks), o.name) for o in free),
            default=(0.0, ""),
        )
        out[slot] = (float(stream), float(best), who)
    return out
