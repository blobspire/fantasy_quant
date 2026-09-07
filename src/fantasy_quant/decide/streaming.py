"""Multi-week streaming for the slots you rent rather than own: D/ST, K, QB, TE.

A defense is not an asset, it is a *schedule*. Season best-minus-worst-starter VORP is
64.0 for D/ST and 20.0 for K against 224.7 for a receiver, and week-to-week the identity
of the good defense changes almost completely. So essentially all of the value in the
slot is in the weekly assignment, and none of it is in which defense sits on your bench
in September. That is a scheduling problem, and this module solves it as one.

**What was measured, and where it contradicts the brief.** Within-week regressions on
2022-2025 ESPN weekly projections joined to nflverse closing lines (`MATCHUP_MODELS`),
demeaned by (season, week) so the fit describes exactly the choice a streamer makes --
who to start *this* week -- rather than the between-week noise:

    position   n     sd(predictable spread)   R^2 with market   R^2 projection only
    TE       3741            3.49                0.307               0.304
    QB       2158            3.16                0.158               0.142
    D/ST     2132            2.30                0.134               0.111
    K        2166            0.68                0.022               0.013

Three things fall out of that table and all three matter:

1. **Kickers are not streamable.** Among startable kickers (projection > 2) the whole
   apparatus predicts 0.68 points of weekly spread, and the implied team total -- the
   thing everyone streams kickers on -- carries R^2 = 0.006 on its own. The brief's
   ranking "K > DST > TE > QB" is exactly backwards at the K end: K is last, by a
   factor of 3.4 against the next-worst position and 5 against the best. This module
   will still solve for K, and will report that the answer is noise, because a surface
   that dresses up 0.68 points as a recommendation is worse than no surface.
2. **ESPN's D/ST projection is under-reactive to the matchup by about 2x and
   under-dispersed by about 1.4x.** ESPN's projection moves -0.33 points per point of
   implied opponent total where the *actual* points move -0.68, and regressing actual on
   the projection alone gives a slope of 1.44 rather than 1.0. So the implied opponent
   total beats the projection outright within a week (R^2 0.159 against 0.138 on
   2024-25), and blending both is worth +1.33 points a week over ESPN's projection alone
   in a leave-one-season-out backtest of "start the best of all 32": +5.10 points/week
   over an average defense against +3.76.
3. **The Vegas term is D/ST-specific.** For QB it is the *team* implied total (+0.42
   points per point) and it adds real signal; for TE and K it is indistinguishable from
   zero and is set to zero here rather than fitted noise.

**A drop is final, and that is the single most consequential thing in this module.**
Marking your own players acquirable in every week -- "he is mine, so I can have him
back" -- turns every roster spot into a free option, and with zero acquisition cost the
solver takes it. On the live grids that produced *drop Jalen Hurts in week 10, re-add
him in week 11* and *drop Colston Loveland in week 10, re-add him in week 11*: not
optimistic advice, unexecutable advice, since a top-five quarterback does not sit on the
wire of a fourteen-team league for seven days. `build_grid(readd_dropped=False)` is the
default and says the honest thing instead -- hold him as long as you like, but once you
drop him he is gone. It costs 1.2 of Wine Wednesday's 43.5 D/ST points and 2.5 of Type
shi's 41.7, and it turns the QB and TE plans from that into zero transactions, which is
the correct answer for a roster whose incumbent beats every free agent in every week.

**Which optimisation problem this actually is.** With one streamed slot, free
acquisition and each streamer usable once, it is a rectangular linear assignment problem
-- totally unimodular, exact in O(n^3) from `scipy.optimize.linear_sum_assignment`. That
is `relax_assignment`. It is a **lower** bound on the *reuse-permitting* streaming IP
(`readd_dropped=True`), because forbidding reuse is a restriction there and not a
relaxation. Under the conservative default it is neither bound nor plan but a
diagnostic: an assignment that starts your incumbent in week five and somebody else in
weeks one to four is infeasible once a drop is final, and on 400 random no-re-add
instances the LAP came out above the true optimum 67 times, by up to 7.1 points. On the
real grids it stays well below it, which is worth reporting rather than assuming.
The **upper** bound is `upper_bound`, the per-week argmax over everyone you could have
acquired by then, which is what an unlimited-reuse zero-cost planner would score.

The honest problem couples the weeks through the roster spot, and the brief's IP is the
right statement of it. Two things about it turned out to be wrong in the brief and both
changed what got built.

**There is a solver.** scipy ships HiGHS as `scipy.optimize.milp`, so the IP can simply
be written out and solved -- `_solve_milp` does exactly that, ~1,000 variables and
~1,000 rows for a 20-candidate 17-week instance, proven optimal in about 10 ms at any
`kappa`. It is the default above one roster slot and the independent check on everything
below it.

**And it does not need a branch-and-bound either.** With `kappa` streamer roster slots
the state is *which streamers you are holding*, and a dynamic program over those states
solves the IP **exactly**, because the flow constraints are precisely a shortest path
through that state graph. For the case that actually occurs -- kappa = 1, one defense on
the roster, swap it whenever it is worth swapping -- the recursion collapses to O(n) per
week:

    V_w(i) = max( f_w(i),  max_j [ f_w(j) - acq_jw ] ),   f_w(j) = c_jw + V_{w+1}(j)

so the full 32-candidate, 17-week instance is solved to optimality in about 3.6
milliseconds -- three times faster than the MILP, and agreeing with it to 1e-6 on every
real-scale instance tested. `solve` reports `optimal=True` only when the answer is
proven: the DP at `kappa = 1`, HiGHS when it reports a proven optimum, and False for the
pruned `method="dp"` path above one roster slot -- which is not idle bookkeeping, since
on six random 20x17 instances that pruned DP took 4.7 seconds and came back up to 5.4
points (3%) short of HiGHS on three of them. `brute_force` re-derives both by exhaustion
on small instances. On the user's real week-1 D/ST grid the four solvers rank
`hold 81.5 <= LAP 116.3 <= exact 124.9 <= per-week argmax 126.1`, and the residual 1.2
is precisely the option value the default declines to spend: an unlimited-reuse planner
would re-sign the Jaguars twice, this one will not.

One correctness note that the exhaustion did *not* catch, because it shared the bug.
`_reward` used to read `grid.floor` only when nothing at all was startable, so a held
candidate worth less than the replacement level was started rather than benched. With
`floor > 0` and a claim priced above zero that left the DP up to three points short of
HiGHS on 33 of 300 random instances -- while still reporting `optimal=True`, and while
`brute_force` agreed with it to the last bit, since both called the same function. Two
implementations of one mistake agreeing is not a proof, and it is the reason the checks
in the test module now run against HiGHS as well as against exhaustion.

`optimality_gap` measures what the rolling-horizon heuristic loses rather than asserting
it loses nothing. On Wine Wednesday's live D/ST grid, in points of a ~125-point
season objective:

    acquisition cost   horizon 1   horizon 2   horizon 3   horizon 4   horizon 6
    0 points             0.000       0.000       0.000       0.000       0.000
    1 point              0.302       0.000       0.000       0.000       0.000
    2 points             3.950       0.415       0.269       0.000       0.000
    3 points             4.620       5.143       4.295       0.713       0.000

At zero cost the myopic plan is exactly optimal -- greedy and optimal coincide when
nothing couples the weeks -- and the horizon only starts to matter once a claim costs
something. Note that four weeks of lookahead is no longer *exactly* optimal at three
points a claim (0.713 short), which is the sort of thing a measured gap tells you and an
asserted one does not; six weeks was optimal on every live instance tested.

**Rolling horizon is how it gets used.** `rolling_plan` optimises the remaining season,
commits only week t, then re-solves. With the projections frozen that is identical to
the full-horizon solve (there is a test pinning it), so the value of re-solving is
entirely informational: **160 of the 272 2026 games had no closing line on 2026-09-07**,
so from week 7 on the market term is simply unavailable today and the grid falls back to
the projection-only model. It becomes available a week at a time, which is the whole
reason to re-solve rather than to commit a season-long plan in September.

**What it says about the user's actual teams, and how much of that to believe.** All
three teams hold a below-average defense with an uncovered week-7 bye, and the
rest-of-season D/ST plan is worth +39.2 to +43.5 model points -- about 2.4 a week -- for
`+2.0pp` to `+3.8pp` of championship probability at 4,000 paired simulations, each
several standard errors clear of zero. Five things bound that claim, and all five are
reported rather than buried:

* Only 13.4-14.9 of those points sit in weeks a bookmaker has priced. `through_week=6`
  prices just those and gives `+0.45pp +/- 0.21`, which clears two standard errors by a
  hair and no more. That is the honest floor, and it is a floor rather than an estimate.
* 7.1 points of it is covering the bye, which needs no model at all.
* Running the simulator on ESPN's own calibrated means instead of the matchup model
  (`apply_matchup_model=False`) still gives `+1.90pp` and +30.7 points, so roughly three
  quarters of the effect survives disbelieving the model's magnitude entirely.
* **The action recommended today is a hold in every one of these cases.** `delta_title`
  prices the whole seventeen-week plan; the week-one move is worth `commit_delta`, which
  the rationale quotes and which is exactly 0.00pp here. A reader who takes the headline
  as the value of `Recommendation.move` is reading it wrong, and the
  `no-action-this-week` tag exists to stop that.
* An independent check: Wine Wednesday's own cross-section of title odds against
  expected wins has a slope of 5.64pp per win (R^2 0.83). 2.56 points a week is 0.42
  extra wins at the measured 1.16pp/point, which predicts +2.34pp -- against the
  simulator's +2.77pp, by a route that shares no code with it.

**And the kicker result, which is the one worth acting on.** The same machinery run on K
finds +9.3 to +14.0 points, of which **+8.1 to +8.4 is simply covering the kicker's bye
week**. The residual matchup edge is 0.9-5.7 points over sixteen weeks, well under half a
point a week, against a fitted within-week spread of 0.68. So the correct kicker advice
is "cover the bye and otherwise stop thinking about it", and `Recommendation.tags`
carries `not-streamable` with `confidence="low"` to say so. `as_recommendation` goes one
step further and *suppresses* a week-one kicker transaction that is not a bye cover:
left to itself the Type shi grid emitted "drop Evan McPherson for Jake Elliott" on a
0.02-point edge at R^2 = 0.022, priced off the season plan at "+1.00pp, significant",
while the simulated value of making that swap and keeping it was -0.27pp. A position the
module has already declared unstreamable must not then hand out transactions in it.

**QB and TE, run as a check, correctly do nothing.** With a final drop, no free agent at
either position beats the incumbent in any week on any of the three rosters, so the plan
is byte-identical to holding, `delta_points` is exactly +0.0, and the recommendation
carries `null-plan`. That tag matters because a plan identical to its baseline has a
paired difference of exactly zero and therefore `stderr == 0.0`, which
`Recommendation.significant` reads as "no Monte Carlo error, hence significant". It is
the opposite: nothing was measured.

**Which acquisition regime we are in.** All three of the user's leagues run rolling
waiver priority, not FAAB, so there is no bid to optimise -- and this was checked
against the live rosters rather than assumed: a 14-team league rosters 15 of the 32
defenses, so 17 are plain free agents that cost nothing at all, not even a claim. The
default is therefore `acquisition_cost=0.0` with `kappa=1`: the binding constraint on
D/ST and K streaming is the single roster slot, not money and not priority.

`acquisition_cost` is the seam for the other regime, and it is in POINTS. Where a
streamer really is on waivers, `decide/waivers.py` prices the priority slot as
`ContinuationTable.threshold(week, priority)` -- but that is in *championship
probability*, so it has to be divided by this league's title-per-point slope before it
comes in here (`Recommendation.delta_title / delta_points` off any confirmed
recommendation is the local estimate of that slope). In a FAAB league pass
`lambda * price` instead. It is deliberately not imported: the conversion is a
league-level quantity the caller already has, and a hard dependency would tie the
streaming solver to a bidding model it does not otherwise need. `max_acquisitions` is
the cardinality form of the same constraint, for a league that meters claims.

The sensitivity is worth knowing before choosing a number. On Wine Wednesday's live
D/ST grid the plan gains +43.5 points with 15 claims at zero cost, +28.9 with 12 at one
point a claim, **+19.2 with 8 at two points a claim**, +8.8 with 4 at four, and nothing
at all above about eight. The edge is real but it is not free of churn: a manager who
will make four moves all season still collects a fifth of it, and one who will make
eight collects nearly half.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import csr_matrix, lil_matrix

from ..core import DST, QB, TE, K, Move, MoveKind, PlayerMove, PlayerOutlook, Recommendation
from ..core import leverage as _leverage
from ..data import nflverse
from ..data.ids import DST_BY_NFLVERSE, team_from_pro_team_id
from ..projections.calibration import CalibrationSet
from ..projections.calibration import load as load_calibration
from ..sim import season as S
from ..sim.distributions import WeeklySampler

log = logging.getLogger(__name__)


class StreamingError(ValueError):
    """The streaming instance is malformed or too large to solve as asked."""


# --------------------------------------------------------------------------------------
# The measured matchup model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchupModel:
    """Within-week coefficients turning a projection and a betting line into a forecast.

    Everything is expressed as a deviation from the *weekly* mean of the candidate pool,
    because that is the form the coefficients were fitted in and because it makes the
    model invariant to the level bias in ESPN's projections (measured at +0.79 points a
    week for D/ST, which cancels out of every difference this module reports anyway).

        c_iw = mean_w(mu) + proj_coef * (mu_iw - mean_w(mu))
                          + team_total_coef * (T_iw - mean_w(T))
                          + opp_total_coef  * (O_iw - mean_w(O))

    with `T`/`O` the Vegas implied team and opponent totals. When no line is posted the
    market terms drop and `proj_only_coef` replaces `proj_coef` -- they are different
    numbers (D/ST: 1.44 alone, 0.52 beside the market) because the projection is partly
    a noisy restatement of the line, and dropping the line without re-fitting the
    projection coefficient would systematically under-react.

    Fitted by OLS on ESPN weekly projections joined to `nflverse` closing spreads and
    totals, seasons 2022-2025, projections above `min_projection` only. The coefficients
    are stable across disjoint season pairs (D/ST joint fit: 2022-23 gives 0.39/-0.42,
    2024-25 gives 0.60/-0.49), which is the reason to trust them at all.
    """

    position_id: int
    label: str
    proj_coef: float
    team_total_coef: float
    opp_total_coef: float
    proj_only_coef: float
    #: Variance of the *actual* explained within-week, market form and projection-only.
    r2: float
    r2_proj_only: float
    #: SD of the fitted value within a week -- the points of spread actually harvestable.
    fitted_sd: float
    #: Player-weeks the fit ran on, and the projection floor it ran above.
    n: int
    min_projection: float
    #: Mean projection in the fitting sample, in ESPN's own default PPR scoring. The
    #: market coefficients are in points-per-point, so they have to be rescaled when a
    #: league scores this position on a different scale; this is the denominator.
    fit_mean_projection: float
    #: mean(actual) - mean(projection) in the fit. Reported, never applied: it is a
    #: level, and every number this module returns is a difference.
    level_bias: float

    @property
    def uses_market(self) -> bool:
        """False when the fitted market terms were indistinguishable from zero."""
        return self.team_total_coef != 0.0 or self.opp_total_coef != 0.0


#: Fitted on 2022-2025. See the module docstring for the table and what it contradicts.
#:
#: TE and K carry zeroed market coefficients on purpose: their fitted values were
#: +0.083/-0.026 and +0.105/-0.073 points per point of implied total, worth 0.003 and
#: 0.009 of R^2, which is fitted noise rather than signal. Writing the noise down and
#: then shipping it is how a model acquires a matchup story it cannot support.
MATCHUP_MODELS: Mapping[int, MatchupModel] = MappingProxyType(
    {
        DST: MatchupModel(
            position_id=DST,
            label="D/ST",
            proj_coef=0.516,
            team_total_coef=0.0,
            opp_total_coef=-0.443,
            proj_only_coef=1.440,
            r2=0.134,
            r2_proj_only=0.111,
            fitted_sd=2.302,
            n=2132,
            min_projection=0.0,
            fit_mean_projection=5.20,
            level_bias=0.79,
        ),
        QB: MatchupModel(
            position_id=QB,
            label="QB",
            proj_coef=0.605,
            team_total_coef=0.422,
            opp_total_coef=0.085,
            proj_only_coef=0.936,
            r2=0.158,
            r2_proj_only=0.142,
            fitted_sd=3.157,
            n=2158,
            min_projection=2.0,
            fit_mean_projection=15.9,
            level_bias=-0.2,
        ),
        TE: MatchupModel(
            position_id=TE,
            label="TE",
            proj_coef=0.996,
            team_total_coef=0.0,
            opp_total_coef=0.0,
            proj_only_coef=1.009,
            r2=0.307,
            r2_proj_only=0.304,
            fitted_sd=3.491,
            n=3741,
            min_projection=2.0,
            fit_mean_projection=6.2,
            level_bias=0.1,
        ),
        K: MatchupModel(
            position_id=K,
            label="K",
            proj_coef=0.320,
            team_total_coef=0.0,
            opp_total_coef=0.0,
            proj_only_coef=0.865,
            r2=0.022,
            r2_proj_only=0.013,
            fitted_sd=0.680,
            n=2166,
            min_projection=2.0,
            fit_mean_projection=6.7,
            level_bias=0.28,
        ),
    }
)

#: Below this much predictable within-week spread, a streaming recommendation is a
#: rounding error dressed as advice. K's 0.68 sits under it and is labelled accordingly.
STREAMABLE_SD = 1.5

#: Fraction of a week's candidates that must carry a posted line before the market form
#: is used for that week. Lines arrive per game-week, all at once or not at all, so this
#: is effectively a switch: 16/16 games priced, or 0/16.
MARKET_COVERAGE = 0.9

#: The lineup slot each streamed position occupies, for building the Move. Slot ids, not
#: position ids -- the two spaces collide at 4 and 15.
POSITION_SLOT: Mapping[int, int] = MappingProxyType({QB: 0, TE: 6, DST: 16, K: 17})


# --------------------------------------------------------------------------------------
# Market and schedule inputs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketSchedule:
    """Implied totals and byes per (nflverse team, week) for one season.

    Built from `nflverse.team_weeks`, which already re-expresses the home-relative
    `spread_line` from each team's own point of view. A null total is "not priced yet",
    never a pick'em: on 2026-09-07 only weeks 1-5 of the 2026 season had closing lines,
    so the honest thing to do with week 9 is to fall back to the projection-only model
    rather than to invent a 44.5.
    """

    season: int
    team_total: Mapping[tuple[str, int], float]
    opponent_total: Mapping[tuple[str, int], float]
    #: (team, week) pairs with an actual game. Absence is a bye.
    plays: frozenset[tuple[str, int]]
    weeks: tuple[int, ...]

    @property
    def priced_weeks(self) -> tuple[int, ...]:
        weeks = {w for (_, w) in self.team_total}
        return tuple(sorted(weeks))

    def bye_by_pro_team_id(self) -> dict[int, int]:
        """proTeamId -> bye week, in the form `SimPanel.from_outlooks` wants.

        Derived from the schedule and never from `player.proTeamId`, which is the
        player's *current* team and so gives a traded player the wrong bye.
        """
        out: dict[int, int] = {}
        for team, defense in DST_BY_NFLVERSE.items():
            idle = [w for w in self.weeks if (team, w) not in self.plays]
            if idle:
                out[defense.pro_team_id] = idle[0]
        return out


def market_schedule(
    season: int,
    *,
    schedule: pl.DataFrame | None = None,
    cache: nflverse.NflverseCache | None = None,
) -> MarketSchedule:
    """Load the season's implied totals and byes."""
    frame = nflverse.schedules([season], cache=cache) if schedule is None else schedule
    weeks = nflverse.team_weeks(frame, season)
    team_total: dict[tuple[str, int], float] = {}
    opp_total: dict[tuple[str, int], float] = {}
    plays: set[tuple[str, int]] = set()
    for row in weeks.iter_rows(named=True):
        key = (str(row["team"]), int(row["week"]))
        if not row["is_bye"]:
            plays.add(key)
        tt, ot = row.get("team_implied_total"), row.get("opponent_implied_total")
        if tt is not None and ot is not None:
            team_total[key] = float(tt)
            opp_total[key] = float(ot)
    return MarketSchedule(
        season=season,
        team_total=team_total,
        opponent_total=opp_total,
        plays=frozenset(plays),
        weeks=tuple(sorted({int(w) for w in weeks["week"].to_list()})),
    )


# --------------------------------------------------------------------------------------
# The streaming grid
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Streamer:
    """One candidate for the streamed slot."""

    player_id: int
    name: str
    position_id: int
    pro_team_id: int
    #: nflverse team abbrev, or "" when the player has no NFL team.
    team: str
    #: The fantasy team rostering him, None when he is a free agent.
    owner: int | None
    mine: bool


@dataclass(frozen=True, slots=True)
class StreamGrid:
    """`(candidate, week)` matchup values plus who may be started or acquired when.

    The three masks are genuinely different questions and conflating any two of them is
    how a plan comes back starting a defense on its bye:

    * `playing` -- has an NFL game that week. A bye is never startable.
    * `available` -- may be *acquired* that week. A defense on a rival's roster is not
      available at any price this module can pay, and by default neither is one you
      dropped: see `build_grid(readd_dropped=...)` for why that is not the same
      question as "is he yours today".
    * `held` -- on your roster right now, which is the DP's initial state. Holding is
      never gated on `available`; only *acquiring* is.
    """

    league_id: int
    season: int
    position_id: int
    my_team_id: int
    model: MatchupModel
    weeks: tuple[int, ...]
    streamers: tuple[Streamer, ...]
    #: (n, W) expected points, already through the matchup model.
    value: np.ndarray
    #: (n, W) the raw calibrated projection the value was built from, kept for reporting.
    projection: np.ndarray
    playing: np.ndarray
    available: np.ndarray
    held: np.ndarray
    #: Weeks where the market form was used rather than the projection-only fallback.
    priced: tuple[bool, ...]
    #: What an empty streamed slot scores, per week. Zero unless a caller supplies a
    #: replacement level -- an unfilled slot really does score nothing.
    floor: np.ndarray

    def __post_init__(self) -> None:
        n, w = len(self.streamers), len(self.weeks)
        for name in ("value", "projection", "playing", "available"):
            got = getattr(self, name).shape
            if got != (n, w):
                raise StreamingError(f"{name} has shape {got}, expected {(n, w)}")
        if self.held.shape != (n,):
            raise StreamingError(f"held has shape {self.held.shape}, expected {(n,)}")
        if len(self.priced) != w:
            raise StreamingError(f"priced has {len(self.priced)} entries, expected {w}")
        if self.floor.shape != (w,):
            raise StreamingError(f"floor has shape {self.floor.shape}, expected {(w,)}")

    @property
    def n(self) -> int:
        return len(self.streamers)

    @property
    def n_weeks(self) -> int:
        return len(self.weeks)

    @property
    def held_index(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.flatnonzero(self.held))

    def index_of(self, player_id: int) -> int:
        for i, s in enumerate(self.streamers):
            if s.player_id == player_id:
                return i
        raise KeyError(f"player {player_id} is not a candidate in this grid")

    def acquirable(self) -> np.ndarray:
        """(n, W) -- could be BOTH signed and started that week: has a game, and is
        either already yours or on the wire. This is the mask for the one-shot models
        (the LAP relaxation and the per-week argmax bound), which sign whoever they
        start in the week they start him."""
        return self.playing & (self.available | self.held[:, None])

    def scored(self) -> np.ndarray:
        """Value with bye cells driven to -inf: what a streamer is worth *if you hold
        him*. Deliberately not masked by `available` -- a defense you signed in week 3
        is yours in week 5 whether or not he was still on the wire then, and folding
        availability in here is how a stash-before-the-bye silently becomes infeasible.
        Acquisition is a constraint on the roster flow, and the solvers enforce it
        there."""
        return np.where(self.playing, self.value, -np.inf).astype(np.float64)


def _week_market(
    grid_teams: Sequence[str],
    week: int,
    market: MarketSchedule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(team_total, opponent_total, has_line) for one week, aligned to `grid_teams`."""
    n = len(grid_teams)
    tt = np.zeros(n)
    ot = np.zeros(n)
    has = np.zeros(n, dtype=bool)
    for i, team in enumerate(grid_teams):
        key = (team, week)
        if key in market.team_total:
            tt[i] = market.team_total[key]
            ot[i] = market.opponent_total[key]
            has[i] = True
    return tt, ot, has


def build_grid(
    outlooks: Iterable[PlayerOutlook],
    *,
    league_id: int,
    season: int,
    position_id: int,
    weeks: Sequence[int],
    ownership: Mapping[int, int],
    my_team_id: int,
    market: MarketSchedule,
    model: MatchupModel | None = None,
    include_rostered: bool = False,
    min_projection: float | None = None,
    floor: float | Sequence[float] = 0.0,
    readd_dropped: bool = False,
) -> StreamGrid:
    """Compile the candidate x week matchup grid for one streamed position.

    `ownership` maps playerId -> fantasy teamId; anyone absent is a free agent. Players
    rostered by a *rival* are dropped by default rather than carried as unavailable
    columns, because a solver that can never pick them only pays for them in runtime --
    pass `include_rostered=True` when you want to see what the league is sitting on.

    `readd_dropped` is the single most consequential switch here and it defaults to
    **False**, which reverses what an earlier version of this module did. Marking your
    own players available in every week says "a player I drop stays mine", and with free
    acquisition the solver then treats every roster spot as a free option: on the live
    week-1 grids it produced *drop Jalen Hurts in week 10, re-add him in week 11* and
    *drop Colston Loveland in week 10, re-add him in week 11* -- plans that are not
    merely optimistic but unexecutable, since a top-five quarterback does not survive a
    week on the wire of a fourteen-team league. Holding is never gated on `available`,
    so the honest statement is the conservative one: keep him for as long as you like,
    but a drop is final. It costs 1.2 of the 43.5 model points on Wine Wednesday's D/ST
    grid and 2.5 of 40.9 on Type shi's, which is the price of not shipping that advice.
    Pass `readd_dropped=True` where re-acquisition really is safe -- a streamed D/ST in a
    shallow league -- and read the difference as the option value of the wire.

    The value is the matchup model's conditional expectation of actual points, which is
    deliberately NOT the projection: for D/ST the projection's within-week deviation is
    multiplied by 0.52 and the market does the rest of the work, so the grid disagrees
    with ESPN by design and by a measured margin.
    """
    model = model or MATCHUP_MODELS.get(position_id)
    if model is None:
        raise StreamingError(
            f"no fitted matchup model for position {position_id}; "
            f"models exist for {sorted(MATCHUP_MODELS)}"
        )
    weeks = tuple(int(w) for w in weeks)
    if not weeks:
        raise StreamingError("a streaming grid needs at least one week")
    threshold = model.min_projection if min_projection is None else float(min_projection)

    rows = [o for o in outlooks if o.position_id == position_id]
    # The scoring scale of the *position in this league*, measured over every player at
    # it who has a game -- not over the candidate pool. Those are different quantities
    # and conflating them is a real bug: the pool excludes whatever rivals have
    # rostered, so on Wine Wednesday's D/ST grid the pool mean is 4.28 against 4.91 for
    # all thirty-two defenses, and the market coefficient was being shrunk by 18% for no
    # reason but that the good defenses were taken. A nuisance parameter meant to carry
    # "this league scores D/ST differently" must not move when a rival makes a claim.
    scale_means = [
        max(wo.mean, 0.0)
        for o in rows
        for w in weeks
        if (wo := o.weeks.get(w)) is not None
        and wo.playing
        and ((team_from_pro_team_id(o.pro_team_id) or ""), w) in market.plays
    ]
    keep: list[tuple[Streamer, PlayerOutlook]] = []
    for o in rows:
        if not all(w in o.weeks for w in weeks):
            continue
        owner = ownership.get(o.player_id)
        mine = owner == my_team_id
        if owner is not None and not mine and not include_rostered:
            continue
        best = max(o.weeks[w].mean for w in weeks)
        if best < threshold and not mine:
            continue
        team = team_from_pro_team_id(o.pro_team_id) or ""
        keep.append(
            (
                Streamer(
                    player_id=o.player_id,
                    name=o.name,
                    position_id=position_id,
                    pro_team_id=o.pro_team_id,
                    team=team,
                    owner=owner,
                    mine=mine,
                ),
                o,
            )
        )
    if not keep:
        raise StreamingError(
            f"no candidates at position {position_id} with an outlook for weeks {weeks}"
        )
    keep.sort(key=lambda pair: -max(pair[1].weeks[w].mean for w in weeks))
    streamers = tuple(s for s, _ in keep)
    teams = [s.team for s in streamers]

    n, n_w = len(streamers), len(weeks)
    projection = np.zeros((n, n_w))
    playing = np.zeros((n, n_w), dtype=bool)
    for i, (s, o) in enumerate(keep):
        for j, w in enumerate(weeks):
            wo = o.weeks[w]
            projection[i, j] = max(wo.mean, 0.0)
            playing[i, j] = bool(wo.playing) and (s.team, w) in market.plays

    # The market coefficients are points per point of implied total, so they carry the
    # scoring scale of the fitting sample. Rescale by the ratio of mean projections and
    # clamp it: a league with wild D/ST bonuses should move the term, a thin week of
    # data should not.
    live_mean = float(np.mean(scale_means)) if scale_means else model.fit_mean_projection
    scale = float(np.clip(live_mean / max(model.fit_mean_projection, 1e-6), 0.5, 2.0))

    value = np.zeros((n, n_w))
    priced: list[bool] = []
    for j, w in enumerate(weeks):
        live = playing[:, j]
        if not live.any():
            priced.append(False)
            continue
        mu = projection[:, j]
        bar = float(mu[live].mean())
        tt, ot, has_line = _week_market(teams, w, market)
        coverage = float(has_line[live].mean())
        use_market = model.uses_market and coverage >= MARKET_COVERAGE
        priced.append(use_market)
        if use_market:
            bar_t = float(tt[live & has_line].mean())
            bar_o = float(ot[live & has_line].mean())
            market_term = scale * (
                model.team_total_coef * np.where(has_line, tt - bar_t, 0.0)
                + model.opp_total_coef * np.where(has_line, ot - bar_o, 0.0)
            )
            value[:, j] = bar + model.proj_coef * (mu - bar) + market_term
        else:
            value[:, j] = bar + model.proj_only_coef * (mu - bar)
    value = np.maximum(value, 0.0)

    available = np.zeros((n, n_w), dtype=bool)
    held = np.zeros(n, dtype=bool)
    for i, s in enumerate(streamers):
        held[i] = s.mine
        # `mine` means "held today", which is the DP's initial state -- not a standing
        # licence to re-sign him after dropping him. See `readd_dropped`.
        available[i, :] = s.owner is None or (s.mine and readd_dropped)

    floor_vec = np.full(n_w, float(floor)) if np.isscalar(floor) else np.asarray(floor, dtype=float)
    return StreamGrid(
        league_id=league_id,
        season=season,
        position_id=position_id,
        my_team_id=my_team_id,
        model=model,
        weeks=weeks,
        streamers=streamers,
        value=value,
        projection=projection,
        playing=playing,
        available=available,
        held=held,
        priced=tuple(priced),
        floor=floor_vec,
    )


# --------------------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """A week-by-week roster and start decision for one streamed slot.

    `start[j]` is the index into `grid.streamers` started in `weeks[j]`, or -1 when the
    slot is left empty (every candidate on a bye, or nothing acquirable). `roster[j]` is
    what is held going into that week, *after* the week's transactions.
    """

    weeks: tuple[int, ...]
    start: tuple[int, ...]
    roster: tuple[tuple[int, ...], ...]
    acquired: tuple[tuple[int, ...], ...]
    dropped: tuple[tuple[int, ...], ...]
    #: Sum of the started values, before any acquisition or roster-slot cost.
    points: float
    #: Total acquisition plus extra-roster-slot cost charged by the objective.
    cost: float
    method: str
    #: True only when the whole feasible state space was enumerated.
    optimal: bool
    #: The zero-cost unlimited-reuse per-week argmax. An upper bound on any plan.
    bound: float

    @property
    def value(self) -> float:
        """The objective: points harvested minus what they cost."""
        return self.points - self.cost

    @property
    def gap(self) -> float:
        """Points per season between this plan and the upper bound. Never negative."""
        return max(self.bound - self.value, 0.0)

    @property
    def n_acquisitions(self) -> int:
        return sum(len(a) for a in self.acquired)

    def started_ids(self, grid: StreamGrid) -> tuple[int | None, ...]:
        return tuple(grid.streamers[i].player_id if i >= 0 else None for i in self.start)

    def week_value(self, grid: StreamGrid) -> np.ndarray:
        """(W,) value harvested each week, floor included."""
        out = np.array(
            [grid.value[i, j] if i >= 0 else grid.floor[j] for j, i in enumerate(self.start)]
        )
        return out

    def value_split(self, grid: StreamGrid, baseline: StreamPlan | None = None) -> ValueSplit:
        """Where the gain over holding actually comes from. Read this before believing it.

        Three decompositions of the same number, and every one of them has changed a
        conclusion here at least once:

        * **bye vs matchup.** Covering the week your incumbent is idle is worth a full
          starter and needs no model at all. On the user's real leagues that is 7.1 of
          the 40.9-point D/ST gain -- and 8.4 of the 11.0-point *kicker* gain, which is
          to say kicker streaming is a bye cover with a rounding error attached.
        * **priced vs unpriced.** A week with no posted line falls back to the
          projection-only fit, which was estimated on ESPN's in-season projections and is
          being applied to a preseason one. The further out the week, the more that
          coefficient is being asked to do, and the corpus cannot measure how much it
          decays. Gain sitting in unpriced weeks is the speculative part.
        * **cost.** Acquisitions and extra roster slots, subtracted.
        """
        baseline = baseline or hold_plan(grid)
        delta = self.week_value(grid) - baseline.week_value(grid)
        bye = np.array([i < 0 for i in baseline.start])
        priced = np.array(grid.priced)
        return ValueSplit(
            total=float(delta.sum()) - self.cost,
            bye_cover=float(delta[bye].sum()),
            matchup=float(delta[~bye].sum()),
            priced=float(delta[priced].sum()),
            unpriced=float(delta[~priced].sum()),
            cost=self.cost,
        )


@dataclass(frozen=True, slots=True)
class ValueSplit:
    """Model points a plan gains over holding, decomposed. All rest-of-season totals."""

    total: float
    bye_cover: float
    matchup: float
    priced: float
    unpriced: float
    cost: float


def _plan_from_choices(
    grid: StreamGrid,
    rosters: Sequence[frozenset[int]],
    starts: Sequence[int],
    *,
    method: str,
    optimal: bool,
    acq: np.ndarray,
    extra_slot_cost: float,
    bound: float,
) -> StreamPlan:
    prev = frozenset(grid.held_index)
    acquired: list[tuple[int, ...]] = []
    dropped: list[tuple[int, ...]] = []
    cost = 0.0
    points = 0.0
    for j, held in enumerate(rosters):
        adds = tuple(sorted(held - prev))
        drops = tuple(sorted(prev - held))
        acquired.append(adds)
        dropped.append(drops)
        cost += float(sum(acq[i, j] for i in adds))
        cost += extra_slot_cost * max(len(held) - 1, 0)
        i = starts[j]
        points += float(grid.value[i, j]) if i >= 0 else float(grid.floor[j])
        prev = held
    return StreamPlan(
        weeks=grid.weeks,
        start=tuple(int(i) for i in starts),
        roster=tuple(tuple(sorted(r)) for r in rosters),
        acquired=tuple(acquired),
        dropped=tuple(dropped),
        points=points,
        cost=cost,
        method=method,
        optimal=optimal,
        bound=bound,
    )


def _acquisition_matrix(
    grid: StreamGrid, acquisition_cost: float | Mapping[int, float] | np.ndarray
) -> np.ndarray:
    """(n, W) cost in points of acquiring each candidate in each week."""
    if isinstance(acquisition_cost, Mapping):
        out = np.zeros((grid.n, grid.n_weeks))
        for i, s in enumerate(grid.streamers):
            out[i, :] = float(acquisition_cost.get(s.player_id, 0.0))
        return out
    arr = np.asarray(acquisition_cost, dtype=float)
    if arr.ndim == 0:
        return np.full((grid.n, grid.n_weeks), float(arr))
    if arr.shape == (grid.n,):
        return np.repeat(arr[:, None], grid.n_weeks, axis=1)
    if arr.shape == (grid.n, grid.n_weeks):
        return arr.astype(float)
    raise StreamingError(f"acquisition_cost has shape {arr.shape}, expected (n,) or (n, W)")


# --------------------------------------------------------------------------------------
# Bounds: the LAP relaxation and the per-week argmax
# --------------------------------------------------------------------------------------


def upper_bound(grid: StreamGrid) -> float:
    """Per-week argmax over everything startable. No plan can beat it.

    Zero acquisition cost, unlimited reuse, one roster slot: any feasible plan starts
    somebody startable each week and pays a non-negative cost to do it, so this
    dominates every plan the IP can produce. It is also exactly the optimum when
    acquisition is free and re-acquisition is allowed, which is why `solve` at zero cost
    closes the gap to 0.0 on a `readd_dropped=True` grid; on the conservative default it
    stays a strict bound, and `StreamPlan.gap` is then the option value of the wire.

    The per-week max is taken *against the floor*, not merely as a fallback for a week
    nobody can cover. Leaving the slot empty is a legal move and worth `floor`, so a
    bound that reads the floor only when every candidate is on a bye is not a bound at
    all: with `floor = 5` over candidates worth 1 it returned 3 against a true optimum
    of 15, and `gap`'s clamp at zero then hid the violation instead of raising it.

    Eligibility is "acquirable in this week *or any earlier one*", not `acquirable()`.
    A streamer signed in week 2 is yours in week 5 whether or not he was still on the
    wire then, so reading the week's own availability mask understates the bound: on a
    grid whose availability moves week to week the DP legitimately scored 24.39 against
    a "bound" of 24.15, and again the clamp in `gap` swallowed it. `build_grid` writes a
    constant mask, so the two agree in production -- which is exactly why this had to be
    found by construction rather than by running it.
    """
    ever = np.logical_or.accumulate(grid.available, axis=1) | grid.held[:, None]
    scored = np.where(grid.playing & ever, grid.value, -np.inf)
    best = np.where(np.isfinite(scored).any(axis=0), scored.max(axis=0, initial=-np.inf), -np.inf)
    floor = np.asarray(grid.floor, dtype=float)
    return float(np.maximum(np.where(np.isfinite(best), best, floor), floor).sum())


def relax_assignment(grid: StreamGrid) -> StreamPlan:
    """The rectangular LAP: one streamed slot, each candidate used at most once, free.

    Exactly the brief's relaxation, solved by `scipy.optimize.linear_sum_assignment` in
    O(n^3). The constraint matrix is an interval bipartite incidence matrix and so
    totally unimodular, which is why the LP optimum is integral and this is exact.

    It is a **lower** bound on the *reuse-permitting* streaming IP -- the one you get
    from `build_grid(readd_dropped=True)` -- and not an upper one: "use each streamer at
    most once" is a restriction that IP does not carry, since ESPN lets you re-add a
    defense you dropped two weeks ago.

    Under the conservative default it is not a bound at all, and saying so is the point
    of this paragraph. An assignment that starts your incumbent in week five while
    somebody else covers weeks one to four needs him back after a drop, which the
    default forbids, so the LAP optimum can sit *above* the achievable one: on 400
    random no-re-add instances it did so 67 times, by up to 7.1 points. Read it there as
    a diagnostic -- how much of the schedule a no-reuse planner could have harvested --
    and read `upper_bound` for the certified ceiling. On the real grids the ordering
    holds comfortably (116.3 against an exact 124.9), which is a measurement and not a
    guarantee.

    A week no candidate can cover is left empty and scores `grid.floor`, which is what
    makes this well defined on a real grid: with fewer candidates than weeks, or a week
    everyone has a bye in, a strict one-per-week assignment is simply infeasible. The
    matrix is therefore augmented with one "leave it empty" row per week, so a perfect
    assignment always exists and no forbidden cell is ever forced into the optimum.
    """
    n, n_w = grid.n, grid.n_weeks
    scored = np.where(grid.acquirable(), grid.value, -np.inf)
    forbidden = -1e9
    cost = np.full((n + n_w, n_w), forbidden)
    cost[:n, :] = np.where(np.isfinite(scored), scored, forbidden)
    for j in range(n_w):
        cost[n + j, j] = float(grid.floor[j])
    rows, chosen = linear_sum_assignment(cost, maximize=True)
    starts = [-1] * n_w
    for r, c in zip(rows, chosen, strict=True):
        if r < n and np.isfinite(scored[r, c]):
            starts[c] = int(r)

    rosters = [frozenset() if i < 0 else frozenset({i}) for i in starts]
    zero = np.zeros((grid.n, grid.n_weeks))
    return _plan_from_choices(
        grid,
        rosters,
        starts,
        method="lap",
        optimal=True,
        acq=zero,
        extra_slot_cost=0.0,
        bound=upper_bound(grid),
    )


def hold_plan(grid: StreamGrid) -> StreamPlan:
    """The null: keep exactly what you have and start the best of it every week.

    This is the baseline every recommendation is a delta against. A week where every
    held candidate is on a bye scores `grid.floor`, which is what an unattended empty
    D/ST slot really scores -- and covering those byes is usually most of what streaming
    buys, so it must not be quietly papered over. It goes through `_reward` so that a
    non-zero floor benches a held candidate worth less than it, exactly as the plan
    would: a baseline that starts a two-point defense against a five-point replacement
    level would credit streaming with three points a week it never earned.
    """
    held = tuple(grid.held_index)
    scored = grid.scored()
    starts = [_reward(grid, scored, held, j)[1] for j in range(grid.n_weeks)]
    zero = np.zeros((grid.n, grid.n_weeks))
    return _plan_from_choices(
        grid,
        [frozenset(held)] * grid.n_weeks,
        starts,
        method="hold",
        optimal=True,
        acq=zero,
        extra_slot_cost=0.0,
        bound=upper_bound(grid),
    )


# --------------------------------------------------------------------------------------
# The exact integer program, as a dynamic program over held sets
# --------------------------------------------------------------------------------------


def _reward(
    grid: StreamGrid, scored: np.ndarray, held: tuple[int, ...], j: int
) -> tuple[float, int]:
    """Best startable value in week `j` from this holding, and who provides it.

    The floor is a *choice*, not only a fallback. With a non-zero replacement level an
    unfilled slot still scores something -- whatever the wire would have streamed into
    it -- so a held candidate worth less than the floor belongs on the bench, and
    reading the floor only when nothing at all is startable silently starts him anyway.
    That is not hypothetical: with `floor > 0` and a claim priced above zero it made the
    DP and `brute_force` come back up to three points short of HiGHS on 33 of 300 random
    instances while both still reported `optimal=True`, and it was invisible to the
    tests precisely because the DP and the brute force shared this function -- the
    exhaustion that was supposed to check the DP was enumerating the same mistake.
    """
    best, who = -np.inf, -1
    for i in held:
        if scored[i, j] > best:
            best, who = scored[i, j], i
    floor = float(grid.floor[j])
    if not np.isfinite(best) or best < floor:
        return floor, -1
    return float(best), who


def solve(
    grid: StreamGrid,
    *,
    kappa: int = 1,
    acquisition_cost: float | Mapping[int, float] | np.ndarray = 0.0,
    extra_slot_cost: float = 0.0,
    max_acquisitions: int | None = None,
    method: str = "auto",
    max_candidates: int = 14,
    max_states: int = 200_000,
) -> StreamPlan:
    """Solve the streaming IP exactly, by dynamic programming over held sets.

    The brief's formulation -- roster flow `h = h_prev + a - d`, start implies hold, one
    started streamer per week, `kappa` streamer roster slots, availability and byes -- is
    a shortest path on the graph whose nodes are `(week, set of streamers held, claims
    used)`. So it needs no branch-and-bound and no LP solver: the DP *is* the exact
    optimum over that graph, and `optimal=True` means the whole graph was enumerated.

    `kappa=1` is the case that occurs, and it collapses to O(n) per week: from holding
    `i` you either keep him or pay for the best replacement, and the "best replacement"
    term does not depend on `i`. A 32-candidate 14-week instance solves in well under a
    millisecond, so there is no reason to approximate it.

    `acquisition_cost` is in POINTS, and is the seam for a priority league: pass
    `decide/waivers.py`'s continuation value for burning waiver priority. In a FAAB
    league pass `lambda * price`. Both of the user's regimes are covered by the default
    of 0.0 for D/ST and K, where the streamed asset is a plain free agent.

    `extra_slot_cost` prices the *second* streamer roster slot in points per week: with
    `kappa=2` you are carrying a defense in place of a bench player, and the honest cost
    of that is what the bench player would have contributed.

    `method` picks the machinery, and "auto" is right unless you are cross-checking:

    * ``"dp"`` -- the held-set recursion. Exact and about 3.6 ms on a live 20-candidate
      17-week grid at `kappa = 1`. At `kappa > 1` the state space is combinatorial, so
      the field is pruned to `max_candidates` and `optimal` comes back False, because an
      optimum over a pruned field is not an optimum.
    * ``"milp"`` -- the brief's integer program handed to `scipy.optimize.milp`, which
      is HiGHS. Exact at any `kappa`, no pruning, roughly 10 ms.
    * ``"auto"`` -- the DP at `kappa = 1`, the MILP above it.

    That split is measured rather than assumed. On six random 20x17 instances the DP and
    HiGHS agree to 1e-6 at `kappa = 1` and the DP is about three times faster; at
    `kappa = 2` the *pruned* DP took 4.7 seconds and came back 0.01 to 5.4 points short
    of HiGHS on three of six, which is exactly the failure `optimal=False` exists to
    announce and exactly the reason the default no longer runs it.

    A note on the brief, which says no solver dependency is available: scipy ships
    HiGHS as `scipy.optimize.milp`, so one is, and this uses it.
    """
    if kappa < 1:
        raise StreamingError("kappa must be at least 1")
    if method not in ("auto", "dp", "milp"):
        raise StreamingError(f"unknown method {method!r}; use 'auto', 'dp' or 'milp'")
    held0 = tuple(grid.held_index)
    if len(held0) > kappa:
        raise StreamingError(
            f"you already hold {len(held0)} candidates at this position but kappa is {kappa}; "
            "raise kappa or drop one before planning"
        )
    claims = grid.n_weeks * kappa if max_acquisitions is None else max(int(max_acquisitions), 0)
    metered = max_acquisitions is not None
    if method == "auto":
        method = "dp" if kappa == 1 else "milp"

    if method == "milp":
        return _solve_milp(
            grid,
            kappa=kappa,
            acq=_acquisition_matrix(grid, acquisition_cost),
            extra_slot_cost=extra_slot_cost,
            max_acquisitions=max_acquisitions,
        )
    if kappa == 1:
        acq = _acquisition_matrix(grid, acquisition_cost)
        return _solve_kappa1(grid, grid.scored(), acq, held0, claims, metered)

    pruned, keep = _prune(grid, max_candidates)
    acq = _acquisition_matrix(grid, acquisition_cost)[keep, :]
    plan = _solve_subsets(
        pruned,
        pruned.scored(),
        acq,
        tuple(pruned.held_index),
        kappa,
        extra_slot_cost,
        claims,
        metered,
        max_states,
    )
    return _reindex(plan, grid, keep, optimal=len(keep) == grid.n)


def _solve_milp(
    grid: StreamGrid,
    *,
    kappa: int,
    acq: np.ndarray,
    extra_slot_cost: float,
    max_acquisitions: int | None,
) -> StreamPlan:
    """The brief's integer program, written out and handed to HiGHS.

    Variables, all indexed `(candidate, week)` and flattened row-major: `h` holds, `y`
    starts, `a` acquires, plus one `s_w` per week counting streamers past the first.
    `a` is left continuous on purpose -- constraint (4) forces `a >= h_w - h_{w-1}` and
    every objective coefficient on it is non-negative, so it settles on the integral
    `max(0, flow)` without paying for another integer variable. `d` (drop) is not a
    variable at all for the same reason: nothing in the objective or the constraints
    reads it, so carrying it would only give the solver a free dimension to wander in.

        (1) y_iw <= h_iw                       start implies hold
        (2) sum_i y_iw <= 1                    one streamed starter, or none
        (3) sum_i h_iw <= kappa                streamer bench cap
        (4) h_iw - h_i,w-1 - a_iw <= 0         roster flow (w=0 reads the current roster)
        (5) sum_i h_iw - s_w <= 1              slots past the first, priced
        (6) sum_iw a_iw <= B                   the acquisition budget, when metered

    Byes and availability are bounds rather than rows: `y` is capped at zero on a bye and
    `a` at zero where the player cannot be signed, and (4) then propagates "never
    acquirable and not currently held" forward into `h = 0` for the whole season.
    """
    n, n_w = grid.n, grid.n_weeks
    size = n * n_w
    n_vars = 3 * size + n_w

    def hv(i: int, w: int) -> int:
        return i * n_w + w

    def av(i: int, w: int) -> int:
        return size + i * n_w + w

    def yv(i: int, w: int) -> int:
        return 2 * size + i * n_w + w

    def sv(w: int) -> int:
        return 3 * size + w

    # Objective, minimised. The floor is a constant per week plus the *excess* a
    # starter earns over it, so a week nobody can beat the floor in simply goes unstarted.
    cost = np.zeros(n_vars)
    for i in range(n):
        for w in range(n_w):
            cost[yv(i, w)] = -(grid.value[i, w] - grid.floor[w])
            cost[av(i, w)] = acq[i, w]
    for w in range(n_w):
        cost[sv(w)] = extra_slot_cost

    start_implies_hold = lil_matrix((size, n_vars))
    flow = lil_matrix((size, n_vars))
    flow_ub = np.zeros(size)
    for i in range(n):
        for w in range(n_w):
            r = i * n_w + w
            start_implies_hold[r, yv(i, w)] = 1
            start_implies_hold[r, hv(i, w)] = -1
            flow[r, hv(i, w)] = 1
            flow[r, av(i, w)] = -1
            if w > 0:
                flow[r, hv(i, w - 1)] = -1
            else:
                flow_ub[r] = 1.0 if grid.held[i] else 0.0

    one_starter = lil_matrix((n_w, n_vars))
    bench_cap = lil_matrix((n_w, n_vars))
    slots = lil_matrix((n_w, n_vars))
    for w in range(n_w):
        for i in range(n):
            one_starter[w, yv(i, w)] = 1
            bench_cap[w, hv(i, w)] = 1
            slots[w, hv(i, w)] = 1
        slots[w, sv(w)] = -1

    constraints = [
        LinearConstraint(csr_matrix(start_implies_hold), -np.inf, 0.0),
        LinearConstraint(csr_matrix(flow), -np.inf, flow_ub),
        LinearConstraint(csr_matrix(one_starter), -np.inf, 1.0),
        LinearConstraint(csr_matrix(bench_cap), -np.inf, float(kappa)),
        LinearConstraint(csr_matrix(slots), -np.inf, 1.0),
    ]
    if max_acquisitions is not None:
        budget = lil_matrix((1, n_vars))
        for i in range(n):
            for w in range(n_w):
                budget[0, av(i, w)] = 1
        constraints.append(
            LinearConstraint(csr_matrix(budget), -np.inf, float(max(max_acquisitions, 0)))
        )

    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)
    upper[3 * size :] = float(n)
    integrality = np.zeros(n_vars)
    integrality[:size] = 1
    integrality[2 * size : 3 * size] = 1
    for i in range(n):
        for w in range(n_w):
            if not grid.playing[i, w]:
                upper[yv(i, w)] = 0.0
            if not grid.available[i, w]:
                upper[av(i, w)] = 0.0

    result = milp(
        c=cost,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lower, upper),
    )
    if not result.success or result.x is None:
        raise StreamingError(f"HiGHS did not solve the streaming IP: {result.message}")
    x = np.asarray(result.x)
    rosters = [frozenset(i for i in range(n) if x[hv(i, w)] > 0.5) for w in range(n_w)]
    starts = [next((i for i in range(n) if x[yv(i, w)] > 0.5), -1) for w in range(n_w)]
    return _plan_from_choices(
        grid,
        rosters,
        starts,
        method=f"milp-kappa{kappa}",
        # `status == 0` is HiGHS reporting a proven optimum, not a time-out or a gap.
        optimal=int(result.status) == 0,
        acq=acq,
        extra_slot_cost=extra_slot_cost,
        bound=upper_bound(grid),
    )


def _prune(grid: StreamGrid, max_candidates: int) -> tuple[StreamGrid, list[int]]:
    """The best `max_candidates` by peak weekly value, plus anything already held."""
    if grid.n <= max_candidates:
        return grid, list(range(grid.n))
    scored = grid.scored()
    peak = np.where(np.isfinite(scored), scored, -np.inf).max(axis=1)
    order = sorted(range(grid.n), key=lambda i: -peak[i])
    keep = sorted(set(order[:max_candidates]) | set(grid.held_index))
    return (
        replace(
            grid,
            streamers=tuple(grid.streamers[i] for i in keep),
            value=grid.value[keep, :],
            projection=grid.projection[keep, :],
            playing=grid.playing[keep, :],
            available=grid.available[keep, :],
            held=grid.held[keep],
        ),
        keep,
    )


def _reindex(
    plan: StreamPlan, grid: StreamGrid, keep: Sequence[int], *, optimal: bool
) -> StreamPlan:
    """Map a plan solved over a pruned candidate list back onto the full grid."""
    if list(keep) == list(range(grid.n)):
        return replace(plan, optimal=optimal)
    m = {i: k for i, k in enumerate(keep)}
    return replace(
        plan,
        start=tuple(m[i] if i >= 0 else -1 for i in plan.start),
        roster=tuple(tuple(m[i] for i in r) for r in plan.roster),
        acquired=tuple(tuple(m[i] for i in a) for a in plan.acquired),
        dropped=tuple(tuple(m[i] for i in d) for d in plan.dropped),
        bound=upper_bound(grid),
        optimal=optimal,
    )


def _solve_kappa1(
    grid: StreamGrid,
    scored: np.ndarray,
    acq: np.ndarray,
    held0: tuple[int, ...],
    claims: int,
    metered: bool,
) -> StreamPlan:
    """O(n * W * claims) exact DP for one streamer roster slot.

    State is `(who you hold, claims spent)`, with the index `n` meaning an empty slot.
    Three moves are available each week and only the third couples to the state you came
    from, which is what makes this linear rather than quadratic in the candidate count:

        keep s          free
        drop to empty   free
        switch to t     costs acq[t, w] and one claim, and t is the same argmax whatever
                        s was -- except that t == s is the "keep" move, so the recursion
                        carries the top TWO switch targets and uses the runner-up
                        exactly when the leader is who you already hold.
    """
    n, n_w = grid.n, grid.n_weeks
    n_states = n + 1
    n_claims = claims + 1

    future = np.zeros((n_states, n_claims))
    choice = np.full((n_w, n_states, n_claims), -1, dtype=np.int32)
    for j in range(n_w - 1, -1, -1):
        # Holding `t` through week j is worth the better of starting him and leaving the
        # slot to the floor -- the same rule `_reward` applies, so the DP, the subset DP,
        # `brute_force` and HiGHS all price a benched streamer identically.
        floor_j = float(grid.floor[j])
        reward = np.maximum(np.where(np.isfinite(scored[:, j]), scored[:, j], floor_j), floor_j)
        # f[t, m] = holding t through week j, then the future with m claims spent.
        f = np.vstack(
            [reward[:, None] + future[:n, :], float(grid.floor[j]) + future[n : n + 1, :]]
        )
        current = np.empty((n_states, n_claims))
        for m in range(n_claims):
            spend = m + 1
            if metered and spend > claims:
                order: list[int] = []
                cand = np.full(n, -np.inf)
            else:
                col = min(spend, n_claims - 1)
                cand = np.where(grid.available[:, j], f[:n, col] - acq[:, j], -np.inf)
                order = [int(i) for i in np.argsort(-cand)[:2] if np.isfinite(cand[i])]
            for s in range(n_states):
                best, best_t = f[s, m], s  # keep (s == n is "stay empty")
                if f[n, m] > best:
                    best, best_t = f[n, m], n
                for t in order:
                    if t == s:
                        continue
                    if cand[t] > best:
                        best, best_t = float(cand[t]), t
                    break
                current[s, m] = best
                choice[j, s, m] = best_t
        future = current

    state = held0[0] if held0 else n
    spent = 0
    rosters: list[frozenset[int]] = []
    starts: list[int] = []
    for j in range(n_w):
        nxt = int(choice[j, state, spent])
        if nxt < n and nxt != state:
            spent = min(spent + 1, claims)
        held = frozenset() if nxt >= n else frozenset({nxt})
        rosters.append(held)
        starts.append(_reward(grid, scored, tuple(held), j)[1])
        state = nxt
    return _plan_from_choices(
        grid,
        rosters,
        starts,
        method="dp-kappa1",
        optimal=True,
        acq=acq,
        extra_slot_cost=0.0,
        bound=upper_bound(grid),
    )


def _solve_subsets(
    grid: StreamGrid,
    scored: np.ndarray,
    acq: np.ndarray,
    held0: tuple[int, ...],
    kappa: int,
    extra_slot_cost: float,
    claims: int,
    metered: bool,
    max_states: int,
) -> StreamPlan:
    """Exact DP over all held sets of size <= kappa. Exponential in kappa, not in n."""
    n, n_w = grid.n, grid.n_weeks
    states: list[tuple[int, ...]] = [
        c for size in range(kappa + 1) for c in itertools.combinations(range(n), size)
    ]
    if len(states) * (claims + 1) > max_states:
        raise StreamingError(
            f"kappa={kappa} over {n} candidates needs {len(states) * (claims + 1)} DP states, "
            f"over the {max_states} cap. Lower max_candidates or use rolling_plan()."
        )
    index = {s: i for i, s in enumerate(states)}
    n_claims = claims + 1
    future = np.zeros((len(states), n_claims))
    choice = np.full((n_w, len(states), n_claims), -1, dtype=np.int32)
    for j in range(n_w - 1, -1, -1):
        reward = np.array([_reward(grid, scored, s, j)[0] for s in states])
        hold_cost = np.array([extra_slot_cost * max(len(s) - 1, 0) for s in states])
        current = np.full((len(states), n_claims), -np.inf)
        for si, s in enumerate(states):
            prev = set(s)
            for m in range(n_claims):
                best, best_t = -np.inf, -1
                for ti, t in enumerate(states):
                    adds = [k for k in t if k not in prev]
                    if any(not grid.available[k, j] for k in adds):
                        continue
                    spend = m + len(adds)
                    if metered and spend > claims:
                        continue
                    total = (
                        reward[ti]
                        - hold_cost[ti]
                        - float(sum(acq[k, j] for k in adds))
                        + future[ti, min(spend, n_claims - 1)]
                    )
                    if total > best:
                        best, best_t = total, ti
                current[si, m] = best
                choice[j, si, m] = best_t
        future = current

    state = index[tuple(sorted(held0))]
    spent = 0
    rosters: list[frozenset[int]] = []
    starts: list[int] = []
    for j in range(n_w):
        nxt = int(choice[j, state, spent])
        if nxt < 0:
            raise StreamingError(f"no feasible holding in week {grid.weeks[j]}")
        held = states[nxt]
        spent = min(spent + len(set(held) - set(states[state])), claims)
        rosters.append(frozenset(held))
        starts.append(_reward(grid, scored, held, j)[1])
        state = nxt
    return _plan_from_choices(
        grid,
        rosters,
        starts,
        method=f"dp-kappa{kappa}",
        optimal=True,
        acq=acq,
        extra_slot_cost=extra_slot_cost,
        bound=upper_bound(grid),
    )


# --------------------------------------------------------------------------------------
# Rolling horizon, brute force, and the measured gap
# --------------------------------------------------------------------------------------


def rolling_plan(
    grid: StreamGrid,
    *,
    horizon: int = 4,
    kappa: int = 1,
    acquisition_cost: float | Mapping[int, float] | np.ndarray = 0.0,
    extra_slot_cost: float = 0.0,
    revalue: Callable[[StreamGrid, int], StreamGrid] | None = None,
) -> StreamPlan:
    """Optimise the next `horizon` weeks, commit week t only, observe, re-solve.

    This is how the plan is actually executed, and it is a heuristic against the
    full-horizon solve for one reason only: a finite horizon cannot see a bye, a cost
    worth saving for, or a playoff-week matchup that sits beyond it. With the grid
    frozen, `horizon >= n_weeks` reproduces `solve` exactly (there is a test).

    Measured on the live 2026 D/ST grids by `optimality_gap`: with free acquisition the
    gap is exactly zero at every horizon, because nothing couples the weeks and the
    myopic plan is the optimal one. At 1, 2 and 3 points per claim a one-week horizon
    gives back 0.6, 4.4 and 4.1 points of a ~100-point objective, and four weeks of
    lookahead was exactly optimal in every case tested. Default accordingly.

    `revalue` is the hook that makes re-solving worth anything: called as
    `revalue(grid, week_index)` before each commitment, it is where a caller folds in
    the lines that have posted since. Without it, re-solving is arithmetic.
    """
    if horizon < 1:
        raise StreamingError("horizon must be at least one week")
    working = grid
    held = frozenset(grid.held_index)
    rosters: list[frozenset[int]] = []
    starts: list[int] = []
    for j in range(grid.n_weeks):
        if revalue is not None:
            working = revalue(working, j)
        window = _slice_weeks(working, j, min(j + horizon, working.n_weeks), held)
        step = solve(
            window,
            kappa=kappa,
            acquisition_cost=_slice_costs(working, acquisition_cost, j, window.n_weeks),
            extra_slot_cost=extra_slot_cost,
        )
        held = frozenset(step.roster[0])
        rosters.append(held)
        starts.append(step.start[0])
    acq = _acquisition_matrix(grid, acquisition_cost)
    return _plan_from_choices(
        grid,
        rosters,
        starts,
        method=f"rolling-{horizon}",
        optimal=False,
        acq=acq,
        extra_slot_cost=extra_slot_cost,
        bound=upper_bound(grid),
    )


def _slice_weeks(grid: StreamGrid, start: int, stop: int, held: frozenset[int]) -> StreamGrid:
    """The grid restricted to `weeks[start:stop]`, with `held` as the new initial state."""
    mask = np.zeros(grid.n, dtype=bool)
    for i in held:
        mask[i] = True
    return replace(
        grid,
        weeks=grid.weeks[start:stop],
        value=grid.value[:, start:stop],
        projection=grid.projection[:, start:stop],
        playing=grid.playing[:, start:stop],
        available=grid.available[:, start:stop],
        held=mask,
        priced=grid.priced[start:stop],
        floor=grid.floor[start:stop],
    )


def _slice_costs(
    grid: StreamGrid,
    acquisition_cost: float | Mapping[int, float] | np.ndarray,
    start: int,
    width: int,
) -> np.ndarray:
    return _acquisition_matrix(grid, acquisition_cost)[:, start : start + width]


def brute_force(
    grid: StreamGrid,
    *,
    kappa: int = 1,
    acquisition_cost: float | Mapping[int, float] | np.ndarray = 0.0,
    extra_slot_cost: float = 0.0,
    max_paths: int = 5_000_000,
) -> StreamPlan:
    """Enumerate every feasible roster path. Only for small instances and for tests.

    The point of this existing is that `solve` claims optimality, and a claim of
    optimality that has never been checked against exhaustion is a comment, not a fact.
    """
    n, n_w = grid.n, grid.n_weeks
    states = [tuple(c) for size in range(kappa + 1) for c in itertools.combinations(range(n), size)]
    if len(states) ** n_w > max_paths:
        raise StreamingError(
            f"{len(states)}^{n_w} paths is past the {max_paths} cap; brute force is for tests"
        )
    acq = _acquisition_matrix(grid, acquisition_cost)
    scored = grid.scored()
    best_value, best_path = -np.inf, None
    for path in itertools.product(states, repeat=n_w):
        prev = set(grid.held_index)
        total = 0.0
        ok = True
        for j, held in enumerate(path):
            adds = [k for k in held if k not in prev]
            if any(not grid.available[k, j] for k in adds):
                ok = False
                break
            total -= float(sum(acq[k, j] for k in adds))
            total -= extra_slot_cost * max(len(held) - 1, 0)
            total += _reward(grid, scored, held, j)[0]
            prev = set(held)
        if ok and total > best_value:
            best_value, best_path = total, path
    if best_path is None:
        raise StreamingError("no feasible roster path exists")
    rosters = [frozenset(h) for h in best_path]
    starts = [_reward(grid, scored, h, j)[1] for j, h in enumerate(best_path)]
    return _plan_from_choices(
        grid,
        rosters,
        starts,
        method="brute-force",
        optimal=True,
        acq=acq,
        extra_slot_cost=extra_slot_cost,
        bound=upper_bound(grid),
    )


@dataclass(frozen=True, slots=True)
class GapReport:
    """What each solver actually scored on one instance, in points per season."""

    exact: float
    lap: float
    hold: float
    bound: float
    rolling: Mapping[int, float]
    brute: float | None = None

    @property
    def rolling_gap(self) -> dict[int, float]:
        return {h: self.exact - v for h, v in self.rolling.items()}

    @property
    def verified(self) -> bool:
        """True when brute force agreed with the DP to within floating point."""
        return self.brute is not None and abs(self.brute - self.exact) < 1e-9


def optimality_gap(
    grid: StreamGrid,
    *,
    kappa: int = 1,
    acquisition_cost: float | Mapping[int, float] | np.ndarray = 0.0,
    horizons: Sequence[int] = (1, 2, 3, 4, 6),
    brute: bool = False,
) -> GapReport:
    """Measure, do not assert. Runs every solver on the same instance and reports."""
    exact = solve(grid, kappa=kappa, acquisition_cost=acquisition_cost)
    rolling = {
        h: rolling_plan(grid, horizon=h, kappa=kappa, acquisition_cost=acquisition_cost).value
        for h in horizons
    }
    return GapReport(
        exact=exact.value,
        lap=relax_assignment(grid).value,
        hold=hold_plan(grid).value,
        bound=upper_bound(grid),
        rolling=rolling,
        brute=brute_force(grid, kappa=kappa, acquisition_cost=acquisition_cost).value
        if brute
        else None,
    )


# --------------------------------------------------------------------------------------
# Turning a plan into a Recommendation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamEvaluation:
    """The paired-CRN simulation of a plan against holding."""

    baseline_title: float
    plan_title: float
    delta_title: float
    stderr: float
    delta_points: float
    #: Model-space points gain, from the grid rather than the simulator. The two should
    #: agree to within the lineup solver's own effects; when they do not, the grid is
    #: promising points the roster cannot actually start.
    model_points: float
    #: Title odds and delta from making week one's move and then holding forever. The
    #: honest floor under `delta_title`: it needs no faith in the unpriced weeks.
    commit_title: float
    commit_delta: float
    commit_stderr: float
    leverage: float
    n_sims: int


def _extended_state(
    state: S.LeagueState, grid: StreamGrid, *, anchor: int | None = None
) -> tuple[S.LeagueState, int | None]:
    """`state` with every streaming candidate in the pool, and an anchor on your roster.

    A free agent has no column in the league's pool, so there is nowhere to put his
    simulated points. Widening the pool -- and only the pool, never a roster -- gives
    every candidate one without giving him to anybody.

    `anchor` is the exception, and it exists for the case where you roster nobody at the
    streamed position at all: there is then no column standing in for the slot, so one
    candidate is signed to your franchise to be that column. Both arms use the same
    anchor, so it cancels; the baseline simply drives it to zero every week.
    """
    have = set(state.pool.player_ids)
    extra = [s for s in grid.streamers if s.player_id not in have]
    out = state
    if extra:
        rows = list(
            zip(
                state.pool.player_ids,
                state.pool.position_ids,
                state.pool.pro_team_ids,
                state.pool.names or tuple("" for _ in state.pool.player_ids),
                strict=True,
            )
        )
        rows.extend((s.player_id, s.position_id, s.pro_team_id, s.name) for s in extra)
        out = replace(out, pool=S.PlayerPool.of(rows))
    if anchor is None:
        return out, None
    me = out.franchise(grid.my_team_id)
    if anchor not in me.player_ids:
        out = out.with_franchise(me.with_players((*me.player_ids, anchor)))
    return out, anchor


def _grid_outlooks(
    grid: StreamGrid,
    base: Sequence[PlayerOutlook],
    *,
    season: int,
    weeks: Sequence[int],
    calibration: CalibrationSet,
    apply_model: bool = True,
) -> list[PlayerOutlook]:
    """`base`, with every candidate's weeks rebuilt from the matchup model's mean.

    Without this the optimiser and the simulator disagree: the plan is chosen on a
    forecast that says Denver is worth 8.2 this week and then scored on ESPN's 6.1. The
    model's output is already a calibrated conditional mean of *actual* points, so it
    goes in through `outlook_from_mean` -- putting it through `outlook` would apply
    ESPN's level correction a second time.
    """
    by_id = {s.player_id: i for i, s in enumerate(grid.streamers)}
    # The same position id the pipeline compiled with. K and D/ST have no fitted curve
    # of their own and fall through to the pooled one inside the calibration set, so
    # this is a label rather than a switch -- but a mismatched label would change the
    # correlation block the sampler puts the player in.
    pos = grid.position_id
    out: list[PlayerOutlook] = []
    for o in base:
        i = by_id.get(o.player_id)
        if i is None:
            out.append(o)
            continue
        rebuilt = dict(o.weeks)
        for j, w in enumerate(grid.weeks if apply_model else ()):
            rebuilt[w] = calibration.outlook_from_mean(
                player_id=o.player_id,
                season=season,
                week=w,
                position_id=pos,
                mean=float(grid.value[i, j]),
                pro_team_id=o.pro_team_id,
                playing=bool(grid.playing[i, j]),
            )
        # The corpus is not guaranteed complete -- the 2026 capture is missing the Saints
        # D/ST in week 8 -- and a candidate the pool now carries must cover every
        # simulated week or `panel_for` refuses the whole tensor.
        for w in weeks:
            if w not in rebuilt:
                rebuilt[w] = calibration.outlook_from_mean(
                    player_id=o.player_id,
                    season=season,
                    week=w,
                    position_id=pos,
                    mean=0.0,
                    pro_team_id=o.pro_team_id,
                    playing=False,
                )
        out.append(replace(o, weeks=rebuilt))
    return out


def evaluate(
    state: S.LeagueState,
    outlooks: Sequence[PlayerOutlook],
    grid: StreamGrid,
    plan: StreamPlan,
    *,
    baseline: StreamPlan | None = None,
    my_team_id: int,
    n_sims: int = 4000,
    seed: int = 11,
    market: MarketSchedule | None = None,
    calibration: CalibrationSet | None = None,
    apply_matchup_model: bool = True,
    efficiency: S.LineupEfficiency | None = None,
) -> StreamEvaluation:
    """Simulate holding against streaming under common random numbers.

    Both arms are drawn from ONE universe and differ only in which column feeds the
    streamed slot each week, so the paired difference isolates the plan instead of
    re-rolling the season. The substitution is exact rather than additive: the
    candidate's realised points are written into the incumbent's column *and* into the
    ex-ante ranking key, so the lineup solver sees the streamed player, honours the bye,
    and the efficiency haircut applies to the same team total it would have applied to.

    `stderr` is the SD of the per-simulation paired difference, not the two arms'
    independent binomial errors. That distinction is the whole point of CRN: on a real
    league the paired error is several times smaller, and using the unpaired one would
    declare every true effect insignificant.

    `apply_matchup_model=True` rebuilds the candidates' *simulated* distributions from
    the same forecast the plan was chosen on, so the optimiser and the simulator agree
    about what a defense is worth. Turning it off scores the plan on ESPN's own
    calibrated means instead, which is the sensitivity worth running before believing
    any of this: on Wine Wednesday it takes the D/ST answer from +2.80pp to +2.05pp, so
    most of the effect does not depend on the matchup model being right about magnitude.
    """
    calibration = calibration or load_calibration("ppr")
    baseline = baseline or hold_plan(grid)
    held = grid.held_index
    anchor_id = None
    if not held:
        first = next((i for i in plan.start if i >= 0), 0)
        anchor_id = grid.streamers[first].player_id
        log.info(
            "no %s on team %d; anchoring the streamed slot on %s",
            grid.model.label,
            grid.my_team_id,
            grid.streamers[first].name,
        )
    ext_state, anchor = _extended_state(state, grid, anchor=anchor_id)
    rows = _grid_outlooks(
        grid,
        outlooks,
        season=state.season,
        weeks=state.weeks,
        calibration=calibration,
        apply_model=apply_matchup_model,
    )
    byes = market.bye_by_pro_team_id() if market is not None else None
    panel = S.panel_for(ext_state, rows, byes=byes)
    draw = WeeklySampler(panel, seed=seed).draw(n_sims)

    points = np.asarray(draw.points)
    rank = S.ex_ante_rank(draw)
    cols = {s.player_id: int(panel.index_of([s.player_id])[0]) for s in grid.streamers}
    week_index = {w: i for i, w in enumerate(ext_state.weeks)}
    slot_id = grid.streamers[held[0]].player_id if held else anchor
    slot_col = cols[slot_id] if slot_id is not None else None

    t = ext_state.team_index[my_team_id]

    def arm(starts: Sequence[int]) -> tuple[S.SeasonResult, float, np.ndarray]:
        pts, rk = points, rank
        if slot_col is not None:
            pts, rk = points.copy(), rank.copy()
            for j, w in enumerate(grid.weeks):
                i = starts[j]
                wi = week_index[w]
                src = cols[grid.streamers[i].player_id] if i >= 0 else None
                if src is None:
                    pts[:, wi, slot_col] = 0.0
                    rk[:, wi, slot_col] = -np.inf
                else:
                    pts[:, wi, slot_col] = points[:, wi, src]
                    rk[:, wi, slot_col] = rank[:, wi, src]
        scores = S.team_week_scores(ext_state, pts, rank=rk, efficiency=efficiency)
        # `SeasonResult.points_for` accumulates only over scheduled head-to-head games,
        # so it silently drops the three playoff weeks -- a quarter of the horizon, and
        # the quarter that decides titles. The starting-lineup total is the honest one.
        started = float(scores[:, :, t].sum(axis=1).mean())
        return S.simulate_from_scores(ext_state, scores, all_play=False), started, scores

    # A third arm that is not decoration: make week one's move and then never touch the
    # slot again. It separates "this swap is right" from "this whole plan is right",
    # which matters because the plan's later weeks lean on projections no market has
    # priced yet, and the user only executes the first week today anyway.
    commit_starts = [
        plan.start[0] if plan.start[0] >= 0 and grid.playing[plan.start[0], j] else -1
        for j in range(grid.n_weeks)
    ]
    base_result, base_points, base_scores = arm(baseline.start)
    plan_result, plan_points, _ = arm(plan.start)
    commit_result, _, _ = arm(commit_starts)

    base_champ = base_result.champions[:, t].astype(np.float64)
    plan_champ = plan_result.champions[:, t].astype(np.float64)
    commit_champ = commit_result.champions[:, t].astype(np.float64)
    diff = plan_champ - base_champ
    stderr = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0

    delta_points = plan_points - base_points
    lev = _first_week_leverage(ext_state, base_scores, my_team_id, grid.weeks[0])
    return StreamEvaluation(
        baseline_title=float(base_champ.mean()),
        plan_title=float(plan_champ.mean()),
        delta_title=float(diff.mean()),
        stderr=stderr,
        delta_points=delta_points,
        model_points=float(plan.points - baseline.points),
        commit_title=float(commit_champ.mean()),
        commit_delta=float((commit_champ - base_champ).mean()),
        commit_stderr=float((commit_champ - base_champ).std(ddof=1) / math.sqrt(n_sims))
        if n_sims > 1
        else 0.0,
        leverage=lev,
        n_sims=n_sims,
    )


def _first_week_leverage(
    state: S.LeagueState, scores: np.ndarray, my_team_id: int, week: int
) -> float:
    """How much a point in this week's matchup is worth, relative to a coin flip.

    Taken off the simulated `(sims, weeks, teams)` scores for the actual matchup rather
    than off a season average, because the point of the number is *this* week: a
    twenty-point favourite gets almost nothing from a better defense however wide the
    league-wide spread is. `sd_diff` comes from the same simulation, so it reflects this
    pair of rosters rather than the corpus-wide 34.4 -- the two should be close, and a
    large disagreement is worth knowing about.
    """
    game = next(
        (
            g
            for g in state.remaining_games
            if week in g.weeks and my_team_id in (g.home_team_id, g.away_team_id)
        ),
        None,
    )
    if game is None:
        return 0.0
    opp = game.away_team_id if game.home_team_id == my_team_id else game.home_team_id
    tindex, windex = state.team_index, state.week_index
    rows = [windex[w] for w in game.weeks if w in windex]
    if not rows:
        return 0.0
    diff = scores[:, rows, tindex[my_team_id]].sum(axis=1) - scores[:, rows, tindex[opp]].sum(
        axis=1
    )
    sd = float(diff.std())
    return float(_leverage(float(diff.mean()), sd if sd > 0 else 34.4))


def _rationale(
    grid: StreamGrid,
    plan: StreamPlan,
    base: StreamPlan,
    ev: StreamEvaluation,
    *,
    suppressed: bool = False,
) -> str:
    """A human-actionable summary: what to do this week, then the rest of the plan."""
    names = [s.name for s in grid.streamers]
    lines: list[str] = []
    incumbent = names[base.start[0]] if base.start[0] >= 0 else "nobody"
    first = names[plan.start[0]] if plan.start[0] >= 0 else "nobody"
    if suppressed:
        lines.append(
            f"Week {grid.weeks[0]}: hold {incumbent}. The plan's week-one swap to {first} is "
            f"worth {grid.value[plan.start[0], 0] - grid.value[base.start[0], 0]:+.2f} points "
            f"against a fitted within-week spread of {grid.model.fitted_sd:.2f} for "
            f"{grid.model.label}, which the model cannot resolve, so it is not recommended."
        )
    elif plan.start[0] != base.start[0]:
        lines.append(f"Week {grid.weeks[0]}: start {first} over {incumbent}.")
    else:
        lines.append(
            f"Week {grid.weeks[0]}: hold {incumbent} -- the move is in the weeks after it, "
            "so there is nothing to execute today."
        )
    schedule = []
    for j, w in enumerate(grid.weeks):
        i = plan.start[j]
        tag = "" if grid.priced[j] else "*"
        who = f"{names[i]}{tag}" if i >= 0 else "(empty)"
        schedule.append(f"w{w} {who}")
    lines.append(" -> ".join(schedule))
    if not grid.model.uses_market:
        lines.append(
            f"* the fitted Vegas terms for {grid.model.label} were indistinguishable from "
            "zero, so every week here is the projection-only model; no line is used."
        )
    elif not all(grid.priced):
        unpriced = [w for w, p in zip(grid.weeks, grid.priced, strict=True) if not p]
        lines.append(
            f"* no closing line posted yet for weeks {unpriced[0]}-{unpriced[-1]}; "
            "those weeks use the projection-only fit and will move as the market opens."
        )
    if plan.start == base.start and plan.n_acquisitions == 0:
        lines.append(
            f"This plan is identical to holding in every week: no free agent at "
            f"{grid.model.label} beats what you already roster in any week of the season, so "
            "the honest answer here is that the position does not need attention. The "
            "delta below is exactly zero by construction, not a measured null."
        )
    split = plan.value_split(grid, base)
    lines.append(
        f"Adopting the WHOLE plan -- {plan.n_acquisitions} acquisitions over "
        f"{grid.n_weeks} weeks -- is worth {split.total:+.1f} model points "
        f"({split.bye_cover:+.1f} from covering the bye, {split.matchup:+.1f} from matchups; "
        f"{split.priced:+.1f} in weeks with a posted line, {split.unpriced:+.1f} without), "
        f"{ev.delta_points:+.1f} simulated points-for, {ev.delta_title * 100:+.2f}pp of title "
        f"(+/-{ev.stderr * 100:.2f}). That is the number `delta_title` carries; it is not "
        "the value of the single move above."
    )
    lines.append(
        f"Making only this week's move and then never touching the slot again is worth "
        f"{ev.commit_delta * 100:+.2f}pp (+/-{ev.commit_stderr * 100:.2f}); the rest is the "
        "value of re-solving every week, which you only collect if you actually do."
    )
    return " ".join(lines)


def recommend(
    sim,
    *,
    position_id: int = DST,
    kappa: int = 1,
    acquisition_cost: float | Mapping[int, float] | np.ndarray = 0.0,
    extra_slot_cost: float = 0.0,
    n_sims: int | None = None,
    seed: int = 11,
    market: MarketSchedule | None = None,
    calibration: CalibrationSet | None = None,
    horizon: int | None = None,
    through_week: int | None = None,
    readd_dropped: bool = False,
    cache: nflverse.NflverseCache | None = None,
) -> Recommendation:
    """The surface: a `pipeline.LeagueSim` in, one comparable `Recommendation` out.

    `sim` is a `pipeline.LeagueSim`; it is untyped here only to keep this module
    importable without pulling the ESPN client in. The recommendation is the *first
    week's* action of the rest-of-season plan, which is the only part you execute now,
    priced by what the whole plan is worth -- because dropping your defense this week is
    only correct if the rest of the schedule agrees.

    Read `delta_title` accordingly: it is the value of *adopting the plan*, not of the
    single move in `Recommendation.move`. When week one's action is a hold the move is
    worth nothing today and the recommendation carries `no-action-this-week` to say so;
    the rationale always quotes what making only this week's move is worth on its own.

    `kappa` is raised to however many players you already roster at the position when
    that is more than you asked for. A roster with two quarterbacks is the ordinary
    case, and refusing to plan for it -- which is what this did -- is a surface that
    fails on the league it was written for rather than an honest constraint.
    """
    state: S.LeagueState = sim.state
    my_team_id = state.my_team_id
    if my_team_id is None:
        raise StreamingError("the league state has no my_team_id; pass one to pipeline.build")
    market = market or market_schedule(state.season, cache=cache)
    ownership = {pid: f.team_id for f in state.franchises for pid in f.player_ids}
    weeks = tuple(w for w in state.weeks if through_week is None or w <= through_week)
    if not weeks:
        raise StreamingError(f"no remaining weeks at or before week {through_week}")
    grid = build_grid(
        sim.outlooks,
        league_id=state.league_id,
        season=state.season,
        position_id=position_id,
        weeks=weeks,
        ownership=ownership,
        my_team_id=my_team_id,
        market=market,
        readd_dropped=readd_dropped,
    )
    rostered = len(grid.held_index)
    if rostered > kappa:
        log.info(
            "team %d already rosters %d at %s; planning with kappa=%d rather than refusing",
            my_team_id,
            rostered,
            grid.model.label,
            rostered,
        )
        kappa = rostered
    plan = (
        solve(grid, kappa=kappa, acquisition_cost=acquisition_cost, extra_slot_cost=extra_slot_cost)
        if horizon is None
        else rolling_plan(
            grid,
            horizon=horizon,
            kappa=kappa,
            acquisition_cost=acquisition_cost,
            extra_slot_cost=extra_slot_cost,
        )
    )
    base = hold_plan(grid)
    ev = evaluate(
        state,
        sim.outlooks,
        grid,
        plan,
        baseline=base,
        my_team_id=my_team_id,
        n_sims=n_sims or sim.n_sims,
        seed=seed,
        market=market,
        calibration=calibration,
    )
    return as_recommendation(grid, plan, base, ev)


def as_recommendation(
    grid: StreamGrid, plan: StreamPlan, base: StreamPlan, ev: StreamEvaluation
) -> Recommendation:
    """Package a solved plan as the common currency, honestly labelled.

    Two labels here exist because the obvious packaging is misleading, and both were
    added after reading what this returned on the live leagues.

    `delta_title` prices the *plan*, and `move` is only its first week. On all three
    real D/ST grids week one's action is a hold, so a reader who takes `delta_title` as
    the value of `move` is told that doing nothing is worth +2.3 to +4.4pp. The
    `no-action-this-week` tag marks exactly that case, and `StreamEvaluation.commit_delta`
    -- quoted in the rationale -- is what this week's move is worth on its own.

    A position whose fitted within-week spread is under `STREAMABLE_SD` has no measured
    matchup signal to act on, so the only transaction it can justify is covering a bye.
    Without that gate the kicker grids emitted *drop Evan McPherson for Jake Elliott* on
    a week-one edge of 0.02 points at R^2 = 0.022, priced at "+1.00pp, significant" off
    the season plan while the simulated value of making that swap and holding it was
    -0.27pp. The plan still shows the bye week; the transaction is suppressed.
    """
    add, drop = plan.start[0], base.start[0]
    streamable = grid.model.fitted_sd >= STREAMABLE_SD
    covers_bye = drop < 0
    suppressed = not streamable and add != drop and not covers_bye
    if suppressed:
        add = drop
    me = grid.my_team_id
    players: tuple[PlayerMove, ...] = ()
    if add < 0 or add == drop:
        kind = MoveKind.HOLD
    elif grid.streamers[add].mine:
        # Already on the roster: this is a start/sit call, not an acquisition.
        kind = MoveKind.LINEUP
    else:
        kind = MoveKind.ADD_DROP
        players = (PlayerMove(player_id=grid.streamers[add].player_id, from_team=None, to_team=me),)
        # Drop the incumbent you are replacing, not an arbitrary held candidate.
        drop_i = drop if drop >= 0 else (grid.held_index[0] if grid.held_index else -1)
        if drop_i >= 0:
            players += (
                PlayerMove(player_id=grid.streamers[drop_i].player_id, from_team=me, to_team=None),
            )

    move = Move(
        kind=kind,
        league_id=grid.league_id,
        players=players,
        bid=None,  # rolling waiver priority: the cost is the priority slot, not money
        lineup={POSITION_SLOT.get(grid.position_id, 0): grid.streamers[add].player_id}
        if add >= 0
        else None,
    )
    # A model with no market term is not "unpriced" -- it never wanted a line. Only the
    # positions whose fit actually uses one can be short of it.
    short_of_market = grid.model.uses_market and not all(grid.priced)
    tags = ["streaming", grid.model.label.lower().replace("/", ""), "waiver-priority"]
    if not streamable:
        tags.append("not-streamable")
    if short_of_market:
        tags.append("partially-unpriced")
    if kind is MoveKind.HOLD:
        tags.append("no-action-this-week")
    if suppressed:
        tags.append("action-suppressed")
    if plan.start == base.start and plan.n_acquisitions == 0:
        # The plan and the baseline are the same season, so the paired difference is
        # identically zero and `stderr` is exactly 0.0 -- which `Recommendation.significant`
        # reads as "no Monte Carlo error, therefore significant". It is the opposite:
        # nothing was measured. The tag is the only place that distinction can live,
        # since the property is part of the shared contract.
        tags.append("null-plan")
    if not plan.optimal:
        tags.append(plan.method)
    confidence = "low" if not streamable else ("medium" if short_of_market else "high")
    rationale = _rationale(grid, plan, base, ev, suppressed=suppressed)
    if not streamable:
        rationale += (
            f" Note: the fitted within-week spread for {grid.model.label} is only "
            f"{grid.model.fitted_sd:.2f} points (R^2 {grid.model.r2:.3f}), so this position is "
            "not meaningfully streamable and the plan should be treated as a tie-break."
        )
    return Recommendation(
        move=move,
        delta_title=ev.delta_title,
        delta_points=ev.delta_points,
        stderr=ev.stderr,
        leverage=ev.leverage,
        rationale=rationale,
        confidence=confidence,
        tags=tuple(tags),
    )


def plan_table(grid: StreamGrid, plan: StreamPlan, base: StreamPlan | None = None) -> list[dict]:
    """One row per week: who to start, what it is worth, and what it replaces."""
    base = base or hold_plan(grid)
    plan_v, base_v = plan.week_value(grid), base.week_value(grid)
    rows = []
    for j, w in enumerate(grid.weeks):
        i, b = plan.start[j], base.start[j]
        rows.append(
            {
                "week": w,
                "start": grid.streamers[i].name if i >= 0 else "(empty)",
                "opponent_priced": grid.priced[j],
                "value": float(plan_v[j]),
                "hold": grid.streamers[b].name if b >= 0 else "(empty)",
                "hold_value": float(base_v[j]),
                "gain": float(plan_v[j] - base_v[j]),
                "add": tuple(grid.streamers[k].name for k in plan.acquired[j]),
                "drop": tuple(grid.streamers[k].name for k in plan.dropped[j]),
            }
        )
    return rows


__all__ = [
    "MARKET_COVERAGE",
    "MATCHUP_MODELS",
    "STREAMABLE_SD",
    "GapReport",
    "MarketSchedule",
    "MatchupModel",
    "StreamEvaluation",
    "StreamGrid",
    "StreamPlan",
    "Streamer",
    "StreamingError",
    "ValueSplit",
    "as_recommendation",
    "brute_force",
    "build_grid",
    "evaluate",
    "hold_plan",
    "market_schedule",
    "optimality_gap",
    "plan_table",
    "recommend",
    "relax_assignment",
    "rolling_plan",
    "solve",
    "upper_bound",
]
