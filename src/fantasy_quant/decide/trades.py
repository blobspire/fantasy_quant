"""Trade discovery: search the league for deals, then decide whether they are deals.

Everything else in this system grades a trade someone already thought of. This module
looks for them. That is the whole reason it exists, and it is why the search half and
the verdict half are separated: the search is allowed to be approximate and wide, the
verdict is not allowed to be either.

**The objective is not chart-sum equality.** It is

    delta(starting-lineup points), with a FREE-AGENT FLOOR, playoff-weighted,
    subject to a strict Pareto improvement for every team involved.

Chart equality is a negotiation heuristic. Two rosters can swap equal "value" and both
get worse, and two rosters can swap wildly unequal value and both get better, because
what a roster is worth is what it *starts*, and what it starts depends on the rest of
the roster. So every candidate is priced by re-solving each affected team's optimal
lineup week by week against `sim/lineup.py`'s free-agent floor, and any candidate where
even one side's gain is <= 0 is discarded before it is ever ranked.

**Where the consolidation credit actually comes from.** On 3,003 community-judged
trades, crediting a freed roster spot lifted verdict accuracy from 61% to 82% -- more
than improving the underlying player values did. The usual implementation bolts on a
constant (FantasyCalc's "a roster spot is worth 425"). That constant is unnecessary
here, because the free-agent floor already produces the effect and produces it in the
right unit:

* A player who is worse than the best available free agent at his slot is worth exactly
  zero to the team giving him up. So the throw-away half of a 2-for-1 is free to
  surrender, and the consolidating side keeps the whole incoming upgrade. A chart model
  charges him full sticker price, which is why chart models hate 2-for-1s.
* The mirror image is charged explicitly: the side receiving two players for one is over
  its roster limit and must cut, and `TradeFinder.settle` performs that cut by greedy
  leave-one-out against the same objective. Nobody's roster grows for free.
* The freed spot itself is then re-filled from the wire, also explicitly, and on all
  three of the user's leagues it comes back worth **exactly zero** -- `settle` never
  signs anybody. That is a theorem, not a measurement artifact: the floor at a slot is
  the best body you could hold there (`wire_pool`), and signing that body cannot beat a
  floor that already assumes you have him. So the 425-point roster spot is real in the
  sense that the *surrendered depth* is free, and not in the sense that the empty seat
  is an asset. It becomes an asset only at a position where the league's wire is
  genuinely empty and the floor is therefore zero, which is why the loop still runs.

**Positional surplus decay.** Incoming value is multiplied by `1 - (rostered -
required)/rostered`, which is just `required/rostered` -- a fourth startable back on a
roster that already has three is worth 58% of his marginal value to that team, and a
fifth 47%. `rostered` counts bodies that beat the wire, not bodies. A headcount screen
charges a team with four sub-replacement running backs a 47% surplus discount on the
back it is starving for, and talks it out of the one trade that helps it.

**Playoff weighting is points-conserving.** `w_p = 1.2` on the bracket weeks and
`w_n = (total_weeks - w_p * playoff_count) / non_playoff_count` on the rest, so the
weights sum to the week count and the scheme reallocates emphasis rather than inflating
every trade by 20%. A weighting that does not conserve makes every trade look better
than doing nothing, which is the failure mode worth designing against.

**Multi-team search is a preference graph with multi-edges.** Each team points at the
teams owning its top-k most-wanted available assets, *all k of them*, and simple cycles
up to length four are enumerated by DFS. Single-edge top-trading-cycles -- each team
points at exactly one team -- degenerates: a published NBA experiment had 17 of 30 teams
pointing at the same team, and fantasy preferences are more concentrated, not less. That
prediction held, and by a wider margin than the NBA figure. Measured on the user's own
leagues at week 1 of 2026:

    ===================  =========================  ====================  ==============
    league               single-edge concentration  single-edge 3-cycles  multi-edge k=8
    ===================  =========================  ====================  ==============
    Wine Wednesday (14)  10 of 14 point at team 13                     0              71
    Blacksburg (12)       8 of 12 point at team 14                     0              41
    Type shi (12)         9 of 12 point at team 4                      0              58
    ===================  =========================  ====================  ==============

Every league's single-edge graph collapses to one 2-cycle and nothing else. There is no
three-way trade to be found in it at all, in any of the three leagues, so a single-edge
implementation would have reported "no multi-team trades exist" and been wrong every
time. `single_edge_targets` is kept as a live diagnostic so the claim can be re-checked
rather than believed.

**Framing.** A rendered proposal leads with what the counterparty *receives*. Sellers
fixate on the good and buyers on the price; an offer written as "I'll give you X for Y"
puts the counterparty's loss first and reads as a demand. `TradeEvaluation.pitch` writes
it the other way round.

What this module deliberately does not do: buy-low/sell-high as a timing heuristic. The
rule is buy undervalued and sell overvalued at any price level, which is what pricing
every asset against the same free-agent floor already implements.

**The two tiers rank differently, and the search must not let the screen decide.** The
screen's objective is a deterministic mean lineup, so surrendered bench depth is worth
exactly zero to it. The confirmation runs against `sim/season.py`'s `ex_ante_rank`, whose
availability mask varies by simulation, so depth there has real option value. The two
orderings are therefore not the same ordering. Measured on the user's own leagues at
20,000 simulations over every Pareto candidate the search returns, the correlation
between the screen's points delta and the confirmed title delta runs from **-0.43 to
+0.72** depending on league and seed, and it is negative in every Wine Wednesday run --
the user's worst team, and the one with the most bench depth for a 2-for-1 to strip. So
the screen is a recall filter and the paired simulation is the ranker: `find_trades`
confirms the whole screened set rather than the screen's top eight, because with a
top-eight cut the best trade in the league was repeatedly outside it (best overall
+0.25pp, best inside the screen's top eight +0.14pp, winner at screen rank 33 of 51).

**What it finds.** Enough to be worth running, and not enough to be worth overselling.
At week 1 of a freshly drafted season the best confirmed trade is +1.4pp of title
probability in Blacksburg (6.4% -> 7.8%, +/-0.15 at 40,000 simulations), about +0.9pp in
Type shi and about +0.4pp in Wine Wednesday. Those are real and they are also the
argmax of forty candidates, which is why `selection_threshold` and not a two-sigma test
decides what gets called significant, and why `_confidence` will not say anything above
"low" about a trade whose measured title delta is negative.

The remaining honest caveat is not statistical. Every one of those trades needs a
counterparty to accept a leg it gains one to three playoff-weighted points from, and the
Blacksburg headline asks a rival to send Jaylen Waddle for a backup quarterback. That
passes the gate as specified -- the objective is Pareto improvement, and the improvement
is genuine on the rival's own lineup -- and no human being accepts it. `min_gain` raises
the gate from *Pareto* to *negotiable* and is plumbed through `find_trades` for exactly
this; at `min_gain=10` almost nothing survives in any league. A caller who wants the
list a manager would actually send should raise it and expect a short answer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

from ..core import (
    Move,
    MoveKind,
    PlayerMove,
    PlayerOutlook,
    Recommendation,
    leverage,
)
from ..data.etr import EtrRankings
from ..sim import season as S
from ..sim.distributions import Draw, WeeklySampler
from ..sim.lineup import LineupPlan, monotone_floor, plan_from_slots
from .opinion import tilt_outlooks
from .wire import all_rostered

if TYPE_CHECKING:  # pragma: no cover - only for the convenience constructor's type
    from ..pipeline import LeagueSim

log = logging.getLogger(__name__)

#: Weight on a bracket week. The rest of the season is re-weighted to conserve the
#: total, so this is a reallocation of emphasis and not a 20% bonus on every trade.
PLAYOFF_WEIGHT = 1.2

#: How close two leave-one-out values have to be before `settle` calls them equal. The
#: usual case is BIT-identical -- a bench player who never starts contributes exactly
#: zero -- so these only catch the near-ties that float arithmetic manufactures.
_TIE_ATOL = 1e-9
_TIE_RTOL = 1e-12

#: Measured on 23,999 paired player-weeks: the SD of (my score - opponent's score).
#: Used only as the fallback when a live tensor is not available to measure it from.
MEASURED_SD_DIFF = 34.4

#: How many of a rival's assets a team points at in the preference graph. Single-edge
#: (k=1) is the degenerate case this default exists to avoid; see the module docstring.
#:
#: Research says k ~ 5, and 5 is too small *here* because half the slots go to the
#: surplus ranking (`_blend`) and the other half concentrate on the same two or three
#: owners. Measured on Blacksburg, a 12-team league: at k=5 the user's own team appears
#: in **zero** cycles of any length and the surface has nothing to say; at k=8 it appears
#: in 11 and the search returns 20 Pareto three-ways. k=12 finds no better trade than
#: k=8 does, so 8 is where the curve flattens rather than where it was convenient.
DEFAULT_TOP_K = 8

#: Players one team may send in one leg. Two covers the 2-for-1 consolidation case,
#: which is the shape the verdict layer was built for.
DEFAULT_MAX_PACKAGE = 2

#: Cycle length cap. 2 is a bilateral trade, 3 a three-way, 4 a four-way. Beyond four
#: the enumeration grows faster than the chance anyone agrees to the trade.
DEFAULT_MAX_TEAMS = 3

#: How many of one owner's players may enter one leg's package pool. Separate from
#: `DEFAULT_TOP_K` on purpose: k sets how many *teams* a suitor points at and is the
#: knob the degeneracy is about, while this sets how wide a package can be built and is
#: the knob the search budget is about. Tying them together forces a choice between a
#: graph too sparse to hold a cycle and packages too numerous to enumerate.
DEFAULT_PER_LEG = 6

#: How many holdable free agents at a position the wire is worth in one week. See
#: `best_free_agents`: this is a model of rolling waiver priority (one claim per run),
#: not a smoothing parameter, and the user's three leagues all run priority.
DEFAULT_WIRE_DEPTH = 3

#: How far an outside ranking set moves the trade board when one is supplied.
#: Mirrors `decide/waivers.DEFAULT_BOARD_WEIGHT`, and carries the same warning: this is
#: the one input in the repo shipping without a measured verdict, because Establish The
#: Run overwrites each chart in place and publishes no history to back-test against.
#: `0.0` is byte-identical to not having the board.
DEFAULT_RANKINGS_WEIGHT = 1.0

POSITION_ABBREV: Mapping[int, str] = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}


class TradeError(ValueError):
    """The trade, or the market it was posed against, is malformed."""


# --------------------------------------------------------------------------------------
# Playoff weighting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeekWeights:
    """Per-week emphasis, summing to the number of weeks.

    Conservation is the whole point. An un-conserved scheme -- 1.2 on the bracket and
    1.0 everywhere else -- inflates every candidate's gain by the same factor, so the
    Pareto gate stops being a gate and "do nothing" stops being a fair comparison.
    """

    weeks: tuple[int, ...]
    playoff_weeks: frozenset[int]
    playoff_weight: float
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.weeks):
            raise TradeError("one weight per week")
        if not math.isclose(sum(self.weights), float(len(self.weeks)), rel_tol=1e-9):
            raise TradeError(
                f"playoff weighting must conserve points: weights sum to "
                f"{sum(self.weights)} over {len(self.weeks)} weeks"
            )

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64)

    def weight_of(self, week: int) -> float:
        return self.weights[self.weeks.index(week)]


def playoff_weights(
    weeks: Sequence[int],
    playoff_weeks: Iterable[int],
    *,
    playoff_weight: float = PLAYOFF_WEIGHT,
) -> WeekWeights:
    """`w_p` on the bracket weeks, `w_n` on the rest, conserving the total.

    `w_n = (total - w_p * n_p) / n_n`. With no bracket weeks left, or no regular-season
    weeks left, there is nothing to reallocate between and every week weighs 1.0 --
    which is right: a trade made in week 16 is already all playoff.
    """
    order = tuple(weeks)
    playoffs = frozenset(w for w in playoff_weeks if w in set(order))
    n_p = len(playoffs)
    n_n = len(order) - n_p
    if n_p == 0 or n_n == 0:
        return WeekWeights(order, playoffs, 1.0, tuple(1.0 for _ in order))
    w_n = (len(order) - playoff_weight * n_p) / n_n
    if w_n <= 0:
        raise TradeError(
            f"playoff_weight {playoff_weight} over {n_p} of {len(order)} weeks leaves "
            f"the regular season a non-positive weight ({w_n:.3f})"
        )
    return WeekWeights(
        weeks=order,
        playoff_weeks=playoffs,
        playoff_weight=playoff_weight,
        weights=tuple(playoff_weight if w in playoffs else w_n for w in order),
    )


# --------------------------------------------------------------------------------------
# Positional surplus
# --------------------------------------------------------------------------------------


def positional_requirements(
    slot_counts: Mapping[int, int], slot_eligibility: Mapping[int, Iterable[int]]
) -> dict[int, float]:
    """How many of each position a team needs to start, splitting flex slots evenly.

    A neutral split rather than the solved flex share from `decide/valuation.py`: this
    number only feeds the surplus *screen*, the lineup objective settles the real
    question, and pulling the flex fixed point into a search loop would cost more than
    the precision is worth here.
    """
    out: dict[int, float] = {}
    for slot, count in slot_counts.items():
        eligible = tuple(slot_eligibility.get(slot, ()))
        if not eligible or count <= 0:
            continue
        share = float(count) / len(eligible)
        for pos in eligible:
            out[pos] = out.get(pos, 0.0) + share
    return out


def surplus_multiplier(rostered: float, required: float) -> float:
    """`1 - (rostered - required)/rostered`, which is just `required/rostered`.

    A team starting 2.33 backs and rostering four values a fourth back at 58% of his
    standalone worth; at five, 47%. Clamped to 1.0 below the requirement -- a team short
    of starters does not get a *bonus*, it gets the player's full value.
    """
    if rostered <= 0:
        return 1.0
    return min(1.0, max(0.0, required / rostered))


# --------------------------------------------------------------------------------------
# Significance under selection
# --------------------------------------------------------------------------------------


#: Family-wise error rate the selection-adjusted threshold controls.
SELECTION_ALPHA = 0.05


def selection_threshold(n_candidates: int, *, alpha: float = SELECTION_ALPHA) -> float:
    """How many standard errors a *selected* delta has to clear to mean anything.

    `core.Recommendation.significant` is a two-sigma test, and two sigma is the right
    test for one pre-specified candidate. It is the wrong test for the candidate that
    *won a search*, and the difference is not academic. Measured on the user's own Type
    shi league at 4,000 simulations, the top-ranked trade came back at +1.12pp +/- 0.49
    -- 2.3 sigma, reported as significant. The same trade at 40,000 simulations is
    +0.53pp +/- 0.13, and at 4,000 simulations under two other seeds it is +0.90pp and
    +0.18pp. The point estimate was inflated roughly twofold by being the maximum of
    eight noisy draws, and the two-sigma label was the argmax passing a test built for a
    single draw.

    So the threshold is Bonferroni over however many candidates the ranking chose from:
    `z = Phi^-1(1 - alpha/(2n))`. One candidate gives back 1.96 and the ordinary test;
    eight gives 2.73; fifty gives 3.20. Deliberately conservative -- the candidates are
    positively correlated, so this over-corrects -- because the cost of a false positive
    here is the user actually proposing the trade.
    """
    n = max(int(n_candidates), 1)
    return float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * n)))


def _resolve(delta: float, stderr: float, z: float, falls: str) -> str:
    """`falls` or `f"{falls}-unclear"` -- two names for one already-negative delta.

    The distinction the surface was missing. A paired title delta near zero has a sign,
    and printing that sign as a flat assertion is the difference between a measurement
    and a coin flip: measured on the live leagues, of **50 counterparty impacts that read
    negative, not one cleared the selection-adjusted threshold** and only 4 cleared even
    a naive two sigma, while the tag was asserted on 45 trades.

    `z` rather than 2.0 because the trade being labelled won a search -- the same
    argument `selection_threshold` makes, applied to the side effects of the winner and
    not only to the winner. A zero standard error means the estimate carries no Monte
    Carlo at all (an unconfirmed or structurally exact impact), and there the sign is the
    answer.
    """
    return falls if _resolved_loss(delta, stderr, z) else f"{falls}-unclear"


def _resolved_loss(delta: float, stderr: float, z: float) -> bool:
    """Whether a non-positive paired delta is a measurement rather than a sign.

    A zero standard error means no Monte Carlo went into the number -- an unconfirmed or
    structurally exact impact -- and there the sign is the answer.

    `>=` rather than `>` so that `z = 0` is EXACTLY the bare sign test this replaced,
    which is what makes the negative control a control: at zero threshold a delta of
    exactly 0.0 has to resolve rather than fall through to "cannot tell". At any real
    threshold the two spellings differ only on exact equality.
    """
    return stderr <= 0.0 or abs(delta) >= z * stderr


# --------------------------------------------------------------------------------------
# The trade itself
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeLeg:
    """One package moving from one team to another. Multi-player by construction."""

    from_team: int
    to_team: int
    player_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.from_team == self.to_team:
            raise TradeError(f"a leg cannot start and end at team {self.from_team}")
        if not self.player_ids:
            raise TradeError("a leg with no players is not a leg")


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A trade as a set of legs, which makes 2-team and N-team the same object.

    The legs of a discovered trade form a cycle -- each team gives to the team that
    wants what it has -- but nothing here requires that, so a hand-written proposal with
    any leg structure can be evaluated by the same machinery.
    """

    league_id: int
    legs: tuple[TradeLeg, ...]

    def __post_init__(self) -> None:
        if not self.legs:
            raise TradeError("a proposal needs at least one leg")
        seen: set[int] = set()
        for leg in self.legs:
            for pid in leg.player_ids:
                if pid in seen:
                    raise TradeError(f"player {pid} moves twice in one trade")
                seen.add(pid)

    @property
    def teams(self) -> tuple[int, ...]:
        return tuple(sorted({t for leg in self.legs for t in (leg.from_team, leg.to_team)}))

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    def received_by(self, team_id: int) -> tuple[int, ...]:
        return tuple(p for leg in self.legs if leg.to_team == team_id for p in leg.player_ids)

    def given_by(self, team_id: int) -> tuple[int, ...]:
        return tuple(p for leg in self.legs if leg.from_team == team_id for p in leg.player_ids)

    def to_move(self) -> Move:
        return Move(
            kind=MoveKind.TRADE,
            league_id=self.league_id,
            players=tuple(
                PlayerMove(player_id=pid, from_team=leg.from_team, to_team=leg.to_team)
                for leg in self.legs
                for pid in leg.player_ids
            ),
        )

    @classmethod
    def from_move(cls, move: Move) -> TradeProposal:
        """The inverse of `to_move`, so a `Move` from any surface can be priced here."""
        if move.kind is not MoveKind.TRADE:
            raise TradeError(f"{move.kind} is not a trade")
        legs: dict[tuple[int, int], list[int]] = {}
        for pm in move.players:
            if pm.from_team is None or pm.to_team is None:
                raise TradeError("a trade leg cannot involve the waiver wire")
            legs.setdefault((pm.from_team, pm.to_team), []).append(pm.player_id)
        return cls(
            league_id=move.league_id,
            legs=tuple(
                TradeLeg(from_team=a, to_team=b, player_ids=tuple(sorted(p)))
                for (a, b), p in sorted(legs.items())
            ),
        )


@dataclass(frozen=True, slots=True)
class TeamImpact:
    """What one team gets out of one trade, in its own units.

    `delta_points` is the objective: playoff-weighted starting-lineup points with the
    free-agent floor, after the forced cut and the wire re-fill. `delta_title` is filled
    in only by the paired CRN confirmation, and is the number that travels across
    leagues.
    """

    team_id: int
    name: str
    received: tuple[int, ...]
    given: tuple[int, ...]
    dropped: tuple[int, ...]
    added: tuple[int, ...]
    before_points: float
    after_points: float
    delta_title: float = 0.0
    delta_title_stderr: float = 0.0
    #: `cut_alternatives[i]` is every player that was EXACTLY as good to cut as
    #: `dropped[i]`. Usually not empty: 94% of forced cuts on the live leagues are ties,
    #: because a deep-bench player who never starts is worth bit-identical zero to the
    #: screen objective. Surfaced rather than hidden, since "cut this one" and "cut any
    #: of these five" are different pieces of advice.
    cut_alternatives: tuple[tuple[int, ...], ...] = ()
    #: The same two numbers under the counterparty's own valuation -- the projections
    #: they are actually looking at -- when a second opinion was supplied. Zero when
    #: there is no second opinion, which is also when `market_delta_points` is zero and
    #: means nothing; read `has_market` first.
    market_before_points: float = 0.0
    market_after_points: float = 0.0

    @property
    def delta_points(self) -> float:
        return self.after_points - self.before_points

    @property
    def has_market(self) -> bool:
        """Whether a second valuation priced this side at all."""
        return self.market_before_points != 0.0 or self.market_after_points != 0.0

    @property
    def market_delta_points(self) -> float:
        """What this trade looks like to the team being asked to accept it.

        Not a prediction that they will. It is the same objective on the same settled
        roster, priced through the projections they can see -- which, in this repo,
        happens to be exactly what every surface computed before a second opinion
        existed, because `pipeline.league_projections` reads ESPN's own numbers.
        """
        return self.market_after_points - self.market_before_points

    @property
    def significant(self) -> bool:
        if self.delta_title_stderr <= 0:
            return True
        return abs(self.delta_title) > 2.0 * self.delta_title_stderr


@dataclass(frozen=True, slots=True)
class TradeEvaluation:
    """A priced proposal. `pareto` is the gate; everything else is explanation."""

    proposal: TradeProposal
    impacts: tuple[TeamImpact, ...]
    #: Post-trade rosters after the forced cut and the wire re-fill, per team.
    rosters: Mapping[int, tuple[int, ...]]
    names: Mapping[int, str] = field(default_factory=dict, repr=False)
    confirmed: bool = False
    #: `team_id -> (sims,)` per-simulation championship difference against the shared
    #: baseline, kept from `confirm_titles`. Two candidates confirmed on the same draw
    #: can therefore be differenced against EACH OTHER as a paired sample, which is the
    #: only honest way to ask "is this one really better than that one" -- comparing two
    #: published means and their standard errors throws away the pairing that makes the
    #: whole confirmation worth running. Not part of the value; float32 to keep forty
    #: candidates at a few megabytes.
    paired: Mapping[int, np.ndarray] | None = field(default=None, compare=False, repr=False)

    @property
    def pareto(self) -> bool:
        """Strict improvement for every team. The gate, not a preference."""
        return all(i.delta_points > 0.0 for i in self.impacts)

    @property
    def title_pareto(self) -> bool:
        """Whether every side also gained *title* probability once simulated.

        Reported rather than gated on, and the two answers routinely disagree. Points
        are what a trade transfers; title probability is a bracket laid on top, and it is
        not monotone in points -- a below-average team can raise its title odds by taking
        on variance it would have to *pay* for in expected points, and a locked-in
        favourite can gain points that buy it nothing. `pareto` is the gate; this is the
        reality check on it, and the check fails far more often than "occasionally":
        across the user's three leagues at 4,000 simulations, **24 to 27 of every 40**
        confirmed Pareto candidates have at least one counterparty whose simulated title
        odds go down. `TradeFinder.recommend` therefore tags those `counterparty-loses`
        and `_rationale` names them, because the pitch quotes that side its points gain
        and a caller reading only the pitch would be quoting a number the module's own
        authoritative tier contradicts.

        Still not the gate. The per-side deltas are individually noisy at affordable
        simulation counts, and a gate on a noisy quantity is a gate on noise.
        """
        return self.confirmed and all(i.delta_title > 0.0 for i in self.impacts)

    @property
    def min_gain(self) -> float:
        return min(i.delta_points for i in self.impacts)

    @property
    def n_teams(self) -> int:
        return self.proposal.n_teams

    def impact_for(self, team_id: int) -> TeamImpact:
        for i in self.impacts:
            if i.team_id == team_id:
                return i
        raise TradeError(f"team {team_id} is not in this trade")

    @property
    def has_market(self) -> bool:
        """Whether a second valuation priced any side of this trade."""
        return any(i.has_market for i in self.impacts)

    def spread(self, team_id: int) -> float:
        """How much more the other side thinks it is gaining than it is, in points.

        The arbitrage, stated as the thing it actually is. For every counterparty, the
        gap between what their own projections say the trade does for them and what
        ours say it does -- summed, because a three-team cycle has two of them.

        Positive is the case worth having: they read the deal as better for them than
        it is, so the trade is cheap to get signed. Negative means we are the ones
        paying up, which is worth seeing rather than hiding.

        Our own side is deliberately absent. The spread is a statement about the
        disagreement, and our side has nothing to disagree with -- `delta_points`
        already carries what we believe, and `Recommendation.delta_title` carries what
        it is worth. Zero when there is only one opinion in the room, which is also
        when it means nothing: check `has_market` first.
        """
        return sum(
            i.market_delta_points - i.delta_points
            for i in self.impacts
            if i.team_id != team_id and i.has_market
        )

    def pitch(self, team_id: int) -> str:
        """The offer written from the counterparties' side of the table.

        Sellers fixate on the good and buyers on the price, so an offer that opens with
        what you want reads as a demand. Every other team's haul is named first, and
        only then what it costs them.

        **Their gain is quoted in their own numbers when we have them.** It used to be
        quoted in ours, which is a number the person reading the offer cannot reproduce
        from anything on their screen -- and the one figure in the message they are most
        likely to go and check. Ours is the right number for deciding whether to send
        the offer; theirs is the right number for writing it.
        """
        me = self.impact_for(team_id)
        parts = []
        for impact in self.impacts:
            if impact.team_id == team_id:
                continue
            gets = ", ".join(self._label(p) for p in impact.received)
            gives = ", ".join(self._label(p) for p in impact.given)
            theirs = impact.market_delta_points if impact.has_market else impact.delta_points
            parts.append(f"{impact.name} gets {gets} (+{theirs:.1f} pts) for {gives}")
        mine = ", ".join(self._label(p) for p in me.received) or "nothing"
        return "; ".join(parts) + f". You get {mine} (+{me.delta_points:.1f} pts)."

    def _label(self, player_id: int) -> str:
        return self.names.get(player_id, str(player_id))


# --------------------------------------------------------------------------------------
# The preference graph
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreferenceEdge:
    """`team_id` wants `player_id`, which `owner_id` has.

    `value` is what the suitor's own starting lineup gains from him, decayed for
    positional surplus; `cost` is what the owner's lineup loses without him. The graph is
    ordered on `value` -- what a team *wants*, which is what a preference is -- and not
    on `gain`, and the difference decides what the whole search can see.

    Ordering on `gain` looks smarter and is a trap. It ranks by where the surplus is, so
    a genuine star is never a target (both marginals are large and cancel) and the only
    edges left point at backup quarterbacks and kickers, whose owners lose nothing by
    parting with them. Run that way against the user's real leagues, every trade the
    search could find was bye-week patchwork worth two to four points a season, and the
    obvious 2-for-1 -- send two starters, get a first-round back -- was not enumerable
    because the back never entered a package pool. `gain` stays as the pruning signal,
    where being a difference is exactly what you want.
    """

    team_id: int
    owner_id: int
    player_id: int
    value: float
    cost: float

    @property
    def gain(self) -> float:
        """How much more the suitor's lineup wants him than the owner's lineup keeps him."""
        return self.value - self.cost

    def __lt__(self, other: PreferenceEdge) -> bool:
        return self.value < other.value


def _blend(ranked: Sequence[PreferenceEdge], n: int) -> tuple[PreferenceEdge, ...]:
    """Half the slots to what a team *wants*, half to where the *surplus* is.

    Not a hedge: the two rankings answer different halves of one question, and a graph or
    a package pool built from either alone cannot express a real trade. Measured on the
    user's three real leagues, and both failure modes are total rather than marginal:

    * **Want only.** Every pool is stars, every package is star-for-star, and
      star-for-star is close to zero-sum once both sides' lineups are re-solved. All
      three leagues returned **nothing at all** -- zero Pareto trades at two, three or
      four teams.
    * **Surplus only.** A star is never in a pool, because his suitor's gain and his
      owner's loss are both large and cancel. What survives is backup-quarterback and
      kicker churn: the best trade in Wine Wednesday was Quentin Johnston for Daniel
      Jones, worth 3.5 points across seventeen weeks, and Jahmyr Gibbs was not
      enumerable in any package because no leg pool could hold him.

    A real package is one of each -- the player the suitor is chasing, and the piece
    whose owner will not miss it -- so both rankings have to reach the pool. `ranked` is
    already in want order; the surplus half is re-sorted here.

    Re-measured after `_paper_gain` was corrected to an actual bound, because the numbers
    above were taken while the prune was silently discarding real trades and both
    failure modes were therefore overstated. What survives: **want-only still returns
    zero Pareto trades in Wine Wednesday and Blacksburg**, which is the load-bearing
    half. What does not: want-only now finds forty in Type shi, and surplus-only finds
    forty in all three rather than kicker churn. So the blend is the only variant that
    works in every league, but it is no longer dominant everywhere -- want-only reaches a
    larger best-for-the-user candidate in Type shi. Treat the split as the safe default
    rather than a settled optimum.
    """
    if n <= 0:
        return ()
    wanted = list(ranked[: (n + 1) // 2])
    spare = sorted(ranked, key=lambda e: (-e.gain, e.player_id))[: n // 2]
    seen: dict[int, PreferenceEdge] = {}
    for edge in (*wanted, *spare):
        seen.setdefault(edge.player_id, edge)
    return tuple(seen.values())


def single_edge_targets(edges: Mapping[int, Sequence[PreferenceEdge]]) -> dict[int, int]:
    """Who each team would point at if it were allowed only its single best edge.

    A diagnostic, kept because the degeneracy it measures is the reason the graph is
    multi-edge. Count the values: if most of the league points at one owner, single-edge
    top-trading-cycles has nothing to cycle through and collapses to bilateral swaps.
    """
    out: dict[int, int] = {}
    for team, team_edges in edges.items():
        if not team_edges:
            continue
        out[team] = max(team_edges).owner_id
    return out


def simple_cycles(
    adjacency: Mapping[int, Iterable[int]], max_length: int = DEFAULT_MAX_TEAMS
) -> list[tuple[int, ...]]:
    """Every simple directed cycle of length 2..`max_length`, each listed once.

    Canonical by minimum node: a cycle is only emitted from its lowest-ranked member and
    never extends into a node ranked below that, so rotations collapse to one
    representative. Both *directions* survive, which is correct -- `a->b->c` and
    `a->c->b` move different players and are different trades.
    """
    if max_length < 2:
        raise TradeError(f"a cycle needs at least two teams, got {max_length}")
    nodes = sorted(adjacency)
    rank = {n: i for i, n in enumerate(nodes)}
    out: list[tuple[int, ...]] = []

    def walk(start: int, path: list[int], on_path: set[int]) -> None:
        for nxt in sorted(set(adjacency.get(path[-1], ()))):
            if nxt not in rank or rank[nxt] < rank[start]:
                continue
            if nxt == start:
                if len(path) >= 2:
                    out.append(tuple(path))
                continue
            if nxt in on_path or len(path) >= max_length:
                continue
            on_path.add(nxt)
            path.append(nxt)
            walk(start, path, on_path)
            path.pop()
            on_path.discard(nxt)

    for node in nodes:
        walk(node, [node], {node})
    return out


# --------------------------------------------------------------------------------------
# The free-agent floor
# --------------------------------------------------------------------------------------


def best_free_agents(
    state: S.LeagueState,
    outlooks: Sequence[PlayerOutlook],
    *,
    rank: int = 1,
    depth: int = DEFAULT_WIRE_DEPTH,
) -> dict[int, np.ndarray]:
    """position -> what the wire is worth at that position in each remaining week.

    "Free agent" is defined against this league's own rosters: anyone with a projection
    who is not in `state.pool`. That is the only definition that makes the floor a
    league-specific number, which it must be -- a 14-team full-PPR wire is a different
    wire from a 12-team half-PPR one.

    **`depth` is not a tuning knob, it is a model of the acquisition rules,** and getting
    it wrong moves every number in this module. Taking the per-week maximum over the
    whole wire -- 371 unrostered players in Wine Wednesday -- models a manager who signs
    whichever kicker projects best *this* week, every week, at every position at once.
    Measured on that league, it prices the kicker slot at 8.44 points a week when the
    best free-agent kicker anyone could actually hold averages 6.3, and the quarterback
    slot at 14.48 against a best holdable 12.8. All three of the user's leagues run
    rolling **waiver priority** and none uses FAAB: one claim per run, and using it drops
    you to last. Nobody gets the whole wire every week.

    So the pool is the top `depth` holdable bodies at the position, ranked over the whole
    horizon, and the weekly floor is the best of those in that week. `depth=1` is "sign
    one body and hold him" and floors a bye week at zero; the default of 3 covers the
    bye with the one claim a priority league actually affords. `depth` past about five
    converges back on the unlimited-wire model and should not be used in a league with
    no FAAB.

    `rank > 1` slides the whole pool down to the *marginal* wire body rather than the
    best one, for a caller who does not believe even the top of the wire is reachable.
    """
    return {
        pos: np.max(np.stack([mu for _, mu in bodies]), axis=0)
        for pos, bodies in wire_pool(state, outlooks, rank=rank, depth=depth).items()
    }


def wire_pool(
    state: S.LeagueState,
    outlooks: Sequence[PlayerOutlook],
    *,
    rank: int = 1,
    depth: int = DEFAULT_WIRE_DEPTH,
) -> dict[int, tuple[tuple[int, np.ndarray], ...]]:
    """position -> the `(player id, weekly means)` the floor is built from, best first.

    Exposed rather than inlined into `best_free_agents` because the floor and the list of
    bodies a short roster may sign have to be *the same players*. When they were not --
    the floor ranked on the horizon total and the signing list on the playoff-weighted
    one -- a body outside the floor pool could beat the floor in a single week, so a
    2-for-1 quietly signed a defense that is not in `state.pool` and the confirmation
    simulation died on it. Same pool, and that is impossible by construction.

    **Rostered means on a franchise, not in the pool.** This read
    `set(state.pool.player_ids)`, which is accidentally right under `pipeline.build` --
    that pools only rostered players, so the two sets are identical on all three live
    leagues -- and wrong the moment anything widens the pool. Under `waivers.augment`,
    which adds the sixty best free agents so they have columns to be simulated in, every
    one of those sixty was then counted as ROSTERED and the floor was read off the dregs
    behind them: measured on the live leagues it came back 0.74 to 8.58 points a week too
    low at every position, and at kicker **0.289 against a true 8.868**, because all
    thirty plausible free-agent kickers had been pulled into the pool.

    Scope: no production path hands this an augmented state today -- `find_trades` is
    only ever called with a `pipeline.build` sim -- so the fix is latent and the test
    below is what keeps it that way.
    """
    rostered = all_rostered(state)
    weeks = state.weeks
    rows: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    for o in outlooks:
        if o.player_id in rostered:
            continue
        mu = _mean_vector(o, weeks)
        # Ranked on the horizon total, then player id, so the pool is a set of real
        # players a manager could roster rather than a per-week envelope of the field.
        rows.setdefault(o.position_id, []).append((-float(mu.sum()), o.player_id, mu))
    out: dict[int, tuple[tuple[int, np.ndarray], ...]] = {}
    for pos, pool in rows.items():
        pool.sort(key=lambda r: (r[0], r[1]))
        start = min(max(rank, 1) - 1, len(pool) - 1)
        out[pos] = tuple((pid, mu) for _, pid, mu in pool[start : start + max(depth, 1)])
    return out


def slot_floor_matrix(
    fa_by_position: Mapping[int, np.ndarray],
    slot_ids: Sequence[int],
    slot_eligibility: Mapping[int, Iterable[int]],
    n_weeks: int,
) -> np.ndarray:
    """`(weeks, slots)` free-agent floor: the best wire body each slot could start.

    Monotone at the *position* level by construction -- a wider slot maxes over a
    superset -- but not necessarily at the *player* level on a roster missing a
    position, which is what `LineupPlan._check_monotone` actually tests. So every solve
    re-lifts this through `monotone_floor` against its own plan rather than trusting it.
    """
    floors = np.zeros((n_weeks, len(slot_ids)), dtype=np.float64)
    for col, slot in enumerate(slot_ids):
        eligible = [p for p in slot_eligibility.get(slot, ()) if p in fa_by_position]
        if not eligible:
            continue
        floors[:, col] = np.max(np.stack([fa_by_position[p] for p in eligible]), axis=0)
    return floors


def _mean_vector(outlook: PlayerOutlook, weeks: Sequence[int]) -> np.ndarray:
    """A player's projected mean per week, zero where he does not play.

    Zero rather than skipped: a bye is a week the roster spot is idle, and against a
    positive floor a zero never starts, so byes and absences need no special case.
    """
    out = np.zeros(len(weeks), dtype=np.float64)
    for i, week in enumerate(weeks):
        wo = outlook.weeks.get(week)
        if wo is not None and wo.playing:
            out[i] = wo.mean
    return out


# --------------------------------------------------------------------------------------
# The finder
# --------------------------------------------------------------------------------------


class TradeFinder:
    """Search and verdict for one league at one moment.

    Two tiers, as `core.MoveEvaluator` prescribes. The screen is deterministic: solve
    every affected team's optimal lineup on projected means against the free-agent
    floor, week by week, and weight the weeks. It costs microseconds and is what the
    search runs on. The confirmation re-simulates the whole league against the *same*
    pre-drawn tensor, so a candidate and the status quo meet identical football and the
    paired difference isolates the trade.

    Not a frozen dataclass because it is mostly cache: lineup plans keyed by a roster's
    position multiset, and objective values keyed by the roster itself. Both hit
    constantly -- a four-way trade search evaluates the same 15-man roster from a dozen
    different candidate paths.
    """

    def __init__(
        self,
        state: S.LeagueState,
        draw: Draw,
        outlooks: Sequence[PlayerOutlook],
        *,
        my_team_id: int | None = None,
        playoff_weight: float = PLAYOFF_WEIGHT,
        floor_rank: int = 1,
        wire_depth: int = DEFAULT_WIRE_DEPTH,
        efficiency: S.LineupEfficiency | None = None,
        market: TradeFinder | None = None,
        roster_limit: int | None = None,
    ) -> None:
        self.state = state
        self.draw = draw
        self.outlooks = tuple(outlooks)
        #: A sibling finder built on the projections the LEAGUE can see, when this one
        #: is built on something else. Everything about it is identical except `_mu`, so
        #: "what does this look like to them" is the same objective on the same settled
        #: roster rather than a second model of anything.
        self.market = market
        self.my_team_id = my_team_id if my_team_id is not None else state.my_team_id
        #: Whose side of the table this search is being run from. `search` sets it;
        #: it defaults to this finder's own team so that `evaluate` and `settle` are
        #: consistent when called directly -- through the `core.MoveEvaluator` pair, say
        #: -- rather than settling a counterparty on our numbers and then pricing him on
        #: his. With no `market` it is irrelevant.
        self._subject: int | None = self.my_team_id
        # Symmetric by default, deliberately. The asymmetric haircut turns every one of
        # the user's below-average teams into a title favourite (see sim/season.py), and
        # a trade surface built on that would recommend standing pat.
        self.efficiency = efficiency or S.LineupEfficiency.symmetric()

        playoff_weeks = {w for rnd in state.playoff_rounds for w in rnd}
        self.weights = playoff_weights(state.weeks, playoff_weeks, playoff_weight=playoff_weight)
        self._w = self.weights.vector

        self.slot_ids = tuple(sorted(state.lineup_slot_counts))
        self._wire = wire_pool(state, self.outlooks, rank=floor_rank, depth=wire_depth)
        self.fa_by_position = {
            pos: np.max(np.stack([mu for _, mu in bodies]), axis=0)
            for pos, bodies in self._wire.items()
        }
        self.week_floor = slot_floor_matrix(
            self.fa_by_position, self.slot_ids, state.slot_eligibility, len(state.weeks)
        )
        #: One number per slot for the simulator, which takes a season-constant floor.
        self.season_floor: Mapping[int, float] = {
            slot: float(self.week_floor[:, col].mean()) for col, slot in enumerate(self.slot_ids)
        }
        self.requirements = positional_requirements(
            state.lineup_slot_counts, state.slot_eligibility
        )

        self._mu: dict[int, np.ndarray] = {}
        self._pos: dict[int, int] = {}
        self._name: dict[int, str] = {}
        for o in self.outlooks:
            self._mu[o.player_id] = _mean_vector(o, state.weeks)
            self._pos[o.player_id] = o.position_id
            self._name[o.player_id] = o.name or str(o.player_id)
        for pid, pos in zip(state.pool.player_ids, state.pool.position_ids, strict=True):
            self._pos.setdefault(pid, pos)
            self._mu.setdefault(pid, np.zeros(len(state.weeks)))
            self._name.setdefault(pid, state.pool.name(pid))

        self.rosters: dict[int, tuple[int, ...]] = {
            f.team_id: tuple(f.player_ids) for f in state.franchises
        }
        #: How many players a team may hold. `roster_limit` is the league's real
        #: `starter_count + bench_slots`, which `decide/waivers.py` already reads off
        #: settings; without it this falls back to each team's CURRENT size, which is
        #: what it always used to be and which quietly asserts that nobody has an open
        #: spot. A team one short of the limit can take an add without cutting anybody,
        #: and modelling that away made every acquisition look like it costs a player.
        self.capacity: dict[int, int] = {
            t: max(len(r), roster_limit) if roster_limit else len(r)
            for t, r in self.rosters.items()
        }
        self.team_names: dict[int, str] = {f.team_id: f.name for f in state.franchises}

        self._plans: dict[tuple[int, ...], tuple[LineupPlan, np.ndarray, np.ndarray]] = {}
        self._rankings: dict[int, tuple[PreferenceEdge, ...]] = {}
        self._weekly: dict[tuple[int, ...], np.ndarray] = {}
        self._adds = self._wire_candidates()
        self._base_scores: np.ndarray | None = None
        self._base_result: S.SeasonResult | None = None
        self._eff_factors: np.ndarray | None = None
        self._sd_diff: float | None = None

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_sim(
        cls,
        sim: LeagueSim,
        *,
        rankings: EtrRankings | None = None,
        rankings_weight: float = DEFAULT_RANKINGS_WEIGHT,
        roster_limit: int | None = None,
        **kwargs,
    ) -> TradeFinder:
        """Build from `pipeline.build`'s output. This is how you get a live league.

        With `rankings`, TWO finders are built and the pair is the arbitrage: this one
        on the re-dealt projections, which is what we believe, and `self.market` on the
        league's own, which is what the counterparty sees. Without them, one finder and
        `market is None`, which is exactly today.

        Applied here rather than in `pipeline.build` on purpose. Every surface reads
        `sim.outlooks`, so re-dealing them upstream would move the odds table and the
        streaming plan too; this board's remit is trades.

        `roster_limit` is read off the league when it can be, the same way
        `waivers.waiver_board` does it. A failure there is not fatal -- the finder
        falls back to current roster sizes, which is what it did before.
        """
        if roster_limit is None:
            roster_limit = _roster_limit(sim)
        kwargs["roster_limit"] = roster_limit
        if rankings is None or rankings_weight == 0.0:
            return cls(sim.state, sim.draw, sim.outlooks, **kwargs)
        outlooks = tilt_outlooks(
            sim.outlooks, rankings, weight=rankings_weight, weeks=sim.state.weeks
        )
        # The confirm scores on the DRAW, not on `_mu`, so the draw has to come from the
        # same projections the screen reads or the two halves are in different
        # currencies. Measured on the live Blacksburg board with `sim.draw` reused: the
        # screen's top row was +17.8 playoff-weighted points and the simulation priced
        # it at +0.10pp +/- 0.36 -- roughly a tenth of what a point buys there --
        # because the tensor still carried ESPN's means. Same seed, same size; it is
        # a different universe from `fq odds` either way and is labelled as such.
        panel = S.panel_for(sim.state, outlooks)
        # `pipeline.build` panels with ESPN's bye table and this call has no access to
        # it; a defence's bye-week outlook still says `playing=True`, so re-panelling
        # from the outlooks alone would put every D/ST back on the field in week 8.
        # The original draw's panel has the right `has_game` on identical axes.
        base = sim.draw.panel
        if (
            np.array_equal(base.player_ids, panel.player_ids)
            and np.array_equal(base.weeks, panel.weeks)
        ):
            panel = replace(panel, has_game=base.has_game)
        else:  # pragma: no cover - the axes are fixed by `state.pool` and `state.weeks`
            log.warning("re-panelled tensor axes differ from the built draw; byes may be lost")
        draw = WeeklySampler(panel, seed=sim.seed).draw(sim.n_sims)
        # The market sibling never simulates -- only `value_of` is ever read off it --
        # so it can carry the original draw without that draw ever being scored.
        market = cls(sim.state, sim.draw, sim.outlooks, **kwargs)
        return cls(sim.state, draw, outlooks, market=market, **kwargs)

    def _side(self, team_id: int) -> TradeFinder:
        """The finder whose numbers `team_id` is reading.

        Ours for the subject, the league's own for everyone else -- and ours for
        everyone when there is only one opinion, which makes every path through here
        identical to the single-finder code it replaced.

        This has to reach every place the search prices a COUNTERPARTY, not just the
        gate. Measured on the live Blacksburg board with the gate alone: the two-opinion
        screen returned 38 trades against 40 and admitted **zero** arbitrage rows,
        because `_paper_gain` and `asset_ranking` were still pruning the other side on
        our numbers and a package that only clears because they overrate their own
        player was thrown away before `evaluate` ever saw it.
        """
        if self.market is None or self._subject is None or team_id == self._subject:
            return self
        return self.market

    def _wire_candidates(self) -> tuple[int, ...]:
        """Exactly the bodies the floor is built from -- see `wire_pool` for why.

        Which makes the answer to "what is a freed roster spot worth" a *theorem* rather
        than a constant: the floor at a slot is the best of these bodies in that week, so
        signing one of them can never beat the floor, and the seat prices out at zero.
        The loop in `settle` still runs, because it is the measurement, and because it
        does find real value at a position where this league's wire is genuinely empty
        and the floor is therefore zero.
        """
        return tuple(pid for bodies in self._wire.values() for pid, _ in bodies)

    # -- the objective -----------------------------------------------------------------

    def live_slots(self, positions: Iterable[int]) -> tuple[dict[int, int], tuple[int, ...]]:
        """Split the starting slots into ones this roster can fill and ones it cannot.

        This split is not cosmetic, and getting it wrong is the most expensive bug this
        module can have. `monotone_floor` lifts every slot's floor to the highest floor
        among the slots *nested inside* it, and nesting is over players: a roster with no
        quarterback makes the QB slot's player set **empty**, an empty set is a subset of
        every other slot, and the whole lineup is then floored at the wire quarterback's
        14.5 points a week. Measured live before this split existed, that turned "trade
        your only QB for a kicker" into a +550-point, +82pp-of-title recommendation in
        Blacksburg -- a slot table's worth of free points conjured out of a set-theory
        edge case.

        A slot no rostered player can fill always scores exactly its own floor, so it is
        removed from the plan and added back as a constant. Same answer, no contamination.
        """
        eligibility = self.state.slot_eligibility
        present = set(positions)
        live: dict[int, int] = {}
        dead: list[int] = []
        for slot, count in self.state.lineup_slot_counts.items():
            if count <= 0:
                continue
            if present & set(eligibility.get(slot, ())):
                live[slot] = count
            else:
                dead.append(slot)
        return live, tuple(sorted(dead))

    def _plan_for(self, positions: tuple[int, ...]) -> tuple[LineupPlan, np.ndarray, np.ndarray]:
        """A plan over the fillable slots, its lifted floor, and the unfillable constant.

        Keyed by the roster's position multiset, which is what actually determines all
        three, and which repeats constantly across a candidate search.
        """
        hit = self._plans.get(positions)
        if hit is not None:
            return hit
        live, dead = self.live_slots(positions)
        index = {slot: col for col, slot in enumerate(self.slot_ids)}
        constant = sum(
            self.state.lineup_slot_counts[s] * self.week_floor[:, index[s]] for s in dead
        )
        if not isinstance(constant, np.ndarray):
            constant = np.zeros(len(self.state.weeks))
        plan = plan_from_slots(live, self.state.slot_eligibility, positions)
        keep = [index[s] for s in plan.floor_slot_ids]
        floor = monotone_floor(plan, self.week_floor[:, keep])
        self._plans[positions] = (plan, floor, constant)
        return plan, floor, constant

    def weekly_points(self, player_ids: Sequence[int]) -> np.ndarray:
        """`(weeks,)` optimal starting-lineup points, with unfilled slots at the wire."""
        # Deduped as well as sorted: the key has to be canonical for the cache to be
        # sound, and a roster holding the same player twice would stack his column twice
        # and let one body fill two slots.
        key = tuple(sorted(set(player_ids)))
        hit = self._weekly.get(key)
        if hit is not None:
            return hit
        if not key:
            # Every slot empty, so every slot scores its own floor -- once per *instance*
            # of the slot, which a plain column sum would get wrong for the two running
            # back seats.
            counts = np.array(
                [self.state.lineup_slot_counts.get(s, 0) for s in self.slot_ids], dtype=np.float64
            )
            total = self.week_floor @ counts
            self._weekly[key] = total
            return total
        positions = tuple(self._pos[p] for p in key)
        plan, floor, constant = self._plan_for(positions)
        mu = np.stack([self._mu[p] for p in key], axis=1)
        total = np.asarray(plan.solve(mu, floor=floor, assignment=False).total, dtype=np.float64)
        total = total + constant
        self._weekly[key] = total
        return total

    def value_of(self, player_ids: Sequence[int]) -> float:
        """The objective: playoff-weighted starting-lineup points over the horizon."""
        return float(self.weekly_points(player_ids) @ self._w)

    # -- roster settlement -------------------------------------------------------------

    def settle(
        self, team_id: int, player_ids: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Bring a post-trade roster back to legal size.

        Returns `(roster, cut, added, tied)`, where `tied[i]` is every player that was
        exactly as good to cut as `cut[i]`.

        This is where the consolidation credit is actually charged and paid, and both
        halves are explicit. Over the limit, the team cuts by greedy leave-one-out
        against the same objective -- receiving two for one is not free. Under it, the
        team signs the best wire body it can, which is the freed spot's value stated in
        points rather than asserted as a constant.

        **The cut used to be decided by ESPN's roster ordering.** `value_of` is a
        whole-roster starting-lineup objective, so a deep-bench player who never cracks a
        lineup contributes exactly zero and removing any of them leaves the objective
        BIT-IDENTICAL. `if value > best_value` keeps the first maximum, and `roster` is
        `dict.fromkeys(player_ids)` over a franchise built from ESPN's own ordering. So
        the answer to "who do you cut" was whoever ESPN happened to list first.

        It is not an edge case: measured on the three live leagues, **65 of 69 forced
        cuts (94%) had at least two bit-exact ties**, with a median tie group of three to
        five players and a maximum of six. The spread between the best and worst
        leave-one-out is 97-140 points, so the choice matters enormously in general --
        it is only among the *top* candidates that it is a dead heat.

        Ties break on the least valuable asset: lowest playoff-weighted rest-of-season
        points, then lowest player id so the result never depends on an input ordering
        at all. `startable_count` already writes `self._mu[pid] @ self._w` for exactly
        this quantity. The equals are returned rather than hidden, because "cut Tank
        Bigsby" and "cut any one of these five, they are indistinguishable" are different
        pieces of advice.
        """
        roster = list(dict.fromkeys(player_ids))
        limit = self.capacity.get(team_id, len(roster))
        cut: list[int] = []
        added: list[int] = []
        tied: list[tuple[int, ...]] = []

        while len(roster) > limit:
            scored = [(self.value_of([p for p in roster if p != pid]), pid) for pid in roster]
            best = max(v for v, _ in scored)
            # A relative tolerance rather than exact equality: letting a 1e-12 difference
            # in a ~100-point objective decide the cut is the same defect one level down.
            tol = max(_TIE_ATOL, _TIE_RTOL * abs(best))
            equals = sorted(pid for v, pid in scored if v >= best - tol)
            best_pid = min(equals, key=self._cut_priority)
            roster.remove(best_pid)
            cut.append(best_pid)
            tied.append(tuple(p for p in equals if p != best_pid))

        held = set(roster)
        while len(roster) < limit:
            base = self.value_of(roster)
            # Strictly positive by a float's width, not by zero: the usual answer here
            # is exactly zero -- the floor already assumes you signed the best free agent
            # -- and a 1e-14 rounding artifact must not read as a signing.
            best_pid, best_gain = None, 1e-6
            for pid in self._adds:
                if pid in held:
                    continue
                gain = self.value_of([*roster, pid]) - base
                if gain > best_gain:
                    best_pid, best_gain = pid, gain
            if best_pid is None:
                # Nothing on the wire beats the floor the empty slot already falls back
                # to, which is the usual answer and the honest one -- see the module
                # docstring on where the consolidation credit really comes from.
                break
            roster.append(best_pid)
            held.add(best_pid)
            added.append(best_pid)

        return tuple(roster), tuple(cut), tuple(added), tuple(tied)

    def _cut_priority(self, player_id: int) -> tuple[float, int]:
        """Sort key for choosing among equally-costly cuts: least valuable asset first.

        Playoff-weighted rest-of-season points, then the player id. The id is not a
        tie-break of convenience -- it is what guarantees the answer does not depend on
        the order ESPN returned the roster in, which is the whole defect.
        """
        return float(self._mu[player_id] @ self._w), int(player_id)

    # -- cheap asset values ------------------------------------------------------------

    def startable_count(self, team_id: int, position: int) -> int:
        """How many bodies a team has at a position that beat the wire.

        Counted against the free-agent floor rather than by headcount, because a roster
        carrying four running backs of whom three are worse than the waiver wire does
        not have a surplus, it has a hole. Charging it surplus decay -- as a headcount
        does -- is how a screen talks a starving team out of the trade it needs.
        """
        wire = self.fa_by_position.get(position)
        total = 0
        for pid in self.rosters.get(team_id, ()):
            if self._pos.get(pid) != position:
                continue
            mu = self._mu[pid]
            edge = mu if wire is None else np.maximum(mu - wire, 0.0)
            if float(edge @ self._w) > 0:
                total += 1
        return total

    def asset_value(self, team_id: int, player_id: int, *, incoming: bool) -> float:
        """What a player is worth to one team: his marginal starting-lineup points.

        Marginal against that team's own lineup, not against a global chart, which is
        the only reading under which the same player is genuinely worth different
        amounts to different rosters. Incoming value is then multiplied by the
        positional surplus decay -- `required/rostered` over the bodies that beat the
        wire -- so a fourth startable back is discounted for a team that already has
        three. Outgoing value is *not* decayed: what you lose is what your lineup loses,
        and discounting it would let a team talk itself into giving away its own depth.
        """
        roster = self.rosters.get(team_id, ())
        if player_id not in self._mu:
            return 0.0
        if not incoming:
            if player_id not in roster:
                return 0.0
            return self.value_of(roster) - self.value_of([p for p in roster if p != player_id])
        base = self.marginal_value(team_id, player_id)
        pos = self._pos.get(player_id, 0)
        count = self.startable_count(team_id, pos) + 1
        return base * surplus_multiplier(float(count), self.requirements.get(pos, 1.0))

    def marginal_value(self, team_id: int, player_id: int) -> float:
        """What adding this player does to a team's lineup, before any surplus decay.

        Split out from `asset_value` because the decay answers a different question from
        the raw number. The decayed value is a *preference* -- it is how the graph
        decides who a team chases, and the decay is what stops a roster with three
        startable backs from chasing a fourth. Anything that needs a *bound* wants this
        one instead: a bound multiplied by something at most one is no longer a bound.
        """
        if player_id not in self._mu:
            return 0.0
        roster = self.rosters.get(team_id, ())
        return self.value_of([*roster, player_id]) - self.value_of(roster)

    def asset_ranking(self, team_id: int) -> tuple[PreferenceEdge, ...]:
        """Every rival asset that would improve this team's lineup, most wanted first.

        Preference, not surplus -- see `PreferenceEdge` for the measurement that settled
        it. Cached, because the whole search reads this list and it costs one lineup
        solve per (team, player).
        """
        hit = self._rankings.get(team_id)
        if hit is not None:
            return hit
        # What this team wants is priced by the numbers THIS team reads, and what it
        # costs the owner by the numbers the OWNER reads. With one opinion both are
        # `self` and this is the original loop.
        wanting = self._side(team_id)
        edges: list[PreferenceEdge] = []
        for owner, roster in self.rosters.items():
            if owner == team_id:
                continue
            owning = self._side(owner)
            for pid in roster:
                value = wanting.asset_value(team_id, pid, incoming=True)
                if value <= 0:
                    continue
                edges.append(
                    PreferenceEdge(
                        team_id=team_id,
                        owner_id=owner,
                        player_id=pid,
                        value=value,
                        cost=owning.asset_value(owner, pid, incoming=False),
                    )
                )
        edges.sort(key=lambda e: (-e.value, e.player_id))
        self._rankings[team_id] = tuple(edges)
        return self._rankings[team_id]

    def preference_edges(
        self, teams: Sequence[int] | None = None, *, top_k: int = DEFAULT_TOP_K
    ) -> dict[int, tuple[PreferenceEdge, ...]]:
        """Each team's `top_k` most-wanted assets on other rosters. MULTI-edge.

        `top_k` sets how many *teams* a team can point at, which is the quantity the
        degeneracy is about: keeping all k -- rather than the single best, as textbook
        top-trading-cycles does -- is what keeps the graph from collapsing onto whoever
        happens to own the league's most misallocated player. How many of an owner's
        players end up in a package is a separate question, answered by `leg_assets`,
        because tying the two together would force a choice between a graph too sparse
        to hold a cycle and packages too wide to enumerate.
        """
        wanted = tuple(teams) if teams is not None else tuple(self.rosters)
        return {team: _blend(self.asset_ranking(team), top_k) for team in wanted}

    def leg_assets(self, team_id: int, owner_id: int, n: int) -> tuple[int, ...]:
        """The `n` assets on `owner_id`'s roster that could plausibly move to `team_id`."""
        ranked = [e for e in self.asset_ranking(team_id) if e.owner_id == owner_id]
        return tuple(e.player_id for e in _blend(ranked, n))

    # -- evaluation --------------------------------------------------------------------

    def evaluate(self, proposal: TradeProposal) -> TradeEvaluation:
        """Price one proposal on the screen objective. No simulation."""
        impacts: list[TeamImpact] = []
        rosters: dict[int, tuple[int, ...]] = {}
        for team in proposal.teams:
            current = self.rosters.get(team)
            if current is None:
                raise TradeError(f"no team {team} in league {self.state.league_id}")
            out = set(proposal.given_by(team))
            missing = out - set(current)
            if missing:
                raise TradeError(f"team {team} does not have {sorted(missing)}")
            after = [p for p in current if p not in out] + list(proposal.received_by(team))
            # Who a team cuts is that team's call, made on the numbers that team reads.
            # Settling a counterparty on ours would price a roster they would not hold.
            settled, cut, added, tied = self._side(team).settle(team, after)
            rosters[team] = settled
            # Then the SAME settled roster is priced twice, so the two numbers differ
            # only in the projections behind them -- which is the whole claim.
            impacts.append(
                TeamImpact(
                    team_id=team,
                    name=self.team_names.get(team, str(team)),
                    received=proposal.received_by(team),
                    given=proposal.given_by(team),
                    dropped=cut,
                    added=added,
                    cut_alternatives=tied,
                    before_points=self.value_of(current),
                    after_points=self.value_of(settled),
                    market_before_points=(
                        0.0 if self.market is None else self.market.value_of(current)
                    ),
                    market_after_points=(
                        0.0 if self.market is None else self.market.value_of(settled)
                    ),
                )
            )
        return TradeEvaluation(
            proposal=proposal,
            impacts=tuple(impacts),
            rosters=rosters,
            names={p: self._name.get(p, str(p)) for i in impacts for p in (*i.received, *i.given)},
        )

    # -- search ------------------------------------------------------------------------

    def search(
        self,
        *,
        for_team: int | None = None,
        max_teams: int = DEFAULT_MAX_TEAMS,
        top_k: int = DEFAULT_TOP_K,
        max_package: int = DEFAULT_MAX_PACKAGE,
        per_leg: int = DEFAULT_PER_LEG,
        min_gain: float = 0.0,
        max_cycles: int = 200,
        per_cycle: int = 24,
        limit: int = 40,
    ) -> list[TradeEvaluation]:
        """Enumerate cycles, build packages, keep the ones that help everybody.

        Three filters in increasing cost order, which is the only way this is tractable:
        the preference graph decides who could plausibly trade with whom, the summed
        marginal values prune packages that cannot clear the Pareto gate even
        optimistically, and only the survivors pay for a full lineup re-solve.

        `min_gain` raises the gate from *Pareto* to *negotiable*, and the distinction is
        not academic. The strict gate is `> 0`, and on the user's real leagues it passes
        trades where the counterparty gains 0.4 playoff-weighted points across seventeen
        weeks. That is a genuine improvement and no human being alive accepts it. Ten
        points is roughly where a side has something to point at when it explains the
        trade to itself. The default stays at the strict gate, because that is the
        objective as specified and because the threshold is a claim about people rather
        than about football; `Recommendation.rationale` always names the weakest side's
        gain so the caller can apply their own.
        """
        target = for_team if for_team is not None else self.my_team_id
        if self.market is not None and target != self._subject:
            # The preference graph is priced from the subject's side of the table, so a
            # different subject is a different graph.
            self._subject = target
            self._rankings.clear()
        edges = self.preference_edges(top_k=top_k)
        adjacency = {team: {e.owner_id for e in team_edges} for team, team_edges in edges.items()}
        cycles = simple_cycles(adjacency, max_teams)
        if target is not None:
            cycles = [c for c in cycles if target in c]
        # A cycle's edge gains are the cheapest possible upper bound on how much trade
        # there is to be had around it, so they order the search budget.
        cycles.sort(key=lambda c: -self._cycle_gain(c, edges))
        cycles = cycles[:max_cycles]

        out: list[TradeEvaluation] = []
        for cycle in cycles:
            for proposal in self._cycle_proposals(cycle, per_leg, max_package, per_cycle):
                ev = self.evaluate(proposal)
                if self._passes(ev, target, min_gain):
                    out.append(ev)
        key = (
            (lambda e: -e.impact_for(target).delta_points)
            if target is not None
            else (lambda e: -e.min_gain)
        )
        out.sort(key=key)
        return _dedupe(out)[:limit]

    def _passes(self, ev: TradeEvaluation, target: int | None, min_gain: float) -> bool:
        """The gate. Pareto by default; the arbitrage when a second opinion exists.

        With one valuation there is one question -- does this help everybody -- and
        `TradeEvaluation.pareto` is it. With two there are two, and they are asked of
        different people: does it help ME under what I believe, and does it help THEM
        under what they can see. A trade that fails the second is not a trade, however
        good it looks from here, because nobody accepts it.

        `pareto` itself is left alone. It is a property of the evaluation and has no
        notion of whose board this is; the subject-aware gate belongs here, where
        `target` already exists. `market is None` collapses this to `pareto` exactly.
        """
        if self.market is None or target is None:
            return ev.pareto and ev.min_gain > min_gain
        # `pareto` is `> 0` and `min_gain` is a floor on the same quantity, so the
        # original gate is `> max(0, min_gain)` -- kept exactly, and applied to each
        # side through the valuation that side is reading.
        floor = max(min_gain, 0.0)
        for impact in ev.impacts:
            side = impact.delta_points if impact.team_id == target else impact.market_delta_points
            if not side > floor:
                return False
        return True

    def _cycle_gain(
        self, cycle: Sequence[int], edges: Mapping[int, Sequence[PreferenceEdge]]
    ) -> float:
        total = 0.0
        for i, team in enumerate(cycle):
            owner = cycle[(i + 1) % len(cycle)]
            total += max((e.gain for e in edges.get(team, ()) if e.owner_id == owner), default=0.0)
        return total

    def _cycle_proposals(
        self,
        cycle: Sequence[int],
        per_leg: int,
        max_package: int,
        limit: int,
    ) -> list[TradeProposal]:
        """Package choices around one cycle, pruned on marginal value as they are built.

        Leg `i` moves players from `cycle[i+1]` to `cycle[i]`, so team `cycle[i]`'s
        books close as soon as legs `i-1` and `i` are chosen. Pruning there rather than
        at the end is what keeps a four-way cycle from enumerating 50,000 packages.
        """
        length = len(cycle)
        pools: list[tuple[int, ...]] = []
        for i, team in enumerate(cycle):
            wanted = self.leg_assets(team, cycle[(i + 1) % length], per_leg)
            if not wanted:
                return []
            pools.append(wanted)

        packages: list[list[tuple[int, ...]]] = [
            [
                combo
                for size in range(1, max_package + 1)
                for combo in combinations(pool, min(size, len(pool)))
            ]
            for pool in pools
        ]
        # combinations() with size > len(pool) yields nothing, and min() above can repeat
        # the full pool; dedupe so a short pool does not double every branch.
        packages = [list(dict.fromkeys(p)) for p in packages]

        out: list[tuple[float, TradeProposal]] = []
        chosen: list[tuple[int, ...]] = []

        def descend(i: int) -> None:
            if len(out) >= limit * 8:
                return
            if i == length:
                team = cycle[0]
                if self._paper_gain(team, chosen[0], chosen[length - 1]) <= 0:
                    return
                legs = tuple(
                    TradeLeg(
                        from_team=cycle[(j + 1) % length],
                        to_team=cycle[j],
                        player_ids=chosen[j],
                    )
                    for j in range(length)
                )
                score = min(
                    self._paper_gain(cycle[j], chosen[j], chosen[(j - 1) % length])
                    for j in range(length)
                )
                out.append((score, TradeProposal(self.state.league_id, legs)))
                return
            for package in packages[i]:
                if i > 0:
                    team = cycle[i]
                    if self._paper_gain(team, package, chosen[i - 1]) <= 0:
                        continue
                chosen.append(package)
                descend(i + 1)
                chosen.pop()

        descend(0)
        out.sort(key=lambda pair: -pair[0])
        return [p for _, p in out[:limit]]

    def _paper_gain(self, team: int, incoming: Sequence[int], outgoing: Sequence[int]) -> float:
        """An optimistic bound on what this leg pair does to `team`. The cheap gate.

        The incoming players are priced against the roster **the outgoing package has
        already left**, and that detail is the whole correctness of the bound. Written
        the obvious way -- each side's marginal taken against the untouched roster and
        subtracted -- it is not optimistic, it is *pessimistic*, and the argument is one
        line of submodularity:

            exact = v(R - out + in) - v(R)
                  = [v(R - out + in) - v(R - out)] - [v(R) - v(R - out)]

        The first bracket is the incoming marginal against the *smaller* roster, which
        submodularity makes no smaller than the marginal against `R`. So the naive form
        is a lower bound on the exact re-solved gain, `descend`'s `<= 0` cut throws away
        trades that would have passed, and the docstring that claimed the opposite was
        describing a bound the code did not have. Measured on the user's three live
        leagues, the naive form understated the exact one-for-one gain in 144 to 166 of
        every 200 candidates tested, by as much as 65 playoff-weighted points -- the
        Blacksburg headline (Jaylen Waddle to the user, +1.5pp of title at 40,000
        simulations) was inside the discarded set and the search could not reach it.

        Taken against `R - out`, the outgoing side is exact and the incoming side is
        optimistic in the one direction a pruning bound is allowed to be: two incoming
        players are jointly worth no more than their separate marginals, so this can wave
        through a candidate that `evaluate` then rejects, and cannot discard one that
        would have worked. A one-for-one comes out exactly equal to the re-solved gain.
        """
        side = self._side(team)
        gone = set(outgoing)
        kept = [p for p in self.rosters.get(team, ()) if p not in gone]
        base = side.value_of(self.rosters.get(team, ()))
        after_out = side.value_of(kept)
        gets = sum(side.value_of([*kept, p]) - after_out for p in incoming)
        return after_out + gets - base

    # -- the simulation half -----------------------------------------------------------

    def franchise_scores(self, team_id: int, roster: Sequence[int]) -> np.ndarray:
        """`(sims, weeks)` weekly totals for one roster, floored at the wire.

        One franchise at a time rather than a whole league in one call, because the
        fillable-slot split is a property of the *roster*: `live_slots` has to remove
        this roster's unfillable slots before `sim/season.py` runs `monotone_floor` over
        them, or the contamination described there reappears inside the simulator, where
        it is much harder to see. The removed slots are added back as a constant, which
        is exactly what they score.

        No efficiency haircut is applied here; the caller multiplies by its own draw so
        that every candidate meets the same set of opposing managers.
        """
        positions = tuple(self._pos[p] for p in roster)
        live, dead = self.live_slots(positions)
        constant = sum(self.state.lineup_slot_counts[s] * self.season_floor[s] for s in dead)
        sub = replace(
            self.state,
            franchises=(self.state.franchise(team_id).with_players(roster),),
            remaining_games=(),
            playoff_rounds=(),
            playoff_team_count=0,
            lineup_slot_counts=live,
        )
        scores = S.team_week_scores(
            sub, self.draw, replacement={s: self.season_floor[s] for s in live}
        )
        return scores[:, :, 0] + float(constant)

    def _ensure_base(self) -> tuple[np.ndarray, S.SeasonResult]:
        if self._base_scores is None or self._base_result is None:
            n_sims = self.draw.points.shape[0]
            self._eff_factors = self.efficiency.draw(self.state, n_sims)
            scores = np.zeros((n_sims, len(self.state.weeks), self.state.size), dtype=np.float32)
            for t, f in enumerate(self.state.franchises):
                scores[:, :, t] = self.franchise_scores(f.team_id, f.player_ids)
            self._base_scores = scores * self._eff_factors[:, None, :]
            self._base_result = S.simulate_from_scores(
                self.state, self._base_scores, all_play=False
            )
        return self._base_scores, self._base_result

    def baseline_title(self, team_id: int) -> float:
        """P(championship) under no move, against the same tensor every candidate meets."""
        _, base = self._ensure_base()
        return float(base.champions.astype(np.float64)[:, self.state.team_index[team_id]].mean())

    @property
    def sd_diff(self) -> float:
        """SD of (my weekly score - my opponent's), measured on this league's own tensor.

        Reported next to the corpus constant of 34.4 rather than assuming it: the
        constant was measured on 12-team PPR, and a 14-team league with thinner rosters
        does not have to agree.
        """
        if self._sd_diff is None:
            scores, _ = self._ensure_base()
            measured = float(np.sqrt(2.0 * scores.var(axis=0).mean()))
            # A degenerate tensor -- one simulation, or a fixture with no variance -- would
            # otherwise hand `leverage` a zero and make every point look infinitely
            # valuable. That is the only case the corpus constant is for.
            usable = math.isfinite(measured) and measured > 0
            self._sd_diff = measured if usable else MEASURED_SD_DIFF
        return self._sd_diff

    def leverage_for(self, team_id: int) -> float:
        """Mean marginal value of a point over this team's remaining matchups.

        `core.leverage` at each scheduled game's projected margin. Near zero means the
        schedule has already decided the games and a points upgrade is buying very
        little win probability -- which is exactly when a trade is worth less than its
        headline points delta suggests.
        """
        index = self.state.week_index
        weekly = {t: self.weekly_points(r) for t, r in self.rosters.items()}
        values: list[float] = []
        for game in self.state.remaining_games:
            if team_id not in (game.home_team_id, game.away_team_id):
                continue
            other = game.away_team_id if game.home_team_id == team_id else game.home_team_id
            cols = [index[w] for w in game.weeks]
            margin = float(weekly[team_id][cols].sum() - weekly[other][cols].sum())
            values.append(leverage(margin, self.sd_diff))
        return float(np.mean(values)) if values else 1.0

    def confirm_titles(self, evaluations: Sequence[TradeEvaluation]) -> list[TradeEvaluation]:
        """Re-simulate each candidate against the same drawn season. Paired, so exact.

        Only the affected franchises' weekly totals are recomputed; every other team
        meets identical football, which is what makes the per-simulation difference a
        clean paired sample rather than two noisy independent estimates. The whole
        league is re-standing-ed regardless, because moving a player changes who his
        new team's opponents beat and therefore who takes the last seat in the bracket.
        """
        base_scores, base = self._ensure_base()
        base_champ = base.champions.astype(np.float64)
        factors = self._eff_factors
        assert factors is not None
        index = self.state.team_index

        pool = set(self.state.pool.player_ids)
        out: list[TradeEvaluation] = []
        for ev in evaluations:
            # A body signed off the wire by `settle` has no column in the tensor, so the
            # simulator cannot start him. He is dropped and the seat falls back to the
            # slot's replacement level, which is what he was worth: the floor is built
            # from exactly these players (`wire_pool`), so this can only ever discard
            # something already priced at zero.
            franchises = tuple(
                self.state.franchise(t).with_players([p for p in r if p in pool])
                for t, r in ev.rosters.items()
            )
            state = self.state
            for f in franchises:
                state = state.with_franchise(f)
            scores = base_scores.copy()
            for f in franchises:
                t = index[f.team_id]
                scores[:, :, t] = (
                    self.franchise_scores(f.team_id, f.player_ids) * factors[:, t : t + 1]
                )
            alt = S.simulate_from_scores(state, scores, all_play=False)
            alt_champ = alt.champions.astype(np.float64)

            impacts = []
            per_team: dict[int, np.ndarray] = {}
            for impact in ev.impacts:
                t = index[impact.team_id]
                paired = alt_champ[:, t] - base_champ[:, t]
                per_team[impact.team_id] = paired.astype(np.float32)
                impacts.append(
                    replace(
                        impact,
                        delta_title=float(paired.mean()),
                        delta_title_stderr=float(paired.std(ddof=1) / math.sqrt(paired.size)),
                    )
                )
            out.append(replace(ev, impacts=tuple(impacts), confirmed=True, paired=per_team))
        return out

    # -- recommendations ---------------------------------------------------------------

    def recommend(
        self,
        evaluations: Sequence[TradeEvaluation],
        *,
        for_team: int | None = None,
        parsimony: bool = True,
    ) -> list[Recommendation]:
        """One `core.Recommendation` per trade, from `for_team`'s side of the table."""
        team = for_team if for_team is not None else self.my_team_id
        if team is None:
            raise TradeError("no team to recommend for; pass for_team or set my_team_id")
        lever = self.leverage_for(team)
        # Every confirmed candidate in this list competed for the top of it, so the
        # winner's own standard error is not the right yardstick for the winner. See
        # `selection_threshold`.
        #
        # `z` is computed over the FULL set, before pruning: multiplicity is a fact
        # about how many candidates competed, and dropping some of them afterwards does
        # not un-compete them.
        n_confirmed = sum(1 for ev in evaluations if ev.confirmed)
        z = selection_threshold(n_confirmed)
        absorbed: dict[frozenset[tuple[int, int, int]], int] = {}
        if parsimony:
            evaluations, absorbed = prune_throw_ins(evaluations, team, z=z)
        out: list[Recommendation] = []
        for ev in evaluations:
            mine = ev.impact_for(team)
            tags = ["trade", f"{ev.n_teams}-team"]
            if ev.confirmed:
                tags.append("confirmed")
            else:
                tags.append("screened")
            # From this team's own side of the table, not the league's: "consolidation"
            # on a recommendation the user is on the receiving end of would read as the
            # opposite of what it is.
            if len(mine.given) > len(mine.received):
                tags.append("consolidating")
            elif len(mine.received) > len(mine.given):
                tags.append("expanding")
            # The arbitrage, tagged only when there are genuinely two opinions to
            # disagree. A sign test on a deterministic difference, not on a noisy
            # paired estimate -- both valuations price the same settled roster with no
            # simulation anywhere, so there is nothing here for a z to guard against.
            if ev.has_market:
                spread = ev.spread(team)
                tags.append("mispriced" if spread > 0.0 else "fairly-priced")
                tags.append(f"spread:{spread:+.2f}")
            if ev.confirmed:
                # The screen said Pareto and the simulation disagreed. Both of these
                # used to be BARE SIGN TESTS on a paired estimate the module's own
                # `TradeEvaluation.title_pareto` docstring already calls noise -- "the
                # per-side deltas are individually noisy at affordable simulation counts,
                # and a gate on a noisy quantity is a gate on noise" -- and then tagged
                # on exactly that. Measured across three seeds on the live leagues they
                # disagreed with themselves on 42-78% (`harmful`) and 35-75%
                # (`counterparty-loses`) of the same forty trades. So each gets three
                # states rather than two, against the selection-adjusted `z`.
                if mine.delta_title <= 0.0:
                    tags.append(_resolve(mine.delta_title, mine.delta_title_stderr, z, "harmful"))
                # STRICTLY negative, as before: a counterparty the trade leaves exactly
                # where it found it has not lost anything, and `min` over an empty
                # sequence is the only other thing that could go here.
                worst = min(
                    (i for i in ev.impacts if i.team_id != team and i.delta_title < 0.0),
                    key=lambda i: i.delta_title,
                    default=None,
                )
                if worst is not None:
                    tags.append(
                        _resolve(
                            worst.delta_title,
                            worst.delta_title_stderr,
                            z,
                            "counterparty-loses",
                        )
                    )
            if ev.title_pareto:
                tags.append("title-pareto")
            n_absorbed = absorbed.get(_move_set(ev), 0)
            if n_absorbed:
                tags.append(f"leanest:{n_absorbed}")
            # How many candidates this row beat, which is NOT how many are published.
            # Parsimony drops candidates after they have competed, and a selection
            # correction computed off the surviving list would quietly shrink the field
            # the winner won -- understating exactly the bias the correction exists for.
            if n_confirmed:
                tags.append(f"considered:{n_confirmed}")
            rec = Recommendation(
                move=ev.proposal.to_move(),
                delta_title=mine.delta_title,
                delta_points=mine.delta_points,
                stderr=mine.delta_title_stderr,
                leverage=lever,
                rationale=self._rationale(ev, team, z=z, n_candidates=n_confirmed),
                confidence=self._confidence(ev, mine, z=z),
                tags=tuple(tags),
            )
            out.append(rec)
        # Title probability is the unit and leads whenever every candidate has one. A
        # half-screened list has no common unit at all, so it falls back to points
        # rather than ranking a measured title delta against a structural zero.
        confirmed = all("confirmed" in r.tags for r in out)
        out.sort(key=(lambda r: -r.delta_title) if confirmed else (lambda r: -r.delta_points))
        return out

    def _confidence(self, ev: TradeEvaluation, mine: TeamImpact, *, z: float = 2.0) -> str:
        """How much of the recommendation to believe, on the confirmed number only.

        Three gates, and the first one is the one that was missing. A confirmed trade
        whose measured title delta is <= 0 is not a low-confidence *recommendation*, it
        is a recommendation against, and labelling it "high" because the measurement was
        precise inverts the word. The live run this was found on returned
        `delta_title=-0.80pp +/- 0.31, confidence="high"` -- a confidently wrong
        recommendation, which is the exact failure this surface exists to avoid.

        `z` is the selection-adjusted threshold from `selection_threshold`, not the
        two-sigma constant, because the trade being labelled is the one that won the
        ranking.
        """
        if not ev.confirmed:
            return "low"
        if mine.delta_title <= 0.0:
            return "low"
        if mine.delta_title_stderr > 0 and abs(mine.delta_title) <= z * mine.delta_title_stderr:
            return "low"
        return "high" if ev.n_teams == 2 else "medium"

    def _rationale(
        self, ev: TradeEvaluation, team: int, *, z: float = 2.0, n_candidates: int = 1
    ) -> str:
        mine = ev.impact_for(team)
        head = ev.pitch(team)
        if ev.has_market:
            # Two questions were asked, so two answers are given. `ev.min_gain` is the
            # weakest side under OUR numbers, and on an arbitrage row that is negative
            # by construction -- printing it as "every side gains" would be false.
            theirs = min(
                (i.market_delta_points for i in ev.impacts if i.team_id != team), default=0.0
            )
            ours_on_them = min(
                (i.delta_points for i in ev.impacts if i.team_id != team), default=0.0
            )
            spread = ev.spread(team)
            body = (
                f" You gain +{mine.delta_points:.1f} playoff-weighted pts by the analyst "
                f"board; the other side gains +{theirs:.1f} by the projections on their "
                f"screen (weakest side)."
            )
            if ours_on_them <= 0.0:
                body += (
                    f" By the analyst board that side is {ours_on_them:+.1f}, so this is "
                    f"the disagreement being traded on, not a deal that helps both by one "
                    f"account -- spread {spread:+.1f} pts."
                )
            else:
                body += f" Both boards call it a gain for them; spread {spread:+.1f} pts."
        else:
            weakest = ev.min_gain
            body = (
                f" Every side gains on its own starting lineup (weakest +{weakest:.1f} "
                f"playoff-weighted pts)."
            )
        if ev.confirmed:
            base = self.baseline_title(team)
            body += (
                f" Your title odds {base * 100:.1f}% -> {(base + mine.delta_title) * 100:.1f}% "
                f"({mine.delta_title * 100:+.2f}pp +/- {mine.delta_title_stderr * 100:.2f})"
            )
            se = mine.delta_title_stderr
            if se > 0 and abs(mine.delta_title) <= z * se:
                body += (
                    f", which is inside the {z:.1f}x standard error this ranking needs to "
                    f"clear once it has picked a winner out of {n_candidates} candidates "
                    f"-- not a significant edge"
                )
            body += "."
            if mine.delta_title <= 0.0:
                body += (
                    " The lineup screen calls this Pareto-improving and the simulation "
                    "disagrees about your side; do not propose it."
                )
            # The pitch quotes each counterparty its *points* gain, which is the gate.
            # If the simulation says that side's title odds fall anyway, saying so is the
            # difference between a trade offer and a trick. But it has to be able to
            # tell: this printed the standard error beside a number it had not tested
            # against it, and on the live leagues not one of the fifty negative
            # counterparty deltas cleared `z`. Separated into what is measured and what
            # is merely the sign of noise.
            down = [i for i in ev.impacts if i.team_id != team and i.delta_title <= 0.0]
            resolved = [i for i in down if _resolved_loss(i.delta_title, i.delta_title_stderr, z)]
            unclear = [i for i in down if i not in resolved]
            if resolved:
                body += " Simulated title odds fall for " + ", ".join(
                    f"{i.name} ({i.delta_title * 100:+.2f}pp +/- {i.delta_title_stderr * 100:.2f})"
                    for i in resolved
                )
                body += " even though its starting lineup gains points."
            if unclear:
                body += (
                    " Simulated title odds read slightly down for "
                    + ", ".join(
                        f"{i.name} ({i.delta_title * 100:+.2f}pp +/- "
                        f"{i.delta_title_stderr * 100:.2f})"
                        for i in unclear
                    )
                    + f", but none of those clears {z:.2f} standard errors, so the "
                    "simulation cannot tell whether that side gains or loses."
                )
        else:
            body += " Screened only; not yet confirmed by simulation."
        if mine.dropped:
            body += (
                " You would have to cut "
                + ", ".join(self._name.get(p, str(p)) for p in mine.dropped)
                + "."
            )
            # 94% of forced cuts are ties, because the screen objective values a
            # never-started bench player at bit-identical zero. Naming one player as
            # though the model had picked him out reads as advice it cannot support.
            equals = sorted({p for group in mine.cut_alternatives for p in group})
            if equals:
                body += (
                    " That cut is a tie: "
                    + ", ".join(self._name.get(p, str(p)) for p in equals)
                    + " cost exactly the same, so pick on something this model does not "
                    "see."
                )
        return head + body

    # -- the MoveEvaluator protocol ----------------------------------------------------

    def screen(self, moves: Sequence[Move]) -> list[Recommendation]:
        """Cheap deterministic estimate for arbitrary candidate trades."""
        evals = [self.evaluate(TradeProposal.from_move(m)) for m in moves]
        return self.recommend(evals)

    def confirm(self, moves: Sequence[Move]) -> list[Recommendation]:
        """Paired CRN simulation. Authoritative, and populates `stderr`."""
        evals = self.confirm_titles([self.evaluate(TradeProposal.from_move(m)) for m in moves])
        return self.recommend(evals)


def _roster_limit(sim: LeagueSim) -> int | None:
    """`starter_count + bench_slots`, or None when ESPN will not say.

    Read defensively and never fatal: the finder's previous behaviour -- capacity is
    whatever each roster currently holds -- is the fallback, so a settings failure costs
    the open-spot modelling and nothing else.
    """
    try:
        roster = sim.league.settings().roster
        return int(roster.starter_count) + int(roster.bench_slots)
    except Exception as err:  # pragma: no cover - live-only path
        log.info("no roster limit for league %s (%s); using current roster sizes", 
                 getattr(sim.state, "league_id", "?"), err)
        return None


def _move_set(ev: TradeEvaluation) -> frozenset[tuple[int, int, int]]:
    """Every `(from, to, player)` this proposal moves. The identity `_dedupe` uses."""
    return frozenset(
        (leg.from_team, leg.to_team, pid) for leg in ev.proposal.legs for pid in leg.player_ids
    )


def prune_throw_ins(
    evaluations: Sequence[TradeEvaluation], team: int, *, z: float = 2.0
) -> tuple[list[TradeEvaluation], dict[frozenset[tuple[int, int, int]], int]]:
    """Drop a trade when a strictly smaller version of it is worth the same.

    Returns `(kept, how many fatter versions each survivor absorbed)`.

    **The measurement this exists for.** On the live Type shi board, "Lawrence for Tate"
    and "Lawrence + Mahomes for Tate" were confirmed at 8,000 simulations under three
    seeds and came out +0.563pp, -0.175pp and +0.125pp apart -- the sign flips. The
    model cannot tell them apart, so which one reaches the top of the board is decided
    by the draw; and the one that won left four quarterbacks on a sixteen-man roster and
    forced Jordyn Tyson to be cut. Twenty of forty candidates had a strictly leaner
    sibling in the same search.

    So when one candidate's moves are a strict SUBSET of another's and the bigger one
    cannot be shown to be better, the bigger one is not a better trade -- it is the same
    trade with a throw-in the simulation cannot price, bought with an extra asset and a
    roster spot. A subset is the right relation because it needs no judgement: same
    deal, fewer players.

    The comparison is paired. Both candidates were simulated against the same baseline
    on the same draw, so their per-simulation differences subtract exactly and the
    standard error of `A - B` is far smaller than either one's own -- which is the whole
    reason `confirm_titles` keeps the arrays. Differencing the two published means
    instead would fail to separate almost anything.

    `z` is the caller's significance bar, normally `selection_threshold(n)`: the same
    bar the board already uses to decide whether its winner is real.
    """
    keys = [_move_set(ev) for ev in evaluations]
    absorbed: dict[frozenset[tuple[int, int, int]], int] = {}
    drop: set[int] = set()

    for i, big in enumerate(evaluations):
        mine = None if big.paired is None else big.paired.get(team)
        if mine is None:
            continue  # never confirmed; nothing to compare
        for j, small in enumerate(evaluations):
            if i == j or not (keys[j] < keys[i]):
                continue
            theirs = None if small.paired is None else small.paired.get(team)
            if theirs is None:
                continue
            diff = mine.astype(np.float64) - theirs.astype(np.float64)
            se = float(diff.std(ddof=1) / math.sqrt(diff.size))
            if float(diff.mean()) > z * se:
                continue  # the extra pieces really do buy something
            drop.add(i)
            absorbed[keys[j]] = absorbed.get(keys[j], 0) + 1
            break

    kept = [ev for i, ev in enumerate(evaluations) if i not in drop]
    return kept, absorbed


def _dedupe(evaluations: Sequence[TradeEvaluation]) -> list[TradeEvaluation]:
    """Collapse proposals that move the same players between the same teams.

    Two different cycles can reach the same trade -- `a->b->c` with an empty-handed leg
    is `a->c` -- and a ranked list that shows the same deal three times is worse than
    useless in front of a human.
    """
    seen: set[frozenset[tuple[int, int, int]]] = set()
    out: list[TradeEvaluation] = []
    for ev in evaluations:
        key = frozenset(
            (leg.from_team, leg.to_team, pid) for leg in ev.proposal.legs for pid in leg.player_ids
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def select_non_overlapping(
    evaluations: Sequence[TradeEvaluation], *, longest_first: bool = True
) -> list[TradeEvaluation]:
    """Greedily pick trades that share no team, longest cycle first.

    Longest-first is the published rule and is kept, but it is a rule about *clearing a
    market*, not about serving one manager: a four-way with three trivial gains outranks
    a two-way with a large one. Within a length the weakest side's gain breaks the tie,
    and `longest_first=False` sorts purely on that, which is what a single user wants.
    """
    ordered = sorted(
        evaluations,
        key=(lambda e: (-e.n_teams, -e.min_gain)) if longest_first else (lambda e: -e.min_gain),
    )
    taken: set[int] = set()
    out: list[TradeEvaluation] = []
    for ev in ordered:
        teams = set(ev.proposal.teams)
        if teams & taken:
            continue
        taken |= teams
        out.append(ev)
    return out


# --------------------------------------------------------------------------------------
# Front door
# --------------------------------------------------------------------------------------


def find_trades(
    sim: LeagueSim,
    *,
    for_team: int | None = None,
    max_teams: int = DEFAULT_MAX_TEAMS,
    top_k: int = DEFAULT_TOP_K,
    max_package: int = DEFAULT_MAX_PACKAGE,
    min_gain: float = 0.0,
    n_confirm: int | None = None,
    include_harmful: bool = False,
    finder: TradeFinder | None = None,
    rankings: EtrRankings | None = None,
    rankings_weight: float = DEFAULT_RANKINGS_WEIGHT,
    parsimony: bool = True,
) -> list[Recommendation]:
    """Search one live league and return confirmed trades, best first.

    **The screen is a recall filter, not a ranker, and `n_confirm` used to assume it was
    both.** The screen's objective is a deterministic mean lineup, so surrendered bench
    depth is worth exactly zero to it, while the confirmation's `ex_ante_rank` sees a
    per-simulation availability mask and therefore does price depth. The two orderings
    are consequently not the same ordering, and on the user's own leagues at 20,000
    simulations the rank correlation between them ranges from **-0.43 to +0.72**
    depending on the league and the seed -- negative in every Wine Wednesday run, which
    is the user's worst team and the one with the most bench depth to sell. With
    `n_confirm=8` the best trade in the league was routinely outside the confirmed set:
    in one Wine Wednesday run the best confirmed candidate overall was +0.25pp and the
    best inside the screen's top eight was +0.14pp, and the winner sat at screen rank 33
    of 51.

    So the default is now to confirm everything the screen returned (about forty
    candidates, five seconds at 4,000 simulations on a 14-team league) and let the paired
    simulation do the ranking it is the authority for. `n_confirm` is kept as a budget
    cap for a caller who cannot afford that, with the cost stated rather than hidden.

    Trades the confirmation says would *lower* this team's title probability are dropped
    rather than ranked last: they passed a points gate, they failed the unit that
    matters, and a list called "recommendations" should not contain them. Pass
    `include_harmful=True` to get the full ranked list for diagnostics.

    `min_gain` is forwarded to `TradeFinder.search` and raises the gate from *Pareto* to
    *negotiable*. It stays at the strict gate by default because that is the objective as
    specified, but it is reachable from here on purpose: the strict gate passes legs a
    counterparty gains half a playoff-weighted point from, and the difference between
    "improves their lineup" and "they would sign it" is the whole distance between this
    module's output and a trade that happens.

    `rankings` closes some of that distance, and is the only thing here that does. With
    a second opinion the gate stops asking one question of everybody and asks two: does
    this help ME under what I believe, and does it help THEM under what they can see. A
    trade that fails the second does not happen no matter how good it looks from here,
    and a trade that passes both is one the numbers on their screen argue for. That is
    as far as this goes -- there is no model of whether they will actually accept, and
    there cannot be, because no accepted-or-rejected trade has ever been recorded here.

    `parsimony` drops a candidate when a strict subset of its own moves is worth the
    same to within the selection-adjusted error -- see `prune_throw_ins`. It is on by
    default because the alternative is publishing a trade that costs an extra asset for
    a difference the simulation cannot measure. `parsimony=False` is the negative
    control and reproduces the previous behaviour exactly.
    """
    engine = finder or TradeFinder.from_sim(
        sim, rankings=rankings, rankings_weight=rankings_weight
    )
    team = for_team if for_team is not None else engine.my_team_id
    screened = engine.search(
        for_team=team,
        max_teams=max_teams,
        top_k=top_k,
        max_package=max_package,
        min_gain=min_gain,
    )
    if not screened:
        return []
    confirmed = engine.confirm_titles(screened if n_confirm is None else screened[:n_confirm])
    out = engine.recommend(confirmed, for_team=team, parsimony=parsimony)
    if include_harmful:
        return out
    return [r for r in out if r.delta_title > 0.0]


__all__ = [
    "DEFAULT_MAX_PACKAGE",
    "DEFAULT_MAX_TEAMS",
    "DEFAULT_PER_LEG",
    "DEFAULT_TOP_K",
    "DEFAULT_WIRE_DEPTH",
    "MEASURED_SD_DIFF",
    "PLAYOFF_WEIGHT",
    "SELECTION_ALPHA",
    "PreferenceEdge",
    "TeamImpact",
    "TradeError",
    "TradeEvaluation",
    "TradeFinder",
    "TradeLeg",
    "TradeProposal",
    "WeekWeights",
    "best_free_agents",
    "find_trades",
    "playoff_weights",
    "positional_requirements",
    "select_non_overlapping",
    "selection_threshold",
    "simple_cycles",
    "single_edge_targets",
    "slot_floor_matrix",
    "surplus_multiplier",
    "wire_pool",
]
