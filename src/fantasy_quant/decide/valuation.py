"""What a player is worth in ONE league, and how far that differs from the market.

A player's value is a function of `(player, league)`. The same receiver in a 14-team
full-PPR league and a 12-team half-PPR one is a different asset twice over: the
scoring moves his points, and the league depth moves the thing he is being compared
against. Nothing in this module may compute, cache or return a league-independent
number -- there is a test whose only job is to catch that regression, because the
failure mode is silent (a global value looks perfectly reasonable in every league).

**Replacement level is solved, not assumed.** The demand for a position is

    N_q = T * (L_q + sum_f alpha_qf * F_f) + T * beta_q

with `T` teams, `L_q` dedicated starting slots, `F_f` slots of flex type `f`, and
`alpha_qf` the share of that flex slot won by position `q`. `alpha` is *endogenous*:
which position wins the flex at the margin depends on the projections, and the
projections are ranked against a baseline that depends on `alpha`. So it is iterated
to a fixed point rather than guessed.

The loop is genuinely circular only because it runs over the *rostered* pool. Each
team rosters `N_q / T` players at `q`; in a given week its flex goes to the best one
left over after the dedicated slots are filled. Hoard receivers and your flex is a
receiver more often, which raises `alpha_WR`, which raises `N_WR`. With no bench
hoarding at all (`beta = {}`) the surplus above the dedicated slots is exactly the
number of flex slots, every survivor starts, and *any* alpha reproduces itself -- a
degenerate fixed point with no information in it. `solve_flex_shares` detects that
saturation and falls back to measuring alpha over the whole projection universe,
which has a unique answer.

**That surplus test is necessary but not sufficient, and the difference is measured.**
`sum(beta) > 0` over the flex-eligible positions only guarantees that *some* surplus
exists; it says nothing about whether the surplus is big enough for the pool to
choose. As `beta` shrinks the map degrades continuously into the identity, and an
iteration that starts at the caller's guess then returns it unchanged reports
`converged=True` while carrying no information at all. Measured on the 2026 corpus,
12-team half PPR: at `sum(beta) = 0.1` the solver hands back `alpha_RB = 1.000` from
an all-RB guess and `0.000` from an all-WR one. So the fixed point is not trusted on
the strength of convergence -- it is re-solved from every one-hot starting guess and
the spread across them is measured. Inside `FLEX_IDENTIFICATION_TOLERANCE` the answer
is reported with the spread attached; outside it, alpha is not identified by the
rostered pool and the open-pool measurement is used instead. `FlexSolution.identified`
and `.guess_spread` carry the verdict; the tests pin both branches.

Three things measured here that the research notes get wrong, reality winning:

* **The bench-hoarding priors are internally inconsistent.** RB 0.5 / WR 0.7 /
  TE 0.2 / QB 0.4 sums to 1.8 bench players per team. All three of the user's
  leagues carry **seven** bench slots, and their real week-1 rosters average
  RB 4.6-4.8 and WR 5.7-5.9 per team -- a bench of 7.07 to 7.25, four times the
  prior. `bench_hoarding_from_rosters` measures it from the league instead;
  `DEFAULT_BENCH_HOARDING` is kept only as the documented starting guess, and
  `sum(beta) == bench slots per team` is the sanity check `value_league` now runs
  and logs.

  For a long time this paragraph was the whole of the fix: the estimator existed,
  said so here, and **had no caller anywhere**. Passing `rosters=` to `value_league`
  is what closes it. The prior does not merely shift every value down, it reorders
  players *across* positions, because it understates the bench most at the positions
  a bench is actually made of. Measured on the three live leagues at week 1 of 2026
  the replacement level moves RB 8.26 -> 3.55 and WR 7.95 -> 5.15 points a week while
  K moves 8.75 -> 8.63 and D/ST 6.47 -> 6.31, so a kicker gains almost nothing and a
  running back gains 4.7 points a week of VORP. **543 to 552 of the ~598 valued
  players change rank**, all 16-17 of the user's own among them: Harrison Butker
  falls 120 -> 188 and the Chargers D/ST 148 -> 214, while Jordan Mason rises
  156 -> 76 and Tank Bigsby 288 -> 177.
* **`draftRanksByRankType` is not in the snapshot corpus.** `snapshot._flatten`
  captures `ownership` (percent owned/started, ADP, auction value) but never the
  draft-rank block, so the market screen runs on ADP, percent rostered and auction
  value. `MarketQuote.draft_rank` is wired through for when the capture is widened.
* **Weekly projections in the corpus do not zero out byes.** Every 2026 player has
  eighteen weekly projection rows, bye included -- and for a D/ST the row carries the
  full projection rather than a token 0.06, which is how every defence used to play
  seventeen games. `pipeline.build` now passes ESPN's bye table to `panel_for`, so
  `has_game` is the authority; this module reads `WeeklyOutlook.playing` and trusts it.

Everything is per week rather than from a season total, deliberately: a player who
misses six weeks and is elite in eleven is not the same asset as a mediocre one who
plays seventeen, and a season-total baseline cannot tell them apart. The playoff
number is the same computation restricted to `LeagueContext.playoff_weeks`, which is
the only one that pays.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..core import DST, QB, RB, TE, WR, K, LeagueContext, PlayerOutlook

if TYPE_CHECKING:  # pragma: no cover - only for the roster-recalibration helper
    from ..espn.league import TeamRoster

log = logging.getLogger(__name__)

#: Research's starting guess for bench hoarding, in players per team beyond the
#: starting requirement. Kept because the fixed point has to start somewhere, and
#: because reproducing the published QB12/RB30/WR42/TE12 figures needs the same
#: inputs they used. It sums to 1.8, which is not a real bench -- see the module
#: docstring, and prefer `bench_hoarding_from_rosters` on a league we can read.
DEFAULT_BENCH_HOARDING: Mapping[int, float] = MappingProxyType(
    {QB: 0.4, RB: 0.5, WR: 0.7, TE: 0.2, K: 0.0, DST: 0.0}
)

#: Ranks used when fitting `pts(rank) = A * exp(-b * rank)`. Research fitted the top
#: 60; the coefficient is sensitive to the window, so it travels with the result.
DEFAULT_SCARCITY_RANKS = 60

#: How far the solved flex shares may move with the starting guess before the fixed
#: point is declared unidentified. Measured, not picked: on the 2026 corpus the healthy
#: configurations spread by 0.0000 (14-team full PPR and both leagues at a measured
#: seven-man bench) to 0.0098 (12-team half PPR at the research prior), while the
#: degenerate ones spread by 0.034 (`sum(beta)=2`), 0.10 (`=1`) and 1.000 (`<=0.1`).
#: 0.02 sits in that gap with a factor of two of headroom on the live side.
FLEX_IDENTIFICATION_TOLERANCE = 0.02

#: Market signals the screen can rank against, and whether a *smaller* number means
#: the field likes the player more.
MARKET_METRICS: Mapping[str, bool] = MappingProxyType(
    {"adp": True, "draft_rank": True, "percent_owned": False, "auction_value": False}
)

#: `(min, max)` outside which a market signal carries no ordering information.
#: Measured on the 2026 capture, 1,036 players: `averageDraftPosition` is **censored**.
#: 845 players sit in [169.0, 171.6] -- ESPN parks everyone undrafted at the last pick
#: of its default draft (191 players fall below 169.0; a 12-team 16-round draft is 192
#: picks) and the spread inside that band is noise. Ranking against it unfiltered
#: makes the screen report nothing but that noise: every fifth-string quarterback
#: comes back as a 300-rank disagreement. `percentOwned` has the same problem at the
#: other end, with 603 players under 0.1%.
MARKET_INFORMATIVE_RANGE: Mapping[str, tuple[float | None, float | None]] = MappingProxyType(
    {
        "adp": (None, 169.0),
        "draft_rank": (None, None),
        "percent_owned": (0.1, None),
        "auction_value": (0.1, None),
    }
)


class ValuationError(ValueError):
    """The league shape or the projection set cannot support a valuation."""


# --------------------------------------------------------------------------------------
# Demand and replacement level
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionDemand:
    """How many players at one position the league as a whole wants rostered.

    The three components are kept apart rather than summed on the way in, because
    they answer different questions: `dedicated_slots` is fixed by the rules,
    `flex_slots` is the solved equilibrium share, and `bench_hoarding` is manager
    behaviour we can measure. Only `rostered` is the baseline.
    """

    position_id: int
    teams: int
    #: L_q -- starting slots that accept this position and nothing else.
    dedicated_slots: float
    #: sum_f alpha_qf * F_f at the solved alpha.
    flex_slots: float
    #: beta_q -- players carried beyond the starting requirement.
    bench_hoarding: float

    @property
    def starters_per_team(self) -> float:
        return self.dedicated_slots + self.flex_slots

    @property
    def per_team(self) -> float:
        return self.starters_per_team + self.bench_hoarding

    @property
    def starters(self) -> float:
        """T * (L_q + sum_f alpha_qf F_f) -- the value-over-last-starter baseline."""
        return self.teams * self.starters_per_team

    @property
    def rostered(self) -> float:
        """N_q. The baseline rank: rank N_q at this position is the wire."""
        return self.teams * self.per_team


@dataclass(frozen=True, slots=True)
class ReplacementLevel:
    """The points a freely available player at one position is worth, week by week.

    Weekly rather than a single season number because the replacement moves: byes
    thin the position, and the N_q-th best receiver in week 9 is not the N_q-th best
    in week 3. A season-average baseline quietly overvalues players whose good weeks
    land when everyone else's are good too.
    """

    position_id: int
    demand: PositionDemand
    by_week: Mapping[int, float]
    #: True when the projection pool is shallower than the demand, so the baseline
    #: is pinned to the worst projected player rather than genuinely resolved.
    supply_limited: bool = False

    @property
    def baseline_rank(self) -> float:
        return self.demand.rostered

    @property
    def per_week(self) -> float:
        return sum(self.by_week.values()) / len(self.by_week) if self.by_week else 0.0

    def points(self, week: int) -> float:
        return self.by_week.get(week, 0.0)


@dataclass(frozen=True, slots=True)
class FlexSolution:
    """The solved flex-demand fixed point.

    `shares[slot][position]` is the fraction of that flex slot won by that position;
    each slot's shares sum to 1.

    `converged` only says the iteration stopped moving, which a degenerate map does
    immediately. `identified` is the number that matters: it says the answer was
    re-derived from every one-hot starting guess and came back the same, and
    `guess_spread` is how far apart those re-derivations actually landed.
    `saturated` records that alpha was measured over the open pool rather than the
    rostered one -- either because the rostered pool had no surplus over the flex
    slots, or because it had too little surplus to identify anything.
    """

    shares: Mapping[int, Mapping[int, float]]
    iterations: int
    converged: bool
    max_change: float
    saturated: bool
    #: False when the rostered-pool answer moved with the starting guess by more than
    #: `FLEX_IDENTIFICATION_TOLERANCE`, in which case `shares` is the open-pool answer.
    identified: bool = True
    #: Largest disagreement between starting guesses, before any fallback.
    guess_spread: float = 0.0
    trace: tuple[Mapping[int, Mapping[int, float]], ...] = field(default=(), repr=False)

    def share(self, slot_id: int, position_id: int) -> float:
        return self.shares.get(slot_id, {}).get(position_id, 0.0)


@dataclass(frozen=True, slots=True)
class ReplacementModel:
    """Solved demand and replacement level for one league, one horizon."""

    league_id: int
    season: int
    teams: int
    from_week: int
    weeks: tuple[int, ...]
    flex: FlexSolution
    levels: Mapping[int, ReplacementLevel]

    def level(self, position_id: int) -> ReplacementLevel | None:
        return self.levels.get(position_id)

    def demand(self, position_id: int) -> PositionDemand | None:
        level = self.levels.get(position_id)
        return level.demand if level is not None else None

    def baseline_ranks(self) -> dict[int, float]:
        """position -> N_q. The headline of the whole fixed point."""
        return {p: lv.demand.rostered for p, lv in sorted(self.levels.items())}

    def describe(self, names: Mapping[int, str] | None = None) -> str:
        names = names or POSITION_ABBREV
        parts = []
        for pos, lv in sorted(self.levels.items()):
            tag = "*" if lv.supply_limited else ""
            parts.append(f"{names.get(pos, pos)}{lv.demand.rostered:.1f}{tag}@{lv.per_week:.2f}")
        flex = ", ".join(
            f"slot {slot}: "
            + "/".join(
                f"{names.get(p, p)} {s:.0%}" for p, s in sorted(sh.items(), key=lambda kv: -kv[1])
            )
            for slot, sh in sorted(self.flex.shares.items())
        )
        head = (
            f"league {self.league_id}/{self.season} {self.teams} teams, weeks "
            f"{self.from_week}-{self.weeks[-1] if self.weeks else self.from_week}: "
            + " ".join(parts)
        )
        if not flex:
            return head
        # A caller reading this needs to know when the shares are the open-pool
        # fallback rather than a solved equilibrium; it changes what they mean.
        note = (
            ""
            if self.flex.identified
            else f" [open pool: alpha moved {self.flex.guess_spread:.3f} with the guess]"
        )
        return f"{head}\n  flex {flex}{note}"


POSITION_ABBREV: Mapping[int, str] = MappingProxyType(
    {QB: "QB", RB: "RB", WR: "WR", TE: "TE", K: "K", DST: "DST"}
)


# --------------------------------------------------------------------------------------
# Player value
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerValue:
    """One player's value in one league, over two horizons.

    `ros_vorp` sums `mu_iw - mu_replacement,q,w` over every remaining week, so a bye
    or an absence costs a replacement's worth of points rather than nothing. That is
    the point: the roster spot is idle that week and the alternative was startable.
    """

    player_id: int
    name: str
    position_id: int
    ros_points: float
    ros_vorp: float
    ros_weeks: int
    playoff_points: float
    playoff_vorp: float
    playoff_weeks: int

    @property
    def ros_vorp_per_week(self) -> float:
        return self.ros_vorp / self.ros_weeks if self.ros_weeks else 0.0

    @property
    def playoff_vorp_per_week(self) -> float:
        return self.playoff_vorp / self.playoff_weeks if self.playoff_weeks else 0.0


@dataclass(frozen=True, slots=True)
class ScarcityCurve:
    """`pts(rank) = a * exp(-b * rank)`, fitted per position over the top ranks.

    `half_life` -- ranks it takes for the position to lose half its production -- is
    the number that matters. A flat curve means the position can be deferred; a steep
    one means the top of it is the whole supply. Research measured QB 0.040 / RB 0.026
    / WR 0.013 / TE 0.039; the fit is sensitive to `n`, so `n` travels with it.
    """

    position_id: int
    a: float
    b: float
    #: R^2 in log space, which is where the line is actually fitted.
    r2: float
    n: int

    @property
    def half_life(self) -> float:
        return math.log(2.0) / self.b if self.b > 0 else math.inf

    def predict(self, rank: float) -> float:
        return self.a * math.exp(-self.b * rank)


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """What the field thinks of a player, from the daily ESPN capture.

    `draft_rank` is `draftRanksByRankType` and is currently always None: the
    snapshot writer does not capture that block. Wired through so widening the
    capture needs no change here.
    """

    player_id: int
    adp: float | None = None
    percent_owned: float | None = None
    auction_value: float | None = None
    draft_rank: float | None = None

    def metric(self, name: str) -> float | None:
        if name not in MARKET_METRICS:
            raise ValueError(f"unknown market metric {name!r}; have {sorted(MARKET_METRICS)}")
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class MarketEdge:
    """One player where our valuation and the market disagree, in ranks.

    `rank_delta = market_rank - our_rank`, so a *positive* delta means we rank him
    higher than the field does. Ranks rather than raw values because ADP, percent
    rostered and auction dollars are not in the same unit as points and never will be.
    """

    player_id: int
    name: str
    position_id: int
    our_rank: int
    market_rank: float
    market_metric: str
    market_value: float
    rank_delta: float
    ros_vorp: float
    playoff_vorp: float

    @property
    def is_buy(self) -> bool:
        return self.rank_delta > 0


@dataclass(frozen=True, slots=True)
class ValuationReport:
    """Everything the valuation layer knows about one league at one moment."""

    league_id: int
    season: int
    name: str
    teams: int
    from_week: int
    replacement: ReplacementModel
    values: tuple[PlayerValue, ...]
    curves: Mapping[int, ScarcityCurve]
    positive_vorp_share: Mapping[int, float]
    market: tuple[MarketEdge, ...]

    def value_for(self, player_id: int) -> PlayerValue | None:
        for v in self.values:
            if v.player_id == player_id:
                return v
        return None

    def top(self, n: int = 20, position_id: int | None = None) -> tuple[PlayerValue, ...]:
        pool = [v for v in self.values if position_id is None or v.position_id == position_id]
        return tuple(pool[:n])

    def buys(self, n: int = 15) -> tuple[MarketEdge, ...]:
        return tuple([e for e in self.market if e.is_buy][:n])

    def fades(self, n: int = 15) -> tuple[MarketEdge, ...]:
        return tuple([e for e in self.market if not e.is_buy][:n])


# --------------------------------------------------------------------------------------
# League shape
# --------------------------------------------------------------------------------------


def starting_shape(
    ctx: LeagueContext, positions: Iterable[int]
) -> tuple[dict[int, float], dict[int, tuple[float, frozenset[int]]]]:
    """`(dedicated slots per position, flex slot -> (count, eligible positions))`.

    Derived entirely from `LeagueContext.starting_slots` and `slot_eligibility`, so a
    superflex, a WR/TE flex or an IDP block all fall out without a special case. A
    slot whose eligibility narrows to exactly one position we have players for is
    dedicated; anything wider is a flex whose share has to be solved.
    """
    known = frozenset(positions)
    dedicated: dict[int, float] = {}
    flex: dict[int, tuple[float, frozenset[int]]] = {}
    for slot, count in sorted(ctx.starting_slots.items()):
        raw = ctx.slot_eligibility.get(slot)
        if raw is None:
            log.warning(
                "league %s starts %d of lineup slot %d but the context has no eligibility "
                "for it; that demand is being dropped.",
                ctx.league_id,
                count,
                slot,
            )
            continue
        eligible = frozenset(raw) & known
        if not eligible:
            log.debug("slot %d accepts no position we project; skipping", slot)
            continue
        if len(eligible) == 1:
            (only,) = eligible
            dedicated[only] = dedicated.get(only, 0.0) + float(count)
        else:
            flex[slot] = (float(count), eligible)
    return dedicated, flex


def uniform_shares(
    flex: Mapping[int, tuple[float, frozenset[int]]],
) -> dict[int, dict[int, float]]:
    """The neutral starting guess: every eligible position equally likely."""
    return {
        slot: {pos: 1.0 / len(eligible) for pos in sorted(eligible)}
        for slot, (_, eligible) in flex.items()
    }


def position_demand(
    ctx: LeagueContext,
    shares: Mapping[int, Mapping[int, float]],
    *,
    positions: Iterable[int],
    bench_hoarding: Mapping[int, float] | None = None,
) -> dict[int, PositionDemand]:
    """N_q = T * (L_q + sum_f alpha_qf F_f) + T * beta_q, for every position."""
    dedicated, flex = starting_shape(ctx, positions)
    bench = dict(DEFAULT_BENCH_HOARDING if bench_hoarding is None else bench_hoarding)
    out: dict[int, PositionDemand] = {}
    for pos in sorted(set(dedicated) | {p for _, (_, e) in flex.items() for p in e}):
        from_flex = sum(
            count * float(shares.get(slot, {}).get(pos, 0.0)) for slot, (count, _) in flex.items()
        )
        out[pos] = PositionDemand(
            position_id=pos,
            teams=ctx.size,
            dedicated_slots=dedicated.get(pos, 0.0),
            flex_slots=from_flex,
            bench_hoarding=float(bench.get(pos, 0.0)),
        )
    return out


# --------------------------------------------------------------------------------------
# The projection universe
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Universe:
    """Projections reshaped for ranking: by position, and by week."""

    positions: frozenset[int]
    #: position -> player ids, best first over the whole horizon.
    ranked: Mapping[int, tuple[int, ...]]
    #: week -> position -> (player id, mu) sorted best first, playable players only.
    weekly: Mapping[int, Mapping[int, tuple[tuple[int, float], ...]]]


def _build_universe(outlooks: Sequence[PlayerOutlook], weeks: Sequence[int]) -> _Universe:
    horizon = set(weeks)
    position_of: dict[int, int] = {}
    totals: dict[int, float] = {}
    per_week: dict[int, dict[int, list[tuple[int, float]]]] = {w: {} for w in weeks}

    for o in outlooks:
        if o.player_id in position_of:
            # A duplicate would enter every weekly ranking twice, pushing the
            # baseline one rank shallower for free. Merging two projection sources
            # without deduplicating is exactly how that happens, so it is loud.
            raise ValuationError(
                f"player {o.player_id} ({o.name}) appears twice in the projection set"
            )
        position_of[o.player_id] = o.position_id
        total = 0.0
        for week, wo in o.weeks.items():
            if week not in horizon:
                continue
            total += wo.mean if wo.playing else 0.0
            if not wo.playing:
                continue
            per_week[week].setdefault(o.position_id, []).append((o.player_id, wo.mean))
        totals[o.player_id] = total

    ranked: dict[int, list[int]] = {}
    for pid, pos in position_of.items():
        ranked.setdefault(pos, []).append(pid)
    for pids in ranked.values():
        # Ties broken on player id so two runs of the same inputs agree exactly.
        pids.sort(key=lambda p: (-totals[p], p))

    weekly = {
        w: {pos: tuple(sorted(rows, key=lambda r: (-r[1], r[0]))) for pos, rows in by_pos.items()}
        for w, by_pos in per_week.items()
    }
    return _Universe(
        positions=frozenset(ranked),
        ranked=MappingProxyType({p: tuple(v) for p, v in ranked.items()}),
        weekly=MappingProxyType(weekly),
    )


def _interpolate_rank(values: Sequence[float], rank: float) -> tuple[float, bool]:
    """Value at a fractional 1-indexed rank. Returns `(value, supply_limited)`.

    N_q is rarely an integer -- a 41.4-deep receiver baseline is a real answer -- and
    rounding it introduces a step change in every player's value at the boundary.

    `supply_limited` is `rank > n`, not `rank >= n`: at exactly `n` the baseline is
    the last projected player, which is a resolved answer, not a pinned one.
    """
    n = len(values)
    if n == 0:
        return 0.0, True
    if rank <= 1.0:
        return values[0], False
    if rank >= n:
        return values[-1], rank > n
    lo = int(math.floor(rank))
    frac = rank - lo
    return values[lo - 1] + (values[lo] - values[lo - 1]) * frac, False


# --------------------------------------------------------------------------------------
# The flex-demand fixed point
# --------------------------------------------------------------------------------------


def _rostered_pool(universe: _Universe, demand: Mapping[int, PositionDemand]) -> set[int]:
    """The top N_q players at each position -- who is off the wire under this alpha."""
    pool: set[int] = set()
    for pos, dem in demand.items():
        take = max(1, int(round(dem.rostered)))
        pool.update(universe.ranked.get(pos, ())[:take])
    return pool


def _measure_shares(
    universe: _Universe,
    ctx: LeagueContext,
    dedicated: Mapping[int, float],
    flex: Mapping[int, tuple[float, frozenset[int]]],
    pool: set[int] | None,
    weeks: Sequence[int],
) -> dict[int, dict[int, float]]:
    """Simulate every week's lineups and count who actually fills each flex slot.

    Greedy, and exact for the nested eligibility ESPN uses (dedicated subset FLEX
    subset SUPERFLEX): fill the dedicated slots best-first, then the flex slots in
    order of how restrictive they are. Aggregated across the league rather than
    per-team, which assumes rosters are efficiently distributed -- true enough at
    equilibrium, and the alternative is simulating a draft inside a fixed point.
    """
    counts: dict[int, dict[int, float]] = {slot: {} for slot in flex}
    order = sorted(flex, key=lambda s: (len(flex[s][1]), s))

    for week in weeks:
        by_pos = universe.weekly.get(week, {})
        remaining: dict[int, list[tuple[int, float]]] = {}
        for pos, rows in by_pos.items():
            keep = [r for r in rows if pool is None or r[0] in pool]
            need = int(round(ctx.size * dedicated.get(pos, 0.0)))
            remaining[pos] = keep[need:]

        for slot in order:
            count, eligible = flex[slot]
            seats = int(round(ctx.size * count))
            if seats <= 0:
                continue
            field_: list[tuple[float, int, int]] = []
            for pos in eligible:
                field_.extend((-mu, pid, pos) for pid, mu in remaining.get(pos, ()))
            field_.sort()
            taken = field_[:seats]
            claimed: set[int] = set()
            for _, pid, pos in taken:
                counts[slot][pos] = counts[slot].get(pos, 0.0) + 1.0
                claimed.add(pid)
            for pos in eligible:
                remaining[pos] = [r for r in remaining.get(pos, ()) if r[0] not in claimed]

    out: dict[int, dict[int, float]] = {}
    for slot, (_, eligible) in flex.items():
        total = sum(counts[slot].values())
        if total <= 0:
            # No candidate ever reached this slot -- the dedicated slots ate the whole
            # pool. Stay neutral rather than invent a winner; with damping < 1 the
            # caller's previous estimate survives, and with damping = 1 the uniform
            # split is the only honest answer available.
            out[slot] = {pos: 1.0 / len(eligible) for pos in sorted(eligible)}
        else:
            out[slot] = {pos: counts[slot].get(pos, 0.0) / total for pos in sorted(eligible)}
    return out


def _one_hot_guesses(
    flex: Mapping[int, tuple[float, frozenset[int]]],
) -> list[dict[int, dict[int, float]]]:
    """One starting guess per flex-eligible position, each as extreme as it gets.

    These are the probes the identification check re-solves from. A one-hot is the
    right probe because the degenerate map is the identity: if the rostered pool
    cannot select, the solver hands a one-hot straight back, and no other starting
    guess makes that as obvious.
    """
    positions = sorted({p for _, eligible in flex.values() for p in eligible})
    neutral = uniform_shares(flex)
    out: list[dict[int, dict[int, float]]] = []
    for pos in positions:
        guess: dict[int, dict[int, float]] = {}
        for slot, (_, eligible) in flex.items():
            if pos in eligible:
                guess[slot] = {p: (1.0 if p == pos else 0.0) for p in sorted(eligible)}
            else:
                guess[slot] = dict(neutral[slot])
        out.append(guess)
    return out


def _max_share_change(
    a: Mapping[int, Mapping[int, float]], b: Mapping[int, Mapping[int, float]]
) -> float:
    worst = 0.0
    for slot in set(a) | set(b):
        left, right = a.get(slot, {}), b.get(slot, {})
        for pos in set(left) | set(right):
            worst = max(worst, abs(left.get(pos, 0.0) - right.get(pos, 0.0)))
    return worst


@dataclass(frozen=True, slots=True)
class _Iteration:
    """One run of the iteration, before anything has been said about identification."""

    shares: dict[int, dict[int, float]]
    iterations: int
    change: float
    saturated: bool
    trace: tuple[Mapping[int, Mapping[int, float]], ...]


def _iterate_shares(
    ctx: LeagueContext,
    universe: _Universe,
    dedicated: Mapping[int, float],
    flex: Mapping[int, tuple[float, frozenset[int]]],
    weeks: Sequence[int],
    *,
    bench_hoarding: Mapping[int, float] | None,
    initial_shares: Mapping[int, Mapping[int, float]] | None,
    max_iterations: int,
    tolerance: float,
    damping: float,
    force_open_pool: bool = False,
) -> _Iteration:
    """The iteration itself: guess -> baselines -> weekly lineups -> guess.

    Says nothing about whether the answer means anything; `solve_flex_shares` decides
    that by running this from several starting guesses and comparing.
    """
    shares: dict[int, dict[int, float]] = (
        {slot: dict(initial_shares[slot]) for slot in flex if slot in initial_shares}
        if initial_shares
        else {}
    )
    for slot, guess in uniform_shares(flex).items():
        shares.setdefault(slot, guess)

    flex_positions = {p for _, eligible in flex.values() for p in eligible}
    seats = sum(ctx.size * count for count, _ in flex.values())
    trace: list[Mapping[int, Mapping[int, float]]] = []
    change = math.inf
    saturated = force_open_pool
    iterations = 0

    for step in range(1, max_iterations + 1):
        iterations = step
        demand = position_demand(
            ctx, shares, positions=universe.positions, bench_hoarding=bench_hoarding
        )
        pool: set[int] | None = None if force_open_pool else _rostered_pool(universe, demand)
        # Surplus over the dedicated slots, measured in seats. If it does not exceed
        # the flex seats then every surviving player starts and the measurement can
        # only echo the guess back -- see the module docstring. Necessary, not
        # sufficient: `solve_flex_shares` checks the sufficient condition by measuring.
        surplus = sum(
            max(0.0, demand[p].rostered - ctx.size * demand[p].dedicated_slots)
            for p in flex_positions
            if p in demand
        )
        if surplus <= seats + 1e-9:
            saturated = True
            pool = None

        measured = _measure_shares(universe, ctx, dedicated, flex, pool, weeks)
        updated = {
            slot: {
                pos: (1.0 - damping) * shares[slot].get(pos, 0.0) + damping * value
                for pos, value in slot_shares.items()
            }
            for slot, slot_shares in measured.items()
        }
        change = _max_share_change(shares, updated)
        shares = updated
        trace.append(MappingProxyType({s: MappingProxyType(dict(v)) for s, v in shares.items()}))
        if change < tolerance:
            break

    if change >= tolerance:
        if damping > 0.5:
            log.warning(
                "flex shares for league %s still moving %.4f after %d undamped passes; "
                "retrying damped.",
                ctx.league_id,
                change,
                max_iterations,
            )
            return _iterate_shares(
                ctx,
                universe,
                dedicated,
                flex,
                weeks,
                bench_hoarding=bench_hoarding,
                initial_shares=shares,
                max_iterations=max_iterations,
                tolerance=tolerance,
                damping=0.5,
                force_open_pool=force_open_pool,
            )
        log.warning(
            "flex shares for league %s did not converge in %d passes (max change %.4f)",
            ctx.league_id,
            max_iterations,
            change,
        )
    return _Iteration(
        shares=shares,
        iterations=iterations,
        change=change,
        saturated=saturated,
        trace=tuple(trace),
    )


def solve_flex_shares(
    ctx: LeagueContext,
    outlooks: Sequence[PlayerOutlook],
    weeks: Sequence[int],
    *,
    bench_hoarding: Mapping[int, float] | None = None,
    initial_shares: Mapping[int, Mapping[int, float]] | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-3,
    damping: float = 1.0,
    identification_tolerance: float = FLEX_IDENTIFICATION_TOLERANCE,
    universe: _Universe | None = None,
) -> FlexSolution:
    """Solve alpha, then prove the answer is not just the starting guess.

    Undamped by default because it measures out at three to six passes over every
    roster shape we have tried (2WR/3WR x FLEX/WR-TE flex x superflex). The feedback
    is positive, though -- a position that wins more flex gets rostered deeper, which
    puts more of it in the pool to win the flex again -- so a two-cycle is possible
    in principle, and a run that fails to converge is retried once at `damping=0.5`
    rather than returning a value that is still moving.

    Convergence is not the acceptance test. The map degrades continuously into the
    identity as bench hoarding shrinks, and the identity converges on pass one at
    whatever it was handed. So the solve is repeated from a one-hot guess on every
    flex-eligible position and the spread across those answers is measured. Within
    `identification_tolerance` the fixed point is real and is returned with the spread
    attached; outside it the rostered pool cannot identify alpha and the guess-free
    open-pool measurement is returned instead, with `identified=False`.
    """
    if not 0.0 < damping <= 1.0:
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    universe = universe or _build_universe(outlooks, weeks)
    dedicated, flex = starting_shape(ctx, universe.positions)
    if not flex:
        return FlexSolution(
            shares={}, iterations=0, converged=True, max_change=0.0, saturated=False
        )

    def run(guess: Mapping[int, Mapping[int, float]] | None, *, open_pool: bool = False):
        return _iterate_shares(
            ctx,
            universe,
            dedicated,
            flex,
            weeks,
            bench_hoarding=bench_hoarding,
            initial_shares=guess,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            force_open_pool=open_pool,
        )

    primary = run(initial_shares)

    # The open-pool branch never reads `shares`, so it is guess-free by construction
    # and there is nothing to probe.
    spread = 0.0
    if not primary.saturated:
        for probe in _one_hot_guesses(flex):
            spread = max(spread, _max_share_change(primary.shares, run(probe).shares))

    if spread > identification_tolerance:
        log.warning(
            "flex shares for league %s move %.4f with the starting guess (tolerance %.4f); "
            "the rostered pool cannot identify alpha at this bench hoarding. Falling back "
            "to the open-pool measurement.",
            ctx.league_id,
            spread,
            identification_tolerance,
        )
        fallback = run(None, open_pool=True)
        return FlexSolution(
            shares=MappingProxyType(
                {s: MappingProxyType(dict(v)) for s, v in fallback.shares.items()}
            ),
            iterations=primary.iterations + fallback.iterations,
            converged=fallback.change < tolerance,
            max_change=fallback.change,
            saturated=True,
            identified=False,
            guess_spread=spread,
            trace=fallback.trace,
        )

    return FlexSolution(
        shares=MappingProxyType({s: MappingProxyType(dict(v)) for s, v in primary.shares.items()}),
        iterations=primary.iterations,
        converged=primary.change < tolerance,
        max_change=primary.change,
        saturated=primary.saturated,
        identified=True,
        guess_spread=spread,
        trace=primary.trace,
    )


def replacement_levels(
    ctx: LeagueContext,
    outlooks: Sequence[PlayerOutlook],
    weeks: Sequence[int],
    demand: Mapping[int, PositionDemand],
    *,
    universe: _Universe | None = None,
) -> dict[int, ReplacementLevel]:
    """The N_q-th best projected player at each position, in each week."""
    universe = universe or _build_universe(outlooks, weeks)
    out: dict[int, ReplacementLevel] = {}
    for pos, dem in demand.items():
        by_week: dict[int, float] = {}
        limited = False
        for week in weeks:
            rows = universe.weekly.get(week, {}).get(pos, ())
            value, short = _interpolate_rank([mu for _, mu in rows], dem.rostered)
            by_week[week] = value
            limited = limited or short
        if limited:
            log.info(
                "position %s in league %s: baseline rank %.1f exceeds the projected pool in at "
                "least one week; replacement pinned to the worst projection.",
                POSITION_ABBREV.get(pos, pos),
                ctx.league_id,
                dem.rostered,
            )
        out[pos] = ReplacementLevel(
            position_id=pos,
            demand=dem,
            by_week=MappingProxyType(by_week),
            supply_limited=limited,
        )
    return out


def build_replacement_model(
    ctx: LeagueContext,
    outlooks: Sequence[PlayerOutlook],
    *,
    from_week: int = 1,
    bench_hoarding: Mapping[int, float] | None = None,
    initial_shares: Mapping[int, Mapping[int, float]] | None = None,
    damping: float = 1.0,
    max_iterations: int = 30,
    tolerance: float = 1e-3,
) -> ReplacementModel:
    """Solve alpha, then N_q, then the weekly replacement level. The whole engine."""
    weeks = remaining_weeks(ctx, from_week)
    if not weeks:
        raise ValuationError(
            f"league {ctx.league_id} has no weeks at or after {from_week}; "
            f"regular season {ctx.regular_season_weeks}, playoffs {ctx.playoff_weeks}"
        )
    if not outlooks:
        raise ValuationError("no projections supplied; replacement level is undefined")

    universe = _build_universe(outlooks, weeks)
    flex = solve_flex_shares(
        ctx,
        outlooks,
        weeks,
        bench_hoarding=bench_hoarding,
        initial_shares=initial_shares,
        damping=damping,
        max_iterations=max_iterations,
        tolerance=tolerance,
        universe=universe,
    )
    demand = position_demand(
        ctx, flex.shares, positions=universe.positions, bench_hoarding=bench_hoarding
    )
    levels = replacement_levels(ctx, outlooks, weeks, demand, universe=universe)
    return ReplacementModel(
        league_id=ctx.league_id,
        season=ctx.season,
        teams=ctx.size,
        from_week=from_week,
        weeks=tuple(weeks),
        flex=flex,
        levels=MappingProxyType(levels),
    )


def remaining_weeks(ctx: LeagueContext, from_week: int) -> tuple[int, ...]:
    """Every scoring period still to be played, regular season and playoffs."""
    return tuple(
        sorted({w for w in (*ctx.regular_season_weeks, *ctx.playoff_weeks) if w >= from_week})
    )


# --------------------------------------------------------------------------------------
# VORP
# --------------------------------------------------------------------------------------


def player_values(
    outlooks: Sequence[PlayerOutlook],
    model: ReplacementModel,
    *,
    playoff_weeks: Sequence[int],
) -> tuple[PlayerValue, ...]:
    """Per-week VORP over the remaining season, and again over the playoff weeks.

    A week the projection layer omits entirely is unknown and is skipped. A week it
    marks `playing=False` is charged the replacement's points, because that is what
    the roster spot costs you: it sits idle while a wire pickup would have scored.
    """
    playoffs = set(playoff_weeks)
    out: list[PlayerValue] = []
    for o in outlooks:
        level = model.levels.get(o.position_id)
        if level is None:
            continue
        ros_pts = ros_vorp = po_pts = po_vorp = 0.0
        ros_n = po_n = 0
        for week in model.weeks:
            wo = o.weeks.get(week)
            if wo is None:
                continue
            mu = wo.mean if wo.playing else 0.0
            edge = mu - level.points(week)
            ros_pts += mu
            ros_vorp += edge
            ros_n += 1
            if week in playoffs:
                po_pts += mu
                po_vorp += edge
                po_n += 1
        out.append(
            PlayerValue(
                player_id=o.player_id,
                name=o.name,
                position_id=o.position_id,
                ros_points=ros_pts,
                ros_vorp=ros_vorp,
                ros_weeks=ros_n,
                playoff_points=po_pts,
                playoff_vorp=po_vorp,
                playoff_weeks=po_n,
            )
        )
    out.sort(key=lambda v: (-v.ros_vorp, v.player_id))
    return tuple(out)


def positive_vorp_share(
    values: Sequence[PlayerValue], *, playoff: bool = False
) -> dict[int, float]:
    """Each position's share of the league's total positive VORP -- the scarcity budget.

    Only positive contributions count. Summing signed VORP would net a position's
    replaceable tail against its stars and report that the deep positions do not
    matter, which is exactly backwards.
    """
    totals: dict[int, float] = {}
    for v in values:
        vorp = v.playoff_vorp if playoff else v.ros_vorp
        if vorp > 0:
            totals[v.position_id] = totals.get(v.position_id, 0.0) + vorp
    grand = sum(totals.values())
    if grand <= 0:
        return {}
    return {pos: total / grand for pos, total in sorted(totals.items())}


# --------------------------------------------------------------------------------------
# Positional scarcity
# --------------------------------------------------------------------------------------


def fit_scarcity_curve(
    points: Sequence[float],
    position_id: int,
    *,
    ranks: int = DEFAULT_SCARCITY_RANKS,
    floor: float = 1e-6,
) -> ScarcityCurve:
    """Least squares on `log pts = log a - b * rank`, over the top `ranks` players.

    Log-linear rather than a nonlinear exponential fit: it is closed-form, it cannot
    fail to converge, and the quantity of interest -- the decay rate -- is a slope in
    that space. `points` must already be sorted best first.
    """
    ys = [p for p in points[:ranks] if p > floor]
    n = len(ys)
    if n < 3:
        raise ValuationError(
            f"need at least 3 positive projections to fit a scarcity curve for position "
            f"{position_id}, got {n}"
        )
    xs = list(range(1, n + 1))
    logs = [math.log(y) for y in ys]
    mx = sum(xs) / n
    my = sum(logs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, logs, strict=True))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, logs, strict=True))
    total = sum((y - my) ** 2 for y in logs)
    return ScarcityCurve(
        position_id=position_id,
        a=math.exp(intercept),
        b=-slope,
        r2=1.0 - resid / total if total > 0 else 1.0,
        n=n,
    )


def scarcity_curves(
    outlooks: Sequence[PlayerOutlook],
    weeks: Sequence[int],
    *,
    ranks: int = DEFAULT_SCARCITY_RANKS,
    per_game: bool = True,
) -> dict[int, ScarcityCurve]:
    """Fit every position's rank-vs-points curve over the same horizon.

    `per_game` divides by weeks played so a bye does not read as scarcity. The decay
    rate is scale-free either way; the intercept is not.
    """
    horizon = set(weeks)
    totals: dict[int, list[float]] = {}
    for o in outlooks:
        played = [wo.mean for w, wo in o.weeks.items() if w in horizon and wo.playing]
        if not played:
            continue
        total = sum(played)
        totals.setdefault(o.position_id, []).append(total / len(played) if per_game else total)
    out: dict[int, ScarcityCurve] = {}
    for pos, values in totals.items():
        values.sort(reverse=True)
        try:
            out[pos] = fit_scarcity_curve(values, pos, ranks=ranks)
        except ValuationError:
            log.debug("too few projections at position %s to fit a scarcity curve", pos)
    return out


# --------------------------------------------------------------------------------------
# Market arbitrage
# --------------------------------------------------------------------------------------


def market_disagreements(
    values: Sequence[PlayerValue],
    quotes: Mapping[int, MarketQuote] | Sequence[MarketQuote],
    *,
    metric: str = "adp",
    limit: int | None = None,
    playoff: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
    use_informative_range: bool = True,
    positions: Iterable[int] | None = None,
) -> tuple[MarketEdge, ...]:
    """Rank players by our VORP and by the market, and return the disagreements.

    Both ranks are computed over the *same* set of players -- only those carrying an
    informative quote -- so an ADP list that prices 190 of 596 players does not make
    everyone outside it look like a screaming buy. `rank_delta` is positive when we
    like a player more than the field.

    `min_value`/`max_value` bound the quotes that count, defaulting to the measured
    `MARKET_INFORMATIVE_RANGE` for the metric. Pass `use_informative_range=False` to
    screen the raw column, which for ADP means screening ESPN's censoring artifact.

    `positions` restricts both rankings to one position, which is usually what you
    want. An all-positions ADP screen structurally reports kickers and defenses as
    buys: ADP encodes the convention "take your kicker last", not a view on value,
    so a K whose VORP puts him 145th overall is drafted 190th in every league on
    earth. Passing `positions=(WR,)` asks the question that has an answer -- we have
    him as WR14 and the field has him as WR31.
    """
    if metric not in MARKET_METRICS:
        raise ValueError(f"unknown market metric {metric!r}; have {sorted(MARKET_METRICS)}")
    ascending = MARKET_METRICS[metric]
    lookup = quotes if isinstance(quotes, Mapping) else {q.player_id: q for q in quotes}
    if use_informative_range:
        default_lo, default_hi = MARKET_INFORMATIVE_RANGE.get(metric, (None, None))
        min_value = default_lo if min_value is None else min_value
        max_value = default_hi if max_value is None else max_value

    wanted = frozenset(positions) if positions is not None else None

    scored: list[tuple[PlayerValue, float]] = []
    for v in values:
        if wanted is not None and v.position_id not in wanted:
            continue
        quote = lookup.get(v.player_id)
        if quote is None:
            continue
        raw = quote.metric(metric)
        if raw is None:
            continue
        if (min_value is not None and raw < min_value) or (
            max_value is not None and raw >= max_value
        ):
            continue
        scored.append((v, float(raw)))
    if not scored:
        return ()

    vorp = (lambda v: v.playoff_vorp) if playoff else (lambda v: v.ros_vorp)
    ours = sorted(scored, key=lambda pair: (-vorp(pair[0]), pair[0].player_id))
    our_rank = {v.player_id: r for r, (v, _) in enumerate(ours, start=1)}

    market_sorted = sorted(
        scored, key=lambda pair: ((pair[1] if ascending else -pair[1]), pair[0].player_id)
    )
    market_rank = {v.player_id: r for r, (v, _) in enumerate(market_sorted, start=1)}

    edges = [
        MarketEdge(
            player_id=v.player_id,
            name=v.name,
            position_id=v.position_id,
            our_rank=our_rank[v.player_id],
            market_rank=market_rank[v.player_id],
            market_metric=metric,
            market_value=raw,
            rank_delta=market_rank[v.player_id] - our_rank[v.player_id],
            ros_vorp=v.ros_vorp,
            playoff_vorp=v.playoff_vorp,
        )
        for v, raw in scored
    ]
    edges.sort(key=lambda e: (-abs(e.rank_delta), e.our_rank))
    return tuple(edges[:limit] if limit else edges)


# --------------------------------------------------------------------------------------
# Bench hoarding, measured
# --------------------------------------------------------------------------------------


def rostered_per_team(
    rosters: Mapping[int, TeamRoster], *, include_ir: bool = True
) -> dict[int, float]:
    """Average players carried at each position, from the league's real rosters.

    IR counts by default: an IR'd player is off the wire, and replacement level is
    about who is actually available, not who is startable.
    """
    # Imported here rather than at module scope so the valuation layer stays free of
    # the ESPN package unless a caller actually hands it ESPN rosters; core.py does
    # the same with NON_STARTING_SLOTS. The slot id lives in one place either way --
    # a second copy of `21` here is exactly how the two id spaces get crossed.
    from ..espn.scoring import SLOT_IR

    counts: dict[int, float] = {}
    teams = len(rosters)
    if not teams:
        return counts
    for roster in rosters.values():
        for entry in roster.entries:
            if not include_ir and entry.lineup_slot_id == SLOT_IR:
                continue
            pos = entry.default_position_id
            counts[pos] = counts.get(pos, 0.0) + 1.0
    return {pos: n / teams for pos, n in sorted(counts.items())}


def bench_hoarding_from_rosters(
    rosters: Mapping[int, TeamRoster],
    demand: Mapping[int, PositionDemand],
    *,
    include_ir: bool = True,
) -> dict[int, float]:
    """beta_q from what the league actually rosters, not from a population prior.

    `beta_q = rostered_q/T - (L_q + sum_f alpha_qf F_f)`, floored at zero. Feed the
    result back in as `bench_hoarding` and re-solve; the shares shift, so one extra
    pass is worth it if the numbers move much.

    Sanity check the caller should run: `sum(beta)` must come out at the league's
    bench slots per team. Research's priors sum to 1.8 against a seven-man bench.
    """
    carried = rostered_per_team(rosters, include_ir=include_ir)
    positions = set(carried) | set(demand)
    out: dict[int, float] = {}
    for pos in sorted(positions):
        dem = demand.get(pos)
        starters = dem.starters_per_team if dem is not None else 0.0
        out[pos] = max(0.0, carried.get(pos, 0.0) - starters)
    return out


# --------------------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------------------


def _measured_bench_hoarding(
    ctx: LeagueContext,
    outlooks: Sequence[PlayerOutlook],
    rosters: Mapping[int, TeamRoster],
    *,
    from_week: int,
    initial_shares: Mapping[int, Mapping[int, float]] | None,
    damping: float,
    bench_slots: int | None,
) -> dict[int, float]:
    """beta from this league's real rosters. The first of `value_league`'s two passes."""
    first = build_replacement_model(
        ctx,
        outlooks,
        from_week=from_week,
        initial_shares=initial_shares,
        damping=damping,
    )
    beta = bench_hoarding_from_rosters(
        rosters, {pos: lv.demand for pos, lv in first.levels.items()}
    )
    total = sum(beta.values())
    if bench_slots is not None and abs(total - bench_slots) > 1.0:
        log.warning(
            "league %s: measured bench hoarding sums to %.2f against %d bench slots a "
            "team; the difference is IR or a slot this does not model",
            ctx.league_id,
            total,
            bench_slots,
        )
    log.info(
        "league %s: bench hoarding measured at %s (sum %.2f) against the prior's %.2f",
        ctx.league_id,
        {POSITION_ABBREV.get(p, p): round(v, 2) for p, v in sorted(beta.items())},
        total,
        sum(DEFAULT_BENCH_HOARDING.values()),
    )
    return beta


def value_league(
    ctx: LeagueContext,
    outlooks: Sequence[PlayerOutlook],
    *,
    from_week: int = 1,
    bench_hoarding: Mapping[int, float] | None = None,
    rosters: Mapping[int, TeamRoster] | None = None,
    bench_slots: int | None = None,
    quotes: Mapping[int, MarketQuote] | Sequence[MarketQuote] | None = None,
    market_metric: str = "adp",
    market_limit: int | None = None,
    initial_shares: Mapping[int, Mapping[int, float]] | None = None,
    scarcity_ranks: int = DEFAULT_SCARCITY_RANKS,
    damping: float = 1.0,
) -> ValuationReport:
    """Value every projected player in one league. The entry point.

    Nothing here is memoized across contexts and nothing is stored on the module:
    call it twice with two leagues and you get two genuinely different answers,
    which is the property `test_valuation.py` exists to defend.

    **Pass `rosters` whenever you have them.** `bench_hoarding` decides how deep the
    rostered pool goes and therefore where replacement level sits, and the fallback
    `DEFAULT_BENCH_HOARDING` is a population prior summing to 1.8 against a real bench
    of 7.07-7.25. It is not a level error that cancels: it understates the bench most at
    the positions a bench is made of, so it reorders players *across* positions. With
    rosters in hand the coefficient is measured from the league instead.

    It has to be two passes, and that is structural rather than sloppy:
    `bench_hoarding_from_rosters` needs a `PositionDemand` per position to subtract the
    starters from what is carried, and demand only exists once a model has been solved.
    So: solve on the prior, measure beta against it, re-solve. The second solve is the
    one that is returned and it is the only one anything downstream sees.

    `bench_slots` is the league's bench count, for the sanity check the estimator's own
    docstring names -- `sum(beta)` must land near it. It is *not* on
    `ctx.lineup_slot_counts`, which is `settings.roster.starting_slots` and has the bench
    filtered out; the caller reads it off `settings.roster.lineup_slot_counts[SLOT_BENCH]`.
    A mismatch is logged rather than raised: it means the league carries IR or an odd
    slot, not that the valuation is unusable.
    """
    if bench_hoarding is None and rosters:
        bench_hoarding = _measured_bench_hoarding(
            ctx,
            outlooks,
            rosters,
            from_week=from_week,
            initial_shares=initial_shares,
            damping=damping,
            bench_slots=bench_slots,
        )
    elif bench_hoarding is None:
        log.info(
            "no rosters for league %s; valuing against DEFAULT_BENCH_HOARDING, which sums "
            "to %.2f against a real bench nearer seven. Replacement level will sit too "
            "shallow and the board will rank kickers and defences too high.",
            ctx.league_id,
            sum(DEFAULT_BENCH_HOARDING.values()),
        )
    model = build_replacement_model(
        ctx,
        outlooks,
        from_week=from_week,
        bench_hoarding=bench_hoarding,
        initial_shares=initial_shares,
        damping=damping,
    )
    playoffs = tuple(w for w in ctx.playoff_weeks if w >= from_week)
    values = player_values(outlooks, model, playoff_weeks=playoffs)
    curves = scarcity_curves(outlooks, model.weeks, ranks=scarcity_ranks)
    market = (
        market_disagreements(values, quotes, metric=market_metric, limit=market_limit)
        if quotes
        else ()
    )
    return ValuationReport(
        league_id=ctx.league_id,
        season=ctx.season,
        name=ctx.name,
        teams=ctx.size,
        from_week=from_week,
        replacement=model,
        values=values,
        curves=MappingProxyType(curves),
        positive_vorp_share=MappingProxyType(positive_vorp_share(values)),
        market=market,
    )
