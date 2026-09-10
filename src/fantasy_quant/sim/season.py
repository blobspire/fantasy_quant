"""Season simulator and playoff bracket: rosters in, championship probability out.

This is the machine every decision surface is ultimately denominated in. A trade, a
waiver claim and a start/sit call are comparable only because each one is priced as a
change in `P(championship)`, and that number comes from here.

It owns three things and delegates everything else. It owns the *state* (who is on
which roster, what the real schedule says, and what has already happened), the
*bracket*, and the *bookkeeping* that turns a tensor of weekly points into standings
and title odds. `sim/distributions.py` draws the tensor; `sim/lineup.py` picks the
starters. Neither is reimplemented here.

Four things separate this from the season simulators in the wild (`ffsimulator` and
its descendants), and each is a correctness issue rather than a refinement.

**1. Played weeks are facts.** The sim starts from the actual standing -- real record,
real points-for, real all-play -- and simulates only the games that have not happened.
A simulator that replays week 1 in week 10 is answering a question nobody asked, and
it will confidently tell a 2-8 team it is a title contender.

**2. Lineups are set ex ante, on projections, never on the drawn outcome.** This is
the easiest catastrophic bug in a season simulator. Pick each week's starters by
looking at the *simulated actuals* and every manager plays with hindsight: `E[max] >
max[E]` inflates every team score far past the measured 121.9 anchor, and it inflates
deep benches most, so the tool starts recommending roster clutter. `rank` is the
ex-ante ordering key and is deliberately a separate argument from the realised points.
Passing a bare tensor with no rank reproduces the hindsight-optimal lineup on purpose
-- it is the upper bound you need to price bench option value -- and
`SeasonResult.hindsight_lineups` records that you did.

**3. The bracket is the league's real one.** `playoff_team_count` teams, seeded on
record with the league's own tiebreak, byes for the top seeds when the field is not a
power of two, and multi-week rounds where `playoffMatchupPeriodLength` says so.
Playoffs are near-random, and the anchor is arithmetic: a team that wins 53% of
individual matchups converts a first-round bye into a title 0.53^2 = 28% of the time
against a coin-flip team's 0.50^2 = 25%. A 6% per-game edge becomes a 12% edge in the
title and no more. Any simulator that reports much more than that has a broken
bracket, and every seed-chasing recommendation built on it is wrong.

**4. Common random numbers throughout.** `simulate(state, tensor)` takes a pre-drawn
outcome, so two candidate rosters are judged against the *same* football, and a null
move returns exactly 0.0 rather than fog.

How much noise that actually removes depends on the metric, and the difference is
large enough that sizing a run off the wrong one will report noise as signal. Measured
here on a 12-team league, 2,000 sims an arm, one starter upgraded 15%, as the ratio of
independent-arm variance to paired variance:

    points-for   ~1000x      (continuous; the football cancels almost exactly)
    wins           ~27x      (fourteen thresholds on that football)
    championship    ~5x      (a bracket is three coin flips laid on top)

So the metric this module is denominated in is the one CRN helps *least*: the paired
standard error on a title-probability difference is about 0.5pp at 2,000 simulations,
and resolving a 0.4pp edge takes on the order of 10,000. Screen candidates on wins or
points-for, where 2,000 is ample, and spend simulations only on the survivors.

Everything is vectorised over the `[sim, week, player]` tensor. No Python loop over
simulations exists in this module; the loops that do exist run over weeks (about 17),
franchises (about 14) and playoff rounds (three).

**The lineup-efficiency asymmetry, stated plainly so it can be argued with.**
`LineupEfficiency.literal()` haircuts opponents' weekly totals while the user's
team is left whole, on the theory that the tool hands the user the optimal lineup and
nobody tells the other eleven managers anything. It is defensible in principle and
indefensible at that number. Measured on the user's three real leagues, week 1 of 2026:

    league          symmetric    with the 0.775 haircut
    Wine Wednesday       2.9%   ->   40.7%
    Blacksburg           5.2%   ->   49.5%
    Type shi             7.5%   ->   52.5%

Nothing about those rosters justifies a coin flip for the title in a 12-team league, so
the SHIPPED DEFAULT IS SYMMETRIC and the haircut is opt-in. The reason 0.775 is wrong
here is specific: 0.775 is manager points over *hindsight*-optimal points, and this simulator
already sets lineups ex ante, which by itself only realises `measure_hindsight_ratio()`
of the hindsight optimum -- 0.86 to 0.90 on these leagues. Applying 0.775 on top charges
the manager twice for the same shortfall; `rescale_for_ex_ante_lineups()` puts the
residual near 0.86-0.90, worth a few points of title probability rather than forty.

The default is left at the literal measured constant because that is what the
calibration file says and this module does not get to quietly overrule it. Pass
`LineupEfficiency.symmetric()` for the honest baseline, `LineupEfficiency.perfect()` to
remove the whole question, and read the numbers above before shipping the default.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import WireLevel
from .distributions import Draw, SimPanel
from .lineup import LineupPlan, monotone_floor, plan_from_slots

if TYPE_CHECKING:  # pragma: no cover - only the type checker needs these
    from ..core import PlayerOutlook, WeeklyOutlook
    from ..espn.league import League, LeagueSettings, Matchup, TeamRoster

log = logging.getLogger(__name__)

#: Manager lineup efficiency, measured: realised points over the best lineup that
#: roster could have started. Applied to opponents only -- see the module docstring
#: for why applying it literally on top of ex-ante lineups double-counts.
OPPONENT_LINEUP_EFFICIENCY = 0.775
OPPONENT_LINEUP_EFFICIENCY_SD = 0.05

#: Corpus anchors: 12-team PPR, nine starters, lineups set on projections.
ANCHOR_TEAM_MEAN = 121.9
ANCHOR_TEAM_SD = 24.35
ANCHOR_TEAM_SKEW = 0.27

#: ESPN's `playoffSeedingRule` is the *tiebreaker*, not the primary sort. Win
#: percentage always seeds first; this only settles teams level on it.
TIEBREAK_POINTS_FOR = "TOTAL_POINTS_SCORED"

#: The efficiency draw is deliberately reproducible: two candidate rosters must meet
#: the same set of opposing managers, not merely the same football.
_EFFICIENCY_SEED = 0xF00D5EED

#: Leagues whose seeding tiebreak we have already grumbled about.
_TIEBREAK_WARNED: set[int] = set()


class SeasonError(ValueError):
    """The league state or the tensor handed in cannot be simulated as given."""


# --------------------------------------------------------------------------------------
# League state
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerPool:
    """The player axis of the tensor: column `i` is `player_ids[i]`.

    One pool for the whole league rather than one per roster, because a candidate move
    hands a player from one franchise to another and the tensor must not have to be
    redrawn when it does. Ids are held ascending to match `SimPanel`, so a panel built
    over the same players lines up column for column.
    """

    player_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    pro_team_ids: tuple[int, ...]
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.player_ids)
        if len(self.position_ids) != n or len(self.pro_team_ids) != n:
            raise SeasonError("PlayerPool arrays must be the same length")
        if self.names and len(self.names) != n:
            raise SeasonError("PlayerPool names must match the player ids")
        if len(set(self.player_ids)) != n:
            raise SeasonError("PlayerPool has duplicate player ids")
        if list(self.player_ids) != sorted(self.player_ids):
            raise SeasonError("PlayerPool ids must be ascending to align with SimPanel")

    @classmethod
    def of(cls, rows: Iterable[tuple[int, int, int, str]]) -> PlayerPool:
        """Build from `(player_id, position_id, pro_team_id, name)` rows, deduped and sorted."""
        by_id = {int(pid): (int(pos), int(team), str(name)) for pid, pos, team, name in rows}
        ids = sorted(by_id)
        return cls(
            player_ids=tuple(ids),
            position_ids=tuple(by_id[i][0] for i in ids),
            pro_team_ids=tuple(by_id[i][1] for i in ids),
            names=tuple(by_id[i][2] for i in ids),
        )

    @property
    def size(self) -> int:
        return len(self.player_ids)

    @property
    def index(self) -> dict[int, int]:
        return {pid: i for i, pid in enumerate(self.player_ids)}

    def columns(self, player_ids: Sequence[int]) -> np.ndarray:
        """Tensor columns for these players, in the order given."""
        index = self.index
        try:
            return np.array([index[int(p)] for p in player_ids], dtype=np.intp)
        except KeyError as err:
            raise SeasonError(f"player {err.args[0]} is not in the pool") from None

    def positions_of(self, player_ids: Sequence[int]) -> np.ndarray:
        return np.asarray(self.position_ids, dtype=np.int64)[self.columns(player_ids)]

    def name(self, player_id: int) -> str:
        if not self.names:
            return str(player_id)
        return self.names[self.index[player_id]]


@dataclass(frozen=True, slots=True)
class Franchise:
    """One team, with everything already played folded into the starting numbers."""

    team_id: int
    name: str
    #: Every player the franchise controls, starters and bench alike. Which of them
    #: start is decided per week by the lineup solver, never by ESPN's slot ids --
    #: the slot a player sits in today says what last week's manager did, not what
    #: this week's optimum is.
    player_ids: tuple[int, ...]
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    #: All-play record over the weeks already played. Half a win for a tie.
    all_play_wins: float = 0.0
    all_play_games: int = 0
    #: The user's own franchise skips the lineup-efficiency haircut.
    is_user: bool = False

    def with_players(self, player_ids: Iterable[int]) -> Franchise:
        return replace(self, player_ids=tuple(player_ids))

    def without(self, player_id: int) -> Franchise:
        return self.with_players(p for p in self.player_ids if p != player_id)


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """One unplayed head-to-head. `weeks` is scoring periods, not matchup periods.

    A matchup period can span several scoring periods -- ESPN's own 2026 template maps
    matchup period 16 onto scoring periods 16 *and* 17 -- so a game's score is a sum
    over `weeks` and never a single week's total.
    """

    matchup_period: int
    weeks: tuple[int, ...]
    home_team_id: int
    away_team_id: int


@dataclass(frozen=True, slots=True)
class LeagueState:
    """Everything the simulator needs, with played weeks already reduced to facts.

    `weeks` is the tensor's week axis: the scoring periods still to be simulated, in
    ascending order. Anything outside it either already happened or is not played.
    """

    league_id: int
    season: int
    name: str
    franchises: tuple[Franchise, ...]
    pool: PlayerPool
    weeks: tuple[int, ...]
    remaining_games: tuple[ScheduledGame, ...]
    #: slotId -> count, starting slots only. Slot ids, NOT position ids.
    lineup_slot_counts: Mapping[int, int]
    #: slotId -> the defaultPositionIds it accepts.
    slot_eligibility: Mapping[int, frozenset[int]]
    playoff_team_count: int
    #: One entry per bracket round, holding that round's scoring periods. A six-team
    #: bracket has three rounds; the first is a bye for the top two seeds.
    playoff_rounds: tuple[tuple[int, ...], ...]
    playoff_seeding_rule: str = TIEBREAK_POINTS_FOR
    playoff_reseed: bool = False
    my_team_id: int | None = None

    def __post_init__(self) -> None:
        if not self.franchises:
            raise SeasonError("a league needs franchises")
        ids = [f.team_id for f in self.franchises]
        if len(set(ids)) != len(ids):
            raise SeasonError("duplicate team ids in state")
        if list(self.weeks) != sorted(set(self.weeks)):
            raise SeasonError("weeks must be ascending and distinct")
        if self.playoff_rounds:
            # A field and a round count that disagree would crown a champion with half
            # the bracket still standing -- silently, since the last round simply reads
            # whichever team happens to sit in slot zero. Refuse rather than pretend.
            needed = self.bracket_size.bit_length() - 1
            if len(self.playoff_rounds) != needed:
                raise SeasonError(
                    f"a {self.playoff_team_count}-team bracket needs {needed} rounds, "
                    f"got {len(self.playoff_rounds)}"
                )
        known = set(self.weeks)
        for game in self.remaining_games:
            missing = [w for w in game.weeks if w not in known]
            if missing:
                raise SeasonError(
                    f"game {game.home_team_id}v{game.away_team_id} needs weeks {missing}, "
                    f"which are not on the tensor axis {self.weeks}"
                )
        for r, weeks in enumerate(self.playoff_rounds, start=1):
            missing = [w for w in weeks if w not in known]
            if missing:
                raise SeasonError(f"playoff round {r} needs weeks {missing}, not on the axis")
        if self.playoff_team_count > len(self.franchises):
            raise SeasonError("more playoff spots than franchises")

    @property
    def size(self) -> int:
        return len(self.franchises)

    @property
    def team_ids(self) -> tuple[int, ...]:
        return tuple(f.team_id for f in self.franchises)

    @property
    def week_index(self) -> dict[int, int]:
        return {w: i for i, w in enumerate(self.weeks)}

    @property
    def team_index(self) -> dict[int, int]:
        return {f.team_id: i for i, f in enumerate(self.franchises)}

    @property
    def regular_season_weeks(self) -> tuple[int, ...]:
        """Remaining scoring periods that carry a head-to-head game."""
        weeks: set[int] = set()
        for game in self.remaining_games:
            weeks.update(game.weeks)
        return tuple(sorted(weeks))

    @property
    def bracket_size(self) -> int:
        """The power of two the field is padded up to. Six teams -> eight slots."""
        size = 1
        while size < max(self.playoff_team_count, 1):
            size *= 2
        return size

    @property
    def bye_count(self) -> int:
        """How many top seeds sit out the first round. Zero in a full bracket."""
        return max(self.bracket_size - self.playoff_team_count, 0)

    def franchise(self, team_id: int) -> Franchise:
        for f in self.franchises:
            if f.team_id == team_id:
                return f
        raise SeasonError(f"no team {team_id} in league {self.league_id}")

    def with_franchise(self, franchise: Franchise) -> LeagueState:
        """This state with one franchise replaced. Every candidate scenario goes through here.

        Membership is checked on the team id, not on whether the tuple changed. A
        `Franchise` is a frozen value, so swapping one in for an equal one leaves the
        tuple identical -- and a "did anything change" guard then reports a team that is
        plainly present as missing. A null candidate (the same roster, priced against
        itself) is a legitimate thing for a search loop to evaluate and must return a
        state, not raise.
        """
        if all(f.team_id != franchise.team_id for f in self.franchises):
            raise SeasonError(f"no team {franchise.team_id} in league {self.league_id}")
        swapped = tuple(franchise if f.team_id == franchise.team_id else f for f in self.franchises)
        return replace(self, franchises=swapped)


# --------------------------------------------------------------------------------------
# Lineup efficiency
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupEfficiency:
    """How much of its own optimal lineup a franchise actually starts.

    SYMMETRIC BY DEFAULT. The asymmetric form is available and defensible -- the tool
    really does hand the user an optimal lineup while nobody advises the other managers
    -- but it is an unverifiable assumption, and at the published 0.775 an identical
    roster in the user's hands scores about 29% more than in a rival's. That is too
    large a claim to make silently on the user's behalf, so it must be opted into.

    Measured on the three real leagues on 2026-09-07, with the literal 0.775:

    ====================  ====================  ==================  =================
    league                weekly win rate       title odds          title, symmetric
    ====================  ====================  ==================  =================
    Wine Wednesday (14)   73.9%                 40.7%               2.9%
    Blacksburg (12)       74.7%                 48.2%               4.7%
    Type shi (12)         77.8%                 52.3%               7.8%
    ====================  ====================  ==================  =================

    Nobody wins three matchups in four on lineup setting alone, so the default as
    written is not believable. The reason is stated in the module docstring: 0.775 is
    manager points over *hindsight*-optimal points, this simulator already sets lineups
    ex ante, and `measure_hindsight_ratio` puts the ex-ante lineup at 0.88-0.90 of the
    hindsight optimum on these same leagues -- so roughly half the measured shortfall is
    charged twice. `calibrated()` divides it back out and lands near 0.87; `symmetric()`
    removes the asymmetry entirely.

    `opponent_sd` spreads the haircut across managers rather than across weeks: lineup
    skill is a property of a manager, so the draw is per (sim, franchise) and held for
    the whole season. Drawing it weekly would average out to a constant and understate
    the real spread between the sharp and the absent managers in a league.
    """

    #: Symmetric by DEFAULT, which is the assumption-free baseline.
    #:
    #: The asymmetry is defensible in principle -- the tool really does hand the user
    #: an optimal lineup and nobody advises the other managers -- but it is an
    #: unverifiable assumption that, at the literal 0.775, flips the user from a
    #: below-baseline team to a title favourite on every one of the three real
    #: leagues. A tool should not manufacture that conclusion in its own defaults.
    #: Opt in with `LineupEfficiency.calibrated(measure_hindsight_ratio(...))`, which
    #: divides out the ex-ante double-count and lands near 0.87.
    opponent_mean: float = 1.0
    opponent_sd: float = 0.0
    user_mean: float = 1.0
    user_sd: float = 0.0

    @classmethod
    def symmetric(cls, value: float = 1.0, sd: float = 0.0) -> LineupEfficiency:
        """Everyone treated the same. This is now also the default."""
        return cls(opponent_mean=value, opponent_sd=sd, user_mean=value, user_sd=sd)

    @classmethod
    def literal(cls) -> LineupEfficiency:
        """The published 0.775 haircut, applied as written.

        Kept reachable so the number in the literature can be reproduced, but it
        double-counts on an ex-ante lineup and is not a sensible default. See
        `rescale_for_ex_ante_lineups`.
        """
        return cls(
            opponent_mean=OPPONENT_LINEUP_EFFICIENCY, opponent_sd=OPPONENT_LINEUP_EFFICIENCY_SD
        )

    @classmethod
    def calibrated(cls, hindsight_ratio: float, sd: float | None = None) -> LineupEfficiency:
        """The asymmetric haircut with the ex-ante double-count divided back out.

        Pass the number `measure_hindsight_ratio` gives you for this league. On ours
        that is 0.88-0.90, which puts the opponent haircut at about 0.87 rather than
        0.775 and the user's weekly win rate back into the plausible fifties.
        """
        return cls(
            opponent_mean=rescale_for_ex_ante_lineups(hindsight_ratio),
            opponent_sd=OPPONENT_LINEUP_EFFICIENCY_SD if sd is None else sd,
        )

    @classmethod
    def perfect(cls) -> LineupEfficiency:
        """No haircut anywhere. Every team scores its own ex-ante optimum.

        The right setting for any test that needs a weekly total to equal a number you
        can add up by hand, and the right setting for a reviewer who wants to see the
        answer with the contentious constant taken out entirely.
        """
        return cls.symmetric(1.0, 0.0)

    @property
    def is_asymmetric(self) -> bool:
        return self.user_mean != self.opponent_mean or self.user_sd != self.opponent_sd

    def draw(self, state: LeagueState, n_sims: int, *, seed: int = _EFFICIENCY_SEED) -> np.ndarray:
        """`(n_sims, n_teams)` multipliers, reproducible for common random numbers.

        Seeded from a constant rather than a caller-supplied generator so that two
        scenarios evaluated against the same tensor also meet the same opposing
        managers. Pass a different `seed` to measure how much the answer moves.
        """
        rng = np.random.default_rng(seed)
        means = np.array(
            [self.user_mean if f.is_user else self.opponent_mean for f in state.franchises]
        )
        sds = np.array([self.user_sd if f.is_user else self.opponent_sd for f in state.franchises])
        draws = means + sds * rng.standard_normal((n_sims, state.size))
        # A manager cannot beat his own optimal lineup, and a haircut past about half
        # is not lineup setting, it is not logging in.
        return np.clip(draws, 0.5, 1.0).astype(np.float32)


def rescale_for_ex_ante_lineups(
    hindsight_ratio: float, measured: float = OPPONENT_LINEUP_EFFICIENCY
) -> float:
    """Convert the measured haircut into the one that belongs on an ex-ante lineup.

    The 0.775 constant is manager points over *hindsight*-optimal points. This
    simulator already sets lineups on projections, and an ex-ante lineup only realises
    `hindsight_ratio` of the hindsight optimum by itself, so applying 0.775 on top
    charges the manager twice for the same shortfall. The residual is `measured /
    hindsight_ratio`, clamped at 1.0 -- an ex-ante optimal lineup cannot be worse than
    a real manager's on average, and a value above 1 means the sign is backwards.

    Measure `hindsight_ratio` on your own league with `measure_hindsight_ratio`.
    """
    if hindsight_ratio <= 0.0:
        raise SeasonError("hindsight_ratio must be positive")
    return min(measured / hindsight_ratio, 1.0)


# --------------------------------------------------------------------------------------
# Weekly team scores
# --------------------------------------------------------------------------------------


def lineup_plans(state: LeagueState) -> tuple[LineupPlan, ...]:
    """One compiled `LineupPlan` per franchise, in `state.franchises` order.

    Per franchise rather than per league because a plan is compiled against a specific
    roster's positions, and because laminarity is a property of *players*: an RB/WR
    slot beside a WR/TE slot is non-laminar in general, but on a roster with no tight
    end it collapses to a nested pair and the fast greedy solver is exact again.
    `sim/lineup.py` works that out; this just builds the plans and reuses them.
    """
    return tuple(
        plan_from_slots(
            state.lineup_slot_counts,
            state.slot_eligibility,
            state.pool.positions_of(f.player_ids),
        )
        for f in state.franchises
    )


def _as_points_and_rank(
    state: LeagueState,
    outcome: np.ndarray | Draw,
    rank: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalise the two ways to hand this module a season.

    A `Draw` carries its own availability mask and projected means, so it can build
    the ex-ante rank itself and the hindsight bug is impossible. A bare array cannot,
    which is why passing one without a rank is the documented opt-in to hindsight.
    """
    if isinstance(outcome, Draw):
        panel = outcome.panel
        if tuple(panel.player_ids.tolist()) != state.pool.player_ids:
            raise SeasonError("the draw's panel and the state's pool hold different players")
        if tuple(panel.weeks.tolist()) != state.weeks:
            raise SeasonError(
                f"the draw covers weeks {panel.weeks.tolist()}, the state needs {list(state.weeks)}"
            )
        points = outcome.points
        if rank is None:
            rank = ex_ante_rank(outcome)
        return points, rank
    return np.asarray(outcome), rank


def ex_ante_rank(draw: Draw) -> np.ndarray:
    """`(sims, weeks, players)` ordering key: the projected mean, minus the unavailable.

    What a manager knows on Sunday morning. The mean is constant across simulations --
    it is a projection, not an outcome -- but availability is not: a back who tore an
    ACL in the simulated week 5 is `-inf` from week 6 on in that simulation and merely
    projected in every other. That is the whole reason the rank is a tensor rather than
    a table.
    """
    mean = np.asarray(draw.panel.mean, dtype=np.float32)
    return np.where(draw.available, mean[None, :, :], -np.inf).astype(np.float32)


def team_week_scores(
    state: LeagueState,
    outcome: np.ndarray | Draw,
    *,
    rank: np.ndarray | None = None,
    efficiency: LineupEfficiency | np.ndarray | None = None,
    plans: Sequence[LineupPlan] | None = None,
    replacement: Mapping[int, float] | float | None = None,
    noise: FloorNoise | None = None,
) -> np.ndarray:
    """`(sims, weeks, teams)` starting-lineup totals.

    `outcome` is either a `distributions.Draw` or a raw `(sims, weeks, players)` array
    of realised points on `state.pool`'s column order and `state.weeks`' week order.

    `rank` is the ex-ante key used to *choose* the lineup, and it is separate from the
    points on purpose. `(weeks, players)` is the fast path: one lineup per team-week
    reused across every simulation, which is what a manager setting a lineup on Sunday
    morning actually does. `(sims, weeks, players)` lets availability vary by
    simulation, which is what you want once injuries are drawn. `None` against a raw
    array ranks on the realised points -- the hindsight-optimal lineup, correct only
    when you are deliberately measuring the upper bound, and a systematic
    overstatement of every team score otherwise. A `Draw` builds its own rank, so the
    mistake is unavailable there.

    `replacement` is what an unfilled slot scores: the best free agent that slot could
    have streamed, per lineupSlotId (or one number for all of them). Leave it `None`
    and an empty slot scores zero, which is right for "how good is this lineup" and
    badly wrong for "what is this player worth" -- see `_floors`.
    """
    points, rank = _as_points_and_rank(state, outcome, rank)
    if points.ndim != 3:
        raise SeasonError(f"points must be (sims, weeks, players), got shape {points.shape}")
    n_sims, n_weeks, n_players = points.shape
    if n_weeks != len(state.weeks):
        raise SeasonError(f"tensor has {n_weeks} weeks, state has {len(state.weeks)}")
    if n_players != state.pool.size:
        raise SeasonError(f"tensor has {n_players} players, pool has {state.pool.size}")

    rank_source = _rank_tensor(points, rank, n_weeks, n_players)

    if plans is None:
        plans = lineup_plans(state)
    if len(plans) != state.size:
        raise SeasonError(f"got {len(plans)} lineup plans for {state.size} franchises")

    scores = np.zeros((n_sims, n_weeks, state.size), dtype=np.float32)
    for t, franchise in enumerate(state.franchises):
        scores[:, :, t] = _franchise_scores(
            state.pool,
            franchise,
            plans[t],
            points,
            rank_source,
            replacement,
            floor_noise=None if noise is None else noise.for_plan(plans[t], t),
        )

    if efficiency is None:
        return scores
    factors = (
        efficiency.draw(state, n_sims)
        if isinstance(efficiency, LineupEfficiency)
        else np.asarray(efficiency, dtype=np.float32)
    )
    if factors.shape != (n_sims, state.size):
        raise SeasonError(f"efficiency must be (sims, teams), got {factors.shape}")
    return scores * factors[:, None, :]


def _rank_tensor(
    points: np.ndarray, rank: np.ndarray | None, n_weeks: int, n_players: int
) -> np.ndarray:
    """Normalise the ordering key to `(sims_or_1, weeks, players)`.

    A `(weeks, players)` rank stays one row deep, so the lineup is solved once and
    broadcast across every simulation. That is not a micro-optimisation: it is two
    orders of magnitude of work removed from the inner loop, and it is legitimate
    precisely because a projection does not vary by simulation.
    """
    if rank is None:
        return points
    out = np.asarray(rank)
    if out.ndim == 2:
        out = out[None, :, :]
    elif out.ndim != 3:
        raise SeasonError("rank must be (weeks, players) or (sims, weeks, players)")
    if out.shape[1:] != (n_weeks, n_players):
        raise SeasonError(f"rank shape {out.shape} does not match the tensor")
    return out


def _franchise_scores(
    pool: PlayerPool,
    franchise: Franchise,
    plan: LineupPlan,
    points: np.ndarray,
    rank_source: np.ndarray,
    replacement: Mapping[int, float] | float | None = None,
    floor_noise: np.ndarray | None = None,
) -> np.ndarray:
    """One franchise's `(sims, weeks)` starting-lineup totals.

    Solve on the ranking key, score on the realisation -- the two are never the same
    array unless the caller asked for hindsight. `allow_empty=True` floors every slot at
    zero, so a projected-zero or unavailable player is left on the bench rather than
    started for a guaranteed nothing.

    `floor_noise` is `(sims, weeks, n_slots)` uniforms ALREADY GATHERED INTO
    `plan.slot_ids` ORDER by the caller. That ordering is a per-roster lexsort -- seat 1
    is the D/ST on one roster and the TE on another -- so a tensor indexed by raw seat
    position is the 3-D version of the transposition this module warns about below, and
    no existing test would catch it. Passing `None` reproduces the deterministic floor
    exactly, which is what every scalar and mean-only caller still gets.
    """
    n_sims, n_weeks = points.shape[0], points.shape[1]
    if not franchise.player_ids:
        return np.zeros((n_sims, n_weeks), dtype=np.float32)
    cols = pool.columns(franchise.player_ids)
    floor, per_slot, credit, omitted = _floors(plan, replacement)
    chosen = plan.solve(rank_source[:, :, cols], floor=floor, assignment=True).assignment
    assert chosen is not None  # assignment=True always populates it
    started = np.broadcast_to(chosen, (n_sims, n_weeks, chosen.shape[-1]))
    got = np.take_along_axis(points[:, :, cols], np.where(started < 0, 0, started), axis=-1)

    # An empty seat is paid what a streamer would actually have scored, not the mean of
    # what a streamer scores. Same distinction the rest of this function already makes:
    # the lineup is CHOSEN on projections and SCORED on the realisation. Without the
    # draw, 15.7% of slot-weeks on a live league contributed a constant with no variance.
    empty = started < 0
    if credit is None or floor_noise is None:
        filled = np.where(empty, per_slot, got)
    else:
        filled = np.where(empty, credit.credit(floor_noise), got)
    total = filled.sum(axis=-1, dtype=np.float32)
    # A slot group with nothing eligible was held out of the solve so it could not lift
    # every other slot; what it really streams is added back here. See `_floors`.
    return total + np.float32(omitted) if omitted else total


class FloorNoise:
    """Uniforms for what each empty seat streams, stable across candidate rosters.

    Two properties do all the work here.

    **Independent per team and per seat.** A single shared body would cancel in
    `Var(A) + Var(B) - 2Cov(A,B)`, and empties are strongly correlated across teams
    because byes are league-wide. Measured on the fixture, sharing one body recovers
    only 7.5% of the spread against 12.6% for independent seats -- 40% of the fix
    thrown away.

    **Keyed on a CANONICAL seat, not a positional index.** `plan.slot_ids` is a
    per-roster lexsort: seat 1 is the D/ST on one roster and the TE on another. A
    tensor indexed by raw position would silently pay the kicker's draw to the tight
    end the moment a roster changed shape, which is exactly the transposition
    `_floors` warns about, one dimension up. The canonical key is
    `(slot_id, occurrence)` derived from the league's own `lineup_slot_counts`, so it
    is fixed for the league and survives any roster change.

    Fixed per `(seed, n_sims)` so common random numbers hold: two candidate rosters
    meet the same football AND the same wire.
    """

    __slots__ = ("_canonical", "_u")

    def __init__(self, state: LeagueState, draw: Draw) -> None:
        seats: list[tuple[int, int]] = []
        for slot, count in sorted(state.lineup_slot_counts.items()):
            seats.extend((int(slot), i) for i in range(int(count)))
        self._canonical = {seat: i for i, seat in enumerate(seats)}
        rng = np.random.default_rng(np.random.SeedSequence(draw.seed, spawn_key=(0xF100_0000,)))
        n_teams = max(len(state.franchises), 1)
        self._u = rng.random(
            (draw.n_sims, len(state.weeks), n_teams, max(len(seats), 1)), dtype=np.float64
        )

    def for_plan(self, plan: LineupPlan, team_index: int) -> np.ndarray:
        """`(sims, weeks, n_slots)` uniforms in this plan's own seat order."""
        seen: dict[int, int] = {}
        cols = []
        for slot in plan.slot_ids:
            occurrence = seen.get(slot, 0)
            seen[slot] = occurrence + 1
            cols.append(self._canonical.get((int(slot), occurrence), 0))
        return self._u[:, :, team_index, :][:, :, np.asarray(cols, dtype=np.intp)]


@dataclass(frozen=True, slots=True)
class _CreditParams:
    """Per-slot hurdle-gamma parameters for what an empty seat is paid.

    Solved so the credit's expectation is EXACTLY the floor the solver committed to.
    Anything else and the lineup decision and the payoff price different rosters.
    """

    p_zero: np.ndarray
    shape: np.ndarray
    scale: np.ndarray

    @classmethod
    def solve(cls, mean: np.ndarray, sd: np.ndarray, p_zero: np.ndarray) -> _CreditParams:
        from ..projections.calibration import hurdle_gamma_from_moments

        p, sh, sc = [], [], []
        for m, s_, z in zip(mean, sd, p_zero, strict=True):
            g = hurdle_gamma_from_moments(float(m), float(s_), float(z))
            p.append(g.p_zero)
            sh.append(g.shape)
            sc.append(g.scale)
        return cls(np.asarray(p), np.asarray(sh), np.asarray(sc))

    @property
    def variance(self) -> np.ndarray:
        """Per-slot variance of the credit. Tier-1 needs it to match tier-2.

        The screen prices a candidate from moments rather than by simulating, so if it
        omitted this the screen would measure a world with a deterministic floor while
        the confirm measured one without. Every add and drop changes how many seats sit
        empty, so the two would disagree on exactly the candidates being ranked.
        """
        q = 1.0 - self.p_zero
        m_pos = self.shape * self.scale
        v_pos = self.shape * self.scale * self.scale
        return q * v_pos + q * self.p_zero * m_pos * m_pos

    def credit(self, u: np.ndarray) -> np.ndarray:
        """`(sims, weeks, slots)` payouts from `(sims, weeks, slots)` uniforms."""
        from .distributions import hurdle_gamma_quantile

        return hurdle_gamma_quantile(
            u, self.p_zero[None, None, :], self.shape[None, None, :], self.scale[None, None, :]
        )


def _as_levels(replacement: Mapping[int, Any]) -> dict[int, WireLevel]:
    """Normalise a mean-only mapping or a full `WireLevel` mapping to WireLevels.

    A mean-only mapping is the historical spelling and stays deterministic: a caller
    who passes `replacement={17: 8.0}` is asserting a number, not a distribution, and
    every test that reasons about that number must keep getting it back exactly.
    """
    return {
        int(k): (v if isinstance(v, WireLevel) else WireLevel(float(v), 0.0, 0.0))
        for k, v in replacement.items()
    }


def _floors(
    plan: LineupPlan, replacement: Mapping[int, float] | float | None
) -> tuple[np.ndarray | None, np.ndarray, _CreditParams | None, float]:
    """`(solver floors, per-slot points for an unfilled slot, credit params, omitted)`.

    `None` means an unfilled slot scores zero, which prices every roster player against
    an *empty* seat. That is the right baseline for "how good is my lineup" and the
    wrong one for "what is this player worth": drop the kicker and the model loses the
    whole eight points rather than the two a waiver-wire kicker would have cost you, so
    a kicker comes out worth more title probability than a starting running back. The
    replacement level from `decide/valuation.py` is what belongs here.

    `lineup.monotone_floor` raises the floors so a wider slot never floors lower than a
    slot nested inside it -- a FLEX can stream whatever the RB slot could -- which is a
    hypothesis of the greedy exactness proof rather than a nicety.

    A `LineupPlan` carries **two** orderings of the same slot groups and they are not
    the same permutation: `floor_slot_ids` is the caller's own eligibility-row order,
    which is what `monotone_floor` takes and returns and what `solve(floor=...)` reads;
    `group_slot_ids` is the plan's internal lexsort of the eligibility matrix. Indexing
    the floor vector by the second is a silent transposition -- both are permutations of
    the same slot ids, so nothing raises -- that credits an empty TE slot with the QB's
    waiver level and vice versa. On a real roster that is several points a week landing
    on the wrong position, which is exactly the number `leave_one_out(replacement=...)`
    is reporting.

    **The empty-group guard lives here, and it did not used to.** `monotone_floor` raises
    a slot's floor to that of every slot whose eligible set it CONTAINS, eligibility is
    computed against this roster, and a roster with nobody at a position leaves that
    group's eligible set empty -- the empty set being contained in every other. So every
    slot on the team lifts to the missing position's floor. It is not a small error:
    dropping the only quarterback off the user's Blacksburg roster lifted all nine slots
    to the QB replacement level and produced a 137.6-point-a-week team with *zero*
    variance at 93% title probability, against 5.8% before the drop. All three real
    rosters carry exactly one quarterback, one kicker and one defence, so it fires on the
    first interesting candidate rather than on an edge case.

    An empty group is therefore handed a floor of zero, which lifts nothing, and the
    points its slots really stream come back as `omitted` for the caller to add to the
    team's weekly total. That is exact rather than approximate: a group with nothing
    eligible takes its floor in every week of every simulation, so the omission is a
    constant. `_franchise_scores` adds it for you; the two callers that reach past it into
    this function have to add it themselves.

    This was reimplemented twice -- `title.TitleEngine._floors_for` and
    `portfolio._floors_for` -- precisely because it was missing from the one place both
    of them route through. Two private copies of a guard is the shape of a guard living
    at the wrong altitude.
    """
    if replacement is None:
        zeros = np.zeros(plan.n_slots, dtype=np.float32)
        return None, zeros, None, 0.0

    # The solve sees MEANS only. The lift is a statement about which body a wider slot
    # could reach, so applying it to a draw would let a manager gain points by leaving
    # the FLEX empty -- hindsight through the back door. Lift the mean; sample around it.
    if not isinstance(replacement, Mapping):
        # A bare scalar asserts a number, not a distribution, and it is uniform -- so the
        # lift is a no-op and there is no empty group to guard against. Unchanged path.
        groups = monotone_floor(plan, replacement)[0]
        index = {slot: g for g, slot in enumerate(plan.floor_slot_ids)}
        per_slot = np.array([groups[index[s]] for s in plan.slot_ids], dtype=np.float32)
        return groups, per_slot, None, 0.0

    levels = _as_levels(replacement)
    omitted = 0.0
    for g in range(plan.n_groups):
        if plan._eligible[g].any():
            continue
        slot = int(plan.group_slot_ids[g])
        # A MEAN, because the group takes its floor in every week of every simulation and
        # the constant part is what the caller adds back. Zeroing the level here zeroes
        # the credit too, since both the solve floor and the hurdle derive from `levels`.
        omitted += float(levels[slot].mean) * plan.group_counts[g]
        levels[slot] = WireLevel(0.0, 0.0, 0.0)

    means = {slot: level.mean for slot, level in levels.items()}
    groups = monotone_floor(plan, means)[0]
    index = {slot: g for g, slot in enumerate(plan.floor_slot_ids)}
    per_slot = np.array([groups[index[s]] for s in plan.slot_ids], dtype=np.float32)

    if not any(level.sd > 0.0 for level in levels.values()):
        return groups, per_slot, None, omitted
    # `per_slot` is the LIFTED mean, so the credit must be re-solved against it rather
    # than against the slot's own unlifted mean, or E[credit] != the solve floor.
    spread = np.array([levels[s].sd for s in plan.slot_ids], dtype=np.float64)
    hurdle = np.array([levels[s].p_zero for s in plan.slot_ids], dtype=np.float64)
    credit = _CreditParams.solve(per_slot.astype(np.float64), spread, hurdle)
    return groups, per_slot, credit, omitted


def measure_hindsight_ratio(state: LeagueState, draw: Draw) -> float:
    """Mean ex-ante team score over mean hindsight-optimal team score.

    The conversion factor `rescale_for_ex_ante_lineups` needs, and a diagnostic in its
    own right: it is the fraction of a roster's realised ceiling that a perfectly
    projected lineup still leaves on the bench, because the ceiling is only knowable on
    Sunday night. Measured on our leagues it sits near 0.88.
    """
    ex_ante = team_week_scores(state, draw).mean()
    hindsight = team_week_scores(state, draw.points, rank=None).mean()
    if hindsight <= 0:
        raise SeasonError("hindsight-optimal scores are non-positive; nothing to measure")
    return float(ex_ante / hindsight)


# --------------------------------------------------------------------------------------
# Bracket geometry
# --------------------------------------------------------------------------------------


def bracket_seed_order(size: int) -> tuple[int, ...]:
    """Zero-based seeds in bracket-slot order for a `size`-team single elimination.

    The standard recursive mirror, so slot pairs `(0,1)`, `(2,3)`, ... are the games and
    the top two seeds can only meet in the final. Eight slots give `1-8, 4-5, 2-7, 3-6`;
    padding seeds 7 and 8 out for a six-team field leaves exactly ESPN's real bracket --
    byes for 1 and 2, `4v5` and `3v6` in the wildcard round.
    """
    if size < 1 or size & (size - 1):
        raise SeasonError(f"bracket size must be a power of two, got {size}")
    order = [0]
    while len(order) < size:
        n = len(order) * 2
        order = [s for pair in ((x, n - 1 - x) for x in order) for s in pair]
    return tuple(order)


def bye_seeds(playoff_team_count: int) -> tuple[int, ...]:
    """Zero-based seeds that sit out the first round. Empty in a full bracket."""
    size = 1
    while size < max(playoff_team_count, 1):
        size *= 2
    return tuple(range(size - playoff_team_count))


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamOutcome:
    """One franchise's distilled season. Probabilities, not counts."""

    team_id: int
    name: str
    expected_wins: float
    expected_losses: float
    expected_ties: float
    expected_points_for: float
    #: Share of all head-to-heads won against *every* team every week. Strips schedule
    #: luck exactly rather than approximately, and has far lower variance than record.
    all_play_pct: float
    make_playoffs: float
    bye: float
    reach_final: float
    championship: float
    #: `seed_distribution[k]` is P(finishing the regular season as the (k+1)th seed).
    seed_distribution: tuple[float, ...]
    mean_seed: float
    championship_stderr: float

    def __lt__(self, other: TeamOutcome) -> bool:
        return self.championship < other.championship


@dataclass(frozen=True, slots=True, eq=False)
class SeasonResult:
    """Per-simulation outcomes, kept as arrays so paired CRN differences stay exact.

    The arrays are `(sims, teams)` and small -- a 2,000 x 14 league is a few hundred
    kilobytes -- so holding them is cheaper than re-running, and a candidate comparison
    must diff them per simulation rather than diffing two summary means. Diffing the
    means throws away exactly the pairing that common random numbers bought.
    """

    league_id: int
    season: int
    team_ids: tuple[int, ...]
    team_names: tuple[str, ...]
    wins: np.ndarray
    losses: np.ndarray
    ties: np.ndarray
    points_for: np.ndarray
    all_play_wins: np.ndarray
    all_play_games: np.ndarray
    seeds: np.ndarray
    made_playoffs: np.ndarray
    byes: np.ndarray
    reached_final: np.ndarray
    champions: np.ndarray
    #: True when lineups were chosen on the realised outcome. An upper bound, not a
    #: forecast; every score in this result is inflated if it is set.
    hindsight_lineups: bool = False

    @property
    def n_sims(self) -> int:
        return int(self.wins.shape[0])

    @property
    def size(self) -> int:
        return len(self.team_ids)

    def title_odds(self) -> dict[int, float]:
        """team_id -> P(championship). Sums to 1 whenever a bracket was simulated."""
        return dict(zip(self.team_ids, self.champions.mean(axis=0).tolist(), strict=True))

    def all_play_pct(self) -> np.ndarray:
        """`(sims, teams)` all-play win rate, played weeks included."""
        return self.all_play_wins / np.maximum(self.all_play_games, 1.0)

    def outcomes(self) -> tuple[TeamOutcome, ...]:
        n = float(self.n_sims)
        champ = self.champions.mean(axis=0)
        seed_hist = np.stack(
            [np.bincount(self.seeds[:, t], minlength=self.size) / n for t in range(self.size)]
        )
        all_play = self.all_play_pct().mean(axis=0)
        out = []
        for t, team_id in enumerate(self.team_ids):
            p = float(champ[t])
            out.append(
                TeamOutcome(
                    team_id=team_id,
                    name=self.team_names[t],
                    expected_wins=float(self.wins[:, t].mean()),
                    expected_losses=float(self.losses[:, t].mean()),
                    expected_ties=float(self.ties[:, t].mean()),
                    expected_points_for=float(self.points_for[:, t].mean()),
                    all_play_pct=float(all_play[t]),
                    make_playoffs=float(self.made_playoffs[:, t].mean()),
                    bye=float(self.byes[:, t].mean()),
                    reach_final=float(self.reached_final[:, t].mean()),
                    championship=p,
                    seed_distribution=tuple(seed_hist[t].tolist()),
                    mean_seed=float(self.seeds[:, t].mean()) + 1.0,
                    championship_stderr=math.sqrt(max(p * (1.0 - p), 0.0) / n),
                )
            )
        return tuple(out)

    def by_team(self, team_id: int) -> TeamOutcome:
        for outcome in self.outcomes():
            if outcome.team_id == team_id:
                return outcome
        raise SeasonError(f"no team {team_id} in this result")

    def table(self, sort_by: str = "championship") -> str:
        """A ranked plain-text table. All-play sits next to record deliberately.

        Record contains schedule luck and all-play does not, so a team whose two
        columns disagree is telling you its record is a story about its opponents.
        """
        rows = sorted(self.outcomes(), key=lambda o: getattr(o, sort_by), reverse=True)
        head = (
            f"{'team':24s} {'W-L':>9s} {'PF':>8s} {'allplay':>8s} "
            f"{'playoff':>8s} {'bye':>7s} {'final':>7s} {'title':>8s}"
        )
        lines = [head, "-" * len(head)]
        for o in rows:
            lines.append(
                f"{o.name[:24]:24s} {o.expected_wins:4.1f}-{o.expected_losses:4.1f} "
                f"{o.expected_points_for:8.1f} {o.all_play_pct * 100:7.1f}% "
                f"{o.make_playoffs * 100:7.1f}% {o.bye * 100:6.1f}% "
                f"{o.reach_final * 100:6.1f}% {o.championship * 100:7.1f}%"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------------------


def _seed_keys(
    state: LeagueState, wins: np.ndarray, ties: np.ndarray, points_for: np.ndarray
) -> np.ndarray:
    """A single sortable float per team: record first, tiebreak second.

    `2*wins + ties` is the integer record score, shifted well clear of any plausible
    points-for total so a tiebreak can never outrank a win. ESPN's `playoffSeedingRule`
    is the *tiebreaker*, not the primary sort -- win percentage always seeds first, and
    reading that field as the sort order is a common and silent way to seed a league
    entirely by points. `TOTAL_POINTS_SCORED` is the only rule evaluable inside the
    simulation; anything else falls back to it with one warning rather than at random,
    because a head-to-head tiebreak needs the full sub-table and rarely decides a seed.
    """
    unusual = state.playoff_seeding_rule != TIEBREAK_POINTS_FOR
    if unusual and state.league_id not in _TIEBREAK_WARNED:
        _TIEBREAK_WARNED.add(state.league_id)
        log.warning(
            "league %s breaks seeding ties by %s; simulating with points-for instead",
            state.league_id,
            state.playoff_seeding_rule,
        )
    record = (2.0 * wins + ties).astype(np.float64)
    return record * 1.0e7 + points_for.astype(np.float64)


def _round_scores(scores: np.ndarray, teams: np.ndarray, week_ids: Sequence[int]) -> np.ndarray:
    """Total across a round's weeks for the team in each bracket slot.

    `teams` is `(sims, slots)` of team indices with `-1` for a bye. A bye scores `-inf`
    so its real opponent advances without the comparison having to be special cased.
    Multi-week rounds are a sum, which is the whole reason this exists: a two-week final
    halves the underdog's variance advantage, and treating it as one week does not.
    """
    real = teams >= 0
    safe = np.where(real, teams, 0)
    total = np.zeros(teams.shape, dtype=np.float32)
    for wi in week_ids:
        total += np.take_along_axis(scores[:, wi, :], safe, axis=1)
    return np.where(real, total, -np.inf)


def simulate_from_scores(
    state: LeagueState,
    scores: np.ndarray,
    *,
    all_play: bool = True,
    hindsight_lineups: bool = False,
) -> SeasonResult:
    """Standings, seeds and a champion from pre-computed `(sims, weeks, teams)` scores.

    Split out from `simulate` because the leave-one-out loop only changes one
    franchise's weekly totals: everything downstream of the scores is shared, and
    re-running it is far cheaper than re-solving a dozen rosters' lineups.
    """
    scores = np.asarray(scores, dtype=np.float32)
    n_sims = scores.shape[0]
    n_teams = state.size
    if scores.shape[1:] != (len(state.weeks), n_teams):
        raise SeasonError(f"scores shape {scores.shape} does not match the state")

    tindex = state.team_index
    windex = state.week_index

    def _start(values: Sequence[float], dtype: type) -> np.ndarray:
        return np.tile(np.asarray(values, dtype=dtype), (n_sims, 1))

    wins = _start([f.wins for f in state.franchises], np.float32)
    losses = _start([f.losses for f in state.franchises], np.float32)
    ties = _start([f.ties for f in state.franchises], np.float32)
    points_for = _start([f.points_for for f in state.franchises], np.float64)

    # -- what is left of the regular season -------------------------------------------
    games = state.remaining_games
    if games:
        home_pts = np.zeros((n_sims, len(games)), dtype=np.float32)
        away_pts = np.zeros((n_sims, len(games)), dtype=np.float32)
        onehot_home = np.zeros((len(games), n_teams), dtype=np.float32)
        onehot_away = np.zeros((len(games), n_teams), dtype=np.float32)
        for g, game in enumerate(games):
            h, a = tindex[game.home_team_id], tindex[game.away_team_id]
            wis = [windex[w] for w in game.weeks]
            home_pts[:, g] = scores[:, wis, h].sum(axis=1)
            away_pts[:, g] = scores[:, wis, a].sum(axis=1)
            onehot_home[g, h] = 1.0
            onehot_away[g, a] = 1.0
        home_win = (home_pts > away_pts).astype(np.float32)
        away_win = (away_pts > home_pts).astype(np.float32)
        drawn = 1.0 - home_win - away_win
        wins += home_win @ onehot_home + away_win @ onehot_away
        losses += away_win @ onehot_home + home_win @ onehot_away
        ties += drawn @ (onehot_home + onehot_away)
        points_for += (home_pts @ onehot_home + away_pts @ onehot_away).astype(np.float64)

    # -- all-play ---------------------------------------------------------------------
    ap_wins = _start([f.all_play_wins for f in state.franchises], np.float64)
    ap_games = _start([float(f.all_play_games) for f in state.franchises], np.float64)
    if all_play and games:
        for w in state.regular_season_weeks:
            week = scores[:, windex[w], :]
            beats = (week[:, :, None] > week[:, None, :]).sum(axis=2)
            level = (week[:, :, None] == week[:, None, :]).sum(axis=2) - 1
            ap_wins += beats + 0.5 * level
            ap_games += float(n_teams - 1)

    # -- seeding ----------------------------------------------------------------------
    keys = _seed_keys(state, wins, ties, points_for)
    order = np.argsort(-keys, axis=1, kind="stable")
    seeds = np.empty_like(order)
    np.put_along_axis(seeds, order, np.tile(np.arange(n_teams), (n_sims, 1)), axis=1)

    n_playoff = state.playoff_team_count
    made = (seeds < n_playoff) if state.playoff_rounds else np.zeros(seeds.shape, dtype=bool)
    # Byes fall out of the bracket geometry rather than the simulation: padding a
    # six-team field up to eight puts the phantom seeds opposite seeds 1 and 2, so the
    # top `bracket_size - playoff_team_count` seeds sit out the first round.
    byes = (seeds < state.bye_count) if state.playoff_rounds else np.zeros(seeds.shape, dtype=bool)
    finalists = np.zeros((n_sims, n_teams), dtype=bool)
    champions = np.zeros((n_sims, n_teams), dtype=bool)

    if state.playoff_rounds and n_playoff >= 2:
        slot_seed = np.asarray(bracket_seed_order(state.bracket_size), dtype=np.intp)
        entrant = order[:, np.clip(slot_seed, 0, n_teams - 1)]
        alive = np.where(slot_seed[None, :] < n_playoff, entrant, -1)
        alive_seed = np.tile(slot_seed, (n_sims, 1))

        rounds = list(state.playoff_rounds)
        for r, weeks in enumerate(rounds):
            if state.playoff_reseed and r > 0:
                # Best remaining seed against worst remaining seed, which is what ESPN
                # does when `playoffReseed` is on. Sort, fold the tail back onto the
                # head, and interleave so slot pairs (0,1), (2,3)... are the new games.
                keep = np.argsort(alive_seed, axis=1, kind="stable")
                alive = np.take_along_axis(alive, keep, axis=1)
                alive_seed = np.take_along_axis(alive_seed, keep, axis=1)
                half = alive.shape[1] // 2
                alive = np.stack([alive[:, :half], alive[:, half:][:, ::-1]], axis=2).reshape(
                    n_sims, -1
                )
                alive_seed = np.stack(
                    [alive_seed[:, :half], alive_seed[:, half:][:, ::-1]], axis=2
                ).reshape(n_sims, -1)

            left, right = alive[:, 0::2], alive[:, 1::2]
            lseed, rseed = alive_seed[:, 0::2], alive_seed[:, 1::2]
            wis = [windex[w] for w in weeks]
            ls = _round_scores(scores, left, wis)
            rs = _round_scores(scores, right, wis)

            if r == len(rounds) - 1:
                for side in (left, right):
                    real = side[:, 0] >= 0
                    finalists[np.nonzero(real)[0], side[real, 0]] = True

            # A playoff tie goes to the better seed: ESPN's rule, and the only tiebreak
            # available that does not invent a coin flip.
            left_wins = (ls > rs) | ((ls == rs) & (lseed < rseed))
            alive = np.where(left_wins, left, right)
            alive_seed = np.where(left_wins, lseed, rseed)

        real = alive[:, 0] >= 0
        champions[np.nonzero(real)[0], alive[real, 0]] = True

    return SeasonResult(
        league_id=state.league_id,
        season=state.season,
        team_ids=state.team_ids,
        team_names=tuple(f.name for f in state.franchises),
        wins=wins,
        losses=losses,
        ties=ties,
        points_for=points_for,
        all_play_wins=ap_wins,
        all_play_games=ap_games,
        seeds=seeds,
        made_playoffs=made,
        byes=byes,
        reached_final=finalists,
        champions=champions,
        hindsight_lineups=hindsight_lineups,
    )


def simulate(
    state: LeagueState,
    outcome: np.ndarray | Draw,
    *,
    rank: np.ndarray | None = None,
    efficiency: LineupEfficiency | np.ndarray | None = None,
    all_play: bool = True,
    plans: Sequence[LineupPlan] | None = None,
    replacement: Mapping[int, float] | float | None = None,
) -> SeasonResult:
    """Simulate the rest of the season and the bracket against a pre-drawn outcome.

    The outcome is an argument rather than something this function draws, so two
    candidate rosters are judged against identical football, and a null move returns
    exactly 0.0 instead of Monte Carlo fog. Measured variance reduction against
    independent arms is about 1000x on points-for, 27x on wins and only 5x on the
    championship indicator -- see the module docstring before choosing a simulation
    count off the title-odds column.

    `efficiency` defaults to the asymmetric haircut described in the module docstring.
    Pass `LineupEfficiency.symmetric()` to remove the asymmetry, or an explicit
    `(sims, teams)` array to hold the same managers fixed across a batch of scenarios.

    `all_play=False` in an inner search loop. All-play is an O(teams^2) comparison per
    week and is the dominant cost of everything downstream of the scores: measured on a
    12-team league at 2,000 simulations, standings plus bracket take 16 ms with it and
    1.9 ms without. It is a reporting column, not an input to the title odds.
    """
    if efficiency is None:
        efficiency = LineupEfficiency()
    hindsight = not isinstance(outcome, Draw) and rank is None
    scores = team_week_scores(
        state, outcome, rank=rank, efficiency=efficiency, plans=plans, replacement=replacement
    )
    return simulate_from_scores(state, scores, all_play=all_play, hindsight_lineups=hindsight)


# --------------------------------------------------------------------------------------
# Leave-one-out contribution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerContribution:
    """What one player is worth to his own franchise, measured by removing him.

    **Not additive.** Removing WR1 promotes WR4 into the lineup, so the sum of every
    receiver's `title_added` badly overstates what the receiving corps is worth, and
    the sum over a whole roster does not equal that franchise's title odds. This is a
    one-term Shapley approximation -- the marginal contribution against exactly one
    coalition, the full roster -- and every surface that displays it must label it as
    such. It answers "what happens if I lose this player", which is the question a
    trade or an injury actually asks; it does not answer "how should credit for this
    team be divided".
    """

    player_id: int
    name: str
    team_id: int
    wins_added: float
    all_play_added: float
    playoff_added: float
    title_added: float
    #: Standard error of the *paired* difference. Common random numbers make this far
    #: smaller than the difference of two independent title probabilities would be.
    title_added_stderr: float

    @property
    def significant(self) -> bool:
        return abs(self.title_added) > 2.0 * self.title_added_stderr

    def __lt__(self, other: PlayerContribution) -> bool:
        """Title added first, then the lower-variance measures behind it.

        Title added is the unit and must lead, but it is a probability difference over
        a bracket and it goes flat in exactly the cases a ranking still has to be
        useful: a league with no playoffs configured, a team already eliminated, a team
        already locked into the one seed. Falling through to playoff odds and then to
        wins added keeps the order meaningful there instead of leaving it to whatever
        the sort happened to find first.
        """
        return (self.title_added, self.playoff_added, self.wins_added) < (
            other.title_added,
            other.playoff_added,
            other.wins_added,
        )


def leave_one_out(
    state: LeagueState,
    outcome: np.ndarray | Draw,
    *,
    player_ids: Iterable[int] | None = None,
    rank: np.ndarray | None = None,
    efficiency: LineupEfficiency | np.ndarray | None = None,
    all_play: bool = True,
    replacement: Mapping[int, float] | float | None = None,
    noise: FloorNoise | None = None,
) -> tuple[PlayerContribution, ...]:
    """Wins added and title added per player, by removing him and re-simulating.

    Only the affected franchise's lineups are re-solved -- everyone else's weekly
    totals are reused verbatim -- but the whole league is re-standing-ed, because
    taking a player off one roster changes who his opponents beat and therefore who
    gets the last playoff seed. That second-order effect is exactly what a per-team
    calculation misses, and it is where most of the difference between "wins added"
    and "title added" comes from.

    Pass `replacement` unless you want the answer to a different question. Without it a
    removed player is replaced by an empty slot, so on a real roster the kicker prices
    out above the RB2 -- drop him and the model concedes eight points a week forever,
    when a waiver-wire kicker costs nothing and scores seven. Measured on the user's own
    leagues, an empty-slot baseline puts a kicker at +5.6pp of title probability and a
    replacement-level baseline puts him near zero, which is the difference between a
    recommendation and a joke.

    Read `PlayerContribution` before displaying any of this: the metric is a one-term
    Shapley approximation and is not additive across players.
    """
    points, rank = _as_points_and_rank(state, outcome, rank)
    if efficiency is None:
        efficiency = LineupEfficiency()
    factors = (
        efficiency.draw(state, points.shape[0])
        if isinstance(efficiency, LineupEfficiency)
        else np.asarray(efficiency, dtype=np.float32)
    )
    plans = lineup_plans(state)
    hindsight = rank is None
    rank_source = _rank_tensor(points, rank, points.shape[1], points.shape[2])
    base_scores = team_week_scores(
        state, points, rank=rank, efficiency=factors, plans=plans, replacement=replacement
    )
    base = simulate_from_scores(state, base_scores, all_play=all_play, hindsight_lineups=hindsight)
    base_champ = base.champions.astype(np.float64)
    base_ap = base.all_play_pct()
    base_playoffs = base.made_playoffs.astype(np.float64)

    owner: dict[int, tuple[int, Franchise]] = {}
    for t, f in enumerate(state.franchises):
        for pid in f.player_ids:
            owner[pid] = (t, f)
    wanted = tuple(player_ids) if player_ids is not None else tuple(owner)

    out: list[PlayerContribution] = []
    for pid in wanted:
        found = owner.get(pid)
        if found is None:
            raise SeasonError(f"player {pid} is not on any roster")
        t, franchise = found
        shortened = franchise.without(pid)
        reduced = state.with_franchise(shortened)
        # Only this franchise's lineups change; every other team's weekly total is the
        # same football, which is what makes the paired difference below exact.
        plan = plan_from_slots(
            state.lineup_slot_counts,
            state.slot_eligibility,
            state.pool.positions_of(shortened.player_ids),
        )
        solo = _franchise_scores(
            state.pool,
            shortened,
            plan,
            points,
            rank_source,
            replacement,
            floor_noise=None if noise is None else noise.for_plan(plan, t),
        )
        scores = base_scores.copy()
        scores[:, :, t] = solo * factors[:, t : t + 1]
        alt = simulate_from_scores(reduced, scores, all_play=all_play, hindsight_lineups=hindsight)
        d_title = base_champ[:, t] - alt.champions.astype(np.float64)[:, t]
        out.append(
            PlayerContribution(
                player_id=pid,
                name=state.pool.name(pid),
                team_id=franchise.team_id,
                wins_added=float((base.wins[:, t] - alt.wins[:, t]).mean()),
                all_play_added=float((base_ap[:, t] - alt.all_play_pct()[:, t]).mean()),
                playoff_added=float(
                    (base_playoffs[:, t] - alt.made_playoffs.astype(np.float64)[:, t]).mean()
                ),
                title_added=float(d_title.mean()),
                title_added_stderr=float(d_title.std(ddof=1) / math.sqrt(d_title.size)),
            )
        )
    return tuple(sorted(out, reverse=True))


# --------------------------------------------------------------------------------------
# Building the panel a draw needs
# --------------------------------------------------------------------------------------


def panel_for(
    state: LeagueState,
    outlooks: Iterable[PlayerOutlook] | Iterable[WeeklyOutlook],
    *,
    byes: Mapping[int, int] | None = None,
) -> SimPanel:
    """A `SimPanel` whose columns line up with `state.pool`, week for week.

    The tensor axes are a contract between three modules, and this is where it is
    enforced: the panel must cover exactly the pool's players and exactly the state's
    remaining weeks, or the simulator silently scores the wrong player in the wrong
    week. A player with no outlook is compiled as having no game rather than dropped,
    which keeps the columns aligned.

    Coverage is checked per *player-week*, not per player. `SimPanel.from_outlooks`
    zero-fills any week it is not given, so a source that returns only the weeks it was
    asked for -- ESPN's `mRoster` hands back about five stat rows per player, not the
    rest of the season -- produces a panel that is almost entirely zeros while
    satisfying a per-player check. That failure is invisible downstream: every team
    scores nothing, every game is a tie, and the league finishes 0-0-14 with title odds
    spread evenly. Demanding one outlook per remaining week turns it into an error at
    the boundary where it can still be attributed. A player who is genuinely out is
    still compiled as `WeeklyOutlook.zeroed()` for each week, which passes.
    """
    rows = list(outlooks)
    flat: list[WeeklyOutlook] = []
    for row in rows:
        weeks = getattr(row, "weeks", None)
        flat.extend(weeks.values() if isinstance(weeks, Mapping) else [row])  # type: ignore[arg-type]
    have = {o.player_id for o in flat}
    missing = [p for p in state.pool.player_ids if p not in have]
    if missing:
        raise SeasonError(
            f"{len(missing)} rostered players have no outlook (e.g. {missing[:5]}); "
            "compile them as zeroed outlooks rather than dropping them, or the tensor "
            "columns stop matching the pool"
        )
    covered: dict[int, set[int]] = {}
    for o in flat:
        covered.setdefault(o.player_id, set()).add(o.week)
    gaps = {
        p: sorted(set(state.weeks) - covered[p])
        for p in state.pool.player_ids
        if not set(state.weeks) <= covered[p]
    }
    if gaps:
        worst = max(gaps.values(), key=len)
        example = next(p for p, g in gaps.items() if g == worst)
        raise SeasonError(
            f"{len(gaps)} rostered players have outlooks for only part of the remaining "
            f"season (e.g. player {example} is missing weeks {worst[:8]}); an unsupplied "
            "week is silently drawn as zero, so a partial source makes every team score "
            "nothing rather than raising. Compile a WeeklyOutlook for every week in "
            "state.weeks -- zeroed() for a player who will not play."
        )
    panel = SimPanel.from_outlooks(
        [o for o in flat if o.player_id in set(state.pool.player_ids)],
        weeks=state.weeks,
        byes=byes,
    )
    if tuple(panel.player_ids.tolist()) != state.pool.player_ids:
        raise SeasonError("panel players do not match the pool; build the pool from the panel")
    return panel


# --------------------------------------------------------------------------------------
# Assembly from a live league
# --------------------------------------------------------------------------------------


def playoff_round_weeks(settings: LeagueSettings) -> tuple[tuple[int, ...], ...]:
    """Scoring periods per bracket round, honouring multi-week rounds.

    ESPN offers three ways to say how long a playoff round is and they disagree:
    `playoffMatchupPeriodLength` (where 0 means unset, not zero weeks),
    `playoffMatchupPeriodLengthByRound` (absent in older seasons, and only binding under
    `variablePlayoffMatchupPeriodLength`), and the `matchupPeriods` map itself. The map
    wins where it covers the round, because it is what the league is actually scheduled
    on. Assuming one week per round is how a two-week final gets silently halved, which
    matters: a two-week final is the single largest reducer of playoff randomness in a
    league's control.

    Every round comes back ascending. ESPN's `matchupPeriods` inner lists are documented
    as unsorted and really are, and an unsorted round would index the tensor's week axis
    out of order -- harmless for a sum, wrong the moment anything reads `weeks[0]`.
    """
    schedule = settings.schedule
    if not schedule.has_playoffs:
        return ()
    rounds = schedule.playoff_round_count
    out = [
        tuple(sorted(schedule.scoring_periods(mp)))
        for mp in schedule.playoff_matchup_periods[:rounds]
    ]
    out = [w for w in out if w]
    if len(out) < rounds:
        # ESPN mapped only part of the bracket: lay the rest end to end on each round's
        # declared length rather than dropping the final. Worth a warning -- a bracket
        # that needs more rounds than the league has scheduled periods usually means
        # `playoffTeamCount` and `matchupPeriods` disagree, and the invented weeks will
        # have no scores in them unless the caller supplies some.
        log.warning(
            "league %s schedules %d playoff matchup periods but a %d-team bracket needs "
            "%d rounds; extending the schedule rather than dropping the final",
            settings.league_id,
            len(out),
            schedule.playoff_team_count,
            rounds,
        )
        cursor = max((max(w) for w in out), default=schedule.matchup_period_count) + 1
        for r in range(len(out) + 1, rounds + 1):
            length = schedule.round_length(r)
            out.append(tuple(range(cursor, cursor + length)))
            cursor += length
    return tuple(out)


def slot_eligibility_from_rosters(
    rosters: Mapping[int, TeamRoster], starting_slots: Mapping[int, int]
) -> dict[int, frozenset[int]]:
    """slotId -> defaultPositionIds, read off the players ESPN says are eligible.

    Derived rather than hand-written because the slot and position id spaces collide at
    4 and 15, and every hand-maintained table eventually gets one of them backwards.
    `RosterEntry.eligible_slots` is ESPN's own answer for that player, so unioning it
    across the league reproduces the league's real flex rules -- including variants we
    have never seen -- with no table to keep current.
    """
    out: dict[int, set[int]] = {s: set() for s, n in starting_slots.items() if n > 0}
    for roster in rosters.values():
        for entry in roster.entries:
            for slot in entry.eligible_slots:
                if slot in out:
                    out[slot].add(entry.default_position_id)
    empty = sorted(s for s, v in out.items() if not v)
    if empty:
        raise SeasonError(f"no rostered player is eligible for starting slots {empty}")
    return {s: frozenset(v) for s, v in out.items()}


def _historical_all_play(matchups: Sequence[Matchup]) -> tuple[dict[int, float], dict[int, int]]:
    """All-play record over the weeks already scored.

    Schedule luck is the largest single source of noise in a fantasy record, and
    all-play removes it exactly rather than approximately. Reconstructing it from
    `pointsByScoringPeriod` on completed matchups costs one field we already fetched.
    """
    per_week: dict[int, dict[int, float]] = {}
    for m in matchups:
        if not m.is_complete:
            continue
        for side in (m.home, m.away):
            if side is None:
                continue
            for week, pts in side.points_by_scoring_period.items():
                per_week.setdefault(int(week), {})[side.team_id] = float(pts)
    wins: dict[int, float] = {}
    games: dict[int, int] = {}
    for scores in per_week.values():
        if len(scores) < 2:
            continue
        for team, pts in scores.items():
            beat = sum(1.0 for other, o in scores.items() if other != team and pts > o)
            level = sum(1.0 for other, o in scores.items() if other != team and pts == o)
            wins[team] = wins.get(team, 0.0) + beat + 0.5 * level
            games[team] = games.get(team, 0) + len(scores) - 1
    return wins, games


def _warn_if_bracket_started(week: int, playoff_rounds: Sequence[Sequence[int]]) -> None:
    """Say so, loudly, when the bracket is already underway.

    Silently re-simulating a game somebody has already won is the kind of error that
    looks like a plausible number, so this refuses to be quiet about it.
    """
    if not playoff_rounds:
        return
    first_playoff_week = min(w for rnd in playoff_rounds for w in rnd)
    if week >= first_playoff_week:
        warnings.warn(
            f"week {week} is inside the playoff bracket (starts week {first_playoff_week}). "
            "Playoff games already played are NOT treated as facts: the bracket is "
            "re-simulated from seeds, so a team that has already advanced is understated. "
            "Regular-season results and seeding remain exact. Treat championship "
            "probabilities as indicative until this is fixed.",
            RuntimeWarning,
            stacklevel=3,
        )


def state_from_league(
    league: League,
    *,
    my_team_id: int | None = None,
    roster_week: int | None = None,
) -> LeagueState:
    """Assemble a `LeagueState` from a live ESPN league.

    Played weeks become the starting record, points-for and all-play; only the matchups
    ESPN still calls `UNDECIDED` are simulated. Playoff matchups are deliberately not
    read from ESPN -- it publishes an empty bracket before the playoffs and a partially
    resolved one after, and neither is the bracket we want -- so the bracket is rebuilt
    from seeds by `playoff_round_weeks`.

    One limitation, stated rather than hidden: if the bracket has already begun, games
    inside it are not treated as facts. The regular season still is, so the seeds are
    right, but a playoff game already won is re-simulated. Everything up to week 15 is
    exact.

    That limitation now WARNS at runtime rather than sitting only in this docstring --
    see `_warn_if_bracket_started`. Championship probabilities computed mid-bracket are
    wrong in a specific direction: a team that has already won its first-round game is
    understated, because the simulation makes it play that game again.
    """
    settings = league.settings()
    schedule = settings.schedule
    teams = league.teams()
    week = roster_week if roster_week is not None else league.current_week()
    _warn_if_bracket_started(week, playoff_round_weeks(settings))
    rosters = league.rosters(week)
    matchups = league.matchups()

    pool = PlayerPool.of(
        (e.player_id, e.default_position_id, e.pro_team_id, e.name)
        for roster in rosters.values()
        for e in roster.entries
    )

    ap_wins, ap_games = _historical_all_play(matchups)
    franchises = tuple(
        Franchise(
            team_id=team.id,
            name=team.name,
            player_ids=tuple(e.player_id for e in rosters[team.id].entries)
            if team.id in rosters
            else (),
            wins=team.record.wins,
            losses=team.record.losses,
            ties=team.record.ties,
            points_for=team.record.points_for,
            points_against=team.record.points_against,
            all_play_wins=ap_wins.get(team.id, 0.0),
            all_play_games=ap_games.get(team.id, 0),
            is_user=my_team_id is not None and team.id == my_team_id,
        )
        for team in sorted(teams.teams, key=lambda t: t.id)
    )

    remaining = tuple(
        ScheduledGame(
            matchup_period=m.matchup_period_id,
            weeks=schedule.scoring_periods(m.matchup_period_id),
            home_team_id=m.home.team_id,
            away_team_id=m.away.team_id,
        )
        for m in matchups
        # Matchup periods past the regular season are ESPN's placeholder playoff rows;
        # the bracket is rebuilt from seeds below.
        if m.matchup_period_id <= schedule.matchup_period_count
        and not m.is_complete
        and m.home is not None
        and m.away is not None
    )

    rounds = playoff_round_weeks(settings)
    weeks: set[int] = set()
    for game in remaining:
        weeks.update(game.weeks)
    for r in rounds:
        weeks.update(r)

    starting = settings.roster.starting_slots
    return LeagueState(
        league_id=settings.league_id,
        season=settings.season,
        name=settings.name,
        franchises=franchises,
        pool=pool,
        weeks=tuple(sorted(weeks)),
        remaining_games=remaining,
        lineup_slot_counts=starting,
        slot_eligibility=slot_eligibility_from_rosters(rosters, starting),
        playoff_team_count=schedule.playoff_team_count,
        playoff_rounds=rounds,
        playoff_seeding_rule=schedule.playoff_seeding_rule,
        playoff_reseed=schedule.playoff_reseed,
        my_team_id=my_team_id,
    )
