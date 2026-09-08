"""Who you are actually playing against: per-manager behavioral estimates from the log.

Every other module in this project models football. This one models the eleven other
people, because ESPN hands over their entire decision history -- every pick, every add,
every drop, every trade proposal, with millisecond timestamps -- and no published work
appears to exist on modelling it. That makes it the largest untapped edge here and also,
by a distance, the one most likely to be noise, so the module is built around admitting
which of its own numbers are real.

**The refusal is the product, and so is the collapse.** Measured on the user's three real
leagues on 2026-09-07, seven of nine traits report *no measurable difference between
managers* and most of those cannot be estimated at all. That is the finding, not a
failure to find one. Earlier drafts of this module -- with a sampled choice set and an
undamped Newton step -- produced confident, actionable-looking recency advice for six of
Blacksburg's twelve managers. Every word of it was an artifact of the estimator, and the
sections below name each artifact and what killed it.

--------------------------------------------------------------------------------------
What the data actually is (measured 2026-09-07)
--------------------------------------------------------------------------------------

    league                    prior seasons served   non-draft transactions
    Wine Wednesday  272150391 none                   13   (2026 only)
    Blacksburg      161496047 2021, 2023, 2024, 2025 2,115 + 8 in 2026
    Type shi        634537479 none                   12   (2026 only)

**The task brief said Wine Wednesday has history and the other two are new. It is the
other way round.** `status.previousSeasons` is empty for Wine Wednesday and for Type shi;
Blacksburg reports 2021-2025 and four of those five serve a full log. Reality wins: the
module reads `previous_seasons` and takes what is there.

**Blacksburg 2022 is a trap.** It answers 200 with twelve teams, 192 draft picks and a
complete-looking settings block -- and `draftDetail.drafted` is False, there are zero
transactions, and every team is on 0.0 points. The league was created and abandoned.
Taking the picks at face value adds a phantom season of 192 holds. `SeasonRecord.played`
exists for exactly this and nothing else.

So there are two regimes and the module is built for both. In a **draft-only league**,
the only evidence in existence is sixteen picks per manager: `draft_recency`,
`name_brand` and `home_team` are estimable and the other six are refused. In a league
**with history**, `add_recency`, `sunk_cost`, `latency`, `activity` and
`trade_receptiveness` come into range for the managers who have been there a while;
`endowment` does not, and the reason is structural.

--------------------------------------------------------------------------------------
Four ESPN facts that decide what is estimable at all
--------------------------------------------------------------------------------------

**1. An executed WAIVER's `proposedDate` is the batch processing time, not the moment
the manager clicked.** Across Blacksburg's four played seasons, 417 of 443 executed
waiver claims fall in UTC hours 7-9 (3-5am ET) and the whole log touches four distinct
hours; the 546 free-agent adds beside them use all 24. So waiver timestamps carry no
information about manager behavior and `latency` reads FREEAGENT only. Using the whole
log -- the obvious thing to do -- yields a beautifully tight and entirely fictional
finding that everybody in the league acts at 3am.

**2. A resolved trade cannot be linked to its proposal.** `TRADE_ACCEPT` and
`TRADE_DECLINE` carry no `items` at all, and their `relatedTransactionId` resolves to a
transaction still present in the log for only **25 of 170** responses -- ESPN retires the
proposal record on resolution. There is therefore no way to recover which players were in
an accepted trade, which kills the textbook endowment estimator (fit `P(accept) =
f(value_in - a * value_out)` and read `a` off as WTA/WTP). What survives is the *ask*
side: the 115 `TRADE_PROPOSAL` records do carry their items with from/to team ids.

**3. Every DRAFT transaction in a season shares one timestamp** -- the draft's completion
time, not the pick's. Pick-level timing does not exist. What does is `auto_drafted`, and
it is worth more than it looks: 22-24% of picks in each of the three 2026 drafts were
made by ESPN, and one to two teams per league autodrafted **every single pick**. Those
managers have no behavior to model and their roster is ESPN's opinion, which is both a
refusal and a finding.

**4. The corpus's 2023 season-total projection column is 88% zeros** (998 of 1,128,
against ~520 in every other season), leaving 106 usable players across all positions.
`MIN_PROJECTION_POOL` skips a season rather than drawing choice sets from a board that
small.

--------------------------------------------------------------------------------------
The estimators, and what each one actually found
--------------------------------------------------------------------------------------

Every trait is a conditional logit, an OLS coefficient, or a mean, each with a real
standard error, then shrunk toward its own league's mean by empirical Bayes. The
`signal_to_noise` on each `Shrinkage` -- between-manager spread over average measurement
error -- is the number that says whether the trait separates anybody:

    trait                  Wine Wednesday   Blacksburg   Type shi
    activity                     --            3.55          --
    latency                      --            2.18          --
    trade_receptiveness          --            1.89          --
    sunk_cost                    --      0.99 CONFOUNDED     --
    draft_recency               0.00           0.00         0.89
    home_team                   0.46           0.00         0.62
    add_recency                  --            0.00          --
    name_brand                  0.00           0.00         0.00
    endowment                    --            0.00          --

`--` means refused for want of sample. **Two traits clear 1.0 and may be acted on, both on
one league, and both largely about the same eleven people.** Everything else is refused,
collapsed, marginal or confounded.

--------------------------------------------------------------------------------------
What an adversarial pass found, and what it changed
--------------------------------------------------------------------------------------

The numbers above are the second set. The first set published four things that were not
there, and each one is now a named guard with a regression test.

**`sunk_cost`'s per-manager ranking is censoring, and the advice came out backwards.**
`weeks_held` is censored at the end of the season, so a manager who drops almost nobody has
a nearly constant outcome and an OLS round slope pinned toward zero *by arithmetic*. Across
Blacksburg's eleven measured managers the fitted round coefficient correlates **+0.87** with
the share of his own draft picks a manager ever releases (and +0.65 with his move rate).
The league's most patient manager -- 21% released, the lowest rate in the league, and its
lowest activity -- came out at the bottom of that ranking and was published as "cuts his own
draft picks: his early-round busts reach the wire", which is the exact inverse of what he
does. `MAX_CENSORING_CORRELATION` now measures the correlation and refuses the ranking. The
league-wide effect (+2.66 weeks per SD earlier, 26% of it removed by the production control)
is real and still readable; what is refused is telling you which manager is different.

**`notable` did not consult whether the trait separated anybody.** Type shi's
`draft_recency` -- twelve managers, one draft, thirteen scoreable picks each, and a
signal-to-noise of 1.00 to four decimal places -- was classified `marginal` by the report,
meaning "read, do not act", and `actions()` then acted on it: "drafts last season's
leaderboard: sell him last year's name", off a raw tilt of **+0.036**. `notable` now
requires `separating`, which is the same line the report already drew.

**The rank-tilt standard error rewarded luck.** The tilt distribution is a spike at zero
plus a ~12% left tail running to -0.97 (a player coming off a lost season who projects
well). With thirteen picks the chance of drawing none of that tail is about 19%, and the
manager who drew none reported `sd/sqrt(n)` of **0.0083** against 0.15-0.30 for everyone
else -- which is not precision, it is luck, and empirical Bayes handed him almost all of his
own weight for it. The within-manager variance is now shrunk toward the league-pooled one
(`TILT_VARIANCE_PRIOR_DF`).

**`counterparty` was a threshold on a shrunk estimate.** Eight of Blacksburg's nine measured
managers had a 95% interval straddling the 0.5 "willing" cutoff; three at 0.565, 0.573 and
0.576 were labelled willing while one at 0.463 was not. Dropping any single season from the
panel moved the willing count between one and five and the unresponsive list between nobody
and two people. A label now requires the manager to be `distinct` from his own league on a
trait that separates, which takes the labelled population from five people to two.

Two further measurements did not change a verdict but did change an interval. Roster moves
are overdispersed (Pearson dispersion 2.87 across manager-seasons), so `activity`'s Poisson
standard error was about 1.7x too small and its published signal-to-noise fell from 6.19 to
3.55; and `latency` correlates **-0.85** with `activity`, so the two surviving traits are
largely one finding about the same people rather than two.

**`draft_recency` and `add_recency`.** For every pick or add, the chosen player's
percentile rank among the alternatives on the *recent* measure minus his rank on the
*forward* measure. Positive means he took the player who looked better last week or last
season than he was going to be worth. The choice set is the real one: for a draft, the
players still on the board at that position; for an add, the live free-agent pool
reconstructed by replaying draft plus every executed roster item in chronological order.

    **Finding: no measurable between-manager spread on either, in any league.**
    Blacksburg's *three corpus-covered* seasons carry the 632 wire adds that land in a
    choice set big enough to score -- 2021 is played and served by ESPN in full, the corpus
    has never captured it, and it therefore contributes none, which the report's
    `seasons_without_football` now says out loud. Across them the tilt averages +0.005 and
    the between-manager standard deviation shrinks to exactly zero. Whatever these
    managers do about last week, they all do equally, and a bias everyone shares is not an
    edge over anyone.

    The brief asks for a coefficient ratio from a regression of adds on prior-week points
    against rest-of-season projection. That regression is here, in `detail` as
    `logit_share`, and it is **not** what gets published, for two measured reasons.
    First, prior-season points and ESPN's own projection correlate at **0.71 inside the
    2026 draft choice sets actually used**, because the projection is largely a function
    of last season; splitting a shared effect between two near-parallel columns is close
    to arbitrary. Second, a manager's add is very often the single best free agent at his
    position, which is textbook separation -- the unpenalized MLE is at infinity and the
    finite answer is set by the ridge. Redrawing a sampled choice set moves the published
    tilt by up to 0.052 against a between-manager spread of 0.038, which is why the pool
    is used whole.

**`sunk_cost`.** Weeks a drafted player was held, regressed on how early he was taken,
controlling for what he actually produced. The control is the entire exercise, and the
confound is large: on Blacksburg, the naive round coefficient averages **+3.58 weeks per
standard deviation earlier in the draft** and the controlled one **+2.66**, so production
accounts for 26% of it. A real effect survives the control -- these managers hold early
picks about two and a half weeks longer than production alone predicts, out of a
seventeen-week season -- but it is a *league-wide* effect and not a discriminator, and what
looked like a discriminator turned out to be censoring. The between-manager spread (0.78) is
the same size as the measurement error (0.79), and the ranking underneath it correlates
+0.87 with how often each manager drops anybody at all, so it is refused.

    Two things suppress the estimate and both point the same way. Hold duration is
    heavily censored -- most drafted players are never dropped -- and the residual round
    effect is not purely irrational, since a slumping first-rounder really is expected to
    bounce back. Read a non-finding here as weak evidence.

**`endowment`.** Mean log ratio of value asked to value offered across a manager's own
proposals, valued at ESPN's preseason season-total projection. `exp(estimate)` is an
implied WTA/WTP; the experimental literature puts that near 2.0.

    **Finding: not estimable, and what little is measurable points the other way.** Only
    two of Blacksburg's twelve managers reached six valued proposals in four seasons, and
    their implied ratios are **0.68 and 0.92** -- they ask for *less* than they offer.
    With two managers there is no between-manager variance to estimate, so the trait
    pools completely and reports itself collapsed. Do not read the 2x prior into this
    data; it is not there, and there is not enough of the data to say it is absent
    either.

**`trade_receptiveness`.** Accepts over accepts-plus-declines. On Blacksburg the range is
1-for-9 to 9-for-9 and the spread survives a season-clustered label permutation (p = 0.002),
so there is something here.

    **It is nonetheless the weakest of the surviving traits, and it was sold as the
    strongest.** Split Blacksburg's history in half and a manager's accept rate over
    2021+2023 correlates **0.28** with his rate over 2024+2025 across eight managers -- a
    correlation whose own standard error is about 0.39, which is to say indistinguishable
    from nothing. One manager went 0-for-7 in 2023 and 4-for-6 in 2024; on the earlier half
    alone this module would have told you not to write to him. It also ignores what was
    offered, and offers are not randomly assigned: one manager received 60% of his proposals
    from a single counterparty and another 42%, so part of what this measures is who writes
    to you rather than how you answer. Read it as "who has said yes before", which is worth
    knowing, and not as a property of the person.

**`name_brand`.** For each drafted player, his percentile rank by projection minus his
rank by ADP, within position, over the draftable pool; positive means more famous than
good. Residualized on overall pick number across the whole league first, because a snake
draft hands the early slots the famous players by construction.

    **Finding: no measurable spread in any of the three leagues.** Between-manager
    standard deviation of exactly zero on all three.

**`home_team`.** Share of a manager's drafted roster from his most-represented NFL team.

    **The 1/32 baseline the brief asks for is wrong, and wrong in the direction that
    manufactures an effect.** Sixteen players drawn from a pool whose NFL teams are
    unevenly represented produce a *maximum* single-team share far above 1/32 by chance,
    because the maximum of thirty-two multinomial counts is not the mean of them.
    Measured on Blacksburg's real 2026 draft: permutation null **0.143**, observed mean
    **0.130**, 1/32 = 0.031. The 1/32 baseline would have called the average manager
    **4.2x stacked** when the truth is that these managers are, very slightly, *less*
    concentrated than chance. The null used here is a permutation of the league's own
    drafted players across its own managers, preserving pick counts, 2,000 times.

**`latency`.** Among the managers who made a free-agent add in a given week, where in the
ordering this one fell, as a percentile. Real spread on Blacksburg: 0.29 (91 adds, first
to the wire) to 0.84.

    The obvious objection is that only a manager's *first* add of the week counts, so a
    manager who makes more adds is taking the minimum of more draws and will look faster for
    nothing. That was tested. Holding every manager's real per-week add counts and giving
    them all one shared arrival-time distribution produces a between-manager spread of 0.030
    against the 0.139 observed, and a signal-to-noise that never once reached the observed
    2.18 in 400 simulations; a shuffle of who-was-fastest within each week's actor set fails
    at p < 0.001 as well. The volume artifact is real and accounts for about a fifth of the
    spread. But `latency` correlates -0.85 with `activity`, so read the two as one thing
    about a person rather than two. `quiet_hours` answers the other half -- on Blacksburg the
least-contested hours are **4-5am and 1-2am ET**, where four seasons produced one to four
adds an hour.

**`activity`.** Log adds-plus-drops per week observed, with a dormancy test on the current
season under the manager's own shrunk rate. The strongest signal in the module by a
distance (S/N 3.55 after the overdispersion correction, 6.19 before it), the least
surprising -- Blacksburg's managers range from 0.76 to 5.39 moves a week, a factor of
seven -- and **the only trait here that passes an out-of-sample test**: a manager's log move
rate over 2021+2023 correlates 0.91 with his rate over 2024+2025.

--------------------------------------------------------------------------------------
Shrinkage, and why every number here is shrunk
--------------------------------------------------------------------------------------

Method-of-moments empirical Bayes, per trait, per league: the between-manager variance is
`tau^2 = max(0, Var(raw) - mean(se^2))`, and each manager moves to `mu + tau^2/(tau^2 +
se_m^2) * (raw - mu)` with posterior sd `sqrt(tau^2 se^2/(tau^2 + se^2) + se_mu^2)`. Three
consequences, all of them load-bearing:

* When the spread across managers is no larger than the measurement error, `tau^2`
  collapses to zero and **every manager is pulled all the way to the league mean**. Seven
  of the nine traits do that on real data. `Shrinkage.collapsed` says so out loud rather
  than leaving the caller to notice that every number is identical.
* A manager with two observations and one with two hundred can hand in the same raw
  estimate and get completely different published ones. That is the whole point.
* The posterior sd keeps the `se_mu` term, so a collapsed trait publishes a real interval
  rather than a zero-width one. A trait that found nothing must not be the most confident
  output in the file.

`vs_league` -- the shrunk estimate minus the league mean -- is what to act on, because
behavioral edges are relative. You do not sell a spiking running back to a manager who
chases recent points; you sell him to the one who chases them harder than his league
does. `TraitEstimate.notable` is that, plus a size bar of one between-manager standard
deviation, and it exists because nine traits across twelve managers is 108 tests and at
two standard errors a handful of false positives is the expected outcome.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .. import corpus
from ..espn.client import EspnClient, EspnError
from ..espn.league import DraftPick, League, Transaction, owner_key

log = logging.getLogger(__name__)

#: Managers act on a US clock and ESPN timestamps are epoch-UTC. Every hour-of-day
#: statement in this module is Eastern, which is where all three leagues live.
LEAGUE_TZ = ZoneInfo("America/New_York")

#: Minimum observations before a trait is published at all. These are not tuned --
#: they are the point at which the estimator's own standard error stops swamping the
#: between-manager spread we ever see, rounded to something a person can remember.
#: A refusal names the number it wanted, so the caller can see how far off it was.
MIN_N: Mapping[str, int] = {
    "draft_recency": 8,
    "add_recency": 12,
    "sunk_cost": 20,
    "endowment": 6,
    "trade_receptiveness": 5,
    "name_brand": 10,
    "home_team": 10,
    "latency": 6,
    "activity": 4,
}

#: A season is unusable for the projection-based traits below this many players carrying
#: a positive season-total projection. Not a style choice: the 2023 capture in this
#: repo's corpus has 998 of 1,128 season projections at exactly zero (against ~520 in
#: every other season), so its usable board is 106 players across all positions and any
#: choice set drawn from it is a different league's draft. A season that fails this is
#: skipped with a log line rather than silently degrading the estimate.
MIN_PROJECTION_POOL = 120

#: Managers who must clear a trait's minimum n before the empirical-Bayes step will
#: believe a between-manager variance at all. Below this the estimate of tau^2 is itself
#: noise, so the module pools completely and reports the trait as collapsed.
MIN_MANAGERS_FOR_SPREAD = 4

#: Weeks of the current season that must have happened before dormancy is even asked
#: about. In week 1 every manager has been quiet since the draft, and testing that
#: against a season-long move rate declares half the league dead.
MIN_WEEKS_FOR_DORMANCY = 3

#: Prior degrees of freedom for shrinking a manager's own within-manager variance toward
#: the league-pooled one in the rank-tilt estimators. Not cosmetic. A manager's tilt
#: sample is a spike near zero plus a ~12% left tail running to -0.97 (a player coming off
#: an injury year, who ranks far lower on last season than on this season's projection).
#: With thirteen picks the chance of drawing none of that tail is about 19%, and the
#: manager who draws none reports a standard error four times smaller than everyone else
#: -- on Type shi's real 2026 draft, 0.0083 against 0.15-0.30 -- which sails through
#: shrinkage and comes out as the most precisely measured manager in the league. Pooling
#: the variance is the same order of assumption the empirical-Bayes step already makes
#: about tau^2, and it removes a pathology that manufactures precision from luck.
TILT_VARIANCE_PRIOR_DF = 8

#: |corr(manager's drop rate, manager's fitted round coefficient)| above which `sunk_cost`
#: declares itself confounded and stops emitting actions. Measured on Blacksburg's real
#: log: **+0.87** across eleven managers. Hold duration is censored at the end of the
#: season, so a manager who drops almost nobody has a nearly constant outcome and a round
#: slope pinned near zero *by arithmetic*, whatever his attachment to early picks. The
#: league's least active manager (0.76 moves/week, 21% of his own picks ever released --
#: the lowest rate in the league) was therefore published as "cuts his own draft picks:
#: his early-round busts reach the wire", which is the exact inverse of what he does.
MAX_CENSORING_CORRELATION = 0.6

#: Traits where a larger number means the manager is more exploitable in that
#: direction, used only to phrase the profile text.
TRAIT_UNITS: Mapping[str, str] = {
    "draft_recency": "percentile rank on last season minus rank on this season's projection",
    "add_recency": "percentile rank on last week minus rank on rest-of-season production",
    "sunk_cost": "extra weeks held per SD earlier in the draft, production held fixed",
    "endowment": "ln(value asked / value offered) in own trade proposals",
    "trade_receptiveness": "accepts / (accepts + declines)",
    "name_brand": "percentile of fame over merit, residual of pick number",
    "home_team": "modal-NFL-team roster share above the permutation null",
    "latency": "percentile position in the week's free-agent ordering (0 = first)",
    "activity": "ln(adds + drops per week observed)",
}

#: Item types that move a player onto or off a roster. A `FUTURE_ROSTER` transaction
#: carries `LINEUP` items and moves nobody, which is why it is absent here.
_ROSTER_ITEM_TYPES = frozenset({"ADD", "DROP", "DRAFT", "TRADE"})

#: Adds off the wire. The behavioral sample: a draft pick is constrained by the board and
#: a trade by a counterparty, but an add is an unforced choice from a known menu.
_WIRE_SOURCES = frozenset({"WAIVER", "FREEAGENT"})


class BehavioralError(RuntimeError):
    """The behavioral panel could not be assembled or a trait was asked for wrongly."""


# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------


def manager_id(swid: str) -> str:
    """Canonical manager key.

    Franchise ids are not stable across seasons and names are not stable at all --
    Blacksburg's team 15 changed hands between 2025 and 2026 while keeping the id. The
    SWID is, so every estimate in this module is keyed on it.
    """
    return owner_key(swid)


# --------------------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeasonRecord:
    """One (league, season) of decision history, with franchises resolved to humans.

    Kept as a plain record rather than a `League` so the estimators can be exercised on
    synthetic logs with no network anywhere, which is what every test here does.
    """

    league_id: int
    season: int
    size: int
    #: team_id -> canonical manager key. Co-owned teams resolve to the primary owner.
    owners: Mapping[int, str]
    #: manager key -> display name, best effort from the member list.
    names: Mapping[str, str]
    #: team_id -> franchise name, for output only.
    team_names: Mapping[int, str]
    picks: tuple[DraftPick, ...]
    transactions: tuple[Transaction, ...]
    final_week: int
    complete: bool
    #: ESPN's own draftDetail.drafted flag. Load-bearing: an abandoned league-season
    #: still ships a full set of placeholder picks (Blacksburg 2022 answers 200 with 192
    #: of them, zero transactions and every team on 0.0 points), so the picks alone are
    #: not evidence that a season happened.
    drafted: bool = True
    #: False when ESPN refused the transaction log for want of credentials, which is a
    #: different thing from a season in which nothing happened.
    log_available: bool = True

    def manager_of(self, team_id: int) -> str | None:
        return self.owners.get(team_id)

    @property
    def managers(self) -> frozenset[str]:
        return frozenset(self.owners.values())

    @property
    def played(self) -> bool:
        """Whether this season actually happened, rather than being a 200 that looks like it.

        `drafted` is the decisive flag; the transaction log is the corroboration, except
        where ESPN refused to serve it, in which case the picks have to do.
        """
        if not self.drafted:
            return False
        return bool(self.transactions) if self.log_available else bool(self.picks)

    @classmethod
    def from_league(cls, league: League) -> SeasonRecord:
        """Pull one league-season. Four calls plus one per scoring period of transactions."""
        settings = league.settings()
        teams = league.teams()
        owners: dict[int, str] = {}
        for team in teams.teams:
            swid = team.primary_owner or (team.owners[0] if team.owners else None)
            if swid:
                owners[team.id] = manager_id(swid)
        names = {
            manager_id(m.id): (f"{m.first_name} {m.last_name}".strip() or m.display_name)
            for m in teams.members
            if m.id
        }
        team_names = {t.id: t.name for t in teams.teams}
        log_ = league.transactions()
        if not log_.available:
            log.warning(
                "transaction log refused for league %s/%s: %s",
                league.league_id,
                league.season,
                log_.reason,
            )
        draft = league.draft()
        final_week = settings.status.final_scoring_period or settings.status.current_week
        return cls(
            league_id=league.league_id,
            season=league.season,
            size=settings.size,
            owners=owners,
            names=names,
            team_names=team_names,
            picks=draft.picks,
            transactions=tuple(log_.transactions),
            final_week=final_week,
            complete=settings.status.current_week >= final_week,
            drafted=draft.drafted,
            log_available=log_.available,
        )


@dataclass(frozen=True, slots=True)
class BehavioralPanel:
    """Every season of history we could reach for one league, plus the season in play.

    `focus_season` is the one profiles are produced for; the others exist only to give
    its managers a sample. A manager who is new to the league this year therefore gets
    a panel entry with one season in it, and most of his traits get refused, which is
    the correct description of what is known about him.
    """

    league_id: int
    focus_season: int
    seasons: tuple[SeasonRecord, ...]

    def __post_init__(self) -> None:
        if not self.seasons:
            raise BehavioralError(f"no seasons in the panel for league {self.league_id}")

    @property
    def focus(self) -> SeasonRecord:
        for s in self.seasons:
            if s.season == self.focus_season:
                return s
        raise BehavioralError(f"season {self.focus_season} not in the panel")

    @property
    def managers(self) -> tuple[str, ...]:
        """Managers in the focus season, in franchise order."""
        focus = self.focus
        return tuple(focus.owners[t] for t in sorted(focus.owners))

    def name(self, manager: str) -> str:
        for s in reversed(self.seasons):
            if manager in s.names:
                return s.names[manager]
        return manager[:8]

    def seasons_for(self, manager: str) -> tuple[SeasonRecord, ...]:
        return tuple(s for s in self.seasons if manager in s.managers and s.played)

    @property
    def history_seasons(self) -> tuple[int, ...]:
        return tuple(s.season for s in self.seasons if s.played)


def build_panel(
    league_id: int,
    season: int,
    *,
    client: EspnClient | None = None,
    history: bool = True,
    max_history: int = 8,
) -> BehavioralPanel:
    """Assemble the panel: this season plus every prior season ESPN still serves.

    Prior seasons come from `settings().status.previous_seasons` and are read from the
    ordinary `seasons/{year}` URL, which answers for them directly -- the
    `leagueHistory` endpoint exists but is not needed, and returns a single-element
    list wrapping the same payload. A season that 404s, 401s or comes back unplayed is
    dropped with a log line rather than failing the panel, because a partially
    reachable history is still worth more than none.
    """
    own_client = client is None
    client = client or _default_client()
    try:
        current = League(client, league_id, season)
        records = [SeasonRecord.from_league(current)]
        prior: tuple[int, ...] = ()
        if history:
            prior = tuple(
                y for y in sorted(current.settings().status.previous_seasons) if y != season
            )[-max_history:]
        for year in prior:
            try:
                record = SeasonRecord.from_league(League(client, league_id, year))
            except EspnError as err:
                log.info("league %s season %s unavailable: %s", league_id, year, err)
                continue
            if not record.played:
                log.info(
                    "league %s season %s exists but was never played; skipped", league_id, year
                )
                continue
            records.append(record)
    finally:
        if own_client:
            client.close()
    records.sort(key=lambda r: r.season)
    return BehavioralPanel(league_id=league_id, focus_season=season, seasons=tuple(records))


def _default_client() -> EspnClient:
    from ..pipeline import client_from_env

    return client_from_env()


# --------------------------------------------------------------------------------------
# Player facts from the corpus
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerFacts:
    """The football side of every estimator, read once through `corpus`.

    Deliberately a flat set of dicts rather than a frame: the estimators index it
    hundreds of thousands of times inside choice-set construction, and a dict lookup is
    the difference between a second and a minute.
    """

    #: (season, player_id) -> season-total actual fantasy points.
    season_actual: Mapping[tuple[int, int], float]
    #: (season, week, player_id) -> that week's actual points.
    week_actual: Mapping[tuple[int, int, int], float]
    #: (season, player_id) -> ESPN's preseason season-total projection.
    season_projection: Mapping[tuple[int, int], float]
    #: (season, player_id) -> ESPN average draft position.
    adp: Mapping[tuple[int, int], float]
    #: player_id -> defaultPositionId, latest season wins.
    position: Mapping[int, int]
    #: (season, player_id) -> NFL team id.
    pro_team: Mapping[tuple[int, int], int]
    #: (season, player_id) -> weeks with an actual row, for per-game rates.
    games: Mapping[tuple[int, int], int]
    seasons: frozenset[int]

    def rest_of_season_ppg(self, season: int, week: int, player_id: int, last_week: int) -> float:
        """Realised points per remaining week from `week` on. Hindsight, and used as such."""
        weeks = range(week, last_week + 1)
        total = sum(self.week_actual.get((season, w, player_id), 0.0) for w in weeks)
        span = max(len(list(weeks)), 1)
        return total / span


def load_player_facts(
    seasons: Iterable[int],
    *,
    root: Path | str = corpus.DEFAULT_ROOT,
    variant: str = "ppr",
) -> PlayerFacts:
    """Season totals, weekly actuals, projections, ADP and NFL team for the seasons asked.

    Reads through `corpus` rather than globbing, so the duplicate captures that would
    otherwise double-count a replayed season are collapsed the one correct way. Seasons
    with no corpus coverage simply come back absent, and every estimator that needs them
    refuses rather than imputing.
    """
    wanted = sorted({int(s) for s in seasons})
    span = sorted({s for s in wanted} | {s - 1 for s in wanted})
    try:
        rows = corpus.load_stat_rows(span, root=root, variant=variant)
    except corpus.CorpusError as err:
        raise BehavioralError(f"behavioral traits need the stat corpus: {err}") from err

    season_actual: dict[tuple[int, int], float] = {}
    week_actual: dict[tuple[int, int, int], float] = {}
    season_projection: dict[tuple[int, int], float] = {}
    position: dict[int, int] = {}
    pro_team: dict[tuple[int, int], int] = {}
    games: dict[tuple[int, int], int] = {}

    keep = rows.select(
        "espn_id",
        "default_position_id",
        "pro_team_id",
        "stat_season",
        "stat_source_id",
        "stat_split_type_id",
        "scoring_period_id",
        "applied_total",
    )
    for r in keep.iter_rows(named=True):
        pid = r["espn_id"]
        season = r["stat_season"]
        if pid is None or season is None:
            continue
        if r["default_position_id"] is not None:
            position[pid] = int(r["default_position_id"])
        if r["pro_team_id"] is not None:
            pro_team[(season, pid)] = int(r["pro_team_id"])
        total = r["applied_total"]
        if total is None:
            continue
        source, split, week = r["stat_source_id"], r["stat_split_type_id"], r["scoring_period_id"]
        if source == corpus.SOURCE_ACTUAL and split == corpus.SPLIT_SEASON:
            season_actual[(season, pid)] = float(total)
        elif source == corpus.SOURCE_ACTUAL and split == corpus.SPLIT_GAME and (week or 0) > 0:
            week_actual[(season, week, pid)] = float(total)
            games[(season, pid)] = games.get((season, pid), 0) + 1
        elif source == corpus.SOURCE_PROJECTED and split == corpus.SPLIT_SEASON:
            season_projection[(season, pid)] = float(total)

    adp: dict[tuple[int, int], float] = {}
    try:
        own = corpus.load_ownership_history(span, root=root, variant=variant)
    except corpus.CorpusError:
        own = pl.DataFrame()
    if not own.is_empty() and "average_draft_position" in own.columns:
        # Ownership history is point-in-time by design; ADP is a preseason quantity, so
        # the earliest capture of each season is the one closest to draft day.
        first = own.sort("captured_at").unique(subset=["espn_id", "request_season"], keep="first")
        for r in first.select("espn_id", "request_season", "average_draft_position").iter_rows():
            pid, season, value = r
            if pid is not None and season is not None and value is not None:
                adp[(int(season), int(pid))] = float(value)

    return PlayerFacts(
        season_actual=season_actual,
        week_actual=week_actual,
        season_projection=season_projection,
        adp=adp,
        position=position,
        pro_team=pro_team,
        games=games,
        seasons=frozenset(wanted),
    )


# --------------------------------------------------------------------------------------
# Estimates, shrinkage and the two fitters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraitEstimate:
    """One trait for one manager, with everything needed to decide whether to believe it.

    `raw` is what this manager's own data says. `estimate` is that pulled toward the
    league mean by `weight`, and `estimate` is the one to act on. `estimable` False
    means the sample never reached `required_n` and no number is published at all --
    `raw` and `estimate` are NaN in that case rather than zero, because zero is a
    claim and NaN is not.
    """

    trait: str
    manager: str
    n: int
    required_n: int
    estimable: bool
    raw: float
    raw_stderr: float
    estimate: float
    stderr: float
    weight: float
    league_mean: float
    #: The empirical-Bayes between-manager standard deviation for this trait in this
    #: league. Zero means the league has no measurable spread and every estimate is the
    #: mean; it is also the yardstick `notable` measures a manager against.
    league_spread: float = 0.0
    #: Mean measurement error across the league's managers on this trait. When
    #: `league_spread` does not clear it, the trait cannot tell managers apart at all.
    league_noise: float = float("inf")
    units: str = ""
    note: str = ""
    #: Set by an estimator that has measured a mechanical channel capable of producing
    #: this trait's whole between-manager spread on its own. A confounded trait may still
    #: be read -- the numbers are what they are -- but it never emits an action, because
    #: the ranking it produces is a re-description of the confound. `sunk_cost` sets this
    #: when the round coefficient tracks how often each manager drops anybody at all.
    confounded: bool = False
    detail: Mapping[str, float] = field(default_factory=dict)

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.estimate - 1.96 * self.stderr, self.estimate + 1.96 * self.stderr)

    @property
    def vs_league(self) -> float:
        """Shrunk estimate minus the league mean. The actionable quantity."""
        return self.estimate - self.league_mean

    @property
    def distinct(self) -> bool:
        """Whether this manager is *statistically* separated from his own league."""
        if not self.estimable or self.stderr <= 0:
            return False
        return abs(self.vs_league) > 2.0 * self.stderr

    @property
    def separating(self) -> bool:
        """Whether this trait can tell *anybody* in this league apart, at all.

        The between-manager spread has to clear the average measurement error. This is the
        same test as `Shrinkage.signal_to_noise >= 1` and the same line `BehavioralReport`
        draws between `usable_traits` and `marginal_traits`, kept on the estimate so that
        `notable` -- and therefore every action -- honours the report's own verdict.
        """
        return self.league_spread > 0 and self.league_spread >= self.league_noise

    @property
    def notable(self) -> bool:
        """Worth acting on: the trait separates, this manager is separated, and by enough.

        Three bars, and all three were put here by a specific false positive measured on
        the user's real leagues on 2026-09-07.

        1. **The trait must separate somebody** (`separating`). Without this bar Type shi's
           `draft_recency` -- a brand-new twelve-team league whose entire evidence is
           thirteen picks per manager, and whose signal-to-noise is 1.00 to four decimal
           places -- emitted "drafts last season's leaderboard: sell him last year's name"
           for one manager off a raw tilt of **+0.036**. The report already classified that
           trait `marginal`, meaning "read, do not act", and then acted on it. A trait
           whose spread does not clear its own measurement error cannot rank anybody.
        2. **Two posterior standard errors** (`distinct`), which is a claim about
           measurement rather than about size.
        3. **A full between-manager standard deviation** of separation, because with four
           seasons of picks behind it a gap of 0.09 in a trait whose whole league-wide
           spread is 0.05 clears the significance bar and means nothing at the table.

        Together these are the module's defence against multiplicity: nine traits across
        twelve managers is 108 tests. Measured against a label-shuffling null on the real
        leagues, the surviving per-trait rate of at least one spurious `notable` runs
        1-2% per trait-league, so across the roughly ten trait-leagues that carry any
        sample at all the family-wise rate is on the order of 10-15%. It is a practical
        filter, not an FDR control, and it is still the reason the advice list is short.
        """
        return (
            not self.confounded
            and self.separating
            and self.distinct
            and abs(self.vs_league) >= self.league_spread
        )

    def describe(self) -> str:
        if not self.estimable:
            return f"{self.trait}: refused (n={self.n}, needs {self.required_n})"
        lo, hi = self.ci95
        flag = "distinct" if self.distinct else "not distinct from league"
        return (
            f"{self.trait}: {self.estimate:+.3f} [{lo:+.3f},{hi:+.3f}] "
            f"(raw {self.raw:+.3f}, n={self.n}, shrink w={self.weight:.2f}, {flag})"
        )


@dataclass(frozen=True, slots=True)
class Shrinkage:
    """What the empirical-Bayes step found across the managers of one league."""

    trait: str
    league_mean: float
    between_sd: float
    mean_stderr: float
    n_managers: int
    #: Non-empty when an estimator measured a mechanical channel that could produce this
    #: trait's whole spread by itself. The text says what the channel was.
    confound: str = ""

    @property
    def signal_to_noise(self) -> float:
        """Between-manager spread over mean measurement error. Below 1 the trait is blind."""
        if self.mean_stderr <= 0 or not math.isfinite(self.mean_stderr):
            return 0.0
        return self.between_sd / self.mean_stderr

    @property
    def separates(self) -> bool:
        """Whether this trait may be acted on: enough managers, real spread, no confound."""
        return (
            not self.collapsed
            and not self.confound
            and self.n_managers >= MIN_MANAGERS_FOR_SPREAD
            and self.signal_to_noise >= 1.0
        )

    @property
    def collapsed(self) -> bool:
        """True when the spread across managers was not bigger than the measurement error.

        This is the honest no-signal verdict: every manager is pulled to the mean and the
        trait carries no information about who is who in this league.
        """
        return self.between_sd <= 0.0


@dataclass(frozen=True, slots=True)
class Fit:
    """A fitted coefficient vector with standard errors."""

    beta: np.ndarray
    stderr: np.ndarray
    n: int
    converged: bool

    def ratio(self, i: int, j: int) -> tuple[float, float]:
        """`b_i / (|b_i| + |b_j|)` and its delta-method standard error.

        The bounded form of the coefficient ratio. The unbounded `b_i / (b_i + b_j)` is
        the same number wherever both coefficients are positive and is unusable where
        they are not, which on this data is often.
        """
        bi, bj = float(self.beta[i]), float(self.beta[j])
        denom = abs(bi) + abs(bj)
        if denom <= 1e-9:
            return 0.0, float("inf")
        value = bi / denom
        # d(bi/D)/dbi = (D - |bi|)/D^2 = |bj|/D^2, and d(bi/D)/dbj = -bi*sign(bj)/D^2.
        dbi = (denom - abs(bi)) / (denom * denom)
        dbj = -bi * _sign(bj) / (denom * denom)
        var = (dbi * float(self.stderr[i])) ** 2 + (dbj * float(self.stderr[j])) ** 2
        return value, math.sqrt(max(var, 0.0))


def _sign(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def _mean_regressor_correlation(designs: Mapping[str, Sequence[np.ndarray]]) -> float:
    """Mean within-choice-set correlation of the two regressors, over every choice used."""
    values: list[float] = []
    for rows in designs.values():
        for design in rows:
            a, b = design[:, 0], design[:, 1]
            if float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
                continue
            values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def conditional_logit(
    designs: Sequence[np.ndarray],
    *,
    ridge: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Fit:
    """McFadden conditional logit by damped Newton-Raphson. Row 0 of each design is the choice.

    Each element of `designs` is one choice occasion: an `(alternatives, k)` matrix whose
    first row is the alternative actually taken. The log-likelihood is
    `sum_c (x_c0 . b - logsumexp(X_c b))`, which is concave, so Newton converges quickly
    when it converges at all.

    **Two details that are not optional on this data, both learned the hard way.**

    *The ridge is per choice occasion*, `lambda = ridge * len(designs)`. A fixed absolute
    penalty is a prior whose strength depends on how much data you have, which is
    backwards; scaled this way `ridge` is a prior precision per observation and behaves
    the same for a manager with 14 adds and one with 125.

    *Every step is line-searched.* Separation is not an edge case here -- a manager's add
    is very often the single highest-scoring free agent at his position, so the
    unpenalized MLE is at infinity. As `beta` grows the softmax saturates, the observed
    information collapses toward zero, and an undamped Newton step is then the gradient
    divided by the ridge alone: measured on Blacksburg's log, undamped iteration returned
    coefficients around -1,400 on standardized regressors, with both signs wrong. Halving
    the step until the penalized log-likelihood actually improves fixes it and costs
    nothing.

    Even so, a ridge-determined coefficient is a prior talking, so the ratio of two of
    them is worth more than either level -- which is why nothing in this module publishes
    a raw beta as a trait.
    """
    usable = [np.asarray(d, dtype=float) for d in designs if len(d) > 1]
    if not usable:
        return Fit(np.zeros(0), np.zeros(0), 0, False)
    k = usable[0].shape[1]
    if any(d.shape[1] != k for d in usable):
        raise BehavioralError("conditional_logit designs disagree on the number of regressors")
    lam = ridge * len(usable)

    def penalized_loglik(b: np.ndarray) -> float:
        total = 0.0
        for design in usable:
            eta = design @ b
            top = float(eta.max())
            total += float(eta[0]) - (top + math.log(float(np.exp(eta - top).sum())))
        return total - lam * float(b @ b)

    beta = np.zeros(k)
    current = penalized_loglik(beta)
    converged = False
    information = np.zeros((k, k))
    for _ in range(max_iter):
        grad = np.zeros(k)
        information = np.zeros((k, k))
        for design in usable:
            eta = design @ beta
            eta -= eta.max()
            weights = np.exp(eta)
            weights /= weights.sum()
            xbar = weights @ design
            grad += design[0] - xbar
            information += (design * weights[:, None]).T @ design - np.outer(xbar, xbar)
        grad -= 2.0 * lam * beta
        try:
            step = np.linalg.solve(information + 2.0 * lam * np.eye(k), grad)
        except np.linalg.LinAlgError:  # pragma: no cover - the ridge keeps this PD
            break
        scale = 1.0
        for _ in range(40):
            candidate = beta + scale * step
            value = penalized_loglik(candidate)
            if value >= current:
                break
            scale *= 0.5
        else:  # pragma: no cover - a concave objective always improves for small enough steps
            break
        beta, previous = beta + scale * step, current
        current = penalized_loglik(beta)
        if abs(current - previous) < tol and np.max(np.abs(scale * step)) < tol:
            converged = True
            break

    try:
        cov = np.linalg.inv(information + 2.0 * lam * np.eye(k))
    except np.linalg.LinAlgError:  # pragma: no cover
        cov = np.eye(k) * np.inf
    stderr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return Fit(beta=beta, stderr=stderr, n=len(usable), converged=converged)


def ols(x: np.ndarray, y: np.ndarray) -> Fit:
    """Ordinary least squares with classical standard errors.

    Uses `lstsq` rather than a normal-equation solve so a design with a collinear column
    -- which happens whenever a manager drafted only one position, or every one of his
    picks survived the season -- degrades to a minimum-norm answer with an honest
    standard error instead of raising.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        stderr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except np.linalg.LinAlgError:  # pragma: no cover
        stderr = np.full(k, np.inf)
    return Fit(beta=beta, stderr=stderr, n=n, converged=True)


def _pooled(
    trait: str,
    raw: Mapping[str, float],
    stderr: Mapping[str, float],
    counts: Mapping[str, int],
    usable: Sequence[str],
    grand: float,
    need: int,
    units: str,
    note: str,
    detail: Mapping[str, Mapping[str, float]],
    errs: np.ndarray,
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Complete pooling: every measured manager gets the league mean and nobody is distinct."""
    seen = set(usable)
    se_grand = float(1.0 / math.sqrt((1.0 / errs**2).sum())) if len(errs) else float("nan")
    out = {
        m: TraitEstimate(
            trait=trait,
            manager=m,
            n=counts.get(m, 0),
            required_n=need,
            estimable=m in seen,
            raw=float(raw[m]) if m in seen else float("nan"),
            raw_stderr=float(stderr[m]) if m in seen else float("nan"),
            estimate=grand if m in seen else float("nan"),
            stderr=se_grand if m in seen else float("nan"),
            weight=0.0,
            league_mean=grand,
            league_noise=float(errs.mean()) if len(errs) else float("inf"),
            units=units,
            note=(
                note
                if m in seen
                else (note or f"n={counts.get(m, 0)} below the {need} this trait needs")
            ),
            detail=dict(detail.get(m, {})),
        )
        for m in raw
    }
    return out, Shrinkage(
        trait=trait,
        league_mean=grand,
        between_sd=0.0,
        mean_stderr=float(errs.mean()),
        n_managers=len(seen),
    )


def empirical_bayes(
    trait: str,
    raw: Mapping[str, float],
    stderr: Mapping[str, float],
    counts: Mapping[str, int],
    *,
    units: str = "",
    note: str = "",
    detail: Mapping[str, Mapping[str, float]] | None = None,
    required_n: int | None = None,
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Shrink each manager's own estimate toward the league mean by its own precision.

    Method of moments: `tau^2 = max(0, Var(raw) - mean(se^2))` is the between-manager
    variance left after taking out what measurement error alone would produce, and the
    shrinkage weight is `tau^2 / (tau^2 + se_m^2)`. A manager measured with error much
    larger than the true spread keeps almost none of his own number; a manager measured
    precisely keeps almost all of it. When `tau^2` hits its floor the league genuinely
    has no measurable spread on this trait and everyone lands on the mean -- `Shrinkage.
    collapsed` says so rather than leaving the caller to infer it from identical output.

    Managers who did not reach `required_n` are carried through as refusals: they take no
    part in the mean or the variance, and come back with NaN.
    """
    need = MIN_N.get(trait, 1) if required_n is None else required_n
    detail = detail or {}
    usable = [
        m
        for m in raw
        if counts.get(m, 0) >= need and np.isfinite(raw[m]) and np.isfinite(stderr.get(m, np.inf))
    ]

    if not usable:
        shrink = Shrinkage(trait, float("nan"), 0.0, float("nan"), 0)
        return {
            m: TraitEstimate(
                trait=trait,
                manager=m,
                n=counts.get(m, 0),
                required_n=need,
                estimable=False,
                raw=float("nan"),
                raw_stderr=float("nan"),
                estimate=float("nan"),
                stderr=float("nan"),
                weight=0.0,
                league_mean=float("nan"),
                units=units,
                note=note or f"no manager reached n={need}",
                detail=dict(detail.get(m, {})),
            )
            for m in raw
        }, shrink

    values = np.array([raw[m] for m in usable])
    errs = np.array([max(stderr[m], 1e-9) for m in usable])
    # Precision-weighted grand mean, iterated with the resulting tau so a single noisy
    # manager cannot drag the centre the estimates are shrunk toward.
    grand = float(values.mean())
    if len(values) < MIN_MANAGERS_FOR_SPREAD:
        # A between-manager variance from two managers is not a variance. Fall back to
        # complete pooling: everyone gets the mean, nobody is reported as distinct, and
        # `Shrinkage.collapsed` says the league had too few measured managers to tell.
        tau2 = 0.0
        return _pooled(trait, raw, stderr, counts, usable, grand, need, units, note, detail, errs)
    tau2 = max(0.0, float(values.var(ddof=1)) - float((errs**2).mean()))
    for _ in range(8):
        w = 1.0 / (tau2 + errs**2)
        grand = float((w * values).sum() / w.sum())
        tau2_new = max(0.0, float((w * (values - grand) ** 2).sum() / w.sum() - (errs**2).mean()))
        if abs(tau2_new - tau2) < 1e-12:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    se_grand = float(1.0 / math.sqrt((1.0 / (tau2 + errs**2)).sum()))

    out: dict[str, TraitEstimate] = {}
    for m in raw:
        n = counts.get(m, 0)
        if m not in usable:
            out[m] = TraitEstimate(
                trait=trait,
                manager=m,
                n=n,
                required_n=need,
                estimable=False,
                raw=float("nan"),
                raw_stderr=float("nan"),
                estimate=float("nan"),
                stderr=float("nan"),
                weight=0.0,
                league_mean=grand,
                units=units,
                note=note or f"n={n} below the {need} this trait needs",
                detail=dict(detail.get(m, {})),
            )
            continue
        se = max(stderr[m], 1e-9)
        weight = tau2 / (tau2 + se * se) if tau2 > 0 else 0.0
        shrunk = grand + weight * (raw[m] - grand)
        # Posterior sd of the shrunk estimate, plus the uncertainty in the league mean it
        # was shrunk toward. Without the second term a collapsed trait publishes a
        # zero-width interval, which is the most overconfident output the module could
        # possibly produce on the traits where it has found nothing.
        post_sd = math.sqrt(tau2 * se * se / (tau2 + se * se) + se_grand * se_grand)
        out[m] = TraitEstimate(
            trait=trait,
            manager=m,
            n=n,
            required_n=need,
            estimable=True,
            raw=float(raw[m]),
            raw_stderr=float(se),
            estimate=float(shrunk),
            stderr=float(post_sd),
            weight=float(weight),
            league_mean=float(grand),
            league_spread=math.sqrt(tau2),
            league_noise=float(errs.mean()),
            units=units,
            note=note,
            detail=dict(detail.get(m, {})),
        )

    shrink = Shrinkage(
        trait=trait,
        league_mean=grand,
        between_sd=math.sqrt(tau2),
        mean_stderr=float(errs.mean()),
        n_managers=len(usable),
    )
    return out, shrink


# --------------------------------------------------------------------------------------
# Event extraction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RosterEvent:
    """One player arriving at or leaving a franchise, in chronological order."""

    season: int
    week: int
    timestamp: int
    team_id: int
    player_id: int
    #: "add" or "drop".
    direction: str
    #: The transaction type that produced it: DRAFT / WAIVER / FREEAGENT / TRADE / ROSTER.
    source: str


def roster_events(record: SeasonRecord) -> tuple[RosterEvent, ...]:
    """Every executed roster movement in one season, ordered as it happened.

    Draft picks come first, in pick order, at a synthetic timestamp before every
    transaction, because ESPN stamps all of a season's DRAFT records with the draft's
    completion time and would otherwise interleave them arbitrarily with week-1 adds.
    Commissioner actions (`is_league_manager`) are dropped -- they are not the manager's
    behavior -- as is anything not `EXECUTED`.
    """
    events: list[RosterEvent] = []
    first_stamp = min((t.proposed_date for t in record.transactions if t.proposed_date), default=0)
    for pick in sorted(record.picks, key=lambda p: p.overall_pick_number):
        events.append(
            RosterEvent(
                season=record.season,
                week=1,
                timestamp=first_stamp - 10_000_000 + pick.overall_pick_number,
                team_id=pick.team_id,
                player_id=pick.player_id,
                direction="add",
                source="DRAFT",
            )
        )

    ordered = sorted(
        (t for t in record.transactions if t.is_executed and not t.is_league_manager),
        key=lambda t: (t.scoring_period_id, t.proposed_date or 0),
    )
    for tx in ordered:
        if tx.type == "DRAFT":
            continue
        for item in tx.items:
            if item.type not in _ROSTER_ITEM_TYPES:
                continue
            stamp = tx.proposed_date or tx.process_date or 0
            if item.type in {"ADD", "DRAFT"} and item.to_team_id:
                events.append(
                    RosterEvent(
                        season=record.season,
                        week=tx.scoring_period_id,
                        timestamp=stamp,
                        team_id=item.to_team_id,
                        player_id=item.player_id,
                        direction="add",
                        source=tx.type,
                    )
                )
            elif item.type == "DROP" and item.from_team_id:
                events.append(
                    RosterEvent(
                        season=record.season,
                        week=tx.scoring_period_id,
                        timestamp=stamp,
                        team_id=item.from_team_id,
                        player_id=item.player_id,
                        direction="drop",
                        source=tx.type,
                    )
                )
            elif item.type == "TRADE":
                if item.from_team_id:
                    events.append(
                        RosterEvent(
                            season=record.season,
                            week=tx.scoring_period_id,
                            timestamp=stamp,
                            team_id=item.from_team_id,
                            player_id=item.player_id,
                            direction="drop",
                            source="TRADE",
                        )
                    )
                if item.to_team_id:
                    events.append(
                        RosterEvent(
                            season=record.season,
                            week=tx.scoring_period_id,
                            timestamp=stamp,
                            team_id=item.to_team_id,
                            player_id=item.player_id,
                            direction="add",
                            source="TRADE",
                        )
                    )
    events.sort(key=lambda e: (0 if e.source == "DRAFT" else 1, e.week, e.timestamp))
    return tuple(events)


@dataclass(frozen=True, slots=True)
class Hold:
    """How long a franchise kept a player it drafted, and how it ended."""

    season: int
    team_id: int
    player_id: int
    round_id: int
    overall_pick_number: int
    weeks_held: int
    released: bool
    release_source: str


def draft_holds(record: SeasonRecord) -> tuple[Hold, ...]:
    """Hold duration for every drafted player, censored at the end of the season.

    A player who leaves by trade is treated the same as one who is dropped -- the hold
    ended -- but `release_source` keeps them separable, because a trade is a different
    decision from a cut and a caller may want only the cuts.
    """
    if not record.picks:
        return ()
    exits: dict[tuple[int, int], RosterEvent] = {}
    for event in roster_events(record):
        if event.direction != "drop":
            continue
        key = (event.team_id, event.player_id)
        if key not in exits:
            exits[key] = event
    out: list[Hold] = []
    for pick in record.picks:
        if pick.player_id == 0:
            continue
        exit_event = exits.get((pick.team_id, pick.player_id))
        if exit_event is not None and exit_event.week <= record.final_week:
            weeks = max(exit_event.week - 1, 0)
            released, source = True, exit_event.source
        else:
            weeks, released, source = record.final_week, False, ""
        out.append(
            Hold(
                season=record.season,
                team_id=pick.team_id,
                player_id=pick.player_id,
                round_id=pick.round_id,
                overall_pick_number=pick.overall_pick_number,
                weeks_held=weeks,
                released=released,
                release_source=source,
            )
        )
    return tuple(out)


def free_agent_adds(record: SeasonRecord) -> tuple[RosterEvent, ...]:
    """Adds off the wire only: no draft picks and no trades. The behavioral sample."""
    return tuple(
        e for e in roster_events(record) if e.direction == "add" and e.source in _WIRE_SOURCES
    )


# --------------------------------------------------------------------------------------
# Trait: draft recency
# --------------------------------------------------------------------------------------


def _standardize(values: np.ndarray) -> np.ndarray:
    sd = float(values.std())
    if sd < 1e-9:
        return np.zeros_like(values)
    return (values - float(values.mean())) / sd


def _alternatives(
    pool: Sequence[int],
    chosen: int,
    n_alternatives: int | None,
    rng: np.random.Generator,
) -> list[int]:
    """The choice set, chosen alternative first. Whole pool by default.

    McFadden's positive-conditioning property means a uniformly sampled subset of the
    alternatives also gives consistent coefficients, and an earlier draft of this module
    sampled 24 of them for speed. **Do not.** Measured on Blacksburg's four played
    seasons, redrawing the sample under five seeds moves a manager's `add_recency` tilt
    by up to **0.052**, against a between-manager spread of **0.038** -- the sampling
    noise is larger than the entire signal the trait is trying to detect. The
    conditional-logit ratio computed on the same sets moves by up to 0.118. And because
    polars' `unique` does not preserve row order the pool itself arrived in a different
    order each read, so the noise was not even seeded: consecutive runs of the same code
    on the same data disagreed. The whole pool costs milliseconds and has neither
    problem, so `n_alternatives` exists only to cap a pathologically large set.

    `pool` must be in a deterministic order -- callers sort it -- because corpus dict
    order is not stable across reads.
    """
    others = [p for p in pool if p != chosen]
    if n_alternatives is not None and len(others) > n_alternatives:
        idx = sorted(rng.choice(len(others), size=n_alternatives, replace=False))
        others = [others[i] for i in idx]
    return [chosen, *others]


def draft_recency(
    panel: BehavioralPanel,
    facts: PlayerFacts,
    *,
    n_alternatives: int | None = None,
    seed: int = 7,
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """How much a manager's draft followed last season's points over this season's projection.

    At every pick, the choice set is the players still on the board at the position he
    actually took, carrying both a prior-season actual and a current-season projection.
    Both regressors are standardized within that choice set, so the coefficients are in
    the same units at pick 3 and pick 150.

    This is the only recency estimator that exists in week 1 of a brand-new league, which
    is why it carries the weight it does here.
    """
    rng = np.random.default_rng(seed)
    designs: dict[str, list[np.ndarray]] = {m: [] for m in panel.managers}
    tilts: dict[str, list[float]] = {m: [] for m in panel.managers}
    counts: dict[str, int] = {m: 0 for m in panel.managers}

    for record in panel.seasons:
        season = record.season
        pool = [
            pid
            for (s, pid), value in facts.season_projection.items()
            if s == season
            and value > 0
            and (season - 1, pid) in facts.season_actual
            and pid in facts.position
        ]
        if len(pool) < MIN_PROJECTION_POOL:
            log.info(
                "season %s has only %d players with a positive projection and a prior-season "
                "actual; skipped for draft_recency",
                season,
                len(pool),
            )
            continue
        by_position: dict[int, list[int]] = {}
        for pid in sorted(pool):
            by_position.setdefault(facts.position[pid], []).append(pid)
        taken: set[int] = set()
        for pick in sorted(record.picks, key=lambda p: p.overall_pick_number):
            manager = record.manager_of(pick.team_id)
            player = pick.player_id
            position = facts.position.get(player)
            if position is None or (season, player) not in facts.season_projection:
                taken.add(player)
                continue
            available = [p for p in by_position.get(position, ()) if p not in taken]
            taken.add(player)
            if manager not in designs or player not in available or len(available) < 3:
                continue
            if pick.auto_drafted:
                # An autodraft pick is ESPN's behavior, not the manager's.
                continue
            rows = _alternatives(available, player, n_alternatives, rng)
            past_raw = [facts.season_actual[(season - 1, p)] for p in rows]
            proj_raw = [facts.season_projection[(season, p)] for p in rows]
            past, proj = _standardize(np.array(past_raw)), _standardize(np.array(proj_raw))
            if float(past.std()) < 1e-9 or float(proj.std()) < 1e-9:
                continue
            designs[manager].append(np.column_stack([past, proj]))
            tilts[manager].append(_percentile_of(past_raw, 0) - _percentile_of(proj_raw, 0))
            counts[manager] += 1

    return _fit_recency_trait(
        "draft_recency",
        tilts,
        designs,
        counts,
        note="rank tilt over the board at each pick; autodraft picks excluded",
    )


def _fit_recency_trait(
    trait: str,
    tilts: Mapping[str, list[float]],
    designs: Mapping[str, list[np.ndarray]],
    counts: Mapping[str, int],
    *,
    note: str,
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Publish the rank tilt, and carry the conditional-logit ratio beside it as detail.

    **Why the published number is the tilt and not the logit ratio the brief asks for.**
    Both answer the same question -- how much of this manager's choice tracked the recent
    number rather than the forward one -- and on this data only one of them is stable.

    The logit ratio is not, for two measured reasons. First, the two regressors can be
    close to the same column: on the real 2026 boards, last season's points and this
    season's ESPN projection correlate at **0.71** inside the choice sets actually used,
    because the projection is largely a function of last season. Splitting a shared
    effect between two near-parallel columns is close to arbitrary. (For in-season adds
    the two are much better separated, at 0.38, so the ratio there fails for the second
    reason rather than the first.) Second, a manager's add is very often the
    highest-scoring free agent at his position, which is textbook separation: the
    unpenalized coefficients run off to infinity and the finite answer is set by the
    ridge. Sampling the choice set instead trades the separation for sampling noise that
    measures *larger than the between-manager spread the trait is trying to detect*.

    The tilt has none of that. For each choice, take the chosen player's percentile rank
    among the alternatives on the recent measure and on the forward measure, and record
    the difference. It is bounded in [-1, 1], it is zero for a manager choosing at
    random, positive for one whose adds rank higher on last week than on what the player
    was worth going forward, and its standard error is the standard error of a mean.
    Because both percentiles come from the same choice set, everything common to that
    week's wire differences out.

    `detail` still carries `logit_share`, `b_recent` and `b_forward` so a caller who
    wants the structural version can have it, clearly labelled.

    **The standard error is partially pooled across managers, and it has to be.** The tilt
    distribution is not remotely normal: it is a spike at or near zero -- the manager took
    a player the two measures agree about -- plus a long left tail running to -0.97 for a
    player coming off a lost season who projects well. On Type shi's real 2026 draft about
    12% of picks land in that tail, so a manager with thirteen picks has roughly a 19%
    chance of drawing none of it, and the one who does reports `sd/sqrt(n)` four times
    smaller than everybody else. That is not precision, it is luck, and empirical Bayes
    rewards it with almost all of its own weight. Each manager's variance is therefore
    shrunk toward the league-pooled within-manager variance with `TILT_VARIANCE_PRIOR_DF`
    prior degrees of freedom.
    """
    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    detail: dict[str, dict[str, float]] = {}
    collinearity = _mean_regressor_correlation(designs)

    # League-pooled within-manager variance, used as the prior below.
    sums = 0.0
    dof = 0
    for values in tilts.values():
        if len(values) >= 3:
            arr = np.array(values, dtype=float)
            sums += float(((arr - arr.mean()) ** 2).sum())
            dof += len(arr) - 1
    pooled_var = sums / dof if dof > 0 else 0.0

    for manager, values in tilts.items():
        rows = designs.get(manager, [])
        info: dict[str, float] = {"regressor_corr": collinearity, "choices": float(len(values))}
        if len(rows) >= 2:
            fit = conditional_logit(rows)
            share, share_se = fit.ratio(0, 1)
            info |= {
                "logit_share": share,
                "logit_share_se": share_se,
                "b_recent": float(fit.beta[0]),
                "b_forward": float(fit.beta[1]),
                "logit_converged": float(fit.converged),
            }
        detail[manager] = info
        if len(values) < 3:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            continue
        arr = np.array(values, dtype=float)
        own_var, own_dof = float(arr.var(ddof=1)), len(arr) - 1
        var = (own_dof * own_var + TILT_VARIANCE_PRIOR_DF * pooled_var) / (
            own_dof + TILT_VARIANCE_PRIOR_DF
        )
        info["own_sd"] = math.sqrt(own_var)
        info["pooled_sd"] = math.sqrt(pooled_var)
        raw[manager] = float(arr.mean())
        err[manager] = max(math.sqrt(var / len(arr)), 1e-9)

    return empirical_bayes(
        trait,
        raw,
        err,
        counts,
        units=TRAIT_UNITS[trait],
        note=note,
        detail=detail,
    )


def _percentile_of(values: Sequence[float], index: int) -> float:
    """Where `values[index]` sits in `values`, in [0, 1]. Ties share the midpoint."""
    arr = np.asarray(values, dtype=float)
    target = float(arr[index])
    below = float((arr < target).sum())
    equal = float((arr == target).sum())
    return (below + 0.5 * (equal - 1.0)) / max(len(arr) - 1, 1)


# --------------------------------------------------------------------------------------
# Trait: in-season add recency
# --------------------------------------------------------------------------------------


def add_recency(
    panel: BehavioralPanel,
    facts: PlayerFacts,
    *,
    n_alternatives: int | None = None,
    seed: int = 11,
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Weight on last week's points versus what the player was really worth from here.

    The choice set is the genuine free-agent pool at the instant of the add,
    reconstructed by replaying draft plus every executed roster item in order -- not a
    week-boundary approximation, because two managers adding in the same week compete for
    the same players and the second one's choice set is genuinely smaller.

    The forward regressor is realised points per remaining week, which is hindsight. It
    is used deliberately: the question is whether the manager weighted a spike beyond
    what the spike turned out to be worth. It is also a *noisy* measure of forward value,
    which attenuates its own coefficient and inflates the recency share, so this
    estimator leans toward finding recency bias and only differences between managers
    facing the same bias should be read.
    """
    rng = np.random.default_rng(seed)
    designs: dict[str, list[np.ndarray]] = {m: [] for m in panel.managers}
    tilts: dict[str, list[float]] = {m: [] for m in panel.managers}
    counts: dict[str, int] = {m: 0 for m in panel.managers}

    for record in panel.seasons:
        season = record.season
        weeks_with_actuals = [w for (s, w, _) in facts.week_actual if s == season]
        if not weeks_with_actuals:
            continue
        last_week = max(weeks_with_actuals)
        pool_by_position: dict[int, list[int]] = {}
        for pid in sorted(pid for (s, pid) in facts.season_actual if s == season):
            if pid in facts.position:
                pool_by_position.setdefault(facts.position[pid], []).append(pid)

        rostered: set[int] = set()
        for event in roster_events(record):
            if event.direction == "drop":
                rostered.discard(event.player_id)
                continue
            manager = record.manager_of(event.team_id)
            player = event.player_id
            week = event.week
            wire_add = event.source in _WIRE_SOURCES
            position = facts.position.get(player)
            if not wire_add or manager not in designs or position is None or week < 2:
                rostered.add(player)
                continue
            available = [
                p
                for p in pool_by_position.get(position, ())
                if p not in rostered and (season, week - 1, p) in facts.week_actual
            ]
            rostered.add(player)
            if player not in available or len(available) < 3 or week > last_week:
                continue
            rows = _alternatives(available, player, n_alternatives, rng)
            recent_raw = [facts.week_actual[(season, week - 1, p)] for p in rows]
            forward_raw = [facts.rest_of_season_ppg(season, week, p, last_week) for p in rows]
            recent = _standardize(np.array(recent_raw))
            forward = _standardize(np.array(forward_raw))
            if float(recent.std()) < 1e-9 or float(forward.std()) < 1e-9:
                continue
            designs[manager].append(np.column_stack([recent, forward]))
            tilts[manager].append(_percentile_of(recent_raw, 0) - _percentile_of(forward_raw, 0))
            counts[manager] += 1

    return _fit_recency_trait(
        "add_recency",
        tilts,
        designs,
        counts,
        note="rank tilt over the live free-agent pool; forward measure is hindsight",
    )


# --------------------------------------------------------------------------------------
# Trait: sunk cost
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SunkCostFit:
    """Naive and production-controlled draft-round effects on hold duration, side by side.

    Kept as its own record because the *gap* between the two is the finding. Publishing
    only the controlled number hides how large the confound was, and the confound is the
    reason the naive version of this trait is worthless.
    """

    manager: str
    n: int
    naive: float
    naive_stderr: float
    controlled: float
    controlled_stderr: float
    production_beta: tuple[float, ...]
    #: Share of this manager's own drafted players that ever left his roster. The outcome
    #: is censored at the end of the season for the rest, so this number is an upper bound
    #: on how much of the round effect his data could possibly show.
    released_share: float = float("nan")


def _hold_frame(
    panel: BehavioralPanel, facts: PlayerFacts
) -> dict[str, list[tuple[float, float, float, float, float]]]:
    """(weeks held, standardized round, season production, early production, released)."""
    rows: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for record in panel.seasons:
        season = record.season
        if not record.complete:
            # An unfinished season has no hold durations, only censoring at today. In week
            # 1 that is sixteen identical observations per manager saying "held all year",
            # which is not data -- and it is enough of it to halve the measured effect.
            continue
        holds = draft_holds(record)
        if not holds:
            continue
        # Standardize production within position-season so a quarterback's 300 points and
        # a kicker's 130 are on one scale; the confound lives in relative quality, not in
        # positional scoring levels.
        by_position: dict[int, list[tuple[float, float]]] = {}
        raw: list[tuple[Hold, int, float, float]] = []
        for hold in holds:
            position = facts.position.get(hold.player_id)
            total = facts.season_actual.get((season, hold.player_id))
            if position is None or total is None:
                continue
            early = sum(
                facts.week_actual.get((season, w, hold.player_id), 0.0) for w in range(1, 5)
            )
            by_position.setdefault(position, []).append((total, early))
            raw.append((hold, position, total, early))
        if not raw:
            continue
        stats = {
            pos: (
                float(np.mean([v[0] for v in vals])),
                max(float(np.std([v[0] for v in vals])), 1e-9),
                float(np.mean([v[1] for v in vals])),
                max(float(np.std([v[1] for v in vals])), 1e-9),
            )
            for pos, vals in by_position.items()
        }
        rounds = np.array([h.round_id for h, *_ in raw], dtype=float)
        round_mean, round_sd = float(rounds.mean()), max(float(rounds.std()), 1e-9)
        for hold, position, total, early in raw:
            manager = record.manager_of(hold.team_id)
            if manager is None:
                continue
            tm, ts, em, es = stats[position]
            rows.setdefault(manager, []).append(
                (
                    float(hold.weeks_held),
                    (hold.round_id - round_mean) / round_sd,
                    (total - tm) / ts,
                    (early - em) / es,
                    float(hold.released),
                )
            )
    return rows


def sunk_cost(
    panel: BehavioralPanel, facts: PlayerFacts
) -> tuple[dict[str, TraitEstimate], Shrinkage, dict[str, SunkCostFit]]:
    """Extra weeks a drafted player was held per SD earlier in the draft, production fixed.

    The control is the whole exercise. Early picks are good players and good players get
    held, so regressing hold duration on draft round alone measures the draft, not the
    manager. Two production controls go in -- season-total points and points through week
    four, both standardized within position-season -- and the trait is the *residual*
    round effect, sign-flipped so positive means the manager held early picks longer than
    their production justified.

    **A second confound, which the production control does not touch, decides whether this
    trait may be acted on at all.** `weeks_held` is censored at the end of the season for
    every player who was never dropped, so a manager who drops almost nobody has a nearly
    constant outcome and an OLS round slope pinned toward zero *by arithmetic*, whatever
    his attachment to his early picks. On Blacksburg's real log the correlation between a
    manager's drop rate and his fitted round coefficient is **+0.87** across eleven
    managers -- the ranking is a re-description of who drops players. The manager at the
    bottom of it releases 21% of his own picks, the lowest rate in the league, and was
    being published as "cuts his own draft picks: his early-round busts reach the wire".
    So the correlation is measured, reported in `detail` as `censoring_corr`, and above
    `MAX_CENSORING_CORRELATION` every estimate is marked `confounded` and stops being
    `notable`. The league-wide effect is unaffected and remains readable; what is refused
    is the per-manager ranking. A Tobit or Cox fit is the real answer and is not here.

    Returns the trait table, its shrinkage, and the naive-versus-controlled fits so a
    caller can see how much of the raw effect was confound.
    """
    rows = _hold_frame(panel, facts)
    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    counts: dict[str, int] = {}
    detail: dict[str, dict[str, float]] = {}
    fits: dict[str, SunkCostFit] = {}

    for manager in panel.managers:
        data = rows.get(manager, [])
        counts[manager] = len(data)
        if len(data) < 4:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            detail[manager] = {}
            continue
        arr = np.array(data, dtype=float)
        y, rnd, season_pts, early_pts = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        released_share = float(arr[:, 4].mean())
        if float(y.std()) < 1e-9 or float(rnd.std()) < 1e-9:
            # Nobody was ever dropped, or every pick came from one round. A zero
            # coefficient with a zero standard error would sail through shrinkage
            # untouched and claim to be the most precisely measured manager in the league.
            raw[manager] = float("nan")
            err[manager] = float("nan")
            detail[manager] = {"degenerate": 1.0}
            continue
        naive = ols(np.column_stack([np.ones_like(rnd), rnd]), y)
        controlled = ols(
            np.column_stack([np.ones_like(rnd), rnd, season_pts, early_pts]),
            y,
        )
        raw[manager] = -float(controlled.beta[1])
        err[manager] = float(controlled.stderr[1])
        detail[manager] = {
            "naive_round": -float(naive.beta[1]),
            "naive_se": float(naive.stderr[1]),
            "controlled_round": -float(controlled.beta[1]),
            "controlled_se": float(controlled.stderr[1]),
            "b_season_points": float(controlled.beta[2]),
            "b_early_points": float(controlled.beta[3]),
            "released_share": released_share,
        }
        fits[manager] = SunkCostFit(
            manager=manager,
            n=len(data),
            naive=-float(naive.beta[1]),
            naive_stderr=float(naive.stderr[1]),
            controlled=-float(controlled.beta[1]),
            controlled_stderr=float(controlled.stderr[1]),
            production_beta=(float(controlled.beta[2]), float(controlled.beta[3])),
            released_share=released_share,
        )

    censoring = _censoring_correlation(fits)
    for info in detail.values():
        info["censoring_corr"] = censoring

    table, shrink = empirical_bayes(
        "sunk_cost",
        raw,
        err,
        counts,
        units=TRAIT_UNITS["sunk_cost"],
        note="OLS of weeks held on draft round with season and week-1-4 production controls",
        detail=detail,
    )

    if math.isfinite(censoring) and abs(censoring) >= MAX_CENSORING_CORRELATION:
        message = (
            f"confounded by censoring: the round coefficient correlates {censoring:+.2f} with "
            "how often each manager drops anybody, and a manager who drops nobody has a "
            "constant outcome and a zero slope by arithmetic. League-wide effect stands; "
            "the per-manager ranking is refused"
        )
        log.info("sunk_cost in league %s: %s", panel.league_id, message)
        table = {
            m: replace(e, confounded=True, note=f"{e.note}; {message}" if e.note else message)
            for m, e in table.items()
        }
        shrink = replace(shrink, confound=message)
    return table, shrink, fits


def _censoring_correlation(fits: Mapping[str, SunkCostFit]) -> float:
    """corr(share of drafted players ever released, fitted round coefficient), across managers.

    The measurement that decides whether `sunk_cost` may rank managers. Needs at least
    `MIN_MANAGERS_FOR_SPREAD` managers and real variation in both columns; anything less
    and it returns NaN, which is read as "not measured" rather than "not confounded".
    """
    shares = np.array([f.released_share for f in fits.values()], dtype=float)
    betas = np.array([f.controlled for f in fits.values()], dtype=float)
    keep = np.isfinite(shares) & np.isfinite(betas)
    shares, betas = shares[keep], betas[keep]
    if len(shares) < MIN_MANAGERS_FOR_SPREAD or shares.std() < 1e-9 or betas.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(shares, betas)[0, 1])


# --------------------------------------------------------------------------------------
# Traits: trade behavior
# --------------------------------------------------------------------------------------


def endowment(
    panel: BehavioralPanel, facts: PlayerFacts
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Implied WTA/WTP from what a manager asks for relative to what he offers.

    Only `TRADE_PROPOSAL` records carry their items, so this is the *ask* side and
    nothing else -- see the module docstring for why the accept-side estimator is not
    available from this API. Value is ESPN's own preseason season-total projection, which
    cancels any uniform rescaling and so makes the ratio comparable across weeks.

    `exp(estimate)` is the implied ratio; the experimental literature's ~2x corresponds to
    an estimate near 0.69. A manager who only ever proposes balanced deals scores zero
    whatever his reservation price actually is, so a null here is not evidence against
    endowment -- it is evidence that his proposals were fair.
    """
    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    counts: dict[str, int] = {}
    detail: dict[str, dict[str, float]] = {}
    samples: dict[str, list[float]] = {m: [] for m in panel.managers}

    for record in panel.seasons:
        season = record.season
        for tx in record.transactions:
            if tx.type != "TRADE_PROPOSAL" or tx.is_league_manager:
                continue
            manager = record.manager_of(tx.team_id)
            if manager not in samples:
                continue
            out_value = 0.0
            in_value = 0.0
            for item in tx.items:
                if item.type != "TRADE":
                    continue
                value = facts.season_projection.get((season, item.player_id))
                if value is None or value <= 0:
                    out_value = in_value = 0.0
                    break
                if item.from_team_id == tx.team_id:
                    out_value += value
                elif item.to_team_id == tx.team_id:
                    in_value += value
            if out_value <= 0 or in_value <= 0:
                continue
            samples[manager].append(math.log(in_value / out_value))

    for manager in panel.managers:
        values = samples.get(manager, [])
        counts[manager] = len(values)
        if len(values) < 2:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            detail[manager] = {}
            continue
        arr = np.array(values)
        raw[manager] = float(arr.mean())
        err[manager] = float(arr.std(ddof=1) / math.sqrt(len(arr)))
        detail[manager] = {"implied_wta_wtp": float(math.exp(arr.mean())), "proposals": len(arr)}

    return empirical_bayes(
        "endowment",
        raw,
        err,
        counts,
        units=TRAIT_UNITS["endowment"],
        note="ask side only; accept-side linkage does not survive ESPN's log",
        detail=detail,
    )


def trade_receptiveness(panel: BehavioralPanel) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Accepts over accepts-plus-declines. Whether it is worth writing to this person.

    `TRADE_ACCEPT` and `TRADE_DECLINE` are recorded against the *responding* franchise
    and carry no items, so this is a rate and not a value model. It is also, on a real
    league, the single most useful number here: a manager who has declined eleven of
    twelve proposals is not a negotiation to price, he is a negotiation to skip.

    The standard error is the binomial one with a half-observation continuity floor, so a
    2-for-2 manager does not come back with a standard error of zero and survive
    shrinkage untouched.
    """
    accepts: dict[str, int] = {m: 0 for m in panel.managers}
    declines: dict[str, int] = {m: 0 for m in panel.managers}
    for record in panel.seasons:
        for tx in record.transactions:
            manager = record.manager_of(tx.team_id)
            if manager not in accepts or tx.is_league_manager:
                continue
            if tx.type == "TRADE_ACCEPT":
                accepts[manager] += 1
            elif tx.type == "TRADE_DECLINE":
                declines[manager] += 1

    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    counts: dict[str, int] = {}
    detail: dict[str, dict[str, float]] = {}
    for manager in panel.managers:
        a, d = accepts[manager], declines[manager]
        n = a + d
        counts[manager] = n
        if n == 0:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            detail[manager] = {"accepts": 0.0, "declines": 0.0}
            continue
        # Add-half smoothing: an all-accept or all-decline record is common at these n
        # and its unsmoothed standard error of zero would defeat the shrinkage step.
        p = (a + 0.5) / (n + 1.0)
        raw[manager] = a / n
        err[manager] = math.sqrt(p * (1 - p) / (n + 1.0))
        detail[manager] = {"accepts": float(a), "declines": float(d)}

    return empirical_bayes(
        "trade_receptiveness",
        raw,
        err,
        counts,
        units=TRAIT_UNITS["trade_receptiveness"],
        note="response-side rate; proposals cannot be linked to their resolutions",
        detail=detail,
    )


# --------------------------------------------------------------------------------------
# Traits: draft-board biases
# --------------------------------------------------------------------------------------


def name_brand(
    panel: BehavioralPanel, facts: PlayerFacts
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Preference for players more famous than their projection justifies.

    For each drafted player, take his percentile rank by ESPN projection and subtract his
    percentile rank by ADP, both within position over the season's draftable pool, so
    positive means "the field drafts him earlier than the numbers say". The per-pick
    values are then residualized on overall pick number across the whole league before
    being averaged per manager: a snake draft hands the early slots the famous players by
    construction, and without that step this trait mostly measures where you picked.
    """
    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    counts: dict[str, int] = {}
    samples: dict[str, list[float]] = {m: [] for m in panel.managers}

    for record in panel.seasons:
        season = record.season
        pool = [
            pid
            for (s, pid), value in facts.season_projection.items()
            if s == season and value > 0 and (season, pid) in facts.adp and pid in facts.position
        ]
        if len(pool) < MIN_PROJECTION_POOL:
            log.info(
                "season %s has only %d players with a positive projection and an ADP; "
                "skipped for name_brand",
                season,
                len(pool),
            )
            continue
        gaps: dict[int, float] = {}
        by_position: dict[int, list[int]] = {}
        for pid in sorted(pool):
            by_position.setdefault(facts.position[pid], []).append(pid)
        for players in by_position.values():
            if len(players) < 4:
                continue
            merit = _percentile_rank([-facts.season_projection[(season, p)] for p in players])
            fame = _percentile_rank([facts.adp[(season, p)] for p in players])
            for pid, m, f in zip(players, merit, fame, strict=True):
                gaps[pid] = m - f

        picks = [p for p in record.picks if p.player_id in gaps and not p.auto_drafted]
        if len(picks) < 10:
            continue
        x = np.column_stack(
            [np.ones(len(picks)), np.array([p.overall_pick_number for p in picks], dtype=float)]
        )
        y = np.array([gaps[p.player_id] for p in picks])
        fit = ols(x, y)
        resid = y - x @ fit.beta
        for pick, value in zip(picks, resid, strict=True):
            manager = record.manager_of(pick.team_id)
            if manager in samples:
                samples[manager].append(float(value))

    for manager in panel.managers:
        values = samples.get(manager, [])
        counts[manager] = len(values)
        if len(values) < 3:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            continue
        arr = np.array(values)
        raw[manager] = float(arr.mean())
        err[manager] = float(arr.std(ddof=1) / math.sqrt(len(arr)))

    return empirical_bayes(
        "name_brand",
        raw,
        err,
        counts,
        units=TRAIT_UNITS["name_brand"],
        note="projection percentile minus ADP percentile, residual of overall pick number",
    )


def _percentile_rank(values: Sequence[float]) -> list[float]:
    """Rank in [0, 1], smallest value at 0. Ties get their average rank."""
    arr = np.asarray(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    # Average tied ranks so a block of identical projections does not order arbitrarily.
    for value in np.unique(arr):
        mask = arr == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return list(ranks / max(len(arr) - 1, 1))


def home_team(
    panel: BehavioralPanel,
    facts: PlayerFacts,
    *,
    reps: int = 2000,
    seed: int = 13,
) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Concentration of a roster in one NFL team, against the right null.

    The 1/32 baseline is wrong here and wrong in the direction that invents an effect.
    Sixteen players drawn at random from a pool whose NFL teams are unevenly represented
    -- and they are: a team with four startable skill players contributes four times as
    many candidates as one with a single back -- produce a *maximum* single-team share
    far above 1/32 by chance alone, because the maximum of thirty-two multinomial counts
    is not the mean of them.

    So the null is a permutation of this league's own drafted players across its own
    managers, preserving each manager's pick count, and the trait is the observed modal
    share minus the permutation mean in permutation standard deviations' natural units
    (shares). `detail` carries the permutation p-value and the modal team so the finding
    is directly usable in a trade offer.
    """
    rng = np.random.default_rng(seed)
    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    counts: dict[str, int] = {}
    detail: dict[str, dict[str, float]] = {}
    accumulated: dict[str, list[tuple[float, float, float, int, int]]] = {
        m: [] for m in panel.managers
    }

    for record in panel.seasons:
        season = record.season
        assignments: list[tuple[str, int]] = []
        for pick in sorted(record.picks, key=lambda p: p.overall_pick_number):
            manager = record.manager_of(pick.team_id)
            team = facts.pro_team.get((season, pick.player_id))
            if manager is None or not team:
                continue
            assignments.append((manager, team))
        if len(assignments) < 40:
            continue
        labels, teams = np.unique([t for _, t in assignments], return_inverse=True)
        managers = [m for m, _ in assignments]
        sizes: dict[str, int] = {}
        for m in managers:
            if m in accumulated:
                sizes[m] = sizes.get(m, 0) + 1
        if not sizes:
            continue

        observed: dict[str, tuple[float, int, int]] = {}
        for manager in sizes:
            own = teams[[i for i, m in enumerate(managers) if m == manager]]
            freq = np.bincount(own, minlength=len(labels))
            top = int(freq.argmax())
            observed[manager] = (float(freq[top] / len(own)), int(labels[top]), len(own))

        null: dict[str, list[float]] = {m: [] for m in sizes}
        for _ in range(reps):
            shuffled = rng.permutation(teams)
            cursor = 0
            for manager, size in sizes.items():
                chunk = shuffled[cursor : cursor + size]
                cursor += size
                null[manager].append(float(np.bincount(chunk, minlength=len(labels)).max() / size))

        for manager, (share, team, size) in observed.items():
            draws = np.array(null[manager])
            excess = share - float(draws.mean())
            sd = max(float(draws.std(ddof=1)), 1e-6)
            pvalue = float((draws >= share - 1e-12).mean())
            accumulated[manager].append((excess, sd, pvalue, team, size))

    for manager in panel.managers:
        seasons = accumulated.get(manager, [])
        n = sum(s[4] for s in seasons)
        counts[manager] = n
        if not seasons:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            detail[manager] = {}
            continue
        excess = float(np.mean([s[0] for s in seasons]))
        sd = float(np.sqrt(np.mean([s[1] ** 2 for s in seasons]) / len(seasons)))
        raw[manager] = excess
        err[manager] = max(sd, 1e-6)
        detail[manager] = {
            "modal_team_id": float(seasons[-1][3]),
            "excess_share": float(excess),
            "permutation_p": float(np.mean([s[2] for s in seasons])),
        }

    return empirical_bayes(
        "home_team",
        raw,
        err,
        counts,
        units=TRAIT_UNITS["home_team"],
        note="permutation null over this league's own drafted players, not 1/32",
        detail=detail,
    )


# --------------------------------------------------------------------------------------
# Traits: clock and activity
# --------------------------------------------------------------------------------------


def _eastern(stamp: int) -> datetime:
    return datetime.fromtimestamp(stamp / 1000.0, UTC).astimezone(LEAGUE_TZ)


def latency(panel: BehavioralPanel) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Where in the week's free-agent ordering a manager falls, as a percentile.

    **Free-agent adds only.** An executed waiver claim's `proposedDate` is the batch
    processing time -- measured on Blacksburg 2021/2024/2025, every one of 342 executed
    claims lands in UTC hours 7-9 and a whole week shares one to three distinct
    timestamps -- so including them would report that everybody acts at 3am with an
    impressively small standard error.

    0.0 is always first to the wire in a given week, 0.5 is the null, 1.0 is always last.
    Weeks in which only one manager acted carry no ordering information and are dropped.
    """
    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    counts: dict[str, int] = {}
    detail: dict[str, dict[str, float]] = {}
    samples: dict[str, list[float]] = {m: [] for m in panel.managers}
    hours: dict[str, list[int]] = {m: [] for m in panel.managers}

    for record in panel.seasons:
        weekly: dict[int, dict[str, int]] = {}
        for event in roster_events(record):
            if event.direction != "add" or event.source != "FREEAGENT" or not event.timestamp:
                continue
            manager = record.manager_of(event.team_id)
            if manager not in samples:
                continue
            first = weekly.setdefault(event.week, {})
            if manager not in first or event.timestamp < first[manager]:
                first[manager] = event.timestamp
            hours[manager].append(_eastern(event.timestamp).hour)
        for actors in weekly.values():
            if len(actors) < 2:
                continue
            order = sorted(actors, key=lambda m: actors[m])
            for rank, manager in enumerate(order):
                samples[manager].append(rank / (len(order) - 1))

    for manager in panel.managers:
        values = samples.get(manager, [])
        counts[manager] = len(values)
        clock = hours.get(manager, [])
        detail[manager] = {
            "median_hour_et": float(np.median(clock)) if clock else float("nan"),
            "free_agent_adds": float(len(clock)),
        }
        if len(values) < 2:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            continue
        arr = np.array(values)
        raw[manager] = float(arr.mean())
        err[manager] = max(float(arr.std(ddof=1) / math.sqrt(len(arr))), 1e-6)

    return empirical_bayes(
        "latency",
        raw,
        err,
        counts,
        units=TRAIT_UNITS["latency"],
        note="free-agent adds only; executed waiver timestamps are the batch run",
        detail=detail,
    )


def quiet_hours(panel: BehavioralPanel, *, top: int = 4) -> tuple[tuple[int, int], ...]:
    """(hour, count) for the least-contested hours of the day, Eastern, free agents only.

    The other half of the latency question: the free-agent wire is uncontested when
    nobody is looking at it, and this says when that is for this specific league rather
    than for fantasy football in general.
    """
    counts = dict.fromkeys(range(24), 0)
    for record in panel.seasons:
        for event in roster_events(record):
            if event.direction == "add" and event.source == "FREEAGENT" and event.timestamp:
                counts[_eastern(event.timestamp).hour] += 1
    return tuple(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[:top])


def activity(panel: BehavioralPanel) -> tuple[dict[str, TraitEstimate], Shrinkage]:
    """Log roster moves per week observed, and whether the manager has gone quiet.

    A dormant manager is a different counterparty, not a worse one: he will not outbid
    you on the wire and he will not answer a trade offer, and both of those are worth
    knowing before you spend a waiver priority or an evening drafting a proposal.

    `detail` carries `weeks_quiet` for the focus season and `dormancy_p`, the probability
    of seeing no moves for that long at the manager's own rate. Under 0.05 he has
    plausibly stopped playing.

    **Roster moves are not Poisson and the standard error says so.** They arrive in bursts
    -- an add and its matching drop are two events at one timestamp, and a manager who
    tinkers does three in a sitting -- so the Poisson `1/sqrt(count)` is too small.
    Measured across Blacksburg's manager-seasons the Pearson dispersion is **2.87**
    (median; max 7.6), meaning the honest standard error is about 1.7x the Poisson one and
    the published signal-to-noise falls from 6.19 to 3.55. Every error here is therefore
    scaled by `sqrt(dispersion)`, estimated from the manager's own season-to-season
    counts. The trait survives easily -- splitting Blacksburg's history in half, a
    manager's log move rate in 2021+2023 correlates **0.91** with his rate in 2024+2025,
    which is the only out-of-sample check in this module that any trait passes cleanly --
    but the interval around it was a third too narrow, and the same dispersion is applied
    to `dormancy_p`, where a too-confident p-value tells you to write off a manager who is
    merely between bursts.
    """
    raw: dict[str, float] = {}
    err: dict[str, float] = {}
    counts: dict[str, int] = {}
    detail: dict[str, dict[str, float]] = {}

    moves: dict[str, int] = {m: 0 for m in panel.managers}
    weeks: dict[str, int] = {m: 0 for m in panel.managers}
    #: manager -> [(moves, weeks observed)] per season, for the dispersion estimate.
    segments: dict[str, list[tuple[int, int]]] = {m: [] for m in panel.managers}
    last_seen: dict[str, int] = {}
    focus = panel.focus

    for record in panel.seasons:
        observed = max(record.final_week if record.complete else _latest_week(record), 1)
        here: dict[str, int] = {m: 0 for m in panel.managers}
        for manager in record.managers:
            if manager in weeks:
                weeks[manager] += observed
        for event in roster_events(record):
            if event.source == "DRAFT":
                continue
            manager = record.manager_of(event.team_id)
            if manager in moves:
                moves[manager] += 1
                here[manager] += 1
                if record.season == focus.season:
                    last_seen[manager] = max(last_seen.get(manager, 0), event.week)
        for manager in record.managers:
            if manager in segments:
                segments[manager].append((here[manager], observed))

    focus_weeks = max(_latest_week(focus), 1)
    for manager in panel.managers:
        n = weeks[manager]
        counts[manager] = n
        if n <= 0:
            raw[manager] = float("nan")
            err[manager] = float("nan")
            detail[manager] = {}
            continue
        rate = (moves[manager] + 0.5) / n
        dispersion = _dispersion(segments[manager], rate)
        raw[manager] = math.log(rate)
        # Quasi-Poisson: Var(log rate) ~ dispersion / count, floored so a zero-move
        # manager is not exact.
        err[manager] = math.sqrt(dispersion) / math.sqrt(moves[manager] + 0.5)
        quiet = max(focus_weeks - last_seen.get(manager, 0), 0)
        testable = focus_weeks >= MIN_WEEKS_FOR_DORMANCY
        detail[manager] = {
            "moves": float(moves[manager]),
            "weeks_observed": float(n),
            "moves_per_week": float(rate),
            "dispersion": float(dispersion),
            "weeks_quiet": float(quiet if testable else 0),
            "dormancy_p": float(math.exp(-rate * quiet / dispersion)) if testable else 1.0,
        }

    return empirical_bayes(
        "activity",
        raw,
        err,
        counts,
        units=TRAIT_UNITS["activity"],
        note="adds and drops only; lineup changes are not roster moves",
        detail=detail,
    )


def _latest_week(record: SeasonRecord) -> int:
    weeks = [t.scoring_period_id for t in record.transactions if t.scoring_period_id]
    return max(weeks) if weeks else 1


def _dispersion(segments: Sequence[tuple[int, int]], rate: float) -> float:
    """Pearson dispersion of a manager's season move counts around his own rate.

    Never below 1.0: the correction may widen an interval that a Poisson assumption made
    too narrow, but it must not narrow one, because "less variable than Poisson" from
    three or four seasons is not a finding.
    """
    usable = [(c, w) for c, w in segments if w > 0]
    if len(usable) < 2 or rate <= 0:
        return 1.0
    chi = sum((c - rate * w) ** 2 / (rate * w) for c, w in usable)
    return max(chi / (len(usable) - 1), 1.0)


# --------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManagerProfile:
    """Everything estimable about one league-mate, plus what to do about it."""

    manager: str
    name: str
    league_id: int
    season: int
    team_id: int
    team_name: str
    seasons_observed: int
    moves_observed: int
    #: Share of this manager's picks ESPN made for him. A high value is not a bias, it is
    #: an absence of behavior: there is nothing to model, and it is also the single
    #: cheapest read on how engaged the person is.
    autodraft_share: float
    traits: Mapping[str, TraitEstimate]

    @property
    def estimable(self) -> tuple[str, ...]:
        return tuple(sorted(t for t, e in self.traits.items() if e.estimable))

    @property
    def refused(self) -> tuple[str, ...]:
        return tuple(sorted(t for t, e in self.traits.items() if not e.estimable))

    @property
    def distinct(self) -> tuple[str, ...]:
        """Traits on which this manager is two posterior SDs from his league's mean."""
        return tuple(sorted(t for t, e in self.traits.items() if e.distinct))

    @property
    def notable(self) -> tuple[str, ...]:
        """Traits that are both measured well and far enough out to change a decision."""
        return tuple(sorted(t for t, e in self.traits.items() if e.notable))

    def get(self, trait: str) -> TraitEstimate | None:
        estimate = self.traits.get(trait)
        return estimate if estimate is not None and estimate.estimable else None

    @property
    def counterparty(self) -> str:
        """One word for how to treat an approach from or to this manager.

        **A threshold on a shrunk estimate is not a classification.** The first version of
        this compared `trade_receptiveness` against fixed cutoffs of 0.2 and 0.5. On
        Blacksburg's real log, eight of the nine measured managers had a 95% interval that
        straddled 0.5, so three managers sitting at 0.565, 0.573 and 0.576 came back
        "willing" while one at 0.463 came back "unknown" -- a coin flip published as a
        category. Dropping any single season from the panel moved the "willing" count
        between one and five and the "unresponsive" list between nobody and two people.

        So a label now requires the manager to be `distinct` from his own league, which
        requires in turn (through `notable`'s machinery) that the trait separates anybody
        at all. Everyone else is "unknown", which is the truthful answer: nine trade
        responses spread over four seasons do not classify a person.
        """
        receptive = self.get("trade_receptiveness")
        active = self.get("activity")
        if receptive is not None and receptive.distinct and receptive.separating:
            if receptive.vs_league < 0 and receptive.estimate < 0.5:
                return "unresponsive"
            if receptive.vs_league > 0 and receptive.estimate > 0.5:
                return "willing"
        if active is not None and active.detail.get("dormancy_p", 1.0) < 0.05:
            return "dormant"
        return "unknown"

    def actions(self) -> tuple[str, ...]:
        """Concrete, defensible moves. Empty when nothing is estimable, which is common."""
        out: list[str] = []
        if self.autodraft_share >= 0.5:
            out.append(
                f"autodrafted {self.autodraft_share:.0%} of his picks: not engaged, and his "
                "roster is ESPN's opinion rather than his own"
            )
        if self.counterparty == "unresponsive":
            receptive = self.traits["trade_receptiveness"]
            lo, hi = receptive.ci95
            out.append(
                f"low priority: accepted {receptive.detail.get('accepts', 0):.0f} of "
                f"{receptive.n} trade responses, {receptive.estimate:.0%} shrunk "
                f"[{lo:.0%},{hi:.0%}] against a league mean of {receptive.league_mean:.0%}. "
                "The rate ignores what was offered and does not replicate across seasons "
                "(split-half correlation 0.28 on the only league with a history), so write "
                "to him last, not never"
            )
        if self.counterparty == "dormant":
            quiet = self.traits["activity"].detail.get("weeks_quiet", 0)
            out.append(
                f"dormant for {quiet:.0f} week(s) at his own rate: he will not contest "
                "the wire and will not answer a proposal"
            )
        for trait, above, below in (
            (
                "add_recency",
                "sell him a player coming off a spike week",
                "will not chase a spike; sell him nothing on momentum",
            ),
            (
                "draft_recency",
                "drafts last season's leaderboard: sell him last year's name",
                "drafts the projection, not the name",
            ),
            (
                "sunk_cost",
                "buy his early-round busts cheap; he will not cut them",
                "lets go of his own draft picks sooner than his league does",
            ),
            (
                "name_brand",
                "sell him fame and buy his unglamorous producers",
                "immune to name value; do not price a household name into an offer",
            ),
        ):
            estimate = self.get(trait)
            if estimate is None or not estimate.notable:
                continue
            phrase = above if estimate.vs_league > 0 else below
            suffix = ""
            if trait == "sunk_cost":
                # The coefficient is fitted on a censored outcome, so it is only readable
                # next to how often this manager drops anybody at all. Carrying the number
                # in the sentence is what stops the advice inverting on a manager whose
                # slope is flat because he never releases a player rather than because he
                # is unsentimental about his picks.
                share = estimate.detail.get("released_share", float("nan"))
                suffix = f", releases {share:.0%} of his own picks"
            out.append(f"{phrase} ({trait} {estimate.vs_league:+.2f} vs league{suffix})")
        endow = self.get("endowment")
        if endow is not None and endow.notable and endow.vs_league > 0:
            out.append(
                f"prices his own players {math.exp(endow.estimate):.2f}x what he pays: "
                "open high or let him propose"
            )
        fast = self.get("latency")
        if fast is not None and fast.notable:
            where = "earlier" if fast.vs_league < 0 else "later"
            out.append(
                f"reaches the wire {where} than his league "
                f"({fast.estimate:.2f} vs {fast.league_mean:.2f} mean position)"
            )
        home = self.get("home_team")
        if home is not None and home.notable and home.vs_league > 0:
            team = int(home.detail.get("modal_team_id", 0))
            out.append(f"stacks NFL team {team} beyond chance: offer him that team's players")
        return tuple(out)


@dataclass(frozen=True, slots=True)
class BehavioralReport:
    """One league's behavioral read, with the shrinkage diagnostics it was built from."""

    league_id: int
    season: int
    #: Every season in the panel that ESPN says was played. NOT the same as the seasons
    #: that fed an estimate -- see `seasons_without_football`.
    seasons_used: tuple[int, ...]
    profiles: tuple[ManagerProfile, ...]
    shrinkage: Mapping[str, Shrinkage]
    quiet_hours: tuple[tuple[int, int], ...]
    sunk_cost_fits: Mapping[str, SunkCostFit]
    #: Seasons in `seasons_used` for which the corpus carries no player results at all, so
    #: every estimator that needs football silently scored none of them. On Blacksburg this
    #: is 2021: ESPN serves all 860 of its transactions and 192 of its picks, the corpus
    #: starts at 2022, and `sunk_cost`, `add_recency`, `draft_recency`, `name_brand` and
    #: `home_team` therefore saw *nothing* from it. Reporting "history [2021, 2023, 2024,
    #: 2025, 2026]" while five of nine traits ran on three seasons is the kind of quiet
    #: sample inflation that makes every interval in the file look better than it is.
    seasons_without_football: tuple[int, ...] = ()

    def _spread(self, low: float, high: float) -> tuple[str, ...]:
        return tuple(
            sorted(
                t
                for t, s in self.shrinkage.items()
                if not s.collapsed
                and s.n_managers >= MIN_MANAGERS_FOR_SPREAD
                and low <= s.signal_to_noise < high
            )
        )

    @property
    def usable_traits(self) -> tuple[str, ...]:
        """Traits whose spread clears the measurement error AND carries no known confound.

        A trait with a measured mechanical channel behind its ranking is listed under
        `confounded_traits` instead, however large its signal-to-noise: `sunk_cost` on a
        league with a long history has a spread that clears its error and a ranking that
        is +0.87 correlated with how often each manager drops anybody.
        """
        return tuple(t for t in self._spread(1.0, float("inf")) if not self.shrinkage[t].confound)

    @property
    def marginal_traits(self) -> tuple[str, ...]:
        """Some real spread, but no larger than the error on it. Read, do not act."""
        return tuple(t for t in self._spread(0.0, 1.0) if not self.shrinkage[t].confound)

    @property
    def confounded_traits(self) -> tuple[str, ...]:
        """Traits whose between-manager ranking is a re-description of something else."""
        return tuple(sorted(t for t, s in self.shrinkage.items() if s.confound))

    def separates(self, trait: str) -> bool:
        """Whether `trait` can rank the managers of this league at all."""
        shrink = self.shrinkage.get(trait)
        return shrink is not None and shrink.separates

    @property
    def dead_traits(self) -> tuple[str, ...]:
        """No measurable spread at all: every manager was pulled to the league mean."""
        known = set(self.usable_traits) | set(self.marginal_traits) | set(self.confounded_traits)
        return tuple(sorted(t for t in self.shrinkage if t not in known))

    def by_manager(self, manager: str) -> ManagerProfile | None:
        for p in self.profiles:
            if p.manager == manager:
                return p
        return None

    def counterparties(self) -> dict[str, tuple[ManagerProfile, ...]]:
        """Managers grouped by how to treat an approach: the "who do I even write to" view.

        `unresponsive` and `dormant` are the ones worth acting on first, because they are
        the only classifications here that save you an evening rather than winning you a
        fraction of a point.
        """
        out: dict[str, list[ManagerProfile]] = {}
        for profile in self.profiles:
            out.setdefault(profile.counterparty, []).append(profile)
        return {k: tuple(v) for k, v in out.items()}

    def rank(self, trait: str) -> tuple[ManagerProfile, ...]:
        """Profiles with an estimable value on `trait`, largest first, or `()`.

        Empty when the trait does not separate this league's managers, which is the
        common case: seven of nine traits collapse on real data, and a collapsed trait
        gives every manager the identical shrunk estimate, so sorting them produces
        franchise order dressed up as a ranking. `separates(trait)` is the question this
        answers, and `shrinkage[trait]` says why the answer was no.
        """
        if not self.separates(trait):
            return ()
        have = [p for p in self.profiles if p.get(trait) is not None]
        return tuple(sorted(have, key=lambda p: -p.traits[trait].estimate))

    def summary(self) -> str:
        lines = [
            f"league {self.league_id} season {self.season} "
            f"| history {list(self.seasons_used)} | {len(self.profiles)} managers",
            f"separating traits (spread > error): {list(self.usable_traits) or 'none'}",
            f"marginal (spread <= error):          {list(self.marginal_traits) or 'none'}",
            f"confounded (ranking is something else): {list(self.confounded_traits) or 'none'}",
            f"no measurable spread at all:         {list(self.dead_traits)}",
        ]
        if self.seasons_without_football:
            lines.append(
                f"NO CORPUS RESULTS for {list(self.seasons_without_football)}: those seasons' "
                "transactions and picks fed nothing to sunk_cost, add_recency, draft_recency, "
                "name_brand or home_team, so those traits saw less history than the line above "
                "implies"
            )
        for trait in self.confounded_traits:
            lines.append(f"    {trait}: {self.shrinkage[trait].confound}")
        groups = self.counterparties()
        lines.append(
            "counterparties: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items()))
        )
        for p in self.profiles:
            actions = p.actions()
            lines.append(
                f"  {p.team_name[:26]:26s} {p.name[:16]:16s} "
                f"[{p.counterparty}] seasons={p.seasons_observed} moves={p.moves_observed} "
                f"estimable={len(p.estimable)} notable={list(p.notable)}"
            )
            lines.extend(f"      - {a}" for a in actions)
        return "\n".join(lines)


def behavioral_report(
    panel: BehavioralPanel,
    facts: PlayerFacts | None = None,
    *,
    root: Path | str = corpus.DEFAULT_ROOT,
    variant: str = "ppr",
    permutation_reps: int = 2000,
) -> BehavioralReport:
    """Run every estimator over the panel and assemble the per-manager profiles.

    Traits whose corpus inputs are missing come back refused rather than absent, so the
    output shape does not depend on how much data happened to be there -- a caller can
    always ask for `sunk_cost` and always find out why it is not available.

    A season ESPN serves in full but the corpus has never captured is the quiet version of
    that problem: the estimators skip it silently, one player at a time, and the report
    still says the history was five seasons deep. `seasons_without_football` names them.
    """
    if facts is None:
        facts = load_player_facts(panel.history_seasons, root=root, variant=variant)

    scored = {s for s, _ in facts.season_actual} | {s for s, _ in facts.season_projection}
    missing = tuple(s for s in panel.history_seasons if s not in scored)
    for season in missing:
        log.warning(
            "league %s season %s is played and served by ESPN but the corpus carries no "
            "player results for it; every football-based trait scored none of it",
            panel.league_id,
            season,
        )

    tables: dict[str, dict[str, TraitEstimate]] = {}
    shrinks: dict[str, Shrinkage] = {}

    tables["draft_recency"], shrinks["draft_recency"] = draft_recency(panel, facts)
    tables["add_recency"], shrinks["add_recency"] = add_recency(panel, facts)
    sunk_table, sunk_shrink, sunk_fits = sunk_cost(panel, facts)
    tables["sunk_cost"], shrinks["sunk_cost"] = sunk_table, sunk_shrink
    tables["endowment"], shrinks["endowment"] = endowment(panel, facts)
    tables["trade_receptiveness"], shrinks["trade_receptiveness"] = trade_receptiveness(panel)
    tables["name_brand"], shrinks["name_brand"] = name_brand(panel, facts)
    tables["home_team"], shrinks["home_team"] = home_team(panel, facts, reps=permutation_reps)
    tables["latency"], shrinks["latency"] = latency(panel)
    tables["activity"], shrinks["activity"] = activity(panel)

    focus = panel.focus
    moves: dict[str, int] = {m: 0 for m in panel.managers}
    for record in panel.seasons:
        for event in roster_events(record):
            manager = record.manager_of(event.team_id)
            if manager in moves and event.source != "DRAFT":
                moves[manager] += 1

    team_names = dict(focus.team_names)
    auto: dict[str, tuple[int, int]] = {}
    for pick in focus.picks:
        manager = focus.manager_of(pick.team_id)
        if manager is None:
            continue
        made, total = auto.get(manager, (0, 0))
        auto[manager] = (made + int(bool(pick.auto_drafted)), total + 1)
    profiles: list[ManagerProfile] = []
    for team_id in sorted(focus.owners):
        manager = focus.owners[team_id]
        traits = {
            trait: table.get(manager, _refusal(trait, manager)) for trait, table in tables.items()
        }
        profiles.append(
            ManagerProfile(
                manager=manager,
                name=panel.name(manager),
                league_id=panel.league_id,
                season=panel.focus_season,
                team_id=team_id,
                team_name=team_names.get(team_id, f"Team {team_id}"),
                seasons_observed=len(panel.seasons_for(manager)),
                moves_observed=moves.get(manager, 0),
                autodraft_share=(
                    auto[manager][0] / auto[manager][1] if auto.get(manager, (0, 0))[1] else 0.0
                ),
                traits=traits,
            )
        )

    return BehavioralReport(
        league_id=panel.league_id,
        season=panel.focus_season,
        seasons_used=panel.history_seasons,
        profiles=tuple(profiles),
        shrinkage=shrinks,
        quiet_hours=quiet_hours(panel),
        sunk_cost_fits=sunk_fits,
        seasons_without_football=missing,
    )


def _refusal(trait: str, manager: str) -> TraitEstimate:
    return TraitEstimate(
        trait=trait,
        manager=manager,
        n=0,
        required_n=MIN_N.get(trait, 1),
        estimable=False,
        raw=float("nan"),
        raw_stderr=float("nan"),
        estimate=float("nan"),
        stderr=float("nan"),
        weight=0.0,
        league_mean=float("nan"),
        units=TRAIT_UNITS.get(trait, ""),
        note="no observations for this manager",
    )


__all__ = [
    "LEAGUE_TZ",
    "MAX_CENSORING_CORRELATION",
    "MIN_MANAGERS_FOR_SPREAD",
    "MIN_N",
    "MIN_PROJECTION_POOL",
    "MIN_WEEKS_FOR_DORMANCY",
    "TILT_VARIANCE_PRIOR_DF",
    "TRAIT_UNITS",
    "BehavioralError",
    "BehavioralPanel",
    "BehavioralReport",
    "Fit",
    "Hold",
    "ManagerProfile",
    "PlayerFacts",
    "RosterEvent",
    "SeasonRecord",
    "Shrinkage",
    "SunkCostFit",
    "TraitEstimate",
    "activity",
    "add_recency",
    "behavioral_report",
    "build_panel",
    "conditional_logit",
    "draft_holds",
    "draft_recency",
    "empirical_bayes",
    "endowment",
    "free_agent_adds",
    "home_team",
    "latency",
    "load_player_facts",
    "manager_id",
    "name_brand",
    "ols",
    "quiet_hours",
    "roster_events",
    "sunk_cost",
    "trade_receptiveness",
]
