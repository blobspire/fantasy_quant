"""Typed league state: settings, franchises, rosters, schedule, draft, transactions.

Everything here returns dataclasses. Raw ESPN dicts do not escape this module, because
the payload is a minefield of near-miss field names and sentinel values and we only want
to get each one wrong once.

Verified live on 2026-09-07 against public league 1241838 (2026 and its 2025 archive) and
against ESPN's own `leaguedefaults/3` template. What the checks turned up, in the order
they will bite you:

* **`mRoster` really does honor `scoringPeriodId`.** Community docs claim otherwise. Asking
  the 2025 league for weeks 1/5/12 returns three genuinely different rosters (17/18/19
  entries, different lineup slots). Without the parameter you get whatever week ESPN feels
  like, which quietly poisons any backfill.
* **`mTransactions2` serves ONE scoring period per call.** Not documented anywhere we found.
  Omitting `scoringPeriodId` returns the *current* period only -- and on a finished season
  that is an empty period, so the `transactions` key is absent entirely and a naive reader
  concludes the league had no transactions. Looping periods 0..17 on the 2025 season yields
  442 transactions with 442 distinct ids; there is no paging and no double counting.
* **`playoffMatchupPeriodLength: 0` does not mean "no playoffs."** ESPN's 2026 PPR template
  ships exactly that, plus `variablePlayoffMatchupPeriodLength: true` and
  `playoffMatchupPeriodLengthByRound {"1": 1, "2": 2}` -- a one-week semifinal and a
  two-week final. League 1241838's 2025 archive omits the `ByRound` key completely, so it
  must be `.get()`. `playoffTeamCount` is the flag for whether playoffs exist at all.
* **`matchupPeriodId != scoringPeriodId`,** and one matchup period can span several scoring
  periods -- the same template maps matchup period 16 to scoring periods 16 *and* 17, so
  assuming one week per playoff round drops the second half of the final. Inner lists are
  documented as unsorted, so they are sorted on the way in.
* **`-1` means unlimited** in `moveLimit`, `acquisitionLimit`, `matchupAcquisitionLimit` and
  every `positionLimits` entry. Treating it as a real bound inverts the constraint. And
  `matchupAcquisitionLimit` has a *second* spelling for it: league 1241838 reports `0.0`
  for 2019/2022/2025/2026 while running unrestricted waivers, where ESPN's own
  `leaguedefaults` template reports `-1.0`. A 0 there is "unset", not "no claims allowed".
* **`status.latestScoringPeriod` runs outside the season at both ends.** The finished 2025
  season reports 19 against a `finalScoringPeriod` of 17; a league that has not opened
  reports 0. It is still the right field to read -- never compute the week from a calendar
  -- but it has to be clamped before it is used as a week.
* **An unrecognized `view=` returns HTTP 200** with a default payload that, on a real league,
  still carries `teams`, `members`, `status` *and* `settings` -- so a top-level key check is
  not enough either. Measured: the `teams` in it hold only `id`/`abbrev`/`owners` and the
  `settings` in it holds only `name`. Taken at face value that is ten nameless teams with
  0-0 records, rosters with zero entries, and a league with no slots and no playoffs, all
  from a 200. Every accessor therefore checks a key the *real* view populates and the
  skeleton does not -- `settings.scheduleSettings`, a team's `record`, a team's `roster`.
* **`mPositionalRatings` ignores `scoringPeriodId`** -- week 5 and no-week return identical
  numbers -- so it is a season-to-date aggregate, not a weekly one.

Two places where the live API disagreed with `docs/RESEARCH.md`, reality winning:

* `draftSettings.type` is a **string** (`"SNAKE"`, `"AUCTION"`), not the documented integer
  enum. Both forms are normalized.
* A private-but-real league returns **401 `AUTH_LEAGUE_NOT_VISIBLE`** while a league id that
  does not exist returns **404 `GENERAL_NOT_FOUND`** -- they are distinguishable without
  cookies after all.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .client import EspnClient, EspnError
from .endpoints import league_url
from .statrows import StatRow, parse_rows, rest_of_season_projection, weekly_projections

log = logging.getLogger(__name__)

#: ESPN's "no limit" sentinel. Appears in roster, move and acquisition limits alike.
UNLIMITED = -1

#: lineupSlotIds that are not part of a starting lineup.
BENCH_SLOT = 20
IR_SLOT = 21
INVALID_SLOT = 22
NON_STARTING_SLOTS = frozenset({BENCH_SLOT, IR_SLOT, INVALID_SLOT})

#: lineupSlotId 7 is OP/superflex; slot 0 is QB. Either route to a two-QB league.
SUPERFLEX_SLOT = 7
QB_SLOT = 0

#: lineupSlotIds 8..15 are the IDP block (DT, DE, LB, DL, CB, S, DB, DP).
IDP_SLOTS = frozenset(range(8, 16))

#: defaultPositionIds, which are a *different* id space from lineupSlotIds and collide with
#: it at 4 and 15. pointsOverrides is keyed by these.
POSITION_TE = 4
POSITION_WR = 3
STAT_RECEPTION = 53

#: `draftSettings.type` is documented as this integer enum but arrives as the string name.
DRAFT_TYPES: Mapping[int, str] = {
    0: "OFFLINE",
    1: "SNAKE",
    2: "AUTOPICK",
    3: "SNAIL",
    4: "AUCTION",
    5: "LINEAR",
}

_MEDIAN_SCORING = "WIN_BONUS_TOP_HALF"


def _limit(value: Any, *, zero_is_unset: bool = False) -> int | None:
    """ESPN limit field -> a real bound, or None for unlimited.

    `zero_is_unset` is for the fields where 0 is a second "no limit" spelling rather than a
    cap of zero; see `matchupAcquisitionLimit` in `parse_settings`.
    """
    if value is None:
        return None
    n = int(value)
    if n == UNLIMITED or (zero_is_unset and n == 0):
        return None
    return n


def _int_keyed(raw: Mapping[str, Any] | None) -> dict[int, int]:
    """ESPN hands back JSON-object maps whose keys are stringified ints."""
    return {int(k): int(v) for k, v in (raw or {}).items()}


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Division:
    id: int
    name: str
    size: int


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    """The schedule block, with the playoff structure actually resolved.

    `matchup_periods` is the authoritative matchup-period -> scoring-period map and is the
    only safe way to convert between the two id spaces; the derived helpers below fall back
    to arithmetic only when ESPN omits it.
    """

    matchup_period_count: int
    matchup_period_length: int
    matchup_periods: Mapping[int, tuple[int, ...]]
    playoff_team_count: int
    playoff_seeding_rule: str
    playoff_reseed: bool
    variable_playoff_length: bool
    playoff_matchup_period_length: int
    playoff_length_by_round: Mapping[int, int]
    divisions: tuple[Division, ...]

    @property
    def has_playoffs(self) -> bool:
        """`playoffMatchupPeriodLength == 0` is not the signal; the team count is."""
        return self.playoff_team_count > 1

    @property
    def playoff_round_count(self) -> int:
        """Rounds implied by the bracket size. A 6-team bracket is 3 rounds with byes."""
        if not self.has_playoffs:
            return 0
        rounds, covered = 0, 1
        while covered < self.playoff_team_count:
            covered *= 2
            rounds += 1
        return rounds

    def round_length(self, round_number: int) -> int:
        """Scoring periods in one playoff round.

        With `variablePlayoffMatchupPeriodLength` the per-round map wins; otherwise the flat
        `playoffMatchupPeriodLength` applies to every round. A 0 anywhere in that chain means
        "unset", not "zero weeks", so it falls through to the regular-season length.
        """
        by_round = self.playoff_length_by_round.get(round_number)
        if self.variable_playoff_length and by_round:
            return by_round
        return self.playoff_matchup_period_length or self.matchup_period_length or 1

    @property
    def regular_season_matchup_periods(self) -> tuple[int, ...]:
        return tuple(range(1, self.matchup_period_count + 1))

    @property
    def playoff_matchup_periods(self) -> tuple[int, ...]:
        observed = tuple(
            mp for mp in sorted(self.matchup_periods) if mp > self.matchup_period_count
        )
        if observed:
            return observed
        if not self.has_playoffs:
            return ()
        start = self.matchup_period_count + 1
        return tuple(range(start, start + self.playoff_round_count))

    def scoring_periods(self, matchup_period: int) -> tuple[int, ...]:
        """Scoring periods covered by one matchup period, ascending."""
        mapped = self.matchup_periods.get(matchup_period)
        if mapped:
            return mapped
        # No map from ESPN: lay the periods out end to end, giving each playoff round the
        # length its bracket entry claims.
        cursor = 1
        for mp in range(1, matchup_period + 1):
            if mp <= self.matchup_period_count:
                length = self.matchup_period_length or 1
            else:
                length = self.round_length(mp - self.matchup_period_count)
            if mp == matchup_period:
                return tuple(range(cursor, cursor + length))
            cursor += length
        return ()

    def matchup_period_for(self, scoring_period: int) -> int | None:
        """Inverse of `scoring_periods`. None when the week is outside the schedule."""
        for mp in sorted(self.matchup_periods):
            if scoring_period in self.matchup_periods[mp]:
                return mp
        for mp in (*self.regular_season_matchup_periods, *self.playoff_matchup_periods):
            if scoring_period in self.scoring_periods(mp):
                return mp
        return None

    @property
    def regular_season_weeks(self) -> tuple[int, ...]:
        weeks: set[int] = set()
        for mp in self.regular_season_matchup_periods:
            weeks.update(self.scoring_periods(mp))
        return tuple(sorted(weeks))

    @property
    def playoff_weeks(self) -> tuple[int, ...]:
        """Scoring periods the bracket is played over.

        Valuation is computed twice -- rest-of-season and playoff-weeks-only -- so this is
        load bearing, and it is exactly where a two-week final gets dropped if you assume
        one week per round.
        """
        weeks: set[int] = set()
        for mp in self.playoff_matchup_periods:
            weeks.update(self.scoring_periods(mp))
        return tuple(sorted(weeks))


@dataclass(frozen=True, slots=True)
class RosterConfig:
    lineup_slot_counts: Mapping[int, int]
    position_limits: Mapping[int, int]
    bench_unlimited: bool
    move_limit: int | None
    uses_undroppable_list: bool

    @property
    def starting_slots(self) -> dict[int, int]:
        """Slot -> count, bench/IR/invalid removed. What the lineup solver fills."""
        return {
            slot: n
            for slot, n in self.lineup_slot_counts.items()
            if n > 0 and slot not in NON_STARTING_SLOTS
        }

    @property
    def starter_count(self) -> int:
        return sum(self.starting_slots.values())

    @property
    def bench_slots(self) -> int:
        return self.lineup_slot_counts.get(BENCH_SLOT, 0)

    @property
    def ir_slots(self) -> int:
        return self.lineup_slot_counts.get(IR_SLOT, 0)

    def position_limit(self, position_id: int) -> int | None:
        """Roster cap for a defaultPositionId; None means unlimited."""
        return _limit(self.position_limits.get(position_id))

    @property
    def is_superflex(self) -> bool:
        counts = self.lineup_slot_counts
        return counts.get(SUPERFLEX_SLOT, 0) > 0 or counts.get(QB_SLOT, 0) >= 2

    @property
    def is_idp(self) -> bool:
        return any(self.lineup_slot_counts.get(slot, 0) > 0 for slot in IDP_SLOTS)


@dataclass(frozen=True, slots=True)
class DraftConfig:
    type: str
    keeper_count: int
    keeper_count_future: int
    auction_budget: int
    order_type: str
    pick_order: tuple[int, ...]
    time_per_selection: int

    @property
    def is_auction(self) -> bool:
        return self.type == "AUCTION"

    @property
    def is_keeper(self) -> bool:
        return self.keeper_count > 0 or self.keeper_count_future > 0


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    uses_faab: bool
    budget: int
    minimum_bid: int
    acquisition_limit: int | None
    matchup_acquisition_limit: int | None
    acquisition_type: str
    waiver_hours: int
    waiver_process_days: tuple[str, ...]
    waiver_process_hour: int


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    scoring_type: str
    enhancement_type: str | None
    home_team_bonus: float
    playoff_home_team_bonus: float
    matchup_tie_rule: str | None
    #: Left raw on purpose: the scoring engine owns turning these into a scorer, and
    #: `pointsOverrides` is keyed by defaultPositionId, which is a trap worth keeping in
    #: one place rather than smeared across modules.
    scoring_items: tuple[dict[str, Any], ...]

    @property
    def is_median_scoring(self) -> bool:
        return self.enhancement_type == _MEDIAN_SCORING

    def _item(self, stat_id: int) -> dict[str, Any] | None:
        for item in self.scoring_items:
            if int(item.get("statId", -1)) == stat_id:
                return item
        return None

    def overrides_for(self, stat_id: int) -> dict[int, float]:
        """`pointsOverrides` for one statId, keyed by defaultPositionId.

        The block may be absent rather than empty, and an override *replaces* `points`
        rather than adding to it -- so a position that is *missing* from this map is not
        unscored, it is scored at `points`. Use `points_for` unless you specifically want
        to know which positions were overridden.
        """
        item = self._item(stat_id)
        if item is None:
            return {}
        return {int(k): float(v) for k, v in (item.get("pointsOverrides") or {}).items()}

    def points_for(self, stat_id: int, position_id: int) -> float:
        """Per-unit points for one statId at one defaultPositionId.

        The override replaces `points` for the positions it names; every other position
        keeps `points`. Live payloads carry partial maps as a matter of course -- league
        1241838 sends statId 53 as `points: 0.0` with overrides on positions 1/2/3/4/15
        only -- so "absent from the map" has to mean `points`, never 0.
        """
        item = self._item(stat_id)
        if item is None:
            return 0.0
        overrides = {int(k): float(v) for k, v in (item.get("pointsOverrides") or {}).items()}
        return overrides.get(position_id, float(item.get("points") or 0.0))

    @property
    def te_premium(self) -> float:
        """Extra points per reception a TE gets over a WR. 0.0 in a flat-PPR league.

        Both sides go through `points_for`, so a league that overrides only one of the two
        positions is measured against the other's base `points` rather than against zero.
        Differencing the raw overrides instead reports a phantom premium when TE alone is
        overridden, and misses a real one when WR alone is.
        """
        return self.points_for(STAT_RECEPTION, POSITION_TE) - self.points_for(
            STAT_RECEPTION, POSITION_WR
        )


@dataclass(frozen=True, slots=True)
class LeagueStatus:
    is_active: bool
    is_expired: bool
    current_matchup_period: int
    latest_scoring_period: int
    first_scoring_period: int
    final_scoring_period: int
    transaction_scoring_period: int
    teams_joined: int
    previous_seasons: tuple[int, ...]

    @property
    def current_week(self) -> int:
        """`latestScoringPeriod`, clamped into the season.

        ESPN is the authority on the week -- never derive it from a calendar -- but a
        finished season overshoots (2025 reported 19 against a final period of 17), and a
        league that has not opened reports 0.
        """
        week = self.latest_scoring_period or self.first_scoring_period or 1
        low = max(self.first_scoring_period, 1)
        high = self.final_scoring_period or week
        return max(low, min(week, high))


@dataclass(frozen=True, slots=True)
class LeagueSettings:
    league_id: int
    season: int
    name: str
    size: int
    is_public: bool
    schedule: ScheduleConfig
    roster: RosterConfig
    draft: DraftConfig
    acquisition: AcquisitionConfig
    scoring: ScoringConfig
    status: LeagueStatus

    @property
    def is_redraft(self) -> bool:
        return not self.draft.is_keeper

    @property
    def format_tags(self) -> tuple[str, ...]:
        """Short labels for the format, for logs and the dashboard header."""
        tags = ["keeper" if self.draft.is_keeper else "redraft"]
        tags.append("auction" if self.draft.is_auction else self.draft.type.lower())
        tags.append("faab" if self.acquisition.uses_faab else "waivers")
        if self.roster.is_superflex:
            tags.append("superflex")
        if self.roster.is_idp:
            tags.append("idp")
        if self.scoring.te_premium > 0:
            tags.append("te-premium")
        if self.scoring.is_median_scoring:
            tags.append("median")
        return tuple(tags)


def parse_settings(payload: Mapping[str, Any], league_id: int, season: int) -> LeagueSettings:
    """Build `LeagueSettings` from an `mSettings` payload."""
    settings = payload.get("settings") or {}
    sched = settings.get("scheduleSettings") or {}
    roster = settings.get("rosterSettings") or {}
    draft = settings.get("draftSettings") or {}
    acq = settings.get("acquisitionSettings") or {}
    scoring = settings.get("scoringSettings") or {}
    status = payload.get("status") or {}

    # Inner lists are unsorted per ESPN's own behavior; sort so week ordering is safe to
    # rely on downstream (a two-week final that arrives as [17, 16] otherwise reads as
    # starting in week 17).
    matchup_periods = {
        int(mp): tuple(sorted(int(sp) for sp in periods))
        for mp, periods in (sched.get("matchupPeriods") or {}).items()
    }

    draft_type = draft.get("type")
    if isinstance(draft_type, int):
        draft_type = DRAFT_TYPES.get(draft_type, str(draft_type))

    return LeagueSettings(
        league_id=league_id,
        season=season,
        name=str(settings.get("name") or ""),
        size=int(settings.get("size") or 0),
        is_public=bool(settings.get("isPublic")),
        schedule=ScheduleConfig(
            matchup_period_count=int(sched.get("matchupPeriodCount") or 0),
            matchup_period_length=int(sched.get("matchupPeriodLength") or 1),
            matchup_periods=matchup_periods,
            playoff_team_count=int(sched.get("playoffTeamCount") or 0),
            playoff_seeding_rule=str(sched.get("playoffSeedingRule") or ""),
            playoff_reseed=bool(sched.get("playoffReseed")),
            variable_playoff_length=bool(sched.get("variablePlayoffMatchupPeriodLength")),
            playoff_matchup_period_length=int(sched.get("playoffMatchupPeriodLength") or 0),
            playoff_length_by_round=_int_keyed(sched.get("playoffMatchupPeriodLengthByRound")),
            divisions=tuple(
                Division(
                    id=int(d.get("id", 0)),
                    name=str(d.get("name") or ""),
                    size=int(d.get("size") or 0),
                )
                for d in (sched.get("divisions") or [])
            ),
        ),
        roster=RosterConfig(
            lineup_slot_counts=_int_keyed(roster.get("lineupSlotCounts")),
            position_limits=_int_keyed(roster.get("positionLimits")),
            bench_unlimited=bool(roster.get("isBenchUnlimited")),
            move_limit=_limit(roster.get("moveLimit")),
            uses_undroppable_list=bool(roster.get("isUsingUndroppableList")),
        ),
        draft=DraftConfig(
            type=str(draft_type or "UNKNOWN"),
            keeper_count=int(draft.get("keeperCount") or 0),
            keeper_count_future=int(draft.get("keeperCountFuture") or 0),
            auction_budget=int(draft.get("auctionBudget") or 0),
            order_type=str(draft.get("orderType") or ""),
            pick_order=tuple(int(t) for t in (draft.get("pickOrder") or [])),
            time_per_selection=int(draft.get("timePerSelection") or 0),
        ),
        acquisition=AcquisitionConfig(
            # NOT `acquisitionType` -- that is the processing model (WAIVERS_TRADITIONAL vs
            # continuous), and it stays set to a waiver value in FAAB leagues.
            uses_faab=bool(acq.get("isUsingAcquisitionBudget")),
            budget=int(acq.get("acquisitionBudget") or 0),
            minimum_bid=int(acq.get("minimumBid") or 0),
            acquisition_limit=_limit(acq.get("acquisitionLimit")),
            # `matchupAcquisitionLimit: 0` is a *second* spelling of "no limit", not a cap
            # of zero. Measured on league 1241838 for 2019/2022/2025/2026: all four report
            # 0.0 while running unrestricted waivers (442 transactions in 2025 alone),
            # where ESPN's own leaguedefaults template reports -1.0 for the same setting.
            # Reading the 0 as a bound tells a waiver planner it may make no claims at
            # all, which is the -1 trap wearing a different hat.
            matchup_acquisition_limit=_limit(
                acq.get("matchupAcquisitionLimit"), zero_is_unset=True
            ),
            acquisition_type=str(acq.get("acquisitionType") or ""),
            waiver_hours=int(acq.get("waiverHours") or 0),
            waiver_process_days=tuple(str(d) for d in (acq.get("waiverProcessDays") or [])),
            waiver_process_hour=int(acq.get("waiverProcessHour") or 0),
        ),
        scoring=ScoringConfig(
            scoring_type=str(scoring.get("scoringType") or ""),
            enhancement_type=scoring.get("scoringEnhancementType"),
            home_team_bonus=float(scoring.get("homeTeamBonus") or 0.0),
            playoff_home_team_bonus=float(scoring.get("playoffHomeTeamBonus") or 0.0),
            matchup_tie_rule=scoring.get("matchupTieRule"),
            scoring_items=tuple(scoring.get("scoringItems") or []),
        ),
        status=LeagueStatus(
            is_active=bool(status.get("isActive")),
            is_expired=bool(status.get("isExpired")),
            current_matchup_period=int(status.get("currentMatchupPeriod") or 0),
            latest_scoring_period=int(status.get("latestScoringPeriod") or 0),
            first_scoring_period=int(status.get("firstScoringPeriod") or 0),
            final_scoring_period=int(status.get("finalScoringPeriod") or 0),
            transaction_scoring_period=int(status.get("transactionScoringPeriod") or 0),
            teams_joined=int(status.get("teamsJoined") or 0),
            previous_seasons=tuple(int(s) for s in (status.get("previousSeasons") or [])),
        ),
    )


# --------------------------------------------------------------------------------------
# Franchises
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Record:
    wins: int
    losses: int
    ties: int
    percentage: float
    points_for: float
    points_against: float
    streak_length: int
    streak_type: str
    games_back: float


@dataclass(frozen=True, slots=True)
class Member:
    """A human. `id` is the SWID, which is how a discovered team maps to its owner."""

    id: str
    display_name: str
    first_name: str
    last_name: str


@dataclass(frozen=True, slots=True)
class Team:
    id: int
    name: str
    abbrev: str
    division_id: int
    owners: tuple[str, ...]
    primary_owner: str | None
    logo: str | None
    playoff_seed: int
    waiver_rank: int
    points_for: float
    points_adjusted: float
    record: Record
    acquisition_budget_spent: float
    acquisitions: int
    trades: int
    drops: int
    keeper_player_ids: tuple[int, ...]
    current_projected_rank: int

    @property
    def points_against(self) -> float:
        return self.record.points_against


@dataclass(frozen=True, slots=True)
class LeagueTeams:
    teams: tuple[Team, ...]
    members: tuple[Member, ...]

    def by_id(self, team_id: int) -> Team:
        for t in self.teams:
            if t.id == team_id:
                return t
        raise KeyError(f"no team {team_id} in this league")

    def member(self, member_id: str) -> Member | None:
        key = owner_key(member_id)
        for m in self.members:
            if owner_key(m.id) == key:
                return m
        return None

    def team_for_owner(self, swid: str) -> Team | None:
        """Which franchise a SWID controls. Co-owned teams list every owner."""
        key = owner_key(swid)
        for t in self.teams:
            if any(owner_key(o) == key for o in t.owners):
                return t
        return None


def owner_key(swid: str) -> str:
    """Canonical form for comparing SWIDs.

    ESPN emits them braced and upper-cased, but a value pasted out of a browser or a config
    file may be neither, and a silent mismatch here shows up as "you own no teams".
    """
    return swid.strip().strip("{}").upper()


def _parse_record(raw: Mapping[str, Any] | None) -> Record:
    r = (raw or {}).get("overall") or {}
    return Record(
        wins=int(r.get("wins") or 0),
        losses=int(r.get("losses") or 0),
        ties=int(r.get("ties") or 0),
        percentage=float(r.get("percentage") or 0.0),
        points_for=float(r.get("pointsFor") or 0.0),
        points_against=float(r.get("pointsAgainst") or 0.0),
        streak_length=int(r.get("streakLength") or 0),
        streak_type=str(r.get("streakType") or "NONE"),
        games_back=float(r.get("gamesBack") or 0.0),
    )


def parse_teams(payload: Mapping[str, Any]) -> LeagueTeams:
    """Build `LeagueTeams` from an `mTeam` payload."""
    teams: list[Team] = []
    for t in payload.get("teams") or []:
        counter = t.get("transactionCounter") or {}
        # Seasons before 2019 split the team name into location + nickname; 2019 onward
        # sends a single `name`. Both shapes still turn up when reading league history.
        name = t.get("name") or " ".join(
            part for part in (t.get("location"), t.get("nickname")) if part
        )
        teams.append(
            Team(
                id=int(t.get("id", 0)),
                name=str(name or t.get("abbrev") or f"Team {t.get('id')}"),
                abbrev=str(t.get("abbrev") or ""),
                division_id=int(t.get("divisionId") or 0),
                owners=tuple(str(o) for o in (t.get("owners") or [])),
                primary_owner=t.get("primaryOwner"),
                logo=t.get("logo"),
                playoff_seed=int(t.get("playoffSeed") or 0),
                waiver_rank=int(t.get("waiverRank") or 0),
                points_for=float(t.get("points") or 0.0),
                points_adjusted=float(t.get("pointsAdjusted") or 0.0),
                record=_parse_record(t.get("record")),
                acquisition_budget_spent=float(counter.get("acquisitionBudgetSpent") or 0.0),
                acquisitions=int(counter.get("acquisitions") or 0),
                trades=int(counter.get("trades") or 0),
                drops=int(counter.get("drops") or 0),
                keeper_player_ids=tuple(
                    int(p) for p in ((t.get("draftStrategy") or {}).get("keeperPlayerIds") or [])
                ),
                current_projected_rank=int(t.get("currentProjectedRank") or 0),
            )
        )

    members = tuple(
        Member(
            id=str(m.get("id") or ""),
            display_name=str(m.get("displayName") or ""),
            first_name=str(m.get("firstName") or ""),
            last_name=str(m.get("lastName") or ""),
        )
        for m in (payload.get("members") or [])
    )
    return LeagueTeams(teams=tuple(teams), members=members)


# --------------------------------------------------------------------------------------
# Rosters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RosterEntry:
    player_id: int
    name: str
    lineup_slot_id: int
    default_position_id: int
    pro_team_id: int
    eligible_slots: tuple[int, ...]
    injury_status: str
    injured: bool
    status: str
    acquisition_type: str
    acquisition_date: int | None
    keeper_value: float
    keeper_value_future: float
    percent_owned: float
    percent_started: float
    stats: tuple[StatRow, ...] = field(repr=False, default=())

    @property
    def is_starting(self) -> bool:
        return self.lineup_slot_id not in NON_STARTING_SLOTS

    @property
    def is_defense(self) -> bool:
        """Team defenses carry negative player ids (e.g. -16001)."""
        return self.player_id < 0

    def weekly_projection(self, season: int, week: int) -> float | None:
        return weekly_projections(self.stats, season).get(week)

    def rest_of_season(self, season: int, from_week: int) -> float:
        """Sum of remaining weekly projections.

        Deliberately not ESPN's season-total row: that is frozen at preseason and never
        revised, which is where the folklore "ESPN is 11% optimistic" comes from.
        """
        return rest_of_season_projection(self.stats, season, from_week)


@dataclass(frozen=True, slots=True)
class TeamRoster:
    team_id: int
    scoring_period: int
    entries: tuple[RosterEntry, ...]

    @property
    def starters(self) -> tuple[RosterEntry, ...]:
        return tuple(e for e in self.entries if e.is_starting)

    @property
    def bench(self) -> tuple[RosterEntry, ...]:
        return tuple(e for e in self.entries if e.lineup_slot_id == BENCH_SLOT)

    @property
    def injured_reserve(self) -> tuple[RosterEntry, ...]:
        return tuple(e for e in self.entries if e.lineup_slot_id == IR_SLOT)

    def by_player_id(self, player_id: int) -> RosterEntry | None:
        for e in self.entries:
            if e.player_id == player_id:
                return e
        return None


def _parse_roster_entry(raw: Mapping[str, Any]) -> RosterEntry:
    ppe = raw.get("playerPoolEntry") or {}
    player = ppe.get("player") or {}
    ownership = player.get("ownership") or {}
    return RosterEntry(
        player_id=int(raw.get("playerId") or player.get("id") or 0),
        name=str(player.get("fullName") or ""),
        lineup_slot_id=int(raw.get("lineupSlotId", BENCH_SLOT)),
        default_position_id=int(player.get("defaultPositionId") or 0),
        # The player-level proTeamId is the player's *current* team. For a historical week
        # the stat row's own proTeamId is the one to read, or a traded player gets the
        # wrong bye; both are kept so callers can pick.
        pro_team_id=int(player.get("proTeamId") or 0),
        eligible_slots=tuple(int(s) for s in (player.get("eligibleSlots") or [])),
        injury_status=str(raw.get("injuryStatus") or player.get("injuryStatus") or "NORMAL"),
        injured=bool(player.get("injured")),
        status=str(raw.get("status") or ""),
        acquisition_type=str(raw.get("acquisitionType") or ""),
        acquisition_date=raw.get("acquisitionDate"),
        keeper_value=float(ppe.get("keeperValue") or 0.0),
        keeper_value_future=float(ppe.get("keeperValueFuture") or 0.0),
        percent_owned=float(ownership.get("percentOwned") or 0.0),
        percent_started=float(ownership.get("percentStarted") or 0.0),
        stats=tuple(parse_rows(player.get("stats") or [])),
    )


def parse_rosters(payload: Mapping[str, Any], scoring_period: int) -> dict[int, TeamRoster]:
    """Build team_id -> `TeamRoster` from an `mRoster` payload."""
    out: dict[int, TeamRoster] = {}
    for t in payload.get("teams") or []:
        entries = ((t.get("roster") or {}).get("entries")) or []
        team_id = int(t.get("id", 0))
        out[team_id] = TeamRoster(
            team_id=team_id,
            scoring_period=scoring_period,
            entries=tuple(_parse_roster_entry(e) for e in entries),
        )
    return out


# --------------------------------------------------------------------------------------
# Schedule / matchups
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchupSide:
    team_id: int
    total_points: float
    points_by_scoring_period: Mapping[int, float]
    wins: int
    losses: int
    ties: int
    adjustment: float


@dataclass(frozen=True, slots=True)
class Matchup:
    id: int
    matchup_period_id: int
    winner: str
    playoff_tier_type: str | None
    home: MatchupSide | None
    away: MatchupSide | None

    @property
    def is_bye(self) -> bool:
        """A playoff bye has only one side."""
        return self.home is None or self.away is None

    @property
    def is_playoff(self) -> bool:
        tier = self.playoff_tier_type
        return bool(tier) and tier != "NONE"

    @property
    def is_championship_bracket(self) -> bool:
        return self.playoff_tier_type == "WINNERS_BRACKET"

    @property
    def is_complete(self) -> bool:
        return self.winner not in ("UNDECIDED", "")

    def side_for(self, team_id: int) -> MatchupSide | None:
        for side in (self.home, self.away):
            if side is not None and side.team_id == team_id:
                return side
        return None

    def opponent_of(self, team_id: int) -> int | None:
        if self.home is not None and self.home.team_id == team_id:
            return self.away.team_id if self.away else None
        if self.away is not None and self.away.team_id == team_id:
            return self.home.team_id if self.home else None
        return None


def _parse_side(raw: Mapping[str, Any] | None) -> MatchupSide | None:
    if not raw:
        return None
    cumulative = raw.get("cumulativeScore") or {}
    return MatchupSide(
        team_id=int(raw.get("teamId", 0)),
        total_points=float(raw.get("totalPoints") or 0.0),
        points_by_scoring_period={
            int(k): float(v) for k, v in (raw.get("pointsByScoringPeriod") or {}).items()
        },
        wins=int(cumulative.get("wins") or 0),
        losses=int(cumulative.get("losses") or 0),
        ties=int(cumulative.get("ties") or 0),
        adjustment=float(raw.get("adjustment") or 0.0),
    )


def parse_matchups(payload: Mapping[str, Any]) -> list[Matchup]:
    """Build matchups from an `mMatchupScore` (or `mMatchup`) payload.

    Only `mMatchupScore`/`mBoxscore` carry `playoffTierType`; `mMatchup` omits it entirely,
    which is verified and is why the client asks for the score view.
    """
    out: list[Matchup] = []
    for m in payload.get("schedule") or []:
        out.append(
            Matchup(
                id=int(m.get("id", 0)),
                matchup_period_id=int(m.get("matchupPeriodId") or 0),
                winner=str(m.get("winner") or "UNDECIDED"),
                playoff_tier_type=m.get("playoffTierType"),
                home=_parse_side(m.get("home")),
                away=_parse_side(m.get("away")),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Draft
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DraftPick:
    id: int
    overall_pick_number: int
    round_id: int
    round_pick_number: int
    team_id: int
    player_id: int
    lineup_slot_id: int
    bid_amount: int
    keeper: bool
    reserved_for_keeper: bool
    nominating_team_id: int
    auto_drafted: bool


@dataclass(frozen=True, slots=True)
class DraftDetail:
    drafted: bool
    in_progress: bool
    complete_date: int | None
    picks: tuple[DraftPick, ...]

    def for_team(self, team_id: int) -> tuple[DraftPick, ...]:
        return tuple(p for p in self.picks if p.team_id == team_id)

    @property
    def keepers(self) -> tuple[DraftPick, ...]:
        return tuple(p for p in self.picks if p.keeper)


def parse_draft(payload: Mapping[str, Any]) -> DraftDetail:
    """Build `DraftDetail` from an `mDraftDetail` payload."""
    detail = payload.get("draftDetail") or {}
    picks = tuple(
        DraftPick(
            id=int(p.get("id", 0)),
            overall_pick_number=int(p.get("overallPickNumber") or 0),
            round_id=int(p.get("roundId") or 0),
            round_pick_number=int(p.get("roundPickNumber") or 0),
            team_id=int(p.get("teamId") or 0),
            player_id=int(p.get("playerId") or 0),
            lineup_slot_id=int(p.get("lineupSlotId", BENCH_SLOT)),
            # In an auction this is the winning bid; in a snake it stays 0.
            bid_amount=int(p.get("bidAmount") or 0),
            keeper=bool(p.get("keeper")),
            reserved_for_keeper=bool(p.get("reservedForKeeper")),
            nominating_team_id=int(p.get("nominatingTeamId") or 0),
            auto_drafted=bool(p.get("autoDraftTypeId")),
        )
        for p in (detail.get("picks") or [])
    )
    return DraftDetail(
        drafted=bool(detail.get("drafted")),
        in_progress=bool(detail.get("inProgress")),
        complete_date=detail.get("completeDate"),
        picks=picks,
    )


# --------------------------------------------------------------------------------------
# Positional ratings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpponentRating:
    pro_team_id: int
    average: float
    rank: int


@dataclass(frozen=True, slots=True)
class PositionRating:
    """ESPN's own points-allowed-by-position table, in this league's scoring.

    Season-to-date and *not* week specific: passing `scoringPeriodId` returns byte-identical
    numbers, verified. Useful as a baseline to beat, not as the DvP model -- raw points
    allowed measures schedule rather than defense.
    """

    position_id: int
    average: float
    by_opponent: Mapping[int, OpponentRating]


def parse_positional_ratings(payload: Mapping[str, Any]) -> dict[int, PositionRating]:
    """Build defaultPositionId -> `PositionRating` from an `mPositionalRatings` payload."""
    block = (payload.get("positionAgainstOpponent") or {}).get("positionalRatings") or {}
    out: dict[int, PositionRating] = {}
    for pos, raw in block.items():
        by_opp = {
            int(team): OpponentRating(
                pro_team_id=int(team),
                average=float(v.get("average") or 0.0),
                rank=int(v.get("rank") or 0),
            )
            for team, v in (raw.get("ratingsByOpponent") or {}).items()
        }
        out[int(pos)] = PositionRating(
            position_id=int(pos),
            average=float(raw.get("average") or 0.0),
            by_opponent=by_opp,
        )
    return out


# --------------------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransactionItem:
    type: str
    player_id: int
    from_team_id: int
    to_team_id: int
    from_lineup_slot_id: int
    to_lineup_slot_id: int
    is_keeper: bool
    overall_pick_number: int


@dataclass(frozen=True, slots=True)
class Transaction:
    id: str
    type: str
    status: str
    execution_type: str
    scoring_period_id: int
    team_id: int
    member_id: str | None
    bid_amount: int
    is_pending: bool
    is_league_manager: bool
    proposed_date: int | None
    process_date: int | None
    related_transaction_id: str | None
    items: tuple[TransactionItem, ...]

    @property
    def is_waiver(self) -> bool:
        return self.type == "WAIVER"

    @property
    def is_executed(self) -> bool:
        return self.status == "EXECUTED"

    @property
    def added_player_ids(self) -> tuple[int, ...]:
        return tuple(i.player_id for i in self.items if i.type == "ADD")

    @property
    def dropped_player_ids(self) -> tuple[int, ...]:
        return tuple(i.player_id for i in self.items if i.type == "DROP")


@dataclass(frozen=True, slots=True)
class TransactionLog:
    """The transaction history, plus whether ESPN was willing to serve it.

    `available=False` means the request was refused for want of credentials rather than the
    league genuinely having no transactions -- the distinction matters because the FAAB and
    behavioral-modeling work silently produces garbage from an empty log.
    """

    transactions: tuple[Transaction, ...]
    weeks_covered: tuple[int, ...]
    available: bool = True
    reason: str | None = None

    def __len__(self) -> int:
        return len(self.transactions)

    def __iter__(self) -> Iterator[Transaction]:
        return iter(self.transactions)

    @property
    def waiver_bids(self) -> tuple[tuple[int, int], ...]:
        """(team_id, bid) for every executed waiver claim, for fitting the FAAB bid CDF."""
        return tuple(
            (t.team_id, t.bid_amount) for t in self.transactions if t.is_waiver and t.is_executed
        )


def parse_transactions(payload: Mapping[str, Any]) -> list[Transaction]:
    """Build transactions from one scoring period's `mTransactions2` payload.

    The `transactions` key is *absent*, not empty, on a period with no activity.
    """
    out: list[Transaction] = []
    for t in payload.get("transactions") or []:
        out.append(
            Transaction(
                id=str(t.get("id") or ""),
                type=str(t.get("type") or ""),
                status=str(t.get("status") or ""),
                execution_type=str(t.get("executionType") or ""),
                scoring_period_id=int(t.get("scoringPeriodId") or 0),
                team_id=int(t.get("teamId") or 0),
                member_id=t.get("memberId"),
                bid_amount=int(t.get("bidAmount") or 0),
                is_pending=bool(t.get("isPending")),
                is_league_manager=bool(t.get("isLeagueManager")),
                proposed_date=t.get("proposedDate"),
                process_date=t.get("processDate"),
                related_transaction_id=t.get("relatedTransactionId"),
                items=tuple(
                    TransactionItem(
                        type=str(i.get("type") or ""),
                        player_id=int(i.get("playerId") or 0),
                        from_team_id=int(i.get("fromTeamId") or 0),
                        to_team_id=int(i.get("toTeamId") or 0),
                        from_lineup_slot_id=int(i.get("fromLineupSlotId", UNLIMITED)),
                        to_lineup_slot_id=int(i.get("toLineupSlotId", UNLIMITED)),
                        is_keeper=bool(i.get("isKeeper")),
                        overall_pick_number=int(i.get("overallPickNumber") or 0),
                    )
                    for i in (t.get("items") or [])
                ),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------------------


def _has_marker(block: Any, markers: Sequence[str]) -> bool:
    """Whether `block`, or any dict inside it, carries one of `markers`.

    An empty list passes: a league really can have no teams yet, and that is not the same
    failure as ESPN handing back a skeleton.
    """
    if isinstance(block, Mapping):
        return any(m in block for m in markers)
    if isinstance(block, list | tuple):
        if not block:
            return True
        return any(isinstance(item, Mapping) and any(m in item for m in markers) for item in block)
    return False


def _is_auth_error(err: EspnError) -> bool:
    """Whether an EspnError is "you are not allowed to see this" rather than a real fault.

    The client raises typed-by-message rather than typed-by-class, so this matches on the
    text it produces for 401. A 404 is deliberately NOT treated as an auth failure: it means
    the league id does not exist, which should surface loudly.
    """
    return str(err).startswith("401")


class League:
    """One (league_id, season). Every accessor returns dataclasses.

    Settings are cached because almost everything else needs the week and the schedule map,
    and there is no reason to re-fetch them per call.
    """

    def __init__(self, client: EspnClient, league_id: int, season: int) -> None:
        self._client = client
        self.league_id = league_id
        self.season = season
        self._url = league_url(season, league_id)
        self._settings: LeagueSettings | None = None

    def __repr__(self) -> str:
        return f"League(league_id={self.league_id}, season={self.season})"

    def _view(self, view: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        query: dict[str, Any] = {"view": view}
        if params:
            query.update(params)
        payload, _ = self._client.get(self._url, params=query)
        if not isinstance(payload, dict):
            raise EspnError(f"ESPN returned a non-object body for view={view!r}")
        return payload

    @staticmethod
    def _require(
        payload: Mapping[str, Any],
        key: str,
        view: str,
        *,
        markers: Sequence[str] = (),
    ) -> Any:
        """Assert that the response really is the view we asked for.

        An unrecognized `view=` returns HTTP 200 carrying a generic league payload, and a
        top-level key check is *not* enough to catch it: measured on league 1241838,
        `view=mBogusView` answers 200 with `teams` (10 of them, carrying only `id`,
        `abbrev` and `owners`), `members`, `status` and a `settings` block holding nothing
        but `name`. So `teams`/`settings` being present proves nothing -- without a deeper
        check `rosters()` returns every team with zero entries and `settings()` returns a
        league with no slots, no playoffs and an unclamped week, both without an error.

        `markers` are keys that only the real view populates: at least one has to appear on
        the block, or on one of its elements when the block is a list.
        """
        block = payload.get(key)
        if block is None:
            raise EspnError(
                f"ESPN returned no {key!r} for view={view!r} on league {payload.get('id')}. "
                "An unrecognized view answers 200 with a generic payload, so check the "
                "view name before suspecting the league."
            )
        if markers and not _has_marker(block, markers):
            raise EspnError(
                f"ESPN's {key!r} for view={view!r} on league {payload.get('id')} carries "
                f"none of {list(markers)}, which the real view always populates. This is "
                "the generic 200 payload an unrecognized or retired view returns; taking "
                "it at face value yields silently empty data."
            )
        return block

    # -- settings ----------------------------------------------------------------------

    def settings(self, refresh: bool = False) -> LeagueSettings:
        if self._settings is None or refresh:
            payload = self._view("mSettings")
            # `settings` alone is in the generic payload too; the sub-blocks are not.
            self._require(
                payload,
                "settings",
                "mSettings",
                markers=("scheduleSettings", "rosterSettings", "scoringSettings"),
            )
            self._settings = parse_settings(payload, self.league_id, self.season)
        return self._settings

    def current_week(self) -> int:
        """ESPN's `latestScoringPeriod`, clamped into the season. Never a computed week."""
        return self.settings().status.current_week

    # -- franchises --------------------------------------------------------------------

    def teams(self) -> LeagueTeams:
        payload = self._view("mTeam")
        # A generic payload also has `teams`, but only with id/abbrev/owners on them.
        self._require(payload, "teams", "mTeam", markers=("record", "transactionCounter", "points"))
        return parse_teams(payload)

    # -- rosters -----------------------------------------------------------------------

    def rosters(self, week: int | None = None) -> dict[int, TeamRoster]:
        """Every team's roster for one scoring period.

        `scoringPeriodId` is always sent. It is honored -- verified by pulling weeks 1, 5
        and 12 of a finished season and getting three different rosters -- and omitting it
        returns an unpredictable week.
        """
        week = self.current_week() if week is None else week
        payload = self._view("mRoster", {"scoringPeriodId": week})
        # The decisive one: the generic payload carries `teams` *without* rosters, so a
        # top-level check would hand back every team with zero entries.
        self._require(payload, "teams", "mRoster", markers=("roster",))
        return parse_rosters(payload, week)

    def roster(self, team_id: int, week: int | None = None) -> TeamRoster:
        rosters = self.rosters(week)
        if team_id not in rosters:
            raise KeyError(f"no team {team_id} in league {self.league_id}")
        return rosters[team_id]

    # -- schedule ----------------------------------------------------------------------

    def matchups(self, matchup_period: int | None = None) -> list[Matchup]:
        """The whole schedule, or one matchup period of it.

        Note the argument is a *matchup* period, not a scoring period; use
        `settings().schedule.matchup_period_for(week)` to convert.
        """
        payload = self._view("mMatchupScore")
        self._require(payload, "schedule", "mMatchupScore")
        games = parse_matchups(payload)
        if matchup_period is None:
            return games
        return [m for m in games if m.matchup_period_id == matchup_period]

    def matchups_for_week(self, week: int) -> list[Matchup]:
        """Matchups covering a scoring period, resolved through `matchupPeriods`."""
        mp = self.settings().schedule.matchup_period_for(week)
        if mp is None:
            return []
        return self.matchups(mp)

    # -- draft -------------------------------------------------------------------------

    def draft(self) -> DraftDetail:
        payload = self._view("mDraftDetail")
        self._require(payload, "draftDetail", "mDraftDetail")
        return parse_draft(payload)

    # -- positional ratings ------------------------------------------------------------

    def positional_ratings(self) -> dict[int, PositionRating]:
        payload = self._view("mPositionalRatings")
        self._require(payload, "positionAgainstOpponent", "mPositionalRatings")
        return parse_positional_ratings(payload)

    # -- transactions ------------------------------------------------------------------

    def transactions(self, weeks: Iterable[int] | None = None) -> TransactionLog:
        """The transaction log, one scoring period per request.

        `mTransactions2` returns a single scoring period and nothing else -- omitting
        `scoringPeriodId` gives you the current period, which on a finished season is empty
        and drops the `transactions` key altogether. So this loops. Verified against a
        finished season: 442 transactions over periods 0-17, all ids distinct.

        Auth-gated in private leagues. When ESPN refuses for want of credentials this
        returns `available=False` rather than raising, so a partially-authenticated sync
        still produces everything else.
        """
        if weeks is None:
            weeks = range(0, self.current_week() + 1)
        wanted = tuple(sorted(set(weeks)))

        collected: list[Transaction] = []
        seen: set[str] = set()
        for week in wanted:
            try:
                payload = self._view("mTransactions2", {"scoringPeriodId": week})
            except EspnError as err:
                if _is_auth_error(err):
                    log.warning(
                        "transactions unavailable for league %s: not authenticated",
                        self.league_id,
                    )
                    return TransactionLog(
                        transactions=tuple(collected),
                        weeks_covered=(),
                        available=False,
                        reason="ESPN refused the transaction log; ESPN_SWID/ESPN_S2 needed.",
                    )
                raise
            for tx in parse_transactions(payload):
                if tx.id and tx.id in seen:
                    continue
                if tx.id:
                    seen.add(tx.id)
                collected.append(tx)

        collected.sort(key=lambda t: (t.scoring_period_id, t.proposed_date or 0))
        return TransactionLog(transactions=tuple(collected), weeks_covered=wanted)


def open_league(
    league_id: int,
    season: int,
    swid: str | None = None,
    espn_s2: str | None = None,
) -> League:
    """Convenience constructor that owns its client. Prefer sharing one client across leagues."""
    return League(EspnClient(swid=swid, espn_s2=espn_s2), league_id, season)


def playoff_weeks_for(settings: LeagueSettings) -> tuple[int, ...]:
    """Playoff scoring periods. Free function so callers need no `League` instance."""
    return settings.schedule.playoff_weeks


def summarize(settings: LeagueSettings, teams: Sequence[Team] | None = None) -> str:
    """One-line league description for logs and CLI output."""
    bits = [
        f"{settings.name!r} ({settings.league_id}/{settings.season})",
        f"{settings.size} teams",
        "/".join(settings.format_tags),
        f"week {settings.status.current_week}",
    ]
    if settings.schedule.has_playoffs:
        weeks = settings.schedule.playoff_weeks
        bits.append(f"playoffs wk {weeks[0]}-{weeks[-1]}" if weeks else "playoffs")
    if teams:
        bits.append(f"{len(teams)} franchises loaded")
    return " | ".join(bits)
