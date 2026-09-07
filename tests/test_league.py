"""League parsing, aimed squarely at the ways ESPN's payload lies to you.

The fixtures are trimmed copies of live 2026/2025 responses, so a shape change upstream
shows up here rather than three modules downstream. Everything except the `network` block
runs offline.
"""

from __future__ import annotations

from typing import Any

import pytest

from fantasy_quant.espn.client import EspnError
from fantasy_quant.espn.league import (
    League,
    ScheduleConfig,
    owner_key,
    parse_draft,
    parse_matchups,
    parse_positional_ratings,
    parse_rosters,
    parse_settings,
    parse_teams,
    parse_transactions,
    summarize,
)

PUBLIC_LEAGUE_ID = 1241838

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

# ESPN's own 2026 PPR template (`leaguedefaults/3`), verbatim in the parts that matter: a
# playoff length of 0 alongside a variable per-round map, a matchup period covering two
# scoring periods, and -- re-checked live on 2026-09-07 -- a `firstScoringPeriod` of 0.
VARIABLE_PLAYOFF_SETTINGS: dict[str, Any] = {
    "id": 3,
    "settings": {
        "name": "Default PPR",
        "size": 10,
        "isPublic": True,
        "scheduleSettings": {
            "matchupPeriodCount": 14,
            "matchupPeriodLength": 1,
            # Deliberately shuffled: ESPN does not guarantee the inner lists are ordered,
            # and a reversed two-week final is the case that breaks naive `[0]` indexing.
            "matchupPeriods": {str(mp): [mp] for mp in range(1, 16)} | {"16": [17, 16]},
            "playoffMatchupPeriodLength": 0,
            "playoffMatchupPeriodLengthByRound": {"1": 1, "2": 2},
            "playoffTeamCount": 4,
            "playoffSeedingRule": "TOTAL_POINTS_SCORED",
            "playoffReseed": False,
            "variablePlayoffMatchupPeriodLength": True,
            "divisions": [{"id": 0, "name": "League Standings", "size": 10}],
        },
        "rosterSettings": {
            "isBenchUnlimited": True,
            "isUsingUndroppableList": True,
            "moveLimit": -1,
            "lineupSlotCounts": {
                "0": 1,
                "2": 2,
                "4": 2,
                "6": 1,
                "16": 1,
                "17": 1,
                "20": 7,
                "21": 1,
                "23": 1,
            },
            "positionLimits": {"1": 4, "2": 8, "3": 8, "4": 3, "5": 3, "6": -1, "16": 3},
        },
        "draftSettings": {
            "type": "SNAKE",
            "keeperCount": 0,
            "keeperCountFuture": 0,
            "auctionBudget": 200,
            "orderType": "DRAFT_START",
            "pickOrder": [1, 2, 3],
            "timePerSelection": 90,
        },
        "acquisitionSettings": {
            # FAAB is `isUsingAcquisitionBudget`. `acquisitionType` is the processing model
            # and stays on a waiver value even in a FAAB league.
            "isUsingAcquisitionBudget": False,
            "acquisitionType": "WAIVERS_TRADITIONAL",
            "acquisitionBudget": 100,
            "acquisitionLimit": -1,
            "matchupAcquisitionLimit": -1.0,
            "minimumBid": 1,
            "waiverHours": 24,
            "waiverProcessDays": ["WEDNESDAY"],
            "waiverProcessHour": 11,
        },
        "scoringSettings": {
            "scoringType": "H2H_POINTS",
            "homeTeamBonus": 0.0,
            "playoffHomeTeamBonus": 0.0,
            "scoringItems": [
                # TE premium: keys are defaultPositionId (4 = TE, 3 = WR), NOT lineupSlotId
                # (where 4 is WR). Getting this backwards is the classic silent mis-score.
                {"statId": 53, "points": 0.0, "pointsOverrides": {"3": 1.0, "4": 1.5}},
                {"statId": 42, "points": 0.1},
            ],
        },
    },
    "status": {
        "isActive": False,
        "currentMatchupPeriod": 1,
        "latestScoringPeriod": 0,
        # 0, not 1: the template has no first period either, so the clamp cannot lean on it.
        "firstScoringPeriod": 0,
        "finalScoringPeriod": 17,
        "transactionScoringPeriod": 0,
        "teamsJoined": 0,
        "previousSeasons": [],
    },
}

# League 1241838's 2025 archive in its schedule, draft, acquisition and status blocks --
# flat playoff length, `ByRound` absent entirely, `latestScoringPeriod` past the end of the
# season -- but NOT a verbatim copy: `scoringEnhancementType` and `moveLimit` are set here
# to exercise median scoring and a real move cap, neither of which that league uses (live
# it sends no enhancement type and `moveLimit: -1`).
FLAT_PLAYOFF_SETTINGS: dict[str, Any] = {
    "settings": {
        "name": "The Keeper League",
        "size": 10,
        "isPublic": True,
        "scheduleSettings": {
            "matchupPeriodCount": 14,
            "matchupPeriodLength": 1,
            "matchupPeriods": {str(mp): [mp] for mp in range(1, 18)},
            "playoffMatchupPeriodLength": 1,
            "playoffTeamCount": 6,
            "playoffSeedingRule": "H2H_RECORD",
            "variablePlayoffMatchupPeriodLength": False,
            "divisions": [
                {"id": 0, "name": "East", "size": 5},
                {"id": 1, "name": "West", "size": 5},
            ],
        },
        "rosterSettings": {
            "lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1, "20": 7, "21": 2, "23": 1},
            "positionLimits": {"1": -1},
            "moveLimit": 40,
        },
        "draftSettings": {"type": "AUCTION", "keeperCount": 2, "keeperCountFuture": 2},
        "acquisitionSettings": {
            "isUsingAcquisitionBudget": True,
            "acquisitionType": "WAIVERS_TRADITIONAL",
            "acquisitionBudget": 100,
            "minimumBid": 0,
        },
        "scoringSettings": {
            "scoringType": "H2H_POINTS",
            "scoringEnhancementType": "WIN_BONUS_TOP_HALF",
            # No pointsOverrides key at all on this item -- absent, not empty.
            "scoringItems": [{"statId": 53, "points": 1.0}],
        },
    },
    "status": {
        "isActive": True,
        "currentMatchupPeriod": 17,
        "latestScoringPeriod": 19,
        "firstScoringPeriod": 1,
        "finalScoringPeriod": 17,
        "transactionScoringPeriod": 19,
        "teamsJoined": 10,
        "previousSeasons": [2024, 2025],
    },
}

TEAM_PAYLOAD: dict[str, Any] = {
    "teams": [
        {
            "id": 1,
            "name": "Team Binish",
            "abbrev": "BINI",
            "divisionId": 0,
            "owners": ["{3A2C2DB2-7702-429A-8C0E-BC6C84DAA2EF}"],
            "primaryOwner": "{3A2C2DB2-7702-429A-8C0E-BC6C84DAA2EF}",
            "playoffSeed": 3,
            "waiverRank": 9,
            "points": 1617.58,
            "record": {
                "overall": {
                    "wins": 7,
                    "losses": 7,
                    "ties": 0,
                    "percentage": 0.5,
                    "pointsFor": 1617.58,
                    "pointsAgainst": 1666.66,
                    "streakLength": 1,
                    "streakType": "WIN",
                    "gamesBack": 3.0,
                }
            },
            "transactionCounter": {"acquisitionBudgetSpent": 61, "acquisitions": 22, "trades": 1},
            "draftStrategy": {"keeperPlayerIds": [4685382]},
        },
        # Pre-2019 shape: the name is split across location + nickname.
        {"id": 2, "location": "Gridiron", "nickname": "Goons", "abbrev": "GG", "owners": []},
    ],
    "members": [
        {
            "id": "{3A2C2DB2-7702-429A-8C0E-BC6C84DAA2EF}",
            "displayName": "justlikepudge",
            "firstName": "Austin",
            "lastName": "Binish",
        }
    ],
}

ROSTER_PAYLOAD: dict[str, Any] = {
    "teams": [
        {
            "id": 1,
            "roster": {
                "entries": [
                    {
                        "playerId": 4685382,
                        "lineupSlotId": 2,
                        "acquisitionType": "DRAFT",
                        "acquisitionDate": 1788144318384,
                        "injuryStatus": "NORMAL",
                        "status": "NORMAL",
                        "playerPoolEntry": {
                            "keeperValue": 28.0,
                            "player": {
                                "id": 4685382,
                                "fullName": "Omarion Hampton",
                                "defaultPositionId": 2,
                                "proTeamId": 24,
                                "eligibleSlots": [2, 3, 23, 7, 20, 21],
                                "injured": False,
                                "ownership": {"percentOwned": 99.1, "percentStarted": 88.0},
                                "stats": [
                                    # Frozen preseason season total -- never revised, and
                                    # exactly what must NOT be used as the forecast.
                                    {
                                        "id": "102026",
                                        "seasonId": 2026,
                                        "statSourceId": 1,
                                        "statSplitTypeId": 0,
                                        "scoringPeriodId": 0,
                                        "appliedTotal": 999.0,
                                        "stats": {},
                                    },
                                    {
                                        "id": "1120261",
                                        "seasonId": 2026,
                                        "statSourceId": 1,
                                        "statSplitTypeId": 1,
                                        "scoringPeriodId": 1,
                                        "appliedTotal": 16.0,
                                        "stats": {},
                                    },
                                    {
                                        "id": "1120262",
                                        "seasonId": 2026,
                                        "statSourceId": 1,
                                        "statSplitTypeId": 1,
                                        "scoringPeriodId": 2,
                                        "appliedTotal": 14.0,
                                        "stats": {},
                                    },
                                    # A 2025 row rides along in the same array.
                                    {
                                        "id": "1120251",
                                        "seasonId": 2025,
                                        "statSourceId": 1,
                                        "statSplitTypeId": 1,
                                        "scoringPeriodId": 1,
                                        "appliedTotal": 500.0,
                                        "stats": {},
                                    },
                                ],
                            },
                        },
                    },
                    {
                        "playerId": -16001,
                        "lineupSlotId": 16,
                        "playerPoolEntry": {
                            "player": {"fullName": "Jaguars D/ST", "defaultPositionId": 16}
                        },
                    },
                    {
                        "playerId": 111,
                        "lineupSlotId": 20,
                        "playerPoolEntry": {"player": {"fullName": "Benchy"}},
                    },
                    {
                        "playerId": 222,
                        "lineupSlotId": 21,
                        "playerPoolEntry": {"player": {"fullName": "Hurty"}},
                    },
                ]
            },
        }
    ]
}

MATCHUP_PAYLOAD: dict[str, Any] = {
    "schedule": [
        {
            "id": 1,
            "matchupPeriodId": 1,
            "winner": "HOME",
            "playoffTierType": "NONE",
            "home": {
                "teamId": 1,
                "totalPoints": 120.5,
                "pointsByScoringPeriod": {"1": 120.5},
                "cumulativeScore": {"wins": 1, "losses": 0, "ties": 0},
            },
            "away": {
                "teamId": 2,
                "totalPoints": 99.0,
                "pointsByScoringPeriod": {"1": 99.0},
                "cumulativeScore": {"wins": 0, "losses": 1, "ties": 0},
            },
        },
        # A playoff bye: one side only.
        {
            "id": 2,
            "matchupPeriodId": 15,
            "winner": "UNDECIDED",
            "playoffTierType": "WINNERS_BRACKET",
            "home": {"teamId": 1, "totalPoints": 0.0},
        },
        {
            "id": 3,
            "matchupPeriodId": 15,
            "winner": "AWAY",
            "playoffTierType": "LOSERS_CONSOLATION_LADDER",
            "home": {"teamId": 5, "totalPoints": 80.0},
            "away": {"teamId": 6, "totalPoints": 91.0},
        },
    ]
}


# What ESPN actually answers for an unrecognized `view=`, captured verbatim from
# `GET .../leagues/1241838?view=mBogusView` on 2026-09-07 (teams/members trimmed to two).
# Note what survives: `settings`, `teams`, `members` and `status` are all present, so a
# top-level key check passes -- but `settings` holds only `name`, the teams hold only
# id/abbrev/owners, and there is no `roster` anywhere.
GENERIC_200_PAYLOAD: dict[str, Any] = {
    "gameId": 1,
    "id": 1241838,
    "members": [{"displayName": "justlikepudge", "id": "{3A2C2DB2}", "isLeagueManager": False}],
    "scoringPeriodId": 19,
    "seasonId": 2025,
    "segmentId": 0,
    "settings": {"name": "The Keeper League"},
    "status": {"currentMatchupPeriod": 17, "isActive": True, "latestScoringPeriod": 19},
    "teams": [
        {"abbrev": "BINI", "id": 1, "owners": ["{3A2C2DB2}"]},
        {"abbrev": "16Ks", "id": 2, "owners": ["{F17564F5}"]},
    ],
}


def _fake_client(payloads: dict[str, Any], calls: list[dict[str, Any]] | None = None) -> Any:
    """Minimal stand-in for EspnClient keyed on the `view` query parameter."""

    class _Fake:
        def get(self, url, params=None, fantasy_filter=None, use_etag=False):  # noqa: ANN001
            params = dict(params or {})
            if calls is not None:
                calls.append(params)
            value = payloads[params["view"]]
            if isinstance(value, Exception):
                raise value
            if callable(value):
                return value(params), {}
            return value, {}

    return _Fake()


# --------------------------------------------------------------------------------------
# Settings: playoff structure
# --------------------------------------------------------------------------------------


def test_playoff_length_zero_still_means_playoffs():
    """`playoffMatchupPeriodLength: 0` is not "no playoffs" -- the team count decides."""
    s = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    assert s.schedule.playoff_matchup_period_length == 0
    assert s.schedule.has_playoffs
    assert s.schedule.playoff_round_count == 2


def test_variable_playoff_lengths_are_read_per_round():
    s = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    sched = s.schedule
    assert sched.round_length(1) == 1
    assert sched.round_length(2) == 2
    assert sched.playoff_matchup_periods == (15, 16)
    # The two-week final contributes both weeks. Assuming one week per round loses week 17,
    # which is the week the money is decided in.
    assert sched.playoff_weeks == (15, 16, 17)


def test_matchup_period_inner_lists_are_sorted():
    """ESPN does not order them; a reversed final would otherwise read as starting late."""
    s = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    assert VARIABLE_PLAYOFF_SETTINGS["settings"]["scheduleSettings"]["matchupPeriods"]["16"] == [
        17,
        16,
    ]
    assert s.schedule.scoring_periods(16) == (16, 17)


def test_matchup_period_is_not_the_scoring_period():
    s = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    assert s.schedule.matchup_period_for(17) == 16
    assert s.schedule.matchup_period_for(16) == 16
    assert s.schedule.matchup_period_for(99) is None


def test_flat_playoff_length_without_by_round_block():
    """`playoffMatchupPeriodLengthByRound` is absent, not empty, in a fixed-length league."""
    s = parse_settings(FLAT_PLAYOFF_SETTINGS, PUBLIC_LEAGUE_ID, 2025)
    sched = s.schedule
    assert sched.playoff_length_by_round == {}
    assert sched.playoff_round_count == 3  # 6 teams -> 3 rounds with byes
    assert all(sched.round_length(r) == 1 for r in (1, 2, 3))
    assert sched.playoff_matchup_periods == (15, 16, 17)
    assert sched.playoff_weeks == (15, 16, 17)


def test_variable_flag_set_but_by_round_missing_does_not_explode():
    """The half-configured case: trust the flag, find no map, fall back rather than KeyError."""
    sched = ScheduleConfig(
        matchup_period_count=13,
        matchup_period_length=1,
        matchup_periods={},
        playoff_team_count=4,
        playoff_seeding_rule="H2H_RECORD",
        playoff_reseed=False,
        variable_playoff_length=True,
        playoff_matchup_period_length=0,
        playoff_length_by_round={},
        divisions=(),
    )
    assert sched.round_length(1) == 1
    assert sched.playoff_matchup_periods == (14, 15)
    # No matchupPeriods map at all: periods are laid out end to end instead.
    assert sched.scoring_periods(14) == (14,)
    assert sched.playoff_weeks == (14, 15)


def test_no_playoffs_when_team_count_is_zero():
    sched = ScheduleConfig(
        matchup_period_count=14,
        matchup_period_length=1,
        matchup_periods={1: (1,)},
        playoff_team_count=0,
        playoff_seeding_rule="",
        playoff_reseed=False,
        variable_playoff_length=False,
        playoff_matchup_period_length=1,
        playoff_length_by_round={},
        divisions=(),
    )
    assert not sched.has_playoffs
    assert sched.playoff_round_count == 0
    assert sched.playoff_matchup_periods == ()
    assert sched.playoff_weeks == ()


# --------------------------------------------------------------------------------------
# Settings: sentinels, flags and missing blocks
# --------------------------------------------------------------------------------------


def test_minus_one_means_unlimited():
    s = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    assert s.roster.move_limit is None
    assert s.acquisition.acquisition_limit is None
    assert s.acquisition.matchup_acquisition_limit is None
    assert s.roster.position_limit(6) is None  # -1 in the payload
    assert s.roster.position_limit(4) == 3  # a real cap survives
    # ...and a real limit is not swallowed by the sentinel handling.
    assert parse_settings(FLAT_PLAYOFF_SETTINGS, 1, 2025).roster.move_limit == 40


def test_latest_scoring_period_is_clamped_both_ways():
    """It overshoots on a finished season and reads 0 before a league opens."""
    finished = parse_settings(FLAT_PLAYOFF_SETTINGS, 1, 2025).status
    assert finished.latest_scoring_period == 19
    assert finished.current_week == 17

    unopened = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026).status
    assert unopened.latest_scoring_period == 0
    assert unopened.current_week == 1


def test_missing_optional_blocks_parse_to_empty_rather_than_raising():
    s = parse_settings({"settings": {}, "status": {}}, 42, 2026)
    assert s.name == ""
    assert s.schedule.matchup_periods == {}
    assert not s.schedule.has_playoffs
    assert s.roster.lineup_slot_counts == {}
    assert s.roster.starter_count == 0
    assert s.scoring.scoring_items == ()
    assert s.draft.type == "UNKNOWN"
    assert s.status.current_week == 1


def test_completely_empty_payload_parses():
    s = parse_settings({}, 0, 2026)
    assert s.size == 0
    assert s.acquisition.uses_faab is False


def test_faab_is_the_budget_flag_not_the_acquisition_type():
    """`acquisitionType` stays WAIVERS_TRADITIONAL in a FAAB league; only the flag moves."""
    non_faab = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    faab = parse_settings(FLAT_PLAYOFF_SETTINGS, 1, 2025)
    assert non_faab.acquisition.acquisition_type == faab.acquisition.acquisition_type
    assert non_faab.acquisition.uses_faab is False
    assert faab.acquisition.uses_faab is True


def test_te_premium_reads_position_ids_not_slot_ids():
    """pointsOverrides keys are defaultPositionId. 4 is TE there and WR in the slot space."""
    s = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    assert s.scoring.overrides_for(53) == {3: 1.0, 4: 1.5}
    assert s.scoring.te_premium == pytest.approx(0.5)
    assert "te-premium" in s.format_tags


def test_te_premium_measures_against_base_points_not_against_zero():
    """An override *replaces* `points`; a position missing from the map keeps `points`.

    Live payloads carry partial maps as a rule -- league 1241838 overrides positions
    1/2/3/4/15 on statId 53 and leaves the rest on `points`. Differencing the raw override
    map instead invents a premium when only TE is listed and misses one when only WR is,
    and disagrees with `scoring.py`, which is the module that actually scores the league.
    """

    def premium(item: dict[str, Any]) -> float:
        return parse_settings(
            {"settings": {"scoringSettings": {"scoringItems": [item]}}}, 1, 2026
        ).scoring.te_premium

    # TE overridden, WR left on the 1.0 base: a 0.5 premium, not a 1.5 one.
    assert premium({"statId": 53, "points": 1.0, "pointsOverrides": {"4": 1.5}}) == pytest.approx(
        0.5
    )
    # WR overridden downward, TE left on the base: still a 0.5 premium, not none at all.
    assert premium({"statId": 53, "points": 1.5, "pointsOverrides": {"3": 1.0}}) == pytest.approx(
        0.5
    )
    # And `points_for` agrees with scoring.py on both sides.
    scoring = parse_settings(
        {"settings": {"scoringSettings": {"scoringItems": [{"statId": 53, "points": 1.0}]}}},
        1,
        2026,
    ).scoring
    assert scoring.points_for(53, 4) == pytest.approx(1.0)
    assert scoring.points_for(53, 16) == pytest.approx(1.0)  # not in the map == base points
    assert scoring.points_for(9999, 4) == 0.0  # a stat the league does not score


def test_matchup_acquisition_limit_of_zero_is_not_a_cap_of_zero():
    """Measured: league 1241838 reports 0.0 for 2019/2022/2025/2026 with unlimited waivers.

    ESPN's own leaguedefaults template writes -1.0 for the same "no limit", so both
    spellings are in the wild. Reading the 0 as a bound tells the waiver planner it may
    make no claims at all.
    """
    payload = {
        "settings": {
            "acquisitionSettings": {"matchupAcquisitionLimit": 0.0, "acquisitionLimit": 3}
        },
        "status": {},
    }
    acq = parse_settings(payload, 1, 2025).acquisition
    assert acq.matchup_acquisition_limit is None
    assert acq.acquisition_limit == 3  # a real cap elsewhere is untouched


def test_absent_points_overrides_is_not_an_error():
    s = parse_settings(FLAT_PLAYOFF_SETTINGS, 1, 2025)
    assert s.scoring.overrides_for(53) == {}
    assert s.scoring.te_premium == 0.0
    assert s.scoring.overrides_for(9999) == {}


def test_median_scoring_detection():
    assert parse_settings(FLAT_PLAYOFF_SETTINGS, 1, 2025).scoring.is_median_scoring
    assert not parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026).scoring.is_median_scoring


def test_draft_type_accepts_the_documented_int_and_the_real_string():
    """RESEARCH.md documents an int enum; the live API sends the string name."""
    as_string = parse_settings(FLAT_PLAYOFF_SETTINGS, 1, 2025)
    assert as_string.draft.type == "AUCTION"
    assert as_string.draft.is_auction

    payload = {"settings": {"draftSettings": {"type": 4}}, "status": {}}
    assert parse_settings(payload, 1, 2025).draft.is_auction


def test_keeper_and_redraft_detection():
    assert parse_settings(FLAT_PLAYOFF_SETTINGS, 1, 2025).draft.is_keeper
    redraft = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    assert redraft.is_redraft
    assert "redraft" in redraft.format_tags


def test_superflex_and_idp_read_the_slot_id_space():
    base = {"settings": {"rosterSettings": {"lineupSlotCounts": {}}}, "status": {}}

    via_op = {"settings": {"rosterSettings": {"lineupSlotCounts": {"0": 1, "7": 1}}}}
    via_two_qbs = {"settings": {"rosterSettings": {"lineupSlotCounts": {"0": 2}}}}
    idp = {"settings": {"rosterSettings": {"lineupSlotCounts": {"0": 1, "11": 3}}}}

    assert not parse_settings(base, 1, 2026).roster.is_superflex
    assert parse_settings(via_op, 1, 2026).roster.is_superflex
    assert parse_settings(via_two_qbs, 1, 2026).roster.is_superflex
    assert parse_settings(idp, 1, 2026).roster.is_idp
    assert not parse_settings(via_op, 1, 2026).roster.is_idp


def test_starting_slots_exclude_bench_ir_and_invalid():
    s = parse_settings(VARIABLE_PLAYOFF_SETTINGS, 3, 2026)
    assert 20 not in s.roster.starting_slots
    assert 21 not in s.roster.starting_slots
    assert s.roster.starter_count == 9  # 1QB 2RB 2WR 1TE 1FLEX 1DST 1K
    assert s.roster.bench_slots == 7
    assert s.roster.ir_slots == 1


def test_summarize_is_stable_enough_to_log():
    line = summarize(parse_settings(FLAT_PLAYOFF_SETTINGS, PUBLIC_LEAGUE_ID, 2025))
    assert "The Keeper League" in line
    assert "keeper" in line
    assert "faab" in line


# --------------------------------------------------------------------------------------
# Teams
# --------------------------------------------------------------------------------------


def test_team_name_falls_back_to_the_pre_2019_split_fields():
    teams = parse_teams(TEAM_PAYLOAD)
    assert teams.by_id(1).name == "Team Binish"
    assert teams.by_id(2).name == "Gridiron Goons"


def test_owner_lookup_tolerates_missing_braces_and_case():
    teams = parse_teams(TEAM_PAYLOAD)
    swid = "{3A2C2DB2-7702-429A-8C0E-BC6C84DAA2EF}"
    assert teams.team_for_owner(swid).id == 1
    assert teams.team_for_owner(swid.strip("{}").lower()).id == 1
    assert teams.team_for_owner("{00000000-0000-0000-0000-000000000000}") is None
    assert owner_key(" {abc} ") == "ABC"


def test_team_record_and_counters():
    team = parse_teams(TEAM_PAYLOAD).by_id(1)
    assert team.record.wins == 7
    assert team.points_against == pytest.approx(1666.66)
    assert team.acquisition_budget_spent == 61
    assert team.keeper_player_ids == (4685382,)
    assert parse_teams(TEAM_PAYLOAD).member(team.owners[0]).display_name == "justlikepudge"


def test_teams_by_id_raises_for_an_unknown_team():
    with pytest.raises(KeyError):
        parse_teams(TEAM_PAYLOAD).by_id(99)


# --------------------------------------------------------------------------------------
# Rosters
# --------------------------------------------------------------------------------------


def test_roster_partitions_starters_bench_and_ir():
    roster = parse_rosters(ROSTER_PAYLOAD, 5)[1]
    assert roster.scoring_period == 5
    assert {e.player_id for e in roster.starters} == {4685382, -16001}
    assert [e.player_id for e in roster.bench] == [111]
    assert [e.player_id for e in roster.injured_reserve] == [222]


def test_team_defenses_carry_negative_player_ids():
    roster = parse_rosters(ROSTER_PAYLOAD, 1)[1]
    assert roster.by_player_id(-16001).is_defense
    assert not roster.by_player_id(4685382).is_defense


def test_rest_of_season_sums_weeklies_and_ignores_the_frozen_total():
    """The 102026 row says 999; the real remaining projection is 16 + 14."""
    entry = parse_rosters(ROSTER_PAYLOAD, 1)[1].by_player_id(4685382)
    assert entry.rest_of_season(2026, from_week=1) == pytest.approx(30.0)
    assert entry.rest_of_season(2026, from_week=2) == pytest.approx(14.0)
    assert entry.weekly_projection(2026, 1) == pytest.approx(16.0)


def test_other_seasons_do_not_leak_into_the_projection():
    """A 2026 request returns 2025 rows in the same array."""
    entry = parse_rosters(ROSTER_PAYLOAD, 1)[1].by_player_id(4685382)
    assert entry.weekly_projection(2025, 1) == pytest.approx(500.0)
    assert entry.rest_of_season(2026, from_week=1) == pytest.approx(30.0)


def test_roster_entry_defaults_when_the_player_block_is_thin():
    roster = parse_rosters(ROSTER_PAYLOAD, 1)[1]
    bench = roster.by_player_id(111)
    assert bench.eligible_slots == ()
    assert bench.stats == ()
    assert bench.percent_owned == 0.0
    assert roster.by_player_id(9999) is None


# --------------------------------------------------------------------------------------
# Matchups
# --------------------------------------------------------------------------------------


def test_matchup_sides_opponents_and_byes():
    games = parse_matchups(MATCHUP_PAYLOAD)
    regular = games[0]
    assert regular.opponent_of(1) == 2
    assert regular.opponent_of(2) == 1
    assert regular.opponent_of(7) is None
    assert regular.side_for(1).total_points == pytest.approx(120.5)
    assert not regular.is_playoff
    assert regular.is_complete

    bye = games[1]
    assert bye.is_bye
    assert bye.opponent_of(1) is None
    assert not bye.is_complete


def test_playoff_tier_distinguishes_the_real_bracket_from_the_consolation_ladder():
    games = parse_matchups(MATCHUP_PAYLOAD)
    assert games[1].is_championship_bracket
    assert games[2].is_playoff
    assert not games[2].is_championship_bracket
    assert not games[0].is_playoff  # "NONE", not absent


def test_matchups_survive_the_view_that_omits_playoff_tier():
    """`mMatchup` has no playoffTierType at all; only mMatchupScore/mBoxscore do."""
    stripped = {
        "schedule": [
            {k: v for k, v in m.items() if k != "playoffTierType"}
            for m in MATCHUP_PAYLOAD["schedule"]
        ]
    }
    games = parse_matchups(stripped)
    assert all(m.playoff_tier_type is None for m in games)
    assert not any(m.is_playoff for m in games)


# --------------------------------------------------------------------------------------
# Draft, ratings, transactions
# --------------------------------------------------------------------------------------


def test_draft_picks_carry_auction_bids_and_keeper_flags():
    payload = {
        "draftDetail": {
            "drafted": True,
            "inProgress": False,
            "completeDate": 1788144318384,
            "picks": [
                {
                    "id": 1,
                    "overallPickNumber": 1,
                    "roundId": 1,
                    "roundPickNumber": 1,
                    "teamId": 1,
                    "playerId": 4685382,
                    "bidAmount": 28,
                    "keeper": True,
                    "lineupSlotId": 2,
                    "autoDraftTypeId": 0,
                },
                {
                    "id": 2,
                    "overallPickNumber": 2,
                    "roundId": 1,
                    "roundPickNumber": 2,
                    "teamId": 2,
                    "playerId": 999,
                    "keeper": False,
                },
            ],
        }
    }
    draft = parse_draft(payload)
    assert draft.drafted
    assert len(draft.picks) == 2
    assert draft.keepers[0].bid_amount == 28
    assert draft.for_team(2)[0].player_id == 999
    assert draft.for_team(2)[0].bid_amount == 0


def test_draft_detail_absent_is_an_empty_draft_not_a_crash():
    assert parse_draft({}).picks == ()
    assert parse_draft({"draftDetail": {}}).drafted is False


def test_positional_ratings_keyed_by_position_and_opponent():
    payload = {
        "positionAgainstOpponent": {
            "positionalRatings": {
                "1": {"average": 15.79, "ratingsByOpponent": {"6": {"average": 23.3, "rank": 32}}},
                "16": {"average": 5.6, "ratingsByOpponent": {}},
            }
        }
    }
    ratings = parse_positional_ratings(payload)
    assert ratings[1].by_opponent[6].rank == 32
    assert ratings[16].by_opponent == {}
    assert parse_positional_ratings({}) == {}


def test_transactions_absent_key_is_empty():
    """A scoring period with no activity drops the key rather than sending an empty list."""
    assert parse_transactions({"status": {}}) == []


def test_transaction_items_and_waiver_bids():
    payload = {
        "transactions": [
            {
                "id": "a",
                "type": "WAIVER",
                "status": "EXECUTED",
                "executionType": "PROCESS",
                "scoringPeriodId": 3,
                "teamId": 4,
                "bidAmount": 26,
                "items": [
                    {
                        "type": "ADD",
                        "playerId": 1,
                        "fromTeamId": 0,
                        "toTeamId": 4,
                        "fromLineupSlotId": -1,
                        "toLineupSlotId": 20,
                    },
                    {
                        "type": "DROP",
                        "playerId": 2,
                        "fromTeamId": 4,
                        "toTeamId": 0,
                        "fromLineupSlotId": 20,
                        "toLineupSlotId": -1,
                    },
                ],
            },
            {
                "id": "b",
                "type": "WAIVER",
                "status": "CANCELED",
                "scoringPeriodId": 3,
                "teamId": 5,
                "bidAmount": 99,
                "items": [],
            },
        ]
    }
    txs = parse_transactions(payload)
    assert txs[0].added_player_ids == (1,)
    assert txs[0].dropped_player_ids == (2,)
    assert txs[0].is_waiver and txs[0].is_executed
    assert not txs[1].is_executed


# --------------------------------------------------------------------------------------
# The League client
# --------------------------------------------------------------------------------------


def test_rosters_always_send_the_scoring_period():
    """mRoster honors scoringPeriodId; omitting it returns an unpredictable week."""
    calls: list[dict[str, Any]] = []
    client = _fake_client({"mSettings": FLAT_PLAYOFF_SETTINGS, "mRoster": ROSTER_PAYLOAD}, calls)
    league = League(client, PUBLIC_LEAGUE_ID, 2025)

    league.rosters(week=5)
    assert calls[-1] == {"view": "mRoster", "scoringPeriodId": 5}

    league.rosters()
    # Defaults to the clamped current week (19 -> 17), never to "no parameter".
    assert calls[-1] == {"view": "mRoster", "scoringPeriodId": 17}


def test_an_unrecognized_view_returning_200_is_caught():
    """ESPN answers a typo'd view with a generic payload; the status code is useless."""
    skeleton = {"id": 1, "gameId": 1, "seasonId": 2025, "status": {}}
    league = League(_fake_client({"mTeam": skeleton}), PUBLIC_LEAGUE_ID, 2025)
    with pytest.raises(EspnError, match="unrecognized view"):
        league.teams()


@pytest.mark.parametrize("view", ["mSettings", "mTeam", "mRoster"])
def test_the_real_generic_payload_is_rejected_not_parsed(view: str):
    """The views whose top-level key the generic payload *also* carries.

    A top-level check passes on all three, which is why it is not the check. Left
    unguarded, `settings()` returns a league with no slots, no playoffs and an unclamped
    week 19; `teams()` returns ten nameless teams with 0-0 records; and `rosters()` returns
    every team with zero entries. All three silently, from an HTTP 200.
    """
    league = League(
        _fake_client(dict.fromkeys(("mSettings", "mTeam", "mRoster"), GENERIC_200_PAYLOAD)),
        PUBLIC_LEAGUE_ID,
        2025,
    )
    call = {"mSettings": league.settings, "mTeam": league.teams, "mRoster": league.rosters}[view]

    # The top-level key really is there -- this is not a missing-key test.
    assert {"settings", "teams"} <= set(GENERIC_200_PAYLOAD)
    with pytest.raises(EspnError, match="generic 200 payload"):
        call()


def test_a_real_view_response_still_passes_the_guard():
    """The guard must reject the skeleton without rejecting a thin but genuine league."""
    league = League(
        _fake_client(
            {
                "mSettings": FLAT_PLAYOFF_SETTINGS,
                "mTeam": TEAM_PAYLOAD,
                "mRoster": ROSTER_PAYLOAD,
            }
        ),
        PUBLIC_LEAGUE_ID,
        2025,
    )
    assert league.settings().name == "The Keeper League"
    assert len(league.teams().teams) == 2
    assert len(league.rosters(week=1)) == 1
    # A league with no franchises yet is empty, not broken.
    empty = League(_fake_client({"mTeam": {"teams": [], "members": []}}), 1, 2026)
    assert empty.teams().teams == ()


def test_transactions_loop_scoring_periods_and_deduplicate():
    """One call per scoring period; ESPN serves exactly one period at a time."""
    per_week = {
        0: {"transactions": [{"id": "x", "type": "ROSTER", "scoringPeriodId": 0, "teamId": 1}]},
        1: {
            "transactions": [
                {
                    "id": "y",
                    "type": "WAIVER",
                    "status": "EXECUTED",
                    "scoringPeriodId": 1,
                    "teamId": 2,
                    "bidAmount": 7,
                },
                # A duplicate id across periods must not be counted twice.
                {"id": "x", "type": "ROSTER", "scoringPeriodId": 1, "teamId": 1},
            ]
        },
        2: {"status": {}},  # no `transactions` key at all
    }
    calls: list[dict[str, Any]] = []
    client = _fake_client(
        {
            "mSettings": FLAT_PLAYOFF_SETTINGS,
            "mTransactions2": lambda p: per_week[p["scoringPeriodId"]],
        },
        calls,
    )
    log = League(client, PUBLIC_LEAGUE_ID, 2025).transactions(weeks=[0, 1, 2])

    assert [c["scoringPeriodId"] for c in calls if c["view"] == "mTransactions2"] == [0, 1, 2]
    assert [t.id for t in log] == ["x", "y"]
    assert log.available
    assert log.weeks_covered == (0, 1, 2)
    assert log.waiver_bids == ((2, 7),)


def test_transactions_degrade_gracefully_when_unauthenticated():
    """Auth-gated in private leagues; an empty log must be distinguishable from a refusal."""
    client = _fake_client(
        {
            "mSettings": FLAT_PLAYOFF_SETTINGS,
            "mTransactions2": EspnError("401 from ESPN: credentials missing or stale."),
        }
    )
    log = League(client, PUBLIC_LEAGUE_ID, 2025).transactions(weeks=[1])
    assert not log.available
    assert log.transactions == ()
    assert "ESPN_S2" in log.reason


def test_transactions_do_not_swallow_a_real_failure():
    """A 404 means the league is gone, which must not masquerade as "not authenticated"."""
    client = _fake_client(
        {
            "mSettings": FLAT_PLAYOFF_SETTINGS,
            "mTransactions2": EspnError("404 from ESPN: private league, or no such league."),
        }
    )
    with pytest.raises(EspnError):
        League(client, PUBLIC_LEAGUE_ID, 2025).transactions(weeks=[1])


def test_matchups_for_week_resolves_through_the_matchup_period_map():
    # The final spans scoring periods 16 and 17 but is matchup period 16. Both weeks have
    # to land on game 9; an implementation that passes the scoring period straight through
    # finds nothing for week 17, which is how a two-week final gets silently dropped.
    schedule = {
        "schedule": [
            *MATCHUP_PAYLOAD["schedule"],
            {
                "id": 9,
                "matchupPeriodId": 16,
                "winner": "UNDECIDED",
                "playoffTierType": "WINNERS_BRACKET",
                "home": {"teamId": 1, "totalPoints": 0.0},
                "away": {"teamId": 2, "totalPoints": 0.0},
            },
            {"id": 10, "matchupPeriodId": 17, "winner": "UNDECIDED", "home": {"teamId": 3}},
        ]
    }
    client = _fake_client({"mSettings": VARIABLE_PLAYOFF_SETTINGS, "mMatchupScore": schedule})
    league = League(client, 3, 2026)

    assert league.settings().schedule.scoring_periods(16) == (16, 17)
    assert [m.id for m in league.matchups_for_week(16)] == [9]
    assert [m.id for m in league.matchups_for_week(17)] == [9]  # not [] and not [10]
    assert [m.id for m in league.matchups_for_week(1)] == [1]
    assert [m.id for m in league.matchups(15)] == [2, 3]
    assert league.matchups_for_week(99) == []


def test_settings_are_fetched_once_and_cached():
    calls: list[dict[str, Any]] = []
    league = League(_fake_client({"mSettings": FLAT_PLAYOFF_SETTINGS}, calls), 1, 2025)
    league.settings()
    league.settings()
    league.current_week()
    assert len(calls) == 1
    league.settings(refresh=True)
    assert len(calls) == 2


# --------------------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_live_public_league_shape():
    from fantasy_quant.espn.client import EspnClient

    with EspnClient() as client:
        league = League(client, PUBLIC_LEAGUE_ID, 2025)
        settings = league.settings()
        assert settings.size == 10
        assert settings.draft.is_auction and settings.draft.is_keeper
        assert settings.acquisition.uses_faab
        assert settings.status.latest_scoring_period > settings.status.final_scoring_period
        assert settings.status.current_week == settings.status.final_scoring_period

        teams = league.teams()
        assert len(teams.teams) == 10
        assert teams.team_for_owner(teams.teams[0].owners[0]).id == teams.teams[0].id

        matchups = league.matchups()
        assert any(m.is_championship_bracket for m in matchups), "mMatchupScore lost its tiers"
        assert any(m.is_bye for m in matchups)

        draft = league.draft()
        assert draft.drafted and draft.picks
        assert max(p.bid_amount for p in draft.picks) > 0  # auction


@pytest.mark.network
def test_live_roster_honors_scoring_period():
    """The claim RESEARCH.md makes and community docs deny. Pin it."""
    from fantasy_quant.espn.client import EspnClient

    with EspnClient() as client:
        league = League(client, PUBLIC_LEAGUE_ID, 2025)
        early = league.roster(1, week=1)
        late = league.roster(1, week=12)

    assert early.scoring_period == 1 and late.scoring_period == 12
    early_ids = {e.player_id for e in early.entries}
    late_ids = {e.player_id for e in late.entries}
    assert early_ids != late_ids, "mRoster ignored scoringPeriodId -- backfills are corrupt"


@pytest.mark.network
def test_live_transactions_need_one_call_per_scoring_period():
    """Omitting scoringPeriodId silently returns only the current period."""
    from fantasy_quant.espn.client import EspnClient
    from fantasy_quant.espn.endpoints import league_url

    with EspnClient() as client:
        payload, _ = client.get(
            league_url(2025, PUBLIC_LEAGUE_ID), params={"view": "mTransactions2"}
        )
        # A finished season's current period is empty, so the key is missing entirely.
        assert "transactions" not in payload

        log = League(client, PUBLIC_LEAGUE_ID, 2025).transactions(weeks=range(0, 18))

    assert log.available
    assert len(log) > 400
    assert len({t.id for t in log}) == len(log)
    assert any(bid > 0 for _, bid in log.waiver_bids)
