"""Scoring engine tests.

The offline fixtures are not invented: the scoring items are league 899513's real
2025 `scoringItems`, and the four stat lines are real rows lifted from that
league's Week 4 boxscore, each with the `appliedTotal` ESPN published. So the
pure-logic tests are an offline replay of the live acceptance test, and the
network test is the same check against ESPN today.

Every trap RESEARCH.md documents gets a test that fails if we regress on it:
position-keyed vs slot-keyed overrides, replace-vs-add, a `0.0` override that is
a real value, absent `pointsOverrides`, and the derived stat buckets that
multi-count if you apply everything present in raw `stats`.
"""

from __future__ import annotations

import copy

import pytest

from fantasy_quant.espn.client import EspnClient
from fantasy_quant.espn.scoring import (
    POS_DST,
    POS_QB,
    POS_RB,
    POS_TE,
    POS_WR,
    LeagueScoring,
    LeagueShape,
    ScoringReproductionError,
    _default_scoring_period,
    check_boxscore,
    fetch_league_settings,
    verify_scoring_reproduction,
)
from fantasy_quant.espn.statrows import SOURCE_ACTUAL, SOURCE_PROJECTED

# --- League 899513 (2025), verbatim: IDP + PPFD + TQB, no per-reception scoring.
IDP_PPFD_ITEMS = [
    {"statId": 3, "points": 0.04},
    {"statId": 4, "points": 4.0},
    {"statId": 19, "points": 2.0},
    {"statId": 20, "points": -2.0},
    {
        "statId": 24,
        "points": 0.0,
        "pointsOverrides": {"1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1, "15": 0.1},
    },
    {
        "statId": 25,
        "points": 0.0,
        "pointsOverrides": {"1": 6.0, "2": 6.0, "3": 6.0, "4": 6.0, "15": 6.0},
    },
    {
        "statId": 26,
        "points": 0.0,
        "pointsOverrides": {"1": 2.0, "2": 2.0, "3": 2.0, "4": 2.0, "15": 2.0},
    },
    {"statId": 42, "points": 0.1},
    {"statId": 43, "points": 6.0},
    {"statId": 44, "points": 2.0},
    {"statId": 63, "points": 6.0},
    {"statId": 72, "points": -2.0},
    {"statId": 79, "points": -2.0},
    {"statId": 82, "points": -3.0},
    {"statId": 86, "points": 1.0},
    {"statId": 88, "points": -1.0},
    {"statId": 89, "points": 0.0, "pointsOverrides": {"16": 5.0}},
    {"statId": 90, "points": 0.0, "pointsOverrides": {"16": 4.0}},
    {"statId": 91, "points": 0.0, "pointsOverrides": {"16": 3.0}},
    {"statId": 92, "points": 0.0, "pointsOverrides": {"16": 1.0}},
    {"statId": 93, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 95, "points": 5.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 96, "points": 4.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 97, "points": 2.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 98, "points": 2.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 99, "points": 4.0, "pointsOverrides": {"16": 1.0}},
    {"statId": 101, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 102, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 103, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 104, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 106, "points": 4.0, "pointsOverrides": {"16": 0.0}},
    {"statId": 107, "points": 0.75, "pointsOverrides": {"16": 0.0}},
    {"statId": 108, "points": 1.5, "pointsOverrides": {"16": 0.0}},
    {"statId": 113, "points": 1.5, "pointsOverrides": {"16": 0.0}},
    {"statId": 114, "points": 0.02, "pointsOverrides": {"16": 0.02}},
    {"statId": 115, "points": 0.02, "pointsOverrides": {"16": 0.02}},
    {"statId": 124, "points": 0.0, "pointsOverrides": {"16": -2.0}},
    {"statId": 125, "points": 0.0, "pointsOverrides": {"16": -4.0}},
    {"statId": 128, "points": 0.0, "pointsOverrides": {"16": 5.0}},
    {"statId": 129, "points": 0.0, "pointsOverrides": {"16": 3.0}},
    {"statId": 130, "points": 0.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 133, "points": 0.0, "pointsOverrides": {"16": -2.0}},
    {"statId": 134, "points": 0.0, "pointsOverrides": {"16": -4.0}},
    {"statId": 135, "points": 0.0, "pointsOverrides": {"16": -5.0}},
    {"statId": 136, "points": 0.0, "pointsOverrides": {"16": -6.0}},
    {"statId": 161, "points": 10.0},
    {"statId": 162, "points": 8.0},
    {"statId": 163, "points": 6.0},
    {"statId": 164, "points": 4.0},
    {"statId": 165, "points": 2.0},
    {"statId": 166, "points": 1.0},
    {"statId": 200, "points": -1.0},
    {"statId": 206, "points": 2.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 209, "points": 1.0, "pointsOverrides": {"16": 1.0}},
    {"statId": 212, "points": 0.0, "pointsOverrides": {"2": 0.5, "3": 0.5, "4": 0.5}},
    {"statId": 213, "points": 0.5},
    {"statId": 214, "points": 0.1},
]

# --- Real Week 4 stat lines from that same league. -------------------------------
FLEX_RB = {
    "lineupSlotId": 23,
    "playerPoolEntry": {
        "id": 4567048,
        "appliedStatTotal": 12.5,
        "player": {
            "id": 4567048,
            "fullName": "Kenneth Walker III",
            "defaultPositionId": 2,
            "stats": [
                {
                    "id": "01401772938",
                    "seasonId": 2025,
                    "statSourceId": 0,
                    "statSplitTypeId": 1,
                    "scoringPeriodId": 4,
                    "appliedTotal": 12.5,
                    "proTeamId": 26,
                    "appliedStats": {"212": 1.0, "213": 0.5, "24": 8.1, "42": 2.9000000000000004},
                    "stats": {
                        "23": 19.0,
                        "24": 81.0,
                        "27": 16.0,
                        "28": 8.0,
                        "29": 4.0,
                        "30": 3.0,
                        "31": 1.0,
                        "33": 3.0,
                        "34": 1.0,
                        "39": 4.263,
                        "40": 81.0,
                        "41": 1.0,
                        "42": 29.0,
                        "47": 5.0,
                        "48": 2.0,
                        "49": 1.0,
                        "50": 1.0,
                        "53": 1.0,
                        "58": 2.0,
                        "59": 32.0,
                        "60": 29.0,
                        "61": 29.0,
                        "155": 1.0,
                        "210": 1.0,
                        "212": 2.0,
                        "213": 1.0,
                    },
                }
            ],
        },
    },
}

PATRIOTS_TQB = {
    "lineupSlotId": 1,
    "playerPoolEntry": {
        "id": -15017,
        "appliedStatTotal": 23.020000000000003,
        "player": {
            "id": -15017,
            "fullName": "Patriots TQB",
            "defaultPositionId": 15,
            "stats": [
                {
                    "id": "01401772847",
                    "seasonId": 2025,
                    "statSourceId": 0,
                    "statSplitTypeId": 1,
                    "scoringPeriodId": 4,
                    "appliedTotal": 23.02,
                    "proTeamId": 17,
                    "appliedStats": {
                        "82": 0.0,
                        "3": 8.120000000000001,
                        "4": 8.0,
                        "212": 0.0,
                        "200": 0.0,
                        "24": 0.9,
                        "88": 0.0,
                        "25": 6.0,
                        "93": 0.0,
                        "206": 0.0,
                        "79": 0.0,
                    },
                    "stats": {
                        "0": 18.0,
                        "1": 14.0,
                        "2": 4.0,
                        "3": 203.0,
                        "4": 2.0,
                        "5": 40.0,
                        "6": 20.0,
                        "7": 10.0,
                        "8": 8.0,
                        "9": 4.0,
                        "10": 2.0,
                        "11": 2.0,
                        "12": 1.0,
                        "13": 0.0,
                        "14": 0.0,
                        "15": 0.0,
                        "17": 0.0,
                        "18": 0.0,
                        "21": 82.35,
                        "22": 203.0,
                        "23": 6.0,
                        "24": 9.0,
                        "25": 1.0,
                        "27": 1.0,
                        "28": 0.0,
                        "29": 0.0,
                        "30": 0.0,
                        "31": 0.0,
                        "32": 0.0,
                        "33": 1.0,
                        "34": 0.0,
                        "35": 0.0,
                        "39": 3.0,
                        "40": 9.0,
                        "45": 0.0,
                        "64": 1.0,
                        "73": 0.0,
                        "76": 0.0,
                        "79": 0.0,
                        "80": 0.0,
                        "81": 0.0,
                        "82": 0.0,
                        "85": 0.0,
                        "88": 0.0,
                        "93": 0.0,
                        "94": 0.0,
                        "105": 0.0,
                        "155": 1.0,
                        "158": 18.0,
                        "175": 1.0,
                        "178": 1.0,
                        "179": 1.0,
                        "200": 0.0,
                        "203": 0.0,
                        "206": 0.0,
                        "210": 1.0,
                        "211": 8.0,
                        "212": 1.0,
                    },
                }
            ],
        },
    },
}

IDP_LINEBACKER = {
    "lineupSlotId": 20,
    "playerPoolEntry": {
        "id": 4243181,
        "appliedStatTotal": 8.25,
        "player": {
            "id": 4243181,
            "fullName": "Nate Landman",
            "defaultPositionId": 11,
            "stats": [
                {
                    "id": "01401772849",
                    "seasonId": 2025,
                    "statSourceId": 0,
                    "statSplitTypeId": 1,
                    "scoringPeriodId": 4,
                    "appliedTotal": 8.25,
                    "proTeamId": 14,
                    "appliedStats": {"113": 1.5, "107": 2.25, "108": 4.5},
                    "stats": {
                        "107": 3.0,
                        "108": 3.0,
                        "109": 6.0,
                        "110": 2.0,
                        "111": 1.0,
                        "113": 1.0,
                        "155": 1.0,
                        "210": 1.0,
                    },
                }
            ],
        },
    },
}

PATRIOTS_DST = {
    "lineupSlotId": 16,
    "playerPoolEntry": {
        "id": -16017,
        "appliedStatTotal": 14.3,
        "player": {
            "id": -16017,
            "fullName": "Patriots D/ST",
            "defaultPositionId": 16,
            "stats": [
                {
                    "id": "01401772847",
                    "seasonId": 2025,
                    "statSourceId": 0,
                    "statSplitTypeId": 1,
                    "scoringPeriodId": 4,
                    "appliedTotal": 14.3,
                    "proTeamId": 17,
                    "appliedStats": {
                        "128": 0.0,
                        "129": 0.0,
                        "130": 0.0,
                        "99": 1.0,
                        "133": 0.0,
                        "134": 0.0,
                        "102": 6.0,
                        "135": 0.0,
                        "136": 0.0,
                        "106": 0.0,
                        "107": 0.0,
                        "108": 0.0,
                        "113": 0.0,
                        "114": 0.96,
                        "115": 3.34,
                        "89": 0.0,
                        "90": 0.0,
                        "91": 3.0,
                        "124": 0.0,
                        "92": 0.0,
                    },
                    "stats": {
                        "89": 0.0,
                        "90": 0.0,
                        "91": 1.0,
                        "92": 0.0,
                        "99": 1.0,
                        "100": 2.0,
                        "102": 1.0,
                        "105": 1.0,
                        "106": 1.0,
                        "107": 32.0,
                        "108": 43.0,
                        "109": 75.0,
                        "110": 25.0,
                        "111": 15.0,
                        "112": 6.0,
                        "113": 2.0,
                        "114": 48.0,
                        "115": 167.0,
                        "116": 4.0,
                        "117": 1.0,
                        "118": 16.0,
                        "119": 6.0,
                        "120": 13.0,
                        "121": 0.0,
                        "122": 0.0,
                        "123": 0.0,
                        "124": 0.0,
                        "127": 326.0,
                        "128": 0.0,
                        "129": 0.0,
                        "130": 0.0,
                        "131": 1.0,
                        "132": 0.0,
                        "133": 0.0,
                        "134": 0.0,
                        "135": 0.0,
                        "136": 0.0,
                        "155": 1.0,
                        "187": 13.0,
                        "188": 0.0,
                        "189": 0.0,
                        "190": 1.0,
                        "191": 0.0,
                        "192": 0.0,
                        "193": 0.0,
                        "194": 0.0,
                        "195": 0.0,
                        "210": 1.0,
                    },
                }
            ],
        },
    },
}

REAL_ENTRIES = [FLEX_RB, PATRIOTS_TQB, IDP_LINEBACKER, PATRIOTS_DST]

# Kenneth Walker's real *projected* Week 4 row, lifted from the same payload as his
# actual one. Projections are the harsher half of the check: every raw count is
# fractional, so every derived bucket (40 = 24, 61 = 42, 47/48, 53) is non-zero and a
# statId applied that shouldn't be cannot hide behind a 0.
WALKER_PROJECTED_ROW = {
    "id": "1120254",
    "seasonId": 2025,
    "statSourceId": 1,
    "statSplitTypeId": 1,
    "scoringPeriodId": 4,
    "appliedTotal": 12.50485147,
    "proTeamId": 0,
    "appliedStats": {
        "24": 5.649639664,
        "25": 2.81244789,
        "26": 0.029701002,
        "42": 1.687339948,
        "43": 0.335951958,
        "44": 0.005308592,
        "63": 0.002664,
        "72": -0.095645546,
        "212": 1.604543928,
        "213": 0.4729000385,
    },
    "stats": {
        "23": 13.66413338,
        "24": 56.49639664,
        "25": 0.468741315,
        "26": 0.014850501,
        "27": 11.0,
        "28": 5.0,
        "29": 2.0,
        "30": 2.0,
        "31": 1.0,
        "33": 2.0,
        "34": 1.0,
        "35": 0.028225692,
        "36": 0.019757984,
        "37": 0.14,
        "38": 0.00476,
        "39": 4.13464909,
        "40": 56.49639664,
        "42": 16.87339948,
        "43": 0.055991993,
        "44": 0.002654296,
        "45": 0.002895588,
        "46": 0.001892267,
        "47": 3.0,
        "48": 1.0,
        "53": 2.304326122,
        "56": 0.006859339,
        "57": 0.000212,
        "58": 2.887895211,
        "60": 7.322487612,
        "61": 16.87339948,
        "62": 0.017504798,
        "63": 0.000444,
        "66": 0.086254396,
        "67": 0.014545983,
        "68": 0.100800379,
        "70": 0.039677022,
        "71": 0.008145751,
        "72": 0.047822773,
        "73": 0.047822773,
        "210": 1.0,
        "212": 3.209087856,
        "213": 0.945800077,
    },
}

# The same roster entry carrying both of that week's rows, which is what ESPN really
# returns. Kept separate so the four-entry fixtures above stay one row per player.
FLEX_RB_BOTH_SOURCES = copy.deepcopy(FLEX_RB)
FLEX_RB_BOTH_SOURCES["playerPoolEntry"]["player"]["stats"].append(
    copy.deepcopy(WALKER_PROJECTED_ROW)
)

# --- League 350313 (2025), verbatim except for the per-item `leagueRanking` /
# --- `leagueTotal` fields ESPN also ships and the parser ignores (kept on the first
# --- item so that tolerance is itself asserted). Three items are `isReverseItem`.
REVERSE_ITEM_ITEMS = [
    {"statId": 3, "points": 0.04, "leagueRanking": 0.0, "leagueTotal": 0.0},
    {"statId": 4, "points": 4.0},
    {"statId": 19, "points": 2.0},
    {"statId": 20, "points": -2.0, "isReverseItem": True},
    {"statId": 24, "points": 0.1},
    {"statId": 25, "points": 6.0},
    {"statId": 26, "points": 2.0},
    {"statId": 42, "points": 0.1},
    {"statId": 43, "points": 6.0},
    {"statId": 44, "points": 2.0},
    {"statId": 53, "points": 0.5},
    {"statId": 63, "points": 6.0},
    {"statId": 72, "points": -2.0, "isReverseItem": True},
    {"statId": 77, "points": 4.0},
    {"statId": 80, "points": 3.0},
    {"statId": 85, "points": -1.0, "isReverseItem": True},
    {"statId": 86, "points": 1.0},
    {"statId": 89, "points": 0.0, "pointsOverrides": {"16": 5.0}},
    {"statId": 90, "points": 0.0, "pointsOverrides": {"16": 4.0}},
    {"statId": 91, "points": 0.0, "pointsOverrides": {"16": 3.0}},
    {"statId": 92, "points": 0.0, "pointsOverrides": {"16": 1.0}},
    {"statId": 93, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 95, "points": 0.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 96, "points": 0.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 97, "points": 0.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 98, "points": 0.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 99, "points": 0.0, "pointsOverrides": {"16": 1.0}},
    {"statId": 101, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 102, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 103, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 104, "points": 6.0, "pointsOverrides": {"16": 6.0}},
    {"statId": 123, "points": 0.0, "pointsOverrides": {"16": -1.0}},
    {"statId": 124, "points": 0.0, "pointsOverrides": {"16": -3.0}},
    {"statId": 125, "points": 0.0, "pointsOverrides": {"16": -5.0}},
    {"statId": 128, "points": 0.0, "pointsOverrides": {"16": 5.0}},
    {"statId": 129, "points": 0.0, "pointsOverrides": {"16": 3.0}},
    {"statId": 130, "points": 0.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 132, "points": 0.0, "pointsOverrides": {"16": -1.0}},
    {"statId": 133, "points": 0.0, "pointsOverrides": {"16": -3.0}},
    {"statId": 134, "points": 0.0, "pointsOverrides": {"16": -5.0}},
    {"statId": 135, "points": 0.0, "pointsOverrides": {"16": -6.0}},
    {"statId": 136, "points": 0.0, "pointsOverrides": {"16": -7.0}},
    {"statId": 198, "points": 5.0},
    {"statId": 201, "points": 5.0},
    {"statId": 206, "points": 2.0, "pointsOverrides": {"16": 2.0}},
    {"statId": 209, "points": 1.0, "pointsOverrides": {"16": 1.0}},
]

# Geno Smith's real Week 4 line in that league: three interceptions thrown (statId 20,
# the reverse item) which ESPN's own appliedStats scores at -6.0, not +6.0.
GENO_SMITH_WEEK_4 = {
    "id": "01401772742",
    "seasonId": 2025,
    "statSourceId": 0,
    "statSplitTypeId": 1,
    "scoringPeriodId": 4,
    "appliedTotal": 9.78,
    "proTeamId": 13,
    "appliedStats": {"3": 4.68, "4": 8.0, "20": -6.0, "24": 3.1},
    "stats": {
        "0": 21.0,
        "1": 14.0,
        "2": 7.0,
        "3": 117.0,
        "4": 2.0,
        "5": 23.0,
        "6": 11.0,
        "7": 5.0,
        "8": 4.0,
        "9": 2.0,
        "10": 1.0,
        "11": 2.0,
        "12": 1.0,
        "13": 1.0,
        "20": 3.0,
        "21": 66.67,
        "22": 117.0,
        "23": 4.0,
        "24": 31.0,
        "27": 6.0,
        "28": 3.0,
        "29": 1.0,
        "30": 1.0,
        "39": 7.75,
        "40": 31.0,
        "73": 3.0,
        "156": 1.0,
        "158": 12.0,
        "175": 2.0,
        "210": 1.0,
        "211": 6.0,
        "212": 1.0,
    },
}

# --- League 1241838 (2025) shape: half PPR with a 0.5 TE premium. -----------------
HALF_PPR_TE_PREMIUM_ITEMS = [
    {"statId": 3, "points": 0.04},
    {"statId": 4, "points": 4.0},
    {"statId": 24, "points": 0.1},
    {"statId": 42, "points": 0.1},
    {"statId": 43, "points": 6.0},
    # points 0.0 with everything real in the overrides -- the shape that makes
    # `ovr[4] - points` report a fake 1.0 premium instead of the true 0.5.
    {
        "statId": 53,
        "points": 0.0,
        "pointsOverrides": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 1.0, "15": 0.5},
    },
]


def scorer(items=IDP_PPFD_ITEMS) -> LeagueScoring:
    return LeagueScoring.from_settings({"scoringSettings": {"scoringItems": items}})


def boxscore(entries=REAL_ENTRIES, week: int = 4, season: int = 2025) -> dict:
    """Wrap roster entries in the `mBoxscore` shape ESPN actually returns."""
    return {
        "id": 899513,
        "seasonId": season,
        "scoringPeriodId": week,
        "schedule": [
            # ESPN returns the whole season's schedule and populates rosters for
            # the requested week only, so most matchups look like this one.
            {"matchupPeriodId": 1, "home": {"teamId": 1}, "away": {"teamId": 2}},
            {
                "matchupPeriodId": week,
                "home": {
                    "teamId": 3,
                    "rosterForCurrentScoringPeriod": {"entries": list(entries)},
                },
                "away": {"teamId": 4, "rosterForCurrentScoringPeriod": {"entries": []}},
            },
        ],
    }


def entries_for_week(week: int, entries=REAL_ENTRIES) -> list[dict]:
    """The real entries with every stat row re-stamped to `week`."""
    out = copy.deepcopy(list(entries))
    for entry in out:
        for row in entry["playerPoolEntry"]["player"]["stats"]:
            row["scoringPeriodId"] = week
    return out


def only(entry) -> tuple[dict, int, float]:
    """(raw stats, defaultPositionId, ESPN's appliedTotal) for one fixture entry."""
    player = entry["playerPoolEntry"]["player"]
    row = player["stats"][0]
    return row["stats"], player["defaultPositionId"], row["appliedTotal"]


# ---------------------------------------------------------------------------
# The three rules
# ---------------------------------------------------------------------------


def test_overrides_are_keyed_by_default_position_not_lineup_slot():
    """The whole ballgame. Kenneth Walker is defaultPositionId 2, lineupSlotId 23.

    Position 2 has an override on rushing yards and rushing first downs; slot 23
    does not exist in that override map, so scoring by slot silently falls back to
    `points` (0.0) and drops two thirds of his score. It would still look correct
    for a D/ST, whose position id and slot id are both 16 -- which is exactly how
    this bug survives in other tools.
    """
    s = scorer()
    raw, position_id, espn = only(FLEX_RB)
    assert s.score(raw, position_id) == pytest.approx(espn)
    assert espn == 12.5

    by_slot = s.score(raw, FLEX_RB["lineupSlotId"])
    assert by_slot == pytest.approx(3.4)
    assert by_slot != pytest.approx(espn)

    # And the collision that hides it: D/ST is 16 in both ID spaces.
    dst_raw, dst_position, dst_espn = only(PATRIOTS_DST)
    assert dst_position == PATRIOTS_DST["lineupSlotId"] == POS_DST
    assert s.score(dst_raw, PATRIOTS_DST["lineupSlotId"]) == pytest.approx(dst_espn)


def test_an_override_replaces_points_it_does_not_add():
    """statId 96 (fumble recovery): points 4.0, pointsOverrides {"16": 2.0}."""
    s = scorer()
    assert s.points_for(96, POS_DST) == 2.0  # not 6.0
    assert s.points_for(96, 11) == 4.0  # a linebacker keeps the base value
    assert s.score({"96": 1.0}, POS_DST) == 2.0


def test_a_zero_override_is_a_real_value_not_an_unset_one():
    """statId 106 (forced fumble): points 4.0, pointsOverrides {"16": 0.0}.

    `overrides.get(pos) or points` -- the natural-looking bug -- pays a D/ST four
    points for a forced fumble ESPN scores at zero. The real Patriots line has
    one, and ESPN's own appliedStats says 106 -> 0.0.
    """
    s = scorer()
    assert s.points_for(106, POS_DST) == 0.0
    assert s.points_for(106, 11) == 4.0

    raw, position_id, espn = only(PATRIOTS_DST)
    assert raw["106"] == 1.0
    assert PATRIOTS_DST["playerPoolEntry"]["player"]["stats"][0]["appliedStats"]["106"] == 0.0
    assert s.score(raw, position_id) == pytest.approx(espn)
    # The "or" bug would show up as exactly the 4.0 it wrongly credits.
    assert s.score(raw, position_id) != pytest.approx(espn + 4.0)


def test_only_stat_ids_with_a_scoring_item_are_applied():
    """Raw `stats` ships derived buckets that restate the same production.

    Walker's line carries 40 (= 24 rushing yards), 61 (= 42 receiving yards),
    41 (= 53 receptions) and 47-50 (the "every N receiving yards" ladders). None
    has a scoringItem in this league, so none may contribute -- applying 61 alone
    would double-count his receiving yards.
    """
    s = scorer()
    raw, position_id, espn = only(FLEX_RB)
    assert raw["40"] == raw["24"] == 81.0
    assert raw["61"] == raw["42"] == 29.0
    assert raw["41"] == raw["53"] == 1.0

    assert s.score(raw, position_id) == pytest.approx(espn)
    contributions = s.breakdown(raw, position_id)
    assert set(contributions) == {"24", "42", "212", "213"}
    for derived in ("40", "41", "47", "48", "49", "50", "61", "58", "59", "60", "155", "210"):
        assert derived not in contributions


def test_defensive_derived_buckets_are_ignored_too():
    """100 = 2x99 and 109 = 107+108, and the PA/YA bands are one-hot."""
    s = scorer()
    raw, position_id, espn = only(PATRIOTS_DST)
    assert raw["100"] == 2 * raw["99"]
    assert raw["109"] == raw["107"] + raw["108"]
    # Exactly one points-allowed band is hot; the rest are present as zeros.
    bands = {b: raw[b] for b in ("89", "90", "91", "92", "121", "122", "123", "124") if b in raw}
    assert sum(bands.values()) == 1.0

    assert s.score(raw, position_id) == pytest.approx(espn)
    contributions = s.breakdown(raw, position_id)
    assert "100" not in contributions
    assert "109" not in contributions


def test_points_overrides_may_be_absent_or_empty():
    """Both forms occur live; `.get` is the only safe read."""
    s = scorer(
        [
            {"statId": 53, "points": 1.0},  # absent entirely
            {"statId": 42, "points": 0.1, "pointsOverrides": {}},  # empty
        ]
    )
    assert s.items["53"].overrides == {}
    assert s.score({"53": 5.0, "42": 100.0}, POS_TE) == pytest.approx(15.0)


def test_reproduces_every_real_stat_line_offline():
    s = scorer()
    for entry in REAL_ENTRIES:
        raw, position_id, espn = only(entry)
        assert s.score(raw, position_id) == pytest.approx(espn, abs=1e-9), entry
        assert s.breakdown(raw, position_id) == pytest.approx(
            entry["playerPoolEntry"]["player"]["stats"][0]["appliedStats"], abs=1e-9
        )


def test_int_keyed_raw_stats_still_score():
    """A Parquet round-trip or a hand-built dict can hand us ints, not strings."""
    s = scorer()
    raw, position_id, espn = only(FLEX_RB)
    assert s.score({int(k): v for k, v in raw.items()}, position_id) == pytest.approx(espn)


def test_unknown_position_scores_the_base_points():
    s = scorer()
    # Position 99 has no override anywhere, so every item falls back to `points`.
    assert s.score({"96": 1.0}, 99) == 4.0
    # ...including the ones whose real value lives only in an override.
    assert s.score({"24": 100.0}, 99) == 0.0


def test_the_scorer_is_callable():
    """`score` is also exposed as `__call__` so it can be passed as a plain callable."""
    s = scorer()
    raw, position_id, espn = only(FLEX_RB)
    assert s(raw, position_id) == s.score(raw, position_id) == pytest.approx(espn)


def test_is_reverse_item_is_metadata_and_never_flips_the_sign():
    """`isReverseItem: true` DOES occur live, and it is not a sign flip.

    League 350313 (2025) ships it on statIds 20, 72 and 85, each with a negative
    `points`. Geno Smith's real Week 4 line there has three interceptions thrown and
    ESPN's own appliedStats scores them -6.0, not +6.0 -- `points` already carries
    the sign, and the flag is ranking metadata ("lower is better"). Flipping it
    would turn his 9.78 into 21.78.
    """
    s = LeagueScoring.from_settings({"scoringSettings": {"scoringItems": REVERSE_ITEM_ITEMS}})
    assert {item.stat_id for item in s.reverse_items} == {"20", "72", "85"}
    assert all(item.points < 0 for item in s.reverse_items)
    assert s.points_for(20, POS_QB) == -2.0

    raw = GENO_SMITH_WEEK_4["stats"]
    assert raw["20"] == 3.0
    assert GENO_SMITH_WEEK_4["appliedStats"]["20"] == -6.0
    assert s.score(raw, POS_QB) == pytest.approx(GENO_SMITH_WEEK_4["appliedTotal"])
    assert s.breakdown(raw, POS_QB) == pytest.approx(GENO_SMITH_WEEK_4["appliedStats"])
    assert s.score(raw, POS_QB) != pytest.approx(GENO_SMITH_WEEK_4["appliedTotal"] + 12.0)

    # The same line also proves 22 == 3 and 40 == 24 stay out, and that fumbles
    # (73) are not mistaken for fumbles lost (72, which this league does score).
    assert raw["22"] == raw["3"] and raw["40"] == raw["24"] and raw["73"] == 3.0
    assert set(s.breakdown(raw, POS_QB)) == {"3", "4", "20", "24"}


def test_unknown_scoring_item_keys_are_ignored():
    """ESPN ships `leagueRanking`/`leagueTotal` on every item; neither is scoring."""
    s = LeagueScoring.from_settings({"scoringSettings": {"scoringItems": REVERSE_ITEM_ITEMS}})
    assert "leagueRanking" in REVERSE_ITEM_ITEMS[0]
    assert s.points_for(3, POS_QB) == 0.04


def test_empty_scoring_items_is_an_error():
    """An unrecognized `view=` returns HTTP 200 with a skeleton, so silence is fatal."""
    with pytest.raises(ValueError, match="scoringItems"):
        LeagueScoring.from_settings({"settings": {"scoringSettings": {"scoringItems": []}}})


def test_from_settings_accepts_payload_settings_or_scoring_settings():
    payload = {"settings": {"scoringSettings": {"scoringItems": IDP_PPFD_ITEMS}}}
    a = LeagueScoring.from_settings(payload)
    b = LeagueScoring.from_settings(payload["settings"])
    c = LeagueScoring.from_settings(payload["settings"]["scoringSettings"])
    assert a.items == b.items == c.items


# ---------------------------------------------------------------------------
# Boxscore reproduction
# ---------------------------------------------------------------------------


def test_check_boxscore_reproduces_the_whole_roster():
    report = check_boxscore(scorer(), boxscore(), season=2025, scoring_period_id=4)
    assert report.exact
    assert report.checked == len(REAL_ENTRIES)
    assert report.matched == report.checked
    assert report.mismatches == ()
    assert report.positions == {POS_RB: 1, 11: 1, 15: 1, POS_DST: 1}
    assert "exact" in report.summary()
    report.raise_if_failed()


def test_check_boxscore_reports_the_offending_player_and_stat():
    """A wrong rule must name the player, the stat, and both totals."""
    broken = copy.deepcopy(IDP_PPFD_ITEMS)
    for item in broken:
        if item["statId"] == 24:  # rushing yards: drop the per-position overrides
            item.pop("pointsOverrides")

    report = check_boxscore(scorer(broken), boxscore(), season=2025, scoring_period_id=4)
    assert not report.exact
    names = {m.player_name for m in report.mismatches}
    assert "Kenneth Walker III" in names

    walker = next(m for m in report.mismatches if m.player_name == "Kenneth Walker III")
    assert walker.espn_total == 12.5
    assert walker.our_total == pytest.approx(4.4)
    assert walker.delta == pytest.approx(-8.1)
    assert walker.raw_stats["24"] == 81.0
    assert walker.default_position_id == POS_RB
    assert walker.lineup_slot_id == 23
    text = walker.describe()
    assert "Kenneth Walker III" in text and "differs" in text and "24" in text

    with pytest.raises(ScoringReproductionError, match="Kenneth Walker III"):
        report.raise_if_failed()


def test_a_boxscore_with_no_rosters_is_a_failure_not_a_pass():
    """Zero rows checked means the payload shape moved, not that scoring is fine."""
    report = check_boxscore(scorer(), boxscore(entries=[]), season=2025, scoring_period_id=4)
    assert report.checked == 0
    assert not report.exact
    assert "NOTHING CHECKED" in report.summary()
    with pytest.raises(ScoringReproductionError):
        report.raise_if_failed()


def test_check_boxscore_ignores_rows_from_other_weeks_and_seasons():
    """A stat array can carry a neighbouring season; only this week's row counts."""
    entry = copy.deepcopy(FLEX_RB)
    stale = copy.deepcopy(entry["playerPoolEntry"]["player"]["stats"][0])
    stale.update(id="002024", seasonId=2024, scoringPeriodId=4, appliedTotal=999.0)
    other_week = copy.deepcopy(entry["playerPoolEntry"]["player"]["stats"][0])
    other_week.update(id="01401772999", scoringPeriodId=5, appliedTotal=999.0)
    entry["playerPoolEntry"]["player"]["stats"] = [
        stale,
        other_week,
        *entry["playerPoolEntry"]["player"]["stats"],
    ]

    report = check_boxscore(scorer(), boxscore([entry]), season=2025, scoring_period_id=4)
    assert report.checked == 1
    assert report.exact


def test_source_filter_narrows_what_is_checked():
    """The fixture must carry BOTH sources or this asserts nothing.

    With only actual rows present, an implementation that ignored `sources`
    entirely would produce exactly the same counts.
    """
    s = scorer()

    def run(**kwargs):
        return check_boxscore(
            s, boxscore([FLEX_RB_BOTH_SOURCES]), season=2025, scoring_period_id=4, **kwargs
        )

    both = run()
    assert both.checked == 2, "the entry carries one actual and one projected row"
    assert both.exact

    actual = run(sources=(SOURCE_ACTUAL,))
    assert actual.checked == 1
    assert actual.exact

    projected = run(sources=(SOURCE_PROJECTED,))
    assert projected.checked == 1
    assert projected.exact

    # An empty source list checks nothing, which `exact` reports as a failure.
    nothing = run(sources=())
    assert nothing.checked == 0
    assert not nothing.exact


def test_a_fractional_projection_row_applies_only_the_scored_stat_ids():
    """Projections are where multi-counting cannot hide behind a zero.

    Walker's projected line has 40 == 24 and 61 == 42 as *fractional* yardage, a
    fractional 53 in a league that does not score receptions at all, and integer
    47/48 ladders. Applying any of them moves the total.
    """
    s = scorer()
    raw = WALKER_PROJECTED_ROW["stats"]
    assert raw["40"] == raw["24"] > 0
    assert raw["61"] == raw["42"] > 0
    assert raw["53"] > 0 and "53" not in s.scored_stat_ids
    assert raw["47"] > 0 and raw["48"] > 0

    assert s.score(raw, POS_RB) == pytest.approx(WALKER_PROJECTED_ROW["appliedTotal"], abs=1e-6)
    assert s.breakdown(raw, POS_RB) == pytest.approx(WALKER_PROJECTED_ROW["appliedStats"], abs=1e-6)


def test_rows_with_no_raw_stats_are_not_counted_as_checked():
    """Every empty-`stats` row in the live corpus carries appliedTotal 0.0.

    Scoring them is a free pass that would only inflate `checked` and make an
    otherwise-empty payload look verified.
    """
    entry = copy.deepcopy(FLEX_RB)
    blank = copy.deepcopy(entry["playerPoolEntry"]["player"]["stats"][0])
    blank.update(id="01401772939", stats={}, appliedStats={}, appliedTotal=0.0)
    entry["playerPoolEntry"]["player"]["stats"].append(blank)

    report = check_boxscore(scorer(), boxscore([entry]), season=2025, scoring_period_id=4)
    assert report.checked == 1
    assert report.exact


# ---------------------------------------------------------------------------
# Picking the week to check, and walking back when it is empty
# ---------------------------------------------------------------------------


def test_latest_scoring_period_is_clamped_to_the_final_one():
    """`latestScoringPeriod` runs past the end of the fantasy season.

    Measured live: a finished 2025 league reports 19 against a `finalScoringPeriod`
    of 17 (league 1241838) or 18 (league 899513), and weeks past the final one carry
    no rosters at all, so an unclamped week would check nothing.
    """
    assert _default_scoring_period(
        {"status": {"firstScoringPeriod": 1, "latestScoringPeriod": 19, "finalScoringPeriod": 17}}
    ) == (17, 1)
    assert _default_scoring_period(
        {"status": {"firstScoringPeriod": 1, "latestScoringPeriod": 19, "finalScoringPeriod": 18}}
    ) == (18, 1)


def test_mid_season_latest_is_not_pushed_forward_to_the_final_week():
    """2026 week 1: latest 1, final 17. Asking for 17 would return an empty payload."""
    assert _default_scoring_period(
        {"status": {"firstScoringPeriod": 1, "latestScoringPeriod": 1, "finalScoringPeriod": 17}}
    ) == (1, 1)
    # No finalScoringPeriod at all -> nothing to clamp against.
    assert _default_scoring_period(
        {"status": {"firstScoringPeriod": 1, "latestScoringPeriod": 5}}
    ) == (5, 1)
    # ...and never a week before the season starts, whatever `status` says.
    assert _default_scoring_period(
        {"status": {"firstScoringPeriod": 4, "latestScoringPeriod": 0, "finalScoringPeriod": 17}}
    ) == (4, 4)
    assert _default_scoring_period({}) == (1, 1)


class _StubClient:
    """Just enough of EspnClient, recording which weeks were actually requested."""

    def __init__(self, settings, boxscores):
        self._settings = settings
        self._boxscores = boxscores
        self.weeks_requested: list[int] = []

    def get(self, url, params=None, **kwargs):
        params = params or {}
        if params.get("view") == "mSettings":
            return self._settings, {}
        week = int(params["scoringPeriodId"])
        self.weeks_requested.append(week)
        return self._boxscores.get(week, {"schedule": []}), {}

    def close(self):
        raise AssertionError("verify_scoring_reproduction closed a client it does not own")


def settings_payload(items=IDP_PPFD_ITEMS, **status) -> dict:
    base = {"firstScoringPeriod": 1, "latestScoringPeriod": 19, "finalScoringPeriod": 17}
    return {
        "settings": {"scoringSettings": {"scoringItems": items}},
        "status": {**base, **status},
    }


def test_verify_walks_back_until_it_finds_a_populated_week():
    client = _StubClient(settings_payload(), {15: boxscore(entries_for_week(15), week=15)})
    report = verify_scoring_reproduction(2025, 899513, client=client)

    assert client.weeks_requested == [17, 16, 15]
    assert report.scoring_period_id == 15
    assert report.checked == len(REAL_ENTRIES)
    assert report.exact


def test_verify_stops_at_max_lookback_and_reports_a_failure():
    client = _StubClient(settings_payload(), {})
    report = verify_scoring_reproduction(2025, 899513, client=client)

    assert client.weeks_requested == [17, 16, 15, 14]  # max_lookback=3 -> four tries
    assert report.checked == 0
    assert not report.exact
    with pytest.raises(ScoringReproductionError):
        report.raise_if_failed()


def test_verify_never_asks_for_a_week_before_the_season_starts():
    client = _StubClient(settings_payload(latestScoringPeriod=2, finalScoringPeriod=0), {})
    verify_scoring_reproduction(2025, 899513, client=client)
    assert client.weeks_requested == [2, 1]


def test_verify_with_an_explicit_week_does_not_walk_back():
    client = _StubClient(settings_payload(), {})
    report = verify_scoring_reproduction(2025, 899513, 9, client=client)
    assert client.weeks_requested == [9]
    assert not report.exact


# ---------------------------------------------------------------------------
# League shape
# ---------------------------------------------------------------------------


def shape(**settings) -> LeagueShape:
    base = {
        "name": "Test",
        "size": 12,
        "scoringSettings": {"scoringType": "H2H_POINTS", "scoringItems": HALF_PPR_TE_PREMIUM_ITEMS},
        "rosterSettings": {"lineupSlotCounts": {}, "positionLimits": {}},
        "draftSettings": {},
        "acquisitionSettings": {},
        "scheduleSettings": {},
    }
    for key, value in settings.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return LeagueShape.from_settings(base)


def test_te_premium_is_te_minus_wr_not_te_minus_points():
    s = shape()
    assert s.points_per_reception == 0.5
    assert s.te_premium == 0.5
    assert s.has_te_premium
    # The trap: `points` is 0.0 here, so ovr[4] - points would claim a 1.0 premium.
    scoring = LeagueScoring.from_settings(
        {"scoringSettings": {"scoringItems": HALF_PPR_TE_PREMIUM_ITEMS}}
    )
    assert scoring.points_for(53, POS_TE) - scoring.items["53"].points == 1.0
    assert scoring.points_for(53, POS_TE) - scoring.points_for(53, POS_WR) == 0.5


def test_flat_ppr_has_no_premium_and_a_league_without_receptions_scores_zero():
    flat = shape(scoringSettings={"scoringItems": [{"statId": 53, "points": 1.0}]})
    assert flat.points_per_reception == 1.0
    assert flat.te_premium == 0.0
    assert not flat.has_te_premium

    ppfd = shape(scoringSettings={"scoringItems": IDP_PPFD_ITEMS})
    assert ppfd.points_per_reception == 0.0
    assert ppfd.points_per_receiving_first_down == 0.5


def test_faab_comes_from_is_using_acquisition_budget_not_acquisition_type():
    """`acquisitionType` is the waiver processing model and never implies money."""
    faab = shape(
        acquisitionSettings={
            "isUsingAcquisitionBudget": True,
            "acquisitionBudget": 100,
            "acquisitionType": "WAIVERS_TRADITIONAL",
        }
    )
    assert faab.uses_faab and faab.faab_budget == 100

    rolling = shape(
        acquisitionSettings={
            "isUsingAcquisitionBudget": False,
            "acquisitionBudget": 100,
            "acquisitionType": "WAIVERS_CONTINUOUS",
        }
    )
    assert not rolling.uses_faab
    assert rolling.acquisition_type == "WAIVERS_CONTINUOUS"


def test_auction_detection_ignores_auction_budget():
    """`auctionBudget` is populated in snake leagues too, so it discriminates nothing."""
    snake = shape(draftSettings={"type": "SNAKE", "auctionBudget": 200})
    assert not snake.is_auction and snake.auction_budget == 200

    assert shape(draftSettings={"type": "AUCTION"}).is_auction
    # Older/ordinal serializations of the same enum.
    assert shape(draftSettings={"type": 4}).is_auction
    assert shape(draftSettings={"type": 1}).draft_type == "SNAKE"
    assert shape(draftSettings={}).draft_type == "UNKNOWN"


def test_keeper_vs_redraft():
    assert shape(draftSettings={"keeperCount": 0, "keeperCountFuture": 0}).is_redraft
    assert shape(draftSettings={"keeperCount": 2, "keeperCountFuture": 2}).is_keeper
    # A league that only starts keeping next year is already a keeper league.
    assert shape(draftSettings={"keeperCount": 0, "keeperCountFuture": 2}).is_keeper


def test_superflex_and_idp_read_lineup_slots_not_positions():
    assert shape(rosterSettings={"lineupSlotCounts": {"7": 1, "0": 1}}).is_superflex
    assert shape(rosterSettings={"lineupSlotCounts": {"0": 2}}).is_superflex
    assert not shape(rosterSettings={"lineupSlotCounts": {"0": 1, "23": 2}}).is_superflex
    # Slot 1 is TQB, which is not a second startable quarterback.
    assert not shape(rosterSettings={"lineupSlotCounts": {"1": 1}}).is_superflex

    # Slot 15 is DP (a defensive player) -- this league is IDP.
    assert shape(rosterSettings={"lineupSlotCounts": {"15": 1}}).is_idp
    assert shape(rosterSettings={"lineupSlotCounts": {"11": 2}}).is_idp
    # Position 15 is TQB. positionLimits is keyed by position, so it must not
    # be mistaken for a defensive slot.
    idp_lookalike = shape(
        rosterSettings={"lineupSlotCounts": {"1": 1, "23": 4}, "positionLimits": {"15": 2}}
    )
    assert not idp_lookalike.is_idp
    assert idp_lookalike.position_limits[15] == 2


def test_median_scoring_needs_the_exact_enhancement_type():
    assert shape(
        scoringSettings={"scoringEnhancementType": "WIN_BONUS_TOP_HALF"}
    ).has_median_scoring
    assert not shape(scoringSettings={"scoringEnhancementType": "NONE"}).has_median_scoring
    assert not shape(scoringSettings={"scoringEnhancementType": None}).has_median_scoring
    assert not shape().has_median_scoring


def test_schedule_settings_are_read_from_their_own_keys():
    """Two ints from the same dict, trivially swappable and never noticed again."""
    s = shape(scheduleSettings={"matchupPeriodCount": 14, "playoffTeamCount": 6})
    assert s.matchup_period_count == 14
    assert s.playoff_team_count == 6

    absent = shape()
    assert absent.matchup_period_count == 0
    assert absent.playoff_team_count == 0


def test_receiving_first_down_rate_is_read_at_wr_not_te():
    """A league can put its TE premium on first downs too; the headline rate is WR's."""
    s = shape(
        scoringSettings={
            "scoringItems": [
                {"statId": 53, "points": 0.5},
                {"statId": 213, "points": 0.0, "pointsOverrides": {"3": 0.5, "4": 1.0}},
            ]
        }
    )
    assert s.points_per_receiving_first_down == 0.5


def test_starting_slots_exclude_bench_and_ir():
    s = shape(
        rosterSettings={"lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "23": 1, "20": 7, "21": 2}}
    )
    assert s.starting_slot_counts == {0: 1, 2: 2, 4: 2, 23: 1}
    assert s.starters == 6


# ---------------------------------------------------------------------------
# Live acceptance test
# ---------------------------------------------------------------------------

# Public 2025 leagues, no auth. Found by scanning random league ids; there are more
# than three, so a new scoring shape is a one-line addition here.
#   1241838  half-PPR + 0.5 TE premium, keeper, auction
#    246497  flat half-PPR
#    899513  IDP + PPFD + TQB, no reception scoring
#    350313  half-PPR with three `isReverseItem: true` items (20, 72, 85)
#   1966012  full PPR with two `isReverseItem: true` items (20, 72)
PUBLIC_LEAGUES = [1241838, 246497, 899513, 350313, 1966012]

# The one live league that pins the isReverseItem reading, and the items it carries.
REVERSE_ITEM_LEAGUE = 350313
REVERSE_ITEM_STAT_IDS = {"20", "72", "85"}


@pytest.mark.network
@pytest.mark.parametrize("league_id", PUBLIC_LEAGUES)
def test_reproduces_espn_applied_total_on_a_real_boxscore(league_id):
    report = verify_scoring_reproduction(2025, league_id)
    assert report.checked > 0, report.summary()
    assert report.exact, report.summary()
    # D/ST is the position most likely to be scored wrong (one-hot bands, its own
    # override map), so "all rows matched" only means something if one was there.
    assert report.positions.get(POS_DST, 0) > 0, report.summary()
    assert len(report.positions) >= 4, report.summary()
    report.raise_if_failed()


@pytest.mark.network
def test_a_live_league_with_reverse_items_reproduces_exactly():
    """The canary for the one rule inferred rather than measured.

    If ESPN ever means something by `isReverseItem` beyond ranking metadata, this
    is the league where it shows up first -- the flagged items are the negative
    ones (interceptions thrown, fumbles lost), so a sign flip is a large, obvious
    divergence rather than a rounding difference.
    """
    with EspnClient() as client:
        settings = fetch_league_settings(client, 2025, REVERSE_ITEM_LEAGUE)
        scoring = LeagueScoring.from_settings(settings)
        report = verify_scoring_reproduction(2025, REVERSE_ITEM_LEAGUE, client=client)

    flagged = {item.stat_id for item in scoring.reverse_items}
    assert flagged == REVERSE_ITEM_STAT_IDS, flagged
    assert all(item.points < 0 for item in scoring.reverse_items)
    assert report.checked > 0, report.summary()
    assert report.exact, report.summary()


@pytest.mark.network
def test_startup_assertion_shape_holds_for_the_current_season():
    """Projection rows exist before Week 1 kicks off, so the canary works preseason."""
    with EspnClient() as client:
        season, _ = client.current_season_and_week()
        report = verify_scoring_reproduction(season, 1241838, client=client)
    assert report.checked > 0, report.summary()
    assert report.exact, report.summary()


@pytest.mark.network
def test_live_league_shape_detection():
    with EspnClient() as client:
        keeper = LeagueShape.from_settings(fetch_league_settings(client, 2025, 1241838))
        idp = LeagueShape.from_settings(fetch_league_settings(client, 2025, 899513))

    assert keeper.is_keeper and not keeper.is_redraft
    assert keeper.is_auction and keeper.auction_budget == 200
    assert keeper.uses_faab and keeper.faab_budget == 100
    assert keeper.points_per_reception == 0.5
    assert keeper.te_premium == 0.5
    assert not keeper.is_idp and not keeper.is_superflex

    assert idp.is_idp
    assert idp.is_redraft
    assert idp.points_per_reception == 0.0
    assert idp.points_per_receiving_first_down == 0.5
