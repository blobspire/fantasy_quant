"""Tests for the generated ESPN constants layer.

The offline fixture below is a hand-transcribed copy of the live position, slot
and pro-team tables. That is deliberate: it makes the pure-logic tests meaningful
without a network call, and the `network` test at the bottom reconciles it against
ESPN so drift shows up as a failure rather than as a wrong lineup in October.
"""

from __future__ import annotations

import json

import pytest

from fantasy_quant.espn.client import EspnError
from fantasy_quant.espn.constants import (
    FALLBACK_PRO_TEAM_ABBREV,
    MISSING_PRO_TEAM_IDS,
    ConstantsError,
    PlatformSettings,
    PositionId,
    ProTeamId,
    SlotId,
    cache_path,
    clear_memo,
    fallback_pro_team_abbrev,
    load_platform_settings,
    position_id,
    pro_team_id,
    read_cache,
    slot_id,
    write_cache,
)
from fantasy_quant.espn.endpoints import platform_settings_url

# defaultPositionId -> abbrev. Note 4 = TE and 15 = TQB.
FIXTURE_POSITIONS: dict[int, str] = {
    0: "POS0",
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    6: "POS6",
    7: "P",
    8: "POS8",
    9: "DT",
    10: "DE",
    11: "LB",
    12: "CB",
    13: "S",
    14: "HC",
    15: "TQB",
    16: "D/ST",
    17: "EDR",
    18: "BE",
}

# lineupSlotId -> (abbrev, eligible defaultPositionIds, starter, bench).
# Note 4 = WR and 15 = DP -- the same two numbers, different meanings.
FIXTURE_SLOTS: dict[int, tuple[str, list[int], bool, bool]] = {
    0: ("QB", [1], True, False),
    1: ("TQB", [15], True, False),
    2: ("RB", [2], True, False),
    3: ("RB/WR", [2, 3], True, False),
    4: ("WR", [3], True, False),
    5: ("WR/TE", [3, 4], True, False),
    6: ("TE", [4], True, False),
    7: ("OP", [1, 2, 3, 4], True, False),
    8: ("DT", [9], True, False),
    9: ("DE", [10], True, False),
    10: ("LB", [11], True, False),
    11: ("DL", [9, 10], True, False),
    12: ("CB", [12], True, False),
    13: ("S", [13], True, False),
    14: ("DB", [12, 13], True, False),
    15: ("DP", [9, 10, 11, 12, 13], True, False),
    16: ("D/ST", [16], True, False),
    17: ("K", [5], True, False),
    18: ("P", [7], True, False),
    19: ("HC", [14], True, False),
    20: ("BE", [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17], False, True),
    21: ("IR", [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17], False, False),
    22: ("INV", [], False, False),
    23: ("FLEX", [2, 3, 4], True, False),
    24: ("EDR", [17], True, False),
    25: ("ALL", [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17], False, False),
}

FIXTURE_BYE_WEEKS = {1: 11, 12: 5, 33: 13, 34: 8}

# A handful of statIds worth pinning by name, from RESEARCH.md's key list. The
# two flags are transcribed from the live 2026 payload, not invented: `derived`
# and `pointsScoringEligible` are near-inverses of each other and NEITHER marks
# the pre-computed buckets, so 47 looks exactly like 42 on both.
FIXTURE_STATS: dict[int, tuple[str, str | None, bool, bool]] = {
    # id: (abbrev, apiIdentifier, derived, pointsScoringEligible)
    0: ("PA", "passing.passingAttempts", False, True),
    3: ("PY", "passing.passingYards", False, True),
    22: ("PYPG", "passing.passingYardsPerGame", True, False),
    41: ("RECS", None, False, False),
    42: ("REY", "receiving.receivingYards", False, True),
    47: ("REY5", None, False, True),
    53: ("REC", "receiving.receptions", False, True),
    96: ("FR", "defensiveInterceptions.fumbleRecoveries", False, True),
    103: ("INTTD", "defensiveInterceptions.interceptionTouchdowns", False, True),
    213: ("REFD", "receiving.receivingFirstDowns", False, True),
}


def _payload(
    *,
    season: int = 2026,
    slots: dict[int, tuple[str, list[int], bool, bool]] | None = None,
    team_abbrev_case: str = "upper",
) -> dict:
    """Build a payload shaped exactly like the live one, from the tables above."""
    slots = FIXTURE_SLOTS if slots is None else slots

    def _abbrev(a: str) -> str:
        return a.upper() if team_abbrev_case == "upper" else a.title()

    return {
        "id": season,
        "settings": {
            "statSettings": {
                "stats": [
                    {
                        "id": sid,
                        "abbrev": abbrev,
                        "displayAbbrev": abbrev,
                        "description": f"stat {sid}",
                        "apiIdentifier": api,
                        "derived": derived,
                        "pointsScoringEligible": eligible,
                    }
                    for sid, (abbrev, api, derived, eligible) in FIXTURE_STATS.items()
                ]
            },
            "positions": [
                {
                    "id": pid,
                    "abbrev": abbrev,
                    "name": abbrev,
                    "apiIdentifiers": [abbrev],
                }
                for pid, abbrev in FIXTURE_POSITIONS.items()
            ],
            "lineupSlots": [
                {
                    "id": sid,
                    "abbrev": abbrev,
                    "name": abbrev,
                    "eligiblePositions": eligible,
                    "starter": starter,
                    "bench": bench,
                }
                for sid, (abbrev, eligible, starter, bench) in slots.items()
            ],
            "proTeams": [
                {
                    "id": tid,
                    "abbrev": _abbrev(abbrev),
                    "location": f"City {tid}",
                    "name": f"Team {tid}",
                    "byeWeek": FIXTURE_BYE_WEEKS.get(tid, 9 if tid else 0),
                }
                for tid, abbrev in FALLBACK_PRO_TEAM_ABBREV.items()
            ],
            "statIdToOverridePosition": {"96": 16, "103": 16},
            "types": {
                # ids start at 1 here; typeNames would be off by one.
                "transactionTypes": [
                    {"id": 1, "name": "TRADE_DECLINE", "abbrev": "DECL"},
                    {"id": 2, "name": "TRADE_PROPOSAL", "abbrev": "PROP"},
                    {"id": 8, "name": "WAIVER", "abbrev": "WAIV"},
                    {"id": 13, "name": "DRAFT", "abbrev": "DRFT"},
                ],
                # and here the first id is negative.
                "transactionStatusTypes": [
                    {"id": -1, "name": "FAILED_UNKNOWN", "abbrev": "UNKWN"},
                    {"id": 0, "name": "PENDING", "abbrev": "PEND"},
                    {"id": 1, "name": "EXECUTED", "abbrev": "EXEC"},
                ],
                "draftTypes": [
                    {"id": 0, "name": "OFFLINE"},
                    {"id": 1, "name": "SNAKE"},
                    {"id": 4, "name": "AUCTION"},
                ],
                "scoringTypes": [{"id": 1, "name": "H2H_POINTS"}],
                "rankTypes": [
                    {"id": 0, "name": "STANDARD"},
                    {"id": 1, "name": "PPR"},
                    {"id": 4, "name": "SUPERFLEX"},
                ],
                "playerStatusTypes": [
                    {"id": 1, "name": "FREEAGENT"},
                    {"id": 2, "name": "ONTEAM"},
                    {"id": 3, "name": "WAIVERS"},
                ],
                "acquisitionTypes": [
                    {"id": 0, "name": "FREEAGENCY"},
                    {"id": 1, "name": "WAIVERS_TRADITIONAL"},
                    {"id": 2, "name": "WAIVERS_CONTINUOUS"},
                ],
            },
            "typeNames": {
                "transactionTypes": ["TRADE_DECLINE", "TRADE_PROPOSAL", "WAIVER", "DRAFT"],
            },
        },
    }


@pytest.fixture
def settings() -> PlatformSettings:
    return PlatformSettings.from_payload(_payload())


class _ExplodingClient:
    """Any network call here is a bug: the cache should have satisfied the request."""

    def get(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("hit the network when a cached payload was available")


class _FailingClient:
    """Stands in for ESPN 404ing a season it has never run."""

    def get(self, *args: object, **kwargs: object) -> None:
        raise EspnError("404 from ESPN: private league, or no such league.")


# The literal body ESPN returns for `seasons/2026` with an unrecognized `view=`:
# HTTP 200, ten keys, no `settings`. Verified live; this is the trap RESEARCH.md
# warns about and the reason a payload is parsed before it is cached.
SKELETON_200 = {
    "abbrev": "FFL",
    "active": True,
    "currentScoringPeriod": {"id": 1},
    "display": True,
    "displayOrder": 1,
    "endDate": 1767225600000,
    "gameId": 1,
    "id": 2026,
    "name": "Fantasy Football",
    "startDate": 1751342400000,
}


class _ScriptedClient:
    """Returns queued bodies in order, counting calls. Nothing is mocked away."""

    def __init__(self, *bodies: dict) -> None:
        self.bodies = list(bodies)
        self.calls = 0

    def get(self, *args: object, **kwargs: object) -> tuple[dict, dict]:
        self.calls += 1
        return self.bodies[min(self.calls - 1, len(self.bodies) - 1)], {}


# ------------------------------------------------------------- the ID-space traps


def test_the_two_id_spaces_disagree_at_4_and_15(settings: PlatformSettings) -> None:
    assert settings.positions.abbrev(position_id(4)) == "TE"
    assert settings.slots.abbrev(slot_id(4)) == "WR"
    assert settings.positions.abbrev(position_id(15)) == "TQB"
    assert settings.slots.abbrev(slot_id(15)) == "DP"


def test_a_slot_id_cannot_be_used_as_a_position_id(settings: PlatformSettings) -> None:
    with pytest.raises(TypeError, match="keyed by PositionId"):
        settings.positions[slot_id(4)]
    with pytest.raises(TypeError, match="keyed by PositionId"):
        settings.positions.get(slot_id(4))


def test_a_position_id_cannot_be_used_as_a_slot_id(settings: PlatformSettings) -> None:
    with pytest.raises(TypeError, match="keyed by SlotId"):
        settings.slots[position_id(4)]
    with pytest.raises(TypeError, match="keyed by SlotId"):
        settings.slots.abbrev(position_id(23))


def test_the_error_names_the_collision(settings: PlatformSettings) -> None:
    with pytest.raises(TypeError, match="disagree at 4 and 15"):
        settings.slots[position_id(4)]


def test_a_raw_int_is_not_an_id(settings: PlatformSettings) -> None:
    # The whole point: you must say which space you are in.
    with pytest.raises(TypeError):
        settings.positions[4]
    with pytest.raises(TypeError):
        settings.slots[23]
    with pytest.raises(TypeError):
        settings.pro_teams[12]


def test_ids_from_different_spaces_never_compare_equal() -> None:
    assert PositionId(4) != SlotId(4)
    assert SlotId(15) != PositionId(15)
    assert ProTeamId(4) != PositionId(4)
    assert PositionId(4) == position_id("4")


def test_a_wrongly_typed_id_misses_even_on_a_hash_collision() -> None:
    # Wrapper types can hash alike; equality is what must not match, or a dict
    # lookup would silently return the wrong row.
    table = {PositionId(4): "TE"}
    assert SlotId(4) not in table
    with pytest.raises(KeyError):
        table[SlotId(4)]


def test_accepts_rejects_the_wrong_id_space(settings: PlatformSettings) -> None:
    flex = settings.slots[slot_id(23)]
    assert flex.accepts(position_id(2))
    with pytest.raises(TypeError):
        flex.accepts(slot_id(2))


# --------------------------------------------------------- flex/superflex derivation


def test_flex_slot_resolves_to_rb_wr_te(settings: PlatformSettings) -> None:
    flex = settings.slots[slot_id(23)]
    assert flex.abbrev == "FLEX"
    eligible = {settings.positions.abbrev(p) for p in flex.eligible_positions}
    assert eligible == {"RB", "WR", "TE"}
    assert flex.is_flex
    assert not flex.is_superflex


def test_superflex_slot_includes_qb(settings: PlatformSettings) -> None:
    op = settings.slots[slot_id(7)]
    assert op.abbrev == "OP"
    eligible = {settings.positions.abbrev(p) for p in op.eligible_positions}
    assert eligible == {"QB", "RB", "WR", "TE"}
    assert op.is_superflex
    assert [s.abbrev for s in settings.slots.superflex_slots] == ["OP"]


def test_superflex_is_derived_not_hardcoded_to_slot_7() -> None:
    # Hypothetical ESPN change: QB eligibility moves from OP to FLEX. Anything that
    # keys on the literal slot 7 gets this backwards.
    slots = dict(FIXTURE_SLOTS)
    slots[7] = ("OP", [2, 3, 4], True, False)
    slots[23] = ("FLEX", [1, 2, 3, 4], True, False)
    moved = PlatformSettings.from_payload(_payload(slots=slots))

    assert moved.slots[slot_id(23)].is_superflex
    assert not moved.slots[slot_id(7)].is_superflex
    assert [s.abbrev for s in moved.slots.superflex_slots] == ["FLEX"]


def test_superflex_finds_qb_by_abbrev_not_by_position_id_1() -> None:
    # The other half of "derived, not hardcoded": QB is looked up by abbrev, so
    # renumbering the QB *position* must not break superflex detection either.
    # Anything that assumes `defaultPositionId == 1` passes the slot-7 test above
    # and fails this one.
    payload = _payload()
    for pos in payload["settings"]["positions"]:
        if pos["id"] == 1:
            pos["id"] = 30
    for slot in payload["settings"]["lineupSlots"]:
        slot["eligiblePositions"] = [30 if p == 1 else p for p in slot["eligiblePositions"]]
    renumbered = PlatformSettings.from_payload(payload)

    assert renumbered.positions.abbrev(position_id(30)) == "QB"
    assert renumbered.positions.get(position_id(1)) is None
    assert renumbered.slots[slot_id(7)].is_superflex
    assert [s.abbrev for s in renumbered.slots.superflex_slots] == ["OP"]
    assert [s.abbrev for s in renumbered.slots.dedicated_qb_slots] == ["QB"]
    assert renumbered.slots.is_superflex_lineup({"0": 1, "7": 1, "20": 6}) is True


def test_bench_and_ir_are_not_flex(settings: PlatformSettings) -> None:
    # Both accept every position, so a naive "more than one eligible position"
    # rule would call the bench a flex slot.
    assert not settings.slots[slot_id(20)].is_flex
    assert not settings.slots[slot_id(21)].is_flex
    assert settings.slots[slot_id(20)].is_bench
    assert "BE" not in {s.abbrev for s in settings.slots.flex_slots}


def test_dedicated_qb_slot_is_found_by_eligibility(settings: PlatformSettings) -> None:
    assert [s.abbrev for s in settings.slots.dedicated_qb_slots] == ["QB"]


def test_slots_accepting_only_returns_starters(settings: PlatformSettings) -> None:
    abbrevs = [s.abbrev for s in settings.slots.slots_accepting(position_id(4))]
    assert abbrevs == ["WR/TE", "TE", "OP", "FLEX"]
    assert "BE" not in abbrevs


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"0": 1, "2": 2, "4": 3, "6": 1, "23": 1, "20": 6}, False),
        ({"0": 1, "7": 1, "2": 2, "4": 2, "20": 6}, True),
        ({"0": 2, "2": 2, "4": 3, "20": 6}, True),
        ({0: 1, 23: 2, 20: 6}, False),
    ],
)
def test_is_superflex_lineup(
    settings: PlatformSettings, counts: dict[str | int, int], expected: bool
) -> None:
    assert settings.slots.is_superflex_lineup(counts) is expected


def test_three_wr_slots_are_not_read_as_tight_ends(settings: PlatformSettings) -> None:
    # lineupSlotCounts key "4" is WR. Reading it in the position space would make
    # this a three-TE league, and pull the wrong replacement level all season.
    counts = {"0": 1, "2": 2, "4": 3, "6": 1, "23": 1, "20": 6}
    starters = {settings.slots.abbrev(slot_id(k)): v for k, v in counts.items() if int(k) != 20}
    assert starters == {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}


# ------------------------------------------------------------------- pro teams


def test_pro_team_table_matches_the_fallback(settings: PlatformSettings) -> None:
    live = {t.id.value: t.abbrev for t in settings.pro_teams}
    assert live == dict(FALLBACK_PRO_TEAM_ABBREV)


def test_pro_team_abbrevs_are_upper_cased_across_seasons() -> None:
    # ESPN returned "Atl"/"Bal"/"Hou" through 2025 and "ATL"/"BAL"/"HOU" in 2026.
    older = PlatformSettings.from_payload(_payload(season=2025, team_abbrev_case="title"))
    assert older.pro_teams.abbrev(pro_team_id(1)) == "ATL"
    assert older.pro_teams.abbrev(pro_team_id(33)) == "BAL"
    assert older.pro_teams.by_abbrev("hou").id == pro_team_id(34)


def test_pro_team_ids_31_and_32_do_not_exist(settings: PlatformSettings) -> None:
    assert not MISSING_PRO_TEAM_IDS & set(FALLBACK_PRO_TEAM_ABBREV)
    for missing in sorted(MISSING_PRO_TEAM_IDS):
        with pytest.raises(KeyError, match="does not exist"):
            settings.pro_teams[pro_team_id(missing)]
        with pytest.raises(KeyError, match="does not exist"):
            fallback_pro_team_abbrev(pro_team_id(missing))


def test_bye_weeks(settings: PlatformSettings) -> None:
    assert settings.pro_teams.bye_week(pro_team_id(12)) == 5
    # Team 0 is the free-agent pseudo-team; byeWeek 0 is "no bye", not week zero.
    assert settings.pro_teams.bye_week(pro_team_id(0)) is None
    assert not settings.pro_teams[pro_team_id(0)].is_real


def test_abbrev_falls_back_when_espn_drops_a_team() -> None:
    payload = _payload()
    teams = payload["settings"]["proTeams"]
    payload["settings"]["proTeams"] = [t for t in teams if t["id"] != 34]
    thin = PlatformSettings.from_payload(payload)

    assert thin.pro_teams.get(pro_team_id(34)) is None
    assert thin.pro_teams.abbrev(pro_team_id(34)) == "HOU"


def test_fallback_table_is_internally_sane() -> None:
    assert len(FALLBACK_PRO_TEAM_ABBREV) == 33
    assert all(a == a.upper() for a in FALLBACK_PRO_TEAM_ABBREV.values())
    assert len(set(FALLBACK_PRO_TEAM_ABBREV.values())) == 33
    assert max(FALLBACK_PRO_TEAM_ABBREV) == 34


# ----------------------------------------------------------------------- stats


def test_stat_lookup(settings: PlatformSettings) -> None:
    assert settings.stat_abbrev(53) == "REC"
    assert settings.stat(3).api_identifier == "passing.passingYards"
    assert settings.stat(103).description
    with pytest.raises(KeyError, match="no statId"):
        settings.stat(9999)


def test_derived_flag_is_not_the_multicount_guard(settings: PlatformSettings) -> None:
    # 47 ("every 5 receiving yards") is a pre-computed bucket that WILL double-count
    # if applied alongside 42, yet ESPN reports derived=false for it. `derived`
    # marks per-game rates. The real guard is "has a scoringItem".
    assert settings.stat(47).derived is False
    assert settings.stat(22).derived is True
    # `pointsScoringEligible` is the other flag someone reaches for, and its name
    # is actively misleading: bucket 47 and raw statId 42 are indistinguishable on
    # both flags, so applying "everything eligible" double-counts receiving yards.
    assert settings.stat(47).points_scoring_eligible is True
    assert settings.stat(42).points_scoring_eligible is True
    assert (settings.stat(47).derived, settings.stat(47).points_scoring_eligible) == (
        settings.stat(42).derived,
        settings.stat(42).points_scoring_eligible,
    )
    # Nor is `pointsScoringEligible` just `not derived`: statId 41 is neither.
    assert settings.stat(41).derived is False
    assert settings.stat(41).points_scoring_eligible is False


def test_stat_override_positions_are_position_ids(settings: PlatformSettings) -> None:
    override = settings.stat_override_position[96]
    assert override == position_id(16)
    assert settings.positions.abbrev(override) == "D/ST"
    # 16 is D/ST in both spaces, but the value is a position id and must stay one.
    assert override != slot_id(16)


# ----------------------------------------------------------------------- enums


def test_enums_come_from_types_not_typenames(settings: PlatformSettings) -> None:
    # typeNames is a flat, display-ordered list. Zipping it against indices would
    # make id 1 "TRADE_PROPOSAL" (it is index 1) instead of "TRADE_DECLINE".
    assert settings.transaction_types.name(1) == "TRADE_DECLINE"
    assert settings.transaction_types.name(13) == "DRAFT"
    assert settings.transaction_types.id("WAIVER") == 8


def test_enum_ids_can_be_negative(settings: PlatformSettings) -> None:
    assert settings.transaction_status_types.name(-1) == "FAILED_UNKNOWN"
    assert settings.transaction_status_types.id("EXECUTED") == 1


def test_the_named_enum_tables_are_all_present(settings: PlatformSettings) -> None:
    assert settings.draft_types.name(4) == "AUCTION"
    assert settings.scoring_types.name(1) == "H2H_POINTS"
    assert settings.rank_types.id("PPR") == 1
    assert settings.rank_types.name(4) == "SUPERFLEX"
    assert settings.player_status_types.id("FREEAGENT") == 1
    assert settings.acquisition_types.name(2) == "WAIVERS_CONTINUOUS"
    with pytest.raises(KeyError, match="no enum"):
        settings.enum("nonsenseTypes")


# --------------------------------------------------------------- payload parsing


def test_skeleton_payload_is_rejected() -> None:
    # An unrecognized `view=` returns HTTP 200 with a default skeleton, so a typo
    # is invisible from the status code. Fail loudly instead of half-parsing.
    with pytest.raises(ConstantsError, match="no `settings` object"):
        PlatformSettings.from_payload({"id": 2026, "gameId": 1})


def test_missing_tables_are_rejected() -> None:
    payload = _payload()
    payload["settings"]["proTeams"] = []
    with pytest.raises(ConstantsError, match="proTeams"):
        PlatformSettings.from_payload(payload)


def test_season_is_read_from_the_payload_when_not_supplied() -> None:
    assert PlatformSettings.from_payload(_payload(season=2024)).season == 2024


# ----------------------------------------------------------------------- cache


def test_cache_round_trips(tmp_path) -> None:
    clear_memo()
    payload = _payload()
    dest = write_cache(2026, payload, tmp_path)

    assert dest == cache_path(2026, tmp_path)
    assert dest.name == "platform_settings_2026.json"
    assert read_cache(2026, tmp_path) == payload

    doc = json.loads(dest.read_text())
    assert doc["season"] == 2026
    assert doc["url"] == platform_settings_url(2026)
    assert doc["fetched_at"]


def test_load_uses_the_cache_and_never_calls_espn(tmp_path) -> None:
    clear_memo()
    write_cache(2026, _payload(), tmp_path)
    loaded = load_platform_settings(2026, client=_ExplodingClient(), root=tmp_path)

    assert loaded.season == 2026
    assert loaded.slots.abbrev(slot_id(23)) == "FLEX"
    assert loaded.positions.abbrev(position_id(4)) == "TE"


def test_load_memoizes_per_root(tmp_path) -> None:
    clear_memo()
    write_cache(2026, _payload(), tmp_path)
    first = load_platform_settings(2026, root=tmp_path, offline=True)
    assert load_platform_settings(2026, root=tmp_path, offline=True) is first

    # A second root is a separate cache; the memo must not serve one for the other,
    # or a tmp_path fixture would leak into the real data/reference cache.
    other = tmp_path / "other"
    with pytest.raises(ConstantsError, match="offline=True"):
        load_platform_settings(2026, root=other, offline=True)

    write_cache(2026, _payload(season=2026), other)
    assert load_platform_settings(2026, root=other, offline=True) is not first

    clear_memo()
    assert load_platform_settings(2026, root=tmp_path, offline=True) is not first


def test_offline_load_without_a_cache_raises(tmp_path) -> None:
    clear_memo()
    with pytest.raises(ConstantsError, match="offline=True"):
        load_platform_settings(2026, root=tmp_path, offline=True)


def test_a_bare_espn_body_dropped_in_by_hand_still_loads(tmp_path) -> None:
    clear_memo()
    path = cache_path(2026, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_payload()))

    loaded = load_platform_settings(2026, client=_ExplodingClient(), root=tmp_path)
    assert loaded.stat_abbrev(53) == "REC"


def test_a_corrupt_cache_is_ignored_rather_than_crashing(tmp_path) -> None:
    clear_memo()
    path = cache_path(2026, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    assert read_cache(2026, tmp_path) is None
    with pytest.raises(ConstantsError, match="offline=True"):
        load_platform_settings(2026, root=tmp_path, offline=True)


def test_a_season_espn_never_ran_reports_the_season_not_a_league(tmp_path) -> None:
    # The client's 404 branch is worded for leagues; on a season URL that reads as
    # a permissions problem when it is really a bad year.
    clear_memo()
    with pytest.raises(ConstantsError, match="platform settings for season 2035"):
        load_platform_settings(2035, client=_FailingClient(), root=tmp_path)
    assert not cache_path(2035, tmp_path).exists()


def test_write_cache_leaves_no_temp_file(tmp_path) -> None:
    write_cache(2026, _payload(), tmp_path)
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["platform_settings_2026.json"]


def test_a_fetched_payload_is_written_to_the_cache(tmp_path) -> None:
    # Without this, the whole caching story is untested offline: the network test
    # that asserts the file exists is deselected in CI.
    clear_memo()
    client = _ScriptedClient(_payload())
    first = load_platform_settings(2026, client=client, root=tmp_path)

    assert client.calls == 1
    assert cache_path(2026, tmp_path).exists()
    assert read_cache(2026, tmp_path) == _payload()

    clear_memo()
    second = load_platform_settings(2026, client=_ExplodingClient(), root=tmp_path)
    assert second.season == first.season
    assert load_platform_settings(2026, root=tmp_path, offline=True).stat_abbrev(53) == "REC"


def test_a_200_skeleton_is_never_cached_and_does_not_poison_the_next_call(tmp_path) -> None:
    # ESPN answers an unrecognized `view=` with HTTP 200 and this ten-key body.
    # Caching it before parsing would make every later load fail from disk, with
    # the network never retried -- one bad response, broken forever.
    clear_memo()
    client = _ScriptedClient(SKELETON_200, _payload())

    with pytest.raises(ConstantsError, match="no `settings` object"):
        load_platform_settings(2026, client=client, root=tmp_path)
    assert not cache_path(2026, tmp_path).exists(), "an unparseable body was persisted"

    clear_memo()
    recovered = load_platform_settings(2026, client=client, root=tmp_path)
    assert client.calls == 2, "the poisoned cache was served instead of refetching"
    assert recovered.stat_abbrev(53) == "REC"
    assert cache_path(2026, tmp_path).exists()


def test_a_structurally_bad_cache_is_refetched_not_fatal(tmp_path) -> None:
    # Valid JSON, wrong content -- a skeleton left by an older build, or a
    # truncated file. `read_cache` only catches JSONDecodeError, so this has to
    # be handled where the payload is parsed.
    clear_memo()
    write_cache(2026, SKELETON_200, tmp_path)
    client = _ScriptedClient(_payload())

    assert load_platform_settings(2026, client=client, root=tmp_path).stat_abbrev(53) == "REC"
    assert client.calls == 1
    assert read_cache(2026, tmp_path) == _payload(), "the good body should replace the bad one"

    # Offline there is nothing to fall back on, but the reason must be reported.
    clear_memo()
    write_cache(2025, SKELETON_200, tmp_path)
    with pytest.raises(ConstantsError, match="unusable"):
        load_platform_settings(2025, root=tmp_path, offline=True)


def test_a_cache_holding_the_wrong_season_is_rejected(tmp_path) -> None:
    # Nothing but a filename ties a cache entry to a season, and the tables really
    # do differ year to year (bye weeks, abbrev case). Serving 2022's tables
    # labelled 2026 is silent and would misroute every bye lookup.
    clear_memo()
    write_cache(2026, _payload(season=2022), tmp_path)
    with pytest.raises(ConstantsError, match="unusable"):
        load_platform_settings(2026, root=tmp_path, offline=True)

    with pytest.raises(ConstantsError, match="season 2022, not 2026"):
        PlatformSettings.from_payload(_payload(season=2022), 2026)

    clear_memo()
    client = _ScriptedClient(_payload(season=2026))
    assert load_platform_settings(2026, client=client, root=tmp_path).season == 2026
    assert client.calls == 1


def test_the_parsed_tables_are_read_only(tmp_path) -> None:
    # One PlatformSettings is memoized per (root, season) and shared by every
    # caller in the process. A stray mutation anywhere would silently rewrite the
    # constants for everyone -- exactly the failure mode this module exists to stop.
    clear_memo()
    write_cache(2026, _payload(), tmp_path)
    loaded = load_platform_settings(2026, root=tmp_path, offline=True)

    # A read-only view refuses assignment with TypeError and has no mutators at
    # all, so both shapes count as "refused".
    for mutate in (
        lambda: loaded.stats.pop(53),
        lambda: loaded.stats.__setitem__(53, None),
        lambda: loaded.enums.pop("draftTypes"),
        lambda: loaded.stat_override_position.pop(96),
        lambda: loaded.positions._by_id.clear(),
        lambda: loaded.slots._by_id.clear(),
        lambda: loaded.pro_teams._by_id.clear(),
        lambda: loaded.pro_teams._by_abbrev.__setitem__("KC", None),
        lambda: FALLBACK_PRO_TEAM_ABBREV.__setitem__(1, "XXX"),
    ):
        with pytest.raises((TypeError, AttributeError)):
            mutate()

    assert load_platform_settings(2026, root=tmp_path, offline=True).stat_abbrev(53) == "REC"


# ------------------------------------------------------------------------- live


@pytest.mark.network
def test_live_payload_still_matches_our_fixture(tmp_path) -> None:
    clear_memo()
    live = load_platform_settings(2026, root=tmp_path)

    assert len(live.stats) == 235, "ESPN's stat dictionary changed size"
    assert live.stat_abbrev(53) == "REC"
    assert live.stat_abbrev(103) == "INTTD"

    assert {p.id.value: p.abbrev for p in live.positions} == FIXTURE_POSITIONS
    assert {
        s.id.value: (s.abbrev, sorted(p.value for p in s.eligible_positions)) for s in live.slots
    } == {sid: (abbrev, sorted(elig)) for sid, (abbrev, elig, _, _) in FIXTURE_SLOTS.items()}

    flex = live.slots[slot_id(23)]
    assert {live.positions.abbrev(p) for p in flex.eligible_positions} == {"RB", "WR", "TE"}
    assert live.slots[slot_id(7)].is_superflex
    assert [s.abbrev for s in live.slots.superflex_slots] == ["OP"]

    # The fixture's flag transcription, checked against ESPN rather than itself.
    for sid, (_abbrev, _api, derived, eligible) in FIXTURE_STATS.items():
        assert live.stat(sid).derived is derived, sid
        assert live.stat(sid).points_scoring_eligible is eligible, sid
    # 47-52 are the pre-computed "every N receiving yards" buckets. If either flag
    # ever separated them from raw statId 42, this module's docstring would be
    # wrong and the scoring layer could use it as the multi-count guard.
    for bucket in range(47, 53):
        assert live.stat(bucket).derived is live.stat(42).derived
        assert live.stat(bucket).points_scoring_eligible is live.stat(42).points_scoring_eligible

    # The cache was written as a side effect of the load.
    assert cache_path(2026, tmp_path).exists()


@pytest.mark.network
def test_fallback_pro_teams_match_live(tmp_path) -> None:
    clear_memo()
    live = load_platform_settings(2026, root=tmp_path)

    assert {t.id.value: t.abbrev for t in live.pro_teams} == dict(FALLBACK_PRO_TEAM_ABBREV)
    assert not MISSING_PRO_TEAM_IDS & {t.id.value for t in live.pro_teams}
    assert all(1 <= t.bye_week <= 18 for t in live.pro_teams if t.is_real)


@pytest.mark.network
def test_live_enum_ids_are_not_list_positions(tmp_path) -> None:
    clear_memo()
    live = load_platform_settings(2026, root=tmp_path)

    assert live.transaction_types.name(1) == "TRADE_DECLINE"
    assert live.transaction_status_types.name(-1) == "FAILED_UNKNOWN"
    for key in (
        "transactionTypes",
        "draftTypes",
        "scoringTypes",
        "rankTypes",
        "playerStatusTypes",
        "transactionStatusTypes",
        "acquisitionTypes",
    ):
        assert len(live.enum(key)) > 0


@pytest.mark.network
def test_prior_season_abbrevs_were_mixed_case_and_are_normalized(tmp_path) -> None:
    # 2025 really does return "Atl"/"Bal"; if a future re-pull loses the
    # normalization every uppercase join downstream breaks silently.
    clear_memo()
    old = load_platform_settings(2025, root=tmp_path)
    assert {t.id.value: t.abbrev for t in old.pro_teams} == dict(FALLBACK_PRO_TEAM_ABBREV)
