"""The ID spine, and the four traps that make a bad join look like a good one.

The offline fixtures are small but they are *real*: every name, id and team below
was copied out of a live `roster_2026.parquet` or `db_playerids.csv` on 2026-09-07,
including the collisions (three Marvin Harrisons, two active Justin Jeffersons).
Synthetic names would not reproduce the failures these tests exist to catch.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import httpx
import polars as pl
import pytest

from fantasy_quant.data.ids import (
    CBS,
    DST_BY_NFLVERSE,
    DST_TEAMS,
    ESPN,
    ESPN_PRO_TEAM_IDS,
    GSIS,
    SLEEPER,
    IdResolver,
    PlayerIds,
    _clean_gsis,
    _clean_id,
    _prepare_dynastyprocess,
    _prepare_roster,
    build_records,
    compact_name,
    ensure_cached,
    espn_pool_coverage,
    is_stale,
    normalize_name,
    normalize_position,
    normalize_team,
    resolve_dst,
    strip_suffix,
)

# --------------------------------------------------------------------------
# Fixtures -- real rows, including the real collisions
# --------------------------------------------------------------------------

ROSTER_ROWS = [
    # name, team, position, gsis, espn, sleeper, week
    ("Amon-Ra St. Brown", "DET", "WR", "00-0036963", "4374302", "7525", 1),
    ("Marvin Harrison Jr.", "ARI", "WR", "00-0039337", "4432708", "11632", 1),
    ("Ja'Marr Chase", "CIN", "WR", "00-0036900", "4362628", "7564", 1),
    ("Kenneth Walker III", "KC", "RB", "00-0037746", "4567048", "8155", 1),
    ("DK Metcalf", "PIT", "WR", "00-0035640", "4047650", "5045", 1),
    ("Audric Estimé", "DEN", "RB", "00-0039893", "4429025", "11557", 1),
    ("Chig Okonkwo", "WAS", "TE", "00-0037741", "4360635", "8110", 1),
    ("Josh Palmer", "BUF", "WR", "00-0036916", "4242546", "7600", 1),
    ("Marquise Brown", "PHI", "WR", "00-0035662", "4241372", "5122", 1),
    ("Bijan Robinson", "ATL", "RB", "00-0038542", "4430807", "8138", 1),
    ("Brian Robinson", "ATL", "RB", "00-0037746x", "4241474", "7611", 1),
    ("Michael Pittman", "PIT", "WR", "00-0036252", "4035687", "6770", 1),
    ("Travis Etienne", "JAX", "RB", "00-0036973", "4239996", "7526", 1),
    ("Tre Harris", "LAC", "WR", "00-0039915", "4685702", "12500", 1),
    ("Dorian Thompson-Robinson", "CLE", "QB", "00-0038579", "4367178", "9226", 1),
    ("Ray-Ray McCloud", "ATL", "WR", "00-0034426", "3116165", "4324", 1),
    # Two *active* Justin Jeffersons in 2026 -- a WR and a linebacker.
    ("Justin Jefferson", "MIN", "WR", "00-0036322", "4262921", "6794", 1),
    ("Justin Jefferson", "CLE", "LB", "00-0039999", "5150249", "13010", 1),
    ("Byron Murphy", "MIN", "DB", "00-0035683", "4038999", "5108", 1),
    ("Byron Murphy II", "SEA", "DL", "00-0039358", "4570040", "11618", 1),
    # The rostered Josh Johnson. Three more people share the name exactly; the
    # one ESPN's pool entry 4390717 means is the free-agent RB in DP_ROWS.
    ("Josh Johnson", "CIN", "QB", "00-0026300", "11394", "260", 1),
    # The only Charger named Williams, and not a receiver. See
    # `test_surname_bonus_needs_a_block_both_filters_survived`.
    ("Marcus Williams", "LAC", "DB", "00-0033894", "3122882", "4021", 1),
    # A 2026 rookie whose nflverse row carries a gsis id and nothing else. His
    # db_playerids row (below) has every other id but a *fabricated* gsis, so the
    # two rows share no join key at all.
    ("Mike Washington Jr.", "LV", "RB", "00-0040878", None, None, 1),
]

# db_playerids ships missing ids as the literal string "NA". These rows carry it
# verbatim, because that is the entire point of this fixture.
DP_ROWS = [
    # name, team, position, mfl, gsis, espn, sleeper, yahoo, draft_year
    ("Marvin Harrison Jr.", "ARI", "WR", "16382", "00-0039337", "4432708", "11632", "NA", "2024"),
    ("Marvin Harrison", "FA*", "WR", "3321", "NA", "NA", "NA", "NA", "1996"),
    ("Marvin Harrison", "IND", "WR", "1234", "NA", "NA", "NA", "NA", "1996"),
    ("Michael Pittman", "PIT", "WR", "13593", "00-0036252", "4035687", "6770", "32723", "2020"),
    ("Michael Pittman", "FA", "RB", "2211", "NA", "NA", "NA", "NA", "1998"),
    ("Tyreek Hill", "FA", "WR", "12783", "00-0033040", "3116406", "3321", "29399", "2016"),
    ("Nick Chubb", "FA", "RB", "13129", "00-0034845", "3128720", "4988", "31005", "2018"),
    ("Chigoziem Okonkwo", "WAS", "TE", "15142", "00-0037741", "4360635", "8110", "NA", "2022"),
    # The espn_id collision: a 2016 tight end sharing an id with a 2016 tight end.
    ("Kyle Carter", "FA", "TE", "12683", "00-0032606", "2582138", "NA", "NA", "2016"),
    ("David Morgan", "MIN", "TE", "12816", "00-0032430", "2582138", "NA", "NA", "2016"),
    # "WAS569019" is not a gsis id. db_playerids fills rookie gsis cells with an
    # ESB-style placeholder rather than "NA" -- five rows do it on the live file.
    ("Mike Washington Jr.", "LVR", "RB", "17482", "WAS569019", "4686658", "13305", "NA", "2026"),
    # A free-agent namesake of a rostered player, and unlike the Marvin Harrison
    # and Michael Pittman pairs above this one *does* carry an espn_id -- so no
    # id-based tiebreak can separate the two. ESPN's own pool lists both.
    ("Josh Johnson", "FA", "RB", "15502", "00-0036799", "4390717", "8051", "33859", "2020"),
    # Two free-agent Mike Williamses, neither with a team. Both are receivers.
    ("Mike Williams", "FA", "WR", "13154", "00-0033536", "3045138", "4068", "30120", "2017"),
    ("Mike Williams", "FA", "WR", "10777", "00-0027702", "13489", "NA", "NA", "2010"),
]


@pytest.fixture(scope="module")
def roster_df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "full_name": name,
                "team": team,
                "position": pos,
                "gsis_id": gsis,
                "espn_id": espn,
                "sleeper_id": sleeper,
                "week": week,
            }
            for name, team, pos, gsis, espn, sleeper, week in ROSTER_ROWS
        ]
    )


@pytest.fixture(scope="module")
def dp_df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "name": name,
                "team": team,
                "position": pos,
                "mfl_id": mfl,
                "gsis_id": gsis,
                "espn_id": espn,
                "sleeper_id": sleeper,
                "yahoo_id": yahoo,
                "draft_year": draft_year,
            }
            for name, team, pos, mfl, gsis, espn, sleeper, yahoo, draft_year in DP_ROWS
        ]
    )


@pytest.fixture(scope="module")
def resolver(roster_df: pl.DataFrame, dp_df: pl.DataFrame) -> IdResolver:
    return IdResolver(build_records(roster_df, dp_df))


# --------------------------------------------------------------------------
# Trap 1: the literal string "NA"
# --------------------------------------------------------------------------


def test_literal_na_is_not_an_id():
    """`bool("NA")` is True. That single fact is the fake-100%-coverage bug."""
    assert _clean_id("NA") is None
    assert _clean_id("N/A") is None
    assert _clean_id("") is None
    assert _clean_id("  ") is None
    assert _clean_id(None) is None
    assert _clean_id("4362628") == "4362628"
    assert _clean_id(4362628) == "4362628"
    # Some readers round-trip numeric id columns through float.
    assert _clean_id("4362628.0") == "4362628"


def test_naive_truthiness_would_report_fake_full_coverage(dp_df: pl.DataFrame):
    """Pins the miscount itself: a naive count says 100%, the truth is lower."""
    naive = sum(1 for value in dp_df["espn_id"] if value)
    real = sum(1 for value in dp_df["espn_id"] if _clean_id(value) is not None)
    truth = sum(1 for row in DP_ROWS if row[5] != "NA")
    assert naive == dp_df.height, "fixture no longer exercises the trap"
    assert naive / dp_df.height == 1.0
    assert real == truth
    assert real < naive


def test_na_never_reaches_a_record(resolver: IdResolver):
    for record in resolver._by_canonical.values():
        for source, value in record.ids.items():
            assert value.upper() not in {"NA", "N/A", ""}, f"{record.name} {source}={value!r}"


def test_na_ids_do_not_collide_into_one_player(resolver: IdResolver):
    """If "NA" were treated as a value, every NA-yahoo player would merge into one."""
    assert resolver.to_canonical("NA", ESPN) is None
    assert resolver.to_canonical("NA", "yahoo") is None
    harrison = resolver.to_canonical("4432708", ESPN)
    assert resolver.get(harrison).name == "Marvin Harrison Jr."


def test_a_fabricated_gsis_id_is_not_a_gsis_id():
    """db_playerids fills rookie gsis cells with an ESB-style token, not "NA".

    Real gsis ids are ``00-0`` plus six digits -- 2,953 of 2,953 nflverse roster
    rows conform. Letting ``WAS569019`` through makes it a canonical id, which
    manufactures a second copy of a player who already exists under his real one.
    """
    assert _clean_gsis("00-0036900") == "00-0036900"
    assert _clean_gsis("WAS569019") is None
    assert _clean_gsis("BAI173035") is None
    assert _clean_gsis("NA") is None
    assert _clean_gsis("00-003690") is None  # one digit short
    assert _clean_gsis("00-00369000") is None  # one digit long


def test_placeholder_gsis_never_enters_the_spine(resolver: IdResolver):
    assert resolver.get("WAS569019") is None
    assert resolver.to_canonical("WAS569019", GSIS) is None
    washington = resolver.record_for("4686658", ESPN)
    assert washington is not None
    assert GSIS not in washington.ids
    assert washington.canonical == "ESPN-4686658"


def test_a_split_player_resolves_to_the_half_that_can_reach_espn(resolver: IdResolver):
    """One rookie, two rows, no shared id -- nflverse has only his gsis.

    Nothing can safely merge them (name similarity is exactly what this module
    refuses to trust), so the name resolver picks the row that carries an ESPN id.
    Every join downstream keys on ESPN; the gsis-only row cannot answer anything.
    """
    match = resolver.resolve_name("Mike Washington Jr.", team="LV", position="RB")
    assert match is not None
    assert resolver.from_canonical(match.canonical, ESPN) == "4686658"
    # The nflverse half is still reachable by its own id -- nothing was dropped.
    assert resolver.get("00-0040878") is not None


def test_roster_rows_without_a_gsis_are_not_collapsed_into_one():
    """`unique` treats null as a value, so a plain dedupe merges every gsis-less row.

    Today's `roster_2026` has exactly one such row, which hides this completely.
    Mid-season the file carries several weeks and more of them, and each one still
    carries the espn/sleeper ids the ESPN join actually needs.
    """
    df = pl.DataFrame(
        [
            # Two different players, neither with a gsis id.
            {"full_name": "No Gsis One", "team": "ARI", "position": "WR", "gsis_id": None,
             "espn_id": "1", "sleeper_id": "a", "week": 1},
            {"full_name": "No Gsis Two", "team": "BUF", "position": "RB", "gsis_id": None,
             "espn_id": "2", "sleeper_id": "b", "week": 1},
            # ...and one player stamped in two weeks, traded in between.
            {"full_name": "Traded Guy", "team": "NYJ", "position": "WR", "gsis_id": "00-0000009",
             "espn_id": "9", "sleeper_id": "c", "week": 1},
            {"full_name": "Traded Guy", "team": "PIT", "position": "WR", "gsis_id": "00-0000009",
             "espn_id": "9", "sleeper_id": "c", "week": 5},
        ]
    )  # fmt: skip
    out = _prepare_roster(df)
    assert sorted(out["full_name"]) == ["No Gsis One", "No Gsis Two", "Traded Guy"]
    # The week stamp still decides which row of a traded player survives.
    traded = out.filter(pl.col("gsis_id") == "00-0000009")
    assert traded.height == 1
    assert traded["team"][0] == "PIT"

    resolver = IdResolver(build_records(df, None))
    assert resolver.record_for("1", ESPN).name == "No Gsis One"
    assert resolver.record_for("2", ESPN).name == "No Gsis Two"


def test_dynastyprocess_dedupe_keeps_the_more_recent_namesake(dp_df: pl.DataFrame):
    """13 espn_ids in the live file are shared by two unrelated players."""
    prepared = _prepare_dynastyprocess(dp_df)
    survivors = prepared.filter(pl.col("espn_id") == "2582138")
    assert survivors.height == 1
    # NA rows must survive the dedupe wholesale, not collapse to one.
    assert prepared.filter(pl.col("espn_id") == "NA").height == 3


# --------------------------------------------------------------------------
# Trap 2: team defenses
# --------------------------------------------------------------------------


def test_thirty_two_defenses_exactly():
    assert len(DST_TEAMS) == 32
    assert len({t.nflverse for t in DST_TEAMS}) == 32
    assert len({t.pro_team_id for t in DST_TEAMS}) == 32
    assert len({t.canonical for t in DST_TEAMS}) == 32


def test_espn_pro_team_ids_match_research():
    """RESEARCH.md's table, re-derived from `chui_default_platformsettings`."""
    expected = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34,
    }  # fmt: skip
    assert set(ESPN_PRO_TEAM_IDS) == expected
    assert 31 not in ESPN_PRO_TEAM_IDS, "31 has never existed"
    assert 32 not in ESPN_PRO_TEAM_IDS, "32 has never existed"
    assert 0 not in ESPN_PRO_TEAM_IDS, "0 is free agency, not a team"


@pytest.mark.parametrize("team", DST_TEAMS, ids=lambda t: t.nflverse)
def test_dst_resolves_from_every_platform_spelling(team):
    """Both directions, all 32, for every id space we actually join on."""
    forms: list[object] = [
        team.nflverse,
        team.espn_abbrev,
        team.sleeper,
        team.pro_team_id,
        team.espn_player_id,
        str(team.espn_player_id),
        team.display_name,  # KeepTradeCut / FantasyCalc
        f"{team.nickname} D/ST",  # ESPN's player-pool fullName
        team.nickname,
        team.canonical,
        team.nflverse.lower(),
        *team.aliases,
    ]
    for form in forms:
        assert resolve_dst(form) is team, f"{form!r} did not resolve to {team.nflverse}"

    # ...and back out again.
    assert DST_BY_NFLVERSE[team.nflverse] is team
    assert team.espn_player_id == -(16_000 + team.pro_team_id)


def test_dst_free_agency_and_nonsense_resolve_to_nothing():
    assert resolve_dst(0) is None
    assert resolve_dst(31) is None
    assert resolve_dst(32) is None
    assert resolve_dst("FA") is None
    assert resolve_dst("") is None
    assert resolve_dst(None) is None
    assert resolve_dst("Ashton Jeanty") is None


def test_relocations_and_platform_abbreviations_agree():
    """The spellings that actually differ between platforms."""
    assert resolve_dst("LAR").nflverse == "LA"  # ESPN and Sleeper say LAR
    assert resolve_dst("LA").nflverse == "LA"  # nflverse says LA
    assert resolve_dst("STL").nflverse == "LA"  # pre-2016
    assert resolve_dst("WSH").nflverse == "WAS"  # ESPN says WSH
    assert resolve_dst("WAS").nflverse == "WAS"  # nflverse and Sleeper say WAS
    assert resolve_dst("OAK").nflverse == "LV"  # pre-2020
    assert resolve_dst("LVR").nflverse == "LV"  # DynastyProcess
    assert resolve_dst("SD").nflverse == "LAC"  # pre-2017
    assert resolve_dst("JAC").nflverse == "JAX"  # DynastyProcess
    assert resolve_dst("GBP").nflverse == "GB"  # DynastyProcess
    assert resolve_dst("Las Vegas Raiders").nflverse == "LV"  # KeepTradeCut


def test_dst_round_trips_through_the_resolver(resolver: IdResolver):
    for team in DST_TEAMS:
        canonical = resolver.to_canonical(team.espn_player_id, ESPN)
        assert canonical == team.canonical
        assert resolver.from_canonical(canonical, ESPN) == str(team.espn_player_id)
        assert resolver.from_canonical(canonical, SLEEPER) == team.sleeper
        assert resolver.get(canonical).position == "DST"
        assert resolver.get(canonical).team == team.nflverse


def test_dst_is_never_fuzzy_matched(resolver: IdResolver):
    """Defenses go through the table or they do not resolve at all."""
    for name in ("Rams D/ST", "Los Angeles Rams", "LAR", "DST-LA"):
        match = resolver.resolve_name(name)
        assert match is not None and match.method == "dst"
        assert match.canonical == "DST-LA"

    # A defense-shaped query with a player position must not fall through to the
    # name matcher and pick up some hapless wide receiver.
    assert resolver.resolve_name("Rams D/ST", position="WR") is None
    # ...and a garbage defense name must not fuzzy-match a real defense.
    assert resolver.resolve_name("Los Angeles Rums", position="DST") is None


# --------------------------------------------------------------------------
# Trap 3: Sleeper is not an ESPN bridge
# --------------------------------------------------------------------------


def test_sleeper_is_a_target_not_a_bridge(resolver: IdResolver):
    """Sleeper ids resolve *to* the spine; nothing routes an ESPN join through it."""
    canonical = resolver.to_canonical("7564", SLEEPER)
    assert canonical == "00-0036900"
    assert resolver.from_canonical(canonical, ESPN) == "4362628"
    # Coverage is reported honestly, so the 25% figure is visible rather than
    # hidden behind a fallback chain.
    rates = {row.source: row.rate for row in resolver.spine_coverage()}
    assert rates[GSIS] >= rates[SLEEPER]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Amon-Ra St. Brown", "amon ra st brown"),
        ("Amon-Ra St.Brown", "amon ra st brown"),
        ("Ja'Marr Chase", "jamarr chase"),
        ("Ja’Marr Chase", "jamarr chase"),  # curly apostrophe
        ("Audric Estimé", "audric estime"),
        ("Tre' Harris", "tre harris"),
        ("Dorian Thompson-Robinson", "dorian thompson robinson"),
        ("  Bijan   Robinson ", "bijan robinson"),
        ("DeVonta Smith", "devonta smith"),
    ],
)
def test_normalize_name(raw: str, expected: str):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Kenneth Walker III", "kenneth walker"),
        ("Deebo Samuel Sr.", "deebo samuel"),
        ("Gardner Minshew II", "gardner minshew"),
        ("Stetson Bennett IV", "stetson bennett"),
        ("Michael Pittman", "michael pittman"),
    ],
)
def test_strip_suffix(raw: str, expected: str):
    assert strip_suffix(normalize_name(raw)) == expected


def test_strip_suffix_never_eats_a_two_token_name():
    """Guard against a surname that happens to look like a suffix."""
    assert strip_suffix("jimmy graham") == "jimmy graham"
    assert strip_suffix("mark v") == "mark v"


def test_compact_name_reconciles_initial_styles():
    assert compact_name("D.K. Metcalf") == compact_name("DK Metcalf") == "dkmetcalf"
    assert compact_name("A.J. Brown") == compact_name("AJ Brown")
    assert compact_name("Amon-Ra St. Brown") == "amonrastbrown"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LAR", "LA"),
        ("LA", "LA"),
        ("WSH", "WAS"),
        ("JAC", "JAX"),
        ("KCC", "KC"),
        (13, "LV"),
        ("13", "LV"),
        (0, None),  # ESPN free agency
        ("FA", None),
        ("FA*", None),
        (None, None),
        ("nonsense", None),
    ],
)
def test_normalize_team(raw: object, expected: str | None):
    assert normalize_team(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1, "QB"),
        (4, "TE"),  # defaultPositionId 4 is TE; lineupSlotId 4 is WR
        (16, "DST"),
        ("PK", "K"),
        ("DEF", "DST"),
        ("D/ST", "DST"),
        ("FB", "RB"),
        ("WR", "WR"),
        ("XX", None),
        ("? ", None),
        (None, None),
    ],
)
def test_normalize_position(raw: object, expected: str | None):
    assert normalize_position(raw) == expected


def test_position_and_slot_id_spaces_are_not_confused():
    """They collide at 4 and 15 -- the top source of silent scoring corruption."""
    assert normalize_position(4) == "TE"  # not WR, which is lineupSlotId 4
    assert normalize_position(16) == "DST"  # agrees in both spaces, by luck


# --------------------------------------------------------------------------
# Id resolution
# --------------------------------------------------------------------------


def test_canonical_prefers_gsis(resolver: IdResolver):
    assert resolver.to_canonical("4362628", ESPN) == "00-0036900"
    assert resolver.to_canonical(4362628, ESPN) == "00-0036900"


def test_translate_round_trips(resolver: IdResolver):
    assert resolver.translate("4362628", ESPN, SLEEPER) == "7564"
    assert resolver.translate("7564", SLEEPER, ESPN) == "4362628"
    assert resolver.translate("00-0036900", GSIS, ESPN) == "4362628"


def test_unknown_ids_return_none_rather_than_guessing(resolver: IdResolver):
    assert resolver.to_canonical("999999999", ESPN) is None
    assert resolver.to_canonical(None, ESPN) is None
    assert resolver.from_canonical("no-such-player", ESPN) is None


def test_dynastyprocess_fills_gaps_without_overwriting_nflverse(resolver: IdResolver):
    """nflverse is refreshed daily; db_playerids is not. Ties go to nflverse."""
    okonkwo = resolver.get("00-0037741")
    assert okonkwo.origin == "nflverse"
    assert okonkwo.name == "Chig Okonkwo"  # nflverse's spelling, not "Chigoziem"
    assert okonkwo.ids["mfl"] == "15142"  # ...but DynastyProcess's extra id


def test_dynastyprocess_only_players_still_resolve(resolver: IdResolver):
    """Free agents fall off a current-season roster file; the fallback catches them."""
    tyreek = resolver.record_for("3116406", ESPN)
    assert tyreek is not None
    assert tyreek.origin == "dynastyprocess"
    assert tyreek.canonical == "00-0033040"


# --------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_espn"),
    [
        # Hyphens, periods and the awkward interaction of both.
        ("Amon-Ra St. Brown", "4374302"),
        ("Amon-Ra St.Brown", "4374302"),
        ("Amon Ra St Brown", "4374302"),
        ("amon-ra st. brown", "4374302"),
        # Apostrophes, straight and curly.
        ("Ja'Marr Chase", "4362628"),
        ("Ja’Marr Chase", "4362628"),
        ("JaMarr Chase", "4362628"),
        ("Tre' Harris", "4685702"),
        ("Tre Harris", "4685702"),
        # Generational suffixes, present on one side or the other.
        ("Marvin Harrison Jr.", "4432708"),
        ("Marvin Harrison", "4432708"),
        ("Kenneth Walker III", "4567048"),
        ("Kenneth Walker", "4567048"),
        ("Michael Pittman Jr.", "4035687"),
        ("Travis Etienne Jr.", "4239996"),
        ("Ray-Ray McCloud III", "3116165"),
        # Initial styling.
        ("D.K. Metcalf", "4047650"),
        ("DK Metcalf", "4047650"),
        # Diacritics.
        ("Audric Estimé", "4429025"),
        ("Audric Estime", "4429025"),
        # Hyphenated surnames written either way.
        ("Dorian Thompson-Robinson", "4367178"),
        ("Dorian Thompson Robinson", "4367178"),
        # Given-name variants that no normalization rule can reach.
        ("Chigoziem Okonkwo", "4360635"),
        ("Joshua Palmer", "4242546"),
    ],
)
def test_fuzzy_matcher_on_known_hard_names(resolver: IdResolver, query: str, expected_espn: str):
    match = resolver.resolve_name(query)
    assert match is not None, f"{query!r} did not resolve"
    assert resolver.from_canonical(match.canonical, ESPN) == expected_espn


def test_ambiguous_name_refuses_rather_than_guessing(resolver: IdResolver):
    """2026 has an active WR and an active LB both named Justin Jefferson.

    A wrong join here is invisible downstream -- it just quietly moves points
    between two players. A missing join is loud. Prefer loud.
    """
    assert resolver.resolve_name("Justin Jefferson") is None
    assert resolver.resolve_name("Justin Jefferson", position="WR").canonical == "00-0036322"
    assert resolver.resolve_name("Justin Jefferson", team="MIN").canonical == "00-0036322"
    assert resolver.resolve_name("Justin Jefferson", "CLE", "LB").canonical == "00-0039999"


def test_suffix_is_not_stripped_into_a_collision(resolver: IdResolver):
    """Byron Murphy and Byron Murphy II are two different, active players."""
    plain = resolver.resolve_name("Byron Murphy")
    junior = resolver.resolve_name("Byron Murphy II")
    assert plain is not None and junior is not None
    assert plain.canonical != junior.canonical
    assert resolver.from_canonical(plain.canonical, ESPN) == "4038999"
    assert resolver.from_canonical(junior.canonical, ESPN) == "4570040"


def test_nickname_needs_full_context(resolver: IdResolver):
    """ "Hollywood Brown" shares nothing with "Marquise Brown" but the surname."""
    assert resolver.resolve_name("Hollywood Brown") is None
    match = resolver.resolve_name("Hollywood Brown", team="PHI", position="WR")
    assert match is not None
    assert match.method == "surname"
    assert resolver.from_canonical(match.canonical, ESPN) == "4241372"


def test_surnames_are_never_crossed(resolver: IdResolver):
    """The nastiest near-collision in the pool: Bijan and Brian Robinson, both ATL RBs.

    Their normalized names score 0.93 on a raw string ratio -- higher than several
    pairs that *are* the same person. Requiring an exact surname plus a margin over
    the runner-up is what keeps them apart.
    """
    bijan = resolver.resolve_name("Bijan Robinson", team="ATL", position="RB")
    brian = resolver.resolve_name("Brian Robinson", team="ATL", position="RB")
    assert resolver.from_canonical(bijan.canonical, ESPN) == "4430807"
    assert resolver.from_canonical(brian.canonical, ESPN) == "4241474"
    # A typo still lands on the right one, because the margin survives.
    typo = resolver.resolve_name("Bijon Robinson", team="ATL", position="RB")
    assert typo is not None and typo.canonical == bijan.canonical
    # A different surname is never a match, however close the given name is.
    assert resolver.resolve_name("Bijan Robertson") is None
    assert resolver.resolve_name("Nonexistent Personage") is None


def test_being_on_this_years_roster_does_not_break_a_namesake_tie(resolver: IdResolver):
    """Four people are named Josh Johnson. Only one is currently rostered.

    "Prefer the currently-rostered row" reads like a harmless tidy-up and is not:
    roster status is not evidence about *which* Josh Johnson a bare name meant.
    Applied as a filter it turned this into ``score=1.0, method="exact"`` on the
    Bengals quarterback -- a confident, unauditable, wrong answer -- while ESPN's
    own pool entry for the name is the free-agent running back.

    Both halves stay reachable the moment the caller supplies any context at all,
    which is the only thing that actually distinguishes them.
    """
    assert resolver.resolve_name("Josh Johnson") is None
    assert resolver.resolve_name("Josh Johnson", position="QB").canonical == "00-0026300"
    assert resolver.resolve_name("Josh Johnson", position="RB").canonical == "00-0036799"
    assert resolver.resolve_name("Josh Johnson", team="CIN").canonical == "00-0026300"


def test_the_espn_id_tiebreak_survives_and_is_the_one_doing_the_work(resolver: IdResolver):
    """Removing the roster tiebreak must not cost the namesakes it was credited with.

    "Michael Pittman Jr." is ambiguous against a 2000s-era running back, and
    "Hollywood Brown" needs a surname block of exactly one. Both are settled by
    the *ESPN id* tiebreak -- the retiree rows have no espn_id at all -- not by
    roster membership. That is why dropping the roster rule is free here and was
    not free for Josh Johnson.
    """
    pittman = resolver.resolve_name("Michael Pittman Jr.")
    assert pittman is not None
    assert resolver.from_canonical(pittman.canonical, ESPN) == "4035687"
    brown = resolver.resolve_name("Hollywood Brown", team="PHI", position="WR")
    assert brown is not None
    assert resolver.from_canonical(brown.canonical, ESPN) == "4241372"


def test_surname_bonus_needs_a_block_both_filters_survived(resolver: IdResolver):
    """ESPN says Mike Williams is a LAC WR. The spine says he has no team at all.

    `_filter` throws away a filter that would empty the block, so the team filter
    (LAC) survives, keeps only Marcus Williams -- a *defensive back* -- and then
    the position filter (WR) empties the block and is discarded. What is left is a
    block narrowed by team alone, and scoring it as a fully qualified
    (team, position) block awarded 0.85 + 0.15*0.2 = 0.88 to a match between
    "Mike" and "Marcus".

    The bonus now requires both filters to have actually survived, so this falls
    back to the plain fuzzy score (0.68) and refuses.
    """
    assert resolver.resolve_name("Mike Williams", team="LAC", position="WR") is None
    # The mechanism, not just the symptom: the block is not a full block.
    block, full = resolver._filter(resolver._by_surname["williams"], "LAC", "WR")
    assert [resolver.get(c).name for c in block] == ["Marcus Williams"]
    assert full is False
    # A block where both filters do survive still earns the bonus.
    _, full_brown = resolver._filter(resolver._by_surname["brown"], "PHI", "WR")
    assert full_brown is True
    assert resolver.resolve_name("Hollywood Brown", team="PHI", position="WR").method == "surname"


def test_two_active_namesakes_with_no_context_refuse(resolver: IdResolver):
    """Two free-agent Mike Williamses, identical on every context field."""
    assert resolver.resolve_name("Mike Williams") is None
    assert resolver.resolve_name("Mike Williams", position="WR") is None


def test_an_id_two_records_claim_resolves_to_nothing():
    """`_prepare_dynastyprocess` deduplicates espn/gsis/sleeper. Nothing else.

    Eleven cbs_ids, four pfr_ids, two fleaflicker_ids, one ktc_id and one
    rotowire_id on the live file are each shared by two unrelated players.
    First-writer-wins made `to_canonical` answer with whichever row was read
    first, while the *other* record still carried the same id -- so an id resolved
    to a person who did not own it. Refusing is the only honest answer, and both
    players stay reachable by the ids that are actually theirs.
    """

    def record(canonical: str, name: str, team: str) -> PlayerIds:
        return PlayerIds(
            canonical=canonical,
            name=name,
            team=team,
            position="DB",
            origin="nflverse",
            ids={GSIS: canonical, CBS: "2866970"},
        )

    records = [
        record("00-0040546", "Cobee Bryant", "KC"),
        record("00-0038136", "Coby Bryant", "SEA"),
    ]
    resolver = IdResolver(records)
    assert resolver.to_canonical("2866970", CBS) is None
    assert resolver.ambiguous_ids(CBS) == frozenset({"2866970"})
    assert resolver.to_canonical("00-0040546", GSIS) == "00-0040546"
    assert resolver.to_canonical("00-0038136", GSIS) == "00-0038136"


def test_every_id_a_record_carries_resolves_back_to_that_record(resolver: IdResolver):
    """The invariant the ambiguity check exists to hold, across the whole spine."""
    for record in resolver._by_canonical.values():
        for source, value in record.ids.items():
            resolved = resolver.to_canonical(value, source)
            assert resolved in (record.canonical, None), (
                f"{source}={value} is carried by {record.name} but resolves to {resolved}"
            )


def test_stale_context_degrades_the_match_rather_than_killing_it(resolver: IdResolver):
    """Sportsbook team fields lag trades. A wrong team must not lose the player."""
    match = resolver.resolve_name("Ja'Marr Chase", team="NYJ", position="WR")
    assert match is not None
    assert resolver.from_canonical(match.canonical, ESPN) == "4362628"


def test_name_resolution_is_cached(resolver: IdResolver):
    """Sportsbook feeds repeat the same few hundred names every refresh."""
    fresh = IdResolver(list(resolver._by_canonical.values()))
    assert fresh.name_cache_size == 0
    first = fresh.resolve_name("Amon-Ra St. Brown")
    assert fresh.name_cache_size == 1
    second = fresh.resolve_name("Amon-Ra St. Brown")
    assert second is first  # the cached object, not an equal one
    assert fresh.name_cache_size == 1
    # Misses are cached too, or a feed full of college players re-scans every time.
    assert fresh.resolve_name("Nobody At All") is None
    assert fresh.name_cache_size == 2
    assert fresh.resolve_name("Nobody At All") is None
    assert fresh.name_cache_size == 2
    # Context is part of the key.
    fresh.resolve_name("Amon-Ra St. Brown", team="DET")
    assert fresh.name_cache_size == 3


def test_name_cache_is_not_slower_than_the_scan(resolver: IdResolver):
    resolver.resolve_name("Amon-Ra St. Brown")
    start = time.perf_counter()
    for _ in range(5_000):
        resolver.resolve_name("Amon-Ra St. Brown")
    assert time.perf_counter() - start < 1.0


# --------------------------------------------------------------------------
# Coverage reporting
# --------------------------------------------------------------------------


def test_coverage_report_counts_honestly(resolver: IdResolver):
    rows = resolver.coverage_report({ESPN: ["4362628", "4374302", "999999", "NA", None]})
    assert len(rows) == 1
    row = rows[0]
    assert row.source == ESPN
    assert row.total == 5
    assert row.resolved == 2
    assert row.rate == pytest.approx(0.4)
    assert "999999" in row.missing_sample


def test_coverage_report_is_zero_safe(resolver: IdResolver):
    (row,) = resolver.coverage_report({ESPN: []})
    assert row.total == 0
    assert row.rate == 0.0


def test_spine_coverage_excludes_defenses(resolver: IdResolver):
    """Defenses would otherwise inflate the ESPN rate and hide real decay."""
    rows = {row.source: row for row in resolver.spine_coverage()}
    players = [r for r in resolver._by_canonical.values() if not r.is_dst]
    assert rows[ESPN].total == len(players)
    assert rows[ESPN].total == len(resolver) - 32


# --------------------------------------------------------------------------
# Cache staleness
# --------------------------------------------------------------------------


def test_is_stale(tmp_path: Path):
    missing = tmp_path / "nope.parquet"
    assert is_stale(missing)

    fresh = tmp_path / "fresh.parquet"
    fresh.write_bytes(b"x")
    assert not is_stale(fresh, dt.timedelta(hours=24))
    assert is_stale(fresh, dt.timedelta(seconds=-1))


def test_ensure_cached_leaves_a_fresh_file_alone(tmp_path: Path):
    dest = tmp_path / "roster.parquet"
    dest.write_bytes(b"cached")
    # A URL that would explode if it were ever fetched.
    ensure_cached("http://127.0.0.1:1/nope", dest, max_age=dt.timedelta(hours=24))
    assert dest.read_bytes() == b"cached"


def test_ensure_cached_serves_stale_rather_than_failing(tmp_path: Path):
    """GitHub being down must degrade the analysis, not stop it."""
    dest = tmp_path / "roster.parquet"
    dest.write_bytes(b"yesterday")
    ensure_cached("http://127.0.0.1:1/nope", dest, max_age=dt.timedelta(seconds=-1))
    assert dest.read_bytes() == b"yesterday"


def test_ensure_cached_raises_when_there_is_nothing_to_fall_back_to(tmp_path: Path):
    # Narrow enough that a TypeError or an AttributeError in `ensure_cached`
    # itself fails the test rather than satisfying it.
    with pytest.raises((httpx.HTTPError, OSError)):
        ensure_cached("http://127.0.0.1:1/nope", tmp_path / "absent.parquet")


def test_offline_mode_never_reaches_the_network(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ensure_cached("http://127.0.0.1:1/nope", tmp_path / "absent.csv", allow_download=False)


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_crosswalks_download_and_build(tmp_path: Path):
    resolver = IdResolver.load(cache_dir=tmp_path)
    assert len(resolver) > 8_000
    assert (tmp_path / "db_playerids.csv").exists()
    assert any(tmp_path.glob("roster_*.parquet"))

    # The "NA" trap, on the real file rather than a fixture.
    raw = pl.read_csv(tmp_path / "db_playerids.csv", infer_schema_length=0)
    naive = sum(1 for value in raw["espn_id"] if value) / raw.height
    assert naive == 1.0, "db_playerids stopped using literal NA; revisit _clean_id"
    real = sum(1 for value in raw["espn_id"] if _clean_id(value) is not None) / raw.height
    assert real < 0.8, f"naive coverage {naive:.1%} vs real {real:.1%}"

    # No id may resolve to a player other than the one whose record carries it.
    # The offline fixture cannot exercise this -- the collisions live in the
    # secondary id spaces of the real 12k-row file (11 cbs_ids, 4 pfr_ids, 2
    # fleaflicker_ids, 1 ktc_id, 1 rotowire_id, each shared by two unrelated
    # players), and `_prepare_dynastyprocess` only deduplicates espn/gsis/sleeper.
    crossed = [
        (record.name, source, value, resolver.get(resolved).name)
        for record in resolver._by_canonical.values()
        for source, value in record.ids.items()
        if (resolved := resolver.to_canonical(value, source)) not in (record.canonical, None)
    ]
    assert not crossed, f"ids resolving to the wrong player: {crossed[:10]}"

    # The primary join keys must be unambiguous outright, not merely consistent.
    for source in (ESPN, GSIS, SLEEPER):
        assert not resolver.ambiguous_ids(source), (
            f"{source} ids are now shared by two records: "
            f"{sorted(resolver.ambiguous_ids(source))[:10]}"
        )


@pytest.mark.network
def test_live_espn_pool_join_rate_is_above_the_floor():
    """The degradation canary. Measured 100.0% overall on 2026-09-07.

    Floors are set well under the measured value so this fires on real breakage
    (an ESPN id-space change, a crosswalk that stopped refreshing) rather than on
    the normal churn of a few deep-bench free agents.
    """
    resolver = IdResolver.load()
    overall, skill = espn_pool_coverage(resolver, limit=600)

    assert overall.total >= 500, "ESPN returned a suspiciously small pool"
    assert overall.rate >= 0.90, (
        f"ESPN join rate fell to {overall.rate:.1%} "
        f"({overall.total - overall.resolved} unmatched, e.g. {overall.missing_sample})"
    )
    assert skill.rate >= 0.92, (
        f"skill-position join rate fell to {skill.rate:.1%}; sample {skill.missing_sample}"
    )


@pytest.mark.network
def test_live_name_matcher_agrees_with_espns_own_ids():
    """The strongest check available: ESPN labels its own pool.

    Each entry gives a name *and* an id. Resolving the name alone and comparing
    against the canonical the id resolves to turns the whole live pool into a
    labelled test set for the fuzzy matcher. Measured 2026-09-07: 600/600 correct
    with team and position context, 594/600 and zero wrong with no context.

    The no-context figure was 598/600 before the "prefer the currently-rostered
    row" tiebreak was removed. Those four extra hits were not free: at pool depth
    1,036 the same tiebreak produced three *wrong* matches, including a Josh
    Johnson delivered at ``score=1.0, method="exact"``. Trading four unresolved
    names for zero cross-wired players is the whole point of this module.

    A wrong match matters far more than an unresolved one -- it moves points
    between two real players and nothing downstream can see it -- so the ceiling
    on wrong matches is much tighter than the floor on resolved ones.
    """
    from fantasy_quant.espn.client import EspnClient
    from fantasy_quant.espn.endpoints import league_default_url

    resolver = IdResolver.load()
    with EspnClient() as client:
        season, _ = client.current_season_and_week()
        entries = client.player_pool(
            league_default_url(season, "ppr"),
            limit=250,
            params={"view": "kona_player_info"},
            max_players=600,
        )

    correct = 0
    wrong: list[tuple[str, str, str]] = []
    unresolved: list[str] = []
    for entry in entries:
        truth = resolver.to_canonical(entry["id"], ESPN)
        if truth is None:
            continue  # counted by the coverage test, not this one
        player = entry["player"]
        match = resolver.resolve_name(
            player["fullName"], player.get("proTeamId"), player.get("defaultPositionId")
        )
        if match is None:
            unresolved.append(player["fullName"])
        elif match.canonical == truth:
            correct += 1
        else:
            wrong.append((player["fullName"], truth, match.canonical))

    total = correct + len(wrong) + len(unresolved)
    assert total >= 500, "ESPN returned a suspiciously small pool"
    assert len(wrong) / total <= 0.01, f"name matcher crossed players: {wrong[:10]}"
    assert correct / total >= 0.95, f"only {correct}/{total} resolved; missed {unresolved[:10]}"


@pytest.mark.network
def test_live_espn_dst_ids_match_the_hard_coded_table():
    """Pins `-16{proTeamId:03d}`, which is how we address a defense in filterIds."""
    from fantasy_quant.espn.client import EspnClient
    from fantasy_quant.espn.endpoints import league_default_url

    with EspnClient() as client:
        season, _ = client.current_season_and_week()
        entries = client.player_pool(
            league_default_url(season, "ppr"),
            limit=40,
            params={"view": "kona_player_info"},
            extra_filter={"filterSlotIds": {"value": [16]}},
        )

    assert len(entries) == 32, f"ESPN listed {len(entries)} defenses, expected 32"
    for entry in entries:
        player = entry["player"]
        team = resolve_dst(entry["id"])
        assert team is not None, f"unmapped D/ST id {entry['id']} ({player['fullName']})"
        assert team.pro_team_id == player["proTeamId"]
        assert team.espn_player_id == entry["id"]
        assert resolve_dst(player["fullName"]) is team
        assert player["defaultPositionId"] == 16
