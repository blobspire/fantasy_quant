"""Fan API discovery and the league registry.

The Fan API's response shape is community-reported and could not be verified (we have no
credentials, and the endpoint 404s without them), so most of this file is the parser being
shown several *plausible* shapes and required to cope with all of them -- including one with
a nesting level removed and one with the wrapper renamed. If ESPN's real body looks like
none of these, `data/reference/fan_api_raw.json` is how a human finds out.

Offline throughout except the one endpoint-reachability check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fantasy_quant.espn.discovery import (
    DiscoveredLeague,
    DiscoveryError,
    discover_leagues,
    fan_api_url,
    fetch_fan_payload,
    manual_leagues,
    parse_fan_payload,
    verify_leagues,
)
from fantasy_quant.registry import (
    Credentials,
    LeagueConfig,
    Objective,
    Registry,
    RegistryDefaults,
    load_credentials,
    normalize_swid,
)

SWID = "{3A2C2DB2-7702-429A-8C0E-BC6C84DAA2EF}"
CREDS = Credentials(swid=SWID, espn_s2="not-a-real-cookie")


# --------------------------------------------------------------------------------------
# Plausible Fan API shapes
# --------------------------------------------------------------------------------------

#: The shape the community reports: preferences -> metaData -> entry -> groups.
CANONICAL: dict[str, Any] = {
    "id": SWID,
    "displayName": "justlikepudge",
    "preferences": [
        {
            "id": "pref-1",
            "typeId": 9,
            "metaData": {
                "entry": {
                    "entryId": 4,
                    "gameId": 1,
                    "seasonId": 2026,
                    "name": "Team Binish",
                    "groups": [{"groupId": 1241838, "groupName": "The Keeper League"}],
                }
            },
        },
        # A fantasy *basketball* team on the same account. gameId 1 is football.
        {
            "id": "pref-2",
            "typeId": 9,
            "metaData": {
                "entry": {
                    "entryId": 7,
                    "gameId": 46,
                    "seasonId": 2026,
                    "groups": [{"groupId": 99999, "groupName": "Hoops"}],
                }
            },
        },
        # A non-team preference: a followed athlete, no league anywhere.
        {"id": "pref-3", "typeId": 1, "metaData": {"athlete": {"id": 4685382}}},
    ],
}

#: A level removed: `metaData` is gone and the entry hangs straight off the preference.
FLATTENED: dict[str, Any] = {
    "preferences": [
        {
            "typeId": 9,
            "entry": {
                "entryId": 4,
                "gameId": 1,
                "seasonId": 2026,
                "groups": [{"groupId": 1241838, "groupName": "The Keeper League"}],
            },
        }
    ]
}

#: A level renamed: `preferences` -> `items`, `metaData` -> `payload`.
RENAMED: dict[str, Any] = {
    "items": [
        {
            "typeId": 9,
            "payload": {
                "fantasyEntry": {
                    "entryId": 4,
                    "gameAbbrev": "ffl",
                    "seasonId": 2026,
                    "groups": [{"groupId": 1241838, "groupName": "The Keeper League"}],
                }
            },
        }
    ]
}

#: No `groups` list at all -- the league id sits on the entry, ids arrive as strings.
NO_GROUPS: dict[str, Any] = {
    "preferences": [
        {
            "typeId": 9,
            "metaData": {
                "entry": {
                    "entryId": "4",
                    "leagueId": "1241838",
                    "gameId": "1",
                    "seasonId": "2026",
                }
            },
        }
    ]
}

#: Season only on an ancestor, and the same league described twice with the halves split
#: across the two records.
SPLIT_ACROSS_LEVELS: dict[str, Any] = {
    "seasonId": 2026,
    "gameId": 1,
    "preferences": [
        {"typeId": 9, "metaData": {"entry": {"groups": [{"groupId": 1241838}]}}},
        {
            "typeId": 9,
            "metaData": {
                "entry": {
                    "entryId": 4,
                    "groups": [{"groupId": 1241838, "groupName": "The Keeper League"}],
                }
            },
        },
    ],
}

ALL_SHAPES = {
    "canonical": CANONICAL,
    "flattened": FLATTENED,
    "renamed": RENAMED,
    "no_groups": NO_GROUPS,
}


def _fake_client(handler) -> Any:  # noqa: ANN001
    class _Fake:
        closed = False

        def get(self, url, params=None, fantasy_filter=None, use_etag=False):  # noqa: ANN001
            result = handler(url, dict(params or {}))
            if isinstance(result, Exception):
                raise result
            return result, {}

        def close(self) -> None:
            self.closed = True

    return _Fake()


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_SHAPES))
def test_every_plausible_shape_yields_the_same_league(name: str):
    """The shape is unverified, so the parser must not depend on any one nesting."""
    found = parse_fan_payload(ALL_SHAPES[name])
    assert [(dl.league_id, dl.season, dl.team_id) for dl in found] == [(1241838, 2026, 4)]


def test_group_name_is_carried_through_when_present():
    assert parse_fan_payload(CANONICAL)[0].name == "The Keeper League"
    assert parse_fan_payload(NO_GROUPS)[0].name is None


def test_other_fantasy_sports_are_filtered_out():
    """gameId 1 is football; the same account's basketball leagues share this endpoint.

    `typeId == 9` means "a fantasy team", not "a fantasy *football* team" -- the basketball
    preference in this fixture carries it too, so treating it as sufficient evidence pulls
    in league ids that 404 against the football API.
    """
    found = parse_fan_payload(CANONICAL)
    assert 99999 not in {dl.league_id for dl in found}
    assert [dl.league_id for dl in found] == [1241838]


def test_a_sport_abbrev_denies_football_even_when_the_type_id_says_fantasy_team():
    hockey = {
        "preferences": [
            {
                "typeId": 9,
                "metaData": {
                    "entry": {
                        "gameAbbrev": "fhl",
                        "seasonId": 2026,
                        "entryId": 3,
                        "groups": [{"groupId": 424242}],
                    }
                },
            }
        ]
    }
    assert parse_fan_payload(hockey) == []


def test_a_team_abbrev_never_denies_football():
    """`abbrev` is also a team's short name; reading it as a sport would drop real leagues."""
    payload = {
        "preferences": [
            {
                "typeId": 9,
                "metaData": {
                    "entry": {
                        "abbrev": "BINI",
                        "seasonId": 2026,
                        "entryId": 4,
                        "groups": [{"groupId": 1241838}],
                    }
                },
            }
        ]
    }
    assert [dl.league_id for dl in parse_fan_payload(payload)] == [1241838]


def test_season_is_inherited_from_an_ancestor_and_records_merge():
    found = parse_fan_payload(SPLIT_ACROSS_LEVELS)
    assert len(found) == 1
    assert found[0].season == 2026
    assert found[0].team_id == 4
    assert found[0].name == "The Keeper League"


def test_a_league_with_no_season_anywhere_is_skipped_not_guessed():
    """The registry keys on (league_id, season); an invented season poisons every join."""
    payload = {"preferences": [{"typeId": 9, "metaData": {"entry": {"groups": [{"groupId": 1}]}}}]}
    assert parse_fan_payload(payload) == []
    # ...unless the caller supplies one explicitly.
    assert parse_fan_payload(payload, season=2026)[0].season == 2026


def test_the_football_filter_can_be_relaxed_when_the_shape_drifts():
    """If gameId/typeId move, filtering on them reports zero leagues -- which is worse."""
    drifted = {"leagues": [{"groupId": 1241838, "seasonId": 2026, "entryId": 4}]}
    assert parse_fan_payload(drifted) == []
    relaxed = parse_fan_payload(drifted, require_football=False)
    assert [dl.league_id for dl in relaxed] == [1241838]
    assert relaxed[0].source == "fan_api"  # source only changes when discover() relabels it


def test_type_id_alone_is_enough_evidence_of_football():
    payload = {
        "preferences": [{"typeId": 9, "entry": {"seasonId": 2026, "groups": [{"groupId": 7}]}}]
    }
    assert [dl.league_id for dl in parse_fan_payload(payload)] == [7]


def test_garbage_payloads_yield_nothing_rather_than_raising():
    for payload in (None, [], {}, "nope", {"preferences": None}, {"a": [[[{"b": 1}]]]}):
        assert parse_fan_payload(payload) == []


def test_booleans_are_not_mistaken_for_ids():
    payload = {
        "preferences": [
            {"typeId": 9, "metaData": {"entry": {"seasonId": 2026, "groups": [{"groupId": True}]}}}
        ]
    }
    assert parse_fan_payload(payload) == []


# --------------------------------------------------------------------------------------
# Manual overrides
# --------------------------------------------------------------------------------------


def test_manual_leagues_are_first_class():
    found = manual_leagues([1241838, 305851, 1241838], season=2026, team_ids={305851: 3})
    assert [dl.league_id for dl in found] == [1241838, 305851]  # deduped, order kept
    assert all(dl.source == "manual" for dl in found)
    assert found[1].team_id == 3


def test_manual_ids_survive_a_dead_fan_api():
    """The unverified path failing must not take the known-good path down with it."""
    result = discover_leagues(
        Credentials(),  # no cookies at all
        manual_league_ids=[1241838],
        season=2026,
        raw_path=None,
    )
    assert not result.fan_api_ok
    assert "manual_league_ids" in result.reason
    assert [dl.league_id for dl in result] == [1241838]


def test_manual_ids_require_a_season():
    with pytest.raises(DiscoveryError, match="season"):
        discover_leagues(CREDS, manual_league_ids=[1], raw_path=None)


def test_manual_entries_win_without_discarding_what_only_discovery_knows(tmp_path: Path):
    """A manual id says which league to pull, not who we are in it.

    `manual_league_ids` is a bare id list, so a manual entry's team_id and name are always
    blank. Letting it win the whole record leaves `team_id=None` on a league we play in --
    the one field the Fan API exists to supply -- and every downstream lookup of "our
    roster" then has nothing to key on.
    """
    client = _fake_client(lambda url, params: CANONICAL)
    result = discover_leagues(
        CREDS,
        manual_league_ids=[1241838],
        season=2026,
        client=client,
        raw_path=tmp_path / "raw.json",
    )
    assert len(result) == 1
    entry = result.leagues[0]
    assert entry.source == "manual"  # provenance stays manual
    assert entry.team_id == 4  # ...but the discovered team id survives
    assert entry.name == "The Keeper League"
    # Merging must not resurrect a league the Fan API alone reported and the manual list
    # deliberately left out -- the merge fills fields, it does not add keys.
    assert [dl.key for dl in result] == [(1241838, 2026)]


# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------


def test_fan_api_url_forces_the_braces():
    assert fan_api_url(SWID).endswith(SWID)
    assert fan_api_url(SWID.strip("{}")).endswith(SWID)
    with pytest.raises(DiscoveryError):
        fan_api_url("")


def test_fetch_requires_both_cookies():
    with pytest.raises(DiscoveryError, match="ESPN_S2"):
        fetch_fan_payload(Credentials(swid=SWID))


def test_fetch_translates_the_clients_league_flavoured_errors():
    from fantasy_quant.espn.client import EspnError

    not_found = _fake_client(
        lambda url, params: EspnError("404 from ESPN (...): private league, or no such league.")
    )
    with pytest.raises(DiscoveryError, match="fan not found"):
        fetch_fan_payload(CREDS, client=not_found, dump_raw=False)

    unauthorized = _fake_client(lambda url, params: EspnError("401 from ESPN: stale"))
    with pytest.raises(DiscoveryError, match="espn_s2"):
        fetch_fan_payload(CREDS, client=unauthorized, dump_raw=False)


def test_raw_body_is_dumped_once_for_a_human_to_read(tmp_path: Path):
    raw = tmp_path / "reference" / "fan_api_raw.json"
    client = _fake_client(lambda url, params: CANONICAL)

    fetch_fan_payload(CREDS, client=client, raw_path=raw)
    assert json.loads(raw.read_text())["preferences"][0]["typeId"] == 9

    # Second call must not clobber the first capture.
    raw.write_text('{"edited": true}')
    fetch_fan_payload(CREDS, client=client, raw_path=raw)
    assert json.loads(raw.read_text()) == {"edited": True}


def test_discovery_falls_back_to_unfiltered_and_labels_the_result(tmp_path: Path, caplog):
    """Filter fields moved: report the leagues and shout, rather than reporting none."""
    drifted = {"leagues": [{"groupId": 1241838, "seasonId": 2026, "entryId": 4}]}
    client = _fake_client(lambda url, params: drifted)
    with caplog.at_level("WARNING"):
        result = discover_leagues(CREDS, client=client, raw_path=tmp_path / "raw.json")

    assert result.fan_api_ok
    assert [dl.league_id for dl in result] == [1241838]
    assert result.leagues[0].source == "fan_api_unfiltered"
    assert "drifted" in caplog.text


def test_discovery_does_not_send_credentials_anywhere_but_the_cookie_jar():
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(url, params):
        seen.append((url, params))
        return CANONICAL

    discover_leagues(CREDS, client=_fake_client(handler), raw_path=None)
    url, params = seen[0]
    assert params == {"featureFlags": "expandAthlete"}
    assert CREDS.espn_s2 not in url


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def test_verify_drops_unreadable_leagues_and_resolves_our_team():
    from fantasy_quant.espn.client import EspnError

    # Shaped like the real responses: `settings` carries its sub-blocks and each team
    # carries a `record`. A generic 200 payload has neither, and `League` rejects it --
    # a fixture that omits them is testing a response ESPN never sends.
    settings = {
        "settings": {
            "name": "The Keeper League",
            "size": 10,
            "scheduleSettings": {"matchupPeriodCount": 14, "playoffTeamCount": 6},
        },
        "status": {"firstScoringPeriod": 1, "finalScoringPeriod": 17},
    }
    teams = {
        "teams": [
            {"id": 1, "name": "Someone Else", "owners": ["{OTHER}"], "record": {"overall": {}}},
            {"id": 6, "name": "Team Binish", "owners": [SWID], "record": {"overall": {}}},
        ],
        "members": [],
    }

    def handler(url, params):
        if "/1241838" in url:
            return settings if params["view"] == "mSettings" else teams
        return EspnError("404 from ESPN: private league, or no such league.")

    candidates = [
        DiscoveredLeague(1241838, 2026, team_id=None),
        DiscoveredLeague(99999, 2026, team_id=7, source="fan_api_unfiltered"),
    ]
    good, bad = verify_leagues(_fake_client(handler), candidates, swid=SWID.lower())

    assert [dl.league_id for dl in good] == [1241838]
    assert good[0].team_id == 6  # resolved from the league itself, not the Fan API
    assert good[0].name == "The Keeper League"
    assert bad[0][0].league_id == 99999
    # 401 and 404 are distinguishable without cookies (measured), and the reason has to
    # say which: "add your cookies" and "fix this id" are different jobs.
    assert bad[0][1].startswith("404")
    assert "wrong id" in bad[0][1]


# --------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------


def test_credentials_never_render_their_values():
    creds = Credentials(swid=SWID, espn_s2="SUPERSECRET")
    assert "SUPERSECRET" not in repr(creds)
    assert "SUPERSECRET" not in str(creds)
    assert "SUPERSECRET" not in f"{creds}"
    assert repr(creds) == "Credentials(swid=set, espn_s2=set)"


def test_credentials_load_from_the_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ESPN_SWID", "3A2C2DB2-7702-429A-8C0E-BC6C84DAA2EF")
    monkeypatch.setenv("ESPN_S2", "  cookie-value  ")
    creds = load_credentials(env_file=tmp_path / "absent.env")
    assert creds.swid == SWID  # braces added back
    assert creds.espn_s2 == "cookie-value"
    assert creds.complete


def test_the_env_example_placeholder_reads_as_unset(monkeypatch, tmp_path: Path):
    """`.env.example` ships {XXXXXXXX-...}; taking it literally 404s the Fan API."""
    monkeypatch.setenv("ESPN_SWID", "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}")
    monkeypatch.setenv("ESPN_S2", "")
    creds = load_credentials(env_file=tmp_path / "absent.env")
    assert creds.swid is None
    assert creds.espn_s2 is None
    assert not creds.complete
    assert normalize_swid(None) is None


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def _registry() -> Registry:
    return Registry(
        defaults=RegistryDefaults(season=2026, objective=Objective.CHAMPIONSHIP),
        leagues={
            (1241838, 2026): LeagueConfig(
                league_id=1241838,
                season=2026,
                name="The Keeper League",
                team_id=6,
                notes="10-team keeper auction",
                tags=("home",),
            ),
            # The league that inverts the whole product's objective.
            (777, 2026): LeagueConfig(
                league_id=777,
                season=2026,
                name="High Stakes",
                objective=Objective.POINTS,
                scoring_variant="half_ppr",
            ),
            (1241838, 2025): LeagueConfig(
                league_id=1241838, season=2025, name="The Keeper League", enabled=False
            ),
        },
        manual_league_ids=(777, 1241838),
    )


def test_registry_round_trips_through_toml(tmp_path: Path):
    original = _registry()
    path = original.save(tmp_path / "config" / "leagues.toml")
    reloaded = Registry.load(path)

    assert reloaded.leagues == original.leagues
    assert reloaded.defaults == original.defaults
    assert reloaded.manual_league_ids == original.manual_league_ids
    # The header comments explaining `objective` must survive into the written file.
    assert "objective" in path.read_text()


def test_registry_round_trips_through_json(tmp_path: Path):
    original = _registry()
    path = original.save(tmp_path / "leagues.json")
    assert Registry.load(path).leagues == original.leagues


def test_registry_keys_on_league_and_season():
    reg = _registry()
    assert len(reg) == 3
    assert reg.get(1241838, 2026).team_id == 6
    assert reg.get(1241838, 2025).team_id is None
    assert reg.get(1241838, 2024) is None
    assert (777, 2026) in reg
    with pytest.raises(KeyError):
        reg.require(1241838, 2024)


def test_active_uses_the_default_season_and_skips_disabled():
    reg = _registry()
    assert [c.key for c in reg.active()] == [(777, 2026), (1241838, 2026)]
    assert [c.key for c in reg.for_season(2025)] == []
    assert [c.key for c in reg.for_season(2025, enabled_only=False)] == [(1241838, 2025)]


def test_points_league_inverts_the_objective():
    """Optimizing weekly win probability destroys value where the payout is points-for."""
    reg = _registry()
    championship = reg.require(1241838, 2026)
    points = reg.require(777, 2026)

    assert championship.objective_weights() == (1.0, 0.0)
    assert not championship.optimizes_points
    assert points.objective_weights() == (0.0, 1.0)
    assert points.optimizes_points


def test_hybrid_must_state_its_split():
    with pytest.raises(ValueError, match="points_weight"):
        LeagueConfig(league_id=1, season=2026, objective=Objective.HYBRID)

    hybrid = LeagueConfig(league_id=1, season=2026, objective="hybrid", points_weight=0.3)
    assert hybrid.objective_weights() == (0.7, 0.3)
    assert not hybrid.optimizes_points


def test_objective_is_validated_loudly():
    with pytest.raises(ValueError, match="unknown objective"):
        LeagueConfig(league_id=1, season=2026, objective="winning")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LeagueConfig(league_id=1, season=2026, points_weight=1.4)
    assert LeagueConfig(league_id=1, season=2026, objective="POINTS ").objective is Objective.POINTS


def test_defaults_apply_to_entries_that_omit_them():
    reg = Registry.from_toml(
        """
[defaults]
season = 2026
objective = "points"
scoring_variant = "half_ppr"

[[leagues]]
league_id = 1241838

[[leagues]]
league_id = 777
season = 2025
objective = "championship"
"""
    )
    inherited = reg.require(1241838, 2026)
    assert inherited.objective is Objective.POINTS
    assert inherited.scoring_variant == "half_ppr"
    assert reg.require(777, 2025).objective is Objective.CHAMPIONSHIP


def test_a_league_without_a_season_anywhere_is_rejected():
    with pytest.raises(ValueError, match="season"):
        Registry.from_toml("[[leagues]]\nleague_id = 1\n")


def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Registry.from_toml(
            "[defaults]\nseason = 2026\n[[leagues]]\nleague_id = 1\n[[leagues]]\nleague_id = 1\n"
        )


def test_missing_config_is_an_empty_registry_not_a_crash(tmp_path: Path):
    reg = Registry.load(tmp_path / "nope.toml")
    assert len(reg) == 0
    with pytest.raises(FileNotFoundError):
        Registry.load(tmp_path / "nope.toml", missing_ok=False)


def test_merge_discovered_fills_gaps_without_clobbering_human_choices():
    reg = _registry()
    discovered = [
        DiscoveredLeague(1241838, 2026, team_id=99, name="Renamed By ESPN"),
        DiscoveredLeague(555, 2026, team_id=2, name="Brand New"),
    ]
    added = reg.merge_discovered(discovered)

    existing = reg.require(1241838, 2026)
    assert existing.team_id == 6  # a configured team id is not second-guessed
    assert existing.name == "The Keeper League"
    assert existing.notes == "10-team keeper auction"

    assert [c.key for c in added] == [(555, 2026)]
    assert reg.require(555, 2026).objective is Objective.CHAMPIONSHIP


def test_merge_discovered_fills_a_blank_team_id():
    reg = Registry(leagues={(1, 2026): LeagueConfig(league_id=1, season=2026)})
    reg.merge_discovered([DiscoveredLeague(1, 2026, team_id=4, name="Found It")])
    assert reg.require(1, 2026).team_id == 4
    assert reg.require(1, 2026).name == "Found It"


def test_merge_discovered_overwrite_is_opt_in():
    reg = _registry()
    reg.merge_discovered(
        [DiscoveredLeague(1241838, 2026, team_id=99, name="Renamed")], overwrite=True
    )
    assert reg.require(1241838, 2026).team_id == 99
    assert reg.require(1241838, 2026).notes == "10-team keeper auction"  # still not clobbered


def test_toml_writer_escapes_quotes_in_names(tmp_path: Path):
    reg = Registry(
        defaults=RegistryDefaults(season=2026),
        leagues={(1, 2026): LeagueConfig(league_id=1, season=2026, name='The "Real" League')},
    )
    path = reg.save(tmp_path / "leagues.toml")
    assert Registry.load(path).require(1, 2026).name == 'The "Real" League'


@pytest.mark.parametrize(
    "name",
    ["carriage\rreturn", "null\x00byte", "vertical\x0btab", "bell\x07", "tab\there", "nl\nhere"],
)
def test_toml_writer_survives_control_characters_in_a_league_name(tmp_path: Path, name: str):
    """League names are user-entered text and TOML forbids a raw control character.

    Emitting one produces a file `tomllib` refuses to read, so `save()` succeeds and the
    next `load()` blows up on a config the tool wrote itself -- the writer's own docstring
    promises it raises rather than doing that.
    """
    reg = Registry(
        defaults=RegistryDefaults(season=2026),
        leagues={(1, 2026): LeagueConfig(league_id=1, season=2026, name=name)},
    )
    path = reg.save(tmp_path / "leagues.toml")
    assert Registry.load(path).require(1, 2026).name == name


def test_points_weight_cannot_silently_override_the_objective():
    """`objective_weights` reads points_weight first, so a stray one inverts the league.

    objective="points" with points_weight=0.0 used to come back as (1.0, 0.0): a league
    that pays for total points, optimized for win probability.
    """
    with pytest.raises(ValueError, match="hybrid"):
        LeagueConfig(league_id=1, season=2026, objective="points", points_weight=0.0)
    with pytest.raises(ValueError, match="hybrid"):
        LeagueConfig(league_id=1, season=2026, objective="championship", points_weight=0.5)


def test_a_default_points_weight_does_not_leak_into_non_hybrid_leagues():
    """`[defaults] points_weight` exists for the hybrid leagues in the file, not for all."""
    reg = Registry.from_toml(
        """
[defaults]
season = 2026
objective = "championship"
points_weight = 0.5

[[leagues]]
league_id = 1

[[leagues]]
league_id = 2
objective = "hybrid"
"""
    )
    plain = reg.require(1, 2026)
    assert plain.points_weight is None
    assert plain.objective_weights() == (1.0, 0.0)
    # ...while a league that really is hybrid still inherits the split.
    assert reg.require(2, 2026).objective_weights() == (0.5, 0.5)


def test_merge_discovered_does_not_hand_a_default_split_to_a_new_league():
    reg = Registry(defaults=RegistryDefaults(season=2026, points_weight=0.5))
    reg.merge_discovered([DiscoveredLeague(1, 2026, team_id=2, name="Found")])
    assert reg.require(1, 2026).objective_weights() == (1.0, 0.0)


# --------------------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_fan_api_rejects_an_unknown_swid():
    """The only thing we can confirm without credentials: the endpoint is up and cookie-gated."""
    creds = Credentials(swid="{00000000-0000-0000-0000-000000000000}", espn_s2="invalid")
    with pytest.raises(DiscoveryError, match="fan not found"):
        fetch_fan_payload(creds, dump_raw=False)


@pytest.mark.network
def test_private_and_nonexistent_leagues_are_distinguishable_without_cookies():
    """RESEARCH.md says they are not. Measured across six ids, they are.

    A real-but-private league answers 401 AUTH_LEAGUE_NOT_VISIBLE; an id that does not
    exist answers 404 GENERAL_NOT_FOUND.
    """
    from fantasy_quant.espn.client import EspnClient

    with EspnClient() as client:
        _, rejected = verify_leagues(
            client,
            [DiscoveredLeague(2000, 2026), DiscoveredLeague(123123123, 2026)],
        )

    reasons = {dl.league_id: reason for dl, reason in rejected}
    assert reasons[2000].startswith("401")
    assert reasons[123123123].startswith("404")
