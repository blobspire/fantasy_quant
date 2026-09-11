"""Report-layer tests.

Four layers, kept apart because they fail for different reasons.

*Selection and formatting* are pure functions over strings and dicts. A failure there is
a CLI-ergonomics failure and touches nothing else.

*Verdicts* are the module's whole reason to exist and get the most tests. The rule under
test is that each surface's OWN significance answer wins: `core.Recommendation.significant`
reads a zero standard error as certainty, which is exactly backwards for the three moves
that produce one (a claim worth nothing, an already-set lineup, a plan identical to
holding), and two sigma is the wrong threshold for a trade that won a search. Every one of
those cases is pinned here, because getting any of them wrong makes the report lie
confidently.

*Payload builders* run against a genuinely synthetic four-team league -- real
`LeagueState`, real `Draw`, real `TitleEngine`, real `decide/lineups` -- so a failure is a
wiring failure against the real surfaces and not against a mock of them. `decide/streaming`
is the one surface that is stubbed, because it fetches a betting market over the network
and this file is offline by construction.

*The CLI* is exercised through `typer.testing.CliRunner` with the payload builders
replaced, so those tests are about argument handling, JSON shape, and what happens when a
league cannot be fetched -- not about the analytics underneath.

Nothing here touches ESPN, and the whole file runs in a couple of seconds.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from fantasy_quant import pipeline as P
from fantasy_quant import registry as R
from fantasy_quant import report
from fantasy_quant.core import (
    Move,
    MoveKind,
    PlayerMove,
    PlayerOutlook,
    Recommendation,
    WeeklyOutlook,
)
from fantasy_quant.sim.distributions import WeeklySampler
from fantasy_quant.sim.season import Franchise, LeagueState, PlayerPool, ScheduledGame, panel_for

runner = CliRunner()


def _wide() -> Console:
    """A console wide enough that an assertion is about the text, not about wrapping."""
    return Console(width=200)


QB, RB, WR, TE, K, DST = 1, 2, 3, 4, 5, 16

#: ESPN's standard redraft start, which is what all three of the user's leagues run.
#: Slot ids, not position ids -- the two spaces collide at 4 and 15.
SLOT_COUNTS = {0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 23: 1}
SLOT_ELIGIBILITY = {
    0: frozenset({QB}),
    2: frozenset({RB}),
    4: frozenset({WR}),
    6: frozenset({TE}),
    16: frozenset({DST}),
    17: frozenset({K}),
    23: frozenset({RB, WR, TE}),
}
ROSTER_POSITIONS = (QB, RB, RB, WR, WR, TE, WR, DST, K)
_BASE_MEANS = (18.0, 14.0, 11.0, 13.0, 10.0, 9.0, 7.0, 8.0, 8.0)


# --------------------------------------------------------------------------------------
# A synthetic league
# --------------------------------------------------------------------------------------


def _outlook(pid: int, pos: int, team: int, means: dict[int, float], name: str) -> PlayerOutlook:
    """A hurdle-gamma outlook whose stated mean the sampler actually reproduces."""
    weeks = {}
    for week, mu in means.items():
        p_zero, shape = 0.05, 2.0
        scale = mu / ((1.0 - p_zero) * shape) if mu > 0 else 0.0
        second = (1.0 - p_zero) * shape * (shape + 1.0) * scale * scale
        weeks[week] = WeeklyOutlook(
            player_id=pid,
            season=2026,
            week=week,
            position_id=pos,
            mean=mu,
            sd=math.sqrt(max(second - mu * mu, 0.0)),
            p_zero=p_zero,
            shape=shape,
            scale=scale,
            pro_team_id=team,
            playing=mu > 0,
        )
    return PlayerOutlook(player_id=pid, name=name, position_id=pos, pro_team_id=team, weeks=weeks)


def _pairs(n: int, week: int) -> list[tuple[int, int]]:
    ids = list(range(n))
    for _ in range(week - 1):
        ids = [ids[0], ids[-1], *ids[1:-1]]
    return [(ids[i], ids[n - 1 - i]) for i in range(n // 2)]


@pytest.fixture(scope="module")
def sim() -> P.LeagueSim:
    """A real `pipeline.LeagueSim` over four hand-built teams and a stubbed `League`.

    A real one rather than a mock: `championship_table` calls `sim.simulate()`, the
    leverage section builds a real `TitleEngine` on `sim.draw`, and the lineup section
    runs the real solver. Only the ESPN half is a stub, because that is the only half
    that would need a network.
    """
    weeks = (1, 2, 3, 4)
    outlooks: list[PlayerOutlook] = []
    rosters: list[tuple[int, ...]] = []
    for t in range(4):
        ids = []
        for j, pos in enumerate(ROSTER_POSITIONS):
            pid = 1000 * (t + 1) + j
            ids.append(pid)
            means = dict.fromkeys(weeks, _BASE_MEANS[j])
            outlooks.append(_outlook(pid, pos, 10 * (t + 1) + j, means, f"P{pid}"))
        rosters.append(tuple(ids))
    pool = PlayerPool.of([(o.player_id, o.position_id, o.pro_team_id, o.name) for o in outlooks])
    games = tuple(
        ScheduledGame(matchup_period=w, weeks=(w,), home_team_id=a + 1, away_team_id=b + 1)
        for w in (1, 2, 3)
        for a, b in _pairs(4, w)
    )
    state = LeagueState(
        league_id=77,
        season=2026,
        name="Synthetic",
        franchises=tuple(
            Franchise(team_id=i + 1, name=f"T{i + 1}", player_ids=rosters[i], is_user=(i == 0))
            for i in range(4)
        ),
        pool=pool,
        weeks=weeks,
        remaining_games=games,
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=2,
        playoff_rounds=((4,),),
        my_team_id=1,
    )
    draw = WeeklySampler(panel_for(state, outlooks), seed=5).draw(200)
    record = SimpleNamespace(wins=1, losses=0, ties=0, points_for=110.0, points_against=98.0)
    teams = SimpleNamespace(
        teams=tuple(
            SimpleNamespace(id=i + 1, record=record, current_projected_rank=i + 1) for i in range(4)
        )
    )
    league = SimpleNamespace(
        teams=lambda: teams,
        rosters=lambda: {
            1: SimpleNamespace(
                starters=tuple(SimpleNamespace(player_id=p) for p in rosters[0] if p % 1000 != 6)
            )
        },
        settings=lambda: SimpleNamespace(
            acquisition=SimpleNamespace(uses_faab=False, budget=0),
            roster=SimpleNamespace(starter_count=9, bench_slots=7),
        ),
    )
    return P.LeagueSim(league=league, state=state, draw=draw, outlooks=outlooks, n_sims=200, seed=5)


@pytest.fixture
def workspace(sim, monkeypatch) -> report.Workspace:
    """A `Workspace` whose ESPN half is the synthetic league and never a socket."""
    monkeypatch.setattr(report.pipeline, "build", lambda *a, **k: sim)
    monkeypatch.setattr(
        report.pipeline, "client_from_env", lambda *a, **k: SimpleNamespace(close=lambda: None)
    )
    return report.Workspace(n_sims=200, seed=5)


_CFG = R.LeagueConfig(league_id=77, season=2026, name="Synthetic", team_id=1)


@pytest.fixture
def cfg() -> R.LeagueConfig:
    return _CFG


CONFIG_TOML = """
[defaults]
season = 2026
objective = "championship"

[[leagues]]
league_id = 111
season = 2026
name = "Wine Wednesday"
team_id = 1

[[leagues]]
league_id = 222
season = 2026
name = "Blacksburg Baddies"
team_id = 1

[[leagues]]
league_id = 333
season = 2026
name = "Wine Cellar"
team_id = 2
enabled = false
"""


@pytest.fixture
def registry_file(tmp_path):
    path = tmp_path / "leagues.toml"
    path.write_text(CONFIG_TOML)
    return path


# --------------------------------------------------------------------------------------
# League selection
# --------------------------------------------------------------------------------------


class TestSelection:
    def test_default_is_every_enabled_league(self, registry_file):
        reg = report.load_registry(registry_file)
        assert [c.league_id for c in report.select_leagues(reg)] == [111, 222]

    def test_by_id(self, registry_file):
        reg = report.load_registry(registry_file)
        assert [c.league_id for c in report.select_leagues(reg, ["222"])] == [222]

    def test_by_name_case_insensitively(self, registry_file):
        reg = report.load_registry(registry_file)
        assert [c.league_id for c in report.select_leagues(reg, ["blacksburg baddies"])] == [222]

    def test_a_prefix_beats_a_substring(self, registry_file):
        """`wine` is a prefix of two names and a substring of the same two; the exact
        match rule has to be tried before either or `--league wine` is ambiguous forever."""
        reg = report.load_registry(registry_file)
        with pytest.raises(report.ReportError, match="ambiguous"):
            report.select_leagues(reg, ["wine"])
        assert [c.league_id for c in report.select_leagues(reg, ["wine w"])] == [111]

    def test_a_disabled_league_is_still_selectable_by_name(self, registry_file):
        """Default means enabled; an explicit request means the user knows what he wants."""
        reg = report.load_registry(registry_file)
        assert [c.league_id for c in report.select_leagues(reg, ["Wine Cellar"])] == [333]

    def test_unknown_name_names_what_is_available(self, registry_file):
        reg = report.load_registry(registry_file)
        with pytest.raises(report.ReportError, match="Blacksburg"):
            report.select_leagues(reg, ["nonsense"])

    def test_ids_and_names_mix_and_deduplicate(self, registry_file):
        reg = report.load_registry(registry_file)
        picked = report.select_leagues(reg, ["111", "Wine Wednesday", "222"])
        assert [c.league_id for c in picked] == [111, 222]

    def test_an_empty_registry_says_so(self, tmp_path):
        with pytest.raises(report.ReportError, match="no leagues configured"):
            report.load_registry(tmp_path / "missing.toml")


# --------------------------------------------------------------------------------------
# Verdicts -- the heart of it
# --------------------------------------------------------------------------------------


def _rec(delta: float, stderr: float = 0.0, *, tags=(), confidence="medium") -> Recommendation:
    return Recommendation(
        move=Move(kind=MoveKind.WAIVER_CLAIM, league_id=77),
        delta_title=delta,
        delta_points=1.0,
        stderr=stderr,
        tags=tuple(tags),
        confidence=confidence,
    )


class TestVerdict:
    def test_an_effect_outside_two_errors_is_actionable(self):
        assert report.verdict_for(_rec(0.03, 0.005)).kind == "act"

    def test_an_effect_inside_two_errors_is_noise(self):
        v = report.verdict_for(_rec(0.005, 0.004))
        assert v.kind == "noise"
        assert not v.significant

    def test_an_exact_zero_is_never_significant(self):
        """`core.Recommendation.significant` calls this True: no error, therefore certain.

        It is the opposite. A claim below the wire floor, an already-optimal lineup and a
        plan identical to holding all produce two bit-identical simulated arms, so the
        delta and its error are both exactly zero and nothing was measured at all.
        """
        assert _rec(0.0, 0.0).significant is True
        assert report.verdict_for(_rec(0.0, 0.0)).kind == "null"

    def test_a_null_plan_is_null_even_with_a_delta(self):
        assert report.verdict_for(_rec(0.02, 0.0, tags=("null-plan",))).kind == "null"

    def test_a_streaming_hold_is_not_null(self):
        """`no-action-this-week` means the move is a hold, not that nothing was measured.

        The plan behind it is a real, simulated +2 to +4pp on all three live leagues.
        Folding the tag into `null` would throw that away.
        """
        v = report.verdict_for(_rec(0.028, 0.0035, tags=("no-action-this-week",)))
        assert v.kind == "act"

    def test_a_trade_is_judged_on_the_selection_adjusted_answer(self):
        """2.5 sigma passes `Recommendation.significant` and fails the search-winner test.

        Live numbers: the top Wine Wednesday trade is +1.03pp +/- 0.41. Two sigma calls
        that significant; `decide/trades.selection_threshold` needs 3.2 once the winner
        was picked out of fifty candidates, and encodes the answer in `confidence`.
        """
        rec = _rec(0.0103, 0.0041, confidence="low")
        assert rec.significant is True
        assert report.verdict_for(rec, surface="trade").kind == "noise"
        # The naive test, for contrast: same numbers, a surface that stands behind them.
        assert report.verdict_for(_rec(0.0103, 0.0041, confidence="medium")).kind == "act"

    def test_a_confident_trade_is_actionable(self):
        rec = _rec(0.0175, 0.0045, confidence="high")
        assert report.verdict_for(rec, surface="trade").kind == "act"

    def test_a_surface_that_rates_its_own_number_low_is_not_overruled(self):
        """The kicker case, and the one this used to get wrong on every surface but trades.

        `decide/streaming.py` sets `confidence="low"` with `not-streamable` when the
        fitted within-week spread at a position is under the floor -- K, at R^2 = 0.022,
        where its own docstring says a surface that dresses 0.68 points up as a
        recommendation is worse than no surface. The plan's PAIRED error is small, so the
        two-sigma test passes it happily: the live Type shi kicker plan is +0.85pp
        +/- 0.46 and a slightly tighter draw renders `act`. It must not.
        """
        rec = _rec(0.0085, 0.0035, confidence="low", tags=("streaming", "k", "not-streamable"))
        assert rec.significant is True  # the shared contract says yes
        v = report.verdict_for(rec, surface="stream")
        assert v.kind == "noise"
        assert "not-streamable" in v.note
        # And the same numbers from a surface that does stand behind them still act.
        assert report.verdict_for(_rec(0.0085, 0.0035), surface="stream").kind == "act"

    def test_a_measured_loss_is_not_called_noise(self):
        """`decide/trades._confidence` returns "low" for a trade that HURTS you too.

        Its own live example is -0.80pp +/- 0.31 -- 2.6 sigma of loss, confirmed. Folding
        that into `noise` prints "inside its own error" over a number the simulation
        resolved perfectly well, in the wrong direction.
        """
        v = report.verdict_for(_rec(-0.0080, 0.0031, confidence="low"), surface="trade")
        assert v.kind == "harm"
        assert "LOSS" in v.note
        assert not v.significant
        # A small negative inside its error is still just noise.
        assert report.verdict_for(_rec(-0.001, 0.004), surface="trade").kind == "noise"

    def test_markers_differ_by_kind(self):
        assert report.verdict_for(_rec(0.03, 0.005)).marker == ""
        assert report.verdict_for(_rec(0.005, 0.004)).marker == "~"
        assert report.verdict_for(_rec(0.0)).marker == "-"
        assert report.verdict_for(_rec(-0.03, 0.005)).marker == "!"


class TestRecPayload:
    def test_players_are_named_and_sided(self):
        move = Move(
            kind=MoveKind.TRADE,
            league_id=77,
            players=(
                PlayerMove(player_id=1, from_team=2, to_team=1),
                PlayerMove(player_id=2, from_team=1, to_team=2),
            ),
        )
        rec = Recommendation(move=move, delta_title=0.01, delta_points=3.0, stderr=0.001)
        body = report.rec_payload(rec, {1: "In", 2: "Out"}, team_id=1)
        assert [p["name"] for p in body["receive"]] == ["In"]
        assert [p["name"] for p in body["send"]] == ["Out"]
        assert body["z"] == pytest.approx(10.0)

    def test_an_unknown_player_falls_back_to_his_id(self):
        move = Move(
            kind=MoveKind.ADD_DROP,
            league_id=77,
            players=(PlayerMove(player_id=99, from_team=None, to_team=1),),
        )
        body = report.rec_payload(Recommendation(move=move, delta_title=0.0, delta_points=0.0), {})
        assert body["players"][0]["name"] == "99"

    def test_no_team_means_no_sides(self):
        """`receive`/`send` are relative to a team; without one they are empty, not wrong."""
        move = Move(
            kind=MoveKind.TRADE,
            league_id=77,
            players=(PlayerMove(player_id=1, from_team=2, to_team=1),),
        )
        body = report.rec_payload(Recommendation(move=move, delta_title=0.0, delta_points=0.0), {})
        assert body["receive"] == [] and body["send"] == []


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


class TestFormatting:
    def test_percentage_points_are_signed(self):
        assert report.pp(0.0175) == "+1.75pp"
        assert report.pp(-0.0175) == "-1.75pp"
        assert report.pp(None) == "-"

    def test_a_slot_is_named_from_what_it_accepts(self):
        """Slot ids and position ids collide at 4 and 15, so a name table is a bug table."""
        assert report.slot_label(4, SLOT_ELIGIBILITY) == "WR"
        assert report.slot_label(6, SLOT_ELIGIBILITY) == "TE"
        assert report.slot_label(23, SLOT_ELIGIBILITY) == "FLEX"
        assert report.slot_label(99, SLOT_ELIGIBILITY) == "99"

    def test_tag_values_are_read_off_the_recommendation(self):
        rec = _rec(0.01, tags=("waiver", "add:Chris Brooks", "pos:RB"))
        assert report.tag_value(rec, "add:") == "Chris Brooks"
        assert report.tag_value(rec, "drop:") == "-"

    def test_a_trade_summary_keeps_the_sentence_that_says_what_you_get(self):
        """`decide/trades` writes the pitch from the other side first, on purpose.

        One sentence of it is a queue row describing somebody else's trade.
        """
        pitch = "Rob gets A for B. You get C (+8.2 pts). Every side gains on its lineup."
        assert "You get C" in report._first_sentences(pitch, sentences=2, limit=220)
        assert "You get C" not in report._first_sentences(pitch)


# --------------------------------------------------------------------------------------
# Payload builders, against the synthetic league
# --------------------------------------------------------------------------------------


class TestPayloads:
    def test_odds_sum_to_one_and_name_the_user(self, workspace, cfg):
        payload = report.odds_payload(workspace, cfg)
        assert sum(t["championship"] for t in payload["teams"]) == pytest.approx(1.0, abs=1e-6)
        assert payload["my_rank"] is not None
        assert [t for t in payload["teams"] if t["is_me"]][0]["team_id"] == 1
        assert payload["teams"] == sorted(payload["teams"], key=lambda t: -t["championship"])

    def test_odds_carry_a_monte_carlo_error_and_say_which_ranks_are_real(self, workspace, cfg):
        """The one table here that used to be published with no error at all.

        It is the first thing `fq weekly` prints and it showed `12.95%` over `11.30%`
        over `11.10%` as a strict order. Re-running the live 14-team Wine Wednesday board
        under five seeds moved ranks 4 through 11 by up to three places each and flipped
        the user's own team between 13th and 14th, with the leader's probability ranging
        12.95-14.60%. The Bernoulli error is one line of arithmetic and was missing.
        """
        payload = report.odds_payload(workspace, cfg)
        n = payload["n_sims"]
        for row in payload["teams"]:
            p = row["championship"]
            assert row["championship_stderr"] == pytest.approx(math.sqrt(p * (1 - p) / n))
        assert payload["teams"][-1]["separated_from_next"] is None
        assert payload["n_ranks"] == len(payload["teams"]) - 1
        assert payload["n_ranks_separated"] <= payload["n_ranks"]
        assert str(payload["n_ranks_separated"]) in payload["ranking_note"]

    def test_four_identical_teams_have_no_separable_ranking(self, workspace, cfg):
        """The synthetic league is four copies of one roster, so the order is pure noise.

        An honest table says nothing is separated. This is the assertion that would fail
        if the separation test were written to always say yes.
        """
        payload = report.odds_payload(workspace, cfg)
        assert payload["n_ranks_separated"] == 0
        assert all(not t.get("separated_from_next") for t in payload["teams"])

    def test_the_separation_test_can_say_yes(self):
        """...and this is the one that would fail if it were written to always say no."""
        assert report._rank_separated(0.60, 0.10, 4000) is True
        # Ranks 2 and 3 on the live Wine Wednesday board: 11.30% over 11.10%, printed as
        # a strict order and 0.2pp apart against a 0.75pp paired error.
        assert report._rank_separated(0.1130, 0.1110, 4000) is False
        # More simulations resolve it; that is what --sims is for and what the note says.
        assert report._rank_separated(0.1130, 0.1110, 4_000_000) is True
        # Mutually exclusive champion indicators are NEGATIVELY correlated -- a sim that
        # crowns one takes the title from the other -- so the paired error is WIDER than
        # the independent one and the naive sqrt(se1^2 + se2^2) sets the bar too low.
        naive = math.hypot(report.champ_stderr(0.113, 4000), report.champ_stderr(0.111, 4000))
        paired = math.sqrt((0.113 + 0.111 - (0.113 - 0.111) ** 2) / 4000)
        assert paired > naive
        # Concretely: the choice of formula decides real orderings. Scan for a gap the
        # naive bar clears and the correct one does not, and require that one exists.
        hi, straddle = 0.113, []
        for step in range(1, 1000):
            lo = hi - step * 1e-4
            if lo <= 0:
                break
            naive_bar = 2 * math.hypot(report.champ_stderr(hi, 4000), report.champ_stderr(lo, 4000))
            if (hi - lo) > naive_bar and not report._rank_separated(hi, lo, 4000):
                straddle.append(lo)
        assert straddle, "the two formulas must disagree somewhere or this is not a real fix"

    def test_odds_carry_espn_facts_when_the_fetch_works(self, workspace, cfg):
        payload = report.odds_payload(workspace, cfg)
        assert payload["teams"][0]["wins"] == 1

    def test_odds_survive_a_standings_fetch_that_fails(self, workspace, cfg, sim, monkeypatch):
        """The simulated table is the answer; ESPN's record is decoration on it."""

        def boom():
            raise RuntimeError("401")

        monkeypatch.setattr(sim.league, "teams", boom)
        payload = report.odds_payload(workspace, cfg)
        assert payload["teams"][0]["wins"] is None

    def test_leverage_is_reported_per_remaining_matchup(self, workspace, cfg):
        payload = report.leverage_payload(workspace, cfg)
        # One row per remaining game for this team, not per week of the season: `0 <=
        # leverage <= 1` and `min(...) <= first` are true by construction and would pass
        # against an empty implementation, so assert the shape instead.
        state = workspace.sim(cfg).state
        mine = [g for g in state.remaining_games if cfg.team_id in (g.home_team_id, g.away_team_id)]
        assert len(payload["weeks"]) == len(mine)
        assert [r["week"] for r in payload["weeks"]] == sorted(g.weeks[0] for g in mine)
        assert payload["this_week"]["week"] == payload["weeks"][0]["week"]
        assert payload["mean_leverage"] == pytest.approx(
            sum(r["leverage"] for r in payload["weeks"]) / len(payload["weeks"]), abs=1e-6
        )

    def test_a_leverage_column_that_is_a_constant_says_it_is_one(self, capsys):
        """ "This week barely matters" is the most valuable line here and in week 1 it
        never fires.

        Live, across all 42 remaining matchups in all three leagues, leverage ran
        0.706-1.000 with a mean of 0.97 and not one row under 0.25 -- at 0-0 the
        projected margin between any two teams is at most ~19 points against an sd_diff
        of 23-30, so every game is a coin flip by construction. A column that is a
        constant printed beside a per-week ranking invites reading an order into it.
        """
        weeks = [
            {
                "week": w,
                "opponent": "T2",
                "margin": 0.5,
                "sd_diff": 28.0,
                "win_probability": 0.51,
                "leverage": lev,
                "decided": False,
            }
            for w, lev in ((1, 1.0), (2, 0.98), (3, 0.92), (4, 0.71))
        ]
        payload = {
            "this_week": {**weeks[0], "points_per_win_pct": 0.0142},
            "least_leveraged": weeks[-1],
            "weeks": weeks,
            "mean_leverage": 0.9025,
        }
        report.render_leverage(payload, _wide())
        out = capsys.readouterr().out
        assert "No remaining matchup is decided" in out
        assert "0.71-1.00" in out
        # ...and it must go away the moment one of them actually is decided.
        capsys.readouterr()
        payload["weeks"] = [*weeks, {**weeks[0], "week": 5, "leverage": 0.05, "decided": True}]
        report.render_leverage(payload, _wide())
        assert "No remaining matchup is decided" not in capsys.readouterr().out

    def test_lineup_reads_the_set_starters_and_names_the_slots(self, workspace, cfg):
        payload = report.lineup_payload(workspace, cfg)
        assert payload["current_known"] is True
        assert {row["slot"] for row in payload["recommended"]} <= {
            "QB",
            "RB",
            "WR",
            "TE",
            "DST",
            "K",
            "FLEX",
        }
        assert len(payload["recommended"]) == sum(SLOT_COUNTS.values())

    def test_an_unreadable_roster_is_said_out_loud(self, workspace, cfg, sim, monkeypatch):
        """Substituting the optimum silently would show "no change" to the one manager
        this surface exists for: the one who has not logged in since the draft."""

        def boom():
            raise RuntimeError("no cookie")

        monkeypatch.setattr(sim.league, "rosters", boom)
        payload = report.lineup_payload(workspace, cfg)
        assert payload["current_known"] is False

    def test_a_section_that_raises_is_recorded_not_raised(self):
        out = report._section("boom", lambda: (_ for _ in ()).throw(ValueError("nope")))
        assert out == {"ok": False, "error": "ValueError: nope"}

    def test_weekly_composes_and_survives_one_broken_section(self, workspace, cfg, monkeypatch):
        monkeypatch.setattr(
            report, "leverage_payload", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        payload = report.weekly_payload(workspace, cfg, sections=("odds", "leverage", "lineup"))
        assert payload["ok"] is True
        assert payload["odds"]["ok"] is True
        assert payload["leverage"] == {"ok": False, "error": "RuntimeError: x"}
        assert payload["lineup"]["ok"] is True
        assert payload["headline"]

    def test_weekly_reports_a_league_it_cannot_build(self, cfg, monkeypatch):
        monkeypatch.setattr(
            report.pipeline, "build", lambda *a, **k: (_ for _ in ()).throw(OSError("ESPN 401"))
        )
        monkeypatch.setattr(
            report.pipeline, "client_from_env", lambda *a, **k: SimpleNamespace(close=lambda: None)
        )
        with report.Workspace() as ws:
            payload = report.weekly_payload(ws, cfg)
        assert payload["ok"] is False
        assert "ESPN 401" in payload["error"]


# --------------------------------------------------------------------------------------
# Headline and the honest empty state
# --------------------------------------------------------------------------------------


def _empty_weekly() -> dict:
    """A league where every surface honestly found nothing."""
    return {
        "ok": True,
        "league_id": 77,
        "season": 2026,
        "name": "Synthetic",
        "team_id": 1,
        "week": 5,
        "waivers": {
            "ok": True,
            "threshold": 0.0011,
            "title_per_point": 0.00056,
            "board": [
                {
                    "add": "Nobody",
                    "position": "RB",
                    "drop": "-",
                    "delta_title": 0.0002,
                    "stderr": 0.0001,
                    "delta_points": 0.4,
                    "significant": True,
                    "verdict": "act",
                    "clears_threshold": False,
                    "leverage": 1.0,
                    "confidence": "medium",
                    "tags": [],
                    "rationale": "",
                }
            ],
            "claims": [],
            "priority": 3,
            "uses_faab": False,
        },
        "lineup": {"ok": True, "n_changes": 0, "changes": [], "leverage": 0.11, "week": 5},
        "trades": {"ok": True, "trades": [], "n_found": 0},
        # A live D/ST plan: several points of title probability for a plan whose week-one
        # action is a hold. Carried, not deleted, and not executable.
        "stream": {
            "ok": True,
            "stream": {
                "action": "hold",
                "position": "DST",
                "significant": True,
                "verdict": "act",
                "delta_title": 0.028,
                "stderr": 0.0035,
                "tags": ["streaming", "no-action-this-week", "partially-unpriced"],
                "receive": [],
                "send": [],
                "plan_summary": {
                    "weeks": 14,
                    "weeks_shown": 8,
                    "total_gain_points": 43.5,
                    "priced_gain_points": 14.9,
                    "unpriced_gain_points": 28.6,
                    "unpriced_share": 0.657,
                    "week_one_gain_points": 0.0,
                    "n_acquisitions": 12,
                },
            },
        },
        "leverage": {
            "ok": True,
            "this_week": {
                "leverage": 0.11,
                "opponent": "T2",
                "points_per_win_pct": 0.16,
            },
        },
    }


class TestEmptyState:
    def test_a_hold_is_stated_in_its_own_numbers(self):
        line = report.headline(_empty_weekly())
        assert line.startswith("Hold")
        assert "0.110pp" in line  # the priority threshold it failed to clear
        assert "already the one to start" in line
        assert "decided" in line  # leverage 0.11
        assert "0.056pp" in line  # what a point is worth here

    def test_a_live_week_is_described_as_live(self):
        payload = _empty_weekly()
        payload["leverage"]["this_week"]["leverage"] = 1.0
        assert "live" in report.headline(payload)

    def test_a_claim_that_does_not_clear_the_threshold_does_not_lead(self):
        """It is a real measured gain AND the wrong thing to do. Both, at once.

        The two reasons to hold read differently on purpose: "nothing clears its own
        error" and "something did and you still cannot execute it" are different facts
        and a user acts on them differently. The second one has to name WHICH gain: the
        largest measured number in a live league is usually the rest-of-season streaming
        plan, and a sentence written for the waiver case reads as a claim about the wire.
        """
        line = report.headline(_empty_weekly())
        assert line.startswith("Hold -- the largest measured gain here is the stream at +2.80pp")
        assert "no-action-this-week" in line
        assert "Nobody" in line  # and the wire is still described, separately

    def test_a_board_of_pure_noise_reads_differently(self):
        payload = _empty_weekly()
        payload["waivers"]["board"][0].update(significant=False, verdict="noise")
        payload["stream"]["stream"].update(significant=False, verdict="noise")
        assert report.headline(payload).startswith("Hold -- nothing here clears its own error")

    def test_an_empty_wire_is_stated_rather_than_left_blank(self):
        payload = _empty_weekly()
        payload["waivers"]["board"] = []
        line = report.headline(payload)
        assert "Nothing on the wire clears the 0.110pp priority threshold" in line

    def test_a_real_action_leads_instead(self):
        payload = _empty_weekly()
        payload["waivers"]["claims"] = payload["waivers"]["board"]
        line = report.headline(payload)
        assert line.startswith("Claim Nobody")
        assert "Worth +0.02pp" in line

    def test_the_queue_says_a_league_held_rather_than_dropping_it(self):
        inner = report._envelope("queue", [_empty_weekly()])
        payload = report.queue_payload(inner)
        # Two rows, both real measurements and neither executable today: a claim that
        # does not clear the cost of spending priority, and a rest-of-season streaming
        # plan whose week-one action is a hold. Listed and flagged, never deleted.
        assert [a["actionable"] for a in payload["actions"]] == [False, False]
        by_surface = {a["surface"]: a for a in payload["actions"]}
        assert by_surface["waiver"]["blockers"] == [
            "below the continuation value of holding waiver priority"
        ]
        assert by_surface["stream"]["blockers"] == ["no-action-this-week"]
        assert payload["holds"] and "Synthetic" in payload["holds"][0]
        assert payload["first_actionable"] is None

    def test_the_local_queue_carries_the_streaming_plan_it_used_to_delete(self):
        """`fq queue` and `fq queue --local` must not describe different worlds.

        `edges.portfolio` puts the three live rest-of-season D/ST plans on TOP of the
        board at +2.0 to +3.8pp -- the largest numbers in the whole tool. The local merge
        dropped them on the floor whenever week one's action was a hold, so the same
        command with one flag either led with them or denied they existed.
        """
        payload = report.queue_payload(report._envelope("queue", [_empty_weekly()]))
        stream = next(a for a in payload["actions"] if a["surface"] == "stream")
        assert stream["delta_title"] == pytest.approx(0.028)
        assert stream["actionable"] is False
        # And it says what the +2.8pp actually buys: 12 adds, none of them this week.
        assert any("12 acquisition(s) over 14 weeks" in c for c in stream["caveats"])
        assert any("+0.0 pts" in c for c in stream["caveats"])
        # It is the biggest number on the board and it is still not the thing to do.
        assert payload["actions"][0]["surface"] != "stream" or payload["first_actionable"] is None

    def test_the_empty_queue_renders(self, capsys):
        report.render_queue(report.queue_payload(report._envelope("queue", [])), _wide())
        assert "Nothing to do in any league" in capsys.readouterr().out

    def test_a_queue_of_nothing_executable_says_so(self, capsys):
        """Not the same thing as an empty queue, and it must not read like one."""
        inner = report._envelope("queue", [_empty_weekly()])
        report.render_queue(report.queue_payload(inner), _wide())
        out = capsys.readouterr().out
        assert "Nothing above is both established and executable today" in out
        assert "below the continuation value" in out


# --------------------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------------------


def _action(delta: float, stderr: float, **kw) -> report.Action:
    base = {
        "league_id": 1,
        "league_name": "L",
        "season": 2026,
        "team_id": 1,
        "surface": "waiver",
        "headline": "do a thing",
        "delta_title": delta,
        "stderr": stderr,
        "leverage": 1.0,
        "significant": abs(delta) > 2 * stderr,
        "verdict": "act" if abs(delta) > 2 * stderr else "noise",
        "confidence": "high",
        "cost": "waiver priority",
        "deadline": "Tuesday night",
    }
    base.update(kw)
    return report.Action(**base)


class TestQueue:
    def test_a_measured_small_effect_outranks_a_noisy_large_one(self):
        """The whole reason the queue exists is to not send the user after noise.

        Live: Wine Wednesday's best claim is +0.17pp +/- 0.01 and its best trade is
        +1.03pp +/- 0.41. The trade has the bigger point estimate and is the one the
        simulation cannot tell from zero.
        """
        claim = _action(0.0017, 0.0001)
        trade = _action(0.0103, 0.0041, surface="trade", significant=False, verdict="noise")
        assert [a.surface for a in report.rank_actions([trade, claim])] == ["waiver", "trade"]

    def test_a_hold_sorts_below_something_you_can_do(self):
        plan = _action(0.028, 0.0035, surface="stream", actionable=False, blockers=("hold",))
        claim = _action(0.0017, 0.0001)
        assert [a.surface for a in report.rank_actions([plan, claim])] == ["waiver", "stream"]

    def test_cost_is_only_a_tiebreak(self):
        free = _action(0.01, 0.001, cost="free", surface="lineup")
        costly = _action(0.01, 0.001, cost="negotiation", surface="trade")
        assert [a.cost for a in report.rank_actions([costly, free])] == ["free", "negotiation"]

    def test_first_actionable_skips_holds_and_noise(self):
        rows = [
            _action(0.028, 0.0035, surface="stream", actionable=False).to_dict(),
            _action(0.0103, 0.0041, significant=False, verdict="noise").to_dict(),
            _action(0.0017, 0.0001).to_dict(),
        ]
        assert report._first_actionable(rows) == 2

    def test_the_top_trade_carries_its_own_selection_bias(self):
        """The number the queue and the headline quote is the max of n noisy draws.

        `decide/trades.selection_threshold` corrects the LABEL and nothing corrects the
        NUMBER, because nothing can without a second independent draw. Measured: the same
        Blacksburg trade (Hubbard and Waddle for Mason and Corum) came back at +1.75,
        +1.75, +1.10, +1.83 and +1.28pp under seeds 1-5 with a per-seed error of
        +/-0.46pp, and its verdict flipped between `act` and `noise` three times to two.
        Quoting "+1.75pp (+/-0.45pp)" with no further comment says the trade is worth
        that, and it does not.
        """
        payload = dict(_empty_weekly())
        payload["trades"] = {
            "ok": True,
            "n_found": 35,
            "trades": [
                {
                    "delta_title": 0.0175,
                    "stderr": 0.0045,
                    "significant": True,
                    "verdict": "act",
                    "confidence": "medium",
                    "leverage": 1.0,
                    "tags": ["trade", "3-team", "confirmed", "counterparty-loses"],
                    "rationale": "Zu gets Mason for Shough. You get Waddle.",
                    "receive": [{"name": "Jaylen Waddle"}],
                    "send": [{"name": "Jordan Mason"}],
                    "partners": [{"name": "Zu"}],
                }
            ],
        }
        trade = next(a for a in report.actions_from(_CFG, payload) if a.surface == "trade")
        assert any("maximum of 35 noisy paired draws" in c for c in trade.caveats)
        assert any("title odds FALL" in c for c in trade.caveats)
        # And the headline that leads on it repeats them rather than ending at the number.
        line = report.headline(payload)
        assert line.startswith("Offer Zu")
        assert "+1.75pp" in line and "Caveat:" in line
        assert "title odds FALL" in line and "maximum of 35" in line

    def test_a_single_candidate_is_not_accused_of_selection(self):
        """The caveat has to be able to be absent, or it is decoration rather than a fact."""
        payload = dict(_empty_weekly())
        payload["trades"] = {
            "ok": True,
            "n_found": 1,
            "trades": [
                {
                    "delta_title": 0.0175,
                    "stderr": 0.0045,
                    "significant": True,
                    "verdict": "act",
                    "confidence": "high",
                    "leverage": 1.0,
                    "tags": ["trade", "2-team", "confirmed"],
                    "rationale": "",
                    "receive": [],
                    "send": [],
                    "partners": [],
                }
            ],
        }
        trade = next(a for a in report.actions_from(_CFG, payload) if a.surface == "trade")
        assert trade.caveats == ()

    def test_an_action_round_trips_to_json(self):
        body = _action(0.01, 0.001, blockers=("hold",), actionable=False).to_dict()
        assert json.loads(json.dumps(body))["blockers"] == ["hold"]
        assert body["z"] == pytest.approx(10.0)


def _install_fake_portfolio(monkeypatch, fake) -> None:
    """Stand a fake `edges.portfolio` in front of the real one, both ways it is reached.

    `sys.modules` alone is not enough once anything has imported the real module: the
    parent package then holds it as an attribute, and `from .edges import portfolio`
    reads the attribute rather than the cache. Miss that and the test quietly exercises
    the real portfolio against a stub client.
    """
    import fantasy_quant.edges as edges_pkg

    monkeypatch.setitem(__import__("sys").modules, "fantasy_quant.edges.portfolio", fake)
    monkeypatch.setattr(edges_pkg, "portfolio", fake, raising=False)


class TestPortfolioHook:
    def test_a_missing_edges_package_degrades_to_the_local_merge(self, monkeypatch, cfg):
        """The module is a sibling being written alongside this one; absence is normal."""
        import builtins

        real = builtins.__import__

        def refuse(name, *args, **kw):
            if "portfolio" in name:
                raise ImportError("no edges")
            return real(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", refuse)
        with report.Workspace() as ws:
            assert report.portfolio_queue(ws, [cfg]) is None

    def test_a_hook_that_raises_degrades_rather_than_crashing(self, monkeypatch, cfg):
        fake = SimpleNamespace(
            action_queue=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
            build_portfolio=lambda *a, **k: SimpleNamespace(stakes=()),
        )
        _install_fake_portfolio(monkeypatch, fake)
        monkeypatch.setattr(
            report.pipeline, "client_from_env", lambda *a, **k: SimpleNamespace(close=lambda: None)
        )
        with report.Workspace() as ws:
            assert report.portfolio_queue(ws, [cfg]) is None

    def test_leagues_in_two_seasons_are_not_one_portfolio(self, cfg):
        """`build_portfolio` puts every league on ONE shared NFL season under one seed.

        Two seasons is not one season, and pretending otherwise would break the very
        coupling that makes the queue's key a measurement.
        """
        other = R.LeagueConfig(league_id=78, season=2025, name="Old", team_id=1)
        with report.Workspace() as ws:
            assert report.portfolio_queue(ws, [cfg, other]) is None

    def test_queue_items_are_converted_and_credited(self, monkeypatch, cfg):
        rec = Recommendation(
            move=Move(kind=MoveKind.HOLD, league_id=77),
            delta_title=0.028,
            delta_points=42.0,
            stderr=0.0035,
            rationale="Week 1: hold Jaguars D/ST -- the move is later. More text here.",
            tags=("streaming", "no-action-this-week"),
        )
        item = SimpleNamespace(
            league_id=77,
            league_name="Synthetic",
            team_id=1,
            surface="streaming",
            rec=rec,
            significant=True,
            actionable=False,
            blockers=("no-action-this-week",),
        )
        fake = SimpleNamespace(
            action_queue=lambda *a, **k: SimpleNamespace(items=(item,), failures=()),
            build_portfolio=lambda *a, **k: SimpleNamespace(stakes=()),
        )
        _install_fake_portfolio(monkeypatch, fake)
        monkeypatch.setattr(
            report.pipeline, "client_from_env", lambda *a, **k: SimpleNamespace(close=lambda: None)
        )
        with report.Workspace() as ws:
            payload = report.portfolio_queue(ws, [cfg])
        assert payload["ranked_by"] == "edges.portfolio.action_queue"
        row = payload["actions"][0]
        assert row["actionable"] is False and row["blockers"] == ["no-action-this-week"]
        assert row["headline"].startswith("Week 1: hold Jaguars D/ST")
        # A hold is not executable, so the queue has nothing to do today and says so.
        assert payload["first_actionable"] is None
        assert payload["holds"]

    def test_the_portfolio_hooks_naive_two_sigma_does_not_win(self, monkeypatch, cfg):
        """`fq queue` is the DEFAULT path and it was undoing this module's whole point.

        `edges.portfolio.QueueItem.significant` is `delta != 0 and rec.significant` --
        the plain two-sigma test. Wine Wednesday's top trade is +1.03pp +/- 0.41
        (z = 2.5), which `decide/trades.py` marks `confidence="low"` against a
        selection-adjusted threshold of 3.2 sigma once the winner was picked out of the
        candidate set. Copying `item.significant` made `fq weekly` render it `~` and
        `fq queue` render it as a thing to go and do -- the same recommendation, two
        opposite labels, from one tool in one run.
        """
        rec = Recommendation(
            move=Move(kind=MoveKind.TRADE, league_id=77),
            delta_title=0.01025,
            delta_points=8.2,
            stderr=0.00414,
            confidence="low",
            tags=("trade", "2-team", "confirmed", "counterparty-loses"),
            rationale="Rob gets A for B. You get C.",
        )
        item = SimpleNamespace(
            league_id=77,
            league_name="Synthetic",
            team_id=1,
            surface="trades",
            rec=rec,
            significant=rec.significant,  # True: the naive test the hook applies
            actionable=True,
            blockers=("counterparty-loses",),
            n_considered=40,
        )
        assert item.significant is True
        action = report._action_from_item(item, season=2026)
        assert action.significant is False
        assert action.verdict == "noise"
        # Both paths attach the same selection sentence, so `--local` does not change it.
        assert any("maximum of 40 noisy paired draws" in c for c in action.caveats)
        # The disagreement is recorded rather than resolved silently in either direction.
        assert any("shown as not significant" in c for c in action.caveats)
        # And the surface's own warning about the counterparty survives the conversion.
        assert any("title odds FALL" in c for c in action.caveats)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


class TestRendering:
    def test_every_section_renders(self, workspace, cfg, capsys, monkeypatch):
        monkeypatch.setattr(report, "stream_payload", _fake_stream)
        payload = report.weekly_payload(workspace, cfg)
        report.render_weekly(payload)
        out = capsys.readouterr().out
        assert "championship odds" in out
        assert "leverage" in out
        assert "lineup" in out
        assert "waivers" in out or "wire" in out

    def test_an_insignificant_row_is_marked(self, capsys):
        report.render_trades(
            {
                "trades": [
                    {
                        "partners": [{"name": "Rob"}],
                        "receive": [{"name": "A"}],
                        "send": [{"name": "B"}],
                        "delta_points": 8.2,
                        "delta_title": 0.0103,
                        "stderr": 0.0041,
                        "z": 2.5,
                        "significant": False,
                        "verdict": "noise",
                    }
                ],
                "n_found": 1,
                "significance_test": "selection-adjusted",
            }
        )
        assert "~" in capsys.readouterr().out

    def test_a_significant_row_is_not_marked(self, capsys):
        report.render_trades(
            {
                "trades": [
                    {
                        "partners": [{"name": "Rob"}],
                        "receive": [{"name": "A"}],
                        "send": [{"name": "B"}],
                        "delta_points": 10.3,
                        "delta_title": 0.0175,
                        "stderr": 0.0045,
                        "z": 3.9,
                        "significant": True,
                        "verdict": "act",
                    }
                ],
                "n_found": 1,
                "significance_test": "selection-adjusted",
            }
        )
        assert "~" not in capsys.readouterr().out

    def test_the_streaming_plan_says_where_its_value_actually_is(self, capsys):
        """The largest number this module prints, and it used to print alone.

        The three live D/ST plans are +2.0/+2.8/+3.8pp, they top `fq queue`, and all
        three facts that change what they mean were computable from the plan table and
        rendered nowhere: week one's gain over holding is +0.0, the plan assumes a dozen
        successful adds, and two thirds of it sits in weeks with no closing line.
        """
        rows = [
            {
                "week": 1,
                "start": "JAX",
                "hold": "JAX",
                "gain": 0.0,
                "opponent_priced": True,
                "add": [],
                "drop": [],
            },
            {
                "week": 2,
                "start": "SF",
                "hold": "JAX",
                "gain": 3.3,
                "opponent_priced": True,
                "add": ["SF"],
                "drop": ["JAX"],
            },
            {
                "week": 7,
                "start": "TEN",
                "hold": "(empty)",
                "gain": 7.1,
                "opponent_priced": False,
                "add": ["TEN"],
                "drop": ["SF"],
            },
        ]
        summary = report._plan_summary(rows, shown=2)
        assert summary["week_one_gain_points"] == 0.0
        assert summary["n_acquisitions"] == 2
        assert summary["unpriced_gain_points"] == pytest.approx(7.1)
        assert summary["unpriced_share"] == pytest.approx(7.1 / 10.4)
        report.render_stream(
            {
                "stream": {
                    "position": "DST",
                    "action": "hold",
                    "delta_title": 0.0277,
                    "stderr": 0.0035,
                    "verdict": "act",
                    "tags": ["streaming", "no-action-this-week", "partially-unpriced"],
                    "caveats": report.caveats_for(["partially-unpriced"]),
                    "plan": rows[:2],
                    "plan_summary": summary,
                }
            },
            _wide(),
        )
        out = capsys.readouterr().out
        assert "2 acquisition(s) over 3 weeks" in out
        assert "+0.0 is week 1 -- the part you can act on today" in out
        assert "68% of it" in out and "no closing line" in out
        assert "showing the first 2 of 3 planned weeks" in out

    def test_a_not_streamable_position_is_not_rendered_as_an_edge(self, capsys):
        """A kicker plan at R^2 = 0.022 must not read like a D/ST one at 0.134."""
        rec = Recommendation(
            move=Move(kind=MoveKind.HOLD, league_id=77),
            delta_title=0.0085,
            delta_points=9.0,
            stderr=0.0035,
            confidence="low",
            tags=("streaming", "k", "not-streamable", "no-action-this-week"),
        )
        verdict = report.verdict_for(rec, surface="stream")
        report.render_stream(
            {
                "stream": {
                    "position": "K",
                    "action": "hold",
                    "delta_title": rec.delta_title,
                    "stderr": rec.stderr,
                    "verdict": verdict.kind,
                    "tags": list(rec.tags),
                    "caveats": report.caveats_for(rec.tags),
                    "plan": [],
                    "plan_summary": {},
                }
            },
            _wide(),
        )
        out = capsys.readouterr().out
        assert "~" in out  # marked as not acted on despite clearing two sigma
        assert "streamable floor" in out and "tie-break" in out

    def test_a_decided_week_says_so(self, capsys):
        report.render_leverage(
            {
                "this_week": {
                    "opponent": "T2",
                    "margin": -40.0,
                    "sd_diff": 20.0,
                    "win_probability": 0.02,
                    "leverage": 0.05,
                    "decided": True,
                    "points_per_win_pct": 0.1,
                    "week": 5,
                },
                "least_leveraged": None,
                "weeks": [
                    {
                        "week": 5,
                        "opponent": "T2",
                        "margin": -40.0,
                        "sd_diff": 20.0,
                        "win_probability": 0.02,
                        "leverage": 0.05,
                        "decided": True,
                    }
                ],
                "mean_leverage": 0.05,
            }
        )
        assert "effectively decided" in capsys.readouterr().out

    def test_a_claim_on_the_threshold_is_not_printed_as_a_fact(self, capsys):
        """`clears?` was a hard `>=` and the claim waterfall is cut on it.

        Live: Blacksburg's fifth claim clears the threshold by +0.003pp against its own
        +/-0.024pp and its sixth misses by -0.004pp, so "Submit 5 claim(s)" is really
        "submit 4, and the fifth is a coin flip". Wine Wednesday's third claim clears by
        +0.008pp against +/-0.038pp. Re-seeding rotated which player held the marginal
        slot while keeping the count.
        """
        board = [
            {
                "add": "Jets D/ST",
                "position": "DST",
                "drop": "X",
                "delta_points": 2.4,
                "delta_title": 0.001515,
                "stderr": 0.000136,
                "verdict": "act",
                "clears_threshold": True,
                "clears_certain": True,
                "clears_margin": 0.000645,
            },
            {
                "add": "Giants D/ST",
                "position": "DST",
                "drop": "X",
                "delta_points": 1.6,
                "delta_title": 0.000898,
                "stderr": 0.000121,
                "verdict": "act",
                "clears_threshold": True,
                "clears_certain": False,
                "clears_margin": 0.000028,
            },
            {
                "add": "49ers D/ST",
                "position": "DST",
                "drop": "X",
                "delta_points": 1.4,
                "delta_title": 0.000829,
                "stderr": 0.000104,
                "verdict": "act",
                "clears_threshold": False,
                "clears_certain": False,
                "clears_margin": -0.000041,
            },
        ]
        report.render_waivers(
            {
                "uses_faab": False,
                "priority": 3,
                "priority_known": True,
                "budget": 0,
                "threshold": 0.00087,
                "baseline_title": 0.0415,
                "title_per_point": 0.00079,
                "title_per_point_stderr": 0.0000389,
                "board": board,
                "claims": board[:2],
                "any_claim": True,
                "waterfall_note": "note",
            },
            _wide(),
        )
        out = capsys.readouterr().out
        assert "yes?" in out and "no?" in out
        assert "2 row(s) marked '?'" in out
        assert "coin flip" in out
        # The rate's own error is shown, because every dTitle above is that rate times a
        # points gain and re-seeding moved it 14% on the live board.
        assert "+/-0.0039" in out

    def test_a_board_clear_of_the_threshold_gets_no_question_marks(self, capsys):
        """...and the warning has to be able to be absent."""
        report.render_waivers(
            {
                "uses_faab": False,
                "priority": 3,
                "priority_known": True,
                "budget": 0,
                "threshold": 0.00087,
                "baseline_title": 0.0415,
                "title_per_point": 0.00079,
                "title_per_point_stderr": 0.0000389,
                "board": [
                    {
                        "add": "Jets D/ST",
                        "position": "DST",
                        "drop": "X",
                        "delta_points": 2.4,
                        "delta_title": 0.001515,
                        "stderr": 0.000136,
                        "verdict": "act",
                        "clears_threshold": True,
                        "clears_certain": True,
                        "clears_margin": 0.000645,
                    }
                ],
                "claims": [],
                "any_claim": False,
                "waterfall_note": "",
            },
            _wide(),
        )
        out = capsys.readouterr().out
        assert "?" not in out.split("clears?")[1]
        assert "coin flip" not in out

    def test_a_broken_lineup_is_shouted_about(self, capsys):
        report.render_lineup(
            {
                "current_known": True,
                "unpriced_current": [1, 2],
                "recommended": [],
                "changes": [],
                "week": 5,
                "margin": 0.0,
                "sd_diff": 30.0,
                "z": 0.0,
                "leverage": 1.0,
                "delta_win_prob": 0.0,
                "points_sacrifice": 0.0,
                "guard": "",
                "recommendation": {
                    "delta_title": 0.0,
                    "stderr": 0.0,
                    "significant": False,
                    "verdict": "null",
                },
            }
        )
        assert "lineup is broken" in capsys.readouterr().out

    def test_a_failed_league_renders_as_one_red_line(self, capsys):
        report.render_weekly({"ok": False, "league_id": 77, "name": "X", "error": "OSError: 401"})
        out = capsys.readouterr().out
        assert "401" in out and "championship odds" not in out


def _fake_waivers(ws, cfg, **kw) -> dict:
    """The board `decide/waivers` would return, minus the five seconds of simulation."""
    return {
        **_empty_weekly()["waivers"],
        "name": "Synthetic",
        "league_id": cfg.league_id,
        "team_id": 1,
        "team_name": "T1",
        "week": 5,
        "priority_known": True,
        "budget": 0,
        "baseline_title": 0.0415,
        "baseline": "waiver board (unfilled slot streams a replacement)",
        "title_per_point_stderr": 0.00003,
        "week_leverage": 1.0,
        "sd_diff": 31.1,
        "n_free_agents": 12,
        "blocks": [],
        "hold": {"delta_title": 0.0, "rationale": "hold"},
        "best": {"delta_title": 0.0, "rationale": "hold"},
        "any_claim": False,
        "waterfall_note": "",
    }


def _fake_stream(ws, cfg, **kw) -> dict:
    """`decide/streaming` fetches a betting market; this file is offline by construction."""
    return {
        "league_id": cfg.league_id,
        "name": "Synthetic",
        "team_id": 1,
        "stream": {
            "position": "DST",
            "action": "hold",
            "delta_title": 0.028,
            "stderr": 0.0035,
            "significant": True,
            "verdict": "act",
            "tags": ["streaming", "no-action-this-week"],
            "plan": [],
            "plan_source": "unavailable",
            "receive": [],
            "send": [],
            "leverage": 1.0,
            "confidence": "medium",
            "rationale": "hold this week.",
        },
    }


# --------------------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, sim, registry_file):
    """Every command wired to the synthetic league, with the heavy surfaces stubbed."""
    monkeypatch.setattr(report.pipeline, "build", lambda *a, **k: sim)
    monkeypatch.setattr(
        report.pipeline, "client_from_env", lambda *a, **k: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(report, "stream_payload", _fake_stream)
    monkeypatch.setattr(report, "waivers_payload", _fake_waivers)
    monkeypatch.setattr(
        report,
        "trades_payload",
        lambda ws, cfg, **kw: {
            "league_id": cfg.league_id,
            "name": "Synthetic",
            "team_id": 1,
            "n_found": 0,
            "min_gain": 0.0,
            "significance_test": "selection-adjusted",
            "trades": [],
        },
    )
    return registry_file


COMMANDS = ["odds", "weekly", "waivers", "trades", "lineup", "stream", "queue"]


class TestCli:
    @pytest.mark.parametrize("command", COMMANDS)
    def test_every_command_renders(self, cli, command):
        args = [command, "--config", str(cli), "--league", "111", "--sims", "50"]
        if command == "queue":
            args.append("--local")
        result = runner.invoke(report.app, args)
        assert result.exit_code == 0, result.output
        assert result.output.strip()

    @pytest.mark.parametrize("command", COMMANDS)
    def test_every_command_emits_parseable_json(self, cli, command):
        args = [command, "--config", str(cli), "--league", "111", "--json"]
        if command == "queue":
            args.append("--local")
        result = runner.invoke(report.app, args)
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["schema_version"] == report.SCHEMA_VERSION
        assert payload["command"] == command
        assert isinstance(payload["leagues"], list) and payload["leagues"]
        assert all("league_id" in row for row in payload["leagues"])

    def test_json_carries_no_rendering(self, cli):
        """A dashboard piping this into `jq` must not get a progress line first."""
        result = runner.invoke(
            report.app, ["odds", "--config", str(cli), "--league", "111", "--json"]
        )
        assert result.output.lstrip().startswith("{")

    def test_selection_by_name_and_by_id_agree(self, cli):
        by_id = runner.invoke(
            report.app, ["odds", "--config", str(cli), "--league", "111", "--json"]
        )
        by_name = runner.invoke(
            report.app, ["odds", "--config", str(cli), "--league", "Wine Wednesday", "--json"]
        )
        assert by_id.exit_code == by_name.exit_code == 0
        a, b = json.loads(by_id.output), json.loads(by_name.output)
        assert a["leagues"][0]["league_id"] == b["leagues"][0]["league_id"] == 111

    def test_the_default_is_every_enabled_league(self, cli):
        result = runner.invoke(report.app, ["odds", "--config", str(cli), "--json"])
        payload = json.loads(result.output)
        assert [row["league_id"] for row in payload["leagues"]] == [111, 222]

    def test_a_league_that_cannot_be_fetched_does_not_take_the_rest_down(
        self, cli, sim, monkeypatch
    ):
        def build(league_id, season, **kw):
            if league_id == 111:
                raise OSError("ESPN returned 401 for league 111")
            return sim

        monkeypatch.setattr(report.pipeline, "build", build)
        result = runner.invoke(report.app, ["odds", "--config", str(cli), "--json"])
        assert result.exit_code == 0
        rows = {row["league_id"]: row for row in json.loads(result.output)["leagues"]}
        assert rows[111]["ok"] is False
        assert "401" in rows[111]["error"]
        assert rows[222]["ok"] is True
        assert rows[222]["teams"]

    def test_a_failed_league_is_visible_in_the_terminal_too(self, cli, monkeypatch):
        monkeypatch.setattr(
            report.pipeline, "build", lambda *a, **k: (_ for _ in ()).throw(OSError("401"))
        )
        result = runner.invoke(report.app, ["odds", "--config", str(cli)])
        assert result.exit_code == 0
        assert "401" in result.output

    def test_an_unknown_league_is_a_usage_error(self, cli):
        result = runner.invoke(report.app, ["odds", "--config", str(cli), "--league", "nope"])
        assert result.exit_code == 2

    def test_an_unknown_skip_section_is_a_usage_error(self, cli):
        result = runner.invoke(
            report.app, ["weekly", "--config", str(cli), "--league", "111", "--skip", "waves"]
        )
        assert result.exit_code == 2

    def test_skip_drops_the_section_from_the_payload(self, cli):
        result = runner.invoke(
            report.app,
            [
                "weekly",
                "--config",
                str(cli),
                "--league",
                "111",
                "--skip",
                "trades",
                "--skip",
                "stream",
                "--json",
            ],
        )
        payload = json.loads(result.output)["leagues"][0]
        assert "odds" in payload and "trades" not in payload and "stream" not in payload

    def test_an_unknown_position_is_a_usage_error(self, cli):
        result = runner.invoke(
            report.app, ["stream", "--config", str(cli), "--league", "111", "--position", "LB"]
        )
        assert result.exit_code == 2


class TestMount:
    def test_mounting_adds_every_command_once(self):
        import typer

        parent = typer.Typer()
        report.mount(parent)
        names = {c.callback.__name__ for c in parent.registered_commands}
        assert {"odds_command", "weekly_command", "queue_command"} <= names
        before = len(parent.registered_commands)
        report.mount(parent)
        assert len(parent.registered_commands) == before

    def test_the_schema_version_is_pinned(self):
        """Bumping it is a decision, not a side effect. A consumer reads this."""
        assert report.SCHEMA_VERSION == 1


def test_no_numpy_scalars_leak_into_json(workspace, cfg):
    """`np.float32` is not JSON-serialisable and every payload here comes off a tensor."""
    payload = report.odds_payload(workspace, cfg)
    text = json.dumps(payload)  # raises TypeError on a numpy scalar
    assert "championship" in text
    assert not any(isinstance(v, np.generic) for t in payload["teams"] for v in t.values())


class TestRenderingAgainstAnAnalystBoard:
    """Both surfaces name the board they were priced against, and say it is unverified."""

    RANKINGS = {
        "kind": "silva",
        "scoring": "half_ppr",
        "matches_league_scoring": False,
        "n": 150,
        "weight": 1.0,
        "file": "silva_top150_half_ppr.csv",
        "unverified": "This board ships without a measured verdict.",
    }

    def test_trades_show_the_spread_and_the_note_and_name_the_board(self, capsys):
        report.render_trades(
            {
                "trades": [
                    {
                        "partners": [{"name": "Rob"}],
                        "receive": [{"name": "Keaton Mitchell"}],
                        "send": [{"name": "Aaron Jones"}],
                        "delta_points": 10.3,
                        "spread": 6.1,
                        "mispriced": True,
                        "notes": {"Keaton Mitchell": "Committee back ceiling."},
                        "delta_title": 0.0175,
                        "stderr": 0.0045,
                        "z": 3.9,
                        "significant": True,
                        "verdict": "act",
                    }
                ],
                "n_found": 1,
                "rankings": self.RANKINGS,
                "priced_on": "analyst board",
                "significance_test": "selection-adjusted",
            },
            _wide(),
        )
        out = capsys.readouterr().out
        assert "spread" in out and "+6.1" in out
        assert "Committee back ceiling." in out
        assert "silva board" in out and "unverified" in out
        # The fallback across scoring formats is said out loud, never silent.
        assert "half_ppr board" in out

    def test_without_a_board_the_trade_table_is_exactly_as_before(self, capsys):
        report.render_trades(
            {
                "trades": [
                    {
                        "partners": [{"name": "Rob"}],
                        "receive": [{"name": "A"}],
                        "send": [{"name": "B"}],
                        "delta_points": 10.3,
                        "delta_title": 0.0175,
                        "stderr": 0.0045,
                        "z": 3.9,
                        "significant": True,
                        "verdict": "act",
                    }
                ],
                "n_found": 1,
                "significance_test": "selection-adjusted",
            },
            _wide(),
        )
        out = capsys.readouterr().out
        assert "spread" not in out and "unverified" not in out

    def test_waivers_show_the_rank_pair_and_the_note(self, capsys):
        report.render_waivers(
            {
                "uses_faab": False,
                "priority": 3,
                "priority_known": True,
                "budget": 0,
                "threshold": 0.00087,
                "baseline_title": 0.0415,
                "title_per_point": 0.00079,
                "title_per_point_stderr": 0.0000389,
                "rankings": {**self.RANKINGS, "matches_league_scoring": True},
                "board": [
                    {
                        "add": "Ryan Flournoy",
                        "position": "WR",
                        "drop": "Denzel Boston",
                        "delta_points": 1.4,
                        "delta_title": 0.0012,
                        "stderr": 0.0001,
                        "verdict": "act",
                        "clears_threshold": True,
                        "clears_certain": True,
                        "clears_margin": 0.0003,
                        "board": "55-over-60",
                        "note": "Underrated playmaker.",
                    },
                    {
                        "add": "Jets D/ST",
                        "position": "DST",
                        "drop": "X",
                        "delta_points": 3.0,
                        "delta_title": 0.0030,
                        "stderr": 0.0001,
                        "verdict": "act",
                        "clears_threshold": True,
                        "clears_certain": True,
                        "clears_margin": 0.002,
                        "board": "-",
                        "note": "-",
                    },
                ],
                "claims": [],
                "any_claim": False,
                "waterfall_note": "",
            },
            _wide(),
        )
        out = capsys.readouterr().out
        assert "55-over-60" in out
        assert "Underrated playmaker." in out
        assert "silva board" in out and "unverified" in out
        # A board with no opinion about a row prints nothing for it, not a dash-as-fact.
        assert "-over-" in out and out.count("-over-") == 1
