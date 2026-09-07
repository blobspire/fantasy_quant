"""Season simulator tests.

Three layers, deliberately separated because they fail for different reasons.

*Bracket arithmetic* is tested through `simulate_from_scores` with hand-drawn normal
team scores. No tensor, no lineups, no projections -- if a bracket test fails it is the
bracket. This is also where the calibration anchor lives, and it is exact rather than
approximate: a team that wins 53% of individual matchups and holds a first-round bye
must win the title 0.53**2 = 28.1% of the time, against a coin-flip team's 25%. The
whole "playoffs are near-random" claim is that one line of arithmetic, and a simulator
that disagrees with it is broken in a way that would inflate every seed-chasing
recommendation the product ever makes.

*Lineup and state plumbing* is tested on tiny hand-built tensors where the right answer
can be read off by eye.

*Calibration against the corpus* draws a real 12-team PPR league out of the Parquet
snapshot already in the repo, simulates it, and checks the team-score moments against
the measured anchor. Offline; skipped if the corpus is absent. Only the tests marked
`network` touch ESPN.

One note on player ids: ESPN gives every D/ST a negative one (`-16{proTeamId:03d}`), and
`sim/distributions.py` currently cannot seed a random stream for a negative id. The
corpus fixture therefore remaps them; the tests of this module's own bookkeeping
deliberately do not, because season.py must handle them as they are, and
`test_negative_player_ids_survive_the_whole_pipeline` is the regression that says so.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm, skew

from fantasy_quant.core import PlayerOutlook
from fantasy_quant.sim.season import (
    _TIEBREAK_WARNED,
    ANCHOR_TEAM_MEAN,
    ANCHOR_TEAM_SD,
    ANCHOR_TEAM_SKEW,
    OPPONENT_LINEUP_EFFICIENCY,
    OPPONENT_LINEUP_EFFICIENCY_SD,
    Franchise,
    LeagueState,
    LineupEfficiency,
    PlayerPool,
    ScheduledGame,
    SeasonError,
    bracket_seed_order,
    bye_seeds,
    ex_ante_rank,
    leave_one_out,
    lineup_plans,
    measure_hindsight_ratio,
    panel_for,
    playoff_round_weeks,
    rescale_for_ex_ante_lineups,
    simulate,
    simulate_from_scores,
    slot_eligibility_from_rosters,
    state_from_league,
    team_week_scores,
)

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "snapshots" / "espn"

#: The measured team-score anchor. Every synthetic score in this file is drawn from it,
#: so a bracket result here is directly comparable to a real league's.
TEAM_SD = ANCHOR_TEAM_SD
TEAM_MEAN = ANCHOR_TEAM_MEAN

#: Points of weekly edge that make a team win exactly 53% of head-to-heads.
EDGE_53 = float(norm.ppf(0.53) * math.sqrt(2.0) * TEAM_SD)

#: ESPN's standard redraft start: 1QB/2RB/2WR/1TE/1FLEX/1DST/1K. Slot ids, not
#: position ids -- the two spaces collide at 4 and 15.
SLOT_COUNTS = {0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 23: 1}
SLOT_ELIGIBILITY = {
    0: frozenset({1}),
    2: frozenset({2}),
    4: frozenset({3}),
    6: frozenset({4}),
    16: frozenset({16}),
    17: frozenset({5}),
    23: frozenset({2, 3, 4}),
}


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _bracket_state(
    n_teams: int = 12,
    playoff_team_count: int = 6,
    rounds: tuple[tuple[int, ...], ...] = ((15,), (16,), (17,)),
    *,
    wins: list[int] | None = None,
    points_for: list[float] | None = None,
    reseed: bool = False,
) -> LeagueState:
    """A league whose regular season is over, so seeds are fixed and only the bracket runs.

    Records descend with the team index unless overridden, which makes team 0 the
    1-seed and lets a test reason about a specific bracket slot instead of a
    distribution over them.
    """
    w = wins if wins is not None else list(range(n_teams - 1, -1, -1))
    pf = points_for if points_for is not None else [1500.0 - 10.0 * i for i in range(n_teams)]
    pool = PlayerPool.of([(1, 1, 1, "nobody")])
    franchises = tuple(
        Franchise(
            team_id=i + 1,
            name=f"T{i + 1}",
            player_ids=(),
            wins=w[i],
            losses=(n_teams - 1) - w[i],
            points_for=pf[i],
        )
        for i in range(n_teams)
    )
    weeks = tuple(sorted({wk for r in rounds for wk in r}))
    return LeagueState(
        league_id=99,
        season=2026,
        name="bracket",
        franchises=franchises,
        pool=pool,
        weeks=weeks,
        remaining_games=(),
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=playoff_team_count,
        playoff_rounds=rounds,
        playoff_reseed=reseed,
    )


def _normal_scores(
    state: LeagueState, n_sims: int, *, edge: float = 0.0, edge_team: int = 0, seed: int = 11
) -> np.ndarray:
    """`(sims, weeks, teams)` independent normal team scores at the measured anchor.

    Independent across weeks on purpose: it makes the bracket's answer exactly
    computable by hand, which is what the anchors below check against.
    """
    rng = np.random.default_rng(seed)
    scores = rng.normal(TEAM_MEAN, TEAM_SD, (n_sims, len(state.weeks), state.size))
    scores[:, :, edge_team] += edge
    return scores.astype(np.float32)


#: Nine forced starters (QB, 2RB, 2WR, TE, FLEX-able WR, DST, K) followed by a bench.
#: The first nine leave exactly one legal lineup, which is what makes a hand-checkable
#: score possible; anything past nine introduces real choice, which is what makes the
#: hindsight-versus-ex-ante gap measurable at all.
_POSITIONS = [1, 2, 2, 3, 3, 4, 3, 16, 5, 3, 2, 4, 3, 2, 1, 3]


def _pool(n_teams: int, per_team: int = 9) -> tuple[PlayerPool, list[tuple[int, ...]]]:
    """A pool of exactly-startable rosters, optionally with a bench past the ninth slot.

    Player id encodes the team so a failure is readable: 1000*team + slot.
    """
    positions = _POSITIONS[:per_team]
    rows: list[tuple[int, int, int, str]] = []
    rosters: list[tuple[int, ...]] = []
    for t in range(n_teams):
        ids = []
        for j, pos in enumerate(positions):
            pid = 1000 * (t + 1) + j
            ids.append(pid)
            rows.append((pid, pos, (t % 32) + 1, f"p{pid}"))
        rosters.append(tuple(ids))
    return PlayerPool.of(rows), rosters


def _round_robin(n: int, weeks: range) -> list[tuple[int, int, int]]:
    """Circle-method schedule: `(week, home_index, away_index)`."""
    ids = list(range(n))
    out = []
    for w in weeks:
        for i in range(n // 2):
            out.append((w, ids[i], ids[n - 1 - i]))
        ids = [ids[0], ids[-1], *ids[1:-1]]
    return out


def _tensor_state(
    n_teams: int = 4,
    weeks: tuple[int, ...] = (1, 2),
    playoff_rounds: tuple[tuple[int, ...], ...] = (),
    playoff_team_count: int = 2,
    per_team: int = 9,
) -> LeagueState:
    pool, rosters = _pool(n_teams, per_team)
    reg = tuple(w for w in weeks if not any(w in r for r in playoff_rounds))
    games = tuple(
        ScheduledGame(matchup_period=w, weeks=(w,), home_team_id=a + 1, away_team_id=b + 1)
        for w, a, b in _round_robin(n_teams, range(min(reg), max(reg) + 1))
    )
    return LeagueState(
        league_id=7,
        season=2026,
        name="tiny",
        franchises=tuple(
            Franchise(team_id=i + 1, name=f"T{i + 1}", player_ids=rosters[i], is_user=(i == 0))
            for i in range(n_teams)
        ),
        pool=pool,
        weeks=weeks,
        remaining_games=games,
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=playoff_team_count,
        playoff_rounds=playoff_rounds,
        my_team_id=1,
    )


def _flat_tensor(state: LeagueState, value: float = 10.0) -> np.ndarray:
    return np.full((1, len(state.weeks), state.pool.size), value, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Bracket geometry
# --------------------------------------------------------------------------------------


class TestBracketGeometry:
    def test_seed_order_is_the_standard_mirror(self):
        """1-8, 4-5, 2-7, 3-6: the top two seeds can only meet in the final."""
        assert bracket_seed_order(8) == (0, 7, 3, 4, 1, 6, 2, 5)
        assert bracket_seed_order(4) == (0, 3, 1, 2)
        assert bracket_seed_order(2) == (0, 1)
        assert bracket_seed_order(1) == (0,)

    def test_padding_a_six_team_field_reproduces_espns_bracket(self):
        """Seeds 1 and 2 bye; the real first-round games are 4v5 and 3v6."""
        order = bracket_seed_order(8)
        pairs = [(order[i], order[i + 1]) for i in range(0, 8, 2)]
        real = [(a + 1, b + 1) for a, b in pairs if a < 6 and b < 6]
        byes = [a + 1 for a, b in pairs if b >= 6] + [b + 1 for a, b in pairs if a >= 6]
        assert sorted(real) == [(3, 6), (4, 5)]
        assert sorted(byes) == [1, 2]

    def test_seed_order_rejects_a_non_power_of_two(self):
        with pytest.raises(SeasonError, match="power of two"):
            bracket_seed_order(6)

    def test_bye_seeds_are_the_top_of_the_padding_gap(self):
        assert bye_seeds(6) == (0, 1)
        assert bye_seeds(5) == (0, 1, 2)
        assert bye_seeds(4) == ()
        assert bye_seeds(8) == ()

    def test_state_derives_bracket_size_and_byes(self):
        state = _bracket_state(playoff_team_count=6)
        assert state.bracket_size == 8
        assert state.bye_count == 2
        four = _bracket_state(playoff_team_count=4, rounds=((16,), (17,)))
        assert four.bracket_size == 4
        assert four.bye_count == 0
        # A field and a round count that disagree would crown a champion with half the
        # bracket still alive, so the state refuses to exist.
        with pytest.raises(SeasonError, match="needs 2 rounds"):
            _bracket_state(playoff_team_count=4)


# --------------------------------------------------------------------------------------
# The calibration anchor
# --------------------------------------------------------------------------------------


class TestPlayoffRandomness:
    """The anchor the whole product's seed-chasing advice rests on.

    A bye seed has to win exactly two games, so its title probability is its per-game
    win rate squared. 0.53 becomes 0.281 and 0.50 becomes 0.250: a 6% edge per game is
    a 12% edge in the title and no more. If these numbers come out high, the bracket is
    wrong and every "climb to the 2-seed" recommendation downstream is overpriced.
    """

    SIMS = 60_000

    def test_a_53_percent_team_with_a_bye_wins_the_title_28_percent(self):
        state = _bracket_state()
        scores = _normal_scores(state, self.SIMS, edge=EDGE_53, edge_team=0)
        result = simulate_from_scores(state, scores)
        assert result.by_team(1).bye == 1.0
        assert result.by_team(1).championship == pytest.approx(0.53**2, abs=0.008)

    def test_a_coin_flip_team_with_a_bye_wins_the_title_25_percent(self):
        state = _bracket_state()
        scores = _normal_scores(state, self.SIMS)
        result = simulate_from_scores(state, scores)
        assert result.by_team(1).championship == pytest.approx(0.25, abs=0.008)

    def test_the_53_percent_edge_is_worth_only_three_points_of_title_odds(self):
        """Stated as a difference so a regression cannot hide behind loose tolerances."""
        state = _bracket_state()
        even = simulate_from_scores(state, _normal_scores(state, self.SIMS))
        better = simulate_from_scores(
            state, _normal_scores(state, self.SIMS, edge=EDGE_53, edge_team=0)
        )
        lift = better.by_team(1).championship - even.by_team(1).championship
        assert lift == pytest.approx(0.53**2 - 0.25, abs=0.008)

    def test_the_same_team_without_a_bye_has_to_win_three_games(self):
        """Seeded third instead of first: 0.53**3, and the bye is worth more than the edge."""
        wins = [9, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        state = _bracket_state(wins=wins)
        scores = _normal_scores(state, self.SIMS, edge=EDGE_53, edge_team=0)
        result = simulate_from_scores(state, scores)
        assert result.by_team(1).bye == 0.0
        assert result.by_team(1).championship == pytest.approx(0.53**3, abs=0.008)


class TestBracketSimulation:
    def test_championship_probabilities_sum_to_one(self):
        state = _bracket_state()
        result = simulate_from_scores(state, _normal_scores(state, 4000))
        assert sum(result.title_odds().values()) == pytest.approx(1.0)
        assert result.champions.sum(axis=1).tolist() == [1] * 4000

    def test_only_playoff_teams_can_win_the_title(self):
        state = _bracket_state()
        result = simulate_from_scores(state, _normal_scores(state, 4000))
        assert not (result.champions & ~result.made_playoffs).any()
        assert not (result.reached_final & ~result.made_playoffs).any()
        assert not (result.champions & ~result.reached_final).any()

    def test_byes_go_to_the_top_two_seeds_and_nobody_else(self):
        state = _bracket_state()
        result = simulate_from_scores(state, _normal_scores(state, 2000))
        assert np.array_equal(result.byes, result.seeds < 2)
        assert result.by_team(1).bye == 1.0
        assert result.by_team(2).bye == 1.0
        assert result.by_team(3).bye == 0.0

    def test_a_four_team_bracket_gives_nobody_a_bye(self):
        state = _bracket_state(playoff_team_count=4, rounds=((16,), (17,)))
        result = simulate_from_scores(state, _normal_scores(state, 2000))
        assert not result.byes.any()
        assert sum(result.title_odds().values()) == pytest.approx(1.0)

    def test_a_dominant_team_wins_nearly_every_title(self):
        state = _bracket_state()
        scores = _normal_scores(state, 4000, edge=300.0, edge_team=0)
        result = simulate_from_scores(state, scores)
        assert result.by_team(1).championship > 0.99

    def test_a_dominant_team_that_misses_the_playoffs_wins_nothing(self):
        """Regular-season record, not weekly scoring, is what gets you into the bracket."""
        wins = [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        state = _bracket_state(wins=wins)
        scores = _normal_scores(state, 2000, edge=300.0, edge_team=0)
        result = simulate_from_scores(state, scores)
        assert result.by_team(1).make_playoffs == 0.0
        assert result.by_team(1).championship == 0.0

    def test_a_two_week_final_favours_the_better_team(self):
        """Doubling a round halves the underdog's variance edge. Measurably, not vaguely."""
        one = _bracket_state(rounds=((15,), (16,), (17,)))
        two = _bracket_state(rounds=((15,), (16,), (17, 18)))
        a = simulate_from_scores(one, _normal_scores(one, 60_000, edge=EDGE_53, edge_team=0))
        b = simulate_from_scores(two, _normal_scores(two, 60_000, edge=EDGE_53, edge_team=0))
        # Round three win rate goes from Phi(d/(sqrt2 sd)) to Phi(2d/(2 sd)).
        expected = 0.53 * float(norm.cdf(2 * EDGE_53 / (2 * TEAM_SD)))
        assert b.by_team(1).championship == pytest.approx(expected, abs=0.008)
        assert b.by_team(1).championship > a.by_team(1).championship

    def test_multi_week_rounds_sum_both_weeks(self):
        """A team that scores 200 in week 17 and 0 in week 18 still beats a steady 90."""
        state = _bracket_state(playoff_team_count=2, rounds=((17, 18),))
        scores = np.zeros((1, 2, 12), dtype=np.float32)
        scores[0, 0, 0], scores[0, 1, 0] = 200.0, 0.0
        scores[0, :, 1] = 90.0
        assert simulate_from_scores(state, scores).champions[0, 0]
        scores[0, 0, 0] = 100.0  # 100 + 0 now loses to 90 + 90
        assert simulate_from_scores(state, scores).champions[0, 1]

    def test_a_playoff_tie_goes_to_the_better_seed(self):
        state = _bracket_state(playoff_team_count=2, rounds=((17,),))
        scores = np.zeros((1, 1, 12), dtype=np.float32)
        scores[0, 0, 0] = scores[0, 0, 1] = 111.0
        result = simulate_from_scores(state, scores)
        assert result.champions[0, 0]  # team 1 is the 1-seed

    def test_reseeding_helps_the_top_seed(self):
        """With reseeding the 1-seed always draws the weakest survivor."""
        fixed = _bracket_state(reseed=False)
        seeded = _bracket_state(reseed=True)
        # Strength decreasing with seed, so reseeding is a real advantage.
        rng = np.random.default_rng(3)
        edges = np.linspace(12.0, -12.0, 12).astype(np.float32)
        base = rng.normal(TEAM_MEAN, TEAM_SD, (40_000, 3, 12)).astype(np.float32) + edges
        a = simulate_from_scores(fixed, base).by_team(1).championship
        b = simulate_from_scores(seeded, base).by_team(1).championship
        assert b > a
        assert sum(simulate_from_scores(seeded, base).title_odds().values()) == pytest.approx(1.0)

    def test_no_bracket_means_no_champion(self):
        state = _bracket_state(playoff_team_count=6, rounds=())
        state = LeagueState(
            **{
                **{f.name: getattr(state, f.name) for f in state.__dataclass_fields__.values()},
                "playoff_rounds": (),
                "weeks": (15,),
            }
        )
        result = simulate_from_scores(state, _normal_scores(state, 100))
        assert not result.champions.any()
        assert not result.made_playoffs.any()
        assert not result.byes.any()


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------


class TestSeeding:
    def test_record_outranks_points_for(self):
        """A 1-win team with 3000 points still seeds below a 2-win team with 100."""
        state = _bracket_state(
            n_teams=4,
            playoff_team_count=2,
            wins=[1, 2, 0, 0],
            points_for=[3000.0, 100.0, 50.0, 40.0],
            rounds=((17,),),
        )
        result = simulate_from_scores(state, _normal_scores(state, 10))
        assert result.by_team(2).mean_seed == 1.0
        assert result.by_team(1).mean_seed == 2.0

    def test_ties_are_broken_by_points_for(self):
        state = _bracket_state(
            n_teams=4,
            playoff_team_count=2,
            wins=[2, 2, 1, 1],
            points_for=[900.0, 1000.0, 10.0, 20.0],
            rounds=((17,),),
        )
        result = simulate_from_scores(state, _normal_scores(state, 10))
        assert result.by_team(2).mean_seed == 1.0  # equal record, more points
        assert result.by_team(1).mean_seed == 2.0

    def test_a_tie_counts_half_a_win(self):
        """1-0-2 and 2-0-0 are the same record, so points-for separates them."""
        state = _bracket_state(
            n_teams=4,
            playoff_team_count=2,
            wins=[2, 2, 1, 1],
            points_for=[100.0, 200.0, 300.0, 400.0],
            rounds=((17,),),
        )
        state = state.with_franchise(
            Franchise(
                team_id=3, name="T3", player_ids=(), wins=1, losses=0, ties=2, points_for=150.0
            )
        )
        result = simulate_from_scores(state, _normal_scores(state, 10))
        # Record score 2*1+2 == 2*2+0, so team 3 sits between team 2 (200 pf) and team 1 (100).
        assert result.by_team(2).mean_seed == 1.0
        assert result.by_team(3).mean_seed == 2.0
        assert result.by_team(1).mean_seed == 3.0
        # And half a win really is half: drop the ties and team 3 falls to last, behind
        # team 4 which has the same one win and more points.
        clean = state.with_franchise(
            Franchise(team_id=3, name="T3", player_ids=(), wins=1, losses=2, points_for=150.0)
        )
        assert simulate_from_scores(clean, _normal_scores(clean, 10)).by_team(3).mean_seed == 4.0

    def test_seed_distribution_is_a_distribution(self):
        state = _bracket_state()
        result = simulate_from_scores(state, _normal_scores(state, 500))
        for outcome in result.outcomes():
            assert sum(outcome.seed_distribution) == pytest.approx(1.0)
            assert len(outcome.seed_distribution) == 12
        # Exactly one team per seed per simulation.
        totals = np.stack([np.array(o.seed_distribution) for o in result.outcomes()]).sum(axis=0)
        assert totals == pytest.approx(np.ones(12))

    def test_an_unsupported_tiebreak_falls_back_to_points_for_with_a_warning(self, caplog):
        state = _bracket_state(
            n_teams=4,
            playoff_team_count=2,
            wins=[2, 2, 1, 1],
            points_for=[900.0, 1000.0, 10.0, 20.0],
            rounds=((17,),),
        )
        state = LeagueState(
            **{
                **{f.name: getattr(state, f.name) for f in state.__dataclass_fields__.values()},
                "league_id": 123456,
                "playoff_seeding_rule": "H2H_RECORD",
            }
        )
        # The "warn once per league" set is module global and outlives the test, so a
        # second run in the same session would see no warning and fail for the wrong
        # reason. Clear this league's entry rather than depend on test ordering.
        _TIEBREAK_WARNED.discard(123456)
        with caplog.at_level("WARNING"):
            result = simulate_from_scores(state, _normal_scores(state, 10))
        assert "H2H_RECORD" in caplog.text
        assert result.by_team(2).mean_seed == 1.0
        # ...and it really is once per league, not once per call.
        caplog.clear()
        with caplog.at_level("WARNING"):
            simulate_from_scores(state, _normal_scores(state, 10))
        assert "H2H_RECORD" not in caplog.text


# --------------------------------------------------------------------------------------
# Played weeks are facts
# --------------------------------------------------------------------------------------


class TestPlayedWeeksAreFacts:
    def test_starting_record_carries_through_untouched(self):
        state = _tensor_state(n_teams=4, weeks=(1, 2))
        boosted = state.with_franchise(
            Franchise(
                team_id=1,
                name="T1",
                player_ids=state.franchise(1).player_ids,
                wins=5,
                losses=1,
                points_for=800.0,
                is_user=True,
            )
        )
        tensor = _flat_tensor(state)
        base = simulate(state, tensor, efficiency=LineupEfficiency.perfect())
        after = simulate(boosted, tensor, efficiency=LineupEfficiency.perfect())
        assert after.by_team(1).expected_wins - base.by_team(1).expected_wins == pytest.approx(5.0)
        assert after.by_team(1).expected_points_for - base.by_team(1).expected_points_for == (
            pytest.approx(800.0)
        )
        # Nobody else moved: the fact is additive, not a re-simulation.
        for t in (2, 3, 4):
            assert after.by_team(t).expected_wins == pytest.approx(base.by_team(t).expected_wins)

    def test_only_the_remaining_schedule_is_played(self):
        """Two weeks on the axis and one game each means at most two more wins."""
        state = _tensor_state(n_teams=4, weeks=(1, 2))
        result = simulate(state, _flat_tensor(state), efficiency=LineupEfficiency.perfect())
        assert len(state.remaining_games) == 4  # 4 teams, 2 weeks, 2 games a week
        played = result.wins + result.losses + result.ties
        assert played.max() == 2.0

    def test_historical_all_play_is_carried_into_the_result(self):
        state = _tensor_state(n_teams=4, weeks=(1, 2))
        prior = state.with_franchise(
            Franchise(
                team_id=1,
                name="T1",
                player_ids=state.franchise(1).player_ids,
                all_play_wins=9.0,
                all_play_games=9,
                is_user=True,
            )
        )
        result = simulate(prior, _flat_tensor(state), efficiency=LineupEfficiency.perfect())
        # Nine prior all-play wins from nine games, then six drawn games at 0.5 each.
        assert result.all_play_wins[0, 0] == pytest.approx(9.0 + 6 * 0.5)
        assert result.all_play_games[0, 0] == pytest.approx(9.0 + 6.0)

    def test_a_game_on_a_week_off_the_axis_is_rejected(self):
        pool, rosters = _pool(2)
        with pytest.raises(SeasonError, match="not on the tensor axis"):
            LeagueState(
                league_id=1,
                season=2026,
                name="x",
                franchises=tuple(
                    Franchise(team_id=i + 1, name=f"T{i}", player_ids=rosters[i]) for i in range(2)
                ),
                pool=pool,
                weeks=(1,),
                remaining_games=(
                    ScheduledGame(matchup_period=9, weeks=(9,), home_team_id=1, away_team_id=2),
                ),
                lineup_slot_counts=SLOT_COUNTS,
                slot_eligibility=SLOT_ELIGIBILITY,
                playoff_team_count=2,
                playoff_rounds=(),
            )


# --------------------------------------------------------------------------------------
# All-play
# --------------------------------------------------------------------------------------


class TestAllPlay:
    def test_all_play_is_zero_sum_across_the_league(self):
        state = _tensor_state(n_teams=4, weeks=(1, 2, 3, 4))
        rng = np.random.default_rng(5)
        tensor = rng.gamma(2.0, 5.0, (200, 4, state.pool.size)).astype(np.float32)
        result = simulate(state, tensor, rank=None, efficiency=LineupEfficiency.perfect())
        assert result.all_play_pct().mean() == pytest.approx(0.5, abs=1e-6)

    def test_all_play_has_lower_variance_than_record(self):
        """The reason both columns are reported: record is mostly schedule."""
        state = _tensor_state(n_teams=12, weeks=tuple(range(1, 15)))
        rng = np.random.default_rng(6)
        tensor = rng.gamma(3.0, 4.0, (400, 14, state.pool.size)).astype(np.float32)
        result = simulate(state, tensor, rank=None, efficiency=LineupEfficiency.perfect())
        record_pct = result.wins / np.maximum(result.wins + result.losses + result.ties, 1.0)
        # Identical rosters, so every point of spread in either column is noise. Record
        # sees fourteen coin flips; all-play sees eleven opponents in each of them, and
        # lands about a third tighter even after the within-week correlation.
        assert result.all_play_pct().std() < 0.7 * record_pct.std()

    def test_a_tie_is_half_an_all_play_win(self):
        state = _tensor_state(n_teams=4, weeks=(1, 2))
        result = simulate(state, _flat_tensor(state), efficiency=LineupEfficiency.perfect())
        assert result.all_play_pct()[0, 0] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Lineups
# --------------------------------------------------------------------------------------


class TestLineups:
    def test_the_best_eligible_players_start_regardless_of_espn_slots(self):
        """Roster is QB, RB, RB, WR, WR, TE, WR, DST, K -- exactly one legal lineup."""
        state = _tensor_state(n_teams=2, weeks=(1,))
        tensor = np.zeros((1, 1, state.pool.size), dtype=np.float32)
        cols = state.pool.columns(state.franchise(1).player_ids)
        tensor[0, 0, cols] = np.arange(1.0, 10.0)
        scores = team_week_scores(state, tensor, rank=None, efficiency=LineupEfficiency.perfect())
        assert scores[0, 0, 0] == pytest.approx(45.0)  # all nine start

    def test_the_flex_takes_the_best_leftover(self):
        """Ten players for nine slots: the weakest flex-eligible one sits."""
        pool, _ = _pool(1)
        extra = 1009
        rows = [
            (pid, pos, 1, f"p{pid}")
            for pid, pos in zip(pool.player_ids, pool.position_ids, strict=True)
        ] + [(extra, 3, 1, "spare")]
        pool = PlayerPool.of(rows)
        franchise = Franchise(team_id=1, name="T1", player_ids=(*range(1000, 1009), extra))
        state = LeagueState(
            league_id=1,
            season=2026,
            name="x",
            franchises=(franchise, Franchise(team_id=2, name="T2", player_ids=())),
            pool=pool,
            weeks=(1,),
            remaining_games=(),
            lineup_slot_counts=SLOT_COUNTS,
            slot_eligibility=SLOT_ELIGIBILITY,
            playoff_team_count=2,
            playoff_rounds=(),
        )
        tensor = np.zeros((1, 1, pool.size), dtype=np.float32)
        tensor[0, 0, pool.columns(franchise.player_ids)] = [
            5.0,  # QB
            5.0,
            5.0,  # RB, RB
            5.0,
            5.0,  # WR, WR
            5.0,  # TE
            1.0,  # third WR: the weak flex candidate
            5.0,  # DST
            5.0,  # K
        ] + [99.0]  # the spare WR, who should take the flex
        scores = team_week_scores(state, tensor, rank=None, efficiency=LineupEfficiency.perfect())
        assert scores[0, 0, 0] == pytest.approx(8 * 5.0 + 99.0)

    def test_a_player_projected_at_zero_is_left_on_the_bench(self):
        """An empty slot scores zero, so nothing is gained by starting a zero."""
        state = _tensor_state(n_teams=2, weeks=(1,))
        cols = state.pool.columns(state.franchise(1).player_ids)
        rank = np.zeros((1, state.pool.size), dtype=np.float32)
        rank[0, cols] = [10, 10, 10, 10, 10, 10, 0, 10, 10]
        tensor = np.zeros((1, 1, state.pool.size), dtype=np.float32)
        tensor[0, 0, cols] = 3.0
        scores = team_week_scores(state, tensor, rank=rank, efficiency=LineupEfficiency.perfect())
        assert scores[0, 0, 0] == pytest.approx(8 * 3.0)  # the projected-zero WR sits

    def test_hindsight_lineups_score_more_than_ex_ante_ones(self):
        """The bug this module is built to avoid, measured rather than asserted away.

        The roster needs a bench for the two to differ at all: with exactly nine
        players for nine slots there is one legal lineup and hindsight buys nothing.
        That is itself the point -- `E[max] > max[E]` is a statement about *optionality*,
        and it is precisely bench depth that a hindsight simulator overvalues.
        """
        state = _tensor_state(n_teams=2, weeks=(1,), per_team=16)
        rng = np.random.default_rng(9)
        tensor = rng.gamma(2.0, 6.0, (3000, 1, state.pool.size)).astype(np.float32)
        rank = np.full((1, state.pool.size), 12.0, dtype=np.float32)
        ex_ante = team_week_scores(state, tensor, rank=rank, efficiency=LineupEfficiency.perfect())[
            :, :, 0
        ].mean()
        hindsight = team_week_scores(
            state, tensor, rank=None, efficiency=LineupEfficiency.perfect()
        )[:, :, 0].mean()
        assert hindsight > ex_ante
        # Seven bench players is worth more than a tenth of a team's weekly score under
        # hindsight and nothing at all ex ante. Same roster, same football.
        assert hindsight / ex_ante > 1.1
        # And with no bench the gap closes to exactly zero.
        forced = _tensor_state(n_teams=2, weeks=(1,))
        forced_tensor = rng.gamma(2.0, 6.0, (200, 1, forced.pool.size)).astype(np.float32)
        forced_rank = np.full((1, forced.pool.size), 12.0, dtype=np.float32)
        eff = LineupEfficiency.perfect()
        assert team_week_scores(forced, forced_tensor, rank=forced_rank, efficiency=eff)[
            :, :, 0
        ].mean() == pytest.approx(
            team_week_scores(forced, forced_tensor, rank=None, efficiency=eff)[:, :, 0].mean()
        )

    def test_the_result_records_that_lineups_were_hindsight(self):
        state = _tensor_state(n_teams=2, weeks=(1,))
        tensor = _flat_tensor(state)
        assert simulate(state, tensor).hindsight_lineups is True
        rank = np.ones((1, state.pool.size), dtype=np.float32)
        assert simulate(state, tensor, rank=rank).hindsight_lineups is False

    def test_a_mismatched_tensor_is_rejected(self):
        state = _tensor_state(n_teams=2, weeks=(1, 2))
        with pytest.raises(SeasonError, match="weeks"):
            team_week_scores(state, np.zeros((1, 5, state.pool.size), dtype=np.float32))
        with pytest.raises(SeasonError, match="players"):
            team_week_scores(state, np.zeros((1, 2, 3), dtype=np.float32))

    def test_lineup_plans_are_one_per_franchise(self):
        state = _tensor_state(n_teams=4, weeks=(1,))
        plans = lineup_plans(state)
        assert len(plans) == 4
        assert all(p.laminar for p in plans)
        assert all(p.n_slots == 9 for p in plans)


class TestLineupEfficiency:
    def test_the_haircut_applies_to_opponents_only(self):
        """Averaged over the draw, not read off one sample: a single opponent factor is
        one N(0.775, 0.05) draw, and a tolerance loose enough to admit it would also
        admit 0.70 or 0.85. 4,000 draws pin the mean to a thousandth."""
        state = _tensor_state(n_teams=2, weeks=(1,))
        tensor = np.repeat(_flat_tensor(state), 4000, axis=0)
        eff = LineupEfficiency.literal()
        scores = team_week_scores(state, tensor, rank=None, efficiency=eff)
        assert scores[:, 0, 0] == pytest.approx(90.0)  # team 1 is the user, never haircut
        assert scores[:, 0, 0].std() == 0.0
        assert scores[:, 0, 1].mean() / 90.0 == pytest.approx(OPPONENT_LINEUP_EFFICIENCY, abs=0.003)
        assert scores[:, 0, 1].std() / 90.0 == pytest.approx(
            OPPONENT_LINEUP_EFFICIENCY_SD, abs=0.003
        )
        assert (scores[:, 0, 1] < scores[:, 0, 0]).all()

    def test_symmetric_efficiency_removes_the_asymmetry(self):
        state = _tensor_state(n_teams=2, weeks=(1,))
        scores = team_week_scores(
            state, _flat_tensor(state), rank=None, efficiency=LineupEfficiency.symmetric()
        )
        assert scores[0, 0, 0] == pytest.approx(scores[0, 0, 1])

    def test_the_draw_is_reproducible_so_two_scenarios_meet_the_same_managers(self):
        state = _tensor_state(n_teams=6, weeks=(1,))
        a = LineupEfficiency.literal().draw(state, 500)
        b = LineupEfficiency.literal().draw(state, 500)
        assert np.array_equal(a, b)
        assert a.shape == (500, 6)
        assert a[:, 0].std() == 0.0  # the user's own factor is fixed at 1.0
        assert a[:, 1].std() > 0.0

    def test_the_literal_haircut_makes_the_user_an_implausible_favourite(self):
        """Why the published 0.775 is NOT the shipped default.

        Two identical rosters, and that haircut alone makes the user win about three
        matchups in four. Pinned so the reason the default is symmetric stays visible.
        """
        state = _tensor_state(n_teams=2, weeks=tuple(range(1, 15)))
        rng = np.random.default_rng(4)
        tensor = rng.gamma(3.0, 4.0, (2000, 14, state.pool.size)).astype(np.float32)
        eff = LineupEfficiency.literal()
        scores = team_week_scores(state, tensor, rank=None, efficiency=eff)
        user_win_rate = (scores[:, :, 0] > scores[:, :, 1]).mean()
        assert user_win_rate > 0.70
        fair = team_week_scores(state, tensor, rank=None, efficiency=LineupEfficiency.symmetric())
        assert (fair[:, :, 0] > fair[:, :, 1]).mean() == pytest.approx(0.5, abs=0.02)

    def test_the_literal_haircut_carries_all_the_way_into_the_title_odds(self):
        """What a caller who never passes `efficiency=` actually gets handed.

        The weekly win rate above is the mechanism; this is the number that reaches a
        user. Twelve *identical* rosters and a real 14-week schedule: symmetric play
        gives the honest 1/12, and the shipped default turns the same roster into a
        better-than-even title favourite with eleven and a half wins. Pinned at league
        level because a haircut that looks like a modest weekly edge compounds through
        fourteen games and a bracket, and no downstream surface can be read without
        knowing it.
        """
        state = _tensor_state(
            n_teams=12,
            weeks=tuple(range(1, 18)),
            playoff_rounds=((15,), (16,), (17,)),
            playoff_team_count=6,
            per_team=16,
        )
        rng = np.random.default_rng(9)
        tensor = rng.gamma(3.0, 4.0, (2000, 17, state.pool.size)).astype(np.float32)
        rank = np.tile(rng.uniform(2.0, 20.0, state.pool.size).astype(np.float32), (17, 1))
        literal_eff = LineupEfficiency.literal()
        loud = simulate(state, tensor, rank=rank, all_play=False, efficiency=literal_eff).by_team(1)
        default = simulate(state, tensor, rank=rank, all_play=False).by_team(1)
        fair = simulate(
            state, tensor, rank=rank, all_play=False, efficiency=LineupEfficiency.symmetric()
        ).by_team(1)
        assert fair.championship == pytest.approx(1.0 / 12.0, abs=0.03)
        assert fair.expected_wins == pytest.approx(7.0, abs=0.5)
        # The SHIPPED default is symmetric, so identical rosters get the honest 1/12.
        assert default.championship == pytest.approx(1.0 / 12.0, abs=0.03)
        # And the published constant, opted into explicitly, is worth 6x the baseline.
        assert loud.championship > 0.45
        assert loud.expected_wins > 11.0

    def test_calibrated_lands_between_the_literal_constant_and_none_at_all(self):
        """The recommended middle ground, and the one the live report is run with."""
        eff = LineupEfficiency.calibrated(0.89)
        assert OPPONENT_LINEUP_EFFICIENCY < eff.opponent_mean < 1.0
        assert eff.opponent_mean == pytest.approx(0.775 / 0.89)
        assert eff.user_mean == 1.0
        assert eff.is_asymmetric

    def test_rescaling_undoes_the_double_count(self):
        assert rescale_for_ex_ante_lineups(0.895) == pytest.approx(0.775 / 0.895)
        # An ex-ante lineup that already realises less than a real manager would means
        # the haircut has the sign backwards; clamp rather than amplify.
        assert rescale_for_ex_ante_lineups(0.5) == 1.0
        with pytest.raises(SeasonError):
            rescale_for_ex_ante_lineups(0.0)


# --------------------------------------------------------------------------------------
# Common random numbers
# --------------------------------------------------------------------------------------


class TestCommonRandomNumbers:
    def test_identical_input_gives_bit_identical_output(self):
        state = _tensor_state(n_teams=6, weeks=tuple(range(1, 13)))
        rng = np.random.default_rng(21)
        tensor = rng.gamma(3.0, 4.0, (400, 12, state.pool.size)).astype(np.float32)
        a = simulate(state, tensor, rank=None)
        b = simulate(state, tensor, rank=None)
        assert np.array_equal(a.champions, b.champions)
        assert np.array_equal(a.wins, b.wins)
        assert np.array_equal(a.seeds, b.seeds)
        assert np.array_equal(a.points_for, b.points_for)

    def test_a_change_on_one_roster_leaves_the_rest_of_the_football_alone(self):
        """The point of CRN: the untouched teams' weekly scores must be identical."""
        state = _tensor_state(n_teams=6, weeks=tuple(range(1, 13)))
        rng = np.random.default_rng(22)
        tensor = rng.gamma(3.0, 4.0, (200, 12, state.pool.size)).astype(np.float32)
        rank = np.full((12, state.pool.size), 10.0, dtype=np.float32)
        before = team_week_scores(state, tensor, rank=rank, efficiency=LineupEfficiency.perfect())
        changed = state.with_franchise(state.franchise(2).without(state.franchise(2).player_ids[0]))
        after = team_week_scores(changed, tensor, rank=rank, efficiency=LineupEfficiency.perfect())
        assert np.array_equal(before[:, :, [0, 2, 3, 4, 5]], after[:, :, [0, 2, 3, 4, 5]])
        assert not np.array_equal(before[:, :, 1], after[:, :, 1])


# --------------------------------------------------------------------------------------
# Leave-one-out
# --------------------------------------------------------------------------------------


class TestLeaveOneOut:
    def test_removing_a_player_nobody_starts_is_exactly_zero(self):
        """Under CRN a null change returns 0.0 rather than Monte Carlo fog."""
        pool, _ = _pool(2)
        rows = [
            (pid, pos, 1, f"p{pid}")
            for pid, pos in zip(pool.player_ids, pool.position_ids, strict=True)
        ] + [(1500, 5, 1, "spare kicker")]
        pool = PlayerPool.of(rows)
        state = LeagueState(
            league_id=1,
            season=2026,
            name="x",
            franchises=(
                Franchise(team_id=1, name="T1", player_ids=(*range(1000, 1009), 1500)),
                Franchise(team_id=2, name="T2", player_ids=tuple(range(2000, 2009))),
            ),
            pool=pool,
            weeks=(1, 2),
            remaining_games=(
                ScheduledGame(matchup_period=1, weeks=(1,), home_team_id=1, away_team_id=2),
                ScheduledGame(matchup_period=2, weeks=(2,), home_team_id=2, away_team_id=1),
            ),
            lineup_slot_counts=SLOT_COUNTS,
            slot_eligibility=SLOT_ELIGIBILITY,
            playoff_team_count=2,
            playoff_rounds=(),
        )
        rng = np.random.default_rng(31)
        tensor = rng.gamma(2.0, 5.0, (500, 2, pool.size)).astype(np.float32)
        rank = np.tile(np.arange(pool.size, 0, -1, dtype=np.float32), (2, 1))
        rank[:, pool.columns([1500])] = 0.1  # strictly the worst kicker
        [spare] = leave_one_out(
            state, tensor, player_ids=[1500], rank=rank, efficiency=LineupEfficiency.perfect()
        )
        assert spare.wins_added == 0.0
        assert spare.title_added == 0.0
        assert spare.title_added_stderr == 0.0

    def test_the_best_player_is_worth_the_most(self):
        # A bracket is needed for `title_added` to be anything but zero, so week 12 is
        # the final and weeks 1-11 are the regular season.
        state = _tensor_state(
            n_teams=6,
            weeks=tuple(range(1, 13)),
            playoff_rounds=((12,),),
            playoff_team_count=2,
        )
        rng = np.random.default_rng(32)
        tensor = rng.gamma(2.0, 4.0, (600, 12, state.pool.size)).astype(np.float32)
        mine = state.franchise(1).player_ids
        # Make one receiver enormous, ex ante and realised alike.
        star = mine[3]
        rank = np.full((12, state.pool.size), 8.0, dtype=np.float32)
        rank[:, state.pool.columns([star])] = 60.0
        tensor[:, :, state.pool.columns([star])] *= 8.0
        got = leave_one_out(
            state,
            tensor,
            player_ids=list(mine),
            rank=rank,
            efficiency=LineupEfficiency.perfect(),
        )
        assert got[0].player_id == star
        assert got[0].wins_added > 0.5
        assert got[0].title_added > 0.0

    def test_contributions_are_not_additive_and_the_docstring_says_so(self):
        """The one-term Shapley caveat, demonstrated rather than merely asserted.

        Two interchangeable receivers each look nearly worthless on their own, because
        removing either promotes the other. Summing them badly understates the pair.
        """
        pool, _ = _pool(2)
        rows = [
            (pid, pos, 1, f"p{pid}")
            for pid, pos in zip(pool.player_ids, pool.position_ids, strict=True)
        ] + [(1600, 3, 1, "twin")]
        pool = PlayerPool.of(rows)
        roster = (*range(1000, 1009), 1600)
        state = LeagueState(
            league_id=1,
            season=2026,
            name="x",
            franchises=(
                Franchise(team_id=1, name="T1", player_ids=roster),
                Franchise(team_id=2, name="T2", player_ids=tuple(range(2000, 2009))),
            ),
            pool=pool,
            weeks=tuple(range(1, 11)),
            remaining_games=tuple(
                ScheduledGame(matchup_period=w, weeks=(w,), home_team_id=1, away_team_id=2)
                for w in range(1, 11)
            ),
            lineup_slot_counts=SLOT_COUNTS,
            slot_eligibility=SLOT_ELIGIBILITY,
            playoff_team_count=2,
            playoff_rounds=(),
        )
        rng = np.random.default_rng(33)
        tensor = rng.gamma(2.0, 5.0, (800, 10, pool.size)).astype(np.float32)
        rank = np.full((10, pool.size), 10.0, dtype=np.float32)
        twins = [1006, 1600]  # the flex WR and his identical replacement
        singles = leave_one_out(
            state, tensor, player_ids=twins, rank=rank, efficiency=LineupEfficiency.perfect()
        )
        both = state.with_franchise(
            state.franchise(1).with_players(p for p in roster if p not in twins)
        )
        base = simulate(state, tensor, rank=rank, efficiency=LineupEfficiency.perfect()).wins.mean(
            axis=0
        )[0]
        without_pair = simulate(
            both, tensor, rank=rank, efficiency=LineupEfficiency.perfect()
        ).wins.mean(axis=0)[0]
        pair_value = base - without_pair
        assert pair_value > sum(s.wins_added for s in singles) + 0.05

    def test_an_unrostered_player_is_an_error(self):
        state = _tensor_state(n_teams=2, weeks=(1,))
        with pytest.raises(SeasonError, match="not on any roster"):
            leave_one_out(state, _flat_tensor(state), player_ids=[123456789])

    def test_without_a_free_agent_floor_the_kicker_is_worth_his_whole_score(self):
        """The empty-slot baseline, and why it is the wrong question for a drop.

        Every player here scores 10. Removing one leaves his slot empty and the team
        loses all 10, so a kicker prices identically to a running back -- which is how
        an empty-slot leave-one-out ends up ranking a streamable kicker above a starter.
        """
        state = _tensor_state(n_teams=2, weeks=(1,))
        tensor = _flat_tensor(state)
        rank = np.full((1, state.pool.size), 10.0, dtype=np.float32)
        eff = LineupEfficiency.perfect()
        full = team_week_scores(state, tensor, rank=rank, efficiency=eff)[0, 0, 0]
        assert full == pytest.approx(90.0)
        kicker = state.franchise(1).player_ids[8]
        thin = state.with_franchise(state.franchise(1).without(kicker))
        assert team_week_scores(thin, tensor, rank=rank, efficiency=eff)[0, 0, 0] == pytest.approx(
            80.0
        )

    def test_a_free_agent_floor_prices_him_against_the_wire_instead(self):
        """With a floor of 8 the same kicker is worth 2 points a week, not 10.

        This is the `optimal_lineup_with_floor` idea from `sim/lineup.py` carried up to
        the season level: a roster player is worth what he adds over what the waiver
        wire would have given you for free, which is the question a drop actually asks.
        """
        state = _tensor_state(n_teams=2, weeks=(1,))
        tensor = _flat_tensor(state)
        rank = np.full((1, state.pool.size), 10.0, dtype=np.float32)
        eff = LineupEfficiency.perfect()
        full = team_week_scores(state, tensor, rank=rank, efficiency=eff, replacement=8.0)
        assert full[0, 0, 0] == pytest.approx(90.0)  # everyone beats the floor, nothing changes
        kicker = state.franchise(1).player_ids[8]
        thin = state.with_franchise(state.franchise(1).without(kicker))
        floored = team_week_scores(thin, tensor, rank=rank, efficiency=eff, replacement=8.0)
        assert floored[0, 0, 0] == pytest.approx(88.0)

    def test_the_floor_benches_a_player_worse_than_the_wire(self):
        state = _tensor_state(n_teams=2, weeks=(1,))
        cols = state.pool.columns(state.franchise(1).player_ids)
        tensor = np.zeros((1, 1, state.pool.size), dtype=np.float32)
        tensor[0, 0, cols] = [10.0] * 8 + [1.0]  # a replacement-level kicker
        rank = np.zeros((1, state.pool.size), dtype=np.float32)
        rank[0, cols] = [10.0] * 8 + [1.0]
        eff = LineupEfficiency.perfect()
        # No floor: he starts for his 1 point because an empty slot scores 0.
        assert team_week_scores(state, tensor, rank=rank, efficiency=eff)[0, 0, 0] == pytest.approx(
            81.0
        )
        # Floor of 6: the slot streams instead and the roster kicker never plays.
        assert team_week_scores(state, tensor, rank=rank, efficiency=eff, replacement=6.0)[
            0, 0, 0
        ] == pytest.approx(86.0)

    def test_each_empty_slot_streams_its_own_replacement_level(self):
        """A *per-slot* floor, which is the only spelling a real waiver level has.

        Every floor test above passes a single scalar, and a scalar is exactly the input
        that cannot detect a transposition: permute seven equal numbers and nothing
        moves. `LineupPlan` carries two different orderings of its slot groups --
        `floor_slot_ids` (the caller's eligibility rows, which is what `monotone_floor`
        returns) and `group_slot_ids` (the plan's internal lexsort) -- and indexing one
        vector by the other is a silent relabelling, not a crash, because both are
        permutations of the same slot ids.

        Here every rostered player projects zero and scores zero, so every slot must
        fall to the wire and the team's total is the sum of the floors, each landing on
        its own slot. Distinct primes so no two slots can be swapped unnoticed.
        """
        state = _tensor_state(n_teams=2, weeks=(1,))
        replacement = {0: 13.0, 2: 7.0, 4: 5.0, 6: 3.0, 16: 6.0, 17: 11.0, 23: 7.0}
        # Already monotone (the FLEX floor is not below any slot nested in it), so
        # `monotone_floor` is the identity and the expected total is arithmetic.
        expected = sum(replacement[s] * n for s, n in SLOT_COUNTS.items())
        assert expected == 13 + 7 + 7 + 5 + 5 + 3 + 6 + 11 + 7
        zeros = np.zeros((1, len(state.weeks), state.pool.size), dtype=np.float32)
        got = team_week_scores(
            state,
            zeros,
            rank=zeros[0],
            efficiency=LineupEfficiency.perfect(),
            replacement=replacement,
        )
        assert got[0, 0, 0] == pytest.approx(expected)

    def test_dropping_a_player_costs_his_own_slots_replacement_not_another_slots(self):
        """The QB is worth his points over the *QB* wire, and the TE over the TE wire.

        Under a transposed floor vector the quarterback's empty seat is credited with
        the tight end's waiver level and vice versa, so the QB prices several points a
        week too high and the TE several too low -- with nothing to show for it but a
        plausible number. This pins the direction on both.
        """
        state = _tensor_state(n_teams=2, weeks=(1,))
        replacement = {0: 13.0, 2: 7.0, 4: 5.0, 6: 3.0, 16: 6.0, 17: 11.0, 23: 7.0}
        eff = LineupEfficiency.perfect()
        cols = state.pool.columns(state.franchise(1).player_ids)
        tensor = np.zeros((1, 1, state.pool.size), dtype=np.float32)
        tensor[0, 0, cols] = 20.0  # everyone beats every floor
        rank = np.zeros((1, state.pool.size), dtype=np.float32)
        rank[0, cols] = 20.0
        full = team_week_scores(state, tensor, rank=rank, efficiency=eff, replacement=replacement)
        assert full[0, 0, 0] == pytest.approx(9 * 20.0)

        roster = state.franchise(1).player_ids
        # _POSITIONS = [QB, RB, RB, WR, WR, TE, WR, DST, K]; the TE and the QB are the
        # two whose floors the transposition swaps.
        for index, slot, name in ((0, 0, "QB"), (5, 6, "TE"), (8, 17, "K")):
            thin = state.with_franchise(state.franchise(1).without(roster[index]))
            got = team_week_scores(thin, tensor, rank=rank, efficiency=eff, replacement=replacement)
            lost = float(full[0, 0, 0] - got[0, 0, 0])
            assert lost == pytest.approx(20.0 - replacement[slot]), name


# --------------------------------------------------------------------------------------
# Negative player ids
# --------------------------------------------------------------------------------------


def test_negative_player_ids_survive_the_whole_pipeline():
    """Every ESPN D/ST id is negative (`-16{proTeamId:03d}`), and every roster has one.

    Nothing here may index an array by a player id: a negative index reads silently
    from the wrong end instead of raising, which is a far worse failure than the crash
    it causes in a random-stream seed. Ordering matters too -- negatives sort first,
    and `SimPanel` sorts the same way, so the pool and the panel stay aligned.
    """
    rows = [
        (-16024, 16, 24, "Chargers D/ST"),
        (-16012, 16, 12, "Chiefs D/ST"),
        (4040715, 1, 21, "QB"),
        (4362238, 2, 4, "RB1"),
        (4239996, 2, 18, "RB2"),
        (4262921, 3, 16, "WR1"),
        (4685278, 3, 3, "WR2"),
        (4723086, 4, 3, "TE"),
        (4569987, 2, 23, "FLEX"),
        (3055899, 5, 12, "K1"),
        (3055898, 5, 12, "K2"),
    ]
    pool = PlayerPool.of(rows)
    assert pool.player_ids[0] == -16024  # negatives sort first, exactly as SimPanel does
    assert pool.name(-16024) == "Chargers D/ST"
    a = tuple(p for p, *_ in rows if p not in {-16012, 3055898})
    b = (-16012, 3055898, 4040715, 4362238, 4239996, 4262921, 4685278, 4723086, 4569987)
    state = LeagueState(
        league_id=1,
        season=2026,
        name="dst",
        franchises=(
            Franchise(team_id=1, name="T1", player_ids=a),
            Franchise(team_id=2, name="T2", player_ids=b),
        ),
        pool=pool,
        weeks=(1, 2),
        remaining_games=(
            ScheduledGame(matchup_period=1, weeks=(1,), home_team_id=1, away_team_id=2),
            ScheduledGame(matchup_period=2, weeks=(2,), home_team_id=2, away_team_id=1),
        ),
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=2,
        playoff_rounds=(),
    )
    tensor = np.zeros((4, 2, pool.size), dtype=np.float32)
    tensor[:, :, pool.columns([-16024])] = 100.0  # the defense carries team 1
    tensor[:, :, pool.columns([-16012])] = 1.0
    scores = team_week_scores(state, tensor, rank=None, efficiency=LineupEfficiency.perfect())
    assert scores[0, 0, 0] == pytest.approx(100.0)
    assert scores[0, 0, 1] == pytest.approx(1.0)
    result = simulate(state, tensor, efficiency=LineupEfficiency.perfect())
    assert result.by_team(1).expected_wins == 2.0
    [contribution] = leave_one_out(
        state, tensor, player_ids=[-16024], efficiency=LineupEfficiency.perfect()
    )
    assert contribution.name == "Chargers D/ST"
    assert contribution.wins_added == 2.0


# --------------------------------------------------------------------------------------
# Schedule and roster plumbing
# --------------------------------------------------------------------------------------


def _schedule(**kw):
    from fantasy_quant.espn.league import ScheduleConfig

    base = dict(
        matchup_period_count=14,
        matchup_period_length=1,
        matchup_periods={i: (i,) for i in range(1, 18)},
        playoff_team_count=6,
        playoff_seeding_rule="TOTAL_POINTS_SCORED",
        playoff_reseed=False,
        variable_playoff_length=False,
        playoff_matchup_period_length=1,
        playoff_length_by_round={},
        divisions=(),
    )
    base.update(kw)
    return SimpleNamespace(league_id=1, schedule=ScheduleConfig(**base))


class TestPlayoffRoundWeeks:
    def test_the_map_wins_when_it_covers_the_bracket(self):
        assert playoff_round_weeks(_schedule()) == ((15,), (16,), (17,))

    def test_a_two_week_final_is_not_halved(self):
        """ESPN's own 2026 template: a four-team bracket, one-week semi, two-week final."""
        settings = _schedule(
            matchup_period_count=15,
            playoff_team_count=4,
            matchup_periods={**{i: (i,) for i in range(1, 16)}, 16: (16,), 17: (17, 18)},
        )
        assert playoff_round_weeks(settings) == ((16,), (17, 18))

    def test_unsorted_inner_lists_are_still_ascending_on_the_way_out(self):
        """ESPN's `matchupPeriods` inner lists are documented as unsorted."""
        settings = _schedule(
            matchup_period_count=15,
            playoff_team_count=4,
            matchup_periods={**{i: (i,) for i in range(1, 16)}, 16: (16,), 17: (18, 17)},
        )
        assert playoff_round_weeks(settings)[-1] == (17, 18)

    def test_no_playoffs_means_no_rounds(self):
        assert playoff_round_weeks(_schedule(playoff_team_count=1)) == ()

    def test_a_missing_map_falls_back_to_declared_lengths(self):
        settings = _schedule(
            matchup_periods={i: (i,) for i in range(1, 15)},
            variable_playoff_length=True,
            playoff_matchup_period_length=0,
            playoff_length_by_round={1: 1, 2: 1, 3: 2},
        )
        rounds = playoff_round_weeks(settings)
        assert len(rounds) == 3
        assert rounds[-1] == (17, 18)


def _entry(player_id: int, position_id: int, eligible: tuple[int, ...], name: str = "x"):
    from fantasy_quant.espn.league import RosterEntry

    return RosterEntry(
        player_id=player_id,
        name=name,
        lineup_slot_id=20,
        default_position_id=position_id,
        pro_team_id=1,
        eligible_slots=eligible,
        injury_status="NORMAL",
        injured=False,
        status="ONTEAM",
        acquisition_type="DRAFT",
        acquisition_date=None,
        keeper_value=0.0,
        keeper_value_future=0.0,
        percent_owned=0.0,
        percent_started=0.0,
    )


class TestSlotEligibilityFromRosters:
    def test_it_reads_espns_own_answer_rather_than_a_hand_table(self):
        from fantasy_quant.espn.league import TeamRoster

        rosters = {
            1: TeamRoster(
                team_id=1,
                scoring_period=1,
                entries=(
                    _entry(1, 1, (0, 7, 20, 21)),
                    _entry(2, 2, (2, 3, 23, 7, 20, 21)),
                    _entry(3, 3, (3, 4, 5, 23, 7, 20, 21)),
                    _entry(4, 4, (5, 6, 23, 7, 20, 21)),
                    _entry(5, 16, (16, 20, 21)),
                    _entry(6, 5, (17, 20, 21)),
                ),
            )
        }
        got = slot_eligibility_from_rosters(rosters, SLOT_COUNTS)
        assert got[0] == frozenset({1})
        assert got[23] == frozenset({2, 3, 4})  # FLEX, derived not assumed
        assert got[4] == frozenset({3})  # slot 4 is WR, position 4 is TE
        assert got[6] == frozenset({4})

    def test_a_starting_slot_nobody_can_fill_is_an_error(self):
        from fantasy_quant.espn.league import TeamRoster

        rosters = {1: TeamRoster(team_id=1, scoring_period=1, entries=(_entry(1, 1, (0, 20)),))}
        with pytest.raises(SeasonError, match="eligible"):
            slot_eligibility_from_rosters(rosters, SLOT_COUNTS)


def test_historical_all_play_reconstructs_the_played_weeks():
    from fantasy_quant.espn.league import Matchup, MatchupSide
    from fantasy_quant.sim.season import _historical_all_play

    def side(team_id: int, pts: dict[int, float]) -> MatchupSide:
        return MatchupSide(
            team_id=team_id,
            total_points=sum(pts.values()),
            points_by_scoring_period=pts,
            wins=0,
            losses=0,
            ties=0,
            adjustment=0.0,
        )

    matchups = [
        Matchup(1, 1, "HOME", None, side(1, {1: 120.0}), side(2, {1: 100.0})),
        Matchup(2, 1, "AWAY", None, side(3, {1: 90.0}), side(4, {1: 130.0})),
        Matchup(3, 2, "UNDECIDED", None, side(1, {2: 0.0}), side(3, {2: 0.0})),
    ]
    wins, games = _historical_all_play(matchups)
    # Week 1 only: 130 > 120 > 100 > 90.
    assert wins == {4: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
    assert games == {1: 3, 2: 3, 3: 3, 4: 3}


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


class TestValidation:
    def test_pool_ids_must_be_ascending_to_match_the_panel(self):
        with pytest.raises(SeasonError, match="ascending"):
            PlayerPool(player_ids=(5, 1), position_ids=(1, 2), pro_team_ids=(1, 1))

    def test_pool_rejects_duplicates(self):
        with pytest.raises(SeasonError, match="duplicate"):
            PlayerPool(player_ids=(1, 1), position_ids=(1, 2), pro_team_ids=(1, 1))

    def test_a_roster_player_outside_the_pool_is_an_error(self):
        state = _tensor_state(n_teams=2, weeks=(1,))
        broken = state.with_franchise(
            Franchise(team_id=1, name="T1", player_ids=(999999,), is_user=True)
        )
        with pytest.raises(SeasonError, match="not in the pool"):
            team_week_scores(broken, _flat_tensor(state))

    def test_more_playoff_spots_than_teams_is_an_error(self):
        with pytest.raises(SeasonError, match="more playoff spots"):
            _bracket_state(n_teams=4, playoff_team_count=6)

    def test_replacing_an_unknown_franchise_is_an_error(self):
        state = _tensor_state(n_teams=2, weeks=(1,))
        with pytest.raises(SeasonError, match="no team"):
            state.with_franchise(Franchise(team_id=77, name="ghost", player_ids=()))

    def test_replacing_a_franchise_with_an_equal_one_is_a_no_op_not_an_error(self):
        """A null candidate is a legitimate thing for a search loop to price.

        `Franchise` is a frozen value type, so swapping one in for an equal one leaves
        the tuple identical. A "did the tuple change" guard then reports a team that is
        plainly in the league as missing -- and the error says `no team 2`, which sends
        the reader hunting for a roster bug that does not exist.
        """
        state = _tensor_state(n_teams=2, weeks=(1,))
        same = state.with_franchise(state.franchise(2))
        assert same.franchises == state.franchises
        assert same.franchise(2) == state.franchise(2)
        # And a genuine null move still prices at exactly zero, which is the property
        # the no-op has to preserve.
        tensor = _flat_tensor(state)
        eff = LineupEfficiency.perfect()
        assert np.array_equal(
            simulate(state, tensor, efficiency=eff).wins,
            simulate(same, tensor, efficiency=eff).wins,
        )


# --------------------------------------------------------------------------------------
# Corpus calibration
# --------------------------------------------------------------------------------------


def _corpus_league(
    *,
    need: dict[int, int],
    slot_counts: dict[int, int],
    n_teams: int = 12,
) -> tuple[LeagueState, list[PlayerOutlook]]:
    """A snake-drafted 12-team PPR league built out of the 2026 snapshot.

    Real ESPN weekly projections through the real calibration curves, so the team-score
    moments this produces are directly comparable to the corpus anchor. D/ST ids are
    remapped positive only because `sim/distributions.py` cannot currently seed a
    stream for a negative id; the mapping is injective and affects nothing but labels.
    """
    import polars as pl

    from fantasy_quant.corpus import SOURCE_PROJECTED, SPLIT_GAME, load_stat_rows
    from fantasy_quant.projections import calibration as cal

    weeks = tuple(range(1, 18))
    rows = load_stat_rows([2026], root=CORPUS).filter(
        (pl.col("stat_split_type_id") == SPLIT_GAME)
        & (pl.col("stat_source_id") == SOURCE_PROJECTED)
        & (pl.col("scoring_period_id").is_in(list(weeks)))
    )
    cset = cal.load(directory=REPO / "data" / "reference")

    proj: dict[int, dict[int, float]] = {}
    meta: dict[int, tuple[int, int, str]] = {}
    for r in rows.iter_rows(named=True):
        pid = r["espn_id"]
        pid = pid if pid >= 0 else 9_000_000 - pid
        proj.setdefault(pid, {})[r["scoring_period_id"]] = r["applied_total"]
        meta[pid] = (r["default_position_id"], r["pro_team_id"], r["full_name"])

    total = {p: sum(v.get(w, 0.0) for w in range(1, 15)) for p, v in proj.items()}
    pools: dict[int, list[int]] = {}
    for p, (pos, _, _) in meta.items():
        pools.setdefault(pos, []).append(p)
    for pos in pools:
        pools[pos].sort(key=lambda p: -total[p])

    rosters: list[list[int]] = [[] for _ in range(n_teams)]
    cursor = {pos: 0 for pos in need}
    draft_order = [slot for slot, n in need.items() for _ in range(n)]
    for rnd, pos in enumerate(draft_order):
        seq = range(n_teams) if rnd % 2 == 0 else reversed(range(n_teams))
        for t in seq:
            rosters[t].append(pools[pos][cursor[pos]])
            cursor[pos] += 1

    outlooks = [
        PlayerOutlook(
            pid,
            meta[pid][2],
            meta[pid][0],
            meta[pid][1],
            {
                w: cset.outlook(
                    player_id=pid,
                    season=2026,
                    week=w,
                    position_id=meta[pid][0],
                    projection=proj[pid].get(w, 0.0),
                    pro_team_id=meta[pid][1],
                    playing=proj[pid].get(w, 0.0) > 0,
                )
                for w in weeks
            },
        )
        for roster in rosters
        for pid in roster
    ]
    pool = PlayerPool.of((p, *meta[p][:2], meta[p][2]) for r in rosters for p in r)
    games = tuple(
        ScheduledGame(matchup_period=w, weeks=(w,), home_team_id=a + 1, away_team_id=b + 1)
        for w, a, b in _round_robin(n_teams, range(1, 15))
    )
    eligibility = {s: SLOT_ELIGIBILITY[s] for s in slot_counts}
    state = LeagueState(
        league_id=0,
        season=2026,
        name="anchor",
        franchises=tuple(
            Franchise(team_id=i + 1, name=f"T{i + 1}", player_ids=tuple(rosters[i]))
            for i in range(n_teams)
        ),
        pool=pool,
        weeks=weeks,
        remaining_games=games,
        lineup_slot_counts=slot_counts,
        slot_eligibility=eligibility,
        playoff_team_count=6,
        playoff_rounds=((15,), (16,), (17,)),
    )
    return state, outlooks


@pytest.mark.skipif(not CORPUS.exists(), reason="no snapshot corpus in the repo")
class TestCorpusAnchor:
    """Simulated team scores against the measured 12-team PPR anchor.

    The anchor is "12-team PPR, **nine skill starters**, optimal lineups: mean 121.9,
    SD 24.35, skew +0.27". Nine *skill* starters means 1QB/2RB/3WR/1TE/2FLEX plus a
    defense and a kicker -- eleven starters in all. Reproducing it at that shape is the
    check; the user's own leagues start seven skill players and land near 105, which is
    the same model with two fewer starters and is not a calibration failure.
    """

    NEED_DEEP = {1: 2, 2: 5, 3: 7, 4: 2, 5: 1, 16: 1}
    SLOTS_DEEP = {0: 1, 2: 2, 4: 3, 6: 1, 16: 1, 17: 1, 23: 2}

    @pytest.fixture(scope="class")
    @classmethod
    def deep(cls):
        from fantasy_quant.sim.distributions import WeeklySampler

        state, outlooks = _corpus_league(need=cls.NEED_DEEP, slot_counts=cls.SLOTS_DEEP)
        draw = WeeklySampler(panel_for(state, outlooks), seed=7).draw(2000)
        return state, draw

    def test_team_score_moments_match_the_corpus(self, deep):
        state, draw = deep
        scores = team_week_scores(state, draw, efficiency=LineupEfficiency.symmetric())
        regular = scores[:, :14, :].ravel()
        assert regular.mean() == pytest.approx(ANCHOR_TEAM_MEAN, rel=0.06)
        assert regular.std() == pytest.approx(ANCHOR_TEAM_SD, rel=0.10)
        assert float(skew(regular)) == pytest.approx(ANCHOR_TEAM_SKEW, abs=0.10)

    def test_a_shallower_lineup_scores_less(self, deep):
        """The user's own 7-skill-starter leagues, same model, ~15 points lighter."""
        from fantasy_quant.sim.distributions import WeeklySampler

        state, outlooks = _corpus_league(
            need={1: 2, 2: 5, 3: 6, 4: 1, 5: 1, 16: 1}, slot_counts=SLOT_COUNTS
        )
        draw = WeeklySampler(panel_for(state, outlooks), seed=7).draw(500)
        scores = team_week_scores(state, draw, efficiency=LineupEfficiency.symmetric())
        assert 95.0 < scores[:, :14, :].mean() < 115.0

    def test_the_hindsight_ratio_is_where_the_double_count_lives(self, deep):
        """0.88-0.92, so the measured 0.775 haircut belongs near 0.85 on ex-ante lineups."""
        state, draw = deep
        ratio = measure_hindsight_ratio(state, draw)
        assert 0.85 < ratio < 0.95
        assert 0.80 < rescale_for_ex_ante_lineups(ratio) < 0.92

    def test_the_ex_ante_rank_never_starts_an_injured_player(self, deep):
        state, draw = deep
        rank = ex_ante_rank(draw)
        assert np.isneginf(rank[~draw.available]).all()
        assert (rank[draw.available] >= 0).all()

    def test_title_probabilities_sum_to_one_on_a_real_pool(self, deep):
        state, draw = deep
        result = simulate(state, draw, efficiency=LineupEfficiency.symmetric())
        assert sum(result.title_odds().values()) == pytest.approx(1.0)
        assert all(0.0 <= o.championship <= 1.0 for o in result.outcomes())
        # Nobody is a lock and nobody is dead: a 12-team league is not that predictable.
        odds = sorted(o.championship for o in result.outcomes())
        assert odds[0] > 0.005
        assert odds[-1] < 0.40

    def test_panel_and_pool_must_agree(self, deep):
        state, _ = deep
        with pytest.raises(SeasonError, match="no outlook"):
            panel_for(state, [])

    def test_a_source_that_covers_only_some_weeks_is_rejected(self):
        """The failure a per-player check waves through, and it is not hypothetical.

        ESPN's `mRoster` returns roughly five stat rows per player -- the periods asked
        for, not the rest of the season. `SimPanel.from_outlooks` zero-fills every week
        it is not handed, so a partial source yields a panel that is almost all zeros:
        every team scores nothing, every game ties, and the standings come out 0-0-14
        with title odds spread evenly across the league. Nothing raises, and the numbers
        look like a modelling opinion rather than a plumbing failure.
        """
        state, outlooks = _corpus_league(need=self.NEED_DEEP, slot_counts=self.SLOTS_DEEP)
        keep = set(state.weeks[:3])
        clipped = [
            PlayerOutlook(
                o.player_id,
                o.name,
                o.position_id,
                o.pro_team_id,
                {w: v for w, v in o.weeks.items() if w in keep},
            )
            for o in outlooks
        ]
        with pytest.raises(SeasonError, match="only part of the remaining season"):
            panel_for(state, clipped)
        # The full source still passes, so the guard is not merely rejecting everything.
        assert panel_for(state, outlooks).n_weeks == len(state.weeks)


# --------------------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------------------


class TestPerformance:
    """The candidate search runs this thousands of times; constants are the design.

    A full 12-team, 17-week, 2,000-simulation season -- lineups re-solved for every
    franchise, the whole remaining schedule played, all-play computed pairwise, and a
    three-round bracket resolved -- has to cost milliseconds, not seconds. Measured on
    an M-series laptop it lands near 0.26s with all-play on and 0.10s without, and the
    leave-one-out loop costs about 50ms per player because only the affected
    franchise's lineups are re-solved.
    """

    def _rig(self):
        state = _tensor_state(
            n_teams=12,
            weeks=tuple(range(1, 18)),
            playoff_rounds=((15,), (16,), (17,)),
            playoff_team_count=6,
            per_team=16,
        )
        rng = np.random.default_rng(101)
        tensor = rng.gamma(2.0, 5.0, (2000, 17, state.pool.size)).astype(np.float32)
        rank = np.tile(rng.uniform(2.0, 20.0, state.pool.size).astype(np.float32), (17, 1))
        return state, tensor, rank

    def test_a_full_season_is_sub_second(self):
        state, tensor, rank = self._rig()
        simulate(state, tensor, rank=rank)  # warm the plan compile and numpy's caches
        start = time.perf_counter()
        for _ in range(3):
            simulate(state, tensor, rank=rank)
        elapsed = (time.perf_counter() - start) / 3
        assert elapsed < 1.5, f"a 2000-sim season took {elapsed:.3f}s"

    def test_dropping_all_play_is_cheaper(self):
        """All-play is the expensive column -- pairwise per week -- and it is optional."""
        state, tensor, rank = self._rig()
        simulate(state, tensor, rank=rank, all_play=False)
        start = time.perf_counter()
        simulate(state, tensor, rank=rank, all_play=False)
        without = time.perf_counter() - start
        start = time.perf_counter()
        simulate(state, tensor, rank=rank, all_play=True)
        assert without <= (time.perf_counter() - start) + 1e-6

    def test_leave_one_out_re_solves_one_roster_not_twelve(self):
        state, tensor, rank = self._rig()
        mine = state.franchise(1).player_ids
        start = time.perf_counter()
        got = leave_one_out(state, tensor, player_ids=list(mine), rank=rank, all_play=False)
        per_player = (time.perf_counter() - start) / len(got)
        assert len(got) == len(mine)
        assert per_player < 0.5, f"{per_player * 1000:.0f} ms per player"


# --------------------------------------------------------------------------------------
# Live leagues
# --------------------------------------------------------------------------------------


@pytest.mark.network
class TestLiveLeagues:
    """Against the user's real leagues. Read-only, and asserts only structure."""

    LEAGUES = ((272150391, 1, 14), (161496047, 1, 12), (634537479, 2, 12))

    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):
        import os

        from dotenv import load_dotenv

        from fantasy_quant.espn.client import EspnClient

        load_dotenv(REPO / ".env")
        swid, s2 = os.environ.get("ESPN_SWID"), os.environ.get("ESPN_S2")
        if not swid or not s2:
            pytest.skip("no ESPN credentials in .env")
        with EspnClient(swid=swid, espn_s2=s2) as c:
            yield c

    def test_state_assembles_from_every_real_league(self, client):
        from fantasy_quant.espn.league import League

        for league_id, my_team, size in self.LEAGUES:
            state = state_from_league(League(client, league_id, 2026), my_team_id=my_team)
            assert state.size == size
            assert state.playoff_team_count == 6
            assert state.playoff_rounds == ((15,), (16,), (17,))
            assert state.weeks[-1] == 17
            assert state.bye_count == 2
            assert len(state.remaining_games) == (size // 2) * 14
            assert state.franchise(my_team).is_user
            # Every roster carries a D/ST, and every D/ST id is negative.
            assert any(p < 0 for p in state.franchise(my_team).player_ids)
            assert all(p.laminar for p in lineup_plans(state))


class TestTheShippedDefaultIsAssumptionFree:
    """The default must not manufacture an edge the user has not earned.

    An asymmetric haircut is defensible in principle but unverifiable in practice,
    and at the published 0.775 it turns a below-baseline roster into a title
    favourite in all three real leagues. Whatever else the tool gets wrong, its
    out-of-the-box answer should not be an artefact of that choice.
    """

    def test_the_default_is_symmetric(self):
        eff = LineupEfficiency()
        assert eff.opponent_mean == 1.0
        assert eff.user_mean == 1.0
        assert eff.opponent_sd == 0.0
        assert not eff.is_asymmetric

    def test_identical_rosters_get_identical_odds_by_default(self):
        """The single clearest statement that no free edge is being handed out."""
        state = _tensor_state(
            n_teams=12,
            weeks=tuple(range(1, 18)),
            playoff_rounds=((15,), (16,), (17,)),
            playoff_team_count=6,
            per_team=16,
        )
        rng = np.random.default_rng(11)
        tensor = rng.gamma(3.0, 4.0, (2000, 17, state.pool.size)).astype(np.float32)
        rank = np.tile(rng.uniform(2.0, 20.0, state.pool.size).astype(np.float32), (17, 1))
        result = simulate(state, tensor, rank=rank, all_play=False)
        odds = [result.by_team(t).championship for t in range(1, 13)]
        assert sum(odds) == pytest.approx(1.0, abs=1e-6)
        assert max(odds) - min(odds) < 0.06, "identical rosters should be near-uniform"

    def test_the_published_constant_is_still_reachable_and_named(self):
        assert LineupEfficiency.literal().opponent_mean == OPPONENT_LINEUP_EFFICIENCY
        assert LineupEfficiency.literal().is_asymmetric
