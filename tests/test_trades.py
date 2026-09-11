"""Trade discovery tests.

The fixtures are hand-built four-team leagues with *constant* weekly projections, so
every objective in this file can be checked by adding numbers up by hand: the week
weights conserve to the week count, so a player worth `m` a week is worth `m * n_weeks`
over the horizon and nothing else has to be believed.

Three claims carry the module and each has a test whose failure would be unambiguous:

* the Pareto gate is a gate, not a preference -- a trade that helps only one side is
  rejected even when the league-wide total goes up;
* the free-agent floor, not a bolted-on constant, is what makes a 2-for-1 favour the
  consolidating side, so the same trade priced without a wire favours it *less*;
* the preference graph must be multi-edge -- the single-edge form is constructed here
  to fail on a concentrated market, which is the market fantasy actually has.

Only `test_confirm_*` draws a tensor, and it draws a small one. Everything runs offline.
"""

from __future__ import annotations

import dataclasses
import math
from itertools import combinations

import numpy as np
import pytest

from fantasy_quant.core import Move, MoveKind, PlayerOutlook, Recommendation, WeeklyOutlook
from fantasy_quant.decide.trades import (
    MEASURED_SD_DIFF,
    PLAYOFF_WEIGHT,
    TradeError,
    TradeEvaluation,
    TradeFinder,
    TradeLeg,
    TradeProposal,
    best_free_agents,
    find_trades,
    playoff_weights,
    positional_requirements,
    select_non_overlapping,
    selection_threshold,
    simple_cycles,
    single_edge_targets,
    surplus_multiplier,
    wire_pool,
)
from fantasy_quant.sim import season as S
from fantasy_quant.sim.distributions import WeeklySampler

QB, RB, WR, TE = 1, 2, 3, 4

#: 1QB/2RB/2WR/1TE/1FLEX, which is the user's real shape with K and D/ST dropped --
#: they never move in a trade and they double the fixture for nothing.
SLOT_COUNTS = {0: 1, 2: 2, 4: 2, 6: 1, 23: 1}
SLOT_ELIGIBILITY = {
    0: frozenset({QB}),
    2: frozenset({RB}),
    4: frozenset({WR}),
    6: frozenset({TE}),
    23: frozenset({RB, WR, TE}),
}

WEEKS = (1, 2, 3, 4)
PLAYOFF_ROUNDS = ((3,), (4,))
SEASON = 2026


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _outlook(player_id: int, position: int, mu: float, name: str) -> PlayerOutlook:
    """A player with the same projection every week. Byes would only obscure the sums."""
    return PlayerOutlook(
        player_id=player_id,
        name=name,
        position_id=position,
        pro_team_id=1 + player_id % 30,
        weeks={
            w: WeeklyOutlook(
                player_id=player_id,
                season=SEASON,
                week=w,
                position_id=position,
                mean=mu,
                # The measured sigma(mu) relation, so a drawn tensor is not absurd.
                sd=3.67 + 0.273 * mu,
                p_zero=0.1,
                shape=2.0,
                scale=max(mu, 0.5) / 2.0,
                pro_team_id=1 + player_id % 30,
                playing=True,
            )
            for w in WEEKS
        },
    )


def _market(
    rosters: dict[int, list[tuple[int, float, str]]],
    wire: list[tuple[int, float, str]] | None = None,
    *,
    my_team_id: int | None = 1,
) -> tuple[S.LeagueState, list[PlayerOutlook]]:
    """A four-team league from `{team_id: [(position, weekly mu, name), ...]}`.

    `wire` is the free-agent pool: projected players on nobody's roster, which is the
    only thing that makes the free-agent floor a league-specific number.
    """
    outlooks: list[PlayerOutlook] = []
    team_players: dict[int, list[int]] = {}
    pid = 100
    for team, players in sorted(rosters.items()):
        ids = []
        for pos, mu, name in players:
            pid += 1
            outlooks.append(_outlook(pid, pos, mu, name))
            ids.append(pid)
        team_players[team] = ids
    for pos, mu, name in wire or []:
        pid += 1
        outlooks.append(_outlook(pid, pos, mu, f"FA {name}"))

    rostered = {p for ids in team_players.values() for p in ids}
    pool = S.PlayerPool.of(
        (o.player_id, o.position_id, o.pro_team_id, o.name)
        for o in outlooks
        if o.player_id in rostered
    )
    franchises = tuple(
        S.Franchise(
            team_id=t,
            name=f"Team {t}",
            player_ids=tuple(team_players[t]),
            is_user=(t == my_team_id),
        )
        for t in sorted(team_players)
    )
    games = (
        S.ScheduledGame(matchup_period=1, weeks=(1,), home_team_id=1, away_team_id=2),
        S.ScheduledGame(matchup_period=1, weeks=(1,), home_team_id=3, away_team_id=4),
        S.ScheduledGame(matchup_period=2, weeks=(2,), home_team_id=1, away_team_id=3),
        S.ScheduledGame(matchup_period=2, weeks=(2,), home_team_id=2, away_team_id=4),
    )
    state = S.LeagueState(
        league_id=42,
        season=SEASON,
        name="fixture",
        franchises=franchises,
        pool=pool,
        weeks=WEEKS,
        remaining_games=games,
        lineup_slot_counts=SLOT_COUNTS,
        slot_eligibility=SLOT_ELIGIBILITY,
        playoff_team_count=4,
        playoff_rounds=PLAYOFF_ROUNDS,
        my_team_id=my_team_id,
    )
    return state, outlooks


def _dummy_draw(state: S.LeagueState, outlooks: list[PlayerOutlook], n_sims: int = 64):
    panel = S.panel_for(state, outlooks)
    return WeeklySampler(panel, seed=7).draw(n_sims)


def _finder(rosters, wire=None, **kwargs) -> TradeFinder:
    state, outlooks = _market(rosters, wire)
    return TradeFinder(state, _dummy_draw(state, outlooks), outlooks, **kwargs)


def _by_name(finder: TradeFinder, name: str) -> int:
    for pid, label in finder._name.items():
        if label == name:
            return pid
    raise KeyError(name)


#: The 2-for-1 fixture, worked out by hand in the module tests below.
#:
#: Team 1 is the consolidator: it starts a two-point tight end behind a seven-point
#: wire, so that tight end is worth exactly nothing to give away. Team 2 is running-back
#: starved and receiver rich, which is what makes the trade Pareto rather than a mugging.
CONSOLIDATION = {
    1: [
        (QB, 18.0, "QB1"),
        (RB, 14.0, "RB good A"),
        (RB, 12.0, "RB traded"),
        (RB, 9.0, "RB3"),
        (RB, 4.0, "RB4"),
        (WR, 13.0, "WR1"),
        (WR, 11.0, "WR2"),
        (WR, 7.0, "WR3"),
        (WR, 5.0, "WR4"),
        (TE, 2.0, "TE sub-wire"),
    ],
    2: [
        (QB, 16.0, "QB2"),
        (RB, 6.0, "RB weak 1"),
        (RB, 5.0, "RB weak 2"),
        (RB, 4.0, "RB weak 3"),
        (RB, 3.0, "RB weak 4"),
        (WR, 22.0, "WR elite"),
        (WR, 18.0, "WR B1"),
        (WR, 17.0, "WR B2"),
        (WR, 16.0, "WR B3"),
        (TE, 11.0, "TE B"),
    ],
    3: [
        (QB, 15.0, "QB3"),
        (RB, 10.0, "RB C1"),
        (RB, 8.0, "RB C2"),
        (RB, 6.0, "RB C3"),
        (WR, 12.0, "WR C1"),
        (WR, 10.0, "WR C2"),
        (WR, 8.0, "WR C3"),
        (WR, 6.0, "WR C4"),
        (TE, 9.0, "TE C"),
        (TE, 8.0, "TE C2"),
    ],
    4: [
        (QB, 14.0, "QB4"),
        (RB, 11.0, "RB D1"),
        (RB, 9.0, "RB D2"),
        (RB, 7.0, "RB D3"),
        (WR, 14.0, "WR D1"),
        (WR, 9.0, "WR D2"),
        (WR, 7.0, "WR D3"),
        (WR, 6.0, "WR D4"),
        (TE, 10.0, "TE D"),
        (TE, 7.0, "TE D2"),
    ],
}

WIRE = [
    (QB, 10.0, "QB"),
    (QB, 9.0, "QB2"),
    (RB, 5.0, "RB"),
    (RB, 4.0, "RB2"),
    (WR, 6.0, "WR"),
    (WR, 5.0, "WR2"),
    (TE, 7.0, "TE"),
    (TE, 6.0, "TE2"),
]


# --------------------------------------------------------------------------------------
# Playoff weighting
# --------------------------------------------------------------------------------------


def test_playoff_weighting_is_points_conserving():
    """1.2 on the bracket, and the regular season gives back exactly what it costs."""
    weeks = tuple(range(1, 18))
    w = playoff_weights(weeks, (15, 16, 17))
    assert sum(w.weights) == pytest.approx(17.0)
    assert w.weight_of(16) == pytest.approx(PLAYOFF_WEIGHT)
    assert w.weight_of(9) == pytest.approx((17 - 1.2 * 3) / 14)
    # The whole point of conserving: an unweighted horizon and a weighted one agree on a
    # player who is the same every week, so the scheme cannot inflate a trade by itself.
    flat = playoff_weights(weeks, ())
    assert sum(flat.weights) == pytest.approx(sum(w.weights))


def test_playoff_weighting_degenerates_to_flat_when_there_is_nothing_to_reallocate():
    assert playoff_weights((15, 16, 17), (15, 16, 17)).weights == (1.0, 1.0, 1.0)
    assert playoff_weights((1, 2, 3), (15,)).weights == (1.0, 1.0, 1.0)


def test_playoff_weighting_refuses_a_weight_that_starves_the_regular_season():
    with pytest.raises(TradeError):
        playoff_weights((1, 2, 3, 4), (2, 3, 4), playoff_weight=1.4)


# --------------------------------------------------------------------------------------
# Positional surplus
# --------------------------------------------------------------------------------------


def test_surplus_multiplier_discounts_the_redundant_and_never_rewards_it():
    required = 2 + 1 / 3  # 2 dedicated RB slots plus a third of a three-way flex
    assert surplus_multiplier(2.0, required) == pytest.approx(1.0)  # short: no bonus
    assert surplus_multiplier(4.0, required) == pytest.approx(1 - (4 - required) / 4)
    assert surplus_multiplier(5.0, required) < surplus_multiplier(4.0, required)
    assert surplus_multiplier(0.0, required) == 1.0


def test_positional_requirements_split_the_flex_evenly():
    req = positional_requirements(SLOT_COUNTS, SLOT_ELIGIBILITY)
    assert req[QB] == pytest.approx(1.0)
    assert req[RB] == pytest.approx(2 + 1 / 3)
    assert req[TE] == pytest.approx(1 + 1 / 3)


def test_surplus_decay_reduces_the_value_of_a_redundant_starter():
    """A fourth startable back is discounted; the discount is exactly the multiplier."""
    finder = _finder(CONSOLIDATION, WIRE)
    back = _by_name(finder, "RB traded")
    crowded = finder.rosters[3]  # three backs, all of them above the wire
    req = finder.requirements[RB]

    raw = finder.value_of([*crowded, back]) - finder.value_of(crowded)
    decayed = finder.asset_value(3, back, incoming=True)
    multiplier = surplus_multiplier(finder.startable_count(3, RB) + 1, req)
    assert raw > 0
    assert multiplier < 1.0
    assert decayed == pytest.approx(raw * multiplier)
    assert decayed < raw


def test_surplus_is_counted_in_startable_bodies_not_headcount():
    """Four backs who are all worse than the wire are a hole, not a surplus.

    Team 2 carries four running backs and three of them are below the waiver wire. A
    headcount screen charges it a 47% surplus discount on the back it desperately needs
    and talks it out of the one trade that helps it; counting against the wire does not.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    headcount = sum(1 for p in finder.rosters[2] if finder._pos[p] == RB)
    assert headcount == 4
    assert finder.startable_count(2, RB) == 1
    req = finder.requirements[RB]
    assert surplus_multiplier(headcount + 1, req) < 1.0
    assert surplus_multiplier(finder.startable_count(2, RB) + 1, req) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# The free-agent floor
# --------------------------------------------------------------------------------------


def _bye_outlook(player_id: int, position: int, per_week: dict[int, float], name: str):
    """A wire body with a real weekly shape, so byes and one-week spikes are visible."""
    base = _outlook(player_id, position, 1.0, name)
    return PlayerOutlook(
        player_id=base.player_id,
        name=name,
        position_id=position,
        pro_team_id=base.pro_team_id,
        weeks={
            w: WeeklyOutlook(
                player_id=player_id,
                season=SEASON,
                week=w,
                position_id=position,
                mean=per_week[w],
                sd=3.67 + 0.273 * per_week[w],
                p_zero=0.1,
                shape=2.0,
                scale=max(per_week[w], 0.5) / 2.0,
                pro_team_id=base.pro_team_id,
                playing=per_week[w] > 0.0,
            )
            for w in WEEKS
        },
    )


def test_wire_depth_models_waiver_priority_not_a_per_week_envelope():
    """The floor is bodies you could hold, not the best projection in the league.

    Taking a per-week maximum over the whole wire models a manager who signs whichever
    kicker projects best every single week at every position at once. All three of the
    user's leagues run rolling waiver *priority* -- one claim per run, and using it drops
    you to last -- so nobody gets that. Measured on Wine Wednesday, the envelope prices
    the kicker slot at 8.44 a week when the best kicker anyone could actually hold
    averages 6.3, and the quarterback slot at 14.48 against a holdable 12.8.
    """
    state, outlooks = _market(CONSOLIDATION, [])
    hold = _bye_outlook(9001, TE, {1: 9.0, 2: 9.0, 3: 9.0, 4: 0.0}, "wire TE, bye in 4")
    backup = _bye_outlook(9002, TE, {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0}, "wire TE 2")
    spike = _bye_outlook(9003, TE, {1: 0.0, 2: 0.0, 3: 0.0, 4: 20.0}, "one-week wonder")
    pool = [*outlooks, hold, backup, spike]

    one = best_free_agents(state, pool, depth=1)[TE]
    assert list(one) == [9.0, 9.0, 9.0, 0.0]  # hold one body; his bye is your problem

    three = best_free_agents(state, pool, depth=3)[TE]
    assert list(three) == [9.0, 9.0, 9.0, 20.0]  # the whole wire is only three deep here

    two = best_free_agents(state, pool, depth=2)[TE]
    assert list(two) == [9.0, 9.0, 9.0, 5.0]  # one claim covers the bye, nothing more

    # And `rank` slides the pool past the top of the wire entirely.
    assert list(best_free_agents(state, pool, rank=2, depth=1)[TE]) == [5.0, 5.0, 5.0, 5.0]


def test_best_free_agents_reads_this_league_s_own_wire():
    state, outlooks = _market(CONSOLIDATION, WIRE)
    fa = best_free_agents(state, outlooks)
    assert fa[TE][0] == pytest.approx(7.0)
    assert fa[RB][0] == pytest.approx(5.0)
    # Rank 2 is the marginal wire body rather than the best one.
    assert best_free_agents(state, outlooks, rank=2)[TE][0] == pytest.approx(6.0)
    # Nobody on a roster is a free agent, however bad he is.
    assert fa[WR][0] == pytest.approx(6.0)


def test_a_slot_no_rostered_player_can_fill_does_not_float_the_whole_lineup():
    """The regression that cost +82pp of title probability in a live league.

    `monotone_floor` lifts a slot's floor to the highest floor among the slots nested
    inside it, and nesting is over players. Strip a roster of its only quarterback and
    the QB slot's player set is empty; an empty set is a subset of every other slot, so
    without the fillable-slot split every running back, receiver and kicker seat gets
    floored at the wire quarterback's 10 points. Measured on Blacksburg before the fix,
    that made "trade Jalen Hurts for a kicker" worth +551 points and +82pp of title.

    The right answer is arithmetic: an unfillable QB slot scores the wire quarterback
    and nothing else moves.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    roster = finder.rosters[1]
    qb = _by_name(finder, "QB1")  # projected 18; the wire quarterback is 10
    qb_less = tuple(p for p in roster if p != qb)

    live, dead = finder.live_slots(finder._pos[p] for p in qb_less)
    assert dead == (0,)  # the QB slot, and only the QB slot
    assert 0 not in live

    assert finder.value_of(roster) == pytest.approx(84.0 * len(WEEKS))
    assert finder.value_of(qb_less) == pytest.approx((84.0 - 18.0 + 10.0) * len(WEEKS))
    assert finder.value_of(roster) - finder.value_of(qb_less) == pytest.approx(
        (18.0 - 10.0) * len(WEEKS)
    )


def test_a_starter_below_the_wire_is_worth_nothing_to_give_away():
    """The rule the whole verdict layer rests on, isolated from any trade."""
    finder = _finder(CONSOLIDATION, WIRE)
    bad_te = _by_name(finder, "TE sub-wire")
    roster = finder.rosters[1]
    without = tuple(p for p in roster if p != bad_te)
    assert finder.value_of(roster) == pytest.approx(finder.value_of(without))

    # Take the wire away and the same player is suddenly worth his full projection,
    # which is the accounting a chart-sum model does and the reason it hates 2-for-1s.
    no_wire = _finder(CONSOLIDATION, [])
    assert no_wire.value_of(roster) - no_wire.value_of(without) == pytest.approx(2.0 * len(WEEKS))


def test_free_agent_floor_makes_a_two_for_one_favour_the_consolidating_side():
    """Team 1 gives two and gets one; the floor is what makes that a good idea."""
    finder = _finder(CONSOLIDATION, WIRE)
    rb = _by_name(finder, "RB traded")
    te = _by_name(finder, "TE sub-wire")
    elite = _by_name(finder, "WR elite")
    proposal = TradeProposal(
        league_id=42,
        legs=(
            TradeLeg(from_team=1, to_team=2, player_ids=(rb, te)),
            TradeLeg(from_team=2, to_team=1, player_ids=(elite,)),
        ),
    )
    ev = finder.evaluate(proposal)
    consolidator = ev.impact_for(1)
    expander = ev.impact_for(2)

    assert ev.pareto
    assert consolidator.delta_points > expander.delta_points > 0
    # 84 -> 94 points a week, times a horizon whose weights sum to the week count.
    assert consolidator.delta_points == pytest.approx(10.0 * len(WEEKS))

    # Price the identical trade with no wire at all and the consolidator is charged the
    # full sticker price of the throw-in, so his gain falls. That difference IS the
    # consolidation credit; no constant is added anywhere.
    dry = _finder(CONSOLIDATION, [])
    assert dry.evaluate(proposal).impact_for(1).delta_points == pytest.approx(8.0 * len(WEEKS))
    assert dry.evaluate(proposal).impact_for(1).delta_points < consolidator.delta_points


def test_the_side_receiving_two_for_one_has_to_cut():
    """Nobody's roster grows for free -- the mirror image of the freed spot."""
    finder = _finder(CONSOLIDATION, WIRE)
    rb = _by_name(finder, "RB traded")
    te = _by_name(finder, "TE sub-wire")
    elite = _by_name(finder, "WR elite")
    ev = finder.evaluate(
        TradeProposal(
            42,
            (
                TradeLeg(1, 2, (rb, te)),
                TradeLeg(2, 1, (elite,)),
            ),
        )
    )
    assert len(ev.impact_for(2).dropped) == 1
    assert len(ev.rosters[2]) == len(finder.rosters[2])
    # And the consolidator's freed seat is filled from the wire or left empty, but it is
    # never allowed to be worth more than the floor already credits it.
    assert len(ev.rosters[1]) <= len(finder.rosters[1])


def test_a_freed_roster_spot_is_priced_from_the_wire_not_asserted():
    """The 425-point roster spot, measured: against a best-available floor it is ~0."""
    finder = _finder(CONSOLIDATION, WIRE)
    roster = finder.rosters[1]
    short = tuple(roster[:-1])
    settled, cut, added, tied = finder.settle(1, short)
    assert not cut
    assert not tied  # nothing was cut, so nothing tied
    # Whatever the wire offers, the seat cannot beat a floor that already assumes you
    # signed the best free agent, so the credit lands at zero rather than at a constant.
    assert finder.value_of(settled) - finder.value_of(short) < 1e-6
    # Not `len(added) == len(settled) - len(short)`, which `settle` satisfies by
    # construction and so cannot fail. On this wire the honest answer is that nothing
    # beats the floor, so nobody is signed and the seat stays empty.
    assert added == ()
    assert settled == short


# --------------------------------------------------------------------------------------
# The Pareto gate
# --------------------------------------------------------------------------------------


def test_pareto_gate_rejects_a_trade_that_helps_only_one_side():
    """League-wide total up, one side down: still rejected."""
    finder = _finder(CONSOLIDATION, WIRE)
    elite = _by_name(finder, "WR elite")
    junk = _by_name(finder, "RB4")
    ev = finder.evaluate(TradeProposal(42, (TradeLeg(2, 1, (elite,)), TradeLeg(1, 2, (junk,)))))
    assert ev.impact_for(1).delta_points > 0
    assert ev.impact_for(2).delta_points < 0
    assert not ev.pareto
    assert sum(i.delta_points for i in ev.impacts) > 0  # the league gained; irrelevant

    found = finder.search(for_team=1, max_teams=2)
    assert all(e.pareto for e in found)
    assert all(e.min_gain > 0 for e in found)


def test_search_never_returns_a_trade_that_hurts_an_uninvolved_third_party_view():
    """Every returned trade is Pareto for everyone it touches, at any cycle length."""
    finder = _finder(CONSOLIDATION, WIRE)
    found = finder.search(for_team=1, max_teams=3)
    assert found, "the fixture is constructed to contain real trades"
    for ev in found:
        assert ev.pareto
        assert set(ev.rosters) == set(ev.proposal.teams)
        for team in ev.proposal.teams:
            assert len(ev.rosters[team]) <= len(finder.rosters[team])


def test_min_gain_raises_the_gate_from_pareto_to_negotiable():
    """A 0.4-point gain is a real improvement and nobody accepts it.

    The strict gate is what the objective specifies and it is what ships. On the user's
    real leagues it also passes three-ways whose weakest side gains 0.4 playoff-weighted
    points across seventeen weeks, so the threshold has to be reachable.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    loose = finder.search(for_team=1, max_teams=3)
    tight = finder.search(for_team=1, max_teams=3, min_gain=8.0)
    assert loose
    assert len(tight) < len(loose)
    assert all(ev.min_gain > 8.0 for ev in tight)
    assert min((ev.min_gain for ev in loose), default=99.0) <= 8.0


# --------------------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------------------


def test_simple_cycles_enumerates_each_direction_once():
    adjacency = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    cycles = simple_cycles(adjacency, 3)
    assert (1, 2) in cycles and (2, 1) not in cycles  # one representative per rotation
    assert (1, 2, 3) in cycles and (1, 3, 2) in cycles  # both directions are real trades
    assert all(len(c) <= 3 for c in cycles)
    assert len(set(cycles)) == len(cycles)


#: Three teams, three needs, arranged so no two of them can help each other alone:
#: 1 needs a tight end and has spare backs, 2 needs a back and has spare receivers,
#: 3 needs a receiver and has spare tight ends. Team 4 owns a player everyone wants
#: more than anything else, which is the concentration that breaks single-edge TTC.
PLANTED_THREE_WAY = {
    1: [
        (QB, 17.0, "P1 QB"),
        (RB, 16.0, "P1 RB1"),
        (RB, 15.0, "P1 RB2"),
        (RB, 14.0, "P1 RB3"),
        (RB, 13.0, "P1 RB spare"),
        (WR, 16.0, "P1 WR1"),
        (WR, 15.0, "P1 WR2"),
        (WR, 14.0, "P1 WR3"),
        (TE, 1.0, "P1 TE hole"),
        (QB, 8.0, "P1 QB2"),
    ],
    2: [
        (QB, 17.0, "P2 QB"),
        (RB, 1.0, "P2 RB hole"),
        (RB, 1.0, "P2 RB hole 2"),
        (WR, 18.0, "P2 WR1"),
        (WR, 17.0, "P2 WR2"),
        (WR, 16.0, "P2 WR3"),
        (WR, 15.0, "P2 WR spare"),
        (TE, 14.0, "P2 TE1"),
        (TE, 13.0, "P2 TE2"),
        (QB, 8.0, "P2 QB2"),
    ],
    3: [
        (QB, 17.0, "P3 QB"),
        (RB, 16.0, "P3 RB1"),
        (RB, 15.0, "P3 RB2"),
        (WR, 1.0, "P3 WR hole"),
        (WR, 1.0, "P3 WR hole 2"),
        (TE, 15.0, "P3 TE1"),
        (TE, 14.0, "P3 TE2"),
        (TE, 13.0, "P3 TE spare"),
        (QB, 8.0, "P3 QB2"),
        (RB, 8.0, "P3 RB3"),
    ],
    # The concentration that breaks single-edge TTC: team 4 owns the single best answer
    # to all three other teams' holes, so everyone's *first* choice is the same owner.
    # It is not deep, though, so those players cost it a lot -- which is exactly why a
    # graph built on surplus cannot see them and one built on preference points every
    # team at the same place. Both halves of the blend are needed here.
    4: [
        (QB, 17.0, "P4 QB"),
        (RB, 25.0, "P4 the best back"),
        (RB, 8.0, "P4 RB2"),
        (RB, 7.0, "P4 RB3"),
        (WR, 25.0, "P4 the best receiver"),
        (WR, 8.0, "P4 WR2"),
        (WR, 7.0, "P4 WR3"),
        (TE, 25.0, "P4 the best tight end"),
        (TE, 8.0, "P4 TE2"),
        (QB, 8.0, "P4 QB2"),
    ],
}


def test_cycle_enumeration_finds_a_planted_three_way():
    finder = _finder(PLANTED_THREE_WAY, WIRE)
    edges = finder.preference_edges(top_k=6)
    adjacency = {t: {e.owner_id for e in es} for t, es in edges.items()}
    cycles = simple_cycles(adjacency, 3)
    # 1 wants a TE (team 3 has them), 3 wants a WR (team 2), 2 wants an RB (team 1).
    assert (1, 3, 2) in cycles

    found = finder.search(for_team=1, max_teams=3, top_k=6)
    three_way = [e for e in found if e.n_teams == 3]
    assert three_way, "the planted three-way must survive the Pareto gate"
    best = three_way[0]
    assert set(best.proposal.teams) >= {1}
    assert best.pareto


def test_multi_edge_avoids_the_degeneracy_single_edge_exhibits():
    """Single-edge TTC collapses on a concentrated market; multi-edge does not.

    Team 4 owns the three best players in the league, so every other team's *single*
    most-wanted asset sits on team 4 and the single-edge graph is a star with no
    three-cycle in it. The planted 1->3->2->1 cycle is still there; only a multi-edge
    graph can see it.
    """
    finder = _finder(PLANTED_THREE_WAY, WIRE)
    edges = finder.preference_edges(top_k=6)

    targets = single_edge_targets(edges)
    assert targets[1] == targets[2] == targets[3] == 4, targets
    single = {t: {o} for t, o in targets.items()}
    assert not [c for c in simple_cycles(single, 3) if len(c) == 3]

    multi = {t: {e.owner_id for e in es} for t, es in edges.items()}
    assert (1, 3, 2) in simple_cycles(multi, 3)


# --------------------------------------------------------------------------------------
# Moves, recommendations and selection
# --------------------------------------------------------------------------------------


def test_proposal_and_move_round_trip():
    proposal = TradeProposal(42, (TradeLeg(1, 2, (7, 8)), TradeLeg(2, 1, (9,))))
    move = proposal.to_move()
    assert move.kind is MoveKind.TRADE
    assert move.teams == frozenset({1, 2})
    assert TradeProposal.from_move(move) == proposal


def test_a_player_cannot_move_twice_in_one_trade():
    with pytest.raises(TradeError):
        TradeProposal(42, (TradeLeg(1, 2, (7,)), TradeLeg(2, 3, (7,))))


def test_evaluate_refuses_a_player_the_team_does_not_own():
    finder = _finder(CONSOLIDATION, WIRE)
    elite = _by_name(finder, "WR elite")  # team 2's
    with pytest.raises(TradeError):
        finder.evaluate(TradeProposal(42, (TradeLeg(1, 2, (elite,)), TradeLeg(2, 1, (elite + 1,)))))


def test_select_non_overlapping_shares_no_team_and_prefers_longer_cycles():
    finder = _finder(PLANTED_THREE_WAY, WIRE)
    found = finder.search(for_team=None, max_teams=3, top_k=6, limit=60)
    picked = select_non_overlapping(found)
    seen: set[int] = set()
    for ev in picked:
        assert not (set(ev.proposal.teams) & seen)
        seen |= set(ev.proposal.teams)
    assert [e.n_teams for e in picked] == sorted((e.n_teams for e in picked), reverse=True)


def test_recommendations_are_framed_around_what_the_counterparty_receives():
    finder = _finder(CONSOLIDATION, WIRE)
    found = finder.search(for_team=1, max_teams=2)
    assert found
    recs = finder.recommend(found[:3], for_team=1)
    assert all(isinstance(r, Recommendation) for r in recs)
    head = recs[0].rationale
    # The counterparty's haul is named before anything the user gives up.
    assert head.index("gets") < head.index("You get")
    assert recs[0].move.kind is MoveKind.TRADE
    assert "trade" in recs[0].tags


# --------------------------------------------------------------------------------------
# The simulation half
# --------------------------------------------------------------------------------------


def test_confirm_populates_delta_title_and_the_stderr_is_the_paired_one():
    """Paired CRN, re-derived here rather than trusted.

    The reported `stderr` has to be the standard error of the *paired* difference. An
    arm's own standard error would be several times larger and would make every effect
    look insignificant; the difference of two independent arms would be larger still.
    So this recomputes both the delta and its error straight from the champion indicator
    matrices and demands an exact match.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    rb = _by_name(finder, "RB traded")
    te = _by_name(finder, "TE sub-wire")
    elite = _by_name(finder, "WR elite")
    ev = finder.evaluate(TradeProposal(42, (TradeLeg(1, 2, (rb, te)), TradeLeg(2, 1, (elite,)))))
    (confirmed,) = finder.confirm_titles([ev])
    assert confirmed.confirmed

    base_scores, base = finder._ensure_base()
    state = finder.state
    for team, roster in confirmed.rosters.items():
        state = state.with_franchise(finder.state.franchise(team).with_players(roster))
    scores = base_scores.copy()
    for team, roster in confirmed.rosters.items():
        col = finder.state.team_index[team]
        scores[:, :, col] = (
            finder.franchise_scores(team, roster) * finder._eff_factors[:, col, None]
        )
    alt = S.simulate_from_scores(state, scores, all_play=False)

    for impact in confirmed.impacts:
        col = finder.state.team_index[impact.team_id]
        paired = alt.champions.astype(float)[:, col] - base.champions.astype(float)[:, col]
        assert impact.delta_title == pytest.approx(paired.mean(), abs=1e-12)
        assert impact.delta_title_stderr == pytest.approx(
            paired.std(ddof=1) / math.sqrt(paired.size), abs=1e-12
        )
        # An unpaired arm's error, for contrast: if the module ever reported that, this
        # test would have to be rewritten, which is the point of pinning it.
        arm = base.champions.astype(float)[:, col]
        assert impact.delta_title_stderr < arm.std(ddof=1) / math.sqrt(arm.size) * 2

    recs = finder.recommend([confirmed], for_team=1)
    assert recs[0].stderr == pytest.approx(confirmed.impact_for(1).delta_title_stderr)
    total = sum(finder.baseline_title(t) for t in finder.rosters)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_a_null_trade_is_exactly_zero_all_the_way_through_the_confirm():
    """CRN's whole point, tested where it can actually break.

    Re-simulating the same score array is a test of `simulate_from_scores`, not of this
    module's pairing. What has to be exact is `confirm_titles` on an evaluation whose
    post-trade rosters are the rosters it started with: every franchise is rebuilt, every
    weekly total recomputed, the whole league re-standing-ed, and the paired difference
    must still come back at a hard zero with a zero standard error. Any leak -- a
    re-drawn tensor, a re-drawn efficiency factor, a franchise rebuilt in a different
    order -- shows up here as a non-zero and nowhere else.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    rb = _by_name(finder, "RB traded")
    elite = _by_name(finder, "WR elite")
    real = finder.evaluate(TradeProposal(42, (TradeLeg(1, 2, (rb,)), TradeLeg(2, 1, (elite,)))))
    null = TradeEvaluation(
        proposal=real.proposal,
        impacts=real.impacts,
        rosters={t: finder.rosters[t] for t in real.proposal.teams},
        names=real.names,
    )
    (confirmed,) = finder.confirm_titles([null])
    for impact in confirmed.impacts:
        assert impact.delta_title == 0.0
        assert impact.delta_title_stderr == 0.0

    # And the real trade is not zero, so the assertion above is testing the null and not
    # a confirm step that returns zero for everything.
    (moved,) = finder.confirm_titles([real])
    assert any(i.delta_title != 0.0 for i in moved.impacts)


def test_measured_sd_diff_is_in_the_right_neighbourhood():
    """The corpus constant is 34.4 on nine starters; seven here should come out lower."""
    finder = _finder(CONSOLIDATION, WIRE)
    assert 10.0 < finder.sd_diff < 60.0
    assert 0.0 < finder.leverage_for(1) <= 1.0


def test_leverage_falls_when_the_matchups_are_already_decided():
    """A team facing only mismatches buys less win probability per point."""
    lopsided = {
        1: [(QB, 40.0, "big QB"), *CONSOLIDATION[1][1:]],
        2: [
            (QB, 1.0, "tiny QB"),
            *[(p, 1.0, f"tiny {i}") for i, (p, _, _) in enumerate(CONSOLIDATION[2][1:])],
        ],
        3: [
            (QB, 1.0, "t3 QB"),
            *[(p, 1.0, f"t3 {i}") for i, (p, _, _) in enumerate(CONSOLIDATION[3][1:])],
        ],
        4: CONSOLIDATION[4],
    }
    even = _finder(CONSOLIDATION, WIRE)
    skewed = _finder(lopsided, WIRE)
    assert skewed.leverage_for(1) < even.leverage_for(1)


def test_move_evaluator_protocol_is_satisfied():
    finder = _finder(CONSOLIDATION, WIRE)
    rb = _by_name(finder, "RB traded")
    elite = _by_name(finder, "WR elite")
    move = Move(
        kind=MoveKind.TRADE,
        league_id=42,
        players=TradeProposal(42, (TradeLeg(1, 2, (rb,)), TradeLeg(2, 1, (elite,))))
        .to_move()
        .players,
    )
    screened = finder.screen([move])
    assert len(screened) == 1 and screened[0].stderr == 0.0
    confirmed = finder.confirm([move])
    assert len(confirmed) == 1
    assert "confirmed" in confirmed[0].tags
    assert math.isfinite(finder.baseline_title(1))


def test_an_empty_roster_scores_every_slot_at_the_wire_once_per_seat():
    """The degenerate roster, which a plain column sum gets wrong at the RB pair."""
    finder = _finder(CONSOLIDATION, WIRE)
    weekly = finder.weekly_points(())
    # 1 QB at 10, 2 RB at 5, 2 WR at 6, 1 TE at 7, 1 FLEX lifted to 7 by monotone_floor.
    assert weekly[0] == pytest.approx(10 + 2 * 5 + 2 * 6 + 7 + 7)
    assert finder.value_of(()) == pytest.approx(weekly[0] * len(WEEKS))


def test_a_duplicated_player_cannot_fill_two_slots():
    finder = _finder(CONSOLIDATION, WIRE)
    roster = finder.rosters[1]
    assert finder.value_of([*roster, roster[0]]) == pytest.approx(finder.value_of(roster))


# --------------------------------------------------------------------------------------
# The pruning bound
# --------------------------------------------------------------------------------------


def _exact_swap(finder: TradeFinder, team: int, incoming, outgoing) -> float:
    """What `evaluate` would actually price this swap at: one re-solve, no shortcuts."""
    roster = finder.rosters[team]
    after = [p for p in roster if p not in set(outgoing)] + list(incoming)
    return finder.value_of(after) - finder.value_of(roster)


def test_the_pruning_bound_is_optimistic_and_therefore_never_discards_a_real_trade():
    """`_cycle_proposals` cuts a branch when `_paper_gain <= 0`, so it had better be a
    bound in the direction that only over-admits.

    The obvious formulation is not. Taking each side's marginal against the *untouched*
    roster and subtracting gives

        naive = [v(R+in) - v(R)] - [v(R) - v(R-out)]

    while the truth is `[v(R-out+in) - v(R-out)] - [v(R) - v(R-out)]`, and submodularity
    makes the incoming marginal against the smaller roster the larger of the two. The
    naive form is a *lower* bound, and pruning on a lower bound throws real trades away:
    on this fixture it understates the exact re-solved gain in 160 of 240 one-for-one
    candidates, by up to 30 playoff-weighted points. This test is written so that
    reintroducing the naive form fails it.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    checked = 0
    for team in finder.rosters:
        incoming = [e.player_id for e in finder.asset_ranking(team)[:5]]
        for out_p in finder.rosters[team][:5]:
            for size in (1, 2):
                for combo in combinations(incoming, size):
                    exact = _exact_swap(finder, team, combo, [out_p])
                    assert finder._paper_gain(team, list(combo), [out_p]) >= exact - 1e-9
                    checked += 1
    assert checked > 100, "the sweep has to be wide enough to catch a one-sided bound"

    # A one-for-one has no package interaction left to be optimistic about, so the bound
    # collapses onto the exact re-solved answer.
    team = 2
    out_p = finder.rosters[team][0]
    for edge in finder.asset_ranking(team)[:5]:
        assert finder._paper_gain(team, [edge.player_id], [out_p]) == pytest.approx(
            _exact_swap(finder, team, [edge.player_id], [out_p])
        )


def test_the_naive_pruning_form_really_does_violate_the_bound_here():
    """The counter-example the test above is guarding, made explicit.

    Without this, `test_the_pruning_bound_is_optimistic...` could be passing because the
    fixture is too easy rather than because the code is right.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    violations = 0
    for team in finder.rosters:
        for edge in finder.asset_ranking(team)[:5]:
            for out_p in finder.rosters[team][:5]:
                naive = finder.asset_value(
                    team, edge.player_id, incoming=True
                ) - finder.asset_value(team, out_p, incoming=False)
                if naive < _exact_swap(finder, team, [edge.player_id], [out_p]) - 1e-9:
                    violations += 1
    assert violations > 20


# --------------------------------------------------------------------------------------
# Significance under selection, and never being confident about harm
# --------------------------------------------------------------------------------------


def test_selection_threshold_is_two_sigma_for_one_candidate_and_stricter_for_many():
    assert selection_threshold(1) == pytest.approx(1.959963, abs=1e-5)
    assert selection_threshold(0) == selection_threshold(1)  # degenerate, not a crash
    assert selection_threshold(8) > selection_threshold(1)
    assert selection_threshold(40) > selection_threshold(8)
    # The measured case: +1.12pp +/- 0.49 was reported as significant off eight
    # candidates and turned out to be +0.53pp +/- 0.13 at ten times the simulations.
    assert 1.12 > 2.0 * 0.49  # what the two-sigma test said
    assert selection_threshold(8) * 0.49 > 1.12  # what the selected test says


def _forced(finder: TradeFinder, delta_title: float, stderr: float) -> TradeEvaluation:
    """A real evaluation with one side's confirmed title delta pinned, for labelling."""
    rb = _by_name(finder, "RB traded")
    elite = _by_name(finder, "WR elite")
    ev = finder.evaluate(TradeProposal(42, (TradeLeg(1, 2, (rb,)), TradeLeg(2, 1, (elite,)))))
    impacts = tuple(
        dataclasses.replace(i, delta_title=delta_title, delta_title_stderr=stderr)
        if i.team_id == 1
        else dataclasses.replace(i, delta_title=0.01, delta_title_stderr=stderr)
        for i in ev.impacts
    )
    return dataclasses.replace(ev, impacts=impacts, confirmed=True)


def test_a_trade_the_simulation_says_is_harmful_is_never_labelled_confident():
    """The live failure this guards: `delta_title=-0.80pp +/- 0.31, confidence="high"`.

    Precision about a loss is not confidence in a recommendation. A confirmed trade whose
    measured title delta is at or below zero has to come back "low", carry the `harmful`
    tag, and say so in words -- otherwise a caller rendering `confidence` next to the
    headline is being told to make the trade.
    """
    finder = _finder(CONSOLIDATION, WIRE)
    harmful = _forced(finder, delta_title=-0.008, stderr=0.0003)  # -0.8pp, ~27 sigma
    (rec,) = finder.recommend([harmful], for_team=1)
    assert rec.significant, "the effect is precisely measured; that is the trap"
    assert rec.confidence == "low"
    assert "harmful" in rec.tags
    assert "do not propose it" in rec.rationale


def test_a_precise_win_is_still_only_confident_after_the_selection_correction():
    """2.5 sigma off one candidate is an edge; off forty it is the winner's curse."""
    finder = _finder(CONSOLIDATION, WIRE)
    one = _forced(finder, delta_title=0.01, stderr=0.004)  # +1.0pp +/- 0.4 == 2.5 sigma
    (alone,) = finder.recommend([one], for_team=1)
    assert alone.confidence == "high"

    many = [one, *[_forced(finder, delta_title=0.0002 * i, stderr=0.004) for i in range(39)]]
    ranked = finder.recommend(many, for_team=1)
    assert ranked[0].delta_title == pytest.approx(0.01)
    assert ranked[0].confidence == "low"
    assert "out of 40 candidates" in ranked[0].rationale


class TestTheWireIsWhatNobodyRosters:
    """`wire_pool` read `set(state.pool.player_ids)` and called it "rostered".

    Accidentally right under `pipeline.build`, which pools only rostered players, so the
    two sets are identical on all three live leagues and no test could tell them apart.
    Wrong the moment anything widens the pool -- and `waivers.augment` does exactly that,
    adding the sixty best free agents so they have columns to be simulated in. Every one
    of those sixty then counted as rostered and the floor was read off the dregs behind
    them: measured on the live leagues, 0.74 to 8.58 points a week too low at every
    position, and at kicker 0.289 against a true 8.868, because all thirty plausible
    free-agent kickers had been pulled into the pool.

    No production path hands `wire_pool` a widened state today, so the fix is latent.
    This is what keeps it that way.
    """

    def test_a_widened_pool_does_not_turn_free_agents_into_rostered_ones(self):
        finder = _finder(CONSOLIDATION, WIRE)
        state = finder.state
        narrow = wire_pool(state, finder.outlooks)
        assert narrow, "the fixture has no wire at all; nothing below can fail"

        # Widen the pool the way `waivers.augment` does: give the best free agents
        # columns, without putting them on anybody's roster.
        on_a_roster = {p for f in state.franchises for p in f.player_ids}
        extra = [o for o in finder.outlooks if o.player_id not in on_a_roster]
        assert extra, "the fixture has no free agents to widen with"
        rows = [
            (pid, pos, team, state.pool.name(pid))
            for pid, pos, team in zip(
                state.pool.player_ids,
                state.pool.position_ids,
                state.pool.pro_team_ids,
                strict=True,
            )
        ]
        rows += [(o.player_id, o.position_id, o.pro_team_id, o.name) for o in extra]
        wide = dataclasses.replace(state, pool=S.PlayerPool.of(rows))
        assert wide.pool.size > state.pool.size

        widened = wire_pool(wide, finder.outlooks)
        # The wire is a property of who is ROSTERED, and widening the pool rosters
        # nobody. Same bodies, same order, whatever the pool holds.
        assert {pos: [pid for pid, _ in bodies] for pos, bodies in widened.items()} == {
            pos: [pid for pid, _ in bodies] for pos, bodies in narrow.items()
        }

    def test_it_reads_franchises_and_not_the_pool(self):
        """The invariant, stated directly: a player on no roster is on the wire."""
        finder = _finder(CONSOLIDATION, WIRE)
        state = finder.state
        on_a_roster = {p for f in state.franchises for p in f.player_ids}
        found = {pid for bodies in wire_pool(state, finder.outlooks).values()
                 for pid, _ in bodies}
        assert found
        assert not (found & on_a_roster)


class TestTheForcedCutIsNotDecidedByEspnsOrdering:
    """`settle` kept the FIRST maximum of a greedy leave-one-out.

    `value_of` is a whole-roster starting-lineup objective, so a deep-bench player who
    never cracks a lineup contributes exactly zero and removing any of them leaves the
    objective bit-identical. The scan ran over `dict.fromkeys(player_ids)` -- ESPN's own
    roster ordering, straight through `self.rosters` -- so the answer to "who do you cut"
    was whoever ESPN happened to list first.

    Measured on the three live leagues: **65 of 69 forced cuts (94%) had at least two
    bit-exact ties**, median tie group three to five, maximum six. The spread between the
    best and worst leave-one-out is 97-140 points, so the choice matters enormously in
    general; it is only among the top candidates that it is a dead heat.
    """

    def _over_the_limit(self, finder, team=1):
        """That team's roster plus enough wire filler to force at least one cut."""
        roster = list(finder.rosters[team])
        limit = finder.capacity.get(team, len(roster))
        wire = [pid for pos in finder._wire for pid, _ in finder._wire[pos]]
        spare = [p for p in wire if p not in roster]
        over = roster + spare
        if len(over) <= limit:
            pytest.skip("this fixture cannot be pushed over its roster limit")
        return over

    def test_the_cut_does_not_move_when_the_input_order_does(self):
        """The defect itself: the answer used to be a function of ESPN's ordering."""
        finder = _finder(CONSOLIDATION, WIRE)
        over = self._over_the_limit(finder)
        forward = finder.settle(1, over)
        backward = finder.settle(1, list(reversed(over)))
        assert forward[1] == backward[1], "the cut moved when the input order did"
        assert sorted(forward[0]) == sorted(backward[0])

    def test_the_cut_is_the_least_valuable_of_the_equals(self):
        finder = _finder(CONSOLIDATION, WIRE)
        _settled, cut, _added, tied = finder.settle(1, self._over_the_limit(finder))
        assert cut
        for dropped, equals in zip(cut, tied, strict=True):
            for other in equals:
                assert finder._cut_priority(dropped) <= finder._cut_priority(other)

    def test_the_equals_are_reported_rather_than_hidden(self):
        '''"Cut this one" and "cut any of these five" are different pieces of advice.'''
        finder = _finder(CONSOLIDATION, WIRE)
        _settled, cut, _added, tied = finder.settle(1, self._over_the_limit(finder))
        assert len(tied) == len(cut)
        assert all(dropped not in equals for dropped, equals in zip(cut, tied, strict=True))
        assert any(equals for equals in tied), "this fixture was supposed to produce a tie"

    def test_a_unique_maximum_is_chosen_exactly_as_before(self):
        """The negative control: where the objective can tell, nothing changed."""
        finder = _finder(CONSOLIDATION, WIRE)
        over = self._over_the_limit(finder)
        scored = [(finder.value_of([p for p in over if p != q]), q) for q in over]
        best = max(v for v, _ in scored)
        winners = [q for v, q in scored if v == best]
        _settled, cut, _added, tied = finder.settle(1, over)
        if len(winners) == 1:
            assert cut[0] == winners[0]
            assert tied[0] == ()
        else:
            # The interesting case, and the live one. Every winner must be reported.
            assert {cut[0], *tied[0]} == set(winners)

    def test_the_tie_is_named_in_the_rationale(self):
        finder = _finder(CONSOLIDATION, WIRE)
        rb = _by_name(finder, "RB traded")
        elite = _by_name(finder, "WR elite")
        ev = finder.evaluate(
            TradeProposal(42, (TradeLeg(1, 2, (rb,)), TradeLeg(2, 1, (elite,))))
        )
        (rec,) = finder.recommend([dataclasses.replace(ev, confirmed=True)], for_team=1)
        mine = ev.impact_for(1)
        if any(mine.cut_alternatives):
            assert "That cut is a tie" in rec.rationale
        else:
            assert "That cut is a tie" not in rec.rationale


def test_a_counterparty_the_simulation_says_loses_is_named_not_buried():
    """The pitch quotes every other side its POINTS gain, which is the gate. When the
    simulation says that side's title odds fall anyway, the rationale has to say so --
    the user is the one who has to send the offer."""
    finder = _finder(CONSOLIDATION, WIRE)
    rb = _by_name(finder, "RB traded")
    elite = _by_name(finder, "WR elite")
    ev = finder.evaluate(TradeProposal(42, (TradeLeg(1, 2, (rb,)), TradeLeg(2, 1, (elite,)))))
    impacts = tuple(
        dataclasses.replace(
            i, delta_title=(0.01 if i.team_id == 1 else -0.005), delta_title_stderr=0.001
        )
        for i in ev.impacts
    )
    (rec,) = finder.recommend(
        [dataclasses.replace(ev, impacts=impacts, confirmed=True)], for_team=1
    )
    assert "counterparty-loses" in rec.tags
    assert "Simulated title odds fall for" in rec.rationale
    assert finder.team_names[2] in rec.rationale


def _tagged(delta_mine, delta_other, stderr, *, n_candidates=40):
    """One confirmed evaluation with hand-set impacts, run through `recommend`."""
    finder = _finder(CONSOLIDATION, WIRE)
    rb = _by_name(finder, "RB traded")
    elite = _by_name(finder, "WR elite")
    ev = finder.evaluate(TradeProposal(42, (TradeLeg(1, 2, (rb,)), TradeLeg(2, 1, (elite,)))))
    impacts = tuple(
        dataclasses.replace(
            i,
            delta_title=(delta_mine if i.team_id == 1 else delta_other),
            delta_title_stderr=stderr,
        )
        for i in ev.impacts
    )
    confirmed = dataclasses.replace(ev, impacts=impacts, confirmed=True)
    # `recommend` derives `z` from how many confirmed candidates competed, so padding the
    # list is how the selection correction is exercised rather than asserted.
    padding = [confirmed] * (n_candidates - 1)
    return finder, finder.recommend([confirmed, *padding], for_team=1)[0]


class TestANoisySignIsNotAnAssertion:
    """`counterparty-loses` and `harmful` were bare sign tests on a noisy paired delta.

    `TradeEvaluation.title_pareto`'s own docstring already said the quantity was noise --
    "the per-side deltas are individually noisy at affordable simulation counts, and a
    gate on a noisy quantity is a gate on noise" -- and then two tags gated on exactly
    that, and `report.py` rendered one of them as a flat assertion that a counterparty's
    odds FALL.

    Measured across three seeds on the live leagues, the same forty trades disagreed with
    themselves about `counterparty-loses` on 75% / 52% / 35% of them and about `harmful`
    on 42% / 78% / 18%. Of the 50 counterparty impacts that read negative, **not one
    cleared the selection-adjusted threshold** and only four cleared a naive two sigma --
    while the tag was asserted on 45 trades.
    """

    def test_a_five_sigma_loss_is_still_asserted(self):
        _finderer, rec = _tagged(0.01, -0.005, 0.0002)
        assert "counterparty-loses" in rec.tags
        assert "counterparty-loses-unclear" not in rec.tags
        assert "Simulated title odds fall for" in rec.rationale

    def test_a_coin_flip_says_it_cannot_tell(self):
        _finderer, rec = _tagged(0.01, -0.0005, 0.002)
        assert "counterparty-loses-unclear" in rec.tags
        assert "counterparty-loses" not in rec.tags
        assert "cannot tell whether that side gains or loses" in rec.rationale
        assert "Simulated title odds fall for" not in rec.rationale

    def test_a_counterparty_that_gains_is_not_tagged_either_way(self):
        _finderer, rec = _tagged(0.01, 0.004, 0.002)
        assert not any(t.startswith("counterparty-loses") for t in rec.tags)

    def test_my_own_side_gets_the_same_three_states(self):
        _f1, resolved = _tagged(-0.01, 0.004, 0.0002)
        assert "harmful" in resolved.tags and "harmful-unclear" not in resolved.tags
        _f2, unclear = _tagged(-0.0005, 0.004, 0.002)
        assert "harmful-unclear" in unclear.tags and "harmful" not in unclear.tags
        # The protection does not depend on the tag: an unconfirmed gain is still low.
        assert unclear.confidence == "low"
        assert resolved.confidence == "low"

    def test_the_threshold_is_selection_adjusted_not_two_sigma(self):
        """The trade being labelled won a search, so its side effects did too."""
        # 2.5 sigma: significant for one pre-specified candidate, not for the winner of 40.
        _f, of_forty = _tagged(0.01, -0.0025, 0.001, n_candidates=40)
        assert "counterparty-loses-unclear" in of_forty.tags
        _f2, of_one = _tagged(0.01, -0.0025, 0.001, n_candidates=1)
        assert "counterparty-loses" in of_one.tags
        assert selection_threshold(40) > 2.0 > 0.0
        assert selection_threshold(1) < 2.5

    def test_a_zero_threshold_is_exactly_the_old_sign_test(self, monkeypatch):
        """The negative control, and it has to be exact.

        With `z = 0` every `-unclear` must collapse back into the bare sign test this
        replaced. Verified on the live leagues too: at `z = 0` no `-unclear` tag survives
        and the resolved counts land exactly on the `-unclear` counts at the real
        threshold (14/10/21 counterparty, 27/29/5 harmful).
        """
        import fantasy_quant.decide.trades as trades_mod

        monkeypatch.setattr(trades_mod, "selection_threshold", lambda *a, **k: 0.0)
        for mine, other in ((0.01, -0.0005), (-0.0005, 0.004), (0.01, -1e-9)):
            _f, rec = _tagged(mine, other, 0.002)
            assert not any(t.endswith("-unclear") for t in rec.tags), (mine, other, rec.tags)
        # ... and the bare signs it produces are the ones the old code produced.
        _f, rec = _tagged(0.01, -0.0005, 0.002)
        assert "counterparty-loses" in rec.tags
        _f, rec = _tagged(-0.0005, 0.004, 0.002)
        assert "harmful" in rec.tags

    def test_an_impact_with_no_monte_carlo_keeps_its_sign(self):
        """A zero standard error is an exact result, not an unmeasured one."""
        _f, rec = _tagged(0.01, -0.005, 0.0)
        assert "counterparty-loses" in rec.tags


# --------------------------------------------------------------------------------------
# The front door
# --------------------------------------------------------------------------------------


class _Sim:
    """The three attributes `TradeFinder.from_sim` reads. `find_trades` is duck-typed."""

    def __init__(self, state, draw, outlooks):
        self.state, self.draw, self.outlooks = state, draw, outlooks


def _sim(rosters=CONSOLIDATION, wire=WIRE):
    state, outlooks = _market(rosters, wire)
    return _Sim(state, _dummy_draw(state, outlooks), outlooks)


def test_find_trades_confirms_the_whole_screened_set_not_the_screens_top_eight():
    """The screen ranks on points and the confirm ranks on title probability, and on the
    user's real leagues those orderings correlate anywhere from -0.43 to +0.72. Cutting
    the confirm to the screen's top eight therefore threw away the best trade in the
    league; the screen is a recall filter, not a ranker."""
    sim = _sim()
    finder = TradeFinder(sim.state, sim.draw, sim.outlooks)
    screened = finder.search(for_team=1, max_teams=3)
    assert len(screened) > 8, "the fixture has to have more candidates than the old cap"
    everything = find_trades(sim, for_team=1, finder=finder, include_harmful=True)
    assert len(everything) == len(screened)
    assert all("confirmed" in r.tags for r in everything)
    capped = find_trades(sim, for_team=1, finder=finder, n_confirm=8, include_harmful=True)
    assert len(capped) == 8


def test_find_trades_does_not_return_a_trade_the_simulation_says_would_hurt():
    sim = _sim()
    finder = TradeFinder(sim.state, sim.draw, sim.outlooks)
    everything = find_trades(sim, for_team=1, finder=finder, include_harmful=True)
    kept = find_trades(sim, for_team=1, finder=finder)
    assert any(r.delta_title <= 0 for r in everything), "the fixture must contain one"
    assert all(r.delta_title > 0 for r in kept)
    assert len(kept) < len(everything)
    assert all("harmful" not in r.tags for r in kept)


def test_find_trades_forwards_min_gain_so_the_documented_remedy_is_reachable():
    """The strict gate passes legs a counterparty gains half a point from. `min_gain` is
    how a caller asks for a trade someone would actually sign, and it was not plumbed
    through the front door at all."""
    sim = _sim()
    finder = TradeFinder(sim.state, sim.draw, sim.outlooks)
    loose = find_trades(sim, for_team=1, finder=finder, include_harmful=True)
    tight = find_trades(sim, for_team=1, finder=finder, min_gain=8.0, include_harmful=True)
    assert len(tight) < len(loose)


# --------------------------------------------------------------------------------------
# sd_diff
# --------------------------------------------------------------------------------------


def test_sd_diff_matches_the_spread_it_claims_to_measure():
    """`sqrt(2 * mean per-week variance)` is only the SD of a matchup margin if the two
    teams' weekly scores really are near-independent with similar variance. Checked
    against the empirical spread of an actual pair rather than asserted."""
    finder = _finder(CONSOLIDATION, WIRE)
    scores, _ = finder._ensure_base()
    empirical = float((scores[:, :, 0] - scores[:, :, 1]).std())
    assert finder.sd_diff == pytest.approx(empirical, rel=0.25)
    assert 0.0 < finder.leverage_for(1) <= 1.0


def test_sd_diff_falls_back_to_the_corpus_constant_on_a_degenerate_tensor():
    """A one-simulation or zero-variance tensor would otherwise hand `core.leverage` a
    zero standard deviation and make every point look infinitely valuable."""
    finder = _finder(CONSOLIDATION, WIRE)
    scores, result = finder._ensure_base()
    finder._base_scores = np.zeros_like(scores)
    finder._sd_diff = None
    assert finder.sd_diff == MEASURED_SD_DIFF
