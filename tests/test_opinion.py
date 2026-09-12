"""The second opinion: a human ranking set, and what it is allowed to say.

Two shapes, and the tests keep them apart. `bench_upgrades` compares ranks to ranks
and reports an ordering. `tilt_outlooks` carries the ordering into the numbers. What
neither may ever do is turn a rank into a price -- see `decide/valuation.py`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fantasy_quant.data.etr import EtrRankings
from fantasy_quant.decide import opinion

QB, RB, WR, TE = 1, 2, 3, 4


def board(rows, *, kind="silva", scoring="half_ppr", comments=True):
    """rows: (espn_id, position_id, overall_rank, pos_rank, player, comment)."""
    frame = pl.DataFrame(
        {
            "espn_id": [r[0] for r in rows],
            "position_id": [r[1] for r in rows],
            "etr_rank": [r[2] for r in rows],
            "pos_rank": [r[3] for r in rows],
            "player": [r[4] for r in rows],
            **({"comment": [r[5] for r in rows]} if comments else {}),
        }
    )
    return EtrRankings(scoring=scoring, kind=kind, path=Path("b.csv"), frame=frame)


BOARD = board(
    [
        (10, RB, 1, 1, "Elite RB", "Bell cow."),
        (11, RB, 5, 2, "Good RB", "Timeshare."),
        (12, RB, 9, 3, "Meh RB", "Backup."),
        (20, WR, 3, 1, "Elite WR", "Target hog."),
        (21, WR, 7, 2, "Meh WR", "Fourth option."),
    ]
)


class TestBenchUpgrades:
    def test_a_free_agent_the_board_ranks_above_somebody_you_hold(self):
        rep = opinion.bench_upgrades(BOARD, roster=[12], free_agents=[11])
        assert len(rep.upgrades) == 1
        up = rep.upgrades[0]
        assert (up.add, up.drop, up.position_id) == (11, 12, RB)
        assert (up.add_rank, up.drop_rank, up.gap) == (2, 3, 1)

    def test_the_board_is_not_asked_to_compare_across_positions(self):
        """An overall rank encodes what a receiver is worth against a running back,
        and this league solves that itself from its own roster shape."""
        rep = opinion.bench_upgrades(BOARD, roster=[12], free_agents=[20])
        assert rep.upgrades == ()

    def test_a_worse_free_agent_is_not_an_upgrade(self):
        assert opinion.bench_upgrades(BOARD, roster=[11], free_agents=[12]).upgrades == ()

    def test_equal_ranks_are_not_an_upgrade(self):
        """The same player on both sides is the degenerate case, and `>` not `>=` is
        what keeps it out."""
        assert opinion.bench_upgrades(BOARD, roster=[11], free_agents=[11]).upgrades == ()

    def test_both_notes_travel_with_the_pair(self):
        """The pitch is half of a waiver claim and all of a trade, and this board is
        the only source in the repo that publishes one."""
        up = opinion.bench_upgrades(BOARD, roster=[12], free_agents=[11]).upgrades[0]
        assert up.add_comment == "Timeshare." and up.drop_comment == "Backup."

    def test_players_the_board_does_not_rank_are_not_guessed_at(self):
        rep = opinion.bench_upgrades(BOARD, roster=[12, 999], free_agents=[11, 998])
        assert len(rep.upgrades) == 1
        assert (rep.roster_covered, rep.roster_total, rep.blind_spot) == (1, 2, 1)
        assert (rep.wire_covered, rep.wire_total) == (1, 2)

    def test_coverage_makes_a_thin_answer_visibly_thin(self):
        """Measured at week 1 of 2026, only 4-7 of the Top 150 are unrostered in the
        user's leagues, so "no upgrades" is the common case and must not be confused
        with "the board had nothing to say"."""
        rep = opinion.bench_upgrades(BOARD, roster=[777], free_agents=[888])
        assert not rep
        assert rep.roster_covered == 0 and rep.roster_total == 1

    def test_the_order_is_determinate_and_not_the_rosters(self):
        """`settle`'s forced cut was decided by ESPN's roster order for a while. The
        same trap, so the same fix: widest gap, then the id pair."""
        rep = opinion.bench_upgrades(BOARD, roster=[12, 11], free_agents=[10])
        assert [(u.add, u.drop) for u in rep.upgrades] == [(10, 12), (10, 11)]
        flipped = opinion.bench_upgrades(BOARD, roster=[11, 12], free_agents=[10])
        assert [(u.add, u.drop) for u in flipped.upgrades] == [(10, 12), (10, 11)]

    def test_limit_keeps_the_widest_gaps(self):
        rep = opinion.bench_upgrades(BOARD, roster=[12, 11], free_agents=[10], limit=1)
        assert [(u.add, u.drop) for u in rep.upgrades] == [(10, 12)]


class TestCrossPosition:
    def test_it_compares_on_the_overall_ladder_and_says_so(self):
        rep = opinion.cross_position_upgrades(BOARD, roster=[12], free_agents=[20])
        assert len(rep.upgrades) == 1
        up = rep.upgrades[0]
        assert up.cross_position is True
        assert (up.add_rank, up.drop_rank) == (3, 9)

    def test_the_same_pair_is_absent_from_the_within_position_screen(self):
        """Kept as two functions rather than one with a flag, because the weaker
        ladder must not be reachable by accident."""
        assert opinion.bench_upgrades(BOARD, roster=[12], free_agents=[20]).upgrades == ()


class TestUpgradesFor:
    """The wire has to come from the outlooks, never from `state.pool`."""

    class _Pool:
        player_ids = (10, 11, 12)
        names = ("Elite RB", "Good RB", "Meh RB")

    class _Franchise:
        def __init__(self, team_id, player_ids):
            self.team_id = team_id
            self.player_ids = player_ids

    class _State:
        def __init__(self, franchises):
            self.franchises = franchises
            self.pool = TestUpgradesFor._Pool()

    class _Outlook:
        def __init__(self, pid, name):
            self.player_id = pid
            self.name = name

    class _Sim:
        def __init__(self, state, outlooks):
            self.state = state
            self.outlooks = outlooks

    def _sim(self):
        state = self._State((self._Franchise(1, (12,)), self._Franchise(2, (10,))))
        rows = [(10, "Elite RB"), (11, "Good RB"), (12, "Meh RB")]
        outlooks = [self._Outlook(p, n) for p, n in rows]
        return self._Sim(state, outlooks)

    def test_the_wire_is_read_off_the_outlooks_not_the_rostered_pool(self):
        """`pipeline.build` pools only rostered players -- measured on all three live
        leagues at week 1 of 2026, every one of `state.pool`'s 194-225 ids is owned.
        Reading the wire off the pool reports a clean zero rather than an error, which
        is exactly how `cc624f6` and `ef36808` each survived as long as they did."""
        rep = opinion.upgrades_for(BOARD, self._sim(), team_id=1)
        assert [(u.add, u.drop) for u in rep.upgrades] == [(11, 12)]

    def test_players_other_franchises_hold_are_not_on_the_wire(self):
        rep = opinion.upgrades_for(BOARD, self._sim(), team_id=1)
        assert all(u.add != 10 for u in rep.upgrades)


class TestBoardsWithoutCommentary:
    def test_a_board_that_publishes_no_notes_still_pairs(self):
        bare = board([(10, RB, 1, 1, "A", None), (12, RB, 9, 3, "C", None)], comments=False)
        up = opinion.bench_upgrades(bare, roster=[12], free_agents=[10]).upgrades[0]
        assert up.add_comment == "" and up.gap == 2


@pytest.mark.network
class TestAgainstTheLiveBoard:
    def test_the_top_150_is_almost_entirely_rostered_by_week_one(self):
        """The honest size of this surface. A 12-team league rosters ~195 players and
        the board is 150 skill players, so the wire overlap is single digits."""
        from fantasy_quant.data import etr

        b, _ = etr.best_available("half_ppr", kind="silva")
        if b is None:
            pytest.skip("no Silva board in data/manual/etr")
        assert b.n == 150
        assert len(b.positional()) == 150


def outlook(pid, pos, means, *, name="", sd=3.0, p_zero=0.1, shape=2.0, scale=4.0):
    from fantasy_quant.core import PlayerOutlook, WeeklyOutlook

    return PlayerOutlook(
        player_id=pid,
        name=name or f"P{pid}",
        position_id=pos,
        pro_team_id=7,
        weeks={
            w: WeeklyOutlook(
                player_id=pid, season=2026, week=w, position_id=pos, mean=m, sd=sd,
                p_zero=p_zero, shape=shape, scale=scale, pro_team_id=7, playing=m > 0,
            )
            for w, m in means.items()
        },
    )


class TestTiltIsAPermutation:
    """The property that keeps replacement level, the wire and the scarcity curves
    exactly where they were: the multiset of values at each position never changes."""

    OUT = [
        outlook(10, RB, {1: 10.0, 2: 10.0}),   # our RB1, board has him RB1
        outlook(11, RB, {1: 8.0, 2: 8.0}),     # our RB2, board has him RB3
        outlook(12, RB, {1: 2.0, 2: 2.0}),     # our RB3, board has him RB2
        outlook(20, WR, {1: 9.0, 2: 9.0}),
        outlook(99, TE, {1: 5.0, 2: 5.0}),     # not on the board at all
    ]

    def test_weight_zero_is_the_identity(self):
        """Every consumer's negative control is this call, so it has to be exact."""
        same = opinion.tilt_outlooks(self.OUT, BOARD, weight=0.0)
        assert same == list(self.OUT)

    def test_the_values_are_re_dealt_not_recomputed(self):
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(self.OUT, BOARD, weight=1.0)}
        # Board order at RB is 10, 11, 12 by pos_rank (1, 2, 3); ours by value is
        # 10 (20.0), 11 (16.0), 12 (4.0). Board has 11 at RB2 and 12 at RB3.
        assert tilted[10].mean_from(1) == pytest.approx(20.0)
        assert tilted[11].mean_from(1) == pytest.approx(16.0)
        assert tilted[12].mean_from(1) == pytest.approx(4.0)

    def test_a_disagreement_swaps_two_players_values_exactly(self):
        """Board ranks 12 above 11; ours has 11 worth 16.0 and 12 worth 4.0."""
        swapped = board(
            [(10, RB, 1, 1, "A", ""), (12, RB, 5, 2, "C", ""), (11, RB, 9, 3, "B", "")]
        )
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(self.OUT, swapped, weight=1.0)}
        assert tilted[12].mean_from(1) == pytest.approx(16.0)
        assert tilted[11].mean_from(1) == pytest.approx(4.0)

    def test_the_multiset_of_values_per_position_is_preserved(self):
        swapped = board(
            [(10, RB, 1, 1, "A", ""), (12, RB, 5, 2, "C", ""), (11, RB, 9, 3, "B", "")]
        )
        for weight in (0.25, 0.5, 1.0):
            tilted = opinion.tilt_outlooks(self.OUT, swapped, weight=weight)
            before = sorted(o.mean_from(1) for o in self.OUT if o.position_id == RB)
            after = sorted(o.mean_from(1) for o in tilted if o.position_id == RB)
            if weight == 1.0:
                assert after == pytest.approx(before)
            # At any weight the total is conserved, which is what replacement level
            # and the scarcity fit actually read.
            assert sum(after) == pytest.approx(sum(before))

    def test_players_the_board_does_not_rank_are_untouched(self):
        """The wire lives in the 448 of 598 projected players the board never sees."""
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(self.OUT, BOARD, weight=1.0)}
        assert tilted[99] is self.OUT[4]

    def test_weight_interpolates_toward_the_board_not_past_it(self):
        swapped = board(
            [(10, RB, 1, 1, "A", ""), (12, RB, 5, 2, "C", ""), (11, RB, 9, 3, "B", "")]
        )
        half = {o.player_id: o for o in opinion.tilt_outlooks(self.OUT, swapped, weight=0.5)}
        assert half[12].mean_from(1) == pytest.approx(10.0)  # halfway from 4.0 to 16.0
        assert half[11].mean_from(1) == pytest.approx(10.0)  # halfway from 16.0 to 4.0


class TestTiltKeepsTheDistributionCoherent:
    def test_byes_and_absences_survive(self):
        """`playing` and `p_zero` are how this simulator models availability. The
        board's ordering must not reach them -- scaling a bye by anything is still a
        bye, and it has to stay one."""
        out = [
            outlook(10, RB, {1: 10.0, 2: 0.0}),
            outlook(11, RB, {1: 8.0, 2: 8.0}),
            outlook(12, RB, {1: 2.0, 2: 2.0}),
        ]
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(out, BOARD, weight=1.0)}
        bye = tilted[10].weeks[2]
        assert bye.mean == 0.0 and bye.playing is False

    def test_the_gamma_stays_consistent_with_its_own_moments(self):
        """`core.WeeklyOutlook` is explicit that mean and sd are the moments of the
        FULL distribution and must not be reconstructed from the gamma alone, so all
        three have to move by the same factor or they stop describing one law."""
        out = [
            outlook(10, RB, {1: 10.0}),
            outlook(11, RB, {1: 8.0}),
            outlook(12, RB, {1: 2.0}),
        ]
        swapped = board(
            [(10, RB, 1, 1, "A", ""), (12, RB, 5, 2, "C", ""), (11, RB, 9, 3, "B", "")]
        )
        before = {o.player_id: o.weeks[1] for o in out}
        after = {o.player_id: o.weeks[1] for o in opinion.tilt_outlooks(out, swapped, weight=1.0)}
        for pid in (11, 12):
            f = after[pid].mean / before[pid].mean
            assert after[pid].sd == pytest.approx(before[pid].sd * f)
            assert after[pid].scale == pytest.approx(before[pid].scale * f)
            assert after[pid].p_zero == before[pid].p_zero
            assert after[pid].shape == before[pid].shape

    def test_a_player_we_do_not_project_at_all_is_left_out(self):
        """A ratio against a near-zero denominator manufactures points out of nothing:
        the 300-row Draft Kit board ranks kickers ESPN projects at 0.00, and pairing
        one of those against a real value hands him 121.5 season points."""
        out = [
            outlook(10, RB, {1: 10.0}),
            outlook(11, RB, {1: 0.0}),
            outlook(12, RB, {1: 2.0}),
        ]
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(out, BOARD, weight=1.0)}
        assert tilted[11].mean_from(1) == 0.0

    def test_an_out_of_range_weight_is_refused(self):
        with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
            opinion.tilt_outlooks([], BOARD, weight=1.5)


class TestPartialSeasonsAreNotTransported:
    """The mechanism's real limit, found by its headline.

    A rank is a rest-of-season opinion and the transport is multiplicative, so a player
    ESPN has at ~0 for eight weeks gets the whole rank-implied total crammed into the
    weeks he plays -- and those are the back half, where the bracket pays three times.
    That player was the top trade target in two of three live leagues.
    """

    def _out(self):
        return [
            outlook(10, RB, {w: 10.0 for w in range(1, 11)}),
            outlook(11, RB, {w: 8.0 for w in range(1, 11)}),
            # Out through week 6 -- six absences, well past the threshold of 3. The
            # calibration's absence is 0.06 with p_zero ~0.84, never an exact zero.
            outlook(12, RB, {**{w: 0.064 for w in range(1, 7)}, **{w: 6.0 for w in range(7, 11)}}),
        ]

    def test_a_bye_plus_one_missed_week_is_still_transported(self):
        """The threshold is 3 because the live board splits 2-versus-8 with nothing in
        between: a bye-plus-one group (Bowers, Henderson) whose transport moves them 5%
        and 12%, and Tyson at 53%. Excluding the first group threw away the analyst
        disagreeing, which is the only reason to read his board."""
        out = [
            outlook(10, RB, {**{w: 10.0 for w in range(1, 11)}, 3: 0.0, 7: 0.0}),  # 80
            outlook(11, RB, {w: 6.0 for w in range(1, 11)}),  # 60
        ]
        swapped = board([(11, RB, 1, 1, "B", ""), (10, RB, 2, 2, "A", "")])
        assert opinion.partial_season(out, swapped) == ()
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(out, swapped, weight=1.0)}
        assert tilted[11].mean_from(1) > out[1].mean_from(1)

    def test_the_absent_player_keeps_his_own_numbers(self):
        # Board has the injured player at RB1 -- an ROS opinion that has netted out the
        # missed games -- which the transport would otherwise read as "RB1 every week".
        hot = board([(12, RB, 1, 1, "C", ""), (10, RB, 2, 2, "A", ""), (11, RB, 3, 3, "B", "")])
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(self._out(), hot, weight=1.0)}
        assert tilted[12] is self._out()[2] or tilted[12] == self._out()[2]
        assert tilted[12].weeks[8].mean == pytest.approx(6.0)

    def test_the_others_are_still_re_dealt_among_themselves(self):
        hot = board([(12, RB, 1, 1, "C", ""), (11, RB, 2, 2, "B", ""), (10, RB, 3, 3, "A", "")])
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(self._out(), hot, weight=1.0)}
        # 10 and 11 swap; the injured player is out of the ladder on both sides.
        assert tilted[11].mean_from(1) == pytest.approx(100.0)
        assert tilted[10].mean_from(1) == pytest.approx(80.0)

    def test_a_bye_is_not_an_absence(self):
        with_bye = [
            outlook(10, RB, {**{w: 10.0 for w in range(1, 11)}, 5: 0.0}),
            outlook(11, RB, {w: 8.0 for w in range(1, 11)}),
        ]
        swapped = board([(11, RB, 1, 1, "B", ""), (10, RB, 2, 2, "A", "")])
        assert opinion.partial_season(with_bye, swapped) == ()
        tilted = {o.player_id: o for o in opinion.tilt_outlooks(with_bye, swapped, weight=1.0)}
        assert tilted[11].mean_from(1) == pytest.approx(90.0)

    def test_partial_season_names_them(self):
        hot = board([(12, RB, 1, 1, "C", ""), (10, RB, 2, 2, "A", "")])
        names = [o.player_id for o in opinion.partial_season(self._out(), hot)]
        assert names == [12]


class TestTheHorizonIsTheRemainingWeeks:
    """`pipeline._fill_weeks` only ADDS weeks, so `sim.outlooks` keeps every week ESPN
    projected -- the played ones included, and a week 18 the league never scores.

    Ranking over "every week present" is right in week 1 and wrong from week 2, which
    is the worst way for it to be wrong: nothing shows in the measurements you take on
    the day you build it.
    """

    #: Week 10 of an 18-week projection set. One player was excellent through
    #: September and is finished; the other is about to carry you. Over the full
    #: season they are identical, which is exactly the confusion being tested.
    EARLY = outlook(10, RB, {**{w: 20.0 for w in range(1, 10)}, **{w: 2.0 for w in range(10, 19)}})
    LATE = outlook(11, RB, {**{w: 2.0 for w in range(1, 10)}, **{w: 20.0 for w in range(10, 19)}})
    REMAINING = tuple(range(10, 18))  # the league's last scored week is 17

    def _board(self):
        # A rest-of-season board, which correctly has the finishing player first.
        return board([(11, RB, 1, 1, "Late", ""), (10, RB, 2, 2, "Early", "")])

    def test_the_ladder_is_built_over_the_remaining_weeks(self):
        out = [self.EARLY, self.LATE]
        tilted = {
            o.player_id: o
            for o in opinion.tilt_outlooks(out, self._board(), weight=1.0, weeks=self.REMAINING)
        }
        ours = sorted(opinion._value(o, frozenset(self.REMAINING)) for o in out)
        after = sorted(opinion._value(o, frozenset(self.REMAINING)) for o in tilted.values())
        assert after == pytest.approx(ours)
        # The board's pick takes the better of the two remaining-week values.
        assert opinion._value(tilted[11], frozenset(self.REMAINING)) == pytest.approx(max(ours))

    def test_over_the_whole_season_the_two_are_indistinguishable(self):
        """Why the wrong horizon is silent rather than loud: it does not crash, it
        ranks two very different players as equals."""
        whole = frozenset(range(1, 19))
        assert opinion._value(self.EARLY, whole) == pytest.approx(
            opinion._value(self.LATE, whole)
        )

    def test_weeks_outside_the_horizon_are_not_scaled(self):
        """A week that was not in the ladder has no business being moved by it --
        including week 18, which `state.weeks` stops short of."""
        out = [self.EARLY, self.LATE]
        tilted = {
            o.player_id: o
            for o in opinion.tilt_outlooks(out, self._board(), weight=1.0, weeks=self.REMAINING)
        }
        for pid, before in ((10, self.EARLY), (11, self.LATE)):
            for week in (1, 5, 9, 18):
                assert tilted[pid].weeks[week].mean == pytest.approx(before.weeks[week].mean)

    def test_an_absence_already_played_is_not_a_reason_to_skip_him(self):
        """The permanent-exclusion half. Tyson's weeks 1-8 stay in the outlooks after
        he returns; counting them would bench the board's opinion of him for the rest
        of the season, which is the opposite of what a rest-of-season rank means."""
        returning = outlook(
            12, RB, {**{w: 0.064 for w in range(1, 9)}, **{w: 8.0 for w in range(9, 19)}}
        )
        healthy = outlook(13, RB, {w: 9.0 for w in range(1, 19)})
        pair = board([(12, RB, 1, 1, "Back", ""), (13, RB, 2, 2, "Fine", "")])
        assert opinion.partial_season([returning, healthy], pair) == (returning,)
        assert opinion.partial_season([returning, healthy], pair, weeks=self.REMAINING) == ()

    def test_he_is_still_skipped_while_the_absence_is_ahead_of_him(self):
        out_now = outlook(
            12, RB, {**{w: 0.064 for w in range(1, 9)}, **{w: 8.0 for w in range(9, 19)}}
        )
        healthy = outlook(13, RB, {w: 9.0 for w in range(1, 19)})
        pair = board([(12, RB, 1, 1, "Out", ""), (13, RB, 2, 2, "Fine", "")])
        assert opinion.partial_season([out_now, healthy], pair, weeks=range(1, 18)) == (out_now,)
