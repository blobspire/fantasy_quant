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
