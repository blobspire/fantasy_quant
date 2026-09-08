"""Market-arbitrage tests.

Four of these exist because a market screen fails *quietly*. A disagreement board that
ranks noise first still prints ten plausible names, an availability filter that leaks
rostered players still produces a board, a velocity computed off misaligned dates still
returns a number, and a predictive check measured in sample always agrees with itself. So
each of those four is pinned against a planted answer rather than against "it ran":

* `test_planted_mispricing_ranks_first` builds a pool the field prices in exactly our order,
  moves one player, and demands that he -- and only he -- comes back.
* `test_availability_excludes_rostered` puts the biggest disagreement on a roster.
* `test_single_capture_*` runs the velocity path against a corpus with one capture, which is
  the state the real 2026 corpus is in today, and against a corpus where a player is missing
  from the middle capture, which is how a date-misaligned join manufactures a trend.
* `test_predictive_check_is_held_out` gives the check a signal that fits the training pairs
  perfectly and reverses on the last one; an in-sample check reports it as a triumph.

A second group exists because the first version of this module passed all of the above and
still overstated three findings. Each of these plants the *artifact* rather than the signal
and demands the screen come back empty-handed:

* `test_dispersion_check_does_not_report_a_HUMP_as_a_signal` and
  `test_a_NON_MONOTONE_confound_needs_more_than_a_straight_line` plant an inverted-U in
  ownership driving both the signal and the outcome, with nothing linking them to each
  other. A linear-in-rank control reports +0.31 for it -- which is the size of the number
  this module used to headline -- and the flexible control has to report nothing.
* `test_board_is_ordered_by_what_the_claim_adds_not_by_the_rank_gap` plants the ordering
  failure the live board actually printed: a near-replacement body with a big rank gap above
  a valuable one with a small gap.
* `test_a_position_that_agrees_perfectly_is_not_infinite_confidence` pins that a zero
  standard error is not certainty. The test it replaces asserted `t < 0` against a `t` of
  negative infinity and therefore could not fail.
* `test_the_direction_family_counts_its_own_multiplicity` runs the whole eight-analyst,
  two-board family so a single p < 0.05 is read against the sixteen tests that produced it.

The live payload fixtures are hand-built from the shapes measured on 2026-09-07, including
the awkward ones: an absent consensus `averageRank`, a player ranked at two slots, and an
analyst who publishes STANDARD as well as PPR. The `network` tests at the bottom reconcile
those shapes against the real endpoint.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import pytest

from fantasy_quant.core import QB, RB, TE, WR
from fantasy_quant.decide.valuation import MarketQuote, PlayerValue
from fantasy_quant.edges.market import (
    CLAY_SOURCE,
    MAX_VORP_TIE,
    MIN_CLAIM_VORP,
    MIN_POSITION_N,
    PUBLISHING_SOURCES,
    RANK_SOURCES,
    SEASON_BOARD,
    AnalystBoard,
    MarketError,
    MarketSnapshot,
    PoolRecord,
    analyst_direction_family,
    analyst_signals,
    availability_board,
    compare_adp,
    corpus_depth,
    crosswalk,
    dispersion_vs_volatility,
    drop_vorp_ties,
    family_verdict,
    field_disagreements,
    gap_vs_direction,
    ownership_momentum,
    parse_boards,
    parse_pool,
    partial_spearman,
    predicts_ownership_change,
    rank_type_for,
    sleeper_flavor_for,
)

# --------------------------------------------------------------------------------------
# Fixtures: a synthetic pool whose "true" order we control
# --------------------------------------------------------------------------------------

SEASON = 2026
CAPTURE = dt.datetime(2026, 9, 7, 12, 0)


def player_value(pid: int, vorp: float, *, position_id: int = WR, name: str = "") -> PlayerValue:
    """A `PlayerValue` carrying nothing but the number the screen ranks on."""
    return PlayerValue(
        player_id=pid,
        name=name or f"P{pid}",
        position_id=position_id,
        ros_points=100.0 + vorp,
        ros_vorp=vorp,
        ros_weeks=17,
        playoff_points=20.0,
        playoff_vorp=vorp / 5.0,
        playoff_weeks=3,
    )


def aligned_market(values: Sequence[PlayerValue]) -> dict[int, MarketQuote]:
    """Quotes that price the pool in exactly our order. Zero disagreement, by construction.

    Every screen below measures a *departure* from this, so the baseline has to be exactly
    flat or the planted signal is competing with the fixture's own noise.
    """
    ordered = sorted(values, key=lambda v: -v.ros_vorp)
    return {
        v.player_id: MarketQuote(
            player_id=v.player_id,
            adp=float(rank),
            percent_owned=100.0 - rank,
            auction_value=float(len(ordered) - rank + 1),
            draft_rank=float(rank),
        )
        for rank, v in enumerate(ordered, start=1)
    }


@pytest.fixture
def flat_pool() -> tuple[list[PlayerValue], dict[int, MarketQuote]]:
    values = [player_value(100 + i, vorp=200.0 - 4.0 * i) for i in range(40)]
    return values, aligned_market(values)


def ranking_entry(source: int, rank: float, *, slot: int = 4, rank_type: str = "PPR") -> dict:
    return {
        "auctionValue": 0,
        "published": True,
        "rank": rank,
        "rankSourceId": source,
        "rankType": rank_type,
        "slotId": slot,
    }


def pool_entry(
    pid: int,
    *,
    name: str,
    position_id: int = WR,
    percent_owned: float = 50.0,
    percent_started: float = 25.0,
    percent_change: float = 0.0,
    adp: float | None = 40.0,
    auction_value: float | None = 5.0,
    draft_ranks: Mapping[str, float] | None = None,
    rankings: Mapping[str, list[dict]] | None = None,
) -> dict:
    """One `kona_player_info` entry in the shape the live endpoint returns it."""
    return {
        "id": pid,
        "onTeamId": 0,
        "player": {
            "id": pid,
            "fullName": name,
            "defaultPositionId": position_id,
            "proTeamId": 12,
            "injuryStatus": "ACTIVE",
            "ownership": {
                "percentOwned": percent_owned,
                "percentStarted": percent_started,
                "percentChange": percent_change,
                "averageDraftPosition": adp,
                "averageDraftPositionPercentChange": 0.0,
                "auctionValueAverage": auction_value,
                "auctionValueAverageChange": 0.0,
            },
            "draftRanksByRankType": {
                rt: {"rank": rank, "auctionValue": 0, "rankSourceId": 0, "rankType": rt}
                for rt, rank in (draft_ranks or {"PPR": 50.0, "STANDARD": 55.0}).items()
            },
            "rankings": dict(rankings or {}),
        },
    }


# --------------------------------------------------------------------------------------
# Parsing the live shapes
# --------------------------------------------------------------------------------------


class TestParsing:
    def test_boards_are_split_by_slot(self) -> None:
        """A dual-eligible player has two boards, not one wide one.

        The live example is Travis Hunter, ranked as a WR by five analysts and as a DB by
        three. Pooling them reports a 73-rank analyst disagreement that is really two
        different questions each answered consistently.
        """
        rankings = {
            "0": [
                *(ranking_entry(s, 10.0, slot=4) for s in (3, 5, 6, 7, 9)),
                *(ranking_entry(s, 90.0, slot=14) for s in (10, 11, 12)),
            ]
        }
        boards = parse_boards(
            pool_entry(1, name="Dual", rankings=rankings)["player"],
            player_id=1,
            name="Dual",
            position_id=WR,
        )
        assert {b.slot_id for b in boards} == {4, 14}
        wide = {b.slot_id: b for b in boards}
        assert wide[4].dispersion == 0.0
        assert wide[4].consensus == 10.0
        assert wide[14].consensus == 90.0

    def test_consensus_source_is_not_an_analyst(self) -> None:
        """`rankSourceId` 0 carries `averageRank` and must not enter the mean."""
        rankings = {
            "0": [
                ranking_entry(7, 2.0),
                ranking_entry(6, 4.0),
                {
                    "rank": 0,
                    "averageRank": 99.0,
                    "rankSourceId": 0,
                    "rankType": "PPR",
                    "slotId": 4,
                },
            ]
        }
        (board,) = parse_boards(
            pool_entry(1, name="X", rankings=rankings)["player"],
            player_id=1,
            name="X",
            position_id=WR,
        )
        assert board.ranks == {6: 4.0, 7: 2.0}
        assert board.consensus == 3.0
        assert board.espn_average == 99.0
        assert board.consensus_lag == pytest.approx(96.0)

    def test_gap_is_leave_one_out(self) -> None:
        board = AnalystBoard(
            player_id=1,
            name="X",
            position_id=WR,
            slot_id=4,
            rank_type="PPR",
            scoring_period=0,
            ranks={7: 10.0, 3: 20.0, 5: 20.0, 6: 20.0, 9: 20.0},
        )
        # Against the full consensus (18.0) the gap would read -8.0; against the other four
        # it is -10.0, which is the number that says how far Clay is from everyone else.
        assert board.gap(CLAY_SOURCE) == pytest.approx(-10.0)
        assert board.clay_gap == pytest.approx(-10.0)
        assert board.gap(99) is None

    def test_single_analyst_board_has_zero_dispersion_not_nan(self) -> None:
        board = AnalystBoard(
            player_id=1,
            name="X",
            position_id=WR,
            slot_id=4,
            rank_type="PPR",
            scoring_period=0,
            ranks={7: 10.0},
        )
        assert board.dispersion == 0.0
        assert board.gap(CLAY_SOURCE) is None

    def test_draft_rank_blends_for_a_half_ppr_league(self) -> None:
        rec = PoolRecord(
            player_id=1,
            name="X",
            position_id=WR,
            pro_team_id=1,
            draft_ranks={"PPR": 10.0, "STANDARD": 20.0},
        )
        assert rec.draft_rank(("PPR",)) == 10.0
        assert rec.draft_rank(("STANDARD",)) == 20.0
        assert rec.draft_rank(("PPR", "STANDARD")) == 15.0
        assert rec.draft_rank(("SUPERFLEX",)) is None

    def test_board_selection_respects_the_scoring_period(self) -> None:
        """Season and week boards are different measurements; see `SEASON_BOARD`."""
        rankings = {
            "0": [ranking_entry(s, 5.0) for s in (3, 5, 6, 7, 9)],
            "1": [ranking_entry(s, 60.0) for s in (3, 5, 6, 7, 9, 10, 11, 12)],
        }
        snap = parse_pool([pool_entry(1, name="X", rankings=rankings)], season=SEASON)
        rec = snap.record(1)
        assert rec is not None
        assert rec.board(("PPR",), scoring_period=SEASON_BOARD).consensus == 5.0
        assert rec.board(("PPR",), scoring_period=1).consensus == 60.0
        # Unfiltered, the widest board wins -- which is the week board here, and is exactly
        # the silent substitution `SEASON_BOARD` exists to prevent.
        assert rec.board(("PPR",), scoring_period=None).consensus == 60.0

    def test_start_rate_survives_zero_ownership(self) -> None:
        rec = PoolRecord(
            player_id=1,
            name="X",
            position_id=WR,
            pro_team_id=1,
            percent_owned=0.0,
            percent_started=0.0,
        )
        assert rec.start_rate is None

    def test_empty_payload_raises(self) -> None:
        with pytest.raises(MarketError):
            parse_pool([], season=SEASON)

    def test_named_sources_cover_the_publishers(self) -> None:
        assert set(PUBLISHING_SOURCES) <= set(RANK_SOURCES)
        assert RANK_SOURCES[CLAY_SOURCE] == "Mike Clay"


class TestRankType:
    def test_reads_the_leagues_own_scoring(self) -> None:
        full = rank_type_for(lambda stats, pos: stats.get("53", 0.0) * 1.0)
        half = rank_type_for(lambda stats, pos: stats.get("53", 0.0) * 0.5)
        standard = rank_type_for(lambda stats, pos: 0.0)
        assert full == ("PPR",)
        assert standard == ("STANDARD",)
        # ESPN publishes nothing between the two, so half PPR reads both and averages.
        assert half == ("PPR", "STANDARD")
        assert sleeper_flavor_for(half) == "adp_half_ppr"
        assert sleeper_flavor_for(full) == "adp_ppr"


# --------------------------------------------------------------------------------------
# Screen 1: the disagreement board
# --------------------------------------------------------------------------------------


class TestDisagreement:
    def test_planted_mispricing_ranks_first(self, flat_pool) -> None:
        """A pool priced exactly in our order, with one player moved, returns that player.

        The point of the fixture is that the *only* disagreement in it is the one planted,
        so first place cannot be won by fixture noise.
        """
        values, quotes = flat_pool
        target = values[30]  # our WR31; the field will be told he is its WR2
        quotes = dict(quotes)
        quotes[target.player_id] = MarketQuote(
            player_id=target.player_id,
            adp=1.5,
            percent_owned=99.0,
            auction_value=60.0,
            draft_rank=2.0,
        )
        edges = field_disagreements(values, quotes, positions=(WR,), min_metrics=4)
        assert edges[0].player_id == target.player_id
        # We rate him far BELOW the field, so this is a sell and every metric says so.
        assert not edges[0].is_buy
        assert edges[0].agreement == 4
        assert abs(edges[1].percentile_delta) < abs(edges[0].percentile_delta) / 3

    def test_planted_buy_ranks_first(self, flat_pool) -> None:
        values, quotes = flat_pool
        target = values[3]  # our WR4, priced by the field as its WR38
        quotes = dict(quotes)
        quotes[target.player_id] = MarketQuote(
            player_id=target.player_id,
            adp=38.0,
            percent_owned=2.0,
            auction_value=1.0,
            draft_rank=38.0,
        )
        edges = field_disagreements(values, quotes, positions=(WR,), min_metrics=4)
        assert edges[0].player_id == target.player_id
        assert edges[0].is_buy
        assert edges[0].rank_delta > 20

    def test_percentile_delta_normalises_across_metric_subsets(self) -> None:
        """A 20-rank gap on a 40-player board is a bigger disagreement than on a 400 one.

        `market_disagreements` ranks each metric over its own informative subset, so raw
        deltas are not comparable between metrics. Pooling the raw numbers would silently
        weight whichever metric priced the most players.
        """
        values = [player_value(200 + i, vorp=100.0 - i) for i in range(40)]
        quotes = aligned_market(values)
        target = values[5]
        # Priced by ADP only, and moved 20 ranks inside that 40-player board.
        quotes[target.player_id] = MarketQuote(player_id=target.player_id, adp=26.0)
        edges = field_disagreements(values, quotes, positions=(WR,), metrics=("adp",))
        top = next(e for e in edges if e.player_id == target.player_id)
        metric = top.per_metric["adp"]
        assert metric.universe == 40
        # Our WR6 against the field's WR25: the ADP of 26.0 slots in below 25 others.
        assert metric.delta == pytest.approx(19.0)
        assert top.percentile_delta == pytest.approx(19.0 / 40.0)

    def test_tie_block_is_dropped(self) -> None:
        """The projection layer's dead tail is not an opinion and must not be ranked.

        Measured on the live 12-team half-PPR league: 42 receivers share a VORP of -73.63
        because ESPN projects each of them the same token 1.1 points for the season. Their
        ranks are player ids, and the first live board duly recommended three of them.
        """
        real = [player_value(300 + i, vorp=100.0 - i) for i in range(5)]
        tail = [player_value(400 + i, vorp=-73.6331) for i in range(10)]
        kept, dropped = drop_vorp_ties([*real, *tail], max_tie=MAX_VORP_TIE)
        assert dropped == 10
        assert {v.player_id for v in kept} == {v.player_id for v in real}

        quotes = aligned_market([*real, *tail])
        edges = field_disagreements([*real, *tail], quotes, positions=(WR,))
        assert all(e.player_id < 400 for e in edges)

    def test_tie_guard_is_per_position(self) -> None:
        """Two positions sharing a VORP is a coincidence, not a tie block."""
        values = [
            player_value(1, vorp=10.0, position_id=WR),
            player_value(2, vorp=10.0, position_id=RB),
            player_value(3, vorp=10.0, position_id=TE),
            player_value(4, vorp=10.0, position_id=QB),
        ]
        kept, dropped = drop_vorp_ties(values, max_tie=1)
        assert dropped == 0
        assert len(kept) == 4

    def test_depth_filter_asks_a_different_question_of_each_direction(self) -> None:
        """A buy must be someone we would roster; a sell, someone THEY would roster."""
        values = [player_value(500 + i, vorp=50.0 - 2.0 * i) for i in range(40)]
        quotes = aligned_market(values)
        # A deep player we like: our WR35, the field's WR40. Below replacement, so no buy.
        deep = values[34]
        quotes[deep.player_id] = MarketQuote(player_id=deep.player_id, draft_rank=40.0)
        # A deep player the field likes: our WR38, their WR3. A real sell.
        chased = values[37]
        quotes[chased.player_id] = MarketQuote(player_id=chased.player_id, draft_rank=3.0)

        edges = field_disagreements(
            values, quotes, positions=(WR,), metrics=("draft_rank",), depth={WR: 20.0}
        )
        found = {e.player_id: e for e in edges}
        assert deep.player_id not in found
        assert chased.player_id in found
        assert not found[chased.player_id].is_buy

    def test_positions_are_screened_separately(self) -> None:
        """A kicker drafted last is not underrated; he is a kicker."""
        wrs = [player_value(600 + i, vorp=80.0 - i, position_id=WR) for i in range(20)]
        qbs = [player_value(700 + i, vorp=80.0 - i, position_id=QB) for i in range(20)]
        values = [*wrs, *qbs]
        quotes = {}
        for rank, v in enumerate(wrs, start=1):
            quotes[v.player_id] = MarketQuote(player_id=v.player_id, draft_rank=float(rank))
        # The whole QB block is priced after every WR, exactly as a real board does it.
        for rank, v in enumerate(qbs, start=1):
            quotes[v.player_id] = MarketQuote(player_id=v.player_id, draft_rank=float(100 + rank))
        edges = field_disagreements(values, quotes, positions=(WR, QB), metrics=("draft_rank",))
        # Ranked within position, nobody disagrees at all.
        assert all(abs(e.rank_delta) < 1e-9 for e in edges)


# --------------------------------------------------------------------------------------
# Screen 5: availability
# --------------------------------------------------------------------------------------


class TestAvailability:
    def test_availability_excludes_rostered(self, flat_pool) -> None:
        """The biggest buy in the league is worth nothing if a rival already has him."""
        values, quotes = flat_pool
        target = values[3]
        runner_up = values[6]
        quotes = dict(quotes)
        for v, rank in ((target, 39.0), (runner_up, 30.0)):
            quotes[v.player_id] = MarketQuote(
                player_id=v.player_id,
                adp=rank,
                percent_owned=100.0 - rank,
                auction_value=float(41 - rank),
                draft_rank=rank,
            )
        edges = field_disagreements(values, quotes, positions=(WR,))
        buys = [e for e in edges if e.is_buy]
        assert buys[0].player_id == target.player_id

        board = availability_board(edges, rostered=[target.player_id])
        assert target.player_id not in {e.player_id for e in board}
        assert board[0].player_id == runner_up.player_id
        assert all(e.available for e in board)

    def test_availability_keeps_only_buys_by_default(self, flat_pool) -> None:
        values, quotes = flat_pool
        quotes = dict(quotes)
        sell = values[2]
        quotes[sell.player_id] = MarketQuote(
            player_id=sell.player_id, adp=1.0, percent_owned=99.9, draft_rank=1.0
        )
        edges = field_disagreements(values, quotes, positions=(WR,))
        board = availability_board(edges, rostered=())
        assert sell.player_id not in {e.player_id for e in board}
        assert all(e.is_buy for e in board)

    def test_availability_is_not_percent_owned(self) -> None:
        """A 0.1%-owned player already on a rival's bench is not available here.

        ESPN's roster rate describes the population; the only thing that decides whether a
        claim is possible is whether somebody in *this* league has him.
        """
        values = [player_value(800 + i, vorp=60.0 - i) for i in range(20)]
        quotes = aligned_market(values)
        quotes[values[1].player_id] = MarketQuote(
            player_id=values[1].player_id,
            adp=19.0,
            percent_owned=0.1,
            auction_value=0.1,
            draft_rank=19.0,
        )
        edges = field_disagreements(values, quotes, positions=(WR,))
        board = availability_board(edges, rostered=[values[1].player_id])
        assert values[1].player_id not in {e.player_id for e in board}

    def test_board_is_ordered_by_what_the_claim_adds_not_by_the_rank_gap(self) -> None:
        """The live failure this pins: a fullback at the top of the biggest league's board.

        `percentile_delta` is mechanically larger the deeper you go -- on the real Wine
        Wednesday board it correlates +0.31 with our own positional rank and -0.40 with VORP
        -- and the available slice is by construction the deep end, so ordering it by the gap
        ordered it by depth. It printed Kyle Juszczyk (+2.2 VORP over 17 weeks, 0.13 a week,
        priced ~22 ranks lower by the field) above Ty Johnson at +37.1. Ranks are how a
        disagreement is found; points are what a claim is worth.
        """
        values = [player_value(900 + i, vorp=140.0 - 4.0 * i) for i in range(40)]
        quotes = dict(aligned_market(values))
        # A near-replacement body the field rates 20 ranks lower than we do.
        fullback = values[32]  # vorp +12, comfortably clear of the claim floor
        # A genuinely valuable free agent the field rates only three ranks lower.
        starter = values[8]  # vorp +108
        quotes[fullback.player_id] = MarketQuote(
            player_id=fullback.player_id,
            adp=41.0,
            percent_owned=0.5,
            auction_value=0.5,
            draft_rank=41.0,
        )
        quotes[starter.player_id] = MarketQuote(
            player_id=starter.player_id,
            adp=12.5,
            percent_owned=87.5,
            auction_value=28.5,
            draft_rank=12.5,
        )
        edges = field_disagreements(values, quotes, positions=(WR,))
        gaps = {e.player_id: e.percentile_delta for e in edges}
        assert gaps[fullback.player_id] > gaps[starter.player_id], "fixture must plant the trap"

        board = availability_board(edges, rostered=())
        assert board[0].player_id == starter.player_id
        assert board[0].ros_vorp > board[-1].ros_vorp
        assert board[0].vorp_per_week == pytest.approx(starter.ros_vorp / 17.0)

    def test_board_drops_players_not_worth_a_roster_spot(self) -> None:
        """A disagreement about a player worth 0.1 points a week is not a move."""
        values = [player_value(950 + i, vorp=40.0 - 2.0 * i) for i in range(30)]
        quotes = dict(aligned_market(values))
        scrub = values[19]  # vorp +2.0, i.e. 0.12 points a week
        quotes[scrub.player_id] = MarketQuote(
            player_id=scrub.player_id,
            adp=30.0,
            percent_owned=1.0,
            auction_value=1.0,
            draft_rank=30.0,
        )
        edges = field_disagreements(values, quotes, positions=(WR,))
        assert scrub.ros_vorp < MIN_CLAIM_VORP
        assert scrub.player_id in {e.player_id for e in edges if e.is_buy}
        assert scrub.player_id not in {e.player_id for e in availability_board(edges, rostered=())}
        # The diagnostic view still shows him, because the filter is a claim bar and not a
        # claim that he does not exist.
        unfiltered = availability_board(edges, rostered=(), min_vorp=None)
        assert scrub.player_id in {e.player_id for e in unfiltered}


# --------------------------------------------------------------------------------------
# Screen 3: ownership momentum
# --------------------------------------------------------------------------------------


def history_frame(rows: Sequence[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "espn_id": pl.Int64,
            "full_name": pl.Utf8,
            "default_position_id": pl.Int64,
            "captured_at": pl.Datetime,
            "percent_owned": pl.Float64,
            "average_draft_position": pl.Float64,
            "percent_change": pl.Float64,
        },
    )


def capture_row(pid: int, when: dt.datetime, owned: float, adp: float, change: float = 0.0):
    return {
        "espn_id": pid,
        "full_name": f"P{pid}",
        "default_position_id": WR,
        "captured_at": when,
        "percent_owned": owned,
        "average_draft_position": adp,
        "percent_change": change,
    }


class TestOwnershipMomentum:
    def test_single_capture_does_not_divide_by_zero(self) -> None:
        """The state the real 2026 corpus is in: one capture, so no denominator.

        `None` rather than `0.0`, because "not moving" and "we cannot tell" are different
        claims and a board that renders them identically is lying about the second one.
        """
        frame = history_frame([capture_row(i, CAPTURE, 50.0, 40.0, change=0.5) for i in (1, 2)])
        tracks = ownership_momentum(frame)
        assert len(tracks) == 2
        assert all(t.owned_velocity is None for t in tracks)
        assert all(t.adp_drift is None for t in tracks)
        assert all(t.span_days == 0.0 for t in tracks)
        assert all(not t.measured for t in tracks)
        # The fallback still orders the board by ESPN's own trailing delta.
        assert all(t.espn_percent_change == 0.5 for t in tracks)

    def test_velocity_is_computed_on_aligned_dates(self) -> None:
        """A player absent from the middle capture must not have his series shifted.

        This is how a date-misaligned join manufactures a trend: stack the daily files, take
        "first" and "last" by row order, and the player who missed a day gets differenced
        against somebody else's window.
        """
        days = [CAPTURE + dt.timedelta(days=d) for d in (0, 1, 2)]
        rows = [
            capture_row(1, days[0], 10.0, 100.0),
            capture_row(2, days[0], 80.0, 20.0),
            # Player 2 is missing from the middle capture entirely.
            capture_row(1, days[1], 20.0, 90.0),
            capture_row(1, days[2], 30.0, 80.0),
            capture_row(2, days[2], 84.0, 18.0),
        ]
        tracks = {t.player_id: t for t in ownership_momentum(history_frame(rows))}
        assert tracks[1].captures == 3
        assert tracks[1].span_days == pytest.approx(2.0)
        assert tracks[1].owned_velocity == pytest.approx(10.0)
        assert tracks[1].adp_drift == pytest.approx(-10.0)
        # Two captures two days apart, not three captures one day apart.
        assert tracks[2].captures == 2
        assert tracks[2].span_days == pytest.approx(2.0)
        assert tracks[2].owned_velocity == pytest.approx(2.0)

    def test_out_of_order_rows_are_sorted_before_differencing(self) -> None:
        days = [CAPTURE + dt.timedelta(days=d) for d in (0, 4)]
        rows = [capture_row(1, days[1], 30.0, 60.0), capture_row(1, days[0], 10.0, 80.0)]
        (track,) = ownership_momentum(history_frame(rows))
        assert track.first_owned == 10.0
        assert track.last_owned == 30.0
        assert track.owned_velocity == pytest.approx(5.0)

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(MarketError):
            ownership_momentum(pl.DataFrame({"espn_id": [1]}))

    def test_empty_history_is_empty(self) -> None:
        assert ownership_momentum(pl.DataFrame()) == ()

    def test_corpus_depth_reports_a_single_capture_as_unusable(self) -> None:
        frame = history_frame([capture_row(i, CAPTURE, 50.0, 40.0) for i in range(5)])
        depth = corpus_depth(SEASON, history=frame)
        assert depth.n_captures == 1
        assert depth.span_days == 0.0
        assert depth.players == 5
        assert not depth.usable
        assert "frozen" in depth.note

    def test_corpus_depth_reports_a_real_series_as_usable(self) -> None:
        rows = [
            capture_row(1, CAPTURE + dt.timedelta(days=d), 10.0 * d, 100.0 - d) for d in range(4)
        ]
        depth = corpus_depth(SEASON, history=history_frame(rows))
        assert depth.n_captures == 4
        assert depth.span_days == pytest.approx(3.0)
        assert depth.usable


# --------------------------------------------------------------------------------------
# Screen 2: the analyst boards and the predictive check
# --------------------------------------------------------------------------------------


class TestAnalystScreen:
    def test_signals_are_sorted_most_bullish_first(self) -> None:
        entries = [
            pool_entry(
                1,
                name="ClayHigh",
                percent_owned=40.0,
                rankings={
                    "0": [ranking_entry(7, 5.0), *(ranking_entry(s, 40.0) for s in (3, 5, 6, 9))]
                },
            ),
            pool_entry(
                2,
                name="ClayLow",
                percent_owned=40.0,
                rankings={
                    "0": [ranking_entry(7, 60.0), *(ranking_entry(s, 20.0) for s in (3, 5, 6, 9))]
                },
            ),
        ]
        snap = parse_pool(entries, season=SEASON)
        signals = analyst_signals(snap, min_percent_owned=1.0)
        assert [s.player_id for s in signals] == [1, 2]
        assert signals[0].is_bullish
        assert signals[0].gap == pytest.approx(-35.0)
        assert signals[-1].gap == pytest.approx(40.0)
        assert signals[0].source == "Mike Clay"

    def test_thin_boards_are_dropped(self) -> None:
        entries = [
            pool_entry(
                1, name="Thin", rankings={"0": [ranking_entry(7, 5.0), ranking_entry(3, 8.0)]}
            )
        ]
        snap = parse_pool(entries, season=SEASON)
        assert analyst_signals(snap) == ()
        assert analyst_signals(snap, min_analysts=2) != ()

    def test_dispersion_check_finds_a_planted_association(self) -> None:
        """A pool where spread and |ownership change| are wired together, plus controls."""
        rng = np.random.default_rng(7)
        entries = []
        for i in range(120):
            spread = float(rng.uniform(0.0, 20.0))
            centre = 20.0 + 0.5 * i
            ranks = [centre - spread, centre, centre + spread, centre, centre]
            entries.append(
                pool_entry(
                    i + 1,
                    name=f"P{i}",
                    percent_owned=float(rng.uniform(5.0, 90.0)),
                    percent_change=float(spread * rng.choice([-1.0, 1.0]) * 0.1),
                    rankings={
                        "0": [
                            ranking_entry(s, r) for s, r in zip((3, 5, 6, 7, 9), ranks, strict=True)
                        ]
                    },
                )
            )
        check = dispersion_vs_volatility(parse_pool(entries, season=SEASON))
        assert check.n == 120
        assert check.statistic > 0.5
        assert check.p_value < 1e-6
        assert not check.held_out  # the whole point: this is an association, not a lead
        assert "CONTEMPORANEOUS" in check.describe()
        # Not `usable`, however small the p-value: contemporaneous is not held out.
        assert not check.usable
        # The effect size travels with the correlation, in the outcome's own units.
        assert check.detail["tercile_top_minus_bottom"] > 0.0
        assert check.detail["tercile_top_minus_bottom_se"] > 0.0
        assert "top vs bottom tercile" in check.describe()

    def test_dispersion_check_does_not_report_a_HUMP_as_a_signal(self) -> None:
        """The confound the controls exist for, planted: dispersion and movement both

        hump-shaped in ownership and conditionally independent of each other. A linear
        control on the rank of ownership cannot remove a hump, so the version of this screen
        that used one reported the confound as a +0.33 association. The screen has to come
        back empty-handed here or it is measuring the hump on the live pool too.
        """
        rng = np.random.default_rng(19)
        entries = []
        for i in range(240):
            owned = float(rng.uniform(1.0, 99.0))
            # One inverted-U in ownership drives BOTH series. Nothing links them to each
            # other except that shared shape, so a control that can see the shape must find
            # nothing and a control that can only draw a straight line must find a lot.
            hump = 1.0 - ((owned - 50.0) / 49.0) ** 2
            spread = max(0.1, 8.0 * hump + float(rng.normal(0.0, 2.0)))
            change = abs(0.5 * hump + float(rng.normal(0.0, 0.25)))
            centre = 20.0 + 0.4 * i
            ranks = [centre - spread, centre, centre + spread, centre, centre]
            entries.append(
                pool_entry(
                    i + 1,
                    name=f"P{i}",
                    percent_owned=owned,
                    percent_change=change * float(rng.choice([-1.0, 1.0])),
                    rankings={
                        "0": [
                            ranking_entry(s, r) for s, r in zip((3, 5, 6, 7, 9), ranks, strict=True)
                        ]
                    },
                )
            )
        snap = parse_pool(entries, season=SEASON)
        check = dispersion_vs_volatility(snap)
        assert check.n == 240
        assert check.p_value > 0.05, f"the hump was reported as a signal: {check.describe()}"
        assert check.verdict == "no association survives the controls"
        # And the diagnostic that says why: the linear control this replaced does fall for it.
        assert abs(check.detail["linear_control_statistic"]) > 0.3
        assert check.detail["linear_control_p"] < 0.01

    def test_direction_check_reports_absence_rather_than_inventing_one(self) -> None:
        rng = np.random.default_rng(11)
        entries = []
        for i in range(120):
            gap = float(rng.uniform(-20.0, 20.0))
            centre = 30.0 + 0.4 * i
            ranks = [centre + gap, centre, centre, centre, centre]
            entries.append(
                pool_entry(
                    i + 1,
                    name=f"P{i}",
                    percent_owned=float(rng.uniform(5.0, 90.0)),
                    percent_change=float(rng.normal(0.0, 0.3)),  # unrelated to the gap
                    rankings={
                        "0": [
                            ranking_entry(s, r) for s, r in zip((7, 3, 5, 6, 9), ranks, strict=True)
                        ]
                    },
                )
            )
        check = gap_vs_direction(parse_pool(entries, season=SEASON))
        assert check.n == 120
        assert check.p_value > 0.05
        assert "no directional signal" in check.verdict
        assert not check.usable

    def test_checks_report_insufficient_data_rather_than_a_number(self) -> None:
        entries = [
            pool_entry(
                i, name=f"P{i}", rankings={"0": [ranking_entry(s, 10.0) for s in (3, 5, 6, 7, 9)]}
            )
            for i in range(1, 5)
        ]
        check = dispersion_vs_volatility(parse_pool(entries, season=SEASON))
        assert check.n == 4
        assert check.verdict == "insufficient data"
        assert not check.usable

    def test_the_direction_family_counts_its_own_multiplicity(self) -> None:
        """Eight analysts on two boards is sixteen tests, and one of them will hit.

        On the live pool exactly one of the sixteen clears p < 0.05 (Clay, week board) --
        against 0.8 expected by chance -- which is the number that turns "significant"
        into "a scan". Here nothing is wired to anything, so the family must report roughly
        its own alpha and the printed line must carry the denominator.
        """
        rng = np.random.default_rng(41)
        entries = []
        for i in range(200):
            centre = 20.0 + 0.4 * i
            ranks = [centre + float(rng.normal(0.0, 6.0)) for _ in PUBLISHING_SOURCES]
            entries.append(
                pool_entry(
                    i + 1,
                    name=f"P{i}",
                    percent_owned=float(rng.uniform(2.0, 98.0)),
                    percent_change=float(rng.normal(0.0, 0.3)),  # wired to nothing
                    rankings={
                        "0": [
                            ranking_entry(s, r)
                            for s, r in zip(PUBLISHING_SOURCES, ranks, strict=True)
                        ]
                    },
                )
            )
        snap = parse_pool(entries, season=SEASON)
        family = analyst_direction_family(snap, scoring_periods=(SEASON_BOARD,))
        assert len(family) == len(PUBLISHING_SOURCES)
        assert {s for s, _, _ in family} == set(PUBLISHING_SOURCES)
        hits = [c for _, _, c in family if c.p_value < 0.05]
        assert len(hits) <= 2, "a null family should not light up"
        line = family_verdict(family)
        assert f"of {len(PUBLISHING_SOURCES)} analyst" in line
        assert "expected by chance" in line


class TestPredictiveCheck:
    def make_history(self, days: Sequence[int], owned: Mapping[int, Sequence[float]]):
        rows = []
        for pid, series in owned.items():
            for day, value in zip(days, series, strict=True):
                rows.append(capture_row(pid, CAPTURE + dt.timedelta(days=day), value, 50.0))
        return history_frame(rows)

    def test_single_capture_corpus_says_so(self) -> None:
        """What the real corpus does today. `n=0`, not a correlation over one point."""
        frame = self.make_history([0], {i: [50.0] for i in range(60)})
        check = predicts_ownership_change(frame, {i: float(i) for i in range(60)})
        assert check.n == 0
        assert check.verdict == "insufficient captures"
        assert "1 distinct capture" in check.note
        assert not check.held_out

    def test_captures_closer_than_the_minimum_gap_do_not_count(self) -> None:
        rows = [capture_row(i, CAPTURE, 50.0, 40.0) for i in range(60)] + [
            capture_row(i, CAPTURE + dt.timedelta(hours=2), 51.0, 40.0) for i in range(60)
        ]
        check = predicts_ownership_change(history_frame(rows), {i: float(i) for i in range(60)})
        assert check.n == 0
        assert check.verdict == "insufficient captures"

    def test_two_captures_are_reported_as_in_sample(self) -> None:
        """One pair is an association. It is allowed, and it is labelled."""
        days = [0, 2]
        owned = {i: [50.0, 50.0 + i * 0.1] for i in range(60)}
        check = predicts_ownership_change(
            self.make_history(days, owned), {i: float(i) for i in range(60)}
        )
        assert check.n == 60
        assert not check.held_out
        assert check.statistic > 0.9
        assert "not held out" in check.verdict

    def test_predictive_check_is_held_out(self) -> None:
        """A signal that fits the training pairs perfectly and reverses on the last one.

        An in-sample check would pool all three pairs, find the training pairs dominate, and
        report a triumph. The held-out design has to report the failure, because the last
        pair is the only one that gets a vote.
        """
        days = [0, 2, 4, 6]
        owned = {}
        for i in range(60):
            base = 50.0
            # Pairs 1 and 2 move WITH the signal; the held-out pair 3 moves against it.
            owned[i] = [base, base + i * 0.1, base + i * 0.2, base + i * 0.2 - i * 0.1]
        frame = self.make_history(days, owned)
        signal = {i: float(i) for i in range(60)}
        check = predicts_ownership_change(frame, signal)
        assert check.held_out
        assert check.n == 60
        assert check.detail["train_pairs"] == 2.0
        assert check.detail["in_sample_statistic"] > 0.9
        assert check.statistic < -0.9  # the held-out pair reverses
        assert check.verdict == "does not survive the held-out pair"

    def test_a_signal_that_really_predicts_is_reported_as_such(self) -> None:
        days = [0, 2, 4, 6]
        owned = {i: [50.0 + i * 0.1 * k for k in range(4)] for i in range(60)}
        check = predicts_ownership_change(
            self.make_history(days, owned), {i: float(i) for i in range(60)}
        )
        assert check.held_out
        assert check.statistic > 0.9
        assert check.verdict == "predicts out of sample"
        assert check.usable

    def test_per_capture_signals_are_aligned_to_their_own_capture(self) -> None:
        """The signal a pair is judged on is the one measured at the pair's FIRST capture."""
        days = [0, 2, 4]
        owned = {i: [50.0, 50.0 + i * 0.1, 50.0 + i * 0.1 - i * 0.1] for i in range(60)}
        frame = self.make_history(days, owned)
        times = sorted(frame["captured_at"].unique().to_list())
        # The held-out pair's own signal is reversed, so a correct alignment must find a
        # POSITIVE held-out statistic even though the raw ownership change is negative.
        signal = {
            times[0]: {i: float(i) for i in range(60)},
            times[1]: {i: -float(i) for i in range(60)},
        }
        check = predicts_ownership_change(frame, signal)
        assert check.held_out
        assert check.statistic > 0.9

    def test_missing_overlap_is_reported_rather_than_correlated(self) -> None:
        days = [0, 2]
        owned = {i: [50.0, 51.0] for i in range(60)}
        check = predicts_ownership_change(self.make_history(days, owned), {1: 1.0, 2: 2.0})
        assert check.verdict == "insufficient overlap"
        assert check.n == 2


class TestPartialSpearman:
    def test_a_control_that_explains_everything_kills_the_correlation(self) -> None:
        z = np.arange(60, dtype=float)
        x = z * 2.0
        y = z * 3.0
        raw, _, _ = partial_spearman(x, y)
        controlled, p, n = partial_spearman(x, y, [z])
        assert raw == pytest.approx(1.0)
        assert abs(controlled) < 1e-6
        assert n == 60
        assert p > 0.05

    def test_too_few_points_returns_nothing_rather_than_one(self) -> None:
        stat, p, n = partial_spearman([1.0, 2.0], [1.0, 2.0])
        assert stat == 0.0
        assert p == 1.0
        assert n == 2

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(MarketError):
            partial_spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0])

    def test_a_NON_MONOTONE_confound_needs_more_than_a_straight_line(self) -> None:
        """The defect this module actually had, in nine lines.

        `x` and `y` are independent given `z` and share nothing but an inverted-U in it.
        A partial correlation that regresses on rank(z) draws a straight line through a hump
        and leaves most of it behind, which is exactly how +0.33 was reported for a
        dispersion/ownership association that is +0.20 when the hump is removed.
        """
        rng = np.random.default_rng(23)
        z = rng.uniform(0.0, 100.0, size=300)
        hump = 1.0 - ((z - 50.0) / 50.0) ** 2
        x = 8.0 * hump + rng.normal(0.0, 2.0, size=300)
        y = 0.5 * hump + rng.normal(0.0, 0.2, size=300)

        naive, naive_p, _ = partial_spearman(x, y, [z], degree=1)
        flexible, flexible_p, _ = partial_spearman(x, y, [z], degree=3)
        assert naive > 0.25 and naive_p < 0.001, "the straight line has to fall for it"
        assert abs(flexible) < 0.12
        assert flexible_p > 0.05

    def test_the_p_value_pays_for_the_controls(self) -> None:
        """Residuals of a k-control fit do not have n - 2 degrees of freedom.

        Handing them to `pearsonr` -- which assumes they do -- spends the controls for free
        and returns a p-value too small. Same correlation, more controls, larger p.
        """
        rng = np.random.default_rng(29)
        x = rng.normal(size=80)
        y = 0.3 * x + rng.normal(size=80)
        one = partial_spearman(x, y, [rng.normal(size=80)], degree=1)
        many = partial_spearman(x, y, [rng.normal(size=80) for _ in range(4)], degree=2)
        assert abs(one[0] - many[0]) < 0.25  # broadly the same association
        assert many[1] > one[1]

    def test_a_control_basis_is_never_allowed_to_eat_the_sample(self) -> None:
        """Degree is reduced rather than honoured when the design would overfit.

        Twenty points and four cubic controls is thirteen columns for twenty observations:
        the basis can fit the noise, and a partial correlation against a saturated design is
        zero by construction rather than by evidence.
        """
        rng = np.random.default_rng(31)
        x = rng.normal(size=20)
        y = 0.8 * x + rng.normal(scale=0.3, size=20)
        stat, _, n = partial_spearman(x, y, [rng.normal(size=20) for _ in range(4)], degree=3)
        assert n == 20
        assert stat > 0.4  # the real association survives, because the design was trimmed


# --------------------------------------------------------------------------------------
# Screen 4: cross-source ADP
# --------------------------------------------------------------------------------------


class FakeSleeperRow:
    """The two attributes `crosswalk` and `compare_adp` read off a `SleeperProjection`."""

    def __init__(self, sleeper_id: str, name: str, position: str, adp: Mapping[str, float]):
        self.sleeper_id = sleeper_id
        self.name = name
        self.position = position
        self._adp = dict(adp)

    def adp(self) -> dict[str, float]:
        return dict(self._adp)


class TestCrossPlatform:
    def snapshot(self) -> MarketSnapshot:
        names = ["Jahmyr Gibbs", "Ja'Marr Chase", "Josh Allen", "Bo Nix", "Brock Purdy"]
        positions = [RB, WR, QB, QB, QB]
        adps = [40.0, 50.0, 10.0, 20.0, 30.0]
        return parse_pool(
            [
                pool_entry(10 + i, name=n, position_id=p, adp=a)
                for i, (n, p, a) in enumerate(zip(names, positions, adps, strict=True))
            ],
            season=SEASON,
        )

    def test_name_join_rescues_the_players_sleeper_has_no_espn_id_for(self) -> None:
        """Sleeper's own `espn_id` misses the recent draft classes, i.e. the top of the board.

        Measured live: 160 of 553 ADP rows resolve by id, and Gibbs, Bijan Robinson,
        Ja'Marr Chase and Puka Nacua are all among the misses. The name join takes it to 530.
        """
        snap = self.snapshot()
        rows = [
            FakeSleeperRow("s1", "Jahmyr Gibbs", "RB", {"adp_ppr": 3.0}),
            FakeSleeperRow("s3", "Josh Allen", "QB", {"adp_ppr": 21.0}),
        ]
        by_id = crosswalk(snap, rows, sleeper_espn_ids={"s3": 12})
        assert by_id["s3"] == 12  # Sleeper's own id wins where it exists
        assert by_id["s1"] == 10  # and the name join covers the rest

    def test_unknown_names_and_positions_are_dropped_not_guessed(self) -> None:
        snap = self.snapshot()
        rows = [
            FakeSleeperRow("s9", "Nobody At All", "WR", {"adp_ppr": 5.0}),
            FakeSleeperRow("s8", "Jahmyr Gibbs", "CB", {"adp_ppr": 5.0}),
        ]
        assert crosswalk(snap, rows) == {}

    def test_stale_sleeper_ids_fall_through_to_the_name_join(self) -> None:
        snap = self.snapshot()
        rows = [FakeSleeperRow("s1", "Jahmyr Gibbs", "RB", {"adp_ppr": 3.0})]
        # An espn_id that is not in this pool must not be trusted just because it is an id.
        assert crosswalk(snap, rows, sleeper_espn_ids={"s1": 999999}) == {"s1": 10}

    def wide_snapshot(self) -> MarketSnapshot:
        """Twelve players: six QBs and six skill, which is enough for a per-position `t`."""
        names = [f"Passer {i}" for i in range(6)] + [f"Runner {i}" for i in range(6)]
        positions = [QB] * 6 + [RB] * 6
        adps = [10.0 + 8.0 * i for i in range(6)] + [12.0 + 8.0 * i for i in range(6)]
        return parse_pool(
            [
                pool_entry(100 + i, name=n, position_id=p, adp=a)
                for i, (n, p, a) in enumerate(zip(names, positions, adps, strict=True))
            ],
            season=SEASON,
        )

    def test_positional_bias_is_measured_with_a_t_statistic(self) -> None:
        """The shape of the real finding: one position shifted, the rest agreeing.

        The gaps are deliberately UNEQUAL. An earlier version of this test gave every
        quarterback the identical gap, which made the standard error zero, which made the
        reported `t` infinite -- and then asserted `t < 0`, which -inf satisfies. The test
        could not fail and the statistic it was pinning was a claim of certainty from n = 3.
        """
        snap = self.wide_snapshot()
        rng = np.random.default_rng(5)
        rows = []
        for i in range(6):
            # Sleeper takes every quarterback ~24 picks later, give or take.
            rows.append(
                FakeSleeperRow(
                    f"q{i}",
                    f"Passer {i}",
                    "QB",
                    {"adp_ppr": 10.0 + 8.0 * i + 24.0 + float(rng.normal(0.0, 3.0))},
                )
            )
            rows.append(
                FakeSleeperRow(
                    f"r{i}",
                    f"Runner {i}",
                    "RB",
                    {"adp_ppr": 12.0 + 8.0 * i - 6.0 + float(rng.normal(0.0, 3.0))},
                )
            )
        result = compare_adp(snap, rows, flavor="adp_ppr")
        assert result.n == 12
        mean_gap, n_qb, t_stat = result.by_position[QB]
        assert n_qb == 6
        assert mean_gap < 0  # ESPN drafts them earlier
        assert math.isfinite(t_stat) and t_stat < -2.0
        assert result.expensive_on_espn(1)[0].position_id == QB
        assert result.cheap_on_espn(1)[0].position_id == RB
        # The two positional rows are one rotation seen twice, not two findings.
        totals = sum(mean * n for mean, n, _ in result.by_position.values())
        assert totals == pytest.approx(0.0, abs=1e-9)

    def test_a_position_that_agrees_perfectly_is_not_infinite_confidence(self) -> None:
        """Zero variance is no standard error, which is not the same as certainty.

        The first version returned `t = +/-inf` here on the argument that a perfectly
        consistent gap is the strongest evidence available. With five players who happen to
        agree it is five observations, and `describe` printed "t = -inf" beside a mean of
        -2.0 as though the board had settled the question.
        """
        # ESPN takes every quarterback before every back and the other platform reverses it
        # exactly, so each QB's rank gap is -6 and each RB's is +6, with no spread at all.
        snap = parse_pool(
            [
                pool_entry(200 + i, name=f"Passer {i}", position_id=QB, adp=10.0 + i)
                for i in range(6)
            ]
            + [
                pool_entry(300 + i, name=f"Runner {i}", position_id=RB, adp=20.0 + i)
                for i in range(6)
            ],
            season=SEASON,
        )
        rows = []
        for i in range(6):
            rows.append(FakeSleeperRow(f"q{i}", f"Passer {i}", "QB", {"adp_ppr": 30.0 + i}))
            rows.append(FakeSleeperRow(f"r{i}", f"Runner {i}", "RB", {"adp_ppr": 5.0 + i}))
        result = compare_adp(snap, rows, flavor="adp_ppr")
        mean_gap, _, t_stat = result.by_position[QB]
        assert mean_gap < 0
        assert not math.isfinite(t_stat)
        assert "n/a" in result.describe()
        assert "-inf" not in result.describe()

    def test_a_position_too_small_to_summarise_gets_no_t(self) -> None:
        """Four players who agree are a rumour with a standard error attached."""
        snap = self.snapshot()  # three QBs, one RB, one WR
        rows = [
            FakeSleeperRow("a", "Jahmyr Gibbs", "RB", {"adp_ppr": 10.0}),
            FakeSleeperRow("b", "Ja'Marr Chase", "WR", {"adp_ppr": 20.0}),
            FakeSleeperRow("c", "Josh Allen", "QB", {"adp_ppr": 50.0}),
            FakeSleeperRow("d", "Bo Nix", "QB", {"adp_ppr": 63.0}),
            FakeSleeperRow("e", "Brock Purdy", "QB", {"adp_ppr": 71.0}),
        ]
        result = compare_adp(snap, rows, flavor="adp_ppr")
        _, n_qb, t_stat = result.by_position[QB]
        assert n_qb == 3 < MIN_POSITION_N
        assert not math.isfinite(t_stat)

    def test_the_censored_adp_band_is_excluded(self) -> None:
        """845 of 1,036 live players sit in ESPN's [169, 171.6] undrafted band.

        Comparing that band to a real ADP produces a board made entirely of the artifact.
        """
        snap = parse_pool(
            [
                pool_entry(1, name="Real Player", position_id=WR, adp=40.0),
                pool_entry(2, name="Undrafted Guy", position_id=WR, adp=169.99),
                pool_entry(3, name="Other Guy", position_id=WR, adp=170.0),
            ],
            season=SEASON,
        )
        rows = [
            FakeSleeperRow("a", "Real Player", "WR", {"adp_ppr": 45.0}),
            FakeSleeperRow("b", "Undrafted Guy", "WR", {"adp_ppr": 60.0}),
            FakeSleeperRow("c", "Other Guy", "WR", {"adp_ppr": 65.0}),
        ]
        censored = compare_adp(snap, rows, flavor="adp_ppr")
        assert censored.n == 1
        raw = compare_adp(snap, rows, flavor="adp_ppr", informative_only=False)
        assert raw.n == 3

    def test_no_overlap_returns_an_empty_comparison_rather_than_raising(self) -> None:
        snap = self.snapshot()
        result = compare_adp(snap, [], flavor="adp_ppr")
        assert result.n == 0
        assert result.gaps == ()
        assert result.describe().startswith("ESPN vs sleeper")


# --------------------------------------------------------------------------------------
# Live reconciliation
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_live_pool_still_carries_the_blocks_this_module_reads() -> None:
    """The two blocks the module depends on, and the eight analysts who publish.

    Both are things ESPN could remove without telling anyone, and both fail silently: an
    absent `rankings` block simply yields an empty analyst screen.
    """
    from fantasy_quant.edges.market import fetch_snapshot
    from fantasy_quant.pipeline import client_from_env

    client = client_from_env()
    try:
        snap = fetch_snapshot(client, 2026, max_players=500)
    finally:
        client.close()

    assert len(snap.records) >= 400
    with_draft_rank = [r for r in snap.records.values() if "PPR" in r.draft_ranks]
    assert len(with_draft_rank) == len(snap.records), "draftRanksByRankType is universal"
    # And uncensored, unlike ADP: a strict total order over the whole pool.
    ranks = [r.draft_ranks["PPR"] for r in with_draft_rank]
    assert len(set(ranks)) == len(ranks)

    boards = snap.boards(("PPR",))
    assert len(boards) >= 200
    sources = {s for b in boards.values() for s in b.ranks}
    assert set(PUBLISHING_SOURCES) <= sources
    assert CLAY_SOURCE in sources


@pytest.mark.network
def test_espn_standard_consensus_is_clay() -> None:
    """The strongest evidence that Clay's board is the one ESPN's game runs on.

    Every STANDARD `averageRank` in the live pool equals his STANDARD rank exactly, because
    he is the only analyst who publishes one.
    """
    from fantasy_quant.edges.market import fetch_snapshot
    from fantasy_quant.pipeline import client_from_env

    client = client_from_env()
    try:
        snap = fetch_snapshot(client, 2026)
    finally:
        client.close()

    compared = 0
    for rec in snap.records.values():
        for board in rec.boards:
            if board.rank_type != "STANDARD" or board.espn_average is None:
                continue
            clay = board.rank_of(CLAY_SOURCE)
            if clay is None:
                continue
            compared += 1
            assert board.espn_average == pytest.approx(clay)
    assert compared > 100
