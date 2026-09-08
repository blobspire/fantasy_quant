"""Where our valuation disagrees with the field, and where the field is about to move.

Everything below was measured on the live 2026 pool and the user's three real leagues on
2026-09-07. Where the measurement disagreed with the brief or with `docs/RESEARCH.md`, the
measurement wins and the disagreement is stated rather than smoothed over.

**The five screens, and which of them actually carry a signal.**

1. *Our VORP against the field's price.* Runs through `decide/valuation.market_disagreements`
   rather than a second ranker, so the informative-range censoring lives in one place. The
   one thing added here is a **fourth market metric that finally works**:
   `draftRanksByRankType`. `decide/valuation.py` documents it as "currently always None: the
   snapshot writer does not capture that block", and that is still true of the corpus -- but
   the block is right there in the live `kona_player_info` payload, and it is the *only*
   uncensored ESPN price we have. Measured on 1,036 players: `averageDraftPosition` is
   censored -- 845 of them sit in [169.0, 171.6] and only 191 carry an informative ADP --
   while `draftRanksByRankType.PPR.rank` is a strict total order over all 1,036, spanning
   1 to 2,627, correlating 0.956 with the informative half of ADP and -0.866 with percent
   rostered. So the ADP screen can see 191 players and the draft-rank screen can see the
   whole pool. Use it. `percent_started` is the brief's fifth signal and is deliberately
   *not* ranked on: it is a 0.951-Spearman copy of `percent_owned`. Its *ratio* to ownership
   is not a copy of anything, and rides along on every row -- see `PoolRecord.start_rate`.

2. *ESPN's analyst boards as a leading indicator.* `player.rankings` really does carry the
   per-analyst boards, and the brief's decoding is right as far as it goes. Four corrections
   from the live payload:

   * **Only eight of the twelve named analysts publish.** Present: 3 Karabell, 5 Cockcroft,
     6 Field Yates, 7 Mike Clay, 9 Bowen, 10 Dopp, 11 Moody, 12 Loza. Absent from every one
     of 1,036 entries: 1 Harris, 2 Berry, 8 Bell. A consensus computed over the brief's
     twelve would silently be a mean over eight.
   * **Ranks are POSITIONAL and are keyed by `slotId`.** Gibbs' rank 1 is RB1, not overall
     1. Ranks must be grouped by `(rankType, slotId)`; collapsing them mixes Travis Hunter's
     WR board with his DB board and manufactures a 73-rank "disagreement" out of nothing.
   * **ESPN's own `averageRank` is not the mean of the published analyst ranks, and it is
     not always available.** It is published for 226 of the 421 ranked players -- it stops
     at a positional depth -- and where both exist it disagrees with the mean of the eight
     for 49 of 216, by 1.2 ranks on average. Isiah Pacheco is the clean example: individual
     ranks 41/49/51/52/53/93/100/100, ESPN's `averageRank` 49.5, which is the mean of the
     five that have not moved. So `averageRank` lags the boards it averages. This module
     therefore computes the consensus itself from the published analysts and keeps ESPN's
     figure beside it as `espn_average`.
   * **ESPN's STANDARD "consensus" IS Clay.** For all 201 players carrying a STANDARD
     `averageRank`, it equals Clay's STANDARD rank exactly, and Clay is the only analyst who
     publishes a STANDARD board at all. That is the strongest available evidence for the
     brief's premise that Clay powers the ESPN game.

   **And then the premise does not survive contact with the data.** The brief asks whether
   a Clay-minus-consensus gap predicts subsequent percent-owned change. Measured: it does
   not. Against ESPN's own trailing ownership delta, over the 237 players with a season-long
   board of at least five analysts and a live ownership between 1% and 99.5%, the partial
   Spearman of the signed Clay gap on the signed ownership change -- controlling for percent
   rostered, consensus rank and position -- is **-0.091 (p = 0.16)**. The sign points the way
   the story wants (Clay bullish, ownership rising) and the effect is indistinguishable from
   zero. Read off the *week-1* board instead of the season board the same statistic comes
   back **+0.141 (p = 0.022)** -- significant, and pointing the other way. A signal whose sign
   depends on which of two near-identical boards you happened to read is not a signal.

   The quantitative version of that, which is the one to keep: run the test over all eight
   publishing analysts on both boards and **exactly one of the sixteen clears p < 0.05**,
   against 0.8 expected by chance, with the other fifteen scattered inside |r| <= 0.09. The
   one hit is that Clay week-board number. `analyst_direction_family` runs the family and
   `family_verdict` prints the count, because a single p-value pulled out of a scan of
   analysts is a story about the scan.

   **What carries a signal is DISPERSION, it is volatility rather than direction, and it is
   half the size the first version of this module reported.** The partial Spearman of the
   analysts' rank spread on the *absolute* ownership change is **+0.187 (p = 0.004,
   n = 248)** on the season board. It was reported as +0.328 (p = 1e-7) until the controls
   were made able to do their job: the confound this screen exists to remove is that ESPN's
   trailing ownership change is hump-shaped in ownership -- flat at both ends, peaking in
   the middle third -- and so is analyst dispersion, and a regression on the *rank* of
   ownership is a straight line, which removes none of a hump. Forty percent of the headline
   was the confound the controls were named after. See `CONTROL_DEGREE`.

   Two further corrections to how that number was sold:

   * **The week-1 board does not survive.** It was cited as the robustness check -- "same
     sign, same story, unlike the gap" -- at +0.161 (p = 0.009). With the hump removed it is
     **+0.065 (p = 0.30)**. The season board is the only board this holds on, and
     `screen_league` now prints both so the qualification travels with the claim.
   * **It is marginal, and marginal in a way a single quoted p-value hides.** The pool was
     pulled twice twenty minutes apart on 2026-09-07: 249 of 1,036 players came back with a
     different `percentChange` and 274 with a different `percentOwned`, and the statistic
     moved from +0.187 (p = 0.004) to +0.142 (p = 0.029) -- inside one standard error
     (~0.064), and across the line most people draw. Reporting "p = 1.2e-07" to two
     significant figures, as the first version did, described a precision that survives
     neither the specification nor the next pull. Two significant figures on the correlation
     and one on the p-value is as much as this measurement can carry.
   * **It is not monotone and it is small.** Tercile means 0.093 / 0.213 / 0.227 look like a
     gradient and are not one: top-minus-middle is **+0.014 +/- 0.048** (z = 0.3). What the
     data supports is that the tightest third is quieter than the other two, by **0.134
     percentage points of roster rate per ESPN period**. That is a real ordering and a
     trivial move, and `PredictiveCheck.effect_size` prints it beside the correlation so the
     unitless number is never read alone.

   And the whole thing is **contemporaneous, not predictive** -- signal and outcome are read
   off the same capture -- which is why `PredictiveCheck.usable` is False for it however
   small the p-value gets. See screen 3 for why a real held-out test is impossible today, and
   `predicts_ownership_change` for the harness that runs the moment it is not.

3. *Ownership momentum from our own captured history.* **There is exactly one 2026 capture.**
   `corpus.load_ownership_history((2026,))` returns 1,036 rows at a single `captured_at`, so
   percent-owned velocity has a denominator of zero and does not exist yet. Worse, it cannot
   be backfilled: the two captures we hold of the 2024 season, 21 months apart, are
   **byte-identical** in every ownership column (1,096 players, mean absolute difference
   0.0, correlation 1.0). ESPN freezes a finished season's ownership block, so the only way
   this series ever exists is forward from here. `corpus_depth` says so out loud and
   `ownership_momentum` returns `velocity=None` rather than dividing by a zero span.
   Until then the only trend available is ESPN's own trailing `percentChange`, which is
   populated for 397 of 1,036 players and is used here as an explicitly labelled proxy.

4. *Cross-source ADP.* Real, structured, and the largest effect in this module. Joining
   ESPN's pool to Sleeper's season ADP (531 of 554 Sleeper rows join; see `crosswalk`) and
   comparing rank for rank over the 161 players with an informative ESPN ADP:
   **ESPN's ADP drafts quarterbacks about 12 ranks earlier than Sleeper does (t = -6.0,
   n = 26) and running backs about 5.5 ranks later (t = +3.4, n = 52)**, with WR and TE
   indistinguishable. Spearman between the two boards is 0.962 overall, so this is a clean
   positional shift rather than general noise, and it survives the three artifacts that
   could have produced it: symmetric selection on both platforms (-13.6, t = -6.0), each
   ESPN ADP band taken separately (t = -3.1 / -3.6 / -2.8), and 5,000 permutations of the
   position labels (95th percentile of the largest |t| is 2.3 against an observed 6.0).

   Three caveats, of which the second was missing and is the one that changes the sentence.
   Sleeper's ADP is drawn from Sleeper's own default league shape and user base, so part of
   the gap is format rather than mispricing. **The finding is about ESPN's ADP column and not
   about ESPN**: run the same comparison against `draftRanksByRankType`, ESPN's other and
   uncensored published price, and the quarterback gap falls to -3.5 (t = -1.3) -- ESPN's own
   two prices disagree about quarterbacks by more than half the effect. And the by-position
   rows sum to zero by construction, so "QB early" and "RB late" are one rotation reported
   twice. All three of the user's leagues have already drafted, which makes this a
   draft-season edge being reported after the draft.

5. *A per-league availability board.* The first four screens are opinions; this is the one
   that produces actions, because a player we rate 40 ranks above the field is worth exactly
   nothing if a rival already has him. Availability is taken from the league's own rosters,
   never from `percentOwned` -- ESPN's global roster rate says what the population did, not
   what your twelve managers did, and the two genuinely disagree. Live check: Justice Hill
   (18.2% rostered globally) is on a roster in Wine Wednesday and free in the other two, and
   Kayshon Boutte (14.3%) is rostered in Wine Wednesday and Type shi and free in Blacksburg.
   Each appears on exactly the boards where he can actually be claimed.

   **This board is ordered by points, not by ranks, and that is a correction.** The rank gap
   it screens on is mechanically larger the deeper you go -- |percentile_delta| correlates
   +0.31 with our own positional rank and -0.40 with VORP, and the free agents this screen
   selects average +2.9 rest-of-season VORP against +84.2 for the rostered players it drops.
   Sorting the available slice by the gap therefore sorts it by depth, and it did: the first
   version put Kyle Juszczyk, a fullback worth 0.13 points a week, at the top of the biggest
   league's board, above a back worth +37. `MIN_CLAIM_VORP` is the floor and the ordering is
   by what the claim adds; the gap stays on the row as the reason he is on the board at all.

**What the first live run of screen 1 printed, and what it took to stop it.** Two filters in
`field_disagreements` are there because the board was useless without them, not because they
are tidy. The projection layer hands every deep-bench body the same token 1.1 points for the
season, so 42 receivers in one league share a VORP of -73.63 and their rank inside that block
is a player id -- the first board's three biggest "buys" were Andre Baccellia, Britain Covey
and JuJu Smith-Schuster. And the projection universe carries 76 quarterbacks against a solved
demand of 23, so "our QB39" sounds mid-pack and is worth -212 points. `MAX_VORP_TIE` removes
the first and `_relevant` removes the second, by asking the two directions different
questions: a buy has to be someone we would roster (`vorp > 0`, inside the position's depth)
and a sell has to be someone THEY would roster (inside the field's own replacement rank).

**League-specificity.** A rank type is chosen per league from that league's own scoring
function (`rank_type_for`), because ESPN publishes PPR and STANDARD boards and nothing in
between: Wine Wednesday reads the PPR board, and the two half-PPR leagues read the mean of
the two. Nothing here caches a league-independent player price.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ..core import DST, QB, RB, TE, WR, K, LeagueContext
from ..corpus import DEFAULT_ROOT, CorpusError, load_ownership_history
from ..decide.valuation import (
    MARKET_INFORMATIVE_RANGE,
    POSITION_ABBREV,
    MarketEdge,
    MarketQuote,
    PlayerValue,
    ValuationReport,
    market_disagreements,
)
from ..espn.endpoints import league_default_url

if TYPE_CHECKING:  # pragma: no cover - only the type checker needs these
    from ..data.sleeper import SleeperProjection
    from ..espn.client import EspnClient
    from ..pipeline import LeagueSim

log = logging.getLogger(__name__)


class MarketError(ValueError):
    """The market payload or corpus as given cannot support a screen."""


# --------------------------------------------------------------------------------------
# ESPN's analyst boards
# --------------------------------------------------------------------------------------

#: `rankSourceId` -> analyst, as documented in the brief. Kept whole because the ids are
#: stable and a source that starts publishing should be picked up by name, not by number.
RANK_SOURCES: Mapping[int, str] = MappingProxyType(
    {
        0: "consensus",
        1: "Harris",
        2: "Berry",
        3: "Karabell",
        5: "Cockcroft",
        6: "Field Yates",
        7: "Mike Clay",
        8: "Bell",
        9: "Bowen",
        10: "Dopp",
        11: "Moody",
        12: "Loza",
    }
)

#: The consensus pseudo-source. Its entries carry `averageRank` and a `rank` of 0.
CONSENSUS_SOURCE = 0

#: Mike Clay. His projections drive the ESPN game itself, which is why his board is the one
#: worth isolating -- and, measured, why ESPN's STANDARD "consensus" is literally his board.
CLAY_SOURCE = 7

#: Measured over all 1,036 entries of the live 2026 pool: these eight and only these eight
#: publish. Harris (1), Berry (2) and Bell (8) appear in the brief and in no payload.
PUBLISHING_SOURCES: tuple[int, ...] = (3, 5, 6, 7, 9, 10, 11, 12)

#: Ranking `slotId` -> `defaultPositionId`. The two id spaces collide (slot 4 is WR, position
#: 4 is TE), and the boards are keyed by slot, so the translation has to be explicit.
RANK_SLOT_POSITION: Mapping[int, int] = MappingProxyType(
    {0: QB, 2: RB, 4: WR, 6: TE, 16: DST, 17: K}
)

#: Which `rankings` block to read. ESPN publishes "0" (season-long) and "1" (the current
#: week), and they are NOT interchangeable even when they look alike. Measured on the live
#: pool: dispersion-vs-ownership-volatility comes out +0.328 (p = 1e-7) on the season board
#: and +0.161 (p = 0.009) on the week board, and the Clay-gap direction test flips SIGN
#: between them (-0.091, p = 0.16 season; +0.141, p = 0.022 week). Reading whichever board
#: happens to be widest silently averages the two. The season board is the default because it
#: is the stable one and it is what a draft-time or trade-time price is set against.
SEASON_BOARD = 0

#: Analysts needed before a board is worth a consensus. Eight publish; below five the
#: spread is measuring who bothered rather than how much they disagree.
MIN_ANALYSTS = 5

#: Points per reception at or above which a league reads ESPN's PPR board, and at or below
#: which it reads STANDARD. Between them (half PPR, which two of the three leagues are) both
#: boards are read and averaged, because ESPN publishes nothing in between.
PPR_THRESHOLD = 0.75
STANDARD_THRESHOLD = 0.25

#: ESPN's receptions statId, used only to ask a league's own scorer what a catch is worth.
RECEPTION_STAT = "53"


@dataclass(frozen=True, slots=True)
class AnalystBoard:
    """Every published analyst rank for one player at one position, in one scoring format.

    Scoped to a single `(scoring_period, rank_type, slot_id)` on purpose. A player eligible
    at two slots is ranked twice by ESPN -- Travis Hunter carries a WR board and a DB board
    -- and pooling them reports a 73-rank analyst disagreement that is really two different
    questions being answered correctly.

    `consensus` is computed here from the published ranks rather than read from ESPN's
    `averageRank`, which is missing for 195 of 421 ranked players and lags the individual
    boards where it is present. `espn_average` keeps ESPN's own figure so the lag itself can
    be looked at; see `consensus_lag`.
    """

    player_id: int
    name: str
    position_id: int
    slot_id: int
    rank_type: str
    scoring_period: int
    #: rankSourceId -> that analyst's positional rank. Never contains the consensus.
    ranks: Mapping[int, float]
    #: ESPN's published `averageRank`, when it publishes one for this player.
    espn_average: float | None = None

    @property
    def n_analysts(self) -> int:
        return len(self.ranks)

    @property
    def consensus(self) -> float:
        """Mean published analyst rank. The field's default board, computed honestly."""
        return float(np.mean(list(self.ranks.values()))) if self.ranks else math.nan

    @property
    def dispersion(self) -> float:
        """Sample SD of the analyst ranks. Zero for a one-analyst board, not undefined."""
        if len(self.ranks) < 2:
            return 0.0
        return float(np.std(list(self.ranks.values()), ddof=1))

    @property
    def consensus_lag(self) -> float | None:
        """`espn_average - consensus`. Positive means ESPN's published number is stale-high.

        Not a signal in itself -- it is the diagnostic that says whether the number your
        league-mates actually see in the app has caught up with the boards behind it.
        """
        if self.espn_average is None or not self.ranks:
            return None
        return float(self.espn_average) - self.consensus

    def rank_of(self, source_id: int) -> float | None:
        value = self.ranks.get(source_id)
        return None if value is None else float(value)

    def gap(self, source_id: int) -> float | None:
        """`rank(source) - mean(every OTHER published rank)`. Negative means more bullish.

        Leave-one-out rather than against the full consensus: an analyst is part of his own
        consensus, so including him shrinks his own disagreement by `1/n` and makes an
        eight-analyst board look 12.5% more agreeable than it is.
        """
        mine = self.ranks.get(source_id)
        if mine is None:
            return None
        others = [v for k, v in self.ranks.items() if k != source_id]
        if not others:
            return None
        return float(mine) - float(np.mean(others))

    @property
    def clay_gap(self) -> float | None:
        return self.gap(CLAY_SOURCE)


# --------------------------------------------------------------------------------------
# The live pool
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoolRecord:
    """One player as the live ESPN pool describes him: price, ownership and boards.

    Deliberately wider than `valuation.MarketQuote`, which carries the four columns the
    disagreement screen ranks on. `percent_started` and `percent_change` are not ranking
    metrics -- the first is conviction given ownership, the second is ESPN's own trailing
    delta -- so they ride here and are reported rather than ranked.
    """

    player_id: int
    name: str
    position_id: int
    pro_team_id: int
    percent_owned: float | None = None
    percent_started: float | None = None
    percent_change: float | None = None
    adp: float | None = None
    adp_percent_change: float | None = None
    auction_value: float | None = None
    auction_value_change: float | None = None
    #: rankType -> ESPN's consensus overall draft rank. Uncensored, unlike ADP.
    draft_ranks: Mapping[str, float] = field(default_factory=dict)
    injury_status: str | None = None
    boards: tuple[AnalystBoard, ...] = ()

    @property
    def start_rate(self) -> float | None:
        """`percent_started / percent_owned` -- conviction among the managers who have him.

        A player rostered by 90% and started by 20% is a handcuff; one rostered by 30% and
        started by 28% is a starter the field has not finished discovering. The two look
        identical on a percent-owned ranking.
        """
        if not self.percent_owned or self.percent_owned <= 0:
            return None
        return float(self.percent_started or 0.0) / float(self.percent_owned)

    def draft_rank(self, rank_types: Sequence[str]) -> float | None:
        """Mean draft rank over the requested boards, for a league between two formats."""
        values = [self.draft_ranks[rt] for rt in rank_types if rt in self.draft_ranks]
        return float(np.mean(values)) if values else None

    def board(
        self, rank_types: Sequence[str], *, scoring_period: int | None = SEASON_BOARD
    ) -> AnalystBoard | None:
        """The widest board for the requested formats and scoring period.

        "Widest" rather than "first" because a player ranked at two slots has two boards and
        the one every analyst filled in is the one carrying information about him. The
        scoring period is filtered rather than pooled -- see `SEASON_BOARD` for the two
        measurements that show pooling them changes the answer and, in one case, its sign.
        """
        wanted = set(rank_types)
        candidates = [
            b
            for b in self.boards
            if b.rank_type in wanted
            and (scoring_period is None or b.scoring_period == scoring_period)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda b: (b.n_analysts, -b.slot_id))

    def quote(self, rank_types: Sequence[str]) -> MarketQuote:
        return MarketQuote(
            player_id=self.player_id,
            adp=self.adp,
            percent_owned=self.percent_owned,
            auction_value=self.auction_value,
            draft_rank=self.draft_rank(rank_types),
        )


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """The whole live pool, parsed once and reused by every screen in every league.

    One pull serves all three leagues: ownership, ADP, auction value and the analyst boards
    are league-independent *inputs*. What is league-specific is which board is read and what
    the player is worth, and both of those are applied downstream.
    """

    season: int
    captured_at: dt.datetime
    variant: str
    records: Mapping[int, PoolRecord]

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.records))

    def record(self, player_id: int) -> PoolRecord | None:
        return self.records.get(int(player_id))

    def quotes(self, rank_types: Sequence[str]) -> dict[int, MarketQuote]:
        return {pid: r.quote(rank_types) for pid, r in self.records.items()}

    def boards(
        self,
        rank_types: Sequence[str],
        *,
        min_analysts: int = MIN_ANALYSTS,
        scoring_period: int | None = SEASON_BOARD,
    ) -> dict[int, AnalystBoard]:
        """player_id -> the board to read, dropping the ones too thin to have a consensus."""
        out: dict[int, AnalystBoard] = {}
        for pid, rec in self.records.items():
            b = rec.board(rank_types, scoring_period=scoring_period)
            if b is not None and b.n_analysts >= min_analysts:
                out[pid] = b
        return out


def _f(value: Any) -> float | None:
    """A float, or None. ESPN sends nulls, and `0` is a real auction value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_boards(player: Mapping[str, Any], *, player_id: int, name: str, position_id: int):
    """`player.rankings` -> one `AnalystBoard` per (period, rankType, slot).

    The consensus entry is pulled out into `espn_average` rather than being counted as a
    ninth analyst; leaving it in would double-weight the mean of the other eight.
    """
    raw = player.get("rankings") or {}
    if not isinstance(raw, Mapping):
        return ()
    grouped: dict[tuple[int, str, int], dict[str, Any]] = {}
    for period, entries in raw.items():
        try:
            scoring_period = int(period)
        except (TypeError, ValueError):
            continue
        for entry in entries or ():
            rank_type = str(entry.get("rankType") or "")
            slot = entry.get("slotId")
            if not rank_type or slot is None:
                continue
            key = (scoring_period, rank_type, int(slot))
            bucket = grouped.setdefault(key, {"ranks": {}, "average": None})
            source = entry.get("rankSourceId")
            if source == CONSENSUS_SOURCE:
                bucket["average"] = _f(entry.get("averageRank"))
                continue
            value = _f(entry.get("rank"))
            if source is None or value is None:
                continue
            bucket["ranks"][int(source)] = value
    return tuple(
        AnalystBoard(
            player_id=player_id,
            name=name,
            position_id=position_id,
            slot_id=slot,
            rank_type=rank_type,
            scoring_period=period,
            ranks=MappingProxyType(dict(sorted(bucket["ranks"].items()))),
            espn_average=bucket["average"],
        )
        for (period, rank_type, slot), bucket in sorted(grouped.items())
        if bucket["ranks"]
    )


def parse_pool(
    entries: Iterable[Mapping[str, Any]],
    *,
    season: int,
    variant: str = "ppr",
    captured_at: dt.datetime | None = None,
    scoring_period: int | None = None,
) -> MarketSnapshot:
    """`kona_player_info` entries -> a `MarketSnapshot`.

    `scoring_period` keeps only the boards published for that period; leave it None to keep
    every board and let `PoolRecord.board` pick the widest. The live payload carries period
    "0" (season-long) and "1" (the current week), and on 2026-09-07 they were near-identical,
    so the distinction has not mattered yet.
    """
    records: dict[int, PoolRecord] = {}
    for entry in entries:
        player = entry.get("player") or {}
        pid = entry.get("id") or player.get("id")
        if pid is None:
            continue
        pid = int(pid)
        name = str(player.get("fullName") or "")
        position_id = int(player.get("defaultPositionId") or 0)
        ownership = player.get("ownership") or {}
        draft_ranks = {
            str(rt): float(block["rank"])
            for rt, block in (player.get("draftRanksByRankType") or {}).items()
            if isinstance(block, Mapping) and block.get("rank") is not None
        }
        boards = parse_boards(player, player_id=pid, name=name, position_id=position_id)
        if scoring_period is not None:
            boards = tuple(b for b in boards if b.scoring_period == scoring_period)
        records[pid] = PoolRecord(
            player_id=pid,
            name=name,
            position_id=position_id,
            pro_team_id=int(player.get("proTeamId") or 0),
            percent_owned=_f(ownership.get("percentOwned")),
            percent_started=_f(ownership.get("percentStarted")),
            percent_change=_f(ownership.get("percentChange")),
            adp=_f(ownership.get("averageDraftPosition")),
            adp_percent_change=_f(ownership.get("averageDraftPositionPercentChange")),
            auction_value=_f(ownership.get("auctionValueAverage")),
            auction_value_change=_f(ownership.get("auctionValueAverageChange")),
            draft_ranks=MappingProxyType(draft_ranks),
            injury_status=player.get("injuryStatus"),
            boards=boards,
        )
    if not records:
        raise MarketError("the player pool payload carried no player entries")
    return MarketSnapshot(
        season=season,
        captured_at=captured_at or dt.datetime.now(),
        variant=variant,
        records=MappingProxyType(records),
    )


def fetch_snapshot(
    client: EspnClient,
    season: int,
    *,
    variant: str = "ppr",
    max_players: int | None = None,
) -> MarketSnapshot:
    """Pull the live pool once. Network.

    Uses the unauthenticated `leaguedefaults` pool rather than a league endpoint: ownership,
    ADP and the analyst boards are league-independent, so one pull serves every league and
    the result is comparable across them.
    """
    entries = client.player_pool(
        league_default_url(season, variant),
        params={"view": "kona_player_info"},
        max_players=max_players,
    )
    return parse_pool(entries, season=season, variant=variant)


def rank_type_for(scorer, *, position_id: int = WR) -> tuple[str, ...]:
    """Which of ESPN's published boards a league should read, from its own scoring.

    ESPN publishes PPR and STANDARD and nothing between them, so a half-PPR league gets both
    and averages. Asked of the league's own scorer rather than a settings string, because the
    thing that matters is what a catch is actually worth here.
    """
    per_reception = float(scorer({RECEPTION_STAT: 1.0}, position_id))
    if per_reception >= PPR_THRESHOLD:
        return ("PPR",)
    if per_reception <= STANDARD_THRESHOLD:
        return ("STANDARD",)
    return ("PPR", "STANDARD")


# --------------------------------------------------------------------------------------
# Screen 1: our value against the field's price
# --------------------------------------------------------------------------------------

#: Metrics the combined screen averages over. `draft_rank` leads because it is the only
#: uncensored one; ADP is kept because it is what a manager recognises.
DEFAULT_METRICS: tuple[str, ...] = ("draft_rank", "adp", "percent_owned", "auction_value")

#: How many players may share a rest-of-season VORP before the whole group is dropped.
#: Measured on Type shi (12-team half PPR): the WR list carries **42 players tied at
#: -73.63**, the TE list 17, the RB list 15 and the QB list 11, because ESPN projects a deep
#: bench body at the same token 1.1 points for the season and every one of them lands on the
#: identical baseline deficit. `valuation.player_values` breaks those ties on player id, so a
#: rank inside the block is a player id rather than an opinion -- and the first live run of
#: this screen duly reported Andre Baccellia, Britain Covey and JuJu Smith-Schuster as the
#: three biggest buys in the league. Three is generous: real ties above the tail are rare.
MAX_VORP_TIE = 3

#: Rounding at which two VORPs count as the same number, in points of rest-of-season value.
VORP_TIE_TOLERANCE = 1e-3

#: How far past a position's replacement rank the screen still cares. `N_q` is where the wire
#: starts, so twice it is already "nobody in this league will roster him"; beyond that both
#: our ranking and the field's are ordering noise about the same irrelevant players.
DEFAULT_DEPTH_MULTIPLE = 2.0


def drop_vorp_ties(
    values: Sequence[PlayerValue],
    *,
    max_tie: int = MAX_VORP_TIE,
    tolerance: float = VORP_TIE_TOLERANCE,
    playoff: bool = False,
) -> tuple[tuple[PlayerValue, ...], int]:
    """Remove the projection layer's tie blocks. Returns `(kept, dropped)`.

    Per position, because a tie only matters against the players it is being ranked against.
    See `MAX_VORP_TIE` for the measurement that makes this necessary rather than tidy.
    """
    key = (lambda v: v.playoff_vorp) if playoff else (lambda v: v.ros_vorp)
    counts: dict[tuple[int, float], int] = {}
    for v in values:
        counts[(v.position_id, round(key(v) / tolerance))] = (
            counts.get((v.position_id, round(key(v) / tolerance)), 0) + 1
        )
    kept = tuple(v for v in values if counts[(v.position_id, round(key(v) / tolerance))] <= max_tie)
    return kept, len(values) - len(kept)


@dataclass(frozen=True, slots=True)
class MetricRank:
    """One player's standing on one market metric, and how deep that metric's board goes."""

    our_rank: int
    market_rank: float
    market_value: float
    #: Players ranked against each other on this metric at this position.
    universe: int

    @property
    def delta(self) -> float:
        return self.market_rank - self.our_rank

    @property
    def percentile_delta(self) -> float:
        """`delta` as a share of the board it was measured on.

        The four metrics do not see the same players -- ADP prices 191 of 1,036 and draft
        rank prices all of them -- so a raw delta of 25 means something different on each.
        Averaging the raw deltas would quietly weight whichever metric happened to rank the
        most players; averaging these does not.
        """
        return self.delta / self.universe if self.universe else 0.0


@dataclass(frozen=True, slots=True)
class FieldDisagreement:
    """One player where our VORP and the field's price disagree, pooled over metrics.

    Sorted and signed on `percentile_delta`: positive means we rank him higher than the field
    does, in units of "share of the position's board". `rank_delta` is the mean raw rank gap
    and is there because it is the number a manager can picture.

    `agreement` is how many of the metrics that priced this player agreed on the sign.
    **It is not four independent confirmations and must not be read as one.** ESPN's four
    published prices are four views of one number: measured on the live pool, the four
    per-player rank deltas correlate 0.73 to 0.87 with each other (`draft_rank` against
    `percent_owned` is 0.88 at the source). Four out of four means "ESPN is internally
    consistent about him", which is the usual case, and one out of four means the metrics
    are fighting, which is the interesting case. The number that carries information is
    `n_metrics`: a player only two metrics priced is a player the field barely prices at all.
    """

    player_id: int
    name: str
    position_id: int
    #: Our rank at this position among the players that survived the tie guard.
    our_rank: int
    rank_delta: float
    percentile_delta: float
    per_metric: Mapping[str, MetricRank]
    ros_vorp: float
    playoff_vorp: float
    #: Weeks `ros_vorp` was accumulated over, so the board can quote points per week.
    ros_weeks: int = 17
    percent_owned: float | None = None
    percent_started: float | None = None
    start_rate: float | None = None
    dispersion: float | None = None
    clay_gap: float | None = None
    available: bool = False

    @property
    def position(self) -> str:
        return POSITION_ABBREV.get(self.position_id, str(self.position_id))

    @property
    def is_buy(self) -> bool:
        return self.percentile_delta > 0

    @property
    def n_metrics(self) -> int:
        return len(self.per_metric)

    @property
    def agreement(self) -> int:
        """Metrics whose own delta shares the sign of the pooled one. NOT independent votes.

        See the class docstring: the four metrics' deltas correlate 0.73-0.87, so this counts
        how internally consistent one source is, not how many sources agree.
        """
        sign = 1.0 if self.percentile_delta >= 0 else -1.0
        return sum(1 for m in self.per_metric.values() if m.delta * sign > 0)

    @property
    def vorp_per_week(self) -> float:
        """Rest-of-season VORP as points a week, which is the number that reads honestly.

        A rank delta of +22 sounds like a finding and `+2.2 VORP` sounds like a rounding
        error; they are the same player. Dividing by the horizon the VORP was accumulated
        over is what turns the second number into the per-week edge a claim actually buys.
        """
        return self.ros_vorp / max(1, self.ros_weeks)


def _relevant(
    pooled_delta: float,
    vorp: float,
    our_rank: int,
    best_market_rank: float,
    baseline_rank: float,
    depth_multiple: float,
) -> bool:
    """Whether a disagreement about this player is a disagreement worth having.

    The two directions have different relevance tests, and collapsing them into one is how
    the first live board ended up recommending third-string quarterbacks.

    * **A buy has to be someone we would actually roster.** VORP is already measured against
      the `N_q`-th best player at the position, so `vorp <= 0` says in our own units that a
      freely available body is as good -- and no amount of the field disliking him changes
      that. He also has to sit inside `depth_multiple` times the position's replacement rank.
      Without this the QB board fills with backups: the projection universe carries 76
      quarterbacks against a solved demand of 23, so "our QB39" sounds mid-pack and is worth
      -212 points of rest-of-season value.
    * **A sell has to be someone THEY would roster.** If the field ranks him past its own
      replacement level too, then nobody is paying for him and there is nothing to sell.
    """
    limit = depth_multiple * baseline_rank
    if pooled_delta > 0:
        return vorp > 0 and our_rank <= limit
    return best_market_rank <= baseline_rank


def field_disagreements(
    values: Sequence[PlayerValue],
    quotes: Mapping[int, MarketQuote],
    *,
    positions: Iterable[int] | None = (QB, RB, WR, TE),
    metrics: Sequence[str] = DEFAULT_METRICS,
    playoff: bool = False,
    min_metrics: int = 1,
    limit: int | None = None,
    snapshot: MarketSnapshot | None = None,
    rank_types: Sequence[str] = ("PPR",),
    scoring_period: int | None = SEASON_BOARD,
    rostered: Iterable[int] | None = None,
    max_tie: int = MAX_VORP_TIE,
    depth: Mapping[int, float] | None = None,
    depth_multiple: float = DEFAULT_DEPTH_MULTIPLE,
) -> tuple[FieldDisagreement, ...]:
    """Rank our VORP against every available market metric, per position, and pool.

    **Per position, always.** `valuation.market_disagreements` says why in its own docstring
    and the live run confirms it: an all-positions ADP screen reports kickers and defenses as
    the biggest buys in every league, because ADP encodes "take your kicker last" rather than
    a view on value. Screening WRs against WRs asks the question that has an answer.

    **Two filters that the first live run proved are not optional.** The tie guard
    (`MAX_VORP_TIE`) removes the projection layer's dead tail, where dozens of players carry
    an identical VORP and their rank is a player id. The relevance filter (`_relevant`) then
    asks each direction its own question -- a buy has to be someone we would roster, a sell
    someone the field would -- and needs `depth=model.baseline_ranks()` to do it; pass it, or
    the screen runs unfiltered. Without them the board is a list of fifth receivers nobody
    will ever roster, which is exactly what it printed.

    Each metric is ranked by `market_disagreements` over its own informative subset, so the
    censoring rules live in one place and this function only pools -- in percentile units,
    because those subsets are different sizes. `min_metrics` is the guard for callers who
    want agreement rather than coverage.
    """
    wanted = None if positions is None else sorted(set(positions))
    owned = {int(p) for p in (rostered or ())}
    screened, dropped = drop_vorp_ties(values, max_tie=max_tie, playoff=playoff)
    if dropped:
        log.info("dropped %d players sitting in a VORP tie block of more than %d", dropped, max_tie)
    meta: dict[int, PlayerValue] = {v.player_id: v for v in screened}

    ours_by_position: dict[int, dict[int, int]] = {}
    for pos in {v.position_id for v in screened}:
        vorp = (lambda v: v.playoff_vorp) if playoff else (lambda v: v.ros_vorp)
        ranked = sorted(
            (v for v in screened if v.position_id == pos), key=lambda v: (-vorp(v), v.player_id)
        )
        ours_by_position[pos] = {v.player_id: r for r, v in enumerate(ranked, start=1)}

    per_player: dict[int, dict[str, MetricRank]] = {}
    groups = [None] if wanted is None else wanted
    for metric in metrics:
        for group in groups:
            edges: tuple[MarketEdge, ...] = market_disagreements(
                screened,
                quotes,
                metric=metric,
                playoff=playoff,
                positions=None if group is None else (group,),
            )
            universe = len(edges)
            for e in edges:
                per_player.setdefault(e.player_id, {})[metric] = MetricRank(
                    our_rank=e.our_rank,
                    market_rank=e.market_rank,
                    market_value=e.market_value,
                    universe=universe,
                )

    out: list[FieldDisagreement] = []
    for pid, seen in per_player.items():
        if len(seen) < min_metrics:
            continue
        value = meta.get(pid)
        if value is None:  # pragma: no cover - market_disagreements only emits known players
            continue
        our_rank = ours_by_position[value.position_id][pid]
        pooled = float(np.mean([m.percentile_delta for m in seen.values()]))
        vorp = value.playoff_vorp if playoff else value.ros_vorp
        if depth is not None:
            baseline = float(depth.get(value.position_id, 0.0))
            best_market = min(m.market_rank for m in seen.values())
            if baseline > 0 and not _relevant(
                pooled, vorp, our_rank, best_market, baseline, depth_multiple
            ):
                continue
        rec = snapshot.record(pid) if snapshot is not None else None
        board = rec.board(rank_types, scoring_period=scoring_period) if rec is not None else None
        out.append(
            FieldDisagreement(
                player_id=pid,
                name=value.name,
                position_id=value.position_id,
                our_rank=our_rank,
                rank_delta=float(np.mean([m.delta for m in seen.values()])),
                percentile_delta=pooled,
                per_metric=MappingProxyType(dict(sorted(seen.items()))),
                ros_vorp=value.ros_vorp,
                playoff_vorp=value.playoff_vorp,
                ros_weeks=value.ros_weeks,
                percent_owned=rec.percent_owned if rec else None,
                percent_started=rec.percent_started if rec else None,
                start_rate=rec.start_rate if rec else None,
                dispersion=board.dispersion if board else None,
                clay_gap=board.clay_gap if board else None,
                available=pid not in owned,
            )
        )
    out.sort(key=lambda d: (-abs(d.percentile_delta), d.our_rank, d.player_id))
    return tuple(out[:limit] if limit else out)


# --------------------------------------------------------------------------------------
# Screen 2: the analyst boards
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnalystSignal:
    """One analyst's disagreement with his peers about one player, with what happened next.

    `gap` is signed the way ranks are: negative means the analyst has him *better* than the
    others do. `dispersion` is the whole board's spread and is the half of this that
    measures out as informative -- see the module docstring.
    """

    player_id: int
    name: str
    position_id: int
    source_id: int
    rank: float
    consensus: float
    gap: float
    dispersion: float
    n_analysts: int
    espn_average: float | None = None
    consensus_lag: float | None = None
    percent_owned: float | None = None
    percent_change: float | None = None

    @property
    def source(self) -> str:
        return RANK_SOURCES.get(self.source_id, str(self.source_id))

    @property
    def position(self) -> str:
        return POSITION_ABBREV.get(self.position_id, str(self.position_id))

    @property
    def is_bullish(self) -> bool:
        return self.gap < 0


def analyst_signals(
    snapshot: MarketSnapshot,
    *,
    source_id: int = CLAY_SOURCE,
    rank_types: Sequence[str] = ("PPR",),
    scoring_period: int | None = SEASON_BOARD,
    min_analysts: int = MIN_ANALYSTS,
    min_percent_owned: float | None = None,
    limit: int | None = None,
) -> tuple[AnalystSignal, ...]:
    """Every player where the named analyst differs from the rest of the board.

    Sorted by the size of the disagreement, most bullish first, so both tails are one slice
    away. `min_percent_owned` exists because the deep pool is full of boards where the
    "disagreement" is two analysts ranking a fullback.
    """
    out: list[AnalystSignal] = []
    boards = snapshot.boards(rank_types, min_analysts=min_analysts, scoring_period=scoring_period)
    for pid, board in boards.items():
        gap = board.gap(source_id)
        rank = board.rank_of(source_id)
        if gap is None or rank is None:
            continue
        rec = snapshot.record(pid)
        owned = rec.percent_owned if rec else None
        if min_percent_owned is not None and (owned is None or owned < min_percent_owned):
            continue
        out.append(
            AnalystSignal(
                player_id=pid,
                name=board.name,
                position_id=RANK_SLOT_POSITION.get(board.slot_id, board.position_id),
                source_id=source_id,
                rank=rank,
                consensus=board.consensus,
                gap=gap,
                dispersion=board.dispersion,
                n_analysts=board.n_analysts,
                espn_average=board.espn_average,
                consensus_lag=board.consensus_lag,
                percent_owned=owned,
                percent_change=rec.percent_change if rec else None,
            )
        )
    out.sort(key=lambda s: (s.gap, s.player_id))
    return tuple(out[:limit] if limit else out)


@dataclass(frozen=True, slots=True)
class PredictiveCheck:
    """The result of asking whether a signal moves the field, and how honestly it was asked.

    `held_out` is the field that decides what this is worth. True means the association was
    estimated on one set of captures and re-measured on captures the estimate never saw --
    which is the only version of this question that answers it. False means the signal and
    the outcome were read off the same capture, which can establish a correlation and cannot
    establish a lead.
    """

    question: str
    method: str
    n: int
    statistic: float
    p_value: float
    held_out: bool
    verdict: str
    note: str = ""
    #: Extra measurements worth printing beside the headline -- tercile means, controls.
    detail: Mapping[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Significant AND held out. A contemporaneous association is never `usable`.

        This property used to be `n > 0 and p < 0.05`, which made the dispersion result --
        signal and outcome read off the same capture -- come back True, in flat contradiction
        of this class's own docstring. Anything filtering a board on `.usable` would have
        been acting on a correlation labelled two lines above as not a lead.
        """
        return self.n > 0 and self.p_value < 0.05 and self.held_out

    @property
    def effect_size(self) -> str:
        """The tercile contrast in the outcome's own units, or "" when there is not one.

        A rank correlation is unitless and reads as big; the thing it is a correlation *of*
        is often not. Printing both is the difference between "p = 1e-7" and "0.13 more
        percentage points of roster rate per period", which is the same finding.
        """
        gap = self.detail.get("tercile_top_minus_bottom")
        if gap is None:
            return ""
        se = self.detail.get("tercile_top_minus_bottom_se", 0.0)
        mid = self.detail.get("tercile_top_minus_middle")
        out = f"top vs bottom tercile {gap:+.3f} +/- {se:.3f}"
        if mid is not None:
            mid_se = self.detail.get("tercile_top_minus_middle_se", 0.0)
            out += f"; top vs MIDDLE {mid:+.3f} +/- {mid_se:.3f}"
        return out

    def describe(self) -> str:
        if self.n == 0:
            return f"{self.question}: {self.verdict} -- {self.note}"
        tag = "held out" if self.held_out else "CONTEMPORANEOUS, not predictive"
        line = (
            f"{self.question}: {self.method} = {self.statistic:+.3f} "
            f"(p = {self.p_value:.2g}, n = {self.n}, {tag}) -- {self.verdict}"
        )
        size = self.effect_size
        return f"{line}\n      effect size: {size}" if size else line


def _rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(values)


#: Degree of the rank polynomial each continuous control is expanded to before it is
#: residualised out. **One is not enough here, and that is measured, not stylistic.** The
#: confound this module has to remove is that ESPN's trailing ownership change is
#: hump-shaped in ownership -- it peaks in deciles 3-5 (mean |change| 0.04 / 0.08 / 0.21 /
#: 0.18 / 0.47 / 0.30 / 0.20 / 0.15 / 0.12 / 0.03 across the live sample) and so does
#: analyst dispersion (2.4 / 3.3 / 5.2 / 7.0 / 6.8 / 3.5 / 3.2 / 2.7 / 1.8 / 1.4). A
#: regression on the *rank* of ownership is a straight line and removes none of a hump, so
#: a linear partial correlation reports the shared hump as an association between the two.
#: Measured on the live pool: dispersion vs |ownership change| is +0.328 with linear
#: controls and +0.187 with cubic ones. Forty percent of the headline was the confound the
#: controls were named after.
CONTROL_DEGREE = 3


def _control_columns(control: Sequence[float], *, degree: int, categorical: bool, n: int):
    values = np.asarray(control, dtype=float)
    if values.size != n:
        raise MarketError(f"a control carries {values.size} values against {n} observations")
    if categorical:
        levels = np.unique(values)[1:]  # one level held out as the reference
        if levels.size == 0:
            return np.empty((n, 0))
        return np.column_stack([(values == lv).astype(float) for lv in levels])
    r = _rank(values)
    spread = float(r.std()) or 1.0
    r = (r - r.mean()) / spread
    return np.column_stack([r**d for d in range(1, max(1, degree) + 1)])


def partial_spearman(
    x: Sequence[float],
    y: Sequence[float],
    controls: Sequence[Sequence[float]] = (),
    *,
    degree: int = CONTROL_DEGREE,
    categorical: Sequence[int] = (),
) -> tuple[float, float, int]:
    """Spearman correlation of `x` and `y` with `controls` flexibly residualised out.

    Rank-transform everything, expand each control into a polynomial in its own rank (or
    into dummies, for the indices named in `categorical`), regress `x` and `y` on that
    design, correlate the residuals. The controls are not optional decoration and neither is
    the expansion -- see `CONTROL_DEGREE` for the measurement that says a straight line
    through the ownership rank leaves 40% of a confound behind.

    The p-value is a two-sided t on `n - 2 - k` degrees of freedom, where `k` is the rank of
    the control design. Handing the residuals to `pearsonr` instead -- which assumes
    `n - 2` -- quietly spends the controls for free and reports a p-value too small.

    `degree` is reduced automatically rather than blindly honoured: a design is never
    allowed more than one column per ten observations, because a control basis that can fit
    the noise removes the signal with it.
    """
    from scipy.stats import pearsonr
    from scipy.stats import t as student_t

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size:
        raise MarketError(f"partial_spearman got {xa.size} x values and {ya.size} y values")
    n = int(xa.size)
    if n < 4:
        return 0.0, 1.0, n

    cats = {int(i) for i in categorical}
    budget = max(1, n // 10)
    deg = max(1, int(degree))
    while True:
        design = np.column_stack(
            [
                np.ones((n, 1)),
                *(
                    _control_columns(c, degree=deg, categorical=i in cats, n=n)
                    for i, c in enumerate(controls)
                ),
            ]
        )
        if design.shape[1] - 1 <= budget or deg == 1:
            break
        deg -= 1

    rx, ry = _rank(xa), _rank(ya)
    scale = max(rx.std(), ry.std(), 1.0)
    rx = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    # A control that fully explains one of the two leaves residuals of pure floating-point
    # dust, and `pearsonr` on dust cheerfully returns 1.0. Compared against the scale of the
    # ranks themselves rather than against zero, because the residuals are never exactly it.
    if min(rx.std(), ry.std()) <= 1e-8 * scale:
        return 0.0, 1.0, n
    stat, _ = pearsonr(rx, ry)
    k = int(np.linalg.matrix_rank(design)) - 1
    df = n - 2 - k
    if df < 1:
        return float(stat), 1.0, n
    t = float(stat) * math.sqrt(df / max(1e-12, 1.0 - float(stat) ** 2))
    return float(stat), float(2.0 * student_t.sf(abs(t), df)), n


def dispersion_vs_volatility(
    snapshot: MarketSnapshot,
    *,
    rank_types: Sequence[str] = ("PPR",),
    scoring_period: int | None = SEASON_BOARD,
    min_analysts: int = MIN_ANALYSTS,
    owned_range: tuple[float, float] = (1.0, 99.5),
) -> PredictiveCheck:
    """Does analyst disagreement go with ownership actually moving? Contemporaneous, weak.

    The outcome is ESPN's own trailing `percentChange`, which is the only ownership delta
    that exists in a single capture -- so this establishes that the two travel together and
    cannot establish which came first. It is labelled `held_out=False` for exactly that
    reason, and `predicts_ownership_change` is the version that will settle it once the
    corpus has two captures of the boards.

    Controls are percent rostered, consensus rank and position slot, each expanded to
    `CONTROL_DEGREE` (slot into dummies) rather than entered as a straight line. That is the
    whole ball game: **both** dispersion and |ownership change| are hump-shaped in ownership,
    peaking together in the third to fifth decile, and a linear control cannot remove a hump.
    Live numbers on the season board: +0.328 (p = 1e-7) with a linear control, +0.187
    (p = 0.004) once the hump is actually removed. The second number is the honest one and is
    what this function returns; the first is carried in `note` so the gap is visible.

    **Expect the second decimal to move.** ESPN repopulates the ownership block continuously
    -- two pulls twenty minutes apart differed on `percentChange` for 249 of 1,036 players --
    and the statistic went +0.187 (p = 0.004) to +0.142 (p = 0.029) between them. That is
    inside one standard error and across the significance line, which is the useful way to
    hold this result: real enough to believe in the ordering, not stable enough to quote to
    three digits or to treat p < 0.05 as a property of the world rather than of the pull.

    Two more things a caller has to see before trading on it, both in `detail`:

    * **The effect is small in the units that matter.** Top-versus-bottom tercile of
      dispersion is 0.227 against 0.093 percentage points of roster rate per ESPN period --
      a difference of 0.13 points of ownership. That is a real ordering and a trivial move.
    * **It is not monotone.** Tercile 1 -> 2 is +0.014 +/- 0.048 (z = 0.3). The bottom third
      is quieter than the other two; the other two are indistinguishable from each other.

    And the week-1 board, which is the module's own robustness check, does **not** survive
    the flexible control: +0.161 (p = 0.009) linear, +0.071 (p = 0.26) cubic. Run this
    function with `scoring_period=1` and read the verdict rather than assuming the season
    board generalises -- `screen_league` runs both and prints both for that reason.
    """
    lo, hi = owned_range
    disp: list[float] = []
    change: list[float] = []
    owned: list[float] = []
    cons: list[float] = []
    slot: list[float] = []
    boards = snapshot.boards(rank_types, min_analysts=min_analysts, scoring_period=scoring_period)
    for pid, board in boards.items():
        rec = snapshot.record(pid)
        if rec is None or rec.percent_owned is None or rec.percent_change is None:
            continue
        if not (lo <= rec.percent_owned <= hi):
            continue
        disp.append(board.dispersion)
        change.append(abs(rec.percent_change))
        owned.append(rec.percent_owned)
        cons.append(board.consensus)
        slot.append(float(board.slot_id))
    question = "does analyst dispersion go with ownership moving"
    if len(disp) < 20:
        return PredictiveCheck(
            question=question,
            method="partial spearman",
            n=len(disp),
            statistic=0.0,
            p_value=1.0,
            held_out=False,
            verdict="insufficient data",
            note=f"only {len(disp)} players carry both a board and an ownership delta",
        )
    stat, p, n = partial_spearman(disp, change, [owned, cons, slot], categorical=(2,))
    linear, linear_p, _ = partial_spearman(disp, change, [owned, cons, slot], degree=1)
    detail = dict(_tercile_means(np.asarray(disp), np.asarray(change)))
    detail.update(_tercile_contrast(np.asarray(disp), np.asarray(change)))
    detail["linear_control_statistic"] = linear
    detail["linear_control_p"] = linear_p
    spread = detail.get("tercile_top_minus_bottom", 0.0)
    if p < 0.05 and stat > 0:
        verdict = (
            "dispersion tracks |ownership change|, contemporaneously and weakly: the loosest "
            f"third of boards moves {spread:+.2f} points of roster rate more per period than "
            "the tightest. Volatility, never direction, and never a lead"
        )
        if p > 0.005:
            verdict += "; and MARGINAL -- the same test on a pull 20 minutes later crossed 0.05"
    else:
        verdict = "no association survives the controls"
    return PredictiveCheck(
        question=question,
        method=(
            f"partial spearman (controls: percent owned, consensus rank, slot; "
            f"degree {CONTROL_DEGREE})"
        ),
        n=n,
        statistic=stat,
        p_value=p,
        held_out=False,
        verdict=verdict,
        note=(
            "outcome is ESPN's own trailing percentChange read from the same capture; a "
            f"linear-in-rank control would report {linear:+.3f} (p = {linear_p:.2g}), and the "
            "difference between the two is the ownership hump that a line cannot remove"
        ),
        detail=MappingProxyType(detail),
    )


def gap_vs_direction(
    snapshot: MarketSnapshot,
    *,
    source_id: int = CLAY_SOURCE,
    rank_types: Sequence[str] = ("PPR",),
    scoring_period: int | None = SEASON_BOARD,
    min_analysts: int = MIN_ANALYSTS,
    owned_range: tuple[float, float] = (1.0, 99.5),
) -> PredictiveCheck:
    """Does one analyst's disagreement go with ownership moving HIS way? Contemporaneous.

    This is the brief's actual hypothesis. A negative statistic supports it: the gap is
    negative when the analyst is bullish and the ownership change is positive when the field
    is adding. Measured on the live pool it comes back small and insignificant.

    **The full family, because one test out of a family is not evidence.** Run over all eight
    publishing analysts on both boards -- sixteen tests -- exactly one clears p < 0.05
    (Clay on the week board, +0.141, p = 0.022) against 0.8 expected by chance, and the
    other fifteen scatter around zero with |r| <= 0.09. That is what a null family looks
    like. Scanning analysts here to find "the one who leads the field" will find one every
    time; `analyst_direction_family` runs the whole family so the finding can be read
    against its own multiplicity instead of on its own.
    """
    lo, hi = owned_range
    gaps: list[float] = []
    change: list[float] = []
    owned: list[float] = []
    cons: list[float] = []
    boards = snapshot.boards(rank_types, min_analysts=min_analysts, scoring_period=scoring_period)
    for pid, board in boards.items():
        rec = snapshot.record(pid)
        gap = board.gap(source_id)
        if rec is None or gap is None or rec.percent_owned is None or rec.percent_change is None:
            continue
        if not (lo <= rec.percent_owned <= hi):
            continue
        gaps.append(gap)
        change.append(rec.percent_change)
        owned.append(rec.percent_owned)
        cons.append(board.consensus)
    question = f"does a {RANK_SOURCES.get(source_id, source_id)} gap go with ownership moving"
    if len(gaps) < 20:
        return PredictiveCheck(
            question=question,
            method="partial spearman",
            n=len(gaps),
            statistic=0.0,
            p_value=1.0,
            held_out=False,
            verdict="insufficient data",
            note=f"only {len(gaps)} players carry both a board and an ownership delta",
        )
    stat, p, n = partial_spearman(gaps, change, [owned, cons])
    verdict = (
        "the gap leads ownership in the expected direction"
        if p < 0.05 and stat < 0
        else "no directional signal: the gap does not say which way the field moves"
    )
    return PredictiveCheck(
        question=question,
        method=(
            f"partial spearman (controls: percent owned, consensus rank; degree {CONTROL_DEGREE})"
        ),
        n=n,
        statistic=stat,
        p_value=p,
        held_out=False,
        verdict=verdict,
        note=(
            "outcome is ESPN's own trailing percentChange read from the same capture; one "
            "analyst on one board is one test out of a family of sixteen -- see "
            "analyst_direction_family"
        ),
    )


def analyst_direction_family(
    snapshot: MarketSnapshot,
    *,
    rank_types: Sequence[str] = ("PPR",),
    scoring_periods: Sequence[int] = (SEASON_BOARD, 1),
    sources: Sequence[int] = PUBLISHING_SOURCES,
) -> tuple[tuple[int, int, PredictiveCheck], ...]:
    """Every (analyst, board) direction test, so a hit can be read against its multiplicity.

    Returns `(source_id, scoring_period, check)` triples. The point is not to find the
    analyst who leads the field; it is that scanning eight analysts across two boards will
    always hand you one p < 0.05 and this is the denominator that says so. Live, on
    2026-09-07: one hit in sixteen, expectation 0.8.
    """
    out: list[tuple[int, int, PredictiveCheck]] = []
    for period in scoring_periods:
        for source in sources:
            out.append(
                (
                    int(source),
                    int(period),
                    gap_vs_direction(
                        snapshot,
                        source_id=int(source),
                        rank_types=rank_types,
                        scoring_period=int(period),
                    ),
                )
            )
    return tuple(out)


def family_verdict(family: Sequence[tuple[int, int, PredictiveCheck]], alpha: float = 0.05) -> str:
    """One line saying how many of a family of tests hit, against how many were expected."""
    tested = [(s, p, c) for s, p, c in family if c.n > 0]
    if not tested:
        return "no analyst-direction test had enough data to run"
    hits = [(s, p, c) for s, p, c in tested if c.p_value < alpha]
    names = ", ".join(f"{RANK_SOURCES.get(s, s)} period {p} {c.statistic:+.3f}" for s, p, c in hits)
    tail = f" ({names})" if names else ""
    return (
        f"{len(hits)} of {len(tested)} analyst x board direction tests clear p < {alpha:g}, "
        f"against {alpha * len(tested):.1f} expected by chance{tail}"
    )


def _tercile_means(signal: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    """Mean outcome in each third of the signal. The sanity check on a rank correlation."""
    if signal.size < 6:
        return {}
    cuts = np.quantile(signal, [1 / 3, 2 / 3])
    bins = np.digitize(signal, cuts)
    return {
        f"tercile_{i}_mean_outcome": float(outcome[bins == i].mean())
        for i in range(3)
        if np.any(bins == i)
    }


def _tercile_contrast(signal: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    """Tercile differences WITH their standard errors, which is what makes them readable.

    A rank correlation says an ordering exists; it does not say how big it is, and it
    certainly does not say the ordering is monotone. Live: top-minus-bottom is +0.134 points
    of roster rate (z = 3.6) and top-minus-middle is +0.014 (z = 0.3). Reporting "monotone
    across terciles, 2.4x" off the three means alone would have been a story about the
    second number, which is noise.
    """
    if signal.size < 12:
        return {}
    cuts = np.quantile(signal, [1 / 3, 2 / 3])
    bins = np.digitize(signal, cuts)
    groups = [outcome[bins == i] for i in range(3)]
    if any(g.size < 2 for g in groups):
        return {}
    means = [float(g.mean()) for g in groups]
    ses = [float(g.std(ddof=1) / math.sqrt(g.size)) for g in groups]
    return {
        "tercile_top_minus_bottom": means[2] - means[0],
        "tercile_top_minus_bottom_se": math.hypot(ses[2], ses[0]),
        "tercile_top_minus_middle": means[2] - means[1],
        "tercile_top_minus_middle_se": math.hypot(ses[2], ses[1]),
    }


def predicts_ownership_change(
    history: pl.DataFrame,
    signal: Mapping[int, float] | Mapping[dt.datetime, Mapping[int, float]],
    *,
    min_gap_days: float = 1.0,
    min_players: int = 30,
    question: str = "does the signal predict subsequent percent-owned change",
) -> PredictiveCheck:
    """The real test: fit on early capture pairs, re-measure on a capture pair held out.

    Not runnable on today's corpus and written anyway, because the difference between "we
    measured it and it does not predict" and "we could not measure it" is the whole value of
    this function. The 2026 corpus holds **one** capture, so there are zero capture pairs and
    this returns `verdict="insufficient captures"` with `n=0`.

    Two limitations worth stating rather than discovering later:

    * The corpus does not store `player.rankings`. `snapshot._flatten` captures the ownership
      block and never the boards, so a per-capture analyst signal does not exist historically
      either. Pass a `{player_id: value}` mapping and it is treated as measured at the first
      capture of each pair -- correct only if the boards move slowly. Pass a
      `{captured_at: {player_id: value}}` mapping and each pair uses its own signal, which is
      what should happen once the capture is widened.
    * A held-out verdict needs at least two disjoint capture pairs, so three captures. With
      exactly two the association is reported with `held_out=False`.
    """
    from scipy.stats import spearmanr

    if history.is_empty():
        return PredictiveCheck(
            question=question,
            method="spearman on capture pairs",
            n=0,
            statistic=0.0,
            p_value=1.0,
            held_out=False,
            verdict="insufficient captures",
            note="the ownership history is empty",
        )
    times = sorted(history["captured_at"].unique().to_list())
    usable_pairs: list[tuple[dt.datetime, dt.datetime]] = []
    for earlier, later in zip(times, times[1:], strict=False):
        if (later - earlier).total_seconds() / 86400.0 >= min_gap_days:
            usable_pairs.append((earlier, later))
    if not usable_pairs:
        return PredictiveCheck(
            question=question,
            method="spearman on capture pairs",
            n=0,
            statistic=0.0,
            p_value=1.0,
            held_out=False,
            verdict="insufficient captures",
            note=(
                f"{len(times)} distinct capture(s) in this history and no pair at least "
                f"{min_gap_days:g} day(s) apart; ownership velocity does not exist yet"
            ),
        )

    per_capture = bool(signal) and isinstance(next(iter(signal)), dt.datetime)

    def pair_rows(pair: tuple[dt.datetime, dt.datetime]) -> tuple[list[float], list[float]]:
        earlier, later = pair
        # Aligned on player id, not on row order: a player absent from one capture must drop
        # out of the pair rather than shift the series by one.
        left = history.filter(pl.col("captured_at") == earlier).select(["espn_id", "percent_owned"])
        right = history.filter(pl.col("captured_at") == later).select(
            ["espn_id", pl.col("percent_owned").alias("later")]
        )
        joined = left.join(right, on="espn_id", how="inner").drop_nulls()
        # `.get` rather than `[]`: a per-capture signal missing a capture is a thin sample,
        # which the overlap guard below reports, not a KeyError out of a screen.
        table = signal.get(earlier, {}) if per_capture else signal  # type: ignore[union-attr]
        xs: list[float] = []
        ys: list[float] = []
        for row in joined.iter_rows(named=True):
            value = table.get(int(row["espn_id"]))  # type: ignore[union-attr]
            if value is None:
                continue
            xs.append(float(value))
            ys.append(float(row["later"]) - float(row["percent_owned"]))
        return xs, ys

    train_pairs = usable_pairs[:-1]
    test_pair = usable_pairs[-1]
    test_x, test_y = pair_rows(test_pair)
    if len(test_x) < min_players:
        return PredictiveCheck(
            question=question,
            method="spearman on capture pairs",
            n=len(test_x),
            statistic=0.0,
            p_value=1.0,
            held_out=bool(train_pairs),
            verdict="insufficient overlap",
            note=(
                f"only {len(test_x)} players carry both a signal and a percent-owned change "
                f"across {test_pair[0]:%Y-%m-%d} -> {test_pair[1]:%Y-%m-%d}"
            ),
        )

    stat, p = spearmanr(test_x, test_y)
    detail: dict[str, float] = {"test_pairs": 1.0, "train_pairs": float(len(train_pairs))}
    if train_pairs:
        tx: list[float] = []
        ty: list[float] = []
        for pair in train_pairs:
            a, b = pair_rows(pair)
            tx.extend(a)
            ty.extend(b)
        if len(tx) >= min_players:
            in_sample, in_p = spearmanr(tx, ty)
            detail["in_sample_statistic"] = float(in_sample)
            detail["in_sample_p"] = float(in_p)
    held_out = bool(train_pairs)
    same_sign = (
        detail["in_sample_statistic"] * float(stat) > 0
        if "in_sample_statistic" in detail
        else False
    )
    if not held_out:
        verdict = "one capture pair only: association measured in sample, not held out"
    elif p < 0.05 and same_sign:
        verdict = "predicts out of sample"
    else:
        verdict = "does not survive the held-out pair"
    return PredictiveCheck(
        question=question,
        method="spearman, fitted on earlier capture pairs, measured on the last",
        n=len(test_x),
        statistic=float(stat),
        p_value=float(p),
        held_out=held_out,
        verdict=verdict,
        note=f"held-out pair {test_pair[0]:%Y-%m-%d} -> {test_pair[1]:%Y-%m-%d}",
        detail=MappingProxyType(detail),
    )


# --------------------------------------------------------------------------------------
# Screen 3: ownership momentum from our own captures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusDepth:
    """How much point-in-time history actually exists. Report it before reporting a trend.

    The corpus is young and the honest statement of that is a number, not an adjective.
    `usable` is False when there are fewer than two captures far enough apart to difference,
    which is the state the 2026 corpus is in today.
    """

    season: int
    captures: tuple[dt.datetime, ...]
    span_days: float
    players: int
    usable: bool
    note: str

    @property
    def n_captures(self) -> int:
        return len(self.captures)


def corpus_depth(
    season: int,
    *,
    root=DEFAULT_ROOT,
    variant: str = "ppr",
    min_gap_days: float = 1.0,
    history: pl.DataFrame | None = None,
) -> CorpusDepth:
    """Count the captures before trusting anything differenced across them.

    Measured on this repo on 2026-09-07: the 2026 PPR corpus holds a single capture, and the
    two captures of 2024 (2024-12-31 and 2026-09-07) are byte-identical across every
    ownership column for all 1,096 shared players. ESPN freezes a finished season's ownership
    block, so a past season's captures are duplicates rather than a series, and this number
    can only go up going forward.
    """
    if history is None:
        try:
            history = load_ownership_history((season,), root=root, variant=variant)
        except CorpusError as err:
            return CorpusDepth(
                season=season,
                captures=(),
                span_days=0.0,
                players=0,
                usable=False,
                note=str(err),
            )
    if history.is_empty():
        return CorpusDepth(
            season=season,
            captures=(),
            span_days=0.0,
            players=0,
            usable=False,
            note=f"no ownership rows for season {season}",
        )
    times = tuple(sorted(history["captured_at"].unique().to_list()))
    span = (times[-1] - times[0]).total_seconds() / 86400.0 if len(times) > 1 else 0.0
    players = int(history["espn_id"].n_unique())
    usable = len(times) > 1 and span >= min_gap_days
    note = (
        f"{len(times)} capture(s) spanning {span:.1f} day(s)"
        if usable
        else (
            f"{len(times)} capture(s): velocity has no denominator yet. A finished season's "
            "ownership block is frozen by ESPN, so this series only accrues forward."
        )
    )
    return CorpusDepth(
        season=season,
        captures=times,
        span_days=span,
        players=players,
        usable=usable,
        note=note,
    )


@dataclass(frozen=True, slots=True)
class OwnershipTrack:
    """One player's point-in-time ownership series, differenced where that is possible.

    `owned_velocity` is `None` rather than `0.0` when the corpus cannot support it. Those are
    different claims -- "not moving" and "we have no idea whether he is moving" -- and a
    board that renders them identically is the reason this is Optional.

    `espn_percent_change` is ESPN's own trailing delta, which exists in a single capture. It
    is the fallback, and it is somebody else's window over somebody else's population.
    """

    player_id: int
    name: str
    position_id: int
    captures: int
    first_at: dt.datetime
    last_at: dt.datetime
    span_days: float
    first_owned: float | None
    last_owned: float | None
    #: Percentage points of roster rate per day. None when there is no span to divide by.
    owned_velocity: float | None
    first_adp: float | None
    last_adp: float | None
    #: Draft-position ranks per day, negative when the field is drafting him earlier.
    adp_drift: float | None
    espn_percent_change: float | None = None

    @property
    def position(self) -> str:
        return POSITION_ABBREV.get(self.position_id, str(self.position_id))

    @property
    def measured(self) -> bool:
        return self.owned_velocity is not None


def ownership_momentum(
    history: pl.DataFrame,
    *,
    min_span_days: float = 1.0,
    limit: int | None = None,
    min_last_owned: float | None = None,
) -> tuple[OwnershipTrack, ...]:
    """Percent-owned velocity and ADP drift per player, from our own captures.

    **Aligned on `(espn_id, captured_at)`, never on row order.** The history is a stack of
    daily files and a player who was absent from one of them would otherwise have his series
    shifted by a day against everyone else's, which produces a velocity out of a join bug.

    Handles the single-capture corpus by construction: `span_days` is zero, so the velocity
    is `None` and nothing is divided. Sorted by absolute velocity where one exists and by
    ESPN's own trailing change where it does not, so the board is still ordered by "who is
    moving" in both states.
    """
    if history.is_empty():
        return ()
    needed = {"espn_id", "captured_at", "percent_owned"}
    missing = needed - set(history.columns)
    if missing:
        raise MarketError(f"ownership history is missing {sorted(missing)}")

    frame = history.sort(["espn_id", "captured_at"])
    grouped = frame.group_by("espn_id").agg(
        [
            pl.col("full_name").last().alias("name"),
            pl.col("default_position_id").last().alias("position_id"),
            pl.len().alias("captures"),
            pl.col("captured_at").first().alias("first_at"),
            pl.col("captured_at").last().alias("last_at"),
            pl.col("percent_owned").first().alias("first_owned"),
            pl.col("percent_owned").last().alias("last_owned"),
            pl.col("average_draft_position").first().alias("first_adp"),
            pl.col("average_draft_position").last().alias("last_adp"),
            pl.col("percent_change").last().alias("espn_percent_change"),
        ]
    )

    out: list[OwnershipTrack] = []
    for row in grouped.iter_rows(named=True):
        span = (row["last_at"] - row["first_at"]).total_seconds() / 86400.0
        enough = row["captures"] > 1 and span >= min_span_days
        if min_last_owned is not None and (row["last_owned"] or 0.0) < min_last_owned:
            continue
        velocity = drift = None
        if enough:
            if row["first_owned"] is not None and row["last_owned"] is not None:
                velocity = (float(row["last_owned"]) - float(row["first_owned"])) / span
            if row["first_adp"] is not None and row["last_adp"] is not None:
                drift = (float(row["last_adp"]) - float(row["first_adp"])) / span
        out.append(
            OwnershipTrack(
                player_id=int(row["espn_id"]),
                name=row["name"] or "",
                position_id=int(row["position_id"] or 0),
                captures=int(row["captures"]),
                first_at=row["first_at"],
                last_at=row["last_at"],
                span_days=span,
                first_owned=row["first_owned"],
                last_owned=row["last_owned"],
                owned_velocity=velocity,
                first_adp=row["first_adp"],
                last_adp=row["last_adp"],
                adp_drift=drift,
                espn_percent_change=row["espn_percent_change"],
            )
        )
    out.sort(
        key=lambda t: (
            -abs(t.owned_velocity if t.owned_velocity is not None else 0.0),
            -abs(t.espn_percent_change or 0.0),
            t.player_id,
        )
    )
    return tuple(out[:limit] if limit else out)


# --------------------------------------------------------------------------------------
# Screen 4: cross-source ADP
# --------------------------------------------------------------------------------------

#: Sleeper's ADP flavor to read for a league, keyed the same way `rank_type_for` decides.
SLEEPER_FLAVORS: Mapping[str, str] = MappingProxyType(
    {"PPR": "adp_ppr", "STANDARD": "adp_std", "HALF": "adp_half_ppr"}
)

#: Sleeper position string -> ESPN `defaultPositionId`, for the name join's tie-break.
SLEEPER_POSITIONS: Mapping[str, int] = MappingProxyType(
    {"QB": QB, "RB": RB, "WR": WR, "TE": TE, "K": K, "DEF": DST}
)

#: Players at a position before its mean rank gap gets a `t` at all. Below this the mean is
#: reported and the `t` is NaN: four players who happen to agree are not a positional shift,
#: and a zero-variance block of three used to come back as t = infinity.
MIN_POSITION_N = 5


@dataclass(frozen=True, slots=True)
class PlatformGap:
    """One player priced differently on two platforms, compared in ranks.

    Ranks rather than raw ADP because the two are not the same unit: ESPN's is an average
    pick in ESPN's default 10-team-ish drafts and Sleeper's is an average pick in Sleeper's,
    and the level difference between the boards says nothing. `rank_delta` is
    `espn_rank - other_rank`, so **positive means the other platform drafts him earlier** and
    ESPN is where he is cheap.
    """

    player_id: int
    name: str
    position_id: int
    espn_adp: float
    other_adp: float
    espn_rank: float
    other_rank: float
    rank_delta: float
    source: str
    flavor: str

    @property
    def position(self) -> str:
        return POSITION_ABBREV.get(self.position_id, str(self.position_id))


@dataclass(frozen=True, slots=True)
class PlatformComparison:
    """Two platforms' draft boards, compared player by player and summarised by position.

    The per-position summary is the part that carries information. Individual gaps are noisy
    -- two ADP samples of different sizes over different user bases -- but a whole position
    moving in one direction with a `t` of six is a format difference or a population
    difference, and either way it is the same mispricing every week of the draft season.

    **Three things about `by_position` that stop it being read as more than it is.**

    * **The rows are not independent of each other.** Both sides are ranks of the *same*
      players, so the deltas sum to exactly zero: live, QB totals -304.5 and RB +283.5 and
      the two are one rotation seen twice, not two findings. Read the largest one and treat
      the rest as its counterweight.
    * **It is a statement about ESPN's ADP column, not about ESPN.** Re-run against ESPN's
      other published price -- the uncensored `draftRanksByRankType` this module otherwise
      prefers -- over the same players and the quarterback gap falls from -11.7 (t = -6.0)
      to -3.5 (t = -1.3). ESPN's two prices disagree with each other about quarterbacks by
      more than half the effect. The gap is real and it is not "ESPN drafts QBs early", it
      is "ESPN's *ADP* drafts QBs early".
    * **It survives the obvious artifacts, which is why it is still here.** Restricting to
      players both platforms price inside their own drafted range moves the QB gap from
      -11.7 to -13.6 (so it is not ESPN's ADP censoring); splitting by ESPN ADP band gives
      t = -3.1 / -3.6 / -2.8 (so it is not the censoring boundary); and permuting the
      position labels 5,000 times puts the 95th percentile of the largest |t| at 2.3 against
      an observed 6.0 (so it is not four positions being tested at once).
    """

    source: str
    flavor: str
    n: int
    spearman: float
    gaps: tuple[PlatformGap, ...]
    #: position_id -> (mean rank gap, n, t statistic)
    by_position: Mapping[int, tuple[float, int, float]]
    joined: int
    unjoined: int

    def cheap_on_espn(self, limit: int = 10) -> tuple[PlatformGap, ...]:
        return tuple(sorted(self.gaps, key=lambda g: -g.rank_delta)[:limit])

    def expensive_on_espn(self, limit: int = 10) -> tuple[PlatformGap, ...]:
        return tuple(sorted(self.gaps, key=lambda g: g.rank_delta)[:limit])

    def describe(self) -> str:
        head = (
            f"ESPN vs {self.source} {self.flavor}: {self.n} players compared "
            f"({self.joined} joined, {self.unjoined} unjoinable), spearman {self.spearman:.3f}"
        )
        lines = [
            head,
            "  positive gap = the other platform drafts him earlier (ESPN cheap)",
            "  the rows sum to zero by construction: one position early forces another late",
        ]
        for pos, (mean, n, t) in sorted(self.by_position.items(), key=lambda kv: -kv[1][0]):
            shown = "  n/a" if not math.isfinite(t) else f"{t:+5.2f}"
            lines.append(
                f"    {POSITION_ABBREV.get(pos, pos):>3} n={n:>3} mean gap {mean:+6.1f} t={shown}"
            )
        return "\n".join(lines)


def _compact(name: str) -> str:
    from ..data.ids import compact_name

    return compact_name(name)


def crosswalk(
    snapshot: MarketSnapshot,
    rows: Sequence[SleeperProjection],
    *,
    sleeper_espn_ids: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """`sleeper_id -> espn_id`, by Sleeper's own id where it has one and by name where not.

    **Sleeper's `espn_id` is not enough on its own and the failure is not random.** Measured
    on the 553 season-projection rows carrying an ADP, only 160 resolve through Sleeper's
    crosswalk, and the ones it misses are the *stars*: Gibbs, Bijan Robinson, Ja'Marr Chase
    and Puka Nacua all come back None while Josh Allen and Lamar Jackson resolve. Sleeper's
    espn_id column is populated for older players and left blank for recent draft classes, so
    an id-only join drops precisely the top of the board -- which is the only part of the
    board a draft-price comparison is about.

    Adding a `compact_name` + position join takes it to 530 of 553. Collisions are dropped
    rather than guessed: two ESPN players sharing a compact name at one position is rare and
    a wrong join is worse than a missing one.
    """
    by_key: dict[tuple[str, int], list[int]] = {}
    for pid, rec in snapshot.records.items():
        if not rec.name:
            continue
        by_key.setdefault((_compact(rec.name), rec.position_id), []).append(pid)

    out: dict[str, int] = {}
    for row in rows:
        direct = (sleeper_espn_ids or {}).get(row.sleeper_id)
        if direct is not None and int(direct) in snapshot.records:
            out[row.sleeper_id] = int(direct)
            continue
        position = SLEEPER_POSITIONS.get((row.position or "").upper())
        if position is None:
            continue
        candidates = by_key.get((_compact(row.name), position))
        if candidates and len(candidates) == 1:
            out[row.sleeper_id] = candidates[0]
    return out


def compare_adp(
    snapshot: MarketSnapshot,
    rows: Sequence[SleeperProjection],
    *,
    flavor: str = "adp_half_ppr",
    source: str = "sleeper",
    sleeper_espn_ids: Mapping[str, int] | None = None,
    informative_only: bool = True,
    limit: int | None = None,
) -> PlatformComparison:
    """Rank ESPN's ADP against another platform's and report where they disagree.

    `informative_only` applies `valuation.MARKET_INFORMATIVE_RANGE["adp"]`, which is not
    optional in practice: 845 of 1,036 ESPN players sit inside the censored band at the last
    pick of ESPN's default draft, and comparing that band to a real ADP produces a board of
    pure artifact.
    """
    from scipy.stats import rankdata, spearmanr

    ids = crosswalk(snapshot, rows, sleeper_espn_ids=sleeper_espn_ids)
    lo, hi = MARKET_INFORMATIVE_RANGE.get("adp", (None, None)) if informative_only else (None, None)

    pairs: list[tuple[PoolRecord, float]] = []
    for row in rows:
        pid = ids.get(row.sleeper_id)
        if pid is None:
            continue
        other = row.adp().get(flavor)
        rec = snapshot.record(pid)
        if other is None or rec is None or rec.adp is None:
            continue
        if (lo is not None and rec.adp < lo) or (hi is not None and rec.adp >= hi):
            continue
        pairs.append((rec, float(other)))

    if len(pairs) < 3:
        return PlatformComparison(
            source=source,
            flavor=flavor,
            n=len(pairs),
            spearman=0.0,
            gaps=(),
            by_position=MappingProxyType({}),
            joined=len(ids),
            unjoined=len(rows) - len(ids),
        )

    espn = np.array([rec.adp for rec, _ in pairs], dtype=float)
    other = np.array([value for _, value in pairs], dtype=float)
    espn_rank = rankdata(espn)
    other_rank = rankdata(other)
    delta = espn_rank - other_rank

    gaps = tuple(
        PlatformGap(
            player_id=rec.player_id,
            name=rec.name,
            position_id=rec.position_id,
            espn_adp=float(rec.adp or 0.0),
            other_adp=value,
            espn_rank=float(espn_rank[i]),
            other_rank=float(other_rank[i]),
            rank_delta=float(delta[i]),
            source=source,
            flavor=flavor,
        )
        for i, (rec, value) in enumerate(pairs)
    )

    by_position: dict[int, tuple[float, int, float]] = {}
    for pos in sorted({g.position_id for g in gaps}):
        values = np.array([g.rank_delta for g in gaps if g.position_id == pos])
        if values.size < 2:
            continue
        se = values.std(ddof=1) / math.sqrt(values.size)
        mean = float(values.mean())
        if values.size < MIN_POSITION_N:
            # Reported, because a position that exists should appear; without a t, because a
            # positional mean over four players is a rumour with a standard error attached.
            by_position[pos] = (mean, int(values.size), math.nan)
            continue
        # A position whose every player carries the identical gap has a zero standard error,
        # and the first version of this reported that as t = +/-inf, on the argument that
        # perfect consistency is the strongest evidence available. It is not: with three
        # players it is three observations that happen to agree, and infinity is a claim of
        # certainty from n = 3. NaN says "no dispersion to divide by", which is the truth,
        # and `describe` prints it as n/a.
        t = mean / se if se > 0 else math.nan
        by_position[pos] = (mean, int(values.size), t)

    ordered = tuple(sorted(gaps, key=lambda g: -abs(g.rank_delta)))
    return PlatformComparison(
        source=source,
        flavor=flavor,
        n=len(pairs),
        spearman=float(spearmanr(espn, other).statistic),
        gaps=ordered[:limit] if limit else ordered,
        by_position=MappingProxyType(by_position),
        joined=len(ids),
        unjoined=len(rows) - len(ids),
    )


def sleeper_flavor_for(rank_types: Sequence[str]) -> str:
    """Sleeper's ADP flavor matching the ESPN board this league reads."""
    if len(rank_types) > 1:
        return SLEEPER_FLAVORS["HALF"]
    return SLEEPER_FLAVORS.get(rank_types[0], SLEEPER_FLAVORS["HALF"])


# --------------------------------------------------------------------------------------
# Screen 5: who is actually available
# --------------------------------------------------------------------------------------


def rostered_in(sim: LeagueSim) -> frozenset[int]:
    """Every player id on a roster in THIS league, which is what availability means here.

    Not `percentOwned`. ESPN's roster rate describes the population; a 4%-owned back sitting
    on a rival's bench is not a waiver target, and a 70%-owned one your league dropped is.
    """
    return frozenset(p for f in sim.state.franchises for p in f.player_ids)


#: Rest-of-season VORP below which a free agent is not a claim. A waiver move costs a roster
#: spot and the player it drops, so the bar is not "positive" but "worth the transaction".
#: Two points over seventeen weeks is 0.12 points a week and is inside the projection
#: layer's own noise; ten is roughly half a point a week, which is a real if small edge.
MIN_CLAIM_VORP = 10.0


def availability_board(
    disagreements: Sequence[FieldDisagreement],
    rostered: Iterable[int],
    *,
    limit: int | None = 20,
    buys_only: bool = True,
    min_agreement: int = 1,
    min_vorp: float | None = MIN_CLAIM_VORP,
) -> tuple[FieldDisagreement, ...]:
    """The disagreement board filtered to players nobody in this league has.

    The whole point of the module lands here: a 40-rank disagreement about a player on a
    rival's roster is an argument, and a 20-rank one about a free agent is a move.

    **Ordered by what the claim adds, not by how surprised the field would be, and that is a
    correction of the first version of this function.** `percentile_delta` is mechanically
    larger the deeper you go: measured on the live Wine Wednesday board, the rank correlation
    of |percentile_delta| with our own positional rank is +0.31 and with rest-of-season VORP
    is -0.40, and the free agents this screen selects average +2.9 VORP against +84.2 for
    the rostered players it drops. So sorting the *available* slice by delta sorts it by
    depth, and it duly did: the first live run of this board put Kyle Juszczyk -- a fullback
    worth +2.2 points over seventeen weeks, 0.13 a week -- above Ty Johnson at +37.1, because
    the field ranks a fullback even lower than we do. Ranks are how the disagreement is
    *found*; points are what a claim is *worth*, and the board a manager acts on has to be
    ordered by the second. The delta stays on every row and stays the qualifying filter.

    `min_vorp` is the second half of the same correction: a disagreement about a player worth
    less than a claim costs is not a move at any rank gap. Pass `None` to see the whole
    slice, which is the diagnostic view rather than the actionable one.
    """
    owned = {int(p) for p in rostered}
    out = [
        d
        for d in disagreements
        if d.player_id not in owned
        and (not buys_only or d.is_buy)
        and d.agreement >= min_agreement
        and (min_vorp is None or d.ros_vorp >= min_vorp)
    ]
    out.sort(key=lambda d: (-d.ros_vorp, -d.percentile_delta, d.player_id))
    return tuple(out[:limit] if limit else out)


# --------------------------------------------------------------------------------------
# The whole thing, per league
# --------------------------------------------------------------------------------------


def context_from(sim: LeagueSim, *, my_team_id: int | None = None) -> LeagueContext:
    """A `core.LeagueContext` for a built `LeagueSim`.

    The state carries the league shape and the settings carry the calendar; the scorer has to
    come from `pipeline.scoring_for`, which re-parses the league's own settings rather than
    trusting ESPN's `appliedTotal`. Note that `pipeline.build` closes any client it created,
    so a caller who wants both a sim and a live pool must own the client.
    """
    from ..pipeline import scoring_for

    settings = sim.league.settings()
    state = sim.state
    return LeagueContext(
        league_id=state.league_id,
        season=state.season,
        name=state.name,
        size=len(state.franchises),
        lineup_slot_counts=state.lineup_slot_counts,
        slot_eligibility=state.slot_eligibility,
        scorer=scoring_for(sim.league).score,
        playoff_team_count=state.playoff_team_count,
        playoff_weeks=settings.schedule.playoff_weeks,
        regular_season_weeks=settings.schedule.regular_season_weeks,
        uses_faab=settings.acquisition.uses_faab,
        faab_budget=settings.acquisition.budget,
        my_team_id=my_team_id if my_team_id is not None else state.my_team_id,
    )


@dataclass(frozen=True, slots=True)
class MarketReport:
    """Every market screen for one league, with each screen's own honesty attached."""

    league_id: int
    season: int
    name: str
    from_week: int
    rank_types: tuple[str, ...]
    captured_at: dt.datetime
    buys: tuple[FieldDisagreement, ...]
    sells: tuple[FieldDisagreement, ...]
    available: tuple[FieldDisagreement, ...]
    analysts: tuple[AnalystSignal, ...]
    dispersion_check: PredictiveCheck
    direction_check: PredictiveCheck
    held_out_check: PredictiveCheck
    corpus: CorpusDepth
    momentum: tuple[OwnershipTrack, ...]
    platform: PlatformComparison | None = None
    #: The same dispersion question asked of the week-1 board. Carried because the season
    #: board's result was sold on "same sign on both boards" and the week board does not in
    #: fact survive the controls, which a reader has to see next to the number it qualifies.
    dispersion_week_check: PredictiveCheck | None = None
    #: One line saying how many of the analyst x board direction tests hit, out of how many
    #: were run. A single p < 0.05 out of sixteen is what the direction check found.
    direction_family: str = ""

    def describe(self, top: int = 8) -> str:
        lines = [
            f"{self.name} ({self.league_id}) season {self.season}, from week {self.from_week}",
            f"  board: {'/'.join(self.rank_types)}"
            f"   pool captured {self.captured_at:%Y-%m-%d %H:%M}",
            "",
            "  WE LIKE THEM MORE THAN THE FIELD (rank delta, positive = we are higher)",
            "  DIAGNOSTIC ORDER -- biggest disagreement first, which is not biggest value:",
            f"  the gap grows with depth, so read the VORP column. '{MIN_CLAIM_VORP:g}' is the",
            "  claim bar and a 'k/n agree' is one source agreeing with itself, not n sources.",
        ]
        for d in self.buys[:top]:
            flag = "FREE" if d.available else "    "
            thin = "  (thin)" if d.ros_vorp < MIN_CLAIM_VORP else ""
            lines.append(
                f"    {flag} {d.name:<22} {d.position:>3} ours#{d.our_rank:<3} "
                f"delta {d.rank_delta:+6.1f} ({d.percentile_delta:+.0%} of board), "
                f"{d.agreement}/{d.n_metrics} agree  VORP {d.ros_vorp:+7.1f} "
                f"({d.vorp_per_week:+.2f}/wk)  owned "
                f"{'--' if d.percent_owned is None else f'{d.percent_owned:5.1f}%'}{thin}"
            )
        lines += ["", "  WE LIKE THEM LESS THAN THE FIELD (sell / do not chase)"]
        for d in self.sells[:top]:
            lines.append(
                f"         {d.name:<22} {d.position:>3} ours#{d.our_rank:<3} "
                f"delta {d.rank_delta:+6.1f} ({d.percentile_delta:+.0%} of board), "
                f"{d.agreement}/{d.n_metrics} agree  VORP {d.ros_vorp:+7.1f} "
                f"({d.vorp_per_week:+.2f}/wk)"
            )
        lines += [
            "",
            "  AVAILABLE IN THIS LEAGUE AND UNDERRATED",
            f"    (ordered by what the claim adds, not by the rank gap; VORP >= "
            f"{MIN_CLAIM_VORP:g} over {self.available[0].ros_weeks if self.available else 17} "
            "weeks)",
        ]
        if not self.available:
            lines.append("    (nothing clears the bar: no free agent here is both underrated")
            lines.append("     by the field and worth a roster spot on our own numbers)")
        for d in self.available[:top]:
            lines.append(
                f"    {d.name:<22} {d.position:>3} ours#{d.our_rank:<3} "
                f"VORP {d.ros_vorp:+7.1f} ({d.vorp_per_week:+.2f}/wk)  "
                f"delta {d.rank_delta:+6.1f} ({d.percentile_delta:+.0%})  owned "
                f"{'--' if d.percent_owned is None else f'{d.percent_owned:5.1f}%'}  "
                f"disp {'--' if d.dispersion is None else f'{d.dispersion:4.1f}'}"
            )
        lines += [
            "",
            "  ANALYST BOARDS (pool-level: league-independent, and identical in every",
            "  league reading the same board -- three reports are one measurement)",
            "    " + self.dispersion_check.describe(),
        ]
        if self.dispersion_week_check is not None:
            lines.append(
                "    same question, WEEK-1 board: " + self.dispersion_week_check.describe()
            )
        lines += [
            "    " + self.direction_check.describe(),
        ]
        if self.direction_family:
            lines.append("      " + self.direction_family)
        lines += [
            "    " + self.held_out_check.describe(),
            "",
            f"  CORPUS: {self.corpus.note}",
        ]
        if self.platform is not None:
            lines += [
                "",
                "  CROSS-PLATFORM (pool-level, and a DRAFT-season edge: all three leagues",
                "  have drafted, so it prices trades rather than picks)",
                "  " + self.platform.describe().replace("\n", "\n  "),
            ]
        return "\n".join(lines)


def screen_league(
    sim: LeagueSim,
    snapshot: MarketSnapshot,
    report: ValuationReport,
    *,
    rank_types: Sequence[str] | None = None,
    sleeper_rows: Sequence[SleeperProjection] | None = None,
    sleeper_espn_ids: Mapping[str, int] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    positions: Iterable[int] | None = (QB, RB, WR, TE),
    top: int = 25,
    min_metrics: int = 2,
    depth_multiple: float = DEFAULT_DEPTH_MULTIPLE,
    history: pl.DataFrame | None = None,
) -> MarketReport:
    """Run every screen on one league. The entry point.

    Takes a built `ValuationReport` rather than building one, because the caller usually has
    several leagues and one snapshot: the pool pull is shared and the valuation is not, and
    making that split explicit is cheaper than hiding a network call in here.
    """
    types = tuple(rank_types) if rank_types else rank_type_for(_scorer(sim))
    quotes = snapshot.quotes(types)
    owned = rostered_in(sim)
    disagreements = field_disagreements(
        report.values,
        quotes,
        positions=positions,
        metrics=metrics,
        snapshot=snapshot,
        rank_types=types,
        rostered=owned,
        min_metrics=min_metrics,
        depth=report.replacement.baseline_ranks(),
        depth_multiple=depth_multiple,
    )
    buys = tuple(d for d in disagreements if d.is_buy)[:top]
    sells = tuple(d for d in disagreements if not d.is_buy)[:top]
    depth = corpus_depth(report.season, history=history)
    hist = history
    if hist is None:
        try:
            hist = load_ownership_history((report.season,))
        except CorpusError:
            hist = pl.DataFrame()
    boards = snapshot.boards(types)
    dispersion_signal = {pid: b.dispersion for pid, b in boards.items()}
    platform = None
    if sleeper_rows:
        platform = compare_adp(
            snapshot,
            sleeper_rows,
            flavor=sleeper_flavor_for(types),
            sleeper_espn_ids=sleeper_espn_ids,
        )
    return MarketReport(
        league_id=report.league_id,
        season=report.season,
        name=report.name,
        from_week=report.from_week,
        rank_types=types,
        captured_at=snapshot.captured_at,
        buys=buys,
        sells=sells,
        available=availability_board(disagreements, owned, limit=top),
        analysts=analyst_signals(snapshot, rank_types=types, min_percent_owned=1.0),
        dispersion_check=dispersion_vs_volatility(snapshot, rank_types=types),
        direction_check=gap_vs_direction(snapshot, rank_types=types),
        held_out_check=predicts_ownership_change(
            hist,
            dispersion_signal,
            question="does analyst dispersion predict subsequent percent-owned change",
        ),
        corpus=depth,
        momentum=ownership_momentum(hist, limit=top) if not hist.is_empty() else (),
        platform=platform,
        dispersion_week_check=dispersion_vs_volatility(
            snapshot, rank_types=types, scoring_period=1
        ),
        direction_family=family_verdict(analyst_direction_family(snapshot, rank_types=types)),
    )


def _scorer(sim: LeagueSim):
    from ..pipeline import scoring_for

    return scoring_for(sim.league).score


__all__ = [
    "CLAY_SOURCE",
    "CONSENSUS_SOURCE",
    "CONTROL_DEGREE",
    "DEFAULT_METRICS",
    "MAX_VORP_TIE",
    "MIN_ANALYSTS",
    "MIN_CLAIM_VORP",
    "MIN_POSITION_N",
    "PUBLISHING_SOURCES",
    "RANK_SLOT_POSITION",
    "RANK_SOURCES",
    "SEASON_BOARD",
    "AnalystBoard",
    "AnalystSignal",
    "CorpusDepth",
    "FieldDisagreement",
    "MarketError",
    "MarketReport",
    "MarketSnapshot",
    "MetricRank",
    "OwnershipTrack",
    "PlatformComparison",
    "PlatformGap",
    "PoolRecord",
    "PredictiveCheck",
    "analyst_direction_family",
    "analyst_signals",
    "availability_board",
    "compare_adp",
    "context_from",
    "corpus_depth",
    "crosswalk",
    "dispersion_vs_volatility",
    "drop_vorp_ties",
    "family_verdict",
    "fetch_snapshot",
    "field_disagreements",
    "gap_vs_direction",
    "ownership_momentum",
    "parse_boards",
    "parse_pool",
    "partial_spearman",
    "predicts_ownership_change",
    "rank_type_for",
    "rostered_in",
    "screen_league",
    "sleeper_flavor_for",
]
