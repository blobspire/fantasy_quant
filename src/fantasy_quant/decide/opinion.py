"""A second opinion on the player pool, and what to do with it.

Everything this repo decides runs on one set of numbers. `pipeline.league_projections`
reads ESPN's own weekly projections out of the corpus and re-scores them under the
league's scoring, so "our valuation" and "the counterparty's valuation" are, today,
the same valuation seen from two sides. That is fine for asking *is this trade good*
and useless for asking *is this trade mispriced*, which needs two opinions.

This module holds the second one. It is a human ranking set -- Establish The Run's,
via `data/etr.py` -- and it enters in exactly two shapes:

* **`bench_upgrades`** compares ranks to ranks and stops there. No points, no
  simulation, no unit conversion: a free agent the analyst ranks above somebody on
  your roster, with both ranks and both of his notes attached. It is checkable by eye,
  which is the entire reason it exists as its own function rather than as an input to
  something bigger.
* **`tilt_outlooks`** carries the ordering into the numbers, for the surfaces that
  need a price rather than a shortlist.

**Neither one ever prices a rank.** `decide/valuation.py` records the reason at length:
comparisons against the field are rank-against-rank, "because ADP, percent rostered and
auction dollars are not in the same unit as points and never will be". Fitting
`points(rank)` and evaluating it on someone else's board would break that, and would
also quietly replace this league's solved replacement level with the curve's. So the
tilt *transports* our own points along the analyst's ordering instead -- see there.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from ..core import PlayerOutlook, WeeklyOutlook
from ..data.etr import EtrRankings
from .wire import all_rostered

log = logging.getLogger(__name__)

#: A player projected below this, per rest-of-season, carries no ordering information
#: and is left out of the transport on both sides. It is a guard against a ratio with
#: a near-zero denominator, and the 300-row Draft Kit board shows why it is needed: it
#: ranks kickers ESPN projects at exactly 0.00, and pairing one of those against a real
#: value hands him 121.5 season points out of nowhere. The Top 150 has no such row, so
#: on the board this was built for the guard never fires -- which is the point of
#: having it before the board that needs it arrives.
MIN_TRANSPORT_POINTS = 1.0

#: A player whose ESPN projection is near zero for more weeks than this is left out of
#: the transport and keeps his own numbers. A bye is one such week.
#:
#: This is the mechanism's real limit, and it was found by its headline rather than by
#: reasoning. The board's rank is a REST-OF-SEASON opinion and ESPN's projection is a
#: per-week profile; the transport is multiplicative, so handing a rank-implied season
#: total to a player ESPN has at 0.06 a week for eight weeks crams the whole total into
#: the weeks he does play. Jordyn Tyson -- "recurring hamstring injuries will sideline
#: Tyson through September", WR51 on the board, 0.06/wk for weeks 1-8 in the corpus --
#: came out of the first version at a 60% higher per-game rate with every point landing
#: in the back half of the season, where the bracket pays about three times. He was the
#: top trade target in two of the user's three leagues. He was the only player on the
#: board with more than a bye missing; TreVeyon Henderson (week 1) was the other.
#: Where both sources agree on games played, a season total and a per-game rate are the
#: same thing up to a constant and the transport is exact. Where they do not, there is
#: no games count in a rank to recover, so the honest answer is not to guess one.
#:
#: **Three, not one, and the gap is measured rather than fitted.** Over the three live
#: leagues at week 1 of 2026 the board's players fall into two groups with nothing
#: between them: a bye-plus-one group at **2** weeks absent (Brock Bowers, TreVeyon
#: Henderson) and Jordyn Tyson at **8**. At a threshold of 1 the first group was
#: excluded for nothing -- transporting them moves their per-game rate by 5% and 12%,
#: which is just the analyst disagreeing, and the analyst disagreeing is the entire
#: point of reading his board. Tyson moves by 53%.
#:
#: Re-tested after the ladder horizon was corrected, on the theory that the horizon bug
#: had been the real cause and this rule could go. **It could not.** With every
#: partial-season player transported, Wine Wednesday's top trade becomes "get Jordyn
#: Tyson for DK Metcalf" at +5.25pp with a spread of **+60.46** -- a counterparty sixty
#: playoff-weighted points out of pocket by their own numbers, which is the units error
#: wearing an edge again. The rule stays; only the threshold moves.
MAX_ABSENT_WEEKS = 3


@dataclass(frozen=True, slots=True)
class Upgrade:
    """One free agent the board ranks above one player you hold.

    `gap` is in ranks, not points, and that is deliberate: the board publishes an
    ordering and nothing else, so an ordering is the most this comparison can
    honestly report. `decide/waivers.py` is where a claim gets a price.
    """

    add: int
    drop: int
    position_id: int
    add_rank: int
    drop_rank: int
    add_name: str = ""
    drop_name: str = ""
    add_comment: str = ""
    drop_comment: str = ""
    #: True when the two were compared on the board's OVERALL ladder rather than
    #: within a position. See `cross_position_upgrades` for why that is weaker.
    cross_position: bool = False

    @property
    def gap(self) -> int:
        """How many ranks better the board says the free agent is."""
        return self.drop_rank - self.add_rank


@dataclass(frozen=True, slots=True)
class UpgradeReport:
    """`upgrades`, plus the coverage that says how much of the roster was even asked."""

    upgrades: tuple[Upgrade, ...]
    roster_covered: int
    roster_total: int
    wire_covered: int
    wire_total: int

    def __bool__(self) -> bool:
        return bool(self.upgrades)

    @property
    def blind_spot(self) -> int:
        """Rostered players the board has no opinion about, so no pair can exist."""
        return self.roster_total - self.roster_covered


def _pairs(
    ladder: Mapping[int, tuple[int, int]],
    roster: Iterable[int],
    free_agents: Iterable[int],
    *,
    names: Mapping[int, str],
    comments: Mapping[int, str],
    cross_position: bool,
    limit: int | None,
) -> tuple[Upgrade, ...]:
    held = [(p, ladder[p]) for p in dict.fromkeys(int(x) for x in roster) if p in ladder]
    wire = [(p, ladder[p]) for p in dict.fromkeys(int(x) for x in free_agents) if p in ladder]

    out: list[Upgrade] = []
    for add, (add_pos, add_rank) in wire:
        for drop, (drop_pos, drop_rank) in held:
            if add_pos != drop_pos or drop_rank <= add_rank:
                continue
            out.append(
                Upgrade(
                    add=add,
                    drop=drop,
                    position_id=add_pos,
                    add_rank=add_rank,
                    drop_rank=drop_rank,
                    add_name=names.get(add, str(add)),
                    drop_name=names.get(drop, str(drop)),
                    add_comment=comments.get(add, ""),
                    drop_comment=comments.get(drop, ""),
                    cross_position=cross_position,
                )
            )
    # Widest gap first, then by the id pair so the order is determinate rather than
    # whatever the roster happened to be in -- the same trap `settle`'s forced cut fell
    # into, where ESPN's roster order decided the answer.
    out.sort(key=lambda u: (-u.gap, u.add, u.drop))
    return tuple(out[:limit] if limit is not None else out)


def bench_upgrades(
    board: EtrRankings,
    roster: Sequence[int],
    free_agents: Iterable[int],
    *,
    names: Mapping[int, str] | None = None,
    limit: int | None = None,
) -> UpgradeReport:
    """Free agents the board ranks above somebody you hold, at the same position.

    Same position, always. A cross-position comparison needs the board's overall
    ladder, which carries the analyst's view of what a tight end is worth against a
    running back -- and this league already solves its own, from its own roster
    shape, at the flex fixed point in `decide/valuation.py`. Within a position that
    question never arises, so the comparison is clean.

    Players the board does not rank cannot be compared and are not guessed at. The
    report carries the coverage so a thin answer is visibly thin rather than
    reading as "no upgrades available".
    """
    ladder = board.positional()
    roster_ids = list(dict.fromkeys(int(p) for p in roster))
    wire_ids = list(dict.fromkeys(int(p) for p in free_agents))
    return UpgradeReport(
        upgrades=_pairs(
            ladder,
            roster_ids,
            wire_ids,
            names=names or {},
            comments=board.comments(),
            cross_position=False,
            limit=limit,
        ),
        roster_covered=sum(1 for p in roster_ids if p in ladder),
        roster_total=len(roster_ids),
        wire_covered=sum(1 for p in wire_ids if p in ladder),
        wire_total=len(wire_ids),
    )


def cross_position_upgrades(
    board: EtrRankings,
    roster: Sequence[int],
    free_agents: Iterable[int],
    *,
    names: Mapping[int, str] | None = None,
    limit: int | None = None,
) -> UpgradeReport:
    """The same comparison on the board's OVERALL ladder, which is weaker.

    Kept separate and named for what it is. An overall rank says a receiver is worth
    more than a running back, and that judgement belongs to the league -- it depends
    on how many flex slots it starts and how deep its benches run, both of which
    `decide/valuation.py` solves per league. Reading it off an analyst's national
    board overwrites a solved, league-specific answer with a generic one.

    Useful anyway as a *shortlist*, which is why it is here: "he has this guy 40
    picks higher than yours" is a real thing to notice even when the ladder it is
    measured on is not the league's own.
    """
    overall = {pid: (0, rank) for pid, rank in board.as_map().items()}
    roster_ids = list(dict.fromkeys(int(p) for p in roster))
    wire_ids = list(dict.fromkeys(int(p) for p in free_agents))
    return UpgradeReport(
        upgrades=_pairs(
            overall,
            roster_ids,
            wire_ids,
            names=names or {},
            comments=board.comments(),
            cross_position=True,
            limit=limit,
        ),
        roster_covered=sum(1 for p in roster_ids if p in overall),
        roster_total=len(roster_ids),
        wire_covered=sum(1 for p in wire_ids if p in overall),
        wire_total=len(wire_ids),
    )


def upgrades_for(
    board: EtrRankings,
    sim,
    team_id: int,
    *,
    limit: int | None = None,
) -> UpgradeReport:
    """`bench_upgrades` against one franchise's roster and the league's actual wire.

    **The wire comes from `sim.outlooks`, not `sim.state.pool`.** `pipeline.build`
    pools only rostered players -- measured on the three live leagues at week 1 of
    2026, `state.pool` holds 194-225 ids and every one of them is owned -- so a free
    agent read off the pool is a free agent that does not exist. Reading the wire off
    the pool is the same defect `cc624f6` and `ef36808` each fixed once already, and
    it reports a clean zero rather than an error, which is how it survives.
    `sim.outlooks` is the corpus-wide 598.

    Ownership is `wire.all_rostered`, the helper that exists for exactly this and that
    `ef36808` fixed `trades.wire_pool` to use after it spent a while calling the whole
    pool rostered.
    """
    state = sim.state
    owned = all_rostered(state)
    roster = next((f.player_ids for f in state.franchises if f.team_id == team_id), ())
    names = {int(o.player_id): o.name for o in sim.outlooks}
    wire = [int(o.player_id) for o in sim.outlooks if int(o.player_id) not in owned]
    return bench_upgrades(board, roster, wire, names=names, limit=limit)


def _scaled_week(week: WeeklyOutlook, factor: float) -> WeeklyOutlook:
    """Scale a hurdle gamma's location, leaving its shape and its zero mass alone.

    The hurdle mean is `(1 - p_zero) * shape * scale` and its sd is linear in `scale`
    too, so scaling `scale`, `mean` and `sd` by one factor keeps the three mutually
    consistent -- which matters because `core.WeeklyOutlook` is explicit that `mean`
    and `sd` are the moments of the FULL distribution and must not be reconstructed
    from the gamma alone.

    `p_zero`, `shape`, `playing`, `pro_team_id` and the week key are untouched, so
    byes, the schedule, the correlation blocks and the hurdle mass all survive.
    """
    return replace(
        week,
        mean=week.mean * factor,
        sd=week.sd * factor,
        scale=week.scale * factor,
    )


def tilt_outlooks(
    outlooks: Sequence[PlayerOutlook],
    board: EtrRankings,
    *,
    weight: float = 1.0,
    weeks: Sequence[int] | None = None,
    min_points: float = MIN_TRANSPORT_POINTS,
    max_absent_weeks: int = MAX_ABSENT_WEEKS,
) -> list[PlayerOutlook]:
    """Our own points, re-dealt along the board's ordering. Within a position.

    The mechanism, which is a **permutation and not a model**: at each position, take
    the players the board ranks and we project, sort them by our own rest-of-season
    mean to get a ladder of values, sort them by the board's positional rank, and pair
    the two up. The board's k-th player takes the ladder's k-th value.

    Three properties fall out, and they are the whole reason for this shape rather
    than a fitted `points(rank)` curve:

    * **No rank is ever priced.** The values are ours throughout; the board supplies
      only the order they are handed out in. `decide/valuation.py` records at length
      why comparisons against an outside opinion stay in rank space, and a fitted
      curve evaluated on someone else's board would break it.
    * **Replacement level does not move.** The multiset of rest-of-season means at
      each position is preserved exactly, so the scarcity curves, the VORP baseline
      and the wire are all where they were. On the live leagues the board covers 150
      of 598 projected players; the other 448 -- which is where the wire lives -- are
      not touched at all.
    * **`weight=0.0` is the identity**, byte for byte, which makes every consumer's
      negative control a one-line call rather than a reconstruction.

    `weight` interpolates the per-player factor rather than the ordering, so
    intermediate values shrink each player toward where we had him rather than
    producing some third ordering neither opinion holds.

    **Players ESPN projects as absent for more than a bye are not transported** --
    see `MAX_ABSENT_WEEKS` for the measurement that made this rule. Their own numbers
    stand, and `partial_season` names them so a caller can say so.

    **Pass `weeks`.** It is the horizon the ladder is built over, and it has to be the
    weeks still to be played -- `state.weeks` -- because the board's rank is a
    rest-of-season opinion. `pipeline._fill_weeks` only ever ADDS weeks to an outlook,
    so `sim.outlooks` keeps every week ESPN projected including the ones already in the
    books; defaulting to "every week present" ranks players by a full-season total.
    At week 1 the two are the same and nothing shows. From week 2 they are not: a
    player who was excellent through September and is finished outranks one who is
    about to carry you, and the ratio handed to the transport is computed on the wrong
    total. It also makes `partial_season` permanent -- Tyson's weeks 1-8 stay in the
    outlooks after he returns, so he would be excluded for the rest of the season.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    if weight == 0.0 or not outlooks:
        return list(outlooks)

    ladder = board.positional()
    horizon = _horizon(outlooks, weeks)

    skipped = partial_season(
        outlooks, board, weeks=horizon, min_points=min_points, max_absent_weeks=max_absent_weeks
    )
    if skipped:
        log.info(
            "tilt: %d board player(s) kept their own numbers, absent beyond a bye: %s",
            len(skipped),
            ", ".join(o.name for o in skipped),
        )
    skip_ids = {int(o.player_id) for o in skipped}

    by_position: dict[int, list[tuple[int, float, int]]] = {}
    for o in outlooks:
        entry = ladder.get(int(o.player_id))
        if entry is None or int(o.player_id) in skip_ids:
            continue
        pos, rank = entry
        value = _value(o, horizon)
        if value < min_points:
            continue
        by_position.setdefault(pos, []).append((int(o.player_id), value, rank))

    factors: dict[int, float] = {}
    for pos, rows in by_position.items():
        # The ladder of values we hold at this position, best first. Ties break on the
        # player id so the deal is determinate rather than dictionary order.
        values = sorted((v for _, v, _ in rows), reverse=True)
        # The order the board would hand them out in. Ties on the board's own rank
        # break the same way, for the same reason.
        order = sorted(rows, key=lambda r: (r[2], r[0]))
        for (pid, ours, _), theirs in zip(order, values, strict=True):
            factors[pid] = (1.0 - weight) + weight * (theirs / ours)
        log.debug("tilt: position %d re-dealt %d values", pos, len(rows))

    out: list[PlayerOutlook] = []
    for o in outlooks:
        factor = factors.get(int(o.player_id))
        if factor is None or factor == 1.0:
            out.append(o)
            continue
        out.append(
            replace(
                o,
                weeks={
                    w: (_scaled_week(wo, factor) if w in horizon else wo)
                    for w, wo in o.weeks.items()
                },
            )
        )
    return out


def _horizon(
    outlooks: Sequence[PlayerOutlook], weeks: Sequence[int] | None
) -> frozenset[int]:
    """The weeks the ladder is built over, and the only ones the transport touches.

    `weeks` is `state.weeks` -- what is still to be played. Falling back to every week
    present in the outlooks is correct only in week 1; see `tilt_outlooks`.
    """
    if weeks is not None:
        return frozenset(int(w) for w in weeks)
    return frozenset(w for o in outlooks for w in o.weeks)


def _value(outlook: PlayerOutlook, horizon: frozenset[int]) -> float:
    """This player's expected points over the horizon.

    `PlayerOutlook.mean_from` is the wrong tool: it sums every week at or after one
    index, which past the start of the season means summing games already played, and
    which here would also pull in week 18 when the league's last scored week is 17.
    """
    return sum(wo.mean for w, wo in outlook.weeks.items() if w in horizon)


def partial_season(
    outlooks: Sequence[PlayerOutlook],
    board: EtrRankings,
    *,
    weeks: Sequence[int] | None = None,
    min_points: float = MIN_TRANSPORT_POINTS,
    max_absent_weeks: int = MAX_ABSENT_WEEKS,
) -> tuple[PlayerOutlook, ...]:
    """Board players ESPN projects as absent for more than `max_absent_weeks`.

    "Absent" is a week under `min_points`: the calibration puts an injured player at
    about 0.06 with `p_zero` near 0.84 rather than at an exact zero, so an equality
    test finds nobody. A bye is one such week and is not an absence.

    Counted over the REMAINING weeks. An absence already played is not a reason to
    distrust a rest-of-season rank -- it is the thing the rank has already priced in --
    and counting it would exclude a returning player for the rest of the season.
    """
    ladder = board.positional()
    horizon = _horizon(outlooks, weeks)
    out: list[PlayerOutlook] = []
    for o in outlooks:
        if int(o.player_id) not in ladder:
            continue
        absent = sum(1 for w, wo in o.weeks.items() if w in horizon and wo.mean < min_points)
        if absent > max_absent_weeks:
            out.append(o)
    return tuple(out)


def positional_ranks(
    outlooks: Sequence[PlayerOutlook], *, weeks: Sequence[int] | None = None
) -> dict[int, tuple[int, int]]:
    """`{espn_id: (position_id, rank within that position)}` by OUR projections.

    The counterpart to `EtrRankings.positional()`, built the same way off the same
    horizon so the two are directly comparable: "RB12 by ESPN, RB18 by the analyst" is
    the arbitrage stated in the one unit both sources publish.

    Ranked on rest-of-season points over the remaining weeks, not on ESPN's own frozen
    preseason ordering -- `PlayerOutlook.mean_from` is explicit that the season-total
    field is never revised, and a rank read off it would go stale the first time
    somebody got hurt. Ties break on the player id so the ordering does not depend on
    dictionary order.
    """
    horizon = _horizon(outlooks, weeks)
    by_position: dict[int, list[tuple[float, int]]] = {}
    for o in outlooks:
        by_position.setdefault(o.position_id, []).append((_value(o, horizon), int(o.player_id)))
    out: dict[int, tuple[int, int]] = {}
    for pos, rows in by_position.items():
        for rank, (_, pid) in enumerate(sorted(rows, key=lambda r: (-r[0], r[1])), start=1):
            out[pid] = (pos, rank)
    return out


def rank_pairs(
    outlooks: Sequence[PlayerOutlook],
    board: EtrRankings | None,
    *,
    weeks: Sequence[int] | None = None,
) -> dict[int, dict[str, object]]:
    """Per player: where ESPN has him, where the analyst has him, and the gap.

    `{espn_id: {"espn": "RB12", "etr": "RB18", "gap": -6}}`. `gap` is positive when the
    analyst likes him MORE than our projections do -- a smaller rank number is better,
    so it is `espn_rank - etr_rank`, the same sign convention as
    `EtrRankings.disagreements`' `etr_edge`.

    `etr` is absent for a player the board does not rank, which is most of them: the
    Top 150 covers 150 of ~598 projected players, and saying so is more useful than
    implying the analyst has an opinion he has not published.
    """
    ours = positional_ranks(outlooks, weeks=weeks)
    theirs = board.positional() if board is not None else {}
    abbrev = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
    out: dict[int, dict[str, object]] = {}
    for pid, (pos, rank) in ours.items():
        row: dict[str, object] = {"espn": f"{abbrev.get(pos, pos)}{rank}", "espn_rank": rank}
        hit = theirs.get(pid)
        if hit is not None and hit[0] == pos:
            row["etr"] = f"{abbrev.get(pos, pos)}{hit[1]}"
            row["etr_rank"] = hit[1]
            row["gap"] = rank - hit[1]
        out[pid] = row
    return out
