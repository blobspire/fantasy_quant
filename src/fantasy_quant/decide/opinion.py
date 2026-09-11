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
from dataclasses import dataclass

from ..data.etr import EtrRankings
from .wire import all_rostered

log = logging.getLogger(__name__)


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
