"""The optimal starting lineup, solved a few million times a second.

Everything above this module -- waiver claims, trades, start/sit, the title-odds
simulation itself -- is a search whose objective is "what does my lineup score",
evaluated once per candidate per simulation per week. So this is the one place in
the system where constant factors are the design, not an afterthought.

**Greedy is exact, and that is a theorem, not a hope.**

Order the starting slots by how restrictive they are and fill each one with the best
player still available. For a *laminar* eligibility family -- any two slots' eligible
sets are nested or disjoint, which is what "dedicated subset of FLEX subset of
SUPERFLEX" means -- this is provably optimal, so the inner loop never runs an LP.

    Proof (exchange). Process slots in increasing order of their eligible set. Let A
    be the slot now being filled, p* the best available player in A, and O an optimal
    completion of the slots not yet fixed. Every remaining slot D containing p*
    satisfies D contains A: p* lies in both, laminarity forces nesting, and D strictly
    inside A would already have been processed. Let q be O's player in A (possibly
    none). Swapping p* into A and q into D changes the total by g(p*) - g(q), where
    g(x) = max(x, f_A) - max(x, f_D) and f is the per-slot floor. Because f_A <= f_D,
    g is non-decreasing: constant f_A - f_D below f_A, rising on [f_A, f_D], zero
    above. p* >= q, so the swap never loses. Induct.

The floor `f` in that proof is the free-agent floor of `optimal_lineup_with_floor`;
with `f == 0` everywhere it degenerates to the ordinary "an empty slot scores zero"
lineup, so one kernel serves both and the proof covers both. The proof needs
`f_A <= f_D` for every slot D solved after A whose eligible set contains A's --
*including when the two sets are equal* -- which is why non-monotone floors are
rejected rather than quietly solved. See `LineupPlan._check_monotone`; the randomized
oracle found the equal-sets case, it was not anticipated.

**Two things measured here that the plan does not say.**

* RESEARCH.md and PLAN.md both assert only that "greedy is provably exact for nested
  slot eligibility"; neither names which real slot table actually violates nesting, and
  the obvious guess -- the IDP block, with its overlapping-looking DL/DB/DP slots -- is
  wrong. Measured against ESPN's own 2026 `lineupSlots` payload, the IDP block is a
  clean tree (DT and DE inside DL, CB and S inside DB, and DL, LB, DB all inside DP),
  and the *only* violation among its 22 starting slots is slot 3 **RB/WR** against slot
  5 **WR/TE**: they share WR and neither contains the other. A league that starts both
  flexes -- an ordinary, non-exotic redraft setting -- is the real trap, and greedy
  loses up to a whole starter there. `test_espn_slot_table_violation_is_rb_wr_vs_wr_te`
  pins this against the cached payload.
* Laminarity is checked over **players, not positions**, and the distinction is worth
  real money. RB/WR against WR/TE is non-laminar as position sets, but on a roster
  carrying no tight end the WR/TE slot's *player* set collapses to a subset of the
  RB/WR slot's and greedy is exact again. Checking the position table would send that
  roster down the exact solver, which costs ~10us a roster-week against greedy's 0.3.
  So the plan compiles the eligibility matrix it is actually given, and the equivalence
  classes it solves over are the distinct columns of that matrix -- derived, never
  assumed to be positions.

**Semantics of an empty slot.** An unfilled slot scores exactly 0, so a player is
started only if he beats his slot's floor *strictly*. Fantasy points go negative (a
three-interception quarterback, a defense that allows 40), and a bye or an injury is
`WeeklyOutlook.zeroed()` -- mean 0, `playing=False`. Both fall out of the same rule:
0 is not greater than 0, so a benched player never displaces an empty slot, and a
negative player never gets started when the slot could simply be left open. Pass
`allow_empty=False` (floor `-inf`) to model a manager who fills every slot regardless;
then an unfillable slot contributes `-inf` and says so loudly. Both solvers honour the
strictness: the assignment-problem fallback is indifferent at a tie, so it unassigns any
slot whose player only *equals* the floor, and `method` therefore never changes which
slots read empty. That matters downstream, where `sim/season.py` picks the lineup on a
projection tensor and then scores the *realised* tensor at those indices -- starting a
projected-zero player there is not free.

**A score that is not a number is not startable.** `-inf` is how `available=False` is
spelled internally, and both kernels drop it; NaN is treated the same way rather than
being allowed to sort to the head of a queue and block every player behind it. The
totals-only and with-assignment paths agree on this, which they did not when the
descending sort was built by reversing an ascending one.

**The free-agent floor treats the wire as unlimited depth.** Each slot independently
falls back to its own floor, so two slots may both claim the same notional streamer.
That is deliberate: the floor exists to stop bench players being credited with value
they do not have, and in a 12-14 team league the wire genuinely holds several
near-replacement bodies at every position. It is a valuation device, not a legality
constraint, and `monotone_floor` documents the one place the approximation bites.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - only for the LeagueContext convenience builder
    from ..core import LeagueContext

log = logging.getLogger(__name__)

Method = Literal["auto", "greedy", "exact"]

#: Cost the exact solver charges for pairing a slot with a player it cannot start.
#: Finite rather than `inf` because `linear_sum_assignment` raises on an infeasible
#: matrix and a finite wall keeps the failure mode "absurd cost", not "exception".
_FORBIDDEN = 1e15
#: Cost of leaving a slot empty when the floor itself is `-inf`. Must sit strictly
#: between a legal pairing and `_FORBIDDEN` so an unfillable slot is left empty rather
#: than filled with an ineligible player.
_UNFILLABLE = 1e12


class LineupError(ValueError):
    """The lineup problem as posed is malformed. Never guessed around."""


# --------------------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------------------


def _as_matrix(eligibility: np.ndarray | Sequence[Sequence[bool]]) -> np.ndarray:
    """A `(slot group, player)` boolean matrix, however it was expressed."""
    matrix = np.asarray(eligibility, dtype=bool)
    if matrix.ndim != 2:
        raise LineupError(f"eligibility must be a 2-D (slot, player) matrix, got {matrix.shape}")
    return matrix


def _pair_relations(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(intersection sizes, set sizes, overlap-but-not-nested mask)` for every pair."""
    counts = matrix.astype(np.int64)
    inter = counts @ counts.T
    sizes = counts.sum(axis=1)
    nested = (inter == sizes[:, None]) | (inter == sizes[None, :])
    crossing = (inter > 0) & ~nested
    return inter, sizes, crossing


def laminar_violations(
    eligibility: np.ndarray | Sequence[Sequence[bool]] | Mapping[int, Iterable[int]],
) -> tuple[tuple[int, int], ...]:
    """Pairs of slots that overlap without nesting, keyed by row index or slot id.

    Accepts either the `(slot, player)` matrix the solver runs on or ESPN's
    `slotId -> eligible defaultPositionIds` mapping, because the two questions --
    "is greedy safe on *this roster*" and "is this league shape safe in general" --
    are different and both get asked.
    """
    if isinstance(eligibility, Mapping):
        slots = sorted(eligibility)
        universe = sorted({p for s in slots for p in eligibility[s]})
        index = {p: i for i, p in enumerate(universe)}
        matrix = np.zeros((len(slots), len(universe)), dtype=bool)
        for row, slot in enumerate(slots):
            for pos in eligibility[slot]:
                matrix[row, index[pos]] = True
        labels: Sequence[int] = slots
    else:
        matrix = _as_matrix(eligibility)
        labels = range(matrix.shape[0])
    if matrix.size == 0:
        return ()
    _, _, crossing = _pair_relations(matrix)
    rows, cols = np.nonzero(np.triu(crossing, k=1))
    return tuple((labels[int(r)], labels[int(c)]) for r, c in zip(rows, cols, strict=True))


def is_laminar(
    eligibility: np.ndarray | Sequence[Sequence[bool]] | Mapping[int, Iterable[int]],
) -> bool:
    """Whether greedy is valid here: every pair of slots nested or disjoint."""
    return not laminar_violations(eligibility)


# --------------------------------------------------------------------------------------
# The compiled plan
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupResult:
    """A solved lineup. `total` carries the caller's leading axes; `assignment` adds one.

    `assignment[..., i]` is the player index in slot instance `i`, or -1 when the slot
    was left empty (or fell back to its free-agent floor). `slot_ids[i]` names it.
    """

    total: np.ndarray
    assignment: np.ndarray | None
    slot_ids: tuple[int, ...]

    @property
    def started(self) -> np.ndarray:
        """Boolean mask over slots: was a real player started here."""
        if self.assignment is None:
            raise LineupError("this result was solved with assignment=False")
        return self.assignment >= 0


@dataclass(frozen=True, slots=True)
class LineupPlan:
    """Everything about a league's starting lineup that does not change per simulation.

    Built once and reused across the whole candidate search: the per-slot eligibility,
    the solve order, the laminarity verdict, and the player equivalence classes. The
    classes are the distinct columns of the eligibility matrix, which for a normal
    roster reproduce the positions exactly but stay correct when they do not.
    """

    #: One entry per slot *instance*, in solve order. Indexes `LineupResult.assignment`.
    slot_ids: tuple[int, ...]
    #: One entry per slot *group* (a slot id and its count), in solve order.
    group_slot_ids: tuple[int, ...]
    group_counts: tuple[int, ...]
    #: The order a *positional* `floor` array is read in, and the order `monotone_floor`
    #: returns: the eligibility rows as the caller supplied them, with zero-count rows
    #: dropped. Deliberately **not** `group_slot_ids`, which is the internal solve order
    #: -- a caller of `optimal_lineup_with_floor` never sees the plan and so cannot know
    #: the solve order, and reading their array in it silently solves a different
    #: problem. Mapping floors are keyed by slot id and are immune either way.
    floor_slot_ids: tuple[int, ...]
    n_players: int
    laminar: bool
    violations: tuple[tuple[int, int], ...]
    #: (group, player) eligibility in solve order.
    _eligible: np.ndarray
    #: Player indices making up each equivalence class.
    _class_players: tuple[np.ndarray, ...]
    #: Class indices each group can draw from.
    _group_classes: tuple[np.ndarray, ...]
    #: Group index pairs `(a, b)`, a before b in solve order, where a's eligible set is
    #: contained in b's. Equal sets are included, and that inclusion is load-bearing.
    _nested_pairs: tuple[tuple[int, int], ...]
    #: `floor_slot_ids` index for each solve-order group: `caller[_floor_perm]` is the
    #: floor vector in solve order.
    _floor_perm: np.ndarray

    # -- construction ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        eligibility: np.ndarray | Sequence[Sequence[bool]],
        slot_counts: Sequence[int] | np.ndarray,
        *,
        slot_ids: Sequence[int] | None = None,
    ) -> LineupPlan:
        """Compile a `(slot group, player)` matrix and its per-group counts."""
        matrix = _as_matrix(eligibility)
        counts = np.asarray(slot_counts, dtype=np.int64)
        if counts.ndim != 1 or counts.shape[0] != matrix.shape[0]:
            raise LineupError(
                f"slot_counts must have one entry per eligibility row: "
                f"{counts.shape} against {matrix.shape}"
            )
        if (counts < 0).any():
            raise LineupError("slot counts must be non-negative")
        ids = tuple(range(matrix.shape[0])) if slot_ids is None else tuple(int(s) for s in slot_ids)
        if len(ids) != matrix.shape[0]:
            raise LineupError("slot_ids must have one entry per eligibility row")
        if len(set(ids)) != len(ids):
            # A repeated id would make a floor mapping and the assignment columns
            # ambiguous, which is a silent wrong answer rather than an error.
            raise LineupError(f"slot ids must be distinct, got {ids}")

        keep = np.nonzero(counts > 0)[0]
        # Most restrictive first. For a laminar family a strict subset is strictly
        # smaller, so ascending size is a valid topological order; the slot id breaks
        # ties between equal (hence mutually nested) sets deterministically. lexsort
        # takes its primary key last.
        sizes = matrix[keep].sum(axis=1)
        order = keep[np.lexsort((np.array([ids[i] for i in keep], dtype=np.int64), sizes))]
        elig = np.ascontiguousarray(matrix[order])
        group_counts = tuple(int(counts[i]) for i in order)
        group_ids = tuple(ids[i] for i in order)
        # Where each solve-order group sat in the caller's own row order. Positional
        # floors are read in *that* order and permuted here, so a caller who never
        # sees the plan cannot silently have their floor vector shuffled.
        caller_pos = {int(row): j for j, row in enumerate(keep)}
        floor_perm = np.array([caller_pos[int(row)] for row in order], dtype=np.intp)

        classes = _equivalence_classes(elig)
        group_classes = tuple(
            np.array([c for c, who in enumerate(classes) if elig[g, who[0]]], dtype=np.intp)
            for g in range(elig.shape[0])
        )

        inter, group_sizes, crossing = _pair_relations(elig)
        rows, cols = np.nonzero(np.triu(crossing, k=1))
        # Reported in slot-id order rather than solve order: these go in front of a
        # human, and "(3, 5)" should mean RB/WR against WR/TE whichever way the roster
        # happened to size them.
        violations = tuple(
            sorted(
                tuple(sorted((group_ids[int(r)], group_ids[int(c)])))
                for r, c in zip(rows, cols, strict=True)
            )
        )
        # `a` is contained in `b` and is solved first. Solve position, not size, is the
        # relation that matters: two slots with *equal* eligible sets are also nested,
        # and their floors have to rise in the order they are filled or greedy burns the
        # best player on the slot with the least to gain from him.
        earlier = np.triu(np.ones(inter.shape, dtype=bool), k=1)
        nested = (inter == group_sizes[:, None]) & earlier
        pairs = tuple((int(a), int(b)) for a, b in zip(*np.nonzero(nested), strict=True))

        instances = tuple(
            sid for sid, n in zip(group_ids, group_counts, strict=True) for _ in range(n)
        )
        return cls(
            slot_ids=instances,
            group_slot_ids=group_ids,
            group_counts=group_counts,
            floor_slot_ids=tuple(ids[int(row)] for row in keep),
            n_players=int(matrix.shape[1]),
            laminar=not violations,
            violations=violations,
            _eligible=elig,
            _class_players=classes,
            _group_classes=group_classes,
            _nested_pairs=pairs,
            _floor_perm=floor_perm,
        )

    @property
    def n_slots(self) -> int:
        return len(self.slot_ids)

    @property
    def n_groups(self) -> int:
        return len(self.group_slot_ids)

    # -- solving -----------------------------------------------------------------------

    def solve(
        self,
        scores: np.ndarray,
        *,
        floor: np.ndarray | Mapping[int, float] | float | None = None,
        available: np.ndarray | None = None,
        allow_empty: bool = True,
        method: Method = "auto",
        assignment: bool = True,
    ) -> LineupResult:
        """Best legal lineup for every entry of a `[..., player]` score tensor.

        `floor` is a per-slot-*group* score a slot falls back to when no available
        player beats it -- the best free agent that slot could stream. Defaults to 0
        (`allow_empty=True`) or `-inf` (`allow_empty=False`). A Mapping is keyed by slot
        id; a positional array is read in `floor_slot_ids` order, which is the order the
        eligibility rows were supplied in, *not* the internal solve order. `available`
        masks players out entirely, for an injury the simulator has already resolved to
        "did not play" rather than "scored zero".
        """
        if method not in ("auto", "greedy", "exact"):
            raise LineupError(f"method must be auto, greedy or exact, got {method!r}")
        eff, lead = self._effective_scores(scores, available)
        floors = self._floor_matrix(floor, allow_empty, eff.shape[0])
        self._check_monotone(floors)
        use_exact = method == "exact" or (method == "auto" and not self.laminar)
        if method == "greedy" and not self.laminar:
            log.warning(
                "league %s eligibility is not laminar (%s); greedy is not exact here",
                self.group_slot_ids,
                self.violations,
            )
        solver = self._exact if use_exact else self._greedy
        total, assign = solver(eff, floors, assignment)
        return LineupResult(
            total=total.reshape(lead),
            assignment=None if assign is None else assign.reshape((*lead, self.n_slots)),
            slot_ids=self.slot_ids,
        )

    # -- input conditioning ------------------------------------------------------------

    def _effective_scores(
        self, scores: np.ndarray, available: np.ndarray | None
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        arr = np.asarray(scores)
        if arr.dtype.kind != "f":
            arr = arr.astype(np.float64)
        if arr.ndim == 0 or arr.shape[-1] != self.n_players:
            raise LineupError(
                f"scores' last axis must be the {self.n_players} players the plan was "
                f"built for, got {arr.shape}"
            )
        if available is not None:
            mask = np.asarray(available, dtype=bool)
            arr = np.where(mask, arr, np.asarray(-np.inf, dtype=arr.dtype))
        lead = arr.shape[:-1]
        # Not `reshape(-1, n)`: an empty roster makes -1 ambiguous and numpy raises.
        rows = int(np.prod(lead, dtype=np.int64))
        return np.ascontiguousarray(arr.reshape(rows, self.n_players)), lead

    def _floor_matrix(
        self,
        floor: np.ndarray | Mapping[int, float] | float | None,
        allow_empty: bool,
        rows: int | None,
    ) -> np.ndarray:
        """Normalize any floor spelling to `(1 or rows, n_groups)` **in solve order**.

        A Mapping is keyed by slot id and needs no permuting. A positional array is read
        in `floor_slot_ids` order -- the caller's own eligibility rows -- and permuted
        into solve order here, because the solve order is an internal lexsort that a
        caller of `optimal_lineup_with_floor` has no way to see.

        `rows=None` skips the batch check, for callers that are reshaping a floor
        rather than solving with one.
        """
        if floor is None:
            fill = 0.0 if allow_empty else -np.inf
            return np.full((1, self.n_groups), fill, dtype=np.float64)
        if isinstance(floor, Mapping):
            missing = [s for s in self.group_slot_ids if s not in floor]
            if missing:
                raise LineupError(f"floor is missing starting slots {missing}")
            return np.array([[float(floor[s]) for s in self.group_slot_ids]], dtype=np.float64)
        arr = np.asarray(floor, dtype=np.float64)
        if arr.ndim == 0:
            return np.full((1, self.n_groups), float(arr), dtype=np.float64)
        if arr.shape[-1] != self.n_groups:
            raise LineupError(
                f"floor's last axis must be the {self.n_groups} slot groups, got {arr.shape}"
            )
        lead = int(np.prod(arr.shape[:-1], dtype=np.int64))
        flat = arr.reshape(lead, self.n_groups)
        if rows is not None and lead not in (1, rows):
            raise LineupError(f"floor has {lead} leading entries for {rows} score rows")
        return np.ascontiguousarray(flat[:, self._floor_perm])

    def _check_monotone(self, floors: np.ndarray) -> None:
        """A slot that accepts everything a narrower one does must not have a lower floor.

        This is the one hypothesis of the exactness proof a caller can actually break,
        and breaking it costs real points: with a dedicated-RB floor of 9.5, a FLEX
        floor of 0 and an RB pair of (10.0, 0.1), greedy returns 10.1 where the optimum
        is 19.5. It is also physically impossible -- whatever free agent the RB slot
        would stream, the FLEX could stream too -- so it means the caller computed the
        floors wrong, and `monotone_floor` is the fix.

        The check deliberately covers *equal* eligible sets too, which is not a nicety:
        the randomized oracle found greedy losing 3.4 points on a roster with no kicker,
        where a WR slot and a WR/K slot collapse to the same players and the WR slot's
        higher floor was filled first. Greedy burned the 20-point receiver on the slot
        that gained 5 from him instead of the one that gained 11.
        """
        for a, b in self._nested_pairs:
            if np.any(floors[:, a] > floors[:, b]):
                relation = (
                    "starts the same players as"
                    if self._eligible[a].sum() == self._eligible[b].sum()
                    else "is a subset of"
                )
                raise LineupError(
                    f"slot {self.group_slot_ids[a]} {relation} slot "
                    f"{self.group_slot_ids[b]} but has the higher free-agent floor; the "
                    "wider slot can always stream the narrower slot's free agent. Fix "
                    "the floors or pass them through monotone_floor()."
                )

    # -- the greedy kernel -------------------------------------------------------------

    def _greedy(
        self, eff: np.ndarray, floors: np.ndarray, assignment: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        rows = eff.shape[0]
        pool_scores, pool_ids = _sorted_pools(eff, self._class_players, with_ids=assignment)
        # How deep into each class's descending queue we have already drawn. This is
        # the entire state of the algorithm, and it is per-row, which is what lets the
        # flex pick a different position in every simulation without a Python loop.
        cursor = np.zeros((rows, len(self._class_players)), dtype=np.intp)
        total = np.zeros(rows, dtype=np.float64)
        assign = np.full((rows, self.n_slots), -1, dtype=np.int32) if assignment else None
        col = 0

        for g, count in enumerate(self.group_counts):
            classes = self._group_classes[g]
            f = floors[:, g : g + 1]
            if classes.size == 0:
                total += float(count) * floors[:, g]
            elif classes.size == 1:
                # A dedicated slot draws a contiguous prefix of one queue, so the whole
                # group resolves in one gather with no per-instance work.
                c = int(classes[0])
                idx = np.minimum(
                    cursor[:, c : c + 1] + np.arange(count), pool_scores[c].shape[1] - 1
                )
                vals = np.take_along_axis(pool_scores[c], idx, axis=1)
                use = vals > f
                total += np.where(use, vals, f).sum(axis=1)
                cursor[:, c] += use.sum(axis=1)
                if assign is not None:
                    ids = np.take_along_axis(pool_ids[c], idx, axis=1)
                    assign[:, col : col + count] = np.where(use, ids, -1)
            else:
                for j in range(count):
                    heads = np.concatenate(
                        [
                            np.take_along_axis(pool_scores[c], cursor[:, c : c + 1], axis=1)
                            for c in classes
                        ],
                        axis=1,
                    )
                    pick = np.argmax(heads, axis=1)
                    best = np.take_along_axis(heads, pick[:, None], axis=1)
                    use = best > f
                    total += np.where(use, best, f)[:, 0]
                    if assign is not None:
                        ids = np.concatenate(
                            [
                                np.take_along_axis(pool_ids[c], cursor[:, c : c + 1], axis=1)
                                for c in classes
                            ],
                            axis=1,
                        )
                        chosen = np.take_along_axis(ids, pick[:, None], axis=1)[:, 0]
                        assign[:, col + j] = np.where(use[:, 0], chosen, -1)
                    taken = use[:, 0]
                    for k, c in enumerate(classes):
                        cursor[:, c] += (pick == k) & taken
            col += count
        return total, assign

    # -- the exact fallback ------------------------------------------------------------

    def _exact(
        self, eff: np.ndarray, floors: np.ndarray, assignment: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Assignment-problem optimum, one Hungarian solve per row.

        Correct for any eligibility structure, and about thirty-five times slower than
        greedy -- measured warm on a 9-slot, 16-man roster: 10-13us a row against 0.3us
        -- so it exists to be the oracle in the tests and the fallback for a league whose
        slots genuinely cross. Slots are the rows; the columns are the players followed
        by one private "leave empty" column per slot carrying that slot's floor. A row at
        a time is the honest cost: `linear_sum_assignment` has no batched form, so a
        14-week 12-team 2,000-sim league costs four or five seconds here against 0.1.

        Time the *second* call, not the first: `scipy.optimize` is imported lazily inside
        this function so `sim.lineup` stays importable without it, and on a 400-row batch
        that one-time import is 130ms -- 330us a row, which is where the "97us a row"
        this docstring used to claim came from.
        """
        from scipy.optimize import linear_sum_assignment

        rows, players = eff.shape
        n_slots = self.n_slots
        slot_rows = np.repeat(self._eligible, self.group_counts, axis=0)
        slot_floor = np.repeat(floors, self.group_counts, axis=1)
        total = np.empty(rows, dtype=np.float64)
        assign = np.full((rows, n_slots), -1, dtype=np.int32)
        cost = np.empty((n_slots, players + n_slots), dtype=np.float64)
        diag = np.arange(n_slots)

        # One trailing zero so an empty slot can be gathered without a branch; an empty
        # roster then still has something to index, which `reshape(-1, 0)` taught us to
        # care about.
        padded = np.empty(players + 1, dtype=np.float64)
        padded[players] = 0.0

        for r in range(rows):
            gain = np.where(slot_rows, eff[r][None, :], -np.inf)
            np.negative(gain, out=cost[:, :players])
            cost[:, :players][~np.isfinite(cost[:, :players])] = _FORBIDDEN
            cost[:, players:] = _FORBIDDEN
            f = slot_floor[r % slot_floor.shape[0]]
            cost[diag, players + diag] = np.where(np.isfinite(f), -f, _UNFILLABLE)
            _, chosen = linear_sum_assignment(cost)
            picked = np.where(chosen < players, chosen, -1)
            padded[:players] = eff[r]
            # The LP is indifferent between a player who exactly ties his slot's floor
            # and leaving the slot empty, and left to itself it fills every slot it can
            # -- so a bye (`WeeklyOutlook.zeroed()`, score 0, floor 0) came back
            # *started*, contradicting the module's rule and disagreeing with greedy on
            # an assignment `sim/season.py` then scores against a different tensor.
            # Unassigning a tie cannot lose points: the LP optimum guarantees no player
            # was placed strictly below his floor, so max(score, floor) == floor here.
            gained = padded[np.where(picked >= 0, picked, players)]
            picked = np.where((picked >= 0) & (gained > f), picked, -1)
            assign[r] = picked
            filled = picked >= 0
            # The total is recomputed from the assignment rather than read off the LP,
            # because the LP works in finite sentinels and the floor may be -inf.
            total[r] = np.where(filled, padded[np.where(filled, picked, players)], f).sum()
        return total, assign if assignment else None


def _equivalence_classes(elig: np.ndarray) -> tuple[np.ndarray, ...]:
    """Players grouped by identical eligibility columns; the never-startable are dropped.

    Two players who can fill exactly the same slots are interchangeable, so the solver
    only ever needs one descending queue per class. On a normal roster the classes come
    out as the positions, but they are read off the matrix rather than assumed, which is
    what keeps a K-less roster or a position nobody rosters from needing a special case.
    """
    if elig.size == 0:
        return ()
    buckets: dict[bytes, list[int]] = {}
    for p in range(elig.shape[1]):
        column = elig[:, p]
        if not column.any():
            continue
        buckets.setdefault(column.tobytes(), []).append(p)
    return tuple(np.array(v, dtype=np.intp) for v in buckets.values())


def _sorted_pools(
    eff: np.ndarray, classes: Sequence[np.ndarray], *, with_ids: bool
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per class, scores sorted descending plus the player index that produced each.

    Each queue carries one trailing `-inf` sentinel so an exhausted class can be read
    without a bounds check: `-inf` never beats a floor, so the cursor stops on its own.

    This sort is the single largest cost in the solver, so the index half is skipped
    when the caller only wants totals: measured on 336k rows of a 16-man roster, the
    sort alone runs in 49ms against 99ms for `argsort` plus the gather, and a candidate
    search that never looks at *which* player started pays the smaller one.

    Both branches must put NaN at the *back*. `np.sort(...)[::-1]` puts it at the front
    -- numpy sorts NaN high -- where it fails `> floor`, stops the cursor, and strands
    every player behind it: one NaN receiver cost 11 points of a 115-point lineup and
    only on the totals-only path, so the two paths silently disagreed. Negating twice
    keeps NaN last, matches `argsort(-x)`, matches the exact solver (which walls NaN off
    as non-finite), and is marginally faster than reversing a view besides.
    """
    rows = eff.shape[0]
    scores: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    pad_score = np.full((rows, 1), -np.inf, dtype=eff.dtype)
    pad_id = np.full((rows, 1), -1, dtype=np.int32)
    for players in classes:
        sub = eff[:, players]
        if with_ids:
            # Stable so equal scores resolve to the earlier roster slot every run; the
            # total is unaffected either way, but a lineup that reshuffles between
            # identical inputs is impossible to diff.
            order = np.argsort(-sub, axis=1, kind="stable")
            ranked = np.take_along_axis(sub, order, axis=1)
            ids.append(np.concatenate([players.astype(np.int32)[order], pad_id], axis=1))
        else:
            ranked = -np.sort(-sub, axis=1)
        scores.append(np.concatenate([ranked, pad_score], axis=1))
    return scores, ids


# --------------------------------------------------------------------------------------
# Front doors
# --------------------------------------------------------------------------------------


def optimal_lineup(
    scores: np.ndarray,
    eligibility: np.ndarray | Sequence[Sequence[bool]],
    slot_counts: Sequence[int] | np.ndarray,
    *,
    slot_ids: Sequence[int] | None = None,
    available: np.ndarray | None = None,
    allow_empty: bool = True,
    method: Method = "auto",
    assignment: bool = True,
) -> LineupResult:
    """Best lineup for a `[..., player]` score tensor. Empty slots score zero.

    Convenience over `LineupPlan.build(...).solve(...)`; build the plan once and reuse
    it when the same league is solved repeatedly, which is every hot path.
    """
    plan = LineupPlan.build(eligibility, slot_counts, slot_ids=slot_ids)
    return plan.solve(
        scores, available=available, allow_empty=allow_empty, method=method, assignment=assignment
    )


def optimal_lineup_with_floor(
    scores: np.ndarray,
    eligibility: np.ndarray | Sequence[Sequence[bool]],
    slot_counts: Sequence[int] | np.ndarray,
    floor: np.ndarray | Mapping[int, float] | float,
    *,
    slot_ids: Sequence[int] | None = None,
    available: np.ndarray | None = None,
    method: Method = "auto",
    assignment: bool = True,
) -> LineupResult:
    """Best lineup where each slot falls back to the best free agent it could stream.

    This is what makes bench evaluation mean anything. Under a plain optimum a fourth
    receiver adds value in every week the top three blank; against a floor he adds
    value only when he beats what the waiver wire would have given you for free, which
    is the actual question a drop decision asks.

    `floor` is per slot *group*, and must not decrease as slots widen -- see
    `LineupPlan._check_monotone` and `monotone_floor`. A Mapping is keyed by slot id; a
    positional array lines up with the rows of `eligibility` (zero-count rows dropped),
    which is the only order a caller of this front door can see.
    """
    plan = LineupPlan.build(eligibility, slot_counts, slot_ids=slot_ids)
    return plan.solve(
        scores, floor=floor, available=available, method=method, assignment=assignment
    )


def monotone_floor(plan: LineupPlan, floor: np.ndarray | Mapping[int, float] | float) -> np.ndarray:
    """Raise every slot's floor to the best floor any slot it contains could stream.

    A FLEX can start the running back the RB slot would have streamed, so its floor is
    at least the RB slot's. Applying this makes any floor vector admissible, at the
    cost of letting two slots claim the same notional free agent -- acceptable because
    the wire is deep, and the reason the raw floors are validated rather than silently
    fixed inside `solve`.

    Takes and returns a vector in `plan.floor_slot_ids` order, so the result feeds
    straight back into `solve(floor=...)`.
    """
    floors = plan._floor_matrix(floor, True, None).copy()
    # `_nested_pairs` comes out ordered by its first index, so by the time any pair
    # (a, b) is reached every lift into `a` has already landed and one pass reaches the
    # transitive closure. That is worth stating because getting it wrong would leave a
    # floor that still fails `_check_monotone` and would look like a solver bug.
    for a, b in plan._nested_pairs:
        np.maximum(floors[:, b], floors[:, a], out=floors[:, b])
    inverse = np.empty_like(plan._floor_perm)
    inverse[plan._floor_perm] = np.arange(plan.n_groups, dtype=np.intp)
    return floors[:, inverse]


# --------------------------------------------------------------------------------------
# Building a plan from a league
# --------------------------------------------------------------------------------------


def plan_from_slots(
    slot_counts: Mapping[int, int],
    slot_eligibility: Mapping[int, Iterable[int]],
    player_positions: Sequence[int] | np.ndarray,
) -> LineupPlan:
    """Compile ESPN's `lineupSlotCounts` and slot eligibility against a roster.

    `slot_counts` and `slot_eligibility` are keyed by **lineupSlotId**;
    `player_positions` are **defaultPositionIds**. The two spaces collide at 4 and 15,
    so mixing them here yields a plausible, silently wrong lineup -- see
    espn/constants.py. Bench, IR and invalid slots are dropped.
    """
    from ..espn.scoring import NON_STARTING_SLOTS

    positions = np.asarray(player_positions, dtype=np.int64)
    if positions.ndim != 1:
        raise LineupError(f"player_positions must be 1-D, got {positions.shape}")
    slots = sorted(s for s, n in slot_counts.items() if n > 0 and s not in NON_STARTING_SLOTS)
    missing = [s for s in slots if s not in slot_eligibility]
    if missing:
        raise LineupError(f"no eligibility for starting slots {missing}")
    matrix = np.zeros((len(slots), positions.shape[0]), dtype=bool)
    for row, slot in enumerate(slots):
        eligible = np.fromiter(slot_eligibility[slot], dtype=np.int64)
        matrix[row] = np.isin(positions, eligible)
    counts = np.array([slot_counts[s] for s in slots], dtype=np.int64)
    return LineupPlan.build(matrix, counts, slot_ids=slots)


def plan_from_context(
    ctx: LeagueContext, player_positions: Sequence[int] | np.ndarray
) -> LineupPlan:
    """`plan_from_slots` against a `LeagueContext`'s own starting slots."""
    return plan_from_slots(ctx.starting_slots, ctx.slot_eligibility, player_positions)
