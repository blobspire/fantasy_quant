"""What an unfilled starting slot can actually stream off the wire.

Extracted so there is exactly ONE answer to this question. There were two, and
they disagreed: `title.streaming_replacement` read the VOLS demand rank -- WR28 in
a 12-team league -- while `waivers.wire_floor` read the genuinely unrostered pool.
In a league that carries five receivers WR28 is ROSTERED, so the first was measuring
the bottom of a roster and calling it the wire. Measured on a real league it claimed
an empty WR slot streams 8.73 points a week against a true best-available of 6.48.

That is not a tidiness problem. Every bench receiver projecting below the phantom
floor contributed exactly zero, so swapping one for another returned +0.000pp +/-
0.000 on bit-identical seasons -- the option value that justifies carrying a bench
at all was silently zero.

**Do not confuse this with `valuation.replacement_levels`.** That is the VORP
baseline: the N_q-th best player in the whole universe, which answers "how much
better than a freely available body is this player" for valuation. This answers
"if this seat were empty on Sunday, what would actually fill it", and the only
players that can fill it are the ones nobody rosters. Both are correct; they are
different questions and the bug was using one where the other belongs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from ..core import PlayerOutlook, WireLevel

#: Second-best rather than best, because the wire is contested: by the time the
#: seat is empty the obvious name is usually gone.
DEFAULT_WIRE_DEPTH = 2


def wire_floor(
    outlooks: Sequence[PlayerOutlook],
    rostered: Iterable[int],
    weeks: Sequence[int],
    slot_eligibility: Mapping[int, frozenset[int]],
    *,
    depth: int = DEFAULT_WIRE_DEPTH,
) -> dict[int, float]:
    """slotId -> the mean weekly points an unfilled slot would stream off this wire.

    The mean alone, for callers that solve a lineup: the start/sit decision is made on
    expectations, so this is the right number to compare a rostered player against.
    `wire_levels` carries the spread as well, for crediting what the seat actually scored.
    """
    return {
        slot: level.mean
        for slot, level in wire_levels(
            outlooks, rostered, weeks, slot_eligibility, depth=depth
        ).items()
    }


def wire_levels(
    outlooks: Sequence[PlayerOutlook],
    rostered: Iterable[int],
    weeks: Sequence[int],
    slot_eligibility: Mapping[int, frozenset[int]],
    *,
    depth: int = DEFAULT_WIRE_DEPTH,
) -> dict[int, WireLevel]:
    """slotId -> the distribution of what an empty seat streams off this wire.

    **The `depth`-th best is taken WEEK BY WEEK, not once for the season.** An empty
    slot is not "the one free agent with the best season total, started seventeen
    times" -- it is "whoever is best on the wire *that week*", and at a streamed
    position the identity changes every week. Measured on the user's three leagues,
    the season-total reading understates the D/ST floor by 1.1-1.3 points a week and
    the QB floor by the same: about 19-23 points of rest-of-season score per slot.

    **The SPREAD is the outcome dispersion of the body who fills the seat, not the
    week-to-week variation of his projection.** Those are different numbers and only
    the first is what a manager experiences. So for each week we find the player who
    is k-th best BY PROJECTION -- the one actually streamed that week -- and take his
    own `sd` and `p_zero`, which already carry the chance he simply blanks.

    Why the spread has to exist at all: crediting an empty seat a constant gave it zero
    variance, and on a live league 15.7% of all slot-weeks fall back to that constant --
    70.6% of the WR3 slot. Measured here, the spread is nearly as large as the mean
    (WR 6.40 +/- 5.33), so treating it as a point value discarded most of what a
    streamed seat actually does. Variance is also what decides whether a team near the
    playoff cut should seek it or avoid it, so the bias landed on the decisions that
    matter most.

    The result is a per-week average, which is what `sim/season.py` wants -- one scalar
    triple per slot. `monotone_floor` then lifts a flex to at least the floors of the
    slots nested inside it, and it lifts the MEAN only: the lift is a statement about
    which body a wider slot could reach, and applying it to a sample would let a manager
    gain points by leaving the FLEX empty, which is hindsight through the back door.
    """
    owned = {int(p) for p in rostered}
    span = tuple(int(w) for w in weeks)
    means: dict[int, list[list[float]]] = {}
    sds: dict[int, list[list[float]]] = {}
    zeros: dict[int, list[list[float]]] = {}
    for o in outlooks:
        if o.player_id in owned:
            continue
        row = [o.weeks[w].mean if w in o.weeks else 0.0 for w in span]
        if not row or sum(row) <= 0.0:
            continue
        means.setdefault(o.position_id, []).append(row)
        sds.setdefault(o.position_id, []).append(
            [o.weeks[w].sd if w in o.weeks else 0.0 for w in span]
        )
        zeros.setdefault(o.position_id, []).append(
            [o.weeks[w].p_zero if w in o.weeks else 1.0 for w in span]
        )

    k = max(int(depth), 1)
    out: dict[int, WireLevel] = {}
    for slot, eligible in slot_eligibility.items():
        rows = [r for pos in eligible for r in means.get(pos, ())]
        if not rows:
            out[slot] = WireLevel(0.0, 0.0, 1.0)
            continue
        mu = np.asarray(rows, dtype=np.float64)
        sd = np.asarray([r for pos in eligible for r in sds.get(pos, ())], dtype=np.float64)
        pz = np.asarray([r for pos in eligible for r in zeros.get(pos, ())], dtype=np.float64)
        # Rank by projection down each week's column independently, then take THAT
        # player's own spread and hurdle -- not the k-th largest of each, which would
        # pair one body's mean with another body's variance.
        pick = kth_best_index(mu, k)
        cols = np.arange(mu.shape[1])
        out[slot] = WireLevel(
            mean=float(mu[pick, cols].mean()),
            sd=float(sd[pick, cols].mean()),
            p_zero=float(pz[pick, cols].mean()),
        )
    return out


def kth_best_index(scores: np.ndarray, depth: int) -> np.ndarray:
    """`(weeks,)` row index of the `depth`-th best candidate in each week's column.

    Extracted so "the k-th best body on the wire, week by week" has ONE definition even
    when two callers rank on different quantities. `wire_levels` ranks on the calibrated
    projection, because that is the scale the simulator pays an empty seat in.
    `decide/streaming.wire_floor` ranks on `StreamGrid.value` -- the matchup model's
    conditional expectation -- because that is the scale its optimiser compares a
    candidate against, and a floor in the wrong units is a floor that benches candidates
    it cannot actually beat.

    Indices rather than values, because the caller usually wants something else off the
    same body: `wire_levels` takes his spread and hurdle, and pairing one body's mean with
    another body's variance is the bug this shape exists to prevent.
    """
    k = max(int(depth), 1)
    return np.argsort(-scores, axis=0)[min(k, scores.shape[0]) - 1]


def all_rostered(state) -> set[int]:
    """Every player id on any franchise in the league."""
    owned: set[int] = set()
    for franchise in state.franchises:
        owned.update(int(p) for p in franchise.player_ids)
    return owned


#: Re-exported: `WireLevel` is defined in core.py so `sim` can use it too.
__all__ = [
    "DEFAULT_WIRE_DEPTH",
    "WireLevel",
    "all_rostered",
    "kth_best_index",
    "wire_floor",
    "wire_levels",
]
