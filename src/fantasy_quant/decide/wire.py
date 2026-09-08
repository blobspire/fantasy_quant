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

from ..core import PlayerOutlook

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
    """slotId -> the weekly points an unfilled slot would stream off this wire.

    **The `depth`-th best is taken WEEK BY WEEK, not once for the season.** The two
    are not the same number and the gap is not small. An empty slot is not "the one
    free agent with the best season total, started seventeen times" -- it is
    "whoever is best on the wire *that week*", and at a streamed position the
    identity changes every week. Measured on the user's three leagues, the
    season-total reading understates the D/ST floor by 1.1-1.3 points a week and the
    QB floor by 1.1-1.3: about 19-23 points of rest-of-season score per slot. Under
    the season-total floor every board came back dominated by "add a second D/ST",
    and the entire gain was the difference between a real streaming slot and a floor
    set at one fixed defense's season average.

    The result is a per-week mean, which is what `sim/season.py` wants for
    `replacement=` -- `_franchise_scores` carries one scalar per slot, so the
    week-to-week variation has to be averaged out here rather than passed through.
    `monotone_floor` then lifts a flex to at least the floors of the slots nested
    inside it, so the values here need not be consistent by construction.
    """
    owned = {int(p) for p in rostered}
    span = tuple(int(w) for w in weeks)
    per_position: dict[int, list[list[float]]] = {}
    for o in outlooks:
        if o.player_id in owned:
            continue
        row = [o.weeks[w].mean if w in o.weeks else 0.0 for w in span]
        if not row or sum(row) <= 0.0:
            continue
        per_position.setdefault(o.position_id, []).append(row)

    k = max(int(depth), 1)
    out: dict[int, float] = {}
    for slot, eligible in slot_eligibility.items():
        rows = [r for pos in eligible for r in per_position.get(pos, ())]
        if not rows:
            out[slot] = 0.0
            continue
        # (players, weeks) sorted best-first down each week's column independently:
        # the streamer is chosen per week, so the k-th best is a different player
        # each week.
        grid = -np.sort(-np.asarray(rows, dtype=np.float64), axis=0)
        out[slot] = float(grid[min(k, grid.shape[0]) - 1].mean())
    return out


def all_rostered(state) -> set[int]:
    """Every player id on any franchise in the league."""
    owned: set[int] = set()
    for franchise in state.franchises:
        owned.update(int(p) for p in franchise.player_ids)
    return owned
