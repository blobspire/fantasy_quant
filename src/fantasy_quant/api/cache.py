"""Server-side memoization, because the cheapest honest answer in this system costs 3s.

`pipeline.build` draws a `(sims, weeks, players)` tensor per league and the decision
surfaces run search on top of it; a waiver board is seconds and a trade search is tens of
seconds. A dashboard that recomputed on every paint would be unusable, and one that
served whatever it computed at breakfast without saying so would be worse: the numbers
here move when a manager sets a lineup or a claim processes, and a stale probability
looks exactly like a fresh one.

So this module keeps three things and nothing else.

**An answer, with the wall clock it was computed at.** `Entry.staleness()` is the only
place `computed_at` and `stale_seconds` are produced, and every API response carries it.
The age is measured on a *monotonic* clock and the timestamp is reported on the wall
clock, which is deliberate: a laptop that sleeps for an hour or has its clock stepped by
NTP must not report a negative age or a cached-in-the-future entry.

**A key that is the whole question.** `(league_id, season, endpoint, sims)` plus the
parameters that change the answer -- a `limit`, a `week`, a `min_gain`. Leaving those out
of the key is the classic memoization bug in this shape of tool: the user asks for the
top twenty waiver claims, gets the six-row answer cached by the weekly report, and has no
way to see that the list was truncated somewhere other than where he asked.

**An explicit invalidate.** No TTL on an analysis, ever. A number that quietly changed
because a cache line aged out is indistinguishable from a number that changed because the
league did, and this tool is used to make decisions worth money. `max_age` exists on
`peek` and is used only for the two things that are genuinely a liveness check rather
than an analysis -- whether ESPN answers, and whether the session cookies still work.

Nothing here knows what an endpoint is or what a league is; it stores values under keys.
The single-flight lock lives here too, because it is the same concern: two browser tabs
asking the same cold question must produce one computation, not two.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "Cache",
    "Entry",
    "Key",
    "KeyedLocks",
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _token(value: Any) -> str:
    """One parameter value as a stable string, so equal questions produce equal keys.

    Booleans are spelled out rather than left as `True`/`False`, and sequences are
    joined in the order given -- `--skip odds --skip trades` is not the same request as
    `--skip trades`, and sorting them here would make it look like one.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_token(v) for v in value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


@dataclass(frozen=True, slots=True, order=True)
class Key:
    """What a cached answer is an answer *to*.

    The four fields the dashboard keys on are named, and everything else that changes
    the answer goes in `params`. `None` for `league_id` is the cross-league case
    (`portfolio`, `queue`, `leagues`, `health`) rather than a missing value.
    """

    endpoint: str
    league_id: int | None = None
    season: int | None = None
    sims: int | None = None
    params: tuple[tuple[str, str], ...] = ()

    @classmethod
    def of(
        cls,
        endpoint: str,
        *,
        league_id: int | None = None,
        season: int | None = None,
        sims: int | None = None,
        **params: Any,
    ) -> Key:
        """A key from the values a request actually carried; `None` parameters drop out.

        Dropping `None` matters for cache hits: a handler that defaults `week=None` and
        one that was given the current week explicitly are asking the same question, and
        keying on the literal `"None"` string would answer it twice.
        """
        return cls(
            endpoint=str(endpoint),
            league_id=None if league_id is None else int(league_id),
            season=None if season is None else int(season),
            sims=None if sims is None else int(sims),
            params=tuple(sorted((str(k), _token(v)) for k, v in params.items() if v is not None)),
        )

    def as_dict(self) -> dict[str, Any]:
        """The key as JSON, for `/api/health`'s cache listing."""
        return {
            "endpoint": self.endpoint,
            "league_id": self.league_id,
            "season": self.season,
            "sims": self.sims,
            "params": dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class Entry:
    """One computed answer and everything the UI needs to distrust it.

    `ok=False` is a first-class entry rather than an absence. A league whose cookies
    expired fails in thirty seconds of HTTP timeouts, and retrying that on every paint
    turns one broken league into a broken dashboard; the caller decides how long a
    failure is allowed to stand by passing `max_age` to `peek`.
    """

    key: Key
    value: Any
    ok: bool
    computed_at: datetime
    #: Monotonic reading at computation time. Ages are measured against this and never
    #: against `computed_at`, so a clock step cannot produce a negative staleness.
    monotonic: float
    compute_seconds: float

    def age(self, *, clock: Callable[[], float] = time.monotonic) -> float:
        return max(clock() - self.monotonic, 0.0)

    def staleness(self, *, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
        """The three fields every API response carries. The only producer of them."""
        return {
            "computed_at": self.computed_at.isoformat(timespec="milliseconds"),
            "stale_seconds": round(self.age(clock=clock), 3),
            "compute_seconds": round(self.compute_seconds, 3),
        }


@dataclass(slots=True)
class Cache:
    """A dict of `Key -> Entry` with ages, an explicit invalidate, and injectable clocks.

    Both clocks are injectable because staleness is a *feature* here and a feature has to
    be testable without sleeping. `clock` is monotonic seconds and `wallclock` is the
    timestamp the UI prints.

    Not thread-safe by design and not thread-shared by construction: every mutation
    happens on the event loop thread, and the expensive work is what gets pushed to a
    worker thread, not the bookkeeping around it.
    """

    clock: Callable[[], float] = time.monotonic
    wallclock: Callable[[], datetime] = _utcnow
    _entries: dict[Key, Entry] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Key) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[Entry]:
        return iter(list(self._entries.values()))

    def peek(self, key: Key, *, max_age: float | None = None) -> Entry | None:
        """The stored answer, or `None`. `max_age` expires it; the default never does.

        `peek` does not compute and does not evict on a miss. Expiry by `max_age` leaves
        the entry in place so a caller who wants a stale-but-present answer -- an
        auth probe that ESPN is currently refusing to re-answer -- can still find it.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        if max_age is not None and entry.age(clock=self.clock) > max_age:
            return None
        return entry

    def entry(
        self, key: Key, value: Any, *, ok: bool = True, compute_seconds: float = 0.0
    ) -> Entry:
        """An entry stamped with both clocks and **not** stored.

        For the one case where an answer has to be returned but must not be served
        again: a computation that finished after the refresh which discarded the
        workspace it was reading. The caller who waited for it still gets it; the next
        caller gets a fresh one.
        """
        return Entry(
            key=key,
            value=value,
            ok=ok,
            computed_at=self.wallclock(),
            monotonic=self.clock(),
            compute_seconds=float(compute_seconds),
        )

    def put(self, key: Key, value: Any, *, ok: bool = True, compute_seconds: float = 0.0) -> Entry:
        entry = self.entry(key, value, ok=ok, compute_seconds=compute_seconds)
        self._entries[key] = entry
        return entry

    def invalidate(
        self,
        *,
        league_id: int | None = None,
        endpoint: str | None = None,
        endpoints: Sequence[str] | None = None,
    ) -> int:
        """Drop matching entries and say how many. No arguments means everything.

        `league_id` deliberately also drops the cross-league entries (`league_id is
        None`): the portfolio and the action queue are built out of every league, so
        refreshing one league and leaving the queue that ranks it in place would show
        the user a queue that disagrees with the league page it came from.
        """
        wanted = set(endpoints or ())
        if endpoint:
            wanted.add(endpoint)
        doomed = [
            k
            for k in self._entries
            if (league_id is None or k.league_id in (league_id, None))
            and (not wanted or k.endpoint in wanted)
        ]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def clear(self) -> int:
        n = len(self._entries)
        self._entries.clear()
        return n

    def stats(self) -> dict[str, Any]:
        """What is currently held, oldest first. Values are never included."""
        rows = sorted(self._entries.values(), key=lambda e: -e.age(clock=self.clock))
        return {
            "entries": len(rows),
            "ok": sum(1 for e in rows if e.ok),
            "failed": sum(1 for e in rows if not e.ok),
            "oldest_seconds": round(rows[0].age(clock=self.clock), 3) if rows else None,
            "keys": [
                {**e.key.as_dict(), "ok": e.ok, **e.staleness(clock=self.clock)} for e in rows
            ],
        }


class KeyedLocks:
    """One `asyncio.Lock` per key, made on demand. Single-flight for cold computations.

    Two tabs open on the same league produce two identical cold requests a millisecond
    apart. Without this they both build the same three-second simulation, and worse, both
    mutate the same `report.Workspace` while doing it. With it the second one waits and
    then finds the cache warm.

    Safe to create locks lazily without a mutex only because every caller is on the event
    loop thread: `asyncio` does not preempt between the `get` and the `setdefault`. The
    dictionary is bounded by the number of leagues plus a handful of cross-league keys,
    so nothing here is ever evicted.
    """

    __slots__ = ("_locks",)

    def __init__(self) -> None:
        self._locks: dict[Hashable, asyncio.Lock] = {}

    def __call__(self, key: Hashable) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def __len__(self) -> int:
        return len(self._locks)

    def held(self) -> tuple[Hashable, ...]:
        """Keys currently computing. `/api/health` reports these as work in flight."""
        return tuple(k for k, lock in self._locks.items() if lock.locked())
