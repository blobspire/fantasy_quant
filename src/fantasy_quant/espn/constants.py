"""ESPN's own constant tables, generated rather than hand-maintained.

    GET {BASE}/seasons/{year}?view=chui_default_platformsettings

is unauthenticated and returns everything the rest of the codebase would otherwise
hard-code: all 235 statIds with their abbrevs and API identifiers, the position and
lineup-slot tables (slots carry `eligiblePositions`, so flex/superflex is *derived*,
never assumed), the 33 pro teams with bye weeks, and every enum ESPN uses. It is
cached to disk per season because a completed season's tables never change.

Everything here exists to prevent one class of bug.

**The two ID spaces collide.** `defaultPositionId` 4 is TE and 15 is TQB, while
`lineupSlotId` 4 is WR and 15 is DP. `lineupSlotCounts` is keyed by slot id;
`pointsOverrides` and `positionLimits` are keyed by position id. Passing one where
the other belongs produces no error and no exception -- just a wrong score for the
rest of the season. So `PositionId`, `SlotId` and `ProTeamId` are distinct wrapper
types here, not `int`, and the lookup tables reject the wrong one at runtime. You
have to say which space you are in; that is the whole point.

Two things measured against the live payload that differ from what you would assume:

* `types` / `typeNames` live under `settings`, and `typeNames` is a bare list of
  names in display order whose index is **not** the enum id -- `transactionTypes`
  starts at id 1 and `transactionStatusTypes` starts at -1. Only `types` carries
  explicit ids, so only `types` is parsed.
* `proTeams[].abbrev` changed case between seasons: 2022 and 2025 return `Atl`,
  `Bal`, `Hou`; 2026 returns `ATL`, `BAL`, `HOU`. Abbrevs are upper-cased on parse,
  because the D/ST join in the ID spine is keyed on nflverse's uppercase team codes
  and a case-sensitive join would silently drop every historical team defense.

Not a guard against multi-counting: neither `derived` nor `pointsScoringEligible`
identifies the pre-computed buckets. `derived` marks per-game rate stats (22 PYPG,
40 RYPG, 61 REYPG) and `pointsScoringEligible` is very nearly its inverse, so the
"every N receiving yards" buckets 47-52 come back `derived: false` and
`pointsScoringEligible: true` -- they look exactly like statId 42 does. The rule
from RESEARCH.md stands unchanged: apply only statIds that carry a `scoringItem`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .client import EspnClient, EspnError
from .endpoints import platform_settings_url

log = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("data/reference")

# ESPN skips these two. Nothing is ever assigned to them; seeing one means an
# upstream parse went wrong, so the lookups say so rather than returning a blank.
MISSING_PRO_TEAM_IDS = frozenset({31, 32})

# Hand-verified floor for the one table we cannot tolerate being unavailable: a
# team abbrev is the join key for every external data source we pull. Live values
# win when the payload is reachable; a network test reconciles the two so drift
# is caught rather than discovered mid-season.
FALLBACK_PRO_TEAM_ABBREV: Mapping[int, str] = MappingProxyType(
    {
        0: "FA",
        1: "ATL",
        2: "BUF",
        3: "CHI",
        4: "CIN",
        5: "CLE",
        6: "DAL",
        7: "DEN",
        8: "DET",
        9: "GB",
        10: "TEN",
        11: "IND",
        12: "KC",
        13: "LV",
        14: "LAR",
        15: "MIA",
        16: "MIN",
        17: "NE",
        18: "NO",
        19: "NYG",
        20: "NYJ",
        21: "PHI",
        22: "ARI",
        23: "PIT",
        24: "LAC",
        25: "SF",
        26: "SEA",
        27: "TB",
        28: "WSH",
        29: "CAR",
        30: "JAX",
        33: "BAL",
        34: "HOU",
    }
)

# Parsed from `types`, which carries explicit ids. `typeNames` is index-ordered and
# would be off by one for every enum that does not start at zero.
_ENUM_SOURCE_KEY = "types"


class ConstantsError(RuntimeError):
    """The platform-settings payload was missing or shaped unexpectedly."""


# --------------------------------------------------------------------------- ids


@dataclass(frozen=True, slots=True, order=True)
class PositionId:
    """A `defaultPositionId`: 1 QB, 2 RB, 3 WR, 4 TE, 5 K, 16 D/ST, 18 BE.

    Deliberately not an int, and deliberately not comparable to `SlotId`. This is
    the key space for `pointsOverrides` and `positionLimits`.
    """

    value: int

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"PositionId({self.value})"


@dataclass(frozen=True, slots=True, order=True)
class SlotId:
    """A `lineupSlotId`: 0 QB, 2 RB, 4 WR, 6 TE, 7 OP, 20 BE, 23 FLEX.

    The key space for `lineupSlotCounts` and for a roster entry's `lineupSlotId`.
    Note 4 and 15 mean different things here than they do as a `PositionId`.
    """

    value: int

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"SlotId({self.value})"


@dataclass(frozen=True, slots=True, order=True)
class ProTeamId:
    """An NFL team id as ESPN numbers them. 0 is free agent; 31 and 32 do not exist."""

    value: int

    def __int__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"ProTeamId({self.value})"


def position_id(raw: int | str) -> PositionId:
    """Wrap a raw `defaultPositionId`. JSON object keys arrive as strings."""
    return PositionId(int(raw))


def slot_id(raw: int | str) -> SlotId:
    """Wrap a raw `lineupSlotId`. JSON object keys arrive as strings."""
    return SlotId(int(raw))


def pro_team_id(raw: int | str) -> ProTeamId:
    return ProTeamId(int(raw))


_COLLIDING = (PositionId, SlotId)


def _require(key: object, expected: type, table: str) -> None:
    """Reject an id from the wrong space. This is the guard the module exists for."""
    if isinstance(key, expected):
        return
    hint = ""
    if expected in _COLLIDING and isinstance(key, _COLLIDING):
        hint = (
            " defaultPositionId and lineupSlotId disagree at 4 and 15, so this would "
            "have resolved to something plausible and wrong. Convert deliberately "
            "if you really meant it."
        )
    raise TypeError(f"{table} is keyed by {expected.__name__}, got {type(key).__name__}.{hint}")


def fallback_pro_team_abbrev(team_id: ProTeamId) -> str:
    """Team abbrev from the hand-verified table, for when ESPN is unreachable."""
    _require(team_id, ProTeamId, "the pro-team table")
    if team_id.value in MISSING_PRO_TEAM_IDS:
        raise KeyError(f"proTeamId {team_id.value} does not exist in ESPN's numbering")
    try:
        return FALLBACK_PRO_TEAM_ABBREV[team_id.value]
    except KeyError:
        raise KeyError(f"unknown proTeamId {team_id.value}") from None


# ------------------------------------------------------------------------ records


@dataclass(frozen=True, slots=True)
class Stat:
    """One entry of `statSettings.stats`. 235 of them, ids 0-234, not contiguous in meaning."""

    id: int
    abbrev: str
    description: str
    api_identifier: str | None
    display_abbrev: str | None
    # True for per-game rate stats only -- see the module docstring; this is NOT
    # the flag that identifies the pre-computed scoring buckets.
    derived: bool
    points_scoring_eligible: bool


@dataclass(frozen=True, slots=True)
class Position:
    id: PositionId
    abbrev: str
    name: str
    # ESPN maps several roster codes onto one position: RB covers FB, K covers PK.
    api_identifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LineupSlot:
    """A lineup slot and, crucially, the positions it accepts.

    `is_flex` and `is_superflex` are computed from `eligible_positions` at parse
    time. Hard-coding slot 23 and slot 7 works right up until ESPN adds a slot.
    """

    id: SlotId
    abbrev: str
    name: str
    eligible_positions: frozenset[PositionId]
    is_starter: bool
    is_bench: bool
    is_flex: bool
    is_superflex: bool

    def accepts(self, position: PositionId) -> bool:
        _require(position, PositionId, "eligible_positions")
        return position in self.eligible_positions


@dataclass(frozen=True, slots=True)
class ProTeam:
    id: ProTeamId
    abbrev: str
    location: str
    name: str
    bye_week: int | None

    @property
    def is_real(self) -> bool:
        """Id 0 is the free-agent pseudo-team, not an NFL franchise."""
        return self.id.value != 0


@dataclass(frozen=True, slots=True)
class EnumMember:
    id: int
    name: str
    abbrev: str | None
    description: str | None


# ------------------------------------------------------------------------- tables


@dataclass(frozen=True, slots=True, eq=False)
class PositionTable:
    """`PositionId` -> `Position`. Rejects a `SlotId` rather than resolving it."""

    _by_id: Mapping[PositionId, Position]
    _by_abbrev: Mapping[str, Position]

    def __getitem__(self, key: PositionId) -> Position:
        _require(key, PositionId, "the position table")
        try:
            return self._by_id[key]
        except KeyError:
            raise KeyError(f"no defaultPositionId {key.value}") from None

    def get(self, key: PositionId) -> Position | None:
        _require(key, PositionId, "the position table")
        return self._by_id.get(key)

    def abbrev(self, key: PositionId) -> str:
        return self[key].abbrev

    def by_abbrev(self, abbrev: str) -> Position:
        try:
            return self._by_abbrev[abbrev.upper()]
        except KeyError:
            raise KeyError(f"no position abbreviated {abbrev!r}") from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, PositionId) and key in self._by_id

    def __iter__(self) -> Iterator[Position]:
        return iter(sorted(self._by_id.values(), key=lambda p: p.id))

    def __len__(self) -> int:
        return len(self._by_id)


@dataclass(frozen=True, slots=True, eq=False)
class SlotTable:
    """`SlotId` -> `LineupSlot`, plus the derived flex questions."""

    _by_id: Mapping[SlotId, LineupSlot]
    _by_abbrev: Mapping[str, LineupSlot]
    _qb_position: PositionId | None

    def __getitem__(self, key: SlotId) -> LineupSlot:
        _require(key, SlotId, "the lineup-slot table")
        try:
            return self._by_id[key]
        except KeyError:
            raise KeyError(f"no lineupSlotId {key.value}") from None

    def get(self, key: SlotId) -> LineupSlot | None:
        _require(key, SlotId, "the lineup-slot table")
        return self._by_id.get(key)

    def abbrev(self, key: SlotId) -> str:
        return self[key].abbrev

    def by_abbrev(self, abbrev: str) -> LineupSlot:
        try:
            return self._by_abbrev[abbrev.upper()]
        except KeyError:
            raise KeyError(f"no lineup slot abbreviated {abbrev!r}") from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, SlotId) and key in self._by_id

    def __iter__(self) -> Iterator[LineupSlot]:
        return iter(sorted(self._by_id.values(), key=lambda s: s.id))

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def flex_slots(self) -> tuple[LineupSlot, ...]:
        """Starting slots that accept more than one position, in id order."""
        return tuple(s for s in self if s.is_flex)

    @property
    def superflex_slots(self) -> tuple[LineupSlot, ...]:
        """Flex slots that accept a quarterback."""
        return tuple(s for s in self if s.is_superflex)

    @property
    def dedicated_qb_slots(self) -> tuple[LineupSlot, ...]:
        """Starting slots that accept quarterbacks and nothing else."""
        qb = self._qb_position
        if qb is None:
            return ()
        return tuple(s for s in self if s.is_starter and s.eligible_positions == frozenset({qb}))

    def slots_accepting(self, position: PositionId) -> tuple[LineupSlot, ...]:
        """Every starting slot a player at this position can legally fill."""
        _require(position, PositionId, "eligible_positions")
        return tuple(s for s in self if s.is_starter and position in s.eligible_positions)

    def is_superflex_lineup(self, lineup_slot_counts: Mapping[str | int, int]) -> bool:
        """Does this league's `lineupSlotCounts` start more quarterbacks than one?

        Takes the raw settings mapping, whose keys are slot ids as JSON strings.
        Both routes are derived: a slot that accepts QB alongside other positions,
        or two-plus of a QB-only slot.
        """
        counts = {slot_id(k): int(v) for k, v in lineup_slot_counts.items()}
        if any(counts.get(s.id, 0) > 0 for s in self.superflex_slots):
            return True
        return any(counts.get(s.id, 0) >= 2 for s in self.dedicated_qb_slots)


@dataclass(frozen=True, slots=True, eq=False)
class ProTeamTable:
    """`ProTeamId` -> `ProTeam`, with a hand-verified abbrev fallback behind it."""

    _by_id: Mapping[ProTeamId, ProTeam]
    _by_abbrev: Mapping[str, ProTeam]

    def __getitem__(self, key: ProTeamId) -> ProTeam:
        _require(key, ProTeamId, "the pro-team table")
        try:
            return self._by_id[key]
        except KeyError:
            if key.value in MISSING_PRO_TEAM_IDS:
                raise KeyError(
                    f"proTeamId {key.value} does not exist in ESPN's numbering"
                ) from None
            raise KeyError(f"no proTeamId {key.value}") from None

    def get(self, key: ProTeamId) -> ProTeam | None:
        _require(key, ProTeamId, "the pro-team table")
        return self._by_id.get(key)

    def abbrev(self, key: ProTeamId) -> str:
        """Live abbrev, falling back to the hand-verified table if ESPN dropped one."""
        team = self.get(key)
        return team.abbrev if team else fallback_pro_team_abbrev(key)

    def by_abbrev(self, abbrev: str) -> ProTeam:
        try:
            return self._by_abbrev[abbrev.upper()]
        except KeyError:
            raise KeyError(f"no pro team abbreviated {abbrev!r}") from None

    def bye_week(self, key: ProTeamId) -> int | None:
        """The team's bye. Read a *stat row's* proTeamId for historical weeks --
        `player.proTeamId` is the current team, so a trade gives a traded player
        the wrong bye."""
        team = self[key]
        return team.bye_week

    def __contains__(self, key: object) -> bool:
        return isinstance(key, ProTeamId) and key in self._by_id

    def __iter__(self) -> Iterator[ProTeam]:
        return iter(sorted(self._by_id.values(), key=lambda t: t.id))

    def __len__(self) -> int:
        return len(self._by_id)


@dataclass(frozen=True, slots=True, eq=False)
class EnumTable:
    """One of ESPN's `types` tables, addressable by id or by name."""

    key: str
    _by_id: Mapping[int, EnumMember]
    _by_name: Mapping[str, EnumMember]

    def __getitem__(self, member_id: int) -> EnumMember:
        try:
            return self._by_id[int(member_id)]
        except KeyError:
            raise KeyError(f"no {self.key} member with id {member_id}") from None

    def name(self, member_id: int) -> str:
        return self[member_id].name

    def id(self, name: str) -> int:
        try:
            return self._by_name[name.upper()].id
        except KeyError:
            raise KeyError(f"no {self.key} member named {name!r}") from None

    def get(self, member_id: int) -> EnumMember | None:
        return self._by_id.get(int(member_id))

    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in sorted(self._by_id.values(), key=lambda m: m.id))

    def __contains__(self, member_id: object) -> bool:
        return isinstance(member_id, int) and int(member_id) in self._by_id

    def __iter__(self) -> Iterator[EnumMember]:
        return iter(sorted(self._by_id.values(), key=lambda m: m.id))

    def __len__(self) -> int:
        return len(self._by_id)


# ---------------------------------------------------------------------- container


@dataclass(frozen=True, slots=True, eq=False)
class PlatformSettings:
    """Everything `chui_default_platformsettings` gives us, for one season."""

    # Every mapping here is a read-only view. One `PlatformSettings` is memoized
    # per (root, season) and handed to every caller in the process, so a stray
    # `.stats.pop(...)` anywhere would corrupt the constants for everyone else --
    # which is precisely the class of silent breakage this module exists to stop.
    season: int
    stats: Mapping[int, Stat]
    positions: PositionTable
    slots: SlotTable
    pro_teams: ProTeamTable
    # statId -> the defaultPositionId whose scoring rules apply. Almost all of it
    # is the D/ST block; it is a position id, never a slot id.
    stat_override_position: Mapping[int, PositionId]
    enums: Mapping[str, EnumTable] = field(default_factory=dict)

    # -- stats

    def stat(self, stat_id: int) -> Stat:
        try:
            return self.stats[int(stat_id)]
        except KeyError:
            raise KeyError(f"no statId {stat_id}") from None

    def stat_abbrev(self, stat_id: int) -> str:
        return self.stat(stat_id).abbrev

    # -- enums

    def enum(self, key: str) -> EnumTable:
        try:
            return self.enums[key]
        except KeyError:
            raise KeyError(
                f"no enum {key!r} in platform settings; have {sorted(self.enums)}"
            ) from None

    @property
    def transaction_types(self) -> EnumTable:
        return self.enum("transactionTypes")

    @property
    def transaction_status_types(self) -> EnumTable:
        return self.enum("transactionStatusTypes")

    @property
    def acquisition_types(self) -> EnumTable:
        return self.enum("acquisitionTypes")

    @property
    def draft_types(self) -> EnumTable:
        return self.enum("draftTypes")

    @property
    def scoring_types(self) -> EnumTable:
        return self.enum("scoringTypes")

    @property
    def rank_types(self) -> EnumTable:
        return self.enum("rankTypes")

    @property
    def player_status_types(self) -> EnumTable:
        return self.enum("playerStatusTypes")

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], season: int | None = None
    ) -> PlatformSettings:
        """Parse the raw response body. `settings` is where everything actually lives."""
        settings = payload.get("settings")
        if not isinstance(settings, Mapping):
            raise ConstantsError(
                "platform-settings payload has no `settings` object; an unrecognized "
                f"view returns HTTP 200 with a skeleton, so check the URL. Keys: {sorted(payload)}"
            )
        body_season = _as_int(payload.get("id"))
        season = int(season) if season is not None else (body_season or 0)
        if not season:
            raise ConstantsError("platform-settings payload carries no season id")
        # A cache file is just a name on disk; nothing stops the wrong year's body
        # sitting under it. The body says which season it is, so check rather than
        # relabel 2022's tables as 2026 and quietly use the wrong byes all year.
        if body_season and body_season != season:
            raise ConstantsError(
                f"platform-settings payload is for season {body_season}, not {season}"
            )

        positions = _parse_positions(settings)
        # Superflex is defined as "accepts a quarterback", so the QB position has to
        # be looked up by abbrev rather than assumed to be id 1.
        try:
            qb: PositionId | None = positions.by_abbrev("QB").id
        except KeyError:
            log.warning("no QB position in %s platform settings; superflex undetectable", season)
            qb = None

        return cls(
            season=season,
            stats=_parse_stats(settings),
            positions=positions,
            slots=_parse_slots(settings, qb),
            pro_teams=_parse_pro_teams(settings),
            stat_override_position=MappingProxyType(
                {
                    int(k): position_id(v)
                    for k, v in (settings.get("statIdToOverridePosition") or {}).items()
                }
            ),
            enums=MappingProxyType(_parse_enums(settings)),
        )


# -------------------------------------------------------------------------- parse


def _as_int(value: object) -> int | None:
    """ESPN is loose about ints-as-strings; a non-numeric value is simply absent."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_stats(settings: Mapping[str, Any]) -> Mapping[int, Stat]:
    raw = ((settings.get("statSettings") or {}).get("stats")) or []
    if not raw:
        raise ConstantsError("platform settings carried no statSettings.stats")
    return MappingProxyType(
        {
            int(s["id"]): Stat(
                id=int(s["id"]),
                abbrev=str(s.get("abbrev") or ""),
                description=str(s.get("description") or ""),
                api_identifier=s.get("apiIdentifier") or None,
                display_abbrev=s.get("displayAbbrev") or None,
                derived=bool(s.get("derived")),
                points_scoring_eligible=bool(s.get("pointsScoringEligible")),
            )
            for s in raw
        }
    )


def _parse_positions(settings: Mapping[str, Any]) -> PositionTable:
    raw = settings.get("positions") or []
    if not raw:
        raise ConstantsError("platform settings carried no positions")
    by_id: dict[PositionId, Position] = {}
    by_abbrev: dict[str, Position] = {}
    for p in raw:
        pos = Position(
            id=position_id(p["id"]),
            abbrev=str(p.get("abbrev") or ""),
            name=str(p.get("name") or ""),
            api_identifiers=tuple(str(a) for a in (p.get("apiIdentifiers") or ())),
        )
        by_id[pos.id] = pos
        # ESPN pads the table with placeholders (POS0, POS6, POS8); first writer
        # wins so a real position is never shadowed by one.
        by_abbrev.setdefault(pos.abbrev.upper(), pos)
    return PositionTable(MappingProxyType(by_id), MappingProxyType(by_abbrev))


def _parse_slots(settings: Mapping[str, Any], qb: PositionId | None) -> SlotTable:
    raw = settings.get("lineupSlots") or []
    if not raw:
        raise ConstantsError("platform settings carried no lineupSlots")
    by_id: dict[SlotId, LineupSlot] = {}
    by_abbrev: dict[str, LineupSlot] = {}
    for s in raw:
        eligible = frozenset(position_id(p) for p in (s.get("eligiblePositions") or ()))
        starter = bool(s.get("starter"))
        is_flex = starter and len(eligible) > 1
        slot = LineupSlot(
            id=slot_id(s["id"]),
            abbrev=str(s.get("abbrev") or ""),
            name=str(s.get("name") or ""),
            eligible_positions=eligible,
            is_starter=starter,
            is_bench=bool(s.get("bench")),
            is_flex=is_flex,
            is_superflex=is_flex and qb is not None and qb in eligible,
        )
        by_id[slot.id] = slot
        by_abbrev.setdefault(slot.abbrev.upper(), slot)
    return SlotTable(MappingProxyType(by_id), MappingProxyType(by_abbrev), qb)


def _parse_pro_teams(settings: Mapping[str, Any]) -> ProTeamTable:
    raw = settings.get("proTeams") or []
    if not raw:
        raise ConstantsError("platform settings carried no proTeams")
    by_id: dict[ProTeamId, ProTeam] = {}
    by_abbrev: dict[str, ProTeam] = {}
    for t in raw:
        bye = t.get("byeWeek")
        team = ProTeam(
            id=pro_team_id(t["id"]),
            # Upper-cased on purpose: ESPN returned "Atl" through 2025 and "ATL"
            # in 2026, and every external join is on uppercase codes.
            abbrev=str(t.get("abbrev") or "").upper(),
            location=str(t.get("location") or ""),
            name=str(t.get("name") or ""),
            bye_week=int(bye) if bye else None,
        )
        by_id[team.id] = team
        by_abbrev.setdefault(team.abbrev, team)
    return ProTeamTable(MappingProxyType(by_id), MappingProxyType(by_abbrev))


def _parse_enums(settings: Mapping[str, Any]) -> dict[str, EnumTable]:
    """Build from `types`, never `typeNames`.

    `typeNames` is a flat list of names in display order. Its index is not the id:
    `transactionTypes` is 1-based and `transactionStatusTypes` starts at -1, so
    zipping names against positions silently mislabels every member.
    """
    raw = settings.get(_ENUM_SOURCE_KEY) or {}
    tables: dict[str, EnumTable] = {}
    for key, members in raw.items():
        if not isinstance(members, list):
            continue
        by_id: dict[int, EnumMember] = {}
        by_name: dict[str, EnumMember] = {}
        for m in members:
            if not isinstance(m, dict) or "id" not in m or "name" not in m:
                continue
            member = EnumMember(
                id=int(m["id"]),
                name=str(m["name"]),
                abbrev=m.get("abbrev") or None,
                description=m.get("description") or None,
            )
            by_id[member.id] = member
            by_name.setdefault(member.name.upper(), member)
        if by_id:
            tables[key] = EnumTable(key, MappingProxyType(by_id), MappingProxyType(by_name))
    return tables


# -------------------------------------------------------------------- fetch/cache


def cache_path(season: int, root: Path = DEFAULT_CACHE_ROOT) -> Path:
    return Path(root) / f"platform_settings_{season}.json"


def fetch_payload(season: int, client: EspnClient | None = None) -> dict[str, Any]:
    """One unauthenticated GET.

    A season ESPN has never run 404s, and the client's 404 message talks about
    private leagues -- which is misleading here, so it is restated.
    """
    url = platform_settings_url(season)
    try:
        if client is not None:
            payload, _ = client.get(url)
        else:
            with EspnClient() as owned:
                payload, _ = owned.get(url)
    except EspnError as exc:
        raise ConstantsError(
            f"could not fetch platform settings for season {season}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConstantsError(f"unexpected platform-settings body from {url}: {type(payload)}")
    return payload


def write_cache(season: int, payload: Mapping[str, Any], root: Path = DEFAULT_CACHE_ROOT) -> Path:
    """Persist the payload verbatim, wrapped with provenance. Written atomically."""
    dest = cache_path(season, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "season": season,
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
        "url": platform_settings_url(season),
        "payload": payload,
    }
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    tmp.replace(dest)
    return dest


def read_cache(season: int, root: Path = DEFAULT_CACHE_ROOT) -> dict[str, Any] | None:
    """The cached payload, or None. A bare ESPN body dropped in by hand also loads."""
    path = cache_path(season, root)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable constants cache %s: %s", path, exc)
        return None
    if not isinstance(doc, dict):
        return None
    inner = doc.get("payload")
    return inner if isinstance(inner, dict) else doc


# Parsing is cheap but the disk read is not free and this gets called from tight
# loops in the valuation code. Keyed on (root, season) so tests using a tmp_path
# never collide with the real cache.
_MEMO: dict[tuple[str, int], PlatformSettings] = {}


def clear_memo() -> None:
    _MEMO.clear()


def load_platform_settings(
    season: int,
    *,
    client: EspnClient | None = None,
    root: Path = DEFAULT_CACHE_ROOT,
    force_refresh: bool = False,
    offline: bool = False,
) -> PlatformSettings:
    """Typed constants for a season: memo, then disk, then ESPN.

    `force_refresh` re-fetches and rewrites the cache -- use it when ESPN changes
    something mid-season, e.g. the 2026 abbrev re-casing. `offline` refuses to hit
    the network and raises if nothing is cached.

    The cache is only ever written from a payload that has already parsed. ESPN
    answers an unrecognized `view=` with HTTP 200 and a ten-key skeleton, and
    persisting one of those would turn a single bad response into a permanently
    poisoned cache that fails identically forever, network or no network.
    """
    # Resolved, so that a relative root does not alias a different directory if
    # the process changes cwd between calls.
    key = (str(Path(root).resolve()), int(season))
    if not force_refresh and key in _MEMO:
        return _MEMO[key]

    settings: PlatformSettings | None = None
    unusable: ConstantsError | None = None

    if not force_refresh:
        cached = read_cache(season, root)
        if cached is not None:
            try:
                settings = PlatformSettings.from_payload(cached, season)
            except ConstantsError as exc:
                # Syntactically fine, semantically not a platform-settings body:
                # a skeleton written by an older build, a truncated file, the
                # wrong season dropped in by hand. Refetch instead of dying.
                unusable = exc
                log.warning(
                    "discarding unusable constants cache %s: %s", cache_path(season, root), exc
                )

    if settings is None:
        if offline:
            why = f"unusable ({unusable})" if unusable is not None else "missing"
            raise ConstantsError(
                f"no cached platform settings for {season} at {cache_path(season, root)} "
                f"-- {why} -- and offline=True"
            )
        payload = fetch_payload(season, client)
        # Parse first; only a payload we can actually use earns a place on disk.
        settings = PlatformSettings.from_payload(payload, season)
        write_cache(season, payload, root)
        log.info("cached ESPN platform settings for %s", season)

    _MEMO[key] = settings
    return settings
