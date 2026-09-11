"""Cross-platform player ID spine.

Every join in the system runs through here, so every silent failure mode in here
becomes a silent failure mode everywhere. Seven of them are already known:

1. **The literal ``"NA"``.** DynastyProcess writes missing ids as the four-byte
   string ``NA``, not as an empty field. Measured on the live file: naive
   truthiness reports ``espn_id`` coverage of **100.0%**; the truth is **65.3%**.
   Every id read here goes through `_clean_id`, and a test pins it.
2. **Team defenses have no crosswalk.** Nobody publishes one, because every
   platform names them differently -- ESPN by numeric ``proTeamId``, Sleeper by
   ``"LV"``, KeepTradeCut by ``"Las Vegas Raiders"``, nflverse by ``"LV"``. In
   testing, *every* join failure was a defense. `DST_TEAMS` is a hard-coded
   32-row table and all D/ST joins route through it. They are never fuzzy-matched.
3. **Sleeper is no longer a usable ESPN crosswalk** (25% and falling). It is
   carried as a *target* id so we can talk to Sleeper, never as a bridge to ESPN.
4. **Suffix stripping creates collisions.** ``"marvin harrison"`` matches three
   rows in db_playerids -- the Cardinals rookie and two 2000s-era namesakes with
   no ESPN id at all. The resolver refuses an ambiguous name rather than picking
   one, which is why `resolve_name` returns ``None`` more often than you expect.
5. **A gsis id that is not a gsis id.** db_playerids fills the `gsis_id` cell for
   some undrafted rookies with an ESB-style placeholder (``WAS569019``) rather
   than ``NA``. It joins to nothing, and taking it at face value manufactures a
   second copy of a player who is already in the spine under his real id. Found
   by this module's own precision harness, not by reading the docs.
6. **Tiebreaks that look like tidy-ups.** Preferring the currently-rostered row
   when two records share a name is not evidence about *which* namesake was
   meant; it just converts a refusal into a confident wrong answer. It made
   ``resolve_name("Josh Johnson")`` return the Bengals quarterback at
   ``score=1.0, method="exact"`` while four people carry that exact name. Only
   the ESPN-id tiebreak survives, because a record with no ESPN id genuinely
   cannot answer an ESPN-keyed question. See `IdResolver._filter`.
7. **An id two records both claim.** `_prepare_dynastyprocess` deduplicates
   espn/gsis/sleeper and nothing else, so 19 secondary ids (11 cbs, 4 pfr, 2
   fleaflicker, 1 ktc, 1 rotowire) are each shared by two unrelated players on
   the live file. First-writer-wins made `to_canonical` answer with a player who
   did not own the id. Those now resolve to None; see `IdResolver.ambiguous_ids`.

Measured against the live ESPN pool on 2026-09-07 with the id as ground truth
for the name: resolving the NAME ONLY lands on the same canonical the id does
600/600 at pool depth 600 and 1027/1028 at depth 1036 with team+position, 594/600
and 1009/1028 with no context at all, and **zero wrong matches** in every
configuration. Traps 6 and 7 were found by extending that harness past the top
600, where the shallow pool had been hiding three wrong matches.

Coverage, measured 2026-09-07 against the live top-600 ESPN pool -- note this is
the direction that matters (ESPN id -> crosswalk row), and it is much healthier
than the source-side fill rates quoted in RESEARCH.md:

| source | ESPN pool joined | source rows carrying an espn_id |
|---|---|---|
| nflverse ``roster_2026`` | 88.2% (93.1% ex-D/ST) | 67.8% all / 86.2% skill positions |
| DynastyProcess ``db_playerids`` | 94.7% (100% ex-D/ST) | 65.3% |
| union | 94.7% (100% ex-D/ST) | -- |

nflverse is still preferred: it is refreshed daily and carries the current team
and position, which the fuzzy matcher needs for blocking. DynastyProcess is the
fallback precisely because it keeps retired and free-agent players that a current
roster file has dropped -- which is exactly the 39 ESPN entries nflverse misses.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from ..paths import data_dir

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

ESPN = "espn"
GSIS = "gsis"
SLEEPER = "sleeper"
YAHOO = "yahoo"
PFR = "pfr"
SPORTRADAR = "sportradar"
ROTOWIRE = "rotowire"
PFF = "pff"
FANTASY_DATA = "fantasy_data"
MFL = "mfl"
FANTASYPROS = "fantasypros"
KTC = "ktc"
CBS = "cbs"
FLEAFLICKER = "fleaflicker"
NFL = "nfl"

SOURCES: tuple[str, ...] = (
    ESPN,
    GSIS,
    SLEEPER,
    YAHOO,
    PFR,
    SPORTRADAR,
    ROTOWIRE,
    PFF,
    FANTASY_DATA,
    MFL,
    FANTASYPROS,
    KTC,
    CBS,
    FLEAFLICKER,
    NFL,
)

_NFLVERSE_ID_COLUMNS: dict[str, str] = {
    "gsis_id": GSIS,
    "espn_id": ESPN,
    "sleeper_id": SLEEPER,
    "yahoo_id": YAHOO,
    "pfr_id": PFR,
    "sportradar_id": SPORTRADAR,
    "rotowire_id": ROTOWIRE,
    "pff_id": PFF,
    "fantasy_data_id": FANTASY_DATA,
}

_DYNASTYPROCESS_ID_COLUMNS: dict[str, str] = {
    "gsis_id": GSIS,
    "espn_id": ESPN,
    "sleeper_id": SLEEPER,
    "yahoo_id": YAHOO,
    "pfr_id": PFR,
    "sportradar_id": SPORTRADAR,
    "rotowire_id": ROTOWIRE,
    "pff_id": PFF,
    "fantasy_data_id": FANTASY_DATA,
    "mfl_id": MFL,
    "fantasypros_id": FANTASYPROS,
    "ktc_id": KTC,
    "cbs_id": CBS,
    "fleaflicker_id": FLEAFLICKER,
    "nfl_id": NFL,
}

# --------------------------------------------------------------------------
# Reference-file cache
# --------------------------------------------------------------------------

REFERENCE_DIR = data_dir("data/reference")

NFLVERSE_ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.parquet"
)
# nflverse publishes a per-tag build stamp. Polling it is a 24-byte request against
# a 540 KB download, so the daily refresh is nearly free.
NFLVERSE_ROSTER_STAMP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/rosters/timestamp.txt"
)
DYNASTYPROCESS_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
)

DEFAULT_MAX_AGE = dt.timedelta(hours=24)

_DOWNLOAD_HEADERS = {"User-Agent": "fantasy_quant/0.1 (+https://github.com/blobspire)"}


# --------------------------------------------------------------------------
# Team defenses -- the hard-coded 32-row table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamDefense:
    """One team defense, keyed on the nflverse ``team_abbr``.

    `espn_player_id` is not guesswork: ESPN exposes each D/ST in the player pool
    as a synthetic negative id of the form ``-16{proTeamId:03d}`` (verified live
    for all 32 on 2026-09-07 -- Texans/proTeamId 34 is ``-16034``). A network test
    pins it, because it is the only way to address a defense in `filterIds`.
    """

    nflverse: str
    pro_team_id: int
    espn_abbrev: str
    sleeper: str
    location: str
    nickname: str
    aliases: tuple[str, ...] = ()

    @property
    def canonical(self) -> str:
        return f"DST-{self.nflverse}"

    @property
    def espn_player_id(self) -> int:
        return -(16_000 + self.pro_team_id)

    @property
    def display_name(self) -> str:
        """The KeepTradeCut / FantasyCalc form, e.g. ``"Las Vegas Raiders"``."""
        return f"{self.location} {self.nickname}"


# proTeamId values are ESPN's, verified against
# `seasons/2026?view=chui_default_platformsettings`. 31 and 32 do not exist.
# Aliases cover the platform-specific and historical spellings we have actually
# seen in the wild: ESPN's WSH/LAR, DynastyProcess's MFL-style GBP/KCC/LVR/JAC,
# pro-football-reference's CRD/RAV/HTX/OTI/CLT/RAI/SDG/RAM, and relocations.
DST_TEAMS: tuple[TeamDefense, ...] = (
    TeamDefense("ARI", 22, "ARI", "ARI", "Arizona", "Cardinals", ("ARZ", "CRD", "PHO")),
    TeamDefense("ATL", 1, "ATL", "ATL", "Atlanta", "Falcons", ()),
    TeamDefense("BAL", 33, "BAL", "BAL", "Baltimore", "Ravens", ("BLT", "RAV")),
    TeamDefense("BUF", 2, "BUF", "BUF", "Buffalo", "Bills", ()),
    TeamDefense("CAR", 29, "CAR", "CAR", "Carolina", "Panthers", ()),
    TeamDefense("CHI", 3, "CHI", "CHI", "Chicago", "Bears", ()),
    TeamDefense("CIN", 4, "CIN", "CIN", "Cincinnati", "Bengals", ()),
    TeamDefense("CLE", 5, "CLE", "CLE", "Cleveland", "Browns", ("CLV",)),
    TeamDefense("DAL", 6, "DAL", "DAL", "Dallas", "Cowboys", ()),
    TeamDefense("DEN", 7, "DEN", "DEN", "Denver", "Broncos", ()),
    TeamDefense("DET", 8, "DET", "DET", "Detroit", "Lions", ()),
    TeamDefense("GB", 9, "GB", "GB", "Green Bay", "Packers", ("GBP", "GNB")),
    TeamDefense("HOU", 34, "HOU", "HOU", "Houston", "Texans", ("HST", "HTX")),
    TeamDefense("IND", 11, "IND", "IND", "Indianapolis", "Colts", ("CLT",)),
    TeamDefense("JAX", 30, "JAX", "JAX", "Jacksonville", "Jaguars", ("JAC",)),
    TeamDefense("KC", 12, "KC", "KC", "Kansas City", "Chiefs", ("KCC", "KAN")),
    TeamDefense("LA", 14, "LAR", "LAR", "Los Angeles", "Rams", ("RAM", "STL")),
    TeamDefense("LAC", 24, "LAC", "LAC", "Los Angeles", "Chargers", ("SD", "SDC", "SDG")),
    TeamDefense("LV", 13, "LV", "LV", "Las Vegas", "Raiders", ("OAK", "LVR", "RAI")),
    TeamDefense("MIA", 15, "MIA", "MIA", "Miami", "Dolphins", ()),
    TeamDefense("MIN", 16, "MIN", "MIN", "Minnesota", "Vikings", ()),
    TeamDefense("NE", 17, "NE", "NE", "New England", "Patriots", ("NEP", "NWE")),
    TeamDefense("NO", 18, "NO", "NO", "New Orleans", "Saints", ("NOS", "NOR")),
    TeamDefense("NYG", 19, "NYG", "NYG", "New York", "Giants", ()),
    TeamDefense("NYJ", 20, "NYJ", "NYJ", "New York", "Jets", ()),
    TeamDefense("PHI", 21, "PHI", "PHI", "Philadelphia", "Eagles", ()),
    TeamDefense("PIT", 23, "PIT", "PIT", "Pittsburgh", "Steelers", ()),
    TeamDefense("SEA", 26, "SEA", "SEA", "Seattle", "Seahawks", ()),
    TeamDefense("SF", 25, "SF", "SF", "San Francisco", "49ers", ("SFO", "SFN")),
    TeamDefense("TB", 27, "TB", "TB", "Tampa Bay", "Buccaneers", ("TBB", "TAM")),
    TeamDefense("TEN", 10, "TEN", "TEN", "Tennessee", "Titans", ("OTI",)),
    TeamDefense("WAS", 28, "WSH", "WAS", "Washington", "Commanders", ("WSH", "WFT")),
)

DST_BY_NFLVERSE: dict[str, TeamDefense] = {t.nflverse: t for t in DST_TEAMS}
DST_BY_PRO_TEAM_ID: dict[int, TeamDefense] = {t.pro_team_id: t for t in DST_TEAMS}

# ESPN's `proTeamId` space, for callers that need to reject an id rather than
# silently map it. 0 is free agency; 31 and 32 have never existed.
ESPN_PRO_TEAM_IDS: frozenset[int] = frozenset(DST_BY_PRO_TEAM_ID)

# ESPN defaultPositionId -> our position label. Note this is the *position* space,
# not the lineup-slot space; the two collide at 4 and 15 (see RESEARCH.md).
ESPN_POSITION_IDS: dict[int, str] = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    7: "P",
    9: "DL",
    10: "DL",
    11: "LB",
    12: "DB",
    13: "DB",
    14: "HC",
    16: "DST",
    17: "LB",
}

_POSITION_ALIASES: dict[str, str] = {
    "PK": "K",
    "FB": "RB",
    "HB": "RB",
    "PN": "P",
    "DEF": "DST",
    "D": "DST",
    "D/ST": "DST",
    "DST": "DST",
    "DEFENSE": "DST",
    "TEAM DEFENSE": "DST",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "EDGE": "LB",
    "EDR": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "OT": "OL",
    "OG": "OL",
    "T": "OL",
    "G": "OL",
    "C": "OL",
}

_UNKNOWN_POSITIONS = frozenset({"", "?", "XX", "NA", "UNK", "NONE"})


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_GSIS_PATTERN = re.compile(r"^00-0\d{6}$")
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_NULLISH = frozenset({"", "NA", "N/A", "NAN", "NONE", "NULL", "-", "--"})


def _clean_id(value: object) -> str | None:
    """Coerce a raw id cell to a real string, or to None.

    This is the "NA" trap. DynastyProcess writes missing ids as the literal string
    ``NA``; polars reads it as a value, `bool("NA")` is True, and the resulting
    coverage report says 100% while a third of the joins fail.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s.upper() in _NULLISH:
        return None
    # Numeric id columns round-trip through float in some readers.
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s or None


def _clean_gsis(value: object) -> str | None:
    """A gsis id, or None -- including when the source called it a gsis id.

    db_playerids fills `gsis_id` for undrafted rookies with an ESB-style
    placeholder (``WAS569019``, ``BAI173035``) instead of leaving it ``NA``. Five
    rows do it today. Those tokens will never match nflverse, so letting one
    become a canonical id manufactures a second copy of a real player -- which is
    exactly how "Mike Washington Jr." ended up in the spine twice. Real gsis ids
    are ``00-0`` plus six digits, without exception across 2,953 roster rows.
    """
    cleaned = _clean_id(value)
    if cleaned is None or not _GSIS_PATTERN.match(cleaned):
        return None
    return cleaned


_ID_CLEANERS = {GSIS: _clean_gsis}


@lru_cache(maxsize=8192)
def normalize_name(name: str) -> str:
    """Lowercase, accent-free, punctuation-free, single-spaced.

    Punctuation becomes a *space*, not nothing: ``"St.Brown"`` and ``"St. Brown"``
    have to land on the same key, and joining them without a separator would fuse
    ``"stbrown"``. The no-separator form is available separately as `compact_name`,
    which is what reconciles ``"D.K. Metcalf"`` with nflverse's ``"DK Metcalf"``.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Apostrophes close up ("Ja'Marr" -> "jamarr"); everything else opens up.
    ascii_only = ascii_only.replace("'", "").replace("’", "")
    return _NON_ALNUM.sub(" ", ascii_only.lower()).strip()


def strip_suffix(normalized: str) -> str:
    """Drop a trailing generational suffix from an already-normalized name.

    Guarded at three tokens so a two-token name is never truncated to one.
    """
    tokens = normalized.split()
    if len(tokens) >= 3 and tokens[-1] in _NAME_SUFFIXES:
        return " ".join(tokens[:-1])
    return normalized


def compact_name(name: str) -> str:
    """Suffix-stripped, alphanumerics only. ``"D.K. Metcalf"`` -> ``"dkmetcalf"``."""
    return strip_suffix(normalize_name(name)).replace(" ", "")


def surname(name: str) -> str:
    """Last token of the suffix-stripped normalized name."""
    tokens = strip_suffix(normalize_name(name)).split()
    return tokens[-1] if tokens else ""


def given_name(name: str) -> str:
    """Everything before the surname."""
    tokens = strip_suffix(normalize_name(name)).split()
    return " ".join(tokens[:-1]) if len(tokens) > 1 else ""


def _build_team_aliases() -> dict[str, str]:
    out: dict[str, str] = {}
    for team in DST_TEAMS:
        keys = {
            team.nflverse,
            team.espn_abbrev,
            team.sleeper,
            *team.aliases,
        }
        for key in keys:
            out[normalize_name(key)] = team.nflverse
        out[normalize_name(team.display_name)] = team.nflverse
    return out


_TEAM_ALIASES: dict[str, str] = _build_team_aliases()


def normalize_team(value: object) -> str | None:
    """Any team spelling -> the nflverse ``team_abbr``.

    Accepts an ESPN ``proTeamId`` (int or numeric string) too. ESPN's 0 is free
    agency and maps to None, which is the right answer: a free agent has no team.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return team_from_pro_team_id(value)
    s = str(value).strip()
    if not s or s.upper() in _NULLISH or s.upper().startswith("FA"):
        return None
    if s.lstrip("-").isdigit():
        return team_from_pro_team_id(int(s))
    return _TEAM_ALIASES.get(normalize_name(s))


def team_from_pro_team_id(pro_team_id: int) -> str | None:
    team = DST_BY_PRO_TEAM_ID.get(pro_team_id)
    return team.nflverse if team else None


def normalize_position(value: object) -> str | None:
    """Any position spelling (or an ESPN ``defaultPositionId``) -> our label."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ESPN_POSITION_IDS.get(value)
    s = str(value).strip().upper()
    if s.lstrip("-").isdigit():
        return ESPN_POSITION_IDS.get(int(s))
    s = s.rstrip("?").strip()
    if not s or s in _UNKNOWN_POSITIONS:
        return None
    if s in _POSITION_ALIASES:
        return _POSITION_ALIASES[s]
    return s


# --------------------------------------------------------------------------
# D/ST resolution -- all defense joins come through here, none are fuzzy
# --------------------------------------------------------------------------


def _build_dst_keys() -> dict[str, TeamDefense]:
    out: dict[str, TeamDefense] = {}
    for team in DST_TEAMS:
        forms = {
            team.nflverse,
            team.espn_abbrev,
            team.sleeper,
            team.nickname,
            team.display_name,
            f"{team.display_name} D/ST",
            f"{team.display_name} Defense",
            f"{team.nickname} D/ST",
            f"{team.nickname} DST",
            f"{team.nflverse} D/ST",
            f"{team.espn_abbrev} D/ST",
            team.canonical,
            *team.aliases,
        }
        for form in forms:
            out.setdefault(normalize_name(form), team)
    return out


_DST_KEYS: dict[str, TeamDefense] = _build_dst_keys()


def resolve_dst(value: object, *, numeric_ids: bool = True) -> TeamDefense | None:
    """Resolve any platform's spelling of a team defense.

    Handles nflverse/ESPN/Sleeper abbreviations, ESPN's numeric ``proTeamId``,
    ESPN's synthetic ``-16xxx`` player id, KeepTradeCut's ``"Las Vegas Raiders"``,
    ESPN's ``"Raiders D/ST"``, and our own ``"DST-LV"`` canonical.

    Never guess at a defense. Every join failure observed in testing was one, and
    a wrong defense is indistinguishable from a right one in a points column.

    Pass ``numeric_ids=False`` when the input is a human display name rather than
    an identifier. Otherwise a name that happens to be all digits is read as a
    ``proTeamId`` -- ``"12"`` would resolve to the Chiefs defense at full
    confidence, which is exactly the silent wrong-join this module exists to
    prevent.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _dst_from_int(value) if numeric_ids else None
    s = str(value).strip()
    if not s:
        return None
    if s.lstrip("-").isdigit():
        return _dst_from_int(int(s)) if numeric_ids else None
    return _DST_KEYS.get(normalize_name(s))


def _dst_from_int(value: int) -> TeamDefense | None:
    if value < 0:
        # ESPN player-pool id: -16{proTeamId:03d}.
        magnitude = -value
        if magnitude // 1000 == 16:
            return DST_BY_PRO_TEAM_ID.get(magnitude % 1000)
        return None
    return DST_BY_PRO_TEAM_ID.get(value)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerIds:
    """One person (or one team defense), with every id we know for them."""

    canonical: str
    name: str
    team: str | None
    position: str | None
    origin: str
    ids: dict[str, str]

    @property
    def is_dst(self) -> bool:
        return self.position == "DST"

    def id_for(self, source: str) -> str | None:
        return self.ids.get(source)


@dataclass(frozen=True, slots=True)
class NameMatch:
    """A name-based resolution, with enough provenance to audit it later."""

    canonical: str
    name: str
    score: float
    method: str


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """Join rate for one source. Print these weekly; watch them decay."""

    source: str
    total: int
    resolved: int
    missing_sample: tuple[str, ...] = ()

    @property
    def rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0


def _dst_records() -> list[PlayerIds]:
    return [
        PlayerIds(
            canonical=team.canonical,
            name=f"{team.nickname} D/ST",
            team=team.nflverse,
            position="DST",
            origin="dst-table",
            ids={ESPN: str(team.espn_player_id), SLEEPER: team.sleeper},
        )
        for team in DST_TEAMS
    ]


def _row_ids(row: Mapping[str, Any], columns: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for column, source in columns.items():
        cleaned = _ID_CLEANERS.get(source, _clean_id)(row.get(column))
        if cleaned is not None:
            out[source] = cleaned
    return out


def build_records(
    roster: pl.DataFrame | None = None,
    dynastyprocess: pl.DataFrame | None = None,
) -> list[PlayerIds]:
    """Merge the sources into one spine, nflverse first.

    Merge key, in order: gsis, then espn, then sleeper. A DynastyProcess row that
    matches an existing person on any of those *fills gaps only* -- it never
    overwrites an nflverse id, because nflverse is refreshed daily and
    db_playerids is not.
    """
    records: dict[str, PlayerIds] = {}
    index: dict[tuple[str, str], str] = {}

    def register(record: PlayerIds) -> None:
        records[record.canonical] = record
        for source, value in record.ids.items():
            index.setdefault((source, value), record.canonical)

    def find(ids: Mapping[str, str]) -> str | None:
        for source in (GSIS, ESPN, SLEEPER):
            value = ids.get(source)
            if value is not None and (hit := index.get((source, value))) is not None:
                return hit
        return None

    for record in _dst_records():
        register(record)

    if roster is not None:
        for row in _prepare_roster(roster).iter_rows(named=True):
            ids = _row_ids(row, _NFLVERSE_ID_COLUMNS)
            canonical = _canonical_for(ids)
            if canonical is None:
                continue
            register(
                PlayerIds(
                    canonical=canonical,
                    name=str(row.get("full_name") or ""),
                    team=normalize_team(row.get("team")),
                    position=normalize_position(row.get("position")),
                    origin="nflverse",
                    ids=ids,
                )
            )

    if dynastyprocess is not None:
        for row in _prepare_dynastyprocess(dynastyprocess).iter_rows(named=True):
            ids = _row_ids(row, _DYNASTYPROCESS_ID_COLUMNS)
            if not ids:
                continue
            existing = find(ids)
            if existing is not None:
                record = records[existing]
                merged = dict(ids)
                merged.update(record.ids)  # nflverse wins on conflict
                updated = PlayerIds(
                    canonical=record.canonical,
                    name=record.name or str(row.get("name") or ""),
                    team=record.team or normalize_team(row.get("team")),
                    position=record.position or normalize_position(row.get("position")),
                    origin=record.origin,
                    ids=merged,
                )
                register(updated)
                continue
            canonical = _canonical_for(ids)
            if canonical is None:
                continue
            register(
                PlayerIds(
                    canonical=canonical,
                    name=str(row.get("name") or ""),
                    team=normalize_team(row.get("team")),
                    position=normalize_position(row.get("position")),
                    origin="dynastyprocess",
                    ids=ids,
                )
            )

    return list(records.values())


def _canonical_for(ids: Mapping[str, str]) -> str | None:
    """Canonical id: gsis when we have it, else a namespaced fallback.

    gsis is the spine because it is nflverse's own key, is stable across seasons,
    and is present on 99.97% of roster rows. The fallbacks exist for retirees that
    only DynastyProcess carries.
    """
    if (gsis := ids.get(GSIS)) is not None:
        return gsis
    if (espn := ids.get(ESPN)) is not None:
        return f"ESPN-{espn}"
    if (mfl := ids.get(MFL)) is not None:
        return f"MFL-{mfl}"
    if (sleeper := ids.get(SLEEPER)) is not None:
        return f"SLEEPER-{sleeper}"
    return None


def _prepare_roster(df: pl.DataFrame) -> pl.DataFrame:
    """One row per player, most recent week wins.

    ``roster_{season}`` is week-stamped; mid-season it carries several weeks and a
    player who changed teams appears more than once. Taking the last week keeps
    the team column current, which is what the fuzzy matcher blocks on.

    Rows with no gsis id are held out of the deduplication rather than run through
    it. `unique` treats null as a value, so a plain ``unique(subset=["gsis_id"])``
    collapses *every* gsis-less row into one -- today's file has a single one and
    hides the bug, but the ids they do carry (espn, sleeper) are exactly what the
    fallback path needs.
    """
    wanted = [
        c
        for c in ("full_name", "team", "position", "week", *_NFLVERSE_ID_COLUMNS)
        if c in df.columns
    ]
    out = df.select(wanted)
    if "week" in out.columns and "gsis_id" in out.columns:
        out = out.sort("week")
        keyed = out.filter(pl.col("gsis_id").is_not_null()).unique(
            subset=["gsis_id"], keep="last", maintain_order=True
        )
        out = pl.concat([keyed, out.filter(pl.col("gsis_id").is_null())])
    return out


def _prepare_dynastyprocess(df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate db_playerids deterministically.

    13 espn_ids and 10 gsis_ids are shared by two rows apiece -- all of them junk
    pairings of a modern player with a same-id 1990s namesake. Keeping the later
    ``draft_year`` picks the live player every time; the mfl_id tiebreak keeps the
    choice stable across file refreshes.
    """
    wanted = [
        c
        for c in ("name", "team", "position", "draft_year", *_DYNASTYPROCESS_ID_COLUMNS)
        if c in df.columns
    ]
    out = df.select(wanted)
    sort_cols = [c for c in ("draft_year", "mfl_id") if c in out.columns]
    if sort_cols:
        ranks = [f"_rank_{c}" for c in sort_cols]
        out = out.with_columns(
            [
                pl.col(c).cast(pl.Int64, strict=False).fill_null(-1).alias(r)
                for c, r in zip(sort_cols, ranks, strict=True)
            ]
        )
        out = out.sort(ranks).drop(ranks)

    out = out.with_row_index("_row")
    for key in ("espn_id", "gsis_id", "sleeper_id"):
        if key not in out.columns:
            continue
        # Missing ids are the string "NA", and there are thousands of them --
        # deduplicating on that value would collapse the whole file to one row.
        blank = pl.col(key).is_null() | pl.col(key).is_in(["NA", ""])
        out = out.filter(blank | (pl.col("_row") == pl.col("_row").max().over(key)))
    return out.drop("_row")


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------

# A surname must match exactly before we will consider a fuzzy hit at all.
# Given-name variation is common ("Josh"/"Joshua", "Chig"/"Chigoziem",
# "Hollywood"/"Marquise"); surname variation is mostly a different person.
_MIN_SCORE = 0.80
_MIN_MARGIN = 0.15
_MIN_GIVEN_RATIO = 0.60


class IdResolver:
    """Maps any supported id, or a bare name, to a canonical internal id.

    Canonical ids are the gsis id where we have one (``00-0036322``), a namespaced
    fallback where we do not (``ESPN-4686658``, ``MFL-17482``, ``SLEEPER-13305``),
    and ``DST-{nflverse_abbr}`` for the 32 team defenses.
    """

    __slots__ = (
        "_by_canonical",
        "_by_source",
        "_ambiguous",
        "_by_exact",
        "_by_stripped",
        "_by_compact",
        "_by_surname",
        "_name_cache",
    )

    def __init__(self, records: Iterable[PlayerIds]) -> None:
        self._by_canonical: dict[str, PlayerIds] = {}
        self._by_source: dict[str, dict[str, str]] = {s: {} for s in SOURCES}
        self._ambiguous: dict[str, set[str]] = {}
        self._by_exact: dict[str, list[str]] = {}
        self._by_stripped: dict[str, list[str]] = {}
        self._by_compact: dict[str, list[str]] = {}
        self._by_surname: dict[str, list[str]] = {}
        self._name_cache: dict[tuple[str, str | None, str | None, float], NameMatch | None] = {}

        for record in records:
            self._by_canonical[record.canonical] = record
            for source, value in record.ids.items():
                table = self._by_source.setdefault(source, {})
                # Two records claiming one id is not a tie to break silently. The
                # secondary id spaces really do collide -- on the live file 11
                # cbs_ids, 4 pfr_ids, 2 fleaflicker_ids, 1 ktc_id and 1
                # rotowire_id are each shared by two unrelated players, because
                # `_prepare_dynastyprocess` only deduplicates espn/gsis/sleeper.
                # First-writer-wins left `to_canonical(cbs_id)` pointing at Coby
                # Bryant while Cobee Bryant's own record still carried it, so the
                # id resolved to a different person than the one who owns it.
                if table.setdefault(value, record.canonical) != record.canonical:
                    self._ambiguous.setdefault(source, set()).add(value)
            # Defenses are deliberately absent from the name indexes. Every
            # defense lookup goes through `resolve_dst`, so leaving "Rams D/ST"
            # out of the player tables removes a whole class of cross-matching.
            if not record.name or record.is_dst:
                continue
            normalized = normalize_name(record.name)
            self._by_exact.setdefault(normalized, []).append(record.canonical)
            self._by_stripped.setdefault(strip_suffix(normalized), []).append(record.canonical)
            self._by_compact.setdefault(compact_name(record.name), []).append(record.canonical)
            self._by_surname.setdefault(surname(record.name), []).append(record.canonical)

    # -- construction -----------------------------------------------------

    @classmethod
    def load(
        cls,
        season: int | None = None,
        *,
        cache_dir: Path = REFERENCE_DIR,
        max_age: dt.timedelta = DEFAULT_MAX_AGE,
        allow_download: bool = True,
    ) -> IdResolver:
        """Build from the cached crosswalks, refreshing them if stale."""
        season = season or default_season()
        roster = load_nflverse_roster(
            season, cache_dir=cache_dir, max_age=max_age, allow_download=allow_download
        )
        dp = load_dynastyprocess(
            cache_dir=cache_dir, max_age=max_age, allow_download=allow_download
        )
        return cls(build_records(roster, dp))

    # -- id lookups -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_canonical)

    def __contains__(self, canonical: object) -> bool:
        return canonical in self._by_canonical

    def get(self, canonical: str) -> PlayerIds | None:
        return self._by_canonical.get(canonical)

    def to_canonical(self, value: object, source: str) -> str | None:
        """Any source id -> canonical id.

        ESPN's synthetic ``-16xxx`` defense ids route through the defense table.
        Only *negative* ESPN ids do -- a bare ``proTeamId`` like 22 is also a
        perfectly good ESPN player id, so it stays with the player index. Pass a
        team id to `resolve_dst` when you mean the defense.

        An id that two unrelated records both claim resolves to None. Some of the
        secondary id spaces really do collide upstream, and answering with
        whichever row happened to be read first is the silent cross-wiring this
        module exists to prevent. Check `ambiguous_ids` if you need to see them.
        """
        if value is None:
            return None
        if source == ESPN and (team := _dst_from_int_like(value)) is not None:
            return team.canonical
        key = _clean_id(value)
        if key is None:
            return None
        if key in self._ambiguous.get(source, ()):
            return None
        return self._by_source.get(source, {}).get(key)

    def ambiguous_ids(self, source: str) -> frozenset[str]:
        """Ids in `source` that more than one record claims. These resolve to None."""
        return frozenset(self._ambiguous.get(source, ()))

    def from_canonical(self, canonical: str, source: str) -> str | None:
        record = self._by_canonical.get(canonical)
        return record.ids.get(source) if record else None

    def translate(self, value: object, from_source: str, to_source: str) -> str | None:
        canonical = self.to_canonical(value, from_source)
        return self.from_canonical(canonical, to_source) if canonical else None

    def record_for(self, value: object, source: str) -> PlayerIds | None:
        canonical = self.to_canonical(value, source)
        return self._by_canonical.get(canonical) if canonical else None

    # -- name resolution --------------------------------------------------

    @property
    def name_cache_size(self) -> int:
        return len(self._name_cache)

    def resolve_name(
        self,
        name: str,
        team: object = None,
        position: object = None,
        *,
        min_score: float = _MIN_SCORE,
    ) -> NameMatch | None:
        """Resolve a bare display name, for sources that publish only strings.

        Underdog and FanDuel hand us names and nothing else, so this is the last
        line of the join. It is deliberately conservative: an ambiguous name
        returns ``None`` rather than a plausible-looking wrong player, because a
        wrong player is invisible downstream and a missing one is loud.

        Cascade, each stage filtered by whatever team/position context is supplied
        and each requiring a unique survivor:

        1. exact normalized name, suffix included
        2. suffix stripped -- ``"Marvin Harrison Jr."`` -> ``"marvin harrison"``
        3. compacted to alphanumerics -- reconciles ``"D.K."`` with ``"DK"``
        4. exact surname within a full (team, position) block
        5. exact surname plus a close given name, with a margin over the runner-up

        `min_score` gates the last two stages. A surname-only match inside a full
        (team, position) block scores 0.85+, so raising the floor above that turns
        the nickname path off for a caller that cannot tolerate it. "Full block"
        means both filters actually survived: `_filter` discards a filter that
        would empty the block, and a block narrowed by team alone does not earn
        the bonus no matter what the caller passed in.

        Results are cached per ``(name, team, position, min_score)``, misses
        included -- a props feed re-posts the same few hundred names every refresh,
        and a third of them are never in the crosswalk at all.
        """
        team_key = normalize_team(team)
        position_key = normalize_position(position)
        cache_key = (name, team_key, position_key, min_score)
        if cache_key in self._name_cache:
            return self._name_cache[cache_key]
        match = self._resolve_name_uncached(name, team_key, position_key, min_score)
        self._name_cache[cache_key] = match
        return match

    def _resolve_name_uncached(
        self,
        name: str,
        team_key: str | None,
        position_key: str | None,
        min_score: float,
    ) -> NameMatch | None:
        # Defenses first, always, and they never leave this branch. A name the
        # table recognizes is a defense; pairing it with a player position is a
        # contradiction in the caller's data, not an invitation to go looking for
        # a wide receiver that happens to be spelled a bit like the Rams.
        if (defense := resolve_dst(name, numeric_ids=False)) is not None:
            if position_key not in (None, "DST"):
                return None
            record = self._by_canonical.get(defense.canonical)
            return NameMatch(record.canonical, record.name, 1.0, "dst") if record else None
        if position_key == "DST":
            return None

        normalized = normalize_name(name)
        if not normalized:
            return None

        for method, table, key in (
            ("exact", self._by_exact, normalized),
            ("suffix", self._by_stripped, strip_suffix(normalized)),
            ("compact", self._by_compact, compact_name(name)),
        ):
            candidates, _ = self._filter(table.get(key, ()), team_key, position_key)
            if len(candidates) == 1:
                record = self._by_canonical[candidates[0]]
                return NameMatch(record.canonical, record.name, 1.0, method)

        return self._fuzzy(name, team_key, position_key, min_score)

    def _fuzzy(
        self,
        name: str,
        team_key: str | None,
        position_key: str | None,
        min_score: float,
    ) -> NameMatch | None:
        candidates, full_block = self._filter(
            self._by_surname.get(surname(name), ()), team_key, position_key
        )
        if not candidates:
            return None

        query_given = given_name(name)

        def given_ratio(canonical: str) -> float:
            other = given_name(self._by_canonical[canonical].name)
            return SequenceMatcher(None, query_given, other).ratio()

        scored = sorted(
            ((given_ratio(c), c) for c in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_ratio, best = scored[0]

        if len(scored) == 1:
            # A unique surname inside a fully specified (team, position) block is
            # itself strong evidence -- it is what rescues nickname entries like
            # "Hollywood Brown", where the given names share nothing at all.
            #
            # `full_block` is not the same question as "did the caller supply both
            # fields". `_filter` discards a filter that would empty the block, so a
            # caller-supplied position can be silently dropped and leave a block
            # narrowed by team alone. Awarding the 0.85 bonus there is how
            # "Mike Williams" (LAC, WR -- a free agent whose spine row carries no
            # team) once resolved to Marcus Williams, the only Charger named
            # Williams left standing after the WR filter was thrown away.
            if full_block:
                score = 0.85 + 0.15 * best_ratio
                method = "surname"
            else:
                score = 0.60 + 0.40 * best_ratio
                method = "fuzzy"
        else:
            if best_ratio < _MIN_GIVEN_RATIO or best_ratio - scored[1][0] < _MIN_MARGIN:
                return None
            score = 0.60 + 0.40 * best_ratio
            method = "fuzzy"

        if score < min_score:
            return None
        record = self._by_canonical[best]
        return NameMatch(record.canonical, record.name, round(score, 4), method)

    def _filter(
        self,
        canonicals: Sequence[str],
        team_key: str | None,
        position_key: str | None,
    ) -> tuple[list[str], bool]:
        """Narrow candidates by context, but never down to nothing.

        Returns ``(candidates, full_block)``. `full_block` is True only when a
        team *and* a position were supplied and *both* survived -- a filter that
        would empty the block is discarded, and a caller must not be told a block
        is fully qualified when half its constraints were thrown away.

        A stale team or an unexpected position label should degrade the match, not
        destroy it, hence the discard. One tiebreak follows:

        *Carries an ESPN id.* Everything downstream keys on ESPN, so a record with
        no ESPN id cannot be the answer to a question we will ask. It also settles
        the case where one person appears twice because the sources share no id at
        all -- a 2026 rookie whose nflverse row has only a gsis and whose
        db_playerids row has everything else -- and it is what disambiguates
        ``"Michael Pittman Jr."`` from the 2000s-era running back of the same name,
        whose db_playerids row has no ESPN id at all.

        There is deliberately *no* "prefer the currently-rostered row" tiebreak.
        It reads as harmless and is not: two distinct people who share a name are
        not disambiguated by one of them being on this year's roster, and applying
        it silently converted real ambiguity into a confident wrong answer. On the
        live 2026 pool it made ``resolve_name("Josh Johnson")`` return the Bengals
        quarterback at ``score=1.0, method="exact"`` when four different players
        carry that exact name, and it pruned the genuine "D.J. Williams" and
        "Mike Williams" rows out of the surname block, handing the fuzzy tier to
        C.J. Williams and Mykel Williams respectively. Without it those three
        return None, which is the loud failure this module is built to prefer.
        """
        out = list(canonicals)
        applied: list[bool] = []
        for key, attr in ((team_key, "team"), (position_key, "position")):
            if key is None:
                applied.append(False)
                continue
            narrowed = [c for c in out if getattr(self._by_canonical[c], attr) == key]
            applied.append(bool(narrowed))
            if narrowed:
                out = narrowed
        if len(out) > 1:
            reachable = [c for c in out if ESPN in self._by_canonical[c].ids]
            if reachable:
                out = reachable
        return out, all(applied)

    # -- coverage ---------------------------------------------------------

    def spine_coverage(self) -> list[SourceCoverage]:
        """Fill rate of each source across the spine itself.

        This is the *source-side* rate -- the one RESEARCH.md quotes. It answers
        "if I hold a canonical id, can I talk to platform X", not "will my ESPN
        pool join". Use `coverage_report` for the second question; it is the one
        that actually predicts a broken dashboard.
        """
        players = [r for r in self._by_canonical.values() if not r.is_dst]
        total = len(players)
        rows = [
            SourceCoverage(source, total, sum(1 for r in players if source in r.ids))
            for source in SOURCES
        ]
        return sorted(rows, key=lambda r: r.rate, reverse=True)

    def coverage_report(
        self,
        observed: Mapping[str, Iterable[object]],
        *,
        sample: int = 5,
    ) -> list[SourceCoverage]:
        """Join rate for ids seen in the wild, per source.

        ``observed`` maps a source name to the ids that source actually handed us
        this run. Degradation shows up here first: a crosswalk that stops being
        refreshed produces a slow decline, and an upstream id-space change
        produces a cliff.
        """
        rows: list[SourceCoverage] = []
        for source, values in observed.items():
            total = 0
            resolved = 0
            missing: list[str] = []
            for value in values:
                total += 1
                if self.to_canonical(value, source) is not None:
                    resolved += 1
                elif len(missing) < sample:
                    missing.append(str(value))
            rows.append(SourceCoverage(source, total, resolved, tuple(missing)))
        return rows


def _dst_from_int_like(value: object) -> TeamDefense | None:
    """ESPN-only: recognize the synthetic ``-16xxx`` defense ids."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _dst_from_int(value) if value < 0 else None
    s = str(value).strip()
    if s.startswith("-") and s[1:].isdigit():
        return _dst_from_int(int(s))
    return None


# --------------------------------------------------------------------------
# Cached downloads
# --------------------------------------------------------------------------


def default_season() -> int:
    """The NFL season currently in play. Rolls over in March, with the league year."""
    today = dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def is_stale(path: Path, max_age: dt.timedelta = DEFAULT_MAX_AGE) -> bool:
    if not path.exists():
        return True
    age = dt.datetime.now(dt.UTC) - dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
    return age > max_age


def _download(url: str, dest: Path, *, timeout: float = 60.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=_DOWNLOAD_HEADERS) as client:
        resp = client.get(url)
        resp.raise_for_status()
        body = resp.content
    if not body:
        raise RuntimeError(f"empty body from {url}; refusing to overwrite {dest}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(dest)


def _fetch_text(url: str, *, timeout: float = 15.0) -> str | None:
    try:
        with httpx.Client(
            follow_redirects=True, timeout=timeout, headers=_DOWNLOAD_HEADERS
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text.strip()
    except (httpx.HTTPError, OSError) as exc:
        log.warning("could not read build stamp %s: %s", url, exc)
        return None


def ensure_cached(
    url: str,
    dest: Path,
    *,
    max_age: dt.timedelta = DEFAULT_MAX_AGE,
    allow_download: bool = True,
    stamp_url: str | None = None,
) -> Path:
    """Return `dest`, downloading it first if it is missing or stale.

    When `stamp_url` is given, the remote build stamp is compared against a local
    sidecar before pulling the payload -- nflverse publishes one per release tag,
    so the daily refresh usually costs 24 bytes instead of half a megabyte.
    Comparing the stamp *text* rather than parsing its timestamp avoids taking a
    dependency on a format ("2026-09-07 09:10:27 EDT") that nobody guarantees.

    A download failure with a cached copy on disk logs a warning and returns the
    stale copy. Yesterday's crosswalk is a far better outcome than no analysis.
    """
    if not is_stale(dest, max_age):
        return dest
    if not allow_download:
        if dest.exists():
            return dest
        raise FileNotFoundError(f"{dest} is missing and downloads are disabled")

    if stamp_url is not None and dest.exists():
        stamp_path = dest.with_suffix(dest.suffix + ".stamp")
        remote = _fetch_text(stamp_url)
        if remote and stamp_path.exists() and stamp_path.read_text().strip() == remote:
            dest.touch()
            log.info("%s unchanged upstream (%s); kept cached copy", dest.name, remote)
            return dest

    try:
        _download(url, dest)
    except (httpx.HTTPError, OSError, RuntimeError) as exc:
        if dest.exists():
            log.warning("refresh of %s failed (%s); using the stale cached copy", dest.name, exc)
            return dest
        raise

    if stamp_url is not None and (remote := _fetch_text(stamp_url)):
        dest.with_suffix(dest.suffix + ".stamp").write_text(remote)
    return dest


def load_nflverse_roster(
    season: int | None = None,
    *,
    cache_dir: Path = REFERENCE_DIR,
    max_age: dt.timedelta = DEFAULT_MAX_AGE,
    allow_download: bool = True,
    max_lookback: int = 1,
) -> pl.DataFrame | None:
    """The preferred crosswalk.

    Falls back a season at a time when a file is not cut yet -- the current
    season's roster does not appear until the league year opens in March.
    `max_lookback` is bounded on purpose: an out-of-range season would otherwise
    walk backwards forever, and each step is a real 404 against GitHub.
    """
    season = season or default_season()
    for attempt in range(max_lookback + 1):
        year = season - attempt
        dest = cache_dir / f"roster_{year}.parquet"
        try:
            ensure_cached(
                NFLVERSE_ROSTER_URL.format(season=year),
                dest,
                max_age=max_age,
                allow_download=allow_download,
                stamp_url=NFLVERSE_ROSTER_STAMP_URL,
            )
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            log.warning("no nflverse roster for %d: %s", year, exc)
            continue
        return pl.read_parquet(dest)
    return None


def load_dynastyprocess(
    *,
    cache_dir: Path = REFERENCE_DIR,
    max_age: dt.timedelta = DEFAULT_MAX_AGE,
    allow_download: bool = True,
) -> pl.DataFrame | None:
    """The fallback crosswalk.

    Read with every column as Utf8 on purpose. Type inference on a file whose
    missing marker is the string ``"NA"`` produces a mix of nulls and literals in
    the same column depending on which rows the sampler happened to see.
    """
    dest = cache_dir / "db_playerids.csv"
    try:
        ensure_cached(DYNASTYPROCESS_URL, dest, max_age=max_age, allow_download=allow_download)
    except (httpx.HTTPError, OSError, RuntimeError, FileNotFoundError) as exc:
        log.warning("db_playerids unavailable: %s", exc)
        return None
    return pl.read_csv(dest, infer_schema_length=0)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def espn_pool_coverage(
    resolver: IdResolver,
    *,
    season: int | None = None,
    limit: int = 600,
) -> tuple[SourceCoverage, SourceCoverage]:
    """Join the live ESPN player pool against the spine.

    Returns ``(overall, skill_positions_only)``. The second is the number to alarm
    on: D/ST always resolves through the hard-coded table, and deep-bench free
    agents legitimately fall out of a current-season roster file.
    """
    from ..espn.client import EspnClient
    from ..espn.endpoints import league_default_url

    with EspnClient() as client:
        if season is None:
            season, _ = client.current_season_and_week()
        entries = client.player_pool(
            league_default_url(season, "ppr"),
            limit=250,
            params={"view": "kona_player_info"},
            max_players=limit,
        )

    skill = {"QB", "RB", "WR", "TE"}
    all_ids: list[object] = []
    skill_ids: list[object] = []
    for entry in entries:
        espn_id = entry.get("id")
        all_ids.append(espn_id)
        position = normalize_position((entry.get("player") or {}).get("defaultPositionId"))
        if position in skill:
            skill_ids.append(espn_id)

    overall = resolver.coverage_report({ESPN: all_ids})[0]
    skill_row = resolver.coverage_report({ESPN: skill_ids})[0]
    return overall, SourceCoverage(
        "espn (QB/RB/WR/TE)", skill_row.total, skill_row.resolved, skill_row.missing_sample
    )


def print_coverage(rows: Sequence[SourceCoverage], title: str = "id coverage") -> None:
    """Print per-source join rates. Run it after every sync; watch for decay."""
    from rich.console import Console
    from rich.table import Table

    table = Table("source", "resolved", "total", "rate", "missing sample", title=title)
    for row in sorted(rows, key=lambda r: r.rate, reverse=True):
        table.add_row(
            row.source,
            f"{row.resolved:,}",
            f"{row.total:,}",
            f"{row.rate:.1%}",
            ", ".join(row.missing_sample[:3]),
        )
    Console().print(table)
