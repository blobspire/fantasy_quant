"""The multi-league registry: which leagues we play, and what each one is trying to win.

Everything in this system is keyed on `(league_id, season)`. A player's value is a function
of `(player, league_settings)` and never a global number, so there is no such thing as "the"
league -- scoring, roster slots and league depth all move replacement level, and the same
roster is worth different things in two of your leagues.

The setting that earns this module its own file is `objective`. The whole product optimizes
championship probability, and that is the *wrong* objective in a league that pays for total
points scored: there, variance is not a tool for climbing the standings, it is pure downside,
and a start/sit call that trades expected points for win probability actively destroys money.
High-stakes formats commonly pay a points-leader bonus and admit wildcards on total points,
so the objective is declared per league:

    championship  maximise P(title); accept -EV points for +EV win probability
    points        maximise expected total points-for; ignore win probability entirely
    hybrid        a payout structure that pays for both; set `points_weight` explicitly

Downstream code should read `objective_weights()` rather than branching on the enum, so a
hybrid league does not need a third code path.

Credentials live here too, because this is the configuration module. Values are read from the
environment (or a `.env`) and are never logged, printed or repr'd.
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

if TYPE_CHECKING:  # pragma: no cover - avoids a registry <-> discovery import cycle
    from .espn.discovery import DiscoveredLeague

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/leagues.toml")

#: `.env.example` ships a braced placeholder; treat it as absent rather than as a real SWID.
_PLACEHOLDER_MARKERS = ("XXXX", "your-", "<")


class Objective(StrEnum):
    """What a league actually pays for."""

    CHAMPIONSHIP = "championship"
    POINTS = "points"
    HYBRID = "hybrid"

    @classmethod
    def parse(cls, value: str | Objective) -> Objective:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as err:
            allowed = ", ".join(o.value for o in cls)
            raise ValueError(f"unknown objective {value!r}; expected one of: {allowed}") from err


@dataclass(frozen=True, slots=True)
class Credentials:
    """ESPN cookies. Never repr'd -- `espn_s2` is a session token."""

    swid: str | None = None
    espn_s2: str | None = None

    @property
    def complete(self) -> bool:
        return bool(self.swid and self.espn_s2)

    def as_kwargs(self) -> dict[str, str | None]:
        """Keyword arguments for `EspnClient`. Both or neither; it ignores a half pair."""
        return {"swid": self.swid, "espn_s2": self.espn_s2}

    def __repr__(self) -> str:
        swid = "set" if self.swid else "unset"
        s2 = "set" if self.espn_s2 else "unset"
        return f"Credentials(swid={swid}, espn_s2={s2})"

    __str__ = __repr__


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip().strip('"').strip("'")
    if not stripped or any(marker in stripped for marker in _PLACEHOLDER_MARKERS):
        return None
    return stripped


def normalize_swid(swid: str | None) -> str | None:
    """ESPN wants the braces. A value pasted without them authenticates as nobody."""
    cleaned = _clean(swid)
    if cleaned is None:
        return None
    return cleaned if cleaned.startswith("{") else "{" + cleaned.strip("{}") + "}"


def load_credentials(env_file: Path | str | None = None) -> Credentials:
    """Read `ESPN_SWID` / `ESPN_S2` from the environment, loading a `.env` first.

    The environment wins over the file, so an explicit export can override a stale `.env`.
    Returns a half-empty `Credentials` rather than raising: Phase 0 and every public league
    work unauthenticated, and the caller decides whether it needs auth.
    """
    load_dotenv(dotenv_path=env_file, override=False)
    creds = Credentials(
        swid=normalize_swid(os.environ.get("ESPN_SWID")),
        espn_s2=_clean(os.environ.get("ESPN_S2")),
    )
    if creds.swid and not creds.espn_s2:
        log.warning("ESPN_SWID is set but ESPN_S2 is not; private leagues will 401.")
    return creds


@dataclass(frozen=True, slots=True)
class RegistryDefaults:
    """Applied to any league that does not override them."""

    season: int | None = None
    objective: Objective = Objective.CHAMPIONSHIP
    points_weight: float | None = None
    scoring_variant: str = "ppr"


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """One league in one season, plus the overrides that change how we optimize it."""

    league_id: int
    season: int
    name: str = ""
    #: Our own franchise. Left None when discovery could not identify it.
    team_id: int | None = None
    objective: Objective = Objective.CHAMPIONSHIP
    #: Only meaningful for `hybrid`; see `objective_weights`.
    points_weight: float | None = None
    #: Which of ESPN's canned pools best approximates this league, for cross-league priors.
    scoring_variant: str = "ppr"
    enabled: bool = True
    notes: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # frozen dataclass: assign through object.__setattr__ so the parsed enum sticks.
        object.__setattr__(self, "objective", Objective.parse(self.objective))
        if self.objective is Objective.HYBRID and self.points_weight is None:
            raise ValueError(
                f"league {self.league_id}/{self.season} is objective=hybrid but sets no "
                "points_weight; a hybrid payout has to say how it splits."
            )
        if self.points_weight is not None and not 0.0 <= self.points_weight <= 1.0:
            raise ValueError(
                f"points_weight for league {self.league_id}/{self.season} must be in [0, 1], "
                f"got {self.points_weight!r}"
            )
        # `objective_weights` reads points_weight ahead of the enum, so a weight attached to
        # a non-hybrid objective silently overrides it: objective="points" with
        # points_weight=0.0 comes back as (1.0, 0.0), the exact opposite of what it says.
        # A split is what `hybrid` *means*, so say so rather than resolving the conflict.
        if self.objective is not Objective.HYBRID and self.points_weight is not None:
            raise ValueError(
                f"league {self.league_id}/{self.season} sets points_weight="
                f"{self.points_weight!r} with objective={self.objective.value}; a split "
                "only means something for objective=hybrid, and attaching one here would "
                'silently override the objective. Use objective="hybrid".'
            )

    @property
    def key(self) -> tuple[int, int]:
        return (self.league_id, self.season)

    def objective_weights(self) -> tuple[float, float]:
        """(championship weight, points weight), summing to 1.

        Read this instead of branching on the enum, so a hybrid payout is a number rather
        than a third code path.
        """
        if self.points_weight is not None:
            return (1.0 - self.points_weight, self.points_weight)
        return (0.0, 1.0) if self.objective is Objective.POINTS else (1.0, 0.0)

    @property
    def optimizes_points(self) -> bool:
        """True when weekly win probability is the wrong thing to maximize here."""
        return self.objective_weights()[1] > 0.5

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "league_id": self.league_id,
            "season": self.season,
            "name": self.name,
            "objective": self.objective.value,
            "scoring_variant": self.scoring_variant,
            "enabled": self.enabled,
        }
        if self.team_id is not None:
            out["team_id"] = self.team_id
        if self.points_weight is not None:
            out["points_weight"] = self.points_weight
        if self.notes:
            out["notes"] = self.notes
        if self.tags:
            out["tags"] = list(self.tags)
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], defaults: RegistryDefaults) -> LeagueConfig:
        if "league_id" not in raw:
            raise ValueError(f"league entry has no league_id: {dict(raw)!r}")
        season = raw.get("season", defaults.season)
        if season is None:
            raise ValueError(
                f"league {raw['league_id']} has no season and [defaults] sets none; "
                "the registry keys on (league_id, season) so it cannot be inferred."
            )
        objective = Objective.parse(raw.get("objective", defaults.objective))
        return cls(
            league_id=int(raw["league_id"]),
            season=int(season),
            name=str(raw.get("name") or ""),
            team_id=int(raw["team_id"]) if raw.get("team_id") is not None else None,
            objective=objective,
            # A `[defaults] points_weight` exists for the hybrid leagues; inheriting it
            # everywhere would quietly turn every championship league into a 50/50 blend.
            points_weight=(
                float(raw["points_weight"])
                if raw.get("points_weight") is not None
                else (defaults.points_weight if objective is Objective.HYBRID else None)
            ),
            scoring_variant=str(raw.get("scoring_variant") or defaults.scoring_variant),
            enabled=bool(raw.get("enabled", True)),
            notes=str(raw.get("notes") or ""),
            tags=tuple(str(t) for t in (raw.get("tags") or [])),
        )


@dataclass(slots=True)
class Registry:
    """Every league we manage, keyed on `(league_id, season)`."""

    defaults: RegistryDefaults = field(default_factory=RegistryDefaults)
    leagues: dict[tuple[int, int], LeagueConfig] = field(default_factory=dict)
    #: League ids to pull even if Fan API discovery misses them. Not a fallback of last
    #: resort -- the Fan API response shape is community-reported and unverified, so this
    #: is the path that is guaranteed to work.
    manual_league_ids: tuple[int, ...] = ()
    path: Path | None = None

    # -- container behavior ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.leagues)

    def __iter__(self) -> Iterator[LeagueConfig]:
        return iter(self.sorted())

    def __contains__(self, key: object) -> bool:
        return key in self.leagues

    def sorted(self) -> list[LeagueConfig]:
        return [self.leagues[k] for k in sorted(self.leagues)]

    def get(self, league_id: int, season: int) -> LeagueConfig | None:
        return self.leagues.get((league_id, season))

    def require(self, league_id: int, season: int) -> LeagueConfig:
        cfg = self.get(league_id, season)
        if cfg is None:
            raise KeyError(
                f"league {league_id} season {season} is not in the registry "
                f"({len(self.leagues)} configured)"
            )
        return cfg

    def for_season(self, season: int, *, enabled_only: bool = True) -> list[LeagueConfig]:
        return [c for c in self.sorted() if c.season == season and (c.enabled or not enabled_only)]

    def active(self) -> list[LeagueConfig]:
        """Enabled leagues in the default season, or all enabled leagues if none is set."""
        if self.defaults.season is None:
            return [c for c in self.sorted() if c.enabled]
        return self.for_season(self.defaults.season)

    # -- mutation ----------------------------------------------------------------------

    def upsert(self, config: LeagueConfig) -> LeagueConfig:
        """Add or replace one league. Returns what is now stored."""
        self.leagues[config.key] = config
        return config

    def merge_discovered(
        self,
        discovered: Iterable[DiscoveredLeague],
        *,
        overwrite: bool = False,
    ) -> list[LeagueConfig]:
        """Fold discovery results in without clobbering hand-tuned overrides.

        An existing entry keeps its objective, notes and tags -- those are human decisions
        that discovery cannot know -- but picks up a `team_id` or `name` it was missing.
        Pass `overwrite=True` only when the config is meant to be regenerated.
        """
        added: list[LeagueConfig] = []
        for item in discovered:
            key = (item.league_id, item.season)
            existing = self.leagues.get(key)
            if existing is None:
                cfg = LeagueConfig(
                    league_id=item.league_id,
                    season=item.season,
                    name=item.name or "",
                    team_id=item.team_id,
                    objective=self.defaults.objective,
                    points_weight=(
                        self.defaults.points_weight
                        if self.defaults.objective is Objective.HYBRID
                        else None
                    ),
                    scoring_variant=self.defaults.scoring_variant,
                )
                self.leagues[key] = cfg
                added.append(cfg)
                continue
            if overwrite:
                self.leagues[key] = replace(
                    existing,
                    name=item.name or existing.name,
                    team_id=item.team_id if item.team_id is not None else existing.team_id,
                )
                continue
            patch: dict[str, Any] = {}
            if not existing.name and item.name:
                patch["name"] = item.name
            if existing.team_id is None and item.team_id is not None:
                patch["team_id"] = item.team_id
            if patch:
                self.leagues[key] = replace(existing, **patch)
        return added

    # -- serialization -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "objective": self.defaults.objective.value,
            "scoring_variant": self.defaults.scoring_variant,
        }
        if self.defaults.season is not None:
            defaults["season"] = self.defaults.season
        if self.defaults.points_weight is not None:
            defaults["points_weight"] = self.defaults.points_weight

        out: dict[str, Any] = {"defaults": defaults}
        if self.manual_league_ids:
            out["discovery"] = {"manual_league_ids": list(self.manual_league_ids)}
        out["leagues"] = [c.to_dict() for c in self.sorted()]
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], path: Path | None = None) -> Registry:
        raw_defaults = raw.get("defaults") or {}
        defaults = RegistryDefaults(
            season=int(raw_defaults["season"]) if raw_defaults.get("season") is not None else None,
            objective=Objective.parse(raw_defaults.get("objective", Objective.CHAMPIONSHIP)),
            points_weight=(
                float(raw_defaults["points_weight"])
                if raw_defaults.get("points_weight") is not None
                else None
            ),
            scoring_variant=str(raw_defaults.get("scoring_variant") or "ppr"),
        )

        leagues: dict[tuple[int, int], LeagueConfig] = {}
        for entry in raw.get("leagues") or []:
            cfg = LeagueConfig.from_dict(entry, defaults)
            if cfg.key in leagues:
                raise ValueError(
                    f"duplicate league entry {cfg.league_id}/{cfg.season}; the registry keys "
                    "on (league_id, season) and cannot hold two."
                )
            leagues[cfg.key] = cfg

        discovery = raw.get("discovery") or {}
        return cls(
            defaults=defaults,
            leagues=leagues,
            manual_league_ids=tuple(int(i) for i in (discovery.get("manual_league_ids") or [])),
            path=path,
        )

    def to_toml(self) -> str:
        return _dump_toml(self.to_dict())

    @classmethod
    def from_toml(cls, text: str, path: Path | None = None) -> Registry:
        return cls.from_dict(tomllib.loads(text), path=path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH, *, missing_ok: bool = True) -> Registry:
        """Read the config. JSON is accepted too, chosen by file suffix.

        A missing file yields an empty registry by default: the very first `fq sync` runs
        before the file exists and should be able to write one from discovery.
        """
        p = Path(path)
        if not p.exists():
            if missing_ok:
                log.info("no league registry at %s; starting empty", p)
                return cls(path=p)
            raise FileNotFoundError(f"no league registry at {p}")
        text = p.read_text(encoding="utf-8")
        raw = json.loads(text) if p.suffix.lower() == ".json" else tomllib.loads(text)
        return cls.from_dict(raw, path=p)

    def save(self, path: Path | str | None = None) -> Path:
        """Write the config, atomically. Returns the path written."""
        dest = Path(path or self.path or DEFAULT_CONFIG_PATH)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = (
            json.dumps(self.to_dict(), indent=2) + "\n"
            if dest.suffix.lower() == ".json"
            else _HEADER + self.to_toml()
        )
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(dest)
        self.path = dest
        return dest


_HEADER = """\
# fantasy_quant league registry. Everything is keyed on (league_id, season).
#
# objective:
#   championship  maximise P(title). Variance is a tool: trail the cut line and you want
#                 more of it, lead it and you want less.
#   points        the league pays for total points-for. Win probability is irrelevant and
#                 optimizing it actively destroys value here.
#   hybrid        pays for both; set points_weight in [0, 1] to say how it splits.
#
# [discovery].manual_league_ids is pulled whether or not Fan API discovery finds them.

"""


# --------------------------------------------------------------------------------------
# A minimal TOML writer
# --------------------------------------------------------------------------------------
#
# The stdlib reads TOML (`tomllib`) but does not write it, and the config schema here is
# small and closed -- a table of scalars, a table of int lists, and an array of tables of
# scalars. That is a dozen lines of emitter versus a dependency, so it is a dozen lines of
# emitter. Anything outside that shape raises rather than emitting something tomllib will
# refuse to read back.


#: TOML basic strings may not carry a raw control character, and only these have a short
#: escape. Anything else below 0x20 (plus DEL) has to go out as \uXXXX or the file we write
#: is one `tomllib` refuses to read back -- an ESPN league name is user-entered text and a
#: stray \r out of a pasted string is enough to do it.
_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _toml_string(value: str) -> str:
    out = ['"']
    for ch in value:
        escape = _TOML_ESCAPES.get(ch)
        if escape is not None:
            out.append(escape)
        elif ch < " " or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, Sequence):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot serialize {type(value).__name__} to TOML: {value!r}")


def _dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append(f"\n[{key}]")
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in value.items())
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                if not isinstance(item, Mapping):
                    raise TypeError(f"top-level array {key!r} must hold tables, got {item!r}")
                lines.append(f"\n[[{key}]]")
                lines.extend(f"{k} = {_toml_value(v)}" for k, v in item.items())
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines).lstrip("\n") + "\n"
