"""nflverse release mirror -- schedule, Vegas lines, weekly stats, usage.

Reads the GitHub release assets directly rather than going through `nflreadpy`
(one more dependency for a URL join) or `nfl_data_py` (deprecated). Every file
lands in a local cache keyed on the release tag's own `timestamp.txt`, so the
daily poll costs 24 bytes instead of re-pulling a multi-megabyte Parquet to
discover it did not change.

**Re-pull mid-week in season.** The NFL issues stat corrections Wednesday and
Thursday, and nflverse rebuilds behind them, so a Tuesday copy of
`stats_player_week_{season}` is not the number that ends up in the record. The
default poll window is short enough to catch that; do not raise it.

Verified against the live releases on 2026-09-07. Traps, all measured:

* **`games.parquet`, not `games.csv`.** Same 7,548 rows either way, but the CSV
  types the `espn` and `old_game_id` crosswalk columns as Int64 while the Parquet
  keeps them String. Those columns exist to join against ESPN, so the string form
  is the one that works.
* **`spread_line` is home-relative** (positive = home favored; corr with home
  margin +0.50 over 2025). Anything team-oriented has to flip sign for the away
  side -- `team_weeks` does it and publishes `team_spread`.
* **Next Gen Stats per-season files are abandoned stubs.**
  `ngs_2024_receiving.csv.gz` holds 8 rows -- four players from the 2024 opener,
  duplicated at week 0 and week 1 -- against 1,435 in the consolidated file for
  the same season. It parses cleanly and looks like data, which is worse than
  being empty. 2025+ per-season files 404 outright. Only the consolidated
  `ngs_{stat}.parquet` is real, and it is the only one this module exposes.
* **The consolidated NGS file mixes season totals in with the weekly rows.**
  `week == 0` is a season aggregate that repeats the weekly rows exactly
  (Ja'Marr Chase, 2024: the week-0 row carries 175 targets and his 17 weekly
  rows sum to 175). A `group_by(player).sum()` over the raw frame therefore
  doubles every player. Filter `week > 0` for weekly work.
* **Depth charts changed schema at 2025.** 2024 and earlier: `season`,
  `club_code`, `week`, `depth_team`, `position`. 2025 and later: `dt`, `team`,
  `pos_grp`/`pos_abb`/`pos_rank`, and *no season or week column at all*. The two
  cannot be concatenated, so a request spanning the boundary is refused.
* **Depth charts are append-only snapshots.** 503,245 rows for 2026 across 171
  `dt` pulls; the CSV is 47 MB, the Parquet 2.3 MB. `latest_only=True` keeps the
  newest snapshot per team. It is not one row per player -- a player legitimately
  appears at several `pos_slot` values in the same snapshot.
* **A team does not always have exactly one bye.** 2022 carries 271 regular-season
  games because the abandoned Bills-Bengals game was never replayed and nflverse
  drops the row outright, so BUF and CIN each show two idle weeks. See
  `bye_weeks` / `idle_weeks`.
* **The current season's weekly files do not exist until games are played.**
  `stats_player_week_2026`, `snap_counts_2026` and `advstats_week_*_2026` all 404
  today. That surfaces as `NflverseNotFound`; pass `allow_missing=True` to skip.

**Do not build on the nflverse `injuries` tag.** Its timestamp still moves daily,
which makes it look alive, but `injuries_2026.parquet` holds 11 rows for the whole
league with `report_status` null on every one. Use MFL
(`api.myfantasyleague.com/{season}/export?TYPE=injuries`) instead.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl

log = logging.getLogger(__name__)

RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
DEFAULT_ROOT = Path("data/cache/nflverse")

# How long a cached file is trusted before the tag's timestamp.txt is polled again.
# Short on purpose: stat corrections land Wed/Thu and nflverse rebuilds behind them.
DEFAULT_POLL_AFTER = dt.timedelta(hours=4)

# nflverse replaced the depth-chart source for 2025; see the module docstring.
DEPTH_CHART_SCHEMA_EPOCH = 2025

_USER_AGENT = "fantasy_quant/0.1 (+https://github.com/blobspire/fantasy_quant)"

_NGS_STATS = ("passing", "receiving", "rushing")
_PFR_STATS = ("pass", "rush", "rec", "def")
# PFR's per-week advanced splits start in 2018; the season rollups go back further.
_PFR_WEEK_FIRST_SEASON = 2018


class NflverseError(RuntimeError):
    """An nflverse fetch or load failed in a way worth surfacing."""


class NflverseNotFound(NflverseError):
    """A release asset does not exist -- usually a season nflverse has not built yet."""


def timestamp_url(tag: str) -> str:
    """The tag's build stamp. 24 bytes, and the only thing worth polling."""
    return f"{RELEASE_BASE}/{tag}/timestamp.txt"


@dataclass(frozen=True, slots=True)
class Asset:
    """One file inside one nflverse release tag."""

    tag: str
    filename: str

    @property
    def url(self) -> str:
        return f"{RELEASE_BASE}/{self.tag}/{self.filename}"

    def __str__(self) -> str:
        return f"{self.tag}/{self.filename}"


SCHEDULES = Asset("schedules", "games.parquet")
PLAYERS = Asset("players", "players.parquet")


@dataclass(frozen=True, slots=True)
class CacheMeta:
    """Sidecar recording what we have and when we last confirmed it.

    `release_stamp` is the verbatim body of the tag's `timestamp.txt` at download
    time. `checked_at` moves on every successful poll even when nothing is
    downloaded -- that is the whole point of polling.
    """

    release_stamp: str
    fetched_at: dt.datetime
    checked_at: dt.datetime
    size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "release_stamp": self.release_stamp,
            "fetched_at": self.fetched_at.isoformat(),
            "checked_at": self.checked_at.isoformat(),
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> CacheMeta:
        return cls(
            release_stamp=str(raw["release_stamp"]),
            fetched_at=_as_utc(dt.datetime.fromisoformat(raw["fetched_at"])),
            checked_at=_as_utc(dt.datetime.fromisoformat(raw["checked_at"])),
            size=int(raw["size"]),
        )


def _as_utc(when: dt.datetime) -> dt.datetime:
    """Naive timestamps predate this field carrying a zone; treat them as UTC."""
    return when.replace(tzinfo=dt.UTC) if when.tzinfo is None else when.astimezone(dt.UTC)


def needs_poll(meta: CacheMeta | None, now: dt.datetime, poll_after: dt.timedelta) -> bool:
    """Whether the tag's `timestamp.txt` is worth a request right now."""
    if meta is None:
        return True
    return now - meta.checked_at >= poll_after


def is_stale(meta: CacheMeta | None, release_stamp: str | None) -> bool:
    """Whether the cached file must be re-downloaded.

    A `None` stamp means the poll itself failed. A possibly-stale local copy beats
    no data, so that case is deliberately *not* stale -- the caller already logged
    the failure.
    """
    if meta is None:
        return True
    if release_stamp is None:
        return False
    return meta.release_stamp != release_stamp


class NflverseCache:
    """Local mirror of nflverse release assets, refreshed on tag timestamp changes."""

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        *,
        poll_after: dt.timedelta = DEFAULT_POLL_AFTER,
        offline: bool = False,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.root = Path(root)
        self.poll_after = poll_after
        self.offline = offline
        self._owns_client = client is None
        self._client = client or httpx.Client(
            follow_redirects=True,  # release downloads 302 to objects.githubusercontent.com
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        )
        # One timestamp poll per tag per process, however many files that tag serves.
        self._stamps: dict[str, str] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> NflverseCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def local_path(self, asset: Asset) -> Path:
        return self.root / asset.tag / asset.filename

    def _meta_path(self, asset: Asset) -> Path:
        return self.local_path(asset).with_name(asset.filename + ".meta.json")

    def _read_meta(self, asset: Asset) -> CacheMeta | None:
        path = self._meta_path(asset)
        if not path.exists():
            return None
        try:
            return CacheMeta.from_dict(json.loads(path.read_text()))
        except (ValueError, TypeError, KeyError, OSError):
            # TypeError included on purpose: a sidecar with a JSON `null` in
            # `size` or `fetched_at` reaches int()/fromisoformat() as None, and
            # a corrupt sidecar must force a refetch, not kill the run.
            log.warning("unreadable cache metadata at %s; refetching", path)
            return None

    def _write_meta(self, asset: Asset, meta: CacheMeta) -> None:
        self._meta_path(asset).write_text(json.dumps(meta.as_dict(), indent=2))

    def release_timestamp(self, tag: str) -> str | None:
        """Body of the tag's `timestamp.txt`, or None if it could not be read."""
        if tag in self._stamps:
            return self._stamps[tag]
        url = timestamp_url(tag)
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("could not poll %s: %s", url, exc)
            return None
        stamp = resp.text.strip()
        self._stamps[tag] = stamp
        return stamp

    def ensure(self, asset: Asset, *, force: bool = False) -> Path:
        """Return a local path to `asset`, downloading only when it actually changed."""
        path = self.local_path(asset)
        meta = self._read_meta(asset)
        if not path.exists():
            meta = None

        if self.offline:
            if meta is None:
                raise NflverseError(f"{asset} is not cached at {path} and the cache is offline.")
            return path

        now = dt.datetime.now(dt.UTC)
        if not force and meta is not None and not needs_poll(meta, now, self.poll_after):
            return path

        stamp = self.release_timestamp(asset.tag)
        if not force and meta is not None and not is_stale(meta, stamp):
            if stamp is None:
                # The poll itself failed. Serving the cached copy is right, but
                # recording a *successful* check is not: it would shut the poll
                # window for another `poll_after` and hide a stat correction
                # that landed while the network was down. Leave `checked_at`
                # where it was so the next call retries.
                return path
            # Unchanged upstream: record that we checked, and skip the big download.
            self._write_meta(asset, CacheMeta(meta.release_stamp, meta.fetched_at, now, meta.size))
            return path

        size = self._download(asset, path)
        self._write_meta(asset, CacheMeta(stamp or "", now, now, size))
        log.info("fetched %s (%s bytes, release %s)", asset, f"{size:,}", stamp or "unknown")
        return path

    def _download(self, asset: Asset, dest: Path) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        written = 0
        try:
            with self._client.stream("GET", asset.url) as resp:
                if resp.status_code == 404:
                    resp.read()
                    raise NflverseNotFound(f"{asset} does not exist ({asset.url}).")
                if resp.status_code != 200:
                    resp.read()
                    raise NflverseError(f"HTTP {resp.status_code} fetching {asset.url}")
                declared = resp.headers.get("content-length")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        written += len(chunk)
                        fh.write(chunk)
            if declared is not None:
                try:
                    expected = int(declared)
                except ValueError:
                    # Surface as NflverseError; a bare ValueError escapes every
                    # caller that guards the fetch with `except NflverseError`.
                    raise NflverseError(
                        f"{asset}: unparseable content-length {declared!r}"
                    ) from None
                if expected != written:
                    # A truncated body would otherwise land as a plausible-looking file.
                    raise NflverseError(
                        f"{asset} truncated: {written} bytes received, {declared} declared."
                    )
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(dest)  # atomic; a half-written file must never be readable
        return written

    def frame(self, asset: Asset, *, force: bool = False) -> pl.DataFrame:
        """Fetch if needed, then read into polars."""
        return _read_frame(self.ensure(asset, force=force))


def _read_frame(path: Path) -> pl.DataFrame:
    name = path.name
    if name.endswith(".parquet"):
        return pl.read_parquet(path)
    if name.endswith((".csv", ".csv.gz")):
        # infer_schema_length=None scans the whole file: nflverse CSVs carry long
        # runs of nulls before the first real value in several columns.
        return pl.read_csv(path, infer_schema_length=None)
    raise NflverseError(f"no reader for {path.name}; use a .parquet or .csv asset.")


_DEFAULT_CACHE: NflverseCache | None = None


def default_cache() -> NflverseCache:
    """Process-wide cache, so one run polls each tag once."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = NflverseCache()
    return _DEFAULT_CACHE


def _cache(cache: NflverseCache | None) -> NflverseCache:
    return cache if cache is not None else default_cache()


def _season_tuple(seasons: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(seasons, str):
        # A str is Iterable, so the generic branch would shred "2026" into
        # (2, 0, 2, 6) and every downstream filter would quietly return nothing.
        raise TypeError(f"seasons must be an int or an iterable of ints, not the str {seasons!r}")
    if isinstance(seasons, int):
        return (seasons,)
    out = tuple(int(s) for s in seasons)
    if not out:
        raise ValueError("no seasons requested")
    return out


def _load_per_season(
    tag: str,
    template: str,
    seasons: tuple[int, ...],
    *,
    cache: NflverseCache,
    allow_missing: bool,
) -> pl.DataFrame:
    """Load one file per season and stack them.

    `diagonal_relaxed` rather than a plain vertical concat: nflverse adds stat
    columns between seasons and narrows integer widths, and a union-with-nulls is
    the honest representation of that.
    """
    frames: list[pl.DataFrame] = []
    missing: list[int] = []
    for season in seasons:
        asset = Asset(tag, template.format(season=season))
        try:
            df = cache.frame(asset)
        except NflverseNotFound:
            if not allow_missing:
                raise NflverseNotFound(
                    f"{asset} is not published. nflverse builds a season's weekly files "
                    f"only once games have been played; pass allow_missing=True to skip it."
                ) from None
            log.warning("%s not published; skipping", asset)
            missing.append(season)
            continue
        if "season" not in df.columns:
            # depth_charts (2025+) carries no season column at all.
            df = df.with_columns(pl.lit(season, dtype=pl.Int32).alias("season"))
        frames.append(df)

    if not frames:
        raise NflverseNotFound(f"none of {list(seasons)} are published under tag {tag!r}.")
    if missing:
        loaded = [s for s in seasons if s not in missing]
        log.info("%s: loaded %s, missing %s", tag, loaded, missing)
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal_relaxed")


def schedules(
    seasons: int | Iterable[int] | None = None,
    *,
    game_types: Sequence[str] | None = None,
    cache: NflverseCache | None = None,
) -> pl.DataFrame:
    """The full game log back to 1999, with closing spread/total and ID crosswalks.

    This is also the free historical Vegas source: `spread_line` and `total_line`
    are closing numbers, and `espn`/`pfr` give the game-id joins. Note the sign
    convention -- `spread_line` is positive when the *home* team is favored.

    Lines are only present once a market has posted, so a future season is mostly
    null: 160 of the 272 2026 games had no spread on 2026-09-07. Treat a null as
    "not priced yet", not as a pick'em.
    """
    # Validate before fetching: a bad argument should not cost a download first.
    wanted = None if seasons is None else _season_tuple(seasons)
    df = _cache(cache).frame(SCHEDULES)
    if wanted is not None:
        df = df.filter(pl.col("season").is_in(list(wanted)))
    if game_types is not None:
        df = df.filter(pl.col("game_type").is_in(list(game_types)))
    return df


def player_week_stats(
    seasons: int | Iterable[int],
    *,
    season_type: str | None = "REG",
    allow_missing: bool = False,
    cache: NflverseCache | None = None,
) -> pl.DataFrame:
    """Weekly player box scores -- the DvP and opportunity input.

    Carries the stable usage metrics directly (`target_share`, `air_yards_share`,
    `wopr`) alongside the unstable ones that need regressing, plus `opponent_team`
    for the two-way offense/defense decomposition. `season_type=None` keeps the
    postseason rows too.
    """
    df = _load_per_season(
        "stats_player",
        "stats_player_week_{season}.parquet",
        _season_tuple(seasons),
        cache=_cache(cache),
        allow_missing=allow_missing,
    )
    if season_type is not None:
        df = df.filter(pl.col("season_type") == season_type)
    return df


def rosters(
    seasons: int | Iterable[int],
    *,
    allow_missing: bool = False,
    cache: NflverseCache | None = None,
) -> pl.DataFrame:
    """Weekly rosters -- the primary ID spine.

    `espn_id` covers 86% of QB/RB/WR/TE/K on the live 2026 file, which beats every
    other free crosswalk. It is a String here and null (not `"NA"`) where absent,
    unlike DynastyProcess's `db_playerids`.
    """
    return _load_per_season(
        "rosters",
        "roster_{season}.parquet",
        _season_tuple(seasons),
        cache=_cache(cache),
        allow_missing=allow_missing,
    )


def snap_counts(
    seasons: int | Iterable[int],
    *,
    allow_missing: bool = False,
    cache: NflverseCache | None = None,
) -> pl.DataFrame:
    """Per-game snap counts and shares, 2012+.

    These post Monday/Tuesday, which is the waiver-timing window: claims are
    already blind by then and most of the league has not looked.

    Unlike `player_week_stats`, this keeps the postseason: the file carries a
    `game_type` column and weeks run 1-22. Playoff weeks do not collide with
    regular-season ones, so a (player, season, week) join is still safe, but
    filter `game_type == "REG"` before any per-week aggregate.
    """
    return _load_per_season(
        "snap_counts",
        "snap_counts_{season}.parquet",
        _season_tuple(seasons),
        cache=_cache(cache),
        allow_missing=allow_missing,
    )


def depth_charts(
    seasons: int | Iterable[int],
    *,
    latest_only: bool = True,
    allow_missing: bool = False,
    cache: NflverseCache | None = None,
) -> pl.DataFrame:
    """Depth charts, deduplicated to the most recent snapshot by default.

    The 2025+ file is append-only: 2026 holds 171 timestamped pulls of all 32
    teams. `latest_only` keeps the newest `dt` per (season, team) -- deliberately
    not one row per player, because a player really does hold several `pos_slot`
    entries in a single snapshot.

    Seasons on either side of 2025 have incompatible schemas and cannot be mixed;
    that is refused rather than silently unioned into a frankenframe. The legacy
    files are week-partitioned rather than append-only, so they are already
    deduplicated and `latest_only` is a no-op on them.
    """
    wanted = _season_tuple(seasons)
    old = [s for s in wanted if s < DEPTH_CHART_SCHEMA_EPOCH]
    new = [s for s in wanted if s >= DEPTH_CHART_SCHEMA_EPOCH]
    if old and new:
        raise NflverseError(
            f"depth charts changed schema at {DEPTH_CHART_SCHEMA_EPOCH}: "
            f"{old} use season/club_code/week/depth_team, {new} use dt/team/pos_*. "
            "Load the two eras separately."
        )

    df = _load_per_season(
        "depth_charts",
        "depth_charts_{season}.parquet",
        wanted,
        cache=_cache(cache),
        allow_missing=allow_missing,
    )
    if not latest_only:
        return df
    if "dt" not in df.columns:
        # Pre-2025 files carry one row per team-week already, and their `week`
        # runs on into the playoffs (19-22) with a null-week SBBYE type on top,
        # so "the latest week" is not a dedupe -- it is a different question.
        log.info(
            "depth charts before %d are already per-week; not deduping",
            DEPTH_CHART_SCHEMA_EPOCH,
        )
        return df
    # dt is an ISO-8601 UTC string, so a lexicographic max is a chronological max.
    return df.filter(pl.col("dt") == pl.col("dt").max().over(["season", "team"]))


def pfr_advstats(
    seasons: int | Iterable[int] | None = None,
    *,
    stat: str = "rec",
    frequency: str = "week",
    allow_missing: bool = False,
    cache: NflverseCache | None = None,
) -> pl.DataFrame:
    """Pro Football Reference advanced splits: broken tackles, drops, air yards.

    `stat` is one of pass/rush/rec/def. `frequency="week"` needs explicit seasons
    (2018+, one file each); `frequency="season"` reads the consolidated rollup and
    filters in memory.
    """
    if stat not in _PFR_STATS:
        raise ValueError(f"unknown stat {stat!r}; expected one of {list(_PFR_STATS)}")

    if frequency == "season":
        wanted_seasons = None if seasons is None else _season_tuple(seasons)
        df = _cache(cache).frame(Asset("pfr_advstats", f"advstats_season_{stat}.parquet"))
        if wanted_seasons is not None:
            df = df.filter(pl.col("season").is_in(list(wanted_seasons)))
        return df
    if frequency != "week":
        raise ValueError(f"unknown frequency {frequency!r}; expected 'week' or 'season'")

    if seasons is None:
        raise ValueError("frequency='week' needs explicit seasons; the files are per-season.")
    wanted = _season_tuple(seasons)
    if too_early := [s for s in wanted if s < _PFR_WEEK_FIRST_SEASON]:
        raise ValueError(
            f"PFR weekly advanced stats start in {_PFR_WEEK_FIRST_SEASON}; got {too_early}."
        )
    return _load_per_season(
        "pfr_advstats",
        f"advstats_week_{stat}_{{season}}.parquet",
        wanted,
        cache=_cache(cache),
        allow_missing=allow_missing,
    )


def nextgen_stats(
    stat: str = "receiving",
    *,
    seasons: int | Iterable[int] | None = None,
    cache: NflverseCache | None = None,
) -> pl.DataFrame:
    """Next Gen Stats -- separation, cushion, air yards, YAC over expected.

    Consolidated file only, and filtered in memory. The per-season assets are
    abandoned stubs from 2024 (8 rows) and absent entirely from 2025; one of
    them parses cleanly and looks like data, which is the worst failure mode
    available.

    **`week == 0` rows are season aggregates, not a week.** They repeat the
    weekly rows exactly -- Ja'Marr Chase's 2024 week-0 row shows 175 targets and
    his 17 weekly rows sum to 175 -- so any `group_by(player).sum()` over the raw
    frame doubles every player. Filter `week > 0` for weekly work. POST rows are
    present too (`season_type`).
    """
    if stat not in _NGS_STATS:
        raise ValueError(f"unknown stat {stat!r}; expected one of {list(_NGS_STATS)}")
    wanted = None if seasons is None else _season_tuple(seasons)
    df = _cache(cache).frame(Asset("nextgen_stats", f"ngs_{stat}.parquet"))
    if wanted is not None:
        df = df.filter(pl.col("season").is_in(list(wanted)))
    return df


def players(*, cache: NflverseCache | None = None) -> pl.DataFrame:
    """The full nflverse player dictionary (every ID space, one row per player)."""
    return _cache(cache).frame(PLAYERS)


# --- schedule views -------------------------------------------------------------

REGULAR_SEASON = "REG"

_REQUIRED_SCHEDULE_COLS = ("season", "game_type", "week", "home_team", "away_team")

# Carried through to the team-week view when present, sign-corrected where needed.
_TEAM_WEEK_CARRY = (
    "game_id",
    "gameday",
    "weekday",
    "gametime",
    "location",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium",
    "espn",
    "pfr",
    "spread_line",
    "total_line",
)


def _regular_season(schedule: pl.DataFrame, season: int) -> pl.DataFrame:
    if missing := [c for c in _REQUIRED_SCHEDULE_COLS if c not in schedule.columns]:
        raise NflverseError(f"schedule frame is missing required columns {missing}")
    reg = schedule.filter((pl.col("season") == season) & (pl.col("game_type") == REGULAR_SEASON))
    if reg.height == 0:
        raise NflverseError(f"no {REGULAR_SEASON} games for season {season} in this schedule.")
    return reg


def team_weeks(
    schedule: pl.DataFrame,
    season: int,
    *,
    from_week: int = 1,
    through_week: int | None = None,
    include_byes: bool = True,
) -> pl.DataFrame:
    """One row per (team, regular-season week), byes included as null-opponent rows.

    Market columns are re-expressed from the team's point of view, which the raw
    schedule does not do:

    * `team_spread` -- positive when *this* team is favored (raw `spread_line` is
      always home-relative, so the away side is negated).
    * `team_implied_total` / `opponent_implied_total` -- the standard
      `total/2 ± spread/2` decomposition, and the thing a schedule-strength model
      actually wants.

    `is_home` is the *nominal* home team, which is what the market prices; seven
    of 2025's regular-season games were at neutral sites. Read the `location`
    column (carried through) when the distinction matters. The raw home-relative
    `spread_line` is carried through as well, so do not reach for it by accident
    when you meant `team_spread`.
    """
    reg = _regular_season(schedule, season)
    carry = [c for c in _TEAM_WEEK_CARRY if c in reg.columns]
    has_spread = "spread_line" in reg.columns
    has_rest = "home_rest" in reg.columns and "away_rest" in reg.columns

    def side(team_col: str, opp_col: str, home: bool) -> pl.DataFrame:
        cols = [
            pl.col("week"),
            pl.col(team_col).alias("team"),
            pl.col(opp_col).alias("opponent"),
            pl.lit(home).alias("is_home"),
        ]
        if has_spread:
            spread = pl.col("spread_line") if home else -pl.col("spread_line")
            cols.append(spread.alias("team_spread"))
        if has_rest:
            cols.append(pl.col("home_rest" if home else "away_rest").alias("team_rest"))
            cols.append(pl.col("away_rest" if home else "home_rest").alias("opponent_rest"))
        cols.extend(pl.col(c) for c in carry)
        return reg.select(cols)

    played = pl.concat(
        [side("home_team", "away_team", True), side("away_team", "home_team", False)]
    )

    week_dtype = reg.schema["week"]
    last_week = int(reg["week"].max())
    grid = pl.DataFrame(
        {"team": sorted(set(reg["home_team"].to_list()) | set(reg["away_team"].to_list()))}
    ).join(
        pl.DataFrame({"week": pl.Series(range(1, last_week + 1), dtype=week_dtype)}),
        how="cross",
    )

    out = grid.join(played, on=["team", "week"], how="left").with_columns(
        pl.col("opponent").is_null().alias("is_bye")
    )
    if has_spread and "total_line" in out.columns:
        half_total = pl.col("total_line") / 2
        half_spread = pl.col("team_spread") / 2
        out = out.with_columns(
            (half_total + half_spread).alias("team_implied_total"),
            (half_total - half_spread).alias("opponent_implied_total"),
        )
    if not include_byes:
        out = out.filter(~pl.col("is_bye"))

    through = last_week if through_week is None else through_week
    return out.filter(pl.col("week").is_between(from_week, through)).sort(["team", "week"])


def idle_weeks(schedule: pl.DataFrame, season: int) -> dict[str, list[int]]:
    """team -> every regular-season week with no game, ascending.

    The unopinionated view. Almost always a single-element list, but see
    `bye_weeks` for when it is not.
    """
    weeks = team_weeks(schedule, season)
    idle = weeks.filter(pl.col("is_bye")).group_by("team").agg(pl.col("week").sort()).sort("team")
    found = {
        team: [int(w) for w in ws]
        for team, ws in zip(idle["team"].to_list(), idle["week"].to_list(), strict=True)
    }
    return {team: found.get(team, []) for team in sorted(set(weeks["team"].to_list()))}


def bye_weeks(schedule: pl.DataFrame, season: int) -> dict[str, int]:
    """team -> bye week, for one regular season.

    Derive byes from the schedule, never from `player.proTeamId`: ESPN's team is
    the player's *current* team, so a traded player inherits the wrong bye.

    Nearly every season gives each team exactly one idle week, but not all of
    them, so this takes the earliest and warns rather than assuming:

    * **2022** carries 271 regular-season games, not 272. The Week 17
      Bills-Bengals game was abandoned after Damar Hamlin's cardiac arrest and
      never replayed, and nflverse just drops the row -- so BUF and CIN each show
      two idle weeks (7 and 17; 10 and 17). The earlier one is the real bye.
    * **1999-2001** had 31 teams, so one club sat out every week and byes are
      spread across weeks 1-17. The familiar weeks 5-14 window is modern only.

    A team with *no* idle week means the frame is filtered or incomplete, and
    there is no honest answer, so that raises. Call `idle_weeks` when you need the
    ambiguity itself rather than a single number.
    """
    idle = idle_weeks(schedule, season)
    if empty := sorted(t for t, w in idle.items() if not w):
        raise NflverseError(
            f"season {season}: {len(empty)} team(s) play every week ({empty}), so they have "
            "no bye. The schedule frame is filtered or incomplete."
        )
    if ambiguous := {t: w for t, w in idle.items() if len(w) > 1}:
        log.warning(
            "season %d: %d team(s) have more than one idle week (%s); taking the earliest",
            season,
            len(ambiguous),
            ambiguous,
        )
    return {team: weeks[0] for team, weeks in idle.items()}


def remaining_opponents(
    schedule: pl.DataFrame,
    season: int,
    from_week: int,
    *,
    include_byes: bool = True,
) -> pl.DataFrame:
    """Rest-of-season opponents per team, from `from_week` inclusive.

    The input to strength-of-schedule work. `from_week` is inclusive so passing
    the current scoring period keeps the week in progress.
    """
    return team_weeks(schedule, season, from_week=from_week, include_byes=include_byes)


def games_remaining(schedule: pl.DataFrame, season: int, from_week: int) -> dict[str, int]:
    """team -> number of regular-season games left from `from_week` inclusive.

    Every team in the season appears, including any with nothing left; a missing
    key would read downstream as a missing team rather than as zero games.
    """
    # Count over the full grid rather than the filtered view, so a `from_week`
    # past the end of the season yields zeros instead of an empty dict.
    counts = (
        team_weeks(schedule, season)
        .group_by("team")
        .agg((~pl.col("is_bye") & (pl.col("week") >= from_week)).sum().alias("n"))
        .sort("team")
    )
    return dict(zip(counts["team"].to_list(), (int(n) for n in counts["n"]), strict=True))
