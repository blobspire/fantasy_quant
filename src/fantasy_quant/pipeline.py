"""End-to-end wiring: an ESPN league id in, championship probabilities out.

Every layer below this one is deliberately independent and testable in isolation,
which leaves the question of who composes them. This module is that answer, and
it is the only place that knows the whole chain:

    league id
      -> espn/league.py        settings, rosters, schedule, standings
      -> espn/scoring.py       this league's own scoring function
      -> core.LeagueContext    the (player, league) pair that makes a value meaningful
      -> corpus / ensemble     component projections per player-week
      -> calibration           component points -> a calibrated WeeklyOutlook
      -> sim/distributions     a correlated [sim, week, player] tensor
      -> sim/season            standings, bracket, championship probability

Two things it exists to guarantee, both of which are silent when wrong:

* **Projections are scored per league.** The same receiver is re-scored through
  each league's own `LeagueScoring`, so a full-PPR league and a half-PPR league
  genuinely disagree about him. Nothing here may compute a league-independent
  player value.
* **The tensor axes line up.** `panel_for` enforces that the panel covers exactly
  the pool's players and exactly the state's remaining weeks; get it wrong and the
  simulator scores the right numbers against the wrong players without complaining.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from . import corpus
from .core import FITTED_POSITIONS, Objective, PlayerOutlook, WeeklyOutlook, WireLevel
from .espn.client import EspnClient
from .espn.endpoints import league_url
from .espn.league import League
from .espn.scoring import LeagueScoring
from .projections.calibration import CalibrationSet
from .projections.calibration import load as load_calibration
from .sim import season as S
from .sim.distributions import Draw, WeeklySampler, espn_bye_weeks

log = logging.getLogger(__name__)

#: Positions we carry a fitted weekly distribution for -- now including K and D/ST,
#: which are genuinely startable (all three leagues start both) and used to be run
#: through the pooled skill line instead. See `projections/calibration`: for a defence
#: the pooled line has the slope the wrong way round, and this is where it reached it.
CALIBRATED_POSITIONS = FITTED_POSITIONS


class PipelineError(RuntimeError):
    """The chain could not be assembled for this league."""


def client_from_env(env_file: Path | str | None = None) -> EspnClient:
    """An authenticated client from ESPN_SWID / ESPN_S2.

    Falls back to an unauthenticated client, which is enough for public leagues
    and for every league-independent endpoint, so a missing cookie degrades to
    reduced access rather than a crash.
    """
    if env_file is not None or not os.environ.get("ESPN_SWID"):
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file or ".env")
        except ImportError:  # pragma: no cover - dotenv is a declared dependency
            pass
    swid, s2 = os.environ.get("ESPN_SWID"), os.environ.get("ESPN_S2")
    if not (swid and s2):
        log.warning("no ESPN_SWID/ESPN_S2 in the environment; private leagues will 401")
        return EspnClient()
    return EspnClient(swid=swid, espn_s2=s2)


def check_objective_supported(objective: Objective | str) -> None:
    """Refuse to run a league whose objective no decision surface implements.

    Every surface currently maximises championship probability. In a league that
    pays for total points-for -- common in high-stakes formats -- that is not
    merely suboptimal, it is backwards near the playoff cut, where the right move
    is to keep scoring rather than to protect a seed. The registry can express
    that objective and nothing downstream honours it yet, so this fails loudly
    rather than quietly optimising the wrong thing.
    """
    obj = Objective(objective)
    if obj is not Objective.CHAMPIONSHIP:
        raise PipelineError(
            f"league objective {obj.value!r} is not implemented: every decision surface "
            "maximises championship probability. In a points-for league that is wrong, "
            'not just approximate. Set objective = "championship" to proceed, or leave '
            "this league out until the surfaces read core.Objective."
        )


def scoring_for(league: League) -> LeagueScoring:
    """This league's own scoring function, parsed from its raw settings payload.

    Deliberately not ESPN's `appliedTotal`: that is scored under whichever canned
    settings the snapshot used, so reusing it would quietly give a half-PPR league
    full-PPR receptions.
    """
    payload, _ = league._client.get(
        league_url(league.season, league.league_id), params={"view": "mSettings"}
    )
    return LeagueScoring.from_settings(payload)


@dataclass(frozen=True, slots=True)
class WeeklyProjection:
    """One player's projected points in one week, already scored for a league."""

    player_id: int
    name: str
    position_id: int
    pro_team_id: int
    week: int
    points: float


def league_projections(
    league: League,
    *,
    weeks: Sequence[int] | None = None,
    variant: str = "ppr",
    root: Path | str = corpus.DEFAULT_ROOT,
) -> list[WeeklyProjection]:
    """Per-week projections for every player in the pool, scored for THIS league.

    Reads component stat lines from the corpus and applies the league's own
    scoring function, rather than trusting ESPN's `appliedTotal`, which is scored
    under whichever canned settings the snapshot used.
    """
    season = league.season
    rows = corpus.load_stat_rows(seasons=(season,), root=root, variant=variant)
    weekly = rows.filter(
        (pl.col("stat_source_id") == corpus.SOURCE_PROJECTED)
        & (pl.col("stat_split_type_id") == corpus.SPLIT_GAME)
        & (pl.col("scoring_period_id") > 0)
    )
    if weeks is not None:
        weekly = weekly.filter(pl.col("scoring_period_id").is_in(list(weeks)))
    if weekly.is_empty():
        raise PipelineError(
            f"no projected weekly rows for season {season} in {root}; run `fq snapshot`"
        )

    scorer = scoring_for(league).score
    out: list[WeeklyProjection] = []
    for r in weekly.iter_rows(named=True):
        stats = dict(zip(r["stat_ids"] or [], r["stat_values"] or [], strict=True))
        pos = r["default_position_id"]
        if pos is None:
            continue
        out.append(
            WeeklyProjection(
                player_id=r["espn_id"],
                name=r["full_name"] or "",
                position_id=pos,
                pro_team_id=r["pro_team_id"] or 0,
                week=r["scoring_period_id"],
                points=float(scorer(stats, pos)),
            )
        )
    return out


def calibrated_outlooks(
    projections: Sequence[WeeklyProjection],
    *,
    season: int,
    calibration: CalibrationSet | None = None,
    variant: str = "ppr",
) -> list[PlayerOutlook]:
    """Turn scored projections into calibrated hurdle-gamma outlooks.

    Every position is passed through with its own id. There used to be a `pos` local
    here meant to route K and D/ST to the pooled fit, and it never did anything -- it
    computed 0 for them and then `pos or p.position_id` handed back the real id anyway.
    The routing happened one layer down, in `CalibrationSet.for_position`, and both
    positions now have fitted curves of their own there.
    """
    cal = calibration or load_calibration(variant)
    by_player: dict[int, dict[int, WeeklyOutlook]] = {}
    meta: dict[int, WeeklyProjection] = {}
    for p in projections:
        outlook = cal.outlook(
            player_id=p.player_id,
            season=season,
            week=p.week,
            position_id=p.position_id,
            projection=max(p.points, 0.0),
            pro_team_id=p.pro_team_id,
        )
        by_player.setdefault(p.player_id, {})[p.week] = outlook
        meta.setdefault(p.player_id, p)
    return [
        PlayerOutlook(
            player_id=pid,
            name=meta[pid].name,
            position_id=meta[pid].position_id,
            pro_team_id=meta[pid].pro_team_id,
            weeks=weeks,
        )
        for pid, weeks in by_player.items()
    ]


def _zeroed(player_id: int, season: int, week: int, position_id: int) -> WeeklyOutlook:
    """A week this player will not play: no points, no variance, not startable."""
    return WeeklyOutlook(
        player_id=int(player_id), season=season, week=week, position_id=position_id,
        mean=0.0, sd=0.0, p_zero=1.0, shape=1.0, scale=1.0, pro_team_id=0, playing=False,
    )


def _fill_weeks(
    outlooks: Sequence[PlayerOutlook], weeks: Sequence[int], season: int
) -> list[PlayerOutlook]:
    """Ensure every outlook covers every remaining week, zeroing the gaps.

    `panel_for` refuses a partial outlook rather than silently drawing the gap as zero
    -- correctly, because a silent zero would make the whole team score nothing that
    week -- so the gaps have to be filled explicitly as "not playing".

    This used to claim ESPN emits no projection row on a bye and that filling the gaps
    was therefore how byes got handled. That is false, and believing it is how every
    defence came to play seventeen games. Measured on the 2026 pool: 32 defences x 18
    weeks is 576 rows and 575 are present -- the one absentee is New Orleans in week 8,
    which is exactly NO's bye, and that single accident is what the sentence was
    generalised from. The other 31 defences all carry a full projection on their own
    bye, averaging 5.00 against a 5.05 season mean. Skill players and kickers are saved
    only because ESPN happens to project them at exactly 0.00 on a bye.

    Byes are handled where they belong, by passing a bye table to `panel_for`, which
    sets `has_game=False`. See `build`.
    """
    want = set(weeks)
    out: list[PlayerOutlook] = []
    for o in outlooks:
        gaps = want - set(o.weeks)
        if not gaps:
            out.append(o)
            continue
        filled = dict(o.weeks)
        for w in gaps:
            filled[w] = _zeroed(o.player_id, season, w, o.position_id)
        out.append(
            PlayerOutlook(
                player_id=o.player_id, name=o.name, position_id=o.position_id,
                pro_team_id=o.pro_team_id, weeks=filled,
            )
        )
    return out


def _fill_unprojected(
    outlooks: Sequence[PlayerOutlook], state: S.LeagueState, season: int
) -> list[PlayerOutlook]:
    """Give every rostered player an outlook, zeroed if ESPN projects him nothing.

    Unsigned free agents get no projection rows at all -- a benched Kareem Hunt on
    proTeamId 0 is the live example. Dropping them would misalign the tensor
    columns against the pool, so they are compiled as playing no games, which is
    what "no team" actually means.
    """
    have = {o.player_id for o in outlooks}
    missing = [pid for pid in state.pool.player_ids if pid not in have]
    weeks = tuple(state.weeks)
    outlooks = _fill_weeks(outlooks, weeks, season)
    if not missing:
        return list(outlooks)
    positions = state.pool.positions_of(missing)
    filled = list(outlooks)
    for pid, pos in zip(missing, positions, strict=True):
        pos = int(pos)
        filled.append(
            PlayerOutlook(
                player_id=int(pid),
                name=state.pool.name(pid),
                position_id=pos,
                pro_team_id=0,
                weeks={
                    w: WeeklyOutlook(
                        player_id=int(pid),
                        season=season,
                        week=w,
                        position_id=pos,
                        mean=0.0,
                        sd=0.0,
                        p_zero=1.0,
                        shape=1.0,
                        scale=1.0,
                        pro_team_id=0,
                        playing=False,
                    )
                    for w in weeks
                },
            )
        )
    log.info("compiled %d rostered players with no projection as zeroed", len(missing))
    return filled


@dataclass(frozen=True, slots=True)
class LeagueSim:
    """Everything needed to evaluate candidate moves in one league.

    Holds the drawn tensor so that comparing rosters uses common random numbers:
    two candidates meet identical football, and the paired difference isolates the
    move instead of re-rolling the season.
    """

    league: League
    state: S.LeagueState
    draw: Draw
    outlooks: Sequence[PlayerOutlook]
    n_sims: int
    seed: int
    #: Cache for `floors()`. Not part of the value -- two sims with the same draw are the
    #: same sim whether or not either has been asked for its floors yet.
    _floors: Mapping[int, WireLevel] | None = field(
        default=None, compare=False, repr=False
    )

    def floors(self) -> Mapping[int, WireLevel]:
        """What an unfilled starting slot streams off this league's wire, per slot id.

        `decide/title.streaming_levels` over `self.outlooks`, which is the same call
        every recommendation surface in this codebase makes. Cached, because it is a
        sort over the whole projection corpus and nothing about it changes between calls.
        """
        if self._floors is None:
            from .decide.title import streaming_levels

            object.__setattr__(
                self,
                "_floors",
                streaming_levels(self.state, self.draw, outlooks=self.outlooks),
            )
        assert self._floors is not None
        return self._floors

    def simulate(self, **kw) -> S.SeasonResult:
        """This league's season, with an unfilled slot floored at the wire.

        **The floor is the default now, and it did not used to be.** `championship_table`
        scored an unfilled slot at ZERO while every recommendation surface floored it at
        the wire, so one league had two published baselines -- a gap disclosed only as a
        footer string on `fq odds` and impossible to reconcile from the output. An empty
        seat does not score nothing; the wire always has a defence.

        It is not a level shift that cancels. Championship probabilities sum to one, so
        this is a relative game, and the teams it helps are the ones carrying the most
        empty seats. Measured on the user's three leagues at week 1 of 2026, **7 to 11 of
        the 12-14 teams change rank**, the biggest single move is -3.53pp and the biggest
        rise +3.25pp, and the user's own odds go UP in all three (+1.57, +1.45, +0.77pp).

        Pass `replacement=0.0` for the old empty-seat convention.
        """
        kw.setdefault("all_play", False)
        if "replacement" not in kw:
            kw["replacement"] = self.floors()
        kw.setdefault(
            "noise",
            S.FloorNoise(self.state, self.draw) if S.has_spread(kw["replacement"]) else None,
        )
        return S.simulate(self.state, self.draw, **kw)


def _byes(season: int) -> Mapping[int, int] | None:
    """proTeamId -> bye week, or None if the table cannot be read.

    A missing bye table is a warning rather than a failure: `data/reference/
    platform_settings_{season}.json` is normally on disk and needs no network, but a
    fresh checkout that is offline should still get a league it can reason about,
    slightly wrong about defences, rather than no league at all.
    """
    try:
        table = espn_bye_weeks(season)
    except Exception as exc:  # noqa: BLE001 - any read failure degrades the same way
        log.warning("no bye table for %d (%s); defences will play every week", season, exc)
        return None
    if not table:
        log.warning("empty bye table for %d; defences will play every week", season)
        return None
    return table


def build(
    league_id: int,
    season: int,
    *,
    my_team_id: int | None = None,
    client: EspnClient | None = None,
    n_sims: int = 4000,
    seed: int = 1,
    variant: str = "ppr",
    root: Path | str = corpus.DEFAULT_ROOT,
    calibration: CalibrationSet | None = None,
    objective: Objective | str = Objective.CHAMPIONSHIP,
    byes: Mapping[int, int] | None = None,
) -> LeagueSim:
    """Assemble one league end to end, ready to simulate.

    `byes` maps proTeamId -> bye week and is resolved from ESPN's own platform-settings
    table when not given. It is not optional in spirit: without it `SimPanel.has_game`
    is True in all seventeen weeks and every defence plays a full season, because ESPN
    projects 31 of 32 defences normally on their own bye. See `_fill_weeks` for the
    false premise that hid this, and `edges/portfolio.ByeExposure` for the audit that
    kept reporting it.
    """
    check_objective_supported(objective)
    own_client = client is None
    client = client or client_from_env()
    try:
        league = League(client, league_id, season)
        state = S.state_from_league(league, my_team_id=my_team_id)
        projections = league_projections(league, variant=variant, root=root)
        outlooks = calibrated_outlooks(
            projections, season=season, calibration=calibration, variant=variant
        )
        outlooks = _fill_unprojected(outlooks, state, season)
        panel = S.panel_for(state, outlooks, byes=byes if byes is not None else _byes(season))
        draw = WeeklySampler(panel, seed=seed).draw(n_sims)
    finally:
        if own_client:
            client.close()
    return LeagueSim(
        league=league,
        state=state,
        draw=draw,
        outlooks=outlooks,
        n_sims=n_sims,
        seed=seed,
    )


def championship_table(sim: LeagueSim) -> list[dict[str, object]]:
    """Every team's odds, best first. Championship probabilities sum to 1."""
    result = sim.simulate()
    rows = []
    for f in sim.state.franchises:
        t = result.by_team(f.team_id)
        rows.append(
            {
                "team_id": f.team_id,
                "name": f.name,
                "is_me": f.team_id == sim.state.my_team_id,
                "playoffs": t.make_playoffs,
                "bye": t.bye,
                "championship": t.championship,
                "expected_wins": t.expected_wins,
            }
        )
    rows.sort(key=lambda r: -float(r["championship"]))
    total = sum(float(r["championship"]) for r in rows)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise PipelineError(f"championship probabilities sum to {total}, not 1.0")
    return rows


#: Re-exported so callers (and tests) can build a League without a second import.
__all__ = [
    "League",
    "LeagueSim",
    "PipelineError",
    "WeeklyProjection",
    "build",
    "calibrated_outlooks",
    "championship_table",
    "check_objective_supported",
    "client_from_env",
    "league_projections",
    "scoring_for",
]
