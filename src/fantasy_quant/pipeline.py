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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import corpus
from .core import Objective, PlayerOutlook, WeeklyOutlook
from .espn.client import EspnClient
from .espn.endpoints import league_url
from .espn.league import League
from .espn.scoring import LeagueScoring
from .projections.calibration import CalibrationSet
from .projections.calibration import load as load_calibration
from .sim import season as S
from .sim.distributions import Draw, WeeklySampler

log = logging.getLogger(__name__)

#: Positions we carry a fitted weekly distribution for. K and D/ST are simulated
#: from their projections with the pooled calibration -- they are genuinely
#: startable (all three leagues start both) and dropping them would understate a
#: team's weekly score by roughly 13-14 points.
CALIBRATED_POSITIONS = (1, 2, 3, 4)


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

    Positions without a fitted curve (K, D/ST) fall back to the pooled fit rather
    than being dropped, because both are started every week in a normal league.
    """
    cal = calibration or load_calibration(variant)
    by_player: dict[int, dict[int, WeeklyOutlook]] = {}
    meta: dict[int, WeeklyProjection] = {}
    for p in projections:
        pos = p.position_id if p.position_id in CALIBRATED_POSITIONS else 0
        outlook = cal.outlook(
            player_id=p.player_id,
            season=season,
            week=p.week,
            position_id=pos or p.position_id,
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

    ESPN emits no projection row for a bye, so a defense's outlook covers 17 of 18
    weeks. `panel_for` refuses a partial outlook rather than silently drawing the
    gap as zero -- correctly, because a silent zero would make the whole team score
    nothing that week -- so the gaps have to be filled explicitly as "not playing".
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

    def simulate(self, **kw) -> S.SeasonResult:
        kw.setdefault("all_play", False)
        return S.simulate(self.state, self.draw, **kw)


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
) -> LeagueSim:
    """Assemble one league end to end, ready to simulate."""
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
        panel = S.panel_for(state, outlooks)
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
