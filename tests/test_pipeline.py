"""End-to-end wiring: league id in, championship probabilities out."""

from __future__ import annotations

import numpy as np
import pytest

from fantasy_quant import pipeline as P
from fantasy_quant.core import PlayerOutlook, WeeklyOutlook

LEAGUES = [
    (272150391, "Wine Wednesday", 1, 14),
    (161496047, "Blacksburg Baddies", 1, 12),
    (634537479, "Type shi season 2 actually", 2, 12),
]


def _outlook(pid: int, week: int, mean: float, pos: int = 3) -> WeeklyOutlook:
    return WeeklyOutlook(
        player_id=pid,
        season=2026,
        week=week,
        position_id=pos,
        mean=mean,
        sd=max(mean * 0.5, 1.0),
        p_zero=0.2,
        shape=2.0,
        scale=max(mean / 2, 0.5),
    )


class TestCalibratedOutlooks:
    def test_projections_become_calibrated_outlooks(self):
        projections = [
            P.WeeklyProjection(
                player_id=1, name="A", position_id=3, pro_team_id=8, week=w, points=12.0
            )
            for w in range(1, 4)
        ]
        got = P.calibrated_outlooks(projections, season=2026)
        assert len(got) == 1
        assert set(got[0].weeks) == {1, 2, 3}
        assert all(0.0 <= o.p_zero <= 1.0 for o in got[0].weeks.values())

    def test_kickers_and_defenses_are_kept_not_dropped(self):
        """All three leagues start both; dropping them loses ~13-14 points a week."""
        projections = [
            P.WeeklyProjection(
                player_id=-16024,
                name="Chargers D/ST",
                position_id=16,
                pro_team_id=24,
                week=1,
                points=7.0,
            ),
            P.WeeklyProjection(
                player_id=999, name="A Kicker", position_id=5, pro_team_id=24, week=1, points=8.0
            ),
        ]
        got = {o.player_id for o in P.calibrated_outlooks(projections, season=2026)}
        assert got == {-16024, 999}

    def test_a_negative_projection_does_not_become_a_negative_mean(self):
        projections = [
            P.WeeklyProjection(
                player_id=1, name="A", position_id=16, pro_team_id=8, week=1, points=-3.0
            )
        ]
        outlook = P.calibrated_outlooks(projections, season=2026)[0].weeks[1]
        assert outlook.mean >= 0.0


class TestFillUnprojected:
    """An unsigned free agent has no projection rows, and dropping him would
    misalign the tensor columns against the roster pool."""

    def _state(self, player_ids):
        from fantasy_quant.sim import season as S

        rows = [(pid, 2, 0, f"P{pid}") for pid in player_ids]
        pool = S.PlayerPool.of(rows)
        return type("S", (), {"pool": pool, "weeks": (1, 2)})()

    def test_a_rostered_player_with_no_projection_is_zeroed_not_dropped(self):
        state = self._state([10, 20])
        given = [
            PlayerOutlook(
                player_id=10,
                name="A",
                position_id=2,
                pro_team_id=8,
                weeks={1: _outlook(10, 1, 9.0), 2: _outlook(10, 2, 9.0)},
            )
        ]
        filled = P._fill_unprojected(given, state, 2026)
        by_id = {o.player_id: o for o in filled}
        assert set(by_id) == {10, 20}
        assert all(w.mean == 0.0 and not w.playing for w in by_id[20].weeks.values())

    def test_nothing_is_added_when_every_player_is_projected(self):
        state = self._state([10])
        given = [
            PlayerOutlook(
                player_id=10,
                name="A",
                position_id=2,
                pro_team_id=8,
                weeks={1: _outlook(10, 1, 9.0), 2: _outlook(10, 2, 9.0)},
            )
        ]
        assert len(P._fill_unprojected(given, state, 2026)) == 1


@pytest.mark.network
class TestAgainstTheRealLeagues:
    """The end-to-end acceptance test. These are the user's actual leagues."""

    @pytest.fixture(scope="class")
    def client(self):
        c = P.client_from_env()
        yield c
        c.close()

    @pytest.mark.parametrize("league_id,name,my_team,size", LEAGUES)
    def test_a_league_simulates_end_to_end(self, client, league_id, name, my_team, size):
        sim = P.build(league_id, 2026, my_team_id=my_team, client=client, n_sims=500)
        table = P.championship_table(sim)
        assert len(table) == size
        # championship_table raises if these do not sum to 1, so this pins the count.
        assert sum(r["championship"] for r in table) == pytest.approx(1.0, abs=1e-6)
        assert any(r["is_me"] for r in table)
        assert all(0.0 <= r["playoffs"] <= 1.0 for r in table)

    def test_the_same_seed_reproduces_exactly(self, client):
        """Common random numbers: two candidate rosters must meet identical football."""
        a = P.build(161496047, 2026, my_team_id=1, client=client, n_sims=300, seed=5)
        b = P.build(161496047, 2026, my_team_id=1, client=client, n_sims=300, seed=5)
        pa = getattr(a.draw, "points", a.draw)
        pb = getattr(b.draw, "points", b.draw)
        assert np.array_equal(pa, pb)

    def test_full_ppr_and_half_ppr_disagree_about_receivers(self, client):
        """The same player, scored through two different leagues' own settings."""
        full = P.scoring_for(P.League(client, 272150391, 2026))
        half = P.scoring_for(P.League(client, 161496047, 2026))
        line = {"53": 6.0, "42": 80.0}  # 6 catches, 80 yards
        assert full.score(line, 3) - half.score(line, 3) == pytest.approx(3.0)


class TestUnsupportedObjectivesFailLoudly:
    """No decision surface reads core.Objective yet; all four maximise title odds.

    In a points-for league that is backwards near the playoff cut, so the entry
    point refuses rather than quietly optimising the wrong quantity.
    """

    def test_championship_is_accepted(self):
        from fantasy_quant.core import Objective

        P.check_objective_supported(Objective.CHAMPIONSHIP)
        P.check_objective_supported("championship")

    @pytest.mark.parametrize("objective", ["points", "hybrid"])
    def test_other_objectives_are_refused_with_a_reason(self, objective):
        with pytest.raises(P.PipelineError, match="not implemented"):
            P.check_objective_supported(objective)

    def test_the_error_says_what_to_do(self):
        with pytest.raises(P.PipelineError, match="championship"):
            P.check_objective_supported("points")
