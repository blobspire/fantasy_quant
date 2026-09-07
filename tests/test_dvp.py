"""Tests for the opponent-adjusted DvP model.

The load-bearing test is `test_recovers_planted_defense_effects`: synthetic
seasons are generated where a named team genuinely suppresses WRs by four points,
and the solver has to find it. Everything else -- convergence, shrinkage
behaviour, the schedule aggregation -- is a property of the estimator that can be
checked without a network. The two live tests are marked `network` and are the
only ones that touch nflverse or ESPN.
"""

from __future__ import annotations

import math
import os

import numpy as np
import polars as pl
import pytest

from fantasy_quant.analysis import dvp

# --------------------------------------------------------------------------------------
# Synthetic seasons with known truth
# --------------------------------------------------------------------------------------

TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]  # fmt: skip


def synthetic_season(
    *,
    seed: int = 11,
    weeks: int = 17,
    per_team: int = 5,
    position: str = "WR",
    season: int = 2025,
    mu: float = 7.0,
    offense_sd: float = 4.0,
    defense_sd: float = 1.5,
    noise_sd: float = 3.0,
    planted: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, dict[str, float], dict[str, float]]:
    """A season where offense and defense effects are known by construction.

    Returns the player-week frame plus the true player and defense effects. The
    schedule is a fresh random pairing each week, which is *kinder* than a real
    NFL schedule (real ones are correlated within division), so a solver that
    cannot recover effects here has no chance on real data.
    """
    rng = np.random.default_rng(seed)
    players = [f"{team}-{i}" for team in TEAMS for i in range(per_team)]
    true_offense = {
        p: float(v) for p, v in zip(players, rng.normal(0, offense_sd, len(players)), strict=True)
    }
    true_defense = {
        t: float(v) for t, v in zip(TEAMS, rng.normal(0, defense_sd, len(TEAMS)), strict=True)
    }
    for team, value in (planted or {}).items():
        true_defense[team] = float(value)

    rows: list[dict[str, object]] = []
    for week in range(1, weeks + 1):
        order = list(rng.permutation(TEAMS))
        for a in range(0, len(order), 2):
            for offense, defense in ((order[a], order[a + 1]), (order[a + 1], order[a])):
                for i in range(per_team):
                    player = f"{offense}-{i}"
                    rows.append(
                        {
                            "player_id": player,
                            "player_display_name": player,
                            "position": position,
                            "season": season,
                            "week": week,
                            "team": offense,
                            "opponent_team": defense,
                            dvp.PPR: mu
                            + true_offense[player]
                            + true_defense[defense]
                            + float(rng.normal(0, noise_sd)),
                        }
                    )
    return pl.DataFrame(rows), true_offense, true_defense


def _centered(values: dict[str, float]) -> dict[str, float]:
    """The decomposition is identified up to a constant sliding into `mu`."""
    mean = float(np.mean(list(values.values())))
    return {k: v - mean for k, v in values.items()}


# --------------------------------------------------------------------------------------
# Recovery -- the real test
# --------------------------------------------------------------------------------------


def test_recovers_planted_defense_effects() -> None:
    """A team that truly suppresses WRs by 4 points is found, and ranked toughest."""
    frame, true_offense, true_defense = synthetic_season(planted={"SEA": -4.0, "CAR": +3.0})
    fit = dvp.fit_position(
        frame, "WR", shrinkage=dvp.Shrinkage(k_off=1.0, k_def=1.0, halflife=None)
    )

    est = _centered({t: e.effect for t, e in fit.defense.items()})
    truth = _centered(true_defense)

    assert fit.converged
    assert est["SEA"] == pytest.approx(truth["SEA"], abs=0.5)
    assert est["CAR"] == pytest.approx(truth["CAR"], abs=0.5)
    rmse = math.sqrt(np.mean([(est[t] - truth[t]) ** 2 for t in TEAMS]))
    assert rmse < 0.5, f"defense RMSE {rmse:.3f}"
    assert np.corrcoef([est[t] for t in TEAMS], [truth[t] for t in TEAMS])[0, 1] > 0.95

    # SEA is the planted floor, so it must come out rank 1 (toughest).
    assert fit.defense["SEA"].rank == 1
    assert fit.defense["CAR"].rank == len(TEAMS)

    off_est = _centered(dict(fit.offense))
    off_true = _centered(true_offense)
    keys = sorted(off_true)
    assert np.corrcoef([off_est[k] for k in keys], [off_true[k] for k in keys])[0, 1] > 0.95


def test_adjustment_removes_schedule_bias_that_fools_the_naive_table() -> None:
    """Bias 1 of 3: opponent quality. All 32 defenses are identical here.

    Half the league draws 13-point offenses three quarters of the time and the
    other half draws 3-point offenses three quarters of the time. Every true
    defense effect is zero, so any gap a table shows between the two halves is
    pure schedule. The naive table shows a large one; the adjustment must not.
    """
    rng = np.random.default_rng(5)
    strong = set(TEAMS[:16])
    rows = []
    for week in range(1, 18):
        for defense in TEAMS:
            # 75% of the time a defense draws from its "own" half of the league.
            draw_strong = (defense in strong) == (rng.random() < 0.75)
            pool = [t for t in TEAMS if (t in strong) == draw_strong and t != defense]
            offense = pool[int(rng.integers(len(pool)))]
            base = 13.0 if offense in strong else 3.0
            for i in range(5):
                rows.append(
                    {
                        "player_id": f"{offense}-{i}",
                        "player_display_name": f"{offense}-{i}",
                        "position": "WR",
                        "season": 2025,
                        "week": week,
                        "team": offense,
                        "opponent_team": defense,
                        dvp.PPR: base + float(rng.normal(0, 3.0)),
                    }
                )
    frame = pl.DataFrame(rows)
    fit = dvp.fit_position(
        frame, "WR", shrinkage=dvp.Shrinkage(k_off=1.0, k_def=1.0, halflife=None)
    )
    assert fit.converged

    def group_gap(pick) -> float:
        hard = [pick(fit.defense[t]) for t in TEAMS if t in strong]
        soft = [pick(fit.defense[t]) for t in TEAMS if t not in strong]
        return float(np.mean(hard) - np.mean(soft))

    naive_gap = group_gap(lambda e: e.naive)
    adjusted_gap = group_gap(lambda e: e.effect)
    assert naive_gap > 3.0, f"the naive table should be badly fooled, got {naive_gap:+.2f}"
    assert abs(adjusted_gap) < naive_gap / 4.0, (
        f"adjusted gap {adjusted_gap:+.2f} vs naive {naive_gap:+.2f}"
    )
    # And the whole spread should collapse, not just the group means.
    naive_sd = float(np.std([e.naive for e in fit.defense.values()]))
    assert fit.spread() < naive_sd / 3.0


# --------------------------------------------------------------------------------------
# Shrinkage behaviour
# --------------------------------------------------------------------------------------


def _one_defense_frame(counts: dict[str, int], deviation: float) -> pl.DataFrame:
    """Each named defense sees `counts[team]` identical +deviation player-games."""
    rows = []
    week = 1
    for team, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "player_id": f"{team}-{i}",
                    "player_display_name": f"{team}-{i}",
                    "position": "WR",
                    "season": 2025,
                    "week": week + i,
                    "team": "OPP",
                    "opponent_team": team,
                    dvp.PPR: 10.0 + deviation,
                }
            )
    # Ballast so the grand mean is 10.0 and the deviation above is a real deviation.
    for i in range(400):
        rows.append(
            {
                "player_id": f"BAL-{i}",
                "player_display_name": f"BAL-{i}",
                "position": "WR",
                "season": 2025,
                "week": 1 + (i % 17),
                "team": "OPP",
                "opponent_team": "BAL",
                dvp.PPR: 10.0,
            }
        )
    return pl.DataFrame(rows)


def test_shrinkage_pulls_a_one_game_sample_far_harder_than_sixteen() -> None:
    """The same raw deviation moves a 1-observation defense far less than a 16."""
    frame = _one_defense_frame({"NYJ": 1, "NYG": 16}, deviation=6.0)
    fit = dvp.fit_position(
        frame, "WR", shrinkage=dvp.Shrinkage(k_off=1e9, k_def=8.0, halflife=None)
    )
    small = fit.defense["NYJ"].effect
    large = fit.defense["NYG"].effect

    # Both saw the identical +6 deviation, so any difference is pure shrinkage.
    assert fit.defense["NYJ"].naive == pytest.approx(fit.defense["NYG"].naive, abs=1e-6)
    assert 0 < small < large
    assert large > 3.0 * small, f"1-game {small:.3f} vs 16-game {large:.3f}"
    # Analytic check: with the offense term pinned out, the estimate is the
    # shrunk mean n*dev/(n+k) up to the intercept shift the ballast absorbs.
    # Tight, because the identity is exact -- a loose tolerance here would pass
    # for any k in a wide band and stop testing the shrinkage at all.
    assert small / large == pytest.approx((1 / (1 + 8)) / (16 / (16 + 8)), rel=1e-6)


def test_more_shrinkage_moves_effects_toward_zero() -> None:
    frame, _, _ = synthetic_season(seed=5)
    spreads = []
    for k_def in (1.0, 30.0, 300.0, 3000.0):
        fit = dvp.fit_position(
            frame, "WR", shrinkage=dvp.Shrinkage(k_off=1.0, k_def=k_def, halflife=None)
        )
        spreads.append(fit.spread())
    assert spreads == sorted(spreads, reverse=True)
    assert spreads[-1] < 0.05, "at k_def=3000 the model should have almost no opinion"


def test_shrinkage_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        dvp.Shrinkage(k_off=-1.0, k_def=30.0)
    with pytest.raises(ValueError):
        dvp.Shrinkage(k_off=1.0, k_def=30.0, halflife=0.0)


# --------------------------------------------------------------------------------------
# Solver mechanics
# --------------------------------------------------------------------------------------


def _index_arrays(
    frame: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    players = sorted(set(frame["player_id"].to_list()))
    teams = sorted(set(frame["opponent_team"].to_list()))
    pi = {p: i for i, p in enumerate(players)}
    ti = {t: i for i, t in enumerate(teams)}
    off = np.array([pi[p] for p in frame["player_id"]], dtype=np.intp)
    dfn = np.array([ti[t] for t in frame["opponent_team"]], dtype=np.intp)
    y = frame[dvp.PPR].to_numpy().astype(float)
    return off, dfn, y, np.ones_like(y), len(players), len(teams)


def test_objective_decreases_monotonically_and_converges() -> None:
    frame, _, _ = synthetic_season(seed=17)
    off, dfn, y, w, n_off, n_def = _index_arrays(frame)
    solution = dvp.solve_two_way(
        off, dfn, y, w, n_off, n_def, dvp.Shrinkage(2.0, 40.0), max_iter=500, tol=1e-10
    )
    assert solution.converged
    obj = solution.objective
    assert len(obj) == solution.iterations
    for earlier, later in zip(obj, obj[1:], strict=False):
        assert later <= earlier + 1e-9, "penalised objective must not increase"


def test_extra_iterations_do_not_move_the_answer() -> None:
    frame, _, _ = synthetic_season(seed=23)
    off, dfn, y, w, n_off, n_def = _index_arrays(frame)
    shrink = dvp.Shrinkage(2.0, 40.0)
    a = dvp.solve_two_way(off, dfn, y, w, n_off, n_def, shrink, max_iter=500, tol=1e-12)
    b = dvp.solve_two_way(off, dfn, y, w, n_off, n_def, shrink, max_iter=5000, tol=1e-12)
    assert np.allclose(a.defense, b.defense, atol=1e-9)
    assert np.allclose(a.offense, b.offense, atol=1e-9)
    assert a.mu == pytest.approx(b.mu, abs=1e-9)


def test_solver_reports_when_it_ran_out_of_iterations() -> None:
    frame, _, _ = synthetic_season(seed=29)
    off, dfn, y, w, n_off, n_def = _index_arrays(frame)
    solution = dvp.solve_two_way(
        off, dfn, y, w, n_off, n_def, dvp.Shrinkage(1.0, 1.0), max_iter=2, tol=1e-12
    )
    assert not solution.converged
    assert solution.iterations == 2


def test_solver_matches_a_direct_ridge_solve() -> None:
    """The alternating fit must equal the closed-form ridge solution it approximates."""
    frame, _, _ = synthetic_season(seed=31, weeks=4, per_team=2)
    off, dfn, y, w, n_off, n_def = _index_arrays(frame)
    shrink = dvp.Shrinkage(2.0, 5.0)
    solution = dvp.solve_two_way(off, dfn, y, w, n_off, n_def, shrink, max_iter=20000, tol=1e-14)

    # Dense design: [intercept | offense dummies | defense dummies], ridge on all
    # but the intercept.
    n = y.size
    design = np.zeros((n, 1 + n_off + n_def))
    design[:, 0] = 1.0
    design[np.arange(n), 1 + off] = 1.0
    design[np.arange(n), 1 + n_off + dfn] = 1.0
    penalty = np.diag(
        np.concatenate([[0.0], np.full(n_off, shrink.k_off), np.full(n_def, shrink.k_def)])
    )
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    assert beta[0] == pytest.approx(solution.mu, abs=1e-6)
    assert np.allclose(beta[1 : 1 + n_off], solution.offense, atol=1e-6)
    assert np.allclose(beta[1 + n_off :], solution.defense, atol=1e-6)


def test_recency_weights() -> None:
    weeks = np.array([1.0, 5.0, 9.0])
    assert np.allclose(dvp.recency_weights(weeks, 9, None), 1.0)
    w = dvp.recency_weights(weeks, 9, 4.0)
    assert w[2] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)
    assert w[0] == pytest.approx(0.25)
    # A future week never gets more than full weight.
    assert dvp.recency_weights(np.array([20.0]), 9, 4.0)[0] == pytest.approx(1.0)


def test_recency_anchor_matches_cross_validation() -> None:
    """`fit_position` must weight the same way `cross_validate` does.

    Load-bearing, because `DEFAULT_SHRINKAGE` was chosen by CV and `k_off`/`k_def`
    are compared against the *sum of weights*. Anchoring the fit on the last week
    observed instead of the week after it would scale every weight by
    2**(1/halflife) relative to the CV that picked the constants -- 12% at
    halflife 6 -- so the shipped constants would shrink harder in production than
    they did in the search, with nothing to show for it.
    """
    frame, _, _ = synthetic_season(seed=113, weeks=9, per_team=2)
    halflife = 6.0
    fit = dvp.fit_position(
        frame, "WR", shrinkage=dvp.Shrinkage(k_off=1.0, k_def=5.0, halflife=halflife)
    )
    # Rebuild the weights a fit through week 9 must have used, and compare their
    # total against the weighted observation counts the solver reported.
    weeks = frame.filter(pl.col("position") == "WR")["week"].to_numpy().astype(float)
    expected_total = float(dvp.recency_weights(weeks, 9 + 1, halflife).sum())
    actual_total = sum(e.weighted_observations for e in fit.defense.values())
    assert actual_total == pytest.approx(expected_total, rel=1e-9)
    # And explicitly: the newest game is NOT at weight 1.0 -- it is one half-life
    # step down, which is what makes the CV constants transfer.
    newest = dvp.recency_weights(np.array([9.0]), 10, halflife)[0]
    assert newest == pytest.approx(0.5 ** (1.0 / halflife))
    assert newest < 1.0


def test_recency_weighting_tracks_a_midseason_change() -> None:
    """A defense that collapses at week 9 should read soft under a short half-life."""
    rows = []
    rng = np.random.default_rng(2)
    for week in range(1, 18):
        shift = -4.0 if week <= 8 else +4.0
        for team in TEAMS:
            for i in range(4):
                effect = shift if team == "DEN" else 0.0
                rows.append(
                    {
                        "player_id": f"P{i}-{week % 3}",
                        "player_display_name": f"P{i}",
                        "position": "WR",
                        "season": 2025,
                        "week": week,
                        "team": "OPP",
                        "opponent_team": team,
                        dvp.PPR: 10.0 + effect + float(rng.normal(0, 1.0)),
                    }
                )
    frame = pl.DataFrame(rows)
    flat = dvp.fit_position(
        frame, "WR", shrinkage=dvp.Shrinkage(k_off=1.0, k_def=5.0, halflife=None)
    )
    recent = dvp.fit_position(
        frame, "WR", shrinkage=dvp.Shrinkage(k_off=1.0, k_def=5.0, halflife=3.0)
    )
    assert abs(flat.effect("DEN")) < 1.0, "uniform weighting averages the two halves away"
    assert recent.effect("DEN") > 2.0, "a 3-game half-life should see the collapse"


# --------------------------------------------------------------------------------------
# Model surface
# --------------------------------------------------------------------------------------


def test_fit_refuses_multiple_seasons() -> None:
    a, _, _ = synthetic_season(seed=1, weeks=3, season=2024)
    b, _, _ = synthetic_season(seed=2, weeks=3, season=2025)
    with pytest.raises(dvp.DvpError, match="within-season"):
        dvp.fit_position(pl.concat([a, b]), "WR")


def test_fit_all_positions_and_lookups() -> None:
    frames = [synthetic_season(seed=i, weeks=8, position=p)[0] for i, p in enumerate(dvp.POSITIONS)]
    model = dvp.fit(pl.concat(frames))
    assert set(model.positions) == set(dvp.POSITIONS)
    assert model.season == 2025
    assert model["WR"].position == "WR"
    with pytest.raises(KeyError):
        model["K"]
    assert model.get("K") is None
    # Unknown teams are "no opinion", not an exception.
    assert model.effect("WR", "XXX") == 0.0
    assert model.effect("K", "SEA") == 0.0
    assert model["WR"].rank("XXX") is None


def test_adjust_is_additive_and_floors_at_zero() -> None:
    frame, _, _ = synthetic_season(seed=41, planted={"SEA": -4.0})
    fit = dvp.fit_position(frame, "WR", shrinkage=dvp.Shrinkage(1.0, 1.0, None))
    assert fit.adjust(12.0, "SEA") == pytest.approx(12.0 + fit.effect("SEA"))
    assert fit.adjust(0.5, "SEA") == 0.0
    assert fit.adjust(12.0, "XXX") == 12.0


def test_team_effect_scales_by_players_per_game() -> None:
    frame, _, _ = synthetic_season(seed=43, per_team=5)
    fit = dvp.fit_position(frame, "WR", shrinkage=dvp.Shrinkage(1.0, 1.0, None))
    assert fit.players_per_game == pytest.approx(5.0)
    assert fit.team_effect("SEA") == pytest.approx(fit.effect("SEA") * 5.0)


def test_naive_team_totals_reconstruct_the_mean() -> None:
    frame, _, _ = synthetic_season(seed=47, per_team=5)
    data = frame.filter(pl.col("position") == "WR")
    totals = dvp.naive_team_totals(data, dvp.PPR)
    assert len(totals) == len(TEAMS)
    # Mean team-total = per-player mean * players per team-game.
    assert float(np.mean(list(totals.values()))) == pytest.approx(
        float(data[dvp.PPR].mean()) * 5.0, rel=1e-6
    )


def test_naive_team_totals_do_not_merge_seasons() -> None:
    """A per-game *sum* does not survive pooling the way a mean does.

    Keying the game on `week` alone folds 2024 week 5 into 2025 week 5 and returns
    about twice the right number, which reads as a plausible team total. The
    result must instead be the pooled mean of the two seasons' team-games.
    """
    a, _, _ = synthetic_season(
        seed=201, weeks=6, season=2024, mu=10.0, noise_sd=0.0, offense_sd=0.0, defense_sd=0.0
    )
    b, _, _ = synthetic_season(
        seed=202, weeks=6, season=2025, mu=20.0, noise_sd=0.0, offense_sd=0.0, defense_sd=0.0
    )
    both = pl.concat([a, b])
    per_season = (
        dvp.naive_team_totals(a, dvp.PPR)["SEA"],
        dvp.naive_team_totals(b, dvp.PPR)["SEA"],
    )
    assert per_season == pytest.approx((50.0, 100.0))  # 5 players x mu
    pooled = dvp.naive_team_totals(both, dvp.PPR)["SEA"]
    assert pooled == pytest.approx(75.0), f"expected the pooled mean, got {pooled}"
    assert pooled < 100.0, "a season-blind game key would return ~150 here"


def test_compare_naive_reports_ranks() -> None:
    frame, _, _ = synthetic_season(seed=53, planted={"SEA": -4.0})
    fit = dvp.fit_position(frame, "WR")
    comparison = dvp.compare_naive(fit)
    assert comparison.teams == len(TEAMS)
    # Recompute the summary from the fit rather than from the comparison's own
    # `moves`, so a bug shared by both would still show.
    deltas = [abs(e.naive_rank - e.rank) for e in fit.defense.values()]
    assert comparison.mean_abs_move == pytest.approx(sum(deltas) / len(deltas))
    assert comparison.max_abs_move == max(deltas)
    assert len(comparison.moves) == len(TEAMS)
    # `moves` is documented biggest-mover-first; downstream reads top_movers().
    assert [abs(n - a) for _, n, a in comparison.moves] == sorted(deltas, reverse=True)
    # A clean synthetic schedule means naive and adjusted mostly agree.
    assert comparison.spearman > 0.8


def test_spearman_against_a_known_case() -> None:
    assert dvp._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert dvp._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Ties get average ranks, so a constant column is undefined, not 0.
    assert math.isnan(dvp._spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert dvp._spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)


def test_ranks_are_toughest_first() -> None:
    frame, _, _ = synthetic_season(seed=59, planted={"SEA": -6.0, "CAR": +6.0})
    fit = dvp.fit_position(frame, "WR", shrinkage=dvp.Shrinkage(1.0, 1.0, None))
    ordered = fit.ranked()
    assert [e.rank for e in ordered] == list(range(1, len(TEAMS) + 1))
    assert ordered[0].team == "SEA"
    assert ordered[-1].team == "CAR"
    assert ordered[0].effect < ordered[-1].effect


# --------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------


def test_cross_validate_prefers_the_true_shrinkage_direction() -> None:
    """On data with a real defense signal, CV must not pick the no-defense baseline.

    Asserted on the *curve* rather than on `gain`. `gain` compares the argmin
    against another point of the same curve, so `result.gain > 0` and
    `best_rmse < baseline_rmse` are both implied by `not isinf(best.k_def)` and
    cannot fail independently of it -- see `test_cv_gain_cannot_report_harm`.
    A fixed-`k_def` comparison can fail, so that is what is checked.
    """
    frame, _, _ = synthetic_season(seed=61, defense_sd=3.0, noise_sd=3.0)
    result = dvp.cross_validate(
        frame,
        "WR",
        k_off_grid=(1.0,),
        k_def_grid=(5.0, 30.0, math.inf),
        halflife_grid=(None,),
        first_test_week=6,
    )
    assert not math.isinf(result.best.k_def)
    by_k = {p.shrinkage.k_def: p.rmse for p in result.curve}
    assert by_k[5.0] < by_k[math.inf], (
        f"a real +3.0 SD defense signal must beat no defense term: {by_k}"
    )
    assert by_k[5.0] < by_k[30.0], "and light shrinkage must beat heavy on a loud signal"
    assert len(result.curve) == 3
    assert all(p.n == result.curve[0].n for p in result.curve)


def test_cv_gain_cannot_report_harm() -> None:
    """`gain` is bounded below by zero by construction, so it is not evidence.

    Pinned because it is exactly the statistic a reader will quote as proof that
    the defense term helps. With the true defense effects set to zero, `k_def=30`
    is genuinely worse out of sample than no defense term at all -- the curve says
    so -- and `gain` still reports 0.0.
    """
    frame, _, _ = synthetic_season(seed=67, defense_sd=0.0, noise_sd=5.0)
    result = dvp.cross_validate(
        frame, "WR", k_off_grid=(1.0,), k_def_grid=(2.0, 30.0, math.inf), halflife_grid=(None,)
    )
    by_k = {p.shrinkage.k_def: p.rmse for p in result.curve}
    real_harm = by_k[30.0] - by_k[math.inf]
    assert real_harm > 0.01, f"expected the defense term to hurt on no-signal data, {by_k}"
    assert result.gain == 0.0, "gain cannot go negative; it silently floors at zero"
    assert result.baseline_rmse == result.best_rmse


def test_cross_validate_prefers_no_defense_when_there_is_none() -> None:
    """With defense_sd = 0 the honest answer is 'do not adjust'."""
    frame, _, _ = synthetic_season(seed=67, defense_sd=0.0, noise_sd=5.0)
    result = dvp.cross_validate(
        frame,
        "WR",
        k_off_grid=(1.0,),
        k_def_grid=(2.0, 30.0, math.inf),
        halflife_grid=(None,),
    )
    assert math.isinf(result.best.k_def), f"picked k_def={result.best.k_def}"


def test_cv_profile_slices_one_axis() -> None:
    frame, _, _ = synthetic_season(seed=71, defense_sd=3.0)
    result = dvp.cross_validate(
        frame,
        "WR",
        k_off_grid=(0.5, 1.0),
        k_def_grid=(5.0, 30.0),
        halflife_grid=(None, 8.0),
    )
    profile = result.profile("k_def")
    assert len(profile) == 2
    assert all(p.shrinkage.k_off == result.best.k_off for p in profile)
    assert all(p.shrinkage.halflife == result.best.halflife for p in profile)
    assert [p.shrinkage.k_def for p in profile] == [5.0, 30.0]
    # None ("no decay") sorts last on the half-life axis, and a k_off of 0.0 must
    # sort first rather than being mistaken for a missing value.
    zero = dvp.cross_validate(
        frame, "WR", k_off_grid=(0.0, 1.0), k_def_grid=(30.0,), halflife_grid=(8.0, None)
    )
    assert [p.shrinkage.halflife for p in zero.profile("halflife")] == [8.0, None]
    assert [p.shrinkage.k_off for p in zero.profile("k_off")] == [0.0, 1.0]
    with pytest.raises(ValueError):
        result.profile("nope")


# --------------------------------------------------------------------------------------
# Strength of schedule
# --------------------------------------------------------------------------------------


def _toy_schedule() -> pl.DataFrame:
    """Four teams, three weeks, one bye. Small enough to check by hand."""
    rows = [
        (2025, 15, "AAA", "SEA", True),
        (2025, 16, "AAA", "CAR", False),
        (2025, 17, "AAA", "SEA", True),
        (2025, 15, "BBB", "CAR", False),
        (2025, 17, "BBB", "CAR", True),
    ]
    return pl.DataFrame(rows, schema=["season", "week", "team", "opponent", "home"], orient="row")


def test_schedule_strength_averages_opponent_effects() -> None:
    frame, _, _ = synthetic_season(seed=73, planted={"SEA": -4.0, "CAR": +4.0})
    fit = dvp.fit_position(frame, "WR", shrinkage=dvp.Shrinkage(1.0, 1.0, None))
    table = dvp.schedule_strength(fit, _toy_schedule(), weeks=dvp.PLAYOFF_WEEKS)

    sea, car = fit.effect("SEA"), fit.effect("CAR")
    assert table["AAA"].games == 3
    assert table["AAA"].total == pytest.approx(2 * sea + car)
    assert table["AAA"].per_game == pytest.approx((2 * sea + car) / 3)
    assert table["AAA"].idle_weeks == ()
    assert table["BBB"].games == 2
    assert table["BBB"].idle_weeks == (16,)
    assert table["BBB"].per_game == pytest.approx(car)
    # BBB only plays the soft defense, so it has the easier schedule -> higher rank number.
    assert table["AAA"].rank == 1
    assert table["BBB"].rank == 2
    assert table["AAA"].opponents == ((15, "SEA"), (16, "CAR"), (17, "SEA"))


def test_playoff_and_rest_of_season_windows() -> None:
    frame, _, _ = synthetic_season(seed=79)
    fit = dvp.fit_position(frame, "WR")
    schedule = _toy_schedule()
    playoff = dvp.playoff_schedule_strength(fit, schedule)
    ros = dvp.rest_of_season_strength(fit, schedule, from_week=15, through_week=17)
    assert playoff["AAA"].weeks == (15, 16, 17)
    assert ros["AAA"].per_game == pytest.approx(playoff["AAA"].per_game)
    with pytest.raises(ValueError):
        dvp.rest_of_season_strength(fit, schedule, from_week=17, through_week=15)
    with pytest.raises(ValueError):
        dvp.schedule_strength(fit, schedule, weeks=[])
    with pytest.raises(dvp.DvpError):
        dvp.schedule_strength(fit, schedule, weeks=[3])


def test_unknown_opponent_contributes_nothing() -> None:
    frame, _, _ = synthetic_season(seed=83, planted={"SEA": -4.0})
    fit = dvp.fit_position(frame, "WR", shrinkage=dvp.Shrinkage(1.0, 1.0, None))
    schedule = pl.DataFrame(
        [(2025, 15, "AAA", "SEA", True), (2025, 16, "AAA", "XXX", False)],
        schema=["season", "week", "team", "opponent", "home"],
        orient="row",
    )
    table = dvp.schedule_strength(fit, schedule, weeks=(15, 16))
    assert table["AAA"].total == pytest.approx(fit.effect("SEA"))
    assert table["AAA"].games == 2


def test_bye_rows_are_not_scored_as_games() -> None:
    """The adapter's team-week grid marks byes with a null opponent.

    A null falls through `effect()`'s unseen-team path to 0.0, so without an
    explicit filter a bye counts as a played average game: the game count is too
    high, `per_game` is pulled toward zero, and `idle_weeks` -- the field that
    exists to surface byes -- comes back empty. Nothing about that output looks
    wrong, which is why it is pinned.
    """
    frame, _, _ = synthetic_season(seed=107, planted={"SEA": -4.0})
    fit = dvp.fit_position(frame, "WR", shrinkage=dvp.Shrinkage(1.0, 1.0, None))
    with_bye = pl.DataFrame(
        [
            (2025, 15, "AAA", "SEA", True),
            (2025, 16, "AAA", None, None),  # the bye row nflverse.team_weeks emits
            (2025, 17, "AAA", "SEA", False),
        ],
        schema=["season", "week", "team", "opponent", "home"],
        orient="row",
    )
    table = dvp.schedule_strength(fit, with_bye, weeks=dvp.PLAYOFF_WEEKS)
    assert table["AAA"].games == 2, "the bye is not a game"
    assert table["AAA"].idle_weeks == (16,)
    assert table["AAA"].per_game == pytest.approx(fit.effect("SEA"))
    assert table["AAA"].total == pytest.approx(2 * fit.effect("SEA"))
    # A frame that cannot answer the question at all is an error, not a zero.
    with pytest.raises(ValueError, match="missing"):
        dvp.schedule_strength(fit, with_bye.drop("opponent"), weeks=dvp.PLAYOFF_WEEKS)


def test_player_schedule_strength_accepts_a_one_shot_iterable() -> None:
    """`weeks` is typed `Iterable[int]`, so a generator is legal input.

    It must be materialised once: draining it on the first position leaves every
    later position with an empty week set and a "needs at least one week" error
    that points nowhere near the cause.
    """
    frames = [synthetic_season(seed=i, weeks=8, position=p)[0] for i, p in enumerate(dvp.POSITIONS)]
    model = dvp.fit(pl.concat(frames))
    players = pl.DataFrame(
        [("p1", "Ace Receiver", "WR", "AAA"), ("p2", "Bo Back", "RB", "AAA")],
        schema=["player_id", "player_display_name", "position", "team"],
        orient="row",
    )
    assert len(model.positions) > 1, "the bug only bites from the second position on"
    out = dvp.player_schedule_strength(
        model, players, _toy_schedule(), weeks=(w for w in dvp.PLAYOFF_WEEKS)
    )
    assert {p.player_id for p in out} == {"p1", "p2"}


def test_player_schedule_strength_joins_position_and_team() -> None:
    frames = [synthetic_season(seed=i, weeks=8, position=p)[0] for i, p in enumerate(dvp.POSITIONS)]
    model = dvp.fit(pl.concat(frames))
    players = pl.DataFrame(
        [
            ("p1", "Ace Receiver", "WR", "AAA"),
            ("p2", "Bo Back", "RB", "BBB"),
            ("p3", "Cal Kicker", "K", "AAA"),
            ("p4", "Dee Deep", "WR", "ZZZ"),
        ],
        schema=["player_id", "player_display_name", "position", "team"],
        orient="row",
    )
    out = dvp.player_schedule_strength(model, players, _toy_schedule(), weeks=dvp.PLAYOFF_WEEKS)
    ids = {p.player_id for p in out}
    assert ids == {"p1", "p2"}, "unfitted position and unscheduled team are dropped, not zeroed"
    assert out == sorted(out, key=lambda p: -p.per_game)
    assert next(p for p in out if p.player_id == "p1").name == "Ace Receiver"

    with pytest.raises(ValueError, match="missing"):
        dvp.player_schedule_strength(
            model, players.drop("team"), _toy_schedule(), weeks=dvp.PLAYOFF_WEEKS
        )


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def test_empty_and_degenerate_inputs() -> None:
    frame, _, _ = synthetic_season(seed=89, weeks=2)
    with pytest.raises(dvp.DvpError, match="no TE rows"):
        dvp.fit_position(frame, "TE")
    with pytest.raises(dvp.DvpError, match="no observations"):
        dvp.solve_two_way(
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.array([]),
            np.array([]),
            0,
            0,
            dvp.Shrinkage(1.0, 1.0),
        )
    with pytest.raises(dvp.DvpError, match="weights are zero"):
        dvp.solve_two_way(
            np.zeros(3, dtype=np.intp),
            np.zeros(3, dtype=np.intp),
            np.ones(3),
            np.zeros(3),
            1,
            1,
            dvp.Shrinkage(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="same length"):
        dvp.solve_two_way(
            np.zeros(3, dtype=np.intp),
            np.zeros(2, dtype=np.intp),
            np.ones(3),
            np.ones(3),
            1,
            1,
            dvp.Shrinkage(1.0, 1.0),
        )


def test_infinite_k_def_pins_the_defense_term_without_a_nan() -> None:
    """The CV baseline runs at k_def = inf; `inf * 0` is NaN if written naively."""
    frame, _, _ = synthetic_season(seed=101, weeks=6)
    off, dfn, y, w, n_off, n_def = _index_arrays(frame)
    solution = dvp.solve_two_way(
        off, dfn, y, w, n_off, n_def, dvp.Shrinkage(1.0, math.inf), max_iter=500
    )
    assert np.all(solution.defense == 0.0)
    assert all(math.isfinite(o) for o in solution.objective)
    assert solution.converged


def test_fit_with_no_positions_is_an_error() -> None:
    frame, _, _ = synthetic_season(seed=103, weeks=3)
    with pytest.raises(dvp.DvpError, match="at least one position"):
        dvp.fit(frame, positions=())


def test_through_week_truncates_the_fit() -> None:
    frame, _, _ = synthetic_season(seed=97, weeks=17)
    early = dvp.fit_position(frame, "WR", through_week=6)
    full = dvp.fit_position(frame, "WR")
    assert early.through_week == 6
    assert full.through_week == 17
    assert early.observations < full.observations
    assert max(e.games for e in early.defense.values()) <= 6


# --------------------------------------------------------------------------------------
# Live data
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_naive_vs_adjusted_on_real_2025() -> None:
    """The headline comparison: how far the adjustment moves a real DvP table."""
    frame = dvp.player_weeks(2025)
    model = dvp.fit(frame)
    moves = {}
    for position in dvp.POSITIONS:
        comparison = dvp.compare_naive(model[position])
        moves[position] = comparison
        print(
            f"{position}: mean |d rank| = {comparison.mean_abs_move:.2f}, "
            f"max = {comparison.max_abs_move}, spearman = {comparison.spearman:+.3f}, "
            f"top movers = {[(t, f'{n}->{a}') for t, n, a in comparison.top_movers(3)]}"
        )
        assert comparison.teams == 32
        assert model[position].converged
        # An adjustment that moved nothing would not be worth the module, and one
        # that reshuffled the table completely would mean the fit is unstable.
        assert 1.0 < comparison.mean_abs_move < 8.0
        assert comparison.spearman > 0.5
    assert moves["WR"].mean_abs_move > 2.5, "WR is where schedule bias bites hardest"


@pytest.mark.network
def test_espn_positional_ratings_are_the_naive_measure() -> None:
    """ESPN's own DvP table is raw points allowed. Proving it is the whole edge."""
    from dotenv import load_dotenv

    from fantasy_quant.espn.client import EspnError
    from fantasy_quant.espn.league import open_league

    load_dotenv("/Users/cole/code/git/fantasy_quant/.env")
    swid, s2 = os.getenv("ESPN_SWID"), os.getenv("ESPN_S2")
    if not (swid and s2):
        pytest.skip("ESPN credentials not configured")

    # 161496047 is the half-PPR league; 2025 is the newest season whose ratings
    # are populated (the 2026 table is present but empty until games are played).
    # Score our side in half-PPR so the comparison is like for like.
    league = open_league(161496047, 2025, swid, s2)
    try:
        ratings = league.positional_ratings()
    except EspnError as err:
        # espn_s2 dies silently, usually mid-season. That is a credentials
        # problem, not a regression in this module -- skip rather than fail.
        if str(err).startswith("401"):
            pytest.skip("ESPN credentials are stale; re-copy espn_s2")
        raise
    frame = dvp.player_weeks(2025)
    model = dvp.fit(frame, points_column=dvp.HALF_PPR)

    for position in dvp.POSITIONS:
        comparison = dvp.compare_with_espn(model[position], ratings, frame)
        print(
            f"{position}: rho(espn, team-total naive) = {comparison.rho_team_total:+.3f}, "
            f"rho(espn, adjusted) = {comparison.rho_adjusted:+.3f}, "
            f"espn mean {comparison.espn_mean:.2f} vs ours {comparison.our_team_total_mean:.2f}, "
            f"mean |d rank| = {comparison.mean_abs_move:.2f} (max {comparison.max_abs_move})"
        )
        assert comparison.teams == 32
        # The claim under test: ESPN publishes the naive team-total measure.
        assert comparison.rho_team_total > 0.95
        assert comparison.our_team_total_mean == pytest.approx(comparison.espn_mean, rel=0.05)
        # And our adjusted table is a materially different ranking.
        assert comparison.rho_adjusted < comparison.rho_team_total
        assert comparison.mean_abs_move > 1.0


@pytest.mark.network
def test_schedule_strength_on_the_live_schedule() -> None:
    from fantasy_quant.data import nflverse

    frame = dvp.player_weeks(2025)
    fit = dvp.fit_position(frame, "WR")
    schedule = dvp.team_schedule(2025)
    assert schedule.height == 272 * 2

    # The adapter's own team-week frame carries bye rows; feeding it directly must
    # give the identical answer rather than scoring 32 byes as average games.
    adapter = nflverse.remaining_opponents(nflverse.schedules(2025), 2025, from_week=1)
    assert adapter.filter(pl.col("opponent").is_null()).height == 32, "byes present in the input"
    ours = dvp.schedule_strength(fit, schedule, weeks=range(1, 19))
    theirs = dvp.schedule_strength(fit, adapter, weeks=range(1, 19))
    assert set(ours) == set(theirs)
    for team, entry in ours.items():
        assert theirs[team].games == entry.games == 17
        assert theirs[team].per_game == pytest.approx(entry.per_game)
        assert theirs[team].idle_weeks == entry.idle_weeks != ()

    playoff = dvp.playoff_schedule_strength(fit, schedule)
    assert len(playoff) == 32
    assert all(entry.games == 3 for entry in playoff.values())
    assert sorted(e.rank for e in playoff.values()) == list(range(1, 33))
    hardest = min(playoff.values(), key=lambda e: e.rank)
    easiest = max(playoff.values(), key=lambda e: e.rank)
    print(
        f"WR playoff SoS 2025: hardest {hardest.team} {hardest.per_game:+.2f}/player-game, "
        f"easiest {easiest.team} {easiest.per_game:+.2f}"
    )
    assert hardest.per_game < easiest.per_game

    ros = dvp.rest_of_season_strength(fit, schedule, from_week=10)
    assert all(1 <= e.games <= 8 for e in ros.values())
