"""Tests for the opportunity / stability / xFP module.

The offline half is the bulk of it and runs on synthetic frames with properties
chosen so the right answer is known in advance: a series with a set autocorrelation
must come back with that autocorrelation, a player handed an absurd touchdown rate
must come out the top of the screen, and the four-way decomposition of actual minus
expected must close to floating-point exactly.

The `network` half re-measures the constants baked into the module against the real
nflverse corpus -- the WOPR refit, the per-opportunity conversion rates, and the
Cooper Kupp 2019 archetype. Those are the numbers that would silently rot if
nflverse restated a season, so they are checked rather than trusted. CI deselects
them with `-m 'not network'`.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from fantasy_quant.analysis import opportunity as op
from fantasy_quant.data import nflverse as nv

# --- synthetic frames -----------------------------------------------------------


def _week_row(
    player_id: str,
    season: int,
    week: int,
    position: str = "WR",
    team: str = "AAA",
    *,
    targets: float = 0.0,
    receptions: float = 0.0,
    receiving_yards: float = 0.0,
    receiving_tds: float = 0.0,
    carries: float = 0.0,
    rushing_yards: float = 0.0,
    rushing_tds: float = 0.0,
    fumbles_lost: float = 0.0,
    buckets: dict[str, int] | None = None,
    team_targets: float = 30.0,
    team_carries: float = 25.0,
    team_air_yards: float = 250.0,
) -> dict[str, object]:
    """One row shaped like `usage_weeks` output, with sane defaults."""
    buckets = buckets or {}
    row: dict[str, object] = {
        "season": season,
        "week": week,
        "player_id": player_id,
        "player_name": player_id,
        "position": position,
        "team": team,
        "opponent": "BBB",
        "targets": targets,
        "receptions": receptions,
        "receiving_yards": receiving_yards,
        "receiving_air_yards": receiving_yards,
        "receiving_yards_after_catch": 0.0,
        "receiving_tds": receiving_tds,
        "receiving_fumbles_lost": 0.0,
        "carries": carries,
        "rushing_yards": rushing_yards,
        "rushing_tds": rushing_tds,
        "rushing_fumbles_lost": fumbles_lost,
        "target_share": targets / team_targets if team_targets else 0.0,
        "air_yards_share": targets / team_targets if team_targets else 0.0,
        "racr": 1.0,
        "pacr": 1.0,
        "fantasy_points_ppr": 0.0,
        "team_targets": team_targets,
        "team_carries": team_carries,
        "team_air_yards": team_air_yards,
        "expected_receptions": receptions,
        "snap_share": 0.8,
        "routes": None,
        "route_participation": None,
    }
    for column in op.BUCKET_COLUMNS:
        row[column] = buckets.get(column, 0)
        row[f"team_{column}"] = max(buckets.get(column, 0), 1) * 4
    return row


def _frame(rows: list[dict[str, object]], wopr: op.WoprWeights = op.CANONICAL_WOPR) -> pl.DataFrame:
    df = pl.DataFrame(rows).with_columns(pl.col("season", "week").cast(pl.Int32))
    return op._derive_metrics(df, wopr)


# --- WOPR -----------------------------------------------------------------------


def test_wopr_matches_hand_computation():
    # 1.5*0.24 + 0.7*0.31 = 0.36 + 0.217 = 0.577
    assert op.CANONICAL_WOPR(0.24, 0.31) == pytest.approx(0.577)
    assert op.CANONICAL_WOPR(0.0, 0.0) == 0.0
    # A pure air-yards hog with no targets is impossible, but the formula is linear
    # and must not special-case it.
    assert op.CANONICAL_WOPR(0.0, 1.0) == pytest.approx(0.7)


def test_wopr_expr_matches_the_scalar_form():
    df = pl.DataFrame({"target_share": [0.1, 0.24, 0.33], "air_yards_share": [0.05, 0.31, 0.4]})
    got = df.select(op.CANONICAL_WOPR.expr().alias("wopr"))["wopr"].to_list()
    pairs = zip(df["target_share"], df["air_yards_share"], strict=True)
    want = [op.CANONICAL_WOPR(t, a) for t, a in pairs]
    assert got == pytest.approx(want)


def test_wopr_rescale_preserves_the_ratio():
    refit = op.WoprWeights(51.5, 3.5)
    rescaled = refit.rescaled(2.2)
    assert rescaled.target_share + rescaled.air_yards_share == pytest.approx(2.2)
    assert rescaled.ratio == pytest.approx(refit.ratio)
    assert op.CANONICAL_WOPR.rescaled(2.2) == op.CANONICAL_WOPR


def test_wopr_rescale_rejects_a_zero_sum():
    with pytest.raises(ValueError, match="sum to zero"):
        op.WoprWeights(1.0, -1.0).rescaled()


def test_wopr_ratio_is_infinite_without_air_yards():
    assert op.WoprWeights(2.2, 0.0).ratio == math.inf


def test_fit_wopr_recovers_planted_coefficients():
    """Points built as 40*ts + 8*ays must fit back to 40/8, ratio 5."""
    rng = np.random.default_rng(11)
    n = 800
    ts = rng.uniform(0.02, 0.35, n)
    ays = np.clip(ts * 1.2 + rng.normal(0, 0.05, n), 0.0, None)
    points = 40.0 * ts + 8.0 * ays
    rows = [
        _week_row(
            f"p{i % 40}",
            2024,
            i % 17 + 1,
            targets=ts[i] * 30.0,
            receptions=points[i],  # reception_points=1.0 makes _y == points
            receiving_yards=0.0,
        )
        for i in range(n)
    ]
    df = _frame(rows).with_columns(target_share=pl.Series(ts), air_yards_share=pl.Series(ays))
    fit = op.fit_wopr_weights(df, positions=("WR",), rescale_to=None)
    assert fit.raw_target_share == pytest.approx(40.0, rel=1e-6)
    assert fit.raw_air_yards_share == pytest.approx(8.0, rel=1e-6)
    assert fit.ratio == pytest.approx(5.0, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert "ratio 5.0" in str(fit)


def test_fit_wopr_refuses_a_frame_it_cannot_fit():
    with pytest.raises(ValueError, match="missing"):
        op.fit_wopr_weights(pl.DataFrame({"target_share": [0.1]}))
    with pytest.raises(ValueError, match="refusing to fit"):
        op.fit_wopr_weights(_frame([_week_row("p", 2024, 1, targets=3.0)]))


# --- field buckets --------------------------------------------------------------


@pytest.mark.parametrize(
    ("yardline", "bucket"),
    [
        (1, "goal_line"),
        (5, "goal_line"),
        (6, "inside_10"),
        (10, "inside_10"),
        (11, "fringe_rz"),
        (20, "fringe_rz"),
        (21, "midfield"),
        (50, "midfield"),
        (51, "own_half"),
        (99, "own_half"),
    ],
)
def test_bucket_boundaries(yardline, bucket):
    assert op.bucket_of(yardline) == bucket


def test_bucket_expr_agrees_with_the_scalar_form():
    df = pl.DataFrame({"yardline_100": list(range(1, 101))})
    got = df.select(op._bucket_expr())["bucket"].to_list()
    assert got == [op.bucket_of(y) for y in range(1, 101)]


def test_red_zone_buckets_are_the_first_three():
    assert op.FIELD_BUCKETS[:3] == op.RED_ZONE_BUCKETS
    assert op.FIELD_BUCKETS[:2] == op.INSIDE_10_BUCKETS
    assert all(op.BUCKET_MAX[b] <= 20 for b in op.RED_ZONE_BUCKETS)


# --- stability ------------------------------------------------------------------


def _ar1_frame(rho: float, n_players: int = 120, n_weeks: int = 17, seed: int = 7) -> pl.DataFrame:
    """A metric with a KNOWN lag-1 autocorrelation of `rho`, in stationary form.

    x[t] = rho*x[t-1] + sqrt(1-rho^2)*eps keeps the marginal variance at 1 whatever
    rho is, so the measured autocorrelation is rho and not rho scaled by a drifting
    variance.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for p in range(n_players):
        x = rng.normal()
        for w in range(1, n_weeks + 1):
            x = rho * x + math.sqrt(max(1.0 - rho * rho, 0.0)) * rng.normal()
            rows.append(
                _week_row(
                    f"p{p}",
                    2024,
                    w,
                    targets=8.0,
                    receptions=5.0,
                    carries=0.0,
                )
                | {"_signal": x}
            )
    return _frame(rows).with_columns(target_share=pl.col("_signal"))


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_stability_recovers_a_known_autocorrelation(rho):
    df = _ar1_frame(rho)
    rows = op.stability_table(
        df, metrics=(op.StabilityMetric("target_share", "targets"),), max_window=4
    )
    assert len(rows) == 1
    assert rows[0].lag1 == pytest.approx(rho, abs=0.035)
    assert rows[0].lag1_n == 120 * 16


def _true_score_frame(n_players: int, k: int, sd_true: float, sd_error: float, seed: int):
    """obs = T_i + e_ig with T ~ N(0, sd_true^2) and e ~ N(0, sd_error^2), iid.

    The classical true-score model, whose k-game reliability is closed form:
    sd_true^2 / (sd_true^2 + sd_error^2/k). Nothing here is derived from the module.
    """
    rng = np.random.default_rng(seed)
    talent = rng.normal(0.0, sd_true, n_players)
    rows = []
    for i in range(n_players):
        for w in range(1, 2 * k + 1):
            rows.append(
                _week_row(f"p{i}", 2024, w, targets=8.0, receptions=5.0)
                | {"_signal": talent[i] + rng.normal(0.0, sd_error)}
            )
    return _frame(rows).with_columns(target_share=pl.col("_signal"))


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_split_half_matches_closed_form_reliability(k):
    """_split_half(k) must be the reliability of a k-game window, not a 2k-game one.

    This is the test that catches a stray Spearman-Brown correction. Two disjoint
    k-game windows are parallel measurements of a k-game window, so their
    correlation IS its reliability; doubling it with 2r/(1+r) silently reports the
    2k figure under the k label, and every "games to 0.70" comes out halved. The
    two answers are far enough apart here (0.43 vs 0.60 at k=3) that no tolerance
    wide enough to be useful can straddle them.
    """
    sd_true, sd_error = 1.0, 2.0
    df = _true_score_frame(3000, k, sd_true, sd_error, seed=400 + k)
    got = op._split_half(df, op.StabilityMetric("target_share", "targets"), k)

    def reliability(games: int) -> float:
        return sd_true**2 / (sd_true**2 + sd_error**2 / games)

    assert got == pytest.approx(reliability(k), abs=0.03)
    assert abs(got - reliability(2 * k)) > 0.03  # and it is NOT the doubled figure


def test_reliability_curve_rises_with_the_window():
    """More games can only help, and the curve must stay inside [0, 1]."""
    df = _true_score_frame(1500, 4, 1.0, 2.0, seed=77)
    rows = op.stability_table(
        df,
        metrics=(op.StabilityMetric("target_share", "targets"),),
        min_games=1,
        min_opportunities_per_game=0.0,
        max_window=4,
    )
    curve = [r for _, r in rows[0].reliability_curve if math.isfinite(r)]
    assert len(curve) == 4
    assert all(0.0 <= r <= 1.0 for r in curve)
    assert curve == sorted(curve)
    # 0.70 needs sd_true^2/(sd_true^2 + 4/n) >= 0.7, i.e. n >= 9.33 -- out of range.
    assert rows[0].games_to_stabilize is None


def test_stability_separates_stable_from_unstable():
    stable = _ar1_frame(0.8, seed=1).with_columns(carry_share=pl.col("target_share"))
    noisy = _ar1_frame(0.02, seed=2)
    df = pl.concat(
        [
            stable.with_columns(pl.col("player_id") + "_s", ypc=pl.col("target_share")),
            noisy.with_columns(pl.col("player_id") + "_n", ypc=pl.col("target_share")),
        ]
    )
    rows = {
        r.metric: r
        for r in op.stability_table(
            df,
            metrics=(
                op.StabilityMetric("carry_share", "carries"),
                op.StabilityMetric("ypc", "targets"),
            ),
            max_window=4,
        )
    }
    # carry_share only exists on the stable half (carries is 0 on the noisy rows,
    # so the volume gate drops them); ypc spans both and lands in between.
    assert rows["carry_share"].lag1 > 0.7
    assert rows["carry_share"].stable
    assert rows["carry_share"].verdict == "project forward"
    assert rows["ypc"].lag1 < 0.6
    assert op.format_stability_table(list(rows.values())).count("\n") == 3


def test_stability_reports_nan_rather_than_inventing_a_number():
    df = _frame([_week_row("p", 2024, w, targets=6.0) for w in range(1, 9)])
    rows = op.stability_table(
        df,
        metrics=(op.StabilityMetric("target_share", "targets"),),
        min_games=1,
        min_opportunities_per_game=0.0,
    )
    assert math.isnan(rows[0].lag1)  # 7 pairs, below the 30 the estimator needs
    assert rows[0].games_to_stabilize is None
    assert not rows[0].stable
    assert "nan" in op.format_stability_table(rows)


def test_lag1_is_next_appearance_not_next_week():
    """lag1 chains consecutive *appearances*, so a gated-out week is skipped.

    Documented behaviour, pinned because the alternative reading (strict week w to
    w+1) is what the name suggests and would silently change every number in the
    published table.
    """
    rows = [_week_row("p", 2024, w, targets=8.0, receptions=5.0) for w in (1, 2, 5, 6)]
    df = _frame(rows).with_columns(target_share=pl.Series([0.1, 0.2, 0.3, 0.4]))
    metric = op.StabilityMetric("target_share", "targets")
    d = op._usable(df, metric).sort("player_id", "season", "week")
    d = d.with_columns(_next_week=pl.col("week").shift(-1).over("player_id", "season"))
    pairs = d.drop_nulls("_next_week")
    assert pairs.height == 3  # weeks 3 and 4 are absent, and 2->5 is still a pair
    assert (pairs["_next_week"] - pairs["week"]).to_list() == [1, 3, 1]
    _, n = op._lag1(df, metric)
    assert n == 3


def test_stability_ignores_a_metric_the_frame_does_not_have():
    df = _ar1_frame(0.5, n_players=40)
    rows = op.stability_table(df, metrics=(op.StabilityMetric("routes_run", "targets"),))
    assert math.isnan(rows[0].lag1)
    assert rows[0].lag1_n == 0


def test_shrink_to_mean_interpolates_by_sample_size():
    assert op.shrink_to_mean(0.5, 0.1, n_observations=0, stabilization=10) == pytest.approx(0.1)
    assert op.shrink_to_mean(0.5, 0.1, n_observations=10, stabilization=10) == pytest.approx(0.3)
    assert op.shrink_to_mean(0.5, 0.1, n_observations=990, stabilization=10) == pytest.approx(0.496)
    with pytest.raises(ValueError):
        op.shrink_to_mean(0.5, 0.1, n_observations=1, stabilization=0)


# --- conversion rates and xFP ---------------------------------------------------


def test_default_rates_reproduce_the_reference_per_opportunity_values():
    """RB carry ~0.60 and target ~1.57 were the reference; ours are 0.611 / 1.687."""
    rates = op.DEFAULT_CONVERSION_RATES
    carry_cells = [rates.cells[("RB", "carry", b)] for b in op.FIELD_BUCKETS]
    carry = sum(c.n * c.points("carry", op.PPR, op.RB) for c in carry_cells) / sum(
        c.n for c in carry_cells
    )
    target_cells = [rates.cells[("*", "target", b)] for b in op.FIELD_BUCKETS]
    target = sum(c.n * c.points("target", op.PPR, op.WR) for c in target_cells) / sum(
        c.n for c in target_cells
    )
    assert carry == pytest.approx(0.611, abs=0.005)
    assert target == pytest.approx(1.687, abs=0.005)
    assert target / carry == pytest.approx(2.76, abs=0.02)

    # Inside the 10 the target premium compresses -- reference 1.79x, ours 1.70x.
    i10 = [(rates.cells[("*", k, b)], k) for k in ("carry", "target") for b in op.INSIDE_10_BUCKETS]
    i10_carry = sum(c.n * c.points(k, op.PPR, op.RB) for c, k in i10 if k == "carry") / sum(
        c.n for c, k in i10 if k == "carry"
    )
    i10_target = sum(c.n * c.points(k, op.PPR, op.WR) for c, k in i10 if k == "target") / sum(
        c.n for c, k in i10 if k == "target"
    )
    assert i10_target / i10_carry == pytest.approx(1.70, abs=0.03)
    assert i10_target / i10_carry < target / carry  # the whole point


def test_conversion_value_is_league_specific():
    cell = op.DEFAULT_CONVERSION_RATES.get("WR", "target", "own_half")
    full = cell.points("target", op.PPR, op.WR)
    half = cell.points("target", op.HALF_PPR, op.WR)
    standard = cell.points("target", op.STANDARD, op.WR)
    assert full > half > standard
    # The whole difference is the reception term: 0.5 * catch_rate.
    assert full - half == pytest.approx(0.5 * cell.catch_rate)
    assert half - standard == pytest.approx(0.5 * cell.catch_rate)


def test_conversion_rates_fall_back_to_the_pooled_row_on_thin_cells():
    rates = op.DEFAULT_CONVERSION_RATES
    # 68 WR carries inside the 5 in seven seasons: not a projection input.
    assert rates.cells[("WR", "carry", "goal_line")].n < rates.min_cell
    assert rates.get("WR", "carry", "goal_line") is rates.cells[("*", "carry", "goal_line")]
    # A fat cell is used directly.
    assert rates.get("WR", "target", "own_half") is rates.cells[("WR", "target", "own_half")]


def test_conversion_rates_raise_on_an_unknown_cell():
    rates = op.ConversionRates(cells={}, seasons=(2024,))
    with pytest.raises(KeyError):
        rates.get("WR", "target", "own_half")


def test_opportunity_value_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="carry.*target"):
        op.OpportunityValue(0.6, 8.0, 0.01, 0.004, 100).stats("kickoff")


def test_expected_points_prices_each_bucket():
    """One goal-line carry must be worth more than one from your own half."""
    rows = [
        _week_row("goal", 2024, 1, "RB", carries=1.0, buckets={"carries_goal_line": 1}),
        _week_row("deep", 2024, 1, "RB", carries=1.0, buckets={"carries_own_half": 1}),
    ]
    out = op.expected_points(_frame(rows))
    by_id = dict(zip(out["player_id"], out["xfp"], strict=True))
    rates = op.DEFAULT_CONVERSION_RATES
    assert by_id["goal"] == pytest.approx(
        rates.get("RB", "carry", "goal_line").points("carry", op.PPR, op.RB)
    )
    assert by_id["deep"] == pytest.approx(
        rates.get("RB", "carry", "own_half").points("carry", op.PPR, op.RB)
    )
    assert by_id["goal"] == pytest.approx(2.479, abs=0.001)
    assert by_id["deep"] == pytest.approx(0.479, abs=0.001)
    assert by_id["goal"] / by_id["deep"] > 5


def test_expected_points_needs_the_play_by_play_columns():
    df = _frame([_week_row("p", 2024, 1, carries=1.0)]).drop("carries_goal_line")
    with pytest.raises(ValueError, match="missing"):
        op.expected_points(df)


def test_expected_points_refuses_an_all_null_field_position_frame():
    df = _frame([_week_row("p", 2024, 1, carries=1.0)]).with_columns(
        [pl.lit(None, dtype=pl.Int32).alias(c) for c in op.BUCKET_COLUMNS]
    )
    with pytest.raises(ValueError, match="entirely null"):
        op.expected_points(df)


def test_decomposition_closes_exactly():
    """actual - xfp must equal the four component terms, with no residual."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(200):
        buckets = {c: int(rng.integers(0, 4)) for c in op.BUCKET_COLUMNS}
        carries = sum(v for k, v in buckets.items() if k.startswith("carries"))
        targets = sum(v for k, v in buckets.items() if k.startswith("targets"))
        rows.append(
            _week_row(
                f"p{i}",
                2024,
                1,
                position=("RB", "WR", "TE")[i % 3],
                targets=float(targets),
                receptions=float(rng.integers(0, targets + 1)),
                receiving_yards=float(rng.integers(0, 120)),
                receiving_tds=float(rng.integers(0, 3)),
                carries=float(carries),
                rushing_yards=float(rng.integers(0, 100)),
                rushing_tds=float(rng.integers(0, 2)),
                fumbles_lost=float(rng.integers(0, 2)),
                buckets=buckets,
            )
        )
    for scorer in (op.PPR, op.HALF_PPR, op.STANDARD):
        d = op.expected_points(_frame(rows), scorer=scorer).with_columns(
            _actual=op.actual_points_expr(scorer)
        )
        price = op.component_prices(scorer, "WR")  # identical across positions here
        parts = (
            price[op.STAT_RECEPTIONS] * (d["receptions"] - d["expected_receptions_rate"])
            + price[op.STAT_RECEIVING_YARDS]
            * (d["receiving_yards"] - d["expected_receiving_yards"])
            + price[op.STAT_RUSHING_YARDS] * (d["rushing_yards"] - d["expected_rushing_yards"])
            + price[op.STAT_RECEIVING_TD] * (d["receiving_tds"] - d["expected_receiving_tds"])
            + price[op.STAT_RUSHING_TD] * (d["rushing_tds"] - d["expected_rushing_tds"])
            + price[op.STAT_FUMBLE_LOST] * (d["fumbles_lost"] - d["expected_fumbles_lost"])
        )
        residual = (d["_actual"] - d["xfp"] - parts).abs().max()
        assert residual < 1e-9


def test_ppr_scorer_matches_espn_stat_ids():
    stats = {
        op.STAT_RECEPTIONS: 6.0,
        op.STAT_RECEIVING_YARDS: 84.0,
        op.STAT_RECEIVING_TD: 1.0,
        op.STAT_RUSHING_YARDS: 12.0,
        op.STAT_RUSHING_TD: 0.0,
        op.STAT_FUMBLE_LOST: 1.0,
    }
    # 6 + 8.4 + 6 + 1.2 + 0 - 2
    assert op.PPR(stats, op.WR) == pytest.approx(19.6)
    assert op.HALF_PPR(stats, op.WR) == pytest.approx(16.6)
    # An unscored statId must be ignored, not crash -- ESPN ships derived buckets.
    assert op.PPR({"999": 500.0}, op.WR) == 0.0


# --- the regression screen ------------------------------------------------------


def _screen_frame() -> pl.DataFrame:
    """Three players with identical usage and wildly different luck."""
    rows = []
    for week in range(1, 13):
        buckets = {
            "targets_own_half": 5,
            "targets_midfield": 3,
            "targets_fringe_rz": 1,
            "targets_goal_line": 1,
        }
        common = dict(targets=10.0, receptions=7.0, receiving_yards=80.0, buckets=buckets)
        rows.append(_week_row("lucky", 2024, week, receiving_tds=2.0, **common))
        rows.append(_week_row("normal", 2024, week, receiving_tds=0.5, **common))
        rows.append(_week_row("cursed", 2024, week, receiving_tds=0.0, **common))
    return _frame(rows)


def test_screen_flags_an_absurd_touchdown_rate():
    out = op.regression_screen(_screen_frame(), season=2024)
    assert [c.player_id for c in out] == ["lucky", "normal", "cursed"]
    lucky, _, cursed = out
    assert lucky.verdict == "sell_high"
    assert lucky.driver == "touchdowns"
    assert lucky.significant
    assert lucky.touchdowns == 24.0
    assert lucky.expected_touchdowns < 10.0
    assert lucky.td_over_expected > 60.0
    assert cursed.gap_per_game < lucky.gap_per_game
    # Identical usage means identical xFP: the whole spread is luck.
    assert lucky.expected_points == pytest.approx(cursed.expected_points)


def test_screen_components_sum_to_the_gap():
    for c in op.regression_screen(_screen_frame(), season=2024):
        total = (
            c.td_over_expected
            + c.catch_over_expected
            + c.yards_over_expected
            + c.fumbles_over_expected
        )
        assert total == pytest.approx(c.points_gap, abs=1e-9)
        assert c.gap_per_game == pytest.approx(c.points_gap / c.games)
        assert c.unrepeatable_points == pytest.approx(c.td_over_expected + c.fumbles_over_expected)


def test_screen_can_rank_on_the_unrepeatable_slice():
    """A yards-driven overperformer outranks a TD-driven one on gap but not on luck."""
    rows = []
    buckets = {"targets_own_half": 8, "targets_midfield": 2}
    for week in range(1, 13):
        rows.append(
            _week_row(
                "yardage",
                2024,
                week,
                targets=10.0,
                receptions=7.0,
                receiving_yards=200.0,
                receiving_tds=0.4,
                buckets=buckets,
            )
        )
        rows.append(
            _week_row(
                "touchdowns",
                2024,
                week,
                targets=10.0,
                receptions=7.0,
                receiving_yards=70.0,
                receiving_tds=2.0,
                buckets=buckets,
            )
        )
    df = _frame(rows)
    by_gap = [c.player_id for c in op.regression_screen(df, season=2024)]
    by_luck = [c.player_id for c in op.regression_screen(df, season=2024, sort_by="unrepeatable")]
    assert by_gap == ["yardage", "touchdowns"]
    assert by_luck == ["touchdowns", "yardage"]

    with pytest.raises(ValueError, match="sort_by"):
        op.regression_screen(df, sort_by="vibes")


def test_screen_is_scoring_aware():
    df = _screen_frame()
    full = {c.player_id: c for c in op.regression_screen(df, season=2024, scorer=op.PPR)}
    half = {c.player_id: c for c in op.regression_screen(df, season=2024, scorer=op.HALF_PPR)}
    # Half PPR pays half as much for receptions, so both sides of the ledger shrink.
    assert half["lucky"].expected_points < full["lucky"].expected_points
    assert half["lucky"].actual_points < full["lucky"].actual_points
    # TD luck is unaffected by the reception rate.
    assert half["lucky"].td_over_expected == pytest.approx(full["lucky"].td_over_expected)


def test_screen_respects_its_thresholds_and_empty_input():
    df = _screen_frame()
    assert op.regression_screen(df, season=2023) == ()
    assert op.regression_screen(df, season=2024, min_games=20) == ()
    assert op.regression_screen(df, season=2024, min_opportunities=1e6) == ()
    assert len(op.regression_screen(df, season=2024, through_week=6)[0:1]) == 1
    assert op.regression_screen(df.head(0)) == ()
    assert op.format_screen(()) == "(no candidates)"


def test_screen_drops_weeks_with_no_play_by_play(caplog):
    df = _screen_frame()
    blanked = df.with_columns(
        [
            pl.when(pl.col("week") == 1).then(None).otherwise(pl.col(c)).alias(c)
            for c in op.BUCKET_COLUMNS
        ]
    )
    with caplog.at_level("WARNING"):
        out = op.regression_screen(blanked, season=2024)
    assert "no play-by-play" in caplog.text
    assert all(c.games == 11 for c in out)  # week 1 dropped, not counted as zero usage


def test_format_screen_renders_both_lists():
    text = op.format_screen(op.regression_screen(_screen_frame(), season=2024), top=2)
    assert "SELL HIGH" in text and "BUY LOW" in text
    assert "lucky" in text and "cursed" in text


def test_format_screen_never_lists_a_player_on_both_sides():
    """With fewer than 2*top candidates the two blocks used to overlap.

    Three players and top=10 printed all three as sell-highs and then all three
    again, reversed, as buy-lows -- so "cursed" appeared as a sell-high.
    """
    candidates = op.regression_screen(_screen_frame(), season=2024)
    assert len(candidates) == 3
    text = op.format_screen(candidates, top=10)
    for c in candidates:
        assert text.count(c.name) == 1, c.name
    sell, buy = text.split("BUY LOW")
    assert "lucky" in sell and "lucky" not in buy
    assert "cursed" in buy and "cursed" not in sell
    # top=1 keeps one on each side and drops the middle.
    one = op.format_screen(candidates, top=1)
    assert one.count("lucky") == 1 and one.count("cursed") == 1
    assert "normal" not in one


# --- graceful degradation -------------------------------------------------------


def test_seasons_tuple_rejects_a_string():
    with pytest.raises(TypeError, match="not"):
        op._seasons_tuple("2026")
    with pytest.raises(ValueError, match="no seasons"):
        op._seasons_tuple([])
    assert op._seasons_tuple(2024) == (2024,)


def test_missing_play_by_play_raises_by_default(tmp_path):
    cache = nv.NflverseCache(tmp_path, offline=True)
    with pytest.raises(nv.NflverseError):
        op.field_position_usage(2099, cache=cache)


def test_missing_play_by_play_is_reported_not_guessed(tmp_path, caplog):
    """allow_missing must still refuse when *nothing* loaded -- silence would lie."""
    cache = nv.NflverseCache(tmp_path, offline=True)
    with caplog.at_level("WARNING"), pytest.raises(nv.NflverseError):
        op.field_position_usage([2098, 2099], cache=cache, allow_missing=True)


def test_usage_weeks_survives_a_season_without_play_by_play(monkeypatch):
    """A missing pbp file must null the bucket columns, never zero them."""
    weekly = pl.DataFrame(
        {
            "season": pl.Series([2026, 2026], dtype=pl.Int32),
            "week": pl.Series([1, 1], dtype=pl.Int32),
            "player_id": ["a", "b"],
            "player_display_name": ["A", "B"],
            "position": ["WR", "RB"],
            "team": ["AAA", "AAA"],
            "opponent_team": ["BBB", "BBB"],
            "targets": [8.0, 2.0],
            "receptions": [6.0, 2.0],
            "receiving_yards": [70.0, 12.0],
            "receiving_air_yards": [80.0, 2.0],
            "receiving_yards_after_catch": [20.0, 10.0],
            "receiving_tds": [1.0, 0.0],
            "receiving_fumbles_lost": [0.0, 0.0],
            "carries": [0.0, 14.0],
            "rushing_yards": [0.0, 61.0],
            "rushing_tds": [0.0, 1.0],
            "rushing_fumbles_lost": [0.0, 0.0],
            "target_share": [0.3, 0.08],
            "air_yards_share": [0.4, 0.01],
            "racr": [0.9, 1.2],
            "pacr": [0.8, 0.8],
            "fantasy_points_ppr": [19.0, 12.1],
        }
    )
    monkeypatch.setattr(nv, "player_week_stats", lambda *a, **k: weekly)
    monkeypatch.setattr(op.nv, "player_week_stats", lambda *a, **k: weekly)

    def no_pbp(*_a, **_k):
        raise nv.NflverseNotFound("play_by_play_2026.parquet does not exist")

    monkeypatch.setattr(op, "_pbp_paths", no_pbp)
    monkeypatch.setattr(
        op,
        "_add_snap_share",
        lambda df, *a, **k: df.with_columns(snap_share=pl.lit(None, dtype=pl.Float64)),
    )

    with pytest.raises(nv.NflverseNotFound):
        op.usage_weeks(2026)

    out = op.usage_weeks(2026, allow_missing=True)
    assert out.height == 2
    assert out["carries_goal_line"].null_count() == 2
    assert out["rz_touch_share"].null_count() == 2
    assert out["croe"].null_count() == 2
    # The metrics that do not need play-by-play still work.
    assert out["target_share"].to_list() == [0.3, 0.08]
    assert out["wopr"].to_list() == pytest.approx([0.73, 0.127])
    assert out["catch_rate"].to_list() == pytest.approx([0.75, 1.0])
    with pytest.raises(ValueError, match="entirely null"):
        op.expected_points(out)


def test_route_participation_is_null_without_a_routes_frame():
    df = _frame([_week_row("p", 2024, 1, targets=5.0)])
    assert df["route_participation"].null_count() == 1


def test_route_participation_is_computed_when_supplied():
    base = (
        pl.DataFrame(
            [
                _week_row("p", 2024, 1, targets=5.0),
                _week_row("q", 2024, 1, targets=2.0),
            ]
        )
        .with_columns(pl.col("season", "week").cast(pl.Int32))
        .drop("routes", "route_participation")
    )
    routes = pl.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "player_id": ["p"],
            "routes": [28.0],
            "team_dropbacks": [35.0],
        }
    )
    out = op._add_route_participation(base, routes)
    assert out.filter(pl.col("player_id") == "p")["route_participation"][0] == pytest.approx(0.8)
    assert out.filter(pl.col("player_id") == "q")["route_participation"][0] is None
    with pytest.raises(ValueError, match="missing"):
        op._add_route_participation(base, routes.drop("routes"))


def test_season_usage_reforms_shares_rather_than_averaging_them():
    """A 20-target game and a 2-target game are not equal evidence."""
    rows = [
        _week_row("p", 2024, 1, targets=20.0, receptions=14.0, team_targets=40.0),
        _week_row("p", 2024, 2, targets=2.0, receptions=2.0, team_targets=4.0),
    ]
    out = op.season_usage(_frame(rows))
    assert out["targets_per_game"][0] == pytest.approx(11.0)
    assert out["games"][0] == 2
    # Catch rate is 14/20 in week 1 and 2/2 in week 2. Averaging the weeks gives
    # 0.85; re-forming from the totals gives 16/22 = 0.727, which is the true rate.
    assert out["catch_rate"][0] == pytest.approx(16.0 / 22.0)
    assert out["catch_rate"][0] != pytest.approx(0.85)


def test_season_usage_uses_the_team_denominator_not_the_players_own_volume():
    """The share must be sum(his)/sum(team's), never his weekly shares averaged.

    Weighting weekly shares by the player's OWN targets looks like the same thing
    and is not, because his targets are the share's own numerator -- it over-weights
    exactly the weeks his share was high. On real 2019-2025 data the wrong one runs
    +16% on target share and +22% on air-yards share.
    """
    rows = [
        # 8 of 40 in a pass-happy week (0.20) and 14 of 20 in a quiet one (0.70).
        _week_row("p", 2024, 1, targets=8.0, team_targets=40.0, team_air_yards=400.0)
        | {"receiving_air_yards": 40.0},
        _week_row("p", 2024, 2, targets=14.0, team_targets=20.0, team_air_yards=100.0)
        | {"receiving_air_yards": 60.0},
    ]
    out = op.season_usage(_frame(rows))
    # 22 of the team's 60 = 0.3667, not (0.20*8 + 0.70*14)/22 = 0.5182.
    assert out["target_share"][0] == pytest.approx(22.0 / 60.0)
    assert out["target_share"][0] != pytest.approx((0.20 * 8 + 0.70 * 14) / 22.0)
    # 100 air yards of the team's 500 = 0.20, not (0.20*40 + 0.70*60)/100 = 0.50.
    assert out["air_yards_share"][0] == pytest.approx(100.0 / 500.0)
    assert out["air_yards_share"][0] != pytest.approx((0.20 * 40 + 0.70 * 60) / 100.0)


def test_season_usage_nulls_air_yards_share_without_a_team_denominator(caplog):
    """No team_air_yards column must mean "not measured", not a biased stand-in."""
    df = _frame([_week_row("p", 2024, 1, targets=8.0)]).drop("team_air_yards")
    with caplog.at_level("WARNING"):
        out = op.season_usage(df)
    assert "team_air_yards" in caplog.text
    assert out["air_yards_share"][0] is None
    assert out["target_share"][0] is not None


# --- live re-measurement --------------------------------------------------------


@pytest.fixture(scope="module")
def live_usage() -> pl.DataFrame:
    try:
        return op.usage_weeks(range(2019, 2026))
    except nv.NflverseError as exc:  # pragma: no cover - network shape
        pytest.skip(f"nflverse unavailable: {exc}")


@pytest.mark.network
def test_live_wopr_reproduces_nflverses_own_column(live_usage):
    """Our canonical WOPR must equal the one nflverse ships, to the bit."""
    raw = nv.player_week_stats(range(2019, 2026), season_type="REG").select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "player_id",
        pl.col("wopr").alias("nflverse_wopr"),
    )
    j = live_usage.join(raw, on=["season", "week", "player_id"], how="inner").drop_nulls(
        "nflverse_wopr"
    )
    assert j.height > 30_000
    assert (j["wopr"] - j["nflverse_wopr"]).abs().max() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.network
def test_live_wopr_refit_matches_the_baked_in_constants(live_usage):
    for reception_points, want in (
        (1.0, op.REFIT_WOPR_PPR),
        (0.5, op.REFIT_WOPR_HALF_PPR),
        (0.0, op.REFIT_WOPR_STANDARD),
    ):
        fit = op.fit_wopr_weights(live_usage, reception_points=reception_points)
        assert fit.weights.target_share == pytest.approx(want.target_share, abs=0.01)
        assert fit.weights.air_yards_share == pytest.approx(want.air_yards_share, abs=0.01)
    # And the headline claim: in PPR the refit ratio is nothing like 1.5/0.7.
    assert op.REFIT_WOPR_PPR.ratio > 5 * op.CANONICAL_WOPR.ratio


@pytest.mark.network
def test_live_conversion_rates_match_the_baked_in_table():
    refit = op.fit_conversion_rates(range(2019, 2026))
    assert set(refit.cells) == set(op.DEFAULT_CONVERSION_RATES.cells)
    for key, cell in op.DEFAULT_CONVERSION_RATES.cells.items():
        other = refit.cells[key]
        assert other.n == cell.n, key
        for field in ("catch_rate", "yards", "td_rate", "fumble_lost_rate"):
            assert getattr(other, field) == pytest.approx(getattr(cell, field), abs=1e-4), key


@pytest.mark.network
def test_live_stability_table_reproduces_the_documented_split(live_usage):
    rows = {r.metric: r for r in op.stability_table(live_usage)}
    documented = {
        "carry_share": 0.875,
        "snap_share": 0.758,
        "wopr": 0.628,
        "air_yards_share": 0.619,
        "target_share": 0.567,
        "rz_touch_share": 0.342,
        "i10_touch_share": 0.263,
        "yac_per_reception": 0.229,
        "catch_rate": 0.120,
        "td_rate": 0.050,
        "ypc": 0.043,
        "croe": 0.035,
    }
    for metric, lag1 in documented.items():
        assert rows[metric].lag1 == pytest.approx(lag1, abs=0.01), metric
    # The split itself, which is the claim that matters.
    assert rows["target_share"].stable and rows["wopr"].stable
    assert not rows["td_rate"].stable and not rows["ypc"].stable

    # The reliability column, uncorrected. These are the numbers a Spearman-Brown
    # step would inflate to 0.824 / 0.894 / 0.145, so pin them.
    reliability_3 = {
        "carry_share": 0.808,
        "snap_share": 0.778,
        "wopr": 0.767,
        "air_yards_share": 0.791,
        "target_share": 0.701,
        "rz_touch_share": 0.498,
        "i10_touch_share": 0.409,
        "yac_per_reception": 0.459,
        "catch_rate": 0.307,
        "td_rate": 0.078,
    }
    for metric, want in reliability_3.items():
        assert rows[metric].split_half_3 == pytest.approx(want, abs=0.01), metric
    assert all(r <= 1.0 for _, r in rows["target_share"].reliability_curve)

    # Target share clears 0.70 at exactly 3 games; the folklore survives without
    # needing a correction to help it over the line.
    assert rows["target_share"].games_to_stabilize == 3
    assert rows["air_yards_share"].games_to_stabilize == 2
    assert rows["carry_share"].games_to_stabilize == 1
    assert rows["td_rate"].games_to_stabilize is None
    # Red-zone share is close but does not get there inside a season.
    assert rows["rz_touch_share"].games_to_stabilize is None
    assert dict(rows["rz_touch_share"].reliability_curve)[8] == pytest.approx(0.699, abs=0.01)


@pytest.mark.network
def test_live_screen_flags_the_kupp_2019_archetype(live_usage):
    out = op.regression_screen(live_usage, season=2019)
    kupp = next(c for c in out if c.name == "Cooper Kupp")
    assert kupp.touchdowns == 10.0
    # Our bucket model says 7.6; ffopportunity's per-play model says 6.15 and the
    # archetype quotes ~5.6. All three agree he banked TDs he had not earned.
    assert 5.0 < kupp.expected_touchdowns < 8.5
    assert kupp.driver == "touchdowns"
    assert kupp.verdict == "sell_high"
    assert kupp.significant
    # And he is in the top slice of the league, not merely positive.
    assert out.index(kupp) < 40


@pytest.mark.network
def test_live_screen_gap_mean_reverts(live_usage):
    """The claim the screen rests on: the gap mostly does not repeat."""
    pairs = []
    for season in range(2019, 2025):
        current = {
            c.player_id: c
            for c in op.regression_screen(
                live_usage, season=season, min_games=8, min_opportunities=60
            )
        }
        nxt = {
            c.player_id: c
            for c in op.regression_screen(
                live_usage, season=season + 1, min_games=8, min_opportunities=60
            )
        }
        pairs.extend(
            (
                c.gap_per_game,
                nxt[pid].gap_per_game,
                c.unrepeatable_per_game,
                nxt[pid].unrepeatable_per_game,
            )
            for pid, c in current.items()
            if pid in nxt
        )
    a = np.array(pairs)
    assert len(a) > 500
    gap_r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    luck_r = float(np.corrcoef(a[:, 2], a[:, 3])[0, 1])
    assert 0.15 < gap_r < 0.40  # documented 0.28
    assert luck_r < 0.20  # documented 0.09 -- the TD slice is the noise
    assert luck_r < gap_r
    # Sell-highs decline, buy-lows improve.
    sell = a[a[:, 0] >= 1.5]
    buy = a[a[:, 0] <= -1.5]
    assert sell[:, 1].mean() < sell[:, 0].mean() / 2
    assert buy[:, 1].mean() > buy[:, 0].mean() / 2


@pytest.mark.network
def test_live_pbp_targets_reconcile_with_the_weekly_file():
    """The check that says our play-by-play filtering is right."""
    seasons = (2023, 2024)
    fp = op.field_position_usage(seasons)
    ours = fp.group_by("season", "player_id").agg(
        sum(pl.col(f"targets_{b}") for b in op.FIELD_BUCKETS).sum().alias("pbp_targets")
    )
    theirs = (
        nv.player_week_stats(seasons, season_type="REG")
        .group_by(pl.col("season").cast(pl.Int32), "player_id")
        .agg(pl.col("targets").sum().alias("weekly_targets"))
    )
    j = ours.join(theirs, on=["season", "player_id"], how="inner")
    assert j.height > 1000
    assert (j["pbp_targets"] - j["weekly_targets"]).abs().max() == 0


@pytest.mark.network
def test_live_ffopportunity_fumble_correction(tmp_path):
    """Their expected side has no fumble term; ours puts one back."""
    try:
        ff = op.ffopportunity_weekly(2024, root=tmp_path)
    except (nv.NflverseError, OSError) as exc:  # pragma: no cover - network shape
        pytest.skip(f"ffopportunity unavailable: {exc}")

    fumbles = ff["rec_fumble_lost"].fill_null(0) + ff["rush_fumble_lost"].fill_null(0)
    # Their actual points are full PPR minus 2 per fumble; verify on the clean rows.
    clean = ff.filter(fumbles == 0).filter(pl.col("position").is_in(["RB", "WR", "TE"]))
    calc = (
        clean["receptions"]
        + 0.1 * (clean["rec_yards_gained"] + clean["rush_yards_gained"])
        + 6.0 * (clean["rec_touchdown"] + clean["rush_touchdown"])
        + 2.0 * (clean["rec_two_point_conv"] + clean["rush_two_point_conv"])
    )
    actual = clean["rec_fantasy_points"] + clean["rush_fantasy_points"]
    assert (actual - calc).abs().max() == pytest.approx(0.0, abs=1e-9)

    # The correction belongs on the EXPECTED side: it adds an expected fumble cost
    # for everyone in proportion to opportunity, and never touches the actual side.
    opportunities = ff["rec_attempt"].fill_null(0) + ff["rush_attempt"].fill_null(0)
    delta = ff["total_fantasy_points_diff_adj"] - ff["total_fantasy_points_diff"]
    want = 2.0 * opportunities * op.FUMBLE_LOST_PER_OPPORTUNITY
    assert (delta - want).abs().max() == pytest.approx(0.0, abs=1e-9)
    assert delta.min() >= 0.0
    assert delta.filter(fumbles > 0).mean() == pytest.approx(
        delta.filter(fumbles == 0).mean(), abs=0.3
    )

    # It removes the mean bias without making fumbling free. A back who fumbled is
    # still behind expectation afterwards -- roughly -2 per fumble, which is the
    # whole point of charging it. The tempting wrong form,
    # `diff + 2*(fumbles - opp*rate)`, zeroes that out.
    adj = ff["total_fantasy_points_diff_adj"]
    assert adj.filter(fumbles == 0).mean() == pytest.approx(0.0, abs=0.15)
    assert adj.filter(fumbles > 0).mean() < -1.5
    wrong = ff["total_fantasy_points_diff"] + 2.0 * (fumbles - opportunities * 0.0042)
    assert wrong.filter(fumbles > 0).mean() > -0.5  # fumbling would cost nothing
