"""Calibration: the mixture algebra, the fitted curves, and the held-out acceptance test.

Everything except the two corpus-gated tests runs offline. The corpus-gated pair is
the point of the module, though: `test_calibration_improves_held_out_mae` is the
acceptance test, and `test_reproduces_research_constants` is the check that we are
reading the same 23,999 player-weeks the published constants came from.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fantasy_quant.core import QB, RB, SKILL_POSITIONS, TE, WR
from fantasy_quant.projections import calibration as cal

# Anchored to the repo, NOT to the working directory. A relative path here means the
# six corpus-gated tests -- the acceptance test among them -- silently skip whenever
# pytest is invoked from anywhere but the repo root, and the run still reports green.
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "data" / "snapshots" / "espn"
HAS_CORPUS = bool(sorted(CORPUS.glob("season=*/variant=ppr/*.parquet")))
needs_corpus = pytest.mark.skipif(not HAS_CORPUS, reason="no ESPN Parquet corpus present")

#: The seasons the published constants and `_DEFAULT_ROWS` were fitted on. Pinned
#: explicitly: the corpus is append-only and this project backfills forward, so a
#: fixture that reads every season on disk stops reproducing 23,999 rows the moment
#: the first 2026 actuals land (measured: 24,428 after one played week).
PINNED_SEASONS = (2022, 2023, 2024, 2025)


# --------------------------------------------------------------------------------------
# The mixture-moment round trip -- the trap the whole module exists to avoid
# --------------------------------------------------------------------------------------

MU_GRID = (0.05, 0.5, 1.0, 2.5, 5.0, 8.0, 12.0, 18.0, 25.0, 40.0)
SD_GRID = (0.5, 1.0, 2.0, 4.0, 7.0, 12.0, 20.0)
P_GRID = (0.0, 0.01, 0.05, 0.185, 0.259, 0.325, 0.5, 0.75, 0.9, 0.98)


def test_mixture_moments_round_trip_across_the_grid():
    """(mu, sd, p) -> gamma -> (mu, sd) must be the identity wherever it is feasible.

    Get this wrong and every simulated score is biased, silently and everywhere.
    """
    checked = 0
    for mu in MU_GRID:
        for p in P_GRID:
            floor = cal.minimum_sd(mu, p)
            for sd in SD_GRID:
                # Infeasible (covered by its own test below), and at the floor itself
                # the positive part is degenerate and the shape cap rounds it.
                if sd <= floor * (1.0 + 1e-6):
                    continue
                g = cal.hurdle_gamma_from_moments(mu, sd, p)
                got_mu, got_sd = g.moments()
                assert got_mu == pytest.approx(mu, rel=1e-9, abs=1e-9)
                assert got_sd == pytest.approx(sd, rel=1e-9, abs=1e-9)
                assert g.shape > 0.0
                assert g.scale > 0.0
                checked += 1
    assert checked > 300


def test_both_mixture_mistakes_are_pinned():
    """The two quiet ways to get the hurdle algebra wrong, at a typical WR's numbers.

    Mistake 1 -- fit the gamma straight to (mean, sd), then bolt `p_zero` on: the
    mixture mean lands at (1-p)*mean, 25.9% low.
    Mistake 2 -- scale `m_pos` correctly but drop the between-component variance: the
    mixture SD lands 26% high.
    """
    mean, sd, p = 9.0, 7.0, 0.259
    q = 1.0 - p

    correct = cal.hurdle_gamma_from_moments(mean, sd, p)
    assert correct.moments() == pytest.approx((mean, sd))

    # Mistake 1: the gamma is the whole distribution, p_zero is an afterthought.
    forgot_scaling = cal.HurdleGamma(p_zero=p, shape=mean**2 / sd**2, scale=sd**2 / mean)
    bad_mean, _ = forgot_scaling.moments()
    assert bad_mean == pytest.approx(q * mean)
    assert bad_mean / mean == pytest.approx(0.741, abs=1e-3)

    # Mistake 2: right mean, but the variance forgets p*(1-p)*m_pos^2.
    m_pos = mean / q
    v_pos = sd * sd / q
    forgot_between = cal.HurdleGamma(p_zero=p, shape=m_pos**2 / v_pos, scale=v_pos / m_pos)
    ok_mean, bad_sd = forgot_between.moments()
    assert ok_mean == pytest.approx(mean)
    assert bad_sd == pytest.approx(8.79, abs=0.01)
    assert bad_sd / sd == pytest.approx(1.256, abs=1e-3)


def test_moments_agree_with_simulation():
    """Closed form vs 400k draws. The algebra is only right if the sampler agrees."""
    rng = np.random.default_rng(20260907)
    for mean, sd, p in ((6.0, 6.5, 0.30), (12.0, 8.0, 0.05), (2.0, 4.0, 0.6)):
        g = cal.hurdle_gamma_from_moments(mean, sd, p)
        draws = rng.gamma(g.shape, g.scale, 400_000)
        draws[rng.random(draws.size) < g.p_zero] = 0.0
        assert draws.mean() == pytest.approx(mean, abs=0.05)
        assert draws.std() == pytest.approx(sd, abs=0.05)


def test_infeasible_sd_is_widened_not_mis_solved():
    """Below `minimum_sd` the mean is preserved and the SD comes back larger, not wrong."""
    mean, p = 10.0, 0.5
    floor = cal.minimum_sd(mean, p)
    assert floor == pytest.approx(10.0)  # m_pos = 20, sd_min = 10*sqrt(.5/.5)
    g = cal.hurdle_gamma_from_moments(mean, sd=1.0, p_zero=p)
    got_mu, got_sd = g.moments()
    assert got_mu == pytest.approx(mean, rel=1e-9)
    assert got_sd == pytest.approx(floor, rel=1e-5)


def test_degenerate_inputs():
    assert cal.hurdle_gamma_from_moments(0.0, 5.0, 0.2).moments() == (0.0, 0.0)
    assert cal.hurdle_gamma_from_moments(-3.0, 5.0, 0.2).p_zero == 1.0
    # p_zero of exactly 1 with a positive mean is contradictory; the cap keeps the
    # mean and makes the blank near-certain rather than raising in a hot loop.
    g = cal.hurdle_gamma_from_moments(8.0, 400.0, 1.0)
    assert g.p_zero == cal.MAX_P_ZERO
    assert g.moments()[0] == pytest.approx(8.0)
    with pytest.raises(ValueError):
        cal.hurdle_gamma_from_moments(5.0, 3.0, 1.5)
    with pytest.raises(ValueError):
        cal.hurdle_gamma_from_moments(5.0, -1.0, 0.2)


def test_mixture_moments_helper_matches_the_object():
    g = cal.hurdle_gamma_from_moments(7.0, 6.0, 0.2)
    assert cal.mixture_moments(g.p_zero, g.shape, g.scale) == g.moments()


# --------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------


def test_hurdle_probability_rises_as_the_projection_falls():
    """A 3-point projection blanks far more often than an 18-point one, at every position."""
    fitted = cal.default_calibration()
    for pid in SKILL_POSITIONS:
        probs = [fitted.p_zero(mu, pid) for mu in (1.0, 3.0, 6.0, 10.0, 15.0, 20.0)]
        assert all(a > b for a, b in zip(probs[:-1], probs[1:], strict=True)), (
            f"position {pid} hurdle is not decreasing in mu: {probs}"
        )
        assert all(0.0 < p < 1.0 for p in probs)
    # And the magnitude matches what the corpus shows: a barely-projected WR is a
    # coin flip to blank; a WR1 almost never is.
    assert fitted.p_zero(1.0, WR) > 0.55
    assert fitted.p_zero(17.0, WR) < 0.05


def test_hurdle_is_clamped_outside_its_fitted_support():
    """No data out there, so no confident extrapolation either."""
    fitted = cal.default_calibration()
    pc = fitted.for_position(RB)
    assert pc.hurdle(-50.0) == pytest.approx(pc.hurdle(pc.hurdle.mu_lo))
    assert pc.hurdle(1_000.0) == pytest.approx(pc.hurdle(pc.hurdle.mu_hi))
    assert 0.0 < pc.hurdle(1_000.0) < 1.0


def test_unfittable_hurdle_falls_back_to_the_positional_constant():
    curve = cal.HurdleCurve.constant(0.185)
    assert curve(0.0) == 0.185
    assert curve(30.0) == 0.185


def test_spread_rises_with_the_mean_and_respects_its_floor():
    fitted = cal.default_calibration()
    for pid in SKILL_POSITIONS:
        assert fitted.sd(20.0, pid) > fitted.sd(5.0, pid) > fitted.sd(0.0, pid) >= cal.SD_FLOOR
    # QB is genuinely flatter than the skill positions; the pooled line is wrong for him.
    qb, wr = fitted.for_position(QB).spread, fitted.for_position(WR).spread
    assert qb.slope < 0.5 * wr.slope
    assert qb.intercept > 1.5 * wr.intercept


def test_source_spread_widens_in_quadrature_and_is_opt_in():
    fitted = cal.default_calibration()
    base = fitted.sd(10.0, WR)
    assert fitted.sd(10.0, WR, source_spread=None) == base
    assert fitted.sd(10.0, WR, source_spread=0.0) == base
    assert fitted.sd(10.0, WR, source_spread=3.0) == pytest.approx(math.hypot(base, 3.0))


def test_level_correction_shrinks_high_projections_and_lifts_low_ones():
    fitted = cal.default_calibration()
    for pid in SKILL_POSITIONS:
        assert fitted.calibrate(20.0, pid) < 20.0
        assert fitted.calibrate(0.0, pid) > 0.0
        assert fitted.calibrate(30.0, pid) > fitted.calibrate(10.0, pid)
    assert fitted.calibrate(-5.0, WR) == 0.0


def test_unknown_positions_fall_through_to_pooled():
    fitted = cal.default_calibration()
    assert fitted.for_position(16) is fitted.pooled  # D/ST
    assert fitted.for_position(5) is fitted.pooled  # K


# --------------------------------------------------------------------------------------
# Outlooks
# --------------------------------------------------------------------------------------


def test_outlook_moments_are_the_full_mixture_moments():
    fitted = cal.default_calibration()
    for pid in SKILL_POSITIONS:
        for projection in (0.4, 3.0, 9.0, 16.0, 24.0):
            o = fitted.outlook(
                player_id=1,
                season=2026,
                week=3,
                position_id=pid,
                projection=projection,
                pro_team_id=12,
            )
            got_mean, got_sd = cal.mixture_moments(o.p_zero, o.shape, o.scale)
            assert o.mean == pytest.approx(got_mean, rel=1e-9)
            assert o.sd == pytest.approx(got_sd, rel=1e-9)
            assert o.mean == pytest.approx(fitted.calibrate(projection, pid), rel=1e-9)
            assert o.sd > 0.0
            assert o.pro_team_id == 12
            assert o.playing


def test_outlook_from_mean_does_not_re_apply_the_level_correction():
    fitted = cal.default_calibration()
    raw = fitted.outlook(player_id=1, season=2026, week=1, position_id=RB, projection=12.0)
    direct = fitted.outlook_from_mean(player_id=1, season=2026, week=1, position_id=RB, mean=12.0)
    assert raw.mean < 12.0
    assert direct.mean == pytest.approx(12.0)


def test_not_playing_is_a_zeroed_outlook():
    fitted = cal.default_calibration()
    o = fitted.outlook(
        player_id=1, season=2026, week=7, position_id=WR, projection=14.0, playing=False
    )
    assert (o.mean, o.sd, o.p_zero, o.playing) == (0.0, 0.0, 1.0, False)


def test_an_infeasible_spread_widens_the_outlook_rather_than_mis_solving_it():
    """A curve set whose sigma(mu) is tighter than its own hurdle allows.

    The shipped curves never reach this -- sigma(mu) sits well above `minimum_sd`
    everywhere a real player lives -- so nothing else in this file covers the path,
    and `outlook_from_mean` is the ensemble's entry point with an arbitrary mean.
    The mean must be preserved exactly and the SD must come back at the floor.
    """
    tight = cal.PositionCalibration(
        position_id=WR,
        level=cal.LevelLine(intercept=0.0, slope=1.0, r2=0.0, n=1000),
        hurdle=cal.HurdleCurve.constant(0.5),
        spread=cal.SpreadLine(intercept=1.0, slope=0.0, floor=cal.SD_FLOOR, n=1000),
        n=1000,
    )
    fitted = cal.CalibrationSet(
        variant="ppr",
        seasons=(2024,),
        fitted_at="x",
        n_pairs=1000,
        source="test",
        positions={WR: tight},
        pooled=tight,
    )
    for mu in (2.0, 10.0, 25.0):
        floor = cal.minimum_sd(mu, 0.5)
        assert fitted.sd(mu, WR) < floor  # the request really is infeasible
        o = fitted.outlook_from_mean(player_id=1, season=2026, week=1, position_id=WR, mean=mu)
        assert o.mean == pytest.approx(mu, rel=1e-9)
        assert o.sd == pytest.approx(floor, rel=1e-5)
        assert o.sd >= floor  # widened, never narrowed below what is achievable
        assert cal.mixture_moments(o.p_zero, o.shape, o.scale) == pytest.approx((o.mean, o.sd))


def test_outlook_of_a_near_zero_projection_stays_valid():
    """The low end is where an infeasible (mu, sd, p) triple would blow up."""
    fitted = cal.default_calibration()
    for pid in SKILL_POSITIONS:
        o = fitted.outlook(player_id=1, season=2026, week=1, position_id=pid, projection=0.01)
        assert 0.0 <= o.p_zero <= 1.0
        assert o.sd >= 0.0
        assert math.isfinite(o.shape) and math.isfinite(o.scale)


# --------------------------------------------------------------------------------------
# Persistence and graceful degradation
# --------------------------------------------------------------------------------------


def test_params_round_trip_through_json(tmp_path: Path):
    original = cal.default_calibration()
    path = original.save(cal.params_path("ppr", tmp_path))
    assert path.exists()

    restored = cal.load("ppr", tmp_path)
    assert restored.source == original.source
    assert restored.seasons == original.seasons
    assert restored.n_pairs == original.n_pairs
    assert restored.to_dict() == original.to_dict()
    for pid in SKILL_POSITIONS:
        assert restored.for_position(pid) == original.for_position(pid)
    assert restored.pooled == original.pooled


def test_load_falls_back_to_defaults_when_no_file(tmp_path: Path):
    loaded = cal.load("ppr", tmp_path / "nothing-here")
    assert loaded.source == "defaults"
    assert loaded.n_pairs == 23999


def test_load_falls_back_on_a_corrupt_or_stale_file(tmp_path: Path):
    path = cal.params_path("ppr", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert cal.load("ppr", tmp_path).source == "defaults"

    payload = cal.default_calibration().to_dict()
    payload["schema_version"] = cal.SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))
    assert cal.load("ppr", tmp_path).source == "defaults"


@pytest.mark.parametrize(
    "text",
    [
        "{not json",  # not JSON at all
        "",  # truncated to nothing
        "null",  # valid JSON, not an object
        "[1, 2, 3]",  # valid JSON, not an object
        "42",
        '"ppr"',
        '{"schema_version": 1, "variant": "ppr", "seasons": [], "fitted_at": "x", "n_pairs": 0}',
        '{"schema_version": 1, "positions": [1, 2], "variant": "ppr", "seasons": [],'
        ' "fitted_at": "x", "n_pairs": 0, "pooled": {}}',
    ],
)
def test_load_never_raises_on_a_malformed_file(tmp_path: Path, text: str):
    """`load` promises a logged warning and the defaults, never an exception.

    A half-written or hand-edited file is often valid JSON of the wrong SHAPE, which
    is a different code path from unparseable bytes and used to escape as an
    AttributeError from `raw.get`.
    """
    path = cal.params_path("ppr", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    loaded = cal.load("ppr", tmp_path)
    assert loaded.source == "defaults"
    assert loaded.for_position(WR).level.slope == pytest.approx(0.9434)


def test_missing_corpus_degrades_gracefully(tmp_path: Path):
    """No Parquet anywhere: empty pairs, shipped defaults, no exception."""
    pairs = cal.load_pairs(root=tmp_path)
    assert pairs.height == 0
    assert list(pairs.columns) == list(cal.PAIR_SCHEMA)

    fitted = cal.fit(root=tmp_path, write=False, directory=tmp_path)
    assert fitted.source == "defaults"
    assert fitted.for_position(WR).level.slope == pytest.approx(0.9434)

    with pytest.raises(ValueError):
        cal.calibration_report(pairs=pairs)


def test_report_refuses_a_leaky_split():
    """Fitting and evaluating on the same season is the easiest way to fake a win."""
    pairs = pl.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["a", "b"],
            "position_id": [WR, WR],
            "season": [2024, 2025],
            "week": [1, 1],
            "projection": [10.0, 11.0],
            "actual": [9.0, 12.0],
        }
    )
    with pytest.raises(ValueError, match="leak"):
        cal.calibration_report(pairs, train_seasons=(2024, 2025), test_seasons=(2025,))


# --------------------------------------------------------------------------------------
# Corpus filters
# --------------------------------------------------------------------------------------


def _row(**kw):
    base = {
        "espn_id": 1,
        "full_name": "Test Player",
        "default_position_id": WR,
        "stat_season": 2026,
        "request_season": 2026,
        "stat_source_id": 1,
        "stat_split_type_id": 1,
        "scoring_period_id": 1,
        "applied_total": 10.0,
        "captured_at": dt.datetime(2026, 9, 1),
    }
    return {**base, **kw}


def _write_corpus(root: Path, rows: list[dict], season: int = 2026) -> None:
    out = root / f"season={season}" / "variant=ppr"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out / f"{season}-09-01.parquet")


def test_load_pairs_drops_the_mixed_season_rows(tmp_path: Path):
    """A 2026 request returns 2025 rows in the same array; they must not be paired."""
    _write_corpus(
        tmp_path,
        [
            _row(stat_source_id=1, applied_total=10.0),
            _row(stat_source_id=0, applied_total=7.0),
            # Same player/week, but last season's rows riding along in the 2026 file.
            _row(stat_season=2025, stat_source_id=1, applied_total=99.0),
            _row(stat_season=2025, stat_source_id=0, applied_total=99.0),
        ],
    )
    pairs = cal.load_pairs(root=tmp_path)
    assert pairs.height == 1
    assert pairs["projection"][0] == 10.0
    assert pairs["actual"][0] == 7.0


def test_load_pairs_drops_non_weekly_rows_and_keeps_the_latest_capture(tmp_path: Path):
    _write_corpus(
        tmp_path,
        [
            # Season totals and the ROS rate are not player-weeks.
            _row(stat_split_type_id=0, scoring_period_id=0, applied_total=300.0),
            _row(stat_split_type_id=2, scoring_period_id=0, stat_source_id=0),
            # The same week captured twice: only the later projection survives, so a
            # season present under two capture dates is not double-weighted.
            _row(applied_total=8.0, captured_at=dt.datetime(2026, 9, 1)),
            _row(applied_total=12.0, captured_at=dt.datetime(2026, 9, 5)),
            _row(stat_source_id=0, applied_total=6.0),
        ],
    )
    pairs = cal.load_pairs(root=tmp_path)
    assert pairs.height == 1
    assert pairs["projection"][0] == 12.0


def test_load_pairs_applies_the_positive_projection_filter(tmp_path: Path):
    _write_corpus(
        tmp_path,
        [
            _row(applied_total=0.0),
            _row(stat_source_id=0, applied_total=0.0),
            _row(scoring_period_id=2, applied_total=4.0),
            _row(scoring_period_id=2, stat_source_id=0, applied_total=3.0),
        ],
    )
    assert cal.load_pairs(root=tmp_path).height == 1
    assert cal.load_pairs(root=tmp_path, min_projection=-1.0).height == 2


# --------------------------------------------------------------------------------------
# The fitter, on data whose truth we know
# --------------------------------------------------------------------------------------


def test_fit_recovers_known_parameters_from_synthetic_hurdle_gamma_data():
    """Simulate a corpus with known level/hurdle/spread and check the fitter finds them."""
    rng = np.random.default_rng(7)
    n = 60_000
    proj = rng.uniform(0.5, 22.0, n)
    true_a, true_b = 0.30, 0.92
    true_alpha, true_beta = 2.4, -1.9
    true_c, true_m = 2.5, 0.38

    mu = true_a + true_b * proj
    p = 1.0 / (1.0 + np.exp(-(true_alpha + true_beta * np.sqrt(mu))))
    sd = true_c + true_m * mu
    actual = np.empty(n)
    for i in range(n):
        g = cal.hurdle_gamma_from_moments(float(mu[i]), float(sd[i]), float(p[i]))
        actual[i] = 0.0 if rng.random() < g.p_zero else rng.gamma(g.shape, g.scale)

    pairs = pl.DataFrame(
        {
            "player_id": np.arange(n),
            "name": ["p"] * n,
            "position_id": np.full(n, RB),
            "season": np.full(n, 2024),
            "week": np.full(n, 1),
            "projection": proj,
            "actual": actual,
        }
    )
    fitted = cal.fit_from_pairs(pairs)
    pc = fitted.for_position(RB)
    assert pc.level.intercept == pytest.approx(true_a, abs=0.15)
    assert pc.level.slope == pytest.approx(true_b, abs=0.02)
    assert pc.hurdle.intercept == pytest.approx(true_alpha, abs=0.15)
    assert pc.hurdle.slope == pytest.approx(true_beta, abs=0.12)
    assert pc.spread.intercept == pytest.approx(true_c, abs=0.35)
    assert pc.spread.slope == pytest.approx(true_m, abs=0.04)


def test_thin_positions_are_left_to_pooled():
    rng = np.random.default_rng(3)
    n = 400
    pairs = pl.DataFrame(
        {
            "player_id": np.arange(n),
            "name": ["p"] * n,
            # 300 RB rows is enough; 100 TE rows is not.
            "position_id": np.where(np.arange(n) < 300, RB, TE),
            "season": np.full(n, 2024),
            "week": np.full(n, 1),
            "projection": rng.uniform(1.0, 20.0, n),
            "actual": rng.uniform(0.0, 30.0, n),
        }
    )
    fitted = cal.fit_from_pairs(pairs)
    assert RB in fitted.positions
    assert TE not in fitted.positions
    assert fitted.for_position(TE) is fitted.pooled


# --------------------------------------------------------------------------------------
# Against the real corpus
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> pl.DataFrame:
    return cal.load_pairs(root=CORPUS, seasons=PINNED_SEASONS)


def test_the_corpus_pin_survives_the_corpus_growing(tmp_path: Path):
    """The pin must key on seasons, not on 'whatever Parquet is on disk today'.

    `fq backfill` appends. An unpinned fixture reproduces 23,999 rows only until the
    first 2026 actuals land, after which `test_reproduces_research_constants` fails
    for a reason that has nothing to do with the calibration.
    """
    for season in (2024, 2026):
        _write_corpus(
            tmp_path,
            [
                _row(stat_season=season, request_season=season, applied_total=10.0),
                _row(
                    stat_season=season,
                    request_season=season,
                    stat_source_id=0,
                    applied_total=7.0,
                ),
            ],
            season=season,
        )
    assert cal.load_pairs(root=tmp_path).height == 2  # unpinned: grows
    assert cal.load_pairs(root=tmp_path, seasons=(2024,)).height == 1  # pinned: stable
    assert cal.load_pairs(root=tmp_path, seasons=(2024,))["season"].to_list() == [2024]


@needs_corpus
def test_the_pinned_corpus_is_exactly_the_seasons_the_defaults_claim(corpus: pl.DataFrame):
    """`default_calibration()` advertises 2022-2025; the fixture must be those seasons."""
    assert tuple(sorted(corpus["season"].unique().to_list())) == PINNED_SEASONS
    assert cal.default_calibration().seasons == PINNED_SEASONS
    assert CORPUS.is_absolute()


@needs_corpus
def test_reproduces_research_constants(corpus: pl.DataFrame):
    """Our pairing must be the same pairing the published constants came from.

    Pairing every weekly row gives 41,970 rows and a QB MAE of 2.37 -- the corpus is
    defined by `projection > 0`, and only that definition lands on the table in
    RESEARCH.md. If this drifts, some other filter changed, not football.
    """
    assert corpus.height == 23_999

    expected = {  # pos: (n, MAE, weekly slope, P(actual <= 0))
        QB: (2210, 5.87, 0.948, 0.032),
        RB: (6312, 4.10, 0.921, 0.185),
        WR: (10004, 4.27, 0.943, 0.259),
        TE: (5473, 3.24, 0.961, 0.325),
    }
    for pid, (n, mae, slope, p_zero) in expected.items():
        s = corpus.filter(pl.col("position_id") == pid)
        p = s["projection"].to_numpy()
        a = s["actual"].to_numpy()
        assert s.height == n
        assert float(np.mean(np.abs(a - p))) == pytest.approx(mae, abs=0.005)
        assert float(np.polyfit(p, a, 1)[0]) == pytest.approx(slope, abs=0.0005)
        assert float(np.mean(a <= 0.0)) == pytest.approx(p_zero, abs=0.0005)


@needs_corpus
def test_calibration_improves_held_out_mae(corpus: pl.DataFrame):
    """ACCEPTANCE TEST. Fit 2022-2024, score 2025. Never the same rows.

    Measured result, stated rather than tuned: MAE improves at QB (-0.28%),
    RB (-1.42%) and WR (-1.49%) and is a wash at TE (+0.11%). ESPN's TE weeklies are
    already mean-calibrated -- the fitted line is 0.211 + 0.961*proj, essentially the
    identity over the range TEs occupy -- and MAE is minimised by the conditional
    MEDIAN, which under this much right skew sits below the mean the calibration
    targets. RMSE, the proper metric for a mean estimate, improves at all four
    positions ON THIS SPLIT -- and only on this split. Leave-one-season-out, the MAE
    directions hold everywhere (RB/WR/QB down, TE up 0.11-0.81%) but RMSE is a near
    wash at the margins: holding out 2024 instead, QB RMSE degrades 0.26% and TE
    0.04%. So the RMSE assertion below pins the shipped 2022-24/2025 split, not a law.
    A paired bootstrap over the 2025 test rows puts the MAE gain beyond doubt at RB
    and WR (p < 0.001) and inside the noise at QB (P(worse) = 0.22).

    So: TE is allowed to be flat, bounded at 0.5%. Nothing else is, and no position
    may lose RMSE.
    """
    report = cal.calibration_report(corpus)
    assert report.pooled.improves_mae, cal.render_report(report)

    for p in report.positions:
        assert p.rmse_calibrated <= p.rmse_raw, f"{p.name} lost RMSE\n{cal.render_report(report)}"
        if p.position_id == TE:
            assert p.mae_delta_pct < 0.5, cal.render_report(report)
        else:
            assert p.improves_mae, f"{p.name} lost MAE\n{cal.render_report(report)}"

    assert set(report.regressions) <= {"TE"}, cal.render_report(report)
    assert {p.name for p in report.positions} == {"QB", "RB", "WR", "TE"}


@needs_corpus
def test_calibrated_slope_is_closer_to_one_than_the_raw_slope(corpus: pl.DataFrame):
    """The level correction's actual job: kill the regression-to-the-mean tilt.

    Also split-specific, and worth knowing before anyone quotes it as a property.
    The correction divides the slope by the trained `b` (~0.92-0.96), so it helps
    exactly when the held-out slope is itself below 1. Holding out 2022 instead, QB's
    raw slope is 1.031 -- already above 1 -- and the correction pushes it to 1.113,
    farther from 1, not nearer. On the shipped 2025 split all four are below 1 and
    all four improve.
    """
    report = cal.calibration_report(corpus)
    for p in report.positions:
        assert abs(p.slope_calibrated - 1.0) < abs(p.slope_raw - 1.0), p.name


@needs_corpus
def test_shipped_defaults_match_a_fresh_fit(corpus: pl.DataFrame):
    """The hard-coded defaults are the fit, not a guess. Refit and diff them."""
    fitted = cal.fit_from_pairs(corpus)
    shipped = cal.default_calibration()
    assert fitted.n_pairs == shipped.n_pairs
    for pid in SKILL_POSITIONS:
        got, want = fitted.for_position(pid), shipped.for_position(pid)
        assert got.n == want.n
        assert got.level.slope == pytest.approx(want.level.slope, abs=5e-4)
        assert got.level.intercept == pytest.approx(want.level.intercept, abs=5e-4)
        assert got.hurdle.slope == pytest.approx(want.hurdle.slope, abs=5e-4)
        assert got.hurdle.intercept == pytest.approx(want.hurdle.intercept, abs=5e-4)
        assert got.spread.slope == pytest.approx(want.spread.slope, abs=5e-4)
        assert got.spread.intercept == pytest.approx(want.spread.intercept, abs=5e-4)


@needs_corpus
def test_modelled_hurdle_and_spread_track_the_held_out_deciles(corpus: pl.DataFrame):
    """The curves have to be right where players actually live, not just on average.

    Loose bounds on purpose -- a decile holds 50-260 held-out rows, so its own
    empirical rate carries a couple of points of noise.

    The known worst miss is QB decile 1 (modelled 30% blank, observed 53%), and it
    is a sample-size problem rather than a model one: 2022-2024 contains only 35 QB
    rows below a calibrated 8 points and they blank 55% of the time, while 2025's 32
    such rows blank 90%. Nobody starts a quarterback projected for 1.5 points, so
    this is documented rather than patched.
    """
    report = cal.calibration_report(corpus)
    for p in report.positions:
        for d in p.deciles:
            assert abs(d.p_zero_modelled - d.p_zero_actual) < 0.25, (p.name, d)
            if d.mean_calibrated < 1.0:
                # A straight line with a positive intercept cannot also be right at
                # mu -> 0. Measured in the bottom held-out decile: RB modelled 2.65 vs
                # observed 1.51 (+76%), WR 2.88 vs 1.95 (+48%), TE 2.51 vs 2.77 (-9%).
                # Nobody starts a sub-one-point player, so this is bounded loosely
                # rather than tightened into a form the linear model cannot deliver.
                assert 0.5 < d.sd_modelled / d.sd_actual < 2.0, (p.name, d)
                continue
            assert d.sd_modelled == pytest.approx(d.sd_actual, rel=0.35), (p.name, d)


@needs_corpus
def test_report_renders_and_names_its_regressions(corpus: pl.DataFrame):
    report = cal.calibration_report(corpus)
    text = cal.render_report(report, cal.fit_from_pairs(corpus))
    assert "MAE raw" in text
    assert "reliability by projection decile" in text
    assert "did NOT improve at: TE" in text
    assert "sigma(mu)" in text
