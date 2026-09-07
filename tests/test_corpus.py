"""The canonical corpus reader, and the duplicate-capture trap it exists to close."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from fantasy_quant import corpus

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CORPUS = REPO_ROOT / "data" / "snapshots" / "espn"
has_corpus = pytest.mark.skipif(
    not REAL_CORPUS.exists(), reason="no local corpus; run `fq backfill`"
)


def _row(
    espn_id: int,
    season: int,
    week: int,
    source: int,
    total: float,
    captured: str,
    split: int = 1,
    pos: int = 3,
) -> dict:
    return {
        "espn_id": espn_id,
        "full_name": f"P{espn_id}",
        "default_position_id": pos,
        "pro_team_id": 8,
        "injury_status": "ACTIVE",
        "injured": False,
        "active": True,
        "eligible_slots": [3],
        "status": "ONTEAM",
        "on_team_id": 1,
        "keeper_value": 0,
        "keeper_value_future": 0,
        "draft_auction_value": 0,
        "percent_owned": 50.0,
        "percent_started": 40.0,
        "percent_change": 0.0,
        "average_draft_position": 10.0,
        "adp_percent_change": 0.0,
        "auction_value_average": 5.0,
        "auction_value_change": 0.0,
        "stat_row_id": f"{source}{split}{season}{week}",
        "stat_season": season,
        "stat_source_id": source,
        "stat_split_type_id": split,
        "scoring_period_id": week,
        "applied_total": total,
        "stat_pro_team_id": 8,
        "stat_ids": ["53"],
        "stat_values": [4.0],
        "captured_at": dt.datetime.fromisoformat(captured),
        "request_season": season,
        "scoring_variant": "ppr",
    }


def _write(tmp_path: Path, season: int, day: str, rows: list[dict]) -> Path:
    d = tmp_path / f"season={season}" / "variant=ppr"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{day}.parquet"
    pl.DataFrame(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "stat_ids": pl.List(pl.Utf8),
            "stat_values": pl.List(pl.Float64),
            "eligible_slots": pl.List(pl.Int64),
        },
    ).write_parquet(dest)
    return dest


class TestDeduplication:
    def test_the_same_season_captured_twice_is_not_double_counted(self, tmp_path):
        """The exact failure this module exists to prevent.

        Two captures of a completed season contain the same played weeks. Naive
        concatenation doubles every row, and nothing errors -- every count, MAE
        and correlation is simply computed on duplicates and looks plausible.
        """
        rows = [_row(1, 2024, w, s, 10.0, "2024-12-31T00:00:00") for w in (1, 2, 3) for s in (0, 1)]
        _write(tmp_path, 2024, "2024-12-31", rows)
        _write(
            tmp_path,
            2024,
            "2026-09-07",
            [{**r, "captured_at": dt.datetime(2026, 9, 7)} for r in rows],
        )

        naive = pl.concat(
            [pl.read_parquet(f) for f in corpus.corpus_files(root=tmp_path)], how="diagonal"
        )
        assert naive.height == 12, "both captures are on disk"
        assert corpus.load_stat_rows(root=tmp_path).height == 6, "each identity kept once"

    def test_the_newest_capture_wins(self, tmp_path):
        """ESPN revises weekly projections all season, so later supersedes earlier."""
        _write(tmp_path, 2026, "2026-09-01", [_row(1, 2026, 5, 1, 10.0, "2026-09-01T00:00:00")])
        _write(tmp_path, 2026, "2026-09-20", [_row(1, 2026, 5, 1, 17.5, "2026-09-20T00:00:00")])
        got = corpus.load_stat_rows(root=tmp_path)
        assert got.height == 1
        assert got["applied_total"][0] == pytest.approx(17.5)

    def test_recency_follows_captured_at_not_glob_order(self, tmp_path):
        """A filename that sorts late must not beat a genuinely newer capture."""
        _write(tmp_path, 2026, "2026-01-01", [_row(1, 2026, 5, 1, 99.0, "2026-12-01T00:00:00")])
        _write(tmp_path, 2026, "2026-02-01", [_row(1, 2026, 5, 1, 11.0, "2026-02-01T00:00:00")])
        got = corpus.load_stat_rows(root=tmp_path)
        assert got["applied_total"][0] == pytest.approx(99.0)

    def test_foreign_season_rows_are_dropped(self, tmp_path):
        """A 2026 request returns 2025 rows in the same array."""
        _write(
            tmp_path,
            2026,
            "2026-09-07",
            [
                _row(1, 2026, 1, 1, 12.0, "2026-09-07T00:00:00"),
                # A 2025 stat row arriving inside the 2026 file, which is what
                # ESPN actually does. request_season is the file's season.
                {**_row(1, 2025, 9, 0, 8.0, "2026-09-07T00:00:00"), "request_season": 2026},
            ],
        )
        assert corpus.load_stat_rows(root=tmp_path).height == 1
        assert corpus.load_stat_rows(root=tmp_path, own_season_only=False).height == 2


class TestOwnershipHistoryIsNotDeduplicated:
    def test_each_capture_is_kept(self, tmp_path):
        """Percent rostered and ADP are point-in-time and unrecoverable.

        Deduplicating them would discard the only genuinely irreplaceable data in
        the corpus -- the reason the daily snapshot exists at all.
        """
        for day, owned in (("2026-09-01", 40.0), ("2026-09-08", 75.0)):
            _write(
                tmp_path,
                2026,
                day,
                [{**_row(1, 2026, 1, 1, 12.0, f"{day}T00:00:00"), "percent_owned": owned}],
            )
        hist = corpus.load_ownership_history(root=tmp_path)
        assert hist.height == 2
        assert sorted(hist["percent_owned"].to_list()) == [40.0, 75.0]


class TestWeeklyPairs:
    def test_pairs_projection_with_actual(self, tmp_path):
        _write(
            tmp_path,
            2025,
            "2025-12-31",
            [
                _row(1, 2025, 1, 1, 12.0, "2025-12-31T00:00:00"),
                _row(1, 2025, 1, 0, 18.0, "2025-12-31T00:00:00"),
                _row(1, 2025, 2, 1, 11.0, "2025-12-31T00:00:00"),  # no actual -> unpaired
            ],
        )
        pairs = corpus.load_weekly_pairs(root=tmp_path)
        assert pairs.height == 1
        assert pairs["projected"][0] == 12.0 and pairs["actual"][0] == 18.0

    def test_zero_projections_are_excluded_by_default(self, tmp_path):
        """~20k deep-bench players are projected at ~0 and never play.

        Including them halves every MAE and inflates the observed zero-rate, so
        the published constants would silently stop meaning what they say.
        """
        _write(
            tmp_path,
            2025,
            "2025-12-31",
            [
                _row(1, 2025, 1, 1, 0.0, "2025-12-31T00:00:00"),
                _row(1, 2025, 1, 0, 0.0, "2025-12-31T00:00:00"),
                _row(2, 2025, 1, 1, 9.0, "2025-12-31T00:00:00"),
                _row(2, 2025, 1, 0, 7.0, "2025-12-31T00:00:00"),
            ],
        )
        assert corpus.load_weekly_pairs(root=tmp_path).height == 1
        assert corpus.load_weekly_pairs(root=tmp_path, positive_projections_only=False).height == 2


class TestMissingCorpus:
    def test_a_missing_corpus_says_what_to_run(self, tmp_path):
        with pytest.raises(corpus.CorpusError, match="fq backfill"):
            corpus.load_stat_rows(root=tmp_path / "nope")

    def test_no_files_is_an_empty_list_not_a_crash(self, tmp_path):
        assert corpus.corpus_files(root=tmp_path / "nope") == []


@has_corpus
class TestAgainstTheRealCorpus:
    def test_reproduces_the_published_constants(self):
        """The published figures were computed before the duplicate 2024 capture existed.

        If dedup regresses, these numbers move and this test says so.
        """
        pairs = corpus.load_weekly_pairs(seasons=(2022, 2023, 2024, 2025), root=REAL_CORPUS)
        assert pairs.height == 23_999
        expected = {
            1: (2210, 5.87, 0.032),
            2: (6312, 4.10, 0.185),
            3: (10004, 4.27, 0.259),
            4: (5473, 3.24, 0.325),
        }
        for pos, (n, mae, hurdle) in expected.items():
            s = pairs.filter(pl.col("default_position_id") == pos)
            assert s.height == n
            err = (s["actual"] - s["projected"]).abs().mean()
            assert err == pytest.approx(mae, abs=0.01)
            assert (s["actual"] <= 0).mean() == pytest.approx(hurdle, abs=0.001)

    def test_the_real_corpus_actually_contains_a_duplicate_to_defend_against(self):
        """Guards the guard: if this stops being true the dedup test proves nothing."""
        raw = pl.concat(
            [pl.read_parquet(f) for f in corpus.corpus_files(root=REAL_CORPUS)], how="diagonal"
        ).filter(pl.col("stat_season") == pl.col("request_season"))
        deduped = corpus.load_stat_rows(root=REAL_CORPUS)
        assert raw.height > deduped.height, "expected at least one duplicated capture"
