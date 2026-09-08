"""The ETR rankings ingest.

ETR has no API and their terms forbid scraping, so this reads the CSV the
download button produces. The export is rankings, not projections, and it ships
the ESPN player id -- both facts shape what the module can and cannot do.
"""

from __future__ import annotations

import pytest

from fantasy_quant.data import etr

HEADER = (
    '"Player","Position","Team","ETR Rank","ADP","Ranking Diff",'
    '"ETR Pos Rank","ADP Pos Rank","Pos Rank Diff","id"'
)


def _csv(tmp_path, name, rows, header=HEADER):
    path = tmp_path / name
    path.write_text("﻿" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


class TestLoading:
    def test_reads_the_export_shape(self, tmp_path):
        p = _csv(
            tmp_path,
            "etr_rankings_full_ppr.csv",
            [
                '"Jahmyr Gibbs","RB","DET","1","1.0","0.0","RB01","RB01","0","4429795"',
                '"Josh Downs","WR","IND","92","111.0","19.0","WR40","WR44","4","4688813"',
            ],
        )
        board = etr.load_file(p)
        assert board.n == 2
        assert board.scoring == "ppr"
        assert board.rank_of(4688813) == 92

    def test_the_bom_does_not_break_the_first_column(self, tmp_path):
        """ETR writes a UTF-8 BOM; a naive read turns `Player` into `\\ufeffPlayer`."""
        p = _csv(
            tmp_path, "x_half_ppr.csv", ['"A","WR","IND","5","6.0","1.0","WR1","WR1","0","111"']
        )
        assert etr.load_file(p).n == 1

    def test_scoring_is_inferred_from_the_filename(self, tmp_path):
        rows = ['"A","WR","IND","5","6.0","1.0","WR1","WR1","0","111"']
        assert etr.load_file(_csv(tmp_path, "etr_half_ppr.csv", rows)).scoring == "half_ppr"
        assert etr.load_file(_csv(tmp_path, "etr_full_ppr.csv", rows)).scoring == "ppr"

    def test_a_drifted_schema_fails_loudly_with_the_columns_it_saw(self, tmp_path):
        """The export schema is undocumented. A silently mis-parsed board is worse
        than a missing one, so this must raise rather than return junk."""
        p = _csv(tmp_path, "etr_ppr.csv", ['"A","1"'], header='"Athlete","Rating"')
        with pytest.raises(etr.EtrError, match="missing"):
            etr.load_file(p)

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(etr.EtrError, match="no ETR export"):
            etr.load_file(tmp_path / "nope.csv")

    def test_rows_without_an_id_are_dropped_not_guessed(self, tmp_path):
        p = _csv(
            tmp_path,
            "etr_ppr.csv",
            [
                '"Good","WR","IND","5","6.0","1.0","WR1","WR1","0","111"',
                '"NoId","WR","IND","6","7.0","1.0","WR2","WR2","0",""',
            ],
        )
        assert etr.load_file(p).n == 1


class TestDisagreements:
    def test_positive_edge_means_etr_likes_him_more_than_the_room(self, tmp_path):
        p = _csv(
            tmp_path,
            "etr_ppr.csv",
            [
                '"Liked","WR","IND","50","120.0","70.0","WR20","WR40","20","1"',
                '"Faded","RB","IND","200","100.0","-100.0","RB60","RB30","-30","2"',
                '"Agreed","TE","IND","30","31.0","1.0","TE5","TE5","0","3"',
            ],
        )
        d = etr.load_file(p).disagreements(minimum=20)
        names = d["player"].to_list()
        assert names[0] == "Liked" and names[-1] == "Faded"
        assert "Agreed" not in names


class TestScoringSubstitution:
    def test_for_league_never_substitutes_the_other_format(self, tmp_path):
        """A full-PPR board on a half-PPR league systematically overrates receivers."""
        _csv(tmp_path, "etr_full_ppr.csv", ['"A","WR","IND","5","6.0","1.0","WR1","WR1","0","111"'])
        assert etr.for_league("ppr", tmp_path) is not None
        assert etr.for_league("half_ppr", tmp_path) is None

    def test_an_empty_directory_is_not_an_error(self, tmp_path):
        assert etr.load_all(tmp_path) == {}
        assert etr.for_league("ppr", tmp_path) is None


class TestCoverage:
    def test_counts_matched_against_a_roster(self, tmp_path):
        p = _csv(
            tmp_path,
            "etr_ppr.csv",
            [
                '"A","WR","IND","5","6.0","1.0","WR1","WR1","0","111"',
                '"B","RB","IND","6","7.0","1.0","RB1","RB1","0","222"',
            ],
        )
        assert etr.coverage(etr.load_file(p), [111, 222, 333]) == (2, 3)


@pytest.mark.network
class TestAgainstTheRealExports:
    """The downloaded boards, if present. Skipped on a clean checkout."""

    def test_both_boards_load_and_carry_espn_ids(self):
        boards = etr.load_all()
        if not boards:
            pytest.skip("no ETR exports in data/manual/etr")
        for scoring, board in boards.items():
            assert board.n >= 100, f"{scoring} board looks truncated"
            assert board.frame["espn_id"].null_count() == 0

    def test_the_ids_join_to_our_player_pool(self):
        """ETR ships the ESPN id, which is why this source needs no fuzzy matching."""
        import polars as pl

        from fantasy_quant import corpus

        boards = etr.load_all()
        if not boards:
            pytest.skip("no ETR exports in data/manual/etr")
        known = set(
            corpus.load_stat_rows(seasons=(2026,)).select(pl.col("espn_id")).to_series().to_list()
        )
        for scoring, board in boards.items():
            ids = board.frame["espn_id"].to_list()
            hit = sum(1 for i in ids if i in known)
            assert hit / len(ids) > 0.95, f"{scoring}: only {hit}/{len(ids)} ids joined"
