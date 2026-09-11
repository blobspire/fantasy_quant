"""The ETR rankings ingest.

ETR has no API and their terms forbid scraping, so this reads the CSV the
download button produces. The export is rankings, not projections; the Draft Kit
chart ships the ESPN player id and Silva's Top 150 does not -- and both facts
shape what the module can and cannot do.
"""

from __future__ import annotations

import pytest

from fantasy_quant.data import etr

HEADER = (
    '"Player","Position","Team","ETR Rank","ADP","Ranking Diff",'
    '"ETR Pos Rank","ADP Pos Rank","Pos Rank Diff","id"'
)

#: Silva's Top 150, which is a different board in the same folder: the rank column
#: is named for him, there is commentary, and there is NO id column.
TOP150 = (
    '"Player","Position","Team","Silva Rank","ADP","Ranking Diff","Rank Change",'
    '"Silva Commentary","Silva Pos Rank","ADP Pos Rank","Pos Rank Diff"'
)


class _Index:
    """Stand-in for `EspnIdIndex`, which refuses a name it cannot place."""

    def __init__(self, by_name):
        self.by_name = by_name

    def from_name(self, name, team=None, position=None):
        return self.by_name.get(name)


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
        for key, board in boards.items():
            assert board.n >= 100, f"{key} board looks truncated"
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
        for key, board in boards.items():
            ids = board.frame["espn_id"].to_list()
            hit = sum(1 for i in ids if i in known)
            assert hit / len(ids) > 0.95, f"{key}: only {hit}/{len(ids)} ids joined"


class TestTheTop150:
    """A second board, in the same folder, under the same scoring, with no id column."""

    ROWS = [
        '"Jahmyr Gibbs","RB","DET","1","1.0","0.0","0","38 TDs in two years.","RB01","RB01","0"',
        '"Puka Nacua","WR","LA","4","4.7","0.7","0","Target hog.","WR02","WR02","0"',
        '"Chris Rodriguez","RB","WSH","118","267.0","149.0","0","Goal line.","RB42","RB56","14"',
    ]
    IDS = _Index({"Jahmyr Gibbs": 4429795, "Puka Nacua": 4426515, "Chris Rodriguez": 4362238})

    def test_the_rank_column_names_the_board_so_two_boards_can_share_a_folder(self, tmp_path):
        """`load_all` keyed on scoring alone, and these two are both half PPR.

        Whichever sorted last silently replaced the other. The rank column's own
        name -- "ETR Rank" against "Silva Rank" -- is the only thing in the export
        that tells them apart.
        """
        draft = ['"A","WR","IND","5","6.0","1.0","WR1","WR1","0","111"']
        _csv(tmp_path, "etr_rankings_half_ppr.csv", draft)
        _csv(tmp_path, "silva_top150_half_ppr.csv", self.ROWS, header=TOP150)
        boards = etr.load_all(tmp_path)
        assert set(boards) == {("etr", "half_ppr"), ("silva", "half_ppr")}
        assert boards[("etr", "half_ppr")].n == 1
        assert boards[("silva", "half_ppr")].n == 3

    def test_names_resolve_to_espn_ids_when_the_board_ships_none(self, tmp_path):
        p = _csv(tmp_path, "silva_top150_half_ppr.csv", self.ROWS, header=TOP150)
        board = etr.load_file(p, id_index=self.IDS)
        assert board.kind == "silva"
        assert board.name_match == (3, 3)
        assert board.rank_of(4362238) == 118

    def test_a_board_that_mostly_fails_to_resolve_is_refused(self, tmp_path):
        """A half-matched board silently re-ranks only the half it matched, which is
        invisible in the output. Loud beats plausible."""
        p = _csv(tmp_path, "silva_top150_half_ppr.csv", self.ROWS, header=TOP150)
        with pytest.raises(etr.EtrError, match="below the"):
            etr.load_file(p, id_index=_Index({"Jahmyr Gibbs": 4429795}))

    def test_positional_is_position_id_and_rank_within_it(self, tmp_path):
        p = _csv(tmp_path, "silva_top150_half_ppr.csv", self.ROWS, header=TOP150)
        pos = etr.load_file(p, id_index=self.IDS).positional()
        assert pos[4429795] == (2, 1)  # RB01
        assert pos[4426515] == (3, 2)  # WR02
        assert pos[4362238] == (2, 42)  # RB42

    def test_the_commentary_survives_the_ingest(self, tmp_path):
        p = _csv(tmp_path, "silva_top150_half_ppr.csv", self.ROWS, header=TOP150)
        assert etr.load_file(p, id_index=self.IDS).comments()[4362238] == "Goal line."

    def test_a_positional_rank_is_derived_when_the_board_omits_one(self, tmp_path):
        """`positional()` is what every consumer reads, so it cannot be optional."""
        p = _csv(
            tmp_path,
            "silva_top150_half_ppr.csv",
            [
                '"Jahmyr Gibbs","RB","DET","1"',
                '"Puka Nacua","WR","LA","4"',
                '"Chris Rodriguez","RB","WSH","118"',
            ],
            header='"Player","Position","Team","Silva Rank"',
        )
        pos = etr.load_file(p, id_index=self.IDS).positional()
        assert pos[4429795] == (2, 1)
        assert pos[4362238] == (2, 2)  # second-ranked RB on this board
        assert pos[4426515] == (3, 1)

    def test_disagreements_refuses_a_board_with_no_field_price(self, tmp_path):
        """ADP only exists on the draft chart. Returning an empty frame would read
        as "ETR agrees with the room about everyone"."""
        p = _csv(
            tmp_path,
            "silva_top150_half_ppr.csv",
            ['"Jahmyr Gibbs","RB","DET","1","RB01"'],
            header='"Player","Position","Team","Silva Rank","Silva Pos Rank"',
        )
        with pytest.raises(etr.EtrError, match="no ADP column"):
            etr.load_file(p, id_index=self.IDS).disagreements()


class TestDeliberateSubstitution:
    """`for_league` refuses the other scoring. `best_available` makes it the caller's call."""

    def test_best_available_falls_back_and_says_that_it_did(self, tmp_path):
        _csv(tmp_path, "silva_top150_half_ppr.csv", ['"A","WR","IND","5","WR1"'],
             header='"Player","Position","Team","Silva Rank","Silva Pos Rank"')
        idx = _Index({"A": 111})
        board, matched = etr.best_available("ppr", tmp_path, kind="silva", id_index=idx)
        assert board is not None and matched is False
        assert board.applies_to("half_ppr") and not board.applies_to("ppr")
        assert etr.for_league("ppr", tmp_path, kind="silva", id_index=idx) is None

    def test_nothing_downloaded_is_not_an_error(self, tmp_path):
        assert etr.best_available("ppr", tmp_path, kind="silva") == (None, False)


class TestArchive:
    def test_a_dated_copy_is_kept_and_never_overwritten(self, tmp_path):
        """ETR overwrites each chart in place and publishes no history, so the
        comparison set has to be accumulated going forward or it never exists."""
        import datetime as dt

        p = _csv(tmp_path, "silva_top150_half_ppr.csv", ['"A","WR","IND","5","WR1"'],
                 header='"Player","Position","Team","Silva Rank","Silva Pos Rank"')
        root = tmp_path / "archive"
        day = dt.date(2026, 9, 10)
        first = etr.archive(p, root=root, today=day)
        assert first == root / "2026-09-10" / "silva_top150_half_ppr.csv"
        assert first.read_text() == p.read_text()
        assert etr.archive(p, root=root, today=day) is None

    def test_a_second_day_is_a_second_copy(self, tmp_path):
        import datetime as dt

        p = _csv(tmp_path, "silva_top150_half_ppr.csv", ['"A","WR","IND","5","WR1"'],
                 header='"Player","Position","Team","Silva Rank","Silva Pos Rank"')
        root = tmp_path / "archive"
        etr.archive(p, root=root, today=dt.date(2026, 9, 10))
        etr.archive(p, root=root, today=dt.date(2026, 9, 17))
        assert sorted(d.name for d in root.iterdir()) == ["2026-09-10", "2026-09-17"]


class TestTheBoardIsFoundFromAnyWorkingDirectory:
    """The dashboard rendered the board-less trade list while `fq trades` from the repo
    root rendered the tilted one, and nothing on either said which.

    `DEFAULT_DIR` was `Path("data/manual/etr")` and `data/reference` (the id crosswalk
    the Top 150 needs, having no id column) was relative too. Both resolve silently to
    nothing from another cwd -- `load_all` returns `{}` and `default_id_index` returns
    an index that matches nobody -- so every surface priced on ESPN alone and said so
    only in a log line.
    """

    def test_the_default_dir_resolves_to_a_real_directory(self):
        from fantasy_quant.paths import data_dir

        assert data_dir(etr.DATA_SUBDIR) == etr.DEFAULT_DIR

    def test_a_missing_directory_says_where_it_looked(self, tmp_path, caplog):
        """Silence is the failure mode being fixed; the warning IS the fix."""
        import logging

        with caplog.at_level(logging.WARNING, logger="fantasy_quant.data.etr"):
            assert etr.load_all(tmp_path / "nope") == {}
        assert any("no ETR directory" in r.message for r in caplog.records)

    def test_a_directory_with_nothing_usable_also_says_so(self, tmp_path, caplog):
        import logging

        (tmp_path / "notes.txt").write_text("not a board", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="fantasy_quant.data.etr"):
            assert etr.load_all(tmp_path) == {}
        assert any("no usable ETR export" in r.message for r in caplog.records)


@pytest.mark.network
class TestTheRealBoardLoadsFromElsewhere:
    def test_the_top_150_resolves_with_the_process_started_anywhere(self, tmp_path, monkeypatch):
        """Both cwd-relative roots at once: the export folder and the id crosswalk."""
        monkeypatch.chdir(tmp_path)
        board, _ = etr.best_available("half_ppr", kind="silva")
        if board is None:
            pytest.skip("no Silva board in data/manual/etr")
        assert board.n == 150
        assert board.name_match == (150, 150)
