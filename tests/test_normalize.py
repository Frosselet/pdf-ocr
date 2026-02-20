"""Tests for docpact.normalize — centralized pipe-table text normalization."""

from docpact.normalize import normalize_pipe_table


# ---------------------------------------------------------------------------
# NBSP
# ---------------------------------------------------------------------------


class TestNBSP:
    def test_nbsp_to_space(self):
        assert normalize_pipe_table("hello\u00a0world") == "hello world"

    def test_nbsp_in_pipe_cell(self):
        assert normalize_pipe_table("| hello\u00a0world |") == "| hello world |"

    def test_multiple_nbsps_collapsed(self):
        # Two NBSPs become two spaces, then collapsed to one
        assert normalize_pipe_table("hello\u00a0\u00a0world") == "hello world"


# ---------------------------------------------------------------------------
# Smart quotes
# ---------------------------------------------------------------------------


class TestSmartQuotes:
    def test_left_single_quote(self):
        assert normalize_pipe_table("\u2018test\u2019") == "'test'"

    def test_left_double_quote(self):
        assert normalize_pipe_table("\u201ctest\u201d") == '"test"'

    def test_mixed_quotes_in_pipe(self):
        result = normalize_pipe_table("| \u201cMOA\u201d \u2018Target\u2019 |")
        assert result == '| "MOA" \'Target\' |'


# ---------------------------------------------------------------------------
# Dashes
# ---------------------------------------------------------------------------


class TestDashes:
    def test_en_dash(self):
        assert normalize_pipe_table("2020\u20132025") == "2020-2025"

    def test_em_dash(self):
        assert normalize_pipe_table("hello\u2014world") == "hello-world"

    def test_dashes_in_pipe_cell(self):
        assert normalize_pipe_table("| Jan\u2013Feb |") == "| Jan-Feb |"


# ---------------------------------------------------------------------------
# Zero-width characters
# ---------------------------------------------------------------------------


class TestZeroWidth:
    def test_zwsp(self):
        assert normalize_pipe_table("hel\u200blo") == "hello"

    def test_zwnj(self):
        assert normalize_pipe_table("hel\u200clo") == "hello"

    def test_zwj(self):
        assert normalize_pipe_table("hel\u200dlo") == "hello"

    def test_bom(self):
        assert normalize_pipe_table("\ufeffhello") == "hello"

    def test_word_joiner(self):
        assert normalize_pipe_table("hel\u2060lo") == "hello"


# ---------------------------------------------------------------------------
# Whitespace collapse
# ---------------------------------------------------------------------------


class TestWhitespaceCollapse:
    def test_double_space(self):
        assert normalize_pipe_table("MOA Target  2025") == "MOA Target 2025"

    def test_triple_space(self):
        assert normalize_pipe_table("a   b") == "a b"

    def test_tab_collapse(self):
        assert normalize_pipe_table("a\t\tb") == "a b"

    def test_single_space_preserved(self):
        assert normalize_pipe_table("a b c") == "a b c"

    def test_trailing_whitespace_stripped(self):
        assert normalize_pipe_table("hello   ") == "hello"

    def test_nbsp_plus_space_collapsed(self):
        # NBSP becomes space, then two spaces collapse
        assert normalize_pipe_table("hello\u00a0 world") == "hello world"


# ---------------------------------------------------------------------------
# Pipe-table structure preservation
# ---------------------------------------------------------------------------


class TestPipeStructure:
    def test_empty_cells_preserved(self):
        """|| (aggregation rows) must be preserved."""
        assert normalize_pipe_table("||total|100|") == "||total|100|"

    def test_separator_row_preserved(self):
        assert normalize_pipe_table("|---|---|---|") == "|---|---|---|"

    def test_standard_cell_format(self):
        """Single spaces in '| cell |' format are preserved."""
        assert normalize_pipe_table("| Port | Vessel | Tons |") == "| Port | Vessel | Tons |"

    def test_compact_format(self):
        """Compact pipe format without spaces."""
        assert normalize_pipe_table("|Port|Vessel|Tons|") == "|Port|Vessel|Tons|"

    def test_section_labels(self):
        assert normalize_pipe_table("WHEAT") == "WHEAT"

    def test_title_lines(self):
        assert normalize_pipe_table("## Table Title") == "## Table Title"

    def test_multiline_table(self):
        text = "## Title\n| A | B |\n|---|---|\n| 1 | 2 |\n||total|"
        assert normalize_pipe_table(text) == text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_clean_text_unchanged(self):
        clean = "| Port | Vessel | Tons |\n|---|---|---|\n| Sydney | MV Star | 5000 |"
        assert normalize_pipe_table(clean) == clean

    def test_double_application(self):
        dirty = "| hello\u00a0\u00a0world\u2019s |"
        once = normalize_pipe_table(dirty)
        twice = normalize_pipe_table(once)
        assert once == twice

    def test_triple_application(self):
        dirty = "| \u201cMOA  Target\u201d\u200b |"
        once = normalize_pipe_table(dirty)
        twice = normalize_pipe_table(once)
        thrice = normalize_pipe_table(twice)
        assert once == twice == thrice


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize_pipe_table("") == ""

    def test_whitespace_only(self):
        assert normalize_pipe_table("   ") == ""

    def test_form_feed_preserved(self):
        """Page separators (\\f) must be preserved."""
        assert normalize_pipe_table("page1\fpage2") == "page1\fpage2"

    def test_multipage(self):
        text = "| A |\n|---|\n| 1 |\f| B |\n|---|\n| 2 |"
        assert normalize_pipe_table(text) == text
