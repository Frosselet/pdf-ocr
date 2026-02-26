"""Tests for XLSX table extraction.

Organized by persona (P1-P4) with heuristic-specific and edge-case tests.
Uses synthetic XLSX fixtures in inputs/xlsx/synthetic/.

Run with: uv run pytest tests/test_xlsx_extractor.py -v
To regenerate fixtures: uv run python tests/generate_synthetic_xlsx.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docpact.xlsx_extractor import (
    _build_column_names_with_forward_fill,
    _detect_number_format_hint,
    _detect_title_row,
    _split_by_blank_runs,
    extract_tables_from_xlsx,
    xlsx_to_markdown,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYNTH_DIR = Path("inputs/xlsx/synthetic")


def _synth(name: str) -> Path:
    p = SYNTH_DIR / f"{name}.xlsx"
    assert p.exists(), (
        f"Synthetic XLSX not found: {p}  "
        f"(run: uv run python tests/generate_synthetic_xlsx.py)"
    )
    return p


# ---------------------------------------------------------------------------
# P1: "The Tidy Analyst" — clean baseline
# ---------------------------------------------------------------------------


class TestTidyAnalyst:
    """P1: Clean single-table sheets with no merges."""

    def test_simple_financial_shape(self) -> None:
        """5 columns, 10 data rows, correct headers."""
        tables = extract_tables_from_xlsx(_synth("p1_simple_financial"))
        assert len(tables) == 1
        t = tables[0]
        assert t.column_names == ["Company", "Revenue", "Profit", "Employees", "Founded"]
        assert len(t.data) == 10
        assert t.source_format == "xlsx"

    def test_simple_financial_data(self) -> None:
        """First row values correct."""
        tables = extract_tables_from_xlsx(_synth("p1_simple_financial"))
        row0 = tables[0].data[0]
        assert row0[0] == "Acme Corp"
        assert row0[1] == "50000"

    def test_inventory_mixed_types(self) -> None:
        """7 columns with mixed types (text, number, date, enum)."""
        tables = extract_tables_from_xlsx(_synth("p1_inventory_mixed"))
        assert len(tables) == 1
        t = tables[0]
        assert len(t.column_names) == 7
        assert t.column_names[0] == "Item"
        assert t.column_names[4] == "Last Restocked"
        assert len(t.data) == 10
        # XH4: date format hint
        assert t.metadata.get("format_hints", {}).get(4) == "date"

    def test_time_series_date_column(self) -> None:
        """Date column + 4 numeric columns, 30 rows."""
        tables = extract_tables_from_xlsx(_synth("p1_time_series"))
        assert len(tables) == 1
        t = tables[0]
        assert t.column_names[0] == "Date"
        assert len(t.data) == 30
        assert t.metadata.get("format_hints", {}).get(0) == "date"

    def test_multi_sheet_enumeration(self) -> None:
        """3 sheets produce 3 tables with correct sheet names."""
        tables = extract_tables_from_xlsx(_synth("p1_multi_sheet"))
        assert len(tables) == 3
        sheet_names = [t.metadata["sheet_name"] for t in tables]
        assert sheet_names == ["Sales", "Targets", "Regions"]

    def test_multi_sheet_filter(self) -> None:
        """sheets= parameter restricts which sheets are processed."""
        tables_all = extract_tables_from_xlsx(_synth("p1_multi_sheet"))
        tables_one = extract_tables_from_xlsx(_synth("p1_multi_sheet"), sheets=["Targets"])
        assert len(tables_all) == 3
        assert len(tables_one) == 1
        assert tables_one[0].metadata["sheet_name"] == "Targets"
        assert tables_one[0].column_names[0] == "Quarter"

    def test_multi_sheet_by_index(self) -> None:
        """sheets= accepts integer indices."""
        tables = extract_tables_from_xlsx(_synth("p1_multi_sheet"), sheets=[2])
        assert len(tables) == 1
        assert tables[0].metadata["sheet_name"] == "Regions"


# ---------------------------------------------------------------------------
# P2: "The Merger" — hierarchical merged headers
# ---------------------------------------------------------------------------


class TestMerger:
    """P2: Tables with merged header cells producing compound headers."""

    def test_budget_compound_headers(self) -> None:
        """2-row header: Revenue/Cost span 2 cols each → compound names."""
        tables = extract_tables_from_xlsx(_synth("p2_budget_merged"))
        assert len(tables) == 1
        t = tables[0]
        assert t.column_names == [
            "Department",
            "Revenue / Q1", "Revenue / Q2",
            "Cost / Q1", "Cost / Q2",
        ]
        assert len(t.data) == 5

    def test_budget_data_correct(self) -> None:
        """Data rows have correct values."""
        tables = extract_tables_from_xlsx(_synth("p2_budget_merged"))
        row0 = tables[0].data[0]
        assert row0[0] == "Engineering"
        assert row0[1] == "450"  # Revenue Q1

    def test_deep_hierarchy_3_levels(self) -> None:
        """3-level header: Year > Quarter > Month."""
        tables = extract_tables_from_xlsx(_synth("p2_deep_hierarchy"))
        assert len(tables) == 1
        t = tables[0]
        assert len(t.data) == 5
        # Check compound header structure
        assert t.column_names[0] == "Region"
        assert t.column_names[1] == "2025 / Q1 / Jan"
        assert t.column_names[2] == "2025 / Q1 / Feb"
        assert t.column_names[3] == "2025 / Q1 / Mar"
        assert t.column_names[4] == "2025 / Q2 / Apr"
        assert t.column_names[7] == "2026 / Q1 / Jan"

    def test_cross_tab_bidirectional(self) -> None:
        """Merged column headers with row group labels."""
        tables = extract_tables_from_xlsx(_synth("p2_cross_tab"))
        assert len(tables) == 1
        t = tables[0]
        # Column headers should have Product A / Revenue, etc.
        assert "Product A / Revenue" in t.column_names
        assert "Product B / Cost" in t.column_names
        assert len(t.data) == 4

    def test_irregular_merges_graceful(self) -> None:
        """Non-rectangular merges with gaps handled gracefully."""
        tables = extract_tables_from_xlsx(_synth("p2_irregular_merges"))
        assert len(tables) == 1
        t = tables[0]
        assert len(t.data) == 4
        # Metrics group should be present
        assert any("Metrics" in h for h in t.column_names)
        assert any("Value A" in h for h in t.column_names)


# ---------------------------------------------------------------------------
# P3: "The Multi-Tasker" — multiple tables per sheet
# ---------------------------------------------------------------------------


class TestMultiTasker:
    """P3: Multiple tables per sheet with gap detection (XH1)."""

    def test_two_tables_blank_rows(self) -> None:
        """2 tables separated by 3 blank rows → 2 StructuredTables."""
        tables = extract_tables_from_xlsx(_synth("p3_two_tables_blank_rows"))
        assert len(tables) == 2
        # Table 1: Name/Score/Grade
        assert tables[0].column_names == ["Name", "Score", "Grade"]
        assert len(tables[0].data) == 5
        # Table 2: City/Population/Area/Density
        assert tables[1].column_names == ["City", "Population", "Area", "Density"]
        assert len(tables[1].data) == 4

    def test_tables_with_titles(self) -> None:
        """Title rows above each table → XH2 title extraction."""
        tables = extract_tables_from_xlsx(_synth("p3_tables_with_titles"))
        assert len(tables) == 2
        assert tables[0].metadata.get("title") == "Employee Performance"
        assert tables[1].metadata.get("title") == "Department Summary"
        # Title should not appear in column names
        assert "Employee Performance" not in tables[0].column_names
        assert "Department Summary" not in tables[1].column_names

    def test_tables_with_titles_data(self) -> None:
        """Title tables have correct data."""
        tables = extract_tables_from_xlsx(_synth("p3_tables_with_titles"))
        assert tables[0].column_names == ["Name", "Department", "Rating"]
        assert len(tables[0].data) == 4
        assert tables[1].column_names == ["Department", "Headcount", "Avg Rating"]
        assert len(tables[1].data) == 3

    def test_tables_with_footnotes(self) -> None:
        """Table with footnotes + blank rows + second table → 2 tables."""
        tables = extract_tables_from_xlsx(_synth("p3_tables_with_footnotes"))
        assert len(tables) == 2
        # Table 1 includes footnotes (single blank rows are within-table)
        assert tables[0].column_names[0] == "Item"
        # Table 2 starts after 2+ blank rows
        assert tables[1].column_names[0] == "Category"
        assert len(tables[1].data) == 4

    def test_side_by_side_tables(self) -> None:
        """2 tables side by side with blank columns between → 2 tables."""
        tables = extract_tables_from_xlsx(_synth("p3_side_by_side"))
        assert len(tables) == 2
        # Left table
        assert tables[0].column_names == ["Product", "Sales", "Returns"]
        assert len(tables[0].data) == 4
        # Right table
        assert tables[1].column_names == ["Region", "Revenue", "Growth"]
        assert len(tables[1].data) == 4

    def test_table_indices_preserved(self) -> None:
        """Each table has correct table_index in metadata."""
        tables = extract_tables_from_xlsx(_synth("p3_two_tables_blank_rows"))
        assert tables[0].metadata["table_index"] == 0
        assert tables[1].metadata["table_index"] == 1

    def test_first_data_values_correct(self) -> None:
        """Side-by-side tables have correct first-row values."""
        tables = extract_tables_from_xlsx(_synth("p3_side_by_side"))
        assert tables[0].data[0][0] == "Widget"
        assert tables[1].data[0][0] == "North"


# ---------------------------------------------------------------------------
# P4: "The Formatter" — visual styling, hidden content
# ---------------------------------------------------------------------------


class TestFormatter:
    """P4: Visual styling, hidden content, date formats."""

    def test_kpi_dashboard_headers(self) -> None:
        """Conditional formatting doesn't affect header detection."""
        tables = extract_tables_from_xlsx(_synth("p4_kpi_dashboard"))
        assert len(tables) == 1
        t = tables[0]
        assert t.column_names == ["KPI", "Target", "Actual", "Status"]
        assert len(t.data) == 6

    def test_banded_report_visual(self) -> None:
        """Zebra rows and borders don't affect extraction."""
        tables = extract_tables_from_xlsx(_synth("p4_banded_report"))
        assert len(tables) == 1
        t = tables[0]
        assert t.column_names == ["ID", "Description", "Category", "Amount", "Date"]
        assert len(t.data) == 8

    def test_hidden_columns_filtered(self) -> None:
        """XH3: Hidden columns are excluded from output."""
        # With hidden filtering (default)
        tables = extract_tables_from_xlsx(_synth("p4_hidden_columns"))
        assert len(tables) == 1
        t = tables[0]
        assert len(t.column_names) == 3
        assert t.column_names == ["Product", "Base Price", "Final Price"]
        # Tax Rate and Tax Amount columns are hidden → excluded

    def test_hidden_columns_included_when_disabled(self) -> None:
        """XH3 can be disabled to include hidden columns."""
        tables = extract_tables_from_xlsx(
            _synth("p4_hidden_columns"), filter_hidden=False,
        )
        assert len(tables) == 1
        t = tables[0]
        assert len(t.column_names) == 5
        assert t.column_names == [
            "Product", "Base Price", "Tax Rate", "Tax Amount", "Final Price",
        ]

    def test_date_formats_detected(self) -> None:
        """XH4: Date number formats produce format hints."""
        tables = extract_tables_from_xlsx(_synth("p4_date_formats"))
        assert len(tables) == 1
        t = tables[0]
        assert t.column_names[0] == "Event"
        fmt = t.metadata.get("format_hints", {})
        # Columns 1-4 all have date formats
        assert fmt.get(1) == "date"
        assert fmt.get(2) == "date"
        assert fmt.get(3) == "date"
        assert fmt.get(4) == "date"

    def test_date_formats_data_values(self) -> None:
        """Date values are converted to strings."""
        tables = extract_tables_from_xlsx(_synth("p4_date_formats"))
        # All date columns should have string representations
        for row in tables[0].data:
            assert row[0]  # Event name exists
            # Date values are non-empty strings
            for ci in range(1, 5):
                assert row[ci], f"Column {ci} has empty date value"


# ---------------------------------------------------------------------------
# XH1: Blank Row/Column Table Boundary — unit tests
# ---------------------------------------------------------------------------


class TestXH1BlankBoundary:
    """Unit tests for the _split_by_blank_runs helper."""

    def test_no_blanks(self) -> None:
        """All non-blank → single segment."""
        result = _split_by_blank_runs([False, False, False, False], min_gap=2)
        assert result == [(0, 3)]

    def test_single_blank_not_separator(self) -> None:
        """Single blank row within threshold → stays in one segment."""
        result = _split_by_blank_runs(
            [False, False, True, False, False], min_gap=2,
        )
        assert result == [(0, 4)]

    def test_two_blanks_separator(self) -> None:
        """Two consecutive blanks → splits into 2 segments."""
        result = _split_by_blank_runs(
            [False, False, True, True, False, False], min_gap=2,
        )
        assert result == [(0, 1), (4, 5)]

    def test_three_blanks_separator(self) -> None:
        """Three consecutive blanks → still splits."""
        result = _split_by_blank_runs(
            [False, True, True, True, False, False], min_gap=2,
        )
        assert result == [(0, 0), (4, 5)]

    def test_all_blank(self) -> None:
        """All blank → no segments."""
        result = _split_by_blank_runs([True, True, True], min_gap=2)
        assert result == []

    def test_leading_blanks(self) -> None:
        """Leading blanks are skipped."""
        result = _split_by_blank_runs(
            [True, True, False, False], min_gap=2,
        )
        assert result == [(2, 3)]


# ---------------------------------------------------------------------------
# XH2: Title Row Detection — unit tests
# ---------------------------------------------------------------------------


class TestXH2TitleDetection:
    """Unit tests for _detect_title_row."""

    def test_title_detected(self) -> None:
        """Single non-empty cell in row 0 with header_count > 1."""
        grid = [
            ["REPORT TITLE", "", ""],
            ["Col A", "Col B", "Col C"],
            ["data", "123", "456"],
        ]
        title, new_grid, new_hc = _detect_title_row(grid, header_count=2)
        assert title == "REPORT TITLE"
        assert new_grid[0] == ["Col A", "Col B", "Col C"]
        assert new_hc == 1

    def test_no_title_when_header_count_1(self) -> None:
        """Title not extracted if header_count == 1."""
        grid = [["Only Header", "", ""], ["data", "123", "456"]]
        title, new_grid, new_hc = _detect_title_row(grid, header_count=1)
        assert title is None
        assert new_grid is grid
        assert new_hc == 1

    def test_no_title_when_multiple_non_empty(self) -> None:
        """Row with 2+ non-empty cells is not a title."""
        grid = [
            ["Col A", "Col B", ""],
            ["Sub1", "Sub2", "Sub3"],
            ["data", "123", "456"],
        ]
        title, new_grid, new_hc = _detect_title_row(grid, header_count=2)
        assert title is None
        assert new_grid is grid
        assert new_hc == 2


# ---------------------------------------------------------------------------
# Forward-fill compound headers — unit tests
# ---------------------------------------------------------------------------


class TestForwardFillHeaders:
    """Unit tests for _build_column_names_with_forward_fill."""

    def test_single_row_no_fill(self) -> None:
        result = _build_column_names_with_forward_fill([["A", "B", "C"]])
        assert result == ["A", "B", "C"]

    def test_forward_fill_propagates(self) -> None:
        """Empty cells get preceding value."""
        result = _build_column_names_with_forward_fill([["X", "", ""]])
        assert result == ["X", "X", "X"]

    def test_two_row_compound(self) -> None:
        """Multi-row header with forward-fill."""
        result = _build_column_names_with_forward_fill([
            ["Region", "Revenue", "", "Cost", ""],
            ["Region", "Q1", "Q2", "Q1", "Q2"],
        ])
        assert result == [
            "Region", "Revenue / Q1", "Revenue / Q2",
            "Cost / Q1", "Cost / Q2",
        ]

    def test_dedup_consecutive(self) -> None:
        """Same value in both rows → no duplication."""
        result = _build_column_names_with_forward_fill([
            ["Region", "Data"],
            ["Region", "Data"],
        ])
        assert result == ["Region", "Data"]

    def test_three_row_hierarchy(self) -> None:
        """Three-level header hierarchy."""
        result = _build_column_names_with_forward_fill([
            ["ID", "2025", "", ""],
            ["ID", "Q1", "", "Q2"],
            ["ID", "Jan", "Feb", "Mar"],
        ])
        assert result == [
            "ID", "2025 / Q1 / Jan", "2025 / Q1 / Feb", "2025 / Q2 / Mar",
        ]

    def test_empty_input(self) -> None:
        assert _build_column_names_with_forward_fill([]) == []


# ---------------------------------------------------------------------------
# XH4: Number Format Interpretation — unit tests
# ---------------------------------------------------------------------------


class TestXH4NumberFormat:
    """Unit tests for _detect_number_format_hint."""

    def test_general_returns_none(self) -> None:
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.number_format = "General"
        assert _detect_number_format_hint(cell) is None

    def test_date_format(self) -> None:
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.number_format = "YYYY-MM-DD"
        assert _detect_number_format_hint(cell) == "date"

    def test_us_date_format(self) -> None:
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.number_format = "MM/DD/YYYY"
        assert _detect_number_format_hint(cell) == "date"

    def test_currency_format(self) -> None:
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.number_format = "$#,##0.00"
        assert _detect_number_format_hint(cell) == "currency"

    def test_percentage_format(self) -> None:
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.number_format = "0%"
        assert _detect_number_format_hint(cell) == "percentage"

    def test_none_format(self) -> None:
        from unittest.mock import MagicMock
        cell = MagicMock()
        cell.number_format = None
        assert _detect_number_format_hint(cell) is None


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestXlsxToMarkdown:
    """Test xlsx_to_markdown convenience function."""

    def test_simple_render(self) -> None:
        md = xlsx_to_markdown(_synth("p1_simple_financial"))
        assert "## Sheet: Financials" in md
        assert "Company" in md
        assert "Acme Corp" in md

    def test_multi_sheet_render(self) -> None:
        md = xlsx_to_markdown(_synth("p1_multi_sheet"))
        assert "## Sheet: Sales" in md
        assert "## Sheet: Targets" in md
        assert "## Sheet: Regions" in md


# ---------------------------------------------------------------------------
# Smoke test: all fixtures
# ---------------------------------------------------------------------------


def test_all_fixtures_smoke() -> None:
    """Every synthetic XLSX file extracts without error."""
    for path in sorted(SYNTH_DIR.glob("*.xlsx")):
        tables = extract_tables_from_xlsx(path)
        assert len(tables) > 0, f"{path.name}: no tables extracted"
        for i, t in enumerate(tables):
            assert len(t.column_names) > 0, f"{path.name} table {i}: no columns"
            assert t.source_format == "xlsx"
            assert "sheet_name" in t.metadata


def test_all_fixtures_have_data() -> None:
    """Every extracted table has at least some data rows."""
    for path in sorted(SYNTH_DIR.glob("*.xlsx")):
        tables = extract_tables_from_xlsx(path)
        for i, t in enumerate(tables):
            assert len(t.data) > 0, f"{path.name} table {i}: no data rows"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        extract_tables_from_xlsx("nonexistent.xlsx")


def test_invalid_sheet_name_ignored() -> None:
    """Invalid sheet names are silently ignored."""
    tables = extract_tables_from_xlsx(
        _synth("p1_simple_financial"),
        sheets=["NonexistentSheet"],
    )
    assert tables == []
