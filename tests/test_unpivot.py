"""Tests for pdf_ocr.unpivot — schema-agnostic pivot detection and unpivoting."""

import pytest

from pdf_ocr.unpivot import (
    PivotDetection,
    UnpivotResult,
    _match_suffix_lists,
    _parse_header_cells,
    _split_compound,
    detect_pivot,
    unpivot_pipe_table,
)


# ===========================================================================
# Helper tests
# ===========================================================================


class TestSplitCompound:
    def test_simple_header(self):
        assert _split_compound("Region") == (None, "Region")

    def test_compound_header(self):
        assert _split_compound("spring crops / Target") == ("spring crops", "Target")

    def test_compound_with_spaces(self):
        assert _split_compound("spring crops / MOA Target 2025") == (
            "spring crops",
            "MOA Target 2025",
        )

    def test_multiple_slashes_only_first_split(self):
        assert _split_compound("A / B / C") == ("A", "B / C")

    def test_empty_string(self):
        assert _split_compound("") == (None, "")


class TestParseHeaderCells:
    def test_standard_pipe_row(self):
        assert _parse_header_cells("| A | B | C |") == ["A", "B", "C"]

    def test_compact_pipe_row(self):
        assert _parse_header_cells("|A|B|C|") == ["A", "B", "C"]

    def test_cells_with_spaces(self):
        assert _parse_header_cells("| spring crops / Target | Region |") == [
            "spring crops / Target",
            "Region",
        ]


class TestMatchSuffixLists:
    def test_identical_suffixes(self):
        result = _match_suffix_lists(["X", "Y", "Z"], ["X", "Y", "Z"], 85.0)
        assert result is not None
        assert len(result) == 3

    def test_fuzzy_match(self):
        result = _match_suffix_lists(
            ["Target 2025", "Actual 2025"],
            ["Targer 2025", "Actual 2025"],  # typo in Target
            85.0,
        )
        assert result is not None
        assert len(result) == 2

    def test_no_match(self):
        result = _match_suffix_lists(["X", "Y"], ["A", "B"], 85.0)
        assert result is None

    def test_single_match_insufficient(self):
        result = _match_suffix_lists(["X", "Y"], ["X", "B"], 85.0)
        assert result is None

    def test_partial_overlap(self):
        result = _match_suffix_lists(
            ["MOA Target 2025", "2025", "%", "2024", "MOA 2024"],
            ["MOA Target 2025", "2025", "2024", "MOA 2024"],
            85.0,
        )
        assert result is not None
        assert len(result) >= 4


# ===========================================================================
# Detection tests
# ===========================================================================


class TestDetectPivotNoCompoundHeaders:
    def test_simple_flat_table(self):
        text = "| Region | Area | Yield |\n|---|---|---|\n| A | 100 | 50 |"
        det = detect_pivot(text)
        assert not det.is_pivoted

    def test_empty_text(self):
        det = detect_pivot("")
        assert not det.is_pivoted

    def test_no_pipe_table(self):
        det = detect_pivot("Just some plain text\nwithout any tables")
        assert not det.is_pivoted


class TestDetectPivotSingleGroup:
    def test_single_prefix_not_pivoted(self):
        text = "| Region | CORN / 2025 | CORN / 2024 |\n|---|---|---|\n| A | 100 | 90 |"
        det = detect_pivot(text)
        assert not det.is_pivoted


class TestDetectPivotTwoGroups:
    def test_two_groups_identical_suffixes(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| row1 | 1 | 2 | 3 | 4 |"
        )
        det = detect_pivot(text)
        assert det.is_pivoted
        assert len(det.groups) == 2
        prefixes = {g.prefix for g in det.groups}
        assert prefixes == {"A", "B"}
        assert det.measure_labels == ["X", "Y"]

    def test_shared_columns_detected(self):
        text = (
            "| Region | Code | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|---|\n"
            "| row1 | r1 | 1 | 2 | 3 | 4 |"
        )
        det = detect_pivot(text)
        assert det.is_pivoted
        assert 0 in det.shared_col_indices  # Region
        assert 1 in det.shared_col_indices  # Code


class TestDetectPivotFuzzyMatch:
    def test_typo_tolerance(self):
        text = (
            "| Region | A / Target 2025 | A / Actual 2025 | B / Targer 2025 | B / Actual 2025 |\n"
            "|---|---|---|---|---|\n"
            "| row1 | 1 | 2 | 3 | 4 |"
        )
        det = detect_pivot(text)
        assert det.is_pivoted
        assert len(det.groups) == 2


class TestDetectPivotUnequalSizes:
    def test_intersection_of_matching(self):
        text = (
            "| Region | A / X | A / Y | A / Z | B / X | B / Y |\n"
            "|---|---|---|---|---|---|\n"
            "| row1 | 1 | 2 | 3 | 4 | 5 |"
        )
        det = detect_pivot(text)
        assert det.is_pivoted
        # X and Y are common, Z is only in A
        assert len(det.measure_labels) == 2
        assert "X" in det.measure_labels
        assert "Y" in det.measure_labels


class TestDetectPivotNonMatchingPrefix:
    def test_non_matching_group_becomes_shared(self):
        text = (
            "| Th.ha. / Region | Th.ha. / MOA 2024 | A / Target | A / 2025 | B / Target | B / 2025 |\n"
            "|---|---|---|---|---|---|\n"
            "| row1 | v1 | 1 | 2 | 3 | 4 |"
        )
        det = detect_pivot(text)
        assert det.is_pivoted
        # Th.ha. group has suffixes ["Region", "MOA 2024"] which don't match ["Target", "2025"]
        # So Th.ha. columns become shared
        prefixes = {g.prefix for g in det.groups}
        assert "A" in prefixes
        assert "B" in prefixes
        assert "Th.ha." not in prefixes
        # Th.ha. column indices should be in shared
        assert 0 in det.shared_col_indices
        assert 1 in det.shared_col_indices


class TestDetectPivotThreeGroups:
    def test_three_groups(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y | C / X | C / Y |\n"
            "|---|---|---|---|---|---|---|\n"
            "| row1 | 1 | 2 | 3 | 4 | 5 | 6 |"
        )
        det = detect_pivot(text)
        assert det.is_pivoted
        assert len(det.groups) == 3


class TestDetectPivotThreshold:
    def test_near_miss_below_threshold(self):
        # "AB" vs "XY" — very low similarity, should fail
        text = (
            "| Region | G1 / AB | G1 / CD | G2 / XY | G2 / CD |\n"
            "|---|---|---|---|---|\n"
            "| row1 | 1 | 2 | 3 | 4 |"
        )
        det = detect_pivot(text, similarity_threshold=85.0)
        # Only CD matches (1 match), AB vs XY doesn't → not pivoted
        assert not det.is_pivoted

    def test_low_threshold_enables_fuzzy_match(self):
        # "Target" vs "Tgt" — similar enough at low threshold
        text = (
            "| Region | G1 / Target | G1 / Actual | G2 / Tgt | G2 / Actual |\n"
            "|---|---|---|---|---|\n"
            "| row1 | 1 | 2 | 3 | 4 |"
        )
        # At default 85 threshold, "Target" vs "Tgt" won't match
        det_strict = detect_pivot(text, similarity_threshold=85.0)
        assert not det_strict.is_pivoted
        # At lower threshold, it should match
        det_lenient = detect_pivot(text, similarity_threshold=40.0)
        assert det_lenient.is_pivoted


# ===========================================================================
# Transformation tests
# ===========================================================================


class TestBasicUnpivot:
    def test_two_groups_shared_col(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| row1 | 1 | 2 | 3 | 4 |\n"
            "| row2 | 5 | 6 | 7 | 8 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        lines = result.text.strip().split("\n")
        # Header: _pivot | Region | X | Y
        header = lines[0]
        assert "_pivot" in header
        assert "Region" in header
        assert "X" in header
        assert "Y" in header
        # 2 original rows × 2 groups = 4 data rows
        data_lines = [l for l in lines if l.startswith("|") and "---" not in l and l != lines[0]]
        assert len(data_lines) == 4

    def test_values_correct(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |"
        )
        result = unpivot_pipe_table(text)
        lines = [l for l in result.text.strip().split("\n") if l.startswith("|") and "---" not in l]
        # Skip header
        data = lines[1:]
        # First row: A group
        cells_a = [c.strip() for c in data[0].strip("|").split("|")]
        assert cells_a[0] == "A"
        assert cells_a[1] == "r1"
        assert cells_a[2] == "1"
        assert cells_a[3] == "2"
        # Second row: B group
        cells_b = [c.strip() for c in data[1].strip("|").split("|")]
        assert cells_b[0] == "B"
        assert cells_b[1] == "r1"
        assert cells_b[2] == "3"
        assert cells_b[3] == "4"


class TestSectionsPreserved:
    def test_section_labels_stay(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "**CENTRAL**\n"
            "| r1 | 1 | 2 | 3 | 4 |\n"
            "**SOUTH**\n"
            "| r2 | 5 | 6 | 7 | 8 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        assert "**CENTRAL**" in result.text
        assert "**SOUTH**" in result.text
        # Check that CENTRAL appears before first data row
        lines = result.text.split("\n")
        central_idx = next(i for i, l in enumerate(lines) if "**CENTRAL**" in l)
        first_data_idx = next(
            i for i, l in enumerate(lines) if l.startswith("|") and "---" not in l and "_pivot" not in l
        )
        assert central_idx < first_data_idx


class TestAggregationExpanded:
    def test_aggregation_rows(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |\n"
            "|| 1 | 2 | 3 | 4 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        # Aggregation row should also be expanded: 2 groups
        data_lines = [
            l for l in result.text.split("\n")
            if l.startswith("|") and "---" not in l and "_pivot" not in l
        ]
        assert len(data_lines) == 4  # 2 rows × 2 groups


class TestTitlePreserved:
    def test_heading_kept(self):
        text = (
            "## My Table\n"
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        assert result.text.startswith("## My Table\n")


class TestPivotColumnFirst:
    def test_pivot_is_first_column(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |"
        )
        result = unpivot_pipe_table(text)
        headers = [c.strip() for c in result.text.split("\n")[0].strip("|").split("|")]
        assert headers[0] == "_pivot"


class TestRowOrder:
    def test_groups_in_left_to_right_order(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |"
        )
        result = unpivot_pipe_table(text)
        data_lines = [
            l for l in result.text.split("\n")
            if l.startswith("|") and "---" not in l and "_pivot" not in l
        ]
        # First expanded row should be A (leftmost), then B
        first_cells = [c.strip() for c in data_lines[0].strip("|").split("|")]
        second_cells = [c.strip() for c in data_lines[1].strip("|").split("|")]
        assert first_cells[0] == "A"
        assert second_cells[0] == "B"


class TestUnchangedWhenNotPivoted:
    def test_flat_table_unchanged(self):
        text = "| Region | Area | Yield |\n|---|---|---|\n| A | 100 | 50 |"
        result = unpivot_pipe_table(text)
        assert not result.was_transformed
        assert result.text == text

    def test_non_table_text_unchanged(self):
        text = "Just some plain text"
        result = unpivot_pipe_table(text)
        assert not result.was_transformed
        assert result.text == text


class TestCustomPivotColumnName:
    def test_custom_name(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |"
        )
        result = unpivot_pipe_table(text, pivot_column_name="Crop")
        headers = [c.strip() for c in result.text.split("\n")[0].strip("|").split("|")]
        assert headers[0] == "Crop"
        assert "_pivot" not in result.text


# ===========================================================================
# Integration-like tests (real-world patterns)
# ===========================================================================


class TestRussianPlantingPattern:
    """Simulates the June DOCX Table 4 pattern with Th.ha. prefix cols."""

    def test_two_crop_groups_with_non_matching_prefix(self):
        text = (
            "| Th.ha. / Region | Th.ha. / MOA 2024 | spring crops / MOA Target 2025 | spring crops / 2025 "
            "| spring grain / MOA Target 2025 | spring grain / 2025 |\n"
            "|---|---|---|---|---|---|\n"
            "**CENTRAL**\n"
            "| Belgorod | 800 | 100 | 90 | 50 | 45 |\n"
            "| Kursk | 700 | 120 | 110 | 60 | 55 |\n"
            "|| 1500 | 220 | 200 | 110 | 100 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        assert result.detection is not None
        assert result.detection.is_pivoted

        # Th.ha. columns should be shared
        prefixes = {g.prefix for g in result.detection.groups}
        assert "spring crops" in prefixes
        assert "spring grain" in prefixes
        assert "Th.ha." not in prefixes

        # Check output structure
        lines = result.text.split("\n")
        header = lines[0]
        assert "_pivot" in header
        assert "Th.ha. / Region" in header
        assert "Th.ha. / MOA 2024" in header
        assert "MOA Target 2025" in header
        assert "2025" in header

        # Section label preserved
        assert "**CENTRAL**" in result.text

        # Count data rows: 3 original rows (2 data + 1 agg) × 2 groups = 6
        data_lines = [
            l for l in lines
            if l.startswith("|") and "---" not in l and "_pivot" not in l
        ]
        assert len(data_lines) == 6

        # Verify crop names in pivot column
        all_pivots = []
        for dl in data_lines:
            cells = [c.strip() for c in dl.strip("|").split("|")]
            all_pivots.append(cells[0])
        assert all_pivots.count("spring crops") == 3
        assert all_pivots.count("spring grain") == 3


class TestRussianPlantingUnequalGroups:
    """spring crops has more sub-columns than spring grain."""

    def test_unequal_groups_intersect(self):
        text = (
            "| Region | A / Target | A / 2025 | A / % | B / Target | B / 2025 |\n"
            "|---|---|---|---|---|---|\n"
            "| r1 | 100 | 90 | 90% | 50 | 45 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        # Target and 2025 match; % is only in A
        assert "Target" in result.detection.measure_labels
        assert "2025" in result.detection.measure_labels
        # % should NOT be in measure_labels since it's only in A
        assert "%" not in result.detection.measure_labels


class TestACEAMotorizationPattern:
    """3 motorization groups with common suffixes."""

    def test_three_motorization_groups(self):
        text = (
            "| Market | BEV / Dec-25 | BEV / % | BEV / YTD | "
            "PHEV / Dec-25 | PHEV / % | PHEV / YTD | "
            "HEV / Dec-25 | HEV / % | HEV / YTD |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n"
            "| Germany | 1000 | 15 | 12000 | 500 | 8 | 6000 | 300 | 5 | 3600 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        assert len(result.detection.groups) == 3
        prefixes = {g.prefix for g in result.detection.groups}
        assert prefixes == {"BEV", "PHEV", "HEV"}

        # 1 row × 3 groups = 3 data rows
        data_lines = [
            l for l in result.text.split("\n")
            if l.startswith("|") and "---" not in l and "_pivot" not in l
        ]
        assert len(data_lines) == 3


class TestSingleCropNotPivoted:
    def test_single_prefix_stays_flat(self):
        text = (
            "| Region | CORN / Target | CORN / 2025 |\n"
            "|---|---|---|\n"
            "| Iowa | 100 | 95 |"
        )
        result = unpivot_pipe_table(text)
        assert not result.was_transformed
        assert result.text == text


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_string(self):
        result = unpivot_pipe_table("")
        assert not result.was_transformed
        assert result.text == ""

    def test_only_header_no_data(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|"
        )
        result = unpivot_pipe_table(text)
        # Should detect as pivoted but produce no data rows
        if result.was_transformed:
            data_lines = [
                l for l in result.text.split("\n")
                if l.startswith("|") and "---" not in l and "_pivot" not in l
            ]
            assert len(data_lines) == 0

    def test_empty_cells_in_data(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 |  | 2 | 3 |  |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        data_lines = [
            l for l in result.text.split("\n")
            if l.startswith("|") and "---" not in l and "_pivot" not in l
        ]
        assert len(data_lines) == 2

    def test_trailing_newline_preserved(self):
        text = (
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |\n"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        assert result.text.endswith("\n")

    def test_multiple_headings_and_text_before_table(self):
        text = (
            "## Table 4\n"
            "Some descriptive text\n"
            "| Region | A / X | A / Y | B / X | B / Y |\n"
            "|---|---|---|---|---|\n"
            "| r1 | 1 | 2 | 3 | 4 |"
        )
        result = unpivot_pipe_table(text)
        assert result.was_transformed
        assert "## Table 4" in result.text
        assert "Some descriptive text" in result.text
