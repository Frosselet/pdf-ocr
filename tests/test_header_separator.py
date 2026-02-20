"""Tests for the ` / ` header separator convention.

Both ``build_column_names_from_headers()`` (heuristics.py) and
``_build_stacked_headers()`` (compress.py) should use fragment-level
deduplication with ` / ` joining instead of word-level dedup with spaces.
"""

from __future__ import annotations

from docpact.heuristics import build_column_names_from_headers


class TestBuildColumnNamesSlash:
    """Verify fragment-level dedup + ` / ` join."""

    def test_two_level_uses_slash(self) -> None:
        headers = [["Group", "Group", "Group"], ["Metric A", "Metric B", "Metric C"]]
        result = build_column_names_from_headers(headers)
        assert result == ["Group / Metric A", "Group / Metric B", "Group / Metric C"]

    def test_single_level_no_slash(self) -> None:
        headers = [["Name", "Age", "City"]]
        result = build_column_names_from_headers(headers)
        assert result == ["Name", "Age", "City"]

    def test_three_levels(self) -> None:
        headers = [["A", "A"], ["B", "B"], ["C1", "C2"]]
        result = build_column_names_from_headers(headers)
        assert result == ["A / B / C1", "A / B / C2"]

    def test_duplicate_dedup(self) -> None:
        """Consecutive identical fragments are deduplicated."""
        headers = [["X", "Y"], ["X", "Z"]]
        result = build_column_names_from_headers(headers)
        assert result == ["X", "Y / Z"]

    def test_empty_fragments_skipped(self) -> None:
        """Empty fragments don't produce spurious separators."""
        headers = [["", ""], ["M", "N"]]
        result = build_column_names_from_headers(headers)
        assert result == ["M", "N"]

    def test_single_column(self) -> None:
        headers = [["Total"]]
        result = build_column_names_from_headers(headers)
        assert result == ["Total"]

    def test_empty_input(self) -> None:
        assert build_column_names_from_headers([]) == []

    def test_whitespace_stripped(self) -> None:
        """Fragments with leading/trailing whitespace are stripped."""
        headers = [["  Group  ", "  Group  "], ["  Val  ", "  Count  "]]
        result = build_column_names_from_headers(headers)
        assert result == ["Group / Val", "Group / Count"]

    def test_numeric_fragments(self) -> None:
        """Numeric cell values are converted to strings."""
        headers = [[2024, 2024], ["Q1", "Q2"]]
        result = build_column_names_from_headers(headers)
        assert result == ["2024 / Q1", "2024 / Q2"]
