"""Tests for the format-agnostic table classification module.

Covers:
A. Unit tests for _parse_pipe_header and _tokenize_header_text
B. Integration tests with hand-crafted tuples
C. Adversarial tests probing failure modes and boundary conditions
D. PDF classification via StructuredTable.to_compressed()

Run with: uv run pytest tests/test_classify.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_ocr.classify import (
    _compute_similarity,
    _keyword_matches,
    _parse_pipe_header,
    _tokenize_header_text,
    classify_tables,
)


# =========================================================================
# A. Unit tests
# =========================================================================


class TestParseHeader:
    """_parse_pipe_header() unit tests."""

    def test_normal_table(self) -> None:
        md = "| Region | Area | Yield |\n| --- | --- | --- |\n| A | 10 | 20 |"
        assert _parse_pipe_header(md) == ["Region", "Area", "Yield"]

    def test_table_with_title(self) -> None:
        md = "## WHEAT\n\n| Region | Area |\n| --- | --- |"
        assert _parse_pipe_header(md) == ["Region", "Area"]

    def test_empty_string(self) -> None:
        assert _parse_pipe_header("") == []

    def test_no_pipe_rows(self) -> None:
        assert _parse_pipe_header("Some prose text.\nNo tables here.") == []

    def test_multiple_pipe_rows_takes_first(self) -> None:
        md = "| A | B |\n| C | D |\n| --- | --- |"
        assert _parse_pipe_header(md) == ["A", "B"]

    def test_only_separator(self) -> None:
        md = "| --- | --- |"
        assert _parse_pipe_header(md) == []

    def test_strips_whitespace(self) -> None:
        md = "|  Region  |  Area  |"
        assert _parse_pipe_header(md) == ["Region", "Area"]


class TestTokenizeHeaderText:
    """_tokenize_header_text() unit tests."""

    def test_filters_years(self) -> None:
        tokens = _tokenize_header_text("Area 2025 Yield 2024")
        assert "2025" not in tokens
        assert "2024" not in tokens
        assert "area" in tokens
        assert "yield" in tokens

    def test_filters_pure_numbers(self) -> None:
        tokens = _tokenize_header_text("Revenue 1,234 Units")
        assert "1,234" not in tokens
        assert "revenue" in tokens
        assert "units" in tokens

    def test_strips_parentheses(self) -> None:
        tokens = _tokenize_header_text("(total)")
        assert "total" in tokens

    def test_strips_dashes(self) -> None:
        tokens = _tokenize_header_text("year–over–year")
        assert "year" in tokens
        assert "over" in tokens

    def test_strips_slashes(self) -> None:
        tokens = _tokenize_header_text("/price/")
        assert "price" in tokens

    def test_strips_brackets(self) -> None:
        tokens = _tokenize_header_text("[Price]")
        assert "price" in tokens

    def test_strips_currency(self) -> None:
        tokens = _tokenize_header_text("$Amount")
        assert "amount" in tokens

    def test_preserves_meaningful_tokens(self) -> None:
        tokens = _tokenize_header_text("Region Area harvested Yield collected")
        assert tokens == {"region", "area", "harvested", "yield", "collected"}


# =========================================================================
# B. Integration tests (hand-crafted tuples)
# =========================================================================


class TestClassifyTablesIntegration:
    """Integration tests with synthetic tuples."""

    def test_single_table_matches_category(self) -> None:
        tables = [
            ("| Export | Volume |\n| --- | --- |\n| A | 10 |",
             {"table_index": 0, "title": None, "row_count": 1, "col_count": 2}),
        ]
        result = classify_tables(tables, {"trade": ["export", "shipment"]})
        assert result[0]["category"] == "trade"

    def test_multiple_tables_different_categories(self) -> None:
        tables = [
            ("| Port | Vessel |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
            ("| Yield | Area harvested |\n| --- | --- |",
             {"table_index": 1, "title": None, "row_count": 5, "col_count": 2}),
        ]
        cats = {
            "shipping": ["port", "vessel"],
            "harvest": ["yield", "area harvested"],
        }
        result = classify_tables(tables, cats)
        assert result[0]["category"] == "shipping"
        assert result[1]["category"] == "harvest"

    def test_min_data_rows_filtering(self) -> None:
        tables = [
            ("| Export | Volume |\n| --- | --- |\n| A | 1 |",
             {"table_index": 0, "title": None, "row_count": 1, "col_count": 2}),
        ]
        result = classify_tables(tables, {"trade": ["export"]}, min_data_rows=5)
        assert result[0]["category"] == "other"

    def test_no_match_gives_other(self) -> None:
        tables = [
            ("| Foo | Bar |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
        ]
        result = classify_tables(tables, {"trade": ["export"]})
        assert result[0]["category"] == "other"

    def test_empty_categories_gives_other(self) -> None:
        tables = [
            ("| A | B |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
        ]
        result = classify_tables(tables, {})
        assert result[0]["category"] == "other"

    def test_title_only_classification(self) -> None:
        """Keywords in title but not in column headers should still match."""
        tables = [
            ("## EXPORT SUMMARY\n\n| Region | Volume |\n| --- | --- |",
             {"table_index": 0, "title": "EXPORT SUMMARY", "row_count": 5, "col_count": 2}),
        ]
        result = classify_tables(tables, {"trade": ["export"]})
        assert result[0]["category"] == "trade"

    def test_similarity_propagation(self) -> None:
        """Table with same structure but no keywords gets propagated."""
        tables = [
            ("| Export | Volume | Port |\n| --- | --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 3}),
            ("| Item | Amount | Location |\n| --- | --- | --- |",
             {"table_index": 1, "title": None, "row_count": 5, "col_count": 3}),
        ]
        cats = {"trade": ["export", "port"]}
        # Without propagation
        result_off = classify_tables(tables, cats, propagate=False)
        assert result_off[1]["category"] == "other"
        # With propagation — similar structure should inherit
        result_on = classify_tables(tables, cats, propagate=True, propagate_threshold=0.0)
        assert result_on[1]["category"] == "trade"


# =========================================================================
# C. Adversarial tests
# =========================================================================


class TestAdversarialWordBoundary:
    """Word boundary adversaries."""

    def test_keyword_not_in_superstring(self) -> None:
        """'port' must NOT match 'transport', 'important', 'export'."""
        assert _keyword_matches("port", "transport costs") is False
        assert _keyword_matches("port", "important data") is False
        assert _keyword_matches("port", "export values") is False

    def test_keyword_not_as_prefix(self) -> None:
        assert _keyword_matches("market", "marketplace") is False

    def test_keyword_not_as_suffix(self) -> None:
        assert _keyword_matches("rice", "price") is False

    def test_suffix_outside_tolerance(self) -> None:
        """'exporter' has suffix '-er' which is NOT in the tolerance list."""
        assert _keyword_matches("export", "exporter data") is False

    def test_double_suffix_no_match(self) -> None:
        assert _keyword_matches("export", "exportings data") is False


class TestAdversarialCompoundHeader:
    """Compound header separator adversaries."""

    def test_keyword_spanning_compound_separator(self) -> None:
        """'area harvested' MUST match column name 'Area / harvested'."""
        tables = [
            ("| Area / harvested | Yield / 2025 |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
        ]
        result = classify_tables(tables, {"crop": ["area harvested"]})
        assert result[0]["category"] == "crop"

    def test_three_level_compound_header(self) -> None:
        """'total volume' MUST match 'Total / Volume / 2025'."""
        tables = [
            ("| Total / Volume / 2025 |\n| --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 1}),
        ]
        result = classify_tables(tables, {"metrics": ["total volume"]})
        assert result[0]["category"] == "metrics"


class TestAdversarialCrossCategory:
    """Cross-category contamination."""

    def test_tie_breaking_by_dict_order(self) -> None:
        """Two categories score equally → first in dict wins."""
        tables = [
            ("| Export | Yield |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
        ]
        cats = {
            "alpha": ["export"],
            "beta": ["yield"],
        }
        result = classify_tables(tables, cats)
        # Both score 1; first dict key wins
        assert result[0]["category"] == "alpha"

    def test_mixed_keywords_highest_count_wins(self) -> None:
        """Category with most hits wins regardless of keyword list size."""
        tables = [
            ("| Export | Port | Vessel | Yield |\n| --- | --- | --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 4}),
        ]
        cats = {
            "harvest": ["yield"],
            "shipping": ["export", "port", "vessel"],
        }
        result = classify_tables(tables, cats)
        assert result[0]["category"] == "shipping"


class TestAdversarialScoring:
    """Scoring edge cases."""

    def test_repeated_column_names_no_double_count(self) -> None:
        """Same keyword matching 5 identical column names still scores 1."""
        tables = [
            ("| Export | Export | Export | Export | Export |\n| --- | --- | --- | --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 5}),
        ]
        cats = {
            "trade": ["export"],
            "other_cat": ["yield", "area"],
        }
        result = classify_tables(tables, cats)
        # "export" keyword matches once in header_text → score 1
        assert result[0]["category"] == "trade"

    def test_category_with_many_keywords_vs_few(self) -> None:
        """Raw hit count, not ratio, determines winner."""
        tables = [
            ("| Export | Port | Vessel |\n| --- | --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 3}),
        ]
        cats = {
            "small": ["export", "port"],  # 2 hits
            "large": ["vessel", "a", "b", "c", "d", "e", "f", "g", "h", "i"],  # 1 hit
        }
        result = classify_tables(tables, cats)
        assert result[0]["category"] == "small"


class TestAdversarialDegenerateInput:
    """Degenerate input adversaries."""

    def test_empty_markdown_string(self) -> None:
        tables = [("", {"table_index": 0, "title": None, "row_count": 0, "col_count": 0})]
        result = classify_tables(tables, {"trade": ["export"]})
        assert result[0]["category"] == "other"

    def test_no_pipe_rows_only_text(self) -> None:
        tables = [
            ("Some prose about exports.\nNo tables here.",
             {"table_index": 0, "title": None, "row_count": 0, "col_count": 0}),
        ]
        result = classify_tables(tables, {"trade": ["export"]})
        assert result[0]["category"] == "other"

    def test_all_numeric_headers(self) -> None:
        tables = [
            ("| 2024 | 2025 | 2026 |\n| --- | --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 3}),
        ]
        result = classify_tables(tables, {"years": ["2025"]})
        # After year filtering in tokenization and ` / ` collapse, header_text
        # is "2024 2025 2026" — keywords do match in raw header_text even if
        # tokens are sparse. The keyword regex still matches in the header_text.
        # But "2025" as a keyword would match in header_text via _keyword_matches.
        # Actually _keyword_matches uses word boundaries on ASCII letters, and
        # "2025" has no ASCII letters, so (?<![a-zA-Z]) and (?![a-zA-Z]) both
        # succeed on digits. The keyword "2025" will match "2025" in header text.
        assert result[0]["category"] == "years"

    def test_single_column_table(self) -> None:
        tables = [
            ("| Export |\n| --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 1}),
        ]
        result = classify_tables(tables, {"trade": ["export"]})
        assert result[0]["category"] == "trade"

    def test_zero_data_rows(self) -> None:
        tables = [
            ("| Export |\n| --- |",
             {"table_index": 0, "title": None, "row_count": 0, "col_count": 1}),
        ]
        result = classify_tables(tables, {"trade": ["export"]}, min_data_rows=1)
        assert result[0]["category"] == "other"

    def test_exactly_at_min_threshold(self) -> None:
        tables = [
            ("| Export |\n| --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 1}),
        ]
        result = classify_tables(tables, {"trade": ["export"]}, min_data_rows=5)
        assert result[0]["category"] == "trade"

    def test_one_below_min_threshold(self) -> None:
        tables = [
            ("| Export |\n| --- |",
             {"table_index": 0, "title": None, "row_count": 4, "col_count": 1}),
        ]
        result = classify_tables(tables, {"trade": ["export"]}, min_data_rows=5)
        assert result[0]["category"] == "other"


class TestAdversarialPropagation:
    """Propagation adversaries."""

    def test_propagation_threshold_zero(self) -> None:
        """threshold=0.0 → all 'other' tables get propagated."""
        tables = [
            ("| Export | Port |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
            ("| Foo | Bar |\n| --- | --- |",
             {"table_index": 1, "title": None, "row_count": 5, "col_count": 2}),
        ]
        result = classify_tables(
            tables, {"trade": ["export"]},
            propagate=True, propagate_threshold=0.0,
        )
        assert result[0]["category"] == "trade"
        assert result[1]["category"] == "trade"  # propagated at threshold=0

    def test_propagation_threshold_one(self) -> None:
        """threshold=1.0 → no propagation (nothing reaches 1.0 unless identical)."""
        tables = [
            ("| Export | Port |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
            ("| Foo | Bar |\n| --- | --- |",
             {"table_index": 1, "title": None, "row_count": 5, "col_count": 2}),
        ]
        result = classify_tables(
            tables, {"trade": ["export"]},
            propagate=True, propagate_threshold=1.0,
        )
        assert result[1]["category"] == "other"

    def test_propagation_wrong_structure(self) -> None:
        """Very different col count (2 vs 10) → stays 'other'."""
        tables = [
            ("| Export | Port |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
            ("| A | B | C | D | E | F | G | H | I | J |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
             {"table_index": 1, "title": None, "row_count": 5, "col_count": 10}),
        ]
        result = classify_tables(
            tables, {"trade": ["export"]},
            propagate=True, propagate_threshold=0.3,
        )
        assert result[1]["category"] == "other"

    def test_propagation_no_donors(self) -> None:
        """All tables are 'other' → no profiles → nothing changes."""
        tables = [
            ("| Foo | Bar |\n| --- | --- |",
             {"table_index": 0, "title": None, "row_count": 5, "col_count": 2}),
            ("| Baz | Qux |\n| --- | --- |",
             {"table_index": 1, "title": None, "row_count": 5, "col_count": 2}),
        ]
        result = classify_tables(
            tables, {"trade": ["export"]},
            propagate=True, propagate_threshold=0.0,
        )
        assert result[0]["category"] == "other"
        assert result[1]["category"] == "other"


class TestAdversarialMultilingual:
    """Multilingual adversaries."""

    def test_accented_suffix_boundary(self) -> None:
        """'export' matches in 'Exporté — résumé' (accented char is not ASCII letter)."""
        assert _keyword_matches("export", "Exporté — résumé") is True

    def test_cyrillic_lookalike_no_match(self) -> None:
        """Cyrillic 'еxроrt' (mixed scripts) should NOT match Latin 'export'."""
        # е = U+0435 (Cyrillic), x = U+0078 (Latin), р = U+0440 (Cyrillic),
        # о = U+043E (Cyrillic), r = U+0072 (Latin), t = U+0074 (Latin)
        cyrillic_mixed = "\u0435x\u0440\u043ert"
        assert _keyword_matches("export", cyrillic_mixed) is False

    def test_cjk_mixed_script(self) -> None:
        """'FOB' matches in 'FOB価格' (CJK chars are not ASCII letters)."""
        assert _keyword_matches("fob", "FOB価格") is True

    def test_german_umlaut_boundary(self) -> None:
        """'fur' should NOT match 'für' but 'export' matches in 'Exportübersicht'."""
        assert _keyword_matches("fur", "für Daten") is False
        assert _keyword_matches("export", "Exportübersicht") is True


class TestAdversarialPunctuation:
    """Punctuation stripping in _tokenize_header_text."""

    def test_en_dash_in_header(self) -> None:
        tokens = _tokenize_header_text("Year–over–year")
        assert "year" in tokens
        assert "over" in tokens

    def test_brackets_in_header(self) -> None:
        tokens = _tokenize_header_text("[Price]")
        assert "price" in tokens

    def test_currency_in_header(self) -> None:
        tokens = _tokenize_header_text("$Amount")
        assert "amount" in tokens

    def test_asterisk_footnote(self) -> None:
        tokens = _tokenize_header_text("Total*")
        assert "total" in tokens


# =========================================================================
# D. PDF classification via StructuredTable.to_compressed()
# =========================================================================


class TestPDFClassification:
    """Classify PDF tables via compress_spatial_text_structured → to_compressed."""

    def test_cbh_shipping_stem(self) -> None:
        """CBH Shipping Stem tables can be classified via the PDF path."""
        from pdf_ocr.compress import compress_spatial_text_structured

        pdf_path = Path("inputs/CBH Shipping Stem 26092025.pdf")
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")

        structured = compress_spatial_text_structured(str(pdf_path))
        compressed = [st.to_compressed() for st in structured]
        assert len(compressed) > 0

        cats = {
            "shipping": ["vessel", "port", "cargo", "grade", "tonnage"],
        }
        result = classify_tables(compressed, cats)
        assert len(result) > 0
        # At least one table should classify as shipping
        shipping = [r for r in result if r["category"] == "shipping"]
        assert len(shipping) > 0

    def test_to_compressed_round_trip(self) -> None:
        """to_compressed() produces valid markdown parseable by _parse_pipe_header."""
        from pdf_ocr.compress import StructuredTable, TableMetadata

        st = StructuredTable(
            metadata=TableMetadata(page_number=1, table_index=0, section_label="TEST"),
            column_names=["Region", "Area harvested / 2025", "Yield"],
            data=[["A", "100", "3.5"], ["B", "200", "4.0"]],
        )
        md, meta = st.to_compressed()
        assert meta["table_index"] == 0
        assert meta["title"] == "TEST"
        assert meta["row_count"] == 2
        assert meta["col_count"] == 3
        assert md.startswith("## TEST")
        cols = _parse_pipe_header(md)
        assert cols == ["Region", "Area harvested / 2025", "Yield"]

    def test_to_compressed_no_section_label(self) -> None:
        """to_compressed() without section_label omits title."""
        from pdf_ocr.compress import StructuredTable, TableMetadata

        st = StructuredTable(
            metadata=TableMetadata(page_number=1, table_index=0, section_label=None),
            column_names=["A", "B"],
            data=[["1", "2"]],
        )
        md, meta = st.to_compressed()
        assert meta["title"] is None
        assert not md.startswith("##")
        assert md.startswith("| A | B |")

    def test_to_compressed_short_rows_padded(self) -> None:
        """Data rows shorter than column_names are padded."""
        from pdf_ocr.compress import StructuredTable, TableMetadata

        st = StructuredTable(
            metadata=TableMetadata(page_number=1, table_index=0, section_label=None),
            column_names=["A", "B", "C"],
            data=[["1"]],  # short row
        )
        md, _ = st.to_compressed()
        data_line = [l for l in md.split("\n") if l.startswith("|") and "---" not in l][1]
        cells = [c.strip() for c in data_line.strip("|").split("|")]
        assert cells == ["1", "", ""]
