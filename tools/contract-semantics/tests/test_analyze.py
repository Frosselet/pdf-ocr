"""Tests for contract_semantics.analyze — document structural profiling."""

from __future__ import annotations

import pytest

from contract_semantics.analyze import (
    AlignedColumn,
    ColumnProfile,
    DocumentProfile,
    MetadataCandidate,
    MultiDocumentProfile,
    TableGroup,
    TableProfile,
    _extract_non_table_text,
    _extract_section_labels,
    _extract_unit_from_header,
    _infer_column_type,
    _profile_table_from_compressed,
    format_analysis_report,
    merge_profiles,
)


# ---------------------------------------------------------------------------
# _infer_column_type
# ---------------------------------------------------------------------------


class TestInferColumnType:
    def test_number_all_integer(self):
        values = ["100", "200", "300"]
        assert _infer_column_type(values, "number") == "int"

    def test_number_with_decimals(self):
        values = ["100.5", "200.3", "300.7"]
        assert _infer_column_type(values, "number") == "float"

    def test_number_with_commas_integer(self):
        values = ["1,000", "2,000", "3,000"]
        assert _infer_column_type(values, "number") == "int"

    def test_string_type(self):
        values = ["hello", "world"]
        assert _infer_column_type(values, "string") == "string"

    def test_date_maps_to_string(self):
        values = ["2025-01-01", "2025-01-02"]
        assert _infer_column_type(values, "date") == "string"

    def test_enum_maps_to_string(self):
        values = ["A", "B", "C"]
        assert _infer_column_type(values, "enum") == "string"

    def test_empty_values_default(self):
        assert _infer_column_type([], "number") == "float"


# ---------------------------------------------------------------------------
# _extract_unit_from_header
# ---------------------------------------------------------------------------


class TestExtractUnitFromHeader:
    def test_parenthesized_unit(self):
        units = _extract_unit_from_header("Quantity (tonnes)")
        assert "tonnes" in units

    def test_comma_separated_unit(self):
        units = _extract_unit_from_header("Area harvested,Th.ha.")
        assert "Th.ha." in units

    def test_no_unit(self):
        units = _extract_unit_from_header("Region")
        assert units == []

    def test_currency_symbol(self):
        units = _extract_unit_from_header("Revenue ($)")
        assert "$" in units or any("$" in u for u in units)


# ---------------------------------------------------------------------------
# _extract_section_labels
# ---------------------------------------------------------------------------


class TestExtractSectionLabels:
    def test_bold_labels(self):
        md = "| A | B |\n| --- | --- |\n**SECTION ONE**\n| 1 | 2 |"
        labels = _extract_section_labels(md)
        assert "SECTION ONE" in labels

    def test_heading_labels(self):
        md = "## My Section\n| A | B |\n| --- | --- |\n| 1 | 2 |"
        labels = _extract_section_labels(md)
        assert "My Section" in labels

    def test_no_labels(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        labels = _extract_section_labels(md)
        assert labels == []


# ---------------------------------------------------------------------------
# _extract_non_table_text
# ---------------------------------------------------------------------------


class TestExtractNonTableText:
    def test_filters_pipe_rows(self):
        text = "Some heading\n| A | B |\n| --- | --- |\n| 1 | 2 |\nFooter text"
        result = _extract_non_table_text(text)
        assert "Some heading" in result
        assert "Footer text" in result
        assert "| A |" not in result


# ---------------------------------------------------------------------------
# _profile_table_from_compressed
# ---------------------------------------------------------------------------


class TestProfileTableFromCompressed:
    def test_basic_flat_table(self):
        md = "| Name | Age | Score |\n| --- | --- | --- |\n| Alice | 30 | 95.5 |\n| Bob | 25 | 87.3 |"
        meta = {"table_index": 0, "title": None, "row_count": 2, "col_count": 3}
        tp = _profile_table_from_compressed(md, meta, "test.docx", 0)

        assert tp.col_count == 3
        assert tp.row_count == 2
        assert len(tp.column_profiles) == 3
        assert tp.column_profiles[0].header_text == "Name"
        assert tp.column_profiles[0].inferred_type == "string"
        assert tp.layout == "flat"

    def test_year_column_detection(self):
        md = "| Region | 2024 | 2025 |\n| --- | --- | --- |\n| Moscow | 100 | 200 |"
        meta = {"table_index": 0, "title": None, "row_count": 1, "col_count": 3}
        tp = _profile_table_from_compressed(md, meta, "test.docx", 0)

        assert tp.column_profiles[1].year_detected is True
        assert tp.column_profiles[2].year_detected is True
        assert tp.column_profiles[0].year_detected is False

    def test_title_from_metadata(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        meta = {"table_index": 0, "title": "Harvest Data", "row_count": 1, "col_count": 2}
        tp = _profile_table_from_compressed(md, meta, "test.docx", 0)
        assert tp.title == "Harvest Data"

    def test_column_unique_counts(self):
        md = (
            "| Category | Value |\n| --- | --- |\n"
            "| A | 10 |\n| B | 20 |\n| A | 30 |\n| C | 40 |"
        )
        meta = {"table_index": 0, "title": None, "row_count": 4, "col_count": 2}
        tp = _profile_table_from_compressed(md, meta, "test.docx", 0)

        cat_col = tp.column_profiles[0]
        assert cat_col.unique_count == 3  # A, B, C
        assert cat_col.total_count == 4


# ---------------------------------------------------------------------------
# merge_profiles
# ---------------------------------------------------------------------------


class TestMergeProfiles:
    def _make_profile(self, doc_path: str, tables: list[TableProfile]) -> DocumentProfile:
        return DocumentProfile(
            doc_path=doc_path,
            doc_format="pdf",
            tables=tables,
        )

    def _make_table(self, doc_path: str, headers: list[str], rows: int = 10) -> TableProfile:
        from contract_semantics.analyze import _YEAR_RE

        col_profiles = [
            ColumnProfile(
                header_text=h,
                inferred_type="string",
                year_detected=bool(_YEAR_RE.fullmatch(h.strip())),
            )
            for h in headers
        ]
        tokens = {w.lower() for h in headers for w in h.split() if w.lower()}
        return TableProfile(
            source_document=doc_path,
            table_index=0,
            column_profiles=col_profiles,
            row_count=rows,
            col_count=len(headers),
            header_tokens=tokens,
        )

    def test_single_document(self):
        tp = self._make_table("doc1.pdf", ["Name", "Value"])
        profile = self._make_profile("doc1.pdf", [tp])

        multi = merge_profiles([profile])
        assert len(multi.table_groups) == 1
        assert len(multi.table_groups[0].aligned_columns) == 2

    def test_empty_profiles(self):
        multi = merge_profiles([])
        assert multi.table_groups == []

    def test_two_similar_documents(self):
        tp1 = self._make_table("doc1.pdf", ["Vessel", "ETA", "Port"])
        tp2 = self._make_table("doc2.pdf", ["Vessel Name", "ETA", "Load Port"])

        p1 = self._make_profile("doc1.pdf", [tp1])
        p2 = self._make_profile("doc2.pdf", [tp2])

        multi = merge_profiles([p1, p2])
        # Tables should be grouped together (similar structure)
        assert len(multi.table_groups) >= 1

    def test_temporal_candidates_merged(self):
        mc1 = MetadataCandidate(text="As of Jan 1, 2025", pattern_name="as_of_date", captured_value="Jan 1, 2025")
        mc2 = MetadataCandidate(text="As of Feb 1, 2025", pattern_name="as_of_date", captured_value="Feb 1, 2025")

        p1 = DocumentProfile(doc_path="doc1.pdf", doc_format="pdf", temporal_candidates=[mc1])
        p2 = DocumentProfile(doc_path="doc2.pdf", doc_format="pdf", temporal_candidates=[mc2])

        multi = merge_profiles([p1, p2])
        assert len(multi.all_temporal_candidates) == 2


# ---------------------------------------------------------------------------
# format_analysis_report
# ---------------------------------------------------------------------------


class TestFormatAnalysisReport:
    def test_single_document_report(self):
        tp = TableProfile(
            source_document="test.pdf",
            table_index=0,
            title="Sales Data",
            layout="flat",
            column_profiles=[
                ColumnProfile(header_text="Product", inferred_type="string", unique_count=5, total_count=10),
                ColumnProfile(header_text="Revenue", inferred_type="float", unique_count=10, total_count=10),
            ],
            row_count=10,
            col_count=2,
        )
        profile = DocumentProfile(
            doc_path="test.pdf",
            doc_format="pdf",
            tables=[tp],
        )
        report = format_analysis_report(profile)
        assert "test.pdf" in report
        assert "Tables detected: 1" in report
        assert "Product" in report
        assert "Revenue" in report

    def test_multi_document_report(self):
        group = TableGroup(
            group_id=0,
            tables=[],
            aligned_columns=[
                AlignedColumn(
                    canonical_header="Vessel",
                    all_headers=["Vessel", "Ship Name"],
                    inferred_type="string",
                ),
            ],
            common_tokens={"vessel"},
        )
        multi = MultiDocumentProfile(
            doc_paths=["a.pdf", "b.pdf"],
            document_profiles=[
                DocumentProfile(doc_path="a.pdf", doc_format="pdf"),
                DocumentProfile(doc_path="b.pdf", doc_format="pdf"),
            ],
            table_groups=[group],
        )
        report = format_analysis_report(multi)
        assert "Documents analyzed: 2" in report
        assert "Vessel" in report
        assert "2 variant(s)" in report
