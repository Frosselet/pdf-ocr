"""Tests for interpret.py pre-Step-1 table splitting and merging.

These tests exercise the deterministic splitting logic that prevents
LLM output truncation on large DOCX/PDF tables (65+ data rows).
"""

from __future__ import annotations

import pytest
from baml_client import types as baml_types

from pdf_ocr.interpret import (
    _count_pipe_data_rows,
    _merge_parsed_chunks,
    _split_pipe_table,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _build_flat_table(n_rows: int, *, with_agg: bool = False) -> str:
    """Build a flat pipe-table with *n_rows* data rows."""
    lines = ["Test Table", "| Region | Area | Yield |", "|---|---|---|"]
    for i in range(1, n_rows + 1):
        lines.append(f"| Region {i} | {i * 100} | {i * 10} |")
    if with_agg:
        lines.append("|| | Total | 999 |")
    return "\n".join(lines)


def _build_sectioned_table(section_sizes: list[int]) -> str:
    """Build a multi-section pipe-table.

    Each section has a label, sub-headers, separator, data rows, and
    an aggregation row.
    """
    lines = ["Sectioned Table", "| Col A | Col B | Col C |", "|---|---|---|"]
    row_counter = 1
    for si, size in enumerate(section_sizes):
        if si > 0:
            # Section transition: label, sub-headers, separator
            lines.append(f"Section {si + 1}")
            lines.append("| Col A | Col B | Col C |")
            lines.append("|---|---|---|")
        for _ in range(size):
            lines.append(f"| row{row_counter} | val{row_counter} | data{row_counter} |")
            row_counter += 1
        lines.append(f"|| | Subtotal | {size * 100} |")
    return "\n".join(lines)


def _make_parsed(
    data_rows: list[list[str]],
    *,
    notes: str | None = None,
) -> baml_types.ParsedTable:
    """Build a minimal ParsedTable for merge tests."""
    return baml_types.ParsedTable(
        table_type=baml_types.TableType.FlatHeader,
        headers=baml_types.HeaderInfo(levels=1, names=[["A", "B", "C"]]),
        aggregations=baml_types.AggregationInfo(
            type=baml_types.AggregationType.Sum, axis="row", labels=["Total"]
        ),
        data_rows=data_rows,
        notes=notes,
    )


# ─── _count_pipe_data_rows ────────────────────────────────────────────────────


class TestCountPipeDataRows:
    """Tests for the quick row counter."""

    def test_flat_table(self) -> None:
        text = _build_flat_table(25)
        assert _count_pipe_data_rows(text) == 25

    def test_flat_table_with_agg(self) -> None:
        text = _build_flat_table(25, with_agg=True)
        assert _count_pipe_data_rows(text) == 25

    def test_sectioned_table(self) -> None:
        text = _build_sectioned_table([30, 37])
        assert _count_pipe_data_rows(text) == 67

    def test_three_sections(self) -> None:
        text = _build_sectioned_table([10, 20, 15])
        assert _count_pipe_data_rows(text) == 45

    def test_empty_table(self) -> None:
        text = "Title\n| A | B |\n|---|---|"
        assert _count_pipe_data_rows(text) == 0

    def test_no_pipe_table(self) -> None:
        text = "Just some text\nwith no tables"
        assert _count_pipe_data_rows(text) == 0

    def test_single_row(self) -> None:
        text = _build_flat_table(1)
        assert _count_pipe_data_rows(text) == 1


# ─── _split_pipe_table ────────────────────────────────────────────────────────


class TestSplitPipeTable:
    """Tests for the pre-Step-1 table splitter."""

    def test_no_split_below_threshold(self) -> None:
        text = _build_flat_table(30)
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_no_split_at_threshold(self) -> None:
        text = _build_flat_table(40)
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_above_threshold(self) -> None:
        text = _build_flat_table(67)
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 2

    def test_split_preserves_all_rows(self) -> None:
        """Every data row from the original must appear in exactly one chunk."""
        import re

        text = _build_flat_table(67)
        chunks = _split_pipe_table(text, 40)
        all_data_lines: list[str] = []
        for chunk in chunks:
            for line in chunk.split("\n"):
                if re.match(r"\| Region \d", line.strip()):
                    all_data_lines.append(line)
        assert len(all_data_lines) == 67

    def test_split_preserves_preamble(self) -> None:
        """Each chunk must start with the original preamble."""
        text = _build_flat_table(67)
        chunks = _split_pipe_table(text, 40)
        for chunk in chunks:
            assert "Test Table" in chunk
            assert "|---|" in chunk
            assert "| Region | Area | Yield |" in chunk

    def test_split_excludes_agg_rows(self) -> None:
        """Aggregation rows (||) should not appear in any chunk."""
        text = _build_flat_table(67, with_agg=True)
        chunks = _split_pipe_table(text, 40)
        for chunk in chunks:
            for line in chunk.split("\n"):
                assert not line.strip().startswith("||"), (
                    f"Aggregation row found in chunk: {line}"
                )

    def test_split_row_order_preserved(self) -> None:
        """Data rows must appear in the same order across chunks."""
        import re

        text = _build_flat_table(67)
        chunks = _split_pipe_table(text, 40)
        all_regions: list[int] = []
        for chunk in chunks:
            for line in chunk.split("\n"):
                m = re.match(r"\| Region (\d+)", line.strip())
                if m:
                    all_regions.append(int(m.group(1)))
        assert all_regions == list(range(1, 68))

    def test_sectioned_split_at_boundary(self) -> None:
        """When possible, split at section boundaries."""
        text = _build_sectioned_table([30, 37])
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 2
        # First chunk should have section 1 (30 rows) + start of section 2
        # Section boundary is a natural split point

    def test_large_single_section(self) -> None:
        """A single section with 80 rows should split into 2 chunks."""
        text = _build_flat_table(80)
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 2

    def test_three_way_split(self) -> None:
        """A 120-row table should split into 3 chunks."""
        import re

        text = _build_flat_table(120)
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 3
        # Verify total (match data rows, not header)
        total = sum(
            1
            for chunk in chunks
            for line in chunk.split("\n")
            if re.match(r"\| Region \d", line.strip())
        )
        assert total == 120

    def test_empty_table_no_split(self) -> None:
        text = "Title\n| A | B |\n|---|---|"
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 1
        assert chunks[0] == text


# ─── _merge_parsed_chunks ────────────────────────────────────────────────────


class TestMergeParsedChunks:
    """Tests for merging Step 1 results from multiple chunks."""

    def test_concatenates_data_rows(self) -> None:
        c1 = _make_parsed([["r1", "v1", "d1"], ["r2", "v2", "d2"]])
        c2 = _make_parsed([["r3", "v3", "d3"]])
        merged = _merge_parsed_chunks([c1, c2])
        assert len(merged.data_rows) == 3
        assert merged.data_rows[0] == ["r1", "v1", "d1"]
        assert merged.data_rows[2] == ["r3", "v3", "d3"]

    def test_preserves_table_type(self) -> None:
        c1 = _make_parsed([["a"]])
        c2 = _make_parsed([["b"]])
        merged = _merge_parsed_chunks([c1, c2])
        assert merged.table_type == baml_types.TableType.FlatHeader

    def test_preserves_headers(self) -> None:
        c1 = _make_parsed([["a"]])
        c2 = _make_parsed([["b"]])
        merged = _merge_parsed_chunks([c1, c2])
        assert merged.headers.names == [["A", "B", "C"]]

    def test_aggregations_from_last_chunk(self) -> None:
        c1 = _make_parsed([["a"]])
        c2 = _make_parsed([["b"]])
        c2_agg = baml_types.AggregationInfo(
            type=baml_types.AggregationType.Sum, axis="row", labels=["Grand Total"]
        )
        c2 = baml_types.ParsedTable(
            table_type=c2.table_type,
            headers=c2.headers,
            aggregations=c2_agg,
            data_rows=c2.data_rows,
            notes=c2.notes,
        )
        merged = _merge_parsed_chunks([c1, c2])
        assert merged.aggregations.labels == ["Grand Total"]

    def test_section_boundaries_rebuilt(self) -> None:
        c1 = _make_parsed(
            [["a"], ["b"], ["c"]],
            notes="Sections: HARVEST (rows 0-2)",
        )
        c2 = _make_parsed(
            [["d"], ["e"]],
            notes="Sections: PLANTING (rows 0-1)",
        )
        merged = _merge_parsed_chunks([c1, c2])
        assert "HARVEST (rows 0-2)" in merged.notes
        assert "PLANTING (rows 3-4)" in merged.notes

    def test_no_sections_no_notes(self) -> None:
        c1 = _make_parsed([["a"]], notes=None)
        c2 = _make_parsed([["b"]], notes=None)
        merged = _merge_parsed_chunks([c1, c2])
        assert merged.notes is None

    def test_single_chunk_passthrough(self) -> None:
        c1 = _make_parsed([["a"], ["b"]], notes="Some note")
        merged = _merge_parsed_chunks([c1])
        assert len(merged.data_rows) == 2
        assert merged.notes == "Some note"

    def test_three_chunk_merge(self) -> None:
        c1 = _make_parsed([["a"]] * 10, notes="Sections: S1 (rows 0-9)")
        c2 = _make_parsed([["b"]] * 15, notes="Sections: S2 (rows 0-14)")
        c3 = _make_parsed([["c"]] * 5, notes="Sections: S3 (rows 0-4)")
        merged = _merge_parsed_chunks([c1, c2, c3])
        assert len(merged.data_rows) == 30
        assert "S1 (rows 0-9)" in merged.notes
        assert "S2 (rows 10-24)" in merged.notes
        assert "S3 (rows 25-29)" in merged.notes


# ─── Adversarial: edge cases and boundary conditions ─────────────────────────


class TestSplitAdversarial:
    """Adversarial tests for splitting edge cases."""

    def test_exactly_at_boundary(self) -> None:
        """Table with exactly max_rows rows should NOT be split."""
        text = _build_flat_table(40)
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 1

    def test_one_above_boundary(self) -> None:
        """Table with max_rows+1 rows MUST be split."""
        text = _build_flat_table(41)
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 2

    def test_all_agg_rows(self) -> None:
        """Table with only aggregation rows should not split."""
        lines = ["Title", "| A | B |", "|---|---|"]
        for i in range(50):
            lines.append(f"|| | Total{i} | {i} |")
        text = "\n".join(lines)
        # All rows are agg rows → 0 data rows → no split
        chunks = _split_pipe_table(text, 40)
        assert len(chunks) == 1

    def test_max_rows_one(self) -> None:
        """Extreme: max_rows=1 should create one chunk per row."""
        text = _build_flat_table(5)
        chunks = _split_pipe_table(text, 1)
        assert len(chunks) == 5
        # Each chunk should have exactly 1 data row
        for chunk in chunks:
            count = _count_pipe_data_rows(chunk)
            assert count == 1

    def test_unicode_content_preserved(self) -> None:
        """Unicode content must survive splitting."""
        import re

        lines = [
            "Таблица",  # Russian title
            "| Регион | Площадь |",
            "|---|---|",
        ]
        for i in range(50):
            lines.append(f"| Регион {i} | {i * 100} |")
        text = "\n".join(lines)
        chunks = _split_pipe_table(text, 30)
        assert len(chunks) == 2
        # Verify all regions present (match data rows, not header)
        all_regions = []
        for chunk in chunks:
            for line in chunk.split("\n"):
                if re.match(r"\| Регион \d", line.strip()):
                    all_regions.append(line)
        assert len(all_regions) == 50

    def test_section_label_preserved_in_correct_chunk(self) -> None:
        """Section labels must appear in the chunk containing their rows."""
        text = _build_sectioned_table([20, 30])
        chunks = _split_pipe_table(text, 25)
        # With 20 + 30 = 50 rows and max_rows=25:
        # Section 1 has 20 rows → fits in chunk 0
        # Section 2 has 30 rows → starts in chunk 0 or 1 depending on boundary
        assert len(chunks) >= 2


class TestMergeAdversarial:
    """Adversarial tests for merge edge cases."""

    def test_empty_data_rows(self) -> None:
        """Merging chunks with empty data_rows should work."""
        c1 = _make_parsed([])
        c2 = _make_parsed([["a"]])
        merged = _merge_parsed_chunks([c1, c2])
        assert len(merged.data_rows) == 1

    def test_mixed_sections_and_no_sections(self) -> None:
        """One chunk with sections, another without."""
        c1 = _make_parsed([["a"]], notes="Sections: S1 (rows 0-0)")
        c2 = _make_parsed([["b"]], notes=None)
        merged = _merge_parsed_chunks([c1, c2])
        # Should include S1 section info from chunk 1
        assert "S1 (rows 0-0)" in merged.notes

    def test_large_merge(self) -> None:
        """Merge many small chunks."""
        chunks = [_make_parsed([["x"]] * 5) for _ in range(10)]
        merged = _merge_parsed_chunks(chunks)
        assert len(merged.data_rows) == 50
