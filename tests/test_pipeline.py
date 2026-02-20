"""Tests for docpact.pipeline — async pipeline orchestration helpers.

All pure-Python, no LLM calls.  Tests that invoke interpretation use
contracts/schemas where the deterministic mapper resolves all columns,
so no API keys or LLM calls are needed.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pandas as pd
import pytest

from docpact.contracts import (
    ContractContext,
    OutputSpec,
    load_contract,
)
from docpact.interpret import CanonicalSchema, ColumnDef, UnpivotStrategy
from docpact.pipeline import (
    DocumentResult,
    compress_and_classify_async,
    interpret_output_async,
    process_document_async,
    save,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _synth(name: str) -> str:
    """Return path to a synthetic DOCX fixture."""
    return str(Path("inputs/docx/synthetic") / f"{name}.docx")


# ─── compress_and_classify_async ─────────────────────────────────────────────


class TestCompressAndClassifyAsync:
    """Tests for compress_and_classify_async()."""

    def test_compress_pdf_single_category(self):
        """Single-category contract skips classification; returns compressed text."""
        cc = load_contract("contracts/au_shipping_stem.json")
        result = asyncio.get_event_loop().run_until_complete(
            compress_and_classify_async(
                "inputs/2857439.pdf",
                cc.categories,
                cc.outputs,
                refine_headers=False,
            )
        )
        assert "vessels" in result
        assert isinstance(result["vessels"], str)
        assert len(result["vessels"]) > 100  # non-trivial text

    def test_compress_pdf_multi_category(self):
        """Multi-category contract uses structured extraction + classification."""
        # ACEA has a single category too, but we can test with a synthetic multi-cat
        # by creating a 2-category dict manually
        categories = {
            "shipping": ["vessel", "cargo", "port"],
            "other": ["nonexistent_keyword_xyz"],
        }
        cc = load_contract("contracts/au_shipping_stem.json")
        # Duplicate output spec under a second category
        specs = dict(cc.outputs)
        specs["other_out"] = OutputSpec(
            name="other_out",
            category="other",
            filename=None,
            schema=cc.outputs["vessels"].schema,
            enrichment={},
            col_specs=cc.outputs["vessels"].col_specs,
        )
        result = asyncio.get_event_loop().run_until_complete(
            compress_and_classify_async(
                "inputs/2857439.pdf",
                categories,
                specs,
                refine_headers=False,
            )
        )
        # "shipping" should match; "other" should not (no matching keywords)
        assert "vessels" in result
        assert "other_out" not in result

    def test_compress_docx_classification(self):
        """DOCX tables are classified and compressed correctly."""
        categories = {
            "shipping": ["cargo", "vessel", "port"],
            "finance": ["revenue", "profit", "margin"],
        }
        shipping_spec = OutputSpec(
            name="vessels",
            category="shipping",
            filename=None,
            schema=CanonicalSchema(columns=[]),
            enrichment={},
            col_specs=[],
        )
        finance_spec = OutputSpec(
            name="financials",
            category="finance",
            filename=None,
            schema=CanonicalSchema(columns=[]),
            enrichment={},
            col_specs=[],
        )
        specs = {"vessels": shipping_spec, "financials": finance_spec}

        result = asyncio.get_event_loop().run_until_complete(
            compress_and_classify_async(
                _synth("multi_category"),
                categories,
                specs,
                refine_headers=False,
            )
        )
        # multi_category.docx has shipping and financial tables
        # At minimum, at least one output should be populated
        assert isinstance(result, dict)
        # Result values for DOCX are list[tuple[str, dict]]
        for data in result.values():
            assert isinstance(data, list)
            assert len(data) > 0
            md, meta = data[0]
            assert isinstance(md, str)
            assert isinstance(meta, dict)


# ─── interpret_output_async ──────────────────────────────────────────────────


class TestInterpretOutputAsync:
    """Tests for interpret_output_async()."""

    def test_schema_deep_copy(self):
        """Caller's schema aliases must NOT be mutated after year template resolution."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "Region name", aliases=["Region"]),
            ColumnDef("Year", "string", "Year", aliases=["{YYYY}", "{YYYY-1}"]),
            ColumnDef("Value", "int", "Numeric value", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={},
            col_specs=[
                {"name": "Region"},
                {"name": "Year"},
                {"name": "Value"},
            ],
        )

        # Save original aliases
        original_aliases = [a for a in schema.columns[1].aliases]

        # Minimal pipe-table that the deterministic mapper can resolve
        pipe_table = (
            "| Region | 2025 | 2024 | Value |\n"
            "|--------|------|------|-------|\n"
            "| Moscow | X    | Y    | 100   |\n"
        )

        asyncio.get_event_loop().run_until_complete(
            interpret_output_async(
                pipe_table,
                spec,
                pivot_years=["2024", "2025"],
            )
        )

        # Original schema aliases must be unchanged
        assert schema.columns[1].aliases == original_aliases

    def test_year_template_resolution(self):
        """Aliases with {YYYY} templates are resolved before interpretation."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "Region name", aliases=["Region"]),
            ColumnDef("Metric", "string", "Metric type", aliases=["MOA Target {YYYY}"]),
            ColumnDef("Value", "int", "Numeric value", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={},
            col_specs=[
                {"name": "Region"},
                {"name": "Metric"},
                {"name": "Value"},
            ],
        )

        pipe_table = (
            "| Region | MOA Target 2025 | Value |\n"
            "|--------|-----------------|-------|\n"
            "| Moscow | 500             | 100   |\n"
        )

        df = asyncio.get_event_loop().run_until_complete(
            interpret_output_async(
                pipe_table,
                spec,
                pivot_years=["2025"],
            )
        )

        # If resolution worked, "MOA Target 2025" matches the resolved alias
        assert isinstance(df, pd.DataFrame)
        # Original aliases must NOT be mutated
        assert schema.columns[1].aliases == ["MOA Target {YYYY}"]

    def test_no_pivot_years_strips_templates(self):
        """When pivot_years is empty, template aliases are stripped."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "Region name", aliases=["Region"]),
            ColumnDef("Value", "int", "Numeric value", aliases=["Value", "{YYYY}"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={},
            col_specs=[{"name": "Region"}, {"name": "Value"}],
        )

        pipe_table = (
            "| Region | Value |\n"
            "|--------|-------|\n"
            "| Moscow | 100   |\n"
        )

        df = asyncio.get_event_loop().run_until_complete(
            interpret_output_async(
                pipe_table,
                spec,
                pivot_years=[],
            )
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Original aliases unchanged
        assert schema.columns[1].aliases == ["Value", "{YYYY}"]

    def test_enrichment_applied(self):
        """Enrichment columns are added to the output DataFrame."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "Region name", aliases=["Region"]),
            ColumnDef("Value", "int", "Numeric value", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={"Campaign": {"source": "constant", "value": "HARVEST"}},
            col_specs=[
                {"name": "Region"},
                {"name": "Value"},
                {"name": "Campaign"},
            ],
        )

        pipe_table = (
            "| Region | Value |\n"
            "|--------|-------|\n"
            "| Moscow | 100   |\n"
        )

        df = asyncio.get_event_loop().run_until_complete(
            interpret_output_async(pipe_table, spec)
        )

        assert "Campaign" in df.columns
        assert list(df["Campaign"]) == ["HARVEST"]

    def test_column_reorder(self):
        """Output columns are reordered to match col_specs order."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "", aliases=["Region"]),
            ColumnDef("Value", "int", "", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={"Campaign": {"source": "constant", "value": "X"}},
            col_specs=[
                {"name": "Campaign"},  # enrichment first
                {"name": "Region"},
                {"name": "Value"},
            ],
        )

        pipe_table = (
            "| Region | Value |\n"
            "|--------|-------|\n"
            "| Test   | 42    |\n"
        )

        df = asyncio.get_event_loop().run_until_complete(
            interpret_output_async(pipe_table, spec)
        )

        assert list(df.columns) == ["Campaign", "Region", "Value"]


# ─── process_document_async ──────────────────────────────────────────────────


class TestProcessDocumentAsync:
    """Tests for process_document_async() — integration, deterministic schemas."""

    def test_process_pdf_returns_document_result(self):
        """Returns DocumentResult with correct fields for a PDF document."""
        cc = load_contract("contracts/au_shipping_stem.json")
        result = asyncio.get_event_loop().run_until_complete(
            process_document_async("inputs/2857439.pdf", cc, refine_headers=False)
        )

        assert isinstance(result, DocumentResult)
        assert result.doc_path == "inputs/2857439.pdf"
        assert isinstance(result.report_date, str)
        assert isinstance(result.compressed_by_category, dict)
        assert isinstance(result.dataframes, dict)
        # Should have the "vessels" output
        assert "vessels" in result.compressed_by_category
        assert "vessels" in result.dataframes
        assert isinstance(result.dataframes["vessels"], pd.DataFrame)
        assert len(result.dataframes["vessels"]) > 0

    def test_document_result_fields_not_none(self):
        """All DocumentResult fields are populated (not None)."""
        cc = load_contract("contracts/au_shipping_stem.json")
        result = asyncio.get_event_loop().run_until_complete(
            process_document_async("inputs/2857439.pdf", cc, refine_headers=False)
        )

        assert result.doc_path is not None
        assert result.report_date is not None
        assert result.compressed_by_category is not None
        assert result.dataframes is not None

    def test_process_docx_returns_document_result(self):
        """Returns DocumentResult for a DOCX document."""
        cc = load_contract("contracts/ru_ag_ministry.json")
        docx_path = "inputs/docx/input/2025-06-24_11-58-45.Russian weekly grain EOW June 20-21 2025-1.docx"
        result = asyncio.get_event_loop().run_until_complete(
            process_document_async(docx_path, cc, refine_headers=False)
        )

        assert isinstance(result, DocumentResult)
        assert result.doc_path == docx_path
        assert isinstance(result.dataframes, dict)
        # DOCX compressed data should be list[tuple[str, dict]]
        for data in result.compressed_by_category.values():
            assert isinstance(data, list)


# ─── DocumentResult dataclass ────────────────────────────────────────────────


class TestDocumentResult:
    """Tests for the DocumentResult dataclass."""

    def test_dataclass_fields(self):
        """DocumentResult has expected fields."""
        dr = DocumentResult(
            doc_path="test.pdf",
            report_date="2025-01-01",
            compressed_by_category={"cat": "text"},
            dataframes={"cat": pd.DataFrame()},
        )
        assert dr.doc_path == "test.pdf"
        assert dr.report_date == "2025-01-01"
        assert dr.compressed_by_category == {"cat": "text"}
        assert "cat" in dr.dataframes

    def test_import_from_package(self):
        """DocumentResult is importable from the top-level package."""
        from docpact import DocumentResult as DR
        assert DR is DocumentResult


# ─── save ─────────────────────────────────────────────────────────────────────


class TestSave:
    """Tests for save() — merge and write output files."""

    @staticmethod
    def _make_result(out_name: str, df: pd.DataFrame) -> DocumentResult:
        return DocumentResult(
            doc_path="test.pdf",
            report_date="",
            compressed_by_category={},
            dataframes={out_name: df},
        )

    def test_save_default_parquet(self, tmp_path):
        """No filenames → Parquet fallback."""
        df = pd.DataFrame({"a": [1, 2]})
        results = [self._make_result("out", df)]
        merged, paths = save(results, tmp_path)
        assert paths["out"].suffix == ".parquet"
        assert paths["out"].exists()
        assert list(merged.keys()) == ["out"]

    def test_save_with_csv_filename(self, tmp_path):
        """.csv filename → CSV output."""
        df = pd.DataFrame({"a": [1, 2]})
        results = [self._make_result("out", df)]
        merged, paths = save(results, tmp_path, filenames={"out": "data.csv"})
        assert paths["out"].suffix == ".csv"
        assert paths["out"].name == "data.csv"
        assert paths["out"].exists()
        # Verify it's actually CSV (readable as text)
        content = paths["out"].read_text()
        assert "a" in content  # header
        assert "1" in content  # data

    def test_save_with_tsv_filename(self, tmp_path):
        """.tsv filename → TSV output."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        results = [self._make_result("out", df)]
        merged, paths = save(results, tmp_path, filenames={"out": "data.tsv"})
        assert paths["out"].suffix == ".tsv"
        content = paths["out"].read_text()
        assert "\t" in content  # tab-separated

    def test_save_merges_multiple_documents(self, tmp_path):
        """DataFrames from multiple documents are concatenated."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        results = [self._make_result("out", df1), self._make_result("out", df2)]
        merged, paths = save(results, tmp_path, filenames={"out": "merged.csv"})
        assert len(merged["out"]) == 2  # two source frames
        # Read back and verify row count
        df_back = pd.read_csv(paths["out"])
        assert len(df_back) == 4
        assert list(df_back["a"]) == [1, 2, 3, 4]
