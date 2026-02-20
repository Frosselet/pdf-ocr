"""Tests for pipeline integration with SemanticContext.

Pure-Python, no LLM calls.  Uses deterministic-resolvable pipe-tables and
synthetic SemanticContext objects.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from docpact.contracts import OutputSpec, SemanticColumnSpec, load_contract
from docpact.interpret import CanonicalSchema, ColumnDef
from docpact.pipeline import (
    DocumentResult,
    interpret_output_async,
    process_document_async,
)
from docpact.semantics import (
    PreFlightReport,
    SemanticContext,
    ValidationReport,
)


# ─── interpret_output_async with semantic_context ─────────────────────────────


class TestInterpretOutputAsyncSemantic:
    """Tests for interpret_output_async() with semantic_context parameter."""

    def test_semantic_aliases_merged(self):
        """Resolved aliases from SemanticContext are merged into schema."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "", aliases=["Region"]),
            ColumnDef("Value", "float", "", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={},
            col_specs=[{"name": "Region"}, {"name": "Value"}],
        )

        # Pipe table has "Oblast" as header, not "Region"
        pipe = (
            "| Oblast | Value |\n"
            "|--------|-------|\n"
            "| Moscow | 100   |\n"
        )

        # Without semantic context, "Oblast" won't match "Region"
        ctx = SemanticContext(
            resolved_aliases={"test": {"Region": ["Oblast", "District"]}},
        )

        df = asyncio.get_event_loop().run_until_complete(
            interpret_output_async(pipe, spec, semantic_context=ctx)
        )

        assert isinstance(df, pd.DataFrame)
        # With semantic context, "Oblast" should match Region via alias merge
        assert "Region" in df.columns

    def test_original_schema_not_mutated(self):
        """Caller's schema aliases must NOT be mutated by alias merge."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "", aliases=["Region"]),
            ColumnDef("Value", "float", "", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={},
            col_specs=[{"name": "Region"}, {"name": "Value"}],
        )

        original_aliases = list(schema.columns[0].aliases)

        ctx = SemanticContext(
            resolved_aliases={"test": {"Region": ["Oblast"]}},
        )
        pipe = (
            "| Region | Value |\n"
            "|--------|-------|\n"
            "| Moscow | 100   |\n"
        )

        asyncio.get_event_loop().run_until_complete(
            interpret_output_async(pipe, spec, semantic_context=ctx)
        )

        assert schema.columns[0].aliases == original_aliases

    def test_no_semantic_context_unchanged(self):
        """Without semantic_context, behavior is identical to before."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "", aliases=["Region"]),
            ColumnDef("Value", "float", "", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={},
            col_specs=[{"name": "Region"}, {"name": "Value"}],
        )

        pipe = (
            "| Region | Value |\n"
            "|--------|-------|\n"
            "| Moscow | 100   |\n"
        )

        df = asyncio.get_event_loop().run_until_complete(
            interpret_output_async(pipe, spec)
        )

        assert isinstance(df, pd.DataFrame)
        assert "Region" in df.columns
        assert "Value" in df.columns

    def test_duplicate_aliases_not_added(self):
        """Aliases already in the schema are not duplicated."""
        schema = CanonicalSchema(columns=[
            ColumnDef("Region", "string", "", aliases=["Region", "Oblast"]),
            ColumnDef("Value", "float", "", aliases=["Value"]),
        ])
        spec = OutputSpec(
            name="test",
            category="test",
            filename=None,
            schema=schema,
            enrichment={},
            col_specs=[{"name": "Region"}, {"name": "Value"}],
        )

        ctx = SemanticContext(
            resolved_aliases={"test": {"Region": ["Oblast", "region"]}},
        )
        pipe = (
            "| Region | Value |\n"
            "|--------|-------|\n"
            "| Moscow | 100   |\n"
        )

        # Should not raise — just verify no duplicates cause issues
        df = asyncio.get_event_loop().run_until_complete(
            interpret_output_async(pipe, spec, semantic_context=ctx)
        )
        assert isinstance(df, pd.DataFrame)


# ─── process_document_async with semantic_context ─────────────────────────────


class TestProcessDocumentAsyncSemantic:
    """Tests for process_document_async() with semantic context."""

    def test_preflight_reports_populated(self):
        """When semantic_context is provided, preflight_reports are populated."""
        cc = load_contract("contracts/au_shipping_stem.json")
        ctx = SemanticContext()  # Empty context, but triggers pre-flight

        result = asyncio.get_event_loop().run_until_complete(
            process_document_async(
                "inputs/2857439.pdf", cc,
                refine_headers=False,
                semantic_context=ctx,
            )
        )

        assert isinstance(result, DocumentResult)
        assert isinstance(result.preflight_reports, dict)
        # Should have a report for "vessels" output
        assert "vessels" in result.preflight_reports
        assert isinstance(result.preflight_reports["vessels"], PreFlightReport)

    def test_validation_reports_populated(self):
        """When semantic_context is provided, validation_reports are populated."""
        cc = load_contract("contracts/au_shipping_stem.json")
        ctx = SemanticContext()

        result = asyncio.get_event_loop().run_until_complete(
            process_document_async(
                "inputs/2857439.pdf", cc,
                refine_headers=False,
                semantic_context=ctx,
            )
        )

        assert isinstance(result.validation_reports, dict)
        # Should have a report for "vessels" (even if empty)
        assert "vessels" in result.validation_reports
        assert isinstance(result.validation_reports["vessels"], ValidationReport)

    def test_no_semantic_context_empty_reports(self):
        """Without semantic_context, reports are empty dicts."""
        cc = load_contract("contracts/au_shipping_stem.json")

        result = asyncio.get_event_loop().run_until_complete(
            process_document_async(
                "inputs/2857439.pdf", cc,
                refine_headers=False,
            )
        )

        assert result.preflight_reports == {}
        assert result.validation_reports == {}

    def test_backward_compatible_dataframes(self):
        """DocumentResult.dataframes still works as before with semantic context."""
        cc = load_contract("contracts/au_shipping_stem.json")
        ctx = SemanticContext()

        result = asyncio.get_event_loop().run_until_complete(
            process_document_async(
                "inputs/2857439.pdf", cc,
                refine_headers=False,
                semantic_context=ctx,
            )
        )

        assert "vessels" in result.dataframes
        assert isinstance(result.dataframes["vessels"], pd.DataFrame)
        assert len(result.dataframes["vessels"]) > 0


# ─── DocumentResult backward compatibility ────────────────────────────────────


class TestDocumentResultBackwardCompat:
    """Verify DocumentResult new fields don't break existing usage."""

    def test_default_empty_reports(self):
        """New fields default to empty dicts."""
        dr = DocumentResult(
            doc_path="test.pdf",
            report_date="",
            compressed_by_category={},
            dataframes={},
        )
        assert dr.preflight_reports == {}
        assert dr.validation_reports == {}

    def test_existing_field_access(self):
        """Accessing existing fields still works."""
        dr = DocumentResult(
            doc_path="test.pdf",
            report_date="2025-01-01",
            compressed_by_category={"cat": "text"},
            dataframes={"cat": pd.DataFrame()},
        )
        assert dr.doc_path == "test.pdf"
        assert dr.report_date == "2025-01-01"
        assert "cat" in dr.dataframes
