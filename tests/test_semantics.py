"""Tests for docpact.semantics — semantic context, pre-flight, and validation.

All pure-Python, no LLM calls or external adapters.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from docpact.contracts import OutputSpec, SemanticColumnSpec, load_contract
from docpact.interpret import CanonicalSchema, ColumnDef
from docpact.semantics import (
    PreFlightFinding,
    PreFlightReport,
    SemanticContext,
    ValidationFinding,
    ValidationReport,
    _extract_all_headers,
    _extract_headers_from_pipe_table,
    preflight_check,
    validate_output,
)


# ─── SemanticContext serialization ────────────────────────────────────────────


class TestSemanticContextSerialization:
    """Tests for SemanticContext to_dict/from_dict/to_json/from_json."""

    def test_round_trip_dict(self):
        ctx = SemanticContext(
            resolved_aliases={"harvest": {"Region": ["Moscow Oblast", "Voronezh Oblast"]}},
            valid_values={"harvest": {"Region": {"Moscow Oblast", "Voronezh Oblast"}}},
            resolved_at="2025-06-20T12:00:00Z",
            adapter_versions={"geonames": "GeoNamesAdapter"},
        )
        d = ctx.to_dict()
        restored = SemanticContext.from_dict(d)

        assert restored.resolved_aliases == ctx.resolved_aliases
        assert restored.valid_values == ctx.valid_values
        assert restored.resolved_at == ctx.resolved_at
        assert restored.adapter_versions == ctx.adapter_versions

    def test_round_trip_json(self, tmp_path):
        ctx = SemanticContext(
            resolved_aliases={"out1": {"col1": ["alias1", "alias2"]}},
            valid_values={"out1": {"col1": {"val1", "val2"}}},
            resolved_at="2025-01-01T00:00:00Z",
        )
        path = tmp_path / "ctx.json"
        ctx.to_json(path)

        # Verify JSON is valid
        with open(path) as f:
            data = json.load(f)
        assert "resolved_aliases" in data
        assert "valid_values" in data

        # Round-trip
        restored = SemanticContext.from_json(path)
        assert restored.resolved_aliases == ctx.resolved_aliases
        assert restored.valid_values == ctx.valid_values

    def test_valid_values_sets_serialized_as_sorted_lists(self):
        ctx = SemanticContext(
            valid_values={"out": {"col": {"cherry", "apple", "banana"}}},
        )
        d = ctx.to_dict()
        assert d["valid_values"]["out"]["col"] == ["apple", "banana", "cherry"]

    def test_empty_context(self):
        ctx = SemanticContext()
        d = ctx.to_dict()
        assert d["resolved_aliases"] == {}
        assert d["valid_values"] == {}
        restored = SemanticContext.from_dict(d)
        assert restored.resolved_aliases == {}
        assert restored.valid_values == {}


class TestSemanticContextHelpers:
    """Tests for aliases_for() and valid_set_for()."""

    def test_aliases_for_existing(self):
        ctx = SemanticContext(
            resolved_aliases={"harvest": {"Region": ["Moscow", "Voronezh"]}},
        )
        assert ctx.aliases_for("harvest", "Region") == ["Moscow", "Voronezh"]

    def test_aliases_for_missing_output(self):
        ctx = SemanticContext(resolved_aliases={})
        assert ctx.aliases_for("nonexistent", "Region") == []

    def test_aliases_for_missing_column(self):
        ctx = SemanticContext(
            resolved_aliases={"harvest": {"Region": ["Moscow"]}},
        )
        assert ctx.aliases_for("harvest", "Metric") == []

    def test_valid_set_for_existing(self):
        ctx = SemanticContext(
            valid_values={"harvest": {"Region": {"Moscow", "Voronezh"}}},
        )
        assert ctx.valid_set_for("harvest", "Region") == {"Moscow", "Voronezh"}

    def test_valid_set_for_missing(self):
        ctx = SemanticContext(valid_values={})
        assert ctx.valid_set_for("harvest", "Region") == set()


# ─── Header extraction ───────────────────────────────────────────────────────


class TestHeaderExtraction:
    """Tests for pipe-table header extraction helpers."""

    def test_single_table(self):
        text = (
            "| Region | Value | Year |\n"
            "|--------|-------|------|\n"
            "| Moscow | 100   | 2025 |\n"
        )
        headers = _extract_headers_from_pipe_table(text)
        assert headers == ["Region", "Value", "Year"]

    def test_with_title_before_table(self):
        text = (
            "Winter wheat harvest\n"
            "\n"
            "| Region | Area harvested | Yield |\n"
            "|--------|----------------|-------|\n"
            "| Moscow | 500            | 25    |\n"
        )
        headers = _extract_headers_from_pipe_table(text)
        assert headers == ["Region", "Area harvested", "Yield"]

    def test_empty_text(self):
        assert _extract_headers_from_pipe_table("") == []

    def test_no_pipe_table(self):
        assert _extract_headers_from_pipe_table("Just some text\nwith no tables") == []

    def test_extract_all_headers_pdf_multipage(self):
        text = (
            "| Region | Value |\n|---|---|\n| A | 1 |\n"
            "\f"
            "| Region | Yield |\n|---|---|\n| B | 2 |\n"
        )
        headers = _extract_all_headers(text)
        assert "Region" in headers
        assert "Value" in headers
        assert "Yield" in headers

    def test_extract_all_headers_docx(self):
        data = [
            ("| Col1 | Col2 |\n|---|---|\n| a | b |", {"title": "T1"}),
            ("| Col3 | Col4 |\n|---|---|\n| c | d |", {"title": "T2"}),
        ]
        headers = _extract_all_headers(data)
        assert headers == ["Col1", "Col2", "Col3", "Col4"]


# ─── Pre-flight check ────────────────────────────────────────────────────────


def _make_spec(
    name: str = "test_output",
    columns: list[ColumnDef] | None = None,
    semantic_columns: dict[str, SemanticColumnSpec] | None = None,
) -> OutputSpec:
    """Create a minimal OutputSpec for testing."""
    if columns is None:
        columns = [
            ColumnDef("Region", "string", "", aliases=["Region"]),
            ColumnDef("Value", "float", "", aliases=["Value", "Amount"]),
        ]
    return OutputSpec(
        name=name,
        category="test",
        filename=None,
        schema=CanonicalSchema(columns=columns),
        enrichment={},
        col_specs=[{"name": c.name} for c in columns],
        semantic_columns=semantic_columns or {},
    )


class TestPreflightCheck:
    """Tests for preflight_check()."""

    def test_full_match(self):
        pipe = "| Region | Value |\n|---|---|\n| Moscow | 100 |\n"
        report = preflight_check(pipe, _make_spec())
        assert report.header_coverage == 1.0
        assert report.unmatched_headers == []
        assert report.missing_aliases == []

    def test_partial_match(self):
        pipe = "| Region | Unknown_col |\n|---|---|\n| Moscow | 100 |\n"
        report = preflight_check(pipe, _make_spec())
        assert report.header_coverage == 0.5
        assert "Unknown_col" in report.unmatched_headers
        # Value column has no matching header
        assert len(report.missing_aliases) > 0

    def test_alias_match(self):
        """Headers matching aliases (not just column names) are recognized."""
        pipe = "| Region | Amount |\n|---|---|\n| Moscow | 100 |\n"
        report = preflight_check(pipe, _make_spec())
        assert report.header_coverage == 1.0

    def test_case_insensitive_match(self):
        pipe = "| REGION | value |\n|---|---|\n| Moscow | 100 |\n"
        report = preflight_check(pipe, _make_spec())
        assert report.header_coverage == 1.0

    def test_no_headers_warning(self):
        report = preflight_check("No tables here", _make_spec())
        assert any(f.severity == "warning" for f in report.findings)

    def test_with_semantic_context_extra_aliases(self):
        """SemanticContext adds extra aliases for matching."""
        ctx = SemanticContext(
            resolved_aliases={"test_output": {"Region": ["Oblast", "District"]}},
        )
        pipe = "| Oblast | Value |\n|---|---|\n| Moscow | 100 |\n"
        report = preflight_check(pipe, _make_spec(), ctx)
        assert report.header_coverage == 1.0

    def test_docx_format(self):
        """DOCX data format (list of tuples) is handled."""
        data = [
            ("| Region | Value |\n|---|---|\n| Moscow | 100 |", {"title": "Test"}),
        ]
        report = preflight_check(data, _make_spec())
        assert report.header_coverage == 1.0


# ─── Post-extraction validation ──────────────────────────────────────────────


class TestValidateOutput:
    """Tests for validate_output()."""

    def test_no_semantic_columns(self):
        """No semantic columns → empty report."""
        df = pd.DataFrame({"Region": ["Moscow"], "Value": [100]})
        spec = _make_spec()
        report = validate_output(df, spec)
        assert report.total_rows == 1
        assert report.invalid_count == 0
        assert report.findings == []

    def test_no_validate_flag(self):
        """Semantic column without validate=True → no validation."""
        spec = _make_spec(
            semantic_columns={
                "Region": SemanticColumnSpec(
                    column_name="Region",
                    concept_uris=[{"uri": "geo:1", "label": "Moscow"}],
                    validate=False,
                ),
            },
        )
        df = pd.DataFrame({"Region": ["InvalidRegion"], "Value": [100]})
        report = validate_output(df, spec)
        assert report.invalid_count == 0

    def test_valid_values(self):
        """All values in valid set → no findings."""
        spec = _make_spec(
            semantic_columns={
                "Region": SemanticColumnSpec(
                    column_name="Region",
                    concept_uris=[
                        {"uri": "geo:1", "label": "Moscow Oblast"},
                        {"uri": "geo:2", "label": "Voronezh Oblast"},
                    ],
                    validate=True,
                ),
            },
        )
        df = pd.DataFrame({"Region": ["Moscow Oblast", "Voronezh Oblast"], "Value": [100, 200]})
        report = validate_output(df, spec)
        assert report.invalid_count == 0
        assert report.valid_count == 2
        assert report.column_summaries["Region"]["valid"] == 2
        assert report.column_summaries["Region"]["invalid"] == 0

    def test_invalid_values(self):
        """Values not in valid set are flagged."""
        spec = _make_spec(
            semantic_columns={
                "Region": SemanticColumnSpec(
                    column_name="Region",
                    concept_uris=[
                        {"uri": "geo:1", "label": "Moscow Oblast"},
                    ],
                    validate=True,
                ),
            },
        )
        df = pd.DataFrame({"Region": ["Moscow Oblast", "Unknown Region"], "Value": [100, 200]})
        report = validate_output(df, spec)
        assert report.invalid_count == 1
        assert report.valid_count == 1
        assert len(report.findings) == 1
        assert report.findings[0].value == "Unknown Region"
        assert report.column_summaries["Region"]["invalid"] == 1
        assert "Unknown Region" in report.column_summaries["Region"]["unknown_values"]

    def test_case_insensitive_validation(self):
        """Validation is case-insensitive."""
        spec = _make_spec(
            semantic_columns={
                "Region": SemanticColumnSpec(
                    column_name="Region",
                    concept_uris=[{"uri": "geo:1", "label": "Moscow Oblast"}],
                    validate=True,
                ),
            },
        )
        df = pd.DataFrame({"Region": ["moscow oblast"], "Value": [100]})
        report = validate_output(df, spec)
        assert report.invalid_count == 0

    def test_empty_values_skipped(self):
        """Empty/NaN values are not flagged."""
        spec = _make_spec(
            semantic_columns={
                "Region": SemanticColumnSpec(
                    column_name="Region",
                    concept_uris=[{"uri": "geo:1", "label": "Moscow"}],
                    validate=True,
                ),
            },
        )
        df = pd.DataFrame({"Region": ["Moscow", "", None], "Value": [100, 200, 300]})
        report = validate_output(df, spec)
        assert report.invalid_count == 0

    def test_semantic_context_enriches_valid_set(self):
        """SemanticContext adds extra valid values beyond concept_uri labels."""
        spec = _make_spec(
            semantic_columns={
                "Region": SemanticColumnSpec(
                    column_name="Region",
                    concept_uris=[{"uri": "geo:1", "label": "Moscow Oblast"}],
                    validate=True,
                ),
            },
        )
        ctx = SemanticContext(
            valid_values={"test_output": {"Region": {"Moscow Oblast", "Moskovskaya oblast'"}}},
        )
        df = pd.DataFrame({"Region": ["Moskovskaya oblast'"], "Value": [100]})
        report = validate_output(df, spec, ctx)
        assert report.invalid_count == 0


# ─── Contract loading with semantic annotations ──────────────────────────────


class TestContractSemanticParsing:
    """Tests for load_contract() semantic annotation parsing."""

    def test_ru_contract_has_semantic_annotations(self):
        ctx = load_contract("contracts/ru_ag_ministry.json")
        assert ctx.has_semantic_annotations is True

    def test_au_contract_no_semantic_annotations(self):
        ctx = load_contract("contracts/au_shipping_stem.json")
        assert ctx.has_semantic_annotations is False

    def test_semantic_columns_populated(self):
        ctx = load_contract("contracts/ru_ag_ministry.json")
        harvest = ctx.outputs["harvest"]
        # Region and Metric have concept_uris in the contract
        assert "Region" in harvest.semantic_columns
        assert "Metric" in harvest.semantic_columns
        # Value does not
        assert "Value" not in harvest.semantic_columns

    def test_semantic_column_spec_fields(self):
        ctx = load_contract("contracts/ru_ag_ministry.json")
        region = ctx.outputs["harvest"].semantic_columns["Region"]
        assert region.column_name == "Region"
        assert len(region.concept_uris) > 0
        assert "uri" in region.concept_uris[0]
        assert "label" in region.concept_uris[0]
        assert isinstance(region.resolve_config, dict)
        assert region.resolve_config.get("source") == "geonames"

    def test_validate_flag_default_false(self):
        ctx = load_contract("contracts/ru_ag_ministry.json")
        region = ctx.outputs["harvest"].semantic_columns["Region"]
        assert region.validate is False

    def test_validate_flag_parsed(self, tmp_path):
        """semantic.validate: true is correctly parsed."""
        contract = {
            "provider": "test",
            "categories": {"cat": {"keywords": ["test"]}},
            "outputs": {
                "out": {
                    "category": "cat",
                    "schema": {
                        "columns": [
                            {
                                "name": "Region",
                                "type": "string",
                                "aliases": ["Region"],
                                "concept_uris": [{"uri": "geo:1", "label": "X"}],
                                "semantic": {"validate": True},
                            },
                        ],
                    },
                },
            },
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(contract))
        ctx = load_contract(path)
        assert ctx.outputs["out"].semantic_columns["Region"].validate is True

    def test_existing_tests_unaffected(self):
        """Verify OutputSpec still has all original fields working."""
        ctx = load_contract("contracts/ru_ag_ministry.json")
        harvest = ctx.outputs["harvest"]
        assert harvest.name == "harvest"
        assert harvest.category == "harvest"
        assert isinstance(harvest.schema, CanonicalSchema)
        assert len(harvest.schema.columns) > 0
        assert isinstance(harvest.enrichment, dict)
        assert isinstance(harvest.col_specs, list)

    def test_planting_semantic_columns(self):
        """Planting output also has semantic columns parsed."""
        ctx = load_contract("contracts/ru_ag_ministry.json")
        planting = ctx.outputs["planting"]
        assert "Region" in planting.semantic_columns
        assert "Crop" in planting.semantic_columns
        # Crop has concept_uris for crops
        crop = planting.semantic_columns["Crop"]
        assert len(crop.concept_uris) > 0
