"""Tests for pdf_ocr.contracts — contract loading and DataFrame helpers.

All pure-Python, no LLM calls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pdf_ocr.contracts import (
    ContractContext,
    OutputSpec,
    enrich_dataframe,
    format_dataframe,
    load_contract,
    resolve_year_templates,
)
from pdf_ocr.interpret import CanonicalSchema, ColumnDef, UnpivotStrategy
from pdf_ocr.report_date import ReportDateConfig


# ─── load_contract ───────────────────────────────────────────────────────────


class TestLoadContract:
    """Tests for load_contract()."""

    def test_load_contract_ru(self):
        ctx = load_contract("contracts/ru_ag_ministry.json")
        assert isinstance(ctx, ContractContext)
        assert ctx.provider == "ru_ag_ministry"
        assert ctx.model == "openai/gpt-4o"
        assert "harvest" in ctx.categories
        assert "planting" in ctx.categories
        assert "harvest" in ctx.outputs
        assert "planting" in ctx.outputs
        assert isinstance(ctx.raw, dict)

    def test_load_contract_unpivot_default(self):
        """Contract without 'unpivot' key defaults to True."""
        ctx = load_contract("contracts/ru_ag_ministry.json")
        assert ctx.unpivot is True

    def test_load_contract_unpivot_string(self, tmp_path):
        """String unpivot value is parsed to UnpivotStrategy enum."""
        contract = {
            "provider": "test",
            "unpivot": "deterministic",
            "categories": {},
            "outputs": {},
        }
        path = tmp_path / "test.json"
        path.write_text(__import__("json").dumps(contract))
        ctx = load_contract(path)
        assert ctx.unpivot == UnpivotStrategy.DETERMINISTIC

    def test_load_contract_unpivot_none(self, tmp_path):
        """String 'none' maps to UnpivotStrategy.NONE."""
        contract = {
            "provider": "test",
            "unpivot": "none",
            "categories": {},
            "outputs": {},
        }
        path = tmp_path / "test.json"
        path.write_text(__import__("json").dumps(contract))
        ctx = load_contract(path)
        assert ctx.unpivot == UnpivotStrategy.NONE

    def test_load_contract_unpivot_bool_false(self, tmp_path):
        """Boolean false is passed through as-is."""
        contract = {
            "provider": "test",
            "unpivot": False,
            "categories": {},
            "outputs": {},
        }
        path = tmp_path / "test.json"
        path.write_text(__import__("json").dumps(contract))
        ctx = load_contract(path)
        assert ctx.unpivot is False

    def test_load_contract_schemas(self):
        """Verify schemas have correct ColumnDef objects with aliases."""
        ctx = load_contract("contracts/ru_ag_ministry.json")
        harvest = ctx.outputs["harvest"]
        assert isinstance(harvest.schema, CanonicalSchema)
        # LLM columns should not include enrichment columns (source: title/report_date/constant)
        col_names = [c.name for c in harvest.schema.columns]
        assert "Region" in col_names
        assert "Metric" in col_names
        assert "Value" in col_names
        # Enrichment columns should NOT be in the schema
        assert "Report_date" not in col_names
        assert "Crop" not in col_names
        assert "Campaign" not in col_names
        assert "Year" not in col_names  # Year is now enrichment (source: constant)
        # Check aliases
        region_col = next(c for c in harvest.schema.columns if c.name == "Region")
        assert isinstance(region_col, ColumnDef)
        assert "Region" in region_col.aliases

    def test_load_contract_enrichment(self):
        """Verify enrichment rules parsed for source: title/report_date/constant."""
        ctx = load_contract("contracts/ru_ag_ministry.json")
        harvest_enrich = ctx.outputs["harvest"].enrichment
        # title source
        assert "Crop" in harvest_enrich
        assert harvest_enrich["Crop"]["source"] == "title"
        # report_date source
        assert "Report_date" in harvest_enrich
        assert harvest_enrich["Report_date"]["source"] == "report_date"
        # constant source
        assert "Campaign" in harvest_enrich
        assert harvest_enrich["Campaign"]["source"] == "constant"
        assert harvest_enrich["Campaign"]["value"] == "HARVESTING"

    def test_load_contract_col_specs(self):
        """col_specs contains ALL columns (LLM + enrichment) for formatting."""
        ctx = load_contract("contracts/ru_ag_ministry.json")
        planting = ctx.outputs["planting"]
        spec_names = [c["name"] for c in planting.col_specs]
        # Should have both LLM and enrichment columns
        assert "Region" in spec_names
        assert "Crop" in spec_names  # format: lowercase
        assert "Report_date" in spec_names  # source: report_date with suffix

    def test_load_contract_output_spec_fields(self):
        """Verify OutputSpec has all expected fields."""
        ctx = load_contract("contracts/au_shipping_stem.json")
        vessels = ctx.outputs["vessels"]
        assert isinstance(vessels, OutputSpec)
        assert vessels.name == "vessels"
        assert vessels.category == "shipping"
        assert vessels.filename == "shipping_stem.csv"
        assert isinstance(vessels.schema, CanonicalSchema)

    def test_load_contract_report_date_config(self):
        """Contract with report_date config gets parsed."""
        ctx = load_contract("contracts/ru_ag_ministry.json")
        assert isinstance(ctx.report_date_config, ReportDateConfig)
        assert ctx.report_date_config.source == "filename"

    def test_load_contract_no_report_date(self):
        """Contract without report_date config returns None."""
        ctx = load_contract("contracts/au_shipping_stem.json")
        assert ctx.report_date_config is None


# ─── resolve_year_templates ──────────────────────────────────────────────────


class TestResolveYearTemplates:
    """Tests for resolve_year_templates()."""

    def test_basic(self):
        assert resolve_year_templates(["{YYYY}"], ["2025"]) == ["2025"]

    def test_offset_minus(self):
        assert resolve_year_templates(["{YYYY-1}"], ["2025"]) == ["2024"]

    def test_offset_plus(self):
        assert resolve_year_templates(["{YYYY+1}"], ["2025"]) == ["2026"]

    def test_inline(self):
        result = resolve_year_templates(["MOA Target {YYYY}"], ["2025"])
        assert result == ["MOA Target 2025"]

    def test_no_years_strips_templates(self):
        result = resolve_year_templates(["{YYYY}", "Region", "{YYYY-1}"], [])
        assert result == ["Region"]

    def test_mixed(self):
        aliases = ["Region", "{YYYY}", "{YYYY-1}"]
        result = resolve_year_templates(aliases, ["2024", "2025"])
        assert result == ["Region", "2025", "2024"]

    def test_multiple_years_uses_latest(self):
        """Uses the last (latest) year in the list."""
        result = resolve_year_templates(["{YYYY}"], ["2023", "2024", "2025"])
        assert result == ["2025"]

    def test_no_templates_passthrough(self):
        aliases = ["Port", "Vessel Name"]
        assert resolve_year_templates(aliases, ["2025"]) == ["Port", "Vessel Name"]

    def test_empty_aliases(self):
        assert resolve_year_templates([], ["2025"]) == []


# ─── enrich_dataframe ────────────────────────────────────────────────────────


class TestEnrichDataframe:
    """Tests for enrich_dataframe()."""

    def test_constant(self):
        df = pd.DataFrame({"val": [1, 2]})
        enrichment = {"Campaign": {"source": "constant", "value": "HARVESTING"}}
        result = enrich_dataframe(df, enrichment)
        assert list(result["Campaign"]) == ["HARVESTING", "HARVESTING"]

    def test_title(self):
        df = pd.DataFrame({"val": [10]})
        enrichment = {"Crop": {"source": "title"}}
        result = enrich_dataframe(df, enrichment, title="Wheat")
        assert list(result["Crop"]) == ["Wheat"]

    def test_title_default(self):
        df = pd.DataFrame({"val": [10]})
        enrichment = {"Crop": {"source": "title"}}
        result = enrich_dataframe(df, enrichment)
        assert list(result["Crop"]) == ["Unknown"]

    def test_report_date(self):
        df = pd.DataFrame({"val": [1]})
        enrichment = {"Report_date": {"source": "report_date"}}
        result = enrich_dataframe(df, enrichment, report_date="June 20-21")
        assert list(result["Report_date"]) == ["June 20-21"]

    def test_report_date_with_suffix(self):
        df = pd.DataFrame({"val": [1]})
        enrichment = {"Report_date": {"source": "report_date", "suffix": " interim report"}}
        result = enrich_dataframe(df, enrichment, report_date="June 20-21")
        assert list(result["Report_date"]) == ["June 20-21 interim report"]

    def test_report_date_no_value(self):
        df = pd.DataFrame({"val": [1]})
        enrichment = {"Report_date": {"source": "report_date"}}
        result = enrich_dataframe(df, enrichment)
        assert list(result["Report_date"]) == [""]

    def test_multiple_enrichments(self):
        df = pd.DataFrame({"val": [1]})
        enrichment = {
            "Crop": {"source": "title"},
            "Campaign": {"source": "constant", "value": "HARVEST"},
        }
        result = enrich_dataframe(df, enrichment, title="Barley")
        assert list(result["Crop"]) == ["Barley"]
        assert list(result["Campaign"]) == ["HARVEST"]

    def test_empty_enrichment(self):
        df = pd.DataFrame({"val": [1, 2]})
        result = enrich_dataframe(df, {})
        assert list(result.columns) == ["val"]


# ─── format_dataframe ────────────────────────────────────────────────────────


class TestFormatDataframe:
    """Tests for format_dataframe()."""

    def test_lowercase(self):
        df = pd.DataFrame({"Crop": ["WHEAT", "Barley"]})
        result = format_dataframe(df, [{"name": "Crop", "format": "lowercase"}])
        assert list(result["Crop"]) == ["wheat", "barley"]

    def test_uppercase(self):
        df = pd.DataFrame({"Port": ["melbourne", "Adelaide"]})
        result = format_dataframe(df, [{"name": "Port", "format": "uppercase"}])
        assert list(result["Port"]) == ["MELBOURNE", "ADELAIDE"]

    def test_titlecase(self):
        df = pd.DataFrame({"Country": ["UNITED KINGDOM", "germany"]})
        result = format_dataframe(df, [{"name": "Country", "format": "titlecase"}])
        assert list(result["Country"]) == ["United Kingdom", "Germany"]

    def test_filter_latest(self):
        df = pd.DataFrame({"Year": [2023, 2024, 2025, 2025], "val": [1, 2, 3, 4]})
        result = format_dataframe(df, [{"name": "Year", "filter": "latest"}])
        assert list(result["Year"]) == [2025, 2025]
        assert list(result["val"]) == [3, 4]

    def test_filter_earliest(self):
        df = pd.DataFrame({"Year": [2023, 2023, 2024], "val": [1, 2, 3]})
        result = format_dataframe(df, [{"name": "Year", "filter": "earliest"}])
        assert list(result["Year"]) == [2023, 2023]
        assert list(result["val"]) == [1, 2]

    def test_filter_all_noop(self):
        df = pd.DataFrame({"Year": [2023, 2024], "val": [1, 2]})
        result = format_dataframe(df, [{"name": "Year", "filter": "all"}])
        assert len(result) == 2

    def test_unknown_format_ignored(self):
        """Non-case format values (like date patterns) are ignored."""
        df = pd.DataFrame({"eta": ["2025-01-01"]})
        result = format_dataframe(df, [{"name": "eta", "format": "YYYY-MM-DD"}])
        assert list(result["eta"]) == ["2025-01-01"]

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"val": [1]})
        result = format_dataframe(df, [{"name": "nonexistent", "format": "lowercase"}])
        assert list(result["val"]) == [1]

    def test_no_specs(self):
        df = pd.DataFrame({"val": [1, 2]})
        result = format_dataframe(df, [])
        assert len(result) == 2

    def test_combined_format_and_filter(self):
        df = pd.DataFrame({
            "Crop": ["WHEAT", "BARLEY", "WHEAT"],
            "Year": [2024, 2025, 2025],
        })
        specs = [
            {"name": "Crop", "format": "lowercase"},
            {"name": "Year", "filter": "latest"},
        ]
        result = format_dataframe(df, specs)
        assert list(result["Crop"]) == ["barley", "wheat"]
        assert list(result["Year"]) == [2025, 2025]
