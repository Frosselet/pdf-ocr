"""Tests for contract_semantics.recommend — contract recommendation engine."""

from __future__ import annotations

import json

import pytest

from contract_semantics.analyze import (
    AlignedColumn,
    ColumnProfile,
    DocumentProfile,
    MetadataCandidate,
    MultiDocumentProfile,
    TableGroup,
    TableProfile,
)
from contract_semantics.recommend import (
    _extract_keywords,
    _generate_category_name,
    _guess_provider_name,
    _to_snake_case,
    compare_contract,
    recommend_contract,
    strip_recommendations,
)


# ---------------------------------------------------------------------------
# _to_snake_case
# ---------------------------------------------------------------------------


class TestToSnakeCase:
    def test_simple(self):
        assert _to_snake_case("NAME OF SHIP") == "name_of_ship"

    def test_with_parens(self):
        assert _to_snake_case("Quantity (tonnes)") == "quantity"

    def test_with_comma_unit(self):
        assert _to_snake_case("Area harvested,Th.ha.") == "area_harvested"

    def test_with_compound_header(self):
        assert _to_snake_case("Group / Metric / 2025") == "group"

    def test_empty(self):
        assert _to_snake_case("") == "column"


# ---------------------------------------------------------------------------
# _guess_provider_name
# ---------------------------------------------------------------------------


class TestGuessProviderName:
    def test_single_doc(self):
        assert _guess_provider_name(["report_2025.pdf"]) == "report_2025"

    def test_multiple_docs_common_prefix(self):
        name = _guess_provider_name(["provider_jan.pdf", "provider_feb.pdf"])
        assert "provider" in name

    def test_empty(self):
        assert _guess_provider_name([]) == "unknown_provider"


# ---------------------------------------------------------------------------
# _generate_category_name
# ---------------------------------------------------------------------------


class TestGenerateCategoryName:
    def test_from_tokens(self):
        g = TableGroup(group_id=0, common_tokens={"shipping", "vessel", "eta"})
        name = _generate_category_name(g)
        assert name == "shipping"  # longest token

    def test_from_title(self):
        tp = TableProfile(source_document="x.pdf", table_index=0, title="Harvest Data")
        g = TableGroup(group_id=0, tables=[tp], common_tokens=set())
        name = _generate_category_name(g)
        assert "harvest" in name

    def test_fallback(self):
        g = TableGroup(group_id=3, common_tokens=set())
        name = _generate_category_name(g)
        assert name == "group_3"


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_basic_extraction(self):
        tp1 = TableProfile(
            source_document="a.pdf",
            table_index=0,
            header_tokens={"vessel", "eta", "port", "commodity"},
        )
        tp2 = TableProfile(
            source_document="b.pdf",
            table_index=0,
            header_tokens={"vessel", "eta", "port", "tonnage"},
        )
        kw = _extract_keywords([tp1, tp2], {"vessel", "eta", "port"})
        assert "vessel" in kw
        assert "port" in kw

    def test_filters_short_tokens(self):
        tp = TableProfile(
            source_document="a.pdf",
            table_index=0,
            header_tokens={"a", "bb", "shipping"},
        )
        kw = _extract_keywords([tp], {"shipping"})
        assert "a" not in kw
        assert "bb" not in kw


# ---------------------------------------------------------------------------
# recommend_contract
# ---------------------------------------------------------------------------


class TestRecommendContract:
    def _make_single_doc_profile(self) -> DocumentProfile:
        tp = TableProfile(
            source_document="test.pdf",
            table_index=0,
            title="Sales Report",
            layout="flat",
            column_profiles=[
                ColumnProfile(header_text="Region", inferred_type="string", unique_count=10, total_count=50),
                ColumnProfile(header_text="Revenue", inferred_type="float"),
                ColumnProfile(header_text="2025", inferred_type="int", year_detected=True),
            ],
            row_count=50,
            col_count=3,
            header_tokens={"region", "revenue"},
            section_labels=["North", "South"],
        )
        return DocumentProfile(
            doc_path="test.pdf",
            doc_format="pdf",
            tables=[tp],
            temporal_candidates=[
                MetadataCandidate(
                    text="As of January 15, 2025",
                    pattern_name="as_of_date",
                    captured_value="January 15, 2025",
                ),
            ],
        )

    def test_basic_structure(self):
        profile = self._make_single_doc_profile()
        draft = recommend_contract(profile, provider_name="Test Provider")

        assert draft["provider"] == "Test Provider"
        assert draft["model"] == "openai/gpt-4o"
        assert "categories" in draft
        assert "outputs" in draft
        assert "_analyzer_version" in draft

    def test_has_schema_columns(self):
        profile = self._make_single_doc_profile()
        draft = recommend_contract(profile)

        # Should have at least one output with schema columns
        for _name, output in draft["outputs"].items():
            assert "schema" in output
            assert "columns" in output["schema"]
            assert len(output["schema"]["columns"]) > 0

    def test_year_column_uses_template(self):
        profile = self._make_single_doc_profile()
        draft = recommend_contract(profile)

        # Find the year column
        found_year = False
        for _name, output in draft["outputs"].items():
            for col in output["schema"]["columns"]:
                if col.get("aliases") == ["{YYYY}"]:
                    found_year = True
                    break
        assert found_year, "Expected a column with {YYYY} template alias"

    def test_section_labels_as_dimension(self):
        profile = self._make_single_doc_profile()
        draft = recommend_contract(profile)

        # Should have a section dimension column with labels as aliases
        found_section = False
        for _name, output in draft["outputs"].items():
            for col in output["schema"]["columns"]:
                aliases = col.get("aliases", [])
                if "North" in aliases and "South" in aliases:
                    found_section = True
                    break
        assert found_section, "Expected section labels as dimension column aliases"

    def test_report_date_suggested(self):
        profile = self._make_single_doc_profile()
        draft = recommend_contract(profile)

        assert "report_date" in draft
        assert draft["report_date"]["source"] == "content"

    def test_custom_model(self):
        profile = self._make_single_doc_profile()
        draft = recommend_contract(profile, model="anthropic/claude-sonnet-4-6")
        assert draft["model"] == "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# strip_recommendations
# ---------------------------------------------------------------------------


class TestStripRecommendations:
    def test_strips_underscore_keys(self):
        contract = {
            "_analyzer_version": "0.1.0",
            "_source_documents": ["test.pdf"],
            "provider": "test",
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {
                                "name": "col",
                                "type": "string",
                                "_recommendation": "Some recommendation",
                                "_detected_type": "string",
                                "aliases": ["Col"],
                            }
                        ]
                    }
                }
            },
        }
        clean = strip_recommendations(contract)
        assert "_analyzer_version" not in clean
        assert "_source_documents" not in clean
        assert "provider" in clean
        col = clean["outputs"]["data"]["schema"]["columns"][0]
        assert "_recommendation" not in col
        assert "_detected_type" not in col
        assert col["aliases"] == ["Col"]

    def test_does_not_mutate_original(self):
        contract = {"_key": "value", "provider": "test"}
        clean = strip_recommendations(contract)
        assert "_key" in contract  # Original untouched
        assert "_key" not in clean


# ---------------------------------------------------------------------------
# compare_contract
# ---------------------------------------------------------------------------


class TestCompareContract:
    def test_uncovered_headers(self, tmp_path):
        contract = {
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {"name": "vessel", "aliases": ["Vessel", "Ship"]},
                            {"name": "port", "aliases": ["Port"]},
                        ]
                    }
                }
            }
        }
        contract_path = tmp_path / "contract.json"
        contract_path.write_text(json.dumps(contract))

        tp = TableProfile(
            source_document="test.pdf",
            table_index=0,
            column_profiles=[
                ColumnProfile(header_text="Vessel", inferred_type="string"),
                ColumnProfile(header_text="ETA", inferred_type="string"),
                ColumnProfile(header_text="Port", inferred_type="string"),
            ],
            col_count=3,
        )
        profile = DocumentProfile(doc_path="test.pdf", doc_format="pdf", tables=[tp])

        report = compare_contract(profile, contract_path)
        assert "Uncovered headers" in report
        assert "ETA" in report

    def test_all_covered(self, tmp_path):
        contract = {
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {"name": "vessel", "aliases": ["Vessel"]},
                            {"name": "port", "aliases": ["Port"]},
                        ]
                    }
                }
            }
        }
        contract_path = tmp_path / "contract.json"
        contract_path.write_text(json.dumps(contract))

        tp = TableProfile(
            source_document="test.pdf",
            table_index=0,
            column_profiles=[
                ColumnProfile(header_text="Vessel", inferred_type="string"),
                ColumnProfile(header_text="Port", inferred_type="string"),
            ],
            col_count=2,
        )
        profile = DocumentProfile(doc_path="test.pdf", doc_format="pdf", tables=[tp])

        report = compare_contract(profile, contract_path)
        assert "All document headers are covered" in report

    def test_unused_aliases(self, tmp_path):
        contract = {
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {
                                "name": "vessel",
                                "aliases": ["Vessel", "Ship Name", "BOAT"],
                            },
                        ]
                    }
                }
            }
        }
        contract_path = tmp_path / "contract.json"
        contract_path.write_text(json.dumps(contract))

        tp = TableProfile(
            source_document="test.pdf",
            table_index=0,
            column_profiles=[
                ColumnProfile(header_text="Vessel", inferred_type="string"),
            ],
            col_count=1,
        )
        profile = DocumentProfile(doc_path="test.pdf", doc_format="pdf", tables=[tp])

        report = compare_contract(profile, contract_path)
        assert "Unused aliases" in report
        assert "boat" in report.lower() or "BOAT" in report

    def test_section_labels_uncovered(self, tmp_path):
        contract = {
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {"name": "port", "aliases": ["GERALDTON"]},
                        ]
                    }
                }
            }
        }
        contract_path = tmp_path / "contract.json"
        contract_path.write_text(json.dumps(contract))

        tp = TableProfile(
            source_document="test.pdf",
            table_index=0,
            column_profiles=[],
            col_count=0,
            section_labels=["GERALDTON", "KWINANA", "ALBANY"],
        )
        profile = DocumentProfile(doc_path="test.pdf", doc_format="pdf", tables=[tp])

        report = compare_contract(profile, contract_path)
        assert "Uncovered section labels" in report
        assert "KWINANA" in report
        assert "ALBANY" in report
