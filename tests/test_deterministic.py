"""Tests for deterministic pipe-table interpretation.

Tests the two core functions:
- ``_parse_pipe_table_deterministic()`` — string-based pipe-table parser
- ``_map_to_schema_deterministic()`` — alias-based schema mapper
"""

from __future__ import annotations

import pytest
from baml_client import types as baml_types

from pdf_ocr.interpret import (
    CanonicalSchema,
    ColumnDef,
    _DeterministicParsed,
    _cell_is_numeric,
    _is_text_data_column,
    _normalize_for_alias_match,
    _parse_pipe_table_deterministic,
    _parse_transposed_pipe_sections,
    _map_to_schema_deterministic,
    _try_deterministic,
    _try_deterministic_transposed,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _schema(*cols: tuple[str, str, str, list[str]]) -> CanonicalSchema:
    """Shorthand to build a CanonicalSchema from (name, type, desc, aliases)."""
    return CanonicalSchema(columns=[
        ColumnDef(name=n, type=t, description=d, aliases=a)
        for n, t, d, a in cols
    ])


# ─── TestParsePipeTable ──────────────────────────────────────────────────────


class TestParsePipeTable:
    """Tests for _parse_pipe_table_deterministic."""

    def test_basic_flat_table(self) -> None:
        text = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |"
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert parsed.headers == ["A", "B", "C"]
        assert len(parsed.data_rows) == 2
        assert parsed.data_rows[0] == ["1", "2", "3"]
        assert parsed.data_rows[1] == ["4", "5", "6"]
        assert parsed.title is None
        assert parsed.sections == []
        assert parsed.aggregation_rows == []

    def test_title_extraction(self) -> None:
        text = "## My Table\n| X | Y |\n|---|---|\n| a | b |"
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert parsed.title == "My Table"

    def test_aggregation_rows(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n|| Total | 2 |"
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.data_rows) == 1
        assert len(parsed.aggregation_rows) == 1
        # || prefix produces an empty first cell from the leading double pipe
        assert parsed.aggregation_rows[0] == ["", "Total", "2"]

    def test_bold_section_labels(self) -> None:
        text = (
            "| Col |\n|---|\n"
            "**Section A**\n| 1 |\n| 2 |\n"
            "**Section B**\n| 3 |"
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.sections) == 2
        assert parsed.sections[0] == ("Section A", 0, 1)
        assert parsed.sections[1] == ("Section B", 2, 2)

    def test_plain_text_section_labels(self) -> None:
        text = (
            "| Col A | Col B |\n|---|---|\n"
            "Section One\n| 1 | x |\n| 2 | y |\n"
            "Section Two\n| 3 | z |"
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.sections) == 2
        assert parsed.sections[0][0] == "Section One"
        assert parsed.sections[1][0] == "Section Two"

    def test_no_pipe_table(self) -> None:
        """Non-table text returns None."""
        text = "This is just some text.\nNo tables here."
        assert _parse_pipe_table_deterministic(text) is None

    def test_repeated_header_rows_skipped(self) -> None:
        """Re-headers between sections are skipped."""
        text = (
            "| A | B |\n|---|---|\n"
            "| 1 | 2 |\n"
            "Section 2\n"
            "| A | B |\n"  # repeated header
            "|---|---|\n"  # repeated separator
            "| 3 | 4 |"
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.data_rows) == 2
        assert parsed.data_rows[1] == ["3", "4"]

    def test_compound_headers(self) -> None:
        text = "| Region | spring crops / 2025 | spring grain / 2025 |\n|---|---|---|\n| Russia | 100 | 50 |"
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert parsed.headers == ["Region", "spring crops / 2025", "spring grain / 2025"]


# ─── TestMapToSchema ─────────────────────────────────────────────────────────


class TestMapToSchema:
    """Tests for _map_to_schema_deterministic."""

    def test_flat_1to1_mapping(self) -> None:
        """Simple flat table with direct alias matches."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Port", "Vessel", "Tonnage"],
            data_rows=[["Melbourne", "Star", "5000"], ["Sydney", "Moon", "3000"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("port", "string", "Port", ["Port"]),
            ("vessel", "string", "Vessel", ["Vessel"]),
            ("tonnage", "float", "Tonnage", ["Tonnage"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2
        assert result.records[0].port == "Melbourne"
        assert result.records[0].vessel == "Star"
        assert result.records[0].tonnage == "5000"

    def test_unmatched_columns_dropped_not_blocking(self) -> None:
        """Unmatched columns are silently dropped, not triggering LLM fallback."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Port", "Unknown Column"],
            data_rows=[["A", "B"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("port", "string", "Port", ["Port"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert result.records[0].port == "A"
        # Unknown Column appears in unmapped_columns for transparency
        assert any("Unknown Column" in col for col in result.unmapped_columns)

    def test_empty_table(self) -> None:
        parsed = _DeterministicParsed(
            title=None, headers=["A"], data_rows=[], sections=[], aggregation_rows=[]
        )
        schema = _schema(("a", "string", "A", ["A"]))
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert result is None

    def test_joined_form_fallback(self) -> None:
        """Stacked header parts joined when individually unmatched."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Port", "Quantity / (tonnes)"],
            data_rows=[["Newcastle", "26914"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("port", "string", "Port", ["Port"]),
            ("tons", "int", "Tonnes", ["Quantity (tonnes)"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert result.records[0].tons == "26914"

    def test_joined_form_not_used_when_parts_match(self) -> None:
        """Joined form only tried when ALL parts are unmatched."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Region", "spring wheat / Value", "spring barley / Value"],
            data_rows=[["Russia", "100", "200"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("crop", "string", "Crop", ["spring wheat", "spring barley"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        # Parts match individually — joined form NOT used, pivot works
        crops = [r.crop for r in result.records]
        assert "spring wheat" in crops
        assert "spring barley" in crops

    def test_case_insensitive_aliases(self) -> None:
        """Alias matching is case-insensitive."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["COUNTRY", "VALUE"],
            data_rows=[["France", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("country", "string", "Country", ["country"]),
            ("value", "float", "Value", ["value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert result.records[0].country == "France"

    def test_skippable_percent_column(self) -> None:
        """% columns are skippable and don't trigger fallback."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Region", "Area", "%"],
            data_rows=[["Russia", "100", "50%"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("area", "float", "Area", ["Area"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None


class TestUnpivotGroups:
    """Tests for compound header unpivoting."""

    def test_two_crop_groups(self) -> None:
        """Russian-style table with two crop groups unpivots correctly."""
        parsed = _DeterministicParsed(
            title="PLANTING",
            headers=[
                "Region",
                "spring crops / MOA Target 2025",
                "spring crops / 2025",
                "spring grain / MOA Target 2025",
                "spring grain / 2025",
            ],
            data_rows=[["Russia", "100", "90", "50", "45"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("area", "float", "Area", ["MOA Target 2025"]),
            ("value", "float", "Value", ["2025"]),
            ("crop", "string", "Crop", ["spring crops", "spring grain"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2  # one per crop group

        # Spring crops
        sc = [r for r in result.records if r.crop == "spring crops"][0]
        assert sc.area == "100"
        assert sc.value == "90"
        assert sc.region == "Russia"

        # Spring grain — this is the critical case
        sg = [r for r in result.records if r.crop == "spring grain"][0]
        assert sg.area == "50"
        assert sg.value == "45"
        assert sg.region == "Russia"

    def test_three_motorization_groups(self) -> None:
        """ACEA-style table with motorization dimension."""
        parsed = _DeterministicParsed(
            title=None,
            headers=[
                "COUNTRY",
                "BATTERY ELECTRIC / Dec-25",
                "PLUG-IN HYBRID / Dec-25",
                "DIESEL / Dec-25",
            ],
            data_rows=[["AUSTRIA", "100", "50", "200"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("country", "string", "Country", ["COUNTRY"]),
            ("motorization", "string", "Power source", ["BATTERY ELECTRIC", "PLUG-IN HYBRID", "DIESEL"]),
            ("registrations", "int", "Count", ["Dec-25"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 3  # one per motorization

        countries = [r.country for r in result.records]
        assert all(c == "AUSTRIA" for c in countries)

        motorizations = sorted(r.motorization for r in result.records)
        assert motorizations == ["BATTERY ELECTRIC", "DIESEL", "PLUG-IN HYBRID"]

        # Check values mapped correctly
        be = [r for r in result.records if r.motorization == "BATTERY ELECTRIC"][0]
        assert be.registrations == "100"

    def test_six_groups(self) -> None:
        """Six unpivot groups (e.g., 6 motorization types)."""
        motor_types = ["BEV", "PHEV", "HEV", "DIESEL", "PETROL", "OTHER"]
        headers = ["COUNTRY"] + [f"{m} / Units" for m in motor_types]
        data = [["GERMANY"] + [str(i * 100) for i in range(1, 7)]]
        parsed = _DeterministicParsed(
            title=None, headers=headers, data_rows=data,
            sections=[], aggregation_rows=[],
        )
        schema = _schema(
            ("country", "string", "Country", ["COUNTRY"]),
            ("motor", "string", "Motor", motor_types),
            ("units", "int", "Units", ["Units"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 6

    def test_no_compound_headers(self) -> None:
        """Non-compound headers → single group, no unpivoting."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Name", "Score"],
            data_rows=[["Alice", "95"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("name", "string", "Name", ["Name"]),
            ("score", "float", "Score", ["Score"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 1


class TestSectionMapping:
    """Tests for section label → schema column mapping."""

    def test_sections_mapped_to_schema(self) -> None:
        parsed = _DeterministicParsed(
            title=None,
            headers=["Item", "Value"],
            data_rows=[["A", "1"], ["B", "2"], ["C", "3"]],
            sections=[("Fruits", 0, 1), ("Vegetables", 2, 2)],
            aggregation_rows=[],
        )
        schema = _schema(
            ("item", "string", "Item", ["Item"]),
            ("value", "float", "Value", ["Value"]),
            ("category", "string", "Category", ["Fruits", "Vegetables"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 3
        assert result.records[0].category == "Fruits"
        assert result.records[1].category == "Fruits"
        assert result.records[2].category == "Vegetables"


class TestMetadata:
    """Tests for MappedTable metadata correctness."""

    def test_deterministic_model_tag(self) -> None:
        parsed = _DeterministicParsed(
            title=None, headers=["X"], data_rows=[["1"]],
            sections=[], aggregation_rows=[],
        )
        schema = _schema(("x", "float", "X", ["X"]))
        result, _ = _map_to_schema_deterministic(parsed, schema)
        assert result is not None
        assert result.metadata.model == "deterministic"

    def test_unpivot_table_type(self) -> None:
        parsed = _DeterministicParsed(
            title=None,
            headers=["Region", "A / Val", "B / Val"],
            data_rows=[["R1", "1", "2"]],
            sections=[], aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("cat", "string", "Cat", ["A", "B"]),
            ("val", "float", "Value", ["Val"]),
        )
        result, _ = _map_to_schema_deterministic(parsed, schema)
        assert result is not None
        assert result.metadata.table_type_inference.table_type == baml_types.TableType.PivotedTable
        assert result.metadata.table_type_inference.mapping_strategy_used == "unpivot"

    def test_flat_table_type(self) -> None:
        parsed = _DeterministicParsed(
            title=None, headers=["X"], data_rows=[["1"]],
            sections=[], aggregation_rows=[],
        )
        schema = _schema(("x", "float", "X", ["X"]))
        result, _ = _map_to_schema_deterministic(parsed, schema)
        assert result is not None
        assert result.metadata.table_type_inference.table_type == baml_types.TableType.FlatHeader

    def test_field_mappings_all_high_confidence(self) -> None:
        parsed = _DeterministicParsed(
            title=None, headers=["X", "Y"],
            data_rows=[["1", "2"]],
            sections=[], aggregation_rows=[],
        )
        schema = _schema(("x", "float", "X", ["X"]), ("y", "float", "Y", ["Y"]))
        result, _ = _map_to_schema_deterministic(parsed, schema)
        assert result is not None
        for fm in result.metadata.field_mappings:
            assert fm.confidence == baml_types.Confidence.High


class TestTryDeterministic:
    """Tests for the _try_deterministic convenience wrapper."""

    def test_returns_mapped_table(self) -> None:
        text = "| Port | Tonnage |\n|---|---|\n| Melbourne | 5000 |"
        schema = _schema(
            ("port", "string", "Port", ["Port"]),
            ("tonnage", "float", "Tonnage", ["Tonnage"]),
        )
        result = _try_deterministic(text, schema)
        assert result is not None
        assert len(result.records) == 1
        assert result.records[0].port == "Melbourne"

    def test_drops_unmatched_columns(self) -> None:
        """Unmatched columns are dropped; mapping still succeeds."""
        text = "| Port | Mystery |\n|---|---|\n| A | B |"
        schema = _schema(("port", "string", "Port", ["Port"]))
        result = _try_deterministic(text, schema)
        assert result is not None
        assert result.records[0].port == "A"

    def test_returns_none_when_all_unmatched(self) -> None:
        """All columns unmatched → no measures/groups → falls back to LLM."""
        text = "| Foo | Bar |\n|---|---|\n| A | B |"
        schema = _schema(("port", "string", "Port", ["Port"]))
        result = _try_deterministic(text, schema)
        assert result is None

    def test_returns_none_on_non_table(self) -> None:
        text = "Just some text, no table here."
        schema = _schema(("x", "string", "X", ["X"]))
        result = _try_deterministic(text, schema)
        assert result is None


class TestRussianPlantingIntegration:
    """Full integration test for Russian planting table pattern."""

    def test_full_table_with_sections(self) -> None:
        text = (
            "## PLANTING\n"
            "| Th.ha. / Region | spring crops / MOA Target 2025 | spring crops / 2025 | "
            "spring grain / MOA Target 2025 | spring grain / 2025 |\n"
            "|---|---|---|---|---|\n"
            "**CENTRAL**\n"
            "| Belgorod | 100 | 90 | 50 | 45 |\n"
            "| Bryansk | 80 | 70 | 40 | 35 |\n"
            "**SOUTH**\n"
            "| Krasnodar | 200 | 180 | 100 | 95 |"
        )
        schema = CanonicalSchema(columns=[
            ColumnDef("region", "string", "Region", aliases=["Region"]),
            ColumnDef("area", "float", "Area", aliases=["MOA Target 2025"]),
            ColumnDef("value", "float", "Value", aliases=["2025"]),
            ColumnDef("crop", "string", "Crop", aliases=["spring crops", "spring grain"]),
            ColumnDef("unit", "string", "Unit", aliases=["Th.ha."]),
            ColumnDef("district", "string", "District", aliases=["CENTRAL", "SOUTH"]),
        ])
        result = _try_deterministic(text, schema)
        assert result is not None
        assert len(result.records) == 6  # 3 rows × 2 crop groups

        # Check section mapping
        central = [r for r in result.records if r.district == "CENTRAL"]
        south = [r for r in result.records if r.district == "SOUTH"]
        assert len(central) == 4  # 2 rows × 2 groups
        assert len(south) == 2  # 1 row × 2 groups

        # Verify spring grain values are not shifted
        sg_belgorod = [
            r for r in result.records
            if r.crop == "spring grain" and r.region == "Belgorod"
        ]
        assert len(sg_belgorod) == 1
        assert sg_belgorod[0].area == "50"
        assert sg_belgorod[0].value == "45"


class TestUnmatchedColumnDrop:
    """Tests for silently dropping unmatched columns."""

    def test_unmatched_columns_with_compound_headers(self) -> None:
        """Th.ha. / Final 2024 type columns resolve deterministically.

        These columns are partially matched (Th.ha. matches a dimension),
        so unmatched parts (MOA 2024, Final 2024) are treated as annotations.
        """
        parsed = _DeterministicParsed(
            title=None,
            headers=[
                "Th.ha. / Region",
                "Th.ha. / MOA 2024",
                "Th.ha. / Final 2024",
                "spring crops / 2025",
                "spring grain / 2025",
            ],
            data_rows=[["Russia", "ref1", "ref2", "90", "45"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("value", "float", "Value", ["2025"]),
            ("crop", "string", "Crop", ["spring crops", "spring grain"]),
            ("unit", "string", "Unit", ["Th.ha."]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2
        # Values from the correct columns (not shifted)
        sc = [r for r in result.records if r.crop == "spring crops"][0]
        assert sc.value == "90"
        sg = [r for r in result.records if r.crop == "spring grain"][0]
        assert sg.value == "45"

    def test_fully_unmatched_column_in_unmapped(self) -> None:
        """Entirely unmatched column appears in unmapped_columns."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Port", "Tonnage", "Random Notes"],
            data_rows=[["Melbourne", "5000", "some note"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("port", "string", "Port", ["Port"]),
            ("tonnage", "float", "Tonnage", ["Tonnage"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert "Random Notes" in result.unmapped_columns

    def test_all_columns_unmatched_falls_back(self) -> None:
        """Table where no column matches any alias returns None (LLM fallback)."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Foo", "Bar", "Baz"],
            data_rows=[["1", "2", "3"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("port", "string", "Port", ["Port"]),
            ("vessel", "string", "Vessel", ["Vessel"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        # No measures or groups found → returns None for LLM fallback
        assert result is None

    def test_empty_header_column_dropped(self) -> None:
        """Empty-string header column is silently dropped."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Region", "", "Area"],
            data_rows=[["Russia", "junk", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("area", "float", "Area", ["Area"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert result.records[0].region == "Russia"
        assert result.records[0].area == "100"


class TestACEAIntegration:
    """Full integration test for ACEA motorization pattern."""

    def test_acea_style_table(self) -> None:
        text = (
            "| COUNTRY | BATTERY ELECTRIC / Dec-25 | BATTERY ELECTRIC / % |"
            " PLUG-IN HYBRID / Dec-25 | PLUG-IN HYBRID / % |"
            " DIESEL / Dec-25 | DIESEL / % |\n"
            "|---|---|---|---|---|---|---|\n"
            "| AUSTRIA | 100 | 10% | 50 | 5% | 200 | 20% |"
        )
        schema = CanonicalSchema(columns=[
            ColumnDef("country", "string", "Country", aliases=["COUNTRY"]),
            ColumnDef("motorization", "string", "Power source",
                      aliases=["BATTERY ELECTRIC", "PLUG-IN HYBRID", "DIESEL"]),
            ColumnDef("registrations", "int", "Count", aliases=["Dec-25"]),
        ])
        result = _try_deterministic(text, schema)
        assert result is not None
        assert len(result.records) == 3

        be = [r for r in result.records if r.motorization == "BATTERY ELECTRIC"][0]
        assert be.registrations == "100"
        assert be.country == "AUSTRIA"


class TestOverlappingAliases:
    """Tests for schemas where multiple columns share the same alias."""

    def test_year_int_and_value_float_share_alias(self) -> None:
        """Year(int) and Value(float) both have alias '2025'.

        Year should become a constant dimension (header text '2025'),
        Value should become a measure (cell data).
        """
        parsed = _DeterministicParsed(
            title=None,
            headers=[
                "Region",
                "spring crops / MOA Target 2025",
                "spring crops / 2025",
                "spring grain / MOA Target 2025",
                "spring grain / 2025",
            ],
            data_rows=[["Russia", "100", "90", "50", "45"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("area", "float", "Area", ["MOA Target 2025"]),
            ("value", "float", "Value", ["2025"]),
            ("crop", "string", "Crop", ["spring crops", "spring grain"]),
            ("year", "int", "Year", ["2025"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2

        # Year should be constant "2025" in every record
        for r in result.records:
            assert r.year == "2025"

        # Value should be cell data, not "2025"
        sc = [r for r in result.records if r.crop == "spring crops"][0]
        assert sc.value == "90"
        assert sc.area == "100"

        sg = [r for r in result.records if r.crop == "spring grain"][0]
        assert sg.value == "45"
        assert sg.area == "50"

    def test_partially_matched_column_not_blocking(self) -> None:
        """Column 'Th.ha. / MOA 2024' has matched Th.ha. but unmatched MOA 2024.

        Since 'Th.ha.' matches, the column is partially matched and unmatched
        parts are treated as annotations — not blocking deterministic mapping.
        """
        parsed = _DeterministicParsed(
            title=None,
            headers=[
                "Th.ha. / Region",
                "Th.ha. / MOA 2024",
                "spring crops / 2025",
                "spring grain / 2025",
            ],
            data_rows=[["Russia", "ref_data", "90", "45"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("value", "float", "Value", ["2025"]),
            ("crop", "string", "Crop", ["spring crops", "spring grain"]),
            ("unit", "string", "Unit", ["Th.ha."]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2

        # Unit should be constant "Th.ha."
        for r in result.records:
            assert r.unit == "Th.ha."
            assert r.region == "Russia"

        # Values from correct columns (not shifted by MOA 2024 col)
        sc = [r for r in result.records if r.crop == "spring crops"][0]
        assert sc.value == "90"
        sg = [r for r in result.records if r.crop == "spring grain"][0]
        assert sg.value == "45"

    def test_full_realistic_russian_with_overlaps(self) -> None:
        """Full realistic Russian table with overlapping Year/Value aliases
        AND extra MOA 2024 reference column."""
        text = (
            "## PLANTING\n"
            "| Th.ha. / Region | Th.ha. / MOA 2024 | "
            "spring crops / MOA Target 2025 | spring crops / 2025 | "
            "spring grain / MOA Target 2025 | spring grain / 2025 |\n"
            "|---|---|---|---|---|---|\n"
            "**CENTRAL**\n"
            "| Belgorod | 999 | 100 | 90 | 50 | 45 |\n"
            "**SOUTH**\n"
            "| Krasnodar | 888 | 200 | 180 | 100 | 95 |"
        )
        schema = CanonicalSchema(columns=[
            ColumnDef("region", "string", "Region", aliases=["Region"]),
            ColumnDef("area", "float", "Area", aliases=["MOA Target 2025"]),
            ColumnDef("value", "float", "Value", aliases=["2025"]),
            ColumnDef("crop", "string", "Crop", aliases=["spring crops", "spring grain"]),
            ColumnDef("unit", "string", "Unit", aliases=["Th.ha."]),
            ColumnDef("year", "int", "Year", aliases=["2025"]),
            ColumnDef("district", "string", "District", aliases=["CENTRAL", "SOUTH"]),
        ])
        result = _try_deterministic(text, schema)
        assert result is not None
        assert len(result.records) == 4  # 2 rows × 2 crop groups

        # Check districts
        central = [r for r in result.records if r.district == "CENTRAL"]
        south = [r for r in result.records if r.district == "SOUTH"]
        assert len(central) == 2
        assert len(south) == 2

        # Check constants
        for r in result.records:
            assert r.unit == "Th.ha."
            assert r.year == "2025"

        # Spring grain Belgorod — critical test (columns not shifted)
        sg_bel = [r for r in central if r.crop == "spring grain"][0]
        assert sg_bel.region == "Belgorod"
        assert sg_bel.area == "50"
        assert sg_bel.value == "45"

        # Spring crops Krasnodar
        sc_kras = [r for r in south if r.crop == "spring crops"][0]
        assert sc_kras.region == "Krasnodar"
        assert sc_kras.area == "200"
        assert sc_kras.value == "180"


# ─── TestNormalizeForAliasMatch ──────────────────────────────────────────────


class TestNormalizeForAliasMatch:
    """Tests for _normalize_for_alias_match()."""

    def test_case_insensitive(self) -> None:
        assert _normalize_for_alias_match("ETA") == "eta"
        assert _normalize_for_alias_match("Vessel Name") == "vessel name"

    def test_paren_spacing(self) -> None:
        assert _normalize_for_alias_match("Quantity(tonnes)") == "quantity (tonnes)"
        assert _normalize_for_alias_match("Quantity (tonnes)") == "quantity (tonnes)"
        assert _normalize_for_alias_match("Quantity  (  tonnes  )") == "quantity (tonnes)"

    def test_whitespace_collapse(self) -> None:
        assert _normalize_for_alias_match("  Name   of   Ship  ") == "name of ship"

    def test_mixed(self) -> None:
        assert _normalize_for_alias_match("QUANTITY(TONNES)") == "quantity (tonnes)"

    def test_double_quote_stripping(self) -> None:
        assert _normalize_for_alias_match('"Loading ""commenced"" or ""completed"""') == "loading commenced or completed"

    def test_quotes_in_alias(self) -> None:
        assert _normalize_for_alias_match('Loading "commenced" or "completed"') == "loading commenced or completed"


# ─── TestTransposedTable ────────────────────────────────────────────────────


class TestParseTransposedPipeSections:
    """Tests for _parse_transposed_pipe_sections()."""

    def test_basic_pipe_rows(self) -> None:
        text = (
            "| Name | Alice | Bob |\n"
            "|---|---|---|\n"
            "| Age | 25 | 30 |"
        )
        rows = _parse_transposed_pipe_sections(text)
        assert rows is not None
        assert len(rows) == 2  # separator skipped
        assert rows[0] == ["Name", "Alice", "Bob"]
        assert rows[1] == ["Age", "25", "30"]

    def test_multiple_fragments(self) -> None:
        """Multiple pipe-table fragments on one page (Queensland pattern)."""
        text = (
            "| Ref | QB001 | QB002 |\n"
            "|---|---|---|\n"
            "| Ship | STAR | MOON |\n"
            "Some text in between\n"
            "| Port | Brisbane | Brisbane |\n"
            "|---|---|---|\n"
            "| ETA | 01-09-2025 | 15-09-2025 |"
        )
        rows = _parse_transposed_pipe_sections(text)
        assert rows is not None
        assert len(rows) == 4

    def test_inconsistent_column_count_returns_none(self) -> None:
        text = "| A | B |\n|---|---|\n| X | Y | Z |"
        assert _parse_transposed_pipe_sections(text) is None

    def test_no_pipe_rows_returns_none(self) -> None:
        text = "Just plain text\nNo tables here"
        assert _parse_transposed_pipe_sections(text) is None

    def test_single_column_returns_none(self) -> None:
        text = "| A |\n|---|\n| 1 |"
        assert _parse_transposed_pipe_sections(text) is None

    def test_csv_continuation_joined(self) -> None:
        """Unterminated CSV-quoted first cell gets continuation joined."""
        text = (
            '|"Loading ""commenced"" or|Completed|Completed|\n'
            "\n"
            '""completed"""\n'
        )
        rows = _parse_transposed_pipe_sections(text)
        assert rows is not None
        assert len(rows) == 1
        assert rows[0][0] == '"Loading ""commenced"" or ""completed"""'
        assert rows[0][1] == "Completed"

    def test_non_csv_continuation_not_joined(self) -> None:
        """Non-CSV text across blank line is NOT joined."""
        text = (
            "| Time of Ship | 14:00 | 15:00 |\n"
            "\n"
            "Date of Arrival\n"
            "\n"
            "| Port | Brisbane | Sydney |"
        )
        rows = _parse_transposed_pipe_sections(text)
        assert rows is not None
        assert len(rows) == 2
        assert rows[0][0] == "Time of Ship"
        assert rows[1][0] == "Port"

    def test_empty_first_cell_skipped(self) -> None:
        """Pipe rows with empty first cell are skipped (not labels)."""
        text = (
            "| Name | Alice | Bob |\n"
            "|---|---|---|\n"
            "|| extra | data |\n"
            "| Age | 25 | 30 |"
        )
        rows = _parse_transposed_pipe_sections(text)
        assert rows is not None
        assert len(rows) == 2
        assert rows[0][0] == "Name"
        assert rows[1][0] == "Age"


class TestTryDeterministicTransposed:
    """Tests for _try_deterministic_transposed()."""

    def test_basic_transposed_2_vessels(self) -> None:
        text = (
            "| Ref # | QB001 | QB002 |\n"
            "|---|---|---|\n"
            "| Ship Name | STAR | MOON |\n"
            "| Port | Brisbane | Sydney |\n"
            "|---|---|---|\n"
            "| ETA | 2025-09-01 | 2025-09-15 |\n"
            "| Tonnes | 30000 | 50000 |\n"
            "| Commodity | Wheat | Barley |"
        )
        schema = _schema(
            ("ref_id", "string", "Ref", ["Ref #"]),
            ("vessel_name", "string", "Vessel", ["Ship Name"]),
            ("load_port", "string", "Port", ["Port"]),
            ("eta", "string", "ETA", ["ETA"]),
            ("tons", "int", "Tonnes", ["Tonnes"]),
            ("commodity", "string", "Commodity", ["Commodity"]),
        )
        result = _try_deterministic_transposed(text, schema)
        assert result is not None
        assert len(result.records) == 2
        assert result.records[0].vessel_name == "STAR"
        assert result.records[0].load_port == "Brisbane"
        assert result.records[0].tons == "30000"
        assert result.records[1].vessel_name == "MOON"
        assert result.records[1].eta == "2025-09-15"
        assert result.metadata.table_type_inference.table_type == baml_types.TableType.TransposedTable

    def test_multi_section_transposed(self) -> None:
        """Multiple pipe-table fragments (Queensland pattern)."""
        text = (
            "| Ref | QB001 | QB002 |\n"
            "|---|---|---|\n"
            "| Ship Name | DARYA | ADAGIO |\n"
            "Some non-pipe text\n"
            "| Port | Brisbane | Brisbane |\n"
            "|---|---|---|\n"
            "| ETA | 15-09-2025 | 21-07-2025 |\n"
            "| Exporter | Arrow | Arrow |\n"
            "| Quantity (tonnes) | 30000 | 30000 |\n"
            "| Commodity | Wheat | Wheat |"
        )
        schema = _schema(
            ("ref_id", "string", "Ref", ["Ref"]),
            ("vessel_name", "string", "Vessel", ["Ship Name"]),
            ("load_port", "string", "Port", ["Port"]),
            ("eta", "string", "ETA", ["ETA"]),
            ("shipper", "string", "Shipper", ["Exporter"]),
            ("tons", "int", "Tonnes", ["Quantity (tonnes)"]),
            ("commodity", "string", "Commodity", ["Commodity"]),
        )
        result = _try_deterministic_transposed(text, schema)
        assert result is not None
        assert len(result.records) == 2
        assert result.records[0].vessel_name == "DARYA"
        assert result.records[0].tons == "30000"

    def test_whitespace_normalization(self) -> None:
        """Quantity(tonnes) matches Quantity (tonnes) via normalization."""
        text = (
            "| Ship Name | STAR | MOON |\n"
            "|---|---|---|\n"
            "| Quantity(tonnes) | 30000 | 50000 |"
        )
        schema = _schema(
            ("vessel_name", "string", "Vessel", ["Ship Name"]),
            ("tons", "int", "Tonnes", ["Quantity (tonnes)"]),
        )
        result = _try_deterministic_transposed(text, schema)
        assert result is not None
        assert result.records[0].tons == "30000"
        assert result.records[1].tons == "50000"

    def test_below_50_pct_match_returns_none(self) -> None:
        """Low match ratio → not transposed."""
        text = (
            "| Unknown Field | A | B |\n"
            "|---|---|---|\n"
            "| Other Field | C | D |\n"
            "| Third Field | E | F |"
        )
        schema = _schema(
            ("vessel_name", "string", "Vessel", ["Ship Name"]),
            ("port", "string", "Port", ["Port"]),
            ("eta", "string", "ETA", ["ETA"]),
            ("tons", "int", "Tonnes", ["Tonnes"]),
        )
        result = _try_deterministic_transposed(text, schema)
        assert result is None

    def test_non_transposed_flat_table_returns_none(self) -> None:
        """Normal flat table (>5 data columns) should not match."""
        text = (
            "| Port | Vessel | ETA | Volume | Commodity | Status |\n"
            "|---|---|---|---|---|---|\n"
            "| Brisbane | STAR | 2025-09-01 | 30000 | Wheat | Completed |"
        )
        schema = _schema(
            ("port", "string", "Port", ["Port"]),
            ("vessel_name", "string", "Vessel", ["Vessel"]),
        )
        # 6 columns total → col_count > 5 → rejected
        result = _try_deterministic_transposed(text, schema)
        assert result is None

    def test_inconsistent_column_count_returns_none(self) -> None:
        """Mismatched pipe row widths → not transposed."""
        text = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| X | Y |"  # only 2 cells
        )
        result = _try_deterministic_transposed(
            text,
            _schema(("a", "string", "A", ["A"]), ("b", "string", "B", ["B"])),
        )
        assert result is None


class TestPreHeaderSectionLabel:
    """Tests for section label detection before the first pipe-table header."""

    def test_pre_header_section_captured(self) -> None:
        text = (
            "GERALDTON\n"
            "| Port | Vessel |\n"
            "|---|---|\n"
            "| A | X |\n"
            "| B | Y |\n"
            "KWINANA\n"
            "| Port | Vessel |\n"
            "|---|---|\n"
            "| C | Z |"
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.sections) == 2
        assert parsed.sections[0] == ("GERALDTON", 0, 1)
        assert parsed.sections[1] == ("KWINANA", 2, 2)

    def test_title_not_confused_with_section(self) -> None:
        text = (
            "## My Table\n"
            "SECTION_A\n"
            "| X |\n"
            "|---|\n"
            "| 1 |"
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert parsed.title == "My Table"
        assert len(parsed.sections) == 1
        assert parsed.sections[0][0] == "SECTION_A"


class TestReHeaderPopping:
    """Tests for popping re-header rows when separator appears in data mode."""

    def test_reheader_with_different_column_count(self) -> None:
        """Re-header with different column count should be popped."""
        text = (
            "SECTION_A\n"
            "| A | B | |\n"  # 3 cells (with empty)
            "|---|---|---|\n"
            "| 1 | 2 | |\n"
            "SECTION_B\n"
            "| A | B |\n"  # 2 cells — re-header with different count
            "|---|---|\n"  # separator triggers pop
            "| 3 | 4 |"
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        # Row "| A | B |" should have been popped as re-header
        assert len(parsed.data_rows) == 2
        assert parsed.data_rows[0][:2] == ["1", "2"]
        assert parsed.data_rows[1][:2] == ["3", "4"]

    def test_no_false_pop_without_separator(self) -> None:
        """Data rows without trailing separator aren't popped."""
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.data_rows) == 2


class TestEmptyColumnAlignment:
    """Tests for aligning short rows when headers have empty columns."""

    def test_short_row_aligned_with_empty_header(self) -> None:
        text = (
            "| A | B | | C |\n"
            "|---|---|---|---|\n"
            "| 1 | 2 | | 3 |\n"  # 4 cells, matches
            "SECTION_B\n"
            "| A | B | C |\n"   # re-header (3 cells)
            "|---|---|---|\n"    # popped
            "| 4 | 5 | 6 |"    # 3 cells — should be aligned to 4
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.data_rows) == 2
        # First row: already 4 cells
        assert parsed.data_rows[0] == ["1", "2", "", "3"]
        # Second row: aligned from 3 to 4 cells
        assert parsed.data_rows[1] == ["4", "5", "", "6"]

    def test_reheader_remaps_different_column_layout(self) -> None:
        """Rows in sections with a re-header are remapped, not dropped."""
        text = (
            "| A | B | | C |\n"
            "|---|---|---|---|\n"
            "| 1 | 2 | | 3 |\n"      # 4 cells, matches global header
            "SECTION_B\n"
            "| A | | B | | C | D |\n"  # re-header (6 cells, different layout)
            "|---|---|---|---|---|---|\n"
            "| x | | y | | z | w |"    # 6 cells — remapped via re-header
        )
        parsed = _parse_pipe_table_deterministic(text)
        assert parsed is not None
        assert len(parsed.data_rows) == 2
        # First row: unchanged (matches global header)
        assert parsed.data_rows[0] == ["1", "2", "", "3"]
        # Second row: remapped — A→col0, B→col1, C→col3 (D has no match, dropped)
        assert parsed.data_rows[1] == ["x", "y", "", "z"]


class TestTransposedStatusLabel:
    """Tests for CSV-escaped status labels in transposed tables."""

    def test_csv_escaped_status_matches_alias(self) -> None:
        """Queensland-style CSV-escaped status label matches after normalization."""
        text = (
            '| Ship Name | STAR | MOON |\n'
            '|---|---|---|\n'
            '| Port | Brisbane | Brisbane |\n'
            '|---|---|---|\n'
            '| ETA | 2025-09-01 | 2025-09-15 |\n'
            '|"Loading ""commenced"" or||Completed|\n'
            '\n'
            '""completed"""\n'
        )
        schema = _schema(
            ("vessel_name", "string", "Vessel", ["Ship Name"]),
            ("load_port", "string", "Port", ["Port"]),
            ("eta", "string", "ETA", ["ETA"]),
            ("status", "string", "Status", ['Loading "commenced" or "completed"']),
        )
        result = _try_deterministic_transposed(text, schema)
        assert result is not None
        assert len(result.records) == 2
        r0 = result.records[0].model_dump()
        r1 = result.records[1].model_dump()
        assert r0.get("status") is None  # first vessel has empty status cell
        assert r1.get("status") == "Completed"


# ─── TestCommaSuffixFallback ────────────────────────────────────────────────


class TestCommaSuffixFallback:
    """Tests for comma-suffix fallback in alias matching."""

    def test_comma_unit_matches_base_alias(self) -> None:
        """'Area harvested,Th.ha.' should match alias 'Area harvested'."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Region", "Area harvested,Th.ha. / 2025", "Area harvested,Th.ha. / 2024"],
            data_rows=[["Russia", "100", "90"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("metric", "string", "Metric", ["Area harvested"]),
            ("value", "float", "Value", ["2025", "2024"]),
            ("year", "int", "Year", ["2025", "2024"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2

    def test_no_false_positive_when_before_comma_not_alias(self) -> None:
        """'Revenue, domestic' with no alias for 'Revenue' → no fallback match."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["Revenue, domestic", "Cost"],
            data_rows=[["100", "200"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("cost", "float", "Cost", ["Cost"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        # Should succeed (unmatched "Revenue, domestic" is just dropped)
        assert not unmatched
        assert result is not None

    def test_full_string_takes_priority(self) -> None:
        """If 'X,Y' itself is an alias, comma fallback should not fire."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["X,Y"],
            data_rows=[["100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("xy", "float", "XY", ["X,Y"]),
            ("x", "float", "X", ["X"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        # Should match "X,Y" exactly, not fall back to "X"
        assert result.records[0].xy == "100"


# ─── TestEmptyHeaderInference ──────────────────────────────────────────────


class TestEmptyHeaderInference:
    """Tests for blank-header text-column inference (Phase 2.5)."""

    def test_single_blank_header_with_text_data(self) -> None:
        """Blank header col with text data → assigned to unmatched string col."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["", "Value"],
            data_rows=[["Russia", "100"], ["France", "200"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2
        assert result.records[0].region == "Russia"
        assert result.records[0].value == "100"

    def test_no_inference_with_two_blank_headers(self) -> None:
        """Two blank-header text columns → ambiguous, no inference."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["", "", "Value"],
            data_rows=[["Russia", "Central", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("district", "string", "District", ["District"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        # Should still succeed (blank cols dropped), but no inference
        assert result is not None
        # Neither blank-header col gets assigned since there are 2
        rec = result.records[0].model_dump()
        assert rec.get("region") is None or rec.get("region") == ""

    def test_no_inference_for_numeric_blank_header(self) -> None:
        """Blank header col with numeric data → no inference."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["", "Name"],
            data_rows=[["100", "Alice"], ["200", "Bob"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("score", "float", "Score", ["Score"]),
            ("name", "string", "Name", ["Name"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        # Inference doesn't fire because col 0 is numeric
        assert result is not None
        rec = result.records[0].model_dump()
        assert rec.get("score") is None or rec.get("score") == ""

    def test_no_inference_with_multiple_unmatched_string_cols(self) -> None:
        """One blank-header text col but two unmatched string cols → no inference."""
        parsed = _DeterministicParsed(
            title=None,
            headers=["", "Value"],
            data_rows=[["Russia", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("district", "string", "District", ["District"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert result is not None
        # Inference doesn't fire since 2 string cols are unmatched
        rec = result.records[0].model_dump()
        assert rec.get("region") is None or rec.get("region") == ""


# ─── TestTitleToSchemaMatching ─────────────────────────────────────────────


class TestTitleToSchemaMatching:
    """Tests for Phase 2.3: title-to-schema matching."""

    def test_title_matches_string_alias(self) -> None:
        """Title matching a schema alias → constant dimension for all records."""
        parsed = _DeterministicParsed(
            title="RICE",
            headers=["Region", "Value"],
            data_rows=[["Russia", "100"], ["France", "200"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("crop", "string", "Crop", ["rice", "wheat"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2
        assert result.records[0].crop == "RICE"
        assert result.records[1].crop == "RICE"
        assert result.records[0].region == "Russia"

    def test_title_enables_blank_header_inference(self) -> None:
        """Title resolves one string col → Phase 2.5 can infer the other."""
        parsed = _DeterministicParsed(
            title="RICE",
            headers=["", "Target 2025", "2025"],
            data_rows=[["Russia", "500", "400"], ["France", "300", "250"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("crop", "string", "Crop", ["rice", "wheat"]),
            ("area", "float", "Target area", ["Target 2025"]),
            ("value", "float", "Value", ["2025"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert len(result.records) == 2
        # Title resolves crop, Phase 2.5 resolves region from blank header
        assert result.records[0].crop == "RICE"
        assert result.records[0].region == "Russia"
        assert result.records[1].region == "France"

    def test_title_no_match(self) -> None:
        """Title with no alias match → no effect."""
        parsed = _DeterministicParsed(
            title="Summary Report",
            headers=["Region", "Value"],
            data_rows=[["Russia", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        # No crop column assigned since title doesn't match any alias
        rec = result.records[0].model_dump()
        assert rec.get("crop") is None

    def test_title_not_used_when_header_already_matches(self) -> None:
        """Title not used when the column is already matched from headers."""
        parsed = _DeterministicParsed(
            title="corn",
            headers=["Region", "rice / Value", "wheat / Value"],
            data_rows=[["Russia", "100", "200"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("crop", "string", "Crop", ["rice", "wheat", "corn"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        # Crop comes from headers "rice" and "wheat" (pivot groups), not title "corn"
        crop_values = [r.crop for r in result.records]
        assert "rice" in crop_values
        assert "wheat" in crop_values
        assert "corn" not in crop_values

    def test_title_case_insensitive(self) -> None:
        """Title matching is case-insensitive via alias normalization."""
        parsed = _DeterministicParsed(
            title="SUNFLOWER",
            headers=["Region", "Value"],
            data_rows=[["Russia", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("crop", "string", "Crop", ["sunflower"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert result.records[0].crop == "SUNFLOWER"

    def test_title_substring_match(self) -> None:
        """Alias as word-boundary substring of title → match."""
        parsed = _DeterministicParsed(
            title="Winter sowing of grains and grasses",
            headers=["Region", "Value"],
            data_rows=[["Russia", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("crop", "string", "Crop", ["grains and grasses", "wheat"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert not unmatched
        assert result is not None
        assert result.records[0].crop == "Winter sowing of grains and grasses"

    def test_title_substring_no_partial_word(self) -> None:
        """Alias must match at word boundaries — no partial word matches."""
        parsed = _DeterministicParsed(
            title="Export price analysis",
            headers=["Region", "Value"],
            data_rows=[["Russia", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            # "port" is a substring of "Export" but not at word boundary
            ("location", "string", "Port", ["port"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert result is not None
        rec = result.records[0].model_dump()
        assert rec.get("location") is None

    def test_title_substring_ambiguous(self) -> None:
        """Two aliases match as substrings → ambiguous, no match."""
        parsed = _DeterministicParsed(
            title="Spring sowing of wheat and barley",
            headers=["Region", "Value"],
            data_rows=[["Russia", "100"]],
            sections=[],
            aggregation_rows=[],
        )
        schema = _schema(
            ("region", "string", "Region", ["Region"]),
            ("crop", "string", "Crop", ["wheat"]),
            ("grain", "string", "Grain type", ["barley"]),
            ("value", "float", "Value", ["Value"]),
        )
        result, unmatched = _map_to_schema_deterministic(parsed, schema)
        assert result is not None
        rec = result.records[0].model_dump()
        # Two different string columns match → ambiguous → no assignment
        assert rec.get("crop") is None
        assert rec.get("grain") is None


# ─── TestCellIsNumeric and TestIsTextDataColumn ────────────────────────────


class TestCellIsNumeric:
    """Tests for _cell_is_numeric helper."""

    def test_integer(self) -> None:
        assert _cell_is_numeric("1234") is True

    def test_comma_decimal(self) -> None:
        assert _cell_is_numeric("1234,5") is True

    def test_year_is_numeric(self) -> None:
        """4-digit years are numeric — no year exclusion at cell level."""
        assert _cell_is_numeric("2025") is True

    def test_text(self) -> None:
        assert _cell_is_numeric("Russia") is False

    def test_empty(self) -> None:
        assert _cell_is_numeric("") is False


class TestIsTextDataColumn:
    """Tests for _is_text_data_column helper."""

    def test_text_column(self) -> None:
        rows = [["Russia"], ["France"], ["Germany"]]
        assert _is_text_data_column(rows, 0) is True

    def test_numeric_column(self) -> None:
        rows = [["100"], ["200"], ["300"]]
        assert _is_text_data_column(rows, 0) is False

    def test_empty_column(self) -> None:
        rows = [[""], [""], [""]]
        assert _is_text_data_column(rows, 0) is False

    def test_mixed_mostly_text(self) -> None:
        rows = [["Russia"], ["100"], ["France"]]
        assert _is_text_data_column(rows, 0) is True
