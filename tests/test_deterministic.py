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
    _parse_pipe_table_deterministic,
    _map_to_schema_deterministic,
    _try_deterministic,
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
