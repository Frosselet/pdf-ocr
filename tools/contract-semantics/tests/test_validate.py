"""Tests for SHACL validation and shape generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph

from contract_semantics.validate import generate_shapes, records_to_graph

pyshacl = pytest.importorskip("pyshacl", reason="pyshacl not installed")


SHAPES_DIR = Path(__file__).resolve().parent.parent / "shapes"


class TestRecordsToGraph:
    def test_basic_conversion(self) -> None:
        records = [
            {"Region": "Moscow Oblast", "Metric": "Yield", "Value": "3.5", "Year": "2025"},
        ]
        schema = [
            {"name": "Region", "type": "string"},
            {"name": "Metric", "type": "string"},
            {"name": "Value", "type": "float"},
            {"name": "Year", "type": "int"},
        ]
        g = records_to_graph(records, schema)
        assert len(list(g.triples((None, None, None)))) > 0

    def test_empty_values_skipped(self) -> None:
        records = [{"Region": "Moscow", "Value": "", "Year": None}]
        g = records_to_graph(records)
        # Should have Region + rdf:type = 2 triples (empty Value and None Year skipped)
        triples = list(g.triples((None, None, None)))
        assert len(triples) == 2

    def test_typed_literals(self) -> None:
        records = [{"Value": "3.5", "Year": "2025"}]
        schema = [
            {"name": "Value", "type": "float"},
            {"name": "Year", "type": "int"},
        ]
        g = records_to_graph(records, schema)
        # Check XSD types are assigned
        from rdflib.namespace import XSD
        from contract_semantics.validate import DOCPACT
        value_obj = list(g.objects(predicate=DOCPACT.Value))[0]
        assert value_obj.datatype == XSD.decimal
        year_obj = list(g.objects(predicate=DOCPACT.Year))[0]
        assert year_obj.datatype == XSD.integer


class TestValidateRecords:
    def test_valid_records(self) -> None:
        from contract_semantics.validate import validate_records

        records = [
            {"Region": "Moscow Oblast", "Metric": "Yield", "Value": "3.5", "Year": "2025"},
            {"Region": "Voronezh Oblast", "Metric": "collected", "Value": "100.0", "Year": "2024"},
        ]
        schema = [
            {"name": "Region", "type": "string"},
            {"name": "Metric", "type": "string"},
            {"name": "Value", "type": "float"},
            {"name": "Year", "type": "int"},
        ]
        shapes_path = SHAPES_DIR / "harvest_output.ttl"
        conforms, report, _graph = validate_records(records, shapes_path, schema)
        assert conforms, f"Validation should pass but failed:\n{report}"

    def test_invalid_metric(self) -> None:
        from contract_semantics.validate import validate_records

        records = [
            {"Region": "Moscow", "Metric": "InvalidMetric", "Value": "3.5", "Year": "2025"},
        ]
        schema = [
            {"name": "Region", "type": "string"},
            {"name": "Metric", "type": "string"},
            {"name": "Value", "type": "float"},
            {"name": "Year", "type": "int"},
        ]
        shapes_path = SHAPES_DIR / "harvest_output.ttl"
        conforms, report, _graph = validate_records(records, shapes_path, schema)
        assert not conforms, "Validation should fail for invalid metric"


class TestGenerateShapes:
    def test_generate_from_contract(self, tmp_path: Path) -> None:
        contract = {
            "outputs": {
                "test": {
                    "schema": {
                        "columns": [
                            {"name": "Name", "type": "string", "aliases": ["Alpha", "Beta"]},
                            {"name": "Count", "type": "int"},
                            {"name": "Score", "type": "float"},
                        ]
                    }
                }
            }
        }
        contract_path = tmp_path / "contract.json"
        contract_path.write_text(json.dumps(contract))

        results = generate_shapes(contract_path, output_dir=tmp_path / "shapes")
        assert "test" in results
        shape_path = Path(results["test"])
        assert shape_path.exists()

        # Verify it's valid Turtle
        g = Graph()
        g.parse(str(shape_path), format="turtle")
        assert len(list(g.triples((None, None, None)))) > 0

    def test_string_column_gets_sh_in(self, tmp_path: Path) -> None:
        contract = {
            "outputs": {
                "test": {
                    "schema": {
                        "columns": [
                            {"name": "Status", "type": "string", "aliases": ["active", "inactive"]},
                        ]
                    }
                }
            }
        }
        contract_path = tmp_path / "contract.json"
        contract_path.write_text(json.dumps(contract))

        results = generate_shapes(contract_path, output_dir=tmp_path / "shapes")
        content = Path(results["test"]).read_text()
        assert "sh:in" in content
        assert '"active"' in content
        assert '"inactive"' in content
