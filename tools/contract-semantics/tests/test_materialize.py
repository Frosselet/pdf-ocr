"""Tests for contract materialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.geonames import GeoNamesAdapter
from contract_semantics.materialize import materialize_contract


@pytest.fixture
def annotated_contract(tmp_path: Path) -> Path:
    """Create a minimal annotated contract for testing."""
    contract = {
        "provider": "test",
        "outputs": {
            "harvest": {
                "category": "harvest",
                "schema": {
                    "description": "Test schema",
                    "columns": [
                        {
                            "name": "Crop",
                            "type": "string",
                            "aliases": ["wheat", "barley"],
                            "concept_uris": [
                                {"uri": "http://aims.fao.org/aos/agrovoc/c_8373", "label": "wheat"},
                                {"uri": "http://aims.fao.org/aos/agrovoc/c_631", "label": "barley"},
                            ],
                            "resolve": {
                                "languages": ["en"],
                                "label_types": ["prefLabel", "altLabel"],
                            },
                        },
                        {
                            "name": "Region",
                            "type": "string",
                            "aliases": ["Region"],
                            "concept_uris": [
                                {"uri": "https://sws.geonames.org/524894/", "label": "Moscow Oblast"},
                            ],
                            "resolve": {
                                "source": "geonames",
                                "languages": ["en", "ru"],
                                "enrich_fields": ["lat", "lng", "countryCode"],
                            },
                        },
                        {
                            "name": "Value",
                            "type": "float",
                            "aliases": ["{YYYY}"],
                        },
                    ],
                },
            }
        },
    }
    path = tmp_path / "annotated.json"
    path.write_text(json.dumps(contract, indent=2))
    return path


class TestMaterializeContract:
    def test_aliases_enriched(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][0]
        assert crop_col["name"] == "Crop"
        # Original aliases preserved
        assert "wheat" in crop_col["aliases"]
        assert "barley" in crop_col["aliases"]
        # New aliases added (e.g. altLabels)
        assert any("common wheat" in a for a in crop_col["aliases"])

    def test_annotation_fields_removed(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][0]
        assert "concept_uris" not in crop_col
        assert "resolve" not in crop_col

    def test_unannotated_columns_unchanged(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        value_col = result["outputs"]["harvest"]["schema"]["columns"][2]
        assert value_col["name"] == "Value"
        assert value_col["aliases"] == ["{YYYY}"]

    def test_output_file_written(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "materialized.json"
        materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
            output_path=out_path,
        )
        assert out_path.exists()
        result = json.loads(out_path.read_text())
        assert result["provider"] == "test"

    def test_geo_sidecar_written(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "geo.json"
        materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
            geo_sidecar_path=sidecar_path,
        )
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text())
        assert "_provenance" in sidecar
        assert "regions" in sidecar
        moscow = sidecar["regions"]["Moscow Oblast"]
        assert moscow["geoname_id"] == 524894
        assert moscow["lat"] == 55.75
        assert moscow["lng"] == 37.62
        assert moscow["countryCode"] == "RU"

    def test_merge_strategy_resolved_only(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
            merge_strategy="resolved_only",
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][0]
        # Manual aliases that aren't in ontology should be gone
        # "wheat" and "barley" are both manual AND resolved, so they stay
        assert "wheat" in crop_col["aliases"]
