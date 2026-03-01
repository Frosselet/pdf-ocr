"""Tests for build_context_data() — the core bridge logic between contract-semantics and docpact.

Tests use build_context_data() which returns a plain dict, avoiding the docpact
import chain (docpact.__init__ -> interpret -> baml_client). The full
build_semantic_context() -> SemanticContext flow is tested in
tests/test_semantic_integration.py in the docpact repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.context import build_context_data
from contract_semantics.geonames import GeoNamesAdapter


@pytest.fixture
def annotated_contract(tmp_path: Path) -> Path:
    """Minimal annotated contract with AGROVOC crops, GeoNames regions, and metrics."""
    contract = {
        "provider": "test",
        "outputs": {
            "harvest": {
                "category": "harvest",
                "schema": {
                    "description": "Test harvest schema",
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
                                "languages": ["en", "ru"],
                                "label_types": ["prefLabel", "altLabel"],
                            },
                        },
                        {
                            "name": "Region",
                            "type": "string",
                            "aliases": ["Region"],
                            "concept_uris": [
                                {"uri": "https://sws.geonames.org/524894/", "label": "Moscow Oblast"},
                                {"uri": "https://sws.geonames.org/472039/", "label": "Voronezh Oblast"},
                            ],
                            "resolve": {
                                "source": "geonames",
                                "languages": ["en", "ru"],
                            },
                        },
                        {
                            "name": "Metric",
                            "type": "string",
                            "aliases": ["Area harvested", "collected", "Yield"],
                            "concept_uris": [
                                {"uri": "http://aims.fao.org/aos/agrovoc/c_330918", "label": "area harvested"},
                            ],
                            "resolve": {
                                "languages": ["en"],
                                "label_types": ["prefLabel"],
                            },
                        },
                        {
                            "name": "Value",
                            "type": "float",
                            "aliases": ["{YYYY}"],
                        },
                    ],
                },
            },
        },
    }
    path = tmp_path / "annotated.json"
    path.write_text(json.dumps(contract, indent=2))
    return path


@pytest.fixture
def no_semantic_contract(tmp_path: Path) -> Path:
    """Contract with no concept_uris at all."""
    contract = {
        "provider": "plain",
        "outputs": {
            "data": {
                "category": "data",
                "schema": {
                    "description": "Plain schema",
                    "columns": [
                        {"name": "Name", "type": "string", "aliases": ["name"]},
                        {"name": "Value", "type": "float", "aliases": ["val"]},
                    ],
                },
            },
        },
    }
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(contract, indent=2))
    return path


class TestBuildContextData:
    def test_build_basic(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Returns dict with resolved_aliases and valid_values populated."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        assert data["resolved_aliases"]
        assert data["valid_values"]
        assert data["resolved_at"]  # ISO timestamp present

    def test_resolved_aliases_structure(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """resolved_aliases has correct {output: {column: [alias, ...]}} nesting."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        assert "harvest" in data["resolved_aliases"]
        harvest = data["resolved_aliases"]["harvest"]
        assert "Crop" in harvest
        assert "Region" in harvest
        assert isinstance(harvest["Crop"], list)
        assert all(isinstance(a, str) for a in harvest["Crop"])

    def test_valid_values_structure(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """valid_values has correct {output: {column: [value, ...]}} nesting."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        assert "harvest" in data["valid_values"]
        harvest_valid = data["valid_values"]["harvest"]
        assert "Crop" in harvest_valid
        assert isinstance(harvest_valid["Crop"], list)

    def test_agrovoc_aliases_resolved(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Crop column gets multilingual labels from AGROVOC (wheat -> пшеница)."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        crop_aliases = data["resolved_aliases"]["harvest"]["Crop"]
        # Russian labels from fixture: пшеница (wheat), ячмень (barley)
        assert "пшеница" in crop_aliases
        assert "ячмень" in crop_aliases
        # English altLabel from fixture
        assert "common wheat" in crop_aliases
        assert "barleycorn" in crop_aliases

    def test_geonames_aliases_resolved(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Region column gets bilingual aliases from GeoNames."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        region_aliases = data["resolved_aliases"]["harvest"]["Region"]
        # Russian alternate names from fixture
        assert "Московская область" in region_aliases
        assert "Воронежская область" in region_aliases
        # English names from fixture
        assert "Moscow Oblast" in region_aliases

    def test_metric_aliases_resolved(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Metric column: manual aliases preserved in union merge."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        # Metric concept c_330918 is not in the fixture graph, so the adapter
        # returns empty. But the manual aliases are preserved in union merge.
        metric_aliases = data["resolved_aliases"]["harvest"]["Metric"]
        assert "Area harvested" in metric_aliases
        assert "collected" in metric_aliases
        assert "Yield" in metric_aliases

    def test_cache_roundtrip(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
        tmp_path: Path,
    ) -> None:
        """Context data serializes to JSON and deserializes identically."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )

        # Write to JSON file
        cache_file = tmp_path / "ctx_cache.json"
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Read back
        with open(cache_file) as f:
            loaded = json.load(f)

        assert loaded["resolved_aliases"] == data["resolved_aliases"]
        assert loaded["valid_values"] == data["valid_values"]
        assert loaded["resolved_at"] == data["resolved_at"]
        assert loaded["adapter_versions"] == data["adapter_versions"]

    def test_merge_strategy_union(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Union strategy: manual + resolved aliases both present."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
            merge_strategy="union",
        )
        crop_aliases = data["resolved_aliases"]["harvest"]["Crop"]
        # Manual aliases preserved
        assert "wheat" in crop_aliases
        assert "barley" in crop_aliases
        # Resolved aliases added
        assert "пшеница" in crop_aliases

    def test_merge_strategy_resolved_only(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """resolved_only: only ontology-derived aliases, manual excluded."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
            merge_strategy="resolved_only",
        )
        crop_aliases = data["resolved_aliases"]["harvest"]["Crop"]
        # Resolved aliases present — these come from the ontology
        # "wheat" and "barley" are ontology prefLabels, so they're still in resolved
        assert "wheat" in crop_aliases
        assert "пшеница" in crop_aliases

    def test_no_semantic_columns(
        self,
        no_semantic_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Contract without concept_uris -> empty context (no crash)."""
        data = build_context_data(
            no_semantic_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        assert data["resolved_aliases"] == {}
        assert data["valid_values"] == {}
        assert data["resolved_at"]  # still populated

    def test_adapter_versions_recorded(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """adapter_versions field populated with adapter info."""
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        assert "agrovoc" in data["adapter_versions"]
        assert data["adapter_versions"]["agrovoc"] == "AgrovocAdapter"
        assert "geonames" in data["adapter_versions"]
        assert data["adapter_versions"]["geonames"] == "GeoNamesAdapter"
