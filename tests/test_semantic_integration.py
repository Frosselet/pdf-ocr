"""Cross-package integration tests: contract-semantics -> docpact pipeline.

Tests the full semantic flow using fixture adapters (no network, no LLM calls).
Verifies that build_context_data() produces data that correctly enriches aliases,
drives pre-flight checks, and validates extracted output when wrapped in a
SemanticContext.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import SKOS

# Make contract_semantics importable from the sibling tools directory
_CS_SRC = Path(__file__).resolve().parent.parent / "tools" / "contract-semantics" / "src"
if str(_CS_SRC) not in sys.path:
    sys.path.insert(0, str(_CS_SRC))

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.context import build_context_data
from contract_semantics.geonames import GeoNamesAdapter
from docpact.contracts import load_contract
from docpact.semantics import (
    SemanticContext,
    preflight_check,
    validate_output,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

AGROVOC = Namespace("http://aims.fao.org/aos/agrovoc/")


def _build_agrovoc_fixture() -> Graph:
    """Replicate the contract-semantics test fixture graph."""
    g = Graph()

    wheat = AGROVOC["c_8373"]
    g.add((wheat, SKOS.prefLabel, Literal("wheat", lang="en")))
    g.add((wheat, SKOS.prefLabel, Literal("пшеница", lang="ru")))
    g.add((wheat, SKOS.altLabel, Literal("common wheat", lang="en")))

    barley = AGROVOC["c_631"]
    g.add((barley, SKOS.prefLabel, Literal("barley", lang="en")))
    g.add((barley, SKOS.prefLabel, Literal("ячмень", lang="ru")))
    g.add((barley, SKOS.altLabel, Literal("barleycorn", lang="en")))

    maize = AGROVOC["c_12332"]
    g.add((maize, SKOS.prefLabel, Literal("maize", lang="en")))
    g.add((maize, SKOS.prefLabel, Literal("кукуруза", lang="ru")))
    g.add((maize, SKOS.altLabel, Literal("corn", lang="en")))

    buckwheat = AGROVOC["c_1094"]
    g.add((buckwheat, SKOS.prefLabel, Literal("buckwheat", lang="en")))
    g.add((buckwheat, SKOS.prefLabel, Literal("гречиха", lang="ru")))

    sunflowers = AGROVOC["c_7440"]
    g.add((sunflowers, SKOS.prefLabel, Literal("sunflowers", lang="en")))
    g.add((sunflowers, SKOS.prefLabel, Literal("подсолнечник", lang="ru")))
    g.add((sunflowers, SKOS.altLabel, Literal("sunflower", lang="en")))

    return g


def _build_geonames_fixture() -> tuple[dict[int, dict], dict[int, list[tuple[str, str]]]]:
    """Replicate the contract-semantics test fixture data."""
    features = {
        524894: {
            "geoname_id": 524894,
            "name": "Moscow Oblast",
            "asciiname": "Moscow Oblast",
            "alternatenames": "Moskovskaya Oblast",
            "lat": 55.75,
            "lng": 37.62,
            "feature_class": "A",
            "feature_code": "ADM1",
            "country_code": "RU",
            "admin1_code": "48",
            "population": 7500000,
        },
        472039: {
            "geoname_id": 472039,
            "name": "Voronezh Oblast",
            "asciiname": "Voronezh Oblast",
            "alternatenames": "Voronezhskaya Oblast",
            "lat": 51.67,
            "lng": 39.21,
            "feature_class": "A",
            "feature_code": "ADM1",
            "country_code": "RU",
            "admin1_code": "86",
            "population": 2300000,
        },
        511180: {
            "geoname_id": 511180,
            "name": "Penza Oblast",
            "asciiname": "Penza Oblast",
            "alternatenames": "Penzenskaya Oblast",
            "lat": 53.12,
            "lng": 44.10,
            "feature_class": "A",
            "feature_code": "ADM1",
            "country_code": "RU",
            "admin1_code": "57",
            "population": 1300000,
        },
    }
    alt_names: dict[int, list[tuple[str, str]]] = {
        524894: [("ru", "Московская область"), ("en", "Moscow Region")],
        472039: [("ru", "Воронежская область")],
        511180: [("ru", "Пензенская область")],
    }
    return features, alt_names


@pytest.fixture
def agrovoc_adapter() -> AgrovocAdapter:
    return AgrovocAdapter(graph=_build_agrovoc_fixture())


@pytest.fixture
def geonames_adapter() -> GeoNamesAdapter:
    features, alt_names = _build_geonames_fixture()
    return GeoNamesAdapter(features=features, alternate_names=alt_names)


@pytest.fixture
def integration_contract(tmp_path: Path) -> Path:
    """Minimal annotated contract for integration testing.

    Uses concept URIs present in the fixture data. Includes semantic.validate=True
    on the Crop column to test post-extraction validation.
    """
    contract = {
        "provider": "integration_test",
        "outputs": {
            "harvest": {
                "category": "harvest",
                "schema": {
                    "description": "Integration test harvest",
                    "columns": [
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
                            "semantic": {
                                "validate": True,
                            },
                        },
                        {
                            "name": "Value",
                            "type": "float",
                            "aliases": ["2025"],
                        },
                    ],
                },
            },
        },
    }
    path = tmp_path / "integration.json"
    path.write_text(json.dumps(contract, indent=2))
    return path


def _build_semantic_context(
    contract_path: Path,
    agrovoc: AgrovocAdapter,
    geonames: GeoNamesAdapter,
) -> SemanticContext:
    """Build a SemanticContext using build_context_data() + from_dict()."""
    data = build_context_data(
        contract_path,
        agrovoc=agrovoc,
        geonames=geonames,
    )
    return SemanticContext.from_dict(data)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSemanticContextEnrichesAliases:
    """Build context -> pass to pipeline -> schema columns get extra aliases."""

    def test_resolved_aliases_available_for_pipeline(
        self,
        integration_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        ctx = _build_semantic_context(
            integration_contract, agrovoc_adapter, geonames_adapter,
        )

        # Verify the aliases_for() accessor works for pipeline consumption
        crop_aliases = ctx.aliases_for("harvest", "Crop")
        assert "wheat" in crop_aliases
        assert "пшеница" in crop_aliases  # Russian from AGROVOC

        region_aliases = ctx.aliases_for("harvest", "Region")
        assert "Moscow Oblast" in region_aliases
        assert "Московская область" in region_aliases  # Russian from GeoNames


class TestPreflightWithRealContext:
    """Build context -> preflight_check() reports coverage against actual headers."""

    def test_preflight_matches_resolved_aliases(
        self,
        integration_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        ctx = _build_semantic_context(
            integration_contract, agrovoc_adapter, geonames_adapter,
        )

        # Load the contract to get the OutputSpec
        cc = load_contract(integration_contract)
        output_spec = cc.outputs["harvest"]

        # Simulate pipe-table data with headers that include a Russian alias
        pipe_table = (
            "| Region | пшеница | 2025 |\n"
            "|--------|---------|------|\n"
            "| Moscow Oblast | 100 | 200 |\n"
        )

        report = preflight_check(pipe_table, output_spec, ctx)

        # "пшеница" should match via resolved aliases (Crop column)
        # "Region" matches directly, "2025" matches Value alias
        assert report.header_coverage > 0.0
        # "пшеница" is a resolved alias for Crop, so it should be matched
        assert "пшеница" not in report.unmatched_headers

    def test_preflight_without_context_misses_resolved(
        self,
        integration_contract: Path,
    ) -> None:
        """Without SemanticContext, Russian aliases are not matched."""
        cc = load_contract(integration_contract)
        output_spec = cc.outputs["harvest"]

        pipe_table = (
            "| Region | пшеница | 2025 |\n"
            "|--------|---------|------|\n"
            "| Moscow Oblast | 100 | 200 |\n"
        )

        # No semantic context — only manual aliases available
        report = preflight_check(pipe_table, output_spec)

        # "пшеница" is NOT in manual aliases, so it should be unmatched
        assert "пшеница" in report.unmatched_headers


class TestValidationWithRealContext:
    """Build context -> validate_output() catches invalid values, passes valid ones."""

    def test_valid_values_pass(
        self,
        integration_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        ctx = _build_semantic_context(
            integration_contract, agrovoc_adapter, geonames_adapter,
        )
        cc = load_contract(integration_contract)
        output_spec = cc.outputs["harvest"]

        df = pd.DataFrame({
            "Region": ["Moscow Oblast", "Voronezh Oblast"],
            "Crop": ["wheat", "barley"],
            "Value": [100.0, 50.0],
        })

        report = validate_output(df, output_spec, ctx)
        assert report.invalid_count == 0

    def test_invalid_value_caught(
        self,
        integration_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        ctx = _build_semantic_context(
            integration_contract, agrovoc_adapter, geonames_adapter,
        )
        cc = load_contract(integration_contract)
        output_spec = cc.outputs["harvest"]

        df = pd.DataFrame({
            "Region": ["Moscow Oblast"],
            "Crop": ["UNKNOWN_CROP"],
            "Value": [100.0],
        })

        report = validate_output(df, output_spec, ctx)
        assert report.invalid_count > 0
        assert any(f.value == "UNKNOWN_CROP" for f in report.findings)

    def test_russian_values_pass_with_context(
        self,
        integration_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Russian crop names (resolved from AGROVOC) are valid."""
        ctx = _build_semantic_context(
            integration_contract, agrovoc_adapter, geonames_adapter,
        )
        cc = load_contract(integration_contract)
        output_spec = cc.outputs["harvest"]

        df = pd.DataFrame({
            "Region": ["Moscow Oblast"],
            "Crop": ["пшеница"],  # Russian for wheat
            "Value": [100.0],
        })

        report = validate_output(df, output_spec, ctx)
        assert report.invalid_count == 0, (
            f"Russian crop 'пшеница' should be valid. Findings: {report.findings}"
        )


class TestContextJsonRoundtripInPipeline:
    """Build -> save to JSON -> load from JSON -> verify same results."""

    def test_roundtrip_preserves_semantic_behavior(
        self,
        integration_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
        tmp_path: Path,
    ) -> None:
        # Build context from adapters
        ctx_original = _build_semantic_context(
            integration_contract, agrovoc_adapter, geonames_adapter,
        )

        # Save to JSON and reload
        cache_file = tmp_path / "ctx.json"
        ctx_original.to_json(cache_file)
        ctx_loaded = SemanticContext.from_json(cache_file)

        # Both should produce identical pipeline behavior
        cc = load_contract(integration_contract)
        output_spec = cc.outputs["harvest"]

        df = pd.DataFrame({
            "Region": ["Moscow Oblast"],
            "Crop": ["пшеница"],
            "Value": [100.0],
        })

        report_original = validate_output(df, output_spec, ctx_original)
        report_loaded = validate_output(df, output_spec, ctx_loaded)

        assert report_original.invalid_count == report_loaded.invalid_count
        assert report_original.valid_count == report_loaded.valid_count

        # Pre-flight should also match
        pipe_table = "| Region | пшеница | 2025 |\n|---|---|---|\n| Moscow | 1 | 2 |\n"
        pf_original = preflight_check(pipe_table, output_spec, ctx_original)
        pf_loaded = preflight_check(pipe_table, output_spec, ctx_loaded)

        assert pf_original.header_coverage == pf_loaded.header_coverage
        assert pf_original.unmatched_headers == pf_loaded.unmatched_headers
