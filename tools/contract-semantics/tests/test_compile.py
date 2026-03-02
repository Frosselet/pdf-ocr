"""Tests for JSON-LD contract compilation (Phase 1.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.compile import compile_contract
from contract_semantics.geonames import GeoNamesAdapter, LEVEL_MAP


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_jsonld(tmp_path: Path) -> Path:
    """Minimal JSON-LD contract with AGROVOC grounding only."""
    contract = {
        "@context": "https://ckir.dev/context/v1",
        "@type": "ExtractionContract",
        "provider": "test",
        "languages": ["en"],
        "outputs": {
            "harvest": {
                "category": "harvest",
                "schema": {
                    "description": "Test schema",
                    "columns": [
                        {
                            "name": "Crop",
                            "type": "string",
                            "grounding": {
                                "ontology": "agrovoc",
                                "concepts": ["wheat", "barley"],
                            },
                        },
                    ],
                },
            }
        },
    }
    path = tmp_path / "test.jsonld"
    path.write_text(json.dumps(contract, indent=2))
    return path


@pytest.fixture
def geo_jsonld(tmp_path: Path) -> Path:
    """JSON-LD contract with GeoNames grounding only."""
    contract = {
        "@context": "https://ckir.dev/context/v1",
        "@type": "ExtractionContract",
        "provider": "test_geo",
        "languages": ["en", "ru"],
        "outputs": {
            "data": {
                "category": "data",
                "schema": {
                    "description": "Test schema",
                    "columns": [
                        {
                            "name": "Region",
                            "type": "string",
                            "aliases": ["Region"],
                            "grounding": {
                                "ontology": "geonames",
                                "country": "RU",
                                "level": "ADM1",
                                "enrich": ["lat", "lng"],
                            },
                        },
                    ],
                },
            }
        },
    }
    path = tmp_path / "test_geo.jsonld"
    path.write_text(json.dumps(contract, indent=2))
    return path


@pytest.fixture
def mixed_jsonld(tmp_path: Path) -> Path:
    """JSON-LD contract with both AGROVOC and GeoNames grounding."""
    contract = {
        "@context": "https://ckir.dev/context/v1",
        "@type": "ExtractionContract",
        "provider": "test_mixed",
        "languages": ["en", "ru"],
        "outputs": {
            "harvest": {
                "category": "harvest",
                "schema": {
                    "description": "Mixed test",
                    "columns": [
                        {
                            "name": "Region",
                            "type": "string",
                            "aliases": ["Region"],
                            "grounding": {
                                "ontology": "geonames",
                                "country": "RU",
                                "level": "ADM1",
                                "enrich": ["lat", "lng", "admin1Code", "countryCode"],
                            },
                        },
                        {
                            "name": "Crop",
                            "type": "string",
                            "aliases": ["spring crops", "spring grain"],
                            "grounding": {
                                "ontology": "agrovoc",
                                "concepts": ["wheat", "barley"],
                                "variants": ["spring"],
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
    path = tmp_path / "test_mixed.jsonld"
    path.write_text(json.dumps(contract, indent=2))
    return path


# ── Adapter Extension Tests ──────────────────────────────────────────────


class TestAgrovocLookupByLabel:
    def test_lookup_by_label_exact(self, agrovoc_adapter: AgrovocAdapter) -> None:
        uri = agrovoc_adapter.lookup_by_label("wheat", language="en")
        assert uri == "http://aims.fao.org/aos/agrovoc/c_8373"

    def test_lookup_by_label_case_insensitive(self, agrovoc_adapter: AgrovocAdapter) -> None:
        uri = agrovoc_adapter.lookup_by_label("Wheat", language="en")
        assert uri == "http://aims.fao.org/aos/agrovoc/c_8373"

    def test_lookup_by_label_not_found(self, agrovoc_adapter: AgrovocAdapter) -> None:
        with pytest.raises(KeyError, match="not found"):
            agrovoc_adapter.lookup_by_label("quinoa", language="en")

    def test_lookup_by_label_russian(self, agrovoc_adapter: AgrovocAdapter) -> None:
        uri = agrovoc_adapter.lookup_by_label("пшеница", language="ru")
        assert uri == "http://aims.fao.org/aos/agrovoc/c_8373"

    def test_lookup_suggestions_in_error(self, agrovoc_adapter: AgrovocAdapter) -> None:
        with pytest.raises(KeyError, match="Similar labels"):
            # "whet" is close to "wheat" — should get fuzzy suggestions
            agrovoc_adapter.lookup_by_label("whet", language="en")


class TestGeoNamesEnumerateFeatures:
    def test_enumerate_features_adm1(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.enumerate_features("RU", "ADM1")
        assert len(results) == 3  # Moscow, Voronezh, Penza
        names = {r.name for r in results}
        assert "Moscow Oblast" in names
        assert "Voronezh Oblast" in names
        assert "Penza Oblast" in names

    def test_enumerate_features_no_match(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.enumerate_features("XX", "ADM1")
        assert results == []

    def test_enumerate_features_unknown_level(self, geonames_adapter: GeoNamesAdapter) -> None:
        with pytest.raises(KeyError, match="Unknown level"):
            geonames_adapter.enumerate_features("RU", "INVALID")

    def test_level_map_coverage(self) -> None:
        """All LEVEL_MAP entries have valid (class, code) tuples."""
        for level, (cls, code) in LEVEL_MAP.items():
            assert isinstance(cls, str) and len(cls) == 1
            assert isinstance(code, str) and len(code) >= 3
            assert level == code  # level key matches feature code


# ── Compilation Pipeline Tests ───────────────────────────────────────────


class TestCompileAgrovocGrounding:
    def test_concept_uris_generated(
        self,
        minimal_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        result = compile_contract(
            minimal_jsonld, agrovoc=agrovoc_adapter
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][0]
        # concept_uris should have been created then consumed by materializer
        assert "concept_uris" not in crop_col  # stripped by materialize
        assert "resolve" not in crop_col
        # Aliases should be populated from resolution
        assert "wheat" in crop_col["aliases"]
        assert "barley" in crop_col["aliases"]

    def test_aliases_from_resolution(
        self,
        minimal_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        result = compile_contract(
            minimal_jsonld, agrovoc=agrovoc_adapter
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][0]
        # Should have altLabels from ontology
        assert "common wheat" in crop_col["aliases"]
        assert "barleycorn" in crop_col["aliases"]


class TestCompileGeoNamesGrounding:
    def test_geonames_grounding(
        self,
        geo_jsonld: Path,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = compile_contract(
            geo_jsonld, geonames=geonames_adapter
        )
        region_col = result["outputs"]["data"]["schema"]["columns"][0]
        # All 3 fixture regions should appear as aliases
        assert "Moscow Oblast" in region_col["aliases"]
        assert "Voronezh Oblast" in region_col["aliases"]
        assert "Penza Oblast" in region_col["aliases"]
        # Manual alias preserved first
        assert region_col["aliases"][0] == "Region"

    def test_geonames_strips_grounding(
        self,
        geo_jsonld: Path,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = compile_contract(
            geo_jsonld, geonames=geonames_adapter
        )
        region_col = result["outputs"]["data"]["schema"]["columns"][0]
        assert "grounding" not in region_col
        assert "concept_uris" not in region_col
        assert "resolve" not in region_col


class TestCompileMixedGrounding:
    def test_mixed_grounding(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        columns = result["outputs"]["harvest"]["schema"]["columns"]

        # Region column
        region_col = columns[0]
        assert "Moscow Oblast" in region_col["aliases"]

        # Crop column
        crop_col = columns[1]
        assert "wheat" in crop_col["aliases"]

        # Value column (no grounding)
        value_col = columns[2]
        assert value_col["aliases"] == ["{YYYY}"]


class TestCompileNoGrounding:
    def test_columns_without_grounding_pass_through(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        value_col = result["outputs"]["harvest"]["schema"]["columns"][2]
        assert value_col["name"] == "Value"
        assert value_col["type"] == "float"
        assert value_col["aliases"] == ["{YYYY}"]


class TestCompileManualAliasesPreserved:
    def test_manual_aliases_appear_first(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        result = compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][1]
        # Manual aliases should appear before resolved ones
        assert crop_col["aliases"][0] == "spring crops"
        assert crop_col["aliases"][1] == "spring grain"


class TestCompileLanguagesFromContract:
    def test_contract_languages_flow_to_resolve(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Contract-level languages flow into resolution (Russian aliases present)."""
        result = compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][1]
        # Russian label should be present due to languages: ["en", "ru"]
        assert "пшеница" in crop_col["aliases"] or "spring пшеница" in crop_col["aliases"]


class TestCompileVariantsToPrefixPatterns:
    def test_variants_produce_prefix_patterns(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """variants: ["spring"] → "spring wheat", "spring barley" etc."""
        result = compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][1]
        assert "spring wheat" in crop_col["aliases"]
        assert "spring barley" in crop_col["aliases"]


class TestCompileEnrichFields:
    def test_enrich_flows_to_sidecar(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "geo.json"
        compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
            geo_sidecar_path=sidecar_path,
        )
        assert sidecar_path.exists()
        sidecar = json.loads(sidecar_path.read_text())
        assert "regions" in sidecar
        moscow = sidecar["regions"]["Moscow Oblast"]
        assert moscow["lat"] == 55.75
        assert moscow["lng"] == 37.62


class TestCompileCompiledFromMetadata:
    def test_compiled_from_field(
        self,
        minimal_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        result = compile_contract(
            minimal_jsonld, agrovoc=agrovoc_adapter
        )
        assert "_compiled_from" in result
        assert result["_compiled_from"] == "test.jsonld"


class TestCompileStripsJsonldFields:
    def test_context_type_stripped(
        self,
        minimal_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        result = compile_contract(
            minimal_jsonld, agrovoc=agrovoc_adapter
        )
        assert "@context" not in result
        assert "@type" not in result
        assert "domain" not in result
        assert "sourceFormat" not in result
        assert "languages" not in result


class TestCompileOutputFile:
    def test_output_file_written(
        self,
        minimal_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        tmp_path: Path,
    ) -> None:
        out_path = tmp_path / "compiled.json"
        compile_contract(
            minimal_jsonld,
            agrovoc=agrovoc_adapter,
            output_path=out_path,
        )
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["provider"] == "test"
        assert "_compiled_from" in loaded


# ── Validation / Error Tests ─────────────────────────────────────────────


class TestCompileValidation:
    def test_missing_context(self, tmp_path: Path) -> None:
        contract = {
            "@type": "ExtractionContract",
            "provider": "test",
            "outputs": {},
        }
        path = tmp_path / "no_context.jsonld"
        path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="@context"):
            compile_contract(path)

    def test_wrong_type(self, tmp_path: Path) -> None:
        contract = {
            "@context": "https://ckir.dev/context/v1",
            "@type": "SomethingElse",
            "provider": "test",
            "outputs": {},
        }
        path = tmp_path / "wrong_type.jsonld"
        path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="ExtractionContract"):
            compile_contract(path)

    def test_unknown_ontology(self, tmp_path: Path) -> None:
        contract = {
            "@context": "https://ckir.dev/context/v1",
            "@type": "ExtractionContract",
            "provider": "test",
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {
                                "name": "X",
                                "type": "string",
                                "grounding": {"ontology": "unknown"},
                            }
                        ]
                    }
                }
            },
        }
        path = tmp_path / "unknown_ontology.jsonld"
        path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="unknown.*Supported"):
            compile_contract(path)

    def test_unresolved_label(
        self,
        tmp_path: Path,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        contract = {
            "@context": "https://ckir.dev/context/v1",
            "@type": "ExtractionContract",
            "provider": "test",
            "languages": ["en"],
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {
                                "name": "Crop",
                                "type": "string",
                                "grounding": {
                                    "ontology": "agrovoc",
                                    "concepts": ["quinoa"],
                                },
                            }
                        ]
                    }
                }
            },
        }
        path = tmp_path / "unresolved.jsonld"
        path.write_text(json.dumps(contract))
        with pytest.raises(KeyError, match="not found"):
            compile_contract(path, agrovoc=agrovoc_adapter)

    def test_missing_agrovoc_adapter(self, tmp_path: Path) -> None:
        contract = {
            "@context": "https://ckir.dev/context/v1",
            "@type": "ExtractionContract",
            "provider": "test",
            "languages": ["en"],
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {
                                "name": "Crop",
                                "type": "string",
                                "grounding": {
                                    "ontology": "agrovoc",
                                    "concepts": ["wheat"],
                                },
                            }
                        ]
                    }
                }
            },
        }
        path = tmp_path / "no_adapter.jsonld"
        path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="AgrovocAdapter"):
            compile_contract(path)

    def test_missing_geonames_adapter(self, tmp_path: Path) -> None:
        contract = {
            "@context": "https://ckir.dev/context/v1",
            "@type": "ExtractionContract",
            "provider": "test",
            "languages": ["en"],
            "outputs": {
                "data": {
                    "schema": {
                        "columns": [
                            {
                                "name": "Region",
                                "type": "string",
                                "grounding": {
                                    "ontology": "geonames",
                                    "country": "RU",
                                    "level": "ADM1",
                                },
                            }
                        ]
                    }
                }
            },
        }
        path = tmp_path / "no_geo_adapter.jsonld"
        path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="GeoNamesAdapter"):
            compile_contract(path)


# ── Acceptance Criteria Tests ────────────────────────────────────────────


class TestAcceptanceCriteria:
    def test_ac1_sme_authors_without_ontology_reference(
        self,
        minimal_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        """AC1: Contract with only labels (no URIs) compiles successfully."""
        result = compile_contract(
            minimal_jsonld, agrovoc=agrovoc_adapter
        )
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][0]
        # Aliases were generated from labels, not URIs
        assert "wheat" in crop_col["aliases"]
        assert "barley" in crop_col["aliases"]
        # No grounding or concept_uris in output
        assert "grounding" not in crop_col
        assert "concept_uris" not in crop_col

    def test_ac2_compiled_output_matches_hand_authored(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """AC2: Compiled output produces equivalent alias structure."""
        result = compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )

        # Region column has all 3 fixture regions
        region_col = result["outputs"]["harvest"]["schema"]["columns"][0]
        region_aliases = set(region_col["aliases"])
        assert "Moscow Oblast" in region_aliases
        assert "Voronezh Oblast" in region_aliases
        assert "Penza Oblast" in region_aliases

        # Crop column has base + variant aliases
        crop_col = result["outputs"]["harvest"]["schema"]["columns"][1]
        crop_aliases = set(crop_col["aliases"])
        assert "wheat" in crop_aliases
        assert "spring wheat" in crop_aliases

        # Provenance is tracked
        assert "_alias_provenance" in crop_col

    def test_ac3_pipeline_output_identical_structure(
        self,
        mixed_jsonld: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """AC3: Compiled contract has the same top-level structure as hand-authored."""
        result = compile_contract(
            mixed_jsonld,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )

        # Standard contract fields present
        assert "provider" in result
        assert "outputs" in result

        # JSON-LD fields stripped
        assert "@context" not in result
        assert "@type" not in result

        # Each column has standard fields, no grounding
        for col in result["outputs"]["harvest"]["schema"]["columns"]:
            assert "name" in col
            assert "type" in col
            assert "grounding" not in col
