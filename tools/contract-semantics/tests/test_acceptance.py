"""Acceptance criteria tests for Phase 1: Semantic Contracts.

Each test maps to one of the four roadmap acceptance criteria:
  AC1: Add region -> bilingual aliases automatically
  AC2: SHACL validation flags out-of-concept values
  AC3: Diff coverage >= 90%
  AC4: Every alias traces to a concept URI
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.context import build_context_data
from contract_semantics.diff import diff_aliases
from contract_semantics.geonames import GeoNamesAdapter
from contract_semantics.materialize import materialize_contract
from contract_semantics.models import ConceptRef, ResolveConfig
from contract_semantics.resolve import resolve_column
from contract_semantics.validate import generate_shapes, validate_records


@pytest.fixture
def annotated_contract(tmp_path: Path) -> Path:
    """Annotated contract with AGROVOC crops and GeoNames regions."""
    contract = {
        "provider": "test_ac",
        "outputs": {
            "harvest": {
                "category": "harvest",
                "schema": {
                    "description": "AC test schema",
                    "columns": [
                        {
                            "name": "Crop",
                            "type": "string",
                            "aliases": ["wheat", "barley", "common wheat"],
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
                            "aliases": ["Moscow Oblast"],
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
                            "name": "Value",
                            "type": "float",
                            "aliases": ["2025"],
                        },
                    ],
                },
            },
        },
    }
    path = tmp_path / "ac_contract.json"
    path.write_text(json.dumps(contract, indent=2))
    return path


class TestAC1BilingualAliases:
    """AC1: Add region -> GeoNames produces bilingual aliases automatically."""

    def test_bilingual_region_aliases(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        data = build_context_data(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )
        region_aliases = data["resolved_aliases"]["harvest"]["Region"]

        # English names present
        assert "Moscow Oblast" in region_aliases
        assert "Voronezh Oblast" in region_aliases

        # Russian names present (from GeoNames alternate names fixture)
        assert "Московская область" in region_aliases
        assert "Воронежская область" in region_aliases

        # No manual Russian translation was required - the adapter resolved them
        # We verify this by checking the aliases come from GeoNames, not manual input
        # The contract only has "Moscow Oblast" as a manual alias

    def test_no_manual_translation_needed(
        self,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Resolving a single GeoNames URI produces Russian name without manual input."""
        aliases = geonames_adapter.resolve_geoname(
            "https://sws.geonames.org/524894/",
            languages=["en", "ru"],
        )
        russian_names = [a.alias for a in aliases if a.language == "ru"]
        assert len(russian_names) > 0
        assert "Московская область" in russian_names


class TestAC2SHACLValidation:
    """AC2: SHACL validation flags values outside known concepts."""

    def test_valid_crops_pass(
        self,
        annotated_contract: Path,
        tmp_path: Path,
    ) -> None:
        # Generate SHACL shapes from the contract
        shapes = generate_shapes(annotated_contract, output_dir=tmp_path / "shapes")
        assert "harvest" in shapes

        # Valid records should pass
        records = [
            {"Crop": "wheat", "Region": "Moscow Oblast", "Value": "100.0"},
            {"Crop": "barley", "Region": "Moscow Oblast", "Value": "50.0"},
        ]
        conforms, report_text, _ = validate_records(
            records, shapes["harvest"],
            schema_columns=[
                {"name": "Crop", "type": "string"},
                {"name": "Region", "type": "string"},
                {"name": "Value", "type": "float"},
            ],
        )
        assert conforms, f"Valid records should conform: {report_text}"

    def test_invalid_crop_flagged(
        self,
        annotated_contract: Path,
        tmp_path: Path,
    ) -> None:
        shapes = generate_shapes(annotated_contract, output_dir=tmp_path / "shapes")

        # "INVALID_CROP" is not in the contract's crop aliases
        records = [
            {"Crop": "INVALID_CROP", "Region": "Moscow Oblast", "Value": "100.0"},
        ]
        conforms, report_text, _ = validate_records(
            records, shapes["harvest"],
            schema_columns=[
                {"name": "Crop", "type": "string"},
                {"name": "Region", "type": "string"},
                {"name": "Value", "type": "float"},
            ],
        )
        assert not conforms, "Record with unknown crop should not conform"
        assert "INVALID_CROP" in report_text


class TestAC3DiffCoverage:
    """AC3: Diff coverage >= 90% for well-annotated columns."""

    def test_crop_coverage_high(
        self,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        """Well-annotated crop column achieves >= 90% coverage."""
        concept_refs = [
            ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_8373", label="wheat"),
            ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_631", label="barley"),
        ]
        # Manual aliases that overlap with ontology labels
        manual_aliases = ["wheat", "barley", "common wheat"]
        resolve_config = ResolveConfig(
            languages=["en"],
            label_types=["prefLabel", "altLabel"],
        )
        result = resolve_column(
            "Crop", concept_refs, manual_aliases, resolve_config, agrovoc_adapter,
        )
        report = diff_aliases(result)

        # All 3 manual aliases should match ontology (wheat=prefLabel,
        # barley=prefLabel, common wheat=altLabel)
        assert result.coverage >= 0.90, (
            f"Coverage {result.coverage:.0%} < 90%. Report:\n{report}"
        )

    def test_diff_report_includes_coverage_line(
        self,
        agrovoc_adapter: AgrovocAdapter,
    ) -> None:
        """Diff report contains a readable coverage line."""
        concept_refs = [
            ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_8373", label="wheat"),
        ]
        manual_aliases = ["wheat"]
        resolve_config = ResolveConfig(languages=["en"], label_types=["prefLabel"])
        result = resolve_column(
            "Crop", concept_refs, manual_aliases, resolve_config, agrovoc_adapter,
        )
        report = diff_aliases(result)
        assert "Coverage:" in report
        assert "100%" in report


class TestAC4AliasProvenance:
    """AC4: Every alias traces to a concept URI."""

    def test_every_resolved_alias_has_provenance(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Every alias in the materialized output has provenance metadata."""
        result = materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )

        for _out_name, out_spec in result["outputs"].items():
            for col in out_spec["schema"]["columns"]:
                if "_alias_provenance" not in col:
                    # Unannotated columns have no provenance — that's fine
                    continue
                prov = col["_alias_provenance"]
                # Every alias in the list must appear in provenance
                for alias in col["aliases"]:
                    assert alias in prov, (
                        f"Alias '{alias}' in column '{col['name']}' "
                        f"has no provenance entry"
                    )

    def test_resolved_aliases_trace_to_concept_uri(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
    ) -> None:
        """Non-manual aliases have concept_uri in their provenance."""
        result = materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
        )

        for _out_name, out_spec in result["outputs"].items():
            for col in out_spec["schema"]["columns"]:
                if "_alias_provenance" not in col:
                    continue
                for alias, meta in col["_alias_provenance"].items():
                    if meta["source"] in ("resolved", "both"):
                        assert "concept_uri" in meta, (
                            f"Resolved alias '{alias}' has no concept_uri"
                        )
                        assert meta["concept_uri"].startswith("http"), (
                            f"concept_uri for '{alias}' is not a valid URI: "
                            f"{meta['concept_uri']}"
                        )

    def test_provenance_serializes_to_json(
        self,
        annotated_contract: Path,
        agrovoc_adapter: AgrovocAdapter,
        geonames_adapter: GeoNamesAdapter,
        tmp_path: Path,
    ) -> None:
        """Provenance data roundtrips through JSON output file."""
        out_path = tmp_path / "materialized.json"
        materialize_contract(
            annotated_contract,
            agrovoc=agrovoc_adapter,
            geonames=geonames_adapter,
            output_path=out_path,
        )
        loaded = json.loads(out_path.read_text())
        crop_col = loaded["outputs"]["harvest"]["schema"]["columns"][0]
        prov = crop_col["_alias_provenance"]

        # Verify roundtrip preserved structure
        assert isinstance(prov, dict)
        assert len(prov) == len(crop_col["aliases"])
        for alias in crop_col["aliases"]:
            assert alias in prov
