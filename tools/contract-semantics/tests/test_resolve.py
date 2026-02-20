"""Tests for generic resolution (resolve_column with prefix patterns)."""

from __future__ import annotations

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.geonames import GeoNamesAdapter
from contract_semantics.models import ConceptRef, ResolveConfig
from contract_semantics.resolve import resolve_column


class TestResolveColumnAgrovoc:
    def test_basic_resolution(self, agrovoc_adapter: AgrovocAdapter) -> None:
        result = resolve_column(
            column_name="Crop",
            concept_refs=[
                ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_8373", label="wheat"),
                ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_631", label="barley"),
            ],
            manual_aliases=["wheat", "barley", "spring wheat"],
            resolve_config=ResolveConfig(
                languages=["en"],
                label_types=["prefLabel"],
            ),
            adapter=agrovoc_adapter,
        )
        assert result.column_name == "Crop"
        assert "wheat" in result.matched
        assert "barley" in result.matched
        assert "spring wheat" in result.manual_only

    def test_prefix_patterns(self, agrovoc_adapter: AgrovocAdapter) -> None:
        result = resolve_column(
            column_name="Crop",
            concept_refs=[
                ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_8373", label="wheat"),
            ],
            manual_aliases=["spring wheat", "wheat"],
            resolve_config=ResolveConfig(
                languages=["en"],
                label_types=["prefLabel"],
                prefix_patterns=["spring {label}", "{label}"],
            ),
            adapter=agrovoc_adapter,
        )
        assert "spring wheat" in result.matched
        assert "wheat" in result.matched
        assert result.coverage == 1.0

    def test_russian_labels_as_resolved_only(self, agrovoc_adapter: AgrovocAdapter) -> None:
        result = resolve_column(
            column_name="Crop",
            concept_refs=[
                ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_8373", label="wheat"),
            ],
            manual_aliases=["wheat"],
            resolve_config=ResolveConfig(
                languages=["en", "ru"],
                label_types=["prefLabel"],
            ),
            adapter=agrovoc_adapter,
        )
        assert "wheat" in result.matched
        assert "пшеница" in result.resolved_only

    def test_coverage_calculation(self, agrovoc_adapter: AgrovocAdapter) -> None:
        result = resolve_column(
            column_name="Crop",
            concept_refs=[
                ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_8373", label="wheat"),
            ],
            manual_aliases=["wheat", "spring crops", "grains and grasses"],
            resolve_config=ResolveConfig(languages=["en"], label_types=["prefLabel"]),
            adapter=agrovoc_adapter,
        )
        # Only "wheat" matches, "spring crops" and "grains and grasses" are manual-only
        assert result.coverage == pytest.approx(1 / 3)


class TestResolveColumnGeonames:
    def test_geonames_resolution(self, geonames_adapter: GeoNamesAdapter) -> None:
        result = resolve_column(
            column_name="Region",
            concept_refs=[
                ConceptRef(uri="https://sws.geonames.org/524894/", label="Moscow Oblast"),
            ],
            manual_aliases=["Region", "Moscow Oblast"],
            resolve_config=ResolveConfig(
                source="geonames",
                languages=["en"],
            ),
            adapter=geonames_adapter,
        )
        assert "Moscow Oblast" in result.matched
        assert "Region" in result.manual_only
        # Inline alternate "Moskovskaya Oblast" should be resolved-only
        assert any("Moskovskaya" in a for a in result.resolved_only)


class TestDeduplication:
    def test_duplicate_aliases_deduplicated(self, agrovoc_adapter: AgrovocAdapter) -> None:
        result = resolve_column(
            column_name="Crop",
            concept_refs=[
                ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_7440", label="sunflowers"),
            ],
            manual_aliases=["sunflower"],
            resolve_config=ResolveConfig(
                languages=["en"],
                label_types=["prefLabel", "altLabel"],
            ),
            adapter=agrovoc_adapter,
        )
        # "sunflower" appears as both altLabel and manual — should be matched once
        aliases = [ra.alias.lower() for ra in result.resolved_aliases]
        # No duplicates in resolved list
        assert len(aliases) == len(set(aliases))


import pytest  # noqa: E402 — needed for approx
