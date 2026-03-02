"""Tests for AGROVOC SKOS adapter."""

from __future__ import annotations

from contract_semantics.agrovoc import AgrovocAdapter


class TestResolveConceptOffline:
    def test_english_preflabels(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_8373",
            languages=["en"],
            label_types=["prefLabel"],
        )
        aliases = {r.alias for r in results}
        assert "wheat" in aliases
        assert all(r.language == "en" for r in results)
        assert all(r.label_type == "prefLabel" for r in results)

    def test_russian_labels(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_8373",
            languages=["ru"],
            label_types=["prefLabel"],
        )
        aliases = {r.alias for r in results}
        assert "пшеница" in aliases

    def test_multilingual(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_8373",
            languages=["en", "ru"],
            label_types=["prefLabel", "altLabel"],
        )
        aliases = {r.alias for r in results}
        assert "wheat" in aliases
        assert "пшеница" in aliases
        assert "common wheat" in aliases

    def test_altlabels(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_12332",
            languages=["en"],
            label_types=["altLabel"],
        )
        aliases = {r.alias for r in results}
        assert "corn" in aliases
        # prefLabel "maize" should NOT be in results
        assert "maize" not in aliases

    def test_narrower_traversal(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_8373",
            languages=["en"],
            label_types=["prefLabel"],
            include_narrower=True,
            narrower_depth=1,
        )
        aliases = {r.alias for r in results}
        assert "wheat" in aliases
        assert "durum wheat" in aliases

    def test_narrower_russian(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_8373",
            languages=["ru"],
            label_types=["prefLabel"],
            include_narrower=True,
        )
        aliases = {r.alias for r in results}
        assert "пшеница" in aliases
        assert "твёрдая пшеница" in aliases

    def test_no_results_for_nonexistent_concept(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_999999",
            languages=["en"],
        )
        assert results == []

    def test_concept_label_field(self, agrovoc_adapter: AgrovocAdapter) -> None:
        results = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_8373",
            languages=["en"],
            label_types=["prefLabel"],
        )
        assert all(r.concept_label == "wheat" for r in results)

    def test_multiple_concepts(self, agrovoc_adapter: AgrovocAdapter) -> None:
        """Resolve multiple concepts independently."""
        wheat = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_8373",
            languages=["en"],
            label_types=["prefLabel"],
        )
        barley = agrovoc_adapter.resolve_concept(
            "http://aims.fao.org/aos/agrovoc/c_631",
            languages=["en"],
            label_types=["prefLabel"],
        )
        assert {r.alias for r in wheat} == {"wheat"}
        assert {r.alias for r in barley} == {"barley"}


class TestLookupByLabel:
    def test_lookup_by_label_exact(self, agrovoc_adapter: AgrovocAdapter) -> None:
        uri = agrovoc_adapter.lookup_by_label("wheat", language="en")
        assert uri == "http://aims.fao.org/aos/agrovoc/c_8373"

    def test_lookup_by_label_case_insensitive(self, agrovoc_adapter: AgrovocAdapter) -> None:
        uri = agrovoc_adapter.lookup_by_label("Wheat", language="en")
        assert uri == "http://aims.fao.org/aos/agrovoc/c_8373"

    def test_lookup_by_label_not_found(self, agrovoc_adapter: AgrovocAdapter) -> None:
        import pytest
        with pytest.raises(KeyError, match="not found"):
            agrovoc_adapter.lookup_by_label("quinoa", language="en")

    def test_lookup_by_label_russian(self, agrovoc_adapter: AgrovocAdapter) -> None:
        uri = agrovoc_adapter.lookup_by_label("пшеница", language="ru")
        assert uri == "http://aims.fao.org/aos/agrovoc/c_8373"
