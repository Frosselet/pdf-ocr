"""Tests for GeoNames adapter."""

from __future__ import annotations

from contract_semantics.geonames import GeoNamesAdapter


class TestResolveGeoname:
    def test_english_name(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.resolve_geoname(
            "https://sws.geonames.org/524894/",
            languages=["en"],
        )
        aliases = {r.alias for r in results}
        assert "Moscow Oblast" in aliases

    def test_russian_alternate_name(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.resolve_geoname(
            "https://sws.geonames.org/524894/",
            languages=["ru"],
        )
        aliases = {r.alias for r in results}
        assert "Московская область" in aliases

    def test_multilingual(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.resolve_geoname(
            524894,  # integer ID
            languages=["en", "ru"],
        )
        aliases = {r.alias for r in results}
        assert "Moscow Oblast" in aliases
        assert "Московская область" in aliases

    def test_inline_alternate_names(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.resolve_geoname(
            524894,
            languages=["en"],
        )
        aliases = {r.alias for r in results}
        assert "Moskovskaya Oblast" in aliases

    def test_concept_uri_format(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.resolve_geoname(
            "https://sws.geonames.org/472039/",
            languages=["en"],
        )
        assert all(r.concept_uri == "https://sws.geonames.org/472039/" for r in results)


class TestEnrichGeoname:
    def test_basic_enrichment(self, geonames_adapter: GeoNamesAdapter) -> None:
        e = geonames_adapter.enrich_geoname("https://sws.geonames.org/524894/")
        assert e.geoname_id == 524894
        assert e.name == "Moscow Oblast"
        assert e.lat == 55.75
        assert e.lng == 37.62
        assert e.country_code == "RU"
        assert e.admin1_code == "48"
        assert e.feature_class == "A"
        assert e.feature_code == "ADM1"
        assert e.population == 7500000

    def test_enrichment_by_id(self, geonames_adapter: GeoNamesAdapter) -> None:
        e = geonames_adapter.enrich_geoname(472039)
        assert e.name == "Voronezh Oblast"
        assert e.admin1_code == "86"

    def test_enrichment_missing_id(self, geonames_adapter: GeoNamesAdapter) -> None:
        import pytest
        with pytest.raises(KeyError):
            geonames_adapter.enrich_geoname(999999)


class TestSearchGeonames:
    def test_search_by_name(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.search_geonames("Moscow")
        assert len(results) >= 1
        assert any(r.name == "Moscow Oblast" for r in results)

    def test_search_with_country_filter(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.search_geonames("Oblast", country="RU")
        assert len(results) == 3  # All three fixtures are RU oblasts

    def test_search_with_feature_code(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.search_geonames(
            "Penza", feature_class="A", feature_code="ADM1"
        )
        assert len(results) == 1
        assert results[0].name == "Penza Oblast"

    def test_search_no_results(self, geonames_adapter: GeoNamesAdapter) -> None:
        results = geonames_adapter.search_geonames("Antarctica")
        assert results == []
