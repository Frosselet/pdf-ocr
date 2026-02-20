"""Shared fixtures: small AGROVOC RDF graph and GeoNames mock data."""

from __future__ import annotations

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import SKOS

from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.geonames import GeoNamesAdapter

AGROVOC = Namespace("http://aims.fao.org/aos/agrovoc/")


def _build_agrovoc_fixture() -> Graph:
    """Build a tiny AGROVOC-like graph for testing.

    Contains:
    - c_8373 (wheat) with English + Russian labels
    - c_631 (barley) with English + Russian labels
    - c_12332 (maize) with English + Russian labels
    - c_1094 (buckwheat) with English label
    - c_7440 (sunflowers) with English + Russian labels
    - c_8373 has a narrower concept c_33556 (durum wheat)
    """
    g = Graph()

    # Wheat
    wheat = AGROVOC["c_8373"]
    g.add((wheat, SKOS.prefLabel, Literal("wheat", lang="en")))
    g.add((wheat, SKOS.prefLabel, Literal("пшеница", lang="ru")))
    g.add((wheat, SKOS.altLabel, Literal("common wheat", lang="en")))

    # Durum wheat (narrower of wheat)
    durum = AGROVOC["c_33556"]
    g.add((wheat, SKOS.narrower, durum))
    g.add((durum, SKOS.prefLabel, Literal("durum wheat", lang="en")))
    g.add((durum, SKOS.prefLabel, Literal("твёрдая пшеница", lang="ru")))

    # Barley
    barley = AGROVOC["c_631"]
    g.add((barley, SKOS.prefLabel, Literal("barley", lang="en")))
    g.add((barley, SKOS.prefLabel, Literal("ячмень", lang="ru")))
    g.add((barley, SKOS.altLabel, Literal("barleycorn", lang="en")))

    # Maize
    maize = AGROVOC["c_12332"]
    g.add((maize, SKOS.prefLabel, Literal("maize", lang="en")))
    g.add((maize, SKOS.prefLabel, Literal("кукуруза", lang="ru")))
    g.add((maize, SKOS.altLabel, Literal("corn", lang="en")))

    # Buckwheat
    buckwheat = AGROVOC["c_1094"]
    g.add((buckwheat, SKOS.prefLabel, Literal("buckwheat", lang="en")))
    g.add((buckwheat, SKOS.prefLabel, Literal("гречиха", lang="ru")))

    # Sunflowers
    sunflowers = AGROVOC["c_7440"]
    g.add((sunflowers, SKOS.prefLabel, Literal("sunflowers", lang="en")))
    g.add((sunflowers, SKOS.prefLabel, Literal("подсолнечник", lang="ru")))
    g.add((sunflowers, SKOS.altLabel, Literal("sunflower", lang="en")))

    return g


def _build_geonames_fixture() -> tuple[dict[int, dict], dict[int, list[tuple[str, str]]]]:
    """Build a small GeoNames mock dataset for testing.

    Contains three Russian admin-1 regions.
    """
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
        524894: [
            ("ru", "Московская область"),
            ("en", "Moscow Region"),
        ],
        472039: [
            ("ru", "Воронежская область"),
        ],
        511180: [
            ("ru", "Пензенская область"),
        ],
    }

    return features, alt_names


@pytest.fixture
def agrovoc_graph() -> Graph:
    """Small AGROVOC RDF fixture graph."""
    return _build_agrovoc_fixture()


@pytest.fixture
def agrovoc_adapter(agrovoc_graph: Graph) -> AgrovocAdapter:
    """AgrovocAdapter backed by the fixture graph."""
    return AgrovocAdapter(graph=agrovoc_graph)


@pytest.fixture
def geonames_adapter() -> GeoNamesAdapter:
    """GeoNamesAdapter backed by the fixture data."""
    features, alt_names = _build_geonames_fixture()
    return GeoNamesAdapter(features=features, alternate_names=alt_names)
