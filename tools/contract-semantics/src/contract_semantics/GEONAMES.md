# GeoNames Adapter

> **Module**: `geonames.py`
> **Public API**: `GeoNamesAdapter`

Resolves GeoNames feature URIs to multilingual alternate names and geographic metadata. Goes beyond alias generation --- enables downstream GIS enrichment (coordinates, admin hierarchy, ISO codes) that makes pipeline outputs directly usable for spatial analytics.

---

## The Problem

Geographic region columns appear in documents with varying names: "Moscow Oblast", "Moskovskaya Oblast", "Московская область". Manually maintaining all variants is tedious. GeoNames already has canonical names and alternates in 100+ languages. Additionally, grounding regions in GeoNames IDs enables post-extraction coordinate enrichment without changing the extraction pipeline.

## The Solution

Three capabilities, two operating modes:

| Capability | Method | Returns |
|---|---|---|
| **Resolve** | `resolve_geoname()` | Multilingual alternate names as `ResolvedAlias` |
| **Enrich** | `enrich_geoname()` | Coordinates, admin codes, population as `GeoEnrichment` |
| **Search** | `search_geonames()` | Feature matches by name with filters as `GeoSearchResult` |
| **Enumerate** | `enumerate_features()` | All features for a country + admin level as `GeoSearchResult` |

| Mode | Data Source | Best For |
|---|---|---|
| **Offline** | Country extract TSV (e.g. `RU.txt` ~5MB) | Batch materialization |
| **Online** | `http://api.geonames.org/` REST API | Interactive annotation |

---

## API

### Construction

```python
from contract_semantics.geonames import GeoNamesAdapter

# Offline: parse country extract TSV
adapter = GeoNamesAdapter.from_file("data/RU.txt")

# Offline with language-tagged alternate names
adapter = GeoNamesAdapter.from_file("data/RU.txt", alt_names_path="data/alternateNamesV2.txt")

# Online: REST API (requires free username from geonames.org)
adapter = GeoNamesAdapter.online(username="myuser")
```

### resolve_geoname()

Return alternate names for a GeoNames feature. Also available as `resolve_concept()` (conforming to the `OntologyAdapter` protocol).

```python
results = adapter.resolve_geoname(
    "https://sws.geonames.org/524894/",  # Moscow Oblast
    languages=["en", "ru"],
)
# Returns: [
#   ResolvedAlias(alias="Moscow Oblast", language="en", label_type="name"),
#   ResolvedAlias(alias="Moskovskaya Oblast", language="en", label_type="altLabel"),
#   ResolvedAlias(alias="Московская область", language="ru", label_type="altLabel"),
# ]
```

Accepts both URI strings and integer IDs:

```python
results = adapter.resolve_geoname(524894, languages=["en"])
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `uri_or_id` | `str \| int` | required | GeoNames URI or integer ID |
| `languages` | `list[str]` | `["en"]` | Language codes to retrieve |

### enrich_geoname()

Return geographic metadata for a feature.

```python
e = adapter.enrich_geoname("https://sws.geonames.org/524894/")
# GeoEnrichment(
#     geoname_id=524894, name="Moscow Oblast",
#     lat=55.75, lng=37.62,
#     country_code="RU", admin1_code="48",
#     feature_class="A", feature_code="ADM1",
#     population=7500000
# )
```

### enumerate_features()

Return all features matching a country + admin level. Used by the compiler for bulk GeoNames grounding.

```python
results = adapter.enumerate_features("RU", "ADM1")
# Returns all ADM1 features in Russia
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `country` | `str` | required | ISO 3166-1 alpha-2 country code |
| `level` | `str` | required | Admin level key from `LEVEL_MAP` |

Raises `KeyError` if level is not in `LEVEL_MAP`.

#### LEVEL_MAP

Maps admin level names to GeoNames `(feature_class, feature_code)` tuples:

| Level | Feature Class | Feature Code |
|---|---|---|
| `ADM1` | `A` | `ADM1` |
| `ADM2` | `A` | `ADM2` |
| `ADM3` | `A` | `ADM3` |
| `PPL` | `P` | `PPL` |
| `PPLA` | `P` | `PPLA` |

### search_geonames()

Search features by name with optional filters.

```python
results = adapter.search_geonames(
    "Voronezh",
    country="RU",
    feature_class="A",
    feature_code="ADM1",
    max_results=5,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Name to search for (substring match offline, full-text online) |
| `country` | `str \| None` | `None` | ISO country code filter |
| `feature_class` | `str \| None` | `None` | GeoNames feature class (`"A"` = admin, `"P"` = populated place) |
| `feature_code` | `str \| None` | `None` | GeoNames feature code (`"ADM1"`, `"PPL"`, etc.) |
| `max_results` | `int` | `10` | Maximum results to return |

---

## Offline Data Format

GeoNames distributes data as tab-separated files. The adapter parses the standard 19-column format used in country extracts (`RU.txt`, `US.txt`, etc.) and `allCountries.txt`:

| Column | Index | Used For |
|---|---|---|
| geonameid | 0 | Primary key |
| name | 1 | Feature name (UTF-8) |
| asciiname | 2 | ASCII transliteration |
| alternatenames | 3 | Comma-separated alternate names |
| latitude | 4 | `GeoEnrichment.lat` |
| longitude | 5 | `GeoEnrichment.lng` |
| feature class | 6 | `"A"`, `"P"`, `"H"`, etc. |
| feature code | 7 | `"ADM1"`, `"PPL"`, etc. |
| country code | 8 | ISO-3166 two-letter code |
| admin1 code | 10 | First-level admin division |
| population | 14 | Population count |

An optional alternate names file provides language-tagged names (columns: alternateNameId, geonameId, isolanguage, alternateName).

---

## Geo Sidecar Files

When `enrich_fields` is configured in the contract's `resolve` block, the materializer produces a geo sidecar JSON alongside the materialized contract:

```json
{
  "_provenance": {"source": "GeoNames", "resolved_at": "2026-02-20"},
  "regions": {
    "Moscow Oblast": {"geoname_id": 524894, "lat": 55.75, "lng": 37.62, "admin1Code": "48", "countryCode": "RU"},
    "Voronezh Oblast": {"geoname_id": 472039, "lat": 51.67, "lng": 39.21, "admin1Code": "86", "countryCode": "RU"}
  }
}
```

This enables a simple post-pipeline GIS join: `df.merge(geo_df, left_on="Region", right_index=True)` to add coordinates to every row.

---

## Test Coverage

15 tests in `tests/test_geonames.py` using a fixture dataset (`conftest.py`) containing three Russian admin-1 regions (Moscow Oblast, Voronezh Oblast, Penza Oblast) with Russian alternate names. Includes 3 tests for `enumerate_features()` (ADM1, no match, level map coverage).
