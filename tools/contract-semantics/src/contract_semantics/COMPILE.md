# Compiler: JSON-LD → Materialized Contract

> **Module**: `compile.py`
> **Public API**: `compile_contract()`

Transforms a JSON-LD contract (human-readable, SME-authored) into a materialized contract (current JSON format, pipeline-ready). Two-step process: resolve grounding blocks to `concept_uris` + `resolve`, then delegate to `materialize._materialize_dict()` for alias expansion.

---

## The Problem

Authoring contracts requires knowing AGROVOC identifiers (`c_8373` for wheat), GeoNames URIs (`sws.geonames.org/524894/`), and `resolve` configuration syntax. SMEs know crops and regions --- they should write labels, not URIs.

## The Solution

```
JSON-LD contract (grounding blocks)
    | compile_contract()
    |   Step 1: grounding → concept_uris + resolve
    |   Step 2: concept_uris → expanded aliases  (reuses materialize)
    v
Materialized contract (current JSON format)
```

---

## API

### compile_contract()

```python
from contract_semantics.compile import compile_contract

result = compile_contract(
    "contracts/ru_ag_ministry.jsonld",
    agrovoc=agrovoc_adapter,
    geonames=geonames_adapter,
    merge_strategy="union",
    output_path="contracts/compiled.json",
    geo_sidecar_path="contracts/geo.json",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `jsonld_path` | `str \| Path` | required | Path to the JSON-LD contract file |
| `agrovoc` | `AgrovocAdapter \| None` | `None` | AGROVOC adapter for concept resolution |
| `geonames` | `GeoNamesAdapter \| None` | `None` | GeoNames adapter for feature enumeration |
| `merge_strategy` | `str` | `"union"` | Alias merge strategy |
| `output_path` | `str \| Path \| None` | `None` | Write materialized contract here |
| `geo_sidecar_path` | `str \| Path \| None` | `None` | Write geo sidecar here |

Returns the materialized contract dict.

---

## Grounding Blocks

### AGROVOC Grounding

```json
{
  "grounding": {
    "ontology": "agrovoc",
    "concepts": ["wheat", "barley"],
    "variants": ["spring", "winter"],
    "labelTypes": ["prefLabel", "altLabel"],
    "includeNarrower": false
  }
}
```

The compiler:
1. Calls `agrovoc.lookup_by_label(label)` for each concept.
2. Builds `concept_uris` with resolved URIs.
3. Generates `resolve` config with `prefix_patterns` from `variants` (e.g., `["spring {label}", "winter {label}", "{label}"]`).
4. Uses contract-level `languages` for `resolve.languages`.

### GeoNames Grounding

```json
{
  "grounding": {
    "ontology": "geonames",
    "country": "RU",
    "level": "ADM1",
    "enrich": ["lat", "lng", "admin1Code", "countryCode"]
  }
}
```

The compiler:
1. Calls `geonames.enumerate_features(country, level)` to get all matching features.
2. Builds `concept_uris` from results.
3. Generates `resolve` config with `feature_class`, `feature_code`, `country` from `LEVEL_MAP`.
4. Uses contract-level `languages` for `resolve.languages`.

---

## JSON-LD Validation

| Check | Error |
|---|---|
| Missing `@context` | `ValueError` with instructions |
| Wrong `@type` | `ValueError` (must be `"ExtractionContract"`) |
| Missing `ontology` in grounding | `ValueError` |
| Unknown ontology | `ValueError` listing supported ontologies |
| AGROVOC grounding without `concepts` | `ValueError` |
| GeoNames grounding without `country` or `level` | `ValueError` |

---

## Field Mapping

| JSON-LD field | Materialized output |
|---|---|
| `@context` | Stripped |
| `@type` | Stripped |
| `domain` | Stripped (metadata only) |
| `sourceFormat` | Stripped (metadata only) |
| `languages` | Consumed by compiler, not in output |
| `grounding` | Replaced by `concept_uris` + `resolve` |
| `_compiled_from` | Added (source filename) |
| All other fields | Passed through unchanged |

---

## Test Coverage

28 tests in `tests/test_compile.py`: adapter extensions (5), compilation pipeline (11), validation/error (6), acceptance criteria (3), plus 3 adapter tests in `test_agrovoc.py` and `test_geonames.py`.
