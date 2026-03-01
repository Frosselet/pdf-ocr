# Materialization

> **Module**: `materialize.py`
> **Public API**: `materialize_contract()`

Reads an annotated contract (with `concept_uris` and `resolve` fields), resolves all concept URIs through the appropriate ontology adapters, merges aliases, and produces two outputs: a materialized contract JSON and an optional geo sidecar file.

---

## The Problem

Annotated contracts contain semantic metadata (`concept_uris`, `resolve` config) that docpact doesn't understand. Before docpact can consume the contract, these annotations need to be resolved into concrete alias lists and stripped from the output.

## The Solution

`materialize_contract()` is the one-step transformation:

```
Annotated contract (concept_uris + resolve)
  -> resolve URIs via adapters
  -> merge with manual aliases
  -> strip annotation fields
  -> write standard contract JSON
  -> optionally write geo sidecar
```

---

## API

### materialize_contract()

```python
from contract_semantics.materialize import materialize_contract
from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.geonames import GeoNamesAdapter

agrovoc = AgrovocAdapter.from_file("data/agrovoc_core.nt")
geonames = GeoNamesAdapter.from_file("data/RU.txt")

result = materialize_contract(
    "contracts/ru_ag_ministry.json",
    agrovoc=agrovoc,
    geonames=geonames,
    merge_strategy="union",
    output_path="contracts/ru_ag_ministry_materialized.json",
    geo_sidecar_path="contracts/ru_ag_ministry_geo.json",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `annotated_path` | `str \| Path` | required | Path to annotated contract JSON |
| `agrovoc` | `AgrovocAdapter \| None` | `None` | AGROVOC adapter (if contract uses AGROVOC concepts) |
| `geonames` | `GeoNamesAdapter \| None` | `None` | GeoNames adapter (if contract uses GeoNames concepts) |
| `merge_strategy` | `str` | `"union"` | How to merge resolved and manual aliases |
| `output_path` | `str \| Path \| None` | `None` | Write materialized contract to this path |
| `geo_sidecar_path` | `str \| Path \| None` | `None` | Write geo sidecar to this path |

**Returns**: The materialized contract as a dict.

---

## Merge Strategies

| Strategy | Behavior |
|---|---|
| `union` | Keep all manual aliases, add resolved ones not already present (case-insensitive dedup) |
| `resolved_only` | Replace manual aliases entirely with resolved ones |
| `manual_priority` | Same as `union` --- manual aliases first, then new resolved ones |

---

## Materialization Steps

For each column with `concept_uris`:

1. Parse `concept_uris` into `ConceptRef` list
2. Parse `resolve` into `ResolveConfig` (defaults to AGROVOC if omitted)
3. Pick adapter based on `resolve.source`
4. Call `resolve_column()` to get `ResolutionResult`
5. Merge aliases according to `merge_strategy`
6. If source is `geonames` and `enrich_fields` configured, collect `GeoEnrichment` data for the sidecar
7. Remove `concept_uris` and `resolve` fields from the column spec

Columns without `concept_uris` pass through unchanged.

---

## Alias Provenance

Each annotated column in the materialized output includes an `_alias_provenance` field that traces every alias to its origin:

```json
{
  "name": "Crop",
  "aliases": ["wheat", "пшеница", "common wheat"],
  "_alias_provenance": {
    "wheat": {
      "source": "both",
      "concept_uri": "http://aims.fao.org/aos/agrovoc/c_8373",
      "language": "en",
      "label_type": "prefLabel"
    },
    "пшеница": {
      "source": "resolved",
      "concept_uri": "http://aims.fao.org/aos/agrovoc/c_8373",
      "language": "ru",
      "label_type": "prefLabel"
    },
    "common wheat": {
      "source": "resolved",
      "concept_uri": "http://aims.fao.org/aos/agrovoc/c_8373",
      "language": "en",
      "label_type": "altLabel"
    }
  }
}
```

| Source | Meaning |
|---|---|
| `manual` | Alias was hand-curated in the contract |
| `resolved` | Alias was discovered via ontology resolution |
| `both` | Alias appears in both manual and resolved sets |

The `_` prefix signals this is metadata, not consumed by docpact's `load_contract()` (which ignores unknown fields). Unannotated columns (no `concept_uris`) do not get `_alias_provenance`.

---

## Geo Sidecar Output

When a GeoNames-grounded column has `enrich_fields` configured, the materializer produces a sidecar JSON file mapping region names to geographic metadata:

```json
{
  "_provenance": {
    "source": "GeoNames",
    "resolved_at": "2026-02-20"
  },
  "regions": {
    "Moscow Oblast": {
      "geoname_id": 524894,
      "lat": 55.75,
      "lng": 37.62,
      "admin1Code": "48",
      "countryCode": "RU"
    }
  }
}
```

Available enrich fields: `lat`, `lng`, `admin1Code`, `countryCode`, `population`.

---

## Test Coverage

9 tests in `tests/test_materialize.py` covering alias enrichment, annotation field removal, unannotated column passthrough, file output, geo sidecar output, `resolved_only` merge strategy, alias provenance presence, provenance exclusion on unannotated columns, and provenance JSON serialization.
