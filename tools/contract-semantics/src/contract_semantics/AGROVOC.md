# AGROVOC SKOS Adapter

> **Module**: `agrovoc.py`
> **Public API**: `AgrovocAdapter`

Resolves AGROVOC concept URIs to multilingual labels using the SKOS vocabulary (Simple Knowledge Organization System). AGROVOC is FAO's multilingual thesaurus covering agriculture, forestry, fisheries, food, and related domains --- over 43,000 concepts in 40+ languages.

---

## The Problem

Contract columns like `Crop` need aliases in multiple languages to match document headers. Manually maintaining alias lists across languages is error-prone and doesn't scale. AGROVOC already has the canonical labels and synonyms --- we just need to query them.

## The Solution

Two operating modes, same interface:

| Mode | Data Source | Startup | Best For |
|---|---|---|---|
| **Offline** | Local N-Triples dump (~50MB) | Slow first load, fast after pickle cache | Batch materialization, CI |
| **Online** | `https://agrovoc.fao.org/sparql` | No setup | Interactive use, small queries |

---

## API

### Construction

```python
from contract_semantics.agrovoc import AgrovocAdapter

# Offline: parse local dump (caches to .pkl for fast reload)
adapter = AgrovocAdapter.from_file("data/agrovoc_core.nt")

# Online: SPARQL endpoint
adapter = AgrovocAdapter.online()

# Direct: pass pre-loaded rdflib Graph
adapter = AgrovocAdapter(graph=my_graph)
```

### resolve_concept()

Resolve a concept URI to multilingual labels.

```python
results = adapter.resolve_concept(
    "http://aims.fao.org/aos/agrovoc/c_8373",  # wheat
    languages=["en", "ru"],
    label_types=["prefLabel", "altLabel"],
    include_narrower=True,
    narrower_depth=1,
)
# Returns: [
#   ResolvedAlias(alias="wheat", language="en", label_type="prefLabel", ...),
#   ResolvedAlias(alias="common wheat", language="en", label_type="altLabel", ...),
#   ResolvedAlias(alias="пшеница", language="ru", label_type="prefLabel", ...),
#   ResolvedAlias(alias="durum wheat", language="en", label_type="prefLabel", ...),  # narrower
# ]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `uri` | `str` | required | AGROVOC concept URI |
| `languages` | `list[str]` | `["en"]` | Language codes to retrieve |
| `label_types` | `list[str]` | `["prefLabel", "altLabel"]` | SKOS label predicates to query |
| `include_narrower` | `bool` | `False` | Traverse `skos:narrower` children |
| `narrower_depth` | `int` | `1` | Max depth for narrower traversal |

---

## Offline Mode Details

The offline adapter parses an N-Triples file into an rdflib `Graph` and queries it via triple pattern matching. On first load, a pickle cache (`.pkl`) is written next to the source file. Subsequent loads use the cache directly (checked via `st_mtime`).

Queried predicates:
- `skos:prefLabel` --- preferred label (one per language)
- `skos:altLabel` --- alternative labels (synonyms, common names)

Narrower traversal follows `skos:narrower` edges recursively up to `narrower_depth` levels.

## Online Mode Details

The online adapter sends SPARQL queries to the public AGROVOC endpoint via httpx. Each `resolve_concept()` call generates one SPARQL SELECT per concept (plus one per narrower child if traversal is enabled). No authentication required.

---

## Test Coverage

9 tests in `tests/test_agrovoc.py` using a fixture graph (`conftest.py`) containing wheat, barley, maize, buckwheat, sunflowers, and durum wheat (narrower of wheat) with English and Russian labels.
