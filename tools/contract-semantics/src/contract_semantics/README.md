# contract-semantics — Ontology-Grounded Contract Authoring

Ground data contracts in linked-data ontologies so aliases can be auto-generated, multilingual labels come for free, and outputs can be validated against SHACL shapes.

---

## Why This Matters

Contract aliases are hand-curated flat strings. Two contracts mentioning "wheat" have no way to know they mean the same thing. A new language or a new synonym means manually editing every contract that uses the term.

Ontology grounding solves this by linking each concept to a canonical URI in a standard vocabulary. Resolution generates aliases automatically from the ontology's multilingual labels. The materialized contract JSON stays backward-compatible --- docpact consumes it unchanged. The semantic intelligence lives entirely in the authoring and validation layers.

---

## Architecture

```
Annotated Contract (concept_uris + resolve config)
 |
 |-- agrovoc.py ---------> AGROVOC SKOS labels (crops, metrics)
 |                          prefLabel / altLabel in en, ru, ...
 |
 |-- geonames.py --------> GeoNames alternate names (regions)
 |                          + geographic enrichment (lat/lng/admin codes)
 |
 |-- resolve.py ----------> OntologyAdapter protocol + prefix pattern expansion
 |                          "spring {label}" x "wheat" -> "spring wheat"
 |
 |-- materialize.py ------> Materialized contract JSON (aliases enriched)
 |                          + geo sidecar JSON (coordinates for GIS joins)
 |
 |-- diff.py -------------> Coverage report: resolved vs manual aliases
 |
 |-- validate.py ---------> SHACL shapes: records -> RDF -> pyshacl validation
 |
 |-- cli.py --------------> Click CLI: materialize, diff, fetch-*, validate
 |
 v
Materialized Contract (standard format, consumed by docpact)
```

## Module Index

| Module | Purpose | Doc |
|---|---|---|
| `models.py` | Pydantic data models shared across all modules | [MODELS.md](MODELS.md) |
| `agrovoc.py` | AGROVOC SKOS adapter (offline rdflib + online SPARQL) | [AGROVOC.md](AGROVOC.md) |
| `geonames.py` | GeoNames adapter (offline TSV + online REST API) | [GEONAMES.md](GEONAMES.md) |
| `resolve.py` | Generic resolution protocol and column resolver | [RESOLVE.md](RESOLVE.md) |
| `materialize.py` | Contract materialization + geo sidecar output | [MATERIALIZE.md](MATERIALIZE.md) |
| `diff.py` | Alias comparison reporting | [DIFF.md](DIFF.md) |
| `validate.py` | SHACL validation and shape generation | [VALIDATE.md](VALIDATE.md) |
| `cli.py` | Click CLI entry points | [CLI.md](CLI.md) |

## Quick Start

```bash
# Install
cd tools/contract-semantics
uv sync

# Download ontology data
contract-semantics fetch-agrovoc
contract-semantics fetch-geonames --country RU

# Compare ontology aliases vs manual aliases
contract-semantics diff contracts/ru_ag_ministry.json

# Materialize: resolve URIs -> enriched contract
contract-semantics materialize contracts/ru_ag_ministry.json \
    -o contracts/ru_ag_ministry_materialized.json \
    --geo-sidecar contracts/ru_ag_ministry_geo.json

# Generate SHACL shapes from contract
contract-semantics generate-shapes contracts/ru_ag_ministry.json

# Validate pipeline output
contract-semantics validate output.csv shapes/harvest_output.ttl
```

## Dependencies

```toml
dependencies = [
    "rdflib>=7.0,<8",    # RDF graph parsing (AGROVOC N-Triples, SHACL Turtle)
    "pydantic>=2.0",      # Data models
    "click>=8.0",         # CLI framework
    "httpx>=0.27",        # HTTP client (SPARQL, GeoNames REST API)
]

[project.optional-dependencies]
shacl = ["pyshacl>=0.26"]  # SHACL validation (validate.py)
```

## Testing

```bash
cd tools/contract-semantics
uv sync --extra dev --extra shacl
uv run pytest tests/ -v
```

45 tests across 6 test files, all using fixture-based mock data (no network calls).

---

## Relationship to docpact

This package is **upstream** of docpact. The contract JSON is the sole interface:

- docpact never imports from `contract_semantics`
- `contract_semantics` never imports from `docpact`
- Unknown fields in contracts (`concept_uris`, `resolve`) are silently ignored by `load_contract()`

The pipeline code never changes. Swap the contract, not the code.
