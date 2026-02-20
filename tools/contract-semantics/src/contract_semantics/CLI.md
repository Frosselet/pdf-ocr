# CLI

> **Module**: `cli.py`
> **Entry point**: `contract-semantics` (installed via `pyproject.toml`)

Click-based command-line interface for all contract-semantics operations. Seven commands covering the full workflow: data fetching, resolution, materialization, diffing, validation, and shape generation.

---

## Commands

### fetch-agrovoc

Download the AGROVOC N-Triples dump for offline resolution.

```bash
contract-semantics fetch-agrovoc
contract-semantics fetch-agrovoc -o /path/to/data/
```

Downloads `agrovoc_core.nt.zip` from FAO's release server. Extract manually after download.

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | `data/` | Output directory |

### fetch-geonames

Download a GeoNames country extract for offline resolution.

```bash
contract-semantics fetch-geonames
contract-semantics fetch-geonames --country US
contract-semantics fetch-geonames -c RU -o /path/to/data/
```

Downloads `{COUNTRY}.zip` from GeoNames export server.

| Option | Default | Description |
|---|---|---|
| `-c`, `--country` | `RU` | ISO country code |
| `-o`, `--output` | `data/` | Output directory |

### diff

Compare ontology-resolved aliases against manual aliases for all annotated columns in a contract.

```bash
contract-semantics diff contracts/ru_ag_ministry.json
```

For each column with `concept_uris`, resolves concepts via the appropriate adapter and prints a coverage report. Adapters are created lazily: AGROVOC uses the local dump if available, otherwise falls back to the public SPARQL endpoint. GeoNames requires a local dump (run `fetch-geonames` first).

### materialize

Resolve all concept URIs in an annotated contract and produce a standard contract with enriched aliases.

```bash
contract-semantics materialize contracts/ru_ag_ministry.json \
    -o contracts/ru_ag_ministry_materialized.json

contract-semantics materialize contracts/ru_ag_ministry.json \
    -o contracts/materialized.json \
    --geo-sidecar contracts/ru_ag_ministry_geo.json \
    --merge union
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output` | required | Output materialized contract path |
| `--geo-sidecar` | none | Output geo sidecar path (GeoNames metadata) |
| `--merge` | `union` | Merge strategy: `union`, `resolved_only`, `manual_priority` |

### validate

Validate a CSV file against a SHACL shapes graph. Exits with code 0 if valid, 1 if violations found.

```bash
contract-semantics validate output.csv shapes/harvest_output.ttl
```

Requires `pyshacl` (`pip install contract-semantics[shacl]`).

### generate-shapes

Auto-generate SHACL shapes from a contract's output schemas.

```bash
contract-semantics generate-shapes contracts/ru_ag_ministry.json
contract-semantics generate-shapes contracts/ru_ag_ministry.json -o shapes/ --output-name harvest
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | `shapes/` | Output directory for shape files |
| `--output-name` | all | Only generate shapes for this output |

---

## Adapter Resolution Order

Both `diff` and `materialize` commands resolve adapters lazily:

| Source | Priority 1 (offline) | Priority 2 (online) |
|---|---|---|
| AGROVOC | `data/agrovoc_core.nt` | Public SPARQL endpoint |
| GeoNames | `data/RU.txt` | Not supported (run `fetch-geonames` first) |

The `data/` directory is relative to the package installation path (`tools/contract-semantics/data/`).

---

## Full Workflow Example

```bash
# 1. Fetch ontology data
contract-semantics fetch-agrovoc
cd data && unzip agrovoc_core.nt.zip && cd ..
contract-semantics fetch-geonames --country RU
cd data && unzip RU.zip && cd ..

# 2. Inspect coverage
contract-semantics diff ../../contracts/ru_ag_ministry.json

# 3. Materialize
contract-semantics materialize ../../contracts/ru_ag_ministry.json \
    -o ../../contracts/ru_ag_ministry_materialized.json \
    --geo-sidecar ../../contracts/ru_ag_ministry_geo.json

# 4. Generate validation shapes
contract-semantics generate-shapes ../../contracts/ru_ag_ministry.json -o shapes/

# 5. Run pipeline with materialized contract, then validate
contract-semantics validate output.csv shapes/harvest_output.ttl
```
