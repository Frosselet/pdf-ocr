# context.py — SemanticContext Bridge Builder

Bridges the gap between `contract_semantics` (ontology resolution) and
`docpact` (pipeline execution) by building a `SemanticContext` data object
that the pipeline can consume.

## Architecture

This module is the **only place** that imports from both packages:

- Reads annotated contracts and resolves concept URIs via `contract_semantics` adapters
- Produces a `docpact.semantics.SemanticContext` (plain dataclass)

The `SemanticContext` is then passed into `process_document_async()` or
`run_pipeline_async()` — docpact never imports from `contract_semantics`.

## Public Functions

### `build_context_data(contract_path, *, agrovoc=None, geonames=None, merge_strategy="union") → dict`

Core resolution logic, independent of `docpact`. Returns a plain dict suitable for constructing a `SemanticContext` via `from_dict()`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `contract_path` | `str \| Path` | Path to annotated contract JSON |
| `agrovoc` | `OntologyAdapter \| None` | AGROVOC adapter (offline or online) |
| `geonames` | `OntologyAdapter \| None` | GeoNames adapter (offline or online) |
| `merge_strategy` | `str` | `"union"` (default), `"resolved_only"`, `"manual_priority"` |

**Returns:** Dict with `resolved_aliases`, `valid_values`, `resolved_at`, and `adapter_versions` keys.

### `build_semantic_context(contract_path, *, agrovoc=None, geonames=None, merge_strategy="union", cache_path=None) → SemanticContext`

Convenience wrapper: calls `build_context_data()` and wraps the result in a `SemanticContext`. Requires `docpact` to be installed.

**Additional parameters:**

| Parameter | Type | Description |
|---|---|---|
| `cache_path` | `str \| Path \| None` | If given, write context to this JSON path |

**Returns:** `SemanticContext` with:

- `resolved_aliases` — merged aliases per output/column
- `valid_values` — concept labels + resolved aliases as valid value sets
- `resolved_at` — ISO timestamp
- `adapter_versions` — adapter class names for provenance

## CLI Command

```bash
contract-semantics build-context contracts/ru_ag_ministry.json -o semantic_context.json
```

Options:
- `--output / -o` (required) — output JSON path
- `--merge` — merge strategy (`union`, `resolved_only`, `manual_priority`)

## Usage

### Python API

```python
from contract_semantics.agrovoc import AgrovocAdapter
from contract_semantics.geonames import GeoNamesAdapter
from contract_semantics.context import build_semantic_context

agrovoc = AgrovocAdapter.from_file("data/agrovoc_core.nt")
geonames = GeoNamesAdapter.from_file("data/RU.txt")

ctx = build_semantic_context(
    "contracts/ru_ag_ministry.json",
    agrovoc=agrovoc,
    geonames=geonames,
    cache_path="semantic_context.json",
)
```

### With the pipeline

```python
from docpact.pipeline import process_document_async
from docpact.contracts import load_contract
from docpact.semantics import SemanticContext

cc = load_contract("contracts/ru_ag_ministry.json")
ctx = SemanticContext.from_json("semantic_context.json")

result = await process_document_async(doc_path, cc, semantic_context=ctx)
print(result.preflight_reports)
print(result.validation_reports)
```
