# semantics.py — Semantic-Aware Pipeline Support

Provides semantic awareness for the docpact pipeline without importing from
`contract_semantics`.  Three capabilities:

1. **Runtime alias enrichment** via `SemanticContext`
2. **Pre-flight header checking** via `preflight_check()`
3. **Post-extraction value validation** via `validate_output()`

## Architecture

The `SemanticContext` is a plain dataclass built externally by
`contract_semantics.context.build_semantic_context()` and passed into pipeline
functions.  docpact never imports from `contract_semantics` — the
`SemanticContext` is the sole interface.

```
contract_semantics          docpact
┌─────────────────┐         ┌──────────────────────┐
│ build_semantic_  │ ──JSON──▶ SemanticContext       │
│ context()        │         │   .aliases_for()      │
│                  │         │   .valid_set_for()     │
└─────────────────┘         └──────────┬───────────┘
                                       │
                            ┌──────────▼───────────┐
                            │ pipeline.py           │
                            │  - alias merge        │
                            │  - preflight_check()  │
                            │  - validate_output()  │
                            └──────────────────────┘
```

## Data Structures

### `SemanticContext`

Pre-resolved semantic data.  Serializable to/from JSON for caching.

| Field | Type | Description |
|---|---|---|
| `resolved_aliases` | `dict[str, dict[str, list[str]]]` | `{output → {column → [alias, ...]}}` |
| `valid_values` | `dict[str, dict[str, set[str]]]` | `{output → {column → {value, ...}}}` |
| `resolved_at` | `str` | ISO timestamp of resolution |
| `adapter_versions` | `dict[str, str]` | Adapter provenance info |

**Methods:**
- `aliases_for(output_name, column_name) → list[str]`
- `valid_set_for(output_name, column_name) → set[str]`
- `to_dict() → dict`, `to_json(path)`, `from_dict(data)`, `from_json(path)`

### `PreFlightReport`

Result of comparing document headers against contract aliases.

| Field | Type | Description |
|---|---|---|
| `findings` | `list[PreFlightFinding]` | Individual info/warning findings |
| `header_coverage` | `float` | Fraction of doc headers matched (0.0–1.0) |
| `unmatched_headers` | `list[str]` | Doc headers not in any alias |
| `missing_aliases` | `list[str]` | Contract aliases with no matching doc header |

### `ValidationReport`

Result of checking extracted values against known valid concept labels.

| Field | Type | Description |
|---|---|---|
| `output_name` | `str` | Output category name |
| `total_rows` | `int` | Total DataFrame rows |
| `valid_count` | `int` | Rows with all validated columns passing |
| `invalid_count` | `int` | Rows with at least one finding |
| `findings` | `list[ValidationFinding]` | Per-value findings |
| `column_summaries` | `dict[str, dict]` | Per-column stats |

## Public Functions

### `preflight_check(data, output_spec, semantic_context=None) → PreFlightReport`

Compares document headers from compressed pipe-table text against all known
aliases (manual + resolved).  Informational only — never blocks extraction.

Uses `_normalize_for_alias_match()` from `interpret.py` for consistent
case-insensitive, whitespace-normalized matching.

### `validate_output(df, output_spec, semantic_context=None) → ValidationReport`

For columns with `validate=True` in their `SemanticColumnSpec`, checks every
value against the combined set of concept URI labels and resolved aliases.
Empty/NaN values are skipped.

Baseline validation works using `concept_uris[*].label` even without a
`SemanticContext`.  Richer validation with `SemanticContext` includes all
multilingual resolved aliases.

## Pipeline Integration

When `semantic_context` is passed to `process_document_async()`:

1. **After compress & classify** — `preflight_check()` runs for each output
2. **In `interpret_output_async()`** — resolved aliases merged into schema
3. **After enrichment** — `validate_output()` runs for each output

Results are stored in `DocumentResult.preflight_reports` and
`DocumentResult.validation_reports` (both default to `{}`).

## Backward Compatibility

- Callers that don't pass `semantic_context` get identical behavior to before
- `DocumentResult` new fields default to empty dicts
- Contracts without `concept_uris` → `semantic_columns` is empty, all features no-op
