# Contract Helpers

> **Module**: `contracts.py`
> **Public API**: `load_contract()`, `resolve_year_templates()`, `enrich_dataframe()`, `format_dataframe()`, `ContractContext`, `OutputSpec`, `SemanticColumnSpec`

Granular, composable building blocks for the prepare phase (contract loading) and transform phase (DataFrame enrichment/formatting). Each helper is independently useful — orchestration stays in the notebook or pipeline.

## Why

JSON data contracts declaratively define the extraction pipeline. This module is the bridge between the JSON file and the typed Python objects that `pipeline.py`, `interpret.py`, and `semantics.py` consume. Separating load from transform keeps each function small and testable.

## Data Classes

### `ContractContext`

Top-level typed representation of a parsed JSON contract.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Human-readable provider name |
| `model` | `str` | LLM model identifier |
| `unpivot` | `UnpivotStrategy \| bool` | Unpivot strategy or bool for backward compat |
| `categories` | `dict[str, list[str]]` | Category name → keyword list |
| `outputs` | `dict[str, OutputSpec]` | Output name → specification |
| `report_date_config` | `ReportDateConfig \| None` | Declarative report-date config |
| `raw` | `dict` | Original JSON dict for custom access |
| `has_semantic_annotations` | `bool` | True if any column has `concept_uris` |

### `OutputSpec`

Per-output specification parsed from the contract.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Output name key (e.g., `"harvest"`) |
| `category` | `str` | Category name for classification |
| `filename` | `str \| None` | Suggested output filename |
| `schema` | `CanonicalSchema` | Schema with LLM-extracted columns only |
| `enrichment` | `dict[str, dict]` | Source-based columns (`"title"`, `"report_date"`, `"constant"`) |
| `col_specs` | `list[dict]` | Raw column specs for formatting/filtering |
| `semantic_columns` | `dict[str, SemanticColumnSpec]` | Columns with `concept_uris` annotations |

### `SemanticColumnSpec`

Semantic metadata for a contract column (parsed from `concept_uris`, `resolve`, `semantic` fields).

| Field | Type | Description |
|-------|------|-------------|
| `column_name` | `str` | Schema column name |
| `concept_uris` | `list[dict]` | Raw concept URI + label dicts |
| `resolve_config` | `dict` | Ontology resolution configuration |
| `validate` | `bool` | Whether to validate extracted values against known labels |

## Prepare Helpers

### `load_contract(path) -> ContractContext`

Parses a JSON contract into typed dataclasses. Separates LLM columns (for `CanonicalSchema`) from enrichment columns (source-based). Extracts semantic annotations when present.

### `resolve_year_templates(aliases, pivot_years) -> list[str]`

Resolves `{YYYY}`, `{YYYY-1}`, `{YYYY+1}` patterns in aliases using document-extracted years. The latest year in `pivot_years` is the base. When `pivot_years` is empty, aliases containing templates are stripped.

## Transform Helpers

### `enrich_dataframe(df, enrichment, *, title=None, report_date=None)`

Adds contract-specified enrichment columns to a DataFrame (in-place):
- `source: "title"` → uses the `title` argument
- `source: "report_date"` → uses the `report_date` argument, with optional suffix
- `source: "constant"` → uses the `value` field from the spec

### `format_dataframe(df, col_specs)`

Applies column-level transformations:
- **Case formats**: `"lowercase"`, `"uppercase"`, `"titlecase"`
- **Row filters**: `"latest"` (keep max), `"earliest"` (keep min)

Other format values (date patterns, number patterns) are handled by `serialize.py`.
