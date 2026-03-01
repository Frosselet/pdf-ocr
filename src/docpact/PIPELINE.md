# Pipeline Orchestration

> **Module**: `pipeline.py`
> **Public API**: `compress_and_classify_async()`, `interpret_output_async()`, `process_document_async()`, `run_pipeline_async()`, `save()`, `DocumentResult`

Async pipeline orchestration helpers for contract-driven document extraction. Four composable layers from granular to convenient.

## Why

The extraction pipeline involves multiple CPU-bound and I/O-bound steps (compression, classification, LLM interpretation, enrichment) that benefit from async concurrency. This module composes the lower-level helpers (`compress.py`, `classify.py`, `interpret.py`, `contracts.py`, `semantics.py`) into reusable async building blocks.

## Four Layers

### Layer 1: `compress_and_classify_async()`

Compresses a document and classifies its tables into output categories.

- **DOCX path**: `classify_docx_tables()` → `compress_docx_tables()` per category
- **PDF path**: `compress_spatial_text()` via `asyncio.to_thread()`, then optionally `compress_spatial_text_structured()` + `classify_tables()` for multi-category contracts
- **Single-category optimization**: Skips classification when only one output exists

Returns `{output_name: compressed_data}`.

### Layer 2: `interpret_output_async()`

Interprets compressed data for one output category end-to-end:

1. Deep-copies the schema (no caller side-effects)
2. Resolves `{YYYY}` alias templates from `pivot_years`
3. Merges resolved aliases from `SemanticContext` (additive)
4. Resolves `{YYYY}` in enrichment constant values
5. Calls `_interpret_pages_batched_async()` for LLM/deterministic interpretation
6. Enriches DataFrame with contract-specified columns
7. Formats DataFrame (case transforms, row filters)
8. Reorders columns to match contract spec

Handles DOCX (per-table enrichment with title) vs PDF (whole-DataFrame enrichment) branching.

### Layer 3: `process_document_async()`

Full per-document pipeline composing Layers 1 and 2:

1. Resolve report date (from contract config)
2. Compress & classify
3. Pre-flight check (when `semantic_context` provided)
4. Extract pivot years (DOCX only)
5. Interpret all output categories concurrently (`asyncio.gather`)
6. Post-extraction validation (when `semantic_context` provided)

Returns a `DocumentResult` dataclass.

### Layer 4: `run_pipeline_async()`

Top-level orchestrator:

1. Load contract (`load_contract()`)
2. Process all documents concurrently (`asyncio.gather`)
3. Merge DataFrames across documents
4. Write output files (Parquet/CSV/TSV)
5. Report timing

## `DocumentResult`

| Field | Type | Description |
|-------|------|-------------|
| `doc_path` | `str` | Path to source document |
| `report_date` | `str` | Resolved report date |
| `compressed_by_category` | `dict` | Compressed data keyed by output name |
| `dataframes` | `dict[str, DataFrame]` | Interpreted DataFrames keyed by output name |
| `preflight_reports` | `dict` | Pre-flight header check results |
| `validation_reports` | `dict` | Post-extraction validation results |

## `save()`

Merges DataFrames across documents and writes each output to file. Format determined by file extension: `.csv` → CSV, `.tsv` → TSV, anything else → Parquet. Filenames come from the contract's `OutputSpec.filename` field.

## API Summary

| Function | Scope | Async |
|----------|-------|-------|
| `compress_and_classify_async()` | Single document → compressed data | Yes |
| `interpret_output_async()` | One output category → DataFrame | Yes |
| `process_document_async()` | Single document → full result | Yes |
| `run_pipeline_async()` | Multiple documents → saved output | Yes |
| `save()` | Merge + write results | No |
