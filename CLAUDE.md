# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF-OCR is a Python project that uses BAML (BoundaryML) to extract structured data from PDF documents via LLMs. It combines PyMuPDF for PDF parsing with LLM providers (OpenAI, Anthropic) for intelligent data extraction.

## Commands

```bash
# Install dependencies
uv sync

# Regenerate BAML client after editing baml_src/ files
uv run baml-cli generate

# Run a quick test of the spatial text renderer
uv run python -c "from pdf_ocr import pdf_to_spatial_text; print(pdf_to_spatial_text('inputs/2857439.pdf'))"
```

## Environment Variables

LLM API keys must be set depending on which client is used:
- `OPENAI_API_KEY` — for GPT-5/GPT-5-mini clients
- `ANTHROPIC_API_KEY` — for Claude clients

## Architecture

### Source Package (`src/pdf_ocr/`)

All application code lives here following the standard Python src layout.

- **`__init__.py`** — Package root, re-exports public API.
- **`spatial_text.py`** — Spatial PDF-to-text renderer. Converts each PDF page into a monospace text grid preserving columns, tables, and scattered text at their correct visual positions. Public function: `pdf_to_spatial_text(pdf_path, *, pages, cluster_threshold, page_separator)`. Also exports `PageLayout` dataclass and `_extract_page_layout()` helper used by `compress.py`.
- **`compress.py`** — Compressed spatial text for LLM consumption. Classifies page regions (tables, text blocks, headings, key-value pairs, scattered) and renders them as markdown tables, flowing paragraphs, and structured key-value lines. Multi-row stacked headers are joined with ` / ` separators (matching the DOCX extractor convention) to enable deterministic compound header parsing. Supports multi-row record merging for shipping stems. Public function: `compress_spatial_text(pdf_path, *, pages, cluster_threshold, page_separator, table_format, merge_multi_row, min_table_rows)`.
- **`interpret.py`** — Table interpretation pipeline with **deterministic-first architecture**: tries alias-based mapping before any LLM call. When every header part (split on ` / `) matches a schema alias, the entire interpretation — including unpivoting compound headers — is done via string matching with zero LLM calls. Falls back to the LLM pipeline only for pages with unmatched columns. LLM path: 2-step (parse then map) and single-shot modes, with auto page-splitting, **pre-step-1 table splitting** for large tables (`step1_max_rows=40`, prevents LLM output truncation on 65+ row tables), batched step 2, and async concurrency. Supports an optional **vision-based schema inference** mode for PDFs with garbled/concatenated headers: pass `pdf_path=` to `interpret_table()` to render pages as images and use a vision LLM to infer column structure before parsing. Includes **deterministic section boundary validation**: after the LLM parses table structure (step 1), Python code counts pipe-table data rows per section directly from the compressed text and corrects any miscounted boundaries before batching. The **`UnpivotStrategy`** enum (`SCHEMA_AGNOSTIC`, `DETERMINISTIC`, `NONE`) controls how pivoted tables are handled: `SCHEMA_AGNOSTIC` pre-unpivots for the LLM via `unpivot.py`, `DETERMINISTIC` lets the deterministic mapper handle pivots and sends the original to the LLM, `NONE` skips all pivot handling. The `unpivot` parameter on all entry points accepts both the enum and `bool` for backward compatibility (`True` → `SCHEMA_AGNOSTIC`, `False` → `NONE`). Key public functions: `interpret_table()`, `interpret_table_single_shot()`, `infer_table_schema_from_image()`, `to_records()`, `to_records_by_page()`.
- **`serialize.py`** — Serialization module for exporting `interpret_table()` results to various formats. Validates records against the CanonicalSchema using Pydantic before serialization, with automatic coercion of OCR artifacts (e.g., `"1,234"` → `1234`, `"(500)"` → `-500`). Applies output formatting from `ColumnDef.format` field — supports date/time patterns (`YYYY-MM-DD`, `HH:mm`), number patterns (`#,###.##`, `+#`, `#%`), and string case transformations (`uppercase`, `camelCase`, `snake_case`, etc.). Public functions: `to_csv(result, schema, *, path=None, include_page=False)`, `to_tsv(...)`, `to_parquet(result, schema, path, *, include_page=False)`, `to_pandas(...)`, `to_polars(...)`. Optional dependencies: `pip install pdf-ocr[dataframes]` for pandas/polars, `pip install pdf-ocr[parquet]` for Parquet support.
- **`normalize.py`** — Centralized pipe-table text normalization. Single function `normalize_pipe_table(text)` that applies lossless, idempotent normalizations to pipe-table markdown: NBSP → space, smart quotes → ASCII, en/em-dash → hyphen, zero-width character removal, whitespace collapse, trailing whitespace strip. Applied automatically at entry points of `classify_tables()`, `interpret_table()`, `interpret_table_single_shot()`, `interpret_table_async()`, and `interpret_tables_async()`.
- **`unpivot.py`** — Schema-agnostic deterministic pivot detection and unpivoting for pipe-tables. Detects repeating compound header groups (e.g. `"crop A / 2025"`, `"crop B / 2025"`) by splitting on ` / `, grouping by prefix, and fuzzy-matching suffix lists across groups using `rapidfuzz.fuzz.ratio()`. When ≥2 groups share ≥2 matching suffixes, transforms the wide pivoted table into long format with a `_pivot` column containing the group prefix. Non-matching groups (e.g. `"Th.ha. / Region"`) become shared columns. Runs as a pre-processing step before the LLM path only (the deterministic mapper handles pivoted tables natively via alias-based group detection, so unpivoting is applied after the deterministic attempt fails). Public functions: `unpivot_pipe_table(text, *, similarity_threshold, min_groups, pivot_column_name)`, `detect_pivot(text, *, similarity_threshold, min_groups)`.
- **`classify.py`** — Format-agnostic table classification. Operates on `list[tuple[str, dict]]` — the compressed pipe-table markdown + metadata tuples produced by both `compress_docx_tables()` (DOCX) and `StructuredTable.to_compressed()` (PDF). Three-layer approach: (1) word-boundary keyword scoring against header text with English suffix tolerance, (2) `min_data_rows` filtering, (3) optional similarity propagation from matched to unmatched tables. The ` / ` compound header separator is collapsed before matching so multi-word keywords like `"area harvested"` work on compound headers like `"Area / harvested / 2025"`. Public function: `classify_tables(tables, categories, *, min_data_rows, propagate, propagate_threshold)`. Also exports `_YEAR_RE` (used by `docx_extractor.py` for pivot value extraction) and helpers `_keyword_matches`, `_compute_similarity`, `_parse_pipe_header`, `_tokenize_header_text`.
- **`contracts.py`** — Contract helpers for loading JSON data contracts and transforming DataFrames. **Prepare helpers**: `load_contract(path)` parses a JSON contract into a typed `ContractContext` dataclass (with `OutputSpec` per output containing the `CanonicalSchema`, enrichment rules, and column specs); `resolve_year_templates(aliases, pivot_years)` resolves `{YYYY}`, `{YYYY-1}`, `{YYYY+1}` patterns in schema aliases using document-extracted years. **Transform helpers**: `enrich_dataframe(df, enrichment, *, title, report_date)` adds contract-specified enrichment columns (`source: "title"/"report_date"/"constant"`); `format_dataframe(df, col_specs)` applies column-level case transformations (`lowercase`/`uppercase`/`titlecase`) and row filters (`latest`/`earliest`). Each helper is small, stateless, and composable — orchestration stays in the notebook.
- **`pipeline.py`** — Async pipeline orchestration helpers. Four composable layers: (1) `compress_and_classify_async(doc_path, categories, output_specs)` — compresses a document and classifies its tables into output categories, with DOCX/PDF branching and `asyncio.to_thread()` for CPU-bound PDF compression; (2) `interpret_output_async(data, output_spec, *, model, unpivot, report_date, pivot_years)` — interprets one output category end-to-end (deep-copies schema, resolves year templates, calls `_interpret_pages_batched_async`, enriches, formats, reorders columns); (3) `process_document_async(doc_path, cc)` — convenience layer composing both helpers with report-date resolution, returning a `DocumentResult` dataclass; (4) `run_pipeline_async(contract_path, doc_paths, output_dir)` — top-level orchestrator that loads the contract, processes all documents concurrently, merges DataFrames across documents, and writes Parquet output. Also exports `save(results, output_dir)` for standalone merge+write. All helpers are independently useful and stateless. The `DocumentResult` dataclass holds `doc_path`, `report_date`, `compressed_by_category`, and `dataframes`.

### BAML Source Layer (`baml_src/`)

This is where AI functions, data models, and LLM client configurations are defined using the BAML DSL. After editing these files, run `uv run baml-cli generate` to regenerate the Python client.

- **`interpret.baml`** — Table interpretation types and LLM functions. Defines the 2-step pipeline (`AnalyzeAndParseTable` → `MapToCanonicalSchema`), single-shot `InterpretTable`, and vision-based variants (`InferTableSchemaFromImage`, `AnalyzeAndParseTableGuided`). Both parse functions include explicit section boundary format instructions (`"Sections: LABEL (rows 0-N), ..."`) so `_parse_section_boundaries()` can regex-parse them. The vision prompt cross-validates column count between image and compressed text pipe-cell count. Types include `ParsedTable`, `MappedTable`, `CanonicalSchema`, `InferredTableSchema`, etc.
- **`resume.baml`** — Defines the `Resume` data model and `ExtractResume` function with prompt template. Contains test cases runnable via the BAML VSCode extension playground.
- **`clients.baml`** — LLM client configurations (OpenAI, Anthropic, plus commented-out Google, Azure, Bedrock, Vertex, Ollama). Includes composite clients: `CustomFast` (round-robin) and `OpenaiFallback` (fallback chain). Defines retry policies (constant and exponential backoff).
- **`generators.baml`** — Code generation config targeting `python/pydantic` with sync client mode. Version must match `baml-py` in `pyproject.toml`.

### Generated Client (`baml_client/`)

Auto-generated Python code — **do not edit manually**. Provides:
- `b.ExtractResume(resume_text)` — sync API (via `sync_client.py`)
- `b.stream.ExtractResume(...)` — streaming with partial results
- Pydantic models in `types.py` matching BAML class definitions

Usage: `from baml_client import b`

### PDF Inputs (`inputs/`)

Test PDF files including shipping statements and edge-case documents (multi-column, overlapping text, tiny fonts).

### Notebooks

- **`capabilities.ipynb`** — Sequential walkthrough of every transformation step (spatial text → compression → classification → schema → LLM interpretation → serialization). Format-agnostic, educational.
- **`pipeline.ipynb`** — End-to-end contract-driven pipelines with async concurrency. Three use cases: Russian agricultural DOCX reports, Australian shipping stems (6 PDF providers), and ACEA car registrations (pivoted table normalization).
- **`walkthrough_legacy.ipynb`** / **`walkthrough_docx_legacy.ipynb`** — Original format-specific notebooks (preserved for reference).

### Data Contracts (`contracts/`)

JSON files that declaratively define extraction pipelines: LLM model, table classification keywords, output schemas (columns, types, aliases, formats), enrichment rules, and unpivot strategy. The pipeline code in `pipeline.ipynb` is 100% generic — swap the contract, not the code.

Top-level contract fields:
- **`provider`** — Human-readable name of the data source.
- **`model`** — LLM model identifier (default `"openai/gpt-4o"`).
- **`unpivot`** — Optional. Controls how pivoted tables are handled. Values: `"schema_agnostic"` (pre-unpivot for LLM, default), `"deterministic"` (deterministic mapper handles pivots, LLM sees original), `"none"` (skip all pivot handling), or boolean `true`/`false` for backward compatibility.
- **`categories`**, **`outputs`**, **`report_date`** — See individual contracts for structure.

- **`ru_ag_ministry.json`** — Russian Ministry of Agriculture weekly grain reports (harvest + planting)
- **`au_shipping_stem.json`** — Australian shipping stem vessel loading records (6 providers, 1 canonical schema)
- **`acea_car_registrations.json`** — European new car registrations by market and power source

## Key Constraints

- Python >=3.13 required
- Project uses src layout — all application code goes in `src/pdf_ocr/`
- BAML version in `generators.baml` must stay in sync with the `baml-py` dependency version in `pyproject.toml`
- The `baml_client/` directory is fully generated; changes belong in `baml_src/`
- **Always regenerate the BAML client** after editing any file in `baml_src/` by running `uv run baml-cli generate`

## BAML Guidelines

- **Mandatory descriptions**: Every BAML `enum` value and `class` field that influences LLM behavior MUST have a `@description()` annotation. These descriptions are embedded in LLM prompts and are critical for correct classification (e.g., `TableType.TransposedTable` must explain what a transposed table looks like so the model can identify it). Missing descriptions cause misclassification bugs. **Exception for self-explanatory enums**: Descriptions are optional when the enum value name is universally unambiguous on its own (e.g., `AggregationType.Sum`, `AggregationType.Min`). If a value's meaning depends on domain context or could be confused with something else, a description is still mandatory.

- **No redundancy between descriptions and prompts**: `@description()` annotations define WHAT a field or value IS (semantic meaning). Prompt templates define HOW to use that information (operational instructions). Do not repeat identification criteria from enum descriptions inside prompt templates — `{{ ctx.output_format }}` already injects all descriptions into the prompt. Prompts should focus on decision processes, mapping strategies, and extraction rules that go beyond what the type definitions express.

- **Domain-agnostic descriptions and examples**: ALL text that reaches the LLM — `@description()` annotations, prompt templates, and inline examples — MUST use generic, domain-neutral language. Use abstract structural terms (e.g., "category labels", "entity names", "measure values", "Item X", "Period A") rather than domain-specific terminology (e.g., "port names", "vessel names", "tonnage", "Wheat", "Oct 2025"). This applies equally to enum value descriptions, class field descriptions, KEY SIGNS lists, and prompt examples. Domain-specific language biases the model toward particular document types and causes misclassification on other domains.

- **Keep descriptions and README in sync**: The `TableType` enum descriptions in `interpret.baml` must match the table in `src/pdf_ocr/README.md` under "Table Types and Mapping Strategies". When updating one, update the other.

## Testing Guidelines

- **Bug fixes must be validated against the original failing case first.** When a concrete bug triggers a fix (e.g., "spring grain columns are wrong in the June DOCX"), the very first test is to reproduce the failure end-to-end and confirm the fix resolves it. Run the actual pipeline on the actual document. Inspect the actual output. Only after the real-world case is confirmed fixed should you write unit tests for edge cases and regressions. Unit tests prove the mechanism works in isolation; they do NOT prove the bug is fixed. A normalizer that passes 34 unit tests but was never run against the document that exposed the problem is untested where it matters.

- **Rigorous, not superficial**: Don't just check that code runs or that output "looks right". Verify actual content accuracy — column names must match the source PDF exactly, data values must be in correct columns, order must be preserved.

- **Compare against source**: For compress/header refinement changes, compare LLM output against the actual PDF. Open the PDF, read the header text, and verify the output matches character-for-character.

- **Test the hard cases**: Bunge has 9 wrapped header rows with complex stacking. 2857439 has hierarchical headers. CBH has side-by-side tables. Queensland has transposed layout. These are the real tests — not whether simple cases work.

- **Verify column count and order**: Count columns in the source PDF header. Count columns in the output. They must match. Verify column order matches left-to-right reading order in the PDF.

- **No hallucination tolerance**: If the LLM produces column names that don't exist in the PDF, or merges/splits columns incorrectly, that's a failure. The goal is faithful reproduction, not creative interpretation.

- **Document expected outputs**: For key test PDFs, document what the correct output should be (column names, column count, data row count) so regressions are immediately visible.

- **Zero failing tests before commit**: Always run the full test suite (`uv run pytest tests/`) before committing or pushing to git. Every test must pass. If any test fails — whether caused by your changes or pre-existing — investigate and fix it before committing. No exceptions. Broken tests that get committed silently become invisible regressions.
