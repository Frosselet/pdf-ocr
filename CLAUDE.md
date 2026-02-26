# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Design Philosophy

**Heuristics are science, not curve-fitting.**

Every deterministic heuristic in this codebase must be designed from first principles — grounded in a human understanding of document structure, not reverse-engineered from specific test cases. We define heuristics explicitly, formally, and with a clear structural rationale that a human would recognize as correct *before* seeing any particular PDF.

Core rules:

1. **General over specific.** A heuristic must describe a *class* of documents, never a single provider. "Tables where field labels occupy column 0 and data occupies columns 1–N" is a structural class. "The Queensland PDF" is not. If a rule only makes sense for one known document, it does not belong in code — it belongs in the contract JSON.

2. **Anticipate the unseen.** When two design choices both solve the known cases but differ in how they handle hypothetical future documents, prefer the one that handles more probable unseen layouts correctly — even if the other is simpler for today's inputs. We optimize for the universe of real-world documents, not for our current test corpus.

3. **Never overfit.** If a heuristic improves results on one provider but degrades or risks degrading results on another (real or plausible), it is wrong — regardless of net metrics. Every heuristic must be safe for all documents that match its structural preconditions.

4. **Contracts carry provider knowledge.** Aliases, port names, formatting quirks, column name variations — all provider-specific knowledge lives in the contract JSON. The code reads contracts; it never hard-codes domain terms. The pipeline is 100% generic; swap the contract, not the code.

5. **Honest testing.** Tests validate that a heuristic works for the structural reason we designed it for, not merely that it produces the right output on known inputs. Adversarial tests (near-miss cases, plausible false positives) are as important as positive cases. A test suite that only confirms known-good cases is a false sense of security.

This is the standard. When in doubt, choose the more principled path — even if it means leaving a known issue open until we find the right general solution.

## CK-IR Specification Authority

docpact is the reference implementation of CK-IR (Canonical Knowledge Intermediate Representation).
The normative specification lives in a separate repository.

- **Spec repo:** `/Volumes/WD Green/dev/git/ckir/ckir/`
- **Spec version:** Formal Spec v0.2 (Draft — Normative)
- **Conformance:** Level 2 (Deterministic) complete + partial Level 3 (Semantic)
- **Authority rule:** When this CLAUDE.md and the ckir spec disagree, the ckir spec is normative.
  If the implementation needs the spec to change, the change must happen in ckir first.

### Current Phase

> **Phase 1: Semantic Contracts — IN PROGRESS**
>
> Do not implement Phase 1.5+ features. If an idea belongs to a later phase,
> note it in a GitHub issue tagged with the phase number and move on.
>
> Phase 1 acceptance criteria (from ckir roadmap):
> 1. Add new region → GeoNames produces bilingual aliases automatically
> 2. SHACL validation flags values outside known concepts
> 3. Alias diff coverage ≥ 90%
> 4. Every alias traces to a concept URI

### Spec File Reference

| Spec File | Purpose | Governs |
|---|---|---|
| `specs/formal-spec-v0.2.md` | Normative CK-IR spec (RFC 2119 language) | All of `src/docpact/` |
| `specs/contract-schema.md` | Contract JSON format reference | `contracts.py`, contract JSON files |
| `specs/canonical-schema-reference.md` | ColumnDef model spec | `contracts.py`, `serialize.py`, `interpret.py` |
| `specs/jsonld-contract-format.md` | JSON-LD authoring format (Phase 1.5) | Future: `ckir compile` command |
| `architecture/blueprint.md` | 6-layer architecture + module map | Overall `src/docpact/` structure |
| `architecture/data-flows.md` | End-to-end document processing | Pipeline stage boundaries |
| `architecture/semantic-brain.md` | Semantic orchestration vision | `semantics.py`, `tools/contract-semantics/` |
| `strategy/roadmap.md` | 5-phase roadmap with acceptance criteria | What to build next |
| `strategy/repo-ecosystem.md` | 3-repo model, trigger maps, sync conventions | Cross-repo coordination |

All paths relative to `/Volumes/WD Green/dev/git/ckir/ckir/`.

### Spec Awareness Protocol

Rules for when Claude Code must consult ckir:

- **Before any architectural change:** Read `formal-spec-v0.2.md` (Sections 3-4) + `blueprint.md`
- **Module-specific rules:**
  - `interpret.py` → formal spec §4.2-4.5, §4.7
  - `classify.py` → formal spec §4.1
  - `contracts.py` → `contract-schema.md` + `canonical-schema-reference.md`
  - `serialize.py` → formal spec §4.9
  - `unpivot.py` → formal spec §4.8
  - `semantics.py` / `tools/contract-semantics/` → `semantic-brain.md`
  - Contract JSON edits → `contract-schema.md`
- **Before Phase 1.5+ work:** Read `roadmap.md` + `jsonld-contract-format.md`
- **When proposing new heuristics:** Verify against spec MUST/SHOULD/MAY requirements
- **Before any significant design decision:** Check `ckir/decisions/` for existing ADRs on the topic

### Roadmap Phase Tracking

| Phase | Name | ckir Status | docpact Status | Key Modules |
|---|---|---|---|---|
| 0 | Document Extraction MVP | DELIVERED | Complete | `interpret.py`, `compress.py`, `classify.py`, `serialize.py`, `pipeline.py` |
| 1 | Semantic Contracts | IN PROGRESS | In progress | `semantics.py`, `contracts.py`, `tools/contract-semantics/` |
| 1.5 | JSON-LD Contract Authoring | PLANNED | Not started | Future: compiler command |
| 2 | Contract Governance | PLANNED | Not started | Future: versioning, testing framework |
| 3 | Platform Services | PLANNED | Not started | Future: REST API, observability |
| 4 | Knowledge Graph | FUTURE | Not started | Future: RDF graph, SPARQL |

## Development Pattern

### Architecture Decision Records (ADRs)

Significant design decisions are recorded in `ckir/decisions/` as lightweight markdown files.
Before proposing an alternative to an existing approach, read the relevant ADR to understand
why the current approach was chosen.

ADR format:
- File: `decisions/NNN-short-title.md`
- Sections: Status, Context, Decision, Consequences
- Written when: a design choice affects multiple modules, reverses a previous approach,
  or involves a non-obvious tradeoff

### Cross-Repo Sync Conventions

- Phase completion → update Roadmap Phase Tracking table above + CHANGELOG.md entry
- Spec-impacting changes → spec must be updated in ckir first
- Never introduce behavior the formal spec prohibits
- Verify features against ckir `examples/` worked examples
- Significant design decision → write ADR in ckir before implementing
- Check ckir `strategy/repo-ecosystem.md` trigger map at phase transitions

### Changelog

`CHANGELOG.md` tracks milestone-level progress. One entry per significant capability or
phase transition. Not per-commit — per-milestone. Format:

    ## Phase 1: Semantic Contracts (in progress)
    ### 2026-02-26 — Semantic-aware pipeline
    - SemanticContext, preflight checks, post-extraction validation
    - build_semantic_context() bridge builder
    - 47 new tests, all 521 tests passing

## Project Overview

docpact is the reference implementation of CK-IR (Canonical Knowledge Intermediate Representation), a specification for transforming unstructured documents (PDF, DOCX, XLSX, PPTX, HTML) into typed, schema-conformant records through declarative JSON contracts. It combines format-specific extractors (PyMuPDF for PDF, python-docx for DOCX, openpyxl for XLSX) with a deterministic-first interpretation pipeline and LLM fallback via BAML.

## Commands

```bash
# Install dependencies
uv sync

# Regenerate BAML client after editing baml_src/ files
uv run baml-cli generate

# Run a quick test of the spatial text renderer
uv run python -c "from docpact import pdf_to_spatial_text; print(pdf_to_spatial_text('inputs/2857439.pdf'))"
```

## Environment Variables

LLM API keys must be set depending on which client is used:
- `OPENAI_API_KEY` — for GPT-5/GPT-5-mini clients
- `ANTHROPIC_API_KEY` — for Claude clients

## Architecture

### Source Package (`src/docpact/`) — 6-Layer Architecture

All application code lives here following the standard Python src layout.

- **`__init__.py`** — Package root, re-exports public API.

**Layer 1: Extraction**

- **`spatial_text.py`** — Spatial PDF-to-text renderer. Converts each PDF page into a monospace text grid preserving columns, tables, and scattered text at their correct visual positions. Public function: `pdf_to_spatial_text(pdf_path, *, pages, cluster_threshold, page_separator)`. Also exports `PageLayout` dataclass and `_extract_page_layout()` helper used by `compress.py`.
- **`compress.py`** — Compressed spatial text for LLM consumption. Classifies page regions (tables, text blocks, headings, key-value pairs, scattered) and renders them as markdown tables, flowing paragraphs, and structured key-value lines. Multi-row stacked headers are joined with ` / ` separators (matching the DOCX extractor convention) to enable deterministic compound header parsing. Supports multi-row record merging for shipping stems. Public function: `compress_spatial_text(pdf_path, *, pages, cluster_threshold, page_separator, table_format, merge_multi_row, min_table_rows)`.
- **`docx_extractor.py`** — DOCX table extractor. Handles `_tc` deduplication for merged cells, gridSpan/vMerge expansion, merge-based header detection, title row extraction, and boundary-aware forward-fill. Per-module docs in `DOCX_EXTRACTOR.md`.
- **`xlsx_extractor.py`** — XLSX/XLS table extractor. Eight XLSX-specific heuristics: **XH1** (blank-row/column table boundary detection — resolves multi-table-per-sheet via `_detect_table_regions()`), **XH2** (title row detection), **XH3** (hidden row/column filtering via `ws.column_dimensions[col].hidden`), **XH4** (number format interpretation for date/currency/percentage hints), **XH5** (header block detection — strips multi-row annotation blocks above tables via sparse-vs-dense row analysis), **XH6** (trailing column trimming — blank-column fence detection + headerless sparse edge trimming removes notes columns, status markers, dead formulas), **XH7** (trailing footnote detection — bottom-up scan strips `*`/`Source:`/`Note:` rows from data), **XH8** (aggregation row detection — dual-channel keyword + bold validation excludes TOTAL/Sum rows). Uses layered header detection: merge-based (horizontal merges + type continuation), type-pattern (TH2), and span-count (H7), taking the maximum. Forward-fill compound headers via `_build_column_names_with_forward_fill()`. Blank data rows (formatting breathing room) stripped post-extraction. Public functions: `extract_tables_from_xlsx()`, `extract_tables_from_excel()`, `xlsx_to_markdown()`. 86 tests, 20 synthetic fixtures across 5 personas (P1-P5). Per-module docs in `XLSX_EXTRACTOR.md`.
- **`pptx_extractor.py`**, **`html_extractor.py`** — Format-specific document extractors (untested, pending persona-driven validation per ADR-003).

**Layer 2: Classification**

- **`classify.py`** — Format-agnostic table classification. Operates on `list[tuple[str, dict]]` — the compressed pipe-table markdown + metadata tuples produced by both `compress_docx_tables()` (DOCX) and `StructuredTable.to_compressed()` (PDF). Three-layer approach: (1) word-boundary keyword scoring against header text with English suffix tolerance, (2) `min_data_rows` filtering, (3) optional similarity propagation from matched to unmatched tables. The ` / ` compound header separator is collapsed before matching so multi-word keywords like `"area harvested"` work on compound headers like `"Area / harvested / 2025"`. Public function: `classify_tables(tables, categories, *, min_data_rows, propagate, propagate_threshold)`. Also exports `_YEAR_RE` (used by `docx_extractor.py` for pivot value extraction) and helpers `_keyword_matches`, `_compute_similarity`, `_parse_pipe_header`, `_tokenize_header_text`.

**Layer 3: Interpretation**

- **`interpret.py`** — Table interpretation pipeline with **deterministic-first architecture**: tries alias-based mapping before any LLM call. The deterministic path runs two detectors per page in order: (1) **transposed table detection** (`_try_deterministic_transposed`) for tables where field labels are rows and data columns are records (e.g., 2 vessels per page with labels in column 0), and (2) **flat table detection** (`_try_deterministic`) for standard row-per-record tables. Alias matching uses `_normalize_for_alias_match()` which normalizes case, whitespace, parenthesis spacing, and strips double-quotes to handle OCR/formatting variation (e.g., `"Quantity(tonnes)"` matches alias `"Quantity (tonnes)"`). A **comma-suffix fallback** strips unit annotations after commas when the full string has no alias match (e.g., `"Area harvested,Th.ha."` → tries `"Area harvested"`). A **joined-form fallback** tries the space-joined form when all parts of a compound header are individually unmatched (e.g., `"Quantity / (tonnes)"` → `"Quantity (tonnes)"`). **Title-to-schema matching** (Phase 2.3) assigns the table title as a constant dimension when it matches a string-type schema column's alias — tries full-text match first, then word-boundary substring fallback (e.g., title "Winter sowing of grains and grasses" contains alias "grains and grasses"). **Blank-header text-column inference** (Phase 2.5) assigns columns with empty headers but text data to unmatched string-type schema columns when exactly one of each exists (accounting for title-matched columns). Columns where every header part (split on ` / `) matches a schema alias are mapped via string matching with zero LLM calls; columns with NO matching alias parts are silently dropped (the contract defines what data we want, not an inventory of everything in the document). **Multi-section tables** with per-section re-headers (different column counts per section) are handled by column remapping: when a section's re-header has a different layout from the global header, named columns are matched and data rows are remapped to the global column order. Section labels (plain text or bold markers) are mapped to schema columns when they match aliases (e.g., port names as `load_port` aliases). Falls back to the LLM pipeline only when no measures or groups can be resolved at all. LLM path: 2-step (parse then map) and single-shot modes, with auto page-splitting, **pre-step-1 table splitting** for large tables (`step1_max_rows=40`, prevents LLM output truncation on 65+ row tables), batched step 2, and async concurrency. Supports an optional **vision-based schema inference** mode for PDFs with garbled/concatenated headers: pass `pdf_path=` to `interpret_table()` to render pages as images and use a vision LLM to infer column structure before parsing. Includes **deterministic section boundary validation**: after the LLM parses table structure (step 1), Python code counts pipe-table data rows per section directly from the compressed text and corrects any miscounted boundaries before batching. The **`UnpivotStrategy`** enum (`SCHEMA_AGNOSTIC`, `DETERMINISTIC`, `NONE`) controls how pivoted tables are handled: `SCHEMA_AGNOSTIC` pre-unpivots for the LLM via `unpivot.py`, `DETERMINISTIC` lets the deterministic mapper handle pivots and sends the original to the LLM, `NONE` skips all pivot handling. The `unpivot` parameter on all entry points accepts both the enum and `bool` for backward compatibility (`True` → `SCHEMA_AGNOSTIC`, `False` → `NONE`). Key public functions: `interpret_table()`, `interpret_table_single_shot()`, `infer_table_schema_from_image()`, `to_records()`, `to_records_by_page()`.
- **`unpivot.py`** — Schema-agnostic deterministic pivot detection and unpivoting for pipe-tables. Detects repeating compound header groups (e.g. `"crop A / 2025"`, `"crop B / 2025"`) by splitting on ` / `, grouping by prefix, and fuzzy-matching suffix lists across groups using `rapidfuzz.fuzz.ratio()`. When ≥2 groups share ≥2 matching suffixes, transforms the wide pivoted table into long format with a `_pivot` column containing the group prefix. Non-matching groups (e.g. `"Th.ha. / Region"`) become shared columns. Runs as a pre-processing step before the LLM path only (the deterministic mapper handles pivoted tables natively via alias-based group detection, so unpivoting is applied after the deterministic attempt fails). Public functions: `unpivot_pipe_table(text, *, similarity_threshold, min_groups, pivot_column_name)`, `detect_pivot(text, *, similarity_threshold, min_groups)`.
- **`normalize.py`** — Centralized pipe-table text normalization. Single function `normalize_pipe_table(text)` that applies lossless, idempotent normalizations to pipe-table markdown: NBSP → space, smart quotes → ASCII, en/em-dash → hyphen, zero-width character removal, whitespace collapse, trailing whitespace strip. Applied automatically at entry points of `classify_tables()`, `interpret_table()`, `interpret_table_single_shot()`, `interpret_table_async()`, and `interpret_tables_async()`.

**Layer 4: Semantic**

- **`contracts.py`** — Contract helpers for loading JSON data contracts and transforming DataFrames. **Prepare helpers**: `load_contract(path)` parses a JSON contract into a typed `ContractContext` dataclass (with `OutputSpec` per output containing the `CanonicalSchema`, enrichment rules, column specs, and `SemanticColumnSpec` for columns with `concept_uris`); `resolve_year_templates(aliases, pivot_years)` resolves `{YYYY}`, `{YYYY-1}`, `{YYYY+1}` patterns in schema aliases using document-extracted years. **Semantic parsing**: `load_contract()` extracts `concept_uris`, `resolve`, and `semantic` fields from contract columns into `SemanticColumnSpec` dataclasses (stored in `OutputSpec.semantic_columns`), and sets `ContractContext.has_semantic_annotations` when any column has concept URIs. **Transform helpers**: `enrich_dataframe(df, enrichment, *, title, report_date)` adds contract-specified enrichment columns (`source: "title"/"report_date"/"constant"`); `format_dataframe(df, col_specs)` applies column-level case transformations (`lowercase`/`uppercase`/`titlecase`) and row filters (`latest`/`earliest`). Each helper is small, stateless, and composable — orchestration stays in the notebook.
- **`semantics.py`** — Semantic-aware pipeline support. Provides three capabilities without importing from `contract_semantics`: (1) `SemanticContext` dataclass — pre-resolved semantic data (aliases and valid value sets) built externally by `contract_semantics.context.build_semantic_context()`, serializable to/from JSON via `to_json()`/`from_json()`; (2) `preflight_check(data, output_spec, semantic_context)` — compares document headers against contract aliases (manual + resolved), reports coverage; (3) `validate_output(df, output_spec, semantic_context)` — checks extracted DataFrame values against known valid concept labels for columns with `validate=True`. All functions are informational — they never block extraction. Uses `_normalize_for_alias_match()` from `interpret.py` for consistent matching. Public types: `SemanticContext`, `PreFlightReport`, `PreFlightFinding`, `ValidationReport`, `ValidationFinding`.

**Layer 5: Serialization**

- **`serialize.py`** — Serialization module for exporting `interpret_table()` results to various formats. Validates records against the CanonicalSchema using Pydantic before serialization, with automatic coercion of OCR artifacts (e.g., `"1,234"` → `1234`, `"(500)"` → `-500`). Applies output formatting from `ColumnDef.format` field — supports date/time patterns (`YYYY-MM-DD`, `HH:mm`), number patterns (`#,###.##`, `+#`, `#%`), and string case transformations (`uppercase`, `camelCase`, `snake_case`, etc.). Public functions: `to_csv(result, schema, *, path=None, include_page=False)`, `to_tsv(...)`, `to_parquet(result, schema, path, *, include_page=False)`, `to_pandas(...)`, `to_polars(...)`. Optional dependencies: `pip install docpact[dataframes]` for pandas/polars, `pip install docpact[parquet]` for Parquet support.

**Layer 6: Orchestration**

- **`pipeline.py`** — Async pipeline orchestration helpers. Four composable layers: (1) `compress_and_classify_async(doc_path, categories, output_specs)` — compresses a document and classifies its tables into output categories, with DOCX/PDF branching and `asyncio.to_thread()` for CPU-bound PDF compression; (2) `interpret_output_async(data, output_spec, *, model, unpivot, report_date, pivot_years, semantic_context)` — interprets one output category end-to-end (deep-copies schema, resolves year templates, merges resolved aliases from `SemanticContext`, calls `_interpret_pages_batched_async`, enriches, formats, reorders columns); (3) `process_document_async(doc_path, cc, *, semantic_context)` — convenience layer composing both helpers with report-date resolution, plus pre-flight checks (after compress) and post-extraction validation (after enrichment) when `semantic_context` is provided, returning a `DocumentResult` dataclass; (4) `run_pipeline_async(contract_path, doc_paths, output_dir, *, semantic_context)` — top-level orchestrator that loads the contract, processes all documents concurrently, merges DataFrames across documents, and writes Parquet output. Also exports `save(results, output_dir)` for standalone merge+write. All helpers are independently useful and stateless. The `DocumentResult` dataclass holds `doc_path`, `report_date`, `compressed_by_category`, `dataframes`, `preflight_reports`, and `validation_reports`.

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
- **`xlsx_extraction.ipynb`** — XLSX extractor companion notebook. Demonstrates all extraction capabilities: basic extraction (P1), merged/compound headers (P2), multi-table detection (XH1), title detection (XH2), hidden content filtering (XH3), number format hints (XH4), visual style heuristics, markdown rendering, and a full fixture survey across all 16 synthetic documents.
- **`walkthrough_legacy.ipynb`** / **`walkthrough_docx_legacy.ipynb`** — Original format-specific notebooks (preserved for reference).

### Data Contracts (`contracts/`)

JSON files that declaratively define extraction pipelines: LLM model, table classification keywords, output schemas (columns, types, aliases, formats), enrichment rules, and unpivot strategy. The pipeline code in `pipeline.ipynb` is 100% generic — swap the contract, not the code.

Top-level contract fields:
- **`provider`** — Human-readable name of the data source.
- **`model`** — LLM model identifier (default `"openai/gpt-4o"`).
- **`unpivot`** — Optional. Controls how pivoted tables are handled. Values: `"schema_agnostic"` (pre-unpivot for LLM, default), `"deterministic"` (deterministic mapper handles pivots, LLM sees original), `"none"` (skip all pivot handling), or boolean `true`/`false` for backward compatibility.
- **`categories`**, **`outputs`**, **`report_date`** — See individual contracts for structure.

- **`ru_ag_ministry.json`** — Russian Ministry of Agriculture weekly grain reports (harvest + planting). Annotated with `concept_uris` for AGROVOC (crops, metrics) and GeoNames (regions) — see `tools/contract-semantics/`.
- **`au_shipping_stem.json`** — Australian shipping stem vessel loading records (6 providers, 1 canonical schema)
- **`acea_car_registrations.json`** — European new car registrations by market and power source

### Contract Semantics Toolkit (`tools/contract-semantics/`)

Separate Python package for ontology-grounded contract authoring and validation. Lives upstream of docpact — the contract JSON is the sole interface. docpact never imports from `contract_semantics`.

- **`models.py`** — Pydantic data models: `ConceptRef`, `ResolveConfig`, `ResolvedAlias`, `ResolutionResult`, `GeoEnrichment`, `GeoSearchResult`.
- **`agrovoc.py`** — AGROVOC SKOS adapter. Offline mode (rdflib Graph from N-Triples dump with pickle cache) and online mode (SPARQL endpoint). Resolves concept URIs to multilingual labels (`skos:prefLabel`, `skos:altLabel`) with optional `skos:narrower` traversal.
- **`geonames.py`** — GeoNames adapter. Offline mode (country extract TSV) and online mode (REST API). Three functions: `resolve_geoname()` (multilingual alternate names), `enrich_geoname()` (lat/lng/admin codes), `search_geonames()` (name search with filters).
- **`resolve.py`** — `OntologyAdapter` protocol (implemented by both adapters) and `resolve_column()` function that dispatches based on `resolve.source`, applies prefix pattern expansion (e.g. `"spring {label}"` × `"wheat"` → `"spring wheat"`), deduplicates, and compares against manual aliases.
- **`materialize.py`** — `materialize_contract()` reads an annotated contract, resolves all concept URIs, merges aliases (union/resolved_only/manual_priority), strips annotation fields, and writes: (1) materialized contract JSON (standard format, consumed by docpact unchanged) and (2) optional geo sidecar JSON mapping region names to GeoNames metadata for post-extraction GIS joins.
- **`diff.py`** — `diff_aliases()` and `diff_all()` produce human-readable reports comparing resolved vs manual aliases: coverage percentage, matched, manual-only, resolved-only with provenance.
- **`validate.py`** — SHACL validation: `records_to_graph()` converts tabular records to RDF, `validate_records()` checks against SHACL shapes via pyshacl, `generate_shapes()` auto-generates shapes from contract schemas. Optional dependency: `pyshacl`.
- **`context.py`** — Bridge builder: `build_semantic_context(contract_path, *, agrovoc, geonames, merge_strategy, cache_path)` resolves all concept URIs in a contract through ontology adapters and builds a `docpact.semantics.SemanticContext` data object. This is the **only module** that imports from both packages — it reads annotated contracts via `contract_semantics` adapters and produces a `SemanticContext` consumed by `docpact.pipeline`. The `SemanticContext` contains resolved aliases (for runtime alias enrichment) and valid value sets (for post-extraction validation), serializable to/from JSON for caching.
- **`cli.py`** — Click CLI with commands: `fetch-agrovoc`, `fetch-geonames`, `diff`, `materialize`, `validate`, `generate-shapes`, `build-context`.

Per-module documentation lives in `.md` files alongside the source: `README.md`, `MODELS.md`, `AGROVOC.md`, `GEONAMES.md`, `RESOLVE.md`, `MATERIALIZE.md`, `DIFF.md`, `VALIDATE.md`, `CLI.md`, `CONTEXT.md`.

## Key Constraints

- Python >=3.13 required
- Project uses src layout — all application code goes in `src/docpact/`
- BAML version in `generators.baml` must stay in sync with the `baml-py` dependency version in `pyproject.toml`
- The `baml_client/` directory is fully generated; changes belong in `baml_src/`
- **Always regenerate the BAML client** after editing any file in `baml_src/` by running `uv run baml-cli generate`
- **Documentation must stay in sync with code.** Both `src/docpact/` and `tools/contract-semantics/src/contract_semantics/` have per-module `.md` files documenting each script file. When modifying a script, you **must** update the corresponding `.md` file in the same commit. This includes: changes to public API signatures, new functions, renamed parameters, altered behavior, and new heuristics. The `.md` file is the authoritative reference for what a module does — if code and docs disagree, that's a bug.

## BAML Guidelines

- **Mandatory descriptions**: Every BAML `enum` value and `class` field that influences LLM behavior MUST have a `@description()` annotation. These descriptions are embedded in LLM prompts and are critical for correct classification (e.g., `TableType.TransposedTable` must explain what a transposed table looks like so the model can identify it). Missing descriptions cause misclassification bugs. **Exception for self-explanatory enums**: Descriptions are optional when the enum value name is universally unambiguous on its own (e.g., `AggregationType.Sum`, `AggregationType.Min`). If a value's meaning depends on domain context or could be confused with something else, a description is still mandatory.

- **No redundancy between descriptions and prompts**: `@description()` annotations define WHAT a field or value IS (semantic meaning). Prompt templates define HOW to use that information (operational instructions). Do not repeat identification criteria from enum descriptions inside prompt templates — `{{ ctx.output_format }}` already injects all descriptions into the prompt. Prompts should focus on decision processes, mapping strategies, and extraction rules that go beyond what the type definitions express.

- **Domain-agnostic descriptions and examples**: ALL text that reaches the LLM — `@description()` annotations, prompt templates, and inline examples — MUST use generic, domain-neutral language. Use abstract structural terms (e.g., "category labels", "entity names", "measure values", "Item X", "Period A") rather than domain-specific terminology (e.g., "port names", "vessel names", "tonnage", "Wheat", "Oct 2025"). This applies equally to enum value descriptions, class field descriptions, KEY SIGNS lists, and prompt examples. Domain-specific language biases the model toward particular document types and causes misclassification on other domains.

- **Keep descriptions and README in sync**: The `TableType` enum descriptions in `interpret.baml` must match the table in `src/docpact/README.md` under "Table Types and Mapping Strategies". When updating one, update the other.

## Testing Guidelines

- **Bug fixes must be validated against the original failing case first.** When a concrete bug triggers a fix (e.g., "spring grain columns are wrong in the June DOCX"), the very first test is to reproduce the failure end-to-end and confirm the fix resolves it. Run the actual pipeline on the actual document. Inspect the actual output. Only after the real-world case is confirmed fixed should you write unit tests for edge cases and regressions. Unit tests prove the mechanism works in isolation; they do NOT prove the bug is fixed. A normalizer that passes 34 unit tests but was never run against the document that exposed the problem is untested where it matters.

- **Rigorous, not superficial**: Don't just check that code runs or that output "looks right". Verify actual content accuracy — column names must match the source PDF exactly, data values must be in correct columns, order must be preserved.

- **Compare against source**: For compress/header refinement changes, compare LLM output against the actual PDF. Open the PDF, read the header text, and verify the output matches character-for-character.

- **Test the hard cases**: Bunge has 9 wrapped header rows with complex stacking. 2857439 has hierarchical headers. CBH has side-by-side tables. Queensland has transposed layout. These are the real tests — not whether simple cases work.

- **Verify column count and order**: Count columns in the source PDF header. Count columns in the output. They must match. Verify column order matches left-to-right reading order in the PDF.

- **No hallucination tolerance**: If the LLM produces column names that don't exist in the PDF, or merges/splits columns incorrectly, that's a failure. The goal is faithful reproduction, not creative interpretation.

- **Document expected outputs**: For key test PDFs, document what the correct output should be (column names, column count, data row count) so regressions are immediately visible.

- **Zero failing tests before commit**: Always run the full test suite (`uv run pytest tests/`) before committing or pushing to git. Every test must pass. If any test fails — whether caused by your changes or pre-existing — investigate and fix it before committing. No exceptions. Broken tests that get committed silently become invisible regressions.
