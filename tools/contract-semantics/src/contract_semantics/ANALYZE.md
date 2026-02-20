# analyze.py — Document Structural Profiling

Analyzes PDF and DOCX documents to extract structural signals that feed into the recommendation engine for draft contract generation. Uses docpact's compression, heuristics, classification, and pivot detection modules read-only.

## Public Functions

### `profile_document(doc_path) → DocumentProfile`

Profile a single document (PDF or DOCX). Runs compression, then analyzes each table's structure: column names, data types, cardinalities, layout detection, section labels, temporal/unit patterns.

**Args:**
- `doc_path: str | Path` — Path to PDF or DOCX file.

**Returns:** `DocumentProfile` with per-table profiles and metadata candidates.

### `merge_profiles(profiles) → MultiDocumentProfile`

Merge multiple `DocumentProfile` instances into a consolidated multi-document profile. Groups structurally similar tables across documents and aligns columns to collect all header variants for each structural position.

**Args:**
- `profiles: list[DocumentProfile]` — List from `profile_document()`.

**Returns:** `MultiDocumentProfile` with grouped tables and aligned columns.

### `format_analysis_report(profile) → str`

Format a human-readable analysis report from a `DocumentProfile` or `MultiDocumentProfile`. Suitable for terminal output.

## Data Models

### `ColumnProfile`

Structural profile of a single table column:
- `header_text: str` — Raw header text from the document
- `inferred_type: str` — Detected type: `"string"`, `"int"`, `"float"`, `"date"`, `"enum"`
- `unique_count: int` — Number of distinct values
- `total_count: int` — Total non-empty values
- `sample_values: list[str]` — Up to 10 sample values (sorted unique)
- `unit_annotations: list[str]` — Units detected in header (e.g. `"tonnes"`, `"Th.ha."`)
- `year_detected: bool` — True if header is a 4-digit year (e.g. `"2025"`)

### `TableProfile`

Structural profile of a single table:
- `source_document: str` — Document path
- `table_index: int` — Position in document
- `title: str | None` — Table heading/title if detected
- `layout: str` — `"flat"`, `"transposed"`, or `"pivoted"`
- `column_profiles: list[ColumnProfile]` — Per-column analysis
- `row_count, col_count: int` — Dimensions
- `section_labels: list[str]` — Bold markers or text dividers within/above table
- `pivot_groups: list[str]` — Pivot group prefixes (for pivoted tables)
- `header_tokens: set[str]` — Lowercased tokens from headers (for similarity)

### `MetadataCandidate`

Potential metadata value found in non-table text:
- `text: str` — Full matched text
- `pattern_name: str` — Pattern type (e.g. `"as_of_date"`, `"period_end"`)
- `captured_value: str` — Extracted value

### `DocumentProfile`

Complete structural profile of a single document:
- `doc_path, doc_format` — Source info
- `tables: list[TableProfile]` — All detected tables
- `temporal_candidates: list[MetadataCandidate]` — Dates, periods, fiscal years
- `unit_candidates: list[MetadataCandidate]` — Currencies, units, scale
- `headings: list[str]` — Document headings (from `##` markers)

### `AlignedColumn`

A column aligned across multiple documents:
- `canonical_header: str` — Most common header text
- `all_headers: list[str]` — Every variant seen across documents
- `sources: list[str]` — Which documents contributed
- `inferred_type, unit_annotations, year_detected` — Merged from sources

### `TableGroup`

A group of structurally similar tables across documents:
- `tables: list[TableProfile]` — Member tables
- `aligned_columns: list[AlignedColumn]` — Column alignment
- `common_tokens: set[str]` — Shared header tokens
- `all_section_labels: list[str]` — Union of section labels
- `layout: str` — Dominant layout type

### `MultiDocumentProfile`

Merged structural profile across multiple documents:
- `doc_paths: list[str]` — All analyzed documents
- `document_profiles: list[DocumentProfile]` — Individual profiles
- `table_groups: list[TableGroup]` — Grouped tables
- `all_temporal_candidates, all_unit_candidates` — Merged metadata

## How It Works

1. **Compression** — PDF: `compress_spatial_text_structured()` → `StructuredTable` objects. DOCX: `compress_docx_tables()` → pipe-table markdown.
2. **Per-table profiling** — For each table: extract column names, detect types via `detect_column_types()`, detect layout via `detect_pivot()` and transposed heuristics, extract section labels and units.
3. **Metadata scanning** — Non-table text is scanned for temporal patterns (`detect_temporal_patterns()`) and unit patterns (`detect_unit_patterns()`).
4. **Multi-document merging** — Tables are grouped by structural similarity (50% column-count ratio + 50% Jaccard token similarity). Columns within groups are aligned by position (same column count) or token overlap (different counts).

## Dependencies

Imports from `docpact` (read-only):
- `docpact.compress` — `compress_spatial_text`, `compress_spatial_text_structured`
- `docpact.docx_extractor` — `compress_docx_tables`
- `docpact.heuristics` — `CellType`, `detect_cell_type`, `detect_column_types`, `detect_temporal_patterns`, `detect_unit_patterns`
- `docpact.classify` — `_parse_pipe_header`, `_tokenize_header_text`, `_compute_similarity`
- `docpact.unpivot` — `detect_pivot`
