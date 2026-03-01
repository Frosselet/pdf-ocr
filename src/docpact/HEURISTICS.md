# Shared Semantic Heuristics

> **Module**: `heuristics.py`
> **Public API**: `detect_cell_type()`, `detect_column_types()`, `is_header_type_pattern()`, `estimate_header_rows()`, `detect_temporal_patterns()`, `detect_unit_patterns()`, `detect_footnote_markers()`
> **Public Types**: `CellType`, `StructuredTable`, `TableMetadata`, `MetadataCategory`, `SearchZone`, `FallbackStrategy`, `MetadataFieldDef`

Format-agnostic heuristics for table structure detection. These encode universal human inference patterns applicable across all document formats (PDF, XLSX, DOCX, PPTX, HTML). They operate on grid-level data (list of rows, each row being list of cells) rather than format-specific geometry.

## Why

Every extractor needs to answer the same structural questions: where do headers end and data begin? What type of data is in each column? Where are dates, footnotes, and units? Rather than reimplementing these decisions per format, this module provides shared heuristics that all extractors delegate to.

Geometry heuristics (y-clustering, x-position alignment) remain in format-specific extractors since they depend on spatial coordinates.

## Heuristics

### TH1: Cell Type Detection (`detect_cell_type`)

Classifies a single cell as `DATE`, `NUMBER`, or `STRING` using pattern matching:

- **Dates**: ISO formats (`YYYY-MM-DD`), slashed (`D/M/Y`), time (`HH:MM`), combined datetime, month-name patterns
- **Numbers**: Digits with optional separators (`,`, `.`), leading signs/currency (`$`, `€`, `(`), trailing `%` or `)`
- **Strings**: Everything else (default fallback)

`ENUM` detection requires column-level analysis (repeated values) and is handled by `detect_column_types()`.

### TH2: Header Type Pattern (`is_header_type_pattern`)

A row is header-like if all non-empty cells are `STRING` type — no dates or numbers. Headers are labels, not data.

### TH3: Column Type Consistency (`detect_column_types`)

Determines the predominant type per column from data rows. Includes `ENUM` detection: columns with few distinct values (≤10% of rows or ≤5 unique) that repeat are classified as `ENUM`.

### H7: Header Row Estimation (`estimate_header_rows`)

Bottom-up span-count analysis:

1. Count non-empty cells per row (the "span count").
2. In the bottom 2/3 of the table, find the 3 most common span counts (must appear ≥2 times and have count ≥2).
3. Scan from the top: the first row whose span count matches a data pattern (or exceeds the max) marks the start of data.
4. All rows above are headers.

Also provides `estimate_header_rows_from_types()` — a secondary method using TH2 type-pattern analysis (consecutive all-string rows from the top).

### RH1: Temporal Pattern Detection (`detect_temporal_patterns`)

Scans text for date and time patterns:
- "As of" dates, period end dates
- Quarter patterns (Q1 FY2025, 1st Quarter 2025)
- Fiscal year patterns (FY2025)
- Year-month patterns (January 2025)
- Date range patterns (01/01/2025 – 12/31/2025)

Returns `(matched_text, pattern_name, captured_value)` tuples.

### RH4: Unit/Currency Pattern Detection (`detect_unit_patterns`)

Scans text for:
- Scale indicators (millions, thousands, 000s, MM)
- Currency codes (USD, EUR, GBP, etc.)
- Currency symbols ($, €, £, ¥)
- Percentage indicators
- Unit names (metric tons, kg, shares)

### RH5: Footnote Marker Detection (`detect_footnote_markers`)

Finds superscript-style markers: Unicode superscripts (¹²³), bracketed numbers ([1], (1)), and symbol markers (*, †, ‡, §, ¶, #).

## Shared Types

### `StructuredTable`

Common output format across all extractors. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `column_names` | `list[str]` | Final flattened header names |
| `data` | `list[list[str]]` | Data rows (each row length == column count) |
| `source_format` | `str` | `"pdf"`, `"xlsx"`, `"docx"`, `"pptx"`, `"html"` |
| `metadata` | `dict \| None` | Format-specific metadata |

### `TableMetadata`

| Field | Type | Description |
|-------|------|-------------|
| `page_number` | `int` | Page/sheet/slide index |
| `table_index` | `int` | For side-by-side tables (0, 1, ...) |
| `section_label` | `str \| None` | Detected section label above table |
| `sheet_name` | `str \| None` | XLSX sheet name |
| `slide_number` | `int \| None` | PPTX slide number |

### Metadata Retrieval Types

Used by `retrieval.py` for search & extract:

- `MetadataCategory` — Enum: TEMPORAL, ENTITY, TABLE_IDENTITY, TABLE_CONTEXT, FOOTNOTE
- `SearchZone` — Enum: TITLE_PAGE, PAGE_HEADER, PAGE_FOOTER, TABLE_CAPTION, COLUMN_HEADER, TABLE_FOOTER, ANYWHERE
- `FallbackStrategy` — Enum: INFER, DEFAULT, PROMPT, FLAG
- `MetadataFieldDef` — Dataclass defining a metadata field to extract (name, category, required, zones, patterns, fallback, default)

## Grid Utilities

| Function | Description |
|----------|-------------|
| `normalize_grid(grid)` | Pad all rows to the same column count |
| `split_headers_from_data(grid, header_count)` | Split grid into header and data portions |
| `build_column_names_from_headers(header_rows)` | Stack header rows with ` / ` separator, deduplicate |
