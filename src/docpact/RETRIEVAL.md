# Metadata Retrieval

> **Module**: `retrieval.py`
> **Public API**: `quick_scan()`, `validate_metadata()`, `apply_fallbacks()`, `search_and_extract()`
> **Public Types**: `RetrievedMetadata`, `ValidationResult`, `SearchExtractResult`

Fast, deterministic metadata extraction from PDF documents. Uses formal retrieval heuristics (RH1-RH5 from `heuristics.py`) to extract document-level and table-level metadata from specific spatial zones before LLM-based table interpretation.

## Why

Documents carry metadata beyond table data — publication dates, fiscal periods, currency scales, entity names. Extracting this metadata deterministically (via regex patterns) before invoking the LLM:
- Provides context for interpretation
- Enables validation gates (abort early if required metadata is missing)
- Reduces LLM prompt complexity

## Flow

```text
Quick Scan (regex/spatial)
  |
  v
Validation Gate
  |
  +---> All required fields found → Deep Extraction (LLM)
  |
  +---> Missing fields → Fallback (default/prompt/flag)
```

## Zone-Based Extraction

The module searches for metadata in specific document zones:

| Zone | Region | Typical Content |
|------|--------|----------------|
| `TITLE_PAGE` | First page, top 40% | Report title, date, publisher |
| `PAGE_HEADER` | Top 15% of any page | Running headers, dates |
| `PAGE_FOOTER` | Bottom 15% of any page | Page numbers, disclaimers |
| `TABLE_CAPTION` | Above table | Table title, description |
| `COLUMN_HEADER` | Within table | Column names, units |
| `TABLE_FOOTER` | Below table | Footnotes, sources |
| `ANYWHERE` | Full page | Catch-all |

Zone bounds are defined as fractions of page height and applied to `PageLayout` row y-positions from `spatial_text.py`.

## API

### `quick_scan()`

Phase 1: Fast regex-based metadata extraction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_input` | `str \| Path` | required | Path to PDF file |
| `metadata_fields` | `list[MetadataFieldDef]` | required | Fields to extract |
| `pages` | `list[int] \| None` | `None` | 0-based page indices (default: first 3) |

Searches each field's zones in priority order. Tries custom patterns first, then built-in category patterns (via `detect_temporal_patterns()`, `detect_unit_patterns()`). Returns `dict[str, RetrievedMetadata]`.

### `validate_metadata()`

Phase 2: Check that all required fields were found.

Returns `ValidationResult` with `passed` flag and `missing` field names.

### `apply_fallbacks()`

Phase 4: Apply fallback strategies for missing fields.

| Strategy | Behavior |
|----------|----------|
| `DEFAULT` | Use schema-defined default value |
| `FLAG` | Return with None value, let caller handle |
| `PROMPT` | Signal that user input is needed |
| `INFER` | Future: LLM-based inference |

### `search_and_extract()`

Main entry point composing all phases: quick scan → validate → apply fallbacks → deep extraction (compress + interpret via LLM).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | `str \| Path` | required | Path to PDF file |
| `schema` | `CanonicalSchema` | required | Schema with columns and optional metadata definitions |
| `model` | `str` | `"openai/gpt-4o"` | LLM model for table interpretation |
| `skip_validation` | `bool` | `False` | Proceed even if metadata is missing |

Returns `SearchExtractResult` with `metadata`, `tables`, `validation`, `warnings`.

## Result Types

### `RetrievedMetadata`

| Field | Type | Description |
|-------|------|-------------|
| `field_name` | `str` | Metadata field name |
| `value` | `str \| None` | Extracted value |
| `source_zone` | `SearchZone` | Zone where found |
| `confidence` | `float` | 0.0–1.0 (custom pattern: 0.9, category pattern: 0.7, not found: 0.0) |
| `pattern_matched` | `str \| None` | Regex that matched |
