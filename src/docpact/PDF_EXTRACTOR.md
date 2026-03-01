# PDF Table Extraction

> **Module**: `pdf_extractor.py`
> **Public API**: `extract_tables_from_pdf()`, `pdf_to_markdown()`

Thin facade aligning the PDF extraction API with the naming convention of other format extractors (`extract_tables_from_xlsx()`, `extract_tables_from_docx()`, etc.).

## Why

PDF extraction logic lives in `spatial_text.py` (raw renderer) and `compress.py` (compression + structured extraction). The public function `compress_spatial_text_structured()` already returns `list[StructuredTable]` — the same type as other extractors — but doesn't follow the `extract_tables_from_<format>()` naming convention. This facade provides the canonical name without duplicating logic.

## API

### `extract_tables_from_pdf()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | `str \| bytes \| Path` | required | Path to PDF file or raw PDF bytes |
| `pages` | `list[int] \| None` | `None` | 0-based page indices (None = all pages) |
| `cluster_threshold` | `float` | `2.0` | Max y-distance (points) for row merging |
| `merge_multi_row` | `bool` | `True` | Merge multi-row records (shipping stems) |
| `min_table_rows` | `int` | `3` | Minimum data rows per table |
| `extract_visual` | `bool` | `True` | Extract visual elements for header heuristics |

Returns `list[StructuredTable]`. Pure delegation to `compress_spatial_text_structured()`.

### `pdf_to_markdown()`

Convenience function for quick inspection. Extracts tables and renders as pipe-table markdown with `## Page N` headers. Mirrors `xlsx_to_markdown()` and `pptx_to_markdown()`.

Same parameters as `extract_tables_from_pdf()` except `extract_visual` (always False for markdown rendering).

## Relationship to Other Modules

```text
pdf_extractor.py  ──delegates──>  compress.py  ──uses──>  spatial_text.py
     (facade)                    (structured)             (raw renderer)
```

No new logic lives here. For implementation details, see `COMPRESS.md` and `SPATIAL_TEXT.md`.
