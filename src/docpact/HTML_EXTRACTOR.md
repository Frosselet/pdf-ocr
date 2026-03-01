# HTML Table Extraction

> **Module**: `html_extractor.py`
> **Public API**: `extract_tables_from_html()`, `html_to_markdown()`

Extracts structured tables from HTML content using selectolax for fast parsing. Handles `<table>` elements with colspan/rowspan, inline styles for visual information, and class names for semantic hints.

## Why

HTML tables have explicit grid structure via `<tr>`/`<td>`/`<th>` tags, but real-world HTML introduces complications:

- **Merged cells**: `colspan` and `rowspan` attributes span cells across the grid
- **Multiple header sources**: `<thead>`, `<th>` tags, background fills — all signal header rows differently
- **Visual style hints**: Inline CSS carries structural information (background colors mark headers, bold marks emphasis)
- **No JavaScript support**: Static parsing only — dynamic pages must be pre-rendered

## Pipeline

```text
HTML input (string or file path)
  |
  v
1. Load content (file read or string pass-through)
  |
  v
2. Parse with selectolax HTMLParser
  |
  v
3. Find all <table> elements
  |
  v
4. Per table: _parse_table()
   -- Find rows in <thead>, <tbody>, <tfoot>, or direct children
   -- Determine grid dimensions (accounting for colspan)
   -- Fill grid with colspan/rowspan expansion
   -- Extract cell styles (background, borders, bold, <th> tags)
   -- Analyze visual structure (header fills, <thead> presence)
  |
  v
5. Normalize grid (pad rows to equal length)
  |
  v
6. Estimate header rows (layered approach)
   -- <thead> row count (highest priority)
   -- Header fill rows (consistent background color)
   -- All-<th> first row
   -- H7 span-count analysis (fallback)
  |
  v
7. Build column names from headers
  |
  v
8. Return StructuredTable objects
```

## Internal Types

| Type | Purpose |
|------|---------|
| `HtmlCellStyle` | Per-cell: background color, border, bold, `<th>` tag flag |
| `HtmlVisualInfo` | Per-table: header fill rows, `<thead>` presence, header row count |
| `HtmlTable` | Intermediate: grid, styles, visual info, table index |

## API

### `extract_tables_from_html()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `html_input` | `str \| Path` | required | HTML string or path to .html file |
| `extract_styles` | `bool` | `True` | Extract visual style information |

Returns `list[StructuredTable]` with `source_format="html"`.

Metadata fields per table:

| Field | Type | Description |
|-------|------|-------------|
| `table_index` | `int` | Table position in document (0-based) |
| `has_thead` | `bool` | Whether the table has an explicit `<thead>` |

### `html_to_markdown()`

Convenience function for quick inspection. Extracts tables and renders as pipe-table markdown with `## Table N` headers.

## Dependencies

Requires `selectolax` (`pip install docpact[html]`).

## Limitations

- No JavaScript execution — content rendered by JS frameworks (React, Vue) must be pre-rendered with a headless browser before extraction
- Color parsing supports hex (`#RGB`, `#RRGGBB`), `rgb()`, and a subset of named colors
- Does not handle `<caption>` elements for table titles
