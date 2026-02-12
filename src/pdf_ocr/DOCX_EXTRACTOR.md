# DOCX Table Extraction

> **Module**: `docx_extractor.py`
> **Public API**: `extract_tables_from_docx()`, `extract_tables_from_word()`, `compress_docx_tables()`, `classify_docx_tables()`, `docx_to_markdown()`

Extracts structured tables from Word documents (.docx, .doc), handles hierarchical merged-cell headers, and produces compressed pipe-table markdown ready for `interpret_table()`.

## Why a Dedicated Extractor?

DOCX tables have explicit grid structure (unlike PDFs where columns are inferred from spatial positions), but python-docx introduces complications:

- **Duplicate `_tc` references**: Horizontally merged cells (gridSpan > 1) appear multiple times in `row.cells`, all pointing to the same XML element
- **Hierarchical headers**: Metric labels span 2+ columns (e.g., "Area harvested" over "2025" / "2024") using gridSpan and vMerge
- **No page boundaries**: Unlike PDFs, DOCX has no native page concept — all tables are in document order

## Pipeline

```text
DOCX file
  |
  v
1. Extract raw grids (_build_grid_from_table)
   -- _tc deduplication for merged cells
   -- gridSpan for horizontal merges
   -- vMerge for vertical merges
  |
  v
2. Detect header rows (_detect_header_rows_from_merges)
   -- scan for merge indicators in first N rows
  |
  v
3. Detect title row
   -- single non-empty cell in row 0 = title
  |
  v
4. Build compound headers (_build_compound_headers)
   -- stack header rows with " / " separator
   -- forward-fill for spanning cells
  |
  v
5. Render pipe-table markdown (compress_docx_table)
   -- strip trailing empty columns
   -- prepend section title
  |
  v
6. interpret_table() --> serialize  (existing, no changes)
```

---

## Heuristics

### DH1: _tc Deduplication (`_build_grid_from_table`)

**Problem**: python-docx returns duplicate `_tc` references for horizontally merged cells. A cell with `gridSpan=2` appears twice in `row.cells`, both pointing to the same XML element. Without deduplication, headers shift right:

```text
WRONG:  Region | Area | Area harvested | (empty) | Area harvested | (empty) | collected | (empty)
                                                   ^^^ DUPLICATED
CORRECT: Region | Area | Area harvested | (empty) | collected | (empty) | Yield | (empty)
```

**Solution**: Track `id(cell._tc)` in a `seen_tcs` set per row. Skip cells whose `_tc` was already processed:

```python
seen_tcs_row: set[int] = set()
for cell in row.cells:
    tc_id = id(cell._tc)
    if tc_id in seen_tcs_row:
        continue
    seen_tcs_row.add(tc_id)
    # ... process cell
```

Applied in both the column-counting pass (grid dimensions) and the cell-filling pass (grid content).

---

### DH2: Merge-Based Header Detection (`_detect_header_rows_from_merges`)

**Problem**: The generic `estimate_header_rows()` from `heuristics.py` uses bottom-up span-count analysis, which works for PDF tables but misses DOCX merge signals. A title row like "WHEAT" (no merges) followed by metric rows (with merges) needs special handling.

**Solution**: Scan the first `max_scan` rows (default 10) for XML merge indicators:

- `gridSpan > 1` — horizontal spanning
- `vMerge` (any value) — vertical merge start or continuation

Find the **last** row with merge indicators. All rows from 0 up to and including that row are headers.

```text
Row 0: "WHEAT"                     → no merges (title)
Row 1: "Area harvested" (span=2)   → merge! last_merge_row = 1
Row 2: "2025" / "2024" (vMerge)    → merge! last_merge_row = 2
Row 3: "Russian Federation" / data → no merges → stop
                                    → header_count = 3
```

**Why scan forward instead of stopping at first non-merge?** Title rows (like crop names) often have no merge indicators but sit above merge-heavy metric/year rows. Stopping at the first non-merge row would produce `header_count = 0` (capped to 1), missing the actual structural headers below.

Returns at least 1 if the table has rows.

---

### DH3: Title Row Detection (`compress_docx_tables`)

**Problem**: Row 0 sometimes contains only the table title (e.g., "WHEAT", "Grains and pulses") spread across one cell, not actual column headers.

**Solution**: If row 0 has exactly one non-empty cell and `header_count > 1`, extract it as the title and exclude it from compound header construction:

```python
non_empty = [c for c in grid[0] if c.strip()]
if len(non_empty) == 1 and header_count > 1:
    title = non_empty[0]
    grid = grid[1:]
    header_count -= 1
```

The title is rendered as `## WHEAT` above the pipe table and included in metadata.

---

### DH4: Compound Header Construction (`_build_compound_headers`)

**Problem**: Hierarchical headers span multiple rows. A metric label ("Area harvested") sits above year labels ("2025", "2024"). These need to be combined into single column names.

**Solution**: For each column position, join non-empty values from header rows with " / ":

```text
Header row 1: ["Region", "Area", "Area harvested", "",    "collected, TMT", "",    "Yield", ""   ]
Header row 2: ["Region", "Area", "2025",           "2024", "2025",           "2024", "2025",  "2024"]
                                  ^^^^              ^^^^
                                  forward-fill      forward-fill
                                  from "Area..."    from "Area..."

Result:        ["Region", "Area", "Area harvested / 2025", "Area harvested / 2024",
                "collected, TMT / 2025", "collected, TMT / 2024", "Yield / 2025", "Yield / 2024"]
```

**Forward-fill**: Empty cells in a header row (positions 3, 5, 7 above) inherit the preceding non-empty cell's text. This propagates the spanning parent's label to its child columns.

**Deduplication**: If the same value appears in multiple header rows for the same column (e.g., "Region" in both rows), it appears only once in the compound name.

---

### DH5: Table Classification (`classify_docx_tables`)

**Problem**: A single DOCX may contain many tables of different types. Processing requires knowing which tables to target.

**Solution**: The caller provides a `categories` dict mapping category names to keyword lists. Header text is collected from all header rows (after title extraction), lowercased, and scanned for keyword presence. The first matching category wins; unmatched tables get `"other"`.

```python
categories = {
    "harvest": ["area harvested", "yield", "collected"],
    "export": ["export", "shipment", "fob"],
}
classes = classify_docx_tables("report.docx", categories)
```

---

## API

### Raw Extraction

```python
from pdf_ocr import extract_tables_from_docx, extract_tables_from_word

# DOCX only
tables = extract_tables_from_docx("report.docx")

# Auto-detect DOCX/DOC
tables = extract_tables_from_word("report.doc")

for table in tables:
    print(f"Columns: {table.column_names}")
    print(f"Rows: {len(table.data)}")
```

### Compressed Markdown (for `interpret_table()`)

```python
from pdf_ocr import compress_docx_tables, classify_docx_tables

# Define your own categories
categories = {
    "harvest": ["area harvested", "yield", "collected"],
    "export": ["export", "shipment", "fob"],
}

# Classify tables
classes = classify_docx_tables("report.docx", categories)
harvest_idx = [c["index"] for c in classes if c["category"] == "harvest"]

# Compress selected tables to pipe-table markdown
results = compress_docx_tables("report.docx", table_indices=harvest_idx)

for markdown, meta in results:
    print(f"Table {meta['table_index']}: {meta['title']}")
    print(markdown[:200])
```

### Full Pipeline (DOCX to structured records)

```python
from pdf_ocr import (
    compress_docx_tables,
    classify_docx_tables,
    extract_pivot_values,
    interpret_table,
    CanonicalSchema,
    to_records,
)

# 1. Classify with your own categories
categories = {
    "harvest": ["area harvested", "yield", "collected"],
    "export": ["export", "shipment", "fob"],
}
classes = classify_docx_tables("harvest_report.docx", categories)
harvest_idx = [c["index"] for c in classes if c["category"] == "harvest"]

# 2. Compress to markdown
results = compress_docx_tables("harvest_report.docx", table_indices=harvest_idx)

# 3. Extract available years from headers (e.g. ["2023", "2024", "2025"])
#    and pick which ones to include in the unpivot
first_md, _ = results[0]
all_years = extract_pivot_values(first_md)  # e.g. ["2024", "2025"]
year_aliases = all_years[-2:]               # last 2 years only

# 4. Define schema with dynamic year aliases
schema = CanonicalSchema.from_dict({
    "description": "Harvest progress by region",
    "columns": [
        {"name": "region", "type": "string", "aliases": ["Region"]},
        {"name": "metric", "type": "string", "aliases": ["Area harvested", "collected", "Yield"]},
        {"name": "year", "type": "int", "aliases": year_aliases},
        {"name": "value", "type": "float", "aliases": []},
    ],
})

# 5. Interpret each table
all_records = []
for markdown, meta in results:
    result = interpret_table(markdown, schema, model="openai/gpt-4o")
    records = to_records(result)
    for rec in records:
        rec["crop"] = meta["title"] or "Unknown"
    all_records.extend(records)
```

**Why dynamic year aliases?** The compound headers contain year values like "Area harvested / 2025" and "Area harvested / 2024". These years change every release. `extract_pivot_values()` reads them from the actual headers, and the `[-2:]` slice selects only the last 2 years. To get all years, use `all_years` directly. To get only the latest, use `all_years[-1:]`.

---

## Parameters

### `extract_tables_from_docx()`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `docx_path` | `str \| Path` | required | Path to .docx file |
| `extract_styles` | `bool` | `True` | Extract visual style info (fill, bold, font size) |

Returns `list[StructuredTable]`.

### `compress_docx_tables()`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `docx_path` | `str \| Path` | required | Path to .docx file |
| `table_indices` | `list[int] \| None` | `None` | Only process these table indices |

Returns `list[tuple[str, dict]]` — `(markdown, metadata)` per table.

Metadata fields:

| Field | Type | Description |
| --- | --- | --- |
| `table_index` | `int` | Table position in document |
| `title` | `str \| None` | Extracted title (e.g., "WHEAT") |
| `row_count` | `int` | Number of data rows (excluding headers) |
| `col_count` | `int` | Number of columns |

### `extract_pivot_values()`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `markdown` | `str` | required | Compressed pipe-table markdown |

Returns `list[str]` — sorted unique years found in the header row (e.g., `["2024", "2025"]`).

### `classify_docx_tables()`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `docx_path` | `str \| Path` | required | Path to .docx file |
| `categories` | `dict[str, list[str]]` | required | Category name → keyword list. First match wins; unmatched tables get `"other"` |

Returns `list[dict]` with fields:

| Field | Type | Description |
| --- | --- | --- |
| `index` | `int` | Table position in document |
| `category` | `str` | One of the caller's category names, or `"other"` |
| `title` | `str \| None` | Extracted title |
| `rows` | `int` | Data row count |
| `cols` | `int` | Column count |

---

## Merge Handling

DOCX tables use two merge mechanisms:

### Horizontal Merge (gridSpan)

A cell with `gridSpan=2` spans two grid columns. The cell's text goes into the first position; the second is left empty.

```xml
<w:tc>
  <w:tcPr><w:gridSpan w:val="2"/></w:tcPr>
  <w:p><w:r><w:t>Area harvested,Th.ha.</w:t></w:r></w:p>
</w:tc>
```

Grid result: `["Area harvested,Th.ha.", ""]` at columns 2-3.

### Vertical Merge (vMerge)

A cell with `vMerge val="restart"` starts a vertical span. Subsequent rows with `vMerge` (no val or val="continue") are continuations — they get empty strings in the grid.

```xml
<!-- Row 1: merge start -->
<w:vMerge w:val="restart"/>  <!-- "Region" → grid[1][0] = "Region" -->

<!-- Row 2: merge continuation -->
<w:vMerge/>                  <!-- "Region" → grid[2][0] = ""  (continuation) -->
```

---

## Visual Style Extraction

When `extract_styles=True`, the extractor captures per-cell visual information:

| Attribute | Source | Usage |
| --- | --- | --- |
| Shading color | `tcPr/shd/@fill` | Header fill detection |
| Bold | `run.bold` | Header emphasis |
| Italic | `run.italic` | Metadata detection |
| Font size | `run.font.size` | Hierarchy detection |

Visual info feeds into `_analyze_visual_structure()` to identify header rows by consistent background fill.

---

## Testing

82 regression tests in `tests/test_docx_extractor.py` cover:

| Test Group | Count | Verifies |
| --- | --- | --- |
| Shape regression | 6 | Table count, column count, row count per file |
| Content regression | 7 | Specific column names and cell values |
| Grid dimensions | 2 | Raw grid rows x cols after _tc dedup |
| _tc dedup correctness | 4 | Merged cells produce correct positions |
| Header detection | 3 | Merge-based header row count |
| Compound headers | 6 | Forward-fill, stacking, deduplication |
| Compress rendering | 4 | Pipe-table output format |
| Compress integration | 5 | End-to-end DOCX to markdown |
| Classification | 11 | Category counts and titles per file |
| Pivot values | 6 | Year extraction from headers |
| Smoke tests | 3 | All files extract/classify/compress without error |
| Synthetic: flat/single | 3 | Flat tables, single-column edge case |
| Synthetic: hmerge | 3 | Compound headers, grid dedup, full-width spans |
| Synthetic: title row | 3 | Title extraction and exclusion from headers |
| Synthetic: vmerge | 2 | Vertical merge header detection and output |
| Synthetic: deep hierarchy | 3 | 3-level headers, header count, pivot values |
| Synthetic: classification | 4 | Multi-table routing, case-insensitive matching |
| Synthetic: multi-table | 2 | Table index filtering, mixed flat/merged |
| Synthetic: edge cases | 5 | Wide tables, unicode, single column, smoke test |

```bash
uv run pytest tests/test_docx_extractor.py -v
```
