# pdf_ocr — Spatial PDF Text Extraction

## Problem

Standard PDF text extraction (PyMuPDF's `get_text()`, `pymupdf4llm`) returns text in stream order — the order objects were written into the PDF file. This rarely matches visual layout. Columns get interleaved, table cells merge into a single line, and scattered labels lose their spatial meaning.

This is especially problematic for tabular documents like shipping stems, loading statements, and financial reports where the relationship between values depends entirely on their position on the page.

## Approach: Spatial Grid Rendering

Instead of reordering text heuristically, we project every text span onto a **monospace character grid** that mirrors the physical page. The result is plain text where columns, tables, and scattered labels appear exactly where they sit in the PDF.

### Pipeline

```
PDF page
  │
  ▼
1. Extract spans ─── page.get_text("dict") ──► list of {text, origin(x,y), bbox}
  │
  ▼
2. Compute cell width ─── median(bbox_width / len(text)) ──► adaptive char width in points
  │
  ▼
3. Cluster y-coords ─── greedy merge within 2pt gap ──► row groups
  │
  ▼
4. Map to grid ─── col = round((x - x_min) / cell_w) ──► (row, col, text) triples
  │
  ▼
5. Render ─── character buffer per row, last-writer-wins ──► plain text
```

### Step 1: Span Extraction

We use `page.get_text("dict")` which returns every text span with:
- **`text`** — the string content
- **`origin`** — the (x, y) baseline coordinate in PDF points (72pt = 1 inch)
- **`bbox`** — the bounding box `[x0, y0, x1, y1]`

Spans are the finest granularity that preserves font/style runs. We strip whitespace-only spans.

### Step 2: Dynamic Cell Width

Fixed character widths fail because PDFs use arbitrary font sizes. Instead, we compute the **median character width** across the page:

```
cell_w = median( (bbox.x1 - bbox.x0) / len(text) )  for all spans with len >= 2
```

- Using the **median** (not mean) makes this robust to outliers like titles in large fonts or footnotes in small fonts.
- We require spans with **2+ characters** to avoid division noise from single-char spans (e.g., bullets, pipe characters).
- Fallback to 6.0pt if a page has only single-char spans.

This single value defines the grid resolution for the entire page.

### Step 3: Row Clustering

PDF text on the "same line" often has slightly different y-coordinates due to font metrics, subscripts, or rendering jitter. We cluster y-values into rows:

1. Sort all unique y-values
2. Greedily merge consecutive values within a **2.0pt threshold**

The 2pt threshold handles sub-point jitter without falsely merging adjacent rows (typical line spacing is 10–14pt).

### Step 4: Grid Mapping

Each span maps to a grid position:

```
col = round((span.x - page_x_min) / cell_w)
row = cluster index from step 3
```

Subtracting `page_x_min` anchors the leftmost text at column 0, avoiding wasted leading whitespace.

### Step 5: Rendering

For each row, we allocate a character buffer and write each span starting at its computed column:

```
buf = [' '] * max_width
for col, text in row_spans:
    for i, ch in enumerate(text):
        buf[col + i] = ch
```

**Last-writer-wins** for overlaps — if two spans occupy the same grid cell, the later one in document order takes precedence. This is the simplest conflict resolution and works well in practice since true overlaps are rare.

Trailing whitespace is stripped from each line.

## API

```python
from pdf_ocr import pdf_to_spatial_text

text = pdf_to_spatial_text("document.pdf")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pdf_path` | `str` | required | Path to the PDF file |
| `pages` | `list[int] \| None` | `None` | 0-based page indices to render; `None` = all |
| `cluster_threshold` | `float` | `2.0` | Max y-distance (points) to merge into one row |
| `page_separator` | `str` | `"\f"` | String inserted between rendered pages |

### Return Value

A single string with all pages joined by `page_separator`. Each page is a block of lines where text spans appear at their spatial grid positions.

## Compressed Spatial Text

The spatial grid preserves layout perfectly but wastes tokens on whitespace padding — a 62,000-character shipping stem is mostly spaces. `compress_spatial_text()` works from the same raw span data but produces a **token-efficient structured representation** that LLMs understand natively.

### Pipeline

```
PageLayout (shared with spatial renderer)
  │
  ▼
1. Span splitting ──► split merged PDF spans using column boundaries from other rows
  │
  ▼
2. Region detection ──► classify row groups as table / text / heading / kv_pairs / scattered
  │
  ▼
3. LLM table fallback ──► if no tables detected and refine_headers=True, ask GPT-4o-mini to find tables
  │
  ▼
4. LLM header refinement ──► for each detected table (refine_headers=True), scan upward for preceding header rows then refine via GPT-4o-mini
  │
  ▼
5. Multi-row merge ──► detect repeating row patterns (e.g., 3-row shipping records) and collapse
  │
  ▼
6. Region rendering ──► markdown tables (with LLM-refined headers when available), flowing paragraphs, key: value lines
  │
  ▼
7. Assembly ──► join regions with blank lines, pages with separator
```

### Span Splitting

PDF creators sometimes encode adjacent cell values as a single text span. For example, a loading statement might store `"7:00:00 PM BUNGE"` as one span covering both the time and exporter columns, or `"33020 WHEAT"` merging a numeric code with a commodity name. The spatial grid renders these faithfully, but the compressor would treat each merged span as a single table cell — breaking pipe table structure.

`_split_merged_spans()` fixes this by using column boundaries observed across **all rows** to split merged spans:

1. **Collect column positions** from every row in the `PageLayout` — each `(col, text)` entry registers its `col` as a known boundary.
2. **For each span**, find positions from *other* rows that fall inside it (with a minimum gap of 5 characters to avoid false splits on narrow columns).
3. **Split at those boundaries** using character-position math: if header rows have separate spans at columns 219 (`EXPORTER`) and 245 (`COMMODITY`), a data span starting at column 208 that covers both positions is split into `"7:00:00 PM"` (col 208) + `"BUNGE"` (col 219).

This approach generalises to any layout — it doesn't need to know which rows are headers. Any row that has finer-grained column boundaries than another provides the split points. The result is a corrected `PageLayout` where each span contains exactly one cell value, enabling clean pipe table rendering downstream.

### Region Detection

Each row is characterized by its span count and column positions. Contiguous rows are classified:

| Region | Detection rule | Rendered as |
|---|---|---|
| **table** | 3+ consecutive rows sharing 2+ aligned column anchors | Markdown pipe table |
| **text** | Consecutive single-span rows, left-aligned | Flowing paragraph |
| **heading** | Single short span, isolated | Plain text line |
| **kv_pairs** | Consecutive 2-span rows (label + value) | `key: value` lines |
| **scattered** | Fallback | Tab-separated spans |

Table detection uses a **column anchor pool**: as rows are scanned, all column positions are accumulated. A new row joins the table if it shares 2+ anchors with the pool. This handles alternating row patterns (e.g., 7-column data rows interleaved with 5-column date/time rows).

**Section label promotion**: Single-span rows inside a table run are checked for content. Numeric values (e.g., `337,000`, `593,810`) are kept as aggregation/totals rows within the table. Non-numeric single-span rows (e.g., `KWINANA`, `ALBANY`) flush the current table run — they become standalone rows classified as headings by the remaining-row pipeline. This prevents section labels from being absorbed into table runs as empty pipe-cell rows, giving the LLM clear section boundaries between table blocks.

### Multi-Row Record Merging

Some PDFs split a single logical record across multiple rows. For example, shipping stems from `2857439.pdf` use 3 rows per ship:

```
Row 1: dates       →  10/07/2025   10/07/2025   06/08/2025   06/08/2025   09/08/2025
Row 2: data        →  Newcastle  ADAGIO  NT25084  ARROW COMMODITIES  Wheat  26,914  Completed
Row 3: times       →  11:45 AM   2:25 PM   8:06 AM   8:06 AM   11:15 PM
```

The compressor detects repeating span-count patterns (here `[5, 7, 5]`) and merges sub-rows, producing cells like `10/07/2025 11:45 AM`. A header offset search skips irregular header rows before finding the repeating body pattern.

### Alignment-Aware Header Matching

PDF tables often have **inconsistent text alignment** between headers and data:
- Headers may be left-aligned or centered, while data is right-aligned (numbers)
- Headers may be positioned slightly before or after the data column they describe
- The same logical column can have span positions 5-10 characters apart between headers and data

Simple start-position matching fails in these cases. For example, a "QUANTITY" header at position 233 won't match a data column at position 239 if the tolerance is only 4.

`_build_stacked_headers()` solves this using **bounding-box overlap** instead of start-position matching:

1. **Compute column bounds** from data rows — for each column, find the `(min_start, max_end)` across all data spans. This captures the full horizontal extent regardless of alignment.

2. **Match headers by overlap** — a header span `[h_start, h_end)` matches a column if it overlaps with `[d_min - margin, d_max)` where `margin=5` captures headers positioned just before the data.

3. **Best-overlap wins** — when a header overlaps multiple columns (common with stacked headers near column boundaries), it's assigned to the column with maximum overlap.

4. **Deduplicate** — consecutive identical words are removed from the final header (handles spans that appear in multiple overlapping header rows).

**Example: Bunge loading statement**

The PDF has 22 columns with 9 stacked header rows. Without alignment-aware matching:

| Column | Header Position | Data Position | Start-based | Overlap-based |
|--------|-----------------|---------------|-------------|---------------|
| 9 | 121-125 "FROM" | 125-135 | ✗ (miss by 0) | ✓ overlap 9 |
| 17 | 233-241 "QUANTITY" | 239-244 | ✗ (miss by 6) | ✓ overlap 5 |
| 20 | 274-286 "DATE LOADING" | 279-289 | ✗ (miss by 5) | ✓ overlap 10 |

Result: All 22 columns now have correct headers.

**Why this matters**: Real-world PDFs are often created in Excel or other tools where humans adjust column widths, merge cells, and use different alignments without consistency. Building heuristics for the chaotic case (mixed alignment, manual formatting) rather than the optimistic case (automated, consistent) produces robust extraction.

### Preceding Header Row Detection

Header rows in complex documents (e.g., loading statements with 9 wrapped header rows) often have different span counts and alignments than data rows, so `_classify_regions` puts them in kv_pairs/scattered/heading regions rather than the table.

`_find_preceding_header_rows` scans upward from the table's first row, collecting rows whose span positions align (within `render_tolerance`) with data column positions. Scanning stops at the first non-matching row, a gap > 2 rows, or a row belonging to another table region.

The preceding rows are **suppressed** from their original non-table region rendering to avoid ambiguous duplication. Non-table regions that lose all their rows to this filtering are dropped entirely; regions that only partially overlap keep their remaining rows.

### LLM Table Detection Fallback

When `refine_headers=True` (the default) and heuristics find no table on a page (e.g., the layout doesn't meet the strict 2+ spans / 2+ shared anchors / 3+ rows requirements), the full page spatial text is sent to GPT-4o-mini for table detection. If a table is found, it's rendered as a pipe table and inserted into the output alongside heuristic-rendered non-table regions.

The LLM call uses `CustomGPT4oMini` defined in `baml_src/compress.baml`. Failures are handled gracefully — any LLM error (network, missing API key, timeout, malformed response) silently falls back to heuristic output.

Setting `refine_headers=False` gives identical behavior to the pure-heuristic path: no LLM calls, no API key needed.

### Transposed Table Detection

Some PDFs use a **transposed layout** where field names are in the first column and data records are in subsequent columns — the transpose of a normal table. This is common in narrow page layouts (A4 portrait) where putting 20+ columns horizontally would require unreadable wrapped headers.

```
Normal table:              Transposed table:
┌──────┬──────┬──────┐     ┌───────────────────┬───────────┬───────────┐
│ Name │ Date │ Qty  │     │ Name              │ Alice     │ Bob       │
├──────┼──────┼──────┤     │ Date              │ 2025-01-15│ 2025-01-16│
│ Alice│ Jan  │ 100  │     │ Quantity          │ 100       │ 200       │
│ Bob  │ Feb  │ 200  │     └───────────────────┴───────────┴───────────┘
└──────┴──────┴──────┘
```

`_is_transposed_table()` detects this layout using structural heuristics:

- **Few columns** (≤5) — one label column + 1-4 record columns
- **Low span variance** (<2.0) — consistent row structure
- **Stable first column** (≥80%) — field labels always present in column 0

When a table region is detected as transposed, the compressor **skips LLM header refinement** — the first column already contains the field labels, so there's nothing to refine. The table is rendered as a pipe table preserving the transposed structure, and the downstream interpret step handles it via `TableType.TransposedTable`.

This avoids sending transposed tables to the LLM, which would incorrectly try to find column headers in the top rows and produce garbled output.

### Side-by-Side Table Splitting

Some PDFs place multiple tables horizontally adjacent on the same page — for example, a "Stock at Port" table next to an "Estimated Dates" table. These tables share the same y-coordinates but are logically separate.

```
┌────────────────────┬───────────┐   ┌──────────────────────┐
│ Port │ Wheat │ Qty │   Total   │   │ Port │ Date Range    │
├──────┼───────┼─────┼───────────┤   ├──────┼───────────────┤
│ ALB  │ 53929 │ ... │   133,620 │   │ ALB  │ 1 - 15 Oct    │
│ ESP  │ 17605 │ ... │    87,753 │   │ ESP  │ 1 - 16 Aug    │
└──────┴───────┴─────┴───────────┘   └──────┴───────────────┘
        Table 1                             Table 2
```

Without splitting, heuristics would merge these into a single wide table with many empty columns.

`_find_horizontal_gaps()` detects **large gaps** (≥40 character positions) between adjacent canonical column positions. When gaps are found, the table is split at those points using `_split_grid_horizontally()`, and each sub-table is rendered as a separate markdown table.

This detection happens **before LLM header refinement** — if horizontal gaps indicate side-by-side tables, the compressor skips the LLM path and uses the splitting render path directly. This ensures each table gets its own header row and separator.

### Table Run Continuity

Tables are detected as contiguous runs of rows that share column positions. Two heuristics prevent false splits and false inclusions:

**Empty cells don't break tables** — A row with fewer filled columns (due to empty cells) still belongs to the table if ≥60% of its column positions match the established structure. This prevents splits when later rows have missing values in the rightmost columns.

**Flowing text rejection** — Rows with high average span length (>12 characters) are rejected as flowing text rather than table data. Tables have short values (~5-8 chars: dates, numbers, codes) while paragraph text has longer phrases (~15+ chars). This prevents disclaimer paragraphs from being misclassified as tables.

### API

```python
from pdf_ocr import compress_spatial_text

text = compress_spatial_text("document.pdf")
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `refine_headers` | `bool` | `True` | Use GPT-4o-mini to refine table headers and detect tables missed by heuristics |
| `pdf_path` | `str` | required | Path to the PDF file |
| `pages` | `list[int] \| None` | `None` | 0-based page indices; `None` = all |
| `cluster_threshold` | `float` | `2.0` | Max y-distance (points) to merge into one row |
| `page_separator` | `str` | `"\f"` | String inserted between pages |
| `table_format` | `str` | `"markdown"` | `"markdown"` for pipe tables or `"tsv"` for tab-separated |
| `merge_multi_row` | `bool` | `True` | Detect and merge multi-row records in tables |
| `min_table_rows` | `int` | `3` | Minimum multi-span rows to classify a region as a table |

### Compression Results

Tested across 14 input PDFs:

| Document type | Spatial chars | Compressed chars | Reduction |
|---|---|---|---|
| Shipping stems (table-heavy) | 3,600–62,000 | 1,800–20,900 | 49–67% |
| KV-pair layouts | 12,700 | 7,700 | 40% |
| Clean tables | 573 | 341 | 40% |
| Mixed content | 400–930 | 350–680 | 16–27% |

### Example

**Spatial grid** (6,471 chars):
```
                                                                                   Shipping Stem Report
                                                                                      Date Generated: 15/09/2025
                                                                                                                                                            Date of       Date of
                                                                                                                                         Quantity
   Port                 Ship Name                       Ref #             Exporter                    Commodity                           Nomination        Nomination     ETA ...
Newcastle               ADAGIO                          NT25084      ARROW COMMODITIES               Wheat               26,914                                           Completed
                                                                                                                                          11:45 AM          2:25 PM       8:06 AM ...
```

**Compressed** (2,129 chars):
```
Shipping Stem Report

Date Generated: 15/09/2025

|Port|Ship Name|Ref #|Exporter|Commodity|...|Nomination|Nomination|ETA|ETB|ETS|Load Status|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Newcastle|ADAGIO|NT25084|ARROW COMMODITIES|Wheat|26,914|10/07/2025 11:45 AM|10/07/2025 2:25 PM|06/08/2025 8:06 AM|...|Completed|
```

## Table Interpretation

Once you have compressed text, `interpret_table()` sends it through a BAML-powered LLM pipeline that extracts structured tabular data and maps it to a **canonical schema** you define. This bridges the gap between raw PDF text and typed, application-ready records.

### Why a Canonical Schema?

Source PDFs use inconsistent column names — one document says "Ship Name", another says "Vessel", a third says "Vessel Name". A canonical schema lets you declare the columns your application expects, along with aliases the LLM should recognise:

```python
from pdf_ocr import CanonicalSchema, ColumnDef

schema = CanonicalSchema(
    description="Shipping stem vessel loading records",
    columns=[
        ColumnDef("port", "string", "Loading port name — may appear as a section header rather than a column", aliases=[]),
        ColumnDef("vessel_name", "string", "Name of the vessel", aliases=["Ship Name", "Vessel"]),
        ColumnDef("commodity", "string", "Type of commodity", aliases=["Commodity"]),
        ColumnDef("quantity_tonnes", "int", "Quantity in tonnes", aliases=["Quantity", "Total"]),
        ColumnDef("eta", "date", "Estimated time of arrival", aliases=["ETA", "Date ETA of Ship"]),
        ColumnDef("status", "string", "Loading status", aliases=["Status", "Load Status"]),
    ],
)
```

Each `ColumnDef` takes:
- **`name`** — canonical column name
- **`type`** — one of `string`, `int`, `float`, `bool`, `date`
- **`description`** — human-readable description the LLM uses for disambiguation
- **`aliases`** — optional list of alternative names this column may appear as in source tables
- **`format`** — optional output format specification (see [Output Formatting](#output-formatting) below)

### Context Inference

Not all schema fields correspond to table columns. Some values live in section headers, document titles, or surrounding text. For example, a shipping stem might group rows under port name headings ("GERALDTON", "KWINANA") without a "Port" column in the table itself.

When a canonical column has **no matching source column** (especially when `aliases` is empty), the LLM looks for its value in:

- Section headers or group labels above or between data groups
- Document title, subtitle, or metadata lines
- Repeated contextual values that apply to all rows in a group

When a value is inferred from context, it is applied to all rows in that section. The `description` field on `ColumnDef` is especially important for context-inferred columns — it tells the LLM where to look.

### Designing a Canonical Schema

A canonical schema should define **structure**, not business logic. Here are guidelines for effective schema design:

#### Do

- **Define what columns exist and their types** — this is the core purpose of a schema
- **Use aliases to match source column variations** — e.g., `aliases=["Ship Name", "Vessel", "Vessel Name"]`
- **Use description for structural guidance** — tell the LLM where to find the value (e.g., "may appear as a section header rather than a column")
- **Use aliases to trigger unpivoting** — when your schema has columns whose aliases match parts of hierarchical/compound headers, the pipeline will automatically unpivot. For example, if headers are `"Battery Electric / December 2025"` and your schema has `car_motorization` with aliases `["battery electric", "plug-in hybrid"]` and `date` with aliases `["<month> <year>"]`, each source row produces multiple output records (one per motorization type × date combination)

#### Don't

- **Don't put filtering logic in descriptions** — "only the latest release" or "exclude totals" are business rules, not structural definitions. The LLM may interpret these inconsistently across batches. Filter downstream in Python/SQL instead.
- **Don't put aggregation logic in descriptions** — "sum of all regions" belongs in post-processing, not extraction
- **Don't rely on the LLM to deduplicate** — extract all matching data, dedupe downstream
- **Don't mix row-filtering with column-mapping** — the schema defines which columns to extract; which rows to keep is a separate concern

#### Example: Separating Structure from Filtering

**Problematic** (mixing concerns):
```python
ColumnDef("registration_count", "int",
          "Number of registrations — only include December 2025, exclude totals",
          aliases=["Quantity"])
```

**Better** (structure only):
```python
ColumnDef("registration_count", "int",
          "Number of registrations",
          aliases=["Quantity"])
```

Then filter downstream:
```python
records = to_records(result)
# Filter to latest period
records = [r for r in records if r['date'] == '2025-12']
# Exclude aggregation rows
records = [r for r in records if r['car_motorization'] != 'Total']
```

This separation makes extraction reliable and filtering explicit/testable.

### Pipeline

Two modes are available:

```
                        ┌──────────────────────────────────┐
  2-step pipeline       │  compressed text                 │
  (default)             │        │                         │
                        │        ▼                         │
                        │  AnalyzeAndParseTable (GPT-4o)   │
                        │        │                         │
                        │        ▼                         │
                        │  ParsedTable (intermediate)      │
                        │        │                         │
                        │        ▼                         │
                        │  MapToCanonicalSchema (GPT-4o)   │
                        │        │                         │
                        │        ▼                         │
                        │  MappedTable (output)            │
                        └──────────────────────────────────┘

                        ┌──────────────────────────────────┐
  Single-shot           │  compressed text + schema        │
  (1 LLM call)         │        │                         │
                        │        ▼                         │
                        │  InterpretTable (GPT-4o)         │
                        │        │                         │
                        │        ▼                         │
                        │  MappedTable (output)            │
                        └──────────────────────────────────┘
```

**2-step** (`interpret_table`): Step 1 analyses the table structure — identifying the table type, extracting headers, separating aggregation rows, and parsing every data row. Step 2 maps parsed columns to canonical columns using names, aliases, and descriptions, coercing types along the way. This is the default because it produces more reliable results on complex tables.

**Single-shot** (`interpret_table_single_shot`): Performs analysis, parsing, and mapping in a single LLM call. Saves one round-trip and works well for simple flat-header tables.

Both modes use GPT-4o by default; override with `model="..."`. Both support context inference and return per-field mapping metadata.

**Vision-guided parsing** (`interpret_table` with `pdf_path=`): Some PDFs have dense tables with stacked/multi-line headers where text extraction produces garbled or concatenated column names (e.g. `"33020 WHEAT"` as a single span). Passing `pdf_path=` to `interpret_table()` enables a vision pre-step:

```
Step 0 (vision):  page image + compressed text → InferredTableSchema
Step 1 (guided):  compressed text + InferredTableSchema → ParsedTable
Step 2 (unchanged): ParsedTable → MappedTable
```

Each page is rendered as an image (150 DPI) and a vision-capable LLM reads the correct column headers from the visual layout. The vision prompt cross-validates between the image and the compressed text: it first counts pipe-separated cells in the data rows to establish a minimum column count, then reads headers from the image paying attention to narrow columns that are easily missed, and finally verifies that the inferred column count is at least as large as the pipe-cell count. Step 1 then uses the guided variant (`AnalyzeAndParseTableGuided`) with the inferred schema to split compound values into the correct columns. The guided prompt includes the same section boundary format specification as the unguided variant so that section-aware batching and section boundary validation work identically. When `pdf_path` is omitted, the pipeline behaves exactly as before (no vision overhead). An optional `vision_model=` parameter allows using a different model for the vision step. Note: the vision prompt examples use domain-neutral terms to avoid biasing the model toward any specific document type.

**Vision cross-validation**: Vision mode should add clarity, not regress quality. After step 1 (guided parsing), we verify that parsed rows have exactly `vision_column_count` cells. If the cell count doesn't match (e.g., vision inferred 23 columns but guided parsing produced 21 cells), the guidance failed — we fall back to non-vision parsing for that page. This ensures vision only helps, never hurts: if vision-guided parsing can't produce rows matching the column count that vision specified, the non-vision path works directly from the data without that broken constraint.

**Multi-page auto-split**: Both `interpret_table` and `interpret_table_single_shot` automatically split input on the page separator (`\f` by default). When multiple pages are detected, all pages are processed **concurrently** via the async BAML client and `asyncio.gather()`, then results are merged into a single `MappedTable`. Single-page input with few rows is processed synchronously with no event loop overhead.

**Step-2 batching** (`interpret_table` only): After step 1 parses each page's rows, step 2 splits them into batches of `batch_size` rows (default 20) before calling the mapping LLM. This prevents output truncation on dense pages — e.g., a page with 78 data rows becomes 4 batches of 20. All batches across all pages run concurrently. Batching is section-aware: when step 1's notes contain section boundaries, rows are split at section edges when possible, and each batch receives re-indexed notes with only its relevant sections. Large sections are split internally at `batch_size` boundaries.

**Section boundary validation**: LLMs often miscount rows in long sectioned tables, producing off-by-one section boundaries that cascade across all subsequent sections (e.g., KWINANA gets 17 rows instead of 18, shifting ALBANY and ESPERANCE). After step 1, Python code deterministically counts pipe-table data rows per section directly from the compressed text (`_detect_sections_from_text`) and cross-references them with the LLM's claimed boundaries (`_validate_section_boundaries`). When the text-detected counts sum to the total data row count, they replace the LLM's boundaries. This ensures correct section labels reach step 2 for context inference (e.g., `load_port` derived from section headers). The detection works by walking the compressed text line-by-line, identifying pipe-table separators (`|---|`), counting data rows (first cell non-empty) vs aggregation rows (first cell empty, e.g., `||...337,000...|`), and pairing each table with the most recent non-pipe, non-tab text line as its section header. Extra detected sections (e.g., a stock-at-port summary table) are filtered out via ordered label-prefix matching against the LLM's section labels.

**Page-keyed results**: `interpret_table()` and `interpret_table_single_shot()` return `dict[int, MappedTable]` keyed by 1-indexed page number. Each page gets its own complete result (records, unmapped columns, mapping notes, metadata). Records contain only canonical schema fields — no internal metadata. Use `to_records(result)` to flatten all pages into a single list, or `to_records_by_page(result)` for `{page: [dicts]}`.

### Table Types and Mapping Strategies

The pipeline recognises five structural patterns, each with a specific mapping strategy:

| Type | Description | Key Signs | Mapping Strategy |
|---|---|---|---|
| `FlatHeader` | Standard table: headers in top row(s), data in subsequent rows. Each column independent. | First column has different values per row (IDs, names). Header row count ≈ data row cell count. | 1:1 — each source row → one output record |
| `HierarchicalHeader` | Tree-structured headers with parent cells spanning multiple children horizontally. | Header row has FEWER cells than data rows (parents span). Compound column names needed. | **Context-dependent**: 1:1 if aliases match full compound names; **Unpivot** if aliases match parts of compound names (see below) |
| `PivotedTable` | Cross-tabulation: categories as rows, time periods/attributes as columns, values in cells. | Column headers are dates/periods. First column has category labels. Cells are numeric values. | **Unpivot** — each row → N records (one per value column); column headers become field values |
| `TransposedTable` | Property sheet: field NAMES as rows (left column), field VALUES as columns. Each column is one record. | First column contains field names (Name, Date, Status). Rows are heterogeneous (different types per row). | **Transpose** — each column → one record; each row → one field |
| `Unknown` | Ambiguous structure | Cannot determine pattern | Best-effort based on content |

**Why table type matters for mapping**: The `AnalyzeAndParseTable` step classifies the table structure and includes this in the `ParsedTable` output. The `MapToCanonicalSchema` step then uses this classification to apply the correct transformation:

- For **pivoted tables**, the model must understand that a single source row like `"Product A | 100 | 200"` with columns `"Category | Jan | Feb"` should produce **two** output records when the schema expects `(category, period, value)` — one for January, one for February. Without explicit guidance, models often try to find a "period" column that doesn't exist.

- For **hierarchical headers with partial alias matches**, the same unpivoting logic applies. When canonical column aliases match *parts* of compound headers (not the full compound name), the model unpivots. For example:
  - Headers: `"Battery Electric / December 2025"`, `"Plug-in Hybrid / December 2025"`, etc.
  - Schema has `car_motorization` with aliases `["battery electric", "plug-in hybrid", ...]` (matches parent part)
  - Schema has `date` with aliases `["<month> <year>"]` (matches child part)
  - Result: Each country row produces multiple records — one per motorization type × date combination

  This is common in statistical reports where dimensions (category, time period) are encoded in hierarchical column headers rather than as separate columns.

- For **transposed tables**, the model inverts the usual row/column relationship — useful for property sheets where each row is a field name and each column is a different entity.

The table type descriptions are embedded in the BAML enum annotations, so the LLM has context about what each type means when parsing and when mapping.

### Dynamic Schema via TypeBuilder

The output `MappedRecord` type uses BAML's `@@dynamic` annotation — it has no fixed fields at definition time. At runtime, `interpret.py` uses a `TypeBuilder` to add one optional property per canonical column:

```python
tb = TypeBuilder()
record_builder = tb.MappedRecord
record_builder.add_property("port", native.string().optional())
record_builder.add_property("quantity_tonnes", native.int().optional())
```

This tells the LLM exactly which fields to produce and their types, without requiring a new BAML class for every schema.

### API

#### Sync

```python
from pdf_ocr import compress_spatial_text, interpret_table, CanonicalSchema, ColumnDef, to_records

compressed = compress_spatial_text("document.pdf")  # may contain \f between pages
schema = CanonicalSchema(columns=[...])

# 2-step (default) — auto-splits, batches step 2, runs all batches concurrently
result = interpret_table(compressed, schema)

# Custom batch size for very wide tables (fewer rows per LLM call)
result = interpret_table(compressed, schema, batch_size=10)

# Single-shot alternative — auto-splits but cannot batch, may truncate on dense pages
result = interpret_table_single_shot(compressed, schema)

# With a fallback model — retries with fallback if primary model fails
result = interpret_table(compressed, schema, model="openai/gpt-4o", fallback_model="openai/gpt-4o-mini")

# Vision-guided parsing — for PDFs with garbled/stacked headers
result = interpret_table(compressed, schema, pdf_path="document.pdf")

# Vision with a different model for the vision step
result = interpret_table(compressed, schema, pdf_path="document.pdf", vision_model="openai/gpt-4o")

# Convert to plain dicts
for record in to_records(result):
    print(record)
```

#### Async (explicit parallel control)

```python
import asyncio
from pdf_ocr.interpret import interpret_tables_async, CanonicalSchema, ColumnDef

pages = compressed.split("\f")
tables = asyncio.run(interpret_tables_async(pages, schema, fallback_model="openai/gpt-4o-mini"))
```

`interpret_tables_async` runs all parse calls concurrently via `asyncio.gather()`, then all map calls concurrently. Use this when you already have an event loop or want explicit control over which text chunks to parallelize. For most cases, `interpret_table()` handles concurrency automatically.

All functions accept an optional `fallback_model` keyword argument. When set, if the primary `model` raises an exception (network error, rate limit, unavailable model), the call is retried with the fallback model.

#### Functions

| Function | Steps | Default model | Use case |
|---|---|---|---|
| `interpret_table(text, schema)` | 2 (parse + map) | GPT-4o | Default; best accuracy. Auto-splits, batches step 2, returns `dict[int, MappedTable]`. Pass `pdf_path=` for vision-guided parsing |
| `interpret_table_single_shot(text, schema)` | 1 | GPT-4o | Simple tables; saves one round-trip. No batching — may truncate on dense pages |
| `analyze_and_parse(text)` | 1 | GPT-4o | Parse only; inspect structure before mapping |
| `analyze_and_parse_guided(text, visual_schema)` | 1 | GPT-4o | Parse guided by a visual schema (for garbled headers) |
| `infer_table_schema_from_image(img_b64, text)` | 1 | GPT-4o | Vision step: infer column structure from a page image |
| `map_to_schema(parsed, schema)` | 1 | GPT-4o | Map a previously parsed table |
| `interpret_tables_async(texts, schema)` | 2 per table | GPT-4o | Explicit async control over pre-split tables |
| `to_records(result)` | — | — | Flatten `dict[int, MappedTable]` (or single `MappedTable`) to `list[dict]` — schema columns only |
| `to_records_by_page(result)` | — | — | Convert to `{page: [dicts]}` — grouped by source page |

All interpretation functions accept `model` and `fallback_model` keyword arguments to override the default model.

#### Output

`interpret_table` returns a `dict[int, MappedTable]` keyed by 1-indexed page number. Each `MappedTable` contains:
- **`records`** — list of `MappedRecord` objects. Dynamic fields match the canonical column names. Access via `record.port`, `record.quantity_tonnes`, etc. Records contain only canonical schema fields. Use `to_records()` for plain dicts or `to_records_by_page()` for page-grouped output.
- **`unmapped_columns`** — source columns that could not be mapped to any canonical column.
- **`mapping_notes`** — optional notes about ambiguous matches or type coercion issues.
- **`metadata`** — an `InterpretationMetadata` object with:
  - **`model`** — the model that produced the result (e.g. `"openai/gpt-4o"`).
  - **`table_type_inference`** — a `TableTypeInference` object with the table type and mapping strategy:
    - `table_type` — the table type from Step 1 (`FlatHeader`, `HierarchicalHeader`, `PivotedTable`, `TransposedTable`, or `Unknown`). This is authoritative — Step 2 does not re-classify.
    - `mapping_strategy_used` — which mapping strategy was applied (e.g. `"1:1 row mapping for FlatHeader"`, `"unpivot for PivotedTable"`)
  - **`field_mappings`** — one `FieldMapping` per canonical column, each containing:
    - `column_name` — the canonical column name
    - `source` — where the value came from (e.g. `"column: Ship Name"`, `"section header"`, `"document title"`)
    - `rationale` — brief explanation of why this mapping was chosen
    - `confidence` — `High` (direct name/alias match), `Medium` (inferred from context), or `Low` (best guess)
  - **`sections_detected`** — section/group labels found in the text (e.g. `["GERALDTON", "KWINANA"]`), or `None` if no sections were detected.

## Output Formatting

The `format` field on `ColumnDef` specifies how values should be formatted in the output. The format is applied both by the LLM during extraction and by the serialization module during export.

### Date & Timestamp Formats

| Format | Example Output | Description |
| --- | --- | --- |
| `YYYY-MM-DD` | `2025-01-15` | ISO date |
| `YYYY-MM` | `2025-01` | Year-month |
| `YYYY` | `2025` | Year only |
| `DD/MM/YYYY` | `15/01/2025` | European date |
| `MM/DD/YYYY` | `01/15/2025` | US date |
| `MMM YYYY` | `Jan 2025` | Short month name |
| `MMMM YYYY` | `January 2025` | Full month name |
| `YYYY-MM-DD HH:mm` | `2025-01-15 14:30` | Date + time (24h) |
| `YYYY-MM-DD HH:mm:ss` | `2025-01-15 14:30:45` | Date + time + seconds |
| `YYYY-MM-DDTHH:mm:ssZ` | `2025-01-15T14:30:45Z` | ISO 8601 UTC |
| `HH:mm` | `14:30` | Time only (24h) |
| `HH:mm:ss` | `14:30:45` | Time with seconds |
| `hh:mm A` | `02:30 PM` | Time (12h with AM/PM) |

### Number Formats

| Format | Example Output | Description |
| --- | --- | --- |
| `#` | `1234` | Plain integer |
| `#.##` | `1234.56` | 2 decimal places |
| `#.####` | `1234.5678` | 4 decimal places |
| `#,###` | `1,234` | Thousands separator |
| `#,###.##` | `1,234.56` | Both |
| `+#` | `+1234` / `-1234` | Explicit sign |
| `#%` | `12.5%` | Percentage |
| `(#)` | `(1234)` | Negative in parentheses |

### String Formats

| Format | Example Output | Description |
| --- | --- | --- |
| `uppercase` | `HELLO WORLD` | All caps |
| `lowercase` | `hello world` | All lower |
| `titlecase` | `Hello World` | Title Case |
| `capitalize` | `Hello world` | First letter only |
| `camelCase` | `helloWorld` | camelCase |
| `PascalCase` | `HelloWorld` | PascalCase |
| `snake_case` | `hello_world` | snake_case |
| `SCREAMING_SNAKE_CASE` | `HELLO_WORLD` | SCREAMING_SNAKE_CASE |
| `kebab-case` | `hello-world` | kebab-case |
| `SCREAMING-KEBAB-CASE` | `HELLO-WORLD` | SCREAMING-KEBAB-CASE |
| `trim` | `hello` | Strip whitespace |

### Format Usage Example

```python
schema = CanonicalSchema(columns=[
    ColumnDef("country", "string", "Country name", aliases=["Country"], format="uppercase"),
    ColumnDef("date", "string", "Registration period", aliases=[], format="YYYY-MM"),
    ColumnDef("registrations", "int", "Number of registrations", aliases=["Units"], format="#,###"),
    ColumnDef("growth", "float", "YoY growth rate", aliases=["% Change"], format="+#.##%"),
])
```

Output:

```csv
country,date,registrations,growth
GERMANY,2025-01,"1,234,567",+12.35%
FRANCE,2024-06,"987,654",-5.67%
```

## Serialization

After interpreting tables, the `serialize` module exports results to common formats — CSV, TSV, Parquet, pandas DataFrames, and polars DataFrames. All functions validate records against the canonical schema using Pydantic before serialization, and apply output formatting specified in the schema's `format` field.

### Features

- **Pydantic validation**: Records are validated against the `CanonicalSchema` types before export
- **OCR artifact coercion**: Handles common OCR artifacts automatically:
  - Comma-separated numbers: `"1,234"` → `1234`
  - Negative in parentheses: `"(500)"` → `-500`
  - Percentage strings: `"12.5%"` → `12.5`
  - Boolean strings: `"yes"`, `"true"`, `"1"` → `True`
  - Empty/whitespace → `None`
- **Output formatting**: Applies format specifications from the schema (dates, numbers, strings)
- **Page column**: Optional `include_page=True` adds a 1-indexed page number column
- **Proper nullable types**: pandas uses `Int64`/`Float64`/`string`/`boolean` nullable dtypes; polars uses native nullable types

### Serialization API

```python
from pdf_ocr import to_csv, to_tsv, to_parquet, to_pandas, to_polars

# CSV/TSV: return string or write to file
csv_str = to_csv(result, schema)                          # Returns CSV string
to_csv(result, schema, path="output.csv")                 # Writes file, returns None
to_csv(result, schema, path="output.csv", include_page=True)  # With page column

tsv_str = to_tsv(result, schema)                          # Same API as to_csv

# Parquet: always writes to file (requires pyarrow)
to_parquet(result, schema, "output.parquet")
to_parquet(result, schema, "output.parquet", include_page=True)

# DataFrames (require pandas/polars)
df = to_pandas(result, schema)                            # pandas.DataFrame
df = to_pandas(result, schema, include_page=True)

pl_df = to_polars(result, schema)                         # polars.DataFrame
pl_df = to_polars(result, schema, include_page=True)
```

### Function Signatures

| Function | Returns | Writes to file | Requires |
| --- | --- | --- | --- |
| `to_csv(result, schema, *, path=None, include_page=False)` | `str \| None` | If `path` provided | stdlib |
| `to_tsv(result, schema, *, path=None, include_page=False)` | `str \| None` | If `path` provided | stdlib |
| `to_parquet(result, schema, path, *, include_page=False)` | `None` | Always | `pyarrow` |
| `to_pandas(result, schema, *, include_page=False)` | `pd.DataFrame` | No | `pandas` |
| `to_polars(result, schema, *, include_page=False)` | `pl.DataFrame` | No | `polars` |

### Type Mapping

| Schema type | Python type | pandas dtype | polars dtype |
| --- | --- | --- | --- |
| `string` | `str \| None` | `string` | `Utf8` |
| `int` | `int \| None` | `Int64` | `Int64` |
| `float` | `float \| None` | `Float64` | `Float64` |
| `bool` | `bool \| None` | `boolean` | `Boolean` |
| `date` | `str \| None` | `string` | `Utf8` |

### Optional Dependencies

Install optional dependencies for DataFrame/Parquet support:

```bash
pip install pdf-ocr[dataframes]  # pandas + polars
pip install pdf-ocr[parquet]     # pyarrow
pip install pdf-ocr[all]         # everything
```

When a required library is missing, functions raise a helpful `ImportError` with installation instructions.

### Validation Errors

If a record fails validation, `SerializationValidationError` is raised:

```python
from pdf_ocr import SerializationValidationError

try:
    csv_str = to_csv(result, schema)
except SerializationValidationError as e:
    print(f"Record {e.record_index} failed: {e}")
    print(f"Column: {e.column_name}")
    print(f"Original error: {e.original_error}")
```

In practice, validation rarely fails because the coercion logic converts invalid values to `None`, which is allowed for all columns.

## Limitations

- **Monospace assumption**: The grid uses a single character width per page. Proportional fonts will have slight alignment imperfections, though the median-based cell width minimizes this.
- **No image/drawing extraction**: Only text spans are rendered.
- **Rotation**: Rotated text is placed at its origin point but the text direction is not adjusted.
- **Very dense pages**: Pages with many overlapping elements at different sizes may produce wide lines due to the single grid resolution.
- **Vision mode cost**: Each page rendered at 150 DPI produces a ~200-400KB PNG image sent to the vision LLM. This adds latency and token cost (one extra LLM call per page for step 0). Only use `pdf_path=` when text-based extraction produces garbled headers.
- **Section boundary validation scope**: The deterministic section detection relies on the compressed text having a repeating pattern of section-header → pipe-table → aggregation-row. It requires at least 2 such sections to activate. Tables without aggregation rows, or with non-standard compressed text formatting, fall back to the LLM's section boundaries as-is.
