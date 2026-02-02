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
1. Region detection ──► classify row groups as table / text / heading / kv_pairs / scattered
  │
  ▼
2. Multi-row merge ──► detect repeating row patterns (e.g., 3-row shipping records) and collapse
  │
  ▼
3. Region rendering ──► markdown tables, flowing paragraphs, key: value lines
  │
  ▼
4. Assembly ──► join regions with blank lines, pages with separator
```

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

### API

```python
from pdf_ocr import compress_spatial_text

text = compress_spatial_text("document.pdf")
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
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

Each `ColumnDef` takes a canonical `name`, a `type` (`string`, `int`, `float`, `bool`, `date`), a human-readable `description` the LLM uses for disambiguation, and optional `aliases`.

### Context Inference

Not all schema fields correspond to table columns. Some values live in section headers, document titles, or surrounding text. For example, a shipping stem might group rows under port name headings ("GERALDTON", "KWINANA") without a "Port" column in the table itself.

When a canonical column has **no matching source column** (especially when `aliases` is empty), the LLM looks for its value in:

- Section headers or group labels above or between data groups
- Document title, subtitle, or metadata lines
- Repeated contextual values that apply to all rows in a group

When a value is inferred from context, it is applied to all rows in that section. The `description` field on `ColumnDef` is especially important for context-inferred columns — it tells the LLM where to look.

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

**Multi-page auto-split**: Both `interpret_table` and `interpret_table_single_shot` automatically split input on the page separator (`\f` by default). When multiple pages are detected, all pages are processed **concurrently** via the async BAML client and `asyncio.gather()`, then results are merged into a single `MappedTable`. Single-page input with few rows is processed synchronously with no event loop overhead.

**Step-2 batching** (`interpret_table` only): After step 1 parses each page's rows, step 2 splits them into batches of `batch_size` rows (default 20) before calling the mapping LLM. This prevents output truncation on dense pages — e.g., a page with 78 data rows becomes 4 batches of 20. All batches across all pages run concurrently. Batching is section-aware: when step 1's notes contain section boundaries, rows are split at section edges when possible, and each batch receives re-indexed notes with only its relevant sections. Large sections are split internally at `batch_size` boundaries.

**Section boundary validation**: LLMs often miscount rows in long sectioned tables, producing off-by-one section boundaries that cascade across all subsequent sections (e.g., KWINANA gets 17 rows instead of 18, shifting ALBANY and ESPERANCE). After step 1, Python code deterministically counts pipe-table data rows per section directly from the compressed text (`_detect_sections_from_text`) and cross-references them with the LLM's claimed boundaries (`_validate_section_boundaries`). When the text-detected counts sum to the total data row count, they replace the LLM's boundaries. This ensures correct section labels reach step 2 for context inference (e.g., `load_port` derived from section headers). The detection works by walking the compressed text line-by-line, identifying pipe-table separators (`|---|`), counting data rows (first cell non-empty) vs aggregation rows (first cell empty, e.g., `||...337,000...|`), and pairing each table with the most recent non-pipe, non-tab text line as its section header. Extra detected sections (e.g., a stock-at-port summary table) are filtered out via ordered label-prefix matching against the LLM's section labels.

**Page-keyed results**: `interpret_table()` and `interpret_table_single_shot()` return `dict[int, MappedTable]` keyed by 1-indexed page number. Each page gets its own complete result (records, unmapped columns, mapping notes, metadata). Records contain only canonical schema fields — no internal metadata. Use `to_records(result)` to flatten all pages into a single list, or `to_records_by_page(result)` for `{page: [dicts]}`.

### Table Types

The pipeline recognises five structural patterns:

| Type | Description | Example |
|---|---|---|
| `FlatHeader` | Single header row, regular data rows | Standard shipping stem |
| `HierarchicalHeader` | Multi-level headers with spanning groups | Quarterly report with "Q1 / Revenue / Costs" |
| `PivotedTable` | Categories as rows, periods/attributes as columns | Commodity volumes by month |
| `TransposedTable` | Fields as rows, records as columns | Single-entity property sheet |
| `Unknown` | Fallback for ambiguous structures | — |

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
  - **`field_mappings`** — one `FieldMapping` per canonical column, each containing:
    - `column_name` — the canonical column name
    - `source` — where the value came from (e.g. `"column: Ship Name"`, `"section header"`, `"document title"`)
    - `rationale` — brief explanation of why this mapping was chosen
    - `confidence` — `High` (direct name/alias match), `Medium` (inferred from context), or `Low` (best guess)
  - **`sections_detected`** — section/group labels found in the text (e.g. `["GERALDTON", "KWINANA"]`), or `None` if no sections were detected.

## Limitations

- **Monospace assumption**: The grid uses a single character width per page. Proportional fonts will have slight alignment imperfections, though the median-based cell width minimizes this.
- **No image/drawing extraction**: Only text spans are rendered.
- **Rotation**: Rotated text is placed at its origin point but the text direction is not adjusted.
- **Very dense pages**: Pages with many overlapping elements at different sizes may produce wide lines due to the single grid resolution.
- **Vision mode cost**: Each page rendered at 150 DPI produces a ~200-400KB PNG image sent to the vision LLM. This adds latency and token cost (one extra LLM call per page for step 0). Only use `pdf_path=` when text-based extraction produces garbled headers.
- **Section boundary validation scope**: The deterministic section detection relies on the compressed text having a repeating pattern of section-header → pipe-table → aggregation-row. It requires at least 2 such sections to activate. Tables without aggregation rows, or with non-standard compressed text formatting, fall back to the LLM's section boundaries as-is.
