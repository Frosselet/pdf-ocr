# Compressed Spatial Text

> **Module**: `compress.py`
> **Public API**: `compress_spatial_text()`, `compress_spatial_text_structured()`

Transforms the spatial grid into a **token-efficient structured representation** — markdown tables, flowing paragraphs, and key-value lines instead of whitespace-heavy grids.

## Why Compress?

The spatial grid preserves layout perfectly but wastes tokens on whitespace padding. A 62,000-character shipping stem is mostly spaces. Compression reduces tokens by 40-67% while preserving structure.

## Pipeline

```
PageLayout (from spatial_text.py)
  │
  ▼
1. Span splitting ──► split merged PDF spans at column boundaries
  │
  ▼
2. Region detection ──► classify row groups as table/text/heading/kv_pairs/scattered
  │
  ▼
3. Multi-row merge ──► collapse multi-row records (e.g., 3-row shipping entries)
  │
  ▼
4. Header refinement ──► find preceding header rows, build stacked headers
  │
  ▼
5. Side-by-side split ──► detect and separate horizontally adjacent tables
  │
  ▼
6. Section label detection ──► extract port names, category labels
  │
  ▼
7. Rendering ──► markdown tables, flowing text, key-value lines
```

---

## Heuristics

### H1: Span Splitting (`_split_merged_spans`)

**Problem**: PDF creators sometimes encode adjacent cells as a single text span:
- `"7:00:00 PM BUNGE"` — time + exporter merged
- `"33020 WHEAT"` — code + commodity merged

**Solution**: Use column boundaries from **all rows** to split merged spans:

1. Collect column positions from every row
2. For each span, find positions from OTHER rows that fall inside it
3. Split at word boundaries (whitespace before/after the character index)
4. Minimum gap of 5 characters to avoid false splits

**Key insight**: Header rows often have finer-grained spans than data rows. Use them to correct data rows.

```python
# If header has spans at columns 219 ("EXPORTER") and 245 ("COMMODITY")
# and data span at 208 covers both positions:
#   "7:00:00 PM BUNGE WHEAT" → ["7:00:00 PM", "BUNGE", "WHEAT"]
```

---

### H2: Region Classification (`_classify_regions`)

**Problem**: A PDF page contains mixed content — tables, paragraphs, headings, key-value pairs.

**Solution**: Classify contiguous row groups by structural patterns:

| Region Type | Detection Rule | Rendered As |
|---|---|---|
| `TABLE` | 3+ rows sharing 2+ aligned column anchors | Markdown pipe table |
| `TEXT` | Consecutive single-span rows, left-aligned | Flowing paragraph |
| `HEADING` | Single short span, isolated | Plain text line |
| `KV_PAIRS` | Consecutive 2-span rows (label + value) | `key: value` lines |
| `SCATTERED` | Fallback for unclassified rows | Tab-separated spans |

---

### H3: Table Detection Pool (`_detect_table_runs`)

**Problem**: Tables have varying row structures — some rows have 7 columns, others 5 (empty cells).

**Solution**: Column anchor pool with overlap-based matching:

```python
pool = set()  # all unique column positions seen in this run

for row in rows:
    overlap_count = count of row's columns within tolerance of pool
    overlap_ratio = overlap_count / len(row.columns)

    if overlap_count >= 2 or overlap_ratio >= 0.6:
        add row to current table run
        add row's columns to pool
    else:
        flush current run, start new one
```

**Thresholds**:
- `overlap_count >= 2`: At least 2 columns must match existing structure
- `overlap_ratio >= 0.6`: Or 60% of columns fit (handles empty cells)
- `col_tolerance`: Dynamic, based on cell width (`max(3, cell_w * 1.5)`)

---

### H4: Section Label vs Table Continuation (`_is_table_continuation`)

**Problem**: Single-span rows inside tables could be:
- **Aggregation rows**: `337,000` (numeric total) — keep in table
- **Section labels**: `KWINANA` (port name) — flush table, start new section

**Solution**: Content-based classification:

```python
def _is_table_continuation(text):
    if empty: return True
    if parenthesized (e.g., "(tonnes)"): return True  # unit annotation
    stripped = remove " ,._$€£%+-"
    return stripped.isdigit()  # numeric = continuation
```

**Effect**: `GERALDTON`, `KWINANA`, `ALBANY` become section headings; `337,000` stays in table.

---

### H5: Flowing Text Rejection

**Problem**: Paragraph text shouldn't be classified as table data.

**Solution**: Reject rows with high average span length:

```python
median_span_len = median of span lengths in multi-span rows
if avg_span_len > median_span_len * 2.0:
    reject as flowing text, not table data
```

**Rationale**: Tables have short values (~5-8 chars: dates, numbers, codes). Paragraphs have long phrases (~15+ chars).

---

### H6: Multi-Row Record Detection (`_detect_multi_row_period`)

**Problem**: Some PDFs split a single logical record across multiple rows:

```
Row 1: dates    →  10/07/2025   10/07/2025   06/08/2025
Row 2: data     →  Newcastle  ADAGIO  NT25084  ARROW  Wheat  26,914
Row 3: times    →  11:45 AM   2:25 PM   8:06 AM
```

**Solution**: Detect repeating span-count patterns:

1. Build span counts per row: `[5, 7, 5, 5, 7, 5, 5, 7, 5, ...]`
2. Try periods 2, 3, 4 with header offsets 0-10
3. Match if 70%+ of groups follow the pattern
4. Merge sub-rows by column position

```python
# Pattern [5, 7, 5] repeated = period 3
# Merge: (col, "10/07/2025") + (col, "11:45 AM") → "10/07/2025 11:45 AM"
```

---

### H7: Header Row Estimation (`_estimate_header_rows`)

**Problem**: Where do headers end and data begin?

**Solution**: Bottom-up span-count analysis:

1. Bottom 2/3 of table establishes "data-like" span counts
2. Find top-3 most common counts in bottom rows (must be ≥2 spans, ≥2 occurrences)
3. Scan from top: first row matching a data pattern = data starts

```python
data_counts = {c for c, freq in bottom_counts.most_common(3) if c >= 2 and freq >= 2}
max_data_count = max(data_counts)

for i, span_count in enumerate(rows):
    if span_count in data_counts or span_count > max_data_count:
        return i  # header row count
```

**Key insight**: Rows with MORE spans than typical are also data (more complete records).

---

### H8: Preceding Header Row Detection (`_find_preceding_header_rows`)

**Problem**: Header rows may be classified as SCATTERED or KV_PAIRS, not part of the TABLE region.

**Solution**: Scan upward from table start:

1. Collect rows whose span positions align with data column positions
2. Stop at: non-matching row, gap > 2 rows, or another table region
3. For single-span rows: require start position alignment AND length < 15

```python
for ri in reversed(rows_above_table):
    if gap > 2: break
    if single_span:
        if start_aligns and len(text) < 15 and not section_label:
            include as header
        else: break
    elif any span aligns: include
    else: break
```

**GERALDTON fix**: Also reject section labels via `_is_table_continuation()`.

---

### H9: Alignment-Aware Header Matching (`_build_stacked_headers`)

**Problem**: Headers and data have inconsistent alignment:
- Headers left-aligned, data right-aligned (numbers)
- Headers positioned 5-10 characters before/after data

**Solution**: Bounding-box overlap instead of start-position matching:

1. **Compute column bounds** from data rows: `(min_start, max_end)` per column
2. **Match headers by overlap**: header span overlaps `[d_min - margin, d_max)`
3. **Best overlap wins**: when header overlaps multiple columns
4. **Tie-breaker**: prefer column whose start is closest to header start
5. **Deduplicate**: remove consecutive identical words

```python
def _overlaps(h_start, h_end, d_min, d_max):
    return h_start < d_max and h_end > (d_min - left_margin)

# left_margin=5 captures headers positioned just before data
```

**Example**: Bunge PDF has 22 columns with 9 stacked header rows. Without overlap matching, columns 9, 17, 19, 20 had no headers. With overlap matching, all 22 columns have correct headers.

---

### H10: Column Unification (`_unify_columns`)

**Problem**: Same logical column appears at slightly different x-positions across rows.

**Solution**: Greedy clustering with running mean:

```python
for c in sorted(all_column_positions):
    if clusters and abs(c - mean(clusters[-1])) <= tolerance:
        clusters[-1].append(c)  # merge
    else:
        clusters.append([c])  # new cluster

canonical = [round(mean(cluster)) for cluster in clusters]
```

**Key**: Uses data rows only (not headers) to avoid phantom columns from header spans at different positions.

---

### H11: Twin Column Merging (`_merge_twin_columns`)

**Problem**: Adjacent columns that are never simultaneously filled = same logical column.

**Solution**: Merge adjacent never-overlapping columns:

```python
for each adjacent column pair (i, i+1):
    if no row has both columns filled:
        merge into single column
```

**Example**: Right-aligned data vs left-aligned data in same logical column.

---

### H12: Horizontal Gap Detection (`_find_horizontal_gaps`)

**Problem**: Side-by-side tables share y-coordinates but are logically separate.

**Solution**: Detect gaps > 3x median inter-column gap:

```python
median_gap = median of gaps between adjacent columns
for gap in gaps:
    if gap > median_gap * 3.0:
        mark as split point
```

**Effect**: "Stock at Port" table and "Maintenance Dates" table become separate entries.

---

### H13: Transposed Table Detection (`_is_transposed_table`)

**Problem**: Some tables have field names as rows, records as columns (transposed).

**Solution**: Structural heuristics:

- **Few columns** (≤5): one label column + 1-4 record columns
- **Low span variance** (<2.0): consistent row structure
- **Stable first column** (≥80%): field labels always in column 0

**Effect**: Skip header refinement for transposed tables — first column IS the headers.

---

### H14: Section Label Detection (`_find_section_label_above`)

**Problem**: Port names like "GERALDTON" appear as HEADING regions above tables.

**Solution**: Look backwards through regions for closest HEADING:

```python
for region in reversed(regions[:table_index]):
    if region.type == HEADING:
        return heading_text if short and not table_continuation
    if region.type == TABLE:
        return None  # hit another table, no section label
    # skip SCATTERED regions (header fragments)
```

---

## Visual Heuristics (Cross-Validation)

Visual heuristics run in parallel with text heuristics. They extract lines, fills, and borders from PDF drawing commands and cross-validate text-based structure detection.

**Philosophy**: Humans use more than text positioning to recognize tables. Visual cues—borders, background colors, zebra striping—trigger instant "this is a table" recognition. These non-text elements are part of the producer's intention and MUST be considered when present.

**Design principle**: Visual cues are not tie-breakers. They are a parallel validation system:
1. If text and visual agree → high confidence
2. If visual exists but text missed it → text heuristic gap (catch silent failure)
3. If text and visual contradict → investigate why (diagnostic signal)

---

### VH1: Table Grid Boundaries (`_detect_visual_grid`)

**Holistic**: A grid of intersecting lines unmistakably signals tabular structure.

**Detection**:
- Extract lines from `page.get_drawings()` (strokes and thin fills)
- Significant lines: ≥10% of page width (h-lines) or ≥2% of page height (v-lines)
- Grid exists if ≥3 significant h-lines AND ≥3 v-lines

**Cross-validation**:
- ✓ Grid bbox matches text-detected TABLE region → confirmed
- ⚠ Grid exists but text classified as SCATTERED → text heuristic gap
- ⚠ Text detected TABLE but no grid → whitespace-only table (valid)

---

### VH2: Header Background Highlighting (`_detect_visual_header`)

**Holistic**: Adding a background color to a row highlights its semantic role as a header.

**Detection**:
- Find fills that overlap with top few rows (header zone)
- Group fills by color, find most common header color
- Match filled rows to text row positions

**Cross-validation**:
- ✓ Fill band matches `_estimate_header_rows()` count → confirmed
- ⚠ Fill suggests N header rows but text says M → investigate
- ⚠ No fills but text detected headers → text-only headers (valid)

---

### VH3: Data Row Alternation — Zebra Striping (`_detect_visual_zebra`)

**Holistic**: Alternating row colors help humans track across wide tables; the pattern signals "these are data rows."

**Detection**:
- Map each row to its fill color
- Check for alternating A-B-A-B pattern in middle/lower section
- Require ≥4 alternations (8 rows) to confirm zebra pattern

**Cross-validation**:
- ✓ Zebra zone matches text-detected data rows → confirmed
- ⚠ Zebra starts at row N but text says headers end at row M → boundary mismatch

---

### VH4: Section Separator Lines (`_detect_visual_separators`)

**Holistic**: A thick horizontal rule signals a break between sections.

**Detection**:
- Find h-lines spanning >80% of table width
- Filter by stroke width > 1.5x median border width
- Position marks section boundary

**Cross-validation**:
- ✓ Separator position matches text-detected section label row → confirmed
- ⚠ Separator exists but no section label in text → visual-only section break

---

### VH5: Cell Border Presence (planned)

**Holistic**: Borders around individual cells confirm each cell is a discrete data container.

*Not yet implemented — will detect closed rectangles around text spans.*

---

### VH6: Exception Highlighting (`_cross_validate_header_rows`)

**Holistic**: Color can highlight EXCEPTIONS, not just headers. A colored first data row means "pay attention to this record," not "this is a header."

**Detection**:
- Visual says "header" (colored fill) but type pattern (TH1) says "data"
- → Exception highlight, not header

**Example**: A first data row with colored background but containing dates/numbers matching data pattern is highlighted for attention, not part of headers.

---

## Type Pattern Heuristics

Type patterns analyze WHAT is in cells, not WHERE they are. This is complementary to spatial heuristics.

---

### TH1: Header Type Pattern (`_is_header_type_pattern`)

**Holistic**: Column headers are LABELS. They describe what's in the column. Labels are strings, not dates or numbers.

**Detection**: A row with only STRING content is likely a header. A row with mixed types (strings + dates + numbers) is likely data.

```python
def _is_header_type_pattern(row):
    for cell in row:
        if _detect_cell_type(cell) in (DATE, NUMBER):
            return False
    return True
```

**Cross-validation with visual**:
- ✓ String-only row + header color → confirmed header
- ⚠ Mixed-type row + header color → EXCEPTION HIGHLIGHT (not header)
- ✓ Mixed-type row + no special color → confirmed data row

---

### TH2: Data Type Consistency (`_detect_column_types`)

**Holistic**: Data columns have consistent types. The "Date" column always has dates. The "Quantity" column always has numbers.

**Detection**:
- Analyze data rows to find predominant type per column
- DATE: matches date patterns (ISO, slashed, time formats)
- NUMBER: digits with optional separators
- ENUM: few unique values that repeat (≤5 distinct or ≤10% unique ratio)
- STRING: default fallback

**Effect**: Disambiguates colored rows. Type signature matching helps distinguish header-like rows from data rows.

---

### TH3: Cell Type Detection (`_detect_cell_type`)

**Detection patterns**:

| Type | Patterns |
|---|---|
| DATE | `YYYY-MM-DD`, `DD/MM/YYYY`, `HH:MM`, `HH:MM AM/PM`, etc. |
| NUMBER | `1,234`, `(500)`, `$1.23`, `45%`, etc. |
| STRING | Everything else |

---

## API

### Markdown Output

```python
from pdf_ocr import compress_spatial_text

text = compress_spatial_text("document.pdf")
```

### Structured Output

```python
from pdf_ocr import compress_spatial_text_structured, StructuredTable, TableMetadata

tables = compress_spatial_text_structured("document.pdf")

for table in tables:
    print(f"Page {table.metadata.page_number}")
    print(f"Section: {table.metadata.section_label}")
    print(f"Columns: {table.column_names}")
    for row in table.data:
        print(row)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pdf_input` | `str \| bytes \| Path` | required | Path to PDF or raw bytes |
| `pages` | `list[int] \| None` | `None` | 0-based page indices |
| `cluster_threshold` | `float` | `2.0` | Y-distance for row merging |
| `refine_headers` | `bool` | `True` | Use LLM for header refinement fallback |
| `table_format` | `str` | `"markdown"` | `"markdown"` or `"tsv"` |
| `merge_multi_row` | `bool` | `True` | Detect/merge multi-row records |
| `min_table_rows` | `int` | `3` | Minimum rows for table classification |
| `extract_visual` | `bool` | `True` | Extract visual elements for cross-validation |

---

## Data Types

### StructuredTable

| Field | Type | Description |
|---|---|---|
| `metadata` | `TableMetadata` | Non-data info |
| `column_names` | `list[str]` | Column headers |
| `data` | `list[list[str]]` | Data rows |

### TableMetadata

| Field | Type | Description |
|---|---|---|
| `page_number` | `int` | 0-based page index |
| `table_index` | `int` | Index within page (side-by-side: 0, 1, ...) |
| `section_label` | `str \| None` | Section heading above table |

---

## Compression Results

| Document Type | Spatial Chars | Compressed | Reduction |
|---|---|---|---|
| Shipping stems | 3,600–62,000 | 1,800–20,900 | 49–67% |
| KV-pair layouts | 12,700 | 7,700 | 40% |
| Clean tables | 573 | 341 | 40% |
| Mixed content | 400–930 | 350–680 | 16–27% |
