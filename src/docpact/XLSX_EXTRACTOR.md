# XLSX Table Extraction

> **Module**: `xlsx_extractor.py`
> **Public API**: `extract_tables_from_xlsx()`, `extract_tables_from_excel()`, `xlsx_to_markdown()`

Extracts structured tables from Excel files (.xlsx, .xls), handles merged-cell headers, multi-table sheets, hidden content filtering, and produces `StructuredTable` objects compatible with the interpretation pipeline.

## Why a Dedicated Extractor?

XLSX tables have explicit grid structure (cells at known row/column positions), unlike PDFs where columns are inferred from spatial clustering. But real-world spreadsheets introduce complications:

- **Merged header cells**: Labels like "Revenue" span multiple columns (e.g., over "Q1" / "Q2") using openpyxl merge ranges
- **Multiple tables per sheet**: Authors place several tables on one worksheet, separated by blank rows/columns
- **Hidden content**: Columns or rows may be hidden (formulas, intermediate calculations) but still present in the file
- **Number formats**: The same value can be displayed as a date, currency, or percentage depending on the cell's format string

## Pipeline

```text
XLSX file
  |
  v
1. Load workbook (openpyxl, data_only=True)
  |
  v
2. XH3: Identify hidden rows/columns (_get_hidden_rows, _get_hidden_cols)
   -- Check ws.column_dimensions[letter].hidden
   -- Check ws.row_dimensions[row].hidden
  |
  v
3. XH1: Detect table regions (_detect_table_regions)
   -- Scan for blank-row runs (≥2 consecutive) → vertical separators
   -- Within each segment, scan for blank-column runs → horizontal separators
   -- Each non-empty rectangle ≥ 2×2 → candidate table
  |
  v
4. Extract grid (_extract_table_from_range)
   -- Expand merged cells (_expand_merged_cells)
   -- Filter hidden rows/columns (XH3)
   -- Collect number format hints (XH4)
   -- Extract visual styles (optional)
  |
  v
5. Estimate header rows (layered approach)
   -- Merge-based: last horizontal merge row + type continuation
   -- Type-pattern: consecutive all-string rows (TH2)
   -- Span-count: bottom-up H7 analysis
   -- Visual fill: header fill rows (VH2)
   -- Take maximum across all methods
  |
  v
6. XH2: Detect title row
   -- Single non-empty cell in row 0 with header_count > 1
   -- Extract as metadata, exclude from headers
  |
  v
7. Build compound headers (_build_column_names_with_forward_fill)
   -- Forward-fill empty cells in each header row
   -- Stack rows with " / " separator
   -- Deduplicate consecutive identical fragments
  |
  v
8. Return StructuredTable objects
   -- column_names, data, source_format="xlsx", metadata
```

---

## Heuristics

### XH1: Blank Row/Column Table Boundary (`_detect_table_regions`)

**Problem**: A single worksheet may contain multiple tables separated by blank space. The previous implementation treated the entire sheet as one table (explicit TODO in the code).

**Solution**: Two-pass boundary detection:

1. **Vertical split**: Scan all rows for blank-row runs. A run of N ≥ 2 consecutive blank rows is a vertical separator. Split the sheet into vertical segments.
2. **Horizontal split**: Within each vertical segment, scan columns for blank-column runs. N ≥ 2 consecutive blank columns is a horizontal separator.
3. **Filter**: Each resulting rectangle must be ≥ 2 rows × 2 columns.

**Rationale**: One blank row is "breathing room within a group" (subtotals, section breaks). Two+ blank rows signal "these are separate things." This matches human visual perception of worksheet layout. The same principle applies to columns for side-by-side tables.

**Implementation**: `_split_by_blank_runs()` is a reusable helper that splits a boolean sequence (blank/non-blank flags) into non-blank segments at gaps of a specified minimum length.

---

### XH2: Title Row Detection (`_detect_title_row`)

**Problem**: Some tables have a title in row 0 (e.g., "Employee Performance") that spans the full width but is not a column header.

**Solution**: If row 0 has exactly one non-empty cell and `header_count > 1`, extract it as the table title. Exclude it from header rows and store in `metadata["title"]`.

**Mirrors**: DOCX's DH3 (title row detection in `compress_docx_tables`).

---

### XH3: Hidden Content Filtering (`_get_hidden_cols`, `_get_hidden_rows`)

**Problem**: Authors hide columns (intermediate formulas, sensitive data) or rows (draft content). These are present in the file but not visible in the spreadsheet.

**Solution**: Before building the grid, check `ws.column_dimensions[letter].hidden` and `ws.row_dimensions[row].hidden`. Exclude hidden rows and columns entirely from the output grid.

**Controllable**: `filter_hidden=False` on `extract_tables_from_xlsx()` disables this behavior.

**Analogy**: This is the XLSX equivalent of PDF page-clipping — content exists in the file but the author chose not to show it.

---

### XH4: Number Format Interpretation (`_detect_number_format_hint`)

**Problem**: A cell with value `45678` could be a number, a date (Excel serial date), or a currency amount. The cell's `number_format` string distinguishes them.

**Solution**: Inspect `cell.number_format` to classify as:
- `"date"` — contains `yyyy`, `mm`, `dd` patterns
- `"currency"` — contains `$`, `€`, `£`, `¥` symbols
- `"percentage"` — contains `%`
- `None` — `"General"` or unrecognized format

Format hints are stored in `metadata["format_hints"]` as a `dict[int, str]` mapping column indices to hint strings. This metadata is informational — it does not alter the extracted values, but downstream consumers can use it for type coercion.

---

### Merge-Based Header Detection (`_estimate_header_rows_from_merges`)

**Problem**: Generic header estimators (`estimate_header_rows`, `estimate_header_rows_from_types`) can miss or under-count headers in XLSX files with merged cells.

**Solution**: Scan `ws.merged_cells.ranges` for **horizontal** merges (spanning multiple columns) in the first `max_scan` rows. Find the last such merge row, then continue scanning with type-pattern analysis (TH2: all-string rows) to catch non-merged header rows below the merges (e.g., month labels below quarter merges).

**Why horizontal only?** Vertical-only merges (spanning rows but one column) indicate row-label grouping in the data body, not header structure. Including them would over-count headers in cross-tab tables.

---

### Forward-Fill Compound Headers (`_build_column_names_with_forward_fill`)

**Problem**: Merged header cells leave empty cells in the grid. For example, "Revenue" merged over columns 2-3 produces `["Revenue", ""]`. The sub-header row has `["Q1", "Q2"]`. Without forward-fill, column 3's header would be just `"Q2"` instead of `"Revenue / Q2"`.

**Solution**: For each header row independently, propagate the last non-empty cell value into subsequent empty cells. Then stack rows with ` / ` separator and deduplicate consecutive identical fragments.

**Result**: `["Department", "Revenue / Q1", "Revenue / Q2", "Cost / Q1", "Cost / Q2"]`

---

## API

### `extract_tables_from_xlsx()`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `xlsx_path` | `str \| Path` | required | Path to .xlsx file |
| `sheets` | `list[str \| int] \| None` | `None` | Sheet names or indices to process (None = all) |
| `extract_styles` | `bool` | `True` | Extract visual style info (fills, borders, fonts) |
| `filter_hidden` | `bool` | `True` | XH3: Exclude hidden rows and columns |

Returns `list[StructuredTable]`.

Metadata fields per table:

| Field | Type | Description |
| --- | --- | --- |
| `page_number` | `int` | Sheet index (0-based) |
| `table_index` | `int` | Table position within the sheet |
| `sheet_name` | `str` | Sheet name |
| `title` | `str` | XH2: Extracted title (if detected) |
| `format_hints` | `dict[int, str]` | XH4: Column index → format hint |

### `extract_tables_from_excel()`

Auto-detects `.xlsx` vs `.xls` format and dispatches to the appropriate extractor. Same parameters as `extract_tables_from_xlsx()`.

### `xlsx_to_markdown()`

Convenience function for quick inspection. Extracts tables and renders as pipe-table markdown with `## Sheet: <name>` headers.

---

## Visual Style Extraction

When `extract_styles=True`, per-cell visual information is captured:

| Attribute | Source | Heuristic |
| --- | --- | --- |
| Fill color | `cell.fill.fgColor` | VH2: Header fill detection |
| Borders | `cell.border.*` | VH1: Grid structure |
| Bold | `cell.font.bold` | VH2: Header emphasis |
| Italic | `cell.font.italic` | Metadata detection |
| Font size | `cell.font.size` | Hierarchy detection |
| Font color | `cell.font.color` | VH5: Conditional formatting |

Visual info feeds into `_analyze_visual_structure()` which detects:
- **VH1**: Grid borders (>50% of cells have borders)
- **VH2**: Header fill rows (first N rows with consistent fill color)
- **VH3**: Zebra striping (alternating row colors in data zone)
- **VH4**: Section separators (rows with thick bottom borders)

---

## Testing

### XLSX tests (`tests/test_xlsx_extractor.py`)

52 tests covering the XLSX extraction pipeline:

| Test Group | Count | Verifies |
| --- | --- | --- |
| P1: Tidy Analyst | 7 | Basic extraction, headers, types, multi-sheet, sheet filter |
| P2: Merger | 5 | Compound headers, deep hierarchy, cross-tab, irregular merges |
| P3: Multi-Tasker | 7 | Blank-row split, titles, footnotes, side-by-side, indices |
| P4: Formatter | 6 | KPI dashboard, banded report, hidden cols, date formats |
| XH1 unit tests | 6 | `_split_by_blank_runs` edge cases |
| XH2 unit tests | 3 | Title detection conditions |
| Forward-fill unit tests | 6 | Single/multi-row, dedup, hierarchy |
| XH4 unit tests | 6 | Number format classification |
| Markdown rendering | 2 | `xlsx_to_markdown` output |
| Smoke tests | 2 | All 16 fixtures extract with data |
| Error handling | 2 | FileNotFoundError, invalid sheet names |

### Synthetic fixtures (`inputs/xlsx/synthetic/`)

16 programmatically generated .xlsx files covering 4 persona archetypes:

| Persona | Files | Structural Focus |
| --- | --- | --- |
| P1: Tidy Analyst | 4 | Clean tables, mixed types, multi-sheet |
| P2: The Merger | 4 | 2-row, 3-row, cross-tab, irregular merges |
| P3: The Multi-Tasker | 4 | Blank-row gaps, titles, footnotes, side-by-side |
| P4: The Formatter | 4 | Conditional formatting, zebra rows, hidden cols, date formats |

Generator: `tests/generate_synthetic_xlsx.py`

```bash
uv run python tests/generate_synthetic_xlsx.py
uv run pytest tests/test_xlsx_extractor.py -v
```
