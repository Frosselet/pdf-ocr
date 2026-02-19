# Table Interpretation

> **Module**: `interpret.py`
> **Public API**: `interpret_table()`, `interpret_table_single_shot()`, `to_records()`

LLM-powered pipeline that extracts structured tabular data from compressed PDF text and maps it to a user-defined canonical schema.

## Why a Canonical Schema?

Source PDFs use inconsistent column names:
- Document A: "Ship Name"
- Document B: "Vessel"
- Document C: "Vessel Name"

A canonical schema lets you declare the columns your application expects, along with aliases the LLM should recognize:

```python
schema = CanonicalSchema(columns=[
    ColumnDef("vessel_name", "string", "Name of the vessel",
              aliases=["Ship Name", "Vessel", "Vessel Name"]),
])
```

## Pipeline Modes

### 2-Step (Default)

```
compressed text
      │
      ▼
normalize_pipe_table()          ← NBSP, smart quotes, dashes, whitespace
      │
      ▼
AnalyzeAndParseTable (GPT-4o)
      │
      ▼
ParsedTable (intermediate)
      │
      ▼
MapToCanonicalSchema (GPT-4o)
      │
      ▼
MappedTable (output)
```

**Step 1**: Analyze structure, classify table type, extract headers, separate aggregation rows, parse all data rows.

**Step 2**: Map parsed columns to canonical columns using names, aliases, and descriptions.

### Single-Shot

```
compressed text + schema
      │
      ▼
normalize_pipe_table()          ← same normalization
      │
      ▼
InterpretTable (GPT-4o)
      │
      ▼
MappedTable (output)
```

One LLM call. Faster but less accurate on complex tables.

### Vision-Guided

```
page image + compressed text
      │
      ▼
normalize_pipe_table()          ← same normalization
      │
      ▼
InferTableSchemaFromImage (GPT-4o vision)
      │
      ▼
InferredTableSchema (column names from image)
      │
      ▼
AnalyzeAndParseTableGuided (GPT-4o)
      │
      ▼
ParsedTable → MapToCanonicalSchema → MappedTable
```

For PDFs with garbled/concatenated headers. Vision reads correct headers from page image.

---

## Heuristics

### H1: Table Type Classification

**Problem**: Different table structures require different mapping strategies.

**Solution**: Classify into 5 types (handled in Step 1):

| Type | Description | Mapping Strategy |
|---|---|---|
| `FlatHeader` | Standard headers in top row(s) | 1:1 row mapping |
| `HierarchicalHeader` | Tree-structured headers (parents span children) | 1:1 or unpivot |
| `PivotedTable` | Cross-tabulation (categories × periods) | Unpivot |
| `TransposedTable` | Field names as rows, records as columns | Transpose |
| `Unknown` | Ambiguous | Best-effort |

**Effect**: Step 2 knows whether to produce one output record per row, or multiple.

---

### H2: Section Boundary Validation (`_validate_section_boundaries`)

**Problem**: LLMs often miscount rows in long sectioned tables:
- Says "KWINANA (rows 10-27)" but it's actually rows 10-28
- Off-by-one cascades to all subsequent sections

**Solution**: Deterministic cross-validation:

1. **Parse LLM's claimed boundaries** from notes: `"Sections: GERALDTON (rows 0-9), KWINANA (rows 10-27)"`
2. **Count pipe-table rows** directly from compressed text (`_detect_sections_from_text`)
3. **If sums match**: Replace LLM boundaries with text-detected counts
4. **If not**: Fall back to LLM boundaries

```python
# Text detection walks through:
#   SECTION_HEADER     → last_header = "GERALDTON"
#   |---|---|...|      → in_data = True
#   |data row 1|       → count++
#   |data row 2|       → count++
#   ||...total...|     → aggregation row, don't count
#
#   KWINANA           → flush previous section, start new
```

---

### H3: Step-1 Pre-Splitting (`_split_pipe_table`)

**Problem**: Large tables (65+ data rows) cause LLM output truncation in Step 1 (`AnalyzeAndParseTable`). Step 1 processes the entire table in a single call, and with many rows the model runs out of output tokens and drops/garbles tail rows.

**Solution**: Pre-split large tables into chunks of `step1_max_rows` (default 40) rows *before* Step 1:

1. **Parse pipe-table structure**: Identify preamble (title, headers, separator), data rows, section labels, aggregation rows
2. **Split at section boundaries** when possible (natural break points)
3. **Split within sections** at `step1_max_rows` boundaries when sections are too large
4. **Exclude aggregation rows** from all chunks (Step 1 ignores them anyway)
5. **Preserve preamble** in every chunk so the LLM can understand the table structure
6. **Run Step 1 concurrently** on all chunks via `asyncio.gather()`
7. **Merge results** back into a single `ParsedTable` — concatenate `data_rows`, rebuild section boundaries with global row indices

```python
# Before: single call truncates at row ~55
parsed = analyze_and_parse(big_table)  # 67 rows → garbled tail

# After: two manageable calls, merged seamlessly
chunks = _split_pipe_table(big_table, 40)  # → [chunk_40_rows, chunk_27_rows]
parsed_chunks = await gather(*(analyze_and_parse(c) for c in chunks))
parsed = _merge_parsed_chunks(parsed_chunks)  # → 67 rows, all correct
```

**Default**: `step1_max_rows=40` — well below the ~65-row truncation threshold, higher than Step-2's `batch_size=20` since Step 1 output per row is smaller.

---

### H4: Step-2 Batching (`_create_batches`)

**Problem**: Dense pages with 78+ data rows cause LLM output truncation in Step 2.

**Solution**: Split into batches of `batch_size` (default 20) rows:

1. **Section-aware splitting**: When sections exist, prefer splits at section boundaries
2. **Large section splitting**: Sections larger than `batch_size` split internally
3. **Batch-local notes**: Each batch gets re-indexed section boundaries

```python
# Original: "Sections: GERALDTON (rows 0-9), KWINANA (rows 10-35)"
# Batch 0 (rows 0-19): "Sections: GERALDTON (rows 0-9), KWINANA (rows 10-19)"
# Batch 1 (rows 20-35): "Sections: KWINANA (rows 0-15)"
```

**All batches run concurrently** via async.

---

### H5: Context Inference

**Problem**: Not all schema fields are table columns. Some values live in:
- Section headers ("GERALDTON", "KWINANA")
- Document titles
- Surrounding text

**Solution**: When a canonical column has no matching source column, the LLM looks in context:

```python
ColumnDef("port", "string",
          "Loading port name — may appear as section header, not column",
          aliases=[])  # empty aliases = look in context
```

**Effect**: Port name inferred from section header, applied to all rows in that section.

---

### H6: Vision Cross-Validation

**Problem**: Vision-guided parsing might regress quality if vision infers wrong column count.

**Solution**: Post-step-1 validation:

```python
if vision_column_count != len(parsed_row.cells):
    # Vision guidance failed — fall back to non-vision parsing
    reparse without vision guidance
```

**Effect**: Vision only helps, never hurts.

---

### H7: Multi-Page Concurrency

**Problem**: Multi-page PDFs should be processed in parallel.

**Solution**: Auto-split on page separator (`\f`), process all pages concurrently:

```python
# Single-threaded for simple input
if len(pages) == 1 and len(rows) < 50:
    sync processing

# Otherwise async
pages = compressed.split("\f")
results = await asyncio.gather(*[process_page(p) for p in pages])
```

---

### H8: Text Normalization (`normalize_pipe_table`)

**Problem**: Different extractors (DOCX, HTML, XLSX, PPTX, PDF) produce inconsistent text: NBSP characters, smart quotes, en/em-dashes, zero-width characters, and double spaces. These cause alias matching failures — e.g., `"MOA Target\u00a0\u00a02025"` (NBSP + double space) doesn't match the contract alias `"MOA Target 2025"`.

**Solution**: A single normalization pass applied at the entry of every public interpretation and classification function:

| Normalization | Characters | Replacement |
|---|---|---|
| NBSP | U+00A0 | Regular space |
| Smart quotes | U+2018/2019, U+201C/201D | ASCII `'` and `"` |
| Dashes | U+2013 en-dash, U+2014 em-dash | Hyphen-minus `-` |
| Zero-width | U+200B, U+200C, U+200D, U+FEFF, U+2060 | Removed |
| Whitespace | Runs of 2+ spaces/tabs | Single space |
| Trailing | Trailing spaces per line | Stripped |

Applied automatically in: `interpret_table()`, `interpret_table_single_shot()`, `interpret_table_async()`, `interpret_tables_async()`, and `classify_tables()`.

**Idempotent**: Safe to call multiple times; already-clean text passes through unchanged.

---

### H9: Deterministic Mapping (`_parse_pipe_table_deterministic`, `_map_to_schema_deterministic`)

**Problem**: The 2-step LLM pipeline is expensive and unreliable for tables with compound ` / ` headers. The LLM shifts columns, misassigns groups, and adds latency/cost for work that is mechanically solvable when the schema has good aliases.

**Root insight**: The compressed pipe-table markdown is already fully structured. Headers, separators, data rows, section labels, and aggregation rows are all explicit syntactic patterns. When every header part matches a schema alias, the entire interpretation is a string-matching + record-building exercise.

**Solution**: Try deterministic first → LLM fallback only for unmatched columns:

```
compressed text + schema
      │
      ▼
_parse_pipe_table_deterministic()     ← string ops only, no LLM
      │
      ▼
_DeterministicParsed (title, headers, data_rows, sections, aggregation_rows)
      │
      ▼
_map_to_schema_deterministic()        ← alias matching, 5 phases
      │
      ├─ all parts matched ──► MappedTable (zero LLM calls)
      │
      └─ unmatched parts ────► fall through to LLM pipeline (H1-H8)
```

**Phase 1 — Alias matching**: Split each header on ` / `, match parts against schema aliases (case-insensitive). Two fallbacks: (1) **comma-suffix fallback** — when a full part has no alias match and contains a comma, the text before the first comma is tried (e.g., `"Area harvested,Th.ha."` → tries `"Area harvested"`); (2) **joined-form fallback** — when ALL parts of a compound header are individually unmatched, the space-joined form is tried (e.g., `"Quantity / (tonnes)"` → joins to `"Quantity (tonnes)"` which matches the alias). Handles stacked headers where a single column name was split across multiple rows.

**Phase 2 — Classification**: Dimension (`string`/`date` → value from header text) vs measure (`int`/`float` → value from cell). Single-part dimension columns → shared (cell-value). Compound label columns → last dimension is shared, earlier parts are constants.

**Phase 2.3 — Title-to-schema matching**: When the table title matches a string-type schema column's alias, that column is assigned as a constant dimension with the title text as its value for all records. Tries full-text match first (e.g., title `"RICE"` matches alias `"rice"`), then falls back to word-boundary substring matching (e.g., title `"Winter sowing of grains and grasses"` contains alias `"grains and grasses"`). Substring fallback requires aliases ≥3 characters and exactly one matching string-type column (ambiguity → skip). Only fires when the matched column isn't already resolved from headers. The title may come from an in-table title row or from the preceding paragraph in the DOCX body. This reduces the pool of unmatched string columns, enabling Phase 2.5 to fire in cases that would otherwise be ambiguous (e.g., both `Region` and `Crop` unmatched → title resolves `Crop`, leaving `Region` as the sole candidate for blank-header inference).

**Phase 2.5 — Blank-header text-column inference**: When exactly one column has a blank header with text data (>50% non-numeric cells), and exactly one string-type schema column has no header match (accounting for title-matched columns from Phase 2.3), they are assigned as a shared column. The "exactly one of each" constraint prevents ambiguous assignments. Doesn't fire for numeric columns, partially-matched columns, or when multiple candidates exist.

**Phase 3 — Unpivot group detection**: Dimension columns appearing with ≥2 distinct values across headers form group dimensions (e.g., `crop` has "spring crops" and "spring grain").

**Phase 4 — Record building**: For each data row × each group, build a record: shared column values from cells, group dimension values from header text, measure values from cells, section label from matching schema column.

**Phase 5 — MappedTable construction**: Build `MappedTable` with `model="deterministic"`, all field mappings with `confidence=High`.

**Integrated into all orchestrators**: `interpret_table()`, `interpret_table_single_shot()`, `_interpret_pages_batched_async()`, `_interpret_pages_async()`, `interpret_table_async()`, `interpret_tables_async()`. Deterministic mapping is skipped when vision mode is enabled (garbled headers need LLM).

**Concrete cases that are 100% deterministic**:
- Russian DOCX — compound headers with ` / `, all header parts covered by aliases
- ACEA PDF — hierarchical headers, all motorization types covered by aliases

---

### H10: Fallback Model Support

**Problem**: Primary model might fail (rate limit, network, unavailable).

**Solution**: Optional `fallback_model` parameter:

```python
result = interpret_table(text, schema,
                         model="openai/gpt-4o",
                         fallback_model="openai/gpt-4o-mini")
```

If primary fails, retry with fallback.

---

## API

### Basic Usage

```python
from pdf_ocr import compress_spatial_text, interpret_table, CanonicalSchema, ColumnDef, to_records

compressed = compress_spatial_text("document.pdf")

schema = CanonicalSchema(columns=[
    ColumnDef("port", "string", "Loading port name", aliases=["Port"]),
    ColumnDef("vessel_name", "string", "Vessel name", aliases=["Ship Name", "Vessel"]),
    ColumnDef("quantity_tonnes", "int", "Quantity in tonnes", aliases=["Quantity", "Tonnes"]),
])

result = interpret_table(compressed, schema)

for record in to_records(result):
    print(record)
```

### Vision-Guided

```python
result = interpret_table(compressed, schema, pdf_path="document.pdf")
```

### Single-Shot

```python
result = interpret_table_single_shot(compressed, schema)
```

### Custom Batch Size

```python
result = interpret_table(compressed, schema, batch_size=10)  # for very wide tables
```

### Async

```python
import asyncio
from pdf_ocr.interpret import interpret_tables_async

pages = compressed.split("\f")
result = asyncio.run(interpret_tables_async(pages, schema))
```

---

## Schema Design

### Do

- **Define structure, not business logic**
- **Use aliases for column name variations**: `aliases=["Ship Name", "Vessel"]`
- **Use description for structural guidance**: "may appear as section header"
- **Use aliases to trigger unpivoting**: when aliases match parts of hierarchical headers

### Don't

- **Don't put filtering logic in descriptions**: "only latest release" → filter downstream
- **Don't put aggregation logic**: "sum of regions" → post-process
- **Don't rely on LLM to deduplicate**: extract all, dedupe downstream

---

## Aliases: The Contract Between Schema and LLM

Aliases are **the primary mechanism** the LLM uses to match source columns to canonical
columns during Step 2 (MapToCanonicalSchema). Getting them right is critical — missing or
incomplete aliases are the #1 cause of mapping failures, NA values, and dropped data.

### How Alias Matching Works

The LLM receives your `CanonicalSchema` (column names, types, descriptions, **aliases**)
alongside the parsed table. For each source column, it checks:

1. Does the column name match any canonical column's **alias** (literal match)?
2. If no alias match, does the **description** suggest a match (semantic inference)?
3. If no match at all, omit the field (→ NULL/NA in output).

**Literal alias matching is far more reliable than description-based inference.**
Descriptions require the LLM to reason about intent; aliases are unambiguous signals.

### Critical Rule: Compound Headers and Unpivoting

When source tables have compound headers with ` / ` separators (e.g.,
`"Category A / Metric 1 / 2025"`), Step 2 splits each header on ` / ` and checks
whether the parts match aliases of different canonical columns. If they do, it
**unpivots**: each data row produces multiple output records.

**The rule (from the BAML prompt):**

> Only unpivot columns where **ALL** header parts match dimension aliases.
> If a column's header part does NOT match any alias, SKIP that column entirely.

This means: **every level of a compound header hierarchy must have a canonical column
with matching aliases.** A missing alias at any level causes the entire column to be
silently dropped.

### Example: What Goes Wrong Without Aliases

Source table (two crop groups side by side):

```
| Region    | spring crops / MOA Target 2025 | spring crops / 2025 | spring grain / MOA Target 2025 | spring grain / 2025 |
|-----------|--------------------------------|---------------------|--------------------------------|---------------------|
| North     | 1000                           | 950                 | 500                            | 480                 |
```

Schema with **incomplete** aliases:

```python
CanonicalSchema(columns=[
    ColumnDef("Region", "string", "Geographic region", aliases=["Region"]),
    ColumnDef("Area", "float", "Target area", aliases=["MOA Target"]),
    ColumnDef("Value", "float", "Actual sown area", aliases=["2025"]),
    ColumnDef("Crop", "string", "Crop name", aliases=[]),      # ← EMPTY!
    ColumnDef("Year", "int", "Reporting year", aliases=["2025"]),
])
```

**Result**: The LLM splits `"spring crops / MOA Target 2025"` into parts:
- `"spring crops"` → no alias match (Crop has `aliases=[]`) → **FAIL**
- `"MOA Target"` → matches Area ✓
- `"2025"` → matches Value/Year ✓

Since "ALL parts" must match, the column should be skipped entirely. In practice,
the LLM may partially comply — handling the first group (left side) through lenient
inference but dropping the second group (right side). This produces correct data for
"spring crops" but NA values for "spring grain".

### Fixed schema:

```python
CanonicalSchema(columns=[
    ColumnDef("Region", "string", "Geographic region", aliases=["Region"]),
    ColumnDef("Area", "float", "Target area", aliases=["MOA Target"]),
    ColumnDef("Value", "float", "Actual sown area", aliases=["2025"]),
    ColumnDef("Crop", "string", "Crop name",
              aliases=["spring crops", "spring grain"]),  # ← explicit group aliases
    ColumnDef("Year", "int", "Reporting year", aliases=["2025"]),
])
```

Now ALL parts of every compound header match an alias → consistent unpivoting for
both groups.

### Alias Checklist

When designing a canonical schema, verify:

| Check | Question | If No |
|-------|----------|-------|
| **Column name variations** | Does every source column name (or partial name) appear as an alias somewhere? | Add aliases for all known variations |
| **Compound header parts** | For ` / `-separated headers, does every segment match an alias on some canonical column? | Add the missing group/dimension aliases |
| **Pivot dimensions** | For year/period columns that get unpivoted, does both the value column AND the dimension column list the pivot values as aliases? | Use `{YYYY}` templates or static year aliases on both |
| **Group labels** | For side-by-side table groups (e.g., two crops in one table), are group labels listed as aliases on the grouping column? | Add all group labels as aliases |

### Year Template Aliases

For dynamic pivot values (years that change per document), use `{YYYY}` templates
in the contract instead of hardcoded values:

```json
{
  "name": "Year",
  "aliases": ["{YYYY}", "{YYYY-1}"]
}
```

The pipeline resolves these at runtime from document headers:
- `{YYYY}` → latest year found (e.g., "2025")
- `{YYYY-1}` → one year before (e.g., "2024")
- `{YYYY+1}` → one year after (e.g., "2026")

This keeps the contract future-proof without hardcoding specific years.

---

## Output

`interpret_table()` returns `dict[int, MappedTable]` keyed by 1-indexed page number.

### MappedTable

| Field | Type | Description |
|---|---|---|
| `records` | `list[MappedRecord]` | Extracted records |
| `unmapped_columns` | `list[str]` | Source columns not mapped |
| `mapping_notes` | `str \| None` | Ambiguous match notes |
| `metadata` | `InterpretationMetadata` | Model, table type, field mappings |

### InterpretationMetadata

| Field | Type | Description |
|---|---|---|
| `model` | `str` | Model used (e.g., "openai/gpt-4o") |
| `table_type_inference` | `TableTypeInference` | Table type + mapping strategy |
| `field_mappings` | `list[FieldMapping]` | Per-column mapping details |
| `sections_detected` | `list[str] \| None` | Section labels found |

### FieldMapping

| Field | Type | Description |
|---|---|---|
| `column_name` | `str` | Canonical column name |
| `source` | `str` | Where value came from |
| `rationale` | `str` | Why this mapping |
| `confidence` | `High \| Medium \| Low` | Mapping confidence |

---

## Table Types Detail

### FlatHeader

Standard layout. Headers in top row(s), data below.

```
| Name  | Date       | Qty  |
|-------|------------|------|
| Alice | 2025-01-15 | 100  |
| Bob   | 2025-01-16 | 200  |
```

**Mapping**: 1:1 (each row → one record)

### HierarchicalHeader

Tree-structured headers. Parent cells span children.

```
|        Sales        |
| Q1     | Q2     |
| Jan|Feb| Mar|Apr|
```

**Mapping**: 1:1 if aliases match full compound names. **Unpivot** if aliases match parts.

### PivotedTable

Cross-tabulation. Categories as rows, periods as columns.

```
| Category | Jan | Feb | Mar |
|----------|-----|-----|-----|
| Product A| 100 | 150 | 200 |
```

**Mapping**: Unpivot (each row → N records, one per value column)

### TransposedTable

Property sheet. Field names as rows.

```
| Field    | Value      |
|----------|------------|
| Name     | Alice      |
| Date     | 2025-01-15 |
| Quantity | 100        |
```

**Mapping**: Transpose (each column → one record)
