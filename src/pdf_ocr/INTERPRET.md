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

### H3: Step-2 Batching (`_create_batches`)

**Problem**: Dense pages with 78+ data rows cause LLM output truncation.

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

### H4: Context Inference

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

### H5: Vision Cross-Validation

**Problem**: Vision-guided parsing might regress quality if vision infers wrong column count.

**Solution**: Post-step-1 validation:

```python
if vision_column_count != len(parsed_row.cells):
    # Vision guidance failed — fall back to non-vision parsing
    reparse without vision guidance
```

**Effect**: Vision only helps, never hurts.

---

### H6: Multi-Page Concurrency

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

### H7: Fallback Model Support

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
