# pdf_ocr — Structured PDF Data Extraction

Extract structured tabular data from PDFs using spatial text analysis and LLM interpretation.

---

## Holistics: Why This Matters

### The Fundamental Gap

Data produced by humans **for humans** is fundamentally different from data produced for machines.

A PDF is the epitome of human-centric data. It preserves:
- **Visual layout** over logical structure
- **Spatial positioning** as semantic meaning
- **Implicit relationships** that human pattern recognition decodes effortlessly

When a human looks at a shipping stem, they instantly see: these columns belong together, this row is a total, that text is a section header. The brain performs dozens of micro-inferences based on position, whitespace, font size, alignment, and context.

Machines see: a stream of text fragments with x,y coordinates.

### The Translation Challenge

This project is not about replacing human interpretation. It's about **encoding the implicit rules** humans use to parse visual data layouts.

Every heuristic in this library represents a discovered pattern:

| Human Inference | Encoded Heuristic |
|---|---|
| "These values are on the same line" | Row clustering within 2pt y-distance |
| "This is a table, not a paragraph" | 3+ rows sharing 2+ aligned column anchors |
| "GERALDTON is a section header" | Single short span that's not numeric |
| "These columns go together" | Bounding-box overlap between headers and data |
| "This row is a subtotal" | Single-span row with numeric content |

### The Long-Term Vision

We start with **holistics** (the philosophy of why) and **heuristics** (practical pattern encodings). This foundation allows us to eventually:

1. **Refine** — as we encounter more edge cases, heuristics become more precise
2. **Generalize** — patterns that work across document types become candidates for axioms
3. **Formalize** — axioms combine into an ontology of human-to-machine data translation

The goal is not a perfect PDF parser. The goal is a **universal understanding** of how humans structure information visually and how to translate that into machine-retro-readable data.

This is the difference between:
- **Naive prompting**: "Extract the table from this PDF" (hoping GenAI figures it out)
- **Principled extraction**: Understanding *why* humans lay out data this way, and encoding that understanding

---

## The Technical Problem

Standard PDF text extraction returns text in stream order — the order objects were written into the PDF. This rarely matches visual layout. Columns get interleaved, tables merge, spatial relationships are lost.

## The Solution

A multi-stage pipeline that preserves spatial structure:

```
PDF
 │
 ├─► spatial_text.py ──► Monospace grid preserving exact positions
 │
 ├─► compress.py ────► Token-efficient markdown tables (40-67% smaller)
 │
 ├─► filter.py ──────► Filter pages by fuzzy title matching
 │
 ├─► interpret.py ───► LLM-powered schema mapping
 │
 └─► serialize.py ───► Export to CSV/Parquet/DataFrames
```

## Quick Start

```python
from pdf_ocr import compress_spatial_text, interpret_table, CanonicalSchema, ColumnDef, to_csv

# 1. Compress PDF to structured text
compressed = compress_spatial_text("document.pdf")

# 2. Define what you want to extract
schema = CanonicalSchema(columns=[
    ColumnDef("vessel_name", "string", "Vessel name", aliases=["Ship Name", "Vessel"]),
    ColumnDef("port", "string", "Loading port", aliases=["Port"]),
    ColumnDef("quantity", "int", "Tonnes", aliases=["Quantity", "Tonnage"]),
    ColumnDef("eta", "date", "Arrival date", aliases=["ETA"], format="YYYY-MM-DD"),
])

# 3. Extract and map to schema
result = interpret_table(compressed, schema)

# 4. Export
csv_str = to_csv(result, schema)
```

## Structured Output

For programmatic access to tables without LLM interpretation:

```python
from pdf_ocr import compress_spatial_text_structured

tables = compress_spatial_text_structured("document.pdf")

for table in tables:
    print(f"Section: {table.metadata.section_label}")
    print(f"Columns: {table.column_names}")
    for row in table.data:
        print(row)
```

---

## Module Documentation

Each module has detailed documentation including all heuristics:

| Module | Description | Documentation |
|---|---|---|
| `spatial_text.py` | Monospace grid rendering | [SPATIAL_TEXT.md](SPATIAL_TEXT.md) |
| `compress.py` | Token-efficient compression | [COMPRESS.md](COMPRESS.md) |
| `filter.py` | Fuzzy title filtering | [FILTER.md](FILTER.md) |
| `interpret.py` | LLM schema mapping | [INTERPRET.md](INTERPRET.md) |
| `serialize.py` | Export to various formats | [SERIALIZE.md](SERIALIZE.md) |

---

## Heuristics Catalog

Each heuristic encodes a human inference pattern. The table shows: what humans do instinctively → how we encode it.

### Spatial Reconstruction (`spatial_text.py`)

> *Translating visual positions into grid coordinates*

| Human Inference | Heuristic | Implementation |
|---|---|---|
| "Text size varies but columns align" | Dynamic cell width | Median character width across page |
| "These words are on the same line" | Row clustering | Greedy merge within 2pt y-distance |
| "Later text replaces earlier" | Last-writer-wins | Document order precedence |

### Structure Recognition (`compress.py`)

> *Identifying tables, headers, sections from spatial patterns*

| Human Inference | Heuristic | Implementation |
|---|---|---|
| "These cells were meant to be separate" | Span splitting | Cross-row column boundary detection |
| "This is a table, that's a paragraph" | Region classification | Span count + column anchor patterns |
| "Rows belong to same table despite gaps" | Table pool detection | 60% column overlap threshold |
| "GERALDTON is a section, 337,000 is data" | Section vs continuation | Numeric content detection |
| "These 3 rows are one record" | Multi-row detection | Repeating span-count patterns |
| "Headers end here, data starts here" | Header estimation | Bottom-up span-count analysis |
| "Those scattered rows are headers" | Preceding header detection | Upward alignment scanning |
| "This header goes with that column" | Alignment-aware matching | Bounding-box overlap |
| "Same column, different x-positions" | Column unification | Greedy clustering with running mean |
| "These columns are really one" | Twin column merging | Never-simultaneous fill detection |
| "Two tables side by side" | Horizontal gap detection | 3x median gap threshold |
| "Field names are rows, not columns" | Transposed detection | Low column count + stable first column |

### Visual Cue Detection (`compress.py`)

> *Cross-validating text heuristics with visual elements*

| Human Inference | Heuristic | Implementation |
|---|---|---|
| "Those lines form a table grid" | Grid boundary detection | H/V line intersection analysis |
| "The colored row is a header" | Header fill detection | Fill color in top rows |
| "Alternating colors = data rows" | Zebra striping | Alternating fill pattern detection |
| "That thick line is a section break" | Section separators | Wide horizontal lines |
| "Colored but has dates = data, not header" | Exception highlighting | Type pattern overrides visual color |

### Type Pattern Analysis (`compress.py`)

> *Distinguishing headers from data by content type*

| Human Inference | Heuristic | Implementation |
|---|---|---|
| "Headers are labels, not values" | Header type pattern | All-string row = likely header |
| "This column always has dates" | Column type consistency | Predominant type per column |
| "YYYY-MM-DD is a date" | Cell type detection | Pattern matching for dates/numbers |
| "Few repeating values = enum" | Enum detection | ≤5 distinct values that repeat |

### Semantic Mapping (`interpret.py`)

> *Connecting extracted structure to application schemas*

| Human Inference | Heuristic | Implementation |
|---|---|---|
| "This is a pivot table" | Table type classification | Structural pattern matching |
| "The LLM miscounted section rows" | Section boundary validation | Deterministic pipe-row counting |
| "Too many rows for one LLM call" | Step-2 batching | Section-aware chunking |
| "Port name is in the section header" | Context inference | Schema column without aliases |
| "Vision made it worse, not better" | Vision cross-validation | Column count verification |

---

## Installation

```bash
pip install pdf-ocr

# Optional dependencies
pip install pdf-ocr[dataframes]  # pandas + polars
pip install pdf-ocr[parquet]     # pyarrow
pip install pdf-ocr[all]         # everything
```

## Environment Variables

```bash
export OPENAI_API_KEY="..."     # For GPT-4o interpretation
export ANTHROPIC_API_KEY="..."  # For Claude (if configured in BAML)
```

---

## Public API Summary

### Core Functions

| Function | Input | Output |
|---|---|---|
| `pdf_to_spatial_text()` | PDF | Monospace text grid |
| `compress_spatial_text()` | PDF | Markdown tables + text |
| `compress_spatial_text_structured()` | PDF | `list[StructuredTable]` |
| `filter_pdf_by_table_titles()` | PDF + search terms | Filtered PDF bytes |
| `interpret_table()` | Compressed text + schema | `dict[int, MappedTable]` |
| `to_csv()` / `to_parquet()` / `to_pandas()` | Result + schema | Various formats |

### Types

| Type | Description |
|---|---|
| `StructuredTable` | Table with metadata, column_names, data |
| `TableMetadata` | page_number, table_index, section_label |
| `CanonicalSchema` | Schema definition with columns |
| `ColumnDef` | Column name, type, description, aliases, format |
| `FilterMatch` | Page, title, search term, score |

---

## Limitations

- **Monospace assumption**: Single character width per page
- **No images**: Only text extracted
- **Vision cost**: Extra LLM call per page when using `pdf_path=`
- **Section detection**: Requires ≥2 sections with aggregation rows

---

## BAML

This project uses [BAML](https://docs.boundaryml.com/) for LLM function definitions. After editing files in `baml_src/`:

```bash
uv run baml-cli generate
```
