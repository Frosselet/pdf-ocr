# pdf_ocr — Structured Document Data Extraction

Extract structured tabular data from PDFs, Excel, Word, PowerPoint, and HTML using spatial text analysis and LLM interpretation.

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
 ├─► retrieval.py ───► Fast metadata extraction (dates, currency, units)
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

## Search & Extract (Metadata + Tables)

Extract document metadata (dates, currency, units) before LLM interpretation:

```python
from pdf_ocr import (
    search_and_extract,
    CanonicalSchema,
    ColumnDef,
    MetadataFieldDef,
    MetadataCategory,
    SearchZone,
    FallbackStrategy,
)

# Define schema with metadata fields
schema = CanonicalSchema(
    columns=[
        ColumnDef("vessel", "string", "Vessel name"),
        ColumnDef("quantity", "int", "Tonnage"),
    ],
    metadata=[
        MetadataFieldDef(
            name="publication_date",
            category=MetadataCategory.TEMPORAL,
            required=True,
            zones=[SearchZone.TITLE_PAGE, SearchZone.PAGE_HEADER],
            patterns=[r"As of\s+(.+?\d{4})"],
            fallback=FallbackStrategy.DEFAULT,
            default="2025-01-01",
        ),
        MetadataFieldDef(
            name="currency",
            category=MetadataCategory.TABLE_CONTEXT,
            zones=[SearchZone.ANYWHERE],
        ),
    ],
)

# Extract metadata + tables in one call
result = search_and_extract("document.pdf", schema)

# Check validation
print(f"Validation passed: {result.validation.passed}")
print(f"Missing fields: {result.validation.missing}")

# Access metadata
for name, meta in result.metadata.items():
    print(f"{name}: {meta.value} (confidence: {meta.confidence})")

# Access tables (same as interpret_table result)
for page, table in result.tables.items():
    print(f"Page {page}: {len(table.records)} records")
```

### Metadata Categories

| Category | Description | Built-in Patterns |
|----------|-------------|-------------------|
| `TEMPORAL` | Dates, periods, fiscal years | "As of", "For the year ended", "Q1 2025", "FY2025" |
| `TABLE_CONTEXT` | Units, currency, scale | "(in millions)", "USD", "$", "metric tons" |
| `ENTITY` | Company names, identifiers | (custom patterns only) |
| `TABLE_IDENTITY` | Table titles, section names | (custom patterns only) |
| `FOOTNOTE` | Footnotes, disclaimers | Superscripts, [1], *, † |

### Search Zones

| Zone | Description |
|------|-------------|
| `TITLE_PAGE` | First page, top 40% |
| `PAGE_HEADER` | Top 15% of any page |
| `PAGE_FOOTER` | Bottom 15% of any page |
| `TABLE_CAPTION` | Text above table |
| `COLUMN_HEADER` | Within table headers |
| `TABLE_FOOTER` | Below table |
| `ANYWHERE` | Full document scan |

### Fallback Strategies

| Strategy | Behavior |
|----------|----------|
| `DEFAULT` | Use `MetadataFieldDef.default` value |
| `FLAG` | Return with `value=None`, let caller handle |
| `PROMPT` | Signal that user input is needed |
| `INFER` | (Future) LLM-based inference |

---

## Module Documentation

Each module has detailed documentation including all heuristics:

| Module | Description | Documentation |
|---|---|---|
| `spatial_text.py` | Monospace grid rendering | [SPATIAL_TEXT.md](SPATIAL_TEXT.md) |
| `compress.py` | Token-efficient compression | [COMPRESS.md](COMPRESS.md) |
| `filter.py` | Fuzzy title filtering | [FILTER.md](FILTER.md) |
| `retrieval.py` | Fast metadata extraction | (see Search & Extract section) |
| `interpret.py` | LLM schema mapping | [INTERPRET.md](INTERPRET.md) |
| `serialize.py` | Export to various formats | [SERIALIZE.md](SERIALIZE.md) |
| `heuristics.py` | Shared semantic heuristics | (see below) |
| `xlsx_extractor.py` | Excel extraction | (see Multi-Format section) |
| `docx_extractor.py` | Word extraction | (see Multi-Format section) |
| `pptx_extractor.py` | PowerPoint extraction | (see Multi-Format section) |
| `html_extractor.py` | HTML extraction | (see Multi-Format section) |

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

### Metadata Retrieval (`retrieval.py`)

> *Extracting document metadata before LLM interpretation*

| Human Inference | Heuristic | Implementation |
|---|---|---|
| "As of December 31, 2025" | RH1: Temporal patterns | Regex for dates, periods, fiscal years |
| "(in millions)" | RH4: Unit patterns | Regex for scale, currency, units |
| "¹ See footnote below" | RH5: Footnote markers | Superscripts, [1], *, † detection |
| "Date is in page header" | Zone-based search | Top 15% = PAGE_HEADER zone |
| "Required field missing" | Validation gate | Required flag + fallback strategies |

---

## Multi-Format Document Extraction

Beyond PDFs, this library extracts tables from Excel, Word, PowerPoint, and HTML documents using the same semantic heuristics.

### Supported Formats

| Format | Extension | Library | Notes |
|--------|-----------|---------|-------|
| **PDF** | `.pdf` | PyMuPDF | Full spatial + visual analysis |
| **Excel 2007+** | `.xlsx` | openpyxl | Merged cells, styles, multi-sheet |
| **Excel 97-2003** | `.xls` | xlrd | Legacy format support |
| **Word 2007+** | `.docx` | python-docx | gridSpan, vMerge handling |
| **Word 97-2003** | `.doc` | LibreOffice | Converts to .docx first |
| **PowerPoint 2007+** | `.pptx` | python-pptx | Table shapes + text clustering |
| **PowerPoint 97-2003** | `.ppt` | LibreOffice | Converts to .pptx first |
| **HTML** | `.html` | selectolax | colspan, rowspan, inline styles |

### Quick Examples

#### Excel (XLSX/XLS)

```python
from pdf_ocr import extract_tables_from_excel

# Auto-detects format (.xlsx or .xls)
tables = extract_tables_from_excel("report.xlsx")

for table in tables:
    print(f"Sheet: {table.metadata['sheet_name']}")
    print(f"Headers: {table.column_names}")
    print(f"Rows: {len(table.data)}")

    for row in table.data[:3]:  # First 3 rows
        print(row)
```

#### Word (DOCX/DOC)

```python
from pdf_ocr import extract_tables_from_word

# Auto-detects format (.docx or .doc)
# Note: .doc requires LibreOffice installed
tables = extract_tables_from_word("document.docx")

for i, table in enumerate(tables):
    print(f"Table {i + 1}: {len(table.column_names)} columns, {len(table.data)} rows")
```

#### PowerPoint (PPTX/PPT)

```python
from pdf_ocr import extract_tables_from_powerpoint

# Extracts both table shapes and text-box tables
tables = extract_tables_from_powerpoint("presentation.pptx")

for table in tables:
    slide = table.metadata.get("slide_number", 0) + 1
    from_shapes = table.metadata.get("from_shapes", False)
    source = " (reconstructed from text boxes)" if from_shapes else ""
    print(f"Slide {slide}{source}: {table.column_names}")
```

#### HTML

```python
from pdf_ocr import extract_tables_from_html

# From file
tables = extract_tables_from_html("page.html")

# From string
html_content = """
<table>
  <thead><tr><th>Name</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Item A</td><td>100</td></tr>
    <tr><td>Item B</td><td>200</td></tr>
  </tbody>
</table>
"""
tables = extract_tables_from_html(html_content)

for table in tables:
    print(f"Columns: {table.column_names}")
    print(f"Data: {table.data}")
```

### Unified StructuredTable Output

All extractors return `list[StructuredTable]` with consistent structure:

```python
@dataclass
class StructuredTable:
    column_names: list[str]      # Header names (stacked if multi-row)
    data: list[list[str]]        # Data rows
    source_format: str           # "pdf", "xlsx", "xls", "docx", "doc", "pptx", "ppt", "html"
    metadata: dict | None        # Format-specific info (sheet_name, slide_number, etc.)
```

### Legacy Format Support

#### Excel 97-2003 (.xls)

Uses `xlrd` library directly — no conversion needed:

```python
from pdf_ocr import extract_tables_from_excel

# Works with both formats
tables = extract_tables_from_excel("legacy_report.xls")
```

#### Word/PowerPoint 97-2003 (.doc/.ppt)

Requires **LibreOffice** installed for automatic conversion:

```bash
# macOS
brew install --cask libreoffice

# Ubuntu/Debian
sudo apt install libreoffice

# Windows
# Download from https://www.libreoffice.org/download/
```

The library auto-detects LibreOffice location:
- macOS: `/Applications/LibreOffice.app/Contents/MacOS/soffice`
- Linux: `/usr/bin/libreoffice` or `/usr/bin/soffice`
- Windows: `C:\Program Files\LibreOffice\program\soffice.exe`

```python
from pdf_ocr import extract_tables_from_word

# Automatically converts .doc → .docx, then extracts
tables = extract_tables_from_word("legacy_document.doc")
```

### Shared Semantic Heuristics

All format extractors use the same semantic heuristics from `heuristics.py`:

| Heuristic | Description | Applies To |
|-----------|-------------|------------|
| **TH1** | Cell type detection (date/number/string) | All formats |
| **TH2** | Header type pattern (all-string row = header) | All formats |
| **TH3** | Column type consistency | All formats |
| **H7** | Header row estimation (bottom-up analysis) | All formats |

Visual heuristics are format-specific:
- **PDF**: Line/fill extraction from drawings
- **XLSX**: Cell.fill, Cell.border, Cell.font
- **DOCX**: cell.shading, table borders
- **PPTX**: Shape fill/stroke
- **HTML**: Inline styles, `<thead>` detection

### Specific Format Extractors

If you only need one format, use the specific extractors:

```python
# Modern formats only
from pdf_ocr import (
    extract_tables_from_xlsx,  # .xlsx only
    extract_tables_from_docx,  # .docx only
    extract_tables_from_pptx,  # .pptx only
    extract_tables_from_html,  # .html
)

# Unified extractors (auto-detect old/new)
from pdf_ocr import (
    extract_tables_from_excel,      # .xlsx + .xls
    extract_tables_from_word,       # .docx + .doc
    extract_tables_from_powerpoint, # .pptx + .ppt
)
```

---

## Installation

```bash
pip install pdf-ocr

# Optional dependencies for multi-format support
pip install pdf-ocr[xlsx]        # Excel 2007+ (.xlsx)
pip install pdf-ocr[xls]         # Excel 97-2003 (.xls)
pip install pdf-ocr[excel]       # Both Excel formats
pip install pdf-ocr[docx]        # Word (.docx, .doc with LibreOffice)
pip install pdf-ocr[pptx]        # PowerPoint (.pptx, .ppt with LibreOffice)
pip install pdf-ocr[html]        # HTML tables
pip install pdf-ocr[formats]     # All document formats
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

### PDF Functions

| Function | Input | Output |
|---|---|---|
| `pdf_to_spatial_text()` | PDF | Monospace text grid |
| `compress_spatial_text()` | PDF | Markdown tables + text |
| `compress_spatial_text_structured()` | PDF | `list[StructuredTable]` |
| `filter_pdf_by_table_titles()` | PDF + search terms | Filtered PDF bytes |
| `search_and_extract()` | PDF + schema with metadata | `SearchExtractResult` |
| `quick_scan()` | PDF + metadata fields | `dict[str, RetrievedMetadata]` |
| `interpret_table()` | Compressed text + schema | `dict[int, MappedTable]` |
| `to_csv()` / `to_parquet()` / `to_pandas()` | Result + schema | Various formats |

### Multi-Format Extractors

| Function | Input | Output |
|---|---|---|
| `extract_tables_from_excel()` | .xlsx/.xls path | `list[StructuredTable]` |
| `extract_tables_from_word()` | .docx/.doc path | `list[StructuredTable]` |
| `extract_tables_from_powerpoint()` | .pptx/.ppt path | `list[StructuredTable]` |
| `extract_tables_from_html()` | .html path or string | `list[StructuredTable]` |
| `extract_tables_from_xlsx()` | .xlsx path only | `list[StructuredTable]` |
| `extract_tables_from_docx()` | .docx path only | `list[StructuredTable]` |
| `extract_tables_from_pptx()` | .pptx path only | `list[StructuredTable]` |

### Shared Heuristic Functions

| Function | Description |
|---|---|
| `detect_cell_type()` | Classify cell as DATE, NUMBER, ENUM, or STRING |
| `detect_column_types()` | Get predominant type for each column |
| `is_header_type_pattern()` | Check if row is all-strings (header-like) |
| `estimate_header_rows()` | Estimate header count from grid structure |
| `detect_temporal_patterns()` | RH1: Find dates, periods, fiscal years in text |
| `detect_unit_patterns()` | RH4: Find scale, currency, units in text |
| `detect_footnote_markers()` | RH5: Find superscripts, [1], *, † markers |

### Types

| Type | Description |
|---|---|
| `StructuredTable` | Table with metadata, column_names, data |
| `GenericStructuredTable` | Cross-format StructuredTable from heuristics.py |
| `TableMetadata` | page_number, table_index, section_label, sheet_name |
| `CellType` | Enum: DATE, NUMBER, ENUM, STRING |
| `CanonicalSchema` | Schema definition with columns and optional metadata fields |
| `ColumnDef` | Column name, type, description, aliases, format |
| `MetadataFieldDef` | Metadata field: name, category, zones, patterns, fallback |
| `MetadataCategory` | Enum: TEMPORAL, ENTITY, TABLE_IDENTITY, TABLE_CONTEXT, FOOTNOTE |
| `SearchZone` | Enum: TITLE_PAGE, PAGE_HEADER, PAGE_FOOTER, etc. |
| `FallbackStrategy` | Enum: INFER, DEFAULT, PROMPT, FLAG |
| `RetrievedMetadata` | Extracted metadata value with confidence |
| `SearchExtractResult` | Combined result: metadata + tables + validation |
| `ValidationResult` | Validation status: passed, found, missing |
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
