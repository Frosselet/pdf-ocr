// Table interpretation: extract structured tabular data from compressed PDF text
// and map it to a user-defined canonical schema.

// ─── Enums ───────────────────────────────────────────────────────────────────

enum TableType {
  FlatHeader
  HierarchicalHeader
  PivotedTable
  TransposedTable
  Unknown
}

enum AggregationType {
  Total
  Sum
  Min
  Max
  Average
  Count
  NoAggregation @alias("None")
}

enum Confidence {
  High   @description("Direct column match by name or alias")
  Medium @description("Inferred from context with reasonable certainty (section header, title, etc.)")
  Low    @description("Best guess — ambiguous source or weak contextual signal")
}

// ─── Intermediate pipeline types ─────────────────────────────────────────────

class HeaderInfo {
  levels int @description("Number of header levels (1 for flat, >1 for hierarchical)")
  names string[][] @description("Header names per level, outer = level, inner = columns")
}

class AggregationInfo {
  type AggregationType
  axis string @description("'row' or 'column' — the axis along which aggregation runs")
  labels string[] @description("Labels identifying aggregation rows/columns, e.g. ['Total', 'Grand Total']")
}

class ParsedTable {
  table_type TableType
  headers HeaderInfo
  aggregations AggregationInfo? @description("Present only if the table contains aggregation rows/columns")
  data_rows string[][] @description("All data rows (excluding headers and aggregation rows). Each inner array is one row of cell values as strings.")
  notes string? @description("Any caveats or observations about the table structure")
}

// ─── Canonical schema + output types ─────────────────────────────────────────

class ColumnDef {
  name string @description("Canonical column name")
  type string @description("Expected type: string, int, float, bool, date")
  description string @description("What this column represents")
  aliases string[] @description("Alternative names this column may appear as in source tables")
}

class CanonicalSchema {
  columns ColumnDef[]
  description string? @description("Optional description of what this schema represents")
}

class MappedRecord {
  @@dynamic
}

class FieldMapping {
  column_name string  @description("Canonical column name from the schema")
  source string       @description("Where the value came from, e.g. 'column: Vessel Name', 'section header', 'document title', 'inferred from context'")
  rationale string    @description("Brief explanation of why this mapping was chosen")
  confidence Confidence
}

class InterpretationMetadata {
  model string                  @description("Model that produced this result, e.g. 'openai/gpt-4o'")
  field_mappings FieldMapping[] @description("One entry per canonical column, explaining how it was resolved")
  sections_detected string[]?   @description("Section/group labels found in the text, if any (e.g. ['GERALDTON', 'KWINANA'])")
}

class MappedTable {
  records MappedRecord[] @description("Mapped data records conforming to the canonical schema")
  unmapped_columns string[] @description("Source columns that could not be mapped to any canonical column")
  mapping_notes string? @description("Notes about the mapping process, e.g. ambiguous matches or type coercion issues")
  metadata InterpretationMetadata @description("Metadata about the interpretation: model used, per-field mapping rationale, detected sections")
}

// ─── Functions ───────────────────────────────────────────────────────────────

function AnalyzeAndParseTable(compressed_text: string) -> ParsedTable {
  client CustomGPT4o
  prompt #"
    You are a table-structure analyst. Given compressed text extracted from a PDF,
    identify the table structure and parse all data rows.

    Steps:
    1. Determine the table type (FlatHeader, HierarchicalHeader, PivotedTable, TransposedTable, or Unknown).
    2. Extract headers — for hierarchical tables, capture each level separately.
    3. Identify any aggregation rows/columns (totals, subtotals, averages, etc.) and separate them.
    4. Parse every data row into an array of string cell values, preserving the original order.
    5. If the text contains multiple table sections separated by labels or headers
       (e.g. data grouped by category, region, time period, or any other dimension),
       record each section label and which data rows belong to it in the notes field.
       Format: "Sections: <label1> (rows 0-N), <label2> (rows N+1-M), ..."

    Important:
    - Keep cell values as strings exactly as they appear (do not reformat numbers or dates).
    - Exclude header rows and aggregation rows from data_rows.
    - If the table has merged cells or spans, repeat the value for each spanned column.

    Compressed text:
    ---
    {{ compressed_text }}
    ---

    {{ ctx.output_format }}
  "#
}

function MapToCanonicalSchema(parsed_table: ParsedTable, schema: CanonicalSchema, model_name: string) -> MappedTable {
  client CustomGPT4o
  prompt #"
    You are a data mapper. Given a parsed table and a canonical schema, map each data row
    to the canonical schema columns.

    Mapping rules:
    1. Match source columns to canonical columns using name, aliases, and description.
    2. If a source column matches multiple canonical columns, pick the best match and note the ambiguity.
    3. Coerce values to the expected type where possible (e.g. "1,234" -> 1234 for int columns).
    4. For date columns, normalize to ISO 8601 format (YYYY-MM-DD) when the date is unambiguous.
    5. If a canonical column has no matching source column, omit it from the record (it will be null).
    6. List any source columns that could not be mapped in unmapped_columns.
    7. CONTEXT INFERENCE: If a canonical column cannot be matched to any source column
       (especially when its aliases list is empty), look for its value in the
       surrounding context:
       - Section headers or group labels that appear above or between data groups
       - Document title, subtitle, or metadata lines
       - Repeated contextual values that apply to all rows in a group
       When a value is inferred from context, apply it to ALL rows in that section.
       The parsed_table's notes field may contain section boundary information.
    8. For EVERY canonical column in the schema, produce a FieldMapping in the
       metadata explaining:
       - Where the value came from (source)
       - Why this mapping was chosen (rationale)
       - How confident you are (High = direct name/alias match,
         Medium = inferred from context with reasonable certainty,
         Low = best guess or ambiguous)
    9. Set metadata.model to "{{ model_name }}" (it will be provided).
    10. If the text had section labels, list them in metadata.sections_detected.

    Parsed table:
    {{ parsed_table }}

    Canonical schema:
    {{ schema }}

    {{ ctx.output_format }}
  "#
}

function InterpretTable(compressed_text: string, schema: CanonicalSchema, model_name: string) -> MappedTable {
  client CustomGPT4o
  prompt #"
    You are a table interpreter. Given compressed text from a PDF and a canonical schema,
    extract and map tabular data in a single pass.

    Steps:
    1. Identify the table structure (flat, hierarchical, pivoted, transposed).
    2. Extract all data rows (exclude headers and aggregation rows like totals).
    3. Map each row's columns to the canonical schema using column names, aliases, and descriptions.
    4. Coerce values to expected types (e.g. "1,234" -> 1234 for int, dates to ISO 8601).
    5. List any source columns that could not be mapped.
    6. CONTEXT INFERENCE: If a canonical column cannot be matched to any source column
       (especially when its aliases list is empty), look for its value in the
       surrounding context:
       - Section headers or group labels that appear above or between data groups
       - Document title, subtitle, or metadata lines
       - Repeated contextual values that apply to all rows in a group
       When a value is inferred from context, apply it to ALL rows in that section.
    7. For EVERY canonical column in the schema, produce a FieldMapping in the
       metadata explaining:
       - Where the value came from (source)
       - Why this mapping was chosen (rationale)
       - How confident you are (High = direct name/alias match,
         Medium = inferred from context with reasonable certainty,
         Low = best guess or ambiguous)
    8. Set metadata.model to "{{ model_name }}" (it will be provided).
    9. If the text had section labels, list them in metadata.sections_detected.

    Important:
    - Keep values faithful to the source; only reformat for type coercion.
    - If a canonical column has no match, omit it from the record.
    - Note any ambiguities or issues in mapping_notes.

    Compressed text:
    ---
    {{ compressed_text }}
    ---

    Canonical schema:
    {{ schema }}

    {{ ctx.output_format }}
  "#
}

// ─── Test cases ──────────────────────────────────────────────────────────────

test flat_shipping_stem {
  functions [AnalyzeAndParseTable]
  args {
    compressed_text #"
      | Port | Vessel | Commodity | Quantity (MT) | ETA | Status |
      |------|--------|-----------|---------------|-----|--------|
      | Portland | MV Ocean Star | Wheat | 52,000 | 2025-10-15 | Loading |
      | Geelong | MV Pacific Tide | Barley | 38,500 | 2025-10-18 | Scheduled |
      | Kwinana | MV Iron Wind | Canola | 41,200 | 2025-10-20 | Scheduled |
      | Total | | | 131,700 | | |
    "#
  }
}

test hierarchical_header {
  functions [AnalyzeAndParseTable]
  args {
    compressed_text #"
      | | Q1 2025 | Q1 2025 | Q2 2025 | Q2 2025 |
      | Region | Revenue | Costs | Revenue | Costs |
      |--------|---------|-------|---------|-------|
      | North | 1,200 | 800 | 1,350 | 850 |
      | South | 980 | 650 | 1,100 | 700 |
      | East | 1,500 | 950 | 1,600 | 1,000 |
    "#
  }
}

test pivoted_with_totals {
  functions [InterpretTable]
  args {
    compressed_text #"
      | Commodity | Oct 2025 | Nov 2025 | Dec 2025 | Total |
      |-----------|----------|----------|----------|-------|
      | Wheat | 52,000 | 48,000 | 55,000 | 155,000 |
      | Barley | 38,500 | 42,000 | 39,500 | 120,000 |
      | Canola | 41,200 | 37,800 | 43,000 | 122,000 |
      | Grand Total | 131,700 | 127,800 | 137,500 | 397,000 |
    "#
    schema {
      description "Monthly commodity shipment volumes"
      columns [
        {
          name "commodity"
          type "string"
          description "Type of commodity"
          aliases ["Commodity", "Product"]
        },
        {
          name "month"
          type "string"
          description "Month and year"
          aliases ["Period", "Month"]
        },
        {
          name "volume_mt"
          type "int"
          description "Volume in metric tonnes"
          aliases ["Quantity", "Volume", "MT"]
        }
      ]
    }
    model_name "openai/gpt-4o"
  }
}

test sectioned_table_context_inference {
  functions [InterpretTable]
  args {
    compressed_text #"
      ## WAREHOUSE A

      | Item | Qty | Unit Price |
      |------|-----|------------|
      | Widget Alpha | 150 | 12.50 |
      | Widget Beta  | 300 | 8.75  |

      ## WAREHOUSE B

      | Item | Qty | Unit Price |
      |------|-----|------------|
      | Widget Alpha | 80  | 12.50 |
      | Gadget Gamma | 200 | 22.00 |
    "#
    schema {
      description "Inventory records by warehouse"
      columns [
        {
          name "warehouse"
          type "string"
          description "Warehouse or location where the inventory is stored — may appear as a section header rather than a table column"
          aliases []
        },
        {
          name "item_name"
          type "string"
          description "Name of the inventory item"
          aliases ["Item", "Product", "SKU"]
        },
        {
          name "quantity"
          type "int"
          description "Number of units in stock"
          aliases ["Qty", "Quantity", "Count"]
        },
        {
          name "unit_price"
          type "float"
          description "Price per unit"
          aliases ["Unit Price", "Price"]
        }
      ]
    }
    model_name "openai/gpt-4o"
  }
}
