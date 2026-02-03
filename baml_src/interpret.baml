// Table interpretation: extract structured tabular data from compressed PDF text
// and map it to a user-defined canonical schema.

// ─── Enums ───────────────────────────────────────────────────────────────────

enum TableType {
  FlatHeader @description("Independent columns with no spanning parent cells. Header text may span multiple rows (stacked/multiline labels) but each column stands alone — combine stacked text into one name per column. 1:1 row-to-record mapping.")
  HierarchicalHeader @description("Tree-structured headers where parent cells span multiple child columns (e.g. 'Q1 2025' spanning 'Revenue' and 'Costs'). Column names are compound paths (e.g. 'Q1 2025 / Revenue'). 1:1 row-to-record mapping. Note: multiline headers with no spanning are FlatHeader, not this.")
  PivotedTable @description("Categories as rows, time periods or attributes as columns — requires unpivoting: each source row produces multiple output records, one per value column")
  TransposedTable @description("Fields as rows, records as columns — requires transposition: each column becomes a record, each row becomes a field")
  Unknown @description("Ambiguous structure — use best judgment based on content")
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
  levels int @description("Number of header rows. For FlatHeader with stacked/multiline labels, this counts the text rows but names[][] should contain the combined column names. For HierarchicalHeader, this counts the hierarchy depth.")
  names string[][] @description("Header names per level, outer = level, inner = columns. For FlatHeader (even with multiline labels), use a single level with combined names. For HierarchicalHeader, each level represents a tier in the tree.")
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

class TableTypeInference {
  table_type TableType          @description("The inferred table structure type")
  rationale string              @description("Brief explanation of why this table type was chosen, citing specific structural evidence")
  confidence Confidence         @description("How confident the inference is based on structural clarity")
}

class InterpretationMetadata {
  model string                  @description("Model that produced this result, e.g. 'openai/gpt-4o'")
  table_type_inference TableTypeInference @description("The inferred table type with rationale explaining the classification")
  field_mappings FieldMapping[] @description("One entry per canonical column, explaining how it was resolved")
  sections_detected string[]?   @description("Section/group labels found in the text, if any (e.g. ['GERALDTON', 'KWINANA'])")
  section_role string?          @description("'context' if a single section label applies to all rows, 'grouping' if multiple labels partition rows into groups")
}

class MappedTable {
  records MappedRecord[] @description("Mapped data records conforming to the canonical schema")
  unmapped_columns string[] @description("Source columns that could not be mapped to any canonical column")
  mapping_notes string? @description("Notes about the mapping process, e.g. ambiguous matches or type coercion issues")
  metadata InterpretationMetadata @description("Metadata about the interpretation: model used, per-field mapping rationale, detected sections")
}

// ─── Vision-based schema inference ──────────────────────────────────────────

class InferredTableSchema {
  column_names string[] @description("Column headers as they visually appear in the table, left to right. For stacked/multiline headers, combine the vertically stacked labels into a single name per column (e.g. 'Unique Slot Reference Number'). For hierarchical headers with spanning parents, use compound paths (e.g. 'Q1 2025 / Revenue').")
  column_count int @description("Total number of data columns")
  header_levels int @description("Number of header rows (1 for single-row, >1 for stacked/multiline or hierarchical)")
  has_spanning_headers bool @description("True if parent cells span multiple child columns (hierarchical). False if headers are just stacked text with no spanning (multiline flat).")
  notes string? @description("Observations about the header structure: merged cells, stacked labels, spanning parent cells, visual groupings, etc.")
}

// ─── Functions ───────────────────────────────────────────────────────────────

function AnalyzeAndParseTable(compressed_text: string) -> ParsedTable {
  client CustomGPT4o
  prompt #"
    You are a table-structure analyst. Given compressed text extracted from a PDF,
    identify the table structure and parse all data rows.

    DETECTING HEADER STRUCTURE:

    Before classifying, examine the header rows carefully:

    1. STACKED/MULTILINE (→ FlatHeader):
       - Each column has text that wraps across multiple rows vertically
       - BUT each column is independent — no cell spans multiple columns horizontally
       - Example: "Unique Slot" / "Reference" / "Number" stacked vertically = one column
         named "Unique Slot Reference Number"
       - Key sign: The number of text fragments per row equals the number of columns

    2. GROUPED/SPANNING (→ HierarchicalHeader):
       - A parent header cell spans HORIZONTALLY across multiple child columns
       - Children are sibling columns that share a common parent cell above them
       - Example: "Date" spanning over "Nominated" and "Accepted" = compound names
         "Date / Nominated" and "Date / Accepted"
       - Key sign: A header row has FEWER cells than the data rows because parents span

    KEY TEST: Do sibling columns share a common parent cell above them that spans
    horizontally? If YES → HierarchicalHeader. If NO → FlatHeader (even with multiline text).

    Steps:
    1. Determine the table type using the header structure tests above:
       - FlatHeader: Each column has its own independent header. This includes tables with
         stacked/multiline headers where text is split across multiple rows but each column
         stands alone (combine the stacked text into one name per column).
       - HierarchicalHeader: Parent header cells SPAN multiple child columns in a tree
         structure (e.g. "Q1 2025" spanning over "Revenue" and "Costs" columns beneath it).
         The key distinction is HORIZONTAL SPANNING — siblings share a common parent cell.
       - PivotedTable: Row labels are categories, column headers are time periods or
         attributes — requires unpivoting.
       - TransposedTable: Field names as rows, records as columns — requires transposition.
       - Unknown: Ambiguous structure.
    2. Extract headers — for hierarchical tables, capture each level separately. For flat
       tables with stacked/multiline headers, combine the text into single column names.
    3. Identify any aggregation rows/columns (totals, subtotals, averages, etc.) and separate them.
    4. Parse every data row into an array of string cell values, preserving the original order.
    5. If the text contains multiple table sections separated by labels or headers
       (e.g. data grouped by category, region, time period, or any other dimension),
       record each section label and which data rows belong to it in the notes field.
       Format: "Sections: <label1> (rows 0-N), <label2> (rows N+1-M), ..."

    Important:
    - Keep cell values as strings exactly as they appear (do not reformat numbers or dates).
    - Exclude header rows and aggregation rows from data_rows.
    - Include ALL other rows, even if they have empty cells, dashes, "N/A", zero or
      missing values, or unusual-looking content. If a row occupies a data position in
      the table it is a data row — do not filter based on cell content.
    - If the table has merged cells or spans, repeat the value for each spanned column.
    - Multiline/stacked headers (where column names wrap across rows but each column is
      independent) should be classified as FlatHeader, NOT HierarchicalHeader.

    Compressed text:
    ---
    {{ compressed_text }}
    ---

    {{ ctx.output_format }}
  "#
}

function InferTableSchemaFromImage(page_img: image, compressed_text: string) -> InferredTableSchema {
  client CustomGPT4o
  prompt #"
    {{ _.role("system") }}
    You are a table structure analyst. You are given:
    1. An image of a PDF page containing a table
    2. Compressed text extracted from the same page (which may have parsing errors)

    Your task: determine the COMPLETE table column structure by combining
    evidence from BOTH the image and the compressed text.

    DETECTING HEADER STRUCTURE:

    Before setting has_spanning_headers, examine the header area carefully:

    1. STACKED/MULTILINE (→ has_spanning_headers=false):
       - Each column has text that wraps across multiple rows vertically
       - BUT each column is independent — no cell spans multiple columns horizontally
       - Example: "Unique Slot" / "Reference" / "Number" stacked vertically = one column
         named "Unique Slot Reference Number"
       - Key visual sign: Column dividers run continuously from top to bottom of header

    2. GROUPED/SPANNING (→ has_spanning_headers=true):
       - A parent header cell spans HORIZONTALLY across multiple child columns
       - Children are sibling columns that share a common parent cell above them
       - Example: "Date" spanning over "Nominated" and "Accepted" = compound names
         "Date / Nominated" and "Date / Accepted"
       - Key visual sign: A cell in an upper row is wider than cells below it,
         with column dividers only appearing in the lower rows

    KEY TEST: Do sibling columns share a common parent cell above them that spans
    horizontally? If YES → has_spanning_headers=true. If NO → false (even with multiline text).

    CRITICAL: Dense tables with small fonts often have narrow columns that are
    easy to miss visually. You MUST cross-validate your column list against the
    data rows in the compressed text.

    Steps:
    1. Count the pipe-separated cells in the FIRST data row of the compressed
       text (the row after the |---|---| separator). This gives you the MINIMUM
       number of columns — some cells may be concatenated (two values joined
       into one cell), so the actual column count may be HIGHER.
    2. Find the table header area in the image. Read EVERY column header left
       to right. Pay special attention to narrow single-word columns between
       wider multi-word columns — these are commonly missed.
       - For stacked/multiline headers (text split across rows but each column
         independent), combine the vertically stacked labels into one name
         (e.g. "Unique Slot" + "Reference" + "Number" → "Unique Slot Reference Number").
       - For hierarchical headers (parent cells spanning horizontally across
         multiple child columns), use compound paths (e.g. "Q1 2025 / Revenue").
    3. Apply the header structure test above to set has_spanning_headers.
    4. Cross-reference with the garbled header lines in the compressed text
       (the lines above the pipe table). Header fragments like individual
       words (e.g. "ID", "STATUS", "CODE") often correspond to real
       columns that may appear narrow in the image.
    5. Verify: your column_count MUST be >= the number of pipe-separated cells
       in the data rows. If it is less, you have missed columns — re-examine
       the image and text headers.
    6. Note how many header rows exist and any structural observations,
       including which data cells appear concatenated (e.g. "42 Smith"
       containing values for two columns).

    {{ ctx.output_format }}

    {{ _.role("user") }}
    {{ page_img }}

    Compressed text (for reference — may contain errors):
    ---
    {{ compressed_text }}
    ---
  "#
}

function AnalyzeAndParseTableGuided(compressed_text: string, visual_schema: InferredTableSchema) -> ParsedTable {
  client CustomGPT4o
  prompt #"
    You are a table-structure analyst. Given compressed text extracted from a PDF
    AND a visual schema describing the correct column structure (inferred from the
    page image), parse the table data rows.

    VISUAL SCHEMA (from image analysis — treat as ground truth for column count
    and header names):
    {{ visual_schema }}

    The compressed text below may have parsing artifacts: column headers may be
    garbled or values from adjacent columns may be concatenated into a single
    cell. Use the visual schema's column_names and column_count to:
    - Correctly identify which part of the text corresponds to which column
    - Split concatenated values when the column count in a row doesn't match
      the expected count (e.g. "33020 WHEAT" should become two cells: "33020"
      and "WHEAT" if they map to separate columns)
    - Use the visual schema's column names as the authoritative header names

    Steps:
    1. Use the visual schema to set the correct headers (column_names →
       headers.names, header_levels → headers.levels).
    2. Determine the table type based on has_spanning_headers:
       - If has_spanning_headers is true → HierarchicalHeader (parent cells span
         multiple child columns in a tree structure)
       - If has_spanning_headers is false → FlatHeader (even with stacked/multiline
         headers, each column is independent)
    3. Identify any aggregation rows/columns and separate them.
    4. Parse every data row into an array of string cell values — the array
       length MUST equal visual_schema.column_count for each row.
    5. If the text contains multiple table sections separated by labels or headers
       (e.g. data grouped by category, region, time period, or any other dimension),
       record each section label and which data rows belong to it in the notes field.
       Format: "Sections: <label1> (rows 0-N), <label2> (rows N+1-M), ..."

    Important:
    - Keep cell values as strings exactly as they appear (do not reformat).
    - Exclude header rows and aggregation rows from data_rows.
    - Include ALL data rows regardless of cell content.
    - If a row has fewer tokens than expected columns, split concatenated
      values to reach the correct column count.
    - If the table has merged cells or spans, repeat the value for each
      spanned column.

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
    You are a data mapper. Given a parsed table and a canonical schema, map data rows
    to the canonical schema columns using a strategy appropriate for the table type.

    TABLE TYPE: {{ parsed_table.table_type }}

    Mapping strategy by table type:
    - FlatHeader: Map each data row directly to one output record. Columns align 1:1.
    - HierarchicalHeader: Column names may be compound from merged headers (e.g. "Q1 / Revenue").
      Match the full compound name against aliases. Each data row produces one record.
    - PivotedTable: Categories are rows, time periods or attributes are columns. You must UNPIVOT:
      for each data row, produce ONE OUTPUT RECORD PER VALUE COLUMN. The column header becomes
      a field value (e.g. column "Period A" → the "period" field gets value "Period A").
      Example: a row "Item X | 100 | 200" with columns "Category | Period A | Period B"
      becomes TWO records: {category: "Item X", period: "Period A", value: 100} and
      {category: "Item X", period: "Period B", value: 200}.
    - TransposedTable: Fields are rows, records are columns. Transpose before mapping:
      each column (after the first) is a separate record, each row label is a field name.
    - Unknown: Use best judgment based on the data structure.

    General mapping rules:
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
    10. If the text had section labels, list them in metadata.sections_detected and set
        metadata.section_role to:
        - "context" if there is only ONE section label that applies to all rows
        - "grouping" if there are MULTIPLE section labels that partition rows into groups
    11. Populate metadata.table_type_inference with:
        - table_type: Use the table type from parsed_table ({{ parsed_table.table_type }})
        - rationale: Explain why this classification applies based on the table structure.
          For FlatHeader vs HierarchicalHeader, explicitly state whether parent cells span
          horizontally across multiple child columns or not.
        - confidence: High if structure is unambiguous, Medium if some irregularities, Low if uncertain

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

    DETECTING HEADER STRUCTURE:

    Before classifying, examine the header rows carefully:

    1. STACKED/MULTILINE (→ FlatHeader):
       - Each column has text that wraps across multiple rows vertically
       - BUT each column is independent — no cell spans multiple columns horizontally
       - Example: "Unique Slot" / "Reference" / "Number" stacked vertically = one column
         named "Unique Slot Reference Number"
       - Key sign: The number of text fragments per row equals the number of columns

    2. GROUPED/SPANNING (→ HierarchicalHeader):
       - A parent header cell spans HORIZONTALLY across multiple child columns
       - Children are sibling columns that share a common parent cell above them
       - Example: "Date" spanning over "Nominated" and "Accepted" = compound names
         "Date / Nominated" and "Date / Accepted"
       - Key sign: A header row has FEWER cells than the data rows because parents span

    KEY TEST: Do sibling columns share a common parent cell above them that spans
    horizontally? If YES → HierarchicalHeader. If NO → FlatHeader (even with multiline text).

    Steps:
    1. Identify the table structure using the header structure tests above and apply the
       appropriate mapping strategy:
       - FlatHeader: Each column has its own independent header. This includes tables with
         stacked/multiline headers where text is split across multiple rows but each column
         stands alone — combine the stacked text into one column name. Map each row to one record.
       - HierarchicalHeader: Parent header cells SPAN HORIZONTALLY across multiple child columns
         in a tree structure (e.g. "Q1 2025" spanning over "Revenue" and "Costs" columns beneath it).
         Column names are compound paths (e.g. "Q1 2025 / Revenue"). Map each row to one record.
         NOTE: Multiline/stacked headers with no horizontal spanning are FlatHeader, not this.
       - PivotedTable: Categories as rows, time periods or attributes as columns. UNPIVOT:
         for each data row, produce ONE RECORD PER VALUE COLUMN. The column header becomes
         a field value. Example: "Item X | 100 | 200" with columns "Category | Period A | Period B"
         becomes TWO records: {category: "Item X", period: "Period A", value: 100} and
         {category: "Item X", period: "Period B", value: 200}.
       - TransposedTable: Fields as rows, records as columns. Transpose: each column is a
         record, each row label is a field name.
    2. Extract all data rows (exclude headers and aggregation rows like totals).
    3. Match source columns to canonical columns using names, aliases, and descriptions.
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
    9. If the text had section labels, list them in metadata.sections_detected and set
       metadata.section_role to:
       - "context" if there is only ONE section label that applies to all rows
       - "grouping" if there are MULTIPLE section labels that partition rows into groups
    10. Populate metadata.table_type_inference with:
        - table_type: The table structure type you identified in step 1
        - rationale: Explain why this classification applies, citing specific structural evidence.
          For FlatHeader vs HierarchicalHeader, explicitly state whether parent cells span
          horizontally across multiple child columns or not.
        - confidence: High if structure is unambiguous, Medium if some irregularities, Low if uncertain

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
