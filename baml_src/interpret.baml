// Table interpretation: extract structured tabular data from compressed PDF text
// and map it to a user-defined canonical schema.

// ─── Enums ───────────────────────────────────────────────────────────────────

enum TableType {
  FlatHeader @description("Standard table: headers in top row(s), data in subsequent rows. Each column is independent (no spanning parent cells). Header text may wrap across multiple rows vertically but each column stands alone. 1:1 row-to-record mapping. Key sign: first column contains different values per row (IDs, names, categories).")
  HierarchicalHeader @description("Tree-structured headers where parent cells span horizontally across multiple child columns (e.g. 'Group A' spanning 'Metric 1' and 'Metric 2'), OR where column names contain ' / ' separators indicating compound names from multi-row header stacking (e.g. 'Group A / Metric 1'). Column names become compound paths (e.g. 'Group A / Metric 1'). Mapping may produce 1:1 records or require unpivoting depending on the canonical schema. Key signs: (1) header row has FEWER cells than data rows because parents span multiple children, OR (2) column headers contain ' / ' separators from compound header construction. Note: stacked/multiline headers with no horizontal spanning AND no ' / ' separators are FlatHeader, not this.")
  PivotedTable @description("Cross-tabulation: row labels are categories, column headers are time periods or attributes, cells are values. Requires unpivoting: each source row produces multiple output records (one per value column). Key sign: column headers are dates, periods, or repeated attribute names; first column contains category labels; cells contain numeric values.")
  TransposedTable @description("Property sheet layout: field NAMES as rows (left column), field VALUES as columns. Each column (after the label column) represents a complete record. Requires transposition: each column becomes one output record, each row provides one field. Key signs: (1) first column contains field names like 'Name', 'Date', 'Status', 'Total'; (2) subsequent columns contain the corresponding values; (3) rows are heterogeneous (mix of dates, numbers, text) because each row is a different field type.")
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
  levels int @description("Number of header rows.")
  names string[][] @description("Header names per level, outer = level, inner = columns.")
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
  data_rows string[][] @description("All data rows. Each inner array is one row of cell values as strings.")
  notes string? @description("Any caveats or observations about the table structure")
}

// ─── Canonical schema + output types ─────────────────────────────────────────

class ColumnDef {
  name string @description("Canonical column name")
  type string @description("Expected type: string, int, float, bool, date")
  description string @description("What this column represents")
  aliases string[] @description("Alternative names this column may appear as in source tables")
  format string? @description("Output format specification. For dates: YYYY-MM-DD, YYYY-MM, HH:mm:ss, etc. For numbers: #,###.## (thousands + decimals), +# (explicit sign), #% (percentage). For strings: uppercase, lowercase, titlecase, camelCase, PascalCase, snake_case, kebab-case, trim.")
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
  source string       @description("Where the value came from, e.g. 'column: Field A', 'section header', 'document title', 'inferred from context'")
  rationale string    @description("Brief explanation of why this mapping was chosen")
  confidence Confidence
}

class TableTypeInference {
  table_type TableType          @description("Table type from Step 1 (do not re-classify)")
  mapping_strategy_used string  @description("Which mapping strategy was applied: '1:1 row mapping', 'unpivot', 'transpose'")
}

class InterpretationMetadata {
  model string                  @description("Model that produced this result, e.g. 'openai/gpt-4o'")
  table_type_inference TableTypeInference @description("The table type from Step 1 and which mapping strategy was applied")
  field_mappings FieldMapping[] @description("One entry per canonical column, explaining how it was resolved")
  sections_detected string[]?   @description("Section/group labels found in the text, if any (e.g. ['GROUP A', 'GROUP B'])")
  section_role string?          @description("Role of section labels: 'context' or 'grouping'")
}

class MappedTable {
  records MappedRecord[] @description("Mapped data records conforming to the canonical schema")
  unmapped_columns string[] @description("Source columns that could not be mapped to any canonical column")
  mapping_notes string? @description("Notes about the mapping process, e.g. ambiguous matches or type coercion issues")
  metadata InterpretationMetadata @description("Metadata about the interpretation: model used, per-field mapping rationale, detected sections")
}

// ─── Vision-based schema inference ──────────────────────────────────────────

class InferredTableSchema {
  column_names string[] @description("Column headers as they visually appear in the table, left to right. One entry per data column.")
  column_count int @description("Total number of data columns")
  header_levels int @description("Number of header rows (1 for single-row, >1 for stacked/multiline or hierarchical)")
  has_spanning_headers bool @description("True if parent cells span multiple child columns (hierarchical). False otherwise.")
  notes string? @description("Observations about the header structure: merged cells, stacked labels, spanning parent cells, visual groupings, etc.")
}

// ─── Functions ───────────────────────────────────────────────────────────────

function AnalyzeAndParseTable(compressed_text: string) -> ParsedTable {
  client CustomGPT4o
  prompt #"
    You are a table-structure analyst. Given compressed text extracted from a PDF,
    identify the table structure and parse all data rows.

    === DATA-FIRST APPROACH ===

    The data rows are the ground truth for column structure. Analyze them FIRST:

    STEP 1: COUNT DATA COLUMNS
    - Find the pipe table (rows with | delimiters)
    - Count cells in the FIRST data row (after |---|---| separator)
    - This is your AUTHORITATIVE column count

    STEP 2: IDENTIFY DATA TYPES PER COLUMN
    - Look at values in each column position across multiple data rows
    - Identify: dates, times, numbers, text, codes, status words, etc.
    - Note any concatenated values (e.g. "1500 Category A" = number + text = 2 columns)

    STEP 3: MAP HEADERS TO DATA COLUMNS
    - Look at the header lines ABOVE the pipe table
    - For stacked/multiline headers, combine vertically aligned text:
      "Unique Slot" / "Reference" / "Number" → "Unique Slot Reference Number"
    - You MUST produce exactly as many header names as data columns

    === TABLE TYPE CLASSIFICATION ===

    Classify using the TableType enum definitions (provided in the output schema).
    Quick decision tree:
    - Header row has FEWER cells than data rows (horizontal spanning)? → HierarchicalHeader
    - Column headers contain " / " separators (compound names from stacked headers)? → HierarchicalHeader
    - Column headers are repeated categories (dates, periods, attributes)? → PivotedTable
    - Field names in left column, values across columns? → TransposedTable
    - Otherwise → FlatHeader (including stacked/multiline headers with no spanning)

    === PARSING RULES ===

    1. headers.names MUST have exactly column_count entries (one per data column)
    2. Each data_row MUST have exactly column_count cells
    3. If a pipe cell contains concatenated values that span multiple columns,
       SPLIT them to maintain consistent column count
    4. Exclude aggregation rows (totals, subtotals) from data_rows
    5. Keep cell values as strings exactly as they appear
    6. Include ALL data rows even with empty cells, dashes, "N/A", or zeros

    SECTION LABELS:
    If the text has sections (e.g. data grouped by region, category, etc.),
    record in notes: "Sections: <label1> (rows 0-N), <label2> (rows N+1-M), ..."

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

    Your task: determine the COMPLETE table column structure. Use a DATA-FIRST
    approach — the data rows are the ground truth for column count.

    === STEP 1: ANALYZE DATA ROWS (Ground Truth) ===

    Look at the DATA ROWS in the image (below the headers):
    1. Pick a complete data row and count the distinct columns visually
    2. Note column boundaries — where values end and new values begin
    3. Identify the DATA TYPE in each column:
       - Dates (DD/MM/YYYY, YYYY-MM-DD, etc.)
       - Times (HH:MM, HH:MM:SS AM/PM)
       - Numbers (quantities, IDs, codes)
       - Text (names, descriptions, status words)
       - Mixed (e.g. "1500 Category A" = number + text = likely 2 columns concatenated)

    The number of distinct data columns is your AUTHORITATIVE column_count.

    === STEP 2: VALIDATE WITH COMPRESSED TEXT ===

    Count pipe-separated cells in the FIRST data row of the compressed text
    (the row after |---|---|). Compare to your visual count:
    - If text has FEWER cells → some values are concatenated in text
    - If text has MORE cells → you may have missed narrow columns visually
    - Note any concatenations (e.g. "7:00 PM Company X" = time + text = 2 columns)

    === STEP 3: MAP HEADERS TO DATA COLUMNS ===

    Now look at the header area and assign a name to EACH data column:
    1. For stacked/multiline headers, combine vertically aligned text into one name
       (e.g. "Unique Slot" / "Reference" / "Number" → "Unique Slot Reference Number")
    2. For hierarchical headers with horizontal spanning, use compound paths
       (e.g. "Date" spanning "Nominated" and "Accepted" → "Date / Nominated", "Date / Accepted")
    3. If a header is unclear, infer from the data type (e.g. column with dates
       and no clear header might be "Date" or "ETA")

    CRITICAL: column_names array length MUST equal column_count from Step 1.

    === STEP 4: DETERMINE HEADER STRUCTURE ===

    Set has_spanning_headers based on visual header structure:
    - FALSE (stacked/multiline): Column dividers run continuously top-to-bottom,
      each column independent even if header text spans multiple rows vertically
    - TRUE (hierarchical): Parent cells span horizontally across multiple child
      columns, column dividers only appear in lower rows under the spanning cell

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
      the expected count (e.g. "1500 Category A" should become two cells: "1500"
      and "Category A" if they map to separate columns)
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
    - FlatHeader: Map each data row directly to one output record.
    - HierarchicalHeader: Check if canonical column aliases match PARTS of compound
      column names (e.g. "Group A / Metric 1"):
      - If canonical columns have aliases that match the parent/child parts of compound headers
        (e.g., "group" aliases ["Group A", "Group B"] and "metric" aliases ["Metric 1", "Metric 2"]
        match parts of "Group A / Metric 1", "Group B / Metric 2"), you must UNPIVOT:
        Each data row produces MULTIPLE records (one per value column).
        The parent and child parts of each compound header become field values.
        CRITICAL: Only unpivot columns where ALL header parts match dimension aliases.
        If a column's header part does NOT match any alias, SKIP that column entirely —
        do not create a record for it. This includes:
        - Derived/calculated columns (e.g., "% change" when metric aliases are ["Revenue", "Costs"])
        - Aggregation columns (e.g., "Total" or "TOTAL" when category aliases are ["Type A", "Type B"])
        Common aggregation labels to skip: "Total", "TOTAL", "Subtotal", "All", "Grand Total", "Sum".
        Unless an aggregation label is explicitly listed in aliases, skip columns containing it.
        ALSO: Skip rows where the row label (first column value) represents an aggregation,
        using the same aggregation labels above.
        Example: row "Region A | 100 | 200 | +50% | 150 | 180 | -10%" with columns
        "Label | Q1 / Revenue | Q1 / Costs | Q1 / % change | Q2 / Revenue | Q2 / Costs | Q2 / % change"
        and schema {label, category (aliases: Q1, Q2), metric (aliases: Revenue, Costs), value}
        becomes FOUR records (% change columns skipped because "% change" is not in metric aliases):
        - {label: "Region A", category: "Q1", metric: "Revenue", value: 100}
        - {label: "Region A", category: "Q1", metric: "Costs", value: 200}
        - {label: "Region A", category: "Q2", metric: "Revenue", value: 150}
        - {label: "Region A", category: "Q2", metric: "Costs", value: 180}
        IMPORTANT: Each record's dimension values must come from THAT SPECIFIC column's header.
        Do NOT use the same value for all records — "Q1" columns produce "Q1", "Q2" columns produce "Q2".
      - If no canonical aliases match parts of compound headers, match the full compound name
        against aliases and produce one record per data row.
    - PivotedTable: UNPIVOT — for each data row, produce ONE OUTPUT RECORD PER VALUE COLUMN.
      The column header becomes a field value (e.g. column "Period A" → the "period" field gets value "Period A").
      Example: a row "Item X | 100 | 200" with columns "Category | Period A | Period B"
      becomes TWO records: {category: "Item X", period: "Period A", value: 100} and
      {category: "Item X", period: "Period B", value: 200}.
    - TransposedTable: TRANSPOSE before mapping:
      1. The first column contains field names — these become your "headers" for alias matching
      2. Each subsequent column is ONE COMPLETE RECORD
      3. Match each row's first-column label against canonical column aliases
      4. Example: If rows are "Name | Alice | Bob" and "Status | Active | Inactive",
         produce TWO records: {name: "Alice", status: "Active"} and {name: "Bob", status: "Inactive"}
      5. EVERY row label must be checked against ALL canonical column aliases
    - Unknown: Use best judgment based on the data structure.

    COMPOUND HEADER DETECTION (applies to ALL table types):
    When column headers contain " / " separators (e.g., "Category A / Metric 1",
    "Category A / Metric 2"), the parts form a hierarchy. Check if canonical
    column aliases match PARTS of these compound headers:
    - Split each compound header on " / " to get its parts
    - If aliases match parts of compound headers, UNPIVOT: each data row produces
      multiple records (one per value column), with matched parts as field values
    - Apply the same aggregation/derived column filtering as HierarchicalHeader
    This ensures compound headers from DOCX/Excel extractors trigger correct
    unpivoting even if the table was not classified as HierarchicalHeader.

    General mapping rules:
    1. Match source columns to canonical columns using name, aliases, and description.
       - If aliases are provided, match against them literally.
       - If aliases are EMPTY, use the description to infer what values should match:
         * For date/time fields: match any value that looks like a year (4 digits), month,
           date, or time period (e.g., "YYYY", "Mon", "Q1", "Month Year")
         * For category fields: match based on semantic similarity to the description
       This allows schemas to be robust without hardcoding specific values.
    2. If a source column matches multiple canonical columns, pick the best match and note the ambiguity.
    3. Coerce values to the expected type where possible (e.g. "1,234" -> 1234 for int columns).
    4. FORMAT OUTPUT VALUES: If a column has a `format` field, apply it to the output:
       - Dates: Use the specified pattern (YYYY-MM-DD, YYYY-MM, HH:mm, etc.)
       - Numbers: Apply formatting (#,### for thousands, #.## for decimals, +# for sign, #% for percentage)
       - Strings: Apply case transformation (uppercase, lowercase, titlecase, camelCase, PascalCase, snake_case, kebab-case, trim)
       If no format is specified, use sensible defaults (ISO 8601 for dates, plain numbers, original case for strings).
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
       - How confident you are (see Confidence enum criteria)
    9. Set metadata.model to "{{ model_name }}" (it will be provided).
    10. If the text had section labels, list them in metadata.sections_detected and set
        metadata.section_role to:
        - "context" if there is only ONE section label that applies to all rows
        - "grouping" if there are MULTIPLE section labels that partition rows into groups
    11. Populate metadata.table_type_inference with:
        - table_type: {{ parsed_table.table_type }} (use this exact value, do not re-classify)
        - mapping_strategy_used: Describe which strategy you applied (e.g., "1:1 row mapping for FlatHeader", "unpivot for PivotedTable")
    12. EXTRACT ALL DATA — DO NOT FILTER: Schema descriptions are metadata about what the field
        represents, NOT filtering instructions. Extract ALL data rows and ALL relevant columns
        from the source table. If a table contains data for multiple time periods, extract
        records for ALL periods. Never skip data because a description mentions "latest",
        "current", "recent", etc. — the user will filter downstream if needed.

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

    KEY TEST: Does a header row have FEWER cells than data rows because parent
    cells span horizontally across multiple children?
    - YES → HierarchicalHeader (use compound 'Parent / Child' names)
    - NO → FlatHeader (even if header text is stacked across multiple rows —
      combine stacked text into single column names)

    See the TableType enum definitions in the output schema for full criteria.

    Steps:
    1. Identify the table structure using the header structure tests above and apply the
       appropriate mapping strategy:
       - FlatHeader: Combine stacked/multiline header text into single column names.
         Map each row to one record.
       - HierarchicalHeader: Column names are compound 'Parent / Child' paths.
         Check if canonical column aliases match PARTS of compound headers. If so, UNPIVOT:
         each data row produces multiple records (one per value column), with parent/child parts
         becoming field values. CRITICAL: Only unpivot columns where ALL header parts match dimension
         aliases. Skip columns where any header part does NOT match aliases — this includes derived
         columns (e.g., "% change") AND aggregation columns (e.g., "Total", "TOTAL", "Subtotal").
         Also skip rows where the row label is an aggregation. Unless explicitly in aliases, skip it.
         If no aliases match parts, map each row to one record.
       - PivotedTable: UNPIVOT — for each data row, produce ONE RECORD PER VALUE COLUMN. The column header becomes
         a field value. Example: "Item X | 100 | 200" with columns "Category | Period A | Period B"
         becomes TWO records: {category: "Item X", period: "Period A", value: 100} and
         {category: "Item X", period: "Period B", value: 200}.
       - TransposedTable: TRANSPOSE — each subsequent column (after the label column) is
         ONE COMPLETE RECORD, each row provides one field.
         (1) First column labels become "headers" for alias matching
         (2) Match each row label against canonical column aliases
         (3) Example: rows "Name | Alice | Bob" and "Status | Active | Inactive" →
             TWO records: {name: "Alice", status: "Active"}, {name: "Bob", status: "Inactive"}

       COMPOUND HEADER DETECTION (applies to ALL table types):
       When column headers contain " / " separators (e.g., "Category A / Metric 1",
       "Category A / Metric 2"), the parts form a hierarchy. Check if canonical
       column aliases match PARTS of these compound headers:
       - Split each compound header on " / " to get its parts
       - If aliases match parts of compound headers, UNPIVOT: each data row produces
         multiple records (one per value column), with matched parts as field values
       - Apply the same aggregation/derived column filtering as HierarchicalHeader
       This ensures compound headers from DOCX/Excel extractors trigger correct
       unpivoting even if the table was not classified as HierarchicalHeader.

    2. Extract all data rows (exclude headers and aggregation rows like totals).
    3. Match source columns to canonical columns using names, aliases, and descriptions.
       When aliases are EMPTY, use the description to infer matches — e.g., a date field
       with empty aliases should match any year (4 digits), month name, or time period.
       IMPORTANT: When unpivoting, each record's dimension value must come from THAT column's
       header — e.g., "Period A" columns → "Period A", "Period B" columns → "Period B". Do NOT
       use the same value for all records.
    4. Coerce values to expected types and FORMAT OUTPUT VALUES per column's `format` field:
       - Dates: Use specified pattern (YYYY-MM-DD, YYYY-MM, HH:mm, etc.)
       - Numbers: Apply formatting (#,### for thousands, #.## for decimals, +# for sign, #% for percentage)
       - Strings: Apply case transformation (uppercase, lowercase, titlecase, camelCase, PascalCase, snake_case, kebab-case, trim)
       If no format specified, use defaults (ISO 8601 for dates, plain numbers, original case).
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
       - How confident you are (see Confidence enum criteria)
    8. Set metadata.model to "{{ model_name }}" (it will be provided).
    9. If the text had section labels, list them in metadata.sections_detected and set
       metadata.section_role to:
       - "context" if there is only ONE section label that applies to all rows
       - "grouping" if there are MULTIPLE section labels that partition rows into groups
    10. Populate metadata.table_type_inference with:
        - table_type: The table structure type you identified in step 1
        - mapping_strategy_used: Describe which strategy you applied (e.g., "1:1 row mapping for FlatHeader", "unpivot for PivotedTable")
    11. EXTRACT ALL DATA — DO NOT FILTER: Schema descriptions are metadata about what the field
        represents, NOT filtering instructions. Extract ALL data rows and ALL relevant columns
        from the source table. If a table contains data for multiple time periods, extract
        records for ALL periods. Never skip data because a description mentions "latest",
        "current", "recent", etc. — the user will filter downstream if needed.

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

// Test: HierarchicalHeader with unpivoting to normalized schema
// This is a common pattern: compound column headers (Category / Period / Metric)
// that need to be unpivoted into a normalized schema with separate columns for
// the category, period, and value.
// IMPORTANT: The "% change" columns should be SKIPPED because "% change" does not
// match any alias for the "period" dimension (which only has ["Period 1", "Period 2"]).
// Expected output: 8 records (2 rows × 2 categories × 2 periods), NOT 12 records.
test hierarchical_header_unpivot {
  functions [MapToCanonicalSchema]
  args {
    parsed_table {
      table_type HierarchicalHeader
      headers {
        levels 2
        names [[
          "Label",
          "Category A / Period 1",
          "Category A / Period 2",
          "Category A / % change",
          "Category B / Period 1",
          "Category B / Period 2",
          "Category B / % change"
        ]]
      }
      aggregations null
      data_rows [
        ["Item X", "25,000", "22,000", "+13.6", "45,000", "52,000", "-13.5"],
        ["Item Y", "48,000", "41,000", "+17.1", "62,000", "71,000", "-12.7"]
      ]
      notes "Sections: GROUP 1 (rows 0-1)"
    }
    schema {
      description "Normalized data with category and period dimensions"
      columns [
        {
          name "label"
          type "string"
          description "Row label or identifier"
          aliases ["Label", "Name"]
        },
        {
          name "category"
          type "string"
          description "Category type (extracted from compound column header)"
          aliases ["Category A", "Category B", "Category C"]
        },
        {
          name "period"
          type "string"
          description "Time period (extracted from compound column header)"
          aliases ["Period 1", "Period 2"]
        },
        {
          name "value"
          type "int"
          description "Numeric value"
          aliases []
        }
      ]
    }
    model_name "openai/gpt-4o"
  }
}
