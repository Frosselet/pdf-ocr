// Compress pipeline: LLM-assisted header refinement and table detection fallback
// for the spatial text compressor.

// ─── Types ──────────────────────────────────────────────────────────────────

class RefinedHeaders {
  header_structure HeaderStructure
    @description("Vertical organization of the column labels.")
  header_row_count int
    @description("Number of rows occupied by column labels.")
  column_names string[]
    @description("Clean column names, one per data column.")
  notes string?
    @description("Observations: merged cells, spanning parents, missing headers, etc.")
}

class DetectedTable {
  layout TableLayout
    @description("Geometric arrangement of records and fields in the table.")
  header_structure HeaderStructure
    @description("Vertical organization of the column labels.")
  column_names string[]
    @description("Column headers, one per data column.")
  data_rows string[][]
    @description("Data rows. Each inner array has one cell value per column.")
  notes string?
    @description("Observations: sections found, aggregation rows excluded, etc.")
}

// ─── Functions ──────────────────────────────────────────────────────────────

function RefineTableHeaders(spatial_excerpt: string, data_column_count: int) -> RefinedHeaders {
  client CustomGPT4oMini
  prompt #"
    You are a table-structure analyst. You receive a monospace spatial text excerpt
    of a table region from a PDF. Your job is to classify the header structure,
    count header rows, and produce clean column names.

    SPATIAL TEXT (monospace — positions matter):
    {{ spatial_excerpt }}

    DATA COLUMN COUNT (from actual data rows): {{ data_column_count }}

    STEPS:
    1. CLASSIFY: Determine header_structure using the HeaderStructure definitions
       and KEY SIGNS in the output schema below.
    2. COUNT: Set header_row_count — how many rows at the top contain column labels
       (not data).
    3. RESOLVE: Produce column_names (exactly {{ data_column_count }} entries):
       - SingleRow: use each header cell directly as the column name.
       - Stacked: concatenate fragments top-to-bottom at the same column position,
         using EXACT original text. Join with single space. Do NOT rephrase,
         abbreviate, or interpret — just flatten the wrapped text verbatim.
         Example: "DATE AT WHICH" + "NOMINATION" + "WAS RECEIVED"
         → "DATE AT WHICH NOMINATION WAS RECEIVED" (not "Nomination Date")
       - Hierarchical: use "Parent / Child" compound paths with exact original text.
       - If a column has no visible header, use empty string "".
    4. Read column positions carefully. The spatial layout shows where each header
       label is positioned — match them to the {{ data_column_count }} data columns.

    CRITICAL: Preserve the original text exactly. Do not rename, rephrase, or
    "clean up" column names. The goal is to flatten wrapped headers, not interpret them.

    {{ ctx.output_format }}
  "#
}

function DetectAndStructureTable(spatial_text: string) -> DetectedTable {
  client CustomGPT4oMini
  prompt #"
    You are a table-detection analyst. You receive the full monospace spatial text of
    a PDF page where automated heuristics found NO tables. Your job is to determine
    if there IS a table on this page, and if so, classify and extract it.

    SPATIAL TEXT (monospace — positions matter):
    {{ spatial_text }}

    STEPS:
    1. DETECT: Look for tabular data — rows of values aligned in columns, with or
       without visible headers, separators, or borders.
    2. CLASSIFY: If a table is found, determine layout and header_structure using the
       enum definitions and KEY SIGNS in the output schema below.
    3. EXTRACT:
       - Produce clean column names using the structure-specific strategy:
         SingleRow=direct, Stacked=concatenate, Hierarchical=compound paths.
       - Extract all data rows (exclude header rows and aggregation/total rows).
       - Each data row must have the same number of cells as column_names.
       - Use empty string "" for missing/empty cells.
       - Keep cell values as strings exactly as they appear.
    4. If there is NO table on this page, return layout=Standard,
       header_structure=SingleRow, empty column_names [] and empty data_rows [].
    5. If the page contains non-table text (headings, paragraphs, metadata),
       ignore it — only extract tabular content.

    {{ ctx.output_format }}
  "#
}
