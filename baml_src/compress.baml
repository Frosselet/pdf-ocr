// Compress pipeline: LLM table detection fallback for the spatial text compressor.
// Header refinement is now handled deterministically by _build_stacked_headers().

// ─── Types ──────────────────────────────────────────────────────────────────

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
