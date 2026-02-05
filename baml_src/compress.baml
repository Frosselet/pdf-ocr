// Compress pipeline: LLM-assisted header refinement and table detection fallback
// for the spatial text compressor.

// ─── Types ──────────────────────────────────────────────────────────────────

class RefinedHeaders {
  header_row_count int
    @description("Number of rows at the top that are headers (not data). Count all rows containing column labels, units, or grouping text.")
  column_names string[]
    @description("Clean column names, one per data column. Combine multiline/stacked text into single names. For hierarchical spanning headers, use 'Parent / Child' paths.")
  notes string?
    @description("Observations: merged cells, stacked labels, spanning parents, missing headers, etc.")
}

class DetectedTable {
  column_names string[]
    @description("Column headers, one per data column")
  data_rows string[][]
    @description("Data rows excluding headers and aggregation/total rows. Each inner array has one cell value per column. Keep values as strings exactly as they appear. Use empty string for empty cells.")
  notes string?
    @description("Observations: sections found, aggregation rows excluded, etc.")
}

// ─── Functions ──────────────────────────────────────────────────────────────

function RefineTableHeaders(spatial_excerpt: string, data_column_count: int) -> RefinedHeaders {
  client CustomGPT4oMini
  prompt #"
    You are a table-structure analyst. You receive a monospace spatial text excerpt
    of a table region from a PDF. Your job is to identify the header rows and produce
    clean column names.

    SPATIAL TEXT (monospace — positions matter):
    {{ spatial_excerpt }}

    DATA COLUMN COUNT (from actual data rows): {{ data_column_count }}

    INSTRUCTIONS:
    1. The data rows (below the headers) have exactly {{ data_column_count }} columns.
       Your column_names array MUST have exactly {{ data_column_count }} entries.
    2. Identify which rows at the top are header rows (column labels, units, grouping
       text) vs. data rows. Set header_row_count accordingly.
    3. Produce clean, human-readable column names:
       - If headers span multiple lines vertically (stacked text), combine them into
         a single name (e.g., "Loading" over "Date" → "Loading Date").
       - If headers are hierarchical with a parent spanning multiple child columns,
         use "Parent / Child" format (e.g., "Region / North", "Region / South").
       - If a column has no visible header, use a descriptive placeholder based on
         the data values (e.g., "Column 1", "Value", "ID").
    4. Read column positions carefully. The spatial layout shows where each header
       label is positioned — match them to the {{ data_column_count }} data columns.

    {{ ctx.output_format }}
  "#
}

function DetectAndStructureTable(spatial_text: string) -> DetectedTable {
  client CustomGPT4oMini
  prompt #"
    You are a table-detection analyst. You receive the full monospace spatial text of
    a PDF page where automated heuristics found NO tables. Your job is to determine
    if there IS a table on this page, and if so, extract it.

    SPATIAL TEXT (monospace — positions matter):
    {{ spatial_text }}

    INSTRUCTIONS:
    1. Look for tabular data: rows of values aligned in columns, with or without
       visible headers, separators, or borders.
    2. If you find a table:
       - Extract clean column names (from headers if present, or inferred from data).
       - Extract all data rows (exclude header rows and aggregation/total rows).
       - Each data row must have the same number of cells as column_names.
       - Use empty string "" for missing/empty cells.
       - Keep cell values as strings exactly as they appear.
    3. If there is NO table on this page, return empty column_names [] and
       empty data_rows [].
    4. If the page contains non-table text (headings, paragraphs, metadata),
       ignore it — only extract tabular content.

    {{ ctx.output_format }}
  "#
}
