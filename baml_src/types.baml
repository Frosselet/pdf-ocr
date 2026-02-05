// Formal type system for PDF table structure classification.
// Shared vocabulary for both compress and interpret pipelines.
// All descriptions are domain-agnostic — abstract structural terms only.

// ─── TableLayout ─────────────────────────────────────────────────────────────
// Geometry of records vs fields. Orthogonal to header structure.

enum TableLayout {
  Standard
    @description(#"
      Each row is one record, each column is one field. The most common layout.
      Records flow top-to-bottom. One-to-one row mapping.
      KEY SIGNS: Uniform cell count across data rows. First column values are
      distinct identifiers (names, codes, dates). Headers sit above data.
    "#)
  Transposed
    @description(#"
      Field names run down the left column, field values run across columns.
      Each value column is one complete record. Requires transposition.
      KEY SIGNS: Left column reads like form labels (Label A, Label B, Label C).
      Rows have mixed types. Multiple value columns hold parallel data for
      different entities.
    "#)
  CrossTab
    @description(#"
      Cross-tabulation: row labels are one dimension, column headers are
      another dimension, cells contain measure values. Requires unpivoting —
      each cell becomes its own record with row label + column label + value.
      KEY SIGNS: Column headers are dates/periods/repeated attributes. First
      column has category labels. Cells are numeric/percentage values.
    "#)
}

// ─── HeaderStructure ─────────────────────────────────────────────────────────
// Vertical organization of column labels.
// Critical distinction: same cell count = Stacked, fewer cells = Hierarchical.

enum HeaderStructure {
  SingleRow
    @description(#"
      One row of complete column labels. Each cell is a full, readable name.
      KEY SIGNS: Header row cell count equals data row cell count. No
      vertically split words. No spanning parents above.
    "#)
  Stacked
    @description(#"
      Labels split across 2+ rows vertically. Each column is independent —
      no horizontal spanning. Fragments must be concatenated top-to-bottom.
      Example: 'Field' (row 1) + 'Name' (row 2) + 'Here' (row 3)
      at the same column position → 'Field Name Here'.
      KEY SIGNS: Multiple header rows with the SAME cell count as data rows.
      Individual cells contain word fragments. Vertical concatenation produces
      sensible names. Typically 2-9 header rows.
      CRITICAL: Same cell count across header rows = Stacked, NOT Hierarchical.
    "#)
  Hierarchical
    @description(#"
      Tree-structured headers where parent labels span horizontally across
      multiple child columns. Column names become compound paths: 'Parent / Child'.
      Example: 'Category A' spans 'Period 1' and 'Period 2'.
      KEY SIGNS: A header row has FEWER cells than the row below it. Parent
      cells centered above children. Compound path names needed.
      CRITICAL: FEWER cells in upper rows than lower rows = Hierarchical.
    "#)
}

// ─── HeaderOrientation ───────────────────────────────────────────────────────
// Reading direction for headers.

enum HeaderOrientation {
  Top
    @description("Headers above data. Records read top-to-bottom. Default for Standard and CrossTab layouts.")
  Left
    @description("Headers in leftmost column. Records read left-to-right. Co-occurs with Transposed layout. Left column contains field-name labels.")
}

// ─── PageSectionType ─────────────────────────────────────────────────────────
// What spatial text reveals about a region on the page.

enum PageSectionType {
  TableHeader
    @description(#"
      Column labels vertically aligned with data columns below. Defines the
      schema of the table that follows. May be SingleRow, Stacked, or
      Hierarchical. Contains label-like text: short words, field names, units.
      KEY SIGNS: Horizontally spaced labels aligned with data columns below.
      Precedes a DataGrid section. No numeric data values.
    "#)
  DataGrid
    @description(#"
      Dense rectangular grid of data values with consistent column positions
      across rows. The body of a table.
      KEY SIGNS: Uniform cell count. Consistent column alignment within 3-5
      char tolerance. Data-value content (numbers, dates, text, codes).
    "#)
  SectionHeading
    @description(#"
      Standalone text naming a group or section. NOT part of a table — labels
      the table block that follows. In multi-table pages, separates blocks
      and provides context (group names, category names).
      KEY SIGNS: Short isolated text, often uppercase. Not aligned with table
      columns. Precedes a table block.
    "#)
  Metadata
    @description(#"
      Document-level information: titles, dates, key-value pairs. Found at
      page top/bottom, outside the main table area.
      KEY SIGNS: 'Label: Value' format. Page header/footer position.
    "#)
  Prose
    @description(#"
      Flowing paragraph text — sentences, not structured data. Should NOT be
      parsed as table data.
      KEY SIGNS: Full sentences with grammar and punctuation. No column
      alignment. Text wraps at page margin. Single span per row.
    "#)
  AggregationRow
    @description(#"
      Summary row with computed values (totals, subtotals). Same column
      alignment as table data but contains aggregate values, not records.
      Should be excluded from data extraction.
      KEY SIGNS: Contains 'Total', 'Subtotal', or similar label. At end of
      table block or section. Values sum the column above.
    "#)
  Footnote
    @description(#"
      Explanatory text, disclaimers, source attributions below main content.
      KEY SIGNS: Bottom of page. Reference markers. Qualifying language.
      Full-width text below tables.
    "#)
}

// ─── MultiRowPattern ─────────────────────────────────────────────────────────
// Whether a single record spans multiple rows.

enum MultiRowPattern {
  SingleRow
    @description("Each row is a complete record. No merging needed. KEY SIGNS: Uniform rows, no alternating row types.")
  RepeatingGroup
    @description(#"
      A single record spans 2-4 consecutive rows following a repeating pattern.
      Example: type-A values on row 1, type-B values on row 2, type-C values
      on row 3 (period=3). Sub-rows at the same column position must be merged.
      KEY SIGNS: Repeating span-count sequence. Alternating row types with
      different value kinds per row. At least 2 complete cycles.
    "#)
}

// ─── TableStructure ──────────────────────────────────────────────────────────
// Bundled classification output for a single table.

class TableStructure {
  layout TableLayout
    @description("Geometric arrangement of records and fields.")
  header_structure HeaderStructure
    @description("How column labels are organized vertically.")
  header_orientation HeaderOrientation
    @description("Whether labels are above (Top) or left of (Left) data.")
  header_row_count int
    @description("Number of rows occupied by headers. Data begins after this count.")
  multi_row_pattern MultiRowPattern
    @description("Whether records span multiple rows.")
  multi_row_period int?
    @description("Rows per record when multi_row_pattern is RepeatingGroup. Null otherwise.")
  data_column_count int
    @description("Number of data columns in the table body.")
  column_names string[]
    @description("Resolved column names, one per data column. For Stacked headers, these are the vertically concatenated names. For Hierarchical, these are compound 'Parent / Child' paths.")
  notes string?
    @description("Observations about structure: ambiguities, confidence, unusual patterns.")
}

// ─── PageSection ─────────────────────────────────────────────────────────────
// Section-level characterization of a page region.

class PageSection {
  type PageSectionType
    @description("What kind of content this section contains.")
  label string?
    @description("Text label for this section (e.g. heading text, group name). Null for unlabeled sections.")
  row_range string?
    @description("Row range in the spatial grid, e.g. '0-3' or '4-35'.")
}
