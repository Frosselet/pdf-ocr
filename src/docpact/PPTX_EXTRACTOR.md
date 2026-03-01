# PowerPoint Table Extraction

> **Module**: `pptx_extractor.py`
> **Public API**: `extract_tables_from_pptx()`, `extract_tables_from_powerpoint()`, `pptx_to_markdown()`

Extracts structured tables from PowerPoint presentations. Handles both explicit table shapes (python-pptx `Table` objects) and text shapes arranged in table-like patterns via geometry heuristics.

## Why

PPTX has two sources of tabular data:

1. **Table shapes**: Explicit tables with grid structure — straightforward extraction via `Table.rows` and `Table.cells`.
2. **Text shapes**: Text boxes arranged spatially to *look* like a table — requires geometry heuristics (H2 y-clustering, H9 x-alignment) to reconstruct the grid.

Legacy `.ppt` files are supported via LibreOffice conversion to `.pptx`.

## Pipeline

```text
PowerPoint file (.pptx or .ppt)
  |
  v
1. Format detection
   -- .pptx: Direct python-pptx parsing
   -- .ppt: Convert via LibreOffice → .pptx, then parse
  |
  v
2. Per slide:
  |
  +---> Table shapes
  |     -- Extract grid from Table.rows / cells
  |     -- Extract cell styles (fill, bold, italic, font size)
  |     -- Analyze visual structure (header fills)
  |
  +---> Text shapes (if include_shape_tables=True)
        -- Extract TextSpan objects (text + x/y/width/height)
        -- H2: Y-clustering → row detection
        -- H9: X-position alignment → column unification
        -- Consistency check (span count variance)
        -- Convert to grid
  |
  v
3. Per table:
   -- Normalize grid (pad rows)
   -- Estimate header rows (visual fill or H7)
   -- Split headers from data
   -- Build column names
  |
  v
4. Return StructuredTable objects
```

## Heuristics

### H2: Y-Position Clustering (`_cluster_y_positions`)

Groups text shapes by vertical position. Shapes with y-positions within a threshold (default 5pt) are placed in the same row. Greedy clustering from sorted positions.

### H9: X-Position Alignment (`_unify_columns`)

Builds canonical column positions from all text shape x-positions. Shapes at similar x-positions (within tolerance, default 10pt) are assigned to the same column. Produces a mapping from each shape's x-position to a column index.

### Shape-Table Detection (`_detect_table_from_shapes`)

After clustering, validates that the result looks like a table:
- Minimum row count (default 3)
- Minimum column count (default 2)
- Span count consistency (max - min ≤ 3 per row)

## Internal Types

| Type | Purpose |
|------|---------|
| `PptxCellStyle` | Per-cell: fill color, bold, italic, font size |
| `PptxVisualInfo` | Per-table: header fill rows |
| `PptxTable` | Intermediate: grid, styles, visual, slide/table indices, `from_shapes` flag |
| `TextSpan` | Text box with position (x, y, width, height) and bold flag |

## API

### `extract_tables_from_pptx()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pptx_path` | `str \| Path` | required | Path to .pptx file |
| `slides` | `list[int] \| None` | `None` | 0-based slide indices (None = all) |
| `include_shape_tables` | `bool` | `True` | Also detect tables from text shapes |
| `extract_styles` | `bool` | `True` | Extract visual style information |

Returns `list[StructuredTable]` with `source_format="pptx"`.

Metadata fields per table:

| Field | Type | Description |
|-------|------|-------------|
| `slide_number` | `int` | Slide index (0-based) |
| `table_index` | `int` | Table position within the slide |
| `from_shapes` | `bool` | True if reconstructed from text shapes |

### `extract_tables_from_powerpoint()`

Auto-detects `.pptx` vs `.ppt` format and dispatches accordingly. Same parameters as `extract_tables_from_pptx()`. Legacy `.ppt` files require LibreOffice.

### `pptx_to_markdown()`

Convenience function. Extracts tables and renders as pipe-table markdown with `## Slide N` headers (with ` (from shapes)` suffix for shape-reconstructed tables).

## Dependencies

- `python-pptx` (`pip install docpact[pptx]`) — for PPTX parsing
- LibreOffice — optional, for legacy `.ppt` conversion
