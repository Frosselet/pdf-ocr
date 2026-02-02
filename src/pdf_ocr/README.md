# pdf_ocr — Spatial PDF Text Extraction

## Problem

Standard PDF text extraction (PyMuPDF's `get_text()`, `pymupdf4llm`) returns text in stream order — the order objects were written into the PDF file. This rarely matches visual layout. Columns get interleaved, table cells merge into a single line, and scattered labels lose their spatial meaning.

This is especially problematic for tabular documents like shipping stems, loading statements, and financial reports where the relationship between values depends entirely on their position on the page.

## Approach: Spatial Grid Rendering

Instead of reordering text heuristically, we project every text span onto a **monospace character grid** that mirrors the physical page. The result is plain text where columns, tables, and scattered labels appear exactly where they sit in the PDF.

### Pipeline

```
PDF page
  │
  ▼
1. Extract spans ─── page.get_text("dict") ──► list of {text, origin(x,y), bbox}
  │
  ▼
2. Compute cell width ─── median(bbox_width / len(text)) ──► adaptive char width in points
  │
  ▼
3. Cluster y-coords ─── greedy merge within 2pt gap ──► row groups
  │
  ▼
4. Map to grid ─── col = round((x - x_min) / cell_w) ──► (row, col, text) triples
  │
  ▼
5. Render ─── character buffer per row, last-writer-wins ──► plain text
```

### Step 1: Span Extraction

We use `page.get_text("dict")` which returns every text span with:
- **`text`** — the string content
- **`origin`** — the (x, y) baseline coordinate in PDF points (72pt = 1 inch)
- **`bbox`** — the bounding box `[x0, y0, x1, y1]`

Spans are the finest granularity that preserves font/style runs. We strip whitespace-only spans.

### Step 2: Dynamic Cell Width

Fixed character widths fail because PDFs use arbitrary font sizes. Instead, we compute the **median character width** across the page:

```
cell_w = median( (bbox.x1 - bbox.x0) / len(text) )  for all spans with len >= 2
```

- Using the **median** (not mean) makes this robust to outliers like titles in large fonts or footnotes in small fonts.
- We require spans with **2+ characters** to avoid division noise from single-char spans (e.g., bullets, pipe characters).
- Fallback to 6.0pt if a page has only single-char spans.

This single value defines the grid resolution for the entire page.

### Step 3: Row Clustering

PDF text on the "same line" often has slightly different y-coordinates due to font metrics, subscripts, or rendering jitter. We cluster y-values into rows:

1. Sort all unique y-values
2. Greedily merge consecutive values within a **2.0pt threshold**

The 2pt threshold handles sub-point jitter without falsely merging adjacent rows (typical line spacing is 10–14pt).

### Step 4: Grid Mapping

Each span maps to a grid position:

```
col = round((span.x - page_x_min) / cell_w)
row = cluster index from step 3
```

Subtracting `page_x_min` anchors the leftmost text at column 0, avoiding wasted leading whitespace.

### Step 5: Rendering

For each row, we allocate a character buffer and write each span starting at its computed column:

```
buf = [' '] * max_width
for col, text in row_spans:
    for i, ch in enumerate(text):
        buf[col + i] = ch
```

**Last-writer-wins** for overlaps — if two spans occupy the same grid cell, the later one in document order takes precedence. This is the simplest conflict resolution and works well in practice since true overlaps are rare.

Trailing whitespace is stripped from each line.

## API

```python
from pdf_ocr import pdf_to_spatial_text

text = pdf_to_spatial_text("document.pdf")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pdf_path` | `str` | required | Path to the PDF file |
| `pages` | `list[int] \| None` | `None` | 0-based page indices to render; `None` = all |
| `cluster_threshold` | `float` | `2.0` | Max y-distance (points) to merge into one row |
| `page_separator` | `str` | `"\f"` | String inserted between rendered pages |

### Return Value

A single string with all pages joined by `page_separator`. Each page is a block of lines where text spans appear at their spatial grid positions.

## Limitations

- **Monospace assumption**: The grid uses a single character width per page. Proportional fonts will have slight alignment imperfections, though the median-based cell width minimizes this.
- **No image/drawing extraction**: Only text spans are rendered.
- **Rotation**: Rotated text is placed at its origin point but the text direction is not adjusted.
- **Very dense pages**: Pages with many overlapping elements at different sizes may produce wide lines due to the single grid resolution.
