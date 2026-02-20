# Spatial Text Extraction

> **Module**: `spatial_text.py`
> **Public API**: `pdf_to_spatial_text()`

Converts PDF pages into monospace text grids where every text span appears at its correct (x, y) position — preserving columns, tables, and scattered text exactly as they appear visually.

## The Problem

Standard PDF text extraction (`PyMuPDF.get_text()`, `pymupdf4llm`) returns text in **stream order** — the order objects were written into the PDF file. This rarely matches visual layout:

- Columns get interleaved
- Table cells merge into single lines
- Scattered labels lose spatial meaning

This is catastrophic for tabular documents where the relationship between values depends entirely on position.

## The Solution: Spatial Grid Rendering

Project every text span onto a **monospace character grid** that mirrors the physical page.

```
PDF page
  │
  ▼
1. Extract spans ─── page.get_text("dict") ──► list of {text, origin(x,y), bbox}
  │
  ▼
2. Compute cell width ─── median(bbox_width / len(text)) ──► adaptive char width
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

---

## Heuristics

### H1: Dynamic Cell Width (`_extract_page_layout`)

**Problem**: Fixed character widths fail because PDFs use arbitrary font sizes.

**Solution**: Compute the **median character width** across all spans:

```python
cell_w = median( (bbox.x1 - bbox.x0) / len(text) )  # for spans with len >= 2
```

**Why median?**
- Robust to outliers (large titles, tiny footnotes)
- Single-char spans excluded (noisy width data)
- Fallback to 6.0pt if page has only single-char spans

**Effect**: A single grid resolution per page that adapts to the dominant font.

---

### H2: Row Clustering (`_extract_page_layout`)

**Problem**: Text on the "same line" often has slightly different y-coordinates due to:
- Font metrics differences
- Subscripts/superscripts
- Rendering jitter

**Solution**: Greedy y-coordinate clustering with **2.0pt threshold**:

```python
for y in sorted(y_values):
    if cluster and abs(y - cluster[-1]) <= 2.0:
        cluster.append(y)  # merge into existing row
    else:
        clusters.append([y])  # start new row
```

**Why 2pt?**
- Handles sub-point jitter
- Typical line spacing is 10-14pt, so no false merges
- Configurable via `cluster_threshold` parameter

---

### H3: Column Mapping

**Problem**: How to convert continuous x-coordinates to discrete grid columns?

**Solution**: Simple linear mapping anchored at leftmost text:

```python
col = round((span.x - page_x_min) / cell_w)
```

**Effect**: Leftmost text at column 0, columns spaced by character width.

---

### H4: Last-Writer-Wins Rendering

**Problem**: What if two spans occupy the same grid cell?

**Solution**: Later spans in document order overwrite earlier ones:

```python
for col, text in row_spans:
    for i, ch in enumerate(text):
        buf[col + i] = ch  # overwrites previous
```

**Rationale**: True overlaps are rare; when they occur, later content is usually the correction.

---

## API

```python
from docpact import pdf_to_spatial_text

text = pdf_to_spatial_text("document.pdf")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pdf_input` | `str \| bytes \| Path` | required | Path to PDF or raw bytes |
| `pages` | `list[int] \| None` | `None` | 0-based page indices; `None` = all |
| `cluster_threshold` | `float` | `2.0` | Max y-distance (points) to merge rows |
| `page_separator` | `str` | `"\f"` | String between pages |

### Return Value

Single string with all pages joined by `page_separator`. Each page is a block of lines where text spans appear at their spatial grid positions.

---

## Visual Element Extraction

In addition to text spans, `spatial_text.py` can extract visual elements (lines, fills) from PDF drawing commands for cross-validation with text heuristics.

### Data Types

```python
@dataclass
class VisualLine:
    orientation: Literal["horizontal", "vertical"]
    start: float   # x for vertical, y for horizontal
    end: float     # x for vertical, y for horizontal
    position: float  # y for horizontal, x for vertical
    width: float   # stroke width
    color: tuple[float, float, float] | None  # RGB 0-1

@dataclass
class VisualFill:
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    color: tuple[float, float, float] | None  # RGB 0-1

@dataclass
class VisualElements:
    h_lines: list[VisualLine]
    v_lines: list[VisualLine]
    fills: list[VisualFill]
```

### How It Works

`_extract_visual_elements(page)` uses PyMuPDF's `page.get_drawings()` to retrieve all drawing commands:

1. **Strokes** (type='s') → Converted to `VisualLine`
2. **Thin fills** (height or width < 2pt) → Converted to `VisualLine` (many PDFs encode lines as thin rectangles)
3. **Regular fills** → Converted to `VisualFill` (cell backgrounds, header highlights)

### PageLayout Enhancement

```python
@dataclass
class PageLayout:
    rows: dict[int, list[tuple[int, str]]]
    row_count: int
    cell_w: float
    visual: VisualElements | None  # NEW: Visual elements
    row_y_positions: list[float]   # NEW: Actual Y coordinates per row
```

The `row_y_positions` field stores the mean Y coordinate for each row cluster, enabling correlation between visual fills and text rows.

### Usage

```python
from docpact.spatial_text import _extract_page_layout, _open_pdf

doc = _open_pdf("document.pdf")
layout = _extract_page_layout(doc[0], extract_visual=True)

if layout.visual:
    print(f"H-lines: {len(layout.visual.h_lines)}")
    print(f"V-lines: {len(layout.visual.v_lines)}")
    print(f"Fills: {len(layout.visual.fills)}")
```

---

## Limitations

- **Monospace assumption**: Single character width per page. Proportional fonts have slight misalignment.
- **No images**: Only text spans extracted.
- **Rotated text**: Placed at origin but direction not adjusted.
- **Dense pages**: Many overlapping elements produce wide lines.
