"""Spatial PDF-to-text renderer.

Converts each PDF page into a monospace text grid where every text span
appears at its correct (x, y) position — preserving columns, tables, and
scattered text exactly as they appear visually.

Also extracts visual elements (lines, fills) for cross-validation with
text-based heuristics.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Visual element data structures
# ---------------------------------------------------------------------------


@dataclass
class VisualLine:
    """A horizontal or vertical line extracted from PDF drawings."""

    orientation: Literal["horizontal", "vertical"]
    start: float  # x for vertical, y for horizontal
    end: float  # x for vertical, y for horizontal
    position: float  # y for horizontal, x for vertical
    width: float  # stroke width
    color: tuple[float, float, float] | None  # RGB 0-1


@dataclass
class VisualFill:
    """A filled rectangle (cell/row background)."""

    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    color: tuple[float, float, float] | None  # RGB 0-1


@dataclass
class VisualElements:
    """All visual elements extracted from a page."""

    h_lines: list[VisualLine] = field(default_factory=list)
    v_lines: list[VisualLine] = field(default_factory=list)
    fills: list[VisualFill] = field(default_factory=list)


def _open_pdf(pdf_input: str | bytes | Path) -> fitz.Document:
    """Open a PDF from various input types.

    Args:
        pdf_input: Path string, Path object, or raw PDF bytes.

    Returns:
        An open fitz.Document.
    """
    if isinstance(pdf_input, bytes):
        return fitz.open(stream=pdf_input, filetype="pdf")
    return fitz.open(pdf_input)


@dataclass
class PageLayout:
    """Structured layout extracted from a single PDF page."""

    rows: dict[int, list[tuple[int, str]]] = field(default_factory=dict)
    row_count: int = 0
    cell_w: float = 6.0
    visual: VisualElements | None = None  # Visual elements (lines, fills)
    row_y_positions: list[float] = field(default_factory=list)  # Actual Y coordinates


# ---------------------------------------------------------------------------
# Visual element extraction
# ---------------------------------------------------------------------------


def _extract_visual_elements(page: fitz.Page) -> VisualElements:
    """Extract visual elements (lines and fills) from a PDF page.

    Uses PyMuPDF's get_drawings() to retrieve all drawing commands. Classifies
    each drawing as either:
    - A line (thin horizontal/vertical fill, height or width < 2pt)
    - A cell/row fill (thicker rectangle)

    Note: Many PDFs encode lines as thin fills rather than strokes. This function
    handles both cases by detecting thin fills and converting them to lines.

    Returns:
        VisualElements with h_lines, v_lines, and fills.
    """
    h_lines: list[VisualLine] = []
    v_lines: list[VisualLine] = []
    fills: list[VisualFill] = []

    # Line detection threshold: fills thinner than this are treated as lines
    line_threshold = 2.0

    drawings = page.get_drawings()

    for d in drawings:
        draw_type = d.get("type")
        rect = d.get("rect")
        if rect is None:
            continue

        x0, y0, x1, y1 = rect
        width = x1 - x0
        height = y1 - y0

        if draw_type == "s":
            # Stroke (actual line)
            color = d.get("color")
            if color and len(color) >= 3:
                color = (color[0], color[1], color[2])
            else:
                color = None
            stroke_width = d.get("width", 1.0)

            # Classify as horizontal or vertical
            if width > height:
                h_lines.append(VisualLine(
                    orientation="horizontal",
                    start=x0,
                    end=x1,
                    position=(y0 + y1) / 2,
                    width=stroke_width,
                    color=color,
                ))
            else:
                v_lines.append(VisualLine(
                    orientation="vertical",
                    start=y0,
                    end=y1,
                    position=(x0 + x1) / 2,
                    width=stroke_width,
                    color=color,
                ))

        elif draw_type == "f":
            # Fill
            fill_color = d.get("fill")
            if fill_color and len(fill_color) >= 3:
                fill_color = (fill_color[0], fill_color[1], fill_color[2])
            else:
                fill_color = None

            # Check if this fill is actually a line (very thin)
            if height < line_threshold and width > height * 5:
                # Horizontal line encoded as thin fill
                h_lines.append(VisualLine(
                    orientation="horizontal",
                    start=x0,
                    end=x1,
                    position=(y0 + y1) / 2,
                    width=height,
                    color=fill_color,
                ))
            elif width < line_threshold and height > width * 5:
                # Vertical line encoded as thin fill
                v_lines.append(VisualLine(
                    orientation="vertical",
                    start=y0,
                    end=y1,
                    position=(x0 + x1) / 2,
                    width=width,
                    color=fill_color,
                ))
            else:
                # Regular cell/row fill
                fills.append(VisualFill(
                    bbox=(x0, y0, x1, y1),
                    color=fill_color,
                ))

    return VisualElements(h_lines=h_lines, v_lines=v_lines, fills=fills)


def _extract_page_layout(
    page: fitz.Page,
    cluster_threshold: float = 2.0,
    extract_visual: bool = False,
) -> PageLayout | None:
    """Extract structured layout from a PDF page.

    Args:
        page: PyMuPDF page object.
        cluster_threshold: Maximum y-distance to merge into the same row.
        extract_visual: If True, also extract visual elements (lines, fills).

    Returns:
        PageLayout with row-grouped spans and column positions,
        or None if the page contains no text.
    """
    data = page.get_text("dict")
    spans: list[dict] = []
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                spans.append({
                    "text": text,
                    "x": span["origin"][0],
                    "y": span["origin"][1],
                    "bbox": span["bbox"],
                })

    if not spans:
        return None

    # Compute dynamic cell width: median of (bbox_width / char_count) for
    # spans with 2+ characters.
    char_widths: list[float] = []
    for s in spans:
        n = len(s["text"])
        if n >= 2:
            bbox_w = s["bbox"][2] - s["bbox"][0]
            char_widths.append(bbox_w / n)

    if not char_widths:
        cell_w = 6.0  # fallback for pages with only single-char spans
    else:
        cell_w = statistics.median(char_widths)

    # Cluster y-coordinates into rows using greedy merge.
    y_values = sorted({s["y"] for s in spans})
    row_clusters: list[list[float]] = []
    for y in y_values:
        if row_clusters and abs(y - row_clusters[-1][-1]) <= cluster_threshold:
            row_clusters[-1].append(y)
        else:
            row_clusters.append([y])

    # Map each y-value to its cluster representative (mean y).
    y_to_row: dict[float, int] = {}
    row_y_positions: list[float] = []  # Mean Y position for each row
    for row_idx, cluster in enumerate(row_clusters):
        for y in cluster:
            y_to_row[y] = row_idx
        row_y_positions.append(sum(cluster) / len(cluster))

    # Group spans by row.
    x_min = min(s["x"] for s in spans)
    rows: dict[int, list[tuple[int, str]]] = {}
    for s in spans:
        row_idx = y_to_row[s["y"]]
        col = round((s["x"] - x_min) / cell_w)
        rows.setdefault(row_idx, []).append((col, s["text"]))

    # Extract visual elements if requested
    visual = _extract_visual_elements(page) if extract_visual else None

    return PageLayout(
        rows=rows,
        row_count=len(row_clusters),
        cell_w=cell_w,
        visual=visual,
        row_y_positions=row_y_positions,
    )


def _render_page_grid(page: fitz.Page, cluster_threshold: float = 2.0) -> str:
    """Render a single PDF page as a spatial text grid."""
    layout = _extract_page_layout(page, cluster_threshold)
    if layout is None:
        return ""

    # Render each row into a character buffer.
    lines: list[str] = []
    for row_idx in range(layout.row_count):
        if row_idx not in layout.rows:
            lines.append("")
            continue
        entries = layout.rows[row_idx]
        # Determine buffer size needed.
        max_end = max(col + len(text) for col, text in entries)
        buf = [" "] * max_end
        for col, text in entries:
            for i, ch in enumerate(text):
                pos = col + i
                if pos < len(buf):
                    buf[pos] = ch
        lines.append("".join(buf).rstrip())

    return "\n".join(lines)


def pdf_to_spatial_text(
    pdf_input: str | bytes | Path,
    *,
    pages: list[int] | None = None,
    cluster_threshold: float = 2.0,
    page_separator: str = "\f",
) -> str:
    """Convert a PDF to spatial text preserving visual layout.

    Args:
        pdf_input: Path to PDF file (str or Path), or raw PDF bytes.
        pages: Optional list of 0-based page indices to render.
            If None, all pages are rendered.
        cluster_threshold: Maximum y-distance (in points) to merge into
            the same text row. Default 2.0.
        page_separator: String inserted between pages. Default is form-feed.

    Returns:
        A single string with all rendered pages joined by *page_separator*.
    """
    doc = _open_pdf(pdf_input)
    page_indices = pages if pages is not None else range(len(doc))
    rendered: list[str] = []
    for idx in page_indices:
        rendered.append(_render_page_grid(doc[idx], cluster_threshold))
    doc.close()
    return page_separator.join(rendered)
