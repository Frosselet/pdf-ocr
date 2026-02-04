"""Spatial PDF-to-text renderer.

Converts each PDF page into a monospace text grid where every text span
appears at its correct (x, y) position — preserving columns, tables, and
scattered text exactly as they appear visually.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


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


def _extract_page_layout(
    page: fitz.Page, cluster_threshold: float = 2.0
) -> PageLayout | None:
    """Extract structured layout from a PDF page.

    Returns a PageLayout with row-grouped spans and column positions,
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
    for row_idx, cluster in enumerate(row_clusters):
        for y in cluster:
            y_to_row[y] = row_idx

    # Group spans by row.
    x_min = min(s["x"] for s in spans)
    rows: dict[int, list[tuple[int, str]]] = {}
    for s in spans:
        row_idx = y_to_row[s["y"]]
        col = round((s["x"] - x_min) / cell_w)
        rows.setdefault(row_idx, []).append((col, s["text"]))

    return PageLayout(
        rows=rows,
        row_count=len(row_clusters),
        cell_w=cell_w,
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
