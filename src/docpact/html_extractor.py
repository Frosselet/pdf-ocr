"""HTML table extraction.

Extracts structured tables from HTML using selectolax for fast parsing.
Handles:
- Standard <table> elements with colspan/rowspan
- Inline styles for visual information
- Class names for semantic hints (static parsing only, no JS)

Note: Does not support JavaScript-rendered content. For dynamic pages,
pre-render with a headless browser before passing to this extractor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from docpact.heuristics import (
    StructuredTable,
    TableMetadata,
    build_column_names_from_headers,
    estimate_header_rows,
    normalize_grid,
    split_headers_from_data,
)

if TYPE_CHECKING:
    from selectolax.parser import HTMLParser, Node


# ---------------------------------------------------------------------------
# Visual element structures
# ---------------------------------------------------------------------------


@dataclass
class HtmlCellStyle:
    """Visual style information for a table cell."""

    background_color: tuple[float, float, float] | None = None
    has_border: bool = False
    is_bold: bool = False
    is_header_tag: bool = False  # True if <th> tag


@dataclass
class HtmlVisualInfo:
    """Visual information for the entire table."""

    header_fill_rows: list[int] = field(default_factory=list)
    has_thead: bool = False  # Table has explicit <thead>
    header_row_count: int = 0  # Rows inside <thead> or <th> count


@dataclass
class HtmlTable:
    """A table extracted from HTML."""

    grid: list[list[str]]
    styles: list[list[HtmlCellStyle]] | None = None
    visual: HtmlVisualInfo | None = None
    table_index: int = 0


# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------


def _parse_color_value(color_str: str | None) -> tuple[float, float, float] | None:
    """Parse CSS color value to RGB tuple (0-1 range).

    Handles:
    - Hex: #RGB, #RRGGBB
    - RGB: rgb(R, G, B)
    - Named colors (subset)
    """
    if not color_str:
        return None

    color_str = color_str.strip().lower()

    # Skip transparent/inherit
    if color_str in ("transparent", "inherit", "initial", "none"):
        return None

    # Hex format
    if color_str.startswith("#"):
        hex_val = color_str[1:]
        if len(hex_val) == 3:
            hex_val = "".join(c * 2 for c in hex_val)
        if len(hex_val) == 6:
            try:
                r = int(hex_val[0:2], 16) / 255.0
                g = int(hex_val[2:4], 16) / 255.0
                b = int(hex_val[4:6], 16) / 255.0
                if (r, g, b) == (1.0, 1.0, 1.0):
                    return None  # Skip white
                return (r, g, b)
            except ValueError:
                return None

    # RGB format
    rgb_match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color_str)
    if rgb_match:
        try:
            r = int(rgb_match.group(1)) / 255.0
            g = int(rgb_match.group(2)) / 255.0
            b = int(rgb_match.group(3)) / 255.0
            if (r, g, b) == (1.0, 1.0, 1.0):
                return None
            return (r, g, b)
        except ValueError:
            return None

    # Common named colors (subset)
    named_colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 0.5, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "orange": (1.0, 0.65, 0.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
        "lightgray": (0.83, 0.83, 0.83),
        "lightgrey": (0.83, 0.83, 0.83),
        "darkgray": (0.66, 0.66, 0.66),
        "darkgrey": (0.66, 0.66, 0.66),
    }
    return named_colors.get(color_str)


def _parse_inline_style(style_str: str | None) -> dict[str, str]:
    """Parse inline CSS style string to dict."""
    if not style_str:
        return {}

    result: dict[str, str] = {}
    for part in style_str.split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            result[key.strip().lower()] = value.strip()
    return result


def _extract_cell_style(node: "Node") -> HtmlCellStyle:
    """Extract visual style from a table cell."""
    style = HtmlCellStyle()

    # Check tag name
    tag = node.tag.lower() if node.tag else ""
    style.is_header_tag = tag == "th"

    # Check for bold (tag or style)
    if style.is_header_tag:
        style.is_bold = True

    # Parse inline style
    style_str = node.attributes.get("style", "")
    css = _parse_inline_style(style_str)

    # Background color
    bg = css.get("background-color") or css.get("background")
    style.background_color = _parse_color_value(bg)

    # Border
    border = css.get("border")
    if border and border not in ("none", "0"):
        style.has_border = True

    # Font weight
    font_weight = css.get("font-weight", "")
    if font_weight in ("bold", "700", "800", "900"):
        style.is_bold = True

    return style


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------


def _get_cell_text(node: "Node") -> str:
    """Get text content from a cell, stripping extra whitespace."""
    text = node.text(deep=True, separator=" ")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_table(table_node: "Node") -> HtmlTable:
    """Parse an HTML <table> element to a grid.

    Handles colspan and rowspan attributes.
    """
    grid: list[list[str]] = []
    styles: list[list[HtmlCellStyle]] = []
    visual = HtmlVisualInfo()

    # Find all rows (in thead, tbody, tfoot, or directly in table)
    rows: list["Node"] = []
    thead_row_count = 0

    # Check for thead
    thead = table_node.css_first("thead")
    if thead:
        visual.has_thead = True
        thead_rows = thead.css("tr")
        thead_row_count = len(thead_rows)
        rows.extend(thead_rows)

    # Check for tbody (or rows directly in table)
    tbody = table_node.css_first("tbody")
    if tbody:
        rows.extend(tbody.css("tr"))
    else:
        # Get direct tr children (excluding thead/tfoot ones)
        for tr in table_node.css("tr"):
            # Skip if already added from thead
            parent = tr.parent
            if parent and parent.tag and parent.tag.lower() == "thead":
                continue
            if parent and parent.tag and parent.tag.lower() == "tfoot":
                continue
            # Skip duplicates
            if tr not in rows:
                rows.append(tr)

    # Check for tfoot
    tfoot = table_node.css_first("tfoot")
    if tfoot:
        rows.extend(tfoot.css("tr"))

    if not rows:
        return HtmlTable(grid=[], styles=[], visual=visual)

    # First pass: determine grid dimensions
    num_rows = len(rows)
    num_cols = 0

    for tr in rows:
        cells = tr.css("td, th")
        row_cols = 0
        for cell in cells:
            colspan = int(cell.attributes.get("colspan", 1) or 1)
            row_cols += colspan
        num_cols = max(num_cols, row_cols)

    if num_cols == 0:
        return HtmlTable(grid=[], styles=[], visual=visual)

    # Initialize grid with empty cells
    grid = [[""] * num_cols for _ in range(num_rows)]
    styles = [[HtmlCellStyle() for _ in range(num_cols)] for _ in range(num_rows)]

    # Track rowspan continuation
    # rowspan_map[col] = (end_row, text, style)
    rowspan_map: dict[int, tuple[int, str, HtmlCellStyle]] = {}

    # Second pass: fill grid
    for row_idx, tr in enumerate(rows):
        cells = tr.css("td, th")
        col_idx = 0

        for cell in cells:
            # Skip past columns occupied by rowspan from previous rows
            while col_idx < num_cols and col_idx in rowspan_map:
                if rowspan_map[col_idx][0] > row_idx:
                    col_idx += 1
                else:
                    del rowspan_map[col_idx]
                    break

            if col_idx >= num_cols:
                break

            # Get cell attributes
            colspan = int(cell.attributes.get("colspan", 1) or 1)
            rowspan = int(cell.attributes.get("rowspan", 1) or 1)

            # Get cell content and style
            text = _get_cell_text(cell)
            cell_style = _extract_cell_style(cell)

            # Fill grid cells
            for r_offset in range(rowspan):
                for c_offset in range(colspan):
                    ri = row_idx + r_offset
                    ci = col_idx + c_offset
                    if ri < num_rows and ci < num_cols:
                        if r_offset == 0 and c_offset == 0:
                            grid[ri][ci] = text
                            styles[ri][ci] = cell_style
                        else:
                            grid[ri][ci] = ""  # Merged cell
                            styles[ri][ci] = cell_style

            # Track rowspan for future rows
            if rowspan > 1:
                for c_offset in range(colspan):
                    rowspan_map[col_idx + c_offset] = (
                        row_idx + rowspan,
                        text,
                        cell_style,
                    )

            col_idx += colspan

    # Analyze visual structure
    visual.header_row_count = thead_row_count

    # Check for header fills
    row_fills: list[tuple[float, float, float] | None] = []
    for ri, row in enumerate(styles[:5]):
        fills = [s.background_color for s in row if s.background_color is not None]
        if fills:
            row_fills.append(fills[0])
        else:
            row_fills.append(None)

    if row_fills and row_fills[0] is not None:
        first_fill = row_fills[0]
        for ri, fill in enumerate(row_fills):
            if fill == first_fill:
                visual.header_fill_rows.append(ri)
            else:
                break

    return HtmlTable(grid=grid, styles=styles, visual=visual)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tables_from_html(
    html_input: str | Path,
    *,
    extract_styles: bool = True,
) -> list[StructuredTable]:
    """Extract structured tables from HTML content.

    Args:
        html_input: HTML string or path to .html file
        extract_styles: If True, extract visual style information.

    Returns:
        List of StructuredTable objects, one per <table> found.

    Raises:
        ImportError: If selectolax is not installed.
        FileNotFoundError: If path is given and file doesn't exist.
    """
    try:
        from selectolax.parser import HTMLParser
    except ImportError as e:
        raise ImportError(
            "selectolax is required for HTML extraction. "
            "Install with: pip install docpact[html]"
        ) from e

    # Load HTML content
    if isinstance(html_input, Path):
        if not html_input.exists():
            raise FileNotFoundError(f"File not found: {html_input}")
        html_content = html_input.read_text(encoding="utf-8")
    elif isinstance(html_input, str) and not html_input.strip().startswith("<"):
        # Assume it's a path
        path = Path(html_input)
        if path.exists():
            html_content = path.read_text(encoding="utf-8")
        else:
            # Treat as HTML content
            html_content = html_input
    else:
        html_content = html_input

    # Parse HTML
    parser = HTMLParser(html_content)
    result: list[StructuredTable] = []

    # Find all tables
    tables = parser.css("table")

    for table_idx, table_node in enumerate(tables):
        # Parse table
        html_table = _parse_table(table_node)
        html_table.table_index = table_idx

        if not html_table.grid or not html_table.grid[0]:
            continue

        # Normalize grid
        grid = normalize_grid(html_table.grid)

        # Estimate header rows
        header_count = 0
        if html_table.visual:
            if html_table.visual.has_thead and html_table.visual.header_row_count > 0:
                header_count = html_table.visual.header_row_count
            elif html_table.visual.header_fill_rows:
                header_count = len(html_table.visual.header_fill_rows)

        if header_count == 0:
            # Check if first row is all <th> tags
            if html_table.styles and html_table.styles[0]:
                if all(s.is_header_tag for s in html_table.styles[0]):
                    header_count = 1

        if header_count == 0:
            header_count = estimate_header_rows(grid)

        # Split headers from data
        header_rows, data_rows = split_headers_from_data(grid, header_count)

        # Build column names
        column_names = build_column_names_from_headers(header_rows)

        # Ensure correct column count
        num_cols = len(grid[0]) if grid else 0
        while len(column_names) < num_cols:
            column_names.append("")

        # Create StructuredTable
        result.append(StructuredTable(
            column_names=column_names,
            data=data_rows,
            source_format="html",
            metadata={
                "table_index": table_idx,
                "has_thead": html_table.visual.has_thead if html_table.visual else False,
            },
        ))

    return result


def html_to_markdown(html_input: str | Path) -> str:
    """Extract tables from HTML and render as markdown.

    Convenience function for quick inspection of HTML content.
    """
    tables = extract_tables_from_html(html_input, extract_styles=False)

    parts: list[str] = []
    for idx, table in enumerate(tables):
        parts.append(f"## Table {idx + 1}\n")

        if table.column_names:
            parts.append("| " + " | ".join(table.column_names) + " |")
            parts.append("| " + " | ".join(["---"] * len(table.column_names)) + " |")

        for row in table.data:
            parts.append("| " + " | ".join(row) + " |")

        parts.append("")  # Blank line between tables

    return "\n".join(parts)
