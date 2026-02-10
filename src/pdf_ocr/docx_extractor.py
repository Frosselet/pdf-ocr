"""Word (DOCX/DOC) table extraction.

Extracts structured tables from Word documents.
- DOCX (Office 2007+): Uses python-docx
- DOC (Office 97-2003): Converts via LibreOffice if available, or raises error

DOCX tables have explicit grid structure, but require handling of:
- Horizontal merges (gridSpan)
- Vertical merges (vMerge)
- Nested tables (treated as text content)

Semantic heuristics from heuristics.py apply for header detection
and type pattern analysis.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pdf_ocr.heuristics import (
    StructuredTable,
    TableMetadata,
    build_column_names_from_headers,
    estimate_header_rows,
    normalize_grid,
    split_headers_from_data,
)

if TYPE_CHECKING:
    from docx.document import Document
    from docx.table import Table, _Cell


# ---------------------------------------------------------------------------
# Visual element structures
# ---------------------------------------------------------------------------


@dataclass
class DocxCellStyle:
    """Visual style information for a table cell."""

    shading_color: tuple[float, float, float] | None = None  # RGB 0-1
    border_top: bool = False
    border_bottom: bool = False
    border_left: bool = False
    border_right: bool = False
    is_bold: bool = False
    is_italic: bool = False
    font_size: float | None = None


@dataclass
class DocxVisualInfo:
    """Visual information for the entire table."""

    header_fill_rows: list[int] = field(default_factory=list)
    has_borders: bool = False


@dataclass
class DocxTable:
    """A table extracted from a Word document with visual info."""

    grid: list[list[str]]
    styles: list[list[DocxCellStyle]] | None = None
    visual: DocxVisualInfo | None = None
    table_index: int = 0


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------


def _parse_hex_color(hex_color: str | None) -> tuple[float, float, float] | None:
    """Parse hex color string to RGB tuple (0-1 range).

    Handles formats: RRGGBB, #RRGGBB, auto, None
    """
    if not hex_color or hex_color.lower() in ("auto", "none"):
        return None

    # Strip # if present
    hex_color = hex_color.lstrip("#")

    if len(hex_color) != 6:
        return None

    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        # Skip pure white (typically means no fill)
        if (r, g, b) == (1.0, 1.0, 1.0):
            return None
        return (r, g, b)
    except ValueError:
        return None


def _extract_cell_style(cell: "_Cell") -> DocxCellStyle:
    """Extract visual style information from a cell."""
    style = DocxCellStyle()

    # Shading (fill color)
    try:
        tc = cell._tc
        tc_pr = tc.tcPr
        if tc_pr is not None:
            shading = tc_pr.shd
            if shading is not None:
                fill = shading.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}fill")
                if fill:
                    style.shading_color = _parse_hex_color(fill)
    except Exception:
        pass

    # Check paragraphs for font info
    try:
        for para in cell.paragraphs:
            for run in para.runs:
                if run.bold:
                    style.is_bold = True
                if run.italic:
                    style.is_italic = True
                if run.font.size:
                    style.font_size = run.font.size.pt
    except Exception:
        pass

    return style


# ---------------------------------------------------------------------------
# Merged cell handling
# ---------------------------------------------------------------------------


def _get_cell_text(cell: "_Cell") -> str:
    """Get text content from a cell, handling nested tables."""
    # Collect text from all paragraphs
    text_parts: list[str] = []
    for para in cell.paragraphs:
        text = para.text.strip()
        if text:
            text_parts.append(text)

    # Note: Nested tables are ignored (their text is in paragraphs)
    return " ".join(text_parts)


def _build_grid_from_table(table: "Table") -> tuple[list[list[str]], list[list[DocxCellStyle]]]:
    """Build a grid from a Word table, handling merged cells.

    Word tables use:
    - gridSpan for horizontal merges (cell spans multiple columns)
    - vMerge for vertical merges (cell continues from above)

    Returns (grid, styles) where grid is list of rows, each row is list of strings.
    """
    # First pass: determine grid dimensions
    num_rows = len(table.rows)
    if num_rows == 0:
        return [], []

    # Count columns from first row (gridSpan-aware)
    num_cols = 0
    for cell in table.rows[0].cells:
        try:
            tc = cell._tc
            grid_span = tc.tcPr.gridSpan if tc.tcPr else None
            span = int(grid_span.val) if grid_span is not None else 1
            num_cols += span
        except Exception:
            num_cols += 1

    if num_cols == 0:
        return [], []

    # Initialize grid
    grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
    styles: list[list[DocxCellStyle]] = [[DocxCellStyle() for _ in range(num_cols)] for _ in range(num_rows)]

    # Track vertical merge state per column
    # vmerge_continue[col] = (row, text, style) of the cell that started the merge
    vmerge_start: dict[int, tuple[int, str, DocxCellStyle]] = {}

    for row_idx, row in enumerate(table.rows):
        col_idx = 0

        for cell in row.cells:
            # Skip if we've already filled this position due to previous cell's span
            while col_idx < num_cols and grid[row_idx][col_idx]:
                col_idx += 1

            if col_idx >= num_cols:
                break

            # Get cell properties
            try:
                tc = cell._tc
                tc_pr = tc.tcPr

                # Grid span (horizontal merge)
                grid_span = tc_pr.gridSpan if tc_pr else None
                h_span = int(grid_span.val) if grid_span is not None else 1

                # Vertical merge
                v_merge = tc_pr.vMerge if tc_pr else None

            except Exception:
                h_span = 1
                v_merge = None

            # Get cell text and style
            cell_text = _get_cell_text(cell)
            cell_style = _extract_cell_style(cell)

            # Handle vertical merge
            if v_merge is not None:
                # Check if this is a continue (val is None or "continue")
                val = v_merge.val if hasattr(v_merge, "val") else None
                if val is None or val == "continue":
                    # This cell continues from the merge start above
                    # Use empty string (the real value is in the start cell)
                    for span_offset in range(h_span):
                        if col_idx + span_offset < num_cols:
                            grid[row_idx][col_idx + span_offset] = ""
                            styles[row_idx][col_idx + span_offset] = cell_style
                    col_idx += h_span
                    continue
                else:
                    # This is a merge restart - treat as new cell
                    vmerge_start[col_idx] = (row_idx, cell_text, cell_style)

            # Fill grid cells for this cell's span
            for span_offset in range(h_span):
                if col_idx + span_offset < num_cols:
                    if span_offset == 0:
                        grid[row_idx][col_idx] = cell_text
                        styles[row_idx][col_idx] = cell_style
                    else:
                        # Spanned cells get empty string
                        grid[row_idx][col_idx + span_offset] = ""
                        styles[row_idx][col_idx + span_offset] = cell_style

            col_idx += h_span

    return grid, styles


# ---------------------------------------------------------------------------
# Visual analysis
# ---------------------------------------------------------------------------


def _analyze_visual_structure(
    styles: list[list[DocxCellStyle]],
) -> DocxVisualInfo:
    """Analyze visual structure of the table."""
    visual = DocxVisualInfo()

    if not styles or not styles[0]:
        return visual

    # Check for header fills in first few rows
    row_fills: list[tuple[float, float, float] | None] = []
    for ri, row in enumerate(styles[:5]):
        fills = [s.shading_color for s in row if s.shading_color is not None]
        if fills:
            row_fills.append(fills[0])
        else:
            row_fills.append(None)

    # Find header rows with consistent fill
    if row_fills and row_fills[0] is not None:
        first_fill = row_fills[0]
        for ri, fill in enumerate(row_fills):
            if fill == first_fill:
                visual.header_fill_rows.append(ri)
            else:
                break

    return visual


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tables_from_docx(
    docx_path: str | Path,
    *,
    extract_styles: bool = True,
) -> list[StructuredTable]:
    """Extract structured tables from a Word document.

    Args:
        docx_path: Path to the .docx file
        extract_styles: If True, extract visual style information.

    Returns:
        List of StructuredTable objects, one per table found.

    Raises:
        ImportError: If python-docx is not installed.
        FileNotFoundError: If the file doesn't exist.
    """
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX extraction. "
            "Install with: pip install pdf-ocr[docx]"
        ) from e

    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")

    doc: Document = Document(path)
    result: list[StructuredTable] = []

    for table_idx, table in enumerate(doc.tables):
        # Extract grid
        grid, styles = _build_grid_from_table(table)

        if not grid or not grid[0]:
            continue

        # Normalize grid
        grid = normalize_grid(grid)

        # Analyze visual structure
        visual = _analyze_visual_structure(styles) if extract_styles else None

        # Estimate header rows
        header_count = 0
        if visual and visual.header_fill_rows:
            header_count = len(visual.header_fill_rows)
        else:
            header_count = estimate_header_rows(grid)

        # Split headers from data
        header_rows, data_rows = split_headers_from_data(grid, header_count)

        # Build column names
        column_names = build_column_names_from_headers(header_rows)

        # Ensure column names list has correct length
        num_cols = len(grid[0]) if grid else 0
        while len(column_names) < num_cols:
            column_names.append("")

        # Create metadata
        metadata = TableMetadata(
            page_number=0,  # DOCX doesn't have clear page boundaries
            table_index=table_idx,
            section_label=None,
        )

        # Create StructuredTable
        result.append(StructuredTable(
            column_names=column_names,
            data=data_rows,
            source_format="docx",
            metadata={
                "page_number": metadata.page_number,
                "table_index": metadata.table_index,
            },
        ))

    return result


# ---------------------------------------------------------------------------
# DOC (Legacy Word 97-2003) extraction via conversion
# ---------------------------------------------------------------------------


def _find_libreoffice() -> str | None:
    """Find LibreOffice executable on the system."""
    # Common paths for LibreOffice
    candidates = [
        "libreoffice",
        "soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",  # macOS
        "/usr/bin/libreoffice",
        "/usr/bin/soffice",
        "C:\\Program Files\\LibreOffice\\program\\soffice.exe",  # Windows
    ]

    for candidate in candidates:
        if shutil.which(candidate):
            return candidate

    return None


def _convert_doc_to_docx(doc_path: Path, output_dir: Path) -> Path | None:
    """Convert a .doc file to .docx using LibreOffice.

    Returns the path to the converted file, or None if conversion fails.
    """
    libreoffice = _find_libreoffice()
    if not libreoffice:
        return None

    try:
        # Run LibreOffice in headless mode to convert
        result = subprocess.run(
            [
                libreoffice,
                "--headless",
                "--convert-to", "docx",
                "--outdir", str(output_dir),
                str(doc_path),
            ],
            capture_output=True,
            timeout=60,  # 1 minute timeout
        )

        if result.returncode != 0:
            return None

        # Find the converted file
        expected_name = doc_path.stem + ".docx"
        converted_path = output_dir / expected_name

        if converted_path.exists():
            return converted_path

        return None

    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None


def _extract_tables_from_doc(
    doc_path: str | Path,
    *,
    extract_styles: bool = True,
) -> list[StructuredTable]:
    """Extract tables from a legacy .doc file by converting to .docx first.

    Requires LibreOffice to be installed for conversion.
    """
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")

    # Check if LibreOffice is available
    if not _find_libreoffice():
        raise ImportError(
            "LibreOffice is required to read legacy .doc files. "
            "Please install LibreOffice or convert the file to .docx format. "
            "Download: https://www.libreoffice.org/download/"
        )

    # Convert in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        converted = _convert_doc_to_docx(path, tmpdir_path)

        if not converted:
            raise RuntimeError(
                f"Failed to convert {path.name} to .docx format. "
                "Please convert the file manually or check your LibreOffice installation."
            )

        # Extract from the converted file
        result = extract_tables_from_docx(converted, extract_styles=extract_styles)

        # Update source format in metadata
        for table in result:
            table.source_format = "doc"

        return result


def extract_tables_from_word(
    word_path: str | Path,
    *,
    extract_styles: bool = True,
) -> list[StructuredTable]:
    """Extract structured tables from a Word document (DOC or DOCX).

    Auto-detects format based on file extension and uses the appropriate
    extraction method.

    Args:
        word_path: Path to the Word file (.docx or .doc)
        extract_styles: If True, extract visual style information.

    Returns:
        List of StructuredTable objects.

    Note: .doc files require LibreOffice for conversion.
    """
    path = Path(word_path)
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return extract_tables_from_docx(path, extract_styles=extract_styles)
    elif suffix == ".doc":
        return _extract_tables_from_doc(path, extract_styles=extract_styles)
    else:
        # Try DOCX first (more common), fall back to DOC
        try:
            return extract_tables_from_docx(path, extract_styles=extract_styles)
        except Exception:
            return _extract_tables_from_doc(path, extract_styles=extract_styles)


def docx_to_markdown(docx_path: str | Path) -> str:
    """Extract tables from Word and render as markdown.

    Convenience function for quick inspection of Word content.
    """
    tables = extract_tables_from_docx(docx_path, extract_styles=False)

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
