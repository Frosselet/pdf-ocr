"""Compressed spatial text for LLM consumption.

Takes a PDF and produces a token-efficient structured text representation
using markdown tables, flowing text, and key-value pairs instead of
whitespace-heavy spatial grids.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import fitz  # PyMuPDF

from pdf_ocr.spatial_text import (
    FontSpan,
    PageLayout,
    VisualElements,
    VisualFill,
    VisualLine,
    _extract_page_layout,
    _open_pdf,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell type detection — for TH1/TH2 heuristics
# ---------------------------------------------------------------------------


class CellType(Enum):
    """Classification of cell content for type pattern analysis."""

    DATE = "date"
    NUMBER = "number"
    ENUM = "enum"  # Repeated categorical value
    STRING = "string"  # Default fallback


# Date patterns for type detection (subset of serialize.py patterns)
_DATE_PATTERNS = [
    # ISO formats
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # YYYY-MM-DD
    re.compile(r"^\d{4}-\d{2}$"),  # YYYY-MM
    re.compile(r"^\d{4}$"),  # YYYY alone (might be year)
    # Slashed formats
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),  # D/M/Y or M/D/Y
    re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),  # D-M-Y or M-D-Y
    # Time formats
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$"),  # HH:MM or HH:MM:SS
    re.compile(r"^\d{1,2}:\d{2}\s*[AaPp][Mm]$"),  # HH:MM AM/PM
    # Combined date-time
    re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}"),  # ISO datetime
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}"),  # D/M/Y HH:MM
    # Month names
    re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$"),  # January 1, 2025
    re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$"),  # 1 January 2025
    re.compile(r"^[A-Za-z]{3,9}\s+\d{4}$"),  # January 2025
]

# Number pattern: digits with optional separators, signs, currency
_NUMBER_PATTERN = re.compile(
    r"^[($€£¥+-]?\s*"  # Optional leading sign/currency/paren
    r"[\d,.\s]+"  # Digits with separators
    r"\s*[)%]?$"  # Optional trailing paren/percent
)


def _detect_cell_type(text: str | None) -> CellType:
    """Classify a cell's content as DATE, NUMBER, or STRING.

    Note: ENUM detection requires column-level analysis (repeated values)
    and cannot be done at the single-cell level. This function returns
    STRING for potential enum values; use _detect_column_types() for
    proper enum detection.
    """
    if text is None:
        return CellType.STRING

    text = str(text).strip()
    if not text:
        return CellType.STRING

    # Check for date patterns first
    for pattern in _DATE_PATTERNS:
        if pattern.match(text):
            return CellType.DATE

    # Check for number pattern
    if _NUMBER_PATTERN.match(text):
        # Verify it's actually numeric (not just punctuation)
        cleaned = re.sub(r"[($€£¥+\-,.\s)%]", "", text)
        if cleaned.isdigit():
            return CellType.NUMBER

    return CellType.STRING


def _detect_column_types(
    grid: list[list[str]],
    skip_header_rows: int = 0,
) -> list[CellType]:
    """Detect the predominant type for each column.

    Analyzes data rows to determine the most common type per column.
    ENUM is detected when a column has few distinct values (≤10% of rows
    or ≤5 unique values) and those values repeat.

    Args:
        grid: Table grid (list of rows, each row is list of cell strings)
        skip_header_rows: Number of header rows to skip in analysis

    Returns:
        List of CellType, one per column
    """
    if not grid:
        return []

    num_cols = len(grid[0]) if grid else 0
    data_rows = grid[skip_header_rows:]

    if not data_rows or num_cols == 0:
        return [CellType.STRING] * num_cols

    # Collect values and types per column
    col_values: list[list[str]] = [[] for _ in range(num_cols)]
    col_types: list[Counter[CellType]] = [Counter() for _ in range(num_cols)]

    for row in data_rows:
        for ci, cell in enumerate(row):
            if ci >= num_cols:
                break
            cell = cell.strip()
            if cell:
                col_values[ci].append(cell)
                col_types[ci][_detect_cell_type(cell)] += 1

    # Determine predominant type per column
    result: list[CellType] = []
    for ci in range(num_cols):
        values = col_values[ci]
        type_counts = col_types[ci]

        if not values:
            result.append(CellType.STRING)
            continue

        # Check for enum: few unique values that repeat
        unique_values = set(values)
        unique_ratio = len(unique_values) / len(values) if values else 1.0
        is_enum = (
            len(unique_values) <= 5
            or unique_ratio <= 0.1
        ) and len(values) >= 3  # Need at least 3 values to detect enum

        if is_enum and type_counts.get(CellType.STRING, 0) > 0:
            result.append(CellType.ENUM)
        elif type_counts:
            # Most common type wins
            result.append(type_counts.most_common(1)[0][0])
        else:
            result.append(CellType.STRING)

    return result


def _row_type_signature(row: list[str]) -> list[CellType]:
    """Get the type signature of a row (list of cell types)."""
    return [_detect_cell_type(cell) for cell in row]


def _is_header_type_pattern(row: list[str]) -> bool:
    """Check if a row has a header-like type pattern (all strings, no dates/numbers).

    TH1 heuristic: Headers are labels (strings), not data (dates, numbers, enums).
    """
    if not row:
        return False

    for cell in row:
        cell_type = _detect_cell_type(cell.strip())
        if cell_type in (CellType.DATE, CellType.NUMBER):
            return False

    return True


# ---------------------------------------------------------------------------
# Visual heuristics — cross-validation with text heuristics
# ---------------------------------------------------------------------------


@dataclass
class VisualGridInfo:
    """Grid structure detected from visual elements (VH1)."""

    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    h_line_count: int
    v_line_count: int
    has_grid: bool  # ≥3 h-lines AND ≥3 v-lines with intersections


@dataclass
class VisualHeaderInfo:
    """Header highlighting detected from visual elements (VH2)."""

    header_fill_rows: list[int]  # Row indices with header-like fills
    header_color: tuple[float, float, float] | None


@dataclass
class VisualZebraInfo:
    """Zebra striping detected from visual elements (VH3)."""

    zebra_start_row: int | None
    zebra_end_row: int | None
    alternating_colors: tuple[tuple[float, float, float], tuple[float, float, float]] | None


@dataclass
class VisualSeparatorInfo:
    """Section separators detected from visual elements (VH4)."""

    separator_y_positions: list[float]  # Y positions of thick horizontal lines


@dataclass
class VisualValidation:
    """Result of visual heuristics analysis."""

    grid: VisualGridInfo | None = None
    header: VisualHeaderInfo | None = None
    zebra: VisualZebraInfo | None = None
    separators: VisualSeparatorInfo | None = None

    # Cross-validation results
    has_visual_elements: bool = False


@dataclass
class CrossValidationResult:
    """Result of comparing text and visual heuristics."""

    # Where both channels agree
    agreements: list[str] = field(default_factory=list)

    # Visual detected something text missed
    text_gaps: list[str] = field(default_factory=list)

    # Text detected something visual missed (often valid - whitespace tables)
    visual_gaps: list[str] = field(default_factory=list)

    # Contradictions requiring investigation
    contradictions: list[str] = field(default_factory=list)


def _detect_visual_grid(
    visual: VisualElements,
    page_width: float,
    page_height: float,
) -> VisualGridInfo | None:
    """VH1: Detect table grid boundaries from visual elements.

    A grid exists if:
    - ≥3 horizontal lines AND ≥3 vertical lines
    - Lines form a bounded structure (intersect or align)
    """
    if not visual.h_lines and not visual.v_lines:
        return None

    # Filter significant lines (spanning reasonable portion of table)
    min_h_len = page_width * 0.1  # At least 10% of page width
    min_v_len = page_height * 0.02  # At least 2% of page height

    sig_h = [l for l in visual.h_lines if (l.end - l.start) >= min_h_len]
    sig_v = [l for l in visual.v_lines if (l.end - l.start) >= min_v_len]

    if len(sig_h) < 3 or len(sig_v) < 3:
        return None

    # Compute bounding box of grid
    h_positions = [l.position for l in sig_h]
    v_positions = [l.position for l in sig_v]

    if not h_positions or not v_positions:
        return None

    bbox = (
        min(v_positions),  # x0
        min(h_positions),  # y0
        max(v_positions),  # x1
        max(h_positions),  # y1
    )

    return VisualGridInfo(
        bbox=bbox,
        h_line_count=len(sig_h),
        v_line_count=len(sig_v),
        has_grid=True,
    )


def _detect_visual_header(
    visual: VisualElements,
    row_y_positions: list[float],
    tolerance: float = 5.0,
) -> VisualHeaderInfo | None:
    """VH2: Detect header background highlighting.

    Headers typically have a distinct fill color in the top portion
    of the table region.
    """
    if not visual.fills or not row_y_positions:
        return None

    # Find fills that overlap with the top few rows
    first_few_rows = row_y_positions[:5] if len(row_y_positions) >= 5 else row_y_positions

    if not first_few_rows:
        return None

    top_y = min(first_few_rows)
    header_zone_end = max(first_few_rows) + tolerance

    # Find fills in the header zone
    header_fills: list[VisualFill] = []
    for fill in visual.fills:
        _, y0, _, y1 = fill.bbox
        # Fill overlaps with header zone
        if y0 <= header_zone_end and y1 >= top_y - tolerance:
            header_fills.append(fill)

    if not header_fills:
        return None

    # Group fills by color
    color_counts: Counter[tuple[float, float, float]] = Counter()
    for fill in header_fills:
        if fill.color:
            color_counts[fill.color] += 1

    if not color_counts:
        return None

    # Most common color is likely the header color
    header_color = color_counts.most_common(1)[0][0]

    # Find which rows have this header color
    header_rows: list[int] = []
    for row_idx, row_y in enumerate(row_y_positions[:10]):  # Check first 10 rows
        for fill in header_fills:
            _, fy0, _, fy1 = fill.bbox
            if fill.color == header_color and fy0 <= row_y <= fy1:
                header_rows.append(row_idx)
                break

    if not header_rows:
        return None

    return VisualHeaderInfo(
        header_fill_rows=header_rows,
        header_color=header_color,
    )


def _detect_visual_zebra(
    visual: VisualElements,
    row_y_positions: list[float],
    tolerance: float = 5.0,
) -> VisualZebraInfo | None:
    """VH3: Detect data row alternation (zebra striping).

    Zebra pattern: alternating fill colors across consecutive rows.
    """
    if not visual.fills or len(row_y_positions) < 4:
        return None

    # Map each row to its fill color (if any)
    row_colors: dict[int, tuple[float, float, float] | None] = {}

    for row_idx, row_y in enumerate(row_y_positions):
        row_color = None
        for fill in visual.fills:
            _, fy0, _, fy1 = fill.bbox
            if fy0 - tolerance <= row_y <= fy1 + tolerance:
                row_color = fill.color
                break
        row_colors[row_idx] = row_color

    # Look for alternating pattern in middle section (skip potential headers)
    start_check = min(3, len(row_y_positions) // 4)
    end_check = len(row_y_positions)

    # Count consecutive alternations
    colors_seen: set[tuple[float, float, float]] = set()
    alternation_count = 0
    prev_color = None

    zebra_start = None
    zebra_end = None

    for row_idx in range(start_check, end_check):
        color = row_colors.get(row_idx)
        if color is None:
            continue

        if prev_color is not None and color != prev_color:
            alternation_count += 1
            if zebra_start is None:
                zebra_start = row_idx - 1
            zebra_end = row_idx
            colors_seen.add(color)
            colors_seen.add(prev_color)

        prev_color = color

    # Need at least 4 alternations (8 rows) to confirm zebra pattern
    if alternation_count >= 4 and len(colors_seen) == 2:
        colors_list = list(colors_seen)
        return VisualZebraInfo(
            zebra_start_row=zebra_start,
            zebra_end_row=zebra_end,
            alternating_colors=(colors_list[0], colors_list[1]),
        )

    return None


def _detect_visual_separators(
    visual: VisualElements,
    table_width: float,
    median_line_width: float | None = None,
) -> VisualSeparatorInfo | None:
    """VH4: Detect section separator lines.

    Separators are horizontal lines that:
    - Span most of the table width (>80%)
    - Are thicker than regular cell borders
    """
    if not visual.h_lines:
        return None

    # Calculate median line width if not provided
    if median_line_width is None and visual.h_lines:
        widths = [l.width for l in visual.h_lines if l.width > 0]
        if widths:
            widths.sort()
            median_line_width = widths[len(widths) // 2]
        else:
            median_line_width = 1.0

    # Find separator lines
    separator_positions: list[float] = []
    width_threshold = table_width * 0.8
    thickness_threshold = median_line_width * 1.5 if median_line_width else 1.5

    for line in visual.h_lines:
        line_len = line.end - line.start
        if line_len >= width_threshold and line.width >= thickness_threshold:
            separator_positions.append(line.position)

    if not separator_positions:
        return None

    return VisualSeparatorInfo(separator_y_positions=sorted(separator_positions))


def _analyze_visual_structure(
    visual: VisualElements | None,
    row_y_positions: list[float],
    page_width: float = 612.0,  # Default letter width
    page_height: float = 792.0,  # Default letter height
) -> VisualValidation:
    """Run all visual heuristics (VH1-VH6) on extracted visual elements.

    Args:
        visual: Visual elements from page (may be None)
        row_y_positions: Y positions of text rows for alignment
        page_width: Page width for relative calculations
        page_height: Page height for relative calculations

    Returns:
        VisualValidation with results from each heuristic
    """
    if visual is None:
        return VisualValidation(has_visual_elements=False)

    has_elements = bool(visual.h_lines or visual.v_lines or visual.fills)
    if not has_elements:
        return VisualValidation(has_visual_elements=False)

    # Run each visual heuristic
    grid = _detect_visual_grid(visual, page_width, page_height)
    header = _detect_visual_header(visual, row_y_positions)
    zebra = _detect_visual_zebra(visual, row_y_positions)

    # For separators, estimate table width from grid or page
    table_width = page_width
    if grid and grid.has_grid:
        table_width = grid.bbox[2] - grid.bbox[0]

    separators = _detect_visual_separators(visual, table_width)

    return VisualValidation(
        grid=grid,
        header=header,
        zebra=zebra,
        separators=separators,
        has_visual_elements=True,
    )


def _cross_validate_header_rows(
    text_header_count: int,
    visual: VisualValidation,
    data_rows: list[list[str]],
) -> CrossValidationResult:
    """Cross-validate header detection between text and visual heuristics.

    Implements VH6 (exception highlighting) and TH1/TH2 (type patterns).
    """
    result = CrossValidationResult()

    if not visual.has_visual_elements:
        return result

    # Compare text-based header count with visual header detection
    if visual.header:
        visual_header_count = len(visual.header.header_fill_rows)

        if visual_header_count == text_header_count:
            result.agreements.append(
                f"Header rows: text={text_header_count}, visual={visual_header_count} (match)"
            )
        else:
            # Check for VH6: exception highlighting
            # If visual suggests more headers but type pattern says data...
            if visual_header_count > text_header_count and data_rows:
                first_disputed_row = data_rows[0] if data_rows else []
                if first_disputed_row and not _is_header_type_pattern(first_disputed_row):
                    result.agreements.append(
                        f"VH6: Row {text_header_count} has header color but data types "
                        f"(exception highlight, not header)"
                    )
                else:
                    result.contradictions.append(
                        f"Header mismatch: text={text_header_count}, visual={visual_header_count}"
                    )
            else:
                result.contradictions.append(
                    f"Header mismatch: text={text_header_count}, visual={visual_header_count}"
                )

    return result


# ---------------------------------------------------------------------------
# Font heuristics — cross-validation with text/visual heuristics
# ---------------------------------------------------------------------------


@dataclass
class FontHierarchyInfo:
    """Font size hierarchy detected from spans (FH1).

    Maps font sizes to structural roles (title, header, body, footnote).
    """

    size_tiers: list[float]  # Distinct size clusters, largest first
    header_size: float | None  # Size associated with header zone
    body_size: float | None  # Most common size in data zone


@dataclass
class FontBoldInfo:
    """Bold pattern analysis (FH2).

    Bold text often signals headers or labels.
    """

    header_bold_ratio: float  # Fraction of bold spans in header zone
    data_bold_ratio: float  # Fraction of bold spans in data zone
    bold_header_rows: list[int]  # Row indices where >50% spans are bold


@dataclass
class FontItalicInfo:
    """Italic pattern analysis (FH3).

    Italic often signals secondary content: captions, citations, metadata.
    """

    italic_positions: list[tuple[int, int]]  # (row, col) of italic spans
    has_caption_below: bool  # Italic text below table area
    metadata_rows: list[int]  # Rows that are entirely/mostly italic


@dataclass
class FontMonospaceInfo:
    """Monospace pattern analysis (FH4).

    Monospace fonts typically indicate structured data: codes, IDs, numbers.
    """

    monospace_columns: list[int]  # Column indices with >50% monospace
    monospace_ratio: float  # Overall monospace span ratio


@dataclass
class FontColorInfo:
    """Text color pattern analysis (FH5).

    Colors encode semantics: red=error, blue=link, gray=secondary.
    """

    header_text_color: tuple[float, float, float] | None  # Most common header color
    exception_colors: list[tuple[float, float, float]]  # Red, orange = warning
    color_groups: dict[tuple[float, float, float], int]  # color -> count


@dataclass
class FontConsistencyInfo:
    """Font family consistency analysis (FH6).

    Multiple font families per column may indicate data quality issues.
    """

    header_font_family: str | None  # Normalized font family in headers
    data_font_family: str | None  # Most common font family in data
    inconsistent_columns: list[int]  # Columns with >2 font families


@dataclass
class FontValidation:
    """Result of font heuristics analysis (FH1-FH6)."""

    hierarchy: FontHierarchyInfo | None = None
    bold: FontBoldInfo | None = None
    italic: FontItalicInfo | None = None
    monospace: FontMonospaceInfo | None = None
    color: FontColorInfo | None = None
    consistency: FontConsistencyInfo | None = None
    has_font_data: bool = False


def _normalize_font_family(font_name: str) -> str:
    """Normalize font name to family (strip style suffixes).

    E.g., "Arial-BoldMT" -> "arial", "TimesNewRomanPS-ItalicMT" -> "timesnewromanps"
    """
    # Remove common style suffixes
    name = font_name.lower()
    for suffix in ("-bold", "-italic", "-regular", "-medium", "-light", "mt", "-oblique"):
        name = name.replace(suffix, "")
    # Remove any remaining suffixes after the last hyphen
    if "-" in name:
        name = name.rsplit("-", 1)[0]
    return name.strip()


def _cluster_sizes(sizes: list[float], gap_threshold: float = 0.2) -> list[float]:
    """Cluster font sizes into tiers.

    Adjacent sizes within gap_threshold proportion of each other are merged.
    Returns cluster representatives (means), sorted largest first.
    """
    if not sizes:
        return []

    sorted_sizes = sorted(set(sizes), reverse=True)
    if len(sorted_sizes) == 1:
        return sorted_sizes

    clusters: list[list[float]] = [[sorted_sizes[0]]]
    for size in sorted_sizes[1:]:
        cluster_mean = sum(clusters[-1]) / len(clusters[-1])
        # Gap threshold: 20% difference means different tier
        if (cluster_mean - size) / cluster_mean > gap_threshold:
            clusters.append([size])
        else:
            clusters[-1].append(size)

    return [sum(c) / len(c) for c in clusters]


def _detect_font_hierarchy(
    font_spans: dict[int, list[tuple[int, FontSpan]]],
    row_y_positions: list[float],
    header_row_estimate: int,
) -> FontHierarchyInfo | None:
    """FH1: Detect font size hierarchy.

    Larger font = more prominent. Cluster sizes into tiers and map to roles.
    """
    if not font_spans:
        return None

    all_sizes: list[float] = []
    header_sizes: list[float] = []
    data_sizes: list[float] = []

    for row_idx, spans in font_spans.items():
        for _, font in spans:
            all_sizes.append(font.size)
            if row_idx < header_row_estimate:
                header_sizes.append(font.size)
            else:
                data_sizes.append(font.size)

    if not all_sizes:
        return None

    size_tiers = _cluster_sizes(all_sizes)

    # Header size = most common in header zone (or largest if no clear winner)
    header_size = None
    if header_sizes:
        size_counts: Counter[float] = Counter(header_sizes)
        header_size = size_counts.most_common(1)[0][0]

    # Body size = most common in data zone
    body_size = None
    if data_sizes:
        size_counts = Counter(data_sizes)
        body_size = size_counts.most_common(1)[0][0]

    return FontHierarchyInfo(
        size_tiers=size_tiers,
        header_size=header_size,
        body_size=body_size,
    )


def _detect_bold_headers(
    font_spans: dict[int, list[tuple[int, FontSpan]]],
    header_row_estimate: int,
) -> FontBoldInfo | None:
    """FH2: Detect bold header pattern.

    Bold = labels, not values. High bold ratio in headers, low in data = pattern.
    """
    if not font_spans:
        return None

    header_total = 0
    header_bold = 0
    data_total = 0
    data_bold = 0
    bold_row_counts: dict[int, tuple[int, int]] = {}  # row -> (bold_count, total)

    for row_idx, spans in font_spans.items():
        row_bold = 0
        row_total = len(spans)
        for _, font in spans:
            if font.is_bold:
                row_bold += 1
                if row_idx < header_row_estimate:
                    header_bold += 1
                else:
                    data_bold += 1

        if row_idx < header_row_estimate:
            header_total += row_total
        else:
            data_total += row_total

        bold_row_counts[row_idx] = (row_bold, row_total)

    header_bold_ratio = header_bold / header_total if header_total > 0 else 0.0
    data_bold_ratio = data_bold / data_total if data_total > 0 else 0.0

    # Rows where >50% spans are bold
    bold_header_rows = [
        row_idx
        for row_idx, (bold_count, total) in bold_row_counts.items()
        if total > 0 and bold_count / total > 0.5
    ]

    return FontBoldInfo(
        header_bold_ratio=header_bold_ratio,
        data_bold_ratio=data_bold_ratio,
        bold_header_rows=sorted(bold_header_rows),
    )


def _detect_italic_patterns(
    font_spans: dict[int, list[tuple[int, FontSpan]]],
    row_y_positions: list[float],
    header_row_estimate: int,
) -> FontItalicInfo | None:
    """FH3: Detect italic pattern.

    Italic = secondary content (captions, citations, metadata).
    """
    if not font_spans:
        return None

    italic_positions: list[tuple[int, int]] = []
    row_italic_ratios: dict[int, float] = {}

    for row_idx, spans in font_spans.items():
        italic_count = 0
        for col, font in spans:
            if font.is_italic:
                italic_positions.append((row_idx, col))
                italic_count += 1
        if spans:
            row_italic_ratios[row_idx] = italic_count / len(spans)

    # Check for caption below table: italic text in last few rows
    max_row = max(font_spans.keys()) if font_spans else 0
    has_caption_below = any(
        row_italic_ratios.get(ri, 0) > 0.5
        for ri in range(max(0, max_row - 2), max_row + 1)
    )

    # Rows that are entirely/mostly italic (>80%)
    metadata_rows = [
        row_idx
        for row_idx, ratio in row_italic_ratios.items()
        if ratio > 0.8
    ]

    return FontItalicInfo(
        italic_positions=italic_positions,
        has_caption_below=has_caption_below,
        metadata_rows=sorted(metadata_rows),
    )


def _detect_monospace_patterns(
    font_spans: dict[int, list[tuple[int, FontSpan]]],
    header_row_estimate: int,
) -> FontMonospaceInfo | None:
    """FH4: Detect monospace pattern.

    Monospace = structured data (codes, IDs, numbers).
    """
    if not font_spans:
        return None

    # Track monospace by column (in data rows only)
    column_monospace: Counter[int] = Counter()
    column_total: Counter[int] = Counter()
    total_spans = 0
    monospace_spans = 0

    for row_idx, spans in font_spans.items():
        if row_idx < header_row_estimate:
            continue  # Skip headers
        for col, font in spans:
            column_total[col] += 1
            total_spans += 1
            if font.is_monospace:
                column_monospace[col] += 1
                monospace_spans += 1

    # Columns with >50% monospace
    monospace_columns = [
        col
        for col in column_total
        if column_total[col] > 0 and column_monospace[col] / column_total[col] > 0.5
    ]

    monospace_ratio = monospace_spans / total_spans if total_spans > 0 else 0.0

    return FontMonospaceInfo(
        monospace_columns=sorted(monospace_columns),
        monospace_ratio=monospace_ratio,
    )


def _detect_color_patterns(
    font_spans: dict[int, list[tuple[int, FontSpan]]],
    header_row_estimate: int,
) -> FontColorInfo | None:
    """FH5: Detect text color patterns.

    Colors encode semantics: red=error, blue=link, gray=secondary.
    """
    if not font_spans:
        return None

    header_colors: Counter[tuple[float, float, float]] = Counter()
    all_colors: Counter[tuple[float, float, float]] = Counter()

    # Exception colors (red/orange hues)
    exception_colors: set[tuple[float, float, float]] = set()

    for row_idx, spans in font_spans.items():
        for _, font in spans:
            color = font.color
            if color is None:
                color = (0.0, 0.0, 0.0)  # Black

            all_colors[color] += 1
            if row_idx < header_row_estimate:
                header_colors[color] += 1

            # Detect exception colors (red/orange: high R, low-medium G, low B)
            r, g, b = color
            if r > 0.6 and g < 0.5 and b < 0.3:
                exception_colors.add(color)

    header_text_color = header_colors.most_common(1)[0][0] if header_colors else None

    return FontColorInfo(
        header_text_color=header_text_color,
        exception_colors=list(exception_colors),
        color_groups=dict(all_colors),
    )


def _detect_font_consistency(
    font_spans: dict[int, list[tuple[int, FontSpan]]],
    header_row_estimate: int,
) -> FontConsistencyInfo | None:
    """FH6: Detect font family consistency.

    >2 font families per column may indicate data quality issues.
    """
    if not font_spans:
        return None

    header_families: Counter[str] = Counter()
    data_families: Counter[str] = Counter()
    column_families: dict[int, set[str]] = {}

    for row_idx, spans in font_spans.items():
        for col, font in spans:
            family = _normalize_font_family(font.font_name)
            if row_idx < header_row_estimate:
                header_families[family] += 1
            else:
                data_families[family] += 1
                column_families.setdefault(col, set()).add(family)

    header_font_family = header_families.most_common(1)[0][0] if header_families else None
    data_font_family = data_families.most_common(1)[0][0] if data_families else None

    # Columns with >2 distinct font families
    inconsistent_columns = [
        col for col, families in column_families.items() if len(families) > 2
    ]

    return FontConsistencyInfo(
        header_font_family=header_font_family,
        data_font_family=data_font_family,
        inconsistent_columns=sorted(inconsistent_columns),
    )


def _analyze_font_structure(
    font_spans: dict[int, list[tuple[int, FontSpan]]],
    row_y_positions: list[float],
    header_row_estimate: int,
) -> FontValidation:
    """Run all font heuristics (FH1-FH6) on extracted font spans.

    Args:
        font_spans: Font data by row (row_idx -> list of (col, FontSpan))
        row_y_positions: Y positions of text rows
        header_row_estimate: Estimated number of header rows

    Returns:
        FontValidation with results from each heuristic
    """
    if not font_spans:
        return FontValidation(has_font_data=False)

    hierarchy = _detect_font_hierarchy(font_spans, row_y_positions, header_row_estimate)
    bold = _detect_bold_headers(font_spans, header_row_estimate)
    italic = _detect_italic_patterns(font_spans, row_y_positions, header_row_estimate)
    monospace = _detect_monospace_patterns(font_spans, header_row_estimate)
    color = _detect_color_patterns(font_spans, header_row_estimate)
    consistency = _detect_font_consistency(font_spans, header_row_estimate)

    return FontValidation(
        hierarchy=hierarchy,
        bold=bold,
        italic=italic,
        monospace=monospace,
        color=color,
        consistency=consistency,
        has_font_data=True,
    )


# ---------------------------------------------------------------------------
# Region types and data structures
# ---------------------------------------------------------------------------


class RegionType(Enum):
    TABLE = "table"
    TEXT = "text"
    HEADING = "heading"
    KV_PAIRS = "kv_pairs"
    SCATTERED = "scattered"


@dataclass
class Region:
    """A contiguous group of rows classified as a single region type."""

    type: RegionType
    row_indices: list[int]
    rows: dict[int, list[tuple[int, str]]]  # subset of PageLayout.rows


@dataclass
class TableMetadata:
    """Non-data/header info for a table."""

    page_number: int
    table_index: int  # For side-by-side tables (0, 1, ...)
    section_label: str | None  # GERALDTON, etc. if detected above table


@dataclass
class StructuredTable:
    """A table in structured format."""

    metadata: TableMetadata
    column_names: list[str]  # Final flattened names from _build_stacked_headers()
    data: list[list[str]]  # Data rows (each row has len == len(column_names))


# ---------------------------------------------------------------------------
# Relative metrics — avoid brittle absolute thresholds
# ---------------------------------------------------------------------------

def _compute_col_tolerance(cell_w: float) -> int:
    """Compute column tolerance relative to cell width.

    Column tolerance determines when two x-positions are "the same column".
    We use max(3, cell_w * 1.5) to ensure:
    - Minimum of 3 for very small fonts (prevents over-fragmentation)
    - Scales with cell width for larger fonts
    """
    return max(3, round(cell_w * 1.5))


def _compute_median_column_gap(canonical: list[int]) -> float:
    """Compute median gap between adjacent canonical column positions."""
    if len(canonical) < 2:
        return 20.0  # fallback
    gaps = [canonical[i + 1] - canonical[i] for i in range(len(canonical) - 1)]
    gaps.sort()
    mid = len(gaps) // 2
    return float(gaps[mid])


def _is_outlier_gap(gap: float, median_gap: float, threshold_multiplier: float = 3.0) -> bool:
    """Check if a gap is an outlier (significantly larger than typical)."""
    return gap > median_gap * threshold_multiplier


# ---------------------------------------------------------------------------
# Span splitting — fix PDF-level merged cells
# ---------------------------------------------------------------------------

def _split_merged_spans(layout: PageLayout, min_gap: int = 5) -> PageLayout:
    """Split text spans that contain multiple column values merged by the PDF.

    Some PDFs encode adjacent cell values as a single text span (e.g.,
    ``"7:00:00 PM BUNGE"`` where the time and exporter should be separate
    cells). The spatial grid positions are correct, but a single span
    occupies two logical columns.

    Fix: collect column start positions from ALL rows (especially header
    rows that have finer-grained spans). When a span's text extends past
    a column boundary defined by *other* rows, split it at the character
    position corresponding to that boundary.

    ``min_gap`` prevents false splits on nearly-overlapping positions
    (e.g., header col 33 vs data col 35 in the same logical column).
    """
    # Collect column starts and which rows own them.
    col_to_rows: dict[int, set[int]] = {}
    for ri, entries in layout.rows.items():
        for col, _text in entries:
            col_to_rows.setdefault(col, set()).add(ri)

    sorted_positions = sorted(col_to_rows)
    if len(sorted_positions) < 2:
        return layout

    changed = False
    new_rows: dict[int, list[tuple[int, str]]] = {}

    for ri, entries in layout.rows.items():
        new_entries: list[tuple[int, str]] = []
        for col, text in entries:
            span_end = col + len(text)

            # Find positions from OTHER rows that fall inside this span,
            # beyond min_gap from the start.
            lo = bisect_right(sorted_positions, col + min_gap)
            splits: list[int] = []
            for idx in range(lo, len(sorted_positions)):
                pos = sorted_positions[idx]
                if pos >= span_end:
                    break
                # Only split if this position is owned by a different row.
                if ri not in col_to_rows[pos]:
                    # Only split at word boundaries — never break mid-word.
                    # A valid split point must have whitespace at or just
                    # before the character index in the span text.
                    char_idx = pos - col
                    at_word_boundary = (
                        (char_idx > 0 and text[char_idx - 1] == " ")
                        or (char_idx < len(text) and text[char_idx] == " ")
                    )
                    if not at_word_boundary:
                        continue
                    splits.append(pos)

            if not splits:
                new_entries.append((col, text))
                continue

            changed = True
            current_start = 0
            for split_col in splits:
                split_idx = split_col - col
                left = text[current_start:split_idx].rstrip()
                if left:
                    new_entries.append((col + current_start, left))
                # Advance past whitespace after the split point.
                next_start = split_idx
                while next_start < len(text) and text[next_start] == " ":
                    next_start += 1
                current_start = next_start

            remaining = text[current_start:].rstrip()
            if remaining:
                new_entries.append((col + current_start, remaining))

        new_rows[ri] = new_entries

    if not changed:
        return layout

    return PageLayout(
        rows=new_rows, row_count=layout.row_count, cell_w=layout.cell_w
    )


# ---------------------------------------------------------------------------
# Row feature extraction
# ---------------------------------------------------------------------------

def _row_column_positions(row: list[tuple[int, str]]) -> list[int]:
    """Return sorted column start positions for spans in a row."""
    return sorted(col for col, _text in row)


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

def _classify_regions(
    layout: PageLayout,
    min_table_rows: int = 3,
    col_tolerance: int = 3,
) -> list[Region]:
    """Classify contiguous row groups into region types."""
    if not layout.rows:
        return []

    row_indices = sorted(layout.rows.keys())

    # Per-row features.
    col_positions: dict[int, list[int]] = {}
    span_counts: dict[int, int] = {}
    for ri in row_indices:
        entries = layout.rows[ri]
        col_positions[ri] = _row_column_positions(entries)
        span_counts[ri] = len(entries)

    # Build row_texts for single-span rows so _detect_table_runs can
    # distinguish numeric aggregation rows from section-label rows.
    row_texts: dict[int, str] = {}
    for ri in row_indices:
        if span_counts[ri] < 2:
            row_texts[ri] = " ".join(text for _, text in layout.rows[ri])

    # Compute average span length per row to distinguish tables from flowing text.
    # Tables have short spans (data values ~5-8 chars), text has long spans (~12+ chars).
    avg_span_lens: dict[int, float] = {}
    for ri in row_indices:
        spans = layout.rows[ri]
        if spans:
            avg_span_lens[ri] = sum(len(t) for _, t in spans) / len(spans)
        else:
            avg_span_lens[ri] = 0.0

    regions: list[Region] = []
    used: set[int] = set()

    # Detect tables.
    table_runs = _detect_table_runs(
        row_indices, col_positions, span_counts, min_table_rows, col_tolerance,
        row_texts=row_texts,
        avg_span_lens=avg_span_lens,
    )
    for run in table_runs:
        used.update(run)
        regions.append(Region(
            type=RegionType.TABLE,
            row_indices=run,
            rows={ri: layout.rows[ri] for ri in run},
        ))

    # Classify remaining rows.
    remaining = [ri for ri in row_indices if ri not in used]
    i = 0
    while i < len(remaining):
        ri = remaining[i]
        entries = layout.rows[ri]
        num_spans = len(entries)
        total_text = " ".join(text for _, text in entries)

        # KV pairs: exactly 2 spans, left one looks like a label.
        if num_spans == 2:
            kv_run = [ri]
            j = i + 1
            while j < len(remaining):
                nri = remaining[j]
                if nri - kv_run[-1] > 2:
                    break
                nri_count = len(layout.rows[nri])
                if nri_count == 2:
                    kv_run.append(nri)
                    j += 1
                elif nri_count == 1:
                    # Allow single-span continuation rows in kv blocks.
                    kv_run.append(nri)
                    j += 1
                else:
                    break
            if len(kv_run) >= 2:
                regions.append(Region(
                    type=RegionType.KV_PAIRS,
                    row_indices=kv_run,
                    rows={r: layout.rows[r] for r in kv_run},
                ))
                used.update(kv_run)
                i = j
                continue

        # Heading: single span, short, not part of a longer text block.
        if num_spans == 1 and len(total_text) < 80:
            is_heading = True
            if i + 1 < len(remaining):
                nri = remaining[i + 1]
                if nri - ri <= 2 and len(layout.rows[nri]) == 1:
                    next_text = layout.rows[nri][0][1]
                    if len(next_text) > 40:
                        is_heading = False
            if is_heading:
                regions.append(Region(
                    type=RegionType.HEADING,
                    row_indices=[ri],
                    rows={ri: layout.rows[ri]},
                ))
                used.add(ri)
                i += 1
                continue

        # Text block: consecutive single-span rows.
        if num_spans == 1:
            text_run = [ri]
            j = i + 1
            while j < len(remaining):
                nri = remaining[j]
                if nri - text_run[-1] > 2:
                    break
                if len(layout.rows[nri]) == 1:
                    text_run.append(nri)
                    j += 1
                else:
                    break
            regions.append(Region(
                type=RegionType.TEXT,
                row_indices=text_run,
                rows={r: layout.rows[r] for r in text_run},
            ))
            used.update(text_run)
            i = j
            continue

        # Scattered fallback.
        regions.append(Region(
            type=RegionType.SCATTERED,
            row_indices=[ri],
            rows={ri: layout.rows[ri]},
        ))
        used.add(ri)
        i += 1

    regions.sort(key=lambda r: r.row_indices[0])
    return regions


def _is_table_continuation(text: str) -> bool:
    """Return True if a single-span row text looks like it belongs inside a table.

    Matches numeric values (totals, subtotals like ``337,000``), empty rows,
    and parenthesized annotations (unit labels like ``(tonnes)``, ``(MWh)``).
    Returns False for section labels (``KWINANA``, ``GERALDTON``) that should
    flush the current table run.
    """
    stripped = text.strip()
    if not stripped:
        return True
    # Parenthesized text is a unit annotation / header note, not a section label.
    if stripped.startswith("(") and stripped.endswith(")"):
        return True
    for ch in " ,._$€£%+-":
        stripped = stripped.replace(ch, "")
    return stripped.isdigit()


def _looks_like_section_label(text: str) -> bool:
    """Return True if text looks like a section label rather than a header fragment.

    Section labels (GERALDTON, KWINANA, NORTH REGION) are:
    - All uppercase
    - Typically location/category names that separate table sections

    Header fragments (Date of, Quantity:, ETA) are:
    - Mixed case, or
    - Short abbreviations (≤3 chars even if uppercase), or
    - Contain punctuation like colons

    This is used in _find_preceding_header_rows to reject section labels
    from being included as header rows.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Contains colon → likely header fragment ("Quantity:", "Date of:")
    if ":" in stripped:
        return False
    # Short all-caps text (≤3 chars) → likely abbreviation header (ETA, ETB, ID)
    if len(stripped) <= 3:
        return False
    # All uppercase → likely section label (GERALDTON, NORTH REGION)
    if stripped.isupper():
        return True
    return False


def _detect_table_runs(
    row_indices: list[int],
    col_positions: dict[int, list[int]],
    span_counts: dict[int, int],
    min_table_rows: int,
    col_tolerance: int,
    row_texts: dict[int, str] | None = None,
    avg_span_lens: dict[int, float] | None = None,
) -> list[list[int]]:
    """Find maximal runs of rows that form table regions.

    Uses a pool of all column anchors seen so far in the run. A new row
    joins the run if it shares 2+ anchors with the pool. This handles
    alternating row patterns (e.g., data rows with 7 columns interleaved
    with date rows with 5 columns that occupy a subset of positions).

    Flowing text rejection uses RELATIVE thresholds: a row is rejected if
    its avg span length is > 2x the median span length AND its spans don't
    align consistently with the column pool.
    """
    if len(row_indices) < min_table_rows:
        return []

    # Compute median span length for relative threshold.
    all_span_lens = []
    if avg_span_lens:
        for ri in row_indices:
            if ri in avg_span_lens and span_counts.get(ri, 0) >= 2:
                all_span_lens.append(avg_span_lens[ri])
    median_span_len = sorted(all_span_lens)[len(all_span_lens) // 2] if all_span_lens else 8.0

    runs: list[list[int]] = []
    current_run: list[int] = []
    # Pool of all unique column anchors in the current run.
    pool: set[int] = set()

    def _pool_overlap(cols: list[int]) -> tuple[int, float]:
        """Count how many of cols match something in the pool.

        Returns (overlap_count, overlap_ratio) where ratio is what fraction
        of the row's columns are accounted for by the pool. A row with empty
        cells may have fewer columns but still belong to the table if most
        of its columns fit within the established structure.
        """
        count = 0
        for c in cols:
            for p in pool:
                if abs(c - p) <= col_tolerance:
                    count += 1
                    break
        ratio = count / len(cols) if cols else 0.0
        return count, ratio

    def _add_to_pool(cols: list[int]) -> None:
        """Add column positions to the pool, merging near-duplicates."""
        for c in cols:
            for p in pool:
                if abs(c - p) <= col_tolerance:
                    break
            else:
                pool.add(c)

    def _flush_run() -> None:
        if current_run and _count_multi_span(current_run, span_counts) >= min_table_rows:
            runs.append(current_run[:])

    for ri in row_indices:
        sc = span_counts[ri]
        cols = col_positions[ri]

        # Reject rows that look like flowing text (long average span length).
        # Use RELATIVE threshold: reject if avg_len > 2x median span length.
        # This adapts to different documents rather than using fixed cutoffs.
        avg_len = (avg_span_lens or {}).get(ri, 0.0)
        if avg_len > median_span_len * 2.0 and sc >= 2:
            # This looks like flowing text, not table data. Flush any current run.
            _flush_run()
            current_run = []
            pool = set()
            continue

        if sc < 2:
            # Single-span rows extend an existing run but don't start one.
            if current_run and ri - current_run[-1] <= 2:
                text = (row_texts or {}).get(ri, "")
                if _is_table_continuation(text):
                    # Numeric value (likely subtotal/aggregate) — keep in table.
                    current_run.append(ri)
                else:
                    # Non-numeric text (likely section label) — flush the run.
                    _flush_run()
                    current_run = []
                    pool = set()
            continue

        if not current_run:
            current_run = [ri]
            pool = set()
            _add_to_pool(cols)
            continue

        gap = ri - current_run[-1]
        overlap_count, overlap_ratio = _pool_overlap(cols)

        # Row belongs to the table if:
        # 1. Reasonable row gap (adjacent or near-adjacent)
        # 2. Either: at least 2 columns overlap, OR most columns (>= 60%) fit
        #    the established structure (handles rows with empty cells)
        if gap <= 2 and (overlap_count >= 2 or overlap_ratio >= 0.6):
            current_run.append(ri)
            _add_to_pool(cols)
        else:
            _flush_run()
            current_run = [ri]
            pool = set()
            _add_to_pool(cols)

    _flush_run()
    return runs


def _count_multi_span(run: list[int], span_counts: dict[int, int]) -> int:
    """Count rows in a run that have 2+ spans."""
    return sum(1 for ri in run if span_counts[ri] >= 2)


# ---------------------------------------------------------------------------
# Multi-row record detection and merging
# ---------------------------------------------------------------------------

def _detect_multi_row_period(
    table: Region,
    span_counts: dict[int, int],
) -> tuple[int, int] | None:
    """Detect repeating multi-row patterns in a table region.

    Looks at span counts per row to find a repeating period (e.g., 3 rows
    per record: dates/data/times). Tries different header offsets to skip
    non-repeating header rows.

    Returns (header_rows, period) or None. header_rows is the number of
    rows at the start that don't follow the repeating pattern.
    """
    indices = table.row_indices
    counts = [span_counts.get(ri, 0) for ri in indices]

    # Try different header offsets (0..10) and periods (2..4).
    max_header = min(10, len(counts) // 2)
    for period in (3, 2, 4):
        for header in range(max_header + 1):
            body = counts[header:]
            if len(body) < period * 2:
                continue
            pattern = body[:period]
            # Skip if all elements in the pattern are the same --
            # uniform rows don't indicate a multi-row record.
            if len(set(pattern)) <= 1:
                continue
            total_groups = len(body) // period
            matches = 0
            for g in range(total_groups):
                group = body[g * period:(g + 1) * period]
                if group == pattern:
                    matches += 1
            if matches >= total_groups * 0.7 and total_groups >= 2:
                return (header, period)

    return None


def _merge_multi_row_records(
    table: Region,
    header_rows: int,
    period: int,
) -> list[list[tuple[int, str]]]:
    """Merge groups of `period` rows into single logical rows.

    Keeps the first `header_rows` as-is, then merges the remaining rows
    in groups of `period`. For each group, spans from all sub-rows are
    combined. If multiple sub-rows have spans at the same column, values
    are joined with space.
    """
    indices = table.row_indices

    # Keep header rows unchanged.
    result: list[list[tuple[int, str]]] = []
    for ri in indices[:header_rows]:
        if ri in table.rows:
            result.append(sorted(table.rows[ri]))

    # Merge body rows.
    body = indices[header_rows:]
    for g in range(0, len(body), period):
        group = body[g:g + period]
        col_values: dict[int, list[str]] = {}
        for ri in group:
            if ri not in table.rows:
                continue
            for col, text in table.rows[ri]:
                col_values.setdefault(col, []).append(text)

        merged: list[tuple[int, str]] = []
        for col in sorted(col_values):
            merged.append((col, " ".join(col_values[col])))
        result.append(merged)

    return result


def _estimate_header_rows(rows: list[list[tuple[int, str]]]) -> int:
    """Estimate header row count using bottom-up span-count analysis.

    Data rows dominate the bottom portion of any table. Find the most
    common span counts there, then scan from the top to find the first
    row matching a data pattern.

    A row is data-like if:
    - Its span count matches one of the common data span counts, OR
    - Its span count is GREATER than the max common data count (meaning
      it's a complete row with more columns filled, like a completed
      shipment with Loading Status and Date/Time Loading Completed).
    """
    n = len(rows)
    if n <= 2:
        return 0

    span_counts = [len(row) for row in rows]

    # Bottom 2/3 establishes what data rows look like.
    bottom_start = max(1, n // 3)
    bottom_counts = span_counts[bottom_start:]

    freq = Counter(bottom_counts)
    # Top-3 most common span counts. Requirements:
    # - span count >= 2 (single-span rows aren't data)
    # - frequency >= 2 (pattern must repeat to be characteristic of data)
    data_counts = {c for c, count in freq.most_common(3) if c >= 2 and count >= 2}

    if not data_counts:
        return 0

    max_data_count = max(data_counts)

    # Scan from top: first row with a data-like span count.
    # A row with MORE spans than typical is still data (more complete).
    for i in range(n):
        if span_counts[i] in data_counts or span_counts[i] > max_data_count:
            return i

    return 0


def _compute_column_bounds(
    data_rows: list[list[tuple[int, str]]],
    col_map: dict[int, int],
    num_cols: int,
) -> list[tuple[int, int]]:
    """Compute column bounds (min_start, max_end) from data rows.

    For each column, finds the leftmost start position and rightmost end
    position across all data spans assigned to that column. This captures
    the full horizontal extent of each column regardless of alignment
    (left, right, or center).

    Returns a list of (min_start, max_end) tuples, one per column.
    Columns with no data get bounds (0, 0).
    """
    bounds: list[list[int]] = [[float('inf'), 0] for _ in range(num_cols)]  # type: ignore[misc]

    for row in data_rows:
        for col, text in row:
            ci = col_map.get(col)
            if ci is not None:
                start = col
                end = col + len(text)
                bounds[ci][0] = min(bounds[ci][0], start)
                bounds[ci][1] = max(bounds[ci][1], end)

    # Convert to tuples, defaulting empty columns to (0, 0)
    result: list[tuple[int, int]] = []
    for min_s, max_e in bounds:
        if min_s == float('inf'):
            result.append((0, 0))
        else:
            result.append((min_s, max_e))

    return result


def _build_stacked_headers(
    header_rows: list[list[tuple[int, str]]],
    col_bounds: list[tuple[int, int]],
    left_margin: int = 5,
) -> list[str]:
    """Build column names from stacked header rows using bounding-box overlap.

    For each column, finds all header text fragments whose horizontal extent
    overlaps with the column's data bounds (extended by left_margin), then
    concatenates them vertically (top to bottom) with spaces.

    Uses bounding-box overlap instead of start-position matching to handle
    mixed alignment (left-aligned headers with right-aligned data, etc.).

    Args:
        header_rows: List of header rows, each row is list of (col, text) tuples.
        col_bounds: List of (min_start, max_end) tuples, one per column.
        left_margin: Extend column bounds leftward by this amount to capture
            headers that are positioned just before the data starts (common
            when headers are left-aligned but data is right-aligned).

    This is deterministic and more reliable than LLM parsing for complex
    stacked headers.
    """
    if not header_rows or not col_bounds:
        return []

    num_cols = len(col_bounds)
    # For each column, collect aligned header fragments in row order.
    col_fragments: list[list[str]] = [[] for _ in range(num_cols)]

    def _overlaps(h_start: int, h_end: int, d_min: int, d_max: int) -> bool:
        """Check if header span [h_start, h_end) overlaps data bounds [d_min, d_max).

        The data bounds are extended leftward by left_margin to capture headers
        that are positioned just before the column's data starts.
        """
        return h_start < d_max and h_end > (d_min - left_margin)

    for row in header_rows:
        # Track which columns got text in this row.
        row_assignments: dict[int, str] = {}
        for col_pos, text in row:
            h_start = col_pos
            h_end = col_pos + len(text)

            # Find all columns this header span overlaps with.
            # If it overlaps multiple, assign to the one with maximum overlap.
            # Tie-breaker: prefer column whose start is closest to header start.
            best_ci = None
            best_overlap = 0
            best_dist = float("inf")
            for ci, (d_min, d_max) in enumerate(col_bounds):
                if d_min == d_max == 0:
                    continue  # Skip empty columns
                if _overlaps(h_start, h_end, d_min, d_max):
                    # Compute overlap amount using extended bounds (d_min - left_margin)
                    # to be consistent with the overlap check.
                    overlap = min(h_end, d_max) - max(h_start, d_min - left_margin)
                    dist = abs(h_start - d_min)
                    if overlap > best_overlap or (overlap == best_overlap and dist < best_dist):
                        best_ci = ci
                        best_overlap = overlap
                        best_dist = dist

            if best_ci is not None:
                # If multiple spans map to same column, join with space.
                if best_ci in row_assignments:
                    row_assignments[best_ci] += " " + text
                else:
                    row_assignments[best_ci] = text

        # Add this row's text to each column.
        for ci in range(num_cols):
            if ci in row_assignments:
                col_fragments[ci].append(row_assignments[ci])

    # Build final column names by joining fragments.
    column_names = []
    for fragments in col_fragments:
        # Deduplicate consecutive identical words (can happen with overlapping
        # header spans that get assigned to the same column).
        deduped_words: list[str] = []
        for fragment in fragments:
            for word in fragment.split():
                if not deduped_words or word != deduped_words[-1]:
                    deduped_words.append(word)
        name = " ".join(deduped_words)
        column_names.append(name)

    return column_names


# ---------------------------------------------------------------------------
# Column unification (shared by markdown and TSV renderers)
# ---------------------------------------------------------------------------

def _unify_columns(
    rows: list[list[tuple[int, str]]],
    col_tolerance: int = 3,
) -> tuple[list[int], dict[int, int]]:
    """Build a canonical column list and mapping from raw col -> col index.

    Uses greedy clustering: columns are sorted, and each is merged into
    the nearest existing canonical column within tolerance. The canonical
    position is updated to the mean of the cluster to improve matching
    for subsequent columns.

    Returns (canonical_positions, col_map).
    """
    all_cols: set[int] = set()
    for row in rows:
        for col, _text in row:
            all_cols.add(col)

    col_sorted = sorted(all_cols)

    # Cluster columns using greedy merge with running mean.
    clusters: list[list[int]] = []  # each cluster is a list of raw cols
    for c in col_sorted:
        merged = False
        if clusters:
            # Check last cluster (since input is sorted, nearest cluster
            # is always the last one).
            last_mean = sum(clusters[-1]) / len(clusters[-1])
            if abs(c - last_mean) <= col_tolerance:
                clusters[-1].append(c)
                merged = True
        if not merged:
            clusters.append([c])

    # Build mapping.
    canonical: list[int] = []
    col_map: dict[int, int] = {}
    for ci, cluster in enumerate(clusters):
        canonical.append(round(sum(cluster) / len(cluster)))
        for c in cluster:
            col_map[c] = ci

    return canonical, col_map


def _rows_to_grid(
    rows: list[list[tuple[int, str]]],
    num_cols: int,
    col_map: dict[int, int],
) -> list[list[str]]:
    """Convert span-based rows to a grid of cell strings."""
    grid: list[list[str]] = []
    for row in rows:
        cells = [""] * num_cols
        for col, text in row:
            ci = col_map[col]
            if cells[ci]:
                cells[ci] += " " + text
            else:
                cells[ci] = text
        grid.append(cells)
    return grid


def _merge_twin_columns(
    grid: list[list[str]],
    num_cols: int,
) -> tuple[list[list[str]], int]:
    """Merge adjacent columns that are never simultaneously filled.

    Two adjacent columns at different x-positions for the same logical
    field produce a 'twin' pattern: one is filled when the other is empty.
    Merging them recovers the true column count.

    Returns (merged_grid, new_num_cols).
    """
    if num_cols <= 1 or not grid:
        return grid, num_cols

    # Build merge groups: adjacent columns that can combine.
    groups: list[list[int]] = [[0]]
    for j in range(1, num_cols):
        current_group = groups[-1]
        # Can column j merge into the current group?
        conflict = False
        for row in grid:
            if row[j] and any(row[g] for g in current_group):
                conflict = True
                break
        if not conflict:
            current_group.append(j)
        else:
            groups.append([j])

    new_num_cols = len(groups)
    if new_num_cols == num_cols:
        return grid, num_cols  # nothing to merge

    # Build merged grid.
    new_grid: list[list[str]] = []
    for row in grid:
        new_row: list[str] = []
        for group in groups:
            val = ""
            for ci in group:
                if row[ci]:
                    val = row[ci] if not val else val + " " + row[ci]
            new_row.append(val)
        new_grid.append(new_row)

    return new_grid, new_num_cols


# ---------------------------------------------------------------------------
# Transposed table detection
# ---------------------------------------------------------------------------

def _is_transposed_table(
    rows: list[list[tuple[int, str]]],
    col_tolerance: int = 3,
) -> bool:
    """Detect if a table is transposed (field names as rows, records as columns).

    Transposed tables have:
    - Few columns (≤5) — one label column + 1-4 record columns
    - Consistent span counts (low variance < 2.0)
    - Stable first column (present in ≥80% of rows)

    These heuristics work even when the table is fragmented into smaller
    regions by the region detector.

    Returns True if the structural pattern matches transposed layout.
    """
    if len(rows) < 3:
        return False

    # 1. Count unique column positions (using unification).
    canonical, _ = _unify_columns(rows, col_tolerance)
    num_cols = len(canonical)

    if num_cols > 5:
        return False  # Too many columns for transposed

    # 2. Span count variance (consistency check).
    # Transposed tables have very consistent structure (variance < 0.5 typically).
    # Standard tables with varying row structures have higher variance.
    span_counts = [len(row) for row in rows]
    mean_spans = sum(span_counts) / len(span_counts)
    variance = sum((c - mean_spans) ** 2 for c in span_counts) / len(span_counts)
    if variance > 2.0:
        return False  # Too variable, not a clean transposed structure

    # 3. First column stability — transposed tables have field labels in col 0.
    # Standard tables often have missing cells or variable first column positions.
    first_col_count = sum(1 for row in rows if row and row[0][0] <= col_tolerance)
    if first_col_count < len(rows) * 0.8:
        return False  # First column not stable

    return True


# ---------------------------------------------------------------------------
# Preceding header row detection
# ---------------------------------------------------------------------------

def _find_preceding_header_rows(
    layout: PageLayout,
    table_start_row: int,
    data_col_positions: list[int],
    tolerance: int,
    table_row_set: set[int] | None = None,
) -> list[int]:
    """Find rows above a table region that are likely header rows.

    Scans upward from table_start_row. A row is a header candidate if
    at least one of its span positions aligns (within tolerance) with
    a data column position. Stops at the first non-matching row, a
    gap > 2, or a row belonging to another table region.

    Single-span rows are filtered more strictly to avoid including
    document titles or metadata that happen to overlap column positions:
    - The span must START at a position close to a column start (not just
      overlap via margin extension)
    - Wide single spans (> 30 chars) at non-column positions are rejected
    """
    all_rows = sorted(ri for ri in layout.rows if ri < table_start_row)

    result: list[int] = []
    for ri in reversed(all_rows):
        # Don't cross into another table region.
        if table_row_set and ri in table_row_set:
            break

        # Gap check.
        ref = result[-1] if result else table_start_row
        if ref - ri > 2:
            break

        entries = layout.rows[ri]

        # Single-span rows need stricter validation to avoid document titles.
        # Header fragments are typically SHORT (e.g., "Quantity", "Date of", "(tonnes)").
        # Document titles and metadata are longer (e.g., "Shipping Stem Report").
        if len(entries) == 1:
            col, text = entries[0]
            # Check if span START aligns with a column (tighter tolerance).
            start_aligns = any(abs(col - dc) <= tolerance for dc in data_col_positions)
            if not start_aligns:
                break
            # Single-span header fragments should be short (< 15 chars typically).
            # Longer single spans are likely titles/metadata, not column labels.
            if len(text) > 15:
                break
            # Reject section labels (GERALDTON, KWINANA, etc.) that look like
            # they could be headers but are actually table region separators.
            # Note: _looks_like_section_label is different from _is_table_continuation —
            # section labels are all-caps location/category names, while header fragments
            # like "Date of" or "Quantity:" should be kept.
            if _looks_like_section_label(text):
                break
            result.append(ri)
        else:
            # Multi-span rows: at least one span must overlap.
            has_overlap = any(
                any(abs(col - dc) <= tolerance for dc in data_col_positions)
                for col, _ in entries
            )
            if has_overlap:
                result.append(ri)
            else:
                break

    result.reverse()
    return result


# ---------------------------------------------------------------------------
# Horizontal table splitting
# ---------------------------------------------------------------------------

def _find_horizontal_gaps(
    canonical: list[int],
    gap_multiplier: float = 3.0,
) -> list[int]:
    """Find large horizontal gaps in canonical column positions.

    Uses relative gap detection: a gap is significant if it's > gap_multiplier
    times the median inter-column gap. This adapts to different font sizes
    and document formats.

    Returns a list of split indices. A gap between canonical[i] and
    canonical[i+1] results in index i+1 being returned (where the right
    side of the split starts).
    """
    if len(canonical) < 2:
        return []

    median_gap = _compute_median_column_gap(canonical)

    splits: list[int] = []
    for i in range(len(canonical) - 1):
        gap = canonical[i + 1] - canonical[i]
        if _is_outlier_gap(gap, median_gap, gap_multiplier):
            splits.append(i + 1)

    return splits


def _split_grid_horizontally(
    grid: list[list[str]],
    splits: list[int],
) -> list[list[list[str]]]:
    """Split a grid into multiple sub-grids at the given column indices.

    Returns a list of grids, one for each horizontal segment.
    """
    if not splits:
        return [grid]

    all_splits = [0] + splits + [len(grid[0]) if grid else 0]
    result: list[list[list[str]]] = []

    for i in range(len(all_splits) - 1):
        start, end = all_splits[i], all_splits[i + 1]
        sub_grid = [row[start:end] for row in grid]
        # Only include non-empty sub-grids (at least one row with content)
        if any(any(cell.strip() for cell in row) for row in sub_grid):
            result.append(sub_grid)

    return result


# ---------------------------------------------------------------------------
# Region renderers
# ---------------------------------------------------------------------------

def _render_table_markdown(
    rows: list[list[tuple[int, str]]],
    col_tolerance: int = 3,
) -> str:
    """Render table rows as a markdown pipe-delimited table.

    If the table has large horizontal gaps (> 40 positions between columns),
    it's split into separate tables. This handles side-by-side tables that
    share the same y-coordinates but are logically separate (e.g., summary
    tables placed next to each other at the bottom of a page).
    """
    if not rows:
        return ""

    canonical, col_map = _unify_columns(rows, col_tolerance)
    num_cols = len(canonical)
    grid = _rows_to_grid(rows, num_cols, col_map)

    # Check for horizontal gaps that indicate side-by-side tables.
    splits = _find_horizontal_gaps(canonical)
    if splits:
        sub_grids = _split_grid_horizontally(grid, splits)
        if len(sub_grids) > 1:
            # Render each sub-grid as a separate table.
            tables: list[str] = []
            for sub_grid in sub_grids:
                table_lines: list[str] = []
                for i, cells in enumerate(sub_grid):
                    line = "|" + "|".join(cells) + "|"
                    table_lines.append(line)
                    if i == 0:
                        table_lines.append("|" + "|".join("---" for _ in cells) + "|")
                tables.append("\n".join(table_lines))
            return "\n\n".join(tables)

    # Single table (no splits).
    lines: list[str] = []
    for i, cells in enumerate(grid):
        line = "|" + "|".join(cells) + "|"
        lines.append(line)
        if i == 0:
            lines.append("|" + "|".join("---" for _ in cells) + "|")

    return "\n".join(lines)


def _render_table_tsv(
    rows: list[list[tuple[int, str]]],
    col_tolerance: int = 3,
) -> str:
    """Render table rows as TSV."""
    if not rows:
        return ""

    canonical, col_map = _unify_columns(rows, col_tolerance)
    num_cols = len(canonical)
    grid = _rows_to_grid(rows, num_cols, col_map)

    lines: list[str] = []
    for cells in grid:
        lines.append("\t".join(cells))

    return "\n".join(lines)


def _render_text(region: Region) -> str:
    """Render a text region as a flowing paragraph."""
    parts: list[str] = []
    for ri in region.row_indices:
        if ri in region.rows:
            for _col, text in region.rows[ri]:
                parts.append(text)
    return " ".join(parts)


def _render_heading(region: Region) -> str:
    """Render a heading region."""
    parts: list[str] = []
    for ri in region.row_indices:
        if ri in region.rows:
            for _col, text in region.rows[ri]:
                parts.append(text)
    return " ".join(parts)


def _render_kv_pairs(region: Region) -> str:
    """Render key-value pair rows as 'key: value' lines.

    Handles multi-row labels by detecting continuation rows (rows where
    the first span starts at the same column as the label in the previous
    row but has no value span).
    """
    lines: list[str] = []
    pending_key: str | None = None
    pending_value: str | None = None

    for ri in region.row_indices:
        if ri not in region.rows:
            continue
        entries = sorted(region.rows[ri])

        if len(entries) == 2:
            if pending_key is not None:
                lines.append(f"{pending_key}: {pending_value or ''}")
            pending_key = entries[0][1]
            pending_value = entries[1][1]
        elif len(entries) == 1:
            if pending_key is not None:
                pending_key += " " + entries[0][1]
            else:
                lines.append(entries[0][1])
        else:
            if pending_key is not None:
                lines.append(f"{pending_key}: {pending_value or ''}")
                pending_key = None
                pending_value = None
            line = "\t".join(text for _, text in entries)
            lines.append(line)

    if pending_key is not None:
        lines.append(f"{pending_key}: {pending_value or ''}")

    return "\n".join(lines)


def _render_scattered(region: Region) -> str:
    """Render scattered spans as tab-separated lines."""
    lines: list[str] = []
    for ri in region.row_indices:
        if ri in region.rows:
            entries = sorted(region.rows[ri])
            lines.append("\t".join(text for _, text in entries))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-assisted helpers
# ---------------------------------------------------------------------------

def _render_detected_table_markdown(detected) -> str:
    """Render a DetectedTable (from LLM fallback) as a markdown pipe table."""
    if not detected.column_names:
        return ""

    num_cols = len(detected.column_names)
    lines = [
        "|" + "|".join(detected.column_names) + "|",
        "|" + "|".join("---" for _ in detected.column_names) + "|",
    ]

    for row in detected.data_rows:
        # Pad/truncate row to match column count.
        cells = (list(row) + [""] * num_cols)[:num_cols]
        lines.append("|" + "|".join(cells) + "|")

    return "\n".join(lines)


def _render_table_markdown_with_headers(
    grid: list[list[str]],
    column_names: list[str],
) -> str:
    """Render a pre-built data grid as markdown pipe table with given column names."""
    if not grid:
        return ""

    num_cols = len(grid[0])
    header_cells = (list(column_names) + [""] * num_cols)[:num_cols]

    lines = [
        "|" + "|".join(header_cells) + "|",
        "|" + "|".join("---" for _ in header_cells) + "|",
    ]

    for cells in grid:
        lines.append("|" + "|".join(cells) + "|")

    return "\n".join(lines)


def _detect_table_with_llm(spatial_text: str):
    """Call LLM to detect a table on a page. Returns DetectedTable or None."""
    try:
        from baml_client.sync_client import b as b_sync
        result = b_sync.DetectAndStructureTable(spatial_text)
        if result.column_names:
            logger.debug(
                "LLM table detection: layout=%s, structure=%s, columns=%d, rows=%d",
                result.layout.value, result.header_structure.value,
                len(result.column_names), len(result.data_rows),
            )
            return result
        return None
    except Exception:
        logger.debug("LLM table detection failed, falling back to heuristic", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Page compression — structured output
# ---------------------------------------------------------------------------

def _find_section_label_above(
    regions: list[Region],
    table_region_index: int,
) -> str | None:
    """Find a HEADING region before a table region.

    Looks for the closest HEADING region that appears before the given table
    region, skipping over SCATTERED regions that represent table header rows.
    These are typically port/section labels like "GERALDTON", "KWINANA", etc.

    Returns the section label text if found, else None.
    """
    if table_region_index <= 0:
        return None

    # Look backwards through regions to find a HEADING
    # Skip SCATTERED regions (which may be table header fragments)
    for i in range(table_region_index - 1, -1, -1):
        region = regions[i]

        if region.type == RegionType.HEADING:
            # Extract the heading text
            text_parts: list[str] = []
            for ri in region.row_indices:
                if ri in region.rows:
                    for _, text in region.rows[ri]:
                        text_parts.append(text)

            heading_text = " ".join(text_parts).strip()

            # Section labels are short and not table continuation text
            if heading_text and len(heading_text) < 50 and not _is_table_continuation(heading_text):
                return heading_text
            return None

        elif region.type == RegionType.TABLE:
            # Hit another table - no section label for this table
            return None

        # Skip SCATTERED regions (likely table header fragments)
        # Continue looking for a HEADING

    return None


def _compress_page_structured(
    page: fitz.Page,
    page_number: int,
    cluster_threshold: float = 2.0,
    merge_multi_row: bool = True,
    min_table_rows: int = 3,
    refine_headers: bool = True,
    extract_visual: bool = True,
) -> list[StructuredTable]:
    """Extract structured tables from a single PDF page.

    Returns a list of StructuredTable objects, one per table found.
    Side-by-side tables are serialized as separate entries.

    Args:
        page: PyMuPDF page object.
        page_number: 0-based page index.
        cluster_threshold: Maximum y-distance for row merging.
        merge_multi_row: If True, detect and merge multi-row records.
        min_table_rows: Minimum rows to classify as table.
        refine_headers: If True, use LLM fallback for missed tables.
        extract_visual: If True, extract visual elements for cross-validation.
    """
    layout = _extract_page_layout(page, cluster_threshold, extract_visual=extract_visual)
    if layout is None:
        return []

    # Split spans that the PDF merged across column boundaries.
    layout = _split_merged_spans(layout)

    # Use relative column tolerance based on cell width.
    col_tolerance = _compute_col_tolerance(layout.cell_w)
    regions = _classify_regions(layout, min_table_rows, col_tolerance)

    if not regions:
        return []

    has_tables = any(r.type == RegionType.TABLE for r in regions)

    # === VISUAL ANALYSIS ===
    # Run visual heuristics for cross-validation (parallel channel)
    visual_validation: VisualValidation | None = None
    if layout.visual is not None:
        # Build row Y positions for visual analysis
        row_y_positions = layout.row_y_positions

        visual_validation = _analyze_visual_structure(
            layout.visual,
            row_y_positions,
            page.rect.width,
            page.rect.height,
        )

        if visual_validation.has_visual_elements:
            logger.debug(
                "Visual elements detected (page %d): grid=%s, header=%s, zebra=%s",
                page_number,
                visual_validation.grid is not None,
                visual_validation.header is not None,
                visual_validation.zebra is not None,
            )

            # Cross-validate: visual detects table but text didn't?
            if visual_validation.grid and visual_validation.grid.has_grid and not has_tables:
                logger.warning(
                    "Visual grid detected (VH1) but no TABLE region found — "
                    "possible text heuristic gap (page %d)",
                    page_number,
                )

    # === FONT ANALYSIS ===
    # Run font heuristics for cross-validation (parallel channel)
    font_validation: FontValidation | None = None
    if layout.font_spans:
        # Estimate header rows from visual cues or default to 2
        header_row_estimate = 2
        if visual_validation and visual_validation.header:
            header_row_estimate = max(visual_validation.header.header_fill_rows) + 1

        font_validation = _analyze_font_structure(
            layout.font_spans,
            layout.row_y_positions,
            header_row_estimate,
        )

        if font_validation.has_font_data:
            logger.debug(
                "Font heuristics detected (page %d): hierarchy=%s, bold=%s, italic=%s, "
                "monospace=%s, color=%s, consistency=%s",
                page_number,
                font_validation.hierarchy is not None,
                font_validation.bold is not None,
                font_validation.italic is not None,
                font_validation.monospace is not None,
                font_validation.color is not None,
                font_validation.consistency is not None,
            )

    # === FALLBACK: no tables detected, ask the LLM ===
    if not has_tables and refine_headers:
        from pdf_ocr.spatial_text import _render_page_grid
        spatial = _render_page_grid(page, cluster_threshold)
        detected = _detect_table_with_llm(spatial)
        if detected and detected.column_names:
            return [StructuredTable(
                metadata=TableMetadata(
                    page_number=page_number,
                    table_index=0,
                    section_label=None,
                ),
                column_names=list(detected.column_names),
                data=[list(row) for row in detected.data_rows],
            )]
        return []

    # === NORMAL PATH: extract tables ===
    structured_tables: list[StructuredTable] = []
    table_counter = 0  # Track table index within page

    # Collect all table row indices for boundary detection.
    all_table_rows: set[int] = set()
    for region in regions:
        if region.type == RegionType.TABLE:
            all_table_rows.update(region.row_indices)

    # Pre-compute preceding header rows for each table.
    table_preceding_headers: dict[int, list[int]] = {}
    preceding_header_rows_set: set[int] = set()

    render_tol = max(col_tolerance, round(10.0 / layout.cell_w))
    for region in regions:
        if region.type != RegionType.TABLE:
            continue
        trows = [sorted(region.rows[ri]) for ri in region.row_indices]
        # Skip transposed tables — they don't have header rows above.
        if _is_transposed_table(trows, render_tol):
            continue
        hc = None
        if merge_multi_row:
            sc = {ri: len(region.rows[ri]) for ri in region.row_indices}
            mr = _detect_multi_row_period(region, sc)
            if mr is not None:
                hc = mr[0]
        if hc is None:
            hc = _estimate_header_rows(trows)
        data_rows = trows[hc:]
        if data_rows:
            can, _ = _unify_columns(data_rows, render_tol)
            preceding = _find_preceding_header_rows(
                layout, region.row_indices[0], can,
                render_tol, all_table_rows,
            )
            table_preceding_headers[region.row_indices[0]] = preceding
            preceding_header_rows_set.update(preceding)

    for region_idx, region in enumerate(regions):
        if region.type != RegionType.TABLE:
            continue

        table_rows = [sorted(region.rows[ri]) for ri in region.row_indices]
        render_tolerance = max(col_tolerance, round(10.0 / layout.cell_w))

        # Detect transposed table structure.
        is_transposed = _is_transposed_table(table_rows, render_tolerance)

        if is_transposed:
            logger.debug("Transposed table detected — using simple extraction")
            # For transposed tables, use simple column unification
            canonical, col_map = _unify_columns(table_rows, render_tolerance)
            num_cols = len(canonical)
            grid = _rows_to_grid(table_rows, num_cols, col_map)

            # First column is typically the label column
            column_names = [""] * num_cols
            if grid and grid[0]:
                column_names[0] = "Field"
                for i in range(1, num_cols):
                    column_names[i] = f"Value {i}" if num_cols > 2 else "Value"

            section_label = _find_section_label_above(regions, region_idx)
            structured_tables.append(StructuredTable(
                metadata=TableMetadata(
                    page_number=page_number,
                    table_index=table_counter,
                    section_label=section_label,
                ),
                column_names=column_names,
                data=grid,
            ))
            table_counter += 1
            continue

        if refine_headers:
            # 1. Detect multi-row pattern (also finds header boundary).
            mr_header_count = None
            period = 1
            if merge_multi_row:
                span_counts = {ri: len(region.rows[ri]) for ri in region.row_indices}
                mr_result = _detect_multi_row_period(region, span_counts)
                if mr_result is not None:
                    mr_header_count, period = mr_result

            # 2. Determine header/data boundary.
            header_count = (
                mr_header_count if mr_header_count is not None
                else _estimate_header_rows(table_rows)
            )

            # 3. Multi-row merge if applicable.
            if period > 1:
                table_rows = _merge_multi_row_records(
                    region, header_count, period
                )

            # 4. Build columns from DATA rows only.
            data_rows = table_rows[header_count:]
            canonical, col_map = _unify_columns(data_rows, render_tolerance)

            # Extend with rightmost columns from header rows.
            if header_count > 0 and canonical:
                rightmost_data = canonical[-1]
                header_rows_spans = table_rows[:header_count]
                for row in header_rows_spans:
                    for col, _ in row:
                        if col > rightmost_data + render_tolerance:
                            already_mapped = any(
                                abs(col - c) <= render_tolerance for c in canonical
                            )
                            if not already_mapped:
                                canonical.append(col)
                canonical.sort()

            # Rebuild col_map for all table rows.
            col_map = {}
            for ci, c in enumerate(canonical):
                col_map[c] = ci
            for row in table_rows:
                for col, _ in row:
                    if col not in col_map:
                        for ci, c in enumerate(canonical):
                            if abs(col - c) <= render_tolerance:
                                col_map[col] = ci
                                break
                        else:
                            nearest_ci = min(
                                range(len(canonical)),
                                key=lambda i: abs(canonical[i] - col),
                            )
                            col_map[col] = nearest_ci

            num_cols = len(canonical)

            # Check for horizontal gaps indicating side-by-side tables.
            horizontal_gaps = _find_horizontal_gaps(canonical)
            if horizontal_gaps:
                logger.debug(
                    "Side-by-side tables detected (gaps at cols %s) — splitting",
                    [canonical[g] for g in horizontal_gaps],
                )
                # Split into separate tables
                all_grid = _rows_to_grid(table_rows, num_cols, col_map)
                sub_grids = _split_grid_horizontally(all_grid, horizontal_gaps)

                # Split canonical columns for each sub-table
                all_splits = [0] + horizontal_gaps + [num_cols]
                sub_canonical_lists = [
                    canonical[all_splits[i]:all_splits[i + 1]]
                    for i in range(len(all_splits) - 1)
                ]

                # Get preceding headers for this table
                preceding = table_preceding_headers.get(region.row_indices[0], [])

                # Find section label above this table
                section_label = _find_section_label_above(regions, region_idx)

                for sub_idx, (sub_grid, sub_canonical) in enumerate(
                    zip(sub_grids, sub_canonical_lists)
                ):
                    if not sub_grid or not any(any(cell for cell in row) for row in sub_grid):
                        continue

                    # Build col_map for this sub-table's columns
                    sub_num_cols = len(sub_canonical)
                    sub_col_map = {c: i for i, c in enumerate(sub_canonical)}

                    # Build column bounds for sub-table
                    # Need to remap data_rows to sub-table's column range
                    sub_data_rows = []
                    start_col_idx = all_splits[sub_idx]
                    end_col_idx = all_splits[sub_idx + 1]
                    for row in data_rows:
                        sub_row = []
                        for col, text in row:
                            ci = col_map.get(col)
                            if ci is not None and start_col_idx <= ci < end_col_idx:
                                # Adjust column position to sub-table coordinates
                                sub_row.append((col, text))
                        if sub_row:
                            sub_data_rows.append(sub_row)

                    # Rebuild sub_col_map from actual positions in sub_data_rows
                    sub_all_cols: set[int] = set()
                    for row in sub_data_rows:
                        for col, _ in row:
                            sub_all_cols.add(col)
                    sub_canonical_actual = sorted(sub_all_cols)
                    sub_col_map = {c: i for i, c in enumerate(sub_canonical_actual)}
                    sub_num_cols = len(sub_canonical_actual)

                    sub_col_bounds = _compute_column_bounds(
                        sub_data_rows, sub_col_map, sub_num_cols
                    )

                    # Build header rows for sub-table (filter spans that overlap)
                    all_header_rows = []
                    for ri in preceding:
                        all_header_rows.append(sorted(layout.rows[ri]))
                    all_header_rows.extend(table_rows[:header_count])

                    sub_column_names = _build_stacked_headers(
                        all_header_rows, sub_col_bounds
                    )

                    # Get data rows (skip header rows)
                    sub_data_grid = sub_grid[header_count:]

                    # Ensure column_names matches grid width
                    grid_width = len(sub_data_grid[0]) if sub_data_grid else 0
                    if len(sub_column_names) < grid_width:
                        sub_column_names.extend([""] * (grid_width - len(sub_column_names)))
                    elif len(sub_column_names) > grid_width:
                        sub_column_names = sub_column_names[:grid_width]

                    structured_tables.append(StructuredTable(
                        metadata=TableMetadata(
                            page_number=page_number,
                            table_index=table_counter,
                            section_label=section_label if sub_idx == 0 else None,
                        ),
                        column_names=sub_column_names,
                        data=sub_data_grid,
                    ))
                    table_counter += 1
            else:
                # Single table (no horizontal splits)
                preceding = table_preceding_headers.get(region.row_indices[0], [])

                if preceding:
                    logger.debug(
                        "Found %d preceding header rows: %s",
                        len(preceding), preceding,
                    )

                # Find section label above this table
                section_label = _find_section_label_above(regions, region_idx)

                # Build stacked headers
                all_header_rows = []
                for ri in preceding:
                    all_header_rows.append(sorted(layout.rows[ri]))
                all_header_rows.extend(table_rows[:header_count])

                col_bounds = _compute_column_bounds(data_rows, col_map, num_cols)
                column_names = _build_stacked_headers(all_header_rows, col_bounds)

                # Grid and (optionally) merge twin columns
                all_grid = _rows_to_grid(table_rows, num_cols, col_map)
                all_grid, merged_num_cols = _merge_twin_columns(all_grid, num_cols)

                if merged_num_cols < num_cols:
                    # Rebuild without twin merging for now
                    all_grid = _rows_to_grid(table_rows, num_cols, col_map)

                # Skip header rows
                grid = all_grid[header_count:]

                # Ensure column_names matches grid width
                grid_width = len(grid[0]) if grid else 0
                if len(column_names) < grid_width:
                    column_names.extend([""] * (grid_width - len(column_names)))
                elif len(column_names) > grid_width:
                    column_names = column_names[:grid_width]

                structured_tables.append(StructuredTable(
                    metadata=TableMetadata(
                        page_number=page_number,
                        table_index=table_counter,
                        section_label=section_label,
                    ),
                    column_names=column_names,
                    data=grid,
                ))
                table_counter += 1
        else:
            # Non-refined path: simple extraction
            if merge_multi_row:
                span_counts = {ri: len(region.rows[ri]) for ri in region.row_indices}
                result = _detect_multi_row_period(region, span_counts)
                if result is not None:
                    header_rows_count, period = result
                    if period > 1:
                        table_rows = _merge_multi_row_records(
                            region, header_rows_count, period
                        )

            canonical, col_map = _unify_columns(table_rows, render_tolerance)
            num_cols = len(canonical)
            grid = _rows_to_grid(table_rows, num_cols, col_map)

            # First row as headers
            column_names = grid[0] if grid else []
            data = grid[1:] if len(grid) > 1 else []

            structured_tables.append(StructuredTable(
                metadata=TableMetadata(
                    page_number=page_number,
                    table_index=table_counter,
                    section_label=None,
                ),
                column_names=column_names,
                data=data,
            ))
            table_counter += 1

    return structured_tables


def _render_structured_table_markdown(table: StructuredTable) -> str:
    """Render a StructuredTable as a markdown pipe table."""
    return _render_table_markdown_with_headers(table.data, table.column_names)


def _render_tables_to_markdown(
    tables: list[StructuredTable],
    non_table_parts: list[str],
) -> str:
    """Render structured tables and non-table parts to markdown string.

    Interleaves non-table content with table markdown.
    """
    # For now, just render tables in order with non-table parts first
    parts: list[str] = list(non_table_parts)
    for table in tables:
        md = _render_structured_table_markdown(table)
        if md:
            parts.append(md)
    return "\n\n".join(p for p in parts if p)


def _compress_page(
    page: fitz.Page,
    cluster_threshold: float = 2.0,
    table_format: str = "markdown",
    merge_multi_row: bool = True,
    min_table_rows: int = 3,
    refine_headers: bool = True,
    extract_visual: bool = True,
) -> str:
    """Compress a single PDF page into structured text.

    Args:
        page: PyMuPDF page object.
        cluster_threshold: Maximum y-distance for row merging.
        table_format: "markdown" or "tsv".
        merge_multi_row: If True, detect and merge multi-row records.
        min_table_rows: Minimum rows to classify as table.
        refine_headers: If True, use LLM fallback for missed tables.
        extract_visual: If True, extract visual elements for cross-validation.
    """
    layout = _extract_page_layout(page, cluster_threshold, extract_visual=extract_visual)
    if layout is None:
        return ""

    # Split spans that the PDF merged across column boundaries.
    layout = _split_merged_spans(layout)

    # Use relative column tolerance based on cell width.
    col_tolerance = _compute_col_tolerance(layout.cell_w)
    regions = _classify_regions(layout, min_table_rows, col_tolerance)

    if not regions:
        return ""

    has_tables = any(r.type == RegionType.TABLE for r in regions)

    # === VISUAL ANALYSIS ===
    # Run visual heuristics for cross-validation (parallel channel)
    visual_validation: VisualValidation | None = None
    if layout.visual is not None:
        # Build row Y positions for visual analysis
        # We need the actual Y coordinates, but layout only has row indices
        # For now, use row indices as proxy (will be refined if needed)
        row_y_positions = layout.row_y_positions

        visual_validation = _analyze_visual_structure(
            layout.visual,
            row_y_positions,
            page.rect.width,
            page.rect.height,
        )

        if visual_validation.has_visual_elements:
            logger.debug(
                "Visual elements detected: grid=%s, header=%s, zebra=%s, separators=%s",
                visual_validation.grid is not None,
                visual_validation.header is not None,
                visual_validation.zebra is not None,
                visual_validation.separators is not None,
            )

            # Cross-validate: visual detects table but text didn't?
            if visual_validation.grid and visual_validation.grid.has_grid and not has_tables:
                logger.warning(
                    "Visual grid detected (VH1) but no TABLE region found by text heuristics — "
                    "possible text heuristic gap"
                )

    # === FONT ANALYSIS ===
    # Run font heuristics for cross-validation (parallel channel)
    font_validation: FontValidation | None = None
    if layout.font_spans:
        # Estimate header rows from visual cues or default to 2
        header_row_estimate = 2
        if visual_validation and visual_validation.header:
            header_row_estimate = max(visual_validation.header.header_fill_rows) + 1

        font_validation = _analyze_font_structure(
            layout.font_spans,
            layout.row_y_positions,
            header_row_estimate,
        )

        if font_validation.has_font_data:
            logger.debug(
                "Font heuristics detected: hierarchy=%s, bold=%s, italic=%s, "
                "monospace=%s, color=%s, consistency=%s",
                font_validation.hierarchy is not None,
                font_validation.bold is not None,
                font_validation.italic is not None,
                font_validation.monospace is not None,
                font_validation.color is not None,
                font_validation.consistency is not None,
            )

    # === FALLBACK: no tables detected, ask the LLM ===
    if not has_tables and refine_headers:
        from pdf_ocr.spatial_text import _render_page_grid
        spatial = _render_page_grid(page, cluster_threshold)
        detected = _detect_table_with_llm(spatial)
        if detected and detected.column_names:
            table_md = _render_detected_table_markdown(detected)
            # Render non-table regions normally, then append LLM table.
            rendered_parts: list[str] = []
            for region in regions:
                if region.type == RegionType.TEXT:
                    rendered_parts.append(_render_text(region))
                elif region.type == RegionType.HEADING:
                    rendered_parts.append(_render_heading(region))
                elif region.type == RegionType.KV_PAIRS:
                    rendered_parts.append(_render_kv_pairs(region))
                elif region.type == RegionType.SCATTERED:
                    rendered_parts.append(_render_scattered(region))
            rendered_parts.append(table_md)
            return "\n\n".join(rendered_parts)

    # === NORMAL PATH: render each region ===
    rendered_parts: list[str] = []

    # Collect all table row indices for boundary detection when scanning
    # for preceding header rows.
    all_table_rows: set[int] = set()
    for region in regions:
        if region.type == RegionType.TABLE:
            all_table_rows.update(region.row_indices)

    # Pre-compute preceding header rows for all table regions so they
    # can be suppressed from non-table region rendering.
    preceding_header_rows: set[int] = set()
    if refine_headers and table_format == "markdown":
        render_tol = max(col_tolerance, round(10.0 / layout.cell_w))
        for region in regions:
            if region.type != RegionType.TABLE:
                continue
            trows = [sorted(region.rows[ri]) for ri in region.row_indices]
            # Skip transposed tables — they don't have header rows above.
            if _is_transposed_table(trows, render_tol):
                continue
            hc = None
            if merge_multi_row:
                sc = {ri: len(region.rows[ri]) for ri in region.row_indices}
                mr = _detect_multi_row_period(region, sc)
                if mr is not None:
                    hc = mr[0]
            if hc is None:
                hc = _estimate_header_rows(trows)
            data_rows = trows[hc:]
            if data_rows:
                can, _ = _unify_columns(data_rows, render_tol)
                preceding_header_rows.update(_find_preceding_header_rows(
                    layout, region.row_indices[0], can,
                    render_tol, all_table_rows,
                ))

    for region in regions:
        if region.type == RegionType.TABLE:
            table_rows = [sorted(region.rows[ri]) for ri in region.row_indices]
            # Use wider tolerance for column unification in rendering.
            render_tolerance = max(col_tolerance, round(10.0 / layout.cell_w))

            # Detect transposed table structure.
            is_transposed = _is_transposed_table(table_rows, render_tolerance)
            if is_transposed:
                logger.debug("Transposed table detected — skipping LLM header refinement")

            if refine_headers and table_format == "markdown" and not is_transposed:
                # 1. Detect multi-row pattern (also finds header boundary).
                mr_header_count = None
                period = 1
                if merge_multi_row:
                    span_counts = {ri: len(region.rows[ri]) for ri in region.row_indices}
                    mr_result = _detect_multi_row_period(region, span_counts)
                    if mr_result is not None:
                        mr_header_count, period = mr_result

                # 2. Determine header/data boundary.
                header_count = (
                    mr_header_count if mr_header_count is not None
                    else _estimate_header_rows(table_rows)
                )

                # 3. Multi-row merge if applicable.
                if period > 1:
                    table_rows = _merge_multi_row_records(
                        region, header_count, period
                    )

                # 4. Build columns from DATA rows only (avoids phantom columns from
                # header spans at different x-positions). But also extend with any
                # rightmost columns from header rows — these may be real columns
                # that are empty in most data rows (e.g., "Loading Completed" columns
                # only filled for completed vessels).
                data_rows = table_rows[header_count:]
                canonical, col_map = _unify_columns(data_rows, render_tolerance)

                # Extend with rightmost columns from header rows if they're beyond
                # the rightmost data column.
                if header_count > 0 and canonical:
                    rightmost_data = canonical[-1]
                    header_rows = table_rows[:header_count]
                    for row in header_rows:
                        for col, _ in row:
                            if col > rightmost_data + render_tolerance:
                                # Check if this position is already covered.
                                already_mapped = any(
                                    abs(col - c) <= render_tolerance for c in canonical
                                )
                                if not already_mapped:
                                    canonical.append(col)
                    canonical.sort()

                # Rebuild col_map to cover ALL table rows (including headers),
                # mapping each position to its nearest canonical column.
                col_map = {}
                for ci, c in enumerate(canonical):
                    col_map[c] = ci
                for row in table_rows:
                    for col, _ in row:
                        if col not in col_map:
                            for ci, c in enumerate(canonical):
                                if abs(col - c) <= render_tolerance:
                                    col_map[col] = ci
                                    break
                            else:
                                # Position not near any canonical column — map to
                                # nearest one (may happen for header spans).
                                nearest_ci = min(
                                    range(len(canonical)),
                                    key=lambda i: abs(canonical[i] - col),
                                )
                                col_map[col] = nearest_ci

                num_cols = len(canonical)

                # Check for horizontal gaps indicating side-by-side tables.
                # These need splitting which isn't compatible with LLM header refinement.
                horizontal_gaps = _find_horizontal_gaps(canonical)
                if horizontal_gaps:
                    logger.debug(
                        "Side-by-side tables detected (gaps at cols %s) — using split rendering",
                        [canonical[g] for g in horizontal_gaps],
                    )
                    rendered_parts.append(_render_table_markdown(table_rows, render_tolerance))
                else:
                    # 5. Find header rows above this table region BEFORE any grid manipulation.
                    preceding = _find_preceding_header_rows(
                        layout, region.row_indices[0], canonical,
                        render_tolerance, all_table_rows,
                    )

                    if preceding:
                        logger.debug(
                            "Found %d preceding header rows: %s",
                            len(preceding), preceding,
                        )

                    # 6. Build stacked headers BEFORE twin merging (uses original column structure).
                    # Collect all header rows: preceding + table-internal headers.
                    all_header_rows = []
                    for ri in preceding:
                        all_header_rows.append(sorted(layout.rows[ri]))
                    all_header_rows.extend(table_rows[:header_count])

                    # Compute column bounds from data rows for bounding-box matching.
                    col_bounds = _compute_column_bounds(data_rows, col_map, num_cols)

                    # Build column names using overlap-based matching.
                    column_names = _build_stacked_headers(all_header_rows, col_bounds)

                    # 7. Grid ALL table rows and merge twin columns.
                    all_grid = _rows_to_grid(table_rows, num_cols, col_map)
                    all_grid, merged_num_cols = _merge_twin_columns(all_grid, num_cols)

                    # If twin merging reduced column count, we need to merge corresponding headers.
                    if merged_num_cols < num_cols:
                        # Rebuild the grid without twin merging for now — twin merging
                        # creates a mismatch between header count and grid columns.
                        # TODO: properly merge headers when columns are merged.
                        all_grid = _rows_to_grid(table_rows, num_cols, col_map)
                        merged_num_cols = num_cols

                    # 8. Skip header rows from the grid.
                    grid = all_grid[header_count:]

                    # 9. Render with deterministic column names and data grid.
                    rendered_parts.append(
                        _render_table_markdown_with_headers(grid, column_names)
                    )
            else:
                # Original path (no LLM or TSV format).
                if merge_multi_row:
                    span_counts = {ri: len(region.rows[ri]) for ri in region.row_indices}
                    result = _detect_multi_row_period(region, span_counts)
                    if result is not None:
                        header_rows, period = result
                        if period > 1:
                            table_rows = _merge_multi_row_records(
                                region, header_rows, period
                            )

                if table_format == "tsv":
                    rendered_parts.append(_render_table_tsv(table_rows, render_tolerance))
                else:
                    rendered_parts.append(
                        _render_table_markdown(table_rows, render_tolerance)
                    )

        else:
            # Filter out rows consumed as preceding table headers.
            # Keep HEADING regions intact — they are section labels
            # (e.g. port names), not column header fragments.
            if preceding_header_rows and region.type != RegionType.HEADING:
                filtered = [ri for ri in region.row_indices
                            if ri not in preceding_header_rows]
                if not filtered:
                    continue
                region = Region(
                    type=region.type,
                    row_indices=filtered,
                    rows={ri: region.rows[ri] for ri in filtered},
                )

            if region.type == RegionType.TEXT:
                rendered_parts.append(_render_text(region))
            elif region.type == RegionType.HEADING:
                rendered_parts.append(_render_heading(region))
            elif region.type == RegionType.KV_PAIRS:
                rendered_parts.append(_render_kv_pairs(region))
            elif region.type == RegionType.SCATTERED:
                rendered_parts.append(_render_scattered(region))

    return "\n\n".join(rendered_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_spatial_text(
    pdf_input: str | bytes | Path,
    *,
    refine_headers: bool = True,
    pages: list[int] | None = None,
    cluster_threshold: float = 2.0,
    page_separator: str = "\f",
    table_format: str = "markdown",
    merge_multi_row: bool = True,
    min_table_rows: int = 3,
    extract_visual: bool = True,
) -> str:
    """Convert a PDF to compressed structured text for LLM consumption.

    Produces token-efficient output using markdown tables, flowing text,
    and key-value pairs instead of whitespace-heavy spatial grids.

    Args:
        pdf_input: Path to PDF file (str or Path), or raw PDF bytes.
        refine_headers: If True, use an LLM (GPT-4o-mini) to refine table
            headers and detect tables missed by heuristics. Default True.
        pages: Optional list of 0-based page indices to render.
            If None, all pages are rendered.
        cluster_threshold: Maximum y-distance (in points) to merge into
            the same text row. Default 2.0.
        page_separator: String inserted between pages. Default is form-feed.
        table_format: "markdown" for pipe-delimited tables or "tsv" for
            tab-separated values. Default "markdown".
        merge_multi_row: If True, detect and merge multi-row records in
            tables (e.g., 3-row shipping stem entries). Default True.
        min_table_rows: Minimum number of multi-span rows to classify a
            region as a table. Default 3.
        extract_visual: If True, extract visual elements (lines, fills) for
            cross-validation with text heuristics. Default True.

    Returns:
        A single string with all compressed pages joined by *page_separator*.
    """
    doc = _open_pdf(pdf_input)
    page_indices = pages if pages is not None else range(len(doc))
    rendered: list[str] = []
    for idx in page_indices:
        rendered.append(_compress_page(
            doc[idx],
            cluster_threshold=cluster_threshold,
            table_format=table_format,
            merge_multi_row=merge_multi_row,
            min_table_rows=min_table_rows,
            refine_headers=refine_headers,
            extract_visual=extract_visual,
        ))
    doc.close()
    return page_separator.join(rendered)


def compress_spatial_text_structured(
    pdf_input: str | bytes | Path,
    *,
    pages: list[int] | None = None,
    cluster_threshold: float = 2.0,
    merge_multi_row: bool = True,
    min_table_rows: int = 3,
    extract_visual: bool = True,
) -> list[StructuredTable]:
    """Extract structured tables from a PDF.

    Returns a list of StructuredTable objects containing:
    - metadata: page_number, table_index (for side-by-side tables), section_label
    - column_names: list of column header strings
    - data: list of rows, each row is a list of cell strings

    Side-by-side tables on the same page are serialized as separate entries
    with incrementing table_index values.

    Args:
        pdf_input: Path to PDF file (str or Path), or raw PDF bytes.
        pages: Optional list of 0-based page indices to process.
            If None, all pages are processed.
        cluster_threshold: Maximum y-distance (in points) to merge into
            the same text row. Default 2.0.
        merge_multi_row: If True, detect and merge multi-row records in
            tables (e.g., 3-row shipping stem entries). Default True.
        min_table_rows: Minimum number of multi-span rows to classify a
            region as a table. Default 3.
        extract_visual: If True, extract visual elements for cross-validation.
            Default True.

    Returns:
        A list of StructuredTable objects, one per table found across
        all processed pages.
    """
    doc = _open_pdf(pdf_input)
    page_indices = pages if pages is not None else range(len(doc))
    all_tables: list[StructuredTable] = []

    for idx in page_indices:
        tables = _compress_page_structured(
            doc[idx],
            page_number=idx,
            cluster_threshold=cluster_threshold,
            merge_multi_row=merge_multi_row,
            min_table_rows=min_table_rows,
            refine_headers=True,
            extract_visual=extract_visual,
        )
        all_tables.extend(tables)

    doc.close()
    return all_tables
