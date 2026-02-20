"""Shared semantic heuristics for table structure detection.

These heuristics encode universal human inference patterns applicable across
all document formats (PDF, XLSX, DOCX, PPTX, HTML). They work on grid-level
data (list of rows, each row being list of cells) rather than format-specific
geometry.

Heuristics included:
- TH1: Cell type detection (date, number, string, enum)
- TH2: Header type pattern (all-string rows are likely headers)
- TH3: Column type consistency
- H7: Header row estimation via bottom-up span-count analysis
- RH1: Temporal pattern detection (dates, periods, fiscal years)
- RH4: Unit/currency pattern detection (scale, currency codes)

Geometry heuristics (y-clustering, x-position alignment) remain in format-specific
extractors since they depend on spatial coordinates.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Metadata retrieval types (for search & extract feature)
# ---------------------------------------------------------------------------


class MetadataCategory(Enum):
    """Categories of document metadata for retrieval heuristics.

    Each category has associated patterns and typical search zones.
    """

    TEMPORAL = "temporal"  # RH1: Dates, periods, fiscal years
    ENTITY = "entity"  # RH2: Company names, identifiers
    TABLE_IDENTITY = "table_identity"  # RH3: Table titles, section names
    TABLE_CONTEXT = "table_context"  # RH4: Units, currency, scale
    FOOTNOTE = "footnote"  # RH5: Footnotes, disclaimers


class SearchZone(Enum):
    """Zones within a document where metadata is commonly found.

    Used to focus pattern matching on high-probability regions.
    """

    TITLE_PAGE = "title_page"  # First page header area (top 40%)
    PAGE_HEADER = "page_header"  # Top 15% of any page
    PAGE_FOOTER = "page_footer"  # Bottom 15% of any page
    TABLE_CAPTION = "table_caption"  # Text immediately above table
    COLUMN_HEADER = "column_header"  # Within table headers
    TABLE_FOOTER = "table_footer"  # Below table
    ANYWHERE = "anywhere"  # Full document scan


class FallbackStrategy(Enum):
    """Strategies for handling missing required metadata.

    When a required metadata field cannot be found via pattern matching,
    these strategies determine the fallback behavior.
    """

    INFER = "infer"  # Pattern-based guess from related context
    DEFAULT = "default"  # Use schema-defined default value
    PROMPT = "prompt"  # Ask user for the value
    FLAG = "flag"  # Return with _missing marker, let caller handle


@dataclass
class MetadataFieldDef:
    """Definition of a metadata field to extract from a document.

    Attributes:
        name: Field name in the result dict
        category: Metadata category (determines default patterns)
        required: If True, validation fails when field is missing
        zones: Search zones to scan (order determines priority)
        patterns: Regex patterns with capture groups for the value
        fallback: Strategy when field is not found
        default: Default value when fallback is DEFAULT
    """

    name: str
    category: MetadataCategory
    required: bool = False
    zones: list[SearchZone] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)  # Regex patterns
    fallback: FallbackStrategy = FallbackStrategy.FLAG
    default: Any = None


# ---------------------------------------------------------------------------
# TH1: Cell type detection
# ---------------------------------------------------------------------------


class CellType(Enum):
    """Classification of cell content for type pattern analysis.

    Used by TH1-TH3 heuristics to distinguish headers (string-only) from
    data rows (containing dates, numbers, enums).
    """

    DATE = "date"
    NUMBER = "number"
    ENUM = "enum"  # Repeated categorical value
    STRING = "string"  # Default fallback


# Date patterns for type detection
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


def detect_cell_type(text: str | None) -> CellType:
    """TH1: Classify a cell's content as DATE, NUMBER, or STRING.

    Note: ENUM detection requires column-level analysis (repeated values)
    and cannot be done at the single-cell level. This function returns
    STRING for potential enum values; use detect_column_types() for
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


def detect_column_types(
    grid: list[list[str]],
    skip_header_rows: int = 0,
) -> list[CellType]:
    """TH3: Detect the predominant type for each column.

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
            cell = cell.strip() if isinstance(cell, str) else str(cell).strip()
            if cell:
                col_values[ci].append(cell)
                col_types[ci][detect_cell_type(cell)] += 1

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
            len(unique_values) <= 5 or unique_ratio <= 0.1
        ) and len(values) >= 3  # Need at least 3 values to detect enum

        if is_enum and type_counts.get(CellType.STRING, 0) > 0:
            result.append(CellType.ENUM)
        elif type_counts:
            # Most common type wins
            result.append(type_counts.most_common(1)[0][0])
        else:
            result.append(CellType.STRING)

    return result


def row_type_signature(row: list[str]) -> list[CellType]:
    """Get the type signature of a row (list of cell types)."""
    return [detect_cell_type(cell) for cell in row]


def is_header_type_pattern(row: list[str]) -> bool:
    """TH2: Check if a row has a header-like type pattern (all strings, no dates/numbers).

    Headers are labels (strings), not data (dates, numbers, enums).
    """
    if not row:
        return False

    for cell in row:
        cell_text = cell.strip() if isinstance(cell, str) else str(cell).strip()
        cell_type = detect_cell_type(cell_text)
        if cell_type in (CellType.DATE, CellType.NUMBER):
            return False

    return True


# ---------------------------------------------------------------------------
# H7: Header row estimation
# ---------------------------------------------------------------------------


def estimate_header_rows(grid: list[list[str]]) -> int:
    """H7: Estimate header row count using bottom-up span-count analysis.

    Data rows dominate the bottom portion of any table. Find the most
    common cell-occupancy patterns there, then scan from the top to find
    the first row matching a data pattern.

    A row is data-like if:
    - Its filled cell count matches one of the common data patterns, OR
    - Its filled cell count is GREATER than the max common count (meaning
      it's a complete row with more columns filled).

    Args:
        grid: Table grid (list of rows, each row is list of cell strings)

    Returns:
        Estimated number of header rows (0 if no clear pattern)
    """
    n = len(grid)
    if n <= 2:
        return 0

    # Count non-empty cells per row (analogous to span_counts in PDF)
    cell_counts = [sum(1 for cell in row if cell.strip()) for row in grid]

    # Bottom 2/3 establishes what data rows look like.
    bottom_start = max(1, n // 3)
    bottom_counts = cell_counts[bottom_start:]

    freq = Counter(bottom_counts)
    # Top-3 most common cell counts. Requirements:
    # - count >= 2 (sparse rows aren't data)
    # - frequency >= 2 (pattern must repeat to be characteristic of data)
    data_counts = {c for c, count in freq.most_common(3) if c >= 2 and count >= 2}

    if not data_counts:
        return 0

    max_data_count = max(data_counts)

    # Scan from top: first row with a data-like cell count.
    # A row with MORE filled cells than typical is still data (more complete).
    for i in range(n):
        if cell_counts[i] in data_counts or cell_counts[i] > max_data_count:
            return i

    return 0


def estimate_header_rows_from_types(grid: list[list[str]]) -> int:
    """Estimate header rows using type pattern analysis.

    Scans from the top and counts consecutive rows that have header-like
    type patterns (all strings, no dates/numbers).

    This is a secondary method that can be used to validate or override
    the span-count based estimation.
    """
    header_count = 0
    for row in grid:
        if is_header_type_pattern(row):
            header_count += 1
        else:
            break
    return header_count


# ---------------------------------------------------------------------------
# Unified structured table format
# ---------------------------------------------------------------------------


@dataclass
class StructuredTable:
    """A table in structured format, used across all extractors.

    This is the common output format for PDF, XLSX, DOCX, PPTX, and HTML
    extractors, enabling shared downstream processing in interpret.py.
    """

    column_names: list[str]  # Final flattened header names
    data: list[list[str]]  # Data rows (each row has len == len(column_names))
    source_format: str  # "pdf", "xlsx", "docx", "pptx", "html"
    metadata: dict[str, str | int | None] | None = None  # Format-specific metadata


@dataclass
class TableMetadata:
    """Non-data/header info for a table."""

    page_number: int
    table_index: int  # For side-by-side tables (0, 1, ...)
    section_label: str | None  # GERALDTON, etc. if detected above table
    sheet_name: str | None = None  # For XLSX
    slide_number: int | None = None  # For PPTX


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------


def normalize_grid(grid: list[list[str]]) -> list[list[str]]:
    """Ensure all rows have the same number of columns.

    Pads shorter rows with empty strings.
    """
    if not grid:
        return grid

    max_cols = max(len(row) for row in grid)
    return [row + [""] * (max_cols - len(row)) for row in grid]


def split_headers_from_data(
    grid: list[list[str]],
    header_row_count: int | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Split a grid into header rows and data rows.

    If header_row_count is None, uses estimate_header_rows() to detect.
    """
    if not grid:
        return [], []

    if header_row_count is None:
        header_row_count = estimate_header_rows(grid)

    headers = grid[:header_row_count]
    data = grid[header_row_count:]
    return headers, data


def build_column_names_from_headers(header_rows: list[list[str]]) -> list[str]:
    """Build column names by stacking header rows vertically.

    For multi-row headers, concatenates values from each row with spaces.
    Deduplicates consecutive identical words.
    """
    if not header_rows:
        return []

    num_cols = max(len(row) for row in header_rows)
    col_fragments: list[list[str]] = [[] for _ in range(num_cols)]

    for row in header_rows:
        for ci, cell in enumerate(row):
            if ci < num_cols:
                text = cell.strip() if isinstance(cell, str) else str(cell).strip()
                if text:
                    col_fragments[ci].append(text)

    # Build final column names: fragment-level dedup + ` / ` join
    # This produces compound headers like "Group / Metric" that match
    # the DOCX extractor's ` / ` separator convention.
    column_names = []
    for fragments in col_fragments:
        deduped: list[str] = []
        for fragment in fragments:
            f = fragment.strip()
            if f and (not deduped or f != deduped[-1]):
                deduped.append(f)
        column_names.append(" / ".join(deduped))

    return column_names


# ---------------------------------------------------------------------------
# RH1: Temporal pattern detection
# ---------------------------------------------------------------------------

# Patterns for detecting temporal metadata (dates, periods, fiscal years)
# Each tuple: (compiled regex, pattern name)
_TEMPORAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "As of" date patterns
    (re.compile(r"[Aa]s\s+of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"), "as_of_date"),
    (re.compile(r"[Aa]s\s+of\s+(\d{1,2}/\d{1,2}/\d{2,4})"), "as_of_date"),
    (re.compile(r"[Aa]s\s+of\s+(\d{4}-\d{2}-\d{2})"), "as_of_date"),
    # Period end patterns
    (
        re.compile(
            r"[Ff]or\s+the\s+(?:year|period|quarter)\s+ended?\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"
        ),
        "period_end",
    ),
    (
        re.compile(
            r"[Ff]or\s+the\s+(?:year|period|quarter)\s+ended?\s+(\d{1,2}/\d{1,2}/\d{2,4})"
        ),
        "period_end",
    ),
    # Quarter patterns
    (re.compile(r"\b(Q[1-4])\s*(?:FY)?(\d{2,4})"), "quarter"),
    (re.compile(r"\b(1st|2nd|3rd|4th)\s+[Qq]uarter\s+(\d{4})"), "quarter"),
    # Fiscal year patterns
    (re.compile(r"\bFY\s*(\d{2,4})"), "fiscal_year"),
    (re.compile(r"\b[Ff]iscal\s+[Yy]ear\s+(\d{4})"), "fiscal_year"),
    # Year-month patterns
    (re.compile(r"\b([A-Za-z]+)\s+(\d{4})\b"), "year_month"),
    # Date range patterns
    (
        re.compile(
            r"(\d{1,2}/\d{1,2}/\d{2,4})\s*[-–—to]+\s*(\d{1,2}/\d{1,2}/\d{2,4})"
        ),
        "date_range",
    ),
    (
        re.compile(
            r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s*[-–—to]+\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})"
        ),
        "date_range",
    ),
]


def detect_temporal_patterns(text: str) -> list[tuple[str, str, str]]:
    """RH1: Detect temporal patterns in text.

    Scans text for date, period, quarter, and fiscal year patterns.

    Args:
        text: Text to scan for temporal patterns

    Returns:
        List of (matched_text, pattern_name, captured_value) tuples.
        For patterns with multiple groups, captured_value is space-joined.
    """
    results: list[tuple[str, str, str]] = []

    for pattern, name in _TEMPORAL_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            groups = match.groups()
            # Join multiple capture groups (e.g., quarter + year)
            captured = " ".join(g for g in groups if g)
            results.append((matched_text, name, captured))

    return results


# ---------------------------------------------------------------------------
# RH4: Unit/currency pattern detection
# ---------------------------------------------------------------------------

# Patterns for detecting unit and currency metadata
_UNIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Scale patterns
    (re.compile(r"\(in\s+(millions?|thousands?|billions?)\)", re.IGNORECASE), "scale"),
    (
        re.compile(
            r"\b(millions?|thousands?|billions?)\s+of\s+(?:dollars|euros|pounds)",
            re.IGNORECASE,
        ),
        "scale",
    ),
    (re.compile(r"(?:USD|EUR|GBP)\s+(000s?|MM|M|K|B)", re.IGNORECASE), "scale"),
    # Currency code patterns
    (re.compile(r"\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|INR)\b"), "currency_code"),
    # Currency symbol patterns
    (re.compile(r"[\$€£¥]"), "currency_symbol"),
    # Percentage patterns
    (re.compile(r"\b(percentage|percent|%)\b", re.IGNORECASE), "percentage"),
    # Unit patterns
    (re.compile(r"\b(metric\s+tons?|tonnes?|MT|kg|lbs?)\b", re.IGNORECASE), "unit"),
    (
        re.compile(r"\b(shares?|units?|contracts?|lots?)\b", re.IGNORECASE),
        "unit",
    ),
]


def detect_unit_patterns(text: str) -> list[tuple[str, str, str]]:
    """RH4: Detect unit and currency patterns in text.

    Scans text for scale indicators (millions, thousands), currency codes/symbols,
    and unit specifications.

    Args:
        text: Text to scan for unit patterns

    Returns:
        List of (matched_text, pattern_name, captured_value) tuples.
        For currency symbols, captured_value is the symbol itself.
    """
    results: list[tuple[str, str, str]] = []

    for pattern, name in _UNIT_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            # Get first capture group if exists, else full match
            captured = match.group(1) if match.lastindex else matched_text
            results.append((matched_text, name, captured))

    return results


# ---------------------------------------------------------------------------
# RH5: Footnote marker detection
# ---------------------------------------------------------------------------


def detect_footnote_markers(text: str) -> list[tuple[str, int | None]]:
    """RH5: Detect footnote markers in text.

    Finds superscript-style markers: ¹²³, [1], (1), *, †, ‡

    Args:
        text: Text to scan for footnote markers

    Returns:
        List of (marker_text, number_or_none) tuples.
        Number is extracted from numeric markers, None for symbol markers.
    """
    results: list[tuple[str, int | None]] = []

    # Unicode superscripts
    superscript_pattern = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+")
    for match in superscript_pattern.finditer(text):
        marker = match.group(0)
        # Convert superscripts to number
        trans = str.maketrans("¹²³⁴⁵⁶⁷⁸⁹⁰", "1234567890")
        num_str = marker.translate(trans)
        results.append((marker, int(num_str) if num_str.isdigit() else None))

    # Bracketed numbers: [1], (1), {1}
    bracket_pattern = re.compile(r"[\[\({\{](\d+)[\]\)}]")
    for match in bracket_pattern.finditer(text):
        results.append((match.group(0), int(match.group(1))))

    # Symbol markers
    symbol_pattern = re.compile(r"[*†‡§¶#]+")
    for match in symbol_pattern.finditer(text):
        results.append((match.group(0), None))

    return results
