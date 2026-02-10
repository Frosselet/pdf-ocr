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

Geometry heuristics (y-clustering, x-position alignment) remain in format-specific
extractors since they depend on spatial coordinates.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum


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

    # Build final column names with deduplication
    column_names = []
    for fragments in col_fragments:
        deduped_words: list[str] = []
        for fragment in fragments:
            for word in fragment.split():
                if not deduped_words or word != deduped_words[-1]:
                    deduped_words.append(word)
        column_names.append(" ".join(deduped_words))

    return column_names
