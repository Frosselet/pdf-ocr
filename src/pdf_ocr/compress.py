"""Compressed spatial text for LLM consumption.

Takes a PDF and produces a token-efficient structured text representation
using markdown tables, flowing text, and key-value pairs instead of
whitespace-heavy spatial grids.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import fitz  # PyMuPDF

from pdf_ocr.spatial_text import PageLayout, _extract_page_layout, _open_pdf

logger = logging.getLogger(__name__)


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


def _anchors_overlap(
    cols_a: tuple[int, ...] | list[int],
    cols_b: tuple[int, ...] | list[int],
    tolerance: int,
) -> int:
    """Count how many column anchors overlap within tolerance."""
    count = 0
    for a in cols_a:
        for b in cols_b:
            if abs(a - b) <= tolerance:
                count += 1
                break
    return count


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

    regions: list[Region] = []
    used: set[int] = set()

    # Detect tables.
    table_runs = _detect_table_runs(
        row_indices, col_positions, span_counts, min_table_rows, col_tolerance,
        row_texts=row_texts,
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


def _detect_table_runs(
    row_indices: list[int],
    col_positions: dict[int, list[int]],
    span_counts: dict[int, int],
    min_table_rows: int,
    col_tolerance: int,
    row_texts: dict[int, str] | None = None,
) -> list[list[int]]:
    """Find maximal runs of rows that form table regions.

    Uses a pool of all column anchors seen so far in the run. A new row
    joins the run if it shares 2+ anchors with the pool. This handles
    alternating row patterns (e.g., data rows with 7 columns interleaved
    with date rows with 5 columns that occupy a subset of positions).
    """
    if len(row_indices) < min_table_rows:
        return []

    runs: list[list[int]] = []
    current_run: list[int] = []
    # Pool of all unique column anchors in the current run.
    pool: set[int] = set()

    def _pool_overlap(cols: list[int]) -> int:
        """Count how many of cols match something in the pool."""
        count = 0
        for c in cols:
            for p in pool:
                if abs(c - p) <= col_tolerance:
                    count += 1
                    break
        return count

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
        overlap = _pool_overlap(cols)

        if overlap >= 2 and gap <= 2:
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
    """
    n = len(rows)
    if n <= 2:
        return 0

    span_counts = [len(row) for row in rows]

    # Bottom 2/3 establishes what data rows look like.
    bottom_start = max(1, n // 3)
    bottom_counts = span_counts[bottom_start:]

    freq = Counter(bottom_counts)
    # Top-3 most common span counts (minimum 2 spans to be data-like).
    data_counts = {c for c, _ in freq.most_common(3) if c >= 2}

    if not data_counts:
        return 0

    # Scan from top: first row with a data-like span count.
    for i in range(n):
        if span_counts[i] in data_counts:
            return i

    return 0


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
# Region renderers
# ---------------------------------------------------------------------------

def _render_table_markdown(
    rows: list[list[tuple[int, str]]],
    col_tolerance: int = 3,
) -> str:
    """Render table rows as a markdown pipe-delimited table."""
    if not rows:
        return ""

    canonical, col_map = _unify_columns(rows, col_tolerance)
    num_cols = len(canonical)
    grid = _rows_to_grid(rows, num_cols, col_map)

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

def _render_spatial_excerpt(layout: PageLayout, row_indices: list[int]) -> str:
    """Render specific rows from a PageLayout as monospace spatial text."""
    lines: list[str] = []
    for row_idx in row_indices:
        if row_idx not in layout.rows:
            lines.append("")
            continue
        entries = layout.rows[row_idx]
        max_end = max(col + len(text) for col, text in entries)
        buf = [" "] * max_end
        for col, text in entries:
            for i, ch in enumerate(text):
                pos = col + i
                if pos < len(buf):
                    buf[pos] = ch
        lines.append("".join(buf).rstrip())
    return "\n".join(lines)


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


def _refine_headers_with_llm(spatial_excerpt: str, data_column_count: int):
    """Call LLM to refine table headers. Returns RefinedHeaders or None on failure."""
    try:
        from baml_client.sync_client import b as b_sync
        refined = b_sync.RefineTableHeaders(spatial_excerpt, data_column_count)
        logger.debug(
            "LLM header refinement: structure=%s, header_rows=%d, columns=%d",
            refined.header_structure.value, refined.header_row_count, len(refined.column_names),
        )
        if len(refined.column_names) != data_column_count:
            logger.debug(
                "LLM returned %d column names, expected %d",
                len(refined.column_names), data_column_count,
            )
        return refined
    except Exception:
        logger.debug("LLM header refinement failed, falling back to heuristic", exc_info=True)
        return None


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
# Page compression
# ---------------------------------------------------------------------------

def _compress_page(
    page: fitz.Page,
    cluster_threshold: float = 2.0,
    table_format: str = "markdown",
    merge_multi_row: bool = True,
    min_table_rows: int = 3,
    refine_headers: bool = True,
) -> str:
    """Compress a single PDF page into structured text."""
    layout = _extract_page_layout(page, cluster_threshold)
    if layout is None:
        return ""

    # Split spans that the PDF merged across column boundaries.
    layout = _split_merged_spans(layout)

    col_tolerance = 3
    regions = _classify_regions(layout, min_table_rows, col_tolerance)

    if not regions:
        return ""

    has_tables = any(r.type == RegionType.TABLE for r in regions)

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

            if refine_headers and table_format == "markdown":
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
                num_cols = len(canonical)

                # 5. Grid data rows and merge twin columns.
                grid = _rows_to_grid(data_rows, num_cols, col_map)
                grid, num_cols = _merge_twin_columns(grid, num_cols)

                # 6. Find header rows above this table region.
                preceding = _find_preceding_header_rows(
                    layout, region.row_indices[0], canonical,
                    render_tolerance, all_table_rows,
                )

                # 7. LLM refines headers with extended spatial context.
                if preceding:
                    logger.debug(
                        "Found %d preceding header rows: %s",
                        len(preceding), preceding,
                    )
                excerpt_rows = preceding + region.row_indices
                spatial_excerpt = _render_spatial_excerpt(layout, excerpt_rows)
                refined = _refine_headers_with_llm(spatial_excerpt, num_cols)

                if refined is not None:
                    if refined.header_structure.value == "Hierarchical":
                        logger.debug(
                            "Hierarchical headers detected — compound names may need unpivoting downstream"
                        )
                    # 8. Render with LLM names and data-derived grid.
                    rendered_parts.append(
                        _render_table_markdown_with_headers(grid, refined.column_names)
                    )
                else:
                    # LLM failed — fall back to heuristic.
                    rendered_parts.append(
                        _render_table_markdown(table_rows, render_tolerance)
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
        ))
    doc.close()
    return page_separator.join(rendered)
