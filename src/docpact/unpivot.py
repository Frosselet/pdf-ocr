"""Schema-agnostic deterministic pivot detection and unpivoting for pipe-tables.

Detects repeating header groups in compound headers (split on ` / `) and
transforms wide pivoted tables into long format by adding a `_pivot` column.

This is a pre-processing step that runs before interpretation (LLM or
deterministic).  After unpivoting, downstream receives a flat table — much
simpler to interpret.

Example::

    Input:
    | Region | spring crops / Target | spring crops / 2025 | spring grain / Target | spring grain / 2025 |
    |---|---|---|---|---|
    | Belgorod | 100 | 90 | 50 | 45 |

    Output:
    | _pivot | Region | Target | 2025 |
    |---|---|---|---|
    | spring crops | Belgorod | 100 | 90 |
    | spring grain | Belgorod | 50 | 45 |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class PivotGroup:
    """A group of columns sharing a common prefix in compound headers."""

    prefix: str
    col_indices: list[int]
    suffixes: list[str]
    matched_suffix_indices: list[int] = field(default_factory=list)


@dataclass
class PivotDetection:
    """Result of pivot structure detection."""

    is_pivoted: bool
    shared_col_indices: list[int] = field(default_factory=list)
    groups: list[PivotGroup] = field(default_factory=list)
    measure_labels: list[str] = field(default_factory=list)


@dataclass
class UnpivotResult:
    """Result of unpivot transformation."""

    text: str
    was_transformed: bool
    detection: PivotDetection | None = None


# ─── Internal helpers ─────────────────────────────────────────────────────────

# Matches a pipe-table header row: starts with |, contains at least one more |
_PIPE_HEADER_RE = re.compile(r"^\|(.+\|)\s*$")
# Matches separator row: | followed by ---+| patterns
_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")
# Matches a section label (bold): **LABEL**
_SECTION_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")
# Matches a heading line: ## HEADING
_HEADING_RE = re.compile(r"^##\s+.+")
# Matches a data row: starts with |, not a separator
_DATA_ROW_RE = re.compile(r"^\|(.+\|)\s*$")


def _parse_header_cells(header_line: str) -> list[str]:
    """Extract cell contents from a pipe-table header row."""
    # Strip leading/trailing pipe and split
    inner = header_line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _split_compound(header: str) -> tuple[str | None, str]:
    """Split a compound header on ' / '.

    Returns (prefix, suffix) for compound headers, or (None, header) for simple.
    """
    parts = header.split(" / ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, header.strip()


def _match_suffix_lists(
    reference: list[str],
    candidate: list[str],
    threshold: float,
) -> list[tuple[int, int, float]] | None:
    """Greedy-match suffixes between two groups.

    Returns [(ref_idx, cand_idx, score)] or None if < 2 matches.
    """
    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for ri, ref in enumerate(reference):
        best_ci, best_score = -1, 0.0
        for ci, cand in enumerate(candidate):
            if ci in used:
                continue
            score = fuzz.ratio(ref.lower(), cand.lower())
            if score > best_score:
                best_score, best_ci = score, ci
        if best_score >= threshold and best_ci >= 0:
            matches.append((ri, best_ci, best_score))
            used.add(best_ci)
    return matches if len(matches) >= 2 else None


def _parse_data_cells(row_line: str) -> list[str]:
    """Extract cell contents from a pipe-table data row."""
    return _parse_header_cells(row_line)


def _build_pipe_row(cells: list[str]) -> str:
    """Build a pipe-table row from cell values."""
    return "| " + " | ".join(cells) + " |"


def _build_separator(num_cols: int) -> str:
    """Build a pipe-table separator row."""
    return "|" + "|".join(["---"] * num_cols) + "|"


# ─── Public API ───────────────────────────────────────────────────────────────


def detect_pivot(
    compressed_text: str,
    *,
    similarity_threshold: float = 85.0,
    min_groups: int = 2,
) -> PivotDetection:
    """Detect pivot structure in a pipe-table without transforming.

    Args:
        compressed_text: Pipe-table markdown text (single page).
        similarity_threshold: Minimum rapidfuzz.fuzz.ratio score (0-100)
            for two suffixes to be considered matching.
        min_groups: Minimum number of groups with matching suffixes to
            consider the table pivoted.

    Returns:
        PivotDetection with is_pivoted=True if pivot structure found.
    """
    not_pivoted = PivotDetection(is_pivoted=False)

    # Find the header line
    lines = compressed_text.split("\n")
    header_line = None
    for line in lines:
        if _PIPE_HEADER_RE.match(line) and not _SEPARATOR_RE.match(line):
            header_line = line
            break

    if header_line is None:
        return not_pivoted

    headers = _parse_header_cells(header_line)
    if len(headers) < 2:
        return not_pivoted

    # Step 1: Split compound headers, group by prefix
    groups_by_prefix: dict[str, list[tuple[int, str]]] = {}
    shared_indices: list[int] = []

    for i, h in enumerate(headers):
        prefix, suffix = _split_compound(h)
        if prefix is None:
            shared_indices.append(i)
        else:
            groups_by_prefix.setdefault(prefix, []).append((i, suffix))

    if len(groups_by_prefix) < min_groups:
        return not_pivoted

    # Build PivotGroup objects
    groups: list[PivotGroup] = []
    for prefix, cols in groups_by_prefix.items():
        indices = [c[0] for c in cols]
        suffixes = [c[1] for c in cols]
        groups.append(PivotGroup(prefix=prefix, col_indices=indices, suffixes=suffixes))

    # Step 2: Try each group as reference, pick the one yielding the most matches.
    # This handles cases where a non-matching group (e.g., Th.ha.) has the most
    # suffixes — we skip it and find the real pivot groups.
    # Sort by suffix count descending so we try the most-suffix groups first.
    groups.sort(key=lambda g: len(g.suffixes), reverse=True)

    best_ref: PivotGroup | None = None
    best_matching: list[PivotGroup] = []
    best_all_matches: list[list[tuple[int, int, float]]] = []
    best_unmatched: list[PivotGroup] = []

    for ref_idx, ref_candidate in enumerate(groups):
        others = [g for j, g in enumerate(groups) if j != ref_idx]
        candidate_matches: list[list[tuple[int, int, float]]] = []
        candidate_matching: list[PivotGroup] = []
        candidate_unmatched: list[PivotGroup] = []

        for g in others:
            m = _match_suffix_lists(ref_candidate.suffixes, g.suffixes, similarity_threshold)
            if m is not None:
                candidate_matches.append(m)
                candidate_matching.append(g)
            else:
                candidate_unmatched.append(g)

        total_matching = 1 + len(candidate_matching)  # ref + matched
        if total_matching >= min_groups and total_matching > 1 + len(best_matching):
            best_ref = ref_candidate
            best_matching = candidate_matching
            best_all_matches = candidate_matches
            best_unmatched = candidate_unmatched

    if best_ref is None or len(best_matching) < (min_groups - 1):
        return not_pivoted

    # Non-matching groups become shared columns
    for g in best_unmatched:
        shared_indices.extend(g.col_indices)

    # Step 3: Find intersection of matched reference indices across all groups
    matched_ref_indices_sets = [
        {m[0] for m in match_list} for match_list in best_all_matches
    ]
    common_ref_indices = set.intersection(*matched_ref_indices_sets)

    if len(common_ref_indices) < 2:
        return not_pivoted

    # Sort by position in reference suffix list
    common_ref_indices_sorted = sorted(common_ref_indices)

    # Build measure labels from reference suffixes
    measure_labels = [best_ref.suffixes[ri] for ri in common_ref_indices_sorted]

    # Build final matching groups list (reference first, then matched in original order)
    matching_groups = [best_ref] + best_matching

    # Set matched_suffix_indices on each group
    best_ref.matched_suffix_indices = common_ref_indices_sorted

    for g_idx, g in enumerate(best_matching):
        match_list = best_all_matches[g_idx]
        match_map = {ri: ci for ri, ci, _ in match_list}
        g.matched_suffix_indices = [match_map[ri] for ri in common_ref_indices_sorted]

    # Sort shared indices
    shared_indices.sort()

    return PivotDetection(
        is_pivoted=True,
        shared_col_indices=shared_indices,
        groups=matching_groups,
        measure_labels=measure_labels,
    )


def unpivot_pipe_table(
    compressed_text: str,
    *,
    similarity_threshold: float = 85.0,
    min_groups: int = 2,
    pivot_column_name: str = "_pivot",
) -> UnpivotResult:
    """Detect and unpivot a pivoted pipe-table. Schema-agnostic.

    Returns original text unchanged if no pivot detected.

    Args:
        compressed_text: Pipe-table markdown text (single page).
        similarity_threshold: Minimum rapidfuzz.fuzz.ratio score (0-100)
            for two suffixes to be considered matching.
        min_groups: Minimum number of groups with matching suffixes.
        pivot_column_name: Name for the new pivot column (default "_pivot").

    Returns:
        UnpivotResult with transformed text if pivoted, original otherwise.
    """
    detection = detect_pivot(
        compressed_text,
        similarity_threshold=similarity_threshold,
        min_groups=min_groups,
    )

    if not detection.is_pivoted:
        return UnpivotResult(
            text=compressed_text,
            was_transformed=False,
            detection=detection,
        )

    # Transform the table
    lines = compressed_text.split("\n")
    output_lines: list[str] = []

    # Determine new column structure:
    # [pivot_column_name] + shared columns (original headers) + measure labels
    shared_headers: list[str] = []
    for i in detection.shared_col_indices:
        # Re-parse the header to get original name
        pass  # We'll get them from the header line

    # Find header line index
    header_idx = None
    sep_idx = None
    for li, line in enumerate(lines):
        if header_idx is None and _PIPE_HEADER_RE.match(line) and not _SEPARATOR_RE.match(line):
            header_idx = li
        elif header_idx is not None and sep_idx is None and _SEPARATOR_RE.match(line):
            sep_idx = li
            break

    if header_idx is None:
        return UnpivotResult(text=compressed_text, was_transformed=False, detection=detection)

    original_headers = _parse_header_cells(lines[header_idx])

    # Build shared column headers (preserve original text, including compound names)
    shared_headers = [original_headers[i] for i in detection.shared_col_indices]

    # New header: pivot col + shared + measures
    new_headers = [pivot_column_name] + shared_headers + detection.measure_labels
    num_new_cols = len(new_headers)

    # Emit lines before the header (titles, headings)
    for li in range(header_idx):
        output_lines.append(lines[li])

    # Emit new header + separator
    output_lines.append(_build_pipe_row(new_headers))
    output_lines.append(_build_separator(num_new_cols))

    # Process remaining lines (after separator)
    start = (sep_idx + 1) if sep_idx is not None else (header_idx + 1)

    for li in range(start, len(lines)):
        line = lines[li]

        # Section labels: pass through as-is
        if _SECTION_RE.match(line):
            output_lines.append(line)
            continue

        # Heading lines: pass through
        if _HEADING_RE.match(line):
            output_lines.append(line)
            continue

        # Empty lines: pass through
        if not line.strip():
            output_lines.append(line)
            continue

        # Non-pipe-table lines: pass through
        if not _DATA_ROW_RE.match(line) or _SEPARATOR_RE.match(line):
            output_lines.append(line)
            continue

        # Data row (including aggregation rows): expand into one row per group
        cells = _parse_data_cells(line)

        # Pad cells if row is shorter than headers
        while len(cells) < len(original_headers):
            cells.append("")

        # Shared column values
        shared_values = [cells[i] if i < len(cells) else "" for i in detection.shared_col_indices]

        # One output row per group
        for group in detection.groups:
            measure_values = []
            for si in group.matched_suffix_indices:
                col_idx = group.col_indices[si]
                measure_values.append(cells[col_idx] if col_idx < len(cells) else "")

            new_cells = [group.prefix] + shared_values + measure_values
            output_lines.append(_build_pipe_row(new_cells))

    return UnpivotResult(
        text="\n".join(output_lines),
        was_transformed=True,
        detection=detection,
    )
