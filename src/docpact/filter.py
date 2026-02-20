"""Filter PDF pages by table titles using fuzzy matching.

Extracts table titles from PDF pages and filters to only pages containing
titles that match the search terms above a given threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from rapidfuzz import fuzz

from docpact.compress import RegionType, _classify_regions
from docpact.spatial_text import _extract_page_layout, _open_pdf


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FilterMatch:
    """A match between a page's title and a search term."""

    page_index: int  # 0-based page index
    matched_title: str  # The concatenated title text that matched
    search_term: str  # The search term it matched against
    score: float  # Fuzzy match score (0-100)


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------

# Patterns for text that should be skipped when building title groups.
_FOOTNOTE_PATTERNS = [
    re.compile(r"^\d+$"),  # Single digits like "2", "3"
    re.compile(r"^Includes\b", re.IGNORECASE),
    re.compile(r"^ACEA\b", re.IGNORECASE),
    re.compile(r"^\*+$"),  # Asterisks
    re.compile(r"^Note:", re.IGNORECASE),
    re.compile(r"^Source:", re.IGNORECASE),
]


def _is_footnote_text(text: str) -> bool:
    """Check if text looks like a footnote marker or footnote text."""
    text = text.strip()
    for pattern in _FOOTNOTE_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _extract_titles_from_page(
    page: fitz.Page,
    cluster_threshold: float = 2.0,
) -> list[tuple[int, str]]:
    """Extract HEADING regions from a page.

    Returns:
        List of (row_index, heading_text) tuples for each HEADING region.
    """
    layout = _extract_page_layout(page, cluster_threshold)
    if layout is None:
        return []

    regions = _classify_regions(layout)
    headings: list[tuple[int, str]] = []

    for region in regions:
        if region.type == RegionType.HEADING:
            # Get the text from this heading region
            for ri in region.row_indices:
                if ri in region.rows:
                    text = " ".join(t for _, t in sorted(region.rows[ri]))
                    if text.strip() and not _is_footnote_text(text):
                        headings.append((ri, text.strip()))

    return headings


def _concatenate_title_groups(
    headings: list[tuple[int, str]],
    max_gap: int = 3,
) -> list[str]:
    """Group consecutive headings and concatenate them.

    Handles multi-line titles like:
    - "NEW CAR REGISTRATIONS BY MARKET AND POWER SOURCE"
    - "MONTHLY"
    Which should become one title: "NEW CAR REGISTRATIONS BY MARKET AND POWER SOURCE MONTHLY"

    Args:
        headings: List of (row_index, text) from _extract_titles_from_page.
        max_gap: Maximum row gap to consider headings as part of same group.

    Returns:
        List of concatenated title strings.
    """
    if not headings:
        return []

    groups: list[list[str]] = []
    current_group: list[str] = [headings[0][1]]
    last_row = headings[0][0]

    for row_idx, text in headings[1:]:
        if row_idx - last_row <= max_gap:
            current_group.append(text)
        else:
            groups.append(current_group)
            current_group = [text]
        last_row = row_idx

    groups.append(current_group)

    return [" ".join(group) for group in groups]


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------


def _fuzzy_match_titles(
    titles: list[str],
    search_terms: list[str],
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Find titles that match any search term above threshold.

    Uses rapidfuzz.fuzz.WRatio for robust matching that handles:
    - Partial matches
    - Word reordering
    - Typos

    Args:
        titles: List of title strings to search.
        search_terms: List of search terms to match against.
        threshold: Minimum score (0-100) for a match.

    Returns:
        List of (title, search_term, score) for matches above threshold.
    """
    matches: list[tuple[str, str, float]] = []

    for title in titles:
        title_lower = title.lower()
        for term in search_terms:
            term_lower = term.lower()
            score = fuzz.WRatio(title_lower, term_lower)
            if score >= threshold:
                matches.append((title, term, score))

    return matches


# ---------------------------------------------------------------------------
# PDF creation
# ---------------------------------------------------------------------------


def _create_filtered_pdf(
    source_doc: fitz.Document,
    page_indices: list[int],
) -> bytes:
    """Create a new PDF containing only the specified pages.

    Args:
        source_doc: The source PDF document.
        page_indices: 0-based indices of pages to include.

    Returns:
        The new PDF as bytes.
    """
    new_doc = fitz.open()
    for idx in page_indices:
        new_doc.insert_pdf(source_doc, from_page=idx, to_page=idx)
    pdf_bytes = new_doc.tobytes()
    new_doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_table_titles(
    pdf_input: str | bytes | Path,
    *,
    cluster_threshold: float = 2.0,
) -> dict[int, list[str]]:
    """Extract potential table titles from each page.

    Useful for debugging and understanding what titles are detected.

    Args:
        pdf_input: Path to PDF file, or raw PDF bytes.
        cluster_threshold: Y-distance threshold for row clustering.

    Returns:
        Dict mapping page index to list of concatenated title strings.
    """
    doc = _open_pdf(pdf_input)
    result: dict[int, list[str]] = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        headings = _extract_titles_from_page(page, cluster_threshold)
        titles = _concatenate_title_groups(headings)
        if titles:
            result[page_idx] = titles

    doc.close()
    return result


def filter_pdf_by_table_titles(
    pdf_input: str | bytes | Path,
    search_terms: str | list[str] | None = None,
    *,
    pages: list[int] | None = None,
    threshold: float = 90.0,
    output_path: str | Path | None = None,
    cluster_threshold: float = 2.0,
    match_any: bool = True,
) -> tuple[bytes, list[FilterMatch]]:
    """Filter PDF to pages by direct selection or fuzzy title matching.

    Two filtering modes:
    - **By pages**: Provide `pages` list of 0-based page indices
    - **By titles**: Provide `search_terms` for fuzzy matching against titles

    Both can be combined (union of results).

    Args:
        pdf_input: Path to PDF file, or raw PDF bytes.
        search_terms: Single search term or list of terms to match against
            table titles. Optional if `pages` is provided.
        pages: List of 0-based page indices to include directly.
            Optional if `search_terms` is provided.
        threshold: Minimum fuzzy match score (0-100). Default 90.0.
        output_path: Optional path to write the filtered PDF.
        cluster_threshold: Y-distance threshold for row clustering.
        match_any: If True, include page if ANY term matches. If False,
            ALL terms must match (not yet implemented).

    Returns:
        Tuple of (filtered_pdf_bytes, list_of_matches).

    Raises:
        ValueError: If neither `search_terms` nor `pages` is provided,
            or if no pages match the criteria.

    Examples:
        Filter by fuzzy title search:

        >>> filtered_bytes, matches = filter_pdf_by_table_titles(
        ...     "report.pdf",
        ...     "new car registrations by market and power source, monthly",
        ... )

        Filter by page numbers (when pages are always the same):

        >>> filtered_bytes, matches = filter_pdf_by_table_titles(
        ...     "report.pdf",
        ...     pages=[2],  # 0-based, so page 3
        ... )

        Combine both (union):

        >>> filtered_bytes, matches = filter_pdf_by_table_titles(
        ...     "report.pdf",
        ...     "monthly registrations",
        ...     pages=[0, 5],  # Also include first and last page
        ... )
    """
    if search_terms is None and pages is None:
        raise ValueError("Must provide either 'search_terms' or 'pages' (or both)")

    # Normalize search_terms to list
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    doc = _open_pdf(pdf_input)
    num_pages = len(doc)
    matches: list[FilterMatch] = []
    matched_pages: set[int] = set()

    # Direct page selection
    if pages is not None:
        for page_idx in pages:
            if 0 <= page_idx < num_pages:
                matched_pages.add(page_idx)
                matches.append(FilterMatch(
                    page_index=page_idx,
                    matched_title="",
                    search_term=f"page:{page_idx}",
                    score=100.0,
                ))
            else:
                doc.close()
                raise ValueError(
                    f"Page index {page_idx} out of range (PDF has {num_pages} pages)"
                )

    # Fuzzy title matching
    if search_terms is not None:
        for page_idx in range(num_pages):
            page = doc[page_idx]
            headings = _extract_titles_from_page(page, cluster_threshold)
            titles = _concatenate_title_groups(headings)

            page_matches = _fuzzy_match_titles(titles, search_terms, threshold)

            for title, term, score in page_matches:
                matches.append(FilterMatch(
                    page_index=page_idx,
                    matched_title=title,
                    search_term=term,
                    score=score,
                ))
                matched_pages.add(page_idx)

    if not matched_pages:
        doc.close()
        raise ValueError(
            f"No pages found matching criteria "
            f"(search_terms={search_terms!r}, pages={pages!r}, threshold={threshold})"
        )

    # Create filtered PDF with matching pages in order
    sorted_pages = sorted(matched_pages)
    filtered_bytes = _create_filtered_pdf(doc, sorted_pages)
    doc.close()

    # Optionally write to file
    if output_path is not None:
        Path(output_path).write_bytes(filtered_bytes)

    return filtered_bytes, matches
