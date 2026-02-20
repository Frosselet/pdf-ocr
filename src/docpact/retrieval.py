"""Metadata retrieval layer for search & extract.

Provides a fast, deterministic metadata extraction phase that runs before
LLM-based table interpretation. Uses formal retrieval heuristics (RH1-RH5)
to extract document-level and table-level metadata from specific zones.

Flow:
    Quick Scan (regex/spatial) → Validation Gate → Deep Extraction (LLM)
                                        ↓ (missing)
                                   Fallback (default/prompt/flag)

Usage::

    from docpact import search_and_extract, CanonicalSchema, MetadataFieldDef

    schema = CanonicalSchema(
        columns=[...],
        metadata=[
            MetadataFieldDef(
                name="publication_date",
                category=MetadataCategory.TEMPORAL,
                required=True,
                zones=[SearchZone.TITLE_PAGE],
                patterns=[r"As of\\s+(.+?\\d{4})"],
            )
        ]
    )
    result = search_and_extract("inputs/annual_report.pdf", schema)
    print(result.metadata)
    print(result.validation.passed)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docpact.heuristics import (
    FallbackStrategy,
    MetadataCategory,
    MetadataFieldDef,
    SearchZone,
    detect_temporal_patterns,
    detect_unit_patterns,
)
from docpact.spatial_text import PageLayout, _extract_page_layout, _open_pdf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedMetadata:
    """A single metadata field value extracted from the document.

    Attributes:
        field_name: Name of the metadata field (from MetadataFieldDef.name)
        value: Extracted value, or None if not found
        source_zone: Zone where the value was found
        confidence: Confidence score (0.0 to 1.0)
        pattern_matched: The regex pattern that matched, if any
    """

    field_name: str
    value: str | None
    source_zone: SearchZone
    confidence: float
    pattern_matched: str | None = None


@dataclass
class ValidationResult:
    """Result of metadata validation against schema requirements.

    Attributes:
        passed: True if all required fields were found
        found: Dict of field_name -> RetrievedMetadata for fields found
        missing: List of required field names that were not found
    """

    passed: bool
    found: dict[str, RetrievedMetadata]
    missing: list[str]


@dataclass
class SearchExtractResult:
    """Complete result from search_and_extract().

    Attributes:
        metadata: Dict of field_name -> RetrievedMetadata
        tables: Dict of page_number -> MappedTable (from deep extraction)
        validation: Validation result for metadata
        warnings: List of warning messages
    """

    metadata: dict[str, RetrievedMetadata]
    tables: dict[int, Any]  # MappedTable, avoid circular import
    validation: ValidationResult
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Zone extraction
# ---------------------------------------------------------------------------


def _extract_zone_text(
    layout: PageLayout,
    zone: SearchZone,
    page_height: float,
    is_first_page: bool = False,
) -> str:
    """Extract text from a specific zone of a page.

    Uses row y-positions from PageLayout to filter rows by vertical zone.

    Args:
        layout: PageLayout with row data and y-positions
        zone: Zone to extract
        page_height: Total page height in points
        is_first_page: Whether this is the first page (for TITLE_PAGE zone)

    Returns:
        Concatenated text from the specified zone.
    """
    # Zone bounds as fraction of page height
    zone_bounds: dict[SearchZone, tuple[float, float]] = {
        SearchZone.PAGE_HEADER: (0.0, 0.15),
        SearchZone.PAGE_FOOTER: (0.85, 1.0),
        SearchZone.TITLE_PAGE: (0.0, 0.4),  # Only valid on first page
        SearchZone.TABLE_CAPTION: (0.0, 1.0),  # Handled specially
        SearchZone.COLUMN_HEADER: (0.0, 1.0),  # Handled specially
        SearchZone.TABLE_FOOTER: (0.0, 1.0),  # Handled specially
        SearchZone.ANYWHERE: (0.0, 1.0),
    }

    # Get zone bounds
    if zone == SearchZone.TITLE_PAGE and not is_first_page:
        return ""

    bounds = zone_bounds.get(zone, (0.0, 1.0))
    min_y = bounds[0] * page_height
    max_y = bounds[1] * page_height

    # Extract text from rows within bounds
    text_parts: list[str] = []
    y_positions = layout.row_y_positions

    for row_idx, row_data in sorted(layout.rows.items()):
        if row_idx < len(y_positions):
            y = y_positions[row_idx]
            if min_y <= y <= max_y:
                row_text = " ".join(text for _, text in sorted(row_data))
                text_parts.append(row_text)

    return "\n".join(text_parts)


def _get_full_page_text(layout: PageLayout) -> str:
    """Get all text from a PageLayout."""
    text_parts: list[str] = []
    for row_data in layout.rows.values():
        row_text = " ".join(text for _, text in sorted(row_data))
        text_parts.append(row_text)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------


def _apply_category_patterns(
    text: str,
    category: MetadataCategory,
) -> list[tuple[str, str, str]]:
    """Apply built-in patterns for a metadata category.

    Args:
        text: Text to scan
        category: Metadata category

    Returns:
        List of (matched_text, pattern_name, captured_value) tuples.
    """
    if category == MetadataCategory.TEMPORAL:
        return detect_temporal_patterns(text)
    elif category == MetadataCategory.TABLE_CONTEXT:
        return detect_unit_patterns(text)
    return []


def _apply_custom_patterns(
    text: str,
    patterns: list[str],
) -> list[tuple[str, str, str]]:
    """Apply custom regex patterns to text.

    Args:
        text: Text to scan
        patterns: List of regex patterns with capture groups

    Returns:
        List of (matched_text, pattern_string, captured_value) tuples.
    """
    results: list[tuple[str, str, str]] = []
    for pattern_str in patterns:
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                # Get first capture group if exists
                captured = match.group(1) if match.lastindex else matched_text
                results.append((matched_text, pattern_str, captured))
        except re.error as e:
            log.warning("Invalid regex pattern %r: %s", pattern_str, e)
    return results


# ---------------------------------------------------------------------------
# Quick scan
# ---------------------------------------------------------------------------


def quick_scan(
    pdf_input: str | Path,
    metadata_fields: list[MetadataFieldDef],
    *,
    pages: list[int] | None = None,
) -> dict[str, RetrievedMetadata]:
    """Phase 1: Fast regex-based metadata extraction.

    Scans specified pages (default: first 3) for metadata using pattern matching.
    Searches each field's zones in order, returning the first match.

    Args:
        pdf_input: Path to PDF file
        metadata_fields: List of MetadataFieldDef specifying what to extract
        pages: 0-based page indices to scan (default: first 3 pages)

    Returns:
        Dict of field_name -> RetrievedMetadata for all matched fields.
    """
    doc = _open_pdf(pdf_input)
    page_count = doc.page_count

    # Default to first 3 pages
    if pages is None:
        pages = list(range(min(3, page_count)))

    # Extract layouts for requested pages
    layouts: list[tuple[int, PageLayout, float]] = []
    for page_idx in pages:
        if 0 <= page_idx < page_count:
            page = doc[page_idx]
            layout = _extract_page_layout(page)
            layouts.append((page_idx, layout, page.rect.height))

    doc.close()

    # Process each metadata field
    results: dict[str, RetrievedMetadata] = {}

    for field_def in metadata_fields:
        # Determine zones to search (default to ANYWHERE if none specified)
        zones = field_def.zones if field_def.zones else [SearchZone.ANYWHERE]

        match_found = False

        for zone in zones:
            if match_found:
                break

            for page_idx, layout, page_height in layouts:
                is_first = page_idx == 0

                # Extract zone text
                if zone == SearchZone.ANYWHERE:
                    zone_text = _get_full_page_text(layout)
                else:
                    zone_text = _extract_zone_text(layout, zone, page_height, is_first)

                if not zone_text.strip():
                    continue

                # Try custom patterns first
                if field_def.patterns:
                    matches = _apply_custom_patterns(zone_text, field_def.patterns)
                    if matches:
                        matched_text, pattern_str, captured = matches[0]
                        results[field_def.name] = RetrievedMetadata(
                            field_name=field_def.name,
                            value=captured,
                            source_zone=zone,
                            confidence=0.9,  # Custom pattern match = high confidence
                            pattern_matched=pattern_str,
                        )
                        match_found = True
                        break

                # Try built-in category patterns
                category_matches = _apply_category_patterns(zone_text, field_def.category)
                if category_matches:
                    matched_text, pattern_name, captured = category_matches[0]
                    results[field_def.name] = RetrievedMetadata(
                        field_name=field_def.name,
                        value=captured,
                        source_zone=zone,
                        confidence=0.7,  # Category pattern = medium confidence
                        pattern_matched=pattern_name,
                    )
                    match_found = True
                    break

        # If no match found, still add entry with None value
        if not match_found:
            results[field_def.name] = RetrievedMetadata(
                field_name=field_def.name,
                value=None,
                source_zone=zones[0] if zones else SearchZone.ANYWHERE,
                confidence=0.0,
                pattern_matched=None,
            )

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_metadata(
    found: dict[str, RetrievedMetadata],
    schema: "CanonicalSchema",  # Forward reference to avoid circular import
) -> ValidationResult:
    """Phase 2: Check required fields are present.

    Args:
        found: Dict of field_name -> RetrievedMetadata from quick_scan()
        schema: CanonicalSchema with metadata field definitions

    Returns:
        ValidationResult with passed flag and list of missing required fields.
    """
    missing: list[str] = []

    for field_def in schema.metadata:
        if field_def.required:
            metadata = found.get(field_def.name)
            if metadata is None or metadata.value is None:
                missing.append(field_def.name)

    return ValidationResult(
        passed=len(missing) == 0,
        found={k: v for k, v in found.items() if v.value is not None},
        missing=missing,
    )


# ---------------------------------------------------------------------------
# Fallback handling
# ---------------------------------------------------------------------------


def apply_fallbacks(
    validation: ValidationResult,
    schema: "CanonicalSchema",
) -> tuple[dict[str, RetrievedMetadata], list[str]]:
    """Phase 4: Apply fallback strategies for missing fields.

    Args:
        validation: ValidationResult from validate_metadata()
        schema: CanonicalSchema with metadata field definitions

    Returns:
        Tuple of (updated metadata dict, list of warning messages).
    """
    result = dict(validation.found)
    warnings: list[str] = []

    # Build lookup for field definitions
    field_defs = {f.name: f for f in schema.metadata}

    for field_name in validation.missing:
        field_def = field_defs.get(field_name)
        if not field_def:
            continue

        fallback = field_def.fallback

        if fallback == FallbackStrategy.DEFAULT:
            if field_def.default is not None:
                result[field_name] = RetrievedMetadata(
                    field_name=field_name,
                    value=str(field_def.default),
                    source_zone=SearchZone.ANYWHERE,
                    confidence=0.5,
                    pattern_matched=None,
                )
                warnings.append(f"Using default value for {field_name}: {field_def.default}")
            else:
                warnings.append(f"No default value for required field: {field_name}")

        elif fallback == FallbackStrategy.FLAG:
            result[field_name] = RetrievedMetadata(
                field_name=field_name,
                value=None,
                source_zone=SearchZone.ANYWHERE,
                confidence=0.0,
                pattern_matched=None,
            )
            warnings.append(f"Required field missing (flagged): {field_name}")

        elif fallback == FallbackStrategy.PROMPT:
            # Caller handles prompting
            warnings.append(f"Required field needs user input: {field_name}")

        elif fallback == FallbackStrategy.INFER:
            # Future: LLM-based inference
            warnings.append(f"Inference not implemented for: {field_name}")

    return result, warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def search_and_extract(
    pdf_path: str | Path,
    schema: "CanonicalSchema",
    *,
    model: str = "openai/gpt-4o",
    skip_validation: bool = False,
) -> SearchExtractResult:
    """Main entry point: Quick scan → Validate → Deep extract.

    Performs fast metadata extraction using regex patterns, validates required
    fields, applies fallback strategies, then runs LLM-based table extraction.

    Args:
        pdf_path: Path to PDF file
        schema: CanonicalSchema with columns and optional metadata definitions
        model: LLM model for table interpretation
        skip_validation: If True, proceed with extraction even if metadata is missing

    Returns:
        SearchExtractResult with metadata, tables, validation, and warnings.
    """
    from docpact.compress import compress_spatial_text
    from docpact.interpret import interpret_table

    warnings: list[str] = []

    # Phase 1: Quick scan for metadata
    if schema.metadata:
        found = quick_scan(pdf_path, schema.metadata)
    else:
        found = {}

    # Phase 2: Validate metadata
    validation = validate_metadata(found, schema)

    if not validation.passed:
        log.warning(
            "Metadata validation failed: missing %s",
            ", ".join(validation.missing),
        )

    # Phase 3: Apply fallbacks if needed
    if not validation.passed:
        found, fallback_warnings = apply_fallbacks(validation, schema)
        warnings.extend(fallback_warnings)

    # Phase 4: Deep extraction (unless validation failed and skip not set)
    tables: dict[int, Any] = {}

    if validation.passed or skip_validation:
        # Compress and interpret
        compressed = compress_spatial_text(pdf_path)
        tables = interpret_table(compressed, schema, model=model, pdf_path=pdf_path)
    else:
        warnings.append("Skipping table extraction due to missing required metadata")

    return SearchExtractResult(
        metadata=found,
        tables=tables,
        validation=validation,
        warnings=warnings,
    )
