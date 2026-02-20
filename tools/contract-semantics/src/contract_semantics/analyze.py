"""Document structural profiling for contract skeleton generation.

Analyzes PDF and DOCX documents to extract structural signals — column names,
data types, table layouts, section labels, temporal patterns, unit annotations —
that feed into the recommendation engine for draft contract generation.

Core helpers (type detection, header parsing, similarity, unit patterns) are
inlined from docpact.heuristics and docpact.classify so that this module is
fully testable without triggering docpact's BAML import chain.  Heavy docpact
imports (compression, docx extraction, pivot detection) are lazy — only loaded
when ``profile_document()`` is called with a real file.

Public functions:
    profile_document(doc_path) → DocumentProfile
    merge_profiles(profiles) → MultiDocumentProfile
    format_analysis_report(profile) → str
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """Structural profile of a single table column."""

    header_text: str
    inferred_type: str  # "string", "int", "float", "date", "enum"
    unique_count: int = 0
    total_count: int = 0
    sample_values: list[str] = field(default_factory=list)
    unit_annotations: list[str] = field(default_factory=list)
    year_detected: bool = False


@dataclass
class TableProfile:
    """Structural profile of a single table."""

    source_document: str
    table_index: int
    title: str | None = None
    layout: str = "flat"  # "flat", "transposed", "pivoted"
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    section_labels: list[str] = field(default_factory=list)
    pivot_groups: list[str] = field(default_factory=list)
    header_tokens: set[str] = field(default_factory=set)
    raw_markdown: str = ""


@dataclass
class MetadataCandidate:
    """A potential metadata value found in non-table text."""

    text: str
    pattern_name: str
    captured_value: str
    location: str = ""  # e.g. "page 1, heading"


@dataclass
class DocumentProfile:
    """Complete structural profile of a single document."""

    doc_path: str
    doc_format: str  # "pdf" or "docx"
    tables: list[TableProfile] = field(default_factory=list)
    temporal_candidates: list[MetadataCandidate] = field(default_factory=list)
    unit_candidates: list[MetadataCandidate] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)


@dataclass
class AlignedColumn:
    """A column aligned across multiple documents.

    Collects all header variants for the same structural column.
    """

    canonical_header: str  # Most common header text
    all_headers: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # doc paths
    inferred_type: str = "string"
    unit_annotations: list[str] = field(default_factory=list)
    year_detected: bool = False


@dataclass
class TableGroup:
    """A group of structurally similar tables across documents."""

    group_id: int
    tables: list[TableProfile] = field(default_factory=list)
    aligned_columns: list[AlignedColumn] = field(default_factory=list)
    common_tokens: set[str] = field(default_factory=set)
    all_section_labels: list[str] = field(default_factory=list)
    layout: str = "flat"


@dataclass
class MultiDocumentProfile:
    """Merged structural profile across multiple documents."""

    doc_paths: list[str] = field(default_factory=list)
    document_profiles: list[DocumentProfile] = field(default_factory=list)
    table_groups: list[TableGroup] = field(default_factory=list)
    all_temporal_candidates: list[MetadataCandidate] = field(default_factory=list)
    all_unit_candidates: list[MetadataCandidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inlined helpers from docpact.heuristics and docpact.classify
#
# These are small, stable functions inlined here so that analyze.py is fully
# self-contained and testable without triggering docpact's __init__.py
# (which eagerly imports serialize → interpret → baml_client).
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_NUM_RE = re.compile(r"^\d[\d\s,.]*$")
_PUNCT_STRIP = string.punctuation + "–—\u2018\u2019\u201c\u201d"


class CellType(Enum):
    """Cell content classification (inlined from docpact.heuristics)."""
    DATE = "date"
    NUMBER = "number"
    ENUM = "enum"
    STRING = "string"


_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{4}-\d{2}$"),
    re.compile(r"^\d{4}$"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$"),
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$"),
    re.compile(r"^\d{1,2}:\d{2}\s*[AaPp][Mm]$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}"),
    re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$"),
    re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$"),
    re.compile(r"^[A-Za-z]{3,9}\s+\d{4}$"),
]

_NUMBER_PATTERN = re.compile(
    r"^[($€£¥+-]?\s*[\d,.\s]+\s*[)%]?$"
)

_UNIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\(in\s+(millions?|thousands?|billions?)\)", re.IGNORECASE), "scale"),
    (re.compile(r"\b(millions?|thousands?|billions?)\s+of\s+(?:dollars|euros|pounds)", re.IGNORECASE), "scale"),
    (re.compile(r"(?:USD|EUR|GBP)\s+(000s?|MM|M|K|B)", re.IGNORECASE), "scale"),
    (re.compile(r"\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|INR)\b"), "currency_code"),
    (re.compile(r"[\$€£¥]"), "currency_symbol"),
    (re.compile(r"\b(percentage|percent|%)\b", re.IGNORECASE), "percentage"),
    (re.compile(r"\b(metric\s+tons?|tonnes?|MT|kg|lbs?)\b", re.IGNORECASE), "unit"),
    (re.compile(r"\b(shares?|units?|contracts?|lots?)\b", re.IGNORECASE), "unit"),
]

_TEMPORAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[Aa]s\s+of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"), "as_of_date"),
    (re.compile(r"[Aa]s\s+of\s+(\d{1,2}/\d{1,2}/\d{2,4})"), "as_of_date"),
    (re.compile(r"[Aa]s\s+of\s+(\d{4}-\d{2}-\d{2})"), "as_of_date"),
    (re.compile(r"[Ff]or\s+the\s+(?:year|period|quarter)\s+ended?\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"), "period_end"),
    (re.compile(r"[Ff]or\s+the\s+(?:year|period|quarter)\s+ended?\s+(\d{1,2}/\d{1,2}/\d{2,4})"), "period_end"),
    (re.compile(r"\b(Q[1-4])\s*(?:FY)?(\d{2,4})"), "quarter"),
    (re.compile(r"\b(1st|2nd|3rd|4th)\s+[Qq]uarter\s+(\d{4})"), "quarter"),
    (re.compile(r"\bFY\s*(\d{2,4})"), "fiscal_year"),
    (re.compile(r"\b[Ff]iscal\s+[Yy]ear\s+(\d{4})"), "fiscal_year"),
    (re.compile(r"\b([A-Za-z]+)\s+(\d{4})\b"), "year_month"),
    (re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})\s*[-–—to]+\s*(\d{1,2}/\d{1,2}/\d{2,4})"), "date_range"),
    (re.compile(r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s*[-–—to]+\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})"), "date_range"),
]


def _detect_cell_type(text: str | None) -> CellType:
    """Classify a cell as DATE, NUMBER, or STRING."""
    if text is None:
        return CellType.STRING
    text = str(text).strip()
    if not text:
        return CellType.STRING
    for pattern in _DATE_PATTERNS:
        if pattern.match(text):
            return CellType.DATE
    if _NUMBER_PATTERN.match(text):
        cleaned = re.sub(r"[($€£¥+\-,.\s)%]", "", text)
        if cleaned.isdigit():
            return CellType.NUMBER
    return CellType.STRING


def _detect_column_types(grid: list[list[str]]) -> list[CellType]:
    """Detect predominant type per column."""
    if not grid:
        return []
    num_cols = len(grid[0]) if grid else 0
    if not grid or num_cols == 0:
        return [CellType.STRING] * num_cols

    col_values: list[list[str]] = [[] for _ in range(num_cols)]
    col_types: list[Counter[CellType]] = [Counter() for _ in range(num_cols)]
    for row in grid:
        for ci, cell in enumerate(row):
            if ci >= num_cols:
                break
            cell = cell.strip() if isinstance(cell, str) else str(cell).strip()
            if cell:
                col_values[ci].append(cell)
                col_types[ci][_detect_cell_type(cell)] += 1

    result: list[CellType] = []
    for ci in range(num_cols):
        values = col_values[ci]
        type_counts = col_types[ci]
        if not values:
            result.append(CellType.STRING)
            continue
        unique_values = set(values)
        unique_ratio = len(unique_values) / len(values) if values else 1.0
        is_enum = (len(unique_values) <= 5 or unique_ratio <= 0.1) and len(values) >= 3
        if is_enum and type_counts.get(CellType.STRING, 0) > 0:
            result.append(CellType.ENUM)
        elif type_counts:
            result.append(type_counts.most_common(1)[0][0])
        else:
            result.append(CellType.STRING)
    return result


def _parse_pipe_header(markdown: str) -> list[str]:
    """Extract column names from the first pipe-delimited header line."""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "|" not in stripped:
            continue
        cells = [c.strip() for c in stripped.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if not cells:
            continue
        if all(re.fullmatch(r"-{2,}", c) or c == "" for c in cells):
            continue
        return cells
    return []


def _tokenize_header_text(header_text: str) -> set[str]:
    """Extract lowercased tokens from header text, stripping years and numbers."""
    text = header_text.lower()
    for ch in _PUNCT_STRIP:
        if ch in text:
            text = text.replace(ch, " ")
    tokens: set[str] = set()
    for word in text.split():
        if not word or _YEAR_RE.fullmatch(word) or _NUM_RE.fullmatch(word):
            continue
        tokens.add(word)
    return tokens


def _compute_similarity(
    table_cols: int,
    table_tokens: set[str],
    profile_cols: float,
    profile_tokens: set[str],
) -> float:
    """50% column-count ratio + 50% header-token Jaccard."""
    col_sim = (
        min(table_cols, profile_cols) / max(table_cols, profile_cols)
        if profile_cols
        else 0.0
    )
    union = table_tokens | profile_tokens
    jaccard = len(table_tokens & profile_tokens) / len(union) if union else 0.0
    return 0.5 * col_sim + 0.5 * jaccard


def _detect_temporal_patterns(text: str) -> list[tuple[str, str, str]]:
    """Detect temporal patterns in text."""
    results: list[tuple[str, str, str]] = []
    for pattern, name in _TEMPORAL_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            groups = match.groups()
            captured = " ".join(g for g in groups if g)
            results.append((matched_text, name, captured))
    return results


def _detect_unit_patterns(text: str) -> list[tuple[str, str, str]]:
    """Detect unit and currency patterns in text."""
    results: list[tuple[str, str, str]] = []
    for pattern, name in _UNIT_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            captured = match.group(1) if match.lastindex else matched_text
            results.append((matched_text, name, captured))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_column_type(values: list[str], cell_type_str: str) -> str:
    """Map CellType enum name to contract type string."""
    mapping = {
        "date": "string",
        "number": "float",
        "enum": "string",
        "string": "string",
    }
    base = mapping.get(cell_type_str.lower(), "string")

    if base == "float" and values:
        all_integer = True
        for v in values:
            cleaned = re.sub(r"[($€£¥+\-,.\s)%]", "", v.strip())
            if not cleaned:
                continue
            stripped = v.strip().replace(",", "").replace(" ", "")
            if "." in stripped:
                parts = stripped.rstrip("%)").split(".")
                if len(parts) == 2 and parts[1] and parts[1] != "0" * len(parts[1]):
                    all_integer = False
                    break
        if all_integer:
            base = "int"

    return base


def _extract_unit_from_header(header: str) -> list[str]:
    """Extract unit annotations from a header string."""
    units: list[str] = []

    for pattern, _name in _UNIT_PATTERNS:
        for match in pattern.finditer(header):
            captured = match.group(1) if match.lastindex else match.group(0)
            if captured not in units:
                units.append(captured)

    if "," in header:
        parts = header.rsplit(",", 1)
        if len(parts) == 2 and len(parts[1].strip()) < 20:
            suffix = parts[1].strip()
            if suffix not in units:
                units.append(suffix)

    paren_match = re.search(r"\(([^)]+)\)", header)
    if paren_match:
        candidate = paren_match.group(1).strip()
        if len(candidate) < 20 and candidate not in units:
            units.append(candidate)

    return units


def _detect_table_layout(
    markdown: str,
    column_names: list[str],
    data: list[list[str]],
) -> tuple[str, list[str]]:
    """Detect table layout: flat, transposed, or pivoted.

    Returns (layout_name, pivot_group_prefixes).
    Transposed detection is self-contained; pivot detection requires docpact
    and gracefully degrades to "flat" if unavailable.
    """
    col_count = len(column_names)

    # Check for transposed: 2-5 columns, first column is all strings (labels)
    if 2 <= col_count <= 5 and data:
        first_col_types = [_detect_cell_type(row[0]) for row in data if row]
        all_string = all(t == CellType.STRING for t in first_col_types)
        other_have_numbers = False
        for row in data:
            for cell in row[1:]:
                if _detect_cell_type(cell) == CellType.NUMBER:
                    other_have_numbers = True
                    break
            if other_have_numbers:
                break
        if all_string and other_have_numbers and len(data) >= 3:
            return "transposed", []

    # Check for pivoted using docpact's detect_pivot (lazy import)
    try:
        from docpact.unpivot import detect_pivot
        detection = detect_pivot(markdown)
        if detection.is_pivoted:
            prefixes = [g.prefix for g in detection.groups]
            return "pivoted", prefixes
    except Exception:
        pass

    return "flat", []


def _extract_section_labels(markdown: str) -> list[str]:
    """Extract section labels from markdown text."""
    labels: list[str] = []
    bold_re = re.compile(r"^\*\*(.+?)\*\*\s*$")
    heading_re = re.compile(r"^##\s+(.+)$")

    in_table = False
    for line in markdown.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped and not stripped.startswith("#"):
            in_table = True
            continue
        m = bold_re.match(stripped)
        if m and in_table:
            label = m.group(1).strip()
            if label and label not in labels:
                labels.append(label)
            continue
        m = heading_re.match(stripped)
        if m:
            label = m.group(1).strip()
            if label and label not in labels:
                labels.append(label)
            continue
        if in_table and not stripped.startswith("|"):
            if len(stripped) < 60 and not stripped.startswith("("):
                if stripped not in labels:
                    labels.append(stripped)

    return labels


def _profile_table_from_structured(
    table: object,
    doc_path: str,
    table_idx: int,
) -> TableProfile:
    """Build a TableProfile from a StructuredTable (PDF path).

    Requires docpact at runtime (called only from _profile_pdf).
    """
    column_names = table.column_names  # type: ignore[attr-defined]
    data = table.data  # type: ignore[attr-defined]
    metadata = table.metadata  # type: ignore[attr-defined]

    col_types = _detect_column_types(data) if data else []

    col_profiles: list[ColumnProfile] = []
    for ci, header in enumerate(column_names):
        values = [row[ci] for row in data if ci < len(row) and row[ci].strip()]
        cell_type = col_types[ci] if ci < len(col_types) else CellType.STRING
        inferred = _infer_column_type(values, cell_type.value)
        unique_vals = set(values)
        col_profiles.append(
            ColumnProfile(
                header_text=header,
                inferred_type=inferred,
                unique_count=len(unique_vals),
                total_count=len(values),
                sample_values=sorted(unique_vals)[:10],
                unit_annotations=_extract_unit_from_header(header),
                year_detected=bool(_YEAR_RE.fullmatch(header.strip())),
            )
        )

    lines = ["| " + " | ".join(column_names) + " |"]
    lines.append("| " + " | ".join("---" for _ in column_names) + " |")
    for row in data[:50]:
        padded = row + [""] * (len(column_names) - len(row))
        lines.append("| " + " | ".join(padded[: len(column_names)]) + " |")
    markdown = "\n".join(lines)

    layout, pivot_groups = _detect_table_layout(markdown, column_names, data)
    section_labels = _extract_section_labels(markdown)

    title = None
    if metadata and hasattr(metadata, "section_label"):
        title = metadata.section_label
    elif isinstance(metadata, dict):
        title = metadata.get("title") or metadata.get("section_label")

    return TableProfile(
        source_document=str(doc_path),
        table_index=table_idx,
        title=title,
        layout=layout,
        column_profiles=col_profiles,
        row_count=len(data),
        col_count=len(column_names),
        section_labels=section_labels,
        pivot_groups=pivot_groups,
        header_tokens=_tokenize_header_text(" ".join(column_names)),
        raw_markdown=markdown,
    )


def _profile_table_from_compressed(
    markdown: str,
    meta: dict,
    doc_path: str,
    table_idx: int,
) -> TableProfile:
    """Build a TableProfile from compressed pipe-table markdown (DOCX path).

    Fully self-contained (no docpact imports needed).
    """
    column_names = _parse_pipe_header(markdown)

    data_rows: list[list[str]] = []
    past_separator = False
    for line in markdown.split("\n"):
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if not cells:
            continue
        if all(re.fullmatch(r"-{2,}", c) or c == "" for c in cells):
            past_separator = True
            continue
        if past_separator:
            data_rows.append(cells)

    col_types = _detect_column_types(data_rows) if data_rows else []

    col_profiles: list[ColumnProfile] = []
    for ci, header in enumerate(column_names):
        values = [
            row[ci] for row in data_rows if ci < len(row) and row[ci].strip()
        ]
        cell_type = col_types[ci] if ci < len(col_types) else CellType.STRING
        inferred = _infer_column_type(values, cell_type.value)
        unique_vals = set(values)
        col_profiles.append(
            ColumnProfile(
                header_text=header,
                inferred_type=inferred,
                unique_count=len(unique_vals),
                total_count=len(values),
                sample_values=sorted(unique_vals)[:10],
                unit_annotations=_extract_unit_from_header(header),
                year_detected=bool(_YEAR_RE.fullmatch(header.strip())),
            )
        )

    layout, pivot_groups = _detect_table_layout(markdown, column_names, data_rows)
    section_labels = _extract_section_labels(markdown)

    return TableProfile(
        source_document=str(doc_path),
        table_index=table_idx,
        title=meta.get("title"),
        layout=layout,
        column_profiles=col_profiles,
        row_count=meta.get("row_count", len(data_rows)),
        col_count=meta.get("col_count", len(column_names)),
        section_labels=section_labels,
        pivot_groups=pivot_groups,
        header_tokens=_tokenize_header_text(" ".join(column_names)),
        raw_markdown=markdown,
    )


def _extract_non_table_text(compressed: str) -> str:
    """Extract non-table text from compressed output for metadata scanning."""
    lines: list[str] = []
    for line in compressed.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|"):
            continue
        if re.fullmatch(r"\|[\s\-|]+\|", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_document(doc_path: str | Path) -> DocumentProfile:
    """Profile a single document (PDF or DOCX) for contract skeleton generation.

    Requires docpact to be importable (compression and extraction modules).

    Args:
        doc_path: Path to PDF or DOCX file.

    Returns:
        DocumentProfile with per-table profiles and metadata candidates.
    """
    path = Path(doc_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _profile_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _profile_docx(path)
    else:
        raise ValueError(f"Unsupported document format: {suffix}")


def _profile_pdf(path: Path) -> DocumentProfile:
    """Profile a PDF document (requires docpact)."""
    from docpact.compress import compress_spatial_text, compress_spatial_text_structured

    structured_tables = compress_spatial_text_structured(str(path))

    tables: list[TableProfile] = []
    for idx, st in enumerate(structured_tables):
        tp = _profile_table_from_structured(st, str(path), idx)
        tables.append(tp)

    compressed = compress_spatial_text(str(path))
    non_table = _extract_non_table_text(compressed)

    temporal = [
        MetadataCandidate(text=m, pattern_name=p, captured_value=c)
        for m, p, c in _detect_temporal_patterns(non_table)
    ]
    units = [
        MetadataCandidate(text=m, pattern_name=p, captured_value=c)
        for m, p, c in _detect_unit_patterns(non_table)
    ]
    headings = [
        line.strip()[3:].strip()
        for line in compressed.split("\n")
        if line.strip().startswith("## ")
    ]

    return DocumentProfile(
        doc_path=str(path),
        doc_format="pdf",
        tables=tables,
        temporal_candidates=temporal,
        unit_candidates=units,
        headings=headings,
    )


def _profile_docx(path: Path) -> DocumentProfile:
    """Profile a DOCX document (requires docpact)."""
    from docpact.docx_extractor import compress_docx_tables

    compressed_tables = compress_docx_tables(str(path))

    tables: list[TableProfile] = []
    all_markdown: list[str] = []
    for idx, (markdown, meta) in enumerate(compressed_tables):
        tp = _profile_table_from_compressed(markdown, meta, str(path), idx)
        tables.append(tp)
        all_markdown.append(markdown)

    full_text = "\n".join(all_markdown)
    non_table = _extract_non_table_text(full_text)

    temporal = [
        MetadataCandidate(text=m, pattern_name=p, captured_value=c)
        for m, p, c in _detect_temporal_patterns(non_table)
    ]
    units = [
        MetadataCandidate(text=m, pattern_name=p, captured_value=c)
        for m, p, c in _detect_unit_patterns(non_table)
    ]
    headings = [
        line.strip()[3:].strip()
        for line in full_text.split("\n")
        if line.strip().startswith("## ")
    ]

    return DocumentProfile(
        doc_path=str(path),
        doc_format="docx",
        tables=tables,
        temporal_candidates=temporal,
        unit_candidates=units,
        headings=headings,
    )


def merge_profiles(profiles: list[DocumentProfile]) -> MultiDocumentProfile:
    """Merge multiple document profiles into a consolidated multi-document profile.

    Groups structurally similar tables across documents and aligns columns
    to collect all header variants for each structural position.

    Args:
        profiles: List of DocumentProfile from profile_document().

    Returns:
        MultiDocumentProfile with grouped tables and aligned columns.
    """
    if not profiles:
        return MultiDocumentProfile()

    if len(profiles) == 1:
        groups = []
        for idx, tp in enumerate(profiles[0].tables):
            aligned = _align_single_table(tp)
            groups.append(
                TableGroup(
                    group_id=idx,
                    tables=[tp],
                    aligned_columns=aligned,
                    common_tokens=tp.header_tokens,
                    all_section_labels=list(tp.section_labels),
                    layout=tp.layout,
                )
            )
        return MultiDocumentProfile(
            doc_paths=[profiles[0].doc_path],
            document_profiles=profiles,
            table_groups=groups,
            all_temporal_candidates=list(profiles[0].temporal_candidates),
            all_unit_candidates=list(profiles[0].unit_candidates),
        )

    all_tables: list[TableProfile] = []
    for p in profiles:
        all_tables.extend(p.tables)

    groups = _group_tables(all_tables)

    all_temporal: list[MetadataCandidate] = []
    all_units: list[MetadataCandidate] = []
    for p in profiles:
        all_temporal.extend(p.temporal_candidates)
        all_units.extend(p.unit_candidates)

    return MultiDocumentProfile(
        doc_paths=[p.doc_path for p in profiles],
        document_profiles=profiles,
        table_groups=groups,
        all_temporal_candidates=all_temporal,
        all_unit_candidates=all_units,
    )


# ---------------------------------------------------------------------------
# Table grouping and column alignment
# ---------------------------------------------------------------------------


def _align_single_table(tp: TableProfile) -> list[AlignedColumn]:
    """Create AlignedColumn entries for a single table's columns."""
    return [
        AlignedColumn(
            canonical_header=cp.header_text,
            all_headers=[cp.header_text],
            sources=[tp.source_document],
            inferred_type=cp.inferred_type,
            unit_annotations=list(cp.unit_annotations),
            year_detected=cp.year_detected,
        )
        for cp in tp.column_profiles
    ]


def _group_tables(tables: list[TableProfile]) -> list[TableGroup]:
    """Group tables by structural similarity (token Jaccard + column count)."""
    threshold = 0.3
    groups: list[TableGroup] = []

    for tp in tables:
        best_group: TableGroup | None = None
        best_sim = 0.0

        for g in groups:
            avg_cols = sum(t.col_count for t in g.tables) / len(g.tables)
            sim = _compute_similarity(
                tp.col_count, tp.header_tokens, avg_cols, g.common_tokens
            )
            if sim > best_sim:
                best_sim = sim
                best_group = g

        if best_group is not None and best_sim >= threshold:
            best_group.tables.append(tp)
            best_group.common_tokens = best_group.common_tokens & tp.header_tokens
            for label in tp.section_labels:
                if label not in best_group.all_section_labels:
                    best_group.all_section_labels.append(label)
            if tp.layout != "flat":
                best_group.layout = tp.layout
        else:
            groups.append(
                TableGroup(
                    group_id=len(groups),
                    tables=[tp],
                    aligned_columns=[],
                    common_tokens=set(tp.header_tokens),
                    all_section_labels=list(tp.section_labels),
                    layout=tp.layout,
                )
            )

    for g in groups:
        g.aligned_columns = _align_columns_in_group(g.tables)

    return groups


def _align_columns_in_group(tables: list[TableProfile]) -> list[AlignedColumn]:
    """Align columns across tables in a group by position + token overlap."""
    if not tables:
        return []
    if len(tables) == 1:
        return _align_single_table(tables[0])

    ref_table = max(tables, key=lambda t: t.col_count)
    aligned: list[AlignedColumn] = _align_single_table(ref_table)

    for tp in tables:
        if tp is ref_table:
            continue
        for ci, cp in enumerate(tp.column_profiles):
            if tp.col_count == ref_table.col_count and ci < len(aligned):
                ac = aligned[ci]
                if cp.header_text not in ac.all_headers:
                    ac.all_headers.append(cp.header_text)
                if tp.source_document not in ac.sources:
                    ac.sources.append(tp.source_document)
                if cp.year_detected:
                    ac.year_detected = True
                for u in cp.unit_annotations:
                    if u not in ac.unit_annotations:
                        ac.unit_annotations.append(u)
                continue

            cp_tokens = set(cp.header_text.lower().split())
            best_idx = -1
            best_overlap = 0
            for ai, ac in enumerate(aligned):
                ac_tokens = set(ac.canonical_header.lower().split())
                overlap = len(cp_tokens & ac_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = ai

            if best_idx >= 0 and best_overlap > 0:
                ac = aligned[best_idx]
                if cp.header_text not in ac.all_headers:
                    ac.all_headers.append(cp.header_text)
                if tp.source_document not in ac.sources:
                    ac.sources.append(tp.source_document)
                if cp.year_detected:
                    ac.year_detected = True
                for u in cp.unit_annotations:
                    if u not in ac.unit_annotations:
                        ac.unit_annotations.append(u)
            else:
                aligned.append(
                    AlignedColumn(
                        canonical_header=cp.header_text,
                        all_headers=[cp.header_text],
                        sources=[tp.source_document],
                        inferred_type=cp.inferred_type,
                        unit_annotations=list(cp.unit_annotations),
                        year_detected=cp.year_detected,
                    )
                )

    return aligned


# ---------------------------------------------------------------------------
# Analysis report formatting
# ---------------------------------------------------------------------------


def format_analysis_report(profile: DocumentProfile | MultiDocumentProfile) -> str:
    """Format a human-readable analysis report from a profile."""
    lines: list[str] = []

    if isinstance(profile, MultiDocumentProfile):
        lines.append("Document Analysis Report")
        lines.append("=" * 60)
        lines.append(f"Documents analyzed: {len(profile.doc_paths)}")
        for dp in profile.doc_paths:
            lines.append(f"  - {dp}")
        lines.append("")

        total_tables = sum(len(dp.tables) for dp in profile.document_profiles)
        flat = sum(1 for dp in profile.document_profiles for t in dp.tables if t.layout == "flat")
        pivoted = sum(1 for dp in profile.document_profiles for t in dp.tables if t.layout == "pivoted")
        transposed = sum(1 for dp in profile.document_profiles for t in dp.tables if t.layout == "transposed")

        layout_parts = []
        if flat:
            layout_parts.append(f"{flat} flat")
        if pivoted:
            layout_parts.append(f"{pivoted} pivoted")
        if transposed:
            layout_parts.append(f"{transposed} transposed")
        layout_str = ", ".join(layout_parts) if layout_parts else "none"

        lines.append(f"Tables detected: {total_tables} ({layout_str})")
        lines.append(f"Table groups: {len(profile.table_groups)}")
        lines.append("")

        for g in profile.table_groups:
            lines.append(f"Group {g.group_id} ({len(g.tables)} tables, layout: {g.layout}):")
            if g.common_tokens:
                lines.append(f"  Common tokens: {', '.join(sorted(g.common_tokens)[:10])}")
            lines.append(f"  Aligned columns: {len(g.aligned_columns)}")
            for ac in g.aligned_columns:
                variants = len(ac.all_headers)
                lines.append(f"    - {ac.canonical_header} ({ac.inferred_type}, {variants} variant(s))")
                if variants > 1:
                    for h in ac.all_headers:
                        lines.append(f"        alias: {h!r}")
            if g.all_section_labels:
                lines.append(f"  Section labels: {', '.join(g.all_section_labels[:10])}")
            lines.append("")

        if profile.all_temporal_candidates:
            lines.append("Metadata Candidates (temporal):")
            seen: set[tuple[str, str]] = set()
            for mc in profile.all_temporal_candidates:
                key = (mc.pattern_name, mc.captured_value)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  - {mc.text!r} ({mc.pattern_name}: {mc.captured_value})")
            lines.append("")

        if profile.all_unit_candidates:
            lines.append("Metadata Candidates (units):")
            seen_u: set[tuple[str, str]] = set()
            for mc in profile.all_unit_candidates:
                key = (mc.pattern_name, mc.captured_value)
                if key not in seen_u:
                    seen_u.add(key)
                    lines.append(f"  - {mc.text!r} ({mc.pattern_name}: {mc.captured_value})")
            lines.append("")

    else:
        lines.append("Document Analysis Report")
        lines.append("=" * 60)
        lines.append(f"Document: {profile.doc_path}")
        lines.append(f"Format: {profile.doc_format.upper()}")
        lines.append("")

        flat = sum(1 for t in profile.tables if t.layout == "flat")
        pivoted = sum(1 for t in profile.tables if t.layout == "pivoted")
        transposed = sum(1 for t in profile.tables if t.layout == "transposed")

        layout_parts = []
        if flat:
            layout_parts.append(f"{flat} flat")
        if pivoted:
            layout_parts.append(f"{pivoted} pivoted")
        if transposed:
            layout_parts.append(f"{transposed} transposed")
        layout_str = ", ".join(layout_parts) if layout_parts else "none"

        lines.append(f"Tables detected: {len(profile.tables)} ({layout_str})")
        if profile.headings:
            lines.append(f"Headings: {len(profile.headings)}")
        lines.append("")

        for tp in profile.tables:
            title_str = f" — {tp.title}" if tp.title else ""
            lines.append(
                f"Table {tp.table_index}{title_str} "
                f"({tp.col_count} cols, {tp.row_count} rows, {tp.layout})"
            )
            for cp in tp.column_profiles:
                parts = [cp.header_text, cp.inferred_type]
                if cp.unique_count:
                    parts.append(f"{cp.unique_count} unique")
                if cp.unit_annotations:
                    parts.append(f"unit: {cp.unit_annotations[0]}")
                if cp.year_detected:
                    parts.append("YEAR")
                lines.append(f"    {' | '.join(parts)}")
            if tp.section_labels:
                labels = ", ".join(tp.section_labels[:5])
                more = f" (+{len(tp.section_labels) - 5} more)" if len(tp.section_labels) > 5 else ""
                lines.append(f"  Section labels: {labels}{more}")
            if tp.pivot_groups:
                lines.append(f"  Pivot groups: {', '.join(tp.pivot_groups)}")
            lines.append("")

        if profile.temporal_candidates:
            lines.append("Metadata Candidates (temporal):")
            for mc in profile.temporal_candidates:
                lines.append(f"  - {mc.text!r} ({mc.pattern_name}: {mc.captured_value})")
            lines.append("")

        if profile.unit_candidates:
            lines.append("Metadata Candidates (units):")
            for mc in profile.unit_candidates:
                lines.append(f"  - {mc.text!r} ({mc.pattern_name}: {mc.captured_value})")
            lines.append("")

    return "\n".join(lines)
