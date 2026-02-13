"""Table interpretation: extract structured tabular data from compressed PDF text
and map it to a user-defined canonical schema.

Provides both sync and async APIs. Multi-page compressed text (pages joined by
form-feed ``\\f``) is automatically split and processed page-by-page.

``interpret_table()`` returns ``dict[int, MappedTable]`` keyed by 1-indexed page
number — each page gets its own records, unmapped columns, mapping notes, and
metadata.  Records contain only canonical schema fields.

Sync usage::

    from pdf_ocr import compress_spatial_text, interpret_table, CanonicalSchema, ColumnDef, to_records

    compressed = compress_spatial_text("inputs/example.pdf")
    schema = CanonicalSchema(columns=[
        ColumnDef("port", "string", "Loading port name", aliases=["Port"]),
        ColumnDef("vessel_name", "string", "Name of the vessel", aliases=["Ship Name"]),
    ])
    result = interpret_table(compressed, schema, model="openai/gpt-4o")
    for r in to_records(result):        # flat list across all pages
        print(r)
    for page, mt in result.items():     # per-page access
        print(f"Page {page}: {len(mt.records)} records")

Async usage (explicit parallel control across pre-split tables)::

    import asyncio
    from pdf_ocr.interpret import interpret_tables_async, CanonicalSchema, ColumnDef

    tables = await interpret_tables_async([page1, page2, page3], schema, model="openai/gpt-4o")
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from baml_client import types as baml_types
from baml_client.sync_client import b as b_sync
from baml_client.async_client import b as b_async
from baml_client.type_builder import TypeBuilder
from baml_py import Image

from pdf_ocr.heuristics import MetadataFieldDef
from pdf_ocr.normalize import normalize_pipe_table
from pdf_ocr.spatial_text import _open_pdf

DEFAULT_MODEL = "openai/gpt-4o"

log = logging.getLogger(__name__)


# ─── Python-side schema definition ───────────────────────────────────────────


@dataclass
class ColumnDef:
    """Definition of a single canonical column.

    Attributes:
        name: Canonical column name.
        type: Expected type - "string", "int", "float", "bool", "date".
        description: What this column represents.
        aliases: Alternative names this column may appear as in source tables.
        format: Output format specification (optional).
            - Dates: YYYY-MM-DD, YYYY-MM, YYYY, DD/MM/YYYY, MM/DD/YYYY, MMM YYYY,
              MMMM YYYY, YYYY-MM-DD HH:mm, YYYY-MM-DD HH:mm:ss, HH:mm, hh:mm A, etc.
            - Numbers: # (plain), #.## (decimals), #,### (thousands), #,###.## (both),
              +# (explicit sign), #% (percentage), (#) (negative in parentheses).
            - Strings: uppercase, lowercase, titlecase, capitalize, camelCase,
              PascalCase, snake_case, SCREAMING_SNAKE_CASE, kebab-case, trim.
    """

    name: str
    type: str  # "string", "int", "float", "bool", "date"
    description: str
    aliases: list[str] = field(default_factory=list)
    format: str | None = None


@dataclass
class CanonicalSchema:
    """User-defined canonical schema for table mapping.

    Attributes:
        columns: Column definitions for table data
        metadata: Optional metadata field definitions for document-level extraction
        description: Optional schema description
    """

    columns: list[ColumnDef]
    metadata: list[MetadataFieldDef] = field(default_factory=list)
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> CanonicalSchema:
        """Build a CanonicalSchema from a plain dict / parsed JSON.

        Expected format::

            {
                "description": "optional schema description",
                "columns": [
                    {
                        "name": "port",
                        "type": "string",
                        "description": "Loading port name",
                        "aliases": ["Port"]
                    },
                    ...
                ],
                "metadata": [
                    {
                        "name": "publication_date",
                        "category": "temporal",
                        "required": true,
                        "zones": ["title_page"],
                        "patterns": ["As of\\\\s+(.+?\\\\d{4})"]
                    },
                    ...
                ]
            }
        """
        from pdf_ocr.heuristics import (
            FallbackStrategy,
            MetadataCategory,
            MetadataFieldDef,
            SearchZone,
        )

        columns = [
            ColumnDef(
                name=c["name"],
                type=c.get("type", "string"),
                description=c.get("description", ""),
                aliases=c.get("aliases", []),
                format=c.get("format"),
            )
            for c in data["columns"]
        ]

        metadata_list: list[MetadataFieldDef] = []
        for m in data.get("metadata", []):
            metadata_list.append(
                MetadataFieldDef(
                    name=m["name"],
                    category=MetadataCategory(m.get("category", "temporal")),
                    required=m.get("required", False),
                    zones=[SearchZone(z) for z in m.get("zones", [])],
                    patterns=m.get("patterns", []),
                    fallback=FallbackStrategy(m.get("fallback", "flag")),
                    default=m.get("default"),
                )
            )

        return cls(
            columns=columns,
            metadata=metadata_list,
            description=data.get("description"),
        )


# ─── TypeBuilder helper ──────────────────────────────────────────────────────

def _build_type_builder(schema: CanonicalSchema) -> TypeBuilder:
    """Construct a BAML TypeBuilder that adds canonical columns as optional
    properties on the dynamic ``MappedRecord`` class.

    Field type factory methods (``string()``, ``int()``, etc.) live on the
    native ``tb._tb`` object, not on the high-level ``FieldType`` class.
    """
    tb = TypeBuilder()
    native = tb._tb  # baml_py.baml_py.TypeBuilder — has .string(), .int(), etc.
    field_type_map = {
        "string": native.string,
        "int": native.int,
        "float": native.float,
        "bool": native.bool,
    }
    record_builder = tb.MappedRecord
    for col in schema.columns:
        ft_fn = field_type_map.get(col.type, native.string)
        record_builder.add_property(col.name, ft_fn().optional())
    return tb


# ─── Conversion helpers ──────────────────────────────────────────────────────


def _to_baml_schema(schema: CanonicalSchema) -> baml_types.CanonicalSchema:
    """Convert Python-side schema dataclasses into BAML Pydantic models."""
    return baml_types.CanonicalSchema(
        columns=[
            baml_types.ColumnDef(
                name=c.name,
                type=c.type,
                description=c.description,
                aliases=c.aliases,
                format=c.format,
            )
            for c in schema.columns
        ],
        description=schema.description,
    )


def _render_page_images(
    pdf_input: str | bytes | Path,
    pages: list[int] | None = None,
    dpi: int = 150,
) -> list[str]:
    """Render PDF pages to base64-encoded PNG strings.

    Args:
        pdf_input: Path to PDF file (str or Path), or raw PDF bytes.
        pages: 0-based page indices. None = all pages.
        dpi: Resolution for rendering (default 150 — good balance of quality/size).

    Returns:
        List of base64-encoded PNG strings, one per page.
    """
    import base64
    import fitz

    doc = _open_pdf(pdf_input)
    indices = pages if pages is not None else list(range(doc.page_count))
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for idx in indices:
        pix = doc[idx].get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        images.append(base64.b64encode(png_bytes).decode("ascii"))
    doc.close()
    return images


@dataclass
class _Batch:
    """A subset of a ParsedTable's data rows for batched step-2 processing."""

    page_index: int  # 0-based page index
    batch_index: int  # 0-based within this page
    data_rows: list[list[str]]
    notes: str | None  # Notes with batch-local row indices


def to_records(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
) -> list[dict]:
    """Convert interpretation result into a flat list of plain dicts.

    Accepts either:
    - ``dict[int, MappedTable]`` (page-keyed result from ``interpret_table``)
    - A single ``MappedTable`` (from lower-level functions)

    Records contain only canonical schema columns (no internal metadata).
    """
    if isinstance(result, dict):
        tables = [result[k] for k in sorted(result)]
    else:
        tables = [result]
    records = []
    for mt in tables:
        for rec in mt.records:
            records.append(rec.model_dump())
    return records


def to_records_by_page(
    result: dict[int, baml_types.MappedTable] | baml_types.MappedTable,
) -> dict[int, list[dict]]:
    """Convert interpretation result into ``{page_number: [records]}``.

    Accepts either:
    - ``dict[int, MappedTable]`` (page-keyed result from ``interpret_table``)
    - A single ``MappedTable`` (grouped under page ``1``)

    Page numbers are 1-indexed.  Records contain only canonical schema columns.
    """
    if isinstance(result, dict):
        return {
            page: [rec.model_dump() for rec in mt.records]
            for page, mt in sorted(result.items())
        }
    return {1: [rec.model_dump() for rec in result.records]}


# ─── Async runner helper ─────────────────────────────────────────────────────


def _run_async(coro):
    """Run a coroutine from sync code, whether or not an event loop is running.

    * **No running loop** (scripts, CLI): uses ``asyncio.run()``.
    * **Running loop** (Jupyter, async frameworks): schedules via
      ``loop.create_task()`` and blocks with ``concurrent.futures``.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — normal script context
        return asyncio.run(coro)

    # Already inside a running loop (e.g. Jupyter) — run in a background thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ─── Section boundary detection & validation ─────────────────────────────────


_SECTION_RE = re.compile(
    r"([A-Z][A-Z0-9 /&\-'().]+?)\s*\(rows?\s+(\d+)\s*-\s*(\d+)\)",
)


def _parse_section_boundaries(
    notes: str | None,
) -> list[tuple[str, int, int]] | None:
    """Parse ``"Sections: LABEL (rows 0-15), LABEL2 (rows 16-55)"`` from notes.

    Returns a list of ``(label, start_row, end_row)`` tuples where rows are
    0-based inclusive indices, or ``None`` when no sections are found.
    """
    if not notes:
        return None
    matches = _SECTION_RE.findall(notes)
    if not matches:
        return None
    return [(label.strip(), int(start), int(end)) for label, start, end in matches]


def _detect_sections_from_text(
    compressed_text: str,
) -> list[tuple[str, int]] | None:
    """Count data rows per section by analysing pipe-table structure.

    Walks through the compressed text looking for the repeating pattern::

        SECTION_HEADER          (non-pipe, no tabs)
        sub-headers / blanks
        |---|---|...|           (pipe separator)
        |data row 1|           (data rows — first cell non-empty)
        ...
        ||...|total|...|       (aggregation row — first cell empty)

    Returns ``[(section_label, data_row_count), ...]`` or ``None`` when
    fewer than 2 sections are found.
    """
    lines = compressed_text.split("\n")
    sections: list[tuple[str, int]] = []
    last_header: str | None = None
    in_data = False
    pipe_rows: list[str] = []

    def _flush() -> None:
        nonlocal pipe_rows
        if last_header and pipe_rows:
            n = len(pipe_rows)
            # Last pipe row with empty first cell is aggregation
            if n > 1 and pipe_rows[-1].startswith("||"):
                n -= 1
            if n > 0:
                sections.append((last_header, n))
        pipe_rows = []

    for line in lines:
        s = line.strip()
        if s.startswith("|---"):
            in_data = True
            pipe_rows = []
            continue
        if in_data:
            if s.startswith("|") and s:
                pipe_rows.append(s)
            else:
                _flush()
                in_data = False
                if s and not s.startswith("|") and "\t" not in s:
                    last_header = s
        else:
            if s and not s.startswith("|") and "\t" not in s:
                last_header = s

    if in_data:
        _flush()

    return sections if len(sections) >= 2 else None


def _validate_section_boundaries(
    notes: str | None,
    total_data_rows: int,
    compressed_text: str,
) -> str | None:
    """Cross-reference LLM section boundaries with the compressed text.

    Counts pipe-table data rows per section directly from the text
    (deterministic) and, when the sum matches *total_data_rows*, rebuilds
    the notes with corrected boundaries.  Falls back to the original
    *notes* when validation cannot be performed.
    """
    detected = _detect_sections_from_text(compressed_text)
    if detected is None:
        return notes

    llm_sections = _parse_section_boundaries(notes)

    # Build matched list: (label, row_count) using detected counts
    if llm_sections is not None and len(llm_sections) == len(detected):
        # Same section count — use LLM labels with detected row counts
        matched = [
            (label, det_count)
            for (label, _, _), (_, det_count) in zip(llm_sections, detected)
        ]
    elif llm_sections is not None:
        # Different counts — try ordered label prefix matching
        matched = []
        det_idx = 0
        for label, _s, _e in llm_sections:
            label_up = label.upper().strip()
            found = False
            while det_idx < len(detected):
                det_up = detected[det_idx][0].upper().strip()
                if det_up.startswith(label_up) or label_up.startswith(det_up):
                    matched.append((label, detected[det_idx][1]))
                    det_idx += 1
                    found = True
                    break
                det_idx += 1
            if not found:
                matched.append((label, _e - _s + 1))
    else:
        # LLM produced no sections — use detected labels and counts
        matched = list(detected)

    # Verify sum matches total data rows
    detected_sum = sum(c for _, c in matched)
    if detected_sum != total_data_rows:
        log.debug(
            "Section validation: detected sum=%d != total_data_rows=%d, keeping original",
            detected_sum,
            total_data_rows,
        )
        return notes

    # Rebuild notes with correct boundaries
    parts = []
    start = 0
    for label, count in matched:
        end = start + count - 1
        parts.append(f"{label} (rows {start}-{end})")
        start = end + 1

    section_str = "Sections: " + ", ".join(parts)

    # Preserve non-section content from original notes
    if notes:
        non_section = _SECTION_RE.sub("", notes)
        non_section = re.sub(r"Sections?\s*:\s*", "", non_section)
        non_section = re.sub(r"^[\s,;]+|[\s,;]+$", "", non_section)
        if non_section:
            return f"{section_str}; {non_section}"
    return section_str


def _build_batch_notes(
    sections: list[tuple[str, int, int]] | None,
    batch_start: int,
    batch_end: int,
    original_notes: str | None,
) -> str | None:
    """Rewrite section row indices to be batch-local for rows [batch_start, batch_end).

    Returns adjusted notes string, or ``None`` if there are no relevant sections
    and no non-section notes content.
    """
    if sections is None:
        return original_notes

    # Find sections that overlap with this batch
    relevant: list[str] = []
    for label, sec_start, sec_end in sections:
        # Overlap check: section [sec_start, sec_end] vs batch [batch_start, batch_end)
        overlap_start = max(sec_start, batch_start)
        overlap_end = min(sec_end, batch_end - 1)  # inclusive end
        if overlap_start > overlap_end:
            continue
        # Reindex to batch-local 0-based
        local_start = overlap_start - batch_start
        local_end = overlap_end - batch_start
        relevant.append(f"{label} (rows {local_start}-{local_end})")

    # Preserve any non-section text from original notes
    non_section = ""
    if original_notes:
        non_section = _SECTION_RE.sub("", original_notes)
        # Clean up leftover "Sections:" prefix and separators
        non_section = re.sub(r"Sections?\s*:\s*", "", non_section)
        non_section = re.sub(r"^[\s,;]+|[\s,;]+$", "", non_section)

    parts: list[str] = []
    if relevant:
        parts.append("Sections: " + ", ".join(relevant))
    if non_section:
        parts.append(non_section)
    return "; ".join(parts) if parts else None


def _create_batches(
    parsed_table: baml_types.ParsedTable,
    page_index: int,
    batch_size: int,
) -> list[_Batch]:
    """Split a ParsedTable's data_rows into batches for step-2 processing.

    Strategy:
    - If rows fit in one batch, return a single batch (fast path).
    - If section boundaries are found in notes, use greedy bin-packing of
      consecutive sections — small adjacent sections combine if they fit;
      large sections are split internally at ``batch_size`` boundaries.
    - If no sections are found, use simple fixed-size chunking.
    """
    rows = parsed_table.data_rows
    n = len(rows)
    notes = parsed_table.notes

    if n <= batch_size:
        return [_Batch(page_index=page_index, batch_index=0, data_rows=rows, notes=notes)]

    sections = _parse_section_boundaries(notes)

    if sections is None:
        # Simple fixed-size chunking
        batches: list[_Batch] = []
        for i in range(0, n, batch_size):
            chunk = rows[i : i + batch_size]
            batch_notes = _build_batch_notes(None, i, i + len(chunk), notes)
            batches.append(
                _Batch(
                    page_index=page_index,
                    batch_index=len(batches),
                    data_rows=chunk,
                    notes=batch_notes,
                )
            )
        return batches

    # Validate section coverage: if sections don't start near row 0 or don't cover
    # most of the data, the LLM's section boundaries are probably wrong. Fall back
    # to simple chunking to avoid losing rows.
    first_start = sections[0][1]
    last_end = sections[-1][2]
    # Count how many rows the sections claim to cover
    claimed_coverage = sum(end - start + 1 for _, start, end in sections)
    # Sections are suspicious if they skip the beginning OR cover less than half the data
    if first_start > batch_size // 2 or claimed_coverage < n // 2:
        log.warning(
            "Section boundaries look invalid (first_start=%d, coverage=%d/%d), "
            "falling back to simple chunking",
            first_start,
            claimed_coverage,
            n,
        )
        batches = []
        for i in range(0, n, batch_size):
            chunk = rows[i : i + batch_size]
            batch_notes = _build_batch_notes(None, i, i + len(chunk), notes)
            batches.append(
                _Batch(
                    page_index=page_index,
                    batch_index=len(batches),
                    data_rows=chunk,
                    notes=batch_notes,
                )
            )
        return batches

    # Section-aware greedy bin-packing
    batches = []
    current_rows: list[list[str]] = []
    current_start = 0  # absolute row index where current batch starts

    def _flush() -> None:
        if current_rows:
            batch_notes = _build_batch_notes(
                sections, current_start, current_start + len(current_rows), notes
            )
            batches.append(
                _Batch(
                    page_index=page_index,
                    batch_index=len(batches),
                    data_rows=list(current_rows),
                    notes=batch_notes,
                )
            )

    for label, sec_start, sec_end in sections:
        sec_rows = rows[sec_start : sec_end + 1]  # inclusive end
        sec_len = len(sec_rows)

        if len(current_rows) + sec_len <= batch_size:
            # Section fits in current batch
            current_rows.extend(sec_rows)
        else:
            if current_rows:
                # Flush current batch before starting this section
                _flush()
                current_rows = []
                current_start = sec_start

            if sec_len <= batch_size:
                # Section fits in a fresh batch
                current_rows = list(sec_rows)
                current_start = sec_start
            else:
                # Large section — split internally
                current_start = sec_start
                for j in range(0, sec_len, batch_size):
                    chunk = sec_rows[j : j + batch_size]
                    current_rows = list(chunk)
                    _flush()
                    current_rows = []
                    current_start = sec_start + j + len(chunk)

    # Flush any remaining rows in current batch
    _flush()

    # Handle trailing rows beyond the last declared section
    last_section_end = sections[-1][2] + 1 if sections else 0
    if last_section_end < n:
        trailing = rows[last_section_end:]
        for i in range(0, len(trailing), batch_size):
            chunk = trailing[i : i + batch_size]
            abs_start = last_section_end + i
            batch_notes = _build_batch_notes(
                sections, abs_start, abs_start + len(chunk), notes
            )
            batches.append(
                _Batch(
                    page_index=page_index,
                    batch_index=len(batches),
                    data_rows=chunk,
                    notes=batch_notes,
                )
            )

    return batches


def _sub_parsed_table(
    parsed_table: baml_types.ParsedTable,
    batch: _Batch,
) -> baml_types.ParsedTable:
    """Build a new ParsedTable for a batch, preserving structural metadata."""
    return baml_types.ParsedTable(
        table_type=parsed_table.table_type,
        headers=parsed_table.headers,
        aggregations=parsed_table.aggregations,
        data_rows=batch.data_rows,
        notes=batch.notes,
    )


# ─── Page splitting & merging ─────────────────────────────────────────────────


def _split_pages(compressed_text: str, page_separator: str = "\f") -> list[str]:
    """Split compressed text on page separator, dropping empty pages."""
    if page_separator not in compressed_text:
        return [compressed_text]
    return [p for p in compressed_text.split(page_separator) if p.strip()]


def _merge_mapped_tables(tables: list[baml_types.MappedTable]) -> baml_types.MappedTable:
    """Merge multiple MappedTable results (e.g. batches) into one.

    - **records**: concatenated from all tables
    - **unmapped_columns**: union, deduplicated, preserving first-seen order
    - **mapping_notes**: non-None notes joined with ``"; "``
    - **metadata**: ``field_mappings`` from the first result; ``sections_detected``
      merged from all results (deduplicated, preserving order)
    """
    all_records = []
    seen_unmapped: dict[str, None] = {}
    notes_parts: list[str] = []

    for mt in tables:
        all_records.extend(mt.records)
        for col in mt.unmapped_columns:
            seen_unmapped.setdefault(col, None)
        if mt.mapping_notes is not None:
            notes_parts.append(mt.mapping_notes)

    first_meta = tables[0].metadata
    merged_sections: list[str] | None = None
    seen_sections: dict[str, None] = {}
    for mt in tables:
        if mt.metadata.sections_detected:
            if merged_sections is None:
                merged_sections = []
            for s in mt.metadata.sections_detected:
                if s not in seen_sections:
                    seen_sections[s] = None
                    merged_sections.append(s)

    merged_metadata = baml_types.InterpretationMetadata(
        model=first_meta.model,
        table_type_inference=first_meta.table_type_inference,
        field_mappings=first_meta.field_mappings,
        sections_detected=merged_sections,
    )

    return baml_types.MappedTable(
        records=all_records,
        unmapped_columns=list(seen_unmapped),
        mapping_notes="; ".join(notes_parts) if notes_parts else None,
        metadata=merged_metadata,
    )


def _group_by_page(
    results: list[baml_types.MappedTable],
    page_numbers: list[int],
) -> dict[int, baml_types.MappedTable]:
    """Group batch results by page number, merging batches within each page.

    *results* and *page_numbers* are aligned 1:1.  Returns a dict keyed by
    1-indexed page number, where each value is a single MappedTable containing
    all records for that page.
    """
    from collections import defaultdict
    page_batches: dict[int, list[baml_types.MappedTable]] = defaultdict(list)
    for mt, page in zip(results, page_numbers):
        page_batches[page].append(mt)
    return {
        page: _merge_mapped_tables(batches) if len(batches) > 1 else batches[0]
        for page, batches in sorted(page_batches.items())
    }


# ─── Pre-Step-1 table splitting ────────────────────────────────────────────────


def _count_pipe_data_rows(compressed_text: str) -> int:
    """Count pipe-table data rows (excluding aggregation rows).

    Walks through *compressed_text* looking for ``|---|`` separators that mark
    the start of data rows, then counts subsequent pipe rows whose first cell
    is non-empty (i.e. not aggregation ``||`` rows).
    """
    total = 0
    in_data = False
    for line in compressed_text.split("\n"):
        s = line.strip()
        if s.startswith("|---"):
            in_data = True
            continue
        if in_data:
            if s.startswith("|") and s:
                if not s.startswith("||"):
                    total += 1
            else:
                in_data = False
    return total


def _split_pipe_table(compressed_text: str, max_rows: int) -> list[str]:
    """Split a compressed pipe-table into chunks when data rows exceed *max_rows*.

    Each chunk retains the preamble (title lines, header pipe rows, separator)
    so Step 1 can understand the table structure.  Aggregation rows (``||``)
    are excluded from all chunks since Step 1 is instructed to ignore them.

    Returns ``[compressed_text]`` unchanged when the table has <= *max_rows*
    data rows.
    """
    lines = compressed_text.split("\n")

    # --- Parse into structural segments ---
    preamble_lines: list[str] = []
    body_items: list[tuple[str, str]] = []  # ("data_row"|"section_label"|"agg_row", line)
    in_data = False
    preamble_done = False
    current_section: str | None = None

    for line in lines:
        s = line.strip()

        if not preamble_done:
            preamble_lines.append(line)
            if s.startswith("|---"):
                preamble_done = True
                in_data = True
            continue

        if in_data:
            if s.startswith("|") and s:
                if s.startswith("||"):
                    body_items.append(("agg_row", line))
                else:
                    body_items.append(("data_row", line))
            else:
                in_data = False
                if s and not s.startswith("|") and "\t" not in s:
                    current_section = s
                    body_items.append(("section_label", line))
        else:
            if s.startswith("|---"):
                in_data = True
                # Don't re-add the separator — preamble already has it
                continue
            if s.startswith("|") and s:
                # Sub-header pipe rows between sections — skip for data purposes
                continue
            if s and not s.startswith("|") and "\t" not in s:
                current_section = s
                body_items.append(("section_label", line))

    # Count total data rows
    data_count = sum(1 for kind, _ in body_items if kind == "data_row")
    if data_count <= max_rows:
        return [compressed_text]

    # --- Group body items into chunks ---
    preamble = "\n".join(preamble_lines)
    chunks: list[str] = []
    current_chunk_lines: list[str] = []
    current_chunk_data_count = 0

    def _flush_chunk() -> None:
        nonlocal current_chunk_lines, current_chunk_data_count
        if current_chunk_lines:
            chunks.append(preamble + "\n" + "\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_chunk_data_count = 0

    pending_section_label: str | None = None

    for kind, line in body_items:
        if kind == "agg_row":
            # Exclude aggregation rows from all chunks
            continue
        if kind == "section_label":
            # If we have data and adding this section would eventually exceed,
            # try to break at this natural boundary
            if current_chunk_data_count >= max_rows:
                _flush_chunk()
            pending_section_label = line
            continue
        # kind == "data_row"
        if current_chunk_data_count >= max_rows:
            _flush_chunk()

        # Add pending section label if any
        if pending_section_label is not None:
            current_chunk_lines.append(pending_section_label)
            pending_section_label = None

        current_chunk_lines.append(line)
        current_chunk_data_count += 1

    _flush_chunk()

    log.info(
        "Pre-split: %d data rows → %d chunks (max_rows=%d)",
        data_count,
        len(chunks),
        max_rows,
    )
    return chunks


def _merge_parsed_chunks(chunks: list[baml_types.ParsedTable]) -> baml_types.ParsedTable:
    """Merge Step 1 results from multiple chunks into a single ``ParsedTable``.

    - ``table_type``: from first chunk
    - ``headers``: from first chunk
    - ``aggregations``: from last chunk that has any, or first chunk
    - ``data_rows``: concatenated from all chunks in order
    - ``notes``: section boundaries rebuilt with correct global row indices
    """
    all_data_rows: list[list[str]] = []
    # Collect per-chunk section info for rebuilding global boundaries
    chunk_sections: list[list[tuple[str, int, int]]] = []

    for chunk in chunks:
        chunk_sections.append(
            _parse_section_boundaries(chunk.notes) or []
        )
        all_data_rows.extend(chunk.data_rows)

    # Find aggregations: use last chunk that has non-empty aggregations
    aggregations = chunks[0].aggregations
    for chunk in reversed(chunks):
        if chunk.aggregations:
            aggregations = chunk.aggregations
            break

    # Rebuild section boundaries with global row indices
    merged_notes: str | None = None
    offset = 0
    global_sections: list[tuple[str, int, int]] = []
    for ci, chunk in enumerate(chunks):
        chunk_rows = len(chunk.data_rows)
        for label, local_start, local_end in chunk_sections[ci]:
            global_sections.append((label, offset + local_start, offset + local_end))
        offset += chunk_rows

    if global_sections:
        parts = [f"{label} (rows {s}-{e})" for label, s, e in global_sections]
        merged_notes = "Sections: " + ", ".join(parts)

    # Preserve non-section notes from first chunk
    if chunks[0].notes and merged_notes:
        non_section = _SECTION_RE.sub("", chunks[0].notes)
        non_section = re.sub(r"Sections?\s*:\s*", "", non_section)
        non_section = re.sub(r"^[\s,;]+|[\s,;]+$", "", non_section)
        if non_section:
            merged_notes = f"{merged_notes}; {non_section}"
    elif chunks[0].notes and not merged_notes:
        merged_notes = chunks[0].notes

    return baml_types.ParsedTable(
        table_type=chunks[0].table_type,
        headers=chunks[0].headers,
        aggregations=aggregations,
        data_rows=all_data_rows,
        notes=merged_notes,
    )


# ─── Deterministic pipe-table parser ──────────────────────────────────────────


@dataclass
class _DeterministicParsed:
    """Result of deterministic pipe-table parsing (no LLM)."""

    title: str | None
    headers: list[str]  # column names from pipe header row
    data_rows: list[list[str]]  # cell values per row
    sections: list[tuple[str, int, int]]  # (label, start_row_inclusive, end_row_inclusive)
    aggregation_rows: list[list[str]]  # ||rows


def _split_pipe_row(line: str) -> list[str]:
    """Split a pipe-delimited row into stripped cell values.

    Handles leading/trailing pipes: ``| A | B | C |`` → ``["A", "B", "C"]``.
    """
    parts = line.split("|")
    # Strip first and last empty strings from leading/trailing |
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.strip() for p in parts]


_TITLE_RE = re.compile(r"^##\s+(.+)")
_SECTION_LABEL_RE = re.compile(r"^\*\*(.+)\*\*$")
_SEPARATOR_RE = re.compile(r"^\|[-| :]+\|$")


def _parse_pipe_table_deterministic(compressed_text: str) -> _DeterministicParsed | None:
    """Parse compressed pipe-table markdown into structured data.

    Returns ``None`` if the text does not contain a recognizable pipe-table.
    """
    lines = compressed_text.split("\n")

    title: str | None = None
    headers: list[str] = []
    data_rows: list[list[str]] = []
    sections: list[tuple[str, int, int]] = []  # (label, start, end) inclusive
    aggregation_rows: list[list[str]] = []

    found_header = False
    found_separator = False
    current_section: str | None = None
    section_start: int | None = None

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Title: ## heading
        if not found_header:
            m = _TITLE_RE.match(s)
            if m:
                title = m.group(1).strip()
                continue

        # Header row: first pipe row before separator
        if not found_header and s.startswith("|") and not _SEPARATOR_RE.match(s):
            headers = _split_pipe_row(s)
            found_header = True
            continue

        # Separator: |---|---|...|
        if found_header and not found_separator and _SEPARATOR_RE.match(s):
            found_separator = True
            continue

        if not found_separator:
            continue

        # After separator: data rows, aggregation rows, section labels, or re-headers

        # Section label: **BOLD TEXT**
        m = _SECTION_LABEL_RE.match(s)
        if m:
            # Close previous section
            if current_section is not None and section_start is not None:
                sections.append((current_section, section_start, len(data_rows) - 1))
            current_section = m.group(1).strip()
            section_start = len(data_rows)
            continue

        # Skip repeated separator/header rows between sections
        if _SEPARATOR_RE.match(s):
            continue
        if s.startswith("|") and not s.startswith("||"):
            # Check if this looks like a re-header (same content as headers)
            candidate = _split_pipe_row(s)
            if candidate == headers:
                continue

        # Aggregation row: || prefix (first cell empty)
        if s.startswith("||"):
            aggregation_rows.append(_split_pipe_row(s))
            continue

        # Data row
        if s.startswith("|"):
            data_rows.append(_split_pipe_row(s))
            continue

        # Plain text section header (no bold markers) — used by some formats
        if not s.startswith("|") and "\t" not in s:
            # Close previous section
            if current_section is not None and section_start is not None:
                sections.append((current_section, section_start, len(data_rows) - 1))
            current_section = s
            section_start = len(data_rows)

    # Close final section
    if current_section is not None and section_start is not None and data_rows:
        sections.append((current_section, section_start, len(data_rows) - 1))

    if not found_separator or not headers:
        return None

    return _DeterministicParsed(
        title=title,
        headers=headers,
        data_rows=data_rows,
        sections=sections,
        aggregation_rows=aggregation_rows,
    )


# ─── Deterministic schema mapper ─────────────────────────────────────────────


@dataclass
class _ColumnGroup:
    """A group of header columns that share the same dimension values."""

    dimensions: list[tuple[ColumnDef, str]]  # (schema_col, header_text_value)
    measures: list[tuple[int, ColumnDef]]  # (header_col_index, schema_col)


def _map_to_schema_deterministic(
    parsed: _DeterministicParsed,
    schema: CanonicalSchema,
) -> tuple[baml_types.MappedTable | None, list[str]]:
    """Try to map parsed pipe-table to schema using alias matching only.

    Returns ``(mapped_table, unmatched_parts)``.  When *unmatched_parts* is
    empty, the mapping is complete.  When non-empty, the caller should fall
    back to LLM interpretation.
    """
    if not parsed.headers or not parsed.data_rows:
        return None, ["empty table"]

    # Phase 1: Match header parts to schema aliases (case-insensitive)
    # Build alias → list of schema columns (multiple columns may share an alias,
    # e.g. Value(float) and Year(int) both have "2025")
    alias_to_cols: dict[str, list[ColumnDef]] = {}
    for col in schema.columns:
        for alias in col.aliases:
            alias_to_cols.setdefault(alias.strip().lower(), []).append(col)

    MEASURE_TYPES = {"int", "float"}
    DIMENSION_TYPES = {"string", "date"}

    def _resolve_alias(part: str) -> list[ColumnDef]:
        """Resolve a header part to matching schema column(s)."""
        matches = alias_to_cols.get(part.lower(), [])
        if len(matches) <= 1:
            return matches
        # Type-based disambiguation: when the same alias matches a dimension
        # (string/date) AND a measure (int/float), keep both — they serve
        # different roles (dimension gets header text, measure gets cell value).
        dims = [c for c in matches if c.type in DIMENSION_TYPES]
        meas = [c for c in matches if c.type in MEASURE_TYPES]
        if dims and meas:
            return matches  # keep all — different roles
        # All measure types: int vs float have different roles — float columns
        # are cell-value measures, int columns often serve as identifiers or
        # period labels (e.g., Year=2025 vs Value=cell_data).
        floats = [c for c in matches if c.type == "float"]
        ints = [c for c in matches if c.type == "int"]
        if floats and ints:
            return matches  # keep all — float=measure, int=pseudo-dimension
        # Same exact type: keep only the first (arbitrary but deterministic)
        return [matches[0]]

    # For each header column, split on ` / ` and match parts
    header_mappings: list[list[tuple[str, list[ColumnDef]]]] = []
    # (part_text, matched_cols) per header column

    for hi, header in enumerate(parsed.headers):
        parts = [p.strip() for p in header.split(" / ")] if " / " in header else [header.strip()]
        col_parts: list[tuple[str, list[ColumnDef]]] = []
        for part in parts:
            col_parts.append((part, _resolve_alias(part)))
        header_mappings.append(col_parts)

    # Phase 2: Classify columns — dimension vs measure
    # Dimension: type in ("string", "date") → value comes from header text
    # Measure: type in ("int", "float") → value comes from cell
    # When the same alias matches both types, the part contributes to BOTH:
    # the dimension column gets the header text, the measure column gets cell data.

    shared_cols: list[tuple[int, ColumnDef]] = []  # (header_col_idx, schema_col)
    group_info: list[dict] = []  # per header column: {"dimensions": [...], "measures": [...], "unmatched": [...]}

    for hi, col_parts in enumerate(header_mappings):
        info: dict = {"dimensions": [], "measures": [], "unmatched": []}
        for part_text, matched_cols in col_parts:
            if not matched_cols:
                info["unmatched"].append(part_text)
            else:
                # When int and float columns share an alias, the float is the
                # measure (cell value) and the int is a dimension (header text
                # value, e.g., Year="2025" vs Value=cell_data).
                has_float = any(c.type == "float" for c in matched_cols)
                for matched_col in matched_cols:
                    if matched_col.type in DIMENSION_TYPES:
                        info["dimensions"].append((matched_col, part_text))
                    elif matched_col.type == "int" and has_float:
                        info["dimensions"].append((matched_col, part_text))
                    else:
                        info["measures"].append((hi, matched_col))
        group_info.append(info)

        # Simple column: single part, single match, dimension → shared column
        if len(col_parts) == 1 and len(col_parts[0][1]) >= 1:
            all_dim = all(c.type in DIMENSION_TYPES for c in col_parts[0][1])
            if all_dim:
                shared_cols.append((hi, col_parts[0][1][0]))

    # Classify unmatched parts: lenient approach
    # - Columns with NO matches at all → entirely skipped (unmapped), not blocking
    # - Columns with SOME matches → unmatched parts are annotations, not blocking
    # - Only flag for LLM fallback when a matched column has ambiguous unmatched parts
    #   that could be important (currently: never — we trust the alias coverage)
    unmatched_parts: list[str] = []
    for hi, info in enumerate(group_info):
        if not info["unmatched"]:
            continue
        has_matches = bool(info["dimensions"] or info["measures"])
        if has_matches:
            # Column has matched parts — unmatched parts are annotations (skip)
            continue
        # Column has NO matches at all — check if it's skippable
        all_skippable = all(
            p in ("%", "%%", "pct", "share") for p in info["unmatched"]
        )
        if all_skippable:
            continue
        # Entirely unmatched column — flag for LLM fallback
        unmatched_parts.extend(info["unmatched"])

    if unmatched_parts:
        return None, unmatched_parts

    # Phase 3: Detect unpivot groups
    # Find dimension columns that appear with ≥2 distinct values across headers
    # These form the group dimension (e.g., "crop" has "spring crops" and "spring grain")
    dim_values: dict[str, list[tuple[int, str]]] = {}  # col_name → [(header_idx, value)]
    for hi, info in enumerate(group_info):
        for matched_col, value in info["dimensions"]:
            dim_values.setdefault(matched_col.name, []).append((hi, value))

    # Group dimensions: appear in >1 header columns with distinct values
    group_dim_names: set[str] = set()
    for col_name, entries in dim_values.items():
        unique_values = {v for _, v in entries}
        if len(unique_values) >= 2:
            group_dim_names.add(col_name)

    # Constant dimensions: non-group dims with a single unique value that appear
    # in ≥2 header columns (e.g., Year="2025" when Year(int) shares alias with
    # Value(float), or Unit="Th.ha." across multiple label columns).
    constant_dims: list[tuple[ColumnDef, str]] = []  # applied to every record
    for col_name, entries in dim_values.items():
        if col_name in group_dim_names:
            continue
        unique_values = {v for _, v in entries}
        if len(unique_values) == 1 and len(entries) >= 2:
            col_def = next(c for c in schema.columns if c.name == col_name)
            value = next(iter(unique_values))
            constant_dims.append((col_def, value))

    # Re-classify compound label columns that contain non-group dimensions.
    # For compound header columns with only non-group dimensions and no measures
    # (e.g., "Th.ha. / Region"):
    # - The LAST dimension part → shared column (cell value, e.g. Region)
    # - Earlier dimension parts → constant dimensions (header text, e.g. Th.ha.)
    # Skip columns whose dims are already fully covered by existing
    # constant_dims / shared_cols (e.g., "Th.ha. / MOA 2024" when Th.ha. is
    # already a constant from another column).
    for hi, info in enumerate(group_info):
        if any(idx == hi for idx, _ in shared_cols):
            continue  # already a shared col
        dims = info["dimensions"]
        has_group_dims = any(col.name in group_dim_names for col, _ in dims)
        if has_group_dims or info["measures"]:
            continue  # participates in groups already
        if not dims:
            continue
        # Check if all dimensions are already covered
        all_covered = all(
            any(cd.name == col.name for cd, _ in constant_dims)
            or any(sc.name == col.name for _, sc in shared_cols)
            for col, _ in dims
        )
        if all_covered:
            continue  # dimensions already accounted for — skip this column
        # Non-group, non-measure compound header column:
        # Last dimension → shared (cell value), rest → constant
        *const_parts, cell_part = dims
        shared_cols.append((hi, cell_part[0]))
        for col_def, val in const_parts:
            if not any(cd.name == col_def.name for cd, _ in constant_dims):
                constant_dims.append((col_def, val))

    # Build column groups
    if not group_dim_names:
        # No unpivoting needed — single group with all measure columns
        measures: list[tuple[int, ColumnDef]] = []
        for hi, info in enumerate(group_info):
            measures.extend(info["measures"])

        if measures or shared_cols:
            groups = [_ColumnGroup(dimensions=list(constant_dims), measures=measures)]
        else:
            return None, ["no measures found"]
    else:
        # Group columns by their group-dimension values
        # Each unique combination of group-dimension values forms a group
        group_key_to_cols: dict[tuple[tuple[str, str], ...], _ColumnGroup] = {}

        for hi, info in enumerate(group_info):
            # Skip shared columns
            if any(idx == hi for idx, _ in shared_cols):
                continue

            # Extract group dimension values for this header column
            group_dims: list[tuple[str, str]] = []
            for matched_col, value in info["dimensions"]:
                if matched_col.name in group_dim_names:
                    group_dims.append((matched_col.name, value))

            if not group_dims and not info["measures"]:
                continue  # no group dims and no measures → skip

            key = tuple(sorted(group_dims))
            if key not in group_key_to_cols:
                # Build dimension list: group dims + constant dims
                dims: list[tuple[ColumnDef, str]] = []
                for matched_col, value in info["dimensions"]:
                    if matched_col.name in group_dim_names:
                        dims.append((matched_col, value))
                dims.extend(constant_dims)
                group_key_to_cols[key] = _ColumnGroup(dimensions=dims, measures=[])

            group_key_to_cols[key].measures.extend(info["measures"])

        groups = list(group_key_to_cols.values())

    if not groups:
        return None, ["no column groups formed"]

    # Detect section-to-schema mapping
    section_col: ColumnDef | None = None
    if parsed.sections:
        # Find a schema column whose aliases match section labels
        for col in schema.columns:
            col_aliases_lower = {a.lower() for a in col.aliases}
            section_labels_lower = {label.lower() for label, _, _ in parsed.sections}
            if section_labels_lower & col_aliases_lower:
                section_col = col
                break

    # Phase 4: Build records
    records: list[dict] = []

    def _build_records_for_range(
        start: int, end: int, section_label: str | None
    ) -> None:
        for row_idx in range(start, end + 1):
            if row_idx >= len(parsed.data_rows):
                break
            row = parsed.data_rows[row_idx]
            for group in groups:
                record: dict = {}
                # Shared columns: value from cell
                for col_idx, schema_col in shared_cols:
                    if col_idx < len(row):
                        record[schema_col.name] = row[col_idx]
                # Group dimension values: value from header text
                for schema_col, value in group.dimensions:
                    record[schema_col.name] = value
                # Measure values: value from cell
                for col_idx, schema_col in group.measures:
                    if col_idx < len(row):
                        record[schema_col.name] = row[col_idx]
                # Section label → matching schema column
                if section_col and section_label:
                    record[section_col.name] = section_label
                records.append(record)

    if parsed.sections:
        for label, start, end in parsed.sections:
            _build_records_for_range(start, end, label)
    else:
        _build_records_for_range(0, len(parsed.data_rows) - 1, None)

    # Phase 5: Build MappedTable
    # Build MappedRecord instances
    mapped_records: list[baml_types.MappedRecord] = []
    for rec in records:
        mapped_records.append(baml_types.MappedRecord(**rec))

    # Build field mappings
    field_mappings: list[baml_types.FieldMapping] = []
    for col in schema.columns:
        # Determine source
        source = "deterministic alias matching"
        if any(col.name == sc.name for _, sc in shared_cols):
            source = "cell value (shared column)"
        elif col.name in group_dim_names:
            source = "header text (unpivot dimension)"
        elif section_col and col.name == section_col.name:
            source = "section label"

        field_mappings.append(baml_types.FieldMapping(
            column_name=col.name,
            source=source,
            rationale="Matched via schema aliases",
            confidence=baml_types.Confidence.High,
        ))

    # Build unmapped columns (header columns not contributing to any schema column)
    unmapped: list[str] = []
    mapped_indices: set[int] = set()
    for idx, _ in shared_cols:
        mapped_indices.add(idx)
    for group in groups:
        for idx, _ in group.measures:
            mapped_indices.add(idx)
    for hi in range(len(parsed.headers)):
        if hi not in mapped_indices and not group_info[hi]["dimensions"]:
            unmapped.append(parsed.headers[hi])

    # Detect sections for metadata
    sections_detected: list[str] | None = None
    if parsed.sections:
        sections_detected = [label for label, _, _ in parsed.sections]

    metadata = baml_types.InterpretationMetadata(
        model="deterministic",
        table_type_inference=baml_types.TableTypeInference(
            table_type=(
                baml_types.TableType.PivotedTable if group_dim_names
                else baml_types.TableType.FlatHeader
            ),
            mapping_strategy_used="unpivot" if group_dim_names else "1:1 row mapping",
        ),
        field_mappings=field_mappings,
        sections_detected=sections_detected,
    )

    mapped_table = baml_types.MappedTable(
        records=mapped_records,
        unmapped_columns=unmapped,
        mapping_notes="Deterministic mapping - all columns resolved via alias matching",
        metadata=metadata,
    )

    return mapped_table, []


def _try_deterministic(
    page_text: str,
    schema: CanonicalSchema,
) -> baml_types.MappedTable | None:
    """Attempt deterministic interpretation of a single page.

    Returns a ``MappedTable`` if fully resolved, or ``None`` to signal
    that LLM fallback is needed.
    """
    parsed = _parse_pipe_table_deterministic(page_text)
    if parsed is None:
        return None
    result, unmatched = _map_to_schema_deterministic(parsed, schema)
    if result is not None and not unmatched:
        log.info("Deterministic mapping: %d records", len(result.records))
        return result
    if unmatched:
        log.debug("Deterministic mapping failed, unmatched: %s", unmatched)
    return None


# ─── Batched async orchestrator ────────────────────────────────────────────────


async def _interpret_pages_batched_async(
    pages: list[str],
    schema: CanonicalSchema,
    *,
    batch_size: int = 20,
    step1_max_rows: int = 40,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    page_images: list[str] | None = None,
    vision_model: str | None = None,
) -> dict[int, baml_types.MappedTable]:
    """Concurrently run the 2-step pipeline with step-2 batching.

    Tries deterministic alias-based mapping first for each page.  Pages that
    cannot be resolved deterministically fall through to the LLM pipeline.
    Deterministic mapping is skipped when *page_images* is provided (vision
    mode implies garbled headers that need LLM).

    Returns a dict keyed by 1-indexed page number, where each value is a
    ``MappedTable`` containing all records for that page (batches merged).

    When *page_images* is provided, a vision-based schema inference step
    (step 0) runs first, and step 1 uses the guided variant that splits
    concatenated values based on the inferred column structure.

    Large tables (more than *step1_max_rows* data rows) are pre-split into
    chunks before Step 1 to prevent LLM output truncation, then merged back
    after Step 1 completes.
    """
    # Try deterministic mapping first (skip when vision is enabled)
    deterministic_results: dict[int, baml_types.MappedTable] = {}
    llm_page_indices: list[int] = []  # original 0-based indices needing LLM

    if page_images is None:
        for pi, page in enumerate(pages):
            det = _try_deterministic(page, schema)
            if det is not None:
                deterministic_results[pi + 1] = det  # 1-indexed
                log.info("Page %d: deterministic mapping (%d records)", pi + 1, len(det.records))
            else:
                llm_page_indices.append(pi)
        if not llm_page_indices:
            return deterministic_results
        # Narrow pages to only those needing LLM
        llm_pages = [pages[i] for i in llm_page_indices]
    else:
        llm_pages = pages
        llm_page_indices = list(range(len(pages)))

    effective_vision_model = vision_model or model

    # Step 0 (optional): infer visual schemas concurrently
    visual_schemas: list[baml_types.InferredTableSchema] | None = None
    if page_images is not None:
        visual_schemas = list(await asyncio.gather(
            *(
                infer_table_schema_from_image_async(
                    img, page, model=effective_vision_model
                )
                for img, page in zip(page_images, pages)
            )
        ))
        for pi, vs in enumerate(visual_schemas):
            log.info(
                "Step 0 page %d: inferred %d columns: %s",
                pi + 1, vs.column_count, vs.column_names,
            )

    # Pre-split large pages to prevent Step 1 output truncation
    expanded_pages: list[str] = []
    page_chunk_counts: list[int] = []  # how many chunks per original page

    for page in llm_pages:
        chunks = _split_pipe_table(page, step1_max_rows)
        expanded_pages.extend(chunks)
        page_chunk_counts.append(len(chunks))

    # Expand vision schemas to match expanded_pages (repeat per chunk)
    expanded_visual_schemas: list[baml_types.InferredTableSchema] | None = None
    if visual_schemas is not None:
        expanded_visual_schemas = []
        for vs, count in zip(visual_schemas, page_chunk_counts):
            expanded_visual_schemas.extend([vs] * count)

    # Step 1: parse all (expanded) pages concurrently
    if expanded_visual_schemas is not None:
        parsed_tables = list(await asyncio.gather(
            *(
                analyze_and_parse_guided_async(
                    page, vs, model=model, fallback_model=fallback_model
                )
                for page, vs in zip(expanded_pages, expanded_visual_schemas)
            )
        ))
    else:
        parsed_tables = list(await asyncio.gather(
            *(
                analyze_and_parse_async(page, model=model, fallback_model=fallback_model)
                for page in expanded_pages
            )
        ))

    # Merge Step 1 chunks back into original pages
    merged_parsed: list[baml_types.ParsedTable] = []
    idx = 0
    for count in page_chunk_counts:
        if count == 1:
            merged_parsed.append(parsed_tables[idx])
        else:
            merged_parsed.append(_merge_parsed_chunks(parsed_tables[idx : idx + count]))
        idx += count
    parsed_tables = merged_parsed

    for pi, pt in enumerate(parsed_tables):
        log.info(
            "Step 1 page %d: parsed %d data rows, %d header levels, type=%s",
            pi + 1,
            len(pt.data_rows),
            pt.headers.levels,
            pt.table_type,
        )
        for lvl, names in enumerate(pt.headers.names):
            log.info("  Step 1 page %d headers[%d]: %s", pi + 1, lvl, names)
        if pt.data_rows:
            log.info("  Step 1 page %d row[0] (%d cells): %s", pi + 1, len(pt.data_rows[0]), pt.data_rows[0])

    # Cross-validate: check parsed row cell counts match vision column count
    if visual_schemas is not None:
        for pi, (pt, vs) in enumerate(zip(parsed_tables, visual_schemas)):
            if pt.data_rows:
                row_len = len(pt.data_rows[0])
                if row_len != vs.column_count:
                    log.warning(
                        "Step 1 page %d: guided parsing produced %d cells but vision "
                        "inferred %d columns — falling back to non-vision parsing",
                        pi + 1, row_len, vs.column_count,
                    )
                    # Re-parse this page without vision guidance
                    fallback_parsed = await analyze_and_parse_async(
                        llm_pages[pi], model=model, fallback_model=fallback_model
                    )
                    parsed_tables[pi] = fallback_parsed
                    log.info(
                        "Step 1 page %d: fallback produced %d cells",
                        pi + 1, len(fallback_parsed.data_rows[0]) if fallback_parsed.data_rows else 0,
                    )

    # Validate section boundaries against compressed text
    for pi, pt in enumerate(parsed_tables):
        corrected = _validate_section_boundaries(
            pt.notes, len(pt.data_rows), llm_pages[pi]
        )
        if corrected != pt.notes:
            log.info(
                "Step 1 page %d: corrected section boundaries: %s",
                pi + 1,
                corrected,
            )
            parsed_tables[pi] = baml_types.ParsedTable(
                table_type=pt.table_type,
                headers=pt.headers,
                aggregations=pt.aggregations,
                data_rows=pt.data_rows,
                notes=corrected,
            )

    # Create batches for all pages
    all_batches: list[_Batch] = []
    for pi, pt in enumerate(parsed_tables):
        page_batches = _create_batches(pt, page_index=pi, batch_size=batch_size)
        all_batches.extend(page_batches)

    log.info(
        "Step 2: %d batches from %d pages (batch_size=%d)",
        len(all_batches),
        len(llm_pages),
        batch_size,
    )

    # Build sub-ParsedTables for each batch
    sub_tables = [
        _sub_parsed_table(parsed_tables[b.page_index], b) for b in all_batches
    ]

    # Step 2: map all batches concurrently
    mapped_tables = await asyncio.gather(
        *(
            map_to_schema_async(sub, schema, model=model, fallback_model=fallback_model)
            for sub in sub_tables
        )
    )

    # Row count validation
    for batch, mapped in zip(all_batches, mapped_tables):
        expected = len(batch.data_rows)
        actual = len(mapped.records)
        if actual < expected:
            log.warning(
                "Batch (page %d, batch %d): expected %d records but got %d",
                batch.page_index + 1,
                batch.batch_index,
                expected,
                actual,
            )

    # Fix table_type in metadata: use the authoritative value from step 1 (ParsedTable)
    # The LLM in step 2 sometimes re-classifies incorrectly, so we override it here.
    for batch, mapped in zip(all_batches, mapped_tables):
        parsed = parsed_tables[batch.page_index]
        if mapped.metadata.table_type_inference.table_type != parsed.table_type:
            log.debug(
                "Fixing table_type metadata: step 2 said %s, step 1 said %s",
                mapped.metadata.table_type_inference.table_type,
                parsed.table_type,
            )
            mapped.metadata.table_type_inference.table_type = parsed.table_type

    # Group batch results by LLM-local page number, then remap to original indices
    page_numbers = [b.page_index + 1 for b in all_batches]  # 1-indexed within llm_pages
    llm_grouped = _group_by_page(list(mapped_tables), page_numbers)

    # Remap LLM results to original page indices and merge with deterministic results
    for llm_page, mt in llm_grouped.items():
        original_idx = llm_page_indices[llm_page - 1]
        deterministic_results[original_idx + 1] = mt
    return deterministic_results


# ─── Sync API ─────────────────────────────────────────────────────────────────


def analyze_and_parse(
    compressed_text: str,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> baml_types.ParsedTable:
    """Analyze table structure and parse data rows (step 1 of 2-step pipeline).

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        model: LLM to use, e.g. ``"openai/gpt-4o"`` or a BAML client name.
        fallback_model: If set, retry with this model when ``model`` fails.
    """
    try:
        return b_sync.AnalyzeAndParseTable(compressed_text, {"client": model})
    except Exception:
        if fallback_model is None:
            raise
        log.warning("Primary model %r failed for AnalyzeAndParseTable, falling back to %r", model, fallback_model)
        return b_sync.AnalyzeAndParseTable(compressed_text, {"client": fallback_model})


def infer_table_schema_from_image(
    page_image_b64: str,
    compressed_text: str,
    *,
    model: str = DEFAULT_MODEL,
) -> baml_types.InferredTableSchema:
    """Infer table column structure from a page image (vision step)."""
    img = Image.from_base64("image/png", page_image_b64)
    return b_sync.InferTableSchemaFromImage(img, compressed_text, {"client": model})


def analyze_and_parse_guided(
    compressed_text: str,
    visual_schema: baml_types.InferredTableSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> baml_types.ParsedTable:
    """Step 1 guided by visual schema (for garbled-header PDFs)."""
    try:
        return b_sync.AnalyzeAndParseTableGuided(
            compressed_text, visual_schema, {"client": model}
        )
    except Exception:
        if fallback_model is None:
            raise
        log.warning("Primary model %r failed for guided parse, falling back to %r", model, fallback_model)
        return b_sync.AnalyzeAndParseTableGuided(
            compressed_text, visual_schema, {"client": fallback_model}
        )


def map_to_schema(
    parsed_table: baml_types.ParsedTable,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> baml_types.MappedTable:
    """Map parsed table to canonical schema (step 2 of 2-step pipeline).

    Args:
        parsed_table: Output from ``analyze_and_parse()``.
        schema: Canonical schema to map to.
        model: LLM to use.
        fallback_model: If set, retry with this model when ``model`` fails.
    """
    tb = _build_type_builder(schema)
    baml_schema = _to_baml_schema(schema)
    try:
        return b_sync.MapToCanonicalSchema(
            parsed_table, baml_schema, model, {"tb": tb, "client": model}
        )
    except Exception:
        if fallback_model is None:
            raise
        log.warning("Primary model %r failed for MapToCanonicalSchema, falling back to %r", model, fallback_model)
        return b_sync.MapToCanonicalSchema(
            parsed_table, baml_schema, fallback_model, {"tb": tb, "client": fallback_model}
        )


def interpret_table(
    compressed_text: str,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    page_separator: str = "\f",
    batch_size: int = 20,
    step1_max_rows: int = 40,
    pdf_path: str | bytes | Path | None = None,
    vision_model: str | None = None,
    unpivot: bool = True,
) -> dict[int, baml_types.MappedTable]:
    """Full 2-step pipeline: analyze+parse, then map to schema.

    Multi-page input (pages joined by *page_separator*) is automatically split
    and all pages are processed concurrently via ``asyncio.gather()``.

    Returns a dict keyed by 1-indexed page number, where each value is a
    ``MappedTable`` containing that page's records, unmapped columns, mapping
    notes, and metadata.  Records contain only canonical schema fields.

    Step 1 (structure parsing) is pre-split: tables with more than
    *step1_max_rows* data rows are split into chunks before the LLM call,
    then merged after.  This prevents output truncation on large tables.

    Step 2 (schema mapping) is batched: each page's parsed rows are split into
    chunks of *batch_size* rows so the LLM can produce complete output without
    truncation.  All batches across all pages run concurrently, then batches
    are merged per-page.

    When *pdf_path* is provided, a vision-based schema inference step runs
    before step 1: each page is rendered as an image and a vision-capable LLM
    infers the correct column structure.  Step 1 then uses a guided variant
    that splits concatenated values based on the inferred schema.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        schema: Canonical schema to map to.
        model: LLM to use for both steps.
        fallback_model: If set, retry each step with this model on failure.
        page_separator: Delimiter between pages (default ``"\\f"``).
        batch_size: Maximum data rows per step-2 LLM call (default 20).
        step1_max_rows: Maximum data rows per step-1 LLM call (default 40).
            Tables exceeding this are pre-split into chunks.
        pdf_path: Path to original PDF (str, Path, or bytes). When provided,
            enables vision-based schema inference (step 0) and guided parsing
            (step 1).
        vision_model: LLM to use for vision step 0. Defaults to *model*.
        unpivot: When True (default), detect and unpivot pivoted tables
            before interpretation.  Pivoted tables have repeating compound
            header groups (e.g. ``"crop A / 2025"``, ``"crop B / 2025"``).
    """
    compressed_text = normalize_pipe_table(compressed_text)
    pages = _split_pages(compressed_text, page_separator)

    if unpivot:
        from pdf_ocr.unpivot import unpivot_pipe_table as _unpivot
        pages = [_unpivot(p).text for p in pages]

    # Render page images when vision is enabled
    page_images: list[str] | None = None
    if pdf_path is not None:
        log.info("Vision enabled: rendering %d page(s)", len(pages))
        page_images = _render_page_images(pdf_path, pages=list(range(len(pages))))

    # Try deterministic mapping first (skip when vision is enabled)
    if page_images is None:
        det_results: dict[int, baml_types.MappedTable] = {}
        llm_needed: list[int] = []
        for pi, page in enumerate(pages):
            det = _try_deterministic(page, schema)
            if det is not None:
                det_results[pi + 1] = det
                log.info("Page %d: deterministic mapping (%d records)", pi + 1, len(det.records))
            else:
                llm_needed.append(pi)
        if not llm_needed:
            return det_results
        # Only LLM-needed pages continue below
        llm_pages = [pages[i] for i in llm_needed]
    else:
        det_results = {}
        llm_needed = list(range(len(pages)))
        llm_pages = pages

    if len(llm_pages) == 1 and page_images is None:
        # Fast path: single page, no vision
        # Pre-split if table is large to prevent Step 1 truncation
        chunks = _split_pipe_table(llm_pages[0], step1_max_rows)
        if len(chunks) == 1:
            # No split needed — try fully sync
            parsed = analyze_and_parse(llm_pages[0], model=model, fallback_model=fallback_model)
        else:
            # Split needed — run Step 1 on chunks concurrently, then merge
            parsed_chunks = _run_async(asyncio.gather(
                *(analyze_and_parse_async(c, model=model, fallback_model=fallback_model) for c in chunks)
            ))
            parsed = _merge_parsed_chunks(list(parsed_chunks))

        # Validate section boundaries against compressed text
        corrected = _validate_section_boundaries(
            parsed.notes, len(parsed.data_rows), llm_pages[0]
        )
        if corrected != parsed.notes:
            log.info("Section boundaries corrected: %s", corrected)
            parsed = baml_types.ParsedTable(
                table_type=parsed.table_type,
                headers=parsed.headers,
                aggregations=parsed.aggregations,
                data_rows=parsed.data_rows,
                notes=corrected,
            )
        batches = _create_batches(parsed, page_index=0, batch_size=batch_size)
        orig_page = llm_needed[0] + 1  # 1-indexed original page number
        if len(batches) == 1:
            # Fast path: single page, fits in one batch — sync, no event loop
            result = map_to_schema(parsed, schema, model=model, fallback_model=fallback_model)
            # Fix table_type: use authoritative value from step 1
            if result.metadata.table_type_inference.table_type != parsed.table_type:
                result.metadata.table_type_inference.table_type = parsed.table_type
            det_results[orig_page] = result
            return det_results
        # Single page but multiple batches — need async for concurrency
        llm_result = _run_async(
            _interpret_pages_batched_async(
                llm_pages, schema, batch_size=batch_size, step1_max_rows=step1_max_rows,
                model=model, fallback_model=fallback_model,
            )
        )
        # Remap: _interpret_pages_batched_async returns 1-indexed keys
        for llm_page, mt in llm_result.items():
            det_results[llm_needed[llm_page - 1] + 1] = mt
        return det_results

    # Multiple pages or vision enabled → batched concurrent processing
    # _interpret_pages_batched_async handles deterministic internally when
    # page_images is None, so pass original pages for vision case
    if page_images is not None:
        return _run_async(
            _interpret_pages_batched_async(
                pages, schema, batch_size=batch_size, step1_max_rows=step1_max_rows,
                model=model, fallback_model=fallback_model,
                page_images=page_images, vision_model=vision_model,
            )
        )
    # Non-vision multi-page: use llm_pages subset
    llm_result = _run_async(
        _interpret_pages_batched_async(
            llm_pages, schema, batch_size=batch_size, step1_max_rows=step1_max_rows,
            model=model, fallback_model=fallback_model,
        )
    )
    for llm_page, mt in llm_result.items():
        det_results[llm_needed[llm_page - 1] + 1] = mt
    return det_results


def interpret_tables(
    compressed_texts: list[str],
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    step1_max_rows: int = 40,
    unpivot: bool = True,
) -> list[baml_types.MappedTable]:
    """Interpret multiple independent tables concurrently.

    Sync wrapper around :func:`interpret_tables_async`.  Each table goes
    through the full 2-step pipeline (parse then map) with all tables
    processed in parallel via ``asyncio.gather()``.

    Use this when you have several pre-split tables (e.g. from
    ``compress_docx_tables()``) that share a schema and should be
    interpreted as fast as possible.

    Args:
        compressed_texts: List of compressed text strings, one per table.
        schema: Canonical schema to map all tables to.
        model: LLM to use for all calls.
        fallback_model: If set, retry each step with this model on failure.
        step1_max_rows: Maximum data rows per step-1 LLM call (default 40).
        unpivot: When True (default), detect and unpivot pivoted tables
            before interpretation.

    Returns:
        List of ``MappedTable`` results, one per input text, in the same
        order as *compressed_texts*.
    """
    return _run_async(
        interpret_tables_async(
            compressed_texts, schema, model=model, fallback_model=fallback_model,
            step1_max_rows=step1_max_rows, unpivot=unpivot,
        )
    )


def interpret_table_single_shot(
    compressed_text: str,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    page_separator: str = "\f",
    unpivot: bool = True,
) -> dict[int, baml_types.MappedTable]:
    """Single-shot: analyze, parse, and map in one LLM call.

    Multi-page input (pages joined by *page_separator*) is automatically split
    and all pages are processed concurrently via ``asyncio.gather()``.
    Single-page input is processed synchronously without an event loop.

    Returns a dict keyed by 1-indexed page number, where each value is a
    ``MappedTable`` for that page.

    .. warning::

       Single-shot mode cannot batch step 2 and is unsuitable for dense pages
       with many data rows (e.g. 50+ rows per page).  The LLM may truncate
       output.  Prefer :func:`interpret_table` which batches step 2
       automatically.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        schema: Canonical schema to map to.
        model: LLM to use.
        fallback_model: If set, retry with this model when ``model`` fails.
        page_separator: Delimiter between pages (default ``"\\f"``).
        unpivot: When True (default), detect and unpivot pivoted tables
            before interpretation.
    """
    compressed_text = normalize_pipe_table(compressed_text)
    pages = _split_pages(compressed_text, page_separator)

    if unpivot:
        from pdf_ocr.unpivot import unpivot_pipe_table as _unpivot
        pages = [_unpivot(p).text for p in pages]

    # Try deterministic mapping first
    det_results: dict[int, baml_types.MappedTable] = {}
    llm_needed: list[int] = []
    for pi, page in enumerate(pages):
        det = _try_deterministic(page, schema)
        if det is not None:
            det_results[pi + 1] = det
        else:
            llm_needed.append(pi)
    if not llm_needed:
        return det_results

    if len(pages) == 1:
        tb = _build_type_builder(schema)
        baml_schema = _to_baml_schema(schema)
        try:
            result = b_sync.InterpretTable(
                pages[0], baml_schema, model, {"tb": tb, "client": model}
            )
        except Exception:
            if fallback_model is None:
                raise
            log.warning("Primary model %r failed for InterpretTable, falling back to %r", model, fallback_model)
            result = b_sync.InterpretTable(
                pages[0], baml_schema, fallback_model, {"tb": tb, "client": fallback_model}
            )
        return {1: result}

    # Multiple pages → run all concurrently via the async BAML client
    results = _run_async(
        _interpret_pages_single_shot_async(pages, schema, model=model, fallback_model=fallback_model)
    )
    return {i + 1: mt for i, mt in enumerate(results)}


# ─── Async API ────────────────────────────────────────────────────────────────


async def _interpret_pages_async(
    pages: list[str],
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    step1_max_rows: int = 40,
) -> list[baml_types.MappedTable]:
    """Concurrently run the 2-step pipeline on multiple pages.

    Tries deterministic alias-based mapping first for each page.  Pages that
    cannot be resolved deterministically fall through to the LLM pipeline.

    Step 1: parse all pages in parallel (with pre-splitting for large tables).
    Step 2: map all parsed tables in parallel (one call per page — no batching).

    .. note::

       This function does **not** batch step 2.  For large pages the LLM may
       truncate output.  :func:`interpret_table` and
       :func:`_interpret_pages_batched_async` add step-2 batching.
       :func:`interpret_tables_async` continues to use this function for
       callers who want explicit control.
    """
    # Try deterministic mapping first
    results: list[baml_types.MappedTable | None] = [None] * len(pages)
    llm_indices: list[int] = []

    for pi, page in enumerate(pages):
        det = _try_deterministic(page, schema)
        if det is not None:
            results[pi] = det
            log.info("Page %d: deterministic mapping (%d records)", pi + 1, len(det.records))
        else:
            llm_indices.append(pi)

    if not llm_indices:
        return [r for r in results if r is not None]

    llm_pages = [pages[i] for i in llm_indices]

    # Pre-split large pages to prevent Step 1 output truncation
    expanded_pages: list[str] = []
    page_chunk_counts: list[int] = []
    for page in llm_pages:
        chunks = _split_pipe_table(page, step1_max_rows)
        expanded_pages.extend(chunks)
        page_chunk_counts.append(len(chunks))

    parsed_tables = list(await asyncio.gather(
        *(analyze_and_parse_async(page, model=model, fallback_model=fallback_model) for page in expanded_pages)
    ))

    # Merge Step 1 chunks back into original pages
    merged_parsed: list[baml_types.ParsedTable] = []
    idx = 0
    for count in page_chunk_counts:
        if count == 1:
            merged_parsed.append(parsed_tables[idx])
        else:
            merged_parsed.append(_merge_parsed_chunks(parsed_tables[idx : idx + count]))
        idx += count

    mapped_tables = list(await asyncio.gather(
        *(map_to_schema_async(parsed, schema, model=model, fallback_model=fallback_model) for parsed in merged_parsed)
    ))

    # Place LLM results back into the correct positions
    for li, orig_idx in enumerate(llm_indices):
        results[orig_idx] = mapped_tables[li]

    return [r for r in results if r is not None]


async def _interpret_pages_single_shot_async(
    pages: list[str],
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> list[baml_types.MappedTable]:
    """Concurrently run single-shot interpretation on multiple pages."""
    tb = _build_type_builder(schema)
    baml_schema = _to_baml_schema(schema)

    async def _one_page(page_text: str) -> baml_types.MappedTable:
        try:
            return await b_async.InterpretTable(
                page_text, baml_schema, model, {"tb": tb, "client": model}
            )
        except Exception:
            if fallback_model is None:
                raise
            log.warning("Primary model %r failed for InterpretTable, falling back to %r", model, fallback_model)
            return await b_async.InterpretTable(
                page_text, baml_schema, fallback_model, {"tb": tb, "client": fallback_model}
            )

    results = await asyncio.gather(*(_one_page(page) for page in pages))
    return list(results)


async def analyze_and_parse_async(
    compressed_text: str,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> baml_types.ParsedTable:
    """Async: analyze table structure and parse data rows.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        model: LLM to use.
        fallback_model: If set, retry with this model when ``model`` fails.
    """
    try:
        return await b_async.AnalyzeAndParseTable(compressed_text, {"client": model})
    except Exception:
        if fallback_model is None:
            raise
        log.warning("Primary model %r failed for AnalyzeAndParseTable, falling back to %r", model, fallback_model)
        return await b_async.AnalyzeAndParseTable(compressed_text, {"client": fallback_model})


async def infer_table_schema_from_image_async(
    page_image_b64: str,
    compressed_text: str,
    *,
    model: str = DEFAULT_MODEL,
) -> baml_types.InferredTableSchema:
    """Async: infer table column structure from a page image."""
    img = Image.from_base64("image/png", page_image_b64)
    return await b_async.InferTableSchemaFromImage(img, compressed_text, {"client": model})


async def analyze_and_parse_guided_async(
    compressed_text: str,
    visual_schema: baml_types.InferredTableSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> baml_types.ParsedTable:
    """Async: step 1 guided by visual schema (for garbled-header PDFs)."""
    try:
        return await b_async.AnalyzeAndParseTableGuided(
            compressed_text, visual_schema, {"client": model}
        )
    except Exception:
        if fallback_model is None:
            raise
        log.warning("Primary model %r failed for guided parse, falling back to %r", model, fallback_model)
        return await b_async.AnalyzeAndParseTableGuided(
            compressed_text, visual_schema, {"client": fallback_model}
        )


async def map_to_schema_async(
    parsed_table: baml_types.ParsedTable,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> baml_types.MappedTable:
    """Async: map parsed table to canonical schema.

    Args:
        parsed_table: Output from ``analyze_and_parse_async()``.
        schema: Canonical schema to map to.
        model: LLM to use.
        fallback_model: If set, retry with this model when ``model`` fails.
    """
    tb = _build_type_builder(schema)
    baml_schema = _to_baml_schema(schema)
    try:
        return await b_async.MapToCanonicalSchema(
            parsed_table, baml_schema, model, {"tb": tb, "client": model}
        )
    except Exception:
        if fallback_model is None:
            raise
        log.warning("Primary model %r failed for MapToCanonicalSchema, falling back to %r", model, fallback_model)
        return await b_async.MapToCanonicalSchema(
            parsed_table, baml_schema, fallback_model, {"tb": tb, "client": fallback_model}
        )


async def interpret_table_async(
    compressed_text: str,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    unpivot: bool = True,
) -> baml_types.MappedTable:
    """Async: full 2-step pipeline for a single table.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        schema: Canonical schema to map to.
        model: LLM to use for both steps.
        fallback_model: If set, retry each step with this model on failure.
        unpivot: When True (default), detect and unpivot pivoted tables
            before interpretation.
    """
    compressed_text = normalize_pipe_table(compressed_text)
    if unpivot:
        from pdf_ocr.unpivot import unpivot_pipe_table as _unpivot
        compressed_text = _unpivot(compressed_text).text
    # Try deterministic mapping first
    det = _try_deterministic(compressed_text, schema)
    if det is not None:
        return det
    parsed = await analyze_and_parse_async(compressed_text, model=model, fallback_model=fallback_model)
    result = await map_to_schema_async(parsed, schema, model=model, fallback_model=fallback_model)
    # Fix table_type: use authoritative value from step 1
    if result.metadata.table_type_inference.table_type != parsed.table_type:
        result.metadata.table_type_inference.table_type = parsed.table_type
    return result


async def interpret_tables_async(
    compressed_texts: list[str],
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    step1_max_rows: int = 40,
    unpivot: bool = True,
) -> list[baml_types.MappedTable]:
    """Async: interpret multiple pre-split tables in parallel.

    Note: ``interpret_table()`` now auto-splits multi-page text on ``\\f``
    and processes pages concurrently.  Use this function when you already
    have an event loop and want to ``await`` the result directly, or when
    you want explicit control over which text chunks to parallelize.

    Uses ``asyncio.gather()`` to run all parse calls concurrently,
    then all map calls concurrently.  Large tables (more than
    *step1_max_rows* data rows) are pre-split into chunks before Step 1
    to prevent LLM output truncation.

    Args:
        compressed_texts: List of compressed text strings, one per table/page.
        schema: Canonical schema to map to.
        model: LLM to use for all calls.
        fallback_model: If set, retry each step with this model on failure.
        step1_max_rows: Maximum data rows per step-1 LLM call (default 40).
        unpivot: When True (default), detect and unpivot pivoted tables
            before interpretation.
    """
    compressed_texts = [normalize_pipe_table(t) for t in compressed_texts]

    if unpivot:
        from pdf_ocr.unpivot import unpivot_pipe_table as _unpivot
        compressed_texts = [_unpivot(t).text for t in compressed_texts]

    # Try deterministic mapping first
    results: list[baml_types.MappedTable | None] = [None] * len(compressed_texts)
    llm_indices: list[int] = []

    for ti, text in enumerate(compressed_texts):
        det = _try_deterministic(text, schema)
        if det is not None:
            results[ti] = det
            log.info("Table %d: deterministic mapping (%d records)", ti + 1, len(det.records))
        else:
            llm_indices.append(ti)

    if not llm_indices:
        return [r for r in results if r is not None]

    llm_texts = [compressed_texts[i] for i in llm_indices]

    # Pre-split large tables to prevent Step 1 output truncation
    expanded_texts: list[str] = []
    table_chunk_counts: list[int] = []
    for text in llm_texts:
        chunks = _split_pipe_table(text, step1_max_rows)
        expanded_texts.extend(chunks)
        table_chunk_counts.append(len(chunks))

    # Step 1: parse all (expanded) tables in parallel
    parsed_tables = list(await asyncio.gather(
        *(analyze_and_parse_async(text, model=model, fallback_model=fallback_model) for text in expanded_texts)
    ))

    # Merge Step 1 chunks back into original tables
    merged_parsed: list[baml_types.ParsedTable] = []
    idx = 0
    for count in table_chunk_counts:
        if count == 1:
            merged_parsed.append(parsed_tables[idx])
        else:
            merged_parsed.append(_merge_parsed_chunks(parsed_tables[idx : idx + count]))
        idx += count

    # Step 2: map all parsed tables in parallel
    mapped_tables = list(await asyncio.gather(
        *(map_to_schema_async(parsed, schema, model=model, fallback_model=fallback_model) for parsed in merged_parsed)
    ))
    # Fix table_type: use authoritative value from step 1
    for parsed, mapped in zip(merged_parsed, mapped_tables):
        if mapped.metadata.table_type_inference.table_type != parsed.table_type:
            mapped.metadata.table_type_inference.table_type = parsed.table_type

    # Place LLM results back into the correct positions
    for li, orig_idx in enumerate(llm_indices):
        results[orig_idx] = mapped_tables[li]

    return [r for r in results if r is not None]
