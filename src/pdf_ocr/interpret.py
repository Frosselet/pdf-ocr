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
    """User-defined canonical schema for table mapping."""

    columns: list[ColumnDef]
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
                ]
            }
        """
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
        return cls(columns=columns, description=data.get("description"))


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


# ─── Batched async orchestrator ────────────────────────────────────────────────


async def _interpret_pages_batched_async(
    pages: list[str],
    schema: CanonicalSchema,
    *,
    batch_size: int = 20,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    page_images: list[str] | None = None,
    vision_model: str | None = None,
) -> dict[int, baml_types.MappedTable]:
    """Concurrently run the 2-step pipeline with step-2 batching.

    Returns a dict keyed by 1-indexed page number, where each value is a
    ``MappedTable`` containing all records for that page (batches merged).

    When *page_images* is provided, a vision-based schema inference step
    (step 0) runs first, and step 1 uses the guided variant that splits
    concatenated values based on the inferred column structure.
    """
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

    # Step 1: parse all pages concurrently
    if visual_schemas is not None:
        parsed_tables = list(await asyncio.gather(
            *(
                analyze_and_parse_guided_async(
                    page, vs, model=model, fallback_model=fallback_model
                )
                for page, vs in zip(pages, visual_schemas)
            )
        ))
    else:
        parsed_tables = list(await asyncio.gather(
            *(
                analyze_and_parse_async(page, model=model, fallback_model=fallback_model)
                for page in pages
            )
        ))

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
                        pages[pi], model=model, fallback_model=fallback_model
                    )
                    parsed_tables[pi] = fallback_parsed
                    log.info(
                        "Step 1 page %d: fallback produced %d cells",
                        pi + 1, len(fallback_parsed.data_rows[0]) if fallback_parsed.data_rows else 0,
                    )

    # Validate section boundaries against compressed text
    for pi, pt in enumerate(parsed_tables):
        corrected = _validate_section_boundaries(
            pt.notes, len(pt.data_rows), pages[pi]
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
        len(pages),
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

    # Group batch results by page
    page_numbers = [b.page_index + 1 for b in all_batches]  # 1-indexed
    return _group_by_page(list(mapped_tables), page_numbers)


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
    pdf_path: str | bytes | Path | None = None,
    vision_model: str | None = None,
) -> dict[int, baml_types.MappedTable]:
    """Full 2-step pipeline: analyze+parse, then map to schema.

    Multi-page input (pages joined by *page_separator*) is automatically split
    and all pages are processed concurrently via ``asyncio.gather()``.

    Returns a dict keyed by 1-indexed page number, where each value is a
    ``MappedTable`` containing that page's records, unmapped columns, mapping
    notes, and metadata.  Records contain only canonical schema fields.

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
        pdf_path: Path to original PDF (str, Path, or bytes). When provided,
            enables vision-based schema inference (step 0) and guided parsing
            (step 1).
        vision_model: LLM to use for vision step 0. Defaults to *model*.
    """
    pages = _split_pages(compressed_text, page_separator)

    # Render page images when vision is enabled
    page_images: list[str] | None = None
    if pdf_path is not None:
        log.info("Vision enabled: rendering %d page(s)", len(pages))
        page_images = _render_page_images(pdf_path, pages=list(range(len(pages))))

    if len(pages) == 1 and page_images is None:
        # Fast path: single page, no vision — try fully sync
        parsed = analyze_and_parse(pages[0], model=model, fallback_model=fallback_model)
        # Validate section boundaries against compressed text
        corrected = _validate_section_boundaries(
            parsed.notes, len(parsed.data_rows), pages[0]
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
        if len(batches) == 1:
            # Fast path: single page, fits in one batch — sync, no event loop
            result = map_to_schema(parsed, schema, model=model, fallback_model=fallback_model)
            # Fix table_type: use authoritative value from step 1
            if result.metadata.table_type_inference.table_type != parsed.table_type:
                result.metadata.table_type_inference.table_type = parsed.table_type
            return {1: result}
        # Single page but multiple batches — need async for concurrency
        return _run_async(
            _interpret_pages_batched_async(
                pages, schema, batch_size=batch_size, model=model, fallback_model=fallback_model
            )
        )

    # Multiple pages or vision enabled → batched concurrent processing
    return _run_async(
        _interpret_pages_batched_async(
            pages, schema, batch_size=batch_size, model=model, fallback_model=fallback_model,
            page_images=page_images, vision_model=vision_model,
        )
    )


def interpret_table_single_shot(
    compressed_text: str,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    page_separator: str = "\f",
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
    """
    pages = _split_pages(compressed_text, page_separator)

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
) -> list[baml_types.MappedTable]:
    """Concurrently run the 2-step pipeline on multiple pages.

    Step 1: parse all pages in parallel.
    Step 2: map all parsed tables in parallel (one call per page — no batching).

    .. note::

       This function does **not** batch step 2.  For large pages the LLM may
       truncate output.  :func:`interpret_table` and
       :func:`_interpret_pages_batched_async` add step-2 batching.
       :func:`interpret_tables_async` continues to use this function for
       callers who want explicit control.
    """
    parsed_tables = await asyncio.gather(
        *(analyze_and_parse_async(page, model=model, fallback_model=fallback_model) for page in pages)
    )
    mapped_tables = await asyncio.gather(
        *(map_to_schema_async(parsed, schema, model=model, fallback_model=fallback_model) for parsed in parsed_tables)
    )
    return list(mapped_tables)


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
) -> baml_types.MappedTable:
    """Async: full 2-step pipeline for a single table.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        schema: Canonical schema to map to.
        model: LLM to use for both steps.
        fallback_model: If set, retry each step with this model on failure.
    """
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
) -> list[baml_types.MappedTable]:
    """Async: interpret multiple pre-split tables in parallel.

    Note: ``interpret_table()`` now auto-splits multi-page text on ``\\f``
    and processes pages concurrently.  Use this function when you already
    have an event loop and want to ``await`` the result directly, or when
    you want explicit control over which text chunks to parallelize.

    Uses ``asyncio.gather()`` to run all parse calls concurrently,
    then all map calls concurrently.

    Args:
        compressed_texts: List of compressed text strings, one per table/page.
        schema: Canonical schema to map to.
        model: LLM to use for all calls.
        fallback_model: If set, retry each step with this model on failure.
    """
    # Step 1: parse all tables in parallel
    parsed_tables = await asyncio.gather(
        *(analyze_and_parse_async(text, model=model, fallback_model=fallback_model) for text in compressed_texts)
    )
    # Step 2: map all parsed tables in parallel
    mapped_tables = await asyncio.gather(
        *(map_to_schema_async(parsed, schema, model=model, fallback_model=fallback_model) for parsed in parsed_tables)
    )
    # Fix table_type: use authoritative value from step 1
    for parsed, mapped in zip(parsed_tables, mapped_tables):
        if mapped.metadata.table_type_inference.table_type != parsed.table_type:
            mapped.metadata.table_type_inference.table_type = parsed.table_type
    return list(mapped_tables)
