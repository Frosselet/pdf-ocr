"""Table interpretation: extract structured tabular data from compressed PDF text
and map it to a user-defined canonical schema.

Provides both sync and async APIs. Multi-page compressed text (pages joined by
form-feed ``\\f``) is automatically split and processed page-by-page, then merged
into a single result.

Sync usage::

    from pdf_ocr import compress_spatial_text, interpret_table, CanonicalSchema, ColumnDef, to_records

    compressed = compress_spatial_text("inputs/example.pdf")
    schema = CanonicalSchema(columns=[
        ColumnDef("port", "string", "Loading port name", aliases=["Port"]),
        ColumnDef("vessel_name", "string", "Name of the vessel", aliases=["Ship Name"]),
    ])
    result = interpret_table(compressed, schema, model="openai/gpt-4o")
    for r in to_records(result):
        print(r)

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

from baml_client import types as baml_types
from baml_client.sync_client import b as b_sync
from baml_client.async_client import b as b_async
from baml_client.type_builder import TypeBuilder

DEFAULT_MODEL = "openai/gpt-4o"

log = logging.getLogger(__name__)


# ─── Python-side schema definition ───────────────────────────────────────────


@dataclass
class ColumnDef:
    """Definition of a single canonical column."""

    name: str
    type: str  # "string", "int", "float", "bool", "date"
    description: str
    aliases: list[str] = field(default_factory=list)


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
            )
            for c in schema.columns
        ],
        description=schema.description,
    )


@dataclass
class _Batch:
    """A subset of a ParsedTable's data rows for batched step-2 processing."""

    page_index: int  # 0-based page index
    batch_index: int  # 0-based within this page
    data_rows: list[list[str]]
    notes: str | None  # Notes with batch-local row indices


def to_records(mapped_table: baml_types.MappedTable) -> list[dict]:
    """Convert a MappedTable into a list of plain dicts.

    Dynamic fields on MappedRecord are stored in pydantic's ``model_extra``,
    since the BAML class uses ``@@dynamic`` (``extra='allow'``).
    """
    results = []
    for rec in mapped_table.records:
        d = rec.model_dump()
        # model_dump() on extra='allow' models includes extra fields at top level
        results.append(d)
    return results


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


# ─── Batching helpers ─────────────────────────────────────────────────────────


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
        non_section = re.sub(r"[,;]\s*$", "", non_section).strip()
        non_section = re.sub(r"^[,;]\s*", "", non_section).strip()

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


def _stamp_page_number(record: baml_types.MappedRecord, page: int) -> None:
    """Write ``_page`` (1-indexed) into a MappedRecord's extra fields."""
    record.__pydantic_extra__["_page"] = page


# ─── Page splitting & merging ─────────────────────────────────────────────────


def _split_pages(compressed_text: str, page_separator: str = "\f") -> list[str]:
    """Split compressed text on page separator, dropping empty pages."""
    if page_separator not in compressed_text:
        return [compressed_text]
    return [p for p in compressed_text.split(page_separator) if p.strip()]


def _merge_mapped_tables(
    results: list[baml_types.MappedTable],
    page_numbers: list[int] | None = None,
) -> baml_types.MappedTable:
    """Combine multiple per-page/batch MappedTable results into one.

    - **records**: concatenated from all tables
    - **unmapped_columns**: union, deduplicated, preserving first-seen order
    - **mapping_notes**: non-None notes joined with ``"; "``
    - **metadata**: ``field_mappings`` from the first result; ``sections_detected``
      merged from all results (deduplicated, preserving order)

    When *page_numbers* is provided it must align 1:1 with *results*.  Each
    record in ``results[i]`` gets ``_page = page_numbers[i]`` stamped into its
    extra fields before concatenation.
    """
    all_records = []
    seen_unmapped: dict[str, None] = {}
    notes_parts: list[str] = []

    for idx, mt in enumerate(results):
        if page_numbers is not None:
            page = page_numbers[idx]
            for rec in mt.records:
                _stamp_page_number(rec, page)
        all_records.extend(mt.records)
        for col in mt.unmapped_columns:
            seen_unmapped.setdefault(col, None)
        if mt.mapping_notes is not None:
            notes_parts.append(mt.mapping_notes)

    # Merge metadata
    first_meta = results[0].metadata
    merged_sections: list[str] | None = None
    seen_sections: dict[str, None] = {}
    for mt in results:
        if mt.metadata.sections_detected:
            if merged_sections is None:
                merged_sections = []
            for s in mt.metadata.sections_detected:
                if s not in seen_sections:
                    seen_sections[s] = None
                    merged_sections.append(s)

    merged_metadata = baml_types.InterpretationMetadata(
        model=first_meta.model,
        field_mappings=first_meta.field_mappings,
        sections_detected=merged_sections,
    )

    return baml_types.MappedTable(
        records=all_records,
        unmapped_columns=list(seen_unmapped),
        mapping_notes="; ".join(notes_parts) if notes_parts else None,
        metadata=merged_metadata,
    )


# ─── Batched async orchestrator ────────────────────────────────────────────────


async def _interpret_pages_batched_async(
    pages: list[str],
    schema: CanonicalSchema,
    *,
    batch_size: int = 20,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> tuple[list[baml_types.MappedTable], list[int]]:
    """Concurrently run the 2-step pipeline with step-2 batching.

    Returns ``(mapped_results, page_numbers)`` where *page_numbers* aligns
    1:1 with *mapped_results* — multiple batches from the same page share
    the same 1-indexed page number.
    """
    # Step 1: parse all pages concurrently
    parsed_tables = await asyncio.gather(
        *(
            analyze_and_parse_async(page, model=model, fallback_model=fallback_model)
            for page in pages
        )
    )

    for pi, pt in enumerate(parsed_tables):
        log.info(
            "Step 1 page %d: parsed %d data rows",
            pi + 1,
            len(pt.data_rows),
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

    # Build page_numbers list aligned 1:1 with mapped_tables
    page_numbers = [b.page_index + 1 for b in all_batches]  # 1-indexed

    return list(mapped_tables), page_numbers


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
) -> baml_types.MappedTable:
    """Full 2-step pipeline: analyze+parse, then map to schema.

    Multi-page input (pages joined by *page_separator*) is automatically split
    and all pages are processed concurrently via ``asyncio.gather()``, then
    results are merged.

    Step 2 (schema mapping) is batched: each page's parsed rows are split into
    chunks of *batch_size* rows so the LLM can produce complete output without
    truncation.  All batches across all pages run concurrently.

    Each output record carries a ``_page`` extra field (1-indexed) indicating
    which source page it originated from.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        schema: Canonical schema to map to.
        model: LLM to use for both steps.
        fallback_model: If set, retry each step with this model on failure.
        page_separator: Delimiter between pages (default ``"\\f"``).
        batch_size: Maximum data rows per step-2 LLM call (default 20).
    """
    pages = _split_pages(compressed_text, page_separator)

    if len(pages) == 1:
        parsed = analyze_and_parse(pages[0], model=model, fallback_model=fallback_model)
        batches = _create_batches(parsed, page_index=0, batch_size=batch_size)
        if len(batches) == 1:
            # Fast path: single page, fits in one batch — sync, no event loop
            result = map_to_schema(parsed, schema, model=model, fallback_model=fallback_model)
            for rec in result.records:
                _stamp_page_number(rec, 1)
            return result
        # Single page but multiple batches — need async for concurrency
        results, page_numbers = _run_async(
            _interpret_pages_batched_async(
                pages, schema, batch_size=batch_size, model=model, fallback_model=fallback_model
            )
        )
        return _merge_mapped_tables(results, page_numbers=page_numbers)

    # Multiple pages → batched concurrent processing
    results, page_numbers = _run_async(
        _interpret_pages_batched_async(
            pages, schema, batch_size=batch_size, model=model, fallback_model=fallback_model
        )
    )
    return _merge_mapped_tables(results, page_numbers=page_numbers)


def interpret_table_single_shot(
    compressed_text: str,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
    page_separator: str = "\f",
) -> baml_types.MappedTable:
    """Single-shot: analyze, parse, and map in one LLM call.

    Multi-page input (pages joined by *page_separator*) is automatically split
    and all pages are processed concurrently via ``asyncio.gather()``, then
    results are merged.  Single-page input is processed synchronously without
    an event loop.

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
            return b_sync.InterpretTable(
                pages[0], baml_schema, model, {"tb": tb, "client": model}
            )
        except Exception:
            if fallback_model is None:
                raise
            log.warning("Primary model %r failed for InterpretTable, falling back to %r", model, fallback_model)
            return b_sync.InterpretTable(
                pages[0], baml_schema, fallback_model, {"tb": tb, "client": fallback_model}
            )

    # Multiple pages → run all concurrently via the async BAML client
    results = _run_async(
        _interpret_pages_single_shot_async(pages, schema, model=model, fallback_model=fallback_model)
    )
    return _merge_mapped_tables(results)


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
    return await map_to_schema_async(parsed, schema, model=model, fallback_model=fallback_model)


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
    return list(mapped_tables)
