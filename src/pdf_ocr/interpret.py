"""Table interpretation: extract structured tabular data from compressed PDF text
and map it to a user-defined canonical schema.

Provides both sync and async APIs. Each call receives one table/page — the caller
splits multi-page compressed text before calling.

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

Async usage (parallel across pre-split tables)::

    import asyncio
    from pdf_ocr.interpret import interpret_tables_async, CanonicalSchema, ColumnDef

    tables = await interpret_tables_async([page1, page2, page3], schema, model="openai/gpt-4o")
"""

from __future__ import annotations

import asyncio
import logging
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
) -> baml_types.MappedTable:
    """Full 2-step pipeline: analyze+parse, then map to schema.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        schema: Canonical schema to map to.
        model: LLM to use for both steps.
        fallback_model: If set, retry each step with this model on failure.
    """
    parsed = analyze_and_parse(compressed_text, model=model, fallback_model=fallback_model)
    return map_to_schema(parsed, schema, model=model, fallback_model=fallback_model)


def interpret_table_single_shot(
    compressed_text: str,
    schema: CanonicalSchema,
    *,
    model: str = DEFAULT_MODEL,
    fallback_model: str | None = None,
) -> baml_types.MappedTable:
    """Single-shot: analyze, parse, and map in one LLM call.

    Args:
        compressed_text: Compressed text from ``compress_spatial_text()``.
        schema: Canonical schema to map to.
        model: LLM to use.
        fallback_model: If set, retry with this model when ``model`` fails.
    """
    tb = _build_type_builder(schema)
    baml_schema = _to_baml_schema(schema)
    try:
        return b_sync.InterpretTable(
            compressed_text, baml_schema, model, {"tb": tb, "client": model}
        )
    except Exception:
        if fallback_model is None:
            raise
        log.warning("Primary model %r failed for InterpretTable, falling back to %r", model, fallback_model)
        return b_sync.InterpretTable(
            compressed_text, baml_schema, fallback_model, {"tb": tb, "client": fallback_model}
        )


# ─── Async API ────────────────────────────────────────────────────────────────


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
