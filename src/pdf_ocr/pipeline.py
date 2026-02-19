"""Pipeline orchestration helpers for contract-driven document extraction.

Four composable async helpers layered from granular to convenient:

1. :func:`compress_and_classify_async` — compress a document and classify its
   tables into output categories (DOCX vs PDF branching, thread-pool wrapping).
2. :func:`interpret_output_async` — interpret one output category end-to-end
   (year template resolution, batched interpretation, enrichment, formatting).
3. :func:`process_document_async` — convenience: full per-document pipeline
   composing the two helpers above plus report-date resolution.
4. :func:`run_pipeline_async` — top-level orchestrator: load contract, process
   all documents concurrently, merge and save results.

Usage::

    from pdf_ocr.pipeline import run_pipeline_async

    results, merged, paths, elapsed = await run_pipeline_async(
        "contracts/au_shipping_stem.json",
        ["inputs/2857439.pdf", "inputs/other.pdf"],
    )

Or use the lower-level helpers directly::

    from pdf_ocr.contracts import load_contract
    from pdf_ocr.pipeline import process_document_async

    cc = load_contract("contracts/au_shipping_stem.json")
    result = await process_document_async("inputs/2857439.pdf", cc)
    print(result.dataframes["vessels"])
"""

from __future__ import annotations

import asyncio
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pdf_ocr.contracts import (
    ContractContext,
    OutputSpec,
    _YEAR_TEMPLATE_RE,
    enrich_dataframe,
    format_dataframe,
    resolve_year_templates,
)
from pdf_ocr.interpret import (
    CanonicalSchema,
    UnpivotStrategy,
    _interpret_pages_batched_async,
    _split_pages,
)
from pdf_ocr.report_date import resolve_report_date
from pdf_ocr.serialize import to_pandas


# ─── Result type ─────────────────────────────────────────────────────────────


@dataclass
class DocumentResult:
    """Result of processing a single document through the full pipeline.

    Attributes:
        doc_path: Path to the source document.
        report_date: Resolved report date string (empty if not configured).
        compressed_by_category: Compressed data keyed by output name.
            Values are ``str`` (PDF pipe-table text) or
            ``list[tuple[str, dict]]`` (DOCX markdown + metadata).
        dataframes: Interpreted DataFrames keyed by output name.
    """

    doc_path: str
    report_date: str
    compressed_by_category: dict[str, str | list[tuple[str, dict]]]
    dataframes: dict[str, pd.DataFrame]


# ─── Layer 1: Compress & classify ────────────────────────────────────────────


async def compress_and_classify_async(
    doc_path: str,
    categories: dict[str, list[str]],
    output_specs: dict[str, OutputSpec],
    *,
    refine_headers: bool = True,
) -> dict[str, str | list[tuple[str, dict]]]:
    """Compress a document and classify its tables into output categories.

    Encapsulates DOCX vs PDF detection, single-category optimization,
    and ``asyncio.to_thread()`` wrapping for CPU-bound PDF compression.

    Returns:
        ``{output_name: compressed_data}`` where values are ``str``
        (PDF pipe-table text) or ``list[tuple[str, dict]]`` (DOCX).
    """
    from pdf_ocr.classify import classify_tables
    from pdf_ocr.compress import compress_spatial_text, compress_spatial_text_structured
    from pdf_ocr.docx_extractor import (
        classify_docx_tables,
        compress_docx_tables,
    )

    is_docx = doc_path.lower().endswith((".docx", ".doc"))
    single_category = len(categories) == 1
    result: dict[str, str | list[tuple[str, dict]]] = {}

    if is_docx:
        classes = classify_docx_tables(doc_path, categories)
        for out_name, spec in output_specs.items():
            cat = spec.category
            indices = [c["index"] for c in classes if c["category"] == cat]
            if indices:
                result[out_name] = compress_docx_tables(
                    doc_path, table_indices=indices
                )
    else:
        compressed_text = await asyncio.to_thread(
            compress_spatial_text, doc_path, refine_headers=refine_headers,
        )

        if single_category:
            for out_name in output_specs:
                result[out_name] = compressed_text
        else:
            structured = await asyncio.to_thread(
                compress_spatial_text_structured, doc_path,
            )
            tuples = [t.to_compressed() for t in structured]
            classes = classify_tables(tuples, categories) if tuples else []
            for out_name, spec in output_specs.items():
                cat = spec.category
                if any(c["category"] == cat for c in classes):
                    result[out_name] = compressed_text

    return result


# ─── Layer 2: Interpret one output ───────────────────────────────────────────


async def interpret_output_async(
    data: str | list[tuple[str, dict]],
    output_spec: OutputSpec,
    *,
    model: str = "openai/gpt-4o",
    unpivot: UnpivotStrategy | bool = True,
    report_date: str = "",
    pivot_years: list[str] | None = None,
) -> pd.DataFrame:
    """Interpret compressed data for one output category, returning a DataFrame.

    Deep-copies the schema before mutating aliases (no caller side-effects).
    Handles DOCX (per-table enrichment) vs PDF (whole-DataFrame enrichment).

    Args:
        data: Compressed table data — ``str`` for PDF, ``list[tuple[str, dict]]``
            for DOCX.
        output_spec: Output specification from the contract.
        model: LLM model identifier.
        unpivot: Unpivot strategy.
        report_date: Resolved report date string for enrichment.
        pivot_years: Document-extracted years for ``{YYYY}`` template resolution.
    """
    # Deep-copy schema to avoid mutating caller's aliases
    schema = copy.deepcopy(output_spec.schema)

    # Resolve {YYYY} alias templates
    if pivot_years is not None:
        has_templates = any(
            _YEAR_TEMPLATE_RE.search(a)
            for col in schema.columns for a in col.aliases
        )
        if has_templates:
            for col in schema.columns:
                col.aliases = resolve_year_templates(col.aliases, pivot_years)

    # Also resolve {YYYY} templates in enrichment constant values
    enrichment = copy.deepcopy(output_spec.enrichment)
    if pivot_years:
        for col_name, spec in enrichment.items():
            if spec.get("source") == "constant" and isinstance(spec.get("value"), str):
                resolved = resolve_year_templates([spec["value"]], pivot_years)
                if resolved:
                    spec["value"] = resolved[0]

    is_docx = isinstance(data, list)

    if is_docx:
        texts = [md for md, _ in data]
        page_results = await _interpret_pages_batched_async(
            texts, schema, model=model, unpivot=unpivot,
        )
        frames = []
        for i, (md, meta) in enumerate(data):
            mapped = page_results.get(i + 1)  # 1-indexed page numbers
            if mapped is None:
                continue
            df = to_pandas(mapped, schema)
            df = enrich_dataframe(
                df, enrichment,
                title=meta.get("title"), report_date=report_date,
            )
            frames.append(df)
        df_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        pages = _split_pages(data, "\f")
        result = await _interpret_pages_batched_async(
            pages, schema, model=model, unpivot=unpivot,
        )
        df_out = to_pandas(result, schema)
        df_out = enrich_dataframe(
            df_out, enrichment, report_date=report_date,
        )

    df_out = format_dataframe(df_out, output_spec.col_specs)
    col_order = [c["name"] for c in output_spec.col_specs if c["name"] in df_out.columns]
    if col_order:
        df_out = df_out[col_order]
    return df_out


# ─── Layer 3: Full document pipeline ─────────────────────────────────────────


async def process_document_async(
    doc_path: str,
    cc: ContractContext,
    *,
    refine_headers: bool = True,
) -> DocumentResult:
    """Compress, classify, interpret, and enrich a single document.

    Composes :func:`compress_and_classify_async` and
    :func:`interpret_output_async` with report-date resolution.
    """
    # Resolve report date
    if cc.report_date_config:
        report_date = await resolve_report_date(cc.report_date_config, doc_path=doc_path)
    else:
        report_date = ""

    # Compress & classify
    compressed_by_category = await compress_and_classify_async(
        doc_path, cc.categories, cc.outputs, refine_headers=refine_headers,
    )

    # Extract pivot years for DOCX (PDF: not yet implemented)
    is_docx = doc_path.lower().endswith((".docx", ".doc"))
    if is_docx:
        from pdf_ocr.docx_extractor import extract_pivot_values

        all_years: set[str] = set()
        for data in compressed_by_category.values():
            if isinstance(data, list):
                for md, _ in data:
                    all_years.update(extract_pivot_values(md))
        pivot_years = sorted(all_years)
    else:
        pivot_years = []

    # Interpret all output categories concurrently
    tasks = [
        _interpret_one(out_name, data, cc, report_date=report_date, pivot_years=pivot_years)
        for out_name, data in compressed_by_category.items()
    ]
    if tasks:
        results = await asyncio.gather(*tasks)
        dataframes = dict(results)
    else:
        dataframes = {}

    return DocumentResult(
        doc_path=doc_path,
        report_date=report_date,
        compressed_by_category=compressed_by_category,
        dataframes=dataframes,
    )


async def _interpret_one(
    out_name: str,
    data: str | list[tuple[str, dict]],
    cc: ContractContext,
    *,
    report_date: str,
    pivot_years: list[str],
) -> tuple[str, pd.DataFrame]:
    """Interpret a single output category — returns ``(name, DataFrame)``."""
    df = await interpret_output_async(
        data,
        cc.outputs[out_name],
        model=cc.model,
        unpivot=cc.unpivot,
        report_date=report_date,
        pivot_years=pivot_years,
    )
    return out_name, df


# ─── Layer 4: Multi-document orchestration ───────────────────────────────────


def save(
    results: list[DocumentResult],
    output_dir: str | Path = "outputs",
    *,
    filenames: dict[str, str] | None = None,
) -> tuple[dict[str, list[pd.DataFrame]], dict[str, Path]]:
    """Merge DataFrames across documents and write each output to file.

    Args:
        results: List of :class:`DocumentResult` from :func:`process_document_async`.
        output_dir: Directory for output files (created if needed).
        filenames: Optional mapping of output name → filename (e.g.
            ``{"vessels": "shipping_stem.csv"}``).  The file extension determines
            the format: ``.csv`` → CSV, ``.tsv`` → TSV, anything else → Parquet.
            When *None* or when an output name has no entry, defaults to
            ``{out_name}.parquet``.

    Returns:
        ``(merged, paths)`` — *merged* maps output name → list of DataFrames,
        *paths* maps output name → written file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list[pd.DataFrame]] = {}

    for result in results:
        for out_name, df in result.dataframes.items():
            if out_name not in merged:
                merged[out_name] = []
            merged[out_name].append(df)

    paths: dict[str, Path] = {}
    for out_name, frames in merged.items():
        df = pd.concat(frames, ignore_index=True)
        fname = filenames.get(out_name, f"{out_name}.parquet") if filenames else f"{out_name}.parquet"
        path = output_dir / fname
        if path.suffix == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix == ".tsv":
            df.to_csv(path, index=False, sep="\t")
        else:
            df.to_parquet(path, index=False)
        paths[out_name] = path
        print(f"  {out_name}: {path} ({len(df)} rows)")

    return merged, paths


async def run_pipeline_async(
    contract_path: str | Path,
    doc_paths: list[str | Path],
    output_dir: str | Path = "outputs",
    *,
    refine_headers: bool = True,
) -> tuple[list[DocumentResult], dict[str, list[pd.DataFrame]], dict[str, Path], float]:
    """Orchestrate the full pipeline with document-level concurrency.

    Loads the contract, processes all documents in parallel via
    :func:`process_document_async`, then merges and saves results.

    Args:
        contract_path: Path to the JSON data contract.
        doc_paths: Paths to the documents to process.
        output_dir: Directory for output files (format driven by contract).
        refine_headers: Whether to refine PDF headers (passed through).

    Returns:
        ``(results, merged, paths, elapsed)`` — *results* is a list of
        :class:`DocumentResult`, *merged*/*paths* from :func:`save`,
        *elapsed* is wall-clock seconds.
    """
    import time

    from pdf_ocr.contracts import load_contract

    t0 = time.perf_counter()

    print("PREPARE")
    cc = load_contract(contract_path)
    filenames = {name: spec.filename for name, spec in cc.outputs.items() if spec.filename}
    print(f"  Contract: {cc.provider}, Model: {cc.model}, Outputs: {list(cc.outputs.keys())}")

    print("\nPROCESS (async — compress + classify + interpret per document)")
    results = await asyncio.gather(
        *[process_document_async(str(p), cc, refine_headers=refine_headers) for p in doc_paths]
    )
    for r in results:
        cats = list(r.dataframes.keys())
        rows = sum(len(df) for df in r.dataframes.values())
        print(f"  {Path(r.doc_path).name[:50]}: {cats} → {rows} records")

    print("\nSAVE")
    merged, paths = save(results, output_dir, filenames=filenames)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return results, merged, paths, elapsed
