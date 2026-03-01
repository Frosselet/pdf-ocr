"""PDF table extraction facade.

Thin wrapper around :mod:`docpact.compress` that aligns the PDF extraction
API with the naming convention used by other format extractors
(``extract_tables_from_xlsx``, ``extract_tables_from_docx``, etc.).

All logic lives in ``compress.py``; this module only re-exports under
the canonical names.
"""

from __future__ import annotations

from pathlib import Path

from docpact.compress import StructuredTable, compress_spatial_text_structured


def extract_tables_from_pdf(
    pdf_path: str | bytes | Path,
    *,
    pages: list[int] | None = None,
    cluster_threshold: float = 2.0,
    merge_multi_row: bool = True,
    min_table_rows: int = 3,
    extract_visual: bool = True,
) -> list[StructuredTable]:
    """Extract structured tables from a PDF file.

    Delegates to :func:`docpact.compress.compress_spatial_text_structured`.

    Args:
        pdf_path: Path to PDF file (str or Path), or raw PDF bytes.
        pages: Optional list of 0-based page indices to process.
            If None, all pages are processed.
        cluster_threshold: Maximum y-distance (in points) to merge into
            the same text row. Default 2.0.
        merge_multi_row: If True, detect and merge multi-row records in
            shipping-stem-style tables.
        min_table_rows: Minimum data rows for a region to be considered
            a table (default 3).
        extract_visual: If True, extract visual elements (fills, lines)
            for header detection heuristics.

    Returns:
        List of :class:`~docpact.compress.StructuredTable` objects.
    """
    return compress_spatial_text_structured(
        pdf_path,
        pages=pages,
        cluster_threshold=cluster_threshold,
        merge_multi_row=merge_multi_row,
        min_table_rows=min_table_rows,
        extract_visual=extract_visual,
    )


def pdf_to_markdown(
    pdf_path: str | bytes | Path,
    *,
    pages: list[int] | None = None,
    cluster_threshold: float = 2.0,
    merge_multi_row: bool = True,
    min_table_rows: int = 3,
) -> str:
    """Extract tables from PDF and render as pipe-table markdown.

    Convenience function mirroring ``xlsx_to_markdown()`` and
    ``pptx_to_markdown()``.

    Args:
        pdf_path: Path to PDF file (str or Path), or raw PDF bytes.
        pages: Optional list of 0-based page indices to process.
        cluster_threshold: Maximum y-distance for row merging.
        merge_multi_row: Merge multi-row shipping-stem records.
        min_table_rows: Minimum data rows per table.

    Returns:
        Pipe-table markdown string with ``## Page N`` headers.
    """
    tables = extract_tables_from_pdf(
        pdf_path,
        pages=pages,
        cluster_threshold=cluster_threshold,
        merge_multi_row=merge_multi_row,
        min_table_rows=min_table_rows,
        extract_visual=False,
    )

    parts: list[str] = []
    for table in tables:
        meta = table.metadata or {}
        page = meta.get("page_number", 0)
        parts.append(f"## Page {page + 1}\n")

        md, _ = table.to_compressed()
        parts.append(md)
        parts.append("")  # blank line between tables

    return "\n".join(parts)
