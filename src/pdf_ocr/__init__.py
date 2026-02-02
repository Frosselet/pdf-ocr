"""pdf-ocr: Extract structured data from PDF documents."""

from pdf_ocr.compress import compress_spatial_text
from pdf_ocr.interpret import (
    CanonicalSchema,
    ColumnDef,
    interpret_table,
    interpret_table_single_shot,
    interpret_tables_async,
    to_records,
)
from pdf_ocr.spatial_text import pdf_to_spatial_text

__all__ = [
    "CanonicalSchema",
    "ColumnDef",
    "compress_spatial_text",
    "interpret_table",
    "interpret_table_single_shot",
    "interpret_tables_async",
    "pdf_to_spatial_text",
    "to_records",
]
