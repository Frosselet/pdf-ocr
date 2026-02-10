"""pdf-ocr: Extract structured data from PDF documents."""

from pdf_ocr import serialize
from pdf_ocr.compress import (
    FontValidation,
    StructuredTable,
    TableMetadata,
    compress_spatial_text,
    compress_spatial_text_structured,
)
from pdf_ocr.filter import FilterMatch, extract_table_titles, filter_pdf_by_table_titles
from pdf_ocr.interpret import (
    CanonicalSchema,
    ColumnDef,
    infer_table_schema_from_image,
    interpret_table,
    interpret_table_single_shot,
    interpret_tables_async,
    to_records,
    to_records_by_page,
)
from pdf_ocr.serialize import (
    SerializationValidationError,
    to_csv,
    to_parquet,
    to_pandas,
    to_polars,
    to_tsv,
)
from pdf_ocr.spatial_text import (
    FontSpan,
    PageLayout,
    VisualElements,
    VisualFill,
    VisualLine,
    pdf_to_spatial_text,
)

__all__ = [
    "CanonicalSchema",
    "ColumnDef",
    "FilterMatch",
    "FontSpan",
    "FontValidation",
    "PageLayout",
    "SerializationValidationError",
    "StructuredTable",
    "TableMetadata",
    "VisualElements",
    "VisualFill",
    "VisualLine",
    "compress_spatial_text",
    "compress_spatial_text_structured",
    "extract_table_titles",
    "filter_pdf_by_table_titles",
    "infer_table_schema_from_image",
    "interpret_table",
    "interpret_table_single_shot",
    "interpret_tables_async",
    "pdf_to_spatial_text",
    "serialize",
    "to_csv",
    "to_pandas",
    "to_parquet",
    "to_polars",
    "to_records",
    "to_records_by_page",
    "to_tsv",
]
