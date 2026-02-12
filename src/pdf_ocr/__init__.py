"""pdf-ocr: Extract structured data from PDF and other document formats."""

from pdf_ocr import serialize
from pdf_ocr.compress import (
    FontValidation,
    StructuredTable,
    TableMetadata,
    compress_spatial_text,
    compress_spatial_text_structured,
)
from pdf_ocr.filter import FilterMatch, extract_table_titles, filter_pdf_by_table_titles
from pdf_ocr.heuristics import (
    CellType,
    FallbackStrategy,
    MetadataCategory,
    MetadataFieldDef,
    SearchZone,
    StructuredTable as GenericStructuredTable,
    detect_cell_type,
    detect_column_types,
    detect_footnote_markers,
    detect_temporal_patterns,
    detect_unit_patterns,
    estimate_header_rows,
    is_header_type_pattern,
)
from pdf_ocr.retrieval import (
    RetrievedMetadata,
    SearchExtractResult,
    ValidationResult,
    apply_fallbacks,
    quick_scan,
    search_and_extract,
    validate_metadata,
)
from pdf_ocr.interpret import (
    CanonicalSchema,
    ColumnDef,
    infer_table_schema_from_image,
    interpret_table,
    interpret_table_single_shot,
    interpret_tables,
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

# Multi-format extractors (lazy imports to avoid optional dependency issues)
# Modern formats
def extract_tables_from_xlsx(*args, **kwargs):
    """Extract tables from XLSX files. Requires: pip install pdf-ocr[xlsx]"""
    from pdf_ocr.xlsx_extractor import extract_tables_from_xlsx as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_docx(*args, **kwargs):
    """Extract tables from DOCX files. Requires: pip install pdf-ocr[docx]"""
    from pdf_ocr.docx_extractor import extract_tables_from_docx as _extract
    return _extract(*args, **kwargs)

def compress_docx_tables(*args, **kwargs):
    """Compress DOCX tables to pipe-table markdown. Requires: pip install pdf-ocr[docx]"""
    from pdf_ocr.docx_extractor import compress_docx_tables as _compress
    return _compress(*args, **kwargs)

def classify_tables(*args, **kwargs):
    """Classify tables from compressed pipe-table markdown + metadata tuples."""
    from pdf_ocr.classify import classify_tables as _classify
    return _classify(*args, **kwargs)

def classify_docx_tables(*args, **kwargs):
    """Classify tables in a DOCX file. Requires: pip install pdf-ocr[docx]"""
    from pdf_ocr.docx_extractor import classify_docx_tables as _classify
    return _classify(*args, **kwargs)

def extract_pivot_values(*args, **kwargs):
    """Extract pivot values (years) from compressed DOCX markdown headers."""
    from pdf_ocr.docx_extractor import extract_pivot_values as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_pptx(*args, **kwargs):
    """Extract tables from PPTX files. Requires: pip install pdf-ocr[pptx]"""
    from pdf_ocr.pptx_extractor import extract_tables_from_pptx as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_html(*args, **kwargs):
    """Extract tables from HTML. Requires: pip install pdf-ocr[html]"""
    from pdf_ocr.html_extractor import extract_tables_from_html as _extract
    return _extract(*args, **kwargs)

# Unified extractors (auto-detect format, support both old and new formats)
def extract_tables_from_excel(*args, **kwargs):
    """Extract tables from Excel files (XLSX or XLS).

    Requires: pip install pdf-ocr[excel] (for both formats)
    Or: pip install pdf-ocr[xlsx] for XLSX only
    Or: pip install pdf-ocr[xls] for XLS only
    """
    from pdf_ocr.xlsx_extractor import extract_tables_from_excel as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_word(*args, **kwargs):
    """Extract tables from Word files (DOCX or DOC).

    Requires: pip install pdf-ocr[docx]
    Note: DOC files also require LibreOffice for conversion.
    """
    from pdf_ocr.docx_extractor import extract_tables_from_word as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_powerpoint(*args, **kwargs):
    """Extract tables from PowerPoint files (PPTX or PPT).

    Requires: pip install pdf-ocr[pptx]
    Note: PPT files also require LibreOffice for conversion.
    """
    from pdf_ocr.pptx_extractor import extract_tables_from_powerpoint as _extract
    return _extract(*args, **kwargs)


__all__ = [
    # Core types
    "CanonicalSchema",
    "CellType",
    "ColumnDef",
    "FallbackStrategy",
    "FilterMatch",
    "FontSpan",
    "FontValidation",
    "GenericStructuredTable",
    "MetadataCategory",
    "MetadataFieldDef",
    "PageLayout",
    "RetrievedMetadata",
    "SearchExtractResult",
    "SearchZone",
    "SerializationValidationError",
    "StructuredTable",
    "TableMetadata",
    "ValidationResult",
    "VisualElements",
    "VisualFill",
    "VisualLine",
    # PDF functions
    "apply_fallbacks",
    "compress_spatial_text",
    "compress_spatial_text_structured",
    "extract_table_titles",
    "filter_pdf_by_table_titles",
    "infer_table_schema_from_image",
    "interpret_table",
    "interpret_table_single_shot",
    "interpret_tables",
    "interpret_tables_async",
    "pdf_to_spatial_text",
    "quick_scan",
    "search_and_extract",
    "serialize",
    "to_csv",
    "to_pandas",
    "to_parquet",
    "to_polars",
    "to_records",
    "to_records_by_page",
    "to_tsv",
    "validate_metadata",
    # Shared heuristics
    "detect_cell_type",
    "detect_column_types",
    "detect_footnote_markers",
    "detect_temporal_patterns",
    "detect_unit_patterns",
    "estimate_header_rows",
    "is_header_type_pattern",
    # Multi-format extractors (specific formats)
    "classify_tables",
    "classify_docx_tables",
    "compress_docx_tables",
    "extract_pivot_values",
    "extract_tables_from_docx",
    "extract_tables_from_html",
    "extract_tables_from_pptx",
    "extract_tables_from_xlsx",
    # Unified extractors (auto-detect old/new formats)
    "extract_tables_from_excel",
    "extract_tables_from_powerpoint",
    "extract_tables_from_word",
]
