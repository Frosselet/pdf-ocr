"""docpact: Extract structured data from PDF and other document formats."""

from docpact import serialize
from docpact.compress import (
    FontValidation,
    StructuredTable,
    TableMetadata,
    compress_spatial_text,
    compress_spatial_text_structured,
)
from docpact.filter import FilterMatch, extract_table_titles, filter_pdf_by_table_titles
from docpact.heuristics import (
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
from docpact.retrieval import (
    RetrievedMetadata,
    SearchExtractResult,
    ValidationResult,
    apply_fallbacks,
    quick_scan,
    search_and_extract,
    validate_metadata,
)
from docpact.contracts import (
    ContractContext,
    OutputSpec,
    SemanticColumnSpec,
    load_contract,
    resolve_year_templates,
    enrich_dataframe,
    format_dataframe,
)
from docpact.pipeline import (
    DocumentResult,
    compress_and_classify_async,
    interpret_output_async,
    process_document_async,
    run_pipeline_async,
    save,
)
from docpact.report_date import ReportDateConfig, resolve_report_date
from docpact.semantics import (
    PreFlightFinding,
    PreFlightReport,
    SemanticContext,
    ValidationFinding,
    ValidationReport,
    preflight_check,
    validate_output,
)
from docpact.interpret import (
    CanonicalSchema,
    ColumnDef,
    UnpivotStrategy,
    infer_table_schema_from_image,
    interpret_table,
    interpret_table_single_shot,
    interpret_tables,
    interpret_tables_async,
    to_records,
    to_records_by_page,
)
from docpact.serialize import (
    SerializationValidationError,
    to_csv,
    to_parquet,
    to_pandas,
    to_polars,
    to_tsv,
)
from docpact.spatial_text import (
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
    """Extract tables from XLSX files. Requires: pip install docpact[xlsx]"""
    from docpact.xlsx_extractor import extract_tables_from_xlsx as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_docx(*args, **kwargs):
    """Extract tables from DOCX files. Requires: pip install docpact[docx]"""
    from docpact.docx_extractor import extract_tables_from_docx as _extract
    return _extract(*args, **kwargs)

def compress_docx_tables(*args, **kwargs):
    """Compress DOCX tables to pipe-table markdown. Requires: pip install docpact[docx]"""
    from docpact.docx_extractor import compress_docx_tables as _compress
    return _compress(*args, **kwargs)

def classify_tables(*args, **kwargs):
    """Classify tables from compressed pipe-table markdown + metadata tuples."""
    from docpact.classify import classify_tables as _classify
    return _classify(*args, **kwargs)

def classify_docx_tables(*args, **kwargs):
    """Classify tables in a DOCX file. Requires: pip install docpact[docx]"""
    from docpact.docx_extractor import classify_docx_tables as _classify
    return _classify(*args, **kwargs)

def extract_pivot_values(*args, **kwargs):
    """Extract pivot values (years) from compressed DOCX markdown headers."""
    from docpact.docx_extractor import extract_pivot_values as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_pptx(*args, **kwargs):
    """Extract tables from PPTX files. Requires: pip install docpact[pptx]"""
    from docpact.pptx_extractor import extract_tables_from_pptx as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_pdf(*args, **kwargs):
    """Extract structured tables from PDF files."""
    from docpact.pdf_extractor import extract_tables_from_pdf as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_html(*args, **kwargs):
    """Extract tables from HTML. Requires: pip install docpact[html]"""
    from docpact.html_extractor import extract_tables_from_html as _extract
    return _extract(*args, **kwargs)

def normalize_pipe_table(*args, **kwargs):
    """Normalize pipe-table markdown text (whitespace, smart quotes, dashes, zero-width chars)."""
    from docpact.normalize import normalize_pipe_table as _normalize
    return _normalize(*args, **kwargs)

def unpivot_pipe_table(*args, **kwargs):
    """Detect and unpivot a pivoted pipe-table. Schema-agnostic."""
    from docpact.unpivot import unpivot_pipe_table as _unpivot
    return _unpivot(*args, **kwargs)

def detect_pivot(*args, **kwargs):
    """Detect pivot structure in a pipe-table without transforming."""
    from docpact.unpivot import detect_pivot as _detect
    return _detect(*args, **kwargs)

# Unified extractors (auto-detect format, support both old and new formats)
def extract_tables_from_excel(*args, **kwargs):
    """Extract tables from Excel files (XLSX or XLS).

    Requires: pip install docpact[excel] (for both formats)
    Or: pip install docpact[xlsx] for XLSX only
    Or: pip install docpact[xls] for XLS only
    """
    from docpact.xlsx_extractor import extract_tables_from_excel as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_word(*args, **kwargs):
    """Extract tables from Word files (DOCX or DOC).

    Requires: pip install docpact[docx]
    Note: DOC files also require LibreOffice for conversion.
    """
    from docpact.docx_extractor import extract_tables_from_word as _extract
    return _extract(*args, **kwargs)

def extract_tables_from_powerpoint(*args, **kwargs):
    """Extract tables from PowerPoint files (PPTX or PPT).

    Requires: pip install docpact[pptx]
    Note: PPT files also require LibreOffice for conversion.
    """
    from docpact.pptx_extractor import extract_tables_from_powerpoint as _extract
    return _extract(*args, **kwargs)


__all__ = [
    # Contract helpers
    "ContractContext",
    "OutputSpec",
    "SemanticColumnSpec",
    "load_contract",
    "resolve_year_templates",
    "enrich_dataframe",
    "format_dataframe",
    # Semantic awareness
    "PreFlightFinding",
    "PreFlightReport",
    "SemanticContext",
    "ValidationFinding",
    "ValidationReport",
    "preflight_check",
    "validate_output",
    # Pipeline orchestration
    "DocumentResult",
    "compress_and_classify_async",
    "interpret_output_async",
    "process_document_async",
    "run_pipeline_async",
    "save",
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
    "UnpivotStrategy",
    "ValidationResult",
    "VisualElements",
    "VisualFill",
    "VisualLine",
    "ReportDateConfig",
    # Report date
    "resolve_report_date",
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
    # Normalization
    "normalize_pipe_table",
    # Unpivoting
    "detect_pivot",
    "unpivot_pipe_table",
    # Multi-format extractors (specific formats)
    "extract_tables_from_pdf",
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
