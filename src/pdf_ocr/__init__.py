"""pdf-ocr: Extract structured data from PDF documents."""

from pdf_ocr.compress import compress_spatial_text
from pdf_ocr.spatial_text import pdf_to_spatial_text

__all__ = ["compress_spatial_text", "pdf_to_spatial_text"]
