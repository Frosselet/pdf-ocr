# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF-OCR is a Python project that uses BAML (BoundaryML) to extract structured data from PDF documents via LLMs. It combines PyMuPDF for PDF parsing with LLM providers (OpenAI, Anthropic) for intelligent data extraction.

## Commands

```bash
# Install dependencies
uv sync

# Regenerate BAML client after editing baml_src/ files
uv run baml-cli generate

# Run a quick test of the spatial text renderer
uv run python -c "from pdf_ocr import pdf_to_spatial_text; print(pdf_to_spatial_text('inputs/2857439.pdf'))"
```

## Environment Variables

LLM API keys must be set depending on which client is used:
- `OPENAI_API_KEY` — for GPT-5/GPT-5-mini clients
- `ANTHROPIC_API_KEY` — for Claude clients

## Architecture

### Source Package (`src/pdf_ocr/`)

All application code lives here following the standard Python src layout.

- **`__init__.py`** — Package root, re-exports public API.
- **`spatial_text.py`** — Spatial PDF-to-text renderer. Converts each PDF page into a monospace text grid preserving columns, tables, and scattered text at their correct visual positions. Public function: `pdf_to_spatial_text(pdf_path, *, pages, cluster_threshold, page_separator)`. Also exports `PageLayout` dataclass and `_extract_page_layout()` helper used by `compress.py`.
- **`compress.py`** — Compressed spatial text for LLM consumption. Classifies page regions (tables, text blocks, headings, key-value pairs, scattered) and renders them as markdown tables, flowing paragraphs, and structured key-value lines. Supports multi-row record merging for shipping stems. Public function: `compress_spatial_text(pdf_path, *, pages, cluster_threshold, page_separator, table_format, merge_multi_row, min_table_rows)`.

### BAML Source Layer (`baml_src/`)

This is where AI functions, data models, and LLM client configurations are defined using the BAML DSL. After editing these files, run `uv run baml-cli generate` to regenerate the Python client.

- **`resume.baml`** — Defines the `Resume` data model and `ExtractResume` function with prompt template. Contains test cases runnable via the BAML VSCode extension playground.
- **`clients.baml`** — LLM client configurations (OpenAI, Anthropic, plus commented-out Google, Azure, Bedrock, Vertex, Ollama). Includes composite clients: `CustomFast` (round-robin) and `OpenaiFallback` (fallback chain). Defines retry policies (constant and exponential backoff).
- **`generators.baml`** — Code generation config targeting `python/pydantic` with sync client mode. Version must match `baml-py` in `pyproject.toml`.

### Generated Client (`baml_client/`)

Auto-generated Python code — **do not edit manually**. Provides:
- `b.ExtractResume(resume_text)` — sync API (via `sync_client.py`)
- `b.stream.ExtractResume(...)` — streaming with partial results
- Pydantic models in `types.py` matching BAML class definitions

Usage: `from baml_client import b`

### PDF Inputs (`inputs/`)

Test PDF files including shipping statements and edge-case documents (multi-column, overlapping text, tiny fonts).

### Notebook (`walkthrough.ipynb`)

Exploratory notebook demonstrating PDF parsing approaches and the spatial text renderer.

## Key Constraints

- Python >=3.13 required
- Project uses src layout — all application code goes in `src/pdf_ocr/`
- BAML version in `generators.baml` must stay in sync with the `baml-py` dependency version in `pyproject.toml`
- The `baml_client/` directory is fully generated; changes belong in `baml_src/`
