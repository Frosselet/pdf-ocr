# pdf-ocr

**Structured data extraction from human-formatted documents.**

PDFs, Word files, spreadsheets, HTML — documents designed for human eyes, not machine consumption. This project extracts their tabular data into canonical schemas using a hybrid architecture: deterministic heuristics first, LLM interpretation only when needed.

The design philosophy, its evolution, and how it fits into the broader landscape of data mesh, metadata management, and agentic architectures is told in two essays:

1. **[The Data Humans Make](pubs/the-data-humans-make.md)** — On the growing gap between how we produce data and how we need to consume it.
2. **[Where the Line Is](pubs/where-the-line-is.md)** — On drawing the boundary between structure and semantics, and why that boundary matters more than which side you're on.

---

## What's Inside

This is a Python library (`src/pdf_ocr/`) with a multi-stage pipeline:

```
Document (PDF / DOCX / XLSX / PPTX / HTML)
 │
 ├─ spatial_text.py ──► Monospace grid preserving exact visual positions
 ├─ compress.py ───────► Token-efficient markdown tables (40-67% smaller)
 ├─ classify.py ───────► Keyword-based table classification
 ├─ interpret.py ──────► Deterministic-first schema mapping, LLM fallback
 ├─ serialize.py ──────► CSV / Parquet / DataFrame export
 │
 └─ contracts/ ────────► JSON data contracts (schema, aliases, enrichment)
```

Orchestration is contract-driven via `pipeline.py`: load a contract, point it at documents, get structured output. See the [notebooks](pipeline.ipynb) for end-to-end examples.

**Full technical documentation, API reference, and heuristics catalog:** [src/pdf_ocr/README.md](src/pdf_ocr/README.md)
