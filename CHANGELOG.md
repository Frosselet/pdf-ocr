# Changelog

Milestone-level progress for docpact, the CK-IR reference implementation.
One entry per significant capability or phase transition.

## Phase 1: Semantic Contracts (in progress)

### 2026-02-26 — Extractor pattern alignment + documentation
- `pdf_extractor.py` facade: `extract_tables_from_pdf()`, `pdf_to_markdown()`
- 11 per-module `.md` documentation files (every `.py` now has a `.md`)
- Lazy import + `__all__` entry for `extract_tables_from_pdf`

### 2026-02-26 — Semantic-aware pipeline
- `SemanticContext` dataclass for pre-resolved ontology data
- Pre-flight header checks (`preflight_check()`)
- Post-extraction value validation (`validate_output()`)
- Runtime alias enrichment from `SemanticContext`
- `build_semantic_context()` bridge in contract-semantics
- `build-context` CLI command
- 47 new tests (521 total), all passing

### 2026-02-26 — Contract semantic annotations
- `SemanticColumnSpec` in contracts.py
- `load_contract()` parses concept_uris, resolve, semantic fields
- `has_semantic_annotations` flag on ContractContext

## Phase 0: Document Extraction MVP (delivered)

### 2026-02-26 — P5 "The Frankenstein" messy real-world Excel fixtures
- 4 new heuristics: XH5 (header block detection), XH6 (trailing column trimming via blank-column fence), XH7 (trailing footnote detection), XH8 (aggregation row detection with dual-channel keyword + bold validation)
- P5 persona: 4 synthetic fixtures (analyst workspace, lookup paradise, dashboard hybrid, living document)
- Blank data row cleanup for formatting breathing room
- 34 new tests (607 total), all passing
- Updated `XLSX_EXTRACTOR.md` with XH5-XH8 documentation

### 2026-02-26 — XLSX extractor validation and hardening
- 4 new heuristics: XH1 (multi-table boundary detection), XH2 (title row), XH3 (hidden content filtering), XH4 (number format hints)
- Merge-based + type-pattern + span-count layered header detection
- Forward-fill compound headers for merged header spans
- Persona-driven synthetic test corpus (ADR-003): 16 fixtures, 4 personas
- 52 new tests (573 total), all passing
- `XLSX_EXTRACTOR.md` per-module documentation

### 2026-02-23 — Package rename
- Renamed package from pdf_ocr to docpact

### Earlier
- ~13,800 LOC extraction pipeline
- 40+ deterministic heuristics
- 3 production contracts (au_shipping_stem, ru_ag_ministry, acea_car_registrations)
- Async pipeline orchestration with multi-document merge
- BAML-based LLM fallback with deterministic validation
