# Pipe-Table Text Normalization

> **Module**: `normalize.py`
> **Public API**: `normalize_pipe_table()`

Centralized, lossless normalization applied to pipe-table markdown before downstream consumption (classification, interpretation). Ensures consistent text regardless of which extractor produced it.

## Why

Different document formats and PDF renderers produce subtly different Unicode for the same visual character — NBSP vs regular space, smart quotes vs ASCII, en-dash vs hyphen. Without normalization, alias matching and keyword scoring silently fail on otherwise-correct text.

## Transformations

All transformations are **lossless** and **idempotent** (applying twice produces the same result as applying once).

| Step | Input | Output | Rationale |
|------|-------|--------|-----------|
| 1 | NBSP (`\u00a0`) | Regular space | PDF extractors often emit NBSP in table cells |
| 2 | Smart quotes (`\u201c` `\u201d` `\u2018` `\u2019`) | ASCII `"` `'` | Word/DOCX formatting artifacts |
| 3 | En-dash (`\u2013`), em-dash (`\u2014`) | Hyphen-minus (`-`) | Consistent range/separator character |
| 4 | Zero-width chars (ZWSP, ZWNJ, ZWJ, BOM, Word Joiner) | Removed | Invisible characters that break string matching |
| 5 | Runs of 2+ spaces/tabs | Single space (per line) | Whitespace collapse for consistent column alignment |
| 6 | Trailing whitespace | Stripped per line | Clean line endings |

## Integration Points

`normalize_pipe_table()` is called automatically at entry points of:
- `classify_tables()` — before keyword scoring
- `interpret_table()` — before alias matching
- `interpret_table_single_shot()` — before LLM prompt construction
- `interpret_table_async()` / `interpret_tables_async()` — before batched interpretation

Callers do not need to normalize explicitly.

## API

### `normalize_pipe_table(text: str) -> str`

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Pipe-table markdown string |

Returns the normalized string.
