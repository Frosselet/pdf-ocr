# Report Date Resolution

> **Module**: `report_date.py`
> **Public API**: `ReportDateConfig`, `resolve_report_date()`

Resolves the report date from declarative contract configuration. Supports both deterministic sources (filename, timestamp, constant) and LLM-based extraction (filename patterns, content parsing).

## Why

Documents rarely contain a clean "report date" field. The date might be embedded in the filename (`harvest_2025-06-20.docx`), inside the document body, or known a priori as a constant. The contract declares *where* to find it; this module does the extraction.

## Sources

| Source | Deterministic? | Description |
|--------|---------------|-------------|
| `"filename"` | LLM | Extract date from the document filename using a hint pattern |
| `"timestamp"` | Yes | Current datetime (for time-of-extraction stamping) |
| `"constant"` | Yes | Fixed value from the contract's `value` field |
| `"metadata"` | — | Not yet implemented |
| `"content"` | LLM | Not yet implemented |
| `"url"` | — | Not yet implemented |
| `"api_response"` | — | Not yet implemented |

LLM-based sources (`filename`, `content`) require a `hint` field describing the expected date format. They call `ExtractReportDate` via the BAML async client.

## API

### `ReportDateConfig`

Dataclass with fields:

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | One of the source types above |
| `hint` | `str \| None` | Pattern hint for LLM extraction |
| `field` | `str \| None` | Field name for metadata/API sources |
| `value` | `str \| None` | Fixed value for constant source |

Constructed from contract JSON via `ReportDateConfig.from_dict(data)`.

### `resolve_report_date(config, *, doc_path=None) -> str`

Async function. Resolves the report date using the configured source. Returns the date as a string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ReportDateConfig` | Declarative configuration |
| `doc_path` | `str \| None` | Path to the document (required for `filename` source) |
