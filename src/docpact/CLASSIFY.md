# Table Classification

> **Module**: `classify.py`
> **Public API**: `classify_tables()`, `_keyword_matches()`, `_compute_similarity()`, `_parse_pipe_header()`, `_tokenize_header_text()`

Format-agnostic table classification on compressed pipe-table markdown. Classifies tables by matching header text against user-defined keyword categories.

## Why

A single document often contains multiple table types (harvest data, export data, pricing). Before interpretation, we need to route each table to the correct output schema. Classification uses keyword scoring on header text — fast, deterministic, and independent of the document format.

## Three-Layer Approach

### Layer 1: Word-Boundary Keyword Scoring

For each table, the header text is extracted from the pipe-table markdown (via `_parse_pipe_header()`), joined into a single string, and matched against each category's keywords using `_keyword_matches()`.

Key features:
- **Word-boundary matching**: Uses `(?<![a-zA-Z])` / `(?![a-zA-Z])` instead of `\b` to avoid misfires on digits/underscores. Prevents `"rice"` from matching `"Price"` and `"port"` from matching `"transport"`.
- **Suffix tolerance**: Each keyword token tolerates English morphological suffixes (`s`, `es`, `ed`, `ing`), so `"export"` matches `"exports"` and `"harvest"` matches `"harvested"`.
- **Compound header collapse**: The ` / ` separator from compound headers is replaced with ` ` before matching, so multi-word keywords like `"area harvested"` work across compound header boundaries.
- **Title prepend**: If the table has a title (from metadata), it's prepended to the header text for matching.

The category with the highest keyword hit count wins. Ties are broken by dict insertion order.

### Layer 2: `min_data_rows` Filtering

Tables with fewer data rows than the threshold are forced to `"other"` immediately. Prevents noise tables (empty, tiny, or metadata-only) from affecting classification.

### Layer 3: Similarity Propagation

When `propagate=True`, structural profiles are built from keyword-matched tables:
- **Column count**: Average column count per category.
- **Header tokens**: Union of tokenized header words (years and pure numbers stripped).

Unmatched `"other"` tables are compared against these profiles using a weighted similarity metric: **50% column-count ratio + 50% header-token Jaccard**. Tables exceeding `propagate_threshold` are re-classified.

Profiles are built only from keyword-matched tables (not propagated ones) to prevent cascading.

## API

### `classify_tables()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tables` | `list[tuple[str, dict]]` | required | `(markdown, meta)` tuples. Each meta must have `table_index`, `title`, `row_count`, `col_count` |
| `categories` | `dict[str, list[str]]` | required | Category name → keyword list |
| `min_data_rows` | `int` | `0` | Tables below this threshold → `"other"` |
| `propagate` | `bool` | `False` | Enable similarity propagation |
| `propagate_threshold` | `float` | `0.3` | Minimum similarity for propagation |

Returns `list[dict]` with fields: `index`, `category`, `title`, `rows`, `cols`.

### Helpers

| Function | Purpose |
|----------|---------|
| `_keyword_matches(keyword, text)` | Word-boundary keyword match with suffix tolerance |
| `_parse_pipe_header(markdown)` | Extract column names from pipe-table header line |
| `_tokenize_header_text(header_text)` | Tokenize header text for similarity, stripping years/numbers/punctuation |
| `_compute_similarity(...)` | Weighted column-count + Jaccard similarity |
| `_YEAR_RE` | Compiled regex for year detection (re-exported, used by `docx_extractor.py`) |

## Integration

Normalization (`normalize_pipe_table()`) is applied automatically at the start of `classify_tables()`.

Input format is shared between DOCX (`compress_docx_tables()`) and PDF (`StructuredTable.to_compressed()`). The DOCX-specific `classify_docx_tables()` in `docx_extractor.py` is a thin wrapper that compresses then delegates here.
