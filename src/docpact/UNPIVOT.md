# Pivot Detection and Unpivoting

> **Module**: `unpivot.py`
> **Public API**: `unpivot_pipe_table()`, `detect_pivot()`
> **Public Types**: `PivotDetection`, `PivotGroup`, `UnpivotResult`

Schema-agnostic deterministic pivot detection and unpivoting for pipe-tables. Detects repeating compound header groups and transforms wide pivoted tables into long format.

## Why

Many real-world tables are *pivoted* — entity attributes are spread across column groups rather than rows. For example, a table might have columns `"wheat / 2025"`, `"wheat / 2024"`, `"barley / 2025"`, `"barley / 2024"` instead of a single `"crop"` column. This makes interpretation harder because the LLM (or deterministic mapper) must understand the group structure.

Unpivoting runs as a **pre-processing step** before the LLM path (after the deterministic mapper attempt fails). The deterministic mapper handles pivoted tables natively via alias-based group detection, so unpivoting is only applied when falling back to LLM interpretation.

## Algorithm

### Detection (`detect_pivot`)

1. **Parse headers**: Split compound headers on ` / ` into `(prefix, suffix)` pairs. Headers without ` / ` are "shared" columns.
2. **Group by prefix**: Build groups of columns sharing the same prefix (e.g., all columns starting with `"wheat"`).
3. **Match suffixes**: For each candidate reference group, compare its suffix list against every other group using `rapidfuzz.fuzz.ratio()`. Two groups match if they share ≥2 suffixes above the similarity threshold (default 85).
4. **Select best reference**: Try each group as the reference and pick the one yielding the most matching groups (≥`min_groups`, default 2).
5. **Handle non-matching groups**: Groups that don't match the reference (e.g., `"Th.ha. / Region"`) become shared columns.
6. **Intersect matched indices**: Only suffixes matched across *all* matching groups become measure labels.

### Unpivoting (`unpivot_pipe_table`)

When pivot structure is detected:

1. Build new header: `[pivot_column_name] + shared columns + measure labels`
2. For each data row, emit one output row per group: `[group prefix] + shared values + measure values`
3. Pass through section labels (`**LABEL**`), headings (`## ...`), and empty lines unchanged

## Example

**Input**:
```
| Region | spring crops / Target | spring crops / 2025 | spring grain / Target | spring grain / 2025 |
|---|---|---|---|---|
| Belgorod | 100 | 90 | 50 | 45 |
```

**Output**:
```
| _pivot | Region | Target | 2025 |
|---|---|---|---|
| spring crops | Belgorod | 100 | 90 |
| spring grain | Belgorod | 50 | 45 |
```

## API

### `detect_pivot()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compressed_text` | `str` | required | Pipe-table markdown (single page) |
| `similarity_threshold` | `float` | `85.0` | Min rapidfuzz.fuzz.ratio score (0-100) for suffix matching |
| `min_groups` | `int` | `2` | Min groups with matching suffixes |

Returns `PivotDetection` with `is_pivoted`, `shared_col_indices`, `groups`, `measure_labels`.

### `unpivot_pipe_table()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compressed_text` | `str` | required | Pipe-table markdown (single page) |
| `similarity_threshold` | `float` | `85.0` | Min score for suffix matching |
| `min_groups` | `int` | `2` | Min matching groups |
| `pivot_column_name` | `str` | `"_pivot"` | Name for the new pivot column |

Returns `UnpivotResult` with `text`, `was_transformed`, `detection`.

## Dependencies

Requires `rapidfuzz` for fuzzy suffix matching.
