# Australian Shipping Stem: Graincorp & CBH Fixes

## Status

All Russian DOCX and other Australian providers (Bunge, Newcastle, Brisbane) are working correctly after recent commits. Two Australian providers remain with issues.

## Issue 1: Graincorp — Logo Table Hijacks Parser

**PDF**: `inputs/shipping-stem-2025-11-13.pdf` (3 pages)
**Contract**: `contracts/au_shipping_stem.json`

### Problem

Page 0 of the PDF has a small 3-column logo table (`GRAINCORP | SHIPPING | STEM`) above the real 18-column data table. `_parse_pipe_table_deterministic()` in `interpret.py` locks onto the logo table's header (the first `|` row + `|---|` separator it encounters), treating the 3-column logo as the header. Consequently:

- The actual 18-column data rows have empty first cells (`|||Gladstone|...`), so the `||` prefix causes them to be misclassified as **aggregation rows** (75 out of 77 pipe rows)
- Only 2 real data rows survive; the deterministic mapper correctly rejects ("no measures found") and falls to LLM
- Pages 1-2 have garbled headers (logo text merged into data table super-header by the compressor) and the same `||` problem → also fall to LLM

Additionally, the single-category shortcut in `pipeline.py` (lines 127-129) skips `classify_tables()` entirely — so even though classification would correctly filter out the logo table, it never runs.

### Recommended Fix: Multi-table-aware parser restart

Modify `_parse_pipe_table_deterministic()` to detect when a **second separator** (`|---|...`) appears in the data stream with a dramatically different column count from the current header. When `new_separator_cols >> current_header_cols` (e.g., 18 vs 3), **restart the parser** with the new, larger table. This is structurally principled: a 3-column decorative table preceding an 18-column data table is a common PDF pattern.

**Heuristic**: When a re-separator is found and the data row immediately before it has significantly more pipe cells than the current header count, discard the previous parse and restart from the new header.

**Secondary fix**: The `||` aggregation heuristic is overly aggressive. Rows like `|||Gladstone|56307|GCOP|DELTA|...` are real data where the first 2 columns (GC Fin Year, Month) are blank carry-forwards. Consider requiring that aggregation rows also have a majority of empty cells or contain "Total" keywords, not just a leading `||`.

---

## Issue 2: CBH — "Stock at Port" Absorbed into Esperance Section

**PDF**: `inputs/CBH Shipping Stem 26092025.pdf` (1 page)
**Contract**: `contracts/au_shipping_stem.json`

### Problem

After the ESPERANCE shipping data (2 vessel rows + total row "90,000"), the PDF continues with "Stock at Port" inventory data and "PORT MAINTENANCE SHUTDOWN DATES" — these are structurally different tables that happen to share some x-positions with the shipping table.

**Compression layer** (`_detect_table_runs` in `compress.py`): Row 55 (spatial) is a single-span numeric "90,000" → `_is_table_continuation` = True (correct). Row 56 ("Stock at Port" with 8 spans) shares 2 column positions with the accumulated pool (col 206 ≈ 205, col 264 ≈ 266) → meets `overlap_count >= 2` → joins the table run. This cascades: rows 57-61 continue matching against the enlarged pool.

**Interpretation layer**: "Stock at Port" appears as a pipe row (`|Stock at Port|(Main|...`), not as plain text, so it's never detected as a section boundary. All 6 trailing rows get `load_port = "ESPERANCE"`.

### Recommended Fix: Post-aggregation overlap tightening

In `_detect_table_runs()` (`compress.py`), after a single-span numeric row (aggregation total), **require a higher overlap threshold** for the next multi-span row to continue the run. Currently: `overlap_count >= 2` or `overlap_ratio >= 0.6`. After an aggregation row, require `overlap_ratio >= 0.5` (at least 50% of columns match the pool).

For the CBH case: "Stock at Port" has 8 columns, only 2 match → ratio 0.25 < 0.5 → run breaks. The shipping data rows typically have 15+ matching columns → ratio > 0.8 → continues normally.

**Alternative**: After a single-span numeric row (total), reset the pool entirely and require the next multi-span rows to independently meet `min_table_rows` to start a new run. This prevents stale pool positions from creating false continuations.

Both approaches are structurally principled: an aggregation total signals a section boundary, and what follows must demonstrate strong structural alignment to be considered part of the same table.

---

## Implementation Order

1. **CBH fix first** — simpler, isolated to `_detect_table_runs` in `compress.py`, lower risk
2. **Graincorp fix second** — more complex, touches `_parse_pipe_table_deterministic` in `interpret.py`, needs careful testing against all providers

## Testing

- Run `uv run pytest tests/ -v` after each change
- Pipeline validation on all 6 Australian providers: Newcastle, Bunge, Graincorp, CBH, Brisbane, and the 2857439 provider
- Ensure Russian DOCX pipeline still passes
