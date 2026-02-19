# Russian DOCX Table Systems — Design Note

## Context

Running all 7 Russian Ministry of Agriculture DOCX reports through the pipeline revealed
systematic extraction failures. The root cause: DOCX headers are **structurally unstable** —
column names, units, separators, and even the presence of columns change between reports.
The current alias-matching approach breaks on every variation.

This note proposes a **structure-first** approach: understand the table's logical system
(pivot axes, shared columns, data layout) before attempting header resolution.

---

## Observed Issues (from 7-document test run)

### Planting Tables (April, June)

| Crop group | Result | Root cause |
|-----------|--------|------------|
| spring crops, spring grain | OK | Headers: `Th.ha. / Region`, `spring crops / MOA Target 2025`, `spring crops / 2025` — Region is suffix in col 0 (1 occurrence → shared column) |
| spring wheat, spring barley | Region = literal `"region"` | Headers: `Region / Th.ha.`, `Region / Final 2024`, `Region / MOA 2024`, `spring wheat / ...` — Region is PREFIX in cols 0-2 (3 occurrences with same value → misclassified as constant dimension) |
| corn | Same Region bug | Same header structure as wheat/barley |
| rice | Region NaN, Crop NaN | Header: `\|  \| MOA Target 2025 \| 2025 \| % \| 2024 \|` — first col is EMPTY, no crop group prefix (crop only in title "RICE") |
| sunflower, soya, rape | OK | Headers: `Region`, `Th.ha. / Final 2024`, `SUNFLOWER / MOA Target 2025`, etc. — Region is a simple single-part column |

**Key insight:** The same logical table (Region × Crop × {Area, Value}) has **4 different
header layouts** across crop groups within the same document. The mapper handles 2 of 4.

### Harvest Tables (July, September, November)

| Issue | Root cause |
|-------|-----------|
| Metric column = NaN everywhere | Header parts like `Area harvested,Th.ha.` don't match alias `Area harvested` (comma+unit suffix). Same for `collected, TMT bunker weight` vs `collected`, `Yield, metric centner/ha` vs `Yield` |
| Works otherwise | Region, Crop (from title), Year, Value all resolve correctly |

### November Planting (different structure entirely)

| Issue | Root cause |
|-------|-----------|
| Area unmatched | Header says `Target area`, alias expects `MOA Target {YYYY}` |
| Crop unmatched | Table title is `None` — no title extracted from DOCX |

---

## Structural Analysis: Two Table Systems

### System 1: Planting Tables

**Logical structure:** Region × Crop × {MOA Target, Actual Sown}

The table is **pivoted by crop group**. Each crop group contributes 2+ data columns
(target area, actual area, %, prior year). The Region column is shared across all groups.

Canonical form (after unpivoting crop groups):
```
| Region | Crop | MOA_Target | Actual | pct | prior_year |
```

Header variations observed across 7 documents:
- Col 0: `Th.ha. / Region` or `Region / Th.ha.` or `Region` or `` (empty!)
- Reference cols: `Th.ha. / Final 2024` or `Region / Final 2024` or `Th.ha. / MOA 2024` (0-3 reference columns, inconsistent)
- Data cols: `{crop} / MOA Target {YYYY}`, `{crop} / {YYYY}`, `{crop} / %`, `{crop} / {YYYY-1}`
- November variant: completely different — `Area planted,Th.ha. / 2025`, `Target area` (pivoted by year, not by crop)

**The stable pattern:** Within each document, the data columns follow a repeating group
structure `{crop} / {measure}` where `{crop}` varies and `{measure}` repeats. The number
of measures per group (2-4) and the number of reference columns (0-3) changes, but the
repeating group pattern is constant.

### System 2: Harvest Tables

**Logical structure:** Region × Metric × Year (pivoted by year)

Each metric (Area harvested, Collected, Yield) has one column per year. The more years
compared, the more columns.

Canonical form (after unpivoting year groups):
```
| Region | Metric | Year | Value |
```

Header pattern: `{metric},{unit} / {year}` — e.g., `Area harvested,Th.ha. / 2025`.
The metric part includes a comma-separated unit suffix that varies (`Th.ha.`, `TMT bunker weight`, `metric centner/ha`).

**The stable pattern:** Columns pair as `{metric} / {year_a}`, `{metric} / {year_b}`.
Each metric name (before the ` / `) repeats once per year. The number of year columns
varies (2025 vs 2024, sometimes 2023).

---

## Proposed Approach: Structure-First Table Understanding

### Core Principle

> **Understand the table's logical system before resolving headers.**
>
> Don't try to match headers to aliases first. Instead, detect the table's structural
> pattern (what's pivoted, what's shared, how many groups, how many measures per group),
> then use that understanding to interpret — and if necessary correct — the headers.

### Step 1: Detect table system from data shape

Before touching headers, analyze the **data section**:

1. **Column count** in data rows (pipe-cell count) — this is ground truth
2. **Header column count** — compare with data. If mismatch → header is malformed,
   needs correction before any alias matching
3. **Repeating column groups** — look for patterns in the data values:
   - Numeric columns that repeat in groups (e.g., 3 numeric cols repeated N times)
   - A leading text column (Region names) that's clearly a shared dimension
4. **Title/section labels** — these often carry the crop name or category

### Step 2: Reconcile headers with data structure

If header column count ≠ data column count:
- The header flattening (`compress.py`) produced an inconsistency
- **Before alias matching, correct the header** using the understood structure:
  - If we detected N repeating groups of M measures → total data cols = 1 (Region) + N*M + reference_cols
  - If the header has more/fewer pipes, re-split or merge header cells

If header column count = data column count:
- Proceed to alias matching, but **informed by the detected structure**:
  - We know which columns are the pivot axis (crop groups or year groups)
  - We know which column is the shared index (Region)
  - We know the group boundaries

### Step 3: Structure-informed alias matching

Instead of matching each header part independently:

1. **Identify the shared/index column** (col 0 or wherever the text column is) → map to Region
2. **Identify reference columns** (between index and first data group) → drop
3. **Identify repeating groups** → extract group prefix (crop name) and measure suffix
4. **Match group prefixes** to Crop aliases
5. **Match measure suffixes** to schema measure aliases (Area, Value, etc.)

This is more robust than part-by-part alias matching because it uses structural
understanding to disambiguate — e.g., "Region" in col 0 is always the shared index,
never a crop group prefix, regardless of how it appears in the compound header.

---

## Open Questions

1. **Where does this logic live?** Is it a pre-processing step before `_try_deterministic()`,
   or a replacement for it? The current `_try_deterministic` is generic (works for shipping,
   ACEA, etc.). The "table system" approach is specific to Russian DOCX structure. Should it
   be a contract-level configuration (`"table_system": "crop_pivot"`) or auto-detected?

2. **Header correction vs data-only parsing?** When headers are unreliable, should we:
   (a) Fix the headers and then run normal alias matching, or
   (b) Skip headers entirely and infer column roles from data patterns + title?

3. **November's year-pivot structure** is fundamentally different from the crop-pivot
   structure used April–September. Same contract output but different table system.
   Handle as two sub-systems, or generalize?

4. **Rice table** has no crop group prefix (only 1 crop → no pivot). This is a degenerate
   case of the crop-pivot system. Should be detectable: if no repeating groups and title
   is a known crop alias → single-crop table.

---

## Bug Fix Queue (independent of table-system redesign)

These are quick fixes that improve results now:

| # | Type | Fix | Impact |
|---|------|-----|--------|
| 1 | **Code** (`interpret.py` L1335-1342) | Constant-dim detection: only classify as constant if the dim appears in ≥1 data column (has measures/group-dims). Dims that ONLY appear in label columns should become shared columns. | Fixes wheat/barley/corn Region bug |
| 2 | **Contract** (`ru_ag_ministry.json`) | Add harvest Metric aliases: `"Area harvested,Th.ha."`, `"collected, TMT bunker weight"`, `"Yield, metric centner/ha"` | Fixes harvest Metric=NaN |
| 3 | **Contract** | Add `"Target area"`, `"Area planted"`, `"Area planted,Th.ha."` as Area aliases | Fixes November planting Area |

---

## Files Referenced

- `src/pdf_ocr/interpret.py` — `_map_to_schema_deterministic()` lines 1206-1470
- `contracts/ru_ag_ministry.json` — harvest/planting schemas
- `src/pdf_ocr/docx_extractor.py` — `compress_docx_tables()` produces the pipe-table headers
- `src/pdf_ocr/compress.py` — header flattening logic (stacked → ` / `-joined)
