# PDF Filtering

> **Module**: `filter.py`
> **Public API**: `filter_pdf_by_table_titles()`, `extract_table_titles()`

Filter PDF pages by table titles using fuzzy matching. Useful when PDFs contain multiple tables but you only need specific ones.

## Use Case

A monthly report PDF has 50 pages with 20 different tables. You need only the "New Car Registrations by Market" table. Instead of processing all pages, filter first:

```python
filtered_bytes, matches = filter_pdf_by_table_titles(
    "report.pdf",
    "new car registrations by market",
    threshold=85.0,
)
# Now process only the matching pages
```

## Pipeline

```
PDF pages
  │
  ▼
1. Extract HEADING regions from each page
  │
  ▼
2. Concatenate consecutive headings (multi-line titles)
  │
  ▼
3. Fuzzy match against search terms
  │
  ▼
4. Create new PDF with matching pages only
```

---

## Heuristics

### H1: Title Extraction (`_extract_titles_from_page`)

**Problem**: Which text on a page is a "table title"?

**Solution**: Reuse the compress.py region classifier — HEADING regions are titles:

```python
regions = _classify_regions(layout)
for region in regions:
    if region.type == RegionType.HEADING:
        extract text from region
```

**Why HEADING?** These are short, isolated single-span rows — exactly what table titles look like.

---

### H2: Footnote Filtering (`_is_footnote_text`)

**Problem**: Footnote markers and annotations look like headings but aren't titles.

**Solution**: Pattern-based rejection:

```python
_FOOTNOTE_PATTERNS = [
    r"^\d+$",           # "2", "3" (footnote numbers)
    r"^Includes\b",     # "Includes electric vehicles"
    r"^ACEA\b",         # Organization abbreviations
    r"^\*+$",           # Asterisks
    r"^Note:",          # Note markers
    r"^Source:",        # Source citations
]
```

**Effect**: Only real titles reach the matching stage.

---

### H3: Multi-Line Title Concatenation (`_concatenate_title_groups`)

**Problem**: Titles may span multiple lines:

```
NEW CAR REGISTRATIONS BY MARKET AND POWER SOURCE
MONTHLY
```

**Solution**: Group consecutive headings within `max_gap=3` rows:

```python
for row_idx, text in headings[1:]:
    if row_idx - last_row <= max_gap:
        current_group.append(text)
    else:
        groups.append(current_group)
        current_group = [text]
```

**Result**: `"NEW CAR REGISTRATIONS BY MARKET AND POWER SOURCE MONTHLY"`

---

### H4: Fuzzy Matching (`_fuzzy_match_titles`)

**Problem**: Exact matching fails when:
- User types "car registrations" but title says "CAR REGISTRATIONS"
- Word order differs
- Minor typos

**Solution**: RapidFuzz `WRatio` for robust matching:

```python
score = fuzz.WRatio(title.lower(), term.lower())
if score >= threshold:
    match!
```

**Why WRatio?**
- Handles partial matches
- Word reordering tolerance
- Case-insensitive
- Typo tolerance
- Score 0-100 (100 = perfect match)

**Default threshold**: 90.0 (high precision, some recall trade-off)

---

## API

### Filter by Title

```python
from docpact import filter_pdf_by_table_titles

filtered_bytes, matches = filter_pdf_by_table_titles(
    "report.pdf",
    "new car registrations by market and power source, monthly",
    threshold=90.0,
)
```

### Filter by Page Number

```python
filtered_bytes, matches = filter_pdf_by_table_titles(
    "report.pdf",
    pages=[2],  # 0-based, so page 3
)
```

### Combine Both (Union)

```python
filtered_bytes, matches = filter_pdf_by_table_titles(
    "report.pdf",
    "monthly registrations",
    pages=[0, 5],  # Also include first and last page
)
```

### Debug: Extract All Titles

```python
from docpact import extract_table_titles

titles = extract_table_titles("report.pdf")
for page_idx, page_titles in titles.items():
    print(f"Page {page_idx}: {page_titles}")
```

---

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pdf_input` | `str \| bytes \| Path` | required | Path to PDF or raw bytes |
| `search_terms` | `str \| list[str] \| None` | `None` | Terms to fuzzy match |
| `pages` | `list[int] \| None` | `None` | 0-based page indices to include |
| `threshold` | `float` | `90.0` | Minimum fuzzy score (0-100) |
| `output_path` | `str \| Path \| None` | `None` | Write filtered PDF to file |
| `cluster_threshold` | `float` | `2.0` | Y-distance for row clustering |
| `match_any` | `bool` | `True` | Include if ANY term matches |

### Return Value

```python
(filtered_pdf_bytes: bytes, matches: list[FilterMatch])
```

### FilterMatch

| Field | Type | Description |
|---|---|---|
| `page_index` | `int` | 0-based page number |
| `matched_title` | `str` | The title text that matched |
| `search_term` | `str` | The search term it matched against |
| `score` | `float` | Fuzzy match score (0-100) |

---

## Threshold Guidelines

| Threshold | Precision | Recall | Use Case |
|---|---|---|---|
| 95+ | Very high | Low | Exact title known |
| 90 | High | Medium | General use (default) |
| 80 | Medium | High | Exploratory search |
| 70 | Low | Very high | Catch-all |
