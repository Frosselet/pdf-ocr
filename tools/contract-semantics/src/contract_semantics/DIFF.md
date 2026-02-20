# Diff

> **Module**: `diff.py`
> **Public API**: `diff_aliases()`, `diff_all()`

Compare auto-generated (ontology-resolved) aliases against manual aliases. Produces human-readable reports showing coverage, matches, gaps, and new discoveries.

---

## The Problem

After resolving concepts against an ontology, you need to understand: how many of my manual aliases are covered? What did the ontology find that I missed? What composite terms remain manual-only? This is the feedback loop that validates whether ontology grounding actually works for a given contract.

## The Solution

Two functions that format `ResolutionResult` objects into readable reports.

---

## API

### diff_aliases()

Report for a single column.

```python
from contract_semantics.diff import diff_aliases

report = diff_aliases(result)
print(report)
```

Output:

```
Column: Crop
Coverage: 55% (6/11)

Matched (manual alias also found via ontology):
  + barley
  + buckwheat
  + corn
  + rice
  + sunflower
  + wheat

Manual-only (not found in ontology — composite terms, domain-specific):
  - grains and grasses
  - spring barley
  - spring crops
  - spring grain
  - spring wheat

Resolved-only (new aliases from ontology — potential additions):
  * ячмень [ru, prefLabel]
  * гречиха [ru, prefLabel]
  * кукуруза [ru, prefLabel]
  * пшеница [ru, prefLabel]
  * spring wheat [en, prefLabel, pattern="spring {label}"]
```

### diff_all()

Combined report for multiple columns with a summary section.

```python
from contract_semantics.diff import diff_all

report = diff_all([crop_result, region_result, metric_result])
print(report)
```

Appends a summary:

```
============================================================
Summary
============================================================
Columns resolved: 3
Overall coverage: 62% (18/29)
New aliases discovered: 47
```

---

## Report Sections

| Section | Symbol | Meaning |
|---|---|---|
| **Matched** | `+` | Manual alias confirmed by ontology |
| **Manual-only** | `-` | Not in ontology (composite terms, domain jargon) |
| **Resolved-only** | `*` | New from ontology, potential addition to contract |

Resolved-only entries include provenance annotations: `[language, label_type]` and `pattern="..."` if generated via prefix pattern expansion.

---

## Test Coverage

5 tests in `tests/test_diff.py` covering full coverage, partial coverage, zero coverage, provenance display, and combined multi-column reports.
