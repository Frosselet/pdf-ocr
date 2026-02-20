# recommend.py — Contract Recommendation Engine

Generates draft contract skeletons from document profiles. Takes a `DocumentProfile` or `MultiDocumentProfile` from `analyze.py` and produces a draft contract JSON with inline recommendations.

## Public Functions

### `recommend_contract(profile, *, provider_name, model) → dict`

Generate a draft contract JSON from a document profile.

**Args:**
- `profile: DocumentProfile | MultiDocumentProfile` — From `analyze.py`.
- `provider_name: str | None` — Optional human-readable provider name. Auto-guessed from filenames if not provided.
- `model: str` — LLM model identifier (default: `"openai/gpt-4o"`).

**Returns:** Dict representing the draft contract JSON with inline `_recommendation` and `_detected_*` guidance fields.

### `compare_contract(profile, existing_path) → str`

Compare a document profile against an existing contract. Flags uncovered headers, unused aliases, type mismatches, and uncovered section labels.

**Args:**
- `profile: DocumentProfile | MultiDocumentProfile`
- `existing_path: str | Path` — Path to existing contract JSON.

**Returns:** Multi-line human-readable gap report.

### `strip_recommendations(contract) → dict`

Strip all `_recommendation`, `_detected_*`, `_analyzer_*` fields from a draft contract. Returns a clean contract JSON suitable for use with docpact's `load_contract()`.

## Draft Contract Structure

The recommended contract follows the standard docpact contract format with additional annotation fields:

```json
{
  "_analyzer_version": "0.1.0",
  "_source_documents": ["report.pdf"],
  "provider": "report",
  "model": "openai/gpt-4o",
  "categories": {
    "category_name": {
      "keywords": ["keyword1", "keyword2"],
      "_recommendation": "3 tables share these header tokens"
    }
  },
  "outputs": {
    "category_name": {
      "category": "category_name",
      "schema": {
        "columns": [
          {
            "name": "column_name",
            "type": "string",
            "aliases": ["Header Variant A", "Header Variant B"],
            "_detected_type": "string",
            "_recommendation": "2 header variants found across documents.",
            "_header_text": "Header Variant A"
          }
        ]
      }
    }
  },
  "report_date": {
    "source": "content",
    "hint": "Found: 'As of January 15, 2025'",
    "_recommendation": "Temporal pattern detected."
  }
}
```

All `_` prefixed fields are stripped by `strip_recommendations()` and are ignored by docpact's `load_contract()`.

## What Gets Detected

| Signal | Contract field | Detection method |
|---|---|---|
| Column names (header text) | `aliases` | Pipe-table header parsing |
| Column data types | `type` | `detect_column_types()` from heuristics |
| Year columns in headers | `aliases: ["{YYYY}"]` | `_YEAR_RE` pattern match |
| Section labels (bold/text) | Dimension column `aliases` | Bold markers, plain text between pipe rows |
| Table titles | `source: "title"` column | `## heading` above table |
| Temporal metadata | `report_date` config | `detect_temporal_patterns()` |
| Unit annotations | `_unit_annotations` | `detect_unit_patterns()` + header parsing |
| Pivoted tables | `unpivot: "deterministic"` | `detect_pivot()` |
| Table similarity groups | `categories` | Token Jaccard + column count ratio |

## What Requires Human Judgment

1. **Canonical column names** — The analyzer proposes snake_case names from headers; the human chooses final names.
2. **Multi-provider alias consolidation** — Recognizing that header variants across providers refer to the same concept.
3. **Ontology grounding** — Whether to add `concept_uris` and `resolve` blocks (AGROVOC, GeoNames).
4. **Output semantics** — Date formats, number formats, which columns matter downstream.
5. **Report date extraction** — Whether the date comes from filename, content, or elsewhere.

## Workflow

```
profile_document("report.pdf")    # or multiple documents
  → merge_profiles([...])
  → recommend_contract(profile)   # draft with recommendations
  → strip_recommendations(draft)  # clean for production use
```

Or with comparison:
```
profile = profile_document("report.pdf")
report = compare_contract(profile, "contracts/existing.json")
print(report)  # gaps between document and contract
```
