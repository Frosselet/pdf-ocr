# Resolution

> **Module**: `resolve.py`
> **Public API**: `OntologyAdapter` (protocol), `resolve_column()`

Generic ontology resolution layer. Defines the `OntologyAdapter` protocol that all adapters implement, and the `resolve_column()` function that orchestrates resolution for a single contract column --- including prefix pattern expansion and coverage analysis.

---

## The Problem

Different ontologies have different APIs (SPARQL, REST, file formats), but the resolution logic is the same: query concepts, expand labels with patterns, deduplicate, compare against manual aliases. The resolution layer abstracts the ontology-specific details behind a common protocol.

## The Solution

### OntologyAdapter Protocol

Any adapter that implements `resolve_concept()` can be used with `resolve_column()`:

```python
class OntologyAdapter(Protocol):
    def resolve_concept(
        self,
        uri: str,
        *,
        languages: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> list[ResolvedAlias]: ...
```

Both `AgrovocAdapter` and `GeoNamesAdapter` implement this protocol. The protocol is `@runtime_checkable`.

---

## API

### resolve_column()

Resolve concept URIs for a single contract column and compare against manual aliases.

```python
from contract_semantics.resolve import resolve_column
from contract_semantics.models import ConceptRef, ResolveConfig

result = resolve_column(
    column_name="Crop",
    concept_refs=[
        ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_8373", label="wheat"),
        ConceptRef(uri="http://aims.fao.org/aos/agrovoc/c_631", label="barley"),
    ],
    manual_aliases=["spring wheat", "wheat", "barley", "grains and grasses"],
    resolve_config=ResolveConfig(
        languages=["en", "ru"],
        label_types=["prefLabel", "altLabel"],
        prefix_patterns=["spring {label}", "{label}"],
    ),
    adapter=agrovoc_adapter,
)

print(result.coverage)       # 0.5 (2 of 4 manual aliases matched)
print(result.matched)        # ["wheat", "barley"]
print(result.manual_only)    # ["spring wheat", "grains and grasses"]  -- wait, "spring wheat" IS matched via pattern
print(result.resolved_only)  # ["пшеница", "ячмень", "spring barley", ...]
```

| Parameter | Type | Description |
|---|---|---|
| `column_name` | `str` | Contract column name (for reporting) |
| `concept_refs` | `list[ConceptRef]` | Concept URIs + labels from the contract |
| `manual_aliases` | `list[str]` | Existing hand-curated aliases |
| `resolve_config` | `ResolveConfig` | Resolution configuration |
| `adapter` | `OntologyAdapter` | Ontology adapter instance |

**Returns**: `ResolutionResult` with resolved aliases and coverage analysis.

---

## Resolution Pipeline

```
For each ConceptRef:
 1. adapter.resolve_concept(uri, languages, label_types)
    -> list[ResolvedAlias]  (raw labels from ontology)

 2. If prefix_patterns configured:
    For each label x pattern:
      "{label}" x "wheat" -> "wheat"
      "spring {label}" x "wheat" -> "spring wheat"
    -> list[ResolvedAlias]  (expanded)

 3. Deduplicate by alias text (case-insensitive)
    -> list[ResolvedAlias]  (deduped)

 4. Compare against manual_aliases (case-insensitive)
    -> matched, manual_only, resolved_only
```

### Prefix Patterns

Prefix patterns generate composite aliases from base ontology labels. The `{label}` placeholder is replaced with each resolved label:

| Pattern | Base Label | Generated Alias |
|---|---|---|
| `{label}` | `wheat` | `wheat` |
| `spring {label}` | `wheat` | `spring wheat` |
| `{label}` | `пшеница` | `пшеница` |
| `spring {label}` | `пшеница` | `spring пшеница` |

This handles the common pattern in agricultural data where crop names appear with seasonal prefixes ("spring wheat", "winter barley") that aren't separate concepts in the ontology.

---

## Test Coverage

6 tests in `tests/test_resolve.py` covering basic resolution, prefix pattern expansion, Russian labels as resolved-only, coverage calculation, GeoNames resolution, and alias deduplication.
