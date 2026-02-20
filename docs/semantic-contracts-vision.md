# Semantic-Driven Contracts — Design Vision

## Context

The current contract JSON already separates **what you want** (schema + aliases) from **what you're given** (document formats, column names, layouts). The pipeline is 100% domain-agnostic. But aliases are hand-curated flat strings with no formal relationship to external knowledge — if two contracts both mention "wheat", there's no way to know they mean the same thing.

## The Idea: Ontology-Grounded Contracts

Instead of manually maintaining alias lists, **derive them from linked-data ontologies** like AGROVOC (FAO's multilingual agricultural thesaurus). A contract column would reference a concept URI rather than flat strings:

- Aliases are **generated at contract resolution time** from the ontology — new synonyms, Russian labels, narrower terms (e.g., "durum wheat" matching "wheat") come for free
- The materialized contract JSON stays backward-compatible — `docpact` consumes it unchanged
- Each alias carries provenance: which ontology version, which concept URI, which resolution rules produced it

## Dependent Repo: `contract-semantics`

This would sit **upstream** of `docpact` as a contract authoring/validation toolkit. Three responsibilities:

1. **Contract grounding (authoring time)** — resolve concept URIs to materialized contract JSON with auto-generated aliases
2. **Pre-extraction constraints** — inject domain intelligence at boundaries (expected crops, regions, plausible ranges) without changing extraction logic
3. **Post-extraction validation** — SHACL shapes validate output records against ontology constraints; validation reports ship with every delivery

## Why It Matters

- **Alias maintenance disappears** — the biggest ongoing cost in current contracts. Ground in AGROVOC and they maintain themselves
- **FAIR-compliant outputs** — ontology-grounded, SHACL-validated datasets are composable with FAO, Eurostat, satellite imagery via shared concept URIs
- **Provider onboarding scales** — new provider = half-day pointing at concept subtrees, not days of manual alias curation
- **Architectural moat** — flat alias lists are trivially replicable; ontology-grounded contracts with SHACL validation require real investment

## Incremental Path

1. **Annotate** — add `concept_uri` fields to existing contract columns (zero risk, pipeline ignores them)
2. **Alias generation pilot** — for Russian agriculture contract, generate aliases from AGROVOC and compare against manual list
3. **SHACL validation prototype** — write shapes for one output, validate existing pipeline results
4. **Contract authoring tool** — if steps 2-3 show value, build the full resolution pipeline

## Architecture

```
Semantic:   Resolve concepts → AGROVOC cereals, GeoNames Russia admin-1
Contract:   Materialized JSON (unchanged format, consumed by docpact)
Extract:    docpact (deterministic + LLM pipeline, unchanged)
Validate:   SHACL shapes → validation report per delivery
```

The pipeline code never changes. Semantic intelligence lives entirely in the contract authoring and validation layers.
