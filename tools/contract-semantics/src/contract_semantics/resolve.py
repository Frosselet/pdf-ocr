"""Generic ontology resolution: OntologyAdapter protocol and resolve_column()."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contract_semantics.models import (
    ConceptRef,
    ResolveConfig,
    ResolvedAlias,
    ResolutionResult,
)


@runtime_checkable
class OntologyAdapter(Protocol):
    """Protocol for ontology adapters (AGROVOC, GeoNames, etc.)."""

    def resolve_concept(
        self,
        uri: str,
        *,
        languages: list[str] | None = None,
        label_types: list[str] | None = None,
    ) -> list[ResolvedAlias]: ...


def resolve_column(
    column_name: str,
    concept_refs: list[ConceptRef],
    manual_aliases: list[str],
    resolve_config: ResolveConfig,
    adapter: OntologyAdapter,
) -> ResolutionResult:
    """Resolve concept URIs for a single column and compare against manual aliases.

    Steps:
    1. For each concept URI, query the adapter for labels in requested languages.
    2. Apply prefix_patterns to expand labels (e.g. "spring {label}" → "spring wheat").
    3. Deduplicate resolved aliases.
    4. Compare against manual aliases (case-insensitive).

    Parameters:
        column_name: Name of the contract column being resolved.
        concept_refs: List of concept URI + label pairs from the contract.
        manual_aliases: Existing hand-curated aliases from the contract.
        resolve_config: Resolution configuration (languages, label_types, patterns).
        adapter: Ontology adapter to use for resolution.

    Returns:
        :class:`ResolutionResult` with resolved aliases and coverage analysis.
    """
    all_resolved: list[ResolvedAlias] = []

    for ref in concept_refs:
        # Resolve base labels from ontology
        labels = adapter.resolve_concept(
            ref.uri,
            languages=resolve_config.languages,
            label_types=resolve_config.label_types,
        )

        if resolve_config.prefix_patterns:
            # Expand each label with prefix patterns
            expanded: list[ResolvedAlias] = []
            for resolved in labels:
                for pattern in resolve_config.prefix_patterns:
                    alias_text = pattern.replace("{label}", resolved.alias)
                    expanded.append(
                        ResolvedAlias(
                            alias=alias_text,
                            concept_uri=resolved.concept_uri,
                            concept_label=resolved.concept_label,
                            language=resolved.language,
                            label_type=resolved.label_type,
                            pattern=pattern,
                        )
                    )
            all_resolved.extend(expanded)
        else:
            all_resolved.extend(labels)

    # Deduplicate by alias text (case-insensitive)
    seen: set[str] = set()
    deduped: list[ResolvedAlias] = []
    for ra in all_resolved:
        key = ra.alias.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(ra)

    # Compare against manual aliases
    resolved_lower = {ra.alias.lower() for ra in deduped}
    manual_lower = {a.lower(): a for a in manual_aliases}

    matched = [a for a in manual_aliases if a.lower() in resolved_lower]
    manual_only = [a for a in manual_aliases if a.lower() not in resolved_lower]
    resolved_only = [
        ra.alias for ra in deduped if ra.alias.lower() not in manual_lower
    ]

    return ResolutionResult(
        column_name=column_name,
        resolved_aliases=deduped,
        manual_aliases=manual_aliases,
        matched=matched,
        manual_only=manual_only,
        resolved_only=resolved_only,
    )
