"""Ontology-grounded contract authoring and validation toolkit for docpact."""

from contract_semantics.models import (
    ConceptRef,
    GeoEnrichment,
    GeoSearchResult,
    ResolveConfig,
    ResolvedAlias,
    ResolutionResult,
)

__all__ = [
    "ConceptRef",
    "GeoEnrichment",
    "GeoSearchResult",
    "ResolveConfig",
    "ResolvedAlias",
    "ResolutionResult",
    # Lazy imports for analyze/recommend (require docpact):
    # from contract_semantics.analyze import profile_document, merge_profiles
    # from contract_semantics.recommend import recommend_contract, compare_contract
    # Lazy import for context builder (requires docpact):
    # from contract_semantics.context import build_semantic_context
]
