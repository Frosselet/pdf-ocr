"""Pydantic data models for ontology-grounded contract resolution."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConceptRef(BaseModel):
    """Reference to a concept in an external ontology."""

    uri: str
    label: str


class ResolveConfig(BaseModel):
    """Resolution configuration for a contract column.

    Controls how concept URIs are resolved into aliases via the
    appropriate ontology adapter.
    """

    source: str = "agrovoc"
    languages: list[str] = Field(default_factory=lambda: ["en"])
    label_types: list[str] = Field(default_factory=lambda: ["prefLabel", "altLabel"])
    prefix_patterns: list[str] = Field(default_factory=list)
    # GeoNames-specific
    feature_class: str | None = None
    feature_code: str | None = None
    country: str | None = None
    enrich_fields: list[str] = Field(default_factory=list)


class ResolvedAlias(BaseModel):
    """A single alias resolved from an ontology concept."""

    alias: str
    concept_uri: str
    concept_label: str
    language: str
    label_type: str
    pattern: str | None = None


class ResolutionResult(BaseModel):
    """Result of resolving a column's concept URIs to aliases."""

    column_name: str
    resolved_aliases: list[ResolvedAlias] = Field(default_factory=list)
    manual_aliases: list[str] = Field(default_factory=list)
    matched: list[str] = Field(default_factory=list)
    manual_only: list[str] = Field(default_factory=list)
    resolved_only: list[str] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of manual aliases also found via resolution."""
        if not self.manual_aliases:
            return 1.0
        return len(self.matched) / len(self.manual_aliases)


class GeoEnrichment(BaseModel):
    """Geographic metadata from GeoNames for a resolved location."""

    geoname_id: int
    name: str
    lat: float
    lng: float
    country_code: str
    admin1_code: str
    feature_class: str
    feature_code: str
    population: int | None = None


class GeoSearchResult(BaseModel):
    """A single GeoNames search result."""

    geoname_id: int
    name: str
    country_code: str
    feature_class: str
    feature_code: str
    admin1_code: str = ""
    lat: float = 0.0
    lng: float = 0.0
