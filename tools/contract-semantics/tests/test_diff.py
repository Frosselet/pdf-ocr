"""Tests for alias diff reporting."""

from __future__ import annotations

from contract_semantics.diff import diff_aliases, diff_all
from contract_semantics.models import ResolvedAlias, ResolutionResult


def _make_result(
    column: str = "Crop",
    matched: list[str] | None = None,
    manual_only: list[str] | None = None,
    resolved_only: list[str] | None = None,
    manual_aliases: list[str] | None = None,
) -> ResolutionResult:
    matched = matched or []
    manual_only = manual_only or []
    resolved_only = resolved_only or []
    manual = manual_aliases or matched + manual_only
    resolved = [
        ResolvedAlias(
            alias=a,
            concept_uri="http://example.com/c_1",
            concept_label="test",
            language="en",
            label_type="prefLabel",
        )
        for a in matched
    ] + [
        ResolvedAlias(
            alias=a,
            concept_uri="http://example.com/c_1",
            concept_label="test",
            language="ru",
            label_type="altLabel",
        )
        for a in resolved_only
    ]
    return ResolutionResult(
        column_name=column,
        resolved_aliases=resolved,
        manual_aliases=manual,
        matched=matched,
        manual_only=manual_only,
        resolved_only=resolved_only,
    )


class TestDiffAliases:
    def test_full_coverage(self) -> None:
        result = _make_result(matched=["wheat", "barley"], manual_only=[], resolved_only=[])
        report = diff_aliases(result)
        assert "100%" in report
        assert "wheat" in report

    def test_partial_coverage(self) -> None:
        result = _make_result(
            matched=["wheat"],
            manual_only=["spring crops"],
            resolved_only=["пшеница"],
        )
        report = diff_aliases(result)
        assert "50%" in report
        assert "spring crops" in report
        assert "пшеница" in report

    def test_zero_coverage(self) -> None:
        result = _make_result(
            matched=[],
            manual_only=["spring crops", "grains and grasses"],
            resolved_only=["wheat", "barley"],
        )
        report = diff_aliases(result)
        assert "0%" in report

    def test_resolved_only_shows_provenance(self) -> None:
        ra = ResolvedAlias(
            alias="spring wheat",
            concept_uri="http://example.com/c_1",
            concept_label="wheat",
            language="en",
            label_type="prefLabel",
            pattern="spring {label}",
        )
        result = ResolutionResult(
            column_name="Crop",
            resolved_aliases=[ra],
            manual_aliases=[],
            matched=[],
            manual_only=[],
            resolved_only=["spring wheat"],
        )
        report = diff_aliases(result)
        assert 'pattern="spring {label}"' in report


class TestDiffAll:
    def test_combined_report(self) -> None:
        results = [
            _make_result("Crop", matched=["wheat"], manual_only=["corn"]),
            _make_result("Region", matched=["Moscow"], manual_only=[]),
        ]
        report = diff_all(results)
        assert "Summary" in report
        assert "Columns resolved: 2" in report
