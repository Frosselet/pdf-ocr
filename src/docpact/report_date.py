"""Report date resolution from declarative contract config."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ReportDateConfig:
    """Declarative config for sourcing a report date."""

    source: str  # "filename", "timestamp", "constant", "metadata", "content", "url", "api_response"
    hint: str | None = None
    field: str | None = None
    value: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ReportDateConfig:
        return cls(
            source=data["source"],
            hint=data.get("hint"),
            field=data.get("field"),
            value=data.get("value"),
        )


def _extract_raw_value(
    config: ReportDateConfig,
    *,
    doc_path: str | None = None,
) -> str:
    """Dispatch on source type to get raw text for date extraction."""
    source = config.source

    if source == "filename":
        if not doc_path:
            raise ValueError("doc_path required for filename source")
        return Path(doc_path).name

    if source == "timestamp":
        return datetime.now().isoformat()

    if source == "constant":
        if not config.value:
            raise ValueError("value required for constant source")
        return config.value

    if source in ("metadata", "content", "url", "api_response"):
        raise NotImplementedError(
            f"report_date source '{source}' is not yet implemented"
        )

    raise ValueError(f"Unknown report_date source: {source}")


# Sources that need LLM extraction (vs. deterministic)
_LLM_SOURCES = {"filename", "content"}


async def resolve_report_date(
    config: ReportDateConfig,
    *,
    doc_path: str | None = None,
) -> str:
    """Resolve report date from config. Uses LLM for filename/content sources."""
    raw = _extract_raw_value(config, doc_path=doc_path)

    if config.source in _LLM_SOURCES:
        if not config.hint:
            raise ValueError(
                f"hint is required for LLM-based source '{config.source}'"
            )
        from baml_client.async_client import b as b_async

        result = await b_async.ExtractReportDate(raw_value=raw, hint=config.hint)
        return result.strip()

    # Deterministic sources — return raw value directly
    return raw
