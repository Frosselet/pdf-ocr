"""Format-agnostic table classification on compressed pipe-table markdown.

Classifies tables by matching header text against user-defined keyword
categories.  Operates on ``list[tuple[str, dict]]`` — the shared format
produced by both ``compress_docx_tables()`` (DOCX) and
``StructuredTable.to_compressed()`` (PDF).

Three-layer approach:
1. Keyword scoring against header text
2. ``min_data_rows`` filtering
3. Optional similarity propagation from matched to unmatched tables
"""

from __future__ import annotations

import re
import string
from functools import lru_cache


# ---------------------------------------------------------------------------
# Helpers (moved from docx_extractor.py)
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_NUM_RE = re.compile(r"^\d[\d\s,.]*$")

# Punctuation to strip from header tokens: all ASCII punctuation plus
# common Unicode punctuation (en-dash, em-dash, smart quotes).
_PUNCT_STRIP = string.punctuation + "–—\u2018\u2019\u201c\u201d"


@lru_cache(maxsize=512)
def _compile_kw_pattern(keyword: str) -> re.Pattern:
    """Compile a word-boundary regex for *keyword* with English suffix tolerance.

    Multi-word keywords are matched with flexible whitespace.  Each token
    tolerates common morphological suffixes (``s``, ``es``, ``ed``, ``ing``)
    so that ``"export"`` matches ``"exports"`` and ``"harvest"`` matches
    ``"harvested"``.  Word boundaries use lookahead/lookbehind on ASCII
    letters rather than ``\\b`` (which misfires on digits/underscores).
    """
    tokens = keyword.split()
    pattern = r"\s+".join(re.escape(t) + r"(?:s|es|ed|ing)?" for t in tokens)
    return re.compile(r"(?<![a-zA-Z])" + pattern + r"(?![a-zA-Z])", re.IGNORECASE)


def _keyword_matches(keyword: str, text: str) -> bool:
    """Return True if *keyword* appears in *text* at a word boundary."""
    return bool(_compile_kw_pattern(keyword).search(text))


# ---------------------------------------------------------------------------
# Header parsing from pipe-table markdown
# ---------------------------------------------------------------------------


def _parse_pipe_header(markdown: str) -> list[str]:
    """Extract column names from the first pipe-delimited header line.

    Skips title lines (``## ...``), blank lines, and separator lines
    (``| --- |``).  Returns an empty list if no pipe header is found.
    """
    for line in markdown.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if "|" not in stripped:
            continue
        # Skip separator rows
        cells = [c.strip() for c in stripped.split("|")]
        # Remove leading/trailing empty strings from split
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if not cells:
            continue
        if all(re.fullmatch(r"-{2,}", c) or c == "" for c in cells):
            continue
        return cells
    return []


# ---------------------------------------------------------------------------
# Tokenization for similarity propagation
# ---------------------------------------------------------------------------


def _tokenize_header_text(header_text: str) -> set[str]:
    """Extract lowercased tokens from header text, stripping years and pure numbers.

    Replaces all ASCII and common Unicode punctuation with spaces before
    splitting, so that "year–over–year" produces {"year", "over"}.
    """
    text = header_text.lower()
    # Replace punctuation characters with spaces to break compound tokens
    for ch in _PUNCT_STRIP:
        if ch in text:
            text = text.replace(ch, " ")
    tokens: set[str] = set()
    for word in text.split():
        if not word or _YEAR_RE.fullmatch(word) or _NUM_RE.fullmatch(word):
            continue
        tokens.add(word)
    return tokens


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------


def _compute_similarity(
    table_cols: int,
    table_tokens: set[str],
    profile_cols: float,
    profile_tokens: set[str],
) -> float:
    """Weighted similarity between a table and a category profile.

    50 % column-count ratio + 50 % header-token Jaccard.
    """
    col_sim = (
        min(table_cols, profile_cols) / max(table_cols, profile_cols)
        if profile_cols
        else 0.0
    )
    union = table_tokens | profile_tokens
    jaccard = len(table_tokens & profile_tokens) / len(union) if union else 0.0
    return 0.5 * col_sim + 0.5 * jaccard


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_tables(
    tables: list[tuple[str, dict]],
    categories: dict[str, list[str]],
    *,
    min_data_rows: int = 0,
    propagate: bool = False,
    propagate_threshold: float = 0.3,
) -> list[dict]:
    """Classify tables from compressed pipe-table markdown + metadata tuples.

    Args:
        tables: ``(markdown, meta)`` tuples as produced by
            ``compress_docx_tables()`` or ``StructuredTable.to_compressed()``.
            Each *meta* dict must contain ``table_index``, ``title``,
            ``row_count``, and ``col_count``.
        categories: Mapping of category name → list of keywords.
        min_data_rows: Tables with fewer data rows are forced to ``"other"``.
        propagate: If True, propagate categories to unmatched tables via
            structural similarity.
        propagate_threshold: Minimum similarity (0-1) for propagation.

    Returns:
        List of dicts with ``index``, ``category``, ``title``, ``rows``,
        ``cols``.
    """
    # Normalize pipe-table text before classification
    from pdf_ocr.normalize import normalize_pipe_table
    tables = [(normalize_pipe_table(md), meta) for md, meta in tables]

    # Lowercase all keywords once
    lower_cats = {
        name: [kw.lower() for kw in kws]
        for name, kws in categories.items()
    }

    results: list[dict] = []
    _table_infos: list[tuple[int, set[str]]] = []  # (col_count, header_tokens)

    for markdown, meta in tables:
        table_idx = meta["table_index"]
        title = meta.get("title")
        row_count = meta.get("row_count", 0)
        col_count = meta.get("col_count", 0)

        # Parse column names from the pipe-table header
        column_names = _parse_pipe_header(markdown)

        # Build header text: join columns, collapse compound separators,
        # prepend title
        header_text = " ".join(column_names).lower().replace(" / ", " ")
        if title:
            header_text = title.lower() + " " + header_text

        # Tokenize for similarity propagation
        header_tokens = _tokenize_header_text(header_text)
        _table_infos.append((col_count, header_tokens))

        # min_data_rows filter
        if row_count < min_data_rows:
            results.append({
                "index": table_idx,
                "category": "other",
                "title": title,
                "rows": row_count,
                "cols": col_count,
            })
            continue

        # Classify: highest keyword-hit score wins
        category = "other"
        best_score = 0
        for cat_name, keywords in lower_cats.items():
            score = sum(1 for kw in keywords if _keyword_matches(kw, header_text))
            if score > best_score:
                best_score = score
                category = cat_name

        results.append({
            "index": table_idx,
            "category": category,
            "title": title,
            "rows": row_count,
            "cols": col_count,
        })

    # Layer 3: Similarity propagation
    if propagate:
        # Build profiles from keyword-matched tables
        profiles: dict[str, dict] = {}
        for i, result in enumerate(results):
            cat = result["category"]
            if cat == "other":
                continue
            t_cols, t_tokens = _table_infos[i]
            if cat not in profiles:
                profiles[cat] = {"col_counts": [], "tokens": set()}
            profiles[cat]["col_counts"].append(t_cols)
            profiles[cat]["tokens"].update(t_tokens)

        # Compute mean col count per profile
        for p in profiles.values():
            p["mean_cols"] = sum(p["col_counts"]) / len(p["col_counts"])

        # Propagate to "other" tables
        for i, result in enumerate(results):
            if result["category"] != "other":
                continue
            t_cols, t_tokens = _table_infos[i]
            best_cat = "other"
            best_sim = propagate_threshold
            for cat, prof in profiles.items():
                sim = _compute_similarity(
                    t_cols, t_tokens, prof["mean_cols"], prof["tokens"],
                )
                if sim > best_sim:
                    best_sim = sim
                    best_cat = cat
            result["category"] = best_cat

    return results
