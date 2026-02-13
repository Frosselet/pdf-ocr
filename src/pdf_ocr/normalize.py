"""Centralized pipe-table text normalization.

Applied after extraction and before consumption (classification, interpretation)
to ensure consistent text regardless of which extractor produced it.

All transformations are lossless and idempotent.
"""

from __future__ import annotations

import re

# Character-level replacements via str.translate
_TRANSLATE = str.maketrans({
    "\u00a0": " ",    # NBSP → space
    "\u2018": "'",    # left single quote → ASCII
    "\u2019": "'",    # right single quote → ASCII
    "\u201c": '"',    # left double quote → ASCII
    "\u201d": '"',    # right double quote → ASCII
    "\u2013": "-",    # en-dash → hyphen-minus
    "\u2014": "-",    # em-dash → hyphen-minus
    "\u200b": None,   # ZWSP → remove
    "\u200c": None,   # ZWNJ → remove
    "\u200d": None,   # ZWJ → remove
    "\ufeff": None,   # BOM → remove
    "\u2060": None,   # Word Joiner → remove
})

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_pipe_table(text: str) -> str:
    """Normalize a pipe-table markdown string for consistent downstream processing.

    Performs:
    1. NBSP → regular space
    2. Smart quotes → ASCII equivalents
    3. En-dash / em-dash → hyphen-minus
    4. Zero-width characters removed (ZWSP, ZWNJ, ZWJ, BOM, WJ)
    5. Runs of 2+ spaces/tabs → single space (per line)
    6. Trailing whitespace stripped per line
    """
    text = text.translate(_TRANSLATE)
    lines = text.split("\n")
    lines = [_MULTI_SPACE_RE.sub(" ", line).rstrip() for line in lines]
    return "\n".join(lines)
