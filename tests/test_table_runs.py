"""Tests for _detect_table_runs post-aggregation overlap tightening.

After a single-span numeric continuation (aggregation total), the next
multi-span row must satisfy BOTH overlap_count >= 2 AND overlap_ratio >= 0.5
to continue the run. This prevents structurally unrelated tables (like CBH's
"Stock at Port") from bleeding into the preceding shipping table.
"""

from __future__ import annotations

from docpact.compress import _detect_table_runs


def _make_args(
    rows: list[tuple[int, list[int], int, str]],
    min_table_rows: int = 2,
    col_tolerance: int = 5,
) -> dict:
    """Build _detect_table_runs kwargs from a compact row spec.

    Each row is (row_index, col_positions, span_count, text).
    """
    row_indices = [r[0] for r in rows]
    col_positions = {r[0]: r[1] for r in rows}
    span_counts = {r[0]: r[2] for r in rows}
    row_texts = {r[0]: r[3] for r in rows}
    return dict(
        row_indices=row_indices,
        col_positions=col_positions,
        span_counts=span_counts,
        min_table_rows=min_table_rows,
        col_tolerance=col_tolerance,
        row_texts=row_texts,
    )


class TestPostAggregationOverlap:
    """Tests for stricter overlap check after single-span continuation."""

    def test_post_aggregation_breaks_on_low_overlap(self) -> None:
        """After aggregation total, rows with 2/8 column overlap → two runs."""
        # Table A: 3 multi-span data rows at positions 10,20,30,...,80
        cols_a = [10, 20, 30, 40, 50, 60, 70, 80]
        rows = [
            (0, cols_a, 8, "| Melbourne | STAR | ..."),
            (1, cols_a, 8, "| Sydney | MOON | ..."),
            # Aggregation total: single span
            (2, [10], 1, "90,000"),
            # Table B: 8 columns but only 2 overlap with table A (10, 20)
            # ratio = 2/8 = 0.25 — below 0.5
            (3, [10, 20, 100, 110, 120, 130, 140, 150], 8, "| Stock | data |"),
            (4, [10, 20, 100, 110, 120, 130, 140, 150], 8, "| Stock | more |"),
        ]
        runs = _detect_table_runs(**_make_args(rows))
        assert len(runs) == 2
        assert runs[0] == [0, 1, 2]  # table A + aggregation
        assert runs[1] == [3, 4]      # table B

    def test_post_aggregation_continues_on_high_overlap(self) -> None:
        """After aggregation total, rows with 6/8 overlap → one run."""
        cols = [10, 20, 30, 40, 50, 60, 70, 80]
        rows = [
            (0, cols, 8, "| data row 1 |"),
            (1, cols, 8, "| data row 2 |"),
            # Aggregation total
            (2, [10], 1, "90,000"),
            # Same structure continues (6/8 = 0.75 overlap)
            (3, [10, 20, 30, 40, 50, 60, 90, 95], 8, "| more data |"),
            (4, [10, 20, 30, 40, 50, 60, 90, 95], 8, "| more data |"),
        ]
        runs = _detect_table_runs(**_make_args(rows))
        assert len(runs) == 1
        assert runs[0] == [0, 1, 2, 3, 4]

    def test_normal_low_ratio_still_joins_without_aggregation(self) -> None:
        """Without prior aggregation, 2 overlapping columns join (OR behavior)."""
        cols_a = [10, 20, 30, 40, 50, 60, 70, 80]
        rows = [
            (0, cols_a, 8, "| data row 1 |"),
            (1, cols_a, 8, "| data row 2 |"),
            # No single-span aggregation in between
            # Next row has 2/8 overlap (ratio 0.25) — normal OR logic passes
            (2, [10, 20, 100, 110, 120, 130, 140, 150], 8, "| stock data |"),
        ]
        runs = _detect_table_runs(**_make_args(rows))
        # Normal behavior: overlap_count >= 2 is enough
        assert len(runs) == 1
        assert runs[0] == [0, 1, 2]

    def test_consecutive_single_span_rows_then_unrelated(self) -> None:
        """Two aggregation totals then unrelated rows → run breaks."""
        cols = [10, 20, 30, 40, 50, 60, 70, 80]
        rows = [
            (0, cols, 8, "| data row |"),
            (1, cols, 8, "| data row |"),
            (2, [10], 1, "50,000"),
            (3, [10], 1, "90,000"),
            # Unrelated table with low overlap
            (4, [10, 20, 100, 110, 120, 130, 140, 150], 8, "| stock |"),
            (5, [10, 20, 100, 110, 120, 130, 140, 150], 8, "| stock |"),
        ]
        runs = _detect_table_runs(**_make_args(rows))
        assert len(runs) == 2
        assert 4 not in runs[0]
