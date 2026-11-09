"""Unit test cho src/metrics.py."""

from __future__ import annotations

import math

import pytest

from src.metrics import (
    first_relevant_rank,
    mcnemar_exact_p,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]


def test_first_relevant_rank() -> None:
    assert first_relevant_rank(RETRIEVED, {"c"}) == 3
    assert first_relevant_rank(RETRIEVED, {"z"}) is None
    assert first_relevant_rank(RETRIEVED, {"d", "b"}) == 2


def test_recall_at_k_is_hit_rate() -> None:
    assert recall_at_k(RETRIEVED, {"c"}, 1) == 0.0
    assert recall_at_k(RETRIEVED, {"c"}, 3) == 1.0
    assert recall_at_k(RETRIEVED, {"c", "z"}, 5) == 1.0
    assert recall_at_k([], {"c"}, 5) == 0.0


def test_reciprocal_rank_respects_cutoff() -> None:
    assert reciprocal_rank(RETRIEVED, {"a"}) == 1.0
    assert reciprocal_rank(RETRIEVED, {"d"}) == pytest.approx(0.25)
    assert reciprocal_rank(RETRIEVED, {"d"}, k=3) == 0.0


def test_ndcg_binary_relevance() -> None:
    assert ndcg_at_k(RETRIEVED, {"a"}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(RETRIEVED, {"b"}, 5) == pytest.approx(1 / math.log2(3))
    # hai chunk đúng ở hạng 1 và 3: DCG = 1 + 1/log2(4); IDCG = 1 + 1/log2(3)
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RETRIEVED, {"a", "c"}, 5) == pytest.approx(expected)
    assert ndcg_at_k(RETRIEVED, {"z"}, 5) == 0.0


def test_mcnemar_exact_p_values() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(4, 1) == pytest.approx(0.375)  # 2 * (C(5,0) + C(5,1)) / 2^5
    assert mcnemar_exact_p(1, 8) == pytest.approx(0.0390625)  # đối xứng theo b, c
    assert mcnemar_exact_p(3, 0) == pytest.approx(0.25)
    assert mcnemar_exact_p(5, 5) == 1.0


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 95) == 5
    assert math.isnan(percentile([], 50))
