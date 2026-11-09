"""Unit test cho Reciprocal Rank Fusion (src/fusion.py)."""

from __future__ import annotations

import pytest

from src.fusion import reciprocal_rank_fusion


def scores_of(fused: list[tuple[str, float]]) -> dict[str, float]:
    return dict(fused)


def test_scores_follow_rrf_formula() -> None:
    fused = scores_of(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_item_found_by_both_retrievers_ranks_first() -> None:
    bm25 = ["x", "shared", "y"]
    dense = ["z", "shared", "w"]
    assert reciprocal_rank_fusion([bm25, dense], k=60)[0][0] == "shared"


def test_k_controls_consensus_versus_top_rank() -> None:
    # "top" chỉ đứng đầu một danh sách; "both" đứng hạng 3 ở cả hai danh sách.
    rankings = [["top", "p", "both"], ["q", "r", "both"]]
    small_k = scores_of(reciprocal_rank_fusion(rankings, k=0))
    large_k = scores_of(reciprocal_rank_fusion(rankings, k=60))
    assert small_k["top"] > small_k["both"]  # 1/1 > 1/3 + 1/3
    assert large_k["both"] > large_k["top"]  # 2/63 > 1/61


def test_weights_change_the_winner() -> None:
    rankings = [["a", "b"], ["b", "a"]]
    assert reciprocal_rank_fusion(rankings, k=60, weights=[2.0, 1.0])[0][0] == "a"
    assert reciprocal_rank_fusion(rankings, k=60, weights=[1.0, 2.0])[0][0] == "b"


def test_duplicates_within_a_list_count_once_at_best_rank() -> None:
    fused = scores_of(reciprocal_rank_fusion([["a", "a", "b"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11)
    assert fused["b"] == pytest.approx(1 / 13)


def test_ties_are_broken_deterministically() -> None:
    # a và b có điểm bằng nhau (hạng 1 ở một danh sách); a xuất hiện trước.
    assert [i for i, _ in reciprocal_rank_fusion([["a"], ["b"]])] == ["a", "b"]
    assert [i for i, _ in reciprocal_rank_fusion([["b"], ["a"]])] == ["b", "a"]


def test_output_sorted_and_top_n_truncates() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"], ["d", "c"]], k=1, top_n=2)
    assert len(fused) == 2
    assert fused[0][1] >= fused[1][1]


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert scores_of(reciprocal_rank_fusion([[], ["a"]])) == pytest.approx({"a": 1 / 61})


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
