"""Test cho src/reranker.py với cross-encoder giả (không cần tải model)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.config import RerankerConfig
from src.lazy import LazyModel
from src.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Điểm = số ký tự chung đầu tiên giữa câu hỏi và đoạn (đủ để kiểm tra thứ tự)."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int, **_: object) -> np.ndarray:
        self.calls.append(len(pairs))
        return np.array([float(passage.count(query)) for query, passage in pairs])


def make_reranker(release_after_use: bool = True) -> tuple[CrossEncoderReranker, FakeCrossEncoder]:
    fake = FakeCrossEncoder()
    reranker = CrossEncoderReranker(RerankerConfig(release_after_use=release_after_use))
    reranker.model = LazyModel("fake-ce", lambda: fake, release_after_use=release_after_use)
    return reranker, fake


def test_rerank_sorts_by_score_and_keeps_ties_stable() -> None:
    reranker, _ = make_reranker()
    candidates = [("a", "x"), ("b", "xe xe"), ("c", "y"), ("d", "xe")]
    result = reranker.rerank("xe", candidates)
    assert [cid for cid, _ in result] == ["b", "d", "a", "c"]
    assert [score for _, score in result] == [2.0, 1.0, 0.0, 0.0]


def test_empty_candidates_do_not_load_model() -> None:
    reranker, _ = make_reranker()
    assert reranker.rerank("xe", []) == []
    assert reranker.model.load_count == 0


def test_model_is_released_after_each_call_by_default() -> None:
    reranker, fake = make_reranker(release_after_use=True)
    reranker.rerank("xe", [("a", "xe")])
    reranker.rerank("xe", [("a", "xe")])
    assert not reranker.model.is_loaded and reranker.model.load_count == 2
    assert fake.calls == [1, 1]


def test_model_is_kept_when_configured() -> None:
    reranker, _ = make_reranker(release_after_use=False)
    reranker.rerank("xe", [("a", "xe")])
    reranker.rerank("xe", [("a", "xe")])
    assert reranker.model.is_loaded and reranker.model.load_count == 1


def test_warmup_reports_load_time_only_when_loading() -> None:
    reranker, _ = make_reranker(release_after_use=False)
    assert reranker.warmup() is not None
    assert reranker.warmup() is None
