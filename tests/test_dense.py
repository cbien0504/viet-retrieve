"""Test cho src/dense.py.

Các test thường dùng vector giả (không cần tải model). Test đánh dấu ``slow`` dùng model e5 thật
(cần có sẵn trong cache Hugging Face) và chỉ chạy khi gọi ``pytest -m slow``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src.config import DenseConfig
from src.dense import DenseIndex, DenseRetriever, E5Embedder


def test_normalize_produces_unit_vectors_without_mutating_input() -> None:
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = DenseIndex.normalize(vectors)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert vectors[0, 0] == 3.0, "không được sửa mảng gốc"


def test_inner_product_on_normalized_vectors_equals_cosine() -> None:
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(20, 8)).astype(np.float32)
    query = rng.normal(size=(1, 8)).astype(np.float32)
    index = DenseIndex(8)
    index.add(corpus)
    scores, ids = index.search(query, top_k=20)
    cosine = (corpus @ query[0]) / (np.linalg.norm(corpus, axis=1) * np.linalg.norm(query[0]))
    assert ids[0][0] == int(np.argmax(cosine))
    assert np.allclose(scores[0], cosine[ids[0]], atol=1e-5)


def test_search_clamps_top_k_to_index_size() -> None:
    index = DenseIndex(4)
    index.add(np.eye(4, dtype=np.float32)[:3])
    scores, ids = index.search(np.ones((1, 4), dtype=np.float32), top_k=10)
    assert ids.shape == (1, 3) and (ids >= 0).all()


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    index = DenseIndex(3)
    index.add(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    path = tmp_path / "faiss.index"
    index.save(path)
    loaded = DenseIndex.load(path)
    assert loaded.size == 2 and loaded.dim == 3


class KeywordEncoder:
    """Encoder giả: vector đếm từ khoá, đủ để kiểm tra logic của DenseRetriever."""

    vocab = ("phép", "lương", "xe")

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return np.array([[q.count(w) + 1e-3 for w in self.vocab] for q in queries], dtype=np.float32)


def test_dense_retriever_returns_ids_sorted_by_similarity() -> None:
    encoder = KeywordEncoder()
    docs = {"a": "nghỉ phép phép", "b": "lương thưởng", "c": "gửi xe"}
    index = DenseIndex(3)
    index.add(encoder.encode_queries(list(docs.values())))
    retriever = DenseRetriever(list(docs), index, encoder)
    results = retriever.search("hỏi về lương", top_k=2)
    assert [cid for cid, _ in results][0] == "b"
    assert len(results) == 2 and results[0][1] >= results[1][1]


def test_dense_retriever_rejects_size_mismatch() -> None:
    index = DenseIndex(3)
    index.add(np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        DenseRetriever(["only_one"], index, KeywordEncoder())


@pytest.fixture(scope="module")
def e5() -> E5Embedder:
    embedder = E5Embedder(DenseConfig())
    try:
        embedder.model.get()
    except Exception as exc:  # pragma: no cover - phụ thuộc mạng / cache
        pytest.skip(f"Không nạp được model e5: {exc}")
    return embedder


@pytest.mark.slow
def test_e5_embeddings_capture_vietnamese_paraphrase(e5: E5Embedder) -> None:
    passages = [
        "Nhân viên được phụ cấp ăn trưa 50.000 đồng cho mỗi ngày làm việc thực tế.",
        "Công ty hỗ trợ 100% phí gửi xe máy hằng tháng tại tầng hầm.",
        "Mật khẩu phải dài tối thiểu 12 ký tự.",
    ]
    index = DenseIndex.normalize(e5.encode_passages(passages))
    query = DenseIndex.normalize(e5.encode_queries(["Công ty có trả tiền cơm trưa không?"]))
    similarities = index @ query[0]
    assert int(np.argmax(similarities)) == 0
    assert index.shape == (3, 384)
