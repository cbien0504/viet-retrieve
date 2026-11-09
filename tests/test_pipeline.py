"""Test tích hợp pipeline với embedder / reranker giả (không cần tải model)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from src.config import Config, PathsConfig, SparseConfig
from src.lazy import LazyModel
from src.pipeline import RetrievalPipeline, build_index, index_is_current

DOCS = {
    "nghi_phep.md": (
        "# Nghỉ phép\n\n## Số ngày\n\nNhân viên có 12 ngày phép năm.\n\n## Đăng ký\n\nĐăng ký phép trên cổng nội bộ."
    ),
    "luong.md": "# Lương\n\n## Kỳ trả lương\n\nLương trả vào ngày 5 hằng tháng.",
    "gui_xe.md": "# Gửi xe\n\n## Phí gửi xe\n\nCông ty hỗ trợ phí gửi xe máy.",
}
VOCAB = ("phép", "lương", "xe", "ngày", "đăng ký")


def keyword_vectors(texts: Sequence[str]) -> np.ndarray:
    return np.array([[t.lower().count(w) + 1e-3 for w in VOCAB] for t in texts], dtype=np.float32)


class FakeEmbedder:
    """Giả lập E5Embedder: vector đếm từ khoá."""

    def __init__(self) -> None:
        self.model: LazyModel[object] = LazyModel("fake-embedder", object)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.model.run(lambda _: keyword_vectors(queries))

    def encode_passages(self, passages: Sequence[str]) -> np.ndarray:
        return self.model.run(lambda _: keyword_vectors(passages))

    def warmup(self) -> float | None:
        was_loaded = self.model.is_loaded
        self.model.get()
        return None if was_loaded else 0.0


class FakeReranker:
    """Giả lập CrossEncoderReranker: điểm = số từ của câu hỏi xuất hiện trong đoạn; giải phóng sau mỗi lần dùng."""

    def __init__(self) -> None:
        self.model: LazyModel[object] = LazyModel("fake-reranker", object, release_after_use=True)

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        words = query.lower().split()

        def score(_: object) -> list[tuple[str, float]]:
            scored = [(cid, float(sum(w in text.lower() for w in words))) for cid, text in candidates]
            return sorted(scored, key=lambda x: -x[1])

        return self.model.run(score)

    def warmup(self) -> float | None:
        was_loaded = self.model.is_loaded
        self.model.run(lambda _: None)
        return None if was_loaded else 0.0


@pytest.fixture()
def config(tmp_path: Path) -> Config:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, text in DOCS.items():
        (raw / name).write_text(text, encoding="utf-8")
    return Config(
        paths=PathsConfig(raw_dir="raw", index_dir="index"),
        sparse=SparseConfig(tokenizer="syllable"),
        root=tmp_path,
    )


def load_fake(config: Config) -> RetrievalPipeline:
    embedder, reranker = FakeEmbedder(), FakeReranker()
    if not index_is_current(config):
        build_index(config, with_dense=True, embedder=embedder)  # type: ignore[arg-type]
    return RetrievalPipeline.load(config, embedder=embedder, reranker=reranker)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_finds_the_relevant_chunk(config: Config, mode: str) -> None:
    pipeline = load_fake(config)
    response = pipeline.search("Phí gửi xe máy được hỗ trợ không?", mode=mode, top_k=3)
    assert response.hits[0].chunk.doc_id == "gui_xe"
    assert [h.rank for h in response.hits] == list(range(1, len(response.hits) + 1))
    assert "total_ms" in response.timings_ms


def test_hybrid_hits_explain_their_components(config: Config) -> None:
    pipeline = load_fake(config)
    hybrid = pipeline.search("lương trả ngày nào", mode="hybrid", top_k=2)
    top = hybrid.hits[0]
    assert top.rrf_score is not None and top.score == top.rrf_score
    assert top.bm25_rank is not None and top.dense_rank is not None
    reranked = pipeline.search("lương trả ngày nào", mode="hybrid_rerank", top_k=2)
    assert reranked.hits[0].rerank_score is not None
    assert {"bm25_ms", "dense_ms", "fusion_ms", "rerank_ms"} <= set(reranked.timings_ms)


def test_reranker_loaded_only_for_rerank_mode_and_released_after(config: Config) -> None:
    pipeline = load_fake(config)
    reranker = pipeline.reranker
    assert reranker is not None
    for mode in ("bm25", "dense", "hybrid"):
        pipeline.search("phép năm", mode=mode)
    assert reranker.model.load_count == 0, "reranker không được nạp khi chế độ không cần"
    pipeline.search("phép năm", mode="hybrid_rerank")
    assert reranker.model.load_count == 1 and not reranker.model.is_loaded


def test_hold_models_keeps_reranker_for_a_batch(config: Config) -> None:
    pipeline = load_fake(config)
    reranker = pipeline.reranker
    assert reranker is not None
    with pipeline.hold_models():
        pipeline.warmup("hybrid_rerank")
        for query in ("phép năm", "lương", "gửi xe"):
            pipeline.search(query, mode="hybrid_rerank")
        assert reranker.model.is_loaded and reranker.model.load_count == 1
    assert not reranker.model.is_loaded


def test_index_is_rebuilt_when_data_changes(config: Config) -> None:
    load_fake(config)
    assert index_is_current(config)
    (config.resolve("raw") / "luong.md").write_text("# Lương\n\nLương trả vào ngày 10.", encoding="utf-8")
    assert not index_is_current(config)
    pipeline = load_fake(config)
    hit = pipeline.search("lương trả ngày nào", mode="bm25", top_k=1).hits[0]
    assert "ngày 10" in hit.chunk.text


def test_bm25_only_index_disables_dense_modes(config: Config) -> None:
    build_index(config, with_dense=False)
    pipeline = RetrievalPipeline.load(config, embedder=FakeEmbedder(), reranker=FakeReranker())  # type: ignore[arg-type]
    assert pipeline.available_modes == ["bm25"]
    assert pipeline.search("gửi xe", mode="bm25").hits
    with pytest.raises(RuntimeError):
        pipeline.search("gửi xe", mode="hybrid")


def test_invalid_mode_raises(config: Config) -> None:
    with pytest.raises(ValueError):
        load_fake(config).search("gửi xe", mode="magic")
