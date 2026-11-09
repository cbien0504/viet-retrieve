"""Ghép các bước: dựng index (chunk + BM25 + FAISS) và truy vấn (BM25 / dense / RRF / RRF + reranker)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from src.chunking import SUPPORTED_SUFFIXES, Chunk, chunk_documents, load_chunks, load_documents, save_chunks
from src.config import MODES, Config
from src.dense import DenseIndex, DenseRetriever, E5Embedder
from src.fusion import reciprocal_rank_fusion
from src.reranker import CrossEncoderReranker
from src.sparse import BM25Retriever, VietnameseTokenizer

logger = logging.getLogger(__name__)
T = TypeVar("T")

CHUNKS_FILE = "chunks.jsonl"
BM25_TOKENS_FILE = "bm25_tokens.json"
FAISS_FILE = "faiss.index"
MANIFEST_FILE = "manifest.json"

DENSE_MODES = frozenset({"dense", "hybrid", "hybrid_rerank"})


@dataclass
class SearchHit:
    """Một kết quả trả về, kèm điểm/hạng của từng thành phần để giải thích."""

    rank: int
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass
class SearchResponse:
    query: str
    mode: str
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)


def data_fingerprint(raw_dir: Path) -> str:
    """SHA-256 trên (đường dẫn, nội dung) của mọi file dữ liệu: đổi dữ liệu -> đổi fingerprint."""
    digest = hashlib.sha256()
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES):
        digest.update(path.relative_to(raw_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest_key(config: Config) -> dict[str, object]:
    """Những gì quyết định nội dung chunk; khác giá trị này thì phải build lại."""
    return {
        "data_fingerprint": data_fingerprint(config.resolve(config.paths.raw_dir)),
        "chunking": config.to_dict()["chunking"],
    }


def _dense_key(config: Config) -> dict[str, object]:
    return {"model_name": config.dense.model_name, "passage_prefix": config.dense.passage_prefix}


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def build_index(config: Config, with_dense: bool = True, embedder: E5Embedder | None = None) -> dict[str, object]:
    """Đọc dữ liệu, chia đoạn, tách từ cho BM25, mã hoá FAISS (tuỳ chọn) và lưu vào ``paths.index_dir``."""
    index_dir = config.resolve(config.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / MANIFEST_FILE).unlink(missing_ok=True)  # manifest chỉ được ghi lại khi build thành công
    started = time.perf_counter()

    documents = load_documents(config.resolve(config.paths.raw_dir))
    if not documents:
        raise ValueError(f"Không có văn bản .md/.txt nào trong {config.paths.raw_dir}")
    chunks = chunk_documents(documents, config.chunking)
    save_chunks(chunks, index_dir / CHUNKS_FILE)
    logger.info("Đã chia %d văn bản thành %d đoạn", len(documents), len(chunks))

    tokenizer = VietnameseTokenizer.from_config(config.sparse)
    texts = [c.index_text(config.chunking.include_headings) for c in chunks]
    _write_json(
        index_dir / BM25_TOKENS_FILE,
        {
            "tokenizer": tokenizer.signature,
            "chunk_ids": [c.chunk_id for c in chunks],
            "tokens": tokenizer.tokenize_many(texts),
        },
    )

    dense_info: dict[str, object] | None = None
    (index_dir / FAISS_FILE).unlink(missing_ok=True)
    if with_dense:
        embedder = embedder or E5Embedder(config.dense)
        dense_started = time.perf_counter()
        index = DenseRetriever.build_index(texts, embedder, config.dense.batch_size)
        index.save(index_dir / FAISS_FILE)
        dense_info = {
            **_dense_key(config),
            "dim": index.dim,
            "seconds": round(time.perf_counter() - dense_started, 2),
        }
        logger.info("Đã mã hoá %d đoạn (dim=%d) trong %.1fs", index.size, index.dim, dense_info["seconds"])

    manifest: dict[str, object] = {
        **_manifest_key(config),
        "dense": dense_info,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    _write_json(index_dir / MANIFEST_FILE, manifest)
    logger.info("Đã lưu index vào %s (%.1fs)", index_dir, manifest["build_seconds"])
    return manifest


def read_manifest(config: Config) -> dict[str, object] | None:
    path = config.resolve(config.paths.index_dir) / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def index_is_current(config: Config) -> bool:
    """Index tồn tại và khớp dữ liệu + cấu hình hiện tại.

    Index dựng không kèm dense (``build --no-dense``) vẫn được coi là hợp lệ cho chế độ BM25.
    """
    index_dir = config.resolve(config.paths.index_dir)
    manifest = read_manifest(config)
    if manifest is None or not (index_dir / CHUNKS_FILE).is_file():
        return False
    if any(manifest.get(k) != v for k, v in _manifest_key(config).items()):
        return False
    dense = manifest.get("dense")
    if dense is None:
        return True
    return (index_dir / FAISS_FILE).is_file() and all(dense.get(k) == v for k, v in _dense_key(config).items())


class RetrievalPipeline:
    """Truy vấn trên index đã dựng theo các chế độ trong ``MODES``.

    - ``bm25``: chỉ BM25.
    - ``dense``: chỉ embedding + FAISS.
    - ``hybrid``: lấy ``fusion.candidates_per_retriever`` ứng viên từ mỗi retriever rồi gộp bằng RRF.
    - ``hybrid_rerank``: như ``hybrid``, sau đó cross-encoder xếp hạng lại ``reranker.candidates`` ứng viên đầu.
    """

    def __init__(
        self,
        config: Config,
        chunks: list[Chunk],
        bm25: BM25Retriever,
        dense: DenseRetriever | None = None,
        embedder: E5Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.config = config
        self.chunks = chunks
        self.chunk_by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker

    @classmethod
    def load(
        cls,
        config: Config,
        auto_build: bool = True,
        embedder: E5Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> RetrievalPipeline:
        """Nạp index từ đĩa; tự build lại nếu thiếu hoặc lỗi thời (khi ``auto_build``).

        Không model nào được nạp ở đây: embedder chỉ nạp ở truy vấn dense/hybrid đầu tiên, reranker chỉ nạp
        khi có truy vấn ``hybrid_rerank``. ``embedder``/``reranker`` cho phép truyền thành phần khác (ví dụ khi test).
        """
        embedder = embedder or E5Embedder(config.dense)
        reranker = reranker or CrossEncoderReranker(config.reranker)
        if not index_is_current(config):
            if not auto_build:
                raise FileNotFoundError("Index chưa có hoặc đã cũ. Chạy: python -m src.cli build")
            logger.info("Index chưa có hoặc đã cũ -> build lại")
            build_index(config, with_dense=True, embedder=embedder)
        index_dir = config.resolve(config.paths.index_dir)
        chunks = load_chunks(index_dir / CHUNKS_FILE)
        chunk_ids = [c.chunk_id for c in chunks]

        tokenizer = VietnameseTokenizer.from_config(config.sparse)
        cached = json.loads((index_dir / BM25_TOKENS_FILE).read_text(encoding="utf-8"))
        if cached.get("tokenizer") == tokenizer.signature and cached.get("chunk_ids") == chunk_ids:
            tokens = cached["tokens"]
        else:
            logger.info("Cấu hình tokenizer đã đổi -> tách từ lại")
            tokens = tokenizer.tokenize_many([c.index_text(config.chunking.include_headings) for c in chunks])
        bm25 = BM25Retriever(chunk_ids, tokens, tokenizer, config.sparse.k1, config.sparse.b)

        dense = None
        if (index_dir / FAISS_FILE).is_file():
            dense = DenseRetriever(chunk_ids, DenseIndex.load(index_dir / FAISS_FILE), embedder)
        pipeline = cls(config, chunks, bm25, dense, embedder if dense else None, reranker)
        pipeline.warmup("bm25")
        return pipeline

    @property
    def available_modes(self) -> list[str]:
        return [m for m in MODES if m not in DENSE_MODES or self.dense is not None]

    def warmup(self, mode: str) -> dict[str, float]:
        """Nạp trước những gì ``mode`` cần; trả về thời gian nạp (giây) của các model vừa được nạp."""
        loaded: dict[str, float] = {}
        self.bm25.tokenizer.tokenize("khởi động")  # lần gọi đầu của underthesea nạp mô hình CRF
        if mode in DENSE_MODES and self.embedder is not None:
            seconds = self.embedder.warmup()
            if seconds is not None:
                loaded["embedder"] = seconds
        if mode == "hybrid_rerank" and self.reranker is not None:
            seconds = self.reranker.warmup()
            if seconds is not None:
                loaded["reranker"] = seconds
        return loaded

    @contextmanager
    def hold_models(self) -> Iterator[None]:
        """Giữ các model đã nạp lười trong suốt khối ``with`` (dùng cho một batch truy vấn).

        Model vẫn chỉ được nạp khi cần; khi rời khối, model có ``release_after_use`` sẽ được giải phóng.
        """
        with ExitStack() as stack:
            if self.embedder is not None:
                stack.enter_context(self.embedder.model.hold())
            if self.reranker is not None:
                stack.enter_context(self.reranker.model.hold())
            yield

    def release_models(self) -> None:
        """Giải phóng mọi model đang nạp."""
        if self.embedder is not None:
            self.embedder.model.release()
        if self.reranker is not None:
            self.reranker.model.release()

    def _require_dense(self) -> DenseRetriever:
        if self.dense is None:
            raise RuntimeError("Index không có phần dense (đã build với --no-dense). Chạy: python -m src.cli build")
        return self.dense

    def search(self, query: str, mode: str | None = None, top_k: int | None = None) -> SearchResponse:
        """Tìm top-k đoạn liên quan đến ``query`` theo ``mode`` (mặc định lấy từ config)."""
        mode = mode or self.config.retrieval.default_mode
        top_k = top_k or self.config.retrieval.top_k
        if mode not in MODES:
            raise ValueError(f"mode phải thuộc {MODES}")
        timings: dict[str, float] = {}
        started = time.perf_counter()
        lazy_models = {"dense_load_ms": self.embedder, "rerank_load_ms": self.reranker}
        loads_before = {key: c.model.load_count for key, c in lazy_models.items() if c is not None}

        def timed(name: str, fn: Callable[[], T]) -> T:
            t0 = time.perf_counter()
            result = fn()
            timings[name] = (time.perf_counter() - t0) * 1000
            return result

        if mode == "bm25":
            results = timed("bm25_ms", lambda: self.bm25.search(query, top_k))
            hits = [
                SearchHit(rank=i, chunk=self.chunk_by_id[cid], score=s, bm25_rank=i, bm25_score=s)
                for i, (cid, s) in enumerate(results, start=1)
            ]
        elif mode == "dense":
            dense = self._require_dense()
            results = timed("dense_ms", lambda: dense.search(query, top_k))
            hits = [
                SearchHit(rank=i, chunk=self.chunk_by_id[cid], score=s, dense_rank=i, dense_score=s)
                for i, (cid, s) in enumerate(results, start=1)
            ]
        else:
            hits = self._hybrid(query, mode, top_k, timed)
        # Model nạp lười ngay trong truy vấn này: ghi riêng thời gian nạp (đã nằm trong dense_ms / rerank_ms).
        for key, component in lazy_models.items():
            if component is not None and component.model.load_count > loads_before[key]:
                timings[key] = (component.model.last_load_seconds or 0.0) * 1000
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return SearchResponse(query=query, mode=mode, hits=hits, timings_ms=timings)

    def _hybrid(
        self, query: str, mode: str, top_k: int, timed: Callable[[str, Callable[[], Any]], Any]
    ) -> list[SearchHit]:
        dense = self._require_dense()
        n_candidates = max(self.config.fusion.candidates_per_retriever, top_k)
        bm25_results = timed("bm25_ms", lambda: self.bm25.search(query, n_candidates))
        dense_results = timed("dense_ms", lambda: dense.search(query, n_candidates))
        fused = timed(
            "fusion_ms",
            lambda: reciprocal_rank_fusion(
                [[cid for cid, _ in bm25_results], [cid for cid, _ in dense_results]],
                k=self.config.fusion.rrf_k,
                weights=[self.config.fusion.bm25_weight, self.config.fusion.dense_weight],
            ),
        )
        bm25_info = {cid: (rank, score) for rank, (cid, score) in enumerate(bm25_results, start=1)}
        dense_info = {cid: (rank, score) for rank, (cid, score) in enumerate(dense_results, start=1)}
        rrf_score = dict(fused)

        # (chunk_id, điểm cuối, điểm rerank)
        ordered: list[tuple[str, float, float | None]]
        if mode == "hybrid":
            ordered = [(cid, score, None) for cid, score in fused[:top_k]]
        else:
            if self.reranker is None:
                raise RuntimeError("Pipeline không có reranker")
            reranker = self.reranker
            candidates = fused[: self.config.reranker.candidates]
            include = self.config.chunking.include_headings
            reranked = timed(
                "rerank_ms",
                lambda: reranker.rerank(
                    query, [(cid, self.chunk_by_id[cid].index_text(include)) for cid, _ in candidates]
                ),
            )
            ordered = [(cid, score, score) for cid, score in reranked]
            # nếu top_k > số ứng viên được rerank: nối phần còn lại theo thứ tự RRF
            ordered += [(cid, score, None) for cid, score in fused[len(candidates) :]]
            ordered = ordered[:top_k]

        hits: list[SearchHit] = []
        for rank, (cid, score, rerank_score) in enumerate(ordered, start=1):
            bm25_rank, bm25_score = bm25_info.get(cid, (None, None))
            dense_rank, dense_score = dense_info.get(cid, (None, None))
            hits.append(
                SearchHit(
                    rank=rank,
                    chunk=self.chunk_by_id[cid],
                    score=score,
                    bm25_rank=bm25_rank,
                    bm25_score=bm25_score,
                    dense_rank=dense_rank,
                    dense_score=dense_score,
                    rrf_score=rrf_score[cid],
                    rerank_score=rerank_score,
                )
            )
        return hits
