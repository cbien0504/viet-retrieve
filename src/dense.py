"""Truy hồi dày (dense): embedding multilingual-e5 + FAISS ``IndexFlatIP``.

Vector được chuẩn hoá L2 trước khi đưa vào index và trước khi tìm kiếm, nên tích vô hướng = cosine.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import faiss
import numpy as np

from src.config import DenseConfig
from src.lazy import LazyModel, quiet_model_loading

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class QueryEncoder(Protocol):
    """Giao diện tối thiểu mà ``DenseRetriever`` cần (cho phép thay bằng encoder giả khi test)."""

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray: ...


class E5Embedder:
    """multilingual-e5 qua sentence-transformers: nạp lười, tự thêm tiền tố ``query: `` / ``passage: ``."""

    def __init__(self, config: DenseConfig) -> None:
        self.config = config
        self.model: LazyModel[SentenceTransformer] = LazyModel(
            f"embedder ({config.model_name})", self._load, release_after_use=config.release_after_use
        )

    def _load(self) -> SentenceTransformer:
        from sentence_transformers import (
            SentenceTransformer,  # import chậm (~vài giây) nên chỉ import khi cần
        )

        quiet_model_loading()
        model = SentenceTransformer(self.config.model_name, device=self.config.device)
        model.max_seq_length = self.config.max_seq_length
        return model

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        inputs = [prefix + text for text in texts]

        def encode(model: SentenceTransformer) -> np.ndarray:
            return model.encode(
                inputs,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,  # chuẩn hoá tường minh ở DenseIndex
                show_progress_bar=False,
            )

        return np.ascontiguousarray(self.model.run(encode), dtype=np.float32)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode(queries, self.config.query_prefix)

    def encode_passages(self, passages: Sequence[str]) -> np.ndarray:
        return self._encode(passages, self.config.passage_prefix)

    def warmup(self) -> float | None:
        """Nạp model (nếu chưa) và chạy thử một lượt; trả về thời gian nạp nếu vừa nạp."""
        was_loaded = self.model.is_loaded
        self.encode_queries(["khởi động"])
        return None if was_loaded else self.model.last_load_seconds


class DenseIndex:
    """FAISS ``IndexFlatIP`` (tìm kiếm chính xác) trên vector đã chuẩn hoá L2."""

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)

    @property
    def dim(self) -> int:
        return int(self.index.d)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """Bản sao float32 đã chuẩn hoá L2 theo từng hàng (không sửa mảng đầu vào)."""
        array = np.array(vectors, dtype=np.float32, copy=True, order="C")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        faiss.normalize_L2(array)
        return array

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(self.normalize(vectors))

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Trả về (điểm cosine, chỉ số) dạng mảng [n_query, k]; chỉ số -1 nghĩa là không đủ kết quả."""
        k = max(1, min(top_k, self.size))
        return self.index.search(self.normalize(queries), k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        index = faiss.read_index(str(path))
        dense = cls(index.d)
        dense.index = index
        return dense


class DenseRetriever:
    """Tìm chunk gần câu hỏi nhất theo cosine giữa embedding câu hỏi và embedding đoạn."""

    def __init__(self, chunk_ids: Sequence[str], index: DenseIndex, encoder: QueryEncoder) -> None:
        if index.size != len(chunk_ids):
            raise ValueError(f"FAISS có {index.size} vector nhưng có {len(chunk_ids)} chunk")
        self.chunk_ids = list(chunk_ids)
        self.index = index
        self.encoder = encoder

    @staticmethod
    def build_index(texts: Sequence[str], embedder: E5Embedder, batch_size: int) -> DenseIndex:
        """Mã hoá kho văn bản theo từng lô ``batch_size`` (không đưa cả kho vào một lần) và thêm dần vào FAISS."""
        index: DenseIndex | None = None
        with embedder.model.hold():
            for start in range(0, len(texts), batch_size):
                vectors = embedder.encode_passages(texts[start : start + batch_size])
                if index is None:
                    index = DenseIndex(vectors.shape[1])
                index.add(vectors)
                logger.debug("Đã mã hoá %d/%d đoạn", min(start + batch_size, len(texts)), len(texts))
        if index is None:
            raise ValueError("Không có văn bản nào để mã hoá")
        return index

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores, ids = self.index.search(self.encoder.encode_queries([query]), top_k)
        return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]
