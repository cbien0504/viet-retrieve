"""Xếp hạng lại (rerank) bằng cross-encoder.

Cross-encoder đọc đồng thời (câu hỏi, đoạn) nên chính xác hơn bi-encoder nhưng chậm hơn nhiều, vì vậy chỉ
áp dụng cho một số ít ứng viên sau RRF. Model nặng (~1.1 GB với bge-reranker-base) nên được nạp lười ở truy
vấn đầu tiên cần xếp hạng lại và (mặc định) giải phóng ngay sau khi xong — xem ``reranker.release_after_use``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.config import RerankerConfig
from src.lazy import LazyModel, quiet_model_loading

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Bao ``sentence_transformers.CrossEncoder`` với cơ chế nạp lười / giải phóng."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self.model: LazyModel[CrossEncoder] = LazyModel(
            f"reranker ({config.model_name})", self._load, release_after_use=config.release_after_use
        )

    def _load(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder  # import chậm nên chỉ import khi cần

        quiet_model_loading()
        return CrossEncoder(self.config.model_name, max_length=self.config.max_length, device=self.config.device)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Điểm liên quan của từng (câu hỏi, đoạn); điểm cao hơn = liên quan hơn."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, passage) for passage in passages]

        def predict(model: CrossEncoder) -> np.ndarray:
            return model.predict(
                pairs, batch_size=self.config.batch_size, show_progress_bar=False, convert_to_numpy=True
            )

        return np.asarray(self.model.run(predict), dtype=np.float32).reshape(-1)

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        """Sắp xếp lại ``candidates`` (id, văn bản) theo điểm cross-encoder giảm dần (hoà điểm giữ thứ tự cũ)."""
        scores = self.score(query, [text for _, text in candidates])
        order = np.argsort(-scores, kind="stable")
        return [(candidates[i][0], float(scores[i])) for i in order]

    def warmup(self) -> float | None:
        """Nạp model (nếu chưa) và chạy thử một lượt; trả về thời gian nạp nếu vừa nạp."""
        was_loaded = self.model.is_loaded
        self.score("khởi động", ["khởi động"])
        return None if was_loaded else self.model.last_load_seconds
