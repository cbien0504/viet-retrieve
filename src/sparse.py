"""Truy hồi thưa (sparse): tách từ tiếng Việt + BM25 (``rank_bm25.BM25Okapi``)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import SparseConfig

# Hư từ / từ để hỏi phổ biến. Từ ghép viết bằng dấu gạch dưới vì underthesea nối âm tiết bằng "_".
# Danh sách cố ý ngắn: BM25 đã tự giảm trọng số từ phổ biến qua IDF; ở đây chỉ loại các từ
# hầu như không mang nội dung, đặc biệt là từ để hỏi xuất hiện trong câu hỏi nhưng không có trong văn bản.
VIETNAMESE_STOPWORDS: frozenset[str] = frozenset(
    """
    à ạ ai anh bị bởi các cái cho chị chứ có_phải còn của cũng do đã đang đâu để đến đó em gì hay hoặc
    khi không là lại mà mình một nào này nên nếu nhé như nhỉ những nữa ra rằng rồi sao sẽ tại thì
    theo thế tôi trong từ tới và vào vậy về vì với được
    bao_nhiêu mấy thế_nào như_thế_nào khi_nào bao_giờ bao_lâu ở_đâu tại_sao vì_sao phải_không hay_không
    """.split()
)

_WORD_RE = re.compile(r"\w+(?:[.,:/\-]\w+)*", re.UNICODE)


class VietnameseTokenizer:
    """Tách từ tiếng Việt cho BM25.

    Backend:
      - ``underthesea``: chuẩn hoá kiểu bỏ dấu (``khoẻ`` -> ``khỏe``) rồi tách từ ghép bằng CRF
        (``nghỉ phép`` -> ``nghỉ_phép``).
      - ``syllable``: tách theo âm tiết bằng regex, không cần thư viện ngoài (baseline / fallback).
    Sau khi tách: chuyển chữ thường, bỏ dấu câu, bỏ stopword (tuỳ chọn).
    """

    def __init__(
        self,
        backend: str = "underthesea",
        lowercase: bool = True,
        remove_stopwords: bool = True,
        stopwords: frozenset[str] | None = None,
    ) -> None:
        if backend not in ("underthesea", "syllable"):
            raise ValueError(f"backend không hỗ trợ: {backend}")
        self.backend = backend
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stopwords = VIETNAMESE_STOPWORDS if stopwords is None else stopwords
        self._segment: Callable[[str], list[str]] = (
            self._underthesea_segmenter() if backend == "underthesea" else self._syllables
        )

    @classmethod
    def from_config(cls, config: SparseConfig) -> VietnameseTokenizer:
        return cls(config.tokenizer, config.lowercase, config.remove_stopwords)

    @property
    def signature(self) -> dict[str, object]:
        """Mô tả cấu hình tokenizer, dùng để kiểm tra cache token còn hợp lệ."""
        return {
            "backend": self.backend,
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords,
            "n_stopwords": len(self.stopwords),
        }

    @staticmethod
    def _underthesea_segmenter() -> Callable[[str], list[str]]:
        try:
            from underthesea import text_normalize, word_tokenize
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise ImportError(
                "Chưa cài underthesea. Cài bằng `pip install underthesea` hoặc đặt sparse.tokenizer: syllable"
            ) from exc

        def segment(text: str) -> list[str]:
            return [word.replace(" ", "_") for word in word_tokenize(text_normalize(text))]

        return segment

    @staticmethod
    def _syllables(text: str) -> list[str]:
        return _WORD_RE.findall(text)

    def tokenize(self, text: str) -> list[str]:
        """Trả về danh sách token (đã chuẩn hoá) của ``text``."""
        text = unicodedata.normalize("NFC", text)
        tokens: list[str] = []
        for token in self._segment(text):
            token = token.strip("_")
            if not token or not any(ch.isalnum() for ch in token):
                continue  # bỏ dấu câu
            if self.lowercase:
                token = token.lower()
            if self.remove_stopwords and token in self.stopwords:
                continue
            tokens.append(token)
        return tokens

    def tokenize_many(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(text) for text in texts]


class BM25Retriever:
    """BM25Okapi trên các chunk đã tách từ; ``search`` trả về (chunk_id, điểm) giảm dần.

    Lưu ý: IDF của ``BM25Okapi`` là ``log((N - n + 0.5) / (n + 0.5))`` nên bằng 0 khi một từ có mặt ở đúng
    một nửa số chunk (ví dụ kho chỉ có 2 chunk); từ có mặt ở hơn một nửa số chunk được gán IDF nhỏ
    ``epsilon * IDF trung bình``. Với kho thực tế (hàng trăm chunk) điều này không đáng kể.
    """

    def __init__(
        self,
        chunk_ids: Sequence[str],
        tokenized_corpus: Sequence[list[str]],
        tokenizer: VietnameseTokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(chunk_ids) != len(tokenized_corpus):
            raise ValueError("chunk_ids và tokenized_corpus phải cùng độ dài")
        if not chunk_ids:
            raise ValueError("Không thể dựng BM25 trên tập rỗng")
        self.chunk_ids = list(chunk_ids)
        self.tokenizer = tokenizer
        self._bm25 = BM25Okapi([list(tokens) for tokens in tokenized_corpus], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        """Điểm BM25 của ``query`` với mọi chunk (theo thứ tự ``chunk_ids``)."""
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunk_ids), dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Top-k chunk có điểm > 0 (chunk không chung token nào với câu hỏi bị loại)."""
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]
