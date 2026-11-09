"""Reciprocal Rank Fusion (RRF) — gộp nhiều danh sách xếp hạng chỉ dựa trên thứ hạng.

Cormack, Clarke & Büttcher (2009): ``score(d) = Σ_i w_i / (k + rank_i(d))``, rank bắt đầu từ 1.
Không cần chuẩn hoá điểm giữa các retriever (điểm BM25 và cosine có thang đo khác nhau),
nên RRF là lựa chọn mặc định đơn giản và bền vững cho hybrid search.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[K]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[K, float]]:
    """Gộp các danh sách xếp hạng (phần tử tốt nhất đứng đầu) thành một danh sách (id, điểm RRF) giảm dần.

    - Phần tử vắng mặt trong một danh sách không nhận điểm từ danh sách đó.
    - Phần tử lặp lại trong cùng một danh sách chỉ được tính ở lần xuất hiện đầu (hạng tốt nhất).
    - Hoà điểm: ưu tiên phần tử có hạng tốt nhất ở bất kỳ danh sách nào, sau đó theo thứ tự xuất hiện đầu tiên
      (kết quả luôn xác định).
    - ``k`` lớn làm phẳng khác biệt giữa các hạng đầu (ưu tiên đồng thuận giữa các retriever);
      ``k`` nhỏ ưu tiên phần tử đứng rất cao ở một danh sách.
    """
    if k < 0:
        raise ValueError("k phải >= 0")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights phải có cùng số phần tử với rankings")

    scores: dict[K, float] = {}
    best_rank: dict[K, int] = {}
    first_seen: dict[K, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[K] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            first_seen.setdefault(item, len(first_seen))

    fused = sorted(scores.items(), key=lambda kv: (-round(kv[1], 12), best_rank[kv[0]], first_seen[kv[0]]))
    return fused if top_n is None else fused[:top_n]
