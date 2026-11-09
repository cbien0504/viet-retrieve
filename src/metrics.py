"""Metric truy hồi với nhãn liên quan nhị phân.

Mỗi câu hỏi có một tập ``relevant`` (thường 1 chunk; có thể nhiều hơn nếu đoạn bằng chứng nằm trong
vùng gối đầu của 2 chunk — khi đó các chunk này tương đương nhau).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def first_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    """Hạng (bắt đầu từ 1) của chunk đúng đầu tiên, ``None`` nếu không có trong danh sách."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return rank
    return None


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1 nếu có ít nhất một chunk đúng trong top-k, ngược lại 0 (hit rate / success@k).

    Vì các chunk đúng của một câu hỏi chứa cùng một đoạn bằng chứng, tìm được một chunk là đủ trả lời.
    Trung bình trên tập câu hỏi = tỷ lệ câu hỏi được "phủ" trong top-k.
    """
    rank = first_relevant_rank(retrieved[:k], relevant)
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """1 / hạng của chunk đúng đầu tiên trong top-k (0 nếu không có). Trung bình = MRR@k."""
    candidates = retrieved if k is None else retrieved[:k]
    rank = first_relevant_rank(candidates, relevant)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """nDCG@k với độ liên quan nhị phân."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(retrieved[:k], start=1) if cid in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mcnemar_exact_p(b: int, c: int) -> float:
    """p-value hai phía của kiểm định McNemar chính xác (nhị thức) cho hai cấu hình chạy trên cùng câu hỏi.

    ``b`` = số câu cấu hình A đúng còn B sai, ``c`` = ngược lại. Các câu cả hai cùng đúng/sai không ảnh hưởng.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def percentile(values: Sequence[float], q: float) -> float:
    """Phân vị ``q`` (0-100) bằng nội suy tuyến tính, không cần numpy."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(pos), math.ceil(pos)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)
