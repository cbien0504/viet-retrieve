# Độ nhạy của Hybrid RRF (chỉ chế độ `hybrid`, không reranker)

> Sinh bởi `python -m eval.sensitivity` lúc 2026-11-09 10:30 trên cùng 50 câu hỏi của bộ đánh giá chính. Chỉ để phân tích — KHÔNG dùng để chọn tham số mặc định.

| Thay đổi | RRF k | Ứng viên/retriever | Trọng số BM25 | Recall@1 | Recall@5 | MRR@10 | MRR keyword | MRR paraphrase | Độ trễ TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mặc định | 60 | 50 | 1.0 | 0.720 | 0.920 | 0.806 | 0.981 | 0.616 | 8.6 |
| rrf_k=10 | 10 | 50 | 1.0 | 0.720 | 0.940 | 0.810 | 0.981 | 0.625 | 7.7 |
| rrf_k=30 | 30 | 50 | 1.0 | 0.720 | 0.920 | 0.808 | 0.981 | 0.620 | 9.1 |
| rrf_k=100 | 100 | 50 | 1.0 | 0.720 | 0.920 | 0.806 | 0.981 | 0.616 | 7.6 |
| candidates_per_retriever=10 | 60 | 10 | 1.0 | 0.720 | 0.940 | 0.802 | 0.981 | 0.607 | 7.8 |
| candidates_per_retriever=20 | 60 | 20 | 1.0 | 0.720 | 0.920 | 0.803 | 0.981 | 0.610 | 8.5 |
| bm25_weight=0.5 | 60 | 50 | 0.5 | 0.760 | 0.900 | 0.828 | 0.981 | 0.663 | 9.3 |
| bm25_weight=0.25 | 60 | 50 | 0.25 | 0.760 | 0.920 | 0.840 | 0.962 | 0.707 | 8.5 |
