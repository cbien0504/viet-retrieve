"""Phân tích độ nhạy của Hybrid RRF theo hằng số k, số ứng viên mỗi retriever và trọng số BM25.

Chạy trên CHÍNH bộ đánh giá nên chỉ dùng để hiểu hành vi của RRF, không dùng để chọn tham số mặc định
(muốn tinh chỉnh cần một tập dev riêng, nếu không sẽ overfit vào 50 câu hỏi này).

Chạy:  python -m eval.sensitivity            -> eval/ablations/rrf_sensitivity.md
"""

from __future__ import annotations

import argparse
import copy
import logging
import statistics
import sys
from datetime import datetime

from eval.evaluate import evaluate_mode, load_eval_set
from src.config import load_config
from src.pipeline import RetrievalPipeline

DEFAULT = {"rrf_k": 60, "candidates_per_retriever": 50, "bm25_weight": 1.0}
GRID: list[dict[str, float]] = [
    {},
    {"rrf_k": 10},
    {"rrf_k": 30},
    {"rrf_k": 100},
    {"candidates_per_retriever": 10},
    {"candidates_per_retriever": 20},
    {"bm25_weight": 0.5},
    {"bm25_weight": 0.25},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.sensitivity", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="eval/ablations/rrf_sensitivity.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    base_config = load_config(args.config)
    items = load_eval_set(base_config)
    pipeline = RetrievalPipeline.load(base_config)
    k_max = max(base_config.eval.ks)
    rows: list[str] = []
    for change in GRID:
        params = {**DEFAULT, **change}
        config = copy.deepcopy(base_config)
        for key, value in params.items():
            setattr(config.fusion, key, value)
        config.validate()
        pipeline.config = config
        name = "default" if not change else "_".join(f"{k}-{v}" for k, v in change.items())
        report = evaluate_mode(
            pipeline,
            items,
            "hybrid",
            config,
            config.resolve(config.eval.runs_dir) / "sensitivity" / f"{name}.jsonl",
        )
        label = "mặc định" if not change else ", ".join(f"{k}={v}" for k, v in change.items())
        rows.append(
            f"| {label} | {params['rrf_k']} | {params['candidates_per_retriever']} | {params['bm25_weight']} | "
            f"{report.mean('recall@1'):.3f} | {report.mean('recall@5'):.3f} | {report.mean(f'mrr@{k_max}'):.3f} | "
            f"{report.mean(f'mrr@{k_max}', 'keyword'):.3f} | {report.mean(f'mrr@{k_max}', 'paraphrase'):.3f} | "
            f"{statistics.fmean(report.latencies):.1f} |"
        )
        print(rows[-1])
    pipeline.release_models()

    output = base_config.resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                "# Độ nhạy của Hybrid RRF (chỉ chế độ `hybrid`, không reranker)",
                "",
                f"> Sinh bởi `python -m eval.sensitivity` lúc {datetime.now().strftime('%Y-%m-%d %H:%M')} trên cùng"
                f" {len(items)} câu hỏi của bộ đánh giá chính. Chỉ để phân tích — KHÔNG dùng để chọn tham số mặc định.",
                "",
                "| Thay đổi | RRF k | Ứng viên/retriever | Trọng số BM25 | Recall@1 | Recall@5 | "
                f"MRR@{k_max} | MRR keyword | MRR paraphrase | Độ trễ TB (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Đã ghi {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
