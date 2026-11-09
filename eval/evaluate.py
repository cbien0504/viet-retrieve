"""Đánh giá các cấu hình truy hồi trên bộ câu hỏi tổng hợp.

Đo Recall@k (k trong ``eval.ks``), MRR@max(k), nDCG@``eval.ndcg_k`` và độ trễ mỗi truy vấn cho các chế độ
``bm25``, ``dense``, ``hybrid`` (RRF) và ``hybrid_rerank`` (RRF + cross-encoder).

- Truy vấn được xử lý theo batch nhỏ (``eval.batch_size``); kết quả từng truy vấn được ghi dần ra
  ``eval/runs/<mode>.jsonl`` nên không giữ toàn bộ kết quả trung gian của mô hình trong RAM.
- Model chỉ được nạp khi chế độ cần đến. Reranker được nạp ở đầu mỗi batch và giải phóng ở cuối batch
  (``reranker.release_after_use``). Thời gian nạp model được ghi riêng, KHÔNG tính vào độ trễ truy vấn.

Chạy:
    python -m eval.evaluate
    python -m eval.evaluate --modes bm25 --set sparse.tokenizer=syllable --output eval/ablations/bm25_syllable.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.make_eval_set import eval_set_signature, meta_path_for
from src.config import MODE_LABELS, MODES, Config, apply_overrides, load_config
from src.lazy import current_rss_mb
from src.metrics import mcnemar_exact_p, ndcg_at_k, percentile, recall_at_k, reciprocal_rank
from src.pipeline import RetrievalPipeline
from src.reranker import CrossEncoderReranker

logger = logging.getLogger("eval")


@dataclass
class ModeReport:
    """Kết quả đánh giá một chế độ."""

    mode: str
    label: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    load_seconds: dict[str, list[float]] = field(default_factory=dict)
    rss_start_mb: float = float("nan")
    rss_peak_mb: float = float("nan")
    rss_end_mb: float = float("nan")
    wall_seconds: float = 0.0

    def mean(self, key: str, qtype: str | None = None) -> float:
        values = [r[key] for r in self.rows if qtype is None or r["type"] == qtype]
        return statistics.fmean(values) if values else float("nan")

    @property
    def latencies(self) -> list[float]:
        return [r["latency_ms"] for r in self.rows]


def load_eval_set(config: Config) -> list[dict[str, Any]]:
    """Đọc bộ đánh giá và kiểm tra nó được tạo với đúng dữ liệu + cách chia đoạn hiện tại."""
    path = config.resolve(config.eval.eval_set_path)
    meta_path = meta_path_for(path)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Chưa có bộ đánh giá. Chạy: python -m eval.make_eval_set")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    current = eval_set_signature(config)
    if any(meta.get(k) != v for k, v in current.items()):
        raise RuntimeError(
            "Bộ đánh giá được tạo với dữ liệu hoặc cấu hình chia đoạn khác. Chạy lại: python -m eval.make_eval_set"
        )
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _batched(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def mode_label(mode: str, config: Config, reranker_model: str | None = None) -> str:
    """Nhãn hiển thị; với chế độ có reranker thì kèm tên model."""
    if mode != "hybrid_rerank":
        return MODE_LABELS[mode]
    return f"{MODE_LABELS[mode]} ({(reranker_model or config.reranker.model_name).split('/')[-1]})"


def evaluate_mode(
    pipeline: RetrievalPipeline,
    items: list[dict[str, Any]],
    mode: str,
    config: Config,
    run_path: Path,
    label: str | None = None,
) -> ModeReport:
    """Chạy toàn bộ câu hỏi qua một chế độ, theo từng batch nhỏ."""
    ks = sorted(config.eval.ks)
    k_max = max(ks)
    ndcg_k = config.eval.ndcg_k
    depth = max(k_max, ndcg_k)
    report = ModeReport(mode=mode, label=label or mode_label(mode, config), rss_start_mb=current_rss_mb())
    report.rss_peak_mb = report.rss_start_mb
    started = time.perf_counter()
    run_path.parent.mkdir(parents=True, exist_ok=True)
    batches = _batched(items, config.eval.batch_size)
    with run_path.open("w", encoding="utf-8") as out:
        for batch_no, batch in enumerate(batches, start=1):
            with pipeline.hold_models():
                for component, seconds in pipeline.warmup(mode).items():
                    report.load_seconds.setdefault(component, []).append(seconds)
                report.rss_peak_mb = max(report.rss_peak_mb, current_rss_mb())
                for item in batch:
                    t0 = time.perf_counter()
                    response = pipeline.search(item["question"], mode=mode, top_k=depth)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    retrieved = [hit.chunk.chunk_id for hit in response.hits]
                    relevant = set(item["relevant_chunk_ids"])
                    rank = next((i for i, cid in enumerate(retrieved, start=1) if cid in relevant), None)
                    row: dict[str, Any] = {
                        "qid": item["qid"],
                        "type": item["type"],
                        "question": item["question"],
                        "relevant": sorted(relevant),
                        "retrieved": retrieved,
                        "first_rank": rank,
                        "latency_ms": round(latency_ms, 3),
                        "timings_ms": {k: round(v, 3) for k, v in response.timings_ms.items()},
                    }
                    for k in ks:
                        row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
                    row[f"mrr@{k_max}"] = reciprocal_rank(retrieved, relevant, k_max)
                    row[f"ndcg@{ndcg_k}"] = ndcg_at_k(retrieved, relevant, ndcg_k)
                    report.rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                report.rss_peak_mb = max(report.rss_peak_mb, current_rss_mb())
            out.flush()
            logger.info(
                "[%s] batch %d/%d xong (%d câu), RSS %.0f MiB",
                report.label,
                batch_no,
                len(batches),
                len(report.rows),
                current_rss_mb(),
            )
    report.wall_seconds = time.perf_counter() - started
    report.rss_end_mb = current_rss_mb()
    return report


def environment_info() -> dict[str, str]:
    """Thông tin máy chạy đánh giá (để số liệu độ trễ có ngữ cảnh)."""
    cpu = platform.processor() or "?"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = {
        "cpu": f"{cpu} ({os.cpu_count()} luồng logic)",
        "python": platform.python_version(),
        "hf_offline": os.environ.get("HF_HUB_OFFLINE", "0"),
    }
    try:
        import torch

        info["torch"] = f"{torch.__version__}, {torch.get_num_threads()} luồng, GPU: {torch.cuda.is_available()}"
    except ImportError:  # pragma: no cover
        info["torch"] = "không cài"
    return info


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if value != value else f"{value:.{digits}f}"


def render_markdown(
    reports: list[ModeReport],
    config: Config,
    items: list[dict[str, Any]],
    pipeline: RetrievalPipeline,
    overrides: list[str],
) -> str:
    """Bảng kết quả + phân tích theo loại câu hỏi + chi tiết từng câu."""
    ks = sorted(config.eval.ks)
    k_max, ndcg_k = max(ks), config.eval.ndcg_k
    metric_keys = [f"recall@{k}" for k in ks] + [f"mrr@{k_max}", f"ndcg@{ndcg_k}"]
    headers = [f"Recall@{k}" for k in ks] + [f"MRR@{k_max}", f"nDCG@{ndcg_k}"]
    types = sorted({i["type"] for i in items})
    env = environment_info()
    n_docs = len({c.doc_id for c in pipeline.chunks})
    lines: list[str] = [
        "# Kết quả đánh giá",
        "",
        f"> Sinh tự động bởi `python -m eval.evaluate` lúc {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        " — số liệu từ lần chạy thực tế, không chỉnh tay.",
        "",
        "## Thiết lập",
        "",
        f"- Dữ liệu: {n_docs} văn bản tổng hợp, {len(pipeline.chunks)} chunk "
        f"(chunk_size={config.chunking.chunk_size}, overlap={config.chunking.chunk_overlap}, "
        f"respect_sections={config.chunking.respect_sections}).",
        f"- Bộ câu hỏi: {len(items)} câu tổng hợp ("
        + ", ".join(f"{sum(i['type'] == t for i in items)} {t}" for t in types)
        + "), mỗi câu có đúng một văn bản nguồn.",
        f"- BM25: tokenizer `{config.sparse.tokenizer}`, stopword={config.sparse.remove_stopwords}, "
        f"k1={config.sparse.k1}, b={config.sparse.b}.",
        f"- Dense: `{config.dense.model_name}` + FAISS IndexFlatIP (vector đã chuẩn hoá L2).",
        f"- Hybrid: RRF k={config.fusion.rrf_k}, mỗi retriever lấy {config.fusion.candidates_per_retriever} ứng viên.",
        f"- Reranker: `{config.reranker.model_name}` (mặc định)"
        + "".join(f", `{m}`" for m in config.eval.extra_rerankers)
        + f" — xếp hạng lại top-{config.reranker.candidates} ứng viên sau RRF.",
        f"- Máy: {env['cpu']}; Python {env['python']}; torch {env['torch']}; HF_HUB_OFFLINE={env['hf_offline']}.",
        f"- Batch đánh giá: {config.eval.batch_size} câu/batch.",
    ]
    if overrides:
        lines.append(f"- Ghi đè cấu hình: `{' '.join(overrides)}`.")
    lines += [
        "",
        "## Kết quả chính",
        "",
        "| Cấu hình | " + " | ".join(headers) + " | Độ trễ TB (ms) | p50 (ms) | p95 (ms) |",
        "|---|" + "---:|" * (len(headers) + 3),
    ]
    for r in reports:
        cells = [_fmt(r.mean(k)) for k in metric_keys]
        lat = r.latencies
        cells += [_fmt(statistics.fmean(lat), 1), _fmt(percentile(lat, 50), 1), _fmt(percentile(lat, 95), 1)]
        lines.append(f"| {r.label} | " + " | ".join(cells) + " |")

    lines += ["", "## Theo loại câu hỏi", ""]
    sub_headers = [f"{t} {h}" for t in types for h in ("R@1", f"R@{ks[1] if len(ks) > 1 else ks[0]}", "MRR")]
    lines.append("| Cấu hình | " + " | ".join(sub_headers) + " |")
    lines.append("|---|" + "---:|" * len(sub_headers))
    k_mid = ks[1] if len(ks) > 1 else ks[0]
    for r in reports:
        cells = []
        for t in types:
            cells += [
                _fmt(r.mean(f"recall@{ks[0]}", t)),
                _fmt(r.mean(f"recall@{k_mid}", t)),
                _fmt(r.mean(f"mrr@{k_max}", t)),
            ]
        lines.append(f"| {r.label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Nạp model và bộ nhớ",
        "",
        "Thời gian nạp model đo riêng, không tính vào độ trễ truy vấn ở trên. RSS = bộ nhớ thường trú của tiến trình,"
        " đo tại các mốc: đầu chế độ, sau khi nạp model, cuối mỗi batch và sau khi giải phóng (cột RSS cuối).",
        "",
        "| Cấu hình | Model được nạp (số lần × TB giây) | RSS đầu (MiB) | RSS đỉnh (MiB) | RSS cuối (MiB) "
        "| Thời gian chạy (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in reports:
        loads = ", ".join(f"{name}: {len(v)} × {statistics.fmean(v):.2f}s" for name, v in r.load_seconds.items()) or "—"
        lines.append(
            f"| {r.label} | {loads} | {_fmt(r.rss_start_mb, 0)} | {_fmt(r.rss_peak_mb, 0)} | "
            f"{_fmt(r.rss_end_mb, 0)} | {_fmt(r.wall_seconds, 1)} |"
        )

    reference = next((r for r in reports if r.mode == "hybrid_rerank"), None)
    if reference is not None and len(reports) > 1:
        ref_hits = {row["qid"]: row["first_rank"] == 1 for row in reference.rows}
        lines += [
            "",
            f"## Kiểm định cặp: {reference.label} so với các cấu hình khác (Recall@1)",
            "",
            "Kiểm định McNemar chính xác trên các câu mà hai cấu hình cho kết quả khác nhau ở hạng 1."
            " p < 0.05 nghĩa là chênh lệch khó do ngẫu nhiên; với 50 câu, chênh lệch vài câu thường chưa đủ.",
            "",
            "| So với | Mặc định đúng, cấu hình kia sai | Mặc định sai, cấu hình kia đúng | p |",
            "|---|---:|---:|---:|",
        ]
        for r in reports:
            if r is reference:
                continue
            other = {row["qid"]: row["first_rank"] == 1 for row in r.rows}
            b = sum(ref_hits[q] and not other[q] for q in ref_hits)
            c = sum(other[q] and not ref_hits[q] for q in ref_hits)
            lines.append(f"| {r.label} | {b} | {c} | {mcnemar_exact_p(b, c):.3f} |")

    lines += [
        "",
        f"## Hạng của chunk đúng theo từng câu hỏi (— = không có trong top-{max(k_max, ndcg_k)})",
        "",
        "<details><summary>Mở bảng chi tiết</summary>",
        "",
        "| Câu | Loại | Câu hỏi | " + " | ".join(r.label for r in reports) + " |",
        "|---|---|---|" + "---:|" * len(reports),
    ]
    by_report = [{row["qid"]: row for row in r.rows} for r in reports]
    for item in items:
        ranks = [rows[item["qid"]]["first_rank"] for rows in by_report]
        lines.append(
            f"| {item['qid']} | {item['type']} | {item['question']} | "
            + " | ".join("—" if rank is None else str(rank) for rank in ranks)
            + " |"
        )
    lines += [
        "",
        "</details>",
        "",
        "## Định nghĩa",
        "",
        "- **Recall@k**: tỷ lệ câu hỏi có ít nhất một chunk đúng trong top-k (hit rate). Mỗi câu hỏi ở đây có"
        " đúng một chunk đúng nên đây cũng là recall thông thường.",
        f"- **MRR@{k_max}**: trung bình 1/hạng của chunk đúng đầu tiên trong top-{k_max} (0 nếu không có).",
        f"- **nDCG@{ndcg_k}**: nDCG với độ liên quan nhị phân.",
        "- **Độ trễ**: thời gian `pipeline.search()` cho một câu hỏi (đã nạp sẵn model), gồm tách từ, mã hoá câu hỏi,"
        " tìm kiếm, gộp và xếp hạng lại.",
        "- Chi tiết từng truy vấn: `eval/runs/<chế độ>.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def _log_summary(report: ModeReport, k_max: int) -> None:
    logger.info(
        "[%s] Recall@1=%.3f Recall@5=%.3f MRR@%d=%.3f | độ trễ TB %.1f ms",
        report.label,
        report.mean("recall@1"),
        report.mean("recall@5"),
        k_max,
        report.mean(f"mrr@{k_max}"),
        statistics.fmean(report.latencies),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=None, help="mặc định: eval.modes trong config")
    parser.add_argument("--output", default=None, help="file markdown kết quả (mặc định: eval.results_path)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="ghi đè config: mục.khoá=giá_trị")
    parser.add_argument("--limit", type=int, default=None, help="chỉ đánh giá N câu đầu (chạy thử nhanh)")
    parser.add_argument(
        "--extra-rerankers",
        nargs="*",
        default=None,
        help="reranker so sánh thêm (mặc định: eval.extra_rerankers)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = apply_overrides(load_config(args.config), args.overrides)
    modes = args.modes or config.eval.modes
    items = load_eval_set(config)[: args.limit]
    pipeline = RetrievalPipeline.load(config)
    missing = {cid for i in items for cid in i["relevant_chunk_ids"]} - set(pipeline.chunk_by_id)
    if missing:
        raise RuntimeError(f"Chunk đúng không có trong index: {sorted(missing)[:5]}... Chạy lại build + make_eval_set")

    output = config.resolve(args.output or config.eval.results_path)
    runs_dir = config.resolve(config.eval.runs_dir)
    if args.output:  # ablation: tách thư mục runs theo tên file kết quả để không ghi đè lần chạy chính
        runs_dir = runs_dir / Path(args.output).stem
    reports: list[ModeReport] = []
    for mode in modes:
        logger.info("=== Đánh giá chế độ: %s (%d câu hỏi) ===", MODE_LABELS[mode], len(items))
        report = evaluate_mode(pipeline, items, mode, config, runs_dir / f"{mode}.jsonl")
        reports.append(report)
        k_max = max(config.eval.ks)
        _log_summary(report, k_max)
    extra = config.eval.extra_rerankers if args.extra_rerankers is None else args.extra_rerankers
    if extra and "hybrid_rerank" in modes:
        default_reranker = pipeline.reranker
        for model_name in extra:
            if pipeline.reranker is not None:
                pipeline.reranker.model.release()  # chỉ đổi reranker; embedder giữ nguyên
            pipeline.reranker = CrossEncoderReranker(replace(config.reranker, model_name=model_name))
            label = mode_label("hybrid_rerank", config, model_name)
            logger.info("=== Đánh giá chế độ: %s (%d câu hỏi) ===", label, len(items))
            slug = model_name.replace("/", "__")
            report = evaluate_mode(
                pipeline, items, "hybrid_rerank", config, runs_dir / f"hybrid_rerank__{slug}.jsonl", label
            )
            reports.append(report)
            _log_summary(report, max(config.eval.ks))
        pipeline.reranker = default_reranker
    pipeline.release_models()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(reports, config, items, pipeline, args.overrides), encoding="utf-8")
    print(f"Đã ghi kết quả vào {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
