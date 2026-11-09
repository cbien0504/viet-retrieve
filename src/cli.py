"""Dòng lệnh: tải model, dựng index, truy vấn, xem thông tin index.

Ví dụ:
    python -m src.cli download-models
    python -m src.cli build
    python -m src.cli search "Tôi được nghỉ phép bao nhiêu ngày?" --mode hybrid_rerank --top-k 5
    python -m src.cli info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from src.config import MODE_LABELS, MODES, load_config
from src.pipeline import MANIFEST_FILE, RetrievalPipeline, build_index

# Chỉ tải những file cần để chạy bằng PyTorch (bỏ bản ONNX/OpenVINO và trọng số trùng lặp).
MODEL_FILE_PATTERNS = ["*.json", "*.txt", "sentencepiece.bpe.model", "tokenizer.json", "1_Pooling/*", "README.md"]


def _cmd_download_models(args: argparse.Namespace) -> int:
    from huggingface_hub import list_repo_files, snapshot_download

    config = load_config(args.config)
    models = [config.dense.model_name, config.reranker.model_name]
    if args.with_extra:
        models += config.eval.extra_rerankers
    for name in dict.fromkeys(models):
        weights = "model.safetensors" if "model.safetensors" in list_repo_files(name) else "pytorch_model.bin"
        path = snapshot_download(
            name, allow_patterns=[*MODEL_FILE_PATTERNS, weights], ignore_patterns=["onnx/*", "openvino/*"]
        )
        print(f"{name} -> {path}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = build_index(config, with_dense=not args.no_dense)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline = RetrievalPipeline.load(config)
    mode = args.mode or config.retrieval.default_mode
    # Model được nạp lười cho đúng chế độ này và giải phóng khi xong (nếu cấu hình yêu cầu).
    with pipeline.hold_models():
        loads = pipeline.warmup(mode)
        response = pipeline.search(args.query, mode=mode, top_k=args.top_k)
    print(f"Câu hỏi: {response.query}")
    print(f"Chế độ: {MODE_LABELS[response.mode]}")
    if loads:
        print("Nạp model: " + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in loads.items()))
    for hit in response.hits:
        parts = [f"score={hit.score:.4f}"]
        if hit.bm25_rank is not None and response.mode != "bm25":
            parts.append(f"bm25#{hit.bm25_rank}")
        if hit.dense_rank is not None and response.mode != "dense":
            parts.append(f"dense#{hit.dense_rank}")
        print(f"\n#{hit.rank}  {'  '.join(parts)}  [{hit.chunk.chunk_id}]  {hit.chunk.heading}")
        for line in hit.chunk.text.split("\n"):
            print(textwrap.indent(textwrap.fill(line, width=100), "    "))
    if not response.hits:
        print("\n(Không tìm thấy đoạn nào phù hợp)")
    timings = ", ".join(f"{k}={v:.1f}" for k, v in response.timings_ms.items())
    print(f"\nThời gian (ms): {timings}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest_path = config.resolve(config.paths.index_dir) / MANIFEST_FILE
    if not manifest_path.is_file():
        print("Chưa có index. Chạy: python -m src.cli build")
        return 1
    print(manifest_path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="Hybrid retrieval tiếng Việt")
    parser.add_argument("--config", default=None, help="đường dẫn config.yaml (mặc định: ./config.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="in log chi tiết")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download-models", help="tải trước model embedding + reranker (chỉ file cần thiết)")
    download.add_argument("--with-extra", action="store_true", help="tải cả reranker so sánh (eval.extra_rerankers)")
    download.set_defaults(func=_cmd_download_models)

    build = sub.add_parser("build", help="đọc data/raw, chia đoạn, dựng BM25 + FAISS")
    build.add_argument("--no-dense", action="store_true", help="chỉ dựng BM25 (không cần tải model embedding)")
    build.set_defaults(func=_cmd_build)

    search = sub.add_parser("search", help="truy vấn")
    search.add_argument("query")
    search.add_argument("--mode", choices=MODES, default=None, help="mặc định: retrieval.default_mode")
    search.add_argument("--top-k", type=int, default=None, help="mặc định: retrieval.top_k")
    search.set_defaults(func=_cmd_search)

    sub.add_parser("info", help="thông tin index đã dựng").set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
