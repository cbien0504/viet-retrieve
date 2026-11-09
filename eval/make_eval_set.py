"""Tạo bộ đánh giá (câu hỏi -> id chunk đúng) từ ``eval/questions_seed.yaml``.

File seed lưu câu hỏi kèm một đoạn trích nguyên văn (evidence) từ văn bản nguồn. Script chia đoạn dữ liệu
theo ``config.yaml`` hiện tại rồi gán mỗi câu hỏi với (các) chunk của đúng văn bản đó có chứa evidence.
Nhờ vậy nhãn luôn khớp với cách chia đoạn: đổi ``chunk_size``/``chunk_overlap`` chỉ cần chạy lại script.

Chạy:  python -m eval.make_eval_set
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.chunking import Chunk, chunk_documents, load_documents
from src.config import Config, load_config
from src.pipeline import data_fingerprint

REQUIRED_FIELDS = ("id", "question", "doc_id", "evidence", "type")
QUESTION_TYPES = ("keyword", "paraphrase")


def _norm(text: str) -> str:
    """So khớp không phân biệt hoa thường, khoảng trắng và dạng Unicode."""
    return " ".join(unicodedata.normalize("NFC", text).split()).lower()


def load_seed(path: Path) -> list[dict[str, Any]]:
    """Đọc và kiểm tra file seed."""
    items = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} phải là một danh sách câu hỏi")
    seen: set[str] = set()
    for item in items:
        missing = [f for f in REQUIRED_FIELDS if not item.get(f)]
        if missing:
            raise ValueError(f"Câu hỏi {item.get('id', '?')} thiếu trường {missing}")
        if item["id"] in seen:
            raise ValueError(f"Trùng id câu hỏi: {item['id']}")
        if item["type"] not in QUESTION_TYPES:
            raise ValueError(f"{item['id']}: type phải thuộc {QUESTION_TYPES}")
        seen.add(item["id"])
    return items


def resolve_eval_set(
    seed: list[dict[str, Any]], chunks: list[Chunk]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Gán câu hỏi với các chunk chứa evidence. Trả về (bản ghi, lỗi, cảnh báo)."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for item in seed:
        evidence = _norm(item["evidence"])
        doc_chunks = by_doc.get(item["doc_id"])
        if not doc_chunks:
            errors.append(f"{item['id']}: không có văn bản '{item['doc_id']}'")
            continue
        relevant = [c.chunk_id for c in doc_chunks if evidence in _norm(c.text)]
        if not relevant:
            errors.append(f"{item['id']}: không tìm thấy evidence trong {item['doc_id']}: {item['evidence']!r}")
            continue
        elsewhere = sorted({c.doc_id for c in chunks if c.doc_id != item["doc_id"] and evidence in _norm(c.text)})
        if elsewhere:
            warnings.append(f"{item['id']}: evidence cũng xuất hiện trong {elsewhere} -> câu hỏi có thể mơ hồ")
        records.append(
            {
                "qid": item["id"],
                "question": item["question"],
                "type": item["type"],
                "doc_id": item["doc_id"],
                "evidence": item["evidence"],
                "relevant_chunk_ids": relevant,
            }
        )
    return records, errors, warnings


def eval_set_signature(config: Config) -> dict[str, Any]:
    """Dữ liệu + cách chia đoạn mà bộ đánh giá phụ thuộc (để evaluate.py kiểm tra khớp với index)."""
    return {
        "data_fingerprint": data_fingerprint(config.resolve(config.paths.raw_dir)),
        "chunking": config.to_dict()["chunking"],
    }


def meta_path_for(eval_set_path: Path) -> Path:
    return eval_set_path.with_suffix(".meta.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.make_eval_set", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = load_seed(config.resolve(config.eval.seed_path))
    chunks = chunk_documents(load_documents(config.resolve(config.paths.raw_dir)), config.chunking)
    records, errors, warnings = resolve_eval_set(seed, chunks)
    for message in warnings:
        print(f"CẢNH BÁO {message}")
    if errors:
        for message in errors:
            print(f"LỖI {message}", file=sys.stderr)
        return 1

    out_path = config.resolve(config.eval.eval_set_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {
        **eval_set_signature(config),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "num_questions": len(records),
        "num_chunks_in_corpus": len(chunks),
    }
    meta_path_for(out_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    types = Counter(r["type"] for r in records)
    multi = sum(len(r["relevant_chunk_ids"]) > 1 for r in records)
    docs = len({r["doc_id"] for r in records})
    print(f"Đã ghi {len(records)} câu hỏi vào {out_path}")
    print(f"  theo loại: {dict(types)} | phủ {docs} văn bản | {multi} câu có >1 chunk đúng (do overlap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
