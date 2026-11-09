"""Unit test cho src/chunking.py."""

from __future__ import annotations

import unicodedata
from itertools import pairwise
from pathlib import Path

import pytest

from src.chunking import (
    Document,
    chunk_document,
    load_chunks,
    load_documents,
    normalize_text,
    parse_document,
    save_chunks,
    split_sentences,
)
from src.config import ChunkingConfig, ConfigError


def make_doc(text: str, doc_id: str = "doc") -> Document:
    title, _ = parse_document(text)
    return Document(doc_id=doc_id, title=title or doc_id, text=normalize_text(text), source=f"{doc_id}.md")


def numbered_sentences(n: int, words_per_sentence: int = 10) -> list[str]:
    """Các câu có nội dung khác nhau, mỗi câu đúng ``words_per_sentence`` từ."""
    filler = " ".join(["nội", "dung"] * words_per_sentence)
    return [f"Câu {i} " + " ".join(filler.split()[: words_per_sentence - 2]) + "." for i in range(n)]


def test_normalize_text_converts_nfd_to_nfc() -> None:
    nfd = unicodedata.normalize("NFD", "Nghỉ phép năm")
    assert nfd != "Nghỉ phép năm"
    assert normalize_text(nfd) == "Nghỉ phép năm"


def test_split_sentences_keeps_numbers_and_abbreviations() -> None:
    text = "Phụ cấp là 1.200.000 đồng mỗi đêm tại TP. Hồ Chí Minh. Hà Nội cũng vậy! Còn tỉnh khác thì sao?"
    assert split_sentences(text) == [
        "Phụ cấp là 1.200.000 đồng mỗi đêm tại TP. Hồ Chí Minh.",
        "Hà Nội cũng vậy!",
        "Còn tỉnh khác thì sao?",
    ]


def test_split_sentences_does_not_split_before_lowercase() -> None:
    assert split_sentences("Xem mục 2. sau đó nộp đơn.") == ["Xem mục 2. sau đó nộp đơn."]


def test_parse_document_extracts_title_sections_and_bullets() -> None:
    text = (
        "# Quy định A\n\nMở đầu.\n\n## 1. Mục một\n\nĐoạn **in đậm**.\n- Ý thứ nhất\n- Ý thứ hai\n\n"
        "### 1.1 Con\n\nChi tiết."
    )
    title, sections = parse_document(text)
    assert title == "Quy định A"
    assert [s.heading for s in sections] == ["", "1. Mục một", "1. Mục một > 1.1 Con"]
    assert sections[1].paragraphs == ("Đoạn in đậm.", "Ý thứ nhất", "Ý thứ hai")


def test_chunks_respect_size_and_cover_all_sentences() -> None:
    sentences = numbered_sentences(30)  # 300 từ, một đoạn văn
    doc = make_doc("# T\n\n## Mục\n\n" + " ".join(sentences))
    config = ChunkingConfig(chunk_size=50, chunk_overlap=20, min_chunk_words=5)
    chunks = chunk_document(doc, config)
    assert len(chunks) > 1
    assert all(c.n_words <= config.chunk_size + config.min_chunk_words for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    sentences = numbered_sentences(20)
    doc = make_doc("## Mục\n\n" + " ".join(sentences))
    config = ChunkingConfig(chunk_size=50, chunk_overlap=20, min_chunk_words=0)
    chunks = chunk_document(doc, config)
    for prev, nxt in pairwise(chunks):
        prev_sents = split_sentences(prev.text)
        next_sents = split_sentences(nxt.text)
        shared = [s for s in next_sents if s in prev_sents]
        assert shared, "đoạn kế tiếp phải lặp lại câu cuối của đoạn trước"
        assert shared == prev_sents[-len(shared) :], "phần gối đầu phải là đuôi của đoạn trước"
        assert sum(len(s.split()) for s in shared) <= config.chunk_overlap


def test_zero_overlap_produces_disjoint_chunks() -> None:
    doc = make_doc("## Mục\n\n" + " ".join(numbered_sentences(20)))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=50, chunk_overlap=0, min_chunk_words=0))
    seen: set[str] = set()
    for chunk in chunks:
        sents = set(split_sentences(chunk.text))
        assert not sents & seen
        seen |= sents


def test_respect_sections_never_mixes_sections() -> None:
    text = "# T\n\n## A\n\nCâu một của mục A.\n\n## B\n\nCâu một của mục B."
    doc = make_doc(text)
    separated = chunk_document(doc, ChunkingConfig(chunk_size=100, chunk_overlap=10, respect_sections=True))
    assert [c.section for c in separated] == ["A", "B"]
    merged = chunk_document(
        doc, ChunkingConfig(chunk_size=100, chunk_overlap=10, respect_sections=False, min_chunk_words=0)
    )
    assert len(merged) == 1 and merged[0].section == "A / B"


def test_sentence_longer_than_chunk_size_is_split() -> None:
    long_sentence = " ".join(f"từ{i}" for i in range(95)) + "."
    doc = make_doc("## Mục\n\n" + long_sentence)
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=40, chunk_overlap=10, min_chunk_words=0))
    assert [c.n_words for c in chunks] == [40, 40, 15]


def test_tiny_tail_is_merged_into_previous_chunk() -> None:
    sentences = numbered_sentences(4, words_per_sentence=20) + ["Hết."]
    doc = make_doc("## Mục\n\n" + " ".join(sentences))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=40, chunk_overlap=0, min_chunk_words=5))
    assert chunks[-1].text.endswith("Hết.")
    assert all(c.n_words >= 5 for c in chunks)


def test_chunk_ids_are_unique_and_deterministic() -> None:
    doc = make_doc("# T\n\n## A\n\n" + " ".join(numbered_sentences(12)), doc_id="quy_dinh")
    config = ChunkingConfig(chunk_size=40, chunk_overlap=10)
    first = chunk_document(doc, config)
    second = chunk_document(doc, config)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert first[0].chunk_id == "quy_dinh#000"


def test_index_text_includes_heading_context() -> None:
    doc = make_doc("# Chính sách nghỉ phép\n\n## 1. Số ngày phép\n\nMười hai ngày.")
    chunk = chunk_document(doc, ChunkingConfig())[0]
    assert chunk.index_text(include_headings=True).startswith("Chính sách nghỉ phép > 1. Số ngày phép\n")
    assert chunk.index_text(include_headings=False) == "Mười hai ngày."


def test_invalid_overlap_raises() -> None:
    doc = make_doc("Một câu.")
    with pytest.raises(ConfigError):
        chunk_document(doc, ChunkingConfig(chunk_size=20, chunk_overlap=20))


def test_load_documents_reads_md_and_txt_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "a_quy_dinh.md").write_text("# Tiêu đề A\n\nNội dung.", encoding="utf-8")
    (tmp_path / "b-ghi-chu.txt").write_text(unicodedata.normalize("NFD", "Không có tiêu đề."), encoding="utf-8")
    (tmp_path / ".an.md").write_text("# Ẩn", encoding="utf-8")
    (tmp_path / "anh.png").write_bytes(b"\x89PNG")
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["a_quy_dinh", "b-ghi-chu"]
    assert [d.title for d in docs] == ["Tiêu đề A", "b ghi chu"]
    assert docs[1].text == "Không có tiêu đề."


def test_empty_document_has_no_chunks() -> None:
    assert chunk_document(make_doc("# Chỉ có tiêu đề\n\n"), ChunkingConfig()) == []


def test_save_and_load_chunks_roundtrip(tmp_path: Path) -> None:
    doc = make_doc("# Tiêu đề\n\n## Mục\n\nNội dung có dấu tiếng Việt.")
    chunks = chunk_document(doc, ChunkingConfig())
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
    assert "tiếng Việt" in path.read_text(encoding="utf-8")
