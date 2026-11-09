"""Unit test cho tokenizer tiếng Việt và BM25 (src/sparse.py)."""

from __future__ import annotations

import unicodedata
from itertools import pairwise

import pytest

from src.sparse import VIETNAMESE_STOPWORDS, BM25Retriever, VietnameseTokenizer


@pytest.fixture(scope="module")
def underthesea_tokenizer() -> VietnameseTokenizer:
    return VietnameseTokenizer("underthesea")


@pytest.fixture(scope="module")
def syllable_tokenizer() -> VietnameseTokenizer:
    return VietnameseTokenizer("syllable")


def test_syllable_tokenizer_lowercases_and_drops_punctuation(syllable_tokenizer: VietnameseTokenizer) -> None:
    tokens = syllable_tokenizer.tokenize("Phụ cấp ăn trưa: 50.000 đồng/ngày, áp dụng từ 01/01/2026!")
    assert tokens == ["phụ", "cấp", "ăn", "trưa", "50.000", "đồng/ngày", "áp", "dụng", "01/01/2026"]


def test_stopwords_are_removed_only_when_enabled() -> None:
    text = "Nhân viên được nghỉ phép và nghỉ ốm"
    with_stop = VietnameseTokenizer("syllable", remove_stopwords=True).tokenize(text)
    without_stop = VietnameseTokenizer("syllable", remove_stopwords=False).tokenize(text)
    assert "và" not in with_stop and "được" not in with_stop
    assert "và" in without_stop and "được" in without_stop


def test_underthesea_joins_compound_words(underthesea_tokenizer: VietnameseTokenizer) -> None:
    tokens = underthesea_tokenizer.tokenize("Nhân viên chính thức được hưởng 12 ngày nghỉ phép năm.")
    assert "nhân_viên" in tokens
    assert "chính_thức" in tokens
    assert "nghỉ_phép" in tokens
    assert "12" in tokens
    assert "." not in tokens and "được" not in tokens


def test_underthesea_normalizes_unicode_and_tone_marks(underthesea_tokenizer: VietnameseTokenizer) -> None:
    nfc = underthesea_tokenizer.tokenize("Bảo hiểm sức khỏe cho nhân viên")
    nfd = underthesea_tokenizer.tokenize(unicodedata.normalize("NFD", "Bảo hiểm sức khỏe cho nhân viên"))
    old_style = underthesea_tokenizer.tokenize("Bảo hiểm sức khoẻ cho nhân viên")
    assert nfc == nfd == old_style


def test_question_words_are_stopwords(underthesea_tokenizer: VietnameseTokenizer) -> None:
    tokens = underthesea_tokenizer.tokenize("Tôi được nghỉ phép bao nhiêu ngày?")
    assert "bao_nhiêu" not in tokens and "tôi" not in tokens
    assert "ngày" in tokens


def test_stopword_list_is_normalized() -> None:
    for word in VIETNAMESE_STOPWORDS:
        assert word == word.lower() == unicodedata.normalize("NFC", word)
        assert " " not in word


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        VietnameseTokenizer("spacy")


@pytest.fixture()
def tiny_bm25(syllable_tokenizer: VietnameseTokenizer) -> BM25Retriever:
    corpus = {
        "phep#0": "Nhân viên có 12 ngày nghỉ phép năm hưởng nguyên lương.",
        "om#0": "Nghỉ ốm từ 2 ngày phải nộp giấy chứng nhận của cơ sở y tế.",
        "luong#0": "Lương được trả vào ngày 5 hằng tháng qua chuyển khoản.",
        "xe#0": "Công ty hỗ trợ phí gửi xe máy tại tầng hầm.",
    }
    ids = list(corpus)
    return BM25Retriever(ids, syllable_tokenizer.tokenize_many(list(corpus.values())), syllable_tokenizer)


def test_bm25_ranks_lexically_matching_chunk_first(tiny_bm25: BM25Retriever) -> None:
    results = tiny_bm25.search("Gửi xe máy có được hỗ trợ không?", top_k=3)
    assert results[0][0] == "xe#0"
    assert all(a[1] >= b[1] for a, b in pairwise(results))


def test_bm25_returns_only_chunks_with_positive_score(tiny_bm25: BM25Retriever) -> None:
    assert tiny_bm25.search("hoàn toàn khác biệt", top_k=4) == []
    assert tiny_bm25.search("???", top_k=4) == []


def test_bm25_respects_top_k(tiny_bm25: BM25Retriever) -> None:
    assert len(tiny_bm25.search("ngày", top_k=1)) == 1


def test_bm25_rejects_mismatched_inputs(syllable_tokenizer: VietnameseTokenizer) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(["a", "b"], [["x"]], syllable_tokenizer)
    with pytest.raises(ValueError):
        BM25Retriever([], [], syllable_tokenizer)
