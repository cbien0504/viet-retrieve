"""Đọc văn bản (.md/.txt) và chia thành các đoạn (chunk) theo cấu trúc mục + câu, có gối đầu.

Thuật toán:
1. Chuẩn hoá Unicode NFC (văn bản tiếng Việt hay lẫn dạng tổ hợp NFD).
2. Tách văn bản thành các mục theo heading Markdown (``#`` là tiêu đề văn bản, ``##`` trở xuống là mục).
3. Mỗi mục tách thành đoạn văn / gạch đầu dòng, rồi thành câu.
4. Gom các câu liên tiếp thành đoạn có tối đa ``chunk_size`` từ. Đoạn kế tiếp lặp lại các câu cuối
   của đoạn trước (tối đa ``chunk_overlap`` từ) để không mất ngữ cảnh ở ranh giới.
"Từ" ở đây là token tách bằng khoảng trắng (với tiếng Việt xấp xỉ một âm tiết).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

from src.config import ChunkingConfig

SUPPORTED_SUFFIXES: tuple[str, ...] = (".md", ".txt")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^[-*+•]\s+")
_NUMBERED_RE = re.compile(r"^\d+[.)]\s+")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_EMPHASIS_RE = re.compile(r"\*\*|__|`")
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")
_ABBREVIATIONS = frozenset({"TP.", "Tp.", "ThS.", "TS.", "PGS.", "GS.", "Th.S", "St."})


def normalize_text(text: str) -> str:
    """Chuẩn hoá Unicode NFC, thống nhất ký tự xuống dòng và khoảng trắng đặc biệt."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace(" ", " ").replace("​", "").replace("﻿", "")


def count_words(text: str) -> int:
    """Số token tách bằng khoảng trắng."""
    return len(text.split())


def _clean_inline(text: str) -> str:
    text = _LINK_RE.sub(r"\1", text)
    text = _EMPHASIS_RE.sub("", text)
    return " ".join(text.split())


def split_sentences(paragraph: str) -> list[str]:
    """Tách một đoạn văn thành câu: cắt sau dấu kết câu khi ký tự kế tiếp là chữ hoa, số hoặc ngoặc mở.

    Không cắt sau các từ viết tắt phổ biến (``TP.``, ``TS.``...). Dấu chấm trong số (``1.200.000``)
    không bị cắt vì không có khoảng trắng phía sau.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(paragraph):
        end = match.end()
        next_char = paragraph[end : end + 1]
        if not next_char or not (next_char.isupper() or next_char.isdigit() or next_char in "\"'“(["):
            continue
        candidate = paragraph[start:end].strip()
        if candidate.split()[-1] in _ABBREVIATIONS:
            continue
        sentences.append(candidate)
        start = end
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@dataclass(frozen=True)
class Section:
    """Một mục của văn bản: heading (có thể rỗng với phần mở đầu) và các đoạn văn."""

    heading: str
    paragraphs: tuple[str, ...]


def parse_document(text: str) -> tuple[str | None, list[Section]]:
    """Phân tích văn bản Markdown/thuần thành (tiêu đề H1 nếu có, danh sách mục).

    Mỗi gạch đầu dòng được coi là một đoạn văn riêng; heading cấp >= 2 được ghép theo phân cấp
    (ví dụ ``"2. Đăng ký > 2.1 Trực tuyến"``).
    """
    title: str | None = None
    sections: list[tuple[str, list[str]]] = [("", [])]
    heading_stack: list[tuple[int, str]] = []
    para_lines: list[str] = []

    def flush_paragraph() -> None:
        if para_lines:
            sections[-1][1].append(" ".join(para_lines))
            para_lines.clear()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            level, heading = len(heading_match.group(1)), _clean_inline(heading_match.group(2))
            if level == 1 and title is None:
                title = heading
                continue
            heading_stack = [(lv, h) for lv, h in heading_stack if lv < level]
            heading_stack.append((level, heading))
            sections.append((" > ".join(h for _, h in heading_stack), []))
            continue
        if _BULLET_RE.match(line) or _NUMBERED_RE.match(line):
            flush_paragraph()
            line = _BULLET_RE.sub("", line)
        para_lines.append(_clean_inline(line))
    flush_paragraph()
    parsed = [Section(heading, tuple(p for p in paras if p)) for heading, paras in sections]
    return title, [s for s in parsed if s.paragraphs]


@dataclass(frozen=True)
class Document:
    """Một văn bản đầu vào đã chuẩn hoá."""

    doc_id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    """Một đoạn văn bản dùng để index và trả về cho người dùng."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    position: int
    n_words: int
    source: str

    @property
    def heading(self) -> str:
        """Ngữ cảnh "Tiêu đề > Mục" của đoạn."""
        return f"{self.title} > {self.section}" if self.section else self.title

    def index_text(self, include_headings: bool = True) -> str:
        """Nội dung dùng để index (BM25 / embedding / reranker)."""
        return f"{self.heading}\n{self.text}" if include_headings else self.text

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Chunk:
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _Unit:
    section: str
    paragraph: int
    text: str
    n_words: int


def _title_from_path(path: Path) -> str:
    return re.sub(r"[_\-]+", " ", path.stem).strip() or path.stem


def load_documents(raw_dir: Path) -> list[Document]:
    """Đọc mọi file ``.md``/``.txt`` (đệ quy, bỏ qua file ẩn) trong ``raw_dir``, sắp xếp theo đường dẫn."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {raw_dir}")
    paths = sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    )
    documents: list[Document] = []
    for path in paths:
        text = normalize_text(path.read_text(encoding="utf-8"))
        title, _ = parse_document(text)
        rel = path.relative_to(raw_dir)
        documents.append(
            Document(
                doc_id=rel.with_suffix("").as_posix(),
                title=title or _title_from_path(path),
                text=text,
                source=rel.as_posix(),
            )
        )
    return documents


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _make_units(sections: list[Section], chunk_size: int) -> list[_Unit]:
    units: list[_Unit] = []
    paragraph_index = 0
    for section in sections:
        for paragraph in section.paragraphs:
            for sentence in split_sentences(paragraph):
                for piece in _split_long_sentence(sentence, chunk_size):
                    units.append(_Unit(section.heading, paragraph_index, piece, count_words(piece)))
            paragraph_index += 1
    return units


def _words(units: Iterable[_Unit]) -> int:
    return sum(u.n_words for u in units)


def pack_units(units: list[_Unit], chunk_size: int, chunk_overlap: int, min_chunk_words: int) -> list[list[_Unit]]:
    """Gom các câu liên tiếp thành nhóm <= ``chunk_size`` từ, gối đầu <= ``chunk_overlap`` từ.

    Phần đuôi cuối cùng có ít hơn ``min_chunk_words`` từ mới được gộp vào nhóm trước (nhóm đó có thể
    vượt ``chunk_size`` tối đa ``min_chunk_words`` từ) để tránh sinh đoạn quá vụn.
    """
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    n_carried = 0
    for unit in units:
        if current and _words(current) + unit.n_words > chunk_size:
            groups.append(current)
            carry: list[_Unit] = []
            for previous in reversed(current):
                if _words(carry) + previous.n_words > chunk_overlap:
                    break
                carry.insert(0, previous)
            while carry and _words(carry) + unit.n_words > chunk_size:
                carry.pop(0)
            current, n_carried = carry, len(carry)
        current.append(unit)
    if current:
        new_units = current[n_carried:]
        if groups and n_carried < len(current) and _words(new_units) < min_chunk_words:
            groups[-1].extend(new_units)
        else:
            groups.append(current)
    return groups


def _join_units(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i > 0:
            parts.append(" " if unit.paragraph == units[i - 1].paragraph else "\n")
        parts.append(unit.text)
    return "".join(parts)


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Chia một văn bản thành danh sách ``Chunk`` theo cấu hình."""
    config.validate()
    _, sections = parse_document(document.text)
    units = _make_units(sections, config.chunk_size)
    if config.respect_sections:
        unit_groups = [list(run) for _, run in groupby(units, key=lambda u: u.section)]
    else:
        unit_groups = [units]

    chunks: list[Chunk] = []
    for group in unit_groups:
        for packed in pack_units(group, config.chunk_size, config.chunk_overlap, config.min_chunk_words):
            text = _join_units(packed)
            section = " / ".join(dict.fromkeys(u.section for u in packed if u.section))
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{position:03d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    section=section,
                    text=text,
                    position=position,
                    n_words=count_words(text),
                    source=document.source,
                )
            )
    return chunks


def chunk_documents(documents: Iterable[Document], config: ChunkingConfig) -> list[Chunk]:
    """Chia nhiều văn bản; thứ tự chunk theo thứ tự văn bản rồi vị trí trong văn bản."""
    return [chunk for document in documents for chunk in chunk_document(document, config)]


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    """Ghi chunk ra JSONL (UTF-8, giữ nguyên dấu tiếng Việt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Đọc chunk từ JSONL."""
    with path.open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
