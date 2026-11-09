"""Đọc và kiểm tra cấu hình từ ``config.yaml`` thành các dataclass có kiểu."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFIG_ENV_VAR = "VIET_RETRIEVE_CONFIG"

MODES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")
MODE_LABELS: dict[str, str] = {
    "bm25": "Chỉ BM25",
    "dense": "Chỉ dense",
    "hybrid": "Hybrid RRF",
    "hybrid_rerank": "Hybrid RRF + reranker",
}


class ConfigError(ValueError):
    """Cấu hình không hợp lệ."""


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    index_dir: str = "data/index"


@dataclass
class ChunkingConfig:
    chunk_size: int = 120
    chunk_overlap: int = 30
    respect_sections: bool = True
    min_chunk_words: int = 12
    include_headings: bool = True

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ConfigError("chunking.chunk_size phải > 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("chunking.chunk_overlap phải nằm trong [0, chunk_size)")
        if self.min_chunk_words < 0:
            raise ConfigError("chunking.min_chunk_words phải >= 0")


@dataclass
class SparseConfig:
    tokenizer: str = "underthesea"
    lowercase: bool = True
    remove_stopwords: bool = True
    k1: float = 1.5
    b: float = 0.75

    def validate(self) -> None:
        if self.tokenizer not in ("underthesea", "syllable"):
            raise ConfigError("sparse.tokenizer phải là 'underthesea' hoặc 'syllable'")


@dataclass
class DenseConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 16
    max_seq_length: int = 512
    device: str = "cpu"
    release_after_use: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError("dense.batch_size phải > 0")


@dataclass
class FusionConfig:
    rrf_k: int = 60
    candidates_per_retriever: int = 50
    bm25_weight: float = 1.0
    dense_weight: float = 1.0

    def validate(self) -> None:
        if self.rrf_k < 0:
            raise ConfigError("fusion.rrf_k phải >= 0")
        if self.candidates_per_retriever <= 0:
            raise ConfigError("fusion.candidates_per_retriever phải > 0")
        if self.bm25_weight < 0 or self.dense_weight < 0:
            raise ConfigError("fusion.*_weight phải >= 0")


@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    candidates: int = 20
    batch_size: int = 8
    max_length: int = 512
    device: str = "cpu"
    release_after_use: bool = True

    def validate(self) -> None:
        if self.candidates <= 0 or self.batch_size <= 0:
            raise ConfigError("reranker.candidates và reranker.batch_size phải > 0")


@dataclass
class RetrievalConfig:
    top_k: int = 5
    default_mode: str = "hybrid_rerank"

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ConfigError("retrieval.top_k phải > 0")
        if self.default_mode not in MODES:
            raise ConfigError(f"retrieval.default_mode phải thuộc {MODES}")


@dataclass
class EvalConfig:
    seed_path: str = "eval/questions_seed.yaml"
    eval_set_path: str = "eval/eval_set.jsonl"
    results_path: str = "eval/results.md"
    runs_dir: str = "eval/runs"
    batch_size: int = 10
    ks: list[int] = field(default_factory=lambda: [1, 5, 10])
    ndcg_k: int = 5
    modes: list[str] = field(default_factory=lambda: list(MODES))
    extra_rerankers: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError("eval.batch_size phải > 0")
        if not self.ks or min(self.ks) <= 0:
            raise ConfigError("eval.ks phải là danh sách số nguyên dương")
        unknown = [m for m in self.modes if m not in MODES]
        if unknown:
            raise ConfigError(f"eval.modes có chế độ không hợp lệ: {unknown}")


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    root: Path = PROJECT_ROOT

    def resolve(self, path: str | Path) -> Path:
        """Chuyển đường dẫn tương đối (tính từ thư mục chứa config) thành tuyệt đối."""
        p = Path(path)
        return p if p.is_absolute() else (self.root / p).resolve()

    def validate(self) -> None:
        for f in fields(self):
            section = getattr(self, f.name)
            if hasattr(section, "validate"):
                section.validate()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("root", None)
        return data


T = TypeVar("T")


def _build_section(cls: type[T], data: dict[str, Any] | None, name: str) -> T:
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Mục '{name}' trong config phải là một mapping")
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Khoá không hợp lệ trong mục '{name}': {sorted(unknown)}")
    return cls(**data)


def apply_overrides(config: Config, overrides: list[str] | None) -> Config:
    """Ghi đè cấu hình dạng ``mục.khoá=giá_trị`` (giá trị parse bằng YAML), ví dụ ``sparse.tokenizer=syllable``."""
    for item in overrides or []:
        key, sep, raw_value = item.partition("=")
        section_name, dot, field_name = key.strip().partition(".")
        section = getattr(config, section_name, None) if section_name != "root" else None
        if not sep or not dot or section is None or field_name not in {f.name for f in fields(section)}:
            raise ConfigError(f"Ghi đè không hợp lệ: '{item}' (định dạng: mục.khoá=giá_trị)")
        setattr(section, field_name, yaml.safe_load(raw_value))
    config.validate()
    return config


def load_config(path: str | Path | None = None) -> Config:
    """Đọc config từ ``path`` (hoặc biến môi trường ``VIET_RETRIEVE_CONFIG``, hoặc ``config.yaml``)."""
    config_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Không tìm thấy file cấu hình: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section_classes: dict[str, type] = {
        "paths": PathsConfig,
        "chunking": ChunkingConfig,
        "sparse": SparseConfig,
        "dense": DenseConfig,
        "fusion": FusionConfig,
        "reranker": RerankerConfig,
        "retrieval": RetrievalConfig,
        "eval": EvalConfig,
    }
    unknown = set(raw) - set(section_classes)
    if unknown:
        raise ConfigError(f"Mục không hợp lệ trong config: {sorted(unknown)}")
    kwargs = {name: _build_section(cls, raw.get(name), name) for name, cls in section_classes.items()}
    config = Config(**kwargs, root=config_path.parent)
    config.validate()
    return config
