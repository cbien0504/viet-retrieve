"""Smoke test giao diện Streamlit bằng AppTest, trên một kho văn bản nhỏ chỉ có index BM25 (không cần model)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import CONFIG_ENV_VAR, PROJECT_ROOT, load_config
from src.pipeline import build_index

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture()
def bm25_only_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "gui_xe.md").write_text("# Gửi xe\n\n## Phí\n\nCông ty hỗ trợ 100% phí gửi xe máy.", encoding="utf-8")
    (raw / "luong.md").write_text("# Lương\n\n## Kỳ trả\n\nLương trả vào ngày 5 hằng tháng.", encoding="utf-8")
    # >= 3 đoạn: với 2 đoạn, IDF của BM25Okapi = log(1.5 / 1.5) = 0 cho mọi từ xuất hiện ở 1 đoạn.
    (raw / "phep.md").write_text("# Nghỉ phép\n\n## Số ngày\n\nNhân viên có 12 ngày phép năm.", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"raw_dir": "raw", "index_dir": "index"},
                "sparse": {"tokenizer": "syllable"},
                "retrieval": {"default_mode": "bm25"},
                "eval": {"eval_set_path": "khong_co.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    build_index(load_config(config_path), with_dense=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))
    return config_path


def test_app_renders_and_answers_a_query(bm25_only_config: Path) -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    assert [r.value for r in app.radio][:1] == ["bm25"], "chỉ có chế độ BM25 khi index không có phần dense"

    app.text_input(key="query").input("phí gửi xe máy").run()
    assert not app.exception
    captions = " ".join(c.value for c in app.caption)
    assert "gui_xe#" in captions
