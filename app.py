"""Demo Streamlit: tìm kiếm quy định nội bộ bằng hybrid retrieval tiếng Việt.

Chạy:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from src.config import MODE_LABELS, load_config
from src.pipeline import RetrievalPipeline, SearchHit, SearchResponse, read_manifest

# Câu hỏi mẫu lấy từ bộ đánh giá, chọn để thấy rõ khác biệt giữa các chế độ.
EXAMPLES = [
    "Đám cưới của nhân viên thì công ty mừng bao nhiêu tiền?",
    "Khi nào thì tôi nhận được tiền công hằng tháng?",
    "Ứng viên có mấy ngày để trả lời offer?",
    "Đi gặp khách hàng ở Đà Nẵng, tôi có được đặt vé bay hay phải đi tàu?",
    "Đội dev có được copy database production về máy để test không?",
    "Khoản chi trên 20 triệu đồng do ai phê duyệt?",
]

st.set_page_config(page_title="Tìm kiếm quy định nội bộ", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Đang nạp index (lần đầu có thể phải dựng index và tải model)…")
def get_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline.load(load_config())


@st.cache_data
def load_gold(path: str) -> dict[str, set[str]]:
    """Câu hỏi của bộ đánh giá -> id đoạn đúng, để đánh dấu khi demo bằng câu hỏi mẫu."""
    file = Path(path)
    if not file.is_file():
        return {}
    records = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["question"]: set(r["relevant_chunk_ids"]) for r in records}


def score_details(hit: SearchHit, mode: str) -> list[str]:
    """Giải thích điểm của một kết quả theo từng thành phần."""
    if mode == "bm25":
        return [f"BM25 {hit.score:.2f}"]
    if mode == "dense":
        return [f"cosine {hit.score:.3f}"]
    parts = [f"rerank {hit.rerank_score:.3f}"] if hit.rerank_score is not None else []
    if hit.rrf_score is not None:
        parts.append(f"RRF {hit.rrf_score:.4f}")
    parts.append(f"BM25 hạng {hit.bm25_rank}" if hit.bm25_rank else "BM25: ngoài danh sách ứng viên")
    parts.append(f"dense hạng {hit.dense_rank}" if hit.dense_rank else "dense: ngoài danh sách ứng viên")
    return parts


def render_timings(response: SearchResponse) -> None:
    t = response.timings_ms
    names = {"bm25_ms": "BM25", "dense_ms": "dense", "fusion_ms": "RRF", "rerank_ms": "rerank"}
    steps = " · ".join(f"{label} {t[key]:.0f} ms" for key, label in names.items() if key in t)
    loads = {"dense_load_ms": "nạp embedder", "rerank_load_ms": "nạp reranker"}
    loaded = ", ".join(f"{t[key]:.0f} ms {label}" for key, label in loads.items() if key in t)
    st.caption(f"⏱ {t['total_ms']:.0f} ms ({steps})" + (f" — trong đó {loaded}" if loaded else ""))


def render_hit(hit: SearchHit, mode: str, gold: set[str]) -> None:
    with st.container(border=True):
        mark = " ✅ đoạn đúng" if hit.chunk.chunk_id in gold else ""
        st.markdown(f"**#{hit.rank} · {html.escape(hit.chunk.heading)}**{mark}")
        st.caption(" · ".join([f"`{hit.chunk.chunk_id}`", *score_details(hit, mode)]))
        body = html.escape(hit.chunk.text).replace("\n", "<br>")
        st.markdown(f"<div style='line-height:1.55'>{body}</div>", unsafe_allow_html=True)


def use_example() -> None:
    if st.session_state.get("example"):
        st.session_state.query = st.session_state.example


try:
    pipeline = get_pipeline()
except Exception as exc:  # hiển thị lỗi thân thiện thay vì traceback
    st.error(f"Không nạp được index: {exc}")
    st.info("Thử dựng lại index bằng lệnh: `python -m src.cli build`")
    st.stop()

config = pipeline.config
manifest = read_manifest(config) or {}
gold_by_question = load_gold(str(config.resolve(config.eval.eval_set_path)))
modes = pipeline.available_modes

with st.sidebar:
    st.header("Cấu hình truy hồi")
    default_mode = config.retrieval.default_mode if config.retrieval.default_mode in modes else modes[0]
    mode = st.radio("Chế độ", modes, index=modes.index(default_mode), format_func=MODE_LABELS.get)
    top_k = st.slider("Số kết quả (top-k)", min_value=1, max_value=10, value=config.retrieval.top_k)
    compare = st.toggle("So sánh tất cả chế độ", value=False)
    st.divider()
    st.markdown(
        f"**Dữ liệu:** {manifest.get('num_documents', '?')} văn bản, {manifest.get('num_chunks', '?')} đoạn  \n"
        f"**Embedding:** `{config.dense.model_name}`  \n"
        f"**Reranker:** `{config.reranker.model_name}`  \n"
        f"**RRF:** k={config.fusion.rrf_k}, {config.fusion.candidates_per_retriever} ứng viên/retriever"
    )
    if pipeline.reranker is not None:
        policy = (
            "nạp khi cần, giải phóng ngay sau mỗi truy vấn"
            if config.reranker.release_after_use
            else "nạp khi cần, giữ lại trong RAM"
        )
        state = "đang nạp" if pipeline.reranker.model.is_loaded else "chưa nạp"
        st.caption(f"Reranker: {policy} (hiện {state}).")
    st.warning("Dữ liệu tổng hợp do AI soạn cho một công ty giả định — không phải quy định thật.", icon="⚠️")

st.title("🔎 Tìm kiếm quy định nội bộ")
st.caption("Hybrid retrieval tiếng Việt: BM25 (underthesea) + embedding e5 + Reciprocal Rank Fusion + cross-encoder")

st.pills("Câu hỏi mẫu", EXAMPLES, key="example", on_change=use_example)
query = st.text_input("Câu hỏi", key="query", placeholder="Ví dụ: Tôi được nghỉ phép bao nhiêu ngày một năm?")

if query.strip():
    gold = gold_by_question.get(query.strip(), set())
    if compare:
        responses: dict[str, SearchResponse] = {}
        with st.spinner("Đang tìm theo tất cả chế độ…"), pipeline.hold_models():
            for m in modes:
                responses[m] = pipeline.search(query, mode=m, top_k=top_k)
        columns = st.columns(len(modes))
        for column, m in zip(columns, modes, strict=True):
            with column:
                st.subheader(MODE_LABELS[m])
                render_timings(responses[m])
                for hit in responses[m].hits:
                    mark = " ✅" if hit.chunk.chunk_id in gold else ""
                    st.markdown(f"{hit.rank}. {html.escape(hit.chunk.heading)}{mark}")
        if gold:
            st.caption("✅ = đoạn đúng theo bộ đánh giá (chỉ hiển thị với câu hỏi mẫu).")
    else:
        with st.spinner(f"Đang tìm ({MODE_LABELS[mode]})…"):
            response = pipeline.search(query, mode=mode, top_k=top_k)
        render_timings(response)
        if not response.hits:
            st.info("Không tìm thấy đoạn nào phù hợp.")
        for hit in response.hits:
            render_hit(hit, mode, gold)
else:
    st.info("Nhập câu hỏi hoặc chọn một câu hỏi mẫu ở trên.")
