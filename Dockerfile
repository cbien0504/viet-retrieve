# syntax=docker/dockerfile:1
# Image chạy demo Streamlit trên CPU. Mặc định tải sẵn model và dựng index lúc build để container chạy offline.
#   docker build -t viet-retrieve .
#   docker run --rm -p 8501:8501 viet-retrieve
# Không tải model lúc build (dùng volume cache riêng):
#   docker build --build-arg PRELOAD_MODELS=0 -t viet-retrieve .
#   docker run --rm -p 8501:8501 -v hf-cache:/models viet-retrieve
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

WORKDIR /app

# torch bản CPU trước: bản mặc định trên PyPI kéo theo ~3 GB thư viện CUDA không cần thiết.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 app && mkdir -p /models && chown app:app /app /models
USER app

COPY --chown=app:app config.yaml app.py ./
COPY --chown=app:app src/ src/
COPY --chown=app:app eval/ eval/
COPY --chown=app:app data/raw/ data/raw/

ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
        python -m src.cli download-models && python -m src.cli build ; \
    fi
# Model đã nằm sẵn trong image -> không gọi mạng khi nạp (nạp nhanh hơn, chạy được offline).
ENV HF_HUB_OFFLINE=${PRELOAD_MODELS}

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
