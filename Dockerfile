# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    # 国内直连 HuggingFace 常被墙，统一走镜像；模型缓存放到共享目录供非 root 用户读取
    HF_ENDPOINT=https://hf-mirror.com \
    FASTEMBED_CACHE_PATH=/opt/models

WORKDIR /app

COPY requirements.txt .
# PIP_INDEX_URL 可在构建时指定（国内服务器建议用镜像源加速）
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --upgrade pip --index-url ${PIP_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt --index-url ${PIP_INDEX_URL}

# 构建期预下载本地语义/重排模型。
# 故意放在 `COPY app` 之前：只要模型名不变，改代码重建时这一层直接命中缓存，不用重新下模型。
ARG EMBED_MODEL=BAAI/bge-small-zh-v1.5
ARG RERANK_MODEL=BAAI/bge-reranker-base
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBED_MODEL}')" \
    && python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='${RERANK_MODEL}')" \
    && chmod -R a+rX /opt/models

COPY app ./app
COPY data ./data

# 以非 root 用户运行
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
