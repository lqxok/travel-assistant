# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
# PIP_INDEX_URL 可在构建时指定（国内服务器建议用镜像源加速）
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --upgrade pip --index-url ${PIP_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt --index-url ${PIP_INDEX_URL}

COPY app ./app
COPY data ./data

# 以非 root 用户运行
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
