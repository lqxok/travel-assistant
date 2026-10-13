"""本地向量化与重排模型的封装（fastembed + ONNX，CPU 推理）。

- `FastEmbedder`：把文本编码成向量，默认 `BAAI/bge-small-zh-v1.5`（中文，512 维）；
- `FastReranker`：cross-encoder 精排，默认 `BAAI/bge-reranker-base`。

统一约定：**懒加载 + 失败降级**。模型只在首次调用时加载；一旦加载或推理失败，
`build_*` 会返回 `None`，检索层据此自动退回纯词面通道，保证没有模型环境（如 CI）
或模型下载失败时服务依旧可用。
"""

from __future__ import annotations

import logging
import math
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

# 这些取值表示「关闭该功能」
_DISABLED = {"", "none", "off", "0", "false"}


class Embedder(Protocol):
    """把一批文本编码成向量。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    """给 (query, 候选文档) 打相关性分，分越高越相关。"""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；任一向量为零向量时返回 0。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class FastEmbedder:
    """基于 fastembed 的文本向量器。"""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        vectors = self._ensure_model().embed(items)
        return [[float(x) for x in vec] for vec in vectors]


class FastReranker:
    """基于 fastembed 的 cross-encoder 重排器。"""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        return [float(s) for s in self._ensure_model().rerank(query, docs)]


def _fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except Exception as exc:  # noqa: BLE001 —— 缺失依赖时静默降级
        logger.warning("未安装 fastembed，语义检索/重排将关闭：%s", exc)
        return False
    return True


def is_disabled(model_name: str | None) -> bool:
    """判断配置值是否表示「关闭」。"""
    return (model_name or "").strip().lower() in _DISABLED


def build_embedder(model_name: str | None) -> Embedder | None:
    """按配置构造向量器；未配置或依赖缺失时返回 None。"""
    if is_disabled(model_name) or not _fastembed_available():
        return None
    return FastEmbedder((model_name or DEFAULT_EMBED_MODEL).strip())


def build_reranker(model_name: str | None) -> Reranker | None:
    """按配置构造重排器；未配置或依赖缺失时返回 None。"""
    if is_disabled(model_name) or not _fastembed_available():
        return None
    return FastReranker((model_name or DEFAULT_RERANK_MODEL).strip())
