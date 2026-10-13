"""本地知识检索（RAG）。

三路通道并行召回，再用 RRF（Reciprocal Rank Fusion）融合，最后可选 cross-encoder 重排：
- 关键词通道：按词面重合度打分，召回广、可解释；
- 意图通道：先按意图收敛候选，再打分，更准；
- 语义通道：用本地 embedding 算余弦相似度，能召回「换了说法」的同义问题（需 embedder）。

融合用 RRF：只依赖各通道的排名，不要求把「词面重合度」和「余弦相似度」标到同一量纲，
是混合检索里成熟且省心的做法。融合后若配置了 reranker，则对候选做一次精排。

本模块只负责「检索」；把它接进对话在编排层完成。三路通道都是「尽力而为」：
缺 embedder / 模型报错时自动少跑一路，不影响其余通道。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.config import BASE_DIR
from app.embedder import Embedder, Reranker, cosine

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

DEFAULT_KB_PATH = BASE_DIR / "data" / "knowledge.json"

# RRF 平滑常数：越大越削弱「排名靠前」的优势，取检索界常用默认值 60
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """英文按词、中文按二元组切分，便于轻量词面匹配。"""
    lowered = text.lower()
    tokens = _WORD_RE.findall(lowered)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(frozen=True, slots=True)
class Doc:
    doc_id: str
    title: str
    text: str
    intent: str = "general"


@dataclass(frozen=True, slots=True)
class Hit:
    doc_id: str
    title: str
    text: str
    score: float
    channel: str  # keyword / intent / vector / rerank


class KnowledgeBase:
    """内存知识库，提供词面、意图、语义三路检索与融合。"""

    def __init__(
        self,
        docs: Sequence[Doc],
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._docs = list(docs)
        self._doc_tokens = {d.doc_id: set(tokenize(f"{d.title} {d.text}")) for d in self._docs}
        self._embedder = embedder
        self._reranker = reranker
        # 文档向量懒计算：首次用到语义通道时才编码，构造 KnowledgeBase 不强依赖模型
        self._doc_vectors: list[list[float]] | None = None

    @classmethod
    def from_json(
        cls,
        path: str | Path = DEFAULT_KB_PATH,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> "KnowledgeBase":
        path = Path(path)
        if not path.exists():
            return cls([], embedder=embedder, reranker=reranker)
        raw = json.loads(path.read_text(encoding="utf-8"))
        docs = [
            Doc(
                doc_id=str(item["doc_id"]),
                title=str(item.get("title", "")),
                text=str(item["text"]),
                intent=str(item.get("intent", "general")),
            )
            for item in raw
        ]
        return cls(docs, embedder=embedder, reranker=reranker)

    @property
    def size(self) -> int:
        return len(self._docs)

    @property
    def has_vector(self) -> bool:
        return self._embedder is not None

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    def warmup(self) -> None:
        """预先算好文档向量并加载模型，避免首个请求承担冷启动延迟。"""
        self._ensure_vectors()
        if self._reranker is not None:
            try:
                self._reranker.score("预热", ["预热"])  # 触发模型加载
            except Exception:  # noqa: BLE001 —— 预热失败不影响功能，只影响首请求速度
                logger.warning("重排模型预热失败，将在首次调用时重试", exc_info=True)

    # ---------- 词面 / 意图通道 ----------
    def _rank(self, query: str, candidates: Sequence[Doc]) -> list[tuple[Doc, float]]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scored: list[tuple[Doc, float]] = []
        for doc in candidates:
            overlap = query_tokens & self._doc_tokens[doc.doc_id]
            if overlap:
                scored.append((doc, len(overlap) / len(query_tokens)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def search_keyword(self, query: str, *, top_k: int) -> list[Hit]:
        ranked = self._rank(query, self._docs)
        return [Hit(d.doc_id, d.title, d.text, s, "keyword") for d, s in ranked[:top_k]]

    def search_by_intent(self, query: str, intent: str | None, *, top_k: int) -> list[Hit]:
        candidates = [d for d in self._docs if intent is None or d.intent == intent]
        ranked = self._rank(query, candidates)
        return [Hit(d.doc_id, d.title, d.text, s, "intent") for d, s in ranked[:top_k]]

    # ---------- 语义通道 ----------
    def _ensure_vectors(self) -> None:
        """懒计算文档向量；失败则关闭语义通道（退化为词面检索）。"""
        if self._doc_vectors is not None or self._embedder is None:
            return
        if not self._docs:
            self._doc_vectors = []
            return
        try:
            texts = [f"{d.title}。{d.text}" for d in self._docs]
            self._doc_vectors = self._embedder.embed(texts)
        except Exception:  # noqa: BLE001 —— 模型不可用不该拖垮检索
            logger.warning("文档向量计算失败，语义通道关闭", exc_info=True)
            self._embedder = None
            self._doc_vectors = None

    def search_vector(self, query: str, *, top_k: int) -> list[Hit]:
        if self._embedder is None:
            return []
        self._ensure_vectors()
        if not self._doc_vectors:
            return []
        try:
            query_vector = self._embedder.embed([query])[0]
        except Exception:  # noqa: BLE001
            logger.warning("查询向量计算失败，本轮跳过语义通道", exc_info=True)
            return []
        scored = [(d, cosine(query_vector, vec)) for d, vec in zip(self._docs, self._doc_vectors)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            Hit(d.doc_id, d.title, d.text, s, "vector")
            for d, s in scored[:top_k]
            if s > 0
        ]

    # ---------- 融合 + 重排 ----------
    async def retrieve(
        self,
        query: str,
        *,
        intent: str | None = None,
        top_k: int = 4,
    ) -> list[Hit]:
        """三路并行召回 → RRF 融合 →（可选）重排，返回 top_k。"""
        per_channel = max(top_k, 4)

        async def keyword_channel() -> list[Hit]:
            return self.search_keyword(query, top_k=per_channel)

        async def intent_channel() -> list[Hit]:
            return self.search_by_intent(query, intent, top_k=per_channel)

        async def vector_channel() -> list[Hit]:
            return self.search_vector(query, top_k=per_channel)

        keyword_hits, intent_hits, vector_hits = await asyncio.gather(
            keyword_channel(), intent_channel(), vector_channel()
        )
        channels = [c for c in (keyword_hits, intent_hits, vector_hits) if c]
        if not channels:
            return []

        fused = self._rrf(channels)
        pool = fused[: min(len(fused), top_k * 2)]
        return self._finalize(query, pool, top_k)

    @staticmethod
    def _rrf(channels: Sequence[Sequence[Hit]], *, k: int = RRF_K) -> list[tuple[Hit, float]]:
        """Reciprocal Rank Fusion：每个通道按排名贡献 1/(k+rank)，同文档累加。"""
        scores: dict[str, float] = {}
        best: dict[str, Hit] = {}
        for hits in channels:
            for rank, hit in enumerate(hits, start=1):
                scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (k + rank)
                prev = best.get(hit.doc_id)
                if prev is None or hit.score > prev.score:
                    best[hit.doc_id] = hit
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(best[doc_id], score) for doc_id, score in ordered]

    def _finalize(
        self, query: str, pool: Sequence[tuple[Hit, float]], top_k: int
    ) -> list[Hit]:
        """对融合候选做（可选）重排，并把分数归一化到 [0,1] 便于展示。"""
        if self._reranker is not None and len(pool) > 1:
            try:
                docs = [f"{hit.title}。{hit.text}" for hit, _ in pool]
                scores = self._reranker.score(query, docs)
                if len(scores) == len(pool):
                    reranked = sorted(
                        zip((hit for hit, _ in pool), scores),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                    return self._to_hits([(h, s) for h, s in reranked], top_k, channel="rerank")
            except Exception:  # noqa: BLE001 —— 重排失败退回融合排序
                logger.warning("重排失败，退回 RRF 融合排序", exc_info=True)
        return self._to_hits(pool, top_k)

    @staticmethod
    def _to_hits(
        pairs: Sequence[tuple[Hit, float]], top_k: int, *, channel: str | None = None
    ) -> list[Hit]:
        if not pairs:
            return []
        values = [score for _, score in pairs]
        low, high = min(values), max(values)
        span = high - low
        result: list[Hit] = []
        for hit, score in pairs[:top_k]:
            normalized = 1.0 if span <= 0 else (score - low) / span
            result.append(
                Hit(hit.doc_id, hit.title, hit.text, round(normalized, 4), channel or hit.channel)
            )
        return result
