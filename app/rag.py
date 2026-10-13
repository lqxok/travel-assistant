"""本地知识检索（RAG）。

两路通道并行召回，再合并去重：
- 关键词通道：按词面重合度打分，召回广；
- 意图通道：先按意图收敛候选，再打分，更准。

本模块只负责「检索」；把它接进对话在编排层完成。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.config import BASE_DIR

_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

DEFAULT_KB_PATH = BASE_DIR / "data" / "knowledge.json"


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
    channel: str  # keyword / intent


class KnowledgeBase:
    """内存知识库，提供关键词与意图两路检索。"""

    def __init__(self, docs: Sequence[Doc]) -> None:
        self._docs = list(docs)
        self._doc_tokens = {d.doc_id: set(tokenize(f"{d.title} {d.text}")) for d in self._docs}

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_KB_PATH) -> "KnowledgeBase":
        path = Path(path)
        if not path.exists():
            return cls([])
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
        return cls(docs)

    @property
    def size(self) -> int:
        return len(self._docs)

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

    async def retrieve(
        self,
        query: str,
        *,
        intent: str | None = None,
        top_k: int = 4,
        keyword_weight: float = 0.85,
        intent_weight: float = 0.95,
    ) -> list[Hit]:
        """并行跑两路通道，合并去重后返回 top_k。"""
        per_channel = max(top_k, 4)

        async def keyword_channel() -> list[Hit]:
            return [
                Hit(h.doc_id, h.title, h.text, h.score * keyword_weight, h.channel)
                for h in self.search_keyword(query, top_k=per_channel)
            ]

        async def intent_channel() -> list[Hit]:
            return [
                Hit(h.doc_id, h.title, h.text, h.score * intent_weight, h.channel)
                for h in self.search_by_intent(query, intent, top_k=per_channel)
            ]

        keyword_hits, intent_hits = await asyncio.gather(keyword_channel(), intent_channel())
        return self._merge(keyword_hits + intent_hits, top_k=top_k)

    @staticmethod
    def _merge(hits: Sequence[Hit], *, top_k: int) -> list[Hit]:
        """同一篇文档只保留最高分，再按分数排序。"""
        best: dict[str, Hit] = {}
        for hit in hits:
            prev = best.get(hit.doc_id)
            if prev is None or hit.score > prev.score:
                best[hit.doc_id] = hit
        ordered = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return ordered[:top_k]
