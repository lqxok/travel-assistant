"""本地知识检索测试（融合/重排用假实现，不依赖真实模型）。"""

from __future__ import annotations

from app.rag import Doc, KnowledgeBase


def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        [
            Doc("d1", "签证与证件", "护照有效期至少 6 个月，签证需要行程单与酒店订单。", "guide"),
            Doc("d2", "旅行预算", "交通约占 40%，住宿约 35%，餐饮约 15%。", "budget"),
        ]
    )


async def test_retrieve_returns_relevant_doc():
    hits = await _kb().retrieve("签证需要准备什么", intent="guide")
    assert hits
    assert hits[0].doc_id == "d1"


async def test_retrieve_dedups_by_doc_id():
    # 同一篇可能被多路通道同时命中，只应出现一次
    hits = await _kb().retrieve("预算 交通 住宿", intent="budget", top_k=5)
    ids = [h.doc_id for h in hits]
    assert len(ids) == len(set(ids))


async def test_retrieve_respects_top_k():
    hits = await _kb().retrieve("签证 预算 交通 住宿", top_k=1)
    assert len(hits) == 1


def test_from_json_missing_file_returns_empty(tmp_path):
    kb = KnowledgeBase.from_json(tmp_path / "not_exist.json")
    assert kb.size == 0


# ---------------- 混合检索 / 重排（假模型） ----------------
class FakeEmbedder:
    """按文本内容给固定二维向量；query「q」与含 alpha 的文本同向。"""

    def embed(self, texts):
        vectors = []
        for text in texts:
            first = 1.0 if "alpha" in text else 0.0
            second = 1.0 if "beta" in text else 0.0
            if text == "q":  # 让 query 同时靠近 alpha 与 beta，制造并列候选
                first = second = 1.0
            vectors.append([first, second])
        return vectors


class FakeReranker:
    """总是认为含 beta 的文档更相关，用来验证重排确实改变了顺序。"""

    def score(self, query, documents):
        return [2.0 if "beta" in doc else 0.0 for doc in documents]


class BrokenEmbedder:
    def embed(self, texts):
        raise RuntimeError("模型不可用")


def _hybrid_kb(**kwargs) -> KnowledgeBase:
    return KnowledgeBase(
        [Doc("a", "A", "alpha", "general"), Doc("b", "B", "beta", "general")], **kwargs
    )


async def test_semantic_channel_recalls_without_lexical_overlap():
    kb = KnowledgeBase(
        [Doc("a", "A", "alpha", "general"), Doc("b", "B", "beta", "general")],
        embedder=FakeEmbedder(),
    )
    # 「q」与任何文档都没有词面重合，只能靠语义通道召回 a
    hits = await kb.retrieve("q", top_k=3)
    assert hits and hits[0].doc_id == "a"
    assert hits[0].channel == "vector"


async def test_reranker_can_reorder_fused_candidates():
    kb = _hybrid_kb(embedder=FakeEmbedder(), reranker=FakeReranker())
    hits = await kb.retrieve("q", top_k=2)
    assert [h.doc_id for h in hits][0] == "b"
    assert hits[0].channel == "rerank"


async def test_broken_embedder_falls_back_to_lexical():
    kb = KnowledgeBase(
        [Doc("d1", "签证与证件", "护照有效期至少 6 个月", "guide")],
        embedder=BrokenEmbedder(),
    )
    hits = await kb.retrieve("签证需要什么", intent="guide")
    assert hits and hits[0].doc_id == "d1"
    assert not kb.has_vector  # 语义通道已自动关闭
