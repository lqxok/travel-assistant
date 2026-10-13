"""本地知识检索测试。"""

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
    # 同一篇可能被两路通道同时命中，只应出现一次
    hits = await _kb().retrieve("预算 交通 住宿", intent="budget", top_k=5)
    ids = [h.doc_id for h in hits]
    assert len(ids) == len(set(ids))


async def test_retrieve_respects_top_k():
    hits = await _kb().retrieve("签证 预算 交通 住宿", top_k=1)
    assert len(hits) == 1


def test_from_json_missing_file_returns_empty(tmp_path):
    kb = KnowledgeBase.from_json(tmp_path / "not_exist.json")
    assert kb.size == 0
