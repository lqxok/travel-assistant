"""编排层测试：检索分支 / 工具分支 / 缺参数追问。"""

from __future__ import annotations

from app.agent import TravelAgent
from app.intents import Intent
from app.rag import Doc, KnowledgeBase


def _kb() -> KnowledgeBase:
    return KnowledgeBase(
        [Doc("d1", "签证与证件", "护照有效期至少 6 个月，需要行程单与酒店订单。", "guide")]
    )


async def test_rag_branch_returns_citations(fake_llm_factory):
    llm = fake_llm_factory(["根据资料，护照有效期应不少于 6 个月。"])
    reply = await TravelAgent(llm, _kb()).handle("去日本旅游要准备什么签证材料")

    assert reply.intent is Intent.GUIDE
    assert reply.citations and reply.citations[0]["doc_id"] == "d1"
    # 生成时应带上检索到的参考资料
    assert "参考资料" in (llm.calls[0]["system"] or "")


async def test_tool_branch_invokes_tool(fake_llm_factory):
    llm = fake_llm_factory(
        [
            '{"origin": "北京", "destination": "成都", "days": 3}',
            "从北京去成都三天，大约 2700 元。",
        ]
    )
    reply = await TravelAgent(llm, _kb()).handle("帮我规划从北京去成都的三天行程")

    assert reply.tool_result is not None
    assert reply.tool_result["itinerary"]["destination"] == "成都"
    assert reply.reply


async def test_missing_param_asks_user(fake_llm_factory):
    # 抽取结果缺少 origin（用户没说出发地）
    llm = fake_llm_factory(['{"destination": "成都", "days": 3}'])
    reply = await TravelAgent(llm, _kb()).handle("帮我规划去成都的三天行程")

    assert reply.tool_result is None
    assert "出发城市" in reply.reply


async def test_budget_branch(fake_llm_factory):
    llm = fake_llm_factory(['{"days": 5, "people": 2}', "两个人 5 天大约 9000 元。"])
    reply = await TravelAgent(llm, _kb()).handle("两个人去日本玩5天大概多少钱")

    assert reply.intent is Intent.BUDGET
    assert reply.tool_result is not None
    assert reply.tool_result["total_cny"] == 9000


async def test_astream_emits_status_delta_done(fake_llm_factory):
    llm = fake_llm_factory(["根据资料，护照有效期应不少于 6 个月。"])
    events = [e async for e in TravelAgent(llm, _kb()).astream("去日本旅游要准备什么签证材料")]

    assert events[0]["type"] == "status"
    assert events[-1]["type"] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "护照" in text
    assert events[-1]["citations"][0]["doc_id"] == "d1"


async def test_astream_fixed_reply_when_param_missing(fake_llm_factory):
    llm = fake_llm_factory(['{"destination": "成都", "days": 3}'])
    events = [e async for e in TravelAgent(llm, _kb()).astream("帮我规划去成都的三天行程")]

    assert events[-1]["type"] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "出发城市" in text
