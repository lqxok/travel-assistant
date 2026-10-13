"""编排层测试：检索分支 / 工具分支 / 缺参数追问。"""

from __future__ import annotations

from app.agent import TravelAgent
from app.intents import Intent
from app.profile import UserProfileStore
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


async def test_multi_turn_includes_history(fake_llm_factory):
    llm = fake_llm_factory(
        [
            '{"origin": "北京", "destination": "成都", "days": 3}',
            "好的，3 天行程如下。",
            '{"origin": "北京", "destination": "成都", "days": 5}',
            "好的，5 天行程如下。",
        ]
    )
    agent = TravelAgent(llm, _kb())
    await agent.handle("帮我规划从北京去成都的三天行程", session_id="s1")
    await agent.handle("帮我规划从北京去成都的五天行程", session_id="s1")

    joined = " ".join(m.content for m in llm.calls[-1]["messages"])
    assert "三天" in joined  # 第二轮带上了第一轮的历史


async def test_without_session_id_no_history(fake_llm_factory):
    llm = fake_llm_factory(
        [
            '{"origin": "北京", "destination": "成都", "days": 3}',
            "好的。",
            '{"origin": "北京", "destination": "成都", "days": 5}',
            "好的。",
        ]
    )
    agent = TravelAgent(llm, _kb())
    await agent.handle("帮我规划从北京去成都的三天行程")
    await agent.handle("帮我规划从北京去成都的五天行程")

    joined = " ".join(m.content for m in llm.calls[-1]["messages"])
    assert "三天" not in joined  # 没传 session_id → 不带历史


async def test_astream_records_memory(fake_llm_factory):
    llm = fake_llm_factory(["你好呀", "第二次回答"])
    agent = TravelAgent(llm, _kb())

    async for _ in agent.astream("你好", session_id="s9"):
        pass

    history = agent._memory.history("s9")
    assert len(history) == 2
    assert history[0].content == "你好"
    assert history[1].role == "assistant"


async def test_user_profile_injected_into_system(fake_llm_factory, tmp_path):
    store = UserProfileStore(tmp_path / "p.json")
    store.add_facts("u1", ["常驻北京"])
    llm = fake_llm_factory(["好的。"])
    agent = TravelAgent(llm, _kb(), profile=store)

    await agent.handle("推荐住哪里", user_id="u1")

    assert "常驻北京" in (llm.calls[0]["system"] or "")


async def test_extracts_and_stores_facts(fake_llm_factory, tmp_path):
    store = UserProfileStore(tmp_path / "p.json")
    # 第一次调用是抽取事实，第二次是生成回答
    llm = fake_llm_factory(['{"facts": ["常驻北京", "不吃辣"]}', "好的。"])
    agent = TravelAgent(llm, _kb(), profile=store)

    await agent.handle("我常驻北京，也不吃辣，帮我推荐住哪里", user_id="u1")

    assert "常驻北京" in store.facts("u1")
    assert "不吃辣" in store.facts("u1")


async def test_no_user_id_no_profile_write(fake_llm_factory, tmp_path):
    store = UserProfileStore(tmp_path / "p.json")
    llm = fake_llm_factory(["好的。"])
    agent = TravelAgent(llm, _kb(), profile=store)

    await agent.handle("我常驻北京，帮我推荐住哪里")

    assert store.user_count == 0


async def test_profile_facts_used_for_tool_params(fake_llm_factory, tmp_path):
    store = UserProfileStore(tmp_path / "p.json")
    store.add_facts("u1", ["常驻北京"])
    llm = fake_llm_factory(['{"destination": "三亚", "days": 3}', "好的。"])
    agent = TravelAgent(llm, _kb(), profile=store)

    await agent.handle("帮我规划去三亚的三天行程", user_id="u1")

    # 抽取参数那次调用应带上长期信息（便于补全出发地）
    assert "常驻北京" in (llm.calls[0]["system"] or "")
