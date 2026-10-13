"""编排：把意图识别、本地检索、工具串成一条链路。

流程：
  用户输入 → 意图识别
    ├ budget / trip_planning → 抽参数 → 调工具 → 让模型润色成自然语言
    └ 其它（guide / transport / stay / general） → 本地检索 → 有据生成（回传引用来源）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.intents import Intent, IntentRecognizer, IntentResult
from app.json_utils import extract_json
from app.llm import LLMClient, Message
from app.rag import Hit, KnowledgeBase
from app.tools import TOOLS, invoke

BASE_SYSTEM = (
    "你是「旅行助手」，帮助用户规划行程、估算预算、解答签证与出行问题。"
    "回答要具体、可执行；不确定或资料没有的信息，不要编造。"
)

# 需要调用工具的意图 → 对应工具名
_TOOL_FOR_INTENT: dict[Intent, str] = {
    Intent.BUDGET: "estimate_budget",
    Intent.TRIP_PLANNING: "plan_trip",
}

# 工具参数字段 → 中文说明（用于缺参数时追问用户）
_FIELD_LABEL: dict[str, str] = {
    "origin": "出发城市",
    "destination": "目的地城市",
    "days": "行程天数",
    "people": "出行人数",
    "style": "出行档次（穷游 / 舒适 / 高端）",
    "start_date": "出发日期",
}


@dataclass(slots=True)
class AgentReply:
    reply: str
    intent: Intent
    confidence: float
    source: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_result: dict[str, Any] | None = None


class TravelAgent:
    """旅行助手编排器。"""

    def __init__(self, llm: LLMClient, knowledge_base: KnowledgeBase) -> None:
        self._llm = llm
        self._kb = knowledge_base
        self._intents = IntentRecognizer(llm)

    async def handle(self, message: str) -> AgentReply:
        ir = await self._intents.recognize(message)
        tool_name = _TOOL_FOR_INTENT.get(ir.intent)
        if tool_name is not None:
            return await self._handle_with_tool(message, ir, tool_name)
        return await self._handle_with_rag(message, ir)

    # ---------- 工具分支 ----------
    async def _handle_with_tool(self, message: str, ir: IntentResult, tool_name: str) -> AgentReply:
        spec = TOOLS[tool_name]
        params = await self._extract_params(message, spec.name, spec.params_model.model_json_schema())
        if params is None:
            return await self._direct_reply(message, ir)

        try:
            result = invoke(tool_name, **params)
        except ValidationError as exc:
            # 用户没说全（如缺出发城市）→ 追问，而不是硬答
            missing = [
                _FIELD_LABEL.get(str(err["loc"][0]), str(err["loc"][0]))
                for err in exc.errors()
                if err.get("type") == "missing"
            ]
            return AgentReply(self._ask_for_missing(missing), ir.intent, ir.confidence, ir.source)
        except Exception:  # noqa: BLE001 —— 其它异常退化为直接回答
            return await self._direct_reply(message, ir)

        reply = await self._render_tool_result(message, tool_name, result)
        return AgentReply(reply, ir.intent, ir.confidence, ir.source, tool_result=result)

    async def _extract_params(
        self, message: str, tool_name: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """让模型从自然语言里抽出工具所需参数。"""
        system = (
            f"请从用户的话里抽取调用工具 `{tool_name}` 所需的参数。"
            f"参数 JSON Schema：{json.dumps(schema, ensure_ascii=False)}。"
            "只输出一个 JSON 对象，字段与 schema 一致；无法确定的字段省略。不要输出其它文字。"
        )
        try:
            raw = await self._llm.complete(
                [Message(role="user", content=message)],
                system=system,
                temperature=0.0,
                max_tokens=300,
            )
            data = extract_json(raw)
        except Exception:  # noqa: BLE001
            return None
        return data or None

    async def _render_tool_result(
        self, message: str, tool_name: str, result: dict[str, Any]
    ) -> str:
        """把工具结果润色成自然语言，不额外编造。"""
        system = (
            "你是「旅行助手」。下面是一次工具调用的结果，请用自然、口语化的中文整理成对用户有用的回答，"
            "保留关键数字，不要添加结果里没有的信息。"
        )
        user = f"用户问题：{message}\n工具：{tool_name}\n结果(JSON)：{json.dumps(result, ensure_ascii=False)}"
        return await self._llm.complete([Message(role="user", content=user)], system=system)

    # ---------- 检索分支 ----------
    async def _handle_with_rag(self, message: str, ir: IntentResult) -> AgentReply:
        hits: list[Hit] = await self._kb.retrieve(message, intent=ir.intent.value, top_k=3)
        if hits:
            context = "\n\n".join(f"[{i}] {h.title}\n{h.text}" for i, h in enumerate(hits, 1))
            system = (
                "你是「旅行助手」。只根据下面提供的参考资料回答用户问题；"
                "资料不足就直说，不要编造。回答要具体、可执行。"
            )
            user = f"参考资料：\n{context}\n\n用户问题：{message}"
        else:
            system = BASE_SYSTEM
            user = message

        reply = await self._llm.complete([Message(role="user", content=user)], system=system)
        citations = [
            {"doc_id": h.doc_id, "title": h.title, "score": round(h.score, 3)} for h in hits
        ]
        return AgentReply(reply, ir.intent, ir.confidence, ir.source, citations=citations)

    # ---------- 兜底 ----------
    @staticmethod
    def _ask_for_missing(missing: list[str]) -> str:
        """缺参数时向用户追问。"""
        if not missing:
            return "要算得更准，我还需要一点信息，麻烦补充一下出发地、天数或人数。"
        return "为了帮你算得更准，我还需要这些信息：" + "、".join(missing) + "。"

    async def _direct_reply(self, message: str, ir: IntentResult) -> AgentReply:
        reply = await self._llm.complete([Message(role="user", content=message)], system=BASE_SYSTEM)
        return AgentReply(reply, ir.intent, ir.confidence, ir.source)
