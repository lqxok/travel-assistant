"""编排：把意图识别、本地检索、工具串成一条链路。

流程：
  用户输入 → 意图识别
    ├ budget / trip_planning → 抽参数 → 调工具 → 让模型润色成自然语言
    └ 其它（guide / transport / stay / general） → 本地检索 → 有据生成（回传引用来源）

多轮：传入相同的 `session_id` 时，会把该会话的历史对话一并带给模型。
对外提供两种用法：
  - `handle()`  一次性返回完整回答
  - `astream()` 逐块产出（status / delta / done），供 SSE 前端使用
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.intents import Intent, IntentRecognizer, IntentResult
from app.json_utils import extract_json
from app.llm import LLMClient, Message
from app.memory import ConversationMemory
from app.profile import UserProfileStore, worth_extracting
from app.rag import Hit, KnowledgeBase
from app.tools import TOOLS, invoke

BASE_SYSTEM = (
    "你是「旅行助手」，帮助用户规划行程、估算预算、解答签证与出行问题。"
    "回答要具体、可执行；不确定或资料没有的信息，不要编造。"
)
RAG_SYSTEM = (
    "你是「旅行助手」。只根据下面提供的参考资料回答用户问题；"
    "资料不足就直说，不要编造。回答要具体、可执行。"
)
RENDER_SYSTEM = (
    "你是「旅行助手」。下面是一次工具调用的结果，请用自然、口语化的中文整理成对用户有用的回答，"
    "保留关键数字，不要添加结果里没有的信息。"
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
    tool_name: str | None = None


@dataclass(slots=True)
class _Prepared:
    """一次回答的"准备结果"：要么给模型一个 prompt 去生成，要么是一个固定回复。"""

    intent: Intent
    confidence: float
    source: str
    stage: str = "生成回答"
    system: str = ""
    user: str = ""
    history: list[Message] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_result: dict[str, Any] | None = None
    tool_name: str | None = None
    fixed_reply: str | None = None


class TravelAgent:
    """旅行助手编排器。"""

    def __init__(
        self,
        llm: LLMClient,
        knowledge_base: KnowledgeBase,
        memory: ConversationMemory | None = None,
        profile: UserProfileStore | None = None,
    ) -> None:
        self._llm = llm
        self._kb = knowledge_base
        self._memory = memory or ConversationMemory()
        self._profile = profile or UserProfileStore()
        self._intents = IntentRecognizer(llm)

    # ---------- 对外：非流式 ----------
    async def handle(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AgentReply:
        await self._update_profile(user_id, message)
        facts = self._profile.facts(user_id) if user_id else []
        prepared = await self._prepare(message, self._history(session_id), facts)
        if prepared.fixed_reply is not None:
            reply = prepared.fixed_reply
        else:
            reply = await self._llm.complete(
                [*prepared.history, Message(role="user", content=prepared.user)],
                system=prepared.system,
            )
        self._remember(session_id, message, reply)
        return AgentReply(
            reply,
            prepared.intent,
            prepared.confidence,
            prepared.source,
            citations=prepared.citations,
            tool_result=prepared.tool_result,
            tool_name=prepared.tool_name,
        )

    # ---------- 对外：流式 ----------
    async def astream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        await self._update_profile(user_id, message)
        facts = self._profile.facts(user_id) if user_id else []
        prepared = await self._prepare(message, self._history(session_id), facts)
        yield {"type": "status", "text": prepared.stage}

        if prepared.fixed_reply is not None:
            full = prepared.fixed_reply
            yield {"type": "delta", "text": full}
        else:
            parts: list[str] = []
            async for piece in self._llm.stream(
                [*prepared.history, Message(role="user", content=prepared.user)],
                system=prepared.system,
            ):
                parts.append(piece)
                yield {"type": "delta", "text": piece}
            full = "".join(parts)

        self._remember(session_id, message, full)
        yield {
            "type": "done",
            "intent": prepared.intent.value,
            "confidence": round(prepared.confidence, 3),
            "source": prepared.source,
            "citations": prepared.citations,
            "tool_result": prepared.tool_result,
            "tool_name": prepared.tool_name,
        }

    # ---------- 内部 ----------
    def _history(self, session_id: str | None) -> list[Message]:
        return self._memory.history(session_id) if session_id else []

    def _remember(self, session_id: str | None, user_text: str, assistant_text: str) -> None:
        if session_id and assistant_text:
            self._memory.append(session_id, user_text, assistant_text)

    async def _prepare(self, message: str, history: list[Message], facts: list[str]) -> _Prepared:
        ir = await self._intents.recognize(message)
        tool_name = _TOOL_FOR_INTENT.get(ir.intent)
        if tool_name is not None:
            prepared = await self._prepare_tool(message, ir, tool_name, history, facts)
        else:
            prepared = await self._prepare_rag(message, ir, history)

        # 把用户长期信息注入系统提示（跨会话生效）
        if facts and prepared.fixed_reply is None:
            joined = "；".join(facts)
            prepared.system = (
                f"{prepared.system}\n\n已知用户长期信息（请自然参考，不要生硬复述）：{joined}"
            )
        return prepared

    async def _update_profile(self, user_id: str | None, message: str) -> None:
        """（必要时）从用户这句话里抽取长期事实并记录。"""
        if not user_id or not worth_extracting(message):
            return
        facts = await self._extract_facts(message)
        if facts:
            self._profile.add_facts(user_id, facts)

    async def _extract_facts(self, message: str) -> list[str]:
        system = (
            "从用户的话里抽取值得长期记住的稳定偏好或事实（例如常驻城市、饮食忌口、"
            "出行习惯、同行情况）。只输出 JSON：{\"facts\": [\"...\"]}；没有则返回 "
            "{\"facts\": []}。每条不超过 20 字，最多 3 条。不要输出其它文字。"
        )
        try:
            raw = await self._llm.complete(
                [Message(role="user", content=message)],
                system=system,
                temperature=0.0,
                max_tokens=200,
            )
            data = extract_json(raw)
            raw_facts = data.get("facts") or []
            return [str(f) for f in raw_facts if str(f).strip()][:3]
        except Exception:  # noqa: BLE001
            return []

    async def _prepare_tool(
        self,
        message: str,
        ir: IntentResult,
        tool_name: str,
        history: list[Message],
        facts: list[str],
    ) -> _Prepared:
        spec = TOOLS[tool_name]
        params = await self._extract_params(
            message, spec.name, spec.params_model.model_json_schema(), history, facts
        )
        if params is None:
            return _Prepared(
                ir.intent, ir.confidence, ir.source, system=BASE_SYSTEM, user=message, history=history
            )

        try:
            result = invoke(tool_name, **params)
        except ValidationError as exc:
            # 用户没说全（如缺出发城市）→ 追问，而不是硬答
            missing = [
                _FIELD_LABEL.get(str(err["loc"][0]), str(err["loc"][0]))
                for err in exc.errors()
                if err.get("type") == "missing"
            ]
            return _Prepared(
                ir.intent, ir.confidence, ir.source, fixed_reply=self._ask_for_missing(missing)
            )
        except Exception:  # noqa: BLE001 —— 其它异常退化为直接回答
            return _Prepared(
                ir.intent, ir.confidence, ir.source, system=BASE_SYSTEM, user=message, history=history
            )

        user = (
            f"用户问题：{message}\n工具：{tool_name}\n"
            f"结果(JSON)：{json.dumps(result, ensure_ascii=False)}"
        )
        return _Prepared(
            ir.intent,
            ir.confidence,
            ir.source,
            stage="整理结果",
            system=RENDER_SYSTEM,
            user=user,
            history=history,
            tool_result=result,
            tool_name=tool_name,
        )

    async def _prepare_rag(
        self, message: str, ir: IntentResult, history: list[Message]
    ) -> _Prepared:
        hits: list[Hit] = await self._kb.retrieve(message, intent=ir.intent.value, top_k=3)
        citations = [
            {"doc_id": h.doc_id, "title": h.title, "score": round(h.score, 3)} for h in hits
        ]
        if not hits:
            return _Prepared(
                ir.intent,
                ir.confidence,
                ir.source,
                system=BASE_SYSTEM,
                user=message,
                history=history,
                citations=citations,
            )
        context = "\n\n".join(f"[{i}] {h.title}\n{h.text}" for i, h in enumerate(hits, 1))
        return _Prepared(
            ir.intent,
            ir.confidence,
            ir.source,
            stage="检索资料",
            system=RAG_SYSTEM,
            user=f"参考资料：\n{context}\n\n用户问题：{message}",
            history=history,
            citations=citations,
        )

    async def _extract_params(
        self,
        message: str,
        tool_name: str,
        schema: dict[str, Any],
        history: list[Message],
        facts: list[str],
    ) -> dict[str, Any] | None:
        """让模型从自然语言里抽出工具所需参数。"""
        system = (
            f"请从用户的话里抽取调用工具 `{tool_name}` 所需的参数。"
            f"参数 JSON Schema：{json.dumps(schema, ensure_ascii=False)}。"
            "只输出一个 JSON 对象，字段与 schema 一致；无法确定的字段省略。不要输出其它文字。"
        )
        if facts:
            system += f"\n已知用户长期信息（可用于补全参数）：{'；'.join(facts)}"
        try:
            raw = await self._llm.complete(
                [*history, Message(role="user", content=message)],
                system=system,
                temperature=0.0,
                max_tokens=300,
            )
            data = extract_json(raw)
        except Exception:  # noqa: BLE001
            return None
        return data or None

    # ---------- 用户长期记忆 ----------
    def clear_profile(self, user_id: str) -> None:
        self._profile.clear(user_id)

    @property
    def profile_user_count(self) -> int:
        return self._profile.user_count

    @staticmethod
    def _ask_for_missing(missing: list[str]) -> str:
        """缺参数时向用户追问。"""
        if not missing:
            return "要算得更准，我还需要一点信息，麻烦补充一下出发地、天数或人数。"
        return "为了帮你算得更准，我还需要这些信息：" + "、".join(missing) + "。"
