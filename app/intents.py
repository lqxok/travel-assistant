"""意图识别：规则快车道 + LLM 慢车道。

- 规则快车道：正则匹配，零成本、低延迟，覆盖高频固定说法；
- LLM 慢车道：规则拿不准（无命中或置信度低）时才调用模型，要求输出结构化 JSON；
- 融合：两路一致时抬升置信度，冲突时取置信更高的一路。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from app.llm import LLMClient, Message


class Intent(str, Enum):
    """旅行意图。"""

    TRIP_PLANNING = "trip_planning"  # 行程规划
    BUDGET = "budget"  # 预算估算
    GUIDE = "guide"  # 签证 / 攻略 / 注意事项
    TRANSPORT = "transport"  # 交通（机票 / 高铁 / 自驾）
    STAY = "stay"  # 住宿
    GENERAL = "general"  # 通用 / 其它


@dataclass(frozen=True)
class _Rule:
    intent: Intent
    priority: int
    patterns: tuple[re.Pattern[str], ...]


_RULES: tuple[_Rule, ...] = (
    _Rule(
        Intent.GUIDE,
        90,
        (
            re.compile(r"(签证|护照|入境|免签|落地签|保险)"),
            re.compile(r"(攻略|注意事项|必备|准备什么|安全注意)"),
        ),
    ),
    _Rule(
        Intent.BUDGET,
        80,
        (
            re.compile(r"(预算|花费|费用|开销|人均)"),
            re.compile(r"(多少钱|贵不贵|报价|要多少)"),
        ),
    ),
    _Rule(
        Intent.TRIP_PLANNING,
        70,
        (
            re.compile(r"(规划|安排|计划|路线)"),
            re.compile(r"(几天|怎么玩|玩几天|行程)"),
        ),
    ),
    _Rule(
        Intent.TRANSPORT,
        65,
        (
            re.compile(r"(机票|航班|高铁|火车|动车|大巴|租车|自驾)"),
            re.compile(r"(怎么去|怎么走|交通)"),
        ),
    ),
    _Rule(
        Intent.STAY,
        60,
        (
            re.compile(r"(酒店|住宿|民宿|青旅|住哪|订房)"),
        ),
    ),
)

# 规则置信度 >= 该阈值时直接用快车道结果，不再调用模型。
_SLOW_LANE_THRESHOLD = 0.8


@dataclass(slots=True)
class IntentResult:
    intent: Intent
    confidence: float
    source: str  # rule / llm / hybrid


class RuleEngine:
    """基于正则的快速分类器。"""

    def __init__(self, rules: tuple[_Rule, ...] = _RULES) -> None:
        self._rules = rules

    def classify(self, text: str) -> tuple[Intent, float] | None:
        text = text.strip()
        if not text:
            return None
        scores: dict[Intent, int] = {}
        for rule in self._rules:
            hits = sum(1 for p in rule.patterns if p.search(text))
            if hits:
                scores[rule.intent] = scores.get(rule.intent, 0) + hits
        if not scores:
            return None

        priority = {r.intent: r.priority for r in self._rules}
        best = max(scores, key=lambda i: (scores[i], priority.get(i, 0)))
        confidence = min(1.0, 0.8 + 0.1 * (scores[best] - 1))
        return best, confidence


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _extract_json(raw: str) -> dict[str, object]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(raw)
    if not match:
        raise json.JSONDecodeError("no json object", raw, 0)
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise TypeError("expected json object")
    return parsed


class IntentRecognizer:
    """快慢车道融合的意图识别器。"""

    def __init__(self, llm: LLMClient, *, slow_lane_threshold: float = _SLOW_LANE_THRESHOLD) -> None:
        self._rules = RuleEngine()
        self._llm = llm
        self._threshold = slow_lane_threshold

    async def recognize(self, text: str) -> IntentResult:
        fast = self._rules.classify(text)
        if fast is not None and fast[1] >= self._threshold:
            return IntentResult(fast[0], fast[1], "rule")

        slow = await self._classify_with_llm(text)
        if fast is None:
            return IntentResult(slow[0], slow[1], "llm")
        if fast[0] is slow[0]:
            return IntentResult(fast[0], min(1.0, (fast[1] + slow[1]) / 2 + 0.05), "hybrid")
        if fast[1] >= slow[1]:
            return IntentResult(fast[0], fast[1], "hybrid")
        return IntentResult(slow[0], slow[1], "hybrid")

    async def _classify_with_llm(self, text: str) -> tuple[Intent, float]:
        allowed = ", ".join(i.value for i in Intent)
        system = (
            "你是旅行助手的意图分类器。请从给定意图中选一个最匹配的，"
            "只输出一个 JSON 对象，字段：intent (字符串), confidence (0-1 小数)。"
            f"可选意图：{allowed}。不要输出任何其它文字。"
        )
        try:
            raw = await self._llm.complete(
                [Message(role="user", content=text)],
                system=system,
                temperature=0.0,
                max_tokens=256,
            )
            data = _extract_json(raw)
            intent = Intent(str(data.get("intent", "general")))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            return intent, confidence
        except Exception:  # noqa: BLE001 —— 慢车道失败不应影响主流程
            return Intent.GENERAL, 0.3
