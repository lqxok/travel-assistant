"""意图识别测试。"""

from __future__ import annotations

from app.intents import Intent, IntentRecognizer, RuleEngine


def test_rule_engine_matches_guide():
    result = RuleEngine().classify("去日本旅游要准备什么签证材料")
    assert result is not None
    assert result[0] is Intent.GUIDE


def test_rule_engine_returns_none_when_no_match():
    assert RuleEngine().classify("你好呀") is None


async def test_recognizer_uses_rule_without_calling_llm(fake_llm_factory):
    llm = fake_llm_factory()
    result = await IntentRecognizer(llm).recognize("帮我规划成都三天行程")

    assert result.intent is Intent.TRIP_PLANNING
    assert result.source == "rule"
    assert llm.calls == []  # 快车道不花模型


async def test_recognizer_falls_back_to_llm(fake_llm_factory):
    llm = fake_llm_factory(['{"intent": "general", "confidence": 0.9}'])
    result = await IntentRecognizer(llm).recognize("你好呀")

    assert result.intent is Intent.GENERAL
    assert result.source == "llm"
    assert len(llm.calls) == 1


async def test_recognizer_degrades_safely_on_bad_json(fake_llm_factory):
    llm = fake_llm_factory(["这不是 JSON"])
    result = await IntentRecognizer(llm).recognize("你好呀")

    assert result.intent is Intent.GENERAL
    assert result.confidence == 0.3
