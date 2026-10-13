"""评测指标与离线评测测试（不需要真实模型）。"""

from __future__ import annotations

from app.rag import KnowledgeBase
from evals.metrics import mean, percentile, rate, recall_at_k
from evals.run_evals import load_cases, run_offline


def test_rate_handles_zero_denominator():
    assert rate(1, 2) == 0.5
    assert rate(0, 0) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], "b", 2)
    assert not recall_at_k(["a", "b"], "b", 1)


def test_mean_and_percentile():
    assert mean([1, 3]) == 2.0
    assert mean([]) == 0.0
    assert 2 <= percentile([1, 2, 3, 4], 50) <= 3
    assert percentile([], 95) == 0.0


def test_cases_are_valid():
    cases = load_cases()
    assert len(cases) >= 20
    types = {c["type"] for c in cases}
    assert {"intent", "retrieval", "tool", "grounded"} <= types
    for case in cases:
        assert "input" in case and "id" in case


async def test_offline_eval_runs_and_is_sane():
    result = await run_offline(load_cases(), KnowledgeBase.from_json())

    assert result["intent_rule"]["coverage"] > 0.5
    assert 0.0 <= result["intent_rule"]["precision_on_covered"] <= 1.0
    assert result["retrieval"]["recall@3"] >= result["retrieval"]["recall@1"] > 0.0
