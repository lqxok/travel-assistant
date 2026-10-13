"""评测入口。

用法：
    python -m evals.run_evals            # 离线指标 + 在线指标（需要 API Key）
    python -m evals.run_evals --offline  # 只跑不需要大模型的指标
    python -m evals.run_evals --limit 3  # 在线部分每组只跑 3 条（省钱）

结果写入 `evals/report.json`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent import TravelAgent
from app.config import load_settings
from app.embedder import build_embedder, build_reranker
from app.intents import IntentRecognizer, RuleEngine
from app.json_utils import extract_json
from app.llm import LLMClient, Message
from app.rag import KnowledgeBase
from evals.metrics import mean, percentile, rate, recall_at_k

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"
REPORT_PATH = Path(__file__).resolve().parent / "report.json"
TOP_K = 3


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _of_type(cases: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [c for c in cases if c.get("type") == kind]


# ---------------- 离线指标（不调用大模型） ----------------
async def run_offline(cases: list[dict[str, Any]], kb: KnowledgeBase) -> dict[str, Any]:
    rules = RuleEngine()

    intent_cases = _of_type(cases, "intent")
    covered = correct = 0
    for case in intent_cases:
        got = rules.classify(case["input"])
        if got is not None:
            covered += 1
            correct += int(got[0].value == case["expected"])

    ret_cases = _of_type(cases, "retrieval")
    hit1 = hit3 = 0
    for case in ret_cases:
        hits = await kb.retrieve(case["input"], top_k=TOP_K)
        ids = [h.doc_id for h in hits]
        hit1 += int(recall_at_k(ids, case["expected_doc"], 1))
        hit3 += int(recall_at_k(ids, case["expected_doc"], TOP_K))

    return {
        "intent_rule": {
            "cases": len(intent_cases),
            "covered": covered,
            "coverage": rate(covered, len(intent_cases)),
            "precision_on_covered": rate(correct, covered),
        },
        "retrieval": {
            "cases": len(ret_cases),
            "recall@1": rate(hit1, len(ret_cases)),
            "recall@3": rate(hit3, len(ret_cases)),
        },
    }


# ---------------- 在线指标（需要大模型） ----------------
async def judge_groundedness(
    llm: LLMClient, question: str, answer: str, context: str
) -> float | None:
    """用模型给"回答是否有据"打分（0~1）；判官失败返回 None（不计入均值）。"""
    system = (
        "你是严格的评审。判断【回答】里的事实性陈述是否都能被【参考资料】支持，"
        "以及资料不足时回答是否明确说明。只输出 JSON：{\"score\": 0~1, \"reason\": \"...\"}"
        "（reason 不超过 20 字）。"
    )
    user = f"【用户问题】{question}\n\n【参考资料】\n{context or '（无）'}\n\n【回答】\n{answer}"
    try:
        # 不限制 max_tokens：推理型模型思考会占用较多预算，给太小会得到空回复
        raw = await llm.complete(
            [Message(role="user", content=user)], system=system, temperature=0.0
        )
        return max(0.0, min(1.0, float(extract_json(raw).get("score", 0.0))))
    except Exception:  # noqa: BLE001 —— 判官失败不应污染分数
        return None


async def run_online(
    cases: list[dict[str, Any]],
    llm: LLMClient,
    agent: TravelAgent,
    kb: KnowledgeBase,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    def cap(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return items[:limit] if limit else items

    # 1) 整体意图准确率（含慢车道）
    recognizer = IntentRecognizer(llm)
    intent_cases = cap(_of_type(cases, "intent"))
    correct = 0
    for case in intent_cases:
        result = await recognizer.recognize(case["input"])
        correct += int(result.intent.value == case["expected"])

    # 2) 工具调用准确率 + 延迟
    tool_cases = cap(_of_type(cases, "tool"))
    tool_ok = 0
    latencies: list[float] = []
    for case in tool_cases:
        start = time.perf_counter()
        reply = await agent.handle(case["input"])
        latencies.append(time.perf_counter() - start)
        tool_ok += int(reply.tool_name == case["expected_tool"])

    # 3) 回答忠实度（LLM 判官）
    grounded_cases = cap(_of_type(cases, "grounded"))
    raw_scores: list[float | None] = []
    for case in grounded_cases:
        start = time.perf_counter()
        reply = await agent.handle(case["input"])
        latencies.append(time.perf_counter() - start)
        hits = await kb.retrieve(case["input"], intent=reply.intent.value, top_k=TOP_K)
        context = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, 1))
        raw_scores.append(await judge_groundedness(llm, case["input"], reply.reply, context))

    scored = [s for s in raw_scores if s is not None]
    return {
        "intent_accuracy": rate(correct, len(intent_cases)),
        "tool_accuracy": rate(tool_ok, len(tool_cases)),
        "groundedness_mean": mean(scored),
        "groundedness_scored": len(scored),
        "groundedness_total": len(raw_scores),
        "latency_s": {
            "n": len(latencies),
            "mean": mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
    }


# ---------------- 输出 ----------------
def print_report(report: dict[str, Any]) -> None:
    off = report["offline"]
    print("=" * 44)
    print(f"评测模式：{report['mode']}    模型：{report.get('model')}")
    print("-" * 44)
    ir = off["intent_rule"]
    print(f"[离线] 规则意图：覆盖 {ir['covered']}/{ir['cases']}（{ir['coverage']}），"
          f"覆盖内准确率 {ir['precision_on_covered']}")
    rt = off["retrieval"]
    print(f"[离线] 检索：Recall@1 = {rt['recall@1']}，Recall@3 = {rt['recall@3']}（{rt['cases']} 条）")
    if "online" in report:
        on = report["online"]
        print(f"[在线] 整体意图准确率：{on['intent_accuracy']}")
        print(f"[在线] 工具调用准确率：{on['tool_accuracy']}")
        print(f"[在线] 回答忠实度：{on['groundedness_mean']}"
              f"（{on['groundedness_scored']}/{on['groundedness_total']} 条成功评分）")
        lat = on["latency_s"]
        print(f"[在线] 延迟：mean {lat['mean']}s / p50 {lat['p50']}s / p95 {lat['p95']}s（{lat['n']} 次）")
    print("=" * 44)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="只跑离线指标")
    parser.add_argument("--limit", type=int, default=None, help="在线部分每组只跑 N 条")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="检索评测启用本地语义通道 + 重排（需已下载模型）",
    )
    args = parser.parse_args()

    settings = load_settings()
    embedder = build_embedder(settings.embed_model) if args.semantic else None
    reranker = build_reranker(settings.rerank_model) if args.semantic else None
    kb = KnowledgeBase.from_json(embedder=embedder, reranker=reranker)
    kb.warmup()
    cases = load_cases()

    offline_only = args.offline or not settings.has_api_key
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline" if offline_only else "full",
        "model": settings.model,
        "retrieval_channels": {
            "keyword": True,
            "intent": True,
            "semantic": kb.has_vector,
            "rerank": kb.has_reranker,
        },
        "offline": await run_offline(cases, kb),
    }
    if offline_only and not args.offline:
        print("（未检测到 API Key，自动只跑离线指标）")

    if not offline_only:
        llm = LLMClient(settings)
        agent = TravelAgent(llm, kb)
        report["online"] = await run_online(cases, llm, agent, kb, limit=args.limit)

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"报告已写入：{REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
