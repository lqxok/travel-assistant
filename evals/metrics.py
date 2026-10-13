"""评测指标：纯函数，便于单独测试。"""

from __future__ import annotations

from typing import Sequence


def rate(numerator: int, denominator: int) -> float:
    """比例；分母为 0 时返回 0.0。"""
    return round(numerator / denominator, 4) if denominator else 0.0


def recall_at_k(ranked_ids: Sequence[str], expected: str, k: int) -> bool:
    """期望文档是否出现在前 k 个结果里。"""
    return expected in list(ranked_ids)[:k]


def mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """简单百分位（p 取 0-100）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 3)
