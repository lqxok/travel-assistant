"""旅行工具：行程草稿与预算估算。

工具是纯函数式的（无外部 IO），参数用 Pydantic 校验，返回结构化结果；
`TOOLS` 注册表供编排层按名字调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class TravelStyle(str, Enum):
    BUDGET = "budget"  # 穷游
    COMFORT = "comfort"  # 舒适
    LUXURY = "luxury"  # 高端


# 人均每天的粗略花费（人民币）
_PER_DAY_CNY: dict[TravelStyle, int] = {
    TravelStyle.BUDGET: 500,
    TravelStyle.COMFORT: 900,
    TravelStyle.LUXURY: 2000,
}

# 花费大致的构成比例
_RATIO: dict[TravelStyle, dict[str, float]] = {
    TravelStyle.BUDGET: {"transport": 0.35, "stay": 0.30, "food": 0.20, "tickets": 0.15},
    TravelStyle.COMFORT: {"transport": 0.40, "stay": 0.35, "food": 0.15, "tickets": 0.10},
    TravelStyle.LUXURY: {"transport": 0.45, "stay": 0.40, "food": 0.10, "tickets": 0.05},
}


class BudgetQuery(BaseModel):
    """预算估算参数。"""

    days: int = Field(..., ge=1, le=60, description="行程天数")
    people: int = Field(1, ge=1, le=9, description="出行人数")
    style: TravelStyle = Field(TravelStyle.COMFORT, description="出行档次")


class TripPlanQuery(BaseModel):
    """行程草稿参数。"""

    origin: str = Field(..., min_length=1, description="出发城市")
    destination: str = Field(..., min_length=1, description="目的地城市")
    days: int = Field(..., ge=1, le=30, description="行程天数")
    people: int = Field(1, ge=1, le=9, description="出行人数")
    style: TravelStyle = Field(TravelStyle.COMFORT, description="出行档次")
    start_date: str | None = Field(None, description="出发日期 YYYY-MM-DD，可选")


def estimate_budget(query: BudgetQuery) -> dict[str, Any]:
    """按档次估算总花费与构成。"""
    per_day = _PER_DAY_CNY[query.style]
    total = per_day * query.days * query.people
    breakdown = {k: round(total * ratio) for k, ratio in _RATIO[query.style].items()}
    return {
        "style": query.style.value,
        "days": query.days,
        "people": query.people,
        "per_person_per_day_cny": per_day,
        "total_cny": total,
        "per_person_cny": round(total / query.people),
        "breakdown_cny": breakdown,
        "note": "为粗略估算，实际以实时报价为准。",
    }


def plan_trip(query: TripPlanQuery) -> dict[str, Any]:
    """生成逐日骨架（含主题）与费用估算。"""
    day_plans: list[dict[str, Any]] = []
    for day in range(1, query.days + 1):
        if day == 1:
            theme = "抵达 + 市区轻松逛"
        elif day == query.days:
            theme = "收尾购物 + 返程"
        else:
            theme = "主景点 / 深度体验"
        day_plans.append({"day": day, "theme": theme})

    budget = estimate_budget(
        BudgetQuery(days=query.days, people=query.people, style=query.style)
    )
    itinerary: dict[str, Any] = {
        "origin": query.origin,
        "destination": query.destination,
        "days": day_plans,
        "estimated_cost_cny": budget["total_cny"],
    }
    if query.start_date:
        try:
            start = date.fromisoformat(query.start_date)
            end = start + timedelta(days=query.days - 1)
            itinerary["date_range"] = f"{start.isoformat()} ~ {end.isoformat()}"
        except ValueError:
            itinerary["date_range"] = None

    return {"itinerary": itinerary, "budget": budget}


@dataclass(frozen=True)
class ToolSpec:
    """一个可被编排层调用的工具。"""

    name: str
    description: str
    params_model: type[BaseModel]
    func: Callable[[Any], dict[str, Any]]


TOOLS: dict[str, ToolSpec] = {
    "estimate_budget": ToolSpec(
        name="estimate_budget",
        description="按天数、人数、档次估算旅行花费总额与构成。",
        params_model=BudgetQuery,
        func=estimate_budget,
    ),
    "plan_trip": ToolSpec(
        name="plan_trip",
        description="根据出发地、目的地、天数生成逐日行程骨架与费用估算。",
        params_model=TripPlanQuery,
        func=plan_trip,
    ),
}


def invoke(name: str, **kwargs: Any) -> dict[str, Any]:
    """按名字调用工具：先校验参数，再执行。"""
    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"未知工具：{name}")
    params = spec.params_model.model_validate(kwargs)
    return spec.func(params)
