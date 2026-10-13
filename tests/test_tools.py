"""旅行工具测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.tools import invoke


def test_estimate_budget_total_and_breakdown():
    out = invoke("estimate_budget", days=5, people=2, style="comfort")

    assert out["total_cny"] == 9000
    assert out["per_person_cny"] == 4500
    assert set(out["breakdown_cny"]) == {"transport", "stay", "food", "tickets"}


def test_plan_trip_builds_day_plans():
    out = invoke(
        "plan_trip", origin="北京", destination="成都", days=3, people=2, style="comfort"
    )

    assert len(out["itinerary"]["days"]) == 3
    assert out["itinerary"]["destination"] == "成都"
    assert out["itinerary"]["estimated_cost_cny"] == 5400


def test_missing_required_param_raises():
    with pytest.raises(ValidationError):
        invoke("plan_trip", destination="成都", days=3)


def test_invalid_style_raises():
    with pytest.raises(ValidationError):
        invoke("estimate_budget", days=3, style="unknown")


def test_unknown_tool_raises():
    with pytest.raises(KeyError):
        invoke("not_a_tool", days=1)
