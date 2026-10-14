"""可观测性测试：请求追踪 ID、限流。"""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import app
from app.observability import RateLimiter


def test_rate_limiter_sliding_window():
    limiter = RateLimiter(window=60)
    assert limiter.allow("ip-1", 2)
    assert limiter.allow("ip-1", 2)
    assert not limiter.allow("ip-1", 2)  # 第三次超限
    assert limiter.allow("ip-2", 2)  # 不同 key 互不影响


def test_response_carries_request_id():
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-Request-ID")


def test_incoming_request_id_is_echoed():
    with TestClient(app) as client:
        resp = client.get("/api/v1/health", headers={"X-Request-ID": "trace-abc"})
        assert resp.headers["X-Request-ID"] == "trace-abc"


def test_rate_limit_returns_429():
    with TestClient(app) as client:
        original = app.state.settings
        app.state.settings = replace(original, rate_limit_per_minute=2)
        try:
            assert client.delete("/api/v1/profile/u1").status_code == 200
            assert client.delete("/api/v1/profile/u1").status_code == 200
            assert client.delete("/api/v1/profile/u1").status_code == 429
        finally:
            app.state.settings = original


def test_health_is_exempt_from_rate_limit():
    with TestClient(app) as client:
        original = app.state.settings
        app.state.settings = replace(original, rate_limit_per_minute=1)
        try:
            for _ in range(5):  # 健康检查不受限流影响（供容器探针使用）
                assert client.get("/api/v1/health").status_code == 200
        finally:
            app.state.settings = original
