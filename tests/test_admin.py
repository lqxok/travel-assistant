"""运营看板接口测试。"""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import app


def test_admin_stats_and_logs():
    with TestClient(app) as client:
        client.post("/api/v1/chat", json={"message": "你好呀"})

        stats = client.get("/api/v1/admin/stats?days=7")
        assert stats.status_code == 200
        body = stats.json()
        assert body["total_calls"] >= 1
        assert "intent_distribution" in body and "latency_ms" in body

        logs = client.get("/api/v1/admin/logs?limit=5")
        assert logs.status_code == 200
        assert logs.json()["count"] >= 1


def test_admin_requires_token_when_configured():
    with TestClient(app) as client:
        original = app.state.settings
        app.state.settings = replace(original, access_token="s3cret")
        try:
            assert client.get("/api/v1/admin/stats").status_code == 401
            ok = client.get("/api/v1/admin/stats", headers={"X-Access-Token": "s3cret"})
            assert ok.status_code == 200
        finally:
            app.state.settings = original
