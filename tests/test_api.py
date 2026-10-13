"""接口测试（用 TestClient；注入假 LLM，避免真实调用）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent import TravelAgent
from app.main import app
from app.rag import KnowledgeBase


def test_health_and_chat(fake_llm_factory):
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        # 替换成使用假 LLM 的编排器
        app.state.agent = TravelAgent(
            fake_llm_factory(["你好，我可以帮你规划行程、估算预算。"]), KnowledgeBase([])
        )
        resp = client.post("/api/v1/chat", json={"message": "你好呀"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"]
        assert "intent" in data and "citations" in data


def test_chat_rejects_empty_message(fake_llm_factory):
    with TestClient(app) as client:
        resp = client.post("/api/v1/chat", json={"message": ""})
        assert resp.status_code == 422  # Pydantic 校验拒绝空消息


def test_chat_stream_endpoint(fake_llm_factory):
    with TestClient(app) as client:
        app.state.agent = TravelAgent(fake_llm_factory(["你好呀，我是旅行助手。"]), KnowledgeBase([]))

        with client.stream("POST", "/api/v1/chat/stream", json={"message": "你好呀"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        assert "data:" in body
        assert '"type": "done"' in body


def test_token_required_when_configured(fake_llm_factory):
    from dataclasses import replace

    with TestClient(app) as client:
        app.state.agent = TravelAgent(fake_llm_factory(["你好"]), KnowledgeBase([]))
        original = app.state.settings
        app.state.settings = replace(original, access_token="s3cret")
        try:
            # 未带口令 → 401
            assert client.post("/api/v1/chat", json={"message": "你好呀"}).status_code == 401
            # 带对口令 → 200
            ok = client.post(
                "/api/v1/chat",
                headers={"X-Access-Token": "s3cret"},
                json={"message": "你好呀"},
            )
            assert ok.status_code == 200
            # 健康检查保持公开（供容器健康检查）
            assert client.get("/api/v1/health").status_code == 200

            # 非 ASCII 口令（走 query）应返回 401，而不是 500
            weird = client.post("/api/v1/chat?token=口令", json={"message": "你好呀"})
            assert weird.status_code == 401
        finally:
            app.state.settings = original


def test_non_ascii_token_can_match(fake_llm_factory):
    """口令本身是中文时，也应当能正确匹配（不是 500）。"""
    from dataclasses import replace

    with TestClient(app) as client:
        app.state.agent = TravelAgent(fake_llm_factory(["你好"] * 3), KnowledgeBase([]))
        original = app.state.settings
        app.state.settings = replace(original, access_token="口令abc")
        try:
            ok = client.post("/api/v1/chat?token=口令abc", json={"message": "你好呀"})
            assert ok.status_code == 200
        finally:
            app.state.settings = original
