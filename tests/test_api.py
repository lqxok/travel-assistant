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
