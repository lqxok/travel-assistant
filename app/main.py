"""旅行助手 API 入口。"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import TravelAgent
from app.config import Settings, load_settings
from app.llm import LLMClient
from app.rag import KnowledgeBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")
    session_id: str | None = Field(None, description="会话标识，用于同一会话内的多轮记忆")
    user_id: str | None = Field(None, description="用户标识，用于跨会话长期记忆")


class ChatResponse(BaseModel):
    reply: str
    model: str
    intent: str
    confidence: float
    source: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_result: dict[str, Any] | None = None
    tool_name: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = load_settings()
    llm = LLMClient(settings)
    knowledge_base = KnowledgeBase.from_json()
    app.state.settings = settings
    app.state.llm = llm
    app.state.agent = TravelAgent(llm, knowledge_base)
    logging.getLogger(__name__).info(
        "旅行助手启动，模型：%s，知识库：%d 条", settings.model, knowledge_base.size
    )
    yield


app = FastAPI(
    title="旅行助手 API",
    version="0.1.0",
    description="个人旅行助手：对话 → 意图路由 → 本地检索 / 旅行工具。",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def require_token(request: Request) -> None:
    """配置了 ACCESS_TOKEN 时校验访问口令（请求头 X-Access-Token 或 query ?token=）。"""
    settings: Settings = request.app.state.settings
    expected = settings.access_token
    if not expected:
        return  # 未配置口令 → 不校验
    provided = request.headers.get("x-access-token") or request.query_params.get("token") or ""
    # 用 bytes 比较：compare_digest 不支持非 ASCII 字符串，直接比 str 会抛异常
    if not secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="访问口令无效或未提供")


# 需要口令的接口（健康检查保持公开，供容器健康检查使用）
guard = [Depends(require_token)]


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/v1/health", tags=["system"])
async def health(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    agent: TravelAgent = request.app.state.agent
    return {
        "status": "ok",
        "model": settings.model,
        "kb_size": agent._kb.size,
        "profile_users": agent.profile_user_count,
        "auth_required": bool(settings.access_token),
    }


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"], dependencies=guard)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """单轮对话：交给编排器处理，返回回答、意图与引用来源。"""
    agent: TravelAgent = request.app.state.agent
    llm: LLMClient = request.app.state.llm
    try:
        result = await agent.handle(
            body.message, session_id=body.session_id, user_id=body.user_id
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"处理失败：{exc}") from exc

    return ChatResponse(
        reply=result.reply,
        model=llm.model,
        intent=result.intent.value,
        confidence=round(result.confidence, 3),
        source=result.source,
        citations=result.citations,
        tool_result=result.tool_result,
        tool_name=result.tool_name,
    )


@app.post("/api/v1/chat/stream", tags=["chat"], dependencies=guard)
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """流式对话：SSE 事件为 status / delta / done / error。"""
    agent: TravelAgent = request.app.state.agent

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in agent.astream(
                body.message, session_id=body.session_id, user_id=body.user_id
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            error = {"type": "error", "error": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/api/v1/profile/{user_id}", tags=["profile"], dependencies=guard)
async def clear_profile(user_id: str, request: Request) -> dict[str, str]:
    """清除某个用户的长期记忆。"""
    agent: TravelAgent = request.app.state.agent
    agent.clear_profile(user_id)
    return {"status": "ok", "user_id": user_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
