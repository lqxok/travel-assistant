"""旅行助手 API 入口。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.admin import router as admin_router
from app.agent import TravelAgent
from app.auth import require_token
from app.cache import StatsCache
from app.config import Settings, load_settings
from app.embedder import build_embedder, build_reranker
from app.llm import LLMClient
from app.rag import KnowledgeBase
from app.store import DEFAULT_DATABASE_URL, ChatStore

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
    embedder = build_embedder(settings.embed_model)
    reranker = build_reranker(settings.rerank_model)
    knowledge_base = KnowledgeBase.from_json(embedder=embedder, reranker=reranker)
    knowledge_base.warmup()  # 预加载向量/重排模型，避免首个请求冷启动

    store = ChatStore(settings.database_url or DEFAULT_DATABASE_URL)
    await store.init()
    cache = StatsCache(settings.redis_url)

    app.state.settings = settings
    app.state.llm = llm
    app.state.agent = TravelAgent(llm, knowledge_base)
    app.state.store = store
    app.state.cache = cache
    logging.getLogger(__name__).info(
        "旅行助手启动，模型：%s，知识库：%d 条，语义检索：%s，重排：%s，日志库：%s，缓存：%s",
        settings.model,
        knowledge_base.size,
        "开" if knowledge_base.has_vector else "关",
        "开" if knowledge_base.has_reranker else "关",
        store.url,
        "开" if cache.enabled else "关",
    )
    yield
    await cache.close()
    await store.close()


app = FastAPI(
    title="旅行助手 API",
    version="0.1.0",
    description="个人旅行助手：对话 → 意图路由 → 本地检索 / 旅行工具。",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# 需要口令的接口（健康检查保持公开，供容器健康检查使用）
guard = [Depends(require_token)]
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    """运营数据看板页面。"""
    return FileResponse(str(STATIC_DIR / "admin.html"))


@app.get("/api/v1/health", tags=["system"])
async def health(request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    agent: TravelAgent = request.app.state.agent
    return {
        "status": "ok",
        "model": settings.model,
        "kb_size": agent._kb.size,
        "vector": agent._kb.has_vector,
        "rerank": agent._kb.has_reranker,
        "profile_users": agent.profile_user_count,
        "auth_required": bool(settings.access_token),
        "database": request.app.state.store.url.split("://", 1)[0],
        "cache": request.app.state.cache.enabled,
    }


async def _log_call(
    request: Request,
    *,
    intent: str,
    source: str,
    tool_name: str | None,
    citations: list[dict[str, Any]],
    latency_ms: int,
    session_id: str | None,
    user_id: str | None,
) -> None:
    """把一次调用落库供看板统计；失败只记日志，不影响对话。"""
    store: ChatStore = request.app.state.store
    channel = citations[0].get("channel") if citations else None
    try:
        await store.record(
            intent=intent,
            source=source,
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_name,
            channel=channel,
            latency_ms=latency_ms,
        )
    except Exception:  # noqa: BLE001 —— 打点失败不该影响回答
        logging.getLogger(__name__).warning("对话日志落库失败", exc_info=True)


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"], dependencies=guard)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """单轮对话：交给编排器处理，返回回答、意图与引用来源。"""
    agent: TravelAgent = request.app.state.agent
    llm: LLMClient = request.app.state.llm
    started = time.perf_counter()
    try:
        result = await agent.handle(
            body.message, session_id=body.session_id, user_id=body.user_id
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"处理失败：{exc}") from exc

    await _log_call(
        request,
        intent=result.intent.value,
        source=result.source,
        tool_name=result.tool_name,
        citations=result.citations,
        latency_ms=int((time.perf_counter() - started) * 1000),
        session_id=body.session_id,
        user_id=body.user_id,
    )
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
        started = time.perf_counter()
        done_event: dict[str, Any] | None = None
        try:
            async for event in agent.astream(
                body.message, session_id=body.session_id, user_id=body.user_id
            ):
                if event.get("type") == "done":
                    done_event = event
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            error = {"type": "error", "error": str(exc)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            return
        done = done_event or {}
        await _log_call(
            request,
            intent=str(done.get("intent", "general")),
            source=str(done.get("source", "")),
            tool_name=done.get("tool_name"),
            citations=done.get("citations") or [],
            latency_ms=int((time.perf_counter() - started) * 1000),
            session_id=body.session_id,
            user_id=body.user_id,
        )

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
