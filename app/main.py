"""旅行助手 API 入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import Settings, load_settings
from app.intents import IntentRecognizer
from app.llm import LLMClient, Message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

SYSTEM_PROMPT = (
    "你是「旅行助手」，帮助用户规划行程、估算预算、解答签证与出行问题。"
    "回答要具体、可执行；涉及价格、签证等时效性信息时，提醒用户以官方最新信息为准，"
    "不确定的内容不要编造。"
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")
    session_id: str | None = Field(None, description="会话标识（后续用于多轮记忆）")


class ChatResponse(BaseModel):
    reply: str
    model: str
    intent: str
    confidence: float
    source: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = load_settings()
    llm = LLMClient(settings)
    app.state.settings = settings
    app.state.llm = llm
    app.state.intents = IntentRecognizer(llm)
    logging.getLogger(__name__).info("旅行助手启动，模型：%s", settings.model)
    yield


app = FastAPI(
    title="旅行助手 API",
    version="0.1.0",
    description="个人旅行助手：对话 → 行程规划 / 预算估算 / 攻略问答。",
    lifespan=lifespan,
)


@app.get("/api/v1/health", tags=["system"])
async def health(request: Request) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    return {"status": "ok", "model": settings.model}


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """单轮对话：先识别意图，再交给模型生成回答。"""
    llm: LLMClient = request.app.state.llm
    recognizer: IntentRecognizer = request.app.state.intents

    result = await recognizer.recognize(body.message)
    try:
        reply = await llm.complete(
            [Message(role="user", content=body.message)], system=SYSTEM_PROMPT
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc

    return ChatResponse(
        reply=reply,
        model=llm.model,
        intent=result.intent.value,
        confidence=round(result.confidence, 3),
        source=result.source,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
