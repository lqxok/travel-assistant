"""LLM 客户端：走 OpenAI 兼容接口，带指数退避重试。

只依赖 `openai` SDK；模型、地址、密钥都来自配置，便于切换供应商。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Message:
    """一条对话消息。"""

    role: str  # system / user / assistant
    content: str


class LLMClient:
    """封装一次对话补全调用，失败按指数退避重试。"""

    def __init__(self, settings: Settings, *, max_retries: int = 2, backoff: float = 1.0) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.api_key or "dummy",
            base_url=settings.base_url,
            timeout=settings.request_timeout,
        )
        self._model = settings.model
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens
        self._max_retries = max_retries
        self._backoff = backoff

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发一次对话补全，返回模型正文。"""
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=payload,  # type: ignore[arg-type]
                    temperature=self._temperature if temperature is None else temperature,
                    max_tokens=self._max_tokens if max_tokens is None else max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM 调用失败（第 %d 次）：%s", attempt + 1, exc)
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff * (2**attempt))

        raise RuntimeError(f"LLM 调用失败：{last_error}")
