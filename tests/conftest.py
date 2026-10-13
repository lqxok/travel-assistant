"""测试公共件：一个可预设回复的假 LLM，避免测试打到真实 API。"""

from __future__ import annotations

import os

import pytest

# 测试默认关闭本地语义/重排模型：既不下载大模型，也让测试保持离线、快速。
# 用 setdefault 的好处是：本地想验证真实模型时，显式设环境变量即可覆盖。
os.environ.setdefault("EMBED_MODEL", "none")
os.environ.setdefault("RERANK_MODEL", "none")


class FakeLLM:
    """按预设回复依次返回；记录每次调用，便于断言。"""

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or [])
        self.calls: list[dict] = []
        self.model = "fake"

    async def complete(self, messages, *, system=None, temperature=None, max_tokens=None):
        self.calls.append({"messages": messages, "system": system})
        if self._replies:
            return self._replies.pop(0)
        return "（默认回答）"

    async def stream(self, messages, *, system=None, temperature=None, max_tokens=None):
        self.calls.append({"messages": messages, "system": system})
        text = self._replies.pop(0) if self._replies else "（默认回答）"
        for ch in text:
            yield ch


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
