"""应用配置：从环境变量与 .env 读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.embedder import DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL

# 项目根目录（app/ 的上一层）
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """运行时配置。"""

    app_name: str
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout: float
    access_token: str = ""
    embed_model: str = DEFAULT_EMBED_MODEL
    rerank_model: str = DEFAULT_RERANK_MODEL
    rate_limit_per_minute: int = 60
    log_format: str = "text"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def load_settings() -> Settings:
    """读取 .env 与环境变量，返回配置对象。"""
    load_dotenv(BASE_DIR / ".env")
    return Settings(
        app_name=os.getenv("APP_NAME", "旅行助手"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        request_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        access_token=os.getenv("ACCESS_TOKEN", "").strip(),
        # 语义检索模型；设为 none/off 可关闭该通道（关闭后仅走词面+意图）
        embed_model=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip(),
        # 重排模型；设为 none/off 可关闭（关闭后用 RRF 融合排序）
        rerank_model=os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL).strip(),
        # 每客户端每分鐘限流次数；设 0 关闭限流
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        # 日志格式：text（可读）/ json（结构化，便于采集）
        log_format=os.getenv("LOG_FORMAT", "text").strip().lower(),
    )
