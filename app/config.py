"""应用配置：从环境变量与 .env 读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        request_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
    )
