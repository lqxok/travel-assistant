"""Redis 缓存（可选）。

给看板统计做短期缓存，削掉重复聚合查询。未配置 `REDIS_URL` 或连不上时自动降级为
「无缓存」，功能不受影响——这也是对外部署时「依赖可选、坏一个不拖垮全站」的做法。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class StatsCache:
    """统计缓存；`enabled` 为 False 时所有操作都是空操作。"""

    def __init__(self, url: str | None = None) -> None:
        self._url = (url or "").strip()
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    async def _ensure_client(self):
        if self._client is None and self._url:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self._url, decode_responses=True)
        return self._client

    async def get_json(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            client = await self._ensure_client()
            raw = await client.get(key)
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001 —— 缓存失效不应影响主流程
            logger.warning("Redis 读取失败，本轮跳过缓存", exc_info=True)
            return None

    async def set_json(self, key: str, value: Any, *, ttl: int = 15) -> None:
        if not self.enabled:
            return
        try:
            client = await self._ensure_client()
            await client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
        except Exception:  # noqa: BLE001
            logger.warning("Redis 写入失败，本轮跳过缓存", exc_info=True)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
