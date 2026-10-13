"""运营数据看板的接口（口令保护）。

给运营/管理员用：看调用量、意图与工具分布、延迟分位、检索命中通道，以及最近日志。
统计结果经 Redis 缓存（未启用 Redis 时直接查库）。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from app.auth import require_token

router = APIRouter(prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(require_token)])


@router.get("/stats")
async def stats(request: Request, days: int = Query(7, ge=1, le=30)) -> dict[str, Any]:
    """看板主数据；命中缓存时 `cached` 为 true。"""
    cache = request.app.state.cache
    key = f"stats:{days}"
    cached = await cache.get_json(key)
    if cached is not None:
        cached["cached"] = True
        return cached

    data = await request.app.state.store.stats(days=days)
    data["cached"] = False
    await cache.set_json(key, data, ttl=15)
    return data


@router.get("/logs")
async def logs(request: Request, limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    """最近的对话调用记录，倒序。"""
    items = await request.app.state.store.recent(limit=limit)
    return {"items": items, "count": len(items)}
