"""访问口令鉴权。

配置了 `ACCESS_TOKEN` 时，受保护接口需带请求头 `X-Access-Token` 或 query `?token=`。
独立成模块，便于 `main.py` 与 `admin.py` 复用同一份守卫逻辑。
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from app.config import Settings


def require_token(request: Request) -> None:
    """校验访问口令；未配置口令则直接放行。"""
    settings: Settings = request.app.state.settings
    expected = settings.access_token
    if not expected:
        return
    provided = request.headers.get("x-access-token") or request.query_params.get("token") or ""
    # 用 bytes 比较：compare_digest 不支持非 ASCII 字符串，直接比 str 会抛异常
    if not secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="访问口令无效或未提供")
