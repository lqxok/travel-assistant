"""可观测性：请求追踪 ID、结构化日志、接口限流。

三件事各司其职：
- **请求追踪**：每个请求分配一个 `request_id`（也接受上游传入的 `X-Request-ID`），
  放进 ContextVar，于是**同一次请求的所有日志**都能带上它，出问题能串起来查；
- **结构化日志**：默认输出可读文本（带 request_id），设 `LOG_FORMAT=json` 则输出 JSON 行，
  方便被 ELK / 云日志直接采集；
- **限流**：单实例内存里的滑动窗口计数，超过阈值返回 429，防止被刷。

限流用「滑动窗口」而不是固定窗口，是为了避免临界点的双倍突发。
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextvars import ContextVar
from datetime import datetime, timezone

# 每个请求一个 ID；放在 ContextVar 里，跨 await 也随协程走，日志过滤器直接读取
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

DEFAULT_WINDOW_SECONDS = 60.0


class RequestIdFilter(logging.Filter):
    """把当前请求的 request_id 注入每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """一条日志一行 JSON，便于日志系统采集。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """带 request_id 的可读文本日志（本地调试友好）。"""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        return f"[{getattr(record, 'request_id', '-')}] {base}"


def setup_logging(level: int = logging.INFO, *, fmt: str = "text") -> None:
    """配置根日志：注入 request_id，并选择文本 / JSON 格式。"""
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            TextFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


class RateLimiter:
    """滑动窗口限流器：每个 key（如客户端 IP）在窗口内最多放行 limit 次。"""

    def __init__(self, window: float = DEFAULT_WINDOW_SECONDS) -> None:
        self._window = window
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, limit: int) -> bool:
        """允许则记一次并返回 True；超限返回 False。"""
        now = time.monotonic()
        bucket = self._hits.setdefault(key, deque())
        cutoff = now - self._window
        while bucket and bucket[0] <= cutoff:  # 丢掉窗口外的旧记录
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True

    def reset(self) -> None:
        self._hits.clear()
