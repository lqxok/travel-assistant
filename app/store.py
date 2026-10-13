"""对话日志存储与统计（运营看板的数据底座）。

生产用 **MySQL**（docker-compose 提供），本地与测试自动回退 **SQLite**——同一套
SQLAlchemy 代码兼容两者，改 `DATABASE_URL` 即可，不需要改业务逻辑。

每次对话调用落一条记录（意图 / 来源 / 工具 / 命中通道 / 延迟），供管理后台聚合展示。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import DateTime, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)

# 未配置 DATABASE_URL 时的默认值（本地/测试零依赖即可跑通）
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///data/app.db"


def _utcnow() -> datetime:
    """统一用「无时区的 UTC」，避免 SQLite / MySQL 在时区比较上不一致。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ChatLog(Base):
    """一次对话调用的记录。"""

    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    intent: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(16))
    tool_name: Mapped[str | None] = mapped_column(String(32), nullable=True)
    channel: Mapped[str | None] = mapped_column(String(16), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class CrawlItem(Base):
    """爬虫抓取到的条目（供脚本自动回填数据库）。"""

    __tablename__ = "crawl_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), index=True)  # 站点标识
    city: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    summary: Mapped[str] = mapped_column(String(2000), default="")
    url: Mapped[str] = mapped_column(String(512), default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class ChatStore:
    """对话日志仓库：写入 + 聚合统计。所有方法都是异步的。"""

    def __init__(self, url: str = DEFAULT_DATABASE_URL) -> None:
        self._url = url
        self._engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def url(self) -> str:
        return self._url

    async def init(self) -> None:
        """建表（幂等）。"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    # ---------- 写入 ----------
    async def record(
        self,
        *,
        intent: str,
        source: str,
        session_id: str | None = None,
        user_id: str | None = None,
        tool_name: str | None = None,
        channel: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        async with self._session() as db:
            db.add(
                ChatLog(
                    session_id=session_id,
                    user_id=user_id,
                    intent=intent,
                    source=source,
                    tool_name=tool_name,
                    channel=channel,
                    latency_ms=latency_ms,
                )
            )
            await db.commit()

    # ---------- 查询 ----------
    async def stats(self, *, days: int = 7) -> dict[str, Any]:
        """看板主数据：总量、今日、分布、延迟分位、近 N 天趋势。"""
        now = _utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        window_start = today_start - timedelta(days=days - 1)

        async with self._session() as db:
            total = (await db.execute(select(func.count()).select_from(ChatLog))).scalar_one()
            today = (
                await db.execute(
                    select(func.count()).select_from(ChatLog).where(ChatLog.created_at >= today_start)
                )
            ).scalar_one()
            sessions = (
                await db.execute(select(func.count(func.distinct(ChatLog.session_id))))
            ).scalar_one()
            users = (
                await db.execute(select(func.count(func.distinct(ChatLog.user_id))))
            ).scalar_one()
            intent_rows = (
                await db.execute(select(ChatLog.intent, func.count()).group_by(ChatLog.intent))
            ).all()
            tool_rows = (
                await db.execute(
                    select(ChatLog.tool_name, func.count())
                    .where(ChatLog.tool_name.is_not(None))
                    .group_by(ChatLog.tool_name)
                )
            ).all()
            channel_rows = (
                await db.execute(
                    select(ChatLog.channel, func.count())
                    .where(ChatLog.channel.is_not(None))
                    .group_by(ChatLog.channel)
                )
            ).all()
            latencies = (await db.execute(select(ChatLog.latency_ms))).scalars().all()
            daily_rows = (
                await db.execute(
                    select(func.date(ChatLog.created_at), func.count())
                    .where(ChatLog.created_at >= window_start)
                    .group_by(func.date(ChatLog.created_at))
                )
            ).all()

        daily_map = {str(row[0])[:10]: row[1] for row in daily_rows}
        trend = [
            {
                "date": (window_start + timedelta(days=offset)).strftime("%m-%d"),
                "count": daily_map.get((window_start + timedelta(days=offset)).strftime("%Y-%m-%d"), 0),
            }
            for offset in range(days)
        ]

        return {
            "total_calls": total,
            "today_calls": today,
            "sessions": sessions,
            "users": users,
            "intent_distribution": {str(k): v for k, v in intent_rows},
            "tool_distribution": {str(k): v for k, v in tool_rows},
            "channel_distribution": {str(k): v for k, v in channel_rows},
            "latency_ms": _latency_summary(list(latencies)),
            "trend": trend,
        }

    async def recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """最近的调用记录（倒序），供日志表展示。"""
        async with self._session() as db:
            rows = (
                await db.execute(select(ChatLog).order_by(ChatLog.id.desc()).limit(limit))
            ).scalars().all()
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_id": r.user_id,
                "intent": r.intent,
                "source": r.source,
                "tool_name": r.tool_name,
                "channel": r.channel,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for r in rows
        ]


    # ---------- 爬虫数据回填 ----------
    async def save_crawl_items(self, items: list[dict[str, Any]]) -> int:
        """批量写入抓取结果，返回写入条数。`city`/`title` 相同视为重复，跳过。"""
        written = 0
        async with self._session() as db:
            existing = set(
                (row[0], row[1])
                for row in (await db.execute(select(CrawlItem.city, CrawlItem.title))).all()
            )
            for item in items:
                key = (item.get("city", ""), item.get("title", ""))
                if key in existing:
                    continue
                db.add(
                    CrawlItem(
                        source=item.get("source", ""),
                        city=item.get("city", ""),
                        title=item.get("title", ""),
                        summary=item.get("summary", ""),
                        url=item.get("url", ""),
                    )
                )
                existing.add(key)
                written += 1
            await db.commit()
        return written

    async def crawl_count(self) -> int:
        async with self._session() as db:
            return (await db.execute(select(func.count()).select_from(CrawlItem))).scalar_one()


def _latency_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)

    def pick(pct: float) -> float:
        index = min(int(len(ordered) * pct / 100), len(ordered) - 1)
        return float(ordered[index])

    return {
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": pick(50),
        "p95": pick(95),
        "max": float(ordered[-1]),
    }
