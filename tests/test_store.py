"""对话日志存储与统计测试（SQLite，本地零依赖）。"""

from __future__ import annotations

from app.store import ChatStore


async def _store(tmp_path) -> ChatStore:
    store = ChatStore(f"sqlite+aiosqlite:///{(tmp_path / 't.db').as_posix()}")
    await store.init()
    return store


async def test_record_and_stats(tmp_path):
    store = await _store(tmp_path)
    try:
        await store.record(
            intent="guide", source="rule", session_id="s1", user_id="u1",
            channel="rerank", latency_ms=100,
        )
        await store.record(
            intent="guide", source="rule", session_id="s1", user_id="u1",
            channel="keyword", latency_ms=300,
        )
        await store.record(
            intent="budget", source="llm", session_id="s2", user_id="u2",
            tool_name="estimate_budget", latency_ms=200,
        )

        stats = await store.stats(days=7)
        assert stats["total_calls"] == 3
        assert stats["sessions"] == 2
        assert stats["users"] == 2
        assert stats["intent_distribution"] == {"guide": 2, "budget": 1}
        assert stats["tool_distribution"] == {"estimate_budget": 1}
        assert stats["channel_distribution"] == {"rerank": 1, "keyword": 1}
        assert stats["latency_ms"]["mean"] == 200.0
        assert stats["latency_ms"]["max"] == 300.0
        assert len(stats["trend"]) == 7
    finally:
        await store.close()


async def test_recent_returns_newest_first(tmp_path):
    store = await _store(tmp_path)
    try:
        await store.record(intent="general", source="rule")
        await store.record(intent="stay", source="rule")

        rows = await store.recent(limit=10)
        assert len(rows) == 2
        assert rows[0]["intent"] == "stay"  # 最新一条在最前
        assert "created_at" in rows[0]
    finally:
        await store.close()


async def test_stats_on_empty_store(tmp_path):
    store = await _store(tmp_path)
    try:
        stats = await store.stats()
        assert stats["total_calls"] == 0
        assert stats["latency_ms"] == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        assert all(point["count"] == 0 for point in stats["trend"])
    finally:
        await store.close()
