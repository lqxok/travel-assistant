"""会话记忆测试。"""

from __future__ import annotations

from app.memory import ConversationMemory


def test_append_and_history():
    memory = ConversationMemory()
    memory.append("s1", "你好", "你好呀")

    history = memory.history("s1")
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content == "你好"


def test_history_isolated_by_session():
    memory = ConversationMemory()
    memory.append("s1", "a", "b")
    assert memory.history("s2") == []


def test_trims_to_max_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.append("s1", f"u{i}", f"a{i}")

    history = memory.history("s1")
    assert len(history) == 4  # 2 轮 × 2 条
    assert history[0].content == "u3"  # 只保留最近的
    assert memory.session_count == 1


def test_clear():
    memory = ConversationMemory()
    memory.append("s1", "a", "b")
    memory.clear("s1")
    assert memory.history("s1") == []
