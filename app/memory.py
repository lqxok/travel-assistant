"""会话记忆：按 session_id 保存最近若干轮对话（进程内存）。

用途：把历史消息一并带给模型，让「那改成 5 天」这类追问能被理解。

注意：进程重启即清空，多实例之间不共享；生产环境可换成 Redis 等。
"""

from __future__ import annotations

from app.llm import Message


class ConversationMemory:
    """按 session_id 维护最近若干轮对话。"""

    def __init__(self, max_turns: int = 6) -> None:
        self._max_turns = max_turns
        self._store: dict[str, list[Message]] = {}

    def history(self, session_id: str) -> list[Message]:
        """返回该会话的历史消息（副本）。"""
        return list(self._store.get(session_id, []))

    def append(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """记录一轮问答，只保留最近 max_turns 轮。"""
        turns = self._store.setdefault(session_id, [])
        turns.append(Message(role="user", content=user_text))
        turns.append(Message(role="assistant", content=assistant_text))
        limit = self._max_turns * 2
        if len(turns) > limit:
            del turns[: len(turns) - limit]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    @property
    def session_count(self) -> int:
        return len(self._store)
