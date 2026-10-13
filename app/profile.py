"""用户长期记忆：跨会话记住用户的稳定偏好/事实，落盘到 JSON。

与会话记忆（`ConversationMemory`）的区别：
- 会话记忆按 `session_id`，只在同一次对话内有效；
- 用户长期记忆按 `user_id`，**跨会话有效**、且**重启不丢**。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import BASE_DIR

DEFAULT_PROFILE_PATH = BASE_DIR / "data" / "profiles.json"

_MAX_FACTS = 10
_MAX_LEN = 40

# 只有出现这些线索时才值得调用模型抽取，避免每轮都花 token
_HINT = re.compile(
    r"(常驻|住在|居住|来自|不吃|忌口|过敏|素食|喜欢|偏爱|习惯|预算|一个人|两个人|带(老人|小孩)|我是)"
)


def worth_extracting(text: str) -> bool:
    """粗略判断这句里是否可能含稳定偏好。"""
    return bool(_HINT.search(text))


class UserProfileStore:
    """按 user_id 保存长期事实，文件持久化。"""

    def __init__(self, path: str | Path = DEFAULT_PROFILE_PATH) -> None:
        self._path = Path(path)
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = {k: list(v) for k, v in raw.items() if isinstance(v, list)}
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def facts(self, user_id: str) -> list[str]:
        return list(self._data.get(user_id, []))

    def add_facts(self, user_id: str, facts: list[str]) -> None:
        """追加事实（去重、限长、只保留最近 _MAX_FACTS 条）。"""
        if not user_id:
            return
        current = self._data.setdefault(user_id, [])
        for fact in facts:
            text = str(fact).strip()[: _MAX_LEN]
            if text and text not in current:
                current.append(text)
        if len(current) > _MAX_FACTS:
            del current[: len(current) - _MAX_FACTS]
        self._save()

    def clear(self, user_id: str) -> None:
        self._data.pop(user_id, None)
        self._save()

    @property
    def user_count(self) -> int:
        return len(self._data)
