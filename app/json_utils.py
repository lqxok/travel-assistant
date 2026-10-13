"""从模型输出里稳健地取出 JSON 对象。"""

from __future__ import annotations

import json
import re

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def extract_json(raw: str) -> dict:
    """把模型返回的文本解析成 dict：先整体解析，失败则退化为抓第一个 {...}。"""
    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"未找到 JSON 对象：{text[:80]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层不是对象")
    return data
