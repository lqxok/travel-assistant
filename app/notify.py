"""飞书群机器人通知（可选）。

配置 `FEISHU_WEBHOOK` 后，脚本可把结果推送到飞书群；未配置则静默跳过。
对应 JD 里「对接飞书等办公平台接口」的低门槛场景。
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


def send_feishu(text: str, *, webhook: str | None = None) -> bool:
    """给飞书自定义机器人发一条文本消息；成功返回 True。"""
    url = (webhook or os.getenv("FEISHU_WEBHOOK", "")).strip()
    if not url:
        logger.info("未配置 FEISHU_WEBHOOK，跳过飞书通知")
        return False
    try:
        resp = httpx.post(
            url,
            json={"msg_type": "text", "content": {"text": text}},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception:  # noqa: BLE001 —— 通知失败不该影响主流程
        logger.warning("飞书通知发送失败", exc_info=True)
        return False
