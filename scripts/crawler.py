"""Playwright 爬虫：抓取公开的城市旅行信息并自动回填数据库（RPA/自动化示例）。

要点：
- **浏览器自动化**（playwright + Chromium），能处理 JS 渲染的页面，比纯 requests 更通用；
- **反爬应对**：伪装 UA / 语言头、请求间随机延迟、失败重试、遵守 robots.txt；
- **结构化抽取 → 落库**：复用 `app.store`（MySQL / SQLite），并可选飞书通知。

依赖单独放置、不装进主镜像（浏览器内核较大）：
    pip install -r scripts/requirements-crawler.txt
    playwright install chromium

用法：
    python -m scripts.crawler --cities 京都 清迈 巴厘岛
    python -m scripts.crawler --cities 京都 --notify
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
from typing import Any
from urllib.parse import quote
from urllib.robotparser import RobotFileParser

from app.config import load_settings
from app.store import DEFAULT_DATABASE_URL, ChatStore

logger = logging.getLogger(__name__)

SOURCE = "zh.wikipedia.org"
BASE = "https://zh.wikipedia.org"
# 站方要求爬虫标识自己；这里用「常见浏览器 UA + 机器人身份」的折中写法
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 TravelAssistantBot/0.1"
)
MIN_SUMMARY_LEN = 40
RETRIES = 3


def city_url(city: str) -> str:
    """城市 → 维基百科词条地址。"""
    return f"{BASE}/wiki/{quote(city)}"


def clean(text: str) -> str:
    """压缩空白，便于入库与展示。"""
    return re.sub(r"\s+", " ", (text or "")).strip()


def allowed_by_robots(url: str, user_agent: str = USER_AGENT) -> bool:
    """遵守 robots.txt（读不到就不拦）。"""
    parser = RobotFileParser()
    parser.set_url(f"{BASE}/robots.txt")
    try:
        parser.read()
    except Exception:  # noqa: BLE001 —— 网络异常时保守放行
        logger.warning("读取 robots.txt 失败，跳过检查")
        return True
    return parser.can_fetch(user_agent, url)


async def fetch_city(page: Any, city: str) -> dict[str, Any] | None:
    """抓取单个城市的词条，返回结构化条目；失败返回 None。"""
    url = city_url(city)
    for attempt in range(1, RETRIES + 1):
        try:
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            title = clean(await page.title()) or city
            paragraphs = await page.eval_on_selector_all(
                "#mw-content-text p", "els => els.map(e => e.innerText)"
            )
            summary = next(
                (clean(p) for p in paragraphs if len(clean(p)) >= MIN_SUMMARY_LEN), ""
            )
            if not summary:
                logger.info("「%s」未取到有效正文", city)
                return None
            return {
                "source": SOURCE,
                "city": city,
                "title": title,
                "summary": summary[:1000],
                "url": url,
            }
        except Exception:  # noqa: BLE001 —— 网络/超时都重试
            logger.warning("抓取「%s」第 %d/%d 次失败", city, attempt, RETRIES)
            await asyncio.sleep(2 * attempt)
    return None


async def crawl(cities: list[str], *, headless: bool = True) -> list[dict[str, Any]]:
    """逐个城市抓取；城市之间加入随机延迟，降低被封风险。"""
    from playwright.async_api import async_playwright

    results: list[dict[str, Any]] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(user_agent=USER_AGENT, locale="zh-CN")
        page = await context.new_page()
        try:
            for city in cities:
                if not allowed_by_robots(city_url(city)):
                    logger.info("robots.txt 不允许抓取：%s，跳过", city)
                    continue
                item = await fetch_city(page, city)
                if item:
                    results.append(item)
                    logger.info("已抓取：%s", city)
                await asyncio.sleep(random.uniform(1.0, 3.0))
        finally:
            await browser.close()
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="抓取城市旅行信息并入库")
    parser.add_argument("--cities", nargs="+", default=["京都", "清迈", "巴厘岛"])
    parser.add_argument("--notify", action="store_true", help="抓完后推送飞书")
    parser.add_argument("--headed", action="store_true", help="显示浏览器窗口（调试用）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    settings = load_settings()

    items = await crawl(args.cities, headless=not args.headed)

    store = ChatStore(settings.database_url or DEFAULT_DATABASE_URL)
    await store.init()
    written = await store.save_crawl_items(items)
    total = await store.crawl_count()
    await store.close()

    logger.info("抓取 %d 条，新增入库 %d 条，库内共 %d 条", len(items), written, total)
    if args.notify:
        from app.notify import send_feishu

        send_feishu(
            f"旅行助手爬虫完成：本轮抓取 {len(items)} 条，新增入库 {written} 条，累计 {total} 条。"
        )


if __name__ == "__main__":
    asyncio.run(main())
