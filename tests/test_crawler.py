"""爬虫脚本的纯函数与数据回填测试（不启动浏览器）。"""

from __future__ import annotations

from app.store import ChatStore
from scripts.crawler import city_url, clean


def test_city_url_encodes_chinese():
    url = city_url("京都")
    assert url.startswith("https://zh.wikipedia.org/wiki/")
    assert "%" in url  # 中文被百分号编码


def test_clean_collapses_whitespace():
    assert clean("  a\n\n b \t c ") == "a b c"
    assert clean("") == ""
    assert clean(None) == ""


async def test_save_crawl_items_dedups(tmp_path):
    store = ChatStore(f"sqlite+aiosqlite:///{(tmp_path / 'c.db').as_posix()}")
    await store.init()
    try:
        items = [
            {"source": "s", "city": "京都", "title": "京都", "summary": "x", "url": "u"},
            {"source": "s", "city": "京都", "title": "京都", "summary": "x", "url": "u"},
            {"source": "s", "city": "清迈", "title": "清迈", "summary": "y", "url": "u"},
        ]
        assert await store.save_crawl_items(items) == 2  # 重复的被跳过
        assert await store.crawl_count() == 2
        assert await store.save_crawl_items(items) == 0  # 再存一次不新增
        assert await store.crawl_count() == 2
    finally:
        await store.close()
