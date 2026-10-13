"""向量/重排封装测试（不加载真实模型）。"""

from __future__ import annotations

from app.embedder import build_embedder, build_reranker, cosine, is_disabled


def test_cosine_basic():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_is_disabled_accepts_common_off_values():
    for value in ("", "none", "OFF", "0", "false", None):
        assert is_disabled(value)
    assert not is_disabled("BAAI/bge-small-zh-v1.5")


def test_builders_return_none_when_disabled():
    # 关闭时不应触发任何模型加载
    assert build_embedder("none") is None
    assert build_reranker("") is None
