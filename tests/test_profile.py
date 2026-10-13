"""用户长期记忆测试。"""

from __future__ import annotations

from app.profile import UserProfileStore, worth_extracting


def test_add_and_get_facts(tmp_path):
    store = UserProfileStore(tmp_path / "p.json")
    store.add_facts("u1", ["常驻北京", "不吃辣"])
    assert store.facts("u1") == ["常驻北京", "不吃辣"]


def test_dedup_and_trim(tmp_path):
    store = UserProfileStore(tmp_path / "p.json")
    store.add_facts("u1", ["a"])
    store.add_facts("u1", ["a", "b"])
    assert store.facts("u1") == ["a", "b"]


def test_persist_across_instances(tmp_path):
    path = tmp_path / "p.json"
    UserProfileStore(path).add_facts("u1", ["常驻北京"])
    assert UserProfileStore(path).facts("u1") == ["常驻北京"]


def test_clear(tmp_path):
    store = UserProfileStore(tmp_path / "p.json")
    store.add_facts("u1", ["a"])
    store.clear("u1")
    assert store.facts("u1") == []


def test_worth_extracting():
    assert worth_extracting("我常驻北京")
    assert worth_extracting("我不吃辣")
    assert not worth_extracting("帮我查一下机票")
