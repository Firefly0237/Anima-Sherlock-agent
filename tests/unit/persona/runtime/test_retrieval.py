"""Tests for anima.persona.runtime.retrieval."""

from __future__ import annotations

import math

import pytest

from anima.persona.contracts.schemas import LoreFact, MemoryRecord
from anima.persona.runtime.retrieval import (
    TenantViolationError,
    build_lore_index,
    build_memory_index,
    rank_by_similarity,
    render_fact_text,
)


class KeywordEmbedder:
    """Deterministic test stub: 3-dim vector = keyword hit counts (test input only)."""

    keywords = ("红茶", "站台", "事故")

    def embed(self, texts):
        vectors = []
        for text in texts:
            vec = [float(text.count(k)) for k in self.keywords]
            vectors.append(vec if any(vec) else [1.0, 1.0, 1.0])
        return vectors


def _fact(fact_id: str, obj: str, *, known: bool = True) -> LoreFact:
    return LoreFact.from_mapping(
        {
            "fact_id": fact_id,
            "subject": "landu_station",
            "predicate": "has_feature",
            "object": obj,
            "aliases": [],
            "valid_from": None,
            "valid_to": None,
            "known_by_persona": known,
            "persona_response": f"我确认{obj}，这是可核查的记录。" if known else None,
            "answer_slots": [[obj], ["可核查"]] if known else [],
            "boundary_forbidden_claims": [] if known else [f"我亲自确认{obj}"],
            "source_ref": "bible_v1",
            "review_status": "draft",
        }
    )


def _memory(
    memory_id: str, obj: str, *, user_id: str = "user-a", status: str = "active"
) -> MemoryRecord:
    return MemoryRecord.from_mapping(
        {
            "memory_id": memory_id,
            "user_id": user_id,
            "persona_id": "test_persona",
            "subject": "authenticated_user",
            "predicate": "favorite_tea",
            "object": obj,
            "valid_from": None,
            "valid_to": None,
            "source_message_id": "msg-0",
            "confidence": 1.0,
            "status": status,
            "created_at": "t0",
            "updated_at": "t0",
        }
    )


FACTS = [
    _fact("lore_000001", "站内茶点铺卖青芜红茶"),
    _fact("lore_000002", "三站台是换乘主站台"),
    _fact("lore_000003", "雾岭七号桥发生过事故"),
]


def test_rank_by_similarity_orders_and_limits():
    embedder = KeywordEmbedder()
    index = build_lore_index(FACTS, embedder)
    [query] = embedder.embed(["有红茶喝吗，红茶"])
    results = rank_by_similarity(index.ids, index.vectors, query, top_k=2, min_score=None)
    assert [r.id for r in results][0] == "lore_000001"
    assert len(results) == 2
    assert results[0].rank == 1 and results[1].rank == 2
    assert results[0].score > results[1].score


def test_min_score_filters():
    embedder = KeywordEmbedder()
    index = build_lore_index(FACTS, embedder)
    [query] = embedder.embed(["红茶"])
    results = rank_by_similarity(index.ids, index.vectors, query, top_k=8, min_score=0.99)
    assert [r.id for r in results] == ["lore_000001"]


def test_vectors_are_normalized():
    embedder = KeywordEmbedder()
    index = build_lore_index(FACTS, embedder)
    for vec in index.vectors:
        assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-9)


def test_unknown_facts_excluded_from_index():
    facts = FACTS + [_fact("lore_000009", "站长不知道的秘密红茶", known=False)]
    index = build_lore_index(facts, KeywordEmbedder())
    assert "lore_000009" not in index.ids


def test_render_fact_text_contains_object_and_aliases():
    fact = LoreFact.from_mapping(
        {
            "fact_id": "lore_000010",
            "subject": "jihong_line",
            "predicate": "terminus_of",
            "object": "白石城",
            "aliases": ["西端终点是白石城"],
            "valid_from": None,
            "valid_to": None,
            "known_by_persona": True,
            "persona_response": "我确认终点是白石城，这条线路记录可以核查。",
            "answer_slots": [["白石城"], ["线路记录"]],
            "boundary_forbidden_claims": [],
            "source_ref": "bible_v1",
            "review_status": "draft",
        }
    )
    text = render_fact_text(fact)
    assert "白石城" in text and "西端终点是白石城" in text


def test_memory_index_enforces_tenant():
    records = [_memory("mem-1", "青芜红茶"), _memory("mem-2", "苦茶", user_id="user-b")]
    with pytest.raises(TenantViolationError):
        build_memory_index(records, KeywordEmbedder(), user_id="user-a", persona_id="test_persona")


def test_memory_index_rejects_non_active_rows():
    records = [_memory("mem-1", "青芜红茶", status="superseded")]
    with pytest.raises(TenantViolationError):
        build_memory_index(records, KeywordEmbedder(), user_id="user-a", persona_id="test_persona")


def test_memory_index_happy_path():
    records = [_memory("mem-1", "青芜红茶"), _memory("mem-3", "站台见闻")]
    index = build_memory_index(
        records, KeywordEmbedder(), user_id="user-a", persona_id="test_persona"
    )
    [query] = KeywordEmbedder().embed(["红茶"])
    results = rank_by_similarity(index.ids, index.vectors, query, top_k=5, min_score=None)
    assert results[0].id == "mem-1"
