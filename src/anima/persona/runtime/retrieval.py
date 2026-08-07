"""Lore and memory retrieval.

Embedding-model-agnostic: callers hand in any object with
`embed(texts) -> list[list[float]]`. Vectors are L2-normalized here so
similarity is a plain dot product. Memory indexing refuses rows outside the
authenticated (user, persona) tenant or in non-active status — vector search
can never bypass tenant filtering. Standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from anima.persona.contracts.schemas import LoreFact, MemoryRecord


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class TenantViolationError(RuntimeError):
    """A memory row outside the authenticated tenant reached the retrieval layer."""


@dataclass(frozen=True)
class VectorIndex:
    ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RetrievedItem:
    id: str
    score: float
    rank: int


def render_fact_text(fact: LoreFact) -> str:
    parts = [
        fact.subject.replace("_", " "),
        fact.predicate.replace("_", " "),
        fact.object,
        *fact.aliases,
    ]
    return " ".join(parts)


def render_memory_text(record: MemoryRecord) -> str:
    return f"{record.predicate.replace('_', ' ')} {record.object}"


def build_lore_index(facts: Sequence[LoreFact], embedder: Embedder) -> VectorIndex:
    """Index only facts the persona actually knows; unknown facts exist solely
    as eval fixtures and must never enter the runtime context."""

    known = [fact for fact in facts if fact.known_by_persona]
    vectors = embedder.embed([render_fact_text(fact) for fact in known])
    return VectorIndex(
        ids=tuple(fact.fact_id for fact in known),
        vectors=tuple(_normalize(vector) for vector in vectors),
    )


def build_memory_index(
    records: Sequence[MemoryRecord],
    embedder: Embedder,
    *,
    user_id: str,
    persona_id: str,
) -> VectorIndex:
    for record in records:
        if record.user_id != user_id or record.persona_id != persona_id:
            raise TenantViolationError(
                f"memory {record.memory_id} belongs to ({record.user_id}, {record.persona_id}), "
                f"expected ({user_id}, {persona_id})"
            )
        if record.status != "active":
            raise TenantViolationError(
                f"memory {record.memory_id} has status {record.status}, expected active"
            )
    vectors = embedder.embed([render_memory_text(record) for record in records])
    return VectorIndex(
        ids=tuple(record.memory_id for record in records),
        vectors=tuple(_normalize(vector) for vector in vectors),
    )


def rank_by_similarity(
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    *,
    top_k: int,
    min_score: float | None,
) -> tuple[RetrievedItem, ...]:
    query = _normalize(query_vector)
    scored = sorted(
        ((sum(q * v for q, v in zip(query, vector)), id_) for id_, vector in zip(ids, vectors)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    results: list[RetrievedItem] = []
    for score, id_ in scored[:top_k]:
        if min_score is not None and score < min_score:
            continue
        results.append(RetrievedItem(id=id_, score=score, rank=len(results) + 1))
    return tuple(results)


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(v / norm for v in vector)
