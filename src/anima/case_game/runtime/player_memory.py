"""Persistent, tenant-scoped player memory for the Sherlock case game.

This module is an orchestration adapter over the existing Persona V4 memory
contracts. It deliberately reuses the production repository, embedding, and
commit planner instead of introducing a second memory implementation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence

from anima.persona.contracts.pack import LoadedPack
from anima.persona.contracts.schemas import MemoryOp, MemoryRecord
from anima.persona.runtime.memory import CommitPlan, plan_memory_commits

PLAYER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
DEFAULT_MEMORY_TOP_K = 5
# Product profile memory uses structured routing and relative BGE ranking.
# Absolute BGE scores are intentionally not filtered across tasks.
DEFAULT_MEMORY_MIN_SCORE: float | None = None

PLAYER_MEMORY_PREDICATE_LABELS = {
    "preferred_name": "称呼 姓名 名字",
    "user_interest": "兴趣 关注的问题",
    "user_location": "所在地 居住地 来自哪里",
    "occupation": "职业 工作 谋生",
    "favorite_topic": "喜欢的话题 偏好话题 闲谈",
    "reading_habit": "阅读习惯 读书习惯",
    "visit_reason": "来访原因 来意",
    "companion_info": "同行者 陪同者",
}

PLAYER_MEMORY_QUERY_CUES = {
    "preferred_name": ("称呼", "姓名", "名字", "叫什么", "叫我", "名叫"),
    "user_interest": ("兴趣", "感兴趣", "关注", "哪类案件"),
    "user_location": ("所在地", "居住地", "住哪里", "住在哪", "来自", "搬到"),
    "occupation": ("职业", "工作", "谋生", "哪一行"),
    "favorite_topic": ("话题", "闲谈", "爱听你讲", "喜欢聊"),
    "reading_habit": ("阅读", "读书", "读侦探小说"),
    "visit_reason": ("来意", "来访", "为什么来", "为何来"),
    "companion_info": ("同行", "陪着", "一起来", "陪同"),
}


class PlayerMemoryError(RuntimeError):
    """The persistent player-memory path could not complete safely."""


class MemoryRepository(Protocol):
    def create_user(
        self,
        user_id: str,
        api_key_hash: str,
        *,
        is_eval: bool = False,
    ) -> None: ...

    def search_memories(
        self,
        user_id: str,
        persona_id: str,
        embedding: Sequence[float],
        top_k: int,
    ) -> list[tuple[MemoryRecord, float]]: ...

    def commit_memory(self, user_id: str, persona_id: str, planner, *, now: str): ...

    def audit_rejected_op(
        self,
        user_id: str,
        persona_id: str,
        op: dict,
        reason: str,
    ) -> None: ...


class MemoryEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class MemoryRetrieval:
    records: tuple[MemoryRecord, ...]
    scores: tuple[float, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record.memory_id for record in self.records)


@dataclass(frozen=True)
class MemoryCommitResult:
    requested: int
    committed: int
    rejected: tuple[dict, ...]


@dataclass(frozen=True)
class PreparedMemoryCommit:
    """A memory mutation planned inside the storage transaction that commits a game turn."""

    requested: int
    user_id: str
    persona_id: str
    now: str
    planner: Callable[
        [Sequence[MemoryRecord]],
        tuple[CommitPlan, dict[str, Sequence[float]]],
    ]

    def result_from_plan(self, plan: CommitPlan) -> MemoryCommitResult:
        return MemoryCommitResult(
            requested=self.requested,
            committed=len(plan.commits),
            rejected=tuple(
                {"op": rejected.op.to_dict(), "reason": rejected.reason}
                for rejected in plan.rejected
            ),
        )


def validate_player_id(player_id: str) -> str:
    value = player_id.strip()
    if not PLAYER_ID_RE.fullmatch(value):
        raise ValueError(
            "player_id must contain 16-128 ASCII letters, digits, underscores, or hyphens"
        )
    return value


def render_player_memory_text(record: MemoryRecord) -> str:
    label = PLAYER_MEMORY_PREDICATE_LABELS.get(
        record.predicate,
        record.predicate.replace("_", " "),
    )
    return f"{record.predicate.replace('_', ' ')} {label} {record.object}"


def route_player_memory_predicates(query: str) -> tuple[str, ...]:
    normalized = "".join(query.casefold().split())
    routed = tuple(
        predicate
        for predicate, cues in PLAYER_MEMORY_QUERY_CUES.items()
        if any("".join(cue.casefold().split()) in normalized for cue in cues)
    )
    if routed:
        return routed
    if "记得" in normalized and "我" in normalized:
        return tuple(PLAYER_MEMORY_PREDICATE_LABELS)
    return ()


class PlayerMemoryService:
    """Retrieve and commit a pseudonymous player's allowlisted profile facts."""

    def __init__(
        self,
        *,
        repository: MemoryRepository,
        embedder: MemoryEmbedder,
        pack: LoadedPack,
        top_k: int = DEFAULT_MEMORY_TOP_K,
        min_score: float | None = DEFAULT_MEMORY_MIN_SCORE,
        now_factory: Callable[[], str] | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("memory top_k must be >= 1")
        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("memory min_score must be in [-1, 1] or None")
        self.repository = repository
        self.embedder = embedder
        self.pack = pack
        self.top_k = top_k
        self.min_score = min_score
        self.now_factory = now_factory or (
            lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        )

    @property
    def runtime_info(self) -> dict:
        return {
            "memory_enabled": True,
            "memory_backend": "postgresql_pgvector",
            "memory_scope": "pseudonymous_player_profile",
            "memory_top_k": self.top_k,
            "memory_min_score": self.min_score,
            "memory_retrieval_policy": "structured_route_then_pgvector_rank",
            "memory_profile_provenance": "source_session_id+source_turn_id",
        }

    def ensure_player(self, player_id: str) -> str:
        value = validate_player_id(player_id)
        digest = hashlib.sha256(f"anima-case-player:{value}".encode("utf-8")).hexdigest()
        user_id = f"case_player_{digest[:32]}"
        try:
            self.repository.create_user(user_id, digest)
        except Exception as exc:
            raise PlayerMemoryError("player memory identity could not be persisted") from exc
        return user_id

    def retrieve(self, player_id: str, query: str) -> MemoryRetrieval:
        user_id = self.ensure_player(player_id)
        routed_predicates = frozenset(route_player_memory_predicates(query))
        if not query.strip() or not routed_predicates:
            return MemoryRetrieval(records=(), scores=())
        try:
            query_vector = self.embedder.embed_query(query)
            hits = self.repository.search_memories(
                user_id,
                self.pack.manifest.persona_id,
                query_vector,
                max(self.top_k, len(self.pack.safety.memory_predicate_allowlist)),
            )
        except Exception as exc:
            raise PlayerMemoryError("player memory retrieval failed") from exc
        retained = tuple(
            (record, score)
            for record, score in hits
            if record.predicate in routed_predicates
            and (self.min_score is None or score >= self.min_score)
        )[: self.top_k]
        return MemoryRetrieval(
            records=tuple(record for record, _score in retained),
            scores=tuple(float(score) for _record, score in retained),
        )

    def commit(
        self,
        player_id: str,
        ops: Sequence[MemoryOp],
        *,
        source_message_id: str,
        source_session_id: str | None = None,
        source_turn_id: str | None = None,
    ) -> MemoryCommitResult:
        if not ops:
            return MemoryCommitResult(requested=0, committed=0, rejected=())
        prepared = self.prepare_commit(
            player_id,
            ops,
            source_message_id=source_message_id,
            source_session_id=source_session_id,
            source_turn_id=source_turn_id,
        )
        return self.apply_prepared(prepared)

    def prepare_commit(
        self,
        player_id: str,
        ops: Sequence[MemoryOp],
        *,
        source_message_id: str,
        source_session_id: str | None = None,
        source_turn_id: str | None = None,
    ) -> PreparedMemoryCommit:
        """Bind untrusted ops now, but defer read→plan→write to the turn transaction."""

        if not ops:
            raise ValueError("cannot prepare an empty memory commit")
        if (source_session_id is None) != (source_turn_id is None):
            raise ValueError(
                "source_session_id and source_turn_id must either both be present or both be null"
            )
        user_id = self.ensure_player(player_id)
        persona_id = self.pack.manifest.persona_id
        now = self.now_factory()
        rebound = tuple(
            MemoryOp(
                op=op.op,
                subject="authenticated_user",
                predicate=op.predicate,
                object=op.object,
                source_message_id=source_message_id,
            )
            for op in ops
        )
        counter = {"value": 0}

        def mint() -> str:
            counter["value"] += 1
            return f"{source_message_id}:mem:{counter['value']}"

        def planner(active: Sequence[MemoryRecord]):
            plan = plan_memory_commits(
                rebound,
                user_id=user_id,
                persona_id=persona_id,
                allowlist=self.pack.safety.memory_predicate_allowlist,
                source_message_id=source_message_id,
                source_session_id=source_session_id,
                source_turn_id=source_turn_id,
                active_memories=active,
                id_factory=mint,
                now=now,
            )
            embeddings = {
                commit.new_record.memory_id: self.embedder.embed(
                    [render_player_memory_text(commit.new_record)]
                )[0]
                for commit in plan.commits
                if commit.new_record is not None
            }
            return plan, embeddings

        return PreparedMemoryCommit(
            requested=len(ops),
            user_id=user_id,
            persona_id=persona_id,
            now=now,
            planner=planner,
        )

    def apply_prepared(self, prepared: PreparedMemoryCommit) -> MemoryCommitResult:
        """Compatibility path for callers without a shared game-state transaction."""

        try:
            plan = self.repository.commit_memory(
                prepared.user_id,
                prepared.persona_id,
                prepared.planner,
                now=prepared.now,
            )
            for rejected in plan.rejected:
                self.repository.audit_rejected_op(
                    prepared.user_id,
                    prepared.persona_id,
                    rejected.op.to_dict(),
                    rejected.reason,
                )
        except Exception as exc:
            raise PlayerMemoryError("player memory commit failed") from exc
        return prepared.result_from_plan(plan)

    def reject(
        self,
        player_id: str,
        ops: Sequence[MemoryOp],
        *,
        reason: str,
    ) -> MemoryCommitResult:
        if not ops:
            return MemoryCommitResult(requested=0, committed=0, rejected=())
        user_id = self.ensure_player(player_id)
        persona_id = self.pack.manifest.persona_id
        rejected = tuple({"op": op.to_dict(), "reason": reason} for op in ops)
        try:
            for item in rejected:
                self.repository.audit_rejected_op(
                    user_id,
                    persona_id,
                    item["op"],
                    item["reason"],
                )
        except Exception as exc:
            raise PlayerMemoryError("player memory rejection audit failed") from exc
        return MemoryCommitResult(
            requested=len(ops),
            committed=0,
            rejected=rejected,
        )

    def close(self) -> None:
        close = getattr(self.repository, "close", None)
        if callable(close):
            close()
