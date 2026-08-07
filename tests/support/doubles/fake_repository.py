"""In-memory Repository double for unit tests.

Mirrors PostgresRepository semantics exactly: tenant filtering before vector
ranking, atomic per-conversation seq, supersede/delete via apply_commit_plan.
It reuses the production commit-plan objects — no business logic is
reimplemented here, only storage.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from typing import Any, Sequence

from anima.persona.contracts.schemas import MemoryRecord
from anima.persona.runtime.memory import CommitPlan
from anima.serve.core.repositories import ConversationRow


class FakeRepository:
    def __init__(self) -> None:
        self.users: dict[str, str] = {}  # api_key_hash -> user_id
        self.eval_users: set[str] = set()
        self.conversations: dict[str, ConversationRow] = {}
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.memories: dict[str, MemoryRecord] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.audits: list[dict[str, Any]] = []
        self.traces: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []
        self._tenant_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def ping(self) -> bool:
        return True

    def create_user(self, user_id: str, api_key_hash: str, *, is_eval: bool = False) -> None:
        self.users[api_key_hash] = user_id
        if is_eval:
            self.eval_users.add(user_id)

    def user_id_for_key(self, api_key_hash: str) -> str | None:
        return self.users.get(api_key_hash)

    def create_conversation(self, row: ConversationRow) -> None:
        self.conversations[row.conversation_id] = row
        self.messages.setdefault(row.conversation_id, [])

    def get_conversation(self, conversation_id: str) -> ConversationRow | None:
        return self.conversations.get(conversation_id)

    def mark_conversation_deleted(self, conversation_id: str) -> None:
        row = self.conversations[conversation_id]
        self.conversations[conversation_id] = replace(row, deleted=True)

    def append_message(self, conversation_id: str, message_id: str, role: str, content: str) -> int:
        msgs = self.messages.setdefault(conversation_id, [])
        seq = (msgs[-1]["seq"] + 1) if msgs else 1
        msgs.append({"message_id": message_id, "seq": seq, "role": role, "content": content})
        return seq

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return [dict(m) for m in self.messages.get(conversation_id, [])]

    def active_memories(self, user_id: str, persona_id: str) -> list[MemoryRecord]:
        return [
            r
            for r in self.memories.values()
            if r.user_id == user_id and r.persona_id == persona_id and r.status == "active"
        ]

    def search_memories(
        self, user_id: str, persona_id: str, embedding: Sequence[float], top_k: int
    ) -> list[tuple[MemoryRecord, float]]:
        # tenant + status filter FIRST, then vector ranking (mirrors the SQL)
        candidates = [
            r
            for r in self.memories.values()
            if r.user_id == user_id and r.persona_id == persona_id and r.status == "active"
        ]
        scored = [
            (record, _cosine(embedding, self.embeddings.get(record.memory_id, [])))
            for record in candidates
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self.memories.get(memory_id)

    def apply_commit_plan(self, plan: CommitPlan, embeddings: dict[str, Sequence[float]]) -> None:
        for commit in plan.commits:
            if commit.superseded_id is not None:
                old = self.memories[commit.superseded_id]
                self.memories[commit.superseded_id] = replace(
                    old, status="superseded", updated_at=plan.now
                )
            if commit.deleted_id is not None:
                old = self.memories[commit.deleted_id]
                self.memories[commit.deleted_id] = replace(
                    old, status="deleted", updated_at=plan.now
                )
            if commit.new_record is not None:
                self.memories[commit.new_record.memory_id] = commit.new_record
                if commit.new_record.memory_id in embeddings:
                    self.embeddings[commit.new_record.memory_id] = list(
                        embeddings[commit.new_record.memory_id]
                    )

    def commit_memory(self, user_id: str, persona_id: str, planner, *, now: str):
        # mirror PostgresRepository: per-tenant lock spans read -> plan -> apply
        with self._locks_guard:
            lock = self._tenant_locks.setdefault(f"{user_id}|{persona_id}", threading.Lock())
        with lock:
            active = self.active_memories(user_id, persona_id)
            plan, embeddings = planner(active)
            self.apply_commit_plan(plan, embeddings)
        return plan

    def audit_rejected_op(
        self, user_id: str, persona_id: str, op: dict[str, Any], reason: str
    ) -> None:
        self.audits.append(
            {"user_id": user_id, "persona_id": persona_id, "op": op, "reason": reason}
        )

    def insert_trace(self, request_id: str, payload: dict[str, Any]) -> None:
        self.traces[request_id] = payload

    def get_trace(self, request_id: str) -> dict[str, Any] | None:
        return self.traces.get(request_id)

    def insert_feedback(
        self,
        user_id: str,
        conversation_id: str | None,
        message_id: str | None,
        kind: str,
        comment: str | None,
    ) -> None:
        self.feedback.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "kind": kind,
                "comment": comment,
            }
        )

    def list_feedback_for_eval(self, limit: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.feedback if row.get("user_id") not in self.eval_users][
            :limit
        ]

    # convenience for tests
    def seed_memory(self, record: MemoryRecord, embedding: Sequence[float]) -> None:
        self.memories[record.memory_id] = record
        self.embeddings[record.memory_id] = list(embedding)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
