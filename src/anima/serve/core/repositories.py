"""PostgreSQL/pgvector persistence for the persona API.

Tenant rule: every memory query filters user_id AND persona_id in SQL
BEFORE any vector ranking — similarity search can never cross tenants.
Message ordering: per-conversation seq is assigned inside the insert statement
itself, so concurrent writers cannot produce duplicate or out-of-order turns.

The pack lore store deliberately stays in-process (packs are the canonical,
hash-versioned artifact; see decision record R3) — PostgreSQL owns user state:
conversations, messages, memories, audits, traces, feedback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from anima.case_game.runtime.game_memory import derive_familiarity_tier
from anima.persona.contracts.schemas import MemoryRecord
from anima.persona.runtime.memory import CommitPlan

MIGRATION_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    api_key_hash TEXT UNIQUE NOT NULL,
    is_eval BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_eval BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    persona_id TEXT NOT NULL,
    persona_version TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS game_sessions (
    session_id TEXT PRIMARY KEY,
    player_id TEXT,
    case_id TEXT NOT NULL,
    state_json JSONB NOT NULL,
    state_version BIGINT NOT NULL DEFAULT 0 CHECK (state_version >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS game_turns (
    turn_id TEXT PRIMARY KEY,
    request_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL REFERENCES game_sessions(session_id),
    state_version_before BIGINT NOT NULL,
    state_version_after BIGINT NOT NULL,
    proposal_json JSONB NOT NULL,
    observation_json JSONB NOT NULL,
    turn_json JSONB NOT NULL,
    before_hash TEXT NOT NULL,
    after_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status = 'committed'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (state_version_after = state_version_before + 1)
);
CREATE INDEX IF NOT EXISTS game_turns_session_version_idx
    ON game_turns (session_id, state_version_after);

CREATE TABLE IF NOT EXISTS game_trace_quarantine (
    quarantine_id TEXT PRIMARY KEY,
    request_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL REFERENCES game_sessions(session_id),
    player_scope_sha256 TEXT,
    case_id TEXT NOT NULL,
    failure_stage TEXT NOT NULL,
    trace_json JSONB NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'quarantine'
        CHECK (review_status IN ('quarantine', 'approved_sft', 'approved_dpo', 'rejected')),
    attempt_count BIGINT NOT NULL DEFAULT 1 CHECK (attempt_count >= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS game_trace_quarantine_status_idx
    ON game_trace_quarantine (review_status, created_at);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, seq)
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    persona_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    source_message_id TEXT NOT NULL,
    source_session_id TEXT,
    source_turn_id TEXT,
    confidence DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'deleted')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    embedding vector(512),
    CHECK ((source_session_id IS NULL) = (source_turn_id IS NULL))
);
ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_session_id TEXT;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_turn_id TEXT;
CREATE INDEX IF NOT EXISTS memories_tenant_idx ON memories (user_id, persona_id, status);
-- : at most one active row per (user, persona, predicate). Defense in depth
-- behind the tenant advisory lock in commit_memory: even a bypassed lock cannot
-- land two concurrent active rows.
CREATE UNIQUE INDEX IF NOT EXISTS memories_active_predicate_uq
    ON memories (user_id, persona_id, predicate) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS memory_audit (
    audit_id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    op JSONB NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS traces (
    request_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT REFERENCES conversations(conversation_id),
    message_id TEXT REFERENCES messages(message_id),
    kind TEXT NOT NULL CHECK (kind IN ('up', 'down', 'ooc', 'fact_error', 'safety_error', 'other')),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class ConversationRow:
    conversation_id: str
    user_id: str
    persona_id: str
    persona_version: str
    deleted: bool


class GameSessionNotFoundError(RuntimeError):
    pass


class GameStateVersionConflictError(RuntimeError):
    pass


class GameRequestConflictError(RuntimeError):
    pass


class Repository(Protocol):
    def ping(self) -> bool: ...
    def create_user(self, user_id: str, api_key_hash: str, *, is_eval: bool = False) -> None: ...
    def user_id_for_key(self, api_key_hash: str) -> str | None: ...
    def create_conversation(self, row: ConversationRow) -> None: ...
    def get_conversation(self, conversation_id: str) -> ConversationRow | None: ...
    def mark_conversation_deleted(self, conversation_id: str) -> None: ...
    def append_message(
        self, conversation_id: str, message_id: str, role: str, content: str
    ) -> int: ...
    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]: ...
    def create_game_session(
        self,
        *,
        session_id: str,
        player_id: str | None,
        case_id: str,
        state_json: Mapping[str, Any],
    ) -> dict[str, Any]: ...
    def get_game_session(self, session_id: str) -> dict[str, Any] | None: ...
    def get_game_turn_by_request_id(self, request_id: str) -> dict[str, Any] | None: ...
    def list_game_turns(self, session_id: str) -> list[dict[str, Any]]: ...
    def player_game_memory_context(
        self, player_id: str, case_id: str, *, recent_limit: int = 5
    ) -> dict[str, Any]: ...
    def quarantine_failed_game_turn(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    def list_committed_game_turns_for_export(self, *, limit: int) -> list[dict[str, Any]]: ...
    def list_failed_game_turns_for_export(self, *, limit: int) -> list[dict[str, Any]]: ...
    def commit_game_turn(self, **kwargs) -> tuple[dict, dict, Any, bool]: ...
    def active_memories(self, user_id: str, persona_id: str) -> list[MemoryRecord]: ...
    def search_memories(
        self, user_id: str, persona_id: str, embedding: Sequence[float], top_k: int
    ) -> list[tuple[MemoryRecord, float]]: ...
    def get_memory(self, memory_id: str) -> MemoryRecord | None: ...
    def seed_memory(self, record: MemoryRecord, embedding: Sequence[float]) -> None: ...
    def apply_commit_plan(
        self, plan: CommitPlan, embeddings: dict[str, Sequence[float]]
    ) -> None: ...
    def commit_memory(self, user_id: str, persona_id: str, planner, *, now: str) -> CommitPlan: ...
    def audit_rejected_op(
        self, user_id: str, persona_id: str, op: dict[str, Any], reason: str
    ) -> None: ...
    def insert_trace(self, request_id: str, payload: dict[str, Any]) -> None: ...
    def get_trace(self, request_id: str) -> dict[str, Any] | None: ...
    def insert_feedback(
        self,
        user_id: str,
        conversation_id: str | None,
        message_id: str | None,
        kind: str,
        comment: str | None,
    ) -> None: ...
    def list_feedback_for_eval(self, limit: int) -> list[dict[str, Any]]: ...


class PostgresRepository:
    """psycopg3 implementation; every method uses a pooled connection."""

    def __init__(self, dsn: str, *, pool_size: int = 8) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(dsn, min_size=1, max_size=pool_size, open=True)

    def migrate(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(MIGRATION_SQL)

    def close(self) -> None:
        self._pool.close()

    def ping(self) -> bool:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def create_user(self, user_id: str, api_key_hash: str, *, is_eval: bool = False) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO users (user_id, api_key_hash, is_eval) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id) DO UPDATE SET is_eval = users.is_eval OR EXCLUDED.is_eval",
                (user_id, api_key_hash, is_eval),
            )

    def user_id_for_key(self, api_key_hash: str) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT user_id FROM users WHERE api_key_hash = %s", (api_key_hash,)
            ).fetchone()
        return row[0] if row else None

    def create_conversation(self, row: ConversationRow) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, user_id, persona_id, persona_version) "
                "VALUES (%s, %s, %s, %s)",
                (row.conversation_id, row.user_id, row.persona_id, row.persona_version),
            )

    def get_conversation(self, conversation_id: str) -> ConversationRow | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT conversation_id, user_id, persona_id, persona_version, deleted "
                "FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationRow(*row)

    def mark_conversation_deleted(self, conversation_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE conversations SET deleted = TRUE WHERE conversation_id = %s",
                (conversation_id,),
            )

    def append_message(self, conversation_id: str, message_id: str, role: str, content: str) -> int:
        # A transaction-scoped advisory lock keyed on the conversation serializes
        # writers on the SAME conversation (different conversations never block),
        # so seq is contiguous and unique with no dup/reorder under concurrency.
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (conversation_id,))
                row = conn.execute(
                    "INSERT INTO messages (message_id, conversation_id, seq, role, content) "
                    "SELECT %s, %s, COALESCE(MAX(seq), 0) + 1, %s, %s FROM messages "
                    "WHERE conversation_id = %s RETURNING seq",
                    (message_id, conversation_id, role, content, conversation_id),
                ).fetchone()
            return int(row[0])

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT message_id, seq, role, content FROM messages WHERE conversation_id = %s ORDER BY seq",
                (conversation_id,),
            ).fetchall()
        return [{"message_id": r[0], "seq": r[1], "role": r[2], "content": r[3]} for r in rows]

    def create_game_session(
        self,
        *,
        session_id: str,
        player_id: str | None,
        case_id: str,
        state_json: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO game_sessions (session_id, player_id, case_id, state_json) "
                "VALUES (%s, %s, %s, %s) "
                "RETURNING session_id, player_id, case_id, state_json, state_version",
                (session_id, player_id, case_id, json.dumps(dict(state_json), ensure_ascii=False)),
            ).fetchone()
        return self._game_session_mapping(row)

    def get_game_session(self, session_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT session_id, player_id, case_id, state_json, state_version "
                "FROM game_sessions WHERE session_id = %s",
                (session_id,),
            ).fetchone()
        return self._game_session_mapping(row) if row is not None else None

    def get_game_turn_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT turn_id, request_id, session_id, state_version_before, "
                "state_version_after, proposal_json, observation_json, turn_json, "
                "before_hash, after_hash, status FROM game_turns WHERE request_id = %s",
                (request_id,),
            ).fetchone()
        return self._game_turn_mapping(row) if row is not None else None

    def list_game_turns(self, session_id: str) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM game_sessions WHERE session_id = %s", (session_id,)
            ).fetchone()
            if exists is None:
                raise GameSessionNotFoundError(f"unknown session_id: {session_id}")
            rows = conn.execute(
                "SELECT turn_id, request_id, session_id, state_version_before, "
                "state_version_after, proposal_json, observation_json, turn_json, "
                "before_hash, after_hash, status FROM game_turns "
                "WHERE session_id = %s ORDER BY state_version_after",
                (session_id,),
            ).fetchall()
        return [self._game_turn_mapping(row) for row in rows]

    def player_game_memory_context(
        self,
        player_id: str,
        case_id: str,
        *,
        recent_limit: int = 5,
    ) -> dict[str, Any]:
        """Return tenant- and case-scoped, non-authoritative game memory.

        Only committed turn metadata is returned. Player text, model answers,
        tool arguments, evidence, and hidden case data are deliberately absent.
        The aggregate familiarity fields may personalize narration but cannot
        participate in tool policy or state transitions.
        """

        if recent_limit < 0:
            raise ValueError("recent_limit must be >= 0")
        with self._pool.connection() as conn:
            stats = conn.execute(
                "SELECT count(gt.turn_id), "
                "count(DISTINCT gs.case_id) FILTER "
                "(WHERE COALESCE((gs.state_json->>'solved')::boolean, false)) "
                "FROM game_sessions gs "
                "LEFT JOIN game_turns gt ON gt.session_id = gs.session_id "
                "AND gt.status = 'committed' "
                "WHERE gs.player_id = %s",
                (player_id,),
            ).fetchone()
            rows = conn.execute(
                "SELECT gt.turn_id, gt.session_id, gs.case_id, "
                "gt.state_version_after, gt.turn_json "
                "FROM game_turns gt "
                "JOIN game_sessions gs ON gs.session_id = gt.session_id "
                "WHERE gs.player_id = %s AND gs.case_id = %s "
                "AND gt.status = 'committed' "
                "ORDER BY gt.created_at DESC, gt.turn_id DESC LIMIT %s",
                (player_id, case_id, recent_limit),
            ).fetchall()
        committed_turn_count = int(stats[0] or 0)
        completed_case_count = int(stats[1] or 0)
        recent_turns = []
        for row in rows:
            turn = _json_value(row[4])
            recent_turns.append(
                {
                    "turn_id": str(row[0]),
                    "session_id": str(row[1]),
                    "case_id": str(row[2]),
                    "state_version_after": int(row[3]),
                    "action": str(turn.get("action") or ""),
                    "accepted": bool(turn.get("accepted")),
                    "message_key": str(turn.get("message_key") or ""),
                    "solved": bool(turn.get("solved")),
                }
            )
        return {
            "authority": "personalization_only_no_state_or_tool_effect",
            "case_id": case_id,
            "recent_committed_turns": recent_turns,
            "familiarity": {
                "committed_turn_count": committed_turn_count,
                "completed_case_count": completed_case_count,
                "tier": derive_familiarity_tier(committed_turn_count, completed_case_count),
            },
        }

    def quarantine_failed_game_turn(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or "")
        session_id = str(payload.get("session_id") or "")
        case_id = str(payload.get("case_id") or "")
        failure_stage = str(payload.get("failure_stage") or "")
        if not all((request_id, session_id, case_id, failure_stage)):
            raise ValueError("failed game trace is missing required identity fields")
        quarantine_id = "game-trace-" + hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:32]
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self._pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO game_trace_quarantine "
                "(quarantine_id, request_id, session_id, player_scope_sha256, case_id, "
                "failure_stage, trace_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (request_id) DO UPDATE SET "
                "trace_json = EXCLUDED.trace_json, failure_stage = EXCLUDED.failure_stage, "
                "attempt_count = game_trace_quarantine.attempt_count + 1, updated_at = now() "
                "RETURNING quarantine_id, request_id, review_status, attempt_count",
                (
                    quarantine_id,
                    request_id,
                    session_id,
                    payload.get("player_scope_sha256"),
                    case_id,
                    failure_stage,
                    encoded,
                ),
            ).fetchone()
        return {
            "quarantine_id": str(row[0]),
            "request_id": str(row[1]),
            "review_status": str(row[2]),
            "attempt_count": int(row[3]),
        }

    def list_committed_game_turns_for_export(self, *, limit: int) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("export limit must be >= 1")
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT gt.turn_id, gt.request_id, gt.session_id, gs.player_id, gs.case_id, "
                "gt.state_version_before, gt.state_version_after, gt.proposal_json, "
                "gt.observation_json, gt.turn_json, gt.before_hash, gt.after_hash, gt.status "
                "FROM game_turns gt JOIN game_sessions gs ON gs.session_id = gt.session_id "
                "WHERE gt.status = 'committed' ORDER BY gt.created_at, gt.turn_id LIMIT %s",
                (limit,),
            ).fetchall()
        keys = (
            "turn_id",
            "request_id",
            "session_id",
            "player_id",
            "case_id",
            "state_version_before",
            "state_version_after",
            "proposal_json",
            "observation_json",
            "turn_json",
            "before_hash",
            "after_hash",
            "status",
        )
        return [dict(zip(keys, row)) for row in rows]

    def list_failed_game_turns_for_export(self, *, limit: int) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("export limit must be >= 1")
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT quarantine_id, request_id, session_id, player_scope_sha256, case_id, "
                "failure_stage, trace_json, review_status, attempt_count "
                "FROM game_trace_quarantine ORDER BY created_at, quarantine_id LIMIT %s",
                (limit,),
            ).fetchall()
        keys = (
            "quarantine_id",
            "request_id",
            "session_id",
            "player_scope_sha256",
            "case_id",
            "failure_stage",
            "trace_json",
            "review_status",
            "attempt_count",
        )
        return [dict(zip(keys, row)) for row in rows]

    def commit_game_turn(
        self,
        *,
        turn_id: str,
        request_id: str,
        session_id: str,
        expected_state_version: int,
        candidate_state: Mapping[str, Any],
        proposal_json: Mapping[str, Any],
        observation_json: Mapping[str, Any],
        turn_json: Mapping[str, Any],
        before_hash: str,
        after_hash: str,
        memory_intent=None,
    ) -> tuple[dict[str, Any], dict[str, Any], Any, bool]:
        """Atomically commit profile memory, candidate state, and one idempotent turn."""

        with self._pool.connection() as conn:
            with conn.transaction():
                session_row = conn.execute(
                    "SELECT session_id, player_id, case_id, state_json, state_version "
                    "FROM game_sessions WHERE session_id = %s FOR UPDATE",
                    (session_id,),
                ).fetchone()
                if session_row is None:
                    raise GameSessionNotFoundError(f"unknown session_id: {session_id}")
                duplicate = conn.execute(
                    "SELECT turn_id, request_id, session_id, state_version_before, "
                    "state_version_after, proposal_json, observation_json, turn_json, "
                    "before_hash, after_hash, status FROM game_turns WHERE request_id = %s",
                    (request_id,),
                ).fetchone()
                if duplicate is not None:
                    duplicate_mapping = self._game_turn_mapping(duplicate)
                    if duplicate_mapping["session_id"] != session_id:
                        raise GameRequestConflictError(
                            f"request_id belongs to another session: {request_id}"
                        )
                    return (
                        self._game_session_mapping(session_row),
                        duplicate_mapping,
                        None,
                        True,
                    )

                current_version = int(session_row[4])
                if current_version != expected_state_version:
                    raise GameStateVersionConflictError(
                        "state_version conflict: "
                        f"expected {expected_state_version}, actual {current_version}"
                    )

                persisted_turn = dict(turn_json)
                memory_plan = None
                if memory_intent is not None:
                    conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))",
                        (f"{memory_intent.user_id}|{memory_intent.persona_id}",),
                    )
                    active = self._active_memories_conn(
                        conn,
                        memory_intent.user_id,
                        memory_intent.persona_id,
                    )
                    memory_plan, embeddings = memory_intent.planner(active)
                    self._apply_conn(conn, memory_plan, embeddings)
                    for rejected in memory_plan.rejected:
                        conn.execute(
                            "INSERT INTO memory_audit (user_id, persona_id, op, reason) "
                            "VALUES (%s, %s, %s, %s)",
                            (
                                memory_intent.user_id,
                                memory_intent.persona_id,
                                json.dumps(rejected.op.to_dict(), ensure_ascii=False),
                                rejected.reason,
                            ),
                        )
                    memory_result = memory_intent.result_from_plan(memory_plan)
                    memory_status = (
                        "committed"
                        if memory_result.committed
                        else ("rejected" if memory_result.rejected else "no_op")
                    )
                    persisted_turn.update(
                        {
                            "memory_status": memory_status,
                            "memory_ops_requested": memory_result.requested,
                            "memory_ops_committed": memory_result.committed,
                            "memory_ops_rejected": len(memory_result.rejected),
                        }
                    )
                    trace = dict(persisted_turn.get("trace_metadata") or {})
                    trace.update(
                        {
                            "memory_commit_status": memory_status,
                            "memory_ops_requested": memory_result.requested,
                            "memory_ops_committed": memory_result.committed,
                            "memory_ops_rejected": list(memory_result.rejected),
                        }
                    )
                    persisted_turn["trace_metadata"] = trace

                next_version = current_version + 1
                persisted_turn.update(
                    {
                        "state_version_before": current_version,
                        "state_version_after": next_version,
                        "before_hash": before_hash,
                        "after_hash": after_hash,
                        "status": "committed",
                    }
                )
                updated = conn.execute(
                    "UPDATE game_sessions SET state_json = %s, state_version = %s, "
                    "updated_at = now() WHERE session_id = %s AND state_version = %s "
                    "RETURNING session_id, player_id, case_id, state_json, state_version",
                    (
                        json.dumps(dict(candidate_state), ensure_ascii=False),
                        next_version,
                        session_id,
                        current_version,
                    ),
                ).fetchone()
                if updated is None:  # pragma: no cover - row lock makes this defensive
                    raise GameStateVersionConflictError("state_version changed during commit")
                conn.execute(
                    "INSERT INTO game_turns (turn_id, request_id, session_id, "
                    "state_version_before, state_version_after, proposal_json, observation_json, "
                    "turn_json, before_hash, after_hash, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'committed')",
                    (
                        turn_id,
                        request_id,
                        session_id,
                        current_version,
                        next_version,
                        json.dumps(dict(proposal_json), ensure_ascii=False),
                        json.dumps(dict(observation_json), ensure_ascii=False),
                        json.dumps(persisted_turn, ensure_ascii=False),
                        before_hash,
                        after_hash,
                    ),
                )
                persisted_row = {
                    "turn_id": turn_id,
                    "request_id": request_id,
                    "session_id": session_id,
                    "state_version_before": current_version,
                    "state_version_after": next_version,
                    "proposal_json": dict(proposal_json),
                    "observation_json": dict(observation_json),
                    "turn_json": persisted_turn,
                    "before_hash": before_hash,
                    "after_hash": after_hash,
                    "status": "committed",
                }
            return self._game_session_mapping(updated), persisted_row, memory_plan, False

    @staticmethod
    def _game_session_mapping(row: Sequence[Any]) -> dict[str, Any]:
        return {
            "session_id": row[0],
            "player_id": row[1],
            "case_id": row[2],
            "state_json": _json_value(row[3]),
            "state_version": int(row[4]),
        }

    @staticmethod
    def _game_turn_mapping(row: Sequence[Any]) -> dict[str, Any]:
        return {
            "turn_id": row[0],
            "request_id": row[1],
            "session_id": row[2],
            "state_version_before": int(row[3]),
            "state_version_after": int(row[4]),
            "proposal_json": _json_value(row[5]),
            "observation_json": _json_value(row[6]),
            "turn_json": _json_value(row[7]),
            "before_hash": row[8],
            "after_hash": row[9],
            "status": row[10],
        }

    _MEMORY_COLUMNS = (
        "memory_id, user_id, persona_id, subject, predicate, object, valid_from, valid_to, "
        "source_message_id, source_session_id, source_turn_id, "
        "confidence, status, created_at, updated_at"
    )

    def _row_to_memory(self, row: Sequence[Any]) -> MemoryRecord:
        keys = [c.strip() for c in self._MEMORY_COLUMNS.split(",")]
        return MemoryRecord.from_mapping(dict(zip(keys, row)))

    def active_memories(self, user_id: str, persona_id: str) -> list[MemoryRecord]:
        with self._pool.connection() as conn:
            return self._active_memories_conn(conn, user_id, persona_id)

    def _active_memories_conn(self, conn, user_id: str, persona_id: str) -> list[MemoryRecord]:
        rows = conn.execute(
            f"SELECT {self._MEMORY_COLUMNS} FROM memories "
            "WHERE user_id = %s AND persona_id = %s AND status = 'active' ORDER BY created_at",
            (user_id, persona_id),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_memories(
        self, user_id: str, persona_id: str, embedding: Sequence[float], top_k: int
    ) -> list[tuple[MemoryRecord, float]]:
        vector_text = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
        with self._pool.connection() as conn:
            rows = conn.execute(
                # tenant + status filter FIRST, vector ordering second
                f"SELECT {self._MEMORY_COLUMNS}, 1 - (embedding <=> %s::vector) AS score FROM memories "
                "WHERE user_id = %s AND persona_id = %s AND status = 'active' AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector_text, user_id, persona_id, vector_text, top_k),
            ).fetchall()
        return [(self._row_to_memory(row[:-1]), float(row[-1])) for row in rows]

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {self._MEMORY_COLUMNS} FROM memories WHERE memory_id = %s", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def seed_memory(self, record: MemoryRecord, embedding: Sequence[float]) -> None:
        """Insert an evaluator fixture under tenant and uniqueness rules."""

        vector_text = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{record.user_id}|{record.persona_id}",),
                )
                conn.execute(
                    "UPDATE memories SET status = 'superseded', updated_at = %s "
                    "WHERE user_id = %s AND persona_id = %s AND predicate = %s AND status = 'active'",
                    (record.updated_at, record.user_id, record.persona_id, record.predicate),
                )
                conn.execute(
                    f"INSERT INTO memories ({self._MEMORY_COLUMNS}, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    (
                        record.memory_id,
                        record.user_id,
                        record.persona_id,
                        record.subject,
                        record.predicate,
                        record.object,
                        record.valid_from,
                        record.valid_to,
                        record.source_message_id,
                        record.source_session_id,
                        record.source_turn_id,
                        record.confidence,
                        record.status,
                        record.created_at,
                        record.updated_at,
                        vector_text,
                    ),
                )

    def apply_commit_plan(self, plan: CommitPlan, embeddings: dict[str, Sequence[float]]) -> None:
        with self._pool.connection() as conn:
            with conn.transaction():
                self._apply_conn(conn, plan, embeddings)

    def _apply_conn(self, conn, plan: CommitPlan, embeddings: dict[str, Sequence[float]]) -> None:
        # supersede/delete run before the insert so the active partial-unique
        # index never sees two active rows for one predicate mid-transaction.
        for commit in plan.commits:
            if commit.superseded_id is not None:
                conn.execute(
                    "UPDATE memories SET status = 'superseded', updated_at = %s WHERE memory_id = %s",
                    (plan.now, commit.superseded_id),
                )
            if commit.deleted_id is not None:
                conn.execute(
                    "UPDATE memories SET status = 'deleted', updated_at = %s WHERE memory_id = %s",
                    (plan.now, commit.deleted_id),
                )
            if commit.new_record is not None:
                record = commit.new_record
                embedding = embeddings.get(record.memory_id)
                vector_text = (
                    "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
                    if embedding is not None
                    else None
                )
                conn.execute(
                    f"INSERT INTO memories ({self._MEMORY_COLUMNS}, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    (
                        record.memory_id,
                        record.user_id,
                        record.persona_id,
                        record.subject,
                        record.predicate,
                        record.object,
                        record.valid_from,
                        record.valid_to,
                        record.source_message_id,
                        record.source_session_id,
                        record.source_turn_id,
                        record.confidence,
                        record.status,
                        record.created_at,
                        record.updated_at,
                        vector_text,
                    ),
                )

    def commit_memory(self, user_id: str, persona_id: str, planner, *, now: str) -> CommitPlan:
        """Serialize read→plan→apply per tenant under one transaction.

        A tenant advisory lock (like append_message's per-conversation lock)
        makes a concurrent commit for the same (user, persona) block until this
        one commits, so it reads the fresh active row and supersedes instead of
        inserting a second active row. `planner(active) -> (CommitPlan, embeddings)`.
        """

        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{user_id}|{persona_id}",)
                )
                active = self._active_memories_conn(conn, user_id, persona_id)
                plan, embeddings = planner(active)
                self._apply_conn(conn, plan, embeddings)
        return plan

    def audit_rejected_op(
        self, user_id: str, persona_id: str, op: dict[str, Any], reason: str
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO memory_audit (user_id, persona_id, op, reason) VALUES (%s, %s, %s, %s)",
                (user_id, persona_id, json.dumps(op, ensure_ascii=False), reason),
            )

    def insert_trace(self, request_id: str, payload: dict[str, Any]) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO traces (request_id, payload) VALUES (%s, %s)",
                (request_id, json.dumps(payload, ensure_ascii=False)),
            )

    def get_trace(self, request_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT payload FROM traces WHERE request_id = %s", (request_id,)
            ).fetchone()
        return row[0] if row else None

    def insert_feedback(
        self,
        user_id: str,
        conversation_id: str | None,
        message_id: str | None,
        kind: str,
        comment: str | None,
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO feedback (user_id, conversation_id, message_id, kind, comment) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, conversation_id, message_id, kind, comment),
            )

    def list_feedback_for_eval(self, limit: int) -> list[dict[str, Any]]:
        if not 1 <= limit <= 5000:
            raise ValueError("feedback export limit must be in [1, 5000]")
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT f.feedback_id, f.user_id, f.conversation_id, f.message_id, f.kind, f.comment, "
                "f.created_at, c.persona_id, m.content, m.seq, t.payload, um.content, um.seq "
                "FROM feedback f "
                "JOIN users u ON u.user_id = f.user_id AND u.is_eval = FALSE "
                "LEFT JOIN conversations c ON c.conversation_id = f.conversation_id "
                "LEFT JOIN messages m ON m.message_id = f.message_id "
                "LEFT JOIN traces t ON t.payload->>'assistant_message_id' = f.message_id "
                "LEFT JOIN messages um ON um.message_id = t.payload->>'user_message_id' "
                "ORDER BY f.feedback_id ASC LIMIT %s",
                (limit,),
            ).fetchall()
            output: list[dict[str, Any]] = []
            for row in rows:
                history: list[dict[str, Any]] = []
                if row[2] is not None and row[12] is not None:
                    history_rows = conn.execute(
                        "SELECT role, content, seq FROM messages "
                        "WHERE conversation_id = %s AND seq < %s ORDER BY seq",
                        (row[2], row[12]),
                    ).fetchall()
                    history = [
                        {"role": history_row[0], "content": history_row[1], "seq": history_row[2]}
                        for history_row in history_rows
                    ]
                output.append(
                    {
                        "feedback_id": int(row[0]),
                        "user_id_hash": hashlib.sha256(str(row[1]).encode("utf-8")).hexdigest()[
                            :16
                        ],
                        "conversation_id_hash": (
                            hashlib.sha256(str(row[2]).encode("utf-8")).hexdigest()[:16]
                            if row[2] is not None
                            else None
                        ),
                        "message_id": row[3],
                        "kind": row[4],
                        "comment": row[5],
                        "created_at": row[6].isoformat() if row[6] is not None else None,
                        "persona_id": row[7],
                        "assistant_answer": row[8],
                        "assistant_seq": row[9],
                        "trace": row[10],
                        "user_message": row[11],
                        "history_before_user": history,
                    }
                )
        return output


def _json_value(value: Any) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise TypeError("stored JSON value must be an object")
    return dict(decoded)
