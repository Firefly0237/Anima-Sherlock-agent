"""Minimal authoritative storage for case sessions and committed turns.

The deterministic engine only produces a candidate state. This module owns
the compare-and-swap boundary that makes a turn visible after rendering,
guards, optional profile-memory planning, and persistence all succeed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from anima.case_game.runtime.game_memory import derive_familiarity_tier
from anima.case_game.runtime.player_memory import (
    MemoryCommitResult,
    PlayerMemoryService,
    PreparedMemoryCommit,
)


class CaseSessionStoreError(RuntimeError):
    """The authoritative case-session store could not complete safely."""


class CaseSessionNotFoundError(CaseSessionStoreError):
    pass


class CaseStateVersionConflictError(CaseSessionStoreError):
    pass


class CaseRequestConflictError(CaseSessionStoreError):
    pass


@dataclass(frozen=True)
class StoredGameSession:
    session_id: str
    player_id: str | None
    case_id: str
    state_json: Mapping[str, Any]
    state_version: int


@dataclass(frozen=True)
class StoredGameTurn:
    turn_id: str
    request_id: str
    session_id: str
    state_version_before: int
    state_version_after: int
    proposal_json: Mapping[str, Any]
    observation_json: Mapping[str, Any]
    turn_json: Mapping[str, Any]
    before_hash: str
    after_hash: str
    status: str


@dataclass(frozen=True)
class GameTurnCommit:
    turn_id: str
    request_id: str
    session_id: str
    expected_state_version: int
    candidate_state: Mapping[str, Any]
    proposal_json: Mapping[str, Any]
    observation_json: Mapping[str, Any]
    turn_json: Mapping[str, Any]
    before_hash: str
    after_hash: str
    memory_intent: PreparedMemoryCommit | None = None


@dataclass(frozen=True)
class GameTurnCommitResult:
    session: StoredGameSession
    turn: StoredGameTurn
    idempotent_replay: bool
    memory_result: MemoryCommitResult | None = None


class CaseSessionStore(Protocol):
    @property
    def runtime_info(self) -> Mapping[str, Any]: ...

    def create_session(
        self,
        *,
        session_id: str,
        player_id: str | None,
        case_id: str,
        state_json: Mapping[str, Any],
    ) -> StoredGameSession: ...

    def get_session(self, session_id: str) -> StoredGameSession | None: ...

    def get_turn_by_request_id(self, request_id: str) -> StoredGameTurn | None: ...

    def list_turns(self, session_id: str) -> list[StoredGameTurn]: ...

    def player_game_memory_context(
        self, player_id: str, case_id: str, *, recent_limit: int = 5
    ) -> Mapping[str, Any]: ...

    def quarantine_failed_turn(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def commit_turn(
        self,
        commit: GameTurnCommit,
        *,
        memory_service: PlayerMemoryService | None,
    ) -> GameTurnCommitResult: ...


@dataclass
class InMemoryCaseSessionStore:
    """Thread-safe store used by unit tests and scripted local smoke only."""

    before_commit: Callable[[GameTurnCommit], None] | None = None
    _sessions: dict[str, StoredGameSession] = field(default_factory=dict, init=False)
    _turns_by_request: dict[str, StoredGameTurn] = field(default_factory=dict, init=False)
    _turns_by_session: dict[str, list[StoredGameTurn]] = field(default_factory=dict, init=False)
    _commit_order: list[StoredGameTurn] = field(default_factory=list, init=False)
    _failed_quarantine: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def runtime_info(self) -> Mapping[str, Any]:
        return {
            "game_session_backend": "in_memory",
            "game_session_persistent": False,
            "game_turn_cas": True,
            "game_memory_atomic_commit": False,
            "failed_trace_quarantine": "in_memory_test_only",
        }

    def create_session(
        self,
        *,
        session_id: str,
        player_id: str | None,
        case_id: str,
        state_json: Mapping[str, Any],
    ) -> StoredGameSession:
        state = _json_mapping(state_json)
        row = StoredGameSession(session_id, player_id, case_id, state, 0)
        with self._lock:
            if session_id in self._sessions:
                raise CaseRequestConflictError(f"duplicate session_id: {session_id}")
            self._sessions[session_id] = row
            self._turns_by_session[session_id] = []
        return row

    def get_session(self, session_id: str) -> StoredGameSession | None:
        with self._lock:
            row = self._sessions.get(session_id)
            return _copy_session(row) if row is not None else None

    def get_turn_by_request_id(self, request_id: str) -> StoredGameTurn | None:
        with self._lock:
            row = self._turns_by_request.get(request_id)
            return _copy_turn(row) if row is not None else None

    def list_turns(self, session_id: str) -> list[StoredGameTurn]:
        with self._lock:
            if session_id not in self._sessions:
                raise CaseSessionNotFoundError(f"unknown session_id: {session_id}")
            return [_copy_turn(row) for row in self._turns_by_session[session_id]]

    def player_game_memory_context(
        self,
        player_id: str,
        case_id: str,
        *,
        recent_limit: int = 5,
    ) -> Mapping[str, Any]:
        if recent_limit < 0:
            raise ValueError("recent_limit must be >= 0")
        with self._lock:
            scoped_sessions = {
                session_id: row
                for session_id, row in self._sessions.items()
                if row.player_id == player_id
            }
            turns = [
                row for row in reversed(self._commit_order) if row.session_id in scoped_sessions
            ]
            recent = []
            for row in turns:
                session = scoped_sessions[row.session_id]
                if session.case_id != case_id:
                    continue
                turn = row.turn_json
                recent.append(
                    {
                        "turn_id": row.turn_id,
                        "session_id": row.session_id,
                        "case_id": session.case_id,
                        "state_version_after": row.state_version_after,
                        "action": str(turn.get("action") or ""),
                        "accepted": bool(turn.get("accepted")),
                        "message_key": str(turn.get("message_key") or ""),
                        "solved": bool(turn.get("solved")),
                    }
                )
                if len(recent) >= recent_limit:
                    break
            completed_cases = {
                row.case_id
                for row in scoped_sessions.values()
                if bool(row.state_json.get("solved"))
            }
            committed_turn_count = len(turns)
            completed_case_count = len(completed_cases)
            return {
                "authority": "personalization_only_no_state_or_tool_effect",
                "case_id": case_id,
                "recent_committed_turns": recent,
                "familiarity": {
                    "committed_turn_count": committed_turn_count,
                    "completed_case_count": completed_case_count,
                    "tier": derive_familiarity_tier(committed_turn_count, completed_case_count),
                },
            }

    def quarantine_failed_turn(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        value = _json_mapping(payload)
        request_id = str(value.get("request_id") or "")
        session_id = str(value.get("session_id") or "")
        if not request_id or session_id not in self._sessions:
            raise CaseSessionStoreError("invalid failed trace quarantine identity")
        quarantine_id = f"in-memory-quarantine:{request_id}"
        with self._lock:
            previous = self._failed_quarantine.get(request_id)
            attempt_count = int(previous.get("attempt_count", 0)) + 1 if previous else 1
            self._failed_quarantine[request_id] = {
                **value,
                "quarantine_id": quarantine_id,
                "review_status": "quarantine",
                "attempt_count": attempt_count,
            }
        return {
            "quarantine_id": quarantine_id,
            "request_id": request_id,
            "review_status": "quarantine",
            "attempt_count": attempt_count,
        }

    def commit_turn(
        self,
        commit: GameTurnCommit,
        *,
        memory_service: PlayerMemoryService | None,
    ) -> GameTurnCommitResult:
        candidate_state = _json_mapping(commit.candidate_state)
        proposal = _json_mapping(commit.proposal_json)
        observation = _json_mapping(commit.observation_json)
        turn_payload = _json_mapping(commit.turn_json)
        with self._lock:
            duplicate = self._turns_by_request.get(commit.request_id)
            if duplicate is not None:
                if duplicate.session_id != commit.session_id:
                    raise CaseRequestConflictError(
                        f"request_id belongs to another session: {commit.request_id}"
                    )
                return GameTurnCommitResult(
                    session=_copy_session(self._sessions[commit.session_id]),
                    turn=_copy_turn(duplicate),
                    idempotent_replay=True,
                )
            current = self._sessions.get(commit.session_id)
            if current is None:
                raise CaseSessionNotFoundError(f"unknown session_id: {commit.session_id}")
            if current.state_version != commit.expected_state_version:
                raise CaseStateVersionConflictError(
                    "state_version conflict: "
                    f"expected {commit.expected_state_version}, actual {current.state_version}"
                )
            if self.before_commit is not None:
                try:
                    self.before_commit(commit)
                except Exception as exc:
                    raise CaseSessionStoreError("in-memory commit hook failed") from exc

            memory_result = None
            if commit.memory_intent is not None:
                if memory_service is None:
                    raise CaseSessionStoreError("memory intent requires a memory service")
                memory_result = memory_service.apply_prepared(commit.memory_intent)
                turn_payload = _attach_memory_result(turn_payload, memory_result)

            next_version = current.state_version + 1
            next_session = StoredGameSession(
                session_id=current.session_id,
                player_id=current.player_id,
                case_id=current.case_id,
                state_json=candidate_state,
                state_version=next_version,
            )
            turn_payload.update(
                {
                    "state_version_before": current.state_version,
                    "state_version_after": next_version,
                    "before_hash": commit.before_hash,
                    "after_hash": commit.after_hash,
                    "status": "committed",
                }
            )
            turn = StoredGameTurn(
                turn_id=commit.turn_id,
                request_id=commit.request_id,
                session_id=commit.session_id,
                state_version_before=current.state_version,
                state_version_after=next_version,
                proposal_json=proposal,
                observation_json=observation,
                turn_json=turn_payload,
                before_hash=commit.before_hash,
                after_hash=commit.after_hash,
                status="committed",
            )
            # Every potentially failing operation above precedes these in-memory assignments.
            self._sessions[commit.session_id] = next_session
            self._turns_by_request[commit.request_id] = turn
            self._turns_by_session[commit.session_id].append(turn)
            self._commit_order.append(turn)
            return GameTurnCommitResult(
                session=_copy_session(next_session),
                turn=_copy_turn(turn),
                idempotent_replay=False,
                memory_result=memory_result,
            )


class PostgresCaseSessionStore:
    """Adapter over PostgresRepository's single-transaction game commit."""

    def __init__(self, repository: Any):
        self.repository = repository

    def close(self) -> None:
        self.repository.close()

    @property
    def runtime_info(self) -> Mapping[str, Any]:
        return {
            "game_session_backend": "postgresql",
            "game_session_persistent": True,
            "game_turn_cas": True,
            "game_memory_atomic_commit": True,
            "failed_trace_quarantine": "postgresql",
        }

    def create_session(
        self,
        *,
        session_id: str,
        player_id: str | None,
        case_id: str,
        state_json: Mapping[str, Any],
    ) -> StoredGameSession:
        try:
            row = self.repository.create_game_session(
                session_id=session_id,
                player_id=player_id,
                case_id=case_id,
                state_json=_json_mapping(state_json),
            )
        except Exception as exc:
            raise _translate_repository_error(exc) from exc
        return _session_from_mapping(row)

    def get_session(self, session_id: str) -> StoredGameSession | None:
        try:
            row = self.repository.get_game_session(session_id)
        except Exception as exc:
            raise _translate_repository_error(exc) from exc
        return _session_from_mapping(row) if row is not None else None

    def get_turn_by_request_id(self, request_id: str) -> StoredGameTurn | None:
        try:
            row = self.repository.get_game_turn_by_request_id(request_id)
        except Exception as exc:
            raise _translate_repository_error(exc) from exc
        return _turn_from_mapping(row) if row is not None else None

    def list_turns(self, session_id: str) -> list[StoredGameTurn]:
        try:
            rows = self.repository.list_game_turns(session_id)
        except Exception as exc:
            raise _translate_repository_error(exc) from exc
        return [_turn_from_mapping(row) for row in rows]

    def player_game_memory_context(
        self,
        player_id: str,
        case_id: str,
        *,
        recent_limit: int = 5,
    ) -> Mapping[str, Any]:
        try:
            return self.repository.player_game_memory_context(
                player_id,
                case_id,
                recent_limit=recent_limit,
            )
        except Exception as exc:
            raise _translate_repository_error(exc) from exc

    def quarantine_failed_turn(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return self.repository.quarantine_failed_game_turn(_json_mapping(payload))
        except Exception as exc:
            raise _translate_repository_error(exc) from exc

    def commit_turn(
        self,
        commit: GameTurnCommit,
        *,
        memory_service: PlayerMemoryService | None,
    ) -> GameTurnCommitResult:
        try:
            session, turn, memory_plan, idempotent = self.repository.commit_game_turn(
                turn_id=commit.turn_id,
                request_id=commit.request_id,
                session_id=commit.session_id,
                expected_state_version=commit.expected_state_version,
                candidate_state=_json_mapping(commit.candidate_state),
                proposal_json=_json_mapping(commit.proposal_json),
                observation_json=_json_mapping(commit.observation_json),
                turn_json=_json_mapping(commit.turn_json),
                before_hash=commit.before_hash,
                after_hash=commit.after_hash,
                memory_intent=commit.memory_intent,
            )
        except Exception as exc:
            raise _translate_repository_error(exc) from exc
        memory_result = (
            commit.memory_intent.result_from_plan(memory_plan)
            if commit.memory_intent is not None and memory_plan is not None
            else None
        )
        return GameTurnCommitResult(
            session=_session_from_mapping(session),
            turn=_turn_from_mapping(turn),
            idempotent_replay=bool(idempotent),
            memory_result=memory_result,
        )


def _attach_memory_result(
    turn_payload: Mapping[str, Any], result: MemoryCommitResult
) -> dict[str, Any]:
    output = _json_mapping(turn_payload)
    status = "committed" if result.committed else ("rejected" if result.rejected else "no_op")
    output.update(
        {
            "memory_status": status,
            "memory_ops_requested": result.requested,
            "memory_ops_committed": result.committed,
            "memory_ops_rejected": len(result.rejected),
        }
    )
    trace = dict(output.get("trace_metadata") or {})
    trace.update(
        {
            "memory_commit_status": status,
            "memory_ops_requested": result.requested,
            "memory_ops_committed": result.committed,
            "memory_ops_rejected": list(result.rejected),
        }
    )
    output["trace_metadata"] = trace
    return output


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - defensive by construction
        raise TypeError("value must encode to a JSON object")
    return decoded


def _copy_session(row: StoredGameSession) -> StoredGameSession:
    return StoredGameSession(
        row.session_id,
        row.player_id,
        row.case_id,
        _json_mapping(row.state_json),
        row.state_version,
    )


def _copy_turn(row: StoredGameTurn) -> StoredGameTurn:
    return StoredGameTurn(
        turn_id=row.turn_id,
        request_id=row.request_id,
        session_id=row.session_id,
        state_version_before=row.state_version_before,
        state_version_after=row.state_version_after,
        proposal_json=_json_mapping(row.proposal_json),
        observation_json=_json_mapping(row.observation_json),
        turn_json=_json_mapping(row.turn_json),
        before_hash=row.before_hash,
        after_hash=row.after_hash,
        status=row.status,
    )


def _session_from_mapping(row: Mapping[str, Any]) -> StoredGameSession:
    return StoredGameSession(
        session_id=str(row["session_id"]),
        player_id=str(row["player_id"]) if row.get("player_id") is not None else None,
        case_id=str(row["case_id"]),
        state_json=_json_mapping(row["state_json"]),
        state_version=int(row["state_version"]),
    )


def _turn_from_mapping(row: Mapping[str, Any]) -> StoredGameTurn:
    return StoredGameTurn(
        turn_id=str(row["turn_id"]),
        request_id=str(row["request_id"]),
        session_id=str(row["session_id"]),
        state_version_before=int(row["state_version_before"]),
        state_version_after=int(row["state_version_after"]),
        proposal_json=_json_mapping(row["proposal_json"]),
        observation_json=_json_mapping(row["observation_json"]),
        turn_json=_json_mapping(row["turn_json"]),
        before_hash=str(row["before_hash"]),
        after_hash=str(row["after_hash"]),
        status=str(row["status"]),
    )


def _translate_repository_error(exc: Exception) -> CaseSessionStoreError:
    name = type(exc).__name__
    message = str(exc)
    if name == "GameSessionNotFoundError":
        return CaseSessionNotFoundError(message)
    if name == "GameStateVersionConflictError":
        return CaseStateVersionConflictError(message)
    if name == "GameRequestConflictError":
        return CaseRequestConflictError(message)
    return CaseSessionStoreError(message or "case session persistence failed")
