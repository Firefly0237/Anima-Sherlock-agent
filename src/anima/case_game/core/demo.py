"""Sherlock case-game facade with candidate-state and atomic commit semantics."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from anima.case_game.core.engine import CaseGameEngine, GameState, GameTurnResult
from anima.case_game.core.loader import load_case_pack
from anima.case_game.core.persona_adapter import (
    guard_case_answer,
    prepare_case_persona_result,
    prepare_case_persona_turn,
)
from anima.case_game.runtime.player_memory import (
    MemoryCommitResult,
    PlayerMemoryError,
    PlayerMemoryService,
    validate_player_id,
)
from anima.case_game.runtime.session_store import (
    CaseRequestConflictError,
    CaseSessionNotFoundError,
    CaseSessionStore,
    CaseSessionStoreError,
    CaseStateVersionConflictError,
    GameTurnCommit,
    InMemoryCaseSessionStore,
    StoredGameSession,
)
from anima.case_game.runtime.tools import (
    CaseToolError,
    CaseToolProposal,
    CaseToolProposalError,
    LiveCaseToolProposer,
    button_tool_proposal,
    execute_case_tool,
)
from anima.persona.contracts.schemas import MemoryOp

_EMPTY_MEMORY_COMMIT = MemoryCommitResult(requested=0, committed=0, rejected=())
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@dataclass(frozen=True)
class HostAnswer:
    """One host/model result before the case-level served-answer guard."""

    text: str
    history_content: str | None = None
    degraded: bool = False
    error_code: str | None = None
    trace_metadata: Mapping[str, Any] = field(default_factory=dict)
    memory_ops: tuple[MemoryOp, ...] = ()


HostAnswerer = Callable[
    [str, GameTurnResult, Sequence[Mapping[str, str]]],
    str | HostAnswer,
]


SCRIPTED_RUNTIME_INFO = {
    "mode": "scripted_smoke",
    "label": "Scripted engine smoke; not live model inference",
    "identity_verified": False,
}


class CaseGameDemoError(ValueError):
    pass


class CaseGamePersistenceError(RuntimeError):
    pass


class CaseGameVersionConflictError(CaseGamePersistenceError):
    pass


@dataclass
class DemoSession:
    session_id: str
    case_id: str
    state: GameState
    state_version: int
    player_id: str | None = field(default=None, repr=False)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    model_history: list[dict[str, str]] = field(default_factory=list, repr=False)


class CaseGameDemo:
    """Small stateful facade for local demo and transcript export."""

    def __init__(
        self,
        levels_root: Path,
        *,
        host_answerer: HostAnswerer | None = None,
        memory_service: PlayerMemoryService | None = None,
        runtime_info: Mapping[str, Any] | None = None,
        tool_proposer: LiveCaseToolProposer | None = None,
        session_store: CaseSessionStore | None = None,
        session_id_factory: Callable[[], str] | None = None,
        request_id_factory: Callable[[], str] | None = None,
        turn_id_factory: Callable[[], str] | None = None,
        memory_source_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.levels_root = Path(levels_root)
        self.engines = {
            path.name: CaseGameEngine(load_case_pack(path))
            for path in sorted(self.levels_root.iterdir())
            if path.is_dir() and (path / "CASE_MANIFEST.json").is_file()
        }
        if not self.engines:
            raise CaseGameDemoError(f"no case packs found under {self.levels_root}")
        self.host_answerer = host_answerer or default_host_answer
        self.memory_service = memory_service
        if self.memory_service is not None and not callable(
            getattr(self.host_answerer, "answer_for_player", None)
        ):
            raise CaseGameDemoError("persistent memory requires a player-aware live host answerer")
        self.runtime_info = dict(runtime_info or SCRIPTED_RUNTIME_INFO)
        self.tool_proposer = tool_proposer
        self.session_store = session_store or InMemoryCaseSessionStore()
        self.runtime_info.update(
            self.memory_service.runtime_info
            if self.memory_service is not None
            else {"memory_enabled": False}
        )
        self.runtime_info.update(self.session_store.runtime_info)
        self.runtime_info.update(
            self.tool_proposer.runtime_info
            if self.tool_proposer is not None
            else {"tool_agent_enabled": False}
        )
        self.session_id_factory = session_id_factory or (lambda: f"game-{uuid.uuid4().hex}")
        self.request_id_factory = request_id_factory or (lambda: f"request-{uuid.uuid4().hex}")
        self.turn_id_factory = turn_id_factory or (lambda: f"turn-{uuid.uuid4().hex}")
        self.memory_source_id_factory = memory_source_id_factory or (
            lambda: f"case-msg-{uuid.uuid4().hex}"
        )

    def list_cases(self) -> dict[str, Any]:
        return {
            "runtime": dict(self.runtime_info),
            "cases": [
                {
                    "case_id": engine.pack.case_id,
                    "title_zh": engine.pack.meta.title_zh,
                    "title_en": engine.pack.meta.title_en,
                    "difficulty": engine.pack.meta.difficulty,
                    "allowed_actions": list(engine.pack.meta.allowed_actions),
                    "objective": _objective(engine),
                }
                for engine in self.engines.values()
            ],
        }

    def start_session(self, case_id: str, *, player_id: str | None = None) -> dict[str, Any]:
        engine = self._engine(case_id)
        normalized_player_id = validate_player_id(player_id) if player_id is not None else None
        if self.memory_service is not None:
            if normalized_player_id is None:
                raise CaseGameDemoError("player_id is required when persistent memory is enabled")
            self.memory_service.ensure_player(normalized_player_id)
        session_id = self.session_id_factory()
        stored = self.session_store.create_session(
            session_id=session_id,
            player_id=normalized_player_id,
            case_id=case_id,
            state_json=_stored_state_payload(engine.new_state()),
        )
        session = DemoSession(
            session_id=stored.session_id,
            case_id=case_id,
            state=engine.state_from_mapping(stored.state_json),
            state_version=stored.state_version,
            player_id=normalized_player_id,
        )
        return self._session_payload(session)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._session_payload(self._session(session_id))

    def submit_turn(
        self,
        session_id: str,
        *,
        action: str,
        player_text: str,
        target_id: str | None = None,
        request_id: str | None = None,
        expected_state_version: int | None = None,
        input_mode: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        normalized_request_id = _validate_request_id(request_id or self.request_id_factory())
        try:
            duplicate = self.session_store.get_turn_by_request_id(normalized_request_id)
        except CaseSessionStoreError as exc:
            raise CaseGamePersistenceError("case turn lookup failed") from exc
        if duplicate is not None:
            if duplicate.session_id != session_id:
                raise CaseGameDemoError(
                    f"request_id belongs to another session: {normalized_request_id}"
                )
            replay_session = self._session(session_id)
            payload = self._session_payload(replay_session)
            replay_turn = dict(duplicate.turn_json)
            replay_turn["idempotent_replay"] = True
            payload["turn"] = replay_turn
            return payload
        if expected_state_version is not None and expected_state_version != session.state_version:
            raise CaseGameVersionConflictError(
                "state_version conflict: "
                f"expected {expected_state_version}, actual {session.state_version}"
            )

        engine = self._engine(session.case_id)
        tool_proposal: CaseToolProposal | None = None
        game_memory_context: Mapping[str, Any] | None = None
        if self.tool_proposer is not None:
            mode = input_mode or ("model" if action in {"", "auto"} else "button")
            try:
                if mode == "model":
                    tool_proposal = self.tool_proposer.propose(
                        engine=engine,
                        state=session.state,
                        player_text=player_text,
                    )
                elif mode == "button":
                    tool_proposal = button_tool_proposal(
                        action=action,
                        player_text=player_text,
                        target_id=target_id,
                        engine=engine,
                        state=session.state,
                        call_id=_tool_call_id(normalized_request_id),
                    )
                else:
                    raise CaseToolError("input_mode must be 'model' or 'button'")
            except (CaseToolError, CaseToolProposalError) as exc:
                failed_result = engine.apply(
                    session.state,
                    "tool_proposal_failed",
                    player_text,
                )
                prepared = prepare_case_persona_result(
                    engine,
                    failed_result,
                    game_session_id=session.session_id,
                    player_text=player_text,
                )
                before_hash = _state_hash(_stored_state_payload(session.state))
                error_code = (
                    exc.code if isinstance(exc, CaseToolProposalError) else "tool_policy_failed"
                )
                trace = (
                    dict(exc.trace_metadata)
                    if isinstance(exc, CaseToolProposalError)
                    else {"tool_proposal_error": str(exc)}
                )
                host_answer = HostAnswer(
                    text="本轮工具提议未通过校验，案件状态未改变。",
                    degraded=True,
                    error_code=error_code,
                    trace_metadata={
                        "agent_input_mode": mode,
                        "agent_tool_execution_status": "not_executed",
                        **trace,
                    },
                )
                memory_commit, memory_status, memory_error = self._reject_memory_ops(
                    session,
                    host_answer,
                    reason=error_code,
                )
                return self._failed_turn_payload(
                    session,
                    action=action or "auto",
                    player_text=player_text,
                    target_id=target_id,
                    request_id=normalized_request_id,
                    prepared=prepared,
                    host_answer=host_answer,
                    served_answer=host_answer.text,
                    guard_blocked=False,
                    before_hash=before_hash,
                    memory_commit=memory_commit,
                    memory_status=memory_status,
                    memory_error=memory_error,
                )
            game_result = execute_case_tool(engine, session.state, tool_proposal)
            action = tool_proposal.action
            target_id = tool_proposal.target_id
            tool_observation = _game_tool_observation(game_result)
            game_memory_context = self._player_game_memory_context(session)
            prepared = prepare_case_persona_result(
                engine,
                game_result,
                game_session_id=session.session_id,
                player_text=player_text,
                agent_tool_call=_tool_call_payload(tool_proposal),
                agent_observation=tool_observation,
                game_memory_context=game_memory_context,
            )
        else:
            game_memory_context = self._player_game_memory_context(session)
            prepared = prepare_case_persona_turn(
                engine,
                session.state,
                game_session_id=session.session_id,
                action=action,
                player_text=player_text,
                target_id=target_id,
                game_memory_context=game_memory_context,
            )
        try:
            player_aware_answerer = getattr(self.host_answerer, "answer_for_player", None)
            if callable(player_aware_answerer) and session.player_id is not None:
                host_value = player_aware_answerer(
                    prepared.persona_user_message,
                    prepared.game_result,
                    tuple(session.model_history),
                    player_id=session.player_id,
                    player_text=player_text,
                )
            else:
                host_value = self.host_answerer(
                    prepared.persona_user_message,
                    prepared.game_result,
                    tuple(session.model_history),
                )
        except Exception as exc:  # noqa: BLE001 - model failures are a product boundary
            host_value = HostAnswer(
                text="本轮模型调用失败，案件状态未改变。",
                degraded=True,
                error_code="model_call_failed",
                trace_metadata={
                    "host_runtime_mode": self.runtime_info.get("mode"),
                    "host_failure_type": type(exc).__name__,
                },
            )
        host_answer = _normalize_host_answer(host_value)
        guarded = guard_case_answer(
            engine,
            prepared.game_result,
            host_answer.text,
            degraded=host_answer.degraded,
        )
        before_state = _stored_state_payload(session.state)
        before_hash = _state_hash(before_state)

        if host_answer.degraded or guarded.blocked:
            memory_commit, memory_status, memory_error = self._reject_memory_ops(
                session,
                host_answer,
                reason=("host_degraded" if host_answer.degraded else "case_guard_blocked"),
            )
            return self._failed_turn_payload(
                session,
                action=action,
                player_text=player_text,
                target_id=target_id,
                request_id=normalized_request_id,
                prepared=prepared,
                host_answer=host_answer,
                served_answer=guarded.served_answer,
                guard_blocked=guarded.blocked,
                before_hash=before_hash,
                memory_commit=memory_commit,
                memory_status=memory_status,
                memory_error=memory_error,
            )

        turn_id = self.turn_id_factory()
        memory_intent = None
        memory_status = "disabled" if self.memory_service is None else "no_op"
        if host_answer.memory_ops:
            if self.memory_service is None or session.player_id is None:
                memory_status = "disabled"
            else:
                try:
                    memory_intent = self.memory_service.prepare_commit(
                        session.player_id,
                        host_answer.memory_ops,
                        source_message_id=self.memory_source_id_factory(),
                        source_session_id=session.session_id,
                        source_turn_id=turn_id,
                    )
                    memory_status = "pending"
                except PlayerMemoryError:
                    return self._failed_turn_payload(
                        session,
                        action=action,
                        player_text=player_text,
                        target_id=target_id,
                        request_id=normalized_request_id,
                        prepared=prepared,
                        host_answer=host_answer,
                        served_answer=guarded.served_answer,
                        guard_blocked=False,
                        before_hash=before_hash,
                        memory_commit=MemoryCommitResult(
                            requested=len(host_answer.memory_ops),
                            committed=0,
                            rejected=(),
                        ),
                        memory_status="failed",
                        memory_error="memory_prepare_failed",
                    )

        candidate_state = _stored_state_payload(prepared.game_result.state)
        after_hash = _state_hash(candidate_state)
        turn_row = {
            "turn_index": len(session.transcript) + 1,
            "turn_id": turn_id,
            "request_id": normalized_request_id,
            "action": action,
            "player_text": player_text,
            "accepted": prepared.game_result.accepted,
            "committed": True,
            "message_key": prepared.game_result.message_key,
            "selected_target_id": prepared.game_result.selected_target_id,
            "stage": prepared.game_result.state.stage,
            "solved": prepared.game_result.state.solved,
            "unlocked_evidence_ids": list(prepared.game_result.unlocked_evidence_ids),
            "solve_status": prepared.game_result.solve_status,
            "score": prepared.game_result.score,
            "covered_feedback_labels": list(prepared.game_result.covered_feedback_labels),
            "missing_feedback_labels": list(prepared.game_result.missing_feedback_labels),
            "host_answer": guarded.served_answer,
            "runtime_mode": self.runtime_info.get("mode"),
            "degraded": host_answer.degraded,
            "error_code": host_answer.error_code,
            "guard_blocked": guarded.blocked,
            "memory_status": memory_status,
            "memory_ops_requested": len(host_answer.memory_ops),
            "memory_ops_committed": 0,
            "memory_ops_rejected": 0,
            "idempotent_replay": False,
            "trace_metadata": {
                **prepared.trace_metadata,
                **({"agent_tool": tool_proposal.to_trace()} if tool_proposal else {}),
                **({"agent_tool_execution_status": "committed"} if tool_proposal else {}),
                **host_answer.trace_metadata,
                **guarded.trace_metadata,
                "memory_commit_status": memory_status,
                "memory_ops_requested": len(host_answer.memory_ops),
                "memory_ops_committed": 0,
                "memory_ops_rejected": [],
            },
        }
        proposal_json = (
            {
                "input_mode": input_mode
                or ("model" if tool_proposal.source == "model" else "button"),
                "source": tool_proposal.source,
                "tool_call": _tool_call_payload(tool_proposal),
                "player_text": player_text,
                "persona_user_message": prepared.persona_user_message,
            }
            if tool_proposal is not None
            else {
                "action": action,
                "player_text": player_text,
                "target_id": target_id,
                "persona_user_message": prepared.persona_user_message,
            }
        )
        observation_json = {
            "accepted": prepared.game_result.accepted,
            "message_key": prepared.game_result.message_key,
            "selected_target_id": prepared.game_result.selected_target_id,
            "unlocked_evidence_ids": list(prepared.game_result.unlocked_evidence_ids),
            "solve_status": prepared.game_result.solve_status,
            "score": prepared.game_result.score,
            "served_answer": guarded.served_answer,
            **(
                {
                    "tool_call_id": tool_proposal.call.call_id,
                    "tool_name": tool_proposal.call.name,
                    "typed_observation": _game_tool_observation(prepared.game_result),
                }
                if tool_proposal is not None
                else {}
            ),
        }
        commit = GameTurnCommit(
            turn_id=str(turn_row["turn_id"]),
            request_id=normalized_request_id,
            session_id=session_id,
            expected_state_version=session.state_version,
            candidate_state=candidate_state,
            proposal_json=proposal_json,
            observation_json=observation_json,
            turn_json=turn_row,
            before_hash=before_hash,
            after_hash=after_hash,
            memory_intent=memory_intent,
        )
        try:
            committed = self.session_store.commit_turn(
                commit,
                memory_service=self.memory_service,
            )
        except PlayerMemoryError:
            return self._failed_turn_payload(
                session,
                action=action,
                player_text=player_text,
                target_id=target_id,
                request_id=normalized_request_id,
                prepared=prepared,
                host_answer=host_answer,
                served_answer=guarded.served_answer,
                guard_blocked=False,
                before_hash=before_hash,
                memory_commit=MemoryCommitResult(
                    requested=len(host_answer.memory_ops),
                    committed=0,
                    rejected=(),
                ),
                memory_status="failed",
                memory_error="memory_commit_failed",
            )
        except CaseStateVersionConflictError as exc:
            raise CaseGameVersionConflictError(str(exc)) from exc
        except CaseSessionNotFoundError as exc:
            raise CaseGameDemoError(str(exc)) from exc
        except (CaseRequestConflictError, CaseSessionStoreError) as exc:
            raise CaseGamePersistenceError("case turn commit failed") from exc

        persisted = self._session(session_id)
        payload = self._session_payload(persisted)
        payload["turn"] = dict(committed.turn.turn_json)
        return payload

    def export_transcript(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        engine = self._engine(session.case_id)
        return {
            "session_id": session.session_id,
            "case_id": session.case_id,
            "case_pack_sha256": engine.pack.manifest_sha256,
            "runtime": dict(self.runtime_info),
            "state_version": session.state_version,
            "final_state": _state_payload(session.state),
            "turns": list(session.transcript),
        }

    def _session_payload(self, session: DemoSession) -> dict[str, Any]:
        engine = self._engine(session.case_id)
        context = engine.model_context(session.state)
        return {
            "session_id": session.session_id,
            "case_id": session.case_id,
            "state_version": session.state_version,
            "title_zh": engine.pack.meta.title_zh,
            "title_en": engine.pack.meta.title_en,
            "objective": _objective(engine),
            "state": _state_payload(session.state),
            "progress": _progress_payload(engine, session.state),
            "evidence_board": context["visible_evidence"],
            "timeline": context["visible_timeline"],
            "investigation_leads": context["available_leads"],
            "allowed_actions": list(engine.pack.meta.allowed_actions),
            "runtime": dict(self.runtime_info),
            "memory": {
                "enabled": self.memory_service is not None,
                "scope": "pseudonymous_player_profile",
                "persistence": "cross_session" if self.memory_service is not None else "disabled",
            },
        }

    def _failed_turn_payload(
        self,
        session: DemoSession,
        *,
        action: str,
        player_text: str,
        target_id: str | None,
        request_id: str,
        prepared,
        host_answer: HostAnswer,
        served_answer: str,
        guard_blocked: bool,
        before_hash: str,
        memory_commit: MemoryCommitResult,
        memory_status: str,
        memory_error: str | None,
    ) -> dict[str, Any]:
        turn_row = {
            "turn_index": len(session.transcript) + 1,
            "turn_id": None,
            "request_id": request_id,
            "action": action,
            "player_text": player_text,
            "accepted": False,
            "candidate_accepted": prepared.game_result.accepted,
            "committed": False,
            "message_key": prepared.game_result.message_key,
            "selected_target_id": target_id,
            "stage": session.state.stage,
            "solved": session.state.solved,
            "unlocked_evidence_ids": [],
            "solve_status": None,
            "score": None,
            "covered_feedback_labels": [],
            "missing_feedback_labels": [],
            "host_answer": served_answer,
            "runtime_mode": self.runtime_info.get("mode"),
            "degraded": host_answer.degraded,
            "error_code": host_answer.error_code
            or ("case_guard_blocked" if guard_blocked else memory_error),
            "guard_blocked": guard_blocked,
            "memory_status": memory_status,
            "memory_ops_requested": memory_commit.requested,
            "memory_ops_committed": memory_commit.committed,
            "memory_ops_rejected": len(memory_commit.rejected),
            "state_version_before": session.state_version,
            "state_version_after": session.state_version,
            "before_hash": before_hash,
            "after_hash": before_hash,
            "status": "failed_precommit",
            "idempotent_replay": False,
            "trace_metadata": {
                **prepared.trace_metadata,
                **host_answer.trace_metadata,
                "guard_blocked": guard_blocked,
                "commit_status": "aborted_before_commit",
                "memory_commit_status": memory_status,
                "memory_ops_requested": memory_commit.requested,
                "memory_ops_committed": memory_commit.committed,
                "memory_ops_rejected": list(memory_commit.rejected),
                **({"memory_error_code": memory_error} if memory_error else {}),
            },
        }
        failure_stage = str(turn_row["error_code"] or "failed_precommit")
        player_scope_sha256 = (
            hashlib.sha256(
                f"anima-game-trace-player:{session.player_id}".encode("utf-8")
            ).hexdigest()
            if session.player_id is not None
            else None
        )
        quarantine_payload = {
            "schema_version": "anima.game_trace_quarantine.v1",
            "request_id": request_id,
            "session_id": session.session_id,
            "player_scope_sha256": player_scope_sha256,
            "case_id": session.case_id,
            "failure_stage": failure_stage,
            "action": action,
            "target_id": target_id,
            "player_text_sha256": hashlib.sha256(player_text.encode("utf-8")).hexdigest(),
            "player_text_chars": len(player_text),
            "state_version": session.state_version,
            "before_hash": before_hash,
            "after_hash": before_hash,
            "candidate_accepted": prepared.game_result.accepted,
            "message_key": prepared.game_result.message_key,
            "trace_metadata": dict(turn_row["trace_metadata"]),
            "review_status": "quarantine",
            "training_eligible": False,
        }
        try:
            quarantine_result = dict(self.session_store.quarantine_failed_turn(quarantine_payload))
            quarantine_status = "persisted"
        except Exception as exc:  # audit persistence must never make candidate state visible
            quarantine_result = {"error_type": type(exc).__name__}
            quarantine_status = "write_failed"
        turn_row["trace_metadata"] = {
            **turn_row["trace_metadata"],
            "trace_quarantine_status": quarantine_status,
            "trace_quarantine": quarantine_result,
        }
        payload = self._session_payload(session)
        payload["turn"] = turn_row
        return payload

    def _reject_memory_ops(
        self,
        session: DemoSession,
        host_answer: HostAnswer,
        *,
        reason: str,
    ) -> tuple[MemoryCommitResult, str, str | None]:
        if self.memory_service is None or session.player_id is None:
            return _EMPTY_MEMORY_COMMIT, "disabled", None
        if not host_answer.memory_ops:
            return _EMPTY_MEMORY_COMMIT, "no_op", None
        try:
            result = self.memory_service.reject(
                session.player_id,
                host_answer.memory_ops,
                reason=reason,
            )
            return result, "rejected", None
        except PlayerMemoryError:
            return (
                MemoryCommitResult(
                    requested=len(host_answer.memory_ops),
                    committed=0,
                    rejected=(),
                ),
                "failed",
                "memory_rejection_audit_failed",
            )

    def _player_game_memory_context(
        self,
        session: DemoSession,
    ) -> Mapping[str, Any] | None:
        if session.player_id is None:
            return None
        try:
            value = self.session_store.player_game_memory_context(
                session.player_id,
                session.case_id,
                recent_limit=5,
            )
        except Exception as exc:  # repository failure must precede all game mutation
            raise CaseGamePersistenceError("game memory context read failed") from exc
        return dict(value)

    def _engine(self, case_id: str) -> CaseGameEngine:
        engine = self.engines.get(case_id)
        if engine is None:
            raise CaseGameDemoError(f"unknown case_id: {case_id}")
        return engine

    def _session(self, session_id: str) -> DemoSession:
        try:
            stored = self.session_store.get_session(session_id)
        except CaseSessionStoreError as exc:
            raise CaseGamePersistenceError("case session read failed") from exc
        if stored is None:
            raise CaseGameDemoError(f"unknown session_id: {session_id}")
        return self._restore_session(stored)

    def _restore_session(self, stored: StoredGameSession) -> DemoSession:
        engine = self._engine(stored.case_id)
        try:
            turns = self.session_store.list_turns(stored.session_id)
        except CaseSessionStoreError as exc:
            raise CaseGamePersistenceError("case turn history read failed") from exc
        transcript = [dict(turn.turn_json) for turn in turns]
        model_history: list[dict[str, str]] = []
        for turn in turns:
            user_content = turn.proposal_json.get("persona_user_message")
            assistant_content = turn.observation_json.get("served_answer")
            if isinstance(user_content, str) and isinstance(assistant_content, str):
                model_history.extend(
                    (
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    )
                )
        return DemoSession(
            session_id=stored.session_id,
            case_id=stored.case_id,
            state=engine.state_from_mapping(stored.state_json),
            state_version=stored.state_version,
            player_id=stored.player_id,
            transcript=transcript,
            model_history=model_history,
        )


def default_host_answer(
    _persona_user_message: str,
    result: GameTurnResult,
    _history: Sequence[Mapping[str, str]] = (),
) -> str:
    if not result.accepted:
        return "这个动作不能推进本案。请回到当前案件和可用行动。"
    if result.action == "hint" and result.hint_text:
        return result.hint_text
    if result.action == "ask":
        if result.followup_question:
            return f"这条问题的答案类别是 {result.answer_class}。{result.followup_question}"
        return "这条问题暂时没有命中可控问答意图。"
    if result.action == "inspect":
        if result.unlocked_evidence_ids:
            return "我已把新的可见线索放到证据板上。"
        return "这次检查没有带来新的可见证据。"
    if result.action == "hypothesize":
        if result.contradiction_ids:
            return "这个假设存在可检验的矛盾，请回到证据板修正。"
        if result.matched_rule_ids:
            return "这个方向已经记录；继续把动机、机会和物证链合并。"
        return "这个假设还没有被当前案包规则支持。"
    if result.action == "solve":
        if result.solve_status == "pass":
            return "你的最终推理已经覆盖关键槽位，本案进入结案复盘。"
        if result.solve_status == "partial":
            return "你的答案只覆盖了一部分关键槽位；继续补足目标、方法或动机链条。"
        return "这个结论还没有抓住案件骨架。"
    if result.action == "recap":
        if result.state.solved:
            return "案件已经结案；下面按全部证据复盘人物、动机、手法与误导项。"
        return "请查看当前证据板、时间线和进度，再选择下一步行动。"
    return "继续。"


def _normalize_host_answer(value: str | HostAnswer) -> HostAnswer:
    if isinstance(value, HostAnswer):
        return value
    return HostAnswer(
        text=str(value),
        trace_metadata={
            "host_runtime_mode": "scripted_smoke",
            "host_parse_status": "not_applicable",
            "host_retry_count": 0,
        },
    )


def _objective(engine: CaseGameEngine) -> str:
    return (
        f"以华生/见习侦探身份推进《{engine.pack.meta.title_zh}》：提问、检查、提出假设，"
        "在证据足够后提交最终推理。"
    )


def _state_payload(state: GameState) -> dict[str, Any]:
    return {
        "case_id": state.case_id,
        "stage": state.stage,
        "unlocked_evidence_ids": list(state.unlocked_evidence_ids),
        "matched_slot_ids": list(state.matched_slot_ids) if state.solved else [],
        "solve_slot_ids": list(state.solve_slot_ids) if state.solved else [],
        "contradiction_ids": list(state.contradiction_ids),
        "hint_tier": state.hint_tier,
        "solved": state.solved,
    }


def _stored_state_payload(state: GameState) -> dict[str, Any]:
    return {
        "case_id": state.case_id,
        "stage": state.stage,
        "unlocked_evidence_ids": list(state.unlocked_evidence_ids),
        "matched_slot_ids": list(state.matched_slot_ids),
        "solve_slot_ids": list(state.solve_slot_ids),
        "contradiction_ids": list(state.contradiction_ids),
        "hint_tier": state.hint_tier,
        "solved": state.solved,
    }


def _state_hash(state: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(state), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tool_call_id(request_id: str) -> str:
    return "call-button-" + hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:32]


def _tool_call_payload(proposal: CaseToolProposal) -> dict[str, Any]:
    return {
        "id": proposal.call.call_id,
        "type": "function",
        "function": {
            "name": proposal.call.name,
            "arguments": dict(proposal.arguments),
        },
    }


def _game_tool_observation(result: GameTurnResult) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "action": result.action,
        "message_key": result.message_key,
        "selected_target_id": result.selected_target_id,
        "unlocked_evidence_ids": list(result.unlocked_evidence_ids),
        "stage": result.state.stage,
        "solved": result.state.solved,
        "solve_status": result.solve_status,
        "score": result.score,
        "covered_feedback_labels": list(result.covered_feedback_labels),
        "missing_feedback_labels": list(result.missing_feedback_labels),
    }


def _validate_request_id(value: str) -> str:
    normalized = value.strip()
    if not _REQUEST_ID_RE.fullmatch(normalized):
        raise ValueError(
            "request_id must contain 1-128 ASCII letters, digits, dots, colons, underscores, or hyphens"
        )
    return normalized


def _progress_payload(engine: CaseGameEngine, state: GameState) -> dict[str, Any]:
    return {
        "stage": state.stage,
        "solved": state.solved,
        "unlocked_evidence": len(state.unlocked_evidence_ids),
        "total_evidence": len(engine.pack.evidence),
        "hint_tier": state.hint_tier,
        "transcript_export_available": True,
    }
