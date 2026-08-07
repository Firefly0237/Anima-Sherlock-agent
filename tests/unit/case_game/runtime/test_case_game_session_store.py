from __future__ import annotations

from collections.abc import Iterator

import pytest

from anima.case_game import HostAnswer
from anima.case_game.core.demo import (
    CaseGameDemo,
    CaseGamePersistenceError,
    CaseGameVersionConflictError,
)
from anima.case_game.runtime.player_memory import (
    MemoryCommitResult,
    PlayerMemoryError,
)
from anima.case_game.runtime.session_store import InMemoryCaseSessionStore
from anima.persona.contracts.schemas import MemoryOp
from tests.support.environment.paths import PROJECT_ROOT

LEVELS = PROJECT_ROOT / "assets" / "cases" / "sherlock" / "levels"


def _ids(prefix: str) -> Iterator[str]:
    index = 0
    while True:
        index += 1
        yield f"{prefix}-{index}"


def _state_snapshot(demo: CaseGameDemo, session_id: str) -> tuple[dict, int, int]:
    payload = demo.get_session(session_id)
    turns = demo.export_transcript(session_id)["turns"]
    return payload["state"], payload["state_version"], len(turns)


@pytest.mark.parametrize(
    ("host", "error_code", "guard_blocked"),
    (
        (
            lambda *_args: (_ for _ in ()).throw(TimeoutError("model timeout")),
            "model_call_failed",
            False,
        ),
        (
            lambda *_args: HostAnswer(
                text="格式修复仍然失败。",
                degraded=True,
                error_code="parse_failed",
            ),
            "parse_failed",
            False,
        ),
        (
            lambda *_args: "Spaulding 就是 John Clay，他正在挖通银行地道。",
            "case_guard_blocked",
            True,
        ),
    ),
)
def test_model_parse_and_guard_failures_leave_state_and_history_unchanged(
    host, error_code: str, guard_blocked: bool
) -> None:
    demo = CaseGameDemo(LEVELS, host_answerer=host)
    session_id = demo.start_session("red_headed_league")["session_id"]
    before = _state_snapshot(demo, session_id)

    result = demo.submit_turn(
        session_id,
        action="ask",
        player_text="助手有什么问题？",
        request_id=f"failure-{error_code}",
        expected_state_version=0,
    )

    assert result["turn"]["committed"] is False
    assert result["turn"]["error_code"] == error_code
    assert result["turn"]["guard_blocked"] is guard_blocked
    assert result["turn"]["before_hash"] == result["turn"]["after_hash"]
    assert _state_snapshot(demo, session_id) == before


class _MemoryWritingHost:
    def answer_for_player(self, *_args, **_kwargs) -> HostAnswer:
        return HostAnswer(
            text="我会记住你的称呼，并继续依据当前证据。",
            memory_ops=(
                MemoryOp.from_mapping(
                    {
                        "op": "add",
                        "subject": "untrusted",
                        "predicate": "preferred_name",
                        "object": "阿远",
                        "source_message_id": "untrusted",
                    }
                ),
            ),
        )


class _FailingMemoryService:
    runtime_info = {
        "memory_enabled": True,
        "memory_backend": "fault_injection",
        "memory_scope": "pseudonymous_player_profile",
    }

    def ensure_player(self, player_id: str) -> str:
        return f"user:{player_id}"

    def prepare_commit(self, *_args, **_kwargs):
        return object()

    def apply_prepared(self, _prepared):
        raise PlayerMemoryError("injected memory commit failure")

    def reject(self, _player_id, ops, *, reason: str) -> MemoryCommitResult:
        return MemoryCommitResult(
            requested=len(ops),
            committed=0,
            rejected=tuple({"reason": reason} for _ in ops),
        )


def test_memory_commit_failure_leaves_state_and_usable_history_unchanged() -> None:
    demo = CaseGameDemo(
        LEVELS,
        host_answerer=_MemoryWritingHost(),
        memory_service=_FailingMemoryService(),  # type: ignore[arg-type]
    )
    session_id = demo.start_session("red_headed_league", player_id="player_atomic_0001")[
        "session_id"
    ]
    before = _state_snapshot(demo, session_id)

    result = demo.submit_turn(
        session_id,
        action="ask",
        player_text="助手有什么问题？",
        request_id="memory-fault-1",
        expected_state_version=0,
    )

    assert result["turn"]["memory_status"] == "failed"
    assert result["turn"]["memory_ops_committed"] == 0
    assert result["turn"]["before_hash"] == result["turn"]["after_hash"]
    assert _state_snapshot(demo, session_id) == before


def test_store_failure_occurs_before_any_state_assignment() -> None:
    def fail(_commit) -> None:
        raise OSError("injected database fault")

    store = InMemoryCaseSessionStore(before_commit=fail)
    demo = CaseGameDemo(LEVELS, session_store=store)
    session_id = demo.start_session("red_headed_league")["session_id"]
    before = _state_snapshot(demo, session_id)

    with pytest.raises(CaseGamePersistenceError, match="case turn commit failed"):
        demo.submit_turn(
            session_id,
            action="ask",
            player_text="助手有什么问题？",
            request_id="db-fault-1",
            expected_state_version=0,
        )

    assert _state_snapshot(demo, session_id) == before


def test_duplicate_request_is_replayed_without_second_model_call_or_state_change() -> None:
    calls = {"value": 0}

    def host(*_args) -> str:
        calls["value"] += 1
        return "只依据当前可见证据继续。"

    demo = CaseGameDemo(LEVELS, host_answerer=host)
    session_id = demo.start_session("red_headed_league")["session_id"]
    first = demo.submit_turn(
        session_id,
        action="ask",
        player_text="助手有什么问题？",
        request_id="idempotent-request-1",
        expected_state_version=0,
    )
    after_first = _state_snapshot(demo, session_id)
    replay = demo.submit_turn(
        session_id,
        action="ask",
        player_text="这段内容不会再次执行",
        request_id="idempotent-request-1",
        expected_state_version=0,
    )

    assert calls["value"] == 1
    assert replay["turn"]["idempotent_replay"] is True
    assert replay["turn"]["turn_id"] == first["turn"]["turn_id"]
    assert _state_snapshot(demo, session_id) == after_first
    assert after_first[1:] == (1, 1)


def test_stale_state_version_fails_before_model_and_does_not_create_turn() -> None:
    calls = {"value": 0}

    def host(*_args) -> str:
        calls["value"] += 1
        return "只依据当前可见证据继续。"

    demo = CaseGameDemo(LEVELS, host_answerer=host)
    session_id = demo.start_session("red_headed_league")["session_id"]
    demo.submit_turn(
        session_id,
        action="ask",
        player_text="助手有什么问题？",
        request_id="version-good-1",
        expected_state_version=0,
    )
    before_conflict = _state_snapshot(demo, session_id)

    with pytest.raises(CaseGameVersionConflictError, match="expected 0, actual 1"):
        demo.submit_turn(
            session_id,
            action="ask",
            player_text="红发会为什么付高薪？",
            request_id="version-stale-1",
            expected_state_version=0,
        )

    assert calls["value"] == 1
    assert _state_snapshot(demo, session_id) == before_conflict


def test_shared_store_recovers_ten_sessions_and_committed_history_after_restart() -> None:
    store = InMemoryCaseSessionStore()
    session_ids = _ids("restart-session")
    request_ids = _ids("restart-request")
    first = CaseGameDemo(
        LEVELS,
        session_store=store,
        session_id_factory=lambda: next(session_ids),
        request_id_factory=lambda: next(request_ids),
    )
    created: list[str] = []
    for index in range(10):
        case_id = "red_headed_league" if index % 2 == 0 else "speckled_band"
        session_id = first.start_session(case_id)["session_id"]
        first.submit_turn(session_id, action="hint", player_text="给我一个提示")
        created.append(session_id)

    restarted = CaseGameDemo(LEVELS, session_store=store)
    for session_id in created:
        restored = restarted.get_session(session_id)
        transcript = restarted.export_transcript(session_id)
        assert restored["state_version"] == 1
        assert len(transcript["turns"]) == 1
        assert transcript["turns"][0]["status"] == "committed"
