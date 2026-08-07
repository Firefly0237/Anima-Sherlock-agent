from __future__ import annotations

import json

import pytest

from anima.case_game import CaseGameDemo, HostAnswer
from anima.case_game.core.demo import CaseGamePersistenceError
from anima.case_game.runtime.game_memory import (
    derive_familiarity_tier,
    normalize_game_memory_context,
)
from anima.case_game.runtime.player_memory import PlayerMemoryService
from anima.case_game.runtime.session_store import InMemoryCaseSessionStore
from anima.persona.contracts.pack import load_pack
from anima.persona.contracts.schemas import MemoryOp
from anima.serve.inference.embedding import HashEmbedder
from tests.support.doubles.fake_repository import FakeRepository
from tests.support.environment.paths import PROJECT_ROOT

LEVELS = PROJECT_ROOT / "assets" / "cases" / "sherlock" / "levels"
PERSONA_PACK = PROJECT_ROOT / "persona_packs" / "public" / "sherlock_holmes"
PLAYER_A = "game_memory_alpha_0001"
PLAYER_B = "game_memory_bravo_0002"


class _ProfileWritingHost:
    def answer_for_player(self, *_args, **_kwargs) -> HostAnswer:
        return HostAnswer(
            text="记下了，阿远。",
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


class _CaptureHost:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, persona_user_message, *_args) -> HostAnswer:
        self.messages.append(persona_user_message)
        return HostAnswer(text="继续按已见事实调查。")


def _payload(message: str) -> dict:
    return json.loads(message.splitlines()[1])


def test_profile_memory_is_bound_to_committed_session_and_turn() -> None:
    repository = FakeRepository()
    memory = PlayerMemoryService(
        repository=repository,
        embedder=HashEmbedder(),
        pack=load_pack(PERSONA_PACK),
        min_score=None,
        now_factory=lambda: "2026-08-03T00:00:00Z",
    )
    demo = CaseGameDemo(
        LEVELS,
        host_answerer=_ProfileWritingHost(),
        memory_service=memory,
        session_store=InMemoryCaseSessionStore(),
        session_id_factory=lambda: "game-memory-provenance-session",
        turn_id_factory=lambda: "game-memory-provenance-turn",
        memory_source_id_factory=lambda: "game-memory-provenance-message",
    )
    session_id = demo.start_session("red_headed_league", player_id=PLAYER_A)["session_id"]

    result = demo.submit_turn(
        session_id,
        action="hint",
        player_text="我叫阿远，请记住，再给一个提示。",
        request_id="game-memory-provenance-request",
        expected_state_version=0,
    )

    user_id = memory.ensure_player(PLAYER_A)
    records = repository.active_memories(user_id, "sherlock_holmes")
    assert result["turn"]["memory_status"] == "committed"
    assert len(records) == 1
    assert records[0].source_message_id == "game-memory-provenance-message"
    assert records[0].source_session_id == session_id
    assert records[0].source_turn_id == "game-memory-provenance-turn"


def test_recent_turns_are_same_player_same_case_and_familiarity_is_derived() -> None:
    host = _CaptureHost()
    session_ids = iter(("alpha-rhl-one", "bravo-rhl", "alpha-spb", "alpha-rhl-two"))
    turn_ids = iter(f"turn-{index:02d}" for index in range(1, 8))
    demo = CaseGameDemo(
        LEVELS,
        host_answerer=host,
        session_store=InMemoryCaseSessionStore(),
        session_id_factory=lambda: next(session_ids),
        turn_id_factory=lambda: next(turn_ids),
    )
    alpha_rhl = demo.start_session("red_headed_league", player_id=PLAYER_A)["session_id"]
    for index in range(3):
        demo.submit_turn(
            alpha_rhl,
            action="hint",
            player_text="给我一个当前提示。",
            request_id=f"alpha-rhl-{index}",
        )
    bravo_rhl = demo.start_session("red_headed_league", player_id=PLAYER_B)["session_id"]
    demo.submit_turn(
        bravo_rhl,
        action="hint",
        player_text="另一个玩家的提示。",
        request_id="bravo-rhl-0",
    )
    alpha_spb = demo.start_session("speckled_band", player_id=PLAYER_A)["session_id"]
    demo.submit_turn(
        alpha_spb,
        action="hint",
        player_text="另一个案件的提示。",
        request_id="alpha-spb-0",
    )
    alpha_rhl_two = demo.start_session("red_headed_league", player_id=PLAYER_A)["session_id"]
    demo.submit_turn(
        alpha_rhl_two,
        action="recap",
        player_text="复盘当前案件。",
        request_id="alpha-rhl-recap",
    )

    memory = _payload(host.messages[-1])["game_memory"]
    assert memory["authority"] == "personalization_only_no_state_or_tool_effect"
    assert memory["familiarity"] == {
        "committed_turn_count": 4,
        "completed_case_count": 0,
        "tier": "established_acquaintance",
    }
    assert len(memory["recent_committed_turns"]) == 3
    assert {row["case_id"] for row in memory["recent_committed_turns"]} == {"red_headed_league"}
    assert {row["session_id"] for row in memory["recent_committed_turns"]} == {"alpha-rhl-one"}
    assert bravo_rhl not in json.dumps(memory)
    assert alpha_spb not in json.dumps(memory)


def test_game_memory_context_cannot_add_tool_or_cross_case_authority() -> None:
    valid = {
        "authority": "personalization_only_no_state_or_tool_effect",
        "case_id": "red_headed_league",
        "recent_committed_turns": [],
        "familiarity": {
            "committed_turn_count": 0,
            "completed_case_count": 0,
            "tier": "new_contact",
        },
    }
    normalized = normalize_game_memory_context(valid, current_case_id="red_headed_league")
    assert normalized["familiarity"]["tier"] == "new_contact"
    assert derive_familiarity_tier(8, 1) == "seasoned_partner"

    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_game_memory_context(
            {**valid, "allowed_tools": ["submit_solution"]},
            current_case_id="red_headed_league",
        )
    with pytest.raises(ValueError, match="crossed case scope"):
        normalize_game_memory_context(valid, current_case_id="speckled_band")


class _BrokenContextStore(InMemoryCaseSessionStore):
    def player_game_memory_context(self, *_args, **_kwargs):
        raise RuntimeError("forced context failure")


def test_game_memory_read_failure_precedes_model_and_state_commit() -> None:
    host = _CaptureHost()
    store = _BrokenContextStore()
    demo = CaseGameDemo(LEVELS, host_answerer=host, session_store=store)
    session_id = demo.start_session("red_headed_league", player_id=PLAYER_A)["session_id"]

    with pytest.raises(CaseGamePersistenceError, match="game memory context read failed"):
        demo.submit_turn(
            session_id,
            action="hint",
            player_text="给我一个提示。",
            request_id="broken-game-memory-context",
            expected_state_version=0,
        )

    assert demo.get_session(session_id)["state_version"] == 0
    assert demo.export_transcript(session_id)["turns"] == []
    assert host.messages == []


def test_failed_turn_is_quarantined_without_raw_player_text_or_state_change() -> None:
    class _DegradedHost:
        def __call__(self, *_args) -> HostAnswer:
            return HostAnswer(
                text="模型结果不可用。",
                degraded=True,
                error_code="forced_model_failure",
            )

    store = InMemoryCaseSessionStore()
    demo = CaseGameDemo(LEVELS, host_answerer=_DegradedHost(), session_store=store)
    session_id = demo.start_session("red_headed_league", player_id=PLAYER_A)["session_id"]
    raw_player_text = "这是不应写入 quarantine 的玩家原文"

    result = demo.submit_turn(
        session_id,
        action="hint",
        player_text=raw_player_text,
        request_id="failed-trace-quarantine",
        expected_state_version=0,
    )

    stored = store._failed_quarantine["failed-trace-quarantine"]
    assert result["state_version"] == 0
    assert result["turn"]["trace_metadata"]["trace_quarantine_status"] == "persisted"
    assert stored["review_status"] == "quarantine"
    assert stored["training_eligible"] is False
    assert stored["before_hash"] == stored["after_hash"]
    assert raw_player_text not in json.dumps(stored, ensure_ascii=False)


def test_game_memory_personalization_does_not_change_case_state_hash() -> None:
    anonymous = CaseGameDemo(LEVELS, host_answerer=_CaptureHost())
    personalized = CaseGameDemo(LEVELS, host_answerer=_CaptureHost())
    anonymous_id = anonymous.start_session("speckled_band")["session_id"]
    personalized_id = personalized.start_session("speckled_band", player_id=PLAYER_A)["session_id"]

    without_memory = anonymous.submit_turn(
        anonymous_id,
        action="hint",
        player_text="给我一个提示。",
        request_id="state-hash-anonymous",
    )
    with_memory = personalized.submit_turn(
        personalized_id,
        action="hint",
        player_text="给我一个提示。",
        request_id="state-hash-personalized",
    )

    assert without_memory["state"] == with_memory["state"]
    assert without_memory["turn"]["after_hash"] == with_memory["turn"]["after_hash"]
