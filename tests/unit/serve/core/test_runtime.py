"""Tests for the request path (handle_message)."""

from __future__ import annotations

import itertools

import pytest

from anima.persona.contracts.pack import load_pack
from anima.persona.contracts.schemas import MemoryRecord
from anima.serve.core.metrics import MetricsRegistry
from anima.serve.core.repositories import ConversationRow
from anima.serve.core.runtime import (
    DEGRADED_FALLBACK,
    AppServices,
    ModelInfo,
    RuntimeConfig,
    build_lore_indexes,
    handle_message,
)
from anima.serve.inference.embedding import HashEmbedder
from tests.support.doubles.fake_repository import FakeRepository
from tests.support.doubles.scripted_model import ScriptedModel, render_output


def _services(pack_dir, responder, *, fail=False, config=None):
    pack = load_pack(pack_dir)
    packs = {pack.manifest.persona_id: pack}
    embedder = HashEmbedder(dim=128)
    repo = FakeRepository()
    model = ScriptedModel(responder, fail=fail)
    services = AppServices(
        packs=packs,
        lore_indexes=build_lore_indexes(packs, embedder),
        repo=repo,
        model=model,
        embedder=embedder,
        metrics=MetricsRegistry(),
        model_info=ModelInfo(base_model="qwen", adapter_id="sft-x", adapter_sha256="abc"),
        config=config or RuntimeConfig(),
        test_mode=True,
    )
    return services, pack, repo, model


def _conversation(pack, user_id="user-a", conversation_id="conv-1"):
    return ConversationRow(
        conversation_id=conversation_id,
        user_id=user_id,
        persona_id=pack.manifest.persona_id,
        persona_version=pack.manifest.version,
        deleted=False,
    )


def _ids():
    counter = itertools.count(1)
    return lambda prefix: f"{prefix}-{next(counter)}"


def _memory(memory_id: str, predicate: str = "favorite_tea", obj: str = "青芜红茶") -> MemoryRecord:
    return MemoryRecord.from_mapping(
        {
            "memory_id": memory_id,
            "user_id": "user-a",
            "persona_id": "yunxiu_stationmaster",
            "subject": "authenticated_user",
            "predicate": predicate,
            "object": obj,
            "valid_from": None,
            "valid_to": None,
            "source_message_id": "msg-seed",
            "confidence": 1.0,
            "status": "active",
            "created_at": "t0",
            "updated_at": "t0",
        }
    )


def _run(services, conversation, message, *, user_id="user-a"):
    return handle_message(
        services,
        user_id=user_id,
        conversation=conversation,
        user_message=message,
        request_id="req-1",
        now="2026-07-12T10:00:00Z",
        id_factory=_ids(),
    )


def test_grounded_answer_and_history_persisted(pack_dir):
    lore_id = "lore_000001"
    services, pack, repo, model = _services(
        pack_dir, lambda s, m: render_output("lore", "站在云岫山北麓。", lore_ids=[lore_id])
    )
    conversation = _conversation(pack)
    result = _run(services, conversation, "站在哪里？")
    assert not result.degraded
    assert result.answer_mode == "lore"
    assert result.served_answer == "站在云岫山北麓。"
    # user + assistant message both persisted, in order
    msgs = repo.list_messages("conv-1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert result.trace["persona_version"] == pack.manifest.version
    assert result.trace["code_commit"] == "unknown"
    assert result.trace["runtime_config_hash"] == "unknown"
    assert result.trace["retrieval_ms"] >= 0.0
    assert result.trace["model_total_ms"] >= result.trace["ttft_ms"]


def test_invalid_lore_citation_fails_closed_without_memory_commit(pack_dir):
    op = {
        "op": "add",
        "subject": "u",
        "predicate": "home_station",
        "object": "青芜镇",
        "source_message_id": "x",
    }
    services, pack, repo, model = _services(
        pack_dir,
        lambda s, m: render_output("lore", "乱说。", lore_ids=["lore_999999"], ops=[op]),
    )
    result = _run(services, _conversation(pack), "问题")
    assert result.invalid_lore_citations == ("lore_999999",)
    assert result.degraded is True
    assert result.error_code == "invalid_citation"
    assert result.served_answer == DEGRADED_FALLBACK
    assert result.memory_ops_committed == 0
    assert repo.active_memories("user-a", pack.manifest.persona_id) == []
    assert [message["role"] for message in repo.list_messages("conv-1")] == ["user"]


def test_legacy_prompt_only_equivalent_disables_read_and_write(pack_dir):
    """Counterexample: the v3 combined-off semantics make write gold impossible."""

    op = {
        "op": "add",
        "subject": "u",
        "predicate": "home_station",
        "object": "青芜镇",
        "source_message_id": "x",
    }
    services, pack, repo, _model = _services(
        pack_dir,
        lambda s, m: render_output("general", "只按角色档案回答。", ops=[op]),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=False,
        ),
    )

    class ExplodingEmbedder:
        def embed_query(self, _text):
            raise AssertionError("prompt-only must not embed or retrieve")

    services.embedder = ExplodingEmbedder()
    result = _run(services, _conversation(pack), "你好")
    assert result.retrieved_lore_ids == ()
    assert result.retrieved_memory_ids == ()
    assert result.memory_ops_committed == 0
    assert result.memory_ops_rejected[0]["reason"] == "memory_write_disabled"
    assert repo.active_memories("user-a", pack.manifest.persona_id) == []
    assert result.trace["lore_retrieval_enabled"] is False
    assert result.trace["memory_retrieval_enabled"] is False
    assert result.trace["memory_write_enabled"] is False


def test_prompt_only_split_control_disables_reads_but_commits_grounded_write(pack_dir):
    """The repaired control keeps prompt-only retrieval-free while allowing a valid write."""

    op = {
        "op": "add",
        "subject": "spoofed",
        "predicate": "home_station",
        "object": "青芜镇",
        "source_message_id": "MODEL_CLAIMED",
    }
    services, pack, repo, model = _services(
        pack_dir,
        lambda s, m: render_output("general", "记住了。", ops=[op]),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=True,
        ),
    )
    repo.seed_memory(_memory("mem-seed", obj="SEED_MEMORY_MUST_NOT_APPEAR"), [1.0] * 128)

    class WriteOnlyEmbedder:
        model_id = "write-only-test"

        def embed_query(self, _text):
            raise AssertionError("prompt-only must not embed a retrieval query")

        def embed(self, texts):
            return [[1.0] * 128 for _ in texts]

    services.embedder = WriteOnlyEmbedder()
    result = _run(services, _conversation(pack), "我明确告诉你，我的家乡站是青芜镇")

    assert result.degraded is False
    assert result.retrieved_lore_ids == ()
    assert result.retrieved_memory_ids == ()
    assert result.trace["retrieved_lore_ids"] == []
    assert result.trace["retrieved_memory_ids"] == []
    assert result.memory_ops_committed == 1
    stored = repo.active_memories("user-a", pack.manifest.persona_id)
    assert any(
        record.predicate == "home_station" and record.object == "青芜镇" for record in stored
    )
    system_prompt = model.calls[0][0]
    assert "SEED_MEMORY_MUST_NOT_APPEAR" not in system_prompt
    assert result.trace["lore_retrieval_enabled"] is False
    assert result.trace["memory_retrieval_enabled"] is False
    assert result.trace["memory_write_enabled"] is True


def test_prompt_only_write_enabled_does_not_synthesize_a_missing_model_op(pack_dir):
    services, pack, repo, _model = _services(
        pack_dir,
        lambda s, m: render_output("general", "我只回应本轮内容。", ops=[]),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=True,
        ),
    )

    result = _run(services, _conversation(pack), "这是一项长期安排。")

    assert result.degraded is False
    assert result.memory_ops_requested == 0
    assert result.memory_ops_committed == 0
    assert result.memory_ops_rejected == ()
    assert repo.active_memories("user-a", pack.manifest.persona_id) == []
    assert result.retrieved_lore_ids == ()
    assert result.retrieved_memory_ids == ()


def test_prompt_only_recall_with_empty_model_delta_stays_empty(pack_dir):
    services, pack, repo, _model = _services(
        pack_dir,
        lambda s, m: render_output("general", "我只根据对话中已经出现的内容回答。", ops=[]),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=True,
        ),
    )
    repo.seed_memory(_memory("mem-seed"), [1.0] * 128)

    result = _run(services, _conversation(pack), "你还记得我先前说过的饮品偏好吗？")

    assert result.retrieved_lore_ids == ()
    assert result.retrieved_memory_ids == ()
    assert result.memory_ops_requested == 0
    assert result.memory_ops_committed == 0
    assert result.memory_ops_rejected == ()
    active = repo.active_memories("user-a", pack.manifest.persona_id)
    assert [(item.memory_id, item.object) for item in active] == [("mem-seed", "青芜红茶")]


def test_prompt_only_duplicate_recall_write_is_rejected_without_host_rewrite(pack_dir):
    repeated_op = {
        "op": "add",
        "subject": "authenticated_user",
        "predicate": "home_station",
        "object": "青芜镇",
        "source_message_id": "model-claimed",
    }
    services, pack, repo, _model = _services(
        pack_dir,
        lambda s, m: render_output("general", "我记得。", ops=[repeated_op]),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=True,
        ),
    )
    repo.seed_memory(_memory("mem-seed", predicate="home_station", obj="青芜镇"), [1.0] * 128)

    result = _run(services, _conversation(pack), "你还记得我先前说过的饮品偏好吗？")

    assert result.retrieved_lore_ids == ()
    assert result.retrieved_memory_ids == ()
    assert result.memory_ops_requested == 1
    assert result.memory_ops_committed == 0
    assert result.memory_ops_rejected[0]["reason"] == "duplicate_active_value"
    assert '"memory_ops"' in result.served_raw
    assert '"home_station"' in result.served_raw
    active = repo.active_memories("user-a", pack.manifest.persona_id)
    assert [(item.memory_id, item.object) for item in active] == [("mem-seed", "青芜镇")]


def test_memory_read_and_write_controls_are_orthogonal(pack_dir):
    op = {
        "op": "add",
        "subject": "u",
        "predicate": "home_station",
        "object": "青芜镇",
        "source_message_id": "x",
    }
    services, pack, repo, _model = _services(
        pack_dir,
        lambda s, m: render_output("memory", "你喜欢青芜红茶。", memory_ids=["mem-seed"], ops=[op]),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=True,
            memory_write_enabled=False,
        ),
    )
    repo.seed_memory(_memory("mem-seed"), [1.0] * 128)

    result = _run(services, _conversation(pack), "我喜欢什么茶？")

    assert result.degraded is False
    assert result.retrieved_lore_ids == ()
    assert result.retrieved_memory_ids == ("mem-seed",)
    assert result.used_memory_ids == ("mem-seed",)
    assert result.memory_ops_committed == 0
    assert result.memory_ops_rejected[0]["reason"] == "memory_write_disabled"
    assert not any(
        memory.predicate == "home_station"
        for memory in repo.active_memories("user-a", pack.manifest.persona_id)
    )


@pytest.mark.parametrize(
    "mapping",
    [
        {
            "retrieval": {"lore_enabled": False, "memory_enabled": False},
            "runtime": {},
        },
        {
            "retrieval": {
                "lore_enabled": False,
                "memory_enabled": False,
                "enabled": False,
            },
            "runtime": {"memory_write_enabled": True},
        },
        {
            "retrieval": {"lore_enabled": False, "memory_enabled": "false"},
            "runtime": {"memory_write_enabled": True},
        },
    ],
)
def test_split_memory_control_parser_rejects_partial_mixed_or_non_boolean(mapping):
    with pytest.raises(ValueError):
        RuntimeConfig.from_mapping(mapping, formal=False)


def test_legacy_combined_control_keeps_old_semantics_only_outside_formal():
    legacy = {
        "retrieval": {"enabled": False},
        "runtime": {"memory_enabled": False},
    }
    pilot = RuntimeConfig.from_mapping(legacy, formal=False)
    assert pilot.lore_retrieval_enabled is False
    assert pilot.memory_retrieval_enabled is False
    assert pilot.memory_write_enabled is False
    with pytest.raises(ValueError, match="explicit split Memory controls"):
        RuntimeConfig.from_mapping(legacy, formal=True)


@pytest.mark.parametrize(
    ("response", "expected_reason"),
    [
        (
            render_output(
                "memory",
                "伪造引用。",
                memory_ids=["not-retrieved"],
                ops=[
                    {
                        "op": "add",
                        "subject": "u",
                        "predicate": "home_station",
                        "object": "青芜镇",
                        "source_message_id": "x",
                    }
                ],
            ),
            "invalid_citation",
        ),
        ("没有结构块的退化输出", "parse_failure"),
        (
            render_output(
                "general",
                "不会写入越界谓词。",
                ops=[
                    {
                        "op": "add",
                        "subject": "u",
                        "predicate": "unlisted_key",
                        "object": "v",
                        "source_message_id": "x",
                    }
                ],
            ),
            "predicate_not_allowlisted",
        ),
    ],
)
def test_prompt_only_split_write_stays_fail_closed(pack_dir, response, expected_reason):
    services, pack, repo, _model = _services(
        pack_dir,
        lambda s, m: response,
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=True,
        ),
    )

    result = _run(services, _conversation(pack), "记住这项资料")

    assert result.memory_ops_committed == 0
    assert repo.active_memories("user-a", pack.manifest.persona_id) == []
    if expected_reason == "invalid_citation":
        assert result.degraded is True
        assert result.error_code == "invalid_citation"
    elif expected_reason == "parse_failure":
        assert result.degraded is True
        assert result.parse_status == "failed"
    else:
        assert result.degraded is False
        assert result.memory_ops_rejected[0]["reason"] == expected_reason


def test_memory_write_commits_and_rebinds_source(pack_dir):
    predicate = "home_station"
    op = {
        "op": "add",
        "subject": "spoofed",
        "predicate": predicate,
        "object": "青芜镇",
        "source_message_id": "MODEL_CLAIMED",
    }
    services, pack, repo, model = _services(
        pack_dir, lambda s, m: render_output("general", "记住了。", ops=[op])
    )
    result = _run(services, _conversation(pack), "以后叫我阿远，我爱喝青芜红茶")
    assert result.memory_ops_committed == 1
    stored = repo.active_memories("user-a", pack.manifest.persona_id)
    assert len(stored) == 1
    # subject + source_message_id rebound to server-authenticated values
    assert stored[0].subject == "authenticated_user"
    assert stored[0].source_message_id == "msg-1"  # the user message id minted this turn
    assert stored[0].object == "青芜镇"


def test_memory_op_outside_allowlist_rejected_and_audited(pack_dir):
    op = {
        "op": "add",
        "subject": "u",
        "predicate": "system_prompt",
        "object": "leak",
        "source_message_id": "x",
    }
    # predicate 'system_prompt' is banned at the schema level, so the model
    # output won't even parse -> exercise an allowlisted-but-not-in-pack predicate
    op = {
        "op": "add",
        "subject": "u",
        "predicate": "unlisted_key",
        "object": "v",
        "source_message_id": "x",
    }
    services, pack, repo, model = _services(
        pack_dir, lambda s, m: render_output("general", "嗯。", ops=[op])
    )
    result = _run(services, _conversation(pack), "记住这个")
    assert result.memory_ops_committed == 0
    assert repo.audits and repo.audits[0]["reason"] == "predicate_not_allowlisted"


def test_format_repair_retry(pack_dir):
    responses = iter(["这是一句不合契约的裸文本", render_output("general", "好的。")])
    services, pack, repo, model = _services(pack_dir, lambda s, m: next(responses))
    result = _run(services, _conversation(pack), "你好")
    assert result.retry_count == 1
    assert result.parse_status == "repaired"
    assert not result.degraded
    assert result.served_answer == "好的。"
    repair_instruction = model.calls[-1][1][-1]["content"]
    assert "</anima_state>" in repair_instruction
    assert repair_instruction.count("</answer>") >= 2
    assert "不得照抄上一条" in repair_instruction
    assert "不超过 220 个汉字" in repair_instruction


@pytest.mark.parametrize("partial_close", ["", "</", "</ans"])
def test_terminal_answer_close_is_bounded_deterministic_repair(pack_dir, partial_close):
    valid = render_output("general", "好的。")
    raw = valid.removesuffix("</answer>") + partial_close
    services, pack, repo, model = _services(pack_dir, lambda s, m: raw)

    result = _run(services, _conversation(pack), "你好")

    assert result.retry_count == 0
    assert result.parse_status == "terminal_closed"
    assert result.degraded is False
    assert result.first_pass_raw == raw
    assert result.served_raw == valid
    assert result.served_answer == "好的。"
    assert len(model.calls) == 1


def test_terminal_answer_close_rejects_nested_markup_and_uses_model_retry(pack_dir):
    responses = iter(
        [
            '<anima_state>{"answer_mode":"general","used_lore_ids":[],"used_memory_ids":[],"memory_ops":[]}</anima_state><answer>好<tool>',
            render_output("general", "好的。"),
        ]
    )
    services, pack, repo, model = _services(pack_dir, lambda s, m: next(responses))

    result = _run(services, _conversation(pack), "你好")

    assert result.retry_count == 1
    assert result.parse_status == "repaired"
    assert result.degraded is False
    assert len(model.calls) == 2


def test_invalid_memory_citation_gets_one_validator_repair_retry(pack_dir):
    full_id = "msg_seed:mem:1"
    truncated_id = "msg_seed"
    responses = iter(
        [
            render_output("memory", "你先前明确说过：青芜红茶。", memory_ids=[truncated_id]),
            render_output("memory", "你先前明确说过：青芜红茶。", memory_ids=[full_id]),
        ]
    )
    services, pack, repo, model = _services(pack_dir, lambda s, m: next(responses))
    repo.seed_memory(_memory(full_id), [1.0] * 128)

    result = _run(services, _conversation(pack), "你还记得我爱喝什么茶吗？")

    assert result.retry_count == 1
    assert result.parse_status == "citation_repaired"
    assert result.degraded is False
    assert result.invalid_memory_citations == ()
    assert result.used_memory_ids == (full_id,)
    assert result.served_answer == "你先前明确说过：青芜红茶。"
    assert len(model.calls) == 2
    repair_instruction = model.calls[-1][1][-1]["content"]
    assert full_id in repair_instruction
    assert truncated_id in repair_instruction
    assert "截断 id" in repair_instruction
    assert "本修复指令之前的最后一条真实用户消息" in repair_instruction
    assert "不得把上一条输出本身当作 memory_ops 的写入证据" in repair_instruction
    assert "纯提问、回忆、确认或要求复述" in repair_instruction


def test_prompt_only_citation_repair_preserves_history_answer_without_ids_or_write(pack_dir):
    invented_id = "invented-memory"
    repeated_op = {
        "op": "add",
        "subject": "authenticated_user",
        "predicate": "preferred_name",
        "object": "岚舟",
        "source_message_id": "model-claimed",
    }
    history_answer = "你刚才说过，希望我称你为岚舟。"
    responses = iter(
        [
            render_output("memory", history_answer, memory_ids=[invented_id], ops=[repeated_op]),
            render_output("memory", history_answer),
        ]
    )
    services, pack, repo, model = _services(
        pack_dir,
        lambda s, m: next(responses),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=True,
        ),
    )
    repo.append_message("conv-1", "history-user", "user", "以后请称我为岚舟。")
    repo.append_message("conv-1", "history-assistant", "assistant", "好，我记住了。")

    result = _run(services, _conversation(pack), "你还记得该怎么称呼我吗？")

    assert result.parse_status == "citation_repaired"
    assert result.degraded is False
    assert result.served_answer == history_answer
    assert result.answer_mode == "memory"
    assert result.retrieved_memory_ids == ()
    assert result.used_memory_ids == ()
    assert result.memory_ops_requested == 0
    assert result.memory_ops_committed == 0
    assert len(model.calls) == 2
    repair_instruction = model.calls[-1][1][-1]["content"]
    assert "必须保留由对话历史支持的核心答案" in repair_instruction
    assert '允许使用 answer_mode="memory"' in repair_instruction
    assert "不得因为合法 memory id 列表为空就改答不知道或转移问题" in repair_instruction
    assert "memory_ops 仍只依据最后一条真实用户消息独立判定" in repair_instruction


def test_citation_repair_rebuilds_a_positive_delta_from_the_real_user_message(pack_dir):
    op = {
        "op": "add",
        "subject": "authenticated_user",
        "predicate": "home_station",
        "object": "通常住在青芜镇",
        "source_message_id": "model-claimed",
    }
    responses = iter(
        [
            render_output("memory", "记住了。", memory_ids=["invented-memory"], ops=[op]),
            render_output("general", "记住了。", ops=[op]),
        ]
    )
    services, pack, repo, model = _services(
        pack_dir,
        lambda s, m: next(responses),
        config=RuntimeConfig(
            lore_retrieval_enabled=False,
            memory_retrieval_enabled=False,
            memory_write_enabled=True,
        ),
    )

    result = _run(services, _conversation(pack), "我通常住在青芜镇。")

    assert result.retry_count == 1
    assert result.parse_status == "citation_repaired"
    assert result.degraded is False
    assert result.retrieved_lore_ids == ()
    assert result.retrieved_memory_ids == ()
    assert result.memory_ops_requested == 1
    assert result.memory_ops_committed == 1
    stored = repo.active_memories("user-a", pack.manifest.persona_id)
    assert [(item.predicate, item.object) for item in stored] == [
        ("home_station", "通常住在青芜镇")
    ]
    repair_instruction = model.calls[-1][1][-1]["content"]
    assert "仅依据本修复指令之前的最后一条真实用户消息独立重判" in repair_instruction
    assert "须重新输出对应事实增量，内容可以与上一条相同" in repair_instruction


def test_invalid_memory_citation_still_fails_closed_after_bad_repair(pack_dir):
    full_id = "msg_seed:mem:1"
    truncated_id = "msg_seed"
    responses = iter(
        [
            render_output("memory", "你先前明确说过：青芜红茶。", memory_ids=[truncated_id]),
            render_output("memory", "你先前明确说过：青芜红茶。", memory_ids=[truncated_id]),
        ]
    )
    services, pack, repo, model = _services(pack_dir, lambda s, m: next(responses))
    repo.seed_memory(_memory(full_id), [1.0] * 128)

    result = _run(services, _conversation(pack), "你还记得我爱喝什么茶吗？")

    assert result.retry_count == 1
    assert result.parse_status == "citation_repaired"
    assert result.degraded is True
    assert result.error_code == "invalid_citation"
    assert result.invalid_memory_citations == (truncated_id,)
    assert result.served_answer == DEGRADED_FALLBACK
    assert [message["role"] for message in repo.list_messages("conv-1")] == ["user"]


def test_first_pass_and_served_are_distinct(pack_dir):
    responses = iter(["坏输出", render_output("general", "修好了。")])
    services, pack, repo, model = _services(pack_dir, lambda s, m: next(responses))
    result = _run(services, _conversation(pack), "你好")
    assert result.first_pass_raw == "坏输出"
    assert "修好了" in result.served_raw
    assert result.trace["raw_first_pass"] == "坏输出"


def test_unrepairable_output_degrades_without_fabrication(pack_dir):
    services, pack, repo, model = _services(pack_dir, lambda s, m: "永远坏的输出")
    result = _run(services, _conversation(pack), "你好")
    assert result.degraded
    assert result.parse_status == "failed"
    # no assistant message is persisted on a degraded turn
    assert [m["role"] for m in repo.list_messages("conv-1")] == ["user"]


def test_model_unavailable_is_degraded_5xx_not_success(pack_dir):
    services, pack, repo, model = _services(pack_dir, lambda s, m: "", fail=True)
    result = _run(services, _conversation(pack), "你好")
    assert result.degraded
    assert result.error_code == "model_unavailable"
    assert result.trace["error_code"] == "model_unavailable"


def test_memory_not_committed_when_output_malformed(pack_dir):
    # a valid-looking op but wrapped in broken output -> must not commit
    services, pack, repo, model = _services(pack_dir, lambda s, m: "没有状态块的裸答案")
    result = _run(services, _conversation(pack), "以后叫我阿远")
    assert result.memory_ops_committed == 0
    assert repo.active_memories("user-a", pack.manifest.persona_id) == []


def test_cross_user_memory_isolation(pack_dir):
    predicate = "home_station"
    op_a = {
        "op": "add",
        "subject": "u",
        "predicate": predicate,
        "object": "青芜镇",
        "source_message_id": "x",
    }
    responses = {
        "user-a": render_output("general", "记住了。", ops=[op_a]),
        "user-b": render_output("general", "好。"),
    }

    def responder(system, messages):
        # user-a writes a memory; user-b writes nothing
        return responses["user-a"] if "青芜" in messages[-1]["content"] else responses["user-b"]

    services, pack, repo, model = _services(pack_dir, responder)
    _run(
        services,
        _conversation(pack, user_id="user-a", conversation_id="conv-a"),
        "我爱喝青芜红茶",
        user_id="user-a",
    )
    # user-b must never see user-a's memory
    assert repo.active_memories("user-b", pack.manifest.persona_id) == []
    assert len(repo.active_memories("user-a", pack.manifest.persona_id)) == 1


def test_internal_error_is_degraded_and_traced_not_500(pack_dir):
    """A retrieval-layer exception must become a traced degraded result, not an
    untraced 500."""

    services, pack, repo, model = _services(pack_dir, lambda s, m: render_output("general", "x"))

    class BoomEmbedder:
        def embed(self, texts):
            return [[1.0] * 128 for _ in texts]

        def embed_query(self, text):
            raise RuntimeError("embedding backend blew up")

    services.embedder = BoomEmbedder()
    result = _run(services, _conversation(pack), "你好")
    assert result.degraded
    assert result.error_code == "internal_error"
    # the failure row is on disk with its request id, not silently dropped
    assert repo.get_trace("req-1") is not None
    assert repo.get_trace("req-1")["degraded"] is True
    assert "conversation_id_hash" in repo.get_trace("req-1")
    assert "conversation_id" not in repo.get_trace("req-1")
    assert "anima_internal_errors_total" in services.metrics.render()


def test_input_too_long_rejected(pack_dir):
    from anima.serve.core.runtime import InputTooLongError

    services, pack, repo, model = _services(
        pack_dir,
        lambda s, m: render_output("general", "x"),
        config=RuntimeConfig(max_input_chars=5),
    )
    with pytest.raises(InputTooLongError):
        _run(services, _conversation(pack), "这是一句超过五个字符的话")
