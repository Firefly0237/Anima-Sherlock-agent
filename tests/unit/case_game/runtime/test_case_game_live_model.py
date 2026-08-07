from __future__ import annotations

import pytest

from anima.case_game import (
    CaseGameDemo,
    CaseGameModelIdentityError,
    LiveCaseAnswerer,
    LiveDPOAnswerer,
)
from tests.support.doubles.scripted_model import ScriptedModel, render_output
from tests.support.environment.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
LEVELS = ROOT / "assets" / "cases" / "sherlock" / "levels"
PERSONA_PACK = ROOT / "persona_packs" / "public" / "sherlock_holmes"


def _identity() -> dict:
    return {
        "backend": "transformers_peft",
        "base_model": "Qwen/Qwen3.6-27B",
        "base_model_revision": "base-revision",
        "adapter_id": "sherlock-dpo",
        "adapter_sha256": "adapter-sha",
        "served_model": "sherlock-dpo",
        "code_commit": "code-commit",
    }


def _answerer(model: ScriptedModel) -> LiveDPOAnswerer:
    return LiveDPOAnswerer(
        model_client=model,
        persona_pack_dir=PERSONA_PACK,
        expected_identity=_identity(),
    )


@pytest.mark.parametrize(
    ("adapter_id", "adapter_sha256", "served_model", "expected_mode"),
    [
        (None, None, "Qwen/Qwen3.6-27B", "live_base"),
        ("sherlock-sft", "sft-sha", "sherlock-sft", "live_sft"),
        ("sherlock-dpo", "dpo-sha", "sherlock-dpo", "live_dpo"),
    ],
)
def test_live_runtime_mode_is_derived_from_exact_identity(
    adapter_id, adapter_sha256, served_model, expected_mode
) -> None:
    identity = {
        **_identity(),
        "adapter_id": adapter_id,
        "adapter_sha256": adapter_sha256,
        "served_model": served_model,
    }
    model = ScriptedModel(
        lambda _system, _messages: render_output("general", "继续。"),
        model=served_model,
        served_models=[served_model],
        identity=identity,
    )
    answerer = LiveCaseAnswerer(
        model_client=model,
        persona_pack_dir=PERSONA_PACK,
        expected_identity=identity,
    )
    assert answerer.runtime_info["mode"] == expected_mode


def test_identity_rejects_half_bound_adapter_and_base_alias() -> None:
    half_bound = {**_identity(), "adapter_sha256": None}
    with pytest.raises(CaseGameModelIdentityError, match="requires adapter_sha256"):
        LiveCaseAnswerer(
            model_client=ScriptedModel(
                lambda *_args: "",
                model="sherlock-dpo",
                identity=half_bound,
            ),
            persona_pack_dir=PERSONA_PACK,
            expected_identity=half_bound,
        )

    bad_base = {
        **_identity(),
        "adapter_id": None,
        "adapter_sha256": None,
        "served_model": "base-alias",
    }
    with pytest.raises(CaseGameModelIdentityError, match="base served_model"):
        LiveCaseAnswerer(
            model_client=ScriptedModel(
                lambda *_args: "",
                model="base-alias",
                identity=bad_base,
            ),
            persona_pack_dir=PERSONA_PACK,
            expected_identity=bad_base,
        )


def test_live_answerer_binds_identity_and_transcript_metadata() -> None:
    model = ScriptedModel(
        lambda _system, _messages: render_output(
            "general",
            "助手的外表细节值得检查，但现在还不能据此跳到结论。",
        ),
        model="sherlock-dpo",
        served_models=["sherlock-dpo"],
        identity=_identity(),
    )
    answerer = _answerer(model)
    demo = CaseGameDemo(
        LEVELS,
        host_answerer=answerer,
        runtime_info=answerer.runtime_info,
        session_id_factory=lambda: "live-session-1",
    )

    session_id = demo.start_session("red_headed_league")["session_id"]
    result = demo.submit_turn(
        session_id,
        action="ask",
        player_text="助手有什么问题？",
    )
    transcript = demo.export_transcript(session_id)

    assert result["runtime"]["mode"] == "live_dpo"
    assert result["runtime"]["identity_verified"] is True
    assert result["turn"]["runtime_mode"] == "live_dpo"
    assert result["turn"]["degraded"] is False
    assert result["turn"]["host_answer"].startswith("本轮调查记录：半薪助手：")
    assert result["turn"]["host_answer"].endswith(
        "助手的外表细节值得检查，但现在还不能据此跳到结论。"
    )
    assert result["turn"]["trace_metadata"]["host_parse_status"] == "ok"
    assert result["turn"]["trace_metadata"]["host_generation_calls"] == 1
    assert transcript["runtime"]["adapter_sha256"] == "adapter-sha"
    assert transcript["turns"][0]["trace_metadata"]["case_pack_sha256"]
    assert len(model.calls) == 1


def test_live_answerer_repairs_format_once() -> None:
    responses = iter(
        (
            "missing contract blocks",
            render_output("general", "格式已经修复，继续按证据调查。"),
        )
    )
    model = ScriptedModel(
        lambda _system, _messages: next(responses),
        model="sherlock-dpo",
        served_models=["sherlock-dpo"],
        identity=_identity(),
    )
    answerer = _answerer(model)
    demo = CaseGameDemo(
        LEVELS,
        host_answerer=answerer,
        runtime_info=answerer.runtime_info,
    )

    session_id = demo.start_session("speckled_band")["session_id"]
    result = demo.submit_turn(
        session_id,
        action="ask",
        player_text="夜里的低哨声有什么意义",
    )

    trace = result["turn"]["trace_metadata"]
    assert result["turn"]["degraded"] is False
    assert trace["host_parse_status"] == "repaired"
    assert trace["host_retry_count"] == 1
    assert trace["host_generation_calls"] == 2
    assert len(model.calls) == 2
    assert "未通过格式解析" in model.calls[1][1][-1]["content"]


def test_live_answerer_fails_startup_on_identity_mismatch() -> None:
    actual = _identity()
    actual["adapter_sha256"] = "wrong-adapter-sha"
    model = ScriptedModel(
        lambda _system, _messages: render_output("general", "不会执行。"),
        model="sherlock-dpo",
        served_models=["sherlock-dpo"],
        identity=actual,
    )

    with pytest.raises(CaseGameModelIdentityError, match="identity mismatch"):
        _answerer(model)


def test_live_answerer_exposes_model_outage_as_degraded_without_scripted_fallback() -> None:
    model = ScriptedModel(
        lambda _system, _messages: render_output("general", "不会到达。"),
        model="sherlock-dpo",
        served_models=["sherlock-dpo"],
        identity=_identity(),
    )
    answerer = _answerer(model)
    model._fail = True
    demo = CaseGameDemo(
        LEVELS,
        host_answerer=answerer,
        runtime_info=answerer.runtime_info,
    )

    session_id = demo.start_session("red_headed_league")["session_id"]
    result = demo.submit_turn(
        session_id,
        action="hint",
        player_text="给我一个提示",
    )

    assert result["turn"]["degraded"] is True
    assert result["turn"]["error_code"] == "model_unavailable"
    assert "未通过运行时校验" in result["turn"]["host_answer"]
    assert "给我一个提示" not in result["turn"]["host_answer"]
    assert result["turn"]["trace_metadata"]["host_runtime_mode"] == "live_dpo"


def test_spoiler_blocked_raw_output_is_not_reintroduced_through_history() -> None:
    responses = iter(
        (
            render_output(
                "general",
                "Spaulding 就是 John Clay，他正在挖通银行地道。",
            ),
            render_output("general", "只依据当前可见证据继续。"),
        )
    )
    model = ScriptedModel(
        lambda _system, _messages: next(responses),
        model="sherlock-dpo",
        served_models=["sherlock-dpo"],
        identity=_identity(),
    )
    answerer = _answerer(model)
    demo = CaseGameDemo(
        LEVELS,
        host_answerer=answerer,
        runtime_info=answerer.runtime_info,
    )
    session_id = demo.start_session("red_headed_league")["session_id"]
    before = demo.get_session(session_id)

    blocked = demo.submit_turn(
        session_id,
        action="ask",
        player_text="助手有什么问题？",
    )
    after_block = demo.get_session(session_id)
    demo.submit_turn(
        session_id,
        action="ask",
        player_text="红发会为什么付高薪？",
    )

    second_call_messages = model.calls[1][1]
    prior_assistant = [
        row["content"] for row in second_call_messages[:-1] if row["role"] == "assistant"
    ]
    assert blocked["turn"]["guard_blocked"] is True
    assert blocked["turn"]["committed"] is False
    assert blocked["turn"]["before_hash"] == blocked["turn"]["after_hash"]
    assert after_block["state"] == before["state"]
    assert after_block["state_version"] == before["state_version"]
    assert prior_assistant == []
