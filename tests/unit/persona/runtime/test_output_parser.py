"""Tests for anima.persona.runtime.output."""

from __future__ import annotations

import json

import pytest

from anima.persona.runtime.output import parse_output

VALID_STATE = {
    "answer_mode": "lore",
    "used_lore_ids": ["lore_000001"],
    "used_memory_ids": [],
    "memory_ops": [],
}


def _render(state: dict, answer: str = "下一班 09:20 发车。") -> str:
    return f"<anima_state>{json.dumps(state, ensure_ascii=False)}</anima_state>\n<answer>{answer}</answer>"


def test_parses_valid_output():
    parsed = parse_output(_render(VALID_STATE))
    assert parsed.ok
    assert parsed.state is not None
    assert parsed.state.answer_mode == "lore"
    assert parsed.state.memory_ops == ()
    assert parsed.answer == "下一班 09:20 发车。"
    assert parsed.errors == ()


def test_missing_state_block():
    parsed = parse_output("<answer>你好。</answer>")
    assert not parsed.ok
    assert any("missing_state_block" in e for e in parsed.errors)
    assert parsed.answer == "你好。"  # answer still recovered for served path


def test_multiple_state_blocks_rejected():
    text = _render(VALID_STATE) + _render(VALID_STATE)
    parsed = parse_output(text)
    assert not parsed.ok
    assert any("multiple_state_blocks" in e for e in parsed.errors)


def test_invalid_state_json():
    text = "<anima_state>{not json}</anima_state><answer>好。</answer>"
    parsed = parse_output(text)
    assert not parsed.ok
    assert any("state_json_invalid" in e for e in parsed.errors)


def test_state_contract_violation_surfaces_field_errors():
    bad = dict(VALID_STATE, answer_mode="chatty")
    parsed = parse_output(_render(bad))
    assert not parsed.ok
    assert any("answer_mode" in e for e in parsed.errors)


def test_empty_answer_rejected():
    parsed = parse_output(_render(VALID_STATE, answer=" "))
    assert not parsed.ok
    assert any("empty_answer" in e for e in parsed.errors)


def test_extra_text_outside_blocks_flagged():
    text = "让我想想。" + _render(VALID_STATE)
    parsed = parse_output(text)
    assert not parsed.ok
    assert any("extra_text" in e for e in parsed.errors)
    # but content is still recovered
    assert parsed.answer == "下一班 09:20 发车。"


def test_whitespace_between_blocks_is_fine():
    text = f"<anima_state>{json.dumps(VALID_STATE)}</anima_state>\n\n <answer>好。</answer>\n"
    assert parse_output(text).ok


@pytest.mark.parametrize("mode", ["abstain", "refuse", "general", "memory"])
def test_all_modes_round_trip(mode):
    state = dict(VALID_STATE, answer_mode=mode, used_lore_ids=[])
    assert parse_output(_render(state)).ok
