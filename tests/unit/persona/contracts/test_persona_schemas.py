"""Contract tests for anima.persona.contracts.schemas."""

from __future__ import annotations

import pytest

from anima.persona.contracts.schemas import (
    ANSWER_MODES,
    REQUIRED_PROFILE_FIELDS,
    AnimaState,
    LoreFact,
    MemoryOp,
    PersonaContractError,
    SafetyPolicy,
    validate_profile,
)


def _lore_mapping(**overrides):
    row = {
        "fact_id": "lore_000042",
        "subject": "yunxiu_station",
        "predicate": "opened_in",
        "object": "启历 402 年",
        "aliases": ["402 年"],
        "valid_from": None,
        "valid_to": None,
        "known_by_persona": True,
        "persona_response": "云岫站在启历 402 年建成，这一年份有站志可查。",
        "answer_slots": [["启历 402 年"], ["站志"]],
        "boundary_forbidden_claims": [],
        "source_ref": "bible_v1",
        "review_status": "draft",
    }
    row.update(overrides)
    return row


class TestLoreFact:
    def test_round_trip(self):
        fact = LoreFact.from_mapping(_lore_mapping())
        assert fact.fact_id == "lore_000042"
        assert fact.aliases == ("402 年",)
        assert fact.answer_slots == (("启历 402 年",), ("站志",))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"fact_id": "fact_1"},
            {"object": ""},
            {"review_status": "approved"},
            {"aliases": ["402 年", "402 年"]},
            {"known_by_persona": "yes"},
            {"answer_slots": [["只够一组"]]},
            {"persona_response": "没有命中任何答案槽。"},
            {"boundary_forbidden_claims": ["不应出现"]},
        ],
    )
    def test_rejects_bad_rows(self, overrides):
        with pytest.raises(PersonaContractError):
            LoreFact.from_mapping(_lore_mapping(**overrides))

    def test_unknown_fact_requires_claim_specific_boundary_markers(self):
        fact = LoreFact.from_mapping(
            _lore_mapping(
                known_by_persona=False,
                persona_response=None,
                answer_slots=[],
                boundary_forbidden_claims=["我在未来亲历了这件事"],
            )
        )
        assert fact.boundary_forbidden_claims == ("我在未来亲历了这件事",)


class TestAnimaState:
    def test_valid_state_parses(self):
        state = AnimaState.from_mapping(
            {
                "answer_mode": "lore",
                "used_lore_ids": ["lore_000001"],
                "used_memory_ids": [],
                "memory_ops": [],
            }
        )
        assert state.answer_mode == "lore"
        assert state.used_lore_ids == ("lore_000001",)

    def test_all_answer_modes_accepted(self):
        for mode in ANSWER_MODES:
            state = AnimaState.from_mapping(
                {"answer_mode": mode, "used_lore_ids": [], "used_memory_ids": [], "memory_ops": []}
            )
            assert state.answer_mode == mode

    @pytest.mark.parametrize(
        "mapping",
        [
            {"answer_mode": "chat", "used_lore_ids": [], "used_memory_ids": [], "memory_ops": []},
            {
                "answer_mode": "lore",
                "used_lore_ids": ["bad_id"],
                "used_memory_ids": [],
                "memory_ops": [],
            },
            {"answer_mode": "lore", "used_lore_ids": [], "used_memory_ids": []},
        ],
    )
    def test_rejects_bad_state(self, mapping):
        with pytest.raises(PersonaContractError):
            AnimaState.from_mapping(mapping)

    def test_memory_ops_nested_validation(self):
        with pytest.raises(PersonaContractError):
            AnimaState.from_mapping(
                {
                    "answer_mode": "memory",
                    "used_lore_ids": [],
                    "used_memory_ids": [],
                    "memory_ops": [
                        {
                            "op": "overwrite",
                            "subject": "u",
                            "predicate": "p",
                            "object": "v",
                            "source_message_id": "m1",
                        }
                    ],
                }
            )


class TestMemoryOp:
    def test_add_requires_object(self):
        with pytest.raises(PersonaContractError):
            MemoryOp.from_mapping(
                {
                    "op": "add",
                    "subject": "authenticated_user",
                    "predicate": "home_station",
                    "object": None,
                    "source_message_id": "m1",
                }
            )

    def test_delete_allows_null_object(self):
        op = MemoryOp.from_mapping(
            {
                "op": "delete",
                "subject": "authenticated_user",
                "predicate": "home_station",
                "object": None,
                "source_message_id": "m1",
            }
        )
        assert op.op == "delete"

    def test_noop_allows_empty_predicate(self):
        op = MemoryOp.from_mapping(
            {
                "op": "noop",
                "subject": "authenticated_user",
                "predicate": None,
                "object": None,
                "source_message_id": "m1",
            }
        )
        assert op.predicate is None


class TestProfileAndSafety:
    def test_profile_requires_all_fields(self):
        profile = {name: "值" for name in REQUIRED_PROFILE_FIELDS}
        assert validate_profile(profile) == []
        broken = dict(profile)
        broken.pop("anti_ooc_rules")
        broken["speech_style"] = " "
        errors = validate_profile(broken)
        assert any("anti_ooc_rules" in e for e in errors)
        assert any("speech_style" in e for e in errors)

    def test_safety_rejects_executable_predicates(self):
        with pytest.raises(PersonaContractError):
            SafetyPolicy.from_mapping(
                {
                    "refusal_style": "站规口吻",
                    "hard_refuse_categories": ["weapons"],
                    "in_character_refusal_examples": ["不行。"],
                    "memory_predicate_allowlist": ["preferred_name", "system_prompt"],
                    "forbidden_assistant_markers": ["作为AI"],
                }
            )

    def test_safety_requires_nonempty_allowlist(self):
        with pytest.raises(PersonaContractError):
            SafetyPolicy.from_mapping(
                {
                    "refusal_style": "站规口吻",
                    "hard_refuse_categories": [],
                    "in_character_refusal_examples": [],
                    "memory_predicate_allowlist": [],
                    "forbidden_assistant_markers": ["作为AI"],
                }
            )
