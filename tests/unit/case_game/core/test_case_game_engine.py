from __future__ import annotations

import json
from dataclasses import replace

from anima.case_game import CaseGameEngine, load_case_pack
from tests.support.environment.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
LEVELS = ROOT / "assets" / "cases" / "sherlock" / "levels"


def _engine(case_id: str) -> CaseGameEngine:
    return CaseGameEngine(load_case_pack(LEVELS / case_id))


def test_core_engine_remains_valid_when_case_identity_changes() -> None:
    original = _engine("red_headed_league").pack
    renamed = replace(
        original,
        meta=replace(original.meta, case_id="case_alpha", case_prefix="alpha."),
    )
    engine = CaseGameEngine(renamed)
    state = engine.new_state()
    assert state.case_id == "case_alpha"
    assert engine.apply(state, "recap", "复盘当前证据").accepted is True
    assert engine.apply(state, "ask", "Roylott 与毒蛇").message_key == "cross_case_reference"


def _context_text(result) -> str:
    return json.dumps(result.model_context, ensure_ascii=False)


def test_loads_both_case_packs_and_eval_categories_are_complete() -> None:
    expected_categories = {
        "opening",
        "ask_answer_class",
        "inspect_unlock",
        "hypothesis_partial",
        "hypothesis_contradiction",
        "hint_non_spoiling",
        "solve_pass",
        "solve_partial",
        "recap",
        "premature_spoiler_request",
        "cross_case_attack",
        "ooc_attack",
        "safety_boundary",
        "multi_turn_stability",
    }
    for case_id in ("red_headed_league", "speckled_band"):
        pack = load_case_pack(LEVELS / case_id)
        assert pack.case_id == case_id
        assert len(pack.eval_cases) == 60
        assert {case.category for case in pack.eval_cases} == expected_categories


def test_opening_context_excludes_hidden_truth_and_spoiler_tables() -> None:
    engine = _engine("speckled_band")
    state = engine.new_state()
    context_text = json.dumps(engine.model_context(state), ensure_ascii=False)
    context = engine.model_context(state)

    assert "solution_slots" not in context
    assert "spoiler_terms" not in context
    assert "spb.slot.dangerous_animal" not in context_text
    assert "spb.sp.final_creature" not in context_text
    assert "斑点带子是 Roylott 控制的毒蛇" not in context_text


def test_ask_unlocks_declared_evidence_and_advances_stage() -> None:
    engine = _engine("red_headed_league")
    state = engine.new_state()

    result = engine.apply(state, "ask", "Spaulding 可疑吗？")
    repeated = engine.apply(state, "ask", "Spaulding 可疑吗？")

    assert result.accepted
    assert result.matched_intent_id == "rhl.q.assistant_role"
    assert result.answer_class == "partly"
    assert result.state.stage == "investigation"
    assert result.unlocked_evidence_ids == ("rhl.ev.004", "rhl.ev.005")
    assert repeated == result


def test_unknown_action_and_cross_case_reference_do_not_mutate_state() -> None:
    engine = _engine("red_headed_league")
    state = engine.new_state()

    unknown = engine.apply(state, "dance", "看线索")
    cross_case = engine.apply(state, "ask", "我想直接查看 spb.ev.001")

    assert not unknown.accepted
    assert unknown.state == state
    assert unknown.message_key == "unknown_action"
    assert not cross_case.accepted
    assert cross_case.state == state
    assert cross_case.message_key == "cross_case_reference"


def test_inspect_unlocks_only_declared_inspection_targets() -> None:
    engine = _engine("speckled_band")
    state = engine.state_from_mapping(
        {
            "stage": "investigation",
            "unlocked_evidence_ids": [
                "spb.ev.001",
                "spb.ev.002",
                "spb.ev.003",
                "spb.ev.004",
                "spb.ev.005",
            ],
        }
    )

    result = engine.apply(state, "inspect", "inspect:spb.place.exterior")

    assert result.accepted
    assert result.unlocked_evidence_ids == ("spb.ev.012",)
    assert "spb.ev.012" in result.state.unlocked_evidence_ids
    assert "spb.ev.013" not in result.state.unlocked_evidence_ids


def test_hypothesis_records_slots_and_contradictions() -> None:
    engine = _engine("speckled_band")
    state = engine.new_state()

    result = engine.apply(state, "hypothesize", "我怀疑是吉普赛人干的。")

    assert result.accepted
    assert result.matched_rule_ids == ("spb.hyp.gypsy_false_lead",)
    assert "spb.slot.false_leads" in result.state.matched_slot_ids
    assert "spb.contra.gypsy_cannot_explain_room" in result.state.contradiction_ids
    assert result.state.stage == "investigation"
    assert "spb.slot.false_leads" not in _context_text(result)


def test_blind_v3_full_hypothesis_uses_canonical_solution_concepts() -> None:
    engine = _engine("red_headed_league")
    state = engine.state_from_mapping(
        {
            "stage": "hypothesis",
            "unlocked_evidence_ids": [f"rhl.ev.{index:03d}" for index in range(1, 13)],
        }
    )

    result = engine.apply(
        state,
        "hypothesize",
        "Spaulding 与假扮 Duncan Ross 的同伙设立红发会，用固定时段把 Wilson 骗离当铺；"
        "Spaulding 借地下室和长时间跪地活动挖掘通往背靠背银行的地道，"
        "解散通知意味着工程已完成、抢劫将近。",
    )

    assert result.message_key == "hypothesis_matched"
    assert set(result.matched_slot_ids) == {
        "rhl.slot.displacement",
        "rhl.slot.assistant_is_culprit",
        "rhl.slot.tunnel_method",
        "rhl.slot.bank_target",
        "rhl.slot.timing",
    }
    assert "rhl.slot.decoy" not in result.matched_slot_ids
    assert not result.contradiction_ids


def test_unrelated_hypothesis_remains_a_miss() -> None:
    engine = _engine("red_headed_league")

    result = engine.apply(engine.new_state(), "hypothesize", "也许只是天气变化造成的误会。")

    assert result.message_key == "hypothesis_miss"
    assert not result.matched_slot_ids


def test_partial_solve_keeps_hidden_solution_evidence_out_of_model_context() -> None:
    engine = _engine("red_headed_league")
    state = engine.state_from_mapping(
        {
            "stage": "solution",
            "unlocked_evidence_ids": ["rhl.ev.002", "rhl.ev.006", "rhl.ev.008"],
        }
    )

    result = engine.apply(state, "solve", "红发会是骗局，目的是骗 Wilson 离开。")

    assert result.solve_status == "partial"
    assert result.state.stage == "solution"
    assert not result.state.solved
    assert "rhl.ev.013" not in result.state.unlocked_evidence_ids
    assert "法国金币" not in _context_text(result)


def test_passing_solve_moves_to_post_case_and_reveals_all_evidence() -> None:
    engine = _engine("red_headed_league")
    state = engine.state_from_mapping(
        {
            "stage": "solution",
            "unlocked_evidence_ids": [
                "rhl.ev.004",
                "rhl.ev.006",
                "rhl.ev.009",
                "rhl.ev.011",
                "rhl.ev.012",
            ],
        }
    )

    result = engine.apply(
        state,
        "solve",
        "红发会是骗局，用高薪把 Wilson 每天固定支开。Spaulding 其实是 John Clay，"
        "他利用地下室挖通附近银行，准备在周六夜里抢金库。",
    )

    assert result.solve_status == "pass"
    assert result.state.stage == "post_case"
    assert result.state.solved
    assert set(result.state.unlocked_evidence_ids) == set(engine.pack.evidence)
    assert "法国金币" in _context_text(result)


def test_spoiler_scanner_flags_terms_before_allowed_stage() -> None:
    engine = _engine("red_headed_league")

    hits = engine.scan_spoilers("答案是不是银行金库？", "premise")

    assert [(hit.spoiler_id, hit.severity) for hit in hits] == [("rhl.sp.bank", "block")]


def test_structured_leads_are_safe_and_resolve_without_hidden_literal_matching() -> None:
    engine = _engine("speckled_band")
    state = engine.state_from_mapping(
        {
            "stage": "investigation",
            "unlocked_evidence_ids": [
                "spb.ev.001",
                "spb.ev.002",
                "spb.ev.003",
                "spb.ev.004",
                "spb.ev.005",
            ],
        }
    )
    leads = engine.available_leads(state)
    bell_rope = next(lead for lead in leads if lead["target_id"] == "spb.ev.013")

    result = engine.apply(
        state,
        bell_rope["action"],
        bell_rope["player_text"],
        target_id=bell_rope["target_id"],
    )

    assert bell_rope["label"] == "检查：床边铃绳"
    assert "毒蛇" not in json.dumps(leads, ensure_ascii=False)
    assert result.selected_target_id == "spb.ev.013"
    assert result.unlocked_evidence_ids == ("spb.ev.013",)
    assert result.state.stage == "hypothesis"


def test_unavailable_or_wrong_action_target_does_not_mutate_state() -> None:
    engine = _engine("red_headed_league")
    state = engine.new_state()

    result = engine.apply(
        state,
        "inspect",
        "尝试错误目标。",
        target_id="rhl.q.assistant_role",
    )

    assert result.accepted is False
    assert result.message_key == "target_unavailable"
    assert result.state == state


def test_hints_advance_one_global_tier_at_a_time_even_after_stage_jump() -> None:
    engine = _engine("speckled_band")
    state = engine.state_from_mapping(
        {
            "stage": "hypothesis",
            "unlocked_evidence_ids": [
                "spb.ev.001",
                "spb.ev.002",
                "spb.ev.003",
                "spb.ev.004",
                "spb.ev.005",
            ],
        }
    )

    observed = []
    for _ in range(4):
        result = engine.apply(state, "hint", "请给下一条提示。")
        observed.append((result.hint_id, result.state.hint_tier))
        state = result.state

    assert observed == [
        ("spb.hint.001", 1),
        ("spb.hint.002", 2),
        ("spb.hint.003", 3),
        ("spb.hint.004", 4),
    ]


def test_explicit_hint_focus_routes_to_authored_tier_and_unlocks_cumulatively() -> None:
    engine = _engine("speckled_band")
    initial_ids = ["spb.ev.001", "spb.ev.002", "spb.ev.003", "spb.ev.004"]
    state = engine.state_from_mapping(
        {
            "stage": "hypothesis",
            "unlocked_evidence_ids": initial_ids,
        }
    )

    result = engine.apply(
        state,
        "hint",
        "再给一档提示，聚焦作案工具与控制方式，尤其是保险柜、牛奶碟、低哨和环状短鞭。",
    )

    assert result.hint_id == "spb.hint.004"
    assert result.state.hint_tier == 4
    assert set(result.unlocked_evidence_ids) == {
        "spb.ev.005",
        "spb.ev.008",
        "spb.ev.011",
        "spb.ev.012",
        "spb.ev.013",
        "spb.ev.014",
        "spb.ev.015",
        "spb.ev.016",
        "spb.ev.017",
    }
    assert "保险柜" in result.hint_text
    assert "短鞭" in result.hint_text


def test_incremental_solve_uses_cumulative_submissions_and_stable_feedback() -> None:
    engine = _engine("red_headed_league")
    state = engine.new_state()

    first = engine.apply(state, "solve", "红发会是骗局，真正目标是银行。")
    second = engine.apply(first.state, "solve", "Spaulding 是 John Clay，并从地下室挖掘地道。")
    final = engine.apply(second.state, "solve", "Wilson 被固定诱离；解散后应立即行动。")

    assert first.solve_status == "partial"
    assert first.covered_feedback_labels == ("骗局性质", "犯罪目标")
    assert first.missing_feedback_labels == ("诱离效果", "作案者", "作案路径", "行动时机")
    assert "红发会是人为骗局/诱饵" not in _context_text(first)
    assert second.solve_status == "partial"
    assert final.solve_status == "pass"
    assert final.state.solved is True


def test_blind_playtest_full_red_headed_solution_scores_as_complete_paraphrase() -> None:
    engine = _engine("red_headed_league")
    result = engine.apply(
        engine.new_state(),
        "solve",
        "所谓红发会是 Spaulding 与冒名 Duncan Ross 制造的诱离骗局。Spaulding 以半薪进入当铺，"
        "使店主八周内固定离店；摄影掩护他在地下室长期挖掘，隧道目标是银行金库。"
        "解散通知意味着通道完成、抢劫将发生，应立即在金库伏击。",
    )

    assert result.solve_status == "pass"
    assert result.score == 1.0


def test_speckled_band_solution_truth_does_not_score_safety_policy_as_case_fact() -> None:
    engine = _engine("speckled_band")

    assert "spb.slot.safety_boundary" not in {
        slot.slot_id for slot in engine.pack.solution.required_slots
    }
    dangerous_slot = next(
        slot
        for slot in engine.pack.solution.required_slots
        if slot.slot_id == "spb.slot.dangerous_animal"
    )
    assert dangerous_slot.label == "斑点带子是 Roylott 控制的毒蛇"
    assert engine.solution_slot_matches("斑点带子其实是继父控制的毒蛇。", dangerous_slot)
