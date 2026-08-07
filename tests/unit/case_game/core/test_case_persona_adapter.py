from __future__ import annotations

import itertools

from anima.case_game import (
    CASE_PERSONA_ADAPTER_ID,
    CaseGameEngine,
    guard_case_answer,
    load_case_pack,
    prepare_case_persona_turn,
)
from anima.serve.core.metrics import MetricsRegistry
from anima.serve.core.repositories import ConversationRow
from anima.serve.core.runtime import AppServices, ModelInfo, build_lore_indexes, handle_message
from anima.serve.inference.embedding import HashEmbedder
from tests.support.doubles.fake_repository import FakeRepository
from tests.support.doubles.scripted_model import ScriptedModel, render_output
from tests.support.environment.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
LEVELS = ROOT / "assets" / "cases" / "sherlock" / "levels"


def _engine(case_id: str) -> CaseGameEngine:
    return CaseGameEngine(load_case_pack(LEVELS / case_id))


def _ids():
    counter = itertools.count(1)
    return lambda prefix: f"{prefix}-{next(counter)}"


def _services(pack_dir, responder):
    pack = __import__("anima.persona.contracts.pack", fromlist=["load_pack"]).load_pack(pack_dir)
    packs = {pack.manifest.persona_id: pack}
    embedder = HashEmbedder(dim=128)
    return AppServices(
        packs=packs,
        lore_indexes=build_lore_indexes(packs, embedder),
        repo=FakeRepository(),
        model=ScriptedModel(responder),
        embedder=embedder,
        metrics=MetricsRegistry(),
        model_info=ModelInfo(base_model="qwen", adapter_id="dpo-sherlock", adapter_sha256="abc"),
        test_mode=True,
    ), pack


def test_adapter_renders_visible_game_context_without_hidden_truth() -> None:
    engine = _engine("speckled_band")
    state = engine.new_state()

    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-1",
        action="ask",
        player_text="Helen 为什么害怕？",
    )

    assert CASE_PERSONA_ADAPTER_ID in turn.persona_user_message
    assert "清晨求助" in turn.persona_user_message
    assert '"solution_slots"' not in turn.persona_user_message
    assert '"spoiler_terms"' not in turn.persona_user_message
    assert '"solution_outline"' not in turn.persona_user_message
    assert "spb.slot.dangerous_animal" not in turn.persona_user_message
    assert "斑点带子是 Roylott 控制的毒蛇" not in turn.persona_user_message
    assert turn.trace_metadata["game_session_id"] == "game-1"
    assert turn.trace_metadata["case_id"] == "speckled_band"


def test_adapter_scopes_unmatched_case_uncertainty_away_from_player_memory() -> None:
    engine = _engine("speckled_band")

    turn = prepare_case_persona_turn(
        engine,
        engine.new_state(),
        game_session_id="game-mixed-memory-case",
        action="ask",
        player_text="你记得我的职业吗？为什么通风口连到隔壁？",
    )

    assert turn.game_result.message_key != "ask_matched"
    assert not turn.game_result.unlocked_evidence_ids
    assert "用户记忆有则据实回答并按契约引用" in turn.persona_user_message
    assert "无则不得称记得" in turn.persona_user_message
    assert "案情未知不得覆盖" in turn.persona_user_message


def test_adapter_blocks_premature_model_spoiler_without_mutating_game_state() -> None:
    engine = _engine("red_headed_league")
    state = engine.new_state()
    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-2",
        action="ask",
        player_text="红发会是真的吗？",
    )

    guarded = guard_case_answer(engine, turn.game_result, "答案是他们要抢银行金库。")

    assert guarded.blocked
    assert guarded.served_answer != "答案是他们要抢银行金库。"
    assert guarded.spoiler_hits[0].spoiler_id == "rhl.sp.bank"
    assert turn.game_result.previous_state == state


def test_blocked_accepted_inspection_preserves_engine_owned_observation() -> None:
    engine = _engine("red_headed_league")
    state = engine.state_from_mapping(
        {
            "stage": "hypothesis",
            "unlocked_evidence_ids": [f"rhl.ev.{index:03d}" for index in range(1, 12)],
        }
    )
    lead = next(row for row in engine.available_leads(state) if row["target_id"] == "rhl.ev.012")
    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-v3-guard-repro",
        action="inspect",
        player_text=lead["player_text"],
        target_id=lead["target_id"],
    )

    guarded = guard_case_answer(
        engine,
        turn.game_result,
        "银行就在当铺背后，因此地下通道必然已经挖通。",
    )

    assert guarded.blocked
    assert guarded.served_answer.startswith("本轮调查记录：邻近银行：")
    assert "City and Suburban Bank" in guarded.served_answer
    assert "地下通道" not in guarded.served_answer
    assert "越过" not in guarded.served_answer
    assert guarded.trace_metadata["case_guard_observation_scaffold_added"] is True
    assert guarded.trace_metadata["case_guard_safe_fallback_applied"] is True
    assert guarded.spoiler_hits[0].spoiler_id == "rhl.sp.tunnel"


def test_adapter_allows_post_case_final_answer() -> None:
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
    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-3",
        action="solve",
        player_text=(
            "红发会是骗局，用高薪把 Wilson 每天固定支开。Spaulding 其实是 John Clay，"
            "他利用地下室挖通附近银行，准备在周六夜里抢金库。"
        ),
    )

    guarded = guard_case_answer(engine, turn.game_result, "银行金库正是目标。")

    assert turn.game_result.state.solved
    assert '"solution_outline"' in turn.persona_user_message
    assert "真正目标是相邻银行及其金库" in turn.persona_user_message
    assert "完整结案复盘" in turn.persona_user_message
    assert guarded.blocked is False
    assert guarded.served_answer.startswith("银行金库正是目标。")
    assert "结案复盘要点" in guarded.served_answer


def test_adapter_scaffolds_incomplete_post_case_final_answer() -> None:
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
    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-final-scaffold",
        action="solve",
        player_text=(
            "红发会是骗局，用高薪把 Wilson 每天固定支开。Spaulding 其实是 John Clay，"
            "他利用地下室挖通附近银行，准备在周六夜里抢金库。"
        ),
    )

    guarded = guard_case_answer(engine, turn.game_result, "通过。红发会是诱饵。")

    assert guarded.blocked is False
    assert guarded.trace_metadata["case_guard_final_scaffold_added"] is True
    assert "结案复盘要点" in guarded.served_answer
    assert "Spaulding/John Clay 是核心罪犯" in guarded.served_answer
    assert "真正目标是相邻银行及其金库" in guarded.served_answer


def test_final_scaffold_uses_the_engine_concept_matcher_without_duplicate_labels() -> None:
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
    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-final-paraphrase",
        action="solve",
        player_text=(
            "红发会是骗局，用高薪把 Wilson 固定支开。Spaulding 就是 John Clay，"
            "他从地下室挖向相邻银行；解散说明准备完成，应立即设伏。"
        ),
    )
    model_answer = (
        "通过。红发会是人为骗局，用来稳定支开 Wilson；Spaulding 即 John Clay，是核心罪犯。"
        "他以地下室摄影为借口掘进，真正目标是相邻银行及其金库。突然解散说明准备完成、行动临近。"
    )

    guarded = guard_case_answer(engine, turn.game_result, model_answer)

    assert guarded.served_answer == model_answer
    assert guarded.trace_metadata["case_guard_final_scaffold_added"] is False


def test_speckled_band_final_contract_requires_direct_canonical_reveal_without_boilerplate() -> (
    None
):
    engine = _engine("speckled_band")
    turn = prepare_case_persona_turn(
        engine,
        engine.new_state(),
        game_session_id="game-speckled-final",
        action="solve",
        player_text=(
            "Roylott 为了钱阻止继女结婚，并借维修把 Helen 搬进 Julia 的房间重演袭击。"
            "门窗无法进入；通风口、假铃绳、固定床、保险柜、牛奶、短鞭、哨声和金属声构成路径。"
            "吉普赛人是误导，斑点带子是他控制的毒蛇。"
        ),
    )

    assert turn.game_result.solve_status == "pass"
    assert "必须明确说出“斑点带子”是 Roylott 控制的毒蛇" in turn.persona_user_message
    assert "不要主动输出安全免责声明" in turn.persona_user_message
    assert "只可做文学层面的高层手法总结" not in turn.persona_user_message


def test_zero_lead_hypothesis_cannot_expose_model_invented_inspection() -> None:
    engine = _engine("red_headed_league")
    state = engine.state_from_mapping(
        {
            "stage": "hypothesis",
            "unlocked_evidence_ids": [f"rhl.ev.{index:03d}" for index in range(1, 13)],
        }
    )
    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-no-leads",
        action="hypothesize",
        player_text="Spaulding 从地下室挖隧道，目标是相邻银行。",
    )
    guarded = guard_case_answer(
        engine,
        turn.game_result,
        "这条证据链方向正确。下一步：检验银行地下金库是否已有异常。",
    )

    assert not engine.available_leads(turn.game_result.state)
    assert "检验银行" not in guarded.served_answer
    assert "提交结案推理" in guarded.served_answer
    assert "请求下一档提示" in guarded.served_answer
    assert guarded.trace_metadata["case_guard_legal_next_step_applied"] is True


def test_adapter_adds_action_specific_npc_contracts() -> None:
    engine = _engine("red_headed_league")
    contradiction_state = engine.state_from_mapping(
        {
            "stage": "investigation",
            "unlocked_evidence_ids": ["rhl.ev.004", "rhl.ev.006", "rhl.ev.010"],
        }
    )

    contradiction_turn = prepare_case_persona_turn(
        engine,
        contradiction_state,
        game_session_id="game-contract-1",
        action="hypothesize",
        player_text="他们只是想偷当铺里的货。",
    )
    recap_turn = prepare_case_persona_turn(
        engine,
        contradiction_state,
        game_session_id="game-contract-2",
        action="recap",
        player_text="把证据链整理一下。",
    )

    assert contradiction_turn.game_result.contradiction_ids
    assert "玩家假设与当前可见证据存在冲突" in contradiction_turn.persona_user_message
    assert "不足以成立" in contradiction_turn.persona_user_message
    assert "只整理当前可见证据" in recap_turn.persona_user_message
    assert "未 solve 通过前" in recap_turn.persona_user_message


def test_structured_inspection_is_bound_to_target_and_serves_observation() -> None:
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
    lead = next(row for row in engine.available_leads(state) if row["target_id"] == "spb.ev.013")

    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-structured-inspect",
        action="inspect",
        player_text=lead["player_text"],
        target_id=lead["target_id"],
    )
    guarded = guard_case_answer(engine, turn.game_result, "请你自行报告铃绳观察。")

    assert turn.trace_metadata["game_selected_target_id"] == "spb.ev.013"
    assert "这次检查已经在游戏内完成" in turn.persona_user_message
    assert "不得要求玩家反过来提供观察结果" in turn.persona_user_message
    assert guarded.served_answer.startswith("本轮调查记录：假铃绳：")
    assert guarded.trace_metadata["case_guard_observation_scaffold_added"] is True


def test_authored_hint_replaces_model_invented_target() -> None:
    engine = _engine("red_headed_league")
    state = engine.state_from_mapping(
        {
            "stage": "investigation",
            "unlocked_evidence_ids": ["rhl.ev.001", "rhl.ev.002", "rhl.ev.003", "rhl.ev.007"],
        }
    )
    turn = prepare_case_persona_turn(
        engine,
        state,
        game_session_id="game-hint-authority",
        action="hint",
        player_text="给提示。",
    )

    guarded = guard_case_answer(engine, turn.game_result, "去查仓库账本和钥匙。")

    assert turn.game_result.hint_id == "rhl.hint.001"
    assert guarded.served_answer == f"提示：{turn.game_result.hint_text}"
    assert "仓库" not in guarded.served_answer
    assert guarded.trace_metadata["case_guard_authoritative_hint"] is True


def test_partial_solve_serves_only_deterministic_feedback_categories() -> None:
    engine = _engine("red_headed_league")
    turn = prepare_case_persona_turn(
        engine,
        engine.new_state(),
        game_session_id="game-solve-feedback",
        action="solve",
        player_text="红发会是骗局，目标是银行。",
    )

    guarded = guard_case_answer(engine, turn.game_result, "你还缺报酬来源、租房合同和八周长度。")

    assert turn.game_result.missing_feedback_labels == (
        "诱离效果",
        "作案者",
        "作案路径",
        "行动时机",
    )
    assert guarded.served_answer == (
        "尚未结案，华生。已覆盖：骗局性质、犯罪目标。"
        "仍需补足：诱离效果、作案者、作案路径、行动时机。"
        "请只围绕这些维度继续补充；此前结案陈述会累计保留。"
    )
    assert "报酬来源" not in guarded.served_answer
    assert guarded.trace_metadata["case_guard_solve_feedback_scaffold_added"] is True


def test_adapter_message_runs_through_existing_persona_runtime(pack_dir) -> None:
    engine = _engine("red_headed_league")
    turn = prepare_case_persona_turn(
        engine,
        engine.new_state(),
        game_session_id="game-4",
        action="ask",
        player_text="助手有什么问题？",
    )

    services, pack = _services(
        pack_dir,
        lambda _system, messages: assert_case_context_and_answer(messages[-1]["content"]),
    )
    conversation = ConversationRow(
        conversation_id="conv-1",
        user_id="user-a",
        persona_id=pack.manifest.persona_id,
        persona_version=pack.manifest.version,
        deleted=False,
    )
    result = handle_message(
        services,
        user_id="user-a",
        conversation=conversation,
        user_message=turn.persona_user_message,
        request_id="req-1",
        now="2026-07-28T10:00:00Z",
        id_factory=_ids(),
    )
    guarded = guard_case_answer(engine, turn.game_result, result.served_answer)

    assert not result.degraded
    assert guarded.blocked is False
    assert guarded.served_answer.startswith("本轮调查记录：半薪助手：")
    assert guarded.served_answer.endswith("我会只依据已解锁证据推进。")
    assert turn.trace_metadata["adapter_id"] == CASE_PERSONA_ADAPTER_ID


def test_full_red_headed_league_script_runs_through_adapter() -> None:
    engine = _engine("red_headed_league")
    steps = (
        ("ask", "助手有什么问题？"),
        ("ask", "他为什么老去地下室？"),
        ("ask", "周围有没有银行？"),
        (
            "solve",
            "红发会是骗局，用高薪把 Wilson 每天固定支开。Spaulding 其实是 John Clay，"
            "他利用地下室挖通附近银行，准备在周六夜里抢金库。",
        ),
    )

    final_turn = _run_adapter_script(engine, steps)

    assert final_turn.game_result.solve_status == "pass"
    assert final_turn.game_result.state.stage == "post_case"
    assert final_turn.game_result.state.solved


def test_full_speckled_band_script_runs_through_adapter() -> None:
    engine = _engine("speckled_band")
    steps = (
        ("ask", "遗产有什么关系？"),
        ("ask", "为什么说是密室？"),
        ("ask", "为什么通风口连到隔壁？"),
        ("ask", "床为什么固定？"),
        ("ask", "保险柜和牛奶说明什么？"),
        ("ask", "短鞭为什么打结？"),
        (
            "solve",
            "Roylott 是凶手，他为了钱阻止继女结婚，让 Helen 面临和 Julia 同样危险。"
            "维修是借口，门窗锁住说明普通入口不成立；通风口、假铃绳、固定床、保险柜、牛奶碟和短鞭是一组物证。"
            "哨声和金属声对应夜间控制，吉普赛人是误导；斑点带子是 Roylott 控制的毒蛇。",
        ),
    )

    final_turn = _run_adapter_script(engine, steps)

    assert final_turn.game_result.solve_status == "pass"
    assert final_turn.game_result.state.stage == "post_case"
    assert final_turn.game_result.state.solved


def _run_adapter_script(engine: CaseGameEngine, steps):
    state = engine.new_state()
    last_turn = None
    for index, (action, player_text) in enumerate(steps, start=1):
        turn = prepare_case_persona_turn(
            engine,
            state,
            game_session_id=f"script-{engine.pack.case_id}",
            action=action,
            player_text=player_text,
        )
        if not turn.game_result.state.solved:
            for slot in engine.pack.solution.required_slots:
                assert slot.slot_id not in turn.persona_user_message, (
                    f"slot leaked at step {index}: {slot.slot_id}"
                )
        guarded = guard_case_answer(engine, turn.game_result, "继续依据已解锁证据推进。")
        assert guarded.blocked is False
        state = turn.game_result.state
        last_turn = turn
    assert last_turn is not None
    return last_turn


def assert_case_context_and_answer(message: str) -> str:
    assert "[案件游戏上下文]" in message
    assert CASE_PERSONA_ADAPTER_ID in message
    assert "rhl.ev.004" in message
    assert "rhl.slot.assistant_is_culprit" not in message
    return render_output("general", "我会只依据已解锁证据推进。")
