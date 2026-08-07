"""Deterministic state engine for Sherlock case packs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from anima.case_game.core.models import (
    CasePack,
    EvidenceCard,
    Hint,
    QuestionIntent,
    SolutionSlot,
    SpoilerHit,
)

_TERM_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_INTERNAL_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*\.[a-z0-9_.-]+$", re.IGNORECASE)
_TERM_SEPARATORS = (
    "是不是",
    "有没有",
    "为什么",
    "怎么",
    "什么",
    "多少",
    "这个",
    "那个",
    "一下",
    "可以",
    "能不能",
    "能否",
    "是否",
    "真的",
    "真名",
    "/",
    "、",
    "，",
    ",",
    "。",
    "；",
    ";",
    "：",
    ":",
    "是",
    "的",
    "与",
    "和",
    "及",
    "或",
    "有",
    "给",
    "拿",
    "说",
    "看",
    "查",
    "问",
    "把",
    "被",
    "从",
    "向",
    "在",
    "为",
    "了",
    "吗",
    "呢",
)
_STOP_TERMS = frozenset(
    {"什么", "怎么", "为什么", "说明", "关键", "真正", "最终", "核心", "案件", "方式", "证据"}
)


@dataclass(frozen=True)
class GameState:
    case_id: str
    stage: str
    unlocked_evidence_ids: tuple[str, ...]
    matched_slot_ids: tuple[str, ...] = ()
    solve_slot_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    hint_tier: int = 0
    solved: bool = False


@dataclass(frozen=True)
class GameTurnResult:
    accepted: bool
    action: str
    state: GameState
    previous_state: GameState
    message_key: str
    selected_target_id: str | None = None
    unlocked_evidence_ids: tuple[str, ...] = ()
    matched_intent_id: str | None = None
    answer_class: str | None = None
    followup_question: str | None = None
    matched_rule_ids: tuple[str, ...] = ()
    matched_slot_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    hint_id: str | None = None
    hint_text: str | None = None
    solve_status: str | None = None
    score: float | None = None
    covered_feedback_labels: tuple[str, ...] = ()
    missing_feedback_labels: tuple[str, ...] = ()
    spoiler_hits: tuple[SpoilerHit, ...] = ()
    model_context: Mapping[str, Any] | None = None


class CaseGameEngine:
    """Apply controlled player actions against one loaded case pack."""

    def __init__(self, pack: CasePack):
        self.pack = pack

    def new_state(self) -> GameState:
        return GameState(
            case_id=self.pack.case_id,
            stage=self.pack.meta.initial_stage,
            unlocked_evidence_ids=self.pack.sort_evidence_ids(
                set(self.pack.meta.initial_evidence_ids)
            ),
        )

    def state_from_mapping(self, value: Mapping[str, Any]) -> GameState:
        stage = value.get("stage", self.pack.meta.initial_stage)
        if not isinstance(stage, str) or stage not in self.pack.reveal_policy.stage_order:
            raise ValueError(f"unknown stage: {stage!r}")
        raw_ids = value.get("unlocked_evidence_ids", self.pack.meta.initial_evidence_ids)
        if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
            raise ValueError("unlocked_evidence_ids must be list[str]")
        unknown = sorted(set(raw_ids) - set(self.pack.evidence))
        if unknown:
            raise ValueError(f"unknown evidence ids: {unknown}")
        return GameState(
            case_id=self.pack.case_id,
            stage=stage,
            unlocked_evidence_ids=self.pack.sort_evidence_ids(set(raw_ids)),
            matched_slot_ids=self.pack.sort_slot_ids(set(value.get("matched_slot_ids", ()))),
            solve_slot_ids=self.pack.sort_slot_ids(set(value.get("solve_slot_ids", ()))),
            contradiction_ids=tuple(value.get("contradiction_ids", ())),
            hint_tier=int(value.get("hint_tier", 0)),
            solved=bool(value.get("solved", False)),
        )

    def apply(
        self,
        state: GameState,
        action: str,
        player_text: str = "",
        *,
        target_id: str | None = None,
    ) -> GameTurnResult:
        if state.case_id != self.pack.case_id:
            raise ValueError(
                f"state case_id {state.case_id!r} does not match pack {self.pack.case_id!r}"
            )
        if action not in self.pack.meta.allowed_actions:
            return self._result(
                False,
                action,
                state,
                state,
                "unknown_action",
                player_text=player_text,
                selected_target_id=target_id,
            )
        if self._has_cross_case_reference(player_text):
            return self._result(
                False,
                action,
                state,
                state,
                "cross_case_reference",
                player_text=player_text,
                selected_target_id=target_id,
            )
        if target_id is not None:
            selected_lead = next(
                (
                    lead
                    for lead in self.available_leads(state)
                    if lead["action"] == action and lead["target_id"] == target_id
                ),
                None,
            )
            if selected_lead is None:
                return self._result(
                    False,
                    action,
                    state,
                    state,
                    "target_unavailable",
                    player_text=player_text,
                    selected_target_id=target_id,
                )
        if action == "reset":
            new_state = self.new_state()
            return self._result(True, action, new_state, state, "reset", player_text=player_text)
        if action == "ask":
            return self._ask(state, player_text, selected_target_id=target_id)
        if action == "inspect":
            return self._inspect(state, player_text, selected_target_id=target_id)
        if action == "hypothesize":
            return self._hypothesize(state, player_text)
        if action == "hint":
            return self._hint(state, player_text)
        if action == "solve":
            return self._solve(state, player_text)
        if action == "recap":
            return self._result(True, action, state, state, "recap", player_text=player_text)
        return self._result(
            False, action, state, state, "unsupported_action", player_text=player_text
        )

    def available_leads(self, state: GameState) -> tuple[dict[str, Any], ...]:
        """Return safe, deterministic ask/inspect affordances for the current state."""

        current = set(state.unlocked_evidence_ids)
        current_rank = self.pack.stage_rank(state.stage)
        max_unlock_rank = min(current_rank + 1, max(self.pack.reveal_policy.stage_order.values()))
        leads: list[dict[str, Any]] = []
        seen_unlock_sets: set[tuple[str, tuple[str, ...]]] = set()

        for intent in self.pack.question_intents:
            if self.pack.stage_rank(intent.min_stage) > current_rank:
                continue
            unlocks = self._safe_unresolved_unlocks(
                intent.unlock_evidence_ids, current, max_unlock_rank
            )
            lead_key = ("ask", unlocks)
            if not unlocks or lead_key in seen_unlock_sets:
                continue
            prompt = _as_question(intent.aliases[0])
            if self._has_blocking_spoiler(prompt, state.stage):
                continue
            seen_unlock_sets.add(lead_key)
            leads.append(
                {
                    "target_id": intent.intent_id,
                    "action": "ask",
                    "label": f"提问：{intent.aliases[0]}",
                    "player_text": prompt,
                    "unlock_count": len(unlocks),
                }
            )

        for evidence_id, card in self.pack.evidence.items():
            if evidence_id in current or card.visibility == "hidden_truth":
                continue
            if self.pack.stage_rank(card.unlock_stage) > max_unlock_rank:
                continue
            targets = [
                condition.removeprefix("inspect:")
                for condition in card.unlock_conditions
                if condition.startswith("inspect:")
            ]
            human_targets = [target for target in targets if not _looks_internal_id(target)]
            if not human_targets:
                continue
            target = human_targets[-1]
            prompt = f"检查{target}，记录可验证的观察结果。"
            if self._has_blocking_spoiler(prompt, state.stage):
                continue
            unlocks = (evidence_id,)
            lead_key = ("inspect", unlocks)
            if lead_key in seen_unlock_sets:
                continue
            seen_unlock_sets.add(lead_key)
            leads.append(
                {
                    "target_id": evidence_id,
                    "action": "inspect",
                    "label": f"检查：{target}",
                    "player_text": prompt,
                    "unlock_count": 1,
                }
            )
        return tuple(leads)

    def scan_spoilers(
        self, text: str, stage: str, *, solved: bool = False
    ) -> tuple[SpoilerHit, ...]:
        if solved:
            return ()
        normalized = _normalize(text)
        hits: list[SpoilerHit] = []
        for spoiler in self.pack.spoiler_terms:
            if self.pack.stage_rank(stage) >= self.pack.stage_rank(spoiler.allowed_stage):
                continue
            for term in spoiler.terms:
                if _normalize(term) in normalized:
                    hits.append(
                        SpoilerHit(
                            spoiler_id=spoiler.spoiler_id,
                            term=term,
                            severity=spoiler.severity,
                            allowed_stage=spoiler.allowed_stage,
                            observed_stage=stage,
                        )
                    )
        return tuple(hits)

    def model_context(
        self, state: GameState, turn: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        safe_turn = dict(turn or {})
        if not state.solved:
            safe_turn["matched_slot_ids"] = []
        visible_evidence = [
            self._evidence_payload(card)
            for evidence_id, card in self.pack.evidence.items()
            if evidence_id in self._model_visible_evidence_ids(state)
        ]
        visible_timeline = [
            {
                "event_id": row.event_id,
                "order": row.order,
                "label": row.label,
                "text": row.text,
                "source_refs": list(row.source_refs),
            }
            for row in self.pack.timeline
            if row.visibility in {"initial", "visible_after_unlock"} or state.solved
        ]
        visible_characters = [
            {
                "character_id": row.character_id,
                "name": row.name,
                "role": row.role,
                "public_description": row.public_description,
                "source_refs": list(row.source_refs),
            }
            for row in self.pack.characters.values()
            if row.visibility in {"initial", "visible_after_unlock"} or state.solved
        ]
        context = {
            "case": {
                "case_id": self.pack.case_id,
                "case_pack_sha256": self.pack.manifest_sha256,
                "title_zh": self.pack.meta.title_zh,
                "title_en": self.pack.meta.title_en,
                "difficulty": self.pack.meta.difficulty,
                "stage": state.stage,
                "solved": state.solved,
                "win_condition": self.pack.meta.win_condition,
            },
            "roles": {
                "player": self.pack.meta.player_role,
                "model": self.pack.meta.model_role,
            },
            "allowed_actions": list(self.pack.meta.allowed_actions),
            "available_leads": list(self.available_leads(state)),
            "visible_evidence": visible_evidence,
            "visible_timeline": visible_timeline,
            "visible_characters": visible_characters,
            "state_summary": {
                "unlocked_evidence_ids": list(state.unlocked_evidence_ids),
                "matched_slot_ids": list(state.matched_slot_ids) if state.solved else [],
                "contradiction_ids": list(state.contradiction_ids),
                "hint_tier": state.hint_tier,
            },
            "turn": safe_turn,
        }
        if state.solved:
            context["solution_outline"] = [
                {
                    "slot_id": slot.slot_id,
                    "label": slot.label,
                    "weight": slot.weight,
                }
                for slot in self.pack.solution.required_slots
            ]
        return context

    def _ask(
        self,
        state: GameState,
        player_text: str,
        *,
        selected_target_id: str | None = None,
    ) -> GameTurnResult:
        intent = (
            next(
                (row for row in self.pack.question_intents if row.intent_id == selected_target_id),
                None,
            )
            if selected_target_id
            else self._match_intent(player_text)
        )
        if intent is None:
            return self._result(
                True,
                "ask",
                state,
                state,
                "ask_unknown",
                player_text=player_text,
                selected_target_id=selected_target_id,
                answer_class="unknown",
            )
        if self.pack.stage_rank(state.stage) < self.pack.stage_rank(intent.min_stage):
            return self._result(
                True,
                "ask",
                state,
                state,
                "ask_locked",
                player_text=player_text,
                selected_target_id=selected_target_id,
                matched_intent_id=intent.intent_id,
                answer_class="locked",
                followup_question=intent.followup_question,
            )
        new_state, unlocked = self._unlock_evidence(state, intent.unlock_evidence_ids)
        return self._result(
            True,
            "ask",
            new_state,
            state,
            "ask_matched",
            player_text=player_text,
            selected_target_id=selected_target_id,
            unlocked_evidence_ids=unlocked,
            matched_intent_id=intent.intent_id,
            answer_class=intent.answer_class,
            followup_question=intent.followup_question,
        )

    def _inspect(
        self,
        state: GameState,
        player_text: str,
        *,
        selected_target_id: str | None = None,
    ) -> GameTurnResult:
        normalized = _normalize(player_text)
        unlock_ids: list[str] = []
        inspected_id: str | None = None
        if selected_target_id is not None:
            inspected_id = selected_target_id
            unlock_ids.append(selected_target_id)
        for evidence_id, card in self.pack.evidence.items():
            if selected_target_id is not None:
                continue
            if _normalize(evidence_id) in normalized or _matches_any(player_text, (card.title,)):
                inspected_id = evidence_id
                unlock_ids.append(evidence_id)
            for condition in card.unlock_conditions:
                if not condition.startswith("inspect:"):
                    continue
                target = condition.removeprefix("inspect:")
                if _normalize(target) in normalized:
                    unlock_ids.append(evidence_id)
        new_state, unlocked = self._unlock_evidence(state, tuple(unlock_ids))
        if not unlocked and inspected_id is None:
            return self._result(
                True,
                "inspect",
                state,
                state,
                "inspect_unknown",
                player_text=player_text,
                selected_target_id=selected_target_id,
            )
        return self._result(
            True,
            "inspect",
            new_state,
            state,
            "inspect_matched",
            player_text=player_text,
            selected_target_id=selected_target_id,
            unlocked_evidence_ids=unlocked,
        )

    def _hypothesize(self, state: GameState, player_text: str) -> GameTurnResult:
        matched_rules = [
            rule
            for rule in self.pack.hypothesis_rules
            if _matches_any(player_text, rule.match_aliases)
        ]
        canonical_slot_ids = {
            slot.slot_id
            for slot in self.pack.solution.required_slots
            if self.solution_slot_matches(player_text, slot)
        }
        turn_slot_ids = canonical_slot_ids | {
            slot_id for rule in matched_rules for slot_id in rule.slot_matches
        }
        if not matched_rules and not turn_slot_ids:
            return self._result(
                True, "hypothesize", state, state, "hypothesis_miss", player_text=player_text
            )
        slot_ids = set(state.matched_slot_ids)
        slot_ids.update(turn_slot_ids)
        contradiction_ids = set(state.contradiction_ids)
        unlock_ids: list[str] = []
        for rule in matched_rules:
            contradiction_ids.update(rule.contradictions)
            unlock_ids.extend(rule.unlock_evidence_ids)
        stage = self._stage_after_hypothesis(state, matched_rules, canonical_slot_ids)
        base_state = GameState(
            case_id=state.case_id,
            stage=stage,
            unlocked_evidence_ids=state.unlocked_evidence_ids,
            matched_slot_ids=self.pack.sort_slot_ids(slot_ids),
            solve_slot_ids=state.solve_slot_ids,
            contradiction_ids=tuple(sorted(contradiction_ids)),
            hint_tier=state.hint_tier,
            solved=state.solved,
        )
        new_state, unlocked = self._unlock_evidence(base_state, tuple(unlock_ids))
        return self._result(
            True,
            "hypothesize",
            new_state,
            state,
            "hypothesis_matched",
            player_text=player_text,
            unlocked_evidence_ids=unlocked,
            matched_rule_ids=tuple(rule.rule_id for rule in matched_rules),
            matched_slot_ids=self.pack.sort_slot_ids(turn_slot_ids),
            contradiction_ids=tuple(sorted(contradiction_ids - set(state.contradiction_ids))),
        )

    def _hint(self, state: GameState, player_text: str) -> GameTurnResult:
        hint = self._next_hint(state, player_text)
        if hint is None:
            return self._result(
                True, "hint", state, state, "hint_exhausted", player_text=player_text
            )
        base_state = GameState(
            case_id=state.case_id,
            stage=state.stage,
            unlocked_evidence_ids=state.unlocked_evidence_ids,
            matched_slot_ids=state.matched_slot_ids,
            solve_slot_ids=state.solve_slot_ids,
            contradiction_ids=state.contradiction_ids,
            hint_tier=max(state.hint_tier, hint.tier),
            solved=state.solved,
        )
        cumulative_unlock_ids = tuple(
            evidence_id
            for authored_hint in sorted(self.pack.hints, key=lambda row: row.tier)
            if state.hint_tier < authored_hint.tier <= hint.tier
            and self.pack.stage_rank(authored_hint.stage) <= self.pack.stage_rank(state.stage)
            for evidence_id in authored_hint.unlock_evidence_ids
        )
        new_state, unlocked = self._unlock_evidence(
            base_state, cumulative_unlock_ids, advance_stage=False
        )
        return self._result(
            True,
            "hint",
            new_state,
            state,
            "hint",
            player_text=player_text,
            unlocked_evidence_ids=unlocked,
            hint_id=hint.hint_id,
            hint_text=hint.text,
        )

    def _solve(self, state: GameState, player_text: str) -> GameTurnResult:
        matched_slots, _ = self._score_solution(player_text)
        solve_slot_ids = self.pack.sort_slot_ids(set(state.solve_slot_ids) | set(matched_slots))
        score = sum(
            slot.weight
            for slot in self.pack.solution.required_slots
            if slot.slot_id in set(solve_slot_ids)
        )
        total = sum(slot.weight for slot in self.pack.solution.required_slots)
        ratio = score / total if total else 0.0
        solve_status = (
            "pass"
            if ratio >= self.pack.solution.pass_threshold
            else ("partial" if score > 0 else "miss")
        )
        slot_ids = set(state.matched_slot_ids) | set(matched_slots)
        solved = solve_status == "pass"
        covered_feedback_labels = tuple(
            slot.feedback_label
            for slot in self.pack.solution.required_slots
            if slot.slot_id in set(solve_slot_ids)
        )
        missing_feedback_labels = tuple(
            slot.feedback_label
            for slot in self.pack.solution.required_slots
            if slot.slot_id not in set(solve_slot_ids)
        )
        stage = "post_case" if solved else self.pack.reveal_policy.solution_stage
        unlocked_ids = set(state.unlocked_evidence_ids)
        if solved:
            unlocked_ids.update(self.pack.evidence)
        new_state = GameState(
            case_id=state.case_id,
            stage=stage,
            unlocked_evidence_ids=self.pack.sort_evidence_ids(unlocked_ids),
            matched_slot_ids=self.pack.sort_slot_ids(slot_ids),
            solve_slot_ids=solve_slot_ids,
            contradiction_ids=state.contradiction_ids,
            hint_tier=state.hint_tier,
            solved=solved,
        )
        return self._result(
            True,
            "solve",
            new_state,
            state,
            "solve",
            player_text=player_text,
            unlocked_evidence_ids=tuple(
                eid
                for eid in new_state.unlocked_evidence_ids
                if eid not in state.unlocked_evidence_ids
            ),
            matched_slot_ids=matched_slots,
            solve_status=solve_status,
            score=ratio,
            covered_feedback_labels=covered_feedback_labels,
            missing_feedback_labels=missing_feedback_labels,
        )

    def _result(
        self,
        accepted: bool,
        action: str,
        state: GameState,
        previous_state: GameState,
        message_key: str,
        *,
        player_text: str,
        selected_target_id: str | None = None,
        unlocked_evidence_ids: tuple[str, ...] = (),
        matched_intent_id: str | None = None,
        answer_class: str | None = None,
        followup_question: str | None = None,
        matched_rule_ids: tuple[str, ...] = (),
        matched_slot_ids: tuple[str, ...] = (),
        contradiction_ids: tuple[str, ...] = (),
        hint_id: str | None = None,
        hint_text: str | None = None,
        solve_status: str | None = None,
        score: float | None = None,
        covered_feedback_labels: tuple[str, ...] = (),
        missing_feedback_labels: tuple[str, ...] = (),
    ) -> GameTurnResult:
        turn = {
            "action": action,
            "accepted": accepted,
            "message_key": message_key,
            "selected_target_id": selected_target_id,
            "unlocked_evidence_ids": list(unlocked_evidence_ids),
            "matched_intent_id": matched_intent_id,
            "answer_class": answer_class,
            "followup_question": followup_question,
            "matched_rule_ids": list(matched_rule_ids),
            "matched_slot_ids": list(matched_slot_ids),
            "contradiction_ids": list(contradiction_ids),
            "hint_id": hint_id,
            "hint_text": hint_text,
            "solve_status": solve_status,
            "score": score,
            "covered_feedback_labels": list(covered_feedback_labels),
            "missing_feedback_labels": list(missing_feedback_labels),
        }
        hits = self.scan_spoilers(player_text, previous_state.stage, solved=previous_state.solved)
        return GameTurnResult(
            accepted=accepted,
            action=action,
            state=state,
            previous_state=previous_state,
            message_key=message_key,
            selected_target_id=selected_target_id,
            unlocked_evidence_ids=unlocked_evidence_ids,
            matched_intent_id=matched_intent_id,
            answer_class=answer_class,
            followup_question=followup_question,
            matched_rule_ids=matched_rule_ids,
            matched_slot_ids=matched_slot_ids,
            contradiction_ids=contradiction_ids,
            hint_id=hint_id,
            hint_text=hint_text,
            solve_status=solve_status,
            score=score,
            covered_feedback_labels=covered_feedback_labels,
            missing_feedback_labels=missing_feedback_labels,
            spoiler_hits=hits,
            model_context=self.model_context(state, turn),
        )

    def _match_intent(self, player_text: str) -> QuestionIntent | None:
        for intent in self.pack.question_intents:
            if _matches_any(player_text, intent.aliases):
                return intent
        return None

    def _next_hint(self, state: GameState, player_text: str) -> Hint | None:
        candidates = [
            hint
            for hint in self.pack.hints
            if hint.tier > state.hint_tier
            and self.pack.stage_rank(hint.stage) <= self.pack.stage_rank(state.stage)
        ]
        if not candidates:
            return None
        focused = [
            hint
            for hint in candidates
            if hint.focus_aliases and _matches_any(player_text, hint.focus_aliases)
        ]
        if focused:
            return min(focused, key=lambda hint: hint.tier)
        return min(candidates, key=lambda hint: hint.tier)

    def _stage_after_hypothesis(
        self, state: GameState, matched_rules, canonical_slot_ids: set[str]
    ) -> str:
        stage = state.stage
        if self.pack.stage_rank(stage) < self.pack.stage_rank("investigation"):
            has_supported_hypothesis = bool(canonical_slot_ids) or any(
                rule.slot_matches and not rule.contradictions for rule in matched_rules
            )
            if has_supported_hypothesis:
                stage = "investigation"
        may_enter_hypothesis = self.pack.stage_rank(state.stage) >= self.pack.stage_rank(
            "investigation"
        ) or any(rule.feedback_level == "solve_ready" for rule in matched_rules)
        if may_enter_hypothesis and self.pack.stage_rank(stage) >= self.pack.stage_rank(
            "investigation"
        ):
            advances_to_hypothesis = (
                len(canonical_slot_ids) >= 2
                and not any(rule.contradictions for rule in matched_rules)
            ) or any(
                rule.feedback_level in {"strong", "solve_ready"} and not rule.contradictions
                for rule in matched_rules
            )
            if advances_to_hypothesis and self.pack.stage_rank(stage) < self.pack.stage_rank(
                "hypothesis"
            ):
                stage = "hypothesis"
        return stage

    def _unlock_evidence(
        self,
        state: GameState,
        evidence_ids: tuple[str, ...],
        *,
        advance_stage: bool = True,
    ) -> tuple[GameState, tuple[str, ...]]:
        current = set(state.unlocked_evidence_ids)
        added: list[str] = []
        target_stage = state.stage
        for evidence_id in evidence_ids:
            card = self.pack.evidence[evidence_id]
            if card.visibility == "hidden_truth" and not state.solved:
                continue
            if evidence_id not in current:
                current.add(evidence_id)
                added.append(evidence_id)
            if advance_stage and self.pack.stage_rank(card.unlock_stage) > self.pack.stage_rank(
                target_stage
            ):
                target_stage = card.unlock_stage
        new_state = GameState(
            case_id=state.case_id,
            stage=target_stage,
            unlocked_evidence_ids=self.pack.sort_evidence_ids(current),
            matched_slot_ids=state.matched_slot_ids,
            solve_slot_ids=state.solve_slot_ids,
            contradiction_ids=state.contradiction_ids,
            hint_tier=state.hint_tier,
            solved=state.solved,
        )
        return new_state, self.pack.sort_evidence_ids(set(added))

    def _score_solution(self, player_text: str) -> tuple[tuple[str, ...], int]:
        matched: list[str] = []
        score = 0
        for slot in self.pack.solution.required_slots:
            if self.solution_slot_matches(player_text, slot):
                matched.append(slot.slot_id)
                score += slot.weight
        return tuple(matched), score

    def solution_slot_matches(self, text: str, slot: SolutionSlot) -> bool:
        """Apply the canonical solution matcher outside solve-state mutation."""

        return _matches_solution_slot(text, slot)

    def _safe_unresolved_unlocks(
        self,
        evidence_ids: tuple[str, ...],
        current: set[str],
        max_unlock_rank: int,
    ) -> tuple[str, ...]:
        return tuple(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id not in current
            and self.pack.evidence[evidence_id].visibility != "hidden_truth"
            and self.pack.stage_rank(self.pack.evidence[evidence_id].unlock_stage)
            <= max_unlock_rank
        )

    def _has_blocking_spoiler(self, text: str, stage: str) -> bool:
        return any(hit.severity in {"block", "repair"} for hit in self.scan_spoilers(text, stage))

    def _model_visible_evidence_ids(self, state: GameState) -> set[str]:
        allowed_by_stage = set(self.pack.reveal_policy.model_visible_by_stage.get(state.stage, ()))
        visible = set(state.unlocked_evidence_ids) & allowed_by_stage
        if not state.solved:
            visible = {
                evidence_id
                for evidence_id in visible
                if self.pack.evidence[evidence_id].visibility != "hidden_truth"
            }
        return visible

    def _evidence_payload(self, card: EvidenceCard) -> dict[str, Any]:
        return {
            "evidence_id": card.evidence_id,
            "title": card.title,
            "text": card.text,
            "source_refs": list(card.source_refs),
            "unsafe_sensitive": card.unsafe_sensitive,
        }

    def _has_cross_case_reference(self, text: str) -> bool:
        normalized = _normalize(text)
        return any(_normalize(term) in normalized for term in self.pack.meta.blocked_external_terms)


def _normalize(text: str) -> str:
    return "".join(_TERM_RE.findall(text.casefold()))


def _matches_any(text: str, aliases: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    for alias in aliases:
        alias_normalized = _normalize(alias)
        if alias_normalized and alias_normalized in normalized:
            return True
        terms = _alias_terms(alias)
        if len(terms) >= 2 and all(term in normalized for term in terms):
            return True
        if len(terms) == 1 and len(terms[0]) >= 2 and terms[0] in normalized:
            return True
    return False


def _matches_solution_slot(text: str, slot: SolutionSlot) -> bool:
    alias_match = _matches_solution_aliases(text, slot.aliases) or _matches_solution_aliases(
        text, (slot.label,)
    )
    group_match = bool(slot.match_groups) and all(
        any(_matches_any(text, (term,)) for term in group) for group in slot.match_groups
    )
    return alias_match or group_match


def _matches_solution_aliases(text: str, aliases: tuple[str, ...]) -> bool:
    """Match aliases without collapsing a relational phrase to one entity."""

    normalized = _normalize(text)
    for alias in aliases:
        alias_normalized = _normalize(alias)
        if alias_normalized and alias_normalized in normalized:
            return True
        terms = _alias_terms(alias)
        if len(terms) >= 2 and all(term in normalized for term in terms):
            return True
    return False


def _as_question(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith(("?", "？", "。", "!", "！")):
        return stripped
    return f"{stripped}？"


def _looks_internal_id(text: str) -> bool:
    return _INTERNAL_ID_RE.fullmatch(text.strip()) is not None


def _alias_terms(alias: str) -> tuple[str, ...]:
    cleaned = alias.casefold()
    for separator in _TERM_SEPARATORS:
        cleaned = cleaned.replace(separator, " ")
    terms = tuple(
        _normalize(term)
        for term in _TERM_RE.findall(cleaned)
        if len(_normalize(term)) >= 2 and _normalize(term) not in _STOP_TERMS
    )
    return terms


def _cjk_ngram_match(text: str, alias: str) -> bool:
    alias_chars = "".join(ch for ch in alias if "\u4e00" <= ch <= "\u9fff")
    text_chars = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    if len(alias_chars) < 4 or len(text_chars) < 4:
        return False
    alias_bigrams = {alias_chars[index : index + 2] for index in range(len(alias_chars) - 1)}
    text_bigrams = {text_chars[index : index + 2] for index in range(len(text_chars) - 1)}
    if not alias_bigrams:
        return False
    overlap = alias_bigrams & text_bigrams
    return len(overlap) >= 2 and len(overlap) / len(alias_bigrams) >= 0.35
