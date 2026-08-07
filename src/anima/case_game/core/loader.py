"""Load and validate Sherlock case-game packs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from anima.case_game.core.models import (
    CasePack,
    CharacterCard,
    Contradiction,
    EvalCase,
    EvidenceCard,
    GamePackMeta,
    Hint,
    HypothesisRule,
    QuestionIntent,
    RevealPolicy,
    SolutionSlot,
    SolutionSlots,
    SpoilerTerm,
    TimelineEvent,
)
from anima.utils.jsonl import iter_jsonl

REQUIRED_CASE_FILES: tuple[str, ...] = (
    "CASE_MANIFEST.json",
    "CASE_BIBLE.md",
    "game_pack.json",
    "evidence_cards.jsonl",
    "timeline.jsonl",
    "characters.jsonl",
    "question_intents.jsonl",
    "hypothesis_rules.jsonl",
    "contradictions.jsonl",
    "hints.jsonl",
    "solution_slots.json",
    "reveal_policy.json",
    "spoiler_terms.jsonl",
    "eval_cases.jsonl",
    "SOURCES.md",
    "LICENSE.md",
)


class CasePackValidationError(ValueError):
    """Aggregated case-pack validation failures."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def load_case_pack(level_dir: Path) -> CasePack:
    """Validate and load one structured Sherlock case level."""

    level_dir = Path(level_dir)
    errors: list[str] = []
    for name in REQUIRED_CASE_FILES:
        if not (level_dir / name).is_file():
            errors.append(f"{name}: file missing")
    if errors:
        raise CasePackValidationError(errors)

    manifest_raw = _read_json(errors, level_dir / "CASE_MANIFEST.json")
    manifest_hash = _sha256(level_dir / "CASE_MANIFEST.json")
    if isinstance(manifest_raw, Mapping):
        files = manifest_raw.get("files")
        if not isinstance(files, Mapping):
            errors.append("CASE_MANIFEST.json: files must be an object")
        else:
            for name, expected in files.items():
                if not isinstance(name, str) or not isinstance(expected, str):
                    errors.append("CASE_MANIFEST.json: files entries must be string:string")
                    continue
                path = level_dir / name
                if not path.is_file():
                    errors.append(f"CASE_MANIFEST.json: listed file missing: {name}")
                    continue
                actual = _sha256(path)
                if actual != expected:
                    errors.append(f"CASE_MANIFEST.json: hash mismatch for {name}")

    game_raw = _read_json(errors, level_dir / "game_pack.json")
    solution_raw = _read_json(errors, level_dir / "solution_slots.json")
    reveal_raw = _read_json(errors, level_dir / "reveal_policy.json")

    meta = _parse_game_meta(errors, game_raw)
    solution = _parse_solution(errors, solution_raw)
    reveal_policy = _parse_reveal_policy(errors, reveal_raw)

    evidence = _load_rows(errors, level_dir, "evidence_cards.jsonl", _parse_evidence, "evidence_id")
    timeline = tuple(
        sorted(
            _load_rows(errors, level_dir, "timeline.jsonl", _parse_timeline, "event_id").values(),
            key=lambda row: row.order,
        )
    )
    characters = _load_rows(errors, level_dir, "characters.jsonl", _parse_character, "character_id")
    question_intents = tuple(
        _load_rows(
            errors, level_dir, "question_intents.jsonl", _parse_question_intent, "intent_id"
        ).values()
    )
    hypothesis_rules = tuple(
        _load_rows(
            errors, level_dir, "hypothesis_rules.jsonl", _parse_hypothesis_rule, "rule_id"
        ).values()
    )
    contradictions = _load_rows(
        errors, level_dir, "contradictions.jsonl", _parse_contradiction, "contradiction_id"
    )
    hints = tuple(
        sorted(
            _load_rows(errors, level_dir, "hints.jsonl", _parse_hint, "hint_id").values(),
            key=lambda row: row.tier,
        )
    )
    spoiler_terms = tuple(
        _load_rows(
            errors, level_dir, "spoiler_terms.jsonl", _parse_spoiler_term, "spoiler_id"
        ).values()
    )
    eval_cases = tuple(
        _load_rows(errors, level_dir, "eval_cases.jsonl", _parse_eval_case, "eval_id").values()
    )

    if meta is not None and solution is not None and reveal_policy is not None:
        errors.extend(
            _validate_references(
                meta=meta,
                solution=solution,
                reveal_policy=reveal_policy,
                evidence=evidence,
                question_intents=question_intents,
                hypothesis_rules=hypothesis_rules,
                contradictions=contradictions,
                hints=hints,
                spoiler_terms=spoiler_terms,
                eval_cases=eval_cases,
            )
        )

    if errors:
        raise CasePackValidationError(errors)
    assert meta is not None and solution is not None and reveal_policy is not None
    return CasePack(
        root=level_dir,
        manifest_sha256=manifest_hash,
        meta=meta,
        evidence=evidence,
        timeline=timeline,
        characters=characters,
        question_intents=question_intents,
        hypothesis_rules=hypothesis_rules,
        contradictions=contradictions,
        hints=hints,
        solution=solution,
        reveal_policy=reveal_policy,
        spoiler_terms=spoiler_terms,
        eval_cases=eval_cases,
    )


def _read_json(errors: list[str], path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}: invalid JSON: {exc}")
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(
    errors: list[str],
    level_dir: Path,
    file_name: str,
    parser: Callable[[list[str], Mapping[str, Any]], Any],
    id_attr: str,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    try:
        parsed = list(iter_jsonl(level_dir / file_name))
    except ValueError as exc:
        errors.append(str(exc))
        return rows
    for line_no, record in parsed:
        row_errors: list[str] = []
        row = parser(row_errors, record)
        errors.extend(f"{file_name}:{line_no}: {error}" for error in row_errors)
        if row_errors:
            continue
        row_id = getattr(row, id_attr)
        if row_id in rows:
            errors.append(f"{file_name}:{line_no}: duplicate {id_attr} {row_id}")
        rows[row_id] = row
    return rows


def _req_str(errors: list[str], value: Mapping[str, Any], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"{key}: required non-empty string")
        return ""
    return raw


def _opt_str(value: Mapping[str, Any], key: str) -> str | None:
    raw = value.get(key)
    return raw if isinstance(raw, str) and raw.strip() else None


def _req_int(errors: list[str], value: Mapping[str, Any], key: str) -> int:
    raw = value.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        errors.append(f"{key}: required integer")
        return 0
    return raw


def _req_float(errors: list[str], value: Mapping[str, Any], key: str) -> float:
    raw = value.get(key)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        errors.append(f"{key}: required number")
        return 0.0
    return float(raw)


def _req_bool(errors: list[str], value: Mapping[str, Any], key: str) -> bool:
    raw = value.get(key)
    if not isinstance(raw, bool):
        errors.append(f"{key}: required boolean")
        return False
    return raw


def _str_tuple(errors: list[str], value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = value.get(key)
    if not isinstance(raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        errors.append(f"{key}: required list of non-empty strings")
        return ()
    if len(set(raw)) != len(raw):
        errors.append(f"{key}: entries must be unique")
    return tuple(raw)


def _str_tuple_groups(
    errors: list[str], value: Mapping[str, Any], key: str
) -> tuple[tuple[str, ...], ...]:
    raw = value.get(key)
    if not isinstance(raw, list) or not raw:
        errors.append(f"{key}: required non-empty list of string lists")
        return ()
    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(raw):
        if (
            not isinstance(group, list)
            or not group
            or any(not isinstance(item, str) or not item.strip() for item in group)
        ):
            errors.append(f"{key}[{index}]: required non-empty list of non-empty strings")
            continue
        if len(set(group)) != len(group):
            errors.append(f"{key}[{index}]: entries must be unique")
        groups.append(tuple(group))
    return tuple(groups)


def _parse_game_meta(errors: list[str], raw: Any) -> GamePackMeta | None:
    if not isinstance(raw, Mapping):
        errors.append("game_pack.json: expected object")
        return None
    return GamePackMeta(
        case_id=_req_str(errors, raw, "case_id"),
        case_prefix=_req_str(errors, raw, "case_prefix"),
        title_zh=_req_str(errors, raw, "title_zh"),
        title_en=_req_str(errors, raw, "title_en"),
        difficulty=_req_int(errors, raw, "difficulty"),
        allowed_actions=_str_tuple(errors, raw, "allowed_actions"),
        initial_stage=_req_str(errors, raw, "initial_stage"),
        initial_evidence_ids=_str_tuple(errors, raw, "initial_evidence_ids"),
        win_condition=_req_str(errors, raw, "win_condition"),
        player_role=_opt_str(raw, "player_role"),
        model_role=_opt_str(raw, "model_role"),
        blocked_external_terms=(
            _str_tuple(errors, raw, "blocked_external_terms")
            if "blocked_external_terms" in raw
            else ()
        ),
        final_recap_requirements=(
            _str_tuple(errors, raw, "final_recap_requirements")
            if "final_recap_requirements" in raw
            else ()
        ),
    )


def _parse_evidence(errors: list[str], raw: Mapping[str, Any]) -> EvidenceCard:
    return EvidenceCard(
        case_id=_req_str(errors, raw, "case_id"),
        evidence_id=_req_str(errors, raw, "evidence_id"),
        title=_req_str(errors, raw, "title"),
        text=_req_str(errors, raw, "text"),
        source_refs=_str_tuple(errors, raw, "source_refs"),
        unlock_stage=_req_str(errors, raw, "unlock_stage"),
        unlock_conditions=_str_tuple(errors, raw, "unlock_conditions"),
        visibility=_req_str(errors, raw, "visibility"),
        supports_slot_ids=_str_tuple(errors, raw, "supports_slot_ids"),
        spoiler=_req_bool(errors, raw, "spoiler"),
        unsafe_sensitive=_req_bool(errors, raw, "unsafe_sensitive"),
    )


def _parse_timeline(errors: list[str], raw: Mapping[str, Any]) -> TimelineEvent:
    return TimelineEvent(
        case_id=_req_str(errors, raw, "case_id"),
        event_id=_req_str(errors, raw, "event_id"),
        order=_req_int(errors, raw, "order"),
        label=_req_str(errors, raw, "label"),
        text=_req_str(errors, raw, "text"),
        source_refs=_str_tuple(errors, raw, "source_refs"),
        visibility=_req_str(errors, raw, "visibility"),
        unlock_conditions=_str_tuple(errors, raw, "unlock_conditions"),
    )


def _parse_character(errors: list[str], raw: Mapping[str, Any]) -> CharacterCard:
    return CharacterCard(
        case_id=_req_str(errors, raw, "case_id"),
        character_id=_req_str(errors, raw, "character_id"),
        name=_req_str(errors, raw, "name"),
        role=_req_str(errors, raw, "role"),
        public_description=_req_str(errors, raw, "public_description"),
        hidden_notes=_req_str(errors, raw, "hidden_notes"),
        visibility=_req_str(errors, raw, "visibility"),
        source_refs=_str_tuple(errors, raw, "source_refs"),
    )


def _parse_question_intent(errors: list[str], raw: Mapping[str, Any]) -> QuestionIntent:
    return QuestionIntent(
        case_id=_req_str(errors, raw, "case_id"),
        intent_id=_req_str(errors, raw, "intent_id"),
        aliases=_str_tuple(errors, raw, "aliases"),
        answer_class=_req_str(errors, raw, "answer_class"),
        min_stage=_req_str(errors, raw, "min_stage"),
        allowed_fact_ids=_str_tuple(errors, raw, "allowed_fact_ids"),
        unlock_evidence_ids=_str_tuple(errors, raw, "unlock_evidence_ids"),
        forbidden_spoiler_ids=_str_tuple(errors, raw, "forbidden_spoiler_ids"),
        followup_question=_req_str(errors, raw, "followup_question"),
    )


def _parse_hypothesis_rule(errors: list[str], raw: Mapping[str, Any]) -> HypothesisRule:
    return HypothesisRule(
        case_id=_req_str(errors, raw, "case_id"),
        rule_id=_req_str(errors, raw, "rule_id"),
        match_aliases=_str_tuple(errors, raw, "match_aliases"),
        slot_matches=_str_tuple(errors, raw, "slot_matches"),
        contradictions=_str_tuple(errors, raw, "contradictions"),
        unlock_evidence_ids=_str_tuple(errors, raw, "unlock_evidence_ids"),
        feedback_level=_req_str(errors, raw, "feedback_level"),
    )


def _parse_contradiction(errors: list[str], raw: Mapping[str, Any]) -> Contradiction:
    return Contradiction(
        case_id=_req_str(errors, raw, "case_id"),
        contradiction_id=_req_str(errors, raw, "contradiction_id"),
        text=_req_str(errors, raw, "text"),
        visibility=_req_str(errors, raw, "visibility"),
        forbidden_spoiler_ids=_str_tuple(errors, raw, "forbidden_spoiler_ids"),
    )


def _parse_hint(errors: list[str], raw: Mapping[str, Any]) -> Hint:
    return Hint(
        case_id=_req_str(errors, raw, "case_id"),
        hint_id=_req_str(errors, raw, "hint_id"),
        tier=_req_int(errors, raw, "tier"),
        stage=_req_str(errors, raw, "stage"),
        text=_req_str(errors, raw, "text"),
        focus_aliases=_str_tuple(errors, raw, "focus_aliases") if "focus_aliases" in raw else (),
        unlock_evidence_ids=_str_tuple(errors, raw, "unlock_evidence_ids"),
        forbidden_spoiler_ids=_str_tuple(errors, raw, "forbidden_spoiler_ids"),
    )


def _parse_solution(errors: list[str], raw: Any) -> SolutionSlots | None:
    if not isinstance(raw, Mapping):
        errors.append("solution_slots.json: expected object")
        return None
    rows_raw = raw.get("required_slots")
    if not isinstance(rows_raw, list):
        errors.append("solution_slots.json: required_slots must be a list")
        rows_raw = []
    rows: list[SolutionSlot] = []
    seen: set[str] = set()
    for index, item in enumerate(rows_raw):
        row_errors: list[str] = []
        if not isinstance(item, Mapping):
            errors.append(f"solution_slots.json: required_slots[{index}] must be an object")
            continue
        row = SolutionSlot(
            slot_id=_req_str(row_errors, item, "slot_id"),
            label=_req_str(row_errors, item, "label"),
            feedback_label=_req_str(row_errors, item, "feedback_label"),
            weight=_req_int(row_errors, item, "weight"),
            aliases=_str_tuple(row_errors, item, "aliases"),
            match_groups=_str_tuple_groups(row_errors, item, "match_groups"),
            visibility=_req_str(row_errors, item, "visibility"),
        )
        errors.extend(
            f"solution_slots.json: required_slots[{index}].{error}" for error in row_errors
        )
        if row.slot_id in seen:
            errors.append(f"solution_slots.json: duplicate slot_id {row.slot_id}")
        seen.add(row.slot_id)
        rows.append(row)
    return SolutionSlots(
        case_id=_req_str(errors, raw, "case_id"),
        pass_threshold=_req_float(errors, raw, "pass_threshold"),
        required_slots=tuple(rows),
    )


def _parse_reveal_policy(errors: list[str], raw: Any) -> RevealPolicy | None:
    if not isinstance(raw, Mapping):
        errors.append("reveal_policy.json: expected object")
        return None
    order_raw = raw.get("stage_order")
    if not isinstance(order_raw, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, int) for k, v in order_raw.items()
    ):
        errors.append("reveal_policy.json: stage_order must be string:int object")
        order_raw = {}
    visible_raw = raw.get("model_visible_by_stage")
    visible: dict[str, tuple[str, ...]] = {}
    if not isinstance(visible_raw, Mapping):
        errors.append("reveal_policy.json: model_visible_by_stage must be an object")
    else:
        for stage, ids in visible_raw.items():
            if (
                not isinstance(stage, str)
                or not isinstance(ids, list)
                or any(not isinstance(item, str) for item in ids)
            ):
                errors.append(
                    "reveal_policy.json: model_visible_by_stage entries must be string:list[string]"
                )
                continue
            visible[stage] = tuple(ids)
    return RevealPolicy(
        case_id=_req_str(errors, raw, "case_id"),
        stages=_str_tuple(errors, raw, "stages"),
        stage_order=dict(order_raw),
        default_stage=_req_str(errors, raw, "default_stage"),
        solution_stage=_req_str(errors, raw, "solution_stage"),
        model_visible_by_stage=visible,
        spoiler_slots=_str_tuple(errors, raw, "spoiler_slots"),
    )


def _parse_spoiler_term(errors: list[str], raw: Mapping[str, Any]) -> SpoilerTerm:
    return SpoilerTerm(
        case_id=_req_str(errors, raw, "case_id"),
        spoiler_id=_req_str(errors, raw, "spoiler_id"),
        terms=_str_tuple(errors, raw, "terms"),
        allowed_stage=_req_str(errors, raw, "allowed_stage"),
        severity=_req_str(errors, raw, "severity"),
    )


def _parse_eval_case(errors: list[str], raw: Mapping[str, Any]) -> EvalCase:
    initial_state = raw.get("initial_state")
    if not isinstance(initial_state, Mapping):
        errors.append("initial_state: required object")
        initial_state = {}
    expected_state_delta = raw.get("expected_state_delta")
    if not isinstance(expected_state_delta, Mapping):
        errors.append("expected_state_delta: required object")
        expected_state_delta = {}
    return EvalCase(
        case_id=_req_str(errors, raw, "case_id"),
        eval_id=_req_str(errors, raw, "eval_id"),
        category=_req_str(errors, raw, "category"),
        initial_state=initial_state,
        action=_req_str(errors, raw, "action"),
        player_text=_req_str(errors, raw, "player_text"),
        expected_state_delta=expected_state_delta,
        forbidden_spoiler_ids=_str_tuple(errors, raw, "forbidden_spoiler_ids"),
        model_owned_axes=_str_tuple(errors, raw, "model_owned_axes"),
        review_status=_req_str(errors, raw, "review_status"),
    )


def _validate_references(
    *,
    meta: GamePackMeta,
    solution: SolutionSlots,
    reveal_policy: RevealPolicy,
    evidence: Mapping[str, EvidenceCard],
    question_intents: tuple[QuestionIntent, ...],
    hypothesis_rules: tuple[HypothesisRule, ...],
    contradictions: Mapping[str, Contradiction],
    hints: tuple[Hint, ...],
    spoiler_terms: tuple[SpoilerTerm, ...],
    eval_cases: tuple[EvalCase, ...],
) -> list[str]:
    errors: list[str] = []
    prefix = meta.case_prefix
    evidence_ids = set(evidence)
    slot_ids = {slot.slot_id for slot in solution.required_slots}
    feedback_labels = [slot.feedback_label for slot in solution.required_slots]
    contradiction_ids = set(contradictions)
    spoiler_ids = {term.spoiler_id for term in spoiler_terms}
    known_spoiler_or_slot = slot_ids | spoiler_ids
    stages = set(reveal_policy.stage_order)

    def check_prefix(label: str, ids: tuple[str, ...] | set[str]) -> None:
        for row_id in ids:
            if not row_id.startswith(prefix):
                errors.append(f"{label}: {row_id} does not start with {prefix}")

    def check_known(label: str, ids: tuple[str, ...], allowed: set[str]) -> None:
        for row_id in ids:
            if row_id not in allowed:
                errors.append(f"{label}: unresolved id {row_id}")

    if meta.case_id != solution.case_id or meta.case_id != reveal_policy.case_id:
        errors.append("case_id: game, solution, and reveal policy must match")
    if meta.initial_stage not in stages:
        errors.append(f"initial_stage: unknown stage {meta.initial_stage}")
    check_known("initial_evidence_ids", meta.initial_evidence_ids, evidence_ids)
    check_prefix("evidence_id", set(evidence))
    check_prefix("solution slot", slot_ids)
    if len(set(feedback_labels)) != len(feedback_labels):
        errors.append("solution_slots.json: feedback_label values must be unique")
    for slot in solution.required_slots:
        if slot.weight <= 0:
            errors.append(f"{slot.slot_id}: weight must be positive")
        if slot.visibility != "hidden_truth":
            errors.append(f"{slot.slot_id}: solution slot visibility must be hidden_truth")
    check_prefix("contradiction_id", contradiction_ids)
    check_prefix("spoiler_id", spoiler_ids)

    for row in evidence.values():
        if row.case_id != meta.case_id:
            errors.append(f"{row.evidence_id}: case_id mismatch")
        if row.unlock_stage not in stages:
            errors.append(f"{row.evidence_id}: unknown unlock_stage {row.unlock_stage}")
        check_known(f"{row.evidence_id}.supports_slot_ids", row.supports_slot_ids, slot_ids)
    for stage, ids in reveal_policy.model_visible_by_stage.items():
        if stage not in stages:
            errors.append(f"model_visible_by_stage: unknown stage {stage}")
        check_known(f"model_visible_by_stage.{stage}", ids, evidence_ids)
    check_known("spoiler_slots", reveal_policy.spoiler_slots, slot_ids)

    for row in question_intents:
        if row.case_id != meta.case_id:
            errors.append(f"{row.intent_id}: case_id mismatch")
        if not row.intent_id.startswith(prefix):
            errors.append(f"{row.intent_id}: bad prefix")
        if row.min_stage not in stages:
            errors.append(f"{row.intent_id}: unknown min_stage {row.min_stage}")
        check_known(f"{row.intent_id}.unlock_evidence_ids", row.unlock_evidence_ids, evidence_ids)
        check_known(
            f"{row.intent_id}.forbidden_spoiler_ids",
            row.forbidden_spoiler_ids,
            known_spoiler_or_slot,
        )
    for row in hypothesis_rules:
        if row.case_id != meta.case_id:
            errors.append(f"{row.rule_id}: case_id mismatch")
        if not row.rule_id.startswith(prefix):
            errors.append(f"{row.rule_id}: bad prefix")
        check_known(f"{row.rule_id}.slot_matches", row.slot_matches, slot_ids)
        check_known(f"{row.rule_id}.contradictions", row.contradictions, contradiction_ids)
        check_known(f"{row.rule_id}.unlock_evidence_ids", row.unlock_evidence_ids, evidence_ids)
    for row in contradictions.values():
        if row.case_id != meta.case_id:
            errors.append(f"{row.contradiction_id}: case_id mismatch")
        check_known(
            f"{row.contradiction_id}.forbidden_spoiler_ids",
            row.forbidden_spoiler_ids,
            known_spoiler_or_slot,
        )
    for row in hints:
        if row.case_id != meta.case_id:
            errors.append(f"{row.hint_id}: case_id mismatch")
        if row.stage not in stages:
            errors.append(f"{row.hint_id}: unknown stage {row.stage}")
        check_known(f"{row.hint_id}.unlock_evidence_ids", row.unlock_evidence_ids, evidence_ids)
        check_known(
            f"{row.hint_id}.forbidden_spoiler_ids", row.forbidden_spoiler_ids, known_spoiler_or_slot
        )
    for row in spoiler_terms:
        if row.case_id != meta.case_id:
            errors.append(f"{row.spoiler_id}: case_id mismatch")
        if row.allowed_stage not in stages:
            errors.append(f"{row.spoiler_id}: unknown allowed_stage {row.allowed_stage}")
    for row in eval_cases:
        if row.case_id != meta.case_id:
            errors.append(f"{row.eval_id}: case_id mismatch")
        check_known(
            f"{row.eval_id}.initial_state.unlocked_evidence_ids",
            tuple(row.initial_state.get("unlocked_evidence_ids", ())),
            evidence_ids,
        )
        check_known(
            f"{row.eval_id}.forbidden_spoiler_ids",
            row.forbidden_spoiler_ids,
            known_spoiler_or_slot,
        )
    return errors
