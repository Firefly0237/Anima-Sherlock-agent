"""Data contracts for the two-level Sherlock case game.

The case pack owns truth. These dataclasses intentionally model only the v1
case-game schema needed by the current two Sherlock levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GamePackMeta:
    case_id: str
    case_prefix: str
    title_zh: str
    title_en: str
    difficulty: int
    allowed_actions: tuple[str, ...]
    initial_stage: str
    initial_evidence_ids: tuple[str, ...]
    win_condition: str
    player_role: str | None
    model_role: str | None
    blocked_external_terms: tuple[str, ...]
    final_recap_requirements: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCard:
    case_id: str
    evidence_id: str
    title: str
    text: str
    source_refs: tuple[str, ...]
    unlock_stage: str
    unlock_conditions: tuple[str, ...]
    visibility: str
    supports_slot_ids: tuple[str, ...]
    spoiler: bool
    unsafe_sensitive: bool


@dataclass(frozen=True)
class TimelineEvent:
    case_id: str
    event_id: str
    order: int
    label: str
    text: str
    source_refs: tuple[str, ...]
    visibility: str
    unlock_conditions: tuple[str, ...]


@dataclass(frozen=True)
class CharacterCard:
    case_id: str
    character_id: str
    name: str
    role: str
    public_description: str
    hidden_notes: str
    visibility: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class QuestionIntent:
    case_id: str
    intent_id: str
    aliases: tuple[str, ...]
    answer_class: str
    min_stage: str
    allowed_fact_ids: tuple[str, ...]
    unlock_evidence_ids: tuple[str, ...]
    forbidden_spoiler_ids: tuple[str, ...]
    followup_question: str


@dataclass(frozen=True)
class HypothesisRule:
    case_id: str
    rule_id: str
    match_aliases: tuple[str, ...]
    slot_matches: tuple[str, ...]
    contradictions: tuple[str, ...]
    unlock_evidence_ids: tuple[str, ...]
    feedback_level: str


@dataclass(frozen=True)
class Contradiction:
    case_id: str
    contradiction_id: str
    text: str
    visibility: str
    forbidden_spoiler_ids: tuple[str, ...]


@dataclass(frozen=True)
class Hint:
    case_id: str
    hint_id: str
    tier: int
    stage: str
    text: str
    focus_aliases: tuple[str, ...]
    unlock_evidence_ids: tuple[str, ...]
    forbidden_spoiler_ids: tuple[str, ...]


@dataclass(frozen=True)
class SolutionSlot:
    slot_id: str
    label: str
    feedback_label: str
    weight: int
    aliases: tuple[str, ...]
    match_groups: tuple[tuple[str, ...], ...]
    visibility: str


@dataclass(frozen=True)
class SolutionSlots:
    case_id: str
    pass_threshold: float
    required_slots: tuple[SolutionSlot, ...]


@dataclass(frozen=True)
class RevealPolicy:
    case_id: str
    stages: tuple[str, ...]
    stage_order: Mapping[str, int]
    default_stage: str
    solution_stage: str
    model_visible_by_stage: Mapping[str, tuple[str, ...]]
    spoiler_slots: tuple[str, ...]


@dataclass(frozen=True)
class SpoilerTerm:
    case_id: str
    spoiler_id: str
    terms: tuple[str, ...]
    allowed_stage: str
    severity: str


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    eval_id: str
    category: str
    initial_state: Mapping[str, Any]
    action: str
    player_text: str
    expected_state_delta: Mapping[str, Any]
    forbidden_spoiler_ids: tuple[str, ...]
    model_owned_axes: tuple[str, ...]
    review_status: str


@dataclass(frozen=True)
class SpoilerHit:
    spoiler_id: str
    term: str
    severity: str
    allowed_stage: str
    observed_stage: str


@dataclass(frozen=True)
class CasePack:
    root: Any
    manifest_sha256: str
    meta: GamePackMeta
    evidence: Mapping[str, EvidenceCard]
    timeline: tuple[TimelineEvent, ...]
    characters: Mapping[str, CharacterCard]
    question_intents: tuple[QuestionIntent, ...]
    hypothesis_rules: tuple[HypothesisRule, ...]
    contradictions: Mapping[str, Contradiction]
    hints: tuple[Hint, ...]
    solution: SolutionSlots
    reveal_policy: RevealPolicy
    spoiler_terms: tuple[SpoilerTerm, ...]
    eval_cases: tuple[EvalCase, ...]

    @property
    def case_id(self) -> str:
        return self.meta.case_id

    @property
    def case_prefix(self) -> str:
        return self.meta.case_prefix

    def stage_rank(self, stage: str) -> int:
        return int(self.reveal_policy.stage_order[stage])

    def sort_evidence_ids(self, evidence_ids: set[str] | tuple[str, ...]) -> tuple[str, ...]:
        selected = set(evidence_ids)
        return tuple(evidence_id for evidence_id in self.evidence if evidence_id in selected)

    def sort_slot_ids(self, slot_ids: set[str] | tuple[str, ...]) -> tuple[str, ...]:
        selected = set(slot_ids)
        return tuple(
            slot.slot_id for slot in self.solution.required_slots if slot.slot_id in selected
        )
