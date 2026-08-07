"""Persona-pack, dialogue-state, and memory contracts.

The maintained design is documented in ``docs/ARCHITECTURE.md`` and
``docs/TRAINING_AND_EVALUATION.md``. This module uses only the standard library.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

ANSWER_MODES: tuple[str, ...] = ("lore", "memory", "general", "abstain", "refuse")
MEMORY_OP_KINDS: tuple[str, ...] = ("add", "update", "delete", "noop")
MEMORY_STATUSES: tuple[str, ...] = ("active", "superseded", "deleted")
REVIEW_STATUSES: tuple[str, ...] = (
    "draft",
    "published",
    "pilot_pass",
    "pilot_fail",
    "human_pass",
    "human_fail",
    "formal_pass",
    "formal_fail",
)
FORMAL_PASS_REVIEW_STATUSES: frozenset[str] = frozenset({"human_pass", "formal_pass"})
FORMAL_FAIL_REVIEW_STATUSES: frozenset[str] = frozenset({"human_fail", "formal_fail"})
CREATED_BY_VALUES: tuple[str, ...] = ("human", "curated")
STYLE_KINDS: tuple[str, ...] = ("positive", "negative")
STYLE_SEMANTIC_TYPES: tuple[str, ...] = (
    "lore",
    "general",
    "ooc",
    "future",
    "unsafe",
    "memory_unknown",
)
PACK_LANGUAGE = "zh-CN"

FACT_ID_RE = re.compile(r"^lore_\d{6}$")
EVENT_ID_RE = re.compile(r"^tl_\d{6}$")
REL_ID_RE = re.compile(r"^rel_\d{4}$")
STYLE_ID_RE = re.compile(r"^style_\d{6}$")
PERSONA_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Memory predicates whose names carry executable/prompt semantics are banned
# outright: user memory stores declarative facts, never instructions.
FORBIDDEN_MEMORY_PREDICATES = frozenset(
    {
        "system_prompt",
        "instruction",
        "instructions",
        "role_override",
        "prompt",
        "command",
        "tool_call",
    }
)

REQUIRED_PROFILE_FIELDS: tuple[str, ...] = (
    "identity",
    "role",
    "self_reference",
    "address_rules",
    "values",
    "goals",
    "fears",
    "taboos",
    "decision_tendencies",
    "speech_style",
    "lexicon",
    "verbosity",
    "humor",
    "emotional_baseline",
    "relationships",
    "knowledge_scope",
    "unknown_response_style",
    "safety_behavior",
    "anti_ooc_rules",
    "assistant_markers_forbidden",
)


class PersonaContractError(ValueError):
    """A persona-layer record violates its declared contract."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _req_str(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    *,
    pattern: re.Pattern[str] | None = None,
    choices: tuple[str, ...] | None = None,
) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key}: required non-empty string")
        return None
    if pattern is not None and not pattern.match(value):
        errors.append(f"{key}: {value!r} does not match required pattern {pattern.pattern}")
    if choices is not None and value not in choices:
        errors.append(f"{key}: {value!r} not in {choices}")
    return value


def _opt_str(errors: list[str], mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key}: must be null or a non-empty string")
        return None
    return value


def _req_bool(errors: list[str], mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        errors.append(f"{key}: required boolean")
        return False
    return value


def _req_int(errors: list[str], mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{key}: required integer")
        return 0
    return value


def _str_tuple(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = True,
    item_pattern: re.Pattern[str] | None = None,
) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        errors.append(f"{key}: required list of non-empty strings")
        return ()
    if len(set(value)) != len(value):
        errors.append(f"{key}: entries must be unique")
    if not allow_empty and not value:
        errors.append(f"{key}: must not be empty")
    if item_pattern is not None:
        for item in value:
            if not item_pattern.match(item):
                errors.append(
                    f"{key}: entry {item!r} does not match required pattern {item_pattern.pattern}"
                )
    return tuple(value)


def _str_groups(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    *,
    min_groups: int = 0,
) -> tuple[tuple[str, ...], ...]:
    value = mapping.get(key)
    if not isinstance(value, list):
        errors.append(f"{key}: required list of alias groups")
        return ()
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for index, group in enumerate(value):
        if (
            not isinstance(group, list)
            or not group
            or any(not isinstance(item, str) or not item.strip() for item in group)
        ):
            errors.append(f"{key}[{index}]: required non-empty list of non-empty strings")
            continue
        if len(set(group)) != len(group):
            errors.append(f"{key}[{index}]: entries must be unique")
        normalized = tuple(sorted("".join(item.casefold().split()) for item in group))
        if normalized in seen:
            errors.append(f"{key}[{index}]: duplicate alias group")
        seen.add(normalized)
        groups.append(tuple(group))
    if len(groups) < min_groups:
        errors.append(f"{key}: requires at least {min_groups} alias groups")
    return tuple(groups)


@dataclass(frozen=True)
class PackLicense:
    name: str
    source: str


@dataclass(frozen=True)
class PackManifest:
    """manifest contract."""

    persona_id: str
    version: str
    language: str
    license: PackLicense
    public: bool
    content_sha256: str
    knowledge_cutoff: str | None
    created_by: str
    human_reviewed: bool
    formal_reviewed: bool
    reviewer_types: tuple[str, ...]
    review_policy_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PackManifest":
        errors: list[str] = []
        persona_id = _req_str(errors, value, "persona_id", pattern=PERSONA_ID_RE)
        version = _req_str(errors, value, "version", pattern=SEMVER_RE)
        language = _req_str(errors, value, "language", choices=(PACK_LANGUAGE,))
        license_raw = value.get("license")
        license_obj = PackLicense("", "")
        if not isinstance(license_raw, Mapping):
            errors.append("license: required object with name/source")
        else:
            license_errors: list[str] = []
            name = _req_str(license_errors, license_raw, "name")
            source = _req_str(license_errors, license_raw, "source")
            errors.extend(f"license.{e}" for e in license_errors)
            license_obj = PackLicense(name or "", source or "")
        public = _req_bool(errors, value, "public")
        content_sha256 = _req_str(errors, value, "content_sha256", pattern=SHA256_RE)
        knowledge_cutoff = _opt_str(errors, value, "knowledge_cutoff")
        created_by = _req_str(errors, value, "created_by", choices=CREATED_BY_VALUES)
        human_reviewed = _req_bool(errors, value, "human_reviewed")
        formal_reviewed_raw = value.get("formal_reviewed", human_reviewed)
        if not isinstance(formal_reviewed_raw, bool):
            errors.append("formal_reviewed: required boolean when provided")
            formal_reviewed = False
        else:
            formal_reviewed = formal_reviewed_raw
        reviewer_types = (
            _str_tuple(errors, value, "reviewer_types")
            if "reviewer_types" in value
            else (("human",) if human_reviewed else ())
        )
        invalid_reviewer_types = sorted(set(reviewer_types) - {"human", "automated"})
        if invalid_reviewer_types:
            errors.append(f"reviewer_types: unsupported entries {invalid_reviewer_types}")
        review_policy_ids = (
            _str_tuple(errors, value, "review_policy_ids")
            if "review_policy_ids" in value
            else (("legacy_human_review",) if human_reviewed else ())
        )
        if formal_reviewed and not reviewer_types:
            errors.append("reviewer_types: required when formal_reviewed=true")
        if formal_reviewed and not review_policy_ids:
            errors.append("review_policy_ids: required when formal_reviewed=true")
        if human_reviewed and "human" not in reviewer_types:
            errors.append("reviewer_types: must include human when human_reviewed=true")
        if human_reviewed and not formal_reviewed:
            errors.append("formal_reviewed: must be true when human_reviewed=true")
        if errors:
            raise PersonaContractError(errors)
        return cls(
            persona_id=persona_id or "",
            version=version or "",
            language=language or "",
            license=license_obj,
            public=public,
            content_sha256=content_sha256 or "",
            knowledge_cutoff=knowledge_cutoff,
            created_by=created_by or "",
            human_reviewed=human_reviewed,
            formal_reviewed=formal_reviewed,
            reviewer_types=reviewer_types,
            review_policy_ids=review_policy_ids,
        )


def is_formal_pass(review_status: str) -> bool:
    """Return whether a row passed an accepted, provenance-bearing review policy."""

    return review_status in FORMAL_PASS_REVIEW_STATUSES


def is_formal_fail(review_status: str) -> bool:
    """Return whether a row explicitly failed an accepted review policy."""

    return review_status in FORMAL_FAIL_REVIEW_STATUSES


@dataclass(frozen=True)
class LoreFact:
    """structured lore fact."""

    fact_id: str
    subject: str
    predicate: str
    object: str
    aliases: tuple[str, ...]
    valid_from: str | None
    valid_to: str | None
    known_by_persona: bool
    persona_response: str | None
    answer_slots: tuple[tuple[str, ...], ...]
    boundary_forbidden_claims: tuple[str, ...]
    source_ref: str
    review_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LoreFact":
        errors: list[str] = []
        fact_id = _req_str(errors, value, "fact_id", pattern=FACT_ID_RE)
        subject = _req_str(errors, value, "subject")
        predicate = _req_str(errors, value, "predicate")
        object_ = _req_str(errors, value, "object")
        aliases = _str_tuple(errors, value, "aliases")
        valid_from = _opt_str(errors, value, "valid_from")
        valid_to = _opt_str(errors, value, "valid_to")
        known_by_persona = _req_bool(errors, value, "known_by_persona")
        persona_response = _opt_str(errors, value, "persona_response")
        answer_slots = _str_groups(
            errors,
            value,
            "answer_slots",
            min_groups=2 if known_by_persona else 0,
        )
        boundary_forbidden_claims = _str_tuple(
            errors,
            value,
            "boundary_forbidden_claims",
            allow_empty=known_by_persona,
        )
        source_ref = _req_str(errors, value, "source_ref")
        review_status = _req_str(errors, value, "review_status", choices=REVIEW_STATUSES)
        if known_by_persona:
            if persona_response is None:
                errors.append("persona_response: required for persona-known facts")
            if boundary_forbidden_claims:
                errors.append(
                    "boundary_forbidden_claims: persona-known facts must not declare future claims"
                )
            if persona_response is not None:
                compact_response = "".join(persona_response.casefold().split())
                for index, group in enumerate(answer_slots):
                    if not any(
                        "".join(alias.casefold().split()) in compact_response for alias in group
                    ):
                        errors.append(f"answer_slots[{index}]: no alias occurs in persona_response")
        else:
            if persona_response is not None:
                errors.append("persona_response: must be null for persona-unknown facts")
            if answer_slots:
                errors.append("answer_slots: persona-unknown facts must not declare answer slots")
        if errors:
            raise PersonaContractError(errors)
        return cls(
            fact_id=fact_id or "",
            subject=subject or "",
            predicate=predicate or "",
            object=object_ or "",
            aliases=aliases,
            valid_from=valid_from,
            valid_to=valid_to,
            known_by_persona=known_by_persona,
            persona_response=persona_response,
            answer_slots=answer_slots,
            boundary_forbidden_claims=boundary_forbidden_claims,
            source_ref=source_ref or "",
            review_status=review_status or "",
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["answer_slots"] = [list(group) for group in self.answer_slots]
        value["boundary_forbidden_claims"] = list(self.boundary_forbidden_claims)
        return value


@dataclass(frozen=True)
class TimelineEvent:
    event_id: str
    order: int
    time_label: str | None
    title: str
    description: str
    source_ref: str
    review_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TimelineEvent":
        errors: list[str] = []
        event_id = _req_str(errors, value, "event_id", pattern=EVENT_ID_RE)
        order = _req_int(errors, value, "order")
        time_label = _opt_str(errors, value, "time_label")
        title = _req_str(errors, value, "title")
        description = _req_str(errors, value, "description")
        source_ref = _req_str(errors, value, "source_ref")
        review_status = _req_str(errors, value, "review_status", choices=REVIEW_STATUSES)
        if errors:
            raise PersonaContractError(errors)
        return cls(
            event_id=event_id or "",
            order=order,
            time_label=time_label,
            title=title or "",
            description=description or "",
            source_ref=source_ref or "",
            review_status=review_status or "",
        )


@dataclass(frozen=True)
class Relationship:
    rel_id: str
    subject: str
    relation: str
    object: str
    stance: str
    description: str
    source_ref: str
    review_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Relationship":
        errors: list[str] = []
        rel_id = _req_str(errors, value, "rel_id", pattern=REL_ID_RE)
        subject = _req_str(errors, value, "subject")
        relation = _req_str(errors, value, "relation")
        object_ = _req_str(errors, value, "object")
        stance = _req_str(errors, value, "stance")
        description = _req_str(errors, value, "description")
        source_ref = _req_str(errors, value, "source_ref")
        review_status = _req_str(errors, value, "review_status", choices=REVIEW_STATUSES)
        if errors:
            raise PersonaContractError(errors)
        return cls(
            rel_id=rel_id or "",
            subject=subject or "",
            relation=relation or "",
            object=object_ or "",
            stance=stance or "",
            description=description or "",
            source_ref=source_ref or "",
            review_status=review_status or "",
        )


@dataclass(frozen=True)
class StyleExample:
    example_id: str
    kind: str
    prompt: str
    response: str
    semantic_type: str
    gold_lore_ids: tuple[str, ...]
    violation_tags: tuple[str, ...]
    review_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StyleExample":
        errors: list[str] = []
        example_id = _req_str(errors, value, "example_id", pattern=STYLE_ID_RE)
        kind = _req_str(errors, value, "kind", choices=STYLE_KINDS)
        prompt = _req_str(errors, value, "prompt")
        response = _req_str(errors, value, "response")
        semantic_type = _req_str(errors, value, "semantic_type", choices=STYLE_SEMANTIC_TYPES)
        gold_lore_ids = _str_tuple(errors, value, "gold_lore_ids", item_pattern=FACT_ID_RE)
        violation_tags = _str_tuple(errors, value, "violation_tags")
        review_status = _req_str(errors, value, "review_status", choices=REVIEW_STATUSES)
        if kind == "negative" and not violation_tags:
            errors.append(
                "violation_tags: negative examples must declare at least one violation tag"
            )
        if kind == "positive" and violation_tags:
            errors.append("violation_tags: positive examples must not carry violation tags")
        if semantic_type == "lore" and not gold_lore_ids:
            errors.append("gold_lore_ids: lore style examples must cite at least one fact")
        if semantic_type != "lore" and gold_lore_ids:
            errors.append("gold_lore_ids: only lore style examples may cite lore facts")
        if errors:
            raise PersonaContractError(errors)
        return cls(
            example_id=example_id or "",
            kind=kind or "",
            prompt=prompt or "",
            response=response or "",
            semantic_type=semantic_type or "",
            gold_lore_ids=gold_lore_ids,
            violation_tags=violation_tags,
            review_status=review_status or "",
        )


@dataclass(frozen=True)
class SafetyPolicy:
    """safety.json: refusal behavior plus the pack-declared memory predicate allowlist."""

    refusal_style: str
    hard_refuse_categories: tuple[str, ...]
    in_character_refusal_examples: tuple[str, ...]
    memory_predicate_allowlist: tuple[str, ...]
    forbidden_assistant_markers: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SafetyPolicy":
        errors: list[str] = []
        refusal_style = _req_str(errors, value, "refusal_style")
        hard_refuse_categories = _str_tuple(errors, value, "hard_refuse_categories")
        in_character_refusal_examples = _str_tuple(errors, value, "in_character_refusal_examples")
        allowlist = _str_tuple(
            errors,
            value,
            "memory_predicate_allowlist",
            allow_empty=False,
            item_pattern=PREDICATE_RE,
        )
        for predicate in allowlist:
            if predicate in FORBIDDEN_MEMORY_PREDICATES:
                errors.append(
                    f"memory_predicate_allowlist: {predicate!r} carries executable semantics and is banned"
                )
        forbidden_assistant_markers = _str_tuple(
            errors, value, "forbidden_assistant_markers", allow_empty=False
        )
        if errors:
            raise PersonaContractError(errors)
        return cls(
            refusal_style=refusal_style or "",
            hard_refuse_categories=hard_refuse_categories,
            in_character_refusal_examples=in_character_refusal_examples,
            memory_predicate_allowlist=allowlist,
            forbidden_assistant_markers=forbidden_assistant_markers,
        )


@dataclass(frozen=True)
class MemoryOp:
    """memory operation emitted inside anima_state.

    The runtime re-binds `subject` and `source_message_id` to authenticated
    values; this schema only enforces shape, not trust.
    """

    op: str
    subject: str
    predicate: str | None
    object: str | None
    source_message_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryOp":
        errors: list[str] = []
        op = _req_str(errors, value, "op", choices=MEMORY_OP_KINDS)
        subject = _req_str(errors, value, "subject")
        source_message_id = _req_str(errors, value, "source_message_id")
        predicate = _opt_str(errors, value, "predicate")
        object_ = _opt_str(errors, value, "object")
        if predicate is not None and not PREDICATE_RE.match(predicate):
            errors.append(
                f"predicate: {predicate!r} does not match required pattern {PREDICATE_RE.pattern}"
            )
        if op in ("add", "update"):
            if predicate is None:
                errors.append(f"predicate: required for op={op}")
            if object_ is None:
                errors.append(f"object: required for op={op}")
        elif op == "delete" and predicate is None:
            errors.append("predicate: required for op=delete")
        if errors:
            raise PersonaContractError(errors)
        return cls(
            op=op or "",
            subject=subject or "",
            predicate=predicate,
            object=object_,
            source_message_id=source_message_id or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRecord:
    """long-term memory row."""

    memory_id: str
    user_id: str
    persona_id: str
    subject: str
    predicate: str
    object: str
    valid_from: str | None
    valid_to: str | None
    source_message_id: str
    confidence: float
    status: str
    created_at: str
    updated_at: str
    source_session_id: str | None = None
    source_turn_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryRecord":
        errors: list[str] = []
        memory_id = _req_str(errors, value, "memory_id")
        user_id = _req_str(errors, value, "user_id")
        persona_id = _req_str(errors, value, "persona_id")
        subject = _req_str(errors, value, "subject")
        predicate = _req_str(errors, value, "predicate", pattern=PREDICATE_RE)
        object_ = _req_str(errors, value, "object")
        valid_from = _opt_str(errors, value, "valid_from")
        valid_to = _opt_str(errors, value, "valid_to")
        source_message_id = _req_str(errors, value, "source_message_id")
        source_session_id = _opt_str(errors, value, "source_session_id")
        source_turn_id = _opt_str(errors, value, "source_turn_id")
        if (source_session_id is None) != (source_turn_id is None):
            errors.append(
                "source_session_id and source_turn_id must either both be present or both be null"
            )
        confidence = value.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= confidence <= 1.0
        ):
            errors.append("confidence: required number in [0, 1]")
            confidence = 0.0
        status = _req_str(errors, value, "status", choices=MEMORY_STATUSES)
        created_at = _req_str(errors, value, "created_at")
        updated_at = _req_str(errors, value, "updated_at")
        if errors:
            raise PersonaContractError(errors)
        return cls(
            memory_id=memory_id or "",
            user_id=user_id or "",
            persona_id=persona_id or "",
            subject=subject or "",
            predicate=predicate or "",
            object=object_ or "",
            valid_from=valid_from,
            valid_to=valid_to,
            source_message_id=source_message_id or "",
            confidence=float(confidence),
            status=status or "",
            created_at=created_at or "",
            updated_at=updated_at or "",
            source_session_id=source_session_id,
            source_turn_id=source_turn_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnimaState:
    """machine-checkable state metadata emitted before the answer."""

    answer_mode: str
    used_lore_ids: tuple[str, ...]
    used_memory_ids: tuple[str, ...]
    memory_ops: tuple[MemoryOp, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnimaState":
        errors: list[str] = []
        answer_mode = _req_str(errors, value, "answer_mode", choices=ANSWER_MODES)
        used_lore_ids = _str_tuple(errors, value, "used_lore_ids", item_pattern=FACT_ID_RE)
        used_memory_ids = _str_tuple(errors, value, "used_memory_ids")
        ops_raw = value.get("memory_ops")
        ops: list[MemoryOp] = []
        if not isinstance(ops_raw, list):
            errors.append("memory_ops: required list")
        else:
            for index, item in enumerate(ops_raw):
                if not isinstance(item, Mapping):
                    errors.append(f"memory_ops[{index}]: must be an object")
                    continue
                try:
                    ops.append(MemoryOp.from_mapping(item))
                except PersonaContractError as exc:
                    errors.extend(f"memory_ops[{index}].{e}" for e in exc.errors)
        if errors:
            raise PersonaContractError(errors)
        return cls(
            answer_mode=answer_mode or "",
            used_lore_ids=used_lore_ids,
            used_memory_ids=used_memory_ids,
            memory_ops=tuple(ops),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_mode": self.answer_mode,
            "used_lore_ids": list(self.used_lore_ids),
            "used_memory_ids": list(self.used_memory_ids),
            "memory_ops": [op.to_dict() for op in self.memory_ops],
        }


def validate_profile(profile: Any) -> list[str]:
    """profile contract: required fields plus all-string discipline."""

    if not isinstance(profile, Mapping):
        return ["profile: must be a JSON object"]
    errors: list[str] = []
    for field_name in REQUIRED_PROFILE_FIELDS:
        value = profile.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"profile.{field_name}: required non-empty string")
    for key, value in profile.items():
        if key not in REQUIRED_PROFILE_FIELDS and (not isinstance(value, str) or not value.strip()):
            errors.append(f"profile.{key}: extra fields must be non-empty strings")
    return errors
