"""Reward-record schema and JSONL helpers.

Implements the frozen reward-record contract from
docs/REWARD_DATA_GUIDE_march7th.md. Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

FOCUS_LABELS: tuple[str, ...] = (
    "Knowledge",
    "Style",
    "Worldview",
    "Emotion",
    "Empathetic",
    "Engagement",
    "Human_Like",
    "Extension",
    "Memory",
    "Safety",
)

VALID_SPLITS: tuple[str, ...] = ("sft", "reward", "eval_heldout")

VALID_SOURCES: tuple[str, ...] = (
    "CharacterBench",
    "RoleBench",
    "model-synth",
    "HSR-canon",
    "HSR-synth",
)

VALID_ROLES: tuple[str, ...] = ("user", "assistant")

EXPECTED_BEHAVIORS: tuple[str, ...] = ("respond", "refuse")

REFUSAL_PROBE_TYPES: tuple[str, ...] = (
    "modern_knowledge",
    "cross_work",
    "future_event",
    "role_conflict",
)

SYNTHETIC_SOURCES: frozenset[str] = frozenset({"model-synth", "benchmark-synth"})

FOCUS_LABEL_SET: frozenset[str] = frozenset(FOCUS_LABELS)
VALID_SPLIT_SET: frozenset[str] = frozenset(VALID_SPLITS)
VALID_SOURCE_SET: frozenset[str] = frozenset(VALID_SOURCES)
VALID_ROLE_SET: frozenset[str] = frozenset(VALID_ROLES)
EXPECTED_BEHAVIOR_SET: frozenset[str] = frozenset(EXPECTED_BEHAVIORS)
REFUSAL_PROBE_TYPE_SET: frozenset[str] = frozenset(REFUSAL_PROBE_TYPES)


@dataclass(frozen=True)
class ConversationTurn:
    """One user/assistant turn in the policy-visible dialogue context."""

    role: str
    content: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConversationTurn":
        return cls(role=_coerce_str(value.get("role")), content=_coerce_str(value.get("content")))


@dataclass(frozen=True)
class SynthMeta:
    """Reproducibility metadata for synthesized gold or DPO fields."""

    generator: str
    prompt_id: str
    lore_sources: tuple[str, ...] = field(default_factory=tuple)
    rejected_strategy: str | None = None
    human_reviewed: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SynthMeta":
        lore_sources = value.get("lore_sources", ())
        if not isinstance(lore_sources, list | tuple):
            lore_sources = ()
        return cls(
            generator=_coerce_str(value.get("generator")),
            prompt_id=_coerce_str(value.get("prompt_id")),
            lore_sources=tuple(str(item) for item in lore_sources),
            rejected_strategy=_optional_str(value.get("rejected_strategy")),
            human_reviewed=value.get("human_reviewed", False),
        )


@dataclass(frozen=True)
class RewardRecord:
    """A single JSONL reward/DPO record."""

    id: str
    character: str
    source_work: str
    character_cluster: int
    profile: str
    conversations: tuple[ConversationTurn, ...]
    # gold_* 和 reference_answer 只给 reward 侧当打分目标，别拼进 policy prompt
    gold_focus: tuple[str, ...]
    gold_focus_attr: str
    reference_answer: str
    rejected_answer: str | None
    source: str
    split: str
    synth_meta: SynthMeta | None = None
    # focus_keys 是可选的 RAIDEN 式可验证 key 列表；默认 None 保证旧数据不受影响
    focus_keys: tuple[Any, ...] | None = None
    # 拒答扩展字段：默认 respond/空，老数据零改动照常通过校验
    expected_behavior: str = "respond"
    refusal_probe_type: str | None = None
    forbidden_terms: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RewardRecord":
        conversations = value.get("conversations", ())
        if not isinstance(conversations, list | tuple):
            conversations = ()

        gold_focus = value.get("gold_focus", ())
        if not isinstance(gold_focus, list | tuple):
            gold_focus = ()

        synth_meta = value.get("synth_meta")
        parsed_synth_meta = None
        if isinstance(synth_meta, Mapping):
            parsed_synth_meta = SynthMeta.from_mapping(synth_meta)

        focus_keys = value.get("focus_keys")
        parsed_focus_keys: tuple[Any, ...] | None
        if focus_keys is None:
            parsed_focus_keys = None
        elif isinstance(focus_keys, list | tuple):
            parsed_focus_keys = tuple(
                dict(item) if isinstance(item, Mapping) else item for item in focus_keys
            )
        else:
            # 非列表值包成单元素 tuple，让 validate_record 报出条目级错误而不是静默丢弃
            parsed_focus_keys = (focus_keys,)

        forbidden_terms = value.get("forbidden_terms", ())
        if not isinstance(forbidden_terms, list | tuple):
            forbidden_terms = ()

        return cls(
            id=_coerce_str(value.get("id")),
            character=_coerce_str(value.get("character")),
            source_work=_coerce_str(value.get("source_work")),
            character_cluster=_coerce_int(value.get("character_cluster")),
            profile=_coerce_str(value.get("profile")),
            conversations=tuple(ConversationTurn.from_mapping(turn) for turn in conversations),
            gold_focus=tuple(str(label) for label in gold_focus),
            gold_focus_attr=_coerce_str(value.get("gold_focus_attr")),
            reference_answer=_coerce_str(value.get("reference_answer")),
            rejected_answer=_optional_str(value.get("rejected_answer")),
            source=_coerce_str(value.get("source")),
            split=_coerce_str(value.get("split")),
            synth_meta=parsed_synth_meta,
            focus_keys=parsed_focus_keys,
            expected_behavior=_coerce_str(value.get("expected_behavior")) or "respond",
            refusal_probe_type=_optional_str(value.get("refusal_probe_type")),
            forbidden_terms=tuple(str(term) for term in forbidden_terms),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["conversations"] = [asdict(turn) for turn in self.conversations]
        data["gold_focus"] = list(self.gold_focus)
        if self.synth_meta is not None:
            data["synth_meta"] = asdict(self.synth_meta)
            data["synth_meta"]["lore_sources"] = list(self.synth_meta.lore_sources)
        else:
            data["synth_meta"] = None
        if self.focus_keys is None:
            data["focus_keys"] = None
        else:
            data["focus_keys"] = [
                dict(item) if isinstance(item, Mapping) else item for item in self.focus_keys
            ]
        data["forbidden_terms"] = list(self.forbidden_terms)
        return data


class SchemaValidationError(ValueError):
    """Raised when a reward record violates the frozen schema."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def validate_record(record: RewardRecord | Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for one reward record.

    Schema checks only; dataset-specific leakage and deduplication checks run separately.
    """

    if not isinstance(record, RewardRecord):
        record = RewardRecord.from_mapping(record)

    errors: list[str] = []

    _require_text(errors, record.id, "id")
    _require_text(errors, record.character, "character")
    _require_text(errors, record.source_work, "source_work")
    _require_text(errors, record.profile, "profile")
    _require_text(errors, record.gold_focus_attr, "gold_focus_attr")
    _require_text(errors, record.reference_answer, "reference_answer")

    if record.character_cluster < -1:
        errors.append("character_cluster must be -1 or a non-negative integer")

    if not record.conversations:
        errors.append("conversations must contain at least one turn")
    for index, turn in enumerate(record.conversations):
        if turn.role not in VALID_ROLE_SET:
            errors.append(f"conversations[{index}].role must be one of {VALID_ROLES}")
        if not turn.content.strip():
            errors.append(f"conversations[{index}].content must be non-empty")

    if not record.gold_focus:
        errors.append("gold_focus must contain at least one label")
    illegal_labels = [label for label in record.gold_focus if label not in FOCUS_LABEL_SET]
    if illegal_labels:
        errors.append(f"gold_focus contains illegal labels: {illegal_labels}")
    if len(set(record.gold_focus)) != len(record.gold_focus):
        errors.append("gold_focus must not contain duplicate labels")

    if record.source not in VALID_SOURCE_SET:
        errors.append(f"source must be one of {VALID_SOURCES}")
    if record.split not in VALID_SPLIT_SET:
        errors.append(f"split must be one of {VALID_SPLITS}")

    if record.source in SYNTHETIC_SOURCES and record.synth_meta is None:
        errors.append("synth_meta is required for synthetic sources")

    if record.synth_meta is not None:
        _require_text(errors, record.synth_meta.generator, "synth_meta.generator")
        _require_text(errors, record.synth_meta.prompt_id, "synth_meta.prompt_id")
        if not isinstance(record.synth_meta.human_reviewed, bool):
            errors.append("synth_meta.human_reviewed must be a boolean")

    if record.focus_keys is not None:
        errors.extend(validate_focus_keys(record.focus_keys))

    errors.extend(_validate_refusal_fields(record))

    if record.rejected_answer is not None:
        if not record.rejected_answer.strip():
            errors.append("rejected_answer must be non-empty when present")
        if record.rejected_answer == record.reference_answer:
            errors.append("rejected_answer must differ from reference_answer")
        if record.synth_meta is None:
            errors.append("synth_meta is required when rejected_answer is present")
        elif not record.synth_meta.rejected_strategy:
            errors.append(
                "synth_meta.rejected_strategy is required when rejected_answer is present"
            )

    return errors


def _validate_refusal_fields(record: RewardRecord) -> list[str]:
    """Return errors for the optional refusal-probe fields."""

    errors: list[str] = []

    if record.expected_behavior not in EXPECTED_BEHAVIOR_SET:
        errors.append(f"expected_behavior must be one of {EXPECTED_BEHAVIORS}")

    if (
        record.refusal_probe_type is not None
        and record.refusal_probe_type not in REFUSAL_PROBE_TYPE_SET
    ):
        errors.append(f"refusal_probe_type must be one of {REFUSAL_PROBE_TYPES}")

    for index, term in enumerate(record.forbidden_terms):
        if not isinstance(term, str) or not term.strip():
            errors.append(f"forbidden_terms[{index}] must be a non-empty string")

    if record.expected_behavior != "refuse":
        if record.refusal_probe_type is not None:
            errors.append("refusal_probe_type is only allowed when expected_behavior is refuse")
        return errors

    if not record.forbidden_terms:
        errors.append("forbidden_terms must be non-empty when expected_behavior is refuse")
    if record.refusal_probe_type is None:
        errors.append("refusal_probe_type is required when expected_behavior is refuse")
    if "Safety" not in record.gold_focus:
        errors.append("gold_focus must include Safety when expected_behavior is refuse")

    # echo 豁免前置到数据侧：用户已说出口的词不能证明角色“知道”，
    # 训练期奖励因此可以退化成纯子串扫描
    last_user_turn = next(
        (turn.content for turn in reversed(record.conversations) if turn.role == "user"),
        "",
    )
    terms = [term for term in record.forbidden_terms if isinstance(term, str) and term.strip()]
    echoed = [term for term in terms if term in last_user_turn]
    if echoed:
        errors.append(f"forbidden_terms must not appear in the last user turn: {echoed}")

    leaked = [term for term in terms if term in record.reference_answer]
    if leaked:
        errors.append(f"reference_answer must not contain forbidden_terms: {leaked}")

    return errors


def validate_focus_keys(focus_keys: Iterable[Any]) -> list[str]:
    """Return errors for an optional focus_keys list (mined verifiable keys)."""

    errors: list[str] = []
    for index, item in enumerate(focus_keys):
        if not isinstance(item, Mapping):
            errors.append(f"focus_keys[{index}] must be an object")
            continue
        key = item.get("key")
        if not isinstance(key, str) or not key.strip():
            errors.append(f"focus_keys[{index}].key must be a non-empty string")
        aliases = item.get("aliases")
        if aliases is not None:
            if not isinstance(aliases, list | tuple) or any(
                not isinstance(alias, str) or not alias.strip() for alias in aliases
            ):
                errors.append(f"focus_keys[{index}].aliases must be a list of non-empty strings")
        focus = item.get("focus")
        if focus is not None and focus not in FOCUS_LABEL_SET:
            errors.append(f"focus_keys[{index}].focus must be one of {FOCUS_LABELS}")
    return errors


def require_valid_record(record: RewardRecord | Mapping[str, Any]) -> RewardRecord:
    """Return a RewardRecord or raise SchemaValidationError."""

    parsed = record if isinstance(record, RewardRecord) else RewardRecord.from_mapping(record)
    errors = validate_record(parsed)
    if errors:
        raise SchemaValidationError(errors)
    return parsed


def load_jsonl(path: str | Path, *, validate: bool = True) -> list[RewardRecord]:
    """Load reward records from JSONL."""

    records: list[RewardRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    [f"line {line_number}: invalid JSON: {exc.msg}"]
                ) from exc
            if not isinstance(payload, Mapping):
                raise SchemaValidationError([f"line {line_number}: record must be a JSON object"])
            record = RewardRecord.from_mapping(payload)
            if validate:
                errors = validate_record(record)
                if errors:
                    raise SchemaValidationError(
                        [f"line {line_number}: {error}" for error in errors]
                    )
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[RewardRecord | Mapping[str, Any]]) -> None:
    """Write reward records to JSONL after schema validation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            parsed = require_valid_record(record)
            handle.write(json.dumps(parsed.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _require_text(errors: list[str], value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name} must be a non-empty string")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -2
