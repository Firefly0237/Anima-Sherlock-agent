"""Load and validate Persona Pack directories.

A pack directory holds six data files plus manifest.json and LICENSE.
Validation aggregates every error instead of stopping at the first one
and `formal=True` refuses any pack that lacks a provenance-bearing review.
Draft content cannot enter a formal training or evaluation path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from anima.persona.contracts.schemas import (
    LoreFact,
    PackManifest,
    PersonaContractError,
    Relationship,
    SafetyPolicy,
    StyleExample,
    TimelineEvent,
    is_formal_pass,
    validate_profile,
)
from anima.utils.jsonl import iter_jsonl

PACK_DATA_FILES: tuple[str, ...] = (
    "profile.json",
    "lore.jsonl",
    "timeline.jsonl",
    "relationships.jsonl",
    "style_examples.jsonl",
    "safety.json",
)

# minimum gates for the original large-scope main persona. Public-domain
# canon personas use a separate main gate because their reliable source surface
# is narrower than a project-owned bible, but still larger than a canary pack.
MAIN_PERSONA_MINIMUMS: dict[str, int] = {
    "profile_fields": 25,
    "lore": 120,
    "timeline": 25,
    "relationships": 10,
    "style_positive": 80,
    "style_negative": 80,
}

PUBLIC_DOMAIN_MAIN_MINIMUMS: dict[str, int] = {
    "profile_fields": 25,
    "lore": 55,
    "timeline": 14,
    "relationships": 10,
    "style_positive": 22,
    "style_negative": 22,
}

GENERALIZATION_MINIMUMS: dict[str, int] = {
    "profile_fields": 20,
    "lore": 45,
    "timeline": 12,
    "relationships": 8,
    "style_positive": 22,
    "style_negative": 22,
}

_ROW_FILES: tuple[tuple[str, type, str], ...] = (
    ("lore.jsonl", LoreFact, "fact_id"),
    ("timeline.jsonl", TimelineEvent, "event_id"),
    ("relationships.jsonl", Relationship, "rel_id"),
    ("style_examples.jsonl", StyleExample, "example_id"),
)


class PackValidationError(ValueError):
    """Aggregated pack validation failures."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def compute_content_sha256(pack_dir: Path) -> str:
    """Deterministic hash over the declared data files (manifest excluded)."""

    pack_dir = Path(pack_dir)
    missing = [name for name in PACK_DATA_FILES if not (pack_dir / name).is_file()]
    if missing:
        raise PackValidationError([f"{name}: data file missing" for name in missing])
    digest = hashlib.sha256()
    for name in PACK_DATA_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((pack_dir / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedPack:
    root: Path
    manifest: PackManifest
    profile: dict[str, Any]
    lore: tuple[LoreFact, ...]
    timeline: tuple[TimelineEvent, ...]
    relationships: tuple[Relationship, ...]
    style_examples: tuple[StyleExample, ...]
    safety: SafetyPolicy

    def unreviewed_row_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for rows, id_attr in (
            (self.lore, "fact_id"),
            (self.timeline, "event_id"),
            (self.relationships, "rel_id"),
            (self.style_examples, "example_id"),
        ):
            ids.extend(
                getattr(row, id_attr) for row in rows if not is_formal_pass(row.review_status)
            )
        return tuple(ids)

    @property
    def is_fully_human_reviewed(self) -> bool:
        rows = (*self.lore, *self.timeline, *self.relationships, *self.style_examples)
        return self.manifest.human_reviewed and all(
            row.review_status == "human_pass" for row in rows
        )

    @property
    def is_fully_formally_reviewed(self) -> bool:
        return self.manifest.formal_reviewed and not self.unreviewed_row_ids()

    def lore_by_id(self) -> dict[str, LoreFact]:
        return {fact.fact_id: fact for fact in self.lore}

    def style_positive(self) -> tuple[StyleExample, ...]:
        return tuple(example for example in self.style_examples if example.kind == "positive")

    def style_negative(self) -> tuple[StyleExample, ...]:
        return tuple(example for example in self.style_examples if example.kind == "negative")


def load_pack(pack_dir: Path, *, formal: bool = False) -> LoadedPack:
    """Validate and load a pack; raise PackValidationError with all errors."""

    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PackValidationError(["manifest.json: file missing"])
    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PackValidationError([f"manifest.json: invalid JSON: {exc}"]) from exc

    errors: list[str] = []
    manifest: PackManifest | None = None
    try:
        manifest = PackManifest.from_mapping(manifest_raw)
    except PersonaContractError as exc:
        errors.extend(f"manifest.json: {e}" for e in exc.errors)

    license_path = pack_dir / "LICENSE"
    if not license_path.is_file() or not license_path.read_text(encoding="utf-8").strip():
        errors.append("LICENSE: file missing or empty")

    missing = [name for name in PACK_DATA_FILES if not (pack_dir / name).is_file()]
    errors.extend(f"{name}: data file missing" for name in missing)

    if not missing and manifest is not None:
        actual = compute_content_sha256(pack_dir)
        if actual != manifest.content_sha256:
            errors.append(
                f"manifest.json: content_sha256 mismatch (manifest {manifest.content_sha256}, actual {actual})"
            )

    profile: dict[str, Any] = {}
    if "profile.json" not in missing:
        profile_raw = _read_json(errors, pack_dir / "profile.json")
        if profile_raw is not None:
            errors.extend(validate_profile(profile_raw))
            if isinstance(profile_raw, dict):
                profile = profile_raw

    rows_by_file: dict[str, tuple[Any, ...]] = {}
    for file_name, row_type, id_attr in _ROW_FILES:
        rows_by_file[file_name] = _load_rows(
            errors, pack_dir, file_name, row_type, id_attr, missing
        )

    lore_by_id = {fact.fact_id: fact for fact in rows_by_file["lore.jsonl"]}
    positives_by_prompt: dict[str, StyleExample] = {}
    negatives_by_prompt: dict[str, StyleExample] = {}
    for example in rows_by_file["style_examples.jsonl"]:
        target = positives_by_prompt if example.kind == "positive" else negatives_by_prompt
        if example.prompt in target:
            errors.append(
                f"style_examples.jsonl: duplicate {example.kind} prompt for "
                f"{target[example.prompt].example_id} and {example.example_id}"
            )
        target[example.prompt] = example
        for fact_id in example.gold_lore_ids:
            fact = lore_by_id.get(fact_id)
            if fact is None:
                errors.append(
                    f"style_examples.jsonl: {example.example_id}: gold lore id {fact_id} not in pack"
                )
            elif not fact.known_by_persona:
                errors.append(
                    f"style_examples.jsonl: {example.example_id}: gold lore id {fact_id} is not persona-known"
                )
            elif example.kind == "positive":
                compact_response = "".join(example.response.casefold().split())
                for index, group in enumerate(fact.answer_slots):
                    if not any(
                        "".join(alias.casefold().split()) in compact_response for alias in group
                    ):
                        errors.append(
                            f"style_examples.jsonl: {example.example_id}: response misses "
                            f"{fact_id} answer_slots[{index}]"
                        )
    for prompt, negative in negatives_by_prompt.items():
        positive = positives_by_prompt.get(prompt)
        if positive is None:
            errors.append(
                f"style_examples.jsonl: {negative.example_id}: negative prompt has no positive pair"
            )
            continue
        if (
            negative.semantic_type != positive.semantic_type
            or negative.gold_lore_ids != positive.gold_lore_ids
        ):
            errors.append(
                f"style_examples.jsonl: {negative.example_id}: pair semantics do not match "
                f"{positive.example_id}"
            )

    orders = [event.order for event in rows_by_file["timeline.jsonl"]]
    if len(set(orders)) != len(orders):
        errors.append("timeline.jsonl: duplicate order values")

    safety: SafetyPolicy | None = None
    if "safety.json" not in missing:
        safety_raw = _read_json(errors, pack_dir / "safety.json")
        if safety_raw is not None:
            try:
                safety = SafetyPolicy.from_mapping(safety_raw)
            except PersonaContractError as exc:
                errors.extend(f"safety.json: {e}" for e in exc.errors)

    if formal:
        if manifest is not None and not manifest.formal_reviewed:
            errors.append("manifest.json: formal use requires formal_reviewed=true")
        for file_name, _row_type, id_attr in _ROW_FILES:
            for row in rows_by_file[file_name]:
                if not is_formal_pass(row.review_status):
                    errors.append(
                        f"{file_name}: {getattr(row, id_attr)}: review_status must be a formal pass for formal use"
                        f" (got {row.review_status})"
                    )

    if errors:
        raise PackValidationError(errors)
    assert manifest is not None and safety is not None  # errors above guarantee this
    return LoadedPack(
        root=pack_dir,
        manifest=manifest,
        profile=profile,
        lore=rows_by_file["lore.jsonl"],
        timeline=rows_by_file["timeline.jsonl"],
        relationships=rows_by_file["relationships.jsonl"],
        style_examples=rows_by_file["style_examples.jsonl"],
        safety=safety,
    )


def check_pack_scale(
    pack: LoadedPack, *, minimums: Mapping[str, int] = MAIN_PERSONA_MINIMUMS
) -> list[str]:
    """scale gate; returns one error per shortfall."""

    counts = {
        "profile_fields": len(pack.profile),
        "lore": len(pack.lore),
        "timeline": len(pack.timeline),
        "relationships": len(pack.relationships),
        "style_positive": len(pack.style_positive()),
        "style_negative": len(pack.style_negative()),
    }
    errors: list[str] = []
    for key, minimum in minimums.items():
        if key not in counts:
            errors.append(f"{key}: unknown scale key")
        elif counts[key] < minimum:
            errors.append(f"{key}: {counts[key]} < required minimum {minimum}")
    return errors


def _read_json(errors: list[str], path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}: invalid JSON: {exc}")
        return None


def _load_rows(
    errors: list[str],
    pack_dir: Path,
    file_name: str,
    row_type: type,
    id_attr: str,
    missing: list[str],
) -> tuple[Any, ...]:
    if file_name in missing:
        return ()
    try:
        parsed = list(iter_jsonl(pack_dir / file_name))
    except ValueError as exc:
        errors.append(str(exc))
        return ()
    rows: list[Any] = []
    seen_ids: set[str] = set()
    for line_no, record in parsed:
        try:
            row = row_type.from_mapping(record)
        except PersonaContractError as exc:
            errors.extend(f"{file_name}:{line_no}: {e}" for e in exc.errors)
            continue
        row_id = getattr(row, id_attr)
        if row_id in seen_ids:
            errors.append(f"{file_name}:{line_no}: duplicate {id_attr} {row_id}")
        seen_ids.add(row_id)
        rows.append(row)
    return tuple(rows)
