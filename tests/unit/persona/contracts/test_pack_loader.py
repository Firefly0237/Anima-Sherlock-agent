"""Tests for anima.persona.contracts.pack."""

from __future__ import annotations

import pytest

from anima.persona.contracts.pack import (
    MAIN_PERSONA_MINIMUMS,
    PackValidationError,
    check_pack_scale,
    compute_content_sha256,
    load_pack,
)
from tests.support.builders.pack_builder import REQUIRED_PROFILE, write_pack


def test_load_valid_draft_pack(pack_dir):
    pack = load_pack(pack_dir)
    assert pack.manifest.persona_id == "yunxiu_stationmaster"
    assert pack.lore[0].fact_id == "lore_000001"
    assert pack.profile["role"] == "枢纽站站长"
    assert not pack.is_fully_human_reviewed


def test_content_hash_changes_with_content(pack_dir):
    before = compute_content_sha256(pack_dir)
    lore_path = pack_dir / "lore.jsonl"
    lore_path.write_text(
        lore_path.read_text(encoding="utf-8").replace("云岫山北麓", "云岫山南麓"), encoding="utf-8"
    )
    assert compute_content_sha256(pack_dir) != before


def test_hash_mismatch_rejected(tmp_path):
    pack_dir = write_pack(tmp_path, corrupt_hash=True)
    with pytest.raises(PackValidationError, match="content_sha256"):
        load_pack(pack_dir)


def test_missing_license_rejected(tmp_path):
    pack_dir = write_pack(tmp_path, skip_license=True)
    with pytest.raises(PackValidationError, match="LICENSE"):
        load_pack(pack_dir)


def test_bad_semver_rejected(tmp_path):
    pack_dir = write_pack(tmp_path, version="v1")
    with pytest.raises(PackValidationError, match="version"):
        load_pack(pack_dir)


def test_duplicate_fact_ids_rejected(tmp_path):
    row = {
        "fact_id": "lore_000001",
        "subject": "s",
        "predicate": "p",
        "object": "o",
        "aliases": [],
        "valid_from": None,
        "valid_to": None,
        "known_by_persona": True,
        "persona_response": "我确认对象是 o，这是一条可核查记录。",
        "answer_slots": [["对象是 o"], ["可核查记录"]],
        "boundary_forbidden_claims": [],
        "source_ref": "bible_v1",
        "review_status": "draft",
    }
    pack_dir = write_pack(tmp_path, lore_rows=[row, dict(row)])
    with pytest.raises(PackValidationError, match="duplicate"):
        load_pack(pack_dir)


def test_missing_profile_field_rejected(tmp_path):
    broken = dict(REQUIRED_PROFILE)
    broken.pop("knowledge_scope")
    bad_dir = write_pack(tmp_path / "broken", profile=broken)
    with pytest.raises(PackValidationError, match="knowledge_scope"):
        load_pack(bad_dir)


def test_formal_requires_provenance_reviewed_manifest_and_rows(tmp_path):
    draft_dir = write_pack(tmp_path / "draft", human_reviewed=False, review_status="draft")
    with pytest.raises(PackValidationError, match="formal_reviewed"):
        load_pack(draft_dir, formal=True)

    # manifest says reviewed but rows still draft -> still rejected
    lying_dir = write_pack(tmp_path / "lying", human_reviewed=True, review_status="draft")
    with pytest.raises(PackValidationError, match="review_status"):
        load_pack(lying_dir, formal=True)

    ok_dir = write_pack(tmp_path / "ok", human_reviewed=True, review_status="human_pass")
    pack = load_pack(ok_dir, formal=True)
    assert pack.is_fully_human_reviewed
    assert pack.is_fully_formally_reviewed


def test_errors_are_aggregated_not_first_only(tmp_path):
    pack_dir = write_pack(tmp_path, version="bad", skip_license=True)
    with pytest.raises(PackValidationError) as excinfo:
        load_pack(pack_dir)
    text = str(excinfo.value)
    assert "version" in text and "LICENSE" in text


def test_scale_gate_reports_all_shortfalls(pack_dir):
    pack = load_pack(pack_dir)
    errors = check_pack_scale(pack, minimums=MAIN_PERSONA_MINIMUMS)
    joined = "\n".join(errors)
    for key in (
        "lore",
        "timeline",
        "relationships",
        "style_positive",
        "style_negative",
        "profile_fields",
    ):
        assert key in joined, f"missing shortfall report for {key}"


def test_scale_gate_passes_small_minimums(pack_dir):
    pack = load_pack(pack_dir)
    tiny = {
        "profile_fields": 20,
        "lore": 1,
        "timeline": 1,
        "relationships": 1,
        "style_positive": 1,
        "style_negative": 1,
    }
    assert check_pack_scale(pack, minimums=tiny) == []
