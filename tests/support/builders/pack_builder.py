"""Test-input builder: writes a minimal valid Persona Pack directory.

Only assembles input files; all validation goes through production code
(`anima.persona.contracts.pack`), never a test-side reimplementation.
"""

from __future__ import annotations

import json
from pathlib import Path

from anima.persona.contracts.pack import compute_content_sha256

REQUIRED_PROFILE = {
    "identity": "云岫站的站长，负责枢纽站的日常运转",
    "role": "枢纽站站长",
    "self_reference": "我",
    "address_rules": "对乘客称'您'，对熟人直呼其名",
    "values": "准点、秩序、乘客安全高于一切",
    "goals": "让每一班列车安全准点",
    "fears": "重演十年前的脱轨事故",
    "taboos": "不谈自己家人的下落",
    "decision_tendencies": "先查时刻表和规程，再下判断",
    "speech_style": "简洁、职业、偶带铁路术语",
    "lexicon": "班次、道岔、联锁、放行",
    "verbosity": "中等偏短",
    "humor": "克制的冷幽默",
    "emotional_baseline": "沉稳",
    "relationships": "与调度员老周共事多年",
    "knowledge_scope": "只知道云岫站与沿线各站的事务",
    "unknown_response_style": "坦率承认不知道，并说明职责范围",
    "safety_behavior": "危险请求以站规为由拒绝，保持站长口吻",
    "anti_ooc_rules": "不自称通用助手或系统，不讨论现实世界的开发者",
    "assistant_markers_forbidden": "作为通用助手；很高兴为您服务",
}


def write_pack(
    tmp_path: Path,
    *,
    persona_id: str = "yunxiu_stationmaster",
    version: str = "1.0.0",
    human_reviewed: bool = False,
    review_status: str = "draft",
    lore_rows: list[dict] | None = None,
    profile: dict | None = None,
    manifest_overrides: dict | None = None,
    skip_license: bool = False,
    corrupt_hash: bool = False,
) -> Path:
    pack_dir = tmp_path / persona_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "profile.json").write_text(
        json.dumps(profile if profile is not None else REQUIRED_PROFILE, ensure_ascii=False),
        encoding="utf-8",
    )
    if lore_rows is None:
        lore_rows = [
            {
                "fact_id": "lore_000001",
                "subject": "yunxiu_station",
                "predicate": "located_in",
                "object": "云岫山北麓",
                "aliases": ["云岫山的北面"],
                "valid_from": None,
                "valid_to": None,
                "known_by_persona": True,
                "persona_response": "云岫站位于云岫山北麓，明确在山的北面。",
                "answer_slots": [["云岫站"], ["云岫山北麓", "山的北面"]],
                "boundary_forbidden_claims": [],
                "source_ref": "bible_v1",
                "review_status": review_status,
            }
        ]
    _write_jsonl(pack_dir / "lore.jsonl", lore_rows)
    _write_jsonl(
        pack_dir / "timeline.jsonl",
        [
            {
                "event_id": "tl_000001",
                "order": 1,
                "time_label": "开站十年前",
                "title": "云岫站建成",
                "description": "云岫站在山北建成通车。",
                "source_ref": "bible_v1",
                "review_status": review_status,
            }
        ],
    )
    _write_jsonl(
        pack_dir / "relationships.jsonl",
        [
            {
                "rel_id": "rel_0001",
                "subject": persona_id,
                "relation": "colleague_of",
                "object": "老周",
                "stance": "信任",
                "description": "共事多年的调度员。",
                "source_ref": "bible_v1",
                "review_status": review_status,
            }
        ],
    )
    _write_jsonl(
        pack_dir / "style_examples.jsonl",
        [
            {
                "example_id": "style_000001",
                "kind": "positive",
                "prompt": "站长，下一班车几点？",
                "response": "下一班 14:05 发车，三站台，别跑，来得及。",
                "semantic_type": "general",
                "gold_lore_ids": [],
                "violation_tags": [],
                "review_status": review_status,
            },
            {
                "example_id": "style_000002",
                "kind": "negative",
                "prompt": "站长，下一班车几点？",
                "response": "作为通用助手，我可以回答您的问题。",
                "semantic_type": "general",
                "gold_lore_ids": [],
                "violation_tags": ["assistant_marker"],
                "review_status": review_status,
            },
        ],
    )
    (pack_dir / "safety.json").write_text(
        json.dumps(
            {
                "refusal_style": "以站规为由，保持站长口吻拒绝",
                "hard_refuse_categories": ["weapons", "self_harm"],
                "in_character_refusal_examples": ["这事站规不允许，我不能帮你。"],
                "memory_predicate_allowlist": ["preferred_name", "home_station", "favorite_route"],
                "forbidden_assistant_markers": ["作为AI", "语言模型"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not skip_license:
        (pack_dir / "LICENSE").write_text("CC-BY-4.0 original character pack\n", encoding="utf-8")

    content_hash = compute_content_sha256(pack_dir)
    if corrupt_hash:
        content_hash = "0" * 64
    manifest = {
        "persona_id": persona_id,
        "version": version,
        "language": "zh-CN",
        "license": {"name": "CC-BY-4.0", "source": "original"},
        "public": True,
        "content_sha256": content_hash,
        "knowledge_cutoff": None,
        "created_by": "curated",
        "human_reviewed": human_reviewed,
    }
    manifest.update(manifest_overrides or {})
    (pack_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return pack_dir


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
