"""Gates on the shipped public Persona Packs.

These run against the real persona_packs/public/ data so any edit that breaks
schema, hash, scale, review-status or IP rules fails CI immediately.
"""

from __future__ import annotations

import pytest

from anima.persona.contracts.pack import (
    GENERALIZATION_MINIMUMS,
    PUBLIC_DOMAIN_MAIN_MINIMUMS,
    PackValidationError,
    check_pack_scale,
    load_pack,
)
from tests.support.environment.paths import PROJECT_ROOT

PACKS_ROOT = PROJECT_ROOT / "persona_packs" / "public"
MAIN_PERSONA = "sherlock_holmes"
GENERALIZATION_PERSONAS: tuple[str, ...] = ()
ALL_PERSONAS = (MAIN_PERSONA, *GENERALIZATION_PERSONAS)

# IP gate: third-party IP vocabulary must never appear in public packs.
THIRD_PARTY_IP_TERMS = (
    "帕姆",
    "星穹",
    "开拓者",
    "崩坏",
    "三月七",
    "米哈游",
    "神探夏洛克",
    "BBC",
    "Netflix",
)
ACTIONABLE_HARM_TERMS = ("三格令", "取三格令", "配比步骤如下")


def test_public_pack_directory_is_sherlock_only():
    personas = tuple(sorted(path.name for path in PACKS_ROOT.iterdir() if path.is_dir()))
    assert personas == ALL_PERSONAS


@pytest.mark.parametrize("persona_id", ALL_PERSONAS)
def test_pack_loads_and_hash_matches(persona_id):
    pack = load_pack(PACKS_ROOT / persona_id)
    assert pack.manifest.persona_id == persona_id
    assert pack.manifest.public is True


@pytest.mark.parametrize("persona_id", ALL_PERSONAS)
def test_drafts_are_refused_for_formal_use(persona_id):
    """Draft packs stay unusable until an accepted provenance-bearing review."""

    with pytest.raises(PackValidationError, match="formal_reviewed"):
        load_pack(PACKS_ROOT / persona_id, formal=True)


def test_main_persona_meets_scale_gate():
    pack = load_pack(PACKS_ROOT / MAIN_PERSONA)
    assert check_pack_scale(pack, minimums=PUBLIC_DOMAIN_MAIN_MINIMUMS) == []
    assert all(fact.aliases for fact in pack.lore)
    assert all(fact.source_ref.startswith("doyle:") for fact in pack.lore)
    assert all(event.source_ref.startswith("doyle:") for event in pack.timeline)
    assert all(relationship.source_ref.startswith("doyle:") for relationship in pack.relationships)


def test_sherlock_lore_has_explicit_non_gameable_answer_contracts():
    pack = load_pack(PACKS_ROOT / MAIN_PERSONA)
    for fact in pack.lore:
        if fact.known_by_persona:
            assert fact.persona_response
            assert fact.persona_response != fact.object
            assert len(fact.answer_slots) >= 2
            assert not fact.boundary_forbidden_claims
            prompt_alias = min(fact.aliases, key=len)
            assert any(
                all(
                    "".join(candidate.split()) not in "".join(prompt_alias.split())
                    for candidate in group
                )
                for group in fact.answer_slots
            ), f"{fact.fact_id}: every answer slot can be satisfied by echoing the prompt alias"
        else:
            assert fact.persona_response is None
            assert not fact.answer_slots
            assert fact.boundary_forbidden_claims
            assert all(claim not in fact.aliases for claim in fact.boundary_forbidden_claims)


def test_sherlock_style_pairs_declare_matching_semantics():
    pack = load_pack(PACKS_ROOT / MAIN_PERSONA)
    positives = {example.prompt: example for example in pack.style_positive()}
    negatives = {example.prompt: example for example in pack.style_negative()}
    assert set(negatives) <= set(positives)
    for prompt, negative in negatives.items():
        positive = positives[prompt]
        assert negative.semantic_type == positive.semantic_type
        assert negative.gold_lore_ids == positive.gold_lore_ids


def test_generalization_personas_meet_scale_gate():
    """Validate every declared auxiliary pack without skipping Sherlock-only releases."""

    for persona_id in GENERALIZATION_PERSONAS:
        pack = load_pack(PACKS_ROOT / persona_id)
        assert check_pack_scale(pack, minimums=GENERALIZATION_MINIMUMS) == []


def test_main_persona_is_free_of_third_party_ip_terms():
    pack_dir = PACKS_ROOT / MAIN_PERSONA
    for path in sorted(pack_dir.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        hits = [term for term in THIRD_PARTY_IP_TERMS if term in text]
        assert not hits, f"{path.name} contains third-party IP terms: {hits}"


def test_public_pack_negatives_do_not_publish_actionable_harm_details():
    pack_dir = PACKS_ROOT / MAIN_PERSONA
    text = (pack_dir / "style_examples.jsonl").read_text(encoding="utf-8")
    hits = [term for term in ACTIONABLE_HARM_TERMS if term in text]
    assert not hits, f"style_examples.jsonl contains actionable harm details: {hits}"


def test_knowledge_cutoffs_are_declared():
    for persona_id in ALL_PERSONAS:
        pack = load_pack(PACKS_ROOT / persona_id)
        assert pack.manifest.knowledge_cutoff, f"{persona_id} must declare a knowledge cutoff"
