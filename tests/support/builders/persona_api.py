"""Shared in-process Persona API assembly for service and evaluation tests."""

from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from anima.persona.contracts.pack import load_pack
from anima.serve.api.persona import create_app
from anima.serve.core.metrics import MetricsRegistry
from anima.serve.core.runtime import AppServices, ModelInfo, RuntimeConfig, build_lore_indexes
from anima.serve.inference.embedding import HashEmbedder
from tests.support.doubles.fake_repository import FakeRepository
from tests.support.doubles.scripted_model import ScriptedModel, render_output

API_KEY = "test-key-abc"


def build_persona_api_test_app(pack_dir, responder=None, *, eval_token=None, config=None):
    pack = load_pack(pack_dir)
    packs = {pack.manifest.persona_id: pack}
    embedder = HashEmbedder(dim=128)
    repository = FakeRepository()
    repository.create_user("user-a", hashlib.sha256(API_KEY.encode()).hexdigest())
    repository.create_user("user-b", hashlib.sha256(b"other-key").hexdigest())
    model = ScriptedModel(
        responder or (lambda _system, _messages: render_output("general", "你好，这里是站长室。"))
    )
    services = AppServices(
        packs=packs,
        lore_indexes=build_lore_indexes(packs, embedder),
        repo=repository,
        model=model,
        embedder=embedder,
        metrics=MetricsRegistry(),
        model_info=ModelInfo(base_model="qwen", adapter_id=None, adapter_sha256=None),
        config=config or RuntimeConfig(),
        test_mode=True,
        eval_token_hash=hashlib.sha256(eval_token.encode()).hexdigest() if eval_token else None,
    )
    return TestClient(create_app(services)), pack, repository
