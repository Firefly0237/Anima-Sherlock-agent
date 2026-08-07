"""Config wiring for anima.serve.api.app (no DB/model connection needed)."""

from __future__ import annotations

import pytest

from anima.persona.contracts.provenance import BASE_MODEL_REVISION, EMBEDDING_MODEL_REVISION
from anima.serve.api.app import load_runtime_config
from anima.serve.core.runtime import RuntimeConfig
from tests.support.environment.paths import PROJECT_ROOT

CONFIG_EXAMPLE = PROJECT_ROOT / "configs" / "runtime.example.yaml"


def test_example_config_parses_with_expected_keys():
    config = load_runtime_config(CONFIG_EXAMPLE)
    assert config["database"]["dsn"].startswith("postgresql://")
    assert config["embedding"]["backend"] == "bge"
    assert config["embedding"]["model_id"] == "BAAI/bge-small-zh-v1.5"
    assert config["embedding"]["revision"] == EMBEDDING_MODEL_REVISION
    assert config["model"]["base_model"]
    assert config["model"]["base_model_revision"] == BASE_MODEL_REVISION
    # thresholds must default to null (calibrated on the dev set, never a constant)
    assert config["retrieval"]["lore_min_score"] is None
    assert config["retrieval"]["memory_min_score"] is None
    parsed = RuntimeConfig.from_mapping(config, formal=False)
    assert parsed.lore_retrieval_enabled is True
    assert parsed.memory_retrieval_enabled is True
    assert parsed.memory_write_enabled is True


def test_formal_runtime_rejects_legacy_combined_memory_control():
    legacy = {
        "retrieval": {"enabled": False},
        "runtime": {"memory_enabled": False},
    }
    with pytest.raises(ValueError, match="explicit split Memory controls"):
        RuntimeConfig.from_mapping(legacy, formal=True)

    pilot = RuntimeConfig.from_mapping(legacy, formal=False)
    assert pilot.lore_retrieval_enabled is False
    assert pilot.memory_retrieval_enabled is False
    assert pilot.memory_write_enabled is False


def test_importing_app_module_has_no_side_effects():
    # importing must not connect to a DB or load a model (factory pattern)
    import anima.serve.api.app as app_module

    assert hasattr(app_module, "create_default_app")
    assert not hasattr(app_module, "app")  # no eager module-level app object


def test_formal_code_commit_requires_hex(monkeypatch, pack_dir):
    from anima.serve.api import app as app_module

    monkeypatch.setenv("ANIMA_CODE_COMMIT", "UNSET")
    with pytest.raises(ValueError, match="hexadecimal"):
        app_module.build_services(
            {
                "embedding": {"backend": "bge", "model_id": "x"},
                "model": {"base_url": "http://model", "base_model": "qwen"},
                "database": {"dsn": "postgresql://unused"},
            },
            packs_root=pack_dir.parent,
            formal=True,
        )
