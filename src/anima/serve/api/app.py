"""Compose real AppServices from a runtime config and expose the ASGI app.

This is the production wiring: PostgresRepository + BGEEmbedder +
OpenAIChatClient, no test doubles. A formal deployment refuses ANIMA_TEST_MODE. Boot with: uvicorn anima.serve.api.app:app (after ANIMA_CONFIG is set),
or `python -m anima.serve.api.app` for a dev run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from anima.persona.contracts.pack import load_pack
from anima.serve.api.persona import create_app
from anima.serve.core.metrics import MetricsRegistry
from anima.serve.core.repositories import PostgresRepository
from anima.serve.core.runtime import AppServices, ModelInfo, RuntimeConfig, build_lore_indexes
from anima.serve.inference.embedding import BGEEmbedder
from anima.serve.inference.model_client import GenerationConfig, OpenAIChatClient


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_services(
    config: dict[str, Any], *, packs_root: str | Path, formal: bool = True
) -> AppServices:
    code_commit = os.environ.get("ANIMA_CODE_COMMIT", "").strip()
    if formal and not re.fullmatch(r"[0-9a-f]{7,64}", code_commit):
        raise ValueError(
            "formal runtime requires a hexadecimal ANIMA_CODE_COMMIT for trace provenance"
        )
    config_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    runtime_config = RuntimeConfig.from_mapping(config, formal=formal)
    emb_cfg = config["embedding"]
    if emb_cfg["backend"] == "bge":
        embedder = BGEEmbedder(
            emb_cfg["model_id"],
            revision=emb_cfg["revision"],
            device=emb_cfg.get("device", "cpu"),
        )
    else:
        raise ValueError(
            f"unsupported embedding backend for a real deployment: {emb_cfg['backend']}"
        )

    model_cfg = config["model"]
    model_client = OpenAIChatClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["base_model"],
        adapter=model_cfg.get("adapter_id"),
        generation=GenerationConfig(
            temperature=model_cfg.get("temperature", 0.7),
            top_p=model_cfg.get("top_p", 0.9),
            max_tokens=model_cfg.get("max_tokens", 512),
            upstream_stream=bool(model_cfg.get("upstream_stream", True)),
        ),
    )
    model_server_identity: dict[str, Any] | None = None
    if formal:
        model_server_identity = model_client.model_identity()
        expected_identity = {
            "backend": "transformers_peft",
            "base_model": model_cfg["base_model"],
            "base_model_revision": model_cfg["base_model_revision"],
            "adapter_id": model_cfg.get("adapter_id"),
            "adapter_sha256": model_cfg.get("adapter_sha256"),
            "served_model": model_cfg.get("adapter_id") or model_cfg["base_model"],
            "code_commit": code_commit,
        }
        mismatches = {
            key: {"actual": model_server_identity.get(key), "expected": value}
            for key, value in expected_identity.items()
            if model_server_identity.get(key) != value
        }
        if mismatches:
            raise ValueError(f"formal model server identity mismatch: {mismatches}")

    packs = {}
    for pack_dir in sorted(Path(packs_root).iterdir()):
        if pack_dir.is_dir():
            pack = load_pack(pack_dir, formal=formal)
            packs[pack.manifest.persona_id] = pack

    repo = PostgresRepository(config["database"]["dsn"])
    repo.migrate()

    return AppServices(
        packs=packs,
        lore_indexes=build_lore_indexes(packs, embedder),
        repo=repo,
        model=model_client,
        embedder=embedder,
        metrics=MetricsRegistry(),
        model_info=ModelInfo(
            base_model=model_cfg["base_model"],
            adapter_id=model_cfg.get("adapter_id"),
            adapter_sha256=model_cfg.get("adapter_sha256"),
            base_model_revision=model_cfg["base_model_revision"],
            embedding_model_revision=emb_cfg["revision"],
        ),
        config=runtime_config,
        test_mode=False,
        eval_token_hash=(
            hashlib.sha256(os.environ["ANIMA_EVAL_TOKEN"].encode("utf-8")).hexdigest()
            if os.environ.get("ANIMA_EVAL_TOKEN")
            else None
        ),
        code_commit=code_commit or "pilot-unpinned",
        runtime_config_hash=config_hash,
        model_server_identity=model_server_identity,
    )


def build_app(config_path: str | Path, packs_root: str | Path):
    config = load_runtime_config(config_path)
    run_mode = str(config.get("run_mode", "formal"))
    if run_mode not in {"pilot", "formal"}:
        raise ValueError("runtime run_mode must be 'pilot' or 'formal'")
    formal = run_mode == "formal"
    services = build_services(config, packs_root=packs_root, formal=formal)
    return create_app(services, formal=formal)


def create_default_app():
    """uvicorn factory (no import-time side effects): `uvicorn anima.serve.api.app:create_default_app --factory`."""

    config_path = os.environ.get("ANIMA_CONFIG", "configs/runtime.example.yaml")
    packs_root = os.environ.get("ANIMA_PACKS_ROOT", "persona_packs/public")
    return build_app(config_path, packs_root)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "anima.serve.api.app:create_default_app",
        factory=True,
        host=os.environ.get("ANIMA_HOST", "127.0.0.1"),
        port=int(os.environ.get("ANIMA_PORT", "8080")),
    )


if __name__ == "__main__":
    main()
