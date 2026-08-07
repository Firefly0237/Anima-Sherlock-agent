"""Deterministic case-game runtime for the Sherlock Mystery Game.

The public package surface is lazy on purpose. Core serving and training
modules import individual ``anima.case_game`` submodules; eagerly importing the
whole runtime here creates a cycle through ``serve.core.runtime`` before its
model constants have been initialised. PEP 562 lazy attributes preserve the
existing ``from anima.case_game import ...`` API without making every submodule
import load the model client and repository stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "CaseGameDemo": ("anima.case_game.core.demo", "CaseGameDemo"),
    "CaseGameDemoError": ("anima.case_game.core.demo", "CaseGameDemoError"),
    "CaseGamePersistenceError": (
        "anima.case_game.core.demo",
        "CaseGamePersistenceError",
    ),
    "CaseGameVersionConflictError": (
        "anima.case_game.core.demo",
        "CaseGameVersionConflictError",
    ),
    "HostAnswer": ("anima.case_game.core.demo", "HostAnswer"),
    "CaseGameEngine": ("anima.case_game.core.engine", "CaseGameEngine"),
    "GameState": ("anima.case_game.core.engine", "GameState"),
    "GameTurnResult": ("anima.case_game.core.engine", "GameTurnResult"),
    "CasePackValidationError": (
        "anima.case_game.core.loader",
        "CasePackValidationError",
    ),
    "load_case_pack": ("anima.case_game.core.loader", "load_case_pack"),
    "CasePack": ("anima.case_game.core.models", "CasePack"),
    "SpoilerHit": ("anima.case_game.core.models", "SpoilerHit"),
    "CASE_PERSONA_ADAPTER_ID": (
        "anima.case_game.core.persona_adapter",
        "CASE_PERSONA_ADAPTER_ID",
    ),
    "CASE_PERSONA_ADAPTER_SHA256": (
        "anima.case_game.core.persona_adapter",
        "CASE_PERSONA_ADAPTER_SHA256",
    ),
    "GuardedCaseAnswer": (
        "anima.case_game.core.persona_adapter",
        "GuardedCaseAnswer",
    ),
    "PreparedCasePersonaTurn": (
        "anima.case_game.core.persona_adapter",
        "PreparedCasePersonaTurn",
    ),
    "guard_case_answer": (
        "anima.case_game.core.persona_adapter",
        "guard_case_answer",
    ),
    "prepare_case_persona_turn": (
        "anima.case_game.core.persona_adapter",
        "prepare_case_persona_turn",
    ),
    "CaseGameModelIdentityError": (
        "anima.case_game.runtime.model",
        "CaseGameModelIdentityError",
    ),
    "LiveCaseAnswerer": ("anima.case_game.runtime.model", "LiveCaseAnswerer"),
    "LiveDPOAnswerer": ("anima.case_game.runtime.model", "LiveDPOAnswerer"),
    "load_expected_model_identity": (
        "anima.case_game.runtime.model",
        "load_expected_model_identity",
    ),
    "MemoryCommitResult": (
        "anima.case_game.runtime.player_memory",
        "MemoryCommitResult",
    ),
    "MemoryRetrieval": (
        "anima.case_game.runtime.player_memory",
        "MemoryRetrieval",
    ),
    "PlayerMemoryError": (
        "anima.case_game.runtime.player_memory",
        "PlayerMemoryError",
    ),
    "PlayerMemoryService": (
        "anima.case_game.runtime.player_memory",
        "PlayerMemoryService",
    ),
    "validate_player_id": (
        "anima.case_game.runtime.player_memory",
        "validate_player_id",
    ),
    "CaseSessionStore": (
        "anima.case_game.runtime.session_store",
        "CaseSessionStore",
    ),
    "InMemoryCaseSessionStore": (
        "anima.case_game.runtime.session_store",
        "InMemoryCaseSessionStore",
    ),
    "PostgresCaseSessionStore": (
        "anima.case_game.runtime.session_store",
        "PostgresCaseSessionStore",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
