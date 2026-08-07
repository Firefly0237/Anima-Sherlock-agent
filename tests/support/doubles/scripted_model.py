"""Scripted ChatModel double for unit tests (never used in formal runs)."""

from __future__ import annotations

import json
from typing import Callable, Sequence

from anima.serve.inference.model_client import GenerationResult, ModelUnavailableError


def render_output(answer_mode: str, answer: str, *, lore_ids=(), memory_ids=(), ops=()) -> str:
    state = {
        "answer_mode": answer_mode,
        "used_lore_ids": list(lore_ids),
        "used_memory_ids": list(memory_ids),
        "memory_ops": [dict(op) for op in ops],
    }
    return f"<anima_state>{json.dumps(state, ensure_ascii=False)}</anima_state><answer>{answer}</answer>"


class ScriptedModel:
    """Returns responses from a callable `responder(system, messages) -> str`,
    or raises ModelUnavailableError if `fail` is set."""

    def __init__(
        self,
        responder: Callable[[str, Sequence[dict[str, str]]], str],
        *,
        model: str = "test-stub",
        served_models: Sequence[str] | None = None,
        identity: dict | None = None,
        fail: bool = False,
        input_tokens: int = 10,
        output_tokens: int = 20,
    ) -> None:
        self._responder = responder
        self._model = model
        self._served_models = list(served_models) if served_models is not None else [model]
        self._identity = dict(identity) if identity is not None else None
        self._fail = fail
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    @property
    def model(self) -> str:
        return self._model

    def list_models(self) -> list[str]:
        if self._fail:
            raise ModelUnavailableError("scripted failure")
        return list(self._served_models)

    def model_identity(self) -> dict:
        if self._fail:
            raise ModelUnavailableError("scripted failure")
        if self._identity is None:
            raise ModelUnavailableError("scripted identity unavailable")
        return dict(self._identity)

    def generate(self, system: str, messages: Sequence[dict[str, str]]) -> GenerationResult:
        self.calls.append((system, [dict(m) for m in messages]))
        if self._fail:
            raise ModelUnavailableError("scripted failure")
        text = self._responder(system, messages)
        return GenerationResult(
            text=text,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            ttft_ms=5.0,
            total_ms=12.0,
        )
