"""OpenAI-compatible chat client for the model service.

Exactly one explicitly configured endpoint. There is deliberately no cloud
fallback, no silent base-model fallback, and no hidden retry: a failed call
raises ModelUnavailableError and the API layer reports degraded=true. The
single allowed format-repair retry is an explicit second `generate` call made
by the caller, never something this client does on its own.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import httpx


class ModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    total_ms: float


@dataclass(frozen=True)
class NativeToolCall:
    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ToolGenerationResult:
    content: str
    tool_calls: tuple[NativeToolCall, ...]
    raw_completion: str
    input_tokens: int
    output_tokens: int
    total_ms: float


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    upstream_stream: bool = True


class OpenAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        adapter: str | None = None,
        api_key: str = "unused",
        timeout_s: float = 60.0,
        generation: GenerationConfig | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._base_model = model
        self._adapter = adapter
        # The formal model service selects the loaded policy by request model.
        # Sending the base name for an adapter arm is therefore forbidden.
        self._served_model = adapter or model
        self._generation = generation or GenerationConfig()
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
            trust_env=False,
        )

    @property
    def model(self) -> str:
        return self._served_model

    @property
    def base_model(self) -> str:
        return self._base_model

    @property
    def adapter(self) -> str | None:
        return self._adapter

    def list_models(self) -> list[str]:
        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"model server unavailable: {exc}") from exc
        return [row["id"] for row in response.json().get("data", [])]

    def model_identity(self) -> dict:
        """Return the server-attested model and adapter identity.

        Model names alone cannot prove which LoRA files are loaded. The
        formal Transformers/PEFT server therefore exposes a small identity
        endpoint containing the pinned base revision and adapter tree hash.
        """

        try:
            response = self._client.get("/v1/anima/model-identity")
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailableError(f"model identity unavailable: {exc}") from exc
        if not isinstance(value, dict):
            raise ModelUnavailableError("malformed model identity response")
        return value

    def generate(self, system: str, messages: Sequence[dict[str, Any]]) -> GenerationResult:
        if self._generation.upstream_stream:
            return self._generate_streamed(system, messages)
        payload = {
            "model": self._served_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self._generation.temperature,
            "top_p": self._generation.top_p,
            "max_tokens": self._generation.max_tokens,
        }
        started = time.monotonic()
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"model call failed: {exc}") from exc
        total_ms = (time.monotonic() - started) * 1000
        body = response.json()
        try:
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(f"malformed model response: {exc}") from exc
        return GenerationResult(
            text=text or "",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            ttft_ms=None,
            total_ms=total_ms,
        )

    def generate_tools(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str = "required",
    ) -> ToolGenerationResult:
        """Request one native OpenAI-compatible tool proposal.

        Tool calls are accepted only from ``message.tool_calls``. This client
        never extracts a function name from natural-language content.
        """

        payload = {
            "model": self._served_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": [dict(row) for row in tools],
            "tool_choice": tool_choice,
            "temperature": self._generation.temperature,
            "top_p": self._generation.top_p,
            "max_tokens": self._generation.max_tokens,
            "stream": False,
        }
        started = time.monotonic()
        try:
            response = self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            raw_calls = message.get("tool_calls") or []
            usage = body.get("usage", {})
            extension = body.get("anima", {})
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelUnavailableError(f"malformed tool response: {exc}") from exc
        calls: list[NativeToolCall] = []
        try:
            for row in raw_calls:
                function = row["function"]
                arguments = function["arguments"]
                if not isinstance(arguments, str):
                    raise TypeError("tool arguments must be a JSON string")
                calls.append(
                    NativeToolCall(
                        call_id=str(row["id"]),
                        name=str(function["name"]),
                        arguments_json=arguments,
                    )
                )
        except (KeyError, TypeError) as exc:
            raise ModelUnavailableError(f"malformed tool call: {exc}") from exc
        total_ms = (time.monotonic() - started) * 1000
        return ToolGenerationResult(
            content=str(message.get("content") or ""),
            tool_calls=tuple(calls),
            raw_completion=str(extension.get("raw_completion") or message.get("content") or ""),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            total_ms=total_ms,
        )

    def _generate_streamed(
        self, system: str, messages: Sequence[dict[str, Any]]
    ) -> GenerationResult:
        """Collect one real upstream stream and preserve TTFT/usage for audit.

        The Runtime still validates the complete state+answer contract before
        committing memory or exposing the served answer. This is upstream
        token streaming, not a claim that the HTTP API forwards unvalidated
        deltas to the user.
        """

        payload = {
            "model": self._served_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self._generation.temperature,
            "top_p": self._generation.top_p,
            "max_tokens": self._generation.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.monotonic()
        first_token_at: float | None = None
        parts: list[str] = []
        usage: dict = {}
        saw_done = False
        try:
            with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ModelUnavailableError(
                            f"malformed model stream JSON: {exc.msg}"
                        ) from exc
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage = event_usage
                    choices = event.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if content:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            parts.append(str(content))
        except ModelUnavailableError:
            raise
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise ModelUnavailableError(f"model stream failed: {exc}") from exc
        if not saw_done:
            raise ModelUnavailableError("model stream ended without [DONE]")
        total_ms = (time.monotonic() - started) * 1000
        return GenerationResult(
            text="".join(parts),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            ttft_ms=(first_token_at - started) * 1000 if first_token_at is not None else None,
            total_ms=total_ms,
        )

    def generate_stream(self, system: str, messages: Sequence[dict[str, Any]]) -> Iterator[str]:
        """Yield text deltas; raises ModelUnavailableError on transport failure."""

        payload = {
            "model": self._served_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self._generation.temperature,
            "top_p": self._generation.top_p,
            "max_tokens": self._generation.max_tokens,
            "stream": True,
        }
        try:
            with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        return
                    delta = json.loads(data).get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"model stream failed: {exc}") from exc


# Backward-compatible import for older callers. The client speaks the
# OpenAI wire protocol and is no longer coupled to one inference engine.
VLLMChatClient = OpenAIChatClient
