from __future__ import annotations

import json

import pytest

from anima.serve.inference.model_client import (
    GenerationConfig,
    ModelUnavailableError,
    OpenAIChatClient,
)


class _Response:
    def __init__(self, body: dict) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        return self.body


class _HTTPClient:
    def __init__(self, body: dict) -> None:
        self.body = body
        self.requests: list[tuple[str, dict]] = []

    def post(self, path: str, *, json: dict) -> _Response:
        self.requests.append((path, json))
        return _Response(self.body)


def _client(body: dict) -> tuple[OpenAIChatClient, _HTTPClient]:
    client = OpenAIChatClient(
        base_url="http://127.0.0.1:9",
        model="base",
        adapter="exact-sft",
        generation=GenerationConfig(upstream_stream=True),
    )
    client._client.close()
    fake = _HTTPClient(body)
    client._client = fake
    return client, fake


def test_generate_tools_consumes_only_openai_message_tool_calls() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ignored prose",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "request_hint",
                                "arguments": '{"focus":"下一档"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "anima": {"raw_completion": "<tool_call>raw</tool_call>"},
    }
    client, fake = _client(body)
    result = client.generate_tools(
        "system",
        [{"role": "user", "content": "提示"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "request_hint",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert result.tool_calls[0].name == "request_hint"
    assert json.loads(result.tool_calls[0].arguments_json) == {"focus": "下一档"}
    assert result.raw_completion == "<tool_call>raw</tool_call>"
    assert fake.requests[0][1]["stream"] is False
    assert fake.requests[0][1]["tool_choice"] == "required"
    assert fake.requests[0][1]["model"] == "exact-sft"


def test_generate_tools_does_not_extract_tool_from_content_and_rejects_non_string_args() -> None:
    client, _fake = _client(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"request_hint"}</tool_call>',
                    }
                }
            ]
        }
    )
    result = client.generate_tools("system", [], tools=[{"type": "function"}])
    assert result.tool_calls == ()

    client, _fake = _client(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "request_hint", "arguments": {}},
                            }
                        ]
                    }
                }
            ]
        }
    )
    with pytest.raises(ModelUnavailableError, match="arguments must be a JSON string"):
        client.generate_tools("system", [], tools=[{"type": "function"}])
