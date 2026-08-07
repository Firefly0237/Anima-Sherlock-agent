"""Unit tests for serve layer: embedding, metrics, model client, runtime."""

from __future__ import annotations

import math

import httpx
import pytest

from anima.serve.core.metrics import MetricsRegistry
from anima.serve.inference.embedding import HashEmbedder
from anima.serve.inference.model_client import (
    GenerationConfig,
    ModelUnavailableError,
    VLLMChatClient,
)


class TestHashEmbedder:
    def test_deterministic_and_normalized(self):
        emb = HashEmbedder(dim=64)
        [a] = emb.embed(["岚渡站的钟楼"])
        [b] = emb.embed(["岚渡站的钟楼"])
        assert a == b
        assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
        assert len(a) == 64

    def test_different_text_differs(self):
        emb = HashEmbedder(dim=128)
        [a] = emb.embed(["青芜红茶"])
        [b] = emb.embed(["岭脊快车"])
        assert a != b

    def test_empty_input(self):
        assert HashEmbedder().embed([]) == []


class TestMetrics:
    def test_counter_and_histogram_render(self):
        m = MetricsRegistry()
        m.inc("anima_requests_total", persona="landu")
        m.inc("anima_requests_total", persona="landu")
        m.observe_ms("anima_total_ms", 120)
        text = m.render()
        assert 'anima_requests_total{persona="landu"} 2' in text
        assert "anima_total_ms_bucket" in text
        assert "anima_total_ms_count 1" in text
        assert "anima_total_ms_sum 120" in text


class TestVLLMClient:
    def _client(self, handler) -> VLLMChatClient:
        transport = httpx.MockTransport(handler)
        client = VLLMChatClient(
            base_url="http://model",
            model="qwen",
            generation=GenerationConfig(max_tokens=32, upstream_stream=False),
        )
        client._client = httpx.Client(transport=transport, base_url="http://model")
        return client

    def test_generate_parses_response(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "<answer>好</answer>"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )

        client = self._client(handler)
        result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert result.text == "<answer>好</answer>"
        assert result.input_tokens == 5 and result.output_tokens == 3

    def test_http_error_raises_model_unavailable(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        client = self._client(handler)
        with pytest.raises(ModelUnavailableError):
            client.generate("sys", [{"role": "user", "content": "hi"}])

    def test_model_identity_is_read_from_attestation_endpoint(self):
        expected = {
            "backend": "transformers_peft",
            "base_model": "qwen",
            "base_model_revision": "commit",
            "adapter_id": "sft-42",
            "adapter_sha256": "a" * 64,
            "served_model": "sft-42",
        }
        client = self._client(lambda request: httpx.Response(200, json=expected))
        assert client.model_identity() == expected

    def test_model_identity_rejects_non_object(self):
        client = self._client(lambda request: httpx.Response(200, json=[]))
        with pytest.raises(ModelUnavailableError, match="malformed model identity"):
            client.model_identity()

    def test_generate_streamed_records_ttft_usage_and_requires_done(self):
        events = "\n".join(
            (
                'data: {"choices":[{"delta":{"content":"<answer>"}}]}',
                'data: {"choices":[{"delta":{"content":"好</answer>"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}',
                "data: [DONE]",
                "",
            )
        )
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, text=events, headers={"content-type": "text/event-stream"}
            )
        )
        client = VLLMChatClient(
            base_url="http://model",
            model="qwen",
            generation=GenerationConfig(max_tokens=32, upstream_stream=True),
        )
        client._client = httpx.Client(transport=transport, base_url="http://model")
        result = client.generate("sys", [{"role": "user", "content": "hi"}])
        assert result.text == "<answer>好</answer>"
        assert result.input_tokens == 5 and result.output_tokens == 3
        assert result.ttft_ms is not None and result.ttft_ms >= 0.0

        broken = httpx.MockTransport(
            lambda request: httpx.Response(200, text='data: {"choices":[]}\n')
        )
        client._client = httpx.Client(transport=broken, base_url="http://model")
        with pytest.raises(ModelUnavailableError, match=r"without \[DONE\]"):
            client.generate("sys", [{"role": "user", "content": "hi"}])

    def test_malformed_body_raises(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True})

        client = self._client(handler)
        with pytest.raises(ModelUnavailableError):
            client.generate("sys", [{"role": "user", "content": "hi"}])

    def test_adapter_is_sent_as_request_model_not_base(self):
        sent: dict = {}

        def handler(request):
            import json as _json

            sent.update(_json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "<answer>x</answer>"}}], "usage": {}},
            )

        transport = httpx.MockTransport(handler)
        client = VLLMChatClient(
            base_url="http://model",
            model="qwen-base",
            adapter="sft-42",
            generation=GenerationConfig(upstream_stream=False),
        )
        client._client = httpx.Client(transport=transport, base_url="http://model")
        client.generate("sys", [{"role": "user", "content": "hi"}])
        # The model service applies the LoRA only when request model == adapter name.
        assert sent["model"] == "sft-42"
        assert client.model == "sft-42" and client.base_model == "qwen-base"

    def test_no_adapter_sends_base(self):
        transport = httpx.MockTransport(
            lambda r: httpx.Response(
                200, json={"choices": [{"message": {"content": "x"}}], "usage": {}}
            )
        )
        client = VLLMChatClient(base_url="http://model", model="qwen-base")
        client._client = httpx.Client(transport=transport, base_url="http://model")
        assert client.model == "qwen-base" and client.adapter is None

    def test_client_ignores_environment_proxy(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:9")
        client = VLLMChatClient(base_url="http://model", model="qwen-base")
        try:
            assert client._client.trust_env is False
        finally:
            client._client.close()
