"""Pinned Transformers/PEFT model server with an OpenAI-compatible API.

This server loads exactly one base revision and, when configured, one LoRA
adapter. The identity endpoint attests the adapter directory tree hash so the
Runtime can reject a process serving different weights under the same name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

try:  # Keep module importable in lightweight test environments without FastAPI.
    from fastapi import Request as FastAPIRequest
except Exception:  # pragma: no cover - exercised only when serving deps are absent.
    FastAPIRequest = Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Match the release/runtime adapter tree hash exactly."""

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def resolve_model_source(
    base_model: str,
    base_model_revision: str,
    base_model_path: Path | None,
) -> tuple[str, dict[str, Any]]:
    """Return a from_pretrained source while keeping public identity pinned.

    Training may use the base model as a local, hash-checked snapshot,
    but runtime identity still needs to attest the upstream model id and
    revision. Loading from a local path avoids network fallback while the
    identity endpoint below continues to expose `base_model` and
    `base_model_revision` from the Runtime config.
    """

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if base_model_path is None:
        kwargs["revision"] = base_model_revision
        return base_model, kwargs
    return str(base_model_path.resolve()), kwargs


def freeze_model_for_inference(model: Any) -> Any:
    """Put a loaded model into non-trainable inference mode before serving."""

    model.eval()
    if hasattr(model, "requires_grad_"):
        model.requires_grad_(False)
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
    if any(getattr(parameter, "requires_grad", False) for parameter in model.parameters()):
        raise RuntimeError("formal model server loaded trainable parameters")
    return model


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    enable_thinking: bool,
    tools: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Render the model chat template with explicit visible-reasoning control."""

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = list(tools)
    return tokenizer.apply_chat_template(list(messages), **kwargs)


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_XML_FUNCTION_RE = re.compile(
    r"^\s*<function=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*(.*?)\s*</function>\s*$",
    re.DOTALL,
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_.:-]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_qwen_tool_calls(text: str) -> tuple[dict[str, Any], ...]:
    """Parse only explicit Qwen tool blocks into OpenAI tool-call objects.

    The frozen Qwen3.6 tokenizer uses ``function/parameter`` XML inside the
    outer block, while earlier Qwen3 templates use one JSON object. Both are
    protocol forms emitted by model-owned templates; prose and bare JSON are
    deliberately ignored.
    """

    calls: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_BLOCK_RE.findall(text):
        if raw.lstrip().startswith("{"):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed Hermes tool JSON: {exc.msg}") from exc
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                raise ValueError("Hermes tool call must contain a string name")
            arguments = value.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError("Hermes tool call arguments must be an object")
            name = value["name"]
        else:
            match = _XML_FUNCTION_RE.fullmatch(raw)
            if match is None:
                raise ValueError("malformed Qwen XML function block")
            name, body = match.groups()
            arguments = {}
            spans: list[tuple[int, int]] = []
            for parameter in _XML_PARAMETER_RE.finditer(body):
                key, value = parameter.groups()
                if key in arguments:
                    raise ValueError(f"duplicate Qwen XML parameter: {key}")
                arguments[key] = value.strip()
                spans.append(parameter.span())
            residual = body
            for start, end in reversed(spans):
                residual = residual[:start] + residual[end:]
            if residual.strip():
                raise ValueError("unexpected content in Qwen XML function block")
        calls.append(
            {
                "id": f"call-{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                },
            }
        )
    return tuple(calls)


# Historical helper name retained for imports; parsing now follows the actual
# frozen tokenizer and supports both Qwen protocol variants.
parse_hermes_tool_calls = parse_qwen_tool_calls


class TransformersPolicy:
    def __init__(
        self,
        *,
        base_model: str,
        base_model_revision: str,
        base_model_path: Path | None,
        adapter_id: str | None,
        adapter_dir: Path | None,
        expected_adapter_sha256: str | None,
        device: str,
        torch_dtype: str,
        load_in_4bit: bool,
        bnb_4bit_quant_type: str,
        bnb_4bit_use_double_quant: bool,
        bnb_4bit_compute_dtype: str,
        enable_thinking: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the requested model-server device")
        if (adapter_id is None) != (adapter_dir is None):
            raise ValueError("adapter_id and adapter_dir must be provided together")
        if adapter_dir is not None and not (adapter_dir / "adapter_config.json").is_file():
            raise ValueError(f"invalid PEFT adapter directory: {adapter_dir}")

        dtype = getattr(torch, torch_dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {torch_dtype}")
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype, None)
        if compute_dtype is None:
            raise ValueError(f"unsupported bnb compute dtype: {bnb_4bit_compute_dtype}")
        computed_adapter_sha = tree_sha256(adapter_dir) if adapter_dir is not None else None
        if computed_adapter_sha != expected_adapter_sha256:
            raise ValueError(
                "adapter tree hash mismatch: "
                f"computed={computed_adapter_sha!r}, expected={expected_adapter_sha256!r}"
            )

        model_source, source_kwargs = resolve_model_source(
            base_model, base_model_revision, base_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(model_source, **source_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        if device == "cuda:0":
            device_map: dict[str, Any] = {"": 0}
        else:
            device_map = {"": device}
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **source_kwargs,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        freeze_model_for_inference(model)

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._enable_thinking = enable_thinking
        self._lock = threading.Lock()
        self._served_model = adapter_id or base_model
        self.identity = {
            "backend": "transformers_peft",
            "base_model": base_model,
            "base_model_revision": base_model_revision,
            "base_model_path": str(base_model_path.resolve())
            if base_model_path is not None
            else None,
            "adapter_id": adapter_id,
            "adapter_sha256": computed_adapter_sha,
            "served_model": self._served_model,
            "device": device,
            "torch_dtype": torch_dtype,
            "quantization": (
                {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": bnb_4bit_quant_type,
                    "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
                    "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
                }
                if load_in_4bit
                else None
            ),
            "chat_template": {"enable_thinking": enable_thinking},
            "code_commit": os.environ.get("ANIMA_CODE_COMMIT", "").strip() or None,
        }

    @property
    def served_model(self) -> str:
        return self._served_model

    def _inputs(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
    ):
        rendered = render_chat_prompt(
            self._tokenizer,
            messages,
            enable_thinking=self._enable_thinking,
            tools=tools,
        )
        return self._tokenizer(rendered, return_tensors="pt").to(self._device)

    def _generation_kwargs(
        self,
        inputs: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        if not 1 <= max_tokens <= 1024:
            raise ValueError("max_tokens must be in [1, 1024]")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        do_sample = temperature > 0
        kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update({"temperature": temperature, "top_p": top_p})
        return kwargs

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[str, int, int]:
        inputs = self._inputs(messages, tools=tools)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        kwargs = self._generation_kwargs(
            inputs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        with self._lock, self._torch.inference_mode():
            output = self._model.generate(**kwargs)
        completion_ids = output[0, prompt_tokens:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return text, prompt_tokens, int(completion_ids.shape[-1])

    def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[Iterator[str], int, dict[str, Any]]:
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=180.0,
        )
        kwargs = self._generation_kwargs(
            inputs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        kwargs["streamer"] = streamer
        state: dict[str, Any] = {"error": None}

        def worker() -> None:
            try:
                with self._lock, self._torch.inference_mode():
                    self._model.generate(**kwargs)
            except BaseException as exc:  # propagated after the stream is closed
                state["error"] = exc
                streamer.on_finalized_text("", stream_end=True)

        thread = threading.Thread(target=worker, name="anima-transformers-generate", daemon=True)
        thread.start()

        def chunks() -> Iterator[str]:
            yield from streamer
            thread.join()
            if state["error"] is not None:
                raise RuntimeError(f"model generation failed: {state['error']}") from state["error"]

        return chunks(), prompt_tokens, state

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def _request_fields(
    payload: dict[str, Any], served_model: str
) -> tuple[list[dict[str, str]], int, float, float]:
    if payload.get("model") != served_model:
        raise ValueError(f"model must be {served_model!r}")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for row in raw_messages:
        if not isinstance(row, dict) or row.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("each message must have a supported role")
        content = row.get("content")
        if not isinstance(content, str):
            raise ValueError("each message content must be a string")
        messages.append({"role": str(row["role"]), "content": content})
    return (
        messages,
        int(payload.get("max_tokens", 192)),
        float(payload.get("temperature", 0.0)),
        float(payload.get("top_p", 1.0)),
    )


def _tool_fields(payload: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str]:
    raw_tools = payload.get("tools")
    tool_choice = str(payload.get("tool_choice", "auto"))
    if tool_choice not in {"auto", "required", "none"}:
        raise ValueError("tool_choice must be 'auto', 'required', or 'none'")
    if raw_tools is None:
        if tool_choice == "required":
            raise ValueError("tool_choice='required' needs tools")
        return None, tool_choice
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("tools must be a non-empty list")
    functions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw_tools:
        if not isinstance(row, dict) or row.get("type") != "function":
            raise ValueError("each tool must have type='function'")
        function = row.get("function")
        if not isinstance(function, dict):
            raise ValueError("each tool must contain a function object")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("tool function names must be non-empty and unique")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError("tool parameters must be an object JSON Schema")
        seen.add(name)
        functions.append(dict(function))
    return functions, tool_choice


def create_app(policy: TransformersPolicy):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="Anima pinned Transformers model server", version="1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": policy.served_model, "object": "model", "owned_by": "anima"}],
        }

    @app.get("/v1/anima/model-identity")
    def model_identity():
        return dict(policy.identity)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: FastAPIRequest):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            messages, max_tokens, temperature, top_p = _request_fields(payload, policy.served_model)
            tools, tool_choice = _tool_fields(payload)
            if tools is not None and bool(payload.get("stream", False)):
                raise ValueError("streaming tool calls are not supported")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request"}}, status_code=400
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if not bool(payload.get("stream", False)):
            try:
                text, prompt_tokens, completion_tokens = policy.generate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                )
            except Exception as exc:
                return JSONResponse(
                    {"error": {"message": str(exc), "type": "generation_error"}},
                    status_code=500,
                )
            tool_parse_status = "not_requested"
            if tools is not None:
                try:
                    tool_calls = parse_qwen_tool_calls(text)
                    tool_parse_status = "ok" if tool_calls else "no_tool_call"
                except ValueError:
                    tool_calls = ()
                    tool_parse_status = "malformed"
            else:
                tool_calls = ()
            if tool_choice == "required" and not tool_calls:
                finish_reason = "stop"
            else:
                finish_reason = "tool_calls" if tool_calls else "stop"
            content = _TOOL_CALL_BLOCK_RE.sub("", text).strip()
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": policy.served_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content or None,
                            **({"tool_calls": list(tool_calls)} if tool_calls else {}),
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "anima": {
                    "raw_completion": text,
                    "tool_parse_status": tool_parse_status,
                },
            }

        def events() -> Iterator[str]:
            chunks, prompt_tokens, _state = policy.stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            completion_text = ""
            for chunk in chunks:
                completion_text += chunk
                event = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": policy.served_model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            completion_tokens = policy.count_tokens(completion_text)
            usage_event = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": policy.served_model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(usage_event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--identity-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument(
        "--bnb-4bit-use-double-quant", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    if config.get("run_mode") not in {"pilot", "formal"}:
        raise SystemExit("model server requires a pilot or formal runtime config")
    model = config["model"]
    adapter_id = model.get("adapter_id")
    expected_adapter_sha = model.get("adapter_sha256")
    if adapter_id is not None and args.adapter_dir is None:
        raise SystemExit("trained runtime config requires --adapter-dir")
    if adapter_id is None and args.adapter_dir is not None:
        raise SystemExit("base runtime config forbids --adapter-dir")

    policy = TransformersPolicy(
        base_model=str(model["base_model"]),
        base_model_revision=str(model["base_model_revision"]),
        base_model_path=args.base_model_path.resolve()
        if args.base_model_path is not None
        else None,
        adapter_id=str(adapter_id) if adapter_id is not None else None,
        adapter_dir=args.adapter_dir.resolve() if args.adapter_dir is not None else None,
        expected_adapter_sha256=str(expected_adapter_sha)
        if expected_adapter_sha is not None
        else None,
        device=args.device,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        enable_thinking=bool(args.enable_thinking),
    )
    if args.identity_output is not None:
        if args.identity_output.exists():
            raise SystemExit(f"refusing to overwrite identity output: {args.identity_output}")
        args.identity_output.parent.mkdir(parents=True, exist_ok=True)
        args.identity_output.write_text(
            json.dumps(policy.identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    import uvicorn

    uvicorn.run(create_app(policy), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
