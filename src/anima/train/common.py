"""Shared helpers for the training entrypoints.

Heavy training libraries are imported lazily inside the functions that train,
so local dry-runs work without torch installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from anima.data.synthesis.reward_records import (
    first_nonempty,
    iter_json_objects,
    normalize_conversations,
    normalize_text_value,
)
from anima.persona.contracts.provenance import BASE_MODEL_ID
from anima.rewards.core.parsing import normalize_focus_labels

FORMAT_INSTRUCTION = (
    "请严格按以下格式回答。不要省略任何标签。只输出一次这个结构，"
    "\\boxed{} 后立刻结束，不要继续续写第二个 <think> 或第二个 \\boxed{}。"
    "不要输出 <|endoftext|>、Human:、user:、角色:、来源作品: 或任何新题目：\n"
    "<think>用简短中文说明角色回复时需要关注什么。"
    "<focus>从 Knowledge, Style, Worldview, Emotion, Empathetic, Engagement, "
    "Human_Like, Extension, Memory, Safety 中选择一个或多个英文标签，用逗号分隔</focus>"
    "<focus_attr>用中文写出本次回复需要体现的角色属性</focus_attr></think>\n"
    "\\boxed{最终给用户的中文角色回复}"
)


OFFICIAL_OUTPUT_CONTRACT = (
    "请用中文扮演指定角色，保持角色身份、语气、价值观和对话上下文。"
    "输出必须是一个 assistant 回复，结构为："
    "<think>简短说明本次角色回复关注点。"
    "<focus>从 Knowledge, Style, Worldview, Emotion, Empathetic, Engagement, "
    "Human_Like, Extension, Memory, Safety 中选择一个或多个英文标签，用逗号分隔</focus>"
    "<focus_attr>用中文写出本次回复体现的角色属性</focus_attr></think>"
    "\\boxed{最终给用户的中文角色回复}"
)


@dataclass(frozen=True)
class PreparedRows:
    rows: list[dict[str, Any]]
    skipped: int
    input_files: list[Path]


class ChatTemplateError(ValueError):
    """Raised when a formal path cannot use the tokenizer chat template."""


def load_config(path: Path) -> dict[str, Any]:
    """Load a training config (JSON object with a ``.yaml`` suffix; PyYAML only as fallback)."""

    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on remote env.
            raise RuntimeError(
                f"{path} is not JSON-compatible and PyYAML is not installed. "
                "Keep configs as JSON objects or install pyyaml."
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping/object")
    return dict(value)


def read_records(paths: Sequence[Path], *, max_records: int | None = None) -> PreparedRows:
    records: list[dict[str, Any]] = []
    input_files: list[Path] = []
    for path in paths:
        files = _collect_input_files(path)
        input_files.extend(files)
        for file_path in files:
            for record in iter_json_objects(file_path):
                records.append(record)
                if max_records is not None and len(records) >= max_records:
                    return PreparedRows(records, skipped=0, input_files=input_files)
    return PreparedRows(records, skipped=0, input_files=input_files)


def build_sft_rows(
    paths: Sequence[Path],
    *,
    max_records: int | None = None,
    require_gold_format: bool = False,
) -> PreparedRows:
    """Build TRL prompt/completion rows for SFTTrainer.

    Rows with gold focus fields get a Character-R1 style completion; raw
    RoleBench rows fall back to their source answer wrapped in the required
    output format.
    """

    prepared = read_records(paths, max_records=max_records)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in prepared.rows:
        answer = select_assistant_answer(record)
        if not answer:
            skipped += 1
            continue

        has_gold = bool(record.get("gold_focus") and record.get("gold_focus_attr"))
        if require_gold_format and not has_gold:
            skipped += 1
            continue

        prompt = render_policy_prompt(record)
        completion = render_sft_completion(record, answer=answer)
        if not prompt.strip() or not completion.strip():
            skipped += 1
            continue

        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "id": str(record.get("id") or f"sft_{len(rows):06d}"),
                "character": _record_character(record),
                "source_work": _record_source_work(record),
            }
        )
    return PreparedRows(rows, skipped=skipped, input_files=prepared.input_files)


def build_sft_chat_rows(
    paths: Sequence[Path],
    *,
    max_records: int | None = None,
    require_gold_format: bool = False,
) -> PreparedRows:
    """Build TRL conversational prompt/completion rows for chat-template SFT.

    Prompt and completion are role messages; turn boundary tokens are left to
    the tokenizer chat template.
    """

    prepared = read_records(paths, max_records=max_records)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in prepared.rows:
        answer = select_assistant_answer(record)
        if not answer:
            skipped += 1
            continue

        has_gold = bool(record.get("gold_focus") and record.get("gold_focus_attr"))
        if require_gold_format and not has_gold:
            skipped += 1
            continue

        prompt_messages = record_to_messages(record)
        completion_content = render_sft_completion_content(record, answer=answer)
        if not prompt_messages or not completion_content.strip():
            skipped += 1
            continue

        rows.append(
            {
                "prompt": prompt_messages,
                "completion": [{"role": "assistant", "content": completion_content}],
                "id": str(record.get("id") or f"sft_chat_{len(rows):06d}"),
                "character": _record_character(record),
                "source_work": _record_source_work(record),
            }
        )
    return PreparedRows(rows, skipped=skipped, input_files=prepared.input_files)


def render_policy_prompt(record: Mapping[str, Any]) -> str:
    """Legacy string prompt renderer kept for dry-runs and old artifacts."""

    profile = _record_profile(record)
    character = _record_character(record)
    source_work = _record_source_work(record)
    conversations = normalize_conversations(dict(record))
    dialogue = render_dialogue(conversations)
    return (
        "你正在扮演一个中文角色。请只根据可见角色卡和对话上下文作答。\n\n"
        f"角色: {character}\n"
        f"来源作品: {source_work}\n"
        f"角色卡:\n{profile}\n\n"
        f"对话上下文:\n{dialogue}\n\n"
        f"{FORMAT_INSTRUCTION}\n"
    )


def record_to_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a standard chat-message view of one role-play record."""

    profile = _record_profile(record)
    character = _record_character(record)
    source_work = _record_source_work(record)

    system_content = "\n\n".join(
        (
            f"角色: {character}\n来源作品: {source_work}\n角色卡:\n{profile}",
            OFFICIAL_OUTPUT_CONTRACT,
        )
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for turn in normalize_conversations(dict(record)):
        role = _chat_role(turn.get("role"), character=character)
        content = str(turn.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    if len(messages) == 1:
        messages.append({"role": "user", "content": "请根据角色卡进行回应。"})
    return messages


def render_messages_with_chat_template(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any | None,
    *,
    mode: str = "formal",
    allow_legacy: bool = False,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render messages with the tokenizer chat template, failing closed for formal paths."""

    if _tokenizer_has_chat_template(tokenizer):
        kwargs = dict(chat_template_kwargs or {})
        rendered = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        return str(rendered)

    if allow_legacy or mode in {"dry_run", "local"}:
        return render_messages_legacy(messages)

    raise ChatTemplateError(
        "formal chat-template rendering requires tokenizer.apply_chat_template "
        "and a non-empty tokenizer.chat_template"
    )


def render_messages_legacy(messages: Sequence[Mapping[str, str]]) -> str:
    """Dry-run-only fallback for environments without the model tokenizer."""

    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts) + "\n"


def render_dialogue(conversations: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for turn in conversations:
        role = str(turn.get("role", "user")).strip() or "user"
        content = str(turn.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def render_sft_completion(record: Mapping[str, Any], *, answer: str) -> str:
    return f"{render_sft_completion_content(record, answer=answer)}<|im_end|>"


def render_sft_completion_content(record: Mapping[str, Any], *, answer: str) -> str:
    focus = normalize_focus_labels(record.get("gold_focus")) or ("Style", "Engagement")
    focus_attr = normalize_text_value(record.get("gold_focus_attr"))
    if not focus_attr:
        character = str(record.get("character") or record.get("role") or "该角色")
        focus_attr = f"保持{character}的身份、语气和互动推进感。"
    return (
        "<think>本次回复需要先保持角色身份，再根据用户问题推进对话。"
        f"<focus>{', '.join(focus)}</focus>"
        f"<focus_attr>{focus_attr}</focus_attr></think>\n"
        f"\\boxed{{{answer.strip()}}}"
    )


def select_assistant_answer(record: Mapping[str, Any]) -> str:
    for key in ("reference_answer", "answer", "response", "generated", "output", "completion"):
        if key not in record:
            continue
        text = _first_text(record[key])
        if text:
            return text
    return ""


def _record_character(record: Mapping[str, Any]) -> str:
    return first_nonempty(dict(record), ("character", "role", "name", "character_name"), "unknown")


def _record_profile(record: Mapping[str, Any]) -> str:
    """Return the character profile text, falling back to a minimal role card."""

    profile = first_nonempty(
        dict(record),
        ("profile", "persona", "description", "character_profile", "system_prompt"),
    )
    if profile:
        return str(profile)
    character = _record_character(record)
    source_work = _record_source_work(record)
    return (
        f"{character}角色卡：来自{source_work}。请保持该角色身份、称谓、语气和价值观，"
        "用中文回应，不要以普通助手口吻回答。"
    )


def _record_source_work(record: Mapping[str, Any]) -> str:
    # RoleBench 原始数据没有作品名字段，这里用角色名回填，别当真正的作品名用
    character = _record_character(record)
    return first_nonempty(
        dict(record), ("source_work", "work", "book", "movie"), f"RoleBench/{character}"
    )


def _chat_role(value: Any, *, character: str | None = None) -> str:
    role = str(value or "user").strip().lower()
    character_role = str(character or "").strip().lower()
    if role in {"assistant", "model", "bot", "角色", "模型"}:
        return "assistant"
    if character_role and role == character_role:
        return "assistant"
    if role in {"system"}:
        return "system"
    if role in {"user", "human", "person", "player", "用户", "玩家"}:
        return "user"
    # 角色扮演数据常把角色名直接写在 speaker role 里，Qwen 模板只认标准角色，
    # 所以叫不出名字的发言一律归 assistant
    if role:
        return "assistant"
    return "user"


def _tokenizer_has_chat_template(tokenizer: Any | None) -> bool:
    return (
        tokenizer is not None
        and callable(getattr(tokenizer, "apply_chat_template", None))
        and bool(getattr(tokenizer, "chat_template", None))
    )


def _first_text(value: Any) -> str:
    if isinstance(value, list | tuple):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    return normalize_text_value(value)


def parse_paths(values: Any) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, str):
        return [Path(values)]
    if isinstance(values, Sequence):
        return [Path(str(value)) for value in values]
    raise TypeError(f"expected path string or list of paths, got {type(values).__name__}")


def build_lora_config(config: Mapping[str, Any]):
    from peft import LoraConfig

    return LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("lora_bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(
            config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )


def load_policy_model_and_tokenizer(
    config: Mapping[str, Any],
    *,
    padding_side: str,
    adapter_path: str | None = None,
):
    """Load a 4-bit policy model plus tokenizer for adapter training."""

    if config.get("use_unsloth", False):
        return _load_unsloth_policy(config, padding_side=padding_side)
    return _load_transformers_policy(config, padding_side=padding_side, adapter_path=adapter_path)


def _load_transformers_policy(
    config: Mapping[str, Any],
    *,
    padding_side: str,
    adapter_path: str | None,
):
    from peft import PeftModel

    model, tokenizer = _load_base_causal_lm(config, padding_side=padding_side)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        return model, tokenizer, None
    return model, tokenizer, build_lora_config(config)


def load_dpo_policy_with_ref_adapter(
    config: Mapping[str, Any],
    *,
    padding_side: str,
    adapter_path: str,
    reference_adapter_name: str | None = "reference",
):
    """Load the DPO policy with the SFT adapter as trainable policy, optionally with a frozen reference copy."""

    from peft import PeftModel

    model, tokenizer = _load_base_causal_lm(config, padding_side=padding_side)
    # Mount the same SFT adapter as a trainable policy and frozen reference
    # when the installed TRL version does not create the reference copy itself.
    model = PeftModel.from_pretrained(
        model, adapter_path, is_trainable=True, adapter_name="default"
    )
    if reference_adapter_name:
        model.load_adapter(adapter_path, adapter_name=reference_adapter_name, is_trainable=False)
    model.set_adapter("default")
    return model, tokenizer, None


def load_frozen_adapter_policy(
    config: Mapping[str, Any],
    *,
    padding_side: str,
    adapter_path: str,
):
    """Load an immutable adapter policy for algorithms needing an explicit reference model."""

    if config.get("use_unsloth", False):
        raise RuntimeError("explicit frozen adapter references are not implemented for Unsloth")
    from peft import PeftModel

    model, _tokenizer = _load_base_causal_lm(config, padding_side=padding_side)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _load_base_causal_lm(config: Mapping[str, Any], *, padding_side: str):
    """Load the quantized base model and tokenizer shared by all policy loaders."""

    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = str(config.get("model_name_or_path", BASE_MODEL_ID))
    strict_local = bool(config.get("require_accepted_model_snapshot", False))
    model_path = Path(model_name).expanduser()
    if strict_local:
        if not model_path.is_absolute() or not model_path.is_dir():
            raise RuntimeError(
                "accepted MODEL snapshot must be an existing absolute local directory"
            )
        if config.get("local_files_only") is not True:
            raise RuntimeError("accepted MODEL loading requires local_files_only=true")
        if config.get("trust_remote_code") is not False:
            raise RuntimeError("accepted MODEL loading requires trust_remote_code=false")
        configured_map = config.get("device_map", "cuda:0")
        if configured_map not in ("cuda:0", 0, {"": 0}, {"": "cuda:0"}):
            raise RuntimeError("accepted MODEL loading requires an explicit single-CUDA device map")
        device_map: Any = {"": 0}
    else:
        device_map = config.get("device_map", "auto")
    revision = str(config.get("model_revision") or "").strip() or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False if strict_local else bool(config.get("trust_remote_code", True)),
        local_files_only=True if strict_local else bool(config.get("local_files_only", False)),
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if bool(config.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=_torch_dtype(
                torch, str(config.get("bnb_4bit_compute_dtype", "bfloat16"))
            ),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False if strict_local else bool(config.get("trust_remote_code", True)),
        local_files_only=True if strict_local else bool(config.get("local_files_only", False)),
        torch_dtype=_torch_dtype(torch, str(config.get("torch_dtype", "bfloat16"))),
        quantization_config=quantization_config,
        device_map=device_map,
    )
    if bool(config.get("gradient_checkpointing", True)):
        model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def _load_unsloth_policy(config: Mapping[str, Any], *, padding_side: str):
    from unsloth import FastLanguageModel

    model_name = str(config.get("model_name_or_path", BASE_MODEL_ID))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        revision=str(config.get("model_revision") or "").strip() or None,
        max_seq_length=int(config.get("max_seq_length", config.get("max_length", 1024))),
        dtype=None,
        load_in_4bit=bool(config.get("load_in_4bit", True)),
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config.get("lora_r", 16)),
        target_modules=list(
            config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        bias=str(config.get("lora_bias", "none")),
        use_gradient_checkpointing=str(config.get("unsloth_gradient_checkpointing", "unsloth")),
        random_state=int(config.get("seed", 42)),
    )
    return model, tokenizer, None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def fail_on_critical_dropped(*, stage: str, dropped: list[str], critical: set[str]) -> None:
    missing = sorted(set(dropped) & critical)
    if missing:
        raise RuntimeError(
            f"{stage} cannot continue because installed TRL does not support critical keys: {missing}"
        )


def supported_config_keys(config_cls: Any) -> set[str]:
    # Prefer dataclass fields, then fall back to the constructor signature.
    # An open **kwargs signature returns an empty set so callers pass all keys.
    fields = getattr(config_cls, "__dataclass_fields__", None)
    if isinstance(fields, dict) and fields:
        return set(fields)

    try:
        import inspect

        signature = inspect.signature(config_cls.__init__)
    except (TypeError, ValueError):
        return set()

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return set()
    return {name for name in params if name != "self"}


def _torch_dtype(torch_module: Any, name: str):
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    if normalized in {"auto", "none"}:
        return "auto"
    raise ValueError(f"unsupported torch dtype: {name}")


def _collect_input_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            child
            for child in path.rglob("*")
            if child.suffix.lower() in {".json", ".jsonl", ".parquet"}
        )
    return [path]
