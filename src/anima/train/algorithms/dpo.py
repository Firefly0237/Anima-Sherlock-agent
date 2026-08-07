"""Direct Preference Optimization entrypoint built on TRL.

Validates ``chosen`` and ``rejected`` pairs strictly. Heavyweight TRL imports
stay inside the training path so local dry-runs remain cheap.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from anima.data.synthesis.reward_records import normalize_text_value
from anima.train.common import (
    PreparedRows,
    fail_on_critical_dropped,
    load_config,
    load_dpo_policy_with_ref_adapter,
    load_policy_model_and_tokenizer,
    optional_int,
    parse_paths,
    read_records,
    record_to_messages,
    render_policy_prompt,
    render_sft_completion,
    render_sft_completion_content,
    supported_config_keys,
    write_json,
)


class DPOPairError(ValueError):
    """Raised when a record cannot be used as a strict DPO pair."""


DPO_COMPLETION_GUARD_KEY = "_dpo_completion_length_audit_passed"


def mark_dpo_completion_length_audit_passed(
    config: dict[str, Any], audit: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    """Record that an exact token audit has enforced DPO length limits."""

    if audit.get("status") != "pass":
        raise ValueError("DPO completion guard requires a passing token audit")
    required_axes = (
        "dpo_prompt_tokens",
        "dpo_chosen_completion_tokens",
        "dpo_rejected_completion_tokens",
    )
    missing = [
        name
        for name in required_axes
        if not isinstance(audit.get(name), Mapping)
        or int((audit.get(name) or {}).get("over_limit", -1)) != 0
    ]
    if missing:
        raise ValueError(
            "DPO token budget guard audit is missing passing axes: " + ", ".join(missing)
        )
    has_total_axis = (
        isinstance(audit.get("dpo_total_tokens"), Mapping)
        and int((audit.get("dpo_total_tokens") or {}).get("over_limit", -1)) == 0
    )
    has_branch_total_axes = all(
        isinstance(audit.get(name), Mapping)
        and int((audit.get(name) or {}).get("over_limit", -1)) == 0
        for name in ("dpo_chosen_total_tokens", "dpo_rejected_total_tokens")
    )
    if not (has_total_axis or has_branch_total_axes):
        raise ValueError(
            "DPO token budget guard audit is missing passing total-length axes: "
            "dpo_total_tokens or dpo_chosen_total_tokens+dpo_rejected_total_tokens"
        )
    axes = list(required_axes)
    if has_total_axis:
        axes.append("dpo_total_tokens")
    else:
        axes.extend(["dpo_chosen_total_tokens", "dpo_rejected_total_tokens"])
    config[DPO_COMPLETION_GUARD_KEY] = {
        "status": "pass",
        "source": source,
        "axes": axes,
    }
    return config[DPO_COMPLETION_GUARD_KEY]


def _dpo_completion_guard(config: Mapping[str, Any], supported: set[str]) -> dict[str, Any]:
    missing_fields: list[str] = []
    if "max_prompt_length" not in supported:
        missing_fields.append("max_prompt_length")
    if not supported.intersection({"max_target_length", "max_completion_length"}):
        missing_fields.append("max_completion_length")
    if not missing_fields:
        return {"status": "pass", "mode": "trl_prompt_completion_length_fields"}
    guard = config.get(DPO_COMPLETION_GUARD_KEY)
    if isinstance(guard, Mapping) and guard.get("status") == "pass":
        return {
            "status": "pass",
            "mode": "token_budget_audit",
            "missing_fields": missing_fields,
            "source": guard.get("source"),
        }
    raise RuntimeError(
        "dpo_config cannot enforce prompt/completion truncation because installed "
        "TRL is missing "
        + ", ".join(missing_fields)
        + ", and no passing DPO token budget audit has been bound"
    )


def _trainer_supports_auto_ref_adapter_copy(trainer_cls: Any | None) -> bool:
    if trainer_cls is None:
        return False
    try:
        import inspect

        source = "\n".join(
            inspect.getsource(member)
            for member in (
                trainer_cls.__init__,
                getattr(trainer_cls, "compute_ref_log_probs", None),
            )
            if member is not None
        )
    except (OSError, TypeError):
        return False
    return 'add_adapter("ref"' in source and "copy_(" in source and 'adapter_name="ref"' in source


def _dpo_reference_adapter_guard(supported: set[str], trainer_cls: Any | None) -> dict[str, Any]:
    adapter_fields = {"model_adapter_name", "ref_adapter_name"}
    if not supported or adapter_fields <= supported:
        return {
            "status": "pass",
            "mode": "dpo_config_adapter_names",
            "policy_adapter_name": "default",
            "reference_adapter_name": "reference",
        }
    if _trainer_supports_auto_ref_adapter_copy(trainer_cls):
        return {
            "status": "pass",
            "mode": "trl_auto_ref_adapter_copy",
            "policy_adapter_name": "default",
            "reference_adapter_name": "ref",
        }
    raise RuntimeError(
        "dpo_config cannot bind the SFT reference adapter because installed TRL "
        "supports neither DPOConfig model_adapter_name/ref_adapter_name nor "
        "automatic frozen ref adapter copying"
    )


def _chunked_selective_log_softmax(logits: Any, index: Any, *, chunk_size: int) -> Any:
    """Memory-bound equivalent of TRL selective_log_softmax for long-sequence DPO."""

    import torch

    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)
    selected_logits = torch.gather(logits, dim=-1, index=index).to(torch.float32)
    logsumexp_values = None
    for chunk in torch.split(logits, int(chunk_size), dim=-1):
        chunk_lse = torch.logsumexp(chunk.to(torch.float32), dim=-1)
        logsumexp_values = (
            chunk_lse if logsumexp_values is None else torch.logaddexp(logsumexp_values, chunk_lse)
        )
    if logsumexp_values is None:
        raise ValueError("logits vocabulary dimension must be non-empty")
    per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)
    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)
    return per_token_logps


def _enable_chunked_selective_log_softmax(chunk_size: int) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunked selective log softmax chunk_size must be positive")
    import trl.trainer.dpo_trainer as dpo_trainer_module
    import trl.trainer.utils as trl_utils

    def chunked(logits: Any, index: Any) -> Any:
        return _chunked_selective_log_softmax(logits, index, chunk_size=chunk_size)

    dpo_trainer_module.selective_log_softmax = chunked
    trl_utils.selective_log_softmax = chunked
    return {"status": "pass", "chunk_size": chunk_size}


def _prepare_single_row_completion_forward(
    model_kwargs: Mapping[str, Any], completion_mask: Any
) -> tuple[dict[str, Any], Any, int]:
    """Trim right padding and request only logits needed by completion log-probs."""

    input_ids = model_kwargs.get("input_ids")
    attention_mask = model_kwargs.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise RuntimeError("completion-only DPO forward requires input_ids and attention_mask")
    if int(input_ids.shape[0]) != 1 or int(attention_mask.shape[0]) != 1:
        raise RuntimeError("completion-only DPO forward requires row_chunk_size=1")

    sequence_length = int(input_ids.shape[1])
    valid_length = int(attention_mask[0].sum().item())
    if valid_length <= 0 or valid_length > sequence_length:
        raise RuntimeError("DPO attention mask has an invalid non-padding length")
    if not bool(attention_mask[0, :valid_length].bool().all().item()) or bool(
        attention_mask[0, valid_length:].bool().any().item()
    ):
        raise RuntimeError("completion-only DPO forward requires contiguous right padding")

    trimmed_kwargs = dict(model_kwargs)
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        value = trimmed_kwargs.get(key)
        if (
            value is not None
            and getattr(value, "ndim", 0) >= 2
            and int(value.shape[0]) == 1
            and int(value.shape[1]) == sequence_length
        ):
            trimmed_kwargs[key] = value[:, :valid_length, ...]
    trimmed_completion_mask = completion_mask[:, :valid_length]
    completion_positions = trimmed_completion_mask[0].bool()
    completion_tokens = int(completion_positions.sum().item())
    if completion_tokens <= 0:
        raise RuntimeError("DPO row has no completion tokens")
    first_completion = valid_length - completion_tokens
    if bool(completion_positions[:first_completion].any().item()) or not bool(
        completion_positions[first_completion:].all().item()
    ):
        raise RuntimeError("completion-only DPO forward requires a contiguous completion suffix")

    # For C completion labels, causal LM loss needs logits from the final C+1
    # input positions and discards the last (non-predictive) logit.
    logits_to_keep = completion_tokens + 1
    trimmed_kwargs["logits_to_keep"] = logits_to_keep
    return trimmed_kwargs, trimmed_completion_mask, logits_to_keep


def _enable_split_dpo_forward_loss(
    row_chunk_size: int, *, completion_only_logits: bool = False
) -> dict[str, Any]:
    """Patch TRL DPO loss to run policy/reference forwards in row chunks."""

    if row_chunk_size <= 0:
        raise ValueError("split DPO forward row_chunk_size must be positive")
    if completion_only_logits and row_chunk_size != 1:
        raise ValueError("completion-only DPO logits require row_chunk_size=1")

    import trl.trainer.dpo_trainer as dpo_trainer_module

    original_compute_loss = dpo_trainer_module.DPOTrainer._compute_loss
    torch = dpo_trainer_module.torch
    F = dpo_trainer_module.F

    def _divergence_name(value: Any) -> str:
        return str(getattr(value, "value", value))

    def _slice_batch(
        mapping: Mapping[str, Any], *, batch_size: int, start: int, end: int
    ) -> dict[str, Any]:
        chunk: dict[str, Any] = {}
        for key, value in mapping.items():
            if (
                hasattr(value, "shape")
                and getattr(value, "ndim", 0) > 0
                and int(value.shape[0]) == batch_size
            ):
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        return chunk

    def _chunk_logps(
        trainer: Any,
        active_model: Any,
        model_kwargs: Mapping[str, Any],
        completion_mask: Any,
    ) -> Any:
        active_kwargs = dict(model_kwargs)
        active_completion_mask = completion_mask
        if completion_only_logits:
            active_kwargs, active_completion_mask, _ = _prepare_single_row_completion_forward(
                active_kwargs, active_completion_mask
            )
        outputs = active_model(**active_kwargs)
        shift_logits = outputs.logits[..., :-1, :]
        target_tokens = int(shift_logits.shape[-2])
        shift_labels = active_kwargs["input_ids"][..., -target_tokens:]
        shift_completion_mask = active_completion_mask[..., -target_tokens:]
        per_token_logps = dpo_trainer_module.selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        return per_token_logps.sum(dim=1)

    def split_compute_loss(
        trainer: Any, model: Any, inputs: Mapping[str, Any], return_outputs: bool
    ):
        if return_outputs:
            raise RuntimeError("return_outputs=True is not supported with split DPO forward loss")
        unsupported = (
            _divergence_name(trainer.f_divergence_type) != "reverse_kl"
            or trainer.ld_alpha is not None
            or trainer.use_weighting
            or trainer.aux_loss_enabled
            or list(trainer.loss_types) != ["sigmoid"]
        )
        if unsupported:
            return original_compute_loss(trainer, model, inputs, return_outputs)

        mode = "train" if trainer.model.training else "eval"
        non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {key: value for key, value in inputs.items() if key not in non_model_keys}
        model_kwargs["use_cache"] = False
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        batch_size = int(input_ids.shape[0])
        if batch_size % 2 != 0:
            raise RuntimeError("split DPO forward expects an even chosen/rejected batch")

        logps_parts = []
        for start in range(0, batch_size, row_chunk_size):
            end = min(start + row_chunk_size, batch_size)
            chunk_kwargs = _slice_batch(model_kwargs, batch_size=batch_size, start=start, end=end)
            chunk_mask = completion_mask[start:end]
            logps_parts.append(_chunk_logps(trainer, model, chunk_kwargs, chunk_mask))
        logps = torch.cat(logps_parts, dim=0)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        if trainer.precompute_ref_logps:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
        else:
            ref_parts = []
            with (
                torch.no_grad(),
                dpo_trainer_module.disable_gradient_checkpointing(
                    trainer.model, trainer.args.gradient_checkpointing_kwargs
                ),
            ):
                if dpo_trainer_module.is_peft_model(model) and trainer.ref_model is None:
                    unwrapped = trainer.accelerator.unwrap_model(model)
                    adapter_name = "ref" if "ref" in unwrapped.peft_config else None
                    with dpo_trainer_module.use_adapter(unwrapped, adapter_name=adapter_name):
                        for start in range(0, batch_size, row_chunk_size):
                            end = min(start + row_chunk_size, batch_size)
                            chunk_kwargs = _slice_batch(
                                model_kwargs, batch_size=batch_size, start=start, end=end
                            )
                            chunk_mask = completion_mask[start:end]
                            ref_parts.append(
                                _chunk_logps(trainer, trainer.model, chunk_kwargs, chunk_mask)
                            )
                else:
                    ref_model = trainer.ref_model
                    for start in range(0, batch_size, row_chunk_size):
                        end = min(start + row_chunk_size, batch_size)
                        chunk_kwargs = _slice_batch(
                            model_kwargs, batch_size=batch_size, start=start, end=end
                        )
                        chunk_mask = completion_mask[start:end]
                        ref_parts.append(_chunk_logps(trainer, ref_model, chunk_kwargs, chunk_mask))
            ref_logps = torch.cat(ref_parts, dim=0)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_score = chosen_logratios - rejected_logratios
        per_sequence_loss = -F.logsigmoid(trainer.beta * delta_score)
        loss_weight = float(trainer.loss_weights[0]) if trainer.loss_weights else 1.0
        loss = per_sequence_loss.mean() * loss_weight

        if mode == "train":
            num_tokens = (
                trainer.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            )
            trainer._total_train_tokens += num_tokens
        trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]

        chosen_rewards = trainer.beta * chosen_logratios.detach()
        rejected_rewards = trainer.beta * rejected_logratios.detach()
        trainer._metrics[mode]["rewards/chosen"].append(
            trainer.accelerator.gather(chosen_rewards).mean().item()
        )
        trainer._metrics[mode]["rewards/rejected"].append(
            trainer.accelerator.gather(rejected_rewards).mean().item()
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        trainer._metrics[mode]["rewards/accuracies"].append(
            trainer.accelerator.gather(reward_accuracies).mean().item()
        )
        margins = chosen_rewards - rejected_rewards
        trainer._metrics[mode]["rewards/margins"].append(
            trainer.accelerator.gather(margins).mean().item()
        )
        trainer._metrics[mode]["logps/chosen"].append(
            trainer.accelerator.gather(chosen_logps.detach()).mean().item()
        )
        trainer._metrics[mode]["logps/rejected"].append(
            trainer.accelerator.gather(rejected_logps.detach()).mean().item()
        )
        trainer._metrics[mode]["split_dpo_forward/row_chunk_size"].append(float(row_chunk_size))
        return loss

    dpo_trainer_module.DPOTrainer._compute_loss = split_compute_loss
    return {
        "status": "pass",
        "row_chunk_size": row_chunk_size,
        "mode": "split_policy_and_reference_rows",
        "supported_loss": "sigmoid_reverse_kl",
        "completion_only_logits": completion_only_logits,
    }


def _enable_in_memory_precompute_ref_logps() -> dict[str, Any]:
    """Patch TRL ref-logp precompute to avoid short-lived Arrow temp files."""

    import trl.trainer.dpo_trainer as dpo_trainer_module

    current = dpo_trainer_module.DPOTrainer._precompute_ref_logps
    if bool(getattr(current, "__anima_in_memory_ref_logps__", False)):
        return {"status": "pass", "mode": "already_enabled"}

    original_precompute_ref_logps = current

    def _add_column(dataset: Any, name: str, values: list[float]) -> Any:
        try:
            return dataset.add_column(name, values)
        except TypeError:
            return dataset.add_column(name=name, column=values)

    def _set_eval(model: Any) -> bool | None:
        if model is None or not hasattr(model, "training"):
            return None
        was_training = bool(model.training)
        if hasattr(model, "eval"):
            model.eval()
        return was_training

    def _restore_training(model: Any, was_training: bool | None) -> None:
        if model is not None and was_training is True and hasattr(model, "train"):
            model.train()

    def in_memory_precompute_ref_logps(
        trainer: Any,
        dataset: Any,
        dataset_name: str,
        batch_size: int,
    ) -> Any:
        column_names = set(getattr(dataset, "column_names", []) or [])
        if {"ref_chosen_logps", "ref_rejected_logps"} <= column_names:
            return dataset

        import torch
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        effective_batch_size = int(batch_size)
        if effective_batch_size <= 0:
            raise ValueError("DPO reference precompute batch_size must be positive")

        args = getattr(trainer, "args", None)
        dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            collate_fn=getattr(trainer, "data_collator"),
            num_workers=int(getattr(args, "dataloader_num_workers", 0)),
            pin_memory=bool(getattr(args, "dataloader_pin_memory", False)),
        )
        accelerator = getattr(trainer, "accelerator")
        dataloader = accelerator.prepare(dataloader)
        gather_for_metrics = getattr(accelerator, "gather_for_metrics", None)
        if gather_for_metrics is None:
            gather_for_metrics = accelerator.gather

        chosen_parts = []
        rejected_parts = []
        model_was_training = _set_eval(getattr(trainer, "model", None))
        ref_was_training = _set_eval(getattr(trainer, "ref_model", None))
        try:
            iterator = tqdm(
                dataloader,
                desc=f"Caching reference log probs for {dataset_name} dataset",
                disable=bool(getattr(args, "disable_tqdm", False)),
            )
            with torch.no_grad():
                for padded_batch in iterator:
                    ref_chosen_logp, ref_rejected_logp = trainer.compute_ref_log_probs(padded_batch)
                    gathered = gather_for_metrics((ref_chosen_logp, ref_rejected_logp))
                    gathered_chosen, gathered_rejected = gathered
                    chosen_parts.append(gathered_chosen.detach().float().cpu())
                    rejected_parts.append(gathered_rejected.detach().float().cpu())
        finally:
            _restore_training(getattr(trainer, "model", None), model_was_training)
            _restore_training(getattr(trainer, "ref_model", None), ref_was_training)

        if not chosen_parts or not rejected_parts:
            raise ValueError("DPO reference precompute received an empty dataset")

        expected_rows = len(dataset)
        ref_chosen_logps = torch.cat(chosen_parts, dim=0)[:expected_rows].tolist()
        ref_rejected_logps = torch.cat(rejected_parts, dim=0)[:expected_rows].tolist()
        if len(ref_chosen_logps) != expected_rows or len(ref_rejected_logps) != expected_rows:
            raise RuntimeError(
                "DPO reference precompute produced a row-count mismatch: "
                f"expected {expected_rows}, got "
                f"{len(ref_chosen_logps)}/{len(ref_rejected_logps)}"
            )

        dataset = _add_column(dataset, "ref_chosen_logps", ref_chosen_logps)
        dataset = _add_column(dataset, "ref_rejected_logps", ref_rejected_logps)
        return dataset

    in_memory_precompute_ref_logps.__anima_in_memory_ref_logps__ = True
    in_memory_precompute_ref_logps.__anima_original_precompute_ref_logps__ = (
        original_precompute_ref_logps
    )
    dpo_trainer_module.DPOTrainer._precompute_ref_logps = in_memory_precompute_ref_logps
    return {"status": "pass", "mode": "in_memory_dataset_columns"}


def _remove_saved_reference_adapter_dirs(output_dir: Path) -> list[str]:
    """Keep the output root unambiguously loadable as the trained policy adapter."""

    removed: list[str] = []
    for name in ("reference", "ref"):
        candidate = output_dir / name
        if candidate.is_dir():
            shutil.rmtree(candidate)
            removed.append(name)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo_qwen3b.yaml"))
    parser.add_argument(
        "--dpo-jsonl", action="append", help="Override/add DPO pair JSON/JSONL input path."
    )
    parser.add_argument("--output-dir", help="Override adapter output directory.")
    parser.add_argument("--sft-adapter", help="Train from an existing SFT LoRA adapter.")
    parser.add_argument("--max-records", type=int, help="Cap records for smoke runs.")
    parser.add_argument(
        "--dpo-schema",
        choices=("legacy", "chat_template"),
        help="Override config.dpo_schema. chat_template uses TRL conversational preference rows.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Normalize pairs and print sample rows only."
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dpo_jsonl:
        config["train_files"] = args.dpo_jsonl
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.sft_adapter:
        config["adapter_path"] = args.sft_adapter
    if args.max_records is not None:
        config["max_records"] = args.max_records
    if args.dpo_schema:
        config["dpo_schema"] = args.dpo_schema

    _assert_beta_positive(config)

    train_files = _dpo_input_paths(config)
    if not train_files:
        raise SystemExit("DPO needs at least one pair file via config.train_files or --dpo-jsonl")

    dpo_schema = str(config.get("dpo_schema", "legacy"))
    prepared = build_dpo_rows(
        train_files,
        max_records=optional_int(config.get("max_records")),
        dpo_schema=dpo_schema,
    )
    if not prepared.rows:
        raise SystemExit(f"No usable DPO pairs found in {[str(path) for path in train_files]}")

    pair_summary = summarize_dpo_rows(prepared.rows)
    adapter_path = str(config.get("adapter_path") or "").strip() or None
    summary = {
        "stage": "dpo",
        "dpo_schema": dpo_schema,
        "config": str(args.config),
        "input_files": [str(path) for path in prepared.input_files],
        "rows": len(prepared.rows),
        "skipped": prepared.skipped,
        "output_dir": str(config.get("output_dir")),
        "adapter_path": str(config.get("adapter_path") or ""),
        "beta": float(config.get("beta", 0.1)),
        "loss_type": str(config.get("loss_type", "sigmoid")),
        "reward_independent": True,
        "target_transform": "explicit_character_r1_target_rendering",
        "reference_policy": reference_policy_semantics(adapter_path),
        "reference_policy_note": reference_policy_note(adapter_path),
        "note": "DPO comparison arm: matched prompts, chosen=reference answer, rejected=synthetically degraded.",
        **pair_summary,
    }

    if args.dry_run:
        print(
            json.dumps(
                {**summary, "sample": prepared.rows[: int(config.get("dry_run_samples", 2))]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    train(
        config, prepared.rows, resume_from_checkpoint=args.resume_from_checkpoint, summary=summary
    )
    return 0


def build_dpo_rows(
    paths: Sequence[Path],
    *,
    max_records: int | None = None,
    dpo_schema: str = "legacy",
) -> PreparedRows:
    """Build strict TRL DPO rows from pair JSONL or reward records."""

    if dpo_schema not in {"legacy", "chat_template"}:
        raise ValueError(f"unsupported dpo_schema={dpo_schema!r}; expected legacy or chat_template")
    prepared = read_records(paths, max_records=max_records)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(prepared.rows):
        try:
            rows.append(_normalize_dpo_record(record, index=index, dpo_schema=dpo_schema))
        except DPOPairError as exc:
            record_id = str(record.get("id") or f"record_{index:06d}")
            raise ValueError(f"DPO pair validation failed for {record_id}: {exc}") from exc
    return PreparedRows(rows=rows, skipped=0, input_files=prepared.input_files)


def train(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    resume_from_checkpoint: str | None,
    summary: dict[str, Any],
    checkpoint_callback: Any = None,
) -> None:
    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    chunked_logsoftmax_guard = None
    if bool(config.get("use_chunked_selective_log_softmax", False)):
        chunked_logsoftmax_guard = _enable_chunked_selective_log_softmax(
            int(config.get("selective_log_softmax_chunk_size", 8192))
        )
    split_forward_loss_guard = None
    if bool(config.get("split_dpo_forward_loss", False)):
        split_forward_loss_guard = _enable_split_dpo_forward_loss(
            int(config.get("split_dpo_forward_row_chunk_size", 1)),
            completion_only_logits=bool(config.get("completion_only_logits", False)),
        )
    in_memory_ref_logps_guard = None
    if bool(config.get("precompute_ref_log_probs", False)):
        in_memory_ref_logps_guard = _enable_in_memory_precompute_ref_logps()

    output_dir = Path(str(config.get("output_dir", "work/models/dpo")))
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = str(config.get("adapter_path") or "").strip() or None
    reference_adapter_guard = (
        _dpo_reference_adapter_guard(supported_config_keys(DPOConfig), DPOTrainer)
        if adapter_path
        else {"status": "not_required", "mode": "base_model_adapter_disabled"}
    )
    if adapter_path:
        model, tokenizer, peft_config = load_dpo_policy_with_ref_adapter(
            config,
            padding_side=str(config.get("padding_side", "right")),
            adapter_path=adapter_path,
            reference_adapter_name=(
                "reference"
                if reference_adapter_guard["mode"] == "dpo_config_adapter_names"
                else None
            ),
        )
    else:
        model, tokenizer, peft_config = load_policy_model_and_tokenizer(
            config,
            padding_side=str(config.get("padding_side", "right")),
            adapter_path=None,
        )
    if str(config.get("dpo_schema", "legacy")) == "chat_template" and not getattr(
        tokenizer, "chat_template", None
    ):
        raise RuntimeError(
            "dpo_schema=chat_template requires a tokenizer with a non-empty chat_template"
        )
    dataset = Dataset.from_list(rows)
    training_args = _build_dpo_config(
        DPOConfig,
        config,
        output_dir,
        use_ref_adapter=bool(adapter_path),
        dpo_trainer_cls=DPOTrainer,
    )

    trainer_kwargs = _dpo_trainer_kwargs(
        DPOTrainer,
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    trainer = DPOTrainer(**trainer_kwargs)
    if checkpoint_callback is not None:
        trainer.add_callback(checkpoint_callback)
    adapter_trainability_audit = (
        audit_dpo_adapter_trainability(trainer.model) if adapter_path else None
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # PEFT/TRL versions use either "reference" or "ref" for the frozen copy.
    # Leaving either one in the deliverable makes "output root = policy adapter"
    # ambiguous, so retain only the trained root adapter.
    removed_reference_adapter_dirs = _remove_saved_reference_adapter_dirs(output_dir)

    metrics = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    write_json(
        output_dir / "anima_dpo_summary.json",
        {
            **summary,
            "train_metrics": metrics,
            "adapter_trainability_audit": adapter_trainability_audit,
            "reference_adapter_guard": reference_adapter_guard,
            "chunked_selective_log_softmax_guard": chunked_logsoftmax_guard,
            "split_dpo_forward_loss_guard": split_forward_loss_guard,
            "in_memory_ref_logps_guard": in_memory_ref_logps_guard,
            "removed_reference_adapter_dirs": removed_reference_adapter_dirs,
            "adapter_path": adapter_path,
            "model_name_or_path": str(config.get("model_name_or_path")),
            "max_length": int(config.get("max_length", 1024)),
            "max_prompt_length": int(config.get("max_prompt_length", 768)),
            "max_target_length": int(
                config.get("max_target_length", config.get("max_completion_length", 256))
            ),
            "precompute_ref_log_probs": bool(config.get("precompute_ref_log_probs", False)),
            "cuda_max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
        },
    )


def audit_dpo_adapter_trainability(model: Any) -> dict[str, int | str | list[str]]:
    """Prove that DPO updates only the policy adapter and freezes its reference copy."""

    default_parameters: list[tuple[str, Any]] = []
    reference_parameters: list[tuple[str, Any]] = []
    unexpected_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        dotted = f".{name}."
        if ".default." in dotted:
            default_parameters.append((name, parameter))
        elif ".reference." in dotted or ".ref." in dotted:
            reference_parameters.append((name, parameter))
        elif bool(parameter.requires_grad):
            unexpected_trainable.append(name)
    if not default_parameters or not reference_parameters:
        raise RuntimeError("DPO model must contain both default and reference adapter parameters")
    frozen_default = [
        name for name, parameter in default_parameters if not bool(parameter.requires_grad)
    ]
    trainable_reference = [
        name for name, parameter in reference_parameters if bool(parameter.requires_grad)
    ]
    if frozen_default:
        raise RuntimeError(f"DPO default adapter contains frozen parameters: {frozen_default[:8]}")
    if trainable_reference:
        raise RuntimeError(
            f"DPO reference adapter contains trainable parameters: {trainable_reference[:8]}"
        )
    if unexpected_trainable:
        raise RuntimeError(
            f"DPO has trainable parameters outside default adapter: {unexpected_trainable[:8]}"
        )
    return {
        "status": "pass",
        "default_parameters": len(default_parameters),
        "trainable_default_parameters": len(default_parameters),
        "reference_parameters": len(reference_parameters),
        "trainable_reference_parameters": 0,
        "reference_adapter_names": ["reference", "ref"],
        "unexpected_trainable_parameters": 0,
    }


def _build_dpo_config(
    DPOConfig: Any,
    config: dict[str, Any],
    output_dir: Path,
    *,
    use_ref_adapter: bool = False,
    dpo_trainer_cls: Any | None = None,
) -> Any:
    requested = {
        "output_dir": str(output_dir),
        "run_name": str(config.get("run_name", output_dir.name)),
        "max_steps": int(config.get("max_steps", 80)),
        "num_train_epochs": float(config.get("num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 4)),
        # LoRA-scale learning rate aligned with established QLoRA DPO recipes.
        "learning_rate": float(config.get("learning_rate", 5e-6)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(config.get("warmup_ratio", 0.1)),
        "beta": float(config.get("beta", 0.1)),
        "loss_type": str(config.get("loss_type", "sigmoid")),
        "label_smoothing": float(config.get("label_smoothing", 0.0)),
        "max_length": int(config.get("max_length", 1024)),
        "max_prompt_length": int(config.get("max_prompt_length", 768)),
        "max_target_length": int(
            config.get("max_target_length", config.get("max_completion_length", 256))
        ),
        "max_completion_length": int(
            config.get("max_completion_length", config.get("max_target_length", 256))
        ),
        "truncation_mode": str(config.get("truncation_mode", "keep_end")),
        "precompute_ref_log_probs": bool(config.get("precompute_ref_log_probs", False)),
        "generate_during_eval": bool(config.get("generate_during_eval", False)),
        "disable_dropout": bool(config.get("disable_dropout", True)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_steps": int(config.get("save_steps", 40)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
        "tf32": bool(config.get("tf32", True)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "optim": str(config.get("optim", "paged_adamw_8bit")),
        "report_to": config.get("report_to", []),
        "seed": int(config.get("seed", 42)),
        "remove_unused_columns": bool(config.get("remove_unused_columns", False)),
    }
    optional_requested: dict[str, Any] = {}
    if "activation_offloading" in config:
        optional_requested["activation_offloading"] = bool(
            config.get("activation_offloading", False)
        )
    if "gradient_checkpointing_kwargs" in config:
        value = config.get("gradient_checkpointing_kwargs")
        if not isinstance(value, Mapping):
            raise ValueError("gradient_checkpointing_kwargs must be a mapping")
        optional_requested["gradient_checkpointing_kwargs"] = dict(value)
    if "torch_empty_cache_steps" in config:
        value = optional_int(config.get("torch_empty_cache_steps"))
        if value is not None:
            optional_requested["torch_empty_cache_steps"] = value
    if "use_cache" in config:
        optional_requested["use_cache"] = bool(config.get("use_cache", False))
    if "skip_memory_metrics" in config:
        optional_requested["skip_memory_metrics"] = bool(config.get("skip_memory_metrics", True))
    requested.update(optional_requested)

    strict_semantics = bool(config.get("strict_config_semantics", False))
    critical = {"beta", "loss_type", "max_length", "max_prompt_length"}
    if strict_semantics:
        critical |= {"label_smoothing", "truncation_mode", "disable_dropout"}
    if use_ref_adapter:
        # Bind the policy and frozen SFT reference explicitly. If this TRL
        # version lacks these fields, its trainer must provide equivalent
        # reference-adapter copying semantics.
        requested["model_adapter_name"] = str(config.get("model_adapter_name", "default"))
        requested["ref_adapter_name"] = str(config.get("ref_adapter_name", "reference"))
        critical |= {"model_adapter_name", "ref_adapter_name"}

    supported = supported_config_keys(DPOConfig)
    if not supported:
        return DPOConfig(**requested)
    reference_guard = (
        _dpo_reference_adapter_guard(supported, dpo_trainer_cls)
        if use_ref_adapter
        else {"status": "not_required", "mode": "base_model_adapter_disabled"}
    )
    if reference_guard.get("mode") == "trl_auto_ref_adapter_copy":
        critical.discard("model_adapter_name")
        critical.discard("ref_adapter_name")
    completion_guard = (
        _dpo_completion_guard(config, supported)
        if strict_semantics
        else {"status": "not_required", "mode": "non_strict"}
    )
    if (
        strict_semantics
        and completion_guard.get("mode") == "token_budget_audit"
        and "max_prompt_length" in completion_guard.get("missing_fields", ())
    ):
        critical.discard("max_prompt_length")

    dropped = sorted(set(requested) - supported)
    if dropped:
        fail_on_critical_dropped(
            stage="dpo_config",
            dropped=dropped,
            critical=critical,
        )
        print(
            json.dumps(
                {
                    "stage": "dpo_config",
                    "dropped_unsupported_keys": dropped,
                    "completion_length_guard": completion_guard,
                    "reference_adapter_guard": reference_guard,
                    "note": "Installed TRL DPOConfig does not expose these fields; training continues without them.",
                },
                ensure_ascii=False,
            )
        )
    return DPOConfig(**{key: value for key, value in requested.items() if key in supported})


def _dpo_trainer_kwargs(
    trainer_cls: Any,
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    tokenizer: Any,
    peft_config: Any,
) -> dict[str, Any]:
    requested = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }

    try:
        import inspect

        params = inspect.signature(trainer_cls.__init__).parameters
    except (TypeError, ValueError):
        return requested

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return requested

    if "processing_class" not in params and "tokenizer" in params:
        requested["tokenizer"] = requested.pop("processing_class")
    return {key: value for key, value in requested.items() if key in params}


def reference_policy_semantics(adapter_path: str | None) -> str:
    """Name which reference policy this run anchors to, for the summary JSON."""

    if adapter_path:
        return "sft_adapter_via_ref_adapter_name"
    return "base_model_adapter_disabled"


def reference_policy_note(adapter_path: str | None) -> str:
    if adapter_path:
        return (
            "The reference policy is the frozen SFT adapter; policy and reference are "
            "bound through explicit adapter names or TRL's reference-copy mechanism."
        )
    return "Without an SFT adapter, the reference policy is the base model with adapters disabled."


def summarize_dpo_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chosen_lengths = [len(_completion_summary_text(row["chosen"])) for row in rows]
    rejected_lengths = [len(_completion_summary_text(row["rejected"])) for row in rows]
    strategies = Counter(str(row.get("rejected_strategy") or "unknown") for row in rows)
    return {
        "chosen_chars": _length_summary(chosen_lengths),
        "rejected_chars": _length_summary(rejected_lengths),
        "rejected_strategies": dict(sorted(strategies.items())),
    }


def _normalize_dpo_record(
    record: Mapping[str, Any], *, index: int, dpo_schema: str
) -> dict[str, Any]:
    chosen_raw = _first_text(record, ("chosen", "reference_answer"))
    rejected_raw = _first_text(record, ("rejected", "rejected_answer"))
    if not chosen_raw:
        raise DPOPairError("chosen/reference_answer must be non-empty")
    if not rejected_raw:
        raise DPOPairError("rejected/rejected_answer must be non-empty")
    if _canonical_pair_text(chosen_raw) == _canonical_pair_text(rejected_raw):
        raise DPOPairError("chosen and rejected must differ")

    if dpo_schema == "chat_template":
        prompt: str | list[dict[str, str]] = record_to_messages(record)
        prompt_ok = bool(prompt)
        chosen: str | list[dict[str, str]] = [
            {
                "role": "assistant",
                "content": _ensure_formatted_completion_content(record, chosen_raw),
            }
        ]
        rejected: str | list[dict[str, str]] = [
            {
                "role": "assistant",
                "content": _ensure_formatted_completion_content(record, rejected_raw),
            }
        ]
    else:
        prompt = _record_prompt(record)
        prompt_ok = bool(prompt.strip())
        chosen = _ensure_formatted_completion(record, chosen_raw)
        rejected = _ensure_formatted_completion(record, rejected_raw)

    if not prompt_ok:
        raise DPOPairError("prompt must be non-empty")

    row: dict[str, Any] = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "id": str(record.get("id") or f"dpo_{index:06d}"),
        "source_record_id": str(
            record.get("source_record_id") or record.get("id") or f"dpo_{index:06d}"
        ),
        "character": _first_text(record, ("character", "role", "name", "character_name"))
        or "unknown",
        "source_work": _first_text(record, ("source_work", "work", "book", "movie")) or "unknown",
    }

    rejected_strategy = _rejected_strategy(record)
    if rejected_strategy:
        row["rejected_strategy"] = rejected_strategy
    return row


def _record_prompt(record: Mapping[str, Any]) -> str:
    prompt = normalize_text_value(record.get("prompt"))
    if prompt:
        return prompt
    return render_policy_prompt(record)


def _ensure_formatted_completion(record: Mapping[str, Any], answer: str) -> str:
    """Keep DPO targets consistent with the Character-R1 output contract."""

    if "<think>" in answer and "\\boxed{" in answer:
        return answer
    return render_sft_completion(record, answer=answer)


def _ensure_formatted_completion_content(record: Mapping[str, Any], answer: str) -> str:
    """Return assistant content for conversational DPO targets."""

    if "<think>" in answer and "\\boxed{" in answer:
        return _strip_known_eos(answer)
    return render_sft_completion_content(record, answer=answer)


def _strip_known_eos(text: str) -> str:
    stripped = text.strip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        if stripped.endswith(token):
            stripped = stripped[: -len(token)].rstrip()
    return stripped


def _completion_summary_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return normalize_text_value(value.get("content")) or str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(_completion_summary_text(item) for item in value)
    return str(value)


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key not in record:
            continue
        text = normalize_text_value(record[key])
        if text:
            return text
    return ""


def _rejected_strategy(record: Mapping[str, Any]) -> str:
    text = normalize_text_value(record.get("rejected_strategy"))
    if text:
        return text
    synth_meta = record.get("synth_meta")
    if isinstance(synth_meta, Mapping):
        return normalize_text_value(synth_meta.get("rejected_strategy"))
    return ""


def _canonical_pair_text(text: str) -> str:
    return "".join(text.split())


def _length_summary(lengths: Sequence[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)}


def _dpo_input_paths(config: Mapping[str, Any]) -> list[Path]:
    value = config.get("train_files")
    if value is None:
        value = config.get("dpo_files")
    if value is None:
        value = config.get("dpo_jsonl")
    return parse_paths(value)


def _assert_beta_positive(config: Mapping[str, Any]) -> None:
    beta = float(config.get("beta", 0.1))
    if beta <= 0.0:
        raise SystemExit(
            f"Refusing DPO with beta={beta}; beta>0 is required for the KL/reference anchor."
        )


if __name__ == "__main__":
    raise SystemExit(main())
