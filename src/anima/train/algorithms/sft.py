"""Supervised fine-tuning entrypoint built on TRL and PEFT.

Use ``--dry-run`` on any candidate JSONL before touching the GPU; it prints the
normalized prompt/completion rows without importing torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from anima.train.common import (
    ChatTemplateError,
    build_sft_chat_rows,
    build_sft_rows,
    fail_on_critical_dropped,
    load_config,
    load_policy_model_and_tokenizer,
    optional_int,
    parse_paths,
    supported_config_keys,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_qwen3b.yaml"))
    parser.add_argument(
        "--train-jsonl", action="append", help="Override/add SFT JSON/JSONL input path."
    )
    parser.add_argument("--output-dir", help="Override adapter output directory.")
    parser.add_argument("--max-records", type=int, help="Cap records for smoke runs.")
    parser.add_argument(
        "--sft-schema",
        choices=("legacy", "chat_template"),
        help="Override config.sft_schema. chat_template uses TRL conversational prompt/completion rows.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Normalize data and print sample rows only."
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.train_jsonl:
        config["train_files"] = args.train_jsonl
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_records is not None:
        config["max_records"] = args.max_records
    if args.sft_schema:
        config["sft_schema"] = args.sft_schema

    train_files = parse_paths(config.get("train_files"))
    if not train_files:
        raise SystemExit(
            "SFT needs at least one train file via config.train_files or --train-jsonl"
        )

    sft_schema = str(config.get("sft_schema", "legacy"))
    if sft_schema == "chat_template":
        prepared = build_sft_chat_rows(
            train_files,
            max_records=optional_int(config.get("max_records")),
            require_gold_format=bool(config.get("require_gold_format", False)),
        )
    elif sft_schema == "legacy":
        prepared = build_sft_rows(
            train_files,
            max_records=optional_int(config.get("max_records")),
            require_gold_format=bool(config.get("require_gold_format", False)),
        )
    else:
        raise SystemExit(f"unsupported sft_schema={sft_schema!r}; expected legacy or chat_template")
    if not prepared.rows:
        raise SystemExit(f"No usable SFT rows found in {[str(path) for path in train_files]}")

    summary = {
        "stage": "sft",
        "sft_schema": sft_schema,
        "config": str(args.config),
        "input_files": [str(path) for path in prepared.input_files],
        "rows": len(prepared.rows),
        "skipped": prepared.skipped,
        "output_dir": str(config.get("output_dir")),
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


def train(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    resume_from_checkpoint: str | None,
    summary: dict[str, Any],
) -> None:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(str(config.get("output_dir", "work/models/sft")))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, peft_config = load_policy_model_and_tokenizer(config, padding_side="right")
    if str(config.get("sft_schema", "legacy")) == "chat_template" and not getattr(
        tokenizer, "chat_template", None
    ):
        raise ChatTemplateError(
            "sft_schema=chat_template requires a tokenizer with a non-empty chat_template"
        )
    dataset = Dataset.from_list(rows)

    training_args = _build_sft_config(SFTConfig, config, output_dir)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(train_result.metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    write_json(
        output_dir / "anima_sft_summary.json",
        {
            **summary,
            "train_metrics": metrics,
            "model_name_or_path": str(config.get("model_name_or_path")),
            "max_seq_length": int(config.get("max_seq_length", config.get("max_length", 1024))),
            "completion_only_loss": bool(config.get("completion_only_loss", True)),
            "assistant_only_loss": bool(config.get("assistant_only_loss", False)),
        },
    )


def _build_sft_config(SFTConfig: Any, config: dict[str, Any], output_dir: Path) -> Any:
    requested = {
        "output_dir": str(output_dir),
        "run_name": str(config.get("run_name", output_dir.name)),
        "max_steps": int(config.get("max_steps", 50)),
        "num_train_epochs": float(config.get("num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_steps": int(config.get("save_steps", 50)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": bool(config.get("bf16", True)),
        "fp16": bool(config.get("fp16", False)),
        "tf32": bool(config.get("tf32", True)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "optim": str(config.get("optim", "paged_adamw_8bit")),
        "max_length": int(config.get("max_seq_length", config.get("max_length", 1024))),
        "eos_token": config.get("eos_token"),
        "packing": bool(config.get("packing", False)),
        "completion_only_loss": bool(config.get("completion_only_loss", True)),
        "assistant_only_loss": bool(config.get("assistant_only_loss", False)),
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

    supported = supported_config_keys(SFTConfig)
    if not supported:
        return SFTConfig(**requested)

    dropped = sorted(set(requested) - supported)
    if dropped:
        fail_on_critical_dropped(
            stage="sft_config",
            dropped=dropped,
            critical={"completion_only_loss", "max_length", "eos_token"},
        )
        print(
            json.dumps(
                {
                    "stage": "sft_config",
                    "dropped_unsupported_keys": dropped,
                    "note": "Installed TRL SFTConfig does not expose these fields; training continues without them.",
                },
                ensure_ascii=False,
            )
        )
    return SFTConfig(**{key: value for key, value in requested.items() if key in supported})


if __name__ == "__main__":
    raise SystemExit(main())
