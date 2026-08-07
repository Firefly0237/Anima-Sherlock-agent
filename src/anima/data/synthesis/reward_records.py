"""Build synthesis seed records from public character data.

Normalizes public raw exports into seed JSONL so the offline synthesis of
``gold_focus`` / ``gold_focus_attr`` / ``reference_answer`` has stable inputs.
Already-synthesized fields are kept if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from anima.utils.jsonl import write_jsonl

SUPPORTED_INPUT_SUFFIXES = {".json", ".jsonl", ".parquet"}


def iter_json_objects(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
                else:
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
        return
    if suffix == ".parquet":
        yield from iter_parquet_objects(path)
        return

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(value, dict):
        for key in ("data", "items", "examples", "records"):
            items = value.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        yield item
                return
        yield value


def iter_parquet_objects(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from HF-style parquet shards."""

    # pyarrow 延迟导入：JSON/JSONL 路径保持纯标准库，训练环境本来就带 pyarrow
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "Reading .parquet inputs requires pyarrow. Activate the training env "
            "or install the dataset stack, then rerun this command."
        ) from exc

    table = pq.read_table(path)
    for index, row in enumerate(table.to_pylist()):
        if isinstance(row, dict):
            yield row
        else:
            raise ValueError(f"{path}:{index}: expected parquet row to decode as object")


def first_nonempty(record: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        text = normalize_text_value(value)
        if text:
            return text
    return default


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list | tuple):
        parts = [normalize_text_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("content", "text", "utterance", "answer", "response", "generated"):
            text = normalize_text_value(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def normalize_conversations(record: dict[str, Any]) -> list[dict[str, str]]:
    conversations = record.get("conversations") or record.get("dialogue") or record.get("history")
    if isinstance(conversations, list):
        normalized: list[dict[str, str]] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or turn.get("speaker") or "user").strip() or "user"
            content = normalize_text_value(
                turn.get("content") or turn.get("text") or turn.get("utterance")
            )
            if content:
                normalized.append({"role": role, "content": content})
        if normalized:
            return normalized

    question = first_nonempty(record, ("question", "instruction", "prompt", "query", "input"))
    if question:
        return [{"role": "user", "content": question}]
    return []


def normalize_seed(
    record: dict[str, Any],
    *,
    source: str,
    index: int,
    id_prefix: str,
    split: str,
) -> dict[str, Any]:
    character = first_nonempty(record, ("character", "role", "name", "character_name"), "unknown")
    source_work = first_nonempty(
        record, ("source_work", "work", "book", "movie"), f"{source}/{character}"
    )
    profile = first_nonempty(
        record,
        ("profile", "persona", "description", "character_profile", "system_prompt"),
    )
    if not profile:
        profile = (
            f"{character}角色卡：来自{source_work}。请保持该角色身份、称谓、语气和价值观，"
            "用中文回应，不要以普通助手口吻回答。"
        )
    conversations = normalize_conversations(record)

    seed = {
        "id": str(record.get("id") or f"{id_prefix}_{index:06d}"),
        "character": character,
        "source_work": source_work,
        "character_cluster": int(record.get("character_cluster", -1) or -1),
        "profile": profile,
        "conversations": conversations,
        "source": source,
        "split": split,
    }

    for key in (
        "gold_focus",
        "gold_focus_attr",
        "reference_answer",
        "rejected_answer",
        "synth_meta",
    ):
        if key in record:
            seed[key] = record[key]
    if "reference_answer" not in seed:
        answer = first_nonempty(record, ("answer", "response", "generated", "reference"))
        if answer:
            seed["reference_answer"] = answer
    return seed


def collect_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES)
            )
        else:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, nargs="+", type=Path, help="Raw JSON/JSONL files or dirs."
    )
    parser.add_argument("--output", required=True, type=Path, help="Normalized seed JSONL.")
    parser.add_argument(
        "--source", required=True, help="Source label, e.g. CharacterBench or RoleBench."
    )
    parser.add_argument("--id-prefix", default="seed")
    parser.add_argument("--split", default="reward", choices=("sft", "reward", "eval_heldout"))
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    input_files = collect_input_files(args.input)
    rows: list[dict[str, Any]] = []
    for file_path in input_files:
        for raw in iter_json_objects(file_path):
            rows.append(
                normalize_seed(
                    raw,
                    source=args.source,
                    index=len(rows),
                    id_prefix=args.id_prefix,
                    split=args.split,
                )
            )
            if args.max_records is not None and len(rows) >= args.max_records:
                break
        if args.max_records is not None and len(rows) >= args.max_records:
            break

    count = write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "input_files": len(input_files),
                "output": str(args.output),
                "records": count,
                "source": args.source,
                "split": args.split,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
