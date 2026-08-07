"""Shared JSONL read/write helpers used across the data, eval, and train layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def iter_jsonl(path: Path, *, encoding: str = "utf-8") -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
            yield line_no, record


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, sort_keys: bool = True) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys))
            handle.write("\n")
            count += 1
    return count
