"""Parse Character-R1 style reward completions.

The policy is expected to emit a private reasoning block with focus metadata
and a boxed answer:

```
<think>...<focus>Knowledge,Style</focus><focus_attr>...</focus_attr>...</think>
\boxed{...}
```
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from anima.data.common.schemas import FOCUS_LABELS as _SCHEMA_FOCUS_LABELS

FOCUS_LABELS: tuple[str, ...] = tuple(_SCHEMA_FOCUS_LABELS)
FOCUS_LABEL_SET: frozenset[str] = frozenset(FOCUS_LABELS)

_THINK_RE = re.compile(r"<\s*think\b[^>]*>(?P<body>.*?)</\s*think\s*>", re.I | re.S)
_FOCUS_RE = re.compile(r"<\s*focus\b[^>]*>(?P<body>.*?)</\s*focus\s*>", re.I | re.S)
_FOCUS_ATTR_RE = re.compile(
    r"<\s*focus_attr\b[^>]*>(?P<body>.*?)</\s*focus_attr\s*>",
    re.I | re.S,
)
_FOCUS_SPLIT_RE = re.compile(r"[,，;；、/|]+|\s+")
_BOXED_START_RE = re.compile(r"\\boxed\s*\{", re.S)

PROMPT_LEAKAGE_MARKERS: tuple[str, ...] = (
    "Human:",
    "User:",
    "Assistant:",
    "System:",
    "human:",
    "user:",
    "assistant:",
    "system:",
    "角色:",
    "来源作品:",
    "角色卡:",
    "对话上下文:",
    "请严格",
    "不要输出",
    "新题目",
    "<|endoftext|>",
)


@dataclass(frozen=True)
class ParsedCompletion:
    """Structured parse result for one model completion."""

    raw: str
    think: str | None
    focus_labels: tuple[str, ...]
    illegal_focus_labels: tuple[str, ...]
    focus_attr: str | None
    boxed_answer: str | None
    boxed_end_index: int | None
    think_count: int
    focus_count: int
    focus_attr_count: int
    boxed_count: int

    @property
    def has_think(self) -> bool:
        return bool(self.think and self.think.strip())

    @property
    def has_focus(self) -> bool:
        return bool(self.focus_labels)

    @property
    def has_focus_attr(self) -> bool:
        return bool(self.focus_attr and self.focus_attr.strip())

    @property
    def has_boxed_answer(self) -> bool:
        return bool(self.boxed_answer and self.boxed_answer.strip())

    @property
    def has_clean_tail(self) -> bool:
        if self.boxed_end_index is None:
            return False
        return _tail_is_clean(self.raw[self.boxed_end_index :])

    @property
    def is_well_formed(self) -> bool:
        return (
            self.has_think
            and self.think_count == 1
            and self.has_focus
            and self.focus_count == 1
            and not self.illegal_focus_labels
            and self.has_focus_attr
            and self.has_boxed_answer
            and self.focus_attr_count == 1
            and self.boxed_count == 1
            and self.has_clean_tail
            and not output_contract_issues(self)
        )


def completion_to_text(completion: Any) -> str:
    """Coerce TRL/plain/chat completion shapes into text."""

    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "generated_text"):
            value = completion.get(key)
            if isinstance(value, str):
                return value
        return str(completion)
    if isinstance(completion, (list, tuple)):
        parts = [completion_to_text(item) for item in completion]
        return "\n".join(part for part in parts if part)
    return str(completion)


def ensure_completion_list(completions: Any) -> list[Any]:
    """Return a batch list while preserving a single chat-message completion."""

    if isinstance(completions, str) or isinstance(completions, dict) or completions is None:
        return [completions]
    if isinstance(completions, (list, tuple)):
        if _looks_like_one_chat_completion(completions):
            return [completions]
        return list(completions)
    return [completions]


def normalize_focus_labels(labels: Any) -> tuple[str, ...]:
    """Normalize a focus field into legal, unique labels in first-seen order."""

    legal, _illegal = split_focus_labels(labels)
    return legal


def split_focus_labels(labels: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a focus field into ``(legal_labels, illegal_labels)``."""

    candidates: list[str] = []
    if labels is None:
        return (), ()
    if isinstance(labels, str):
        candidates.extend(_split_focus_text(labels))
    elif isinstance(labels, Iterable):
        for label in labels:
            if isinstance(label, str):
                candidates.extend(_split_focus_text(label))
            else:
                candidates.append(str(label).strip())
    else:
        candidates.append(str(labels).strip())

    legal: list[str] = []
    illegal: list[str] = []
    seen_legal: set[str] = set()
    seen_illegal: set[str] = set()
    for label in candidates:
        label = _clean_focus_token(label)
        if not label:
            continue
        if label in FOCUS_LABEL_SET:
            if label not in seen_legal:
                legal.append(label)
                seen_legal.add(label)
        elif label not in seen_illegal:
            illegal.append(label)
            seen_illegal.add(label)
    return tuple(legal), tuple(illegal)


def parse_completion(completion: Any) -> ParsedCompletion:
    """Extract think, focus labels, focus attributes, and boxed answer."""

    raw = completion_to_text(completion)
    think_matches = list(_THINK_RE.finditer(raw))
    think_match = think_matches[0] if think_matches else None
    think = _strip_or_none(think_match.group("body")) if think_match else None
    search_area = think if think is not None else raw

    focus_texts = [match.group("body") for match in _FOCUS_RE.finditer(search_area)]
    legal_focus: list[str] = []
    illegal_focus: list[str] = []
    for focus_text in focus_texts:
        legal, illegal = split_focus_labels(focus_text)
        for label in legal:
            if label not in legal_focus:
                legal_focus.append(label)
        for label in illegal:
            if label not in illegal_focus:
                illegal_focus.append(label)

    attr_matches = list(_FOCUS_ATTR_RE.finditer(search_area))
    attr_match = attr_matches[0] if attr_matches else None
    focus_attr = _strip_or_none(attr_match.group("body")) if attr_match else None
    boxed_starts = list(_BOXED_START_RE.finditer(raw))
    boxed_answer, boxed_end_index = _extract_boxed(raw)

    return ParsedCompletion(
        raw=raw,
        think=think,
        focus_labels=tuple(legal_focus),
        illegal_focus_labels=tuple(illegal_focus),
        focus_attr=focus_attr,
        boxed_answer=boxed_answer,
        boxed_end_index=boxed_end_index,
        think_count=len(think_matches),
        focus_count=len(focus_texts),
        focus_attr_count=len(attr_matches),
        boxed_count=len(boxed_starts),
    )


def parse_completions(completions: Any) -> list[ParsedCompletion]:
    return [parse_completion(completion) for completion in ensure_completion_list(completions)]


def prompt_leakage_markers(completion: Any) -> tuple[str, ...]:
    """Return prompt/header markers leaked into the completion text."""

    raw = (
        completion.raw
        if isinstance(completion, ParsedCompletion)
        else completion_to_text(completion)
    )
    return tuple(marker for marker in PROMPT_LEAKAGE_MARKERS if marker in raw)


def output_contract_issues(completion: Any) -> tuple[str, ...]:
    """Return schema-validity issues for a Character-R1 style completion."""

    parsed = (
        completion if isinstance(completion, ParsedCompletion) else parse_completion(completion)
    )
    issues: list[str] = []
    stripped = parsed.raw.strip()

    if not stripped:
        issues.append("empty_completion")
    elif not stripped.lower().startswith("<think"):
        issues.append("prefix_before_think")
    if not parsed.has_think:
        issues.append("missing_think")
    if parsed.think_count > 1:
        issues.append("multi_think")
    if not parsed.has_focus:
        issues.append("missing_focus")
    if parsed.focus_count > 1:
        issues.append("multi_focus")
    # 非法 focus 标签不判契约失败：契约只管结构，语义扣分在 focus 奖励的 F1 分母里
    if not parsed.has_focus_attr:
        issues.append("missing_focus_attr")
    if parsed.focus_attr_count > 1:
        issues.append("multi_focus_attr")
    if not parsed.has_boxed_answer:
        issues.append("missing_boxed")
    if parsed.boxed_count > 1:
        issues.append("multi_boxed")
    if parsed.boxed_count > 0 and parsed.boxed_end_index is None:
        issues.append("malformed_boxed")
    if parsed.boxed_end_index is not None and not parsed.has_clean_tail:
        issues.append("text_after_first_boxed")
    if prompt_leakage_markers(parsed):
        issues.append("prompt_leakage")

    return tuple(dict.fromkeys(issues))


def is_output_contract_valid(completion: Any) -> bool:
    return not output_contract_issues(completion)


def output_contract_report(completion: Any) -> dict[str, Any]:
    parsed = (
        completion if isinstance(completion, ParsedCompletion) else parse_completion(completion)
    )
    issues = output_contract_issues(parsed)
    markers = prompt_leakage_markers(parsed)
    return {
        "status": "pass" if not issues else "fail",
        "score": 1.0 if not issues else 0.0,
        "issues": list(issues),
        "illegal_focus_labels": list(parsed.illegal_focus_labels),
        "think_count": parsed.think_count,
        "focus_count": parsed.focus_count,
        "focus_attr_count": parsed.focus_attr_count,
        "boxed_count": parsed.boxed_count,
        "has_clean_tail": parsed.has_clean_tail,
        "prompt_leakage": {
            "status": "fail" if markers else "pass",
            "markers": list(markers),
        },
    }


def _looks_like_one_chat_completion(value: Sequence[Any]) -> bool:
    return bool(value) and all(
        isinstance(item, dict) and "role" in item and "content" in item for item in value
    )


def _split_focus_text(text: str) -> list[str]:
    return [part for part in _FOCUS_SPLIT_RE.split(text) if part.strip()]


def _clean_focus_token(token: str) -> str:
    return token.strip(" \t\r\n'\"`[](){}<>:：.!！?？")


def _strip_or_none(text: str) -> str | None:
    stripped = text.strip()
    return stripped or None


def _extract_boxed(text: str) -> tuple[str | None, int | None]:
    match = _BOXED_START_RE.search(text)
    if not match:
        return None, None

    start = match.end()
    depth = 1
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return _strip_or_none(text[start:index]), index + 1
        index += 1
    return None, None


def _tail_is_clean(tail: str) -> bool:
    allowed_special = (
        "<|im_end|>",
        "<pad>",
    )
    stripped = tail.strip()
    previous = None
    while stripped and stripped != previous:
        previous = stripped
        for token in allowed_special:
            if stripped.startswith(token):
                stripped = stripped[len(token) :].strip()
                break
    return stripped == ""
