"""Parse the output contract: one <anima_state> block plus one <answer> block.

The parser recovers whatever content it can (the served path may still show a
recovered answer after a single format-repair retry), but `ok` is strict: any
contract violation keeps first-pass scoring honest. Standard library only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from anima.persona.contracts.schemas import AnimaState, PersonaContractError

STATE_BLOCK_RE = re.compile(r"<anima_state>(.*?)</anima_state>", re.DOTALL)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class ParsedOutput:
    ok: bool
    state: AnimaState | None
    answer: str | None
    errors: tuple[str, ...]


def parse_output(text: str) -> ParsedOutput:
    errors: list[str] = []

    state_blocks = STATE_BLOCK_RE.findall(text)
    if not state_blocks:
        errors.append("missing_state_block")
    elif len(state_blocks) > 1:
        errors.append("multiple_state_blocks")

    answer_blocks = ANSWER_BLOCK_RE.findall(text)
    if not answer_blocks:
        errors.append("missing_answer_block")
    elif len(answer_blocks) > 1:
        errors.append("multiple_answer_blocks")

    state: AnimaState | None = None
    if state_blocks:
        try:
            state_raw = json.loads(state_blocks[0])
        except json.JSONDecodeError as exc:
            errors.append(f"state_json_invalid: {exc.msg}")
        else:
            if not isinstance(state_raw, dict):
                errors.append("state_json_invalid: not an object")
            else:
                try:
                    state = AnimaState.from_mapping(state_raw)
                except PersonaContractError as exc:
                    errors.extend(f"state_contract: {e}" for e in exc.errors)

    answer: str | None = None
    if answer_blocks:
        answer = answer_blocks[0].strip()
        if not answer:
            errors.append("empty_answer")
            answer = None

    stripped = STATE_BLOCK_RE.sub("", ANSWER_BLOCK_RE.sub("", text), count=1)
    if stripped.strip():
        errors.append("extra_text_outside_blocks")

    return ParsedOutput(ok=not errors, state=state, answer=answer, errors=tuple(errors))
