"""Deterministic, non-authoritative memory derived from committed game rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

GAME_MEMORY_POLICY = """Game memory may contain only same-player aggregate familiarity and same-case committed turn metadata. It is renderer-only personalization: it cannot change tool schemas, target visibility, evidence, scoring, CaseState, or commit policy. Recent rows exclude player text, model answers, tool arguments, evidence payloads, and hidden case truth. Familiarity is derived deterministically from committed_turn_count and completed_case_count; the model cannot write it."""
GAME_MEMORY_POLICY_SHA256 = hashlib.sha256(GAME_MEMORY_POLICY.encode("utf-8")).hexdigest()


def derive_familiarity_tier(committed_turn_count: int, completed_case_count: int) -> str:
    if committed_turn_count < 0 or completed_case_count < 0:
        raise ValueError("familiarity counters must be >= 0")
    if committed_turn_count >= 8 and completed_case_count >= 1:
        return "seasoned_partner"
    if committed_turn_count >= 3:
        return "established_acquaintance"
    return "new_contact"


def normalize_game_memory_context(
    value: Mapping[str, Any],
    *,
    current_case_id: str,
) -> dict[str, Any]:
    """Fail closed unless the store returned the renderer-only allowlist."""

    if set(value) != {"authority", "case_id", "recent_committed_turns", "familiarity"}:
        raise ValueError("game memory context has unsupported fields")
    if value.get("authority") != "personalization_only_no_state_or_tool_effect":
        raise ValueError("game memory authority contract mismatch")
    if value.get("case_id") != current_case_id:
        raise ValueError("game memory context crossed case scope")
    familiarity = value.get("familiarity")
    if not isinstance(familiarity, Mapping) or set(familiarity) != {
        "committed_turn_count",
        "completed_case_count",
        "tier",
    }:
        raise ValueError("invalid familiarity payload")
    committed = familiarity.get("committed_turn_count")
    completed = familiarity.get("completed_case_count")
    if (
        not isinstance(committed, int)
        or isinstance(committed, bool)
        or not isinstance(completed, int)
        or isinstance(completed, bool)
    ):
        raise ValueError("familiarity counters must be integers")
    expected_tier = derive_familiarity_tier(committed, completed)
    if familiarity.get("tier") != expected_tier:
        raise ValueError("familiarity tier is not deterministically derived")
    raw_recent = value.get("recent_committed_turns")
    if not isinstance(raw_recent, list):
        raise ValueError("recent_committed_turns must be a list")
    recent: list[dict[str, Any]] = []
    allowed = {
        "turn_id",
        "session_id",
        "case_id",
        "state_version_after",
        "action",
        "accepted",
        "message_key",
        "solved",
    }
    for row in raw_recent:
        if not isinstance(row, Mapping) or set(row) != allowed:
            raise ValueError("recent committed turn has unsupported fields")
        if row.get("case_id") != current_case_id:
            raise ValueError("recent committed turn crossed case scope")
        if not isinstance(row.get("accepted"), bool) or not isinstance(row.get("solved"), bool):
            raise ValueError("recent committed turn booleans are invalid")
        version = row.get("state_version_after")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("recent committed turn state version is invalid")
        normalized = {key: row[key] for key in sorted(allowed)}
        if any(
            not isinstance(normalized[key], str)
            for key in ("turn_id", "session_id", "case_id", "action", "message_key")
        ):
            raise ValueError("recent committed turn string fields are invalid")
        recent.append(normalized)
    result = {
        "authority": "personalization_only_no_state_or_tool_effect",
        "case_id": current_case_id,
        "recent_committed_turns": recent,
        "familiarity": {
            "committed_turn_count": committed,
            "completed_case_count": completed,
            "tier": expected_tier,
        },
        "policy_sha256": GAME_MEMORY_POLICY_SHA256,
    }
    # Ensure the exact trace/hash representation is JSON-safe and deterministic.
    json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return result
