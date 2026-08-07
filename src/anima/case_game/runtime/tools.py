"""Native six-tool contract and fail-closed policy for the Sherlock case game."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from anima.case_game.core.engine import CaseGameEngine, GameState, GameTurnResult
from anima.serve.inference.model_client import (
    ModelUnavailableError,
    NativeToolCall,
    ToolGenerationResult,
)

CASE_TOOL_SCHEMA_VERSION = "sherlock.case_tools.v1"
CASE_TOOL_POLICY_VERSION = "sherlock.case_tool_policy.v1"
MAX_TOOL_TEXT_LENGTH = 2000
_CALL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _text_schema(description: str) -> dict[str, Any]:
    return {
        "type": "string",
        "description": description,
        "minLength": 1,
        "maxLength": MAX_TOOL_TEXT_LENGTH,
    }


_CASE_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "ask_case_question",
            "description": "Ask one currently available, evidence-bounded case question.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_id": _text_schema(
                        "Exact target_id from an available ask lead in the current observation."
                    ),
                    "question": _text_schema("The player's question, without invented evidence."),
                },
                "required": ["target_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_case_target",
            "description": "Inspect one currently available case target.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_id": _text_schema(
                        "Exact target_id from an available inspect lead in the current observation."
                    ),
                    "inspection_request": _text_schema(
                        "What the player wants checked; do not invent the result."
                    ),
                },
                "required": ["target_id", "inspection_request"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_hypothesis",
            "description": "Submit one case hypothesis for deterministic evidence checking.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "hypothesis": _text_schema(
                        "The player's hypothesis and its evidence-based rationale."
                    )
                },
                "required": ["hypothesis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_hint",
            "description": "Request exactly one next authored, non-spoiling hint.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "focus": _text_schema(
                        "The player's requested focus, or a general request for the next hint."
                    )
                },
                "required": ["focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "description": "Submit the player's complete current deduction for deterministic scoring.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "solution": _text_schema(
                        "The player's proposed culprit, motive, method, and evidence chain."
                    )
                },
                "required": ["solution"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recap_case",
            "description": "Recap only the evidence and conclusions visible in the current state.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "request": _text_schema("What the player wants included in the recap.")
                },
                "required": ["request"],
            },
        },
    },
)

TOOL_ACTIONS = {
    "ask_case_question": ("ask", "question"),
    "inspect_case_target": ("inspect", "inspection_request"),
    "submit_hypothesis": ("hypothesize", "hypothesis"),
    "request_hint": ("hint", "focus"),
    "submit_solution": ("solve", "solution"),
    "recap_case": ("recap", "request"),
}
ACTION_TO_TOOL = {action: name for name, (action, _field) in TOOL_ACTIONS.items()}

CASE_TOOL_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(_CASE_TOOLS, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()
CASE_TOOL_POLICY_SHA256 = hashlib.sha256(
    (
        CASE_TOOL_POLICY_VERSION
        + "|exactly_one|schema_then_state_policy|no_session_player_case_args|"
        "ask_inspect_targets_must_be_available|post_case_recap_only|no_execution_before_repair"
    ).encode("utf-8")
).hexdigest()


class CaseToolError(ValueError):
    pass


class CaseToolProposalError(CaseToolError):
    def __init__(self, code: str, detail: str, *, trace_metadata: Mapping[str, Any]) -> None:
        super().__init__(detail)
        self.code = code
        self.trace_metadata = dict(trace_metadata)


@dataclass(frozen=True)
class CaseToolProposal:
    call: NativeToolCall
    arguments: Mapping[str, str]
    action: str
    player_text: str
    target_id: str | None
    source: str
    retry_count: int = 0
    trace_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_trace(self) -> dict[str, Any]:
        return {
            "schema_version": CASE_TOOL_SCHEMA_VERSION,
            "schema_sha256": CASE_TOOL_SCHEMA_SHA256,
            "policy_version": CASE_TOOL_POLICY_VERSION,
            "policy_sha256": CASE_TOOL_POLICY_SHA256,
            "source": self.source,
            "call_id": self.call.call_id,
            "name": self.call.name,
            "arguments": dict(self.arguments),
            "action": self.action,
            "target_id": self.target_id,
            "retry_count": self.retry_count,
            "schema_status": "pass",
            "policy_status": "pass",
            **dict(self.trace_metadata),
        }


class NativeToolModel(Protocol):
    def generate_tools(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str = "required",
    ) -> ToolGenerationResult: ...


def case_tool_schemas() -> list[dict[str, Any]]:
    return copy.deepcopy(list(_CASE_TOOLS))


def validate_native_tool_call(
    calls: Sequence[NativeToolCall],
    *,
    engine: CaseGameEngine,
    state: GameState,
    source: str,
    retry_count: int = 0,
    trace_metadata: Mapping[str, Any] | None = None,
) -> CaseToolProposal:
    if len(calls) != 1:
        raise CaseToolError(f"exactly one tool call is required; got {len(calls)}")
    call = calls[0]
    if not _CALL_ID_RE.fullmatch(call.call_id):
        raise CaseToolError("tool call id is invalid")
    spec = next(
        (row["function"] for row in _CASE_TOOLS if row["function"]["name"] == call.name),
        None,
    )
    if spec is None:
        raise CaseToolError(f"unknown case tool: {call.name}")
    try:
        value = json.loads(call.arguments_json)
    except json.JSONDecodeError as exc:
        raise CaseToolError(f"tool arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise CaseToolError("tool arguments must be an object")
    schema = spec["parameters"]
    allowed = set(schema["properties"])
    required = set(schema["required"])
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    if missing:
        raise CaseToolError(f"missing required tool arguments: {missing}")
    if extra:
        raise CaseToolError(f"unexpected tool arguments: {extra}")
    normalized: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(raw, str):
            raise CaseToolError(f"tool argument {key} must be a string")
        text = raw.strip()
        if not text or len(text) > MAX_TOOL_TEXT_LENGTH:
            raise CaseToolError(
                f"tool argument {key} must contain 1-{MAX_TOOL_TEXT_LENGTH} characters"
            )
        normalized[key] = text

    action, text_field = TOOL_ACTIONS[call.name]
    target_id = normalized.get("target_id")
    available = {
        (str(row["action"]), str(row["target_id"])) for row in engine.available_leads(state)
    }
    if action in {"ask", "inspect"} and (action, str(target_id)) not in available:
        raise CaseToolError(f"target is not currently available for {action}: {target_id}")
    if state.solved and action != "recap":
        raise CaseToolError("only recap_case is legal after the case is solved")
    return CaseToolProposal(
        call=call,
        arguments=normalized,
        action=action,
        player_text=normalized[text_field],
        target_id=target_id,
        source=source,
        retry_count=retry_count,
        trace_metadata=dict(trace_metadata or {}),
    )


def button_tool_proposal(
    *,
    action: str,
    player_text: str,
    target_id: str | None,
    engine: CaseGameEngine,
    state: GameState,
    call_id: str,
) -> CaseToolProposal:
    name = ACTION_TO_TOOL.get(action)
    if name is None:
        raise CaseToolError(f"button action has no case tool: {action}")
    _mapped_action, text_field = TOOL_ACTIONS[name]
    arguments = {text_field: player_text}
    if target_id is not None:
        arguments["target_id"] = target_id
    call = NativeToolCall(
        call_id=call_id,
        name=name,
        arguments_json=json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    )
    return validate_native_tool_call(
        (call,), engine=engine, state=state, source="button", trace_metadata={"model_calls": 0}
    )


def execute_case_tool(
    engine: CaseGameEngine,
    state: GameState,
    proposal: CaseToolProposal,
) -> GameTurnResult:
    """The single executor used by model proposals and button fallback."""

    return engine.apply(
        state,
        proposal.action,
        proposal.player_text,
        target_id=proposal.target_id,
    )


class LiveCaseToolProposer:
    def __init__(
        self,
        model_client: NativeToolModel,
        *,
        max_retries: int = 1,
        raw_hash: Callable[[bytes], str] | None = None,
    ) -> None:
        if max_retries not in {0, 1}:
            raise ValueError("max_retries must be 0 or 1")
        self.model_client = model_client
        self.max_retries = max_retries
        self.raw_hash = raw_hash or (lambda value: hashlib.sha256(value).hexdigest())

    @property
    def runtime_info(self) -> dict[str, Any]:
        return {
            "tool_agent_enabled": True,
            "tool_schema_version": CASE_TOOL_SCHEMA_VERSION,
            "tool_schema_sha256": CASE_TOOL_SCHEMA_SHA256,
            "tool_policy_version": CASE_TOOL_POLICY_VERSION,
            "tool_policy_sha256": CASE_TOOL_POLICY_SHA256,
            "tool_count": len(_CASE_TOOLS),
            "tool_format_retries": self.max_retries,
        }

    def propose(
        self,
        *,
        engine: CaseGameEngine,
        state: GameState,
        player_text: str,
    ) -> CaseToolProposal:
        context = _proposal_context(engine, state)
        system = (
            "You are the tool-selection policy for a constrained Sherlock case game. "
            "Return exactly one native tool call and no prose. Never invent a target_id. "
            "For ask/inspect, copy one target_id whose action matches from available_leads. "
            "Do not place session_id, player_id, case_id, state_version, hidden truth, or tool "
            "results in arguments. The deterministic executor, not you, decides observations."
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": json.dumps(
                    {"current_observation": context, "player_request": player_text},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ]
        generations: list[ToolGenerationResult] = []
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            if attempt:
                messages.extend(
                    (
                        {
                            "role": "assistant",
                            "content": generations[-1].raw_completion or generations[-1].content,
                        },
                        {
                            "role": "user",
                            "content": (
                                "The previous proposal was not executable: "
                                + errors[-1]
                                + ". Return exactly one corrected native tool call."
                            ),
                        },
                    )
                )
            try:
                generated = self.model_client.generate_tools(
                    system,
                    messages,
                    tools=case_tool_schemas(),
                    tool_choice="required",
                )
            except ModelUnavailableError as exc:
                raise CaseToolProposalError(
                    "tool_model_unavailable",
                    str(exc),
                    trace_metadata={
                        **self.runtime_info,
                        "tool_proposal_status": "failed",
                        "tool_proposal_error": "model_unavailable",
                        "tool_model_calls": len(generations),
                    },
                ) from exc
            generations.append(generated)
            raw = generated.raw_completion or generated.content
            try:
                return validate_native_tool_call(
                    generated.tool_calls,
                    engine=engine,
                    state=state,
                    source="model",
                    retry_count=attempt,
                    trace_metadata={
                        "model_calls": len(generations),
                        "input_tokens": sum(row.input_tokens for row in generations),
                        "output_tokens": sum(row.output_tokens for row in generations),
                        "total_ms": sum(row.total_ms for row in generations),
                        "raw_completion_sha256": self.raw_hash(raw.encode("utf-8")),
                    },
                )
            except CaseToolError as exc:
                errors.append(str(exc))
        raise CaseToolProposalError(
            "tool_validation_failed",
            errors[-1],
            trace_metadata={
                **self.runtime_info,
                "tool_proposal_status": "failed",
                "tool_proposal_error": errors[-1],
                "tool_model_calls": len(generations),
                "tool_retry_count": max(0, len(generations) - 1),
                "raw_completion_sha256": self.raw_hash(
                    (generations[-1].raw_completion or generations[-1].content).encode("utf-8")
                ),
            },
        )


def _proposal_context(engine: CaseGameEngine, state: GameState) -> dict[str, Any]:
    context = engine.model_context(state)
    return {
        "stage": context["case"]["stage"],
        "solved": context["case"]["solved"],
        "available_leads": context["available_leads"],
        "visible_evidence": [
            {
                "evidence_id": row["evidence_id"],
                "title": row["title"],
                "text": row["text"],
            }
            for row in context["visible_evidence"]
        ],
        "hint_tier": context["state_summary"]["hint_tier"],
        "allowed_tool_names": list(TOOL_ACTIONS),
    }
