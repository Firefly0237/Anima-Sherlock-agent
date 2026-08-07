"""Pinned live-model answerer for the Sherlock case-game demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from anima.case_game.core.demo import HostAnswer
from anima.case_game.core.engine import GameTurnResult
from anima.case_game.runtime.player_memory import (
    PlayerMemoryError,
    PlayerMemoryService,
    route_player_memory_predicates,
)
from anima.persona.contracts.pack import load_pack
from anima.persona.contracts.schemas import MemoryRecord
from anima.persona.runtime.context import build_context
from anima.persona.runtime.output import ParsedOutput, parse_output
from anima.serve.core.runtime import REPAIR_INSTRUCTION
from anima.serve.inference.model_client import GenerationResult, ModelUnavailableError

LIVE_RUNTIME_MODE = "live_adapter"  # Backward-compatible module constant; instances derive mode.
DEGRADED_MODEL_ANSWER = (
    "本轮模型回复未通过运行时校验，因此没有把未经验证的内容交给你。"
    "案件状态仍由规则引擎记录；请重试这一行动。"
)
MEMORY_GROUNDING_REPAIR_INSTRUCTION = """上一回答没有完整遵守本轮用户长期记忆约束。请重写同一轮回答：
- 已检索记录如下；若非空，必须从系统[用户长期记忆]区块准确复述其 object，并在 used_memory_ids 中逐字引用完整 memory id：{memory_refs}
- 下列玩家资料字段在本轮没有记录；若非空，必须明确说明当前没有对应记录，不得声称记得：{missing_predicates}
- 案件动作未命中或案件证据不足只约束案情子问题，不能覆盖已有的玩家资料。
- 仍须回答同一轮的案情部分、遵守剧透与安全边界，并保持福尔摩斯口吻。
- 只输出一个 <anima_state> JSON 块和一个 <answer> 块。
检测到的问题：{errors}"""
NO_MEMORY_ACKNOWLEDGEMENTS = (
    "没有记录",
    "未记录",
    "没有这项资料",
    "没有对应资料",
    "尚未告诉",
    "未曾告诉",
    "还没有告诉",
    "没有告诉",
    "不记得",
    "没有保存",
    "记录里没有",
    "资料中没有",
)
FALSE_MEMORY_CLAIMS = ("当然记得", "我记得", "还记得你")

IDENTITY_FIELDS = (
    "backend",
    "base_model",
    "base_model_revision",
    "adapter_id",
    "adapter_sha256",
    "served_model",
    "code_commit",
)


class CaseGameModelIdentityError(ValueError):
    pass


class ChatModelClient(Protocol):
    def list_models(self) -> list[str]: ...

    def model_identity(self) -> dict[str, Any]: ...

    def generate(
        self,
        system: str,
        messages: Sequence[dict[str, str]],
    ) -> GenerationResult: ...


def load_expected_model_identity(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CaseGameModelIdentityError("model identity contract must be a JSON object")
    return _identity_contract(value)


class LiveCaseAnswerer:
    """Generate one guarded Sherlock turn from an identity-bound model service."""

    def __init__(
        self,
        *,
        model_client: ChatModelClient,
        persona_pack_dir: Path,
        expected_identity: Mapping[str, Any],
        memory_service: PlayerMemoryService | None = None,
        max_history_turns: int = 12,
        max_format_retries: int = 1,
        max_memory_grounding_retries: int = 1,
    ) -> None:
        if max_history_turns < 0:
            raise ValueError("max_history_turns must be >= 0")
        if max_format_retries not in {0, 1}:
            raise ValueError("max_format_retries must be 0 or 1")
        if max_memory_grounding_retries not in {0, 1}:
            raise ValueError("max_memory_grounding_retries must be 0 or 1")
        self.model_client = model_client
        self.pack = load_pack(persona_pack_dir)
        self.memory_service = memory_service
        if self.memory_service is not None:
            memory_manifest = self.memory_service.pack.manifest
            if (
                memory_manifest.persona_id != self.pack.manifest.persona_id
                or memory_manifest.version != self.pack.manifest.version
                or memory_manifest.content_sha256 != self.pack.manifest.content_sha256
            ):
                raise ValueError(
                    "player memory service and live answerer must use the same persona pack"
                )
        self.expected_identity = _identity_contract(expected_identity)
        self.max_history_turns = max_history_turns
        self.max_format_retries = max_format_retries
        self.max_memory_grounding_retries = max_memory_grounding_retries
        self.actual_identity = self._verify_identity()
        self.runtime_mode = _runtime_mode(self.actual_identity)

    @property
    def runtime_info(self) -> dict[str, Any]:
        info = {
            "mode": self.runtime_mode,
            "label": "Live identity-verified Sherlock model",
            "identity_verified": True,
            **{field: self.actual_identity[field] for field in IDENTITY_FIELDS},
        }
        info.update(
            self.memory_service.runtime_info
            if self.memory_service is not None
            else {"memory_enabled": False}
        )
        info["memory_grounding_max_retries"] = self.max_memory_grounding_retries
        return info

    def __call__(
        self,
        persona_user_message: str,
        game_result: GameTurnResult,
        history: Sequence[Mapping[str, str]],
    ) -> HostAnswer:
        return self._answer(
            persona_user_message,
            game_result,
            history,
            memories=(),
            memory_scores=(),
            memory_query_predicates=(),
        )

    def answer_for_player(
        self,
        persona_user_message: str,
        game_result: GameTurnResult,
        history: Sequence[Mapping[str, str]],
        *,
        player_id: str,
        player_text: str,
    ) -> HostAnswer:
        if self.memory_service is None:
            return self(persona_user_message, game_result, history)
        memory_query_predicates = route_player_memory_predicates(player_text)
        try:
            retrieval = self.memory_service.retrieve(player_id, player_text)
        except PlayerMemoryError as exc:
            built = build_context(
                self.pack,
                retrieved_lore=(),
                memories=(),
                history=history,
                user_message=persona_user_message,
                max_history_turns=self.max_history_turns,
            )
            return self._degraded_answer(
                error_code="memory_unavailable",
                context_hash=built.context_hash,
                prompt_version=built.prompt_version,
                retry_count=0,
                parse_status="not_started",
                parse_errors=(),
                generations=(),
                detail=str(exc),
                memory_query_predicates=memory_query_predicates,
            )
        return self._answer(
            persona_user_message,
            game_result,
            history,
            memories=retrieval.records,
            memory_scores=retrieval.scores,
            memory_query_predicates=memory_query_predicates,
        )

    def _answer(
        self,
        persona_user_message: str,
        _game_result: GameTurnResult,
        history: Sequence[Mapping[str, str]],
        *,
        memories: Sequence[MemoryRecord],
        memory_scores: Sequence[float],
        memory_query_predicates: Sequence[str],
    ) -> HostAnswer:
        built = build_context(
            self.pack,
            retrieved_lore=(),
            memories=memories,
            history=history,
            user_message=persona_user_message,
            max_history_turns=self.max_history_turns,
        )
        generations: list[GenerationResult] = []
        retry_count = 0
        memory_grounding_retry_count = 0
        memory_grounding_errors: tuple[str, ...] = ()
        parse_status = "failed"
        parsed: ParsedOutput | None = None
        try:
            first = self.model_client.generate(built.system_prompt, built.messages)
            generations.append(first)
            parsed = parse_output(first.text)
            parse_status = "ok" if parsed.ok else "failed"
            if not parsed.ok and self.max_format_retries:
                retry_count = 1
                repaired = self.model_client.generate(
                    built.system_prompt,
                    (
                        *built.messages,
                        {"role": "assistant", "content": first.text},
                        {"role": "user", "content": REPAIR_INSTRUCTION},
                    ),
                )
                generations.append(repaired)
                parsed = parse_output(repaired.text)
                parse_status = "repaired" if parsed.ok else "failed"

            if parsed.ok and parsed.state is not None and parsed.answer is not None:
                memory_grounding_errors = _memory_grounding_errors(
                    parsed,
                    memories,
                    memory_query_predicates,
                )
                if memory_grounding_errors and self.max_memory_grounding_retries:
                    memory_grounding_retry_count = 1
                    retry_count += 1
                    grounded = self.model_client.generate(
                        built.system_prompt,
                        (
                            *built.messages,
                            {"role": "assistant", "content": generations[-1].text},
                            {
                                "role": "user",
                                "content": _render_memory_grounding_repair_instruction(
                                    memories,
                                    memory_query_predicates,
                                    parsed,
                                    memory_grounding_errors,
                                ),
                            },
                        ),
                    )
                    generations.append(grounded)
                    parsed = parse_output(grounded.text)
                    if parsed.ok and parsed.state is not None and parsed.answer is not None:
                        memory_grounding_errors = _memory_grounding_errors(
                            parsed,
                            memories,
                            memory_query_predicates,
                        )
                    else:
                        memory_grounding_errors = (
                            "memory_grounding_repair_invalid_output",
                            *(parsed.errors if parsed is not None else ()),
                        )
                    parse_status = (
                        "memory_grounding_repaired"
                        if parsed.ok and not memory_grounding_errors
                        else "failed"
                    )
        except ModelUnavailableError as exc:
            return self._degraded_answer(
                error_code="model_unavailable",
                context_hash=built.context_hash,
                prompt_version=built.prompt_version,
                retry_count=retry_count,
                parse_status=parse_status,
                parse_errors=(),
                generations=generations,
                detail=str(exc),
                memories=memories,
                memory_scores=memory_scores,
                memory_grounding_retry_count=memory_grounding_retry_count,
                memory_grounding_errors=memory_grounding_errors,
                memory_query_predicates=memory_query_predicates,
            )

        if memory_grounding_errors:
            return self._degraded_answer(
                error_code="memory_grounding_failed",
                context_hash=built.context_hash,
                prompt_version=built.prompt_version,
                retry_count=retry_count,
                parse_status="failed",
                parse_errors=memory_grounding_errors,
                generations=generations,
                memories=memories,
                memory_scores=memory_scores,
                memory_grounding_retry_count=memory_grounding_retry_count,
                memory_grounding_errors=memory_grounding_errors,
                memory_query_predicates=memory_query_predicates,
            )

        if parsed is None or not parsed.ok or parsed.answer is None:
            return self._degraded_answer(
                error_code="model_output_invalid",
                context_hash=built.context_hash,
                prompt_version=built.prompt_version,
                retry_count=retry_count,
                parse_status="failed",
                parse_errors=parsed.errors if parsed is not None else (),
                generations=generations,
                memories=memories,
                memory_scores=memory_scores,
                memory_grounding_retry_count=memory_grounding_retry_count,
                memory_grounding_errors=memory_grounding_errors,
                memory_query_predicates=memory_query_predicates,
            )

        assert parsed.state is not None
        retrieved_memory_ids = {record.memory_id for record in memories}
        invalid_lore_ids = tuple(parsed.state.used_lore_ids)
        invalid_memory_ids = tuple(
            memory_id
            for memory_id in parsed.state.used_memory_ids
            if memory_id not in retrieved_memory_ids
        )
        if invalid_lore_ids or invalid_memory_ids:
            errors = tuple(
                [f"invalid_lore_citation:{item}" for item in invalid_lore_ids]
                + [f"invalid_memory_citation:{item}" for item in invalid_memory_ids]
            )
            return self._degraded_answer(
                error_code="invalid_citation",
                context_hash=built.context_hash,
                prompt_version=built.prompt_version,
                retry_count=retry_count,
                parse_status=parse_status,
                parse_errors=errors,
                generations=generations,
                memories=memories,
                memory_scores=memory_scores,
                memory_grounding_retry_count=memory_grounding_retry_count,
                memory_grounding_errors=memory_grounding_errors,
                memory_query_predicates=memory_query_predicates,
            )

        served_generation = generations[-1]
        return HostAnswer(
            text=parsed.answer,
            history_content=served_generation.text,
            trace_metadata=self._trace_metadata(
                context_hash=built.context_hash,
                prompt_version=built.prompt_version,
                retry_count=retry_count,
                parse_status=parse_status,
                parse_errors=(),
                generations=generations,
                memories=memories,
                memory_scores=memory_scores,
                used_memory_ids=parsed.state.used_memory_ids,
                answer_mode=parsed.state.answer_mode,
                memory_grounding_retry_count=memory_grounding_retry_count,
                memory_grounding_errors=memory_grounding_errors,
                memory_query_predicates=memory_query_predicates,
            ),
            memory_ops=parsed.state.memory_ops,
        )

    def _verify_identity(self) -> dict[str, Any]:
        try:
            actual = self.model_client.model_identity()
            served_models = self.model_client.list_models()
        except ModelUnavailableError as exc:
            raise CaseGameModelIdentityError(f"model identity unavailable: {exc}") from exc
        mismatches = {
            field: {
                "expected": self.expected_identity[field],
                "actual": actual.get(field),
            }
            for field in IDENTITY_FIELDS
            if actual.get(field) != self.expected_identity[field]
        }
        expected_served = str(self.expected_identity["served_model"])
        if expected_served not in served_models:
            mismatches["listed_models"] = {
                "expected": expected_served,
                "actual": list(served_models),
            }
        if mismatches:
            raise CaseGameModelIdentityError(
                "live model identity mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        return {field: actual[field] for field in IDENTITY_FIELDS}

    def _degraded_answer(
        self,
        *,
        error_code: str,
        context_hash: str,
        prompt_version: str,
        retry_count: int,
        parse_status: str,
        parse_errors: Sequence[str],
        generations: Sequence[GenerationResult],
        detail: str | None = None,
        memories: Sequence[MemoryRecord] = (),
        memory_scores: Sequence[float] = (),
        memory_grounding_retry_count: int = 0,
        memory_grounding_errors: Sequence[str] = (),
        memory_query_predicates: Sequence[str] = (),
    ) -> HostAnswer:
        trace = self._trace_metadata(
            context_hash=context_hash,
            prompt_version=prompt_version,
            retry_count=retry_count,
            parse_status=parse_status,
            parse_errors=parse_errors,
            generations=generations,
            memories=memories,
            memory_scores=memory_scores,
            memory_grounding_retry_count=memory_grounding_retry_count,
            memory_grounding_errors=memory_grounding_errors,
            memory_query_predicates=memory_query_predicates,
        )
        trace["host_error_code"] = error_code
        if detail:
            trace["host_error_detail"] = detail
        return HostAnswer(
            text=DEGRADED_MODEL_ANSWER,
            degraded=True,
            error_code=error_code,
            trace_metadata=trace,
        )

    def _trace_metadata(
        self,
        *,
        context_hash: str,
        prompt_version: str,
        retry_count: int,
        parse_status: str,
        parse_errors: Sequence[str],
        generations: Sequence[GenerationResult],
        memories: Sequence[MemoryRecord] = (),
        memory_scores: Sequence[float] = (),
        used_memory_ids: Sequence[str] = (),
        answer_mode: str | None = None,
        memory_grounding_retry_count: int = 0,
        memory_grounding_errors: Sequence[str] = (),
        memory_query_predicates: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "host_runtime_mode": self.runtime_mode,
            "host_model_identity": {
                field: self.actual_identity[field] for field in IDENTITY_FIELDS
            },
            "host_context_hash": context_hash,
            "host_prompt_version": prompt_version,
            "host_parse_status": parse_status,
            "host_parse_errors": list(parse_errors),
            "host_retry_count": retry_count,
            "host_generation_calls": len(generations),
            "host_input_tokens": sum(row.input_tokens for row in generations),
            "host_output_tokens": sum(row.output_tokens for row in generations),
            "host_total_ms": sum(row.total_ms for row in generations),
            "host_ttft_ms": generations[-1].ttft_ms if generations else None,
            "host_retrieved_memory_ids": [record.memory_id for record in memories],
            "host_retrieved_memory_scores": [float(score) for score in memory_scores],
            "host_used_memory_ids": list(used_memory_ids),
            "host_answer_mode": answer_mode,
            "host_memory_grounding_retries": memory_grounding_retry_count,
            "host_memory_grounding_errors": list(memory_grounding_errors),
            "host_memory_query_predicates": list(memory_query_predicates),
        }


# Compatibility alias for historical imports and receipts. New code should use
# LiveCaseAnswerer because the same runtime evaluates Base, SFT, and DPO arms.
LiveDPOAnswerer = LiveCaseAnswerer


def _runtime_mode(identity: Mapping[str, Any]) -> str:
    adapter_id = str(identity.get("adapter_id") or "").lower()
    if not adapter_id:
        return "live_base"
    if "dpo" in adapter_id:
        return "live_dpo"
    if "sft" in adapter_id:
        return "live_sft"
    return "live_adapter"


def _memory_grounding_errors(
    parsed: ParsedOutput,
    memories: Sequence[MemoryRecord],
    memory_query_predicates: Sequence[str],
) -> tuple[str, ...]:
    if parsed.state is None or parsed.answer is None:
        return ()
    if parsed.state.answer_mode == "refuse":
        return ()
    mutated_predicates = _mutated_memory_predicates(parsed)
    required = tuple(record for record in memories if record.predicate not in mutated_predicates)
    used_ids = frozenset(parsed.state.used_memory_ids)
    answer = "".join(parsed.answer.split())
    errors: list[str] = []
    for record in required:
        if record.memory_id not in used_ids:
            errors.append(f"missing_memory_citation:{record.memory_id}")
        value = "".join(record.object.split())
        if value and value not in answer:
            errors.append(f"missing_memory_value:{record.memory_id}")
    available_predicates = {record.predicate for record in required}
    missing_predicates = tuple(
        predicate
        for predicate in memory_query_predicates
        if predicate not in available_predicates and predicate not in mutated_predicates
    )
    if missing_predicates:
        if not any(marker in parsed.answer for marker in NO_MEMORY_ACKNOWLEDGEMENTS):
            errors.append("missing_memory_abstention:" + ",".join(missing_predicates))
        if not required and any(marker in parsed.answer for marker in FALSE_MEMORY_CLAIMS):
            errors.append("false_memory_claim_without_record:" + ",".join(missing_predicates))
    return tuple(errors)


def _render_memory_grounding_repair_instruction(
    memories: Sequence[MemoryRecord],
    memory_query_predicates: Sequence[str],
    parsed: ParsedOutput,
    errors: Sequence[str],
) -> str:
    mutated_predicates = _mutated_memory_predicates(parsed)
    refs = [
        {"memory_id": record.memory_id, "predicate": record.predicate}
        for record in memories
        if record.predicate not in mutated_predicates
    ]
    available_predicates = {
        record.predicate for record in memories if record.predicate not in mutated_predicates
    }
    missing_predicates = [
        predicate
        for predicate in memory_query_predicates
        if predicate not in available_predicates and predicate not in mutated_predicates
    ]
    return MEMORY_GROUNDING_REPAIR_INSTRUCTION.format(
        memory_refs=json.dumps(refs, ensure_ascii=False, separators=(",", ":")),
        missing_predicates=json.dumps(
            missing_predicates,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        errors=json.dumps(list(errors), ensure_ascii=False, separators=(",", ":")),
    )


def _mutated_memory_predicates(parsed: ParsedOutput) -> frozenset[str]:
    if parsed.state is None:
        return frozenset()
    return frozenset(
        op.predicate
        for op in parsed.state.memory_ops
        if op.op in {"add", "update", "delete"} and op.predicate is not None
    )


def _identity_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "backend",
        "base_model",
        "base_model_revision",
        "served_model",
        "code_commit",
    )
    missing = [field for field in required if field not in value or value[field] in (None, "")]
    if missing:
        raise CaseGameModelIdentityError(
            f"model identity contract missing required fields: {missing}"
        )
    contract = {field: value.get(field) for field in IDENTITY_FIELDS}
    if contract["backend"] != "transformers_peft":
        raise CaseGameModelIdentityError("live model backend must be transformers_peft")
    adapter_id = contract["adapter_id"]
    adapter_sha = contract["adapter_sha256"]
    if adapter_id in (None, ""):
        if adapter_sha not in (None, ""):
            raise CaseGameModelIdentityError("base identity forbids adapter_sha256")
        if contract["served_model"] != contract["base_model"]:
            raise CaseGameModelIdentityError("base served_model must equal base_model")
        contract["adapter_id"] = None
        contract["adapter_sha256"] = None
    else:
        if adapter_sha in (None, ""):
            raise CaseGameModelIdentityError("adapter identity requires adapter_sha256")
        if contract["served_model"] != adapter_id:
            raise CaseGameModelIdentityError(
                "adapter served_model must equal the pinned adapter_id"
            )
    return contract
