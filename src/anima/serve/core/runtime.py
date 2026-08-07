"""Deterministic request processing for the persona runtime.

One deterministic path per turn, no model-driven routing: validate → retrieve
lore+memory (when their independent read controls allow it) → assemble context →
generate → parse → validate citations → commit memory ops (under a separate write
control) → trace. first_pass and served are kept distinct; one narrowly
bounded terminal-tag recovery may precede at most one model format-repair retry
and one citation-repair retry. A model failure yields degraded=true, never a
fabricated success. Standard library only (model/embedder/repo are
injected).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from anima.persona.contracts.pack import LoadedPack
from anima.persona.contracts.schemas import MemoryOp
from anima.persona.runtime.context import PROMPT_VERSION, build_context
from anima.persona.runtime.memory import plan_memory_commits
from anima.persona.runtime.output import parse_output
from anima.persona.runtime.retrieval import (
    Embedder,
    VectorIndex,
    build_lore_index,
    rank_by_similarity,
    render_memory_text,
)
from anima.serve.core.metrics import MetricsRegistry
from anima.serve.core.repositories import ConversationRow, Repository
from anima.serve.inference.model_client import GenerationResult, ModelUnavailableError

REPAIR_INSTRUCTION = (
    "你上一条输出未通过格式解析。请重写完整回复，不得照抄上一条。"
    "输出必须且只能依次包含一个 <anima_state> JSON 块和一个 <answer> 块，"
    "两个块都必须完整闭合，块外不得有任何文字。"
    "请保留核心结论并删去重复铺陈，使 <answer> 正文不超过 220 个汉字。"
    "结束输出前逐字检查：必须同时存在 </anima_state> 和 </answer>，"
    "并且最后一个非空白字符序列必须是 </answer>。"
)
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"
CITATION_REPAIR_INSTRUCTION = (
    "你上一条输出的引用 id 未通过校验。请只重新输出一个 <anima_state> JSON 块和一个 <answer> 块，"
    "块外不要有任何其他文字。\n"
    "回答依据、引用 id 和写入增量必须分别重判，不得互相推导。"
    "若核心答案可由本修复指令之前的对话历史中当前用户的明确陈述支持，"
    '必须保留由对话历史支持的核心答案；允许使用 answer_mode="memory"，'
    "但不得因为合法 memory id 列表为空就改答不知道或转移问题。\n"
    "规则：used_lore_ids 只能从下列完整 lore id 中逐字选择：{lore_ids}。\n"
    "used_memory_ids 只能从下列完整 memory id 中逐字选择：{memory_ids}。\n"
    "不得写 current_user、memory_1、memory_2、msg id 或截断 id；如果没有使用该类证据，必须写 []；"
    "只依据对话历史作答时 used_memory_ids 必须写 []。\n"
    "重写时须仅依据本修复指令之前的最后一条真实用户消息独立重判 memory_ops；本修复指令不是用户事实。"
    "不得把上一条输出本身当作 memory_ops 的写入证据，也不得从历史或回答复写旧事实；"
    "若原始真实用户消息确有新增、更正或删除的稳定事实，须重新输出对应事实增量，内容可以与上一条相同；"
    "若该真实消息只是纯提问、回忆、确认或要求复述而没有新事实增量，memory_ops 必须写 []。"
    "memory_ops 仍只依据最后一条真实用户消息独立判定，answer_mode=memory 不授予写入权限。\n"
    "本次非法 lore id：{invalid_lore_ids}。\n"
    "本次非法 memory id：{invalid_memory_ids}。"
)
DEGRADED_FALLBACK = "抱歉，我这会儿有点走神，没能理清这句话，你能再说一遍吗？"


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...
    def generate(self, system: str, messages: Sequence[dict[str, str]]) -> GenerationResult: ...


class QueryEmbedder(Embedder, Protocol):
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RuntimeConfig:
    lore_retrieval_enabled: bool = True
    memory_retrieval_enabled: bool = True
    memory_write_enabled: bool = True
    lore_top_k: int = 8
    memory_top_k: int = 5
    lore_min_score: float | None = None
    memory_min_score: float | None = None
    max_history_turns: int = 12
    max_input_chars: int = 2000
    max_format_retries: int = 1
    max_citation_retries: int = 1

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, formal: bool) -> "RuntimeConfig":
        """Parse explicit read/write controls, retaining only safe pilot compatibility.

        Formal runtimes must bind all three controls into their config identity. A
        legacy pilot config may still use ``retrieval.enabled`` plus the combined
        ``runtime.memory_enabled`` switch; its old semantics are preserved by mapping
        that one Memory value to *both* read and write. Mixed or partial forms are
        rejected so an omitted field can never silently widen a formal arm.
        """

        retrieval = config.get("retrieval", {})
        runtime = config.get("runtime", {})
        if not isinstance(retrieval, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError("runtime retrieval/runtime sections must be mappings")

        split_fields = (
            (retrieval, "lore_enabled"),
            (retrieval, "memory_enabled"),
            (runtime, "memory_write_enabled"),
        )
        split_present = [name in section for section, name in split_fields]
        legacy_present = "enabled" in retrieval or "memory_enabled" in runtime

        if formal:
            if legacy_present or not all(split_present):
                raise ValueError(
                    "formal runtime requires explicit split Memory controls: "
                    "retrieval.lore_enabled, retrieval.memory_enabled, and "
                    "runtime.memory_write_enabled; legacy combined controls are forbidden"
                )
        elif any(split_present):
            if not all(split_present) or legacy_present:
                raise ValueError(
                    "split Memory controls must be complete and cannot be mixed with legacy controls"
                )

        if all(split_present):
            lore_retrieval_enabled = _strict_bool(
                retrieval["lore_enabled"], "retrieval.lore_enabled"
            )
            memory_retrieval_enabled = _strict_bool(
                retrieval["memory_enabled"], "retrieval.memory_enabled"
            )
            memory_write_enabled = _strict_bool(
                runtime["memory_write_enabled"], "runtime.memory_write_enabled"
            )
        else:
            lore_retrieval_enabled = _strict_bool(
                retrieval.get("enabled", True), "retrieval.enabled"
            )
            legacy_memory_enabled = _strict_bool(
                runtime.get("memory_enabled", True), "runtime.memory_enabled"
            )
            memory_retrieval_enabled = legacy_memory_enabled
            memory_write_enabled = legacy_memory_enabled

        return cls(
            lore_retrieval_enabled=lore_retrieval_enabled,
            memory_retrieval_enabled=memory_retrieval_enabled,
            memory_write_enabled=memory_write_enabled,
            lore_top_k=int(retrieval.get("lore_top_k", 8)),
            memory_top_k=int(retrieval.get("memory_top_k", 5)),
            lore_min_score=retrieval.get("lore_min_score"),
            memory_min_score=retrieval.get("memory_min_score"),
            max_history_turns=int(runtime.get("max_history_turns", 12)),
            max_input_chars=int(runtime.get("max_input_chars", 2000)),
            max_format_retries=int(runtime.get("max_format_retries", 1)),
            max_citation_retries=int(runtime.get("max_citation_retries", 1)),
        )


@dataclass(frozen=True)
class ModelInfo:
    base_model: str
    adapter_id: str | None
    adapter_sha256: str | None
    base_model_revision: str | None = None
    embedding_model_revision: str | None = None
    prompt_version: str = PROMPT_VERSION


@dataclass
class AppServices:
    packs: dict[str, LoadedPack]
    lore_indexes: dict[str, VectorIndex]
    repo: Repository
    model: ChatModel
    embedder: QueryEmbedder
    metrics: MetricsRegistry
    model_info: ModelInfo
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    test_mode: bool = False
    eval_token_hash: str | None = None
    code_commit: str = "unknown"
    runtime_config_hash: str = "unknown"
    model_server_identity: dict[str, Any] | None = None


def build_lore_indexes(packs: dict[str, LoadedPack], embedder: Embedder) -> dict[str, VectorIndex]:
    return {persona_id: build_lore_index(pack.lore, embedder) for persona_id, pack in packs.items()}


class InputTooLongError(ValueError):
    pass


@dataclass
class MessageResult:
    request_id: str
    served_answer: str
    first_pass_raw: str
    served_raw: str
    parse_status: str  # ok | terminal_closed | repaired | citation_repaired | failed
    answer_mode: str | None
    retry_count: int
    degraded: bool
    error_code: str | None
    used_lore_ids: tuple[str, ...]
    used_memory_ids: tuple[str, ...]
    retrieved_lore_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    invalid_lore_citations: tuple[str, ...]
    invalid_memory_citations: tuple[str, ...]
    memory_ops_requested: int
    memory_ops_committed: int
    memory_ops_rejected: tuple[dict[str, Any], ...]
    user_message_id: str
    assistant_message_id: str | None
    ttft_ms: float | None
    total_ms: float
    input_tokens: int
    output_tokens: int
    trace: dict[str, Any]


def handle_message(
    services: AppServices,
    *,
    user_id: str,
    conversation: ConversationRow,
    user_message: str,
    request_id: str,
    now: str,
    id_factory: Callable[[str], str],
) -> MessageResult:
    config = services.config
    metrics = services.metrics
    metrics.inc("anima_requests_total", persona=conversation.persona_id)
    started = time.monotonic()

    if len(user_message) > config.max_input_chars:
        metrics.inc("anima_rejected_total", reason="input_too_long")
        raise InputTooLongError(f"user_message exceeds {config.max_input_chars} characters")

    pack = services.packs[conversation.persona_id]
    user_msg_id = id_factory("msg")
    repo = services.repo
    repo.append_message(conversation.conversation_id, user_msg_id, "user", user_message)

    try:
        return _process_message(
            services,
            user_id=user_id,
            conversation=conversation,
            user_message=user_message,
            request_id=request_id,
            now=now,
            id_factory=id_factory,
            pack=pack,
            user_msg_id=user_msg_id,
            started=started,
        )
    except Exception as exc:  # retrieval/commit/etc.: never leak an untraced 500
        metrics.inc("anima_internal_errors_total")
        return _degraded_result(
            services,
            request_id=request_id,
            conversation=conversation,
            user_id=user_id,
            user_message_id=user_msg_id,
            first_pass_raw="",
            retrieved_lore_ids=(),
            retrieved_memory_ids=(),
            retry_count=0,
            error_code="internal_error",
            started=started,
            now=now,
            detail=str(exc),
        )


def _process_message(
    services: AppServices,
    *,
    user_id: str,
    conversation: ConversationRow,
    user_message: str,
    request_id: str,
    now: str,
    id_factory: Callable[[str], str],
    pack: LoadedPack,
    user_msg_id: str,
    started: float,
) -> MessageResult:
    config = services.config
    metrics = services.metrics
    repo = services.repo

    # --- retrieval (tenant filtering happens inside repo.search_memories) ---
    retrieval_started = time.monotonic()
    query_vec = (
        services.embedder.embed_query(user_message)
        if config.lore_retrieval_enabled or config.memory_retrieval_enabled
        else None
    )
    retrieved_lore: tuple[tuple[Any, float], ...] = ()
    if config.lore_retrieval_enabled:
        assert query_vec is not None
        lore_index = services.lore_indexes[conversation.persona_id]
        lore_hits = rank_by_similarity(
            lore_index.ids,
            lore_index.vectors,
            query_vec,
            top_k=config.lore_top_k,
            min_score=config.lore_min_score,
        )
        lore_by_id = pack.lore_by_id()
        retrieved_lore = tuple((lore_by_id[hit.id], hit.score) for hit in lore_hits)

    memories = ()
    retained_memory_hits: tuple[tuple[Any, float], ...] = ()
    if config.memory_retrieval_enabled:
        assert query_vec is not None
        mem_hits = repo.search_memories(
            user_id, conversation.persona_id, query_vec, config.memory_top_k
        )
        retained_memory_hits = tuple(
            (record, score)
            for record, score in mem_hits
            if config.memory_min_score is None or score >= config.memory_min_score
        )
        memories = tuple(record for record, _score in retained_memory_hits)
    retrieval_ms = (time.monotonic() - retrieval_started) * 1000
    metrics.observe_ms("anima_retrieval_ms", retrieval_ms)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in repo.list_messages(conversation.conversation_id)
        if m["message_id"] != user_msg_id
    ]

    built = build_context(
        pack,
        retrieved_lore=retrieved_lore,
        memories=memories,
        history=history,
        user_message=user_message,
        max_history_turns=config.max_history_turns,
    )

    # --- generation + bounded validator-triggered repair retries ---
    retrieved_lore_ids = tuple(fact.fact_id for fact, _ in retrieved_lore)
    retrieved_memory_ids = tuple(record.memory_id for record in memories)
    retrieved_lore_scores = tuple(float(score) for _fact, score in retrieved_lore)
    retrieved_memory_scores = tuple(float(score) for _record, score in retained_memory_hits)
    error_code: str | None = None
    retry_count = 0
    first_pass_raw = ""
    served_gen: GenerationResult | None = None

    try:
        gen1 = services.model.generate(built.system_prompt, built.messages)
        first_pass_raw = gen1.text
        served_gen = gen1
        parsed = parse_output(gen1.text)
        parse_status = "ok" if parsed.ok else "failed"
        terminal_repair = _repair_terminal_answer_close(gen1.text)
        if not parsed.ok and terminal_repair is not None:
            served_gen = replace(gen1, text=terminal_repair)
            parsed = parse_output(terminal_repair)
            parse_status = "terminal_closed"
            metrics.inc("anima_terminal_tag_repairs_total")
        elif not parsed.ok and config.max_format_retries > 0:
            retry_count = 1
            repair_messages = [
                *built.messages,
                {"role": "assistant", "content": gen1.text},
                {"role": "user", "content": REPAIR_INSTRUCTION},
            ]
            gen2 = services.model.generate(built.system_prompt, repair_messages)
            served_gen = gen2
            parsed = parse_output(gen2.text)
            parse_status = "repaired" if parsed.ok else "failed"

        if parsed.ok and parsed.state is not None and config.max_citation_retries > 0:
            invalid_lore, invalid_memory = _invalid_citations(
                parsed.state,
                retrieved_lore_ids=retrieved_lore_ids,
                retrieved_memory_ids=retrieved_memory_ids,
            )
            if invalid_lore or invalid_memory:
                retry_count += 1
                metrics.inc("anima_citation_repair_retries_total")
                repair_messages = [
                    *built.messages,
                    {"role": "assistant", "content": served_gen.text},
                    {
                        "role": "user",
                        "content": _render_citation_repair_instruction(
                            retrieved_lore_ids=retrieved_lore_ids,
                            retrieved_memory_ids=retrieved_memory_ids,
                            invalid_lore_ids=invalid_lore,
                            invalid_memory_ids=invalid_memory,
                        ),
                    },
                ]
                gen3 = services.model.generate(built.system_prompt, repair_messages)
                served_gen = gen3
                parsed = parse_output(gen3.text)
                parse_status = "citation_repaired" if parsed.ok else "failed"
    except ModelUnavailableError as exc:
        metrics.inc("anima_model_errors_total")
        return _degraded_result(
            services,
            request_id=request_id,
            conversation=conversation,
            user_id=user_id,
            user_message_id=user_msg_id,
            first_pass_raw=first_pass_raw,
            retrieved_lore_ids=retrieved_lore_ids,
            retrieved_memory_ids=retrieved_memory_ids,
            retry_count=retry_count,
            error_code="model_unavailable",
            started=started,
            now=now,
            detail=str(exc),
            retrieval_ms=retrieval_ms,
            context_hash=built.context_hash,
            retrieved_lore_scores=retrieved_lore_scores,
            retrieved_memory_scores=retrieved_memory_scores,
        )

    served_raw = served_gen.text
    served_answer = parsed.answer if parsed.answer is not None else DEGRADED_FALLBACK
    degraded = not parsed.ok
    answer_mode = parsed.state.answer_mode if parsed.state is not None else None

    invalid_lore: tuple[str, ...] = ()
    invalid_memory: tuple[str, ...] = ()
    memory_ops_requested = 0
    memory_ops_committed = 0
    rejected: tuple[dict[str, Any], ...] = ()

    if parsed.state is not None:
        invalid_lore, invalid_memory = _invalid_citations(
            parsed.state,
            retrieved_lore_ids=retrieved_lore_ids,
            retrieved_memory_ids=retrieved_memory_ids,
        )
        memory_ops_requested = len(parsed.state.memory_ops)

    if invalid_lore or invalid_memory:
        degraded = True
        error_code = "invalid_citation"
        served_answer = DEGRADED_FALLBACK
        metrics.inc("anima_invalid_citations_total")

    # Commit memory only from a fully valid, grounded served output. A parsed
    # answer with invented evidence ids is still untrusted.
    if not degraded and parsed.state is not None and parsed.state.memory_ops:
        if config.memory_write_enabled:
            committed, rejected = _commit_memory(
                services,
                user_id,
                conversation.persona_id,
                pack,
                parsed.state.memory_ops,
                user_msg_id,
                now,
            )
            memory_ops_committed = committed
        else:
            rejected = tuple(
                {"op": op.to_dict(), "reason": "memory_write_disabled"}
                for op in parsed.state.memory_ops
            )
            for item in rejected:
                repo.audit_rejected_op(user_id, conversation.persona_id, item["op"], item["reason"])

    assistant_msg_id: str | None = None
    if not degraded:
        assistant_msg_id = id_factory("msg")
        repo.append_message(
            conversation.conversation_id, assistant_msg_id, "assistant", served_answer
        )

    total_ms = (time.monotonic() - started) * 1000
    metrics.observe_ms("anima_total_ms", total_ms)
    metrics.observe_ms("anima_model_total_ms", served_gen.total_ms)
    if served_gen.ttft_ms is not None:
        metrics.observe_ms("anima_ttft_ms", served_gen.ttft_ms)
    metrics.inc("anima_input_tokens_total", served_gen.input_tokens)
    metrics.inc("anima_output_tokens_total", served_gen.output_tokens)
    if degraded:
        metrics.inc("anima_degraded_total")

    trace = _build_trace(
        services,
        request_id=request_id,
        conversation=conversation,
        user_id=user_id,
        user_message_id=user_msg_id,
        assistant_message_id=assistant_msg_id,
        first_pass_raw=first_pass_raw,
        served_raw=served_raw,
        parse_status=parse_status,
        retry_count=retry_count,
        retrieved_lore_ids=retrieved_lore_ids,
        retrieved_memory_ids=retrieved_memory_ids,
        used_lore_ids=parsed.state.used_lore_ids if parsed.state else (),
        used_memory_ids=parsed.state.used_memory_ids if parsed.state else (),
        invalid_lore=invalid_lore,
        invalid_memory=invalid_memory,
        memory_ops_requested=memory_ops_requested,
        memory_ops_committed=memory_ops_committed,
        rejected=rejected,
        degraded=degraded,
        error_code=error_code,
        gen=served_gen,
        retrieval_ms=retrieval_ms,
        context_hash=built.context_hash,
        total_ms=total_ms,
        now=now,
        retrieved_lore_scores=retrieved_lore_scores,
        retrieved_memory_scores=retrieved_memory_scores,
    )
    services.repo.insert_trace(request_id, trace)

    return MessageResult(
        request_id=request_id,
        served_answer=served_answer,
        first_pass_raw=first_pass_raw,
        served_raw=served_raw,
        parse_status=parse_status,
        answer_mode=answer_mode,
        retry_count=retry_count,
        degraded=degraded,
        error_code=error_code,
        used_lore_ids=parsed.state.used_lore_ids if parsed.state else (),
        used_memory_ids=parsed.state.used_memory_ids if parsed.state else (),
        retrieved_lore_ids=retrieved_lore_ids,
        retrieved_memory_ids=retrieved_memory_ids,
        invalid_lore_citations=invalid_lore,
        invalid_memory_citations=invalid_memory,
        memory_ops_requested=memory_ops_requested,
        memory_ops_committed=memory_ops_committed,
        memory_ops_rejected=rejected,
        user_message_id=user_msg_id,
        assistant_message_id=assistant_msg_id,
        ttft_ms=served_gen.ttft_ms,
        total_ms=total_ms,
        input_tokens=served_gen.input_tokens,
        output_tokens=served_gen.output_tokens,
        trace=trace,
    )


def _repair_terminal_answer_close(text: str) -> str | None:
    """Recover only a missing terminal ``</answer>`` delimiter.

    This does not repair JSON, invent answer text, tolerate extra blocks, or
    hide the raw first pass. It accepts a valid state block followed by one
    non-empty answer body whose only structural defect is a missing (possibly
    partially emitted) terminal close tag. The caller keeps the original text
    for first-pass scoring and records ``parse_status=terminal_closed``.
    """

    parsed = parse_output(text)
    if parsed.ok or parsed.state is None:
        return None
    if len(parsed.errors) != 2 or set(parsed.errors) != {
        "missing_answer_block",
        "extra_text_outside_blocks",
    }:
        return None

    stripped = text.rstrip()
    if stripped.count(_ANSWER_OPEN) != 1 or _ANSWER_CLOSE in stripped:
        return None
    _prefix, body = stripped.rsplit(_ANSWER_OPEN, 1)

    partial_length = 0
    for length in range(len(_ANSWER_CLOSE) - 1, 0, -1):
        if body.endswith(_ANSWER_CLOSE[:length]):
            partial_length = length
            break
    content = body[:-partial_length] if partial_length else body
    if not content.strip() or "<" in content:
        return None

    candidate = stripped + _ANSWER_CLOSE[partial_length:]
    return candidate if parse_output(candidate).ok else None


def _invalid_citations(
    state: Any,
    *,
    retrieved_lore_ids: Sequence[str],
    retrieved_memory_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    retrieved_lore_set = set(retrieved_lore_ids)
    retrieved_memory_set = set(retrieved_memory_ids)
    invalid_lore = tuple(i for i in state.used_lore_ids if i not in retrieved_lore_set)
    invalid_memory = tuple(i for i in state.used_memory_ids if i not in retrieved_memory_set)
    return invalid_lore, invalid_memory


def _render_citation_repair_instruction(
    *,
    retrieved_lore_ids: Sequence[str],
    retrieved_memory_ids: Sequence[str],
    invalid_lore_ids: Sequence[str],
    invalid_memory_ids: Sequence[str],
) -> str:
    return CITATION_REPAIR_INSTRUCTION.format(
        lore_ids=json.dumps(list(retrieved_lore_ids), ensure_ascii=False),
        memory_ids=json.dumps(list(retrieved_memory_ids), ensure_ascii=False),
        invalid_lore_ids=json.dumps(list(invalid_lore_ids), ensure_ascii=False),
        invalid_memory_ids=json.dumps(list(invalid_memory_ids), ensure_ascii=False),
    )


def _commit_memory(
    services: AppServices,
    user_id: str,
    persona_id: str,
    pack: LoadedPack,
    ops: Sequence[MemoryOp],
    user_msg_id: str,
    now: str,
) -> tuple[int, tuple[dict[str, Any], ...]]:
    # re-bind subject and source_message_id to authenticated server values
    rebound = [
        MemoryOp(
            op=op.op,
            subject="authenticated_user",
            predicate=op.predicate,
            object=op.object,
            source_message_id=user_msg_id,
        )
        for op in ops
    ]
    counter = {"n": 0}

    def mint() -> str:
        counter["n"] += 1
        return f"{user_msg_id}:mem:{counter['n']}"

    def planner(active):
        # runs inside the repo's per-tenant lock, so `active` is a fresh read:
        # a concurrent write is already committed and gets superseded, not duplicated
        plan = plan_memory_commits(
            rebound,
            user_id=user_id,
            persona_id=persona_id,
            allowlist=pack.safety.memory_predicate_allowlist,
            source_message_id=user_msg_id,
            active_memories=active,
            id_factory=mint,
            now=now,
        )
        embeddings = {
            commit.new_record.memory_id: services.embedder.embed(
                [render_memory_text(commit.new_record)]
            )[0]
            for commit in plan.commits
            if commit.new_record is not None
        }
        return plan, embeddings

    plan = services.repo.commit_memory(user_id, persona_id, planner, now=now)
    for rejected_op in plan.rejected:
        services.repo.audit_rejected_op(
            user_id, persona_id, rejected_op.op.to_dict(), rejected_op.reason
        )
    return len(plan.commits), tuple(
        {"op": r.op.to_dict(), "reason": r.reason} for r in plan.rejected
    )


def _degraded_result(
    services: AppServices,
    *,
    request_id: str,
    conversation: ConversationRow,
    user_id: str,
    user_message_id: str,
    first_pass_raw: str,
    retrieved_lore_ids: tuple[str, ...],
    retrieved_memory_ids: tuple[str, ...],
    retry_count: int,
    error_code: str,
    started: float,
    now: str,
    detail: str,
    retrieval_ms: float | None = None,
    context_hash: str | None = None,
    retrieved_lore_scores: tuple[float, ...] = (),
    retrieved_memory_scores: tuple[float, ...] = (),
) -> MessageResult:
    total_ms = (time.monotonic() - started) * 1000
    services.metrics.observe_ms("anima_total_ms", total_ms)
    services.metrics.inc("anima_degraded_total")
    trace = _build_trace(
        services,
        request_id=request_id,
        conversation=conversation,
        user_id=user_id,
        user_message_id=user_message_id,
        assistant_message_id=None,
        first_pass_raw=first_pass_raw,
        served_raw="",
        parse_status="failed",
        retry_count=retry_count,
        retrieved_lore_ids=retrieved_lore_ids,
        retrieved_memory_ids=retrieved_memory_ids,
        used_lore_ids=(),
        used_memory_ids=(),
        invalid_lore=(),
        invalid_memory=(),
        memory_ops_requested=0,
        memory_ops_committed=0,
        rejected=(),
        degraded=True,
        error_code=error_code,
        gen=None,
        retrieval_ms=retrieval_ms,
        context_hash=context_hash,
        total_ms=total_ms,
        now=now,
        error_detail=detail,
        retrieved_lore_scores=retrieved_lore_scores,
        retrieved_memory_scores=retrieved_memory_scores,
    )
    services.repo.insert_trace(request_id, trace)
    return MessageResult(
        request_id=request_id,
        served_answer=DEGRADED_FALLBACK,
        first_pass_raw=first_pass_raw,
        served_raw="",
        parse_status="failed",
        answer_mode=None,
        retry_count=retry_count,
        degraded=True,
        error_code=error_code,
        used_lore_ids=(),
        used_memory_ids=(),
        retrieved_lore_ids=retrieved_lore_ids,
        retrieved_memory_ids=retrieved_memory_ids,
        invalid_lore_citations=(),
        invalid_memory_citations=(),
        memory_ops_requested=0,
        memory_ops_committed=0,
        memory_ops_rejected=(),
        user_message_id=user_message_id,
        assistant_message_id=None,
        ttft_ms=None,
        total_ms=total_ms,
        input_tokens=0,
        output_tokens=0,
        trace=trace,
    )


def _build_trace(
    services: AppServices,
    *,
    request_id: str,
    conversation: ConversationRow,
    user_id: str,
    user_message_id: str,
    assistant_message_id: str | None,
    first_pass_raw: str,
    served_raw: str,
    parse_status: str,
    retry_count: int,
    retrieved_lore_ids: tuple[str, ...],
    retrieved_memory_ids: tuple[str, ...],
    used_lore_ids: tuple[str, ...],
    used_memory_ids: tuple[str, ...],
    invalid_lore: tuple[str, ...],
    invalid_memory: tuple[str, ...],
    memory_ops_requested: int,
    memory_ops_committed: int,
    rejected: tuple[dict[str, Any], ...],
    degraded: bool,
    error_code: str | None,
    gen: GenerationResult | None,
    retrieval_ms: float | None,
    context_hash: str | None,
    total_ms: float,
    now: str,
    error_detail: str | None = None,
    retrieved_lore_scores: tuple[float, ...] = (),
    retrieved_memory_scores: tuple[float, ...] = (),
) -> dict[str, Any]:
    info = services.model_info
    server_identity = services.model_server_identity
    server_identity_hash = (
        hashlib.sha256(
            json.dumps(
                server_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if server_identity is not None
        else None
    )
    return {
        "request_id": request_id,
        "conversation_id_hash": _hash(conversation.conversation_id),
        "user_id_hash": _hash(user_id),
        "persona_id": conversation.persona_id,
        "persona_version": conversation.persona_version,
        "persona_content_sha256": services.packs[conversation.persona_id].manifest.content_sha256,
        "base_model": info.base_model,
        "base_model_revision": info.base_model_revision,
        "adapter_id": info.adapter_id,
        "adapter_sha256": info.adapter_sha256,
        "embedding_model": getattr(services.embedder, "model_id", type(services.embedder).__name__),
        "embedding_model_revision": info.embedding_model_revision,
        "lore_retrieval_enabled": services.config.lore_retrieval_enabled,
        "memory_retrieval_enabled": services.config.memory_retrieval_enabled,
        "memory_write_enabled": services.config.memory_write_enabled,
        "prompt_version": info.prompt_version,
        "code_commit": services.code_commit,
        "runtime_config_hash": services.runtime_config_hash,
        "model_server_backend": (
            server_identity.get("backend") if server_identity is not None else None
        ),
        "model_server_identity_sha256": server_identity_hash,
        "model_server_identity_verified": server_identity is not None,
        "retrieved_lore_ids": list(retrieved_lore_ids),
        "retrieved_memory_ids": list(retrieved_memory_ids),
        "retrieved_lore_scores": list(retrieved_lore_scores),
        "retrieved_memory_scores": list(retrieved_memory_scores),
        "used_lore_ids": list(used_lore_ids),
        "used_memory_ids": list(used_memory_ids),
        "invalid_lore_citations": list(invalid_lore),
        "invalid_memory_citations": list(invalid_memory),
        "raw_first_pass": first_pass_raw,
        "served_raw": served_raw,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "context_hash": context_hash,
        "parse_status": parse_status,
        "retry_count": retry_count,
        "memory_ops_requested": memory_ops_requested,
        "memory_ops_committed": memory_ops_committed,
        "memory_ops_rejected": list(rejected),
        "degraded": degraded,
        "error_code": error_code,
        "error_detail": error_detail,
        "ttft_ms": gen.ttft_ms if gen else None,
        "model_total_ms": gen.total_ms if gen else None,
        "retrieval_ms": retrieval_ms,
        "total_ms": total_ms,
        "input_tokens": gen.input_tokens if gen else 0,
        "output_tokens": gen.output_tokens if gen else 0,
        "test_mode": services.test_mode,
        "created_at": now,
    }


def _hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _strict_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value
