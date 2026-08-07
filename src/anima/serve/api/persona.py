"""FastAPI persona API.

user_id is always derived server-side from the Bearer API key — never read
from the request body. Conversation and memory access is ownership-checked
against that authenticated user. A formal deployment refuses to start in
ANIMA_TEST_MODE or with a stubbed backend. Streaming uses SSE; the
answer is streamed only after the <anima_state> block has been parsed, so no
unvalidated content ever reaches the user.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from anima.persona.contracts.schemas import MemoryRecord
from anima.persona.runtime.retrieval import render_memory_text
from anima.serve.core.repositories import ConversationRow
from anima.serve.core.runtime import AppServices, InputTooLongError, MessageResult, handle_message
from anima.serve.inference.model_client import ModelUnavailableError


class CreateConversationRequest(BaseModel):
    persona_id: str = Field(min_length=1)


class MessageRequest(BaseModel):
    content: str = Field(min_length=1)
    stream: bool = False


class FeedbackRequest(BaseModel):
    conversation_id: str | None = None
    message_id: str | None = None
    kind: str
    comment: str | None = None


class EvalUserRequest(BaseModel):
    user_id: str = Field(min_length=8, max_length=128)
    api_key: str = Field(min_length=16, max_length=256)


class EvalMemoryFixtureRequest(BaseModel):
    persona_id: str = Field(min_length=1)
    fixture_id: str = Field(min_length=1, max_length=128)
    predicate: str = Field(min_length=1, max_length=64)
    object: str = Field(min_length=1, max_length=1000)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def create_app(services: AppServices, *, formal: bool = False) -> FastAPI:
    # : ANIMA_TEST_MODE *存在即拒跑* — membership, not truthiness, so an
    # empty-string value (ANIMA_TEST_MODE=) still refuses a formal start.
    if formal and (services.test_mode or "ANIMA_TEST_MODE" in os.environ):
        raise RuntimeError(
            "refusing to start a formal deployment with a test backend or ANIMA_TEST_MODE set"
        )

    app = FastAPI(title="Anima Persona API", version=services.model_info.prompt_version)

    def authenticate(authorization: str = Header(default="")) -> str:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        key = authorization[len("Bearer ") :].strip()
        user_id = services.repo.user_id_for_key(_key_hash(key))
        if user_id is None:
            raise HTTPException(status_code=401, detail="unknown api key")
        return user_id

    def authenticate_eval(x_anima_eval_token: str = Header(default="")) -> None:
        expected = services.eval_token_hash
        supplied = _key_hash(x_anima_eval_token) if x_anima_eval_token else ""
        if expected is None or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=404, detail="not found")

    def _owned_conversation(conversation_id: str, user_id: str) -> ConversationRow:
        row = services.repo.get_conversation(conversation_id)
        if row is None or row.user_id != user_id or row.deleted:
            raise HTTPException(status_code=404, detail="conversation not found")
        return row

    @app.post("/v1/conversations")
    def create_conversation(
        body: CreateConversationRequest, user_id: str = Depends(authenticate)
    ) -> dict[str, Any]:
        pack = services.packs.get(body.persona_id)
        if pack is None:
            raise HTTPException(status_code=404, detail="unknown persona")
        conversation_id = _id("conv")
        services.repo.create_conversation(
            ConversationRow(
                conversation_id=conversation_id,
                user_id=user_id,
                persona_id=body.persona_id,
                persona_version=pack.manifest.version,
                deleted=False,
            )
        )
        return {
            "conversation_id": conversation_id,
            "persona_id": body.persona_id,
            "persona_version": pack.manifest.version,
        }

    @app.post("/v1/conversations/{conversation_id}/messages")
    def post_message(
        conversation_id: str, body: MessageRequest, user_id: str = Depends(authenticate)
    ):
        conversation = _owned_conversation(conversation_id, user_id)
        request_id = _id("req")
        try:
            result = handle_message(
                services,
                user_id=user_id,
                conversation=conversation,
                user_message=body.content,
                request_id=request_id,
                now=_now(),
                id_factory=_id,
            )
        except InputTooLongError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        # degraded turns share one failure contract across streaming and
        # non-streaming: HTTP 503, so an SLO monitor keyed on status can't count
        # a streamed model failure as success.
        if result.degraded:
            return _json_response(_result_payload(result), status_code=503)
        if body.stream:
            return StreamingResponse(_sse(result), media_type="text/event-stream")
        return _result_payload(result)

    @app.get("/v1/conversations/{conversation_id}")
    def get_conversation(
        conversation_id: str, user_id: str = Depends(authenticate)
    ) -> dict[str, Any]:
        conversation = _owned_conversation(conversation_id, user_id)
        return {
            "conversation_id": conversation.conversation_id,
            "persona_id": conversation.persona_id,
            "persona_version": conversation.persona_version,
            "messages": services.repo.list_messages(conversation_id),
        }

    @app.delete("/v1/conversations/{conversation_id}")
    def delete_conversation(
        conversation_id: str, user_id: str = Depends(authenticate)
    ) -> dict[str, Any]:
        _owned_conversation(conversation_id, user_id)
        services.repo.mark_conversation_deleted(conversation_id)
        return {"deleted": conversation_id}

    @app.get("/v1/memories")
    def list_memories(persona_id: str, user_id: str = Depends(authenticate)) -> dict[str, Any]:
        records = services.repo.active_memories(user_id, persona_id)
        return {"memories": [r.to_dict() for r in records]}

    @app.delete("/v1/memories/{memory_id}")
    def delete_memory(memory_id: str, user_id: str = Depends(authenticate)) -> dict[str, Any]:
        record = services.repo.get_memory(memory_id)
        if record is None or record.user_id != user_id:
            raise HTTPException(status_code=404, detail="memory not found")
        from anima.persona.runtime.memory import CommitPlan, PlannedCommit

        plan = CommitPlan(
            commits=(
                PlannedCommit(
                    action="delete", op=_synthetic_delete_op(record), deleted_id=memory_id
                ),
            ),
            rejected=(),
            now=_now(),
        )
        services.repo.apply_commit_plan(plan, {})
        return {"deleted": memory_id}

    @app.get("/v1/personas")
    def list_personas() -> dict[str, Any]:
        return {
            "personas": [
                {
                    "persona_id": p.manifest.persona_id,
                    "version": p.manifest.version,
                    "public": p.manifest.public,
                    "human_reviewed": p.manifest.human_reviewed,
                }
                for p in services.packs.values()
            ]
        }

    @app.post("/v1/feedback")
    def post_feedback(
        body: FeedbackRequest, user_id: str = Depends(authenticate)
    ) -> dict[str, Any]:
        if body.kind not in ("up", "down", "ooc", "fact_error", "safety_error", "other"):
            raise HTTPException(status_code=422, detail="invalid feedback kind")
        if body.message_id is not None and body.conversation_id is None:
            raise HTTPException(status_code=422, detail="message_id requires conversation_id")
        if body.conversation_id is not None:
            # feedback may only be attached to the caller's own conversation/message,
            # else one user could poison another's data-flywheel signal
            _owned_conversation(body.conversation_id, user_id)
            if body.message_id is not None:
                owned = {m["message_id"] for m in services.repo.list_messages(body.conversation_id)}
                if body.message_id not in owned:
                    raise HTTPException(status_code=404, detail="message not in conversation")
        services.repo.insert_feedback(
            user_id, body.conversation_id, body.message_id, body.kind, body.comment
        )
        return {"ok": True}

    @app.post("/v1/eval/users")
    def create_eval_user(
        body: EvalUserRequest, _eval: None = Depends(authenticate_eval)
    ) -> dict[str, Any]:
        services.repo.create_user(body.user_id, _key_hash(body.api_key), is_eval=True)
        return {"user_id": body.user_id}

    @app.post("/v1/eval/memories")
    def seed_eval_memory(
        body: EvalMemoryFixtureRequest,
        user_id: str = Depends(authenticate),
        _eval: None = Depends(authenticate_eval),
    ) -> dict[str, Any]:
        pack = services.packs.get(body.persona_id)
        if pack is None:
            raise HTTPException(status_code=404, detail="unknown persona")
        if body.predicate not in pack.safety.memory_predicate_allowlist:
            raise HTTPException(status_code=422, detail="memory predicate not allowlisted")
        now = _now()
        record = MemoryRecord(
            memory_id=_id("eval_mem"),
            user_id=user_id,
            persona_id=body.persona_id,
            subject="authenticated_user",
            predicate=body.predicate,
            object=body.object,
            valid_from=None,
            valid_to=None,
            source_message_id=f"eval_fixture:{body.fixture_id}",
            confidence=1.0,
            status="active",
            created_at=now,
            updated_at=now,
        )
        [embedding] = services.embedder.embed([render_memory_text(record)])
        services.repo.seed_memory(record, embedding)
        return {"fixture_id": body.fixture_id, "memory_id": record.memory_id}

    @app.get("/v1/eval/feedback")
    def export_eval_feedback(
        limit: int = 1000,
        _eval: None = Depends(authenticate_eval),
    ) -> dict[str, Any]:
        if not 1 <= limit <= 5000:
            raise HTTPException(status_code=422, detail="limit must be in [1, 5000]")
        rows = services.repo.list_feedback_for_eval(limit)
        return {"feedback": rows, "rows": len(rows)}

    @app.get("/v1/traces/{request_id}")
    def get_trace(request_id: str, user_id: str = Depends(authenticate)) -> dict[str, Any]:
        """Return the caller's own trace for eval/audit replay.

        The endpoint exposes model outputs and retrieval ids, never the rendered
        system prompt. Ownership is checked with the same short hash persisted
        by the runtime; an unknown or foreign trace is indistinguishable.
        """

        trace = services.repo.get_trace(request_id)
        if trace is None or trace.get("user_id_hash") != _short_hash(user_id):
            raise HTTPException(status_code=404, detail="trace not found")
        return trace

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        checks = {
            "db": services.repo.ping(),
            "packs": bool(services.packs) and bool(services.lore_indexes),
            "model": _model_ready(services, formal=formal),
        }
        ready = all(checks.values())
        return _json_response({"ready": ready, "checks": checks}, status_code=200 if ready else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return services.metrics.render()

    return app


def _sse(result: MessageResult):
    for chunk in _chunk_text(result.served_answer):
        yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
    meta = {
        "done": True,
        "request_id": result.request_id,
        "answer_mode": result.answer_mode,
        "degraded": result.degraded,
        "persona_version": result.trace.get("persona_version"),
    }
    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 12):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _result_payload(result: MessageResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "answer": result.served_answer,
        "answer_mode": result.answer_mode,
        "persona_version": result.trace.get("persona_version"),
        "model": {
            "base_model": result.trace.get("base_model"),
            "adapter_id": result.trace.get("adapter_id"),
        },
        "degraded": result.degraded,
        "error_code": result.error_code,
        "used_lore_ids": list(result.used_lore_ids),
        "used_memory_ids": list(result.used_memory_ids),
        "memory_ops_committed": result.memory_ops_committed,
    }


def _json_response(payload: dict[str, Any], *, status_code: int):
    from fastapi.responses import JSONResponse

    return JSONResponse(content=payload, status_code=status_code)


def _model_ready(services: AppServices, *, formal: bool) -> bool:
    if services.test_mode and not formal:
        return True
    try:
        served = services.model.list_models()  # type: ignore[attr-defined]
    except (ModelUnavailableError, AttributeError):
        return bool(services.test_mode and not formal)
    expected_served = services.model_info.adapter_id or services.model_info.base_model
    if expected_served not in served:
        return False
    if not formal:
        return True
    try:
        identity = services.model.model_identity()  # type: ignore[attr-defined]
    except (ModelUnavailableError, AttributeError):
        return False
    expected_identity = {
        "base_model": services.model_info.base_model,
        "base_model_revision": services.model_info.base_model_revision,
        "adapter_id": services.model_info.adapter_id,
        "adapter_sha256": services.model_info.adapter_sha256,
        "served_model": expected_served,
        "code_commit": services.code_commit,
    }
    return identity.get("backend") == "transformers_peft" and all(
        identity.get(key) == value for key, value in expected_identity.items()
    )


def _synthetic_delete_op(record):
    from anima.persona.contracts.schemas import MemoryOp

    return MemoryOp(
        op="delete",
        subject="authenticated_user",
        predicate=record.predicate,
        object=None,
        source_message_id="user_delete_request",
    )


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
