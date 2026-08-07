"""Validated long-term memory operations.

`plan_memory_commits` turns model-emitted MemoryOps into an auditable commit
plan: subject and source_message_id are re-bound to authenticated server
values, predicates must sit in the pack allowlist, and the
one-active-row-per-(user, persona, subject, predicate) invariant is enforced
by superseding, never duplicating. Rejected ops keep their reason for the
audit trail. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from anima.persona.contracts.schemas import MemoryOp, MemoryRecord

MEMORY_SUBJECT = "authenticated_user"


class MemoryInvariantError(RuntimeError):
    """The store violated the single-active-row invariant."""


@dataclass(frozen=True)
class PlannedCommit:
    action: str  # add | supersede_add | delete
    op: MemoryOp
    new_record: MemoryRecord | None = None
    superseded_id: str | None = None
    deleted_id: str | None = None


@dataclass(frozen=True)
class RejectedOp:
    op: MemoryOp
    reason: str


@dataclass(frozen=True)
class CommitPlan:
    commits: tuple[PlannedCommit, ...]
    rejected: tuple[RejectedOp, ...]
    now: str


def plan_memory_commits(
    ops: Iterable[MemoryOp],
    *,
    user_id: str,
    persona_id: str,
    allowlist: Sequence[str],
    source_message_id: str,
    source_session_id: str | None = None,
    source_turn_id: str | None = None,
    active_memories: Sequence[MemoryRecord],
    id_factory: Callable[[], str],
    now: str,
) -> CommitPlan:
    if (source_session_id is None) != (source_turn_id is None):
        raise ValueError(
            "source_session_id and source_turn_id must either both be present or both be null"
        )
    active_by_predicate: dict[str, MemoryRecord] = {}
    for record in active_memories:
        if record.status != "active":
            continue
        if record.user_id != user_id or record.persona_id != persona_id:
            raise MemoryInvariantError(
                f"active memory {record.memory_id} belongs to ({record.user_id}, {record.persona_id}), "
                f"expected ({user_id}, {persona_id}); tenant filtering must happen before planning"
            )
        if record.predicate in active_by_predicate:
            raise MemoryInvariantError(
                f"two active rows for predicate {record.predicate!r}: "
                f"{active_by_predicate[record.predicate].memory_id}, {record.memory_id}"
            )
        active_by_predicate[record.predicate] = record

    commits: list[PlannedCommit] = []
    rejected: list[RejectedOp] = []
    allowed = frozenset(allowlist)

    for op in ops:
        if op.op == "noop":
            continue
        assert op.predicate is not None  # schema guarantees for add/update/delete
        if op.predicate not in allowed:
            rejected.append(RejectedOp(op=op, reason="predicate_not_allowlisted"))
            continue

        current = active_by_predicate.get(op.predicate)
        if op.op in ("add", "update"):
            if current is not None and current.object == op.object:
                rejected.append(RejectedOp(op=op, reason="duplicate_active_value"))
                continue
            new_record = MemoryRecord(
                memory_id=id_factory(),
                user_id=user_id,
                persona_id=persona_id,
                subject=MEMORY_SUBJECT,
                predicate=op.predicate,
                object=op.object or "",
                valid_from=None,
                valid_to=None,
                source_message_id=source_message_id,
                confidence=1.0,
                status="active",
                created_at=now,
                updated_at=now,
                source_session_id=source_session_id,
                source_turn_id=source_turn_id,
            )
            if current is not None:
                commits.append(
                    PlannedCommit(
                        action="supersede_add",
                        op=op,
                        new_record=new_record,
                        superseded_id=current.memory_id,
                    )
                )
            else:
                commits.append(PlannedCommit(action="add", op=op, new_record=new_record))
            active_by_predicate[op.predicate] = new_record
        elif op.op == "delete":
            if current is None:
                rejected.append(RejectedOp(op=op, reason="delete_target_missing"))
                continue
            pending = _pending_commit_for(commits, current.memory_id)
            if pending is not None:
                # the delete cancels an add planned earlier in this same turn
                commits.remove(pending)
                if pending.superseded_id is not None:
                    commits.append(
                        PlannedCommit(action="delete", op=op, deleted_id=pending.superseded_id)
                    )
            else:
                commits.append(PlannedCommit(action="delete", op=op, deleted_id=current.memory_id))
            del active_by_predicate[op.predicate]

    return CommitPlan(commits=tuple(commits), rejected=tuple(rejected), now=now)


def apply_plan(
    active_memories: Sequence[MemoryRecord], plan: CommitPlan
) -> tuple[MemoryRecord, ...]:
    """Reference in-memory application of a plan; the SQL repository mirrors this."""

    records = {record.memory_id: record for record in active_memories}
    for commit in plan.commits:
        if commit.superseded_id is not None:
            old = records[commit.superseded_id]
            records[commit.superseded_id] = replace(old, status="superseded", updated_at=plan.now)
        if commit.deleted_id is not None:
            old = records[commit.deleted_id]
            records[commit.deleted_id] = replace(old, status="deleted", updated_at=plan.now)
        if commit.new_record is not None:
            records[commit.new_record.memory_id] = commit.new_record
    return tuple(records.values())


def _pending_commit_for(commits: list[PlannedCommit], memory_id: str) -> PlannedCommit | None:
    for commit in commits:
        if commit.new_record is not None and commit.new_record.memory_id == memory_id:
            return commit
    return None
