from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self
from uuid import UUID, uuid4, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from swarm_forge.contracts.claim import AttemptStatus
from swarm_forge.contracts.operation import OperationContext
from swarm_forge.contracts.task import TaskStatus


def utc_now() -> datetime:
    return datetime.now(UTC)

class EventType(StrEnum):
    RUN_CREATED = "run_created"
    TASK_CREATED = "task_created"
    TASK_STATUS_CHANGED = "task_status_changed"
    TASK_ATTEMPTED = "task_attempted"
    TASK_ATTEMPT_STARTED = "task_attempt_started"
    TASK_ATTEMPT_ENDED = "task_attempt_ended"
    ARTIFACT_CREATED = "artifact_created"
    POLICY_ALLOWED = "policy_allowed"
    POLICY_DENIED = "policy_denied"
    TEST_COMPLETED = "test_completed"
    HUMAN_APPROVAL = "human_approval"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    event_type: EventType

    timestamp: datetime = Field(default_factory=utc_now)
    correlation_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    prev_hash: str | None = None
    event_hash: str | None = None


_LIFECYCLE_EVENT_NAMESPACE = UUID("55de08b5-79a0-46e2-bbc2-1b6d5d474365")
_FAILED_TO_READY_CAUSES = frozenset({"retry_task", "recover_expired_attempts"})


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Payload numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Payload object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError("Payload must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _event_id_for(operation_id: str, discriminator: str) -> str:
    identity = json.dumps([operation_id, discriminator], separators=(",", ":"))
    return str(uuid5(_LIFECYCLE_EVENT_NAMESPACE, identity))


class LifecycleEvent(BaseModel):
    """Immutable lifecycle event intent, ready for future outbox persistence.

    This is separate from ``AuditEvent`` so lifecycle idempotency metadata does
    not alter the persisted audit envelope or its hash format. The outbox must
    enforce event-ID uniqueness, reject an event ID reused with different
    immutable content, enforce uniqueness of (operation_id, idempotency_key),
    and reject idempotency-key reuse with conflicting operation content.
    Individual model instances cannot guarantee cross-instance uniqueness.

    For ``FAILED`` to ``READY``, ``retry_task`` and
    ``recover_expired_attempts`` are canonical operation-cause values
    introduced by this LifecycleEvent contract. They are not
    ``transition_cause`` values previously stored or emitted by TaskStore.
    ``retry_task`` denotes an explicit retry through
    ``TaskStore.retry_task()`` and does not require attempt identity.
    ``recover_expired_attempts`` denotes recovery through
    ``TaskStore.recover_expired_attempts()`` and requires the identity of the
    recovered attempt. TaskStore records ``reason_code='lease_expired'`` as
    the reason an attempt ended; that reason code is distinct from the
    lifecycle event's ``transition_cause``. Integration code must explicitly
    map each TaskStore operation to the corresponding contract value and must
    not assume TaskStore already stores or emits ``transition_cause``.

    Supported trust-boundary APIs are ``for_operation``, ``model_validate``,
    ``model_validate_json``, and the validated ``model_copy`` override. A
    consumer must revalidate with ``LifecycleEvent.model_validate(value)`` or
    ``model_validate_json`` before persistence or processing. Pydantic's
    ``model_construct`` and direct ``BaseModel.model_copy`` calls bypass this
    contract and must not be used at lifecycle trust boundaries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    event_discriminator: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    source: str = Field(min_length=1)

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    event_type: EventType
    occurred_at: datetime = Field(default_factory=utc_now)

    attempt_id: str | None = None
    attempt_no: int | None = Field(default=None, ge=1)
    worker_id: str | None = None
    claim_request_id: str | None = None
    previous_state: TaskStatus | None = None
    new_state: TaskStatus | None = None
    attempt_outcome: AttemptStatus | None = None
    transition_cause: str | None = None
    reason_code: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "event_id",
        "idempotency_key",
        "operation_id",
        "event_discriminator",
        "correlation_id",
        "source",
        "run_id",
        "task_id",
        "attempt_id",
        "worker_id",
        "claim_request_id",
        "transition_cause",
        "reason_code",
    )
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @field_serializer("payload")
    def serialize_payload(self, value: dict[str, Any]) -> dict[str, Any]:
        return _thaw_json(value)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a fully revalidated copy; payloads are always detached.

        Unlike BaseModel.model_copy, updates are validated. Identity fields
        must continue to agree with the deterministic event-ID rule. The
        ``deep`` argument is retained for API compatibility; this method
        reconstructs all fields and deeply freezes payloads on every copy.
        """
        data = self.model_dump(mode="python")
        if update:
            data.update(update)
        return type(self).model_validate(data)

    @model_validator(mode="after")
    def validate_event_shape(self) -> LifecycleEvent:
        expected_event_id = _event_id_for(
            self.operation_id, self.event_discriminator
        )
        if self.event_id != expected_event_id:
            raise ValueError("event_id must match operation_id and event_discriminator")
        object.__setattr__(self, "event_id", expected_event_id)
        object.__setattr__(self, "payload", _freeze_json(self.payload))

        if (self.attempt_id is None) != (self.attempt_no is None):
            raise ValueError("attempt_id and attempt_no must be supplied together")

        if self.event_type == EventType.TASK_CREATED:
            if (
                self.previous_state is not None
                or self.new_state != TaskStatus.CREATED
                or self._has_attempt_fields()
                or self.attempt_outcome is not None
                or self.transition_cause is not None
                or self.reason_code is not None
                or self.worker_id is not None
                or self.claim_request_id is not None
            ):
                raise ValueError(
                    "TASK_CREATED requires CREATED state and no attempt fields"
                )
        elif self.event_type == EventType.TASK_STATUS_CHANGED:
            if self.previous_state is None or self.new_state is None:
                raise ValueError("TASK_STATUS_CHANGED requires both states")
            if not self.transition_cause:
                raise ValueError("TASK_STATUS_CHANGED requires transition_cause")
            if (
                self.attempt_outcome is not None
                or self.worker_id is not None
                or self.claim_request_id is not None
            ):
                raise ValueError("TASK_STATUS_CHANGED contains unrelated fields")
            if (
                self.previous_state == TaskStatus.READY
                and self.new_state == TaskStatus.RUNNING
            ):
                raise ValueError(
                    "READY to RUNNING must use TASK_ATTEMPT_STARTED"
                )
            if (
                self.previous_state == TaskStatus.FAILED
                and self.new_state == TaskStatus.READY
            ):
                if self.transition_cause not in _FAILED_TO_READY_CAUSES:
                    raise ValueError(
                        "FAILED to READY requires transition_cause 'retry_task' "
                        "or 'recover_expired_attempts'"
                    )
                if (
                    self.transition_cause == "recover_expired_attempts"
                    and not self._has_attempt_fields()
                ):
                    raise ValueError(
                        "Recovery FAILED to READY requires attempt identity"
                    )
            if self.previous_state in {
                TaskStatus.RUNNING,
                TaskStatus.TESTING,
                TaskStatus.REVIEW,
            } and not self._has_attempt_fields():
                raise ValueError(
                    "Active attempt transitions require attempt identity"
                )
            # Import locally to keep direct state_machine imports from cycling
            # through contracts.__init__ back into this module.
            from swarm_forge.workflow.state_machine import validate_transition

            validate_transition(self.previous_state, self.new_state)
        elif self.event_type == EventType.TASK_ATTEMPT_STARTED:
            if (
                self.attempt_id is None
                or self.attempt_no is None
                or self.worker_id is None
                or self.claim_request_id is None
                or self.attempt_outcome is not None
                or self.transition_cause is not None
                or self.reason_code is not None
            ):
                raise ValueError("TASK_ATTEMPT_STARTED requires claim context only")
            if (
                self.previous_state != TaskStatus.READY
                or self.new_state != TaskStatus.RUNNING
            ):
                raise ValueError(
                    "TASK_ATTEMPT_STARTED must describe READY to RUNNING"
                )
        elif self.event_type == EventType.TASK_ATTEMPT_ENDED:
            # This event records only the attempt outcome. Any task status
            # transition is represented by a separate TASK_STATUS_CHANGED.
            if self.attempt_id is None or self.attempt_no is None:
                raise ValueError("TASK_ATTEMPT_ENDED requires attempt identity")
            if self.attempt_outcome is None or self.attempt_outcome == AttemptStatus.RUNNING:
                raise ValueError("TASK_ATTEMPT_ENDED requires a terminal outcome")
            if (
                self.previous_state is not None
                or self.new_state is not None
                or self.transition_cause is not None
                or self.worker_id is not None
                or self.claim_request_id is not None
            ):
                raise ValueError(
                    "TASK_ATTEMPT_ENDED records only the attempt outcome"
                )
        else:
            raise ValueError("LifecycleEvent only supports lifecycle event types")
        return self

    def _has_attempt_fields(self) -> bool:
        return self.attempt_id is not None or self.attempt_no is not None

    @classmethod
    def for_operation(
        cls,
        *,
        context: OperationContext,
        idempotency_key: str,
        event_discriminator: str,
        run_id: str,
        task_id: str,
        event_type: EventType,
        **event_fields: Any,
    ) -> LifecycleEvent:
        """Build a lifecycle event with explicit operation context.

        Event identity is UUIDv5(operation_id, event_discriminator), so
        rebuilding the same event is stable and distinct events in one
        operation use distinct discriminators. ``idempotency_key`` is a
        separate logical-operation key, scoped by operation_id. The outbox
        must enforce uniqueness and reject reuse with different event content.
        """
        return cls(
            event_id=_event_id_for(context.operation_id, event_discriminator),
            idempotency_key=idempotency_key,
            operation_id=context.operation_id,
            event_discriminator=event_discriminator,
            correlation_id=context.correlation_id,
            source=context.source,
            run_id=run_id,
            task_id=task_id,
            event_type=event_type,
            **event_fields,
        )
