from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)

class EventType(StrEnum):
    RUN_CREATED = "run_created"
    TASK_CREATED = "task_created"
    TASK_STATUS_CHANGED = "task_status_changed"
    TASK_ATTEMPTED = "task_attempted"
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
