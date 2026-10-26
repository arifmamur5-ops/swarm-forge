from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from swarm_forge.contracts.task import TaskSpec


class ClaimSkipReason(StrEnum):
    ACTIVE_ATTEMPT_EXISTS = "active_attempt_exists"
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    DEPENDENCY_CROSS_RUN = "dependency_cross_run"
    DEPENDENCY_INVALID = "dependency_invalid"
    DEPENDENCY_MISSING = "dependency_missing"
    DEPENDENCY_NOT_ACCEPTED = "dependency_not_accepted"
    SELF_DEPENDENCY = "self_dependency"
    TASK_INVALID = "task_invalid"


class AttemptStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


class SkippedClaimCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    reason: ClaimSkipReason


class TaskClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task: TaskSpec
    attempt_id: str = Field(min_length=1)
    attempt_no: int = Field(ge=1)
    worker_id: str = Field(min_length=1)
    lease_expires_at: datetime

    @field_validator("worker_id")
    @classmethod
    def reject_blank_worker_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("worker_id must not be blank")
        return value


class ClaimResult(BaseModel):
    """Immutable receipt for a claim request, including an empty result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: TaskClaim | None = None
    skipped_candidates: tuple[SkippedClaimCandidate, ...] = ()
