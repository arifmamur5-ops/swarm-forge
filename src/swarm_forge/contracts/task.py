from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)

class AgentRole(StrEnum):
    ANALYST = "analyst"
    ARCHITECT = "architect"
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    QA = "qa"
    SECURITY_REVIEWER = "security_reviewer"
    INTEGRATOR = "integrator"

class TaskStatus(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    READY = "ready"
    RUNNING = "running"
    TESTING = "testing"
    REVIEW = "review"
    ACCEPTED = "accepted"
    BLOCKED = "blocked"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

class TaskBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(default=3, ge=1, le=5)
    timeout_seconds: int = Field(default=300, ge=1, le=1800)
    max_changed_files: int = Field(default=10, ge=1, le=50)
    max_input_tokens: int = Field(default=8000, ge=1, le=50000)
    max_output_tokens: int = Field(default=4000, ge=1, le=20000)

class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    run_id: str = Field(min_length=1)
    parent_task_id: str | None = None

    agent_role: AgentRole
    goal: str = Field(min_length=1, max_length=2000)

    depends_on: tuple[str, ...] = ()
    input_refs: tuple[str, ...] = ()

    output_contract: str = Field(min_length=1, max_length=200)
    allowed_tools: tuple[str, ...] = ()
    budget: TaskBudget = Field(default_factory=TaskBudget)

    status: TaskStatus = TaskStatus.CREATED
    attempt: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("task_id", "run_id", "output_contract")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator("goal")
    @classmethod
    def reject_blank_goal(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Task goal must not be blank")
        return value

    @field_validator("depends_on")
    @classmethod
    def reject_duplicate_dependencies(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Duplicate dependencies are not allowed")
        return value

class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    success: bool
    summary: str = Field(min_length=1, max_length=2000)
    artifact_refs: tuple[str, ...] = ()
    test_command: str | None = None
    test_exit_code: int | None = None
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
