import pytest
from pydantic import ValidationError

from swarm_forge.contracts import (
    AgentRole,
    ArtifactKind,
    ArtifactRecord,
    AuditEvent,
    EventType,
    TaskBudget,
    TaskSpec,
    TaskStatus,
)


def make_task(**overrides: object) -> TaskSpec:
    data: dict[str, object] = {
        "run_id": "run-test-001",
        "agent_role": AgentRole.IMPLEMENTER,
        "goal": "Implementasikan fungsi contoh",
        "output_contract": "code_patch.v1",
        "allowed_tools": ("read_repo", "run_tests"),
    }
    data.update(overrides)
    return TaskSpec.model_validate(data)

def test_task_gets_id_and_default_budget() -> None:
    task = make_task()

    assert task.task_id
    assert task.budget.max_attempts == 3
    assert task.status == TaskStatus.CREATED

def test_blank_goal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_task(goal="   ")

def test_unknown_task_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_task(unapproved_shell=True)

def test_duplicate_dependencies_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_task(depends_on=("task-a", "task-a"))

def test_budget_has_upper_bound() -> None:
    with pytest.raises(ValidationError):
        TaskBudget(max_attempts=100)

def test_artifact_requires_sha256() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord(
            task_id="task-001",
            kind=ArtifactKind.CODE_PATCH,
            uri="artifacts/example.patch",
            sha256="not-a-hash",
            schema_version="1.0",
        )

def test_audit_event_has_correlation_id() -> None:
    event = AuditEvent(
        run_id="run-test-001",
        task_id="task-001",
        event_type=EventType.TASK_CREATED,
        correlation_id="corr-test-001",
        payload={"source": "unit_test"},
    )

    assert event.event_id
    assert event.prev_hash is None
    assert event.event_hash is None

def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_task(status="magic_success")
