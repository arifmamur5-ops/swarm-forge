from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from swarm_forge.contracts.claim import AttemptStatus, TaskClaim
from swarm_forge.contracts.task import AgentRole, TaskBudget, TaskSpec, TaskStatus
from swarm_forge.orchestrator.workflow_engine import (
    WorkflowEngine,
    WorkflowValidationError,
)
from swarm_forge.storage.task_store import TaskStore


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(db_path=tmp_path / "workflow.db")


@pytest.fixture
def engine(store: TaskStore) -> WorkflowEngine:
    return WorkflowEngine(store)


def make_task(
    task_id: str,
    *,
    run_id: str = "run-1",
    depends_on: tuple[str, ...] = (),
    created_at: datetime | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        run_id=run_id,
        agent_role=AgentRole.PLANNER,
        goal=f"Goal for {task_id}",
        depends_on=depends_on,
        output_contract="test-output",
        created_at=created_at or datetime.now(UTC),
    )


def transition_claim(
    engine: WorkflowEngine,
    claim: TaskClaim,
    expected_task_status: TaskStatus,
    new_task_status: TaskStatus,
) -> TaskSpec:
    return engine.transition_attempt_status(
        task_id=claim.task.task_id,
        attempt_id=claim.attempt_id,
        attempt_no=claim.attempt_no,
        expected_attempt_status=AttemptStatus.RUNNING,
        expected_task_status=expected_task_status,
        new_task_status=new_task_status,
    )


def test_task_without_dependencies_becomes_ready_and_can_be_claimed(
    engine: WorkflowEngine,
    store: TaskStore,
) -> None:
    task = make_task("task-1")

    engine.register_task(task)
    validated = engine.validate_task(task.task_id)
    assert validated.status == TaskStatus.VALIDATED

    changed = engine.refresh_ready_tasks()
    assert [item.task_id for item in changed] == ["task-1"]

    claim_result = engine.claim_next_task(
        worker_id="worker-1", claim_request_id="engine-claim-1"
    )
    claimed = claim_result.claim
    assert claimed is not None
    assert claimed.task.task_id == "task-1"
    assert claimed.task.status == TaskStatus.RUNNING
    assert claimed.attempt_no == 1
    assert store.get_task("task-1").status == TaskStatus.RUNNING
    assert store.get_task("task-1").attempt == 1


def test_dependent_task_waits_until_dependency_is_accepted(
    engine: WorkflowEngine,
    store: TaskStore,
) -> None:
    parent = make_task("parent")
    child = make_task("child", depends_on=("parent",))

    engine.register_task(parent)
    engine.register_task(child)
    engine.validate_task("parent")
    engine.validate_task("child")

    changed = engine.refresh_ready_tasks()
    assert [item.task_id for item in changed] == ["parent"]
    assert store.get_task("parent").status == TaskStatus.READY
    assert store.get_task("child").status == TaskStatus.VALIDATED

    # Complete the parent's lifecycle through the existing state machine.
    parent_claim = engine.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="parent-claim"
    )
    assert parent_claim.claim is not None
    assert parent_claim.claim.task.task_id == "parent"
    transition_claim(
        engine, parent_claim.claim, TaskStatus.RUNNING, TaskStatus.TESTING
    )
    transition_claim(
        engine, parent_claim.claim, TaskStatus.TESTING, TaskStatus.REVIEW
    )
    transition_claim(
        engine, parent_claim.claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED
    )

    changed = engine.refresh_ready_tasks()
    assert [item.task_id for item in changed] == ["child"]
    assert store.get_task("child").status == TaskStatus.READY


def test_missing_dependency_leaves_task_created(
    engine: WorkflowEngine,
    store: TaskStore,
) -> None:
    engine.register_task(make_task("child", depends_on=("missing",)))

    with pytest.raises(WorkflowValidationError, match="does not exist"):
        engine.validate_task("child")

    assert store.get_task("child").status == TaskStatus.CREATED


def test_cross_run_dependency_is_rejected(
    engine: WorkflowEngine,
    store: TaskStore,
) -> None:
    engine.register_task(make_task("parent", run_id="run-A"))
    engine.register_task(make_task("child", run_id="run-B", depends_on=("parent",)))

    with pytest.raises(WorkflowValidationError, match="different run"):
        engine.validate_task("child")

    assert store.get_task("child").status == TaskStatus.CREATED


def test_self_dependency_is_rejected(
    engine: WorkflowEngine,
    store: TaskStore,
) -> None:
    engine.register_task(make_task("loop", depends_on=("loop",)))

    with pytest.raises(WorkflowValidationError, match="depend on itself"):
        engine.validate_task("loop")

    assert store.get_task("loop").status == TaskStatus.CREATED


def test_dependency_cycle_is_rejected(
    engine: WorkflowEngine,
    store: TaskStore,
) -> None:
    engine.register_task(make_task("task-A", depends_on=("task-B",)))
    engine.register_task(make_task("task-B", depends_on=("task-A",)))

    with pytest.raises(WorkflowValidationError, match="cycle detected"):
        engine.validate_task("task-A")

    assert store.get_task("task-A").status == TaskStatus.CREATED


def test_ready_tasks_have_deterministic_order(
    engine: WorkflowEngine,
) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    engine.register_task(make_task("task-B", created_at=timestamp))
    engine.register_task(make_task("task-A", created_at=timestamp))

    engine.validate_task("task-B")
    engine.validate_task("task-A")
    engine.refresh_ready_tasks()

    assert [task.task_id for task in engine.get_ready_tasks()] == ["task-A", "task-B"]


def test_recovery_preserves_persisted_states(tmp_path: Path) -> None:
    db_path = tmp_path / "recovery.db"

    store1 = TaskStore(db_path=db_path)
    engine1 = WorkflowEngine(store1)

    engine1.register_task(make_task("ready-task", run_id="run-ready"))
    engine1.register_task(make_task("running-task", run_id="run-running"))

    engine1.validate_task("ready-task")
    engine1.validate_task("running-task")
    engine1.refresh_ready_tasks()

    running_claim = engine1.claim_next_task(
        run_id="run-running",
        worker_id="worker-1",
        claim_request_id="running-recovery-setup",
    )
    assert running_claim.claim is not None

    store2 = TaskStore(db_path=db_path)
    engine2 = WorkflowEngine(store2)

    ready = engine2.recover()

    assert [task.task_id for task in ready] == ["ready-task"]
    assert store2.get_task("ready-task").status == TaskStatus.READY
    assert store2.get_task("running-task").status == TaskStatus.RUNNING


def test_recovery_marks_task_ready_when_dependencies_are_accepted(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recovery-dependency.db"

    store1 = TaskStore(db_path=db_path)
    engine1 = WorkflowEngine(store1)

    engine1.register_task(make_task("parent"))
    engine1.register_task(make_task("child", depends_on=("parent",)))
    engine1.validate_task("parent")
    engine1.validate_task("child")

    # Parent sudah selesai sebelum proses restart.
    store1.update_status("parent", TaskStatus.READY)
    parent_claim = engine1.claim_next_task(
        run_id="run-1",
        worker_id="worker-1",
        claim_request_id="accepted-dependency-parent",
    )
    assert parent_claim.claim is not None
    transition_claim(
        engine1, parent_claim.claim, TaskStatus.RUNNING, TaskStatus.TESTING
    )
    transition_claim(
        engine1, parent_claim.claim, TaskStatus.TESTING, TaskStatus.REVIEW
    )
    transition_claim(
        engine1, parent_claim.claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED
    )

    store2 = TaskStore(db_path=db_path)
    engine2 = WorkflowEngine(store2)

    ready = engine2.recover()

    assert [task.task_id for task in ready] == ["child"]
    assert store2.get_task("child").status == TaskStatus.READY
    assert store2.get_task("parent").status == TaskStatus.ACCEPTED


def test_engine_recover_requeues_expired_attempt(tmp_path: Path) -> None:
    db_path = tmp_path / "workflow-expired-recovery.db"
    store = TaskStore(db_path)
    engine = WorkflowEngine(store)
    task = make_task("expired-workflow-task")
    engine.register_task(task)
    engine.validate_task(task.task_id)
    engine.refresh_ready_tasks()
    result = engine.claim_next_task(
        worker_id="worker-recover", claim_request_id="workflow-expired-claim"
    )
    assert result.claim is not None
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE task_attempts SET lease_expires_at = ? WHERE attempt_id = ?",
            ("2000-01-01T00:00:00+00:00", result.claim.attempt_id),
        )

    recovered = WorkflowEngine(TaskStore(db_path)).recover()

    assert [item.task_id for item in recovered] == [task.task_id]
    assert store.get_task(task.task_id).status == TaskStatus.READY


@pytest.mark.parametrize("max_attempts", [4, 5])
def test_workflow_registration_rejects_budget_above_global_cap(
    engine: WorkflowEngine, max_attempts: int
) -> None:
    task = make_task(
        f"over-cap-registration-{max_attempts}"
    ).model_copy(update={"budget": TaskBudget(max_attempts=max_attempts)})

    with pytest.raises(ValueError, match="global maximum of 3"):
        engine.register_task(task)

    assert engine.store.list_tasks() == []


def test_workflow_cannot_validate_historical_created_task_above_cap(
    engine: WorkflowEngine,
) -> None:
    task = make_task("historical-created-over-cap")
    engine.register_task(task)
    with sqlite3.connect(engine.store.db_path) as connection:
        connection.execute(
            "UPDATE tasks SET budget_json = ? WHERE task_id = ?",
            (TaskBudget(max_attempts=4).model_dump_json(), task.task_id),
        )

    with pytest.raises(WorkflowValidationError, match="global max_attempts"):
        engine.validate_task(task.task_id)

    assert engine.store.get_task(task.task_id).status == TaskStatus.CREATED


def test_workflow_recovery_does_not_requeue_expired_historical_task_above_cap(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "workflow-historical-over-cap.db"
    store = TaskStore(db_path)
    engine = WorkflowEngine(store)
    task = make_task("workflow-historical-expired")
    engine.register_task(task)
    engine.validate_task(task.task_id)
    engine.refresh_ready_tasks()
    claim_result = engine.claim_next_task(
        worker_id="worker-historical-expired",
        claim_request_id="claim-historical-expired",
    )
    assert claim_result.claim is not None

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE tasks SET budget_json = ? WHERE task_id = ?",
            (TaskBudget(max_attempts=5).model_dump_json(), task.task_id),
        )
        connection.execute(
            "UPDATE task_attempts SET lease_expires_at = ? WHERE attempt_id = ?",
            ("2000-01-01T00:00:00+00:00", claim_result.claim.attempt_id),
        )

    assert WorkflowEngine(TaskStore(db_path)).recover() == []
    assert store.get_task(task.task_id).status == TaskStatus.FAILED
    assert store.get_task(task.task_id).attempt == 1
