from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from swarm_forge.contracts.task import AgentRole, TaskSpec, TaskStatus
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

    claimed = engine.claim_next_task()
    assert claimed is not None
    assert claimed.task_id == "task-1"
    assert claimed.status == TaskStatus.RUNNING
    assert claimed.attempt == 1
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
    store.update_status("parent", TaskStatus.RUNNING)
    store.update_status("parent", TaskStatus.TESTING)
    store.update_status("parent", TaskStatus.REVIEW)
    store.update_status("parent", TaskStatus.ACCEPTED)

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

    engine1.register_task(make_task("ready-task"))
    engine1.register_task(make_task("running-task"))

    engine1.validate_task("ready-task")
    engine1.validate_task("running-task")
    engine1.refresh_ready_tasks()

    store1.update_status("running-task", TaskStatus.RUNNING)

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
    store1.update_status("parent", TaskStatus.RUNNING)
    store1.update_status("parent", TaskStatus.TESTING)
    store1.update_status("parent", TaskStatus.REVIEW)
    store1.update_status("parent", TaskStatus.ACCEPTED)

    store2 = TaskStore(db_path=db_path)
    engine2 = WorkflowEngine(store2)

    ready = engine2.recover()

    assert [task.task_id for task in ready] == ["child"]
    assert store2.get_task("child").status == TaskStatus.READY
    assert store2.get_task("parent").status == TaskStatus.ACCEPTED
