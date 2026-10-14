from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from swarm_forge.contracts.task import AgentRole, TaskBudget, TaskSpec, TaskStatus
from swarm_forge.storage.task_store import (
    DuplicateTaskError,
    TaskNotFoundError,
    TaskStore,
)
from swarm_forge.workflow.state_machine import InvalidStateTransitionError


@pytest.fixture
def task_store(tmp_path: Path) -> TaskStore:
    db_file = tmp_path / "test_swarm.db"
    return TaskStore(db_path=db_file)


def test_create_and_get_task_full_fields(task_store: TaskStore) -> None:
    spec = TaskSpec(
        task_id="custom-task-001",
        run_id="run-123",
        parent_task_id="parent-000",
        agent_role=AgentRole.ARCHITECT,
        goal="Design resilient architecture",
        depends_on=("task-dep-1", "task-dep-2"),
        input_refs=("ref-input-1",),
        output_contract="Architecture Blueprint",
        allowed_tools=("read_file", "search"),
        budget=TaskBudget(
            max_attempts=4,
            timeout_seconds=600,
            max_changed_files=15,
            max_input_tokens=10000,
            max_output_tokens=5000,
        ),
        attempt=1,
    )

    task_store.save_task(spec)
    retrieved = task_store.get_task(spec.task_id)

    assert retrieved.task_id == spec.task_id
    assert retrieved.run_id == spec.run_id
    assert retrieved.parent_task_id == spec.parent_task_id
    assert retrieved.agent_role == spec.agent_role
    assert retrieved.goal == spec.goal
    assert retrieved.depends_on == ("task-dep-1", "task-dep-2")
    assert retrieved.input_refs == ("ref-input-1",)
    assert retrieved.output_contract == spec.output_contract
    assert retrieved.allowed_tools == ("read_file", "search")
    assert retrieved.budget.max_attempts == 4
    assert retrieved.budget.timeout_seconds == 600
    assert retrieved.budget.max_changed_files == 15
    assert retrieved.budget.max_input_tokens == 10000
    assert retrieved.budget.max_output_tokens == 5000
    assert retrieved.status == TaskStatus.CREATED
    assert retrieved.attempt == 1
    assert retrieved.created_at == spec.created_at


def test_duplicate_task_raises_error(task_store: TaskStore) -> None:
    spec = TaskSpec(task_id="task-dup", run_id="run-1", agent_role=AgentRole.PLANNER, goal="Plan", output_contract="plan")
    task_store.save_task(spec)

    with pytest.raises(DuplicateTaskError):
        task_store.save_task(spec)


def test_task_not_found_raises_error(task_store: TaskStore) -> None:
    with pytest.raises(TaskNotFoundError):
        task_store.get_task("non-existent-id")


def test_list_tasks_with_and_without_filter(task_store: TaskStore) -> None:
    task1 = TaskSpec(run_id="run-A", agent_role=AgentRole.ANALYST, goal="Goal A", output_contract="contract A")
    task2 = TaskSpec(run_id="run-B", agent_role=AgentRole.QA, goal="Goal B", output_contract="contract B")

    task_store.save_task(task1)
    task_store.save_task(task2)

    all_tasks = task_store.list_tasks()
    assert len(all_tasks) == 2

    run_a_tasks = task_store.list_tasks(run_id="run-A")
    assert len(run_a_tasks) == 1
    assert run_a_tasks[0].task_id == task1.task_id


def test_update_status_valid_transition(task_store: TaskStore) -> None:
    spec = TaskSpec(run_id="run-1", agent_role=AgentRole.IMPLEMENTER, goal="Code", output_contract="code")
    task_store.save_task(spec)

    updated = task_store.update_status(spec.task_id, TaskStatus.VALIDATED)
    assert updated.status == TaskStatus.VALIDATED

    # Verify persistence
    retrieved = task_store.get_task(spec.task_id)
    assert retrieved.status == TaskStatus.VALIDATED


def test_update_status_invalid_transition_raises_and_rolls_back(task_store: TaskStore) -> None:
    spec = TaskSpec(run_id="run-1", agent_role=AgentRole.IMPLEMENTER, goal="Code", output_contract="code")
    task_store.save_task(spec)

    with pytest.raises(InvalidStateTransitionError):
        task_store.update_status(spec.task_id, TaskStatus.RUNNING)  # CREATED -> RUNNING is illegal

    # Verify status remains unchanged (atomicity / rollback check)
    retrieved = task_store.get_task(spec.task_id)
    assert retrieved.status == TaskStatus.CREATED


def test_persistence_across_restarts(tmp_path: Path) -> None:
    db_file = tmp_path / "persist.db"
    
    # Store and close session
    store1 = TaskStore(db_path=db_file)
    spec = TaskSpec(run_id="run-persist", agent_role=AgentRole.ARCHITECT, goal="Design", output_contract="design")
    store1.save_task(spec)
    store1.update_status(spec.task_id, TaskStatus.VALIDATED)

    # Re-open store with same file path
    store2 = TaskStore(db_path=db_file)
    retrieved = store2.get_task(spec.task_id)

    assert retrieved.task_id == spec.task_id
    assert retrieved.status == TaskStatus.VALIDATED
    assert retrieved.goal == "Design"


def test_in_memory_database_support_and_close() -> None:
    store = TaskStore(db_path=":memory:")
    spec = TaskSpec(run_id="run-mem", agent_role=AgentRole.QA, goal="Test memory", output_contract="test")
    store.save_task(spec)
    retrieved = store.get_task(spec.task_id)
    assert retrieved.task_id == spec.task_id
    
    # Test explicit close lifecycle method
    store.close()


def test_in_memory_duplicate_task_handling() -> None:
    store = TaskStore(db_path=":memory:")
    spec = TaskSpec(task_id="task-mem-dup", run_id="run-mem", agent_role=AgentRole.PLANNER, goal="Memory dup test", output_contract="test")
    store.save_task(spec)

    with pytest.raises(DuplicateTaskError):
        store.save_task(spec)

    retrieved = store.get_task("task-mem-dup")
    assert retrieved.goal == "Memory dup test"
    store.close()


def test_sql_execution_error_triggers_rollback(task_store: TaskStore) -> None:
    spec = TaskSpec(
        task_id="task-sql-err",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="SQL error test",
        output_contract="code",
    )
    task_store.save_task(spec)

    conn = sqlite3.connect(task_store.db_path)
    try:
        conn.execute(
            """
            CREATE TRIGGER fail_task_status_update
            BEFORE UPDATE OF status ON tasks
            BEGIN
                SELECT RAISE(ABORT, 'forced status update failure');
            END
            """
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="forced status update failure"):
        task_store.update_status(spec.task_id, TaskStatus.VALIDATED)

    retrieved = task_store.get_task(spec.task_id)
    assert retrieved.status == TaskStatus.CREATED
