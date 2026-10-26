from __future__ import annotations

import inspect
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event

import pytest
from pydantic import ValidationError

from swarm_forge.contracts.claim import AttemptStatus, TaskClaim
from swarm_forge.contracts.task import AgentRole, TaskBudget, TaskSpec, TaskStatus
from swarm_forge.storage import task_store as task_store_module
from swarm_forge.storage.task_store import (
    AttemptFencingError,
    AttemptLimitExceededError,
    ClaimRequestConflictError,
    DuplicateTaskError,
    TaskAttemptRequiredError,
    TaskClaimRequiredError,
    TaskNotFoundError,
    TaskStore,
)
from swarm_forge.workflow.state_machine import InvalidStateTransitionError


@pytest.fixture
def task_store(tmp_path: Path) -> TaskStore:
    db_file = tmp_path / "test_swarm.db"
    return TaskStore(db_path=db_file)


def make_legacy_database(db_file: Path, spec: TaskSpec) -> None:
    """Create the original tasks-only schema directly, before migrations."""
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                parent_task_id TEXT,
                agent_role TEXT NOT NULL,
                goal TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                budget_json TEXT NOT NULL,
                depends_on_json TEXT NOT NULL,
                input_refs_json TEXT NOT NULL,
                output_contract TEXT NOT NULL,
                allowed_tools_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, run_id, parent_task_id, agent_role, goal, status,
                attempt, created_at, budget_json, depends_on_json, input_refs_json,
                output_contract, allowed_tools_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.task_id,
                spec.run_id,
                spec.parent_task_id,
                spec.agent_role.value,
                spec.goal,
                spec.status.value,
                spec.attempt,
                spec.created_at.isoformat(),
                spec.budget.model_dump_json(),
                json.dumps(list(spec.depends_on)),
                json.dumps(list(spec.input_refs)),
                spec.output_contract,
                json.dumps(list(spec.allowed_tools)),
            ),
        )


def insert_attempt(
    connection: sqlite3.Connection,
    *,
    attempt_id: str,
    task_id: str,
    attempt_no: int,
    status: str,
) -> None:
    connection.execute(
        """
        INSERT INTO task_attempts (
            attempt_id, task_id, attempt_no, status, started_at, lease_expires_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            attempt_id,
            task_id,
            attempt_no,
            status,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:05:00+00:00" if status == "running" else None,
        ),
    )


def transition_claim(
    store: TaskStore,
    claim: TaskClaim,
    expected_task_status: TaskStatus,
    new_task_status: TaskStatus,
) -> TaskSpec:
    return store.transition_attempt_status(
        task_id=claim.task.task_id,
        attempt_id=claim.attempt_id,
        attempt_no=claim.attempt_no,
        expected_attempt_status=AttemptStatus.RUNNING,
        expected_task_status=expected_task_status,
        new_task_status=new_task_status,
    )


def claim_task_at_review(store: TaskStore, task_id: str) -> TaskClaim:
    task = TaskSpec(
        task_id=task_id,
        run_id="run-fencing",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Prepare task for fenced completion",
        output_contract="patch",
        status=TaskStatus.READY,
    )
    store.save_task(task)
    result = store.claim_next_task(
        run_id=task.run_id,
        worker_id=f"worker-{task_id}",
        claim_request_id=f"claim-{task_id}",
    )
    assert result.claim is not None
    transition_claim(store, result.claim, TaskStatus.RUNNING, TaskStatus.TESTING)
    transition_claim(store, result.claim, TaskStatus.TESTING, TaskStatus.REVIEW)
    return result.claim


def test_new_database_initializes_attempt_history_schema(tmp_path: Path) -> None:
    db_file = tmp_path / "new-schema.db"

    TaskStore(db_path=db_file)

    with sqlite3.connect(db_file) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(task_attempts)")
        }
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(task_attempts)"
        ).fetchall()

    assert version == TaskStore.SCHEMA_VERSION
    assert {
        "attempt_id",
        "task_id",
        "attempt_no",
        "worker_id",
        "status",
        "started_at",
        "heartbeat_at",
        "lease_expires_at",
        "finished_at",
        "reason_code",
    } <= columns
    assert any(row[2] == "tasks" and row[3] == "task_id" for row in foreign_keys)


def test_migration_preserves_legacy_non_running_task(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy-non-running.db"
    spec = TaskSpec(
        task_id="legacy-validated",
        run_id="run-legacy",
        agent_role=AgentRole.PLANNER,
        goal="Preserve old task",
        output_contract="plan",
        status=TaskStatus.VALIDATED,
    )
    make_legacy_database(db_file, spec)

    migrated = TaskStore(db_path=db_file)
    persisted = migrated.get_task(spec.task_id)

    assert persisted == spec
    with sqlite3.connect(db_file) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?",
            (spec.task_id,),
        ).fetchone()[0] == 0


def test_migration_keeps_legacy_running_task_without_lease_or_attempt(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "legacy-running.db"
    spec = TaskSpec(
        task_id="legacy-running",
        run_id="run-legacy",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Do not fabricate a lease",
        output_contract="patch",
        status=TaskStatus.RUNNING,
        attempt=1,
    )
    make_legacy_database(db_file, spec)

    migrated = TaskStore(db_path=db_file)

    assert migrated.get_task(spec.task_id) == spec
    with sqlite3.connect(db_file) as connection:
        rows = connection.execute(
            "SELECT status, lease_expires_at FROM task_attempts WHERE task_id = ?",
            (spec.task_id,),
        ).fetchall()

    assert rows == []


def test_attempt_history_migration_is_idempotent(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy-repeat.db"
    spec = TaskSpec(
        task_id="legacy-repeat",
        run_id="run-legacy",
        agent_role=AgentRole.PLANNER,
        goal="Migrate repeatedly",
        output_contract="plan",
        status=TaskStatus.RUNNING,
        attempt=1,
    )
    make_legacy_database(db_file, spec)

    TaskStore(db_path=db_file)
    TaskStore(db_path=db_file)

    with sqlite3.connect(db_file) as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == TaskStore.SCHEMA_VERSION
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?",
            (spec.task_id,),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT status FROM tasks WHERE task_id = ?", (spec.task_id,)
        ).fetchone()[0] == TaskStatus.RUNNING.value


def test_schema_version_one_migrates_to_claim_request_schema(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "schema-version-one.db"
    task = TaskSpec(
        task_id="version-one-task",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Preserve attempt history through migration",
        output_contract="patch",
    )
    store = TaskStore(db_path=db_path)
    store.save_task(task)
    with sqlite3.connect(db_path) as connection:
        insert_attempt(
            connection,
            attempt_id="version-one-attempt",
            task_id=task.task_id,
            attempt_no=1,
            status="failed",
        )
        connection.execute("DROP TABLE claim_requests")
        connection.execute("PRAGMA user_version = 1")

    migrated = TaskStore(db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == TaskStore.SCHEMA_VERSION
        )
        assert connection.execute(
            "SELECT attempt_id, attempt_no, status FROM task_attempts "
            "WHERE task_id = ?",
            (task.task_id,),
        ).fetchone() == ("version-one-attempt", 1, "failed")
        assert connection.execute(
            "SELECT COUNT(*) FROM claim_requests"
        ).fetchone()[0] == 0
    assert migrated.get_task(task.task_id) == task


def test_attempt_number_must_be_unique_per_task(task_store: TaskStore) -> None:
    task = TaskSpec(
        task_id="attempt-unique-task",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Attempt uniqueness",
        output_contract="patch",
    )
    task_store.save_task(task)
    connection = sqlite3.connect(task_store.db_path)
    try:
        insert_attempt(
            connection,
            attempt_id="attempt-unique-1",
            task_id=task.task_id,
            attempt_no=1,
            status="failed",
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            insert_attempt(
                connection,
                attempt_id="attempt-unique-2",
                task_id=task.task_id,
                attempt_no=1,
                status="failed",
            )
    finally:
        connection.close()


def test_only_one_running_attempt_per_task(task_store: TaskStore) -> None:
    task = TaskSpec(
        task_id="single-running-task",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Running uniqueness",
        output_contract="patch",
    )
    task_store.save_task(task)
    connection = sqlite3.connect(task_store.db_path)
    try:
        insert_attempt(
            connection,
            attempt_id="running-attempt-1",
            task_id=task.task_id,
            attempt_no=1,
            status="running",
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            insert_attempt(
                connection,
                attempt_id="running-attempt-2",
                task_id=task.task_id,
                attempt_no=2,
                status="running",
            )
    finally:
        connection.close()


def test_attempt_history_enforces_foreign_key_and_safe_reason_code(
    task_store: TaskStore,
) -> None:
    connection = task_store._get_connection()
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            insert_attempt(
                connection,
                attempt_id="attempt-missing-task",
                task_id="missing-task",
                attempt_no=1,
                status="failed",
            )

        task = TaskSpec(
            task_id="reason-code-task",
            run_id="run-1",
            agent_role=AgentRole.IMPLEMENTER,
            goal="Reason code validation",
            output_contract="patch",
        )
        connection.rollback()
        task_store.save_task(task)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO task_attempts (
                    attempt_id, task_id, attempt_no, status, started_at, reason_code
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "attempt-unsafe-reason",
                    task.task_id,
                    1,
                    "failed",
                    "2026-01-01T00:00:00+00:00",
                    "raw credential value",
                ),
            )
    finally:
        connection.close()


def test_in_memory_task_store_enforces_foreign_keys() -> None:
    store = TaskStore(db_path=":memory:")
    connection = store._get_connection()
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            insert_attempt(
                connection,
                attempt_id="memory-attempt-missing-task",
                task_id="missing-task",
                attempt_no=1,
                status="failed",
            )
    finally:
        store.close()


def test_migration_failure_rolls_back_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_file = tmp_path / "migration-retry.db"
    spec = TaskSpec(
        task_id="migration-retry-task",
        run_id="run-legacy",
        agent_role=AgentRole.PLANNER,
        goal="Retry a failed migration",
        output_contract="plan",
        status=TaskStatus.VALIDATED,
    )
    make_legacy_database(db_file, spec)

    original_get_connection = TaskStore._get_connection

    def deny_attempt_index(store: TaskStore) -> sqlite3.Connection:
        connection = original_get_connection(store)

        def authorizer(
            action: int,
            argument_one: str | None,
            _argument_two: str | None,
            _database_name: str | None,
            _trigger_name: str | None,
        ) -> int:
            if (
                action == sqlite3.SQLITE_CREATE_INDEX
                and argument_one == "idx_task_attempts_one_running_per_task"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        return connection

    monkeypatch.setattr(TaskStore, "_get_connection", deny_attempt_index)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        TaskStore(db_path=db_file)

    with sqlite3.connect(db_file) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute(
            "SELECT status FROM tasks WHERE task_id = ?", (spec.task_id,)
        ).fetchone()[0] == TaskStatus.VALIDATED.value
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'task_attempts'
            """
        ).fetchone()[0] == 0

    monkeypatch.setattr(TaskStore, "_get_connection", original_get_connection)
    migrated = TaskStore(db_path=db_file)
    assert migrated.get_task(spec.task_id) == spec
    with sqlite3.connect(db_file) as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == TaskStore.SCHEMA_VERSION
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts"
        ).fetchone()[0] == 0


def test_newer_schema_version_is_rejected_without_modification(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "future-schema.db"
    spec = TaskSpec(
        task_id="future-schema-task",
        run_id="run-legacy",
        agent_role=AgentRole.PLANNER,
        goal="Preserve future schema",
        output_contract="plan",
    )
    make_legacy_database(db_file, spec)
    future_version = TaskStore.SCHEMA_VERSION + 1
    with sqlite3.connect(db_file) as connection:
        connection.execute(f"PRAGMA user_version = {future_version}")

    with sqlite3.connect(db_file) as connection:
        schema_before = connection.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name
            """
        ).fetchall()
        task_before = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (spec.task_id,)
        ).fetchone()

    with pytest.raises(RuntimeError, match="newer than supported"):
        TaskStore(db_path=db_file)

    with sqlite3.connect(db_file) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == future_version
        assert connection.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name
            """
        ).fetchall() == schema_before
        assert connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (spec.task_id,)
        ).fetchone() == task_before


def test_running_attempt_requires_a_lease_expiry(task_store: TaskStore) -> None:
    task = TaskSpec(
        task_id="running-requires-lease",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Require lease metadata",
        output_contract="patch",
    )
    task_store.save_task(task)
    connection = task_store._get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO task_attempts (
                    attempt_id, task_id, attempt_no, status, started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "running-without-lease",
                    task.task_id,
                    1,
                    "running",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
    finally:
        connection.close()


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


def test_loading_persisted_naive_created_at_rejects_legacy_value(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="naive-created-at-task",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Reject ambiguous persisted timestamps",
        output_contract="patch",
    )
    task_store.save_task(spec)
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            "UPDATE tasks SET created_at = ? WHERE task_id = ?",
            ("2026-01-01T00:00:00", spec.task_id),
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        task_store.get_task(spec.task_id)


def test_update_status_rolls_back_when_persisted_task_is_invalid(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="naive-created-at-update",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Do not commit a mutation with invalid persisted data",
        output_contract="patch",
    )
    task_store.save_task(spec)
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            "UPDATE tasks SET created_at = ? WHERE task_id = ?",
            ("2026-01-01T00:00:00", spec.task_id),
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        task_store.update_status(spec.task_id, TaskStatus.VALIDATED)

    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status, attempt, created_at FROM tasks WHERE task_id = ?",
            (spec.task_id,),
        ).fetchone() == (
            TaskStatus.CREATED.value,
            0,
            "2026-01-01T00:00:00",
        )


def test_duplicate_task_raises_error(task_store: TaskStore) -> None:
    spec = TaskSpec(
        task_id="task-dup",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="Plan",
        output_contract="plan",
    )
    task_store.save_task(spec)

    with pytest.raises(DuplicateTaskError):
        task_store.save_task(spec)


def test_task_not_found_raises_error(task_store: TaskStore) -> None:
    with pytest.raises(TaskNotFoundError):
        task_store.get_task("non-existent-id")


def test_list_tasks_with_and_without_filter(task_store: TaskStore) -> None:
    task1 = TaskSpec(
        run_id="run-A",
        agent_role=AgentRole.ANALYST,
        goal="Goal A",
        output_contract="contract A",
    )
    task2 = TaskSpec(
        run_id="run-B",
        agent_role=AgentRole.QA,
        goal="Goal B",
        output_contract="contract B",
    )

    task_store.save_task(task1)
    task_store.save_task(task2)

    all_tasks = task_store.list_tasks()
    assert len(all_tasks) == 2

    run_a_tasks = task_store.list_tasks(run_id="run-A")
    assert len(run_a_tasks) == 1
    assert run_a_tasks[0].task_id == task1.task_id


def test_update_status_valid_transition(task_store: TaskStore) -> None:
    spec = TaskSpec(
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Code",
        output_contract="code",
    )
    task_store.save_task(spec)

    updated = task_store.update_status(spec.task_id, TaskStatus.VALIDATED)
    assert updated.status == TaskStatus.VALIDATED

    # Verify persistence
    retrieved = task_store.get_task(spec.task_id)
    assert retrieved.status == TaskStatus.VALIDATED


def test_generic_failed_to_ready_rejects_exhausted_retry_budget(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="generic-retry-exhausted",
        run_id="run-generic-retry",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Do not bypass the retry budget",
        output_contract="patch",
        status=TaskStatus.FAILED,
        attempt=2,
        budget=TaskBudget(max_attempts=2),
    )
    task_store.save_task(spec)

    with pytest.raises(AttemptLimitExceededError):
        task_store.update_status(spec.task_id, TaskStatus.READY)

    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.FAILED
    assert persisted.attempt == 2


def test_generic_failed_to_ready_allows_retry_when_budget_remains(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="generic-retry-available",
        run_id="run-generic-retry",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Allow an available retry",
        output_contract="patch",
        status=TaskStatus.FAILED,
        attempt=1,
        budget=TaskBudget(max_attempts=2),
    )
    task_store.save_task(spec)

    updated = task_store.update_status(spec.task_id, TaskStatus.READY)

    assert updated.status == TaskStatus.READY
    assert updated.attempt == 1
    assert task_store.get_task(spec.task_id).attempt == 1


def test_update_status_invalid_transition_raises_and_rolls_back(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Code",
        output_contract="code",
    )
    task_store.save_task(spec)

    with pytest.raises(TaskClaimRequiredError):
        task_store.update_status(
            spec.task_id, TaskStatus.RUNNING
        )

    # Verify status remains unchanged (atomicity / rollback check)
    retrieved = task_store.get_task(spec.task_id)
    assert retrieved.status == TaskStatus.CREATED


def test_save_task_rejects_running_without_atomic_claim(task_store: TaskStore) -> None:
    spec = TaskSpec(
        task_id="cannot-save-running",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="RUNNING requires attempt history",
        output_contract="patch",
        status=TaskStatus.RUNNING,
        attempt=1,
    )

    with pytest.raises(TaskClaimRequiredError):
        task_store.save_task(spec)

    with pytest.raises(TaskNotFoundError):
        task_store.get_task(spec.task_id)


def test_persistence_across_restarts(tmp_path: Path) -> None:
    db_file = tmp_path / "persist.db"

    # Store and close session
    store1 = TaskStore(db_path=db_file)
    spec = TaskSpec(
        run_id="run-persist",
        agent_role=AgentRole.ARCHITECT,
        goal="Design",
        output_contract="design",
    )
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
    spec = TaskSpec(
        run_id="run-mem",
        agent_role=AgentRole.QA,
        goal="Test memory",
        output_contract="test",
    )
    store.save_task(spec)
    retrieved = store.get_task(spec.task_id)
    assert retrieved.task_id == spec.task_id

    # Test explicit close lifecycle method
    store.close()


def test_in_memory_duplicate_task_handling() -> None:
    store = TaskStore(db_path=":memory:")
    spec = TaskSpec(
        task_id="task-mem-dup",
        run_id="run-mem",
        agent_role=AgentRole.PLANNER,
        goal="Memory dup test",
        output_contract="test",
    )
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


def test_retry_task_returns_failed_task_to_ready_without_increment(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="task-retry-success",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Retry implementation",
        output_contract="code",
        budget=TaskBudget(max_attempts=3),
        status=TaskStatus.FAILED,
        attempt=1,
    )
    task_store.save_task(spec)

    retried = task_store.retry_task(spec.task_id)

    assert retried.status == TaskStatus.READY
    assert retried.attempt == 1

    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.READY
    assert persisted.attempt == 1


def test_retry_task_rejects_exhausted_attempt_budget(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="task-retry-exhausted",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Exhausted retry",
        output_contract="code",
        budget=TaskBudget(max_attempts=2),
        status=TaskStatus.FAILED,
        attempt=2,
    )
    task_store.save_task(spec)

    with pytest.raises(AttemptLimitExceededError):
        task_store.retry_task(spec.task_id)

    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.FAILED
    assert persisted.attempt == 2


def test_retry_task_rejects_non_failed_task(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="task-retry-invalid-state",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Invalid retry state",
        output_contract="code",
        status=TaskStatus.CREATED,
    )
    task_store.save_task(spec)

    with pytest.raises(InvalidStateTransitionError):
        task_store.retry_task(spec.task_id)

    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.CREATED
    assert persisted.attempt == 0


def test_retry_task_rejects_unknown_task(task_store: TaskStore) -> None:
    with pytest.raises(TaskNotFoundError):
        task_store.retry_task("missing-retry-task")


def test_full_retry_lifecycle_increments_attempt_only_on_start(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="task-retry-lifecycle",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Verify retry lifecycle",
        output_contract="code",
        budget=TaskBudget(max_attempts=2),
        status=TaskStatus.READY,
    )
    task_store.save_task(spec)

    first_result = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="retry-claim-1"
    )
    assert first_result.claim is not None
    assert first_result.claim.task.status == TaskStatus.RUNNING
    assert first_result.claim.attempt_no == 1

    failed = transition_claim(
        task_store,
        first_result.claim,
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
    )
    assert failed.status == TaskStatus.FAILED
    assert failed.attempt == 1

    retried = task_store.retry_task(spec.task_id)
    assert retried.status == TaskStatus.READY
    assert retried.attempt == 1

    second_result = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="retry-claim-2"
    )
    assert second_result.claim is not None
    assert second_result.claim.task.status == TaskStatus.RUNNING
    assert second_result.claim.attempt_no == 2

    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.attempt == 2
    with sqlite3.connect(task_store.db_path) as connection:
        attempts = connection.execute(
            "SELECT attempt_no, status FROM task_attempts WHERE task_id = ? "
            "ORDER BY attempt_no",
            (spec.task_id,),
        ).fetchall()
    assert attempts == [(1, "failed"), (2, "running")]


def test_claim_atomically_updates_status_counter_and_attempt_history(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="task-start-attempt",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Implement feature",
        output_contract="code",
    )
    task_store.save_task(spec)
    task_store.update_status(spec.task_id, TaskStatus.VALIDATED)
    task_store.update_status(spec.task_id, TaskStatus.READY)

    result = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="atomic-claim"
    )

    assert result.claim is not None
    assert result.claim.task.status == TaskStatus.RUNNING
    assert result.claim.attempt_no == 1
    assert result.claim.worker_id == "worker-1"
    assert result.claim.attempt_id
    assert result.claim.lease_expires_at > datetime.now(UTC)
    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.RUNNING
    assert persisted.attempt == 1
    with sqlite3.connect(task_store.db_path) as connection:
        attempt_rows = connection.execute(
            "SELECT attempt_id, attempt_no, worker_id, status, lease_expires_at "
            "FROM task_attempts WHERE task_id = ?",
            (spec.task_id,),
        ).fetchall()
        request_count = connection.execute(
            "SELECT COUNT(*) FROM claim_requests WHERE claim_request_id = ?",
            ("atomic-claim",),
        ).fetchone()[0]
    assert len(attempt_rows) == 1
    assert attempt_rows[0][0] == result.claim.attempt_id
    assert attempt_rows[0][1:4] == (1, "worker-1", "running")
    assert attempt_rows[0][4] is not None
    assert request_count == 1


def test_claim_skips_exhausted_budget_without_creating_attempt(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="task-attempt-limit",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Implement feature",
        output_contract="code",
        budget=TaskBudget(max_attempts=1),
        attempt=1,
        status=TaskStatus.READY,
    )
    task_store.save_task(spec)

    result = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="exhausted-claim"
    )

    assert result.claim is None
    assert result.skipped_candidates[0].task_id == spec.task_id
    assert result.skipped_candidates[0].reason.value == "attempt_budget_exhausted"
    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.READY
    assert persisted.attempt == 1
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?", (spec.task_id,)
        ).fetchone()[0] == 0


def test_claim_returns_empty_for_non_ready_task_without_increment(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="task-not-ready",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Implement feature",
        output_contract="code",
    )
    task_store.save_task(spec)

    result = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="not-ready-claim"
    )

    assert result.claim is None
    persisted = task_store.get_task(spec.task_id)
    assert persisted.status == TaskStatus.CREATED
    assert persisted.attempt == 0


def test_invalid_or_exhausted_candidate_does_not_block_eligible_task(
    task_store: TaskStore,
) -> None:
    first_created = datetime(2026, 1, 1, tzinfo=UTC)
    exhausted = TaskSpec(
        task_id="first-exhausted",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="Exhausted earlier candidate",
        output_contract="plan",
        status=TaskStatus.READY,
        attempt=1,
        budget=TaskBudget(max_attempts=1),
        created_at=first_created,
    )
    eligible = TaskSpec(
        task_id="second-eligible",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="Claim next eligible candidate",
        output_contract="plan",
        status=TaskStatus.READY,
        created_at=first_created + timedelta(seconds=1),
    )
    task_store.save_task(exhausted)
    task_store.save_task(eligible)

    result = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="skip-exhausted"
    )

    assert result.claim is not None
    assert result.claim.task.task_id == eligible.task_id
    assert [(item.task_id, item.reason.value) for item in result.skipped_candidates] == [
        (exhausted.task_id, "attempt_budget_exhausted")
    ]


def test_claim_skips_invalid_persisted_timestamp_and_reports_candidate(
    task_store: TaskStore,
) -> None:
    invalid = TaskSpec(
        task_id="invalid-persisted-candidate",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="Skip invalid persisted data",
        output_contract="plan",
        status=TaskStatus.READY,
    )
    eligible = TaskSpec(
        task_id="valid-after-invalid-candidate",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="Still claim a valid task",
        output_contract="plan",
        status=TaskStatus.READY,
    )
    task_store.save_task(invalid)
    task_store.save_task(eligible)
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            "UPDATE tasks SET created_at = ? WHERE task_id = ?",
            ("2026-01-01T00:00:00", invalid.task_id),
        )

    result = task_store.claim_next_task(
        run_id="run-1",
        worker_id="worker-1",
        claim_request_id="skip-invalid-persisted-task",
    )

    assert result.claim is not None
    assert result.claim.task.task_id == eligible.task_id
    assert [(item.task_id, item.reason.value) for item in result.skipped_candidates] == [
        (invalid.task_id, "task_invalid")
    ]


def test_claim_orders_timezone_aware_candidates_deterministically(
    task_store: TaskStore,
) -> None:
    offset = timedelta(hours=2)
    candidates = (
        ("z-later", datetime(2026, 1, 1, 13, tzinfo=UTC)),
        ("b-tie", datetime(2026, 1, 1, 12, tzinfo=UTC)),
        ("early", datetime(2026, 1, 1, 13, tzinfo=timezone(offset))),
        ("a-tie", datetime(2026, 1, 1, 14, tzinfo=timezone(offset))),
    )
    for task_id, created_at in candidates:
        task_store.save_task(
            TaskSpec(
                task_id=task_id,
                run_id="run-1",
                agent_role=AgentRole.IMPLEMENTER,
                goal=f"Claim {task_id}",
                output_contract="patch",
                status=TaskStatus.READY,
                created_at=created_at,
            )
        )

    claimed_ids: list[str] = []
    for index in range(len(candidates)):
        result = task_store.claim_next_task(
            run_id="run-1",
            worker_id="worker-1",
            claim_request_id=f"ordered-claim-{index}",
        )
        assert result.claim is not None
        claimed_ids.append(result.claim.task.task_id)
        transition_claim(
            task_store,
            result.claim,
            TaskStatus.RUNNING,
            TaskStatus.CANCELLED,
        )

    assert claimed_ids == ["early", "a-tie", "b-tie", "z-later"]


def test_unmet_dependency_is_skipped_and_reported(task_store: TaskStore) -> None:
    dependency = TaskSpec(
        task_id="unaccepted-dependency",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="Dependency is not accepted",
        output_contract="plan",
        status=TaskStatus.VALIDATED,
    )
    dependent = TaskSpec(
        task_id="dependent-ready-task",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="Must wait for dependency",
        output_contract="plan",
        depends_on=(dependency.task_id,),
        status=TaskStatus.READY,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    eligible = TaskSpec(
        task_id="independent-ready-task",
        run_id="run-1",
        agent_role=AgentRole.PLANNER,
        goal="No dependency",
        output_contract="plan",
        status=TaskStatus.READY,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    task_store.save_task(dependency)
    task_store.save_task(dependent)
    task_store.save_task(eligible)

    result = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="dependency-check"
    )

    assert result.claim is not None
    assert result.claim.task.task_id == eligible.task_id
    assert [(item.task_id, item.reason.value) for item in result.skipped_candidates] == [
        (dependent.task_id, "dependency_not_accepted")
    ]


def test_attempt_insert_failure_rolls_back_entire_claim(task_store: TaskStore) -> None:
    spec = TaskSpec(
        task_id="attempt-insert-fails",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Rollback attempt insertion",
        output_contract="patch",
        status=TaskStatus.READY,
    )
    task_store.save_task(spec)
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_attempt_insert
            BEFORE INSERT ON task_attempts
            BEGIN
                SELECT RAISE(ABORT, 'forced attempt insert failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced attempt insert failure"):
        task_store.claim_next_task(
            run_id="run-1", worker_id="worker-1", claim_request_id="attempt-fail"
        )

    persisted = task_store.get_task(spec.task_id)
    assert (persisted.status, persisted.attempt) == (TaskStatus.READY, 0)
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM task_attempts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM claim_requests").fetchone()[0] == 0


def test_request_insert_failure_rolls_back_entire_claim(task_store: TaskStore) -> None:
    spec = TaskSpec(
        task_id="request-insert-fails",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Rollback request insertion",
        output_contract="patch",
        status=TaskStatus.READY,
    )
    task_store.save_task(spec)
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_claim_request_insert
            BEFORE INSERT ON claim_requests
            BEGIN
                SELECT RAISE(ABORT, 'forced request insert failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced request insert failure"):
        task_store.claim_next_task(
            run_id="run-1", worker_id="worker-1", claim_request_id="request-fail"
        )

    persisted = task_store.get_task(spec.task_id)
    assert (persisted.status, persisted.attempt) == (TaskStatus.READY, 0)
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM task_attempts").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM claim_requests").fetchone()[0] == 0


def test_claim_request_replays_success_after_store_reopen(task_store: TaskStore) -> None:
    spec = TaskSpec(
        task_id="claim-replay-task",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Replay successful claim",
        output_contract="patch",
        status=TaskStatus.READY,
    )
    task_store.save_task(spec)
    initial = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="stable-claim-id"
    )
    assert initial.claim is not None
    transition_claim(
        task_store,
        initial.claim,
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
    )
    task_store.retry_task(spec.task_id)
    task_store.close()

    reopened = TaskStore(db_path=task_store.db_path)
    replay = reopened.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="stable-claim-id"
    )

    assert replay == initial
    assert replay.claim is not None
    assert replay.claim.task.task_id == spec.task_id
    current = reopened.get_task(spec.task_id)
    assert (current.status, current.attempt) == (TaskStatus.READY, 1)
    with sqlite3.connect(reopened.db_path) as connection:
        attempts = connection.execute(
            "SELECT status FROM task_attempts WHERE task_id = ?", (spec.task_id,)
        ).fetchall()
        assert attempts == [("failed",)]
        assert connection.execute(
            "SELECT COUNT(*) FROM claim_requests WHERE claim_request_id = ?",
            ("stable-claim-id",),
        ).fetchone()[0] == 1


def test_empty_claim_request_replays_empty_and_new_id_polls_again(
    task_store: TaskStore,
) -> None:
    empty = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="empty-claim"
    )
    assert empty.claim is None

    task = TaskSpec(
        task_id="arrived-after-empty-claim",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Available to a later poll",
        output_contract="patch",
        status=TaskStatus.READY,
    )
    task_store.save_task(task)

    replay = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="empty-claim"
    )
    next_poll = task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="new-poll-id"
    )

    assert replay == empty
    assert replay.claim is None
    assert next_poll.claim is not None
    assert next_poll.claim.task.task_id == task.task_id


def test_claim_request_id_is_global_and_bound_to_worker_and_run(
    task_store: TaskStore,
) -> None:
    task_store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="global-request-id"
    )

    with pytest.raises(ClaimRequestConflictError):
        task_store.claim_next_task(
            run_id="run-2", worker_id="worker-2", claim_request_id="global-request-id"
        )


def test_two_store_connections_cannot_claim_same_task_twice(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "concurrent-claim.db"
    first_store = TaskStore(db_path=db_path)
    second_store = TaskStore(db_path=db_path)
    spec = TaskSpec(
        task_id="single-concurrent-claim",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Only one worker can claim this task",
        output_contract="patch",
        status=TaskStatus.READY,
    )
    first_store.save_task(spec)
    barrier = Barrier(2)

    def claim(store: TaskStore, worker_id: str, request_id: str):
        barrier.wait()
        return store.claim_next_task(
            run_id="run-1", worker_id=worker_id, claim_request_id=request_id
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(claim, first_store, "worker-1", "request-1")
        second_future = executor.submit(claim, second_store, "worker-2", "request-2")
        results = [first_future.result(), second_future.result()]

    claims = [result.claim for result in results if result.claim is not None]
    assert len(claims) == 1
    assert sum(result.claim is None for result in results) == 1
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT status, attempt FROM tasks WHERE task_id = ?", (spec.task_id,)
        ).fetchone() == (TaskStatus.RUNNING.value, 1)
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?", (spec.task_id,)
        ).fetchone()[0] == 1


def test_legacy_running_task_is_never_claimed(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-running-claim.db"
    legacy = TaskSpec(
        task_id="legacy-running-claim",
        run_id="run-1",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Remain fail-closed",
        output_contract="patch",
        status=TaskStatus.RUNNING,
        attempt=1,
    )
    make_legacy_database(db_path, legacy)
    store = TaskStore(db_path=db_path)

    result = store.claim_next_task(
        run_id="run-1", worker_id="worker-1", claim_request_id="legacy-claim"
    )

    assert result.claim is None
    assert store.get_task(legacy.task_id) == legacy
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?",
            (legacy.task_id,),
        ).fetchone()[0] == 0


def test_valid_attempt_completes_task_and_persists_consistent_terminal_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "fenced-completion.db"
    store = TaskStore(db_path=db_path)
    claim = claim_task_at_review(store, "fenced-completion-task")

    completed = transition_claim(
        store, claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED
    )
    assert completed.status == TaskStatus.ACCEPTED
    store.close()

    reopened = TaskStore(db_path=db_path)
    assert reopened.get_task(claim.task.task_id).status == TaskStatus.ACCEPTED
    with sqlite3.connect(db_path) as connection:
        attempt = connection.execute(
            "SELECT attempt_id, attempt_no, status, finished_at FROM task_attempts "
            "WHERE task_id = ?",
            (claim.task.task_id,),
        ).fetchone()
    assert attempt[:3] == (claim.attempt_id, claim.attempt_no, "succeeded")
    assert attempt[3] is not None


def test_wrong_attempt_id_is_rejected_without_changing_task_or_attempt(
    task_store: TaskStore,
) -> None:
    claim = claim_task_at_review(task_store, "wrong-attempt-id-task")

    with pytest.raises(AttemptFencingError):
        task_store.transition_attempt_status(
            task_id=claim.task.task_id,
            attempt_id="not-the-claimed-attempt",
            attempt_no=claim.attempt_no,
            expected_attempt_status=AttemptStatus.RUNNING,
            expected_task_status=TaskStatus.REVIEW,
            new_task_status=TaskStatus.ACCEPTED,
        )

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("running",)


def test_wrong_attempt_number_is_rejected(task_store: TaskStore) -> None:
    claim = claim_task_at_review(task_store, "wrong-attempt-number-task")

    with pytest.raises(AttemptFencingError):
        task_store.transition_attempt_status(
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no + 1,
            expected_attempt_status=AttemptStatus.RUNNING,
            expected_task_status=TaskStatus.REVIEW,
            new_task_status=TaskStatus.ACCEPTED,
        )

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW


def test_wrong_task_id_is_rejected(task_store: TaskStore) -> None:
    claim = claim_task_at_review(task_store, "wrong-task-id-owner")

    with pytest.raises(TaskNotFoundError):
        task_store.transition_attempt_status(
            task_id="missing-task-id",
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            expected_attempt_status=AttemptStatus.RUNNING,
            expected_task_status=TaskStatus.REVIEW,
            new_task_status=TaskStatus.ACCEPTED,
        )


def test_attempt_cannot_change_status_of_another_claimed_task(
    task_store: TaskStore,
) -> None:
    first_claim = claim_task_at_review(task_store, "first-fenced-owner")
    second_claim = claim_task_at_review(task_store, "second-fenced-owner")

    with pytest.raises(AttemptFencingError):
        task_store.transition_attempt_status(
            task_id=second_claim.task.task_id,
            attempt_id=first_claim.attempt_id,
            attempt_no=first_claim.attempt_no,
            expected_attempt_status=AttemptStatus.RUNNING,
            expected_task_status=TaskStatus.REVIEW,
            new_task_status=TaskStatus.ACCEPTED,
        )

    assert task_store.get_task(first_claim.task.task_id).status == TaskStatus.REVIEW
    assert task_store.get_task(second_claim.task.task_id).status == TaskStatus.REVIEW


def test_terminal_attempt_cannot_complete_task_twice(task_store: TaskStore) -> None:
    claim = claim_task_at_review(task_store, "terminal-attempt-task")
    transition_claim(task_store, claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED)

    with pytest.raises(AttemptFencingError):
        transition_claim(task_store, claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED)

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.ACCEPTED


def test_wrong_expected_attempt_or_task_status_is_rejected(
    task_store: TaskStore,
) -> None:
    claim = claim_task_at_review(task_store, "wrong-expected-status-task")

    with pytest.raises(AttemptFencingError):
        task_store.transition_attempt_status(
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            expected_attempt_status=AttemptStatus.FAILED,
            expected_task_status=TaskStatus.REVIEW,
            new_task_status=TaskStatus.ACCEPTED,
        )
    with pytest.raises(AttemptFencingError):
        task_store.transition_attempt_status(
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            expected_attempt_status=AttemptStatus.RUNNING,
            expected_task_status=TaskStatus.TESTING,
            new_task_status=TaskStatus.REVIEW,
        )

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW


def test_expired_attempt_cannot_change_task_status(task_store: TaskStore) -> None:
    claim = claim_task_at_review(task_store, "expired-attempt-task")
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            "UPDATE task_attempts SET lease_expires_at = ? WHERE attempt_id = ?",
            ("2000-01-01T00:00:00+00:00", claim.attempt_id),
        )

    with pytest.raises(AttemptFencingError):
        transition_claim(
            task_store, claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED
        )

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("running",)


def test_generic_status_api_cannot_bypass_attempt_fencing(
    task_store: TaskStore,
) -> None:
    claim = claim_task_at_review(task_store, "generic-status-bypass-task")

    with pytest.raises(TaskAttemptRequiredError):
        task_store.update_status(claim.task.task_id, TaskStatus.ACCEPTED)

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("running",)


def test_legacy_running_without_attempt_remains_fail_closed(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-running-fenced.db"
    legacy = TaskSpec(
        task_id="legacy-running-fenced",
        run_id="run-legacy",
        agent_role=AgentRole.IMPLEMENTER,
        goal="No synthetic attempt",
        output_contract="patch",
        status=TaskStatus.RUNNING,
        attempt=1,
    )
    make_legacy_database(db_path, legacy)
    store = TaskStore(db_path=db_path)

    with pytest.raises(TaskAttemptRequiredError):
        store.update_status(legacy.task_id, TaskStatus.FAILED)
    with pytest.raises(AttemptFencingError):
        store.transition_attempt_status(
            task_id=legacy.task_id,
            attempt_id="no-legacy-attempt",
            attempt_no=legacy.attempt,
            expected_attempt_status=AttemptStatus.RUNNING,
            expected_task_status=TaskStatus.RUNNING,
            new_task_status=TaskStatus.FAILED,
        )

    assert store.get_task(legacy.task_id) == legacy
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?",
            (legacy.task_id,),
        ).fetchone()[0] == 0


def test_attempt_update_failure_rolls_back_task_status(task_store: TaskStore) -> None:
    claim = claim_task_at_review(task_store, "attempt-write-rollback-task")
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            f"""
            CREATE TRIGGER fail_attempt_finish
            BEFORE UPDATE OF status ON task_attempts
            WHEN OLD.attempt_id = '{claim.attempt_id}'
            BEGIN
                SELECT RAISE(ABORT, 'forced attempt finish failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced attempt finish failure"):
        transition_claim(task_store, claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED)

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status, finished_at FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("running", None)


def test_task_update_failure_rolls_back_completion(task_store: TaskStore) -> None:
    claim = claim_task_at_review(task_store, "task-write-rollback-task")
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            f"""
            CREATE TRIGGER fail_task_finish
            BEFORE UPDATE OF status ON tasks
            WHEN OLD.task_id = '{claim.task.task_id}'
            BEGIN
                SELECT RAISE(ABORT, 'forced task finish failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced task finish failure"):
        transition_claim(task_store, claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED)

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status, finished_at FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("running", None)


def expire_attempt(store: TaskStore, claim: TaskClaim) -> None:
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE task_attempts SET lease_expires_at = ? WHERE attempt_id = ?",
            ("2000-01-01T00:00:00+00:00", claim.attempt_id),
        )


def test_heartbeat_extends_only_the_matching_active_worker_lease(
    task_store: TaskStore,
) -> None:
    claim = claim_task_at_review(task_store, "heartbeat-valid")
    before = datetime.now(UTC)

    expiry = task_store.heartbeat_attempt(
        task_id=claim.task.task_id,
        attempt_id=claim.attempt_id,
        attempt_no=claim.attempt_no,
        worker_id=claim.worker_id,
    )
    after = datetime.now(UTC)

    assert expiry > claim.lease_expires_at
    assert before + timedelta(seconds=claim.task.budget.timeout_seconds) <= expiry
    assert expiry <= after + timedelta(seconds=claim.task.budget.timeout_seconds)
    with sqlite3.connect(task_store.db_path) as connection:
        row = connection.execute(
            "SELECT heartbeat_at, lease_expires_at FROM task_attempts "
            "WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone()
    heartbeat_at = datetime.fromisoformat(row[0])
    assert before <= heartbeat_at <= after
    assert row[1] == expiry.isoformat()


def test_heartbeat_rejects_wrong_worker_and_expired_attempt(
    task_store: TaskStore,
) -> None:
    claim = claim_task_at_review(task_store, "heartbeat-invalid")
    with pytest.raises(AttemptFencingError):
        task_store.heartbeat_attempt(
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            worker_id="different-worker",
        )

    expire_attempt(task_store, claim)
    with pytest.raises(AttemptFencingError):
        task_store.heartbeat_attempt(
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            worker_id=claim.worker_id,
        )


def test_heartbeat_waiting_for_write_lock_uses_time_after_lock(
    task_store: TaskStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = claim_task_at_review(task_store, "heartbeat-delayed-lock")
    expiry = datetime(2030, 1, 1, tzinfo=UTC)
    before_expiry = expiry - timedelta(seconds=1)
    after_expiry = expiry + timedelta(seconds=1)
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            "UPDATE task_attempts SET lease_expires_at = ? WHERE attempt_id = ?",
            (expiry.isoformat(), claim.attempt_id),
        )

    clock = {"now": before_expiry}
    monkeypatch.setattr(task_store_module, "_utc_now", lambda: clock["now"])
    lock_holder_store = TaskStore(task_store.db_path)
    lock_holder = lock_holder_store._get_connection()
    lock_holder.execute("BEGIN IMMEDIATE")
    begin_attempted = Event()
    original_get_connection = task_store._get_connection

    class SignalingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, statement: str, parameters: tuple = ()):
            if statement == "BEGIN IMMEDIATE":
                begin_attempted.set()
            return self.connection.execute(statement, parameters)

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

    monkeypatch.setattr(
        task_store,
        "_get_connection",
        lambda: SignalingConnection(original_get_connection()),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        heartbeat = executor.submit(
            task_store.heartbeat_attempt,
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            worker_id=claim.worker_id,
        )
        assert begin_attempted.wait(timeout=2)
        clock["now"] = after_expiry
        lock_holder.commit()
        with pytest.raises(AttemptFencingError):
            heartbeat.result(timeout=5)
    lock_holder.close()

    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT heartbeat_at, lease_expires_at, status FROM task_attempts "
            "WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == (
            None,
            expiry.isoformat(),
            AttemptStatus.RUNNING.value,
        )


def test_recovery_before_heartbeat_prevents_lease_resurrection(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recovery-before-heartbeat.db"
    recovery_store = TaskStore(db_path)
    claim = claim_task_at_review(recovery_store, "recovery-first-heartbeat")
    heartbeat_store = TaskStore(db_path)
    expire_attempt(recovery_store, claim)

    assert len(recovery_store.recover_expired_attempts()) == 1
    with pytest.raises(AttemptFencingError):
        heartbeat_store.heartbeat_attempt(
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            worker_id=claim.worker_id,
        )

    reopened = TaskStore(db_path)
    assert reopened.get_task(claim.task.task_id).status == TaskStatus.READY
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == (AttemptStatus.ABANDONED.value,)


def test_valid_heartbeat_competes_safely_with_recovery(tmp_path: Path) -> None:
    db_path = tmp_path / "heartbeat-recovery-race.db"
    first_store = TaskStore(db_path)
    claim = claim_task_at_review(first_store, "heartbeat-recovery-race")
    second_store = TaskStore(db_path)
    barrier = Barrier(2)

    def heartbeat() -> datetime:
        barrier.wait()
        return first_store.heartbeat_attempt(
            task_id=claim.task.task_id,
            attempt_id=claim.attempt_id,
            attempt_no=claim.attempt_no,
            worker_id=claim.worker_id,
        )

    def recover() -> list[TaskSpec]:
        barrier.wait()
        return second_store.recover_expired_attempts()

    with ThreadPoolExecutor(max_workers=2) as executor:
        heartbeat_future = executor.submit(heartbeat)
        recovery_future = executor.submit(recover)
        renewed_expiry = heartbeat_future.result(timeout=5)
        recovered = recovery_future.result(timeout=5)

    assert recovered == []
    assert renewed_expiry > claim.lease_expires_at
    reopened = TaskStore(db_path)
    assert reopened.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT status, lease_expires_at FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone()
    assert row[0] == AttemptStatus.RUNNING.value
    assert datetime.fromisoformat(row[1]) == renewed_expiry


@pytest.mark.parametrize(
    ("lease_expires_at", "completion_wins"),
    [
        ("2999-01-01T00:00:00+00:00", True),
        ("2000-01-01T00:00:00+00:00", False),
    ],
)
def test_completion_and_recovery_compete_atomically(
    tmp_path: Path,
    lease_expires_at: str,
    completion_wins: bool,
) -> None:
    db_path = tmp_path / f"completion-recovery-{completion_wins}.db"
    completion_store = TaskStore(db_path)
    claim = claim_task_at_review(completion_store, f"completion-race-{completion_wins}")
    recovery_store = TaskStore(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE task_attempts SET lease_expires_at = ? WHERE attempt_id = ?",
            (lease_expires_at, claim.attempt_id),
        )
    barrier = Barrier(2)

    def complete() -> TaskSpec | None:
        barrier.wait()
        try:
            return transition_claim(
                completion_store, claim, TaskStatus.REVIEW, TaskStatus.ACCEPTED
            )
        except AttemptFencingError:
            return None

    def recover() -> list[TaskSpec]:
        barrier.wait()
        return recovery_store.recover_expired_attempts()

    with ThreadPoolExecutor(max_workers=2) as executor:
        completion_future = executor.submit(complete)
        recovery_future = executor.submit(recover)
        completed = completion_future.result(timeout=5)
        recovered = recovery_future.result(timeout=5)

    reopened = TaskStore(db_path)
    persisted_task = reopened.get_task(claim.task.task_id)
    with sqlite3.connect(db_path) as connection:
        attempt_status = connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone()[0]

    assert persisted_task.attempt == 1
    if completion_wins:
        assert completed is not None
        assert recovered == []
        assert persisted_task.status == TaskStatus.ACCEPTED
        assert attempt_status == AttemptStatus.SUCCEEDED.value
    else:
        assert completed is None
        assert [task.task_id for task in recovered] == [claim.task.task_id]
        assert persisted_task.status == TaskStatus.READY
        assert attempt_status == AttemptStatus.ABANDONED.value


def test_expired_attempt_is_abandoned_and_task_requeued_atomically(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "expired-recovery.db"
    store = TaskStore(db_path)
    claim = claim_task_at_review(store, "expired-retry")
    expire_attempt(store, claim)

    recovered = store.recover_expired_attempts()

    assert len(recovered) == 1
    assert recovered[0].status == TaskStatus.READY
    assert recovered[0].attempt == 1
    with sqlite3.connect(db_path) as connection:
        attempt = connection.execute(
            "SELECT status, finished_at, reason_code FROM task_attempts "
            "WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone()
    assert attempt[0] == "abandoned"
    assert attempt[1] is not None
    assert attempt[2] == "lease_expired"
    store.close()

    reopened = TaskStore(db_path)
    assert reopened.get_task(claim.task.task_id).status == TaskStatus.READY
    assert reopened.recover_expired_attempts() == []


def test_expired_attempt_exhaustion_fails_task_without_increment(
    task_store: TaskStore,
) -> None:
    spec = TaskSpec(
        task_id="expired-exhausted",
        run_id="run-exhausted",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Exhaust after crash",
        output_contract="patch",
        status=TaskStatus.READY,
        budget=TaskBudget(max_attempts=1),
    )
    task_store.save_task(spec)
    claim_result = task_store.claim_next_task(
        run_id=spec.run_id,
        worker_id="worker-exhausted",
        claim_request_id="claim-exhausted",
    )
    assert claim_result.claim is not None
    claim = claim_result.claim
    transition_claim(task_store, claim, TaskStatus.RUNNING, TaskStatus.TESTING)
    transition_claim(task_store, claim, TaskStatus.TESTING, TaskStatus.REVIEW)
    expire_attempt(task_store, claim)

    recovered = task_store.recover_expired_attempts()

    assert [task.status for task in recovered] == [TaskStatus.FAILED]
    assert task_store.get_task(spec.task_id).attempt == 1
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("abandoned",)


def test_recovery_task_write_failure_rolls_back_attempt_abandonment(
    task_store: TaskStore,
) -> None:
    claim = claim_task_at_review(task_store, "recovery-rollback")
    expire_attempt(task_store, claim)
    with sqlite3.connect(task_store.db_path) as connection:
        connection.execute(
            f"""
            CREATE TRIGGER fail_recovery_task_update
            BEFORE UPDATE OF status ON tasks
            WHEN OLD.task_id = '{claim.task.task_id}'
            BEGIN
                SELECT RAISE(ABORT, 'forced recovery task update failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced recovery task update failure"):
        task_store.recover_expired_attempts()

    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status, finished_at, reason_code FROM task_attempts "
            "WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("running", None, None)


def test_recovery_waiting_for_write_lock_uses_time_after_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "recovery-delayed-lock.db"
    store = TaskStore(db_path)
    claim = claim_task_at_review(store, "recovery-delayed-lock")
    expiry = datetime(2030, 1, 1, tzinfo=UTC)
    before_expiry = expiry - timedelta(seconds=1)
    after_expiry = expiry + timedelta(seconds=1)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE task_attempts SET lease_expires_at = ? WHERE attempt_id = ?",
            (expiry.isoformat(), claim.attempt_id),
        )

    clock = {"now": before_expiry}
    monkeypatch.setattr(task_store_module, "_utc_now", lambda: clock["now"])
    lock_holder_store = TaskStore(db_path)
    lock_holder = lock_holder_store._get_connection()
    lock_holder.execute("BEGIN IMMEDIATE")
    begin_attempted = Event()
    original_get_connection = store._get_connection

    class SignalingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, statement: str, parameters: tuple = ()):
            if statement == "BEGIN IMMEDIATE":
                begin_attempted.set()
            return self.connection.execute(statement, parameters)

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

    monkeypatch.setattr(
        store,
        "_get_connection",
        lambda: SignalingConnection(original_get_connection()),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        recovery = executor.submit(store.recover_expired_attempts)
        assert begin_attempted.wait(timeout=2)
        clock["now"] = after_expiry
        lock_holder.commit()
        recovered = recovery.result(timeout=5)
    lock_holder.close()

    assert [task.task_id for task in recovered] == [claim.task.task_id]
    assert recovered[0].status == TaskStatus.READY
    assert recovered[0].attempt == 1
    reopened = TaskStore(db_path)
    assert reopened.get_task(claim.task.task_id).status == TaskStatus.READY
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT status, finished_at FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == (AttemptStatus.ABANDONED.value, after_expiry.isoformat())


def test_recovery_does_not_abandon_unexpired_attempt(
    task_store: TaskStore,
) -> None:
    claim = claim_task_at_review(task_store, "recovery-unexpired")

    assert task_store.recover_expired_attempts() == []
    assert task_store.get_task(claim.task.task_id).status == TaskStatus.REVIEW
    with sqlite3.connect(task_store.db_path) as connection:
        assert connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == (AttemptStatus.RUNNING.value,)


def test_recovery_api_does_not_accept_caller_selected_time() -> None:
    assert "now" not in inspect.signature(TaskStore.recover_expired_attempts).parameters


def test_recovery_leaves_legacy_running_without_attempt_untouched(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-recovery.db"
    legacy = TaskSpec(
        task_id="legacy-recovery-running",
        run_id="legacy-run",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Remain fail-closed",
        output_contract="patch",
        status=TaskStatus.RUNNING,
        attempt=1,
    )
    make_legacy_database(db_path, legacy)
    store = TaskStore(db_path)

    assert store.recover_expired_attempts() == []
    assert store.get_task(legacy.task_id) == legacy
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM task_attempts WHERE task_id = ?",
            (legacy.task_id,),
        ).fetchone() == (0,)


def test_two_stores_recover_an_expired_attempt_only_once(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent-recovery.db"
    first_store = TaskStore(db_path)
    claim = claim_task_at_review(first_store, "concurrent-expiry")
    expire_attempt(first_store, claim)
    second_store = TaskStore(db_path)

    barrier = Barrier(2)

    def recover(store: TaskStore) -> list[TaskSpec]:
        barrier.wait()
        return store.recover_expired_attempts()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(recover, (first_store, second_store)))

    assert sum(len(result) for result in results) == 1
    assert first_store.get_task(claim.task.task_id).status == TaskStatus.READY
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT status FROM task_attempts WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone() == ("abandoned",)
