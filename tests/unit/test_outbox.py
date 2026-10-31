from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from swarm_forge.contracts import (
    EventType,
    LifecycleEvent,
    OperationContext,
    TaskStatus,
)
from swarm_forge.storage import outbox as outbox_module
from swarm_forge.storage.outbox import (
    EventDiscriminatorConflictError,
    EventIdentityConflictError,
    OperationIdempotencyConflictError,
    OperationWriteResult,
    OutboxManifestMismatchError,
    OutboxTransactionRequiredError,
    get_lifecycle_event,
    list_pending_lifecycle_events,
    persist_operation,
)
from swarm_forge.storage.task_store import TaskStore


def make_event(
    *,
    operation_id: str = "operation-001",
    idempotency_key: str = "request-001",
    discriminator: str = "task-created",
    event_type: EventType = EventType.TASK_CREATED,
    payload: dict[str, object] | None = None,
) -> LifecycleEvent:
    return LifecycleEvent.for_operation(
        context=OperationContext(
            operation_id=operation_id,
            correlation_id="correlation-001",
            source="workflow_engine",
        ),
        idempotency_key=idempotency_key,
        event_discriminator=discriminator,
        run_id="run-001",
        task_id="task-001",
        event_type=event_type,
        new_state=(
            TaskStatus.CREATED
            if event_type == EventType.TASK_CREATED
            else TaskStatus.VALIDATED
        ),
        **(
            {}
            if event_type == EventType.TASK_CREATED
            else {
                "previous_state": TaskStatus.CREATED,
                "transition_cause": "validation",
            }
        ),
        payload=payload or {},
    )


def make_events(
    *, operation_id: str = "operation-001", idempotency_key: str = "request-001"
) -> tuple[LifecycleEvent, LifecycleEvent]:
    return (
        make_event(
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            discriminator="task-created",
        ),
        make_event(
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            discriminator="task-validated",
            event_type=EventType.TASK_STATUS_CHANGED,
        ),
    )


def begin_store_transaction(store: TaskStore) -> sqlite3.Connection:
    connection = store._get_connection()
    connection.execute("BEGIN IMMEDIATE")
    return connection


def insert_raw_outbox_row(
    connection: sqlite3.Connection, event: LifecycleEvent
) -> None:
    canonical = outbox_module._canonical_event(event)
    timestamp = "2026-02-01T00:00:00.000000+00:00"
    connection.execute(
        """
        INSERT INTO lifecycle_outbox (
            event_id, operation_id, idempotency_key, event_discriminator,
            schema_version, run_id, task_id, attempt_id, correlation_id,
            event_type, occurred_at, event_json, status, retry_count,
            next_attempt_at, created_at
        ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
        """,
        (
            event.event_id,
            event.operation_id,
            event.idempotency_key,
            event.event_discriminator,
            event.run_id,
            event.task_id,
            event.attempt_id,
            event.correlation_id,
            event.event_type.value,
            event.occurred_at.isoformat(),
            canonical.event_json,
            timestamp,
            timestamp,
        ),
    )


def test_new_database_has_schema_version_three_and_outbox_tables(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "new-outbox.db")
    connection = store._get_connection()
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "lifecycle_outbox_operations" in table_names
        assert "lifecycle_outbox" in table_names
    finally:
        connection.close()


def test_v2_migration_preserves_task_attempt_and_claim_data(tmp_path: Path) -> None:
    db_path = tmp_path / "v2-outbox.db"
    store = TaskStore(db_path)
    task = store.list_tasks()
    assert task == []
    connection = store._get_connection()
    try:
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, run_id, agent_role, goal, status, attempt, created_at,
                budget_json, depends_on_json, input_refs_json, output_contract,
                allowed_tools_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-task",
                "legacy-run",
                "implementer",
                "old task",
                "failed",
                1,
                "2026-01-01T00:00:00+00:00",
                '{"max_attempts": 3, "timeout_seconds": 60}',
                "[]",
                "[]",
                "result",
                "[]",
            ),
        )
        connection.execute(
            """
            INSERT INTO task_attempts (
                attempt_id, task_id, attempt_no, status, started_at
            ) VALUES ('legacy-attempt', 'legacy-task', 1, 'failed', ?)
            """,
            ("2026-01-01T00:00:00+00:00",),
        )
        connection.execute(
            """
            INSERT INTO claim_requests (
                claim_request_id, worker_id, run_id, response_json, created_at
            ) VALUES ('legacy-claim', 'worker-1', 'legacy-run', '{}', ?)
            """,
            ("2026-01-01T00:00:00+00:00",),
        )
        connection.commit()

        connection.execute("DROP TABLE lifecycle_outbox")
        connection.execute("DROP TABLE lifecycle_outbox_operations")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()

    migrated = TaskStore(db_path)
    check = migrated._get_connection()
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 3
        assert check.execute(
            "SELECT task_id FROM tasks WHERE task_id = 'legacy-task'"
        ).fetchone()[0] == "legacy-task"
        assert check.execute(
            "SELECT attempt_id FROM task_attempts WHERE attempt_id = 'legacy-attempt'"
        ).fetchone()[0] == "legacy-attempt"
        assert check.execute(
            "SELECT claim_request_id FROM claim_requests WHERE claim_request_id = 'legacy-claim'"
        ).fetchone()[0] == "legacy-claim"
        assert check.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 0
    finally:
        check.close()


def test_v3_migration_failure_rolls_back_schema_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "migration-rollback.db"
    original = TaskStore(db_path)
    connection = original._get_connection()
    try:
        connection.execute("DROP TABLE lifecycle_outbox")
        connection.execute("DROP TABLE lifecycle_outbox_operations")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()

    original_get_connection = TaskStore._get_connection

    def deny_outbox_index(store: TaskStore) -> sqlite3.Connection:
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
                and argument_one == "idx_lifecycle_outbox_pending"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        return connection

    monkeypatch.setattr(TaskStore, "_get_connection", deny_outbox_index)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        TaskStore(db_path)

    check = sqlite3.connect(db_path)
    try:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 2
        assert check.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name IN "
            "('lifecycle_outbox', 'lifecycle_outbox_operations')"
        ).fetchone()[0] == 0
    finally:
        check.close()


def test_insert_read_list_and_transaction_ownership(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-primitives.db")
    connection = begin_store_transaction(store)
    events = make_events()
    result = persist_operation(
        connection,
        operation_id="operation-001",
        idempotency_key="request-001",
        operation_fingerprint="fingerprint-001",
        events=events,
    )

    assert result == OperationWriteResult.CREATED
    assert connection.in_transaction
    assert get_lifecycle_event(connection, events[0].event_id) == events[0]
    assert list_pending_lifecycle_events(connection) == list(events)

    observer = sqlite3.connect(store.db_path)
    try:
        assert observer.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 0
    finally:
        observer.close()
    connection.commit()
    connection.close()


def test_write_api_requires_active_transaction_before_writing(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-no-transaction.db")
    connection = store._get_connection()
    event = make_event()
    with pytest.raises(OutboxTransactionRequiredError, match="active"):
        persist_operation(
            connection,
            operation_id=event.operation_id,
            idempotency_key=event.idempotency_key,
            operation_fingerprint="fingerprint-001",
            events=[event],
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 0
    connection.close()


def test_caller_rollback_after_success_removes_receipt_and_events(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "outbox-caller-rollback-success.db")
    connection = begin_store_transaction(store)
    events = make_events()
    persist_operation(
        connection,
        operation_id="operation-001",
        idempotency_key="request-001",
        operation_fingerprint="fingerprint-001",
        events=events,
    )
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 2
    connection.rollback()
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 0
    connection.close()


def test_caller_rollback_removes_partial_write_after_late_insert_failure(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "outbox-partial-rollback.db")
    connection = begin_store_transaction(store)
    events = make_events()
    connection.execute(
        """
        CREATE TEMP TRIGGER fail_second_outbox_event
        BEFORE INSERT ON lifecycle_outbox
        WHEN NEW.event_discriminator = 'task-validated'
        BEGIN
            SELECT RAISE(ABORT, 'injected event conflict');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected event conflict"):
        persist_operation(
            connection,
            operation_id="operation-001",
            idempotency_key="request-001",
            operation_fingerprint="fingerprint-001",
            events=events,
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 1

    connection.rollback()
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 0
    connection.close()


def test_register_and_insert_primitives_are_idempotent(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-insert.db")
    connection = begin_store_transaction(store)
    event = make_events()[0]

    assert outbox_module._register_operation(
        connection,
        operation_id=event.operation_id,
        idempotency_key=event.idempotency_key,
        operation_fingerprint="fingerprint-001",
        events=[event],
    )
    assert outbox_module._insert_lifecycle_event(connection, event)
    assert not outbox_module._insert_lifecycle_event(connection, event)
    connection.rollback()
    connection.close()


def test_exact_operation_replay_does_not_duplicate_events(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-replay.db")
    connection = begin_store_transaction(store)
    events = make_events()
    arguments = {
        "operation_id": "operation-001",
        "idempotency_key": "request-001",
        "operation_fingerprint": "fingerprint-001",
        "events": events,
    }
    assert persist_operation(connection, **arguments) == OperationWriteResult.CREATED
    assert persist_operation(connection, **arguments) == OperationWriteResult.REPLAYED
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 2
    connection.rollback()
    connection.close()


def test_replay_detects_an_extra_event_row_outside_manifest(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-extra-row.db")
    connection = begin_store_transaction(store)
    expected = make_event()
    persist_operation(
        connection,
        operation_id=expected.operation_id,
        idempotency_key=expected.idempotency_key,
        operation_fingerprint="fingerprint-001",
        events=[expected],
    )
    insert_raw_outbox_row(
        connection,
        make_event(discriminator="unmanifested-event"),
    )

    with pytest.raises(OutboxManifestMismatchError, match="exactly match"):
        persist_operation(
            connection,
            operation_id=expected.operation_id,
            idempotency_key=expected.idempotency_key,
            operation_fingerprint="fingerprint-001",
            events=[expected],
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox WHERE operation_id = ? "
        "AND idempotency_key = ?",
        (expected.operation_id, expected.idempotency_key),
    ).fetchone()[0] == 2
    connection.rollback()
    connection.close()


def test_replay_detects_a_missing_event_row(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-missing-row.db")
    connection = begin_store_transaction(store)
    events = make_events()
    persist_operation(
        connection,
        operation_id="operation-001",
        idempotency_key="request-001",
        operation_fingerprint="fingerprint-001",
        events=events,
    )
    connection.execute(
        "DELETE FROM lifecycle_outbox WHERE event_id = ?", (events[1].event_id,)
    )

    with pytest.raises(OutboxManifestMismatchError, match="exactly match"):
        persist_operation(
            connection,
            operation_id="operation-001",
            idempotency_key="request-001",
            operation_fingerprint="fingerprint-001",
            events=events,
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox WHERE operation_id = ? "
        "AND idempotency_key = ?",
        ("operation-001", "request-001"),
    ).fetchone()[0] == 1
    connection.rollback()
    connection.close()


def test_event_with_wrong_idempotency_key_is_rejected_before_write(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "outbox-wrong-key.db")
    connection = begin_store_transaction(store)
    with pytest.raises(OutboxManifestMismatchError, match="match the operation"):
        persist_operation(
            connection,
            operation_id="operation-001",
            idempotency_key="request-001",
            operation_fingerprint="fingerprint-001",
            events=[make_event(idempotency_key="request-other")],
        )
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 0
    connection.rollback()
    connection.close()


def test_same_idempotency_key_on_different_operations_is_distinct(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "outbox-key-scope.db")
    connection = begin_store_transaction(store)
    first = make_event(operation_id="operation-one", idempotency_key="shared-key")
    second = make_event(operation_id="operation-two", idempotency_key="shared-key")

    assert first.event_id != second.event_id
    assert persist_operation(
        connection,
        operation_id=first.operation_id,
        idempotency_key=first.idempotency_key,
        operation_fingerprint="fingerprint-one",
        events=[first],
    ) == OperationWriteResult.CREATED
    assert persist_operation(
        connection,
        operation_id=second.operation_id,
        idempotency_key=second.idempotency_key,
        operation_fingerprint="fingerprint-two",
        events=[second],
    ) == OperationWriteResult.CREATED
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations "
        "WHERE idempotency_key = 'shared-key'"
    ).fetchone()[0] == 2
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox WHERE event_id IN (?, ?)",
        (first.event_id, second.event_id),
    ).fetchone()[0] == 2
    connection.rollback()
    connection.close()


def test_operation_fingerprint_conflict_is_rejected(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-operation-conflict.db")
    connection = begin_store_transaction(store)
    events = make_events()
    persist_operation(
        connection,
        operation_id="operation-001",
        idempotency_key="request-001",
        operation_fingerprint="fingerprint-001",
        events=events,
    )
    with pytest.raises(OperationIdempotencyConflictError):
        persist_operation(
            connection,
            operation_id="operation-001",
            idempotency_key="request-001",
            operation_fingerprint="fingerprint-002",
            events=events,
        )
    assert connection.execute(
        "SELECT operation_fingerprint FROM lifecycle_outbox_operations"
    ).fetchone()[0] == "fingerprint-001"
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 2
    connection.rollback()
    connection.close()


def test_manifest_conflict_is_rejected(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-manifest-conflict.db")
    connection = begin_store_transaction(store)
    original = [make_events()[0]]
    persist_operation(
        connection,
        operation_id="operation-001",
        idempotency_key="request-001",
        operation_fingerprint="fingerprint-001",
        events=original,
    )
    with pytest.raises(OutboxManifestMismatchError):
        persist_operation(
            connection,
            operation_id="operation-001",
            idempotency_key="request-001",
            operation_fingerprint="fingerprint-001",
            events=[make_event(discriminator="different-event")],
        )
    connection.rollback()
    connection.close()


def test_replay_with_same_event_id_but_different_content_is_rejected(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "outbox-event-conflict.db")
    connection = begin_store_transaction(store)
    original = make_event(payload={"value": "original"})
    persist_operation(
        connection,
        operation_id=original.operation_id,
        idempotency_key=original.idempotency_key,
        operation_fingerprint="fingerprint-001",
        events=[original],
    )

    changed = make_event(payload={"value": "changed"})
    assert changed.event_id == original.event_id
    with pytest.raises(EventIdentityConflictError):
        persist_operation(
            connection,
            operation_id=changed.operation_id,
            idempotency_key=changed.idempotency_key,
            operation_fingerprint="fingerprint-001",
            events=[changed],
        )
    assert connection.execute(
        "SELECT event_json FROM lifecycle_outbox WHERE event_id = ?",
        (original.event_id,),
    ).fetchone()[0] == outbox_module._canonical_event(original).event_json
    connection.rollback()
    connection.close()


def test_canonical_content_ignores_payload_dictionary_key_order(
    tmp_path: Path,
) -> None:
    store = TaskStore(tmp_path / "outbox-canonical-order.db")
    connection = begin_store_transaction(store)
    first = make_event(payload={"outer": {"a": 1, "b": 2}})
    reordered = first.model_copy(
        update={"payload": {"outer": {"b": 2, "a": 1}}}
    )

    assert first.event_id == reordered.event_id
    assert persist_operation(
        connection,
        operation_id=first.operation_id,
        idempotency_key=first.idempotency_key,
        operation_fingerprint="fingerprint-001",
        events=[first],
    ) == OperationWriteResult.CREATED
    assert persist_operation(
        connection,
        operation_id=reordered.operation_id,
        idempotency_key=reordered.idempotency_key,
        operation_fingerprint="fingerprint-001",
        events=[reordered],
    ) == OperationWriteResult.REPLAYED
    connection.rollback()
    connection.close()


def test_non_finite_payload_number_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_event(payload={"invalid": float("nan")})


def test_one_operation_can_contain_multiple_events(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-multiple-events.db")
    connection = begin_store_transaction(store)
    events = make_events()
    assert persist_operation(
        connection,
        operation_id="operation-001",
        idempotency_key="request-001",
        operation_fingerprint="fingerprint-001",
        events=events,
    ) == OperationWriteResult.CREATED
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 2
    connection.rollback()
    connection.close()


def test_discriminator_conflict_is_rejected(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-discriminator-conflict.db")
    connection = begin_store_transaction(store)
    first = make_event(idempotency_key="request-001", payload={"value": 1})
    persist_operation(
        connection,
        operation_id=first.operation_id,
        idempotency_key=first.idempotency_key,
        operation_fingerprint="fingerprint-001",
        events=[first],
    )
    second = make_event(idempotency_key="request-002", payload={"value": 2})
    with pytest.raises(EventDiscriminatorConflictError):
        persist_operation(
            connection,
            operation_id=second.operation_id,
            idempotency_key=second.idempotency_key,
            operation_fingerprint="fingerprint-002",
            events=[second],
        )
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox_operations"
    ).fetchone()[0] == 1
    connection.rollback()
    connection.close()


def test_outbox_composite_foreign_key_is_enforced(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-composite-fk.db")
    connection = begin_store_transaction(store)
    event = make_event()
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        insert_raw_outbox_row(connection, event)
    assert connection.execute("SELECT COUNT(*) FROM lifecycle_outbox").fetchone()[0] == 0
    connection.rollback()
    connection.close()


def test_outbox_check_constraints_are_enforced(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "outbox-checks.db")
    connection = begin_store_transaction(store)
    event = make_event()
    persist_operation(
        connection,
        operation_id=event.operation_id,
        idempotency_key=event.idempotency_key,
        operation_fingerprint="fingerprint-001",
        events=[event],
    )
    connection.commit()
    for sql, parameters in (
        (
            "UPDATE lifecycle_outbox SET status = 'unknown' WHERE event_id = ?",
            (event.event_id,),
        ),
        (
            "UPDATE lifecycle_outbox SET retry_count = -1 WHERE event_id = ?",
            (event.event_id,),
        ),
        (
            "UPDATE lifecycle_outbox SET lease_token = 'token' WHERE event_id = ?",
            (event.event_id,),
        ),
        (
            "UPDATE lifecycle_outbox SET status = 'delivered' WHERE event_id = ?",
            (event.event_id,),
        ),
        (
            """UPDATE lifecycle_outbox
               SET delivered_at = '2026-01-01T00:00:00Z'
               WHERE event_id = ?""",
            (event.event_id,),
        ),
    ):
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(sql, parameters)
        connection.rollback()
        row = connection.execute(
            """
            SELECT status, retry_count, lease_token, lease_expires_at, delivered_at
            FROM lifecycle_outbox WHERE event_id = ?
            """,
            (event.event_id,),
        ).fetchone()
        assert tuple(row) == ("pending", 0, None, None, None)
    connection.close()
