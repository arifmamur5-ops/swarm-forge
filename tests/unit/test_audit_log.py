import sqlite3

import pytest

from swarm_forge.storage.audit_log import AuditLog


def test_append_and_read_event(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")

    event = log.append(
        run_id="run-001",
        task_id="task-001",
        event_type="task_created",
        correlation_id="corr-001",
        payload={"role": "implementer"},
    )

    events = log.list_events()

    assert len(events) == 1
    assert events[0]["event_id"] == event["event_id"]
    assert events[0]["payload"] == {"role": "implementer"}
    assert log.verify_chain()

def test_hash_chain_links_events(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")

    first = log.append(
        run_id="run-001",
        event_type="run_created",
        correlation_id="corr-001",
        payload={"step": 1},
    )
    second = log.append(
        run_id="run-001",
        event_type="task_created",
        correlation_id="corr-001",
        payload={"step": 2},
    )

    assert second["prev_hash"] == first["event_hash"]
    assert log.verify_chain()

def test_events_survive_reopening_database(tmp_path) -> None:
    db_path = tmp_path / "audit.db"
    AuditLog(db_path).append(
        run_id="run-001",
        event_type="run_created",
        correlation_id="corr-001",
        payload={"persistent": True},
    )

    reopened = AuditLog(db_path)

    assert len(reopened.list_events()) == 1
    assert reopened.verify_chain()

def test_payload_tampering_is_detected(tmp_path) -> None:
    db_path = tmp_path / "audit.db"
    log = AuditLog(db_path)

    log.append(
        run_id="run-001",
        event_type="run_created",
        correlation_id="corr-001",
        payload={"safe": True},
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE audit_events SET payload_json = ?",
            ('{"safe":false}',),
        )

    assert not log.verify_chain()

def test_blank_run_id_is_rejected(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")

    with pytest.raises(ValueError):
        log.append(
            run_id=" ",
            event_type="run_created",
            correlation_id="corr-001",
            payload={},
        )

def test_filter_events_by_run_id(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")

    for run_id in ("run-a", "run-b", "run-a"):
        log.append(
            run_id=run_id,
            event_type="run_created",
            correlation_id=f"corr-{run_id}",
            payload={},
        )

    assert len(log.list_events(run_id="run-a")) == 2
    assert len(log.list_events(run_id="run-b")) == 1
