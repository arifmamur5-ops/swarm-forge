from datetime import UTC, datetime

import pytest

from swarm_forge.contracts import AuditEvent, EventType
from swarm_forge.storage.audit_log import AuditLog


def make_event() -> AuditEvent:
    return AuditEvent(
        event_id="event-test-001",
        run_id="run-test-001",
        task_id="task-test-001",
        event_type=EventType.TASK_CREATED,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        correlation_id="corr-test-001",
        payload={"agent_role": "implementer"},
    )

def test_append_event_returns_hashed_model(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")

    stored = log.append_event(make_event())

    assert isinstance(stored, AuditEvent)
    assert stored.event_id == "event-test-001"
    assert stored.prev_hash == "0" * 64
    assert stored.event_hash is not None
    assert len(stored.event_hash) == 64

def test_append_event_rejects_caller_supplied_hash(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")
    event = make_event().model_copy(
        update={"event_hash": "a" * 64}
    )

    with pytest.raises(ValueError, match="Hash fields"):
        log.append_event(event)

def test_event_type_is_persisted(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")

    log.append_event(make_event())

    events = log.list_events("run-test-001")

    assert len(events) == 1
    assert events[0]["event_type"] == EventType.TASK_CREATED.value
    assert events[0]["payload"] == {"agent_role": "implementer"}


def test_append_event_preserves_id_and_timestamp(tmp_path) -> None:
    log = AuditLog(tmp_path / "audit.db")
    original = make_event()

    returned = log.append_event(original)
    persisted = log.list_events("run-test-001")[0]

    assert returned.event_id == original.event_id
    assert returned.timestamp == original.timestamp
    assert persisted["event_id"] == original.event_id
    assert persisted["timestamp"] == original.timestamp.isoformat()
    assert returned.event_hash == persisted["event_hash"]
    assert log.verify_chain()
