from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from swarm_forge.contracts import (
    AgentRole,
    ArtifactKind,
    ArtifactRecord,
    AttemptStatus,
    AuditEvent,
    EventType,
    LifecycleEvent,
    OperationContext,
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

def test_task_created_at_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_task(created_at=datetime(2026, 1, 1))

def test_task_created_at_accepts_timezone_aware_datetime() -> None:
    created_at = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=7)))

    task = make_task(created_at=created_at)

    assert task.created_at == created_at
    assert task.created_at.utcoffset() == timedelta(hours=7)

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


def make_operation_context(**overrides: str) -> OperationContext:
    data = {
        "operation_id": "operation-001",
        "correlation_id": "correlation-001",
        "source": "workflow_engine",
    }
    data.update(overrides)
    return OperationContext.model_validate(data)


def test_operation_context_requires_explicit_nonblank_ids() -> None:
    with pytest.raises(ValidationError):
        OperationContext(
            operation_id="operation-001",
            correlation_id=" ",
            source="workflow_engine",
        )


def test_operation_context_is_immutable() -> None:
    context = make_operation_context()

    with pytest.raises(ValidationError):
        context.correlation_id = "other-correlation"  # type: ignore[misc]


def test_lifecycle_event_replay_preserves_event_identity() -> None:
    context = make_operation_context()
    first = make_created_lifecycle_event(context=context)
    rebuilt = make_created_lifecycle_event(context=context)
    replay = LifecycleEvent.model_validate_json(first.model_dump_json())

    assert first.event_id == rebuilt.event_id == replay.event_id
    assert first.event_id != first.idempotency_key
    assert first.idempotency_key == replay.idempotency_key
    assert first.operation_id == context.operation_id
    assert first.correlation_id == context.correlation_id


def make_created_lifecycle_event(
    *, context: OperationContext, discriminator: str = "task-created"
) -> LifecycleEvent:
    return LifecycleEvent.for_operation(
        context=context,
        idempotency_key="task-created:task-001",
        event_discriminator=discriminator,
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_CREATED,
        new_state=TaskStatus.CREATED,
    )


def make_validated_lifecycle_event() -> LifecycleEvent:
    return LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key="task-validated:task-001",
        event_discriminator="task-validated",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_STATUS_CHANGED,
        previous_state=TaskStatus.CREATED,
        new_state=TaskStatus.VALIDATED,
        transition_cause="validation",
        payload={"details": {"labels": ["original"]}},
    )


def test_lifecycle_event_copy_without_updates_preserves_event() -> None:
    event = make_validated_lifecycle_event()

    copied = event.model_copy()
    deep_copied = event.model_copy(deep=True)

    assert copied == event
    assert copied.model_dump(mode="json") == event.model_dump(mode="json")
    assert copied.payload is not event.payload
    assert deep_copied.model_dump(mode="json") == event.model_dump(mode="json")


def test_lifecycle_event_copy_valid_payload_update_is_deeply_immutable() -> None:
    event = make_validated_lifecycle_event()
    update_payload = {"nested": [{"value": "updated"}]}

    copied = event.model_copy(update={"payload": update_payload})
    update_payload["nested"][0]["value"] = "changed outside"

    assert copied.payload["nested"][0]["value"] == "updated"
    with pytest.raises((AttributeError, TypeError)):
        copied.payload["nested"].append({"value": "mutated"})
    with pytest.raises(TypeError):
        copied.payload["nested"][0]["value"] = "mutated"
    assert event.payload["details"]["labels"][0] == "original"


def test_lifecycle_event_copy_rejects_invalid_update() -> None:
    event = make_validated_lifecycle_event()

    with pytest.raises(ValidationError):
        event.model_copy(update={"new_state": TaskStatus.ACCEPTED})


def test_lifecycle_event_copy_cannot_change_event_id_independently() -> None:
    event = make_validated_lifecycle_event()

    with pytest.raises(ValidationError, match="event_id must match"):
        event.model_copy(update={"event_id": "unrelated-event-id"})


def test_lifecycle_event_copy_does_not_change_original() -> None:
    event = make_validated_lifecycle_event()
    original = event.model_dump(mode="json")

    copied = event.model_copy(update={"payload": {"new": [1, 2]}})

    assert event.model_dump(mode="json") == original
    assert copied.payload == {"new": (1, 2)}


def test_distinct_events_in_one_operation_have_distinct_ids() -> None:
    context = make_operation_context()

    created = make_created_lifecycle_event(context=context)
    status_changed = LifecycleEvent.for_operation(
        context=context,
        idempotency_key="task-validated:task-001",
        event_discriminator="task-validated",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_STATUS_CHANGED,
        previous_state=TaskStatus.CREATED,
        new_state=TaskStatus.VALIDATED,
        transition_cause="validation",
    )

    assert created.operation_id == status_changed.operation_id
    assert created.event_id != status_changed.event_id


def test_lifecycle_payload_nested_dictionary_is_immutable() -> None:
    event = make_created_lifecycle_event(context=make_operation_context())
    nested = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key="payload-dict:task-001",
        event_discriminator="payload-dict",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_CREATED,
        new_state=TaskStatus.CREATED,
        payload={"details": {"state": "created"}},
    )

    assert event.payload == {}
    with pytest.raises(TypeError):
        nested.payload["details"]["state"] = "changed"


def test_lifecycle_payload_nested_list_is_immutable() -> None:
    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key="payload-list:task-001",
        event_discriminator="payload-list",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_CREATED,
        new_state=TaskStatus.CREATED,
        payload={"items": [{"value": 1}]},
    )

    with pytest.raises((AttributeError, TypeError)):
        event.payload["items"].append({"value": 2})
    with pytest.raises(TypeError):
        event.payload["items"][0]["value"] = 2


def test_lifecycle_event_serialization_round_trip_preserves_all_fields() -> None:
    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key="claim:claim-request-001",
        event_discriminator="attempt-start",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_ATTEMPT_STARTED,
        attempt_id="attempt-001",
        attempt_no=1,
        worker_id="worker-001",
        claim_request_id="claim-request-001",
        previous_state=TaskStatus.READY,
        new_state=TaskStatus.RUNNING,
        payload={"labels": ["one", "two"], "details": {"ok": True}},
    )

    restored = LifecycleEvent.model_validate_json(event.model_dump_json())

    assert restored.model_dump(mode="json") == event.model_dump(mode="json")


def test_lifecycle_event_model_validate_paths_refreeze_payload() -> None:
    event = make_validated_lifecycle_event()
    dumped = event.model_dump()

    from_mapping = LifecycleEvent.model_validate(dumped)
    from_instance = LifecycleEvent.model_validate(event)
    dumped["payload"]["details"]["labels"].append("outside")

    assert from_mapping.model_dump(mode="json") == event.model_dump(mode="json")
    assert from_instance.model_dump(mode="json") == event.model_dump(mode="json")
    with pytest.raises((AttributeError, TypeError)):
        from_mapping.payload["details"]["labels"].append("inside")


def test_lifecycle_event_model_construct_requires_trust_boundary_revalidation() -> None:
    event = make_validated_lifecycle_event()
    constructed = LifecycleEvent.model_construct(
        **{
            **event.model_dump(),
            "event_id": "invalid-event-id",
            "payload": {"nested": []},
        }
    )

    with pytest.raises(ValidationError, match="event_id must match"):
        LifecycleEvent.model_validate(constructed)


def test_lifecycle_event_attempt_start_has_ready_to_running_transition() -> None:
    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key="claim:claim-request-001",
        event_discriminator="attempt-start",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_ATTEMPT_STARTED,
        attempt_id="attempt-001",
        attempt_no=1,
        worker_id="worker-001",
        claim_request_id="claim-request-001",
        previous_state=TaskStatus.READY,
        new_state=TaskStatus.RUNNING,
    )

    assert event.previous_state == TaskStatus.READY
    assert event.new_state == TaskStatus.RUNNING


def test_lifecycle_event_rejects_invalid_attempt_start_transition() -> None:
    with pytest.raises(ValidationError, match="READY to RUNNING"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="claim:claim-request-002",
            event_discriminator="attempt-start",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_ATTEMPT_STARTED,
            attempt_id="attempt-002",
            attempt_no=1,
            worker_id="worker-001",
            claim_request_id="claim-request-002",
            previous_state=TaskStatus.CREATED,
            new_state=TaskStatus.RUNNING,
        )


def test_lifecycle_status_change_requires_explicit_cause() -> None:
    with pytest.raises(ValidationError, match="transition_cause"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="retry:operation-003",
            event_discriminator="retry-status-change",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_STATUS_CHANGED,
            previous_state=TaskStatus.FAILED,
            new_state=TaskStatus.READY,
        )


def test_failed_to_ready_explicit_retry_does_not_require_attempt_identity() -> None:
    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key="retry:task-001",
        event_discriminator="retry-failed-to-ready",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_STATUS_CHANGED,
        previous_state=TaskStatus.FAILED,
        new_state=TaskStatus.READY,
        transition_cause="retry_task",
    )

    assert event.attempt_id is None
    assert event.attempt_no is None


def test_failed_to_ready_recovery_requires_attempt_identity() -> None:
    with pytest.raises(ValidationError, match="Recovery FAILED to READY"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="recovery:task-001",
            event_discriminator="recovery-failed-to-ready",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_STATUS_CHANGED,
            previous_state=TaskStatus.FAILED,
            new_state=TaskStatus.READY,
            transition_cause="recover_expired_attempts",
        )

    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key="recovery:task-001",
        event_discriminator="recovery-failed-to-ready",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_STATUS_CHANGED,
        previous_state=TaskStatus.FAILED,
        new_state=TaskStatus.READY,
        transition_cause="recover_expired_attempts",
        attempt_id="attempt-001",
        attempt_no=1,
    )
    assert event.attempt_id == "attempt-001"
    assert event.attempt_no == 1


@pytest.mark.parametrize("cause", ["recovery", " recover_expired_attempts", "unknown"])
def test_failed_to_ready_rejects_unrecognized_causes(cause: str) -> None:
    with pytest.raises(ValidationError, match="requires transition_cause"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key=f"unknown-cause:{cause}",
            event_discriminator=f"unknown-cause-{cause}",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_STATUS_CHANGED,
            previous_state=TaskStatus.FAILED,
            new_state=TaskStatus.READY,
            transition_cause=cause,
        )


def test_lifecycle_attempt_end_requires_terminal_outcome() -> None:
    with pytest.raises(ValidationError, match="terminal outcome"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="attempt-end:attempt-001",
            event_discriminator="attempt-end",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_ATTEMPT_ENDED,
            attempt_id="attempt-001",
            attempt_no=1,
            attempt_outcome=AttemptStatus.RUNNING,
        )


def test_lifecycle_status_change_rejects_impossible_transition() -> None:
    with pytest.raises(ValidationError, match="Invalid status transition"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="impossible-transition:task-001",
            event_discriminator="impossible-transition",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_STATUS_CHANGED,
            previous_state=TaskStatus.CREATED,
            new_state=TaskStatus.ACCEPTED,
            transition_cause="workflow",
        )


def test_lifecycle_status_change_rejects_claim_transition() -> None:
    with pytest.raises(ValidationError, match="must use TASK_ATTEMPT_STARTED"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="claim-as-status-change:task-001",
            event_discriminator="claim-as-status-change",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_STATUS_CHANGED,
            previous_state=TaskStatus.READY,
            new_state=TaskStatus.RUNNING,
            transition_cause="claim",
        )


@pytest.mark.parametrize(
    ("previous", "new"),
    [
        (TaskStatus.RUNNING, TaskStatus.TESTING),
        (TaskStatus.TESTING, TaskStatus.REVIEW),
        (TaskStatus.REVIEW, TaskStatus.ACCEPTED),
    ],
)
def test_active_attempt_transition_requires_attempt_identity(
    previous: TaskStatus, new: TaskStatus
) -> None:
    with pytest.raises(ValidationError, match="require attempt identity"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key=f"active:{previous.value}:{new.value}",
            event_discriminator=f"active-{previous.value}-{new.value}",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_STATUS_CHANGED,
            previous_state=previous,
            new_state=new,
            transition_cause="attempt",
        )

    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key=f"active:{previous.value}:{new.value}",
        event_discriminator=f"active-{previous.value}-{new.value}",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_STATUS_CHANGED,
        previous_state=previous,
        new_state=new,
        transition_cause="attempt",
        attempt_id="attempt-001",
        attempt_no=1,
    )
    assert event.attempt_id == "attempt-001"


@pytest.mark.parametrize(
    ("previous", "new"),
    [
        (TaskStatus.CREATED, TaskStatus.VALIDATED),
        (TaskStatus.FAILED, TaskStatus.READY),
        (TaskStatus.CREATED, TaskStatus.CANCELLED),
    ],
)
def test_administrative_transition_does_not_require_attempt_identity(
    previous: TaskStatus, new: TaskStatus
) -> None:
    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key=f"admin:{previous.value}:{new.value}",
        event_discriminator=f"admin-{previous.value}-{new.value}",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_STATUS_CHANGED,
        previous_state=previous,
        new_state=new,
        transition_cause=(
            "retry_task"
            if previous == TaskStatus.FAILED and new == TaskStatus.READY
            else "administrative"
        ),
    )

    assert event.attempt_id is None


def test_lifecycle_event_rejects_contradictory_shapes() -> None:
    with pytest.raises(ValidationError, match="attempt fields"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="created-with-attempt:task-001",
            event_discriminator="created-with-attempt",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_CREATED,
            new_state=TaskStatus.CREATED,
            attempt_id="attempt-001",
            attempt_no=1,
        )

    with pytest.raises(ValidationError, match="only the attempt outcome"):
        LifecycleEvent.for_operation(
            context=make_operation_context(),
            idempotency_key="attempt-end-with-state:attempt-001",
            event_discriminator="attempt-end-with-state",
            run_id="run-001",
            task_id="task-001",
            event_type=EventType.TASK_ATTEMPT_ENDED,
            attempt_id="attempt-001",
            attempt_no=1,
            attempt_outcome=AttemptStatus.SUCCEEDED,
            new_state=TaskStatus.ACCEPTED,
        )


@pytest.mark.parametrize(
    "outcome",
    [AttemptStatus.SUCCEEDED, AttemptStatus.FAILED, AttemptStatus.ABANDONED],
)
def test_lifecycle_attempt_end_accepts_terminal_attempt_statuses(
    outcome: AttemptStatus,
) -> None:
    event = LifecycleEvent.for_operation(
        context=make_operation_context(),
        idempotency_key=f"attempt-end:{outcome.value}",
        event_discriminator=f"attempt-end-{outcome.value}",
        run_id="run-001",
        task_id="task-001",
        event_type=EventType.TASK_ATTEMPT_ENDED,
        attempt_id="attempt-001",
        attempt_no=1,
        attempt_outcome=outcome,
    )

    assert event.attempt_outcome == outcome

def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_task(status="magic_success")
