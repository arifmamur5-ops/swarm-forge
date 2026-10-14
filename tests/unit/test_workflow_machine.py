import pytest

from swarm_forge.contracts.task import TaskStatus
from swarm_forge.workflow.state_machine import (
    VALID_TRANSITIONS,
    InvalidStateTransitionError,
    is_terminal_status,
    validate_transition,
)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TaskStatus.CREATED, TaskStatus.VALIDATED),
        (TaskStatus.CREATED, TaskStatus.REJECTED),
        (TaskStatus.VALIDATED, TaskStatus.READY),
        (TaskStatus.READY, TaskStatus.RUNNING),
        (TaskStatus.RUNNING, TaskStatus.TESTING),
        (TaskStatus.TESTING, TaskStatus.REVIEW),
        (TaskStatus.REVIEW, TaskStatus.ACCEPTED),
        (TaskStatus.BLOCKED, TaskStatus.READY),
    ],
)
def test_valid_transitions(current: TaskStatus, target: TaskStatus) -> None:
    # Should not raise exception
    validate_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TaskStatus.CREATED, TaskStatus.RUNNING),
        (TaskStatus.READY, TaskStatus.ACCEPTED),
        (TaskStatus.ACCEPTED, TaskStatus.RUNNING),
        (TaskStatus.FAILED, TaskStatus.READY),
        (TaskStatus.CANCELLED, TaskStatus.VALIDATED),
    ],
)
def test_invalid_transitions(current: TaskStatus, target: TaskStatus) -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(current, target)


def test_transition_to_same_status_raises() -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(TaskStatus.CREATED, TaskStatus.CREATED)


def test_terminal_statuses() -> None:
    assert is_terminal_status(TaskStatus.ACCEPTED) is True
    assert is_terminal_status(TaskStatus.FAILED) is True
    assert is_terminal_status(TaskStatus.REJECTED) is True
    assert is_terminal_status(TaskStatus.CANCELLED) is True

    assert is_terminal_status(TaskStatus.CREATED) is False
    assert is_terminal_status(TaskStatus.READY) is False
    assert is_terminal_status(TaskStatus.RUNNING) is False


def test_all_statuses_covered_in_transitions() -> None:
    for status in TaskStatus:
        assert status in VALID_TRANSITIONS
