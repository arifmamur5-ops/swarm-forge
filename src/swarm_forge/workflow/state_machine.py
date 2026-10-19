from __future__ import annotations

from swarm_forge.contracts.task import TaskStatus


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid task status transition is attempted."""

    def __init__(self, current_status: TaskStatus, target_status: TaskStatus) -> None:
        super().__init__(
            f"Invalid status transition from '{current_status}' to '{target_status}'"
        )
        self.current_status = current_status
        self.target_status = target_status


# Definisi aturan transisi status yang valid (Allowed Transitions)
VALID_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset(
        {
            TaskStatus.VALIDATED,
            TaskStatus.REJECTED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.VALIDATED: frozenset(
        {
            TaskStatus.READY,
            TaskStatus.BLOCKED,
            TaskStatus.REJECTED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.READY: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.TESTING,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.TESTING: frozenset(
        {
            TaskStatus.REVIEW,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
        }
    ),
    TaskStatus.REVIEW: frozenset(
        {
            TaskStatus.ACCEPTED,
            TaskStatus.READY,
            TaskStatus.REJECTED,
        }
    ),
    TaskStatus.BLOCKED: frozenset(
        {
            TaskStatus.READY,
            TaskStatus.CANCELLED,
        }
    ),
    # Status terminal (kecuali FAILED yang bisa di-retry kembali ke READY):
    TaskStatus.ACCEPTED: frozenset(),
    TaskStatus.FAILED: frozenset(
        {
            TaskStatus.READY,
        }
    ),
    TaskStatus.REJECTED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


def is_terminal_status(status: TaskStatus) -> bool:
    """Check if a task status is terminal (no further transitions allowed)."""
    return len(VALID_TRANSITIONS.get(status, frozenset())) == 0


def validate_transition(current_status: TaskStatus, target_status: TaskStatus) -> None:
    """Validate whether a task status transition is allowed.

    Same-status transitions are rejected. Invalid transitions raise
    InvalidStateTransitionError.
    """
    allowed_targets = VALID_TRANSITIONS.get(current_status, frozenset())
    if target_status not in allowed_targets:
        raise InvalidStateTransitionError(current_status, target_status)
