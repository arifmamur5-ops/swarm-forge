from __future__ import annotations

from datetime import datetime

from swarm_forge.contracts.claim import AttemptStatus, ClaimResult
from swarm_forge.contracts.task import TaskSpec, TaskStatus
from swarm_forge.policy.attempt_policy import ATTEMPT_POLICY
from swarm_forge.storage.task_store import TaskNotFoundError, TaskStore


class WorkflowValidationError(ValueError):
    """Raised when a task cannot be safely validated for workflow execution."""


class WorkflowEngine:
    """Deterministic workflow orchestration backed by TaskStore."""

    def __init__(self, store: TaskStore) -> None:
        self.store = store

    def register_task(self, task: TaskSpec) -> TaskSpec:
        """Register a new task. Registration never executes a task."""
        if task.status != TaskStatus.CREATED:
            raise WorkflowValidationError("New tasks must start with CREATED status.")

        self.store.save_task(task)
        return self.store.get_task(task.task_id)

    def validate_task(self, task_id: str) -> TaskSpec:
        """Validate dependency references before moving CREATED to VALIDATED."""
        task = self.store.get_task(task_id)

        if task.status != TaskStatus.CREATED:
            raise WorkflowValidationError(
                f"Task '{task_id}' must be CREATED before validation; "
                f"current status is '{task.status.value}'."
            )
        if not ATTEMPT_POLICY.accepts_new_budget(task.budget):
            raise WorkflowValidationError(
                "Task budget exceeds the global max_attempts policy."
            )

        for dependency_id in task.depends_on:
            if dependency_id == task.task_id:
                raise WorkflowValidationError(
                    f"Task '{task_id}' cannot depend on itself."
                )

            dependency = self._get_dependency(
                dependency_id, expected_run_id=task.run_id
            )
            if dependency.run_id != task.run_id:
                raise WorkflowValidationError(
                    f"Dependency '{dependency_id}' belongs to a different run."
                )

        self._assert_no_dependency_cycle(task)
        return self.store.update_status(task_id, TaskStatus.VALIDATED)

    def refresh_ready_tasks(self, run_id: str | None = None) -> list[TaskSpec]:
        """Move validated tasks to READY when all dependencies are ACCEPTED."""
        changed: list[TaskSpec] = []

        tasks = self.store.list_tasks(run_id=run_id)
        tasks.sort(key=lambda task: (task.created_at, task.task_id))

        for task in tasks:
            if task.status != TaskStatus.VALIDATED:
                continue
            if not ATTEMPT_POLICY.accepts_new_budget(task.budget):
                continue

            dependencies = [
                self._get_dependency(dependency_id, expected_run_id=task.run_id)
                for dependency_id in task.depends_on
            ]

            if all(
                dependency.status == TaskStatus.ACCEPTED for dependency in dependencies
            ):
                changed.append(self.store.update_status(task.task_id, TaskStatus.READY))

        return changed

    def get_ready_tasks(self, run_id: str | None = None) -> list[TaskSpec]:
        """Return READY tasks in stable creation-time and ID order."""
        tasks = [
            task
            for task in self.store.list_tasks(run_id=run_id)
            if task.status == TaskStatus.READY
        ]
        return sorted(tasks, key=lambda task: (task.created_at, task.task_id))

    def claim_next_task(
        self,
        *,
        worker_id: str,
        claim_request_id: str,
        run_id: str | None = None,
    ) -> ClaimResult:
        """Atomically claim the next eligible task without executing it."""
        return self.store.claim_next_task(
            run_id=run_id,
            worker_id=worker_id,
            claim_request_id=claim_request_id,
        )

    def transition_attempt_status(
        self,
        *,
        task_id: str,
        attempt_id: str,
        attempt_no: int,
        expected_attempt_status: AttemptStatus,
        expected_task_status: TaskStatus,
        new_task_status: TaskStatus,
    ) -> TaskSpec:
        """Apply a workflow transition using the claim's attempt fence."""
        return self.store.transition_attempt_status(
            task_id=task_id,
            attempt_id=attempt_id,
            attempt_no=attempt_no,
            expected_attempt_status=expected_attempt_status,
            expected_task_status=expected_task_status,
            new_task_status=new_task_status,
        )

    def heartbeat_attempt(
        self,
        *,
        task_id: str,
        attempt_id: str,
        attempt_no: int,
        worker_id: str,
    ) -> datetime:
        """Renew the lease for the matching active worker attempt."""
        return self.store.heartbeat_attempt(
            task_id=task_id,
            attempt_id=attempt_id,
            attempt_no=attempt_no,
            worker_id=worker_id,
        )

    def recover(self, run_id: str | None = None) -> list[TaskSpec]:
        """Recover expired attempts and return a snapshot of all READY tasks.

        The returned list is the current READY queue after recovery, not only
        the tasks transitioned by this invocation.
        """
        self.store.recover_expired_attempts(run_id=run_id)
        self.refresh_ready_tasks(run_id=run_id)
        return self.get_ready_tasks(run_id=run_id)

    def _get_dependency(self, dependency_id: str, expected_run_id: str) -> TaskSpec:
        try:
            dependency = self.store.get_task(dependency_id)
        except TaskNotFoundError as exc:
            raise WorkflowValidationError(
                f"Dependency '{dependency_id}' does not exist."
            ) from exc

        if dependency.run_id != expected_run_id:
            raise WorkflowValidationError(
                f"Dependency '{dependency_id}' belongs to a different run."
            )

        return dependency

    def _assert_no_dependency_cycle(self, task: TaskSpec) -> None:
        """Reject dependency graphs that lead back to the task being validated."""
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id == task.task_id:
                raise WorkflowValidationError(
                    f"Dependency cycle detected for task '{task.task_id}'."
                )

            if task_id in visited:
                return

            visited.add(task_id)
            node = self._get_dependency(task_id, expected_run_id=task.run_id)

            for dependency_id in node.depends_on:
                visit(dependency_id)

        for dependency_id in task.depends_on:
            visit(dependency_id)
