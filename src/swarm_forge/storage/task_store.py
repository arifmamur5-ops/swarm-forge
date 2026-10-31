from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from swarm_forge.contracts.claim import (
    AttemptStatus,
    ClaimResult,
    ClaimSkipReason,
    SkippedClaimCandidate,
    TaskClaim,
)
from swarm_forge.contracts.task import AgentRole, TaskBudget, TaskSpec, TaskStatus
from swarm_forge.policy.attempt_policy import ATTEMPT_POLICY
from swarm_forge.workflow.state_machine import validate_transition


def _utc_now() -> datetime:
    """Internal clock seam; callers cannot choose lease timestamps."""
    return datetime.now(UTC)


class DuplicateTaskError(ValueError):
    """Raised when attempting to create a task with an existing task_id."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task with ID '{task_id}' already exists.")
        self.task_id = task_id


class TaskNotFoundError(KeyError):
    """Raised when a requested task cannot be found in the store."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task with ID '{task_id}' not found.")
        self.task_id = task_id


class AttemptLimitExceededError(ValueError):
    """Raised when a task has exhausted its allowed execution attempts."""

    def __init__(self, task_id: str, max_attempts: int) -> None:
        super().__init__(
            f"Task '{task_id}' has exhausted its maximum of {max_attempts} attempts."
        )
        self.task_id = task_id
        self.max_attempts = max_attempts


class TaskClaimRequiredError(ValueError):
    """Raised when a caller tries to enter RUNNING outside the claim operation."""


class TaskAttemptRequiredError(ValueError):
    """Raised when an active task transition lacks attempt identity."""


class AttemptFencingError(ValueError):
    """Raised when an attempt/task identity or expected state does not match."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"Attempt is not active for task '{task_id}'.")
        self.task_id = task_id


class ClaimRequestConflictError(ValueError):
    """Raised when a globally unique request ID is reused by another caller."""

    def __init__(self, claim_request_id: str) -> None:
        super().__init__(f"Claim request ID '{claim_request_id}' is already in use.")
        self.claim_request_id = claim_request_id


class TaskStore:
    """SQLite-backed persistent repository for TaskSpec objects."""

    SCHEMA_VERSION = 3

    def __init__(self, db_path: str | Path = "swarm_forge.db") -> None:
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self._persistent_conn: sqlite3.Connection | None = None

        # Handle in-memory database explicitly by keeping a single persistent connection open
        if str(self.db_path) == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA foreign_keys = ON")

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        if (
            str(self.db_path) != ":memory:"
            and self.db_path != Path(":memory:")
            and self.db_path.parent != Path(".")
        ):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        try:
            # Version 0 is the existing tasks-only schema. Keep the task table
            # intact and install attempt history as one atomic migration.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
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

            current_version = int(
                conn.execute("PRAGMA user_version").fetchone()[0]
            )
            if current_version > self.SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {current_version} is newer than "
                    f"supported version {self.SCHEMA_VERSION}."
                )

            if current_version < 1:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_attempts (
                        attempt_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL REFERENCES tasks(task_id),
                        attempt_no INTEGER NOT NULL CHECK (attempt_no > 0),
                        worker_id TEXT,
                        status TEXT NOT NULL CHECK (
                            status IN ('running', 'succeeded', 'failed', 'abandoned')
                        ),
                        started_at TEXT NOT NULL,
                        heartbeat_at TEXT,
                        lease_expires_at TEXT,
                        finished_at TEXT,
                        reason_code TEXT CHECK (
                            reason_code IS NULL OR (
                                length(reason_code) BETWEEN 1 AND 100
                                AND reason_code NOT GLOB '*[^a-z0-9_.-]*'
                            )
                        ),
                        CHECK (
                            status != 'running' OR (
                                lease_expires_at IS NOT NULL
                                AND finished_at IS NULL
                            )
                        ),
                        UNIQUE (task_id, attempt_no)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        idx_task_attempts_one_running_per_task
                    ON task_attempts(task_id)
                    WHERE status = 'running'
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_task_attempts_expiry
                    ON task_attempts(lease_expires_at)
                    WHERE status = 'running'
                    """
                )
                # Legacy RUNNING tasks intentionally remain untouched. They
                # have no trustworthy lease or attempt identity, so recovery
                # must not treat them as active or make them READY.
                conn.execute("PRAGMA user_version = 1")
                current_version = 1

            if current_version < 2:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS claim_requests (
                        claim_request_id TEXT PRIMARY KEY,
                        worker_id TEXT NOT NULL,
                        run_id TEXT,
                        response_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute("PRAGMA user_version = 2")
                current_version = 2

            if current_version < 3:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lifecycle_outbox_operations (
                        operation_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        operation_fingerprint TEXT NOT NULL,
                        event_manifest TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (operation_id, idempotency_key)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lifecycle_outbox (
                        sequence_no INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        operation_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        event_discriminator TEXT NOT NULL,
                        schema_version INTEGER NOT NULL CHECK (schema_version > 0),
                        run_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        attempt_id TEXT,
                        correlation_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        event_json TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('pending', 'delivered', 'poison')
                        ),
                        retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
                        next_attempt_at TEXT NOT NULL,
                        lease_token TEXT,
                        lease_expires_at TEXT,
                        delivered_at TEXT,
                        last_error TEXT,
                        last_error_at TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE (operation_id, event_discriminator),
                        FOREIGN KEY (operation_id, idempotency_key)
                            REFERENCES lifecycle_outbox_operations(
                                operation_id, idempotency_key
                            ),
                        CHECK (
                            (lease_token IS NULL AND lease_expires_at IS NULL)
                            OR (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
                        ),
                        CHECK (
                            (status = 'delivered' AND delivered_at IS NOT NULL)
                            OR (status != 'delivered' AND delivered_at IS NULL)
                        )
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_lifecycle_outbox_pending
                    ON lifecycle_outbox(status, sequence_no)
                    WHERE status = 'pending'
                    """
                )
                conn.execute("PRAGMA user_version = 3")

            conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def close(self) -> None:
        """Explicitly close persistent connection if active (e.g. for :memory: databases)."""
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None

    def save_task(self, task: TaskSpec) -> None:
        """Persist a new TaskSpec into the database, relying on primary key constraints for atomicity."""
        if task.status == TaskStatus.RUNNING:
            raise TaskClaimRequiredError(
                "Tasks must enter RUNNING through an atomic claim."
            )
        ATTEMPT_POLICY.validate_new_budget(task.budget)

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, run_id, parent_task_id, agent_role, goal,
                    status, attempt, created_at, budget_json,
                    depends_on_json, input_refs_json, output_contract, allowed_tools_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.run_id,
                    task.parent_task_id,
                    task.agent_role.value,
                    task.goal,
                    task.status.value,
                    task.attempt,
                    task.created_at.isoformat(),
                    task.budget.model_dump_json(),
                    json.dumps(list(task.depends_on)),
                    json.dumps(list(task.input_refs)),
                    task.output_contract,
                    json.dumps(list(task.allowed_tools)),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise DuplicateTaskError(task.task_id) from e
        finally:
            if self._persistent_conn is None:
                conn.close()

    def get_task(self, task_id: str) -> TaskSpec:
        """Retrieve a TaskSpec by its task_id."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                raise TaskNotFoundError(task_id)
            return self._row_to_task_spec(row)
        finally:
            if self._persistent_conn is None:
                conn.close()

    def list_tasks(self, run_id: str | None = None) -> list[TaskSpec]:
        """List all tasks, optionally filtered by run_id."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if run_id is not None:
                cursor.execute("SELECT * FROM tasks WHERE run_id = ?", (run_id,))
            else:
                cursor.execute("SELECT * FROM tasks")
            rows = cursor.fetchall()
            return [self._row_to_task_spec(row) for row in rows]
        finally:
            if self._persistent_conn is None:
                conn.close()

    @staticmethod
    def _ensure_attempt_budget_remaining(
        task_id: str, attempt: int, max_attempts: int
    ) -> None:
        if (
            max_attempts > ATTEMPT_POLICY.max_attempts
            or attempt >= max_attempts
        ):
            raise AttemptLimitExceededError(
                task_id, min(max_attempts, ATTEMPT_POLICY.max_attempts)
            )

    def update_status(self, task_id: str, new_status: TaskStatus) -> TaskSpec:
        """Atomically update a task's status with workflow state machine validation.

        Uses BEGIN IMMEDIATE to acquire a write lock immediately and prevent race conditions.
        """
        if new_status == TaskStatus.RUNNING:
            raise TaskClaimRequiredError(
                "Tasks must enter RUNNING through an atomic claim."
            )

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")

            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                conn.rollback()
                raise TaskNotFoundError(task_id)

            current_status = TaskStatus(row["status"])

            active_attempt = conn.execute(
                """
                SELECT 1 FROM task_attempts
                WHERE task_id = ? AND status = 'running' LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if current_status in {
                TaskStatus.RUNNING,
                TaskStatus.TESTING,
                TaskStatus.REVIEW,
            } or active_attempt is not None:
                raise TaskAttemptRequiredError(
                    "Active task status changes require attempt identity."
                )

            # Validate via workflow state machine
            validate_transition(current_status, new_status)
            if new_status == TaskStatus.READY:
                budget = TaskBudget.model_validate_json(row["budget_json"])
                self._ensure_attempt_budget_remaining(
                    task_id, row["attempt"], budget.max_attempts
                )

            cursor.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                (new_status.value, task_id),
            )
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            updated_row = cursor.fetchone()
            if not updated_row:
                raise TaskNotFoundError(task_id)
            result = self._row_to_task_spec(updated_row)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    @staticmethod
    def _terminal_attempt_status(new_task_status: TaskStatus) -> AttemptStatus | None:
        if new_task_status == TaskStatus.ACCEPTED:
            return AttemptStatus.SUCCEEDED
        if new_task_status in {TaskStatus.FAILED, TaskStatus.REJECTED}:
            return AttemptStatus.FAILED
        if new_task_status in {TaskStatus.BLOCKED, TaskStatus.CANCELLED}:
            return AttemptStatus.ABANDONED
        if new_task_status == TaskStatus.READY:
            return AttemptStatus.FAILED
        return None

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
        """Transition a task only when its current attempt identity and states match."""
        if not task_id.strip() or not attempt_id.strip():
            raise ValueError("task_id and attempt_id must not be blank")
        if attempt_no < 1:
            raise ValueError("attempt_no must be at least 1")
        if expected_attempt_status != AttemptStatus.RUNNING:
            raise AttemptFencingError(task_id)

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            task_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task_row is None:
                raise TaskNotFoundError(task_id)

            attempt_row = conn.execute(
                """
                SELECT status, lease_expires_at FROM task_attempts
                WHERE attempt_id = ? AND task_id = ? AND attempt_no = ?
                """,
                (attempt_id, task_id, attempt_no),
            ).fetchone()
            if attempt_row is None:
                raise AttemptFencingError(task_id)

            current_task = self._row_to_task_spec(task_row)
            if (
                AttemptStatus(attempt_row["status"]) != expected_attempt_status
                or current_task.status != expected_task_status
                or current_task.attempt != attempt_no
            ):
                raise AttemptFencingError(task_id)

            if new_task_status == TaskStatus.READY:
                self._ensure_attempt_budget_remaining(
                    task_id,
                    current_task.attempt,
                    current_task.budget.max_attempts,
                )

            try:
                lease_expires_at = datetime.fromisoformat(
                    attempt_row["lease_expires_at"]
                )
            except (TypeError, ValueError) as exc:
                raise AttemptFencingError(task_id) from exc
            if (
                lease_expires_at.tzinfo is None
                or lease_expires_at.astimezone(UTC) <= datetime.now(UTC)
            ):
                raise AttemptFencingError(task_id)

            validate_transition(expected_task_status, new_task_status)
            terminal_attempt_status = self._terminal_attempt_status(new_task_status)
            transition_time = datetime.now(UTC).isoformat()

            task_update = conn.execute(
                """
                UPDATE tasks SET status = ?
                WHERE task_id = ? AND status = ? AND attempt = ?
                  AND EXISTS (
                      SELECT 1 FROM task_attempts
                      WHERE attempt_id = ? AND task_id = ? AND attempt_no = ?
                        AND status = ? AND lease_expires_at > ?
                  )
                """,
                (
                    new_task_status.value,
                    task_id,
                    expected_task_status.value,
                    attempt_no,
                    attempt_id,
                    task_id,
                    attempt_no,
                    expected_attempt_status.value,
                    transition_time,
                ),
            )
            if task_update.rowcount != 1:
                raise AttemptFencingError(task_id)

            if terminal_attempt_status is not None:
                attempt_update = conn.execute(
                    """
                    UPDATE task_attempts
                    SET status = ?, finished_at = ?
                    WHERE attempt_id = ? AND task_id = ? AND attempt_no = ?
                      AND status = ? AND lease_expires_at > ?
                    """,
                    (
                        terminal_attempt_status.value,
                        transition_time,
                        attempt_id,
                        task_id,
                        attempt_no,
                        expected_attempt_status.value,
                        transition_time,
                    ),
                )
                if attempt_update.rowcount != 1:
                    raise AttemptFencingError(task_id)

            updated_task = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if updated_task is None:
                raise TaskNotFoundError(task_id)
            result = self._row_to_task_spec(updated_task)
            conn.commit()
            return result
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def heartbeat_attempt(
        self,
        *,
        task_id: str,
        attempt_id: str,
        attempt_no: int,
        worker_id: str,
    ) -> datetime:
        """Extend an active attempt lease, returning its new expiry time."""
        if not task_id.strip() or not attempt_id.strip() or not worker_id.strip():
            raise ValueError("task_id, attempt_id, and worker_id must not be blank")
        if attempt_no < 1:
            raise ValueError("attempt_no must be at least 1")

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Take the authoritative time only after obtaining SQLite's write
            # lock; a queued heartbeat must not use a pre-expiry timestamp.
            heartbeat_time = _utc_now()
            heartbeat_at = heartbeat_time.isoformat()
            row = conn.execute(
                """
                SELECT t.status, t.attempt, t.budget_json,
                       a.worker_id, a.status AS attempt_status, a.lease_expires_at
                FROM tasks AS t
                JOIN task_attempts AS a ON a.task_id = t.task_id
                WHERE t.task_id = ? AND a.attempt_id = ? AND a.attempt_no = ?
                """,
                (task_id, attempt_id, attempt_no),
            ).fetchone()
            if (
                row is None
                or row["worker_id"] != worker_id
                or row["attempt_status"] != AttemptStatus.RUNNING.value
                or row["attempt"] != attempt_no
                or row["status"]
                not in {
                    TaskStatus.RUNNING.value,
                    TaskStatus.TESTING.value,
                    TaskStatus.REVIEW.value,
                }
            ):
                raise AttemptFencingError(task_id)
            try:
                old_expiry = datetime.fromisoformat(row["lease_expires_at"])
            except (TypeError, ValueError) as exc:
                raise AttemptFencingError(task_id) from exc
            if old_expiry.tzinfo is None or old_expiry.astimezone(UTC) <= heartbeat_time:
                raise AttemptFencingError(task_id)

            timeout_seconds = TaskBudget.model_validate_json(
                row["budget_json"]
            ).timeout_seconds
            new_expiry = heartbeat_time + timedelta(seconds=timeout_seconds)
            update = conn.execute(
                """
                UPDATE task_attempts
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE attempt_id = ? AND task_id = ? AND attempt_no = ?
                  AND worker_id = ? AND status = 'running'
                  AND lease_expires_at > ?
                """,
                (
                    heartbeat_at,
                    new_expiry.isoformat(),
                    attempt_id,
                    task_id,
                    attempt_no,
                    worker_id,
                    heartbeat_at,
                ),
            )
            if update.rowcount != 1:
                raise AttemptFencingError(task_id)
            conn.commit()
            return new_expiry
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def recover_expired_attempts(
        self,
        *,
        run_id: str | None = None,
    ) -> list[TaskSpec]:
        """Abandon expired active attempts and requeue or fail their tasks atomically.

        Legacy RUNNING rows without a matching attempt are intentionally ignored.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Evaluate expiry only after the write lock is held, so waiting on
            # another writer cannot make this recovery use a stale timestamp.
            recovery_time = _utc_now()
            recovery_at = recovery_time.isoformat()
            query = """
                SELECT t.*, a.attempt_id, a.attempt_no, a.lease_expires_at
                FROM tasks AS t
                JOIN task_attempts AS a ON a.task_id = t.task_id
                WHERE a.status = 'running'
            """
            parameters: tuple[str, ...] = ()
            if run_id is not None:
                query += " AND t.run_id = ?"
                parameters = (run_id,)
            rows = conn.execute(query, parameters).fetchall()
            recovered: list[TaskSpec] = []
            for row in rows:
                if row["attempt"] != row["attempt_no"]:
                    continue
                if row["status"] not in {
                    TaskStatus.RUNNING.value,
                    TaskStatus.TESTING.value,
                    TaskStatus.REVIEW.value,
                }:
                    continue
                try:
                    expiry = datetime.fromisoformat(row["lease_expires_at"])
                except (TypeError, ValueError):
                    continue
                if expiry.tzinfo is None or expiry.astimezone(UTC) > recovery_time:
                    continue

                current_status = TaskStatus(row["status"])
                budget = TaskBudget.model_validate_json(row["budget_json"])
                exhausted = not ATTEMPT_POLICY.has_remaining_attempts(
                    budget, row["attempt"]
                )
                target_status = TaskStatus.FAILED if exhausted else TaskStatus.READY
                if exhausted:
                    validate_transition(current_status, TaskStatus.FAILED)
                elif current_status == TaskStatus.REVIEW:
                    validate_transition(current_status, TaskStatus.READY)
                else:
                    validate_transition(current_status, TaskStatus.FAILED)
                    validate_transition(TaskStatus.FAILED, TaskStatus.READY)

                attempt_update = conn.execute(
                    """
                    UPDATE task_attempts
                    SET status = 'abandoned', finished_at = ?, reason_code = 'lease_expired'
                    WHERE attempt_id = ? AND task_id = ? AND attempt_no = ?
                      AND status = 'running' AND lease_expires_at <= ?
                    """,
                    (
                        recovery_at,
                        row["attempt_id"],
                        row["task_id"],
                        row["attempt_no"],
                        recovery_at,
                    ),
                )
                if attempt_update.rowcount != 1:
                    raise AttemptFencingError(row["task_id"])

                if not exhausted and current_status != TaskStatus.REVIEW:
                    failed_update = conn.execute(
                        """
                        UPDATE tasks SET status = ?
                        WHERE task_id = ? AND status = ? AND attempt = ?
                        """,
                        (
                            TaskStatus.FAILED.value,
                            row["task_id"],
                            current_status.value,
                            row["attempt_no"],
                        ),
                    )
                    if failed_update.rowcount != 1:
                        raise AttemptFencingError(row["task_id"])

                final_update = conn.execute(
                    """
                    UPDATE tasks SET status = ?
                    WHERE task_id = ? AND attempt = ? AND status = ?
                    """,
                    (
                        target_status.value,
                        row["task_id"],
                        row["attempt_no"],
                        (
                            TaskStatus.FAILED.value
                            if not exhausted and current_status != TaskStatus.REVIEW
                            else current_status.value
                        ),
                    ),
                )
                if final_update.rowcount != 1:
                    raise AttemptFencingError(row["task_id"])
                updated = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
                ).fetchone()
                if updated is None:
                    raise TaskNotFoundError(row["task_id"])
                recovered.append(self._row_to_task_spec(updated))
            conn.commit()
            return recovered
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def claim_next_task(
        self,
        *,
        run_id: str | None = None,
        worker_id: str,
        claim_request_id: str,
    ) -> ClaimResult:
        """Atomically select, start, and persist one task claim or an empty result.

        Request IDs are globally unique within this TaskStore database. Replays
        return the stored receipt, including the original empty result.
        """
        if not worker_id.strip():
            raise ValueError("worker_id must not be blank")
        if not claim_request_id.strip():
            raise ValueError("claim_request_id must not be blank")
        if run_id is not None and not run_id.strip():
            raise ValueError("run_id must not be blank")

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior_request = conn.execute(
                """
                SELECT worker_id, run_id, response_json
                FROM claim_requests WHERE claim_request_id = ?
                """,
                (claim_request_id,),
            ).fetchone()
            if prior_request is not None:
                if (
                    prior_request["worker_id"] != worker_id
                    or prior_request["run_id"] != run_id
                ):
                    raise ClaimRequestConflictError(claim_request_id)
                result = ClaimResult.model_validate_json(prior_request["response_json"])
                conn.commit()
                return result

            candidates_query = "SELECT * FROM tasks WHERE status = ?"
            parameters: tuple[str, ...] = (TaskStatus.READY.value,)
            if run_id is not None:
                candidates_query += " AND run_id = ?"
                parameters += (run_id,)
            candidate_rows = conn.execute(candidates_query, parameters).fetchall()

            candidates: list[TaskSpec] = []
            skipped: list[SkippedClaimCandidate] = []
            for candidate_row in candidate_rows:
                try:
                    candidates.append(self._row_to_task_spec(candidate_row))
                except (KeyError, TypeError, ValueError):
                    skipped.append(
                        SkippedClaimCandidate(
                            task_id=str(candidate_row["task_id"]),
                            reason=ClaimSkipReason.TASK_INVALID,
                        )
                    )
            candidates.sort(key=lambda task: (task.created_at, task.task_id))

            claim: TaskClaim | None = None
            for task in candidates:
                reason = self._claim_skip_reason(conn, task)
                if reason is not None:
                    skipped.append(
                        SkippedClaimCandidate(task_id=task.task_id, reason=reason)
                    )
                    continue

                validate_transition(task.status, TaskStatus.RUNNING)
                max_attempts = min(
                    task.budget.max_attempts, ATTEMPT_POLICY.max_attempts
                )
                attempt_no = task.attempt + 1
                attempt_id = str(uuid4())
                started_at = datetime.now(UTC)
                lease_expires_at = started_at + timedelta(
                    seconds=task.budget.timeout_seconds
                )
                update = conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, attempt = attempt + 1
                    WHERE task_id = ? AND status = ? AND attempt = ?
                      AND attempt < ?
                    """,
                    (
                        TaskStatus.RUNNING.value,
                        task.task_id,
                        TaskStatus.READY.value,
                        task.attempt,
                        max_attempts,
                    ),
                )
                if update.rowcount != 1:
                    raise RuntimeError(
                        f"Could not atomically claim task '{task.task_id}'."
                    )

                conn.execute(
                    """
                    INSERT INTO task_attempts (
                        attempt_id, task_id, attempt_no, worker_id, status,
                        started_at, lease_expires_at
                    ) VALUES (?, ?, ?, ?, 'running', ?, ?)
                    """,
                    (
                        attempt_id,
                        task.task_id,
                        attempt_no,
                        worker_id,
                        started_at.isoformat(),
                        lease_expires_at.isoformat(),
                    ),
                )
                claimed_task = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?", (task.task_id,)
                ).fetchone()
                if claimed_task is None:
                    raise TaskNotFoundError(task.task_id)
                claim = TaskClaim(
                    task=self._row_to_task_spec(claimed_task),
                    attempt_id=attempt_id,
                    attempt_no=attempt_no,
                    worker_id=worker_id,
                    lease_expires_at=lease_expires_at,
                )
                break

            result = ClaimResult(claim=claim, skipped_candidates=tuple(skipped))
            conn.execute(
                """
                INSERT INTO claim_requests (
                    claim_request_id, worker_id, run_id, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    claim_request_id,
                    worker_id,
                    run_id,
                    result.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
            return result
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def _claim_skip_reason(
        self, conn: sqlite3.Connection, task: TaskSpec
    ) -> ClaimSkipReason | None:
        if not ATTEMPT_POLICY.has_remaining_attempts(task.budget, task.attempt):
            return ClaimSkipReason.ATTEMPT_BUDGET_EXHAUSTED

        active_attempt = conn.execute(
            """
            SELECT 1 FROM task_attempts
            WHERE task_id = ? AND status = 'running' LIMIT 1
            """,
            (task.task_id,),
        ).fetchone()
        if active_attempt is not None:
            return ClaimSkipReason.ACTIVE_ATTEMPT_EXISTS

        for dependency_id in task.depends_on:
            if dependency_id == task.task_id:
                return ClaimSkipReason.SELF_DEPENDENCY
            dependency_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (dependency_id,)
            ).fetchone()
            if dependency_row is None:
                return ClaimSkipReason.DEPENDENCY_MISSING
            try:
                dependency = self._row_to_task_spec(dependency_row)
            except (KeyError, TypeError, ValueError):
                return ClaimSkipReason.DEPENDENCY_INVALID
            if dependency.run_id != task.run_id:
                return ClaimSkipReason.DEPENDENCY_CROSS_RUN
            if dependency.status != TaskStatus.ACCEPTED:
                return ClaimSkipReason.DEPENDENCY_NOT_ACCEPTED
        return None

    def retry_task(self, task_id: str) -> TaskSpec:
        """Atomically return a FAILED task to READY if attempts remain.

        The attempt counter is unchanged here. It increments only when
        claim_next_task() successfully starts the next execution.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise TaskNotFoundError(task_id)

            current_status = TaskStatus(row["status"])
            validate_transition(current_status, TaskStatus.READY)

            budget = TaskBudget.model_validate_json(row["budget_json"])
            max_attempts = min(budget.max_attempts, ATTEMPT_POLICY.max_attempts)
            self._ensure_attempt_budget_remaining(
                task_id, row["attempt"], budget.max_attempts
            )

            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = ?
                WHERE task_id = ?
                  AND status = ?
                  AND attempt < ?
                """,
                (
                    TaskStatus.READY.value,
                    task_id,
                    TaskStatus.FAILED.value,
                    max_attempts,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Could not atomically retry task '{task_id}'.")

            updated_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if updated_row is None:
                raise TaskNotFoundError(task_id)

            result = self._row_to_task_spec(updated_row)
            conn.commit()
            return result
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def _row_to_task_spec(self, row: sqlite3.Row) -> TaskSpec:
        budget_data = json.loads(row["budget_json"])
        depends_on_tuple = tuple(json.loads(row["depends_on_json"]))
        input_refs_tuple = tuple(json.loads(row["input_refs_json"]))
        allowed_tools_tuple = tuple(json.loads(row["allowed_tools_json"]))

        return TaskSpec(
            task_id=row["task_id"],
            run_id=row["run_id"],
            parent_task_id=row["parent_task_id"],
            agent_role=AgentRole(row["agent_role"]),
            goal=row["goal"],
            depends_on=depends_on_tuple,
            input_refs=input_refs_tuple,
            output_contract=row["output_contract"],
            allowed_tools=allowed_tools_tuple,
            budget=TaskBudget(**budget_data),
            status=TaskStatus(row["status"]),
            attempt=row["attempt"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
