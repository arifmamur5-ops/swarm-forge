from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from swarm_forge.contracts.task import AgentRole, TaskBudget, TaskSpec, TaskStatus
from swarm_forge.workflow.state_machine import validate_transition


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

class TaskStore:
    """SQLite-backed persistent repository for TaskSpec objects."""

    def __init__(self, db_path: str | Path = "swarm_forge.db") -> None:
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self._persistent_conn: sqlite3.Connection | None = None
        
        # Handle in-memory database explicitly by keeping a single persistent connection open
        if str(self.db_path) == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.row_factory = sqlite3.Row

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if str(self.db_path) != ":memory:" and self.db_path != Path(":memory:") and self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        try:
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
            conn.commit()
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

    def update_status(self, task_id: str, new_status: TaskStatus) -> TaskSpec:
        """Atomically update a task's status with workflow state machine validation.

        Uses BEGIN IMMEDIATE to acquire a write lock immediately and prevent race conditions.
        """
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
            
            # Validate via workflow state machine
            validate_transition(current_status, new_status)

            cursor.execute(
                "UPDATE tasks SET status = ? WHERE task_id = ?",
                (new_status.value, task_id),
            )
            conn.commit()

            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            updated_row = cursor.fetchone()
            if not updated_row:
                raise TaskNotFoundError(task_id)
            return self._row_to_task_spec(updated_row)
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._persistent_conn is None:
                conn.close()

    def start_attempt(self, task_id: str) -> TaskSpec:
        """Atomically increment attempt and transition READY to RUNNING."""
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise TaskNotFoundError(task_id)

            current_status = TaskStatus(row["status"])
            validate_transition(current_status, TaskStatus.RUNNING)

            budget_data = json.loads(row["budget_json"])
            max_attempts = budget_data["max_attempts"]
            if row["attempt"] >= max_attempts:
                raise AttemptLimitExceededError(task_id, max_attempts)

            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = ?, attempt = attempt + 1
                WHERE task_id = ? AND status = ? AND attempt < ?
                """,
                (
                    TaskStatus.RUNNING.value,
                    task_id,
                    TaskStatus.READY.value,
                    max_attempts,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Could not atomically start attempt for task '{task_id}'."
                )

            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise TaskNotFoundError(task_id)

            result = self._row_to_task_spec(row)
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
