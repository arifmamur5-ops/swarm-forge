from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from swarm_forge.contracts.event import AuditEvent

GENESIS_HASH = "0" * 64

def utc_now() -> datetime:
    return datetime.now(UTC)

def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

def calculate_event_hash(
    *,
    event_id: str,
    run_id: str,
    task_id: str | None,
    event_type: str,
    timestamp: str,
    correlation_id: str,
    payload_json: str,
    prev_hash: str,
) -> str:
    content = canonical_json(
        {
            "event_id": event_id,
            "run_id": run_id,
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "correlation_id": correlation_id,
            "payload_json": payload_json,
            "prev_hash": prev_hash,
        }
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

class AuditLog:
    """Append-only SQLite audit log with a SHA-256 hash chain."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                )
                """
            )

    def _append_record(
        self,
        *,
        event_id: str,
        run_id: str,
        task_id: str | None,
        event_type: str,
        timestamp: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not run_id.strip():
            raise ValueError("run_id must not be blank")
        if not event_type.strip():
            raise ValueError("event_type must not be blank")
        if not correlation_id.strip():
            raise ValueError("correlation_id must not be blank")

        payload_json = canonical_json(payload)

        with self._connect() as connection:
            # Serialize concurrent writers before reading the chain tip.
            connection.execute("BEGIN IMMEDIATE")

            previous = connection.execute(
                """
                SELECT event_hash
                FROM audit_events
                ORDER BY sequence DESC
                LIMIT 1
                """
            ).fetchone()

            prev_hash = (
                str(previous["event_hash"])
                if previous is not None
                else GENESIS_HASH
            )

            event = {
                "event_id": event_id,
                "run_id": run_id,
                "task_id": task_id,
                "event_type": event_type,
                "timestamp": timestamp,
                "correlation_id": correlation_id,
                "payload_json": payload_json,
                "prev_hash": prev_hash,
            }

            event_hash = calculate_event_hash(
                event_id=event_id,
                run_id=run_id,
                task_id=task_id,
                event_type=event_type,
                timestamp=timestamp,
                correlation_id=correlation_id,
                payload_json=payload_json,
                prev_hash=prev_hash,
            )

            connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, run_id, task_id, event_type,
                    timestamp, correlation_id, payload_json,
                    prev_hash, event_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    run_id,
                    task_id,
                    event_type,
                    timestamp,
                    correlation_id,
                    payload_json,
                    prev_hash,
                    event_hash,
                ),
            )

            return {
                **event,
                "payload": payload,
                "event_hash": event_hash,
            }

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        correlation_id: str,
        payload: dict[str, Any],
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a new event using a generated ID and timestamp."""
        return self._append_record(
            event_id=str(uuid4()),
            run_id=run_id,
            task_id=task_id,
            event_type=event_type,
            timestamp=utc_now().isoformat(),
            correlation_id=correlation_id,
            payload=payload,
        )

    def append_event(self, event: AuditEvent) -> AuditEvent:
        """Persist an AuditEvent without changing its ID or timestamp."""
        if event.prev_hash is not None or event.event_hash is not None:
            raise ValueError(
                "Hash fields must be assigned by AuditLog, not by the caller"
            )

        stored = self._append_record(
            event_id=event.event_id,
            run_id=event.run_id,
            task_id=event.task_id,
            event_type=event.event_type.value,
            timestamp=event.timestamp.isoformat(),
            correlation_id=event.correlation_id,
            payload=event.payload,
        )

        return event.model_copy(
            update={
                "prev_hash": stored["prev_hash"],
                "event_hash": stored["event_hash"],
            }
        )

    def list_events(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if run_id is None:
                rows = connection.execute(
                    "SELECT * FROM audit_events ORDER BY sequence"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM audit_events
                    WHERE run_id = ?
                    ORDER BY sequence
                    """,
                    (run_id,),
                ).fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            events.append(event)

        return events

    def verify_chain(self) -> bool:
        try:
            events = self.list_events()
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            # Stored payloads are untrusted input during verification. Database
            # errors are deliberately not caught here.
            return False

        expected_prev_hash = GENESIS_HASH

        for event in events:
            if not isinstance(event["payload"], dict):
                return False
            if event["prev_hash"] != expected_prev_hash:
                return False

            try:
                payload_json = canonical_json(event["payload"])
                calculated = calculate_event_hash(
                    event_id=event["event_id"],
                    run_id=event["run_id"],
                    task_id=event["task_id"],
                    event_type=event["event_type"],
                    timestamp=event["timestamp"],
                    correlation_id=event["correlation_id"],
                    payload_json=payload_json,
                    prev_hash=event["prev_hash"],
                )
            except (TypeError, ValueError, RecursionError):
                return False

            if calculated != event["event_hash"]:
                return False

            expected_prev_hash = event["event_hash"]

        return True
