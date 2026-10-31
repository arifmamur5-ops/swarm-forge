"""SQLite persistence primitives for immutable lifecycle outbox events.

All functions accept a caller-owned connection. They never open a connection,
begin a transaction, commit, or roll back.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import NamedTuple

from swarm_forge.contracts.event import LifecycleEvent

LIFECYCLE_EVENT_SCHEMA_VERSION = 1


class OutboxError(RuntimeError):
    """Base class for lifecycle outbox contract conflicts."""


class OperationIdempotencyConflictError(OutboxError):
    """An operation idempotency key was reused with a different fingerprint."""


class OutboxManifestMismatchError(OutboxError):
    """An operation replay has a different or incomplete event manifest."""


class OutboxTransactionRequiredError(OutboxError):
    """A write operation was called without an active caller-owned transaction."""


class EventIdentityConflictError(OutboxError):
    """An event identity was reused with different immutable event content."""


class EventDiscriminatorConflictError(EventIdentityConflictError):
    """An operation discriminator was reused with different event content."""


class OperationWriteResult(StrEnum):
    CREATED = "created"
    REPLAYED = "replayed"


class _CanonicalEvent(NamedTuple):
    event: LifecycleEvent
    event_json: str


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_event(event: LifecycleEvent) -> _CanonicalEvent:
    # Revalidate a plain serialized mapping. This also protects this persistence
    # boundary from LifecycleEvent instances made through Pydantic escape hatches.
    event_json = _canonical_json(event.model_dump(mode="json"))
    validated = LifecycleEvent.model_validate_json(event_json)
    return _CanonicalEvent(validated, event_json)


def _canonical_events(
    events: Sequence[LifecycleEvent], *, operation_id: str, idempotency_key: str
) -> tuple[list[_CanonicalEvent], str]:
    if not events:
        raise OutboxManifestMismatchError("An operation must manifest at least one event")

    canonical_events = [_canonical_event(event) for event in events]
    discriminators = [item.event.event_discriminator for item in canonical_events]
    event_ids = [item.event.event_id for item in canonical_events]
    if len(set(discriminators)) != len(discriminators) or len(set(event_ids)) != len(
        event_ids
    ):
        raise OutboxManifestMismatchError(
            "An operation manifest cannot contain duplicate event identities"
        )

    for item in canonical_events:
        if (
            item.event.operation_id != operation_id
            or item.event.idempotency_key != idempotency_key
        ):
            raise OutboxManifestMismatchError(
                "Every event must match the operation receipt identity"
            )

    manifest_items = sorted(
        (
            item.event.event_discriminator,
            item.event.event_id,
        )
        for item in canonical_events
    )
    return canonical_events, _canonical_json(manifest_items)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _require_active_transaction(connection: sqlite3.Connection) -> None:
    if not connection.in_transaction:
        raise OutboxTransactionRequiredError(
            "Lifecycle outbox writes require an active caller-owned transaction"
        )


def _preflight_event_conflicts(
    connection: sqlite3.Connection,
    events: Sequence[_CanonicalEvent],
) -> None:
    """Reject currently visible identity/content conflicts before any insert."""
    for canonical in events:
        event = canonical.event
        existing = connection.execute(
            """
            SELECT operation_id, idempotency_key, event_discriminator, event_json
            FROM lifecycle_outbox WHERE event_id = ?
            """,
            (event.event_id,),
        ).fetchone()
        if existing is None:
            existing = connection.execute(
                """
                SELECT operation_id, idempotency_key, event_discriminator, event_json
                FROM lifecycle_outbox
                WHERE operation_id = ? AND event_discriminator = ?
                """,
                (event.operation_id, event.event_discriminator),
            ).fetchone()
        if existing is None:
            continue
        if (
            existing["operation_id"] == event.operation_id
            and existing["event_discriminator"] == event.event_discriminator
            and existing["idempotency_key"] == event.idempotency_key
            and existing["event_json"] == canonical.event_json
        ):
            continue
        if (
            existing["operation_id"] == event.operation_id
            and existing["event_discriminator"] == event.event_discriminator
            and existing["idempotency_key"] != event.idempotency_key
        ):
            raise EventDiscriminatorConflictError(
                "Operation event discriminator was reused with different content"
            )
        if (
            existing["operation_id"] == event.operation_id
            and existing["event_discriminator"] == event.event_discriminator
        ):
            raise EventIdentityConflictError(
                "Event identity was reused with different immutable content"
            )
        raise EventIdentityConflictError(
            "Event ID is already assigned to another immutable event"
        )


def _check_operation_receipt(
    connection: sqlite3.Connection,
    *,
    operation_id: str,
    idempotency_key: str,
    operation_fingerprint: str,
    manifest: str,
) -> bool:
    existing = connection.execute(
        """
        SELECT operation_fingerprint, event_manifest
        FROM lifecycle_outbox_operations
        WHERE operation_id = ? AND idempotency_key = ?
        """,
        (operation_id, idempotency_key),
    ).fetchone()
    if existing is None:
        return False
    if existing["operation_fingerprint"] != operation_fingerprint:
        raise OperationIdempotencyConflictError(
            "Operation idempotency key was reused with different content"
        )
    if existing["event_manifest"] != manifest:
        raise OutboxManifestMismatchError(
            "Operation replay has a different event manifest"
        )
    return True


def _register_operation(
    connection: sqlite3.Connection,
    *,
    operation_id: str,
    idempotency_key: str,
    operation_fingerprint: str,
    events: Sequence[LifecycleEvent],
) -> bool:
    """Internal receipt primitive; prefer :func:`persist_operation`.

    Returns True for a new receipt and False for an exact fingerprint and
    manifest match. This function does not verify event rows or commit. The
    receipt and every event must be persisted in one caller-owned transaction;
    the caller must roll back that transaction after any exception.
    """
    _require_active_transaction(connection)
    if not operation_fingerprint.strip():
        raise ValueError("operation_fingerprint must not be blank")
    _, manifest = _canonical_events(
        events, operation_id=operation_id, idempotency_key=idempotency_key
    )
    if _check_operation_receipt(
        connection,
        operation_id=operation_id,
        idempotency_key=idempotency_key,
        operation_fingerprint=operation_fingerprint,
        manifest=manifest,
    ):
        return False

    connection.execute(
        """
        INSERT INTO lifecycle_outbox_operations (
            operation_id, idempotency_key, operation_fingerprint,
            event_manifest, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            operation_id,
            idempotency_key,
            operation_fingerprint,
            manifest,
            _utc_timestamp(),
        ),
    )
    return True


def _insert_canonical_event(
    connection: sqlite3.Connection,
    canonical: _CanonicalEvent,
) -> bool:
    _require_active_transaction(connection)
    event = canonical.event
    receipt = connection.execute(
        """
        SELECT event_manifest FROM lifecycle_outbox_operations
        WHERE operation_id = ? AND idempotency_key = ?
        """,
        (event.operation_id, event.idempotency_key),
    ).fetchone()
    if receipt is None:
        raise OutboxManifestMismatchError(
            "Register the operation receipt before inserting its lifecycle events"
        )
    manifest = json.loads(receipt["event_manifest"])
    if [event.event_discriminator, event.event_id] not in manifest:
        raise OutboxManifestMismatchError(
            "Event identity is not present in the operation manifest"
        )

    existing_id = connection.execute(
        "SELECT operation_id, idempotency_key, event_discriminator, event_json "
        "FROM lifecycle_outbox WHERE event_id = ?",
        (event.event_id,),
    ).fetchone()
    if existing_id is not None:
        if (
            existing_id["operation_id"] == event.operation_id
            and existing_id["event_discriminator"] == event.event_discriminator
        ):
            if existing_id["event_json"] != canonical.event_json:
                if existing_id["idempotency_key"] != event.idempotency_key:
                    raise EventDiscriminatorConflictError(
                        "Operation event discriminator was reused with different content"
                    )
                raise EventIdentityConflictError(
                    "Event ID was replayed with different immutable content"
                )
            return False
        raise EventIdentityConflictError(
            "Event ID is already assigned to another immutable event"
        )

    existing_discriminator = connection.execute(
        """
        SELECT event_id, event_json FROM lifecycle_outbox
        WHERE operation_id = ? AND event_discriminator = ?
        """,
        (event.operation_id, event.event_discriminator),
    ).fetchone()
    if existing_discriminator is not None:
        if (
            existing_discriminator["event_id"] != event.event_id
            or existing_discriminator["event_json"] != canonical.event_json
        ):
            raise EventDiscriminatorConflictError(
                "Operation event discriminator was reused with different content"
            )
        return False

    timestamp = _utc_timestamp()
    connection.execute(
        """
        INSERT INTO lifecycle_outbox (
            event_id, operation_id, idempotency_key, event_discriminator,
            schema_version, run_id, task_id, attempt_id, correlation_id,
            event_type, occurred_at, event_json, status, retry_count,
            next_attempt_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
        """,
        (
            event.event_id,
            event.operation_id,
            event.idempotency_key,
            event.event_discriminator,
            LIFECYCLE_EVENT_SCHEMA_VERSION,
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
    return True


def _insert_lifecycle_event(
    connection: sqlite3.Connection,
    event: LifecycleEvent,
) -> bool:
    """Internal single-row primitive; prefer :func:`persist_operation`."""
    _require_active_transaction(connection)
    canonical = _canonical_event(event)
    return _insert_canonical_event(connection, canonical)


def persist_operation(
    connection: sqlite3.Connection,
    *,
    operation_id: str,
    idempotency_key: str,
    operation_fingerprint: str,
    events: Sequence[LifecycleEvent],
) -> OperationWriteResult:
    """Persist one complete operation in the caller's active transaction.

    The manifest and all event rows are written atomically by the caller. The
    caller must roll back the transaction if this function raises. The
    repository never begins, commits, or rolls back that transaction.
    """
    _require_active_transaction(connection)
    canonical_events, manifest = _canonical_events(
        events, operation_id=operation_id, idempotency_key=idempotency_key
    )
    if not operation_fingerprint.strip():
        raise ValueError("operation_fingerprint must not be blank")
    receipt_exists = _check_operation_receipt(
        connection,
        operation_id=operation_id,
        idempotency_key=idempotency_key,
        operation_fingerprint=operation_fingerprint,
        manifest=manifest,
    )
    _preflight_event_conflicts(connection, canonical_events)
    is_new = not receipt_exists
    if is_new:
        _register_operation(
            connection,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            operation_fingerprint=operation_fingerprint,
            events=[item.event for item in canonical_events],
        )
    if is_new:
        for item in canonical_events:
            _insert_canonical_event(connection, item)
        return OperationWriteResult.CREATED

    rows = connection.execute(
        """
        SELECT event_id, event_discriminator, event_json
        FROM lifecycle_outbox
        WHERE operation_id = ? AND idempotency_key = ?
        """,
        (operation_id, idempotency_key),
    ).fetchall()
    expected_by_identity = {
        (item.event.event_discriminator, item.event.event_id): item.event_json
        for item in canonical_events
    }
    actual_by_identity = {
        (row["event_discriminator"], row["event_id"]): row["event_json"]
        for row in rows
    }
    if len(rows) != len(actual_by_identity) or set(actual_by_identity) != set(
        expected_by_identity
    ):
        raise OutboxManifestMismatchError(
            "Persisted event rows do not exactly match the operation manifest"
        )
    for identity, expected_json in expected_by_identity.items():
        if actual_by_identity[identity] != expected_json:
            raise EventIdentityConflictError(
                "Event identity was replayed with different immutable content"
            )

    return OperationWriteResult.REPLAYED


def get_lifecycle_event(
    connection: sqlite3.Connection, event_id: str
) -> LifecycleEvent | None:
    row = connection.execute(
        "SELECT schema_version, event_json FROM lifecycle_outbox WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        return None
    if row["schema_version"] != LIFECYCLE_EVENT_SCHEMA_VERSION:
        raise OutboxError(
            f"Unsupported lifecycle event schema version {row['schema_version']}"
        )
    return LifecycleEvent.model_validate_json(row["event_json"])


def list_pending_lifecycle_events(
    connection: sqlite3.Connection, *, limit: int = 100
) -> list[LifecycleEvent]:
    if limit < 1:
        raise ValueError("limit must be positive")
    rows = connection.execute(
        """
        SELECT schema_version, event_json FROM lifecycle_outbox
        WHERE status = 'pending'
        ORDER BY sequence_no
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    events: list[LifecycleEvent] = []
    for row in rows:
        if row["schema_version"] != LIFECYCLE_EVENT_SCHEMA_VERSION:
            raise OutboxError(
                f"Unsupported lifecycle event schema version {row['schema_version']}"
            )
        events.append(LifecycleEvent.model_validate_json(row["event_json"]))
    return events
