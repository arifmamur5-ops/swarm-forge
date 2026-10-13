from swarm_forge.contracts.event import EventType
from swarm_forge.contracts.task import AgentRole, TaskSpec
from swarm_forge.policy.gate import PolicyGate
from swarm_forge.storage.audit_log import AuditLog


def make_task(*, tools: tuple[str, ...]) -> TaskSpec:
    return TaskSpec(
        run_id="run-policy-audit",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Implement a small, tested change",
        output_contract="Code changes with passing tests",
        allowed_tools=tools,
    )


def test_allowed_decision_is_persisted_to_audit_log(
    tmp_path,
) -> None:
    audit_log = AuditLog(tmp_path / "audit.sqlite3")
    task = make_task(tools=("read_file", "run_tests"))

    decision = PolicyGate().evaluate_and_audit(
        task,
        audit_log,
        correlation_id="correlation-allowed",
    )

    events = audit_log.list_events(run_id=task.run_id)

    assert decision.allowed is True
    assert len(events) == 1
    assert events[0]["event_type"] == EventType.POLICY_ALLOWED.value
    assert events[0]["task_id"] == task.task_id
    assert events[0]["correlation_id"] == "correlation-allowed"
    assert events[0]["payload"] == {
        "allowed": True,
        "reasons": [],
        "policy_version": "1.0",
    }
    assert audit_log.verify_chain() is True


def test_denied_decision_and_reasons_are_persisted(
    tmp_path,
) -> None:
    audit_log = AuditLog(tmp_path / "audit.sqlite3")
    task = make_task(tools=("read_file", "shell"))

    decision = PolicyGate().evaluate_and_audit(
        task,
        audit_log,
        correlation_id="correlation-denied",
    )

    events = audit_log.list_events(run_id=task.run_id)

    assert decision.allowed is False
    assert decision.reasons == ("tools_not_allowed:shell",)
    assert len(events) == 1
    assert events[0]["event_type"] == EventType.POLICY_DENIED.value
    assert events[0]["payload"] == {
        "allowed": False,
        "reasons": ["tools_not_allowed:shell"],
        "policy_version": "1.0",
    }
    assert audit_log.verify_chain() is True


def test_audit_log_records_both_decisions_in_one_hash_chain(
    tmp_path,
) -> None:
    audit_log = AuditLog(tmp_path / "audit.sqlite3")
    gate = PolicyGate()

    gate.evaluate_and_audit(
        make_task(tools=("read_file",)),
        audit_log,
        correlation_id="correlation-1",
    )
    gate.evaluate_and_audit(
        make_task(tools=("shell",)),
        audit_log,
        correlation_id="correlation-2",
    )

    events = audit_log.list_events()

    assert [event["event_type"] for event in events] == [
        EventType.POLICY_ALLOWED.value,
        EventType.POLICY_DENIED.value,
    ]
    assert events[1]["prev_hash"] == events[0]["event_hash"]
    assert audit_log.verify_chain() is True
