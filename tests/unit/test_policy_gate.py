from pathlib import Path

import pytest

from swarm_forge.contracts.task import AgentRole, TaskBudget, TaskSpec
from swarm_forge.policy.attempt_policy import ATTEMPT_POLICY, AttemptPolicy
from swarm_forge.policy.gate import PolicyGate
from swarm_forge.storage.task_store import TaskStore


def make_task(
    *,
    tools: tuple[str, ...] = ("read_file", "run_tests"),
    budget: TaskBudget | None = None,
    attempt: int = 0,
) -> TaskSpec:
    return TaskSpec(
        run_id="run-policy-test",
        agent_role=AgentRole.IMPLEMENTER,
        goal="Implement a small, tested change",
        output_contract="Code changes with passing tests",
        allowed_tools=tools,
        budget=budget or TaskBudget(),
        attempt=attempt,
    )


def test_allows_task_with_permitted_tools() -> None:
    decision = PolicyGate().evaluate(make_task())

    assert decision.allowed is True
    assert decision.reasons == ()
    assert decision.policy_version == "1.0"


def test_rejects_unknown_or_dangerous_tool() -> None:
    task = make_task(tools=("read_file", "shell"))

    decision = PolicyGate().evaluate(task)

    assert decision.allowed is False
    assert "tools_not_allowed:shell" in decision.reasons


def test_rejects_multiple_disallowed_tools() -> None:
    task = make_task(tools=("network", "read_secrets", "shell"))

    decision = PolicyGate().evaluate(task)

    assert decision.allowed is False
    assert "tools_not_allowed:network,read_secrets,shell" in decision.reasons


@pytest.mark.parametrize(
    ("budget", "attempt", "expected_reason"),
    [
        (TaskBudget(max_attempts=4), 0, "max_attempts_exceeded"),
        (TaskBudget(max_attempts=5), 0, "max_attempts_exceeded"),
        (TaskBudget(max_attempts=3), 3, "max_attempts_exceeded"),
        (TaskBudget(timeout_seconds=601), 0, "timeout_seconds_exceeded"),
        (TaskBudget(max_changed_files=11), 0, "max_changed_files_exceeded"),
        (TaskBudget(max_input_tokens=8001), 0, "max_input_tokens_exceeded"),
        (TaskBudget(max_output_tokens=4001), 0, "max_output_tokens_exceeded"),
    ],
)
def test_rejects_budget_over_policy_limit(
    budget: TaskBudget,
    attempt: int,
    expected_reason: str,
) -> None:
    decision = PolicyGate().evaluate(make_task(budget=budget, attempt=attempt))

    assert decision.allowed is False
    assert expected_reason in decision.reasons


def test_allows_budget_at_exact_limit() -> None:
    budget = TaskBudget(
        max_attempts=3,
        timeout_seconds=600,
        max_changed_files=10,
        max_input_tokens=8000,
        max_output_tokens=4000,
    )
    decision = PolicyGate().evaluate(make_task(budget=budget, attempt=2))

    assert decision.allowed is True
    assert decision.reasons == ()


def test_can_configure_a_stricter_tool_allowlist() -> None:
    gate = PolicyGate(allowed_tools=frozenset({"read_file"}))
    task = make_task(tools=("read_file", "run_tests"))

    decision = gate.evaluate(task)

    assert decision.allowed is False
    assert decision.reasons == ("tools_not_allowed:run_tests",)


def test_rejects_invalid_policy_limits() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        PolicyGate(max_attempts=0)
    with pytest.raises(ValueError, match="global attempt policy"):
        PolicyGate(max_attempts=4)
    with pytest.raises(ValueError, match="fixed at 3"):
        AttemptPolicy(max_attempts=4)
    with pytest.raises(ValueError, match="max_input_tokens"):
        PolicyGate(max_input_tokens=0)


@pytest.mark.parametrize("max_attempts", [1, 3])
def test_global_attempt_policy_accepts_budgets_within_cap(max_attempts: int) -> None:
    budget = TaskBudget(max_attempts=max_attempts)

    assert ATTEMPT_POLICY.accepts_new_budget(budget) is True
    assert PolicyGate().evaluate(make_task(budget=budget)).allowed is True


@pytest.mark.parametrize(
    ("max_attempts", "allowed"), [(1, True), (3, True), (4, False), (5, False)]
)
def test_policy_gate_and_task_store_agree_on_new_budget(
    tmp_path: Path, max_attempts: int, allowed: bool
) -> None:
    task = make_task(budget=TaskBudget(max_attempts=max_attempts))
    store = TaskStore(tmp_path / f"attempt-policy-{max_attempts}.db")

    assert PolicyGate().evaluate(task).allowed is allowed
    if allowed:
        store.save_task(task)
        assert store.get_task(task.task_id).budget == task.budget
    else:
        with pytest.raises(ValueError, match="global maximum of 3"):
            store.save_task(task)
        assert store.list_tasks() == []
