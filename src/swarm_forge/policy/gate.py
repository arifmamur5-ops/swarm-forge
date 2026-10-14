from swarm_forge.contracts.event import AuditEvent, EventType
from swarm_forge.contracts.policy import PolicyDecision
from swarm_forge.contracts.task import TaskSpec
from swarm_forge.storage.audit_log import AuditLog

DEFAULT_ALLOWED_TOOLS = frozenset(
    {
        "read_file",
        "list_files",
        "search_text",
        "write_patch",
        "git_diff",
        "run_tests",
        "ruff_check",
        "mypy_check",
    }
)


class PolicyGate:
    """Evaluate tasks against deterministic, fail-closed rules."""

    def __init__(
        self,
        *,
        allowed_tools: frozenset[str] = DEFAULT_ALLOWED_TOOLS,
        max_attempts: int = 3,
        max_timeout_seconds: int = 600,
        max_changed_files: int = 10,
        max_input_tokens: int = 8000,
        max_output_tokens: int = 4000,
        policy_version: str = "1.0",
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_timeout_seconds < 1:
            raise ValueError("max_timeout_seconds must be at least 1")
        if max_changed_files < 1:
            raise ValueError("max_changed_files must be at least 1")
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be at least 1")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")

        self._allowed_tools = allowed_tools
        self._max_attempts = max_attempts
        self._max_timeout_seconds = max_timeout_seconds
        self._max_changed_files = max_changed_files
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._policy_version = policy_version

    def evaluate(self, task: TaskSpec) -> PolicyDecision:
        """Evaluate a task without writing to external storage."""
        reasons: list[str] = []

        unknown_tools = sorted(
            set(task.allowed_tools) - self._allowed_tools
        )
        if unknown_tools:
            reasons.append(
                f"tools_not_allowed:{','.join(unknown_tools)}"
            )

        if task.attempt >= task.budget.max_attempts or task.budget.max_attempts > self._max_attempts:
            reasons.append("max_attempts_exceeded")

        if task.budget.timeout_seconds > self._max_timeout_seconds:
            reasons.append("timeout_seconds_exceeded")

        if task.budget.max_changed_files > self._max_changed_files:
            reasons.append("max_changed_files_exceeded")

        if task.budget.max_input_tokens > self._max_input_tokens:
            reasons.append("max_input_tokens_exceeded")

        if task.budget.max_output_tokens > self._max_output_tokens:
            reasons.append("max_output_tokens_exceeded")

        return PolicyDecision(
            allowed=not reasons,
            reasons=tuple(reasons),
            policy_version=self._policy_version,
        )

    def evaluate_and_audit(
        self,
        task: TaskSpec,
        audit_log: AuditLog,
        *,
        correlation_id: str,
    ) -> PolicyDecision:
        """Evaluate a task and persist its decision before returning."""
        decision = self.evaluate(task)

        event_type = (
            EventType.POLICY_ALLOWED
            if decision.allowed
            else EventType.POLICY_DENIED
        )

        event = AuditEvent(
            run_id=task.run_id,
            task_id=task.task_id,
            event_type=event_type,
            correlation_id=correlation_id,
            payload={
                "allowed": decision.allowed,
                "reasons": list(decision.reasons),
                "policy_version": decision.policy_version,
            },
        )

        # If persistence fails, propagate the error instead of returning
        # a decision that was expected to be audited.
        audit_log.append_event(event)

        return decision
