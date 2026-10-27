from __future__ import annotations

from dataclasses import dataclass

from swarm_forge.contracts.task import TaskBudget

GLOBAL_MAX_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class AttemptPolicy:
    """Immutable global limit applied by policy and lifecycle persistence."""

    max_attempts: int = GLOBAL_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if self.max_attempts != GLOBAL_MAX_ATTEMPTS:
            raise ValueError(
                f"The global max_attempts policy is fixed at {GLOBAL_MAX_ATTEMPTS}"
            )

    def accepts_new_budget(self, budget: TaskBudget) -> bool:
        return budget.max_attempts <= self.max_attempts

    def has_remaining_attempts(self, budget: TaskBudget, attempt: int) -> bool:
        return self.accepts_new_budget(budget) and attempt < budget.max_attempts

    def validate_new_budget(self, budget: TaskBudget) -> None:
        if not self.accepts_new_budget(budget):
            raise ValueError(
                f"Task budget max_attempts ({budget.max_attempts}) exceeds "
                f"the global maximum of {self.max_attempts}."
            )


ATTEMPT_POLICY = AttemptPolicy()
