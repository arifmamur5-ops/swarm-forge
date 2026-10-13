from pydantic import BaseModel, ConfigDict, Field


class PolicyDecision(BaseModel):
    """Deterministic decision about whether a task may proceed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    policy_version: str = "1.0"
