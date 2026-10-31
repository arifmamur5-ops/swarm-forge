from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OperationContext(BaseModel):
    """Explicit immutable identity and correlation context for an operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @field_validator("operation_id", "correlation_id", "source")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value must not be blank")
        return value
