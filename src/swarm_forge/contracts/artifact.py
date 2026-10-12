from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ArtifactKind(StrEnum):
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    CODE_PATCH = "code_patch"
    TEST_REPORT = "test_report"
    SECURITY_REPORT = "security_report"
    DOCUMENTATION = "documentation"
    RELEASE_REPORT = "release_report"

class ValidationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"

class ArtifactValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ValidationStatus = ValidationStatus.PENDING
    validator_name: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=2000)
    checked_at: datetime = Field(default_factory=utc_now)

class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(min_length=1)
    kind: ArtifactKind
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    schema_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    validation: ArtifactValidation = Field(
        default_factory=lambda: ArtifactValidation(
            validator_name="not_checked",
            message="Artifact has not been validated yet",
        )
    )
