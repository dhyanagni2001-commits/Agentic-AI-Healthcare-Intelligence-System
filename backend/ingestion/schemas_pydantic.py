"""Pydantic boundary-validation models for raw ingestion rows.

These validate shape/range at the CSV-row level, before a row is turned
into the stdlib `HospitalRecord` dataclass used by the rest of the app.
Validation failures are reported as `ValidationIssue`s (reusing the dataclass
already defined in backend/models/schemas.py) rather than silently dropped.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from backend.models.schemas import RiskLevel, ValidationIssue


class HospitalRecordIn(BaseModel):
    facility_id: str = Field(min_length=1)
    facility_name: str = Field(min_length=1)
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    hospital_type: Optional[str] = None
    ownership: Optional[str] = None
    overall_rating: Optional[int] = Field(default=None, ge=1, le=5)

    @field_validator("state")
    @classmethod
    def _state_len(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("state must be a 2-letter code")
        return v


class DoctorAggregateIn(BaseModel):
    affiliated_hospital_id: str = Field(min_length=1)
    affiliated_hospital_name: str = Field(min_length=1)
    specialty: str = Field(min_length=1)
    doctor_count: int = Field(ge=0)


class DepartmentSummaryIn(BaseModel):
    affiliated_hospital_name: str = Field(min_length=1)
    department: str = Field(min_length=1)
    doctor_count: int = Field(ge=0)


def issues_from_validation_error(facility_id: str, exc) -> list[ValidationIssue]:
    """Convert a pydantic ValidationError into ValidationIssue records."""
    out = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", ())) or "unknown"
        out.append(ValidationIssue(field=field, issue=err.get("msg", "invalid"),
                                    severity=RiskLevel.MEDIUM))
    return out
