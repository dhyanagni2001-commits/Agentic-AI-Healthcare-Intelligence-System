"""Clinical validation engine — stdlib only."""
from __future__ import annotations
from typing import List
from backend.models.schemas import (
    HospitalRecord, ValidationResult, ValidationIssue, RiskLevel
)


def _issues(h: HospitalRecord) -> List[ValidationIssue]:
    out: List[ValidationIssue] = []

    if h.capabilities.emergency_services and not h.capabilities.icu:
        out.append(ValidationIssue("capabilities.icu",
            "Hospital has Emergency Services but no ICU — critical patients at risk",
            RiskLevel.HIGH))

    if h.capabilities.surgery and h.doctor_count == 0:
        out.append(ValidationIssue("doctor_count",
            "Surgery capability reported but zero doctors on record",
            RiskLevel.HIGH))

    if not h.city:
        out.append(ValidationIssue("city", "City is missing", RiskLevel.MEDIUM))
    if not h.state:
        out.append(ValidationIssue("state", "State is missing", RiskLevel.MEDIUM))

    if not h.phone:
        out.append(ValidationIssue("phone", "No phone number on record", RiskLevel.LOW))

    if (h.capabilities.emergency_services
            and h.quality.overall_rating is not None
            and h.quality.overall_rating <= 2):
        out.append(ValidationIssue("quality.overall_rating",
            f"Emergency-capable hospital has low rating ({h.quality.overall_rating}/5)",
            RiskLevel.HIGH))

    if h.doctor_count == 0 and not any(vars(h.capabilities).values()):
        out.append(ValidationIssue("capabilities",
            "No doctors and no capabilities — record may be incomplete",
            RiskLevel.MEDIUM))

    below = sum(1 for v in [
        h.quality.mortality_comparison, h.quality.safety_comparison,
        h.quality.readmission_comparison, h.quality.effectiveness_comparison,
    ] if v and "below" in str(v).lower())
    if below >= 2:
        out.append(ValidationIssue("quality",
            f"{below} quality metrics are below national average",
            RiskLevel.HIGH))

    if h.confidence_score < 0.5:
        out.append(ValidationIssue("confidence_score",
            f"Low confidence score ({h.confidence_score:.0%}) — manual review recommended",
            RiskLevel.MEDIUM))

    return out


def _risk(issues: List[ValidationIssue]) -> RiskLevel:
    if not issues: return RiskLevel.LOW
    sevs = [i.severity for i in issues]
    if RiskLevel.CRITICAL in sevs: return RiskLevel.CRITICAL
    highs = sevs.count(RiskLevel.HIGH)
    if highs >= 3: return RiskLevel.CRITICAL
    if highs >= 1: return RiskLevel.HIGH
    if RiskLevel.MEDIUM in sevs: return RiskLevel.MEDIUM
    return RiskLevel.LOW


def validate_hospital(h: HospitalRecord) -> ValidationResult:
    issues   = _issues(h)
    risk     = _risk(issues)
    valid    = risk not in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    warnings = [f"[LOW] {i.field}: {i.issue}" for i in issues if i.severity == RiskLevel.LOW]
    return ValidationResult(
        facility_id=h.facility_id, valid=valid,
        issues=issues, risk_level=risk,
        confidence_score=h.confidence_score, warnings=warnings,
    )


def validate_many(hospitals: List[HospitalRecord]) -> List[ValidationResult]:
    return [validate_hospital(h) for h in hospitals]
