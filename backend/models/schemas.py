"""
Data models for the Healthcare Intelligence Platform.
Uses Python stdlib dataclasses only — zero external dependencies.
Compatible with Python 3.9+
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class GapType(str, Enum):
    NO_EMERGENCY      = "no_emergency_services"
    NO_ICU            = "no_icu"
    LOW_DOCTOR_DENSITY= "low_doctor_density"
    NO_SPECIALIST     = "no_specialist"
    LOW_QUALITY       = "low_quality_metrics"


class RecommendationType(str, Enum):
    INVEST       = "invest_in_facility"
    TRANSFER     = "patient_transfer"
    DEPLOY_STAFF = "deploy_staff"
    UPGRADE      = "upgrade_capability"
    CLOSE_GAP    = "close_service_gap"


# ── Hospital ──────────────────────────────────────────────────────────────────

@dataclass
class HospitalCapabilities:
    emergency_services: bool = False
    icu:                bool = False
    surgery:            bool = False
    maternity:          bool = False
    pediatrics:         bool = False
    rehabilitation:     bool = False
    mental_health:      bool = False
    cardiac_care:       bool = False
    oncology:           bool = False
    trauma_center:      bool = False


@dataclass
class HospitalQualityMetrics:
    overall_rating:                 Optional[int]  = None
    mortality_comparison:           Optional[str]  = None
    safety_comparison:              Optional[str]  = None
    readmission_comparison:         Optional[str]  = None
    patient_experience_comparison:  Optional[str]  = None
    effectiveness_comparison:       Optional[str]  = None
    timeliness_comparison:          Optional[str]  = None
    meets_ehr_criteria:             Optional[bool] = None


@dataclass
class HospitalRecord:
    facility_id:      str
    facility_name:    str
    address:          Optional[str] = None
    city:             Optional[str] = None
    state:            Optional[str] = None
    zip_code:         Optional[str] = None
    county:           Optional[str] = None
    phone:            Optional[str] = None
    hospital_type:    Optional[str] = None
    ownership:        Optional[str] = None
    capabilities:     HospitalCapabilities  = field(default_factory=HospitalCapabilities)
    quality:          HospitalQualityMetrics = field(default_factory=HospitalQualityMetrics)
    doctor_count:     int           = 0
    department_count: int           = 0
    departments:      List[str]     = field(default_factory=list)
    confidence_score: float         = 1.0
    notes:            Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["capabilities"] = asdict(self.capabilities)
        d["quality"]      = asdict(self.quality)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HospitalRecord":
        caps  = HospitalCapabilities(**d.pop("capabilities", {}))
        qual  = HospitalQualityMetrics(**d.pop("quality", {}))
        depts = d.pop("departments", [])
        return HospitalRecord(capabilities=caps, quality=qual, departments=depts, **d)


# ── Validation ────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    field:    str
    issue:    str
    severity: RiskLevel


@dataclass
class ValidationResult:
    facility_id:      str
    valid:            bool
    issues:           List[ValidationIssue]
    risk_level:       RiskLevel
    confidence_score: float
    warnings:         List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facility_id":      self.facility_id,
            "valid":            self.valid,
            "issues":           [{"field": i.field, "issue": i.issue,
                                  "severity": i.severity.value} for i in self.issues],
            "risk_level":       self.risk_level.value,
            "confidence_score": self.confidence_score,
            "warnings":         self.warnings,
        }


# ── Gaps ──────────────────────────────────────────────────────────────────────

@dataclass
class HealthcareGap:
    gap_type:           GapType
    description:        str
    affected_facilities: List[str]
    severity:           RiskLevel
    region:             Optional[str] = None
    population_impact:  Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_type":            self.gap_type.value,
            "description":         self.description,
            "affected_facilities": self.affected_facilities,
            "severity":            self.severity.value,
            "region":              self.region,
            "population_impact":   self.population_impact,
        }


@dataclass
class GapAnalysisResult:
    state:               Optional[str]
    city:                Optional[str]
    gaps:                List[HealthcareGap]
    overall_risk:        RiskLevel
    summary:             str
    facilities_analyzed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state":               self.state,
            "city":                self.city,
            "gaps":                [g.to_dict() for g in self.gaps],
            "overall_risk":        self.overall_risk.value,
            "summary":             self.summary,
            "facilities_analyzed": self.facilities_analyzed,
        }


# ── Recommendations ───────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    type:                RecommendationType
    priority:            RiskLevel
    title:               str
    description:         str
    target_facility_ids: List[str]     = field(default_factory=list)
    target_region:       Optional[str] = None
    rationale:           str           = ""
    estimated_impact:    Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":                self.type.value,
            "priority":            self.priority.value,
            "title":               self.title,
            "description":         self.description,
            "target_facility_ids": self.target_facility_ids,
            "target_region":       self.target_region,
            "rationale":           self.rationale,
            "estimated_impact":    self.estimated_impact,
        }


# ── Agent ─────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    step_name:      str
    description:    str
    data_used:      List[str]
    output_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    query:               str
    answer:              str
    reasoning_steps:     List[ReasoningStep]    = field(default_factory=list)
    hospitals_referenced: List[str]             = field(default_factory=list)
    gaps_identified:     List[HealthcareGap]    = field(default_factory=list)
    recommendations:     List[Recommendation]   = field(default_factory=list)
    confidence:          float                  = 0.8
    data_sources:        List[str]              = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query":                self.query,
            "answer":               self.answer,
            "reasoning_steps":      [s.to_dict() for s in self.reasoning_steps],
            "hospitals_referenced": self.hospitals_referenced,
            "gaps_identified":      [g.to_dict() for g in self.gaps_identified],
            "recommendations":      [r.to_dict() for r in self.recommendations],
            "confidence":           self.confidence,
            "data_sources":         self.data_sources,
        }


# ── Parse Response ────────────────────────────────────────────────────────────

@dataclass
class ParseResponse:
    success:          bool
    raw_input:        str
    hospital:         Optional[HospitalRecord] = None
    processing_notes: List[str]                = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":          self.success,
            "raw_input":        self.raw_input,
            "hospital":         self.hospital.to_dict() if self.hospital else None,
            "processing_notes": self.processing_notes,
        }
