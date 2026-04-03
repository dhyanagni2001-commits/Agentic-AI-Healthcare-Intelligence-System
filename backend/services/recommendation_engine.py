"""Recommendation engine — stdlib only."""
from __future__ import annotations
from typing import List
from backend.models.schemas import (
    GapAnalysisResult, GapType, HealthcareGap,
    HospitalRecord, Recommendation, RecommendationType, RiskLevel
)

_PRIORITY = {RiskLevel.CRITICAL:0, RiskLevel.HIGH:1, RiskLevel.MEDIUM:2, RiskLevel.LOW:3}


def _recs_for(gap: HealthcareGap, hospitals: List[HospitalRecord]) -> List[Recommendation]:
    out: List[Recommendation] = []
    ids = gap.affected_facilities

    if gap.gap_type == GapType.NO_EMERGENCY:
        out.append(Recommendation(RecommendationType.INVEST, gap.severity,
            f"Establish Emergency Services in {gap.region}",
            f"{len(ids)} facilities lack ER. Immediate investment required.",
            ids[:3], gap.region,
            "Emergency services are a baseline patient safety requirement.",
            "Could reduce emergency mortality by 20-40%."))
        out.append(Recommendation(RecommendationType.TRANSFER, RiskLevel.HIGH,
            f"Patient Transfer Protocol for {gap.region}",
            "Create formal transfer agreements with nearby ER-capable hospitals.",
            [], gap.region, "Interim measure until local ER is established."))

    elif gap.gap_type == GapType.NO_ICU:
        out.append(Recommendation(RecommendationType.UPGRADE, gap.severity,
            f"Upgrade ICU Capacity in {gap.region}",
            "Fund priority facilities to add critical care units.",
            ids[:2], gap.region,
            "ICU shortages lead to preventable deaths during surge events.",
            "Target: at least 1 ICU-capable hospital per 50,000 residents."))

    elif gap.gap_type == GapType.LOW_DOCTOR_DENSITY:
        out.append(Recommendation(RecommendationType.DEPLOY_STAFF, gap.severity,
            f"Deploy Physicians to {gap.region}",
            "Consider incentive programmes, loan forgiveness, or rotational deployments.",
            ids[:5], gap.region,
            "Adequate physician staffing is essential for quality outcomes.",
            "Bring average doctor count to 10+ per facility."))

    elif gap.gap_type == GapType.NO_SPECIALIST:
        out.append(Recommendation(RecommendationType.CLOSE_GAP, gap.severity,
            f"Fill Specialist Gap in {gap.region}",
            gap.description, [], gap.region,
            "Specialist absence forces patients into long-distance care.",
            "Establish at least one specialist centre per missing specialty."))

    elif gap.gap_type == GapType.LOW_QUALITY:
        out.append(Recommendation(RecommendationType.INVEST, gap.severity,
            f"Quality Improvement Programme for {gap.region}",
            "Target safety, infection control, and readmission-reduction programmes.",
            ids[:5], gap.region,
            "Below-average safety scores correlate with higher patient harm rates.",
            "Bring all facilities to national average within 24 months."))

    return out


def generate_recommendations(gap_result: GapAnalysisResult,
                              hospitals: List[HospitalRecord]) -> List[Recommendation]:
    recs: List[Recommendation] = []
    for gap in gap_result.gaps:
        recs.extend(_recs_for(gap, hospitals))

    seen: set = set()
    unique = []
    for r in recs:
        if r.title not in seen:
            seen.add(r.title)
            unique.append(r)

    unique.sort(key=lambda r: _PRIORITY.get(r.priority, 99))
    return unique
