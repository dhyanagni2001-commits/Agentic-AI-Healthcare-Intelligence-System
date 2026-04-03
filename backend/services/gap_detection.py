"""Healthcare gap detection — stdlib only."""
from __future__ import annotations
from typing import Dict, List, Optional
from collections import defaultdict
from backend.models.schemas import (
    HospitalRecord, HealthcareGap, GapAnalysisResult, GapType, RiskLevel
)

_BELOW = lambda v: v is not None and "below" in str(v).lower()


def _er_gap(hospitals, region):
    total = len(hospitals); with_er = sum(1 for h in hospitals if h.capabilities.emergency_services)
    ratio = with_er / total
    if ratio < 0.40:
        return HealthcareGap(GapType.NO_EMERGENCY,
            f"Only {with_er}/{total} ({ratio:.0%}) hospitals have emergency services in {region}",
            [h.facility_id for h in hospitals if not h.capabilities.emergency_services],
            RiskLevel.CRITICAL if ratio < 0.2 else RiskLevel.HIGH, region)

def _icu_gap(hospitals, region):
    total = len(hospitals); with_icu = sum(1 for h in hospitals if h.capabilities.icu)
    ratio = with_icu / total
    if ratio < 0.25:
        return HealthcareGap(GapType.NO_ICU,
            f"Only {with_icu}/{total} ({ratio:.0%}) hospitals have ICU in {region}",
            [h.facility_id for h in hospitals if not h.capabilities.icu],
            RiskLevel.CRITICAL if ratio < 0.1 else RiskLevel.HIGH, region)

def _doc_gap(hospitals, region):
    avg = sum(h.doctor_count for h in hospitals) / len(hospitals)
    if avg < 5:
        return HealthcareGap(GapType.LOW_DOCTOR_DENSITY,
            f"Average doctor count is {avg:.1f} per hospital in {region}",
            [h.facility_id for h in hospitals if h.doctor_count < 5],
            RiskLevel.CRITICAL if avg == 0 else RiskLevel.HIGH, region,
            "Residents face long wait times and limited specialist access")

def _spec_gap(hospitals, region):
    has_c = any(h.capabilities.cardiac_care for h in hospitals)
    has_o = any(h.capabilities.oncology     for h in hospitals)
    has_p = any(h.capabilities.pediatrics   for h in hospitals)
    missing = [s for s, ok in [("cardiac care",has_c),("oncology",has_o),("pediatrics",has_p)] if not ok]
    if missing:
        return HealthcareGap(GapType.NO_SPECIALIST,
            f"No hospital in {region} offers: {', '.join(missing)}",
            [h.facility_id for h in hospitals],
            RiskLevel.HIGH if len(missing) >= 2 else RiskLevel.MEDIUM, region,
            f"Patients requiring {', '.join(missing)} must travel outside region")

def _qual_gap(hospitals, region):
    below = sum(1 for h in hospitals if _BELOW(h.quality.safety_comparison))
    if below / len(hospitals) > 0.5:
        return HealthcareGap(GapType.LOW_QUALITY,
            f"{below}/{len(hospitals)} hospitals below national average for safety in {region}",
            [h.facility_id for h in hospitals if _BELOW(h.quality.safety_comparison)],
            RiskLevel.HIGH, region)


def _overall(gaps):
    if not gaps: return RiskLevel.LOW
    sevs = [g.severity for g in gaps]
    if RiskLevel.CRITICAL in sevs: return RiskLevel.CRITICAL
    if sevs.count(RiskLevel.HIGH) >= 2: return RiskLevel.CRITICAL
    if RiskLevel.HIGH in sevs: return RiskLevel.HIGH
    if RiskLevel.MEDIUM in sevs: return RiskLevel.MEDIUM
    return RiskLevel.LOW


def analyse_region(hospitals: List[HospitalRecord],
                   state: Optional[str] = None,
                   city:  Optional[str] = None) -> GapAnalysisResult:
    region = ", ".join(p for p in [city, state] if p) or "selected region"
    if not hospitals:
        return GapAnalysisResult(state, city, [], RiskLevel.LOW, "No hospitals to analyse", 0)

    gaps = [g for g in [_er_gap(hospitals, region), _icu_gap(hospitals, region),
                         _doc_gap(hospitals, region), _spec_gap(hospitals, region),
                         _qual_gap(hospitals, region)] if g is not None]
    risk = _overall(gaps)
    return GapAnalysisResult(state, city, gaps, risk,
        f"{len(gaps)} gap(s) found in {region}, overall risk: {risk.upper()}",
        len(hospitals))


def analyse_by_state(all_h: List[HospitalRecord]) -> Dict[str, GapAnalysisResult]:
    by_state: Dict[str, List[HospitalRecord]] = defaultdict(list)
    for h in all_h:
        if h.state: by_state[h.state].append(h)
    return {s: analyse_region(hs, state=s) for s, hs in by_state.items()}


def get_medical_deserts(all_h: List[HospitalRecord], top_n: int = 10) -> List[GapAnalysisResult]:
    order = {RiskLevel.CRITICAL:4,RiskLevel.HIGH:3,RiskLevel.MEDIUM:2,RiskLevel.LOW:1}
    results = list(analyse_by_state(all_h).values())
    results.sort(key=lambda r: (-order[r.overall_risk], -len(r.gaps)))
    return results[:top_n]
