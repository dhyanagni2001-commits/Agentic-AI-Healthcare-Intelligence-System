"""
Data ingestion service.
Loads and merges the four CSV files into HospitalRecord objects.
Caches to JSON for fast restarts. Pure stdlib — no dependencies.
"""
from __future__ import annotations
import csv, json, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from backend.models.schemas import (
    HospitalRecord, HospitalCapabilities, HospitalQualityMetrics
)
from backend.ingestion.clean import CleaningReport, clean_str, clean_bool, clean_int
from backend.ingestion.schemas_pydantic import HospitalRecordIn, issues_from_validation_error
from pydantic import ValidationError

log = logging.getLogger(__name__)
DATA_DIR   = Path(__file__).parent.parent.parent / "data"
CACHE_FILE = DATA_DIR / "cache.json"

_REPORT = CleaningReport()


# ── Helpers ───────────────────────────────────────────────────────────────────
# Thin wrappers over backend.ingestion.clean so existing call sites in this
# module don't need to thread the CleaningReport through every call.

def _bool(v: str) -> bool:
    return clean_bool(v, "unspecified", _REPORT)

def _int(v: str, default=None) -> Optional[int]:
    return clean_int(v, "unspecified", _REPORT, default=default)

def _str(v: str) -> Optional[str]:
    return clean_str(v, "unspecified", _REPORT)


def _capabilities(row: Dict) -> HospitalCapabilities:
    t = str(row.get("Hospital Type", "")).lower()
    er = _bool(row.get("Emergency Services", "No"))
    return HospitalCapabilities(
        emergency_services = er,
        icu        = er or any(k in t for k in ("acute", "critical", "general")),
        surgery    = any(k in t for k in ("acute", "surgical", "general", "critical")),
        maternity  = any(k in t for k in ("maternity", "women", "obstetric", "children")),
        pediatrics = any(k in t for k in ("children", "pediatric")),
        rehabilitation = "rehabilitation" in t,
        mental_health  = any(k in t for k in ("psychiatric", "mental", "behavioral")),
        cardiac_care   = "cardiac" in t,
        oncology       = "cancer" in t,
        trauma_center  = "trauma" in t or (er and "acute" in t),
    )


def _quality(row: Dict) -> HospitalQualityMetrics:
    ehr_raw = _str(row.get("Meets criteria for meaningful use of EHRs", ""))
    return HospitalQualityMetrics(
        overall_rating               = _int(row.get("Hospital overall rating", "")),
        mortality_comparison         = _str(row.get("Mortality national comparison", "")),
        safety_comparison            = _str(row.get("Safety of care national comparison", "")),
        readmission_comparison       = _str(row.get("Readmission national comparison", "")),
        patient_experience_comparison= _str(row.get("Patient experience national comparison", "")),
        effectiveness_comparison     = _str(row.get("Effectiveness of care national comparison", "")),
        timeliness_comparison        = _str(row.get("Timeliness of care national comparison", "")),
        meets_ehr_criteria           = _bool(ehr_raw) if ehr_raw else None,
    )


def _confidence(r: HospitalRecord) -> float:
    score = sum([
        0.25 if r.facility_name else 0,
        0.15 if r.city  else 0,
        0.15 if r.state else 0,
        0.20 if any(vars(r.capabilities).values()) else 0,
        0.10 if r.doctor_count > 0 else 0,
        0.15,  # base
    ])
    if r.capabilities.emergency_services and r.doctor_count == 0:
        score -= 0.15
    return max(0.1, round(score, 2))


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_hospitals() -> Dict[str, HospitalRecord]:
    out: Dict[str, HospitalRecord] = {}
    issue_count = 0
    with open(DATA_DIR / "all_us_hospitals.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid  = _str(row.get("Facility ID", ""))
            name = _str(row.get("Facility Name", ""))
            if not fid or not name:
                continue

            quality = _quality(row)
            try:
                HospitalRecordIn(
                    facility_id=fid, facility_name=name,
                    city=_str(row.get("City")), state=_str(row.get("State")),
                    zip_code=_str(row.get("ZIP Code")),
                    hospital_type=_str(row.get("Hospital Type")),
                    ownership=_str(row.get("Hospital Ownership")),
                    overall_rating=quality.overall_rating,
                )
            except ValidationError as e:
                for issue in issues_from_validation_error(fid, e):
                    issue_count += 1
                    log.debug(f"Validation issue for {fid}: {issue.field} — {issue.issue}")

            out[fid] = HospitalRecord(
                facility_id   = fid,
                facility_name = name,
                address       = _str(row.get("Address")),
                city          = _str(row.get("City")),
                state         = _str(row.get("State")),
                zip_code      = _str(row.get("ZIP Code")),
                county        = _str(row.get("County Name")),
                phone         = _str(row.get("Phone Number")),
                hospital_type = _str(row.get("Hospital Type")),
                ownership     = _str(row.get("Hospital Ownership")),
                capabilities  = _capabilities(row),
                quality       = quality,
            )
    log.info(f"Loaded {len(out)} hospitals — {issue_count} Pydantic validation issues flagged")
    return out


def _load_dept_summary() -> Dict[str, Tuple[int, List[str]]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(DATA_DIR / "all_us_department_summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hosp = _str(row.get("affiliated_hospital_name", ""))
            dept = _str(row.get("department", ""))
            cnt  = _int(str(row.get("doctor_count", "0")), 0) or 0
            if hosp and dept:
                grouped[hosp][dept] = grouped[hosp].get(dept, 0) + cnt
    return {h: (sum(d.values()), list(d.keys())) for h, d in grouped.items()}


def _enrich(hospitals: Dict[str, HospitalRecord],
            summary: Dict[str, Tuple[int, List[str]]]) -> None:
    name_to_id = {r.facility_name: fid for fid, r in hospitals.items()}
    for hname, (total, depts) in summary.items():
        fid = name_to_id.get(hname)
        if not fid:
            continue
        h = hospitals[fid]
        h.doctor_count     = total
        h.department_count = len(depts)
        h.departments      = depts
        dl = [d.lower() for d in depts]
        c  = h.capabilities
        if any("icu" in d or "intensive" in d or "critical" in d for d in dl): c.icu = True
        if any("surg" in d for d in dl):                                        c.surgery = True
        if any("mater" in d or "obstet" in d or "labor" in d for d in dl):     c.maternity = True
        if any("pediatr" in d or "children" in d for d in dl):                 c.pediatrics = True
        if any("cardiac" in d or "cardio" in d for d in dl):                   c.cardiac_care = True
        if any("oncol" in d or "cancer" in d for d in dl):                     c.oncology = True
        if any("psych" in d or "mental" in d or "behav" in d for d in dl):     c.mental_health = True
        if any("rehab" in d or "physical med" in d for d in dl):               c.rehabilitation = True
        if any("trauma" in d or "emergency" in d for d in dl):                 c.trauma_center = True
    for h in hospitals.values():
        h.confidence_score = _confidence(h)


# ── Public API ────────────────────────────────────────────────────────────────

def load_all_hospitals(force_reload: bool = False) -> Dict[str, HospitalRecord]:
    if not force_reload and CACHE_FILE.exists():
        try:
            raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            hospitals = {k: HospitalRecord.from_dict(v) for k, v in raw.items()}
            log.info(f"Loaded {len(hospitals)} hospitals from cache")
            return hospitals
        except Exception as e:
            log.warning(f"Cache load failed ({e}), rebuilding…")

    hospitals = _load_hospitals()
    _enrich(hospitals, _load_dept_summary())
    _REPORT.log_summary(log)

    try:
        CACHE_FILE.write_text(
            json.dumps({k: v.to_dict() for k, v in hospitals.items()}, ensure_ascii=False),
            encoding="utf-8"
        )
        log.info("Cache written")
    except Exception as e:
        log.warning(f"Cache write failed: {e}")

    return hospitals
