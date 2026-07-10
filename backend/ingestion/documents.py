"""Generates the four "searchable document" types that get embedded:
hospital profiles, doctor (specialty) aggregates, department summaries,
and gap reports.

A SearchableDocument is the seam between ingestion and the embedding layer:
{id, doc_type, text, metadata}. `text` is what gets embedded; `metadata`
is what the UI/agent need back (facility_id, scores, etc.) without
re-embedding or re-parsing text.
"""
from __future__ import annotations
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
from collections import defaultdict

from backend.models.schemas import HospitalRecord, GapAnalysisResult
from backend.ingestion.aggregate_doctors import DoctorAggregate
from backend.ingestion.clean import CleaningReport, clean_str, clean_int

DATA_DIR = Path(__file__).parent.parent.parent / "data"

DOC_TYPE_HOSPITAL = "hospital_profile"
DOC_TYPE_DOCTOR = "doctor_aggregate"
DOC_TYPE_DEPARTMENT = "department_summary"
DOC_TYPE_GAP = "gap_report"


@dataclass
class SearchableDocument:
    id: str
    doc_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hospital_document(h: HospitalRecord) -> SearchableDocument:
    caps = [k.replace("_", " ") for k, v in vars(h.capabilities).items() if v]
    text = " | ".join([
        h.facility_name,
        f"{h.city or ''} {h.state or ''}".strip(),
        h.hospital_type or "",
        f"emergency services {'yes' if h.capabilities.emergency_services else 'no'}",
        "capabilities: " + (", ".join(caps) if caps else "none recorded"),
        f"{h.doctor_count} doctors across {h.department_count} departments",
        "departments: " + ", ".join(h.departments[:15]),
        f"overall rating {h.quality.overall_rating or 'unrated'}/5",
    ]).strip()
    return SearchableDocument(
        id=f"hospital::{h.facility_id}",
        doc_type=DOC_TYPE_HOSPITAL,
        text=text.lower(),
        metadata={
            "facility_id": h.facility_id,
            "facility_name": h.facility_name,
            "city": h.city,
            "state": h.state,
        },
    )


def doctor_aggregate_document(agg: DoctorAggregate) -> SearchableDocument:
    names = ", ".join(agg.sample_doctor_names) or "no names on file"
    creds = ", ".join(sorted(agg.credentials)) or "unspecified"
    text = (
        f"{agg.hospital_name} — {agg.specialty}: {agg.doctor_count} physicians. "
        f"Credentials: {creds}. Sample physicians: {names}."
    ).lower()
    return SearchableDocument(
        id=f"doctor::{agg.hospital_id}::{agg.specialty}",
        doc_type=DOC_TYPE_DOCTOR,
        text=text,
        metadata={
            "facility_id": agg.hospital_id,
            "facility_name": agg.hospital_name,
            "specialty": agg.specialty,
            "doctor_count": agg.doctor_count,
            "sample_doctor_names": agg.sample_doctor_names,
        },
    )


def department_summary_document(hospital_id: str, hospital_name: str,
                                 department: str, doctor_count: int) -> SearchableDocument:
    text = (
        f"{hospital_name} — {department} department: {doctor_count} doctors assigned."
    ).lower()
    return SearchableDocument(
        id=f"department::{hospital_id}::{department}",
        doc_type=DOC_TYPE_DEPARTMENT,
        text=text,
        metadata={
            "facility_id": hospital_id,
            "facility_name": hospital_name,
            "department": department,
            "doctor_count": doctor_count,
        },
    )


def gap_report_document(gap_result: GapAnalysisResult) -> SearchableDocument:
    region = ", ".join(p for p in [gap_result.city, gap_result.state] if p) or "region"
    gap_lines = "; ".join(g.description for g in gap_result.gaps) or "no gaps detected"
    text = f"Gap analysis for {region}: {gap_result.summary}. Gaps: {gap_lines}.".lower()
    return SearchableDocument(
        id=f"gap::{gap_result.state or 'na'}::{gap_result.city or 'na'}",
        doc_type=DOC_TYPE_GAP,
        text=text,
        metadata={
            "state": gap_result.state,
            "city": gap_result.city,
            "overall_risk": gap_result.overall_risk.value,
        },
    )


def load_department_summary_documents(report: CleaningReport,
                                       name_to_id: Dict[str, str],
                                       csv_path: Path = None) -> List[SearchableDocument]:
    """Groups all_us_department_summary.csv by (hospital, department) and
    emits one document per pair, resolving hospital name -> facility_id
    via the lookup built during hospital ingestion."""
    csv_path = csv_path or (DATA_DIR / "all_us_department_summary.csv")
    grouped: Dict[tuple, int] = defaultdict(int)

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hname = clean_str(row.get("affiliated_hospital_name"), "affiliated_hospital_name", report)
            dept = clean_str(row.get("department"), "department", report)
            cnt = clean_int(row.get("doctor_count", "0"), "doctor_count", report, default=0) or 0
            if not hname or not dept:
                report.record("skipped_missing_key", "department_row")
                continue
            grouped[(hname, dept)] += cnt

    docs = []
    for (hname, dept), cnt in grouped.items():
        hid = name_to_id.get(hname)
        if not hid:
            report.record("unmatched_hospital_name", "department_summary")
            continue
        docs.append(department_summary_document(hid, hname, dept, cnt))
    return docs
