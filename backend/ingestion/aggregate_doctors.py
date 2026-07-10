"""Aggregates the raw per-doctor CSV (536K+ rows) into per
(hospital, specialty) documents (~15-25K rows).

Embedding one vector per individual doctor row would mean encoding 536K
texts (~25-30 min on CPU) and ~824MB of float32 vectors — too large for a
1GB-RAM free-tier EC2 instance. Retrieval value for this system lives at
"does this hospital have cardiologists", not the individual-physician
level, so we aggregate and keep a handful of sample names for citation
traceability.
"""
from __future__ import annotations
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

from backend.ingestion.clean import CleaningReport, clean_str

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class DoctorAggregate:
    hospital_id: str
    hospital_name: str
    specialty: str
    doctor_count: int = 0
    sample_doctor_names: List[str] = field(default_factory=list)
    credentials: set = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "hospital_id": self.hospital_id,
            "hospital_name": self.hospital_name,
            "specialty": self.specialty,
            "doctor_count": self.doctor_count,
            "sample_doctor_names": self.sample_doctor_names,
            "credentials": sorted(self.credentials),
        }


def aggregate_doctors(report: CleaningReport,
                       csv_path: Path = None) -> Dict[str, DoctorAggregate]:
    """Stream the doctor CSV and group by (hospital_id, specialty).

    Returns a dict keyed by "{hospital_id}::{specialty}".
    """
    csv_path = csv_path or (DATA_DIR / "all_us_doctors.csv")
    groups: Dict[str, DoctorAggregate] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hid = clean_str(row.get("affiliated_hospital_id"), "affiliated_hospital_id", report)
            hname = clean_str(row.get("affiliated_hospital_name"), "affiliated_hospital_name", report)
            specialty = clean_str(row.get("specialty"), "specialty", report) \
                or clean_str(row.get("department"), "department", report)
            if not hid or not hname or not specialty:
                report.record("skipped_missing_key", "doctor_row")
                continue

            key = f"{hid}::{specialty}"
            agg = groups.get(key)
            if agg is None:
                agg = DoctorAggregate(hospital_id=hid, hospital_name=hname, specialty=specialty)
                groups[key] = agg

            agg.doctor_count += 1
            cred = clean_str(row.get("doctor_credential"), "doctor_credential", report)
            if cred:
                agg.credentials.add(cred)
            if len(agg.sample_doctor_names) < 5:
                first = clean_str(row.get("doctor_first_name"), "doctor_first_name", report) or ""
                last = clean_str(row.get("doctor_last_name"), "doctor_last_name", report) or ""
                name = f"{first} {last}".strip()
                if name:
                    agg.sample_doctor_names.append(name)

    return groups
