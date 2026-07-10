"""End-to-end ingestion entry point: loads hospitals, aggregates doctors,
and produces the full set of SearchableDocuments ready for embedding.

Run standalone: python3 -m backend.ingestion.pipeline
"""
from __future__ import annotations
import logging
from typing import Dict, List, Tuple

from backend.models.schemas import HospitalRecord
from backend.services.data_loader import load_all_hospitals
from backend.ingestion.clean import CleaningReport
from backend.ingestion.aggregate_doctors import aggregate_doctors
from backend.ingestion.documents import (
    SearchableDocument, hospital_document, doctor_aggregate_document,
    load_department_summary_documents,
)

log = logging.getLogger(__name__)


def build_all_documents() -> Tuple[List[SearchableDocument], Dict[str, HospitalRecord]]:
    report = CleaningReport()

    hospitals = load_all_hospitals()
    name_to_id = {h.facility_name: fid for fid, h in hospitals.items()}

    docs: List[SearchableDocument] = [hospital_document(h) for h in hospitals.values()]
    log.info(f"Built {len(docs)} hospital_profile documents")

    doctor_groups = aggregate_doctors(report)
    doctor_docs = [doctor_aggregate_document(agg) for agg in doctor_groups.values()]
    docs.extend(doctor_docs)
    log.info(f"Built {len(doctor_docs)} doctor_aggregate documents "
              f"(from {sum(a.doctor_count for a in doctor_groups.values())} raw doctor rows)")

    dept_docs = load_department_summary_documents(report, name_to_id)
    docs.extend(dept_docs)
    log.info(f"Built {len(dept_docs)} department_summary documents")

    report.log_summary(log)
    log.info(f"Total searchable documents: {len(docs)}")
    return docs, hospitals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_all_documents()
