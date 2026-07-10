"""Smoke tests for the FAISS + sentence-transformers retrieval layer.

Requires the venv deps (sentence-transformers, faiss-cpu) — unlike
test_all.py, this is NOT zero-dependency. Builds a small synthetic index
directly (bypassing the full CSV ingestion pipeline) so it runs in
seconds rather than minutes.
"""
from __future__ import annotations
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.schemas import HospitalRecord, HospitalCapabilities
from backend.ingestion.documents import SearchableDocument
from backend.services.vector_store import FaissVectorStore
from backend.services.hybrid_index import VectorHospitalIndex


def _hospital(fid, name, city, state, **caps) -> HospitalRecord:
    return HospitalRecord(facility_id=fid, facility_name=name, city=city, state=state,
                           capabilities=HospitalCapabilities(**caps))


def _doc(fid, text) -> SearchableDocument:
    return SearchableDocument(id=f"hospital::{fid}", doc_type="hospital_profile",
                               text=text, metadata={"facility_id": fid})


def _build_synthetic_index() -> VectorHospitalIndex:
    hospitals = {
        "TX001": _hospital("TX001", "Houston Emergency Center", "Houston", "TX",
                            emergency_services=True, icu=True),
        "CA001": _hospital("CA001", "LA Children's Hospital", "Los Angeles", "CA",
                            pediatrics=True),
        "NY001": _hospital("NY001", "NYC Cardiac Institute", "New York", "NY",
                            cardiac_care=True, emergency_services=True),
    }
    docs = [
        _doc("TX001", "houston emergency center has icu and emergency services in houston tx"),
        _doc("CA001", "la children's hospital offers pediatrics in los angeles ca"),
        _doc("NY001", "nyc cardiac institute offers cardiac care and emergency services in new york ny"),
    ]
    idx = VectorHospitalIndex()
    idx._records = hospitals
    idx._store = FaissVectorStore()
    idx._store.build(docs)
    idx._built = True
    return idx


class TestVectorHospitalIndex(unittest.TestCase):

    def setUp(self):
        self.idx = _build_synthetic_index()

    def test_search_returns_relevant_top_result(self):
        results = self.idx.search("hospital with icu and emergency care", top_k=3)
        self.assertGreater(len(results), 0)
        top_hospital, score = results[0]
        self.assertEqual(top_hospital.facility_id, "TX001")
        self.assertIsInstance(score, float)

    def test_state_filter(self):
        results = self.idx.search("hospital", top_k=5, state_filter="CA")
        self.assertTrue(all(h.state == "CA" for h, _ in results))

    def test_city_filter(self):
        results = self.idx.search("hospital", top_k=5, city_filter="New York")
        self.assertTrue(all("new york" in (h.city or "").lower() for h, _ in results))

    def test_filter_by_state_legacy_signature(self):
        self.assertEqual(len(self.idx.filter_by_state("TX")), 1)

    def test_filter_by_capability_legacy_signature(self):
        caps = self.idx.filter_by_capability("emergency_services")
        ids = {h.facility_id for h in caps}
        self.assertEqual(ids, {"TX001", "NY001"})

    def test_get_by_id(self):
        self.assertEqual(self.idx.get_by_id("CA001").facility_name, "LA Children's Hospital")
        self.assertIsNone(self.idx.get_by_id("NOPE"))

    def test_unbuilt_raises(self):
        fresh = VectorHospitalIndex()
        with self.assertRaises(RuntimeError):
            fresh.search("anything")


if __name__ == "__main__":
    unittest.main(verbosity=2)
