"""API tests for backend/main_fastapi.py.

Patches the lifespan's data/index loaders with a small synthetic index
(see tests/test_embedding_retrieval.py) so this runs in seconds instead
of the minutes/hours a full ~257K-document FAISS build would take.
"""
from __future__ import annotations
import sys, os, unittest
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from tests.test_embedding_retrieval import _build_synthetic_index


def _make_client() -> TestClient:
    idx = _build_synthetic_index()
    with patch("backend.services.data_loader.load_all_hospitals", return_value=idx.get_all()), \
         patch("backend.services.hybrid_index.build_index", return_value=idx):
        import backend.main_fastapi as m
        client = TestClient(m.app)
        client.__enter__()  # runs lifespan startup while the patches are active
    return client


class TestHealthAndStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _make_client()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["hospitals_loaded"], 3)
        self.assertTrue(body["index_ready"])

    def test_stats(self):
        r = self.client.get("/stats")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total_hospitals"], 3)

    def test_list_hospitals_filtered_by_state(self):
        r = self.client.get("/hospitals", params={"state": "TX"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["hospitals"][0]["facility_id"], "TX001")

    def test_get_hospital_by_id(self):
        r = self.client.get("/hospitals/CA001")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["facility_name"], "LA Children's Hospital")

    def test_get_hospital_404(self):
        r = self.client.get("/hospitals/NOPE")
        self.assertEqual(r.status_code, 404)

    def test_gaps_for_state(self):
        r = self.client.get("/gaps", params={"state": "TX"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"], "TX")

    def test_gaps_no_match_404(self):
        r = self.client.get("/gaps", params={"state": "ZZ"})
        self.assertEqual(r.status_code, 404)

    def test_parse_success(self):
        r = self.client.post("/parse", json={"text": "Houston Emergency Center in Houston TX with ICU"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["success"])

    def test_parse_text_too_short(self):
        r = self.client.post("/parse", json={"text": "a"})
        self.assertEqual(r.status_code, 400)

    def test_validate_by_facility_id(self):
        r = self.client.post("/validate", json={"facility_id": "TX001"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["facility_id"], "TX001")

    def test_validate_unknown_facility_id(self):
        r = self.client.post("/validate", json={"facility_id": "NOPE"})
        self.assertEqual(r.status_code, 404)

    def test_validate_requires_facility_id_or_hospital(self):
        r = self.client.post("/validate", json={})
        self.assertEqual(r.status_code, 400)

    def test_query_returns_relevant_hospital(self):
        r = self.client.post("/query", json={"query": "hospital with icu and emergency care"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("TX001", r.json()["hospitals_referenced"])

    def test_query_requires_nonempty_query(self):
        r = self.client.post("/query", json={"query": "   "})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
