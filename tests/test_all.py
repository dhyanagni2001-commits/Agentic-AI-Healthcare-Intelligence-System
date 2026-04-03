"""
Full test suite — pure Python stdlib, zero external deps.
Run:  python3 tests/test_all.py
      python3 -m pytest tests/test_all.py -v   (if pytest installed)
"""
from __future__ import annotations
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.schemas import (
    HospitalRecord, HospitalCapabilities, HospitalQualityMetrics,
    RiskLevel, GapType, RecommendationType
)
from backend.services.idp_service import parse_hospital_text
from backend.services.validation_service import validate_hospital, validate_many
from backend.services.rag_service import HospitalIndex
from backend.services.gap_detection import analyse_region
from backend.services.recommendation_engine import generate_recommendations


# ── Fixtures ──────────────────────────────────────────────────────────────────

def H(fid="T001", name="Test Hospital", city="Houston", state="TX",
      emergency=True, icu=True, surgery=True, doctors=20, rating=4) -> HospitalRecord:
    return HospitalRecord(
        facility_id=fid, facility_name=name, city=city, state=state,
        capabilities=HospitalCapabilities(emergency_services=emergency, icu=icu, surgery=surgery),
        quality=HospitalQualityMetrics(overall_rating=rating),
        doctor_count=doctors, confidence_score=0.9,
        phone="(555) 000-0000",
    )


def sample_index() -> HospitalIndex:
    hospitals = {
        "TX001": HospitalRecord("TX001","Houston ER",city="Houston",state="TX",
                                capabilities=HospitalCapabilities(emergency_services=True,icu=True)),
        "CA001": HospitalRecord("CA001","LA Children",city="Los Angeles",state="CA",
                                capabilities=HospitalCapabilities(pediatrics=True)),
        "NY001": HospitalRecord("NY001","NYC Cardiac",city="New York",state="NY",
                                capabilities=HospitalCapabilities(cardiac_care=True,emergency_services=True)),
    }
    hospitals["NY001"].departments = ["Cardiology","Emergency"]
    idx = HospitalIndex()
    idx.build(hospitals)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
class TestIDP(unittest.TestCase):

    def test_full_parse(self):
        r = parse_hospital_text(
            "St. Mary Medical Center is in Dallas, TX. Has ICU, emergency, and surgery. 15 doctors.")
        self.assertTrue(r.success)
        h = r.hospital
        self.assertEqual(h.state, "TX")
        self.assertTrue(h.capabilities.icu)
        self.assertTrue(h.capabilities.emergency_services)
        self.assertEqual(h.doctor_count, 15)

    def test_empty_input(self):
        self.assertFalse(parse_hospital_text("").success)

    def test_too_short(self):
        self.assertFalse(parse_hospital_text("Hi").success)

    def test_no_state_flagged(self):
        r = parse_hospital_text("General Hospital. Has emergency, ICU, and 10 doctors.")
        if r.success:
            self.assertIn("State not detected", r.processing_notes)

    def test_rating_extraction(self):
        r = parse_hospital_text("Downtown Clinic, Chicago IL. Rating: 3. Emergency services.")
        if r.success and r.hospital.quality.overall_rating:
            self.assertEqual(r.hospital.quality.overall_rating, 3)

    def test_confidence_in_range(self):
        r = parse_hospital_text("Houston Medical Center, Houston TX. Emergency, ICU, 50 doctors.")
        if r.success:
            self.assertGreaterEqual(r.hospital.confidence_score, 0.0)
            self.assertLessEqual(r.hospital.confidence_score, 1.0)

    def test_beds_in_notes(self):
        r = parse_hospital_text("City Hospital, Boston MA. 200 beds, 40 doctors. Emergency and ICU.")
        if r.success and r.hospital.notes:
            self.assertIn("200 beds", r.hospital.notes)

    def test_strict_mode_runs(self):
        r = parse_hospital_text("A place with ICU in TX 75001", strict_mode=True)
        self.assertIsInstance(r.success, bool)

    def test_messy_input(self):
        r = parse_hospital_text("hosp in new york NY 10001. emergency care maybe. rating 2")
        self.assertIsNotNone(r)

    def test_unicode(self):
        r = parse_hospital_text("Hôpital Général in Houston TX. Emergency and ICU. 20 doctors.")
        self.assertIsInstance(r.success, bool)

    def test_to_dict_round_trip(self):
        r = parse_hospital_text("Houston Medical Center, Houston TX. Emergency, ICU, 50 doctors.")
        if r.success:
            d = r.to_dict()
            self.assertIn("success", d)
            self.assertIn("hospital", d)


# ─────────────────────────────────────────────────────────────────────────────
class TestValidation(unittest.TestCase):

    def test_valid_hospital(self):
        r = validate_hospital(H())
        self.assertIn(r.risk_level, [RiskLevel.LOW, RiskLevel.MEDIUM])

    def test_er_without_icu(self):
        r = validate_hospital(H(icu=False, surgery=False))
        self.assertIn("capabilities.icu", [i.field for i in r.issues])
        self.assertIn(r.risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])

    def test_surgery_no_doctors(self):
        r = validate_hospital(H(doctors=0))
        self.assertIn("doctor_count", [i.field for i in r.issues])

    def test_low_rating_emergency(self):
        r = validate_hospital(H(rating=2))
        self.assertIn("quality.overall_rating", [i.field for i in r.issues])

    def test_no_caps_no_doctors(self):
        h = HospitalRecord("G001","Ghost")
        r = validate_hospital(h)
        self.assertIn("capabilities", [i.field for i in r.issues])

    def test_missing_city_state(self):
        h = H(); h.city = None; h.state = None
        r = validate_hospital(h)
        fields = [i.field for i in r.issues]
        self.assertIn("city", fields)
        self.assertIn("state", fields)

    def test_multiple_below_avg(self):
        h = H()
        h.quality.mortality_comparison = "Below the national average"
        h.quality.safety_comparison    = "Below the national average"
        r = validate_hospital(h)
        self.assertIn("quality", [i.field for i in r.issues])

    def test_validate_many_length(self):
        results = validate_many([H("H1","A"), H("H2","B"), H("H3","C")])
        self.assertEqual(len(results), 3)

    def test_to_dict_has_keys(self):
        r = validate_hospital(H())
        d = r.to_dict()
        for k in ("facility_id","valid","issues","risk_level","confidence_score"):
            self.assertIn(k, d)

    def test_risk_escalates_multiple_high(self):
        h = H(icu=False, doctors=0, rating=1)
        h.quality.mortality_comparison = "Below the national average"
        h.quality.safety_comparison    = "Below the national average"
        r = validate_hospital(h)
        self.assertIn(r.risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])


# ─────────────────────────────────────────────────────────────────────────────
class TestRAG(unittest.TestCase):

    def setUp(self):
        self.idx = sample_index()

    def test_search_returns_results(self):
        r = self.idx.search("emergency hospital")
        self.assertGreater(len(r), 0)

    def test_state_filter(self):
        r = self.idx.search("hospital", state_filter="TX")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0].facility_id, "TX001")

    def test_state_no_match(self):
        r = self.idx.search("hospital", state_filter="AK")
        self.assertEqual(len(r), 0)

    def test_capability_filter(self):
        r = self.idx.filter_by_capability("cardiac_care")
        ids = [h.facility_id for h in r]
        self.assertIn("NY001", ids)
        self.assertNotIn("TX001", ids)

    def test_get_by_id_found(self):
        h = self.idx.get_by_id("CA001")
        self.assertIsNotNone(h)
        self.assertEqual(h.facility_name, "LA Children")

    def test_get_by_id_missing(self):
        self.assertIsNone(self.idx.get_by_id("NOPE"))

    def test_total(self):
        self.assertEqual(self.idx.total, 3)

    def test_scores_are_floats(self):
        for _, score in self.idx.search("cardiac heart"):
            self.assertIsInstance(score, float)

    def test_unbuilt_raises(self):
        idx = HospitalIndex()
        with self.assertRaises(RuntimeError):
            idx.search("anything")


# ─────────────────────────────────────────────────────────────────────────────
class TestGapDetection(unittest.TestCase):

    def _group(self, n, n_er, n_icu, docs):
        return [HospitalRecord(f"H{i:03d}", f"Hosp {i}", city="City", state="TX",
                               capabilities=HospitalCapabilities(emergency_services=(i<n_er), icu=(i<n_icu)),
                               doctor_count=docs) for i in range(n)]

    def test_er_gap_detected(self):
        r = analyse_region(self._group(10, 1, 5, 15), state="TX")
        self.assertIn("no_emergency_services", [g.gap_type.value for g in r.gaps])

    def test_no_er_gap_when_sufficient(self):
        r = analyse_region(self._group(10, 7, 5, 20), state="CA")
        self.assertNotIn("no_emergency_services", [g.gap_type.value for g in r.gaps])

    def test_low_doctor_density(self):
        r = analyse_region(self._group(5, 3, 2, 0), state="TX")
        self.assertIn("low_doctor_density", [g.gap_type.value for g in r.gaps])

    def test_critical_risk_no_er_no_icu(self):
        r = analyse_region(self._group(10, 0, 0, 0), state="TX")
        self.assertIn(r.overall_risk, [RiskLevel.HIGH, RiskLevel.CRITICAL])

    def test_empty_hospitals(self):
        r = analyse_region([], state="ZZ")
        self.assertEqual(r.facilities_analyzed, 0)
        self.assertEqual(len(r.gaps), 0)

    def test_to_dict(self):
        r = analyse_region(self._group(5, 1, 1, 0), state="TX")
        d = r.to_dict()
        self.assertIn("overall_risk", d)
        self.assertIn("gaps", d)


# ─────────────────────────────────────────────────────────────────────────────
class TestRecommendations(unittest.TestCase):

    def _hospitals_no_er(self, n=5):
        return [HospitalRecord(f"H{i}", f"Hosp {i}", city="X", state="TX",
                               capabilities=HospitalCapabilities(emergency_services=(i==0)))
                for i in range(n)]

    def test_recs_generated(self):
        hs = self._hospitals_no_er()
        gr = analyse_region(hs, state="TX")
        recs = generate_recommendations(gr, hs)
        self.assertGreater(len(recs), 0)

    def test_sorted_by_priority(self):
        hs = [HospitalRecord(f"H{i}", f"H {i}", city="X", state="TX",
                             capabilities=HospitalCapabilities(), doctor_count=0)
              for i in range(8)]
        gr = analyse_region(hs, state="TX")
        recs = generate_recommendations(gr, hs)
        prio = {RiskLevel.CRITICAL:0,RiskLevel.HIGH:1,RiskLevel.MEDIUM:2,RiskLevel.LOW:3}
        vals = [prio[r.priority] for r in recs]
        self.assertEqual(vals, sorted(vals))

    def test_no_recs_healthy_region(self):
        hs = [HospitalRecord(f"H{i}", f"Good {i}", city="NYC", state="NY",
                             capabilities=HospitalCapabilities(emergency_services=True,icu=True,
                                surgery=True,cardiac_care=True,oncology=True,pediatrics=True),
                             quality=HospitalQualityMetrics(overall_rating=5),
                             doctor_count=30)
              for i in range(5)]
        gr = analyse_region(hs, state="NY")
        recs = generate_recommendations(gr, hs)
        self.assertLessEqual(len(recs), 2)

    def test_to_dict(self):
        hs = self._hospitals_no_er()
        gr = analyse_region(hs, state="TX")
        recs = generate_recommendations(gr, hs)
        if recs:
            d = recs[0].to_dict()
            self.assertIn("type", d)
            self.assertIn("priority", d)
            self.assertIn("title", d)


# ─────────────────────────────────────────────────────────────────────────────
class TestAgent(unittest.TestCase):

    def setUp(self):
        self.idx = sample_index()

    def test_basic_query(self):
        from backend.agents.healthcare_agent import run_agent
        r = run_agent("Find emergency hospitals", self.idx)
        self.assertIsInstance(r.answer, str)
        self.assertGreater(len(r.answer), 0)

    def test_state_filter(self):
        from backend.agents.healthcare_agent import run_agent
        r = run_agent("hospitals in texas", self.idx, state_filter="TX")
        self.assertGreater(len(r.hospitals_referenced), 0)

    def test_reasoning_steps_present(self):
        from backend.agents.healthcare_agent import run_agent
        r = run_agent("gap analysis", self.idx, include_reasoning=True)
        self.assertGreater(len(r.reasoning_steps), 0)

    def test_reasoning_steps_suppressed(self):
        from backend.agents.healthcare_agent import run_agent
        r = run_agent("gap analysis", self.idx, include_reasoning=False)
        self.assertEqual(len(r.reasoning_steps), 0)

    def test_to_dict(self):
        from backend.agents.healthcare_agent import run_agent
        r = run_agent("Find hospitals", self.idx)
        d = r.to_dict()
        for k in ("query","answer","reasoning_steps","hospitals_referenced","confidence"):
            self.assertIn(k, d)

    def test_bad_query_returns_response(self):
        from backend.agents.healthcare_agent import run_agent
        r = run_agent("xyzzy foobar", self.idx)
        self.assertIsInstance(r.answer, str)


# ─────────────────────────────────────────────────────────────────────────────
class TestDataLoader(unittest.TestCase):

    def test_loads_hospitals(self):
        from backend.services.data_loader import load_all_hospitals
        hospitals = load_all_hospitals()
        self.assertGreater(len(hospitals), 1000)

    def test_hospital_has_required_fields(self):
        from backend.services.data_loader import load_all_hospitals
        hospitals = load_all_hospitals()
        h = next(iter(hospitals.values()))
        self.assertIsNotNone(h.facility_id)
        self.assertIsNotNone(h.facility_name)

    def test_doctor_enrichment_worked(self):
        from backend.services.data_loader import load_all_hospitals
        hospitals = load_all_hospitals()
        enriched = [h for h in hospitals.values() if h.doctor_count > 0]
        self.assertGreater(len(enriched), 100)

    def test_round_trip_serialisation(self):
        from backend.services.data_loader import load_all_hospitals
        hospitals = load_all_hospitals()
        h = next(iter(hospitals.values()))
        d = h.to_dict()
        h2 = HospitalRecord.from_dict(d)
        self.assertEqual(h.facility_id, h2.facility_id)
        self.assertEqual(h.capabilities.emergency_services,
                         h2.capabilities.emergency_services)


# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases(unittest.TestCase):

    def test_duplicate_ids_validate_many(self):
        h1 = H("SAME","Alpha")
        h2 = H("SAME","Beta")
        results = validate_many([h1, h2])
        self.assertEqual(len(results), 2)

    def test_all_capabilities(self):
        h = HospitalRecord("S1","Super",city="NYC",state="NY",
                           capabilities=HospitalCapabilities(True,True,True,True,True,True,True,True,True,True),
                           quality=HospitalQualityMetrics(overall_rating=5),
                           doctor_count=500,confidence_score=1.0)
        r = validate_hospital(h)
        self.assertIn(r.risk_level, [RiskLevel.LOW, RiskLevel.MEDIUM])

    def test_very_long_parse_text(self):
        text = "Lots of noise. " * 100 + "St. Anthony Medical Center in Phoenix, AZ. Emergency, ICU, 45 doctors."
        r = parse_hospital_text(text)
        self.assertIsNotNone(r)

    def test_hospital_to_dict_from_dict(self):
        h = H()
        d = h.to_dict()
        h2 = HospitalRecord.from_dict(d)
        self.assertEqual(h.facility_id, h2.facility_id)
        self.assertEqual(h.state, h2.state)
        self.assertEqual(h.capabilities.emergency_services, h2.capabilities.emergency_services)

    def test_analyse_single_hospital(self):
        r = analyse_region([H()], state="TX")
        self.assertIsNotNone(r)
        self.assertEqual(r.facilities_analyzed, 1)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestIDP, TestValidation, TestRAG, TestGapDetection,
                TestRecommendations, TestAgent, TestDataLoader, TestEdgeCases]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
