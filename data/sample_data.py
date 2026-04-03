"""
Sample dataset demo — runs without the server.
python3 data/sample_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.idp_service import parse_hospital_text
from backend.services.validation_service import validate_hospital

SAMPLES = [
    ("Complete record",
     "St. Catherine Medical Center in Austin, TX 78701. Level I Trauma Center with emergency, ICU, surgery, maternity, pediatrics. 45 doctors, 80 nurses. Rating: 4. 250 beds."),
    ("Rural minimal",
     "Community clinic in Plainview, TX 79072. Basic emergency services. 5 doctors. Rating: 3."),
    ("Contradictory — surgery, no doctors",
     "Downtown Surgical Center, Chicago IL 60601. Full surgical suite. 0 doctors on staff. Emergency operational. Rating: 2."),
    ("ICU but no emergency",
     "Northside Critical Care, Denver CO 80202. Specialises in ICU and ventilator management. No emergency department. 12 specialists."),
    ("Missing location",
     "General Hospital. Has emergency room, surgery, ICU. 30 doctors. Rating 4 out of 5."),
    ("Very minimal",
     "Clinic in rural AR. 2 doctors."),
    ("High performing",
     "Massachusetts General Hospital, Boston MA 02114. Emergency, ICU, surgery, cardiac care, oncology, pediatrics, maternity. 500 doctors. Rating: 5."),
]

SAMPLE_QUERIES = [
    "What are the healthcare gaps in Texas?",
    "Find hospitals without emergency services in California",
    "Which states have the lowest doctor density?",
    "Show me high-risk hospitals in Florida",
    "Recommend where to deploy additional doctors",
]

if __name__ == "__main__":
    print("=" * 62)
    print("HEALTHIQ — SAMPLE DATASET DEMONSTRATION")
    print("=" * 62)

    for label, text in SAMPLES:
        print(f"\n{'─'*52}")
        print(f"  {label}")
        result = parse_hospital_text(text.strip())
        if result.success:
            h = result.hospital
            caps = [k for k, v in vars(h.capabilities).items() if v]
            vr = validate_hospital(h)
            print(f"  Name:       {h.facility_name}")
            print(f"  Location:   {h.city or '?'}, {h.state or '?'}")
            print(f"  Confidence: {h.confidence_score:.0%}")
            print(f"  Caps:       {', '.join(caps) or 'none'}")
            print(f"  Doctors:    {h.doctor_count}")
            print(f"  Risk Level: {vr.risk_level.upper()}")
            if vr.issues:
                for i in vr.issues:
                    if i.severity.value in ("high", "critical"):
                        print(f"    ⚠ [{i.severity.upper()}] {i.issue}")
        else:
            print(f"  Parse failed: {result.processing_notes}")

    print(f"\n{'='*62}")
    print("SAMPLE AI QUERIES (POST /query):")
    for q in SAMPLE_QUERIES:
        print(f"  • {q}")
    print()
