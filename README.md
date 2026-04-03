# HealthIQ — Agentic AI Healthcare Intelligence Platform

> **Zero external dependencies for the backend.**  
> Runs with plain `python3 server.py` — no pip install required.

---

## Quick Start (2 commands)

```bash
# 1. Start the backend (first run ~30s to build data cache)
python3 server.py

# 2. Start the frontend (separate terminal)
cd frontend && npm install && npm start
```

- Backend: http://localhost:8000  
- Frontend: http://localhost:3000  
- API test: http://localhost:8000/health

---

## Project Structure

```
healthcare_platform/
├── server.py                      ← Start here. Pure stdlib HTTP server.
├── backend/
│   ├── models/
│   │   └── schemas.py             ← All dataclasses (HospitalRecord, etc.)
│   ├── services/
│   │   ├── data_loader.py         ← Merges 4 CSVs → HospitalRecord objects
│   │   ├── idp_service.py         ← Parse raw text → structured hospital
│   │   ├── validation_service.py  ← 8-rule clinical validation engine
│   │   ├── rag_service.py         ← TF-IDF search index
│   │   ├── gap_detection.py       ← Medical desert identification
│   │   └── recommendation_engine.py ← Actionable recommendations
│   └── agents/
│       └── healthcare_agent.py    ← 5-node reasoning agent
├── frontend/
│   └── src/
│       ├── App.js / App.css
│       ├── pages/
│       │   ├── Dashboard.js
│       │   ├── QueryPage.js       ← AI chat interface
│       │   ├── HospitalsPage.js   ← Browse + filter hospitals
│       │   ├── GapsPage.js        ← Gap analysis + recommendations
│       │   └── ParsePage.js       ← IDP text parser
│       └── utils/api.js
├── data/
│   ├── all_us_hospitals.csv       ← 5,335 hospitals
│   ├── all_us_doctors.csv         ← 536,723 doctors
│   ├── all_us_hospital_doctor_mapping.csv
│   └── all_us_department_summary.csv
└── tests/
    └── test_all.py                ← 55 tests, pure stdlib
```

---

## Data Sources

| File | Rows | Content |
|------|------|---------|
| all_us_hospitals.csv | 5,335 | Name, location, type, quality metrics |
| all_us_doctors.csv | 536,723 | Specialty, credentials, affiliation |
| all_us_hospital_doctor_mapping.csv | 536,723 | Doctor ↔ hospital links |
| all_us_department_summary.csv | 100,539 | Doctor counts per department |

---

## API Endpoints

All endpoints return JSON. CORS is open for development.

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "hospitals_loaded": 5335, "index_ready": true}
```

### GET /stats
```bash
curl http://localhost:8000/stats
```

### GET /hospitals
```bash
# All hospitals (paginated)
curl "http://localhost:8000/hospitals?page=1&per_page=20"

# Filter by state + emergency services
curl "http://localhost:8000/hospitals?state=TX&has_emergency=true"

# Filter by minimum rating
curl "http://localhost:8000/hospitals?state=CA&min_rating=4"
```

Query params: `state`, `city`, `has_emergency` (true/false), `min_rating` (1-5), `hospital_type`, `page`, `per_page`

### GET /hospitals/{id}
```bash
curl http://localhost:8000/hospitals/TX00001
```

### GET /gaps
```bash
# Gap analysis for a state
curl "http://localhost:8000/gaps?state=TX"

# Gap analysis for a city
curl "http://localhost:8000/gaps?state=CA&city=Los Angeles"
```

### POST /parse
Parse free-form hospital text into structured data.
```bash
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{
    "text": "St. Mary Hospital in Houston TX. Has ICU, emergency, and surgery. 30 doctors. Rating: 4.",
    "strict_mode": false
  }'
```

### POST /validate
```bash
# Validate by facility ID
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"facility_id": "TX00001"}'
```

### POST /query  ← Main agent endpoint
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the healthcare gaps in Texas?",
    "state_filter": "TX",
    "include_reasoning": true,
    "max_results": 15
  }'
```

Response includes:
- `answer` — human-readable response
- `reasoning_steps` — full agent trace (which data was used at each step)
- `gaps_identified` — structured gap list
- `recommendations` — prioritised action list
- `hospitals_referenced` — facility IDs used

---

## Run Tests

```bash
# No dependencies needed — pure stdlib
python3 tests/test_all.py
```

Expected output: `Ran 55 tests in ~1s — OK`

---

## Frontend Pages

| Page | URL | What it does |
|------|-----|-------------|
| Dashboard | / | Stats, charts, quick queries |
| AI Query | /query | Chat interface with reasoning trace |
| Hospitals | /hospitals | Browse + filter all 5,335 hospitals |
| Gap Analysis | /gaps | Select state → see gaps + recommendations |
| Parse Text | /parse | Paste hospital text → extract structured data |

---

## Agent Architecture (5-node graph)

```
User Query
    │
    ▼
[parse_query]        Extract intent, state, city, capability
    │
    ▼
[retrieve_data]      TF-IDF semantic search + location filters
    │
    ▼
[validate_data]      Run 8-rule clinical validation
    │
    ▼
[detect_gaps]        Check ER%, ICU%, doctor density, specialties
    │
    ▼
[generate_recommendation]   Map gaps → prioritised recommendations
    │
    ▼
AgentResponse (with full reasoning trace)
```

Every step records which hospital IDs were used → full traceability.

---

## Optional: FastAPI version (better performance)

If you want to use FastAPI instead of the stdlib server:

```bash
pip install fastapi uvicorn pydantic

# Then run:
uvicorn backend.main_fastapi:app --reload --port 8000
```

The stdlib server (`server.py`) and FastAPI server produce identical JSON responses.

---

## Optional: Upgrade RAG to real embeddings

Replace the TF-IDF index with FAISS + sentence-transformers:

```python
# In backend/services/rag_service.py
pip install faiss-cpu sentence-transformers

from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [record_to_text(h) for h in hospitals.values()]
embeddings = model.encode(texts).astype(np.float32)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
```

The public `.search()` API stays the same — no agent changes needed.

---

## Validation Rules

| Rule | Severity | Condition |
|------|----------|-----------|
| Emergency without ICU | HIGH | ER=True but ICU=False |
| Surgery without doctors | HIGH | Surgery=True but doctors=0 |
| Low rating + emergency | HIGH | ER=True and rating ≤ 2 |
| Multiple below average | HIGH | 2+ quality metrics below national |
| Missing city/state | MEDIUM | No location data |
| No capabilities + no doctors | MEDIUM | Likely incomplete record |
| Low confidence score | MEDIUM | Confidence < 50% |
| Missing phone | LOW | No contact info |

Risk levels: LOW → MEDIUM → HIGH → CRITICAL

---

## Gap Detection Thresholds

| Gap | Threshold | Severity |
|-----|-----------|----------|
| Emergency coverage | < 40% of hospitals have ER | HIGH/CRITICAL |
| ICU coverage | < 25% of hospitals have ICU | HIGH/CRITICAL |
| Doctor density | < 5 avg doctors per hospital | HIGH/CRITICAL |
| Safety quality | > 50% below national average | HIGH |
| Missing specialty | Cardiac/oncology/pediatrics absent | MEDIUM/HIGH |
