# HealthIQ — Agentic AI Healthcare Intelligence Platform

A full-stack healthcare intelligence system built on a **5-agent LangGraph reasoning pipeline**, FAISS semantic search, RAG answer synthesis (Gemini 2.5 Flash), and a React dashboard — spanning **5,335 US hospitals** and **536,723 doctors**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (5-node StateGraph) |
| Semantic search | FAISS + sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM / RAG synthesis | Gemini 2.5 Flash (Google AI Studio) |
| Backend (production) | FastAPI + Pydantic v2 |
| Backend (lightweight) | Python stdlib only (no ML deps) |
| Frontend | React |
| Data | 4 CSV files — hospitals, doctors, departments, summaries |

---

## Agent Pipeline

Every query runs through a 5-node LangGraph graph. Nodes share one `AgentState` object; each node reads the state, does its work, and returns only the fields it updates.

```
Query
  │
  ▼
[1] Query Planner     — extracts intent, state/city filter, capability filter
  │
  ▼
[2] Retrieval         — FAISS semantic search (or TF-IDF fallback), location filter
  │
  ▼
[3] Validation        — flags missing data, low confidence, quality issues
  │
  ▼
[4] Gap Analysis      — computes ER/ICU coverage, doctor density, specialist gaps
  │
  ▼
[5] Recommendation    — maps each gap to prioritised staffing/investment actions
  │
  ▼
RAG Answer Synthesis  — LLM narrates the computed facts (Gemini 2.5 Flash)
```

The graph is compiled once at import time and re-invoked per request. The retriever (FAISS or TF-IDF) is injected at call time via `config["configurable"]`, so the same compiled graph handles both backends without rebuilding.

---

## Quick Start

### Production (FastAPI + FAISS + LLM)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY=your_key_here   # free at aistudio.google.com/apikey

uvicorn backend.main_fastapi:app --port 8000
cd frontend && npm install && npm start
```

First run builds the FAISS index (~3–5 min for 257K vectors). Subsequent restarts load from disk cache in seconds.

> **macOS note:** The `KMP_DUPLICATE_LIB_OK` and `OMP_NUM_THREADS` flags are set automatically in code to prevent a known libomp conflict between PyTorch and FAISS on macOS. No manual env var setup needed.

### Lightweight (stdlib only, no ML stack)

```bash
pip install pydantic
python3 server.py
cd frontend && npm install && npm start
```

Uses TF-IDF retrieval and returns deterministic template answers (no LLM key needed).

| URL | Description |
|---|---|
| `http://localhost:8000` | Backend API |
| `http://localhost:3000` | React frontend |
| `http://localhost:8000/docs` | FastAPI auto-docs (production only) |
| `http://localhost:8000/health` | Health check |

---

## Configuration

| Env var | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (production) | — | Google AI Studio key (free tier) |
| `GROK_API_KEY` | No | — | xAI Grok key (optional secondary LLM) |
| `LLM_PROVIDER` | No | `gemini` | `gemini` or `grok` |

---

## API Reference

All endpoints are available on both the production FastAPI server and the lightweight stdlib server.

### GET `/health`
Returns server status and whether the index is ready.

### GET `/stats`
Dataset-wide statistics: total hospitals, states covered, emergency services coverage, doctor totals, type distribution.

### GET `/hospitals`
Paginated list of hospitals with optional filters.

| Query param | Type | Example |
|---|---|---|
| `page` | int | `1` |
| `per_page` | int | `20` |
| `state` | string | `TX` |
| `city` | string | `Houston` |
| `has_emergency` | bool | `true` |
| `min_rating` | int | `3` |
| `hospital_type` | string | `Acute Care` |

### GET `/hospitals/{facility_id}`
Returns a single hospital record by CMS facility ID.

### GET `/gaps`
Runs gap analysis for a region.

| Query param | Type | Example |
|---|---|---|
| `state` | string | `TX` |
| `city` | string | `Houston` |

### POST `/query`
Main agentic endpoint. Runs the full 5-agent pipeline and returns an LLM-synthesised answer.

```json
{
  "query": "Which hospitals in Texas lack ICU?",
  "state_filter": "TX",
  "city_filter": null,
  "include_reasoning": true,
  "max_results": 10
}
```

Response includes: `answer`, `reasoning_steps` (one per agent), `gaps_identified`, `recommendations`, `retrieved_documents`, `confidence`.

### POST `/parse`
Parse free-text hospital descriptions into structured `HospitalRecord` objects using regex-based IDP.

```json
{ "text": "St. Mary Medical Center in Dallas, TX. Has ICU, emergency. 15 doctors.", "strict_mode": false }
```

### POST `/validate`
Validate a hospital record for clinical data quality issues (missing capabilities, low ratings, no doctors).

```json
{ "facility_id": "670055" }
```


---

## Project Structure

```
.
├── backend/
│   ├── agents/
│   │   └── healthcare_agent.py     # LangGraph 5-agent graph + RAG synthesis
│   ├── ingestion/
│   │   ├── aggregate_doctors.py    # streams 536K rows → per-(hospital,specialty) aggregates
│   │   ├── clean.py                # loggable CSV field cleaning (nullish, bool, int)
│   │   ├── documents.py            # builds SearchableDocument objects for embedding
│   │   ├── pipeline.py             # end-to-end ingestion entry point
│   │   └── schemas_pydantic.py     # Pydantic boundary validators for raw CSV rows
│   ├── models/
│   │   └── schemas.py              # stdlib dataclasses: HospitalRecord, AgentResponse, etc.
│   ├── prompts/
│   │   └── templates.py            # RAG prompt templates (kept separate for easy tuning)
│   ├── services/
│   │   ├── data_loader.py          # loads + merges 4 CSVs → HospitalRecord dict, JSON cache
│   │   ├── embedding_service.py    # singleton sentence-transformers model (all-MiniLM-L6-v2)
│   │   ├── gap_detection.py        # rule-based ER/ICU/doctor/specialist/quality gap checks
│   │   ├── hybrid_index.py         # VectorHospitalIndex: FAISS search + filter fallback
│   │   ├── idp_service.py          # regex-based intelligent document parsing
│   │   ├── legacy_tfidf_service.py # TF-IDF index (stdlib fallback + benchmark baseline)
│   │   ├── llm_service.py          # pluggable LLM providers: Gemini (default), Grok
│   │   ├── rag_pipeline.py         # explicit RAG: embed → retrieve → context → LLM
│   │   ├── recommendation_engine.py# maps detected gaps → prioritised recommendations
│   │   ├── validation_service.py   # clinical data quality validation rules
│   │   └── vector_store.py         # FAISS IndexFlatIP wrapper with metadata sidecar
│   └── main_fastapi.py             # FastAPI app: routes, Pydantic models, lifespan
├── frontend/                       # React dashboard
├── data/                           # CSV data files (gitignored — large files)
├── tests/
│   ├── test_all.py                 # 55 unit tests (stdlib only — no ML deps)
│   ├── test_embedding_retrieval.py # FAISS/embeddings integration tests
│   ├── test_main_fastapi.py        # FastAPI endpoint tests
│   └── test_rag_pipeline.py        # RAG pipeline tests
├── server.py                       # Lightweight stdlib HTTP server (no FastAPI)
└── requirements.txt
```

---

## Running Tests

```bash
# Core tests (no ML stack needed)
python3 tests/test_all.py

# Or with pytest
python3 -m pytest tests/test_all.py -v

# Full integration tests (requires FAISS + sentence-transformers)
python3 -m pytest tests/ -v
```

---

## Data Scale

| Dataset | Records |
|---|---|
| US Hospitals | 5,335 |
| Doctors | 536,723 |
| Departments | aggregated per hospital |
| Searchable documents (embedded) | ~250K vectors |

Data sourced from CMS (Centers for Medicare & Medicaid Services) public datasets.

---

## Retrieval Evaluation: FAISS vs TF-IDF Baseline

8 queries evaluated against verifiable ground truth (state + capability label, e.g. "cardiac care New York" → NY hospitals with `cardiac_care=True`). Measured **without** location filter to test pure retrieval quality.

| Metric | TF-IDF (baseline) | FAISS + embeddings | FAISS + cap-boost |
|---|---|---|---|
| Precision@5 | 0.42 | 0.90 | **0.95** |
| Precision@10 | 0.39 | 0.91 | **0.96** |

**Cap-boost** re-ranks FAISS candidates by applying a 1.3× score multiplier to hospitals that possess the queried capability (detected by the Query Planner agent). This is implemented in `VectorHospitalIndex.search()` via the `cap_filter` parameter and wired through the agent pipeline — it's live in the production query path, not just an eval trick.

Adding the location filter barely moves FAISS (P@10: 0.96 → 0.97), confirming the embedding does the real work — not the filter. TF-IDF is unchanged either way, meaning it finds the right state via keyword match but fails to rank by capability relevance.

The gap is clearest on semantic synonym queries:

| Query | TF-IDF P@10 | FAISS P@10 | FAISS + cap-boost P@10 |
|---|---|---|---|
| "cardiac care New York" | 0.00 | 0.70 | 0.70 |
| "oncology cancer hospital Illinois" | 0.00 | **1.00** | **1.00** |
| "pediatric hospital Florida" | 0.10 | 1.00 | **1.00** |
| "emergency hospital Texas" | 0.50 | 0.80 | **1.00** |

TF-IDF scores zero on "cardiac care" and "oncology cancer" because `care` and `cancer` are high-frequency tokens with near-zero IDF weight in a healthcare corpus. The embedding model captures semantic meaning regardless of exact token overlap. The TF-IDF index is kept as a zero-dependency fallback and retrieval benchmark.

---

## Validation Catching Real Data Issues

Running the validation agent against the full dataset surfaces concrete problems in the CMS data. Example — `TX00026`:

```
Query:    "Find surgical hospitals in Austin, TX"
Hospital: Texas Veterans Hospital — Austin, TX (Acute Care)
Risk:     CRITICAL

Issues flagged by the validation agent:
  [HIGH] doctor_count       — Surgery capability reported but zero doctors on record
  [HIGH] quality.overall_rating — Emergency-capable hospital has a 1/5 rating
  [HIGH] quality             — 3 quality metrics below national average
                               (mortality, safety, readmission)
```

This is a real record in the CMS dataset. The hospital reports surgical and emergency capability but has no staff on record and sits at the bottom of every quality metric. Without the validation step, it would appear as a top result for "surgical hospitals in Austin" with no warning. The validation agent flags it as CRITICAL and the recommendation engine generates a quality-improvement action for it.

Across the full dataset: **628 hospitals** report surgery capability with zero doctors, and **984** are emergency-capable with an overall rating of ≤ 2/5 — both caught automatically on every query.
