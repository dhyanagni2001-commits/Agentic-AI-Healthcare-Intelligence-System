"""
Healthcare Intelligence Platform — FastAPI backend.

Same endpoints as server.py (the stdlib fallback), rebuilt with Pydantic
request/response models and async-aware routing. Uses the production
retriever (FAISS + sentence-transformers, backend.services.hybrid_index)
instead of the legacy TF-IDF index.

Run with:
    source .venv/bin/activate
    uvicorn backend.main_fastapi:app --reload --port 8000
"""
from __future__ import annotations
import os
# Must be set before torch or faiss load their OpenMP runtime (macOS libomp conflict)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import logging
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.schemas import HospitalRecord  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main_fastapi")

_hospitals: Dict[str, HospitalRecord] = {}
_index: Any = None


# ── Request logging middleware ────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        log.info(f"{request.url.path} {response.status_code} {elapsed_ms:.1f}ms")
        return response


# ── Startup/shutdown ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hospitals, _index
    from backend.services.embedding_service import _get_model
    from backend.services.data_loader import load_all_hospitals
    from backend.services.hybrid_index import build_index
    # Load torch + SentenceTransformer before faiss to avoid libomp conflict on macOS
    _get_model()
    log.info("Loading hospital data…")
    _hospitals = load_all_hospitals()
    log.info(f"Loaded {len(_hospitals)} hospitals — building FAISS index…")
    _index = build_index(_hospitals)
    log.info("Ready.")
    yield


app = FastAPI(title="HealthIQ", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)


# ── Pydantic models (mirror backend/models/schemas.py dataclasses) ───────────

class HospitalCapabilitiesModel(BaseModel):
    emergency_services: bool = False
    icu: bool = False
    surgery: bool = False
    maternity: bool = False
    pediatrics: bool = False
    rehabilitation: bool = False
    mental_health: bool = False
    cardiac_care: bool = False
    oncology: bool = False
    trauma_center: bool = False


class HospitalQualityMetricsModel(BaseModel):
    overall_rating: Optional[int] = None
    mortality_comparison: Optional[str] = None
    safety_comparison: Optional[str] = None
    readmission_comparison: Optional[str] = None
    patient_experience_comparison: Optional[str] = None
    effectiveness_comparison: Optional[str] = None
    timeliness_comparison: Optional[str] = None
    meets_ehr_criteria: Optional[bool] = None


class HospitalRecordModel(BaseModel):
    facility_id: str
    facility_name: str
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    county: Optional[str] = None
    phone: Optional[str] = None
    hospital_type: Optional[str] = None
    ownership: Optional[str] = None
    capabilities: HospitalCapabilitiesModel = Field(default_factory=HospitalCapabilitiesModel)
    quality: HospitalQualityMetricsModel = Field(default_factory=HospitalQualityMetricsModel)
    doctor_count: int = 0
    department_count: int = 0
    departments: List[str] = Field(default_factory=list)
    confidence_score: float = 1.0
    notes: Optional[str] = None


class ValidationIssueModel(BaseModel):
    field: str
    issue: str
    severity: str


class ValidationResultModel(BaseModel):
    facility_id: str
    valid: bool
    issues: List[ValidationIssueModel]
    risk_level: str
    confidence_score: float
    warnings: List[str] = Field(default_factory=list)


class HealthcareGapModel(BaseModel):
    gap_type: str
    description: str
    affected_facilities: List[str]
    severity: str
    region: Optional[str] = None
    population_impact: Optional[str] = None


class GapAnalysisResultModel(BaseModel):
    state: Optional[str] = None
    city: Optional[str] = None
    gaps: List[HealthcareGapModel]
    overall_risk: str
    summary: str
    facilities_analyzed: int


class RecommendationModel(BaseModel):
    type: str
    priority: str
    title: str
    description: str
    target_facility_ids: List[str] = Field(default_factory=list)
    target_region: Optional[str] = None
    rationale: str = ""
    estimated_impact: Optional[str] = None


class ReasoningStepModel(BaseModel):
    step_name: str
    description: str
    data_used: List[str]
    output_summary: str


class AgentResponseModel(BaseModel):
    query: str
    answer: str
    reasoning_steps: List[ReasoningStepModel] = Field(default_factory=list)
    hospitals_referenced: List[str] = Field(default_factory=list)
    gaps_identified: List[HealthcareGapModel] = Field(default_factory=list)
    recommendations: List[RecommendationModel] = Field(default_factory=list)
    confidence: float = 0.8
    data_sources: List[str] = Field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list)


class ParseResponseModel(BaseModel):
    success: bool
    raw_input: str
    hospital: Optional[HospitalRecordModel] = None
    processing_notes: List[str] = Field(default_factory=list)
    validation: Optional[ValidationResultModel] = None


class HealthResponse(BaseModel):
    status: str
    hospitals_loaded: int
    index_ready: bool


class StatsResponse(BaseModel):
    total_hospitals: int
    states_covered: int
    hospitals_with_emergency: int
    hospitals_without_emergency: int
    average_rating: Optional[float] = None
    high_risk_hospitals: int
    total_doctors: int
    total_departments: int
    top_states_by_hospital_count: Dict[str, int]
    hospital_type_distribution: Dict[str, int]


class HospitalListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    hospitals: List[HospitalRecordModel]


class ParseRequest(BaseModel):
    text: str
    strict_mode: bool = False


class ValidateRequest(BaseModel):
    facility_id: Optional[str] = None
    hospital: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    query: str
    state_filter: Optional[str] = None
    city_filter: Optional[str] = None
    include_reasoning: bool = True
    max_results: int = 10


# ── Routes ─────────────────────────────────────────────────────────────────────
# Defined as plain `def` (not `async def`): the underlying services
# (FAISS search, sentence-transformers encode, LLM HTTP calls) are
# synchronous, so FastAPI dispatches these to its worker threadpool
# instead of blocking the event loop.

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", hospitals_loaded=len(_hospitals),
                           index_ready=_index is not None and _index.total > 0)


@app.get("/stats", response_model=StatsResponse)
def stats():
    items = list(_hospitals.values())
    total = len(items)
    if not total:
        raise HTTPException(503, "No data loaded")
    ratings = [h.quality.overall_rating for h in items if h.quality.overall_rating]
    avg = round(sum(ratings) / len(ratings), 2) if ratings else None

    def is_high_risk(h):
        metrics = [h.quality.mortality_comparison, h.quality.safety_comparison,
                   h.quality.readmission_comparison, h.quality.effectiveness_comparison]
        return sum(1 for m in metrics if m and "below" in m.lower()) >= 2

    state_c = Counter(h.state for h in items if h.state)
    type_c = Counter(h.hospital_type for h in items if h.hospital_type)
    return StatsResponse(
        total_hospitals=total,
        states_covered=len(state_c),
        hospitals_with_emergency=sum(1 for h in items if h.capabilities.emergency_services),
        hospitals_without_emergency=sum(1 for h in items if not h.capabilities.emergency_services),
        average_rating=avg,
        high_risk_hospitals=sum(1 for h in items if is_high_risk(h)),
        total_doctors=sum(h.doctor_count for h in items),
        total_departments=sum(h.department_count for h in items),
        top_states_by_hospital_count=dict(state_c.most_common(10)),
        hospital_type_distribution=dict(type_c.most_common(15)),
    )


@app.get("/hospitals", response_model=HospitalListResponse)
def list_hospitals(page: int = 1, per_page: int = 20, state: Optional[str] = None,
                    city: Optional[str] = None, has_emergency: Optional[bool] = None,
                    min_rating: Optional[int] = None, hospital_type: Optional[str] = None):
    items = list(_hospitals.values())
    if state:
        items = [h for h in items if (h.state or "").upper() == state.upper()]
    if city:
        items = [h for h in items if city.lower() in (h.city or "").lower()]
    if has_emergency is True:
        items = [h for h in items if h.capabilities.emergency_services]
    if has_emergency is False:
        items = [h for h in items if not h.capabilities.emergency_services]
    if min_rating is not None:
        items = [h for h in items if h.quality.overall_rating and h.quality.overall_rating >= min_rating]
    if hospital_type:
        items = [h for h in items if hospital_type.lower() in (h.hospital_type or "").lower()]

    total = len(items)
    page = max(1, page)
    per_page = min(100, max(1, per_page))
    start = (page - 1) * per_page
    return HospitalListResponse(
        total=total, page=page, per_page=per_page,
        hospitals=[HospitalRecordModel(**h.to_dict()) for h in items[start:start + per_page]],
    )


@app.get("/hospitals/{facility_id}", response_model=HospitalRecordModel)
def get_hospital(facility_id: str):
    h = _hospitals.get(facility_id)
    if not h:
        raise HTTPException(404, f"Hospital '{facility_id}' not found")
    return HospitalRecordModel(**h.to_dict())


@app.get("/gaps", response_model=GapAnalysisResultModel)
def gaps(state: Optional[str] = None, city: Optional[str] = None):
    from backend.services.gap_detection import analyse_region
    items = list(_hospitals.values())
    if state:
        items = [h for h in items if (h.state or "").upper() == state.upper()]
    if city:
        items = [h for h in items if city.lower() in (h.city or "").lower()]
    if not items:
        raise HTTPException(404, "No hospitals found for the specified region")
    result = analyse_region(items, state=state, city=city)
    return GapAnalysisResultModel(**result.to_dict())


@app.post("/parse", response_model=ParseResponseModel)
def parse(body: ParseRequest):
    from backend.services.idp_service import parse_hospital_text
    from backend.services.validation_service import validate_hospital
    if not body.text or len(body.text.strip()) < 3:
        raise HTTPException(400, "text field is required (min 3 chars)")
    result = parse_hospital_text(body.text, strict_mode=body.strict_mode)
    d = result.to_dict()
    if result.success and result.hospital:
        d["validation"] = validate_hospital(result.hospital).to_dict()
    return ParseResponseModel(**d)


@app.post("/validate", response_model=ValidationResultModel)
def validate(body: ValidateRequest):
    from backend.services.validation_service import validate_hospital
    if body.facility_id:
        h = _hospitals.get(body.facility_id)
        if not h:
            raise HTTPException(404, f"Hospital '{body.facility_id}' not found")
    elif body.hospital:
        try:
            h = HospitalRecord.from_dict(dict(body.hospital))
        except Exception as e:
            raise HTTPException(400, f"Invalid hospital object: {e}")
    else:
        raise HTTPException(400, "Provide facility_id or hospital")
    return ValidationResultModel(**validate_hospital(h).to_dict())


@app.post("/query", response_model=AgentResponseModel)
def query(body: QueryRequest):
    from backend.agents.healthcare_agent import run_agent
    if _index is None or _index.total == 0:
        raise HTTPException(503, "Search index not ready")
    q = body.query.strip()
    if not q:
        raise HTTPException(400, "query field is required")
    resp = run_agent(
        query=q, index=_index, state_filter=body.state_filter, city_filter=body.city_filter,
        include_reasoning=body.include_reasoning, max_results=min(50, max(1, body.max_results)),
    )
    return AgentResponseModel(**resp.to_dict())
