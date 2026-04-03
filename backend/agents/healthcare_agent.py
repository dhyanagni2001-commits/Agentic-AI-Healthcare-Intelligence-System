"""5-node LangGraph-style reasoning agent — stdlib only."""
from __future__ import annotations
import re, logging
from typing import Dict, List, Optional
from backend.models.schemas import (
    AgentResponse, GapAnalysisResult, HealthcareGap,
    HospitalRecord, Recommendation, ReasoningStep, RiskLevel
)
from backend.services.rag_service import HospitalIndex
from backend.services.validation_service import validate_hospital
from backend.services.gap_detection import analyse_region
from backend.services.recommendation_engine import generate_recommendations

log = logging.getLogger(__name__)

_STATE_NAMES = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA",
    "colorado":"CO","connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA",
    "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS",
    "kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA",
    "michigan":"MI","minnesota":"MN","mississippi":"MS","missouri":"MO","montana":"MT",
    "nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
    "new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND",
    "ohio":"OH","oklahoma":"OK","oregon":"OR","pennsylvania":"PA","rhode island":"RI",
    "south carolina":"SC","south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT",
    "vermont":"VT","virginia":"VA","washington":"WA","west virginia":"WV",
    "wisconsin":"WI","wyoming":"WY",
}
_ABBREVS = set(_STATE_NAMES.values())

_INTENTS = {
    "gap_analysis":    [r"gap",r"desert",r"under.?served",r"missing",r"lack",r"shortage",r"risk"],
    "recommendation":  [r"recommend",r"suggest",r"invest",r"improve",r"deploy",r"send patient",r"where should"],
    "hospital_search": [r"hospital",r"facility",r"clinic",r"find",r"list",r"show",r"which"],
    "statistics":      [r"how many",r"count",r"total",r"average",r"stat",r"summary"],
}

_CAP_KW = {
    "icu":["icu","intensive care"],"emergency_services":["emergency","er "],
    "surgery":["surgery","surgical"],"maternity":["maternity","obstetric"],
    "pediatrics":["pediatric","children"],"cardiac_care":["cardiac","heart"],
    "oncology":["cancer","oncol"],"mental_health":["mental health","psych"],
    "rehabilitation":["rehab"],
}


# ── State object ──────────────────────────────────────────────────────────────

class AgentState:
    def __init__(self, query: str, max_results: int = 10):
        self.query = query
        self.max_results = max_results
        self.intents: List[str] = []
        self.state_filter: Optional[str] = None
        self.city_filter:  Optional[str] = None
        self.cap_filter:   Optional[str] = None
        self.retrieved: List[HospitalRecord] = []
        self.gap_result: Optional[GapAnalysisResult] = None
        self.recommendations: List[Recommendation] = []
        self.steps: List[ReasoningStep] = []

    def add_step(self, name, desc, data_ids, summary):
        self.steps.append(ReasoningStep(name, desc, data_ids, summary))


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_parse_query(state: AgentState) -> AgentState:
    q = state.query.lower()
    intents = [k for k, pats in _INTENTS.items() if any(re.search(p,q) for p in pats)]
    state.intents = intents or ["hospital_search"]

    # State
    for name, abbr in _STATE_NAMES.items():
        if name in q:
            state.state_filter = abbr; break
    if not state.state_filter:
        m = re.search(r"\b([A-Z]{2})\b", state.query)
        if m and m.group(1) in _ABBREVS:
            state.state_filter = m.group(1)

    # City
    m = re.search(r"\bin\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)", state.query)
    if m: state.city_filter = m.group(1)

    # Capability
    for cap, kws in _CAP_KW.items():
        if any(kw in q for kw in kws):
            state.cap_filter = cap; break

    state.add_step("parse_query", f"Analysed: '{state.query}'", [],
        f"Intents={state.intents}, state={state.state_filter}, "
        f"city={state.city_filter}, capability={state.cap_filter}")
    return state


def node_retrieve_data(state: AgentState, index: HospitalIndex) -> AgentState:
    q = state.query + (f" {state.cap_filter}" if state.cap_filter else "")
    results = index.search(q, top_k=state.max_results,
                           state_filter=state.state_filter, city_filter=state.city_filter)

    if not results and state.state_filter:
        hs = index.filter_by_state(state.state_filter)
        results = [(h, 0.5) for h in hs[:state.max_results]]

    if state.cap_filter and len(results) < 5:
        existing = {h.facility_id for h, _ in results}
        for h in index.filter_by_capability(state.cap_filter):
            if h.facility_id not in existing:
                results.append((h, 0.3))
            if len(results) >= state.max_results: break

    state.retrieved = [h for h, _ in results]
    state.add_step("retrieve_data", f"Searched index: '{q}'",
        [h.facility_id for h in state.retrieved],
        f"Retrieved {len(state.retrieved)} hospitals"
        + (f" in {state.state_filter}" if state.state_filter else ""))
    return state


def node_validate_data(state: AgentState) -> AgentState:
    high_risk, total_issues = [], 0
    for h in state.retrieved:
        r = validate_hospital(h)
        if r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            high_risk.append(h.facility_name)
        total_issues += len(r.issues)
    state.add_step("validate_data", f"Validated {len(state.retrieved)} hospitals",
        [h.facility_id for h in state.retrieved],
        f"{len(high_risk)} high-risk, {total_issues} total issues"
        + (f". High-risk: {', '.join(high_risk[:3])}" if high_risk else ""))
    return state


def node_detect_gaps(state: AgentState) -> AgentState:
    if not state.retrieved:
        state.add_step("detect_gaps","No hospitals","[]","Skipped")
        return state
    gr = analyse_region(state.retrieved, state=state.state_filter, city=state.city_filter)
    state.gap_result = gr
    state.add_step("detect_gaps", f"Gap analysis on {len(state.retrieved)} hospitals",
        [h.facility_id for h in state.retrieved],
        f"{len(gr.gaps)} gap(s), overall risk: {gr.overall_risk.upper()}")
    return state


def node_generate_recommendation(state: AgentState) -> AgentState:
    if not state.gap_result or not state.gap_result.gaps:
        state.add_step("generate_recommendation","No gaps","[]","No recommendations needed")
        return state
    recs = generate_recommendations(state.gap_result, state.retrieved)
    state.recommendations = recs
    state.add_step("generate_recommendation", f"{len(recs)} recommendations",
        [h.facility_id for h in state.retrieved],
        f"Top: {recs[0].title if recs else 'none'}")
    return state


# ── Answer synthesis ──────────────────────────────────────────────────────────

def _answer(state: AgentState) -> str:
    parts = []
    region = ", ".join(p for p in [state.city_filter, state.state_filter] if p) or "the queried region"

    if "statistics" in state.intents and state.retrieved:
        total = len(state.retrieved)
        er = sum(1 for h in state.retrieved if h.capabilities.emergency_services)
        avg_d = sum(h.doctor_count for h in state.retrieved) / total if total else 0
        parts.append(f"**Statistics for {region}:**\n"
                     f"- Hospitals analysed: {total}\n"
                     f"- With emergency services: {er}/{total}\n"
                     f"- Average doctors per hospital: {avg_d:.1f}")

    if "hospital_search" in state.intents and state.retrieved:
        rows = "\n".join(
            f"- {h.facility_name} ({h.city or '?'}, {h.state or '?'}) "
            f"— {h.hospital_type or 'N/A'}, Rating: {h.quality.overall_rating or 'N/A'}/5"
            for h in state.retrieved[:8]
        )
        parts.append(f"**Hospitals in {region}:**\n{rows}")

    if state.gap_result and state.gap_result.gaps:
        gtext = "\n".join(f"- [{g.severity.upper()}] {g.description}"
                          for g in state.gap_result.gaps)
        parts.append(f"\n**Healthcare Gaps ({state.gap_result.overall_risk.upper()} risk):**\n{gtext}")
    elif "gap_analysis" in state.intents:
        parts.append(f"\n✅ No critical healthcare gaps detected in {region}.")

    if state.recommendations:
        rtext = "\n".join(f"- [{r.priority.upper()}] **{r.title}**: {r.description}"
                          for r in state.recommendations[:5])
        parts.append(f"\n**Recommendations:**\n{rtext}")

    if not parts:
        if state.retrieved:
            parts.append(f"Found {len(state.retrieved)} hospitals. "
                         f"Top: {state.retrieved[0].facility_name} in "
                         f"{state.retrieved[0].city}, {state.retrieved[0].state}.")
        else:
            parts.append("No hospitals found for your query. Try broadening the search or removing filters.")

    return "\n\n".join(parts)


# ── Public entry point ────────────────────────────────────────────────────────

def run_agent(query: str, index: HospitalIndex,
              state_filter: Optional[str] = None,
              city_filter:  Optional[str] = None,
              include_reasoning: bool = True,
              max_results: int = 10) -> AgentResponse:
    state = AgentState(query, max_results)
    if state_filter: state.state_filter = state_filter
    if city_filter:  state.city_filter  = city_filter

    try:
        state = node_parse_query(state)
        state = node_retrieve_data(state, index)
        state = node_validate_data(state)
        state = node_detect_gaps(state)
        state = node_generate_recommendation(state)
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        return AgentResponse(query=query, answer=f"Agent error: {e}", confidence=0.0)

    gaps = state.gap_result.gaps if state.gap_result else []
    return AgentResponse(
        query=query,
        answer=_answer(state),
        reasoning_steps=state.steps if include_reasoning else [],
        hospitals_referenced=[h.facility_id for h in state.retrieved],
        gaps_identified=gaps,
        recommendations=state.recommendations,
        confidence=min(1.0, 0.5 + 0.1 * len(state.retrieved)),
        data_sources=[h.facility_name for h in state.retrieved[:5]],
    )
