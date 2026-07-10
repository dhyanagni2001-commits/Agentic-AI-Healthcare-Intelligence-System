"""5-agent reasoning graph, built on LangGraph.

Five agents share one `AgentState` (Query Planner -> Retrieval ->
Validation -> Gap Analysis -> Recommendation), compiled once at import
time and re-invoked per query. The retriever (FAISS/embeddings or legacy
TF-IDF) is injected per-call via LangGraph's `config["configurable"]`
rather than baked into the compiled graph, so the same graph instance
serves every request without rebuilding.

Accepts any index object that duck-types the legacy search()/filter_by_state()/
filter_by_capability()/get_by_id() interface (TfidfHospitalIndex or
VectorHospitalIndex) — kept import-light at runtime (no torch/faiss
dependency) so this module and the zero-dependency server.py fallback
keep working without the embeddings stack installed; VectorHospitalIndex
is only imported for static type-checking, never at runtime.
"""
from __future__ import annotations
import re, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, START, END
from langgraph.types import RunnableConfig

from backend.models.schemas import (
    AgentResponse, GapAnalysisResult, HealthcareGap,
    HospitalRecord, Recommendation, ReasoningStep, RiskLevel
)
from backend.services.validation_service import validate_hospital
from backend.services.gap_detection import analyse_region
from backend.services.recommendation_engine import generate_recommendations
from backend.services.llm_service import generate as llm_generate, LLMNotConfiguredError
from backend.prompts.templates import RAG_ANSWER_SYSTEM, RAG_ANSWER_PROMPT, format_evidence_block

if TYPE_CHECKING:
    from backend.services.hybrid_index import VectorHospitalIndex

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


# ── Shared state (LangGraph schema) ───────────────────────────────────────────

@dataclass
class AgentState:
    query: str
    max_results: int = 10
    intents: List[str] = field(default_factory=list)
    state_filter: Optional[str] = None
    city_filter:  Optional[str] = None
    cap_filter:   Optional[str] = None
    retrieved: List[HospitalRecord] = field(default_factory=list)
    gap_result: Optional[GapAnalysisResult] = None
    recommendations: List[Recommendation] = field(default_factory=list)
    steps: List[ReasoningStep] = field(default_factory=list)


# ── Agents ─────────────────────────────────────────────────────────────────────
# Each agent is a LangGraph node: `AgentState -> Partial<AgentState>`.

def query_planner_agent(state: AgentState) -> Dict[str, Any]:
    """Extracts intent, state, city, and capability/specialty filters from
    the raw query text."""
    q = state.query.lower()
    intents = [k for k, pats in _INTENTS.items() if any(re.search(p, q) for p in pats)]
    intents = intents or ["hospital_search"]

    state_filter = state.state_filter
    for name, abbr in _STATE_NAMES.items():
        if name in q:
            state_filter = abbr; break
    if not state_filter:
        m = re.search(r"\b([A-Z]{2})\b", state.query)
        if m and m.group(1) in _ABBREVS:
            state_filter = m.group(1)

    city_filter = state.city_filter
    m = re.search(r"\bin\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)", state.query)
    if m:
        city_filter = m.group(1)

    cap_filter = state.cap_filter
    for cap, kws in _CAP_KW.items():
        if any(kw in q for kw in kws):
            cap_filter = cap; break

    step = ReasoningStep("query_planner", f"Analysed: '{state.query}'", [],
        f"Intents={intents}, state={state_filter}, city={city_filter}, capability={cap_filter}")
    return {
        "intents": intents, "state_filter": state_filter, "city_filter": city_filter,
        "cap_filter": cap_filter, "steps": state.steps + [step],
    }


def retrieval_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Searches the retriever (FAISS/embeddings or legacy TF-IDF, injected
    via config["configurable"]["index"]) for hospitals matching the
    planned query/filters."""
    index = config["configurable"]["index"]
    q = state.query + (f" {state.cap_filter}" if state.cap_filter else "")
    results = index.search(q, top_k=state.max_results,
                           state_filter=state.state_filter, city_filter=state.city_filter,
                           cap_filter=state.cap_filter)

    if not results and state.state_filter:
        hs = index.filter_by_state(state.state_filter)
        results = [(h, 0.5) for h in hs[:state.max_results]]

    if state.cap_filter and len(results) < 5:
        existing = {h.facility_id for h, _ in results}
        for h in index.filter_by_capability(state.cap_filter):
            if h.facility_id not in existing:
                results.append((h, 0.3))
            if len(results) >= state.max_results: break

    retrieved = [h for h, _ in results]
    step = ReasoningStep("retrieval", f"Searched index: '{q}'",
        [h.facility_id for h in retrieved],
        f"Retrieved {len(retrieved)} hospitals"
        + (f" in {state.state_filter}" if state.state_filter else ""))
    return {"retrieved": retrieved, "steps": state.steps + [step]}


def validation_agent(state: AgentState) -> Dict[str, Any]:
    """Flags missing/inconsistent data and low-confidence records among the
    retrieved hospitals."""
    high_risk, total_issues = [], 0
    for h in state.retrieved:
        r = validate_hospital(h)
        if r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            high_risk.append(h.facility_name)
        total_issues += len(r.issues)
    step = ReasoningStep("validation", f"Validated {len(state.retrieved)} hospitals",
        [h.facility_id for h in state.retrieved],
        f"{len(high_risk)} high-risk, {total_issues} total issues"
        + (f". High-risk: {', '.join(high_risk[:3])}" if high_risk else ""))
    return {"steps": state.steps + [step]}


def gap_analysis_agent(state: AgentState) -> Dict[str, Any]:
    """Computes ICU/ER coverage, doctor density, and specialty availability
    for the retrieved region."""
    if not state.retrieved:
        step = ReasoningStep("gap_analysis", "No hospitals", [], "Skipped")
        return {"steps": state.steps + [step]}
    gr = analyse_region(state.retrieved, state=state.state_filter, city=state.city_filter)
    step = ReasoningStep("gap_analysis", f"Gap analysis on {len(state.retrieved)} hospitals",
        [h.facility_id for h in state.retrieved],
        f"{len(gr.gaps)} gap(s), overall risk: {gr.overall_risk.upper()}")
    return {"gap_result": gr, "steps": state.steps + [step]}


def recommendation_agent(state: AgentState) -> Dict[str, Any]:
    """Maps detected gaps to prioritised staffing/infrastructure/investment
    recommendations."""
    if not state.gap_result or not state.gap_result.gaps:
        step = ReasoningStep("recommendation", "No gaps", [], "No recommendations needed")
        return {"steps": state.steps + [step]}
    recs = generate_recommendations(state.gap_result, state.retrieved)
    step = ReasoningStep("recommendation", f"{len(recs)} recommendations",
        [h.facility_id for h in state.retrieved],
        f"Top: {recs[0].title if recs else 'none'}")
    return {"recommendations": recs, "steps": state.steps + [step]}


# ── Graph assembly (compiled once, reused across every run_agent call) ───────

_builder = StateGraph(AgentState)
_builder.add_node("query_planner", query_planner_agent)
_builder.add_node("retrieval", retrieval_agent)
_builder.add_node("validation", validation_agent)
_builder.add_node("gap_analysis", gap_analysis_agent)
_builder.add_node("recommendation", recommendation_agent)
_builder.add_edge(START, "query_planner")
_builder.add_edge("query_planner", "retrieval")
_builder.add_edge("retrieval", "validation")
_builder.add_edge("validation", "gap_analysis")
_builder.add_edge("gap_analysis", "recommendation")
_builder.add_edge("recommendation", END)
_compiled_graph = _builder.compile()


# ── Answer synthesis ──────────────────────────────────────────────────────────

def _llm_answer(state: AgentState, retrieved_documents: List[Dict]) -> Optional[str]:
    """Grounded LLM narration over the already-computed gaps/recommendations/
    retrieved hospitals. Returns None if the LLM isn't configured, so the
    caller can fall back to the deterministic template — the rule-based
    computations are the source of truth either way; the LLM only narrates."""
    region = ", ".join(p for p in [state.city_filter, state.state_filter] if p) or "the queried region"
    facts = [f"{len(state.retrieved)} hospitals retrieved for {region}."]
    if state.gap_result:
        facts.append(f"Gap analysis: {state.gap_result.summary}")
        facts.extend(f"Gap — [{g.severity.upper()}] {g.description}" for g in state.gap_result.gaps)
    if state.recommendations:
        facts.extend(f"Recommendation — [{r.priority.upper()}] {r.title}: {r.description}"
                      for r in state.recommendations[:5])

    evidence_block = format_evidence_block(retrieved_documents) if retrieved_documents else \
        "\n".join(f"- {h.facility_name} ({h.city}, {h.state})" for h in state.retrieved[:8])
    prompt = RAG_ANSWER_PROMPT.format(query=state.query, evidence_block=evidence_block,
                                       facts_block="\n".join(f"- {f}" for f in facts))
    try:
        return llm_generate(prompt, system=RAG_ANSWER_SYSTEM)
    except LLMNotConfiguredError as e:
        log.info(f"LLM not configured, using template answer: {e}")
        return None


def _template_answer(state: AgentState) -> str:
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

def run_agent(query: str, index: VectorHospitalIndex,
              state_filter: Optional[str] = None,
              city_filter:  Optional[str] = None,
              include_reasoning: bool = True,
              max_results: int = 10) -> AgentResponse:
    initial = AgentState(query=query, max_results=max_results,
                          state_filter=state_filter, city_filter=city_filter)

    try:
        result = _compiled_graph.invoke(initial, config={"configurable": {"index": index}})
        state = AgentState(**result)
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        return AgentResponse(query=query, answer=f"Agent error: {e}", confidence=0.0)

    gaps = state.gap_result.gaps if state.gap_result else []

    retrieved_documents: List[Dict] = []
    if hasattr(index, "search_documents"):
        try:
            raw = index.search_documents(state.query, top_k=max_results)
            retrieved_documents = [{
                "id": doc.id, "doc_type": doc.doc_type,
                "snippet": doc.text[:220], "score": round(score, 4),
                "metadata": doc.metadata,
            } for doc, score in raw]
        except Exception as e:
            log.warning(f"search_documents failed, continuing without evidence panel: {e}")

    answer = _llm_answer(state, retrieved_documents) or _template_answer(state)

    return AgentResponse(
        query=query,
        answer=answer,
        reasoning_steps=state.steps if include_reasoning else [],
        hospitals_referenced=[h.facility_id for h in state.retrieved],
        gaps_identified=gaps,
        recommendations=state.recommendations,
        confidence=min(1.0, 0.5 + 0.1 * len(state.retrieved)),
        data_sources=[h.facility_name for h in state.retrieved[:5]],
        retrieved_documents=retrieved_documents,
    )
