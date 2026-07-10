"""Explicit RAG pipeline: query -> embed -> FAISS retrieval -> context
construction -> LLM -> grounded answer.

Self-contained and independently testable (with a mocked LLMProvider) —
backend/agents/healthcare_agent.py calls this for answer synthesis,
optionally passing extra computed facts (gap/recommendation results) as
additional grounding.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from backend.models.schemas import HospitalRecord
from backend.services.hybrid_index import VectorHospitalIndex
from backend.services.llm_service import generate as llm_generate, LLMNotConfiguredError
from backend.prompts.templates import (
    RAG_ANSWER_SYSTEM, RAG_ANSWER_PROMPT, format_evidence_block,
)

log = logging.getLogger(__name__)

SNIPPET_LEN = 220


@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_hospitals: List[HospitalRecord] = field(default_factory=list)
    retrieved_physicians: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_hospitals": [h.to_dict() for h in self.retrieved_hospitals],
            "retrieved_physicians": self.retrieved_physicians,
            "retrieved_documents": self.retrieved_documents,
            "confidence": self.confidence,
        }


def _basic_facts(hospitals: List[HospitalRecord]) -> List[str]:
    """Cheap, always-available grounding facts derived directly from
    retrieved records — independent of the gap-detection engine, so this
    pipeline works standalone."""
    if not hospitals:
        return ["No hospitals were retrieved for this query."]
    total = len(hospitals)
    er = sum(1 for h in hospitals if h.capabilities.emergency_services)
    icu = sum(1 for h in hospitals if h.capabilities.icu)
    avg_doctors = sum(h.doctor_count for h in hospitals) / total
    return [
        f"{total} hospitals retrieved.",
        f"{er}/{total} have emergency services.",
        f"{icu}/{total} have an ICU.",
        f"Average doctor count across retrieved hospitals: {avg_doctors:.1f}.",
    ]


def run_rag(query: str,
            index: VectorHospitalIndex,
            top_k: int = 10,
            state_filter: Optional[str] = None,
            city_filter: Optional[str] = None,
            extra_facts: Optional[List[str]] = None) -> RAGResult:
    # 1. Embedding + FAISS retrieval (raw documents across all types)
    raw_docs = index.search_documents(query, top_k=top_k)

    retrieved_documents: List[Dict[str, Any]] = []
    retrieved_physicians: List[Dict[str, Any]] = []
    hospital_ids_seen: List[str] = []

    for doc, score in raw_docs:
        retrieved_documents.append({
            "id": doc.id,
            "doc_type": doc.doc_type,
            "snippet": doc.text[:SNIPPET_LEN],
            "score": round(score, 4),
            "metadata": doc.metadata,
        })
        if doc.doc_type == "doctor_aggregate":
            retrieved_physicians.append(doc.metadata)
        fid = doc.metadata.get("facility_id")
        if fid and fid not in hospital_ids_seen:
            hospital_ids_seen.append(fid)

    # 2. Resolve hospital records referenced by any retrieved document type
    retrieved_hospitals = [h for fid in hospital_ids_seen
                            if (h := index.get_by_id(fid)) is not None]
    if state_filter:
        retrieved_hospitals = [h for h in retrieved_hospitals
                                if (h.state or "").upper() == state_filter.upper()]
    if city_filter:
        retrieved_hospitals = [h for h in retrieved_hospitals
                                if city_filter.lower() in (h.city or "").lower()]

    # 3. Context construction
    evidence_block = format_evidence_block(retrieved_documents)
    facts = _basic_facts(retrieved_hospitals) + (extra_facts or [])
    facts_block = "\n".join(f"- {f}" for f in facts)
    prompt = RAG_ANSWER_PROMPT.format(query=query, evidence_block=evidence_block,
                                       facts_block=facts_block)

    # 4. LLM generation (grounded — facts/evidence are computed, not invented)
    try:
        answer = llm_generate(prompt, system=RAG_ANSWER_SYSTEM)
        confidence = min(1.0, 0.5 + 0.05 * len(retrieved_documents))
    except LLMNotConfiguredError as e:
        log.warning(f"LLM not configured, falling back to fact summary: {e}")
        answer = "LLM is not configured. Computed facts:\n" + facts_block
        confidence = 0.3

    return RAGResult(
        query=query,
        answer=answer,
        retrieved_hospitals=retrieved_hospitals,
        retrieved_physicians=retrieved_physicians,
        retrieved_documents=retrieved_documents,
        confidence=confidence,
    )
