"""Explicit RAG pipeline: query -> embed -> FAISS retrieval -> context
construction -> LLM -> grounded answer.

Self-contained and independently testable (with a mocked LLMProvider) —
backend/agents/healthcare_agent.py calls this for answer synthesis,
optionally passing extra computed facts (gap/recommendation results) as
additional grounding.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

from backend.models.schemas import HospitalRecord
from backend.services.hybrid_index import VectorHospitalIndex
from backend.services.llm_service import (
    generate as llm_generate, stream as llm_stream, LLMError, LLMNotConfiguredError,
)
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


def _retrieve_and_build_prompt(query: str,
                               index: VectorHospitalIndex,
                               top_k: int,
                               state_filter: Optional[str],
                               city_filter: Optional[str],
                               extra_facts: Optional[List[str]],
                               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                          List[HospitalRecord], str, str]:
    """Steps 1-3 shared by run_rag and stream_rag: retrieval, hospital
    resolution, and prompt construction. Returns (documents, physicians,
    hospitals, facts_block, prompt)."""
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
    return retrieved_documents, retrieved_physicians, retrieved_hospitals, facts_block, prompt


def _fallback_answer(facts_block: str, err: LLMError) -> str:
    reason = "not configured" if isinstance(err, LLMNotConfiguredError) else "unavailable"
    return f"LLM is {reason}. Computed facts:\n" + facts_block


def run_rag(query: str,
            index: VectorHospitalIndex,
            top_k: int = 10,
            state_filter: Optional[str] = None,
            city_filter: Optional[str] = None,
            extra_facts: Optional[List[str]] = None) -> RAGResult:
    docs, physicians, hospitals, facts_block, prompt = _retrieve_and_build_prompt(
        query, index, top_k, state_filter, city_filter, extra_facts)

    # 4. LLM generation (grounded — facts/evidence are computed, not invented)
    try:
        answer = llm_generate(prompt, system=RAG_ANSWER_SYSTEM)
        confidence = min(1.0, 0.5 + 0.05 * len(docs))
    except LLMError as e:
        log.warning(f"LLM unavailable, falling back to fact summary: {e}")
        answer = _fallback_answer(facts_block, e)
        confidence = 0.3

    return RAGResult(
        query=query,
        answer=answer,
        retrieved_hospitals=hospitals,
        retrieved_physicians=physicians,
        retrieved_documents=docs,
        confidence=confidence,
    )


def stream_rag(query: str,
               index: VectorHospitalIndex,
               top_k: int = 10,
               state_filter: Optional[str] = None,
               city_filter: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Streaming variant of run_rag. Yields events in a fixed order:

      {"event": "sources", "data": {"documents": [...], "hospital_ids": [...]}}
      {"event": "token",   "data": {"text": "..."}}            (0..n times)
      {"event": "error",   "data": {"message": "...", "partial": bool}}  (LLM runtime failure only)
      {"event": "done",    "data": {"source_ids": [...], "fallback": bool,
                                     "fallback_reason": None|"llm_not_configured"|"llm_unavailable", timings}}

    Sources are emitted before any token, so citations survive even if
    generation fails mid-stream; `done` repeats the source ids so a client
    that only keeps the final event can still attribute the answer.
    """
    t0 = time.perf_counter()
    docs, _, hospitals, facts_block, prompt = _retrieve_and_build_prompt(
        query, index, top_k, state_filter, city_filter, None)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    source_ids = [d["id"] for d in docs]
    yield {"event": "sources", "data": {
        "documents": docs, "hospital_ids": [h.facility_id for h in hospitals]}}

    fallback = False
    fallback_reason: Optional[str] = None
    n_chunks = 0
    first_token_ms: Optional[float] = None
    try:
        for text in llm_stream(prompt, system=RAG_ANSWER_SYSTEM):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t0) * 1000
            n_chunks += 1
            yield {"event": "token", "data": {"text": text}}
    except LLMError as e:
        fallback = True
        if isinstance(e, LLMNotConfiguredError):  # deliberate config, not a failure
            fallback_reason = "llm_not_configured"
        else:
            fallback_reason = "llm_unavailable"
            log.warning(f"LLM stream failed after {n_chunks} chunks: {e}")
            yield {"event": "error", "data": {"message": str(e), "fallback": True,
                                              "partial": n_chunks > 0}}
        yield {"event": "token", "data": {"text": ("\n\n" if n_chunks else "")
                                          + _fallback_answer(facts_block, e)}}

    yield {"event": "done", "data": {
        "source_ids": source_ids,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "chunks": n_chunks,
        "retrieval_ms": round(retrieval_ms, 2),
        "first_token_ms": round(first_token_ms, 2) if first_token_ms is not None else None,
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }}
