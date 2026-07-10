"""VectorHospitalIndex — FAISS+embeddings retrieval with the same call
signature as the legacy TF-IDF `HospitalIndex` (backend/services/
legacy_tfidf_service.py), so backend/agents/healthcare_agent.py only needs
its import swapped, not rewritten.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

from backend.models.schemas import HospitalRecord
from backend.ingestion.documents import SearchableDocument
from backend.ingestion.pipeline import build_all_documents
from backend.services.vector_store import FaissVectorStore, DEFAULT_DIR

log = logging.getLogger(__name__)


class VectorHospitalIndex:
    def __init__(self):
        self._records: Dict[str, HospitalRecord] = {}
        self._store = FaissVectorStore()
        self._built = False

    def build(self, hospitals: Dict[str, HospitalRecord]) -> None:
        """Builds the full multi-document-type FAISS index. `hospitals` is
        accepted for interface parity with the legacy index (and used for
        filter_by_state/filter_by_capability/get_by_id); the embedded
        corpus itself also includes doctor-aggregate and department docs
        pulled fresh from the ingestion pipeline."""
        self._records = hospitals
        if self._store.load(DEFAULT_DIR):
            self._built = True
            return
        docs, loaded_hospitals = build_all_documents()
        if not self._records:
            self._records = loaded_hospitals
        self._store.build(docs)
        self._store.save(DEFAULT_DIR)
        self._built = True

    def search(self,
               query: str,
               top_k: int = 10,
               state_filter: Optional[str] = None,
               city_filter: Optional[str] = None,
               cap_filter: Optional[str] = None) -> List[Tuple[HospitalRecord, float]]:
        if not self._built:
            raise RuntimeError("Index not built — call build() first")

        # Over-fetch so location + capability filtering still leaves top_k results.
        raw = self._store.search(query, top_k=top_k * 15)
        seen: set = set()
        candidates: List[Tuple[HospitalRecord, float, float]] = []  # (h, raw_score, boosted_score)
        for doc, score in raw:
            fid = doc.metadata.get("facility_id")
            if not fid or fid in seen:
                continue
            h = self._records.get(fid)
            if not h:
                continue
            if state_filter and (h.state or "").upper() != state_filter.upper():
                continue
            if city_filter and city_filter.lower() not in (h.city or "").lower():
                continue
            seen.add(fid)
            # Boost hospitals that have the queried capability so they rank higher
            # than semantically similar hospitals that lack it.
            has_cap = cap_filter and vars(h.capabilities).get(cap_filter, False)
            boosted = score * 1.3 if has_cap else score
            candidates.append((h, round(score, 4), boosted))

        candidates.sort(key=lambda x: -x[2])
        results = [(h, s) for h, s, _ in candidates[:top_k]]

        if not results and (state_filter or city_filter):
            for h in self.filter_by_state(state_filter) if state_filter else self._records.values():
                if city_filter and city_filter.lower() not in (h.city or "").lower():
                    continue
                results.append((h, 0.5))
                if len(results) >= top_k:
                    break

        return results

    def search_documents(self, query: str, top_k: int = 10) -> List[Tuple[SearchableDocument, float]]:
        """Raw retrieval across ALL document types (hospital/doctor/department/
        gap), without collapsing to one-per-hospital — used by the RAG
        pipeline for evidence/citations, where doc_type and snippet text
        matter, unlike the legacy-compatible `search()` above."""
        if not self._built:
            raise RuntimeError("Index not built — call build() first")
        return self._store.search(query, top_k=top_k)

    def filter_by_state(self, state: str) -> List[HospitalRecord]:
        return [h for h in self._records.values() if (h.state or "").upper() == state.upper()]

    def filter_by_capability(self, cap: str) -> List[HospitalRecord]:
        return [h for h in self._records.values() if vars(h.capabilities).get(cap, False)]

    def get_by_id(self, fid: str) -> Optional[HospitalRecord]:
        return self._records.get(fid)

    def get_all(self) -> Dict[str, HospitalRecord]:
        return self._records

    @property
    def total(self) -> int:
        return len(self._records)


_INDEX: Optional[VectorHospitalIndex] = None


def get_index() -> VectorHospitalIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = VectorHospitalIndex()
    return _INDEX


def build_index(hospitals: Dict[str, HospitalRecord]) -> VectorHospitalIndex:
    idx = get_index()
    idx.build(hospitals)
    return idx
