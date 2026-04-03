"""TF-IDF hospital search index — stdlib only."""
from __future__ import annotations
import re, math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from backend.models.schemas import HospitalRecord


def _to_text(h: HospitalRecord) -> str:
    caps = [k.replace("_", " ") for k, v in vars(h.capabilities).items() if v]
    return " | ".join([
        h.facility_name,
        f"{h.city or ''} {h.state or ''}",
        h.hospital_type or "",
        f"emergency {'yes' if h.capabilities.emergency_services else 'no'}",
        " ".join(caps),
        f"{h.doctor_count} doctors",
        " ".join(h.departments[:15]),
    ]).lower()


def _tok(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HospitalIndex:
    def __init__(self):
        self._records: Dict[str, HospitalRecord] = {}
        self._tf: Dict[str, Dict[str, float]] = {}
        self._idf: Dict[str, float] = {}
        self._built = False

    def build(self, hospitals: Dict[str, HospitalRecord]) -> None:
        self._records = hospitals
        texts = {fid: _to_text(h) for fid, h in hospitals.items()}

        doc_counts: Dict[str, Dict[str, int]] = {}
        for fid, text in texts.items():
            toks = _tok(text)
            cnt: Dict[str, int] = defaultdict(int)
            for t in toks: cnt[t] += 1
            total = len(toks) or 1
            doc_counts[fid] = cnt
            self._tf[fid] = {t: c / total for t, c in cnt.items()}

        n = len(texts)
        df: Dict[str, int] = defaultdict(int)
        for cnt in doc_counts.values():
            for t in cnt: df[t] += 1
        self._idf = {t: math.log((n + 1) / (f + 1)) + 1 for t, f in df.items()}
        self._built = True

    def search(self,
               query: str,
               top_k: int = 10,
               state_filter: Optional[str] = None,
               city_filter: Optional[str] = None) -> List[Tuple[HospitalRecord, float]]:
        if not self._built:
            raise RuntimeError("Index not built — call build() first")

        scores: Dict[str, float] = defaultdict(float)
        for tok in _tok(query):
            idf = self._idf.get(tok, 0)
            for fid, tf_map in self._tf.items():
                scores[fid] += tf_map.get(tok, 0) * idf

        # Build candidate list; when location filter active, include 0-score docs in region
        all_fids = list(scores.keys()) + [f for f in self._records if f not in scores]
        seen: set = set()
        results: List[Tuple[HospitalRecord, float]] = []
        for fid in sorted(all_fids, key=lambda f: -scores.get(f, 0)):
            if fid in seen: continue
            seen.add(fid)
            h = self._records.get(fid)
            if not h: continue
            if state_filter and (h.state or "").upper() != state_filter.upper(): continue
            if city_filter and city_filter.lower() not in (h.city or "").lower(): continue
            s = scores.get(fid, 0)
            if s > 0 or state_filter or city_filter:
                results.append((h, round(s, 4)))
            if len(results) >= top_k: break
        return results

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


_INDEX: Optional[HospitalIndex] = None


def get_index() -> HospitalIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = HospitalIndex()
    return _INDEX


def build_index(hospitals: Dict[str, HospitalRecord]) -> HospitalIndex:
    idx = get_index()
    idx.build(hospitals)
    return idx
