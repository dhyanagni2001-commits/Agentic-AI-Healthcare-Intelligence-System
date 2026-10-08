"""Retrieval-quality audit for the README's Precision@10 claim (0.39 -> 0.96).

Measures RETRIEVAL only (which hospitals come back), never answer text —
answer quality is scored separately by backend/evaluation/answer_quality.py.

Query set: backend/evaluation/retrieval_queries.jsonl (frozen, seeded).
  dev  — 8 queries reconstructing the README protocol (the original script
         was never committed, so this is a reconstruction, not a rerun).
  test — held-out; nothing in the system was tuned on it:
         field         state + capability-field labels (README protocol),
                       canonical keywords and paraphrases the planner lacks
         hospital_type / ownership
                       labels from attributes the capability boost never reads
         known_item    one correct facility per query (MRR / Hit@10)

Systems
  base_rate        expected P@10 of 10 random in-state hospitals (analytic)
  filter_only      no retrieval: in-state rows with the detected capability
  tfidf_raw / dense_raw      query text only, no filters
  dense_state      dense + planner state filter, no capability boost
  tfidf_agent / dense_agent  production path: query_planner -> retrieval_agent

    python -m backend.evaluation.retrieval_audit --out reports/retrieval_audit.json
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

from backend.services.embedding_service import _get_model  # torch before faiss
from backend.agents.healthcare_agent import AgentState, query_planner_agent, retrieval_agent
from backend.models.schemas import HospitalRecord

HERE = Path(__file__).parent
QUERIES = HERE / "retrieval_queries.jsonl"
K = 10


def is_relevant(q: dict, h: HospitalRecord) -> bool:
    if q["label"] == "known_item":
        return h.facility_id == q["facility_id"]
    if (h.state or "").upper() != q["state"]:
        return False
    if q["label"] == "field":
        return bool(vars(h.capabilities).get(q["cap"]))
    if q["label"] == "hospital_type":
        return h.hospital_type == q["hospital_type"]
    if q["label"] == "ownership":
        return h.ownership == q["ownership"]
    raise ValueError(q["label"])


def precision_at_k(rels: List[bool]) -> float:
    return sum(rels[:K]) / K


def ndcg_at_k(rels: List[bool], n_relevant: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rels[:K]) if r)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(K, n_relevant)))
    return dcg / idcg if idcg else 0.0


def rr_at_k(rels: List[bool]) -> float:
    return next((1 / (i + 1) for i, r in enumerate(rels[:K]) if r), 0.0)


def bootstrap_ci(xs: List[float], n: int = 2000, seed: int = 0) -> List[float]:
    if not xs:
        return [float("nan")] * 2
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3)]


def make_systems(hospitals: Dict[str, HospitalRecord], dense, tfidf) -> Dict[str, Callable]:
    by_state = defaultdict(list)
    for h in hospitals.values():
        by_state[(h.state or "").upper()].append(h)

    def plan(q):
        return query_planner_agent(AgentState(query=q["query"], max_results=K))

    def agent(index):
        def run(q):
            p = plan(q)
            st = AgentState(query=q["query"], max_results=K, **{k: p[k] for k in
                            ("intents", "state_filter", "city_filter", "cap_filter")})
            out = retrieval_agent(st, {"configurable": {"index": index}})
            return out["retrieved"][:K]
        return run

    def filter_only(q):
        p = plan(q)
        pool = by_state.get(p["state_filter"] or "", list(hospitals.values()))
        cap = p["cap_filter"]
        return [h for h in pool if not cap or vars(h.capabilities).get(cap)][:K]

    def dense_state(q):
        p = plan(q)
        return [h for h, _ in dense.search(q["query"], top_k=K, state_filter=p["state_filter"])]

    return {
        "filter_only": filter_only,
        "tfidf_raw": lambda q: [h for h, _ in tfidf.search(q["query"], top_k=K)],
        "dense_raw": lambda q: [h for h, _ in dense.search(q["query"], top_k=K)],
        "dense_state": dense_state,
        "tfidf_agent": agent(tfidf),
        "dense_agent": agent(dense),
    }


def base_rate(q: dict, hospitals: Dict[str, HospitalRecord]) -> float:
    if q["label"] == "known_item":
        return K / len(hospitals) / K  # one relevant item among all hospitals
    pool = [h for h in hospitals.values() if (h.state or "").upper() == q["state"]]
    return sum(is_relevant(q, h) for h in pool) / len(pool)


def evaluate(queries: List[dict], hospitals, systems) -> Dict:
    groups = defaultdict(list)
    for q in queries:
        g = f"{q['split']}/{q['label']}"
        if q["label"] == "field" and q["split"] == "test":
            g += "/para" if "-para-" in q["id"] else "/canon"
        groups[g].append(q)

    report = {}
    for g, qs in sorted(groups.items()):
        row = {"n_queries": len(qs)}
        br = [base_rate(q, hospitals) for q in qs]
        row["base_rate"] = {"p@10": round(sum(br) / len(br), 3)}
        for name, fn in systems.items():
            p, nd, rr, lat = [], [], [], []
            for q in qs:
                t0 = time.perf_counter()
                got = fn(q)
                lat.append(time.perf_counter() - t0)
                rels = [is_relevant(q, h) for h in got]
                n_rel = 1 if q["label"] == "known_item" else \
                    sum(is_relevant(q, h) for h in hospitals.values())
                p.append(precision_at_k(rels))
                nd.append(ndcg_at_k(rels, n_rel))
                rr.append(rr_at_k(rels))
            m = {"p@10": round(sum(p) / len(p), 3), "p@10_ci95": bootstrap_ci(p),
                 "ndcg@10": round(sum(nd) / len(nd), 3),
                 "median_latency_ms": round(sorted(lat)[len(lat) // 2] * 1000, 1)}
            if qs[0]["label"] == "known_item":
                m = {"mrr@10": round(sum(rr) / len(rr), 3), "mrr@10_ci95": bootstrap_ci(rr),
                     "hit@10": round(sum(r > 0 for r in rr) / len(rr), 3),
                     "median_latency_ms": m["median_latency_ms"]}
            row[name] = m
        report[g] = row
    return report


def planner_diagnostics(queries: List[dict]) -> Dict:
    """How often the planner turns a state name into a bogus city filter."""
    state_q = [q for q in queries if "state" in q]
    bogus = 0
    for q in state_q:
        p = query_planner_agent(AgentState(query=q["query"]))
        if p["city_filter"] and p["state_filter"] == q["state"] and \
                p["city_filter"].lower() in q["query"].lower() and \
                p["city_filter"].upper() not in {"", None} and \
                p["city_filter"].lower() == _state_name(q["state"]).lower():
            bogus += 1
    return {"state_scoped_queries": len(state_q), "state_parsed_as_city": bogus}


def _state_name(abbr: str) -> str:
    from backend.agents.healthcare_agent import _STATE_NAMES
    return next((k for k, v in _STATE_NAMES.items() if v == abbr), "")


def main():
    ap = argparse.ArgumentParser(description="Retrieval audit (P@10, nDCG@10, MRR@10)")
    ap.add_argument("--out", type=Path, default=Path("reports/retrieval_audit.json"))
    ap.add_argument("--label", default="current")
    a = ap.parse_args()

    from backend.services.data_loader import load_all_hospitals
    from backend.services.hybrid_index import VectorHospitalIndex
    from backend.services.legacy_tfidf_service import TfidfHospitalIndex
    _get_model()
    hospitals = load_all_hospitals()
    dense = VectorHospitalIndex()
    dense.build(hospitals)
    tfidf = TfidfHospitalIndex()
    tfidf.build(hospitals)

    queries = [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]
    report = {"label": a.label, "k": K, "queries_file": str(QUERIES.relative_to(HERE.parent.parent)),
              "n_hospitals": len(hospitals), "n_vectors": dense._store.total,
              "planner": planner_diagnostics(queries),
              "results": evaluate(queries, hospitals, make_systems(hospitals, dense, tfidf))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))

    cols = ["base_rate", "filter_only", "tfidf_raw", "dense_raw", "dense_state", "tfidf_agent", "dense_agent"]
    print(f"[{a.label}] planner: {report['planner']}")
    print(f"{'group':28} {'n':>3} " + " ".join(f"{c:>11}" for c in cols))
    for g, row in report["results"].items():
        metric = "mrr@10" if "known_item" in g else "p@10"
        vals = [row[c].get(metric, row[c].get("p@10")) for c in cols]
        print(f"{g + ' ' + metric:28} {row['n_queries']:>3} " + " ".join(f"{v:>11.3f}" for v in vals))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
