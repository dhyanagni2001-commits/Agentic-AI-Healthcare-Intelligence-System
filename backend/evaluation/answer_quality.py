"""Answer-quality (grounding) evaluation — kept separate from retrieval quality.

Retrieval can be perfect while the answer still cites hospitals or numbers
the evidence never contained. For each query this runs the RAG pipeline with
the configured LLM and scores the generated text against what was retrieved:

  unsupported_citation_rate  hospital names in the answer that are real
                             dataset names but absent from retrieved evidence
  unsupported_number_rate    numbers in the answer absent from evidence+facts
                             (percentages derived from "a/b" facts count as supported)
  out_of_state_citation_rate hospitals cited for a state-scoped query that do
                             not exist in that state (e.g. West Virginia for Virginia)
  fallback_rate              answers where the LLM failed or was not configured

These are automatic proxies for faithfulness, not human judgments. With no
LLM configured the report says `pending` instead of scoring template text.

    LLM_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL=... \\
      python -m backend.evaluation.answer_quality --out reports/answer_quality.json
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from backend.evaluation.retrieval_audit import QUERIES

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
_RATIO = re.compile(r"(\d+)\s*/\s*(\d+)")


def _cited_names(answer: str, names: Iterable[str]) -> List[str]:
    """Longest-first, non-overlapping, word-bounded name matches, so
    "West Virginia Community Hospital" is not also counted as
    "Virginia Community Hospital"."""
    text = answer.lower()
    found = []
    for n in sorted(set(names), key=len, reverse=True):
        pat = re.compile(r"(?<![\w])" + re.escape(n.lower()) + r"(?![\w])")
        if pat.search(text):
            found.append(n)
            text = pat.sub(" ", text)
    return sorted(found)


def score_answer(answer: str, evidence_text: str, all_names: Iterable[str],
                 name_states: Dict[str, set] = None, query_state: str = None) -> Dict:
    """Grounding checks for one answer. `evidence_text` is everything the model
    was shown (evidence snippets + computed facts + the question)."""
    ev = evidence_text.lower()
    cited = _cited_names(answer, all_names)
    unsupported_cites = [n for n in cited if n.lower() not in ev]
    out_of_state = [n for n in cited if query_state and name_states
                    and query_state not in name_states.get(n, set())]
    nums = _NUM.findall(re.sub(r"24\s*/\s*7", " ", answer))  # "24/7" is not a claim
    ev_nums = set(_NUM.findall(evidence_text))
    for a, b in _RATIO.findall(evidence_text):  # "7/10 have ..." supports "70%"
        if int(b):
            ev_nums.add(str(round(100 * int(a) / int(b))))
    ev_vals = {float(n) for n in ev_nums}  # "208" is supported by "208.0"
    unsupported_nums = [n for n in nums if float(n) not in ev_vals]
    return {"cited": len(cited), "unsupported_citations": unsupported_cites,
            "out_of_state_citations": out_of_state,
            "numbers": len(nums), "unsupported_numbers": unsupported_nums}


def aggregate(rows: List[Dict]) -> Dict:
    scored = [r for r in rows if not r["fallback"]]
    cites = sum(r["cited"] for r in scored)
    nums = sum(r["numbers"] for r in scored)
    return {
        "n_queries": len(rows),
        "fallback_rate": round(1 - len(scored) / len(rows), 3) if rows else None,
        "unsupported_citation_rate": round(sum(len(r["unsupported_citations"]) for r in scored) / cites, 3) if cites else None,
        "out_of_state_citation_rate": round(sum(len(r.get("out_of_state_citations", [])) for r in scored) / cites, 3) if cites else None,
        "unsupported_number_rate": round(sum(len(r["unsupported_numbers"]) for r in scored) / nums, 3) if nums else None,
        "total_citations": cites,
        "total_numbers": nums,
        "answers_with_any_issue": sum(bool(r["unsupported_citations"] or r["unsupported_numbers"]
                                           or r.get("out_of_state_citations")) for r in scored),
    }


def main():
    ap = argparse.ArgumentParser(description="Answer grounding evaluation")
    ap.add_argument("--out", type=Path, default=Path("reports/answer_quality.json"))
    ap.add_argument("--limit", type=int, default=30)
    a = ap.parse_args()

    from backend.services.llm_service import provider_info
    info = provider_info()
    report: Dict = {"llm": info, "temperature": os.environ.get("LLM_TEMPERATURE", "server default")}
    if not info["configured"]:
        report["status"] = "pending: no LLM configured; set LLM_PROVIDER (e.g. vllm) and rerun"
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(report["status"])
        return

    from backend.services.embedding_service import _get_model
    from backend.services.data_loader import load_all_hospitals
    from backend.services.hybrid_index import VectorHospitalIndex
    from backend.services.rag_pipeline import _retrieve_and_build_prompt, run_rag
    _get_model()
    hospitals = load_all_hospitals()
    index = VectorHospitalIndex()
    index.build(hospitals)
    names = {h.facility_name for h in hospitals.values() if h.facility_name}
    name_states: Dict[str, set] = {}
    for h in hospitals.values():
        name_states.setdefault(h.facility_name, set()).add((h.state or "").upper())

    qs = [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]
    qs = [q for q in qs if q["split"] == "test" and q["label"] != "known_item"][:a.limit]
    rows = []
    for q in qs:
        res = run_rag(q["query"], index, top_k=10)
        *_, prompt = _retrieve_and_build_prompt(q["query"], index, 10, None, None, None)
        fallback = res.answer.startswith("LLM is ")
        rows.append({"id": q["id"], "query": q["query"], "fallback": fallback,
                     **score_answer(res.answer, prompt, names, name_states, q.get("state")),
                     "answer": res.answer})
    report.update({"status": "scored", "summary": aggregate(rows), "rows": rows})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
