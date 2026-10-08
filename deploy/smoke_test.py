"""Post-deploy smoke check for a running HealthIQ API (compose, kind, or GPU cluster).

    python deploy/smoke_test.py http://localhost:8000              # LLM optional
    python deploy/smoke_test.py http://localhost:8000 --expect-llm # fail if answers fell back

Checks liveness, readiness, /query, and /query/stream event order with
source preservation. Exits non-zero on the first failed check.
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import Any, List, Tuple

import httpx

QUERY = {"query": "pediatric hospitals in California", "max_results": 3}


def parse_sse(text: str) -> List[Tuple[str, Any]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((fields.get("event", "message"), json.loads(fields.get("data", "null"))))
    return events


def run_checks(client, expect_llm: bool = False) -> List[str]:
    """Returns passed-check descriptions; raises AssertionError on failure.
    `client` is anything with httpx-style get/post (httpx.Client, TestClient)."""
    passed = []

    r = client.get("/health")
    assert r.status_code == 200, f"/health -> {r.status_code}"
    passed.append("liveness /health 200")

    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["ready"], f"/ready -> {r.status_code} {r.text[:200]}"
    llm = r.json()["llm"]
    passed.append(f"readiness /ready 200, {r.json()['hospitals_loaded']} hospitals, llm={llm}")
    if expect_llm:
        assert llm["configured"], f"LLM expected but not configured: {llm}"

    r = client.post("/query", json=QUERY)
    assert r.status_code == 200 and r.json()["answer"], f"/query -> {r.status_code}"
    passed.append(f"/query 200, {len(r.json()['hospitals_referenced'])} hospitals referenced")

    r = client.post("/query/stream", json=QUERY)
    assert r.status_code == 200, f"/query/stream -> {r.status_code}"
    assert r.headers["content-type"].startswith("text/event-stream")
    evs = parse_sse(r.text)
    names = [e for e, _ in evs]
    assert names[0] == "sources" and names[-1] == "done", f"bad event order: {names}"
    assert "token" in names, "no token events"
    src = [d["id"] for d in evs[0][1]["documents"]]
    done = evs[-1][1]
    assert src and done["source_ids"] == src, "source ids not preserved through stream"
    if expect_llm:
        assert not done["fallback"], f"LLM fell back: {done.get('fallback_reason')}"
    passed.append(f"/query/stream sources->{names.count('token')} tokens->done, "
                  f"{len(src)} sources preserved, fallback={done['fallback']}, "
                  f"retrieval_ms={done.get('retrieval_ms')}, first_token_ms={done.get('first_token_ms')}")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url")
    ap.add_argument("--expect-llm", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=120)
    a = ap.parse_args()
    with httpx.Client(base_url=a.url.rstrip("/"), timeout=a.timeout_s) as client:
        try:
            for line in run_checks(client, a.expect_llm):
                print("PASS", line)
        except (AssertionError, httpx.HTTPError) as e:
            print("FAIL", e)
            return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
