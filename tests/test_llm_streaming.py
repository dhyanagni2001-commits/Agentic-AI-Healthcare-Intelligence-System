"""vLLM provider, failure handling, streaming, and source preservation.

The provider tests talk HTTP to benchmarks/mock_openai_server.py (an
OpenAI-compatible stub), so they exercise the real `openai` SDK request,
SSE parsing, timeout, and error paths — not a patched function. No model
or GPU is involved.
"""
from __future__ import annotations
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.mock_openai_server import MockConfig, start_in_thread
from backend.services import llm_service
from backend.services.llm_service import (
    VLLMProvider, LLMNotConfiguredError, LLMUnavailableError, get_provider, reset_provider,
)
from backend.services.rag_pipeline import stream_rag
from tests.test_embedding_retrieval import _build_synthetic_index

FAST_FAIL_ENV = {"LLM_TIMEOUT_S": "0.5", "LLM_MAX_RETRIES": "0"}


class _MockServerCase(unittest.TestCase):
    cfg = MockConfig(tokens=6)

    def setUp(self):
        self.server, self.base_url = start_in_thread(self.cfg)
        self.env = patch.dict(os.environ, FAST_FAIL_ENV)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()

    def provider(self) -> VLLMProvider:
        return VLLMProvider(base_url=self.base_url, model="mock-model")


class TestVLLMConfig(unittest.TestCase):
    def tearDown(self):
        reset_provider()

    def test_missing_endpoint_is_not_configured(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "vllm"}, clear=False):
            os.environ.pop("VLLM_BASE_URL", None)
            reset_provider()
            with self.assertRaises(LLMNotConfiguredError):
                get_provider()

    def test_env_selects_vllm(self):
        env = {"LLM_PROVIDER": "vllm", "VLLM_BASE_URL": "http://x:8001/v1", "VLLM_MODEL": "m"}
        with patch.dict(os.environ, env):
            reset_provider()
            p = get_provider()
            self.assertEqual((p.name, p.model, p.base_url), ("vllm", "m", "http://x:8001/v1"))

    def test_unknown_and_none_providers_degrade_instead_of_crashing(self):
        for name in ("none", "does-not-exist"):
            with patch.dict(os.environ, {"LLM_PROVIDER": name}):
                reset_provider()
                info = llm_service.provider_info()
                self.assertFalse(info["configured"], name)


class TestVLLMProviderHappyPath(_MockServerCase):
    def test_generate(self):
        self.assertEqual(self.provider().generate("hi"), "".join(f"tok{i} " for i in range(6)))

    def test_stream_yields_incremental_chunks(self):
        chunks = list(self.provider().stream("hi", system="sys"))
        self.assertEqual(len(chunks), 6)
        self.assertEqual(chunks[0], "tok0 ")


class TestVLLMTimeout(_MockServerCase):
    cfg = MockConfig(hang_s=2.0)

    def test_timeout_raises_unavailable(self):
        with self.assertRaises(LLMUnavailableError):
            self.provider().generate("hi")


class TestVLLMUpstreamError(_MockServerCase):
    cfg = MockConfig(fail_status=500)

    def test_5xx_raises_unavailable(self):
        with self.assertRaises(LLMUnavailableError):
            list(self.provider().stream("hi"))


class TestVLLMMidStreamDrop(_MockServerCase):
    cfg = MockConfig(tokens=6, drop_after=2)

    def test_partial_stream_then_unavailable(self):
        got = []
        with self.assertRaises(LLMUnavailableError):
            for c in self.provider().stream("hi"):
                got.append(c)
        self.assertEqual(len(got), 2)


class TestConnectionRefused(unittest.TestCase):
    def test_unreachable_server_raises_unavailable(self):
        with patch.dict(os.environ, FAST_FAIL_ENV):
            p = VLLMProvider(base_url="http://127.0.0.1:9/v1", model="m")
            with self.assertRaises(LLMUnavailableError):
                p.generate("hi")


# ── stream_rag: event order and source preservation ──────────────────────────

class _FakeProvider:
    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after = chunks, fail_after

    def stream(self, prompt, system=None):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise LLMUnavailableError("boom")
            yield c


class TestStreamRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = _build_synthetic_index()

    def _events(self, provider):
        with patch.object(llm_service, "get_provider", return_value=provider):
            return list(stream_rag("pediatric hospital in los angeles", self.idx, top_k=3))

    def test_sources_first_then_tokens_then_done(self):
        evs = self._events(_FakeProvider(["a", "b", "c"]))
        names = [e["event"] for e in evs]
        self.assertEqual(names, ["sources", "token", "token", "token", "done"])
        self.assertEqual("".join(e["data"]["text"] for e in evs if e["event"] == "token"), "abc")

    def test_done_repeats_exact_source_ids(self):
        evs = self._events(_FakeProvider(["x"]))
        sources = [d["id"] for d in evs[0]["data"]["documents"]]
        self.assertTrue(sources)
        self.assertEqual(evs[-1]["data"]["source_ids"], sources)
        self.assertFalse(evs[-1]["data"]["fallback"])
        self.assertIsNotNone(evs[-1]["data"]["first_token_ms"])

    def test_mid_stream_failure_keeps_sources_and_partial_text(self):
        evs = self._events(_FakeProvider(["a", "b", "c"], fail_after=1))
        names = [e["event"] for e in evs]
        self.assertEqual(names[:2], ["sources", "token"])
        self.assertIn("error", names)
        err = next(e for e in evs if e["event"] == "error")["data"]
        self.assertTrue(err["partial"])
        self.assertEqual(evs[-1]["data"]["fallback_reason"], "llm_unavailable")
        self.assertEqual(evs[-1]["data"]["source_ids"],
                         [d["id"] for d in evs[0]["data"]["documents"]])
        self.assertIn("Computed facts", evs[-2]["data"]["text"])

    def test_not_configured_streams_fact_fallback(self):
        with patch.object(llm_service, "get_provider", side_effect=LLMNotConfiguredError("no")):
            evs = list(stream_rag("cardiac", self.idx, top_k=2))
        self.assertEqual(evs[0]["event"], "sources")
        self.assertIn("LLM is not configured", evs[-2]["data"]["text"])
        self.assertNotIn("error", [e["event"] for e in evs])  # deliberate config, not a failure
        self.assertEqual(evs[-1]["data"]["fallback_reason"], "llm_not_configured")


# ── API: /query/stream SSE, /query fallback, /ready ──────────────────────────

def _parse_sse(text: str):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


class TestStreamingAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_main_fastapi import _make_client
        cls.client = _make_client()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_sse_stream_with_sources(self):
        with patch.object(llm_service, "get_provider", return_value=_FakeProvider(["Hi ", "there"])):
            r = self.client.post("/query/stream", json={"query": "cardiac care", "max_results": 3})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        evs = _parse_sse(r.text)
        self.assertEqual(evs[0][0], "sources")
        self.assertEqual(evs[-1][0], "done")
        self.assertEqual(evs[-1][1]["source_ids"], [d["id"] for d in evs[0][1]["documents"]])

    def test_sse_rejects_empty_query(self):
        self.assertEqual(self.client.post("/query/stream", json={"query": "  "}).status_code, 400)

    def test_query_falls_back_when_llm_times_out(self):
        """Regression: a provider timeout used to escape _llm_answer and 500."""
        with patch("backend.agents.healthcare_agent.llm_generate",
                   side_effect=LLMUnavailableError("timeout")):
            r = self.client.post("/query", json={"query": "hospitals in TX"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["answer"])

    def test_ready_reports_index_and_llm(self):
        r = self.client.get("/ready")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ready"])
        self.assertIn("llm", r.json())


if __name__ == "__main__":
    unittest.main()


# ── Deployment smoke check + benchmark harness self-checks ───────────────────

class TestDeploySmokeCheck(unittest.TestCase):
    """Runs deploy/smoke_test.py's checks in-process against the app."""

    def test_smoke_checks_pass_without_llm(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy"))
        from smoke_test import run_checks
        from tests.test_main_fastapi import _make_client
        client = _make_client()
        try:
            with patch.dict(os.environ, {"LLM_PROVIDER": "none"}):
                reset_provider()
                self.assertEqual(len(run_checks(client)), 4)
                with self.assertRaises(AssertionError):
                    run_checks(client, expect_llm=True)
        finally:
            reset_provider()
            client.__exit__(None, None, None)


class TestBenchHarness(unittest.TestCase):
    """The harness must measure a known synthetic latency and count failures."""

    def _run(self, cfg, tmp):
        from benchmarks import bench_serving
        server, url = start_in_thread(cfg)
        try:
            return bench_serving.main(["--target", "llm", "--url", url, "--concurrency", "2",
                                       "--num-requests", "4", "--warmup", "0", "--max-tokens", "8",
                                       "--timeout-s", "5", "--out-dir", tmp, "--label", "t"])
        finally:
            server.shutdown()
            server.server_close()

    def test_measures_known_token_delay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            lv = self._run(MockConfig(tokens=8, token_delay_ms=20), tmp)["levels"][0]
        self.assertEqual(lv["succeeded"], 4)
        self.assertGreaterEqual(lv["itl_ms"]["p50"], 18)
        self.assertLess(lv["itl_ms"]["p50"], 60)
        self.assertEqual(lv["output_tokens_total"], 32)

    def test_counts_http_errors(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rep = self._run(MockConfig(fail_status=503), tmp)
        lv = rep["levels"][0]
        self.assertEqual(lv["error_rate"], 1.0)
        self.assertEqual(lv["errors"], {"HTTP 503": 4})
        self.assertIn("hardware", rep)
        self.assertEqual(rep["workload"]["num_prompts"], 24)


class TestPlannerStateNotCity(unittest.TestCase):
    """Regression: 'in California' was parsed as city=California, which matched
    no hospital and bypassed semantic retrieval (see docs/RETRIEVAL_AUDIT.md)."""

    def test_state_name_is_not_a_city(self):
        from backend.agents.healthcare_agent import AgentState, query_planner_agent
        p = query_planner_agent(AgentState(query="Pediatric hospitals in California"))
        self.assertEqual((p["state_filter"], p["city_filter"]), ("CA", None))

    def test_real_city_still_parsed(self):
        from backend.agents.healthcare_agent import AgentState, query_planner_agent
        p = query_planner_agent(AgentState(query="Find hospitals in Houston, TX"))
        self.assertEqual((p["state_filter"], p["city_filter"]), ("TX", "Houston"))

    def test_er_keyword_needs_word_boundary(self):
        from backend.agents.healthcare_agent import AgentState, query_planner_agent
        self.assertEqual(query_planner_agent(AgentState(
            query="Which hospitals in Ohio offer pediatric care?"))["cap_filter"], "pediatrics")
        self.assertEqual(query_planner_agent(AgentState(
            query="Nearest ER in Texas"))["cap_filter"], "emergency_services")


class TestAnswerQualityScorer(unittest.TestCase):
    def test_flags_unsupported_citation_and_number(self):
        from backend.evaluation.answer_quality import score_answer, aggregate
        names = ["Houston Emergency Center", "LA Children's Hospital"]
        evidence = "1. [hospital_profile] houston emergency center has 12 doctors"
        good = score_answer("Houston Emergency Center has 12 doctors.", evidence, names)
        bad = score_answer("LA Children's Hospital has 40 beds.", evidence, names)
        self.assertEqual((good["unsupported_citations"], good["unsupported_numbers"]), ([], []))
        self.assertEqual(bad["unsupported_citations"], ["LA Children's Hospital"])
        self.assertEqual(bad["unsupported_numbers"], ["40"])
        agg = aggregate([{**good, "fallback": False}, {**bad, "fallback": False},
                         {"fallback": True}])
        self.assertEqual(agg["out_of_state_citation_rate"], 0.0)
        self.assertEqual(agg["unsupported_citation_rate"], 0.5)
        self.assertAlmostEqual(agg["fallback_rate"], 0.333, places=3)

    def test_derived_percent_24_7_and_overlapping_names(self):
        from backend.evaluation.answer_quality import score_answer
        names = ["Virginia Community Hospital", "West Virginia Community Hospital"]
        states = {"Virginia Community Hospital": {"VA"}, "West Virginia Community Hospital": {"WV"}}
        evidence = "west virginia community hospital ... - 7/10 have emergency services. avg 208.0"
        r = score_answer("West Virginia Community Hospital is open 24/7; 70% have an ER; avg 208.",
                         evidence, names, states, "VA")
        self.assertEqual(r["unsupported_numbers"], [])
        self.assertEqual(r["unsupported_citations"], [])
        self.assertEqual(r["out_of_state_citations"], ["West Virginia Community Hospital"])
        self.assertEqual(r["cited"], 1)
