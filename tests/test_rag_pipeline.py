"""RAG pipeline tests with a mocked LLMProvider — no API key required."""
from __future__ import annotations
import sys, os, unittest
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_embedding_retrieval import _build_synthetic_index
from backend.services.rag_pipeline import run_rag
from backend.services.llm_service import LLMNotConfiguredError


class TestRAGPipelineMockedLLM(unittest.TestCase):

    def setUp(self):
        self.idx = _build_synthetic_index()

    @patch("backend.services.rag_pipeline.llm_generate")
    def test_grounded_answer_uses_llm(self, mock_generate):
        mock_generate.return_value = "Houston Emergency Center has ICU and ER capabilities."
        result = run_rag("hospital with icu and emergency care", self.idx, top_k=3)

        mock_generate.assert_called_once()
        self.assertEqual(result.answer, "Houston Emergency Center has ICU and ER capabilities.")
        self.assertGreater(len(result.retrieved_documents), 0)
        self.assertGreater(result.confidence, 0.5)

    @patch("backend.services.rag_pipeline.llm_generate")
    def test_evidence_passed_to_prompt(self, mock_generate):
        mock_generate.return_value = "answer"
        run_rag("cardiac care emergency", self.idx, top_k=3)
        prompt_arg = mock_generate.call_args.args[0]
        self.assertIn("Retrieved evidence", prompt_arg)
        self.assertIn("Computed facts", prompt_arg)

    @patch("backend.services.rag_pipeline.llm_generate")
    def test_falls_back_when_llm_not_configured(self, mock_generate):
        mock_generate.side_effect = LLMNotConfiguredError("no key")
        result = run_rag("pediatrics", self.idx, top_k=3)
        self.assertIn("LLM is not configured", result.answer)
        self.assertLess(result.confidence, 0.5)

    @patch("backend.services.rag_pipeline.llm_generate")
    def test_state_filter_narrows_hospitals(self, mock_generate):
        mock_generate.return_value = "answer"
        result = run_rag("hospital", self.idx, top_k=5, state_filter="CA")
        self.assertTrue(all(h.state == "CA" for h in result.retrieved_hospitals))

    @patch("backend.services.rag_pipeline.llm_generate")
    def test_state_filter_scopes_evidence_in_prompt(self, mock_generate):
        mock_generate.return_value = "answer"
        result = run_rag("cardiac care emergency", self.idx, top_k=3, state_filter="TX")
        self.assertEqual({d["metadata"]["facility_id"] for d in result.retrieved_documents}, {"TX001"})
        self.assertNotIn("nyc cardiac institute", mock_generate.call_args.args[0])

    @patch("backend.services.rag_pipeline.llm_generate")
    def test_state_named_in_query_scopes_evidence(self, mock_generate):
        # Regression: York, PA reached answers about New York because the RAG
        # path never applied the planner's location (docs/RETRIEVAL_AUDIT.md).
        mock_generate.return_value = "answer"
        result = run_rag("cardiac care hospitals in Texas", self.idx, top_k=3)
        self.assertEqual({d["metadata"]["facility_id"] for d in result.retrieved_documents}, {"TX001"})
        self.assertTrue(all(h.state == "TX" for h in result.retrieved_hospitals))

    @patch("backend.services.rag_pipeline.llm_generate")
    def test_to_dict_serializable(self, mock_generate):
        mock_generate.return_value = "answer"
        result = run_rag("hospital", self.idx, top_k=3)
        d = result.to_dict()
        self.assertIn("retrieved_documents", d)
        self.assertIn("retrieved_hospitals", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
