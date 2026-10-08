# Retrieval Audit: the "Precision@10 0.39 → 0.96" Claim

**Verdict:** the claim can't be reproduced as stated, and the metric behind it doesn't measure retrieval quality.

- After two planner bugs found here are fixed, the 0.96 figure is matched on held-out canonical-keyword queries.
- The labels are the same capability field that the reranker boosts, and a plain database filter with no retrieval scores 1.00.
- Dense retrieval does improve on TF-IDF in non-circular tests: known-item MRR@10 is 0.94 vs 0.24 (both on query text, no filters). On hospital-type queries it doesn't: TF-IDF is level with it.

Reproduce: `python -m backend.evaluation.retrieval_audit --out reports/retrieval_audit_fixed.json`. It takes about 1 minute once the FAISS index exists. Raw results are in `reports/retrieval_audit_as_shipped.json` and `reports/retrieval_audit_fixed.json`.

## What was wrong with the original evaluation

1. **The script was never committed.** `.gitignore` lists `backend/evaluation/results/`, but no evaluation code exists in git history. The 8 queries and the scoring code are unknown, so the dev rows below are a *reconstruction* from the README's description.
2. **The labels are circular.** A result counts as relevant if `state` matches and `capabilities.<cap>` is true. The capability-aware reranker multiplies the score by 1.3 when that *same field* is true. The metric rewards reading the label.
3. **The base rate is high.** 63–83% of hospitals carry each capability flag, because the flags come from broad keyword rules on hospital type and departments. Ten random in-state hospitals already score P@10 ≈ 0.69.
4. **The baseline is too weak.** The TF-IDF baseline scores *below* random-in-state (0.15–0.25 without filters). A 0.39 → 0.96 jump mostly measures "starts using the state filter".
5. **Too small, no held-out set, no CIs.** It used 8 queries with no confidence intervals and no separation between tuning and test queries.

## Bugs the audit found (fixed in `backend/agents/healthcare_agent.py`)

| Bug | Effect | Evidence |
|---|---|---|
| The planner treats "in California" as `city_filter="California"` | No hospital matches the city, so retrieval falls back to the **first 10 in-state rows in CSV order** and ranking is skipped entirely | 84/84 state-scoped audit queries affected; production P@10 0.65, below the 0.69 base rate |
| The emergency keyword `"er "` matches inside "off**er** ", "cent**er** " | The wrong capability gets boosted | `filter_only` 0.83 → 1.00 on canonical queries after the fix |
| The planner checks "virginia" before "west virginia" | West Virginia queries are scoped to VA | Found in review; no audit query names West Virginia, so no metric changes |
| The RAG/streaming path ignores the query's location (fixed in `rag_pipeline.py` and `hybrid_index.py`) | Out-of-state evidence reaches the LLM prompt | 7/198 wrong-state citations → 0/195 ([below](#answer-quality-kept-separate-from-retrieval)) |

Regression tests: `tests/test_llm_streaming.py::TestPlannerStateNotCity` and the state-scoping tests in `tests/test_rag_pipeline.py`.

## Held-out query set

`backend/evaluation/retrieval_queries.jsonl` is frozen and generated with seed `20261007`. Nothing in the system was tuned on the `test` split.

| Group | n | Relevance label | Independent of the boost? |
|---|---:|---|---|
| dev/field | 8 | state + capability field (reconstructed README protocol) | no |
| test/field/canon | 24 | same, planner keywords ("pediatric", "oncology") | no |
| test/field/para | 24 | same, paraphrases the planner lacks ("kids and infants", "chemotherapy") | no |
| test/hospital_type | 16 | state + `hospital_type` | yes |
| test/ownership | 12 | state + `ownership` | yes |
| test/known_item | 40 | the one facility named in the query | yes |

## Results (after the planner fixes)

P@10 unless noted; 95% bootstrap CI over queries in brackets. Systems:
- `base` is the expected score of random in-state hospitals.
- `filter` is a lookup with no retrieval.
- `TF-IDF, no filters` and `dense, no filters` search the query text only (`tfidf_raw`, `dense_raw`).
- `dense+state` is FAISS with the state filter and no boost.
- `TF-IDF agent` and **`dense agent`** run the full planner path (planner → retriever with state/city filters → capability boost). `dense agent` is production.

| Group | base | filter | TF-IDF, no filters | TF-IDF agent | dense, no filters | dense+state | **dense agent** |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev/field | 0.69 | 1.00 | 0.20 | 0.58 | 0.90 | 0.93 | **1.00** |
| test/field/canon | 0.69 | 1.00 | 0.25 [0.14, 0.36] | 0.47 [0.32, 0.63] | 0.90 [0.86, 0.95] | 0.92 [0.88, 0.96] | **1.00** |
| test/field/para | 0.71 | 0.62 | 0.15 [0.08, 0.25] | 0.20 [0.11, 0.32] | 0.86 [0.80, 0.92] | 0.90 [0.84, 0.95] | **0.90** [0.84, 0.95] |
| test/hospital_type | 0.17 | 0.23 | 0.43 [0.36, 0.51] | 0.47 [0.37, 0.57] | 0.45 [0.31, 0.60] | 0.45 | **0.44** [0.29, 0.60] |
| test/ownership | 0.16 | 0.14 | 0.16 | 0.20 | 0.19 [0.12, 0.27] | 0.19 | **0.19** [0.12, 0.27] |
| test/known_item (MRR@10) | ≈0 | 0.02 | 0.24 [0.14, 0.36] | 0.43 [0.31, 0.55] | 0.94 [0.88, 0.99] | 0.97 | **0.97** [0.93, 1.00] |

As shipped (before the fixes), the production agent scored 0.65 on dev, 0.63 on test/canon and 0.61 on test/para. That is roughly the base rate, because ranking was being bypassed.

## Interpretation

- **The capability boost adds nothing measurable beyond structured filtering.** On canonical queries, agent and filter both score 1.00. On paraphrases the planner misses the capability, so the boost never fires and agent = dense+state = 0.90.
- **Semantic retrieval does help where labels are independent.**
  - Known-item MRR@10: 0.94 for dense vs 0.24 for TF-IDF, both with no filters (0.97 vs 0.43 through the agent path).
  - Hospital-type P@10: dense reaches 0.45 vs a 0.17 base rate, but TF-IDF does just as well (0.43 with no filters, 0.47 as an agent; all CIs overlap). The type is spelled out in the document text, so lexical matching is enough.
  - Ownership is at chance (0.19 vs 0.16), since that attribute is barely represented in the embedded text.
- **An honest replacement claim:** "On 40 held-out known-item queries, dense retrieval reached MRR@10 0.94 vs 0.24 for TF-IDF (query text only, no filters)". Report capability-label precision only next to its 0.69 base rate.

## Answer quality, kept separate from retrieval

`backend/evaluation/answer_quality.py` scores generated answers against what was retrieved. It counts hospital names cited but absent from evidence, numbers absent from evidence and facts, and the fallback rate. It's an automatic faithfulness proxy, not a human judgment.

**Status: scored with a local model, after the RAG state-filter fix.** The model was Qwen2.5-1.5B-Instruct Q4_K_M on llama.cpp (Apple M5 Pro), at temperature 0, over 30 held-out queries. Two runs produced identical answers (30/30). Raw output, including every full answer, is in `reports/answer_quality_llamacpp_qwen1.5b.json`.

```bash
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --port 8001 --alias qwen2.5-1.5b-instruct-q4_k_m
LLM_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL=qwen2.5-1.5b-instruct-q4_k_m LLM_TEMPERATURE=0 \
  python -m backend.evaluation.answer_quality --out reports/answer_quality_llamacpp_qwen1.5b.json
```

| Metric | Before the fix | After the fix |
|---|---|---|
| Fallback rate | 0 / 30 | 0 / 30 |
| Cited hospital not in the retrieved evidence | 0 / 198 citations | 0 / 195 citations |
| Cited hospital in the wrong state | 7 / 198 (3.5%), in 3 answers | **0 / 195** |
| Number not supported by evidence or facts | 8 / 279 (2.9%) | 7 / 260 (2.7%) |
| Answers with any issue | 7 / 30 | 5 / 30 |

**The fix.** Before, `run_rag` and `/query/stream` never applied the planner's location, and even an explicit `state_filter` only narrowed the hospital list, not the evidence documents in the prompt. So "Manhattan Healthcare" (Manhattan, KS) and "York Healthcare" (York, PA) reached answers about New York, and three West Virginia hospitals reached an answer about Virginia, where the model presented them as in-state. Now `search_documents` filters documents by location, and the RAG path parses the state/city from the query the same way the planner does (`parse_location`). The `/query` evidence panel uses the same filter. Regression tests: `tests/test_rag_pipeline.py::test_state_named_in_query_scopes_evidence` and `test_state_filter_scopes_evidence_in_prompt`.

**Remaining flags, all checked by hand and all genuine:** every one is a "collectively N doctors" sum that doesn't match the numbers listed in the same answer. For example, a California answer lists counts summing to 181 but claims 128 and 218, and a Virginia answer lists 68 but claims 30.

**Not caught automatically:**
- Reading a fact under the wrong label, e.g. presenting the overall average doctor count (352.7) as a per-hospital oncology figure.
- Listing the same hospital twice (a Maryland answer repeats two entries).

The scorer only checks that names and numbers exist in the evidence, not that they're used correctly, so these rates are a lower bound. A 1.5B 4-bit model is also much weaker than what you'd serve in production.

## Data caveat

`data/all_us_hospitals.csv` looks synthetic:
- Names like "Mount Jessica Memorial Hospital".
- A Houston address with ZIP 31287 (a Georgia prefix) and "San Antonio County".
- Hospital types spread almost uniformly (850–920 each), whereas real CMS data is dominated by acute-care hospitals.

All metrics here describe this dataset, not real US hospitals. Confirm the provenance before describing it as "public hospital records".

## Limitations

- Labels are rule-derived, not human relevance judgments, and the known-item queries use exact names.
- The query set is small: 124 queries, with 12–40 per group, so CIs are wide.
- Only `all-MiniLM-L6-v2` was evaluated, on one dataset.
