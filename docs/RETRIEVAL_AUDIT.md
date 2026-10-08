# Retrieval Audit: the "Precision@10 0.39 → 0.96" Claim

**Verdict:** the claim can't be reproduced as stated, and the metric behind it doesn't measure retrieval quality.

- After two planner bugs found here are fixed, the 0.96 figure is matched on held-out canonical-keyword queries.
- The labels are the same capability field that the reranker boosts, and a plain database filter with no retrieval scores 1.00.
- Dense retrieval does improve on TF-IDF in non-circular tests: known-item MRR@10 is 0.94 vs 0.24.

Reproduce: `python -m backend.evaluation.retrieval_audit --out reports/retrieval_audit_fixed.json`. It takes about 1 minute once the FAISS index exists. Raw results are in `reports/retrieval_audit_as_shipped.json` and `reports/retrieval_audit_fixed.json`.

## What was wrong with the original evaluation

1. **The script was never committed.** `.gitignore` lists `backend/evaluation/results/`, but no evaluation code exists in git history. The 8 queries and the scoring code are unknown, so the dev rows below are a *reconstruction* from the README's description.
2. **The labels are circular.** A result counts as relevant if `state` matches and `capabilities.<cap>` is true. The capability-aware reranker multiplies the score by 1.3 when that *same field* is true. The metric rewards reading the label.
3. **The base rate is high.** 63–83% of hospitals carry each capability flag, because the flags come from broad keyword rules on hospital type and departments. Ten random in-state hospitals already score P@10 ≈ 0.69.
4. **The baseline is too weak.** The TF-IDF baseline scores *below* random-in-state (0.20–0.25 without filters). A 0.39 → 0.96 jump mostly measures "starts using the state filter".
5. **Too small, no held-out set, no CIs.** It used 8 queries with no confidence intervals and no separation between tuning and test queries.

## Bugs the audit found (fixed in `backend/agents/healthcare_agent.py`)

| Bug | Effect | Evidence |
|---|---|---|
| The planner treats "in California" as `city_filter="California"` | No hospital matches the city, so retrieval falls back to the **first 10 in-state rows in CSV order** and ranking is skipped entirely | 84/84 state-scoped audit queries affected; production P@10 0.65, below the 0.69 base rate |
| The emergency keyword `"er "` matches inside "off**er** ", "cent**er** " | The wrong capability gets boosted | `filter_only` 0.83 → 1.00 on canonical queries after the fix |

Regression tests: `tests/test_llm_streaming.py::TestPlannerStateNotCity`.

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
- `dense+state` is FAISS with the state filter and no boost.
- `agent` is the production path (planner → FAISS → capability boost).

| Group | base | filter | TF-IDF agent | dense, no filters | dense+state | **agent** |
|---|---:|---:|---:|---:|---:|---:|
| dev/field | 0.69 | 1.00 | 0.58 | 0.90 | 0.93 | **1.00** |
| test/field/canon | 0.69 | 1.00 | 0.47 [0.32, 0.63] | 0.90 [0.86, 0.95] | 0.92 [0.88, 0.96] | **1.00** |
| test/field/para | 0.71 | 0.62 | 0.20 [0.11, 0.32] | 0.86 [0.80, 0.92] | 0.90 [0.84, 0.95] | **0.90** [0.84, 0.95] |
| test/hospital_type | 0.17 | 0.23 | 0.47 [0.37, 0.57] | 0.45 [0.31, 0.60] | 0.45 | **0.44** [0.29, 0.60] |
| test/ownership | 0.16 | 0.14 | 0.20 | 0.19 [0.12, 0.27] | 0.19 | **0.19** [0.12, 0.27] |
| test/known_item (MRR@10) | ≈0 | 0.02 | 0.43 [0.31, 0.55] | 0.94 [0.88, 0.99] | 0.97 | **0.97** [0.93, 1.00] |

As shipped (before the fixes), the production agent scored 0.65 on dev, 0.63 on test/canon and 0.61 on test/para. That is roughly the base rate, because ranking was being bypassed.

## Interpretation

- **The capability boost adds nothing measurable beyond structured filtering.** On canonical queries, agent and filter both score 1.00. On paraphrases the planner misses the capability, so the boost never fires and agent = dense+state = 0.90.
- **Semantic retrieval does help where labels are independent.**
  - Known-item MRR@10: 0.94 for dense vs 0.24 for raw TF-IDF.
  - Hospital-type P@10: 0.45 vs a 0.17 base rate.
  - Ownership is at chance (0.19 vs 0.16), since that attribute is barely represented in the embedded text.
- **An honest replacement claim:** "On 40 held-out known-item queries, dense retrieval reached MRR@10 0.94 vs 0.24 for TF-IDF". Report capability-label precision only next to its 0.69 base rate.

## Answer quality, kept separate from retrieval

`backend/evaluation/answer_quality.py` scores generated answers against what was retrieved. It counts hospital names cited but absent from evidence, numbers absent from evidence and facts, and the fallback rate. It's an automatic faithfulness proxy, not a human judgment.

**Status: scored with a local model.** The model was Qwen2.5-1.5B-Instruct Q4_K_M on llama.cpp (Apple M5 Pro), at temperature 0, over 30 held-out queries. Two runs produced identical answers (30/30). Raw output, including every full answer, is in `reports/answer_quality_llamacpp_qwen1.5b.json`.

```bash
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --port 8001 --alias qwen2.5-1.5b-instruct-q4_k_m
LLM_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL=qwen2.5-1.5b-instruct-q4_k_m LLM_TEMPERATURE=0 \
  python -m backend.evaluation.answer_quality --out reports/answer_quality_llamacpp_qwen1.5b.json
```

| Metric | Result |
|---|---|
| Fallback rate | 0 / 30 |
| Cited hospital not in the retrieved evidence | 0 / 198 citations |
| Cited hospital in the wrong state | 7 / 198 (3.5%), in 3 answers |
| Number not supported by evidence or facts | 8 / 279 (2.9%) |
| Answers with any issue | 7 / 30 |

Every flag was checked by hand, and all are genuine:
- **Wrong state:** "Manhattan Healthcare" (Manhattan, KS) and "York Healthcare" (York, PA) appear in answers about New York. Three West Virginia hospitals appear in an answer about Virginia. They came from retrieval: `run_rag` and `/query/stream` don't apply the planner's state filter, so word-similar hospitals from other states reach the prompt. The model then presents them as in-state.
- **Wrong arithmetic:** 5 "collectively N doctors" sums don't match the numbers listed in the same answer (e.g. 128 claimed vs 181 listed). There is also a "45/208 = 21.5%" ratio that mixes unrelated quantities.

**Not caught automatically:** reading a fact under the wrong label, e.g. presenting "average doctor count 255.7" as an oncology-department figure. The scorer only checks that names and numbers exist in the evidence, not that they're used correctly, so these rates are a lower bound. A 1.5B 4-bit model is also much weaker than what you'd serve in production.

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
