# Architecture

```
                ┌───────────── HealthIQ API (FastAPI, CPU) ────────────────┐
 client ──HTTP──▶ /query         LangGraph agent: planner → retrieval →     │
                │                 validation → gaps → recommendations → LLM │
 client ──SSE───▶ /query/stream  RAG: embed → FAISS → prompt → LLM stream  │
                │ /health /ready                                           │
                │   │ all-MiniLM-L6-v2 (384-d)   FAISS IndexFlatIP 257,700 │
                └───┼──────────────────────────────────────────┬──────────┘
                    │ data/ (CSV + persisted index)            │ llm_service
                    ▼                                          ▼
              PVC / bind mount              LLM_PROVIDER = vllm | gemini | grok | none
                                            vllm → vLLM server (OpenAI API, 1 GPU)
```

## LLM provider layer (`backend/services/llm_service.py`)

- `LLMProvider` has two methods, `generate()` and `stream()`. The default `stream()` yields one chunk, so every provider works with the streaming endpoint.
- `OpenAICompatibleProvider` is shared by **vLLM** and **Grok**. It reuses the existing `openai` dependency and adds a timeout (`LLM_TIMEOUT_S`), retries (`LLM_MAX_RETRIES`) and `max_tokens`. Gemini, the existing default, keeps its own SDK and also gains a timeout and native streaming.
- **Errors fall into two types.**
  - `LLMNotConfiguredError` means a missing key or endpoint, or `none`.
  - `LLMUnavailableError` means a timeout, connection failure, 4xx/5xx, or a broken stream.

  Both inherit from `LLMError`. Every caller catches `LLMError` and falls back to computed facts or templates. Before this change, a timeout escaped `/query` as a 500.
- **Why vLLM goes through its OpenAI-compatible server instead of an in-process engine.** The API pod stays CPU-only and small. The GPU pod scales separately and can be replaced by any OpenAI-compatible server (TensorRT-LLM/Triton, NIM).

## Streaming protocol (`POST /query/stream`, `text/event-stream`)

| Order | Event | Payload |
|---|---|---|
| 1 | `sources` | retrieved documents (`id`, `doc_type`, `snippet`, `score`, `metadata`) and `hospital_ids` |
| 2..n | `token` | `{"text": "..."}` |
| optional | `error` | LLM runtime failure only: `{"message", "partial"}`, followed by a fallback `token` |
| last | `done` | `source_ids` (same as `sources`), `fallback`, `fallback_reason`, `retrieval_ms`, `first_token_ms`, `total_ms` |

Sources are sent before generation, so citations survive a mid-stream failure. `done` repeats the IDs for clients that keep only the final event. `LLM_PROVIDER=none` is deliberate configuration, so it produces a fallback answer with no `error` event.

## Deployment

| Path | GPU | LLM | How |
|---|---|---|---|
| Local Python | no | none / Gemini | `uvicorn backend.main_fastapi:app` |
| Docker Compose | no | none / Gemini / mock | `docker compose up api`, `--profile mock` |
| Docker Compose | yes | vLLM | `--profile gpu` |
| Kubernetes (kind) | no | none | `deploy/k8s/overlays/kind` |
| Kubernetes | yes | vLLM | `deploy/k8s/overlays/gpu` |

**Kubernetes** (`deploy/k8s`, Kustomize):
- **base:** Namespace, ConfigMap (non-secret config), optional Secret `healthiq-secrets`, a data PVC, and the API Deployment and Service.
  - Requests are 1 CPU / 1.5 Gi and limits 2 CPU / 3 Gi. Measured steady state is about 1.0 GiB.
  - A startup probe on `/health` allows 15 minutes for index load or build.
  - Readiness uses `/ready`, which returns 503 until the index is loaded. It reports the LLM's status but doesn't require it.
  - Liveness uses `/health`.
  - The pod runs as non-root UID 10001.
- **gpu overlay:** adds a vLLM Deployment and Service.
  - Image `vllm/vllm-openai:v0.31.0` (the version benchmarked on the T4), Qwen2.5-1.5B-Instruct.
  - Resources: `nvidia.com/gpu: 1`, 8–16 Gi memory, a memory-backed `/dev/shm`, a GPU toleration, and a 20-minute startup budget.
  - It switches the ConfigMap to `LLM_PROVIDER=vllm`.
  - It needs the NVIDIA device plugin.
- **Scaling note:** the data PVC is ReadWriteOnce, so the API runs 1 replica. Scaling out means baking the index into a read-only volume or image.

The API image is 2.08 GB: CPU-only torch plus the embedding model baked in, so pods need no Hugging Face access at runtime. CSVs and the index are mounted, not baked.

## Evaluation and benchmarking

- `backend/evaluation/retrieval_audit.py` measures retrieval quality: P@10, nDCG@10 and MRR@10 with bootstrap CIs on held-out queries.
- `backend/evaluation/answer_quality.py` measures answer grounding separately.
- `benchmarks/bench_serving.py` measures serving latency and throughput.
- See [RETRIEVAL_AUDIT.md](RETRIEVAL_AUDIT.md) and [BENCHMARKS.md](BENCHMARKS.md).
