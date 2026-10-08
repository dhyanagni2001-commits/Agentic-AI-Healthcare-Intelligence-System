# Serving Benchmarks

The harness is `benchmarks/bench_serving.py`. It is async httpx and streaming-aware, with no new dependencies. Every run writes `benchmarks/results/<label>.json`, which records:
- **Hardware:** client CPU/RAM, `nvidia-smi` if present, and a free-text `--gpu-note` for remote servers.
- **Model:** from `/v1/models` or `/ready`.
- **Workload:** file, SHA-256, prompt count, `max_tokens`, `temperature=0`.
- **Concurrency:** level, request count and warmup.
- **Code version:** git commit.

| Metric | Definition |
|---|---|
| TTFT | request start → first content chunk |
| ITL | gap between consecutive content chunks (vLLM streams ≈1 token per chunk) |
| TPOT | (E2E − TTFT) / (output tokens − 1), per request |
| E2E | request start → final byte |
| Throughput | successful requests/s and output tokens/s over the level's wall time |
| Errors | HTTP errors, timeouts, broken streams, app `error` events and LLM fallbacks, counted by cause |

Output tokens come from vLLM's `usage` (via `stream_options.include_usage`) when the server returns it; otherwise from chunk count. The result file records which source was used (`token_count_source`).

## Verified result: application overhead on CPU (no model)

**This is not an LLM benchmark.** The LLM is `benchmarks/mock_openai_server.py`, which emits 64 tokens at a fixed 10 ms each. The run isolates what HealthIQ adds around the model: embedding the query on CPU, a flat FAISS search over 257,700 vectors, prompt construction and SSE relay.

- **Server:** API in Docker Desktop (linux/arm64 VM on Apple M5 Pro, 6 vCPU, 9.7 GiB), CPU-only, `docker compose --profile mock`
- **Workload:** `benchmarks/workloads/healthcare_queries.jsonl` (24 queries, sha256 `834b5968b9727a9e`), top_k=10, 48 requests per level, 2 warmup
- **Raw:** `benchmarks/results/cpu-docker-mockllm-app.json`

| Concurrency | ok/n | req/s | TTFT p50 / p99 (ms) | server retrieval p50 / p99 (ms) | ITL p50 (ms) | E2E p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48/48 | 1.35 | 38.5 / 47.0 | 23.3 / 31.8 | 10.9 | 742 / 761 |
| 4 | 48/48 | 5.44 | 51.6 / 60.9 | 33.6 / 39.0 | 10.7 | 734 / 748 |
| 16 | 48/48 | 19.05 | 162.6 / 202.6 | 113.8 / 177.6 | 10.5 | 837 / 876 |

**Reading it:**
- **Token relay overhead is small.** ITL stays within about 1 ms of the mock's 10 ms per token at every level.
- **Retrieval is the CPU bottleneck under load.** Server-side retrieval p50 rises from 23 ms to 114 ms at concurrency 16, because query embedding and the exact flat search compete for 6 vCPUs. That accounts for most of the TTFT growth.
- **Next levers:**
  - Batch query embeddings across concurrent requests.
  - Move embedding to the GPU next to vLLM.
  - Switch to an IVF/HNSW index if the corpus grows.

**Failure check (same stack, mock LLM stopped):** 8/8 requests were recorded as errors with cause `vllm stream failed: Connection error`. The API still returned HTTP 200 with sources and a computed-facts fallback, and no 500s occurred.

## Verified result: real model on a laptop (llama.cpp, not vLLM)

This measures a real LLM, but with **llama.cpp on an Apple M5 Pro, not vLLM on an NVIDIA GPU**. The provider code is the same because both expose the OpenAI-compatible API. The serving engines differ, so these numbers say nothing about vLLM throughput.

- **Server:** `llama-server` 0.4.1 (build 10964), `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`, 4 parallel slots, context 8192. Same machine as the client (M5 Pro, 24 GB).
- **Workload:** same 24-query file, `max_tokens=128`, temperature 0, 32 requests per level, 2 warmup. Model-only token counts come from server `usage`.
- **Raw:** `benchmarks/results/llamacpp-m5pro-qwen1.5b-q4-llm.json` and `…-app.json`

**Model only** (`--target llm`):

| Concurrency | ok/n | req/s | output tok/s | TTFT p50 / p99 (ms) | ITL p50 (ms) | E2E p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32/32 | 1.50 | 172 | 31.9 / 33.1 | 5.5 | 716 / 809 |
| 4 | 32/32 | 3.06 | 349 | 42.6 / 111.5 | 10.3 | 1380 / 1479 |
| 8 | 32/32 | 3.23 | 369 | 1236 / 1368 | 9.8 | 2470 / 2704 |

**End to end through HealthIQ** (`--target app`: retrieval + RAG prompt + stream):

| Concurrency | ok/n | req/s | TTFT p50 / p99 (ms) | server retrieval p50 (ms) | ITL p50 (ms) | E2E p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32/32 | 1.29 | 117.7 / 302.7 (cold prompt cache) | 10.6 | 5.5 | 792 / 1100 |
| 4 | 32/32 | 2.97 | 41.7 / 106.5 | 13.8 | 10.3 | 1386 / 1466 |
| 8 | 32/32 | 3.02 | 1356 / 1447 | 15.6 | 10.4 | 2654 / 2803 |

**Reading it:**
- **Throughput stops scaling at the slot count.** Going from 1 to 4 concurrent requests doubles output tokens/s (172 → 349). At 8 it barely moves (369), and TTFT jumps to about 1.2 s because requests queue for one of the 4 slots. This is the limit continuous batching in vLLM is designed to remove, which is why the vLLM run still matters.
- **Per-token latency roughly doubles under load** (5.5 → 10 ms) as 4 sequences share the hardware.
- **Prompt caching skews TTFT across levels.** The first level runs with a cold cache: 118 ms TTFT for the full RAG prompt. Later levels reuse cached prefixes of the same 24 prompts. Rerunning concurrency 1 with a warm cache gave a TTFT p50 of **25 ms**. Read the cold number as first-request latency. For clean per-level comparisons, restart the server or use a non-repeating workload.
- **Retrieval is not the bottleneck here** (11–16 ms). Generation dominates end-to-end latency.

## Pending: GPU validation

No NVIDIA GPU was available, so **no vLLM TTFT, ITL or throughput numbers exist yet.** The llama.cpp numbers above are not a substitute. Run these on a GPU host (one T4, L4 or A10 is enough for the default `Qwen/Qwen2.5-1.5B-Instruct`):

```bash
# 1. Model serving alone
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 --max-model-len 4096
python -m benchmarks.bench_serving --target llm --url http://localhost:8001/v1 \
  --concurrency 1,4,16,32 --num-requests 64 --max-tokens 128 \
  --label vllm-qwen1.5b-<gpu> --gpu-note "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"

# 2. End to end through HealthIQ (retrieval + vLLM + SSE)
LLM_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  uvicorn backend.main_fastapi:app --port 8000
python -m benchmarks.bench_serving --target app --url http://localhost:8000 \
  --concurrency 1,4,16 --num-requests 48 --label app-vllm-<gpu> --gpu-note "..."

# 3. Answer grounding with the real model
python -m backend.evaluation.answer_quality --out reports/answer_quality.json
```

Compare (1) with (2) to separate model latency from application overhead. Include the result JSONs when quoting any number.

## Harness self-checks (automated)

`tests/test_llm_streaming.py::TestBenchHarness` runs the harness against the mock server and checks two things:
- It measures a known 20 ms/token delay as ITL p50 between 18 and 60 ms, with exact token totals.
- It reports injected HTTP 503s as a 100% error rate, keyed by cause.
