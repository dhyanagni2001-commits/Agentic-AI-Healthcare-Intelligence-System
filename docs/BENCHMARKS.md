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

## Verified result: vLLM on an NVIDIA T4 (model only)

- **Server:** **vLLM 0.31.0** (V1 engine) on Kaggle, 1× Tesla T4 16 GB (one of two, pinned with `CUDA_VISIBLE_DEVICES=0`), driver 580.178.04, PyTorch 2.13.0+cu130
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, fp16 (`--dtype half`; the T4 has no bf16), `--max-model-len 4096`, `--gpu-memory-utilization 0.90`
- **Workload:** same 24-query file (byte-identical upload), `max_tokens=128`, temperature 0, **64 requests per level**, 2 warmup. The client ran in the same notebook, so there are no network hops.
- **Engine config (from `vllm.log`):**
  - attention backend `TRITON_ATTN`, since FlashAttention needs Ampere or newer and the T4 is Turing (sm75)
  - CUDA graphs in `FULL_AND_PIECEWISE` mode; prefix caching and chunked prefill on (the defaults)
  - KV cache **9.08 GiB = 340,064 tokens**, so vLLM reports a maximum concurrency of **83×** at 4,096 tokens per request
  - weights and non-torch memory 3.24 GiB; engine init 67.9 s, including 20.6 s of compilation
- **Raw:** `benchmarks/results/vllm-t4-qwen1.5b.json` *(still to copy in from the Kaggle output)*

| Concurrency | ok/n | req/s | output tok/s | TTFT p50 / p99 (ms) | ITL p50 (ms) | TPOT p50 (ms) | E2E p50 / p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64/64 | 0.60 | 54 | 40.7 / 49.3 | 18.3 | 18.2 | 1584 / 2620 |
| 4 | 64/64 | 2.59 | 234 | 47.3 / 58.0 | 15.7 | 15.8 | 1364 / 2089 |
| 8 | 64/64 | 4.82 | 431 | 52.1 / 69.2 | 16.4 | 16.6 | 1396 / 2166 |
| 16 | 64/64 | 8.06 | 723 | 57.0 / 99.2 | 17.9 | 18.4 | 1552 / 2407 |
| 32 | 64/64 | 11.74 | 1060 | 104.5 / 162.6 | 23.2 | 23.3 | 2042 / 3145 |

**Reading it:**
- **Continuous batching scales.** Output throughput rises **19.6×** from concurrency 1 to 32 (54 → 1,060 tok/s), and there are no errors at any level.
- **Latency holds under load.** TTFT p50 stays within 41–57 ms up to concurrency 16 and reaches 105 ms at 32. Per-token latency rises only from 18 to 23 ms.
- **The contrast with llama.cpp is in the shape of the curve, not the absolute numbers.** llama.cpp, with 4 fixed slots, flattened at 369 tok/s with 1.2 s TTFT at concurrency 8. vLLM at concurrency 8 gave 431 tok/s at 52 ms and kept climbing. The hardware differs (M5 Pro vs T4), as do the precision (4-bit vs fp16) and the request count (32 vs 64), so don't compare per-request speed directly. Single-stream ITL is actually lower on the M5 Pro (5.5 vs 18 ms), as expected for a 4-bit model on high-bandwidth unified memory.
- **Against the hardware limit:** single-stream decoding is bound by memory bandwidth. Reading 3.24 GiB of weights at the T4's 320 GB/s peak takes about 10.9 ms per token, a ceiling of about 92 tok/s. The measured 18.3 ms is about **59% of peak bandwidth**, which is reasonable given Triton attention (no FlashAttention on Turing) and scheduler overhead.
- **The prompts are short.** System prompt plus query is about 50 tokens. Real HealthIQ RAG prompts carry 10 evidence snippets (up to 220 characters each) plus computed facts, about 450–800 tokens, roughly 10× longer, so prefill and TTFT will be noticeably higher, especially under concurrency. The end-to-end `--target app` run on vLLM measures that and is still pending.
- **The run repeats.** A second identical run reproduced output throughput within ±1.1% at every level (e.g. 1,060 → 1,048 tok/s at concurrency 32) and TTFT p50 within 0.5 ms up to concurrency 16.
- **p99 is noisy.** With 64 requests per level, p99 is close to the slowest single request: at concurrency 32, TTFT p99 was 163 ms in one run and 212 ms in the other. E2E spread also reflects output length (anywhere up to 128 tokens). Repeat runs would give variance bounds.
- **The ceiling isn't reached yet.** The KV cache could hold about 83 full-length requests, and concurrency 32 still increases throughput. A sweep to 64–128 would find the saturation point.
- **Caveat: prefix caching.** The workload cycles 24 short prompts, and vLLM's prefix cache is on by default. Repeated prompts can skip part of prefill, which slightly flatters TTFT. With prompts of only about 40 tokens the effect is small, but a non-repeating workload would remove it.

## Still pending

The model-only vLLM run is done (above). For RAG-sized prompts with prefix caching off, concurrency 1–128 and 3 repeats, paste `benchmarks/kaggle_rag_sweep.py` into a GPU notebook. It uses `benchmarks/workloads/healthcare_rag_prompts.jsonl`, the real HealthIQ prompts for the 24 queries. Still to run on a GPU host: **(2)** end to end through HealthIQ with vLLM, **(3)** answer grounding with the fp16 model, and the Kubernetes `gpu` overlay. Commands:

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
