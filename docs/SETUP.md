# Setup and Demo

You need Python 3.11 and the CSVs in `data/`. Docker is optional. A GPU is needed only for vLLM.

## 1. Local, CPU, no LLM (about 2 minutes)

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt pytest   # or python3.11 -m venv + pip
source .venv/bin/activate
python -m pytest -q                                   # 106 tests, ~10 s, no network/GPU
LLM_PROVIDER=none uvicorn backend.main_fastapi:app --port 8000
```

The first start builds the FAISS index: 257,700 vectors in 72 s on an Apple M5 Pro. It's saved to `data/vector_index/`, and later starts load it in a few seconds.

## 2. Demo

```bash
curl -s localhost:8000/ready                         # index + LLM status
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"Pediatric hospitals in California"}' | python -m json.tool | head -30
curl -N -X POST localhost:8000/query/stream -H 'content-type: application/json' \
  -d '{"query":"ICU hospitals in Florida","max_results":3}'
```

Expected stream: `event: sources`, then `token` events, then `event: done` whose `source_ids` match the sources. To watch real token-by-token streaming without a GPU, point the API at the mock OpenAI server:

```bash
python -m benchmarks.mock_openai_server --port 8001 --token-delay-ms 30 &
LLM_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL=mock-model \
  uvicorn backend.main_fastapi:app --port 8000
```

The mock emits placeholder tokens (`tok0 tok1 …`). It demonstrates the transport, not answer quality.

For **real answers without an NVIDIA GPU**, use llama.cpp (`brew install llama.cpp`). It speaks the same OpenAI API, so the `vllm` provider talks to it unchanged:

```bash
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M --port 8001 --alias qwen2.5-1.5b-instruct-q4_k_m   # ~1 GB download
LLM_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8001/v1 VLLM_MODEL=qwen2.5-1.5b-instruct-q4_k_m \
  LLM_TEMPERATURE=0 uvicorn backend.main_fastapi:app --port 8000
python deploy/smoke_test.py http://localhost:8000 --expect-llm
```

Label any numbers from this setup as llama.cpp, not vLLM. The React UI (`cd frontend && npm install && npm start`) uses `/query` and is unchanged.

## 3. Docker Compose

```bash
docker compose up --build api                                    # CPU, LLM_PROVIDER=none
LLM_PROVIDER=gemini GEMINI_API_KEY=... docker compose up api     # free-tier Gemini
LLM_PROVIDER=vllm VLLM_BASE_URL=http://mock-llm:8000/v1 VLLM_MODEL=mock-model \
  docker compose --profile mock up                               # streaming demo, CPU
LLM_PROVIDER=vllm docker compose --profile gpu up                # vLLM, NVIDIA GPU
python deploy/smoke_test.py http://localhost:8000 [--expect-llm]
```

## 4. Kubernetes

CPU path, local, using [kind](https://kind.sigs.k8s.io):

```bash
docker build -t healthiq-api:dev .
kind create cluster --name healthiq --config deploy/kind-config.yaml   # run from repo root
kind load docker-image healthiq-api:dev --name healthiq
kubectl apply -k deploy/k8s/overlays/kind
kubectl -n healthiq rollout status deploy/healthiq-api
kubectl -n healthiq port-forward svc/healthiq-api 8000:8000 &
python deploy/smoke_test.py http://localhost:8000
kind delete cluster --name healthiq
```

GPU cluster (needs the NVIDIA device plugin):
1. Push the image to your registry and set it in `deploy/k8s/base/api.yaml`.
2. Copy the CSVs into the `healthiq-data` PVC.
3. Run:

```bash
kubectl apply -k deploy/k8s/overlays/gpu
kubectl -n healthiq create secret generic healthiq-secrets --from-literal=HF_TOKEN=...   # optional
python deploy/smoke_test.py http://<api-service> --expect-llm
```

## 5. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `vllm`, `grok`, or `none` |
| `VLLM_BASE_URL` / `VLLM_MODEL` | – | e.g. `http://vllm:8000/v1`, `Qwen/Qwen2.5-1.5B-Instruct` |
| `VLLM_API_KEY` | `EMPTY` | only needed if vLLM was started with `--api-key` |
| `GEMINI_API_KEY` / `GROK_API_KEY` | – | for those providers |
| `LLM_TIMEOUT_S` / `LLM_MAX_RETRIES` / `LLM_MAX_TOKENS` | 30 / 1 / 512 | apply to every provider |
| `LLM_TEMPERATURE` | server default | OpenAI-compatible providers; set `0` for reproducible evaluation |

## 6. Evaluate and benchmark

```bash
python -m backend.evaluation.retrieval_audit --out reports/retrieval_audit_fixed.json
python -m backend.evaluation.answer_quality                     # needs an LLM
python -m benchmarks.bench_serving --target app --url http://localhost:8000 --label my-run
```

See [BENCHMARKS.md](BENCHMARKS.md) and [RETRIEVAL_AUDIT.md](RETRIEVAL_AUDIT.md).
