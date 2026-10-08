"""Streaming latency/throughput benchmark for vLLM and the HealthIQ API.

Targets
  llm  POST {url}/chat/completions (stream=True) — any OpenAI-compatible
       server, e.g. `vllm serve`. Measures model serving.
  app  POST {url}/query/stream (HealthIQ SSE) — measures end to end:
       embedding + FAISS retrieval + prompt build + LLM stream + SSE.

Metrics per concurrency level
  TTFT   time to first content chunk
  ITL    inter-chunk latency (vLLM streams ~1 token per chunk)
  TPOT   (e2e - ttft) / (output_tokens - 1), per request
  E2E    request start to final byte
  throughput: requests/s and output tokens/s over the level's wall time
  errors: count and rate (HTTP errors, timeouts, broken streams, app `error` events)

Every result file records hardware, model, workload hash, and settings so a
run can be reproduced and compared. Example:

  python -m benchmarks.bench_serving --target llm --url http://localhost:8001/v1 \\
      --concurrency 1,4,16 --num-requests 64 --max-tokens 128 --label vllm-l4
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

HERE = Path(__file__).parent
DEFAULT_WORKLOAD = HERE / "workloads" / "healthcare_queries.jsonl"
SYSTEM_PROMPT = ("You are a healthcare infrastructure analyst. Answer concisely "
                 "using general knowledge; do not invent specific statistics.")


@dataclass
class RequestResult:
    ok: bool
    ttft_s: Optional[float] = None
    e2e_s: Optional[float] = None
    chunk_gaps_s: List[float] = field(default_factory=list)
    output_tokens: int = 0
    token_count_source: str = "chunks"
    error: Optional[str] = None
    server_retrieval_ms: Optional[float] = None


def load_workload(path: Path) -> List[str]:
    return [json.loads(line)["query"] for line in path.read_text().splitlines() if line.strip()]


async def _iter_sse(resp: httpx.Response):
    """Yields (event, data_str) from an SSE response."""
    event, data = "message", []
    async for line in resp.aiter_lines():
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        yield event, "\n".join(data)


async def one_llm_request(client: httpx.AsyncClient, url: str, model: str, prompt: str,
                          max_tokens: int) -> RequestResult:
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}]}
    r = RequestResult(ok=False)
    t0 = time.perf_counter()
    last = None
    try:
        async with client.stream("POST", f"{url}/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                r.error = f"HTTP {resp.status_code}"
                return r
            done = False
            async for _, data in _iter_sse(resp):
                if data == "[DONE]":
                    done = True
                    break
                obj = json.loads(data)
                if obj.get("usage") and obj["usage"].get("completion_tokens") is not None:
                    r.output_tokens = obj["usage"]["completion_tokens"]
                    r.token_count_source = "usage"
                choices = obj.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if r.ttft_s is None:
                        r.ttft_s = now - t0
                    else:
                        r.chunk_gaps_s.append(now - last)
                    last = now
                    if r.token_count_source == "chunks":
                        r.output_tokens += 1
            if not done:
                r.error = "stream ended without [DONE]"
                return r
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        r.error = type(e).__name__
        return r
    r.e2e_s = time.perf_counter() - t0
    r.ok = r.ttft_s is not None
    if not r.ok:
        r.error = "no content"
    return r


async def one_app_request(client: httpx.AsyncClient, url: str, query: str,
                          max_results: int) -> RequestResult:
    r = RequestResult(ok=False)
    t0 = time.perf_counter()
    last = None
    try:
        async with client.stream("POST", f"{url}/query/stream",
                                 json={"query": query, "max_results": max_results}) as resp:
            if resp.status_code != 200:
                r.error = f"HTTP {resp.status_code}"
                return r
            got_done = False
            async for event, data in _iter_sse(resp):
                obj = json.loads(data)
                if event == "token":
                    now = time.perf_counter()
                    if r.ttft_s is None:
                        r.ttft_s = now - t0
                    else:
                        r.chunk_gaps_s.append(now - last)
                    last = now
                    r.output_tokens += 1
                elif event == "error":
                    r.error = "app error event: " + str(obj.get("message"))[:80]
                elif event == "done":
                    got_done = True
                    r.server_retrieval_ms = obj.get("retrieval_ms")
                    if obj.get("fallback") and not r.error:
                        r.error = "llm fallback"
            if not got_done:
                r.error = r.error or "stream ended without done"
                return r
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        r.error = type(e).__name__
        return r
    r.e2e_s = time.perf_counter() - t0
    r.ok = r.error is None and r.ttft_s is not None
    return r


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _dist_ms(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"n": 0}
    ms = [x * 1000 for x in xs]
    return {"n": len(ms), "mean": round(statistics.fmean(ms), 2),
            **{f"p{p}": round(_pct(ms, p), 2) for p in (50, 90, 99)}}


async def run_level(args, prompts: List[str], concurrency: int, model: str) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(args.timeout_s, connect=10.0)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def task(i: int) -> RequestResult:
            p = prompts[i % len(prompts)]
            async with sem:
                if args.target == "llm":
                    return await one_llm_request(client, args.url, model, p, args.max_tokens)
                return await one_app_request(client, args.url, p, args.max_results)

        for i in range(args.warmup):  # warmup: sequential, not recorded
            await task(i)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(task(i) for i in range(args.num_requests)))
        wall = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    tpot = [(r.e2e_s - r.ttft_s) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    errors: Dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[r.error or "unknown"] = errors.get(r.error or "unknown", 0) + 1
    out_tokens = sum(r.output_tokens for r in ok)
    retrieval = [r.server_retrieval_ms / 1000 for r in ok if r.server_retrieval_ms is not None]
    return {
        "concurrency": concurrency,
        "num_requests": len(results),
        "succeeded": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 4) if results else None,
        "errors": errors,
        "wall_time_s": round(wall, 3),
        "request_throughput_rps": round(len(ok) / wall, 3) if wall else None,
        "output_token_throughput_tps": round(out_tokens / wall, 2) if wall else None,
        "output_tokens_total": out_tokens,
        "token_count_source": sorted({r.token_count_source for r in ok}),
        "ttft_ms": _dist_ms([r.ttft_s for r in ok]),
        "itl_ms": _dist_ms([g for r in ok for g in r.chunk_gaps_s]),
        "tpot_ms": _dist_ms(tpot),
        "e2e_ms": _dist_ms([r.e2e_s for r in ok]),
        **({"server_retrieval_ms": _dist_ms(retrieval)} if retrieval else {}),
    }


def _cmd(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info(gpu_note: Optional[str]) -> Dict[str, Any]:
    info = {"client_platform": platform.platform(), "client_machine": platform.machine(),
            "client_cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    if sys.platform == "darwin":
        info["client_cpu"] = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _cmd(["sysctl", "-n", "hw.memsize"])
        info["client_mem_gb"] = round(int(mem) / 2**30, 1) if mem else None
    info["local_nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                     "--format=csv,noheader"])
    # The server may be remote: say what it runs on, since the client can't see it.
    info["server_hardware_note"] = gpu_note
    return info


def discover_model(args) -> Dict[str, Any]:
    try:
        if args.target == "llm":
            data = httpx.get(f"{args.url}/models", timeout=10).json()
            return {"model": args.model or data["data"][0]["id"],
                    "served_models": [m["id"] for m in data["data"]]}
        ready = httpx.get(f"{args.url}/ready", timeout=10).json()
        return {"model": ready.get("llm", {}).get("model"), "app_llm": ready.get("llm"),
                "hospitals_loaded": ready.get("hospitals_loaded")}
    except Exception as e:  # metadata only; the benchmark itself will surface real failures
        return {"model": args.model, "discovery_error": str(e)}


def _git_commit() -> Optional[str]:
    return _cmd(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"])


def print_table(levels: List[Dict[str, Any]]) -> None:
    hdr = f"{'conc':>4} {'ok/n':>7} {'err%':>5} {'req/s':>7} {'tok/s':>8} " \
          f"{'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} {'TPOT p50':>9} {'E2E p50':>8} {'E2E p99':>8}"
    print(hdr)
    for lv in levels:
        g = lambda k, p: lv[k].get(p) if lv[k].get("n") else None  # noqa: E731
        f = lambda v: f"{v:.1f}" if v is not None else "-"  # noqa: E731
        print(f"{lv['concurrency']:>4} {lv['succeeded']:>3}/{lv['num_requests']:<3} "
              f"{lv['error_rate'] * 100:>5.1f} {lv['request_throughput_rps']:>7.2f} "
              f"{lv['output_token_throughput_tps']:>8.1f} {f(g('ttft_ms', 'p50')):>9} "
              f"{f(g('ttft_ms', 'p99')):>9} {f(g('itl_ms', 'p50')):>8} {f(g('tpot_ms', 'p50')):>9} "
              f"{f(g('e2e_ms', 'p50')):>8} {f(g('e2e_ms', 'p99')):>8}")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["llm", "app"], required=True)
    ap.add_argument("--url", required=True, help="llm: base incl. /v1; app: API root")
    ap.add_argument("--model", help="llm target: model name (default: first from /v1/models)")
    ap.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-results", type=int, default=10, help="app target: retrieval top_k")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--gpu-note", help="server hardware, e.g. '1x NVIDIA L4 24GB, driver 550'")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args(argv)
    args.url = args.url.rstrip("/")

    prompts = load_workload(args.workload)
    model_meta = discover_model(args)
    model = model_meta.get("model") or args.model or "unknown"
    levels = []
    for c in (int(x) for x in args.concurrency.split(",")):
        lv = asyncio.run(run_level(args, prompts, c, model))
        levels.append(lv)
    report = {
        "label": args.label,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "target": args.target,
        "url": args.url,
        "model": model_meta,
        "hardware": hardware_info(args.gpu_note),
        "workload": {"file": str(args.workload.relative_to(HERE.parent)) if args.workload.is_relative_to(HERE.parent) else str(args.workload),
                     "sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest()[:16],
                     "num_prompts": len(prompts), "max_tokens": args.max_tokens,
                     "max_results": args.max_results, "temperature": 0.0,
                     "warmup": args.warmup, "num_requests_per_level": args.num_requests},
        "levels": levels,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[{args.label}] target={args.target} model={model}")
    print_table(levels)
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
