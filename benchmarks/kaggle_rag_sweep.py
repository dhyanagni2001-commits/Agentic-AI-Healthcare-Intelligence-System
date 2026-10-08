# RAG-sized vLLM sweep for a Kaggle/Colab GPU notebook — paste as ONE cell.
#
# Prereqs in the notebook: vLLM installed (torchaudio removed), bench_serving.py
# written to /kaggle/working (via %%writefile), and rag_prompts.txt attached as
# a dataset (generated from benchmarks/workloads/healthcare_rag_prompts.jsonl:
# real HealthIQ RAG prompts for the 24 benchmark queries).
#
# What it does:
#   1. restarts vLLM with prefix caching OFF (repeated long prompts would
#      otherwise hide prefill cost),
#   2. measures real prompt lengths with vLLM's own tokenizer (/tokenize),
#   3. sweeps concurrency 1..128, 3 repeats, >= 2x concurrency requests per level,
#   4. prints mean ± std across repeats and zips everything for download.
import glob, json, os, shutil, statistics, subprocess, sys, time, urllib.request

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
URL = "http://localhost:8001"
LEVELS = [1, 4, 8, 16, 32, 64, 128]
REPS = 3
OUT = "/kaggle/working/rag_sweep"
os.makedirs(OUT, exist_ok=True)

RAG = glob.glob("/kaggle/input/**/rag_prompts.txt", recursive=True)
assert RAG, "rag_prompts.txt not found under /kaggle/input — attach the dataset first"
RAG = RAG[0]
assert os.path.exists("bench_serving.py"), "bench_serving.py missing — rerun the %%writefile cell"
print("workload:", RAG)


def get(path, data=None, timeout=5):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


# 1. restart vLLM without prefix caching
try:
    proc.terminate(); proc.wait(timeout=60)  # noqa: F821 — from the earlier start cell, if present
except Exception:
    pass
subprocess.run(["pkill", "-f", "vllm"])  # also stops a lingering EngineCore child
for _ in range(60):  # wait until GPU 0 memory is actually released
    used = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    if used.isdigit() and int(used) < 1000:
        break
    time.sleep(2)
print("GPU 0 memory used before restart (MiB):", used)
log = open(f"{OUT}/vllm_nocache.log", "w")
proc = subprocess.Popen(
    ["vllm", "serve", MODEL, "--port", "8001", "--dtype", "half", "--max-model-len", "4096",
     "--gpu-memory-utilization", "0.90", "--no-enable-prefix-caching"],
    stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
t0 = time.time()
for _ in range(120):
    if proc.poll() is not None:
        print(open(f"{OUT}/vllm_nocache.log").read()[-3000:])
        raise SystemExit(f"vLLM exited with code {proc.returncode}")
    try:
        get("/health", timeout=2); break
    except Exception:
        time.sleep(5)
print(f"vLLM ready in {time.time() - t0:.0f}s")
cfg = open(f"{OUT}/vllm_nocache.log").read()
print("prefix caching disabled:", "enable_prefix_caching=False" in cfg)

# 2. real prompt lengths, counted by vLLM's tokenizer (system prompt of the harness not included)
prompts = [json.loads(l)["query"] for l in open(RAG) if l.strip()]
try:
    counts = sorted(get("/tokenize", {"model": MODEL, "prompt": p})["count"] for p in prompts)
    prompt_stats = {"n": len(counts), "min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1],
                    "source": "vllm /tokenize"}
except Exception as e:  # non-fatal: fall back to a rough chars/4 estimate
    counts = sorted(len(p) // 4 for p in prompts)
    prompt_stats = {"n": len(counts), "min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1],
                    "source": f"estimate chars/4 (/tokenize failed: {e})"}
print("RAG prompt tokens:", prompt_stats)

def _out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


VER = next((l.split()[-1] for l in _out([sys.executable, "-m", "pip", "show", "vllm"]).splitlines()
            if l.startswith("Version")), "?")
GPU = _out(["nvidia-smi", "-i", "0", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader"]) or "unknown GPU"
NOTE = (f"{GPU} (1 of 2, CUDA_VISIBLE_DEVICES=0); vllm {VER}; fp16; prefix caching OFF; "
        f"RAG prompts median {prompt_stats['median']} tokens")

# 3. sweep
results = {c: [] for c in LEVELS}
start = time.time()
for rep in range(1, REPS + 1):
    for c in LEVELS:
        n = max(32, 2 * c)
        label = f"vllm-t4-rag-c{c}-r{rep}"
        r = subprocess.run([sys.executable, "bench_serving.py", "--target", "llm", "--url", URL + "/v1",
                            "--workload", RAG, "--concurrency", str(c), "--num-requests", str(n),
                            "--warmup", "1", "--max-tokens", "128", "--timeout-s", "300",
                            "--label", label, "--out-dir", OUT, "--gpu-note", NOTE],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(f"benchmark failed at c={c}")
        lv = json.load(open(f"{OUT}/{label}.json"))["levels"][0]
        results[c].append(lv)
        print(f"rep {rep} c={c:>3}  ok {lv['succeeded']:>3}/{lv['num_requests']:<3} "
              f"tok/s {lv['output_token_throughput_tps']:>7.1f}  TTFT p50 {lv['ttft_ms'].get('p50', float('nan')):>7.1f} ms  "
              f"ITL p50 {lv['itl_ms'].get('p50', float('nan')):>5.1f} ms   [{(time.time() - start) / 60:.1f} min]", flush=True)


# 4. mean ± std across repeats
def ms(c, key, stat="p50"):
    xs = [lv[key].get(stat) for lv in results[c] if lv[key].get(stat) is not None]
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


summary = {"note": NOTE, "model": MODEL, "vllm": VER, "prompt_tokens": prompt_stats,
           "workload": os.path.basename(RAG), "max_tokens": 128, "reps": REPS, "levels": {}}
print(f"\nmean ± std over {REPS} repeats — RAG prompts (median {prompt_stats['median']} tokens), prefix caching OFF")
print(f"{'conc':>4} {'n/rep':>5} {'err':>4} {'tok/s':>15} {'req/s':>12} {'TTFT p50 ms':>15} {'TTFT p99 ms':>15} {'ITL p50 ms':>12} {'E2E p50 ms':>15}")
for c in LEVELS:
    tps = [lv["output_token_throughput_tps"] for lv in results[c]]
    rps = [lv["request_throughput_rps"] for lv in results[c]]
    errs = sum(lv["num_requests"] - lv["succeeded"] for lv in results[c])
    row = {"tok_s": (statistics.fmean(tps), statistics.stdev(tps)), "req_s": (statistics.fmean(rps), statistics.stdev(rps)),
           "ttft_p50": ms(c, "ttft_ms"), "ttft_p99": ms(c, "ttft_ms", "p99"),
           "itl_p50": ms(c, "itl_ms"), "e2e_p50": ms(c, "e2e_ms"), "errors": errs,
           "requests_per_rep": results[c][0]["num_requests"]}
    summary["levels"][c] = row
    f = lambda t: f"{t[0]:.1f} ± {t[1]:.1f}"  # noqa: E731
    print(f"{c:>4} {row['requests_per_rep']:>5} {errs:>4} {f(row['tok_s']):>15} {f(row['req_s']):>12} "
          f"{f(row['ttft_p50']):>15} {f(row['ttft_p99']):>15} {f(row['itl_p50']):>12} {f(row['e2e_p50']):>15}")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
shutil.make_archive("/kaggle/working/vllm-t4-rag-sweep", "zip", OUT)
print(f"\ndone in {(time.time() - start) / 60:.1f} min -> download /kaggle/working/vllm-t4-rag-sweep.zip")
