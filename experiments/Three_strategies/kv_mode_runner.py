import csv
import os
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path
from statistics import mean, stdev

import requests
from transformers import AutoTokenizer


# ============================================================
# GLOBAL CONFIG
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"

OUTPUT_DIR = Path("./kv_experiment_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREFIX_TOKEN_LIST = [256]
REPEATS = 3

MAX_TOKENS_STAGE1 = 8
MAX_TOKENS_STAGE2 = 32

TOOL_WAIT_SECONDS = 2.0
DO_WARMUP = False
USE_COMPLETIONS_API = True
TEMPERATURE = 0.0
POST_REQUEST_SLEEP_SECONDS = 1.0

GPU_MEMORY_UTILIZATION = 0.5
MAX_MODEL_LEN = 1000

# LMCache env for swap mode
LMCACHE_CHUNK_SIZE = "256"
LMCACHE_LOCAL_CPU = "True"
LMCACHE_MAX_LOCAL_CPU_SIZE = "5.0"

STOP_SERVER_AFTER_RUN = False
SERVER_START_TIMEOUT = 180

PID_FILE = OUTPUT_DIR / "active_server.pid"


# ============================================================
# HELPERS
# ============================================================

class LogFollower:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.offset = 0
        if self.path and self.path.exists():
            self.offset = self.path.stat().st_size

    def read_new(self) -> str:
        if self.path is None or not self.path.exists():
            return ""
        with self.path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(self.offset)
            data = f.read()
            self.offset = f.tell()
        return data


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def safe_get_text(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def request_completion(prompt: str, max_tokens: int, temperature: float = 0.0):
    t0 = now_ms()

    if USE_COMPLETIONS_API:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }
        url = f"{BASE_URL}/v1/completions"
    else:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }
        url = f"{BASE_URL}/v1/chat/completions"

    r = requests.post(url, json=payload, timeout=300)
    t1 = now_ms()
    r.raise_for_status()
    data = r.json()

    if USE_COMPLETIONS_API:
        text = data["choices"][0]["text"]
    else:
        text = data["choices"][0]["message"]["content"]

    return {
        "latency_ms": t1 - t0,
        "text": text,
        "raw_json": data,
    }


def build_exact_prefix(tokenizer, target_tokens: int, unique_tag: str) -> str:
    base_unit = (
        "This is a reusable shared context for KV cache benchmarking. "
        "It contains repeated factual but irrelevant narrative so that the tokenizer "
        "produces a long stable prefix for cache reuse experiments. "
    )

    tag_text = f"\n[RUN_TAG={unique_tag}]\n"
    tag_ids = tokenizer.encode(tag_text, add_special_tokens=False)

    usable = max(32, target_tokens - len(tag_ids))
    text = ""

    while True:
        text += base_unit
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= usable:
            ids = ids[:usable]
            prefix_body = tokenizer.decode(ids, skip_special_tokens=True)
            return prefix_body + tag_text


# ============================================================
# LOG PARSING
# ============================================================
def parse_lmcache_log_chunk(text: str) -> dict:
    """
    Parse LMCache request-level log info.

    Works with logs like:
    Reqid: cmpl-xxx-0, Total tokens 285, Inference Engine computed tokens: 256,
    LMCache hit tokens: 256, need to load: 0

    [req_id=cmpl-xxx-0] Stored 256 out of total 256 tokens.
    size: 0.0273 GB, cost 7.7667 ms, throughput: 3.5207 GB/s;
    offload_time: 7.6513 ms, put_time: 0.0531 ms
    """
    out = {
        "cache_total_tokens": None,
        "cache_computed_tokens": None,
        "cache_hit_tokens": None,
        "cache_need_to_load": None,
        "cache_hit_rate": None,
        "cache_stored_tokens": None,
        "cache_store_total_tokens": None,
        "cache_stored_kv_gb": None,
        "cache_store_cost_ms": None,
        "cache_store_throughput_gbps": None,
        "cache_offload_time_ms": None,
        "cache_put_time_ms": None,
    }

    # More robust request-line parser:
    # - Reqid can be any non-comma string, not just digits
    # - tolerate optional colon / extra spaces
    hit_pat = re.compile(
        r"Reqid:\s*([^,]+),\s*"
        r"Total tokens\s*:?\s*(\d+),\s*"
        r"Inference Engine computed tokens:\s*(\d+),\s*"
        r"LMCache hit tokens:\s*(\d+),\s*"
        r"need to load:\s*(\d+)",
        re.IGNORECASE | re.DOTALL,
    )

    # Robust store-line parser
    store_pat = re.compile(
        r"Stored\s+(\d+)\s+out of total\s+(\d+)\s+tokens\.\s*"
        r"size:\s*([0-9.]+)\s*GB,\s*"
        r"cost\s*([0-9.]+)\s*ms,\s*"
        r"throughput:\s*([0-9.]+)\s*GB/s;\s*"
        r"offload_time:\s*([0-9.]+)\s*ms,\s*"
        r"put_time:\s*([0-9.]+)\s*ms",
        re.IGNORECASE | re.DOTALL,
    )

    hit_matches = list(hit_pat.finditer(text))
    if hit_matches:
        m = hit_matches[-1]
        # group(1) is reqid string, not used here
        total_tokens = int(m.group(2))
        computed_tokens = int(m.group(3))
        hit_tokens = int(m.group(4))
        need_to_load = int(m.group(5))

        out["cache_total_tokens"] = total_tokens
        out["cache_computed_tokens"] = computed_tokens
        out["cache_hit_tokens"] = hit_tokens
        out["cache_need_to_load"] = need_to_load
        out["cache_hit_rate"] = hit_tokens / total_tokens if total_tokens > 0 else None

    store_matches = list(store_pat.finditer(text))
    if store_matches:
        m = store_matches[-1]
        out["cache_stored_tokens"] = int(m.group(1))
        out["cache_store_total_tokens"] = int(m.group(2))
        out["cache_stored_kv_gb"] = float(m.group(3))
        out["cache_store_cost_ms"] = float(m.group(4))
        out["cache_store_throughput_gbps"] = float(m.group(5))
        out["cache_offload_time_ms"] = float(m.group(6))
        out["cache_put_time_ms"] = float(m.group(7))

    return out


# ============================================================
# PROMETHEUS METRICS
# ============================================================

INTERESTING_METRICS = {
    "lmcache:num_store_requests",
    "lmcache:num_retrieve_requests",
    "lmcache:num_lookup_requests",
    "lmcache:num_requested_tokens",
    "lmcache:num_hit_tokens",
    "lmcache:num_stored_tokens",
    "lmcache:num_lookup_tokens",
    "lmcache:num_lookup_hits",
    "lmcache:num_vllm_hit_tokens",
    "lmcache:num_prompt_tokens",
    "lmcache:retrieve_hit_rate",
    "lmcache:lookup_hit_rate",
    "lmcache:time_to_retrieve",
    "lmcache:time_to_store",
    "lmcache:time_to_lookup",
    "lmcache:retrieve_speed",
    "lmcache:store_speed",
    "lmcache:local_cache_usage",
    "lmcache:remote_cache_usage",
    "lmcache:local_storage_usage",
    "lmcache:local_cpu_evict_count",
    "lmcache:local_cpu_evict_keys_count",
    "lmcache:local_cpu_evict_failed_count",
    "lmcache:local_cpu_hot_cache_count",
    "lmcache:local_cpu_keys_in_request_count",
    "lmcache:active_memory_objs_count",
    "lmcache:pinned_memory_objs_count",
    "lmcache:forced_unpin_count",
    "lmcache:lmcache_is_healthy",
    "lmcache:kv_msg_queue_size",
    "lmcache:remote_put_task_num",
    "vllm:gpu_prefix_cache_hits_total",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
}


def fetch_metrics_snapshot() -> dict[str, float]:
    text = safe_get_text(f"{BASE_URL}/metrics")
    if text is None:
        return {}

    metrics = {}
    line_re = re.compile(
        r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$'
    )

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = line_re.match(line)
        if not m:
            continue

        metric_name = m.group(1)
        value = float(m.group(3))

        if metric_name.endswith("_created"):
            continue

        if metric_name not in INTERESTING_METRICS:
            continue

        metrics[metric_name] = metrics.get(metric_name, 0.0) + value

    return metrics


def diff_metrics(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in sorted(keys)}


def flatten_metrics(before: dict[str, float], after: dict[str, float], prefix: str) -> dict:
    delta = diff_metrics(after, before)

    gauge_like = {
        "lmcache:retrieve_hit_rate",
        "lmcache:lookup_hit_rate",
        "lmcache:retrieve_speed",
        "lmcache:store_speed",
        "lmcache:local_cache_usage",
        "lmcache:remote_cache_usage",
        "lmcache:local_storage_usage",
        "lmcache:local_cpu_hot_cache_count",
        "lmcache:local_cpu_keys_in_request_count",
        "lmcache:active_memory_objs_count",
        "lmcache:pinned_memory_objs_count",
        "lmcache:lmcache_is_healthy",
        "lmcache:kv_msg_queue_size",
        "lmcache:remote_put_task_num",
    }

    out = {}

    for k in sorted(set(before) | set(after)):
        safe_name = k.replace(":", "_").replace(".", "_")
        if k in gauge_like:
            out[f"{prefix}_{safe_name}_after"] = after.get(k)
        else:
            out[f"{prefix}_{safe_name}_delta"] = delta.get(k)

    return out


def warmup():
    prompt = "Warmup request. Reply with a short sentence."
    _ = request_completion(prompt, max_tokens=8, temperature=0.0)


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(rows: list[dict], mode: str):
    if not rows:
        print(f"\nNo valid rows collected for mode={mode}.")
        return

    groups = {}
    for r in rows:
        key = r["prefix_tokens"]
        groups.setdefault(key, []).append(r)

    print(f"\n================ SUMMARY ({mode}) ================\n")

    def fmt(vals):
        vals = [x for x in vals if x is not None]
        if not vals:
            return "N/A"
        if len(vals) == 1:
            return f"{vals[0]:.4f}"
        return f"{mean(vals):.4f} ± {stdev(vals):.4f}"

    for prefix_tokens in sorted(groups):
        items = groups[prefix_tokens]

        s1 = [x.get("stage1_latency_ms") for x in items]
        s2 = [x.get("stage2_latency_ms") for x in items]
        e2e = [x.get("e2e_two_stage_ms") for x in items]
        s2_hit_rate = [x.get("stage2_cache_hit_rate") for x in items]

        print(
            f"mode={mode:9s} | "
            f"prefix={prefix_tokens:5d} | "
            f"stage1_ms={fmt(s1):>18s} | "
            f"stage2_ms={fmt(s2):>18s} | "
            f"e2e_ms={fmt(e2e):>18s} | "
            f"s2_hit_rate={fmt(s2_hit_rate):>18s}"
        )


# ============================================================
# SERVER MANAGEMENT
# ============================================================

def is_server_ready() -> bool:
    txt = safe_get_text(f"{BASE_URL}/metrics")
    return txt is not None


def wait_until_server_ready(timeout_s: int = SERVER_START_TIMEOUT):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if is_server_ready():
            return
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready within {timeout_s} seconds.")


def read_pid_file() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def write_pid_file(pid: int):
    PID_FILE.write_text(str(pid))


def remove_pid_file():
    if PID_FILE.exists():
        PID_FILE.unlink()


def kill_pid(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    for _ in range(20):
        try:
            os.kill(pid, 0)
            time.sleep(1)
        except ProcessLookupError:
            return

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_existing_server():
    pid = read_pid_file()
    if pid is not None:
        print(f"Stopping previous managed server pid={pid} ...")
        kill_pid(pid)
        remove_pid_file()

    try:
        result = subprocess.run(
            ["bash", "-lc", f"lsof -ti tcp:{PORT}"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        for p in pids:
            try:
                print(f"Killing process on port {PORT}: pid={p}")
                os.kill(int(p), signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        pass

def build_server_command_and_env(mode: str):
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--port", str(PORT),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(MAX_MODEL_LEN),
    ]
    env = os.environ.copy()

    if mode == "preserve":
        cmd += ["--enable-prefix-caching"]

    elif mode == "swap":
        cmd += [
            "--enable-prefix-caching",
            "--kv-transfer-config",
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
        ]
        env["LMCACHE_CHUNK_SIZE"] = LMCACHE_CHUNK_SIZE
        env["LMCACHE_LOCAL_CPU"] = LMCACHE_LOCAL_CPU
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = LMCACHE_MAX_LOCAL_CPU_SIZE

    elif mode == "discard":
        # Explicitly disable vLLM automatic prefix caching.
        cmd += ["--no-enable-prefix-caching"]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cmd, env


def start_server(mode: str) -> Path:
    log_path = LOG_DIR / f"{mode}_server.log"
    if log_path.exists():
        log_path.unlink()

    cmd, env = build_server_command_and_env(mode)

    print("\nStarting server with command:")
    print(" ".join(cmd))

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    write_pid_file(proc.pid)
    print(f"Started server pid={proc.pid}, log={log_path}")
    wait_until_server_ready()
    print("Server is ready.")
    return log_path


# ============================================================
# MAIN RUNNER
# ============================================================

def run_experiment(mode: str):
    if mode not in {"preserve", "swap", "discard"}:
        raise ValueError(f"Unsupported mode: {mode}")

    out_csv = OUTPUT_DIR / f"results_{mode}.csv"

    print(f"Loading tokenizer for {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    stop_existing_server()
    log_path = start_server(mode)
    follower = LogFollower(log_path)

    rows = []

    if DO_WARMUP:
        print(f"Running warmup for mode={mode} ...")
        warmup()
        time.sleep(POST_REQUEST_SLEEP_SECONDS)
        _ = follower.read_new()

    for prefix_tokens in PREFIX_TOKEN_LIST:
        for rep in range(REPEATS):
            run_id = f"{mode}_{prefix_tokens}_{rep}_{uuid.uuid4().hex[:8]}"
            print(f"\n=== mode={mode}, prefix={prefix_tokens}, repeat={rep+1}/{REPEATS}, run_id={run_id} ===")

            shared_prefix = build_exact_prefix(tokenizer, prefix_tokens, unique_tag=run_id)

            stage1_suffix = (
                "\nYou are about to call an external tool. "
                "Before the tool call, reply with a very short acknowledgement.\n"
                "Answer:"
            )

            stage2_suffix = (
                "\nTool result: status=success; value=42; explanation=the lookup finished.\n"
                "Now continue the response in one concise paragraph.\n"
                "Answer:"
            )

            prompt_stage1 = shared_prefix + stage1_suffix
            prompt_stage2 = shared_prefix + stage2_suffix

            _ = follower.read_new()
            metrics_before_1 = fetch_metrics_snapshot()

            try:
                res1 = request_completion(
                    prompt_stage1,
                    max_tokens=MAX_TOKENS_STAGE1,
                    temperature=TEMPERATURE,
                )
                time.sleep(POST_REQUEST_SLEEP_SECONDS)
                metrics_after_1 = fetch_metrics_snapshot()
                log_after_1 = follower.read_new()
            except Exception as e:
                print(f"stage1 failed: {e}")
                rows.append({
                    "run_id": run_id,
                    "mode": mode,
                    "prefix_tokens": prefix_tokens,
                    "repeat": rep,
                    "tool_wait_s": TOOL_WAIT_SECONDS,
                    "error": f"stage1_failed: {e}",
                })
                continue

            time.sleep(TOOL_WAIT_SECONDS)

            _ = follower.read_new()
            metrics_before_2 = fetch_metrics_snapshot()

            try:
                res2 = request_completion(
                    prompt_stage2,
                    max_tokens=MAX_TOKENS_STAGE2,
                    temperature=TEMPERATURE,
                )
                time.sleep(POST_REQUEST_SLEEP_SECONDS)
                metrics_after_2 = fetch_metrics_snapshot()
                log_after_2 = follower.read_new()
            except Exception as e:
                print(f"stage2 failed: {e}")
                rows.append({
                    "run_id": run_id,
                    "mode": mode,
                    "prefix_tokens": prefix_tokens,
                    "repeat": rep,
                    "tool_wait_s": TOOL_WAIT_SECONDS,
                    "stage1_latency_ms": res1["latency_ms"],
                    "error": f"stage2_failed: {e}",
                })
                continue

            cache_stage1 = parse_lmcache_log_chunk(log_after_1)
            cache_stage2 = parse_lmcache_log_chunk(log_after_2)

            flat_stage1_metrics = flatten_metrics(metrics_before_1, metrics_after_1, prefix="stage1_metrics")
            flat_stage2_metrics = flatten_metrics(metrics_before_2, metrics_after_2, prefix="stage2_metrics")

            row = {
                "run_id": run_id,
                "mode": mode,
                "prefix_tokens": prefix_tokens,
                "repeat": rep,
                "tool_wait_s": TOOL_WAIT_SECONDS,

                "stage1_latency_ms": res1["latency_ms"],
                "stage2_latency_ms": res2["latency_ms"],
                "e2e_two_stage_ms": res1["latency_ms"] + res2["latency_ms"] + TOOL_WAIT_SECONDS * 1000.0,

                "stage1_text": res1["text"],
                "stage2_text": res2["text"],

                "stage1_cache_total_tokens": cache_stage1["cache_total_tokens"],
                "stage1_cache_hit_tokens": cache_stage1["cache_hit_tokens"],
                "stage1_cache_computed_tokens": cache_stage1["cache_computed_tokens"],
                "stage1_cache_need_to_load": cache_stage1["cache_need_to_load"],
                "stage1_cache_hit_rate": cache_stage1["cache_hit_rate"],
                "stage1_cache_stored_tokens": cache_stage1["cache_stored_tokens"],
                "stage1_cache_store_total_tokens": cache_stage1["cache_store_total_tokens"],
                "stage1_cache_stored_kv_gb": cache_stage1["cache_stored_kv_gb"],
                "stage1_cache_store_cost_ms": cache_stage1["cache_store_cost_ms"],
                "stage1_cache_store_throughput_gbps": cache_stage1["cache_store_throughput_gbps"],
                "stage1_cache_offload_time_ms": cache_stage1["cache_offload_time_ms"],
                "stage1_cache_put_time_ms": cache_stage1["cache_put_time_ms"],

                "stage2_cache_total_tokens": cache_stage2["cache_total_tokens"],
                "stage2_cache_hit_tokens": cache_stage2["cache_hit_tokens"],
                "stage2_cache_computed_tokens": cache_stage2["cache_computed_tokens"],
                "stage2_cache_need_to_load": cache_stage2["cache_need_to_load"],
                "stage2_cache_hit_rate": cache_stage2["cache_hit_rate"],
            }

            row.update(flat_stage1_metrics)
            row.update(flat_stage2_metrics)
            rows.append(row)

            print(
                f"mode={mode}, prefix={prefix_tokens}, rep={rep}, "
                f"s1_ms={row['stage1_latency_ms']:.1f}, "
                f"s2_ms={row['stage2_latency_ms']:.1f}, "
                f"e2e_ms={row['e2e_two_stage_ms']:.1f}, "
                f"s2_hit={row.get('stage2_cache_hit_tokens')}/{row.get('stage2_cache_total_tokens')}, "
                f"s2_rate={row.get('stage2_cache_hit_rate')}"
            )

            fieldnames = sorted(set().union(*(r.keys() for r in rows)))
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    print(f"\nSaved results to: {out_csv}")
    summarize_results(rows, mode)

    if STOP_SERVER_AFTER_RUN:
        stop_existing_server()