#!/usr/bin/env python3
"""
Chunk reuse experiment for LMCache + vLLM, with optional auto-start of server.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
from lmcache.v1.token_database import ChunkedTokenDatabase
from transformers import AutoTokenizer


# ----------------------------
# Prometheus parsing utilities
# ----------------------------

def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            key, value = line.rsplit(" ", 1)
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def metric_name_variants(metric_name: str) -> List[str]:
    """
    Build compatible metric-name variants for different exporters:
    - original form, e.g. lmcache:num_hit_tokens
    - prometheus counter suffix form, e.g. lmcache:num_hit_tokens_total
    - sanitized form, e.g. lmcache_num_hit_tokens
    - sanitized counter suffix form, e.g. lmcache_num_hit_tokens_total
    """
    base = metric_name
    base_sanitized = metric_name.replace(":", "_")
    variants = [base, f"{base}_total", base_sanitized, f"{base_sanitized}_total"]
    # Keep order and deduplicate.
    return list(dict.fromkeys(variants))


def fetch_metrics(base_url: str, timeout: float = 10.0) -> Dict[str, float]:
    text = fetch_metrics_text(base_url, timeout=timeout)
    return parse_prometheus_metrics(text)


def fetch_metrics_text(base_url: str, timeout: float = 10.0) -> str:
    url = f"{base_url.rstrip('/')}/metrics"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def sum_metric(metrics: Dict[str, float], metric_prefix: str) -> float:
    total = 0.0
    for k, v in metrics.items():
        if k == metric_prefix or k.startswith(metric_prefix + "{"):
            total += v
    return total


def sum_metric_compatible(metrics: Dict[str, float], metric_name: str) -> float:
    total = 0.0
    for metric_prefix in metric_name_variants(metric_name):
        total += sum_metric(metrics, metric_prefix)
    return total


def latest_gauge(metrics: Dict[str, float], metric_prefix: str) -> float:
    return sum_metric_compatible(metrics, metric_prefix)


def snapshot_selected_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "lmcache_num_hit_tokens": sum_metric_compatible(metrics, "lmcache:num_hit_tokens"),
        "lmcache_num_lookup_hits": sum_metric_compatible(metrics, "lmcache:num_lookup_hits"),
        "lmcache_num_vllm_hit_tokens": sum_metric_compatible(
            metrics, "lmcache:num_vllm_hit_tokens"
        ),
        "lmcache_retrieve_hit_rate": latest_gauge(metrics, "lmcache:retrieve_hit_rate"),
        "lmcache_lookup_hit_rate": latest_gauge(metrics, "lmcache:lookup_hit_rate"),
        "lmcache_time_to_retrieve_sum": sum_metric_compatible(
            metrics, "lmcache:time_to_retrieve_sum"
        ),
        "lmcache_time_to_retrieve_count": sum_metric_compatible(
            metrics, "lmcache:time_to_retrieve_count"
        ),
        "lmcache_time_to_store_sum": sum_metric_compatible(metrics, "lmcache:time_to_store_sum"),
        "lmcache_time_to_store_count": sum_metric_compatible(
            metrics, "lmcache:time_to_store_count"
        ),
        "lmcache_time_to_lookup_sum": sum_metric_compatible(
            metrics, "lmcache:time_to_lookup_sum"
        ),
        "lmcache_time_to_lookup_count": sum_metric_compatible(
            metrics, "lmcache:time_to_lookup_count"
        ),
        "lmcache_local_cache_usage": latest_gauge(metrics, "lmcache:local_cache_usage"),
        "lmcache_remote_cache_usage": latest_gauge(metrics, "lmcache:remote_cache_usage"),
        "lmcache_local_storage_usage": latest_gauge(metrics, "lmcache:local_storage_usage"),
        "lmcache_active_memory_objs_count": latest_gauge(
            metrics, "lmcache:active_memory_objs_count"
        ),
        "lmcache_pinned_memory_objs_count": latest_gauge(
            metrics, "lmcache:pinned_memory_objs_count"
        ),
        "lmcache_local_cpu_hot_cache_count": latest_gauge(
            metrics, "lmcache:local_cpu_hot_cache_count"
        ),
        "lmcache_local_cpu_keys_in_request_count": latest_gauge(
            metrics, "lmcache:local_cpu_keys_in_request_count"
        ),
    }


GAUGE_ABSOLUTE_KEYS = [
    "lmcache_retrieve_hit_rate",
    "lmcache_lookup_hit_rate",
    "lmcache_local_cache_usage",
    "lmcache_remote_cache_usage",
    "lmcache_local_storage_usage",
    "lmcache_active_memory_objs_count",
    "lmcache_pinned_memory_objs_count",
    "lmcache_local_cpu_hot_cache_count",
    "lmcache_local_cpu_keys_in_request_count",
]


def print_metrics_probe(metrics_base_url: str, timeout: float) -> None:
    try:
        metrics = fetch_metrics(metrics_base_url, timeout=timeout)
    except Exception as e:
        print(f"[Metrics Probe] failed to fetch metrics from {metrics_base_url}: {e}")
        return

    lmcache_keys = [k for k in metrics if "lmcache" in k.lower()]
    nonzero_keys = [(k, v) for k, v in metrics.items() if "lmcache" in k.lower() and v != 0.0]
    print(
        f"[Metrics Probe] endpoint={metrics_base_url} "
        f"lmcache_metric_count={len(lmcache_keys)} nonzero_lmcache_metric_count={len(nonzero_keys)}"
    )

    interesting = [
        "lmcache:num_hit_tokens",
        "lmcache:num_lookup_hits",
        "lmcache:num_vllm_hit_tokens",
        "lmcache:retrieve_hit_rate",
        "lmcache:lookup_hit_rate",
    ]
    for name in interesting:
        print(f"[Metrics Probe] {name}={sum_metric_compatible(metrics, name)}")


def save_raw_metrics_snapshot(
    metrics_base_url: str,
    timeout: float,
    output_path: Optional[str],
) -> None:
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    text = fetch_metrics_text(metrics_base_url, timeout=timeout)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def post_json(base_url: str, path: str, timeout: float = 10.0) -> Dict[str, object]:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_json(base_url: str, path: str, timeout: float = 10.0) -> Dict[str, object]:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def control_chunk_statistics(chunk_stats_base_url: str, action: str, timeout: float) -> None:
    path = f"/chunk_statistics/{action}"
    result = post_json(chunk_stats_base_url, path, timeout=timeout)
    if result.get("status") != "success":
        raise RuntimeError(f"chunk_statistics {action} failed: {result}")


def get_chunk_statistics_status(
    chunk_stats_base_url: str, timeout: float
) -> Dict[str, object]:
    return get_json(chunk_stats_base_url, "/chunk_statistics/status", timeout=timeout)


def read_chunk_hash_records(output_dir: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    root = Path(output_dir)
    if not root.exists():
        return out
    for file_path in sorted(root.glob("chunk_hashes_*.jsonl")):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        out.append(item)
        except Exception:
            continue
    return out


def collect_chunk_hashes(records: List[Dict[str, object]]) -> set[str]:
    hashes: set[str] = set()
    for rec in records:
        rec_hashes = rec.get("chunk_hashes", [])
        if isinstance(rec_hashes, list):
            for h in rec_hashes:
                if isinstance(h, str):
                    hashes.add(h)
    return hashes


def clear_chunk_hash_files(output_dir: str) -> int:
    root = Path(output_dir)
    if not root.exists():
        return 0
    removed = 0
    for file_path in root.glob("chunk_hashes_*.jsonl"):
        try:
            file_path.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def delta_metrics(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in sorted(set(before) | set(after)):
        out[k] = after.get(k, 0.0) - before.get(k, 0.0)
    return out


# ----------------------------
# Prompt construction utilities
# ----------------------------

def make_exact_token_ids(
    tokenizer: AutoTokenizer,
    target_tokens: int,
    unit_text: str = "tool result value success explanation ",
) -> List[int]:
    if target_tokens <= 0:
        return []

    current_ids: List[int] = []

    unit_ids = tokenizer.encode(unit_text, add_special_tokens=False)
    if not unit_ids:
        raise ValueError("unit_text tokenizes to empty ids, please change unit_text.")

    while len(current_ids) + len(unit_ids) <= target_tokens:
        current_ids.extend(unit_ids)

    remainder = target_tokens - len(current_ids)
    if remainder > 0:
        filler = "x " * (remainder * 4)
        filler_ids = tokenizer.encode(filler, add_special_tokens=False)[:remainder]
        current_ids.extend(filler_ids)

    return current_ids[:target_tokens]


def token_len_ids(token_ids: List[int]) -> int:
    return len(token_ids)


def make_segment_token_ids(
    tokenizer: AutoTokenizer,
    label: str,
    target_tokens: int,
) -> List[int]:
    """
    Build a labeled token-id segment with exactly target_tokens length.
    """
    prefix_ids = tokenizer.encode(f"{label}: ", add_special_tokens=False)
    if len(prefix_ids) >= target_tokens:
        return prefix_ids[:target_tokens]

    body_tokens = target_tokens - len(prefix_ids)
    body_ids = make_exact_token_ids(
        tokenizer=tokenizer,
        target_tokens=body_tokens,
        unit_text=f"{label.lower()} context detail evidence rationale ",
    )
    return (prefix_ids + body_ids)[:target_tokens]


def build_requests(
    tokenizer: AutoTokenizer,
    chunk0_token_ids: List[int],
    chunk_token_ids: List[int],
    segment_tokens: int = 256,
) -> Dict[str, List[int]]:
    a0 = make_segment_token_ids(tokenizer, "A0", segment_tokens)
    b0 = make_segment_token_ids(tokenizer, "B0", segment_tokens)
    b00 = make_segment_token_ids(tokenizer, "B00", segment_tokens)

    a1 = make_segment_token_ids(tokenizer, "A1", segment_tokens)
    b1 = make_segment_token_ids(tokenizer, "B1", segment_tokens)

    b2 = make_segment_token_ids(tokenizer, "B2", segment_tokens)

    a = make_segment_token_ids(tokenizer, "A", segment_tokens)
    b = make_segment_token_ids(tokenizer, "B", segment_tokens)

    a5 = make_segment_token_ids(tokenizer, "A5", segment_tokens)
    b5 = make_segment_token_ids(tokenizer, "B5", segment_tokens)

    return {
        "R0": a0 + chunk0_token_ids + b0,
        "R00": a0 + chunk_token_ids + b00,
        "R1": a1 + chunk_token_ids + b1,
        "R2": a1 + chunk_token_ids + b2,
        "R3": a + b + chunk_token_ids,
        "R4": chunk_token_ids + a + b,
        "R5": chunk_token_ids + a5 + b5,
    }


def split_token_ids_by_pattern(
    token_ids: List[int],
    pattern_ids: List[int],
) -> List[List[int]]:
    """Split token IDs by exact pattern; keep empty segments for leading/trailing separators."""
    if not pattern_ids:
        return [token_ids]
    out: List[List[int]] = []
    n = len(token_ids)
    m = len(pattern_ids)
    start = 0
    i = 0
    while i <= n - m:
        if token_ids[i : i + m] == pattern_ids:
            out.append(token_ids[start:i])
            start = i + m
            i = start
            continue
        i += 1
    out.append(token_ids[start:])
    return out


def to_hex_hash(hash_val: int) -> str:
    if hash_val < 0:
        hash_val = hash_val & ((1 << 64) - 1)
    return hex(hash_val)


def compute_segment_hashes_hex(
    token_ids: List[int],
    segment_sep_ids: List[int],
    hash_db: ChunkedTokenDatabase,
) -> List[str]:
    segs = split_token_ids_by_pattern(token_ids, segment_sep_ids)
    return [to_hex_hash(hash_db._hash_tokens(seg)) for seg in segs]


def get_lmcache_blend_sep_ids(
    tokenizer: AutoTokenizer,
    blend_special_str: str,
) -> List[int]:
    """
    Match LMCache SegmentTokenDatabase behavior:
    sep_tokens = tokenizer.encode(config.blend_special_str)[1:]
    """
    encoded = tokenizer.encode(blend_special_str)
    if not encoded:
        return []
    return encoded[1:]


# ----------------------------
# Server management
# ----------------------------

def is_server_ready(base_url: str, timeout: float = 3.0) -> bool:
    checks = [
        f"{base_url.rstrip('/')}/",
        f"{base_url.rstrip('/')}/health",
        f"{base_url.rstrip('/')}/v1/models",
        f"{base_url.rstrip('/')}/metrics",
    ]
    for url in checks:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False


def read_log_tail(path: str, max_lines: int = 80) -> str:
    if not os.path.exists(path):
        return "(log file not found)"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip() or "(log file is empty)"
    except Exception as e:
        return f"(failed to read log tail: {e})"


def wait_for_server(
    base_url: str,
    ready_timeout: float,
    proc: Optional[subprocess.Popen] = None,
    server_log: Optional[str] = None,
) -> None:
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            exit_code = proc.returncode
            log_tail = read_log_tail(server_log) if server_log else "(no server_log provided)"
            raise RuntimeError(
                f"Server process exited early with code {exit_code} before becoming ready.\n"
                f"base_url={base_url}\n"
                f"server_log={server_log}\n"
                f"--- server log tail ---\n{log_tail}"
            )
        if is_server_ready(base_url):
            return
        time.sleep(2.0)
    log_tail = read_log_tail(server_log) if server_log else "(no server_log provided)"
    raise RuntimeError(
        f"Server did not become ready within {ready_timeout} seconds: {base_url}\n"
        f"server_log={server_log}\n"
        f"--- server log tail ---\n{log_tail}"
    )


def start_vllm_server(
    model: str,
    port: int,
    server_log: str,
    gpu_memory_utilization: float,
    extra_args: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]

    if extra_args:
        cmd.extend(extra_args)

    log_f = open(server_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    proc._log_file_handle = log_f  # type: ignore[attr-defined]
    return proc


def stop_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        try:
            if hasattr(proc, "_log_file_handle"):
                proc._log_file_handle.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    finally:
        try:
            if hasattr(proc, "_log_file_handle"):
                proc._log_file_handle.close()  # type: ignore[attr-defined]
        except Exception:
            pass


# ----------------------------
# Request execution
# ----------------------------

@dataclass
class RequestResult:
    request_name: str
    prompt_tokens: int
    elapsed_seconds: float
    completion_text: str
    metrics_delta: Dict[str, float]
    metrics_after: Dict[str, float]
    chunk_observation: Dict[str, object]


def run_completion(
    base_url: str,
    model: str,
    prompt_token_ids: List[int],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Tuple[float, str]:
    url = f"{base_url.rstrip('/')}/v1/completions"
    # vLLM OpenAI-compatible APIs differ by version. Try common token-id forms.
    candidate_payloads = [
        {
            "model": model,
            "prompt_token_ids": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        {
            "model": model,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        {
            "model": model,
            "prompt": [prompt_token_ids],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    ]

    last_error: Optional[Exception] = None
    last_body = ""
    for payload in candidate_payloads:
        start = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        if resp.status_code == 200:
            data = resp.json()
            text = ""
            if "choices" in data and data["choices"]:
                text = data["choices"][0].get("text", "")
            return elapsed, text

        try:
            last_body = resp.text
            resp.raise_for_status()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        "All completion payload formats failed. "
        f"last_error={last_error}; response_body={last_body[:1000]}"
    )


def run_one_request(
    base_url: str,
    metrics_base_url: str,
    model: str,
    tokenizer: AutoTokenizer,
    request_name: str,
    prompt_token_ids: List[int],
    max_tokens: int,
    temperature: float,
    timeout: float,
    sleep_between_requests: float,
    before_raw_metrics_path: Optional[str] = None,
    after_raw_metrics_path: Optional[str] = None,
    chunk_statistics_enabled: bool = False,
    chunk_hash_output_dir: Optional[str] = None,
    chunk_details_dir: Optional[str] = None,
    chunk_detail_prefix: Optional[str] = None,
    known_segment_hashes: Optional[Set[str]] = None,
    segment_sep_ids: Optional[List[int]] = None,
    segment_hash_db: Optional[ChunkedTokenDatabase] = None,
) -> RequestResult:
    before_chunk_records: List[Dict[str, object]] = []
    before_chunk_hashes: set[str] = set()
    if chunk_statistics_enabled and chunk_hash_output_dir:
        before_chunk_records = read_chunk_hash_records(chunk_hash_output_dir)
        before_chunk_hashes = collect_chunk_hashes(before_chunk_records)

    save_raw_metrics_snapshot(
        metrics_base_url=metrics_base_url,
        timeout=timeout,
        output_path=before_raw_metrics_path,
    )
    before = snapshot_selected_metrics(fetch_metrics(metrics_base_url, timeout=timeout))
    elapsed, text = run_completion(
        base_url=base_url,
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    if sleep_between_requests > 0:
        time.sleep(sleep_between_requests)
    save_raw_metrics_snapshot(
        metrics_base_url=metrics_base_url,
        timeout=timeout,
        output_path=after_raw_metrics_path,
    )
    after = snapshot_selected_metrics(fetch_metrics(metrics_base_url, timeout=timeout))
    delta = delta_metrics(before, after)
    chunk_observation: Dict[str, object] = {}
    segment_detail_data: Optional[Dict[str, object]] = None

    if (
        known_segment_hashes is not None
        and segment_sep_ids is not None
        and segment_hash_db is not None
    ):
        segment_hashes = compute_segment_hashes_hex(
            token_ids=prompt_token_ids,
            segment_sep_ids=segment_sep_ids,
            hash_db=segment_hash_db,
        )
        segment_hashes_set = set(segment_hashes)
        segment_hit_hashes = sorted(segment_hashes_set & known_segment_hashes)
        segment_new_hashes = sorted(segment_hashes_set - known_segment_hashes)
        known_before = len(known_segment_hashes)
        known_segment_hashes.update(segment_hashes_set)

        chunk_observation.update(
            {
                "segment_hashes_count": len(segment_hashes),
                "segment_unique_hashes_count": len(segment_hashes_set),
                "segment_new_hashes_count": len(segment_new_hashes),
                "segment_hit_hashes_count": len(segment_hit_hashes),
                "segment_known_hashes_before_count": known_before,
                "segment_known_hashes_after_count": len(known_segment_hashes),
            }
        )
        segment_detail_data = {
            "segment_hashes": segment_hashes,
            "segment_new_hashes": segment_new_hashes,
            "segment_hit_hashes": segment_hit_hashes,
            "segment_separator_token_ids": segment_sep_ids,
        }

    if chunk_statistics_enabled and chunk_hash_output_dir:
        after_chunk_records = read_chunk_hash_records(chunk_hash_output_dir)
        # Record strategy appends jsonl records; take newly appended slice as this request.
        appended_records = after_chunk_records[len(before_chunk_records) :]
        request_hashes = collect_chunk_hashes(appended_records)
        newly_added_hashes = sorted(request_hashes - before_chunk_hashes)
        hit_hashes = sorted(request_hashes & before_chunk_hashes)

        chunk_observation = {
            "request_chunk_hashes_count": len(request_hashes),
            "new_chunk_hashes_count": len(newly_added_hashes),
            "hit_chunk_hashes_count": len(hit_hashes),
            "known_unique_hashes_before_count": len(before_chunk_hashes),
            "known_unique_hashes_after_count": len(collect_chunk_hashes(after_chunk_records)),
        }

        if chunk_details_dir:
            os.makedirs(chunk_details_dir, exist_ok=True)
            safe_prefix = chunk_detail_prefix or request_name
            detail_path = os.path.join(
                chunk_details_dir, f"{safe_prefix}_{int(time.time() * 1000)}.json"
            )
            detail_data = {
                "request_name": request_name,
                "request_chunk_hashes": sorted(request_hashes),
                "new_chunk_hashes": newly_added_hashes,
                "hit_chunk_hashes": hit_hashes,
                "appended_records_count": len(appended_records),
                "segment_observation": segment_detail_data,
            }
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(detail_data, f, indent=2)
            chunk_observation["chunk_details_path"] = detail_path
    elif segment_detail_data is not None and chunk_details_dir:
        os.makedirs(chunk_details_dir, exist_ok=True)
        safe_prefix = chunk_detail_prefix or request_name
        detail_path = os.path.join(
            chunk_details_dir, f"{safe_prefix}_{int(time.time() * 1000)}.json"
        )
        detail_data = {
            "request_name": request_name,
            "segment_observation": segment_detail_data,
        }
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_data, f, indent=2)
        chunk_observation["chunk_details_path"] = detail_path

    return RequestResult(
        request_name=request_name,
        prompt_tokens=token_len_ids(prompt_token_ids),
        elapsed_seconds=elapsed,
        completion_text=text,
        metrics_delta=delta,
        metrics_after=after,
        chunk_observation=chunk_observation,
    )


# ----------------------------
# Experiment loop
# ----------------------------

def run_experiment(args: argparse.Namespace) -> List[Dict[str, object]]:
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    resolved_blend_special_str = (
        args.blend_special_str if args.blend_special_str is not None else args.separator
    )
    segment_sep_ids = get_lmcache_blend_sep_ids(tokenizer, resolved_blend_special_str)
    segment_hash_db = ChunkedTokenDatabase()
    if args.mode == "blending":
        print(
            f"[Segment Debug] blend_special_str={resolved_blend_special_str!r} "
            f"segment_sep_ids={segment_sep_ids} sep_len={len(segment_sep_ids)}"
        )

    chunk_lengths = [int(x) for x in args.chunk_token_lengths.split(",") if x.strip()]
    rows: List[Dict[str, object]] = []
    valid_cases: List[Tuple[int, List[int], Dict[str, List[int]], int]] = []

    for chunk_len in chunk_lengths:
        chunk0_token_ids = make_exact_token_ids(
            tokenizer,
            chunk_len,
            unit_text="chunk0 independent baseline context value ",
        )
        chunk_token_ids = make_exact_token_ids(
            tokenizer,
            chunk_len,
            unit_text="tool result value success explanation ",
        )
        if chunk0_token_ids == chunk_token_ids:
            # Ensure R0 chunk is distinct from shared chunk.
            chunk0_token_ids = make_exact_token_ids(
                tokenizer,
                chunk_len,
                unit_text="chunk0 unique unrelated baseline signal ",
            )
        requests_map = build_requests(
            tokenizer=tokenizer,
            chunk0_token_ids=chunk0_token_ids,
            chunk_token_ids=chunk_token_ids,
            segment_tokens=args.segment_token_length,
        )
        max_prompt_tokens = max(token_len_ids(p) for p in requests_map.values())
        total_budget = max_prompt_tokens + args.max_tokens

        if args.max_model_len is not None and total_budget > args.max_model_len:
            print(
                f"[Skip] chunk={chunk_len} exceeds context budget: "
                f"max_prompt_tokens={max_prompt_tokens}, max_tokens={args.max_tokens}, "
                f"total={total_budget}, max_model_len={args.max_model_len}"
            )
            continue

        valid_cases.append((chunk_len, chunk_token_ids, requests_map, max_prompt_tokens))

    if not valid_cases:
        raise ValueError(
            "No valid chunk lengths remain after max_model_len budget checks. "
            "Try reducing --chunk-token-lengths or --max-tokens, or increasing --max-model-len."
        )

    for chunk_len, _chunk_token_ids, requests_map, max_prompt_tokens in valid_cases:
        print(
            f"[Case] chunk={chunk_len} | max_prompt_tokens={max_prompt_tokens} "
            f"| max_tokens={args.max_tokens} | total_budget={max_prompt_tokens + args.max_tokens}"
        )

        for repeat_idx in range(args.repeats):
            run_id = f"{args.mode}_chunk{chunk_len}_rep{repeat_idx+1}"
            known_segment_hashes: Set[str] = set()

            for request_name in ["R0", "R00", "R1", "R2", "R3", "R4", "R5"]:
                prompt_token_ids = requests_map[request_name]
                result = run_one_request(
                    base_url=args.base_url,
                    metrics_base_url=args.metrics_base_url,
                    model=args.model,
                    tokenizer=tokenizer,
                    request_name=request_name,
                    prompt_token_ids=prompt_token_ids,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    sleep_between_requests=args.sleep_between_requests,
                    before_raw_metrics_path=(
                        os.path.join(
                            args.raw_metrics_dir,
                            f"{run_id}_{request_name}_before.prom",
                        )
                        if args.raw_metrics_dir
                        else None
                    ),
                    after_raw_metrics_path=(
                        os.path.join(
                            args.raw_metrics_dir,
                            f"{run_id}_{request_name}_after.prom",
                        )
                        if args.raw_metrics_dir
                        else None
                    ),
                    chunk_statistics_enabled=args.enable_chunk_statistics,
                    chunk_hash_output_dir=args.chunk_hash_output_dir,
                    chunk_details_dir=args.chunk_details_dir,
                    chunk_detail_prefix=f"{run_id}_{request_name}",
                    known_segment_hashes=known_segment_hashes,
                    segment_sep_ids=segment_sep_ids,
                    segment_hash_db=segment_hash_db,
                )

                row: Dict[str, object] = {
                    "mode": args.mode,
                    "run_id": run_id,
                    "repeat": repeat_idx + 1,
                    "request_name": request_name,
                    "chunk_token_length": chunk_len,
                    "prompt_tokens": result.prompt_tokens,
                    "elapsed_seconds": result.elapsed_seconds,
                    "completion_preview": result.completion_text[:120].replace("\n", " "),
                }
                row.update(result.metrics_delta)
                for gauge_key in GAUGE_ABSOLUTE_KEYS:
                    row[f"{gauge_key}_after"] = result.metrics_after.get(gauge_key, 0.0)
                row.update(result.chunk_observation)
                rows.append(row)

    return rows


def save_csv(rows: List[Dict[str, object]], output_csv: str) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["request_name"]), int(row["chunk_token_length"]))
        grouped.setdefault(key, []).append(row)

    print("\n=== Summary ===")
    for (request_name, chunk_len), group in sorted(grouped.items()):
        elapsed = [float(r["elapsed_seconds"]) for r in group]
        hit_tokens = [float(r.get("lmcache_num_hit_tokens", 0.0)) for r in group]
        lookup_hits = [float(r.get("lmcache_num_lookup_hits", 0.0)) for r in group]
        print(
            f"{request_name} | chunk={chunk_len:4d} | "
            f"latency_mean={statistics.mean(elapsed):.4f}s | "
            f"hit_tokens_delta_mean={statistics.mean(hit_tokens):.2f} | "
            f"lookup_hits_delta_mean={statistics.mean(lookup_hits):.2f}"
        )


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk reuse experiment with LMCache observability metrics."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--metrics-base-url",
        default=None,
        help="Metrics endpoint base URL. If omitted, auto-resolved (prefer LMCache internal API).",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--mode", required=True, choices=["prefix_only", "blending"])
    parser.add_argument("--output-csv", default="chunk_reuse_metrics_results.csv")
    parser.add_argument("--chunk-token-lengths", default="128,256,512")
    parser.add_argument(
        "--segment-token-length",
        type=int,
        default=256,
        help="Approximate token length for each context segment (A/A1/A2/A5/B/B1/B2/B5).",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--separator",
        default=" # # ",
        help="Separator inserted between logical segments/chunks in prompt token IDs.",
    )
    parser.add_argument(
        "--blend-special-str",
        default=None,
        help=(
            "Special delimiter string for blending. If omitted, uses --separator exactly "
            "(no strip) so tokenizer boundary matches prompt construction."
        ),
    )
    parser.add_argument(
        "--blend-check-layers",
        default="1",
        help="Passed to LMCACHE_BLEND_CHECK_LAYERS in blending mode. Example: '1' or '1,3,5'.",
    )
    parser.add_argument(
        "--blend-recompute-ratios",
        default="0.15",
        help="Passed to LMCACHE_BLEND_RECOMPUTE_RATIOS in blending mode. Example: '0.15'.",
    )
    parser.add_argument(
        "--blend-thresholds",
        default=None,
        help="Optional LMCACHE_BLEND_THRESHOLDS in blending mode. Example: '0.1' or '0.1,0.2'.",
    )
    parser.add_argument(
        "--disable-blend-layerwise",
        action="store_true",
        help="Disable LMCACHE_USE_LAYERWISE in blending mode (enabled by default).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.5)
    parser.add_argument(
        "--raw-metrics-dir",
        default="raw_metrics",
        help="Directory to dump raw Prometheus snapshots before/after each request. Set empty string to disable.",
    )
    parser.add_argument(
        "--enable-chunk-statistics",
        action="store_true",
        help="Enable LMCache chunk-level statistics collection.",
    )
    parser.add_argument(
        "--disable-chunk-statistics",
        action="store_true",
        help="Disable LMCache chunk-level statistics collection.",
    )
    parser.add_argument(
        "--chunk-statistics-strategy",
        default="file_hash",
        choices=["file_hash", "memory_bloom_filter"],
        help="Chunk statistics strategy for LMCache.",
    )
    parser.add_argument(
        "--chunk-hash-output-dir",
        default="chunk_hashes",
        help="Output dir for file_hash chunk statistics records.",
    )
    parser.add_argument(
        "--chunk-details-dir",
        default="chunk_details",
        help="Output dir for per-request chunk-level analysis results.",
    )
    parser.add_argument(
        "--chunk-stats-base-url",
        default=None,
        help="Chunk statistics API base URL. Defaults to metrics base URL.",
    )

    # Auto-start server options
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-log", default="vllm_server.log")
    parser.add_argument("--server-ready-timeout", type=float, default=600.0)
    parser.add_argument(
        "--internal-api-server-port-start",
        type=int,
        default=6999,
        help="LMCache internal API server base port (scheduler). Worker0 uses +1.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--kv-transfer-config",
        default='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
        help="Passed directly to vllm serve --kv-transfer-config",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Pass --enable-prefix-caching to vllm serve",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=65536,
        help="--max-model-len for vllm serve. Default lowered for KV-memory stability.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass --enforce-eager to vllm serve (disables cudagraph wrappers).",
    )

    parser.set_defaults(enable_chunk_statistics=True)
    args = parser.parse_args()
    if args.disable_chunk_statistics:
        args.enable_chunk_statistics = False
    return args


def normalize_kv_transfer_config(raw_config: str) -> str:
    """
    Compatibility shim:
    Some older examples use LMCacheConnectorV1Dynamic, but newer vLLM builds
    register LMCacheConnectorV1.
    """
    try:
        parsed = json.loads(raw_config)
    except Exception:
        return raw_config

    if isinstance(parsed, dict) and parsed.get("kv_connector") == "LMCacheConnectorV1Dynamic":
        parsed["kv_connector"] = "LMCacheConnectorV1"
        print(
            "[Compat] Rewriting kv_connector from LMCacheConnectorV1Dynamic "
            "to LMCacheConnectorV1 for this vLLM build."
        )
        return json.dumps(parsed)
    return raw_config


def build_server_extra_args(args: argparse.Namespace) -> List[str]:
    normalized_kv_config = normalize_kv_transfer_config(args.kv_transfer_config)
    extra_args: List[str] = [
        "--kv-transfer-config",
        normalized_kv_config,
    ]
    if args.enable_prefix_caching:
        extra_args.append("--enable-prefix-caching")
    if args.max_model_len is not None:
        extra_args.extend(["--max-model-len", str(args.max_model_len)])
    if args.enforce_eager or args.mode == "blending":
        # Blending path expects the raw model class (e.g., LlamaForCausalLM),
        # while cudagraph may wrap it as CUDAGraphWrapper and break LMCache
        # model-type inference.
        extra_args.append("--enforce-eager")
    return extra_args


def build_env_for_mode(
    mode: str,
    separator: str,
    blend_special_str: Optional[str],
    blend_check_layers: str,
    blend_recompute_ratios: str,
    blend_thresholds: Optional[str],
    disable_blend_layerwise: bool,
    internal_api_server_port_start: int,
    enable_chunk_statistics: bool,
    chunk_statistics_strategy: str,
    chunk_hash_output_dir: str,
) -> Dict[str, str]:
    """
    You can edit these defaults based on your environment.
    """
    env = {
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5.0",
        "LMCACHE_INTERNAL_API_SERVER_ENABLED": "True",
        "LMCACHE_INTERNAL_API_SERVER_PORT_START": str(internal_api_server_port_start),
        "LMCACHE_ENABLE_CHUNK_STATISTICS": "True"
        if enable_chunk_statistics
        else "False",
        "LMCACHE_CHUNK_STATISTICS_STRATEGY": chunk_statistics_strategy,
    }
    # Keep hash behavior stable across processes, following blendsmall.py practice.
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    extra_config = {}
    if chunk_statistics_strategy == "file_hash":
        extra_config["chunk_statistics_file_output_dir"] = chunk_hash_output_dir
    if extra_config:
        env["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)

    if mode == "blending":
        resolved_special_str = (
            blend_special_str
            if blend_special_str is not None
            else separator
        )
        env["LMCACHE_ENABLE_BLENDING"] = "True"
        env["LMCACHE_BLEND_SPECIAL_STR"] = resolved_special_str
        env["LMCACHE_USE_LAYERWISE"] = "False" if disable_blend_layerwise else "True"
        env["LMCACHE_BLEND_CHECK_LAYERS"] = blend_check_layers
        env["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = blend_recompute_ratios
        if blend_thresholds is not None and blend_thresholds.strip() != "":
            env["LMCACHE_BLEND_THRESHOLDS"] = blend_thresholds
    else:
        env["LMCACHE_ENABLE_BLENDING"] = "False"

    return env


def resolve_metrics_base_url(args: argparse.Namespace) -> str:
    if args.metrics_base_url:
        return args.metrics_base_url
    if args.start_server:
        # In single-worker setups, scheduler uses port_start and worker0 uses port_start + 1.
        worker0_port = args.internal_api_server_port_start + 1
        return f"http://127.0.0.1:{worker0_port}"
    return args.base_url


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    args = parse_args()
    server_proc: Optional[subprocess.Popen] = None
    if args.raw_metrics_dir == "":
        args.raw_metrics_dir = None
    if args.chunk_hash_output_dir == "":
        args.chunk_hash_output_dir = None
    if args.chunk_details_dir == "":
        args.chunk_details_dir = None
    args.metrics_base_url = resolve_metrics_base_url(args)
    if args.chunk_stats_base_url is None:
        args.chunk_stats_base_url = args.metrics_base_url

    try:
        if args.start_server:
            args.base_url = f"http://127.0.0.1:{args.server_port}"
            extra_args = build_server_extra_args(args)
            resolved_blend_special_str = (
                args.blend_special_str
                if args.blend_special_str is not None
                else args.separator
            )
            env_overrides = build_env_for_mode(
                args.mode,
                args.separator,
                args.blend_special_str,
                args.blend_check_layers,
                args.blend_recompute_ratios,
                args.blend_thresholds,
                args.disable_blend_layerwise,
                args.internal_api_server_port_start,
                args.enable_chunk_statistics,
                args.chunk_statistics_strategy,
                args.chunk_hash_output_dir or "chunk_hashes",
            )

            print(f"Starting vLLM server on {args.base_url} ...")
            print(f"Collecting metrics from {args.metrics_base_url} ...")
            print(f"Chunk statistics API at {args.chunk_stats_base_url} ...")
            if args.mode == "blending":
                print(f"Using blend special string: {resolved_blend_special_str!r}")
                print(
                    "Blending env: "
                    f"LMCACHE_USE_LAYERWISE={'False' if args.disable_blend_layerwise else 'True'}, "
                    f"LMCACHE_BLEND_CHECK_LAYERS={args.blend_check_layers}, "
                    f"LMCACHE_BLEND_RECOMPUTE_RATIOS={args.blend_recompute_ratios}, "
                    f"LMCACHE_BLEND_THRESHOLDS={args.blend_thresholds}"
                )
                if resolved_blend_special_str not in args.separator:
                    print(
                        "[Warning] blend special string does not appear in --separator; "
                        "blending chunk boundary detection may fail."
                    )
            server_proc = start_vllm_server(
                model=args.model,
                port=args.server_port,
                server_log=args.server_log,
                gpu_memory_utilization=args.gpu_memory_utilization,
                extra_args=extra_args,
                env_overrides=env_overrides,
            )
            wait_for_server(
                args.base_url,
                args.server_ready_timeout,
                proc=server_proc,
                server_log=args.server_log,
            )
            print("Server is ready.")

        else:
            print(f"Collecting metrics from {args.metrics_base_url} ...")
            print(f"Chunk statistics API at {args.chunk_stats_base_url} ...")
            wait_for_server(args.base_url, 30.0)

        if args.enable_chunk_statistics:
            if args.chunk_statistics_strategy == "file_hash" and args.chunk_hash_output_dir:
                removed = clear_chunk_hash_files(args.chunk_hash_output_dir)
                print(
                    f"[ChunkStats] cleaned {removed} old file_hash record files "
                    f"from {args.chunk_hash_output_dir}"
                )
            control_chunk_statistics(args.chunk_stats_base_url, "reset", args.timeout)
            control_chunk_statistics(args.chunk_stats_base_url, "start", args.timeout)
            status = get_chunk_statistics_status(args.chunk_stats_base_url, args.timeout)
            print(
                "[ChunkStats] enabled="
                f"{status.get('enabled')} strategy={args.chunk_statistics_strategy} "
                f"total_chunks={status.get('total_chunks')} unique_chunks={status.get('unique_chunks')}"
            )

        print_metrics_probe(args.metrics_base_url, args.timeout)
        rows = run_experiment(args)
        save_csv(rows, args.output_csv)
        print_summary(rows)
        print(f"\nSaved results to: {args.output_csv}")
        if args.enable_chunk_statistics:
            end_status = get_chunk_statistics_status(args.chunk_stats_base_url, args.timeout)
            print(
                "[ChunkStats End] "
                f"total_requests={end_status.get('total_requests')} "
                f"total_chunks={end_status.get('total_chunks')} "
                f"unique_chunks={end_status.get('unique_chunks')} "
                f"reuse_rate={end_status.get('reuse_rate')}"
            )

    finally:
        if args.start_server:
            print("Stopping server ...")
            stop_vllm_server(server_proc)


if __name__ == "__main__":
    main()
