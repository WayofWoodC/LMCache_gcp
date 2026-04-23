#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Motivation experiment for chunk KV reuse with LMCache blending.

Goal:
- Treat one fixed text chunk as a "tool-result-like" reusable chunk.
- Compare prefix-only reuse vs blending-based reuse.
- Check whether the same chunk can still be reused when inserted into
  different prompt contexts or different positions.

This version is designed for SINGLE-SERVER runs:
- Run once with a prefix-only server
- Run once with a blending-enabled server
- Compare the two output CSV files afterward

Assumptions:
- The server is already launched.
- Each run uses only one server endpoint.
- Prefix-only and blending are compared through two separate experiment runs.
- The same model/tokenizer is used across all runs.
- Chunk boundaries are explicitly marked by a fixed separator string.
- We do not consider actual tool-calls, tool-cache, reuse frequency,
  or output quality in this initial experiment.

Outputs:
- A CSV file with request latency, response text, and optional metrics delta.
"""

import os
import re
import csv
import json
import time
import uuid
import argparse
from typing import Dict, List, Optional

import requests
from transformers import AutoTokenizer


# ============================================================
# 0. Configuration helpers
# ============================================================

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the currently running server, e.g. http://127.0.0.1:8000"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prefix_only", "blending"],
        help="Experiment mode for the current run."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name served by vLLM, e.g. meta-llama/Llama-3.2-3B-Instruct"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path. Defaults to --model if not provided."
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="motivation_chunk_reuse.csv"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300
    )

    parser.add_argument(
        "--chunk-token-lengths",
        type=str,
        default="256,512,1024",
        help="Comma-separated chunk lengths in tokens, e.g. 256,512,1024"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64
    )

    parser.add_argument(
        "--separator",
        type=str,
        default=" # # ",
        help="Explicit separator string used for chunk boundaries."
    )

    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="If set, scrape /metrics before and after each request."
    )

    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=1.0
    )

    return parser


def ensure_dir_for_file(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ============================================================
# 1. HTTP helpers
# ============================================================

def completions_request(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict:
    """
    Send one completion request to a vLLM OpenAI-compatible endpoint.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    t1 = time.perf_counter()
    response.raise_for_status()
    data = response.json()

    return {
        "latency_ms": (t1 - t0) * 1000.0,
        "response_json": data,
    }


def get_text_from_completion_response(resp_json: Dict) -> str:
    """
    Extract text from OpenAI-style completion response.
    """
    try:
        return resp_json["choices"][0]["text"]
    except Exception:
        return ""


def scrape_metrics(base_url: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch raw Prometheus metrics from /metrics.
    """
    url = base_url.rstrip("/") + "/metrics"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


# ============================================================
# 2. Metrics parsing
# ============================================================

def parse_prometheus_metrics(metrics_text: str) -> Dict[str, float]:
    """
    Parse Prometheus metrics text into a dictionary.

    Notes:
    - This parser is intentionally generic.
    - It only keeps plain scalar lines of the form:
      metric_name value
    - Labels are ignored for simplicity.

    If your metrics contain multiple labeled variants for the same name,
    this function will keep the last one it sees.
    """
    result = {}
    if metrics_text is None:
        return result

    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Match patterns like:
        # metric_name 123
        # metric_name{label="x"} 123
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+eE0-9\.]+)$', line)
        if not m:
            continue

        metric_name = m.group(1)
        metric_value = m.group(3)

        try:
            result[metric_name] = float(metric_value)
        except ValueError:
            pass

    return result


def diff_metric_dict(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    """
    Compute after - before for metrics that exist in either dictionary.
    """
    keys = set(before.keys()) | set(after.keys())
    delta = {}
    for k in keys:
        delta[k] = after.get(k, 0.0) - before.get(k, 0.0)
    return delta


def pick_lmcache_related_metrics(delta_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Keep only metrics that look LMCache/vLLM-cache related.

    This function is intentionally broad because metric names may differ
    across versions. You can tighten it later after checking your actual /metrics.
    """
    keep = {}
    patterns = [
        "lmcache",
        "cache",
        "kv",
        "prefix",
        "blend",
        "load",
        "store",
        "hit",
        "miss",
        "token",
    ]

    for k, v in delta_metrics.items():
        lk = k.lower()
        if any(p in lk for p in patterns):
            keep[k] = v

    return keep


# ============================================================
# 3. Prompt/chunk construction
# ============================================================

def make_base_chunk_source_text() -> str:
    """
    Construct a sufficiently long reusable text source.
    This emulates a tool-result-like chunk.

    You can replace this with any fixed content you want.
    """
    paragraphs = [
        "The company reported steady revenue growth across multiple quarters, "
        "driven by subscription expansion, better pricing discipline, and lower churn.",

        "Management highlighted improvements in operating margin, with cost control "
        "coming from cloud optimization, process automation, and more efficient staffing.",

        "On the macro side, inflation remained elevated but showed signs of moderation, "
        "while labor market conditions stayed relatively resilient.",

        "The market reaction was initially mixed because forward guidance was cautious, "
        "even though the historical results were above consensus expectations.",

        "Analysts also focused on free cash flow conversion, capital allocation policy, "
        "and whether recent performance can be sustained over the next few quarters.",

        "Several risk factors were repeatedly mentioned, including foreign exchange exposure, "
        "slower enterprise demand, and uncertainty in customer budget cycles.",

        "The overall tone of the report was constructive but not aggressively optimistic, "
        "suggesting a gradual improvement rather than a sharp turnaround.",

        "From a valuation perspective, investors may reassess multiples if margin expansion "
        "continues and top-line growth remains stable under a soft-landing scenario.",
    ]

    # Repeat enough times so we can slice to different token lengths.
    text = "\n".join(paragraphs * 200)
    return text


def truncate_text_to_token_length(tokenizer, text: str, target_tokens: int) -> str:
    """
    Truncate text to approximately target token length.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def make_chunk(separator: str, chunk_text: str) -> str:
    """
    Build a chunk with explicit boundaries.
    """
    return f"{separator}{chunk_text}{separator}"


def build_request_templates(chunk: str) -> Dict[str, str]:
    """
    Build the request templates required by the experiment.

    Request 1: prompt1.1 + chunk + prompt1.2
    Request 2: prompt2.1 + chunk + prompt2.2
    Request 3: prompt3.1 + prompt3.2 + chunk
    Request 4: chunk + prompt3.1 + prompt3.2

    The prefixes are intentionally different to reduce accidental
    prefix-only reuse outside the shared chunk.
    """
    prompt11 = (
        "You are given a reference passage below. "
        "Summarize the business outlook in three concise bullet points.\n\n"
    )
    prompt12 = (
        "\n\nFocus on growth, cost structure, and near-term risks."
    )

    prompt21 = (
        "Please read the following passage carefully. "
        "Extract the main macroeconomic implications in plain language.\n\n"
    )
    prompt22 = (
        "\n\nKeep the answer short and mention inflation, labor, and demand."
    )

    prompt31 = (
        "Read the materials and answer the question that follows.\n\n"
        "Question: What signals might matter most for investors over the next quarter?\n\n"
    )
    prompt32 = (
        "Use a brief paragraph and avoid quoting the passage directly.\n\n"
    )

    requests_map = {
        "request0_store": chunk,
        "request1_mid_contextA": prompt11 + chunk + prompt12,
        "request2_mid_contextB": prompt21 + chunk + prompt22,
        "request3_suffix": prompt31 + prompt32 + chunk,
        "request4_prefix": chunk + prompt31 + prompt32,
    }
    return requests_map


# ============================================================
# 4. Experiment runner
# ============================================================

def run_one_request(
    *,
    mode: str,
    base_url: str,
    model: str,
    request_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    enable_metrics: bool,
) -> Dict:
    """
    Run one request and optionally record metrics delta.
    """
    metrics_delta_json = None

    metrics_before_raw = None
    metrics_after_raw = None

    if enable_metrics:
        metrics_before_raw = scrape_metrics(base_url, timeout=30)

    out = completions_request(
        base_url=base_url,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    if enable_metrics:
        metrics_after_raw = scrape_metrics(base_url, timeout=30)
        before = parse_prometheus_metrics(metrics_before_raw or "")
        after = parse_prometheus_metrics(metrics_after_raw or "")
        delta = diff_metric_dict(before, after)
        delta_filtered = pick_lmcache_related_metrics(delta)
        metrics_delta_json = json.dumps(delta_filtered, ensure_ascii=False)

    response_text = get_text_from_completion_response(out["response_json"])

    return {
        "mode": mode,
        "request_name": request_name,
        "latency_ms": out["latency_ms"],
        "response_text": response_text,
        "metrics_delta_json": metrics_delta_json,
    }


def run_experiment(args):
    ensure_dir_for_file(args.output_csv)

    tokenizer_name = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    chunk_token_lengths = [
        int(x.strip()) for x in args.chunk_token_lengths.split(",") if x.strip()
    ]
    separator = args.separator

    mode = args.mode
    base_url = args.base_url

    # Build one long reusable source text.
    source_text = make_base_chunk_source_text()

    rows = []

    for chunk_len in chunk_token_lengths:
        # Build the reusable chunk for this token length.
        chunk_text = truncate_text_to_token_length(tokenizer, source_text, chunk_len)
        chunk = make_chunk(separator, chunk_text)

        # Build request templates.
        request_map = build_request_templates(chunk)

        for repeat in range(args.repeats):
            run_id = str(uuid.uuid4())[:8]

            # ------------------------------------------------------------
            # Step 1: Store chunk KV using request0_store
            # ------------------------------------------------------------
            store_result = run_one_request(
                mode=mode,
                base_url=base_url,
                model=args.model,
                request_name="request0_store",
                prompt=request_map["request0_store"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                enable_metrics=args.enable_metrics,
            )

            rows.append({
                "run_id": run_id,
                "repeat": repeat,
                "mode": mode,
                "base_url": base_url,
                "chunk_tokens": chunk_len,
                "request_name": store_result["request_name"],
                "latency_ms": round(store_result["latency_ms"], 3),
                "response_text": store_result["response_text"],
                "metrics_delta_json": store_result["metrics_delta_json"],
            })

            time.sleep(args.sleep_between_requests)

            # ------------------------------------------------------------
            # Step 2: Run the four evaluation requests
            # ------------------------------------------------------------
            eval_request_order = [
                "request1_mid_contextA",
                "request2_mid_contextB",
                "request3_suffix",
                "request4_prefix",
            ]

            for request_name in eval_request_order:
                result = run_one_request(
                    mode=mode,
                    base_url=base_url,
                    model=args.model,
                    request_name=request_name,
                    prompt=request_map[request_name],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    enable_metrics=args.enable_metrics,
                )

                rows.append({
                    "run_id": run_id,
                    "repeat": repeat,
                    "mode": mode,
                    "base_url": base_url,
                    "chunk_tokens": chunk_len,
                    "request_name": result["request_name"],
                    "latency_ms": round(result["latency_ms"], 3),
                    "response_text": result["response_text"],
                    "metrics_delta_json": result["metrics_delta_json"],
                })

                time.sleep(args.sleep_between_requests)

    write_csv(args.output_csv, rows)
    print(f"Saved results to: {args.output_csv}")


# ============================================================
# 5. CSV output
# ============================================================

def write_csv(path: str, rows: List[Dict]):
    fieldnames = [
        "run_id",
        "repeat",
        "mode",
        "base_url",
        "chunk_tokens",
        "request_name",
        "latency_ms",
        "response_text",
        "metrics_delta_json",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run_experiment(args)