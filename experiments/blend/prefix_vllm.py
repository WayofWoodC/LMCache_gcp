#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
vLLM end-to-end common-prefix-hit experiment (LMCache non-blend mode).

Goal:
- Reuse the same request/prompt construction as segment_hit_vllm.py:
  - r1: c1 + c2 + c3
  - r2: c2 + c1 + c3
  - r3: c2 + c1 + c3
  - r4: c4 + c1 + c3
  - r5: c1 + c2 + c5
- Run LMCache in non-blending mode (common-prefix behavior).
- Print per-request LMCache metrics (lookup/retrieve/store latency and hit tokens).

Notes:
- This script keeps vLLM prefix-caching disabled by default to focus on LMCache
  common-prefix behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.cache_engine import LMCacheEngineBuilder

REQUEST_ORDER = ["warmup", "r1", "r2", "r3", "r4", "r5"]
RequestName = str
SegmentRange = tuple[str, int, int]


@dataclass
class PromptPack:
    prompts: dict[RequestName, list[int]]
    segment_ranges: dict[RequestName, list[SegmentRange]]


@dataclass
class StatsCursor:
    lookup_id: int
    retrieve_id: int
    store_id: int


@dataclass
class RequestMetrics:
    phase: str
    request: str
    prompt_tokens: int
    request_ms: float
    lookup_ms: float
    retrieve_ms: float
    store_ms: float
    lookup_hit_tokens: int
    retrieve_hit_tokens: int
    store_tokens: int
    lookup_events: int
    retrieve_events: int
    store_events: int


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def to_python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return to_python_value(value.item())
        return [to_python_value(v) for v in value.detach().cpu().flatten().tolist()]
    if isinstance(value, dict):
        return {str(k): to_python_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python_value(v) for v in value]
    if isinstance(value, tuple):
        return [to_python_value(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return to_python_value(item())
    return str(value)


def to_int(value: Any) -> int:
    return int(to_python_value(value))


def to_float(value: Any) -> float:
    return float(to_python_value(value))


def slugify_component(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if not s:
        s = "na"
    return s[:max_len]


def build_run_label(args: argparse.Namespace, run_tag: str) -> str:
    model_slug = slugify_component(args.model.split("/")[-1])
    return (
        f"{run_tag}_prefixvllm_commonprefix_{model_slug}"
        f"_c{args.chunk_size}_s{args.segment_chunks}_w{args.warmup_chunks}"
        f"_ly{int(bool(args.use_layerwise))}_apc{int(bool(args.enable_vllm_prefix_caching))}"
    )


def configure_multiprocessing_for_vllm() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    mp.set_start_method("spawn", force=True)


def setup_environment_variables(
    *,
    chunk_size: int,
    blend_special_str: str,
    use_disk: bool,
    use_layerwise: bool,
) -> None:
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "False"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True" if use_layerwise else "False"

    # Must allow store so later requests can hit cached common prefix chunks.
    os.environ["LMCACHE_FORCE_SKIP_SAVE"] = "0"

    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(
    *,
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    enable_vllm_prefix_caching: bool,
):
    from vllm import LLM
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_vllm_prefix_caching,
        enforce_eager=True,
        disable_log_stats=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)


def build_prompt_pack(
    tokenizer: AutoTokenizer,
    blend_special_str: str,
    *,
    chunk_size: int,
    segment_chunks: int,
    warmup_chunks: int,
) -> PromptPack:
    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    sep_ids = enc(blend_special_str)
    system_text = (
        "You are a helpful assistant. "
        "Please answer the user's question briefly."
    )
    sys_prompt = enc(system_text)

    per_segment_tokens = max(chunk_size, chunk_size * max(1, segment_chunks))
    warmup_tokens = max(chunk_size, chunk_size * max(1, warmup_chunks))

    def make_chunk_aligned_segment(seed_text: str, target_tokens: int) -> list[int]:
        seed = enc(seed_text)
        if not seed:
            fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
            seed = [int(fallback_id)]
        out: list[int] = []
        while len(out) < target_tokens:
            out.extend(seed)
        return out[:target_tokens]

    c1 = make_chunk_aligned_segment("Hello, how are you?", per_segment_tokens)
    c2 = make_chunk_aligned_segment("Hello, what's up?", per_segment_tokens)
    c3 = make_chunk_aligned_segment("Hi, what are you up to?", per_segment_tokens)
    c4 = make_chunk_aligned_segment("Hello, how is it going?", per_segment_tokens)
    c5 = make_chunk_aligned_segment("Hi, nice to meet you!", per_segment_tokens)

    warmup_prompt = make_chunk_aligned_segment("Nice to meet you.", warmup_tokens)
    tail = enc("Hello, my name is")

    def join_segments(segments: list[tuple[str, list[int]]]) -> tuple[list[int], list[SegmentRange]]:
        prompt: list[int] = list(sys_prompt)
        ranges: list[SegmentRange] = [("sys", 0, len(sys_prompt))]
        for name, token_ids in segments:
            prompt.extend(sep_ids)
            st = len(prompt)
            prompt.extend(token_ids)
            ed = len(prompt)
            ranges.append((name, st, ed))
        prompt.extend(sep_ids)
        st = len(prompt)
        prompt.extend(tail)
        ranges.append(("tail", st, len(prompt)))
        return prompt, ranges

    request_segments: dict[RequestName, list[tuple[str, list[int]]]] = {
        "r1": [("c1", c1), ("c2", c2), ("c3", c3)],
        "r2": [("c2", c2), ("c1", c1), ("c3", c3)],
        "r3": [("c2", c2), ("c1", c1), ("c3", c3)],
        "r4": [("c4", c4), ("c1", c1), ("c3", c3)],
        "r5": [("c1", c1), ("c2", c2), ("c5", c5)],
    }

    prompts: dict[RequestName, list[int]] = {"warmup": warmup_prompt}
    segment_ranges: dict[RequestName, list[SegmentRange]] = {
        "warmup": [("warmup", 0, len(warmup_prompt))]
    }

    for req_name, segments in request_segments.items():
        prompt, ranges = join_segments(segments)
        prompts[req_name] = prompt
        segment_ranges[req_name] = ranges

    return PromptPack(prompts=prompts, segment_ranges=segment_ranges)


def get_stats_cursor(monitor: LMCStatsMonitor) -> StatsCursor:
    return StatsCursor(
        lookup_id=monitor.lookup_request_id,
        retrieve_id=monitor.retrieve_request_id,
        store_id=monitor.store_request_id,
    )


def collect_request_metrics(
    *,
    monitor: LMCStatsMonitor,
    before: StatsCursor,
    after: StatsCursor,
    phase: str,
    request_name: str,
    prompt_tokens: int,
    request_ms: float,
) -> RequestMetrics:
    lookup_stats = [
        monitor.lookup_requests[idx]
        for idx in range(before.lookup_id, after.lookup_id)
        if idx in monitor.lookup_requests
    ]
    retrieve_stats = [
        monitor.retrieve_requests[idx]
        for idx in range(before.retrieve_id, after.retrieve_id)
        if idx in monitor.retrieve_requests
    ]
    store_stats = [
        monitor.store_requests[idx]
        for idx in range(before.store_id, after.store_id)
        if idx in monitor.store_requests
    ]

    return RequestMetrics(
        phase=phase,
        request=request_name,
        prompt_tokens=to_int(prompt_tokens),
        request_ms=to_float(request_ms),
        lookup_ms=to_float(sum(stat.time_to_lookup() for stat in lookup_stats) * 1000.0),
        retrieve_ms=to_float(sum(stat.time_to_retrieve() for stat in retrieve_stats) * 1000.0),
        store_ms=to_float(sum(stat.time_to_store() for stat in store_stats) * 1000.0),
        lookup_hit_tokens=to_int(sum(stat.hit_tokens for stat in lookup_stats)),
        retrieve_hit_tokens=to_int(sum(stat.local_hit_tokens for stat in retrieve_stats)),
        store_tokens=to_int(sum(stat.num_tokens for stat in store_stats)),
        lookup_events=to_int(len(lookup_stats)),
        retrieve_events=to_int(len(retrieve_stats)),
        store_events=to_int(len(store_stats)),
    )


def run_one_request(
    *,
    llm,
    monitor: LMCStatsMonitor,
    phase: str,
    request_name: str,
    prompt: list[int],
    sampling_params,
) -> RequestMetrics:
    before = get_stats_cursor(monitor)
    st = now_ms()
    llm.generate(
        prompts={"prompt_token_ids": prompt},
        sampling_params=sampling_params,
    )
    ed = now_ms()
    after = get_stats_cursor(monitor)

    return collect_request_metrics(
        monitor=monitor,
        before=before,
        after=after,
        phase=phase,
        request_name=request_name,
        prompt_tokens=len(prompt),
        request_ms=ed - st,
    )


def print_request_line(metric: RequestMetrics, chunk_size: int) -> None:
    prefix_hit_chunks = metric.lookup_hit_tokens // chunk_size if chunk_size > 0 else 0
    print(
        f"[{metric.phase}:{metric.request}] prompt_tokens={metric.prompt_tokens} "
        f"prefix_hit_chunks={prefix_hit_chunks} "
        f"lookup_hit_tokens={metric.lookup_hit_tokens} "
        f"retrieve_hit_tokens={metric.retrieve_hit_tokens} "
        f"store_tokens={metric.store_tokens} "
        f"lookup_ms={metric.lookup_ms:.3f} "
        f"retrieve_ms={metric.retrieve_ms:.3f} "
        f"store_ms={metric.store_ms:.3f} "
        f"request_ms={metric.request_ms:.3f} "
        f"events(l/r/s)={metric.lookup_events}/{metric.retrieve_events}/{metric.store_events}"
    )


def write_results(
    *,
    request_rows: list[RequestMetrics],
    results_dir: str,
    run_label: str,
    args: argparse.Namespace,
) -> dict[str, Path]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}

    req_json = out_dir / f"{run_label}_requests.json"
    req_csv = out_dir / f"{run_label}_requests.csv"
    req_payload = [to_python_value(asdict(row)) for row in request_rows]

    req_json.write_text(json.dumps(req_payload, indent=2), encoding="utf-8")
    with req_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RequestMetrics.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(req_payload)

    output_paths["request_json"] = req_json
    output_paths["request_csv"] = req_csv

    totals = {
        "request_ms": to_float(sum(r["request_ms"] for r in req_payload)),
        "lookup_ms": to_float(sum(r["lookup_ms"] for r in req_payload)),
        "retrieve_ms": to_float(sum(r["retrieve_ms"] for r in req_payload)),
        "store_ms": to_float(sum(r["store_ms"] for r in req_payload)),
        "lookup_hit_tokens": to_int(sum(r["lookup_hit_tokens"] for r in req_payload)),
        "retrieve_hit_tokens": to_int(sum(r["retrieve_hit_tokens"] for r in req_payload)),
        "store_tokens": to_int(sum(r["store_tokens"] for r in req_payload)),
        "lookup_events": to_int(sum(r["lookup_events"] for r in req_payload)),
        "retrieve_events": to_int(sum(r["retrieve_events"] for r in req_payload)),
        "store_events": to_int(sum(r["store_events"] for r in req_payload)),
    }

    summary_payload = {
        "run_label": run_label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "request_count": len(req_payload),
        "totals": totals,
        "mean_ms": {
            "request_ms": to_float(totals["request_ms"] / max(1, len(req_payload))),
            "lookup_ms": to_float(totals["lookup_ms"] / max(1, len(req_payload))),
            "retrieve_ms": to_float(totals["retrieve_ms"] / max(1, len(req_payload))),
            "store_ms": to_float(totals["store_ms"] / max(1, len(req_payload))),
        },
    }

    summary_json = out_dir / f"{run_label}_summary.json"
    summary_json.write_text(json.dumps(to_python_value(summary_payload), indent=2), encoding="utf-8")
    output_paths["summary_json"] = summary_json

    summary_csv = out_dir / f"{run_label}_summary.csv"
    summary_rows = [
        {"section": "request_totals_ms", "metric": "request", "value": totals["request_ms"]},
        {"section": "request_totals_ms", "metric": "lookup", "value": totals["lookup_ms"]},
        {"section": "request_totals_ms", "metric": "retrieve", "value": totals["retrieve_ms"]},
        {"section": "request_totals_ms", "metric": "store", "value": totals["store_ms"]},
        {
            "section": "request_mean_ms",
            "metric": "request",
            "value": summary_payload["mean_ms"]["request_ms"],
        },
        {
            "section": "request_mean_ms",
            "metric": "lookup",
            "value": summary_payload["mean_ms"]["lookup_ms"],
        },
        {
            "section": "request_mean_ms",
            "metric": "retrieve",
            "value": summary_payload["mean_ms"]["retrieve_ms"],
        },
        {
            "section": "request_mean_ms",
            "metric": "store",
            "value": summary_payload["mean_ms"]["store_ms"],
        },
        {
            "section": "request_tokens",
            "metric": "lookup_hit_tokens",
            "value": totals["lookup_hit_tokens"],
        },
        {
            "section": "request_tokens",
            "metric": "retrieve_hit_tokens",
            "value": totals["retrieve_hit_tokens"],
        },
        {
            "section": "request_tokens",
            "metric": "store_tokens",
            "value": totals["store_tokens"],
        },
        {
            "section": "request_events",
            "metric": "lookup_events",
            "value": totals["lookup_events"],
        },
        {
            "section": "request_events",
            "metric": "retrieve_events",
            "value": totals["retrieve_events"],
        },
        {
            "section": "request_events",
            "metric": "store_events",
            "value": totals["store_events"],
        },
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)
    output_paths["summary_csv"] = summary_csv

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Same segment separator string as segment_hit_vllm.py prompt builder.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--segment-chunks", type=int, default=2)
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--use-disk", action="store_true")
    parser.add_argument(
        "--use-layerwise",
        action="store_true",
        default=True,
        help="Use layerwise GPU connector path (recommended for compatibility/stability).",
    )
    parser.add_argument(
        "--no-layerwise",
        action="store_false",
        dest="use_layerwise",
        help="Disable layerwise path and use non-layerwise KV transfer.",
    )
    parser.add_argument(
        "--enable-vllm-prefix-caching",
        action="store_true",
        help="Enable vLLM built-in APC in addition to LMCache connector path.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/gcp-vm/projects/LMCache/experiments/blend/results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    configure_multiprocessing_for_vllm()
    setup_environment_variables(
        chunk_size=args.chunk_size,
        blend_special_str=args.blend_special_str,
        use_disk=args.use_disk,
        use_layerwise=args.use_layerwise,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_pack = build_prompt_pack(
        tokenizer=tokenizer,
        blend_special_str=args.blend_special_str,
        chunk_size=args.chunk_size,
        segment_chunks=args.segment_chunks,
        warmup_chunks=args.warmup_chunks,
    )

    max_prompt_tokens = max(len(v) for v in prompt_pack.prompts.values())
    if max_prompt_tokens + args.max_tokens > args.max_model_len:
        raise RuntimeError(
            "Prompt length exceeds max_model_len budget: "
            f"max_prompt_tokens={max_prompt_tokens}, max_tokens={args.max_tokens}, "
            f"max_model_len={args.max_model_len}."
        )

    print(f"Using model: {args.model}")
    print("mode=common-prefix (LMCache non-blend)")
    print(f"LMCACHE_CHUNK_SIZE={os.getenv('LMCACHE_CHUNK_SIZE')}")
    print(f"LMCACHE_ENABLE_BLENDING={os.getenv('LMCACHE_ENABLE_BLENDING')}")
    print(f"LMCACHE_USE_LAYERWISE={os.getenv('LMCACHE_USE_LAYERWISE')}")
    print(f"LMCACHE_FORCE_SKIP_SAVE={os.getenv('LMCACHE_FORCE_SKIP_SAVE')}")
    print(f"enable_vllm_prefix_caching={args.enable_vllm_prefix_caching}")
    print()

    print("Prompt token lengths:")
    for name in REQUEST_ORDER:
        print(f"  {name:<8} -> {len(prompt_pack.prompts[name])} tokens")
    print()

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    rows: list[RequestMetrics] = []

    with build_llm_with_lmcache(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_vllm_prefix_caching=args.enable_vllm_prefix_caching,
    ) as llm:
        engine = LMCacheEngineBuilder.get(ENGINE_NAME)
        if engine is None:
            raise RuntimeError("Failed to get LMCache engine instance")
        monitor = engine.stats_monitor

        request_sequence = REQUEST_ORDER if not args.skip_warmup else REQUEST_ORDER[1:]

        print("Phase 1: vLLM end-to-end requests (common-prefix mode)")
        for name in request_sequence:
            row = run_one_request(
                llm=llm,
                monitor=monitor,
                phase="request",
                request_name=name,
                prompt=prompt_pack.prompts[name],
                sampling_params=sampling_params,
            )
            rows.append(row)
            print_request_line(row, args.chunk_size)

            if args.sleep_between > 0:
                time.sleep(args.sleep_between)
        print()

    req_req = sum(r.request_ms for r in rows)
    req_lookup = sum(r.lookup_ms for r in rows)
    req_ret = sum(r.retrieve_ms for r in rows)
    req_store = sum(r.store_ms for r in rows)
    req_lookup_hits = sum(r.lookup_hit_tokens for r in rows)
    req_ret_hits = sum(r.retrieve_hit_tokens for r in rows)

    print("Latency summary (ms):")
    print(
        "  request_phase: "
        f"request={req_req:.3f}, lookup={req_lookup:.3f}, "
        f"retrieve={req_ret:.3f}, store={req_store:.3f}"
    )
    print("Hit summary:")
    print(f"  lookup_hit_tokens={req_lookup_hits}")
    print(f"  retrieve_hit_tokens={req_ret_hits}")

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_label = build_run_label(args, run_tag)
    output_paths = write_results(
        request_rows=rows,
        results_dir=args.results_dir,
        run_label=run_label,
        args=args,
    )
    for k, path in output_paths.items():
        print(f"Saved {k}: {path}")


if __name__ == "__main__":
    main()
