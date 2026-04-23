#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
vLLM end-to-end segment-hit experiment.

This script runs real vLLM generation requests with LMCacheConnectorV1 and
collects request-level latency plus LMCache lookup/retrieve/store costs.

Request construction:
- r1: c1 + c2 + c3
- r2: c2 + c1 + c3
- r3: c2 + c1 + c3
- r4: c4 + c1 + c3
- r5: c1 + c2 + c5

Default flow:
- request phase only: warmup + r1 + r2 + r3 + r4 + r5

Optional flow:
- add --precompute-chunks to prepend: pre_warmup + pre_chunk1 + pre_chunk2 + pre_chunk3

Notes:
- This is true vLLM E2E (not direct CB_* calls).
- Metrics come from LMCache stats monitor in the connector path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.cache_engine import LMCacheEngineBuilder


REQUEST_ORDER = ["warmup", "r1", "r2", "r3", "r4", "r5"]
PRECOMPUTE_ORDER = ["pre_warmup", "pre_chunk1", "pre_chunk2", "pre_chunk3"]


@dataclass
class PromptPack:
    requests: dict[str, list[int]]
    precompute: dict[str, list[int]]


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


def configure_multiprocessing_for_vllm() -> None:
    # Keep EngineCore in-process so LMCache stats are visible in this process.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    mp.set_start_method("spawn", force=True)


def setup_environment_variables(
    use_disk: bool,
    blend_special_str: str,
    enable_sparse: bool,
    chunk_size: int,
) -> None:
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "False"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"

    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(model: str):
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
        max_model_len=12000,
        gpu_memory_utilization=0.8,
        # Keep LMCache blending path active, disable vLLM APC for cleaner comparison.
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)


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
        help="Special separator string for blending boundaries.",
    )
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Use disk as LMCache backend instead of CPU memory.",
    )
    parser.add_argument(
        "--enable-sparse",
        action="store_true",
        help="Enable sparse blending path if supported.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size used by LMCache.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Generation max_tokens for each request.",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Sleep seconds between requests.",
    )
    parser.add_argument(
        "--precompute-chunks",
        action="store_true",
        help="Run pre_warmup + pre_chunk1/2/3 before request phase.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/gcp-vm/projects/LMCache/experiments/blend/results",
        help="Directory to save raw metrics as CSV/JSON.",
    )
    return parser.parse_args()


def build_prompt_pack(tokenizer: AutoTokenizer, blend_special_str: str) -> PromptPack:
    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    sep_ids = enc(blend_special_str)

    system_text = (
        "You are a helpful assistant. "
        "Please answer the user's question briefly."
    )
    sys_prompt = enc(system_text)

    c1 = enc(("Hello, how are you? " * 500).strip())
    c2 = enc(("Hello, what's up? " * 500).strip())
    c3 = enc(("Hi, what are you up to? " * 500).strip())
    c4 = enc(("Hello, how is it going? " * 500).strip())
    c5 = enc(("Hi, nice to meet you! " * 500).strip())

    warmup_prompt = enc(("Nice to meet you. " * 500).strip())
    tail = enc("Hello, my name is")

    def compose_chunks(chunks: list[list[int]]) -> list[int]:
        prompt = list(sys_prompt)
        for chunk in chunks:
            prompt += sep_ids
            prompt += chunk
        prompt += sep_ids
        prompt += tail
        return prompt

    requests = {
        "warmup": warmup_prompt,
        "r1": compose_chunks([c1, c2, c3]),
        "r2": compose_chunks([c2, c1, c3]),
        "r3": compose_chunks([c2, c1, c3]),
        "r4": compose_chunks([c4, c1, c3]),
        "r5": compose_chunks([c1, c2, c5]),
    }

    precompute = {
        "pre_warmup": warmup_prompt,
        "pre_chunk1": c1,
        "pre_chunk2": c2,
        "pre_chunk3": c3,
    }

    return PromptPack(requests=requests, precompute=precompute)


def get_stats_cursor(monitor: LMCStatsMonitor) -> StatsCursor:
    return StatsCursor(
        lookup_id=monitor.lookup_request_id,
        retrieve_id=monitor.retrieve_request_id,
        store_id=monitor.store_request_id,
    )


def collect_request_metrics(
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
        prompt_tokens=prompt_tokens,
        request_ms=request_ms,
        lookup_ms=sum(stat.time_to_lookup() for stat in lookup_stats) * 1000.0,
        retrieve_ms=sum(stat.time_to_retrieve() for stat in retrieve_stats) * 1000.0,
        store_ms=sum(stat.time_to_store() for stat in store_stats) * 1000.0,
        lookup_hit_tokens=sum(stat.hit_tokens for stat in lookup_stats),
        retrieve_hit_tokens=sum(stat.local_hit_tokens for stat in retrieve_stats),
        store_tokens=sum(stat.num_tokens for stat in store_stats),
        lookup_events=len(lookup_stats),
        retrieve_events=len(retrieve_stats),
        store_events=len(store_stats),
    )


def run_one_request(
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
    blend_hits = metric.lookup_hit_tokens // chunk_size if chunk_size > 0 else 0
    print(
        f"[{metric.phase}:{metric.request}] prompt_tokens={metric.prompt_tokens} "
        f"blend_hits={blend_hits} "
        f"lookup_hit_tokens={metric.lookup_hit_tokens} "
        f"retrieve_hit_tokens={metric.retrieve_hit_tokens} "
        f"store_tokens={metric.store_tokens} "
        f"lookup_ms={metric.lookup_ms:.3f} "
        f"retrieve_ms={metric.retrieve_ms:.3f} "
        f"store_ms={metric.store_ms:.3f} "
        f"request_ms={metric.request_ms:.3f} "
        f"events(l/r/s)={metric.lookup_events}/{metric.retrieve_events}/{metric.store_events}"
    )


def write_results(rows: list[RequestMetrics], results_dir: str, run_tag: str) -> tuple[Path, Path]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{run_tag}_segment-hit-normal-prefill.json"
    csv_path = out_dir / f"{run_tag}_segment-hit-normal-prefill.csv"

    payload = [asdict(row) for row in rows]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = list(RequestMetrics.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload)

    return json_path, csv_path


def sum_metrics(rows: list[RequestMetrics]) -> tuple[float, float, float, float]:
    return (
        sum(r.request_ms for r in rows),
        sum(r.lookup_ms for r in rows),
        sum(r.retrieve_ms for r in rows),
        sum(r.store_ms for r in rows),
    )


def main() -> None:
    args = parse_args()
    configure_multiprocessing_for_vllm()

    setup_environment_variables(
        use_disk=args.use_disk,
        blend_special_str=args.blend_special_str,
        enable_sparse=args.enable_sparse,
        chunk_size=args.chunk_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_pack = build_prompt_pack(
        tokenizer=tokenizer,
        blend_special_str=os.getenv("LMCACHE_BLEND_SPECIAL_STR", "# #"),
    )

    print(f"Using model: {args.model}")
    print(f"LMCACHE_CHUNK_SIZE={os.getenv('LMCACHE_CHUNK_SIZE')}")
    print(f"LMCACHE_ENABLE_BLENDING={os.getenv('LMCACHE_ENABLE_BLENDING')}")
    print(f"LMCACHE_BLEND_SPECIAL_STR={os.getenv('LMCACHE_BLEND_SPECIAL_STR')}")
    print(f"LMCACHE_BLEND_RECOMPUTE_RATIOS={os.getenv('LMCACHE_BLEND_RECOMPUTE_RATIOS')}")
    print(f"LMCACHE_USE_LAYERWISE={os.getenv('LMCACHE_USE_LAYERWISE')}")
    print(f"VLLM_ENABLE_V1_MULTIPROCESSING={os.getenv('VLLM_ENABLE_V1_MULTIPROCESSING')}")
    print(f"VLLM_WORKER_MULTIPROC_METHOD={os.getenv('VLLM_WORKER_MULTIPROC_METHOD')}")
    print()

    print("Prompt token lengths:")
    for name in REQUEST_ORDER:
        print(f"  {name:<8} -> {len(prompt_pack.requests[name])} tokens")
    if args.precompute_chunks:
        print("Precompute docs:")
        for name in PRECOMPUTE_ORDER:
            print(f"  {name:<10} -> {len(prompt_pack.precompute[name])} tokens")
    print()

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    rows: list[RequestMetrics] = []

    with build_llm_with_lmcache(model=args.model) as llm:
        engine = LMCacheEngineBuilder.get(ENGINE_NAME)
        if engine is None:
            raise RuntimeError("Failed to get LMCache engine instance")

        monitor = engine.stats_monitor

        if args.precompute_chunks:
            print("Phase 1: precompute")
            for name in PRECOMPUTE_ORDER:
                row = run_one_request(
                    llm=llm,
                    monitor=monitor,
                    phase="precompute",
                    request_name=name,
                    prompt=prompt_pack.precompute[name],
                    sampling_params=sampling_params,
                )
                rows.append(row)
                print_request_line(row, args.chunk_size)
                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)
            print()

        print("Phase 2: request (vLLM end-to-end generation)")
        for name in REQUEST_ORDER:
            row = run_one_request(
                llm=llm,
                monitor=monitor,
                phase="request",
                request_name=name,
                prompt=prompt_pack.requests[name],
                sampling_params=sampling_params,
            )
            rows.append(row)
            print_request_line(row, args.chunk_size)
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)
        print()

    precompute_rows = [r for r in rows if r.phase == "precompute"]
    request_rows = [r for r in rows if r.phase == "request"]

    pre_req, pre_lookup, pre_ret, pre_store = sum_metrics(precompute_rows)
    req_req, req_lookup, req_ret, req_store = sum_metrics(request_rows)
    all_req, all_lookup, all_ret, all_store = sum_metrics(rows)

    print("Latency summary (ms):")
    if precompute_rows:
        print(
            "  precompute_phase: "
            f"request={pre_req:.3f}, lookup={pre_lookup:.3f}, "
            f"retrieve={pre_ret:.3f}, store={pre_store:.3f}"
        )
    print(
        "  request_phase: "
        f"request={req_req:.3f}, lookup={req_lookup:.3f}, "
        f"retrieve={req_ret:.3f}, store={req_store:.3f}"
    )
    print(
        "  total: "
        f"request={all_req:.3f}, lookup={all_lookup:.3f}, "
        f"retrieve={all_ret:.3f}, store={all_store:.3f}"
    )

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, csv_path = write_results(rows=rows, results_dir=args.results_dir, run_tag=run_tag)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
