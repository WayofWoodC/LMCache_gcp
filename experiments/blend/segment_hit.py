#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Segment-hit experiment: vLLM real KV compute + CB_* store/lookup/retrieve.

What this script does:
1) Run vLLM (LMCacheConnectorV1) for r1 so KV is computed by real model forward.
2) Export r1 KV from vLLM paged buffer into a contiguous [2, L, T, D] tensor.
3) Register that tensor to blend_server_v2 via CB_REGISTER_KV_CACHE.
4) Store c1/c2/c3 via CB_STORE_PRE_COMPUTED.
5) Run CB_LOOKUP_PRE_COMPUTED_V2 (and optional CB_RETRIEVE_PRE_COMPUTED_V2)
   on r1..r5 to validate segment-hit behavior.
6) Optionally run vLLM end-to-end requests for r2..r5 and print request latency.

Request construction:
- r1: c1 + c2 + c3
- r2: c2 + c1 + c3
- r3: c2 + c1 + c3
- r4: c4 + c1 + c3
- r5: c1 + c2 + c5

Note:
- CB retrieval in this script writes into the exported contiguous CB tensor.
  It is for validating CB segment-hit mechanism and CB timing.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Iterable
import multiprocessing as mp
import os
import time

import torch
import zmq
from transformers import AutoTokenizer

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.mp_observability.config import DEFAULT_PROMETHEUS_CONFIG
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper, IPCCacheEngineKey, KVCache
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class


RequestName = str
SegmentRange = tuple[str, int, int]
_CAPTURED_CONNECTORS: list[object] = []


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
    name: str
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


@dataclass
class CBLookupMetrics:
    name: str
    hits: int
    hit_tokens: int
    lookup_ms: float
    retrieve_ms: float


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def configure_multiprocessing_for_vllm() -> None:
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
    # Disable built-in blending path; this script validates explicit CB_* path.
    os.environ["LMCACHE_ENABLE_BLENDING"] = "False"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "False"

    # Avoid LMCache auto-save interference for this experiment.
    os.environ.setdefault("LMCACHE_FORCE_SKIP_SAVE", "True")

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


def install_connector_probe() -> None:
    """Patch LMCacheConnectorV1Impl.__init__ to capture live connector instances."""
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    if getattr(LMCacheConnectorV1Impl, "_segment_hit_probe_installed", False):
        return

    original_init = LMCacheConnectorV1Impl.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _CAPTURED_CONNECTORS.append(self)

    LMCacheConnectorV1Impl.__init__ = wrapped_init  # type: ignore[method-assign]
    LMCacheConnectorV1Impl._segment_hit_probe_installed = True  # type: ignore[attr-defined]


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


def get_worker_connector_impl() -> object:
    for impl in reversed(_CAPTURED_CONNECTORS):
        role = str(getattr(impl, "_role", "")).lower()
        if "worker" in role:
            return impl

    if _CAPTURED_CONNECTORS:
        return _CAPTURED_CONNECTORS[-1]

    raise RuntimeError("Could not capture LMCacheConnectorV1Impl instance")


def server_process_runner(
    host: str,
    port: int,
    chunk_size: int,
    cpu_buffer_size_gb: float,
) -> None:
    from lmcache.v1.multiprocess.blend_server_v2 import run_cache_server

    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=int(cpu_buffer_size_gb * 1024**3),
                use_lazy=True,
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )

    run_cache_server(
        storage_manager_config=storage_manager_config,
        prometheus_config=DEFAULT_PROMETHEUS_CONFIG,
        host=host,
        port=port,
        chunk_size=chunk_size,
    )


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

    prompts: dict[RequestName, list[int]] = {}
    segment_ranges: dict[RequestName, list[SegmentRange]] = {}
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
    monitor: LMCStatsMonitor,
    before: StatsCursor,
    after: StatsCursor,
    name: str,
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
        name=name,
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
        name=request_name,
        prompt_tokens=len(prompt),
        request_ms=ed - st,
    )


def print_request_line(metric: RequestMetrics, chunk_size: int) -> None:
    blend_hits = metric.lookup_hit_tokens // chunk_size if chunk_size > 0 else 0
    print(
        f"[vllm:{metric.name}] prompt_tokens={metric.prompt_tokens} "
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


def create_cb_cache_key(
    token_ids: tuple[int, ...],
    request_id: str,
    model_name: str,
    worker_id: int | None = 0,
) -> IPCCacheEngineKey:
    return IPCCacheEngineKey(
        model_name=model_name,
        world_size=1,
        worker_id=worker_id,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id=request_id,
    )


def get_segment_bounds(ranges: list[SegmentRange], segment_name: str) -> tuple[int, int]:
    for name, st, ed in ranges:
        if name == segment_name:
            return st, ed
    raise ValueError(f"Segment {segment_name} not found")


def locate_segment(ranges: list[SegmentRange], st: int, ed: int) -> str:
    for name, seg_st, seg_ed in ranges:
        if st < seg_ed and ed > seg_st:
            return name
    return "unknown"


def summarize_segments(matches: Iterable, ranges: list[SegmentRange]) -> str:
    counts: dict[str, int] = {}
    for m in matches:
        seg = locate_segment(ranges, m.cur_st, m.cur_ed)
        counts[seg] = counts.get(seg, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def build_slot_mapping_from_tracker_or_linear(
    worker_impl: object,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    block_size = int(getattr(worker_impl, "_block_size"))
    trackers = getattr(worker_impl, "_request_trackers", {})

    for tracker in trackers.values():
        token_ids = getattr(tracker, "token_ids", [])
        block_ids = getattr(tracker, "allocated_block_ids", [])
        if len(token_ids) < num_tokens:
            continue
        if len(block_ids) * block_size < num_tokens:
            continue

        block_tensor = torch.tensor(block_ids, dtype=torch.long, device=device)
        offsets = torch.arange(block_size, dtype=torch.long, device=device).unsqueeze(0)
        slot_mapping = (offsets + block_tensor.unsqueeze(1) * block_size).flatten()
        return slot_mapping[:num_tokens].contiguous()

    # Fallback assumption: first request usually maps linearly from 0.
    return torch.arange(num_tokens, dtype=torch.long, device=device)


def export_request_kv_to_plain(worker_impl: object, num_tokens: int) -> torch.Tensor:
    kv_cache_dict = getattr(worker_impl, "kv_caches", None)
    if not kv_cache_dict:
        raise RuntimeError("worker connector kv_caches is empty")

    lmcache_engine = getattr(worker_impl, "lmcache_engine", None)
    if lmcache_engine is None:
        raise RuntimeError("worker connector has no lmcache_engine")

    gpu_connector = lmcache_engine.gpu_connector
    if gpu_connector is None:
        raise RuntimeError("LMCache engine has no gpu_connector")

    kv_caches = list(kv_cache_dict.values())
    dev = kv_caches[0].device

    slot_mapping = build_slot_mapping_from_tracker_or_linear(worker_impl, num_tokens, dev)

    shape = gpu_connector.get_shape(num_tokens)
    if len(shape) != 4 or shape[0] != 2:
        raise RuntimeError(
            f"Expected non-MLA KV shape [2,L,T,D], got {tuple(shape)}"
        )

    plain_kv = torch.empty(shape, dtype=kv_caches[0].dtype, device=dev)

    mem_obj = SimpleNamespace(
        tensor=plain_kv,
        metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
    )

    gpu_connector.from_gpu(
        mem_obj,
        0,
        num_tokens,
        kvcaches=kv_caches,
        slot_mapping=slot_mapping,
    )
    torch.cuda.synchronize(dev)
    return plain_kv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--blend-special-str", default="# #")

    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1)

    parser.add_argument("-d", "--use-disk", action="store_true")
    parser.add_argument("--enable-sparse", action="store_true")

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5567)
    parser.add_argument("--cpu-buffer-size-gb", type=float, default=5.0)
    parser.add_argument("--start-server", action="store_true")

    parser.add_argument("--retrieve-after-lookup", action="store_true")
    parser.add_argument("--retrieve-offset", type=int, default=0)
    parser.add_argument("--max-match-rows", type=int, default=12)

    parser.add_argument(
        "--run-vllm-r2-r5",
        action="store_true",
        help="Also run vLLM real generation for r2..r5 and print request latency.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    configure_multiprocessing_for_vllm()
    setup_environment_variables(
        use_disk=args.use_disk,
        blend_special_str=args.blend_special_str,
        enable_sparse=args.enable_sparse,
        chunk_size=args.chunk_size,
    )
    install_connector_probe()

    server_proc: mp.Process | None = None
    if args.start_server:
        server_proc = mp.Process(
            target=server_process_runner,
            args=(args.host, args.port, args.chunk_size, args.cpu_buffer_size_gb),
            daemon=True,
        )
        server_proc.start()
        time.sleep(3)

    server_url = f"tcp://{args.host}:{args.port}"
    context = zmq.Context.instance()
    client = MessageQueueClient(server_url=server_url, context=context)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_pack = build_prompt_pack(tokenizer, args.blend_special_str)

    print(f"server_url={server_url}")
    print(f"model={args.model}")
    print(f"chunk_size={args.chunk_size}")
    print(f"LMCACHE_ENABLE_BLENDING={os.getenv('LMCACHE_ENABLE_BLENDING')}")
    print(f"LMCACHE_FORCE_SKIP_SAVE={os.getenv('LMCACHE_FORCE_SKIP_SAVE')}")
    for req_name in ["r1", "r2", "r3", "r4", "r5"]:
        print(f"{req_name}: prompt_tokens={len(prompt_pack.prompts[req_name])}")
    print()

    instance_id = os.getpid() + 10000
    model_name_for_cb = "segment-hit-vllm-cb"

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    vllm_rows: list[RequestMetrics] = []
    cb_rows: list[CBLookupMetrics] = []

    plain_kv: torch.Tensor | None = None

    try:
        client.submit_request(
            RequestType.CLEAR,
            [],
            get_response_class(RequestType.CLEAR),
        ).result(timeout=20)

        with build_llm_with_lmcache(args.model) as llm:
            engine = LMCacheEngineBuilder.get(ENGINE_NAME)
            if engine is None:
                raise RuntimeError("Failed to get LMCache engine instance")
            monitor = engine.stats_monitor

            # --------------------------------------------------------------
            # Step 1: vLLM real compute on r1
            # --------------------------------------------------------------
            r1_metric = run_one_request(
                llm=llm,
                monitor=monitor,
                request_name="r1",
                prompt=prompt_pack.prompts["r1"],
                sampling_params=sampling_params,
            )
            vllm_rows.append(r1_metric)
            print_request_line(r1_metric, args.chunk_size)

            worker_impl = get_worker_connector_impl()
            plain_kv = export_request_kv_to_plain(
                worker_impl=worker_impl,
                num_tokens=len(prompt_pack.prompts["r1"]),
            )
            print(f"[export] plain_kv_shape={tuple(plain_kv.shape)} dtype={plain_kv.dtype}")

            kv_cache: KVCache = [CudaIPCWrapper(plain_kv)]
            client.submit_request(
                RequestType.CB_REGISTER_KV_CACHE,
                [instance_id, kv_cache, model_name_for_cb, 1],
                get_response_class(RequestType.CB_REGISTER_KV_CACHE),
            ).result(timeout=20)

            # --------------------------------------------------------------
            # Step 2: CB store c1/c2/c3 from r1 offsets
            # --------------------------------------------------------------
            r1_tokens = prompt_pack.prompts["r1"]
            r1_ranges = prompt_pack.segment_ranges["r1"]
            for seg_name in ["c1", "c2", "c3"]:
                st, ed = get_segment_bounds(r1_ranges, seg_name)
                seg_tokens = tuple(r1_tokens[st:ed])
                key = create_cb_cache_key(
                    token_ids=seg_tokens,
                    request_id=f"store-r1-{seg_name}",
                    model_name=model_name_for_cb,
                )

                event = torch.cuda.Event(interprocess=True)
                event.record()

                t0 = now_ms()
                ok = (
                    client.submit_request(
                        RequestType.CB_STORE_PRE_COMPUTED,
                        [key, st, instance_id, event.ipc_handle()],
                        get_response_class(RequestType.CB_STORE_PRE_COMPUTED),
                    )
                    .to_cuda_future()
                    .result(timeout=40)
                )
                t1 = now_ms()
                print(
                    f"[store][{seg_name}] token_len={len(seg_tokens)} "
                    f"offset={st} ok={ok} latency_ms={t1 - t0:.3f}"
                )
            print()

            # --------------------------------------------------------------
            # Step 3: CB lookup/retrieve for r1..r5
            # --------------------------------------------------------------
            total_lookup_ms = 0.0
            total_retrieve_ms = 0.0

            for req_name in ["r1", "r2", "r3", "r4", "r5"]:
                req_tokens = tuple(prompt_pack.prompts[req_name])
                key = create_cb_cache_key(
                    token_ids=req_tokens,
                    request_id=f"lookup-{req_name}",
                    model_name=model_name_for_cb,
                )

                t0 = now_ms()
                matches = client.submit_request(
                    RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
                    [key],
                    get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
                ).result(timeout=30)
                t1 = now_ms()

                lookup_ms = t1 - t0
                total_lookup_ms += lookup_ms

                matches_sorted = sorted(matches, key=lambda m: m.cur_st)
                hit_tokens = len(matches_sorted) * args.chunk_size
                seg_summary = summarize_segments(matches_sorted, prompt_pack.segment_ranges[req_name])

                print(
                    f"[lookup][{req_name}] hits={len(matches_sorted)} "
                    f"hit_tokens={hit_tokens} latency_ms={lookup_ms:.3f} "
                    f"segments={seg_summary}"
                )
                for m in matches_sorted[: args.max_match_rows]:
                    seg = locate_segment(prompt_pack.segment_ranges[req_name], m.cur_st, m.cur_ed)
                    print(
                        f"  cur=[{m.cur_st},{m.cur_ed}) seg={seg:<6} "
                        f"old=[{m.old_st},{m.old_ed}) hash={m.hash.hex()[:16]}..."
                    )
                if len(matches_sorted) > args.max_match_rows:
                    print(f"  ... and {len(matches_sorted) - args.max_match_rows} more")

                retrieve_ms = 0.0
                if args.retrieve_after_lookup:
                    event2 = torch.cuda.Event(interprocess=True)
                    event2.record()

                    t2 = now_ms()
                    ok2 = (
                        client.submit_request(
                            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
                            [key, matches_sorted, args.retrieve_offset, instance_id, event2.ipc_handle()],
                            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
                        )
                        .to_cuda_future()
                        .result(timeout=40)
                    )
                    t3 = now_ms()

                    retrieve_ms = t3 - t2
                    total_retrieve_ms += retrieve_ms
                    print(
                        f"[retrieve][{req_name}] ok={ok2} offset={args.retrieve_offset} "
                        f"latency_ms={retrieve_ms:.3f}"
                    )
                print()

                cb_rows.append(
                    CBLookupMetrics(
                        name=req_name,
                        hits=len(matches_sorted),
                        hit_tokens=hit_tokens,
                        lookup_ms=lookup_ms,
                        retrieve_ms=retrieve_ms,
                    )
                )

            print("CB totals:")
            print(f"  lookup_total_ms={total_lookup_ms:.3f}")
            if args.retrieve_after_lookup:
                print(f"  retrieve_total_ms={total_retrieve_ms:.3f}")
            print()

            # --------------------------------------------------------------
            # Step 4: Optional vLLM E2E latency for r2..r5
            # --------------------------------------------------------------
            if args.run_vllm_r2_r5:
                print("vLLM request latency (r2..r5):")
                for req_name in ["r2", "r3", "r4", "r5"]:
                    row = run_one_request(
                        llm=llm,
                        monitor=monitor,
                        request_name=req_name,
                        prompt=prompt_pack.prompts[req_name],
                        sampling_params=sampling_params,
                    )
                    vllm_rows.append(row)
                    print_request_line(row, args.chunk_size)
                print()

        if vllm_rows:
            print("vLLM totals:")
            print(f"  request_total_ms={sum(r.request_ms for r in vllm_rows):.3f}")
            print(f"  lookup_total_ms={sum(r.lookup_ms for r in vllm_rows):.3f}")
            print(f"  retrieve_total_ms={sum(r.retrieve_ms for r in vllm_rows):.3f}")
            print(f"  store_total_ms={sum(r.store_ms for r in vllm_rows):.3f}")

    finally:
        try:
            client.submit_request(
                RequestType.CB_UNREGISTER_KV_CACHE,
                [instance_id],
                get_response_class(RequestType.CB_UNREGISTER_KV_CACHE),
            ).result(timeout=15)
        except Exception:
            pass

        try:
            client.submit_request(
                RequestType.CLEAR,
                [],
                get_response_class(RequestType.CLEAR),
            ).result(timeout=15)
        except Exception:
            pass

        client.close()

        if plain_kv is not None:
            del plain_kv
            torch.cuda.empty_cache()

        if server_proc is not None and server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.kill()
                server_proc.join(timeout=2)


if __name__ == "__main__":
    main()
