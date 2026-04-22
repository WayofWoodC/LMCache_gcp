#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
vLLM end-to-end segment-hit experiment with optional blend-server verification.

Goal:
- Compute KV via real vLLM forward (no offline KV precompute).
- Enable LMCache blend mode for segment reuse.
- Test the r1-r5 construction:
  - r1: c1 + c2 + c3
  - r2: c2 + c1 + c3
  - r3: c2 + c1 + c3
  - r4: c4 + c1 + c3
  - r5: c1 + c2 + c5

Modes:
1) vLLM E2E blend mode (default)
   - warmup -> r1 -> r2 -> r3 -> r4 -> r5
   - prints LMCache lookup/retrieve/store timings and hit tokens.

2) Optional blend-server CB verification (--cb-verify)
   - after r1, export r1 KV from vLLM paged buffer
   - register KV buffer to blend server
   - store c1/c2/c3 from r1 offsets with CB_STORE_PRE_COMPUTED
   - run CB_LOOKUP_PRE_COMPUTED_V2 (and optional retrieve) on r1..r5

Examples:
- Pure vLLM E2E blend mode:
  python segment_hit_vllm.py --model meta-llama/Llama-3.2-3B-Instruct

- Start local blend server and run CB verification too:
  python segment_hit_vllm.py --start-server --cb-verify
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Literal

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


REQUEST_ORDER = ["warmup", "r1", "r2", "r3", "r4", "r5"]
CB_LOOKUP_ORDER = ["r1", "r2", "r3", "r4", "r5"]
_CAPTURED_CONNECTORS: list[object] = []
CBProtocol = Literal["v1", "v2"]
ServerBackend = Literal["blend_v1", "blend_v2"]

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


@dataclass
class CBLookupMetrics:
    request: str
    hits: int
    hit_tokens: int
    lookup_ms: float
    retrieve_ms: float
    segment_summary: str


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
    use_layerwise: bool,
    blend_check_layers: str,
    blend_recompute_ratios: str,
) -> None:
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True" if use_layerwise else "False"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = blend_check_layers
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = blend_recompute_ratios

    # For this experiment, we want stores enabled so later requests can hit.
    os.environ["LMCACHE_FORCE_SKIP_SAVE"] = "False"

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
        # Keep LMCache blending active, disable APC for a cleaner experiment.
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


def choose_server_backend(server_backend: str) -> tuple[ServerBackend, str | None]:
    if server_backend == "v1":
        return "blend_v1", None
    if server_backend == "v2":
        from lmcache.v1.multiprocess import blend_server_v2  # noqa: F401

        return "blend_v2", None
    if server_backend != "auto":
        raise ValueError(f"Unknown server backend: {server_backend}")

    try:
        from lmcache.v1.multiprocess import blend_server_v2  # noqa: F401

        return "blend_v2", None
    except Exception as e:
        return "blend_v1", f"{type(e).__name__}: {e}"


def server_process_runner(
    host: str,
    port: int,
    chunk_size: int,
    cpu_buffer_size_gb: float,
    backend: ServerBackend,
) -> None:
    storage_manager_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=int(cpu_buffer_size_gb * 1024**3),
                use_lazy=True,
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )

    if backend == "blend_v1":
        from lmcache.v1.multiprocess import blend_server
        from lmcache.v1.multiprocess.config import MPServerConfig

        run_cache_server = blend_server.run_cache_server
        param_names = set(inspect.signature(run_cache_server).parameters)

        # Legacy API.
        if {
            "storage_manager_config",
            "prometheus_config",
            "host",
            "port",
            "chunk_size",
        }.issubset(param_names):
            run_cache_server(
                storage_manager_config=storage_manager_config,
                prometheus_config=DEFAULT_PROMETHEUS_CONFIG,
                host=host,
                port=port,
                chunk_size=chunk_size,
            )
            return

        # Newer API with MPServerConfig.
        if {"mp_config", "storage_manager_config", "prometheus_config"}.issubset(
            param_names
        ):
            mp_cfg_fields = set(inspect.signature(MPServerConfig).parameters)
            mp_kwargs: dict[str, object] = {
                "host": host,
                "port": port,
                "chunk_size": chunk_size,
            }
            if "max_workers" in mp_cfg_fields:
                mp_kwargs["max_workers"] = 1
            if "hash_algorithm" in mp_cfg_fields:
                mp_kwargs["hash_algorithm"] = "blake3"

            mp_config = MPServerConfig(**mp_kwargs)

            kwargs: dict[str, object] = {
                "mp_config": mp_config,
                "storage_manager_config": storage_manager_config,
                "prometheus_config": DEFAULT_PROMETHEUS_CONFIG,
            }
            if "telemetry_config" in param_names:
                try:
                    from lmcache.v1.mp_observability.telemetry.config import (
                        DEFAULT_TELEMETRY_CONFIG,
                    )

                    kwargs["telemetry_config"] = DEFAULT_TELEMETRY_CONFIG
                except Exception:
                    pass

            run_cache_server(**kwargs)
            return

        raise RuntimeError(
            "Unsupported blend_server.run_cache_server signature: "
            f"{sorted(param_names)}"
        )

    from lmcache.v1.multiprocess import blend_server_v2

    run_cache_server = blend_server_v2.run_cache_server
    param_names = set(inspect.signature(run_cache_server).parameters)

    # Compatible with old-style v2 API used in tests:
    # run_cache_server(storage_manager_config, prometheus_config, host, port, chunk_size, ...)
    if {
        "storage_manager_config",
        "prometheus_config",
        "host",
        "port",
        "chunk_size",
    }.issubset(param_names):
        run_cache_server(
            storage_manager_config=storage_manager_config,
            prometheus_config=DEFAULT_PROMETHEUS_CONFIG,
            host=host,
            port=port,
            chunk_size=chunk_size,
        )
        return

    # Compatible with newer v2 API:
    # run_cache_server(mp_config, storage_manager_config, obs_config, ...)
    if {"mp_config", "storage_manager_config", "obs_config"}.issubset(param_names):
        from lmcache.v1.multiprocess.config import MPServerConfig

        mp_cfg_fields = set(inspect.signature(MPServerConfig).parameters)
        mp_kwargs: dict[str, object] = {
            "host": host,
            "port": port,
            "chunk_size": chunk_size,
        }
        if "max_workers" in mp_cfg_fields:
            mp_kwargs["max_workers"] = 1
        if "max_gpu_workers" in mp_cfg_fields:
            mp_kwargs["max_gpu_workers"] = 1
        if "max_cpu_workers" in mp_cfg_fields:
            mp_kwargs["max_cpu_workers"] = 1
        if "hash_algorithm" in mp_cfg_fields:
            mp_kwargs["hash_algorithm"] = "blake3"

        mp_config = MPServerConfig(**mp_kwargs)

        obs_config = DEFAULT_PROMETHEUS_CONFIG
        try:
            # Available in newer observability API.
            from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG

            obs_config = DEFAULT_OBSERVABILITY_CONFIG
        except Exception:
            pass

        run_cache_server(
            mp_config=mp_config,
            storage_manager_config=storage_manager_config,
            obs_config=obs_config,
        )
        return

    raise RuntimeError(
        "Unsupported blend_server_v2.run_cache_server signature: "
        f"{sorted(param_names)}"
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

    warmup_prompt = enc(("Nice to meet you. " * 500).strip())
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


def summarize_hit_ranges(hit_ranges: Iterable[tuple[int, int]], ranges: list[SegmentRange]) -> str:
    counts: dict[str, int] = {}
    for st, ed in hit_ranges:
        seg = locate_segment(ranges, st, ed)
        counts[seg] = counts.get(seg, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def get_probe_request_types(backend: ServerBackend | None) -> list[RequestType]:
    # v2 explicitly supports PING; NOOP is a useful compatibility fallback.
    if backend == "blend_v2":
        return [RequestType.PING, RequestType.NOOP]
    if backend == "blend_v1":
        return [RequestType.NOOP]
    # Unknown/external backend: probe both.
    return [RequestType.PING, RequestType.NOOP]


def wait_for_blend_server_ready(
    *,
    client: MessageQueueClient,
    server_url: str,
    backend: ServerBackend | None,
    server_proc: mp.Process | None,
    timeout_s: float,
    per_probe_timeout_s: float,
    poll_interval_s: float,
) -> RequestType:
    start = time.perf_counter()
    probe_types = get_probe_request_types(backend)
    last_err: Exception | None = None

    while time.perf_counter() - start < timeout_s:
        if server_proc is not None and not server_proc.is_alive():
            raise RuntimeError(
                "Blend server process exited while waiting for readiness. "
                "Please check child-process traceback above."
            )

        for rt in probe_types:
            try:
                client.submit_request(
                    rt,
                    [],
                    get_response_class(rt),
                ).result(timeout=per_probe_timeout_s)
                return rt
            except Exception as e:
                last_err = e

        time.sleep(poll_interval_s)

    err_msg = f"{type(last_err).__name__}: {last_err}" if last_err else "no response"
    raise RuntimeError(
        f"Cannot reach blend server at {server_url} within {timeout_s:.1f}s. "
        f"Last probe error: {err_msg}"
    )


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
        raise RuntimeError(f"Expected non-MLA KV shape [2,L,T,D], got {tuple(shape)}")

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


def run_cb_segment_verification(
    *,
    args: argparse.Namespace,
    client: MessageQueueClient,
    prompt_pack: PromptPack,
    cb_protocol: CBProtocol,
) -> list[CBLookupMetrics]:
    cb_rows: list[CBLookupMetrics] = []
    plain_kv: torch.Tensor | None = None

    instance_id = os.getpid() + 10000
    model_name_for_cb = "segment-hit-vllm-cb"

    try:
        client.submit_request(
            RequestType.CLEAR,
            [],
            get_response_class(RequestType.CLEAR),
        ).result(timeout=20)

        worker_impl = get_worker_connector_impl()
        plain_kv = export_request_kv_to_plain(
            worker_impl=worker_impl,
            num_tokens=len(prompt_pack.prompts["r1"]),
        )
        print(f"[cb][export] plain_kv_shape={tuple(plain_kv.shape)} dtype={plain_kv.dtype}")

        kv_cache: KVCache = [CudaIPCWrapper(plain_kv)]
        client.submit_request(
            RequestType.CB_REGISTER_KV_CACHE,
            [instance_id, kv_cache, model_name_for_cb, 1],
            get_response_class(RequestType.CB_REGISTER_KV_CACHE),
        ).result(timeout=20)

        # Store c1/c2/c3 using offsets in r1.
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
                f"[cb][store][{seg_name}] token_len={len(seg_tokens)} "
                f"offset={st} ok={ok} latency_ms={t1 - t0:.3f}"
            )
        print()

        # Lookup/retrieve on r1..r5 to validate segment hits by request.
        for req_name in CB_LOOKUP_ORDER:
            req_tokens = tuple(prompt_pack.prompts[req_name])
            key = create_cb_cache_key(
                token_ids=req_tokens,
                request_id=f"lookup-{req_name}",
                model_name=model_name_for_cb,
            )

            retrieve_ms = 0.0

            if cb_protocol == "v2":
                t0 = now_ms()
                matches = client.submit_request(
                    RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
                    [key],
                    get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED_V2),
                ).result(timeout=30)
                t1 = now_ms()

                lookup_ms = t1 - t0
                matches_sorted = sorted(matches, key=lambda m: m.cur_st)
                hits = len(matches_sorted)
                hit_tokens = sum((m.cur_ed - m.cur_st) for m in matches_sorted)
                seg_summary = summarize_segments(
                    matches_sorted, prompt_pack.segment_ranges[req_name]
                )

                print(
                    f"[cb][lookup][{req_name}] protocol=v2 hits={hits} "
                    f"hit_tokens={hit_tokens} latency_ms={lookup_ms:.3f} "
                    f"segments={seg_summary}"
                )
                for m in matches_sorted[: args.max_match_rows]:
                    seg = locate_segment(
                        prompt_pack.segment_ranges[req_name], m.cur_st, m.cur_ed
                    )
                    print(
                        f"  cur=[{m.cur_st},{m.cur_ed}) seg={seg:<6} "
                        f"old=[{m.old_st},{m.old_ed}) hash={m.hash.hex()[:16]}..."
                    )
                if len(matches_sorted) > args.max_match_rows:
                    print(f"  ... and {len(matches_sorted) - args.max_match_rows} more")

                if args.cb_retrieve_after_lookup:
                    event2 = torch.cuda.Event(interprocess=True)
                    event2.record()

                    t2 = now_ms()
                    ok2 = (
                        client.submit_request(
                            RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
                            [
                                key,
                                matches_sorted,
                                args.cb_retrieve_offset,
                                instance_id,
                                event2.ipc_handle(),
                            ],
                            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED_V2),
                        )
                        .to_cuda_future()
                        .result(timeout=40)
                    )
                    t3 = now_ms()

                    retrieve_ms = t3 - t2
                    print(
                        f"[cb][retrieve][{req_name}] protocol=v2 ok={ok2} "
                        f"offset={args.cb_retrieve_offset} latency_ms={retrieve_ms:.3f}"
                    )
            else:
                t0 = now_ms()
                hit_ranges = client.submit_request(
                    RequestType.CB_LOOKUP_PRE_COMPUTED,
                    [key],
                    get_response_class(RequestType.CB_LOOKUP_PRE_COMPUTED),
                ).result(timeout=30)
                t1 = now_ms()

                lookup_ms = t1 - t0
                ranges_sorted = sorted(hit_ranges, key=lambda x: x[0])
                hits = len(ranges_sorted)
                hit_tokens = sum((ed - st) for st, ed in ranges_sorted)
                seg_summary = summarize_hit_ranges(
                    ranges_sorted, prompt_pack.segment_ranges[req_name]
                )

                print(
                    f"[cb][lookup][{req_name}] protocol=v1 hits={hits} "
                    f"hit_tokens={hit_tokens} latency_ms={lookup_ms:.3f} "
                    f"segments={seg_summary}"
                )
                for st, ed in ranges_sorted[: args.max_match_rows]:
                    seg = locate_segment(prompt_pack.segment_ranges[req_name], st, ed)
                    print(f"  cur=[{st},{ed}) seg={seg:<6}")
                if len(ranges_sorted) > args.max_match_rows:
                    print(f"  ... and {len(ranges_sorted) - args.max_match_rows} more")

                if args.cb_retrieve_after_lookup:
                    event2 = torch.cuda.Event(interprocess=True)
                    event2.record()

                    t2 = now_ms()
                    ok2 = (
                        client.submit_request(
                            RequestType.CB_RETRIEVE_PRE_COMPUTED,
                            [
                                key,
                                ranges_sorted,
                                args.cb_retrieve_offset,
                                instance_id,
                                event2.ipc_handle(),
                            ],
                            get_response_class(RequestType.CB_RETRIEVE_PRE_COMPUTED),
                        )
                        .to_cuda_future()
                        .result(timeout=40)
                    )
                    t3 = now_ms()

                    retrieve_ms = t3 - t2
                    print(
                        f"[cb][retrieve][{req_name}] protocol=v1 ok={ok2} "
                        f"offset={args.cb_retrieve_offset} latency_ms={retrieve_ms:.3f}"
                    )
            print()

            cb_rows.append(
                CBLookupMetrics(
                    request=req_name,
                    hits=hits,
                    hit_tokens=hit_tokens,
                    lookup_ms=lookup_ms,
                    retrieve_ms=retrieve_ms,
                    segment_summary=seg_summary,
                )
            )

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

        if plain_kv is not None:
            del plain_kv
            torch.cuda.empty_cache()

    return cb_rows


def write_results(
    request_rows: list[RequestMetrics],
    cb_rows: list[CBLookupMetrics],
    results_dir: str,
    run_tag: str,
) -> dict[str, Path]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}

    req_json = out_dir / f"{run_tag}_segment-hit-vllm-requests.json"
    req_csv = out_dir / f"{run_tag}_segment-hit-vllm-requests.csv"
    req_payload = [asdict(row) for row in request_rows]
    req_json.write_text(json.dumps(req_payload, indent=2), encoding="utf-8")
    with req_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RequestMetrics.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(req_payload)

    output_paths["request_json"] = req_json
    output_paths["request_csv"] = req_csv

    if cb_rows:
        cb_json = out_dir / f"{run_tag}_segment-hit-vllm-cb.json"
        cb_csv = out_dir / f"{run_tag}_segment-hit-vllm-cb.csv"
        cb_payload = [asdict(row) for row in cb_rows]
        cb_json.write_text(json.dumps(cb_payload, indent=2), encoding="utf-8")
        with cb_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(CBLookupMetrics.__dataclass_fields__.keys()))
            writer.writeheader()
            writer.writerows(cb_payload)

        output_paths["cb_json"] = cb_json
        output_paths["cb_csv"] = cb_csv

    return output_paths


def sum_request_metrics(rows: list[RequestMetrics]) -> tuple[float, float, float, float]:
    return (
        sum(r.request_ms for r in rows),
        sum(r.lookup_ms for r in rows),
        sum(r.retrieve_ms for r in rows),
        sum(r.store_ms for r in rows),
    )


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

    parser.add_argument("-d", "--use-disk", action="store_true")
    parser.add_argument("--enable-sparse", action="store_true")

    parser.add_argument(
        "--use-layerwise",
        action="store_true",
        default=True,
        help="Use layerwise retrieve path (recommended for blending).",
    )
    parser.add_argument(
        "--no-layerwise",
        action="store_false",
        dest="use_layerwise",
        help="Disable layerwise retrieve path.",
    )
    parser.add_argument(
        "--blend-check-layers",
        type=str,
        default="1",
        help="Value for LMCACHE_BLEND_CHECK_LAYERS, e.g. '1' or '1,2'.",
    )
    parser.add_argument(
        "--blend-recompute-ratios",
        type=str,
        default="0.15",
        help="Value for LMCACHE_BLEND_RECOMPUTE_RATIOS, e.g. '0.15'.",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup request and start directly from r1.",
    )

    parser.add_argument("--start-server", action="store_true")
    parser.add_argument(
        "--server-backend",
        choices=["auto", "v1", "v2"],
        default="auto",
        help=(
            "Server backend to launch when --start-server is set. "
            "auto: try blend_server_v2 then fallback to blend_server."
        ),
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5567)
    parser.add_argument("--cpu-buffer-size-gb", type=float, default=5.0)

    parser.add_argument(
        "--cb-verify",
        action="store_true",
        help="Run CB server verification after r1 (requires a running blend server).",
    )
    parser.add_argument(
        "--cb-protocol",
        choices=["auto", "v1", "v2"],
        default="auto",
        help=(
            "CB lookup/retrieve protocol. auto picks v2 for blend_server_v2, "
            "or v1 for blend_server fallback."
        ),
    )
    parser.add_argument("--cb-retrieve-after-lookup", action="store_true")
    parser.add_argument("--cb-retrieve-offset", type=int, default=0)
    parser.add_argument("--max-match-rows", type=int, default=12)
    parser.add_argument(
        "--server-ready-timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for blend server readiness when cb-verify is enabled.",
    )
    parser.add_argument(
        "--server-ready-probe-timeout",
        type=float,
        default=2.0,
        help="Per-probe timeout in seconds for server readiness checks.",
    )
    parser.add_argument(
        "--server-ready-poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds between server readiness probes.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/gcp-vm/projects/LMCache/experiments/blend/results",
        help="Directory to save raw metrics as CSV/JSON.",
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
        use_layerwise=args.use_layerwise,
        blend_check_layers=args.blend_check_layers,
        blend_recompute_ratios=args.blend_recompute_ratios,
    )
    install_connector_probe()

    server_proc: mp.Process | None = None
    selected_server_backend: ServerBackend | None = None
    server_backend_fallback_reason: str | None = None
    if args.start_server:
        try:
            selected_server_backend, server_backend_fallback_reason = choose_server_backend(
                args.server_backend
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot initialize requested server backend '{args.server_backend}': "
                f"{type(e).__name__}: {e}"
            ) from e
        server_proc = mp.Process(
            target=server_process_runner,
            args=(
                args.host,
                args.port,
                args.chunk_size,
                args.cpu_buffer_size_gb,
                selected_server_backend,
            ),
            daemon=True,
        )
        server_proc.start()
        time.sleep(3)
        if not server_proc.is_alive():
            raise RuntimeError(
                "Blend server process exited during startup. "
                "Please check child-process traceback above."
            )

    selected_cb_protocol: CBProtocol = "v2"
    if args.cb_protocol == "v1":
        selected_cb_protocol = "v1"
    elif args.cb_protocol == "v2":
        selected_cb_protocol = "v2"
    else:
        if selected_server_backend == "blend_v1":
            selected_cb_protocol = "v1"
        else:
            selected_cb_protocol = "v2"

    if args.cb_verify and selected_server_backend == "blend_v1" and selected_cb_protocol == "v2":
        raise RuntimeError(
            "cb-protocol=v2 is incompatible with blend_server (v1 backend). "
            "Use --cb-protocol v1, or launch a working blend_server_v2."
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
    print(f"LMCACHE_BLEND_CHECK_LAYERS={os.getenv('LMCACHE_BLEND_CHECK_LAYERS')}")
    print(f"LMCACHE_BLEND_RECOMPUTE_RATIOS={os.getenv('LMCACHE_BLEND_RECOMPUTE_RATIOS')}")
    print(f"LMCACHE_USE_LAYERWISE={os.getenv('LMCACHE_USE_LAYERWISE')}")
    print(f"LMCACHE_FORCE_SKIP_SAVE={os.getenv('LMCACHE_FORCE_SKIP_SAVE')}")
    print(f"VLLM_ENABLE_V1_MULTIPROCESSING={os.getenv('VLLM_ENABLE_V1_MULTIPROCESSING')}")
    print(f"VLLM_WORKER_MULTIPROC_METHOD={os.getenv('VLLM_WORKER_MULTIPROC_METHOD')}")
    if args.cb_verify or args.start_server:
        print(f"blend_server_url=tcp://{args.host}:{args.port}")
    if args.start_server:
        print(f"blend_server_backend={selected_server_backend}")
        if server_backend_fallback_reason:
            print(f"blend_server_backend_fallback_reason={server_backend_fallback_reason}")
    if args.cb_verify:
        print(f"cb_protocol={selected_cb_protocol}")
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
    cb_rows: list[CBLookupMetrics] = []

    cb_client: MessageQueueClient | None = None
    if args.cb_verify:
        server_url = f"tcp://{args.host}:{args.port}"
        context = zmq.Context.instance()
        cb_client = MessageQueueClient(server_url=server_url, context=context)
        ready_via = wait_for_blend_server_ready(
            client=cb_client,
            server_url=server_url,
            backend=selected_server_backend,
            server_proc=server_proc,
            timeout_s=args.server_ready_timeout,
            per_probe_timeout_s=args.server_ready_probe_timeout,
            poll_interval_s=args.server_ready_poll_interval,
        )
        print(f"blend_server_ready_via={ready_via.name}")

    try:
        with build_llm_with_lmcache(model=args.model) as llm:
            engine = LMCacheEngineBuilder.get(ENGINE_NAME)
            if engine is None:
                raise RuntimeError("Failed to get LMCache engine instance")
            monitor = engine.stats_monitor

            request_sequence = REQUEST_ORDER if not args.skip_warmup else REQUEST_ORDER[1:]

            print("Phase 1: vLLM end-to-end requests")
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

                # Run CB verification right after r1 so exported KV is exactly from r1.
                if args.cb_verify and name == "r1":
                    if cb_client is None:
                        raise RuntimeError("CB verify requested but MessageQueueClient is not ready")
                    print()
                    print(
                        "Phase 2: CB segment verification "
                        f"(backend={selected_server_backend}, protocol={selected_cb_protocol})"
                    )
                    cb_rows = run_cb_segment_verification(
                        args=args,
                        client=cb_client,
                        prompt_pack=prompt_pack,
                        cb_protocol=selected_cb_protocol,
                    )
                    print("Phase 2 done")
                    print()

                if args.sleep_between > 0:
                    time.sleep(args.sleep_between)
            print()

        req_req, req_lookup, req_ret, req_store = sum_request_metrics(rows)
        print("Latency summary (ms):")
        print(
            "  request_phase: "
            f"request={req_req:.3f}, lookup={req_lookup:.3f}, "
            f"retrieve={req_ret:.3f}, store={req_store:.3f}"
        )

        if cb_rows:
            print("CB summary:")
            print(f"  total_hits={sum(r.hits for r in cb_rows)}")
            print(f"  total_hit_tokens={sum(r.hit_tokens for r in cb_rows)}")
            print(f"  lookup_total_ms={sum(r.lookup_ms for r in cb_rows):.3f}")
            print(f"  retrieve_total_ms={sum(r.retrieve_ms for r in cb_rows):.3f}")

        run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_paths = write_results(
            request_rows=rows,
            cb_rows=cb_rows,
            results_dir=args.results_dir,
            run_tag=run_tag,
        )
        for k, path in output_paths.items():
            print(f"Saved {k}: {path}")

    finally:
        if cb_client is not None:
            cb_client.close()

        if server_proc is not None and server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.kill()
                server_proc.join(timeout=2)


if __name__ == "__main__":
    main()
