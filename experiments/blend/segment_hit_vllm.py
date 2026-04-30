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

2) Optional blend-server CB verification (--cb-verify, V2 by default)
   - after r1, export r1 KV from vLLM paged buffer
   - register KV buffer to blend server
   - store c1/c2/c3 from r1 offsets with CB_STORE_PRE_COMPUTED
   - run CB_LOOKUP_PRE_COMPUTED_V2 (and optional retrieve) on r1..r5

Examples:
- Pure vLLM E2E blend mode:
  python segment_hit_vllm.py --model meta-llama/Llama-3.2-3B-Instruct

- Start local blend server and run CB verification too:
  python segment_hit_vllm.py --start-server --cb-verify --server-backend v2

Recommended for quick/stable runs on current blend-server-v2:
  python segment_hit_vllm.py --start-server --cb-verify \
      --server-backend v2 \
      --segment-chunks 2 --warmup-chunks 2 \
      --max-model-len 4096
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
from typing import Any, Iterable, Literal

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
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper, IPCCacheEngineKey, KVCache
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class


REQUEST_ORDER = ["warmup", "r1", "r2", "r3", "r4", "r5"]
CB_LOOKUP_ORDER = ["r1", "r2", "r3", "r4", "r5"]
_CAPTURED_CONNECTORS: list[object] = []
CBProtocol = Literal["v2"]
ServerBackend = Literal["blend_v2"]

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


def to_python_value(value: Any) -> Any:
    """Convert torch/numpy-like scalars and containers to JSON-safe Python values."""
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
    # Fallback for scalar-like objects (e.g. numpy scalars) exposing item().
    item = getattr(value, "item", None)
    if callable(item):
        return to_python_value(item())
    return str(value)


def to_int(value: Any) -> int:
    return int(to_python_value(value))


def to_float(value: Any) -> float:
    return float(to_python_value(value))


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
    # Use "0" to avoid ambiguous bool("False") parsing in some call paths.
    os.environ["LMCACHE_FORCE_SKIP_SAVE"] = "0"

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
def build_llm_with_lmcache(
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
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
    if server_backend not in ("auto", "v2"):
        raise ValueError(
            "This repo version supports only blend_server_v2. "
            f"Got --server-backend={server_backend!r}."
        )
    from lmcache.v1.multiprocess import blend_server_v2  # noqa: F401

    return "blend_v2", None


def server_process_runner(
    host: str,
    port: int,
    chunk_size: int,
    cpu_buffer_size_gb: float,
    backend: ServerBackend,
) -> None:
    if backend != "blend_v2":
        raise RuntimeError(f"Unsupported backend in current repo: {backend}")

    from lmcache.v1.multiprocess.blend_server_v2 import run_cache_server
    from lmcache.v1.multiprocess.config import MPServerConfig

    mp_cfg_fields = set(inspect.signature(MPServerConfig).parameters)
    mp_kwargs: dict[str, object] = {
        "host": host,
        "port": port,
        "chunk_size": chunk_size,
    }
    if "engine_type" in mp_cfg_fields:
        mp_kwargs["engine_type"] = "blend"
    if "max_workers" in mp_cfg_fields:
        mp_kwargs["max_workers"] = 1
    if "max_gpu_workers" in mp_cfg_fields:
        mp_kwargs["max_gpu_workers"] = 1
    if "max_cpu_workers" in mp_cfg_fields:
        mp_kwargs["max_cpu_workers"] = 1
    if "hash_algorithm" in mp_cfg_fields:
        mp_kwargs["hash_algorithm"] = "blake3"

    mp_config = MPServerConfig(**mp_kwargs)

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
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


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
            # Extremely defensive fallback for tokenizers that may return [] on short text.
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

    warmup_prompt = make_chunk_aligned_segment("This is just a warmup message.", warmup_tokens)
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
    engine_fmt = getattr(lmcache_engine, "fmt", None)

    def _infer_layerwise_fmt() -> MemoryFormat:
        if len(shape) != 3:
            raise RuntimeError(f"Expected layerwise 3D shape, got {tuple(shape)}")
        # Layerwise connectors expose either [2, T, D] or [T, 2, D].
        if shape[0] == 2:
            return MemoryFormat.KV_2TD
        if shape[1] == 2:
            return MemoryFormat.KV_T2D
        if engine_fmt in (MemoryFormat.KV_2TD, MemoryFormat.KV_T2D):
            return engine_fmt
        raise RuntimeError(
            "Expected non-MLA layerwise shape [2,T,D] or [T,2,D], "
            f"got {tuple(shape)}"
        )

    def _export_via_batched_from_gpu(layer_fmt: MemoryFormat) -> torch.Tensor:
        num_layers = int(
            getattr(lmcache_engine, "num_layers", 0)
            or getattr(gpu_connector, "num_layers", 0)
        )
        if num_layers <= 0:
            raise RuntimeError(
                "Cannot determine num_layers for batched_from_gpu export "
                f"(engine={getattr(lmcache_engine, 'num_layers', None)}, "
                f"connector={getattr(gpu_connector, 'num_layers', None)})"
            )

        hidden_dim = int(shape[2])
        if layer_fmt == MemoryFormat.KV_2TD:
            layer_buffers = [
                torch.empty((2, num_tokens, hidden_dim), dtype=kv_caches[0].dtype, device=dev)
                for _ in range(num_layers)
            ]
        elif layer_fmt == MemoryFormat.KV_T2D:
            layer_buffers = [
                torch.empty((num_tokens, 2, hidden_dim), dtype=kv_caches[0].dtype, device=dev)
                for _ in range(num_layers)
            ]
        else:
            raise RuntimeError(f"Unsupported layerwise format for export: {layer_fmt}")

        layer_memory_objs = [
            [SimpleNamespace(tensor=buf, metadata=SimpleNamespace(fmt=layer_fmt))]
            for buf in layer_buffers
        ]

        batched_ret = gpu_connector.batched_from_gpu(
            layer_memory_objs,
            [0],
            [num_tokens],
            kvcaches=kv_caches,
            slot_mapping=slot_mapping,
            sync=True,
        )
        if batched_ret is not None:
            for _ in batched_ret:
                pass

        # CB server expects a single contiguous plain tensor [2, L, T, D].
        merged = torch.empty(
            (2, num_layers, num_tokens, hidden_dim),
            dtype=kv_caches[0].dtype,
            device=dev,
        )
        for layer_id, buf in enumerate(layer_buffers):
            if layer_fmt == MemoryFormat.KV_2TD:
                merged[:, layer_id].copy_(buf, non_blocking=True)
            else:  # KV_T2D
                merged[:, layer_id].copy_(buf.permute(1, 0, 2), non_blocking=True)
        return merged

    if len(shape) == 4:
        # [2, L, T, D]
        if shape[0] != 2:
            raise RuntimeError(
                "Expected non-MLA KV shape [2,L,T,D], "
                f"got {tuple(shape)} (shape[0] must be 2)"
            )
        plain_kv = torch.empty(shape, dtype=kv_caches[0].dtype, device=dev)
        mem_obj = SimpleNamespace(
            tensor=plain_kv,
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        )
        try:
            gpu_connector.from_gpu(
                mem_obj,
                0,
                num_tokens,
                kvcaches=kv_caches,
                slot_mapping=slot_mapping,
            )
        except NotImplementedError:
            # Some layerwise connectors only support batched_from_gpu.
            plain_kv = _export_via_batched_from_gpu(
                engine_fmt if engine_fmt in (MemoryFormat.KV_2TD, MemoryFormat.KV_T2D) else MemoryFormat.KV_2TD
            )
    elif len(shape) == 3:
        plain_kv = _export_via_batched_from_gpu(_infer_layerwise_fmt())
    else:
        raise RuntimeError(
            "Expected non-MLA KV shape [2,L,T,D], [2,T,D], or [T,2,D], "
            f"got {tuple(shape)}"
        )

    if dev.type == "cuda":
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
                f"[cb][lookup][{req_name}] protocol={cb_protocol} hits={hits} "
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
                    f"[cb][retrieve][{req_name}] protocol={cb_protocol} ok={ok2} "
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
    args: argparse.Namespace | None = None,
) -> dict[str, Path]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}

    req_json = out_dir / f"{run_tag}_segment-hit-vllm-requests.json"
    req_csv = out_dir / f"{run_tag}_segment-hit-vllm-requests.csv"
    req_payload = [to_python_value(asdict(row)) for row in request_rows]
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
        cb_payload = [to_python_value(asdict(row)) for row in cb_rows]
        cb_json.write_text(json.dumps(cb_payload, indent=2), encoding="utf-8")
        with cb_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(CBLookupMetrics.__dataclass_fields__.keys()))
            writer.writeheader()
            writer.writerows(cb_payload)

        output_paths["cb_json"] = cb_json
        output_paths["cb_csv"] = cb_csv

    req_summary = {
        "request_count": len(req_payload),
        "totals_ms": {
            "request": to_float(sum(r["request_ms"] for r in req_payload)),
            "lookup": to_float(sum(r["lookup_ms"] for r in req_payload)),
            "retrieve": to_float(sum(r["retrieve_ms"] for r in req_payload)),
            "store": to_float(sum(r["store_ms"] for r in req_payload)),
        },
        "mean_ms": {
            "request": to_float(sum(r["request_ms"] for r in req_payload) / max(1, len(req_payload))),
            "lookup": to_float(sum(r["lookup_ms"] for r in req_payload) / max(1, len(req_payload))),
            "retrieve": to_float(sum(r["retrieve_ms"] for r in req_payload) / max(1, len(req_payload))),
            "store": to_float(sum(r["store_ms"] for r in req_payload) / max(1, len(req_payload))),
        },
        "totals_tokens": {
            "prompt_tokens": to_int(sum(r["prompt_tokens"] for r in req_payload)),
            "lookup_hit_tokens": to_int(sum(r["lookup_hit_tokens"] for r in req_payload)),
            "retrieve_hit_tokens": to_int(sum(r["retrieve_hit_tokens"] for r in req_payload)),
            "store_tokens": to_int(sum(r["store_tokens"] for r in req_payload)),
        },
        "events": {
            "lookup_events": to_int(sum(r["lookup_events"] for r in req_payload)),
            "retrieve_events": to_int(sum(r["retrieve_events"] for r in req_payload)),
            "store_events": to_int(sum(r["store_events"] for r in req_payload)),
        },
    }

    cb_summary = {
        "request_count": len(cb_rows),
        "total_hits": to_int(sum(r.hits for r in cb_rows)),
        "total_hit_tokens": to_int(sum(r.hit_tokens for r in cb_rows)),
        "lookup_total_ms": to_float(sum(r.lookup_ms for r in cb_rows)),
        "retrieve_total_ms": to_float(sum(r.retrieve_ms for r in cb_rows)),
        "lookup_mean_ms": to_float(
            sum(r.lookup_ms for r in cb_rows) / max(1, len(cb_rows))
        ),
        "retrieve_mean_ms": to_float(
            sum(r.retrieve_ms for r in cb_rows) / max(1, len(cb_rows))
        ),
    }

    summary_payload = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args) if args is not None else None,
        "request_summary": req_summary,
        "cb_summary": cb_summary,
        "files": {
            "request_json": str(req_json),
            "request_csv": str(req_csv),
            "cb_json": str(output_paths.get("cb_json", "")),
            "cb_csv": str(output_paths.get("cb_csv", "")),
        },
    }

    summary_json = out_dir / f"{run_tag}_segment-hit-vllm-summary.json"
    summary_json.write_text(json.dumps(to_python_value(summary_payload), indent=2), encoding="utf-8")
    output_paths["summary_json"] = summary_json

    summary_csv = out_dir / f"{run_tag}_segment-hit-vllm-summary.csv"
    summary_rows = [
        {"section": "request_totals_ms", "metric": "request", "value": req_summary["totals_ms"]["request"]},
        {"section": "request_totals_ms", "metric": "lookup", "value": req_summary["totals_ms"]["lookup"]},
        {"section": "request_totals_ms", "metric": "retrieve", "value": req_summary["totals_ms"]["retrieve"]},
        {"section": "request_totals_ms", "metric": "store", "value": req_summary["totals_ms"]["store"]},
        {"section": "request_mean_ms", "metric": "request", "value": req_summary["mean_ms"]["request"]},
        {"section": "request_mean_ms", "metric": "lookup", "value": req_summary["mean_ms"]["lookup"]},
        {"section": "request_mean_ms", "metric": "retrieve", "value": req_summary["mean_ms"]["retrieve"]},
        {"section": "request_mean_ms", "metric": "store", "value": req_summary["mean_ms"]["store"]},
        {
            "section": "request_tokens",
            "metric": "lookup_hit_tokens",
            "value": req_summary["totals_tokens"]["lookup_hit_tokens"],
        },
        {
            "section": "request_tokens",
            "metric": "retrieve_hit_tokens",
            "value": req_summary["totals_tokens"]["retrieve_hit_tokens"],
        },
        {
            "section": "request_tokens",
            "metric": "store_tokens",
            "value": req_summary["totals_tokens"]["store_tokens"],
        },
        {"section": "cb", "metric": "total_hits", "value": cb_summary["total_hits"]},
        {"section": "cb", "metric": "total_hit_tokens", "value": cb_summary["total_hit_tokens"]},
        {"section": "cb", "metric": "lookup_total_ms", "value": cb_summary["lookup_total_ms"]},
        {"section": "cb", "metric": "retrieve_total_ms", "value": cb_summary["retrieve_total_ms"]},
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)
    output_paths["summary_csv"] = summary_csv

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
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max_model_len. Keep this close to prompt lengths for stability.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM gpu_memory_utilization.",
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
        "--segment-chunks",
        type=int,
        default=2,
        help=(
            "Per-segment token length in units of chunk_size. "
            "c1..c5 are built to chunk-aligned lengths."
        ),
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=2,
        help="Warmup prompt token length in units of chunk_size.",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup request and start directly from r1.",
    )

    parser.add_argument("--start-server", action="store_true")
    parser.add_argument(
        "--server-backend",
        choices=["auto", "v2"],
        default="v2",
        help=(
            "Server backend to launch when --start-server is set. "
            "Current repo supports blend_server_v2 only."
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
        choices=["auto", "v2"],
        default="auto",
        help=(
            "CB lookup/retrieve protocol. Current repo uses v2 only "
            "(CB_LOOKUP_PRE_COMPUTED_V2 / CB_RETRIEVE_PRE_COMPUTED_V2)."
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_pack = build_prompt_pack(
        tokenizer=tokenizer,
        blend_special_str=os.getenv("LMCACHE_BLEND_SPECIAL_STR", "# #"),
        chunk_size=args.chunk_size,
        segment_chunks=args.segment_chunks,
        warmup_chunks=args.warmup_chunks,
    )
    max_prompt_tokens = max(len(v) for v in prompt_pack.prompts.values())
    if max_prompt_tokens + args.max_tokens > args.max_model_len:
        raise RuntimeError(
            "Prompt length exceeds max_model_len budget: "
            f"max_prompt_tokens={max_prompt_tokens}, max_tokens={args.max_tokens}, "
            f"max_model_len={args.max_model_len}. "
            "Increase --max-model-len or decrease --segment-chunks/--warmup-chunks."
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
    print(f"max_model_len={args.max_model_len}")
    print(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"segment_chunks={args.segment_chunks}")
    print(f"warmup_chunks={args.warmup_chunks}")
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
        with build_llm_with_lmcache(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as llm:
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
            args=args,
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
