# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
import argparse
import contextlib
import csv
import os
import time
from typing import Dict, List

import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = "# #",
    enable_sparse: bool = False,
    internal_api_server_port_start: int = 6999,
    use_layerwise: bool = False,
):
    # LMCache chunk size
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending configs
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True" if use_layerwise else "False"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    os.environ["LMCACHE_INTERNAL_API_SERVER_ENABLED"] = "True"
    os.environ["LMCACHE_INTERNAL_API_SERVER_PORT_START"] = str(
        internal_api_server_port_start
    )

    # Optional but often useful for stable hashing behavior
    os.environ.setdefault("PYTHONHASHSEED", "0")

    # Avoid HF Xet downloader instability on some environments
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
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=12000,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        enforce_eager=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
    metrics_base_url: str | None = None,
):
    before = fetch_metrics(metrics_base_url) if metrics_base_url else {}
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt},
        sampling_params=sampling_params,
    )
    elapsed = time.time() - start
    after = fetch_metrics(metrics_base_url) if metrics_base_url else {}
    delta = diff_metrics(before, after) if before and after else {}

    print("-" * 80)
    print(f"[{req_str}] prompt_tokens={len(prompt)}")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        # Best-effort vLLM-side hit/caching hints.
        if hasattr(output, "num_cached_tokens"):
            print(f"vllm_num_cached_tokens={getattr(output, 'num_cached_tokens')}")
        if hasattr(output, "metrics") and output.metrics is not None:
            m = output.metrics
            for key in [
                "num_cached_tokens",
                "num_prefill_tokens",
                "num_decode_tokens",
            ]:
                if hasattr(m, key):
                    print(f"vllm_metrics_{key}={getattr(m, key)}")

    if delta:
        print("LMCache metric deltas:")
        for k in [
            "lmcache:num_lookup_hits_total",
            "lmcache:num_hit_tokens_total",
            "lmcache:num_vllm_hit_tokens_total",
            "lmcache:time_to_lookup_count_total",
            "lmcache:time_to_retrieve_count_total",
            "lmcache:time_to_store_count_total",
        ]:
            print(f"  {k}: {delta.get(k, 0.0)}")
    print(f"Generation took {elapsed:.2f} seconds, {req_str} request done.")
    print("-" * 80)
    return elapsed, delta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Use disk as LMCache backend instead of CPU memory.",
    )
    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Special separator string for blending.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path (defaults to --model).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load tokenizer from local cache/files only (offline-friendly).",
    )
    parser.add_argument(
        "--enable-sparse",
        action="store_true",
        help="Enable sparse blending path if supported.",
    )
    parser.add_argument(
        "--use-layerwise",
        action="store_true",
        help="Enable LMCACHE_USE_LAYERWISE=True. Default is False for stability.",
    )
    parser.add_argument(
        "--segment-token-length",
        type=int,
        default=256,
        help="Token length for A/A0/A1/A5/B/B0/B00/B1/B2/B5 segments.",
    )
    parser.add_argument(
        "--chunk-token-length",
        type=int,
        default=256,
        help="Token length for chunk/chunk0.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Generation max tokens per request.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between requests.",
    )
    parser.add_argument(
        "--internal-api-server-port-start",
        type=int,
        default=6999,
        help="LMCache internal API server base port (worker0 usually uses +1).",
    )
    parser.add_argument(
        "--metrics-base-url",
        default="http://127.0.0.1:7000",
        help="LMCache metrics endpoint base URL, e.g. http://127.0.0.1:7000",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path to save per-request metric deltas.",
    )
    return parser.parse_args()


def fetch_metrics(metrics_base_url: str) -> Dict[str, float]:
    try:
        resp = requests.get(f"{metrics_base_url.rstrip('/')}/metrics", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Metrics] fetch failed: {e}")
        return {}
    out: Dict[str, float] = {}
    for raw in resp.text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            k, v = line.rsplit(" ", 1)
            out[k] = float(v)
        except Exception:
            continue
    return out


def metric_sum(metrics: Dict[str, float], name: str) -> float:
    total = 0.0
    for k, v in metrics.items():
        if k == name or k.startswith(name + "{"):
            total += v
    return total


def diff_metrics(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    wanted = [
        "lmcache:num_lookup_hits_total",
        "lmcache:num_hit_tokens_total",
        "lmcache:num_vllm_hit_tokens_total",
        "lmcache:time_to_lookup_count_total",
        "lmcache:time_to_retrieve_count_total",
        "lmcache:time_to_store_count_total",
    ]
    return {k: metric_sum(after, k) - metric_sum(before, k) for k in wanted}


def make_exact_token_ids(
    tokenizer: AutoTokenizer,
    target_tokens: int,
    unit_text: str,
) -> List[int]:
    if target_tokens <= 0:
        return []
    unit_ids = tokenizer.encode(unit_text, add_special_tokens=False)
    if not unit_ids:
        raise ValueError("unit_text tokenizes to empty ids")
    out: List[int] = []
    while len(out) + len(unit_ids) <= target_tokens:
        out.extend(unit_ids)
    if len(out) < target_tokens:
        filler = tokenizer.encode("x " * (target_tokens * 4), add_special_tokens=False)
        out.extend(filler[: target_tokens - len(out)])
    return out[:target_tokens]


def make_segment_token_ids(
    tokenizer: AutoTokenizer,
    label: str,
    target_tokens: int,
) -> List[int]:
    prefix_ids = tokenizer.encode(f"{label}: ", add_special_tokens=False)
    if len(prefix_ids) >= target_tokens:
        return prefix_ids[:target_tokens]
    body = make_exact_token_ids(
        tokenizer,
        target_tokens - len(prefix_ids),
        f"{label.lower()} context detail evidence rationale ",
    )
    return prefix_ids + body


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    blend_special_str: str,
    segment_tokens: int,
    chunk_tokens: int,
) -> Dict[str, List[int]]:
    sep_ids = tokenizer.encode(blend_special_str, add_special_tokens=False)
    sys_prompt = tokenizer.encode(
        "You are a helpful assistant. Please answer briefly.",
        add_special_tokens=False,
    )

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

    chunk = make_exact_token_ids(
        tokenizer,
        chunk_tokens,
        "tool result value success explanation ",
    )
    chunk0 = make_exact_token_ids(
        tokenizer,
        chunk_tokens,
        "chunk0 independent baseline context value ",
    )
    if chunk0 == chunk:
        chunk0 = make_exact_token_ids(
            tokenizer,
            chunk_tokens,
            "chunk0 unique unrelated baseline signal ",
        )

    # Keep warmup as a plain long text (no separator), similar to blendsmall,
    # to avoid triggering blend segmentation on the very first request.
    warmup = tokenizer.encode(
        ("Warmup unrelated calibration sentence. " * 500).strip(),
        add_special_tokens=False,
    )

    return {
        "warmup": warmup,
        "R0": sys_prompt + sep_ids + a0 + sep_ids + chunk0 + sep_ids + b0,
        "R00": sys_prompt + sep_ids + a0 + sep_ids + chunk + sep_ids + b00,
        "R1": sys_prompt + sep_ids + a1 + sep_ids + chunk + sep_ids + b1,
        "R2": sys_prompt + sep_ids + a1 + sep_ids + chunk + sep_ids + b2,
        "R3": sys_prompt + sep_ids + a + sep_ids + b + sep_ids + chunk,
        "R4": sys_prompt + sep_ids + chunk + sep_ids + a + sep_ids + b,
        "R5": sys_prompt + sep_ids + chunk + sep_ids + a5 + sep_ids + b5,
    }


def save_rows_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_environment_variables(
        use_disk=args.use_disk,
        blend_special_str=args.blend_special_str,
        enable_sparse=args.enable_sparse,
        internal_api_server_port_start=args.internal_api_server_port_start,
        use_layerwise=args.use_layerwise,
    )

    tokenizer_name = args.tokenizer or model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        local_files_only=args.local_files_only,
    )

    requests_map = build_prompt_ids(
        tokenizer=tokenizer,
        blend_special_str=os.getenv("LMCACHE_BLEND_SPECIAL_STR", args.blend_special_str),
        segment_tokens=args.segment_token_length,
        chunk_tokens=args.chunk_token_length,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    print(f"Using model: {model}")
    print(f"LMCACHE_CHUNK_SIZE={os.getenv('LMCACHE_CHUNK_SIZE')}")
    print(f"LMCACHE_ENABLE_BLENDING={os.getenv('LMCACHE_ENABLE_BLENDING')}")
    print(f"LMCACHE_BLEND_SPECIAL_STR={os.getenv('LMCACHE_BLEND_SPECIAL_STR')}")
    print(f"LMCACHE_BLEND_RECOMPUTE_RATIOS={os.getenv('LMCACHE_BLEND_RECOMPUTE_RATIOS')}")
    print(f"LMCACHE_USE_LAYERWISE={os.getenv('LMCACHE_USE_LAYERWISE')}")
    print(f"Metrics endpoint={args.metrics_base_url}")
    print()

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        order = ["warmup", "R0", "R00", "R1", "R2", "R3", "R4", "R5"]
        rows: List[Dict[str, object]] = []
        for idx, req in enumerate(order):
            try:
                elapsed, delta = print_output(
                    llm,
                    requests_map[req],
                    sampling_params,
                    req,
                    metrics_base_url=args.metrics_base_url,
                )
            except Exception as e:
                print(f"[Fatal] request={req} failed: {e}")
                print(
                    "Suggestion: keep --use-layerwise disabled (default), "
                    "and retry. If needed, reduce --chunk-token-length."
                )
                raise
            row: Dict[str, object] = {
                "request_name": req,
                "prompt_tokens": len(requests_map[req]),
                "elapsed_seconds": elapsed,
            }
            row.update(delta)
            rows.append(row)
            if idx != len(order) - 1:
                time.sleep(args.sleep_seconds)

        if args.output_csv:
            save_rows_csv(rows, args.output_csv)
            print(f"Saved per-request metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
