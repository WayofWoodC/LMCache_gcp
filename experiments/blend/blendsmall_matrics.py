# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
import argparse
import contextlib
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.token_database import ChunkedTokenDatabase


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = "# #",
    enable_sparse: bool = False,
):
    # LMCache chunk size
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending configs
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"

    # Optional but often useful for stable hashing behavior
    os.environ.setdefault("PYTHONHASHSEED", "0")

    # Avoid HF Xet downloader instability on some environments
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Enable chunk statistics so we can inspect chunk reuse details.
    os.environ["LMCACHE_ENABLE_CHUNK_STATISTICS"] = "True"
    os.environ["LMCACHE_CHUNK_STATISTICS_AUTO_START_STATISTICS"] = "True"
    os.environ["LMCACHE_CHUNK_STATISTICS_STRATEGY"] = "file_hash"

    extra_config = {
        "chunk_statistics_file_output_dir": "./chunk_hashes",
    }
    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        extra_config["enable_sparse"] = True
    os.environ["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)

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
        if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt},
        sampling_params=sampling_params,
    )
    elapsed = time.time() - start

    print("-" * 80)
    print(f"[{req_str}] prompt_tokens={len(prompt)}")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {elapsed:.2f} seconds, {req_str} request done.")
    print("-" * 80)


def locate_segment(
    segment_ranges: list[tuple[str, int, int]],
    start: int,
    end: int,
) -> str:
    for name, seg_start, seg_end in segment_ranges:
        if start < seg_end and end > seg_start:
            return name
    return "unknown"


def get_chunk_entries(
    prompt_tokens: list[int],
    segment_ranges: list[tuple[str, int, int]],
    chunk_size: int,
) -> list[dict]:
    token_db = ChunkedTokenDatabase()
    token_db.chunk_size = chunk_size
    entries: list[dict] = []
    for idx, (start, end, hash_val) in enumerate(
        token_db.process_tokens(tokens=prompt_tokens, make_key=False)
    ):
        h = int(hash_val)
        if h < 0:
            h = h & ((1 << 64) - 1)
        entries.append(
            {
                "chunk_idx": idx,
                "start": start,
                "end": end,
                "segment": locate_segment(segment_ranges, start, end),
                "hash_hex": hex(h),
            }
        )
    return entries


def print_chunk_reuse_report(
    req_str: str,
    prompt_tokens: list[int],
    segment_ranges: list[tuple[str, int, int]],
    chunk_size: int,
    seen_hashes: set[str],
):
    entries = get_chunk_entries(prompt_tokens, segment_ranges, chunk_size)
    reused = [e for e in entries if e["hash_hex"] in seen_hashes]
    new_chunks = [e for e in entries if e["hash_hex"] not in seen_hashes]

    for e in entries:
        seen_hashes.add(e["hash_hex"])

    print(
        f"[Chunk hash view][{req_str}] total_chunks={len(entries)}, "
        f"reused_chunks={len(reused)}, new_chunks={len(new_chunks)}"
    )
    if reused:
        print(f"[Chunk hash view][{req_str}] reused chunk details:")
        for e in reused[:20]:
            print(
                f"  chunk_idx={e['chunk_idx']}, token_range=[{e['start']},{e['end']}), "
                f"segment={e['segment']}, hash={e['hash_hex']}"
            )
        if len(reused) > 20:
            print(f"  ... and {len(reused) - 20} more reused chunks")


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
        "--enable-sparse",
        action="store_true",
        help="Enable sparse blending path if supported.",
    )
    return parser.parse_args()


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    blend_special_str: str,
):
    sep_ids = tokenizer.encode(blend_special_str, add_special_tokens=False)

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def build_segment_ids(tag: str, base: str, target_tokens: int = 2400) -> list[int]:
        unit_ids = enc(f"[{tag}] {base} ")
        if not unit_ids:
            unit_ids = enc(tag)
        repeat = (target_tokens + len(unit_ids) - 1) // len(unit_ids)
        return (unit_ids * repeat)[:target_tokens]

    # Keep all major segments at roughly similar token length while staying
    # safely below max_model_len.
    chunk0_prompt = build_segment_ids(
        "chunk0",
        "Warmup-only chunk for priming; this content should not be reused in R1-R5.",
    )
    shared_chunk_prompt = build_segment_ids(
        "chunk",
        "Shared reusable chunk that should be reused across R1-R5 in different layouts.",
    )

    A0_prompt = build_segment_ids(
        "A0",
        "Warmup prefix A0 with standalone semantics and unique lexical patterns.",
    )
    B0_prompt = build_segment_ids(
        "B0",
        "Warmup suffix B0 asks for concise continuation and remains unique.",
    )
    A1_prompt = build_segment_ids(
        "A1",
        "Prefix A1 introduces topic alpha and instruction style one.",
    )
    B1_prompt = build_segment_ids(
        "B1",
        "Suffix B1 enforces response format one with unique marker.",
    )
    A2_prompt = build_segment_ids(
        "A2",
        "Prefix A2 introduces topic beta and instruction style two.",
    )
    B2_prompt = build_segment_ids(
        "B2",
        "Suffix B2 enforces response format two with unique marker.",
    )
    A_prompt = build_segment_ids(
        "A",
        "Prefix A (for R3) differs from A1/A2 while staying similar in length.",
    )
    A4_prompt = build_segment_ids(
        "A4",
        "Prefix A4 is new and appears after chunk in R4 only.",
    )
    B4_prompt = build_segment_ids(
        "B4",
        "Suffix B4 is new and appears in R4 only.",
    )
    A5_prompt = build_segment_ids(
        "A5",
        "Prefix A5 is new and used to compare prefix efficiency against blend.",
    )
    B5_prompt = build_segment_ids(
        "B5",
        "Suffix B5 is new and used together with A5 after shared chunk.",
    )

    def join_segments(segments: list[tuple[str, list[int]]]):
        prompt: list[int] = []
        ranges: list[tuple[str, int, int]] = []
        for i, (name, seg_ids) in enumerate(segments):
            if i > 0:
                prompt += sep_ids
            start = len(prompt)
            prompt += seg_ids
            end = len(prompt)
            ranges.append((name, start, end))
        return prompt, ranges

    # Target workflow:
    # R0: A0 + chunk0 + B0
    # R1: A1 + chunk + B1
    # R2: A2 + chunk + B2
    # R3: A + B2 + chunk
    # R4: chunk + A4 + B4
    # R5: chunk + A5 + B5
    r0_segments = [
        ("A0", A0_prompt),
        ("chunk0", chunk0_prompt),
        ("B0", B0_prompt),
    ]
    r1_segments = [
        ("A1", A1_prompt),
        ("chunk", shared_chunk_prompt),
        ("B1", B1_prompt),
    ]
    r2_segments = [
        ("A2", A2_prompt),
        ("chunk", shared_chunk_prompt),
        ("B2", B2_prompt),
    ]
    r3_segments = [("A", A_prompt), ("B2", B2_prompt), ("chunk", shared_chunk_prompt)]
    r4_segments = [
        ("chunk", shared_chunk_prompt),
        ("A4", A4_prompt),
        ("B4", B4_prompt),
    ]
    r5_segments = [
        ("chunk", shared_chunk_prompt),
        ("A5", A5_prompt),
        ("B5", B5_prompt),
    ]

    r0_prompt, r0_ranges = join_segments(r0_segments)
    r1_prompt, r1_ranges = join_segments(r1_segments)
    r2_prompt, r2_ranges = join_segments(r2_segments)
    r3_prompt, r3_ranges = join_segments(r3_segments)
    r4_prompt, r4_ranges = join_segments(r4_segments)
    r5_prompt, r5_ranges = join_segments(r5_segments)

    prompt_map = {
        "R0 (warmup: A0+chunk0+B0)": r0_prompt,
        "R1 (A1+chunk+B1)": r1_prompt,
        "R2 (A2+chunk+B2)": r2_prompt,
        "R3 (A+B2+chunk)": r3_prompt,
        "R4 (chunk+A4+B4)": r4_prompt,
        "R5 (chunk+A5+B5)": r5_prompt,
    }
    segment_map = {
        "R0 (warmup: A0+chunk0+B0)": r0_ranges,
        "R1 (A1+chunk+B1)": r1_ranges,
        "R2 (A2+chunk+B2)": r2_ranges,
        "R3 (A+B2+chunk)": r3_ranges,
        "R4 (chunk+A4+B4)": r4_ranges,
        "R5 (chunk+A5+B5)": r5_ranges,
    }
    return prompt_map, segment_map


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_environment_variables(
        use_disk=args.use_disk,
        blend_special_str=args.blend_special_str,
        enable_sparse=args.enable_sparse,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    prompt_map, segment_map = build_prompt_ids(
        tokenizer=tokenizer,
        blend_special_str=os.getenv("LMCACHE_BLEND_SPECIAL_STR", "# #"),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=1,
    )

    print(f"Using model: {model}")
    print(f"LMCACHE_CHUNK_SIZE={os.getenv('LMCACHE_CHUNK_SIZE')}")
    print(f"LMCACHE_ENABLE_BLENDING={os.getenv('LMCACHE_ENABLE_BLENDING')}")
    print(f"LMCACHE_BLEND_SPECIAL_STR={os.getenv('LMCACHE_BLEND_SPECIAL_STR')}")
    print(f"LMCACHE_BLEND_RECOMPUTE_RATIOS={os.getenv('LMCACHE_BLEND_RECOMPUTE_RATIOS')}")
    print(f"LMCACHE_USE_LAYERWISE={os.getenv('LMCACHE_USE_LAYERWISE')}")
    print(
        f"LMCACHE_ENABLE_CHUNK_STATISTICS={os.getenv('LMCACHE_ENABLE_CHUNK_STATISTICS')}"
    )
    print(
        "LMCACHE_CHUNK_STATISTICS_STRATEGY="
        f"{os.getenv('LMCACHE_CHUNK_STATISTICS_STRATEGY')}"
    )
    print(f"LMCACHE_EXTRA_CONFIG={os.getenv('LMCACHE_EXTRA_CONFIG')}")
    print()
    print("Request lengths:")
    for req_name, req_prompt in prompt_map.items():
        print(f"  {req_name}: {len(req_prompt)} tokens")
    print()

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        chunk_size = int(os.getenv("LMCACHE_CHUNK_SIZE", "256"))
        seen_hashes: set[str] = set()

        request_order = [
            "R0 (warmup: A0+chunk0+B0)",
            "R1 (A1+chunk+B1)",
            "R2 (A2+chunk+B2)",
            "R3 (A+B2+chunk)",
            "R4 (chunk+A4+B4)",
            "R5 (chunk+A5+B5)",
        ]

        for idx, req_name in enumerate(request_order):
            print_output(llm, prompt_map[req_name], sampling_params, req_name)
            print_chunk_reuse_report(
                req_name,
                prompt_map[req_name],
                segment_map[req_name],
                chunk_size,
                seen_hashes,
            )
            if idx < len(request_order) - 1:
                time.sleep(1)


if __name__ == "__main__":
    main()
