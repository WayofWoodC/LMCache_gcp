# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
import argparse
import contextlib
import os
import time

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
    """
    Build token-id prompts with stable blending boundaries.

    Required structures:
    - R0: A0 + chunk0 + B0
    - R1: A1 + chunk + B1
    - R2: A2 + chunk + B2
    - R3: A + B + chunk
    - R4: chunk + A + B
    - R5: chunk + A5 + B5
    """

    sep_ids = tokenizer.encode(blend_special_str, add_special_tokens=False)

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def join_segments(*segments: list[int]) -> list[int]:
        prompt_ids = []
        for i, seg in enumerate(segments):
            if i > 0:
                prompt_ids += sep_ids
            prompt_ids += seg
        return prompt_ids

    # Main reusable chunk used by R1 / R2 / R3 / R4 / R5
    chunk_text = ("Hello, how are you? " * 500).strip()
    chunk_ids = enc(chunk_text)

    # Separate chunk only for R0
    chunk0_text = ("Nice to meet you. " * 500).strip()
    chunk0_ids = enc(chunk0_text)

    # A segments
    A0_text = (
        "This is prefix A0. You are a helpful assistant. Please answer the user's question briefly. "
    )
    A1_text = (
        "This is prefix A1.You are a helpful assistant. Please answer the user's question briefly. "
        
    )
    A2_text = (
        "You are a helpful assistant. Please answer the user's question briefly. This is prefix A2."
    )
    A_text = (
        "As a helpful assistant, this is prefix A."
    )
    A5_text = (
        "You are also a helpful assistant, and this is prefix A5."
    )

    # B segments
    B0_text = "This is suffix B0. Reply with one short continuation."
    B1_text = "This is suffix B1. Reply with one short continuation."
    B2_text = "This is suffix B2. Reply with one short continuation."
    B_text = "This is suffix B. Reply with one short continuation."
    B5_text = "This is suffix B5. Reply with one short continuation."

    A0_ids = enc(A0_text)
    A1_ids = enc(A1_text)
    A2_ids = enc(A2_text)
    A_ids = enc(A_text)
    A5_ids = enc(A5_text)

    B0_ids = enc(B0_text)
    B1_ids = enc(B1_text)
    B2_ids = enc(B2_text)
    B_ids = enc(B_text)
    B5_ids = enc(B5_text)

    prompts = {
        "R0": join_segments(A0_ids, chunk0_ids, B0_ids),
        "R1": join_segments(A1_ids, chunk_ids, B1_ids),
        "R2": join_segments(A2_ids, chunk_ids, B2_ids),
        "R3": join_segments(A_ids, B_ids, chunk_ids),
        "R4": join_segments(chunk_ids, A_ids, B_ids),
        "R5": join_segments(chunk_ids, A5_ids, B5_ids),
    }

    return prompts


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

    prompts = build_prompt_ids(
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
    print()

    for name, prompt in prompts.items():
        print(f"{name}: {len(prompt)} tokens")
    print()

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        request_order = ["R0", "R1", "R2", "R3", "R4", "R5"]

        for i, req_name in enumerate(request_order):
            print_output(llm, prompts[req_name], sampling_params, req_name)
            if i < len(request_order) - 1:
                time.sleep(1)


if __name__ == "__main__":
    main()