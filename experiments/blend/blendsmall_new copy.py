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
        enable_prefix_caching=True,  # true of false
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
    # Keep pure tokenized text; do not use Mistral-specific hardcoded token IDs.
    # For this experiment, we only need stable long token sequences for KV reuse,
    # not chat-template-perfect prompting.

    sep_ids = tokenizer.encode(blend_special_str, add_special_tokens=False)

    system_text = (
        "You are a helpful assistant. "
        "Please answer the user's question briefly."
    )
    sys_prompt = tokenizer.encode(system_text, add_special_tokens=False)

    chunk1_text = ("Hello, how are you? " * 500).strip()
    chunk2_text = ("Hello, what's up? " * 500).strip()
    chunk3_text = ("Hi, what are you up to? " * 500).strip()
    chunk4_text = ("Hello, how is it going? " * 500).strip()
    chunk5_text = ("Hi, nice to meet you! " * 500).strip()

    chunk1_prompt = tokenizer.encode(chunk1_text, add_special_tokens=False)
    chunk2_prompt = tokenizer.encode(chunk2_text, add_special_tokens=False)
    chunk3_prompt = tokenizer.encode(chunk3_text, add_special_tokens=False)
    chunk4_prompt = tokenizer.encode(chunk4_text, add_special_tokens=False)
    chunk5_prompt = tokenizer.encode(chunk5_text, add_special_tokens=False)

    warmup_text = ("Nice to meet you. " * 500).strip()
    warmup_prompt = tokenizer.encode(warmup_text, add_special_tokens=False)

    tail1 = tokenizer.encode("Hello, my name is", add_special_tokens=False)
    tail2 = tokenizer.encode("Hello, how are you?", add_special_tokens=False)
    tail3 = tokenizer.encode("Hello, what's up?", add_special_tokens=False)
    tail4 = tokenizer.encode("Hi, how you doing?", add_special_tokens=False)
    
    first_prompt = (
        sys_prompt
        + sep_ids
        + chunk1_prompt
        + sep_ids
        + chunk2_prompt
        + sep_ids
        + chunk3_prompt
        + sep_ids
        + tail1
    )

    second_prompt = (
        sys_prompt
        + sep_ids
        + chunk2_prompt
        + sep_ids
        + chunk1_prompt
        + sep_ids
        + chunk3_prompt
        + sep_ids
        + tail2
    )

    third_prompt = (
        sys_prompt
        + sep_ids
        + chunk2_prompt
        + sep_ids
        + chunk1_prompt
        + sep_ids
        + chunk3_prompt
        + sep_ids
        + tail3
    )

    fourth_prompt = (
        sys_prompt
        + sep_ids
        + chunk2_prompt
        + sep_ids
        + chunk1_prompt
        + sep_ids
        + chunk4_prompt
        + sep_ids
        + tail4
    )

    fifth_prompt = (
        sys_prompt
        + sep_ids
        + chunk1_prompt
        + sep_ids
        + chunk2_prompt
        + sep_ids
        + chunk5_prompt
        + sep_ids
        + tail4
    )

    return warmup_prompt, first_prompt, second_prompt, third_prompt, fourth_prompt, fifth_prompt


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

    warmup_prompt, first_prompt, second_prompt, third_prompt, fourth_prompt, fifth_prompt = build_prompt_ids(
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

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        print_output(llm, warmup_prompt, sampling_params, "warmup")

        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        print_output(
            llm,
            second_prompt,
            sampling_params,
            "second (warming up blend code path)",
        )

        time.sleep(1)

        print_output(llm, third_prompt, sampling_params, "third")
        time.sleep(1)
        print_output(llm, fourth_prompt, sampling_params, "fourth")
        time.sleep(1)
        print_output(llm, fifth_prompt, sampling_params, "fifth")

if __name__ == "__main__":
    main()