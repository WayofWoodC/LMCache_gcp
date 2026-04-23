# SPDX-License-Identifier: Apache-2.0
# Standard
import time

# Third Party
from lmcache_vllm.blend_adapter import (
    OfflineKVPreCompute,
    combine_input_prompt_chunks,
)
from lmcache_vllm.vllm import LLM, SamplingParams
import lmcache_vllm


# ============================================================
# 1. Load chunk texts
# ============================================================
context_files = ["chunk1.txt", "chunk2.txt"]
chunks = []

for context_file in context_files:
    with open(context_file, "r", encoding="utf-8") as fin:
        context = fin.read()
    chunks.append(context)


# ============================================================
# 2. Build model
# ============================================================
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    gpu_memory_utilization=0.7,
    tensor_parallel_size=1,
)

sampling_params_generation = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=30,
)


# ============================================================
# 3. Build prompt pieces
#    For Llama Instruct, using chat template is safer.
# ============================================================
tokenizer = llm.get_tokenizer()

system_text = "You are a helpful assistant."
prefix_text = "Here are two document chunks from the user:\n\n"
question_text = "\n\nQuestion: What does this document mainly talk about? Answer briefly."

# Use chat template for the final full request.
# We will still keep the chunk texts as separate pieces for blending.
messages = [
    {"role": "system", "content": system_text},
    {
        "role": "user",
        "content": prefix_text + chunks[0] + chunks[1] + question_text,
    },
]

full_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# ------------------------------------------------------------
# To preserve chunk-level structure for KV blending,
# we separately prepare:
#   [system/prefix part] + [chunk1] + [chunk2] + [question suffix]
#
# We derive prefix/suffix by splitting around chunk1+chunk2.
# ------------------------------------------------------------
joined_chunks = chunks[0] + chunks[1]
split_idx = full_prompt.find(joined_chunks)

if split_idx == -1:
    raise ValueError(
        "Could not locate concatenated chunks inside the final chat-formatted prompt. "
        "Please verify chunk contents and prompt construction."
    )

sys_prompt = full_prompt[:split_idx]
question = full_prompt[split_idx + len(joined_chunks):]


# ============================================================
# 4. Offline precompute KV for each chunk
# ============================================================
print("-------------- Pre-computing KV cache for chunks -------------------")
offline_precompute = OfflineKVPreCompute(llm)
for i, chunk in enumerate(chunks, start=1):
    offline_precompute.precompute_kv(chunk)
    print(f"Precomputed KV for chunk {i}, length={len(chunk)} chars")

time.sleep(3)
print("Running the real query here!")


# ============================================================
# 5. Real blended query
# ============================================================
user_prompt = [sys_prompt, chunks[0], chunks[1], question]
user_prompt = combine_input_prompt_chunks(user_prompt)

outputs = llm.generate(user_prompt, sampling_params_generation)

for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Newly generated text: {generated_text!r}")

    if hasattr(output, "metrics") and output.metrics is not None:
        if (
            hasattr(output.metrics, "first_token_time")
            and hasattr(output.metrics, "first_scheduled_time")
            and output.metrics.first_token_time is not None
            and output.metrics.first_scheduled_time is not None
        ):
            ttft = output.metrics.first_token_time - output.metrics.first_scheduled_time
            print(f"Time to first token: {ttft:.3f} seconds")


# ============================================================
# 6. Graceful exit
# ============================================================
lmcache_vllm.close_lmcache_engine()