import argparse
from distserve import OfflineLLM, SamplingParams
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='The model to use', default='facebook/opt-125m')
args = parser.parse_args()

# Sample prompts.
prompts = [
    "Life blooms like a flower. Far away or by the road. Waiting",
    "A quick brown fox",
    "Artificial intelligence is",
    "To be or not to be,",
    "one two three four"
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=64, stop=["\n"]
)

# Create an LLM for offline inference.
llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model,
        tokenizer=None
    ),
    disagg_parallel_config=DisaggParallelConfig(
        context=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        ),
        decoding=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
    ),
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=1024,
        gpu_memory_utilization=0.9,
        cpu_swap_space=1.0
    ),
    context_sched_config=ContextStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    ),
    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    )
)

# Generate texts from the prompts. The output is a list of Request objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# Print the outputs.
for prompt, step_outputs in zip(prompts, outputs):
    print(
        f"Prompt: {prompt!r}, Generated text: {''.join([step_output.new_token for step_output in step_outputs])} ({len(step_outputs)} tokens generated)."
    )
