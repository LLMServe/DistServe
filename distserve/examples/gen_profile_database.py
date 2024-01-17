"""
This script generates a profiling database for the Skip-join MLFQ scheduler.
"""
import argparse
import random
import time
import os

from distserve.llm import OfflineLLM
from distserve import SamplingParams
from distserve.profiling import (
    ProfilingDatabase,
    ProfilingResult,
    ParallelConfig,
    PromptConfig,
)
from distserve.profiling import bs_config, in_len_config


def main(args, llm):
    if not os.path.exists(args.file_path):
        pd = ProfilingDatabase(args.file_path, new_database=True)
    else:
        pd = ProfilingDatabase(args.file_path, new_database=False)
    for idx, input_len in enumerate(in_len_config):
        print(f"Profiling input length {input_len}")
        max_tokens = 16
        random.seed(args.seed)
        if args.beam_width == 1:
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )
        else:
            sampling_params = SamplingParams(
                temperature=1,
                top_p=1.0,
                max_tokens=max_tokens,
                ignore_eos=True,
                best_of=args.beam_width,
                n=args.beam_width,
                use_beam_search=True,
            )
        for idy, batch_size in enumerate(bs_config):
            print(f"---Profiling batch size {batch_size}")
            for _i in range(batch_size):
                token_ids = [random.randint(0, 50257) for _ in range(input_len)]
                llm.llm_engine.add_request(
                    prompt=None,
                    prompt_token_ids=token_ids,
                    sampling_params=sampling_params,
                )

            latency_list = []
            for _ in range(max_tokens):
                start_time = time.time()
                try:
                    llm.llm_engine.step()
                except Exception as e:
                    print("Exception encountered when running llm.llm_engine.step(): ", e)
                    break # FIXME(sunyh): just a hack
                end_time = time.time()
                latency_list.append(end_time - start_time)

            parallel_config = ParallelConfig(
                int(args.pipeline_parallel_size), int(args.tensor_parallel_size)
            )
            prompt_config = PromptConfig(
                batch_size,
                input_len,
                args.beam_width,
            )
            if args.model not in pd.results:
                pd.results[args.model] = ProfilingResult(
                    args.model,
                    {parallel_config: {prompt_config: latency_list}},
                )
            else:
                pd.get(args.model).add_result(
                    parallel_config, prompt_config, latency_list
                )

    pd.materialize()

    # sanity check
    print(
        pd.get(args.model).get_latency_list(
            pp=args.pipeline_parallel_size,
            tp=args.tensor_parallel_size,
            batch_size=1,
            beam_width=1,
            in_len=1,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile the LLM model inference.")

    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    parser.add_argument("--sched-policy", type=str, default="fcfs")
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048 * 2048)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--file-path", type=str, default="profile_database")
    parser.add_argument("--use-dummy-weights", action="store_true")

    args = parser.parse_args()

    llm = OfflineLLM(
        args.model,
        tokenizer=args.tokenizer,
        seed=args.seed,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        max_num_blocks_per_req=args.max_num_blocks_per_req,
        sched_policy=args.sched_policy,
        use_dummy_weights=args.use_dummy_weights,
        max_batch_size=bs_config[-1],
        max_tokens_per_batch=args.max_tokens_per_batch,
        profiling_file=args.file_path,
    )

    main(args, llm)
