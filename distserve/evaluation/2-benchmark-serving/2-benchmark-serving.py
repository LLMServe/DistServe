"""Benchmark online serving throughput.
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional
import os
import sys

import aiohttp
import numpy as np
from tqdm import tqdm

from structs import TestRequest, Dataset, RequestResult
from backends import BACKEND_TO_PORTS

pbar: Optional[tqdm] = None

def sample_requests(dataset_path: str, num_prompts: int) -> List[TestRequest]:
    """
    sample_requests: Sample the given number of requests from the dataset.
    """
    dataset = Dataset.load(dataset_path)
    if num_prompts > len(dataset.reqs):
        raise ValueError(
            f"Number of prompts ({num_prompts}) is larger than the dataset size ({len(dataset.reqs)})."
        )
    return random.sample(dataset.reqs, num_prompts)


async def get_request(
    input_requests: List[TestRequest],
    process_name: str = "possion",
    request_rate: float = 1.0,
    cv: float = 1.0,
) -> AsyncGenerator[TestRequest, None]:
    interval_lens = len(input_requests)
    input_requests = iter(input_requests)

    if request_rate not in [float("inf"), 0.0]:
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(interval_lens)]
        elif process_name == "gamma":
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        elif process_name == "possion":
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        else:
            raise ValueError(
                f"Unsupported prosess name: {process_name}, we currently support uniform, gamma and possion."
            )
    for idx, request in enumerate(input_requests):
        yield request
        if request_rate == float("inf") or request_rate == 0.0:
            continue

        interval = intervals[idx]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> RequestResult:
    global pbar
    if backend == "deepspeed":
        payload = {
            "prompt": prompt,
            "max_tokens": output_len,
            "min_new_tokens": output_len,
            "max_new_tokens": output_len,
            "stream": True,
            "max_length": int((prompt_len + output_len)*1.2+10) # *1.2 to prevent tokenization error
        }
        
        request_start_time = time.perf_counter()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3*3600)) as session:
            token_timestamps = []
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for data in response.content.iter_any():
                            token_timestamps.append(time.perf_counter())
                        complete_time = time.perf_counter()
                    else:
                        print(response)
                        print(response.status)
                        print(response.reason)
                        sys.exit(1)
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
                print(e)
                sys.exit(1)
        request_end_time = time.perf_counter()
        
        pbar.update(1)
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=token_timestamps,
            lifetime_events=None
        )
    else:
        headers = {"User-Agent": "Benchmark Client"}
        if backend == "distserve" or backend == "vllm":
            pload = {
                "prompt": prompt,
                "n": 1,
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "temperature": 0.0 if use_beam_search else 1.0,
                "top_p": 1.0,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": False,
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # The maximum length of the input is 2048, limited by the embedding
        # table size.
        assert prompt_len+output_len < 2048
        
        request_start_time = time.perf_counter()
        request_output = None

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(api_url, headers=headers, json=pload) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                try:
                    output = json.loads(output)
                except:
                    print("Failed to parse the response:")
                    print(output)
                    continue

                # Re-send the request if it failed.
                if "error" not in output:
                    request_output = output
                    break
                else:
                    print(f"Failed to process the request: {output['error']}")
                    print(f"Resending the request: {pload}")

        request_end_time = time.perf_counter()
        
        pbar.update(1)        
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=request_output["timestamps"],
            lifetime_events=request_output.get("lifetime_events", None)
        )


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[TestRequest],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    request_cv: float = 1.0,
    process_name: str = "possion",
) -> List[RequestResult]:
    tasks: List[asyncio.Task] = []
    async for request in get_request(
        input_requests, process_name, request_rate, request_cv
    ):
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                request.prompt,
                request.prompt_len,
                request.output_len,
                best_of,
                use_beam_search,
            )
        )
        tasks.append(task)
    request_results = await asyncio.gather(*tasks)
    return request_results


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    input_requests = sample_requests(
        args.dataset, args.num_prompts
    )
    print("Sampling done. Start benchmarking...")

    global pbar
    pbar = tqdm(total=args.num_prompts)
    benchmark_start_time = time.time()
    request_results = asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            args.request_cv,
            args.process_name,
        )
    )
    benchmark_end_time = time.time()
    pbar.close()
    
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput:")
    print(f"\t{args.num_prompts / benchmark_time:.2f} requests/s")
    print(f"\t{sum([req.prompt_len + req.output_len for req in input_requests]) / benchmark_time:.2f} tokens/s")
    print(f"\t{sum([req.output_len for req in input_requests]) / benchmark_time:.2f} output tokens/s")

    with open(args.output, "w") as f:
        json.dump(request_results, f, default=vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend", type=str, default="distserve", choices=["distserve", "vllm", "deepspeed"]
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the (preprocessed) dataset."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts-req-rates", type=str, required=True,
        help="[(num_prompts, request_rate), ...] where num_prompts is the number of prompts to process and request_rate is the number of requests per second.",
    )
    parser.add_argument(
        "--request-cv",
        type=float,
        default=1.0,
        help="the coefficient of variation of the gap between" "the requests.",
    )
    parser.add_argument(
        "--process-name",
        type=str,
        default="possion",
        choices=["possion", "gamma", "uniform"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    
    parser.add_argument(
        "--exp-result-root",
        type=str,
        default=None,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: env var EXP_RESULT_ROOT)"
    )
    parser.add_argument(
        "--exp-result-dir",
        type=str,
        required=True,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: <model_name>-<dataset.name>)"
    )
    parser.add_argument(
        "--exp-result-prefix",
        type=str,
        default=None,
        help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default: <backend>)"
    )
    
    args = parser.parse_args()
    
    if args.exp_result_root == None:
        if "EXP_RESULT_ROOT" not in os.environ:
            print(f"Error: EXP_RESULT_ROOT is not set in environment variables")
            sys.exit(1)
        args.exp_result_root = os.getenv("EXP_RESULT_ROOT")
        
    if args.exp_result_prefix == None:
        args.exp_result_prefix = args.backend
        
    if args.port == None:
        args.port = BACKEND_TO_PORTS[args.backend]
        
    num_prompts_request_rates = eval(args.num_prompts_req_rates)
    for (num_prompts, request_rate) in num_prompts_request_rates:
        print("===================================================================")
        print(f"Running with num_prompts={num_prompts}, request_rate={request_rate}")
        args.num_prompts = num_prompts
        args.request_rate = request_rate
        output_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{args.exp_result_prefix}-{num_prompts}-{request_rate}.exp")
        main(args)
        time.sleep(1)
        