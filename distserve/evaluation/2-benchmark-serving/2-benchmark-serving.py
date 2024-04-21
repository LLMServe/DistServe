"""
The client during serving performance benchmarking
"""
import sys, os
import argparse
import time
import random
import asyncio
import requests
from typing import Optional
from tqdm.asyncio import tqdm

import numpy as np

# From https://stackoverflow.com/a/287944/16569836
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
from distserve.simulator.utils import TestRequest, Dataset, ReqResult, dump_req_result_list
from metrics import BenchmarkMetrics
import backends

async def run_some_requests(
    backend: str,
    host: str, port: int,
    requests: list[TestRequest],
    timestamps: list[float],
    output_descrip_str: str,
    verbose: bool
) -> list[ReqResult]:
    """
    Issue a bunch of requests on their corresponding timestamps, and return the ReqResults
    """

    pbar_args = {
        "ncols": 90,
        "smoothing": 0.05,
    }
    issued_pbar = tqdm(total=len(requests), desc="Iss", colour="#aaeebb", **pbar_args)
    first_token_generated_pbar = tqdm(total=len(requests), desc="TFT", colour="#ffffff", **pbar_args)
    finished_pbar = tqdm(total=len(requests), desc="Fin", colour="#66ccff", **pbar_args)
    
    outputs = []
    last_print_outputs_len = 0
    last_print_time = time.time()

    async def run_one_request(
        request: TestRequest,
        timestamp: float
    ) -> Optional[ReqResult]:
        """
        Issue one request on the given timestamp, and then return the ReqResult
        """
        await asyncio.sleep(timestamp)
        issued_pbar.update(1)
        issued_pbar.refresh()
        request_func = backends.BACKEND_TO_REQUEST_FUNCS[backend]
        output = await request_func(host, port, request, first_token_generated_pbar, finished_pbar, verbose)

        if output is None:
            return
        outputs.append(output)
        nonlocal last_print_outputs_len
        nonlocal last_print_time
        if len(outputs)-last_print_outputs_len > len(requests)*0.1 or \
            time.time() - last_print_time > 60:
            last_print_outputs_len = len(outputs)
            last_print_time = time.time()
            part_metrics = BenchmarkMetrics.from_req_results(outputs)
            print(f"\n\n\n{output_descrip_str}")
            print(f"TFT Gap: {issued_pbar.n - first_token_generated_pbar.n}")
            print(f"FIN Gap: {issued_pbar.n - finished_pbar.n}")
            print(part_metrics)
            issued_pbar.refresh()
            first_token_generated_pbar.refresh()
            finished_pbar.refresh()

    tasks = []
    for (request, timestamp) in zip(requests, timestamps):
        task = asyncio.create_task(run_one_request(request, timestamp))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
        
    outputs = [x for x in outputs if x is not None]
    issued_pbar.close()
    first_token_generated_pbar.close()
    finished_pbar.close()
    return outputs


def benchmark_serving(
    backend: str,
    host: str, port: int,
    dataset: Dataset,
    req_timestamps: list[float],
    num_prompts: int,
    request_rate: float,
    verbose: bool
) -> list[ReqResult]:
    """
    Perform online serving benchmark under the given num_prompts and request_rate
    """
    # Generate requests and timestamps
    if len(dataset.reqs) < num_prompts:
        print(f"Warning: dataset only has {len(dataset.reqs)} requests, but we are asked to process {num_prompts} prompts")
        while len(dataset.reqs) < num_prompts:
            dataset.reqs += dataset.reqs
    if len(req_timestamps) < num_prompts:
        print(f"Error: req_timestamps only has {len(req_timestamps)} requests, but we are asked to process {num_prompts} prompts")
        sys.exit(1)
    requests = dataset.reqs[:num_prompts]
    timestamps = np.array(req_timestamps[:num_prompts])
    # for req in requests:
    #     print(req.prompt_len)

    # Scale timestamps to [0, num_prompts/request_rate]
    timestamps -= timestamps[0]
    timestamps *= (num_prompts/request_rate) / timestamps[-1]
    timestamps = timestamps.tolist()

    output_descrip_str = f"{backend}, {dataset.dataset_name}, ({num_prompts}, {request_rate})"
    benchmark_result = asyncio.run(run_some_requests(backend, host, port, requests, timestamps, output_descrip_str, verbose))
    return benchmark_result


def generate_poisson_process(num_data_points: int = 40000, lam: float = 1) -> list[float]:
    """
    Generate a list of timestamps that follows a Poisson process
    """
    result = []
    t = 0
    for _ in range(num_data_points):
        t += np.random.exponential(1/lam)
        result.append(t)
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    port = args.port if args.port is not None else backends.BACKEND_TO_PORTS[backend]

    dataset = Dataset.load(args.dataset)
    random.shuffle(dataset.reqs)
    print(f"Loaded dataset {dataset.dataset_name} ({len(dataset.reqs)} requests)")

    if not args.uniform_distrib:
        req_timestamps = generate_poisson_process()
        print("Using Poisson distribution")
    else:
        req_timestamps = list(map(float, range(0, 100000)))
        print("Using uniform distribution")

    # Complete missing fields in args
    meta = requests.get(f"http://{args.host}:{port+1}").json()
    print(meta)
    if meta["backend"] != backend:
        print(f"{bcolors.FAIL}Error: The metadata server is running on backend {meta.backend}, but we are benchmarking on backend {backend}{bcolors.ENDC}")
        sys.exit(1)
    if args.exp_result_root == None:
        if "EXP_RESULT_ROOT" not in os.environ:
            print(f"{bcolors.FAIL}Error: EXP_RESULT_ROOT is not set in environment variables{bcolors.ENDC}")
            sys.exit(1)
        args.exp_result_root = os.getenv("EXP_RESULT_ROOT")
    if args.exp_result_dir == None:
        args.exp_result_dir = ('-'.join(meta["model"].split('/')[1:])) + '-' + dataset.dataset_name
    if args.exp_result_prefix == None:
        args.exp_result_prefix = backend
        
    num_prompts_and_request_rates = eval(args.num_prompts_req_rates)
    for (num_prompts, request_rate) in num_prompts_and_request_rates:
        print(f"{bcolors.OKGREEN}==============================================={bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Running on {num_prompts=} {request_rate=}{bcolors.ENDC}")
        result = benchmark_serving(
            backend,
            args.host, port,
            dataset,
            req_timestamps,
            num_prompts,
            request_rate,
            args.verbose
        )
        metrics = BenchmarkMetrics.from_req_results(result)
        print(metrics)
        if not args.dont_save:
            exp_result_filename = f"{args.exp_result_prefix}-{num_prompts}-{request_rate}"
            if args.uniform_distrib:
                exp_result_filename += "-uniform"
            exp_result_filename += ".exp"

            exp_result_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
            os.makedirs(exp_result_dir, exist_ok=True)
            exp_result_path = os.path.join(exp_result_dir, exp_result_filename)

            dump_req_result_list(result, exp_result_path)
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=list(backends.BACKEND_TO_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num-prompts-req-rates",
        type=str,
        required=True,
        help="[(num_prompts, request_rate), ...] where num_prompts is the number of prompts to process and request_rate is the number of requests per second.",
    )
    parser.add_argument(
        "--dont-save",
        action="store_true",
        help="If this flag is set, then we won't save the benchmark results to disk",
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
        default=None,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: <model_name>-<dataset.name>)"
    )
    parser.add_argument(
        "--exp-result-prefix",
        type=str,
        default=None,
        help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default: <backend>)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--uniform-distrib",
        action="store_true",
        help="Use uniform distribution instead of Poisson"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print prompts & completions"
    )

    args = parser.parse_args()

    main(args)