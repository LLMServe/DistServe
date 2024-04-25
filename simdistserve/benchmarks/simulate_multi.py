import argparse
import os
import shlex
import time
from itertools import product
from pathlib import Path
from subprocess import Popen
from typing import List

import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-cmdline", type=str, required=True,
        help="Command line to run the client. "
             "e.g. python3 simulate_dist.py --backend distserve --N {N} "
             "--workload {dataset} "
             "--rate {rate} "
             "--tp-prefill {tp_prefill} --pp-prefill {pp_prefill} "
             "--tp-decode {tp_decode} --pp-decode {pp_decode} "
             "--output {file_prefix}.latency.csv "
             "--output-request-info {file_prefix}.request_info.csv "
             "--output-request-event {file_prefix}.request_event.csv "
             "--output-request-latency {file_prefix}.request_latency.csv "
             "--output-worker {file_prefix}.worker.csv "
    )
    # "--file_prefix {file_prefix} "
    parser.add_argument(
        "--file-prefix", type=str, default='',
    )
    # --N
    parser.add_argument(
        '--N', type=str, default=None,
        help="List of number of prompts to run the experiment at, e.g. [1000]"
    )
    # --base-N
    parser.add_argument(
        '--base-N', type=str, default=None,
        help="List of number base number of prompts to run the experiment. "
             "N = base_N * rate"
    )
    # --dataset
    parser.add_argument(
        '--workload', type=str, default="['sharegpt']",
        help="List of dataset to run the experiment at, e.g. ['sharegpt', 'longbench', 'humaneval']"
    )
    # --per-gpu-rate, that eventually goes into --rate
    parser.add_argument(
        '--per-gpu-rate', type=str, default="[1, 2, 3, 4, 5]",
        help="List of request rate per GPU to run the experiment at, e.g. [1, 2, 3, 4, 5]. "
             "This will make into the `rate` parameter when running actual experiment. "
    )
    # --tp-prefill
    parser.add_argument(
        '--tp-prefill', type=str, default="[1]",
        help="List of TP for prefill to run the experiment at, e.g. [1]"
    )
    # --pp-prefill
    parser.add_argument(
        '--pp-prefill', type=str, default="[1]",
        help="List of PP for prefill to run the experiment at, e.g. [1]"
    )
    # --tp-decode
    parser.add_argument(
        '--tp-decode', type=str, default="[1]",
        help="List of TP for decode to run the experiment at, e.g. [1]"
    )
    # --pp-decode
    parser.add_argument(
        '--pp-decode', type=str, default="[1]",
        help="List of PP for decode to run the experiment at, e.g. [1]"
    )
    # --total-gpu
    parser.add_argument(
        '--total-gpu', type=int, default=32,
        help="Total number of GPUs available"
    )
    args = parser.parse_args()

    # Compose the actual running arguments
    template_cmdline = args.client_cmdline
    workloads: List[str] = eval(args.workload)
    Ns: List[int] = eval(args.N) if args.N else []
    base_Ns: List[int] = eval(args.base_N)
    per_gpu_rates: List[float] = eval(args.per_gpu_rate)
    tp_prefills: List[int] = eval(args.tp_prefill)
    pp_prefills: List[int] = eval(args.pp_prefill)
    tp_decodes: List[int] = eval(args.tp_decode)
    pp_decodes: List[int] = eval(args.pp_decode)
    file_prefix = args.file_prefix
    total_gpus = args.total_gpu

    # Check the dataset directory exist
    dataset_root = os.environ.get('DATASET', '/app/dataset')
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root directory {dataset_root} does not exist. "
                         f"Please set the environment variable `DATASET` to the directory with data.")

    # Produce the actual cmds
    cmds = []
    for workload in workloads:
        for base_N in base_Ns:
            # for N in Ns: # TODO: Use `Ns` or `base_Ns` - determine here.
            for tp_prefill, pp_prefill, tp_decode, pp_decode in product(
                tp_prefills, pp_prefills, tp_decodes, pp_decodes
            ):
                ngpu = tp_prefill * pp_prefill + tp_decode * pp_decode
                if ngpu > total_gpus:
                    continue

                # TODO: Employ binary search if necessary
                for per_gpu_rate in per_gpu_rates:
                    rate = per_gpu_rate * ngpu
                    N = base_N * ngpu
                    format_vals = dict(
                        N=N,
                        workload=workload,
                        rate=rate,
                        tp_prefill=tp_prefill,
                        pp_prefill=pp_prefill,
                        tp_decode=tp_decode,
                        pp_decode=pp_decode,
                    )
                    file_prefix_template = args.file_prefix
                    file_prefix = file_prefix_template.format(**format_vals)
                    cur_cmdline = template_cmdline.format(**format_vals, file_prefix=file_prefix)
                    cmds.append((cur_cmdline, file_prefix))
                    pass

    # Run and wait for all subprocesses to finish
    procs = []
    max_concurrent_procs = os.cpu_count() - 1
    for cmd, file_prefix in tqdm.tqdm(cmds):
        if len(procs) >= max_concurrent_procs:
            while True:
                found = False
                for p in procs:
                    if p.poll() is not None:
                        procs.remove(p)
                        found = True
                        break
                if found:
                    break
                time.sleep(0.1)

        fout = open(f"{file_prefix}.log", 'w')
        ferr = open(f"{file_prefix}.err", 'w')
        p = Popen(
            shlex.split(cmd),
            stdout=fout,
            stderr=ferr,
        )
        procs.append(p)
        pass

    # Done
    print("All done!")

    pass
