import os
import time
from multiprocessing import Process
from time import sleep

import pandas as pd
from tqdm import tqdm

from simdistserve.benchmarks.search_binary import run_binary_search
from simdistserve.benchmarks.search_configs import get_distserve_configs
from simdistserve.constants import ModelTypes


def main(num_node, num_gpu_per_node, is_dist_high: bool = True, run_parallel: bool = True):
    configs = get_distserve_configs(
        ModelTypes.opt_13b, num_node, num_gpu_per_node, is_dist_high
    )

    max_cpu_count = os.cpu_count() - 2

    processes = []
    for pid, config in tqdm(enumerate(configs), total=len(configs)):

        args = (
            ModelTypes.opt_13b,
            config,
            "distserve",
            (200, 100, 90, 90),
        )
        kwargs = dict(max_per_gpu_rate=5, pid=pid, esp=0.25, N=300, )

        if not run_parallel:
            r = run_binary_search(*args, **kwargs)
            print(f"config={config}, result={r}")
            continue

        proc = Process(
            target=run_binary_search,
            args=args,
            kwargs=kwargs
        )
        if len(processes) >= max_cpu_count:
            # Find one process that has ended
            found = False
            while not found:
                for i in range(len(processes)):
                    if not processes[i].is_alive():
                        processes[i].join()
                        processes.pop(i)
                        found = True
                        break
                sleep(0.2)

        proc.start()
        processes.append(proc)
        pass


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_parallel = not args.debug

    data = []
    for ngpu in [2, 4, 8, 16, 32]:
        start = time.perf_counter()
        main(ngpu, 1, is_dist_high=True)
        end = time.perf_counter()
        duration = end - start
        data.append({
            "name": "DistHigh",
            "ngpu": ngpu,
            "duration": duration
        })
        print(f"DistHigh({ngpu=}):{duration}s")

    for ngpu_per_node, num_node in [(2, 1), (4, 1), (8, 1), (8, 2), (8, 4)]:
        ngpu = ngpu_per_node * num_node
        start = time.perf_counter()
        main(num_node, ngpu_per_node, is_dist_high=False, run_parallel=run_parallel)
        end = time.perf_counter()
        duration = end - start
        data.append({
            "name": "DistLow",
            "ngpu": ngpu,
            "duration": duration
        })
        print(f"DistLow({ngpu_per_node=},{num_node=}):{duration}s")

    df = pd.DataFrame(data)
    df.to_csv("runtime_result.csv", index=False)

    pass
