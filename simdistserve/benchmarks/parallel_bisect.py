import os
import time
from multiprocessing import Process, Manager
from time import sleep

import pandas as pd
from tqdm import tqdm

from simdistserve.benchmarks.search_binary import run_binary_search
from simdistserve.benchmarks.search_configs import get_distserve_configs, get_vllm_config
from simdistserve.constants import ModelTypes

# Restrict runtime to <= 32 CPU core.
# RunPod encounters problem when using `os.cpu_count()`
# to query the number of CPUs
MAX_CPU_COUNT = min(os.cpu_count() - 2, 32)


def main(
    num_node: int, num_gpu_per_node: int, model_type: ModelTypes,
    is_dist_high: bool = True,
    backend: str = "distserve", attainment=(200, 100, 90, 90),
    max_per_gpu_rate=5, esp=0.25, N=1000,
    max_cpu_count=MAX_CPU_COUNT,
):
    """
    :return result: dict that maps config to the best_per_gpu_rate (int)
    """
    if backend == "distserve":
        configs = get_distserve_configs(
            model_type, num_node, num_gpu_per_node, is_dist_high
        )
    elif backend == "vllm":
        configs = get_vllm_config(
            model_type, num_node * num_gpu_per_node
        )

    processes = []
    # Add a multiproc shared dict
    with Manager() as manager:
        result = manager.dict()
        pbar = tqdm(enumerate(configs), total=len(configs))
        for pid, config in pbar:
            proc = Process(
                target=run_binary_search,
                args=(
                    model_type, config,
                    backend, attainment,
                ),
                kwargs=dict(
                    max_per_gpu_rate=max_per_gpu_rate,
                    pid=pid, esp=esp,
                    N=N, result=result,
                    debug=True,
                )
            )
            if len(processes) >= max_cpu_count:
                # Pop a process that has finished running
                found = False
                while not found:
                    for i in range(len(processes)):
                        if not processes[i].is_alive():
                            processes[i].join()
                            processes.pop(i)
                            found = True
                            pbar.update(1)
                            break
                    sleep(0.2)

            proc.start()
            processes.append(proc)
            pass
        for proc in processes:
            pbar.update(1)
            proc.join()
        result = dict(result)
        return result


simulate_bisect_search = main

if __name__ == '__main__':
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
        main(num_node, ngpu_per_node, is_dist_high=False)
        end = time.perf_counter()
        duration = end - start
        data.append({
            "name": "DistLow",
            "ngpu": ngpu,
            "duration": duration
        })
        print(f"DistLow({ngpu_per_node=},{num_node=}):{duration}s")

    df = pd.DataFrame(data)
    df.to_csv("parallel_bisect.csv", index=False)

    pass
