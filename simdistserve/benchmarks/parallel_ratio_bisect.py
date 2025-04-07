import os
import time
from multiprocessing import Process, Manager
from time import sleep

import pandas as pd
from tqdm import tqdm
from fractions import Fraction

from simdistserve.benchmarks.search_binary import run_binary_search
from simdistserve.benchmarks.search_configs import get_distserve_configs, get_vllm_config
from simdistserve.constants import ModelTypes

# Restrict runtime to <= 32 CPU core.
# RunPod encounters problem when using `os.cpu_count()`
# to query the number of CPUs
MAX_CPU_COUNT = min(os.cpu_count() - 2, 32)


def main(
    prefill_pp: int, prefill_tp: int,
    decode_pp: int, decode_tp: int,
    model_type: ModelTypes,
    is_dist_high: bool = True,
    backend: str = "distserve", attainment=(200, 100, 90, 90),
    max_prefill_instance=8, max_decode_instance=8,
    max_per_gpu_rate=5, 
    kv_cache_mem_per_gpu=54, kv_transfer_bw=80,
    esp=0.25, N=1000,
    max_cpu_count=MAX_CPU_COUNT,
):
    """
    :return result: dict that maps config to the best_per_gpu_rate (int)
    """
    configs = [(1, prefill_tp, prefill_pp, decode_tp, decode_pp)]
    
    if backend == "distserve":
        ratios = []
        for prefill_instance in range(1, max_prefill_instance + 1):
            for decode_instance in range(1, max_decode_instance + 1):
                frac = Fraction(prefill_instance, decode_instance)
                ratios.append((frac.numerator, frac.denominator))
        ratios = list(set(ratios))
    else:
        raise ValueError(f"Unsupported backend for ratio search: {backend}")

    processes = []
    # Add a multiproc shared dict
    with Manager() as manager:
        result = manager.dict()
        pbar = tqdm(enumerate(ratios), total=len(ratios))
        for pid, ratio in pbar:
            proc = Process(
                target=run_binary_search,
                args=(
                    model_type, configs[0],
                    backend, attainment,
                ),
                kwargs=dict(
                    kv_cache_mem_per_gpu=kv_cache_mem_per_gpu,
                    kv_transfer_bw=kv_transfer_bw,
                    max_per_gpu_rate=max_per_gpu_rate,
                    prefill_instance=ratio[0],
                    decode_instance=ratio[1],
                    pid=pid, esp=esp,
                    N=N, result=result,
                    ratio_search=True,
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


simulate_bisect_ratio_search = main

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
