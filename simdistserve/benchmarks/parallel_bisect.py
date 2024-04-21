import os
from collections import deque
from multiprocessing import Process, Manager, Value, freeze_support

from tqdm import tqdm

from simdistserve.benchmarks.search_binary import run_binary_search
from simdistserve.benchmarks.search_configs import get_distserve_configs, get_vllm_config
from simdistserve.constants import ModelTypes


# get_distserve_configs(ModelTypes.opt_13b, 4, 8, False)
# get_vllm_config(ModelTypes.opt_13b, 32)

def main():
    configs = get_distserve_configs(ModelTypes.opt_13b, 4, 8, True)

    manager = Manager()
    max_cpu_count = os.cpu_count() - 2
    # add a multiprocess dict that makes the values returned
    # shared_dict = manager.dict()

    processes = deque([])
    for pid, config in tqdm(enumerate(configs), total=len(configs)):

        proc = Process(
            target=run_binary_search,
            args=(
                ModelTypes.opt_13b,
                config,
                "distserve",
                (200, 100, 90, 90),
            ),
            kwargs=dict(
                max_per_gpu_rate=5,
                pid=pid,
                esp=0.25,
            )
        )
        if len(processes) >= max_cpu_count:
            p = processes.popleft()
            p.join()

        proc.start()
        processes.append(proc)
        pass


if __name__ == '__main__':
    freeze_support()
    main()
    pass

# configs = get_distserve_configs(ModelTypes.opt_13b, 4, 8, True)
# print(configs)
