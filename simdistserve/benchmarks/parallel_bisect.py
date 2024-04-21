import os
from collections import deque
from multiprocessing import Process, Manager, Value, freeze_support

from tqdm import tqdm

from simdistserve.benchmarks.search_binary import run_binary_search
from simdistserve.benchmarks.search_configs import get_distserve_configs, get_vllm_config
from simdistserve.constants import ModelTypes


# run_binary_search(
#     ModelTypes.opt_13b,
#     (1, 1, 1, 1, 1),
#     "distserve",
#     (200, 100, 90, 90),
#     max_per_gpu_rate=16,
# )


# get_distserve_configs(ModelTypes.opt_13b, 4, 8, False)
# get_vllm_config(ModelTypes.opt_13b, 32)

def main():
    configs = get_distserve_configs(ModelTypes.opt_13b, 4, 8, True)

    manager = Manager()
    # shared_lock = manager.Lock()
    shared_lock = None
    shared_best_goodput = manager.Value('best_goodput', 0)
    shared_best_config = manager.Value('best_config', None)
    max_cpu_count = os.cpu_count() - 2

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
                max_per_gpu_rate=16,
                shared_lock=shared_lock,
                shared_best_goodput=shared_best_goodput,
                shared_best_config=shared_best_config,
                pid=pid,
            )
        )
        if len(processes) >= max_cpu_count:
            p = processes.popleft()
            p.join()

        proc.start()
        processes.append(proc)
        pass

    print("Final result: ", shared_best_config.value, shared_best_goodput.value)


if __name__ == '__main__':
    freeze_support()
    main()
    pass

# configs = get_distserve_configs(ModelTypes.opt_13b, 4, 8, True)
# print(configs)
