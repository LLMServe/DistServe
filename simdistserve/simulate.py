import argparse

from simdistserve.benchmarks.parallel_bisect import simulate_bisect_search
from simdistserve.constants import ModelTypes


def parse_args():
    parser = argparse.ArgumentParser("Simulate DistServe or vLLM to find the optimal configuration.")
    parser.add_argument("--ngpu-per-node", type=int, default=4)
    parser.add_argument("--num-node", type=int, default=1)
    parser.add_argument("--is-high-affinity", action="store_true")
    parser.add_argument("--backend", type=str, default="distserve",
                        help="Choose from: distserve, vllm")
    parser.add_argument("--workload", type=str, default="sharegpt",
                        help="Choose from: sharegpt, humaneval, longbench")
    parser.add_argument("--prefill-target", type=int, default=200,
                        help="Prefill TTFT attainment target in ms (default 200ms)")
    parser.add_argument("--decode-target", type=int, default=100,
                        help="Decode TPOT attainment target in ms (default 100ms)")
    parser.add_argument("--prefill-percentage", type=int, default=90,
                        help="Percentage of prefill target (default P90)")
    parser.add_argument("--decode-percentage", type=int, default=90,
                        help="Percentage of prefill target (default P90)")
    parser.add_argument("--max-per-gpu-rate", type=int, default=5,
                        help="Max per GPU rate to search (default 5)")
    parser.add_argument("--esp", type=float, default=0.25,
                        help="Stopping criteria: `high - low < esp` (default esp = 0.25)")
    parser.add_argument("--N", type=int, default=300,
                        help="Number of samples to simulate (default 1000)")
    parser.add_argument("--model-type", type=str, default="opt_13b",
                        help="Model type to simulate (opt_13b, opt_66b, opt_175b)")

    args = parser.parse_args()
    args.model_type = ModelTypes.model_str_to_object(args.model_type)
    return args


def find_best_config(config_to_best_per_gpu_rate):
    best_config = None
    best_ngpu = float("inf")
    best_per_gpu_rate = 0
    for config, per_gpu_rate in config_to_best_per_gpu_rate.items():
        pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode = config
        num_gpu = pp_cross * (tp_prefill * pp_prefill + tp_decode * pp_decode)
        if per_gpu_rate > best_per_gpu_rate or (per_gpu_rate == best_per_gpu_rate and num_gpu < best_ngpu):
            best_config = config
            best_per_gpu_rate = per_gpu_rate
            best_ngpu = num_gpu

    return best_config, best_per_gpu_rate


def check_dataset_env_var():
    import os
    if "DATASET" in os.environ:
        return
    raise KeyError(
        "Please set the environment variable `DATASET` to the path of the workload datasets. "
        "For user who started the environment with `DistServe-AE-GPU` docker image, "
        "simply do:\nexport DATASET=`/app/dataset`\n"
        "See the `repro-dataset.md` to prepare for workload dataset if you are using your custom environment."
    )


if __name__ == '__main__':
    # def main(num_node, num_gpu_per_node, is_dist_high: bool = True):
    args = parse_args()
    print(args)

    result = simulate_bisect_search(
        args.num_node,
        args.ngpu_per_node,
        model_type=args.model_type,
        is_dist_high=args.is_high_affinity,
        backend=args.backend,
        attainment=(args.prefill_target, args.decode_target, args.prefill_percentage, args.decode_percentage),
        max_per_gpu_rate=args.max_per_gpu_rate,
        esp=args.esp,
        N=args.N,
    )
    # print(result)
    best_config, best_per_gpu_rate = find_best_config(result)
    pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode = best_config
    print(f"Best per GPU rate: {best_per_gpu_rate:.2f}")
    print(f"Best config: pp_cross={pp_cross}, "
          f"tp_prefill={tp_prefill}, pp_prefill={pp_prefill}, "
          f"tp_decode={tp_decode}, pp_decode={pp_decode}")
