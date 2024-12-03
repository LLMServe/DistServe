import argparse

from simdistserve.benchmarks.parallel_ratio_bisect import simulate_bisect_ratio_search
from simdistserve.constants import ModelTypes


def parse_args():
    parser = argparse.ArgumentParser("Simulate DistServe or vLLM to find the optimal configuration.")
    parser.add_argument("--prefill-tp", type=int, default=8,
                        help="Prefill TP num (default 8)")
    parser.add_argument("--prefill-pp", type=int, default=1,
                        help="Prefill PP num (default 1)")
    parser.add_argument("--decode-tp", type=int, default=8,
                        help="Decode TP num (default 8)")
    parser.add_argument("--decode-pp", type=int, default=1,
                        help="Decode PP num (default 1)")
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
    parser.add_argument("--kv-cache-mem-per-gpu", type=int, default=10,
                        help="KV cache memory per GPU in GB (default 10GB)")
    parser.add_argument("--kv-transfer-bw", type=int, default=80,
                        help="KV transfer bandwidth in Gbps (default 10Gbps)")
    parser.add_argument("--esp", type=float, default=0.25,
                        help="Stopping criteria: `high - low < esp` (default esp = 0.25)")
    parser.add_argument("--N", type=int, default=300,
                        help="Number of samples to simulate (default 1000)")
    parser.add_argument("--model-type", type=str, default="opt_13b",
                        help="Model type to simulate (opt_13b, opt_66b, opt_175b)")

    args = parser.parse_args()
    args.model_type = ModelTypes.model_str_to_object(args.model_type)
    return args


def find_best_config(config_to_best_per_gpu_rate, backend):
    best_config = None
    best_ngpu = float("inf")
    best_per_gpu_rate = 0
    num_gpu = 0
    for config, per_gpu_rate in config_to_best_per_gpu_rate.items():
        if backend == 'distserve':
            pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode = config
            num_gpu = pp_cross * (tp_prefill * pp_prefill + tp_decode * pp_decode)
        elif backend == 'vllm':
            tp, pp = config
            num_gpu = tp * pp

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

    if args.backend != "distserve":
        raise ValueError(f"Unsupported backend: {args.backend }")


    result = simulate_bisect_ratio_search(
        prefill_tp=args.prefill_tp,
        prefill_pp=args.prefill_pp,
        decode_tp=args.decode_tp,
        decode_pp=args.decode_pp,
        model_type=args.model_type,
        is_dist_high=args.is_high_affinity,
        backend=args.backend,
        attainment=(args.prefill_target, args.decode_target, args.prefill_percentage, args.decode_percentage),
        max_per_gpu_rate=args.max_per_gpu_rate,
        kv_cache_mem_per_gpu=args.kv_cache_mem_per_gpu,
        kv_transfer_bw=args.kv_transfer_bw,
        esp=args.esp,
        N=args.N,
    )
    best_config = None
    max_per_gpu_rate = 0
    for config, per_gpu_rate in result.items():
        if per_gpu_rate > max_per_gpu_rate:
            best_config = config
            max_per_gpu_rate = per_gpu_rate
    print(f"Best config: prefill_instance={best_config[0]}, decode_instance={best_config[1]}, per_gpu_rate={max_per_gpu_rate}")
