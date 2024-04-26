import time

from simdistserve.benchmarks.simulate_dist import run_experiment, parse_args
from simdistserve.constants import ModelTypes


def run_binary_search(
    model_type: ModelTypes,
    # config
    # - distserve: (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode)
    # - vllm: (tp, pp)
    config,
    backend: str,
    containment_targets: '(prefill_target, decode_target, prefill_containment, decode_containment)',
    max_per_gpu_rate: int = 16,
    pid=0,
    esp=0.5,
    N=1000,
    debug=False,
):
    N = str(N)

    #
    # make config args
    #
    if backend == 'distserve':
        (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode) = config
        num_gpu = pp_cross * (tp_prefill * pp_prefill + tp_decode * pp_decode)
        config_args = [
            '--tp-prefill', f'{pp_cross * tp_prefill}',
            '--pp-prefill', f'{pp_cross * pp_prefill}',
            '--tp-decode', f'{tp_decode}',
            '--pp-decode', f'{pp_decode}',
        ]
    else:
        (tp, pp) = config
        num_gpu = tp * pp
        config_args = [
            '--tp-prefill', f'{tp}',
            '--pp-prefill', f'{pp}',
            '--tp-decode', f'{tp}',
            '--pp-decode', f'{tp}',
        ]

    prefill_target, decode_target, prefill_containment, decode_containment = containment_targets

    #
    # bisect the integer range between 1 <= per_gpu_rate <= max_per_gpu_rate
    #
    low = 0
    high = max_per_gpu_rate
    best_per_gpu_rate = 0

    fixed_args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--N', N,
        '--prefill-containment', prefill_containment,  # P90
        '--prefill-target', prefill_target,  # ms
        '--decode-containment', decode_containment,  # P90
        '--decode-target', decode_target,  # ms
        '--model', ModelTypes.formalize_model_name(model_type),
        '--workload', 'sharegpt',
        '--slas', '[]',
        '--slo-scales', '[1]',
    ]

    time_durations = []
    while (high - low) > esp:
        # Run simulation
        this_rate = (low + high) / 2
        rate = this_rate * num_gpu
        args = [*fixed_args, *config_args, '--rate', rate, ]
        args = [str(i) for i in args]
        args = parse_args(args)
        try:
            start_time = time.time()
            is_prefill_contained, is_decode_contained, df = run_experiment(args)
            end_time = time.time()
            time_durations.append((config, this_rate, end_time - start_time))
        except Exception as e:
            if debug:
                import traceback
                traceback.print_exc()
            return None

        # Update the range
        success = is_prefill_contained and is_decode_contained
        if not success:
            # The experiment not passing the attainment threshold
            high = this_rate
            continue

        # Experiment passed the attainment threshold
        low = this_rate
        best_per_gpu_rate = this_rate
        pass
    return best_per_gpu_rate


if __name__ == '__main__':
    args = parse_args()
    run_binary_search(
        ModelTypes.opt_13b,
        (1, 1, 1, 1, 1),
        "distserve",
        (200, 100, 90, 90),
        max_per_gpu_rate=16,
        is_debug=args.debug,
    )
