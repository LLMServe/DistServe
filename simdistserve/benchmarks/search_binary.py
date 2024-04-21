from simdistserve.benchmarks.simulate_dist import run_experiment, parse_args
from simdistserve.constants import ModelTypes

from contextlib import nullcontext


def run_binary_search(
    model_type: ModelTypes,
    # config
    # - distserve: (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode)
    # - vllm: (tp, pp)
    config,
    backend: str,
    containment_targets: '(prefill_target, decode_target, prefill_containment, decode_containment)',
    max_per_gpu_rate: int = 16,
    shared_lock=None,
    shared_best_goodput=None,
    shared_best_config=None,
    pid=0,
):
    if shared_lock is None:
        shared_lock = nullcontext()

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
    low = shared_best_goodput.value
    high = max_per_gpu_rate
    best_per_gpu_rate = 0
    fixed_args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--N', '1000',
        '--prefill-containment', prefill_containment,  # P90
        '--prefill-target', prefill_target,  # ms
        '--decode-containment', decode_containment,  # P90
        '--decode-target', decode_target,  # ms
        '--model', ModelTypes.formalize_model_name(model_type),
        '--workload', 'sharegpt',
        '--slas', '[]',
        '--slo-scales', '[1]',
    ]

    while (high - low) > 0.5:
        with shared_lock:
            print(pid, 'shared', config, shared_best_goodput.value, shared_best_config.value, (low, high))
            low = max(shared_best_goodput.value, low)
        # Run simulation
        this_rate = (low + high) / 2
        rate = this_rate * num_gpu
        args = [*fixed_args, *config_args, '--rate', rate, ]
        args = [str(i) for i in args]
        args = parse_args(args)
        try:
            is_prefill_contained, is_decode_contained, df = run_experiment(args)
        except Exception as e:
            return None

        # Update the range
        success = is_prefill_contained and is_decode_contained
        if not success:
            # The experiment not passing the attainment threshold
            high = this_rate
            continue
        print(pid, config, success, this_rate)

        # Experiment passed the attainment threshold
        low = this_rate
        best_per_gpu_rate = this_rate
        if best_per_gpu_rate < shared_best_goodput.value:
            low = shared_best_goodput.value
            print(pid, 'ignored', low, config)
            continue

        # Update the global config if detected the global is even better
        with shared_lock:
            print(pid, 'read', config, shared_best_goodput.value)
            if shared_best_goodput.value < best_per_gpu_rate:
                print(pid, 'update', best_per_gpu_rate, config)
                shared_best_config.value = config
                shared_best_goodput.value = best_per_gpu_rate
        pass

    print(pid, 'Finish', config, best_per_gpu_rate)
    return best_per_gpu_rate


if __name__ == '__main__':
    run_binary_search(
        ModelTypes.opt_13b,
        (1, 1, 1, 1, 1),
        "distserve",
        (200, 100, 90, 90),
        max_per_gpu_rate=16,
    )
