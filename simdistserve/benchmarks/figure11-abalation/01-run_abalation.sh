#!/bin/bash

# if model is opt_66b, then set these variables

workloads=$1

# See paper Table 1 and Figure 9 for the arguments
if [ $workloads == "opt_13b_sharegpt" ]; then
    total_gpu=4
    per_gpu_rate='[1,2,3,4,5]'
    base_N='[100]'
    model="opt_13b"
    prefill_target=200
    decode_target=100
    chosen_per_gpu_rate=2
elif [ $workloads == "opt_66b_sharegpt" ]; then
    total_gpu=8
    per_gpu_rate='[0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]'
    base_N='[25]'
    model="opt_66b"
    prefill_target=400
    decode_target=100
    chosen_per_gpu_rate=0.25
elif [ $workloads == "opt_175b_sharegpt" ]; then
    total_gpu=32
    per_gpu_rate='[0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125]'
    base_N='[25]'
    model="opt_175b"
    prefill_target=4000
    decode_target=200
    chosen_per_gpu_rate=0.25
elif [ $workloads == "opt_66b_humaneval" ]; then
    total_gpu=8
    per_gpu_rate='[0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875, 2.125]'
    base_N='[50]'
    model="opt_66b"
    prefill_target=125
    decode_target=200
elif [ $workloads == "opt_66b_longbench" ]; then
    total_gpu=8
    per_gpu_rate='[0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625]'
    base_N='[25]'
    model="opt_66b"
    prefill_target=1500
    decode_target=150
else
    echo "Usage: bash 01-run_abalation.sh [opt_13b_sharegpt|opt_66b_sharegpt|opt_175b_sharegpt|opt_66b_humaneval|opt_66b_longbench]"
    exit 1
fi



# Abalation study with DistServe and vLLM
rm -rf result || true
mkdir -p result
# Production code for abalation study
config_tpl="--model ${model} --prefill-target ${prefill_target} --decode-target ${decode_target} --tp-prefill {tp_prefill} --pp-prefill {pp_prefill} --tp-decode {tp_decode} --pp-decode {pp_decode}"
output_tpl_prod="--output {file_prefix}.latency.csv"
output_tpl_debug="--output {file_prefix}.latency.csv --output-request-info  {file_prefix}.request_info.csv --output-request-event {file_prefix}.request_event.csv --output-request-latency {file_prefix}.request_latency.csv --output-worker {file_prefix}.worker.csv"
exec_path=$(realpath ../simulate_dist.py)

# if env has ABALATION_DEBUG=1, then use debug output
if [ -z "$ABALATION_DEBUG" ]; then
    output_tpl=$output_tpl_prod
else
    output_tpl=$output_tpl_debug
    echo "Debug mode: printing all worker output"
fi


# Run experiment for DistServe
client_cmdline="python3 ${exec_path} --backend distserve --N {N} --workload {workload} --rate {rate} ${config_tpl} ${output_tpl}"
file_prefix='result/simulate-distserve-{workload}-n{N}-r{rate}-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}'
python ../simulate_multi.py \
--client-cmdline "$client_cmdline" \
--total-gpu $total_gpu \
--per-gpu-rate "$per_gpu_rate" --workload '["sharegpt"]' \
--tp-prefill "[1,2,4,8]" --pp-prefill "[1,2,4,8]" --tp-decode "[1,2,4,8]" --pp-decode "[1,2,4,8]" \
--file-prefix $file_prefix --base-N $base_N


# Run experiment for vLLM
# Note: `--xp-decode 0` is set becuase vLLM only use `--xp-prefill` as parallelism setting in the script.
client_cmdline="python3 ${exec_path} --backend vllm --N {N} --workload {workload} --rate {rate} ${config_tpl} ${output_tpl}"
file_prefix='result/simulate-vllm-{workload}-n{N}-r{rate}-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}'
python ../simulate_multi.py \
--client-cmdline "$client_cmdline" \
--total-gpu $total_gpu \
--per-gpu-rate "$per_gpu_rate" --workload '["sharegpt"]' \
--tp-prefill "[1,2,4,8]" --pp-prefill "[1,2,4,8]" --tp-decode "[0]" --pp-decode "[0]" \
--file-prefix $file_prefix --base-N $base_N



# Draw figure
mkdir -p visual figure
python 02-draw_rate_abalation.py --target "($prefill_target, $decode_target)"
python 03-draw_slo_abalation.py --target "($prefill_target, $decode_target)" --per_gpu_rate $chosen_per_gpu_rate
python 04-draw_abalation_curve.py