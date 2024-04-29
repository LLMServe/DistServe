#!/bin/bash

# Abalation study with DistServe and vLLM with OPT-66B ShareGPT

total_gpu=32
per_gpu_rate='[0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]'
base_N='[25]'
model="opt_66b"
prefill_target=400
decode_target=100


# Clean up result directory
rm -rf result || true
mkdir -p result

# Production code for abalation study
config_tpl="--model ${model} --prefill-target ${prefill_target} --decode-target ${decode_target} --tp-prefill {tp_prefill} --pp-prefill {pp_prefill} --tp-decode {tp_decode} --pp-decode {pp_decode}"
output_tpl="--output {file_prefix}.latency.csv"
exec_path=$(realpath ../simulate_dist.py)

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
# Note: vLLM only use `--xp-prefill` as parallelism setting in the script.
client_cmdline="python3 ${exec_path} --backend vllm --N {N} --workload {workload} --rate {rate} ${config_tpl} ${output_tpl}"
file_prefix='result/simulate-vllm-{workload}-n{N}-r{rate}-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}'
python ../simulate_multi.py \
--client-cmdline "$client_cmdline" \
--total-gpu $total_gpu \
--per-gpu-rate "$per_gpu_rate" --workload '["sharegpt"]' \
--tp-prefill "[1,2,4,8]" --pp-prefill "[1,2,4,8]" --tp-decode "[0]" --pp-decode "[0]" \
--file-prefix $file_prefix --base-N $base_N


