#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

echo "Starting vllm client... (for full evaluation, OPT-66B, ShareGPT)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend vllm --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(100, 0.25), (100, 0.5), (100, 1), (100, 1.5), (100, 2), (100, 2.5), (100, 3)]" --exp-result-dir opt-66b-sharegpt

echo "Starting vllm client... (for full evaluation, OPT-66B, HumanEval)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend vllm --dataset /app/dataset/humaneval.ds --num-prompts-req-rates "[(200, 0.5), (200, 1.5), (200, 2.5), (200, 3.5), (200, 4.5), (200, 5.5), (200, 6.5), (200, 7.5), (200, 8.5)]" --exp-result-dir opt-66b-humaneval

echo "Starting vllm client... (for full evaluation, OPT-66B, LongBench)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend vllm --dataset /app/dataset/longbench.ds --num-prompts-req-rates "[(100, 0.25), (100, 0.5), (100, 0.75), (100, 1), (100, 1.25), (100, 1.5), (100, 1.75), (100, 2), (100, 2.25), (100, 2.5)]" --exp-result-dir opt-66b-longbench
