#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

echo "Starting distllm client... (for full evaluation, OPT-66B, ShareGPT)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(200, 0.5), (200, 1), (200, 2), (200, 3), (200, 4), (200, 5), (200, 6)]" --exp-result-dir opt-66b-sharegpt

echo "Starting distllm client... (for full evaluation, OPT-66B, HumanEval)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /app/dataset/humaneval.ds --num-prompts-req-rates "[(400, 1), (400, 3), (400, 5), (400, 7), (400, 9), (400, 11), (400, 13), (400, 15), (400, 17)]" --exp-result-dir opt-66b-humaneval

echo "Starting distllm client... (for full evaluation, OPT-66B, LongBench)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /app/dataset/longbench.ds --num-prompts-req-rates "[(200, 0.5), (200,  1), (200, 1.5), (200, 2), (200, 2.5), (200, 3), (200, 3.5), (200, 4), (200, 4.5), (200, 5)]" --exp-result-dir opt-66b-longbench
