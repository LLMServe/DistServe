#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

echo "Starting vllm client... (for full evaluation, OPT-13B)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend vllm --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(100, 0.25), (100, 0.5), (100, 1), (100, 1.5), (100, 2), (100, 2.25), (100, 2.5), (100, 3)]" --exp-result-dir opt-13b-sharegpt
