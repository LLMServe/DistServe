#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/common.sh

echo "Starting distllm client... (for kick-the-tires)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(10, 1)]" --exp-result-dir opt-125m-sharegpt --verbose
