#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

echo "Starting vllm server... (for full evaluation, OPT-66B)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py --backend vllm --model facebook/opt-66b
