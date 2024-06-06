#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/common.sh

echo "Starting distllm server... (for kick-the-tires)"
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py --backend distserve --model facebook/opt-125m
