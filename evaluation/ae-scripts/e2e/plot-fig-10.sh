#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

echo "Plotting figure 10..."
python3 /app/distserve/distserve/evaluation/ae-scripts/e2e/_plot.py 10
echo "Done. Please check the figures in the /workspace/plots directory."
