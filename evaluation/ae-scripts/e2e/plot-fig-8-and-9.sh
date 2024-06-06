#!/bin/bash
. /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

echo "Plotting figure 8..."
python3 /app/distserve/distserve/evaluation/ae-scripts/e2e/_plot.py 8
echo "Plotting figure 9..."
python3 /app/distserve/distserve/evaluation/ae-scripts/e2e/_plot.py 9
echo "Done. Please check the figures in the /workspace/plots directory."
