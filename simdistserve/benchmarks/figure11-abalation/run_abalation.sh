#!/bin/bash

cd "$(dirname "$0")"

# Run abalation study
bash 01-run_abalation.sh

sleep 10 # in case data frames have not written to disk (in some file systems)
echo "All task complete. Now draw figures"
# Draw figure
mkdir -p visual figure
python 02-draw_rate_abalation.py --target "(400, 100)"
python 03-draw_slo_abalation.py --target "(400, 100)" --per_gpu_rate 0.375
python 04-draw_abalation_curve.py --rates "[0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]"

cp figure/abalation.png /workspace/