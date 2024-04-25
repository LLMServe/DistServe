#!/bin/bash

workload=opt_66b_sharegpt
bash 01-run_abalation.sh $workload
# Draw figure
mkdir -p visual figure
python 02-draw_rate_abalation.py --target "(400, 100)"
python 03-draw_slo_abalation.py --target "(400, 100)" --per_gpu_rate 0.125
python 04-draw_abalation_curve.py --rates "[0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]"