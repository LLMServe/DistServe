#!/bin/bash

workload=opt_13b_sharegpt
bash 01-run_abalation.sh $workload
# Draw figure
mkdir -p visual figure
python 02-draw_rate_abalation.py --target "(200, 100)"
python 03-draw_slo_abalation.py --target "(200, 100)" --per_gpu_rate 1
python 04-draw_abalation_curve.py --rates "[1,2,3,4,5]"