#!/bin/bash

workload=opt_175b_sharegpt
bash 01-run_abalation.sh $workload
mkdir -p visual figure
python 02-draw_rate_abalation.py --target "(4000, 100)"
python 03-draw_slo_abalation.py --target "(4000, 100)" --per_gpu_rate 0.1
python 04-draw_abalation_curve.py --rates "[0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125]"