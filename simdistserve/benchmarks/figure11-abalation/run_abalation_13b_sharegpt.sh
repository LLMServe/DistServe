#!/bin/bash

workload=opt_13b_sharegpt
now=$(date +"%Y%m%d_%H%M%S")
bash 01-run_abalation.sh $workload
mkdir -p arxiv-data/$workload/$now
cp -r figure result visual arxiv-data/$workload/$now