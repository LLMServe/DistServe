#!/bin/bash

workload=opt_13b_sharegpt
now=$(date +"%Y%m%d_%H%M%S")
sh 01-run_abalation.sh $workload
mkdir -p arxiv-data/$workload/$now
mv figure result visual arxiv-data/$workload/$now