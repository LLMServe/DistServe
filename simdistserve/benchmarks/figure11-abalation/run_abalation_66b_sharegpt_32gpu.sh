#!/bin/bash

workload=opt_66b_sharegpt_32gpu
now=$(date +"%Y%m%d_%H%M%S")
sh 01-run_abalation.sh $workload
mkdir -p arxiv-data/$workload/$now
cp -r figure result visual arxiv-data/$workload/$now