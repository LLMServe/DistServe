#!/bin/bash

workload=opt_13b_sharegpt
sh 01-run_abalation.sh $workload
mkdir -p arxiv-data/$workload
cp -r figure result visual arxiv-data/$workload/