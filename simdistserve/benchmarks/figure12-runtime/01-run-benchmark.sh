#!/bin/bash
N_GPUs=(2 4 8 16 32)


run_sharegpt_13b() {
  #  Run one sharegpt 13b workload and return the runtime
  exec_path=$(realpath ../simulate_dist.py)
  echo "foo"
}

# Simulate Dist-low

# Simulate Dist-high