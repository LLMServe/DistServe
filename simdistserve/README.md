# DistServe Simulator

Simulator that finds the optimal paralleism strategy for DistServe and vLLM.

Given a specific workload, number of nodes, and GPU per node, the simulator finds the optimal parallelism strategy that
maximize the goodput of the workload.

## Quick Start

Find the optimal parallelism configuration for OPT-13B with ShareGPT workload in one node 4xA100 80G SXM:

### Dataset Preparation

To reproduce the dataset, please follow [this instruction](../../distserve/evaluation/docs/repro-dataset.md).
For user who started the environment with `DistLLM-AE-GPU` docker image,
we've preprocessed the datasets under `/app/dataset`.

Set your environment variable `DATASET` to the folder with preprocessed ShareGPT dataset.

```bash
export DATASET=/path/to/dataset
#export DATASET=/app/dataset
```

### Run experiment

```bash
python -m simdistserve.simulate \
    --num-node 1 --ngpu-per-node 4 \
    --model-type "facebook/opt-13b" \
    --workload sharegpt --backend distserve \
    --prefill-target 200 --decode-target 100 \
    --prefill-percentage 90 --decode-percentage 90 \
    --max-per-gpu-rate 5 \
    --esp 0.25 \
    --N 300
```

This experiment runs ShareGPT with 1 node with 4 A100 (80G) GPU. The model is facebook/opt-13b. We set the prefill SLA
to 200 ms, and decode SLA to 100 ms. The `prefill-percentage` and `decode-percentage` indicate the percentage of
requests to attain, in this case, P90 (90%). The `max-per-gpu-rate` is set to 5 to restrict the binary search runtime,
and we set the stopping criteria of binary search to ` hi - lo < 0.25`. The number of requests is set to 300.

Ideally you should get the following result:

```text
Best per GPU rate: 1.56
Best config: pp_cross=1, tp_prefill=2, pp_prefill=1, tp_decode=1, pp_decode=1
```

## Architecture

The simulator is written on top of `simpy`, a discrete event simulator build natively in Python. 

In the high level, our simulator is composed of the following core components (under the `base` module):

- `Worker`. A worker simulates the behavior of one or more GPU that executes in lock step (i.e. a group of `TP` number of GPUs with tensor parallelism). Worker can be a prefill worker or a decode worker, depending on its assignment in the cluster. Multiple workers can be chained together to form a pipeline. Setting the corresponding `PP` value in the simulator changes the pipeline parallelism for the cluster.
- `Scheduler`. The scheduler forwards the requests to the workers, with specified arrival rate and process (e.g. Poisson, Gamma, fixed rate).
- `Cluster`: The cluster is just a wrapper around the workers and scheduler. `DisaggCluster` and `VLLMCluster` are the available options to choose from.

On top of the core components, we implement the embarrassingly parallel search of the optimal parallelism configuration. The search is done by a binary search algorithm that tries to find the optimal per GPU rate that maximizes the goodput of the workload. The algorithm is currently under `benchmarks` module.
