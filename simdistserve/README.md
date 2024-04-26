# DistServe Simulator

Simulator that finds the optimal paralleism strategy for DistServe / vLLM.

Given a specific workload, number of nodes, and GPU per node, the simulator finds the optimal parallelism strategy that
maximize the goodput of the workload.

## Quick Start

If you have installed `distserve`, you should have also installed this `simdiserve` package.

To run OPT13B with ShareGPT workload to find the best config with one 4xA100 80G SXM node (4 GPU per node):

TODO: ...

## Architecture

```
 base
```