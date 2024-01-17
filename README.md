# DistServe

A disaggregated Large Language Model (LLM) serving system.

## Installation

- git clone the repo
- create a conda environment: `conda env create -f environment.yml`
- activate the conda environment: `conda activate distserve`
- clone the C++ backend, SwiftTransformer, via `git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive`
- build the SwiftTransformer library `cd SwiftTransformer; cmake -B build; cmake --build build -j$(nproc)`

## Launching

### Launchng a Ray Cluster

DistLLM relies on [ray](https://ray.io) to implement parallelism. If you does not launch a ray cluster in advance, ray will automatically initiate a mini cluster, consisting only the current node. You may start the ray cluster manually in advance if you want to use multiple nodes for inferencing.

### a

We provide some offline examples to play with, you may try `distserve/exmples/offline.py`.

To launch the api server, try out `distserve/api_server/distserve_api_server.py`