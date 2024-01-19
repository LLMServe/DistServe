# DistServe

A disaggregated Large Language Model (LLM) serving system.

Code & artifacts for the paper <TODO> (link: <TODO>)

It is fast with:
- Prefill stage and decoding stage disaggregation
- preemptive scheduling
- various scheduling algorithms (FCFS / MLFQ / ...)
- continuous batching (iterational-level scheduling and selective batching)
- pure C++ GPT implementation
- flash attention kernel (xFormers' kernels for non-GQA models and hand-written high-performance kernels for models with GQA)

It is memory efficient with:
- paged attention kernel
- proactive memory swapping

It is scalable with:
- megatron-LM tensor parallelism
- streaming pipeline parallelism

It supports:
- OPT (facebook/opt-1.3b, facebook/opt-6.7b, ...)
- LLaMA2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b, ...) (will be supported in the future)

## Installation

### Requirements

- NVIDIA GPU with compute capability >= 6.0 (Pascal architecture). Maxwell architecture is not supported since it does not support FP16.
- The NVIDIA GPU driver.
- CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- NCCL: https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#down

### Installation

- git clone the repo
- Create a conda environment: `conda env create -f environment.yml`
- Activate the conda environment: `conda activate distserve`
- Clone the C++ backend, SwiftTransformer, via `git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive`
- Build the SwiftTransformer library `cd SwiftTransformer; cmake -B build; cmake --build build -j$(nproc)`
- On successful builds, you should see `libst_pybinding.so` under the `SwiftTransformer/build/lib` directory

## Launching

### Launchng a Ray Cluster

DistLLM relies on [ray](https://ray.io) to achieve parallelism. If you does not launch a ray cluster in advance, ray will automatically initiate a mini cluster, consisting only all gpus on the current node. You may need to start the ray cluster manually in advance if you want to use multiple nodes for inference.

### Run the Example

We provide an offline examples to play with, you may try `distserve/exmples/offline.py`.

### Launch the API Server

To launch the api server, try out `distserve/api_server/distserve_api_server.py`


