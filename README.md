# DistServe

DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding
computation. Existing LLM serving systems colocate the two
phases and batch the computation of prefill and decoding
across all users and requests. We find that this strategy not
only leads to strong prefill-decoding interferences but also
couples the resource allocation and parallelism plans for both
phases. In DistServe, you can simply set the parallelism configs and scheduling strategies for the two phases and it will work just like a single instance which handles the KV-Cache communication and memory management automatically.  

It supports:
- GPT-2 (gpt2, gpt2-xl, ...)
- OPT (facebook/opt-1.3b, facebook/opt-6.7b, ...)
- LLaMA2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b, ...)

## Build && Install
```shell
# clone the project
git clone git@github.com:LLMServe/DistServe.git && cd DistServe

# setup the distserve conda environment
conda env create -f environment.yml && conda activate distserve

# clone and build the SwiftTransformer library  
git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive && cmake -B build && cmake --build build -j$(nproc) && cd ..

# install distserve
pip install -e .
```

## Launching

### Launch Ray Cluster

DistServe relies on [ray](https://ray.io) to implement workers. If you do not launch a ray cluster in advance, it will automatically initiate a cluster consisting of all the gpus on the current node. You may need to start the ray cluster manually in advance if you want to use multiple nodes for inference.

### Run offline Example

We provide an offline examples to play with, you may try `distserve/exmples/offline.py`.

### Launch the API Server

To launch the online server API server, try out `distserve/api_server/distserve_api_server.py`

## Citation
If you use DistServe for your research, please cite our [paper](https://arxiv.org/abs/2401.09670):
```
@misc{zhong2024distserve,
      title={DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving}, 
      author={Yinmin Zhong and Shengyu Liu and Junda Chen and Jianbo Hu and Yibo Zhu and Xuanzhe Liu and Xin Jin and Hao Zhang},
      year={2024},
      eprint={2401.09670},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
