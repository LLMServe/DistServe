# DistServe Reproducibility Guide

This file will guide you through the process of reproducing the results of the paper "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving".

Since it is rediculously hard to rent a node with eight NVIDIA A100 80GB SXM GPUs, and (at least for us) impossible to rent three of them with Infiniband interconnect, we will provide a screencast for the OPT-175B end-to-end experiment (which needs three nodes).

For the OPT-13B end-to-end experiment (which runs on 4 GPUs) and the OPT-66B experiment (which runs on 8 GPUs), we provide a python script for you to SHUA JI QI. Ideally it grabs a 4 GPU node in 10 minutes and an 8 GPU node in about 24 hours. We also provide corresponding screen recordings, if you find it annoying to SHUA JI QI.

Here is a high level overview of the whole process:
1. Create a docker container instance (pod) from our image as a GPU instance on RunPod (other cloud providers are also supported)
2. Run some toy examples to verify DistServe and two baselines (vLLM and DeepSpeed-MI) are working (for the kick-the-tires step)
3. Run the full experiments
4. Post-process the results and generate the figures

During the kick-the-tires step, only step 1 and 2 are required, which takes about TODO human-minutes and TODO compute-minutes. The full experiments will take about TODO human-hours and TODO compute-hours.

NOTE. To save your time, we've preprocessed the datasets and saved them to `/app/dataset`, so you can skip the dataset preparation step. If you want to reproduce the dataset, please refer to `docs/repro-dataset.md`.

## Step 1. Create a GPU instance on RunPod

*TODO human-minutes + TODO compute-minutes*

(For kick-the-tire, 2 GPUs are sufficient, which can be easily obtained on RunPod)

(Use the template called `distserve-evaluation`)

## Step 2. Kick-the-tires

*15 human-minutes + 20 compute-minutes*

Some high-level overviews and notes:
- From now on, we need to use two terminals simultaneously, one for the server (i.e. the inference engine), and one for the client (i.e. the load generator). We will refer to them as `S-terminal` and `C-terminal` respectively.
- We will use a wrapper script, `/app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py`, to launch the API server. The script will print the command it uses to launch the server, which can be used to inspect the startup parameters.
- The load generator locates at `/app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py`. Given a target dataset and a list of (num_prompt, request_rate)s, it runs serval rounds of experiments, each with a given (num_prompt, request_rate) pair, and save the result to a file located at `/workspace/exp-results/<model-name>-<dataset-name>/<backend>-<num_prompt>-<request_rate>.exp`, for example, `/workspace/exp-results/opt-1.3b-sharegpt/vllm-50-2.exp`

Now we can run some toy examples to verify DistServe and two baselines (vLLM and DeepSpeed) are working:

### vLLM

On the `S-terminal`, run the following command:

```bash
micromamba activate distserve	# NOTE. This is not a typo. This environment is for the wrapper script
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py --backend vllm --model facebook/opt-1.3b
```

Wait until the server is ready (i.e. `# GPU blocks: XXX, # CPU blocks: XXX` pops up)

On the `C-terminal`, run the following command:

```bash
micromamba activate distserve
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend vllm --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(50, 2)]" --verbose
```

*Note. Here we add the `--verbose` flag to print out all prompts & responses for a simple correctness check. On the formal experiments (@zym polish this plz), we will not use this flag.*

Ideally it should run without any error, and generate a file `/workspace/exp-results/opt-1.3b-sharegpt/vllm-50-2.exp`.

### DeepSpeed

On the `S-terminal`, run the following command:

```bash
micromamba activate distserve
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py --backend deepspeed --model facebook/opt-1.3b
```

Wait until the server is ready (i.e. `Uvicorn running on http://0.0.0.0:8300 (Press CTRL+C to quit)` pops up)

*Note. DeepSpeed compiles some CUDA kernels on the first start up, which takes several minutes.*

On the `C-terminal`, run the following command:

```bash
micromamba activate distserve
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend deepspeed --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(50, 2)]" --verbose
```

Ideally it should generate a file `/workspace/exp-results/opt-1.3b-sharegpt/deepspeed-50-2.exp`.

### DistServe

On the `S-terminal`, run the following command:

```bash
micromamba activate distserve
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py --backend distserve --model facebook/opt-1.3b
```

Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `C-terminal`, run the following command:

```bash
micromamba activate distserve
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(50, 2)]" --verbose
```

Ideally it should generate a file `/workspace/exp-results/opt-1.3b-sharegpt/distserve-50-2.exp`.
