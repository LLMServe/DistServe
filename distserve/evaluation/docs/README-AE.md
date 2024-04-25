# DistLLM Artifact Evaluation Guide

This is the artifact of the paper "DistLLM: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving". We are going to guide you through the process of reproducing the main results in the paper.

Here is a high level overview of the whole process:
1. Environment Setup: Create a GPU instance on [RunPod](https://www.runpod.io/) from our provided template with all the environment already setup.
2. Kick-the-tires: Run some toy examples to verify DistLLM and baselines (vLLM and DeepSpeed-MII) are working.
3. Full evaluation: Reproduce all the main results in the paper.

## Environment Setup

Follow the steps below to create a GPU instance on [RunPod](https://www.runpod.io/) with template `distserve-evaluation`: 
- Log in to [RunPod](https://www.runpod.io/) with the credentials provided in hotcrp.
- Click `Pods` in the left toolbar.
- Click `+ Deploy`.
- Choose `A100 SXM 80GB`.
- Click `Change Template` and choose `distserve-evaluation`.
- Choose `GPU Count`: For Kick-the-tires, 2 GPUs are sufficient, which is usually always available on RunPod. For full evaluation, 8 GPUs are required and we provide a script to grab the machine automatically because 8xA100-SXM machine is a ridiculously popular resource on clouds and it usually takes over 1 day to grab the machine.
- Click `Deploy On-Demand`: If the button is grey, it means this resource is not currently available.

**It is appreciated to stop the instance when you finish the review process each time, we pay real dollars for the GPU hours :)**

### Dataset Preprocessing
To save your time, we've preprocessed the datasets in advance and saved them to `/app/dataset` in the template. If you want to reproduce the dataset, please follow [this instruction](repro-dataset.md).

## Kick-the-tires
*15 human-minutes + 20 compute-minutes*

Some high-level overviews and notes:
- From now on, we need to use two terminals simultaneously, one for the server (i.e. the inference engine), and one for the client (i.e. the load generator). We will refer to them as `S-terminal` and `C-terminal` respectively.
- We will use a wrapper script, `/app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py`, to launch the API server. The script will print the command it uses to launch the server, which can be used to inspect the startup parameters.
- The load generator locates at `/app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py`. Given a target dataset and a list of (num_prompt, request_rate)s, it runs serval rounds of experiments, each with a given (num_prompt, request_rate) pair, and save the result to a file located at `/workspace/exp-results/<model-name>-<dataset-name>/<backend>-<num_prompt>-<request_rate>.exp`, for example, `/workspace/exp-results/opt-1.3b-sharegpt/vllm-50-2.exp`

Now we can run some toy examples to verify DistLLM and two baselines (vLLM and DeepSpeed) are working:

### vLLM

On the `S-terminal`, run the following command:

```bash
micromamba activate distserve
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py --backend vllm --model facebook/opt-1.3b
```

Wait until the server is ready (i.e. `# GPU blocks: XXX, # CPU blocks: XXX` pops up)

On the `C-terminal`, run the following command:

```bash
micromamba activate distserve
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend vllm --dataset /app/dataset/sharegpt.ds --num-prompts-req-rates "[(50, 2)]" --verbose
```

*Note. Here we add the `--verbose` flag to print out all prompts & responses for a simple correctness check. In the full evaluation section, we will not use this flag.*

Ideally it should run without any error, and generate a file `/workspace/exp-results/opt-1.3b-sharegpt/vllm-50-2.exp`.

### DeepSpeed-MII

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

### DistLLM

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



## Full Evaluation

### End-to-end Experiments (Section 6.2, Figure. 8 + Figure. 9)
The OPT-175B experiment of DistLLM requires four 8xA100-SXM-80GB machines. On common cloud providers like AWS or Runpod, this experiment costs over 2000$ in total for each run. Due to the limited budget, it is too expensive for us to reproduce the OPT-175B experiment (Figure. 8c) so we reuse the data in our paper. But we do provide the scripts for interested ones who have enough resources to produce the results by themselves.

The OPT-13B and OPT-66B experiments require one 8xA100-SXM-80GB machine. However, due to the shortage of 8xA100-SXM-80GB machines recently, it takes 1-2 days to grab even a single machine with automatic script (which we will provide you). So for reviewers who do not want to experience this tedious machine-grabbing process, we provide the [screencast]() of producing the results in each figure. 

If you successfully obtain one 8xA100-SXM-80GB machine, run

TODO: Command
### Latency Breakdown (Section 6.3, Figure. 10) 
Due to the same budget reason, we cannot afford to reproduce the OPT-175B experiment in the left figure of Figure. 10. However, we provide a OPT-66B version which can also verify our claim in this Section that the transmission time is negligible compared to computation in DistLLM.

We also provide the [screencast]() of producing the results in Figure. 10 in case the reviewers do not want to experience the machine-grabbing process.

If you successfully obtain one 8xA100-SXM-80GB machine, run

TODO: Command

### Ablation Studies (Section 6.4, Figure. 11)

*Compute Time: 5 min*

The abalation study is sufficient to run on CPU-only instance. 

To allocate a CPU instance in RunPod, follow these steps:

- Log in to [RunPod](https://www.runpod.io/) with the credentials provided in hotcrp.
- Click `Storage` in the left toolbar. Select `OSDI24 DistServe Abalation` and click Deploy.
- Select the `CPU` on the top. Then select `Compute-Optimized` instance with `32 vCPUs`. 
- Name the pod, for example `OSDI Abalation Eval`
- Check the pod template is `Runpod Ubuntu (runpod/base:0.5.1-cpu)`
- Click `Deploy On-Demand`



#### Run Abalation Study

Once you connect to the instance, run the following to start benchmarking:

```bash
# Enable the python virtual environment
cd /workspace
source venv/bin/activate

# Run abalation study
cd /workspace/DistServe/simdistserve/benchmarks/figure11-abalation
rm -rf result visual figure # clean existing results if exists
bash 01-run_abalation.sh opt_13b_sharegpt

# Draw figures
mkdir -p visual figure
python 02-draw_rate_abalation.py --target "(200, 100)"
python 03-draw_slo_abalation.py --target "(200, 100)" --per_gpu_rate 1
python 04-draw_abalation_curve.py --rates "[1,2,3,4,5]"

# See the figure `abalation.png`
cd /workspace/DistServe/simdistserve/benchmarks/figure11-abalation/figure
```



TODO: How should reviewer see the file?
