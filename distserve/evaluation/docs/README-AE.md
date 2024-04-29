# DistLLM Artifact Evaluation Guide

This is the artifact of the paper "DistLLM: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving". We are going to guide you through the process of reproducing the main results in the paper.

Here is a high level overview of the whole process:
1. Environment Setup: Create a GPU instance on [RunPod](https://www.runpod.io/) from our provided template with all the environment already setup.
2. Kick-the-tires: Run some toy examples to verify DistLLM and vLLM are working.
3. Full evaluation: Reproduce all the main results in the paper.

## Environment Setup

We use the cloud provider [Runpod](https://www.runpod.io/) to create compute instances and run all the experiments on them. We provided credentials for you to log in to Runpod in hotcrp and you can start the instance from a template with all the environments set up already. Different experiments require a different number of compute resources, please follow the detailed guidance below in each section to create the instance.

**It is appreciated to stop the instance when you finish the review process each time, we pay real dollars for the GPU hours :)**

### Dataset Preprocessing
To save your time, we've preprocessed the datasets in advance and saved them to `/app/dataset` in the template. If you want to reproduce the dataset, please follow [this instruction](repro-dataset.md).

## Kick-the-tires
*15 human-minutes + 15 compute-minutes*

Follow the steps below to create a instance with two `A100 SXM 80GB` GPUs on [RunPod](https://www.runpod.io/) with template `DistLLM-AE-GPU`: 
- Log in to [RunPod](https://www.runpod.io/) with the credentials provided in hotcrp.
- Switch the account from `osdi24ae` to `Hao Lab@UCSD` using the upper right button.
- Click `Pods` in the left toolbar.
- Click `+ Deploy`.
- Choose `A100 SXM 80GB`.
- Click `Change Template` and choose `DistLLM-AE-GPU`.
- Choose `GPU Count`: For Kick-the-tires, 2 GPUs are sufficient, which is usually always available on RunPod. 
- Click `Deploy On-Demand`: If the button is grey, it means this resource is not currently available.


When the instance is started, you can ssh into the instance  in your terminal. Remember to provide your public key on the hotcrp so that we can give you the access to the instance you create. Here are some high-level overviews and notes:
- From now on, we need to use two terminals simultaneously, one for the server (i.e. the inference engine), and one for the client (i.e. the load generator). We will refer to them as `S-terminal` and `C-terminal` respectively.
- We will use a wrapper script, `/app/distserve/distserve/evaluation/2-benchmark-serving/2-start-api-server.py`, to launch the API server. The script will print the command it uses to launch the server, which can be used to inspect the startup parameters.
- The load generator locates at `/app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py`. Given a target dataset and a list of (num_prompt, request_rate)s, it runs serval rounds of experiments, each with a given (num_prompt, request_rate) pair, and save the result to a file located at `/workspace/exp-results/<model-name>-<dataset-name>/<backend>-<num_prompt>-<request_rate>.exp`, for example, `/workspace/exp-results/opt-13b-sharegpt/vllm-50-2.exp`

*Note. "DistServe" and "DistLLM" are used interchangeably in the codebase, they are the same thing.*

Now we can run some toy examples to verify DistLLM and vLLM are working:

### vLLM

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/vllm-server.sh
```

Wait until the server is ready (i.e. `# GPU blocks: XXX, # CPU blocks: XXX` pops up)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/vllm-client.sh
```

In the script we add the `--verbose` flag to print out all prompts && responses for a simple correctness check. In the full evaluation section, we will not use this flag.

Ideally it should run without any error, and generate a file `/workspace/exp-results/opt-125m-sharegpt/vllm-10-1.exp`.

### DistLLM

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/distllm-server.sh
```

Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/kick-the-tires/distllm-client.sh
```

Ideally it should generate a file `/workspace/exp-results/opt-125m-sharegpt/distserve-10-1.exp`. The file should contain a JSON object which looks like:

```
[{"prompt_len": 1135, "output_len": 12, "start_time": 200915.496689009, "end_time": 200915.565055445, "token_timestamps": [...]}, ...]
```

## Full Evaluation

### End-to-end Experiments (Section 6.2, Figure. 8 + Figure. 9)
*15 human-minutes + 90 compute-minutes*

The OPT-175B experiment of DistLLM requires four 8xA100-SXM-80GB machines. On common cloud providers like AWS or RunPod, this experiment costs over 2000$ in total for each run. Due to the limited budget, it is too expensive for us to reproduce the OPT-175B experiment (Figure. 8c) so we reuse the data in our paper. But we do provide the scripts for interested ones who have enough resources to produce the results by themselves.

For OPT-13B and OPT-66B End-to-end Experiments, 8 GPUs are required and we provide a script to grab the machine automatically because 8xA100-SXM machine is a ridiculously popular resource on clouds and it usually takes over 1 day to grab the machine. For instructions on how to use this script, please refer to [this file](grab-machine.md).

For reviewers who do not want to experience this tedious machine-grabbing process, we provide the [screencast](https://drive.google.com/drive/folders/1QCEkpV4Wi2WUutFnDR46NrsSTDXr8lL3?usp=sharing) of producing the results in each figure. 

If you successfully obtain one 8xA100-SXM-80GB machine, please follow the instructions below to reproduce the results in Figure. 8 and Figure. 9.

Let's start with the OPT-13B experiment in Figure. 8:

First for vLLM:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-vllm-server.sh
```
Wait until the server is ready (i.e. `# GPU blocks: XXX, # CPU blocks: XXX` pops up)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-vllm-client.sh
```
Wait until the client finishes (i.e. exits without any error)

---
Then for DistLLM:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-distllm-server.sh
```
Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-13b-distllm-client.sh
```
Wait until the client finishes (i.e. exits without any error)

---

And then let's move on to the OPT-66B experiment in Figure. 8:

First for vLLM:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-vllm-server.sh
```
Wait until the server is ready (i.e. `# GPU blocks: XXX, # CPU blocks: XXX` pops up)

On the `C-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-vllm-client.sh
```
This script runs all three datasets (ShareGPT, HumanEval, LongBench) in sequence, which will take a while (~30 minutes).

Wait until the client finishes (i.e. exits without any error)

---

Then for DistLLM:

On the `S-terminal`, execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-distllm-server.sh
```
Wait until the server is ready (i.e. the engine begins to print its status once per second)

On the `C-terminal`, execute
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/opt-66b-distllm-client.sh
```
It will also take a while (~30 minutes).
Wait until the client finishes (i.e. exits without any error)

---

Finally is to run the plotting script: execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/plot-fig-8-and-9.sh
```
Plots will be saved under `/workspace/plots`.

### Latency Breakdown (Section 6.3, Figure. 10) 

Due to the same budget reason, we cannot afford to reproduce the OPT-175B experiment in the left figure of Figure. 10. However, we provide a OPT-66B version which can also verify our claim in this Section that the transmission time is negligible compared to computation in DistLLM.

We also provide the [screencast](https://drive.google.com/drive/folders/1QCEkpV4Wi2WUutFnDR46NrsSTDXr8lL3?usp=sharing) of producing the results in Figure. 10 in case the reviewers do not want to experience the machine-grabbing process.

If you have successfully obtained one 8xA100-SXM-80GB machine, after running end-to-end experiments above, you can execute 
```bash
bash /app/distserve/distserve/evaluation/ae-scripts/e2e/plot-fig-10.sh
```
to generate Figure. 10. Plots will be saved under `/workspace/plots`.

### Ablation Studies (Section 6.4, Figure. 11)
Follow the steps below to create a instance with one `RTX3090` GPU on [RunPod](https://www.runpod.io/) with template `DistLLM-AE-GPU`: 
- Log in to [RunPod](https://www.runpod.io/) with the credentials provided in hotcrp.
- Switch the account from `osdi24ae` to `Hao Lab@UCSD` using the upper right button.
- Click `Pods` in the left toolbar.
- Click `+ Deploy`.
- Choose `RTX3090`.
- Click `Change Template` and choose `DistLLM-AE-GPU`.
- Choose `GPU Count`: 1 GPU is sufficient. In fact, the ablation experiment can be run on CPU instances, but the template we provide requires GPU environment.
- Click `Deploy On-Demand`: If the button is grey, it means this resource is not currently available.

TODO: @cjd plot to pdf instead of png

Execute xxx

... ...

Plots will be saved under `/workspace/plots`.