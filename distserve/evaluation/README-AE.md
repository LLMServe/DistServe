# DistServe Reproducibility Guide

This file will guide you through the process of reproducing the results of the paper "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving".

Here is a high level overview of the whole process:
1. Create a docker container from our image as a GPU instance on RunPod (other cloud providers are also supported)
2. Download the raw datasets and preprocess them
3. Run some toy examples to verify DistServe and two baselines (vLLM and DeepSpeed-MI) are working (for the kick-the-tires step)
4. Run the full experiments
5. Post-process the results and generate the figures

During the kick-the-tires step, only step 1, 2, and 3 are required, which takes about 21 human-minutes and 28 compute-minutes. The full experiments will take about TODO human-hours and TODO compute-hours.

## Step 1. Create a GPU instance on RunPod

*TODO human-minutes + TODO compute-minutes*

(For kick-the-tire, 1 GPU is sufficient. For the full experiments, 8 GPUs are necessary.)

(Use the template called `distserve-evaluation`)

## Step 2. Download Datasets

*3 human-minutes + 3 compute-minutes*

Please follow the following steps to download all datasets

```bash
mkdir -p /workspace/dataset/raw
cd /workspace/dataset/raw

# Download the "ShareGPT" dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Download the "HumanEval" dataset
wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip HumanEval.jsonl.gz

# Download the "LongBench" dataset
wget "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip?download=true" -O longbench.zip
unzip longbench.zip
mv data longbench
```

Now you should have `HumanEval.jsonl`, `ShareGPT_V3_unfiltered_cleaned_split.json`, and a folder `longbench` under `/workspace/dataset/raw`.

## Step 3. Preprocess datasets

*3 human-minutes + 10 compute-minutes*

Now we start to preprocess the datasets:

```bash
cd /app/distserve/distserve/evaluation/
micromamba activate distserve	# NOTE. micromamba is a lightweight conda-like package manager

# Preprocess the "ShareGPT" dataset
python3 2-benchmark-serving/0-prepare-dataset.py --dataset sharegpt --dataset-path /workspace/dataset/raw/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer facebook/opt-13b --output-path /workspace/dataset/sharegpt.ds

# Preprocess the "HumanEval" dataset
python3 2-benchmark-serving/0-prepare-dataset.py --dataset humaneval --dataset-path /workspace/dataset/raw/HumanEval.jsonl --tokenizer facebook/opt-13b --output-path /workspace/dataset/humaneval.ds

# Preprocess the "LongBench" dataset
python3 2-benchmark-serving/0-prepare-dataset.py --dataset longbench --dataset-path /workspace/dataset/raw/longbench/ --tokenizer facebook/opt-13b --output-path /workspace/dataset/longbench.ds
```

Now you should have `sharegpt.ds`, `humaneval.ds`, and `longbench.ds` under `/workspace/dataset/`.

## Step 4. Kick-the-tires

*15 human-minutes + 15 compute-minutes*

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
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend vllm --dataset /workspace/dataset/sharegpt.ds --num-prompts-req-rates "[(50, 2)]" --verbose
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
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend deepspeed --dataset /workspace/dataset/sharegpt.ds --num-prompts-req-rates "[(50, 2)]" --verbose
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
python3 /app/distserve/distserve/evaluation/2-benchmark-serving/2-benchmark-serving.py --backend distserve --dataset /workspace/dataset/sharegpt.ds --num-prompts-req-rates "[(50, 2)]" --verbose
```

Ideally it should generate a file `/workspace/exp-results/opt-1.3b-sharegpt/distserve-50-2.exp`.
