# Dataset Preprocessing Instructions

This doc will guide you through the dataset preprocessing procedure in our paper.

We assume that your dataset directory is located at `$DATASET`. You may use `export DATASET=/path/to/dataset` to set the environment variable.

## Step 1. Download Datasets

*3 human-minutes + 3 compute-minutes*

Please follow the following steps to download all datasets

```bash
mkdir -p $DATASET/raw
cd $DATASET/raw

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

Now you should have `HumanEval.jsonl`, `ShareGPT_V3_unfiltered_cleaned_split.json`, and a folder `longbench` under `$DATASET/raw`.



## Step 2. Preprocess datasets

*3 human-minutes + 10 compute-minutes*

Now we start to preprocess the datasets:

```bash
conda activate distserve

# Preprocess the "ShareGPT" dataset
python3 2-benchmark-serving/0-prepare-dataset.py --dataset sharegpt --dataset-path $DATASET/raw/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer facebook/opt-13b --output-path $DATASET/sharegpt.ds

# Preprocess the "HumanEval" dataset
python3 2-benchmark-serving/0-prepare-dataset.py --dataset humaneval --dataset-path $DATASET/raw/HumanEval.jsonl --tokenizer facebook/opt-13b --output-path $DATASET/humaneval.ds

# Preprocess the "LongBench" dataset
python3 2-benchmark-serving/0-prepare-dataset.py --dataset longbench --dataset-path $DATASET/raw/longbench/ --tokenizer facebook/opt-13b --output-path $DATASET/longbench.ds
```

Now you should have `sharegpt.ds`, `humaneval.ds`, and `longbench.ds` under `$DATASET/`.


