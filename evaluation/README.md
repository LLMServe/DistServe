# DistServe - Evaluation

This folder contains code and detailed instructions to evaluate DistServe's performance, as
well as all the necessary scripts to reproduce the experiments in our paper.

## Folder Structure

This folder contains the following sub-folders:
- `0-test-single-forward-performance`: Scripts for testing the time consumption of a single forward pass of DistServe and vLLM. Used for: 1) verifying the efficiency of SwiftTransformer, which is DistServe's "data path". 2) generating a profiler database for fitting the execution time estimation model.
- `1-fit-time-usage`: Scripts for fitting the execution time estimation model of DistServe.
- `2-benchmark-serving`: Scripts for benchmarking the online serving performance of DistServe, vLLM, and DeepSpeed-MII.
- `ae-scripts`: Automated scripts to produce experiment results.
- `docs`: Detailed instructions to run some toy examples and reproduce the experiment results in our paper.
- `Dockerfile.evaluation`: Dockerfile for building the image for artifact evaluation.

## Building the Docker Image for Artifact Evaluation

NOTE. This docker image is for the artifact evaluation process only, i.e., you want to exactly reproduce the experiment results in our paper. To deploy DistServe, please refer to the instructions in the project's README.

To build the Docker image, please run:

```bash
# (Assume your current working directory is DistServe/evaluation)

# Clone the (modified) version of vLLM
git clone https://github.com/interestingLSY/vllm.git --branch distserve-baseline-vllm

# Clone the (modified) version of DeepSpeed
git clone https://github.com/interestingLSY/DeepSpeed-MII.git --branch distserve-baseline
```

Then, clone `SwiftTransformer` to the root directory, and run `git submodule update --init --recursive` inside the `SwiftTransformer` directory.

Then we prepare the dataset. Please follow the instructions in `docs/repro-dataset.md`, and set `$DATASET` to "./dataset".

Finally, run the following command to build the Docker image (Assume your current working directory is distserve/evaluation):

```bash
docker build -f Dockerfile.evaluation -t distserve-evaluation:latest ../../
```

## Instructions

If you accidentally use [RunPod](https://www.runpod.io/) as the cloud provider, you can use our [script](./docs/grab-machine.md) to grab machine automatically.

Then follow [this document](./docs/README-AE.md) to reproduce all the results in our paper.