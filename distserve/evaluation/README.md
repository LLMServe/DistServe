# DistServe - Evaluation

This folder contains code & instructions to evaluate DistServe's performance, as
well as all the necessary scripts to reproduce the experiments in the paper.

**For AE reviewers, please read docs/AE-guide.md.**

## Folder Structure

This folder contains the following subfolders:
- `0-test-single-forward-performance`: Scripts for testing the time consumption of a single forward pass of DistServe and vLLM. Used for: 1) verifying the efficiency of SwiftTransformer, which is DistServe's "data path". 2) generating a profiler database for fitting the execution time estimation model.
- `1-fit-time-usage`: Scripts for fitting the execution time estimation model of DistServe.
- `2-benchmark-serving`: Scripts for benchmarking the online serving performance of DistServe, vLLM, and DeepSpeed-MII.
- `Dockerfile.evaluation`: Dockerfile for building the image for artifact evaluation.

## Building the Docker Image

NOTE. This docker image is for the artifact evaluation process only. To deploy DistServe, please refer to the dockerfile in the root directory.

To build the Docker image, please:

```bash
# (Assume your current working directory is distserve/evaluation)

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
