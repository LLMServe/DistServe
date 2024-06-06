## Set up the conda environment for vLLM

We use version v0.2.3, the last version released before the paper submission.

```bash
# Create the conda env
conda create -n distserve-vllm python=3.10.14
conda activate distserve-vllm

# Clone the repo
git clone git@github.com:interestingLSY/vllm.git
cd vllm
git checkout distserve-baseline-vllm	# This is the branch we used. It's based on v0.2.3, with some minor changes.

# Install packages
pip install -r requirements.txt
pip install -e .
```