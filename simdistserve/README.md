# DistServe Simulator

## Environment Setup

```bash
# Environment (prepared in runpod)
apt update
apt install -y python-is-python3 git 

# Install Python virtual env
cd /workspace
python3.11 -m virtualenv venv
source venv/bin/activate
python -m pip install simpy tqdm matplotlib pandas joblib shlex 


```
## Run benchmark
```bash
# Enable the python virtual environment
cd /workspace
source venv/bin/activate

# Run the benchmark
cd /workspace/DistServe/simdistserve/benchmarks
bash run_abalation_13b_sharegpt.sh

# See the figure `abalation.png`
cd /workspace/DistServe/simdistserve/benchmarks/figure
```
