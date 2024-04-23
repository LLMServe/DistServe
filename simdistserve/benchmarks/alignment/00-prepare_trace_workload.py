from copy import deepcopy

import pandas as pd
from pathlib import Path
import json

from simdistserve.benchmarks.simulate_dist import load_workload


def get_trace_file(file):
    with open(file, "r") as f:
        # read one line
        a = f.readline().strip()
        reqs = eval(a)
    c = [(
        r['prompt_len'],
        r['output_len'],
        r['issue_time'],
    ) for r in reqs]
    base_time = min([i[2] for i in c])
    c = sorted(c, key=lambda x: x[2])
    d = [(i[0], i[1], i[2] - base_time) for i in c]
    return d


root_dor = Path("data-ground-truth") / ("opt-13b-sharegpt")
files = list(
    root_dor.glob("distserve*.exp")
) + list(
    root_dor.glob("vllm*.exp")
)

trace_file_repr = {}
result = {}
for file in files:
    a = get_trace_file(file)
    result[file.name] = a
    b = [
        {
            "prompt_len": i[0],
            "output_len": i[1],
            "start_time": i[2]
        }
        for i in a
    ]
    trace_file_repr[file.name] = b

# workload = load_workload("sharegpt", 1000, 1, 1, 0, "gamma")

trace_root_dor = Path("data-ground-truth") / ("opt-13b-sharegpt-workload")
for filename, trace in trace_file_repr.items():
    file = trace_root_dor / filename
    # Write the JSON to the file
    with open(file, "w") as f:
        json.dump(trace, f)
    pass
