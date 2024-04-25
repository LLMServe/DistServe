import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

max_machine = 4
max_gpu_per_node = 8


def parse_args(args_=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='(400, 100)')
    parser.add_argument("--per_gpu_rate", type=float, default=0.5)
    args = parser.parse_args(args_)
    return args


args = parse_args()
chosen_per_gpu_rate = args.per_gpu_rate
target = eval(args.target)

Path("figure").mkdir(exist_ok=True)
Path("visual").mkdir(exist_ok=True)

# Get all files with format '*.latency.csv' from root_dir
# root_dir = Path("fig11-abalation-log")
root_dir = Path("result")
latency_file_paths = sorted(list(root_dir.glob("*.latency.csv")))
columns = ['backend', 'rate', 'target', 'attainment', 'latency']

dfs = []
namespaces = []
for latency_file_path in latency_file_paths:
    try:
        df = pd.read_csv(latency_file_path)
        dfs.append(df)
    except pd.errors.EmptyDataError:
        pass

big_df = pd.concat(dfs, ignore_index=True)
big_df['ngpu'] = big_df['tp_prefill'] * big_df['pp_prefill'] + big_df['tp_decode'] * big_df['pp_decode']
big_df['per_gpu_rate'] = big_df['rate'] / big_df['ngpu']
big_df['goodput@90'] = big_df.apply(
    lambda x: x['rate'] / x['ngpu'] if x['attainment'] >= 90 else 0,
    axis=1,
)
big_df['target'] = big_df['target'].apply(eval)
big_df = big_df[big_df['per_gpu_rate'] == chosen_per_gpu_rate]

big_df

slos = [0.4, 0.6, 0.8, 1, 1.2]
targets = [
    (target[0] * slo, target[1] * slo)
    for slo in slos
]
target_to_slo = {
    target: slo
    for target, slo in zip(targets, slos)
}

big_df = big_df[big_df['target'].isin(targets)]
big_df = big_df.copy()

big_df['slo'] = big_df['target'].apply(lambda x: target_to_slo[x])


def can_fit_low_affinity_distserve(x):
    a, b, c, d = x['tp_prefill'], x['pp_prefill'], x['tp_decode'], x['pp_decode']
    for pp_common in range(1, max_machine + 1):
        bp = b / pp_common
        dp = d / pp_common
        # If either bp or dp is not int, skip
        if int(bp) != bp or int(dp) != dp:
            continue
        # Check if the segment can be placed inside a node
        if a * bp + c * dp <= max_gpu_per_node:
            return True
        pass
    return False


def can_fit_low_affinity(x):
    if x['backend'] == 'distserve':
        return can_fit_low_affinity_distserve(x)
    else:
        return True
    pass


big_df['low_affin'] = big_df.apply(can_fit_low_affinity, axis=1)

big_df

figure_11_right_df = big_df.copy()

figure_11_distserve_high = figure_11_right_df[
    (figure_11_right_df['backend'] == 'distserve')
]
figure_11_distserve_low = figure_11_right_df[
    (figure_11_right_df['backend'] == 'distserve') & (figure_11_right_df['low_affin'])
    ]
figure_11_vllm_high = figure_11_right_df[
    (figure_11_right_df['backend'] == 'vllm')
]
figure_11_vllm_low = figure_11_right_df[
    (figure_11_right_df['backend'] == 'vllm') & (figure_11_right_df['low_affin'])
    ]


def get_top_config(df):
    strictest_slo = min(slos)
    r = df[df['slo'] == strictest_slo].sort_values(
        by=['goodput@90', 'attainment'],
        ascending=False,
    ).iloc[0][[
        "tp_prefill",
        "pp_prefill",
        "tp_decode",
        "pp_decode",
    ]]
    return r


big_df = big_df.sort_values(by=['per_gpu_rate', 'slo', ], ascending=False)


def get_top_config(df):
    strictest_slo = min(slos)
    r = df[df['slo'] == strictest_slo].sort_values(
        by=['goodput@90', 'attainment'],
        ascending=False,
    ).iloc[0][[
        "tp_prefill",
        "pp_prefill",
        "tp_decode",
        "pp_decode",
    ]]
    return r


def add_matplotlib_trace(fig, df: 'DataFrame', trace: str):
    tp_prefill, pp_prefill, tp_decode, pp_decode = get_top_config(df)
    config_df = df[
        (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill) &
        (df['tp_decode'] == tp_decode) & (df['pp_decode'] == pp_decode)
        ]
    config_df = config_df.sort_values(by=['slo'], ascending=False)
    if 'vllm' in trace:
        name = f"{trace}-p{tp_prefill}{pp_prefill}"
        pass
    else:
        name = f"{trace}-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}"
        pass

    fig.plot(
        config_df['slo'],
        config_df['attainment'],
        label=name,
        marker='o',
    )
    return config_df['attainment'].tolist()


# Plot a line chart with 4 curves
# x-axis: per_gpu_rate
# y-axis: attainment

fig, ax = plt.subplots()
a = add_matplotlib_trace(ax, figure_11_distserve_high, "disthigh")
b = add_matplotlib_trace(ax, figure_11_distserve_low, "distlow")
c = add_matplotlib_trace(ax, figure_11_vllm_high, "vllm++")
d = add_matplotlib_trace(ax, figure_11_vllm_low, "vllm")
plt.title("Figure 11: Abalation Study (DistServe and vLLM)")
plt.xlabel("SLO Scale")
plt.ylabel("SLO Attainment (%)")
plt.xticks(slos)
plt.gca().invert_xaxis()
plt.legend()

# save the plot 
fig.savefig("figure/figure_11b.png")

data_points = {
    "dist++": a,
    "dist": b,
    "vllm++": c,
    "vllm": d,
}
with open("figure/figure_11b.json", "w") as f:
    json.dump(data_points, f)
