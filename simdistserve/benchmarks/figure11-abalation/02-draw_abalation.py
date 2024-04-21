#!/usr/bin/env python
# coding: utf-8

# In[17]:


from pathlib import Path

import pandas as pd

# Get all files with format '*.latency.csv' from root_dir
# root_dir = Path("fig11-abalation-log")
root_dir = Path("result")
latency_file_paths = sorted(list(root_dir.glob("*.latency.csv")))
experiment_log_paths = sorted(list(root_dir.glob("*.log")))
columns = ['backend', 'rate', 'target', 'attainment', 'latency']

# In[18]:


dfs = []
namespaces = []
for latency_file_path, experiment_log_path in zip(latency_file_paths, experiment_log_paths):
    # read experiment_log_path and log the namespace
    with open(experiment_log_path, 'r') as f:
        exp_args = f.read()
        exp_args = eval(exp_args)
        namespaces.append(exp_args)

    df = pd.read_csv(latency_file_path)
    dfs.append(df)

# In[19]:


big_df = pd.concat(dfs, ignore_index=True)
big_df['ngpu'] = big_df['tp_prefill'] * big_df['pp_prefill'] + big_df['tp_decode'] * big_df['pp_decode']
big_df['per_gpu_rate'] = big_df['rate'] / big_df['ngpu']
big_df['goodput@90'] = big_df.apply(
    lambda x: x['rate'] / x['ngpu'] if x['attainment'] >= 90 else 0,
    axis=1,
)

# In[20]:


big_df

# In[21]:


max_machine = 4
max_gpu_per_node = 8


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

# In[22]:


big_df.sort_values(by=['backend', 'per_gpu_rate', 'tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode'])

# In[23]:


target = '(200.0, 100.0)'
figure_11_left_df = big_df[
    (big_df['pd'] == 'both')
    & (big_df['target'] == target)
    ].copy()

figure_11_left_df = figure_11_left_df.sort_values(by=[
    'backend', 'tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode',
    'rate'
])
# Choose the config with the best goodput in each group
figure_11_distserve_high = figure_11_left_df[
    (figure_11_left_df['backend'] == 'distserve')
]
figure_11_distserve_low = figure_11_left_df[
    (figure_11_left_df['backend'] == 'distserve')
    & (figure_11_left_df['low_affin'])
    ]
figure_11_vllm_high = figure_11_left_df[
    (figure_11_left_df['backend'] == 'vllm')
]
figure_11_vllm_low = figure_11_left_df[
    (figure_11_left_df['backend'] == 'vllm')
    & (figure_11_left_df['pp_prefill'] == 1)
    ]


# In[24]:


def find_best_config(df):
    # Filter the DataFrame to include only rows where goodput@90 > 0
    filtered_df = df[df['goodput@90'] > 0]

    # Group by the specified columns and find the row with the maximum 'rate' in each group
    grouped_df = filtered_df.groupby(['tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode'])
    max_rate_df = grouped_df['rate'].idxmax()

    # Retrieve the rows with the maximum 'rate' from the original DataFrame using the indices
    best_configs = df.loc[max_rate_df]

    # Return the DataFrame containing the best configurations
    # return best_configs[['tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode']]
    return best_configs


# In[25]:


figure_11_distserve_high

# In[26]:


figure_11_distserve_low

# In[27]:


figure_11_vllm_high

# In[28]:


figure_11_vllm_low

# In[29]:


# Plot the `figure_11_distserve_high`for some configurations
# tp_prefill = 1, pp_prefill = 1, tp_decode = 1, pp_decode = 1
# x-axis: rate
# y-axis: attainment
# find all combination of tp_prefill, pp_prefill, tp_decode, pp_decode
import plotly.graph_objects as go

fig = go.Figure()
configs = figure_11_distserve_high[['tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode']].drop_duplicates()
df = figure_11_distserve_high

for tp_prefill, pp_prefill, tp_decode, pp_decode in configs.values:
    config_df = df[
        (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill) &
        (df['tp_decode'] == tp_decode) & (df['pp_decode'] == pp_decode)
        ]
    # plot this inside a plotly plot
    fig.add_trace(go.Scatter(
        x=config_df['per_gpu_rate'], y=config_df['attainment'],
        mode='lines+markers', name=f"p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-distserve"
    ))

# fig add title
fig.update_layout(
    title="DistServe",
    xaxis_title="Per-GPU Rate (tokens/s)",
    yaxis_title="Attainment (%)",
    legend_title="Configuration"
)

fig.show()
# Export to html
fig.write_html("visual/figure_11_distserve_high.html")

# In[30]:


# Plot the `figure_11_vllm_high`for some configurations
# tp_prefill = 1, pp_prefill = 1
# x-axis: rate
# y-axis: attainment
# find all combination of tp_prefill, pp_prefill
import plotly.graph_objects as go

fig = go.Figure()
configs = figure_11_vllm_high[['tp_prefill', 'pp_prefill']].drop_duplicates()
df = figure_11_vllm_high

for tp_prefill, pp_prefill in configs.values:
    config_df = df[
        (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill)
        ]
    # plot this inside a plotly plot
    fig.add_trace(go.Scatter(
        x=config_df['per_gpu_rate'], y=config_df['attainment'],
        mode='lines+markers', name=f"p{tp_prefill}{pp_prefill}-vllm"
    ))

# fig add title
fig.update_layout(
    title="vLLM++",
    xaxis_title="Per-GPU Rate (tokens/s)",
    yaxis_title="Attainment (%)",
    legend_title="Configuration"
)
fig.show()
# Export to html
fig.write_html("visual/figure_11_vllm_high.html")

# In[31]:


import plotly.graph_objects as go

fig = go.Figure()

# Plot the `figure_11_distserve_high`for some configurations
# tp_prefill = 1, pp_prefill = 1, tp_decode = 1, pp_decode = 1
# x-axis: rate
# y-axis: attainment
# find all combination of tp_prefill, pp_prefill, tp_decode, pp_decode

configs = figure_11_distserve_high[['tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode']].drop_duplicates()
df = figure_11_distserve_high

for tp_prefill, pp_prefill, tp_decode, pp_decode in configs.values:
    config_df = df[
        (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill) &
        (df['tp_decode'] == tp_decode) & (df['pp_decode'] == pp_decode)
        ]
    # plot this inside a plotly plot
    fig.add_trace(go.Scatter(
        x=config_df['per_gpu_rate'], y=config_df['attainment'],
        mode='lines+markers', name=f"p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-distserve"
    ))

# Plot the `figure_11_vllm_high`for some configurations
# tp_prefill = 1, pp_prefill = 1
# x-axis: rate
# y-axis: attainment
# find all combination of tp_prefill, pp_prefill

configs = figure_11_vllm_high[['tp_prefill', 'pp_prefill']].drop_duplicates()
df = figure_11_vllm_high

for tp_prefill, pp_prefill in configs.values:
    config_df = df[
        (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill)
        ]
    # plot this inside a plotly plot
    fig.add_trace(go.Scatter(
        x=config_df['per_gpu_rate'], y=config_df['attainment'],
        mode='lines+markers', name=f"p{tp_prefill}{pp_prefill}-vllm"
    ))

# fig add title
fig.update_layout(
    title="Figure 11: Abalation Study (DistServe and vLLM)",
    xaxis_title="Per-GPU Rate (tokens/s)",
    yaxis_title="Attainment (%)",
    legend_title="Configuration"
)
fig.show()
fig.write_html("visual/figure_11.full.html")

# In[43]:


import plotly.graph_objects as go

fig = go.Figure()
configs = figure_11_vllm_high[['tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode']].drop_duplicates()

# Case 1: DistServe-High
df = figure_11_distserve_high
tp_prefill, pp_prefill, tp_decode, pp_decode = 4, 2, 1, 1
config_df = df[
    (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill) &
    (df['tp_decode'] == tp_decode) & (df['pp_decode'] == pp_decode)
    ]
# plot this inside a plotly plot
fig.add_trace(go.Scatter(
    x=config_df['per_gpu_rate'], y=config_df['attainment'],
    mode='lines+markers', name=f"disthigh-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}"
))

# Case 2: DistServe-Low
df = figure_11_distserve_low
tp_prefill, pp_prefill, tp_decode, pp_decode = 4, 1, 1, 1
config_df = df[
    (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill) &
    (df['tp_decode'] == tp_decode) & (df['pp_decode'] == pp_decode)
    ]
# plot this inside a plotly plot
fig.add_trace(go.Scatter(
    x=config_df['per_gpu_rate'], y=config_df['attainment'],
    mode='lines+markers', name=f"distlow-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}"
))

# Case 3: vLLM++
df = figure_11_vllm_high
tp_prefill, pp_prefill = 4, 1
config_df = df[
    (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill)
    ]
# plot this inside a plotly plot
fig.add_trace(go.Scatter(
    x=config_df['per_gpu_rate'], y=config_df['attainment'],
    mode='lines+markers', name=f"vllm++-p{tp_prefill}{pp_prefill}"
))

# Case 4: vLLM
df = figure_11_vllm_low
tp_prefill, pp_prefill = 4, 1
config_df = df[
    (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill)
    ]
# plot this inside a plotly plot
fig.add_trace(go.Scatter(
    x=config_df['per_gpu_rate'], y=config_df['attainment'],
    mode='lines+markers', name=f"vllm-p{tp_prefill}{pp_prefill}"
))

fig.update_layout(
    title="Figure 11: Abalation Study (DistServe and vLLM)<br>"
          "<sup>The figure shows that DistHigh > DistLow > vLLM++ > vLLM (vLLM++ and vLLM overlaps) </sup>",
    xaxis_title="Per-GPU Rate (tokens/s)",
    yaxis_title="Attainment (%)",
    legend_title="Configuration"
)
fig.show()
fig.write_html("visual/figure_11.html")


# In[ ]:
# Generate the matplotlib