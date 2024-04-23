import json
from pathlib import Path

from simdistserve.benchmarks.simulate_dist import parse_args
from simdistserve.benchmarks.simulate_dist import run_experiment
import pandas as pd


def get_trace_file(file):
    with open(file, "r") as f:
        # read one line
        a = f.readline().strip()
        reqs = eval(a)
    c = [(
        r['prompt_len'],
        r['output_len'],
        r['issue_time'],
        r['ttft_ms'],
        r['tpot_ms'],
    ) for r in reqs]
    base_time = min([i[2] for i in c])
    c = sorted(c, key=lambda x: x[2])
    d = [(i[0], i[1], i[2] - base_time, i[3], i[4]) for i in c]
    return d


def get_vllm_comparison(
    target_ftl=200,
    target_tpot=100,
    N=300,
    rate=3,
):
    name = f'vllm-{N}-{rate}'
    tp_prefill, pp_prefill, tp_decode, pp_decode = 1, 1, 1, 1
    alignment_output_dir = Path("./alignment-output/")
    data_input_dir = Path("./data-ground-truth/") / 'opt-13b-sharegpt'
    ground_truth_input_dir = Path("./data-ground-truth/") / 'opt-13b-sharegpt-workload'
    data_input_path = str(
        (data_input_dir / f'{name}.exp').absolute()
    )
    original_data_trace = get_trace_file(data_input_path)

    original_data_trace_df = pd.DataFrame(
        original_data_trace, columns=['prompt_len', 'output_len', 'issue_time', 'ttft_ms', 'tpot_ms']
    )
    workload_path = str(
        (ground_truth_input_dir / f'{name}.exp').absolute()
    )
    output_path = str(
        (alignment_output_dir / f'{name}.latency.csv').absolute()
    )
    output_request_latency_path = str(
        (alignment_output_dir / f'{name}.latency.latency.csv').absolute()
    )
    with open(workload_path) as f:
        workload_tuples = json.load(f)

    args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--backend', 'vllm',
        '--prefill-containment', '90',  # P90
        '--prefill-target', '200',  # ms
        '--decode-containment', '90',  # P90
        '--decode-target', '100',  # ms
        '--model', 'opt_13b',
        '--workload', workload_path,
        '--output', output_path,
        '--output-request-latency', output_request_latency_path,
        '--rate', str(rate),
        '--N', str(N),
        '--pp-prefill', f'{pp_prefill}',
        '--pp-decode', f'{pp_decode}',
        '--tp-prefill', f'{tp_prefill}',
        '--tp-decode', f'{tp_decode}',

    ]
    outputs = {}
    args = parse_args(args_=args)
    p_contained, d_contained, latency_df = run_experiment(args, outputs=outputs)

    # Compare the latency of the original data and the simulated data
    a = outputs['per_request_latency_df']
    b = original_data_trace_df
    c = pd.concat([b, a, ], axis=1)
    c = c[['prompt_len', 'output_len', 'ttft_ms', 'first_token_latency', 'tpot_ms', 'tpot']]
    c['ftl_rerr'] = (c['first_token_latency'] - c['ttft_ms']) / c['ttft_ms']
    c['tpot_rerr'] = (c['tpot'] - c['tpot_ms']) / c['tpot_ms']

    # Compare the attention rate
    c['sim_attn'] = (c['first_token_latency'] < target_ftl) & (c['tpot'] < target_tpot)
    c['real_attn'] = (c['ttft_ms'] < target_ftl) & (c['tpot_ms'] < target_tpot)
    N = len(c)
    real_attn = c['real_attn'].sum() / N * 100
    sim_attn = c['sim_attn'].sum() / N * 100
    return real_attn, sim_attn


def get_distserve_comparison(
    target_ftl=200,
    target_tpot=100,
    N=300,
    rate=3,
):
    name = f'distserve-c21d11-{N}-{rate}'
    tp_prefill, pp_prefill, tp_decode, pp_decode = 2, 1, 1, 1
    alignment_output_dir = Path("./alignment-output/")
    data_input_dir = Path("./data-ground-truth/") / 'opt-13b-sharegpt'
    ground_truth_input_dir = Path("./data-ground-truth/") / 'opt-13b-sharegpt-workload'
    data_input_path = str(
        (data_input_dir / f'{name}.exp').absolute()
    )
    original_data_trace = get_trace_file(data_input_path)

    original_data_trace_df = pd.DataFrame(
        original_data_trace, columns=['prompt_len', 'output_len', 'issue_time', 'ttft_ms', 'tpot_ms']
    )
    workload_path = str(
        (ground_truth_input_dir / f'{name}.exp').absolute()
    )
    output_path = str(
        (alignment_output_dir / f'{name}.latency.csv').absolute()
    )
    output_request_latency_path = str(
        (alignment_output_dir / f'{name}.latency.latency.csv').absolute()
    )
    with open(workload_path) as f:
        workload_tuples = json.load(f)

    args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--backend', 'distserve',
        '--prefill-containment', '90',  # P90
        '--prefill-target', '200',  # ms
        '--decode-containment', '90',  # P90
        '--decode-target', '100',  # ms
        '--model', 'opt_13b',
        '--workload', workload_path,
        '--output', output_path,
        '--output-request-latency', output_request_latency_path,
        '--rate', f'3',
        '--N', '300',
        '--pp-prefill', f'{pp_prefill}',
        '--pp-decode', f'{pp_decode}',
        '--tp-prefill', f'{tp_prefill}',
        '--tp-decode', f'{tp_decode}',

    ]
    outputs = {}
    args = parse_args(args_=args)
    p_contained, d_contained, latency_df = run_experiment(args, outputs=outputs)

    # Compare the latency of the original data and the simulated data
    a = outputs['per_request_latency_df']
    b = original_data_trace_df
    c = pd.concat([b, a, ], axis=1)
    c = c[['prompt_len', 'output_len', 'ttft_ms', 'first_token_latency', 'tpot_ms', 'tpot']]
    c['ftl_rerr'] = (c['first_token_latency'] - c['ttft_ms']) / c['ttft_ms']
    c['tpot_rerr'] = (c['tpot'] - c['tpot_ms']) / c['tpot_ms']

    # Compare the attention rate
    c['sim_attn'] = (c['first_token_latency'] < target_ftl) & (c['tpot'] < target_tpot)
    c['real_attn'] = (c['ttft_ms'] < target_ftl) & (c['tpot_ms'] < target_tpot)
    N = len(c)
    real_attn = c['real_attn'].sum() / N * 100
    sim_attn = c['sim_attn'].sum() / N * 100
    return real_attn, sim_attn


results = []


for rate in [3, 4.5, 6, 7.5, 9, 10.5, 12]:
    N = 300
    real_attn, sim_attn = get_distserve_comparison(
        target_ftl=200,
        target_tpot=100,
        N=N,
        rate=rate
    )
    results.append({
        "real_attn": real_attn,
        "sim_attn": sim_attn,
        "rate": rate,
        "N": N,
        "backend": "distserve",
    })



for N, rate in [
    (100, 1), (100, 2), (150, 1.5), (200, 2),
    (250, 2.5), (300, 3), (350, 3.5), (400, 4),
]:
    real_attn, sim_attn = get_vllm_comparison(
        target_ftl=200,
        target_tpot=100,
        N=N,
        rate=rate
    )
    results.append({
        "real_attn": real_attn,
        "sim_attn": sim_attn,
        "rate": rate,
        "N": N,
        "backend": "vllm",
    })
    pass

df = pd.DataFrame(results)
rmse = (
           df['real_attn'] - df['sim_attn']
       ).apply(lambda x: x ** 2).mean() ** 0.5
df['diff'] = df['real_attn'] - df['sim_attn']
print(df.to_markdown(floatfmt=".2f"))
now = pd.Timestamp.now()
logf = open(
    f"/Users/mike/Project/DistServe/simdistserve/scratch/alignment/logs/"
    + now.strftime("%Y%m%d_%H%M%S") + ".txt", "w"
)
print(df.to_markdown(floatfmt=".2f"), file=logf)

t = "/Users/mike/Project/DistServe/simdistserve/estimators/profile_data/profiler-a100-80g.json"
with open(t) as f:
    data = json.load(f)
a = data['facebook/opt-13b']
print(a)
print(a, file=logf)

print(f"RMSE: {rmse:.2f}")
print(f"RMSE: {rmse:.2f}", file=logf)
