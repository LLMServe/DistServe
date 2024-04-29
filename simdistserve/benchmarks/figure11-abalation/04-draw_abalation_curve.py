import json

import matplotlib.pyplot as plt

fontsize = 18
markersize = 8
att_target = 90
ylabel = "SLO Attainment (%)"

plt.rcParams.update({'font.size': fontsize})
plt.figure(figsize=(10, 3))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Draw ablation curve')
    # rates
    parser.add_argument('--rates', type=str, default="[1,2,3,4,5]")
    # SLO scales
    parser.add_argument('--slo_scales', type=str, default="[0.4,0.6,0.8,1.0,1.2]")
    return parser.parse_args()


args = parse_args()
rates = eval(args.rates)
SLO_scales = eval(args.slo_scales)

## rate
plt.subplot(1, 2, 1)
xlabel = "Per-GPU Rate (req/s)"
with open("figure/figure_11a.json") as f:
    data = json.load(f)
    distllm_optimal_SLO_att = data['dist++']
    distllm_real_SLO_att = data['dist']
    vllm_plus_SLO_att = data['vllm++']
    vllm_SLO_att = data['vllm']

plt.plot(rates, distllm_optimal_SLO_att, label='DistLLM-High', marker="o", markersize=markersize)
plt.plot(rates, distllm_real_SLO_att, label='DistLLM-Low', marker="o", markersize=markersize)
plt.plot(rates, vllm_plus_SLO_att, label='vLLM++', marker="o", markersize=markersize)
plt.plot(rates, vllm_SLO_att, label='vLLM', marker="o", markersize=markersize)
plt.plot([rates[0], rates[-1]], [att_target, att_target], '--')
# plt.xticks(rates, rates)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8], [0, 0.2, 0.4, 0.6, 0.8])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim(0, 105)
plt.yticks([0, 50, 100], [0, 50, 100])

## SLO Scale
plt.subplot(1, 2, 2)
xlabel = "SLO Scale"
with open("figure/figure_11b.json") as f:
    data = json.load(f)
    distllm_optimal_SLO_att = data['dist++']
    distllm_real_SLO_att = data['dist']
    vllm_plus_SLO_att = data['vllm++']
    vllm_SLO_att = data['vllm']

plt.plot(SLO_scales, distllm_optimal_SLO_att, label='DistLLM-High', marker="o", markersize=markersize)
plt.plot(SLO_scales, distllm_real_SLO_att, label='DistLLM-Low', marker="o", markersize=markersize)
plt.plot(SLO_scales, vllm_plus_SLO_att, label='vLLM++', marker="o", markersize=markersize)
plt.plot(SLO_scales, vllm_SLO_att, label='vLLM', marker="o", markersize=markersize)
plt.plot([SLO_scales[0], SLO_scales[-1]], [att_target, att_target], '--')
plt.xticks(SLO_scales, reversed([f"{i:.1f}" for i in SLO_scales]))
plt.xlabel(xlabel)
plt.ylim(0, 105)
plt.yticks([0, 50, 100], [0, 50, 100])

# plt.legend(frameon=False, bbox_to_anchor = (0.75, 1.3, 0, 0), ncol=2,
#            bbox_transform = plt.gcf().transFigure, columnspacing=1)

plt.legend(frameon=False, bbox_to_anchor=(0.95, 1.1, 0, 0), ncol=4,
           bbox_transform=plt.gcf().transFigure, columnspacing=1)
plt.savefig("figure/ablation.png", bbox_inches="tight")
plt.savefig("figure/ablation.pdf", bbox_inches="tight")
