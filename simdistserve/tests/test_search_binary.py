# get_distserve_configs(ModelTypes.opt_13b, 4, 8, True)
from itertools import product

from simdistserve.benchmarks.search_configs import get_distserve_configs
from simdistserve.constants import ModelTypes


def test_search_binary():
    for (
        model,
        num_node,
        num_gpu_per_node,
        is_high_affinity,
    ) in product(
        (ModelTypes.opt_13b, ModelTypes.opt_66b, ModelTypes.opt_175b),
        (1, 2, 3, 4),
        (1, 2, 4, 8),
        (True, False),
    ):
        configs = get_distserve_configs(model, num_node, num_gpu_per_node, is_high_affinity)
        # print(
        #     f"model={model}, num_node={num_node}, num_gpu_per_node={num_gpu_per_node}, "
        #     f"is_high_affinity={is_high_affinity}, num_configs={len(configs)}"
        # )
    pass


if __name__ == '__main__':
    test_search_binary()
