import marshal
import random
from contextlib import contextmanager
from os import PathLike
from typing import List

import numpy as np

from simdistserve.base.request import Request


@contextmanager
def numpy_seed(seed):
    if seed is None:
        yield
        return

    state = np.random.get_state()  # Save the current state
    try:
        np.random.seed(seed)  # Set the new seed
        yield
    finally:
        np.random.set_state(state)  # Restore the original state


def convert_interarrival_to_absolutearrival(x: 'List[float]'):
    y = 0
    result = []
    for t in x:
        y += t
        result.append(y)
    return result


def convert_absolutearrival_to_interarrival(x: 'List[float]'):
    result = [0]
    for i in range(1, len(x)):
        delay = x[i] - x[i - 1]
        delay *= 1000
        result.append(delay)
    return result


def convert_pd_pair_to_request(pairs: 'list[tuple[int, int]]') -> 'List[Request]':
    result = []
    for i, (prefill_length, output_length) in enumerate(pairs):
        r = Request(
            env=None,
            req_id=i,
            prefill_length=prefill_length,
            output_lens=output_length,
        )
        result.append(r)
    return result


class NamedList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = None

    def set_name(self, name):
        self.name = name
        return self


def get_fixed_interarrival(n, delay: float):
    """Fixed interarrival delay (ms)."""
    assert n > 0
    data = [0] + [delay] * (n - 1)
    result = NamedList(data).set_name(f'fixed(delay={delay})')
    return result


def get_poisson_interarrival(n: int, rate: float, seed=None):
    """
    Return the list of inter-arrival time (ms).
    Note: the 0-th element is 0 - the first request always have 0-delay.
    See the processing function for why.
    """
    return get_gamma_interarrival(n, rate, 1, seed=seed)


def get_gamma_interarrival(n: int, rate: float, cv: float, seed=None):
    assert n > 0
    with numpy_seed(seed):
        shape = 1 / (cv * cv)
        scale = cv * cv / rate
        result = np.random.gamma(shape, scale, size=n - 1)
        result *= 1000

    data = [0] + list(result)
    result = NamedList(data).set_name(f'gamma(rate={rate}, cv={cv}, seed={seed})')
    return result


def sample_requests(dataset_path: PathLike, num_prompts: int) -> 'list[(int, int)]':
    """
    sample_requests: Sample the given number of requests from the dataset.
    :param dataset_path: The path to the dataset.
    :param num_prompts: The number of prompts to sample.
    :return: A list of prompts and decode lengths.
    """
    with open(dataset_path, 'rb') as f:
        # {dataset_name:str, data:list[(prompt:str, prompt_len:int, output_len:int)]}
        dataset = marshal.load(f)
    dataset = dataset['data']
    result = random.sample(dataset, num_prompts)
    result = [(p, d) for (_, p, d) in result]

    # Generate requests
    requests = [
        Request(
            env=None,
            req_id=i,
            prefill_length=prefill_length,
            output_lens=output_length,
        )
        for i, (prefill_length, output_length) in enumerate(result)
    ]
    return requests
