import contextlib
import itertools
import time
from functools import reduce
from itertools import chain
from typing import List

_verbose = True


@contextlib.contextmanager
def set_debug_verbosity(value):
    global _verbose
    old_val = _verbose
    _verbose = value
    yield
    _verbose = old_val


# TODO: Not thread safe.
def debugf(*args, **kwargs):
    if _verbose:
        print(*args, **kwargs)
    return None


def set_next_worker(x, y):
    x.next_worker = y
    return y


@contextlib.contextmanager
def timeit():
    start = time.time()
    yield
    end = time.time()
    print(f"Time elapsed: {end - start:.4f}s")


def grid_search(grid):
    for values in itertools.product(*grid.values()):
        yield dict(zip(grid.keys(), values))


def grid_total_job(grid):
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


def cyclic_chain(workers: 'List[Worker]'):
    return reduce(set_next_worker, chain(workers, (workers[0],)))


def irange(*args):
    """Return [1, x]"""
    if len(args) == 1:
        x = args[0]
        return range(1, x + 1)
    if len(args) == 2:
        x, y = args
        return range(x, y + 1)
    if len(args) == 3:
        x, y, z = args
        return range(x, y + 1, z)
    raise ValueError(f"args={args}")
