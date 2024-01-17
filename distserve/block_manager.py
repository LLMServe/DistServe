from typing import List, Callable
from enum import Enum
from dataclasses import dataclass

from distserve.config import ModelConfig, ParallelConfig, CacheConfig
from distserve.request import Request, BatchedRequests
from distserve.logger import init_logger
from distserve.utils import Stage

logger = init_logger(__name__)


class BlockLocation(Enum):
    """The location of a block"""

    GPU = "gpu"
    CPU = "cpu"


class BlockManager:
    """A Block Manager that maintains the key-value cache in block-level"""

    """For subroutines and algorithms related to swapping, please refer to
    the big comment block above swap_requests()"""

    def __init__(
        self,
        stage: Stage,
        max_num_gpu_blocks: int,
        max_num_cpu_blocks: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        engine_remote_call_all_workers_async: Callable,
    ):
        self.stage = stage
        self.max_num_gpu_blocks = max_num_gpu_blocks
        self.max_num_cpu_blocks = max_num_cpu_blocks
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.engine_remote_call_all_workers_async = engine_remote_call_all_workers_async

        self.free_gpu_blocks_list = list(range(max_num_gpu_blocks))
        self.free_cpu_blocks_list = list(range(max_num_cpu_blocks))

        self.swapping_gpu_blocks_list = []
        self.swapping_cpu_blocks_list = []

        # request_id => [block0_id, block1_id, ...]
        # If the blocks of the request are on GPU, then block0_id, block1_id are GPU block
        # ids, and vice versa
        self.block_table = {}

        # request_id => BlockLocation
        self.request_location = {}

    def get_num_avail_gpu_blocks(self) -> int:
        """Get the number of available GPU blocks"""
        return len(self.free_gpu_blocks_list) + len(self.swapping_gpu_blocks_list)

    def get_num_avail_cpu_blocks(self) -> int:
        """Get the number of available CPU blocks"""
        return len(self.free_cpu_blocks_list) + len(self.swapping_cpu_blocks_list)

    def _get_free_blocks(self, num_blocks: int, location: BlockLocation) -> List[int]:
        """Get free blocks from the free block pool indicated by `location`"""
        """When `location` is GPU, the returned blocks are on GPU, and vice versa"""
        assert location in [BlockLocation.GPU, BlockLocation.CPU]
        if location == BlockLocation.GPU:
            num_avail_blocks = self.get_num_avail_gpu_blocks()
            assert (
                num_avail_blocks >= num_blocks
            ), f"not enough free blocks on GPU, requested {num_blocks}, available {num_avail_blocks}"
            if len(self.free_gpu_blocks_list) < num_blocks:
                # Need to "flush" self.swapping_gpu_blocks_list, i.e. make sure all
                # swapping-out operations have finished, thus blocks in self.swapping_gpu_blocks_list
                # can be moved to self.free_gpu_blocks_list
                self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
                self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
                self.swapping_gpu_blocks_list = []
            blocks = self.free_gpu_blocks_list[:num_blocks]
            self.free_gpu_blocks_list = self.free_gpu_blocks_list[num_blocks:]
        else:
            num_avail_blocks = self.get_num_avail_cpu_blocks()
            assert (
                num_avail_blocks >= num_blocks
            ), f"not enough free blocks on CPU, requested {num_blocks}, available {num_avail_blocks}"
            if len(self.free_cpu_blocks_list) < num_blocks:
                # Need to "flush" self.swapping_cpu_blocks_list, i.e. make sure all
                # swapping-in operations have finished, thus blocks in self.swapping_cpu_blocks_list
                # can be moved to self.free_cpu_blocks_list
                self.engine_remote_call_all_workers_async("wait_for_all_swap_in")
                self.free_cpu_blocks_list += self.swapping_cpu_blocks_list
                self.swapping_cpu_blocks_list = []
            blocks = self.free_cpu_blocks_list[:num_blocks]
            self.free_cpu_blocks_list = self.free_cpu_blocks_list[num_blocks:]
        return blocks

    def get_allocated_num_blocks(self, request_id: int) -> int:
        """Get the number of allocated blocks for a request"""
        return len(self.block_table.get(request_id, []))

    def get_location(self, request_id: int) -> BlockLocation:
        """Get the kvcache blocks location of a request"""
        return self.request_location.get(request_id, None)

    def get_num_blocks_needed(self, request: Request):
        """Get the number of blocks needed for a request"""
        return (
            request.get_input_len()
            + request.get_output_len()
            + self.cache_config.block_size
            - 1
        ) // self.cache_config.block_size

    def get_num_append_blocks_needed(self, request: Request) -> int:
        """Get the number of blocks needed for a request already in GPU"""
        assert (
            self.request_location[request.request_id] == BlockLocation.GPU
        ), f"request {request.request_id} is not on GPU when calling get_num_append_blocks_needed"
        num_blocks_cur = len(self.block_table[request.request_id])
        num_blocks_needed = self.get_num_blocks_needed(request)
        return num_blocks_needed - num_blocks_cur

    def allocate_blocks(self, request: Request):
        """Allocate blocks for a request"""
        # Make sure the request is not already allocated or its blocks are on GPU
        assert (
            request.request_id not in self.block_table
            or self.request_location.get(request.request_id, None) == BlockLocation.GPU
        ), f"request {request.request_id} is currently on CPU. Please migrate it to GPU before allocating ore blocks"

        num_blocks_needed = self.get_num_blocks_needed(request)
        if request.request_id not in self.block_table:
            # This request has not been allocated before
            self.block_table[request.request_id] = self._get_free_blocks(
                num_blocks_needed, BlockLocation.GPU
            )
            self.request_location[request.request_id] = BlockLocation.GPU
        else:
            assert self.request_location[request.request_id] == BlockLocation.GPU
            num_blocks_cur = len(self.block_table[request.request_id])
            if num_blocks_cur < num_blocks_needed:
                self.block_table[request.request_id] += self._get_free_blocks(
                    num_blocks_needed - num_blocks_cur, BlockLocation.GPU
                )

    def allocate_blocks_batched(self, batch_requests: BatchedRequests):
        """Allocate blocks for a batch of requests"""
        for request in batch_requests.requests:
            self.allocate_blocks(request)

    def free_blocks(self, request_id: int):
        """Free blocks for a request"""
        assert request_id in self.block_table, f"request {request_id} not allocated"
        if self.request_location[request_id] == BlockLocation.GPU:
            self.free_gpu_blocks_list += self.block_table.pop(request_id)
        else:
            self.free_cpu_blocks_list += self.block_table.pop(request_id)
        self.request_location.pop(request_id)

    def free_blocks_batched(self, requests: List[Request]):
        """Free blocks for a batch of requests"""
        for request in requests:
            self.free_blocks(request.request_id)

    def get_block_table(self, request_id: int) -> List[int]:
        """Get the block table for a request"""
        assert request_id in self.block_table, f"request {request_id} not allocated"
        return self.block_table[request_id]
    
    def get_partial_block_table(self, request_ids: List[int]) -> List[List[int]]:
        """Get the block table for a batch of requests"""
        block_table = []
        for request_id in request_ids:
            assert request_id in self.block_table, f"request {request_id} not allocated"
            block_ids = self.block_table.get(request_id, [])
            block_table.append(block_ids)
        return block_table

    def __repr__(self) -> str:
        return (
            f"BlockManager(max_num_gpu_blocks={self.max_num_gpu_blocks}, "
            f"max_num_cpu_blocks={self.max_num_cpu_blocks}, "
            f"blocksize={self.cache_config.block_size})"
        )

    def print_block_usage(self):
        num_cpu_blocks_used = (
            self.max_num_cpu_blocks
            - len(self.free_cpu_blocks_list)
            - len(self.swapping_cpu_blocks_list)
        )
        logger.info(
            f"({self.stage}) CPU blocks: {num_cpu_blocks_used} / {self.max_num_cpu_blocks} "
            f"({num_cpu_blocks_used / self.max_num_cpu_blocks * 100:.2f}%) "
            f"used, ({len(self.swapping_cpu_blocks_list)} swapping in)"
        )
        num_gpu_blocks_used = (
            self.max_num_gpu_blocks
            - len(self.free_gpu_blocks_list)
            - len(self.swapping_gpu_blocks_list)
        )
        logger.info(
            f"({self.stage}) GPU blocks: {num_gpu_blocks_used} / {self.max_num_gpu_blocks} "
            f"({num_gpu_blocks_used / self.max_num_gpu_blocks * 100:.2f}%) "
            f"used, ({len(self.swapping_gpu_blocks_list)} swapping out)"
        )

    """The following methods are used for swapping
    We use the term "swap in" to mean moving blocks from CPU to GPU, and
    "swap out" to mean moving blocks from GPU to CPU.

    The followings explain our logic and code layout for swapping:

    # Swapping

    ## Overview

    We use swap_in as the example here. Swapping-out is similar.

    The scheduler calls LLMEngine.swap_in_request, which is a thin wrapper
    around BlockManager.swap_in_requests. The latter allocate blocks (i.e. GPU
    blocks when swapping in and CPU blocks when swapping out) for the requests
    and then call ParaWorker.swap_blocks on every worker.

    scheduler -> LLMEngine.swap_in_requests ->
        BlockManager.swap_in_requests -> ParaWorker.swap_blocks
    
    ## Synchonization

    We need to deal with two types of synchronization:

    Requirement 1: When calling step(), we need to ensure that all blocks that
        are related to the request are on GPU, and have finished swapping if
        we've issued a swap-in operation on them before.

    Requirement 2: When swapping in/out, we need to ensure that the target blocks
        (i.e. blocks that we are copying to) are free. If there were another
        swapping operation which use them as source, we must wait for that operation
        to finish.
    
    Requirement 3: When swapping in/out, the source blocks must be ready, i.e.
        if we have issued swapping-in operation on request #0 before and now I
        want to swap it out, then the swap-in operation must have finished.

    After every swapping in/out operation, we push a CUDA event into the corresponding
    CUDA stream. We maintain a dict called `swap_event_table`, which maps request_ids,
    to the latest cuda event related to that request.

    When calling step(), we iterate through all requests in the batch, retrieve
    the CUDA event from `swap_event_table`, and wait for it to finish. This ensures
    Requirement 1.

    When swapping in/out, we call event.wait() on the corresponding swap out/in
    event and use the corresponding stream (if we are swapping-in then it is the
    swap-in stream, vice versa) as the argument. Wait() makes all future work
    submitted to the given stream wait for this event, so the swapping operation
    that I'm going to issue will wait for the previous swapping operation to finish.
    This ensures Requirement 3.

    For Requirement 2, a possible solution is to mark the CUDA event of every block
    and wait for them to finish. However, this is not efficient. Instead, for
    each device (CPU/GPU), we maintain two lists: `free_blocks_list` and
    `swapping_blocks_list`. The former contains blocks that are we are sure to
    be free (it does not contain any useful data), and the latter contains blocks
    which we have issued a swapping operation from but we are not sure if the operation
    has finished.
    
    When we want to allocate a block, we first check if there are enough free blocks
    in `free_blocks_list`. If unfortunately this is not the case, we let all
    workers to finish all operations in `swapping_blocks_list` and then move
    all blocks in `swapping_blocks_list` to `free_blocks_list`. This ensures
    Requirement 2 and the overhead is small.

    ## Further Optimization

    To optimize more aggressively, consider the following idea: if we mark every
    CUDA event in the same stream with an increasing id, then if "The id associated
    with the most synced CUDA event" is larger than "The id associated with the
    CUDA event that we are waiting for", then we can skip the waiting.

    Currently we do not implement this idea because we want to make sure the
    correctness of the code first. We will implement this idea in the future.
    """

    def swap_requests(self, requests: List[Request], is_swap_in: bool):
        """Swap blocks for a batch of requests
        If `is_swap_in` is True, then swap in blocks from CPU to GPU, and vice versa
        """
        cur_location = BlockLocation.CPU if is_swap_in else BlockLocation.GPU
        target_location = BlockLocation.GPU if is_swap_in else BlockLocation.CPU
        source_block_ids = []  # block ids on cur_location
        target_block_ids = []  # block ids on target_location
        for request in requests:
            assert (
                request.request_id in self.block_table
            ), f"request {request.request_id} not allocated"
            assert (
                self.request_location[request.request_id] == cur_location
            ), f"request {request.request_id} is on {target_location} now"
            old_block_ids = self.block_table[request.request_id]
            new_block_ids = self._get_free_blocks(len(old_block_ids), target_location)
            source_block_ids += old_block_ids
            target_block_ids += new_block_ids
            self.block_table[request.request_id] = new_block_ids
            self.request_location[request.request_id] = target_location
            if cur_location == BlockLocation.CPU:
                self.swapping_cpu_blocks_list += old_block_ids
            else:
                self.swapping_gpu_blocks_list += old_block_ids
        self.engine_remote_call_all_workers_async(
            "swap_blocks", requests, source_block_ids, target_block_ids, is_swap_in
        )

    def swap_in_requests(self, requests: List[Request]):
        """Swap in blocks for a batch of requests"""
        self.swap_requests(requests, is_swap_in=True)

    def swap_out_requests(self, requests: List[Request]):
        """Swap out blocks for a batch of requests"""
        self.swap_requests(requests, is_swap_in=False)

    def is_all_requests_on_gpu(self, requests: BatchedRequests):
        """Check if all requests in a batch are on GPU"""
        for request in requests.requests:
            if self.request_location[request.request_id] == BlockLocation.CPU:
                return False
        return True