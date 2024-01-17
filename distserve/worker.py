# Adapted from github.com/vllm/worker/worker.py
import copy
import time
from typing import List, Tuple, Optional
import socket

import ray
import torch
import torch.distributed

from distserve.config import ModelConfig, CacheConfig, ParallelConfig
from distserve.request import Request, BatchedRequests
from distserve.utils import set_random_seed, cudaMemoryIpcHandle, Stage
from distserve.models import get_model_op
from distserve.utils import get_gpu_memory, set_random_seed, GB, MB
from distserve.logger import init_logger
from distserve.downloader import download_and_convert_weights

logger = init_logger(__name__)


@ray.remote(num_cpus=0, num_gpus=1)
class ParaWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache, the KV swap and executing the model on the GPU.
    In case of distributed inference, each worker is assigned a partition of
    the model.

    """

    def __init__(
        self,
        worker_id: int,
        stage: Stage,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig = ParallelConfig(),
        tensor_parallel_id: List[int] = None,
        pipeline_parallel_id: List[int] = None,
    ) -> None:
        self.worker_id = worker_id
        self.stage = stage
        self.model = None
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.tensor_parallel_id = tensor_parallel_id
        self.pipeline_parallel_id = pipeline_parallel_id
        self.gpu_id = ray.get_gpu_ids()[0]
        
        self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)
        
        # K/V cache on GPU
        self.k_cache = None
        self.v_cache = None
        # K/V swap on CPU
        self.k_swap = None
        self.v_swap = None
        # CUDA streams for swapping in and out
        self.swap_in_stream = torch.cuda.Stream()
        self.swap_out_stream = torch.cuda.Stream()
        # The swap_event_table, refer to block_manager.py for more details
        self.swap_event_table = {}
        # The latest swap event in each stream
        # Used when we need to wait for all swap events to finish
        self.latest_swap_in_event = None
        self.latest_swap_out_event = None
        # Statistics
        self.execution_time = 0.0
        self.blocked_swapping_time = 0.0

    def ready(self):
        """
        Ray functions queue inside one single actor to be executed in order.
        If ready is called, the actor is ready.
        """
        logger.info(f"Worker {self.stage}.#{self.worker_id} created on host {socket.gethostname()} and gpu #{self.gpu_id}")
        pass

    def init_model(self):
        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model_op(
            self.model_config, self.parallel_config, self.cache_config
        )
        self.model.init_communicator(self.tensor_parallel_id, self.pipeline_parallel_id)
        torch.cuda.synchronize()
        if self.model_config.use_dummy_weights:
            self.model.init_dummy_weights()
        else:
            path = download_and_convert_weights(self.model_config)
            self.model.load_weight(path)
        torch.cuda.synchronize()
        logger.info(f"(worker {self.stage}.#{self.worker_id}) model {self.model_config.model} loaded")

    def init_kvcache_and_swap(self, num_gpu_blocks, num_cpu_blocks) -> (cudaMemoryIpcHandle, cudaMemoryIpcHandle):
        """
        Allocate the K/V cache and swap.
        
        Return K/V cache's memory handle
        """
        # kv shape is [num_gpu_blocks, num_layers, num_local_heads, block_size, head_dim]
        # profile the GPU to get num_gpu_blocks
        kv_cache_shape = (
            num_gpu_blocks,
            self.model_config.get_num_layers(self.parallel_config),
            self.model_config.get_num_heads(self.parallel_config),
            self.cache_config.block_size,
            self.model_config.get_head_size(),
        )
        self.k_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        self.v_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        # kv swap is [num_cpu_blocks, num_layers, num_local_heads, block_size, head_dim]
        # We pin memory here in order to leverage cudaMemcpyAsync when swapping
        kv_swap_shape = (num_cpu_blocks,) + kv_cache_shape[1:]
        self.k_swap = torch.empty(
            kv_swap_shape, dtype=self.model_config.get_torch_dtype(), device="cpu", pin_memory=True
        )
        self.v_swap = torch.empty(
            kv_swap_shape, dtype=self.model_config.get_torch_dtype(), device="cpu", pin_memory=True
        )
        torch.cuda.synchronize()
        
        return torch.ops.block_migration_ops.get_ipc_mem_handle(self.k_cache), \
               torch.ops.block_migration_ops.get_ipc_mem_handle(self.v_cache)

    def _get_block_size_in_bytes(
        self,
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        # the shape of one slot in k/v cache is [num_layers, num_local_heads, block_size, head_dim]
        num_layers = model_config.get_num_layers(parallel_config)
        num_heads = model_config.get_num_heads(parallel_config)
        head_dim = model_config.get_head_size()

        key_cache_size = num_layers * num_heads * block_size * head_dim
        total = key_cache_size * 2
        dtype_size = model_config.get_dtype_size()
        return total * dtype_size

    @torch.inference_mode()
    def _profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # GPU and CPU blocks that can be allocated with the remaining free memory.

        # Profile memory usage with max_batch_size requests and the total
        # number of tokens equal to max_tokens_per_batch.
        total_gpu_memory = get_gpu_memory()
        peak_runtime_memory = (
            total_gpu_memory * 0.01
            + self.model_config.get_model_size_in_bytes(
                parallel_config=self.parallel_config
            )
        )
        logger.info(f"runtime peak memory: {peak_runtime_memory / GB:.3f} GB")
        logger.info(f"total GPU memory: {total_gpu_memory / GB:.3f} GB")
        block_size_in_bytes = self._get_block_size_in_bytes(
            block_size, self.model_config, self.parallel_config
        )
        logger.info(
            f"kv cache size for one token: {block_size_in_bytes / block_size / MB:.5f} MB"
        )
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_runtime_memory)
            // block_size_in_bytes
        )
        num_cpu_blocks = int(cpu_swap_space // block_size_in_bytes)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        logger.info(f"num_gpu_blocks: {num_gpu_blocks}")
        num_cpu_blocks = max(num_cpu_blocks, 0)
        logger.info(f"num_cpu_blocks: {num_cpu_blocks}")

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def step(
        self,
        request_ids: List[int],
        input_tokens_batched,
        first_token_indexes,
        block_table,
    ) -> Tuple[Optional[List[int]], int]:
        """Run one step of inference on the batch of requests."""

        start = time.time()
        # Check whether synchronization is necessary
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # We let the current stream wait for the swap event
                # This is non-blocking (It just stop the current stream instead
                # of chocking the CPU)
                self.swap_event_table[request_id].wait(torch.cuda.current_stream())
                self.swap_event_table.pop(request_id, None)
        self.blocked_swapping_time += time.time() - start

        start = time.time()
        # print(f"Worker {self.stage}.#{self.worker_id} Step begin")
        # run forward
        generated_tokens_ids = self.model.forward(
            input_tokens_batched,
            first_token_indexes,
            self.k_cache,
            self.v_cache,
            block_table,
        )
        self.execution_time += time.time() - start
        # print(f"Worker {self.stage}.#{self.worker_id} Step end")

        return generated_tokens_ids

    def register_kvcache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kvcache_ipc_mem_handles: List[List[Tuple[cudaMemoryIpcHandle, cudaMemoryIpcHandle]]]
    ):
        for pp_rank, stage_workers in enumerate(kvcache_ipc_mem_handles):
            for tp_rank, mem_handle in enumerate(stage_workers):
                tmp_parallel_config = copy.deepcopy(context_parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = pp_rank
                tmp_parallel_config.tensor_parallel_rank = tp_rank
                torch.ops.block_migration_ops.register_ipc_mem_handle(
                    kvcache_ipc_mem_handles[pp_rank][tp_rank][0],
                    kvcache_ipc_mem_handles[pp_rank][tp_rank][1],
                    self.model_config.get_num_layers(),
                    self.model_config.get_num_heads(),
                    tmp_parallel_config.to_list(),
                    self.parallel_config.to_list()
                )
                
        torch.cuda.synchronize()
                
    def migrate_blocks(
        self,
        context_block_indexes: List[int],
        context_parallel_config: ParallelConfig,
        decoding_block_indexes: List[int]
    ):
        torch.ops.block_migration_ops.migrate_blocks(
            context_parallel_config.pipeline_parallel_size,
            context_parallel_config.tensor_parallel_size,
            context_block_indexes,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_rank,
            self.parallel_config.tensor_parallel_rank,
            decoding_block_indexes,
            self.k_cache,
            self.v_cache
        )
        
    def swap_blocks(
        self,
        request_ids: List[int],
        source_block_ids: List[int],
        target_block_ids: List[int],
        is_swap_in: bool,
    ):
        """Swap some blocks between CPU and GPU
        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False
        """

        # print(f"Swap {source_block_ids} ({'CPU' if is_swap_in else 'GPU'}) to {target_block_ids} ({'GPU' if is_swap_in else 'CPU'})")
        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream

        # Record event
        event = torch.cuda.Event()
        event.record(stream)

        # Save that event
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # If we've issued another swapping operation before, we shall wait it
                # Pay attention to the difference between wait() and synchronize()
                self.swap_event_table[request_id].wait(stream)
            self.swap_event_table[request_id] = event
        if is_swap_in:
            self.latest_swap_in_event = event
        else:
            self.latest_swap_out_event = event

        # Swap
        with torch.cuda.stream(stream):
            torch.ops.swapping_ops.swap(
                source_block_ids,
                target_block_ids,
                is_swap_in,
                self.k_cache,
                self.v_cache,
                self.k_swap,
                self.v_swap,
            )

    def clear_request_resource(self, request_id: int):
        """Clear the resources associated with the request."""
        """This is called by LLMEngine when a request is finished or aborted"""
        # Clear the swap event table
        self.swap_event_table.pop(request_id, None)

    def clear_request_resource_batched(self, requests: List[Request]):
        """Clear the resources associated with the requests."""
        for request in requests:
            self.clear_request_resource(request.request_id)

    def wait_for_all_swap_in(self):
        """Wait for all swap in to finish"""
        if self.latest_swap_in_event is not None:
            self.latest_swap_in_event.synchronize()
            self.latest_swap_in_event = None

    def wait_for_all_swap_out(self):
        """Wait for all swap out to finish"""
        if self.latest_swap_out_event is not None:
            self.latest_swap_out_event.synchronize()
            self.latest_swap_out_event = None
