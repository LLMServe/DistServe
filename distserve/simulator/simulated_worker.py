"""
A forged worker that imitates the behavior of the real worker without using any GPUs
"""
import time
from typing import List, Tuple
import asyncio

from .config import SimulatorConfig
from .estimator import Estimator

from distserve.config import ModelConfig, CacheConfig, ParallelConfig
from distserve.request import Request, BatchedRequests
from distserve.utils import Stage, set_random_seed, GB, MB
from distserve.logger import init_logger

logger = init_logger(__name__)

RAY_OVERHEAD_BLOCKING = 1    # In ms
RAY_OVERHEAD_NONBLOCKING = 4 # In ms

class SimulatedWorker:
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
        simulator_config: SimulatorConfig,
        parallel_config: ParallelConfig = ParallelConfig(),
        tensor_parallel_id: List[int] = None,   # Although the type is list[int], it is actually a NCCL unique ID
        pipeline_parallel_id: List[int] = None, # Same as above
    ) -> None:
        self.worker_id = worker_id
        self.stage = stage
        self.model = None
        self.model_config = model_config
        self.simulator_config = simulator_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.tensor_parallel_id = tensor_parallel_id
        self.pipeline_parallel_id = pipeline_parallel_id
        self.estimator = Estimator(
            self.simulator_config.profiler_data_path,
            self.model_config.model,
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size
        )
        
    def ready(self):
        return

    def init_model(self):
        return

    def init_kvcache_and_swap(self, num_gpu_blocks, num_cpu_blocks):
        return

    def _get_block_size_in_bytes(
        self,
    ) -> int:
        # the shape of one slot in k/v cache is [num_layers, num_local_heads, block_size, head_dim]
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_heads = self.model_config.get_num_heads(self.parallel_config)
        head_dim = self.model_config.get_head_size()

        key_cache_size = num_layers * num_heads * self.cache_config.block_size * head_dim
        total = key_cache_size * 2
        dtype_size = self.model_config.get_dtype_size()
        return total * dtype_size

    def _profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        peak_runtime_memory = (
            self.simulator_config.gpu_mem_size_gb * GB * 0.01
            + self.model_config.get_model_size_in_bytes(
                parallel_config=self.parallel_config
            )
        )
        logger.info(f"Total GPU memory: {self.simulator_config.gpu_mem_size_gb} GB")
        logger.info(f"Runtime peak memory (est.): {peak_runtime_memory / GB:.3f} GB")
        block_size_in_bytes = self._get_block_size_in_bytes()
        logger.info(
            f"kv cache size for one token: {block_size_in_bytes / block_size / MB:.5f} MB"
        )
        num_gpu_blocks = int(
            (self.simulator_config.gpu_mem_size_gb*GB * gpu_memory_utilization - peak_runtime_memory)
            // block_size_in_bytes
        )
        num_cpu_blocks = int(cpu_swap_space // block_size_in_bytes)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        logger.info(f"num_gpu_blocks: {num_gpu_blocks}")
        num_cpu_blocks = max(num_cpu_blocks, 0)
        logger.info(f"num_cpu_blocks: {num_cpu_blocks}")

        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    async def _simulate_ray_overhead(self):
        time.sleep(RAY_OVERHEAD_BLOCKING / 1000.0)
        await asyncio.sleep(RAY_OVERHEAD_NONBLOCKING / 1000.0)
        return
    
    async def step(
        self,
        request_ids: List[int],
        input_tokens_batched,
        first_token_indexes,
        block_table,
    ) -> List[int]:
        """Run one step of inference on the batch of requests."""
        await self._simulate_ray_overhead()
        batch_size = len(request_ids)
        generated_tokens_ids = [102 for _ in range(batch_size)] # 102 is token "a"
        if self.stage == Stage.CONTEXT:
            estimated_time = self.estimator.estimate_prefill_time_ms(
                sum([len(tokens) for tokens in input_tokens_batched]),
                sum([len(tokens)**2 for tokens in input_tokens_batched])
            )
        else:
            estimated_time = self.estimator.estimate_decoding_time_ms(
                sum([num_previous_tokens+1 for num_previous_tokens in first_token_indexes]),
                batch_size
            )
        await asyncio.sleep(estimated_time / 1000.0)

        return generated_tokens_ids

    def register_kvcache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kvcache_ipc_mem_handles
    ):
        return
                
    async def migrate_blocks(
        self,
        context_block_indexes: List[int],
        context_parallel_config: ParallelConfig,
        decoding_block_indexes: List[int]
    ):
        await self._simulate_ray_overhead()
        estimated_time_ms = self._get_block_size_in_bytes() * len(context_block_indexes) / (600*GB) * 1000
        estimated_time_ms += 0.1
        await asyncio.sleep(estimated_time_ms / 1000.0)
        return
        
    async def swap_blocks(
        self,
        request_ids: List[int],
        source_block_ids: List[int],
        target_block_ids: List[int],
        is_swap_in: bool,
    ):
        await self._simulate_ray_overhead()
        estimated_time_ms = self._get_block_size_in_bytes() * len(source_block_ids) / (32*GB) * 1000
        await asyncio.sleep(estimated_time_ms / 1000.0)
        return

    async def clear_request_resource(self, request_id: int):
        """Clear the resources associated with the request."""
        """This is called by LLMEngine when a request is finished or aborted"""
        await self._simulate_ray_overhead()
        return

    async def clear_request_resource_batched(self, requests: List[Request]):
        """Clear the resources associated with the requests."""
        await self._simulate_ray_overhead()
        return

    async def wait_for_all_swap_in(self):
        """Wait for all swap in to finish"""
        await self._simulate_ray_overhead()
        return

    async def wait_for_all_swap_out(self):
        """Wait for all swap out to finish"""
        await self._simulate_ray_overhead()
        return
