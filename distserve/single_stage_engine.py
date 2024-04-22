import time, copy
from typing import Callable, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import asyncio

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

import torch

from distserve.logger import init_logger
from distserve.config import (
    ModelConfig,
    ParallelConfig,
    CacheConfig,
)
from distserve.request import (
    Request, 
    BatchedRequests,
    MigratingRequest
)
from distserve.utils import Counter, cudaMemoryIpcHandle, Stage
from distserve.lifetime import LifetimeEvent, LifetimeEventType
from distserve.tokenizer import get_tokenizer
from distserve.block_manager import BlockManager
from distserve.worker import ParaWorker
from distserve.context_stage_scheduler import ContextStageSchedConfig, ContextStageScheduler, get_context_stage_scheduler
from distserve.decoding_stage_scheduler import DecodingStageSchedConfig, DecodingStageScheduler, get_decoding_stage_scheduler
from distserve.simulator.config import SimulatorConfig
from distserve.simulator.simulated_worker import SimulatedWorker
from distserve.simulator.utils import Barrier

logger = init_logger(__name__)

# Sleep for this many seconds when there is no request in ContextStageLLMEngine.step()
# We need to sleep for a while because the whole program is a asyncio-based,
# event driven, single thread program. We save some CPU time for other coroutines.
SLEEP_WHEN_CONTEXT_NO_REQUEST = 0.003

# Sleep for this many seconds when there is no request in DecodingStageLLMEngine.step()
SLEEP_WHEN_DECODING_NO_REQUEST = 0.003

# Sleep for this many seconds in each event loop, useful for debugging
SLEEP_IN_EACH_EVENT_LOOP = 0

# Print engine status every this many seconds
PRINT_STATUS_INTERVAL = 1

class StepOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(self, request: Request, new_token: str, new_token_id: int):
        self.request = request
        self.request_id = request.request_id
        self.prompt = request.prompt
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = request.is_finished

    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )

    
class SingleStageLLMEngine(ABC):
    """
    SingleStageLLMEngine: An LLMEngine that runs either the context stage or the decoding stage.
    
    This class is the base class for ContextStageLLMEngine and DecodingStageLLMEngine.
    """
    @abstractmethod
    def _get_scheduler(self) -> ContextStageScheduler | DecodingStageScheduler:
        raise NotImplementedError()
    
    def _free_request_resources(self, request_id: int) -> None:
        self.block_manager.free_blocks(request_id)
        self._remote_call_all_workers_async("clear_request_resource", request_id)
    
    def __init__(
        self,
        stage: Stage,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig | DecodingStageSchedConfig,
        simulator_config: SimulatorConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],   # The LLMEngine's callback function when a new StepOutput of a particular request is generated
        engine_on_new_lifetime_event_callback: Optional[Callable[[int, LifetimeEvent, bool], None]] = None,   # The LLMEngine's callback function when a new LifetimeEvent of a particular request is generated
    ):
        self.stage = stage
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sched_config = sched_config
        self.simulator_config = simulator_config
        self.engine_on_new_step_output_callback = engine_on_new_step_output_callback
        self.engine_on_new_lifetime_event_callback = engine_on_new_lifetime_event_callback

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )

        self.placement_groups = placement_groups
        
        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers = []
    
    async def initialize(self):
        """Initialize workers, load models and initialize k/v cache
        
        We seperate this function from __init__ because we want to run it in an async way
        to enable parallel initialization between Engines.
        """
        logger.info(f"Initializing {self.stage.name} workers")
        await self._init_workers()
        
        logger.info(f"Initializing {self.stage.name} models")
        await self._init_model()
        
        logger.info(f"Initializing {self.stage.name} kvcaches")
        self.num_gpu_blocks, self.num_cpu_blocks = await self._init_kvcache()

        self.block_manager = BlockManager(
            self.stage,
            self.num_gpu_blocks,
            self.num_cpu_blocks,
            self.model_config,
            self.parallel_config,
            self.cache_config,
            self._remote_call_all_workers_async,
        )
        
        self.scheduler: ContextStageScheduler | DecodingStageScheduler = self._get_scheduler()

        logger.info(f"Scheduler: {self.scheduler}")
        logger.info(f"Block manager: {self.block_manager}")


    async def _init_workers(self):
        """
        for each pipeline stage, create tensor_parallel_size workers
        each worker will be assigned a GPU
        the worker will be placed in the corresponding placement group
        """
        logger.info("Initializing workers")
        
        if self.simulator_config.is_simulator_mode:
            for i in range(self.parallel_config.pipeline_parallel_size):
                workers = []
                tensor_parallel_barrier = Barrier(self.parallel_config.tensor_parallel_size)
                for j in range(self.parallel_config.tensor_parallel_size):
                    tmp_parallel_config = copy.deepcopy(self.parallel_config)
                    tmp_parallel_config.pipeline_parallel_rank = i
                    tmp_parallel_config.tensor_parallel_rank = j
                    worker = SimulatedWorker(
                        worker_id=(i*self.parallel_config.tensor_parallel_size+j),
                        stage=self.stage,
                        model_config=self.model_config,
                        cache_config=self.cache_config,
                        simulator_config=self.simulator_config,
                        parallel_config=tmp_parallel_config,
                        tensor_parallel_barrier=tensor_parallel_barrier,
                    )
                    workers.append(worker)
                self.workers.append(workers)
        
        else:
            layer_per_placement_group = self.model_config.get_num_layers() // len(self.placement_groups)
            layer_per_pp = self.model_config.get_num_layers(self.parallel_config)
            pp_per_placement_group = layer_per_placement_group // layer_per_pp
            
            pp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
            
            init_handlers = []
            for i in range(self.parallel_config.pipeline_parallel_size):
                workers = []
                placement_group_index = i // pp_per_placement_group
                tp_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
                cur_placement_group = self.placement_groups[placement_group_index]
                for j in range(self.parallel_config.tensor_parallel_size):
                    tmp_parallel_config = copy.deepcopy(self.parallel_config)
                    tmp_parallel_config.pipeline_parallel_rank = i
                    tmp_parallel_config.tensor_parallel_rank = j
                    worker = ParaWorker.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=cur_placement_group
                        )
                    ).remote(
                        worker_id=(i*self.parallel_config.tensor_parallel_size+j),
                        stage=self.stage,
                        model_config=self.model_config,
                        cache_config=self.cache_config,
                        parallel_config=tmp_parallel_config,
                        pipeline_parallel_id=pp_id,
                        tensor_parallel_id=tp_id,
                    )
                    workers.append(worker)
                    init_handlers.append(worker.ready.remote())
                self.workers.append(workers)
                
            await asyncio.wait(init_handlers)

    async def _init_model(self):
        """
        init model by call init_model() on all workers
        """
        handlers = self._remote_call_all_workers_async("init_model")
        await asyncio.wait(handlers)

    async def _init_kvcache(self):
        """
        Profile available blocks and initialize k/v cache on all workers
        """
        logger.info("Profiling available blocks")
        if self.simulator_config.is_simulator_mode:
            num_gpu_blocks, num_cpu_blocks = self.workers[0][0]._profile_num_available_blocks(
                self.cache_config.block_size,
                self.cache_config.gpu_memory_utilization,
                self.cache_config.cpu_swap_space,
            )
        else:
            num_gpu_blocks, num_cpu_blocks = await self.workers[0][0]._profile_num_available_blocks.remote(
                self.cache_config.block_size,
                self.cache_config.gpu_memory_utilization,
                self.cache_config.cpu_swap_space,
            )
            
        logger.info(f"Profiling result: num_gpu_blocks: {num_gpu_blocks}, num_cpu_blocks: {num_cpu_blocks}")
        if self.stage == Stage.CONTEXT:
            # Do not set to 0 to avoid division by 0
            logger.info(f"The engine performs context stage, setting num_cpu_blocks to 1")
            num_cpu_blocks = 1
        logger.info("Allocating kv cache")
        kv_cache_mem_handles_1d = await asyncio.gather(*self._remote_call_all_workers_async(
            "init_kvcache_and_swap", num_gpu_blocks, num_cpu_blocks
        ))
        
        # Gather the address of kv cache for block migration
        self.kv_cache_mem_handles = []
        for stage in self.workers:
            kv_cache_mem_handles = []
            for worker in stage:
                kv_cache_mem_handles.append(kv_cache_mem_handles_1d.pop(0))
            self.kv_cache_mem_handles.append(kv_cache_mem_handles)
        
        return num_gpu_blocks, num_cpu_blocks

    def _remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers, return the futures immediately
        """
        if self.simulator_config.is_simulator_mode:
            handlers = []
            for stage in self.workers:
                for worker in stage:
                    item = getattr(worker, func_name)(*args)
                    if asyncio.iscoroutine(item):
                        handlers.append(asyncio.create_task(item))
                    else:
                        async def wrapper():
                            return item
                        handlers.append(asyncio.create_task(wrapper()))
            return handlers
        else:
            handlers = []
            for stage in self.workers:
                for worker in stage:
                    handlers.append(getattr(worker, func_name).remote(*args))
            return handlers

    def abort_request(self, request_id: int):
        """
        abort_request: Abort one request and free its resources
        """
        # Currently there may be some race conditions here,
        # so we just do nothing
        # TODO. Implement request abortion
        logger.warn(f"Request abortion is not implemented yet")
        return
        self.scheduler.abort_request(request_id)
        self._free_request_resources(request_id)
    
    @abstractmethod
    async def start_event_loop(self):
        raise NotImplementedError()
    
    @abstractmethod
    async def print_engine_status(self):
        raise NotImplementedError()
        
    
class ContextStageLLMEngine(SingleStageLLMEngine):
    def _get_scheduler(self) -> ContextStageScheduler:
        return get_context_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.block_manager
        )
    
    def __init__(
        self,
        bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: ContextStageSchedConfig,
        simulator_config: SimulatorConfig,
        placement_groups: List[PlacementGroup],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(
            Stage.CONTEXT,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            simulator_config,
            placement_groups,
            engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback
        )
        
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []
        
        self.bridge_queue = bridge_queue
    
    def add_request(self, request: Request):
        self.scheduler.add_request(request)
    
    def _free_request_resources(self, request_id: int):
        super()._free_request_resources(request_id)
        
    async def _step(self):
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        
        Note2. Pipeline parallel is not tested yet
        """
        # pick next batch from scheduler
        batched_requests = self.scheduler.get_next_batch_and_pop()
        if len(batched_requests) == 0:
            # Two cases may cause len(batched_requests) == 0:
            # 1. No request in the waiting queue
            # 2. No enough free blocks (e.g. the decoding stage is too slow)
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_CONTEXT_NO_REQUEST)
        else:
            logger.info(f"(context) Forwarding with lengths {[len(request.prompt_token_ids) for request in batched_requests.requests]}")
            # allocate blocks as needed
            self.block_manager.allocate_blocks_batched(batched_requests)
            
            # Log down the lifetime event
            for request in batched_requests.requests:
                self.engine_on_new_lifetime_event_callback(
                    request.request_id,
                    LifetimeEvent(LifetimeEventType.ContextBegin)
                )
                
            # push the batch into pipeline
            batched_requests.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batched_requests)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                batched_requests.get_request_ids(),
                batched_requests.get_input_tokens_batched(),
                batched_requests.get_first_token_indexes(),
                self.block_manager.get_partial_block_table(
                    batched_requests.get_request_ids()
                ),
            )
            
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                # No request in the batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                generated_tokens_ids = await self.batches_ret_futures[0]
                    
                end_time = time.time()
                generated_tokens = [
                    self.tokenizer.decode(gen_token_id)
                    for gen_token_id in generated_tokens_ids
                ]

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(
                    generated_tokens, generated_tokens_ids, end_time
                )
                
                self.scheduler.on_finish_requests(finished_batch)
                
                for request, new_token, new_token_id in zip(
                    finished_batch.requests, generated_tokens, generated_tokens_ids
                ):
                    step_output = StepOutput(request, new_token, new_token_id)
                    self.engine_on_new_lifetime_event_callback(
                        request.request_id,
                        LifetimeEvent(LifetimeEventType.ContextEnd)
                    )
                    self.engine_on_new_step_output_callback(
                        request.request_id,
                        step_output
                    )

                # Cannot free blocks now! The decoding stage may still need them!

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
                
                # Inform the user that the request has finished the context stage
                for request in finished_batch.requests:
                    if not request.is_finished:
                        # Push the request into the bridge queue if it is not finished
                        migrating_req = MigratingRequest(
                            request,
                            self.block_manager.get_block_table(request.request_id),
                            self.parallel_config,
                        )
                        self.bridge_queue.put_nowait(migrating_req) # This won't panic because the queue is unbounded
                    else:
                        self._free_request_resources(request.request_id)
    
    def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest):
        """
        Called when the decoding engine finishes migrating the blocks of the request.
        """
        self._free_request_resources(migrated_request.req.request_id)
        self.scheduler.on_request_migrated(migrated_request)
        
    async def start_event_loop(self):
        async def event_loop1():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        async def event_loop2():
            while True:
                self.print_engine_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)

        await asyncio.gather(event_loop1(), event_loop2())
        
    def print_engine_status(self):
        self.scheduler.print_status()
        

class DecodingStageLLMEngine(SingleStageLLMEngine):
    def _get_scheduler(self) -> DecodingStageScheduler:
        return get_decoding_stage_scheduler(
            self.sched_config,
            self.parallel_config,
            self.block_manager,
            self._migrate_blocks
        )
        
    def __init__(
        self,
        bridge_queue: asyncio.Queue[MigratingRequest],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: DecodingStageSchedConfig,
        simulator_config: SimulatorConfig,
        placement_groups: List[PlacementGroup],
        clear_migrated_blocks_callback: Callable[[Request], None],
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):
        super().__init__(
            Stage.DECODING,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            simulator_config,
            placement_groups,
            engine_on_new_step_output_callback,
            engine_on_new_lifetime_event_callback
        )
        
        self.bridge_queue = bridge_queue
        self.clear_migrated_blocks_callback = clear_migrated_blocks_callback
        
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline = []
        self.batches_ret_futures = []
        
    async def register_kvcache_mem_handles(
        self,
        context_parallel_config: ParallelConfig,
        kv_cache_mem_handles: List[List[Tuple[cudaMemoryIpcHandle, cudaMemoryIpcHandle]]]
    ):
        """
        Distribute kv cache memory IPC handles to workers and workers will
        register those handles.
        """
        self.kv_cache_mem_handles = kv_cache_mem_handles
        await asyncio.wait(self._remote_call_all_workers_async(
            "register_kvcache_mem_handles",
            context_parallel_config,
            kv_cache_mem_handles
        ))
    
    def _free_request_resources(self, request_id: int):
        super()._free_request_resources(request_id)
        self.request_events.pop(request_id)
        self.request_outputs.pop(request_id)
        
    async def _migrate_blocks(
        self,
        migrating_req: MigratingRequest
    ) -> None:
        """
        Migrate one request from the context engine to the decoding engine
        
        This function will be called be the decoding stage scheduler
        
        This function performs the following steps:
        - Allocate blocks on the decoding engine's side
        - Transfer the blocks
        - Clear the blocks on the context engine's side
        """
        # Allocate blocks on the decoding engine's side
        
        # Here we temporarily backup the generated tokens and generated token ids
        # since we are going to overwrite them later when allocating blocks
        generated_token_bkup = migrating_req.req.generated_tokens
        generated_token_ids_bkup = migrating_req.req.generated_token_ids
        migrating_req.req.generated_tokens = []
        migrating_req.req.generated_token_ids = []
        self.block_manager.allocate_blocks(migrating_req.req)
        migrating_req.req.generated_tokens = generated_token_bkup
        migrating_req.req.generated_token_ids = generated_token_ids_bkup
        
        target_block_indexes = self.block_manager.get_block_table(migrating_req.req.request_id)
        assert len(target_block_indexes) == len(migrating_req.block_indexes)
        
        # Transfer the blocks
        self.engine_on_new_lifetime_event_callback(
            migrating_req.req.request_id,
            LifetimeEvent(LifetimeEventType.MigrationBegin)
        )
        await asyncio.wait(self._remote_call_all_workers_async(
            "migrate_blocks",
            migrating_req.block_indexes,
            migrating_req.context_parallel_config,
            target_block_indexes
        ))
        self.engine_on_new_lifetime_event_callback(
            migrating_req.req.request_id,
            LifetimeEvent(LifetimeEventType.MigrationEnd)
        )
    
        # Clear the blocks on the context engine's side
        self.clear_migrated_blocks_callback(migrating_req)
            
    async def _step(self) -> None:
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """

        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size

        # pick next batch from scheduler
        # this may trigger migration if some requests are still at context stage
        # this may trigger swap_in if some requests have been swapped out to CPU
        # this may also trigger swap_out if GPU blocks are not enough
        batched_requests = self.scheduler.get_next_batch()

        if len(batched_requests) == 0:
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_DECODING_NO_REQUEST)
        else:
            # Log down the lifetime event
            for request in batched_requests.requests:
                self.engine_on_new_lifetime_event_callback(
                    request.request_id,
                    LifetimeEvent(LifetimeEventType.DecodingBegin),
                    True
                )
                
            # Allocate blocks as needed
            self.block_manager.allocate_blocks_batched(batched_requests)

            # Check if all requests are on GPU (i.e. not swapped out)
            assert self.block_manager.is_all_requests_on_gpu(
                batched_requests
            ), "Some requests are currently swapped out to CPU"

            # push the batch into pipeline
            batched_requests.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batched_requests)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                batched_requests.get_request_ids(),
                batched_requests.get_input_tokens_batched(),
                batched_requests.get_first_token_indexes(),
                self.block_manager.get_partial_block_table(
                    batched_requests.get_request_ids()
                ),
            )
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        # output buffer
        finished_reqs = []

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                generated_tokens_ids = await self.batches_ret_futures[0]
                end_time = time.time()
                generated_tokens = [
                    self.tokenizer.decode(gen_token_id)
                    for gen_token_id in generated_tokens_ids
                ]

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(
                    generated_tokens, generated_tokens_ids, end_time
                )

                for request, new_token, new_token_id in zip(
                    finished_batch.requests, generated_tokens, generated_tokens_ids
                ):
                    self.engine_on_new_step_output_callback(
                        request.request_id,
                        StepOutput(request, new_token, new_token_id)
                    )
                    if request.is_finished:
                        self.engine_on_new_lifetime_event_callback(
                            request.request_id,
                            LifetimeEvent(LifetimeEventType.DecodingEnd)
                        )
                finished_reqs = self.scheduler.pop_finished_requests()

                # free blocks for finished requests
                self.block_manager.free_blocks_batched(finished_reqs)
                self._remote_call_all_workers_async(
                    "clear_request_resource_batched", finished_reqs
                )

                # pop the finished batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

        # proactive request migraion
        await self.scheduler.post_process()
    
    async def start_event_loop(self):
        async def event_loop1():
            # Event loop 1. Add migrating request to the scheduler
            while True:
                migrating_req = await self.bridge_queue.get()
                await self.scheduler.add_request(migrating_req)
                self.bridge_queue.task_done()
        
        async def event_loop2():
            # Event loop 2. Run step()
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        async def event_loop3():
            # Event loop 3. Print engine status
            while True:
                self.print_engine_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)
                
        await asyncio.gather(event_loop1(), event_loop2(), event_loop3())
    
    def print_engine_status(self):
        self.block_manager.print_block_usage()
        self.scheduler.print_status()
        