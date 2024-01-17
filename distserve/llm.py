import time
from typing import List, Union, Optional, AsyncGenerator

import asyncio
from tqdm import tqdm

from distserve.config import (
    ModelConfig,
    ParallelConfig,
    CacheConfig,
    DisaggParallelConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.single_stage_engine import StepOutput
from distserve.engine import LLMEngine
from distserve.logger import init_logger
from distserve.request import Request, SamplingParams


logger = init_logger(__name__)


class OfflineLLM:
    """A Large Language Model (LLM) for offline inference.
    It wraps around the LLMEngine and provides the **generate** interface to do
    offline inference on a list of prompts, which only return when all the prompts
    finish generation. If you want to do online inference where each user can asynchronously
    get the generation results in a streaming fashion, please refer to the **AsyncLLM** class.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig
    ):
        self.engine = LLMEngine(
            model_config,
            disagg_parallel_config,
            cache_config,
            context_sched_config,
            decoding_sched_config
        )
        
        asyncio.run(self.engine.initialize())

    def generate(
        self,
        prompts: Optional[Union[List[str], str]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ) -> List[List[StepOutput]]:
        if prompts is None and prompt_token_ids is None:
            raise ValueError("prompts or prompt_token_ids must be provided")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError(
                    "The lengths of prompts and prompt_token_ids must be the same."
                )

        num_requests = len(prompts) if prompts is not None else len(prompt_token_ids)
        if sampling_params is None:
            sampling_params = [SamplingParams()] * num_requests
        elif isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * num_requests
        else:
            assert (
                len(sampling_params) == num_requests
            ), f"prompts should pair with the list of sampling parameters, \
                 but got {num_requests} prompts and {len(sampling_params)} sampling parameters"

        async def deal_with_request_coroutine(req_index: int) -> List[StepOutput]:
            prompt = prompts[req_index] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[req_index]
            step_outputs = []
            async for step_output in self.engine.generate(prompt, token_ids, sampling_params[req_index]):
                step_outputs.append(step_output)
            return step_outputs
        
        async def generate_main() -> List[List[StepOutput]]:
            request_tasks = []
            for i in range(num_requests):
                request_tasks.append(asyncio.create_task(deal_with_request_coroutine(i)))
            event_loop_task = asyncio.create_task(self.engine.start_all_event_loops())
            result = await asyncio.gather(*request_tasks)
            event_loop_task.cancel()
            return result

        return asyncio.run(generate_main())
    

class AsyncLLM:
    """A Large Language Model (LLM) for online inference."""

    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        context_sched_config: ContextStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig
    ):
        self.engine = LLMEngine(
            model_config,
            disagg_parallel_config,
            cache_config,
            context_sched_config,
            decoding_sched_config
        )
        
        asyncio.run(self.engine.initialize())

    async def start_event_loop(self):
        """Start the underlying LLMEngine's event loop
        
        This function should be called at the beginning of the server.
        """
        await self.engine.start_all_event_loops()
        
    async def generate(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> AsyncGenerator[StepOutput, None]:
        """Generate outputs for a single request.

        This method is a coroutine. It adds the request into the engine, and
        yields the StepOutput objects from the LLMEngine for the request.

        Args:
            request_id: The unique id of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            sampling_params: The sampling parameters of the request.

        Yields:
            The output `StepOutput` objects from the LLMEngine for the
            request.
        """
        if prompt is None and prompt_token_ids is None:
            raise ValueError("prompt or prompt_token_ids must be provided")

        arrival_time = time.time()
        async for step_output in  self.engine.generate(
            prompt,
            prompt_token_ids,
            sampling_params,
            arrival_time,
            request_id
        ):
            yield step_output
            
        # Here the engine has all the request's lifetime events in engine.request_lifetime_events[request_id]
        # But unfortunately I don't know how to return it and when to clear it...
        # TODO Find a reliable way to return and clear the lifetime events

    def get_and_pop_request_lifetime_events(self, request_id: str):
        return self.engine.request_lifetime_events.pop(request_id)
    
    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """

        logger.info(f"Aborted request {request_id}.")
        self.engine.abort_request(request_id)

