import argparse
import json
from typing import AsyncGenerator, List, Tuple
import asyncio
import time
import traceback
import sys, os
import signal

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from distserve.llm import AsyncLLM
from distserve.request import SamplingParams
from distserve.utils import random_uuid
from distserve.logger import init_logger
from distserve.single_stage_engine import StepOutput
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.lifetime import json_encode_lifetime_events

import ray

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    logger.info("Received a request.")
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(
        request_id, prompt=prompt, sampling_params=sampling_params
    )

    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for step_output in results_generator:
                text_output = step_output.request.get_response()
                ret = {"text": text_output}
                yield (json.dumps(ret) + "\0").encode("utf-8")

        async def abort_request() -> None:
            await engine.abort(request_id)

        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        final_outputs: List[Tuple[StepOutput, float]] = []   # (step_output, timestamp)
        async for step_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_outputs.append((step_output, time.time()))

        request_events = engine.get_and_pop_request_lifetime_events(request_id)
        text_output = prompt + ''.join([step_output[0].new_token for step_output in final_outputs])
        ret = {
            "text": text_output,
            "timestamps": [step_output[1] for step_output in final_outputs],
            "lifetime_events": json_encode_lifetime_events(request_events)
        }
        return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy-weights", action="store_true")
    
    parser.add_argument("--context-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-tensor-parallel-size", type=int, default=1)
    
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=int, default=1)
    
    parser.add_argument("--context-sched-policy", type=str, default="fcfs")
    parser.add_argument("--context-max-batch-size", type=int, default=256)
    parser.add_argument("--context-max-tokens-per-batch", type=int, default=4096)
    
    parser.add_argument("--decoding-sched-policy", type=str, default="fcfs")
    parser.add_argument("--decoding-max-batch-size", type=int, default=256)
    parser.add_argument("--decoding-max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--decoding-profiling-file", type=str, default=None)
    parser.add_argument("--decoding-proactive-offloading", action="store_true")
    parser.add_argument("--decoding-num-min-free-blocks-threshold", type=int, default=0)
    parser.add_argument("--decoding-num-queues-for-prediction", type=int, default=2)
    parser.add_argument("--decoding-use-skip-join", action="store_true")
    
    args = parser.parse_args()
    
    ray.init()

    engine = AsyncLLM(
        model_config=ModelConfig(
            model=args.model,
            tokenizer=args.tokenizer,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            use_dummy_weights=args.use_dummy_weights
        ),
        disagg_parallel_config=DisaggParallelConfig(
            context=ParallelConfig(
                tensor_parallel_size=args.context_tensor_parallel_size,
                pipeline_parallel_size=args.context_pipeline_parallel_size
            ),
            decoding=ParallelConfig(
                tensor_parallel_size=args.decoding_tensor_parallel_size,
                pipeline_parallel_size=args.decoding_pipeline_parallel_size
            )
        ),
        cache_config=CacheConfig(
            block_size=args.block_size,
            max_num_blocks_per_req=args.max_num_blocks_per_req,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_swap_space=args.swap_space
        ),
        context_sched_config=ContextStageSchedConfig(
            policy=args.context_sched_policy,
            max_batch_size=args.context_max_batch_size,
            max_tokens_per_batch=args.context_max_tokens_per_batch
        ),
        decoding_sched_config=DecodingStageSchedConfig(
            policy=args.decoding_sched_policy,
            max_batch_size=args.decoding_max_batch_size,
            max_tokens_per_batch=args.decoding_max_tokens_per_batch,
            profiling_file=args.decoding_profiling_file,
            model_name=args.model,
            proactive_offloading=args.decoding_proactive_offloading,
            num_min_free_blocks_threshold=args.decoding_num_min_free_blocks_threshold,
            num_queues_for_prediction=args.decoding_num_queues_for_prediction,
            use_skip_join=args.decoding_use_skip_join,
            waiting_block_prop_threshold=0.05
        )
    )

    uvicorn_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    
    async def main_coroutine():
        task2 = asyncio.create_task(uvicorn_server.serve())
        
        async def start_event_loop_wrapper():
            try:
                task = asyncio.create_task(engine.start_event_loop())
                await task
            except Exception as e:
                traceback.print_exc()
                task2.cancel()
                os._exit(1) # Kill myself, or it will print tons of errors. Don't know why.
        
        task1 = asyncio.create_task(start_event_loop_wrapper())
        
        try:
            await task2
        except:
            # This is a workaround
            # When task1 exited for some reason (e.g. error in the engine),
            # task2 will raise many exceptions, which is annoying and I do 
            # not know why
            pass
    
    asyncio.run(main_coroutine())
    