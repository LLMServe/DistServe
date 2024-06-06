"""
Usage example:

python -m distserve.api_server.distserve_api_server \\
    --host 0.0.0.0 \\
    --port {port} \\
    --model {args.model} \\
    --tokenizer {args.model} \\
    \\
    --context-tensor-parallel-size {context_tp} \\
    --context-pipeline-parallel-size {context_pp} \\
    --decoding-tensor-parallel-size {decoding_tp} \\
    --decoding-pipeline-parallel-size {decoding_pp} \\
    \\
    --block-size 16 \\
    --max-num-blocks-per-req 128 \\
    --gpu-memory-utilization 0.95 \\
    --swap-space 16 \\
    \\
    --context-sched-policy fcfs \\
    --context-max-batch-size 128 \\
    --context-max-tokens-per-batch 8192 \\
    \\
    --decoding-sched-policy fcfs \\
    --decoding-max-batch-size 1024 \\
    --decoding-max-tokens-per-batch 65536
"""

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

import distserve
import distserve.engine
from distserve.llm import AsyncLLM
from distserve.request import SamplingParams
from distserve.utils import random_uuid, set_random_seed
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
        # Currently we do not support request abortion, so we comment this line.
        # TODO implement request abortion.
        # background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        final_outputs: List[Tuple[StepOutput, float]] = []   # (step_output, timestamp)
        async for step_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)
                return Response(status_code=499)
            final_outputs.append((step_output, time.perf_counter()))

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
    
    distserve.engine.add_engine_cli_args(parser)
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    ray.init()
    
    engine = AsyncLLM.from_engine_args(args)

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
    