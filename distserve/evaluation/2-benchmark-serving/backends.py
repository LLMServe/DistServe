"""
Adapted from vLLM (https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py)
"""
import json
import os, sys
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

from structs import TestRequest, ReqResult

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

async def _async_request_vllm_like_interface(
    api_url: str,
    payload: dict,
    request: TestRequest,
    first_token_generated_pbar: tqdm,
    finished_pbar: tqdm,
    verbose: bool   # Print the prompt and completion
) -> ReqResult:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        issue_time = time.perf_counter()
        first_token_time = 0
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    num_tokens = 0
                    async for data in response.content.iter_any():
                        if first_token_time == 0:
                            first_token_generated_pbar.update(1)
                            first_token_generated_pbar.refresh()
                            first_token_time = time.perf_counter()
                        num_tokens += 1
                    complete_time = time.perf_counter()
                    if abs(num_tokens-request.output_len) > 2:
                        print(f"WARNING: num_tokens ({num_tokens}) != request.output_len ({request.output_len})")
                    if verbose:
                        print()
                        print(f"Prompt: {request.prompt}")
                        print(f"Completion: {data.decode('utf-8').strip(chr(0))}")
                else:
                    print(response)
                    print(response.status)
                    print(response.reason)
                    sys.exit(1)
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
            print(e)
            sys.exit(1)

        finished_pbar.update(1)
        finished_pbar.refresh()
        return ReqResult.from_http_request_result(
            request.prompt_len,
            request.output_len,
            issue_time,
            first_token_time,
            complete_time
        )
        
async def async_request_vllm(
    host: str,
    port: int,
    request: TestRequest,
    first_token_generated_pbar: tqdm,
    finished_pbar: tqdm,
    verbose: bool
) -> ReqResult:
    api_url = f"http://{host}:{port}/generate"
    payload = {
        "prompt": request.prompt,
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": request.output_len,
        "ignore_eos": True,
        "stream": True,
    }
    return await _async_request_vllm_like_interface(
        api_url,
        payload,
        request,
        first_token_generated_pbar,
        finished_pbar,
        verbose
    )   

async def async_request_lightllm(
    host: str,
    port: int,
    request: TestRequest,
    first_token_generated_pbar: tqdm,
    finished_pbar: tqdm,
    verbose: bool
) -> Optional[ReqResult]:
    api_url = f"http://{host}:{port}/generate_stream"
    payload = {
        "inputs": request.prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": request.output_len,
        }
    }
    return await _async_request_vllm_like_interface(
        api_url,
        payload,
        request,
        first_token_generated_pbar,
        finished_pbar,
        verbose
    )
        
async def async_request_deepspeed(
    host: str,
    port: int,
    request: TestRequest,
    first_token_generated_pbar: tqdm,
    finished_pbar: tqdm,
    verbose: bool
) -> Optional[ReqResult]:
    api_url = f"http://{host}:{port}/generate"
    payload = {
        "prompt": request.prompt,
        "max_tokens": request.output_len,
        "min_new_tokens": request.output_len,
        "max_new_tokens": request.output_len,
        "stream": True,
        "max_length": int((request.prompt_len + request.output_len)*1.2+10) # *1.2 to prevent tokenization error
    }
    return await _async_request_vllm_like_interface(
        api_url,
        payload,
        request,
        first_token_generated_pbar,
        finished_pbar,
        verbose
    )

BACKEND_TO_PORTS = {
    "vllm": 8100,
    "lightllm": 8200,
    "deepspeed": 8300,
    "distserve": 8400
}

BACKEND_TO_REQUEST_FUNCS = {
    "vllm": async_request_vllm,
    "lightllm": async_request_lightllm,
    "deepspeed": async_request_deepspeed,
    "distserve": async_request_vllm,
}
