import os, sys
import argparse
import aiohttp
import asyncio
import json

prompts = [
    "To be or not to be",
    "One two three four five",
    "A shoulder for the past, let out the",
    "Life blooms like a flower, far",
    "Genshin Impact is"
]

async def main(args: argparse.Namespace):
    async def task(prompt: str):
        payload = {
            "prompt": prompt,
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 50,
            "ignore_eos": False,
            "stream": True,
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            generated_text = ""
            async with session.post(url=url, json=payload) as response:
                async for data in response.content.iter_any():
                    generated_text = json.loads(data.decode("utf-8"))["text"]
            print(f"{prompt} | {generated_text}")

    url = f"http://{args.host}:{args.port}/generate"
    tasks = []
    for prompt in prompts:
        tasks.append(asyncio.create_task(task(prompt)))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="localhost"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8000"
    )

    asyncio.run(main(parser.parse_args()))
    