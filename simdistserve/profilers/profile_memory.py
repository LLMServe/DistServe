import time

from fastserve.config import ModelConfig, ParallelConfig

MB = 1 << 20
GB = 1 << 30

try:
    import torch

    total_gpu_memory_ = torch.cuda.get_device_properties(0).total_memory
except:
    total_gpu_memory_ = 80 * GB
    pass


def _get_block_size_in_bytes(
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


def get_model_possible_pp(model):
    model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
    total_num_hidden_layers = model_config.hf_config.num_hidden_layers
    possible_pp = []
    for pp in range(1, 1 + total_num_hidden_layers):
        if total_num_hidden_layers % pp == 0:
            possible_pp.append(pp)
    return possible_pp


def get_model_possible_tp(model):
    model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
    total_num_attention_heads = model_config.hf_config.num_attention_heads
    possible_tp = []
    for tp in range(1, 1 + total_num_attention_heads):
        if total_num_attention_heads % tp == 0:
            possible_tp.append(tp)
    return possible_tp


def measure_stats(
    model="facebook/opt-13b", tp=1, pp=1,
    total_gpu_memory=total_gpu_memory_,
    block_size=16,
    gpu_memory_utilization=0.95,
):
    try:
        result = dict(locals())

        model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
        total_num_hidden_layers = model_config.hf_config.num_hidden_layers
        total_num_attention_heads = model_config.hf_config.num_attention_heads
        if total_num_hidden_layers % pp != 0:
            return None
        if total_num_attention_heads % tp != 0:
            return None

        parallel_config = ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp
        )
        model_in_bytes = model_config.get_model_size_in_bytes(
            parallel_config=parallel_config
        )
        model_size = model_in_bytes / GB
        block_size_in_bytes = _get_block_size_in_bytes(
            block_size, model_config, parallel_config
        )
        peak_runtime_memory = (
            total_gpu_memory * 0.01
            + model_in_bytes
        )
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_runtime_memory)
            // block_size_in_bytes
        )
        max_num_tokens = num_gpu_blocks * block_size
        kv_size_in_byte = (block_size_in_bytes / block_size / MB)
        result.update(
            dict(
                total_gpu_memory=total_gpu_memory,
                model_size=model_size,
                peak_runtime_memory=peak_runtime_memory,
                kv_size_in_byte=kv_size_in_byte,
                num_gpu_blocks=num_gpu_blocks,
                max_num_tokens=max_num_tokens,
            )
        )
        return result
    except Exception as e:
        # print(f"Error {model} {tp} {pp}: {e}")
        pass
    return None


def get_all_possible_tp_pp():
    for model in ["facebook/opt-13b", "facebook/opt-66b", "facebook/opt-175b"]:
        tps = get_model_possible_tp(model)
        pps = get_model_possible_pp(model)
        print((model, tps, pps))


def get_all_configs():
    configs = []

    for model in ("facebook/opt-13b", "facebook/opt-66b", "facebook/opt-175b"):
        for tp in get_model_possible_tp(model):
            for pp in get_model_possible_pp(model):
                a = measure_stats(model, tp, pp)
                print(model, tp, pp)
                configs.append(a)

    configs = [i for i in configs if i is not None]
    print(configs)
    return configs


# if __name__ == '__main__':
#     get_all_configs()
#     # get_all_possible_tp_pp
#     pass

def get_params(model):
    model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
    total_num_hidden_layers = model_config.hf_config.num_hidden_layers
    total_num_attention_heads = model_config.hf_config.num_attention_heads
    return total_num_hidden_layers, total_num_attention_heads


for model in ["facebook/opt-13b", "facebook/opt-66b", "facebook/opt-175b"]:
    a, b = get_params(model)
    print(f"{model},{a},{b}")
    pass
