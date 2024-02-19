import torch

from distserve.config import ModelConfig, ParallelConfig, CacheConfig


def get_model_op(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
):
    model_name = model_config.model
    model_type = model_config.hf_config.model_type
    match model_type:
      case 'opt':
        return torch.classes.gpt_ops.OptOp(
            model_config.hf_config.vocab_size,
            model_config.get_max_model_len(),
            model_config.get_hidden_size(),
            model_config.get_num_layers(),
            model_config.get_num_heads(),
            model_config.get_head_size(),
            model_config.dtype,
            cache_config.block_size,
            cache_config.max_num_blocks_per_req,
            parallel_config.to_list(),
        )
      case 'gpt2':
        return torch.classes.gpt_ops.Gpt2Op(
            model_config.hf_config.vocab_size,
            model_config.get_max_model_len(),
            model_config.get_hidden_size(),
            model_config.get_num_layers(),
            model_config.get_num_heads(),
            model_config.get_head_size(),
            model_config.dtype,
            cache_config.block_size,
            cache_config.max_num_blocks_per_req,
            parallel_config.to_list(),
        )
      case 'llama':
        return torch.classes.gpt_ops.Llama2Op(
            model_config.hf_config.vocab_size,
            model_config.get_max_model_len(),
            model_config.get_hidden_size(),
            model_config.get_num_layers(),
            model_config.get_q_heads(),
            model_config.get_num_heads(),
            model_config.get_head_size(),
            model_config.get_ffn_inter_dim(),
            model_config.dtype,
            cache_config.block_size,
            cache_config.max_num_blocks_per_req,
            parallel_config.to_list(),
        )
    raise NotImplementedError(f"model {model_name} not supported")
