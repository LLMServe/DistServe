from typing import Optional, List

import torch
from transformers import AutoConfig

from distserve.utils import GB


class CacheConfig:
    """Configuration for the key-value cache.

    Args:
        block_size: Number of tokens in a block.
        max_num_blocks_per_req: Maximum number of blocks each request can have.
        gpu_memory_utilization: The maximum percentage of GPU memory that can be used.
        cpu_swap_space: The maximum CPU swap space in bytes that can be used.
    """

    def __init__(
        self,
        block_size: int,
        max_num_blocks_per_req: int,
        gpu_memory_utilization: int = 0.9,
        cpu_swap_space: int = 0,
    ):
        self.block_size = block_size
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.gpu_memory_utilization = gpu_memory_utilization
        self.cpu_swap_space = cpu_swap_space * GB


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        tensor_parallel_size: number of tensor parallel groups.
        tensor_parallel_rank: rank in the tensor parallel group.
        pipeline_parallel_size: number of pipeline parallel groups.
        pipeline_parallel_rank: rank in the pipeline parallel group.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_rank: int = 0,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_rank = pipeline_parallel_rank

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self.use_parallel = self.world_size > 1

    def to_list(self) -> List[int]:
        return [
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
        ]

    def is_last_stage(self) -> bool:
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1


class DisaggParallelConfig:
    """Configuration for disaggregated execution.

    Args:
        context: Context stage parallel config
        decoding: Decoding stage parallel config
    """

    def __init__(
        self,
        context: ParallelConfig,
        decoding: ParallelConfig,
    ) -> None:
        self.context = context
        self.decoding = decoding
    
    def get_num_workers(self) -> int:
        """Get the total number of workers (GPUs) needed."""
        return self.context.world_size + self.decoding.world_size


class ContextStageSchedConfig:
    """Configuration for the context stage scheduler.

    Args:
        policy: The scheduling policy.
        max_batch_size: The maximum number of requests in a batch.
        max_tokens_per_batch: The maximum number of input tokens in a batch.
    """
    def __init__(
        self,
        policy: str,
        max_batch_size: int,
        max_tokens_per_batch: int,
        parallel_config: ParallelConfig = None,
    ):
        assert policy in [
            "fcfs"
        ], f"policy {policy} not supported"
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.parallel_config = parallel_config
    
class DecodingStageSchedConfig:
    """Configuration for the decoding stage scheduler.

    Args:
        policy: The scheduling policy.
        max_batch_size: The maximum number of requests in a batch.
        max_tokens_per_batch: The maximum number of input tokens in a batch.
    """

    def __init__(
        self,
        policy: str,
        max_batch_size: int,
        max_tokens_per_batch: int,
        model_name: str = None,
        waiting_block_prop_threshold: float = 0.05
    ):
        assert policy in [
            "fcfs",
            "srpt",
            "mlfq",
            "sj-mlfq",
        ], f"policy {policy} not supported"
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.model_name = model_name
        self.waiting_block_prop_threshold = waiting_block_prop_threshold

_TORCH_DTYPE_MAP = {"fp16": torch.half, "fp32": torch.float32}


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Model name or path.
        tokenizer: Tokenizer name or path.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
            Default to "auto".
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        dtype: Data type of the model. Default to "fp16".
        seed: Random seed for reproducing.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str],
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        dtype: str = "fp16",
        seed: int = 1,
        use_dummy_weights: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer if tokenizer else model
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.seed = seed
        self._verify_args()
        self.hf_config = self._get_hf_config()
        self.use_dummy_weights = use_dummy_weights

    def _verify_args(self):
        assert self.dtype in [
            "fp16",
            "fp32",
        ], f"dtype must be either 'fp16' or 'fp32'."

    def _get_hf_config(self):
        try:
            config = AutoConfig.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code
            )
        except:
            raise ValueError(
                f"Failed to load the model config, please check the model name or path: {self.model}"
            )
        return config

    def get_dtype_size(self) -> int:
        if self.dtype == "fp16":
            return 2
        elif self.dtype == "fp32":
            return 4
        else:
            raise NotImplementedError(f"dtype {self.dtype} not supported")

    def get_torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPE_MAP[self.dtype]

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_ffn_inter_dim(self) -> int:
        # For LLaMA-2:
        return self.hf_config.intermediate_size

    def get_q_heads(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        # For LLaMA-2:
        return (
            self.hf_config.num_attention_heads
                // parallel_config.tensor_parallel_size
        )
    
    def get_num_heads(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        # For GPTBigCode & Falcon:
        # Note: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        new_decoder_arch_falcon = self.hf_config.model_type == "falcon" and getattr(
            self.hf_config, "new_decoder_architecture", False
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            return 1

        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return self.hf_config.n_head_kv // parallel_config.tensor_parallel_size

        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (
                self.hf_config.num_key_value_heads
                // parallel_config.tensor_parallel_size
            )

        # Normal case:
        total_num_attention_heads = self.hf_config.num_attention_heads
        assert total_num_attention_heads % parallel_config.tensor_parallel_size == 0, (
            f"Total number of attention heads ({total_num_attention_heads}) "
            f"must be divisible by the size of tensor parallel group "
            f"({parallel_config.tensor_parallel_size})."
        )
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_max_model_len(self) -> int:
        max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        for key in possible_keys:
            max_len_key = getattr(self.hf_config, key, None)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        return max_model_len

    def get_num_layers(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        assert total_num_hidden_layers % parallel_config.pipeline_parallel_size == 0, (
            f"Number of layers ({total_num_hidden_layers}) must be divisible "
            f"by the size of pipeline parallel group "
            f"({parallel_config.pipeline_parallel_size})."
        )
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_model_size_in_bytes(
        self, parallel_config: ParallelConfig = ParallelConfig()
    ) -> int:
        total_params = (
            self.hf_config.vocab_size * self.get_hidden_size()  # vocab embed
            + self.get_max_model_len() * self.get_hidden_size()  # position embed
            + 4
            * self.get_num_layers(parallel_config)
            * (self.get_hidden_size() ** 2)  # attention
            / parallel_config.tensor_parallel_size # attention is divided by tp
            + 8
            * self.get_num_layers(parallel_config)
            * (self.get_hidden_size() ** 2)  # FFN
            / parallel_config.tensor_parallel_size # FFN is divided by tp
            + 5 * self.get_num_layers(parallel_config) * self.get_hidden_size()  # bias
        ) 
        return total_params * self.get_dtype_size()
