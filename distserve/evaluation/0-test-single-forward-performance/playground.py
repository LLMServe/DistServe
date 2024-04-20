import copy, time
import torch
import ray
from transformers import AutoTokenizer

from distserve.config import ModelConfig, CacheConfig, ParallelConfig
from distserve.models import get_model_op
from distserve.utils import get_gpu_memory, set_random_seed, GB, MB
from distserve.downloader import download_and_convert_weights

import copy

stem_token_ids = None
def get_input_ids(model_dir: str, num_tokens: int) -> torch.Tensor:
    """
    Generate input ids for testing
    """
    global stem_token_ids
    if stem_token_ids is None:
        text = " ".join([str(i) for i in range(1, 1000)])
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        stem_token_ids = tokenizer(text).input_ids
    input_ids_list = copy.deepcopy(stem_token_ids)
    while len(input_ids_list) < num_tokens:
        input_ids_list += input_ids_list
    input_ids = torch.tensor(input_ids_list[:num_tokens], dtype=torch.int32, device="cpu")
    return input_ids

BLOCK_SIZE = 16

@ray.remote(num_gpus=1)
class Worker:
    def __init__(
        self,
        worker_id: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig = ParallelConfig(),
        tensor_parallel_id: list[int] = None,
        pipeline_parallel_id: list[int] = None
    ):
        self.worker_id = worker_id
        self.model = None
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.tensor_parallel_id = tensor_parallel_id
        self.pipeline_parallel_id = pipeline_parallel_id
        self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)
    
    def init_model(self):
        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model_op(
            self.model_config, self.parallel_config, self.cache_config
        )
        self.model.init_communicator(self.tensor_parallel_id, self.pipeline_parallel_id)
        torch.cuda.synchronize()
        if self.model_config.use_dummy_weights:
            self.model.init_dummy_weights()
        else:
            path = download_and_convert_weights(self.model_config)
            self.model.load_weight(path)
        torch.cuda.synchronize()
        print(f"(worker #{self.worker_id}) model {self.model_config.model} loaded")
        
    def init_kvcache(self, num_gpu_blocks: int):
        kv_cache_shape = (
            num_gpu_blocks,
            self.model_config.get_num_layers(self.parallel_config),
            self.model_config.get_num_heads(self.parallel_config),
            self.cache_config.block_size,
            self.model_config.get_head_size(),
        )
        print(f"Number of blocks: {num_gpu_blocks}")
        print(f"Estimated kv cache size: {2*kv_cache_shape[0]*kv_cache_shape[1]*kv_cache_shape[2]*kv_cache_shape[3]*kv_cache_shape[4]*2/GB:.2f} GB")
        self.k_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        self.v_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        torch.cuda.synchronize()
    
    def step(
        self,
        input_tokens: list[list[int]],
        first_token_indexes: list[int],
        block_Table: list[list[int]]
    ):
        generated_token_ids = self.model.forward(
            input_tokens,
            first_token_indexes,
            self.k_cache,
            self.v_cache,
            block_Table
        )
        return generated_token_ids
        
class DistServeSUT:
    def __init__(
        self,
        tp_world_size: int = 1
    ):
        self.model_config = ModelConfig(
            model = "facebook/opt-6.7b",
            tokenizer = "facebook/opt-6.7b",
            dtype = "fp16",
        )
        self.cache_config = CacheConfig(
            block_size = BLOCK_SIZE,
            max_num_blocks_per_req = 2048 // BLOCK_SIZE + 2,
            gpu_memory_utilization = 0.92
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer)
        self.tp_world_size = tp_world_size
        
        print("Creating workers...")
        self.workers = []
        pp_nccl_comm_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        tp_nccl_comm_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        for tp_id in range(self.tp_world_size):
            parallel_config = ParallelConfig(
                tensor_parallel_rank = tp_id,
                tensor_parallel_size = self.tp_world_size,
                pipeline_parallel_rank = 0,
                pipeline_parallel_size = 1
            )
            worker = Worker.remote(
                worker_id = tp_id,
                model_config = self.model_config,
                cache_config = self.cache_config,
                parallel_config = parallel_config,
                tensor_parallel_id = tp_nccl_comm_id,
                pipeline_parallel_id = pp_nccl_comm_id
            )
            self.workers.append(worker)
        
        print("Loading weights...")
        ray.get([worker.init_model.remote() for worker in self.workers])
        
        print("Initializing kv cache...")
        block_needed = 1024
        ray.get([worker.init_kvcache.remote(block_needed) for worker in self.workers])

    def inference(
        self,
        prompt_lens_list: list[int],
        output_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], float, list[float]]:
        batch_size = len(prompt_lens_list)
        
        input_ids = get_input_ids(self.model_config.model, sum(prompt_lens_list)).tolist()
        prompt_token_ids = []
        for (i, prompt_len) in enumerate(prompt_lens_list):
            prompt_token_ids.append(input_ids[:prompt_len])
            input_ids = input_ids[prompt_len:]
        
        max_block_needed_for_one_req = max([
            (prompt_len+output_len-1+BLOCK_SIZE-1)//BLOCK_SIZE
            for prompt_len in prompt_lens_list
        ])
        block_table = []
        next_block_idx = 0
        for prompt_len in prompt_lens_list:
            block_needed = (prompt_len+output_len-1+BLOCK_SIZE-1)//BLOCK_SIZE
            block_table.append(
                list(range(next_block_idx, next_block_idx+block_needed)) + [-1000]*(max_block_needed_for_one_req-block_needed)
            )
            next_block_idx += block_needed
        
        predict_ids = [
            []
            for _ in range(batch_size)
        ]
        
        # Prefill phase
        prefill_start_time = time.perf_counter()
        last_turn_output_ids = ray.get([
            worker.step.remote(
                prompt_token_ids,
                [0 for _ in range(batch_size)],
                block_table
            )
            for worker in self.workers
        ])[0]
        prefill_end_time = time.perf_counter()
        prefill_time_usage = (prefill_end_time - prefill_start_time)*1000
        for i in range(batch_size):
            predict_ids[i].append(last_turn_output_ids[i])
        
        # Decoding phase
        decoding_time_usages = []
        for step in range(output_len-1):
            decoding_start_time = time.perf_counter()
            last_turn_output_ids = ray.get([
                worker.step.remote(
                    [
                        [x]
                        for x in last_turn_output_ids
                    ],
                    [prompt_len+step for prompt_len in prompt_lens_list],
                    block_table
                )
                for worker in self.workers
            ])[0]
            decoding_end_time = time.perf_counter()
            decoding_time_usages.append((decoding_end_time - decoding_start_time)*1000)
            for i in range(batch_size):
                predict_ids[i].append(last_turn_output_ids[i])
        
        predict_texts = [
            self.tokenizer.decode(predict_id, skip_special_tokens=True)
            for predict_id in predict_ids
        ]
        
        return input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usages
    
if __name__ == "__main__":
    sut = DistServeSUT()
    prompt_lens = [304,554,561,615, 1472, 1761, 1840]
    input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usages = sut.inference(prompt_lens, 16)
    print(predict_texts)
    print(prefill_time_usage)
    print(decoding_time_usages)