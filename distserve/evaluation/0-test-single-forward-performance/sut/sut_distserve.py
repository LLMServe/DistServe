import copy, time
import torch
import ray
from transformers import AutoTokenizer

from distserve.config import ModelConfig, CacheConfig, ParallelConfig
from distserve.models import get_model_op
from distserve.utils import get_gpu_memory, set_random_seed, GB, MB
from distserve.downloader import download_and_convert_weights

import sys
sys.path.append("..")	# Add the parent directory to the system path so that we can import structs.py
from structs import *
from .abstract_sut import SystemUnderTest, get_input_ids

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
        
class DistServeSUT(SystemUnderTest):
    def __init__(
        self,
        worker_param: WorkerParam,
        input_params: list[InputParam]
    ):
        self.worker_param = worker_param
        
        self.model_config = ModelConfig(
            model = worker_param.model_dir,
            tokenizer = worker_param.model_dir,
            dtype = "fp16",
            use_dummy_weights = worker_param.use_dummy_weights
        )
        self.cache_config = CacheConfig(
            block_size = BLOCK_SIZE,
            max_num_blocks_per_req = 2048 // BLOCK_SIZE + 2,
            gpu_memory_utilization = 0.92
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer)
        
        print("Creating workers...")
        self.workers = []
        pp_nccl_comm_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        tp_nccl_comm_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        for tp_id in range(self.worker_param.tp_world_size):
            parallel_config = ParallelConfig(
                tensor_parallel_rank = tp_id,
                tensor_parallel_size = self.worker_param.tp_world_size,
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
            # worker = Worker(
            #     worker_id = tp_id,
            #     model_config = self.model_config,
            #     cache_config = self.cache_config,
            #     parallel_config = parallel_config,
            #     tensor_parallel_id = tp_nccl_comm_id,
            #     pipeline_parallel_id = pp_nccl_comm_id
            # )
            self.workers.append(worker)
        
        print("Loading weights...")
        ray.get([worker.init_model.remote() for worker in self.workers])
        # self.workers[0].init_model()
        
        print("Initializing kv cache...")
        block_needed = 0
        for input_param in input_params:
            cur_block_needed = input_param.batch_size * ((input_param.input_len+input_param.output_len-1+BLOCK_SIZE-1)//BLOCK_SIZE)
            block_needed = max(block_needed, cur_block_needed)
        ray.get([worker.init_kvcache.remote(block_needed) for worker in self.workers])
        # self.workers[0].init_kvcache(block_needed)

    def inference(
        self,
        input_param: InputParam
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], float, list[float]]:
        input_ids = get_input_ids(self.worker_param.model_dir, input_param.input_len*input_param.batch_size)
        prompt_token_ids = input_ids.view(input_param.batch_size, input_param.input_len).tolist()
        
        block_needed_for_one_req = (input_param.input_len+input_param.output_len-1+BLOCK_SIZE-1)//BLOCK_SIZE
        block_table = [
            list(range(block_needed_for_one_req*i, block_needed_for_one_req*(i+1)))
            for i in range(input_param.batch_size)
        ]
        
        predict_ids = [
            []
            for _ in range(input_param.batch_size)
        ]
        
        # Prefill phase
        prefill_start_time = time.perf_counter()
        last_turn_output_ids = ray.get([
            worker.step.remote(
                prompt_token_ids,
                [0 for _ in range(input_param.batch_size)],
                block_table
            )
            for worker in self.workers
        ])[0]
        # last_turn_output_ids = self.workers[0].step(
        #     prompt_token_ids,
        #     [0 for _ in range(input_param.batch_size)],
        #     block_table
        # )
        prefill_end_time = time.perf_counter()
        prefill_time_usage = (prefill_end_time - prefill_start_time)*1000
        for i in range(input_param.batch_size):
            predict_ids[i].append(last_turn_output_ids[i])
        
        # Decoding phase
        decoding_time_usages = []
        for step in range(input_param.output_len-1):
            decoding_start_time = time.perf_counter()
            last_turn_output_ids = ray.get([
                worker.step.remote(
                    [
                        [x]
                        for x in last_turn_output_ids
                    ],
                    [input_param.input_len+step for _ in range(input_param.batch_size)],
                    block_table
                )
                for worker in self.workers
            ])[0]
            # last_turn_output_ids = self.workers[0].step(
            #     [
            #         [x]
            #         for x in last_turn_output_ids
            #     ],
            #     [input_param.input_len+step for _ in range(input_param.batch_size)],
            #     block_table
            # )
            decoding_end_time = time.perf_counter()
            decoding_time_usages.append((decoding_end_time - decoding_start_time)*1000)
            for i in range(input_param.batch_size):
                predict_ids[i].append(last_turn_output_ids[i])
        
        predict_texts = [
            self.tokenizer.decode(predict_id, skip_special_tokens=True)
            for predict_id in predict_ids
        ]
        
        return input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usages
    
    