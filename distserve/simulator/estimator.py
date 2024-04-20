"""
An estimator class that can estimate the time consumption of one prefill/decoding pass
"""
import json

class Estimator:
    def __init__(
        self,
        profiler_data_path: str,
        model_name: str,
        tp_world_size: int,
        pp_world_size: int
    ):
        self.profiler_data = json.load(open(profiler_data_path, "r"))
        self.model_name = model_name
        self.tp_world_size = tp_world_size
        self.pp_world_size = pp_world_size
        self.pp_factor = 1.0 / self.pp_world_size   # TODO maybe need to be multiplied
        
        self.prefill_abc = self.profiler_data[model_name][str(tp_world_size)]["prefill"]
        self.decoding_abc = self.profiler_data[model_name][str(tp_world_size)]["decoding"]
    
    def estimate_prefill_time_ms(self, num_total_tokens: int, sum_num_tokens_sqr: int) -> float:
        return (self.prefill_abc[0] + self.prefill_abc[1]*num_total_tokens + self.prefill_abc[2]*sum_num_tokens_sqr) * self.pp_factor

    def estimate_decoding_time_ms(self, num_total_tokens: int, batch_size: int) -> float:
        return (self.decoding_abc[0] + self.decoding_abc[1]*num_total_tokens + self.decoding_abc[2]*batch_size) * self.pp_factor
    
    