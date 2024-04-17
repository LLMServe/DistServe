import math
import numpy as np
import dataclasses

from structs import TestRequest, Dataset, ReqResult

@dataclasses.dataclass
class BenchmarkMetrics:
    num_requests: int
    test_duration_ms: float

    request_throughput: float
    avg_per_token_latency_ms: float
    avg_input_token_latency_ms: float
    avg_ttft_ms: float
    avg_tpot_ms: float
    
    @staticmethod
    def from_req_results(data: list[ReqResult]):
        test_duration_ms = float(np.max([req.complete_time for req in data]) - np.min([req.issue_time for req in data]))*1000
        return BenchmarkMetrics(
            len(data),
            test_duration_ms,
            len(data) / (test_duration_ms / 1000),
            float(np.mean([req.latency_ms / (req.prompt_len+req.output_len) for req in data])),
            float(np.mean([req.ttft_ms / req.prompt_len for req in data])),
            float(np.mean([req.ttft_ms for req in data])),
            float(np.mean([req.tpot_ms for req in data if req.output_len != 0]))
        )
    
    def __str__(self) -> str:
        result = "{\n"
        for field in dataclasses.fields(self):
            result += f"\t{field.name}: {getattr(self, field.name)}\n"
        result += "}"
        return result
    