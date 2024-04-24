import dataclasses
from typing import List
import marshal

@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """
    
    prompt: str
    prompt_len: int
    output_len: int
    
@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    data: List[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "data": [(req.prompt, req.prompt_len, req.output_len) for req in self.data]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["data"]]
        )
        
import dataclasses
import numpy as np
from typing import List
import json

from distserve.lifetime import LifetimeEvent, LifetimeEventType, json_decode_lifetime_events

class RequestResult:
    """
    A class for storing the results of a single request
    """
    
    def __init__(
        self,
        prompt_len: int,
        output_len: int,
        start_time: float,
        end_time: float,
        token_timestamps: List[float],
        lifetime_events: List[LifetimeEvent] = None
    ):
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.start_time = start_time
        self.end_time = end_time
        self.token_timestamps = token_timestamps
        self.lifecycle_events = lifetime_events
        
        self.latency = end_time - start_time
        self.ftl = token_timestamps[0] - start_time
        self.tpot = 0 if output_len == 1 else (token_timestamps[-1] - token_timestamps[0]) / (output_len-1)

def read_request_results(path: str) -> List[RequestResult]:
    with open(path, "r") as f:
        request_results: List[RequestResult] = [
            RequestResult(
                item["prompt_len"],
                item["output_len"],
                item["start_time"],
                item["end_time"],
                item["token_timestamps"],
                json_decode_lifetime_events(item["lifecycle_events"]) if item.get("lifecycle_events", None) is not None else None
            )
            for item in json.load(f)
        ]
    return request_results

def count_valid_results(request_results: list[RequestResult], ftl: float, tpot: float) -> int:
    """
    count_valid_results: Count the number of requests that satisfy the given FTL and TPOT.
    """
    count = 0
    for req in request_results:
        if req.ftl <= ftl and req.tpot <= tpot:
            count += 1
    return count

def get_slo_attainment(request_results: list[RequestResult], ftl: float, tpot: float) -> float:
    """
    get_slo_attainment: Get the SLO attainment of the given request results under the given FTL and TPOT.
    """
    return count_valid_results(request_results, ftl, tpot) / len(request_results)
