"""
sut.py - Implement the abstract SystemUnderTest class
"""
import copy

from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer

from structs import *

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

class SystemUnderTest(ABC):
    @abstractmethod
    def __init__(
        self,
        worker_param: WorkerParam,
        input_params: list[InputParam]
    ):
        pass

    @abstractmethod
    def inference(
        self,
        input_param: InputParam
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], float, list[float]]:
        # Return: input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usages
        pass
    