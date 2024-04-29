import dataclasses
from typing import List, Literal, Optional

from simdistserve.base.request import Request

Evalstr = str


@dataclasses.dataclass
class WorkloadComment:
    type_: Literal['Poisson', 'Gamma', 'Fixed']
    rate_: float
    cv: float
    workload: Literal['ShareGPT', 'FixedLength']
    comment: str = None


@dataclasses.dataclass
class DisaggRunParam:
    name: str  # Experiment name
    arrival: 'List[float]'
    requests: 'List[Request] | Evalstr'
    N_prefill_instance: int
    N_decode_instance: int
    # TODO: Get PP_prefill_inter, PP_decode_inter, PP_prefill_intra, PP_decode_intra, TP_prefill, TP_decode
    PP_prefill: int
    PP_decode: int
    # worker configs
    prefill_max_batch_size: int
    model_type: 'ModelTypes | str'
    TP_Prefill: int
    TP_Decode: int
    chunked_prefill_max_tokens: int

    TP: int = 1  # TODO: Deprecate TP - instead, use TP_prefill
    workload_comment: Optional[WorkloadComment] = None

    def __hash__(self):
        arrival = hash(tuple(self.arrival))
        requests = hash(tuple(self.requests) if isinstance(self.requests, list) else self.requests)
        # hash everything except for workload_comment, and the original arrival and requests (use new)
        return hash((
            self.name,
            arrival,
            requests,
            self.N_prefill_instance,
            self.N_decode_instance,
            self.PP_prefill,
            self.PP_decode,
            self.prefill_max_batch_size,
            self.model_type,
            self.TP,
            self.chunked_prefill_max_tokens,
        ))

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"N_prefill_instance={self.N_prefill_instance}, "
            f"N_decode_instance={self.N_decode_instance}, "
            f"PP_prefill={self.PP_prefill}, "
            f"PP_decode={self.PP_decode}, "
            f"prefill_max_batch_size={self.prefill_max_batch_size}, "
            f"model_type={self.model_type}, "
            f"TP={self.TP}, "
            f"chunked_prefill_max_tokens={self.chunked_prefill_max_tokens}"
            f")"
        )

    __repr__ = __str__


@dataclasses.dataclass
class VLLMRunParam:
    name: str  # Experiment name
    arrival: 'List[float]'
    requests: 'List[Request] | Evalstr'
    N_instance: int
    PP: int
    # worker configs
    prefill_max_batch_size: int
    model_type: 'ModelTypes | str'
    TP_Prefill: int
    TP_Decode: int
    chunked_prefill_max_tokens: int

    TP: int = 1  # TODO: Deprecate TP - instead, use TP_prefill

    def __hash__(self):
        arrival = hash(tuple(self.arrival))
        requests = hash(tuple(self.requests) if isinstance(self.requests, list) else self.requests)
        # hash everything except for workload_comment, and the original arrival and requests (use new)
        return hash((
            self.name,
            arrival,
            requests,
            self.N_instance,
            self.PP,
            self.prefill_max_batch_size,
            self.model_type,
            self.TP,
            self.chunked_prefill_max_tokens,
        ))

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"N_instance={self.N_instance}, "
            f"PP={self.PP}, "
            f"prefill_max_batch_size={self.prefill_max_batch_size}, "
            f"model_type={self.model_type}, "
            f"TP={self.TP}, "
            f"chunked_prefill_max_tokens={self.chunked_prefill_max_tokens}"
            f")"
        )

    __repr__ = __str__
