"""Sampling parameters for text generation."""
from typing import List, Optional, Union, Tuple
import time

from distserve.utils import Counter
from distserve.config import (
    ParallelConfig
)

class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        use_beam_search: Whether to use beam search instead of sampling.
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
    """

    _SAMPLING_EPS = 1e-5

    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Union[None, str, List[str]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
    ) -> None:
        self.n = n
        self.best_of = best_of if best_of is not None else n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        if stop is None:
            self.stop = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = list(stop)
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.logprobs = logprobs

        self._verify_args()
        if self.use_beam_search:
            self._verity_beam_search()
        elif self.temperature < self._SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self._verify_greedy_sampling()

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.best_of < self.n:
            raise ValueError(
                f"best_of must be greater than or equal to n, "
                f"got n={self.n} and best_of={self.best_of}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disable), or at least 1, " f"got {self.top_k}."
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}.")

    def _verity_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError(
                "best_of must be greater than 1 when using beam "
                f"search. Got {self.best_of}."
            )
        if self.temperature > self._SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")

    def _verify_greedy_sampling(self) -> None:
        if self.best_of > 1:
            raise ValueError(
                "best_of must be 1 when using greedy sampling." f"Got {self.best_of}."
            )
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"best_of={self.best_of}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"use_beam_search={self.use_beam_search}, "
            f"stop={self.stop}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"logprobs={self.logprobs})"
        )


class Request:
    """A request contains the user's prompt, generated tokens and related information.
    Args:
        arrival_time: the absolute or relative time when the request arrives.
        request_id: the unique identifier for the request.
        prompt: the prompt provided by the user.
        prompt_token_ids: the token ids of the prompt.
        sampling_params: sampling parameters for the request.
        priority: the priority of this request, default is 0.
    """

    def __init__(
        self,
        arrival_time: float,
        request_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams = SamplingParams(),
        priority: int = 0
    ):
        # static states
        self.arrival_time = arrival_time
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.sampling_params = sampling_params

        # dynamic states
        self.generated_tokens = []
        self.generated_token_ids = []
        self.is_finished = False
        self.is_running = False

        self.process_time = 0.0
        self.last_step_time = 0.0

        self.priority = priority

    def get_priority(self) -> int:
        return self.priority
    
    def set_priority(self, priority: int) -> None:
        self.priority = priority

    def _check_finish_condition(self):
        if self.get_output_len() >= self.sampling_params.max_tokens:
            self.is_finished = True

        if not self.sampling_params.ignore_eos:
            if self.get_output_len() and (
                self.generated_tokens[-1] in self.sampling_params.stop
            ):
                self.is_finished = True

    def add_generated_token(self, token: str, token_id: int):
        if self.get_output_len() > self.sampling_params.max_tokens:
            raise ValueError(
                f"The generated tokens exceed the maximum output length {self.sampling_params.max_tokens}\
                for request {self.request_id}."
            )
        self.generated_tokens.append(token)
        self.generated_token_ids.append(token_id)
        self._check_finish_condition()

    def is_context_stage(self) -> bool:
        return len(self.generated_tokens) == 0

    def get_input_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        assert len(self.generated_tokens) == len(self.generated_token_ids)
        return len(self.generated_token_ids)

    def get_response(self) -> str:
        return "".join(self.generated_tokens)

    def get_input_tokens_ids(self) -> List[int]:
        """The token ids of the input tokens for the next iteration.
        For request in the context stage, this is equal to the prompt_token_ids.
        For request in the decoding stage, this is equal to the newly generated token.
        """
        if self.is_context_stage():
            return self.prompt_token_ids
        else:
            # this is generally not true, if speculative decoding is used
            return [self.generated_token_ids[-1]]

    def get_num_input_tokens(self) -> int:
        return len(self.get_input_tokens_ids())

    def get_first_new_token_index(self) -> int:
        """The index of the first newly generated tokens.
        In the decoding phase, only the input tokens need to compute QKV. The index
        of the first token in the input tokens of next round is needed to do positional
        embedding and decoding phase kernel correctly.

        Note: Currently, the last token in self.output_tokens is the first newly generated token.
        This might not be true if speculative decoding is used.
        """
        return (
            0
            if self.is_context_stage()
            else self.get_input_len() + self.get_output_len() - 1
        )

    def get_process_time(self) -> float:
        return self.process_time

    def reset_process_time(self) -> None:
        self.process_time = 0.0

    def add_process_time(self, running_time: float) -> None:
        self.process_time += running_time

    def get_kvcache_slots(self) -> float:
        """The number of kvcache slots needed for the request.
        The number of kvcache slots is the total number of tokens in the request.
        """
        return self.get_input_len() + self.get_output_len()

    def __repr__(self) -> str:
        return (
            f"Request(arrival_time = {self.arrival_time}, "
            f"request_id={self.request_id}, "
            f"prompt={self.prompt}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"generated_tokens={self.generated_tokens}, "
            f"generated_token_ids={self.generated_token_ids}, "
            f"is_context_stage={self.is_context_stage()}, "
            f"is_finished={self.is_finished})"
        )

    def __str__(self) -> str:
        return f"Request {self.request_id}: {self.prompt} {self.get_response()}"


class BatchedRequests:
    def __init__(
        self,
        requests: Optional[List[Request]] = None,
    ) -> None:
        if requests is None:
            self.requests = []
        else:
            self.requests = requests
        self.start_time = None

    def __len__(self):
        return len(self.requests)

    def __str__(self) -> str:
        return f"BatchedRequests: {self.requests}"

    def __repr__(self) -> str:
        return f"BatchedRequests: {self.requests}"

    def add_request(self, request: Request):
        assert (
            request.request_id not in self.get_request_ids()
        ), f"request {request.request_id} already exists"
        self.requests.append(request)

    def pop_finished_requests(self) -> List[Request]:
        finished_requests, unfinished_requests = [], []
        for request in self.requests:
            if request.is_finished:
                finished_requests.append(request)
            else:
                unfinished_requests.append(request)
        self.requests = unfinished_requests
        return finished_requests

    def start_one_iteration(self, start_time):
        """Update the start time of the batch before its execution of iteration."""
        assert self.start_time is None, "the batch has already started one iteration"
        self.start_time = start_time
        self.is_running = True

    def finish_one_iteration(
        self,
        generated_tokens: List[str],
        generated_tokens_ids: List[int],
        end_time: float,
    ):
        """Update the requests in the batch after it finishes one iteration
        Note: the order of generated tokens should align with self.requests.
        """
        assert self.start_time is not None, "the batch has not been started"
        for request, generated_token, generated_token_id in zip(
            self.requests, generated_tokens, generated_tokens_ids
        ):
            request.last_step_time = end_time
            request.add_process_time(end_time - self.start_time)
            request.add_generated_token(generated_token, generated_token_id)
        self.start_time = None
        self.is_running = False

    #### General Getters
    def get_request_ids(self) -> List[int]:
        return [request.request_id for request in self.requests]

    def get_kvcache_slots(self) -> int:
        return sum([request.get_kvcache_slots() for request in self.requests])

    def get_num_input_tokens(self) -> int:
        return sum([request.get_num_input_tokens() for request in self.requests])

    #### Getters for the GPT operator parameters ####
    def get_input_tokens_batched(self) -> List[List[int]]:
        return [request.get_input_tokens_ids() for request in self.requests]

    def get_first_token_indexes(self) -> List[int]:
        return [request.get_first_new_token_index() for request in self.requests]

    def get_is_context_stage(self) -> List[bool]:
        return [request.is_context_stage() for request in self.requests]


def create_request(
    prompt: Optional[str],
    prompt_token_ids: Optional[List[str]],
    sampling_params: SamplingParams,
    request_counter: Counter,
    tokenizer,
    arrival_time: Optional[float] = None,
    request_id: Optional[int] = None,
) -> Request:
    if request_id is None:
        request_id = next(request_counter)
    if prompt_token_ids is None:
        assert prompt is not None
        prompt_token_ids = tokenizer.encode(prompt)
    if prompt is None:
        assert prompt_token_ids is not None
        prompt = tokenizer.decode(prompt_token_ids)
    if arrival_time is None:
        arrival_time = time.time()
    return Request(
        arrival_time,
        request_id,
        prompt,
        prompt_token_ids,
        sampling_params,
    )


class MigratingRequest:
    """
    MigratingRequest: elements in the "bridge" queue.
    
    Each MigratingRequest represents a request that:
      - Has finished the context stage
      - Has not yet acceptted by the decoding stage
      - Its block is still on context stage's GPU memory (i.e. migration needed)
      
    Those requests are produced by ContextStageLLMEngine, queued in the "bridge"
    queue, and finally consumed by DecodingStageLLMEngine, which forms a
    producer-consumer pattern.
    
    For more information about the design & implementation of disaggregation,
    please refer to engine.py.
    """
    
    def __init__(
        self,
        req: Request,
        block_indexes: List[int],
        context_parallel_config: ParallelConfig,
    ):
        self.req = req
        self.block_indexes = block_indexes
        self.context_parallel_config = context_parallel_config
        