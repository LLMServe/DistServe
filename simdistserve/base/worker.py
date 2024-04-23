"""
Worker class for simulation. One worker class manages a TP group.
"""
import random
import warnings
from collections import deque
from typing import Optional, List, Iterable, TYPE_CHECKING, Union, TypedDict, Literal
from uuid import UUID

from simdistserve.estimators.time_estimator import get_prefill_time, get_decode_time

if TYPE_CHECKING:
    from simdistserve.base.scheduler import Scheduler
    from simdistserve.base.request import Request


# TODO: (Refactor) Make this a configuration.
class WorkerConfig(TypedDict):
    """Behaviors of worker."""
    TP_Prefill: int  # Tensor parallelism for prefill (default = 1)
    TP_Decode: int  # Tensor parallelism for decode (default = 1)
    model_type: str  # Model type for prefill/decode time calculation (default = ModelType.opt_13b)
    prefill_max_batch_size: int  # Maximum number of prefill request in a batch (default = 10**7)
    decode_max_batch_size: int  # Maximum number of decode request in a batch (default = 10**7)
    prefill_max_tokens: int  # Max tokens in prefill iteration (default = 10**7)
    decode_max_tokens: int  # Max tokens in a iteration forward (default = 10**7)
    enable_chunked_prefill: Optional[bool]  # Enable memory pressure simulation (default = False)
    engine_type: Literal["distserve", "vllm"]  # Engine type for prefill/decode time calculation (default = "distserve")

    # TODO: Deprecated
    TP: Optional[int]  # Tensor parallelism (default = 1)
    pass


class Worker:
    def __init__(
        self, env, wid,
        cluster=None,
        is_last_in_pipeline: bool = False,
        pipe_rank: int = None,
        should_request_stay: bool = True,
        prefill_max_batch_size: int = 10 ** 7,
        decode_max_batch_size: int = 10 ** 7,
        global_scheduler: 'Scheduler' = None,
        model_type: str = None,
        TP: int = 1,
        TP_Prefill: int = None,
        TP_Decode: int = None,
        enable_chunked_prefill=False,
        prefill_max_tokens=10 ** 7,
        decode_max_tokens=10 ** 7,
        decode_back_pressure: float = 0.9,
        engine_type: Literal["distserve", "vllm"] = "distserve",
    ):
        self.env = env
        self.cluster = cluster  # Refer to the cluster of init.
        self.wid = wid
        self.pipe_rank = pipe_rank
        self.is_last_in_pipeline = is_last_in_pipeline
        self.next_worker: 'Optional[Worker]' = None
        self.model_type = model_type

        # TODO: (Deprecate) TP should be deprecate in favor of TP_prefill and TP_decode.
        self.TP = TP
        self.TP_Prefill = TP_Prefill
        self.TP_Decode = TP_Decode
        if (self.TP_Prefill is None) and (self.TP_Decode is None):
            warnings.warn(f"TP_Prefill and TP_Decode are not set. Default to {TP = } only apply to prefill.")
            self.TP_Prefill = TP
            self.TP_Decode = 1
        elif (self.TP_Prefill is not None) and (self.TP_Decode is not None):
            # Using the new TP_prefill and TP_decode value, instead of TP.
            pass
        elif (self.TP_Prefill is None) or (self.TP_Decode is None):
            warnings.warn(f"{TP = } will be deprecated soon. Use TP_Prefill and TP_Decode.")
            self.TP_Prefill = TP
            self.TP_Decode = 1
            pass

        # Same request should stay in the same worker.
        # If set to false, then it will forward to the global scheduler.
        self.global_scheduler = global_scheduler
        self.should_request_stay: bool = should_request_stay
        # Maximum number requests to fill in prefill batch. (Default 0 => 10 ** 7, big enough number)
        self.prefill_max_batch_size: int = prefill_max_batch_size if prefill_max_batch_size > 0 else 10 ** 7
        self.decode_max_batch_size: int = decode_max_batch_size if decode_max_batch_size > 0 else 10 ** 7
        # Maximum number of tokens for a prefill request to batch.
        self.prefill_max_tokens: int = prefill_max_tokens if prefill_max_tokens > 0 else 10 ** 7
        self.decode_max_tokens: int = decode_max_tokens if decode_max_tokens > 0 else 10 ** 7
        # Enable chunked prefill (if True) or prioritization scheduling (if False)
        self.enable_chunked_prefill: bool = enable_chunked_prefill
        # Decode worker stop accepting incoming request when this is full.
        self.decode_back_pressure = decode_back_pressure

        self.prefill_queue: 'deque[Request]' = deque()
        self.decode_queue: 'deque[Request]' = deque()
        self._prefill_ips: int = 0  # Elements in progress for prefill
        self._decode_ips: int = 0  # Elements in progress for decode
        self._wakeup_event = env.event()
        self.log: 'list[tuple[float, str, int, int, int, list[int], list[int]]]' = []

        # Simulate scheduler delay in terms of number of decode rounds.
        self._prefill_sched_delay: int = 0
        self.engine_type = engine_type
        pass

    @property
    def is_first_in_pipeline(self):
        return self.pipe_rank == 0

    @property
    def has_back_pressure(self) -> bool:
        threshold = int(self.decode_max_batch_size * self.decode_back_pressure)
        return sum(r.current_context_len for r in self.decode_queue) > threshold

    def __repr__(self):
        return f"Worker {self.wid}"

    def _log_event(self, event, num_tokens: int = 0, prefill_bs=0, decode_bs=0,
                   prefill_len_list=None, decode_len_list=None):
        if prefill_len_list is None:
            prefill_len_list = []
        if decode_len_list is None:
            decode_len_list = []
        item = (self.env.now, event, num_tokens, prefill_bs, decode_bs, prefill_len_list, decode_len_list)
        self.log.append(item)
        # print(item)
        return

    def run(self):
        while True:
            if not (self.prefill_queue or self.decode_queue):
                yield self._wakeup_event

            if self.prefill_queue and not self.has_back_pressure:
                yield from self.do_prefill()
            else:
                yield from self.do_decode()

            self._log_event("wait")
            pass

        pass

    def add_ray_overhead(self, sum_of_tokens) -> int:
        base_overhead = 2
        k = 0.0001
        delay = base_overhead + sum_of_tokens * k
        return delay

    # run = run_with_schedule_delay

    def wakeup(self):
        self._wakeup_event.succeed()
        self._wakeup_event = self.env.event()
        return

    def forward_prefill(self, items):
        # if items is not iterable, then make it iterable
        if not items:
            return
        if not isinstance(items, Iterable):
            items = [items]

        self.next_worker.prefill_queue.extend(items)
        self.next_worker.wakeup()
        return

    def forward_decode(self, items: Union['Request', Iterable['Request']], to_scheduler: bool = False):
        if not items:
            return
        if not isinstance(items, Iterable):
            items = [items]

        if not to_scheduler:
            self.next_worker.decode_queue.extend(items)
            self.next_worker.wakeup()
            return

        for item in items:
            self.global_scheduler.schedule_decode(item)
        return

    def _enter_decodes(self, remaining_tok_in_batch: int) -> 'List[Request]':
        # decode_max_tokens

        # Acceptable decode requests is capped by the remaining allowed tokens in this batch.
        decode_max_tokens = self.decode_max_tokens
        _decode_len = min(remaining_tok_in_batch, len(self.decode_queue))
        decode_reqs = []
        for i in range(_decode_len):
            req = self.decode_queue[0]
            # TODO: could be off by one (current_context_len + 1)
            if (req.current_context_len + 1) > decode_max_tokens:
                print(f"### Decode request is too big. Decode max token limit reached: {self.decode_max_tokens}.")
                break
            decode_max_tokens -= (req.current_context_len + 1)
            decode_reqs.append(self.decode_queue.popleft())
        for r in decode_reqs:
            r.do_decode(wid=self.wid)
        return decode_reqs

    def _enter_prefill(self) -> 'List[Request]':
        result: 'List[Request]' = []

        # Limit the maximum prefill requests to handle.
        max_request_size = min(self.prefill_max_batch_size, len(self.prefill_queue))

        # TODO: (Refactor) This logic becomes spaghetti.
        # If worker is not the first in pipeline, then it will just identify the chunks of prefill.
        if not self.is_first_in_pipeline:
            # Then just fetch all decode with the same chunk-id.
            chunk_id = self.prefill_queue[0].chunk_id
            for i in range(max_request_size):
                candidate: 'Request' = self.prefill_queue[0]
                if candidate.chunk_id != chunk_id:
                    break
                result.append(self.prefill_queue.popleft())
            pass

        else:
            # Worker is the first in pipeline, then it will do chunked prefill.
            chunk_size = 0
            prefill_max_tokens = self.prefill_max_tokens
            # chunk_id assign as uuid
            chunk_id = UUID(int=random.getrandbits(128))
            for _ in range(max_request_size):
                candidate: 'Request' = self.prefill_queue[0]

                if self.enable_chunked_prefill:
                    # The prefill portion that we picked from the candidate.
                    sched_size = min(
                        # The to-schedule size is the minimum of
                        # (1) the remaining prefill size of the candidate, and
                        # (2) the maximum allowed size of a chunked-prefill batch.
                        # This way we greedily cut and schedule the prefill chunk.
                        candidate.remain_prefill_lens,
                        prefill_max_tokens - chunk_size  # max batch size in a chunked-prefill batch - chunk size
                    )
                    if sched_size <= 0:
                        break
                else:
                    # If the whole request can fit into the chunk,
                    # then just schedule the whole request.
                    sched_size = candidate.remain_prefill_lens
                    if sched_size > prefill_max_tokens:
                        break
                    pass

                # Candidate is picked. Now fill in the chunked-prefill information.
                candidate.current_prefill_lens = sched_size
                candidate.remain_prefill_lens -= sched_size
                prefill_max_tokens -= sched_size
                candidate.chunk_id = chunk_id
                chunk_size += sched_size
                assert candidate.remain_prefill_lens >= 0
                result.append(self.prefill_queue.popleft())
                pass

        if self.prefill_queue:
            print("### Prefill queue is not empty. Prefill max token limit reached.")
            pass
        for i in result:
            i.do_prefill(wid=self.wid)
        return result

    def _exit_prefill(self, prefill_items: List['Request']):
        for item in prefill_items:
            next_wid = self.next_worker.wid if self.next_worker else None
            item.finish_prefill(is_finished_one_round=self.is_last_in_pipeline, wid=self.wid, next_wid=next_wid)
            if not self.is_last_in_pipeline or (item.remain_prefill_lens > 0):
                # Finish one chunk of prefill. Now forward to the next worker
                # (or head of worker) to do the rest of the parts.
                self.forward_prefill(item)
                continue

            # Arrive at worker who is at the last of pipeline.
            if item.should_finish():
                # ... just a sanity check to avoid any infinite loop.
                continue
            self.forward_decode(item, to_scheduler=(not self.should_request_stay))
        return

    def _exit_decode(self, decode_reqs):
        if not decode_reqs:
            return
        next_wid = self.next_worker.wid if self.next_worker else None
        for r in decode_reqs:
            r.finish_decode(is_finished_one_round=self.is_last_in_pipeline, next_wid=next_wid)
        next_decode_batch = tuple(r for r in decode_reqs if not r.should_finish())
        self.forward_decode(next_decode_batch)
        return

    def do_prefill(self):
        prefill_items: 'List[Request]' = self._enter_prefill()
        if self.enable_chunked_prefill:
            remaining_tok_in_batch = self.prefill_max_tokens - sum(x.current_prefill_lens for x in prefill_items)
            decode_reqs = self._enter_decodes(remaining_tok_in_batch)
        else:
            decode_reqs = []
        # TODO: (Refactor) The `num_tokens` may be used inaccurately in the get prefill time function.
        num_tokens = sum(x.current_prefill_lens for x in prefill_items)
        num_tokens += len(decode_reqs)

        self._log_event(
            "do_prefill",
            num_tokens=num_tokens,
            prefill_bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )

        # Get prefill time wrt total number of tokens.
        delay = get_prefill_time(
            num_tokens,
            bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            pp=self.cluster.PP_prefill,
            model_type=self.model_type, TP=self.TP_Prefill,
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            engine_type=self.engine_type,
            # __prefill_reqs=prefill_items,
            # __decode_reqs=decode_reqs,
        )
        num_tokens = sum(x.current_context_len for x in (prefill_items + decode_reqs))
        if self.is_first_in_pipeline:
            delay += self.add_ray_overhead(num_tokens)
        # Set the number of prefills in progress such that the scheduler get proper information about the worker.
        self._prefill_ips = len(prefill_items)
        yield self.env.timeout(delay)
        self._prefill_ips = 0
        self._exit_prefill(prefill_items)
        self._exit_decode(decode_reqs)
        return

    def do_decode(self):
        decode_reqs = self._enter_decodes(self.decode_max_tokens)
        batch_size = len(decode_reqs)
        self._log_event(
            "do_decode", num_tokens=batch_size, decode_bs=batch_size,
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )
        _token_generated_list = [x.current_context_len + 1 for x in decode_reqs]
        delay = get_decode_time(batch_size, pp=self.cluster.PP_decode,
                                model_type=self.model_type, TP=self.TP_Decode,
                                token_generated_list=_token_generated_list,
                                engine_type=self.engine_type, )
        num_tokens = sum(x.current_context_len for x in decode_reqs)
        if self.is_first_in_pipeline:
            delay += self.add_ray_overhead(num_tokens)
        yield self.env.timeout(delay)
        self._exit_decode(decode_reqs)
        return

    pass
