from queue import Queue
from typing import List, TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from simdistserve.base.request import Request
    from simdistserve.base.worker import Worker


class Scheduler:
    def __init__(self, env, prefill_heads, decode_heads):
        self.env = env
        self._prefill_heads: 'List[Worker]' = prefill_heads
        self._prefill_queues = [i.prefill_queue for i in self._prefill_heads]
        self._decode_heads: 'List[Worker]' = decode_heads
        self._decode_queues = [i.decode_queue for i in self._decode_heads]
        pass

    @staticmethod
    def _find_best_worker_and_queue(workers, queues) -> 'Tuple[Worker, Union[Queue, List]]':
        # Peak the queue to find the least loaded worker.
        # Assume round-robin
        # Add the pending tasks in prefill
        worker, queue = min(zip(workers, queues), key=lambda x: x[0]._prefill_ips + len(x[1]))
        return worker, queue

    @staticmethod
    def _sched_request(req, worker, queue):
        queue.append(req)
        worker.wakeup()
        return

    def schedule_new_req(self, req: 'Request'):
        if req.counter < 0:
            return self.schedule_prefill(req)
        # This is for the 'decode-only' case.
        return self.schedule_decode(req)

    def schedule_prefill(self, req: 'Request'):
        assert req.counter < 0
        worker, queue = self._find_best_worker_and_queue(self._prefill_heads, queues=self._prefill_queues)
        self._sched_request(req, worker, queue)
        return

    def schedule_decode(self, req: 'Request'):
        assert req.counter >= 0
        if req.should_finish():
            # Force request to quit.
            req.finish_decode()
            return

        worker, queue = self._find_best_worker_and_queue(self._decode_heads, queues=self._decode_queues)
        req.wait_decode(worker.wid) # Artifact to prevent request having FTL != 0 when decode only.
        self._sched_request(req, worker, queue)
        return

    pass


# TODO: (Deprecate)
def put_request(env, scheduler: 'Scheduler', delays, requests):
    for r, delay in zip(requests, delays):
        r.init()
        scheduler.schedule_new_req(r)
        yield env.timeout(delay)
    return


def put_request_at_time(env, scheduler: 'Scheduler', time, request: 'Request'):
    yield env.timeout(time)
    request.init()
    scheduler.schedule_new_req(request)
    return


def put_requests_with_interarrivals(env, scheduler: 'Scheduler', inter_arrivals, requests):
    """Put requests with the inter-arrivals."""
    assert len(inter_arrivals) == len(requests), (
        f"Number of requests ({len(requests)}) and inter-arrivals ({len(inter_arrivals)}) "
        f"should be the same."
    )
    wake_time = 0
    for r, ts in zip(requests, inter_arrivals):
        if r.env is None:
            r.env = env
        assert r.env == env
        wake_time += ts
        env.process(put_request_at_time(env, scheduler, wake_time, r))
    return
