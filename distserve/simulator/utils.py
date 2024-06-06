import dataclasses
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
    reqs: list[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
        )
        
@dataclasses.dataclass
class ReqResult:
    prompt_len: int
    output_len: int

    issue_time: float
    first_token_time: float
    complete_time: float
    
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    
    @staticmethod
    def from_http_request_result(
        prompt_len: int,
        output_len: int,
        issue_time: float,
        first_token_time: float,
        complete_time: float
    ):
        """
        benchmark-serving will call this function once a request is completed
        """
        return ReqResult(
            prompt_len,
            output_len,
            issue_time,
            first_token_time,
            complete_time,
            (first_token_time - issue_time)*1000,
            ((complete_time - first_token_time) / (output_len-1) if output_len > 1 else 0)*1000,
            (complete_time - issue_time)*1000
        )

def dump_req_result_list(reqs: list[ReqResult], output_path: str, print_metrics: bool = True):
    """
    Dump the request result to a file
    
    The first line in the file contains the string representation of the list of ReqResult.

    If print_metrics is True, then metrics will be appended to the file for better human readability.
    """
    with open(output_path, "w") as f:
        f.write(str([dataclasses.asdict(d) for d in reqs]) + "\n")
        if print_metrics:
            from metrics import BenchmarkMetrics
            metrics = BenchmarkMetrics.from_req_results(reqs)
            f.write(str(metrics))

def load_req_result_list(path: str) -> list[ReqResult]:
    with open(path, "r") as f:
        text = f.readline().strip()
        return [ReqResult(**d) for d in eval(text)]

# A barrier for synchronizing a fixed number of tasks
# Copied from https://github.com/python/cpython/blob/3.12/Lib/asyncio/locks.py
import enum
import asyncio
class _BarrierState(enum.Enum):
    FILLING = 'filling'
    DRAINING = 'draining'
    RESETTING = 'resetting'
    BROKEN = 'broken'
    
class Barrier(asyncio.mixins._LoopBoundMixin):
    """Asyncio equivalent to threading.Barrier

    Implements a Barrier primitive.
    Useful for synchronizing a fixed number of tasks at known synchronization
    points. Tasks block on 'wait()' and are simultaneously awoken once they
    have all made their call.
    """

    def __init__(self, parties):
        """Create a barrier, initialised to 'parties' tasks."""
        if parties < 1:
            raise ValueError('parties must be > 0')

        self._cond = asyncio.locks.Condition() # notify all tasks when state changes

        self._parties = parties
        self._state = _BarrierState.FILLING
        self._count = 0       # count tasks in Barrier

    def __repr__(self):
        res = super().__repr__()
        extra = f'{self._state.value}'
        if not self.broken:
            extra += f', waiters:{self.n_waiting}/{self.parties}'
        return f'<{res[1:-1]} [{extra}]>'

    async def __aenter__(self):
        # wait for the barrier reaches the parties number
        # when start draining release and return index of waited task
        return await self.wait()

    async def __aexit__(self, *args):
        pass

    async def wait(self):
        """Wait for the barrier.

        When the specified number of tasks have started waiting, they are all
        simultaneously awoken.
        Returns an unique and individual index number from 0 to 'parties-1'.
        """
        async with self._cond:
            await self._block() # Block while the barrier drains or resets.
            try:
                index = self._count
                self._count += 1
                if index + 1 == self._parties:
                    # We release the barrier
                    await self._release()
                else:
                    await self._wait()
                return index
            finally:
                self._count -= 1
                # Wake up any tasks waiting for barrier to drain.
                self._exit()

    async def _block(self):
        # Block until the barrier is ready for us,
        # or raise an exception if it is broken.
        #
        # It is draining or resetting, wait until done
        # unless a CancelledError occurs
        await self._cond.wait_for(
            lambda: self._state not in (
                _BarrierState.DRAINING, _BarrierState.RESETTING
            )
        )

        # see if the barrier is in a broken state
        if self._state is _BarrierState.BROKEN:
            raise ValueError("Barrier aborted")

    async def _release(self):
        # Release the tasks waiting in the barrier.

        # Enter draining state.
        # Next waiting tasks will be blocked until the end of draining.
        self._state = _BarrierState.DRAINING
        self._cond.notify_all()

    async def _wait(self):
        # Wait in the barrier until we are released. Raise an exception
        # if the barrier is reset or broken.

        # wait for end of filling
        # unless a CancelledError occurs
        await self._cond.wait_for(lambda: self._state is not _BarrierState.FILLING)

        if self._state in (_BarrierState.BROKEN, _BarrierState.RESETTING):
            raise ValueError("Abort or reset of barrier")

    def _exit(self):
        # If we are the last tasks to exit the barrier, signal any tasks
        # waiting for the barrier to drain.
        if self._count == 0:
            if self._state in (_BarrierState.RESETTING, _BarrierState.DRAINING):
                self._state = _BarrierState.FILLING
            self._cond.notify_all()

    async def reset(self):
        """Reset the barrier to the initial state.

        Any tasks currently waiting will get the BrokenBarrier exception
        raised.
        """
        async with self._cond:
            if self._count > 0:
                if self._state is not _BarrierState.RESETTING:
                    #reset the barrier, waking up tasks
                    self._state = _BarrierState.RESETTING
            else:
                self._state = _BarrierState.FILLING
            self._cond.notify_all()

    async def abort(self):
        """Place the barrier into a 'broken' state.

        Useful in case of error.  Any currently waiting tasks and tasks
        attempting to 'wait()' will have BrokenBarrierError raised.
        """
        async with self._cond:
            self._state = _BarrierState.BROKEN
            self._cond.notify_all()

    @property
    def parties(self):
        """Return the number of tasks required to trip the barrier."""
        return self._parties

    @property
    def n_waiting(self):
        """Return the number of tasks currently waiting at the barrier."""
        if self._state is _BarrierState.FILLING:
            return self._count
        return 0

    @property
    def broken(self):
        """Return True if the barrier is in a broken state."""
        return self._state is _BarrierState.BROKEN