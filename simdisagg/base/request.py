"""
Request class for the simulation.
"""
# Define the request *E_*vents
E_INIT = "init"
E_WAIT_PREFILL = "wait_prefill"
E_DO_PREFILL = "do_prefill"
E_WAIT_DECODE = "wait_decode"
E_DO_DECODE = "do_decode"
E_FINISH_PREFILL = "finish_prefill"
E_FINISH_DECODE = "finish_decode"
E_EXIT_SYSTEM = "exit_system"


class Request:

    # def __hash__(self):
    #     return hash((self.prefill_lens, self.output_lens))

    def __str__(self):
        return (
            f'Request('
            f'id={self.req_id},'
            f',prefill={self.prefill_lens}'
            f',output={self.output_lens}'
            f')'
        )

    __repr__ = __str__

    def __init__(
        self,
        env: 'simpy.Environment' = None,
        req_id: int = None,
        req_init_counter=-1,
        # TODO: (Refactor) Change name to `prefill_length` and `output_length`.
        prefill_length: int = 512,
        output_lens: int = 128,
        schedule_wait: int = 0
    ):
        assert req_id is not None, f'Request ID is not set.'
        self.env = env
        self.req_id = req_id
        # counter: int
        #  - counter < 0: steps to prefill. Only use the value `-1`. Reserve the negative space for future extension.
        #  - counter >=0: steps to decode. Maximum value is `output_lens - 1`, when the request should exit system.
        self.counter = req_init_counter
        self.log: 'list[tuple[float, str, int]]' = []
        self._terminated = False
        self.prefill_lens = prefill_length
        self.output_lens = output_lens
        self.schedule_wait = schedule_wait
        # `remain_prefill_lens`: length of prefill unscheduled.
        # Worker changes this value when it schedules the prefill.
        self.remain_prefill_lens: int = prefill_length
        # `current_prefill_lens`: length of prefill active.
        # Worker (usually the first worker in a prefill pipeline)
        # set this value when it schedules this request for chunked-prefill.
        self.current_prefill_lens: int = 0
        # Worker (usually the first worker in a prefill pipeline)
        # set this value if a request belongs to a particular chunk
        # The last worker in the pipeline unset this value at a chunk's end.
        self.chunk_id = None

    def _log_event(self, event, wid=-1):
        if not self.env:
            raise ValueError("Request.env is not set.")
        if self._terminated:
            return
        self.log.append((self.env.now, event, wid))
        return

    def init(self):
        self._log_event(E_INIT)

    def wait_prefill(self, wid=None):
        self._log_event(E_WAIT_PREFILL, wid=wid)

    def do_prefill(self, wid=None):
        self._log_event(E_DO_PREFILL, wid=wid)

    def wait_decode(self, wid=None):
        self._log_event(E_WAIT_DECODE, wid=wid)

    def do_decode(self, wid=None):
        self._log_event(E_DO_DECODE, wid=wid)

    def _reset_chunked_prefill_metadata(self):
        """Reset the metadata of chunked prefill."""
        self.chunk_id = None
        self.current_prefill_lens = 0
        return

    def finish_prefill(self, is_finished_one_round=False, wid=None, next_wid=None):
        if not is_finished_one_round:
            self.wait_prefill(wid=next_wid)
            return

        # Reset some properties of self.
        # TODO: (Rename) We only implement batched prefill, not chunked prefill
        self._reset_chunked_prefill_metadata()
        if self.remain_prefill_lens > 0:
            self.wait_prefill(wid=next_wid)
            return

        # All the prefills has been done.
        # Reset counter to 0
        # TODO: Should we do self.counter += 1?
        self.counter = 0
        # Hack to ensure "wait_decode" appears at least once.
        self.wait_decode(wid=next_wid)
        if not self.should_finish():
            return
        self._log_event(E_EXIT_SYSTEM)
        self._terminated = True
        return

    def finish_decode(self, is_finished_one_round=False, next_wid=None):
        if is_finished_one_round:
            self.counter += 1
        self.wait_decode(wid=next_wid)
        if self.should_finish():
            self._log_event(E_EXIT_SYSTEM)
            self._terminated = True
        return

    def should_finish(self):
        return self.counter >= self.output_lens
