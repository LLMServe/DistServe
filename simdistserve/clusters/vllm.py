from functools import reduce
from itertools import chain
from typing import Optional

from simdistserve.base.scheduler import Scheduler
from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.utils import set_next_worker


class VLLMCluster:
    def __init__(
        self,
        env,
        N_instance: int = 1,
        PP: int = 1,
        worker_configs: 'Optional[WorkerConfig]' = None,
    ):
        worker_kwargs = dict(
            global_scheduler=None,
            # is_last_in_pipeline
            # should_request_stay: bool = True,
            # prefill_max_batch_size: int = 0,
            # global_scheduler: 'Scheduler' = None,
            **(worker_configs or {})
        )
        instances = []

        worker_id = 0
        for inst_id in range(N_instance):
            instance = []
            for i, p in enumerate(range(PP)):
                worker = Worker(env, worker_id, cluster=self, PP=PP, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1

            # Cyclically chain instance within a GPU
            reduce(
                set_next_worker,
                chain(instance, (instance[0],))
            )
            instance[-1].is_last_in_pipeline = True
            instances.append(instance)
            pass

        self.env = env
        self.PP_prefill = PP
        self.PP_decode = PP
        self.instances = instances
        self.scheduler = Scheduler(env, prefill_heads=[
            i[0] for i in instances
        ], decode_heads=[
            i[0] for i in instances
        ])
        pass

    def get_all_workers(self):
        return list(chain.from_iterable(self.instances))

    def run(self):
        for instance in self.instances:
            for worker in instance:
                self.env.process(worker.run())
        return self
