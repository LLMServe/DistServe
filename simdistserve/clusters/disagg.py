from functools import reduce
from itertools import chain
from typing import List, Optional

from simdistserve.base.scheduler import Scheduler
from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.utils import set_next_worker


class DisaggCluster:
    def __init__(
        self,
        env,
        N_prefill_instance: int = 1,
        N_decode_instance: int = 1,
        PP_prefill: int = 1,
        PP_decode: int = 1,
        worker_configs: 'Optional[WorkerConfig]' = None,
    ):
        prefill_instances = []
        decode_instances = []

        worker_kwargs = dict(
            global_scheduler=None,
            # is_last_in_pipeline
            # should_request_stay: bool = True,
            # prefill_max_batch_size: int = 0,
            # global_scheduler: 'Scheduler' = None,
            **(worker_configs or {})
        )

        worker_id = 0
        for inst_id in range(N_prefill_instance):
            instance = []
            for i, p in enumerate(range(PP_prefill)):
                worker = Worker(env, worker_id, cluster=self, PP=PP_prefill, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1

            # Cyclically chain instance within a GPU
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            instance[-1].should_request_stay = False
            prefill_instances.append(instance)
            pass

        for inst_id in range(N_decode_instance):
            instance = []
            for i, p in enumerate(range(PP_decode)):
                worker = Worker(env, worker_id, cluster=self, PP=PP_decode, pipe_rank=i, **worker_kwargs)
                instance.append(worker)
                worker_id += 1

            # Cyclically chain instance within a GPU
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            decode_instances.append(instance)
            pass

        scheduler = Scheduler(env, prefill_heads=[
            i[0] for i in prefill_instances
        ], decode_heads=[
            i[0] for i in decode_instances
        ])

        for last_in_prefill in (instances[-1] for instances in prefill_instances):
            last_in_prefill.global_scheduler = scheduler

        self.env = env
        self.PP_prefill = PP_prefill
        self.PP_decode = PP_decode
        self.prefill_instances: 'List[List[Worker]]' = prefill_instances
        self.decode_instances: 'List[List[Worker]]' = decode_instances
        self.scheduler: 'Scheduler' = scheduler
        pass

    def get_all_workers(self):
        return list(
            chain(
                chain(*self.prefill_instances),
                chain(*self.decode_instances),
            )
        )

    def run(self):
        for instance in chain(self.prefill_instances, self.decode_instances):
            for worker in instance:
                self.env.process(worker.run())
        return self
