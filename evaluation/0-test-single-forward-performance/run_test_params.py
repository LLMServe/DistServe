import numpy as np

from structs import *
from db import RecordManager
from sut.abstract_sut import SystemUnderTest

def run_test_params(
    sut_class: type[SystemUnderTest],
    db_path: str,
    testing_params: list[TestParamGroup],
    warmup_rounds: int = 1,
    measure_rounds: int = 3,
    skip_duplicated: bool = True,
    store_into_db: bool = True
):
    """
    Create a SUT, run it on a series of TestParamGroups, and store the results into a database
    """
    record_manager = RecordManager(db_path)
    total_num_params = sum([len(test_param_group.input_params) for test_param_group in testing_params])
    num_finished_params = 0
    print(f"Total number of params to run: {total_num_params}")
    for test_param_group in testing_params:
        worker_param = test_param_group.worker_param
        # Check if we've already tested this set of parameters
        all_records_exist = True
        for input_param in test_param_group.input_params:
            if record_manager.query_record(worker_param, input_param) is None:
                all_records_exist = False
                break
        if all_records_exist and skip_duplicated:
            print(f"Record for test param group with {worker_param=} already exists. Continued")
            num_finished_params += len(test_param_group.input_params)
            continue

        print("==================================")
        print("==================================")
        print(f"Creating SUT with worker param {worker_param}")
        sut = sut_class(worker_param, test_param_group.input_params)

        for input_param in test_param_group.input_params:
            print("--------------------")
            print(f"Progress: {num_finished_params} / {total_num_params} ({num_finished_params/total_num_params*100:.2f}%)")
            print(f"Input param: {input_param}")
            if skip_duplicated and record_manager.query_record(worker_param, input_param) is not None:
                print("Skipped")
                num_finished_params += 1
                continue

            # Warm up the workers
            print(f"Warming up")
            for _ in range(warmup_rounds):
                sut.inference(input_param)

            # Benchmark
            print(f"Running")
            prefill_time_usages = []    # [measure_rounds]
            decoding_time_usages = []   # [measure_rounds*(output_len-1)]
            for _ in range(measure_rounds):
                input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usage = sut.inference(input_param)
                prefill_time_usages.append(prefill_time_usage)
                decoding_time_usages.extend(decoding_time_usage)
            avg_prefill_time_usage = np.mean(prefill_time_usages)
            avg_decoding_time_usage = np.median(decoding_time_usages)
            prefill_time_stddev = np.std(prefill_time_usages)
            decoding_time_stddev = np.std(decoding_time_usages)

            print(f"Pred output[0]: {predict_texts[0]}")
            print(f"Prefill time usage: avg {avg_prefill_time_usage}, stddev {prefill_time_stddev}")
            print(f"Decoding time usage: avg {avg_decoding_time_usage}, stddev {decoding_time_stddev}")
            # print(f"Prefill time usages: {prefill_time_usages}")
            # print(f"Decoding time usages: {decoding_time_usages}")
            if store_into_db:
                record_manager.update_or_insert_record(
                    worker_param,
                    input_param,
                    avg_prefill_time_usage,
                    avg_decoding_time_usage,
                    prefill_time_stddev,
                    decoding_time_stddev
                )
            num_finished_params += 1
        del sut
        