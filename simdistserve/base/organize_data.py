from typing import TypedDict
import numpy as np
import pandas as pd


class Request_t(TypedDict):
    req_id: int
    prefill_lens: int
    output_lens: int
    pass


class RequestLog_t(TypedDict):
    start_time: float
    end_time: float
    event_type: str
    req_id: int
    duration: float
    pass


class WorkerLog_t(TypedDict):
    start_time: float
    end_time: float
    event_type: str
    worker_id: int
    duration: float
    decode_batch_size: int
    prefill_batch: 'list[int]'
    decode_batch: 'list[int]'
    pass


class LatencyDist_t(TypedDict):
    first_token_latency: float
    decoding_latency: float
    tpot: float
    inv_tpot_ms: float
    inv_tpot_s: float
    pass


def organize_request_df(requests) -> 'DataFrame[Request_t]':
    """Describe the property of each request."""
    request_df = pd.DataFrame([
        {
            'req_id': r.req_id,
            'prefill_lens': r.prefill_lens,
            'output_lens': r.output_lens,
        }
        for r in requests
    ])
    return request_df


def transform_request_log_to_df(req: 'Request') -> 'DataFrame[RequestLog_t]':
    """
    Transform the request log into a DataFrame.
    :param req:
    :return: DataFrame
        req_id, start_time, end_time, duration, event_type
    """
    df = pd.DataFrame(req.log, columns=['start_time', 'event_type', 'worker_id'])
    df['req_id'] = req.req_id
    df['duration'] = df['start_time'].shift(-1) - df['start_time']
    df['duration'] = df['duration'].fillna(0)
    df['end_time'] = df['start_time'] + df['duration']
    return df


def organize_request_event_df(requests) -> 'DataFrame[RequestLog_t]':
    """Aggregate all request event logs into a single DataFrame."""
    request_event_df = pd.concat([
        transform_request_log_to_df(r)
        for r in requests
    ])
    return request_event_df


def transform_worker_log_to_df(worker: 'Worker') -> 'DataFrame[WorkerLog_t]':
    if not worker.log:
        return None
    df = pd.DataFrame(worker.log, columns=[
        'start_time', 'event_type', 'num_tokens', 'prefill_bs', 'decode_bs',
        'prefill_batch',
        'decode_batch'
    ])
    df['worker_id'] = worker.wid
    df['duration'] = df['start_time'].shift(-1) - df['start_time']
    df['duration'] = df['duration'].fillna(0)
    df['end_time'] = df['start_time'] + df['duration']
    return df


def organize_worker_event_df(cluster) -> 'DataFrame[WorkerLog_t]':
    """Aggregate all worker event logs into a single DataFrame."""
    worker_event_df = pd.concat([
        transform_worker_log_to_df(w)
        for w in cluster.get_all_workers()
    ])
    return worker_event_df


def calculate_per_request_latency(
    df: 'DataFrame[RequestLog_t]',
    output_lens: 'pd.Series' = None
) -> 'DataFrame[LatencyDist_t]':
    assert isinstance(output_lens, pd.Series) or output_lens is None, \
        f'output_lens must be a pd.Series, got {type(output_lens)}'
    # First token latency: time between first event and the first `wait_decode`
    # Decoding latency: time between first event and the last event
    first_event = df[df.event_type == 'init'].groupby('req_id').start_time.min()
    first_wait_decode = df[df.event_type == 'wait_decode'].groupby('req_id').start_time.min()
    last_event = df[df.event_type == 'exit_system'].groupby('req_id').end_time.max()

    # Then, calculate the first token latency and decoding latency for each req_id
    first_token_latency = first_wait_decode - first_event
    decoding_latency = last_event - first_wait_decode
    total_latency = last_event - first_event

    dist_df = pd.DataFrame({
        'first_token_latency': first_token_latency,
        'decoding_latency': decoding_latency,
        'total_latency': total_latency,
    })

    if output_lens is not None:
        # If we have the request information, then we can also calculate the average time per-output-token.
        # Calculate the average time per output token in each request.
        tpot = decoding_latency.div(output_lens).replace([np.inf, - np.inf], 0)
        dist_df['tpot'] = tpot
        dist_df['inv_tpot_ms'] = 1 / tpot
        dist_df['inv_tpot_s'] = 1000 / tpot

    return dist_df
