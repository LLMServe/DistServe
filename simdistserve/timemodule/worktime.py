# Fit a model where prefill does not have an intercept, and decode does have one.
import json
from pathlib import Path

from simdistserve.constants import ModelTypes


def load_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "profiler-a100-80g.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


profile_data = load_profile_data()


def get_prefill_time(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, **kw):
    a, b, c = profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]["prefill"]
    pp_factor = 1 / pp
    pp_const = 1 * pp  # TODO: Modulate the PP overhead
    num_total_tokens = sum(prefill_len_list)
    sum_num_tokens_sqr = sum([x ** 2 for x in prefill_len_list])
    delay = a + b * num_total_tokens + c * sum_num_tokens_sqr
    delay = delay * pp_factor + pp_const
    return delay


def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None, **kw):
    a, b, c = profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]["decode"]
    pp_factor = 1 / pp
    pp_const = 1 * pp  # TODO: Modulate the PP overhead
    num_total_tokens = sum(token_generated_list)
    batch_size = num_requests
    delay = a + b * num_total_tokens + c * batch_size
    delay = delay * pp_factor + pp_const
    return delay
