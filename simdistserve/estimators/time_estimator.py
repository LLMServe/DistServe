# Fit a model where prefill does not have an intercept, and decode does have one.
import json
from pathlib import Path

from simdistserve.constants import ModelTypes


def load_distserve_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "profiler-a100-80g.distserve.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


def load_vllm_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "profiler-a100-80g.vllm.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


distserve_profile_data = load_distserve_profile_data()
vllm_profile_data = load_vllm_profile_data()


def get_prefill_time(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, engine_type="distserve", **kw):
    if engine_type == "distserve":
        params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        a, b, c = params["prefill"]
    else:
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        a, b, c = params["prefill"]

    f = 1
    a, b, c = (a * f, b * f, c * f)
    pp_factor = 1 / pp
    pp_const = 1 * pp  # TODO: Modulate the PP overhead
    num_total_tokens = sum(prefill_len_list)
    sum_num_tokens_sqr = sum([x ** 2 for x in prefill_len_list])
    delay = a + b * num_total_tokens + c * sum_num_tokens_sqr
    delay = delay * pp_factor + pp_const
    return delay


def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None,
                    engine_type="distserve", **kw):
    batch_size = num_requests
    if engine_type == "distserve":
        params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params[
            "decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            a, b, c = params["decoding_smallbs"]
        else:
            a, b, c = params["decoding_largebs"]
    else:
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params[
            "decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            a, b, c = params["decoding_smallbs"]
        else:
            a, b, c = params["decoding_largebs"]
        pass
    f = 1
    pp_factor = 1 / pp
    # pp_const = 1 * pp  # TODO: Modulate the PP overhead
    pp_const = 0
    num_total_tokens = sum(token_generated_list)

    delay = a + b * num_total_tokens + c * batch_size
    delay = delay * pp_factor + pp_const
    delay *= f
    return delay
