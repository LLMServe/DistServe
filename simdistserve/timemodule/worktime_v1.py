# Fit a model where prefill does not have an intercept, and decode does have one.
from math import isnan
from pathlib import Path

import pandas as pd


class ModelTypes:
    opt_13b = 'OPT-13B'
    opt_66b = 'OPT-66B'
    opt_175b = 'OPT-175B'

    @staticmethod
    def formalize_model_name(x):
        if x == 'opt13b': return ModelTypes.opt_13b
        if x == 'opt66b': return ModelTypes.opt_66b
        if x == 'opt175b': return ModelTypes.opt_175b
        raise ValueError(x)


def setup_time_params():
    model = pd.read_csv(Path(__file__).parent / 'worktime_v1.csv')
    prefill_params = {
        (ModelTypes.formalize_model_name(model_type), tp, pp): (a, b, c)
        for _, (model_type, pd, a, b, c, tp, pp) in model.iterrows()
        if pd == 'prefill'
    }

    decode_params = {
        (ModelTypes.formalize_model_name(model_type), tp, pp): (a, b, c)
        for _, (model_type, pd, a, b, c, tp, pp) in model.iterrows()
        if pd == 'decode'
    }

    return prefill_params, decode_params


prefill_params, decode_params = setup_time_params()


def get_prefill_time(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, **kw):
    assert prefill_len_list, 'prefill `prefill_len_list` is required'
    overhead = 0
    if model_type == ModelTypes.opt_13b:
        if (model_type, TP, pp) in prefill_params:
            a, b, c = prefill_params[(model_type, TP, pp)]
        else:
            # Fall back to TP = 1
            a, b, c = prefill_params[(model_type, 1, 1)]
            (a, b, c) = (a / TP / pp, b / TP / pp, c / TP / pp)
            overhead = 1 * TP

    elif model_type == ModelTypes.opt_66b:
        if (model_type, TP, pp) in prefill_params:
            a, b, c = prefill_params[(model_type, TP, pp)]
        else:
            # Fall back to TP = 2, PP = 1, and normalize it to TP = 1, PP = 2
            a, b, c = prefill_params[(model_type, 2, 1)]
            (a, b, c) = (a / TP / pp, b / TP / pp, c / TP / pp)
            (a, b, c) = (a * 2, b * 2, c * 2)
            overhead = 1 * TP
    elif model_type == ModelTypes.opt_175b:
        if (model_type, TP, pp) in prefill_params:
            a, b, c = prefill_params[(model_type, TP, pp)]
        else:
            # Fall back to TP = 2, PP = 4, normalize it to TP = 1 and PP = 1
            a, b, c = prefill_params[(model_type, 2, 4)]
            (a, b, c) = (a * 8, b * 8, c * 8)
            (a, b, c) = (a / TP / pp, b / TP / pp, c / TP / pp)
            overhead = 1 * TP
    else:
        raise NotImplementedError(model_type)

    delay = a * num_tokens + b * sum((i // 64) ** 2 for i in prefill_len_list) + c + overhead
    # print(f'>>>>prefill|{prefill_len_list}|{delay}|{pp}|{TP}')
    assert not isnan(delay)
    return delay


def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None, **kw):
    assert token_generated_list, 'decode `token_generated_list` is required'

    overhead = 0
    if model_type == ModelTypes.opt_13b:
        if TP == 1:
            a, b, c = 0.00058323, 0.02713492, 19.5175
        elif TP == 2:
            a, b, c = 0.0003346309566065, 0.0242961697383897, 13.604431926258682
        elif TP == 4:
            a, b, c = 0.0001987465644909, 0.0111104359855963, 9.392290369222366
        else:
            # Fall back to TP = 1
            a, b, c = 0.00058323, 0.02713492, 19.5175
            (a, b, c) = (a / TP / pp, b / TP / pp, c / TP / pp)
            overhead = 1 * TP
            pass
    elif model_type == ModelTypes.opt_66b:
        if TP == 2:
            a, b, c = 0.0009380319482417, 0.0296114928613491, 40.20930149627845
        elif TP == 4:
            a, b, c = 0.0008319271363898, 0.0075814479902322, 28.73783470990596
        else:
            a, b, c = 0.0009380319482417, 0.0296114928613491, 40.20930149627845
            (a, b, c) = (a / TP / pp, b / TP / pp, c / TP / pp)
            (a, b, c) = (a * 2, b * 2, c * 2)
            overhead = 1 * TP

    else:
        raise NotImplementedError(model_type)

    delay = a * sum(token_generated_list) + b * num_requests + c + overhead
    # print(f'>>>>decode|{token_generated_list}|{delay}|{pp}|{TP}')
    assert not isnan(delay)
    return delay / pp / TP
