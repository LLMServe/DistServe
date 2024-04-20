# Fit a model where prefill does not have an intercept, and decode does have one.

from simdistserve.constants import ModelTypes

opt_13b_params = {(1, 1): (
    [0.09482354604158061, 4.03088651711854e-06, 18.891404031181914],
    [0.0006109319663589057, 0.06818472528436295, 19.878099045083218]), (1, 8): (
    [-4558.887156658196, 71.23285024757712, 2.2555561121921532],
    [71.22214629752114, 0, 2.5190309970412765]), (4, 1): (
    [0.0324359612452919, 1.4551978520879437e-06, 12.64555955944732],
    [0.0002088168996943625, -0.03482556407211749, 9.22092145851236]), (1, 2): (
    [0.04792105116000226, 1.88847757163527e-06, 11.161935389560918],
    [0.0003322802214276166, -0.051424246153820574, 10.341547636031812]), (2, 1): (
    [0.06373414217499217, 2.0426077577567835e-06, 17.73259930652419],
    [0.00036709471825591145, 0.018816958156899988, 17.350085537432527]), (2, 2): (
    [0.030488963388914707, 1.0579946461106713e-06, 11.583810492933228],
    [0.00016739034217305715, 0.002035068323032907, 7.9132009388562725])}


def get_prefill_time(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, **kw):
    assert prefill_len_list, 'prefill `prefill_len_list` is required'
    if model_type == ModelTypes.opt_13b:
        if (TP, pp) in opt_13b_params:
            a, b, c = opt_13b_params[(TP, pp)][0]
        else:
            raise NotImplementedError(f"Model type {model_type} with {TP = } and {pp = } not implemented")
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    x = sum(prefill_len_list)
    x2 = sum(i ** 2 for i in prefill_len_list)
    delay = a * x + b * x2 + c
    return delay


def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None, **kw):
    assert token_generated_list, 'decode `token_generated_list` is required'
    if model_type == ModelTypes.opt_13b:
        if (TP, pp) in opt_13b_params:
            a, b, c = opt_13b_params[(TP, pp)][1]
        else:
            raise NotImplementedError(f"Model type {model_type} with {TP = } and {pp = } not implemented")
    x = sum(token_generated_list)
    bs = len(token_generated_list)
    delay = a * x + b * bs + c
    return delay
