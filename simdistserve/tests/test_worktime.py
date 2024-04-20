from simdistserve.constants import ModelTypes
from simdistserve.timemodule.worktime import get_prefill_time, get_decode_time


def test_get_prefill_time():
    print("bs,time")
    for bs in [1, 2, 4, 8, 16, 32]:
        batch = [64] * bs
        a = get_prefill_time(
            num_tokens=sum(batch),
            pp=1, TP=1, bs=bs, decode_bs=0,
            model_type=ModelTypes.opt_13b,
            prefill_len_list=batch,
        )
        print(f"{bs},{a:.2f},")
    pass


def test_get_decode_time():
    print("bs,time")
    for bs in [1, 2, 4, 8, 16, 32]:
        batch = [64] * bs
        a = get_decode_time(
            bs, pp=1, TP=1,
            token_generated_list=batch,
        )
        print(f"{bs},{a:.2f},")
    pass


if __name__ == '__main__':
    print("== Prefill Time ==")
    test_get_prefill_time()
    print("== Decode Time ==")
    test_get_decode_time()
