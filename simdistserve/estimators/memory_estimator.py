from pathlib import Path
import pandas as pd

from simdistserve.constants import ModelTypes


def load_profile_data():
    profile_data_path = Path(__file__).parent / "profile_data" / "max_num_tokens.csv"
    with open(profile_data_path) as f:
        _profile_data = pd.read_csv(f)

    result = {}
    for _, row in _profile_data.iterrows():
        model = row['model']
        tp = row['tp']
        pp = row['pp']
        if model not in result:
            result[model] = {}
        result[model][(tp, pp)] = row['max_num_tokens']

    return result


# { model: { (tp, pp): max_num_tokens } }
max_num_tokens_data = load_profile_data()


def is_model_runnable(model: ModelTypes, tp: int, pp: int) -> bool:
    if model not in max_num_tokens_data:
        return False
    if (tp, pp) not in max_num_tokens_data[model]:
        return False
    return True


def get_max_num_tokens(model: ModelTypes, tp: int, pp: int) -> int:
    model: str = ModelTypes.formalize_model_name(model)
    max_num_tokens = max_num_tokens_data[model][(tp, pp)]
    return max_num_tokens


model_hyperparams = {
    # model: (layers, heads)
    "facebook/opt-13b": (40, 40),
    "facebook/opt-66b": (64, 72),
    "facebook/opt-175b": (96, 96),
}


def get_model_possible_pp(model: str):
    total_num_layers = model_hyperparams[model][0]
    possible_pp = []
    for pp in range(1, 1 + total_num_layers):
        if total_num_layers % pp == 0:
            possible_pp.append(pp)
    return possible_pp


def get_model_possible_tp(model: str):
    total_num_attention_heads = model_hyperparams[model][1]
    possible_tp = []
    for tp in range(1, 1 + total_num_attention_heads):
        if total_num_attention_heads % tp == 0:
            possible_tp.append(tp)
    return possible_tp
