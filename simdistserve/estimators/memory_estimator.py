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


max_num_tokens_data = load_profile_data()


def get_max_num_tokens(model: ModelTypes, tp: int, pp: int) -> int:
    model: str = ModelTypes.formalize_model_name(model)
    max_num_tokens = max_num_tokens_data[model][(tp, pp)]
    return max_num_tokens
