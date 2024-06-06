"""
# Introduction

This program reads data (batch size, input len, prefill time usage, decoding
time usage) from a sqlite database, and fits a model to predict the prefill/decoding
time usage.

# Methodology

For the prefill stage, we assume the time usage to be:

A + B*#total_tokens + C*\sum_{i=1}^{batch_size} input_len_i^2

Where #total_tokens = batch_size * input_len

While for the decoding stage, we assume the time usage to be:

A + B*#previous_tokens + C*batch_size

Where #previous_tokens = batch_size * input_len

We use the least squares method ("最小二乘法" in Chinese) to fit the model, with
the goal of minimizing \sum relative_error^2.
"""
import math
import dataclasses
from typing import Callable
import json

import numpy as np
import csv
import sqlite3
import argparse

@dataclasses.dataclass
class DataPoint:
    model: str
    tp_world_size: int
    
    batch_size: int
    input_len: int
    
    prefill_time: float
    decoding_time: float

def read_all_data_points(db_path: str) -> list[DataPoint]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT tag, tp_world_size, batch_size, input_len, avg_prefill_time_usage, avg_decoding_time_usage FROM records")
    return [
        DataPoint(
            model = row[0],
            tp_world_size = row[1],
            batch_size = row[2],
            input_len = row[3],
            prefill_time = row[4],
            decoding_time = row[5]
        )
        for row in cur.fetchall()
    ]

def fit_one_abc(
    data_points: list[DataPoint],
    b_coef_getter: Callable[[DataPoint], float],
    c_coef_getter: Callable[[DataPoint], float],
    t_coef_getter: Callable[[DataPoint], float],
    weight_getter: Callable[[DataPoint], float]
) -> tuple[float, float, float]:
    a_matrix = []
    b_vec = []
    for dp in data_points:
        b = b_coef_getter(dp)
        c = c_coef_getter(dp)
        t = t_coef_getter(dp)
        weight = weight_getter(dp)
        a_matrix.append([
            1/t*weight, b/t*weight, c/t*weight
        ])
        b_vec.append(
            weight
        )
    a_matrix = np.array(a_matrix)
    b_vec = np.array(b_vec)
    answer, _, _, _ = np.linalg.lstsq(a_matrix, b_vec, rcond=None)
    
    print(f"{'bs':>3s}  {'ilen':>6s}  {'actual':>9s}  {'pred':>9s}  {'rel_err':>6s}")
    rel_errs = []
    for dp in data_points:
        b = b_coef_getter(dp)
        c = c_coef_getter(dp)
        t = t_coef_getter(dp)
        pred_time_usage = answer[0] + answer[1]*b + answer[2]*c
        cur_rel_err = (pred_time_usage - t)/t
        rel_errs.append(cur_rel_err)
        print(f"{dp.batch_size:3d}  {dp.input_len:6d}  {t:9.2f}  {pred_time_usage:9.2f}  {cur_rel_err*100:6.2f}%")
    
    rel_errs = np.array(rel_errs)
    print(f"Max relative error: {np.max(np.abs(rel_errs))*100:.2f}%")
    print(f"Avg relative error: {np.mean(np.abs(rel_errs))*100:.2f}%")
    print(f"Avg sqrt(relerr^2): {np.sqrt(np.mean(rel_errs**2))*100:.2f}%")
    
    return answer

def main(args: argparse.Namespace):
    print(args)
    input_path = args.input
    output_path = args.output
    
    data_points = read_all_data_points(input_path)
    
    data_points.sort(key=lambda dp: (dp.model, dp.tp_world_size, dp.batch_size, dp.input_len))
    
    models_and_tp_sizes = []
    for dp in data_points:
        if (dp.model, dp.tp_world_size) not in models_and_tp_sizes:
            models_and_tp_sizes.append((dp.model, dp.tp_world_size))
    
    DECODING_LARGE_SMALL_BS_THRESHOLD = 96-1
    
    result = {}
    for (model, tp_world_size) in models_and_tp_sizes:
        print(f"Fitting model {model} with tp_world_size {tp_world_size} (Prefill stage)")
        cur_data_points = [
            dp
            for dp in data_points
            if dp.model == model and dp.tp_world_size == tp_world_size
        ]
        prefill_abc = fit_one_abc(
            cur_data_points,
            lambda dp: dp.batch_size*dp.input_len,
            lambda dp: dp.batch_size*dp.input_len**2,
            lambda dp: dp.prefill_time,
            lambda dp: 1
        )
        print(prefill_abc)
        
        print(f"Fitting model {model} with tp_world_size {tp_world_size} (Decoding stage, small batch size)")
        cur_data_points = [
            dp
            for dp in data_points
            if dp.model == model and dp.tp_world_size == tp_world_size
            if dp.batch_size <= DECODING_LARGE_SMALL_BS_THRESHOLD
        ]
        decoding_smallbs_abc = fit_one_abc(
            cur_data_points,
            lambda dp: dp.batch_size*dp.input_len,
            lambda dp: dp.batch_size,
            lambda dp: dp.decoding_time,
            lambda dp: 1
        )
        print(decoding_smallbs_abc)
        
        print(f"Fitting model {model} with tp_world_size {tp_world_size} (Decoding stage, large batch size)")
        cur_data_points = [
            dp
            for dp in data_points
            if dp.model == model and dp.tp_world_size == tp_world_size
            if dp.batch_size > DECODING_LARGE_SMALL_BS_THRESHOLD
        ]
        decoding_largebs_abc = fit_one_abc(
            cur_data_points,
            lambda dp: dp.batch_size*dp.input_len,
            lambda dp: dp.batch_size,
            lambda dp: dp.decoding_time,
            lambda dp: 1
        )
        print(decoding_largebs_abc)
        
        if model not in result:
            result[model] = {}
        result[model][tp_world_size] = {
            "decoding_large_small_bs_threshold": DECODING_LARGE_SMALL_BS_THRESHOLD,
            "prefill": prefill_abc.tolist(),
            "decoding_smallbs": decoding_smallbs_abc.tolist(),
            "decoding_largebs": decoding_largebs_abc.tolist()
        }
    
    with open(output_path, "w") as f:
        f.write(json.dumps(result, indent=4))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input sqlite database")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output json file")
    args = parser.parse_args()
    main(args)