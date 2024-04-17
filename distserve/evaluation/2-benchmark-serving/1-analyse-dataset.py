import os, sys
import argparse

import numpy as np
import histoprint

from structs import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Path to the dataset (produced by 0-prepare-dataset.py)")
    args = parser.parse_args()
    
    dataset_path = args.dataset
    dataset = Dataset.load(dataset_path)
    
    prompt_lens = np.array([req.prompt_len for req in dataset.reqs])
    output_lens = np.array([req.output_len for req in dataset.reqs])
    
    # Draw a histogram of prompt_len
    print("Distribution of prompt_len:")
    histoprint.print_hist(
        np.histogram(
            prompt_lens,
            bins = 40,
            range = (0, 2047)
        ),
        bg_colors="c",
    )
    print()
 
    print("Distribution of output_len:")
    histoprint.print_hist(
        np.histogram(
            output_lens,
            bins = 40,
            range = (0, 2047)
        ),
        bg_colors="c",
    )
    print()
    histoprint.print_hist(
        np.histogram(
            output_lens,
            bins = 40,
            range = (0, 256)
        ),
        bg_colors="c",
    )
    print()
 
    print("Distribution of prompt_len+output_len:")
    histoprint.print_hist(
        np.histogram(
            prompt_lens+output_lens,
            bins = 40,
            range = (0, 2047)
        ),
        bg_colors="c",
    )
    
    print(f"Prompt len mean: {np.mean(prompt_lens)}")
    print(f"Output len mean: {np.mean(output_lens)}")
    