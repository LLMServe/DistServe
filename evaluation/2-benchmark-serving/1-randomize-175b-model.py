import os, sys
import tqdm
import safetensors
import torch
from safetensors.torch import save_file

MODEL_DIR = "/dev/shm/forged-175b-model"

if __name__ == "__main__":
    st_files = []
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".safetensors"):
                st_files.append(file)
                
    for file in tqdm.tqdm(st_files):
        print(file)
        tensors: dict[str, torch.Tensor] = {}
        with safetensors.safe_open(os.path.join(MODEL_DIR, file), framework="pt", device="cuda:0") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        for key in tensors:
            tensors[key].normal_(mean=0.0, std=0.001)
        save_file(tensors, os.path.join(MODEL_DIR, file))
        