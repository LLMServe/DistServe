"""Download *.bin files from huggingface and convert model weights into 
SwiftTransformer's format.

Default path where facebook/opt-125m is saved: 
    ~/.cache/distserve/models--facebook--opt-125m/
"""
import filelock
import os
import torch

from distserve.config import ModelConfig
from distserve.logger import init_logger
from huggingface_hub import snapshot_download
from typing import Optional
from .converter import convert_weights

logger = init_logger(__name__)

# Constants.
# REPO_ID_SEPARATOR - this substring is not allowed in repo_ids on hf.co
# and is the canonical one we use for serialization of repo ids elsewhere.
REPO_ID_SEPARATOR = "--"

# cache - where the converted weights are saved
default_cache = os.path.join(os.path.expanduser("~"), ".cache", "distserve")
DISTSERVE_CACHE = os.getenv("DISTSERVE_CACHE", default_cache)

# MODEL_REGISTRY - Supported models
MODEL_REGISTRY = {
    "opt",
    "llama",
    "gpt2"
}

def repo_folder_name(*, repo_id: str, repo_type: str = "model") -> str:
    """Return a serialized version of a hf.co repo name and type, safe for 
    disk storage as a single non-nested folder.

    Example: models--julien-c--EsperBERTo-small
    """
    # remove all `/` occurrences to correctly convert repo to directory name
    parts = [f"{repo_type}s", *repo_id.split("/")]
    return REPO_ID_SEPARATOR.join(parts)

def get_lock(model: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else default_cache
    os.makedirs(lock_dir, exist_ok=True)
    lock_file_name = model.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock

def prepare_hf_model_weigths(
    model_name: str,
    cache_dir: Optional[str] = None
) -> str:
    """Download model weights from huggingface."""
    allow_patterns = "*.bin"
    hf_folder = snapshot_download(model_name,
                                    allow_patterns=allow_patterns,
                                    cache_dir=cache_dir)
    return os.path.join(hf_folder, allow_patterns)

def download_and_convert_weights(model_config: ModelConfig) -> str:
    """(interface) Function for downloading and converting weights.
    
    A user is allowed to pass a huggingface repo name or the local folder where
    model is located to `OfflineLLM` or `AsyncLLM` as the `model` argument. 
    This function first decides where it is. If it is local, directly return 
    the path. Otherwise, it will download and convert weights from the 
    huggingface hub the first time the repo is called on. When running the 
    same model in the future, it will return the path where the converted model 
    weights were saved. Similarly, it won't download the same model from
    huggingface twice.

    We mark whether the model already exists with an empty file called `done`.
    """
    model_name_or_path = model_config.model
    with get_lock(model_name_or_path):
        """
        We use a file lock here to prevent multiple processes on the same machine
        from downloading the same model weights from huggingface at the same time.

        Some explanation is necessary here:
        - We use file lock instead of a global lock (provided by ray or something)
          because that, whether two workers (processes) need to download the same
          weight multiple times or not, depends on whether the two workers reside
          on the same machine (i.e. use the same filesystem) or not. So filelock
          is the most suitable solution here.
          What's more, this also works under network filesystems (e.g. NFS).
        - We warp the whole codeblock inside the lock, instead of just the "downloading
          and converting" part, because otherwise the following situation may happen:
          - Worker A founds out that the model weights are not downloaded yet, so it
            decides to download and convert the weights.
          - Just before worker A begins downloading, another worker, worker B, finds
            out that the model weights are not downloaded yet, so it also decides to
            download the weights.
          If this happens, then the two workers will download the same weights twice,
          no matter whether we use a lock or not. So we should warp the entire codeblock
          (including the "check whether the weights are downloaded" part) inside the lock.
        """
        torch_dtype = {"fp16": torch.float16, "fp32": torch.float32}
        try:
            dtype = torch_dtype[model_config.dtype]
        except KeyError:
            raise ValueError(
                f"Unknown dtype {dtype}, expected one of {torch_dtype.keys()}")
        model = model_config.hf_config.model_type
        assert model in MODEL_REGISTRY, \
            f"Unknown model {model}, expected one of {MODEL_REGISTRY}"
        
        # if the user provides a local path
        is_local = os.path.isdir(model_name_or_path)
        
        # if the model weights have already been downloaded and converted before
        cache_dir = DISTSERVE_CACHE
        storage_folder = \
            os.path.join(cache_dir, 
                        repo_folder_name(repo_id=model_name_or_path)) + '/'
        done_file = os.path.join(storage_folder, "done")
        if os.path.exists(done_file):
            logger.info(f"Find cached model weights in {storage_folder}.")    
            return storage_folder
        
        # download and convert model weights
        hf_files = ""
        if is_local:
            hf_files = model_name_or_path
        else:
            hf_files = prepare_hf_model_weigths(model_name_or_path)
        convert_weights(hf_files, storage_folder, dtype, model)
        file = open(done_file, 'w')
        file.close()
        return storage_folder
