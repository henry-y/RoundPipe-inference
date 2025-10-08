import os
import json
from typing import *

import threading
import torch
from safetensors import safe_open


class HfShardLoader:
    """Lazy loader for HuggingFace sharded safetensors weights.

    Usage:
        loader = HfShardLoader(model_dir)
        t = loader.load_to_cpu(param_name, dtype, shape)  # returns pinned CPU tensor
    """

    def __init__(self, model_dir: str):
        index_json = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_json):
            raise FileNotFoundError(f"HF safetensors index not found: {index_json}")
        with open(index_json, "r") as f:
            idx = json.load(f)
        self.name_to_file: Dict[str, str] = idx.get("weight_map", {})
        if not self.name_to_file:
            raise ValueError("Invalid safetensors index.json: missing weight_map")
        self.model_dir = model_dir
        self._lock = threading.Lock()

    def load_to_cpu(self, name: str, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
        shard = self.name_to_file.get(name)
        if shard is None:
            raise KeyError(f"Parameter/buffer not found in index: {name}")
        path = os.path.join(self.model_dir, shard)
        with self._lock:
            with safe_open(path, framework="pt", device="cpu") as f:
                t = f.get_tensor(name)
        if t.dtype != dtype or tuple(t.shape) != tuple(shape):
            raise RuntimeError(f"dtype/shape mismatch for {name}: {t.dtype}/{tuple(t.shape)} vs {dtype}/{tuple(shape)}")
        try:
            return t.pin_memory()
        except RuntimeError:
            return t.contiguous().pin_memory()


