"""Deterministic seeds and bounded runtime caches."""

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data/HF_MODELS")


def stable_seed(seed, *parts):
    payload = ":".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def check_output_path(path):
    resolved = Path(path).expanduser().resolve()
    if resolved == ASSET_ROOT or ASSET_ROOT in resolved.parents:
        raise ValueError("Outputs and caches cannot be placed under /data/HF_MODELS")
    return resolved


def configure_runtime(cpu_threads=2):
    cache = PROJECT_ROOT / ".cache"
    paths = {
        "HF_HOME": cache / "huggingface",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "TORCH_HOME": cache / "torch",
        "MPLCONFIGDIR": cache / "matplotlib",
        "TRITON_CACHE_DIR": cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
        "TMPDIR": cache / "tmp",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    return cache


def seed_training(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def json_write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
