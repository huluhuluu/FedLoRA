"""Small public contracts for custom algorithms."""

from dataclasses import dataclass, field
from models.adapter_state import AdapterState


@dataclass
class Workload:
    samples_processed: int = 0
    unique_samples: int = 0
    tokens_processed: int = 0
    batches: int = 0
    optimizer_steps: int = 0
    upload_bytes: int = 0
    download_bytes: int = 0
    physical_seconds: float = 0.0
    operation: str = "train"
    batch_size: int = 1
    seq_len: int = 128
    rank: int = 8
    precision: str = "fp32"
    device: str = "cpu"
    hardware_profile: str = "cpu"
    compute_signature: str = ""


@dataclass
class ClientUpdate:
    client_id: int
    round_id: int
    global_version: int
    state: AdapterState
    num_samples: int
    metrics: dict = field(default_factory=dict)
    workload: Workload = field(default_factory=Workload)
    optimizer_state: dict | None = None
    worker_id: int = 0
    scaler_state: dict | None = None


@dataclass
class AggregationResult:
    state: AdapterState
    metrics: dict = field(default_factory=dict)
    workload: Workload = field(default_factory=lambda: Workload(operation="aggregate"))


@dataclass
class Context:
    round_id: int
    global_version: int
    seed: int
    rng: object = None


def context(config, operation, round_id=0, version=0):
    import numpy as np
    from utils.helper import stable_seed

    seed = stable_seed(config["experiment"]["seed"], operation, round_id)
    return Context(round_id, version, seed, np.random.default_rng(seed))
