"""Resolve execution resources and create one worker per device."""

import json
import os
from pathlib import Path


def devices_for(config):
    spec = config["execution"]
    if spec["devices"]:
        devices = spec["devices"]
    else:
        manifest = json.loads(Path(spec["resources_file"]).read_text())
        nodes = [
            node for node in manifest["nodes"] if node["name"] == spec["local_node"]
        ]
        if len(nodes) != 1:
            raise ValueError(
                "execution.local_node must identify a resource manifest node"
            )
        devices = nodes[0]["devices"]
    if not devices or not (
        all(d == "cpu" for d in devices) or all(d.startswith("cuda:") for d in devices)
    ):
        raise ValueError("A node needs CPU workers or CUDA devices")
    return devices


def cuda_visible(devices, env):
    mapping = env.get("CUDA_VISIBLE_DEVICES")
    mapping = mapping.split(",") if mapping else None
    return ",".join(
        mapping[int(d.split(":")[1])] if mapping else d.split(":")[1] for d in devices
    )


def rank_device(config):
    devices = devices_for(config)
    return "cpu" if devices[0] == "cpu" else "cuda:" + os.environ["LOCAL_RANK"]


def create_executor(config, bundle=None, initialize=True):
    if config["execution"]["backend"] == "distributed":
        from execution.distributed import DistributedExecutor

        return DistributedExecutor(config, bundle, initialize)
    from execution.local import SerialExecutor, LocalExecutor

    executor = (
        LocalExecutor if config["execution"]["backend"] == "local" else SerialExecutor
    )
    return executor(config, bundle, initialize)
