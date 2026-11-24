"""Launch this node's ranks, including clusters with unequal device counts."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--resume")
    args = parser.parse_args()
    from utils.helper import PROJECT_ROOT, configure_runtime

    configure_runtime(2)
    manifest = json.loads(Path(args.resources).read_text())
    nodes = manifest["nodes"]
    if len({node["name"] for node in nodes}) != len(nodes):
        raise ValueError("Resource node names must be unique")
    offset = 0
    local = None
    for node in nodes:
        if not node["devices"]:
            raise ValueError("Each node needs a device")
        if node["name"] == args.node:
            local = node
            rank_offset = offset
        offset += len(node["devices"])
    if local is None:
        raise ValueError(f"Unknown node: {args.node}")
    devices = local["devices"]
    cpu = all(device == "cpu" for device in devices)
    if not cpu and (
        any(not device.startswith("cuda:") for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("A node needs distinct CUDA devices or CPU workers")
    env = os.environ.copy()
    if local.get("interface"):
        env["GLOO_SOCKET_IFNAME"] = local["interface"]
    if not cpu:
        from execution import cuda_visible

        env["CUDA_VISIBLE_DEVICES"] = cuda_visible(devices, env)
    env.update(
        MASTER_ADDR=str(manifest["master_addr"]),
        MASTER_PORT=str(manifest["master_port"]),
        WORLD_SIZE=str(offset),
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--set",
        "execution.backend=distributed",
        "--set",
        "execution.devices="
        + json.dumps(
            ["cpu"] * len(devices)
            if cpu
            else [f"cuda:{i}" for i in range(len(devices))]
        ),
    ]
    for override in args.set:
        command.extend(["--set", override])
    if args.resume:
        command.extend(["--resume", str(Path(args.resume).resolve())])
    processes = []
    try:
        for local_rank in range(len(devices)):
            child_env = {
                **env,
                "RANK": str(rank_offset + local_rank),
                "LOCAL_RANK": str(local_rank),
                "LOCAL_WORLD_SIZE": str(len(devices)),
            }
            processes.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=child_env))
        while any(process.poll() is None for process in processes):
            failed = next(
                (
                    process.returncode
                    for process in processes
                    if process.poll() not in (None, 0)
                ),
                None,
            )
            if failed is not None:
                raise RuntimeError(f"A local rank failed with exit code {failed}")
            time.sleep(0.2)
        if any(process.returncode for process in processes):
            raise RuntimeError("A local rank failed")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
