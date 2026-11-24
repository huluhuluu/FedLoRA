"""Sequential named comparison cases with per-case parameter grids."""

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid


def variants(base, spec):
    from utils.config import (
        merge,
        set_value,
        validate,
        validate_components,
        resolve_paths,
    )

    base_dir = (
        Path(base["_config_path"]).resolve().parent
        if base.get("_config_path")
        else Path.cwd()
    )
    for case in spec.get("cases", [{"name": "grid"}]):
        config = merge(base, case.get("overrides", {}))
        if config["experiment"]["mode"] != base["experiment"]["mode"]:
            if "aggregation" not in case.get("overrides", {}):
                federated = config["experiment"]["mode"] == "federated"
                config["aggregation"] = {
                    "mode": "flat" if federated else "none",
                    "method": None,
                    "params": {"weighting": "samples"} if federated else {},
                }
            config["evaluation"]["unit"] = (
                case.get("overrides", {}).get("evaluation", {}).get("unit")
            )
        if "task" in case.get("overrides", {}).get(
            "model", {}
        ) and "task" not in case.get("overrides", {}):
            config["task"]["type"] = case["overrides"]["model"]["task"]
        grid = {**spec.get("grid", {}), **case.get("grid", {})}
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            resolved = json.loads(json.dumps(config))
            parameters = dict(zip(keys, values))
            for key, value in parameters.items():
                set_value(resolved, key, value)
            resolve_paths(resolved, base_dir)
            validate(resolved)
            validate_components(resolved, construct=True)
            yield case["name"], parameters, resolved


def launch(config, path):
    from utils.helper import PROJECT_ROOT

    command = [sys.executable, str(PROJECT_ROOT / "main.py"), "--config", str(path)]
    env = os.environ.copy()
    if config["execution"]["backend"] == "distributed":
        from execution import devices_for

        if config["execution"].get("resources_file"):
            manifest = json.loads(
                Path(config["execution"]["resources_file"]).read_text()
            )
            if len(manifest["nodes"]) != 1:
                raise ValueError(
                    "For multi-node suites launch generated case configs with tools.launch on every node"
                )
        devices = devices_for(config)
        if all(device.startswith("cuda:") for device in devices):
            from execution import cuda_visible

            env["CUDA_VISIBLE_DEVICES"] = cuda_visible(devices, env)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--node-rank=0",
            f"--nproc-per-node={len(devices)}",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            *command[1:],
        ]
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write case configs without starting training",
    )
    args = parser.parse_args()
    from utils.config import load_config, resolve, validate
    from utils.helper import check_output_path, configure_runtime
    from utils.helper import json_write

    configure_runtime(2)
    spec = json.loads(args.suite.read_text())
    base_path = Path(resolve(spec["base"], args.suite.resolve().parent))
    base = load_config(base_path)
    root = check_output_path(
        resolve(
            spec.get("output_root", base["output"]["root"]), args.suite.resolve().parent
        )
    )
    base["output"]["root"] = str(root)
    validate(base)
    planned = list(variants(base, spec))
    directory = root / (
        "suite-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    inputs = directory / "configs"
    inputs.mkdir(parents=True)
    json_write(directory / "suite.json", spec)
    summary = []
    for index, (case, parameters, config) in enumerate(planned):
        config["output"]["root"] = str(directory / "runs")
        config["experiment"]["name"] = f"suite-{index:04d}"
        if (
            config["execution"]["backend"] == "distributed"
            and config["evaluation"]["devices"]
        ):
            from execution import devices_for

            devices = devices_for(config)
            config["evaluation"]["worker_ids"] = [
                devices.index(device) for device in config["evaluation"]["devices"]
            ]
            config["evaluation"]["devices"] = []
        path = inputs / f"{index:04d}.json"
        json_write(path, config)
        if args.prepare_only:
            continue
        launch(config, path)
        run = next((directory / "runs").glob(f"*-suite-{index:04d}-*"))
        with (run / "rounds.csv").open() as stream:
            row = list(csv.DictReader(stream))[-1]
        if (run / "evaluations.csv").exists():
            with (run / "evaluations.csv").open() as stream:
                evaluation = list(csv.DictReader(stream))[-1]
            row.update(
                {
                    key: value
                    for key, value in evaluation.items()
                    if key.startswith("eval_")
                }
            )
            row["eval_model_version"] = evaluation["model_version"]
        summary.append(
            {
                "case": case,
                "mode": config["experiment"]["mode"],
                "run": str(run),
                **{key: json.dumps(value) for key, value in parameters.items()},
                **row,
            }
        )
        with (directory / "summary.csv").open("w", newline="") as stream:
            fields = list(dict.fromkeys(key for value in summary for key in value))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
    print(directory)


if __name__ == "__main__":
    main()
