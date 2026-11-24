"""Build a compute timing profile from recorded training workloads."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    from utils.helper import check_output_path
    from utils.helper import json_write

    path = check_output_path(args.run.resolve())
    config = json.loads((path / "config.resolved.json").read_text())
    if config["experiment"].get("mode", "federated") != "federated":
        raise ValueError(
            "Client timing profiles apply to federated runs; centralized global-step time is recorded directly"
        )
    observations = {}
    for line in (path / "events.jsonl").read_text().splitlines():
        for client in json.loads(line).get("clients", []):
            workload = client["workload"]
            steps = workload.get("batches") or workload["optimizer_steps"]
            if not steps:
                continue
            key = (
                config["model"]["path"],
                config.get("task", {}).get(
                    "type", config["model"].get("task", "classification")
                ),
                workload["precision"],
                workload["rank"],
                workload["batch_size"],
                workload["seq_len"],
                workload["hardware_profile"],
                workload.get("compute_signature", ""),
            )
            observations.setdefault(key, []).append(
                workload["physical_seconds"] / steps
            )
    fields = (
        "model",
        "task",
        "precision",
        "rank",
        "batch_size",
        "seq_len",
        "hardware_profile",
        "compute_signature",
    )
    entries = [
        {
            **dict(zip(fields, key)),
            "step_seconds": statistics.median(values),
            "observations": len(values),
        }
        for key, values in observations.items()
    ]
    json_write(
        path / "timing_profile.json",
        {"schema_version": 1, "source": "measured_training", "entries": entries},
    )
    print(path / "timing_profile.json")


if __name__ == "__main__":
    main()
