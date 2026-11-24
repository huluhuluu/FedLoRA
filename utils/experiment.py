"""Bounded experiment artifacts and atomic adapter-only checkpoints."""

import csv
import hashlib
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
import torch
from utils.helper import check_output_path, json_write

FIELDS = [
    "round",
    "step",
    "epoch",
    "samples_processed",
    "optimizer_steps",
    "learning_rate",
    "global_version",
    "selected",
    "train_loss",
    "virtual_round_seconds",
    "virtual_total_seconds",
    "physical_round_seconds",
    "uploaded_bytes",
    "downloaded_bytes",
]


def identity(config):
    import copy

    filtered = copy.deepcopy(
        {
            key: value
            for key, value in config.items()
            if key not in ("output", "execution", "evaluation", "_config_path")
        }
    )
    filtered["experiment"].pop("name", None)
    filtered["runtime"].pop("cpu_threads", None)
    if config["experiment"]["mode"] == "centralized":
        filtered["training"]["budget"].pop("value", None)
        for key in (
            "clients",
            "partition",
            "grouping",
            "selection",
            "aggregation",
            "simulation",
            "timing",
        ):
            filtered.pop(key, None)
    else:
        filtered.pop("training", None)
        filtered["simulation"].pop("rounds", None)
        if filtered["timing"]["mode"] == "off":
            filtered["timing"] = {"mode": "off"}
    return hashlib.sha256(json.dumps(filtered, sort_keys=True).encode()).hexdigest()


def load_checkpoint(path, config):
    if path is None:
        return None
    from pathlib import Path
    from utils.config import load_config

    saved = torch.load(path, map_location="cpu", weights_only=True)
    mode = config["experiment"]["mode"]
    if saved.get("kind", "federated") != mode:
        raise ValueError(f"Resume requires a {mode} training checkpoint")
    if saved["identity"] != identity(config):
        # Older checkpoints hashed inactive options as well; their resolved inputs allow migration.
        sidecar = Path(path).parent / "config.resolved.json"
        if not sidecar.is_file():
            raise ValueError(
                "Resume identity differs; keep the original config.resolved.json beside older checkpoints"
            )
        if identity(load_config(sidecar)) != identity(config):
            raise ValueError("Resume configuration changes training inputs")
    return saved


class Experiment:
    def __init__(self, config, clients, partitions, initial_state, parent=None):
        root = check_output_path(config["output"]["root"])
        name = config["experiment"]["name"]
        if not name or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in name
        ):
            raise ValueError(
                "experiment.name accepts letters, numbers, underscore and dash"
            )
        self.path = root / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + name
            + "-"
            + uuid.uuid4().hex[:8]
        )
        # Task construction may require label metadata; declarations are class-level.
        from tasks import task_class

        names = task_class(config["task"]["type"]).metric_names
        self.eval_fields = ["eval_" + name for name in names]
        self.fields = FIELDS + self.eval_fields
        self.path.mkdir(parents=True, exist_ok=False)
        json_write(self.path / "config.resolved.json", config)
        if config["experiment"]["mode"] == "federated":
            json_write(self.path / "clients.json", clients)
            import numpy as np

            np.savez_compressed(
                self.path / "partitions.npz",
                **{
                    str(cid): np.asarray(ids, dtype=np.int64)
                    for cid, ids in partitions.items()
                },
            )
        layout = initial_state.to_dict()
        layout.pop("tensors")
        json_write(self.path / "adapter_layout.json", layout)
        import importlib.metadata as md

        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        json_write(
            self.path / "environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "versions": {
                    name: md.version(name)
                    for name in ("torch", "transformers", "peft", "datasets")
                },
                "git_head": git.stdout.strip(),
                "parent_checkpoint": str(parent) if parent else None,
                "git_dirty": bool(
                    subprocess.run(
                        ["git", "status", "--porcelain"], capture_output=True, text=True
                    ).stdout.strip()
                ),
            },
        )
        with (self.path / "rounds.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writeheader()
        self.status("running")

    def status(self, value, error=None):
        json_write(self.path / "status.json", {"status": value, "error": error})

    def record(self, row, event):
        with (self.path / "rounds.csv").open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writerow(
                {key: row.get(key, "") for key in self.fields}
            )
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def checkpoint(self, value):
        temporary = self.path / "checkpoint_last.pt.tmp"
        torch.save(value, temporary)
        os.replace(temporary, self.path / "checkpoint_last.pt")

    def event(self, value):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def dispatch(self, value):
        with (self.path / "dispatch.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")

    def record_evaluation(self, value):
        path = self.path / "evaluations.csv"
        fields = [
            "model_version",
            "unit",
            "index",
            "worker_ids",
            *self.eval_fields,
            "physical_seconds",
        ]
        row = {
            "model_version": value["model_version"],
            **value["progress"],
            "worker_ids": json.dumps(value["worker_ids"]),
            **{
                "eval_" + key: item
                for key, item in value["metrics"].items()
                if key != "physical_seconds"
            },
            "physical_seconds": value["metrics"]["physical_seconds"],
        }
        exists = path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        with (self.path / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    def save_best(self, value):
        temporary = self.path / "best_adapter.pt.tmp"
        torch.save(value, temporary)
        os.replace(temporary, self.path / "best_adapter.pt")
