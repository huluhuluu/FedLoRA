"""JSON inputs normalize once; validation follows the active training mode."""

import copy
import json
import math
from pathlib import Path
from utils.helper import PROJECT_ROOT, check_output_path

DEFAULT = {
    "schema_version": 1,
    "experiment": {"name": "fedlora", "seed": 42, "mode": "federated"},
    "task": {},
    "training": {
        "batch_size": 8,
        "micro_batch_size": 2,
        "budget": {"unit": "steps", "value": 100},
        "optimizer": {"name": "adamw", "lr": 0.0002},
        "max_grad_norm": 1.0,
        "scheduler": {"name": "constant", "warmup_steps": 0},
    },
    "evaluation": {
        "mode": "sync",
        "every": None,
        "unit": None,
        "at_end": True,
        "devices": [],
        "worker_ids": [],
        "batch_size": 4,
        "pending_policy": "latest",
        "save_best": False,
        "metric": "loss",
        "direction": "min",
    },
    "runtime": {"precision": "fp32", "cpu_threads": 2},
    "model": {
        "source": "local",
        "path": "",
        "head_policy": "global_trainable",
        "adapter": {
            "target_modules": None,
            "rank": 8,
            "alpha": 16,
        },
    },
    "data": {
        "source": "local",
        "path": "",
        "loader": "huggingface",
        "task": "agnews",
        "train_split": "train",
        "eval_split": "test",
        "seq_len": 128,
        "max_train_samples": None,
        "max_eval_samples": None,
    },
    "clients": {
        "count": 2,
        "defaults": {
            "adapter": {"rank": 8, "alpha": 16},
            "train": {
                "batch_size": 8,
                "budget": {"unit": "epochs", "value": 1},
                "optimizer": {
                    "name": "sgd",
                    "lr": 0.01,
                    "momentum": 0.5,
                    "state_policy": "reset",
                },
            },
            "system": {
                "hardware_profile": "reference",
                "time_multiplier": 1.0,
                "upload_mbps": 100,
                "download_mbps": 100,
                "latency_ms": 0,
            },
        },
        "profiles": {},
        "assignments": [],
        "overrides": [],
    },
    "partition": {"method": "iid", "params": {}},
    "initialization": {"method": "kaiming", "params": {}},
    "grouping": {"method": "none", "params": {}, "update_every": 0},
    "selection": {"method": "uniform", "params": {"clients_per_round": 2}},
    "aggregation": {
        "mode": "flat",
        "method": None,
        "params": {"weighting": "samples"},
    },
    "simulation": {"rounds": 1, "on_empty_round": "skip"},
    "execution": {
        "backend": "serial",
        "resources_file": None,
        "local_node": None,
        "devices": [],
        "placement": "balanced_count",
        "timeout_seconds": None,
    },
    "timing": {
        "mode": "off",
        "assumed_step_seconds": 0.1,
        "assumed_aggregation_seconds": 0.0,
        "profile_file": None,
    },
    "plugins": [],
    "output": {
        "root": str(PROJECT_ROOT / "experiments/runs"),
        "checkpoint_every": 10,
    },
}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result.get(key, {}), value)
            if isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def set_value(config, dotted, value):
    parts = dotted.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    if dotted == "model.task":
        config.setdefault("task", {})["type"] = value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path, base):
    path = Path(path).expanduser()
    return (
        str((base / path).resolve()) if not path.is_absolute() else str(path.resolve())
    )


def load_config(path=None, overrides=()):
    base = Path(path).resolve().parent if path else PROJECT_ROOT
    raw = read_json(path) if path else {}
    if "file" in raw.get("clients", {}):
        raw["clients"] = read_json(resolve(raw["clients"]["file"], base))
    config = merge(DEFAULT, raw)
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("--set expects key=JSON-value")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        set_value(config, key, parsed)
    resolve_paths(config, base)
    config["_config_path"] = str(Path(path).resolve()) if path else None
    validate(config)
    return config


def resolve_paths(config, base):
    for section in ("model", "data"):
        if config[section]["source"] == "local" and config[section]["path"]:
            config[section]["path"] = resolve(config[section]["path"], base)
    config["output"]["root"] = str(
        check_output_path(resolve(config["output"]["root"], base))
    )
    for section, field in (("execution", "resources_file"), ("timing", "profile_file")):
        if config[section].get(field):
            config[section][field] = resolve(config[section][field], base)


def expand_clients(spec):
    clients = {cid: copy.deepcopy(spec["defaults"]) for cid in range(spec["count"])}
    assigned = set()
    for item in spec.get("assignments", []):
        profile = spec.get("profiles", {})[item["profile"]]
        for cid in item["client_ids"]:
            if cid not in clients or cid in assigned:
                raise ValueError(f"Invalid or duplicate client assignment: {cid}")
            assigned.add(cid)
            clients[cid] = merge(clients[cid], profile)
            clients[cid]["profile"] = item["profile"]
    for item in spec.get("overrides", []):
        for cid in item["client_ids"]:
            if cid not in clients:
                raise ValueError(f"Unknown client: {cid}")
            clients[cid] = merge(
                clients[cid], {k: v for k, v in item.items() if k != "client_ids"}
            )
    return clients


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def normalize(config):
    mode = config["experiment"]["mode"]
    if mode not in ("federated", "centralized"):
        raise ValueError("experiment.mode must be federated or centralized")
    legacy_task = config["model"].pop("task", None)
    config["task"]["type"] = (
        config["task"].get("type") or legacy_task or "classification"
    )
    # Old single-choice switches remain accepted at the input boundary only.
    fixed = [
        ("runtime", "compile", False),
        ("model.adapter", "backend", "peft_lora"),
        ("initialization", "scope", "global"),
        (
            "execution",
            "client_training",
            {"backend": "single_gpu", "gpus_per_client": 1},
        ),
        ("evaluation", "placement", "shared_pool"),
        ("evaluation", "max_pending", 1),
        ("simulation", "protocol", "sync"),
        ("timing", "network", "independent_links"),
        ("timing", "include_evaluation", False),
    ]
    for section, key, expected in fixed:
        cursor = config
        for part in section.split("."):
            cursor = cursor[part]
        value = cursor.pop(key, expected)
        active = not (
            mode == "centralized" and section in ("simulation", "timing")
        ) and not (section == "evaluation" and config["evaluation"]["mode"] == "off")
        if active and value != expected:
            raise ValueError(f"{section}.{key} currently supports only {expected!r}")
    execution = config["execution"]
    legacy_device = execution.pop("device", None)
    if legacy_device is not None and (
        execution["backend"] == "serial" or not execution["devices"]
    ):
        execution["devices"] = [legacy_device]
    execution["devices"] = [
        "cuda:0" if d == "cuda" else d for d in execution["devices"]
    ]
    if not execution["devices"] and not execution.get("resources_file"):
        execution["devices"] = ["cpu"]
    aggregation = config["aggregation"]
    if mode == "centralized":
        if aggregation.get("method") is not None:
            raise ValueError(
                "Centralized training uses gradient updates; aggregation.method must be null"
            )
        aggregation.update(mode="none", params={})
    elif aggregation.get("method") is None:
        aggregation["method"] = "fedavg"
    evaluation = config["evaluation"]
    trigger = evaluation.pop("trigger", None)
    if trigger:
        evaluation["every"] = trigger.get("every", evaluation["every"])
        evaluation["unit"] = trigger.get("unit", evaluation["unit"])
    legacy_every = config["output"].pop("eval_every", 1)
    if evaluation["every"] is None:
        evaluation["every"] = legacy_every
    if evaluation["unit"] is None:
        evaluation["unit"] = "step" if mode == "centralized" else "round"


def validate(config):
    normalize(config)
    if config["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    if not isinstance(config["task"]["type"], str) or not config["task"]["type"]:
        raise ValueError("task.type must be a task registration name")
    if config["runtime"]["precision"] not in ("fp32", "bf16", "fp16"):
        raise ValueError("Unknown runtime.precision")
    integer(config["runtime"]["cpu_threads"], "runtime.cpu_threads")
    integer(config["data"]["seq_len"], "data.seq_len")
    integer(config["output"]["checkpoint_every"], "output.checkpoint_every", 0)
    execution = config["execution"]
    if execution["backend"] not in ("serial", "local", "distributed"):
        raise ValueError("Unknown execution backend")
    timeout = execution["timeout_seconds"]
    if timeout is not None and (
        type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError(
            "execution.timeout_seconds must be finite and positive, or null"
        )
    if config["model"]["head_policy"] not in (
        "global_trainable",
        "private_trainable",
        "frozen",
    ):
        raise ValueError("Unknown model.head_policy")
    if config["experiment"]["mode"] == "federated":
        integer(config["clients"]["count"], "clients.count")
        integer(config["simulation"]["rounds"], "simulation.rounds")
        integer(config["grouping"]["update_every"], "grouping.update_every", 0)
        if config["simulation"]["on_empty_round"] not in ("skip", "error"):
            raise ValueError("on_empty_round must be skip or error")
        clients = expand_clients(config["clients"])
        for client in clients.values():
            adapter, train, system = (
                client["adapter"],
                client["train"],
                client["system"],
            )
            integer(adapter["rank"], "client rank")
            integer(train["batch_size"], "client batch_size")
            integer(train["budget"]["value"], "client budget.value")
            if adapter["alpha"] <= 0 or train["budget"]["unit"] not in (
                "epochs",
                "steps",
            ):
                raise ValueError(
                    "Client alpha must be positive and budget unit epochs or steps"
                )
            if train["optimizer"]["name"] not in ("sgd", "adamw") or train["optimizer"][
                "state_policy"
            ] not in ("reset", "retain"):
                raise ValueError("Invalid client optimizer name or state_policy")
            if (
                min(
                    system["upload_mbps"],
                    system["download_mbps"],
                    system["time_multiplier"],
                )
                <= 0
                or system["latency_ms"] < 0
            ):
                raise ValueError(
                    "Client rates must be positive and latency nonnegative"
                )
        first = max(clients.values(), key=lambda c: c["adapter"]["rank"])["adapter"]
        config["model"]["adapter"].update(rank=first["rank"], alpha=first["alpha"])
        agg = config["aggregation"]
        if agg["mode"] not in ("flat", "hierarchical"):
            raise ValueError("aggregation.mode must be flat or hierarchical")
        if agg["mode"] == "flat" and agg["method"] == "fedavg":
            layouts = {
                (c["adapter"]["rank"], c["adapter"]["alpha"] / c["adapter"]["rank"])
                for c in clients.values()
            }
            if len(layouts) != 1:
                raise ValueError(
                    "FedAvg needs equal ranks/scaling; use SVD for heterogeneous ranks"
                )
        if (
            config["timing"]["mode"] != "off"
            and min(
                config["timing"]["assumed_step_seconds"],
                config["timing"]["assumed_aggregation_seconds"],
            )
            < 0
        ):
            raise ValueError("Assumed timing costs cannot be negative")
    else:
        if config["model"]["head_policy"] == "private_trainable":
            raise ValueError("Centralized training uses a global or frozen head")
        if config["timing"]["mode"] != "off":
            raise ValueError(
                "Centralized time is measured directly; timing.mode must be off"
            )
        train = config["training"]
        for key in ("batch_size", "micro_batch_size"):
            integer(train[key], "training." + key)
        integer(train["budget"]["value"], "training.budget.value")
        integer(config["model"]["adapter"]["rank"], "model.adapter.rank")
        integer(train["scheduler"]["warmup_steps"], "warmup_steps", 0)
        if train["budget"]["unit"] not in ("steps", "epochs") or train["optimizer"][
            "name"
        ] not in ("sgd", "adamw"):
            raise ValueError("Invalid centralized budget or optimizer")
        if (
            train["scheduler"]["name"] not in ("constant", "linear", "cosine")
            or config["model"]["adapter"]["alpha"] <= 0
        ):
            raise ValueError("Invalid centralized scheduler or alpha")
    evaluation = config["evaluation"]
    if evaluation["mode"] not in ("off", "sync", "async"):
        raise ValueError("Unknown evaluation mode")
    if evaluation["mode"] != "off":
        integer(evaluation["every"], "evaluation.every")
        integer(evaluation["batch_size"], "evaluation.batch_size")
        expected = "step" if config["experiment"]["mode"] == "centralized" else "round"
        if evaluation["unit"] != expected:
            raise ValueError(f"evaluation.unit must be {expected}")
        if evaluation["pending_policy"] not in ("latest", "block") or evaluation[
            "direction"
        ] not in ("min", "max"):
            raise ValueError("Invalid evaluation queue policy or metric direction")
        if evaluation["devices"] and evaluation["worker_ids"]:
            raise ValueError("Choose evaluation.devices or worker_ids")
        if evaluation["mode"] == "async" and execution["backend"] == "serial":
            raise ValueError("Use local or distributed for async evaluation")
    output = Path(config["output"]["root"]).resolve()
    for section in ("model", "data"):
        asset = config[section]
        if asset["source"] == "local" and asset["path"]:
            source = Path(asset["path"]).resolve()
            if output == source or source in output.parents:
                raise ValueError(
                    f"output.root cannot be inside the local {section} source"
                )


def validate_components(config, construct=False):
    from algorithm.registry import load_plugins, validate_spec, build
    from tasks import task_class

    load_plugins(config["plugins"])
    check = build if construct else validate_spec
    task = task_class(config["task"]["type"])
    evaluation = config["evaluation"]
    if (
        evaluation["mode"] != "off"
        and evaluation["save_best"]
        and evaluation["metric"] not in task.metric_names
    ):
        raise ValueError("evaluation.metric must be declared in task.metric_names")
    check("initialization", config["initialization"])
    if config["experiment"]["mode"] == "federated":
        from simulation.timing import VirtualClock  # noqa: F401 - register timing backends

        for kind in ("partition", "grouping", "selection"):
            check(kind, config[kind])
        agg = config["aggregation"]
        for spec in (
            [agg]
            if agg["mode"] == "flat"
            else [agg["within_group"], agg["across_groups"]]
        ):
            check("aggregation", spec)
        if config["timing"]["mode"] != "off":
            check(
                "timing",
                {"method": config["timing"]["mode"]},
                timing_config=config["timing"],
                model={**config["model"], "task": config["task"]["type"]},
            )
