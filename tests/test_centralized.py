"""Centralized training must match global-batch SGD and preserve state across resume."""

import json
import pytest
import torch
from training.centralized import BatchStream
from training.loop import make_optimizer, forward_batch
from utils.config import merge, validate


def central_config(config, **changes):
    result = merge(
        config,
        {
            "experiment": {"mode": "centralized"},
            "timing": {"mode": "off"},
            "aggregation": {"mode": "none", "method": None},
            "model": {"adapter": {"rank": 2, "alpha": 4}},
            "training": {
                "batch_size": 4,
                "micro_batch_size": 2,
                "budget": {"unit": "steps", "value": 3},
                "optimizer": {"name": "sgd", "lr": 0.03, "momentum": 0.5},
                "max_grad_norm": None,
            },
        },
    )
    return merge(result, changes)


def run(config, resume=None):
    from main import run as run_experiment

    validate(config)
    return run_experiment(config, resume)


@pytest.mark.parametrize("task", ["classification", "causal_lm"])
def test_global_minibatch_matches_standard_sgd(config, tmp_path, task):
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

    original = config["model"]["path"]
    target = tmp_path / "dropout-free"
    model_config = GPT2Config.from_pretrained(original)
    model_config.attn_pdrop = model_config.resid_pdrop = model_config.embd_pdrop = 0
    GPT2LMHeadModel.from_pretrained(original, config=model_config).save_pretrained(
        target
    )
    AutoTokenizer.from_pretrained(original).save_pretrained(target)
    changes = {"model": {"path": str(target)}, "task": {"type": task}}
    if task == "causal_lm":
        from datasets import Dataset, DatasetDict, load_from_disk

        raw = load_from_disk(config["data"]["path"])
        rows = raw["train"][:]
        rows["text"] = [
            "news world",
            "news world sports",
            "news world sports business science",
            "news sports business science",
        ] * 4
        train = Dataset.from_dict(rows, features=raw["train"].features)
        data = tmp_path / "variable-length-data"
        DatasetDict(train=train, test=train.select(range(4))).save_to_disk(data)
        changes["data"] = {"path": str(data)}
    cfg = central_config(config, **changes)
    trained = run(cfg)
    from models.peft_adapter import PeftAdapter

    reference = PeftAdapter(cfg, trained.bundle.info, "cpu")
    optimizer = make_optimizer(
        [p for p in reference.model.parameters() if p.requires_grad],
        cfg["training"]["optimizer"],
    )
    stream = BatchStream(len(trained.bundle.train), 4, cfg["experiment"]["seed"])
    reference.model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = forward_batch(reference, trained.bundle, stream.next())
        loss.backward()
        optimizer.step()
    for key, value in reference.export().tensors.items():
        assert torch.allclose(
            value, trained.state.tensors[key], atol=3e-7, rtol=1e-5
        ), key


@pytest.mark.parametrize("scheduler", ["constant", "linear"])
def test_central_resume_retains_optimizer_sampler_scheduler(config, scheduler):
    cfg = central_config(
        config,
        training={
            "optimizer": {"name": "adamw", "lr": 0.001},
            "scheduler": {"name": scheduler, "total_steps": 10},
        },
    )
    expected = run(cfg)
    first = run(merge(cfg, {"training": {"budget": {"value": 1}}}))
    resumed = run(cfg, first.experiment.path / "checkpoint_last.pt")
    for key, value in expected.state.tensors.items():
        assert torch.equal(value, resumed.state.tensors[key]), key
    assert (
        expected.optimizer.state_dict()["param_groups"]
        == resumed.optimizer.state_dict()["param_groups"]
    )
    checkpoint = torch.load(
        resumed.experiment.path / "checkpoint_last.pt", weights_only=True
    )
    assert checkpoint["stream"] == expected.stream.state_dict()
    assert checkpoint["next_step"] == 3
    assert checkpoint["scaler"] == expected.scaler.state_dict()


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_central_worker_migration_and_async_evaluation(config, precision):
    cfg = central_config(config, runtime={"precision": precision})
    expected = run(cfg)
    parallel = run(
        merge(
            cfg,
            {
                "execution": {"backend": "local", "devices": ["cpu", "cpu"]},
                "evaluation": {
                    "mode": "async",
                    "worker_ids": [1],
                    "every": 1,
                    "save_best": True,
                },
                "output": {"trace_dispatch": True},
            },
        )
    )
    for key, value in expected.state.tensors.items():
        assert torch.equal(value, parallel.state.tensors[key]), key
    evaluations = [
        json.loads(line)
        for line in (parallel.experiment.path / "evaluations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert evaluations[-1]["progress"] == {"unit": "step", "index": 3}
    assert evaluations[-1]["model_version"] == parallel.version
    best = torch.load(parallel.experiment.path / "best_adapter.pt", weights_only=True)
    assert best["kind"] == "best_adapter"
    assert best["model_version"] in [row["model_version"] for row in evaluations]


def test_custom_task_and_label_field(config, tmp_path):
    from datasets import DatasetDict, load_from_disk

    raw = load_from_disk(config["data"]["path"])
    path = tmp_path / "custom-columns"
    DatasetDict(
        {
            name: data.rename_column("text", "content").rename_column("label", "target")
            for name, data in raw.items()
        }
    ).save_to_disk(path)
    cfg = central_config(
        config,
        task={"type": "classification_f1"},
        plugins=["algorithm.custom.task_example"],
        data={"path": str(path), "text_fields": ["content"], "label_field": "target"},
    )
    trained = run(cfg)
    assert (
        "eval_macro_f1"
        in (trained.experiment.path / "rounds.csv").read_text().splitlines()[0]
    )
    assert (
        "macro_f1"
        in json.loads(
            (trained.experiment.path / "evaluations.jsonl").read_text().splitlines()[-1]
        )["metrics"]
    )


def test_distributed_centralized_async_matches_serial(config, tmp_path):
    import socket
    import subprocess
    import sys

    cfg = central_config(config)
    expected = run(cfg)
    distributed = merge(
        cfg,
        {
            "execution": {"backend": "distributed", "device": "cpu"},
            "evaluation": {"mode": "async", "worker_ids": [2], "every": 1},
            "output": {"root": str(tmp_path / "distributed")},
        },
    )
    path = tmp_path / "distributed.json"
    path.write_text(json.dumps(distributed))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--node-rank=0",
            "--nproc-per-node=3",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "main.py",
            "--config",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run_path = next((tmp_path / "distributed").iterdir())
    checkpoint = torch.load(run_path / "checkpoint_last.pt", weights_only=True)
    for key, value in expected.state.tensors.items():
        assert torch.equal(value, checkpoint["global_state"]["tensors"][key]), key
    evaluations = [
        json.loads(line)
        for line in (run_path / "evaluations.jsonl").read_text().splitlines()
    ]
    assert evaluations[-1]["progress"]["index"] == 3
