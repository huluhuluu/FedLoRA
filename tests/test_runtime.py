import json
import subprocess
import sys
import pytest
import torch
from utils.config import merge, validate


def run(config, resume=None):
    from main import run as run_experiment

    validate(config)
    return run_experiment(config, resume)


@pytest.mark.parametrize("policy", ["reset", "retain"])
def test_resume_matches_uninterrupted(config, policy):
    config = merge(
        config,
        {"clients": {"defaults": {"train": {"optimizer": {"state_policy": policy}}}}},
    )
    uninterrupted = run(merge(config, {"simulation": {"rounds": 2}}))
    first = run(config)
    resumed = run(
        merge(config, {"simulation": {"rounds": 2}}),
        first.experiment.path / "checkpoint_last.pt",
    )
    assert resumed.version == 2
    for key, value in uninterrupted.global_state.tensors.items():
        assert torch.equal(value, resumed.global_state.tensors[key]), key
    assert resumed.clock.elapsed == uninterrupted.clock.elapsed
    checkpoint = torch.load(
        resumed.experiment.path / "checkpoint_last.pt", weights_only=True
    )
    assert "global_state" in checkpoint and "model" not in checkpoint
    assert (resumed.experiment.path / "checkpoint_last.pt").stat().st_size < 100_000


@pytest.mark.parametrize("policy", ["global_trainable", "private_trainable", "frozen"])
def test_heterogeneous_rank_and_head_policy(config, policy):
    cfg = merge(
        config,
        {
            "model": {"head_policy": policy},
            "aggregation": {"method": "svd"},
            "clients": {
                "overrides": [{"client_ids": [1], "adapter": {"rank": 3, "alpha": 9}}]
            },
            "simulation": {"rounds": 2},
        },
    )
    engine = run(cfg)
    assert engine.version == 2
    extras = engine.global_state.extra_roles
    assert extras
    initial = engine.executor.initial_state
    if policy == "frozen":
        assert all(
            torch.equal(initial.tensors[k], engine.global_state.tensors[k])
            for k in extras
        )
    elif policy == "global_trainable":
        assert any(
            not torch.equal(initial.tensors[k], engine.global_state.tensors[k])
            for k in extras
        )
    if policy == "private_trainable":
        assert all(state.private for state in engine.persistent.values())
    else:
        assert all(not state.private for state in engine.persistent.values())


def test_causal_lm_and_eos_mask(config):
    cfg = merge(config, {"model": {"task": "causal_lm"}})
    engine = run(cfg)
    assert engine.executor.worker.net.tokenizer.padding_side == "right"
    assert engine.global_state.extra_roles == {}
    rows = [
        json.loads(line)
        for line in (engine.experiment.path / "events.jsonl").read_text().splitlines()
    ]
    assert rows[0]["evaluation"]["perplexity"] > 0
    from tasks.causal_lm import Task

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[3, 2, 2]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            }

    labels = Task(cfg["data"], {}).collate(Tokenizer(), [{"text": "news"}])["labels"]
    assert labels.tolist() == [[3, 2, -100]]


@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_two_local_workers_match_serial(config, heterogeneous, precision):
    config = merge(config, {"runtime": {"precision": precision}})
    if heterogeneous:
        config = merge(
            config,
            {
                "aggregation": {"method": "svd"},
                "simulation": {"rounds": 2},
                "clients": {
                    "defaults": {"train": {"optimizer": {"state_policy": "retain"}}},
                    "overrides": [
                        {"client_ids": [1], "adapter": {"rank": 3, "alpha": 9}}
                    ],
                },
            },
        )
    serial = run(config)
    for block in serial.global_state.blocks.values():
        assert serial.global_state.tensors[block.a_key].dtype == torch.float32
        assert serial.global_state.tensors[block.b_key].dtype == torch.float32
    parallel = run(
        merge(config, {"execution": {"backend": "local", "devices": ["cpu", "cpu"]}})
    )
    for key, value in serial.global_state.tensors.items():
        assert torch.equal(value, parallel.global_state.tensors[key]), key
    event = json.loads(
        (parallel.experiment.path / "events.jsonl").read_text().splitlines()[0]
    )
    assert {client["worker_id"] for client in event["clients"]} == {0, 1}


def test_output_guard_and_bad_config(config):
    from utils.helper import check_output_path

    with pytest.raises(ValueError):
        check_output_path("/data/HF_MODELS/test")
    with pytest.raises(ValueError):
        validate(
            merge(
                config,
                {
                    "clients": {
                        "overrides": [{"client_ids": [1], "adapter": {"rank": 3}}]
                    }
                },
            )
        )


def test_hierarchical_matches_flat_and_custom_selection(config):
    common = merge(
        config,
        {
            "clients": {"count": 3},
            "selection": {"params": {"clients_per_round": 3}},
            "grouping": {
                "method": "predefined",
                "params": {"groups": {"a": [0, 1], "b": [2]}},
            },
        },
    )
    flat = run(common)
    grouped = run(
        merge(
            common,
            {
                "aggregation": {
                    "mode": "hierarchical",
                    "within_group": {
                        "method": "fedavg",
                        "params": {"weighting": "samples"},
                    },
                    "across_groups": {
                        "method": "fedavg",
                        "params": {"weighting": "samples"},
                    },
                }
            },
        )
    )
    for key, value in flat.global_state.tensors.items():
        assert torch.allclose(value, grouped.global_state.tensors[key], atol=2e-7), key
    custom = run(
        merge(
            config,
            {
                "plugins": ["algorithm.custom.example"],
                "selection": {"method": "priority", "params": {"clients_per_round": 1}},
                "clients": {"overrides": [{"client_ids": [1], "priority": 10}]},
            },
        )
    )
    event = json.loads(
        (custom.experiment.path / "events.jsonl").read_text().splitlines()[0]
    )
    assert event["selected"] == [1]


def test_custom_initialization_hook(config):
    normal = run(config)
    custom = run(
        merge(
            config,
            {
                "plugins": ["algorithm.custom.example"],
                "initialization": {"method": "scaled_kaiming", "params": {"gain": 3}},
            },
        )
    )
    for block in normal.executor.initial_state.blocks.values():
        a = normal.executor.initial_state.tensors[block.a_key]
        assert torch.equal(a * 3, custom.executor.initial_state.tensors[block.a_key])


def test_distributed_resume_with_idle_rank(config, tmp_path):
    import socket

    common = merge(
        config,
        {"clients": {"defaults": {"train": {"optimizer": {"state_policy": "retain"}}}}},
    )
    expected = run(merge(common, {"simulation": {"rounds": 2}}))
    first = run(common)
    cfg = merge(
        common,
        {
            "simulation": {"rounds": 2},
            "execution": {"backend": "distributed", "device": "cpu"},
            "output": {"root": str(tmp_path / "distributed")},
        },
    )
    path = tmp_path / "distributed.json"
    path.write_text(json.dumps(cfg))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        "--nproc-per-node=3",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "main_fed.py",
        "--config",
        str(path),
        "--resume",
        str(first.experiment.path / "checkpoint_last.pt"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run_path = next((tmp_path / "distributed").iterdir())
    checkpoint = torch.load(run_path / "checkpoint_last.pt", weights_only=True)
    for key, value in expected.global_state.tensors.items():
        assert torch.equal(value, checkpoint["global_state"]["tensors"][key]), key
    event = json.loads((run_path / "events.jsonl").read_text().splitlines()[0])
    assert {u["worker_id"] for u in event["clients"]} == {0, 1}


def test_bert_text_pairs(config, assets, tmp_path):
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        PreTrainedTokenizerFast,
    )
    from tokenizers.processors import TemplateProcessing
    from datasets import load_from_disk, DatasetDict

    model_dir = tmp_path / "bert"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(assets[0])
    tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]"})
    tokenizer.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.cls_token_id),
            ("[SEP]", tokenizer.sep_token_id),
        ],
    )
    tokenizer.save_pretrained(model_dir)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            num_labels=4,
        )
    ).save_pretrained(model_dir)
    raw = load_from_disk(assets[1])
    data_dir = tmp_path / "pairs"
    DatasetDict(
        {
            key: value.rename_column("text", "premise").add_column(
                "hypothesis", ["news sports"] * len(value)
            )
            for key, value in raw.items()
        }
    ).save_to_disk(data_dir)
    cfg = merge(
        config,
        {
            "model": {
                "path": str(model_dir),
                "adapter": {"target_modules": ["query", "value"]},
            },
            "data": {"path": str(data_dir), "task": "mnli", "seq_len": 12},
        },
    )
    engine = run(cfg)
    net = engine.executor.worker.net
    batch = net.task.collate(net.tokenizer, [engine.bundle.train[0]])
    assert "token_type_ids" in batch and batch["token_type_ids"].max() == 1
    assert len(engine.global_state.blocks) == 2
