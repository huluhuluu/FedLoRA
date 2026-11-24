"""Exercise real dispatch with delayed CPU-only jobs, without loading a backbone."""

import queue
import time
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from execution.pool import JobPool
from models.adapter_state import AdapterState
from utils.config import merge, validate
from utils.data import load_data
from utils.experiment import Experiment


class FakePool(JobPool):
    def __init__(self, config):
        validate(config)
        from algorithm.registry import load_plugins

        load_plugins(config["plugins"])
        bundle = load_data(config)
        bundle.evaluation = bundle.evaluation.select(range(3))
        self.initialize(config, bundle, ["cpu", "cpu"])
        self.responses = queue.Queue()
        self.threads = ThreadPoolExecutor(2)
        self.trace_values = []
        self.initial_state = AdapterState(
            {"marker": torch.tensor([1.0])}, {}, {}, "fixture"
        )

    def trace(self, value):
        self.trace_values.append(value)
        super().trace(value)

    def send(self, worker, job_id, kind, payload):
        def execute():
            time.sleep(0.09 if kind == "evaluate" else 0.025)
            if kind == "evaluate":
                _, _, indices = payload
                result = {
                    "statistics": {
                        "loss_sum": float(len(indices)),
                        "weight": len(indices),
                        "correct": sum(i % 2 == 0 for i in indices),
                    },
                    "physical_seconds": 0.09,
                }
            else:
                result = payload[1]
            self.responses.put((worker, job_id, ("ok", result)))

        self.threads.submit(execute)

    def receive(self, timeout):
        return self.responses.get(timeout=timeout)

    def close(self):
        self.flush_evaluations()
        self.threads.shutdown()


def test_eval_workers_return_during_pending_training(config):
    cfg = merge(
        config,
        {
            "execution": {"backend": "local", "devices": ["cpu", "cpu"]},
            "evaluation": {"mode": "async", "worker_ids": [1], "every": 1},
        },
    )
    pool = FakePool(cfg)
    try:
        pool.evaluate_version(pool.initial_state, 0, {"unit": "round", "index": 1})
        assert pool.run_jobs("gradient", pool.initial_state, list(range(16))) == list(
            range(16)
        )
        pool.flush_evaluations()
        events = pool.trace_values
        start = next(
            e for e in events if e["event"] == "job_started" and e["kind"] == "evaluate"
        )
        end = next(
            e
            for e in events
            if e["event"] == "job_finished" and e["kind"] == "evaluate"
        )
        assert any(
            e["event"] == "job_started"
            and e["kind"] == "gradient"
            and e["worker_id"] == 0
            and start["started_at"] <= e["started_at"] < end["completed_at"]
            for e in events
        )
        assert any(
            e["event"] == "job_started"
            and e["kind"] == "gradient"
            and e["worker_id"] == 1
            and e["started_at"] >= end["completed_at"]
            for e in events
        )
    finally:
        pool.close()


@pytest.mark.parametrize("policy,versions", [("latest", [1, 3]), ("block", [1, 2, 3])])
def test_bounded_pending_policy_and_versioned_best(config, policy, versions):
    cfg = merge(
        config,
        {
            "execution": {"backend": "local", "devices": ["cpu", "cpu"]},
            "evaluation": {
                "mode": "async",
                "worker_ids": [1],
                "every": 1,
                "pending_policy": policy,
                "save_best": True,
            },
        },
    )
    pool = FakePool(cfg)
    experiment = Experiment(cfg, {}, {}, pool.initial_state)
    pool.attach(experiment)
    try:
        for version in range(1, 4):
            pool.evaluate_version(
                pool.initial_state, version, {"unit": "round", "index": version}
            )
        pool.initial_state.tensors["marker"].fill_(99)
        pool.flush_evaluations()
        values = [
            json.loads(line)
            for line in (experiment.path / "evaluations.jsonl").read_text().splitlines()
        ]
        assert [v["model_version"] for v in values] == versions
        best = torch.load(experiment.path / "best_adapter.pt", weights_only=True)
        assert best["global_state"]["tensors"]["marker"].item() == 1
        assert best["model_version"] == 1
    finally:
        pool.close()


def test_eval_shards_merge_counts_before_metrics(config):
    cfg = merge(
        config,
        {
            "execution": {"backend": "local", "devices": ["cpu", "cpu"]},
            "evaluation": {"mode": "sync", "worker_ids": [0, 1]},
        },
    )
    pool = FakePool(cfg)
    try:
        result = pool.evaluate(pool.initial_state)
        assert result["accuracy"] == pytest.approx(2 / 3)
    finally:
        pool.close()
