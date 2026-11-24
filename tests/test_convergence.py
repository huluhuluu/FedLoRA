"""Regression coverage for resource lifetime, bounded dispatch and resume inputs."""

import copy
import gc
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import torch
from algorithm.registry import load_plugins, register
from execution.pool import JobPool
from models.adapter_state import AdapterState
from utils.config import merge, validate, load_config
from test_centralized import central_config, run


def test_disabled_evaluation_needs_no_eval_split_or_client_configuration(config):
    cfg = central_config(
        config,
        clients={"count": 0},
        simulation={"rounds": 0},
        evaluation={
            "mode": "off",
            "worker_ids": [99],
            "every": 0,
            "batch_size": 0,
            "save_best": True,
            "metric": "unused",
        },
        data={"eval_split": "missing"},
        output={"checkpoint_every": 0},
    )
    trainer = run(cfg)
    assert trainer.bundle.evaluation is None
    assert not (trainer.experiment.path / "evaluations.csv").exists()
    assert (
        json.loads((trainer.experiment.path / "status.json").read_text())["status"]
        == "completed"
    )


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_cleanup_reports_failure_without_masking_original(
    config, monkeypatch, evaluation_fails
):
    from execution.local import SerialExecutor
    import models.test

    if evaluation_fails:

        def fail_evaluation(*args, **kwargs):
            raise ValueError("evaluation sentinel")

        monkeypatch.setattr(models.test, "evaluate_statistics", fail_evaluation)

    def fail_close(*args, **kwargs):
        raise RuntimeError("cleanup sentinel")

    monkeypatch.setattr(SerialExecutor, "close", fail_close)
    message = "evaluation sentinel" if evaluation_fails else "cleanup sentinel"
    with pytest.raises(RuntimeError, match=message):
        run(central_config(config))
    path = next(Path(config["output"]["root"]).iterdir())
    status = json.loads((path / "status.json").read_text())
    assert status["status"] == "failed" and message in status["error"]


def test_distributed_constructor_failure_releases_processes(config, tmp_path):
    cfg = central_config(
        config,
        execution={"backend": "distributed", "devices": ["cpu"]},
        evaluation={"worker_ids": [9]},
    )
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(cfg))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    code = r"""
import multiprocessing as mp,sys
import torch.distributed as dist
from utils.config import load_config
from main import run
try:
    try:run(load_config(sys.argv[1]))
    except ValueError as exc:assert "worker_ids" in str(exc)
    else:raise AssertionError("invalid worker accepted")
    assert not mp.active_children()
    assert not dist.is_initialized()
finally:
    for child in mp.active_children():
        child.terminate();child.join(timeout=3)
    if dist.is_initialized():dist.destroy_process_group()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=40,
        env={
            **os.environ,
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "GLOO_SOCKET_IFNAME": "lo",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_distributed_suite_uses_the_configured_gpu_list(monkeypatch):
    from execution import rank_device
    from tools.run_suite import launch

    cfg = load_config(
        "experiments/configs/agnews_gpt2_shared_smoke.json",
        ["execution.backend=distributed"],
    )
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,1,0")
    assert rank_device(cfg) == "cuda:2"
    with patch("tools.run_suite.subprocess.run") as invoked:
        launch(cfg, Path("unused.json"))
    command = invoked.call_args.args[0]
    assert "--nproc-per-node=4" in command
    assert invoked.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3,2,1,0"


def test_gradient_stream_is_ordered_bounded_and_reuses_snapshots(config):
    counters = {"live": 0, "peak": 0, "full_states": 0}
    lock = threading.Lock()

    class Result:
        def __init__(self, index):
            self.index, self.payload = index, bytearray(128 * 1024)
            with lock:
                counters["live"] += 1
                counters["peak"] = max(counters["peak"], counters["live"])

        def __del__(self):
            with lock:
                counters["live"] -= 1

    class Pool(JobPool):
        def __init__(self, cfg):
            self.responses = queue.Queue()
            self.threads = ThreadPoolExecutor(2)
            self.initialize(cfg, type("Bundle", (), {"info": {}})(), ["cpu", "cpu"])

        def send(self, worker, job_id, kind, payload):
            state, index = payload
            counters["full_states"] += int(state is not None)

            def compute():
                time.sleep(0.08 if index == 0 else 0.001)
                self.responses.put((worker, job_id, ("ok", Result(index))))

            self.threads.submit(compute)

        def receive(self, timeout):
            return self.responses.get(timeout=timeout)

    cfg = merge(config, {"evaluation": {"mode": "off"}})
    validate(cfg)
    load_plugins([])
    pool = Pool(cfg)
    state = AdapterState({"marker": torch.ones(1)}, {}, {}, "fixture")
    try:
        assert [
            result.index for result in pool.gradients(state, list(range(80)))
        ] == list(range(80))
    finally:
        pool.threads.shutdown()
    gc.collect()
    assert counters["full_states"] == 2
    assert counters["peak"] <= 7
    assert counters["live"] == 0


def test_disabled_best_metric_and_large_perplexity_are_serializable(config):
    from test_evaluation_pool import FakePool
    from utils.experiment import Experiment

    cfg = merge(
        config,
        {
            "task": {"type": "causal_lm"},
            "evaluation": {"mode": "sync", "save_best": False, "metric": "unused"},
        },
    )
    pool = FakePool(cfg)
    experiment = Experiment(cfg, {}, {}, pool.initial_state)
    pool.attach(experiment)
    finalize = pool.task.finalize
    pool.task.finalize = lambda totals: finalize(
        {**totals, "loss_sum": 701 * totals["weight"]}
    )
    try:
        metrics = pool.evaluate(pool.initial_state)
        assert metrics["perplexity"] is None
        row = json.loads((experiment.path / "evaluations.jsonl").read_text())
        assert row["metrics"]["perplexity"] is None
    finally:
        pool.close()


def test_stateful_timing_continues_after_resume(config):
    @register("timing", "counting_test")
    class Counting:
        def __init__(self, timing_config, model):
            self.count = 0

        def seconds(self, workload, client):
            self.count += 1
            return self.count

        def state_dict(self):
            return {"count": self.count}

        def load_state_dict(self, state):
            self.count = state["count"]

    from test_runtime import run as federated_run

    cfg = merge(
        config, {"timing": {"mode": "counting_test"}, "simulation": {"rounds": 2}}
    )
    expected = federated_run(copy.deepcopy(cfg))
    first = federated_run(merge(cfg, {"simulation": {"rounds": 1}}))
    resumed = federated_run(cfg, first.experiment.path / "checkpoint_last.pt")
    assert resumed.clock.elapsed == expected.clock.elapsed
    assert resumed.clock.backend.count == expected.clock.backend.count


def test_legacy_resume_ignores_inactive_options_and_skips_initialization(
    config, monkeypatch
):
    from algorithm.initialization import Kaiming
    from test_runtime import run as federated_run

    expected = federated_run(merge(config, {"simulation": {"rounds": 2}}))
    first = federated_run(copy.deepcopy(config))
    path = first.experiment.path / "checkpoint_last.pt"
    saved = torch.load(path, weights_only=True)
    saved["identity"] = "legacy-broader-configuration-hash"
    torch.save(saved, path)

    def unexpected(*args):
        raise AssertionError("initializer reran during resume")

    monkeypatch.setattr(Kaiming, "peft_value", unexpected)
    cfg = merge(
        config,
        {
            "experiment": {"name": "renamed"},
            "training": {"batch_size": 999},
            "runtime": {"cpu_threads": 1},
            "simulation": {"rounds": 2},
        },
    )
    resumed = federated_run(cfg, path)
    for key, value in expected.global_state.tensors.items():
        assert torch.equal(value, resumed.global_state.tensors[key])


def test_worker_process_exit_is_reported_without_a_task_timeout(config):
    from execution.local import LocalExecutor
    from utils.data import load_data

    cfg = merge(
        config,
        {
            "execution": {"backend": "local", "devices": ["cpu"]},
            "evaluation": {"mode": "off"},
        },
    )
    validate(cfg)
    load_plugins([])
    executor = LocalExecutor(cfg, load_data(cfg), initialize=False)
    try:
        executor.processes[0].kill()
        executor.processes[0].join(timeout=3)
        with pytest.raises(RuntimeError, match="exited"):
            executor.receive(None)
    finally:
        executor.close(cancel=True)
    assert not executor.processes[0].is_alive()


def test_slow_client_does_not_block_dispatch_to_other_workers(config):
    from types import SimpleNamespace

    times = {}

    class Pool(JobPool):
        def __init__(self, cfg):
            self.responses = queue.Queue()
            self.threads = ThreadPoolExecutor(2)
            self.initialize(cfg, SimpleNamespace(info={}), ["cpu", "cpu"])

        def send(self, worker, job_id, kind, payload):
            task = payload[1]

            def compute():
                times[("start", task.client_id)] = time.perf_counter()
                time.sleep(0.15 if task.client_id == 0 else 0.002)
                times[("end", task.client_id)] = time.perf_counter()
                self.responses.put(
                    (worker, job_id, ("ok", SimpleNamespace(client_id=task.client_id)))
                )

            self.threads.submit(compute)

        def receive(self, timeout):
            return self.responses.get(timeout=timeout)

    cfg = merge(config, {"evaluation": {"mode": "off"}})
    validate(cfg)
    load_plugins([])
    pool = Pool(cfg)
    state = AdapterState({"marker": torch.ones(1)}, {}, {}, "fixture")
    tasks = [SimpleNamespace(client_id=i, global_version=0) for i in range(24)]
    try:
        assert [result.client_id for result in pool.run_clients(state, tasks)] == list(
            range(24)
        )
    finally:
        pool.threads.shutdown()
    assert times[("start", 12)] < times[("end", 0)]
