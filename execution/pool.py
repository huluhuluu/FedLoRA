"""Bounded task dispatch; evaluation workers rejoin training as they become idle."""

import time
from collections import deque
from tasks import build_task
from execution.evaluation import Evaluator


class JobPool:
    is_leader = True

    def initialize(self, config, bundle, devices):
        self.config, self.bundle, self.worker_devices = config, bundle, list(devices)
        self.task = build_task(config, bundle.info)
        self.busy, self.sequence, self.worker_states = {}, 0, {}
        self.evaluator = Evaluator(self, config)

    def attach(self, experiment):
        self.experiment = experiment
        self.evaluator.attach(experiment)
        from utils.helper import json_write

        json_write(
            experiment.path / "workers.json",
            {
                "workers": [
                    {"worker_id": i, "device": device}
                    for i, device in enumerate(self.worker_devices)
                ],
                "evaluation_workers": self.evaluator.workers,
            },
        )

    def submit(self, worker, kind, payload, tag):
        self.sequence += 1
        request = {
            "id": self.sequence,
            "kind": kind,
            "tag": tag,
            "worker_id": worker,
            "started_at": time.perf_counter(),
        }
        if kind == "train":
            request.update(
                model_version=payload[1].global_version, logical_id=payload[1].client_id
            )
        elif kind == "gradient" and isinstance(payload[1], dict):
            request.update(
                model_version=payload[1]["version"], logical_id=payload[1]["task_id"]
            )
        elif kind == "evaluate":
            request.update(model_version=self.evaluator.active["version"])
        state, *arguments = payload
        # One in-flight job per worker makes a single cached CPU reference sufficient.
        reference = None if self.worker_states.get(worker) is state else state
        self.worker_states[worker] = state
        self.busy[worker] = request
        self.send(worker, request["id"], kind, (reference, *arguments))
        self.trace({"event": "job_started", **request})
        return request["id"]

    def trace(self, event):
        if getattr(self, "experiment", None) and self.config["output"].get(
            "trace_dispatch", False
        ):
            self.experiment.dispatch(event)

    def receive_one(self):
        timeout = self.config["execution"]["timeout_seconds"]
        if timeout is not None:
            now = time.perf_counter()
            remaining = [
                timeout - (now - request["started_at"])
                for request in self.busy.values()
            ]
            timeout = min(remaining)
            if timeout <= 0:
                raise TimeoutError("Execution task deadline exceeded")
        worker, job_id, response = self.receive(timeout)
        request = self.busy.pop(worker)
        if request["id"] != job_id:
            raise RuntimeError("Worker response does not match its active job")
        status, result = response
        self.trace(
            {"event": "job_finished", **request, "completed_at": time.perf_counter()}
        )
        if status != "ok":
            raise RuntimeError(f"Worker {worker} failed:\n{result}")
        if request["tag"][0] == "eval":
            self.evaluator.complete(request, result, worker)
            completed = None
        else:
            completed = (request["tag"][1], result)
        self.evaluator.dispatch()
        return completed

    def drive_evaluation(self):
        self.evaluator.dispatch()
        if self.busy:
            self.receive_one()
        elif self.evaluator.active is not None or self.evaluator.pending is not None:
            raise RuntimeError("Evaluation has no runnable worker")

    def iter_jobs(self, kind, state, tasks, ordered=True):
        pending = deque(enumerate(tasks))
        waiting, ready, next_index = set(), {}, 0
        # Fixed-order reduction remains deterministic without retaining an entire global batch.
        window = (2 if ordered else 1) * len(self.worker_devices)
        while pending or waiting or ready:
            while next_index in ready:
                result = ready.pop(next_index)
                next_index += 1
                yield result
            if not pending and not waiting:
                break
            self.evaluator.dispatch()
            for worker in range(len(self.worker_devices)):
                if not pending or len(waiting) + len(ready) >= window:
                    break
                if worker in self.busy:
                    continue
                index, task = pending.popleft()
                self.submit(worker, kind, (state, task), ("train", index))
                waiting.add(index)
            if not self.busy:
                raise RuntimeError("No worker available for pending training")
            completed = self.receive_one()
            if completed is not None:
                index, result = completed
                waiting.remove(index)
                if ordered:
                    ready[index] = result
                else:
                    yield result

    def run_jobs(self, kind, state, tasks):
        return list(self.iter_jobs(kind, state, tasks))

    def run_clients(self, state, tasks):
        if self.config["execution"]["placement"] == "balanced_work":

            def cost(task):
                train, budget = task.config["train"], task.config["train"]["budget"]
                steps = (
                    budget["value"]
                    if budget["unit"] == "steps"
                    else budget["value"]
                    * (
                        (len(task.indices) + train["batch_size"] - 1)
                        // train["batch_size"]
                    )
                )
                return steps * train["batch_size"] * task.config["adapter"]["rank"]

            tasks = sorted(tasks, key=cost, reverse=True)
        elif self.config["execution"]["placement"] != "balanced_count":
            raise ValueError("Unknown execution.placement")
        return sorted(
            self.iter_jobs("train", state, tasks, ordered=False),
            key=lambda value: value.client_id,
        )

    def gradients(self, state, jobs):
        return self.iter_jobs("gradient", state, jobs)

    def evaluate_version(self, state, version, progress, clients=None, final=False):
        return self.evaluator.request(state, version, progress, clients, final)

    def evaluate(self, state, client=None):
        return self.evaluator.request(
            state,
            0,
            {"unit": "round", "index": 0},
            [client] if client else None,
            force=True,
        )

    def flush_evaluations(self):
        self.evaluator.flush()
