"""Versioned evaluation snapshots on workers temporarily borrowed from training."""

import time
import numpy as np
from tasks import merge_statistics
from simulation.client import ClientState
from utils.helper import cpu_tree


class Evaluator:
    def __init__(self, pool, config):
        self.pool, self.spec = pool, config["evaluation"]
        self.active, self.pending, self.last = None, None, None
        self.sequence, self.best = 0, None
        if self.spec["mode"] == "off":
            self.workers = []
            return
        devices = self.spec["devices"]
        if self.spec["worker_ids"]:
            self.workers = list(self.spec["worker_ids"])
        elif devices:
            self.workers = []
            for device in devices:
                matches = [
                    i for i, value in enumerate(pool.worker_devices) if value == device
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"evaluation device {device} must identify one worker; use worker_ids for repeated devices"
                    )
                self.workers.append(matches[0])
        else:
            self.workers = [
                len(pool.worker_devices) - 1 if self.spec["mode"] == "async" else 0
            ]
        if len(set(self.workers)) != len(self.workers) or any(
            i not in range(len(pool.worker_devices)) for i in self.workers
        ):
            raise ValueError("Invalid evaluation worker_ids")

    def attach(self, experiment):
        self.experiment = experiment

    def event(self, value):
        if hasattr(self, "experiment"):
            self.experiment.event(value)

    def request(self, state, version, progress, clients=None, final=False, force=False):
        spec = self.spec
        if spec["mode"] == "off":
            return {}
        due = (
            force
            or progress["index"] % spec["every"] == 0
            or (final and spec["at_end"])
        )
        if not due:
            if final:
                self.flush()
            return {}
        while self.pending is not None and spec["pending_policy"] == "block":
            self.pool.drive_evaluation()
        self.sequence += 1
        # Only trainable adapter/head snapshots are copied, never the backbone.
        job = {
            "id": self.sequence,
            "state": state.clone(),
            "version": version,
            "progress": dict(progress),
            "created": time.perf_counter(),
            "clients": [],
        }
        for config, persistent in clients or []:
            job["clients"].append((config, ClientState(cpu_tree(persistent.private))))
        if self.pending is not None:
            self.event(
                {
                    "event": "evaluation_replaced",
                    "model_version": self.pending["version"],
                    "progress": self.pending["progress"],
                }
            )
        self.pending = job
        self.dispatch()
        if spec["mode"] == "sync" or final or force:
            while self.last is None or self.last["job_id"] != job["id"]:
                self.pool.drive_evaluation()
            return self.last["metrics"]
        return {}

    def dispatch(self):
        if self.active is None and self.pending is not None:
            self.active, self.pending = self.pending, None
            job = self.active
            chunks = [
                part.tolist()
                for part in np.array_split(
                    np.arange(len(self.pool.bundle.evaluation)), len(self.workers)
                )
                if len(part)
            ]
            clients = job["clients"] or [None]
            job["todo"] = [
                (cid, chunk, client)
                for cid, client in enumerate(clients)
                for chunk in chunks
            ]
            job["inflight"], job["results"] = 0, []
            job["started"] = time.perf_counter()
        job = self.active
        if job is None:
            return
        for worker in self.workers:
            if worker not in self.pool.busy and job["todo"]:
                cid, indices, client = job["todo"].pop(0)
                self.pool.submit(
                    worker,
                    "evaluate",
                    (job["state"], client, indices),
                    ("eval", job["id"], cid),
                )
                job["inflight"] += 1

    def complete(self, request, result, worker):
        job = self.active
        job["inflight"] -= 1
        job["results"].append((request["tag"][2], result, worker))
        if job["todo"] or job["inflight"]:
            return
        values = []
        for cid in range(len(job["clients"]) or 1):
            stats = merge_statistics(
                result["statistics"] for c, result, _ in job["results"] if c == cid
            )
            values.append(self.pool.task.finalize(stats))
        metrics = {
            key: (
                None
                if any(value[key] is None for value in values)
                else sum(value[key] for value in values) / len(values)
            )
            for key in values[0]
        }
        elapsed = time.perf_counter() - job["started"]
        metrics["physical_seconds"] = elapsed
        row = {
            "job_id": job["id"],
            "model_version": job["version"],
            "progress": job["progress"],
            "metrics": metrics,
            "worker_ids": sorted({worker for _, _, worker in job["results"]}),
            "submitted_at": job["created"],
            "started_at": job["started"],
            "completed_at": time.perf_counter(),
        }
        self.last = row
        if hasattr(self, "experiment"):
            self.experiment.record_evaluation(row)
            metric = metrics[self.spec["metric"]] if self.spec["save_best"] else None
            improved = metric is not None and (
                self.best is None
                or (
                    metric < self.best
                    if self.spec["direction"] == "min"
                    else metric > self.best
                )
            )
            if improved:
                self.best = metric
                self.experiment.save_best(
                    {
                        "kind": "best_adapter",
                        "model_version": job["version"],
                        "global_state": job["state"].to_dict(),
                        "metrics": metrics,
                        "clients": [
                            {"config": cfg, "private": persistent.private}
                            for cfg, persistent in job["clients"]
                        ],
                    }
                )
        self.active = None

    def flush(self):
        while self.active is not None or self.pending is not None:
            self.pool.drive_evaluation()
