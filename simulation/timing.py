"""Virtual client time is independent of real GPU queues."""

import json
import math
from pathlib import Path
from algorithm.registry import register


@register("timing", "analytic")
class Analytic:
    def __init__(self, timing_config, model):
        self.spec = timing_config

    def seconds(self, workload, client):
        return (
            (workload.batches or workload.optimizer_steps)
            * self.spec["assumed_step_seconds"]
            * client["system"]["time_multiplier"]
        )


@register("timing", "profiled")
class Profiled:
    def __init__(self, timing_config, model):
        self.model = model
        self.entries = json.loads(Path(timing_config["profile_file"]).read_text())[
            "entries"
        ]

    def seconds(self, workload, client):
        expected = {
            "model": self.model["path"],
            "task": self.model["task"],
            "precision": workload.precision,
            "rank": workload.rank,
            "batch_size": workload.batch_size,
            "seq_len": workload.seq_len,
            "hardware_profile": client["system"]["hardware_profile"],
        }
        if workload.compute_signature:
            expected["compute_signature"] = workload.compute_signature
        matches = [
            entry
            for entry in self.entries
            if all(entry.get(key) == value for key, value in expected.items())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Timing profile requires exactly one entry matching {expected}"
            )
        if (
            not math.isfinite(matches[0]["step_seconds"])
            or matches[0]["step_seconds"] <= 0
        ):
            raise ValueError("Profile step_seconds must be finite and positive")
        return (
            matches[0]["step_seconds"]
            * (workload.batches or workload.optimizer_steps)
            * client["system"]["time_multiplier"]
        )


class VirtualClock:
    def __init__(self, config):
        from algorithm.registry import build

        self.spec, self.elapsed = config["timing"], 0.0
        mode = self.spec["mode"]
        self.backend = (
            None
            if mode == "off"
            else build(
                "timing",
                {"method": mode},
                timing_config=self.spec,
                model={
                    **config["model"],
                    "task": config["task"].get("type", "classification"),
                },
            )
        )

    def client_seconds(self, update, client):
        if self.backend is None:
            return 0.0
        system = client["system"]
        download = update.workload.download_bytes * 8 / (system["download_mbps"] * 1e6)
        upload = update.workload.upload_bytes * 8 / (system["upload_mbps"] * 1e6)
        return (
            self.backend.seconds(update.workload, client)
            + download
            + upload
            + 2 * system["latency_ms"] / 1000
        )

    def advance(self, updates, clients, groups, hierarchical=False):
        if not updates:
            return 0.0, {}
        durations = {
            u.client_id: self.client_seconds(u, clients[u.client_id]) for u in updates
        }
        aggregation = self.spec["assumed_aggregation_seconds"] if self.backend else 0
        if hierarchical:
            group_times = [
                max((durations[cid] for cid in ids if cid in durations), default=0)
                + aggregation
                for ids in groups.values()
                if any(cid in durations for cid in ids)
            ]
            elapsed = max(group_times, default=0) + aggregation
        else:
            elapsed = max(durations.values(), default=0) + aggregation
        self.elapsed += elapsed
        return elapsed, durations
