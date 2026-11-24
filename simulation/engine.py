"""Synchronous rounds over logical clients, independent of worker placement."""

import time
from dataclasses import asdict
import numpy as np
from algorithm.interfaces import ClientUpdate, context
from algorithm.registry import build
from models.adapter_state import AdapterState
from simulation.client import ClientState, ClientTask
from simulation.timing import VirtualClock
from utils.config import expand_clients
from utils.experiment import Experiment, identity, load_checkpoint
from utils.helper import stable_seed


class Engine:
    def __init__(self, config, bundle, executor, resume=None, checkpoint=None):
        self.config, self.bundle, self.executor = config, bundle, executor
        self.clients = expand_clients(config["clients"])
        self.algorithms = {
            kind: build(kind, config[kind])
            for kind in ("partition", "grouping", "selection")
        }
        agg = config["aggregation"]
        self.hierarchical = agg["mode"] == "hierarchical"
        self.aggregators = (
            [
                build("aggregation", agg["within_group"]),
                build("aggregation", agg["across_groups"]),
            ]
            if self.hierarchical
            else [build("aggregation", agg)]
        )
        self.clock = VirtualClock(config)
        self.global_state, self.version, self.next_round = executor.initial_state, 0, 0
        self.persistent = {cid: ClientState() for cid in self.clients}
        self.observations = None
        saved = (
            checkpoint if checkpoint is not None else load_checkpoint(resume, config)
        )
        if saved:
            self.global_state = AdapterState.from_dict(saved["global_state"])
            self.version, self.next_round = saved["global_version"], saved["next_round"]
            self.persistent = {
                int(cid): ClientState.from_dict(value)
                for cid, value in saved["clients"].items()
            }
            self.partitions, self.groups = saved["partitions"], saved["groups"]
            self.clock.elapsed = saved["virtual_seconds"]
            for key, algorithm in self._all_algorithms().items():
                if key in saved["algorithms"]:
                    algorithm.load_state_dict(saved["algorithms"][key])
        else:
            self.partitions = self.algorithms["partition"].partition(
                bundle.train, self.clients, context(config, "partition")
            )
            flattened = [
                int(index) for ids in self.partitions.values() for index in ids
            ]
            if sorted(self.partitions) != sorted(self.clients) or sorted(
                flattened
            ) != list(range(len(bundle.train))):
                raise ValueError(
                    "Partition must assign every training sample exactly once"
                )
            self.partitions = {
                cid: list(map(int, ids)) for cid, ids in self.partitions.items()
            }
            self.groups = self._group(0)
        if self.next_round >= config["simulation"]["rounds"]:
            raise ValueError(
                "Resume needs simulation.rounds greater than the completed round count"
            )
        self.experiment = Experiment(
            config, self.clients, self.partitions, self.global_state, resume
        )
        executor.attach(self.experiment)

    def _all_algorithms(self):
        return {
            **self.algorithms,
            **{f"aggregation_{i}": value for i, value in enumerate(self.aggregators)},
            **(
                {"timing": self.clock.backend} if self.clock.backend is not None else {}
            ),
        }

    def _group(self, round_id):
        if self.observations is None:
            if self.config["grouping"]["method"] in (
                "none",
                "predefined",
                "resource_profile",
            ):
                self.observations = {}
            else:
                label_values = (
                    list(self.bundle.train["label"])
                    if "label" in self.bundle.train.column_names
                    else [0] * len(self.bundle.train)
                )
                names = sorted(set(label_values))
                observations = {}
                labels = np.asarray(label_values)
                for cid, indices in self.partitions.items():
                    counts = np.asarray(
                        [(labels[indices] == label).sum() for label in names],
                        dtype=float,
                    )
                    observations[cid] = {
                        "num_samples": len(indices),
                        "label_distribution": (counts / max(len(indices), 1)).tolist(),
                    }
                self.observations = observations
        groups = self.algorithms["grouping"].group(
            self.clients,
            self.observations,
            context(self.config, "grouping", round_id, self.version),
        )
        if sorted(cid for ids in groups.values() for cid in ids) != sorted(
            self.clients
        ):
            raise ValueError("Grouping must cover each client exactly once")
        return groups

    def _aggregate(self, updates, round_id):
        ctx = context(self.config, "aggregation", round_id, self.version)
        if not self.hierarchical:
            result = self.aggregators[0].aggregate(updates, self.global_state, ctx)
            self.aggregation_events.append(
                {
                    "stage": "global",
                    "metrics": result.metrics,
                    "workload": asdict(result.workload),
                }
            )
            return result.state
        group_updates = []
        for gid, ids in enumerate(self.groups.values()):
            subset = [update for update in updates if update.client_id in ids]
            if subset:
                result = self.aggregators[0].aggregate(subset, self.global_state, ctx)
                self.aggregation_events.append(
                    {
                        "stage": f"group_{gid}",
                        "metrics": result.metrics,
                        "workload": asdict(result.workload),
                    }
                )
                state = result.state
                group_updates.append(
                    ClientUpdate(
                        gid,
                        round_id,
                        self.version,
                        state,
                        sum(u.num_samples for u in subset),
                    )
                )
        result = self.aggregators[1].aggregate(group_updates, self.global_state, ctx)
        self.aggregation_events.append(
            {
                "stage": "across_groups",
                "metrics": result.metrics,
                "workload": asdict(result.workload),
            }
        )
        return result.state

    def checkpoint(self, next_round):
        algorithms = {
            key: value.state_dict()
            for key, value in self._all_algorithms().items()
            if hasattr(value, "state_dict")
        }
        self.experiment.checkpoint(
            {
                "kind": "federated",
                "schema_version": 1,
                "identity": identity(self.config),
                "next_round": next_round,
                "global_version": self.version,
                "global_state": self.global_state.to_dict(),
                "clients": {
                    cid: state.to_dict() for cid, state in self.persistent.items()
                },
                "partitions": self.partitions,
                "groups": self.groups,
                "virtual_seconds": self.clock.elapsed,
                "algorithms": algorithms,
            }
        )

    def run(self):
        config = self.config
        print(f"EXPERIMENT_DIR={self.experiment.path}", flush=True)
        for round_id in range(self.next_round, config["simulation"]["rounds"]):
            start = time.perf_counter()
            interval = config["grouping"]["update_every"]
            if interval and round_id and round_id % interval == 0:
                self.groups = self._group(round_id)
            eligible = {
                cid: spec for cid, spec in self.clients.items() if self.partitions[cid]
            }
            selected = self.algorithms["selection"].select(
                eligible,
                self.groups,
                context(config, "selection", round_id, self.version),
            )
            if len(set(selected)) != len(selected) or any(
                cid not in eligible for cid in selected
            ):
                raise ValueError("Selection returned duplicate or ineligible clients")
            if not selected and config["simulation"]["on_empty_round"] != "skip":
                raise ValueError("No eligible client selected")
            tasks = [
                ClientTask(
                    cid,
                    round_id,
                    self.version,
                    self.clients[cid],
                    self.partitions[cid],
                    stable_seed(config["experiment"]["seed"], "train", round_id, cid),
                    self.persistent[cid],
                )
                for cid in selected
            ]
            updates = (
                self.executor.run_clients(self.global_state, tasks) if tasks else []
            )
            if sorted(u.client_id for u in updates) != sorted(selected):
                raise ValueError("Workers returned an incomplete round")
            if any(
                u.global_version != self.version or u.round_id != round_id
                for u in updates
            ):
                raise ValueError("Stale client update")
            self.aggregation_events = []
            if updates:
                self.global_state = self._aggregate(updates, round_id)
                self.version += 1
                for update in updates:
                    self.persistent[update.client_id] = ClientState(
                        update.state.private_tensors(),
                        update.optimizer_state,
                        update.scaler_state,
                    )
            virtual, durations = self.clock.advance(
                updates, self.clients, self.groups, self.hierarchical
            )
            personalized = (
                [(self.clients[cid], self.persistent[cid]) for cid in selected]
                if "private" in self.global_state.extra_roles.values()
                else None
            )
            evaluation = self.executor.evaluate_version(
                self.global_state,
                self.version,
                {"unit": "round", "index": round_id + 1},
                personalized,
                final=round_id + 1 == config["simulation"]["rounds"],
            )
            processed = sum(u.metrics["loss_weight"] for u in updates)
            row = {
                "round": round_id + 1,
                "global_version": self.version,
                "selected": len(selected),
                "train_loss": sum(
                    u.metrics["loss"] * u.metrics["loss_weight"] for u in updates
                )
                / processed
                if processed
                else "",
                **{
                    "eval_" + key: value
                    for key, value in evaluation.items()
                    if key != "physical_seconds"
                },
                "virtual_round_seconds": virtual,
                "virtual_total_seconds": self.clock.elapsed,
                "physical_round_seconds": time.perf_counter() - start,
                "uploaded_bytes": sum(u.workload.upload_bytes for u in updates),
                "downloaded_bytes": sum(u.workload.download_bytes for u in updates),
            }
            self.experiment.record(
                row,
                {
                    "round": round_id + 1,
                    "selected": selected,
                    "groups": self.groups,
                    "evaluation": evaluation,
                    "virtual_client_seconds": durations,
                    "aggregation": self.aggregation_events,
                    "clients": [
                        {
                            "client_id": u.client_id,
                            "worker_id": u.worker_id,
                            "metrics": u.metrics,
                            "workload": asdict(u.workload),
                        }
                        for u in updates
                    ],
                },
            )
            interval = config["output"]["checkpoint_every"]
            if (
                interval
                and (round_id + 1) % interval == 0
                or round_id + 1 == config["simulation"]["rounds"]
            ):
                self.checkpoint(round_id + 1)
        self.executor.flush_evaluations()
