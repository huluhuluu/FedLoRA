"""Standard synchronous minibatch training with one persistent optimizer."""

import math
import time
import numpy as np
import torch
from models.adapter_state import AdapterState
from utils.experiment import Experiment, identity, load_checkpoint
from utils.helper import stable_seed
from training.loop import make_optimizer


class BatchStream:
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.cursor, self.cached_epoch = 0, 0, None

    def next(self):
        if self.cursor == self.size:
            self.epoch += 1
            self.cursor = 0
        if self.cached_epoch != self.epoch:
            self.order = np.random.default_rng(
                stable_seed(self.seed, "central-data", self.epoch)
            ).permutation(self.size)
            self.cached_epoch = self.epoch
        result = self.order[self.cursor : self.cursor + self.batch_size].tolist()
        self.cursor += len(result)
        return result

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.cached_epoch = None


class CentralizedTrainer:
    def __init__(self, config, bundle, executor, resume=None, checkpoint=None):
        self.config, self.bundle, self.executor = config, bundle, executor
        spec = config["training"]
        budget = spec["budget"]
        self.total_steps = (
            budget["value"]
            if budget["unit"] == "steps"
            else budget["value"] * math.ceil(len(bundle.train) / spec["batch_size"])
        )
        self.stream = BatchStream(
            len(bundle.train), spec["batch_size"], config["experiment"]["seed"]
        )
        saved = (
            checkpoint if checkpoint is not None else load_checkpoint(resume, config)
        )
        self.state = (
            AdapterState.from_dict(saved["global_state"])
            if saved
            else executor.initial_state
        )
        self.version, self.next_step = (
            (saved["global_version"], saved["next_step"]) if saved else (0, 0)
        )
        if self.next_step >= self.total_steps:
            raise ValueError("Resume requires a larger total training budget")
        self.parameters = {
            key: torch.nn.Parameter(value.clone())
            for key, value in sorted(self.state.tensors.items())
            if self.state.role(key) == "global"
        }
        self.optimizer = make_optimizer(self.parameters.values(), spec["optimizer"])
        scheduler = spec["scheduler"]
        horizon = (
            saved["scheduler_horizon"]
            if saved
            else scheduler.get("total_steps") or self.total_steps
        )
        self.horizon = horizon

        def multiplier(step):
            warmup = scheduler["warmup_steps"]
            if warmup and step < warmup:
                return (step + 1) / warmup
            fraction = min(1, max(0, (step - warmup) / max(1, horizon - warmup)))
            return (
                1
                if scheduler["name"] == "constant"
                else 1 - fraction
                if scheduler["name"] == "linear"
                else 0.5 * (1 + math.cos(math.pi * fraction))
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, multiplier)
        # GPU workers differentiate using this shared scale; CPU checks/updates it once per global batch.
        self.scaler = torch.amp.GradScaler(
            "cpu", enabled=config["runtime"]["precision"] == "fp16"
        )
        if self.scaler.is_enabled():
            self.scaler.scale(
                torch.zeros(())
            )  # Initialize the shared CPU scale once before workers differentiate.
        if saved:
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])
            self.scaler.load_state_dict(saved["scaler"])
            self.stream.load_state_dict(saved["stream"])
        self.experiment = Experiment(config, {}, {}, self.state, resume)
        executor.attach(self.experiment)

    def snapshot(self):
        values = {
            key: value.clone()
            for key, value in self.state.tensors.items()
            if key not in self.parameters
        }
        values.update(
            {key: value.detach().clone() for key, value in self.parameters.items()}
        )
        self.state = self.state.with_tensors(values)
        return self.state

    def checkpoint(self, next_step):
        self.experiment.checkpoint(
            {
                "kind": "centralized",
                "schema_version": 1,
                "identity": identity(self.config),
                "next_step": next_step,
                "global_version": self.version,
                "global_state": self.state.to_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "scheduler_horizon": self.horizon,
                "stream": self.stream.state_dict(),
            }
        )

    def run(self):
        print(f"EXPERIMENT_DIR={self.experiment.path}", flush=True)
        config, spec = self.config, self.config["training"]
        for step in range(self.next_step, self.total_steps):
            start = time.perf_counter()
            indices = self.stream.next()
            micro = spec["micro_batch_size"]
            jobs = [
                {
                    "task_id": i,
                    "version": self.version,
                    "indices": indices[offset : offset + micro],
                    "seed": stable_seed(
                        config["experiment"]["seed"], "central-gradient", step, i
                    ),
                    "scale": self.scaler.get_scale(),
                }
                for i, offset in enumerate(range(0, len(indices), micro))
            ]
            self.optimizer.zero_grad(set_to_none=True)
            totals, gradients, reports = {}, {}, []
            for index, result in enumerate(self.executor.gradients(self.state, jobs)):
                if result["version"] != self.version or result["task_id"] != index:
                    raise ValueError("Stale or missing centralized gradient")
                statistics = result["statistics"]
                for key, value in statistics.items():
                    totals[key] = totals.get(key, 0) + value
                for key, value in result["gradients"].items():
                    if key in gradients:
                        gradients[key].add_(value, alpha=statistics["weight"])
                    else:
                        gradients[key] = value.mul_(statistics["weight"])
                reports.append(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in ("gradients", "statistics")
                    }
                )
            metrics = self.executor.task.finalize(totals)
            for key, parameter in self.parameters.items():
                parameter.grad = gradients.get(key)
                if parameter.grad is not None:
                    parameter.grad.div_(totals["weight"])
            learning_rate = self.optimizer.param_groups[0]["lr"]
            self.scaler.unscale_(self.optimizer)
            if spec.get("max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    list(self.parameters.values()), spec["max_grad_norm"]
                )
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            updated = self.scaler.get_scale() >= old_scale
            if updated:
                self.version += 1
                self.scheduler.step()
            if updated:
                self.snapshot()
            final = step + 1 == self.total_steps
            evaluation = self.executor.evaluate_version(
                self.state,
                self.version,
                {"unit": "step", "index": step + 1},
                final=final,
            )
            self.experiment.record(
                {
                    "step": step + 1,
                    "epoch": self.stream.epoch,
                    "global_version": self.version,
                    "train_loss": metrics["loss"],
                    **{
                        "eval_" + key: value
                        for key, value in evaluation.items()
                        if key != "physical_seconds"
                    },
                    "physical_round_seconds": time.perf_counter() - start,
                    "virtual_round_seconds": 0,
                    "virtual_total_seconds": 0,
                    "samples_processed": len(indices),
                    "optimizer_steps": int(updated),
                    "learning_rate": learning_rate,
                },
                {
                    "step": step + 1,
                    "mode": "centralized",
                    "evaluation": evaluation,
                    "train": metrics,
                    "gradients": reports,
                },
            )
            interval = config["output"]["checkpoint_every"]
            if interval and (step + 1) % interval == 0 or final:
                self.checkpoint(step + 1)
        self.executor.flush_evaluations()
