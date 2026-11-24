"""Common batch computation; orchestration owns optimizer lifetime."""

import time
import torch
from utils.helper import cpu_tree, seed_training


def make_optimizer(parameters, spec):
    options = {
        key: value for key, value in spec.items() if key not in ("name", "state_policy")
    }
    if spec["name"] == "sgd":
        return torch.optim.SGD(parameters, **options)
    if spec["name"] == "adamw":
        options.pop(
            "momentum", None
        )  # A switched client profile may inherit the SGD default.
        return torch.optim.AdamW(parameters, **options)
    raise ValueError("optimizer.name must be sgd or adamw")


def forward_batch(net, bundle, indices):
    batch = {
        key: value.to(net.device)
        for key, value in net.task.collate(net.tokenizer, bundle.rows(indices)).items()
    }
    with net.autocast():
        output = net.model(**batch)
        loss = net.task.compute_loss(output, batch)
        output.loss = loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    return batch, output, loss


def gradient_batch(net, bundle, state, job, worker_id):
    net.prepare(state)
    seed_training(job["seed"])
    net.model.train()
    net.model.zero_grad(set_to_none=True)
    if net.device.type == "cuda":
        torch.cuda.synchronize(net.device)
    start = time.perf_counter()
    batch, output, loss = forward_batch(net, bundle, job["indices"])
    (loss * job["scale"]).backward()
    statistics = net.task.statistics(output, batch)
    gradients = {
        key: cpu_tree(parameter.grad)
        for key, parameter in net.parameter_map().items()
        if parameter.grad is not None
    }
    if net.device.type == "cuda":
        torch.cuda.synchronize(net.device)
    return {
        "task_id": job["task_id"],
        "version": job["version"],
        "gradients": gradients,
        "statistics": statistics,
        "samples": len(job["indices"]),
        "tokens": int(batch["attention_mask"].sum()),
        "worker_id": worker_id,
        "physical_seconds": time.perf_counter() - start,
    }


def compute_signature(state, config, train):
    import hashlib
    import json

    layout = {
        key: (list(value.shape), str(value.dtype), state.role(key))
        for key, value in state.tensors.items()
    }
    value = {
        "layout": layout,
        "optimizer": train["optimizer"],
        "clip": train.get("max_grad_norm"),
        "dropout": config["model"]["adapter"].get("dropout", 0),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
