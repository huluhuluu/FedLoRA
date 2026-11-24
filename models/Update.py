"""Bounded local training on one reusable backbone per worker."""

import math
import time
import numpy as np
import torch
from algorithm.Fed import client_state
from algorithm.interfaces import ClientUpdate, Workload
from utils.helper import seed_training, cpu_tree
from training.loop import make_optimizer, forward_batch


def train_client(net, bundle, global_state, task, worker_id=0):
    state = client_state(global_state, task.config, task.seed)
    state.tensors.update(task.persistent.private)
    net.prepare(state)
    net.invalidate()
    # Adapter construction must not change dropout RNG when tasks migrate between workers.
    seed_training(task.seed)
    model = net.model
    model.train()
    spec = task.config["train"]
    opt = spec["optimizer"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(parameters, opt)
    if opt["state_policy"] == "retain" and task.persistent.optimizer:
        optimizer.load_state_dict(task.persistent.optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=net.precision == "fp16")
    if task.persistent.scaler:
        scaler.load_state_dict(task.persistent.scaler)
    successful_steps = 0
    batch_size, budget = spec["batch_size"], spec["budget"]
    batches_per_epoch = math.ceil(len(task.indices) / batch_size)
    steps = (
        budget["value"] * batches_per_epoch
        if budget["unit"] == "epochs"
        else budget["value"]
    )
    totals, processed, tokens = {}, 0, 0
    generator = np.random.default_rng(task.seed)
    order, cursor = [], 0
    if net.device.type == "cuda":
        torch.cuda.synchronize(net.device)
    start = time.perf_counter()
    for _ in range(steps):
        if cursor >= len(order):
            order = generator.permutation(task.indices).tolist()
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        optimizer.zero_grad(set_to_none=True)
        batch, output, loss = forward_batch(net, bundle, indices)
        scaler.scale(loss).backward()
        if spec.get("max_grad_norm"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, spec["max_grad_norm"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        successful_steps += int(scaler.get_scale() >= old_scale)
        for key, value in net.task.statistics(output, batch).items():
            totals[key] = totals.get(key, 0) + value
        processed += len(indices)
        tokens += int(batch["attention_mask"].sum())
    if net.device.type == "cuda":
        torch.cuda.synchronize(net.device)
    elapsed = time.perf_counter() - start
    result = net.export()
    from training.loop import compute_signature

    workload = Workload(
        compute_signature=compute_signature(result, net.config, spec),
        samples_processed=processed,
        unique_samples=min(processed, len(task.indices)),
        tokens_processed=tokens,
        batches=steps,
        optimizer_steps=successful_steps,
        upload_bytes=result.payload_bytes(),
        download_bytes=state.payload_bytes(),
        physical_seconds=elapsed,
        batch_size=batch_size,
        seq_len=net.config["data"]["seq_len"],
        rank=task.config["adapter"]["rank"],
        precision=net.precision,
        device=str(net.device),
        hardware_profile=net.config["execution"].get("hardware_profile")
        or (
            torch.cuda.get_device_name(net.device)
            if net.device.type == "cuda"
            else "cpu"
        ),
    )
    return ClientUpdate(
        task.client_id,
        task.round_id,
        task.global_version,
        result,
        len(task.indices),
        {**net.task.finalize(totals), "loss_weight": totals["weight"]},
        workload,
        cpu_tree(optimizer.state_dict()) if opt["state_policy"] == "retain" else None,
        worker_id,
        scaler.state_dict(),
    )
