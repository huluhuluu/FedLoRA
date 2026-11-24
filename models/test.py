"""Evaluation returns mergeable statistics for data shards."""

__test__ = False
import time
import torch
from algorithm.Fed import client_state


@torch.no_grad()
def evaluate_statistics(net, bundle, state, client=None, indices=None):
    if client is not None:
        config, persistent = client
        state = client_state(state, config, 0)
        state.tensors.update(persistent.private)
    net.prepare(state)
    net.model.eval()
    totals = {}
    indices = list(range(len(bundle.evaluation))) if indices is None else indices
    batch_size = net.config.get("evaluation", {}).get("batch_size", 4)
    start = time.perf_counter()
    for offset in range(0, len(indices), batch_size):
        batch = {
            key: value.to(net.device)
            for key, value in net.task.collate(
                net.tokenizer, bundle.rows(indices[offset : offset + batch_size], False)
            ).items()
        }
        with net.autocast():
            output = net.model(**batch)
            output.loss = net.task.compute_loss(output, batch)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        for key, value in net.task.statistics(output, batch).items():
            totals[key] = totals.get(key, 0) + value
    return {"statistics": totals, "physical_seconds": time.perf_counter() - start}


def evaluate(net, bundle, state, client=None):
    result = evaluate_statistics(net, bundle, state, client)
    return {
        **net.task.finalize(result["statistics"]),
        "physical_seconds": result["physical_seconds"],
    }
