"""Aggregation over named adapters; scaling is part of the model delta."""

import math
import time
from dataclasses import replace
import torch
from algorithm.interfaces import AggregationResult, Workload


def weights(updates, weighting):
    if weighting not in ("samples", "uniform"):
        raise ValueError("weighting must be samples or uniform")
    values = [u.num_samples if weighting == "samples" else 1 for u in updates]
    total = sum(values)
    if not updates or total <= 0:
        raise ValueError("Aggregation requires nonempty positive-weight updates")
    return [value / total for value in values]


def factorize(b, a, rank, scale):
    # QR reduces the SVD to the summed adapter ranks, without forming B @ A.
    qb, rb = torch.linalg.qr(b.float(), mode="reduced")
    qa, ra = torch.linalg.qr(a.float().T, mode="reduced")
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    kept = min(rank, len(s))
    root = (s[:kept] / scale).clamp_min(0).sqrt()
    out_b = torch.zeros(b.shape[0], rank)
    out_a = torch.zeros(rank, a.shape[1])
    out_b[:, :kept] = (qb @ u[:, :kept]) * root
    out_a[:kept] = root[:, None] * (vh[:kept] @ qa.T)
    return out_b, out_a, s


def client_state(state, config, seed):
    """Project global deltas to a client's rank while preserving zero-init gradients."""
    rank, alpha = config["adapter"]["rank"], config["adapter"]["alpha"]
    lora_keys = {
        key for block in state.blocks.values() for key in (block.a_key, block.b_key)
    }
    tensors = {k: v.clone() for k, v in state.tensors.items() if k not in lora_keys}
    blocks = {}
    generator = torch.Generator().manual_seed(seed)
    for name, block in state.blocks.items():
        a, b = state.tensors[block.a_key], state.tensors[block.b_key]
        scale = alpha / rank
        if rank == block.rank:
            ca, cb = a.clone(), b * (block.scale / scale)
        elif not torch.count_nonzero(b):
            ca = torch.empty(rank, block.input_dim).uniform_(
                -1 / math.sqrt(block.input_dim),
                1 / math.sqrt(block.input_dim),
                generator=generator,
            )
            ca[: min(rank, block.rank)] = a[: min(rank, block.rank)]
            cb = torch.zeros(block.output_dim, rank)
        elif rank > block.rank:
            ca = torch.empty(rank, block.input_dim).uniform_(
                -1 / math.sqrt(block.input_dim),
                1 / math.sqrt(block.input_dim),
                generator=generator,
            )
            ca[: block.rank] = a
            cb = torch.zeros(block.output_dim, rank)
            cb[:, : block.rank] = b * (block.scale / scale)
        else:
            cb, ca, _ = factorize(b * block.scale, a, rank, scale)
        unused = (ca == 0).all(dim=1) & (cb == 0).all(dim=0)
        if unused.any():
            ca[unused] = torch.empty(int(unused.sum()), block.input_dim).uniform_(
                -1 / math.sqrt(block.input_dim),
                1 / math.sqrt(block.input_dim),
                generator=generator,
            )
        tensors[block.a_key], tensors[block.b_key] = ca.to(a.dtype), cb.to(b.dtype)
        blocks[name] = replace(block, rank=rank, alpha=alpha, scale=scale)
    return state.with_tensors(tensors, blocks)


class FedAvg:
    def __init__(self, weighting="samples"):
        self.weighting = weighting

    def aggregate(self, updates, previous, context):
        start = time.perf_counter()
        ws = weights(updates, self.weighting)
        layout = updates[0].state.blocks
        if layout.keys() != previous.blocks.keys():
            raise ValueError("FedAvg requires common target modules")
        for update in updates:
            if update.state.base_reference != previous.base_reference:
                raise ValueError(
                    "Cannot aggregate adapters with different frozen backbones"
                )
            if update.state.blocks != layout:
                raise ValueError(
                    "FedAvg requires identical adapter layouts and scaling"
                )
        tensors = {
            key: value.clone() if previous.role(key) != "global" else value
            for key, value in previous.tensors.items()
        }
        for key, value in tensors.items():
            if previous.role(key) == "global":
                tensors[key] = sum(
                    u.state.tensors[key].float() * w for u, w in zip(updates, ws)
                ).to(value.dtype)
        return AggregationResult(
            previous.with_tensors(tensors, layout),
            workload=Workload(
                operation="aggregate", physical_seconds=time.perf_counter() - start
            ),
        )


class SVD:
    def __init__(self, weighting="samples", server_rank=None, energy=None):
        self.weighting, self.server_rank, self.energy = weighting, server_rank, energy
        if server_rank is not None and (
            type(server_rank) is not int or server_rank < 1
        ):
            raise ValueError("server_rank must be a positive integer")
        if energy is not None and not 0 < energy <= 1:
            raise ValueError("energy must be in (0, 1]")

    def aggregate(self, updates, previous, context):
        start = time.perf_counter()
        ws = weights(updates, self.weighting)
        tensors = {
            key: value.clone() if previous.role(key) != "global" else value
            for key, value in previous.tensors.items()
        }
        blocks = {}
        lora_keys = {
            key
            for block in previous.blocks.values()
            for key in (block.a_key, block.b_key)
        }
        for update in updates:
            if (
                update.state.base_reference != previous.base_reference
                or update.state.blocks.keys() != previous.blocks.keys()
            ):
                raise ValueError("SVD requires common backbone and target modules")
        for name, block in previous.blocks.items():
            bs, aa = [], []
            for update, weight in zip(updates, ws):
                own = update.state.blocks[name]
                root = math.sqrt(weight * own.scale)
                bs.append(update.state.tensors[own.b_key].float() * root)
                aa.append(update.state.tensors[own.a_key].float() * root)
            bcat, acat = torch.cat(bs, 1), torch.cat(aa, 0)
            rank = self.server_rank or max(
                block.rank, *(u.state.blocks[name].rank for u in updates)
            )
            if not torch.count_nonzero(bcat):
                old_a = tensors[block.a_key]
                generator = torch.Generator().manual_seed(
                    context.seed if context else 0
                )
                a = torch.empty(rank, block.input_dim).uniform_(
                    -1 / math.sqrt(block.input_dim),
                    1 / math.sqrt(block.input_dim),
                    generator=generator,
                )
                a[: min(rank, block.rank)] = old_a[: min(rank, block.rank)]
                tensors[block.a_key] = a.to(old_a.dtype)
                tensors[block.b_key] = torch.zeros(
                    block.output_dim, rank, dtype=tensors[block.b_key].dtype
                )
                blocks[name] = replace(block, rank=rank, scale=block.alpha / rank)
                continue
            b, a, singular = factorize(bcat, acat, rank, block.alpha / rank)
            if self.energy and singular.square().sum() > 0:
                rank = min(
                    rank,
                    int(
                        torch.searchsorted(
                            singular.square().cumsum(0) / singular.square().sum(),
                            self.energy,
                        )
                    )
                    + 1,
                )
                adjustment = math.sqrt(rank / b.shape[1])
                b, a = b[:, :rank] * adjustment, a[:rank] * adjustment
            tensors[block.a_key], tensors[block.b_key] = (
                a.to(tensors[block.a_key].dtype),
                b.to(tensors[block.b_key].dtype),
            )
            blocks[name] = replace(block, rank=rank, scale=block.alpha / rank)
        for key in tensors.keys() - lora_keys:
            if previous.role(key) == "global":
                tensors[key] = sum(
                    u.state.tensors[key].float() * w for u, w in zip(updates, ws)
                ).to(tensors[key].dtype)
        return AggregationResult(
            previous.with_tensors(tensors, blocks),
            workload=Workload(
                operation="aggregate", physical_seconds=time.perf_counter() - start
            ),
        )
