import numpy as np
import pytest
import torch
from algorithm.Fed import FedAvg, SVD, client_state
from algorithm.interfaces import ClientUpdate, Context
from algorithm.sampling import IID, Shard, Dirichlet
from models.adapter_state import AdapterState, LoRABlockSpec


def adapter(rank=2, alpha=4, zero=False):
    g = torch.Generator().manual_seed(rank)
    return AdapterState(
        {
            "a": torch.randn(rank, 5, generator=g),
            "b": torch.zeros(4, rank) if zero else torch.randn(4, rank, generator=g),
            "head": torch.tensor([float(rank)]),
            "private": torch.tensor([99.0]),
        },
        {"layer": LoRABlockSpec("a", "b", rank, alpha, alpha / rank, 5, 4)},
        {"head": "global", "private": "private"},
        "base",
    )


def ctx():
    return Context(0, 0, 42, np.random.default_rng(42))


@pytest.mark.parametrize("method", [IID(), Shard(5), Dirichlet(0.1)])
@pytest.mark.parametrize("clients", [2, 20])
def test_partition_conservation(method, clients):
    class Data:
        def __len__(self):
            return 13

        def __getitem__(self, key):
            return [i % 3 for i in range(13)]

    result = method.partition(Data(), range(clients), ctx())
    assert sorted(index for ids in result.values() for index in ids) == list(range(13))
    assert result == method.partition(Data(), range(clients), ctx())


def test_scaled_svd_and_head_weighting():
    previous = adapter(4, 8)
    first, second = adapter(1, 3), adapter(2, 10)
    updates = [ClientUpdate(0, 0, 0, first, 1), ClientUpdate(1, 0, 0, second, 3)]
    result = SVD().aggregate(updates, previous, ctx()).state
    expected = (
        0.25 * (first.tensors["b"] @ first.tensors["a"]) * 3
        + 0.75 * (second.tensors["b"] @ second.tensors["a"]) * 5
    )
    assert torch.allclose(
        result.tensors["b"] @ result.tensors["a"] * 2, expected, atol=2e-5
    )
    assert result.tensors["head"].item() == 1.75
    assert result.tensors["private"].item() == 99


def test_zero_initialization_stays_trainable_and_rank_projection():
    state = adapter(zero=True)
    result = SVD().aggregate([ClientUpdate(0, 0, 0, state, 1)], state, ctx()).state
    assert torch.equal(result.tensors["a"], state.tensors["a"])
    projected = client_state(state, {"adapter": {"rank": 4, "alpha": 16}}, 3)
    assert projected.tensors["a"].abs().sum() > 0
    assert projected.tensors["b"].abs().sum() == 0
    assert projected.blocks["layer"].scale == 4
    state = adapter()
    projected = client_state(state, {"adapter": {"rank": 4, "alpha": 16}}, 3)
    assert torch.allclose(
        projected.tensors["b"] @ projected.tensors["a"] * 4,
        state.tensors["b"] @ state.tensors["a"] * 2,
    )


def test_fedavg_rejects_incompatible_layouts():
    with pytest.raises(ValueError, match="identical"):
        FedAvg().aggregate(
            [
                ClientUpdate(0, 0, 0, adapter(1), 2),
                ClientUpdate(1, 0, 0, adapter(2), 2),
            ],
            adapter(),
            ctx(),
        )
    state = adapter()
    state.base_reference = "other"
    with pytest.raises(ValueError, match="backbones"):
        FedAvg().aggregate([ClientUpdate(0, 0, 0, state, 2)], adapter(), ctx())


def test_virtual_time_uses_logical_clients():
    from algorithm.interfaces import Workload
    from simulation.timing import VirtualClock
    from utils.config import DEFAULT, merge

    spec = merge(DEFAULT, {"timing": {"mode": "analytic", "assumed_step_seconds": 2}})
    from algorithm.registry import load_plugins

    load_plugins()
    clock = VirtualClock(spec)
    clients = {
        0: {
            "system": {
                "time_multiplier": 1,
                "download_mbps": 8,
                "upload_mbps": 8,
                "latency_ms": 0,
            }
        },
        1: {
            "system": {
                "time_multiplier": 3,
                "download_mbps": 8,
                "upload_mbps": 8,
                "latency_ms": 0,
            }
        },
    }
    updates = [
        ClientUpdate(
            i,
            0,
            0,
            adapter(),
            1,
            workload=Workload(optimizer_steps=2, upload_bytes=1_000_000),
        )
        for i in clients
    ]
    duration, _ = clock.advance(updates, clients, {"all": [0, 1]})
    assert duration == 13


def test_zero_singular_directions_and_empty_round():
    state = adapter()
    state.tensors["a"].zero_()
    state.tensors["b"].zero_()
    projected = client_state(state, {"adapter": {"rank": 2, "alpha": 4}}, 3)
    assert projected.tensors["a"].abs().sum() > 0
    assert not projected.tensors["b"].count_nonzero()
    from simulation.timing import VirtualClock
    from utils.config import DEFAULT, merge
    from algorithm.registry import load_plugins

    load_plugins()
    clock = VirtualClock(
        merge(
            DEFAULT, {"timing": {"mode": "analytic", "assumed_aggregation_seconds": 5}}
        )
    )
    assert clock.advance([], {}, {}) == (0.0, {})
    assert clock.elapsed == 0


def test_profile_matching_is_explicit(tmp_path):
    import json
    from algorithm.interfaces import Workload
    from simulation.timing import Profiled

    path = tmp_path / "profile.json"
    entry = {
        "model": "fixture",
        "task": "classification",
        "precision": "fp32",
        "rank": 2,
        "batch_size": 1,
        "seq_len": 8,
        "hardware_profile": "fixture-cpu",
        "step_seconds": 0.5,
    }
    path.write_text(json.dumps({"entries": [entry]}))
    profiler = Profiled(
        {"profile_file": str(path)}, {"path": "fixture", "task": "classification"}
    )
    work = Workload(rank=2, batch_size=1, seq_len=8, optimizer_steps=3)
    client = {"system": {"hardware_profile": "fixture-cpu", "time_multiplier": 2}}
    assert profiler.seconds(work, client) == 3
    work.rank = 3
    with pytest.raises(ValueError, match="matching"):
        profiler.seconds(work, client)


def test_negative_time_configuration():
    from utils.config import DEFAULT, merge, validate

    with pytest.raises(ValueError, match="negative"):
        validate(
            merge(DEFAULT, {"timing": {"mode": "analytic", "assumed_step_seconds": -1}})
        )


def test_group_balanced_selection_does_not_starve_later_groups():
    from algorithm.selection import GroupBalanced

    clients = dict.fromkeys(range(8))
    groups = {str(i): [2 * i, 2 * i + 1] for i in range(4)}
    seen = set()
    for seed in range(100):
        selected = GroupBalanced(2).select(
            clients, groups, Context(seed, 0, seed, np.random.default_rng(seed))
        )
        assert len({cid // 2 for cid in selected}) == 2
        seen.update(selected)
    assert seen == set(clients)


def test_svd_zero_update_replaces_old_delta_and_obeys_rank_cap():
    previous = adapter(2)
    zero = adapter(2, zero=True)
    result = (
        SVD(server_rank=1)
        .aggregate([ClientUpdate(0, 0, 0, zero, 1)], previous, ctx())
        .state
    )
    assert result.blocks["layer"].rank == 1
    assert result.tensors["a"].abs().sum() > 0
    assert result.tensors["b"].count_nonzero() == 0


def test_fedavg_accepts_common_group_layout_after_svd():
    previous = adapter(2)
    reduced = (
        SVD(server_rank=1)
        .aggregate([ClientUpdate(0, 0, 0, previous, 1)], previous, ctx())
        .state
    )
    result = (
        FedAvg().aggregate([ClientUpdate(0, 0, 0, reduced, 1)], previous, ctx()).state
    )
    assert result.blocks == reduced.blocks
    assert torch.equal(result.tensors["a"], reduced.tensors["a"])


def test_energy_rank_reuses_one_factorization(monkeypatch):
    import algorithm.Fed as module

    previous = adapter(4, 8)
    states = [adapter(1, 3), adapter(2, 10)]
    updates = [ClientUpdate(0, 0, 0, states[0], 1), ClientUpdate(1, 0, 0, states[1], 3)]
    dense = sum(
        weight * state.blocks["layer"].scale * (state.tensors["b"] @ state.tensors["a"])
        for weight, state in zip([0.25, 0.75], states)
    )
    calls = []
    factorize = module.factorize

    def counted(*args, **kwargs):
        calls.append(True)
        return factorize(*args, **kwargs)

    monkeypatch.setattr(module, "factorize", counted)
    result = SVD(energy=0.5).aggregate(updates, previous, ctx()).state
    rank = result.blocks["layer"].rank
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    expected = (u[:, :rank] * s[:rank]) @ vh[:rank]
    actual = result.blocks["layer"].scale * (result.tensors["b"] @ result.tensors["a"])
    assert len(calls) == 1
    assert torch.allclose(actual, expected, atol=2e-5)
