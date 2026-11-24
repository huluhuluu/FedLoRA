"""Example plugin: choose clients by an explicit priority."""

from algorithm.initialization import Kaiming
from algorithm.registry import register


@register("selection", "priority")
class Priority:
    def __init__(self, clients_per_round=2):
        self.count = clients_per_round

    def select(self, clients, groups, context):
        # The engine exposes configuration here; use a declared client priority.
        return sorted(clients, key=lambda cid: (-clients[cid].get("priority", 0), cid))[
            : self.count
        ]


@register("initialization", "scaled_kaiming")
class ScaledKaiming(Kaiming):
    def __init__(self, gain=1.0):
        self.gain = gain

    def initialize(self, state, net, data, context):
        result = state.clone()
        for block in result.blocks.values():
            result.tensors[block.a_key].mul_(self.gain)
        return result
