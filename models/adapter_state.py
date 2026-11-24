"""Named adapter tensors shared by models, algorithms and workers."""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class LoRABlockSpec:
    a_key: str
    b_key: str
    rank: int
    alpha: float
    scale: float
    input_dim: int
    output_dim: int
    module_kind: str = "linear"
    weight_orientation: str = "out_in"


@dataclass
class AdapterState:
    tensors: dict
    blocks: dict
    extra_roles: dict
    base_reference: str
    layout_version: int = 1

    def with_tensors(self, tensors, blocks=None):
        return replace(
            self, tensors=tensors, blocks=self.blocks if blocks is None else blocks
        )

    def clone(self):
        return self.with_tensors(
            {k: v.detach().cpu().clone() for k, v in self.tensors.items()}
        )

    def role(self, key):
        return self.extra_roles.get(key, "global")

    def payload_bytes(self, shared_only=True):
        return sum(
            t.numel() * t.element_size()
            for k, t in self.tensors.items()
            if not shared_only or self.role(k) == "global"
        )

    def private_tensors(self):
        return {
            k: t.detach().cpu().clone()
            for k, t in self.tensors.items()
            if self.role(k) == "private"
        }

    def to_dict(self):
        return {
            "tensors": self.tensors,
            "blocks": {k: asdict(v) for k, v in self.blocks.items()},
            "extra_roles": self.extra_roles,
            "base_reference": self.base_reference,
            "layout_version": self.layout_version,
        }

    @classmethod
    def from_dict(cls, state):
        return cls(
            tensors=state["tensors"],
            blocks={k: LoRABlockSpec(**v) for k, v in state["blocks"].items()},
            extra_roles=state["extra_roles"],
            base_reference=state["base_reference"],
            layout_version=state.get("layout_version", 1),
        )
