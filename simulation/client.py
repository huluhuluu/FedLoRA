"""Logical clients remain independent of physical worker placement."""

from dataclasses import dataclass, field


@dataclass
class ClientState:
    private: dict = field(default_factory=dict)
    optimizer: dict | None = None
    scaler: dict | None = None

    def to_dict(self):
        return {
            "private": self.private,
            "optimizer": self.optimizer,
            "scaler": self.scaler,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass
class ClientTask:
    client_id: int
    round_id: int
    global_version: int
    config: dict
    indices: list
    seed: int
    persistent: ClientState = field(default_factory=ClientState)
