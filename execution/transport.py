"""Serialize CPU states without one shared-memory descriptor per tensor."""

import pickle
from datetime import timedelta


def send_message(channel, value):
    channel.put(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def receive_message(channel, timeout=None):
    return pickle.loads(channel.get(timeout=timeout))


def send_rank_message(value, dst):
    import torch
    import torch.distributed as dist

    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    dist.send(torch.tensor([len(data)], dtype=torch.int64), dst=dst)
    dist.send(torch.frombuffer(bytearray(data), dtype=torch.uint8), dst=dst)


def receive_rank_message(src):
    import torch
    import torch.distributed as dist

    header = torch.empty(1, dtype=torch.int64)
    # Gloo applies the group timeout to idle object receives. Only the header uses a long idle wait.
    dist.irecv(header, src=src).wait(timeout=timedelta(days=365))
    data = torch.empty(int(header.item()), dtype=torch.uint8)
    dist.recv(data, src=src)
    return pickle.loads(data.numpy().tobytes())
