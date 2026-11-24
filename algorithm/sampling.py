"""Client partitions preserve every sample exactly once."""

import numpy as np


class IID:
    def partition(self, dataset, clients, context):
        indices = context.rng.permutation(len(dataset))
        return {
            cid: part.tolist()
            for cid, part in enumerate(np.array_split(indices, len(clients)))
        }


class Shard:
    def __init__(self, num_shards=200):
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        self.num_shards = num_shards

    def partition(self, dataset, clients, context):
        labels = np.asarray(dataset["label"])
        order = np.argsort(labels, kind="stable")
        chunks = np.array_split(order, min(self.num_shards, len(order)))
        result = {cid: [] for cid in range(len(clients))}
        for position, shard in enumerate(context.rng.permutation(len(chunks))):
            result[position % len(clients)].extend(chunks[shard].tolist())
        return result


class Dirichlet:
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive")
        self.alpha = alpha

    def partition(self, dataset, clients, context):
        labels = np.asarray(dataset["label"])
        result = {cid: [] for cid in range(len(clients))}
        for label in np.unique(labels):
            indices = context.rng.permutation(np.flatnonzero(labels == label))
            probabilities = context.rng.dirichlet(np.full(len(clients), self.alpha))
            counts = context.rng.multinomial(len(indices), probabilities)
            offset = 0
            for cid, count in enumerate(counts):
                result[cid].extend(indices[offset : offset + count].tolist())
                offset += count
        return result
