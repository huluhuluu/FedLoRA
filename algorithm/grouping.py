"""Grouping is independent of physical GPU placement."""

import numpy as np


class NoneGrouping:
    def group(self, clients, observations, context):
        return {"all": sorted(clients)}


class Predefined:
    def __init__(self, groups):
        self.groups = groups

    def group(self, clients, observations, context):
        flattened = [cid for ids in self.groups.values() for cid in ids]
        if sorted(flattened) != sorted(clients):
            raise ValueError("Predefined groups must cover each client exactly once")
        return {str(key): list(ids) for key, ids in self.groups.items()}


class ResourceProfile:
    def group(self, clients, observations, context):
        groups = {}
        for cid, config in clients.items():
            groups.setdefault(config.get("profile", "default"), []).append(cid)
        return groups


class Distribution:
    def __init__(self, num_groups=2, iterations=20):
        if min(num_groups, iterations) < 1:
            raise ValueError("num_groups and iterations must be positive")
        self.num_groups, self.iterations = num_groups, iterations

    def group(self, clients, observations, context):
        ids = sorted(clients)
        features = np.asarray(
            [observations[cid]["label_distribution"] for cid in ids], dtype=float
        )
        k = min(self.num_groups, len(ids))
        centers = features[context.rng.choice(len(ids), k, replace=False)].copy()
        for _ in range(self.iterations):
            labels = (
                ((features[:, None, :] - centers[None, :, :]) ** 2)
                .sum(axis=2)
                .argmin(axis=1)
            )
            updated = centers.copy()
            for group in range(k):
                if np.any(labels == group):
                    updated[group] = features[labels == group].mean(axis=0)
            if np.allclose(updated, centers):
                break
            centers = updated
        return {
            str(group): [cid for cid, label in zip(ids, labels) if label == group]
            for group in range(k)
            if np.any(labels == group)
        }
